"""Nucleo generalizado de cenarios (ESPEC_BASE_CASOS_2026-11-14, secao 1).

Um unico caminho de calculo (ScenarioEvaluator.evaluate) serve:
  - a formulacao restrita de 3 objetivos (Constrained3Problem),
  - a formulacao penalizada de 5 objetivos (Penalized5Problem),
  - o gerador de base para IC (features pre-manobra).

As definicoes fisicas sao as do artigo SEPOC 2026 (src/sepoc_study.py e
src/nsga2_optimizer.py), sem alteracao.

Limitacao MEDIDA (2026-11-14): py-dss-interface 2.3.0 envia os comandos ao
engine codificados em ASCII (py_dss_interface/models/Text/Text.py:14); um
master.dss em caminho com acentos (como esta pasta) gera UnicodeEncodeError
no `compile`. Por isso `build_instance` grava o .dss canonico em
work_dir/<instance_id>/ e, se esse caminho nao for ASCII, compila a partir
de uma copia em diretorio ASCII (variavel de ambiente NSGA2VCB_DSS_ROOT ou
<tempdir>/nsga2vcb_dss/).

Scc (feature de IC, correcao F1 de 2026-11-14): o metodo analitico so vale
para fonte unica sem gerador nem interligacao (P1); fora disso devolve NaN
com aviso, e o ScenarioEvaluator (scc_method="auto") usa o faultstudy do
OpenDSS em subprocesso. MEDIDO antes da correcao (work/verif/verif_scc.log):
o analitico dava 74,205 MVA em MV_A de um caso tipo P2, contra 145,474 (N-0)
e 78,555 MVA (N-1) do faultstudy.
"""
from __future__ import annotations

import cmath
import copy
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config_loader import SystemData, deep_merge, load_system
from .dss_builder import build as build_dss
from .powerflow_runner import ScenarioState, run_maneuver
from .protection_logic import evaluate_protection, total_idle_motor_kW

log = logging.getLogger(__name__)

_SAFE_ID = re.compile(r"^[A-Za-z0-9_.\-]+$")


# ===========================================================================
# Instancias
# ===========================================================================
@dataclass
class PlantInstance:
    """Instancia de planta: dados + alvo + nivel de carga + overrides do yaml."""

    instance_id: str
    plant_id: str
    data_dir: Path
    target_motor_id: int
    static_load_mult: float = 1.0
    overrides: dict = field(default_factory=dict)


def ascii_compile_dir(preferred: Path) -> Path:
    """Diretorio para o OpenDSS compilar: `preferred` se for ASCII, senao um
    espelho ASCII deterministico (hash do caminho preferido)."""
    preferred = Path(os.path.abspath(str(preferred)))
    if str(preferred).isascii():
        return preferred
    root = Path(os.environ.get("NSGA2VCB_DSS_ROOT",
                               Path(tempfile.gettempdir()) / "nsga2vcb_dss"))
    root = Path(os.path.abspath(str(root)))
    if not str(root).isascii():
        raise RuntimeError(
            f"Diretorio de compilacao OpenDSS nao-ASCII: {root}. Defina "
            "NSGA2VCB_DSS_ROOT com um caminho ASCII.")
    tag = hashlib.sha1(str(preferred).encode("utf-8")).hexdigest()[:12]
    return root / f"{preferred.name if preferred.name.isascii() else 'inst'}_{tag}"


def _apply_target(system: SystemData, target_motor_id: int) -> None:
    tid = int(target_motor_id)
    row = system.motor_by_id(tid)  # KeyError se inexistente
    tm = system.config.setdefault("target_motor", {})
    tm["motor_id"] = tid
    tm["name"] = str(row["name"])
    system.motors["is_target"] = system.motors["motor_id"].astype(int) == tid


def _apply_static_load_mult(system: SystemData, mult: float) -> None:
    if float(mult) == 1.0:
        return  # nao toca os valores (preserva bytes do .dss original)
    m = float(mult)
    loads = system.loads.copy()
    loads["kW"] = [round(float(v) * m, 6) for v in loads["kW"]]
    loads["kvar"] = [round(float(v) * m, 6) for v in loads["kvar"]]
    system.loads = loads


def _normalize_contingency(cfg: dict) -> None:
    """Em modo N-0 o builder liga todos os trafos; zera transformers_out para
    que powerflow_runner.collect_snapshot meça todos (coerencia)."""
    ct = cfg.get("contingency") or {}
    if ct and ct.get("mode", "N-0") == "N-0" and ct.get("transformers_out"):
        ct["transformers_out"] = []


def build_instance(inst: PlantInstance, work_dir: str | Path) -> tuple[SystemData, Path]:
    """Carrega, aplica alvo/carga/overrides e gera o .dss da instancia.

    Ordem (secao 1 da especificacao): (1) carrega data_dir; (2) alvo;
    (3) multiplica kW e kvar das cargas estaticas; (4) merge profundo dos
    overrides no yaml; (5) gera o .dss em work_dir/<instance_id>/.

    Retorna (SystemData, master_path) onde master_path e compilavel pelo
    OpenDSS (ASCII; ver docstring do modulo).
    """
    if not _SAFE_ID.match(inst.instance_id):
        raise ValueError(f"instance_id invalido (use [A-Za-z0-9_.-]): {inst.instance_id!r}")
    # Resolver ANTES de qualquer engine OpenDSS (que altera o cwd).
    data_dir = Path(os.path.abspath(str(inst.data_dir)))
    out_dir = Path(os.path.abspath(str(work_dir))) / inst.instance_id

    system = load_system(data_dir)
    _apply_target(system, inst.target_motor_id)
    _apply_static_load_mult(system, inst.static_load_mult)
    if inst.overrides:
        system.config = deep_merge(system.config, inst.overrides)
        tid_cfg = int(system.config["target_motor"]["motor_id"])
        if tid_cfg != int(inst.target_motor_id):
            raise ValueError("overrides nao podem trocar target_motor.motor_id; "
                             "use PlantInstance.target_motor_id")
    _normalize_contingency(system.config)

    out_dir.mkdir(parents=True, exist_ok=True)
    master, network = build_dss(system, out_dir)
    meta = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(inst).items()}
    meta["data_dir"] = str(data_dir)
    comp_dir = ascii_compile_dir(out_dir)
    if comp_dir != out_dir:
        comp_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(master, comp_dir / master.name)
        shutil.copyfile(network, comp_dir / network.name)
        master = comp_dir / master.name
    meta["compile_master"] = str(master)
    (out_dir / "instance.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")
    return system, Path(master)


# ===========================================================================
# Utilidades de topologia
# ===========================================================================
def transformers_out(cfg: dict) -> set[str]:
    """Trafos fora de servico, mesma regra de dss_builder._get_transformers_out."""
    ct = cfg.get("contingency", {}) or {}
    if not ct or ct.get("mode", "N-0") == "N-0":
        return set()
    return set(ct.get("transformers_out", []) or [])


def in_service_transformers(cfg: dict) -> list[dict]:
    out = transformers_out(cfg)
    return [t for t in cfg.get("transformers", []) or [] if str(t["name"]) not in out]


def _sag_monitored_buses(cfg: dict) -> list[str] | None:
    cons = cfg.get("constraints", {}) or {}
    buses = (cons.get("sag_monitored_buses") or cfg.get("sag_monitored_buses")
             or (cfg.get("operational_limits", {}) or {}).get("sag_monitored_buses"))
    return [str(b) for b in buses] if buses else None


def _loading_transformers(cfg: dict) -> list[str] | None:
    cons = cfg.get("constraints", {}) or {}
    trs = cons.get("loading_transformers")
    return [str(t) for t in trs] if trs else None


def _bus_key(name: Any) -> str:
    """Nome de barra normalizado (sem sufixo de nos '.1.2.3', minusculo)."""
    return str(name).split(".")[0].strip().lower()


def _flag(val: Any) -> bool:
    """yes/true/1/on => True; NaN/None/vazio => False (mesma regra do builder)."""
    if val is None or (isinstance(val, float) and val != val):
        return False
    if isinstance(val, str):
        return val.strip().lower() in {"yes", "true", "1", "on", "y", "t"}
    return bool(val)


def scc_analytic_inapplicable_reason(system: SystemData, bus: str) -> str | None:
    """Motivo pelo qual o Scc analitico NAO vale para `bus` (None = vale).

    O modelo analitico so representa: uma unica fonte, sem geradores, e a
    barra `bus` alimentada apenas por trafos em servico ligados direto a
    barra da fonte, sem outro caminho ate uma fonte. Conservador: qualquer
    das condicoes abaixo invalida o metodo (achado F1, 2026-11-14):
      - mais de uma fonte declarada em `sources:`;
      - qualquer gerador declarado em `generators:`;
      - linha de manobra (switch=yes) habilitada na componente de `bus`;
      - trafo em servico da barra da fonte para outra barra da mesma
        componente de `bus` (caminho paralelo, ex. via interligacao);
      - nenhum trafo em servico ligando a barra da fonte a `bus`.
    A componente e o grafo de linhas habilitadas e trafos em servico, sem
    a barra da fonte.
    """
    cfg = system.config
    sources = cfg.get("sources") or []
    if len(sources) > 1:
        return f"{len(sources)} fontes em `sources:` (so a primeira entra no analitico)"
    gens = cfg.get("generators") or []
    if gens:
        return "gerador(es) local(is) declarado(s): " + ", ".join(
            str(g.get("name", "?")) for g in gens)
    src_bus = _bus_key(sources[0].get("bus", cfg["system"]["source_bus"]) if sources
                       else cfg["system"]["source_bus"])
    mv_main = str(cfg["system"]["mv_main_bus"])
    target = _bus_key(bus)
    if target == src_bus:
        return None  # fonte unica, sem gerador: Scc = Thevenin da fonte

    # Grafo: linhas habilitadas + trafos em servico, sem a barra da fonte.
    adj: dict[str, set[str]] = {}
    switch_lines: list[tuple[str, str, str]] = []

    def _edge(a: str, b: str) -> None:
        if src_bus in (a, b):
            return
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    lines = system.lines
    has_en = "enabled" in lines.columns
    has_sw = "switch" in lines.columns
    for _, row in lines.iterrows():
        if has_en and not _flag(row["enabled"]):
            continue
        a, b = _bus_key(row["bus_from"]), _bus_key(row["bus_to"])
        _edge(a, b)
        if has_sw and _flag(row["switch"]):
            switch_lines.append((str(row["name"]), a, b))
    src_trs: list[tuple[str, str]] = []
    for tr in in_service_transformers(cfg):
        hv = _bus_key(tr.get("bus_hv", cfg["system"]["source_bus"]))
        lv = _bus_key(tr.get("bus_lv", mv_main))
        if hv == src_bus:
            src_trs.append((str(tr["name"]), lv))
        else:
            _edge(hv, lv)
    comp = {target}
    stack = [target]
    while stack:
        for nb in adj.get(stack.pop(), ()):
            if nb not in comp:
                comp.add(nb)
                stack.append(nb)
    sw = [nm for nm, a, b in switch_lines if a in comp or b in comp]
    if sw:
        return "linha(s) de manobra habilitada(s) ligada(s) a barra: " + ", ".join(sw)
    par = [nm for nm, lv in src_trs if lv in comp and lv != target]
    if par:
        return "caminho paralelo pela fonte via trafo(s): " + ", ".join(par)
    if not any(lv == target for _, lv in src_trs):
        return "nenhum trafo em servico liga a barra da fonte a esta barra"
    return None


def scc_mva_analytic(system: SystemData, bus: str, warn: bool = True) -> float:
    """Potencia de curto-circuito trifasica (MVA) na barra `bus`, analitica.

    Modelo: Thevenin da fonte principal (isc3, X/R) em serie com o paralelo
    dos trafos EM SERVICO que ligam a barra da fonte a `bus` (Z = %R + j%X,
    com %R = %loadloss, como o OpenDSS). Cargas e motores nao contribuem.
    Validado contra faultstudy do OpenDSS na P1 (MV_BUS, N-1: 74,205 MVA
    nos dois; work/verif/verif_scc.log).

    Devolve NaN (e emite RuntimeWarning se `warn`) quando o modelo nao
    representa a topologia: varias fontes, gerador, interligacao (linha
    switch=yes), caminho paralelo ou barra sem trafo direto da fonte (ver
    `scc_analytic_inapplicable_reason`). Para essas, use
    `scc_mva_faultstudy`. A barra da propria fonte (planta de uma fonte, sem
    gerador) devolve o Scc da fonte. Definicao do grupo de IC pode diferir.
    """
    cfg = system.config
    reason = scc_analytic_inapplicable_reason(system, bus)
    if reason is not None:
        if warn:
            warnings.warn(f"scc_mva_analytic({bus}): metodo analitico nao se aplica "
                          f"({reason}); devolvendo NaN. Use scc_mva_faultstudy.",
                          RuntimeWarning, stacklevel=2)
        return float("nan")
    sources = cfg.get("sources")
    if sources:
        src = sources[0]
        src_bus = str(src.get("bus", cfg["system"]["source_bus"]))
        kv = float(src.get("basekv", cfg["system"]["base_kV_HV"]))
    else:
        src = cfg["source"]
        src_bus = str(cfg["system"]["source_bus"])
        kv = float(cfg["system"]["base_kV_HV"])
    s_base = float(cfg["system"]["base_MVA"])
    ssc_src = math.sqrt(3.0) * kv * float(src["isc3_kA"])
    zs = (s_base / ssc_src) * cmath.exp(1j * math.atan(float(src["x1r1"])))
    if bus.lower() == src_bus.lower():
        return s_base / abs(zs)
    mv_main = str(cfg["system"]["mv_main_bus"])
    ys = 0j
    for tr in in_service_transformers(cfg):
        hv = str(tr.get("bus_hv", cfg["system"]["source_bus"]))
        lv = str(tr.get("bus_lv", mv_main))
        if hv.lower() == src_bus.lower() and lv.lower() == bus.lower():
            z_own = complex(float(tr["loss_load_percent"]), float(tr["xhl_percent"])) / 100.0
            ys += 1.0 / (z_own * s_base / (float(tr["kva"]) / 1000.0))
    if ys == 0:  # pragma: no cover (coberto por scc_analytic_inapplicable_reason)
        return float("nan")
    return s_base / abs(zs + 1.0 / ys)


# Script do faultstudy, executado em PROCESSO SEPARADO: instancias DSS() no
# mesmo processo compartilham o engine (work/check_engine_sharing.py), e um
# compile aqui trocaria o circuito do avaliador.
_FAULTSTUDY_SCRIPT = r"""
import json, math, sys
from py_dss_interface import DSS
master, bus = sys.argv[1], sys.argv[2]
d = DSS()
d.text('compile "' + master + '"')
d.text("solve mode=faultstudy")
d.circuit.set_active_bus(bus)
name = str(d.bus.name)
isc = list(d.bus.isc)
mags = [math.hypot(isc[i], isc[i + 1]) for i in range(0, len(isc) - 1, 2)]
print("SCC_JSON:" + json.dumps({"bus": name, "kv_base_ln": float(d.bus.kv_base),
                                "isc_A": mags}))
"""

_SCC_FS_CACHE: dict[tuple[str, str], float] = {}


def _master_digest(master_path: Path) -> str:
    """Hash do master e do generated_network ao lado (chave do cache)."""
    h = hashlib.sha1()
    for p in (master_path, master_path.parent / "generated_network.dss"):
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()


def scc_mva_faultstudy(master_path: str | Path, bus: str, timeout_s: float = 180.0) -> float:
    """Scc trifasico (MVA) na barra `bus` pelo faultstudy do OpenDSS.

    Roda em subprocesso (nao toca o engine do processo atual): compila o
    master, `solve mode=faultstudy`, e usa Scc = sqrt(3) * kV_LL * media
    de |Isc| nas fases 1..3 (Bus.Isc). Inclui todas as fontes, trafos,
    interligacoes e geradores habilitados no .dss compilado. Mesma formula
    de work/verif/verif_scc.py, que confere com o analitico na P1.
    Resultado em cache por (hash do .dss, barra) no processo.
    """
    master = Path(os.path.abspath(str(master_path)))
    if not str(master).isascii():
        raise ValueError(f"master.dss em caminho nao-ASCII: {master}. Use build_instance.")
    key = (_master_digest(master), _bus_key(bus))
    if key in _SCC_FS_CACHE:
        return _SCC_FS_CACHE[key]
    proc = subprocess.run([sys.executable, "-c", _FAULTSTUDY_SCRIPT, str(master), str(bus)],
                          capture_output=True, text=True, timeout=timeout_s,
                          cwd=str(master.parent))
    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith("SCC_JSON:"):
            payload = json.loads(line[len("SCC_JSON:"):])
    if proc.returncode != 0 or payload is None:
        raise RuntimeError(f"faultstudy falhou (rc={proc.returncode}): "
                           f"{proc.stderr.strip()[-800:]}")
    if _bus_key(payload["bus"]) != _bus_key(bus):
        raise KeyError(f"barra {bus} inexistente no circuito (ativa: {payload['bus']})")
    mags = payload["isc_A"][:3]
    if len(mags) < 3:
        raise ValueError(f"barra {bus} com {len(mags)} nos; Scc trifasico indefinido")
    kv_ll = float(payload["kv_base_ln"]) * math.sqrt(3.0)
    scc = math.sqrt(3.0) * kv_ll * (sum(mags) / 3.0) / 1000.0
    _SCC_FS_CACHE[key] = scc
    return scc


# Metodo do Scc na feature de IC. "auto": analitico quando o modelo se
# aplica (P1: igual ao faultstudy), senao faultstudy em subprocesso.
SCC_METHODS = ("auto", "analytic", "faultstudy")


# ===========================================================================
# Avaliador de cenarios
# ===========================================================================
class ScenarioEvaluator:
    """Mantem um engine OpenDSS e avalia cenarios liga/desliga.

    `evaluate(motors_on)` executa run_maneuver (4 instantaneos) e devolve
    todas as grandezas brutas (ver ESPEC secao 1). As restricoes `g` sao
    nomeadas: planta 1 (sem `sag_monitored_buses`) => g1, g2, g3.
    """

    def __init__(self, system: SystemData, master_path: str | Path,
                 scc_method: str = "auto"):
        """`scc_method` (feature Scc_MVA): "auto" (analitico se aplicavel,
        senao faultstudy), "analytic" (NaN + aviso se nao aplicavel) ou
        "faultstudy". Na P1 os tres dao o mesmo valor (74,205 MVA em N-1)."""
        from .opendss_interface import OpenDSSInterface

        if scc_method not in SCC_METHODS:
            raise ValueError(f"scc_method invalido: {scc_method!r}; use {SCC_METHODS}")

        self.system = system
        cfg = system.config
        self.master_path = Path(os.path.abspath(str(master_path)))
        if not str(self.master_path).isascii():
            raise ValueError(
                "master.dss em caminho nao-ASCII (py-dss-interface 2.3.0 codifica "
                f"comandos em ASCII): {self.master_path}. Use build_instance.")
        self.target_motor_id = int(cfg["target_motor"]["motor_id"])
        self.other_motor_ids = [int(r["motor_id"]) for _, r in system.motors.iterrows()
                                if int(r["motor_id"]) != self.target_motor_id]
        self.other_kw_total = float(sum(float(r["kW"]) for _, r in system.motors.iterrows()
                                        if int(r["motor_id"]) != self.target_motor_id))
        lim = cfg["operational_limits"]
        self.v_ir = float(lim["v_min_inrush_pu"])
        self.v_sw = float(lim["v_max_switching_pu"])
        pen = cfg.get("penalty", {}) or {}
        self.pen_uncv = float(pen.get("unconverged", 5.0))
        self.pen_crit = float(pen.get("critical_violation", 3.0))

        # Barras monitoradas para o afundamento (g1).
        self.sag_buses = _sag_monitored_buses(cfg)
        # Trafos supervisionados (g3): padrao = primeiro em servico (sepoc_study).
        snap_out = set((cfg.get("contingency", {}) or {}).get("transformers_out", []) or [])
        ins = in_service_transformers(cfg)
        self.n_transformers = len(cfg.get("transformers", []) or [])
        self.n_in_service = len(ins)
        names = _loading_transformers(cfg)
        self.loading_explicit = names is not None
        if names is None:
            names = [str(ins[0]["name"])] if ins else []
        by_name = {str(t["name"]): t for t in cfg.get("transformers", []) or []}
        self.loading_trs: list[tuple[str, float, float]] = []
        for nm in names:
            if nm not in by_name:
                raise KeyError(f"trafo {nm} inexistente em transformers")
            if nm in snap_out:
                raise ValueError(f"trafo {nm} esta em transformers_out; nao ha medicao")
            t = by_name[nm]
            self.loading_trs.append((nm, float(t.get("kva_af", t["kva"])), float(t["kva"])))

        self.g_names = (["g1"] if self.sag_buses is None
                        else [f"g1_{b}" for b in self.sag_buses])
        self.g_names.append("g2")
        if not self.loading_explicit:
            self.g_names.append("g3")
        else:
            self.g_names += [f"g3_{nm}" for nm in names]

        # Features estaticas do alvo (IC).
        trow = system.motor_by_id(self.target_motor_id)
        self.target_bus = str(trow["bus"])
        self.scc_method = scc_method
        self.scc_note = scc_analytic_inapplicable_reason(system, self.target_bus)
        if scc_method == "faultstudy" or (scc_method == "auto" and self.scc_note is not None):
            try:
                scc = scc_mva_faultstudy(self.master_path, self.target_bus)
                self.scc_method_used = "faultstudy"
            except Exception as exc:  # NaN explicito, nunca valor errado silencioso
                warnings.warn(f"Scc_MVA por faultstudy falhou ({exc}); feature = NaN",
                              RuntimeWarning, stacklevel=2)
                scc, self.scc_method_used = float("nan"), "faultstudy_failed"
        else:
            scc = scc_mva_analytic(system, self.target_bus)  # NaN + aviso se nao aplica
            self.scc_method_used = "analytic"
        self.static_features = {
            "Scc_MVA": scc,
            "Pn_kW": float(trow["kW"]),
            "Ip_In": float(trow["inrush_multiplier"]),
            "cosphi_p": float(trow.get("starting_pf", 0.25)),
            "eta": float(trow["efficiency"]),
            "C_N1": int(self.n_in_service < self.n_transformers),
        }

        self.dss = OpenDSSInterface()
        self.dss.compile(self.master_path)
        # MEDIDO (2026-11-14, work/check_engine_sharing.py): instancias DSS()
        # do py-dss-interface no MESMO processo compartilham o engine; um
        # compile de outro avaliador troca o circuito deste. Guarda: o
        # datapath do engine (diretorio do master compilado) e conferido a
        # cada avaliacao; se mudou, recompila. Use uma instancia por processo.
        self._fingerprint = self._datapath()
        self.n_recompiles = 0

    def _datapath(self) -> str:
        try:
            return str(self.dss.dss.dssinterface.datapath)
        except Exception:  # pragma: no cover
            return ""

    def ensure_active(self) -> None:
        """Recompila se outro compile (outro avaliador/codigo) trocou o circuito."""
        if self._datapath() != self._fingerprint:
            log.warning("engine OpenDSS compartilhado trocou de circuito; recompilando %s",
                        self.master_path)
            self.dss.compile(self.master_path)
            self._fingerprint = self._datapath()
            self.n_recompiles += 1

    # ------------------------------------------------------------------
    def bits_to_motors_on(self, x) -> dict[int, bool]:
        x = np.asarray(x).astype(bool)
        return {mid: bool(x[k]) for k, mid in enumerate(self.other_motor_ids)}

    def _fallback(self, motors_on: dict[int, bool], exc: Exception) -> dict[str, Any]:
        g = {n: 1.0 for n in self.g_names}
        return {
            "exception": True, "error": str(exc), "converged": False,
            "v_min_inrush": np.nan, "v_max_post": np.nan,
            "v_min_inrush_by_bus": {}, "tr_load_af": np.nan,
            "tr_load_inrush_af": np.nan, "tr_load_nom": np.nan,
            "tr_load_af_by_tr": {},
            # Mesmos valores de recuo de sepoc_study (f5 = todas desligadas).
            "f1": np.nan, "f2": np.nan, "f3": 60.0, "f4": 500.0,
            "f5": self.other_kw_total,
            "g": g, "g_raw": dict(g), "penalty": np.nan,
            "critical_violation": False,
            "n_on": int(sum(motors_on.get(m, True) for m in self.other_motor_ids)),
            "features": {"Pprev_kW": np.nan, **self.static_features},
        }

    def evaluate(self, motors_on: dict[int, bool]) -> dict[str, Any]:
        """Avalia um cenario; nao faz nenhum comando DSS alem de run_maneuver."""
        scen = ScenarioState(motors_on=motors_on, target_motor_id=self.target_motor_id)
        self.ensure_active()
        try:
            snaps = run_maneuver(self.dss, self.system, scen)
        except Exception as exc:  # pragma: no cover
            log.error("run_maneuver falhou: %s", exc)
            return self._fallback(motors_on, exc)

        conv = (snaps.pre_energ.converged and snaps.inrush.converged
                and snaps.pre_desl.converged and snaps.post_desl.converged)
        v_min_inrush = snaps.inrush.v_min
        v_max_post = snaps.post_desl.v_max
        by_bus = dict(snaps.inrush.bus_voltages_pu_min)

        # Carregamentos (p.u. do AF e do nominal/AN).
        def _s(snap, nm):
            return snap.transformer_SkVA.get(nm, 0.0)

        load_af: dict[str, float] = {}
        load_nom: dict[str, float] = {}
        inrush_af: dict[str, float] = {}
        for nm, s_af, s_nom in self.loading_trs:
            sus = max(_s(snaps.pre_energ, nm) / s_af, _s(snaps.pre_desl, nm) / s_af,
                      _s(snaps.post_desl, nm) / s_af)
            load_af[nm] = sus
            load_nom[nm] = max(_s(snaps.pre_energ, nm), _s(snaps.pre_desl, nm),
                               _s(snaps.post_desl, nm)) / s_nom
            inrush_af[nm] = _s(snaps.inrush, nm) / s_af
        first = self.loading_trs[0][0] if self.loading_trs else None
        tr_sustained = load_af[first] if first else 0.0
        tr_inrush = inrush_af[first] if first else 0.0
        tr_nom = load_nom[first] if first else 0.0

        # Restricoes brutas (<= 0 viavel).
        g_raw: dict[str, float] = {}
        if self.sag_buses is None:
            g_raw["g1"] = self.v_ir - v_min_inrush
        else:
            low = {k.lower(): v for k, v in by_bus.items()}
            for b in self.sag_buses:
                if b.lower() not in low:
                    raise KeyError(f"barra monitorada {b} inexistente no circuito")
                g_raw[f"g1_{b}"] = self.v_ir - low[b.lower()]
        g_raw["g2"] = v_max_post - self.v_sw
        if not self.loading_explicit:
            g_raw["g3"] = tr_sustained - 1.0
        else:
            for nm, _, _ in self.loading_trs:
                g_raw[f"g3_{nm}"] = load_af[nm] - 1.0
        # Nao convergencia => todas valem 1 (como no artigo).
        g = dict(g_raw) if conv else {n: 1.0 for n in self.g_names}

        viol_inr = evaluate_protection(self.system, snaps.inrush, use_inrush_limits=True)
        viol_pos = evaluate_protection(self.system, snaps.post_desl, use_inrush_limits=True)
        viol_pre = evaluate_protection(self.system, snaps.pre_desl, use_inrush_limits=False)
        f3 = float(viol_inr.cascade_count + viol_pos.cascade_count + viol_pre.cascade_count)
        f4 = snaps.pre_desl.total_losses_kW
        f5 = total_idle_motor_kW(self.system, motors_on, self.target_motor_id)
        g1_vals = [v for k, v in g_raw.items() if k.startswith("g1")]
        f1 = max(0.0, max(g1_vals))  # P1: max(0, V_ir - Vmin); P2: max_s
        f2 = max(0.0, v_max_post - self.v_sw)

        penalty = 1.0
        if not conv:
            penalty *= self.pen_uncv
        critical = bool(viol_inr.critical_violation or viol_pos.critical_violation)
        if critical:
            penalty *= self.pen_crit

        return {
            "exception": False, "error": "", "converged": bool(conv),
            "v_min_inrush": v_min_inrush, "v_max_post": v_max_post,
            "v_min_inrush_by_bus": by_bus,
            "tr_load_af": tr_sustained, "tr_load_inrush_af": tr_inrush,
            "tr_load_nom": tr_nom, "tr_load_af_by_tr": load_af,
            "f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5,
            "g": g, "g_raw": g_raw, "penalty": penalty,
            "critical_violation": critical,
            "n_on": int(sum(motors_on.get(m, True) for m in self.other_motor_ids)),
            "features": {"Pprev_kW": snaps.pre_energ.total_power_kW,
                         **self.static_features},
        }

    def evaluate_bits(self, x) -> dict[str, Any]:
        return self.evaluate(self.bits_to_motors_on(x))


def evaluator_for(inst: PlantInstance, work_dir: str | Path) -> ScenarioEvaluator:
    """Atalho: build_instance + ScenarioEvaluator."""
    system, master = build_instance(inst, work_dir)
    return ScenarioEvaluator(system, master)


def copy_system(system: SystemData) -> SystemData:
    """Copia profunda de SystemData (para variar instancias sem efeito colateral)."""
    return SystemData(config=copy.deepcopy(system.config), buses=system.buses.copy(),
                      lines=system.lines.copy(), motors=system.motors.copy(),
                      loads=system.loads.copy(), data_dir=system.data_dir)
