"""Gerador da base OpenDSS para o trabalho de IC (ESPEC secao 5).

Uso:
  python -m src.ic.dataset --config experiments/ic/config.yaml \
      --strategy stratified_k --n 1500 --seed 1 --out experiments/ic/demo/x.csv \
      [--workers 4] [--block-size 250] [--resume]
  python -m src.ic.dataset --config experiments/ic/config.yaml \
      --validate 40 --seed 7 --out work/ic/validate [--plants P1 P2]

Espaco amostrado (ver experiments/ic/config.yaml): planta, contingencia
N-0/N-1, isc3 da fonte (faz o Scc variar), parametros do alvo (Pn, Ip/In,
cos phi_p, eta) por LHS ou entre os motores existentes, multiplicador da
carga estatica e o vetor liga/desliga das demais maquinas.

Estrategias do vetor liga/desliga:
  bernoulli     cada maquina ligada com prob. p (padrao 0,5);
  stratified_k  k ~ U{0..n} ligadas, subconjunto aleatorio de tamanho k;
  exhaustive    todos os 2^n vetores para uma condicao fixa (por blocos);
  nsga2_history bits unicos de results_sepoc/history_all.csv (P1, N-1).

Definicoes (podem diferir das do grupo de IC; documentado no README):
  Pprev_kW  potencia ativa total fornecida pela fonte no instantaneo
            PRE-ENERG (alvo desligado), inclui perdas;
  Scc_MVA   potencia de curto-circuito trifasica na barra do alvo, Thevenin
            da rede (cargas nao contribuem): analitico na P1; quando o
            analitico nao se aplica (P2: gerador/interligacao), faultstudy
            do OpenDSS no proprio engine logo apos recompilar o master
            (mesma definicao de scenario_core.scc_mva_faultstudy);
  C_N1      1 se a contingencia N-1 esta aplicada.

Desempenho: o circuito e compilado UMA vez por (processo, planta, alvo) e
cada amostra e aplicada por comandos `Edit` (fonte, trafos, cargas estaticas,
cargas do alvo); `--validate` compara com a recompilacao completa.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import datetime as _dt
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..config_loader import SystemData, load_system
from ..dss_builder import build as build_dss
from ..motor_model import compute_motor_electrical
from ..scenario_core import (PlantInstance, ScenarioEvaluator, ascii_compile_dir,
                             build_instance, scc_analytic_inapplicable_reason,
                             scc_mva_analytic, scc_mva_faultstudy)
from . import CONT_TARGETS, FEATURES, PROVENANCE, STRATEGIES

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "ic" / "config.yaml"
DATASET_VERSION = "ic-dataset-1 (2026-11-14)"

# Colunas da saida: as da especificacao, na ordem, e depois extras.
SPEC_COLUMNS = FEATURES + ["bits", "n_on"] + CONT_TARGETS + ["y"] + PROVENANCE
EXTRA_COLUMNS = ["plant", "target_motor_id", "contingency", "isc3_kA", "static_load_mult",
                 "pf_run", "converged", "v_min_label", "tr_load_inrush_af", "y_sepoc", "f3",
                 "scc_method", "eval_ms"]
SPEC_KEYS = ["sample_id", "plant", "target_motor_id", "contingency", "isc3_kA",
             "static_load_mult", "Pn_kW", "Ip_In", "cosphi_p", "eta", "pf_run", "bits"]


# ===========================================================================
# Configuracao
# ===========================================================================
def _abs(p: str | Path) -> Path:
    p = Path(p)
    return Path(os.path.abspath(str(p if p.is_absolute() else REPO_ROOT / p)))


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Le o yaml e resolve caminhos para absolutos (antes de qualquer engine,
    pois o py-dss-interface altera o cwd)."""
    path = _abs(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(path)
    for pid, p in (cfg.get("plants") or {}).items():
        p["data_dir"] = str(_abs(p["data_dir"]))
    cfg.setdefault("run", {})["work_dir"] = str(_abs(cfg["run"].get("work_dir", "work/ic")))
    nh = cfg.setdefault("nsga2_history", {})
    nh["path"] = str(_abs(nh.get("path", "results_sepoc/history_all.csv")))
    return cfg


def config_digest(cfg: dict) -> str:
    c = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.sha1(json.dumps(c, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ===========================================================================
# Planta base (sem OpenDSS)
# ===========================================================================
@dataclass
class PlantBase:
    plant_id: str
    data_dir: Path
    target_id: int
    system: SystemData
    other_ids: list[int]
    n1_out: list[str]
    trafo_names: list[str]
    isc3_kA: float
    isc1_ratio: float
    target_row: dict

    @property
    def n_bits(self) -> int:
        return len(self.other_ids)


_PLANT_CACHE: dict[tuple, PlantBase] = {}


def plant_base(cfg: dict, plant_id: str, target_id: int | None = None) -> PlantBase:
    """Dados estaticos da planta/alvo (cache por processo)."""
    pc = cfg["plants"][plant_id]
    data_dir = Path(pc["data_dir"])
    key0 = (plant_id, str(data_dir), target_id, str(pc.get("n1_transformers_out")))
    if key0 in _PLANT_CACHE:
        return _PLANT_CACHE[key0]
    system = load_system(data_dir)
    if target_id is None:
        target_id = pc.get("target_motor_id") or system.config["target_motor"]["motor_id"]
    key = (plant_id, str(data_dir), int(target_id), str(pc.get("n1_transformers_out")))
    if key in _PLANT_CACHE:
        _PLANT_CACHE[key0] = _PLANT_CACHE[key]
        return _PLANT_CACHE[key]
    tid = int(target_id)
    trow = system.motor_by_id(tid)
    other = [int(m) for m in system.motors["motor_id"] if int(m) != tid]
    ct = system.config.get("contingency", {}) or {}
    n1 = pc.get("n1_transformers_out") or ct.get("transformers_out") or []
    srcs = system.config.get("sources")
    src = srcs[0] if srcs else system.config["source"]
    pb = PlantBase(plant_id=plant_id, data_dir=data_dir, target_id=tid, system=system,
                   other_ids=other, n1_out=[str(t) for t in n1],
                   trafo_names=[str(t["name"]) for t in system.config["transformers"]],
                   isc3_kA=float(src["isc3_kA"]),
                   isc1_ratio=float(src["isc1_kA"]) / float(src["isc3_kA"]),
                   target_row={k: (v.item() if hasattr(v, "item") else v)
                               for k, v in trow.to_dict().items()})
    _PLANT_CACHE[key] = pb
    _PLANT_CACHE[key0] = pb
    return pb


def _existing_ids(cfg: dict, plant_id: str) -> list[int]:
    ids = ((cfg.get("target") or {}).get("existing_ids") or {}).get(plant_id)
    if ids:
        return [int(i) for i in ids]
    system = load_system(cfg["plants"][plant_id]["data_dir"])
    cand = system.config.get("target_candidates")
    return [int(i) for i in (cand or [system.config["target_motor"]["motor_id"]])]


def _pf_run(cfg: dict, pb: PlantBase) -> float:
    v = (cfg.get("target") or {}).get("pf_running", "base")
    return float(pb.target_row["pf"]) if v in (None, "base") else float(v)


def _base_target_params(cfg: dict, pb: PlantBase) -> dict:
    r = pb.target_row
    return {"target_motor_id": pb.target_id, "Pn_kW": float(r["kW"]),
            "Ip_In": float(r["inrush_multiplier"]), "cosphi_p": float(r.get("starting_pf", 0.25)),
            "eta": float(r["efficiency"]), "pf_run": float(r["pf"])}


# ===========================================================================
# Planos de amostragem (deterministicos pela semente; sem OpenDSS)
# ===========================================================================
def _lerp(u: np.ndarray, rng_: list[float]) -> np.ndarray:
    lo, hi = float(rng_[0]), float(rng_[1])
    return lo + u * (hi - lo)


def bits_bernoulli(rng: np.random.Generator, n_bits: int, p: float) -> str:
    return "".join("1" if b else "0" for b in (rng.random(n_bits) < p))


def bits_stratified_k(rng: np.random.Generator, n_bits: int) -> str:
    k = int(rng.integers(0, n_bits + 1))
    on = set(rng.choice(n_bits, size=k, replace=False).tolist()) if k else set()
    return "".join("1" if i in on else "0" for i in range(n_bits))


def bits_from_index(i: int, n_bits: int) -> str:
    """Indice da enumeracao -> bits (MSB = 1a maquina de other_ids)."""
    return format(int(i), f"0{n_bits}b")


def plan_random(cfg: dict, strategy: str, n: int, seed: int) -> pd.DataFrame:
    """Plano para bernoulli / stratified_k: condicao por LHS + vetor liga/desliga."""
    from scipy.stats import qmc

    if strategy not in ("bernoulli", "stratified_k"):
        raise ValueError(strategy)
    n = int(n)
    rng = np.random.default_rng(int(seed))
    probs = cfg.get("plant_probs") or {"P1": 1.0}
    pids = list(probs)
    pw = np.array([float(probs[k]) for k in pids], dtype=float)
    plants = rng.choice(pids, size=n, p=pw / pw.sum()) if len(pids) > 1 else np.array(pids * n)
    # Dimensoes LHS: contingencia, isc3, carga, Pn, Ip/In, cos phi_p, eta.
    U = qmc.LatinHypercube(d=7, seed=rng).random(n)
    p_n1 = float(cfg["contingency"]["p_N1"])
    cont = np.where(U[:, 0] < p_n1, "N-1", "N-0")
    isc3 = np.round(_lerp(U[:, 1], cfg["source"]["isc3_kA_range"]), 5)
    mult = np.round(_lerp(U[:, 2], cfg["static_load_mult_range"]), 6)
    tcfg = cfg["target"]
    mode = tcfg.get("mode", "lhs")
    ex_ids = {pid: _existing_ids(cfg, pid) for pid in pids} if mode == "existing" else {}
    rows = []
    for i in range(n):
        pid = str(plants[i])
        if mode == "lhs":
            pb = plant_base(cfg, pid)
            tp = {"target_motor_id": pb.target_id,
                  "Pn_kW": round(float(_lerp(U[i, 3], tcfg["Pn_kW_range"])), 3),
                  "Ip_In": round(float(_lerp(U[i, 4], tcfg["Ip_In_range"])), 5),
                  "cosphi_p": round(float(_lerp(U[i, 5], tcfg["cosphi_p_range"])), 5),
                  "eta": round(float(_lerp(U[i, 6], tcfg["eta_range"])), 5),
                  "pf_run": _pf_run(cfg, pb)}
        elif mode == "existing":
            tid = int(rng.choice(ex_ids[pid]))
            pb = plant_base(cfg, pid, tid)
            tp = _base_target_params(cfg, pb)
        else:
            raise ValueError(f"target.mode invalido: {mode}")
        if strategy == "bernoulli":
            bits = bits_bernoulli(rng, pb.n_bits, float(cfg["switching"]["bernoulli_p"]))
        else:
            bits = bits_stratified_k(rng, pb.n_bits)
        rows.append({"sample_id": i, "plant": pid, "contingency": str(cont[i]),
                     "isc3_kA": float(isc3[i]), "static_load_mult": float(mult[i]),
                     **tp, "bits": bits})
    return pd.DataFrame(rows, columns=SPEC_KEYS)


def exhaustive_condition(cfg: dict) -> tuple[dict, PlantBase]:
    ex = cfg["exhaustive"]
    pid = ex.get("plant", "P1")
    pb = plant_base(cfg, pid)
    if ex.get("target", "base") != "base":
        raise ValueError("exhaustive.target suporta apenas 'base'")
    cond = {"plant": pid, "contingency": ex.get("contingency", "N-1"),
            "isc3_kA": float(ex.get("isc3_kA") or pb.isc3_kA),
            "static_load_mult": float(ex.get("static_load_mult", 1.0)),
            **_base_target_params(cfg, pb)}
    return cond, pb


def plan_exhaustive_range(cfg: dict, start: int, stop: int) -> pd.DataFrame:
    cond, pb = exhaustive_condition(cfg)
    rows = [{"sample_id": i, **cond, "bits": bits_from_index(i, pb.n_bits)}
            for i in range(int(start), int(stop))]
    return pd.DataFrame(rows, columns=SPEC_KEYS)


def plan_history(cfg: dict, n: int | None, seed: int) -> tuple[pd.DataFrame, dict]:
    """Bits unicos do historico do NSGA-II (condicao do artigo SEPOC)."""
    nh = cfg["nsga2_history"]
    pb = plant_base(cfg, "P1")
    h = pd.read_csv(nh["path"], dtype={"bits": str})
    h["bits"] = h["bits"].str.zfill(pb.n_bits)
    algos = nh.get("algorithms")
    if algos:
        h = h[h["algorithm"].isin(list(algos))]
    u = h.drop_duplicates(subset="bits", keep="first").reset_index(drop=True)
    info = {"history_path": nh["path"], "algorithms": algos, "n_history_rows": int(len(h)),
            "n_unique_bits": int(len(u))}
    cap = n if n is not None else nh.get("max_unique")
    if cap is not None and int(cap) < len(u):
        if nh.get("selection", "random") == "first":
            u = u.iloc[: int(cap)]
        else:
            idx = np.sort(np.random.default_rng(int(seed)).choice(len(u), int(cap), replace=False))
            u = u.iloc[idx]
        info["truncated"] = True
    else:
        info["truncated"] = False
    info["n_selected"] = int(len(u))
    cond = {"plant": "P1", "contingency": "N-1", "isc3_kA": pb.isc3_kA,
            "static_load_mult": 1.0, **_base_target_params(cfg, pb)}
    df = pd.DataFrame([{"sample_id": k, **cond, "bits": b} for k, b in enumerate(u["bits"])],
                      columns=SPEC_KEYS)
    df["hist_algorithm"] = u["algorithm"].to_numpy()
    df["hist_seed"] = u["seed"].to_numpy()
    df["hist_v_min_inrush"] = u["v_min_inrush"].to_numpy()
    df["hist_feasible"] = u["feasible"].to_numpy()
    return df, info


# ===========================================================================
# Rotulo
# ===========================================================================
def label_value(row: dict, lab: dict) -> int:
    """y = 1 (viavel) / 0 (critico); regra configuravel (padrao: artigo de IC)."""
    vals = [row.get("v_min_label"), row.get("v_max_post")]
    basis = lab.get("loading_basis", "nominal")
    load = row.get("tr_load_nom") if basis == "nominal" else row.get("tr_load_af")
    if basis not in ("nominal", "af"):
        raise ValueError(f"label.loading_basis invalido: {basis}")
    vals.append(load)
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals):
        return 0
    if lab.get("require_converged", True) and not bool(row.get("converged")):
        return 0
    ok = (float(row["v_min_label"]) > float(lab.get("v_min_gt", 0.85))
          and float(row["v_max_post"]) <= float(lab.get("v_max_post_le", 1.08))
          and float(load) <= float(lab.get("loading_max", 1.0)))
    return int(ok)


# ===========================================================================
# Avaliador com Edit
# ===========================================================================
def _fmt(v: float) -> str:
    return repr(float(v))


class ICScenarioEvaluator(ScenarioEvaluator):
    """ScenarioEvaluator do nucleo (sem edita-lo) + aplicacao da condicao.

    `apply_edits=True`: a condicao (fonte, trafos, cargas estaticas, cargas
    do alvo) e aplicada por `Edit` no circuito compilado. `False`: o .dss ja
    reflete a condicao (caminho de referencia por recompilacao); so a
    contabilidade em Python (contingencia, trafos medidos) e atualizada.
    """

    def __init__(self, system: SystemData, master_path: str | Path, base: PlantBase,
                 cfg: dict, apply_edits: bool = True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            super().__init__(system, master_path, scc_method="auto")
        self.base = base
        self.cfg = cfg
        self.apply_edits = apply_edits
        self.scc_cfg_method = (cfg.get("scc") or {}).get("method", "auto")
        self.trafos = [(str(t["name"]), float(t.get("kva_af", t["kva"])), float(t["kva"]))
                       for t in system.config["transformers"]]
        self.static_loads = [(str(r["name"]), float(r["kW"]), float(r["kvar"]),
                              str(r["initial_state"]))
                             for _, r in base.system.loads.iterrows()]
        trow = system.motor_by_id(self.target_motor_id)
        self.target_name = str(trow["name"])
        self.motor_names = [str(m) for m in system.motors["name"]]
        self.label_cfg = cfg.get("label") or {}
        self._scc_cache: dict[tuple, float] = {}

    # -------------------------------------------------------------- condicao
    def _set_bookkeeping(self, spec: dict) -> set[str]:
        cfg = self.system.config
        out = set(self.base.n1_out) if spec["contingency"] == "N-1" else set()
        if spec["contingency"] not in ("N-0", "N-1"):
            raise ValueError(f"contingencia invalida: {spec['contingency']}")
        cfg["contingency"] = {"mode": spec["contingency"], "transformers_out": sorted(out)}
        isc3 = float(spec["isc3_kA"])
        isc1 = round(isc3 * self.base.isc1_ratio, 6)
        if cfg.get("sources"):
            cfg["sources"][0]["isc3_kA"] = isc3
            cfg["sources"][0]["isc1_kA"] = isc1
        if cfg.get("source"):
            cfg["source"]["isc3_kA"] = isc3
            cfg["source"]["isc1_kA"] = isc1
        # Trafos medidos: todos os em servico (g3_<nome> no AF; rotulo usa AN).
        self.loading_trs = [t for t in self.trafos if t[0] not in out]
        self.loading_explicit = True
        self.n_in_service = len(self.loading_trs)
        g = (["g1"] if self.sag_buses is None else [f"g1_{b}" for b in self.sag_buses])
        self.g_names = g + ["g2"] + [f"g3_{t[0]}" for t in self.loading_trs]
        return out

    def _edit_source_trafos(self, spec: dict, out: set[str]) -> None:
        d = self.dss.dss
        isc3 = float(spec["isc3_kA"])
        isc1 = round(isc3 * self.base.isc1_ratio, 6)
        d.text(f"Edit Vsource.source isc3={isc3 * 1000.0:.2f} isc1={isc1 * 1000.0:.2f}")
        for nm, _, _ in self.trafos:
            d.text(f"Edit Transformer.{nm} enabled={'no' if nm in out else 'yes'}")

    def _scc_engine_faultstudy(self, bus: str, spec: dict, out: set[str]) -> float:
        """Faultstudy no proprio engine, a partir de um circuito RECEM
        compilado (mesmo master, ~ms) com fonte e trafos editados.

        MEDIDO (2026-11-14, work/logs/ic_check_p2_scc.log): logo apos o
        compile o faultstudy nao depende das cargas (P2: 77,348 x 77,352 MVA
        com cargas desligadas), mas depois de fluxos de potencia o
        resultado muda com o estado do gerador (P2: 78,47 x 77,35 MVA).
        Por isso recompila antes. As cargas sao reeditadas em seguida e os
        motores ajustados por run_maneuver.
        """
        d = self.dss.dss
        self.dss.compile(self.master_path)
        self._fingerprint = self._datapath()
        self._edit_source_trafos(spec, out)
        d.text("solve mode=faultstudy")
        d.circuit.set_active_bus(bus)
        if str(d.bus.name).lower() != bus.split(".")[0].lower():
            d.text("set mode=snap")
            raise KeyError(f"barra {bus} inexistente")
        isc = list(d.bus.isc)
        mags = [math.hypot(isc[i], isc[i + 1]) for i in range(0, len(isc) - 1, 2)][:3]
        kv_ll = float(d.bus.kv_base) * math.sqrt(3.0)
        d.text("set mode=snap")
        return math.sqrt(3.0) * kv_ll * (sum(mags) / len(mags)) / 1000.0

    def _scc(self, spec: dict, out: set[str]) -> tuple[float, str]:
        m = self.scc_cfg_method
        if not self.apply_edits:
            # Referencia: metodo do nucleo sobre o .dss completo da condicao.
            return float(self.static_features["Scc_MVA"]), "core_" + self.scc_method_used
        if m == "analytic" or (m == "auto" and
                               scc_analytic_inapplicable_reason(self.system, self.target_bus) is None):
            return scc_mva_analytic(self.system, self.target_bus, warn=False), "analytic"
        if m in ("auto", "engine_faultstudy"):
            # Cache por condicao (contingencia, isc3): na enumeracao exaustiva
            # (condicao fixa) evita recompilar a cada amostra.
            key = (spec["contingency"], round(float(spec["isc3_kA"]), 6))
            if key not in self._scc_cache:
                self._scc_cache[key] = self._scc_engine_faultstudy(self.target_bus, spec, out)
            return self._scc_cache[key], "engine_faultstudy"
        raise ValueError(f"scc.method invalido: {m}")

    def apply_spec(self, spec: dict) -> tuple[float, str]:
        """Aplica a condicao da amostra. Sempre reaplica tudo (nao depende
        do estado anterior do engine)."""
        self.ensure_active()  # antes das edicoes: um recompile apagaria as edicoes
        out = self._set_bookkeeping(spec)
        d = self.dss.dss
        if self.apply_edits:
            self._edit_source_trafos(spec, out)
        scc, method = self._scc(spec, out)
        if self.apply_edits:
            m = float(spec["static_load_mult"])
            for nm, kw, kvar, st in self.static_loads:
                if m != 1.0:  # mesma regra/arredondamento de build_instance
                    kw, kvar = round(kw * m, 6), round(kvar * m, 6)
                d.text(f"Edit Load.{nm} kW={_fmt(kw)} kvar={_fmt(kvar)} "
                       f"enabled={'yes' if st == 'on' else 'no'}")
            e = compute_motor_electrical(_target_row(self.base, spec))
            kva_lr = e.kVA_in * e.inrush_multiplier
            kw_lr = kva_lr * e.starting_pf
            kvar_lr = math.sqrt(max(kva_lr ** 2 - kw_lr ** 2, 0.0))
            d.text(f"Edit Load.{self.target_name} kW={e.kW_in:.3f} kvar={e.kvar_in:.3f}")
            d.text(f"Edit Load.{self.target_name}_INRUSH kW={kw_lr:.3f} kvar={kvar_lr:.3f}")
        return scc, method

    # -------------------------------------------------------------- amostra
    def evaluate_spec(self, spec: dict) -> dict:
        t0 = time.perf_counter()
        scc, method = self.apply_spec(spec)
        bits = str(spec["bits"])
        if len(bits) != len(self.other_motor_ids):
            raise ValueError(f"bits com {len(bits)} posicoes; esperado {len(self.other_motor_ids)}")
        motors_on = {mid: bits[k] == "1" for k, mid in enumerate(self.other_motor_ids)}
        r = self.evaluate(motors_on)
        eval_ms = (time.perf_counter() - t0) * 1000.0
        by_tr = r.get("tr_load_af_by_tr") or {}
        kv = {nm: (af, kva) for nm, af, kva in self.trafos}
        load_af = max(by_tr.values()) if by_tr else float("nan")
        load_nom = (max(v * kv[nm][0] / kv[nm][1] for nm, v in by_tr.items())
                    if by_tr else float("nan"))
        vb = self.label_cfg.get("vmin_buses")
        if vb:
            low = {k.lower(): v for k, v in (r.get("v_min_inrush_by_bus") or {}).items()}
            v_lab = min(low.get(str(b).lower(), float("nan")) for b in vb)
        else:
            v_lab = r["v_min_inrush"]
        g_raw = r.get("g_raw") or {}
        row = {
            "Pprev_kW": r["features"]["Pprev_kW"], "Scc_MVA": scc,
            "Pn_kW": float(spec["Pn_kW"]), "Ip_In": float(spec["Ip_In"]),
            "cosphi_p": float(spec["cosphi_p"]), "eta": float(spec["eta"]),
            "C_N1": int(spec["contingency"] == "N-1"),
            "bits": bits, "n_on": int(bits.count("1")),
            "v_min_inrush": r["v_min_inrush"], "v_max_post": r["v_max_post"],
            "tr_load_nom": load_nom, "tr_load_af": load_af,
            "plant": spec["plant"], "target_motor_id": int(spec["target_motor_id"]),
            "contingency": spec["contingency"], "isc3_kA": float(spec["isc3_kA"]),
            "static_load_mult": float(spec["static_load_mult"]), "pf_run": float(spec["pf_run"]),
            "converged": bool(r["converged"]), "v_min_label": v_lab,
            "tr_load_inrush_af": r["tr_load_inrush_af"],
            "y_sepoc": int(bool(r["converged"]) and bool(g_raw)
                           and all(float(v) <= 0 for v in g_raw.values())),
            "f3": r["f3"], "scc_method": method, "eval_ms": eval_ms,
        }
        row["y"] = label_value(row, self.label_cfg)
        return row


def _target_row(pb: PlantBase, spec: dict) -> dict:
    row = dict(pb.target_row)
    row.update({"kW": float(spec["Pn_kW"]), "inrush_multiplier": float(spec["Ip_In"]),
                "starting_pf": float(spec["cosphi_p"]), "efficiency": float(spec["eta"]),
                "pf": float(spec["pf_run"])})
    return row


# ===========================================================================
# Execucao (por processo)
# ===========================================================================
_EVALUATORS: dict[tuple, ICScenarioEvaluator] = {}


def get_evaluator(cfg: dict, plant: str, target_id: int) -> ICScenarioEvaluator:
    """Um avaliador por (planta, alvo) por processo; .dss em diretorio proprio."""
    key = (plant, int(target_id), config_digest(cfg))
    if key not in _EVALUATORS:
        pb = plant_base(cfg, plant, target_id)
        inst = PlantInstance(f"ic_{plant}_t{int(target_id)}_p{os.getpid()}", plant,
                             pb.data_dir, int(target_id))
        system, master = build_instance(inst, cfg["run"]["work_dir"])
        _EVALUATORS[key] = ICScenarioEvaluator(system, master, pb, cfg, apply_edits=True)
    return _EVALUATORS[key]


def evaluate_specs(cfg: dict, specs: pd.DataFrame, strategy: str, seed: int) -> pd.DataFrame:
    rows = []
    extra_cols = [c for c in specs.columns if c.startswith("hist_")]
    for spec in specs.to_dict("records"):
        ev = get_evaluator(cfg, spec["plant"], spec["target_motor_id"])
        row = ev.evaluate_spec(spec)
        row.update({"strategy": strategy, "seed": int(seed), "sample_id": int(spec["sample_id"])})
        for c in extra_cols:
            row[c] = spec[c]
        rows.append(row)
    cols = SPEC_COLUMNS + EXTRA_COLUMNS + extra_cols
    return pd.DataFrame(rows, columns=cols)


def _write_atomic_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _block_job(cfg: dict, strategy: str, seed: int, block_path: str,
               specs: pd.DataFrame | None, rng_: tuple[int, int] | None) -> tuple[str, int, float]:
    t0 = time.perf_counter()
    if specs is None:
        specs = plan_exhaustive_range(cfg, *rng_)
    df = evaluate_specs(cfg, specs, strategy, seed)
    _write_atomic_csv(df, Path(block_path))
    return block_path, len(df), time.perf_counter() - t0


def _versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform(),
           "dataset_version": DATASET_VERSION}
    for mod in ("py_dss_interface", "numpy", "pandas", "scipy", "sklearn"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:  # pragma: no cover
            out[mod] = None
    return out


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def run_dataset(cfg: dict, strategy: str, n: int | None, seed: int, out: str | Path,
                workers: int | None = None, block_size: int | None = None,
                resume: bool = False, cmdline: str = "") -> dict:
    """Gera o dataset por blocos (retomavel) e grava CSV + .meta.json."""
    if strategy not in STRATEGIES:
        raise ValueError(f"estrategia invalida: {strategy}; use {STRATEGIES}")
    started = _now()
    t_wall = time.perf_counter()
    out = Path(os.path.abspath(str(out)))
    out.parent.mkdir(parents=True, exist_ok=True)
    blocks_dir = out.parent / (out.stem + ".blocks")
    workers = int(workers or cfg["run"].get("workers", 1))
    bs = int(block_size or cfg["run"].get("block_size", 250))

    plan_info: dict[str, Any] = {}
    jobs: list[tuple[str, pd.DataFrame | None, tuple[int, int] | None, int]] = []
    if strategy == "exhaustive":
        cond, pb = exhaustive_condition(cfg)
        total = 2 ** pb.n_bits
        n_eff = total if n is None else min(int(n), total)
        plan_info = {"condition": cond, "n_bits": pb.n_bits, "n_total_space": total,
                     "n_evaluated_indices": n_eff, "truncated": n_eff < total}
        for b, s in enumerate(range(0, n_eff, bs)):
            e = min(s + bs, n_eff)
            jobs.append((str(blocks_dir / f"block_{b:06d}.csv"), None, (s, e), e - s))
    else:
        if strategy == "nsga2_history":
            plan, plan_info = plan_history(cfg, n, seed)
        else:
            if n is None:
                raise ValueError("--n e obrigatorio para bernoulli/stratified_k (sem teto)")
            plan = plan_random(cfg, strategy, int(n), seed)
        for b, s in enumerate(range(0, len(plan), bs)):
            part = plan.iloc[s: s + bs].reset_index(drop=True)
            jobs.append((str(blocks_dir / f"block_{b:06d}.csv"), part, None, len(part)))

    sig = {"strategy": strategy, "n": n, "seed": int(seed), "block_size": bs,
           "config_digest": config_digest(cfg), "dataset_version": DATASET_VERSION}
    sig_path = blocks_dir / "plan.json"
    if blocks_dir.exists() and sig_path.exists():
        old = json.loads(sig_path.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError(f"{blocks_dir} ja existe; use --resume ou outro --out")
        if old != sig:
            raise ValueError(f"--resume com plano diferente do gravado em {sig_path}: {old} != {sig}")
    blocks_dir.mkdir(parents=True, exist_ok=True)
    sig_path.write_text(json.dumps(sig, indent=2), encoding="utf-8")

    def _done(path: str, expected: int) -> bool:
        p = Path(path)
        if not p.is_file():
            return False
        try:
            return len(pd.read_csv(p, usecols=["sample_id"])) == expected
        except Exception:
            return False

    todo = [j for j in jobs if not _done(j[0], j[3])]
    reused = len(jobs) - len(todo)
    log.info("%s: %d blocos (%d reaproveitados), workers=%d", strategy, len(jobs), reused, workers)
    block_times: list[float] = []
    if workers <= 1:
        for path, specs, rr, _ in todo:
            _, _, dt = _block_job(cfg, strategy, seed, path, specs, rr)
            block_times.append(dt)
    else:
        ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = [ex.submit(_block_job, cfg, strategy, seed, path, specs, rr)
                    for path, specs, rr, _ in todo]
            for f in cf.as_completed(futs):
                block_times.append(f.result()[2])

    df = pd.concat([pd.read_csv(j[0], dtype={"bits": str}) for j in jobs], ignore_index=True)
    df = df.sort_values("sample_id", kind="stable").reset_index(drop=True)
    _write_atomic_csv(df, out)
    wall = time.perf_counter() - t_wall
    em = df["eval_ms"].to_numpy(float)
    meta = {
        "created": started, "finished": _now(), "cmdline": cmdline,
        "strategy": strategy, "n_requested": n, "seed": int(seed), "out": str(out),
        "n_rows": int(len(df)), "n_blocks": len(jobs), "n_blocks_reused": reused,
        "n_blocks_computed": len(todo), "workers": workers, "block_size": bs,
        "wall_s_this_invocation": wall,
        "rows_per_s_wall": (sum(j[3] for j in todo) / wall) if todo else None,
        "eval_ms_per_sample": {"mean": float(np.mean(em)), "median": float(np.median(em)),
                               "p95": float(np.percentile(em, 95)), "max": float(np.max(em)),
                               "note": "tempo de Edit+4 fluxos+extracao por amostra, num processo"},
        "class_balance": {"y1": int(df["y"].sum()), "y0": int((1 - df["y"]).sum())},
        "n_unconverged": int((~df["converged"].astype(bool)).sum()),
        "scc_methods": df["scc_method"].value_counts().to_dict(),
        "plan_info": plan_info, "label_rule": cfg.get("label"),
        "definitions": {
            "Pprev_kW": "P ativa total da fonte no instantaneo PRE-ENERG (alvo desligado), com perdas",
            "Scc_MVA": "Scc trifasico na barra do alvo, Thevenin sem cargas (analitico ou faultstudy)",
            "bits": "bit k = maquina other_ids[k] (motor_ids da planta sem o alvo, ordem do csv)",
            "tr_load_nom": "max sobre trafos em servico de max(S PRE-ENERG, PRE-DESL, POS-DESL)/kVA AN",
            "tr_load_af": "idem, sobre kVA AF",
            "y_sepoc": "viabilidade da formulacao restrita (g1, g2, g3 por trafo em servico no AF)",
        },
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "config_path": cfg.get("_config_path"), "config_digest": config_digest(cfg),
        "versions": _versions(),
    }
    meta_path = out.parent / (out.stem + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str, ensure_ascii=True), encoding="utf-8")
    return meta


# ===========================================================================
# Validacao: Edit x recompilacao; Scc analitico x faultstudy
# ===========================================================================
def _overrides_for(pb: PlantBase, spec: dict) -> dict:
    isc3 = float(spec["isc3_kA"])
    isc1 = round(isc3 * pb.isc1_ratio, 6)
    cfg = pb.system.config
    ov: dict[str, Any] = {}
    if cfg.get("sources"):
        srcs = copy.deepcopy(cfg["sources"])
        srcs[0]["isc3_kA"], srcs[0]["isc1_kA"] = isc3, isc1
        ov["sources"] = srcs
    if cfg.get("source"):
        ov["source"] = {"isc3_kA": isc3, "isc1_kA": isc1}
    ov["contingency"] = ({"mode": "N-1", "transformers_out": list(pb.n1_out)}
                         if spec["contingency"] == "N-1" else {"mode": "N-0", "transformers_out": []})
    return ov


def build_reference(cfg: dict, spec: dict, work_dir: Path, tag: str) -> tuple[SystemData, Path, PlantBase]:
    """.dss COMPLETO da condicao (sem Edit): build_instance + motor alvo alterado."""
    pb = plant_base(cfg, spec["plant"], int(spec["target_motor_id"]))
    inst = PlantInstance(tag, spec["plant"], pb.data_dir, int(spec["target_motor_id"]),
                         float(spec["static_load_mult"]), _overrides_for(pb, spec))
    system, _ = build_instance(inst, work_dir)
    mot = system.motors.copy()
    i = mot.index[mot["motor_id"].astype(int) == int(spec["target_motor_id"])][0]
    tr = _target_row(pb, spec)
    for c in ("kW", "inrush_multiplier", "starting_pf", "efficiency", "pf"):
        mot[c] = mot[c].astype(float)
        mot.loc[i, c] = tr[c]
    system.motors = mot
    out_dir = Path(work_dir) / tag
    master, network = build_dss(system, out_dir)
    comp = ascii_compile_dir(out_dir)
    if comp != out_dir:
        comp.mkdir(parents=True, exist_ok=True)
        (comp / "master.dss").write_bytes(master.read_bytes())
        (comp / "generated_network.dss").write_bytes(network.read_bytes())
        master = comp / "master.dss"
    return system, master, pb


def validate(cfg: dict, n: int, seed: int, out_dir: str | Path,
             plants: list[str] | None = None) -> dict:
    """Compara o caminho Edit com a recompilacao em `n` amostras por planta."""
    out_dir = Path(os.path.abspath(str(out_dir)))
    out_dir.mkdir(parents=True, exist_ok=True)
    plants = plants or ["P1"]
    specs = []
    for pid in plants:
        c = copy.deepcopy(cfg)
        c["plant_probs"] = {pid: 1.0}
        p = plan_random(c, "stratified_k", n, seed)
        pb = plant_base(cfg, pid)
        base = {"plant": pid, "isc3_kA": pb.isc3_kA, "static_load_mult": 1.0,
                **_base_target_params(cfg, pb)}
        ext = pd.DataFrame([{**base, "contingency": "N-1", "bits": "1" * pb.n_bits},
                            {**base, "contingency": "N-0", "bits": "1" * pb.n_bits},
                            {**base, "contingency": "N-1", "bits": "0" * pb.n_bits}])
        specs.append(pd.concat([p, ext], ignore_index=True))
    plan = pd.concat(specs, ignore_index=True)
    plan["sample_id"] = np.arange(len(plan))
    # Edit: ordem embaralhada (estado anterior do engine diferente a cada amostra).
    order = np.random.default_rng(seed + 1).permutation(len(plan))
    t0 = time.perf_counter()
    edit = evaluate_specs(cfg, plan.iloc[order].reset_index(drop=True), "validate_edit", seed)
    t_edit = time.perf_counter() - t0
    edit = edit.sort_values("sample_id").reset_index(drop=True)
    refs, t_ref = [], 0.0
    for spec in plan.to_dict("records"):
        t1 = time.perf_counter()
        tag = f"icref_{spec['plant']}_{int(spec['sample_id']):04d}_p{os.getpid()}"
        system, master, pb = build_reference(cfg, spec, out_dir / "ref_dss", tag)
        ev = ICScenarioEvaluator(system, master, pb, cfg, apply_edits=False)
        row = ev.evaluate_spec(spec)
        t_ref += time.perf_counter() - t1
        row["Scc_faultstudy_subproc"] = scc_mva_faultstudy(master, ev.target_bus)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            row["Scc_analytic_ref"] = scc_mva_analytic(system, ev.target_bus, warn=False)
        row["sample_id"] = int(spec["sample_id"])
        refs.append(row)
        del ev
    ref = pd.DataFrame(refs).sort_values("sample_id").reset_index(drop=True)
    cmp_cols = ["v_min_inrush", "v_max_post", "Pprev_kW", "tr_load_nom", "tr_load_af",
                "tr_load_inrush_af", "Scc_MVA"]
    comp = plan[["sample_id", "plant", "contingency", "isc3_kA", "static_load_mult",
                 "Pn_kW", "bits"]].copy()
    summary: dict[str, Any] = {"n_samples": int(len(plan)), "plants": plants, "seed": int(seed),
                               "t_edit_s": t_edit, "t_ref_s": t_ref,
                               "ms_per_sample_edit": 1000 * t_edit / len(plan),
                               "ms_per_sample_rebuild_compile": 1000 * t_ref / len(plan)}
    for c in cmp_cols:
        comp[f"{c}_edit"] = edit[c].to_numpy(float)
        comp[f"{c}_ref"] = ref[c].to_numpy(float)
        comp[f"d_{c}"] = comp[f"{c}_edit"] - comp[f"{c}_ref"]
    comp["Scc_faultstudy_subproc"] = ref["Scc_faultstudy_subproc"].to_numpy(float)
    comp["Scc_analytic_ref"] = ref["Scc_analytic_ref"].to_numpy(float)
    comp["scc_method_edit"] = edit["scc_method"].to_numpy()
    comp["d_Scc_edit_vs_fs"] = comp["Scc_MVA_edit"] - comp["Scc_faultstudy_subproc"]
    comp["d_Scc_analytic_vs_fs"] = comp["Scc_analytic_ref"] - comp["Scc_faultstudy_subproc"]
    comp["y_edit"], comp["y_ref"] = edit["y"].to_numpy(), ref["y"].to_numpy()
    comp["conv_edit"], comp["conv_ref"] = edit["converged"].to_numpy(), ref["converged"].to_numpy()
    for pid in plants:
        m = comp["plant"] == pid
        s: dict[str, Any] = {}
        for c in cmp_cols + ["Scc_edit_vs_fs", "Scc_analytic_vs_fs"]:
            col = f"d_{c}"
            v = comp.loc[m, col].abs()
            s[f"max_abs_{col}"] = None if v.isna().all() else float(v.max())
        s["label_mismatch"] = int((comp.loc[m, "y_edit"] != comp.loc[m, "y_ref"]).sum())
        s["converged_mismatch"] = int((comp.loc[m, "conv_edit"] != comp.loc[m, "conv_ref"]).sum())
        base_n1 = comp[m & (comp["contingency"] == "N-1")
                       & (comp["bits"].str.count("1") == comp["bits"].str.len())]
        if len(base_n1):
            r0 = base_n1.iloc[-1]
            s["base_N1_all_on"] = {"Scc_edit": float(r0["Scc_MVA_edit"]),
                                   "Scc_faultstudy_subproc": float(r0["Scc_faultstudy_subproc"]),
                                   "v_min_inrush_edit": float(r0["v_min_inrush_edit"]),
                                   "Pprev_kW_edit": float(r0["Pprev_kW_edit"])}
        summary[pid] = s
    comp.to_csv(out_dir / "validate_edit_vs_recompile.csv", index=False)
    (out_dir / "validate_summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                                   encoding="utf-8")
    return summary


# ===========================================================================
# CLI
# ===========================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Base OpenDSS para IC (ESPEC secao 5)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--strategy", choices=STRATEGIES)
    ap.add_argument("--n", type=int, default=None, help="tamanho (sem teto); exhaustive: corta nos n primeiros indices")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--block-size", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--validate", type=int, default=None, metavar="N",
                    help="valida Edit x recompilacao em N amostras por planta")
    ap.add_argument("--plants", nargs="*", default=None)
    ap.add_argument("--history-algorithms", nargs="*", default=None,
                    help="nsga2_history: algoritmos do historico (padrao: config)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(a.config)
    cmd = "python -m src.ic.dataset " + " ".join(argv if argv is not None else sys.argv[1:])
    if a.validate:
        s = validate(cfg, a.validate, a.seed, a.out, a.plants)
        print(json.dumps(s, indent=2, default=str))
        return 0
    if not a.strategy:
        ap.error("--strategy e obrigatorio (ou use --validate)")
    if a.plants:  # sobrepoe plant_probs (uniforme entre as plantas listadas)
        cfg["plant_probs"] = {p: 1.0 for p in a.plants}
    if a.history_algorithms:  # sobrepoe nsga2_history.algorithms
        cfg["nsga2_history"]["algorithms"] = list(a.history_algorithms)
    meta = run_dataset(cfg, a.strategy, a.n, a.seed, a.out, a.workers, a.block_size,
                       a.resume, cmd)
    print(json.dumps({k: meta[k] for k in ("strategy", "n_rows", "n_blocks", "n_blocks_reused",
                                           "workers", "wall_s_this_invocation", "rows_per_s_wall",
                                           "eval_ms_per_sample", "class_balance",
                                           "n_unconverged", "scc_methods", "plan_info")},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
