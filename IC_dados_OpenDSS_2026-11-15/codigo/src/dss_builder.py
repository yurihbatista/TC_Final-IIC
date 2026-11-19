"""Construtor de arquivos OpenDSS .dss a partir dos CSVs.

Suporta:
  - N transformadores em paralelo (lista em system_config.yaml > transformers).
  - Contingencia N-1 declarativa: o YAML pode listar trafos fora de operacao
    em contingency.transformers_out.
  - Topologia alimentador unico (todos os motores conectados a uma unica barra MV).

Gera dss/master.dss e dss/generated_network.dss.

Generalizacao ADITIVA (2026-11-14, ESPEC_BASE_CASOS secao 3); todos os
campos sao opcionais e, ausentes, o .dss sai identico ao original:
  - transformers[*].bus_hv / bus_lv (padrao: system.source_bus / mv_main_bus);
  - sources: lista; a primeira vira o Circuit, as demais Vsource
    (contingency.sources_out desabilita Vsources adicionais);
  - generators: lista de Generator (chaves extras viram parametros DSS;
    contingency.generators_out desabilita);
  - lines.csv: colunas opcionais `switch` (Line switch=yes) e `enabled`;
  - Voltagebases inclui bases extras (ex.: 0,48 kV) presentes em buses.csv,
    transformadores ou sources.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from .config_loader import SystemData
from .motor_model import compute_motor_electrical, motor_as_dss_load, motor_as_dss_load_inrush

HEADER = (
    "! =============================================================\n"
    "! Sistema gerado automaticamente por nsga2-vcb-decision\n"
    "! Nao editar manualmente - regerar via dss_builder.build()\n"
    "! =============================================================\n"
)


def build(system: SystemData, output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(os.path.abspath(os.path.expanduser(str(output_dir))))
    output_dir.mkdir(parents=True, exist_ok=True)
    master_path = output_dir / "master.dss"
    network_path = output_dir / "generated_network.dss"
    _write_master(system, master_path)
    _write_network(system, network_path)
    return master_path, network_path


def _primary_source(cfg: dict) -> tuple[str, float, dict]:
    """(barra, basekv, dict da fonte) da fonte que vira o Circuit.

    Generalizacao aditiva (2026-11-14): se `sources:` (lista) existir, a
    primeira entrada vira o Circuit; senao, usa `system.source_bus`,
    `system.base_kV_HV` e `source:` (comportamento original).
    """
    sources = cfg.get("sources")
    if sources:
        first = sources[0]
        bus = first.get("bus", cfg["system"]["source_bus"])
        basekv = first.get("basekv", cfg["system"]["base_kV_HV"])
        return bus, basekv, first
    return cfg["system"]["source_bus"], cfg["system"]["base_kV_HV"], cfg["source"]


def _truthy(val) -> bool:
    """Interpreta yes/true/1/on como verdadeiro; NaN/None/vazio como falso."""
    if val is None:
        return False
    if isinstance(val, float) and val != val:  # NaN
        return False
    if isinstance(val, str):
        return val.strip().lower() in ("yes", "y", "true", "t", "1", "on", "closed")
    return bool(val)


def _extra_voltage_bases(system: SystemData, existing: list[float]) -> list[float]:
    """Bases de tensao (kV LL) presentes no sistema e ausentes de `existing`.

    Fontes: buses.csv (base_kV), transformadores (kV_HV, kV_LV) e
    `sources:` (basekv). Ordem decrescente. Na planta 1 o resultado e vazio,
    mantendo o master.dss identico ao original.
    """
    cand: list[float] = []
    if "base_kV" in system.buses.columns:
        cand += [float(v) for v in system.buses["base_kV"].dropna()]
    for tr in system.config.get("transformers", []) or []:
        for key in ("kV_HV", "kV_LV"):
            if key in tr:
                cand.append(float(tr[key]))
    for s in system.config.get("sources", []) or []:
        if "basekv" in s:
            cand.append(float(s["basekv"]))
    extra: list[float] = []
    for v in sorted(set(cand), reverse=True):
        if all(abs(v - float(e)) > 1e-9 for e in list(existing) + extra):
            extra.append(v)
    return extra


def _write_master(system: SystemData, path: Path) -> None:
    cfg = system.config
    src_bus, base_kv_hv, src = _primary_source(cfg)
    base_kv_mv = cfg["system"]["base_kV_MV"]
    freq = cfg["system"]["frequency_Hz"]
    # pu/angle: padrao original "1.00"/"0"; so mudam se `sources:` declarar.
    pu_txt = src.get("pu", "1.00") if cfg.get("sources") else "1.00"
    ang_txt = src.get("angle", 0) if cfg.get("sources") else 0

    content = HEADER
    content += "Clear\n"
    content += f"Set DefaultBaseFrequency={freq}\n"
    content += (
        f"New Circuit.{cfg['system']['name']} "
        f"bus1={src_bus}.1.2.3 phases=3 "
        f"basekv={base_kv_hv} BaseMVA={cfg['system']['base_MVA']} "
        f"pu={pu_txt} angle={ang_txt} frequency={freq} "
        f"isc3={src['isc3_kA'] * 1000.0:.2f} isc1={src['isc1_kA'] * 1000.0:.2f} "
        f"x1r1={src['x1r1']} x0r0={src['x0r0']}\n"
    )
    content += f"\nRedirect {Path('generated_network.dss').name}\n"
    content += "\n! Bases de tensao por barra\n"
    bases = [base_kv_hv, base_kv_mv]
    extra = _extra_voltage_bases(system, [float(b) for b in bases])
    content += f"Set Voltagebases=[{', '.join(str(b) for b in bases + extra)}]\n"
    content += "Calcvoltagebases\n"
    content += "\nSet Mode=snap\n"
    content += "Set Maxiterations=100\n"
    content += "Set Maxcontroliter=50\n"
    content += "Set tolerance=1e-6\n"
    path.write_text(content, encoding="utf-8")


def _get_transformers_out(cfg: dict) -> set[str]:
    """Retorna o conjunto de nomes de trafos em contingencia (fora de servico)."""
    ct = cfg.get("contingency", {})
    if not ct or ct.get("mode", "N-0") == "N-0":
        return set()
    return set(ct.get("transformers_out", []))


def _write_network(system: SystemData, path: Path) -> None:
    cfg = system.config
    mv_main = cfg["system"]["mv_main_bus"]
    src = cfg["system"]["source_bus"]

    # Lista de trafos (compat: aceita 'transformer' singular legado OU 'transformers' lista)
    if "transformers" in cfg:
        trafos = cfg["transformers"]
    elif "transformer" in cfg:
        trafos = [cfg["transformer"]]
    else:
        raise KeyError("system_config.yaml: defina 'transformers' (lista) ou 'transformer' (objeto)")

    out_set = _get_transformers_out(cfg)
    mode = cfg.get("contingency", {}).get("mode", "N-0")

    lines_out = [HEADER, f"! Modo de contingencia: {mode}"]
    if out_set:
        lines_out.append(f"! Transformadores FORA de servico: {sorted(out_set)}")

    # ---------- Fontes adicionais (aditivo 2026-11-14; so se `sources:` tiver >1) ----------
    extra_sources = (cfg.get("sources") or [])[1:]
    if extra_sources:
        ct = cfg.get("contingency", {}) or {}
        src_out = set(ct.get("sources_out", []) or []) if mode != "N-0" else set()
        freq = cfg["system"]["frequency_Hz"]
        lines_out.append("\n! --- Fontes adicionais (Vsource) ---")
        for s in extra_sources:
            en = "no" if s["name"] in src_out else "yes"
            lines_out.append(
                f"New Vsource.{s['name']} bus1={s['bus']}.1.2.3 phases=3 "
                f"basekv={s['basekv']} pu={s.get('pu', 1.0)} angle={s.get('angle', 0)} "
                f"frequency={freq} "
                f"isc3={s['isc3_kA'] * 1000.0:.2f} isc1={s['isc1_kA'] * 1000.0:.2f} "
                f"x1r1={s['x1r1']} x0r0={s['x0r0']} enabled={en}"
            )

    # ---------- Transformadores em paralelo ----------
    lines_out.append("\n! --- Transformadores ---")
    for tr in trafos:
        in_service = tr["name"] not in out_set
        enabled = "yes" if in_service else "no"
        flag = " [IN-SERVICE]" if in_service else " [OUT-OF-SERVICE]"
        # bus_hv / bus_lv opcionais (aditivo 2026-11-14); padrao = original.
        bus_hv = tr.get("bus_hv", src)
        bus_lv = tr.get("bus_lv", mv_main)
        lines_out.append(f"! {tr['name']}{flag}")
        # IMPORTANTE: continuation '~' deve estar no INICIO de nova linha fisica
        lines_out.append(
            f"New Transformer.{tr['name']} phases=3 windings=2 xhl={tr['xhl_percent']} "
            f"%loadloss={tr['loss_load_percent']} %noloadloss={tr['loss_no_load_percent']} "
            f"enabled={enabled}"
        )
        lines_out.append(
            f"~ wdg=1 bus={bus_hv}.1.2.3 conn={tr['conn_HV']} kV={tr['kV_HV']} kVA={tr['kva']}"
        )
        lines_out.append(
            f"~ wdg=2 bus={bus_lv}.1.2.3 conn={tr['conn_LV']} kV={tr['kV_LV']} kVA={tr['kva']}"
        )

    # ---------- Linhas / cabos (se houver) ----------
    if not system.lines.empty:
        lines_out.append("\n! --- Linhas / cabos ---")
        has_switch = "switch" in system.lines.columns
        has_enabled = "enabled" in system.lines.columns
        for _, row in system.lines.iterrows():
            # Coluna opcional `enabled` (aditivo): falso => enabled=no.
            en_sfx = "" if (not has_enabled or _truthy(row["enabled"])) else " enabled=no"
            # Coluna opcional `switch` (aditivo): verdadeiro => Line switch=yes
            # (impedancia minima do OpenDSS; sem linecode).
            if has_switch and _truthy(row["switch"]):
                # normamps (correcao de 2026-11-14): sem ele o OpenDSS assume
                # 400 A e o carregamento da interligacao entra em f3 de forma
                # artificial (P2: TIE_AB a 244,7 %). So quando ampacity_A
                # existe e esta preenchido; a P1 nao tem linhas e nao muda.
                amp = row["ampacity_A"] if "ampacity_A" in system.lines.columns else None
                amp_ok = amp is not None and str(amp).strip().lower() not in ("", "nan", "none")
                amp_sfx = f" normamps={amp}" if amp_ok else ""
                lines_out.append(
                    f"New Line.{row['name']} phases=3 "
                    f"bus1={row['bus_from']}.1.2.3 bus2={row['bus_to']}.1.2.3 "
                    f"switch=yes{amp_sfx}{en_sfx}"
                )
                continue
            lc_name = f"LC_{row['name']}"
            r1 = float(row["r1_ohm_km"]) / 1000.0
            x1 = float(row["x1_ohm_km"]) / 1000.0
            c1 = float(row["c1_nf_km"])
            c1_uF_per_m = c1 * 1e-6
            r0 = r1 * 3.0
            x0 = x1 * 3.5
            c0_uF_per_m = c1_uF_per_m * 0.5
            lines_out.append(
                f"New Linecode.{lc_name} nphases=3 r1={r1:.6f} x1={x1:.6f} c1={c1_uF_per_m:.6e} "
                f"r0={r0:.6f} x0={x0:.6f} c0={c0_uF_per_m:.6e} units=m normamps={row['ampacity_A']}"
            )
            lines_out.append(
                f"New Line.{row['name']} phases=3 "
                f"bus1={row['bus_from']}.1.2.3 bus2={row['bus_to']}.1.2.3 "
                f"linecode={lc_name} length={row['length_m']} units=m{en_sfx}"
            )

    # ---------- Cargas estaticas ----------
    lines_out.append("\n! --- Cargas estaticas ---")
    for _, row in system.loads.iterrows():
        enabled = "yes" if row["initial_state"] == "on" else "no"
        lines_out.append(
            f"New Load.{row['name']} Bus1={row['bus']}.1.2.3 Phases={row['phases']} Conn=wye "
            f"kV={row['kV']} kW={row['kW']} kvar={row['kvar']} Model=1 "
            f"Vminpu=0.85 Vmaxpu=1.20 enabled={enabled} "
            f"!critical={row['critical']} sheddable={row['sheddable']} load_id={row['load_id']}"
        )

    # ---------- Motores (regime + inrush) ----------
    lines_out.append("\n! --- Motores (modelo simplificado em regime + inrush em standby) ---")
    for _, row in system.motors.iterrows():
        elec = compute_motor_electrical(row, in_inrush=False)
        lines_out.append(motor_as_dss_load(elec, state=row["initial_state"]))
        lines_out.append(motor_as_dss_load_inrush(elec))

    # ---------- Geradores locais (aditivo 2026-11-14; so se `generators:`) ----------
    gens = cfg.get("generators") or []
    if gens:
        ct = cfg.get("contingency", {}) or {}
        gen_out = set(ct.get("generators_out", []) or []) if mode != "N-0" else set()
        lines_out.append("\n! --- Geradores ---")
        skip = {"name", "bus", "phases", "description", "enabled"}
        for g in gens:
            en = "no" if (g["name"] in gen_out or
                          ("enabled" in g and not _truthy(g["enabled"]))) else "yes"
            # Demais chaves do yaml viram parametros do elemento, na ordem do yaml
            # (ex.: kV, kW, kvar, model, Vpu, maxkvar, minkvar, kVA, xdp, xdpp, H).
            params = " ".join(f"{k}={v}" for k, v in g.items() if k not in skip)
            lines_out.append(
                f"New Generator.{g['name']} bus1={g['bus']}.1.2.3 "
                f"phases={g.get('phases', 3)} {params} enabled={en}"
            )

    # ---------- Medidores ----------
    lines_out.append("\n! --- Medidores ---")
    # Energymeter no primeiro trafo IN-SERVICE
    em_target = None
    for tr in trafos:
        if tr["name"] not in out_set:
            em_target = tr["name"]; break
    if em_target:
        lines_out.append(f"New Energymeter.EM_MAIN element=Transformer.{em_target} terminal=1")

    path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")


def patch_motor_state(network_dss_path: Path, motor_name: str, enabled: bool) -> None:
    """Habilita/desabilita um motor diretamente no arquivo .dss (uso opcional)."""
    txt = network_dss_path.read_text(encoding="utf-8")
    target = f"Load.{motor_name} "
    new_lines = []
    for line in txt.splitlines():
        if line.startswith("New " + target) and "enabled=" in line:
            new_state = "yes" if enabled else "no"
            line = _replace_kv(line, "enabled", new_state)
        new_lines.append(line)
    network_dss_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def _replace_kv(line: str, key: str, value: str) -> str:
    pattern = re.compile(rf"({key}=)\S+", re.IGNORECASE)
    return pattern.sub(rf"\1{value}", line)
