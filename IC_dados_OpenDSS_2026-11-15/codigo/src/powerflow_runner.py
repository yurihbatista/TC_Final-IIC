"""Executor de fluxo de potencia e avaliacao de cenarios de manobra.

Reescopo (2026-05): foco em manobra (energizacao E desligamento) do motor
alvo. Para cada combinacao de estado on/off das demais maquinas avaliamos
quatro snapshots em quase-estatico:
  1. PRE-ENERG  : motor alvo OFF, demais conforme cenario
  2. INRUSH     : motor alvo em partida (locked-rotor)
  3. PRE-DESL   : motor alvo ON nominal
  4. POS-DESL   : motor alvo OFF apos abertura
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config_loader import SystemData
from .opendss_interface import OpenDSSInterface

log = logging.getLogger(__name__)


@dataclass
class ScenarioState:
    motors_on: dict[int, bool] = field(default_factory=dict)
    target_motor_id: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PowerFlowSnapshot:
    converged: bool
    iterations: int
    bus_voltages_pu: dict[str, float]
    bus_voltages_pu_min: dict[str, float]
    bus_voltages_pu_max: dict[str, float]
    line_loading_pct: dict[str, float]
    total_power_kW: float
    total_reactive_kvar: float
    total_losses_kW: float
    motor_currents_A: dict[str, float]
    label: str = ""
    transformer_SkVA: dict[str, float] = field(default_factory=dict)

    @property
    def v_max(self) -> float:
        return max(self.bus_voltages_pu_max.values()) if self.bus_voltages_pu_max else 1.0

    @property
    def v_min(self) -> float:
        return min(self.bus_voltages_pu_min.values()) if self.bus_voltages_pu_min else 1.0


def _set_motor_state(dss, motor_name: str, running: bool, in_inrush: bool = False) -> None:
    if running and not in_inrush:
        dss.set_motor_running(motor_name, running=True)
    elif in_inrush:
        dss.set_motor_inrush(motor_name, in_inrush=True)
    else:
        dss.set_load_enabled(motor_name, False)
        dss.set_load_enabled(f"{motor_name}_INRUSH", False)


def _apply_other_motors(dss, system: SystemData, scen: ScenarioState) -> None:
    target_id = scen.target_motor_id
    for _, row in system.motors.iterrows():
        mid = int(row["motor_id"])
        if mid == target_id:
            continue
        running = scen.motors_on.get(mid, True)
        _set_motor_state(dss, str(row["name"]), running=running, in_inrush=False)


def collect_snapshot(dss, system: SystemData, label: str = "") -> PowerFlowSnapshot:
    converged = dss.solve()
    p, q = dss.get_total_power_kW()
    motor_currents = {}
    for _, row in system.motors.iterrows():
        try:
            currents = dss.get_element_currents(f"Load.{row['name']}")
            motor_currents[str(row["name"])] = float(np.max(currents))
        except Exception:
            motor_currents[str(row["name"])] = 0.0

    # Carregamento dos transformadores em servico (para restricao ANSI 49).
    tr_SkVA: dict[str, float] = {}
    ct = system.config.get("contingency", {})
    out_set = set(ct.get("transformers_out", []) or [])
    for tr in system.config.get("transformers", []):
        nm = str(tr["name"])
        if nm not in out_set and hasattr(dss, "get_transformer_SkVA"):
            tr_SkVA[nm] = dss.get_transformer_SkVA(nm)
    return PowerFlowSnapshot(
        transformer_SkVA=tr_SkVA,
        converged=converged,
        iterations=int(dss.dss.solution.iterations),
        bus_voltages_pu=dss.get_bus_voltages_pu(),
        bus_voltages_pu_min=dss.get_bus_voltages_pu_min(),
        bus_voltages_pu_max=dss.get_bus_voltages_pu_max(),
        line_loading_pct=dss.get_line_loading_percent(),
        total_power_kW=p,
        total_reactive_kvar=q,
        total_losses_kW=dss.get_total_losses_kW(),
        motor_currents_A=motor_currents,
        label=label,
    )


@dataclass
class ManeuverSnapshots:
    pre_energ: PowerFlowSnapshot
    inrush: PowerFlowSnapshot
    pre_desl: PowerFlowSnapshot
    post_desl: PowerFlowSnapshot


def run_maneuver(dss, system: SystemData, scen: ScenarioState) -> ManeuverSnapshots:
    target_id = scen.target_motor_id
    target_row = system.motor_by_id(target_id)
    target_name = str(target_row["name"])
    _apply_other_motors(dss, system, scen)
    _set_motor_state(dss, target_name, running=False, in_inrush=False)
    pre_energ = collect_snapshot(dss, system, label="pre_energ")
    _set_motor_state(dss, target_name, running=False, in_inrush=True)
    inrush = collect_snapshot(dss, system, label="inrush")
    _set_motor_state(dss, target_name, running=True, in_inrush=False)
    pre_desl = collect_snapshot(dss, system, label="pre_desl")
    _set_motor_state(dss, target_name, running=False, in_inrush=False)
    post_desl = collect_snapshot(dss, system, label="post_desl")
    return ManeuverSnapshots(pre_energ=pre_energ, inrush=inrush,
                              pre_desl=pre_desl, post_desl=post_desl)


def base_case_snapshot(master_path, system: SystemData) -> PowerFlowSnapshot:
    """Fluxo base: motor alvo ON, todas as outras maquinas ON."""
    dss = OpenDSSInterface()
    dss.compile(master_path)
    target_id = int(system.config["target_motor"]["motor_id"])
    scen = ScenarioState(
        motors_on={int(r["motor_id"]): True for _, r in system.motors.iterrows()
                   if int(r["motor_id"]) != target_id},
        target_motor_id=target_id,
    )
    _apply_other_motors(dss, system, scen)
    target_name = str(system.motor_by_id(target_id)["name"])
    _set_motor_state(dss, target_name, running=True, in_inrush=False)
    return collect_snapshot(dss, system, label="base")
