"""Logica de protecao e contagem de violacoes de tensao/carregamento."""

from __future__ import annotations

from dataclasses import dataclass

from .config_loader import SystemData
from .powerflow_runner import PowerFlowSnapshot


@dataclass
class ProtectionViolations:
    overvoltage_buses: list[str]
    undervoltage_buses: list[str]
    overloaded_lines: list[str]
    cascade_count: int
    critical_violation: bool
    summary: str


def evaluate_protection(
    system: SystemData,
    snapshot: PowerFlowSnapshot,
    use_inrush_limits: bool = False,
) -> ProtectionViolations:
    """Avalia violacoes em um snapshot.

    Parameters
    ----------
    use_inrush_limits : bool
        Se True, aplica os limites mais frouxos (admitidos durante partida e
        manobra de abertura) em vez dos limites operacionais nominais.
    """
    lim = system.config["operational_limits"]
    if use_inrush_limits:
        v_min_lim = float(lim["v_min_inrush_pu"])
        v_max_lim = float(lim["v_max_switching_pu"])
    else:
        v_min_lim = float(lim["v_min_pu"])
        v_max_lim = float(lim["v_max_pu"])

    v_pickup_over = float(lim["protection_overvoltage_pu"])
    v_pickup_under = float(lim["protection_undervoltage_pu"])
    therm_max = float(lim["thermal_loading_max_percent"])

    over: list[str] = []
    under: list[str] = []
    critical = False

    for bus, vmax in snapshot.bus_voltages_pu_max.items():
        if vmax > v_max_lim:
            over.append(bus)
            if vmax >= v_pickup_over:
                critical = True

    for bus, vmin in snapshot.bus_voltages_pu_min.items():
        if vmin < v_min_lim:
            under.append(bus)
            if vmin <= v_pickup_under:
                critical = True

    overloaded = [ln for ln, ld in snapshot.line_loading_pct.items() if ld > therm_max]

    cascade = len(over) + len(under) + len(overloaded)
    summary = (
        f"OV={len(over)} | UV={len(under)} | Overload={len(overloaded)} | "
        f"crit={critical} | snapshot={snapshot.label}"
    )

    return ProtectionViolations(
        overvoltage_buses=over,
        undervoltage_buses=under,
        overloaded_lines=overloaded,
        cascade_count=cascade,
        critical_violation=critical,
        summary=summary,
    )


def total_idle_motor_kW(system: SystemData, motors_on: dict[int, bool],
                         target_motor_id: int) -> float:
    """Soma a potencia mecanica dos motores em off (excluindo o motor alvo)."""
    total = 0.0
    for _, row in system.motors.iterrows():
        mid = int(row["motor_id"])
        if mid == target_motor_id:
            continue
        if not motors_on.get(mid, True):
            total += float(row["kW"])
    return total
