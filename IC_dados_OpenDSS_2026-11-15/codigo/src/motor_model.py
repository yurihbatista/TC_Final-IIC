"""Modelagem de motores de indução em regime permanente / quase-estático.

Justificativa da escolha:
  O OpenDSS dispõe do elemento *Load* com modelos 1 (P, Q constantes),
  2 (impedância constante) e 5 (corrente constante) — adequados para
  representar motores de indução em regime permanente. Para a análise
  de manobras de VCB em frequência industrial, este modelo simplificado
  é suficiente porque o objetivo é avaliar fluxo de potência, sobre-
  tensão pós-abertura e violação de limites — não o transitório de
  reignição em si, que demanda EMTP (vide export_to_emtp.py).

  Modelo avançado (UPFC, IndMach012) existe no OpenDSS para estudos
  dinâmicos, mas o agente de decisão Silva (2026) opera em snapshot
  quase-estático conforme metodologia da dissertação.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class MotorElectrical:
    """Resultado do cálculo elétrico do motor."""

    motor_id: int
    name: str
    bus: str
    kV: float
    kW_in: float            # Potência ativa drenada da rede
    kvar_in: float          # Potência reativa drenada da rede
    kVA_in: float           # Potência aparente
    pf: float
    efficiency: float
    inrush_multiplier: float
    nominal_current_A: float
    inrush_current_A: float
    starting_pf: float


def compute_motor_electrical(motor_row, in_inrush: bool = False) -> MotorElectrical:
    """Calcula grandezas elétricas do motor a partir da linha de motors.csv.

    Parameters
    ----------
    motor_row : pd.Series
        Linha do dataframe motors.csv.
    in_inrush : bool
        Se True, retorna corrente de partida (locked-rotor) ao invés da nominal.
    """
    kW_mech = float(motor_row["kW"])
    kV = float(motor_row["kV"])
    pf = float(motor_row["pf"])
    eff = float(motor_row["efficiency"])
    inrush_mult = float(motor_row["inrush_multiplier"])
    starting_pf = float(motor_row.get("starting_pf", 0.25))

    # Potência elétrica de entrada = mecânica / rendimento
    kW_in = kW_mech / eff
    kVA_in = kW_in / pf
    kvar_in = math.sqrt(max(kVA_in**2 - kW_in**2, 0.0))

    # Corrente nominal (trifásica): I = kVA / (sqrt(3) * kV)
    inom = (kVA_in * 1e3) / (math.sqrt(3) * kV * 1e3)
    iinr = inom * inrush_mult

    return MotorElectrical(
        motor_id=int(motor_row["motor_id"]),
        name=str(motor_row["name"]),
        bus=str(motor_row["bus"]),
        kV=kV,
        kW_in=kW_in,
        kvar_in=kvar_in,
        kVA_in=kVA_in,
        pf=pf,
        efficiency=eff,
        inrush_multiplier=inrush_mult,
        nominal_current_A=inom,
        inrush_current_A=iinr,
        starting_pf=starting_pf,
    )


def motor_as_dss_load(elec: MotorElectrical, state: str = "on") -> str:
    """Gera string DSS representando o motor como carga (model=1).

    Para a partida direta em quase-estático, usa-se uma carga adicional
    com corrente equivalente a inrush_multiplier × Inom durante o cenário
    de partida (gerado por dss_builder em modo separado).
    """
    enabled = "yes" if state == "on" else "no"
    return (
        f"New Load.{elec.name} Bus1={elec.bus}.1.2.3 Phases=3 Conn=wye "
        f"kV={elec.kV:.4f} kW={elec.kW_in:.3f} kvar={elec.kvar_in:.3f} "
        f"Model=1 Vminpu=0.6 Vmaxpu=1.2 enabled={enabled} "
        f"!motor_id={elec.motor_id}"
    )


def motor_as_dss_load_inrush(elec: MotorElectrical) -> str:
    """Representa motor em partida (locked-rotor) com fator de potência baixo.

    Aproximação: durante locked-rotor a corrente é I_lr ≈ inrush_mult × Inom
    com fator de potência tipicamente 0,2–0,35. Modela-se como carga modelo 2
    (impedância constante) com cosφ = starting_pf para emular o afundamento.
    """
    # Potência aparente equivalente em LR
    kVA_lr = elec.kVA_in * elec.inrush_multiplier
    kW_lr = kVA_lr * elec.starting_pf
    kvar_lr = math.sqrt(max(kVA_lr**2 - kW_lr**2, 0.0))
    return (
        f"New Load.{elec.name}_INRUSH Bus1={elec.bus}.1.2.3 Phases=3 Conn=wye "
        f"kV={elec.kV:.4f} kW={kW_lr:.3f} kvar={kvar_lr:.3f} "
        f"Model=2 Vminpu=0.4 Vmaxpu=1.2 enabled=no "
        f"!motor_id={elec.motor_id}_inrush"
    )
