"""Camada de abstracao sobre py-dss-interface.

Justificativa: py-dss-interface e cross-platform, mantida ativamente e oferece
API pythonica proxima ao COM oficial do EPRI. Concentra todas as chamadas DSS
neste modulo para isolar o codigo de aplicacao de mudancas de API futuras.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class OpenDSSInterface:
    """Wrapper estavel sobre o engine OpenDSS."""

    def __init__(self) -> None:
        try:
            import py_dss_interface  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "py-dss-interface nao esta instalado. Execute: pip install py-dss-interface"
            ) from e
        from py_dss_interface import DSS

        self.dss = DSS()
        self._compiled_path: Path | None = None

    # -------------------------------------------------------------- compile
    def compile(self, master_path: str | Path) -> None:
        # Forca path absoluto INDEPENDENTE do cwd corrente, pois
        # py-dss-interface altera o cwd ao instanciar o engine.
        master_path = Path(os.path.abspath(os.path.expanduser(str(master_path))))
        if not master_path.is_file():
            raise FileNotFoundError(f"master.dss nao encontrado: {master_path}")
        self.dss.text(f'compile "{master_path}"')
        self._compiled_path = master_path
        log.info("OpenDSS compilado a partir de %s", master_path)

    def command(self, cmd: str) -> None:
        self.dss.text(cmd)

    # ----------------------------------------------------------------- solve
    def solve(self) -> bool:
        self.dss.text("Solve")
        converged = bool(self.dss.solution.converged)
        if not converged:
            log.warning("Fluxo de potencia NAO convergiu.")
        return converged

    # ------------------------------------------------------------- toggles
    def set_load_enabled(self, load_name: str, enabled: bool) -> None:
        state = "yes" if enabled else "no"
        self.dss.text(f"Edit Load.{load_name} enabled={state}")

    def set_motor_running(self, motor_name: str, running: bool) -> None:
        self.set_load_enabled(motor_name, running)
        self.set_load_enabled(f"{motor_name}_INRUSH", False)

    def set_motor_inrush(self, motor_name: str, in_inrush: bool) -> None:
        self.set_load_enabled(motor_name, not in_inrush)
        self.set_load_enabled(f"{motor_name}_INRUSH", in_inrush)

    def scale_load(self, multiplier: float) -> None:
        self.dss.text(f"Set LoadMult={multiplier:.6f}")

    # ----------------------------------------------------- voltage extraction
    def get_bus_voltages_pu(self) -> dict[str, float]:
        """Retorna magnitude media por barra em pu (media das tres fases)."""
        out: dict[str, float] = {}
        for name in self.dss.circuit.buses_names:
            self.dss.circuit.set_active_bus(name)
            vmag = np.array(self.dss.bus.vmag_angle)
            if vmag.size == 0:
                continue
            mags = vmag[::2]
            base_kv = self.dss.bus.kv_base
            if base_kv <= 0:
                continue
            # py-dss-interface ja retorna kv_base em LINHA-NEUTRO (kV LN).
            # Para 4,16 kV LL, kv_base = 4.16/sqrt(3) ~ 2.40 kV.
            v_ln_base = base_kv * 1000.0
            v_pu_phase = mags / v_ln_base
            out[name] = float(np.mean(v_pu_phase))
        return out

    def get_bus_voltages_pu_min(self) -> dict[str, float]:
        """Retorna a MINIMA entre as tres fases por barra (em pu).
        Usado como criterio de sag (afundamento) durante energizacao.
        """
        out: dict[str, float] = {}
        for name in self.dss.circuit.buses_names:
            self.dss.circuit.set_active_bus(name)
            vmag = np.array(self.dss.bus.vmag_angle)
            if vmag.size == 0:
                continue
            mags = vmag[::2]
            base_kv = self.dss.bus.kv_base
            if base_kv <= 0:
                continue
            v_ln_base = base_kv * 1000.0
            v_pu_phase = mags / v_ln_base
            out[name] = float(np.min(v_pu_phase))
        return out

    def get_bus_voltages_pu_max(self) -> dict[str, float]:
        """Retorna a maxima entre as tres fases por barra (em pu)."""
        out: dict[str, float] = {}
        for name in self.dss.circuit.buses_names:
            self.dss.circuit.set_active_bus(name)
            vmag = np.array(self.dss.bus.vmag_angle)
            if vmag.size == 0:
                continue
            mags = vmag[::2]
            base_kv = self.dss.bus.kv_base
            if base_kv <= 0:
                continue
            # py-dss-interface ja retorna kv_base em LINHA-NEUTRO (kV LN).
            # Para 4,16 kV LL, kv_base = 4.16/sqrt(3) ~ 2.40 kV.
            v_ln_base = base_kv * 1000.0
            v_pu_phase = mags / v_ln_base
            out[name] = float(np.max(v_pu_phase))
        return out

    # ------------------------------------------------------- power & losses
    def get_total_losses_kW(self) -> float:
        losses = self.dss.circuit.losses
        return float(losses[0]) / 1000.0

    def get_total_power_kW(self) -> tuple[float, float]:
        p, q = self.dss.circuit.total_power
        return float(-p), float(-q)

    def get_line_loading_percent(self) -> dict[str, float]:
        """Retorna o carregamento percentual de cada linha.

        Robusto a circuitos sem linhas: py-dss-interface retorna
        ['None'] quando count==0, o que provoca SIGSEGV ao ativar o
        elemento. Evita-se isto checando 'count' primeiro.
        """
        out: dict[str, float] = {}
        try:
            n_lines = int(self.dss.lines.count)
        except Exception:
            n_lines = 0
        if n_lines <= 0:
            return out
        for line in self.dss.lines.names:
            if not line or line.lower() == "none":
                continue
            self.dss.lines.name = line
            self.dss.circuit.set_active_element(f"Line.{line}")
            currents = self.dss.cktelement.currents_mag_ang
            mags = np.array(currents[::2])
            i_max = float(np.max(mags)) if mags.size else 0.0
            i_norm = float(self.dss.lines.norm_amps) if self.dss.lines.norm_amps else 0.0
            loading = (i_max / i_norm * 100.0) if i_norm > 0 else 0.0
            out[line] = loading
        return out

    def get_transformer_SkVA(self, name: str) -> float:
        """Potencia aparente (kVA) no terminal 1 do transformador `name`.

        Soma P e Q dos condutores do terminal primario e retorna |S|.
        Retorna 0.0 se o elemento nao existir ou estiver desabilitado.
        """
        try:
            self.dss.circuit.set_active_element(f"Transformer.{name}")
            powers = np.array(self.dss.cktelement.powers, dtype=float)
            if powers.size == 0:
                return 0.0
            n_cond = powers.size // 2  # metade = terminal 1 (P1,Q1,P2,Q2,...)
            p = powers[0:n_cond:2].sum()
            q = powers[1:n_cond:2].sum()
            return float(np.hypot(p, q))
        except Exception:
            return 0.0

    def get_element_currents(self, full_name: str) -> np.ndarray:
        self.dss.circuit.set_active_element(full_name)
        currents = np.array(self.dss.cktelement.currents_mag_ang)
        if currents.size == 0:
            return np.zeros(3)
        return currents[::2]

    # --------------------------------------------------------------- summary
    def summary(self) -> dict[str, Any]:
        return {
            "converged": bool(self.dss.solution.converged),
            "iterations": int(self.dss.solution.iterations),
            "total_power_kW": self.get_total_power_kW(),
            "total_losses_kW": self.get_total_losses_kW(),
            "bus_voltages_pu_mean": self.get_bus_voltages_pu(),
            "bus_voltages_pu_max": self.get_bus_voltages_pu_max(),
            "line_loading_percent": self.get_line_loading_percent(),
        }
