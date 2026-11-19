"""Carregador de configuração e arquivos CSV de descrição do sistema.

Centraliza a leitura de:
  - system_config.yaml (parâmetros globais)
  - buses.csv, lines.csv, motors.csv, loads.csv (topologia)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass
class SystemData:
    """Container imutável das tabelas e configuração do sistema."""

    config: dict[str, Any]
    buses: pd.DataFrame
    lines: pd.DataFrame
    motors: pd.DataFrame
    loads: pd.DataFrame
    data_dir: Path = field(default_factory=Path)

    @property
    def base_kV_MV(self) -> float:
        return float(self.config["system"]["base_kV_MV"])

    @property
    def frequency_Hz(self) -> int:
        return int(self.config["system"]["frequency_Hz"])

    def motor_by_id(self, motor_id: int) -> pd.Series:
        df = self.motors.loc[self.motors["motor_id"] == int(motor_id)]
        if df.empty:
            raise KeyError(f"motor_id={motor_id} inexistente em motors.csv")
        return df.iloc[0]


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def deep_merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """Merge profundo (aditivo, 2026-11-14): devolve NOVO dict.

    Chaves de `override` substituem as de `base`; dicts aninhados sao
    mesclados recursivamente; listas e escalares sao substituidos inteiros.
    `base` e `override` nao sao modificados.
    """
    import copy

    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_system(data_dir: str | Path,
                overrides: dict[str, Any] | None = None) -> SystemData:
    """Carrega o sistema completo a partir do diretório /data.

    `overrides` (opcional, aditivo em 2026-11-14): merge profundo aplicado
    sobre o system_config.yaml lido. Sem overrides, comportamento original.
    """
    data_dir = Path(data_dir)
    config = load_config(data_dir / "system_config.yaml")
    if overrides:
        config = deep_merge(config, overrides)

    buses = pd.read_csv(data_dir / "buses.csv")
    lines = pd.read_csv(data_dir / "lines.csv")
    motors = pd.read_csv(data_dir / "motors.csv")
    loads = pd.read_csv(data_dir / "loads.csv")

    _validate(buses, lines, motors, loads)
    return SystemData(
        config=config,
        buses=buses,
        lines=lines,
        motors=motors,
        loads=loads,
        data_dir=data_dir,
    )


def _validate(buses: pd.DataFrame, lines: pd.DataFrame, motors: pd.DataFrame, loads: pd.DataFrame) -> None:
    bus_names = set(buses["name"].astype(str))

    for _, row in lines.iterrows():
        if row["bus_from"] not in bus_names or row["bus_to"] not in bus_names:
            raise ValueError(f"Linha {row['name']} referencia barra inexistente.")

    for _, row in motors.iterrows():
        if row["bus"] not in bus_names:
            raise ValueError(f"Motor {row['name']} na barra inexistente {row['bus']}.")

    for _, row in loads.iterrows():
        if row["bus"] not in bus_names:
            raise ValueError(f"Carga {row['name']} na barra inexistente {row['bus']}.")

    # Garante presenca de exatamente um motor alvo (is_target=true)
    if "is_target" in motors.columns:
        n_target = int(motors["is_target"].sum())
        if n_target != 1:
            raise ValueError(
                f"motors.csv deve ter exatamente um motor com is_target=true "
                f"(encontrados: {n_target})."
            )
