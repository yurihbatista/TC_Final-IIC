"""Fixtures compartilhadas dos testes de correspondencia artigo<->codigo."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config_loader import load_system

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def pytest_configure(config):
    # 2026-11-14: registra o marcador usado em test_regression_sepoc.py e
    # test_plant2.py (antes gerava PytestUnknownMarkWarning). Para pular os
    # lentos: pytest -m "not slow".
    config.addinivalue_line("markers", "slow: testes lentos (OpenDSS/NSGA completos)")


@pytest.fixture(scope="session")
def system():
    return load_system(DATA_DIR)


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    """Compila o circuito OpenDSS uma unica vez por sessao."""
    pytest.importorskip("py_dss_interface")
    from src.dss_builder import build as build_dss
    sysd = load_system(DATA_DIR)
    out = tmp_path_factory.mktemp("dss")
    master, _ = build_dss(sysd, out)
    return sysd, master
