"""Testes da base OpenDSS para IC (ESPEC secao 5): features, rotulo, Edit x
recompilacao, Scc, retomada por blocos e curva de aprendizagem."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ic import FEATURES, dataset as D, learning_curve as LC

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ic" / "config.yaml"


@pytest.fixture()
def cfg(tmp_path):
    c = D.load_config(CONFIG)
    c["run"]["work_dir"] = str(tmp_path / "work")
    return c


# ----------------------------------------------------------------- planos
def test_plan_deterministic_and_in_ranges(cfg):
    a = D.plan_random(cfg, "stratified_k", 200, 5)
    b = D.plan_random(cfg, "stratified_k", 200, 5)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == D.SPEC_KEYS
    for col, key in (("Pn_kW", "Pn_kW_range"), ("Ip_In", "Ip_In_range"),
                     ("cosphi_p", "cosphi_p_range"), ("eta", "eta_range")):
        lo, hi = cfg["target"][key]
        assert a[col].between(lo, hi).all()
    lo, hi = cfg["source"]["isc3_kA_range"]
    assert a["isc3_kA"].between(lo, hi).all()
    # LHS na dimensao da contingencia: fracao exata p_N1.
    assert (a["contingency"] == "N-1").sum() == 100
    k = a["bits"].str.count("1")
    assert k.min() == 0 and k.max() == 19  # stratified_k cobre toda a faixa
    assert a["bits"].str.len().eq(19).all()


def test_bernoulli_and_index_bits(cfg):
    p = D.plan_random(cfg, "bernoulli", 400, 9)
    frac = p["bits"].str.count("1").sum() / (400 * 19)
    assert abs(frac - 0.5) < 0.02
    assert D.bits_from_index(5, 4) == "0101"
    allb = {D.bits_from_index(i, 5) for i in range(32)}
    assert len(allb) == 32


def test_label_rule():
    lab = {"v_min_gt": 0.85, "v_max_post_le": 1.08, "loading_basis": "nominal",
           "loading_max": 1.0, "require_converged": True}
    ok = {"v_min_label": 0.86, "v_max_post": 1.0, "tr_load_nom": 0.9, "tr_load_af": 0.75,
          "converged": True}
    assert D.label_value(ok, lab) == 1
    assert D.label_value({**ok, "v_min_label": 0.85}, lab) == 0      # estrito
    assert D.label_value({**ok, "v_max_post": 1.081}, lab) == 0      # sobretensao
    assert D.label_value({**ok, "tr_load_nom": 1.1}, lab) == 0       # carregamento AN
    assert D.label_value({**ok, "tr_load_nom": 1.1}, {**lab, "loading_basis": "af"}) == 1
    assert D.label_value({**ok, "converged": False}, lab) == 0
    assert D.label_value({**ok, "v_min_label": float("nan")}, lab) == 0


# ----------------------------------------------------------------- OpenDSS
def test_p1_reference_features_and_scc(cfg):
    pytest.importorskip("py_dss_interface")
    pb = D.plant_base(cfg, "P1")
    base = {"sample_id": 0, "plant": "P1", "isc3_kA": pb.isc3_kA, "static_load_mult": 1.0,
            **D._base_target_params(cfg, pb)}
    ev = D.get_evaluator(cfg, "P1", 1)
    on = ev.evaluate_spec({**base, "contingency": "N-1", "bits": "1" * 19})
    off = ev.evaluate_spec({**base, "contingency": "N-1", "bits": "0" * 19})
    n0 = ev.evaluate_spec({**base, "contingency": "N-0", "bits": "1" * 19})
    # Artigo de IC: Scc = 74,20 MVA em N-1; artigo SEPOC: Vmin 0,755 / 0,866.
    assert on["Scc_MVA"] == pytest.approx(74.205, abs=0.01)
    assert on["C_N1"] == 1 and n0["C_N1"] == 0
    assert n0["Scc_MVA"] > on["Scc_MVA"] + 30
    assert on["v_min_inrush"] == pytest.approx(0.755, abs=0.0005)
    assert off["v_min_inrush"] == pytest.approx(0.866, abs=0.0005)
    assert on["y"] == 0 and off["y"] == 1
    assert on["Pprev_kW"] > off["Pprev_kW"] + 8000
    assert set(FEATURES) <= set(on)


def test_edit_matches_recompile(cfg, tmp_path):
    """Edit (fonte, trafos, cargas, alvo) == recompilacao completa.
    Sob mutacao (p.ex. sem o Edit do isc3 ou das cargas do alvo) falha."""
    pytest.importorskip("py_dss_interface")
    s = D.validate(cfg, 3, 13, tmp_path / "val", ["P1"])["P1"]
    assert s["max_abs_d_v_min_inrush"] < 1e-9
    assert s["max_abs_d_v_max_post"] < 1e-9
    assert s["max_abs_d_tr_load_nom"] < 1e-9
    assert s["max_abs_d_Pprev_kW"] < 0.5          # warm start do fluxo (tol 1e-6)
    assert s["max_abs_d_Scc_edit_vs_fs"] < 1e-3   # analitico x faultstudy OpenDSS
    assert s["label_mismatch"] == 0 and s["converged_mismatch"] == 0
    assert s["base_N1_all_on"]["Scc_faultstudy_subproc"] == pytest.approx(74.205, abs=0.01)


def test_nsga2_history_recomputed(cfg):
    pytest.importorskip("py_dss_interface")
    plan, info = D.plan_history(cfg, 15, 0)
    assert info["n_unique_bits"] >= 15 and info["truncated"]
    out = D.evaluate_specs(cfg, plan, "nsga2_history", 0)
    assert np.nanmax(np.abs(out["v_min_inrush"] - out["hist_v_min_inrush"])) < 1e-9
    assert (out["y_sepoc"].astype(bool) == out["hist_feasible"].astype(bool)).all()


def test_run_dataset_blocks_resume_and_meta(cfg, tmp_path):
    pytest.importorskip("py_dss_interface")
    out = tmp_path / "ds" / "x.csv"
    m1 = D.run_dataset(cfg, "stratified_k", 30, 3, out, workers=1, block_size=10)
    assert m1["n_rows"] == 30 and m1["n_blocks"] == 3
    a = pd.read_csv(out, dtype={"bits": str})
    assert list(a.columns[:len(D.SPEC_COLUMNS)]) == D.SPEC_COLUMNS
    meta = json.loads((out.parent / "x.meta.json").read_text(encoding="utf-8"))
    assert meta["eval_ms_per_sample"]["mean"] > 0 and "versions" in meta
    with pytest.raises(FileExistsError):
        D.run_dataset(cfg, "stratified_k", 30, 3, out, workers=1, block_size=10)
    with pytest.raises(ValueError):
        D.run_dataset(cfg, "stratified_k", 31, 3, out, workers=1, block_size=10, resume=True)
    (out.parent / "x.blocks" / "block_000001.csv").unlink()   # bloco perdido
    m2 = D.run_dataset(cfg, "stratified_k", 30, 3, out, workers=1, block_size=10, resume=True)
    assert m2["n_blocks_reused"] == 2 and m2["n_blocks_computed"] == 1
    b = pd.read_csv(out, dtype={"bits": str})
    cols = [c for c in a.columns if c != "eval_ms"]
    pd.testing.assert_frame_equal(a[cols], b[cols], check_exact=False, atol=0.5)


def test_exhaustive_blocks(cfg, tmp_path):
    pytest.importorskip("py_dss_interface")
    out = tmp_path / "ex" / "e.csv"
    m = D.run_dataset(cfg, "exhaustive", 8, 0, out, workers=1, block_size=4)
    assert m["plan_info"]["n_total_space"] == 2 ** 19 and m["plan_info"]["truncated"]
    e = pd.read_csv(out, dtype={"bits": str})
    assert list(e["bits"]) == [D.bits_from_index(i, 19) for i in range(8)]
    assert (e["contingency"] == "N-1").all() and e["Scc_MVA"].nunique() == 1


# ----------------------------------------------------------------- curva
def _synthetic(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.normal(size=n) for f in FEATURES})
    df["y"] = (df["Pprev_kW"] - df["Scc_MVA"] + 0.3 * rng.normal(size=n) < 0).astype(int)
    df["v_min_inrush"] = 0.85 + 0.02 * rng.normal(size=n)
    df["n_on"] = rng.integers(0, 20, n)
    df["strategy"] = "synthetic"
    return df


def test_grid_and_sufficient_rule():
    assert LC.n_grid(100, 2, 1500) == [100, 200, 400, 800, 1500]
    summ = pd.DataFrame({"group": "g", "n_train": [100, 200, 400, 800],
                         "auc_mean": [0.90, 0.97, 0.995, 0.997],
                         "auc_width": [0.10, 0.03, 0.01, 0.0]})
    r = LC.sufficient_n(summ, "auc", 0.005, 0.02)["g"]
    assert r["n_sufficient"] == 400
    r2 = LC.sufficient_n(summ, "auc", 0.001, 0.02)["g"]
    assert r2["n_sufficient"] is None    # so o n maximo satisfaria (trivial)


def test_sufficient_rule_independent_reference_and_pool_frac():
    """Achado IC-V1: com referencia de n_max (1 ajuste) o n* pode sair
    favorecido; com referencia independente o proprio n_max entra no plato,
    e max_pool_frac tira da regra os n perto de N. O resultado declara que
    n* e relativo a n_max."""
    summ = pd.DataFrame({"group": "g", "n_train": [100, 200, 400, 800, 1500],
                         "n_reps": [20, 20, 20, 20, 1], "pool_size": 1500,
                         "acc_mean": [0.90, 0.940, 0.947, 0.950, 0.951],
                         "acc_width": [0.04, 0.019, 0.014, 0.010, 0.0]})
    summ["frac_of_pool"] = summ["n_train"] / summ["pool_size"]
    old = LC.sufficient_n(summ, "acc", 0.005, 0.02)["g"]
    assert old["n_sufficient"] == 400 and old["reference_kind"] == "n_max"
    assert old["relative_to_n_max"] is True and old["n_max"] == 1500
    # Referencia independente mais alta (0,954): 400 e 800 deixam o plato,
    # e o n_max (1 ajuste) entra no teste do plato mas nunca vira n*.
    ref = {"g": {"value": 0.954, "sd": 0.0, "n": 6000, "n_fits": 1, "kind": "independent"}}
    ind = LC.sufficient_n(summ, "acc", 0.005, 0.02, reference=ref)["g"]
    assert ind["n_sufficient"] == 800 and ind["reference_kind"] == "independent"
    assert ind["reference_n"] == 6000
    ref2 = {"g": {"value": 0.957, "sd": 0.0, "n": 6000, "n_fits": 1, "kind": "independent"}}
    assert LC.sufficient_n(summ, "acc", 0.005, 0.02, reference=ref2)["g"]["n_sufficient"] is None
    # So o n_max (1 ajuste, IC de largura 0 por construcao) passa no plato:
    # nao vira n*.
    ref3 = {"g": {"value": 0.9555, "sd": 0.0, "n": 6000, "n_fits": 1, "kind": "independent"}}
    assert LC.sufficient_n(summ, "acc", 0.005, 0.02, reference=ref3)["g"]["n_sufficient"] is None
    # max_pool_frac = 0,5: 800/1500 = 0,53 fica fora da regra.
    fr = LC.sufficient_n(summ, "acc", 0.005, 0.02, max_pool_frac=0.5)["g"]
    assert fr["n_sufficient"] == 400 and fr["n_rule_points"] == 3
    fr2 = LC.sufficient_n(summ, "acc", 0.005, 0.02, reference=ref, max_pool_frac=0.5)["g"]
    assert fr2["n_sufficient"] is None


def test_run_n_max_cap_and_independent_reference(tmp_path):
    """n_max_cap iguala o n_max entre bases de tamanhos diferentes e a
    referencia independente vem do pool informado (sem vazamento)."""
    pytest.importorskip("matplotlib")
    paths = {}
    for name, n, s in (("data", 600, 11), ("ref", 1200, 12), ("test", 400, 13)):
        d = _synthetic(n, s)
        d["C_N1"] = (d["C_N1"] > 0).astype(int)
        paths[name] = tmp_path / f"{name}.csv"
        d.to_csv(paths[name], index=False)
    cfg_lc = {"n_start": 50, "factor": 2.0, "n_reps": 4, "seed": 1,
              "n_max_cap": 300, "reference": "independent", "reference_n": 600, "reference_reps": 3}
    res = LC.run(cfg_lc, {"v_min_gt": 0.85}, [str(paths["data"])], tmp_path / "out",
                 test_data=[str(paths["test"])], reference_data=[str(paths["ref"])])
    assert res["pools"] == {"synthetic": 300} and res["pools_before_cap"] == {"synthetic": 600}
    assert res["n_sufficient_is_relative_to_n_max"] is True
    g = res["reference"]["groups"]["synthetic"]
    assert g["status"] == "ok" and g["pool_size"] == 1200 and g["overlap_ref_train"] == 0
    assert g["auc"]["n"] == 600 and g["auc"]["n_fits"] == 3
    for m in LC.METRICS:
        r = res["n_sufficient"][m]["synthetic"]
        assert r["reference_kind"] == "independent" and r["n_max"] == 300
        assert r["reference_value"] == pytest.approx(g[m]["value"])
    summ = pd.read_csv(tmp_path / "out" / "curve_summary.csv")
    assert summ["n_train"].max() == 300 and (summ["pool_size"] == 300).all()
    with pytest.raises(ValueError):
        LC.run({**cfg_lc, "n_max_cap": None}, {}, [str(paths["data"])], tmp_path / "o2",
               test_data=[str(paths["test"])], reference_data=None)


def test_learning_curve_and_bias_synthetic():
    pool, test = _synthetic(600, 1), _synthetic(400, 2)
    curve = LC.learning_curve(pool, test, [50, 100, 600], 5, seed=0, group="s")
    summ = LC.summarize(curve)
    assert list(summ["n_train"]) == [50, 100, 600]
    assert summ["auc_mean"].iloc[-1] > 0.95
    assert (curve.loc[curve.n_train == 600, "rep"] == 0).all()   # pool inteiro: 1 rep
    b = LC.bias_report(pool, 0.85, 0.01)
    expect = float(((pool["v_min_inrush"] - 0.85).abs() < 0.01).mean())
    assert b["frac_near_boundary"].iloc[0] == pytest.approx(expect)
