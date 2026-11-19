"""Curva de aprendizagem e comparacao de vies de amostragem (ESPEC secao 5).

Uso:
  python -m src.ic.learning_curve --config experiments/ic/config.yaml \
      --data experiments/ic/demo/p1_stratified_k_n1500_s1.csv experiments/ic/demo/p1_bernoulli_n1500_s2.csv \
      --test-data experiments/ic/demo/test_*.csv \
      --external experiments/ic/demo/p1_nsga2_history_s3.csv \
      --out experiments/ic/demo/learning_curve

O que faz:
  1. Conjunto de teste FIXO (`--test-data`, ou fracao estratificada de
     `--data` se ausente). Nunca entra no treino.
  2. Para cada grupo de treino (coluna `strategy`; e o pool unido quando ha
     mais de um), n_train numa grade geometrica (100, 200, 400, ... ate o
     disponivel) com `n_reps` reamostragens: pipeline StandardScaler +
     LogisticRegression L2 (C configuravel), metricas acuracia, recall da
     classe 0 e AUC no teste fixo.
  3. Regra objetiva de "n suficiente": menor n tal que, para ele e todos os
     maiores da grade, |media(n) - referencia| <= epsilon E a largura do
     intervalo (percentis das reamostragens, nivel ci_level) <= ci_width_max.
     O n* e SEMPRE relativo ao n_max do pool (o resultado declara isso):
     - `reference: n_max` (padrao antigo): referencia = UM ajuste no pool
       inteiro; n* tende a crescer com o tamanho da base e nao e propriedade
       da planta (achado IC-V1, 2026-11-14);
     - `reference: independent`: referencia = media de `reference_reps`
       ajustes num pool INDEPENDENTE (`--reference-data`, de preferencia
       maior), da mesma estrategia; o proprio n_max entra no teste do plato;
     - `n_max_cap`: teto comum do pool por grupo (subamostra estratificada de
       semente fixa); para comparar plantas, usar o MESMO valor;
     - `max_pool_frac`: so entram na regra os n com n/N <= fracao (perto de N
       as subamostras sem reposicao se sobrepoem, fracao esperada n/N, o que
       encolhe a largura do IC).
  4. Validacao externa (`--external`, p.ex. cenarios do NSGA-II): modelo
     treinado no pool inteiro de cada grupo, avaliado no conjunto externo;
     conta vetores de features identicos entre externo e treino/teste
     (vazamento).
  5. Vies: distribuicao de Pprev e fracao perto da fronteira
     |Vmin - V_ir| < delta por estrategia (todos os arquivos informados).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import FEATURES

log = logging.getLogger(__name__)
METRICS = ("accuracy", "recall0", "auc")

# Parametros da regra de "n suficiente" (achado IC-V1). Ficam aqui, e nao no
# experiments/ic/config.yaml, porque `dataset.config_digest` inclui o yaml
# inteiro e acrescentar chaves invalidaria o --resume das bases ja geradas.
# Podem ser sobrepostos no bloco `learning_curve:` do yaml ou pela CLI.
RULE_DEFAULTS: dict[str, Any] = {
    "reference": "n_max",    # [A DECIDIR pelo grupo] n_max (1 ajuste no pool inteiro) | independent
    "reference_n": None,     # [A DECIDIR pelo grupo] None = pool de referencia inteiro (1 ajuste)
    "reference_reps": 20,    # [A DECIDIR pelo grupo] ajustes quando reference_n < pool de referencia
    "n_max_cap": None,       # [A DECIDIR pelo grupo] teto comum do pool; comparar plantas com o MESMO
    "max_pool_frac": 1.0,    # [A DECIDIR pelo grupo] so n/N <= fracao entra na regra (1.0 = antigo)
}


# ===========================================================================
# Utilidades
# ===========================================================================
def load_data(paths: list[str | Path]) -> pd.DataFrame:
    dfs = [pd.read_csv(p, dtype={"bits": str}) for p in paths]
    for p, d in zip(paths, dfs):
        d["source_file"] = Path(p).name
    return pd.concat(dfs, ignore_index=True)


def n_grid(n_start: int, factor: float, n_max: int) -> list[int]:
    """Grade geometrica n_start, n_start*factor, ... (< n_max) + n_max."""
    if n_max < 2:
        return []
    out, n = [], float(n_start)
    while int(round(n)) < n_max:
        out.append(int(round(n)))
        n *= float(factor)
    out.append(int(n_max))
    return sorted(set(v for v in out if v >= 2))


def stratified_split(df: pd.DataFrame, frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    tr, te = train_test_split(df, test_size=frac, random_state=seed, stratify=df["y"])
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def subsample_idx(y: np.ndarray, n: int, rng: np.random.Generator, mode: str) -> np.ndarray:
    """Indices de uma subamostra de tamanho n (sem reposicao)."""
    N = len(y)
    if n >= N:
        return np.arange(N)
    if mode == "simple":
        return np.sort(rng.choice(N, n, replace=False))
    if mode != "stratified":
        raise ValueError(f"subsample invalido: {mode}")
    # Alocacao proporcional por classe (minimo 1 por classe presente).
    classes = np.unique(y)
    alloc = {c: max(1, int(round(n * np.mean(y == c)))) for c in classes}
    big = max(classes, key=lambda c: np.sum(y == c))
    alloc[big] += n - sum(alloc.values())
    idx = [rng.choice(np.flatnonzero(y == c), min(alloc[c], int(np.sum(y == c))), replace=False)
           for c in classes]
    return np.sort(np.concatenate(idx))


def make_model(C: float, max_iter: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # L2 explicito: sklearn >= 1.8 depreca `penalty` (usa l1_ratio=0 = L2).
    if _has_l1_ratio_api():
        lr = LogisticRegression(C=C, l1_ratio=0.0, max_iter=max_iter)
    else:  # pragma: no cover
        lr = LogisticRegression(C=C, penalty="l2", max_iter=max_iter)
    return make_pipeline(StandardScaler(), lr)


def _has_l1_ratio_api() -> bool:
    """sklearn >= 1.8 depreca `penalty`; usa l1_ratio=0 (L2) nessas versoes."""
    import sklearn

    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    return (major, minor) >= (1, 8)


def metrics(model, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

    pred = model.predict(X)
    out = {"accuracy": float(accuracy_score(y, pred)),
           "recall0": float(recall_score(y, pred, pos_label=0, zero_division=0))
           if np.any(y == 0) else float("nan")}
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, model.predict_proba(X)[:, 1]))
    else:
        out["auc"] = float("nan")
    return out


# ===========================================================================
# Curva de aprendizagem
# ===========================================================================
def learning_curve(pool: pd.DataFrame, test: pd.DataFrame, grid: list[int], n_reps: int,
                   C: float = 1.0, max_iter: int = 2000, subsample: str = "stratified",
                   seed: int = 0, group: str = "all",
                   features: list[str] | None = None) -> pd.DataFrame:
    feats = features or FEATURES
    X, y = pool[feats].to_numpy(float), pool["y"].to_numpy(int)
    Xt, yt = test[feats].to_numpy(float), test["y"].to_numpy(int)
    rng = np.random.default_rng(seed)
    rows = []
    for n in grid:
        if n > len(pool):
            continue
        reps = 1 if n >= len(pool) else n_reps
        for r in range(reps):
            idx = subsample_idx(y, n, rng, subsample)
            row: dict[str, Any] = {"group": group, "n_train": int(n), "rep": r,
                                   "n_pos": int(y[idx].sum()), "pool_size": int(len(pool))}
            if len(np.unique(y[idx])) < 2:
                row.update({m: float("nan") for m in METRICS})
                row["status"] = "uma_classe"
            else:
                mdl = make_model(C, max_iter).fit(X[idx], y[idx])
                row.update(metrics(mdl, Xt, yt))
                row["status"] = "ok"
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(curve: pd.DataFrame, ci_level: float = 0.95) -> pd.DataFrame:
    a = (1.0 - ci_level) / 2.0
    out = []
    for (g, n), d in curve.groupby(["group", "n_train"]):
        row: dict[str, Any] = {"group": g, "n_train": int(n), "n_reps": int(len(d)),
                               "n_valid": int(d[METRICS[0]].notna().sum())}
        if "pool_size" in d:
            # frac_of_pool = n/N = fracao esperada de pontos comuns entre duas
            # subamostras sem reposicao de tamanho n do mesmo pool de tamanho N.
            N = int(d["pool_size"].iloc[0])
            row.update({"pool_size": N, "frac_of_pool": float(n) / N if N else float("nan")})
        for m in METRICS:
            v = d[m].dropna().to_numpy(float)
            if len(v):
                lo, hi = np.quantile(v, [a, 1 - a])
                row.update({f"{m}_mean": float(v.mean()), f"{m}_sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                            f"{m}_lo": float(lo), f"{m}_hi": float(hi), f"{m}_width": float(hi - lo)})
            else:
                row.update({f"{m}_{k}": float("nan") for k in ("mean", "sd", "lo", "hi", "width")})
        out.append(row)
    return pd.DataFrame(out).sort_values(["group", "n_train"]).reset_index(drop=True)


def sufficient_n(summary: pd.DataFrame, metric: str, eps: float, width_max: float,
                 reference: dict[str, dict[str, Any]] | None = None,
                 max_pool_frac: float = 1.0) -> dict[str, Any]:
    """Menor n da grade a partir do qual o plato (eps) e a largura do IC
    (width_max) valem para todos os n maiores. Por grupo.

    reference: {grupo: {"value", "sd", "kind", "n", "n_fits", ...}}. Sem
    entrada para o grupo, a referencia e a media no maior n da curva (um
    ajuste no pool inteiro: kind "n_max", comportamento antigo). Com
    referencia externa, o maior n tambem e testado no plato.
    max_pool_frac: so entram na regra os n com n/N <= fracao.
    Pontos com uma so reamostragem (pool inteiro) nunca viram n* e sua
    largura de IC (0 por construcao) nao conta como evidencia."""
    res: dict[str, Any] = {}
    for g, d in summary.groupby("group"):
        d = d.sort_values("n_train").reset_index(drop=True)
        n_max = int(d["n_train"].iloc[-1])
        pool_n = int(d["pool_size"].iloc[-1]) if "pool_size" in d else n_max
        ref_in = (reference or {}).get(str(g))
        if ref_in is not None and ref_in.get("value") is not None and np.isfinite(ref_in["value"]):
            ref, kind = float(ref_in["value"]), str(ref_in.get("kind", "independent"))
            cand = d
        else:
            ref, kind = float(d[f"{metric}_mean"].iloc[-1]), "n_max"
            cand = d.iloc[:-1]  # o proprio n_max satisfaz trivialmente (diferenca 0, IC de largura 0)
        frac = cand["n_train"] / pool_n if pool_n else cand["n_train"] * np.nan
        cand = cand[frac <= float(max_pool_frac) + 1e-12]
        single = (cand["n_reps"] < 2).to_numpy() if "n_reps" in cand else np.zeros(len(cand), bool)
        plateau = ((cand[f"{metric}_mean"] - ref).abs() <= eps).to_numpy()
        width_ok = ((cand[f"{metric}_width"] <= width_max).to_numpy() | single)
        okv = plateau & width_ok
        n_star = None
        for i in range(len(okv)):
            if not single[i] and okv[i:].all():
                n_star = int(cand["n_train"].iloc[i])
                break
        if n_star is None:
            note = ("nao estabelecido: nenhum n com reamostragens satisfaz a regra ate o n_max; "
                    "aumentar a base ou rever eps/ci_width_max")
        else:
            note = "n* relativo a n_max (tamanho do pool) e a referencia; ver README"
        res[str(g)] = {"metric": metric, "epsilon": eps, "ci_width_max": width_max,
                       "reference_kind": kind,
                       "reference_n_max": n_max, "reference_value": ref,
                       "reference_sd": (float(ref_in.get("sd", float("nan"))) if kind != "n_max"
                                        else 0.0),
                       "reference_n": (int(ref_in.get("n", 0)) if kind != "n_max" else n_max),
                       "reference_n_fits": (int(ref_in.get("n_fits", 0)) if kind != "n_max" else 1),
                       "n_max": n_max, "pool_size": pool_n, "relative_to_n_max": True,
                       "max_pool_frac": float(max_pool_frac),
                       "n_rule_points": int(len(cand)),
                       "n_sufficient": n_star,
                       "frac_of_pool_at_n_star": (float(n_star) / pool_n if n_star is not None and pool_n
                                                  else None),
                       "note": note}
    return res


def reference_from_pool(ref_pool: pd.DataFrame, test: pd.DataFrame, n_ref: int | None, n_fits: int,
                        C: float, max_iter: int, subsample: str, seed: int,
                        features: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Referencia de plato a partir de um pool INDEPENDENTE do pool da curva.

    n_ref None (ou >= tamanho do pool): um ajuste no pool de referencia
    inteiro. Senao: media de `n_fits` ajustes em subamostras de tamanho n_ref.
    Devolve {metrica: {value, sd, n, n_fits}}."""
    feats = features or FEATURES
    N = len(ref_pool)
    n = N if (n_ref is None or int(n_ref) >= N) else int(n_ref)
    reps = 1 if n >= N else max(1, int(n_fits))
    curve = learning_curve(ref_pool, test, [n], reps, C, max_iter, subsample, seed, "ref", feats)
    out: dict[str, dict[str, Any]] = {}
    for m in METRICS:
        v = curve[m].dropna().to_numpy(float)
        out[m] = {"value": float(v.mean()) if len(v) else float("nan"),
                  "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                  "n": int(n), "n_fits": int(len(v)), "pool_size": int(N)}
    return out


# ===========================================================================
# Vies e validacao externa
# ===========================================================================
def bias_report(df: pd.DataFrame, v_ir: float, delta: float, by: str = "strategy") -> pd.DataFrame:
    rows = []
    for g, d in df.groupby(by):
        p = d["Pprev_kW"].to_numpy(float)
        q = np.quantile(p, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
        near = (d["v_min_inrush"] - v_ir).abs() < delta
        rows.append({by: g, "n": int(len(d)), "Pprev_mean": float(p.mean()), "Pprev_sd": float(p.std(ddof=1)),
                     "Pprev_min": q[0], "Pprev_p05": q[1], "Pprev_p25": q[2], "Pprev_p50": q[3],
                     "Pprev_p75": q[4], "Pprev_p95": q[5], "Pprev_max": q[6],
                     "n_on_mean": float(d["n_on"].mean()),
                     "frac_near_boundary": float(near.mean()), "n_near_boundary": int(near.sum()),
                     "frac_y1": float(d["y"].mean()),
                     "frac_N1": float(d["C_N1"].mean()),
                     "Scc_min": float(d["Scc_MVA"].min()), "Scc_max": float(d["Scc_MVA"].max()),
                     "n_unique_feature_vectors": int(d[FEATURES].round(9).drop_duplicates().shape[0])})
    return pd.DataFrame(rows)


def external_validation(pools: dict[str, pd.DataFrame], test: pd.DataFrame, ext: pd.DataFrame,
                        C: float, max_iter: int) -> pd.DataFrame:
    rows = []
    key = lambda d: set(map(tuple, d[FEATURES].round(9).to_numpy()))  # noqa: E731
    ext_keys = key(ext)
    test_keys = key(test)
    for g, pool in pools.items():
        mdl = make_model(C, max_iter).fit(pool[FEATURES].to_numpy(float), pool["y"].to_numpy(int))
        m_ext = metrics(mdl, ext[FEATURES].to_numpy(float), ext["y"].to_numpy(int))
        m_test = metrics(mdl, test[FEATURES].to_numpy(float), test["y"].to_numpy(int))
        rows.append({"group": g, "n_train": int(len(pool)), "n_external": int(len(ext)),
                     **{f"ext_{k}": v for k, v in m_ext.items()},
                     **{f"test_{k}": v for k, v in m_test.items()},
                     "overlap_ext_train": int(len(ext_keys & key(pool))),
                     "overlap_ext_test": int(len(ext_keys & test_keys))})
    return pd.DataFrame(rows)


# ===========================================================================
# Figuras
# ===========================================================================
def plot_curves(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, m in zip(axes, METRICS):
        for g, d in summary.groupby("group"):
            ax.plot(d["n_train"], d[f"{m}_mean"], marker="o", label=str(g))
            ax.fill_between(d["n_train"], d[f"{m}_lo"], d[f"{m}_hi"], alpha=0.2)
        ax.set_xscale("log")
        ax.set_xlabel("n de treino")
        ax.set_title({"accuracy": "acuracia", "recall0": "recall classe 0", "auc": "AUC"}[m])
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_bias(df: pd.DataFrame, v_ir: float, path: Path, by: str = "strategy") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    bins = np.linspace(df["Pprev_kW"].min(), df["Pprev_kW"].max(), 40)
    for g, d in df.groupby(by):
        axes[0].hist(d["Pprev_kW"], bins=bins, density=True, histtype="step", lw=1.5, label=str(g))
        axes[1].hist(d["v_min_inrush"], bins=40, density=True, histtype="step", lw=1.5, label=str(g))
    axes[0].set_xlabel("Pprev (kW)")
    axes[0].set_ylabel("densidade")
    axes[1].axvline(v_ir, color="k", ls="--", lw=1)
    axes[1].set_xlabel("Vmin na partida (pu)")
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ===========================================================================
# CLI
# ===========================================================================
def run(cfg_lc: dict, label_cfg: dict, data: list[str], out: str | Path,
        test_data: list[str] | None = None, external: list[str] | None = None,
        compare: list[str] | None = None, pooled: bool = True,
        reference_data: list[str] | None = None) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    seed = int(cfg_lc.get("seed", 0))
    cfg_lc = {**RULE_DEFAULTS, **cfg_lc}
    ref_kind = str(cfg_lc.get("reference", "n_max"))
    if ref_kind not in ("n_max", "independent"):
        raise ValueError(f"learning_curve.reference invalido: {ref_kind}")
    if ref_kind == "independent" and not reference_data:
        raise ValueError("reference: independent exige --reference-data (pool independente)")
    cap = cfg_lc.get("n_max_cap")
    cap = None if cap in (None, "", "null") else int(cap)
    max_pool_frac = float(cfg_lc.get("max_pool_frac", 1.0))
    df = load_data(data)
    if test_data:
        test = load_data(test_data)
        pool_all = df
        test_desc = {"kind": "arquivos", "files": [Path(p).name for p in test_data]}
    else:
        frac = float(cfg_lc.get("test_fraction", 0.3))
        pool_all, test = stratified_split(df, frac, seed)
        test_desc = {"kind": "split", "fraction": frac}
    test_desc.update({"n": int(len(test)), "frac_y1": float(test["y"].mean())})
    pools = {str(g): d.reset_index(drop=True) for g, d in pool_all.groupby("strategy")}
    if pooled and len(pools) > 1:
        pools["pooled"] = pool_all.reset_index(drop=True)
    pools_before_cap = {g: int(len(p)) for g, p in pools.items()}
    if cap is not None:
        # Teto comum de n_max: subamostra estratificada (por y) de semente fixa.
        for k, g in enumerate(list(pools)):
            p = pools[g]
            if len(p) > cap:
                rng_cap = np.random.default_rng(seed + 50_000 + k)
                idx = subsample_idx(p["y"].to_numpy(int), cap, rng_cap, "stratified")
                pools[g] = p.iloc[idx].reset_index(drop=True)
    curves = []
    for k, (g, pool) in enumerate(pools.items()):
        grid = n_grid(int(cfg_lc.get("n_start", 100)), float(cfg_lc.get("factor", 2.0)), len(pool))
        curves.append(learning_curve(pool, test, grid, int(cfg_lc.get("n_reps", 20)),
                                     float(cfg_lc.get("C", 1.0)), int(cfg_lc.get("max_iter", 2000)),
                                     cfg_lc.get("subsample", "stratified"), seed + 1000 * k, g))
    curve = pd.concat(curves, ignore_index=True)
    summ = summarize(curve, float(cfg_lc.get("ci_level", 0.95)))
    # Referencia do plato por grupo e metrica.
    refs: dict[str, dict[str, dict[str, Any]]] = {m: {} for m in METRICS}
    ref_desc: dict[str, Any] = {"kind": ref_kind}
    if ref_kind == "independent":
        rdf = load_data(reference_data)
        ref_pools = {str(g): d.reset_index(drop=True) for g, d in rdf.groupby("strategy")}
        if "pooled" in pools:
            ref_pools["pooled"] = rdf.reset_index(drop=True)
        key = lambda d: set(map(tuple, d[FEATURES].round(9).to_numpy()))  # noqa: E731
        rkeys, tkeys = key(rdf), key(test)
        ref_desc.update({"files": [Path(p).name for p in reference_data],
                         "overlap_ref_test": int(len(rkeys & tkeys)), "groups": {}})
        n_ref = cfg_lc.get("reference_n")
        n_ref = None if n_ref in (None, "", "null") else int(n_ref)
        for k, g in enumerate(pools):
            rp = ref_pools.get(g)
            if rp is None:
                ref_desc["groups"][g] = {"status": "sem pool de referencia da mesma estrategia; "
                                                   "usa n_max"}
                continue
            r = reference_from_pool(rp, test, n_ref, int(cfg_lc.get("reference_reps", 20)),
                                    float(cfg_lc.get("C", 1.0)), int(cfg_lc.get("max_iter", 2000)),
                                    cfg_lc.get("subsample", "stratified"), seed + 90_000 + k)
            for m in METRICS:
                refs[m][g] = {**r[m], "kind": "independent"}
            ref_desc["groups"][g] = {"status": "ok", "pool_size": int(len(rp)),
                                     "overlap_ref_train": int(len(rkeys & key(pools[g]))),
                                     **{m: r[m] for m in METRICS}}
    rule = {m: sufficient_n(summ, m, float(cfg_lc.get("epsilon", 0.005)),
                            float(cfg_lc.get("ci_width_max", 0.02)),
                            reference=refs[m] or None, max_pool_frac=max_pool_frac)
            for m in METRICS}
    curve.to_csv(out / "curve_raw.csv", index=False)
    summ.to_csv(out / "curve_summary.csv", index=False)
    plot_curves(summ, out / "learning_curve.png")
    res: dict[str, Any] = {"data": [Path(p).name for p in data], "test": test_desc,
                           "pools": {g: int(len(p)) for g, p in pools.items()},
                           "pools_before_cap": pools_before_cap, "n_max_cap": cap,
                           "reference": ref_desc, "max_pool_frac": max_pool_frac,
                           "n_sufficient_is_relative_to_n_max": True,
                           "primary_metric": cfg_lc.get("primary_metric", "auc"),
                           "n_sufficient": rule, "config": cfg_lc}
    v_ir = float(label_cfg.get("v_min_gt", 0.85))
    delta = float(cfg_lc.get("delta_boundary", 0.01))
    if external:
        ext = load_data(external)
        ev = external_validation(pools, test, ext, float(cfg_lc.get("C", 1.0)),
                                 int(cfg_lc.get("max_iter", 2000)))
        ev.to_csv(out / "external_validation.csv", index=False)
        res["external"] = ev.to_dict("records")
    bias_files = list(dict.fromkeys(list(data) + list(test_data or []) + list(external or [])
                                    + list(compare or [])))
    bdf = load_data(bias_files)
    bdf["strategy_file"] = bdf["strategy"] + " [" + bdf["source_file"] + "]"
    bias = bias_report(bdf, v_ir, delta, by="strategy_file")
    bias.to_csv(out / "bias_by_strategy.csv", index=False)
    plot_bias(bdf, v_ir, out / "bias_pprev_vmin.png", by="strategy_file")
    res["bias"] = bias.to_dict("records")
    res["delta_boundary"] = delta
    (out / "learning_curve_result.json").write_text(json.dumps(res, indent=2, default=str),
                                                    encoding="utf-8")
    return res


def main(argv: list[str] | None = None) -> int:
    from .dataset import DEFAULT_CONFIG, load_config

    ap = argparse.ArgumentParser(description="Curva de aprendizagem / vies (ESPEC secao 5)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--test-data", nargs="*", default=None)
    ap.add_argument("--external", nargs="*", default=None)
    ap.add_argument("--compare", nargs="*", default=None, help="outros arquivos so para o vies")
    ap.add_argument("--no-pooled", action="store_true")
    ap.add_argument("--reference-data", nargs="*", default=None,
                    help="pool independente para a referencia do plato (com --reference independent)")
    ap.add_argument("--reference", choices=["n_max", "independent"], default=None,
                    help="sobrepoe learning_curve.reference do config")
    ap.add_argument("--n-max-cap", type=int, default=None,
                    help="sobrepoe learning_curve.n_max_cap (teto comum do pool)")
    ap.add_argument("--max-pool-frac", type=float, default=None,
                    help="sobrepoe learning_curve.max_pool_frac")
    ap.add_argument("--reference-n", type=int, default=None,
                    help="sobrepoe learning_curve.reference_n")
    ap.add_argument("--reference-reps", type=int, default=None,
                    help="sobrepoe learning_curve.reference_reps")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(a.config)
    cfg_lc = dict(cfg.get("learning_curve") or {})
    for k, v in (("reference", a.reference), ("n_max_cap", a.n_max_cap),
                 ("max_pool_frac", a.max_pool_frac), ("reference_n", a.reference_n),
                 ("reference_reps", a.reference_reps)):
        if v is not None:
            cfg_lc[k] = v
    res = run(cfg_lc, cfg.get("label") or {}, a.data, a.out,
              a.test_data, a.external, a.compare, pooled=not a.no_pooled,
              reference_data=a.reference_data)
    print(json.dumps({k: res[k] for k in ("pools", "pools_before_cap", "test", "reference",
                                          "n_sufficient")}, indent=2, default=str))
    if "external" in res:
        print(json.dumps(res["external"], indent=2, default=str))
    print(json.dumps(res["bias"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
