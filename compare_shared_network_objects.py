#!/usr/bin/env python3
"""
Compare LS-LR-NVAR and Chen TV-LASSO on shared objects, not raw A vs Granger A.

(1) Granger / VAR operators
    Ours:  Phi_{k,t} = (I + W_t) odot B_{k,t}
    Chen:  local-linear lag coefficients alpha_k(tau)
    Strength S_ij = max_k |coeff_{k,ij}|, diagonal removed.

(2) Matched-sparsity / matched-density overlap
    Threshold our |Phi| (or A-probability) so the edge count equals Chen's
    Granger graph at that date, then score overlap and Jaccard.

Writes data/shared_network_comparison.json and data/shared_network_comparison.html.
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from chen_tv_nvar_6m_forecast import (
    BANDWIDTH,
    LAG_P,
    N_FACTORS,
    local_linear_stage,
    paper_pca,
)
from compare_networks_over_time import EDGE_TOL, N_GRID, granger_adj
from ls_lr_nvar_6m_forecast import HORIZON, LSLRNVAR

DATA_DIR = Path(__file__).resolve().parent / "data"
SEED = 0
OURS_NPZ = DATA_DIR / "lslrnvar_operators_train.npz"
CHEN_NPZ = DATA_DIR / "chen_operator_grid.npz"


def offdiag_vec(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return np.asarray(M)[mask]


def strength_from_phi(phis: list[np.ndarray], t: int) -> np.ndarray:
    S = np.maximum(np.abs(phis[0][t]), np.abs(phis[1][t]))
    np.fill_diagonal(S, 0.0)
    return S


def strength_from_alpha(alpha: np.ndarray, n: int, p: int) -> np.ndarray:
    S = np.zeros((n, n))
    for k in range(p):
        S = np.maximum(S, np.abs(alpha[:, k * n : (k + 1) * n]))
    np.fill_diagonal(S, 0.0)
    return S


def topk_adj(S: np.ndarray, k: int) -> np.ndarray:
    n = S.shape[0]
    A = np.zeros((n, n))
    if k <= 0:
        return A
    flat = np.abs(S).copy()
    np.fill_diagonal(flat, -np.inf)
    take = min(k, n * (n - 1))
    idx = np.argpartition(flat.ravel(), -take)[-take:]
    A.ravel()[idx] = 1.0
    np.fill_diagonal(A, 0.0)
    return A


def overlap_scores(A: np.ndarray, B: np.ndarray) -> dict[str, float]:
    a = A > 0
    b = B > 0
    inter = float(np.sum(a & b))
    union = float(np.sum(a | b))
    na, nb = float(a.sum()), float(b.sum())
    return {
        "n_ours": na,
        "n_chen": nb,
        "intersection": inter,
        "overlap": (inter / na) if na else np.nan,
        "jaccard": (inter / union) if union else np.nan,
        "recall_in_dense": (inter / nb) if nb else np.nan,
    }


def rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size < 3 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    rho = spearmanr(x, y).correlation
    return float(rho) if rho is not None and np.isfinite(rho) else float("nan")


def kendall_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size < 3 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    tau = kendalltau(x, y, nan_policy="omit").correlation
    return float(tau) if tau is not None and np.isfinite(tau) else float("nan")


def norm_frobenius(A: np.ndarray, B: np.ndarray) -> float:
    a = np.asarray(A, dtype=float).copy()
    b = np.asarray(B, dtype=float).copy()
    np.fill_diagonal(a, 0.0)
    np.fill_diagonal(b, 0.0)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-15 and nb < 1e-15:
        return 0.0
    if na < 1e-15 or nb < 1e-15:
        return 1.0
    return float(np.linalg.norm(a / na - b / nb))


def companion_radius(A1: np.ndarray, A2: np.ndarray) -> float:
    n = A1.shape[0]
    C = np.zeros((2 * n, 2 * n))
    C[:n, :n] = A1
    C[:n, n:] = A2
    C[n:, :n] = np.eye(n)
    ev = np.linalg.eigvals(C)
    return float(np.max(np.abs(ev)))


def nanmean(xs: list[float]) -> float:
    a = np.asarray(xs, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def load_or_fit_ours(
    train_ret: np.ndarray,
    dates: pd.DatetimeIndex,
    ours_npz: Path,
) -> tuple[np.ndarray, list[np.ndarray]]:
    if ours_npz.exists():
        z = np.load(ours_npz)
        A_mean = z["A_mean"]
        phis = [z["Phi1"], z["Phi2"]]
        if A_mean.shape[0] != len(dates):
            raise RuntimeError("Cached operator length does not match training dates")
        print(f"Loading cached LS-LR-NVAR operators from {ours_npz.name}")
        return A_mean, phis
    print("Fitting LS-LR-NVAR and collecting posterior-mean A_t and Phi_t ...")
    np.random.seed(SEED)
    model = LSLRNVAR()
    model.fit(train_ret, collect_A=True)
    if model.A_mean is None or model.Phi_mean is None:
        raise RuntimeError("LS-LR-NVAR did not store A_mean / Phi_mean")
    A_mean = model.A_mean
    phis = model.Phi_mean
    np.savez_compressed(
        ours_npz,
        A_mean=A_mean,
        Phi1=phis[0],
        Phi2=phis[1],
        dates=dates.strftime("%Y-%m-%d").to_numpy(),
    )
    return A_mean, phis


def load_or_fit_chen(
    Y: np.ndarray,
    dates: pd.DatetimeIndex,
    label: str,
    key: str,
    chen_npz: Path,
) -> dict:
    if chen_npz.exists():
        z = np.load(chen_npz, allow_pickle=False)
        needed = {f"{key}_alpha", f"{key}_t", f"{key}_tau"}
        if needed.issubset(z.files):
            print(f"Loading cached Chen {label} operators from {chen_npz.name}")
            return {
                "alpha": z[f"{key}_alpha"],
                "t": z[f"{key}_t"].astype(int),
                "tau": z[f"{key}_tau"],
                "dates": z[f"{key}_dates"],
            }
    T, n = Y.shape
    taus = np.linspace(0.18, 1.0, N_GRID)
    print(f"Chen {label}: LassoCV at tau=1 ...")
    _, _, _, lams = local_linear_stage(Y, p=LAG_P, tau=1.0, h=BANDWIDTH, n_obs=T, lasso=True, verbose=True)
    alphas = np.zeros((N_GRID, n, LAG_P * n))
    t_idx = np.zeros(N_GRID, dtype=int)
    date_s = []
    for i, tau in enumerate(taus):
        t = min(T - 1, max(0, int(round(tau * T)) - 1))
        print(f"  {label}  {i + 1}/{N_GRID}  tau={tau:.2f}  {dates[t].date()}")
        _, alpha, _, _ = local_linear_stage(
            Y, p=LAG_P, tau=tau, h=BANDWIDTH, n_obs=T, lasso=True, lam_vec=lams, verbose=False
        )
        alphas[i] = alpha
        t_idx[i] = t
        date_s.append(dates[t].date().isoformat())
    return {"alpha": alphas, "t": t_idx, "tau": taus, "dates": np.array(date_s)}


def save_chen_cache(fa: dict, direct: dict, chen_npz: Path) -> None:
    np.savez_compressed(
        chen_npz,
        fa_alpha=fa["alpha"],
        fa_t=fa["t"],
        fa_tau=fa["tau"],
        fa_dates=fa["dates"],
        dir_alpha=direct["alpha"],
        dir_t=direct["t"],
        dir_tau=direct["tau"],
        dir_dates=direct["dates"],
    )


def score_grid(
    A_mean: np.ndarray,
    phis: list[np.ndarray],
    chen: dict,
    n: int,
    p: int,
) -> list[dict]:
    rows = []
    for i in range(len(chen["tau"])):
        t = int(chen["t"][i])
        alpha = chen["alpha"][i]
        S_o = strength_from_phi(phis, t)
        S_c = strength_from_alpha(alpha, n, p)
        G = granger_adj(alpha, n, p, EDGE_TOL)
        k = int(G.sum())
        A_phi = topk_adj(S_o, k)
        A_prob = topk_adj(A_mean[t], k)
        A_dense = (A_mean[t] >= 0.5).astype(float)
        np.fill_diagonal(A_dense, 0.0)
        ov_phi = overlap_scores(A_phi, G)
        ov_prob = overlap_scores(A_prob, G)
        ov_dense = overlap_scores(A_dense, G)
        vo, vc = offdiag_vec(S_o), offdiag_vec(S_c)
        A1o, A2o = phis[0][t], phis[1][t]
        A1c, A2c = alpha[:, :n], alpha[:, n:]
        rows.append(
            {
                "date": str(chen["dates"][i]),
                "tau": float(chen["tau"][i]),
                "n_chen_edges": k,
                "spearman_strength": rank_corr(vo, vc),
                "kendall_strength": kendall_corr(vo, vc),
                "norm_frobenius": norm_frobenius(S_o, S_c),
                "rho_ours": companion_radius(A1o, A2o),
                "rho_chen": companion_radius(A1c, A2c),
                "spearman_out_influence": rank_corr(S_o.sum(axis=1), S_c.sum(axis=1)),
                "spearman_in_influence": rank_corr(S_o.sum(axis=0), S_c.sum(axis=0)),
                "matched_phi_overlap": ov_phi["overlap"],
                "matched_phi_jaccard": ov_phi["jaccard"],
                "matched_A_overlap": ov_prob["overlap"],
                "matched_A_jaccard": ov_prob["jaccard"],
                "chen_inside_dense_A": ov_dense["recall_in_dense"],
                "spearman_degree_phi": rank_corr(A_phi.sum(axis=1), G.sum(axis=1)),
                "spearman_degree_A": rank_corr(A_prob.sum(axis=1), G.sum(axis=1)),
            }
        )
    return rows


def _r(xs: list[float], nd: int = 4) -> list[float | None]:
    out = []
    for v in xs:
        if v is None or not np.isfinite(v):
            out.append(None)
        else:
            out.append(round(float(v), nd))
    return out


def build_payload(fa_rows: list[dict], dir_rows: list[dict], explained: float) -> dict:
    labels = [datetime.fromisoformat(r["date"]).strftime("%b %y") for r in dir_rows]

    def pack(rows: list[dict], name: str) -> dict:
        keys = [
            "spearman_strength",
            "kendall_strength",
            "norm_frobenius",
            "rho_ours",
            "rho_chen",
            "spearman_out_influence",
            "spearman_in_influence",
            "matched_phi_overlap",
            "matched_phi_jaccard",
            "matched_A_overlap",
            "matched_A_jaccard",
            "chen_inside_dense_A",
            "spearman_degree_phi",
            "spearman_degree_A",
            "n_chen_edges",
        ]
        series = {k: _r([r[k] for r in rows]) for k in keys}
        means = {k: (None if not np.isfinite(nanmean([r[k] for r in rows])) else round(nanmean([r[k] for r in rows]), 4)) for k in keys}
        return {"name": name, "series": series, "means": means}

    return {
        "window": {
            "start": dir_rows[0]["date"],
            "end": dir_rows[-1]["date"],
            "n_grid": len(dir_rows),
            "factor_var_explained": explained,
        },
        "labels": labels,
        "iso_dates": [r["date"] for r in dir_rows],
        "fa": pack(fa_rows, "Chen factor-adjusted LASSO"),
        "direct": pack(dir_rows, "Chen direct LASSO"),
        "ours_rho": _r([r["rho_ours"] for r in dir_rows]),
    }


def html_report(payload: dict) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)
    fa, d = payload["fa"]["means"], payload["direct"]["means"]
    w = payload["window"]

    def pct(x):
        return "—" if x is None else f"{100 * x:.1f}%"

    def num(x, nd=3):
        return "—" if x is None else f"{x:.{nd}f}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Shared-object network comparison — LS-LR-NVAR vs Chen TV-LASSO</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #181818; --text: #e6e6e6; --muted: #9a9a9a; --faint: #6e6e6e;
    --card: #222222; --stroke: #333333;
    --ours: #3fb950; --fa: #f85149; --dir: #d29922; --accent: #79c0ff;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 24px; background: var(--bg); color: var(--text);
    font: 14px/1.5 ui-sans-serif, system-ui, Segoe UI, sans-serif;
  }}
  .wrap {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{ font-size: 24px; font-weight: 600; margin: 0 0 8px; }}
  h2 {{ font-size: 18px; font-weight: 600; margin: 28px 0 8px; }}
  h3 {{ font-size: 15px; font-weight: 600; margin: 20px 0 8px; }}
  p.lead, p.note {{ color: var(--muted); margin: 0 0 12px; }}
  p.caption {{ color: var(--faint); font-size: 12px; margin: 6px 0 0; }}
  .interpret {{
    border-left: 2px solid #4a6a88; padding: 8px 12px; margin: 10px 0 4px;
    color: var(--text); background: #1c2228;
  }}
  .interpret strong {{ display: block; color: var(--accent); font-size: 12px; margin-bottom: 4px; }}
  .pills {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 16px; }}
  .pill {{
    border: 1px solid var(--stroke); background: transparent; color: var(--muted);
    border-radius: 999px; padding: 3px 10px; font-size: 12px;
  }}
  .pill.active {{ background: #2a2a2a; color: var(--text); border-color: #555; }}
  .callout {{
    border: 1px solid #2f4f6f; background: #1c2833; padding: 12px 14px; border-radius: 8px;
    margin: 16px 0 20px;
  }}
  .callout strong {{ display: block; margin-bottom: 4px; }}
  .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 16px 0 8px; }}
  .stat .v {{ font-size: 22px; font-weight: 600; }}
  .stat .l {{ color: var(--muted); font-size: 12px; }}
  .stat.ours .v {{ color: var(--ours); }}
  .stat.fa .v {{ color: var(--fa); }}
  .stat.dir .v {{ color: var(--dir); }}
  .chart {{ background: var(--card); border: 1px solid var(--stroke); border-radius: 8px; padding: 12px; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; border: 1px solid var(--stroke); }}
  th, td {{ padding: 7px 10px; border-bottom: 1px solid var(--stroke); text-align: left; }}
  th {{ background: #202020; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tr:nth-child(even) {{ background: #1c1c1c; }}
  hr {{ border: 0; border-top: 1px solid var(--stroke); margin: 28px 0; }}
  @media (max-width: 800px) {{
    .stats, .grid2 {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Shared-object comparison — LS-LR-NVAR vs Chen TV-LASSO</h1>
  <p class="lead">
    The named adjacency matrices are different scientific objects. This report compares them after mapping both models onto the same quantities: the VAR lag operator, coefficient-strength rankings, and graphs with a matched number of edges.
    Grid: {w['n_grid']} dates from {w['start']} to {w['end']}. Chen factor-adjusted uses idiosyncratics after PCA k = 3 ({w['factor_var_explained']*100:.1f}% of return variance).
  </p>
  <div class="pills">
    <span class="pill active">Same 36-date tau grid</span>
    <span class="pill">Ours: posterior-mean Phi and A</span>
    <span class="pill">Chen: local-linear LASSO alpha</span>
  </div>

  <div class="stats">
    <div class="stat dir"><div class="v">{num(d['spearman_strength'])}</div><div class="l">Mean Spearman of |Phi| vs |alpha| (direct)</div></div>
    <div class="stat fa"><div class="v">{num(fa['spearman_strength'])}</div><div class="l">Mean Spearman of |Phi| vs |alpha| (factor-adj.)</div></div>
    <div class="stat dir"><div class="v">{pct(d['matched_phi_overlap'])}</div><div class="l">Mean matched-sparsity overlap (direct)</div></div>
    <div class="stat dir"><div class="v">{num(d['spearman_out_influence'])}</div><div class="l">Mean out-influence Spearman (direct)</div></div>
  </div>

  <div class="callout">
    <strong>What is being compared</strong>
    <b>Operator / Granger.</b> LS-LR-NVAR contributes Phi<sub>k,t</sub> = (I + W<sub>t</sub>) ⊙ B<sub>k,t</sub>, the matrix that actually multiplies lagged returns. Chen contributes the local-linear lag blocks alpha<sub>k</sub>(τ). Strength S<sub>ij</sub> is the largest absolute lag coefficient from j to i (own lags dropped).
    <br/><br/>
    <b>Matched sparsity.</b> At each date we keep Chen’s LASSO support (k edges) and keep the k largest |Phi<sub>ij</sub>| (or k largest A-probabilities). Overlap equals precision equals recall at that matched k. Jaccard is |∩| / |∪|.
    <br/><br/>
    Direct LASSO is the clean operator match (both estimated on raw returns). Factor-adjusted alpha is estimated on idiosyncratics, so that column is a residual-Granger comparison, not a reduced-form VAR match.
  </div>

  <h2>Average scores</h2>
  <p class="note">Means over dates where the metric is defined (Chen empty graphs drop rank correlations and overlap). Normalized Frobenius is ||Ŝ_ours − Ŝ_chen|| after each strength matrix is scaled to unit F-norm (0 = identical direction, √2 ≈ 1.41 = orthogonal).</p>
  <table>
    <thead>
      <tr><th>Measure</th><th class="num">vs Chen direct</th><th class="num">vs Chen factor-adjusted</th></tr>
    </thead>
    <tbody>
      <tr><td>Spearman of off-diagonal |Phi| vs |alpha|</td><td class="num">{num(d['spearman_strength'])}</td><td class="num">{num(fa['spearman_strength'])}</td></tr>
      <tr><td>Kendall of off-diagonal |Phi| vs |alpha|</td><td class="num">{num(d['kendall_strength'])}</td><td class="num">{num(fa['kendall_strength'])}</td></tr>
      <tr><td>Normalized Frobenius distance of S</td><td class="num">{num(d['norm_frobenius'])}</td><td class="num">{num(fa['norm_frobenius'])}</td></tr>
      <tr><td>Spearman of out-influence (row sums of S)</td><td class="num">{num(d['spearman_out_influence'])}</td><td class="num">{num(fa['spearman_out_influence'])}</td></tr>
      <tr><td>Spearman of in-influence (column sums of S)</td><td class="num">{num(d['spearman_in_influence'])}</td><td class="num">{num(fa['spearman_in_influence'])}</td></tr>
      <tr><td>Matched-sparsity overlap of top-|Phi| vs Granger</td><td class="num">{pct(d['matched_phi_overlap'])}</td><td class="num">{pct(fa['matched_phi_overlap'])}</td></tr>
      <tr><td>Chance overlap at the same k (k / 870)</td><td class="num">{pct(None if d['n_chen_edges'] is None else d['n_chen_edges'] / 870)}</td><td class="num">{pct(None if fa['n_chen_edges'] is None else fa['n_chen_edges'] / 870)}</td></tr>
      <tr><td>Matched-sparsity Jaccard of top-|Phi| vs Granger</td><td class="num">{pct(d['matched_phi_jaccard'])}</td><td class="num">{pct(fa['matched_phi_jaccard'])}</td></tr>
      <tr><td>Matched-density overlap of top-P(A) vs Granger</td><td class="num">{pct(d['matched_A_overlap'])}</td><td class="num">{pct(fa['matched_A_overlap'])}</td></tr>
      <tr><td>Matched-density Jaccard of top-P(A) vs Granger</td><td class="num">{pct(d['matched_A_jaccard'])}</td><td class="num">{pct(fa['matched_A_jaccard'])}</td></tr>
      <tr><td>Out-degree Spearman of those matched graphs</td><td class="num">{num(d['spearman_degree_phi'])}</td><td class="num">{num(fa['spearman_degree_phi'])}</td></tr>
      <tr><td>Share of Chen Granger edges inside A ≥ 0.5</td><td class="num">{pct(d['chen_inside_dense_A'])}</td><td class="num">{pct(fa['chen_inside_dense_A'])}</td></tr>
      <tr><td>Mean companion spectral radius (Chen)</td><td class="num">{num(d['rho_chen'])}</td><td class="num">{num(fa['rho_chen'])}</td></tr>
    </tbody>
  </table>
  <p class="caption">LS-LR-NVAR mean companion radius on this grid: {num(nanmean([x for x in payload['ours_rho'] if x is not None]))}. Source: compare_shared_network_objects.py.</p>

  <h2>Coefficient-strength agreement</h2>
  <p class="note">Rank correlation of the 870 off-diagonal |Phi<sub>ij</sub>| vs |alpha<sub>ij</sub>| values. Near 0 means the models do not rank the same directed pairs as strong lagged links.</p>
  <div class="chart"><canvas id="spearman" height="90"></canvas></div>
  <p class="caption">Spearman ρ · dates with no Chen variation (empty or one-valued alpha) are omitted.</p>
  <div class="interpret">
    <strong>Interpretation</strong>
    Spearman ρ ranks the 870 off-diagonal |Φ<sub>ij</sub>| against |α<sub>ij</sub>|. Its purpose is to test whether LS-LR-NVAR and Chen put the same directed pairs at the top of the lagged-strength list, regardless of coefficient scale.
    <br/><br/>
    They do not. Versus Chen direct the correlation stays near zero (mean −0.009): slightly positive in 2019–2020 (at most +0.05) and negative after 2024 (about −0.07 at the origin). Versus Chen factor-adjusted it also stays inside ±0.08, with gaps when the residual graph is empty. Neither Chen spec ranks lagged pairs the way LS-LR-NVAR does.
  </div>

  <div class="grid2">
    <div>
      <h3>Kendall τ of |Phi| vs |alpha|</h3>
      <div class="chart"><canvas id="kendall" height="140"></canvas></div>
      <p class="caption">Same pairs as the Spearman panel; more robust to a few large coefficients.</p>
      <div class="interpret">
        <strong>Interpretation</strong>
        Kendall τ is the same pairwise ranking test as Spearman, but it is less sensitive to a few very large coefficients. Its purpose is to check that any agreement (or disagreement) is not driven by LASSO outliers.
        <br/><br/>
        LS-LR-NVAR and Chen still disagree. Versus direct LASSO the mean is −0.007 and the path tracks Spearman; versus factor-adjusted it is +0.004. Down-weighting large α does not bring the two models’ lag orderings together.
      </div>
    </div>
    <div>
      <h3>Normalized Frobenius distance</h3>
      <div class="chart"><canvas id="frobenius" height="140"></canvas></div>
      <p class="caption">Scale-free distance of strength matrices. Smaller is closer. A random direction is around 1.4.</p>
      <div class="interpret">
        <strong>Interpretation</strong>
        Normalized Frobenius is ||Ŝ<sub>ours</sub> − Ŝ<sub>chen</sub>|| after each strength matrix is scaled to unit length. Its purpose is to measure how close the two lag-strength patterns are once units are removed (0 = the same direction, ≈1.41 = orthogonal).
        <br/><br/>
        LS-LR-NVAR’s S and Chen’s S stay far apart: 1.20–1.38 versus direct (mean 1.31) and mean 1.33 versus factor-adjusted. The only modest dip is 2019–2020, when Chen is densest; even then the operators do not point the same way. After scale is stripped, the two models’ strength patterns are nearly orthogonal.
      </div>
    </div>
  </div>

  <h2>Who transmits shocks</h2>
  <p class="note">Out-influence is the row sum of S (how strongly name i is driven by others). In-influence is the column sum (how strongly name j drives others). Spearman is over the 30 names.</p>
  <div class="grid2">
    <div>
      <h3>Out-influence rank correlation</h3>
      <div class="chart"><canvas id="outinf" height="140"></canvas></div>
      <div class="interpret">
        <strong>Interpretation</strong>
        Out-influence is the row sum of S: how strongly each name is driven by the others. Spearman of those 30 scores asks whether LS-LR-NVAR and Chen agree on which stocks are the most inbound.
        <br/><br/>
        They do not. Versus Chen direct the correlation wanders between about −0.23 and +0.13 (mean −0.02) and never stays positive. Versus factor-adjusted it is noisier (few residual edges) with mean +0.04. The names most driven in Φ are not the names most driven in Chen’s α.
      </div>
    </div>
    <div>
      <h3>In-influence rank correlation</h3>
      <div class="chart"><canvas id="ininf" height="140"></canvas></div>
      <div class="interpret">
        <strong>Interpretation</strong>
        In-influence is the column sum of S: how strongly each name drives the others. Spearman of those 30 scores asks whether the two models agree on who sends lagged shocks.
        <br/><br/>
        Here LS-LR-NVAR and Chen disagree most clearly. Versus direct LASSO the mean is −0.13, and from 2022 onward the correlation is about −0.3 to −0.48. Names Chen treats as Granger sources (GS, AXP, JPM at the origin) tend to be weak transmitters in Φ, while LS-LR-NVAR’s association hubs (HD, CSCO, HON) are not Chen’s sources. The models disagree on who sends shocks, not only on who receives them.
      </div>
    </div>
  </div>
  <p class="caption">Direct LASSO is the relevant series for reduced-form influence. Factor-adjusted influence is among idiosyncratics.</p>

  <h2>Companion spectral radius</h2>
  <p class="note">Largest absolute eigenvalue of the VAR(2) companion built from (Phi<sub>1</sub>, Phi<sub>2</sub>) vs (alpha<sub>1</sub>, alpha<sub>2</sub>). This is a property of the operator, not of the named network drawing.</p>
  <div class="chart"><canvas id="rho" height="90"></canvas></div>
  <p class="caption">LS-LR-NVAR last-origin draws are capped at 0.92 in the forecast program; the path here is the unrestricted posterior-mean operator at each date.</p>
  <div class="interpret">
    <strong>Interpretation</strong>
    The companion spectral radius is the largest absolute eigenvalue of the VAR(2) companion built from (Φ<sub>1</sub>, Φ<sub>2</sub>) or (α<sub>1</sub>, α<sub>2</sub>). Its purpose is to compare how persistent each model’s lag operator is — a property of the dynamics, not of the named network drawing.
    <br/><br/>
    The two models do not share a persistence path. LS-LR-NVAR stays in a 0.25–0.40 band and only rises to 0.49 at the origin (moderate, stable). Chen direct tracks its own density cycle: near 0.50 in 2020, essentially 0 in 2022 when the Granger graph collapses, then 0.30 by February 2026. Chen factor-adjusted is ~0 on almost every date because residual lags are tiny. LS-LR-NVAR’s operator is steadier; Chen’s shrinks and expands with sparsity.
  </div>

  <h2>Matched-sparsity overlap</h2>
  <p class="note">At each date Chen has k Granger edges. Keep the k largest |Phi<sub>ij</sub>| from LS-LR-NVAR. Overlap is the share of those k that coincide with Chen. A random directed graph with the same k would overlap at about k / 870.</p>
  <div class="chart"><canvas id="overlapPhi" height="90"></canvas></div>
  <p class="caption">Matched-sparsity overlap and Jaccard of top-|Phi| vs Chen Granger support.</p>
  <div class="interpret">
    <strong>Interpretation</strong>
    At each date Chen has k Granger edges. Matched-sparsity overlap keeps the k largest |Φ<sub>ij</sub>| from LS-LR-NVAR and reports the share that coincide with Chen (Jaccard is |∩| / |∪|). The purpose is to ask whether the two models select the same directed links once they are forced to the same number of edges. The grey line is chance, k / 870.
    <br/><br/>
    They do not select the same links. Versus Chen direct, overlap rises with Chen’s 2019–2020 density to about 25% (chance that day is ~23%), then is 0 from January 2022 onward — below chance (still 2–9% while Chen is sparse). The mean (7.9%) is no better than average chance (9.8%). Versus factor-adjusted the mean overlap is 0.5% against a 1% chance baseline. After matching k, LS-LR-NVAR’s strongest lags are not Chen’s Granger edges.
  </div>

  <h3>Matched-density overlap using A probabilities</h3>
  <p class="note">Same k, but rank pairs by posterior P(A<sub>ij</sub>=1) instead of |Phi|. This asks whether Chen’s Granger links sit among our strongest contemporaneous associations, not among our strongest lags.</p>
  <div class="chart"><canvas id="overlapA" height="90"></canvas></div>
  <p class="caption">The A ≥ 0.5 graph covers ~79% of pairs, so almost every Chen edge falls inside it by construction. That series is shown only as a caution, not as evidence of agreement.</p>
  <div class="interpret">
    <strong>Interpretation</strong>
    This is the same matched-k overlap, but pairs are ranked by LS-LR-NVAR’s posterior P(A<sub>ij</sub>=1) instead of |Φ<sub>ij</sub>|. Its purpose is to test whether Chen’s Granger links sit among our strongest contemporaneous associations. The green series is the share of Chen edges that fall inside the dense A ≥ 0.5 graph (~79% of pairs).
    <br/><br/>
    The association network does not recover Chen’s Granger graph. Top-P(A) overlap versus direct averages 6.8%, below chance. The green series near 80% is mechanical: because LS-LR-NVAR’s 0.5-graph already covers most pairs, Chen edges land inside it without being special. Versus factor-adjusted, matched top-P(A) overlap is 0. Our contemporaneous hubs are not a shortlist of Chen’s lagged edges.
  </div>

  <h2>Matched-graph degree ranks</h2>
  <p class="note">Spearman of the 30 out-degrees on the two k-edge graphs. Agreement here means the models name the same hubs after the edge counts are forced to match.</p>
  <div class="chart"><canvas id="deg" height="90"></canvas></div>
  <p class="caption">Out-degree Spearman on matched top-|Phi| graphs (solid idea) and matched top-P(A) graphs.</p>
  <div class="interpret">
    <strong>Interpretation</strong>
    After both graphs are reduced to the same k edges, Spearman of the 30 out-degrees asks whether LS-LR-NVAR and Chen name the same hubs. Top-|Φ| is the Granger-operator comparison; top-P(A) is the association-network comparison.
    <br/><br/>
    They name different hubs. Mean Spearman versus Chen direct is −0.07 on top-|Φ| degrees and 0.00 on top-P(A) degrees; versus factor-adjusted it is −0.05. LS-LR-NVAR’s high-degree names stay HD, CSCO, HON; Chen direct concentrates out-degree on GS, AXP, JPM by the origin. Matching the edge count does not reconcile who is central.
  </div>

  <hr/>
  <h2>Protocol</h2>
  <div class="grid2">
    <div>
      <h3>LS-LR-NVAR</h3>
      <p class="note">Blocked Gibbs, 24 sweeps, 12 burn-in, thin 2 (6 kept draws). Posterior mean of daily Phi<sub>k,t</sub> and A<sub>t</sub>. Evaluated at the training index nearest each Chen tau.</p>
    </div>
    <div>
      <h3>Chen TV-LASSO</h3>
      <p class="note">Equation-wise LassoCV at tau = 1, then that lambda on 36 local-linear fits (h = 0.20, p = 2). Granger edge if any lag is nonzero. Own lags dropped. Direct: raw returns. Factor-adjusted: PCA residuals.</p>
    </div>
  </div>
  <p class="caption">Programs: compare_shared_network_objects.py, ls_lr_nvar_6m_forecast.py, chen_tv_nvar_6m_forecast.py · data/shared_network_comparison.json</p>

  <hr/>
  <h2>Overall assessment</h2>
  <p class="note">
    Purpose of this section: put the chart-by-chart scores on one footing and say whether LS-LR-NVAR and Chen TV-LASSO recover the same dynamic network for these 30 DJIA names. Direct LASSO is the clean comparison (both estimated on raw returns). Factor-adjusted LASSO is Chen’s preferred spec for correlated panels, but its α is residual Granger after PCA, not a reduced-form match to Φ.
  </p>
  <div class="callout">
    <strong>Verdict</strong>
    After mapping both models onto the same objects — lag operators, matched-k edges, influence ranks, and companion persistence — LS-LR-NVAR and Chen do not describe the same network. They are complementary views (contemporaneous association that modulates lags vs sparse time-varying Granger), not two estimates of one graph. Treating our A<sub>t</sub> as Chen’s Granger adjacency, or the reverse, would be a category error.
  </div>
  <div class="grid2">
    <div>
      <h3>Where they differ</h3>
      <p class="note">
        Lag-strength rankings are uncorrelated: mean Spearman of |Φ| vs |α| is {num(d['spearman_strength'])} versus direct and {num(fa['spearman_strength'])} versus factor-adjusted. Normalized Frobenius stays near 1.3 (random is ~1.41), so the strength matrices are nearly orthogonal. Matched-k overlap versus direct is {pct(d['matched_phi_overlap'])}, no better than chance ({pct(None if d['n_chen_edges'] is None else d['n_chen_edges'] / 870)}); versus factor-adjusted it is {pct(fa['matched_phi_overlap'])}. In-influence is the sharpest split (mean {num(d['spearman_in_influence'])} vs direct): Chen’s late-sample sources are GS, AXP, JPM; LS-LR-NVAR’s hubs stay HD, CSCO, HON. Persistence also diverges — our companion radius is stable (~0.32, 0.49 at the origin); Chen direct follows its density cycle (≈0.50 in 2020, ≈0 in 2022, 0.30 at the origin); factor-adjusted is essentially 0.
      </p>
    </div>
    <div>
      <h3>What that implies</h3>
      <p class="note">
        LS-LR-NVAR is the model of a dense, almost undirected, slowly moving association network that damps lags through (I+W)⊙B. Chen is the model of a sparse, hub-and-spoke Granger graph that thickens around 2020 and thins in 2022. Factor adjustment removes most of those Granger links, which is why Chen’s paper-recommended spec looks empty next to ours. The 6-month open-loop forecast comparison is consistent with this: LS-LR-NVAR beat Chen factor-adjusted on path MAPE (10.81% vs 12.22%), while Chen direct was slightly tighter (10.18%) — a mean-path ranking, not evidence that the two networks coincide.
      </p>
    </div>
  </div>

  <h2>Why propose LS-LR-NVAR rather than use Chen et al.</h2>
  <p class="note">
    The case for our model is the estimand, not a claim that every sparse VAR is obsolete. Chen, Li, Li and Linton (JoE 2025) estimate a time-varying <em>Granger</em> network by local-linear LASSO. For equity panels they recommend first removing common factors, then shrinking the idiosyncratics. That is the right tool if the object of interest is a few directed lagged links after market factors have been taken out. It is the wrong default if the object is the locally stationary <em>association</em> structure that actually enters a return VAR.
  </p>
  <div class="grid2">
    <div>
      <h3>What our model is for</h3>
      <p class="note">
        Equity returns are densely and almost symmetrically dependent. A 30-name DJIA panel has a correlation floor that already covers most pairs; a probit network with low-rank latent factors is a natural model of that graph. LS-LR-NVAR puts that graph into the mean: y<sub>t</sub> is driven by a low-rank local level and by (I+W<sub>t</sub>)⊙B<sub>t</sub>, with B<sub>t</sub> itself low-rank and locally stationary. The network is a parameter of the dynamics, not a post-hoc drawing of nonzero LASSO coefficients. That lets us talk about density, reciprocity, clustering, and slowly moving hubs (HD, CSCO, HON, SHW) as features of the market, rather than as artifacts of a penalty.
      </p>
    </div>
    <div>
      <h3>Why Chen is not a substitute</h3>
      <p class="note">
        On the same DJIA sample, Chen’s factor-adjusted Granger graph has mean density 1.1% and is often empty; the direct LASSO graph cycles from ~24% around 2020 to ~2% in 2022. After we map both models onto Φ versus α, rank correlations sit on zero and matched-k overlap is at or below chance. The two estimators are not two readings of one network. Using Chen in place of LS-LR-NVAR would throw away the contemporaneous, reciprocal structure that our model is written to capture, and — under Chen’s own recommended spec for correlated panels — would also worsen the 6-month open-loop path (MAPE 12.22% and Spearman 0.07 versus 10.81% and 0.19).
      </p>
    </div>
  </div>
  <div class="callout">
    <strong>How we would state the contribution</strong>
    We recommend LS-LR-NVAR when the aim is to estimate a locally stationary low-rank network VAR for a correlated financial panel: a dense association network, a low-rank local mean, and a low-rank lag loading, with uncertainty from a blocked Gibbs sampler. We do not recommend it as a sparse Granger detector. Chen remaining useful as a complementary check: if the question is which directed lags survive L1 after factors, their Section 5 specification is the benchmark we already beat on this holdout; if the question is raw-return sparsity, their direct LASSO can be slightly tighter on path MAPE (10.18%) and should be reported as a robustness comparator, not as the model of the association network.
  </div>
  <p class="note">
    Limitations we should keep in the paper: the implemented probit uses a correlation-threshold floor, so reported density on equities is partly inherited from contemporaneous |corr|; on a weak-signal blocked-SBM simulation that floor does not fire and Â<sub>t</sub> collapses. Gibbs here used a short run (24 sweeps, 6 kept draws). These are implementation and identification caveats, not reasons to prefer Chen’s estimand for the problem we state.
  </p>
</div>
<script>
const DATA = {data_json};
const colors = {{
  ours: "#3fb950", fa: "#f85149", direct: "#d29922",
  grid: "#333333", text: "#9a9a9a"
}};
Chart.defaults.color = colors.text;
Chart.defaults.borderColor = colors.grid;
Chart.defaults.font.family = "ui-sans-serif, system-ui, Segoe UI, sans-serif";
Chart.defaults.spanGaps = true;

function lineSet(label, data, color) {{
  return {{
    label, data, borderColor: color, backgroundColor: color,
    tension: 0.15, pointRadius: 2, pointHoverRadius: 4, borderWidth: 2
  }};
}}
function lineChart(id, datasets, yLabel, beginAtZero) {{
  return new Chart(document.getElementById(id), {{
    type: "line",
    data: {{ labels: DATA.labels, datasets }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ position: "top" }} }},
      scales: {{
        y: {{ beginAtZero: !!beginAtZero, title: {{ display: true, text: yLabel }} }},
        x: {{ ticks: {{ maxRotation: 60, minRotation: 45, autoSkip: true, maxTicksLimit: 18 }} }}
      }}
    }}
  }});
}}

lineChart("spearman", [
  lineSet("vs Chen direct LASSO", DATA.direct.series.spearman_strength, colors.direct),
  lineSet("vs Chen factor-adjusted", DATA.fa.series.spearman_strength, colors.fa)
], "Spearman ρ");

lineChart("kendall", [
  lineSet("vs Chen direct LASSO", DATA.direct.series.kendall_strength, colors.direct),
  lineSet("vs Chen factor-adjusted", DATA.fa.series.kendall_strength, colors.fa)
], "Kendall τ");

lineChart("frobenius", [
  lineSet("vs Chen direct LASSO", DATA.direct.series.norm_frobenius, colors.direct),
  lineSet("vs Chen factor-adjusted", DATA.fa.series.norm_frobenius, colors.fa)
], "Normalized ||S_ours − S_chen||_F", true);

lineChart("outinf", [
  lineSet("vs Chen direct LASSO", DATA.direct.series.spearman_out_influence, colors.direct),
  lineSet("vs Chen factor-adjusted", DATA.fa.series.spearman_out_influence, colors.fa)
], "Spearman ρ");

lineChart("ininf", [
  lineSet("vs Chen direct LASSO", DATA.direct.series.spearman_in_influence, colors.direct),
  lineSet("vs Chen factor-adjusted", DATA.fa.series.spearman_in_influence, colors.fa)
], "Spearman ρ");

lineChart("rho", [
  lineSet("LS-LR-NVAR Phi", DATA.ours_rho, colors.ours),
  lineSet("Chen direct alpha", DATA.direct.series.rho_chen, colors.direct),
  lineSet("Chen factor-adjusted alpha", DATA.fa.series.rho_chen, colors.fa)
], "Spectral radius", true);

lineChart("overlapPhi", [
  lineSet("Overlap vs direct", DATA.direct.series.matched_phi_overlap, colors.direct),
  lineSet("Chance vs direct (k/870)", DATA.direct.series.n_chen_edges.map(k => k == null ? null : k / 870), "#8b8b8b"),
  lineSet("Jaccard vs direct", DATA.direct.series.matched_phi_jaccard, "#b08800"),
  lineSet("Overlap vs factor-adj.", DATA.fa.series.matched_phi_overlap, colors.fa)
], "Share of matched-k edges", true);

lineChart("overlapA", [
  lineSet("top-P(A) overlap vs direct", DATA.direct.series.matched_A_overlap, colors.direct),
  lineSet("Chen edges inside A ≥ 0.5 (direct)", DATA.direct.series.chen_inside_dense_A, colors.ours)
], "Share", true);

lineChart("deg", [
  lineSet("top-|Phi| degrees vs direct", DATA.direct.series.spearman_degree_phi, colors.direct),
  lineSet("top-P(A) degrees vs direct", DATA.direct.series.spearman_degree_A, colors.ours),
  lineSet("top-|Phi| degrees vs factor-adj.", DATA.fa.series.spearman_degree_phi, colors.fa)
], "Spearman ρ");
</script>
</body>
</html>
"""


def _project_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def main(data_dir: Path | str | None = None, out: Path | str | None = None, open_browser: bool = True) -> Path:
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    ours_npz = data_dir / "lslrnvar_A_mean_train_v2.npz"
    if not ours_npz.exists():
        ours_npz = data_dir / "lslrnvar_operators_train.npz"
    chen_npz = data_dir / "chen_operator_grid.npz"

    prices = pd.read_csv(data_dir / "djia_adj_close_wide_10y.csv", parse_dates=["Date"], index_col="Date")
    prices = prices.sort_index().ffill().dropna(how="all")
    log_ret = np.diff(np.log(prices.values), axis=0)
    train_ret = log_ret[:-HORIZON]
    ret_dates = prices.index[1:-HORIZON]
    n = train_ret.shape[1]
    print(f"Train returns {train_ret.shape}  {ret_dates[0].date()} to {ret_dates[-1].date()}")

    A_mean, phis = load_or_fit_ours(train_ret, ret_dates, ours_npz)

    need_chen = not chen_npz.exists()
    mu = train_ret.mean(axis=0)
    _, _, U, explained = paper_pca(train_ret - mu, N_FACTORS)
    print(f"PCA k={N_FACTORS}, variance explained={explained:.3f}")
    chen_fa = load_or_fit_chen(U, ret_dates, "factor-adjusted", "fa", chen_npz)
    chen_dir = load_or_fit_chen(train_ret, ret_dates, "direct", "dir", chen_npz)
    if need_chen or not chen_npz.exists():
        save_chen_cache(chen_fa, chen_dir, chen_npz)
        print(f"Saved {chen_npz.name}")

    fa_rows = score_grid(A_mean, phis, chen_fa, n, LAG_P)
    dir_rows = score_grid(A_mean, phis, chen_dir, n, LAG_P)
    payload = build_payload(fa_rows, dir_rows, explained)

    json_path = data_dir / "shared_network_comparison.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out_path = Path(out) if out is not None else data_dir / "shared_network_comparison.html"
    out_path.write_text(html_report(payload), encoding="utf-8")

    dm, fm = payload["direct"]["means"], payload["fa"]["means"]
    print(f"Wrote {out_path}")
    print(
        "Direct   Spearman |Phi|vs|a| {0}  matched overlap {1}  out-inf {2}  F {3}".format(
            dm["spearman_strength"], dm["matched_phi_overlap"], dm["spearman_out_influence"], dm["norm_frobenius"]
        )
    )
    print(
        "FA       Spearman |Phi|vs|a| {0}  matched overlap {1}  out-inf {2}  F {3}".format(
            fm["spearman_strength"], fm["matched_phi_overlap"], fm["spearman_out_influence"], fm["norm_frobenius"]
        )
    )
    if open_browser:
        webbrowser.open(out_path.resolve().as_uri())
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare shared network objects and write HTML.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    main(data_dir=args.data_dir, out=args.out, open_browser=not args.no_open)
