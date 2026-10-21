#!/usr/bin/env python3
"""
Compare time-varying network measures: LS-LR-NVAR vs Chen TV-LASSO.

Our model: posterior-mean probit adjacency A_t from the blocked Gibbs sampler.
Chen et al.: local Granger graph at tau = t/n, edge i<-j if any lag coefficient
of j in the i equation is nonzero after local-linear LASSO (paper eqs. 2.5, 3.4).
Primary Chen spec is the Section 5 factor-adjusted idiosyncratic VAR.

Measures are computed on the training sample (through 2026-02-05) and then
aggregated to month-end so the two models share a time axis.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from chen_tv_nvar_6m_forecast import (
    BANDWIDTH,
    LAG_P,
    N_FACTORS,
    local_linear_stage,
    paper_pca,
)
from ls_lr_nvar_6m_forecast import HORIZON, LSLRNVAR

DATA_DIR = Path(__file__).resolve().parent / "data"
SEED = 0
N_GRID = 36
EDGE_TOL = 1e-8


def directed_measures(A: np.ndarray) -> dict[str, float]:
    A = np.asarray(A, dtype=float).copy()
    n = A.shape[0]
    np.fill_diagonal(A, 0.0)
    e = float(A.sum())
    denom = n * (n - 1)
    density = e / denom if denom else 0.0
    out_deg = A.sum(axis=1)
    in_deg = A.sum(axis=0)
    mutual = float(np.sum((A > 0) & (A.T > 0)) / 2.0)
    reciprocity = (2.0 * mutual / e) if e > 0 else 0.0

    U = np.maximum(A, A.T)
    np.fill_diagonal(U, 0.0)
    und_e = float(U.sum() / 2.0)
    und_density = (2.0 * und_e) / denom if denom else 0.0
    clustering = transitivity(U)
    n_comp, largest = undirected_components(U)
    gini = degree_gini(out_deg)
    return {
        "n_edges": e,
        "density": density,
        "mean_out_degree": float(out_deg.mean()),
        "mean_in_degree": float(in_deg.mean()),
        "reciprocity": reciprocity,
        "undirected_density": und_density,
        "clustering": clustering,
        "n_components": float(n_comp),
        "largest_component_share": largest / n,
        "out_degree_gini": gini,
        "max_out_degree": float(out_deg.max()),
    }


def transitivity(U: np.ndarray) -> float:
    """Global clustering on an undirected 0/1 adjacency."""
    U = (U > 0).astype(float)
    np.fill_diagonal(U, 0.0)
    triangles = float(np.trace(U @ U @ U) / 6.0)
    deg = U.sum(axis=1)
    triples = float(np.sum(deg * (deg - 1) / 2.0))
    if triples <= 0:
        return 0.0
    return 3.0 * triangles / triples


def undirected_components(U: np.ndarray) -> tuple[int, int]:
    n = U.shape[0]
    adj = (U + U.T) > 0
    seen = np.zeros(n, dtype=bool)
    n_comp = 0
    largest = 0
    for i in range(n):
        if seen[i]:
            continue
        n_comp += 1
        stack = [i]
        seen[i] = True
        size = 0
        while stack:
            v = stack.pop()
            size += 1
            nbrs = np.where(adj[v])[0]
            for w in nbrs:
                if not seen[w]:
                    seen[w] = True
                    stack.append(int(w))
        largest = max(largest, size)
    isolated = int((~seen).sum())
    n_comp += isolated
    return n_comp, largest


def degree_gini(deg: np.ndarray) -> float:
    x = np.sort(np.asarray(deg, dtype=float))
    n = x.size
    s = x.sum()
    if s <= 0 or n == 0:
        return 0.0
    return float(2.0 * np.sum(np.arange(1, n + 1) * x) / (n * s) - (n + 1) / n)


def granger_adj(alpha: np.ndarray, n: int, p: int, tol: float = EDGE_TOL) -> np.ndarray:
    """alpha is (n, p*n): row i, block k is lag-k coefficients on all sources."""
    A = np.zeros((n, n))
    for k in range(p):
        block = np.abs(alpha[:, k * n : (k + 1) * n]) > tol
        A = np.maximum(A, block.astype(float))
    np.fill_diagonal(A, 0.0)
    return A


def top_names(A: np.ndarray, tickers: list[str], k: int = 5) -> list[str]:
    out_deg = A.sum(axis=1)
    order = np.argsort(-out_deg)
    return [f"{tickers[i]} ({int(out_deg[i])})" for i in order[:k] if out_deg[i] > 0]


def monthly_from_daily(dates: pd.DatetimeIndex, A_path: np.ndarray, tickers: list[str]) -> pd.DataFrame:
    """A_path: (T, n, n) posterior mean probabilities. Threshold 0.5 for topology."""
    rows = []
    df_idx = pd.Series(np.arange(len(dates)), index=dates)
    month_end = df_idx.resample("ME").last().dropna()
    for stamp, t in month_end.items():
        t = int(t)
        P = A_path[t]
        A = (P >= 0.5).astype(float)
        m = directed_measures(A)
        m["date"] = stamp.date().isoformat()
        m["expected_density"] = float(P.sum() / (P.shape[0] * (P.shape[0] - 1)))
        m["top_out_degree"] = ", ".join(top_names(A, tickers))
        rows.append(m)
    return pd.DataFrame(rows)


def chen_grid(
    Y: np.ndarray,
    dates: pd.DatetimeIndex,
    tickers: list[str],
    p: int,
    h: float,
    n_grid: int,
    label: str,
) -> pd.DataFrame:
    T, n = Y.shape
    taus = np.linspace(0.18, 1.0, n_grid)
    print(f"Chen {label}: choosing equation-wise LASSO penalties at tau=1 ...")
    _, _, _, lams = local_linear_stage(Y, p=p, tau=1.0, h=h, n_obs=T, lasso=True, verbose=True)
    rows = []
    for i, tau in enumerate(taus):
        t = min(T - 1, max(0, int(round(tau * T)) - 1))
        print(f"  {label}  {i + 1}/{n_grid}  tau={tau:.2f}  {dates[t].date()}")
        _, alpha, _, _ = local_linear_stage(
            Y, p=p, tau=tau, h=h, n_obs=T, lasso=True, lam_vec=lams, verbose=False
        )
        A = granger_adj(alpha, n, p)
        m = directed_measures(A)
        m["date"] = dates[t].date().isoformat()
        m["tau"] = float(tau)
        m["expected_density"] = m["density"]
        m["top_out_degree"] = ", ".join(top_names(A, tickers))
        rows.append(m)
    return pd.DataFrame(rows)


def main() -> None:
    np.random.seed(SEED)
    prices = pd.read_csv(DATA_DIR / "djia_adj_close_wide_10y.csv", parse_dates=["Date"], index_col="Date")
    prices = prices.sort_index().ffill().dropna(how="all")
    tickers = prices.columns.tolist()
    log_ret = np.diff(np.log(prices.values), axis=0)
    train_ret = log_ret[:-HORIZON]
    ret_dates = prices.index[1:-HORIZON]
    print(f"Train returns {train_ret.shape}  {ret_dates[0].date()} to {ret_dates[-1].date()}")

    ours_npz = DATA_DIR / "lslrnvar_A_mean_train.npz"
    if ours_npz.exists():
        print(f"Loading cached LS-LR-NVAR A_mean from {ours_npz.name}")
        A_mean = np.load(ours_npz)["A_mean"]
        if A_mean.shape[0] != len(ret_dates):
            raise RuntimeError("Cached A_mean length does not match training dates")
    else:
        print("Fitting LS-LR-NVAR and collecting posterior-mean A_t ...")
        model = LSLRNVAR()
        model.fit(train_ret, collect_A=True)
        if model.A_mean is None:
            raise RuntimeError("LS-LR-NVAR did not store A_mean")
        A_mean = model.A_mean
        np.savez_compressed(ours_npz, A_mean=A_mean, dates=ret_dates.strftime("%Y-%m-%d").to_numpy())
    ours_m = monthly_from_daily(ret_dates, A_mean, tickers)
    ours_m.insert(0, "model", "LS-LR-NVAR")

    print("Chen factor-adjusted TV-LASSO Granger networks on a time grid ...")
    mu = train_ret.mean(axis=0)
    _, _, U, explained = paper_pca(train_ret - mu, N_FACTORS)
    print(f"  PCA k={N_FACTORS}, variance explained={explained:.3f}")
    chen_fa = chen_grid(U, ret_dates, tickers, LAG_P, BANDWIDTH, N_GRID, "factor-adjusted")
    chen_fa.insert(0, "model", "Chen factor-adjusted LASSO")

    print("Chen direct TV-LASSO Granger networks on a time grid ...")
    chen_dir = chen_grid(train_ret, ret_dates, tickers, LAG_P, BANDWIDTH, N_GRID, "direct")
    chen_dir.insert(0, "model", "Chen direct LASSO")

    long = pd.concat([ours_m, chen_fa, chen_dir], ignore_index=True)
    long.to_csv(DATA_DIR / "network_measures_over_time.csv", index=False)

    metric_cols = [
        "density",
        "expected_density",
        "n_edges",
        "mean_out_degree",
        "reciprocity",
        "clustering",
        "largest_component_share",
        "out_degree_gini",
        "n_components",
        "max_out_degree",
    ]

    def series_for(df):
        return {
            "dates": df["date"].tolist(),
            **{c: [round(float(v), 4) for v in df[c].tolist()] for c in metric_cols},
            "top_start": df.iloc[0]["top_out_degree"],
            "top_mid": df.iloc[len(df) // 2]["top_out_degree"],
            "top_end": df.iloc[-1]["top_out_degree"],
        }

    payload = {
        "window": {
            "start": str(ret_dates[0].date()),
            "end": str(ret_dates[-1].date()),
            "n_stocks": len(tickers),
            "ours_resolution": "month-end of daily posterior-mean A_t (threshold 0.5)",
            "chen_resolution": f"{N_GRID} local-linear LASSO Granger graphs, tau in [0.18, 1]",
            "factor_var_explained": explained,
        },
        "summary": {},
        "ours": series_for(ours_m),
        "chen_fa": series_for(chen_fa),
        "chen_dir": series_for(chen_dir),
    }
    for name, df in (
        ("LS-LR-NVAR", ours_m),
        ("Chen factor-adjusted LASSO", chen_fa),
        ("Chen direct LASSO", chen_dir),
    ):
        payload["summary"][name] = {
            "mean_density": float(df["density"].mean()),
            "end_density": float(df["density"].iloc[-1]),
            "mean_reciprocity": float(df["reciprocity"].mean()),
            "mean_clustering": float(df["clustering"].mean()),
            "mean_gini": float(df["out_degree_gini"].mean()),
            "mean_lcc": float(df["largest_component_share"].mean()),
        }
        print(
            f"{name:32s}  mean density {df['density'].mean():.3f}  "
            f"end {df['density'].iloc[-1]:.3f}  "
            f"recip {df['reciprocity'].mean():.3f}  "
            f"clust {df['clustering'].mean():.3f}"
        )

    (DATA_DIR / "network_measures_over_time.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print("Saved network_measures_over_time.csv and network_measures_over_time.json")


if __name__ == "__main__":
    main()
