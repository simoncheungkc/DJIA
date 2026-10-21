#!/usr/bin/env python3
"""Rebuild network rankings with a non-degenerate eigenvector.

Two graphs are scored:

  1. Rolling Pearson |corr| > 0.35 over the last 126 forecast origins
     (the proxy used in the rolling notebook — not the Gibbs network).
  2. If lslrnvar_A_mean_train_v2.npz exists, the Gibbs posterior-mean
     probit probabilities (threshold 0.5), averaged over the training path.

Eigenvector centrality is the principal eigenvector of the symmetrised
binary adjacency, L2-normalised. It is never computed on row-stochastic W.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
HORIZON = 126
NET_WINDOW = 252
THRESHOLD = 0.35
CORE_Q = 0.95


def eigen_centrality(A: np.ndarray) -> np.ndarray:
    S = 0.5 * (np.asarray(A, dtype=float) + np.asarray(A, dtype=float).T)
    np.fill_diagonal(S, 0.0)
    n = S.shape[0]
    if float(S.sum()) <= 0:
        return np.zeros(n)
    vals, vecs = np.linalg.eigh(S)
    x = np.real(vecs[:, int(np.argmax(vals))])
    x = np.abs(x)
    nrm = np.linalg.norm(x)
    return x / nrm if nrm > 0 else x


def rolling_corr_networks(prices: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], pd.DatetimeIndex]:
    tickers = prices.columns.tolist()
    log_ret = np.diff(np.log(prices.values), axis=0)
    ret_dates = prices.index[1:]
    T, n = log_ret.shape
    start = max(5 * 252, T - HORIZON)
    H = min(HORIZON, T - start)
    A_series = np.zeros((H, n, n))
    C_series = np.zeros((H, n, n))
    dates = ret_dates[start : start + H]
    for i, t in enumerate(range(start, start + H)):
        R = log_ret[max(0, t - NET_WINDOW) : t]
        C = np.corrcoef(R.T)
        C = np.nan_to_num(C, nan=0.0)
        np.fill_diagonal(C, 0.0)
        A = (np.abs(C) > THRESHOLD).astype(float)
        np.fill_diagonal(A, 0.0)
        A_series[i] = A
        C_series[i] = C
    return A_series, C_series, tickers, dates


def score_path(A_series: np.ndarray, tickers: list[str]) -> pd.DataFrame:
    H, n, _ = A_series.shape
    degree = A_series.sum(axis=2)
    eigen = np.vstack([eigen_centrality(A_series[t]) for t in range(H)])
    t_idx = np.arange(H, dtype=float)
    eigen_trend = np.array([np.polyfit(t_idx, eigen[:, j], 1)[0] for j in range(n)])
    degree_trend = np.array([np.polyfit(t_idx, degree[:, j], 1)[0] for j in range(n)])
    persistence = (degree > 0).mean(axis=0)
    df = pd.DataFrame(
        {
            "ticker": tickers,
            "avg_degree": degree.mean(axis=0),
            "avg_eigen_cent": eigen.mean(axis=0),
            "eigen_trend": eigen_trend,
            "degree_trend": degree_trend,
            "persistence": persistence,
            "is_core": persistence >= CORE_Q,
        }
    )
    df["importance"] = df["persistence"] * df["avg_degree"]
    df = df.sort_values(["avg_eigen_cent", "avg_degree"], ascending=False).reset_index(drop=True)
    df["eigen_rank"] = np.arange(1, n + 1)
    deg = df.sort_values("avg_degree", ascending=False).reset_index(drop=True)
    deg["degree_rank"] = np.arange(1, n + 1)
    return df, deg


def main() -> None:
    prices = pd.read_csv(DATA_DIR / "djia_adj_close_wide_10y.csv", parse_dates=["Date"], index_col="Date")
    prices = prices.sort_index().ffill().dropna(how="all")
    A_series, C_series, tickers, dates = rolling_corr_networks(prices)
    eigen_df, deg_df = score_path(A_series, tickers)

    risk_path = DATA_DIR / "djia_risk_adjusted_ranking.csv"
    risk = pd.read_csv(risk_path) if risk_path.exists() else None

    deg_out = deg_df.copy()
    deg_out["rank"] = deg_out["degree_rank"]
    deg_out.to_csv(DATA_DIR / "network_importance_ranking.csv", index=False)
    deg_out[deg_out["is_core"]].to_csv(DATA_DIR / "network_core_stocks.csv", index=False)
    deg_out.head(6).to_csv(DATA_DIR / "network_top6_stocks.csv", index=False)

    eigen_df.to_csv(DATA_DIR / "centrality_trajectories_summary.csv", index=False)

    payload = {
        "protocol": "rolling Pearson |corr| > 0.35, 252-day window, last 126 origins",
        "not_gibbs": True,
        "n_days": int(A_series.shape[0]),
        "start": str(dates[0].date()),
        "end": str(dates[-1].date()),
        "mean_density": float(A_series.sum() / (A_series.shape[0] * A_series.shape[1] * (A_series.shape[1] - 1))),
        "density_path": [float(A_series[t].sum() / (A_series.shape[1] * (A_series.shape[1] - 1))) for t in range(A_series.shape[0])],
    }

    if risk is not None:
        merged = eigen_df.merge(
            risk.rename(columns={"rank": "risk_rank"})[["ticker", "risk_adj_score", "risk_rank"]],
            on="ticker",
            how="left",
        )
        merged = merged.merge(deg_df[["ticker", "degree_rank"]], on="ticker", how="left")
        merged.to_csv(DATA_DIR / "network_plus_risk_ranking.csv", index=False)
        top_deg = set(deg_df.nsmallest(10, "degree_rank")["ticker"] if "degree_rank" in deg_df else deg_df.head(10)["ticker"])
        # deg_df is sorted by degree already
        top_deg = set(deg_df.head(10)["ticker"])
        top_eig = set(eigen_df.head(10)["ticker"])
        top_risk = set(risk.head(10)["ticker"])
        payload["risk_top10"] = risk.head(10)["ticker"].tolist()
        payload["degree_top10"] = deg_df.head(10)["ticker"].tolist()
        payload["eigen_top10"] = eigen_df.head(10)["ticker"].tolist()
        payload["intersection_degree_risk"] = sorted(top_deg & top_risk)
        payload["intersection_eigen_risk"] = sorted(top_eig & top_risk)
        rows = []
        for _, r in risk.iterrows():
            tkr = r["ticker"]
            drow = deg_df[deg_df["ticker"] == tkr].iloc[0]
            erow = eigen_df[eigen_df["ticker"] == tkr].iloc[0]
            rows.append(
                {
                    "ticker": tkr,
                    "risk_rank": int(r["rank"]),
                    "risk_score": float(r["risk_adj_score"]),
                    "degree_rank": int(drow["degree_rank"]),
                    "avg_degree": float(drow["avg_degree"]),
                    "eigen_rank": int(erow["eigen_rank"]),
                    "avg_eigen_cent": float(erow["avg_eigen_cent"]),
                    "risk_top10": int(r["rank"]) <= 10,
                    "degree_top10": int(drow["degree_rank"]) <= 10,
                    "eigen_top10": int(erow["eigen_rank"]) <= 10,
                }
            )
        payload["comparison"] = rows
        print("Degree AND risk top-10:", payload["intersection_degree_risk"])
        print("Eigen AND risk top-10:", payload["intersection_eigen_risk"])

    for label, fname, csv_pref in (
        ("gibbs", "lslrnvar_A_mean_train_v2.npz", "gibbs"),
        ("tvp", "tvplrnvarsv_A_mean_train.npz", "tvp"),
    ):
        npz = DATA_DIR / fname
        if not npz.exists():
            continue
        z = np.load(npz)
        P = z["A_mean"]
        A_hard = (P >= 0.5).astype(float)
        g_eigen, g_deg = score_path(A_hard, tickers)
        g_deg.to_csv(DATA_DIR / f"{csv_pref}_network_importance_ranking.csv", index=False)
        g_eigen.to_csv(DATA_DIR / f"{csv_pref}_centrality_trajectories_summary.csv", index=False)
        payload[label] = {
            "mean_expected_density": float(P.sum() / (P.shape[0] * P.shape[1] * (P.shape[1] - 1))),
            "mean_hard_density": float(A_hard.sum() / (A_hard.shape[0] * A_hard.shape[1] * (A_hard.shape[1] - 1))),
            "degree_top10": g_deg.head(10)["ticker"].tolist(),
            "eigen_top10": g_eigen.head(10)["ticker"].tolist(),
        }
        print(f"{label} degree top-10:", payload[label]["degree_top10"])
        print(f"{label} eigen top-10:", payload[label]["eigen_top10"])

    (DATA_DIR / "revised_ranking_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    dens = payload["density_path"]
    print(f"Rolling proxy density: start {dens[0]:.3f}  end {dens[-1]:.3f}  mean {payload['mean_density']:.3f}")
    print("Saved revised ranking CSVs and data/revised_ranking_summary.json")


if __name__ == "__main__":
    main()
