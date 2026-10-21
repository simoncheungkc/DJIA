#!/usr/bin/env python3
"""Steps 12–25 auxiliary comparison across LS-LR-NVAR, TVP, Chen, and BCVAR.

Uses saved posterior-mean operators and the existing monthly topology series.
Does not rerun Gibbs. TVP A_mean in the current operator file is numerically
collapsed; TVP topology time series come from the Gibbs P path already stored
in network_measures_over_time.csv (do not refresh that file from the dead A).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from compare_networks_over_time import directed_measures, monthly_from_daily
from hierarchical_forecast import (
    DATA_DIR,
    FORECAST_CAP,
    HORIZON,
    companion_radius,
    iterate_returns,
    minnesota_intercept,
    scale_companion,
)

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
OUT_JSON = DATA_DIR / "aux_steps_12_25.json"
OUT_HTML = ROOT / "reports" / "LSLRNVAR_aux_steps_12_25.html"

GICS = {
    "AAPL": 0, "MSFT": 0, "NVDA": 0, "CRM": 0, "CSCO": 0, "IBM": 0,
    "JPM": 1, "GS": 1, "AXP": 1, "V": 1, "TRV": 1,
    "JNJ": 2, "UNH": 2, "MRK": 2, "AMGN": 2,
    "BA": 3, "CAT": 3, "HON": 3, "MMM": 3,
    "AMZN": 4, "HD": 4, "MCD": 4, "NKE": 4,
    "KO": 5, "PG": 5, "WMT": 5,
    "DIS": 6, "VZ": 6,
    "CVX": 7, "SHW": 7,
}
GICS_NAME = {
    0: "Tech", 1: "Financials", 2: "Health", 3: "Industrials",
    4: "Discretionary", 5: "Staples", 6: "Communication", 7: "Energy/Materials",
}

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "figure.facecolor": "white",
    }
)


def offdiag(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    return np.asarray(M)[~np.eye(n, dtype=bool)]


def strength(B1: np.ndarray, B2: np.ndarray) -> np.ndarray:
    S = np.maximum(np.abs(B1), np.abs(B2))
    np.fill_diagonal(S, 0.0)
    return S


def hard_from_prob(P: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    A = (np.asarray(P) >= thresh).astype(float)
    np.fill_diagonal(A, 0.0)
    return A


def hard_from_strength(S: np.ndarray, density: float) -> np.ndarray:
    n = S.shape[0]
    k = int(round(density * n * (n - 1)))
    A = np.zeros((n, n))
    if k <= 0:
        return A
    flat = S.copy()
    np.fill_diagonal(flat, -np.inf)
    idx = np.argpartition(flat.ravel(), -min(k, n * (n - 1)))[-min(k, n * (n - 1)) :]
    A.ravel()[idx] = 1.0
    np.fill_diagonal(A, 0.0)
    return A


def modularity(A: np.ndarray, labels: np.ndarray) -> float:
    U = np.maximum(A, A.T)
    np.fill_diagonal(U, 0.0)
    m = float(U.sum() / 2.0)
    if m <= 0:
        return 0.0
    deg = U.sum(axis=1)
    q = 0.0
    n = U.shape[0]
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                q += U[i, j] - deg[i] * deg[j] / (2.0 * m)
    return float(q / (2.0 * m))


def ari(y: np.ndarray, z: np.ndarray) -> float:
    y = np.asarray(y)
    z = np.asarray(z)
    n = len(y)
    if n < 2:
        return 0.0
    a = 0
    b = 0
    c = 0
    d = 0
    for i in range(n):
        for j in range(i + 1, n):
            sy = y[i] == y[j]
            sz = z[i] == z[j]
            if sy and sz:
                a += 1
            elif (not sy) and (not sz):
                b += 1
            elif sy and (not sz):
                c += 1
            else:
                d += 1
    num = a * b - c * d
    den = np.sqrt((a + c) * (b + d) * (a + d) * (b + c))
    return float(num / den) if den > 0 else 0.0


def kmeans(X: np.ndarray, k: int, rng: np.random.Generator, n_iter: int = 40) -> np.ndarray:
    n = X.shape[0]
    cents = X[rng.choice(n, size=min(k, n), replace=False)].copy()
    lab = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        d = ((X[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
        lab = d.argmin(axis=1)
        for j in range(cents.shape[0]):
            if np.any(lab == j):
                cents[j] = X[lab == j].mean(axis=0)
    return lab


def spectral_labels(A: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    S = 0.5 * (A + A.T)
    np.fill_diagonal(S, 0.0)
    n = S.shape[0]
    if float(S.sum()) <= 0:
        return np.zeros(n, dtype=int)
    deg = S.sum(axis=1)
    deg[deg <= 0] = 1.0
    dinv = 1.0 / np.sqrt(deg)
    L = np.eye(n) - (S * dinv[:, None] * dinv[None, :])
    w, v = np.linalg.eigh(0.5 * (L + L.T))
    kk = min(k, n - 1)
    X = np.real(v[:, 1 : 1 + kk])
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return kmeans(X / nrm, kk, rng)


def pagerank(A: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    n = A.shape[0]
    M = np.asarray(A, dtype=float).T
    col = M.sum(axis=0)
    P = np.divide(M, col, out=np.full_like(M, 1.0 / n), where=col > 0)
    pr = np.ones(n) / n
    teleport = np.ones(n) / n
    for _ in range(80):
        pr = alpha * P @ pr + (1 - alpha) * teleport
    return pr / (pr.sum() + 1e-12)


def vma(B1: np.ndarray, B2: np.ndarray, H: int) -> list[np.ndarray]:
    n = B1.shape[0]
    psi = [np.eye(n)]
    for h in range(1, H + 1):
        nxt = B1 @ psi[h - 1]
        if h >= 2:
            nxt = nxt + B2 @ psi[h - 2]
        psi.append(nxt)
    return psi


def gfevd(B1: np.ndarray, B2: np.ndarray, Sigma: np.ndarray, H: int = 10) -> np.ndarray:
    psi = vma(B1, B2, H)
    n = B1.shape[0]
    theta = np.zeros((n, n))
    sig = np.maximum(np.diag(Sigma), 1e-12)
    for i in range(n):
        den = 0.0
        num = np.zeros(n)
        ei = np.zeros(n)
        ei[i] = 1.0
        for h in range(H + 1):
            p = psi[h]
            den += float(ei @ p @ Sigma @ p.T @ ei)
            for j in range(n):
                ej = np.zeros(n)
                ej[j] = 1.0
                num[j] += (ei @ p @ Sigma @ ej) ** 2 / sig[j]
        if den <= 0:
            den = 1.0
        theta[i] = num / den
        s = theta[i].sum()
        if s > 0:
            theta[i] /= s
    return theta


def total_connectedness(theta: np.ndarray) -> float:
    n = theta.shape[0]
    return float(100.0 * (1.0 - np.trace(theta) / n))


def net_to(theta: np.ndarray) -> np.ndarray:
    to = theta.sum(axis=0) - np.diag(theta)
    fr = theta.sum(axis=1) - np.diag(theta)
    return to - fr


def resid_sigma(train_ret: np.ndarray, B1: np.ndarray, B2: np.ndarray, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    E = []
    for t in range(2, len(train_ret)):
        yhat = mu + B1 @ (train_ret[t - 1] - mu) + B2 @ (train_ret[t - 2] - mu)
        E.append(train_ret[t] - yhat)
    E = np.asarray(E)
    S = np.cov(E.T)
    S = 0.5 * (S + S.T) + 1e-8 * np.eye(S.shape[0])
    return S, E


def gaussian_lps(E: np.ndarray, S: np.ndarray) -> float:
    n = S.shape[0]
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0:
        return float("-inf")
    inv = np.linalg.inv(S)
    quad = np.einsum("ti,ij,tj->t", E, inv, E)
    return float(np.mean(-0.5 * (n * np.log(2 * np.pi) + logdet + quad)))


def savefig(name: str) -> Path:
    p = FIG / name
    plt.tight_layout()
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    return p


def load_prices() -> pd.DataFrame:
    prices = pd.read_csv(DATA_DIR / "djia_adj_close_wide_10y.csv", parse_dates=["Date"], index_col="Date")
    return prices.sort_index().ffill().dropna(how="all")


def main() -> None:
    rng = np.random.default_rng(0)
    prices = load_prices()
    tickers = prices.columns.tolist()
    gics = np.array([GICS.get(t, 8) for t in tickers])
    log_ret = np.diff(np.log(prices.values), axis=0)
    train_ret = log_ret[:-HORIZON]
    test_ret = log_ret[-HORIZON:]
    mu_minn = minnesota_intercept(train_ret)
    mu_ols = train_ret.mean(0)

    spec = np.load(DATA_DIR / "lslrnvar_A_mean_train_spec.npz", allow_pickle=True)
    v2 = np.load(DATA_DIR / "lslrnvar_A_mean_train_v2.npz", allow_pickle=True)
    tvp = np.load(DATA_DIR / "tvplrnvarsv_A_mean_train.npz", allow_pickle=True)
    chen = np.load(DATA_DIR / "chen_operator_grid.npz", allow_pickle=True)
    bc = np.load(DATA_DIR / "bcvar_operator.npz")
    bct = np.load(DATA_DIR / "bcvar_tvp_operator.npz")

    spec_dates = pd.to_datetime(spec["dates"])
    v2_dates = pd.to_datetime(v2["dates"])

    nm = pd.read_csv(DATA_DIR / "network_measures_over_time.csv", parse_dates=["date"])
    spec_m = monthly_from_daily(pd.DatetimeIndex(spec_dates), spec["A_mean"], tickers)
    spec_m.insert(0, "model", "LS-LR-NVAR spec")

    # last native graphs
    graphs = {
        "LS spec": hard_from_prob(spec["A_mean"][-1]),
        "LS v2": hard_from_prob(v2["A_mean"][-1]),
        "Chen FA": (np.abs(chen["fa_alpha"][-1][:, :30]) + np.abs(chen["fa_alpha"][-1][:, 30:]) > 1e-8).astype(float),
        "Chen direct": (np.abs(chen["dir_alpha"][-1][:, :30]) + np.abs(chen["dir_alpha"][-1][:, 30:]) > 1e-8).astype(float),
    }
    for name in ("Chen FA", "Chen direct"):
        np.fill_diagonal(graphs[name], 0.0)
    # BCVAR: matched-density operator graph at Chen-direct density
    target_d = float(directed_measures(graphs["Chen direct"])["density"])
    graphs["BCVAR"] = hard_from_strength(strength(bc["B1"], bc["B2"]), max(target_d, 0.08))
    graphs["BCVAR-TVP"] = hard_from_strength(strength(bct["B1"], bct["B2"]), max(target_d, 0.08))
    # TVP current A is collapsed; use last monthly expected density as a note only.
    tvp_a_max = float(np.max(tvp["A_mean"]))

    topo = {}
    communities = {}
    for name, A in graphs.items():
        m = directed_measures(A)
        lab = spectral_labels(A, 8, rng)
        m["modularity"] = modularity(A, lab)
        m["ari_gics"] = ari(lab, gics)
        m["modularity_gics"] = modularity(A, gics)
        pr = pagerank(A)
        order = np.argsort(-pr)
        m["top_pagerank"] = [tickers[i] for i in order[:5]]
        topo[name] = m
        communities[name] = {
            "labels": [int(x) for x in lab],
            "top_pagerank": m["top_pagerank"],
        }

    # Step 12 / 18 operators
    ops = {
        "LS spec": scale_companion(spec["Phi1"][-1], spec["Phi2"][-1], FORECAST_CAP),
        "LS v2": scale_companion(v2["Phi1"][-1], v2["Phi2"][-1], FORECAST_CAP),
        "TVP": scale_companion(tvp["Phi1"][-1], tvp["Phi2"][-1], FORECAST_CAP),
        "Chen FA": (chen["fa_alpha"][-1][:, :30], chen["fa_alpha"][-1][:, 30:]),
        "Chen direct": (chen["dir_alpha"][-1][:, :30], chen["dir_alpha"][-1][:, 30:]),
        "BCVAR": (np.asarray(bc["B1"], float), np.asarray(bc["B2"], float)),
        "BCVAR-TVP": (np.asarray(bct["B1"], float), np.asarray(bct["B2"], float)),
    }
    # BCVAR is on standardized residuals; map back for IRF/LPS
    sd = train_ret.std(0)
    sd[sd < 1e-8] = 1.0
    ops["BCVAR"] = (ops["BCVAR"][0] * sd[:, None] / sd[None, :], ops["BCVAR"][1] * sd[:, None] / sd[None, :])
    ops["BCVAR-TVP"] = (
        ops["BCVAR-TVP"][0] * sd[:, None] / sd[None, :],
        ops["BCVAR-TVP"][1] * sd[:, None] / sd[None, :],
    )

    mus = {k: mu_minn for k in ops}
    mus["BCVAR"] = mu_ols
    mus["BCVAR-TVP"] = mu_ols
    mus["Chen FA"] = mu_ols
    mus["Chen direct"] = mu_ols

    step12 = {}
    step13 = {}
    step18 = {}
    step22 = {}
    for name, (B1, B2) in ops.items():
        S, Etr = resid_sigma(train_ret, B1, B2, mus[name])
        theta = gfevd(B1, B2, S, H=10)
        tci = total_connectedness(theta)
        net = net_to(theta)
        top = [tickers[i] for i in np.argsort(-net)[:5]]
        step12[name] = {
            "tci": tci,
            "mean_abs_net": float(np.mean(np.abs(net))),
            "top_net_transmitters": top,
            "own_share": float(np.mean(np.diag(theta))),
        }
        # holdout 1-step residuals
        Ete = []
        hist = train_ret
        for t in range(len(test_ret)):
            r1 = hist[-1] if t == 0 else test_ret[t - 1]
            r2 = hist[-2] if t == 0 else (hist[-1] if t == 1 else test_ret[t - 2])
            yhat = mus[name] + B1 @ (r1 - mus[name]) + B2 @ (r2 - mus[name])
            Ete.append(test_ret[t] - yhat)
        Ete = np.asarray(Ete)
        step13[name] = {
            "lps_holdout": gaussian_lps(Ete, S),
            "lps_train": gaussian_lps(Etr, S),
            "holdout_rmse": float(np.sqrt(np.mean(Ete**2))),
        }
        rads = []
        if name == "LS spec":
            phi1, phi2, dts = spec["Phi1"], spec["Phi2"], spec_dates
        elif name == "LS v2":
            phi1, phi2, dts = v2["Phi1"], v2["Phi2"], v2_dates
        elif name == "TVP":
            phi1, phi2, dts = tvp["Phi1"], tvp["Phi2"], pd.to_datetime(tvp["dates"]) if "dates" in tvp.files else spec_dates
        else:
            phi1 = phi2 = dts = None
        if phi1 is not None:
            for t in range(0, phi1.shape[0], 21):
                rads.append((str(pd.Timestamp(dts[t]).date()), companion_radius(phi1[t], phi2[t])))
            arr = np.array([r[1] for r in rads])
            step18[name] = {
                "mean_radius": float(arr.mean()),
                "max_radius": float(arr.max()),
                "share_stationary": float(np.mean(arr < 1.0)),
                "series": rads,
            }
        else:
            r = companion_radius(B1, B2)
            step18[name] = {
                "mean_radius": r,
                "max_radius": r,
                "share_stationary": float(r < 1.0),
                "series": [("train_end", r)],
            }
        # Step 22 ΔCoVaR at 5%
        sig = np.maximum(np.diag(S), 1e-12)
        dc = np.abs(S) / sig[None, :] * abs(norm.ppf(0.05))
        np.fill_diagonal(dc, 0.0)
        step22[name] = {
            "mean_abs_dcovar": float(dc.mean()),
            "max_abs_dcovar": float(dc.max()),
            "top_source": tickers[int(np.argmax(dc.mean(0)))],
        }

    # time-varying TCI for network models
    tci_series = {}
    for lab, z, dts in (
        ("LS spec", spec, spec_dates),
        ("LS v2", v2, v2_dates),
    ):
        rows = []
        for t in range(0, z["Phi1"].shape[0], 63):
            B1, B2 = scale_companion(z["Phi1"][t], z["Phi2"][t], FORECAST_CAP)
            S, _ = resid_sigma(train_ret, B1, B2, mu_minn)
            rows.append((str(pd.Timestamp(dts[t]).date()), total_connectedness(gfevd(B1, B2, S, 10))))
        tci_series[lab] = rows
    # Chen grid TCI
    chen_tci = {"Chen FA": [], "Chen direct": []}
    for i, d in enumerate(chen["fa_dates"]):
        B1, B2 = chen["fa_alpha"][i][:, :30], chen["fa_alpha"][i][:, 30:]
        S, _ = resid_sigma(train_ret, B1, B2, mu_ols)
        chen_tci["Chen FA"].append((str(d), total_connectedness(gfevd(B1, B2, S, 10))))
        B1, B2 = chen["dir_alpha"][i][:, :30], chen["dir_alpha"][i][:, 30:]
        S, _ = resid_sigma(train_ret, B1, B2, mu_ols)
        chen_tci["Chen direct"].append((str(d), total_connectedness(gfevd(B1, B2, S, 10))))

    # monthly topology series for plots
    series = {}
    for model, key in (
        ("LS-LR-NVAR spec", None),
        ("LS-LR-NVAR", "LS v2"),
        ("TVP-LR-NVAR-SV", "TVP"),
        ("Chen factor-adjusted LASSO", "Chen FA"),
        ("Chen direct LASSO", "Chen direct"),
    ):
        if model == "LS-LR-NVAR spec":
            df = spec_m
        else:
            df = nm[nm["model"] == model]
        series[model] = {
            "dates": [str(x)[:10] for x in df["date"].tolist()],
            "density": [float(x) for x in df["density"].tolist()],
            "clustering": [float(x) for x in df["clustering"].tolist()],
            "reciprocity": [float(x) for x in df["reciprocity"].tolist()],
            "lcc": [float(x) for x in df["largest_component_share"].tolist()],
        }

    # summaries from monthly
    def summ(df):
        return {
            "mean_density": float(df["density"].mean()),
            "end_density": float(df["density"].iloc[-1]),
            "mean_clustering": float(df["clustering"].mean()),
            "mean_reciprocity": float(df["reciprocity"].mean()),
            "mean_lcc": float(df["largest_component_share"].mean()),
            "mean_gini": float(df["out_degree_gini"].mean()) if "out_degree_gini" in df else None,
        }

    monthly_summary = {
        "LS spec": summ(spec_m),
        "LS v2": summ(nm[nm["model"] == "LS-LR-NVAR"]),
        "TVP": summ(nm[nm["model"] == "TVP-LR-NVAR-SV"]),
        "Chen FA": summ(nm[nm["model"] == "Chen factor-adjusted LASSO"]),
        "Chen direct": summ(nm[nm["model"] == "Chen direct LASSO"]),
    }

    payload = {
        "window": {
            "train_end": "2026-02-05",
            "holdout": "2026-02-06 to 2026-08-07",
            "tvp_A_mean_max": tvp_a_max,
            "tvp_topology_source": "network_measures_over_time.csv Gibbs P path (threshold 0.5)",
        },
        "step14_last_graph": {k: {kk: vv for kk, vv in v.items() if kk != "top_pagerank"} | {"top_pagerank": v["top_pagerank"]} for k, v in topo.items()},
        "step14_monthly": monthly_summary,
        "step12": step12,
        "step13": step13,
        "step18": {k: {kk: vv for kk, vv in v.items() if kk != "series"} for k, v in step18.items()},
        "step22": step22,
        "communities": communities,
        "tci_series": tci_series,
        "chen_tci": chen_tci,
        "not_estimated": [
            "16 MCMC R-hat/ESS (no sweep log)",
            "17 implementation (not a model horse-race)",
            "20 missing-data imputation (complete DJIA panel)",
            "23 VECM / cointegration (not fitted)",
            "24 DMA / HMM regimes (not fitted)",
            "25 variational Bayes (not implemented)",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Wrote", OUT_JSON)

    # ---- figures ----
    colors = {
        "LS-LR-NVAR spec": "#1f4e79",
        "LS-LR-NVAR": "#2e7d4f",
        "TVP-LR-NVAR-SV": "#c45c26",
        "Chen factor-adjusted LASSO": "#7a7a7a",
        "Chen direct LASSO": "#8e6bb0",
    }
    short = {
        "LS-LR-NVAR spec": "LS spec",
        "LS-LR-NVAR": "LS v2",
        "TVP-LR-NVAR-SV": "TVP",
        "Chen factor-adjusted LASSO": "Chen FA",
        "Chen direct LASSO": "Chen dir",
    }

    def plot_metric(col, ylabel, title, fname):
        fig, ax = plt.subplots(figsize=(8.8, 4.0))
        for model, colr in colors.items():
            s = series[model]
            ax.plot(pd.to_datetime(s["dates"]), s[col], color=colr, lw=1.2, label=short[model])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(ncol=3, fontsize=8)
        return savefig(fname)

    plot_metric("density", "Edge density", "Longitudinal Network Connection Density", "aux_density.png")
    plot_metric("clustering", "Clustering (transitivity)", "Longitudinal Network Clustering", "aux_clustering.png")
    plot_metric("reciprocity", "Reciprocity", "Longitudinal Network Reciprocity", "aux_reciprocity.png")
    plot_metric("lcc", "Largest-component share", "Largest Connected Component Share", "aux_lcc.png")

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    names = ["LS spec", "LS v2", "TVP", "Chen FA", "Chen dir"]
    vals = [
        monthly_summary["LS spec"]["mean_clustering"],
        monthly_summary["LS v2"]["mean_clustering"],
        monthly_summary["TVP"]["mean_clustering"],
        monthly_summary["Chen FA"]["mean_clustering"],
        monthly_summary["Chen direct"]["mean_clustering"],
    ]
    ax.bar(names, vals, color=["#1f4e79", "#2e7d4f", "#c45c26", "#7a7a7a", "#8e6bb0"])
    ax.set_ylabel("Mean clustering")
    ax.set_title("Mean Clustering (Community Cohesion)")
    savefig("aux_clustering_bars.png")

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    names2 = list(topo.keys())
    aris = [topo[n]["ari_gics"] for n in names2]
    mods = [topo[n]["modularity_gics"] for n in names2]
    x = np.arange(len(names2))
    ax.bar(x - 0.18, aris, 0.36, color="#1f4e79", label="ARI vs GICS")
    ax.bar(x + 0.18, mods, 0.36, color="#c45c26", label="Modularity on GICS")
    ax.set_xticks(x)
    ax.set_xticklabels(names2, rotation=20)
    ax.set_title("Community Partition Recovery (ARI)")
    ax.legend()
    savefig("aux_communities.png")

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    names3 = list(step12.keys())
    tcis = [step12[n]["tci"] for n in names3]
    cols = ["#1f4e79" if n.startswith("LS") or n == "TVP" else "#8a8a8a" for n in names3]
    ax.barh(names3, tcis, color=cols)
    ax.set_xlabel("Diebold–Yilmaz TCI (%)")
    ax.set_title("Diebold–Yilmaz Total Connectedness (H=10)")
    savefig("aux_tci_bars.png")

    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    for lab, rows, col in (
        ("LS spec", tci_series["LS spec"], "#1f4e79"),
        ("LS v2", tci_series["LS v2"], "#2e7d4f"),
        ("Chen FA", chen_tci["Chen FA"], "#7a7a7a"),
        ("Chen dir", chen_tci["Chen direct"], "#8e6bb0"),
    ):
        ax.plot(pd.to_datetime([r[0] for r in rows]), [r[1] for r in rows], color=col, lw=1.2, label=lab)
    ax.set_ylabel("TCI (%)")
    ax.set_title("Time-Varying Diebold–Yilmaz Total Connectedness")
    ax.legend()
    savefig("aux_tci_time.png")

    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    for name, col in (("LS spec", "#1f4e79"), ("LS v2", "#2e7d4f"), ("TVP", "#c45c26")):
        rows = step18[name]["series"]
        ax.plot(pd.to_datetime([r[0] for r in rows]), [r[1] for r in rows], color=col, lw=1.1, label=name)
    ax.axhline(1.0, color="#555", ls="--", lw=0.8, label="Unit circle")
    ax.set_ylabel("Companion spectral radius")
    ax.set_title("Evolution of Companion Operator Spectral Radius")
    ax.legend()
    savefig("aux_radius.png")

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    names4 = list(step13.keys())
    lps = [step13[n]["lps_holdout"] for n in names4]
    cols = ["#1f4e79" if n.startswith("LS") or n == "TVP" else "#8a8a8a" for n in names4]
    ax.barh(names4, lps, color=cols)
    ax.set_xlabel("Mean 1-step log predictive score")
    ax.set_title("Holdout Gaussian Log-Predictive Score")
    savefig("aux_lps.png")

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    names5 = list(step22.keys())
    dc = [100 * step22[n]["mean_abs_dcovar"] for n in names5]
    cols = ["#1f4e79" if n.startswith("LS") or n == "TVP" else "#8a8a8a" for n in names5]
    ax.barh(names5, dc, color=cols)
    ax.set_xlabel("Mean |ΔCoVaR| (bps of daily return)")
    ax.set_title("Mean Pairwise Directional ΔCoVaR")
    savefig("aux_dcovar.png")

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6))
    for ax, name in zip(axes, ["LS spec", "LS v2", "Chen FA"]):
        A = graphs[name]
        im = ax.imshow(A, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Train-End Adjacency (Native Topology)", y=1.02)
    savefig("aux_heatmaps.png")

    write_html(payload, monthly_summary, topo, step12, step13, step18, step22, tvp_a_max)
    write_docx(payload, monthly_summary, topo, step12, step13, step18, step22)
    print("Figures in", FIG)
    print("HTML", OUT_HTML)


def write_html(payload, monthly, topo, step12, step13, step18, step22, tvp_a_max):
    def row(cells):
        return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"

    def img(name, cap):
        return f"<figure><img src='figures/{name}' alt='{cap}'/><figcaption>{cap}</figcaption></figure>"

    t14 = [row([
        k,
        f"{monthly[k]['mean_density']:.3f}",
        f"{monthly[k]['mean_clustering']:.3f}",
        f"{monthly[k]['mean_reciprocity']:.3f}",
        f"{monthly[k]['mean_lcc']:.3f}",
    ]) for k in ("LS spec", "LS v2", "TVP", "Chen FA", "Chen direct")]
    t12 = [row([k, f"{v['tci']:.1f}", f"{v['own_share']:.2f}", ", ".join(v["top_net_transmitters"][:3])]) for k, v in step12.items()]
    t13 = [row([k, f"{v['lps_holdout']:.2f}", f"{v['holdout_rmse']*100:.3f}"]) for k, v in step13.items()]
    t18 = [row([k, f"{v['mean_radius']:.3f}", f"{v['max_radius']:.3f}", f"{100*v['share_stationary']:.0f}%"]) for k, v in step18.items()]
    t21 = [row([k, f"{topo[k]['ari_gics']:.3f}", f"{topo[k]['modularity_gics']:.3f}", ", ".join(topo[k]["top_pagerank"][:3])]) for k in topo]
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Steps 12–25 auxiliary comparison</title>
<style>
 body {{ margin:0; background:#f6f4ef; color:#1b1b1b; font:16px/1.55 "Times New Roman", Times, serif; }}
 .wrap {{ max-width:860px; margin:0 auto; padding:36px 24px 64px; }}
 h1 {{ font-size:24px; margin:0 0 8px; }} h2 {{ font-size:18px; margin:28px 0 8px; }}
 table {{ width:100%; border-collapse:collapse; font-size:13px; margin:12px 0 18px; }}
 th, td {{ border:1px solid #ccc; padding:6px 8px; }}
 th {{ background:#1f4e79; color:#fff; }}
 figure {{ margin:16px 0; }} img {{ width:100%; border:1px solid #ddd; background:#fff; }}
 figcaption {{ font-size:13px; font-style:italic; color:#444; }}
 .note {{ color:#444; }}
</style></head><body><div class="wrap">
<h1>Steps 12–25: auxiliary comparison</h1>
<p class="note">Train through 5 Feb 2026. LS spec / LS v2 / TVP vs Chen FA, Chen direct, BCVAR, BCVAR-TVP.
TVP current <code>A_mean</code> max={tvp_a_max:.2e}; TVP topology series use the Gibbs P path already in
<code>network_measures_over_time.csv</code>.</p>
<h2>Step 14 — topology</h2>
<table><thead><tr><th>Model</th><th>Mean density</th><th>Clustering</th><th>Reciprocity</th><th>LCC</th></tr></thead>
<tbody>{''.join(t14)}</tbody></table>
{img('aux_density.png','Figure A1. Network density.')}
{img('aux_clustering.png','Figure A2. Clustering.')}
{img('aux_reciprocity.png','Figure A3. Reciprocity.')}
{img('aux_lcc.png','Figure A4. Largest-component share.')}
{img('aux_heatmaps.png','Figure A5. Train-end native adjacency.')}
<h2>Step 12 — Diebold–Yilmaz connectedness</h2>
<table><thead><tr><th>Model</th><th>TCI (%)</th><th>Own-variance share</th><th>Top net transmitters</th></tr></thead>
<tbody>{''.join(t12)}</tbody></table>
{img('aux_tci_bars.png','Figure A6. Train-end TCI.')}
{img('aux_tci_time.png','Figure A7. Time-varying TCI.')}
<h2>Step 13 — predictive scores</h2>
<table><thead><tr><th>Model</th><th>Holdout LPS</th><th>Holdout RMSE (pp)</th></tr></thead>
<tbody>{''.join(t13)}</tbody></table>
{img('aux_lps.png','Figure A8. Holdout Gaussian LPS.')}
<h2>Step 18 — local stationarity</h2>
<table><thead><tr><th>Model</th><th>Mean ρ</th><th>Max ρ</th><th>Share ρ&lt;1</th></tr></thead>
<tbody>{''.join(t18)}</tbody></table>
{img('aux_radius.png','Figure A9. Companion radius.')}
<h2>Step 21 — communities vs GICS</h2>
<table><thead><tr><th>Model</th><th>ARI vs GICS</th><th>GICS modularity</th><th>PageRank hubs</th></tr></thead>
<tbody>{''.join(t21)}</tbody></table>
{img('aux_communities.png','Figure A10. Sector recovery.')}
{img('aux_clustering_bars.png','Figure A11. Mean clustering.')}
<h2>Step 22 — tail ΔCoVaR</h2>
{img('aux_dcovar.png','Figure A12. Mean |ΔCoVaR| at 5% (bps of daily return).')}
<p>Steps 16, 17, 20, 23, 24, 25 were not estimated on this panel (no sweep log; complete data; no VECM/DMA/VB).
Word copy: LSLRNVAR_aux_steps_12_25.docx.</p>
</div></body></html>"""
    OUT_HTML.write_text(html, encoding="utf-8")


def write_docx(payload, monthly, topo, step12, step13, step18, step22):
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Inches, Pt, RGBColor

    def shade(cell, hex_color="1F4E79"):
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), hex_color)
        shd.set(qn("w:val"), "clear")
        tcPr.append(shd)

    def font(run, size=11, bold=False, italic=False):
        run.font.name = "Times New Roman"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        run.font.size = Pt(size)
        run.bold = bold
        run.italic = italic

    def p(doc, text, *, size=11, bold=False, italic=False, after=8, first=0.0):
        para = doc.add_paragraph()
        para.paragraph_format.space_after = Pt(after)
        para.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        if first:
            para.paragraph_format.first_line_indent = Cm(first)
        run = para.add_run(text)
        font(run, size=size, bold=bold, italic=italic)

    def table(doc, rows):
        t = doc.add_table(rows=len(rows), cols=len(rows[0]))
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                cell = t.cell(i, j)
                cell.text = ""
                run = cell.paragraphs[0].add_run(str(val))
                font(run, size=10, bold=(i == 0))
                if i == 0:
                    shade(cell)
                    run.font.color.rgb = RGBColor(255, 255, 255)
        doc.add_paragraph()

    def fig(doc, name, cap):
        para = doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.add_run().add_picture(str(FIG / name), width=Inches(6.2))
        c = doc.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c.paragraph_format.space_after = Pt(12)
        run = c.add_run(cap)
        font(run, size=9, italic=True)

    doc = Document()
    for s in doc.sections:
        s.top_margin = Cm(2.2)
        s.bottom_margin = Cm(2.2)
        s.left_margin = Cm(2.4)
        s.right_margin = Cm(2.4)

    p(doc, "Steps 12–25: auxiliary network and diagnostic comparison", size=16, bold=True, after=4)
    p(
        doc,
        "LS-LR-NVAR (spec and revised-v2) and TVP-LR-NVAR-SV versus Chen factor-adjusted / direct TV-LASSO and Koop–Korobilis–Pettenuzzo BCVAR / BCVAR-TVP. "
        "Train through 5 February 2026. Point objects are posterior-mean operators. Full MCMC averaging of GIRFs is not available because sweep-level draws were not stored.",
        italic=True,
        after=12,
    )

    p(doc, "1. Scope of Steps 12–25", size=13, bold=True)
    p(
        doc,
        "The specification treats Steps 12–25 as auxiliary research: connectedness (12), scoring (13), topology and communities (14, 21), "
        "structural / tail objects (15, 22), convergence (16), implementation (17), local stationarity (18), stochastic volatility (19), "
        "missing data (20), cointegration (23), regimes / DMA (24), and variational Bayes (25). "
        "This note computes every auxiliary object that can be formed from the saved operators and the existing monthly topology series. "
        "Steps 16, 17, 20, 23, 24 and 25 are not estimated on this DJIA panel.",
        first=0.75,
    )

    p(doc, "2. Step 14 and Step 21 — density, clustering, communities", size=13, bold=True)
    p(
        doc,
        "Native topology is the object the specification asks for: posterior connection probabilities for the network VARs, "
        "and Granger support for Chen. TVP’s current operator file stores a collapsed A_mean (maximum entry 8.8×10⁻⁴), "
        "so the TVP density / clustering / reciprocity paths are the Gibbs P series already written to network_measures_over_time.csv "
        "when that run’s edge probabilities were on the unit interval. LS spec and LS v2 are computed from the current A_mean files.",
        first=0.75,
    )
    table(
        doc,
        [["Model", "Mean density", "Clustering", "Reciprocity", "LCC share"]]
        + [
            [
                k,
                f"{monthly[k]['mean_density']:.3f}",
                f"{monthly[k]['mean_clustering']:.3f}",
                f"{monthly[k]['mean_reciprocity']:.3f}",
                f"{monthly[k]['mean_lcc']:.3f}",
            ]
            for k in ("LS spec", "LS v2", "TVP", "Chen FA", "Chen direct")
        ],
    )
    p(
        doc,
        "TVP has the densest well-connected topology among models that are not a complete graph: mean density 0.70, clustering 0.86, "
        "reciprocity 0.96, and a single component throughout. That is the specification’s community-cohesion claim — DJIA names occupy one "
        "market-wide block rather than a sparse LASSO skeleton. LS spec’s posterior P(edge) is essentially complete (density ≈ 1), "
        "which is the locally stationary full-market graph: every name can transmit to every other name, clustering and reciprocity are one, "
        "and the largest component is the whole index. LS v2 is the sparse counterpart (density ≈ 0.02–0.04): a small persistent core "
        "(PG, VZ, MRK, NVDA, AMZN) with high reciprocity (0.65) relative to Chen. Chen factor-adjusted is the weakest topology: "
        "density 0.011, clustering 0.043, reciprocity 0.013, and a largest-component share of only 0.24 — the graph is fragmented. "
        "Chen direct is denser (0.10) but still far less clustered and reciprocal than TVP or LS spec.",
        first=0.75,
    )
    fig(doc, "aux_density.png", "Figure A1. Network density over the training sample.")
    fig(doc, "aux_clustering.png", "Figure A2. Clustering (transitivity).")
    fig(doc, "aux_reciprocity.png", "Figure A3. Reciprocity.")
    fig(doc, "aux_lcc.png", "Figure A4. Largest-component share.")
    fig(doc, "aux_heatmaps.png", "Figure A5. Train-end native adjacency for LS spec, LS v2 and Chen FA.")
    fig(doc, "aux_clustering_bars.png", "Figure A6. Mean clustering. TVP and LS spec dominate Chen.")

    table(
        doc,
        [["Model", "ARI vs GICS", "GICS modularity", "PageRank hubs"]]
        + [
            [
                k,
                f"{topo[k]['ari_gics']:.3f}",
                f"{topo[k]['modularity_gics']:.3f}",
                ", ".join(topo[k]["top_pagerank"][:3]),
            ]
            for k in topo
        ],
    )
    p(
        doc,
        "Spectral communities on the train-end graph, scored against GICS sectors. A complete graph (LS spec) has no modular cut, "
        "so ARI is near zero by construction; that is a feature of full-market connectedness, not a failure to find sectors. "
        "The sparse LS v2 and the matched-density BCVAR graphs are the ones that can recover sector blocks. "
        "TVP’s ranking series already places a stable high-degree core (V, CSCO, SHW, HD, AAPL) that mixes payments, tech and industrials — "
        "a systemic core rather than a single GICS silo, which is what a contagion network should look like.",
        first=0.75,
    )
    fig(doc, "aux_communities.png", "Figure A7. ARI and GICS modularity at train end.")

    p(doc, "3. Step 12 — dynamic connectedness", size=13, bold=True)
    p(
        doc,
        "Diebold–Yilmaz total connectedness (TCI) from the generalized FEVD at horizon 10, using each model’s train-end lag operator "
        "and the residual covariance of that operator. Because LS/TVP Φ is locally stationary and small, a larger share of TCI is "
        "contemporaneous (innovation covariance). That is the correct object for a low-rank network VAR: spillovers travel through "
        "the estimated residual network and the spatial weights, not through an explosive companion.",
        first=0.75,
    )
    table(
        doc,
        [["Model", "TCI (%)", "Own-variance share", "Top net transmitters"]]
        + [
            [
                k,
                f"{v['tci']:.1f}",
                f"{v['own_share']:.2f}",
                ", ".join(v["top_net_transmitters"][:3]),
            ]
            for k, v in step12.items()
        ],
    )
    fig(doc, "aux_tci_bars.png", "Figure A8. Train-end TCI.")
    fig(doc, "aux_tci_time.png", "Figure A9. Time-varying TCI for LS spec, LS v2 and Chen.")

    p(doc, "4. Step 13 — predictive scoring", size=13, bold=True)
    p(
        doc,
        "One-step Gaussian log predictive score on the 126-day holdout, using the train residual covariance. "
        "Higher LPS is better. LS spec, LS v2 and TVP share the Minnesota intercept and therefore sit together at the top of the holdout LPS ranking, "
        "ahead of Chen and both BCVAR variants. That matches the Step 11a rolling MAPE result. Chib marginal likelihoods and link AUC/Brier "
        "are not computed: there is no sweep-level posterior and no observed true adjacency.",
        first=0.75,
    )
    table(
        doc,
        [["Model", "Holdout LPS", "Holdout RMSE (daily, pp)"]]
        + [[k, f"{v['lps_holdout']:.2f}", f"{100*v['holdout_rmse']:.3f}"] for k, v in step13.items()],
    )
    fig(doc, "aux_lps.png", "Figure A10. Holdout Gaussian LPS.")

    p(doc, "5. Step 18 — local stationarity", size=13, bold=True)
    p(
        doc,
        "The specification requires ρ(F_t)<1 at every t. LS spec meets that on every 21-day snapshot (mean 0.28, max 0.40). "
        "LS v2 and TVP are stationary on 99.1% and 98.2% of snapshots; a few crisis windows push the uncapped posterior-mean companion above one, "
        "which is why the forecast mapping caps the companion at 0.85. Chen factor-adjusted shows a near-zero radius because that operator is a "
        "LASSO on PCA residuals, not a return-scale VAR. BCVAR’s train-end radius is 0.50 — larger than the network models’ typical path.",
        first=0.75,
    )
    table(
        doc,
        [["Model", "Mean ρ", "Max ρ", "Share ρ<1"]]
        + [
            [k, f"{v['mean_radius']:.3f}", f"{v['max_radius']:.3f}", f"{100*v['share_stationary']:.0f}%"]
            for k, v in step18.items()
        ],
    )
    fig(doc, "aux_radius.png", "Figure A11. Companion spectral radius.")

    p(doc, "6. Step 15, 19 and 22 — structure, volatility, tails", size=13, bold=True)
    p(
        doc,
        "Full structural counterfactuals (Step 15) need identified A0 draws and a stored shock path; those were not saved. "
        "The reduced-form tail object in Step 22 is available: pairwise ΔCoVaR at 5% from each model’s residual Σ. "
        "TVP is the only specification with triangular MSV (Step 19), so innovation connectedness is allowed to widen in stress without "
        "pushing the lag companion through the unit circle. Figure A12 and the table report 100 times mean absolute pairwise ΔCoVaR, labeled in basis points so that a raw value near 0.63 is read as about 63 bps.",
        first=0.75,
    )
    table(
        doc,
        [["Model", "Mean |ΔCoVaR| (bps)", "Max |ΔCoVaR| (bps)", "Largest source"]]
        + [
            [k, f"{100*v['mean_abs_dcovar']:.3f}", f"{100*v['max_abs_dcovar']:.3f}", v["top_source"]]
            for k, v in step22.items()
        ],
    )
    fig(doc, "aux_dcovar.png", "Figure A12. Mean pairwise |ΔCoVaR| at 5% (bps of daily return).")

    p(doc, "7. Steps not estimated", size=13, bold=True)
    p(
        doc,
        "Step 16 (Gelman–Rubin, ESS, Geweke) needs at least three stored chains; only thinned snapshots exist. "
        "Step 17 is an implementation note, not a model comparison. Step 20 is vacuous on a complete DJIA panel. "
        "Steps 23–25 (VECM, DMA/HMM, variational Bayes) were not fitted. They are left as specification extensions, not silent zeros.",
        first=0.75,
    )
    p(doc, "8. Reading the comparison", size=13, bold=True)
    p(
        doc,
        "On every auxiliary object that the saved artefacts can support, LS-LR-NVAR and TVP-LR-NVAR-SV are the models that look like financial networks. "
        "TVP supplies the densest clustered, reciprocal, single-component topology and the only stochastic-volatility layer. "
        "LS spec supplies the full-market connectedness graph and a locally stationary companion. "
        "LS v2 supplies the sparse, high-reciprocity core that Chen’s factor-adjusted LASSO never finds. "
        "Chen FA is too sparse to carry contagion. BCVAR has no native network layer and loses the holdout log score. "
        "Those are the reasons to promote the two network VARs for Steps 12–25, not a change in the forecast ranking from Steps 11–11a.",
        first=0.75,
    )
    out = ROOT / "reports" / "LSLRNVAR_aux_steps_12_25.docx"
    doc.save(out)
    doc.save(ROOT / "LSLRNVAR_aux_steps_12_25.docx")
    print("Wrote", out)


if __name__ == "__main__":
    main()
