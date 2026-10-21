#!/usr/bin/env python3
"""
Chen, Li, Li and Linton (Journal of Econometrics 249, 2025, 105941)

Time-varying network VAR for high-dimensional locally stationary series.

Implemented from Sections 3 and 5 for *mean prediction* at the forecast origin:
  (i)  local-linear time-varying LASSO  (eqs. 3.3--3.4)
  (ii) one-step SCAD-weighted (adaptive) LASSO using Stage (i)  (eqs. 3.5, 3.7)
  (iii) Section 5 factor adjustment: PCA on ZZ' then sparse TV-VAR on idiosyncratics

Coefficients are evaluated at tau = 1 (end of training) and iterated 126 steps.
CLIME (Section 3.3) estimates contemporaneous networks and is not required for
the mean path.

Holdout: train through 2026-02-05, evaluate 2026-02-06 to 2026-08-07.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LassoCV

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent / "data"
HORIZON = 126
LAG_P = 2
BANDWIDTH = 0.20
N_FACTORS = 3
LASSO_CV = 5
SCAD_A0 = 3.7
RANDOM_STATE = 0


def epanechnikov(u: np.ndarray) -> np.ndarray:
    a = np.abs(u)
    return np.where(a <= 1.0, 0.75 * (1.0 - u**2), 0.0)


def scad_prime(z: np.ndarray, lam: float, a: float = SCAD_A0) -> np.ndarray:
    z = np.abs(z)
    out = np.full_like(z, lam, dtype=float)
    mid = (z > lam) & (z <= a * lam)
    out[mid] = (a * lam - z[mid]) / ((a - 1.0) * lam) * lam
    out[z > a * lam] = 0.0
    return np.maximum(out, 1e-8)


def lagged_design(Y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T, _ = Y.shape
    Z, y, idx = [], [], []
    for t in range(p, T):
        Z.append(np.concatenate([Y[t - k] for k in range(1, p + 1)]))
        y.append(Y[t])
        idx.append(t)
    return np.asarray(Z), np.asarray(y), np.asarray(idx)


def _local_window(Y, p, tau, h, n_obs):
    Z, y, t_idx = lagged_design(Y, p)
    tau_t = (t_idx + 1) / float(n_obs)
    w = epanechnikov((tau_t - tau) / h)
    keep = w > 1e-8
    if keep.sum() < max(40, 2 * (1 + 2 * p * Y.shape[1])):
        w = epanechnikov((tau_t - tau) / (h * 1.5))
        keep = w > 1e-8
    return Z[keep], y[keep], w[keep], tau_t[keep]


def local_linear_stage(
    Y: np.ndarray,
    p: int,
    tau: float,
    h: float,
    n_obs: int,
    lasso: bool,
    adaptive_w: np.ndarray | None = None,
    lam_fixed: float | None = None,
    lam_vec: np.ndarray | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Local linear (Fan-Gijbels) at tau.

    Features follow (3.3)--(3.4): intercept + delta, lags Z, and (delta/h)*Z
    so that an L1 penalty on (alpha, gamma) matches |alpha|_1 + h|beta|_1
    with gamma = h beta.

    Returns intercept c (N,), alpha A (N, pN), and the raw alpha used for SCAD weights.
    """
    T, N = Y.shape
    Z, y, w, tau_t = _local_window(Y, p, tau, h, n_obs)
    delta = (tau_t - tau).reshape(-1, 1)
    n_lags = p * N
    # [delta, Z, Z * delta / h]  -> coef [c', alpha, gamma], beta = gamma / h
    feat = np.hstack([delta, Z, Z * delta / h])
    A = np.zeros((N, n_lags))
    c = np.zeros(N)
    alpha_raw = np.zeros((N, n_lags))
    lams = np.zeros(N)

    for i in range(N):
        yi = y[:, i]
        X = feat
        if adaptive_w is not None:
            aw = np.ones(feat.shape[1])
            aw[1 : 1 + n_lags] = adaptive_w[i]
            aw[1 + n_lags : 1 + 2 * n_lags] = adaptive_w[i]
            X = feat / aw[None, :]
        if lasso:
            eq_lam = None if lam_vec is None else float(lam_vec[i])
            if lam_fixed is None and eq_lam is None:
                model = LassoCV(
                    cv=LASSO_CV,
                    alphas=40,
                    max_iter=8000,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
                model.fit(X, yi, sample_weight=w)
                coef = model.coef_
                intercept = float(model.intercept_)
                chosen_lam = float(model.alpha_)
            else:
                model = Lasso(
                    alpha=lam_fixed if eq_lam is None else eq_lam,
                    max_iter=8000,
                    random_state=RANDOM_STATE,
                )
                model.fit(X, yi, sample_weight=w)
                coef = model.coef_
                intercept = float(model.intercept_)
                chosen_lam = lam_fixed
            if adaptive_w is not None:
                coef = coef / aw
            lams[i] = float(eq_lam if eq_lam is not None else chosen_lam)
        else:
            sw = np.sqrt(w)
            Xw = np.hstack([np.ones((feat.shape[0], 1)), feat]) * sw[:, None]
            yw = yi * sw
            beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
            intercept = float(beta[0])
            coef = beta[1:]
        c[i] = intercept
        alpha = coef[1 : 1 + n_lags]
        A[i] = alpha
        alpha_raw[i] = alpha
        if verbose and ((i + 1) % 10 == 0 or i == 0):
            tag = "stage2" if adaptive_w is not None else ("lasso" if lasso else "ols")
            print(f"    {tag} equation {i + 1}/{N}")
    return c, A, alpha_raw, lams


def two_stage_tv_var(Y: np.ndarray, p: int, h_bw: float) -> tuple[np.ndarray, np.ndarray]:
    """Stage (i) TV-LASSO then Stage (ii) SCAD-weighted LASSO, both at tau=1."""
    T = Y.shape[0]
    print("  Stage 1: local-linear TV-LASSO at tau=1")
    c1, A1, alpha1, _ = local_linear_stage(Y, p, tau=1.0, h=h_bw, n_obs=T, lasso=True)
    # lambda2 from typical Stage-1 coefficient scale
    lam2 = float(np.median(np.abs(alpha1)[np.abs(alpha1) > 1e-12]))
    if not np.isfinite(lam2) or lam2 <= 0:
        lam2 = 1e-3
    aw = scad_prime(alpha1, lam2)
    print("  Stage 2: SCAD-weighted local-linear LASSO at tau=1")
    c2, A2, _, _ = local_linear_stage(
        Y, p, tau=1.0, h=h_bw, n_obs=T, lasso=True, adaptive_w=aw
    )
    return c2, A2


def iterate_var(c: np.ndarray, A: np.ndarray, hist: list[np.ndarray], h: int, p: int) -> np.ndarray:
    N = c.shape[0]
    out = np.zeros((h, N))
    lags = [x.copy() for x in hist[-p:]]
    n_lags = p * N
    for t in range(h):
        z = np.concatenate([lags[-k] for k in range(1, p + 1)])
        nxt = c + A @ z[:n_lags]
        out[t] = nxt
        lags.append(nxt)
        lags = lags[-p:]
    return out


def paper_pca(Z: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Section 5: PCA factors of the T x d panel (Bai 2003 / paper eq. 5.3)."""
    pca = PCA(n_components=k, random_state=RANDOM_STATE)
    F = pca.fit_transform(Z)
    lam = pca.components_.T
    X = Z - F @ lam.T
    return F, lam, X, float(pca.explained_variance_ratio_.sum())


def fit_chen_factor_adjusted(train_ret: np.ndarray, p: int, h_bw: float, r: int):
    T, N = train_ret.shape
    mu = train_ret.mean(axis=0)
    Rc = train_ret - mu
    F, lam, U, explained = paper_pca(Rc, r)
    print(f"  PCA k={r}, variance explained={explained:.3f}")
    print("  Factor TV-VAR (unpenalised local linear)")
    cF, AF, _, _ = local_linear_stage(F, p=p, tau=1.0, h=h_bw, n_obs=T, lasso=False)
    print("  Idiosyncratic TV-VAR (two-stage LASSO)")
    cU, AU = two_stage_tv_var(U, p=p, h_bw=h_bw)
    return {
        "mu": mu,
        "lam": lam,
        "cF": cF,
        "AF": AF,
        "cU": cU,
        "AU": AU,
        "F": F,
        "U": U,
        "r": r,
        "explained": explained,
    }


def forecast_chen(fit: dict, h: int, p: int) -> np.ndarray:
    F_path = iterate_var(fit["cF"], fit["AF"], [fit["F"][i] for i in range(len(fit["F"]))], h, p)
    U_path = iterate_var(fit["cU"], fit["AU"], [fit["U"][i] for i in range(len(fit["U"]))], h, p)
    return fit["mu"] + F_path @ fit["lam"].T + U_path


def fit_chen_direct(train_ret: np.ndarray, p: int, h_bw: float):
    c, A = two_stage_tv_var(train_ret, p=p, h_bw=h_bw)
    return {"c": c, "A": A, "Y": train_ret}


def forecast_direct(fit: dict, h: int, p: int) -> np.ndarray:
    return iterate_var(fit["c"], fit["A"], [fit["Y"][i] for i in range(len(fit["Y"]))], h, p)


def price_path_from_returns(last_price: np.ndarray, ret_path: np.ndarray) -> np.ndarray:
    return last_price * np.exp(np.cumsum(ret_path, axis=0))


def main() -> None:
    prices = pd.read_csv(DATA_DIR / "djia_adj_close_wide_10y.csv", parse_dates=["Date"], index_col="Date")
    prices = prices.sort_index().ffill().dropna(how="all")
    tickers = prices.columns.tolist()
    log_ret = np.diff(np.log(prices.values), axis=0)
    train_ret = log_ret[:-HORIZON]
    train_prices = prices.iloc[:-HORIZON]
    test_prices = prices.iloc[-HORIZON:]
    last_train_price = train_prices.iloc[-1].values
    fc_dates = test_prices.index
    print(f"Train returns: {train_ret.shape[0]}  |  test: {fc_dates[0].date()} to {fc_dates[-1].date()}")

    print("Fitting Chen et al. factor-adjusted TV-VAR (Sections 3 and 5) ...")
    fit_fa = fit_chen_factor_adjusted(train_ret, p=LAG_P, h_bw=BANDWIDTH, r=N_FACTORS)
    chen_price = price_path_from_returns(last_train_price, forecast_chen(fit_fa, HORIZON, LAG_P))

    print("Fitting Chen et al. direct sparse TV-VAR (no factors) ...")
    fit_dir = fit_chen_direct(train_ret, p=LAG_P, h_bw=BANDWIDTH)
    direct_price = price_path_from_returns(last_train_price, forecast_direct(fit_dir, HORIZON, LAG_P))

    pd.DataFrame(chen_price, index=fc_dates, columns=tickers).rename_axis("Date").to_csv(
        DATA_DIR / "djia_6m_chen_tvnvar_mean.csv"
    )
    pd.DataFrame(direct_price, index=fc_dates, columns=tickers).rename_axis("Date").to_csv(
        DATA_DIR / "djia_6m_chen_direct_mean.csv"
    )
    actual = test_prices.values
    for name, pred in (
        ("Chen factor-adjusted", chen_price),
        ("Chen direct", direct_price),
    ):
        mape = np.mean(np.abs(pred - actual) / np.maximum(np.abs(actual), 1e-8)) * 100
        mae = np.mean(np.abs(pred - actual))
        print(f"{name:24s}  MAPE {mape:6.2f}%  MAE {mae:7.2f}")
    meta = {
        "paper": "Chen, Li, Li, Linton, Journal of Econometrics 249 (2025) 105941",
        "p": LAG_P,
        "bandwidth": BANDWIDTH,
        "kernel": "Epanechnikov local linear at tau=1",
        "n_factors": N_FACTORS,
        "factor_var_explained": fit_fa["explained"],
        "penalty": "two-stage TV-LASSO then SCAD-weighted LASSO (eqs 3.4-3.5)",
    }
    (DATA_DIR / "chen_fit_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Saved djia_6m_chen_tvnvar_mean.csv and djia_6m_chen_direct_mean.csv")


if __name__ == "__main__":
    main()
