#!/usr/bin/env python3
"""
Bayesian Compressed VAR (Koop, Korobilis, Pettenuzzo, JoE 2019).

Triangular (equation-by-equation) compressed VAR with Achlioptas random
projections and BIC-weighted BMA, as in their Sections 3 and 5.

  Y_t = c + B_1 Y_{t-1} + ... + B_p Y_{t-p} + e_t

is written in lower-triangular form so contemporaneous dependence sits in A
(ones on the diagonal). Equation i is a univariate regression of y_{i,t} on
lags, intercept, and contemporaneous y_{j,t} for j < i. Those regressors are
randomly compressed (first own lag left uncompressed). Posterior means are
analytical (flat / non-informative conjugate). R random compressions are
averaged with BIC weights.

BCVAR-TVP uses the Koop-Korobilis forgetting-factor filter on the compressed
coefficients (lambda=0.98, kappa=0.94).

Holdout: train through 2026-02-05, evaluate 2026-02-06 to 2026-08-07.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
HORIZON = 126
LAG_P = 2
N_COMPRESS = 50
SEED = 0
PHI_LO, PHI_HI = 0.1, 0.8
RIDGE = 1e-8
COMPANION_CAP = 0.95
LAMBDA0 = 0.98
KAPPA0 = 0.94


def lagged_rows(Y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    T, n = Y.shape
    rows = []
    targets = []
    for t in range(p, T):
        z = [1.0]
        for k in range(1, p + 1):
            z.extend(Y[t - k])
        rows.append(z)
        targets.append(Y[t])
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


def draw_projection(m: int, k: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    """Achlioptas / Guhaniyogi-Dunson sparse Rademacher, then row-orthonormalize."""
    if m <= 0 or k <= 0:
        return np.zeros((0, k))
    u = rng.random((m, k))
    val = np.sqrt(1.0 / max(phi, 1e-8))
    Phi = np.zeros((m, k))
    Phi[u < phi / 2.0] = val
    Phi[u > 1.0 - phi / 2.0] = -val
    q, r = np.linalg.qr(Phi.T, mode="reduced")
    keep = np.abs(np.diag(r)) > 1e-10
    if not np.any(keep):
        keep[0] = True
    return q[:, keep].T


def ols_beta(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, int]:
    nobs, m = X.shape
    gram = X.T @ X + RIDGE * np.eye(m)
    beta = np.linalg.solve(gram, X.T @ y)
    resid = y - X @ beta
    sse = float(resid @ resid)
    return beta, sse, nobs


def bic_from_sse(sse: float, nobs: int, npar: int) -> float:
    sse = max(sse, 1e-12)
    return float(nobs * np.log(sse / nobs) + npar * np.log(nobs))


def companion_radius(c: np.ndarray, B_lags: list[np.ndarray]) -> float:
    n = c.shape[0]
    p = len(B_lags)
    C = np.zeros((p * n, p * n))
    for k, Bk in enumerate(B_lags):
        C[:n, k * n : (k + 1) * n] = Bk
    if p > 1:
        C[n:, :-n] = np.eye((p - 1) * n)
    return float(np.max(np.abs(np.linalg.eigvals(C))))


def scale_lags(c: np.ndarray, B_lags: list[np.ndarray], cap: float) -> tuple[np.ndarray, list[np.ndarray]]:
    rad = companion_radius(c, B_lags)
    if rad > cap:
        s = cap / (rad + 1e-12)
        B_lags = [s * B for B in B_lags]
    return c, B_lags


def iterate_var(c: np.ndarray, B_lags: list[np.ndarray], hist: list[np.ndarray], h: int) -> np.ndarray:
    p = len(B_lags)
    n = c.shape[0]
    out = np.zeros((h, n))
    lags = [x.copy() for x in hist[-p:]]
    for t in range(h):
        nxt = c.copy()
        for k, Bk in enumerate(B_lags):
            nxt = nxt + Bk @ lags[-(k + 1)]
        nxt = np.clip(nxt, -0.08, 0.08)
        out[t] = nxt
        lags.append(nxt)
        lags = lags[-p:]
    return out


def fit_one_compression(
    Zlag: np.ndarray,
    Yt: np.ndarray,
    p: int,
    rng: np.random.Generator,
    compress_contemp: bool,
) -> tuple[np.ndarray, list[np.ndarray], float, np.ndarray]:
    """
    Equation-by-equation compressed triangular VAR.

    Returns reduced-form intercept c, lag matrices B_1..B_p, total BIC,
    and stacked lag operator (n, p*n) for Granger support.
    """
    nobs, n = Yt.shape
    k_lag = 1 + p * n
    bic_sum = 0.0
    intercepts = np.zeros(n)
    gamma_lags = np.zeros((n, p * n))
    A = np.eye(n)

    for i in range(n):
        extra = Yt[:, :i] if i > 0 else np.zeros((nobs, 0))
        Z = np.hstack([Zlag, extra]) if extra.size else Zlag.copy()
        ki = Z.shape[1]
        own_lag = 1 + i
        keep = np.zeros(ki, dtype=bool)
        keep[0] = True
        keep[own_lag] = True
        if (not compress_contemp) and i > 0:
            keep[k_lag:] = True
        rest = ~keep
        phi = float(rng.uniform(PHI_LO, PHI_HI))
        k_rest = int(rest.sum())
        m_max = max(1, int(np.floor(5.0 * np.log(max(ki, 2)))))
        m = int(rng.integers(1, m_max + 1))
        m = min(m, k_rest, nobs - 2) if k_rest else 0
        Phi = draw_projection(m, k_rest, phi, rng) if m > 0 else np.zeros((0, k_rest))
        parts = [Z[:, keep]]
        if Phi.size:
            parts.append(Z[:, rest] @ Phi.T)
        X = np.hstack(parts)
        y = Yt[:, i]
        beta, sse, n_used = ols_beta(X, y)
        npar = X.shape[1]
        bic_sum += bic_from_sse(sse, n_used, npar)

        theta = np.zeros(ki)
        n_keep = int(keep.sum())
        theta[keep] = beta[:n_keep]
        if Phi.size:
            theta[rest] = Phi.T @ beta[n_keep:]
        intercepts[i] = theta[0]
        gamma_lags[i] = theta[1:k_lag]
        if i > 0:
            A[i, :i] = -theta[k_lag : k_lag + i]

    Ainv = np.linalg.inv(A)
    c = Ainv @ intercepts
    Gl = Ainv @ gamma_lags
    B_lags = [Gl[:, k * n : (k + 1) * n] for k in range(p)]
    c, B_lags = scale_lags(c, B_lags, COMPANION_CAP)
    Gl = np.hstack(B_lags)
    return c, B_lags, bic_sum, Gl


def forgetting_filter(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Last-origin coefficient from a forgetting-factor Kalman filter."""
    T, m = X.shape
    theta = np.zeros(m)
    P = np.eye(m)
    sig2 = 1e-4
    for t in range(T):
        x = X[t]
        lam = LAMBDA0 + (1.0 - LAMBDA0) * np.exp(-0.5 * ((y[t - 1] - X[t - 1] @ theta) ** 2 / max(sig2, 1e-12) if t else 0.0))
        lam = float(np.clip(lam, LAMBDA0, 1.0))
        P = P / lam
        s = float(x @ P @ x) + sig2
        K = (P @ x) / s
        e = float(y[t] - x @ theta)
        theta = theta + K * e
        P = P - np.outer(K, x) @ P
        P = 0.5 * (P + P.T) + 1e-10 * np.eye(m)
        sig2 = KAPPA0 * sig2 + (1.0 - KAPPA0) * e * e
        sig2 = float(np.clip(sig2, 1e-10, 1.0))
    return theta


def fit_one_compression_tvp(
    Zlag: np.ndarray,
    Yt: np.ndarray,
    p: int,
    rng: np.random.Generator,
    compress_contemp: bool,
) -> tuple[np.ndarray, list[np.ndarray], float, np.ndarray]:
    nobs, n = Yt.shape
    k_lag = 1 + p * n
    bic_sum = 0.0
    intercepts = np.zeros(n)
    gamma_lags = np.zeros((n, p * n))
    A = np.eye(n)

    for i in range(n):
        extra = Yt[:, :i] if i > 0 else np.zeros((nobs, 0))
        Z = np.hstack([Zlag, extra]) if extra.size else Zlag.copy()
        ki = Z.shape[1]
        own_lag = 1 + i
        keep = np.zeros(ki, dtype=bool)
        keep[0] = True
        keep[own_lag] = True
        if (not compress_contemp) and i > 0:
            keep[k_lag:] = True
        rest = ~keep
        phi = float(rng.uniform(PHI_LO, PHI_HI))
        k_rest = int(rest.sum())
        m_max = max(1, int(np.floor(5.0 * np.log(max(ki, 2)))))
        m = int(rng.integers(1, m_max + 1))
        m = min(m, k_rest, nobs - 2) if k_rest else 0
        Phi = draw_projection(m, k_rest, phi, rng) if m > 0 else np.zeros((0, k_rest))
        parts = [Z[:, keep]]
        if Phi.size:
            parts.append(Z[:, rest] @ Phi.T)
        X = np.hstack(parts)
        y = Yt[:, i]
        beta = forgetting_filter(X, y)
        resid = y - X @ beta
        sse = float(resid @ resid)
        bic_sum += bic_from_sse(sse, nobs, X.shape[1])

        theta = np.zeros(ki)
        n_keep = int(keep.sum())
        theta[keep] = beta[:n_keep]
        if Phi.size:
            theta[rest] = Phi.T @ beta[n_keep:]
        intercepts[i] = theta[0]
        gamma_lags[i] = theta[1:k_lag]
        if i > 0:
            A[i, :i] = -theta[k_lag : k_lag + i]

    Ainv = np.linalg.inv(A)
    c = Ainv @ intercepts
    Gl = Ainv @ gamma_lags
    B_lags = [Gl[:, k * n : (k + 1) * n] for k in range(p)]
    c, B_lags = scale_lags(c, B_lags, COMPANION_CAP)
    return c, B_lags, bic_sum, np.hstack(B_lags)


def bma_fit(
    Y: np.ndarray,
    p: int,
    R: int,
    rng: np.random.Generator,
    compress_contemp: bool,
    tvp: bool,
) -> dict:
    Zlag, Yt = lagged_rows(Y, p)
    bics = []
    models = []
    fitter = fit_one_compression_tvp if tvp else fit_one_compression
    for r in range(R):
        c, B_lags, bic, Gl = fitter(Zlag, Yt, p, rng, compress_contemp)
        models.append((c, B_lags, Gl))
        bics.append(bic)
        if (r + 1) % 10 == 0 or r == 0:
            print(f"    compression {r + 1}/{R}  BIC={bic:.1f}")
    bics = np.asarray(bics)
    psi = bics - bics.min()
    w = np.exp(-0.5 * psi)
    w = w / w.sum()
    n = Y.shape[1]
    c_bar = np.zeros(n)
    B_bar = [np.zeros((n, n)) for _ in range(p)]
    G_bar = np.zeros((n, p * n))
    for wi, (c, B_lags, Gl) in zip(w, models):
        c_bar += wi * c
        for k in range(p):
            B_bar[k] += wi * B_lags[k]
        G_bar += wi * Gl
    c_bar, B_bar = scale_lags(c_bar, B_bar, COMPANION_CAP)
    return {
        "c": c_bar,
        "B": B_bar,
        "G": G_bar,
        "weights": w,
        "bics": bics,
        "ess": float(1.0 / np.sum(w**2)),
    }


def granger_density(G: np.ndarray, n: int, p: int, tol: float = 1e-4) -> float:
    S = np.zeros((n, n))
    for k in range(p):
        S = np.maximum(S, np.abs(G[:, k * n : (k + 1) * n]))
    np.fill_diagonal(S, 0.0)
    return float((S > tol).mean() * n / (n - 1))


def price_path(last_price: np.ndarray, ret: np.ndarray) -> np.ndarray:
    return last_price * np.exp(np.cumsum(ret, axis=0))


def unstandardize_path(path_z: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return path_z * sd + mu


def main() -> None:
    rng = np.random.default_rng(SEED)
    prices = pd.read_csv(DATA_DIR / "djia_adj_close_wide_10y.csv", parse_dates=["Date"], index_col="Date")
    prices = prices.sort_index().ffill().dropna(how="all")
    tickers = prices.columns.tolist()
    log_ret = np.diff(np.log(prices.values), axis=0)
    train_ret = log_ret[:-HORIZON]
    train_prices = prices.iloc[:-HORIZON]
    test_prices = prices.iloc[-HORIZON:]
    last_train_price = train_prices.iloc[-1].values
    fc_dates = test_prices.index
    hist = [train_ret[i] for i in range(len(train_ret))]

    mu = train_ret.mean(axis=0)
    sd = train_ret.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Yz = (train_ret - mu) / sd
    hist_z = [(train_ret[i] - mu) / sd for i in range(len(train_ret))]

    specs = [
        ("BCVAR", True, False, "djia_6m_bcvar_mean.csv"),
        ("BCVAR-TVP", True, True, "djia_6m_bcvar_tvp_mean.csv"),
    ]
    meta = {
        "paper": "Koop, Korobilis, Pettenuzzo, Journal of Econometrics 210 (2019) 135-154",
        "p": LAG_P,
        "R": N_COMPRESS,
        "projection": "Achlioptas / Guhaniyogi-Dunson, phi ~ U[0.1,0.8], m ~ U{1,...,5 ln k_i}",
        "bma": "BIC weights, equation-by-equation triangular VAR",
        "own_lag_uncompressed": True,
        "specs": {},
    }
    print(f"Train returns {train_ret.shape}  test {fc_dates[0].date()} to {fc_dates[-1].date()}")
    for name, compress_c, tvp, fname in specs:
        print(f"Fitting {name} (R={N_COMPRESS}, compress contemporaneous={compress_c}, tvp={tvp})")
        fit = bma_fit(Yz, LAG_P, N_COMPRESS, rng, compress_c, tvp)
        path_z = iterate_var(fit["c"], fit["B"], hist_z, HORIZON)
        path = unstandardize_path(path_z, mu, sd)
        path = np.clip(path, -0.03, 0.03)
        pred_price = price_path(last_train_price, path)
        pd.DataFrame(pred_price, index=fc_dates, columns=tickers).rename_axis("Date").to_csv(DATA_DIR / fname)
        actual = test_prices.values
        mape = np.mean(np.abs(pred_price - actual) / np.maximum(np.abs(actual), 1e-8)) * 100
        mae = np.mean(np.abs(pred_price - actual))
        dens = granger_density(fit["G"], train_ret.shape[1], LAG_P)
        print(f"  {name:12s}  MAPE {mape:6.2f}%  MAE {mae:7.2f}  BMA ESS {fit['ess']:.1f}  Granger dens {dens:.3f}")
        np.savez(
            DATA_DIR / f"{name.lower().replace('-', '_')}_operator.npz",
            c=fit["c"],
            B1=fit["B"][0],
            B2=fit["B"][1],
            G=fit["G"],
            weights=fit["weights"],
        )
        meta["specs"][name] = {
            "file": fname,
            "avg_mape": float(mape),
            "avg_mae": float(mae),
            "bma_ess": fit["ess"],
            "granger_density": dens,
            "n_compressions": N_COMPRESS,
        }
        print(f"  Saved {fname}")
    (DATA_DIR / "bcvar_fit_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
