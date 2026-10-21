#!/usr/bin/env python3
"""
LS-LR-NVAR Gibbs Steps 0-10 and predictive Steps 11 / 11a from
"Bayesian Locally stationary low-rank Network VAR model for financial
time series specification.docx".

In-sample y_t are daily log-returns. Network: mu_ij = u_i'v_j + u_{i,t}'v_{j,t}
(no probit intercept, no correlation floor). W is the row-normalisation of
binary A. Dynamic states are random walks. Step 11 uses lagged (I+W) odot B
and averages posterior predictive simulations. Step 11a is expanding-window
evaluation with the origin-T posterior (warm-start re-estimation at every
day is not repeated; actual lags are updated).

Holdout: train through 2026-02-05, evaluate 2026-02-06 to 2026-08-07.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from hierarchical_forecast import forecast_from_phi
from ls_lr_nvar_6m_forecast import (
    DATA_DIR,
    HORIZON,
    LAG_P,
    RB,
    R0,
    RMU,
    RZ,
    SEED,
    LSLRNVAR,
    apply_phi,
    latent_mean,
    network_from_latents,
    sample_A_bernoulli,
    sample_probit_z,
    spectral_radius,
)

SAMPLER_VERSION = "spec-steps-0-10"
RHO_RW = 1.0
N_SWEEP = 36
BURN = 18
THIN = 2
N_MC = 20  # extra predictive sims per posterior draw (same integral, less MC noise)


def spec_row_normalize(A: np.ndarray) -> np.ndarray:
    """W_ij = A_ij / d_i if d_i>0 else 0 (spec Step 0 / Step 11)."""
    W = np.asarray(A, dtype=np.float64).copy()
    if W.ndim == 2:
        np.fill_diagonal(W, 0.0)
        deg = W.sum(axis=1, keepdims=True)
        out = np.zeros_like(W)
        np.divide(W, deg, out=out, where=deg > 0)
        return out
    n = W.shape[1]
    W[:, np.arange(n), np.arange(n)] = 0.0
    deg = W.sum(axis=2, keepdims=True)
    out = np.zeros_like(W)
    np.divide(W, deg, out=out, where=deg > 0)
    return out


def _svd_sym_factors(M: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    r = min(r, U.shape[1])
    U = U[:, :r] * np.sqrt(np.maximum(s[:r], 0.0))
    V = Vt[:r, :].T * np.sqrt(np.maximum(s[:r], 0.0))
    for ell in range(r):
        if U[0, ell] < 0:
            U[:, ell] *= -1.0
            V[:, ell] *= -1.0
    return U, V


class LSLRNVARSpec(LSLRNVAR):
    """Specification Gibbs: random-walk states, row-normalized A, no MH."""

    def __init__(self, p=LAG_P, r0=R0, rmu=RMU, rb=RB, rz=RZ):
        super().__init__(p=p, r0=r0, rmu=rmu, rb=rb, rz=rz)
        self.rho_mean = RHO_RW
        self.rho_net = RHO_RW
        self.delta = 1e-8

    def initialize(self, Y: np.ndarray) -> None:
        super().initialize(Y)
        self.corr_boost = np.zeros((self.n, self.n))
        self.P, self.W = self._rebuild_PW(sample_a=True)

    def _z_mean(self) -> np.ndarray:
        return latent_mean(
            self.u, self.v, self.u_t, self.v_t,
            corr_boost=None, intercept=0.0,
        )

    def _rebuild_PW(self, sample_a: bool = True) -> tuple[np.ndarray, np.ndarray]:
        _, _, P = network_from_latents(
            self.u, self.v, self.u_t, self.v_t,
            corr_boost=None, intercept=0.0, delta=1e-8, hard=False,
        )
        if sample_a:
            self.A = sample_A_bernoulli(P)
        else:
            self.A = (P >= 0.5).astype(np.float64)
            n = self.A.shape[-1]
            self.A[:, np.arange(n), np.arange(n)] = 0.0
        W = spec_row_normalize(self.A)
        return P, W

    def update_network_static(self) -> None:
        n, T, r = self.n, self.T, self.rz
        z_mean = self._z_mean()
        z = sample_probit_z(z_mean, self.A)
        r_stat = z - np.einsum("tir,tjr->tij", self.u_t, self.v_t)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            S = np.eye(r) / self.sigma_z2
            m = np.zeros(r)
            for j in np.where(mask)[0]:
                S = S + T * np.outer(self.v[j], self.v[j])
                m = m + r_stat[:, i, j].sum() * self.v[j]
            cov = np.linalg.inv(S + 1e-6 * np.eye(r))
            self.u[i] = np.random.multivariate_normal(cov @ m, cov)
        for j in range(n):
            mask = np.ones(n, dtype=bool)
            mask[j] = False
            S = np.eye(r) / self.sigma_z2
            m = np.zeros(r)
            for i in np.where(mask)[0]:
                S = S + T * np.outer(self.u[i], self.u[i])
                m = m + r_stat[:, i, j].sum() * self.u[i]
            cov = np.linalg.inv(S + 1e-6 * np.eye(r))
            self.v[j] = np.random.multivariate_normal(cov @ m, cov)
        self.P, self.W = self._rebuild_PW(sample_a=True)
        self.n_mh_static += 1
        self.n_mh_static_ok += 1

    def update_network_dynamic(self) -> None:
        from ls_lr_nvar_6m_forecast import ar_innov_ssq, ffbs, _clip_tau, draw_ig

        n, T, r = self.n, self.T, self.rz
        z_mean = self._z_mean()
        z = sample_probit_z(z_mean, self.A)
        r_dyn = z - (self.u @ self.v.T)[None, :, :]
        Sigma_obs = np.eye(n - 1)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            obs = r_dyn[:, i, mask]
            H = self.v_t[:, mask, :]
            self.u_t[:, i, :] = ffbs(obs, H, Sigma_obs, self.tau_u, rho=self.rho_net)
        for j in range(n):
            mask = np.ones(n, dtype=bool)
            mask[j] = False
            obs = r_dyn[:, mask, j]
            H = self.u_t[:, mask, :]
            self.v_t[:, j, :] = ffbs(obs, H, Sigma_obs, self.tau_v, rho=self.rho_net)
        self.tau_u = _clip_tau(draw_ig(2.0 + 0.5 * n * r * (T - 1), 0.01 + 0.5 * ar_innov_ssq(self.u_t, self.rho_net)))
        self.tau_v = _clip_tau(draw_ig(2.0 + 0.5 * n * r * (T - 1), 0.01 + 0.5 * ar_innov_ssq(self.v_t, self.rho_net)))
        self.P, self.W = self._rebuild_PW(sample_a=True)
        self.n_mh_dyn += 1
        self.n_mh_dyn_ok += 1

    def step10_identify(self) -> None:
        qf, rf = np.linalg.qr(self.F, mode="reduced")
        self.F = qf
        self.lam = self.lam @ rf.T

        extra = np.einsum("nr,tnk,tk->tr", self.F, self.Ft, self.gam)
        self.lam = self.lam + extra
        pperp = np.eye(self.n) - self.F @ self.F.T
        self.Ft = np.einsum("ij,tjk->tik", pperp, self.Ft)
        for t in range(self.T):
            col = self.Ft[t, :, 0]
            nrm = float(np.linalg.norm(col)) + 1e-12
            self.Ft[t, :, 0] = col / nrm
            self.gam[t, 0] *= nrm
            if self.Ft[t, 0, 0] < 0:
                self.Ft[t] *= -1.0
                self.gam[t] *= -1.0

        self.u, self.v = _svd_sym_factors(self.u @ self.v.T, self.rz)
        for t in range(self.T):
            self.u_t[t], self.v_t[t] = _svd_sym_factors(self.u_t[t] @ self.v_t[t].T, self.rz)

        for i in range(self.p):
            for t in range(self.T):
                self.U[i, t], self.V[i, t] = _svd_sym_factors(
                    self.U[i, t] @ self.V[i, t].T, self.rb
                )

        g = np.eye(self.n) + self.W[-1]
        for i in range(self.p):
            b = self.U[i, -1] @ self.V[i, -1].T
            rad = spectral_radius(g * b)
            if rad > 0.95:
                s = 0.95 / (rad + 1e-12)
                self.U[i] *= np.sqrt(s)
                self.V[i] *= np.sqrt(s)

        self.mu = self.lam @ self.F.T + np.einsum("tnr,tr->tn", self.Ft, self.gam)
        self.P, self.W = self._rebuild_PW(sample_a=False)

    def snapshot(self) -> dict:
        p = self.p
        t0 = self.T - p
        b_tail = []
        for i in range(p):
            b_tail.append(
                np.einsum("tnr,tjr->tnj", self.U[i, t0:], self.V[i, t0:])
            )
        return {
            "Y_tail": self.Y[-p:].copy(),
            "mu_tail": self.mu[-p:].copy(),
            "W_tail": self.W[-p:].copy(),
            "B_tail": b_tail,
            "F": self.F.copy(),
            "lam": self.lam[-1].copy(),
            "Ft": self.Ft[-1].copy(),
            "gam": self.gam[-1].copy(),
            "u": self.u.copy(),
            "v": self.v.copy(),
            "u_t": self.u_t[-1].copy(),
            "v_t": self.v_t[-1].copy(),
            "U": [self.U[i, -1].copy() for i in range(p)],
            "V": [self.V[i, -1].copy() for i in range(p)],
            "Sigma": self.Sigma.copy(),
            "tau_u": float(self.tau_u),
            "tau_v": float(self.tau_v),
            "tau_lam": float(self.tau_lam),
            "tau_gam": float(self.tau_gam),
            "tau_F": float(self.tau_F),
            "tau_U": self.tau_U.copy(),
            "tau_V": self.tau_V.copy(),
            "mu_T": self.mu[-1].copy(),
            "rad": float(spectral_radius((np.eye(self.n) + self.W[-1]) * (self.U[0, -1] @ self.V[0, -1].T))),
        }

    def fit(self, Y: np.ndarray, n_sweep=N_SWEEP, burn=BURN, thin=THIN, collect_A: bool = False):
        self.initialize(Y)
        draws = []
        A_acc = None
        Phi_acc = None
        n_keep = 0
        print(
            f"LS-LR-NVAR Gibbs {SAMPLER_VERSION}: T={self.T}, n={self.n}, "
            f"sweeps={n_sweep}, burn={burn}, RW rho={self.rho_mean}"
        )
        for s in range(n_sweep):
            self.update_network_static()
            self.update_network_dynamic()
            self.update_mean()
            self.update_ar()
            self.update_sigma()
            self.step10_identify()
            if (s + 1) % 5 == 0 or s == 0:
                dens = float(self.A[-1].sum() / (self.n * (self.n - 1)))
                print(
                    f"  sweep {s + 1}/{n_sweep}  hard density(T)={dens:.3f}  "
                    f"tau_U={self.tau_U[0]:.4f}  |mu_T|={np.mean(np.abs(self.mu[-1])):.4f}  "
                    f"rad={spectral_radius((np.eye(self.n)+self.W[-1])*(self.U[0,-1]@self.V[0,-1].T)):.3f}",
                    flush=True,
                )
            if s >= burn and (s - burn) % thin == 0:
                draws.append(self.snapshot())
                if collect_A:
                    A_acc = self.P.copy() if A_acc is None else A_acc + self.P
                    phis = self.current_phi_path()
                    if Phi_acc is None:
                        Phi_acc = [pp.copy() for pp in phis]
                    else:
                        for k, pp in enumerate(phis):
                            Phi_acc[k] += pp
                    n_keep += 1
        if collect_A and n_keep:
            self.A_mean = A_acc / n_keep
            self.Phi_mean = [pp / n_keep for pp in Phi_acc]
        else:
            self.A_mean = None
            self.Phi_mean = None
        print(f"  kept {len(draws)} snapshots")
        return draws


def _rw(x: np.ndarray, tau: float) -> np.ndarray:
    return x + np.sqrt(max(tau, 1e-12)) * np.random.randn(*x.shape)


def _future_network(u, v, u_t, v_t) -> np.ndarray:
    mu_z = u @ v.T + u_t @ v_t.T
    np.fill_diagonal(mu_z, -8.0)
    p = norm.cdf(mu_z)
    np.fill_diagonal(p, 0.0)
    a = sample_A_bernoulli(p[None, ...])[0]
    return spec_row_normalize(a)


def step11_forecast(draw: dict, h: int, p: int, include_shock: bool = True) -> np.ndarray:
    """Spec Step 11: lagged (I+W)⊙B, RW states, optional shocks."""
    n = draw["F"].shape[0]
    out = np.zeros((h, n))
    y_hist = list(np.asarray(draw["Y_tail"], dtype=float))
    mu_hist = list(np.asarray(draw["mu_tail"], dtype=float))
    w_hist = list(np.asarray(draw["W_tail"], dtype=float))
    b_hist = [list(np.asarray(b, dtype=float)) for b in draw["B_tail"]]
    u = np.asarray(draw["u"], dtype=float)
    v = np.asarray(draw["v"], dtype=float)
    u_t = np.asarray(draw["u_t"], dtype=float)
    v_t = np.asarray(draw["v_t"], dtype=float)
    f = np.asarray(draw["F"], dtype=float)
    lam = np.asarray(draw["lam"], dtype=float).copy()
    ft = np.asarray(draw["Ft"], dtype=float).copy()
    gam = np.asarray(draw["gam"], dtype=float).copy()
    u_now = [np.asarray(x, dtype=float).copy() for x in draw["U"]]
    v_now = [np.asarray(x, dtype=float).copy() for x in draw["V"]]
    sigma = np.asarray(draw["Sigma"], dtype=float)
    chol = np.linalg.cholesky(0.5 * (sigma + sigma.T) + 1e-8 * np.eye(n))
    for _t in range(h):
        lam = _rw(lam, draw["tau_lam"])
        gam = _rw(gam, draw["tau_gam"])
        ft = _rw(ft, draw["tau_F"])
        mu = f @ lam + ft @ gam
        nxt = mu.copy()
        for i in range(1, p + 1):
            nxt = nxt + apply_phi(w_hist[-i], b_hist[i - 1][-i], y_hist[-i] - mu_hist[-i])
        if include_shock:
            nxt = nxt + chol @ np.random.randn(n)
        out[_t] = nxt
        u_t = _rw(u_t, draw["tau_u"])
        v_t = _rw(v_t, draw["tau_v"])
        w_new = _future_network(u, v, u_t, v_t)
        b_new = []
        for i in range(p):
            u_now[i] = _rw(u_now[i], float(draw["tau_U"][i]))
            v_now[i] = _rw(v_now[i], float(draw["tau_V"][i]))
            b_new.append(u_now[i] @ v_now[i].T)
        y_hist.append(nxt)
        mu_hist.append(mu.copy())
        w_hist.append(w_new)
        for i in range(p):
            b_hist[i].append(b_new[i])
        y_hist = y_hist[-p:]
        mu_hist = mu_hist[-p:]
        w_hist = w_hist[-p:]
        for i in range(p):
            b_hist[i] = b_hist[i][-p:]
    return out


def price_path(last_price: np.ndarray, ret: np.ndarray) -> np.ndarray:
    return last_price * np.exp(np.cumsum(ret, axis=0))


def predictive_mean(draws: list[dict], h: int, p: int, n_mc: int, include_shock: bool) -> np.ndarray:
    acc = None
    n_path = 0
    for d in draws:
        for _ in range(n_mc):
            path = step11_forecast(d, h, p, include_shock=include_shock)
            acc = path if acc is None else acc + path
            n_path += 1
    return acc / max(n_path, 1)


def rebase_draw(draw: dict, y_hist: np.ndarray, mu_hist: np.ndarray) -> dict:
    d = dict(draw)
    d["Y_tail"] = np.asarray(y_hist, dtype=float).copy()
    d["mu_tail"] = np.asarray(mu_hist, dtype=float).copy()
    return d


def run_step11a(draws: list[dict], ret_all: np.ndarray, t0: int, horizons=(1, 5, 21)) -> dict:
    """Expanding-window forward simulation (spec Step 11a, frozen posterior)."""
    p = LAG_P
    t_all = ret_all.shape[0]
    out = {h: {"ae_ret": []} for h in horizons}
    last_o = t_all - max(horizons)
    n_o = 0
    hmax = max(horizons)
    for origin in range(t0, last_o):
        y_hist = ret_all[origin - p + 1 : origin + 1]
        mu_hist = y_hist.copy()
        paths = []
        for d in draws:
            dd = rebase_draw(d, y_hist, mu_hist)
            paths.append(step11_forecast(dd, hmax, p, include_shock=False))
        mean_path = np.mean(np.stack(paths, axis=0), axis=0)
        for h in horizons:
            err = mean_path[h - 1] - ret_all[origin + h]
            out[h]["ae_ret"].append(float(np.mean(np.abs(err))))
        n_o += 1
    summary = {}
    for h in horizons:
        summary[str(h)] = {
            "ret_mae": float(np.mean(out[h]["ae_ret"])),
            "n_origins": n_o,
        }
    return summary


def main() -> None:
    np.random.seed(SEED)
    prices = pd.read_csv(DATA_DIR / "djia_adj_close_wide_10y.csv", parse_dates=["Date"], index_col="Date")
    prices = prices.sort_index().ffill().dropna(how="all")
    tickers = prices.columns.tolist()
    log_ret = np.diff(np.log(prices.values), axis=0)
    train_ret = log_ret[:-HORIZON]
    train_prices = prices.iloc[:-HORIZON]
    test_prices = prices.iloc[-HORIZON:]
    last_train_price = train_prices.iloc[-1].values
    print(
        f"LS-LR-NVAR spec Steps 0-10 + 11  train {train_ret.shape}  "
        f"test {test_prices.index[0].date()} to {test_prices.index[-1].date()}"
    )

    prev = DATA_DIR / "djia_6m_lslrnvar_mean.csv"
    bak = DATA_DIR / "djia_6m_lslrnvar_revised_v2_mean.csv"
    if prev.exists() and not bak.exists():
        shutil.copy2(prev, bak)

    model = LSLRNVARSpec()
    draws = model.fit(train_ret, collect_A=True)
    if draws:
        print(
            f"  mean |mu_T|={np.mean([np.mean(np.abs(d['mu_T'])) for d in draws]):.4f}  "
            f"mean rad={np.mean([d['rad'] for d in draws]):.3f}"
        )
    if model.A_mean is not None:
        ret_dates = prices.index[1:-HORIZON]
        np.savez_compressed(
            DATA_DIR / "lslrnvar_A_mean_train_spec.npz",
            A_mean=model.A_mean,
            Phi1=model.Phi_mean[0] if model.Phi_mean else np.array([]),
            Phi2=model.Phi_mean[1] if model.Phi_mean else np.array([]),
            dates=ret_dates.strftime("%Y-%m-%d").to_numpy(),
        )

    if model.Phi_mean is not None:
        pred_price = forecast_from_phi(
            train_ret, last_train_price, model.Phi_mean[0][-1], model.Phi_mean[1][-1], HORIZON
        )
    else:
        mean_ret = predictive_mean(draws, HORIZON, LAG_P, N_MC, include_shock=False)
        mean_ret = np.nan_to_num(mean_ret, nan=0.0, posinf=0.0, neginf=0.0)
        pred_price = price_path(last_train_price, mean_ret)
    pd.DataFrame(pred_price, index=test_prices.index, columns=tickers).rename_axis("Date").to_csv(
        DATA_DIR / "djia_6m_lslrnvar_mean.csv"
    )
    actual = test_prices.values
    mape = np.mean(np.abs(pred_price - actual) / np.maximum(np.abs(actual), 1e-8), axis=0) * 100
    mae = np.mean(np.abs(pred_price - actual), axis=0)
    print(f"LS-LR-NVAR spec Step 11  MAPE {mape.mean():.2f}%  MAE {mae.mean():.2f}")

    print("Step 11a expanding-window rolling (frozen posterior, actual lags) ...")
    t0 = len(train_ret) - 1
    step11a = run_step11a(draws, log_ret, t0)
    for h, d in step11a.items():
        print(f"  horizon {h}d  ret MAE {d['ret_mae']*100:.3f}pp  origins {d['n_origins']}")

    meta = {
        "sampler_version": SAMPLER_VERSION,
        "spec": "Steps 0-10 Gibbs, Step 11 6m predictive simulation, Step 11a rolling",
        "avg_mape": float(mape.mean()),
        "avg_mae": float(mae.mean()),
        "n_draws": len(draws),
        "n_sweep": N_SWEEP,
        "burn": BURN,
        "n_mc": N_MC,
        "rho": RHO_RW,
        "network_W": "row-normalized binary A (spec); probit intercept 0; no corr floor",
        "step11a": step11a,
        "step11a_note": "frozen origin-T posterior; actual return lags; not daily warm-start MCMC",
    }
    (DATA_DIR / "lslrnvar_fit_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Saved data/djia_6m_lslrnvar_mean.csv")


if __name__ == "__main__":
    main()
