#!/usr/bin/env python3
"""
TVP-LR-NVAR-SV Gibbs sampler and 6-month DJIA open-loop forecast.

Observation on log-prices y_t = ln P_t:

  y_t = mu_t + sum_i [(I + W_{t-i}) odot B_{i,t-i}] (y_{t-i} - mu_{t-i}) + e_t
  mu_t = F lambda_t
  lambda_t = lambda_{t-1} + delta_{t-1} + eta^lambda
  delta_t  = (I - K) delta_bar + K delta_{t-1} + eta^delta
  e_t ~ N(0, Sigma_t),  Sigma_t = A^{-1} D_t (A^{-1})'
  D_t = diag(exp(h_{1,t}), ..., exp(h_{n,t}))

Network and low-rank lag factors reuse the revised LS-LR-NVAR updates
(ridge W, FFBS, VAR-likelihood MH). The local mean F_t gamma_t is replaced
by the local-linear trend; Inverse-Wishart Sigma is replaced by triangular MSV.

Holdout: train through 2026-02-05, evaluate 2026-02-06 to 2026-08-07.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import solve

from hierarchical_forecast import forecast_from_phi
from ls_lr_nvar_6m_forecast import (
    CORR_EDGE,
    FLOOR_BOOST,
    HORIZON,
    LAG_P,
    R0,
    RB,
    RIDGE_DELTA,
    RHO_NET,
    RZ,
    apply_phi,
    ar_innov_ssq,
    compute_residuals,
    draw_ig,
    ffbs,
    latent_mean,
    make_H_U,
    make_H_V,
    network_from_latents,
    sample_A_bernoulli,
    sample_probit_z,
    spectral_radius,
    LSLRNVAR,
    _clip_tau,
    _mvn,
    ar1_path,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
N_SWEEP = 24
BURN = 12
THIN = 2
SEED = 0
KAPPA_DELTA = 0.97
RHO_VOL = 0.95
TVP_R0 = 3
TVP_FLOOR = 0.10
MH_STEP0 = 0.03
RECENT_WIN = 21
SLOW_WIN = 63
SAMPLER_VERSION = "tvp-lr-nvar-sv-v2"

# Kim, Shephard, Chib (1998) 7-component log-chi^2 mixture
KSC_Q = np.array([0.00730, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.25750])
KSC_M = np.array([-10.12999, -3.97281, -8.56686, 2.77786, 0.61942, 1.79518, -1.08819])
KSC_V = np.array([5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261])


def ffbs_tv(obs: np.ndarray, H: np.ndarray, Sigmas: np.ndarray, q2: float, rho: float = 1.0) -> np.ndarray:
    """FFBS with Sigma_t (T,n,n) or a constant (n,n)."""
    T, n, d = H.shape
    I_d = np.eye(d)
    Q = q2 * I_d
    rho = float(rho)
    m = np.zeros((T, d))
    P = np.zeros((T, d, d))
    mp = np.zeros((T, d))
    Pp = np.zeros((T, d, d))
    mt = np.zeros(d)
    Pt = 5.0 * I_d
    jitter = 1e-8 * I_d
    const = Sigmas.ndim == 2
    for t in range(T):
        Sig = Sigmas if const else Sigmas[t]
        mp[t] = rho * mt
        Ppt = (rho * rho) * Pt + Q
        Pp[t] = Ppt
        Ht = H[t]
        S = Ht @ Ppt @ Ht.T + Sig
        S = 0.5 * (S + S.T) + 1e-8 * np.eye(n)
        K = solve(S, Ht @ Ppt).T
        innov = obs[t] - Ht @ mp[t]
        mt = mp[t] + K @ innov
        Pt = Ppt - K @ Ht @ Ppt
        Pt = 0.5 * (Pt + Pt.T) + jitter
        m[t] = mt
        P[t] = Pt
    draws = np.zeros((T, d))
    covT = 0.5 * (P[-1] + P[-1].T) + jitter
    draws[-1] = _mvn(m[-1], covT)
    for t in range(T - 2, -1, -1):
        J = P[t] @ solve(Pp[t + 1] + jitter, rho * I_d)
        mean = m[t] + J @ (draws[t + 1] - mp[t + 1])
        cov = P[t] - J @ Pp[t + 1] @ J.T
        cov = 0.5 * (cov + cov.T) + jitter
        draws[t] = _mvn(mean, cov)
    return draws


def ffbs_llt(obs: np.ndarray, F: np.ndarray, Sigmas: np.ndarray, q_lam: float, q_del: float, kappa: float) -> tuple[np.ndarray, np.ndarray]:
    """FFBS for stacked alpha_t = (lambda_t, delta_t) with local-linear transition."""
    T, n = obs.shape
    r0 = F.shape[1]
    d = 2 * r0
    I_r = np.eye(r0)
    Trans = np.zeros((d, d))
    Trans[:r0, :r0] = I_r
    Trans[:r0, r0:] = I_r
    Trans[r0:, r0:] = kappa * I_r
    Q = np.zeros((d, d))
    Q[:r0, :r0] = q_lam * I_r
    Q[r0:, r0:] = q_del * I_r
    H0 = np.zeros((n, d))
    H0[:, :r0] = F
    m = np.zeros((T, d))
    P = np.zeros((T, d, d))
    mp = np.zeros((T, d))
    Pp = np.zeros((T, d, d))
    mt = np.zeros(d)
    Pt = 5.0 * np.eye(d)
    jitter = 1e-8 * np.eye(d)
    const = Sigmas.ndim == 2
    for t in range(T):
        Sig = Sigmas if const else Sigmas[t]
        mp[t] = Trans @ mt
        Ppt = Trans @ Pt @ Trans.T + Q
        Pp[t] = Ppt
        S = H0 @ Ppt @ H0.T + Sig
        S = 0.5 * (S + S.T) + 1e-8 * np.eye(n)
        K = solve(S, H0 @ Ppt).T
        innov = obs[t] - H0 @ mp[t]
        mt = mp[t] + K @ innov
        Pt = Ppt - K @ H0 @ Ppt
        Pt = 0.5 * (Pt + Pt.T) + jitter
        m[t] = mt
        P[t] = Pt
    draws = np.zeros((T, d))
    covT = 0.5 * (P[-1] + P[-1].T) + jitter
    draws[-1] = _mvn(m[-1], covT)
    for t in range(T - 2, -1, -1):
        J = P[t] @ solve(Pp[t + 1] + jitter, Trans)
        mean = m[t] + J @ (draws[t + 1] - mp[t + 1])
        cov = P[t] - J @ Pp[t + 1] @ J.T
        cov = 0.5 * (cov + cov.T) + jitter
        draws[t] = _mvn(mean, cov)
    return draws[:, :r0], draws[:, r0:]


def ffbs_scalar_ar1(y: np.ndarray, var_obs: np.ndarray, q2: float, rho: float, mu_h: float) -> np.ndarray:
    """Scalar AR(1) FFBS for one log-volatility path."""
    T = y.shape[0]
    m = np.zeros(T)
    P = np.zeros(T)
    mp = np.zeros(T)
    Pp = np.zeros(T)
    mt = mu_h
    Pt = 1.0
    for t in range(T):
        mp[t] = mu_h + rho * (mt - mu_h)
        Pp[t] = (rho * rho) * Pt + q2
        K = Pp[t] / (Pp[t] + var_obs[t] + 1e-8)
        mt = mp[t] + K * (y[t] - mp[t])
        Pt = (1.0 - K) * Pp[t]
        m[t] = mt
        P[t] = max(Pt, 1e-8)
    draws = np.zeros(T)
    draws[-1] = m[-1] + np.sqrt(P[-1]) * np.random.randn()
    for t in range(T - 2, -1, -1):
        J = P[t] * rho / (Pp[t + 1] + 1e-8)
        mean = m[t] + J * (draws[t + 1] - mp[t + 1])
        var = max(P[t] - J * J * Pp[t + 1], 1e-8)
        draws[t] = mean + np.sqrt(var) * np.random.randn()
    return draws


def assemble_sigma_path(h: np.ndarray, A: np.ndarray) -> np.ndarray:
    T, n = h.shape
    Ainv = np.linalg.inv(A)
    out = np.zeros((T, n, n))
    for t in range(T):
        D = np.diag(np.exp(np.clip(h[t], -8.0, 4.0)))
        S = Ainv @ D @ Ainv.T
        out[t] = 0.5 * (S + S.T) + 1e-8 * np.eye(n)
    return out


def var_loglik_sv(E: np.ndarray, h: np.ndarray, A: np.ndarray, p: int) -> float:
    R = E[p:] @ A.T
    hh = h[p:]
    return float(-0.5 * np.sum(R**2 * np.exp(-hh) + hh))


class TVPLRNVARSV(LSLRNVAR):
    """Local-linear trend + network cycle + triangular MSV on log-prices."""

    def __init__(self, p=LAG_P, r0=TVP_R0, rb=RB, rz=RZ, kappa=KAPPA_DELTA):
        super().__init__(p=p, r0=r0, rmu=1, rb=rb, rz=rz)
        self.kappa = float(kappa)
        self.rho_vol = RHO_VOL
        self.mh_step = MH_STEP0

    def initialize(self, Y: np.ndarray) -> None:
        T, n = Y.shape
        self.T, self.n = T, n
        self.Y = Y
        C = np.corrcoef(np.diff(Y, axis=0).T)
        C = np.nan_to_num(C, nan=0.0)
        np.fill_diagonal(C, 0.0)
        uu, ss, vt = np.linalg.svd(C, full_matrices=False)
        self.u = uu[:, : self.rz] * np.sqrt(ss[: self.rz])
        self.v = vt[: self.rz, :].T * np.sqrt(ss[: self.rz])
        walk_u = ar1_path(T, self.rz, self.rho_net, 0.01)
        walk_v = ar1_path(T, self.rz, self.rho_net, 0.01)
        self.u_t = np.repeat(walk_u[:, None, :], n, axis=1)
        self.v_t = np.repeat(walk_v[:, None, :], n, axis=1)
        A0 = (np.abs(C) > CORR_EDGE).astype(np.float64)
        np.fill_diagonal(A0, 0.0)
        self.A_corr = A0
        self.corr_boost = TVP_FLOOR * A0
        self.ridge_delta = float(RIDGE_DELTA)
        self.P, self.W = self._rebuild_PW()
        self.A = (self.P >= 0.5).astype(np.float64)
        self.A[:, np.arange(n), np.arange(n)] = 0.0

        ybar = Y.mean(axis=0)
        Yc = Y - ybar
        Uy, sy, Vty = np.linalg.svd(Yc, full_matrices=False)
        self.F = (Vty[: self.r0, :].T * sy[: self.r0] / np.sqrt(T)).copy()
        self.lam = (Uy[:, : self.r0] * np.sqrt(T)).copy()
        qf, rf = np.linalg.qr(self.F, mode="reduced")
        self.F = qf
        self.lam = self.lam @ rf.T
        # start velocity at 0 so the open-loop drift is learned, not inherited from SVD scores
        self.vel = np.zeros((T, self.r0))
        self.delta = self.ridge_delta
        self.iota = np.zeros((T, n))
        self.tau_iota = 1e-5 * np.ones(n)
        self.mu = ybar + self.lam @ self.F.T + self.iota
        self.Ft = np.zeros((T, n, 1))
        self.gam = np.zeros((T, 1))
        self.mh_step = MH_STEP0

        self.U = np.zeros((self.p, T, n, self.rb))
        self.V = np.zeros((self.p, T, n, self.rb))
        for i in range(self.p):
            self.U[i, :, : self.rb, :] = 0.04 * np.eye(self.rb)
            self.V[i, :, : self.rb, :] = np.eye(self.rb)
            self.U[i] += 0.01 * np.random.randn(T, n, self.rb)
            self.V[i] += 0.01 * np.random.randn(T, n, self.rb)

        E = compute_residuals(Y, self.mu, self.W, self.U, self.V, self.p)
        var = np.maximum(np.var(E[self.p :], axis=0), 1e-8)
        self.h = np.repeat(np.log(var)[None, :], T, axis=0)
        self.A_chol = np.eye(n)
        self.mu_h = np.log(var)
        self.phi_h = np.full(n, self.rho_vol)
        self.tau_h = 0.05 * np.ones(n)
        self.Sigma = np.diag(var)
        self.Sigma_path = assemble_sigma_path(self.h, self.A_chol)
        self.tau_u = 1e-4
        self.tau_v = 1e-4
        self.tau_lam = 1e-4
        self.tau_del = 1e-4
        self.tau_gam = 1e-4
        self.tau_F = 1e-4
        self.tau_U = 1e-4 * np.ones(self.p)
        self.tau_V = 1e-4 * np.ones(self.p)
        self.sigma_z2 = 1.0

    def _rebuild_PW(self) -> tuple[np.ndarray, np.ndarray]:
        ridge = float(getattr(self, "ridge_delta", RIDGE_DELTA))
        _, W, P = network_from_latents(
            self.u, self.v, self.u_t, self.v_t,
            corr_boost=self.corr_boost, delta=ridge, hard=False,
        )
        return P, W

    def _sigma_inv(self) -> np.ndarray:
        try:
            return np.linalg.inv(self.Sigma)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(self.Sigma)

    def _ll(self, W=None) -> float:
        E = compute_residuals(self.Y, self.mu, self.W if W is None else W, self.U, self.V, self.p)
        return var_loglik_sv(E, self.h, self.A_chol, self.p)

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

        ll0 = self._ll()
        step = float(getattr(self, "mh_step", MH_STEP0))
        u_star = self.u + step * np.random.randn(*self.u.shape)
        v_star = self.v + step * np.random.randn(*self.v.shape)
        _, W_star, P_star = network_from_latents(
            u_star, v_star, self.u_t, self.v_t,
            corr_boost=self.corr_boost, delta=self.ridge_delta, hard=False,
        )
        ll1 = self._ll(W_star)
        prior0 = -0.5 / self.sigma_z2 * (np.sum(self.u**2) + np.sum(self.v**2))
        prior1 = -0.5 / self.sigma_z2 * (np.sum(u_star**2) + np.sum(v_star**2))
        self.n_mh_static += 1
        if np.log(np.random.rand()) < (ll1 - ll0 + prior1 - prior0):
            self.u, self.v = u_star, v_star
            self.P, self.W = P_star, W_star
            self.A = sample_A_bernoulli(self.P)
            self.n_mh_static_ok += 1
            self.mh_step = min(0.12, step * 1.06)
        else:
            self.P, self.W = self._rebuild_PW()
            self.mh_step = max(0.006, step * 0.90)

    def update_network_dynamic(self) -> None:
        n, T, r = self.n, self.T, self.rz
        ll0 = self._ll()
        u_t0, v_t0 = self.u_t.copy(), self.v_t.copy()
        W0, P0, A0 = self.W.copy(), self.P.copy(), self.A.copy()
        tau_u0, tau_v0 = self.tau_u, self.tau_v

        z_mean = self._z_mean()
        z = sample_probit_z(z_mean, self.A)
        r_dyn = z - (self.u @ self.v.T)[None, :, :] - self.corr_boost[None, :, :]
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
        self.P, self.W = self._rebuild_PW()
        ll1 = self._ll()
        self.n_mh_dyn += 1
        if np.log(np.random.rand()) < (ll1 - ll0):
            self.A = sample_A_bernoulli(self.P)
            self.n_mh_dyn_ok += 1
        else:
            self.u_t, self.v_t = u_t0, v_t0
            self.W, self.P, self.A = W0, P0, A0
            self.tau_u, self.tau_v = tau_u0, tau_v0

    def update_mean(self) -> None:
        Y, p = self.Y, self.p
        T, n = self.T, self.n
        ybar = Y.mean(axis=0)
        for _ in range(2):
            E = np.zeros_like(Y)
            for t in range(p, T):
                r = Y[t].copy()
                for i in range(1, p + 1):
                    B = self.U[i - 1, t - i] @ self.V[i - 1, t - i].T
                    r = r - apply_phi(self.W[t - i], B, Y[t - i] - self.mu[t - i])
                E[t] = r
            E[:p] = Y[:p]
            # Common local-linear trend on the residual after the idiosyncratic level
            common_obs = E - ybar - self.iota
            self.lam, self.vel = ffbs_llt(
                common_obs, self.F, self.Sigma_path, self.tau_lam, self.tau_del, self.kappa,
            )
            resid = common_obs
            gram = self.lam.T @ self.lam + 1e-3 * np.eye(self.r0)
            fhat = resid.T @ self.lam @ np.linalg.inv(gram)
            self.F = fhat + 0.005 * np.random.randn(*fhat.shape)
            self.tau_lam = _clip_tau(draw_ig(2.0 + 0.5 * self.r0 * (T - 1), 0.01 + 0.5 * np.sum(np.diff(self.lam, axis=0) ** 2)))
            self.tau_del = _clip_tau(draw_ig(2.0 + 0.5 * self.r0 * (T - 1), 0.01 + 0.5 * np.sum((self.vel[1:] - self.kappa * self.vel[:-1]) ** 2)))
            qf, rf = np.linalg.qr(self.F, mode="reduced")
            self.F = qf
            self.lam = self.lam @ rf.T
            self.vel = self.vel @ rf.T
            drift = self.vel @ self.F.T
            cap = 0.002
            scale = np.maximum(np.max(np.abs(drift), axis=1, keepdims=True) / cap, 1.0)
            self.vel = self.vel / scale
            common = ybar + self.lam @ self.F.T
            gap = E - common
            for j in range(n):
                vobs = np.clip(self.Sigma_path[p:, j, j], 1e-6, 1.0)
                path = ffbs_scalar_ar1(gap[p:, j], vobs, self.tau_iota[j], 0.999, 0.0)
                self.iota[p:, j] = path
                self.iota[:p, j] = path[0]
                innov = np.diff(self.iota[:, j])
                self.tau_iota[j] = float(np.clip(draw_ig(2.0 + 0.5 * (T - 1), 0.001 + 0.5 * np.sum(innov**2)), 1e-8, 5e-4))
            self.mu = common + self.iota

    def update_ar(self) -> None:
        Y, p, n, T = self.Y, self.p, self.n, self.T
        d = n * self.rb
        rho = 0.98
        for i in range(p):
            lag = i + 1
            Ts = T - lag
            obs = np.zeros((Ts, n))
            HU = np.zeros((Ts, n, d))
            HV = np.zeros((Ts, n, d))
            Sig = self.Sigma_path[lag:]
            for s in range(Ts):
                t = s + lag
                r = Y[t] - self.mu[t]
                for k in range(1, p + 1):
                    if k == lag:
                        continue
                    B = self.U[k - 1, t - k] @ self.V[k - 1, t - k].T
                    r = r - apply_phi(self.W[t - k], B, Y[t - k] - self.mu[t - k])
                obs[s] = r
                x = Y[t - lag] - self.mu[t - lag]
                W = self.W[t - lag]
                HU[s] = make_H_U(self.V[i, t - lag], W, x)
                HV[s] = make_H_V(self.U[i, t - lag], W, x)
            u_draw = ffbs_tv(obs, HU, Sig, self.tau_U[i], rho=rho)
            Umat = np.zeros((T, n, self.rb))
            Umat[lag:] = u_draw.reshape(Ts, self.rb, n).transpose(0, 2, 1)
            Umat[:lag] = Umat[lag]
            self.U[i] = Umat
            for s in range(Ts):
                t = s + lag
                x = Y[t - lag] - self.mu[t - lag]
                HV[s] = make_H_V(self.U[i, t - lag], self.W[t - lag], x)
            v_draw = ffbs_tv(obs, HV, Sig, self.tau_V[i], rho=rho)
            Vmat = np.zeros((T, n, self.rb))
            Vmat[lag:] = v_draw.reshape(Ts, self.rb, n).transpose(0, 2, 1)
            Vmat[:lag] = Vmat[lag]
            self.V[i] = Vmat
            self.tau_U[i] = _clip_tau(draw_ig(2.0 + 0.5 * d * (T - 1), 0.01 + 0.5 * ar_innov_ssq(self.U[i], rho)))
            self.tau_V[i] = _clip_tau(draw_ig(2.0 + 0.5 * d * (T - 1), 0.01 + 0.5 * ar_innov_ssq(self.V[i], rho)))
        self._stabilize_ar()

    def _stabilize_ar(self) -> None:
        n = self.n
        G = np.eye(n) + self.W[-1]
        for i in range(self.p):
            B = self.U[i, -1] @ self.V[i, -1].T
            rad = spectral_radius(G * B)
            if rad > 0.88:
                s = 0.88 / (rad + 1e-12)
                self.U[i] *= np.sqrt(s)
                self.V[i] *= np.sqrt(s)

    def update_sigma(self) -> None:
        E = compute_residuals(self.Y, self.mu, self.W, self.U, self.V, self.p)
        n, T, p = self.n, self.T, self.p
        Uorth = E @ self.A_chol.T
        for j in range(1, n):
            X = E[p:, :j] * np.exp(-0.5 * self.h[p:, j : j + 1])
            y = E[p:, j] * np.exp(-0.5 * self.h[p:, j])
            XtX = X.T @ X + 0.1 * np.eye(j)
            xty = X.T @ y
            cov = np.linalg.inv(XtX)
            a = np.random.multivariate_normal(cov @ xty, cov)
            self.A_chol[j, :j] = -a
        Uorth = E @ self.A_chol.T
        for j in range(n):
            ystar = np.log(Uorth[p:, j] ** 2 + 1e-4)
            resid = ystar - self.h[p:, j] + 1.2704
            logw = np.log(KSC_Q)[None, :] - 0.5 * np.log(KSC_V)[None, :] - 0.5 * (resid[:, None] - KSC_M[None, :]) ** 2 / KSC_V[None, :]
            logw = logw - logw.max(axis=1, keepdims=True)
            w = np.exp(logw)
            w = w / w.sum(axis=1, keepdims=True)
            s = np.array([np.random.choice(7, p=w[t]) for t in range(w.shape[0])])
            y_obs = ystar - KSC_M[s] + 1.2704
            v_obs = KSC_V[s]
            path = ffbs_scalar_ar1(y_obs, v_obs, self.tau_h[j], self.phi_h[j], self.mu_h[j])
            self.h[p:, j] = path
            self.h[:p, j] = path[0]
            innov = path[1:] - self.mu_h[j] - self.phi_h[j] * (path[:-1] - self.mu_h[j])
            self.tau_h[j] = float(np.clip(draw_ig(2.5 + 0.5 * (len(path) - 1), 0.025 + 0.5 * np.sum(innov**2)), 1e-4, 0.5))
            Xh = np.column_stack([np.ones(len(path) - 1), path[:-1]])
            yh = path[1:]
            beta = np.linalg.lstsq(Xh, yh, rcond=None)[0]
            phi = float(np.clip(beta[1], -0.98, 0.98))
            mu = float(beta[0] / max(1.0 - phi, 1e-3)) if abs(phi) < 0.98 else float(path.mean())
            self.phi_h[j] = phi
            self.mu_h[j] = float(np.clip(mu, -8.0, 2.0))
        self.Sigma_path = assemble_sigma_path(self.h, self.A_chol)
        self.Sigma = self.Sigma_path[-1]

    def snapshot(self) -> dict:
        t = self.T - 1
        B = [self.U[i, t] @ self.V[i, t].T for i in range(self.p)]
        Y = self.Y
        w1 = min(RECENT_WIN, t)
        w2 = min(SLOW_WIN, t)
        ret_fast = (Y[t] - Y[t - w1]) / max(w1, 1)
        ret_slow = (Y[t] - Y[t - w2]) / max(w2, 1)
        return {
            "mu": self.mu[t].copy(),
            "iota": self.iota[t].copy(),
            "W": self.W[t].copy(),
            "B": [b.copy() for b in B],
            "Y_tail": [self.Y[self.T - k].copy() for k in range(self.p, 0, -1)],
            "mu_tail": [self.mu[self.T - k].copy() for k in range(self.p, 0, -1)],
            "F": self.F.copy(),
            "lam": self.lam[t].copy(),
            "delta": self.vel[t].copy(),
            "kappa": self.kappa,
            "ybar": self.Y.mean(axis=0).copy(),
            "ret_fast": ret_fast,
            "ret_slow": ret_slow,
        }

    def fit(self, Y: np.ndarray, n_sweep=N_SWEEP, burn=BURN, thin=THIN, collect_A: bool = False) -> list[dict]:
        self.initialize(Y)
        draws = []
        A_acc = None
        Phi_acc = None
        n_keep = 0
        print(
            f"TVP-LR-NVAR-SV Gibbs {SAMPLER_VERSION}: T={self.T}, n={self.n}, "
            f"sweeps={n_sweep}, burn={burn}, kappa={self.kappa}"
        )
        for s in range(n_sweep):
            self.update_network_static()
            self.update_network_dynamic()
            self.update_mean()
            self.update_ar()
            self.update_sigma()
            if (s + 1) % 5 == 0 or s == 0:
                dens = float(self.P[-1].sum() / (self.n * (self.n - 1)))
                acc_s = self.n_mh_static_ok / max(self.n_mh_static, 1)
                acc_d = self.n_mh_dyn_ok / max(self.n_mh_dyn, 1)
                drift = float(np.max(np.abs(self.vel[-1] @ self.F.T)))
                gap = float(np.max(np.abs(self.mu[-1] - self.Y[-1])))
                print(
                    f"  sweep {s + 1}/{n_sweep}  E[density](T)={dens:.3f}  "
                    f"mean|h|={np.mean(np.abs(self.h[-1])):.2f}  "
                    f"MH static={acc_s:.2f} dyn={acc_d:.2f}  "
                    f"|F d|_inf={drift:.4f}  |mu-y|_inf={gap:.3f}  "
                    f"mh_step={getattr(self, 'mh_step' , 0):.3f}",
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


def mean_forecast_tvp(
    draw: dict,
    h: int,
    p: int,
    *,
    max_drift: float = 0.0016,
    max_cycle: float = 0.04,
    max_disp: float = 0.80,
    max_level_gap: float = 0.15,
    max_ret: float = 0.025,
) -> np.ndarray:
    """Open-loop log-price path via compounded daily returns.

    Daily drift mixes the factor velocity with each name's recent mean return.
    The network acts on the cycle x = y - mu; the implied return is
    drift + (x_new - x_old).
    """
    n = draw["mu"].shape[0]
    out = np.zeros((h, n))
    y_hist = [x.copy() for x in draw["Y_tail"]]
    mu_hist = [x.copy() for x in draw["mu_tail"]]
    last_y = y_hist[-1]
    mu = last_y + np.clip(np.asarray(draw["mu"], dtype=float) - last_y, -max_level_gap, max_level_gap)
    W = draw["W"]
    B = draw["B"]
    F = np.asarray(draw["F"])
    vel = np.asarray(draw["delta"], dtype=float)
    factor_d = np.clip(F @ vel, -max_drift, max_drift)
    ret_fast = np.asarray(draw.get("ret_fast", np.zeros(n)), dtype=float)
    ret_slow = np.asarray(draw.get("ret_slow", np.zeros(n)), dtype=float)
    stock_d = 0.65 * ret_fast + 0.35 * ret_slow
    mkt = float(np.mean(stock_d))
    drift = 0.45 * stock_d + 0.25 * mkt + 0.30 * factor_d
    drift = np.clip(drift, -max_drift, max_drift)
    kappa = float(draw.get("kappa", KAPPA_DELTA))
    y = last_y.copy()
    for t in range(h):
        x_lags = [y_hist[-i] - mu_hist[-i] for i in range(1, p + 1)]
        x_old = x_lags[0]
        x_new = np.zeros(n)
        for i in range(p):
            x_new = x_new + apply_phi(W, B[i], x_lags[i])
        x_new = np.clip(x_new, -max_cycle, max_cycle)
        ret = drift + (x_new - x_old)
        ret = np.clip(ret, -max_ret, max_ret)
        y = y + ret
        mu = mu + drift
        drift = kappa * drift
        y = last_y + np.clip(y - last_y, -max_disp, max_disp)
        out[t] = y
        y_hist.append(y.copy())
        mu_hist.append(mu.copy())
        y_hist = y_hist[-p:]
        mu_hist = mu_hist[-p:]
    return out


def main() -> None:
    np.random.seed(SEED)
    prices = pd.read_csv(DATA_DIR / "djia_adj_close_wide_10y.csv", parse_dates=["Date"], index_col="Date")
    prices = prices.sort_index().ffill().dropna(how="all")
    tickers = prices.columns.tolist()
    train_prices = prices.iloc[:-HORIZON]
    test_prices = prices.iloc[-HORIZON:]
    Y = np.log(train_prices.values)
    print(f"Train log-prices {Y.shape}  test {test_prices.index[0].date()} to {test_prices.index[-1].date()}")

    model = TVPLRNVARSV()
    draws = model.fit(Y, collect_A=True)
    if model.A_mean is not None:
        np.savez_compressed(
            DATA_DIR / "tvplrnvarsv_A_mean_train.npz",
            A_mean=model.A_mean,
            Phi1=model.Phi_mean[0] if model.Phi_mean else np.array([]),
            Phi2=model.Phi_mean[1] if model.Phi_mean else np.array([]),
            dates=train_prices.index.strftime("%Y-%m-%d").to_numpy(),
        )
    last_train_price = train_prices.iloc[-1].values
    train_ret = np.diff(Y, axis=0)
    if model.Phi_mean is not None:
        pred_price = forecast_from_phi(
            train_ret, last_train_price, model.Phi_mean[0][-1], model.Phi_mean[1][-1], HORIZON
        )
    else:
        last = Y[-1]
        safe = []
        for d in draws:
            path = mean_forecast_tvp(d, HORIZON, LAG_P)
            if np.all(np.isfinite(path)) and np.max(np.abs(path - last)) < 1.5:
                safe.append(path)
        if not safe:
            raise RuntimeError("all forecast draws left the log-price scale")
        mean_log = np.mean(np.stack(safe, axis=0), axis=0)
        pred_price = np.exp(mean_log)
        print(f"Using {len(safe)}/{len(draws)} finite forecast draws")

    prev = DATA_DIR / "djia_6m_tvplrnvarsv_mean.csv"
    v2 = DATA_DIR / "djia_6m_tvplrnvarsv_v2_mean.csv"
    if prev.exists() and not v2.exists():
        prev.replace(v2)
    out = pd.DataFrame(pred_price, index=test_prices.index, columns=tickers)
    out.index.name = "Date"
    out.to_csv(DATA_DIR / "djia_6m_tvplrnvarsv_mean.csv")

    actual = test_prices.values
    mae = np.mean(np.abs(pred_price - actual), axis=0)
    rmse = np.sqrt(np.mean((pred_price - actual) ** 2, axis=0))
    mape = np.mean(np.abs(pred_price - actual) / np.maximum(np.abs(actual), 1e-8), axis=0) * 100
    print(f"TVP-LR-NVAR-SV  MAPE {mape.mean():.2f}%  MAE {mae.mean():.2f}  RMSE {rmse.mean():.2f}")
    meta = {
        "sampler_version": SAMPLER_VERSION,
        "observation": "log-prices",
        "trend": "local-linear common factor plus idiosyncratic local level",
        "forecast": "Minnesota 5% premium + 1/3 name means + frozen origin network VAR",
        "covariance": "triangular MSV (KSC mixture)",
        "avg_mae": float(mae.mean()),
        "avg_rmse": float(rmse.mean()),
        "avg_mape": float(mape.mean()),
        "n_draws": len(draws),
        "p": LAG_P,
        "n_sweep": N_SWEEP,
        "burn": BURN,
        "kappa_delta": KAPPA_DELTA,
        "mh_static_accept": model.n_mh_static_ok / max(model.n_mh_static, 1),
        "mh_dyn_accept": model.n_mh_dyn_ok / max(model.n_mh_dyn, 1),
    }
    (DATA_DIR / "tvplrnvarsv_fit_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Saved data/djia_6m_tvplrnvarsv_mean.csv")


if __name__ == "__main__":
    main()
