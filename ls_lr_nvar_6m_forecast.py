#!/usr/bin/env python3
"""
Bayesian locally stationary low-rank Network VAR (LS-LR-NVAR), revised Gibbs.

Observation equation (log-returns), p lags:

  y_t = mu_t + sum_i [ (I + W_{t-i}) odot B_{i,t-i} ] (y_{t-i} - mu_{t-i}) + e_t
  e_t ~ N(0, Sigma)

  mu_t = F lambda_t + F_t gamma_t
  B_{i,t} = U_{i,t} V_{i,t}'

Revisions relative to the first draft:
  * Kalman maps H^U, H^V are the n x n r_B row-decoupled operators
    (never n^2 Kronecker / diag(w) identities).
  * Local-mean and network dynamic factors are stationary AR(1) with rho=0.98,
    not unconstrained random walks.
  * W is a ridge row-normalisation of the probit probabilities Phi(mu),
    not a hard sign(mu) graph and not A / d_i.
  * Named adjacency A_t is Bernoulli(Phi(mu)); the VAR likelihood MH ratio
    contains no auxiliary phi(z-mu) terms.
  * Dynamic latents are updated by FFBS, never by a joint T-dimensional
    random walk. A global MH using the VAR likelihood accepts/rejects the
    FFBS draw.
  * Correlation enters only as a prior tilt on mu, not as a hard floor on A.

Holdout: train through 2026-02-05, evaluate 2026-02-06 to 2026-08-07 (126 days).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import solve
from scipy.stats import invwishart, norm, truncnorm

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent / "data"
HORIZON = 126
LAG_P = 2
R0 = 2
RMU = 1
RB = 2
RZ = 2
N_SWEEP = 24
BURN = 12
THIN = 2
SEED = 0
CORR_EDGE = 0.30
RHO_MEAN = 0.98
RHO_NET = 0.98
RIDGE_DELTA = 1.0
FLOOR_BOOST = 0.35
PROBIT_INTERCEPT = -0.4
SAMPLER_VERSION = "revised-v2"


def ridge_row_normalize(P: np.ndarray, delta: float = RIDGE_DELTA) -> np.ndarray:
    """W_ij = P_ij / (sum_ell P_iell + delta). Isolated rows stay near zero."""
    W = np.asarray(P, dtype=np.float64).copy()
    if W.ndim == 2:
        np.fill_diagonal(W, 0.0)
        deg = W.sum(axis=1, keepdims=True)
        return W / (deg + float(delta))
    W[:, np.arange(W.shape[1]), np.arange(W.shape[1])] = 0.0
    deg = W.sum(axis=2, keepdims=True)
    return W / (deg + float(delta))


def row_normalize(A: np.ndarray, delta: float = RIDGE_DELTA) -> np.ndarray:
    """Backwards-compatible name: ridge row-normalisation."""
    return ridge_row_normalize(A, delta=delta)


def apply_phi(W: np.ndarray, B: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Apply Phi x with Phi = (I + W) odot B.

    The identity is inside the Hadamard product, so the diagonal of Phi equals the
    diagonal of B. Scaling B by c scales Phi by c, and c = 0 yields the zero map.
    """
    n = W.shape[0]
    G = np.eye(n) + W
    return (G * B) @ x


def make_H_U(V: np.ndarray, W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Map vec(U) (column-stacked) to ((I+W) odot (U V')) x.

    Row k of the operator is U_{k,.} · c_k with
    c_k = sum_j G_{kj} x_j V_{j,.}.  Hence H is n x (n r), block-ℓ = diag(c^{(ℓ)}).
    """
    n, r = V.shape
    G = np.eye(n) + W
    H = np.zeros((n, n * r))
    gx = G * x[None, :]
    for ell in range(r):
        c = gx @ V[:, ell]
        H[:, ell * n : (ell + 1) * n] = np.diag(c)
    return H


def make_H_V(U: np.ndarray, W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Map vec(V) (column-stacked) to ((I+W) odot (U V')) x."""
    n, r = U.shape
    G = np.eye(n) + W
    H = np.zeros((n, n * r))
    for ell in range(r):
        H[:, ell * n : (ell + 1) * n] = (G * U[:, ell][:, None]) * x[None, :]
    return H


def ffbs(obs: np.ndarray, H: np.ndarray, Sigma: np.ndarray, q2: float, rho: float = 1.0) -> np.ndarray:
    """Durbin–Koopman simulation smoother for x_t = rho x_{t-1} + eta_t."""
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
    for t in range(T):
        mp[t] = rho * mt
        Ppt = (rho * rho) * Pt + Q
        Pp[t] = Ppt
        Ht = H[t]
        S = Ht @ Ppt @ Ht.T + Sigma
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


def _mvn(mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    z = np.random.randn(mean.shape[0])
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        eigval, eigvec = np.linalg.eigh(cov)
        eigval = np.clip(eigval, 1e-10, None)
        L = eigvec * np.sqrt(eigval)
    return mean + L @ z


def ffbs_const_H(
    obs: np.ndarray,
    Hmat: np.ndarray,
    Sigma: np.ndarray,
    q2: float,
    rho: float = 1.0,
) -> np.ndarray:
    T = obs.shape[0]
    H = np.repeat(Hmat[None, :, :], T, axis=0)
    return ffbs(obs, H, Sigma, q2, rho=rho)


def _clip_tau(x: float, lo: float = 1e-8, hi: float = 5e-4) -> float:
    return float(np.clip(x, lo, hi))


def spectral_radius(M: np.ndarray) -> float:
    try:
        return float(np.max(np.abs(np.linalg.eigvals(M))))
    except np.linalg.LinAlgError:
        return 10.0


def draw_ig(a: float, b: float) -> float:
    a = max(a, 1.01)
    b = max(b, 1e-8)
    return float(1.0 / np.random.gamma(a, 1.0 / b))


def sample_trunc_norm(mean: np.ndarray, lower: bool) -> np.ndarray:
    if lower:
        a = (0.0 - mean) / 1.0
        b = np.full_like(mean, np.inf)
    else:
        a = np.full_like(mean, -np.inf)
        b = (0.0 - mean) / 1.0
    return truncnorm.rvs(a, b, loc=mean, scale=1.0)


def ar1_path(T: int, dim: int, rho: float, scale: float) -> np.ndarray:
    x = np.zeros((T, dim))
    x[0] = scale * np.random.randn(dim)
    for t in range(1, T):
        x[t] = rho * x[t - 1] + scale * np.random.randn(dim)
    return x


def latent_mean(
    u: np.ndarray,
    v: np.ndarray,
    u_t: np.ndarray,
    v_t: np.ndarray,
    corr_boost: np.ndarray | None = None,
    intercept: float = PROBIT_INTERCEPT,
) -> np.ndarray:
    mu = intercept + (u @ v.T)[None, :, :] + np.einsum("tir,tjr->tij", u_t, v_t)
    if corr_boost is not None:
        mu = mu + corr_boost[None, :, :]
    n = mu.shape[-1]
    mu[:, np.arange(n), np.arange(n)] = -8.0
    return mu


def network_from_latents(
    u: np.ndarray,
    v: np.ndarray,
    u_t: np.ndarray,
    v_t: np.ndarray,
    corr_boost: np.ndarray | None = None,
    intercept: float = PROBIT_INTERCEPT,
    delta: float = RIDGE_DELTA,
    hard: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (A_or_P, W, P). A is 1{P>=0.5} if hard else P itself."""
    mu = latent_mean(u, v, u_t, v_t, corr_boost=corr_boost, intercept=intercept)
    P = norm.cdf(mu)
    n = P.shape[-1]
    P[:, np.arange(n), np.arange(n)] = 0.0
    W = ridge_row_normalize(P, delta=delta)
    A = (P >= 0.5).astype(np.float64) if hard else P.copy()
    A[:, np.arange(n), np.arange(n)] = 0.0
    return A, W, P


def sample_A_bernoulli(P: np.ndarray) -> np.ndarray:
    A = (np.random.rand(*P.shape) < P).astype(np.float64)
    n = A.shape[-1]
    A[:, np.arange(n), np.arange(n)] = 0.0
    return A


def sample_probit_z(mean: np.ndarray, A: np.ndarray) -> np.ndarray:
    n = A.shape[-1]
    eye = np.eye(n, dtype=bool)
    pos = (A > 0.5) & ~eye
    neg = (~(A > 0.5)) & ~eye
    z = np.zeros_like(mean)
    if pos.any():
        z[pos] = sample_trunc_norm(mean[pos], True)
    if neg.any():
        z[neg] = sample_trunc_norm(mean[neg], False)
    return z


def compute_residuals(
    Y: np.ndarray,
    mu: np.ndarray,
    W: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    p: int,
) -> np.ndarray:
    T, n = Y.shape
    E = np.zeros_like(Y)
    for t in range(p, T):
        r = Y[t] - mu[t]
        for i in range(1, p + 1):
            B = U[i - 1, t - i] @ V[i - 1, t - i].T
            r = r - apply_phi(W[t - i], B, Y[t - i] - mu[t - i])
        E[t] = r
    return E


def var_loglik(E: np.ndarray, Sigma_inv: np.ndarray, p: int) -> float:
    R = E[p:]
    return -0.5 * float(np.sum((R @ Sigma_inv) * R))


def ar_innov_ssq(x: np.ndarray, rho: float) -> float:
    return float(np.sum((x[1:] - rho * x[:-1]) ** 2))


class LSLRNVAR:
    def __init__(self, p=LAG_P, r0=R0, rmu=RMU, rb=RB, rz=RZ):
        self.p = p
        self.r0 = r0
        self.rmu = rmu
        self.rb = rb
        self.rz = rz
        self.rho_mean = RHO_MEAN
        self.rho_net = RHO_NET
        self.delta = RIDGE_DELTA
        self.n_mh_static = 0
        self.n_mh_static_ok = 0
        self.n_mh_dyn = 0
        self.n_mh_dyn_ok = 0

    def initialize(self, Y: np.ndarray) -> None:
        T, n = Y.shape
        self.T, self.n = T, n
        self.Y = Y
        C = np.corrcoef(Y.T)
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
        self.corr_boost = FLOOR_BOOST * A0
        self.P, self.W = self._rebuild_PW()
        self.A = (self.P >= 0.5).astype(np.float64)
        self.A[:, np.arange(n), np.arange(n)] = 0.0

        Yc = Y - Y.mean(axis=0)
        Uy, sy, Vty = np.linalg.svd(Yc, full_matrices=False)
        self.F = (Vty[: self.r0, :].T * sy[: self.r0] / np.sqrt(T)).copy()
        self.lam = (Uy[:, : self.r0] * np.sqrt(T)).copy()
        self.Ft = np.zeros((T, n, self.rmu))
        self.gam = np.zeros((T, self.rmu))
        self.mu = self.lam @ self.F.T

        self.U = np.zeros((self.p, T, n, self.rb))
        self.V = np.zeros((self.p, T, n, self.rb))
        for i in range(self.p):
            self.U[i, :, : self.rb, :] = 0.04 * np.eye(self.rb)
            self.V[i, :, : self.rb, :] = np.eye(self.rb)
            self.U[i] += 0.01 * np.random.randn(T, n, self.rb)
            self.V[i] += 0.01 * np.random.randn(T, n, self.rb)

        E = compute_residuals(Y, self.mu, self.W, self.U, self.V, self.p)
        S = E[self.p :].T @ E[self.p :] / max(T - self.p, 1)
        self.Sigma = S + 1e-4 * np.eye(n)
        self.tau_u = 1e-4
        self.tau_v = 1e-4
        self.tau_lam = 1e-4
        self.tau_gam = 1e-4
        self.tau_F = 1e-4
        self.tau_U = 1e-4 * np.ones(self.p)
        self.tau_V = 1e-4 * np.ones(self.p)
        self.sigma_z2 = 1.0

    def _sigma_inv(self) -> np.ndarray:
        try:
            return np.linalg.inv(self.Sigma)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(self.Sigma)

    def _z_mean(self) -> np.ndarray:
        return latent_mean(self.u, self.v, self.u_t, self.v_t, corr_boost=self.corr_boost)

    def _rebuild_PW(self) -> tuple[np.ndarray, np.ndarray]:
        _, W, P = network_from_latents(
            self.u, self.v, self.u_t, self.v_t,
            corr_boost=self.corr_boost, delta=self.delta, hard=False,
        )
        return P, W

    def _set_network_from_latents(self) -> None:
        self.P, self.W = self._rebuild_PW()
        self.A = sample_A_bernoulli(self.P)

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

        # MH on static factors: proposal is symmetric RW; target is VAR lik + N(0,1) prior.
        # Auxiliary z densities cancel and are not in the ratio.
        Sigma_inv = self._sigma_inv()
        E0 = compute_residuals(self.Y, self.mu, self.W, self.U, self.V, self.p)
        ll0 = var_loglik(E0, Sigma_inv, self.p)
        kappa = 0.04
        u_star = self.u + kappa * np.random.randn(*self.u.shape)
        v_star = self.v + kappa * np.random.randn(*self.v.shape)
        _, W_star, P_star = network_from_latents(
            u_star, v_star, self.u_t, self.v_t,
            corr_boost=self.corr_boost, delta=self.delta, hard=False,
        )
        E1 = compute_residuals(self.Y, self.mu, W_star, self.U, self.V, self.p)
        ll1 = var_loglik(E1, Sigma_inv, self.p)
        prior0 = -0.5 / self.sigma_z2 * (np.sum(self.u**2) + np.sum(self.v**2))
        prior1 = -0.5 / self.sigma_z2 * (np.sum(u_star**2) + np.sum(v_star**2))
        self.n_mh_static += 1
        if np.log(np.random.rand()) < (ll1 - ll0 + prior1 - prior0):
            self.u, self.v = u_star, v_star
            self.P, self.W = P_star, W_star
            self.A = sample_A_bernoulli(self.P)
            self.n_mh_static_ok += 1
        else:
            self.P, self.W = self._rebuild_PW()

    def update_network_dynamic(self) -> None:
        n, T, r = self.n, self.T, self.rz
        Sigma_inv = self._sigma_inv()
        ll0 = var_loglik(compute_residuals(self.Y, self.mu, self.W, self.U, self.V, self.p), Sigma_inv, self.p)
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
        ll1 = var_loglik(compute_residuals(self.Y, self.mu, self.W, self.U, self.V, self.p), Sigma_inv, self.p)
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
        Sigma = self.Sigma
        rho = self.rho_mean
        for _ in range(2):
            E = np.zeros_like(Y)
            for t in range(p, T):
                r = Y[t].copy()
                for i in range(1, p + 1):
                    B = self.U[i - 1, t - i] @ self.V[i - 1, t - i].T
                    r = r - apply_phi(self.W[t - i], B, Y[t - i] - self.mu[t - i])
                E[t] = r
            dyn = np.einsum("tnr,tr->tn", self.Ft, self.gam)
            estat = E - dyn
            estat[:p] = 0.0
            self.lam = ffbs_const_H(estat, self.F, Sigma, self.tau_lam, rho=rho)
            Lam = self.lam[p:].T
            Estat = estat[p:].T
            Sll = Lam @ Lam.T + 1e-4 * np.eye(self.r0)
            try:
                Sinv = np.linalg.inv(self.Sigma)
            except np.linalg.LinAlgError:
                Sinv = np.linalg.pinv(self.Sigma)
            prec = np.kron(Sll, Sinv) + np.eye(n * self.r0) / 4.0
            rhs = (Sinv @ Estat @ Lam.T).ravel(order="F")
            cov = np.linalg.inv(prec + 1e-8 * np.eye(prec.shape[0]))
            mean = cov @ rhs
            vecF = np.random.multivariate_normal(mean, cov)
            self.F = vecF.reshape((n, self.r0), order="F")
            self.tau_lam = _clip_tau(draw_ig(2.0 + 0.5 * self.r0 * (T - 1), 0.01 + 0.5 * ar_innov_ssq(self.lam, rho)))

            edyn = E - self.lam @ self.F.T
            edyn[:p] = 0.0
            Hgam = self.Ft
            self.gam = ffbs(edyn, Hgam, Sigma, self.tau_gam, rho=rho)
            d = n * self.rmu
            Hft = np.zeros((T, n, n * self.rmu))
            for t in range(T):
                for ell in range(self.rmu):
                    Hft[t, :, ell * n : (ell + 1) * n] = self.gam[t, ell] * np.eye(n)
            ft = ffbs(edyn, Hft, Sigma, self.tau_F, rho=rho)
            self.Ft = ft.reshape(T, self.rmu, n).transpose(0, 2, 1)
            self.tau_gam = _clip_tau(draw_ig(2.0 + 0.5 * self.rmu * (T - 1), 0.01 + 0.5 * ar_innov_ssq(self.gam, rho)))
            self.tau_F = _clip_tau(draw_ig(2.0 + 0.5 * d * (T - 1), 0.01 + 0.5 * ar_innov_ssq(self.Ft, rho)))
            self.mu = self.lam @ self.F.T + np.einsum("tnr,tr->tn", self.Ft, self.gam)

    def update_ar(self) -> None:
        Y, p, n, T = self.Y, self.p, self.n, self.T
        Sigma = self.Sigma
        d = n * self.rb
        rho = self.rho_mean
        for i in range(p):
            lag = i + 1
            Ts = T - lag
            obs = np.zeros((Ts, n))
            HU = np.zeros((Ts, n, d))
            HV = np.zeros((Ts, n, d))
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
            u_draw = ffbs(obs, HU, Sigma, self.tau_U[i], rho=rho)
            Umat = np.zeros((T, n, self.rb))
            Umat[lag:] = u_draw.reshape(Ts, self.rb, n).transpose(0, 2, 1)
            Umat[:lag] = Umat[lag]
            self.U[i] = Umat
            for s in range(Ts):
                t = s + lag
                x = Y[t - lag] - self.mu[t - lag]
                HV[s] = make_H_V(self.U[i, t - lag], self.W[t - lag], x)
            v_draw = ffbs(obs, HV, Sigma, self.tau_V[i], rho=rho)
            Vmat = np.zeros((T, n, self.rb))
            Vmat[lag:] = v_draw.reshape(Ts, self.rb, n).transpose(0, 2, 1)
            Vmat[:lag] = Vmat[lag]
            self.V[i] = Vmat
            for s in range(Ts):
                t = s + lag
                x = Y[t - lag] - self.mu[t - lag]
                HU[s] = make_H_U(self.V[i, t - lag], self.W[t - lag], x)
            self.tau_U[i] = _clip_tau(draw_ig(2.0 + 0.5 * d * (T - 1), 0.01 + 0.5 * ar_innov_ssq(self.U[i], rho)))
            self.tau_V[i] = _clip_tau(draw_ig(2.0 + 0.5 * d * (T - 1), 0.01 + 0.5 * ar_innov_ssq(self.V[i], rho)))
        self._stabilize_ar()

    def _stabilize_ar(self) -> None:
        n = self.n
        G = np.eye(n) + self.W[-1]
        for i in range(self.p):
            B = self.U[i, -1] @ self.V[i, -1].T
            rad = spectral_radius(G * B)
            if rad > 0.92:
                s = 0.92 / (rad + 1e-12)
                self.U[i] *= np.sqrt(s)
                self.V[i] *= np.sqrt(s)

    def update_sigma(self) -> None:
        E = compute_residuals(self.Y, self.mu, self.W, self.U, self.V, self.p)
        S = E[self.p :].T @ E[self.p :]
        nu0 = self.n + 2
        S0 = np.eye(self.n)
        self.Sigma = np.asarray(invwishart.rvs(df=nu0 + (self.T - self.p), scale=S0 + S))
        self.Sigma = 0.5 * (self.Sigma + self.Sigma.T) + 1e-6 * np.eye(self.n)

    def snapshot(self) -> dict:
        t = self.T - 1
        B = [self.U[i, t] @ self.V[i, t].T for i in range(self.p)]
        return {
            "mu": self.mu[t].copy(),
            "W": self.W[t].copy(),
            "B": [b.copy() for b in B],
            "Y_tail": [self.Y[self.T - k].copy() for k in range(self.p, 0, -1)],
            "mu_tail": [self.mu[self.T - k].copy() for k in range(self.p, 0, -1)],
            "F": self.F.copy(),
            "lam": self.lam[t].copy(),
            "Ft": self.Ft[t].copy(),
            "gam": self.gam[t].copy(),
            "rho": self.rho_mean,
        }

    def current_phi_path(self) -> list[np.ndarray]:
        eye = np.eye(self.n)
        G = eye + self.W
        out = []
        for i in range(self.p):
            B = np.einsum("tir,tjr->tij", self.U[i], self.V[i])
            out.append(G * B)
        return out

    def fit(self, Y: np.ndarray, n_sweep=N_SWEEP, burn=BURN, thin=THIN, collect_A: bool = False) -> list[dict]:
        self.initialize(Y)
        draws = []
        A_acc = None
        Phi_acc = None
        n_keep = 0
        print(
            f"LS-LR-NVAR Gibbs {SAMPLER_VERSION}: T={self.T}, n={self.n}, "
            f"sweeps={n_sweep}, burn={burn}, rho={self.rho_mean}, delta={self.delta}"
        )
        for s in range(n_sweep):
            self.update_network_static()
            self.update_network_dynamic()
            self.update_mean()
            self.update_ar()
            self.update_sigma()
            if (s + 1) % 10 == 0 or s == 0:
                dens = float(self.P[-1].sum() / (self.n * (self.n - 1)))
                acc_s = self.n_mh_static_ok / max(self.n_mh_static, 1)
                acc_d = self.n_mh_dyn_ok / max(self.n_mh_dyn, 1)
                print(
                    f"  sweep {s + 1}/{n_sweep}  E[density](T)={dens:.3f}  "
                    f"tau_U={self.tau_U[0]:.4f}  MH static={acc_s:.2f} dyn={acc_d:.2f}"
                )
            if s >= burn and (s - burn) % thin == 0:
                draws.append(self.snapshot())
                if collect_A:
                    A_acc = self.P.copy() if A_acc is None else A_acc + self.P
                    phis = self.current_phi_path()
                    if Phi_acc is None:
                        Phi_acc = [p.copy() for p in phis]
                    else:
                        for k, p in enumerate(phis):
                            Phi_acc[k] += p
                    n_keep += 1
        if collect_A and n_keep:
            self.A_mean = A_acc / n_keep
            self.Phi_mean = [p / n_keep for p in Phi_acc]
        else:
            self.A_mean = None
            self.Phi_mean = None
        print(
            f"  kept {len(draws)} snapshots  "
            f"MH accept static={self.n_mh_static_ok}/{self.n_mh_static} "
            f"dyn={self.n_mh_dyn_ok}/{self.n_mh_dyn}"
        )
        return draws


from hierarchical_forecast import forecast_from_phi, minnesota_intercept, scale_companion, iterate_returns


def mean_forecast(draw: dict, h: int, p: int, train_ret: np.ndarray | None = None) -> np.ndarray:
    """Open-loop returns: Minnesota intercept + frozen origin network VAR."""
    n = draw["mu"].shape[0]
    B1 = np.asarray(draw["W"] * 0 + (np.eye(n) + draw["W"]) * draw["B"][0], dtype=float)
    B2 = (np.eye(n) + draw["W"]) * draw["B"][1]
    y1 = np.asarray(draw["Y_tail"][-1], dtype=float)
    y2 = np.asarray(draw["Y_tail"][-2], dtype=float)
    if train_ret is not None:
        mu = minnesota_intercept(train_ret)
    else:
        mu = np.clip(np.asarray(draw["mu"], dtype=float), -0.002, 0.002)
    B1, B2 = scale_companion(B1, B2, 0.85)
    return iterate_returns(mu, B1, B2, y1, y2, h)


def price_path(last_price: np.ndarray, ret: np.ndarray) -> np.ndarray:
    return last_price * np.exp(np.cumsum(ret, axis=0))


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
    print(f"Train returns {train_ret.shape}  test {test_prices.index[0].date()} to {test_prices.index[-1].date()}")

    model = LSLRNVAR()
    draws = model.fit(train_ret, collect_A=True)
    if model.A_mean is not None:
        ret_dates = prices.index[1:-HORIZON]
        np.savez_compressed(
            DATA_DIR / "lslrnvar_A_mean_train_v2.npz",
            A_mean=model.A_mean,
            Phi1=model.Phi_mean[0] if model.Phi_mean else np.array([]),
            Phi2=model.Phi_mean[1] if model.Phi_mean else np.array([]),
            dates=ret_dates.strftime("%Y-%m-%d").to_numpy(),
        )
    if model.Phi_mean is not None:
        pred_price = forecast_from_phi(train_ret, last_train_price, model.Phi_mean[0][-1], model.Phi_mean[1][-1], HORIZON)
    else:
        paths = np.stack([mean_forecast(d, HORIZON, LAG_P, train_ret) for d in draws], axis=0)
        mean_ret = np.nan_to_num(paths.mean(axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        pred_price = price_path(last_train_price, mean_ret)

    out = pd.DataFrame(pred_price, index=test_prices.index, columns=tickers)
    out.index.name = "Date"
    out.to_csv(DATA_DIR / "djia_6m_lslrnvar_revised_v2_mean.csv")

    actual = test_prices.values
    mae = np.mean(np.abs(pred_price - actual), axis=0)
    rmse = np.sqrt(np.mean((pred_price - actual) ** 2, axis=0))
    mape = np.mean(np.abs(pred_price - actual) / np.maximum(np.abs(actual), 1e-8), axis=0) * 100
    print(f"LS-LR-NVAR  MAPE {mape.mean():.2f}%  MAE {mae.mean():.2f}  RMSE {rmse.mean():.2f}")
    acc_s = model.n_mh_static_ok / max(model.n_mh_static, 1)
    acc_d = model.n_mh_dyn_ok / max(model.n_mh_dyn, 1)
    meta = {
        "sampler_version": SAMPLER_VERSION,
        "avg_mae": float(mae.mean()),
        "avg_rmse": float(rmse.mean()),
        "avg_mape": float(mape.mean()),
        "n_draws": len(draws),
        "p": LAG_P,
        "ranks": {"r0": R0, "rmu": RMU, "rb": RB, "rz": RZ},
        "n_sweep": N_SWEEP,
        "burn": BURN,
        "rho_mean": RHO_MEAN,
        "rho_net": RHO_NET,
        "ridge_delta": RIDGE_DELTA,
        "floor_boost": FLOOR_BOOST,
        "mh_static_accept": acc_s,
        "mh_dyn_accept": acc_d,
    }
    (DATA_DIR / "lslrnvar_fit_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Saved data/djia_6m_lslrnvar_revised_v2_mean.csv")


if __name__ == "__main__":
    main()
