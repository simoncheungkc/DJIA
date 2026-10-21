#!/usr/bin/env python3
"""
Simulate y from a time-varying blocked random network VAR.

Network: directed stochastic block model (SBM) with K communities.
Time: n_regimes SBM graphs are drawn at equally spaced knots. By default
those anchors are blended with a cosine (smoothstep) weight so A_t and W_t
change smoothly between knots. Set transition="step" for the old
piecewise-constant path.

Observation equation matches LS-LR-NVAR, plus a contemporaneous
network shock so pairwise |corr| tracks the graph (needed for the
LS-LR-NVAR correlation floor):

    y_t = mu + sum_k [(I + W_{t-k}) odot B_k] (y_{t-k} - mu) + e_t

    W_t = row-normalisation of A_soft
    e_t ~ N(0, sigma^2 R_t)

    R_t[i,j] ≈ contemp_rho * max(A_soft_ij, A_soft_ji)  (PSD projection)

Sparsity is controlled by --density: expected directed off-diagonal
edge probability. Within-block edges are `block_ratio` times more
likely than between-block edges, with both scaled to hit `density`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"


def sbm_probs(n: int, labels: np.ndarray, density: float, block_ratio: float) -> np.ndarray:
    """Pairwise edge probabilities with E[A_ij | i!=j] = density."""
    if not (0.0 < density < 1.0):
        raise ValueError("density must be in (0, 1)")
    if block_ratio < 1.0:
        raise ValueError("block_ratio must be >= 1 (within >= between)")
    same = labels[:, None] == labels[None, :]
    np.fill_diagonal(same, False)
    n_off = n * (n - 1)
    n_in = int(same.sum())
    n_out = n_off - n_in
    # n_in * r * p_out + n_out * p_out = density * n_off
    p_out = (density * n_off) / max(n_in * block_ratio + n_out, 1)
    p_in = block_ratio * p_out
    if p_in > 0.999:
        p_in = 0.999
        p_out = (density * n_off - n_in * p_in) / max(n_out, 1)
    p_out = float(np.clip(p_out, 1e-6, 0.999))
    p_in = float(np.clip(p_in, p_out, 0.999))
    P = np.where(labels[:, None] == labels[None, :], p_in, p_out)
    np.fill_diagonal(P, 0.0)
    return P


def draw_adj(P: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    A = (rng.random(P.shape) < P).astype(np.float64)
    np.fill_diagonal(A, 0.0)
    return A


def row_normalize(A: np.ndarray) -> np.ndarray:
    deg = A.sum(axis=1, keepdims=True)
    return np.divide(A, deg, out=np.zeros_like(A), where=deg > 0)


def companion_radius(phis: list[np.ndarray]) -> float:
    n = phis[0].shape[0]
    p = len(phis)
    C = np.zeros((p * n, p * n))
    for k, Ak in enumerate(phis):
        C[:n, k * n : (k + 1) * n] = Ak
    if p > 1:
        C[n:, :-n] = np.eye((p - 1) * n)
    return float(np.max(np.abs(np.linalg.eigvals(C))))


def make_B(
    n: int,
    p: int,
    rng: np.random.Generator,
    scale: float = 0.22,
    own_lag: float | None = None,
    net_lag: float | None = None,
) -> list[np.ndarray]:
    """
    Lag loadings so Phi = (I+W) odot B = own_lag * I + net_lag * W (lag 1),
    with geometric decay on later lags. Strong own_lag makes y predictable;
    net_lag is the row-normalised network channel.
    """
    own = 1.15 * scale if own_lag is None else float(own_lag)
    net = scale if net_lag is None else float(net_lag)
    out = []
    for k in range(p):
        decay = 0.45**k
        B = (net * decay) * np.ones((n, n))
        np.fill_diagonal(B, own * decay)
        B = B + 0.008 * rng.normal(size=(n, n))
        B = 0.5 * (B + B.T)
        np.fill_diagonal(B, own * decay)
        out.append(B)
    return out


def cosine_blend(u: float) -> float:
    """Smoothstep on [0, 1]: 0 at the left knot, 1 at the right, flat derivative at both."""
    u = float(np.clip(u, 0.0, 1.0))
    return 0.5 - 0.5 * np.cos(np.pi * u)


def scale_B_for_stability(
    B: list[np.ndarray],
    W: np.ndarray,
    cap: float,
) -> list[np.ndarray]:
    n = W.shape[0]
    G = np.eye(n) + W
    phis = [G * b for b in B]
    rad = companion_radius(phis)
    if rad > cap:
        s = cap / (rad + 1e-12)
        B = [s * b for b in B]
    return B


def phi_at(W: np.ndarray, B: list[np.ndarray]) -> list[np.ndarray]:
    G = np.eye(W.shape[0]) + W
    return [G * b for b in B]


def _to_correlation(M: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    w = np.maximum(w, 1e-6)
    R = (V * w) @ V.T
    d = np.sqrt(np.clip(np.diag(R), 1e-12, None))
    R = R / np.outer(d, d)
    return 0.5 * (R + R.T)


def corr_from_network(
    A_soft: np.ndarray,
    rho: float,
    background: float = 0.06,
    n_proj: int = 12,
) -> np.ndarray:
    """
    PSD correlation that stays high on (symmetrized) hard edges and low off them.

    Alternating projection onto the edge pattern and the PSD-correlation
    cone, so non-edges do not get pulled above the estimator floor (0.30)
    the way a single eig-clip of rho * A does.
    """
    n = A_soft.shape[0]
    if rho <= 0.0:
        return np.eye(n)
    S = np.maximum(A_soft, A_soft.T)
    np.fill_diagonal(S, 0.0)
    edge = (S >= 0.5).astype(np.float64)
    bg = float(np.clip(background, 0.0, rho))
    R = np.where(edge > 0, rho, bg)
    np.fill_diagonal(R, 1.0)
    for _ in range(max(int(n_proj), 1)):
        R = _to_correlation(R)
        R = np.where(edge > 0, rho, bg)
        np.fill_diagonal(R, 1.0)
    return _to_correlation(R)


def simulate(
    n: int = 30,
    T: int = 800,
    p: int = 2,
    n_blocks: int = 3,
    n_regimes: int = 8,
    density: float = 0.10,
    block_ratio: float = 4.0,
    sigma: float = 0.006,
    rho_cap: float = 0.90,
    seed: int = 0,
    transition: str = "smooth",
    b_scale: float = 0.22,
    burn: int = 0,
    contemp_rho: float = 0.48,
    contemp_bg: float = 0.04,
    own_lag: float | None = None,
    net_lag: float | None = None,
) -> dict:
    """
    density: expected directed density of A_t (sparsity = 1 - density).
    n_blocks: number of communities.
    n_regimes: number of SBM anchors along the sample.
    transition: "smooth" blends neighboring anchors with a cosine weight;
                "step" holds each draw fixed until the next knot.
    contemp_rho: target |corr| on hard (symmetrized) edges.
    contemp_bg: target |corr| off the edge set. Keep this below the
                estimator floor (0.30) so recovered density can track truth.
    """
    if transition not in ("smooth", "step"):
        raise ValueError("transition must be 'smooth' or 'step'")
    rng = np.random.default_rng(seed)
    labels = np.array_split(np.arange(n), n_blocks)
    group = np.empty(n, dtype=int)
    for k, idx in enumerate(labels):
        group[idx] = k

    T_all = T + max(int(burn), 0)
    knots = np.linspace(0, T_all - 1, n_regimes).astype(int)
    P0 = sbm_probs(n, group, density, block_ratio)
    anchors = [draw_adj(P0, rng) for _ in range(n_regimes)]
    A_path = np.zeros((T_all, n, n))
    W_path = np.zeros((T_all, n, n))
    A_soft = np.zeros((T_all, n, n))
    regime_id = np.zeros(T_all, dtype=int)
    B = make_B(n, p, rng, scale=b_scale, own_lag=own_lag, net_lag=net_lag)
    B = scale_B_for_stability(B, row_normalize(anchors[0]), rho_cap)

    for t in range(T_all):
        if t <= knots[0]:
            r, soft = 0, anchors[0]
        elif t >= knots[-1]:
            r, soft = n_regimes - 1, anchors[-1]
        else:
            r = int(np.searchsorted(knots, t, side="right") - 1)
            span = max(int(knots[r + 1] - knots[r]), 1)
            u = (t - knots[r]) / span
            if transition == "smooth":
                lam = cosine_blend(u)
                soft = (1.0 - lam) * anchors[r] + lam * anchors[r + 1]
            else:
                soft = anchors[r]
        hard = (soft >= 0.5).astype(np.float64)
        np.fill_diagonal(hard, 0.0)
        A_path[t] = hard
        A_soft[t] = soft
        W_path[t] = row_normalize(soft)
        regime_id[t] = r

    mu = np.zeros(n)
    Y = np.zeros((T_all, n))
    Y[:p] = rng.normal(0.0, sigma, size=(p, n))
    for t in range(p, T_all):
        nxt = mu.copy()
        for k in range(1, p + 1):
            phis = phi_at(W_path[t - k], B)
            nxt = nxt + phis[k - 1] @ (Y[t - k] - mu)
        R = corr_from_network(A_soft[t], contemp_rho, background=contemp_bg)
        try:
            L = np.linalg.cholesky(R + 1e-8 * np.eye(n))
        except np.linalg.LinAlgError:
            L = np.eye(n)
        nxt = nxt + sigma * (L @ rng.normal(size=n))
        Y[t] = nxt

    if burn > 0:
        Y = Y[burn:]
        A_path = A_path[burn:]
        W_path = W_path[burn:]
        A_soft = A_soft[burn:]
        regime_id = regime_id[burn:]
    dens = A_path.sum(axis=(1, 2)) / (n * (n - 1))
    return {
        "Y": Y,
        "A": A_path,
        "W": W_path,
        "A_soft": A_soft,
        "B": B,
        "group": group,
        "regime_id": regime_id,
        "density_path": dens,
        "settings": {
            "n": n,
            "T": T,
            "p": p,
            "n_blocks": n_blocks,
            "n_regimes": n_regimes,
            "density": density,
            "sparsity": 1.0 - density,
            "block_ratio": block_ratio,
            "sigma": sigma,
            "rho_cap": rho_cap,
            "seed": seed,
            "transition": transition,
            "b_scale": b_scale,
            "own_lag": own_lag,
            "net_lag": net_lag,
            "burn": burn,
            "contemp_rho": contemp_rho,
            "contemp_bg": contemp_bg,
            "realized_mean_density": float(dens.mean()),
        },
    }


def save_sim(sim: dict, out_dir: Path, prefix: str = "sim_block_nvar") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tickers = [f"N{i:02d}" for i in range(sim["Y"].shape[1])]
    dates = pd.RangeIndex(sim["Y"].shape[0], name="t")
    pd.DataFrame(sim["Y"], index=dates, columns=tickers).to_csv(out_dir / f"{prefix}_y.csv")
    pd.DataFrame(
        {
            "regime": sim["regime_id"],
            "density": sim["density_path"],
        },
        index=dates,
    ).to_csv(out_dir / f"{prefix}_meta.csv")
    np.savez_compressed(
        out_dir / f"{prefix}_networks.npz",
        A=sim["A"],
        W=sim["W"],
        group=sim["group"],
        regime_id=sim["regime_id"],
        **{f"B{k+1}": b for k, b in enumerate(sim["B"])},
    )
    (out_dir / f"{prefix}_settings.json").write_text(
        json.dumps(sim["settings"], indent=2), encoding="utf-8"
    )
    return out_dir / f"{prefix}_y.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate y from time-blocked SBM networks.")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--T", type=int, default=800)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--n-blocks", type=int, default=3)
    parser.add_argument("--n-regimes", type=int, default=8)
    parser.add_argument(
        "--density",
        type=float,
        default=0.10,
        help="Expected directed density of A_t in (0,1). Sparsity is 1-density.",
    )
    parser.add_argument("--block-ratio", type=float, default=4.0)
    parser.add_argument("--sigma", type=float, default=0.006)
    parser.add_argument("--transition", choices=("smooth", "step"), default="smooth")
    parser.add_argument("--b-scale", type=float, default=0.22)
    parser.add_argument(
        "--contemp-rho",
        type=float,
        default=0.48,
        help="Target contemporaneous |corr| on hard edges.",
    )
    parser.add_argument(
        "--contemp-bg",
        type=float,
        default=0.04,
        help="Target contemporaneous |corr| off the edge set.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefix", type=str, default="sim_block_nvar")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    sim = simulate(
        n=args.n,
        T=args.T,
        p=args.p,
        n_blocks=args.n_blocks,
        n_regimes=args.n_regimes,
        density=args.density,
        block_ratio=args.block_ratio,
        sigma=args.sigma,
        seed=args.seed,
        transition=args.transition,
        b_scale=args.b_scale,
        contemp_rho=args.contemp_rho,
        contemp_bg=args.contemp_bg,
    )
    path = save_sim(sim, args.out_dir, args.prefix)
    s = sim["settings"]
    print(
        f"Wrote {path}  n={s['n']} T={s['T']}  "
        f"target density={s['density']:.3f}  realized={s['realized_mean_density']:.3f}  "
        f"sparsity={s['sparsity']:.3f}  blocks={s['n_blocks']}  "
        f"regimes={s['n_regimes']}  transition={s['transition']}"
    )


if __name__ == "__main__":
    main()
