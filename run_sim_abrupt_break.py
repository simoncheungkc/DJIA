"""Appendix B.6: abrupt community swap on a non-bilinear block VAR.

The data-generating transition is a dense block matrix that jumps once at t = T/2.
It is not built from smooth bilinear latent coordinates. Each replication fits
TVP-LR-NVAR-SV and LS-LR-NVAR (v2) and records spectral ARI against the true labels.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_aux_steps_12_25 import ari, spectral_labels
from ls_lr_nvar_6m_forecast import LSLRNVAR
from tvp_lr_nvar_sv_6m_forecast import TVPLRNVARSV

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "sim_abrupt_break.json"
FIG = ROOT / "reports" / "figures" / "sim_abrupt_break_ari.png"

N = 30
T = 800
P = 2
N_BLOCKS = 3
BREAK = T // 2
SWAP = 5
N_REP = 4
SWEEP = 12
BURN = 6
THIN = 3
SIGMA = 0.005
OWN = 0.15
WITHIN = 0.12
BETWEEN = 0.0
NOISE = 0.004
RHO_CAP = 0.80


def labels_of(n: int, n_blocks: int) -> np.ndarray:
    lab = np.empty(n, dtype=int)
    for k, idx in enumerate(np.array_split(np.arange(n), n_blocks)):
        lab[idx] = k
    return lab


def block_phi(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(labels)
    B = np.full((n, n), BETWEEN)
    same = labels[:, None] == labels[None, :]
    B[same] = WITHIN
    np.fill_diagonal(B, OWN)
    B = B + NOISE * rng.normal(size=(n, n))
    np.fill_diagonal(B, OWN)
    return B


def scale_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = a.shape[0]
    C = np.zeros((2 * n, 2 * n))
    C[:n, :n] = a
    C[:n, n:] = b
    C[n:, :n] = np.eye(n)
    rad = float(np.max(np.abs(np.linalg.eigvals(C))))
    if rad > RHO_CAP:
        s = RHO_CAP / rad
        a = s * a
        b = s * b
    return a, b


def simulate(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    lab0 = labels_of(N, N_BLOCKS)
    lab1 = lab0.copy()
    a = np.flatnonzero(lab0 == 0)[:SWAP]
    c = np.flatnonzero(lab0 == 2)[:SWAP]
    lab1[a] = 2
    lab1[c] = 0
    Phi1_0, Phi2_0 = scale_pair(block_phi(lab0, rng), 0.45 * block_phi(lab0, rng))
    Phi1_1, Phi2_1 = scale_pair(block_phi(lab1, rng), 0.45 * block_phi(lab1, rng))
    Y = np.zeros((T, N))
    Y[:P] = rng.normal(0.0, SIGMA, size=(P, N))
    labels = np.zeros((T, N), dtype=int)
    for t in range(T):
        labels[t] = lab0 if t < BREAK else lab1
    for t in range(P, T):
        if t < BREAK:
            nxt = Phi1_0 @ Y[t - 1] + Phi2_0 @ Y[t - 2]
        else:
            nxt = Phi1_1 @ Y[t - 1] + Phi2_1 @ Y[t - 2]
        Y[t] = nxt + SIGMA * rng.normal(size=N)
    return {"Y": Y, "labels": labels}


def ari_path(strength: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros(T)
    for t in range(T):
        S = np.abs(strength[t])
        np.fill_diagonal(S, 0.0)
        out[t] = ari(spectral_labels(S, N_BLOCKS, rng), labels[t])
    return out


def ols_ari_path(Y: np.ndarray, labels: np.ndarray, seed: int, window: int = 80) -> np.ndarray:
    """Spectral ARI of a rolling VAR(1), using only observations inside one regime when the window crosses the break."""
    rng = np.random.default_rng(seed)
    out = np.full(T, np.nan)
    for t in range(window, T):
        a = t - window
        if a < BREAK <= t:
            a = BREAK
        if t - a < 40:
            continue
        X = Y[a:t]
        Z = Y[a + 1 : t + 1]
        coef = np.linalg.lstsq(X, Z, rcond=None)[0].T
        S = np.abs(coef)
        np.fill_diagonal(S, 0.0)
        out[t] = ari(spectral_labels(S, N_BLOCKS, rng), labels[t])
    return out


def fit_path(model, Y: np.ndarray) -> np.ndarray:
    model.fit(Y, n_sweep=SWEEP, burn=BURN, thin=THIN, collect_A=True)
    phi = model.Phi_mean[0]
    return np.abs(phi)


def main() -> None:
    t0 = time.perf_counter()
    rows = []
    paths = []
    for rep in range(N_REP):
        sim = simulate(1000 + rep)
        print(f"replication {rep + 1}/{N_REP}", flush=True)
        tvp = fit_path(TVPLRNVARSV(), sim["Y"])
        ls = fit_path(LSLRNVAR(), sim["Y"])
        tvp_ari = ari_path(tvp, sim["labels"], 2000 + rep)
        ls_ari = ari_path(ls, sim["labels"], 3000 + rep)
        ols_ari = ols_ari_path(sim["Y"], sim["labels"], 4000 + rep)
        paths.append((tvp_ari, ls_ari, ols_ari))
        pre = slice(BREAK - 40, BREAK)
        windows = {
            "pre_40": (pre.start, pre.stop),
            "post_0_30": (BREAK, BREAK + 30),
            "post_30_50": (BREAK + 30, BREAK + 50),
            "post_50_100": (BREAK + 50, BREAK + 100),
        }
        row = {"rep": rep}
        for name, (a, b) in windows.items():
            row[f"tvp_{name}"] = float(np.nanmean(tvp_ari[a:b]))
            row[f"ls_{name}"] = float(np.nanmean(ls_ari[a:b]))
            row[f"ols_{name}"] = float(np.nanmean(ols_ari[a:b]))
        rows.append(row)
        print("  " + ", ".join(f"{k}={v:.3f}" for k, v in row.items() if k != "rep"), flush=True)
    stack_tvp = np.nanmean([p[0] for p in paths], axis=0)
    stack_ls = np.nanmean([p[1] for p in paths], axis=0)
    stack_ols = np.nanmean([p[2] for p in paths], axis=0)
    summary = {}
    keys = [k for k in rows[0] if k != "rep"]
    for k in keys:
        vals = np.array([r[k] for r in rows])
        summary[k] = {"mean": float(vals.mean()), "se": float(vals.std(ddof=1) / np.sqrt(len(vals)))}
    payload = {
        "design": {
            "n": N,
            "T": T,
            "p": P,
            "break": BREAK,
            "swap": SWAP,
            "replications": N_REP,
            "sweeps": SWEEP,
            "burn": BURN,
            "thin": THIN,
            "dgp": "dense block VAR with one discrete membership swap; not a bilinear latent-space walk",
        },
        "summary": summary,
        "tvp_ari_mean_path": np.round(stack_tvp[::10], 4).tolist(),
        "ls_ari_mean_path": np.round(stack_ls[::10], 4).tolist(),
        "ols_ari_mean_path": np.round(stack_ols[::10], 4).tolist(),
        "elapsed_sec": time.perf_counter() - t0,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(T)
    ax.plot(x, stack_ols, color="0.35", label="Rolling least squares")
    ax.plot(x, stack_tvp, color="#c45c26", label="TVP-LR-NVAR-SV")
    ax.plot(x, stack_ls, color="#1f4e79", label="LS-LR-NVAR (v2)")
    ax.axvline(BREAK, color="0.4", ls="--", lw=1)
    ax.set_xlabel("Time index")
    ax.set_ylabel("ARI versus true blocks")
    ax.set_title("Community Recovery after an Abrupt Membership Swap")
    ax.legend()
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=140)
    plt.close()
    print(f"wrote {OUT} in {payload['elapsed_sec']:.0f}s")


if __name__ == "__main__":
    main()
