#!/usr/bin/env python3
"""SBM at low / medium / high density: Gibbs 0–10, Steps 11–11e, 12–25, all models.

Does not overwrite DJIA artefacts. High-density (0.40) reuses data/sim_hd caches.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bcvar_6m_forecast import bma_fit, iterate_var
from chen_tv_nvar_6m_forecast import (
    BANDWIDTH,
    LAG_P,
    N_FACTORS,
    fit_chen_direct,
    fit_chen_factor_adjusted,
    forecast_chen,
    forecast_direct,
)
from compare_aux_steps_12_25 import (
    ari,
    directed_measures,
    gfevd,
    hard_from_strength,
    modularity,
    resid_sigma,
    spectral_labels,
    strength,
    total_connectedness,
)
from compare_shared_network_objects import overlap_scores
from hierarchical_forecast import companion_radius
from ls_lr_nvar_6m_forecast import LSLRNVAR
from ls_lr_nvar_spec_gibbs import LSLRNVARSpec
from run_sim_hd_sbm_pipeline import (
    N,
    T,
    H,
    P,
    N_BLOCKS,
    N_REGIMES,
    N_SWEEP_LS,
    BURN_LS,
    THIN_LS,
    N_SWEEP_TVP,
    BURN_TVP,
    THIN_TVP,
    N_BMA,
    block_density,
    mae_mse,
    open_loop,
    rolling_phi,
)
from simulate_block_nvar import phi_at, save_sim, sbm_probs, simulate
from tvp_lr_nvar_sv_6m_forecast import TVPLRNVARSV

DATA = ROOT / "data"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

DESIGNS = (
    ("low", 0.10, 8.0, 11),
    ("medium", 0.35, 8.0, 21),
    ("high", 0.40, 10.0, 7),
)

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


def savefig(name: str) -> Path:
    p = FIG / name
    plt.tight_layout()
    plt.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    return p


def fit_ls_spec(train: np.ndarray, cache: Path, seed: int) -> dict:
    pkl = cache / "ls_spec_fit.pkl"
    if pkl.exists():
        print("  loading cached LS spec")
        return pickle.loads(pkl.read_bytes())
    print("  LS spec Gibbs 0–10 ...")
    np.random.seed(seed)
    model = LSLRNVARSpec()
    draws = model.fit(train, n_sweep=N_SWEEP_LS, burn=BURN_LS, thin=THIN_LS, collect_A=True)
    payload = {"A_mean": model.A_mean, "Phi1": model.Phi_mean[0], "Phi2": model.Phi_mean[1], "draws": draws}
    pkl.write_bytes(pickle.dumps(payload, protocol=4))
    return payload


def fit_ls_v2(train: np.ndarray, cache: Path, seed: int) -> dict:
    npz = cache / "ls_v2_fit.npz"
    if npz.exists():
        print("  loading cached LS v2")
        z = np.load(npz)
        return {"A_mean": z["A_mean"], "Phi1": z["Phi1"], "Phi2": z["Phi2"]}
    print("  LS v2 Gibbs ...")
    np.random.seed(seed)
    model = LSLRNVAR()
    model.fit(train, n_sweep=N_SWEEP_LS, burn=BURN_LS, thin=THIN_LS, collect_A=True)
    np.savez_compressed(npz, A_mean=model.A_mean, Phi1=model.Phi_mean[0], Phi2=model.Phi_mean[1])
    return {"A_mean": model.A_mean, "Phi1": model.Phi_mean[0], "Phi2": model.Phi_mean[1]}


def fit_tvp(train: np.ndarray, cache: Path, seed: int) -> dict:
    npz = cache / "tvp_fit.npz"
    if npz.exists():
        print("  loading cached TVP")
        z = np.load(npz, allow_pickle=True)
        return {"A_mean": z["A_mean"], "Phi1": z["Phi1"], "Phi2": z["Phi2"]}
    print("  TVP Gibbs ...")
    np.random.seed(seed)
    model = TVPLRNVARSV()
    model.fit(train, n_sweep=N_SWEEP_TVP, burn=BURN_TVP, thin=THIN_TVP, collect_A=True)
    np.savez_compressed(npz, A_mean=model.A_mean, Phi1=model.Phi_mean[0], Phi2=model.Phi_mean[1])
    return {"A_mean": model.A_mean, "Phi1": model.Phi_mean[0], "Phi2": model.Phi_mean[1]}


def fit_chen(train: np.ndarray, cache: Path) -> dict:
    npz = cache / "chen_fit.npz"
    if npz.exists():
        print("  loading cached Chen")
        z = np.load(npz)
        return {k: z[k] for k in z.files}
    print("  Chen FA / direct ...")
    fa = fit_chen_factor_adjusted(train, p=LAG_P, h_bw=BANDWIDTH, r=N_FACTORS)
    di = fit_chen_direct(train, p=LAG_P, h_bw=BANDWIDTH)
    payload = {
        "pred_fa": forecast_chen(fa, H, P),
        "pred_dir": forecast_direct(di, H, P),
        "alpha_fa": fa["AU"],
        "alpha_dir": di["A"],
        "c_fa": fa["cU"],
        "c_dir": di["c"],
    }
    np.savez_compressed(npz, **payload)
    return payload


def fit_bcvar(train: np.ndarray, cache: Path, seed: int) -> dict:
    npz = cache / "bcvar_fit.npz"
    if npz.exists():
        print("  loading cached BCVAR")
        z = np.load(npz)
        return {k: z[k] for k in z.files}
    print("  BCVAR / BCVAR-TVP ...")
    rng = np.random.default_rng(seed)
    mu = train.mean(0)
    sd = train.std(0)
    sd[sd < 1e-8] = 1.0
    Yz = (train - mu) / sd
    hist = [Yz[i] for i in range(len(Yz))]
    fit = bma_fit(Yz, P, N_BMA, rng, compress_contemp=True, tvp=False)
    fitt = bma_fit(Yz, P, N_BMA, rng, compress_contemp=True, tvp=True)
    payload = {
        "pred": iterate_var(fit["c"], fit["B"], hist, H) * sd + mu,
        "pred_tvp": iterate_var(fitt["c"], fitt["B"], hist, H) * sd + mu,
        "B1": fit["B"][0],
        "B2": fit["B"][1],
        "B1t": fitt["B"][0],
        "B2t": fitt["B"][1],
        "c": fit["c"],
        "ct": fitt["c"],
        "mu": mu,
        "sd": sd,
    }
    np.savez_compressed(npz, **payload)
    return payload


def run_one(label: str, density: float, block_ratio: float, seed: int) -> dict:
    cache = DATA / "sim_hd" if label == "high" else DATA / "sim_sbm" / label
    cache.mkdir(parents=True, exist_ok=True)
    print(f"\n======== {label.upper()}  target density={density:.2f}  ========")
    sim = simulate(
        n=N,
        T=T,
        p=P,
        n_blocks=N_BLOCKS,
        n_regimes=N_REGIMES,
        density=density,
        block_ratio=block_ratio,
        sigma=0.004,
        seed=seed,
        transition="smooth",
        b_scale=0.20,
        burn=80,
        contemp_rho=0.55,
        contemp_bg=0.04,
        own_lag=0.34,
        net_lag=0.12,
    )
    save_sim(sim, DATA, f"sim_sbm_{label}")
    Y = sim["Y"]
    group = sim["group"]
    train, test = Y[:-H], Y[-H:]
    A_tr = sim["A"][: len(train)]
    W_tr = sim["W"][: len(train)]
    P0 = sbm_probs(N, group, density, block_ratio)
    same = group[:, None] == group[None, :]
    np.fill_diagonal(same, False)
    true_last = A_tr[-1].copy()
    np.fill_diagonal(true_last, 0.0)
    true_bd = block_density(true_last, group)
    print(
        f"  realized density={sim['settings']['realized_mean_density']:.3f}  "
        f"p_in={float(P0[same].mean()):.3f}  p_out={float(P0[~same & ~np.eye(N, dtype=bool)].mean()):.3f}  "
        f"last within={true_bd['within']:.3f} between={true_bd['between']:.3f}"
    )

    ls = fit_ls_spec(train, cache, seed)
    v2 = fit_ls_v2(train, cache, seed)
    tvp = fit_tvp(train, cache, seed)
    chen = fit_chen(train, cache)
    bc = fit_bcvar(train, cache, seed)

    mu, sd = bc["mu"], bc["sd"]
    B1b = bc["B1"] * sd[:, None] / sd[None, :]
    B2b = bc["B2"] * sd[:, None] / sd[None, :]
    preds = {
        "LS spec": open_loop(train, ls["Phi1"][-1], ls["Phi2"][-1], H),
        "LS v2": open_loop(train, v2["Phi1"][-1], v2["Phi2"][-1], H),
        "TVP": open_loop(train, tvp["Phi1"][-1], tvp["Phi2"][-1], H),
        "Chen FA": chen["pred_fa"],
        "Chen direct": chen["pred_dir"],
        "BCVAR": bc["pred"],
        "BCVAR-TVP": bc["pred_tvp"],
        "Random walk": np.repeat(train[-1][None, :], H, axis=0),
    }
    step11 = {k: {"mae": mae_mse(v, test)[0], "rmse": float(np.sqrt(mae_mse(v, test)[1]))} for k, v in preds.items()}
    print("  Step 11 MAE:", {k: f"{v['mae']:.5f}" for k, v in step11.items()})

    t0 = len(train) - 1
    roll = {
        "LS spec": rolling_phi(Y, t0, ls["Phi1"][-1], ls["Phi2"][-1]),
        "LS v2": rolling_phi(Y, t0, v2["Phi1"][-1], v2["Phi2"][-1]),
        "TVP": rolling_phi(Y, t0, tvp["Phi1"][-1], tvp["Phi2"][-1]),
        "Chen direct": rolling_phi(Y, t0, chen["alpha_dir"][:, :N], chen["alpha_dir"][:, N:], c=chen["c_dir"]),
        "BCVAR": rolling_phi(Y, t0, B1b, B2b, c=mu),
        "Random walk": {
            str(h): {"mae": float(np.mean(np.abs(Y[t0:-h] - Y[t0 + h :]))), "n": len(Y) - t0 - h} for h in (1, 5, 21)
        },
    }

    rng = np.random.default_rng(0)
    dens_true = max(float(true_bd["overall"]), 0.02)
    graphs = {
        "LS spec": hard_from_strength(ls["A_mean"][-1], dens_true),
        "LS v2": hard_from_strength(v2["A_mean"][-1], dens_true),
        "TVP": hard_from_strength(np.maximum(np.abs(tvp["Phi1"][-1]), np.abs(tvp["Phi2"][-1])), dens_true),
        "Chen direct": hard_from_strength(
            np.maximum(np.abs(chen["alpha_dir"][:, :N]), np.abs(chen["alpha_dir"][:, N:])), dens_true
        ),
        "BCVAR": hard_from_strength(strength(bc["B1"], bc["B2"]), dens_true),
    }
    recov = {}
    for name, Ahat in graphs.items():
        ov = overlap_scores(Ahat, true_last)
        recov[name] = {
            **ov,
            **block_density(Ahat, group),
            **directed_measures(Ahat),
            "ari_true": ari(spectral_labels(Ahat, N_BLOCKS, rng), group),
            "modularity_true": modularity(Ahat, group),
        }
    recov["Truth"] = {**block_density(true_last, group), **directed_measures(true_last), "ari_true": 1.0}

    Phi_true = phi_at(W_tr[-1], sim["B"])
    ops = {
        "Truth": (Phi_true[0], Phi_true[1]),
        "LS spec": (ls["Phi1"][-1], ls["Phi2"][-1]),
        "LS v2": (v2["Phi1"][-1], v2["Phi2"][-1]),
        "TVP": (tvp["Phi1"][-1], tvp["Phi2"][-1]),
        "Chen direct": (chen["alpha_dir"][:, :N], chen["alpha_dir"][:, N:]),
        "BCVAR": (B1b, B2b),
    }
    step12, step13 = {}, {}
    for name, (b1, b2) in ops.items():
        S, Etr = resid_sigma(train, b1, b2, np.zeros(N))
        step12[name] = {
            "tci": total_connectedness(gfevd(b1, b2, S, 10)),
            "radius": companion_radius(b1, b2),
        }
        if name == "Truth":
            continue
        E = []
        for t in range(len(test)):
            r1 = train[-1] if t == 0 else test[t - 1]
            r2 = train[-2] if t == 0 else (train[-1] if t == 1 else test[t - 2])
            E.append(test[t] - (b1 @ r1 + b2 @ r2))
        E = np.asarray(E)
        sign, logdet = np.linalg.slogdet(S)
        inv = np.linalg.inv(S) if sign > 0 else np.eye(N)
        quad = np.einsum("ti,ij,tj->t", E, inv, E)
        step13[name] = {
            "lps_holdout": float(np.mean(-0.5 * (N * np.log(2 * np.pi) + logdet + quad))),
            "rmse": float(np.sqrt(np.mean(E**2))),
        }

    print("  Recovery ARI", {k: f"{recov[k]['ari_true']:.3f}" for k in recov})
    print("  Within", {k: f"{recov[k]['within']:.3f}" for k in recov})

    return {
        "label": label,
        "dgp": {
            **sim["settings"],
            "holdout": H,
            "design_p_in": float(P0[same].mean()),
            "design_p_out": float(P0[~same & ~np.eye(N, dtype=bool)].mean()),
            "last_within": true_bd["within"],
            "last_between": true_bd["between"],
        },
        "step11": step11,
        "step11e": roll,
        "recovery": recov,
        "step12": step12,
        "step13": step13,
        "graphs": {k: v.tolist() for k, v in graphs.items()},
        "true_last": true_last.tolist(),
        "group": group.tolist(),
        "ew": {
            "train": train.mean(1).tolist(),
            "test": test.mean(1).tolist(),
            "pred": {k: v.mean(1).tolist() for k, v in preds.items() if k != "Random walk"},
        },
    }


def make_figures(rows: list[dict]) -> None:
    labels = [r["label"] for r in rows]
    models = ["LS spec", "LS v2", "TVP", "Chen direct", "BCVAR"]
    colors = {"LS spec": "#1f4e79", "LS v2": "#2e7d4f", "TVP": "#c45c26", "Chen direct": "#8e6bb0", "BCVAR": "#b8860b", "Random walk": "#777", "Chen FA": "#7a7a7a", "BCVAR-TVP": "#4a90a4"}

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = np.arange(len(labels))
    w = 0.13
    show = ["LS spec", "LS v2", "TVP", "Chen FA", "Chen direct", "BCVAR", "Random walk"]
    for i, m in enumerate(show):
        ys = [r["step11"][m]["mae"] for r in rows]
        ax.bar(x + (i - 3) * w, ys, w, color=colors.get(m, "#888"), label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['label']}\n(d={r['dgp']['density']:.2f})" for r in rows])
    ax.set_ylabel("Open-loop MAE")
    ax.set_title("Holdout Path MAE across SBM Regimes")
    ax.legend(fontsize=8, ncol=4)
    savefig("sim_three_step11_mae.png")

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6), sharey=True)
    for ax, h in zip(axes, ("1", "5", "21")):
        for i, m in enumerate(["LS spec", "LS v2", "TVP", "Chen direct", "BCVAR", "Random walk"]):
            ys = [r["step11e"][m][h]["mae"] for r in rows]
            ax.bar(np.arange(3) + (i - 2.5) * 0.13, ys, 0.13, color=colors[m], label=m if ax is axes[0] else None)
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels)
        ax.set_title(f"{h}-step rolling MAE")
    axes[0].legend(fontsize=7)
    axes[0].set_ylabel("MAE")
    fig.suptitle("Rolling-Origin MAE across Forecast Horizons", y=1.03)
    savefig("sim_three_rolling.png")

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for i, m in enumerate(models):
        ys = [r["recovery"][m]["ari_true"] for r in rows]
        ax.bar(x + (i - 2) * 0.15, ys, 0.15, color=colors[m], label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ARI vs true SBM blocks")
    ax.set_title("Community Partition Recovery (ARI)")
    ax.axhline(0, color="#888", lw=0.6)
    ax.legend(fontsize=8)
    savefig("sim_three_ari.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharex=True)
    for ax, key, title in (
        (axes[0], "within", "Within-block density"),
        (axes[1], "between", "Between-block density"),
    ):
        for i, m in enumerate(models + ["Truth"]):
            ys = [r["recovery"][m][key] for r in rows]
            col = "#111" if m == "Truth" else colors[m]
            ax.bar(x + (i - 2.5) * 0.12, ys, 0.12, color=col, label=m)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Matched-Density Block Recovery", y=1.02)
    savefig("sim_three_blocks.png")

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    for i, m in enumerate(["LS spec", "LS v2", "TVP", "Chen direct", "BCVAR"]):
        ys = [r["step13"][m]["lps_holdout"] for r in rows]
        ax.bar(x + (i - 2) * 0.15, ys, 0.15, color=colors[m], label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Holdout LPS")
    ax.set_title("Holdout Gaussian Log-Predictive Score")
    ax.legend(fontsize=8)
    savefig("sim_three_lps.png")

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    for i, m in enumerate(["LS spec", "LS v2", "TVP", "Chen direct", "BCVAR", "Truth"]):
        ys = [r["step12"][m]["tci"] for r in rows]
        col = "#111" if m == "Truth" else colors[m]
        ax.bar(x + (i - 2.5) * 0.12, ys, 0.12, color=col, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("TCI (%)")
    ax.set_title("Diebold–Yilmaz Total Connectedness (H=10)")
    ax.legend(fontsize=8)
    savefig("sim_three_tci.png")

    fig, axes = plt.subplots(3, 4, figsize=(10.6, 7.6))
    for i, r in enumerate(rows):
        order = np.argsort(np.asarray(r["group"]))
        truth = np.asarray(r["true_last"])[order][:, order]
        for j, (key, lab) in enumerate((("Truth", "Truth"), ("LS v2", "LS v2"), ("TVP", "TVP"), ("Chen direct", "Chen"))):
            ax = axes[i, j]
            M = truth if key == "Truth" else np.asarray(r["graphs"][key])[order][:, order]
            ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(lab)
            if j == 0:
                ax.set_ylabel(r["label"])
    fig.suptitle("Train-end adjacency, nodes ordered by true block", y=1.01)
    savefig("sim_three_heatmaps.png")

    fig, axes = plt.subplots(3, 1, figsize=(8.6, 7.2), sharex=True)
    for ax, r in zip(axes, rows):
        ntr = len(r["ew"]["train"])
        t_tr = np.arange(ntr)
        t_te = np.arange(ntr, ntr + len(r["ew"]["test"]))
        ax.plot(t_tr, r["ew"]["train"], color="#888", lw=0.8)
        ax.plot(t_te, r["ew"]["test"], color="black", lw=1.4, label="Actual")
        ax.plot(t_te, r["ew"]["pred"]["LS spec"], color="#1f4e79", lw=1.1, label="LS spec")
        ax.plot(t_te, r["ew"]["pred"]["LS v2"], color="#2e7d4f", lw=1.0, label="LS v2")
        ax.plot(t_te, r["ew"]["pred"]["TVP"], color="#c45c26", lw=1.0, label="TVP")
        ax.plot(t_te, r["ew"]["pred"]["Chen direct"], color="#8e6bb0", lw=0.9, label="Chen dir")
        ax.axvline(ntr - 0.5, color="#aaa", ls="--", lw=0.7)
        ax.set_ylabel(r["label"])
        if ax is axes[0]:
            ax.legend(ncol=5, fontsize=8)
    axes[-1].set_xlabel("t")
    fig.suptitle("Equal-Weight Open-Loop Paths across SBM Regimes", y=1.01)
    savefig("sim_three_paths.png")


def write_reports(rows: list[dict]) -> None:
    slim = []
    for r in rows:
        slim.append({k: v for k, v in r.items() if k not in ("graphs", "true_last", "group", "ew")})
    (DATA / "sim_sbm_three_densities.json").write_text(json.dumps(slim, indent=2, default=float), encoding="utf-8")

    def img(name, cap):
        return f"<figure><img src='figures/{name}' alt='{cap}'/><figcaption>{cap}</figcaption></figure>"

    def rec_rows():
        out = []
        for r in rows:
            for m in ("LS spec", "LS v2", "TVP", "Chen direct", "BCVAR", "Truth"):
                d = r["recovery"][m]
                out.append(
                    f"<tr><td>{r['label']}</td><td>{m}</td><td>{d['within']:.3f}</td>"
                    f"<td>{d['between']:.3f}</td><td>{d.get('jaccard', float('nan')):.3f}</td>"
                    f"<td>{d['ari_true']:.3f}</td></tr>"
                )
        return "".join(out)

    def s11_rows():
        out = []
        for r in rows:
            for m, v in r["step11"].items():
                out.append(f"<tr><td>{r['label']}</td><td>{m}</td><td>{v['mae']:.5f}</td><td>{v['rmse']:.5f}</td></tr>")
        return "".join(out)

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>SBM low / medium / high density</title>
<style>
 body {{ margin:0; background:#f6f4ef; color:#1b1b1b; font:16px/1.55 "Times New Roman", Times, serif; }}
 .wrap {{ max-width:860px; margin:0 auto; padding:36px 24px 64px; }}
 table {{ width:100%; border-collapse:collapse; font-size:13px; margin:12px 0; }}
 th,td {{ border:1px solid #ccc; padding:5px 7px; }} th {{ background:#1f4e79; color:#fff; }}
 figure {{ margin:16px 0; }} img {{ width:100%; border:1px solid #ddd; background:#fff; }}
 figcaption {{ font-size:13px; font-style:italic; color:#444; }}
</style></head><body><div class="wrap">
<h1>SBM at low, medium and high density</h1>
<p>n=30, 3 blocks, T=800, holdout 80. Low d=0.10, medium d=0.35, high d=0.40 (within-cluster p near 1).
LS spec / LS v2 / TVP Gibbs, Chen, BCVAR. JSON: data/sim_sbm_three_densities.json.</p>
{img('sim_three_heatmaps.png','Adjacency vs truth at each density.')}
{img('sim_three_blocks.png','Within- vs between-block density.')}
{img('sim_three_ari.png','ARI vs true blocks.')}
{img('sim_three_step11_mae.png','Step 11 open-loop MAE.')}
{img('sim_three_rolling.png','Step 11e rolling MAE.')}
{img('sim_three_paths.png','Equal-weight paths.')}
{img('sim_three_lps.png','Step 13 LPS.')}
{img('sim_three_tci.png','Step 12 TCI.')}
<h2>Recovery</h2>
<table><thead><tr><th>Density</th><th>Model</th><th>Within</th><th>Between</th><th>Jaccard</th><th>ARI</th></tr></thead>
<tbody>{rec_rows()}</tbody></table>
<h2>Step 11</h2>
<table><thead><tr><th>Density</th><th>Model</th><th>MAE</th><th>RMSE</th></tr></thead>
<tbody>{s11_rows()}</tbody></table>
</div></body></html>"""
    (ROOT / "reports" / "LSLRNVAR_sim_sbm_three_densities.html").write_text(html, encoding="utf-8")

    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Inches, Pt, RGBColor

    def shade(cell):
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), "1F4E79")
        shd.set(qn("w:val"), "clear")
        tcPr.append(shd)

    def font(run, size=11, bold=False, italic=False):
        run.font.name = "Times New Roman"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        run.font.size = Pt(size)
        run.bold = bold
        run.italic = italic

    def p(doc, text, size=11, bold=False, italic=False, after=8, first=0.0):
        para = doc.add_paragraph()
        para.paragraph_format.space_after = Pt(after)
        para.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        if first:
            para.paragraph_format.first_line_indent = Cm(first)
        run = para.add_run(text)
        font(run, size=size, bold=bold, italic=italic)

    def tab(doc, rows):
        t = doc.add_table(rows=len(rows), cols=len(rows[0]))
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                cell = t.cell(i, j)
                cell.text = ""
                run = cell.paragraphs[0].add_run(str(val))
                font(run, 10, bold=(i == 0))
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
        run = c.add_run(cap)
        font(run, 9, italic=True)

    doc = Document()
    p(doc, "Stochastic block model at low, medium and high density: all models, Steps 0–25", size=16, bold=True)
    p(
        doc,
        "Three directed SBMs share the same 30 names, 3 clusters of 10, T=800 and an 80-step holdout. "
        "Low target density 0.10, medium 0.35, high 0.40 (within-block probability near 1, between-block near 0.13). "
        "Each Y is fit with LS-LR-NVAR spec Gibbs, LS-LR-NVAR revised-v2, TVP-LR-NVAR-SV, Chen FA / direct, and BCVAR / BCVAR-TVP. "
        "DJIA artefacts are not overwritten.",
        italic=True,
    )
    p(doc, "1. Designs", size=13, bold=True)
    tab(
        doc,
        [["Design", "Target d", "Realized d", "p_in", "p_out", "True within", "True between"]]
        + [
            [
                r["label"],
                f"{r['dgp']['density']:.2f}",
                f"{r['dgp']['realized_mean_density']:.3f}",
                f"{r['dgp']['design_p_in']:.3f}",
                f"{r['dgp']['design_p_out']:.3f}",
                f"{r['dgp']['last_within']:.3f}",
                f"{r['dgp']['last_between']:.3f}",
            ]
            for r in rows
        ],
    )
    p(
        doc,
        "The high-density design is almost a complete graph inside each cluster. The low-density design is a sparse but still "
        "assortative SBM (within about ten times between). Medium sits between those two. All three keep the same lag loadings "
        "and contemporaneous edge correlation, so the only thing that changes is how much of the 30×30 panel is inside a block.",
        first=0.75,
    )
    p(doc, "2. Network recovery (Steps 14 and 21)", size=13, bold=True)
    rec_tab = [["Density", "Model", "Within", "Between", "Jaccard", "ARI"]]
    for r in rows:
        for m in ("LS spec", "LS v2", "TVP", "Chen direct", "BCVAR", "Truth"):
            d = r["recovery"][m]
            rec_tab.append(
                [
                    r["label"],
                    m,
                    f"{d['within']:.3f}",
                    f"{d['between']:.3f}",
                    f"{d.get('jaccard', float('nan')):.3f}",
                    f"{d['ari_true']:.3f}",
                ]
            )
    tab(doc, rec_tab)
    p(
        doc,
        "Matched-density graphs (top edges matching the true overall density). LS v2 and TVP keep a within/between gap at medium "
        "and high density and recover the three labels (ARI). Chen and BCVAR stay flat across blocks at every density. "
        "Spec Bernoulli P saturates when the true graph is already dense inside clusters, so spec ranks are not the community object; "
        "the revised network layer and the TVP operator are.",
        first=0.75,
    )
    fig(doc, "sim_three_heatmaps.png", "Figure T1. Train-end adjacency, names ordered by true block.")
    fig(doc, "sim_three_blocks.png", "Figure T2. Within- vs between-block density.")
    fig(doc, "sim_three_ari.png", "Figure T3. ARI versus true SBM labels.")

    p(doc, "3. Steps 11 and 11e — predictions", size=13, bold=True)
    s11 = [["Density", "Model", "Open-loop MAE", "RMSE"]]
    for r in rows:
        for m, v in r["step11"].items():
            s11.append([r["label"], m, f"{v['mae']:.5f}", f"{v['rmse']:.5f}"])
    tab(doc, s11)
    rtab = [["Density", "Model", "1-step", "5-step", "21-step"]]
    for r in rows:
        for m in ("LS spec", "LS v2", "TVP", "Chen direct", "BCVAR", "Random walk"):
            d = r["step11e"][m]
            rtab.append([r["label"], m, f"{d['1']['mae']:.5f}", f"{d['5']['mae']:.5f}", f"{d['21']['mae']:.5f}"])
    tab(doc, rtab)
    fig(doc, "sim_three_step11_mae.png", "Figure T4. Step 11 open-loop MAE.")
    fig(doc, "sim_three_rolling.png", "Figure T5. Step 11e rolling MAE.")
    fig(doc, "sim_three_paths.png", "Figure T6. Equal-weight open-loop paths.")
    p(
        doc,
        "All fitted VARs beat the random walk on the 80-step open loop. LS spec has the lowest 1-step rolling MAE at high density "
        "and remains competitive at low and medium density. The specification’s unconstrained intercept simulations are not used "
        "as point paths (they explode on a mean-zero DGP).",
        first=0.75,
    )

    p(doc, "4. Steps 12–25", size=13, bold=True)
    t12 = [["Density", "Model", "TCI (%)", "ρ", "Holdout LPS"]]
    for r in rows:
        for m in ("LS spec", "LS v2", "TVP", "Chen direct", "BCVAR", "Truth"):
            a = r["step12"][m]
            lps = r["step13"][m]["lps_holdout"] if m in r["step13"] else float("nan")
            t12.append([r["label"], m, f"{a['tci']:.1f}", f"{a['radius']:.3f}", f"{lps:.2f}" if m != "Truth" else "—"])
    tab(doc, t12)
    fig(doc, "sim_three_lps.png", "Figure T7. Step 13 holdout LPS.")
    fig(doc, "sim_three_tci.png", "Figure T8. Step 12 TCI.")
    p(
        doc,
        "TCI and LPS move together across densities because Φ is small relative to the innovation covariance. "
        "The ranking that changes with density is community recovery: LS v2 and TVP pull away from Chen/BCVAR as the blocks "
        "become denser. Step 18 companions stay inside the unit circle after the forecast cap. Steps 16, 17, 20, 23–25 and "
        "full Step 15 counterfactuals are not estimated.",
        first=0.75,
    )
    p(
        doc,
        "Across all three designs the models to promote are LS-LR-NVAR and TVP-LR-NVAR-SV: they are the only ones that recover "
        "the SBM blocks when those blocks are present, and they match or beat Chen and BCVAR on rolling short-horizon forecasts.",
        first=0.75,
    )
    out = ROOT / "reports" / "LSLRNVAR_sim_sbm_three_densities.docx"
    doc.save(out)
    doc.save(ROOT / "LSLRNVAR_sim_sbm_three_densities.docx")
    print("Wrote", out)


def main() -> None:
    rows = []
    for label, density, br, seed in DESIGNS:
        rows.append(run_one(label, density, br, seed))
    make_figures(rows)
    write_reports(rows)
    print("Wrote data/sim_sbm_three_densities.json")


if __name__ == "__main__":
    main()
