#!/usr/bin/env python3
"""
SBM simulation: TVP-LR-NVAR-SV vs Chen TV-LASSO vs BCVAR (Koop et al. 2019).

Same blocked-network DGP as compare_sim_models.py. All three models forecast
the held-out tail of y (stationary network VAR). TVP-LR-NVAR-SV is fit on y
as the observation (local-linear trend should stay near flat). Chen and BCVAR
are the same reduced-form specs used on DJIA returns.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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
from compare_networks_over_time import EDGE_TOL, granger_adj
from compare_shared_network_objects import (
    norm_frobenius,
    offdiag_vec,
    overlap_scores,
    rank_corr,
    strength_from_alpha,
    topk_adj,
)
from compare_sim_models import (
    DENSITIES,
    H,
    N,
    P,
    SEED0,
    T,
    TRANSITION,
    corr_floor_density,
    directed_density,
    mae_mse,
    mape,
    one_step_chen,
    one_step_phi,
    simulate_for_density,
    strength_phi_list,
)
from ls_lr_nvar_6m_forecast import CORR_EDGE
from simulate_block_nvar import phi_at
from tvp_lr_nvar_sv_6m_forecast import TVPLRNVARSV, mean_forecast_tvp

DATA_DIR = Path(__file__).resolve().parent / "data"
CACHE = DATA_DIR / "sim_study_pred"
N_SWEEP = 16
BURN_S = 8
THIN_S = 2
N_COMPRESS = 30


def fit_tvp(Y: np.ndarray, cache: Path) -> dict:
    if cache.exists():
        print(f"  loading cached TVP-LR-NVAR-SV {cache.name}")
        z = np.load(cache, allow_pickle=True)
        return {
            "A_mean": z["A_mean"],
            "Phi1": z["Phi1"],
            "Phi2": z["Phi2"],
            "pred": z["pred"],
        }
    print("  fitting TVP-LR-NVAR-SV ...")
    np.random.seed(0)
    model = TVPLRNVARSV()
    draws = model.fit(Y, n_sweep=N_SWEEP, burn=BURN_S, thin=THIN_S, collect_A=True)
    if model.A_mean is None or model.Phi_mean is None:
        raise RuntimeError("TVP-LR-NVAR-SV did not store A_mean / Phi_mean")
    sd = float(np.std(Y))
    paths = np.stack(
        [
            mean_forecast_tvp(
                d, H, P,
                max_drift=0.25 * sd,
                max_cycle=4.0 * sd,
                max_disp=10.0 * sd,
                max_level_gap=2.0 * sd,
            )
            for d in draws
        ],
        axis=0,
    )
    pred = np.nan_to_num(paths.mean(axis=0), nan=Y[-1])
    np.savez_compressed(
        cache,
        A_mean=model.A_mean,
        Phi1=model.Phi_mean[0],
        Phi2=model.Phi_mean[1],
        pred=pred,
    )
    return {
        "A_mean": model.A_mean,
        "Phi1": model.Phi_mean[0],
        "Phi2": model.Phi_mean[1],
        "pred": pred,
    }


def fit_chen(Y: np.ndarray, cache: Path) -> dict:
    if cache.exists():
        print(f"  loading cached Chen {cache.name}")
        z = np.load(cache, allow_pickle=True)
        return {
            "pred_fa": z["pred_fa"],
            "pred_dir": z["pred_dir"],
            "alpha_dir": z["alpha_dir"],
            "alpha_fa": z["alpha_fa"],
            "c_dir": z["c_dir"],
            "c_fa": z["c_fa"],
        }
    print("  fitting Chen ...")
    fit_fa = fit_chen_factor_adjusted(Y, p=LAG_P, h_bw=BANDWIDTH, r=N_FACTORS)
    pred_fa = forecast_chen(fit_fa, H, P)
    fit_dir = fit_chen_direct(Y, p=LAG_P, h_bw=BANDWIDTH)
    pred_dir = forecast_direct(fit_dir, H, P)
    np.savez_compressed(
        cache,
        pred_fa=pred_fa,
        pred_dir=pred_dir,
        alpha_dir=fit_dir["A"],
        alpha_fa=fit_fa["AU"],
        c_dir=fit_dir["c"],
        c_fa=fit_fa["cU"],
    )
    return {
        "pred_fa": pred_fa,
        "pred_dir": pred_dir,
        "alpha_dir": fit_dir["A"],
        "alpha_fa": fit_fa["AU"],
        "c_dir": fit_dir["c"],
        "c_fa": fit_fa["cU"],
    }


def fit_bcvar(Y: np.ndarray, cache: Path) -> dict:
    if cache.exists():
        print(f"  loading cached BCVAR {cache.name}")
        z = np.load(cache, allow_pickle=True)
        return {
            "pred": z["pred"],
            "pred_tvp": z["pred_tvp"],
            "G": z["G"],
            "G_tvp": z["G_tvp"],
            "c": z["c"],
            "c_tvp": z["c_tvp"],
            "B1": z["B1"],
            "B2": z["B2"],
            "B1_tvp": z["B1_tvp"],
            "B2_tvp": z["B2_tvp"],
            "mu": z["mu"] if "mu" in z.files else None,
            "sd": z["sd"] if "sd" in z.files else None,
        }
    print("  fitting BCVAR / BCVAR-TVP ...")
    rng = np.random.default_rng(0)
    mu = Y.mean(axis=0)
    sd = Y.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Yz = (Y - mu) / sd
    hist_z = [Yz[i] for i in range(len(Yz))]
    fit = bma_fit(Yz, P, N_COMPRESS, rng, compress_contemp=True, tvp=False)
    fit_t = bma_fit(Yz, P, N_COMPRESS, rng, compress_contemp=True, tvp=True)
    pred_z = iterate_var(fit["c"], fit["B"], hist_z, H)
    pred_tz = iterate_var(fit_t["c"], fit_t["B"], hist_z, H)
    pred = pred_z * sd + mu
    pred_tvp = pred_tz * sd + mu
    np.savez_compressed(
        cache,
        pred=pred,
        pred_tvp=pred_tvp,
        G=fit["G"],
        G_tvp=fit_t["G"],
        c=fit["c"],
        c_tvp=fit_t["c"],
        B1=fit["B"][0],
        B2=fit["B"][1],
        B1_tvp=fit_t["B"][0],
        B2_tvp=fit_t["B"][1],
        mu=mu,
        sd=sd,
    )
    return {
        "pred": pred,
        "pred_tvp": pred_tvp,
        "G": fit["G"],
        "G_tvp": fit_t["G"],
        "c": fit["c"],
        "c_tvp": fit_t["c"],
        "B1": fit["B"][0],
        "B2": fit["B"][1],
        "B1_tvp": fit_t["B"][0],
        "B2_tvp": fit_t["B"][1],
        "mu": mu,
        "sd": sd,
    }


def score_one(density: float) -> dict:
    seed = SEED0 + int(round(100 * density))
    print(f"\n=== SBM density={density:.2f}  seed={seed} ===")
    sim = simulate_for_density(density)
    Y = sim["Y"]
    train, test = Y[:-H], Y[-H:]
    A_tr = sim["A"][:-H]
    W_tr = sim["W"][:-H]
    B_true = sim["B"]
    n = train.shape[1]
    t0 = train.shape[0] - 1

    CACHE.mkdir(parents=True, exist_ok=True)
    tag = f"d{int(round(100 * density)):03d}"
    tvp = fit_tvp(train, CACHE / f"{tag}_tvp.npz")
    chen = fit_chen(train, CACHE / f"{tag}_chen.npz")
    bc = fit_bcvar(train, CACHE / f"{tag}_bcvar.npz")

    A_hat = (tvp["A_mean"] >= 0.5).astype(float)
    true_dens = directed_density(A_tr)
    A_last = A_tr[t0].copy()
    np.fill_diagonal(A_last, 0.0)
    G_dir = granger_adj(chen["alpha_dir"], n, P, EDGE_TOL)
    G_fa = granger_adj(chen["alpha_fa"], n, P, EDGE_TOL)
    G_bc = granger_adj(bc["G"], n, P, EDGE_TOL)
    G_bct = granger_adj(bc["G_tvp"], n, P, EDGE_TOL)

    Phi_true = phi_at(W_tr[t0], B_true)
    S_true = strength_phi_list(Phi_true)
    S_tvp = strength_phi_list([tvp["Phi1"][t0], tvp["Phi2"][t0]])
    S_dir = strength_from_alpha(chen["alpha_dir"], n, P)

    k_true = int(A_last.sum())
    ov_tvp = overlap_scores(A_hat[t0], A_last)
    ov_dir = overlap_scores(G_dir, A_last)
    ov_bc = overlap_scores(G_bc, A_last)

    pred_tvp_ol = tvp["pred"][:H]
    pred_tvp_os = one_step_phi(Y, t0, tvp["Phi1"][t0], tvp["Phi2"][t0], H)
    chen_ol_d = chen["pred_dir"][:H]
    chen_ol_f = chen["pred_fa"][:H]
    chen_os_d = one_step_chen(Y, t0, chen["c_dir"], chen["alpha_dir"], H, P)
    chen_os_f = one_step_chen(Y, t0, chen["c_fa"], chen["alpha_fa"], H, P)
    bc_ol = bc["pred"][:H]
    bc_ol_t = bc["pred_tvp"][:H]
    mu_y = np.asarray(bc["mu"]) if bc.get("mu") is not None else train.mean(axis=0)
    sd_y = np.asarray(bc["sd"]) if bc.get("sd") is not None else train.std(axis=0)
    sd_y = np.where(sd_y < 1e-8, 1.0, sd_y)
    bc_os = np.zeros((H, n))
    bc_os_t = np.zeros((H, n))
    for t in range(H):
        i = t0 + 1 + t
        z1 = (Y[i - 1] - mu_y) / sd_y
        z2 = (Y[i - 2] - mu_y) / sd_y
        bc_os[t] = mu_y + sd_y * (bc["c"] + bc["B1"] @ z1 + bc["B2"] @ z2)
        bc_os_t[t] = mu_y + sd_y * (bc["c_tvp"] + bc["B1_tvp"] @ z1 + bc["B2_tvp"] @ z2)

    models_ol = {
        "TVP-LR-NVAR-SV": pred_tvp_ol,
        "Chen direct": chen_ol_d,
        "Chen FA": chen_ol_f,
        "BCVAR": bc_ol,
        "BCVAR-TVP": bc_ol_t,
    }
    models_os = {
        "TVP-LR-NVAR-SV": pred_tvp_os,
        "Chen direct": chen_os_d,
        "Chen FA": chen_os_f,
        "BCVAR": bc_os,
        "BCVAR-TVP": bc_os_t,
    }

    row = {
        "true_density_target": density,
        "true_density_realized": true_dens,
        "corr_floor_density": corr_floor_density(train, CORR_EDGE),
        "tvp_binary_density": directed_density(A_hat),
        "tvp_expected_density": directed_density(tvp["A_mean"]),
        "chen_direct_density": directed_density(G_dir),
        "chen_fa_density": directed_density(G_fa),
        "bcvar_density": directed_density(G_bc),
        "bcvar_tvp_density": directed_density(G_bct),
        "tvp_jaccard": ov_tvp["jaccard"],
        "chen_direct_jaccard": ov_dir["jaccard"],
        "bcvar_jaccard": ov_bc["jaccard"],
        "tvp_spearman_Phi": rank_corr(offdiag_vec(S_tvp), offdiag_vec(S_true)),
        "chen_direct_spearman_Phi": rank_corr(offdiag_vec(S_dir), offdiag_vec(S_true)),
        "tvp_frobenius_Phi": norm_frobenius(S_tvp, S_true),
        "chen_direct_frobenius_Phi": norm_frobenius(S_dir, S_true),
        "tvp_matched_overlap": overlap_scores(topk_adj(tvp["A_mean"][t0], k_true), A_last)["overlap"],
    }
    for name, pred in models_ol.items():
        mae, mse = mae_mse(pred, test)
        key = name.lower().replace(" ", "_").replace("-", "_")
        row[f"openloop_mae_{key}"] = mae
        row[f"openloop_mse_{key}"] = mse
        row[f"openloop_mape_{key}"] = mape(pred, test)
    for name, pred in models_os.items():
        mae, mse = mae_mse(pred, test)
        key = name.lower().replace(" ", "_").replace("-", "_")
        row[f"onestep_mae_{key}"] = mae
        row[f"onestep_mse_{key}"] = mse
        row[f"onestep_mape_{key}"] = mape(pred, test)

    os_keys = {k: row[f"onestep_mae_{k.lower().replace(' ', '_').replace('-', '_')}"] for k in models_os}
    ol_keys = {k: row[f"openloop_mae_{k.lower().replace(' ', '_').replace('-', '_')}"] for k in models_ol}
    row["onestep_winner"] = min(os_keys, key=os_keys.get)
    row["openloop_winner"] = min(ol_keys, key=ol_keys.get)
    print(
        f"  1-step MAE  TVP {row['onestep_mae_tvp_lr_nvar_sv']:.5f}  "
        f"Chen dir {row['onestep_mae_chen_direct']:.5f}  "
        f"BCVAR {row['onestep_mae_bcvar']:.5f}  winner {row['onestep_winner']}"
    )
    print(
        f"  open-loop MAE  TVP {row['openloop_mae_tvp_lr_nvar_sv']:.5f}  "
        f"Chen dir {row['openloop_mae_chen_direct']:.5f}  "
        f"BCVAR {row['openloop_mae_bcvar']:.5f}  winner {row['openloop_winner']}"
    )
    return row


def write_html(rows: list[dict], path: Path) -> None:
    def cell(r, key, nd=5):
        v = r.get(key, float("nan"))
        if not np.isfinite(v):
            return "—"
        return f"{v:.{nd}f}"

    body = []
    for r in rows:
        body.append(
            "<tr>"
            f"<td>{r['true_density_target']:.2f}</td>"
            f"<td class='num'>{r['true_density_realized']:.3f}</td>"
            f"<td class='num'>{r['tvp_binary_density']:.3f}</td>"
            f"<td class='num'>{r['chen_direct_density']:.3f}</td>"
            f"<td class='num'>{r['bcvar_density']:.3f}</td>"
            f"<td class='num'>{cell(r, 'onestep_mae_tvp_lr_nvar_sv')}</td>"
            f"<td class='num'>{cell(r, 'onestep_mae_chen_direct')}</td>"
            f"<td class='num'>{cell(r, 'onestep_mae_bcvar')}</td>"
            f"<td class='num'>{cell(r, 'openloop_mae_tvp_lr_nvar_sv')}</td>"
            f"<td class='num'>{cell(r, 'openloop_mae_chen_direct')}</td>"
            f"<td class='num'>{cell(r, 'openloop_mae_bcvar')}</td>"
            f"<td>{r['onestep_winner']}</td>"
            f"<td>{r['openloop_winner']}</td>"
            "</tr>"
        )
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>SBM simulation: TVP-LR-NVAR-SV vs Chen vs BCVAR</title>
<style>
  body {{ margin:0; padding:24px; background:#181818; color:#e6e6e6;
         font:14px/1.5 ui-sans-serif, system-ui, Segoe UI, sans-serif; }}
  .wrap {{ max-width:1200px; margin:0 auto; }}
  h1 {{ font-size:22px; font-weight:600; margin:0 0 8px; }}
  p {{ color:#9a9a9a; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; border:1px solid #333; }}
  th, td {{ padding:7px 8px; border-bottom:1px solid #333; text-align:left; }}
  th {{ background:#202020; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
</style></head><body><div class="wrap">
<h1>SBM blocked-network forecast comparison</h1>
<p>
  DGP: stochastic block network VAR, n={N}, T={T}, holdout H={H}, p={P},
  densities {', '.join(f'{d:.2f}' for d in DENSITIES)}.
  TVP-LR-NVAR-SV Gibbs ({N_SWEEP} sweeps), Chen TV-LASSO (direct and factor-adjusted),
  BCVAR / BCVAR-TVP (Koop, Korobilis, Pettenuzzo 2019, R={N_COMPRESS}).
</p>
<table>
<thead><tr>
<th>Target dens.</th><th class="num">True</th><th class="num">TVP dens.</th>
<th class="num">Chen dir</th><th class="num">BCVAR</th>
<th class="num">1s TVP</th><th class="num">1s Chen</th><th class="num">1s BCVAR</th>
<th class="num">OL TVP</th><th class="num">OL Chen</th><th class="num">OL BCVAR</th>
<th>1-step win</th><th>Open-loop win</th>
</tr></thead>
<tbody>{''.join(body)}</tbody>
</table>
<p>MAE on the H-step holdout of y. Source: compare_sim_three_models.py.</p>
</div></body></html>
"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    rows = [score_one(d) for d in DENSITIES]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "dgp": {
            "n": N, "T": T, "H": H, "p": P,
            "densities": list(DENSITIES),
            "transition": TRANSITION,
        },
        "tvp": {"n_sweep": N_SWEEP, "burn": BURN_S, "thin": THIN_S},
        "bcvar": {"R": N_COMPRESS, "paper": "Koop, Korobilis, Pettenuzzo, JoE 2019"},
        "rows": rows,
    }
    (DATA_DIR / "sim_three_model_comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_html(rows, DATA_DIR / "sim_three_model_comparison.html")
    print("Saved data/sim_three_model_comparison.json / .html")


if __name__ == "__main__":
    main()
