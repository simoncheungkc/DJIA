#!/usr/bin/env python3
"""
Step 11a / rolling-origin forecasts (user: Step 11e).

Specification Step 11a: at each origin tau, use information I_tau and iterate
the network VAR forward. Daily Gibbs warm-start is not repeated; Phi is the
train-end posterior mean operator. The intercept is the Minnesota predictive
mean computed on returns up to that origin (expanding window).

Horizons: 1, 5, 21 trading days. Holdout: 2026-02-06 to 2026-08-07.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hierarchical_forecast import (
    DATA_DIR,
    FORECAST_CAP,
    iterate_returns,
    last_operators,
    minnesota_intercept,
    scale_companion,
)

HORIZON = 126
P = 2
HORIZONS = (1, 5, 21)


def price_from_returns(last: np.ndarray, rets: np.ndarray) -> np.ndarray:
    return last * np.exp(np.cumsum(rets, axis=0))


def load_prices() -> pd.DataFrame:
    prices = pd.read_csv(DATA_DIR / "djia_adj_close_wide_10y.csv", parse_dates=["Date"], index_col="Date")
    return prices.sort_index().ffill().dropna(how="all")


def chen_direct_operator(train_ret: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(DATA_DIR / "chen_operator_grid.npz")
    a = z["dir_alpha"][-1]
    n = train_ret.shape[1]
    resid = []
    for t in range(P, len(train_ret)):
        zlag = np.concatenate([train_ret[t - k] for k in range(1, P + 1)])
        resid.append(train_ret[t] - a @ zlag)
    c = np.mean(np.asarray(resid), axis=0) if resid else np.zeros(n)
    return c, a


def iterate_phi(h: int, r1: np.ndarray, r2: np.ndarray, mu: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    return iterate_returns(mu, b1, b2, r1, r2, h)


def iterate_generic(h: int, r1: np.ndarray, r2: np.ndarray, step) -> np.ndarray:
    out = np.zeros((h, r1.shape[0]))
    a, b = r1.copy(), r2.copy()
    for t in range(h):
        nxt = step(a, b)
        out[t] = nxt
        b, a = a, nxt
    return out


def write_html(payload: dict, path: Path) -> None:
    rows = []
    for h in payload["horizons"]:
        rec = payload["by_horizon"][str(h)]
        for name, d in rec.items():
            rows.append(
                f"<tr><td>{h}</td><td>{name}</td>"
                f"<td class='num'>{d['price_mape']:.2f}%</td>"
                f"<td class='num'>{d['price_mae']:.2f}</td>"
                f"<td class='num'>{d['ret_mae']*100:.3f} pp</td>"
                f"<td class='num'>{d['ret_rmsfe']*100:.3f} pp</td>"
                f"<td class='num'>{d['n_origins']}</td></tr>"
            )
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Step 11a rolling-origin DJIA forecasts</title>
<style>
  body {{ margin:0; padding:24px; background:#181818; color:#e6e6e6;
         font:14px/1.5 ui-sans-serif, system-ui, Segoe UI, sans-serif; }}
  .wrap {{ max-width:1040px; margin:0 auto; }}
  h1 {{ font-size:22px; margin:0 0 8px; }}
  p {{ color:#9a9a9a; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; border:1px solid #333; }}
  th, td {{ padding:7px 10px; border-bottom:1px solid #333; text-align:left; }}
  th {{ background:#202020; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
</style></head><body><div class="wrap">
<h1>Step 11a rolling-origin forecasts</h1>
<p>
  Holdout {payload['window']['test_start']} to {payload['window']['test_end']}.
  Network operators frozen at {payload['window']['train_end']}. Each origin uses
  the true return history up to that day and an expanding-window Minnesota intercept.
  Horizons are 1, 5 and 21 trading days. Spec Step 11a; daily Gibbs warm-start is not repeated.
</p>
<table>
<thead><tr><th>Horizon</th><th>Model</th><th class="num">Price MAPE</th>
<th class="num">Price MAE</th><th class="num">Return MAE</th><th class="num">Return RMSFE</th>
<th class="num">Origins</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
<p>Return MAE / RMSFE are on the horizon log-return. Source: rolling_forecast_compare.py.</p>
</div></body></html>
"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    prices = load_prices()
    pmat = prices.values
    logp = np.log(pmat)
    ret = np.diff(logp, axis=0)
    t_price = len(prices)
    t_train_end = t_price - HORIZON - 1
    train_ret = ret[:t_train_end]
    test_start = t_train_end + 1
    print(
        f"Step 11a rolling  train through {prices.index[t_train_end].date()}  "
        f"holdout {prices.index[test_start].date()} to {prices.index[-1].date()}"
    )

    spec_b1, spec_b2 = last_operators(DATA_DIR / "lslrnvar_A_mean_train_spec.npz")
    v2_b1, v2_b2 = last_operators(DATA_DIR / "lslrnvar_A_mean_train_v2.npz")
    tvp_b1, tvp_b2 = last_operators(DATA_DIR / "tvplrnvarsv_A_mean_train.npz")
    spec_b1, spec_b2 = scale_companion(spec_b1, spec_b2, FORECAST_CAP)
    v2_b1, v2_b2 = scale_companion(v2_b1, v2_b2, FORECAST_CAP)
    tvp_b1, tvp_b2 = scale_companion(tvp_b1, tvp_b2, FORECAST_CAP)

    chen_c, chen_a = chen_direct_operator(train_ret)
    bc = np.load(DATA_DIR / "bcvar_operator.npz")
    bct = np.load(DATA_DIR / "bcvar_tvp_operator.npz")
    mu_bc = train_ret.mean(axis=0)
    sd = train_ret.std(axis=0)
    sd[sd < 1e-8] = 1.0

    def step_chen(r1, r2, _mu=None):
        return chen_c + chen_a @ np.concatenate([r1, r2])

    def step_bcvar(r1, r2, op, _mu=None):
        z1 = (r1 - mu_bc) / sd
        z2 = (r2 - mu_bc) / sd
        zhat = op["c"] + op["B1"] @ z1 + op["B2"] @ z2
        return np.clip(zhat * sd + mu_bc, -0.03, 0.03)

    last_origin = t_price - 1 - max(HORIZONS)
    names = [
        "LS-LR-NVAR spec",
        "LS-LR-NVAR v2",
        "TVP-LR-NVAR-SV",
        "Chen direct",
        "BCVAR",
        "BCVAR-TVP",
        "Random walk",
    ]
    by_h = {str(h): {name: {"ae": [], "ape": [], "ret_ae": [], "ret_se": []} for name in names} for h in HORIZONS}

    n_origins = 0
    for origin in range(t_train_end, last_origin + 1):
        if origin < 2:
            continue
        r1 = ret[origin - 1]
        r2 = ret[origin - 2]
        hist = ret[:origin]
        mu = minnesota_intercept(hist)
        last_px = pmat[origin]
        hmax = max(HORIZONS)
        paths = {
            "LS-LR-NVAR spec": iterate_phi(hmax, r1, r2, mu, spec_b1, spec_b2),
            "LS-LR-NVAR v2": iterate_phi(hmax, r1, r2, mu, v2_b1, v2_b2),
            "TVP-LR-NVAR-SV": iterate_phi(hmax, r1, r2, mu, tvp_b1, tvp_b2),
            "Chen direct": iterate_generic(hmax, r1, r2, lambda a, b: step_chen(a, b)),
            "BCVAR": iterate_generic(hmax, r1, r2, lambda a, b: step_bcvar(a, b, bc)),
            "BCVAR-TVP": iterate_generic(hmax, r1, r2, lambda a, b: step_bcvar(a, b, bct)),
            "Random walk": np.zeros((hmax, r1.shape[0])),
        }
        n_origins += 1
        for name, path in paths.items():
            px = price_from_returns(last_px, path)
            for h in HORIZONS:
                tgt = origin + h
                ae = np.abs(px[h - 1] - pmat[tgt])
                ape = ae / np.maximum(np.abs(pmat[tgt]), 1e-8)
                r_hat = np.log(px[h - 1] / last_px)
                r_act = logp[tgt] - logp[origin]
                err = r_hat - r_act
                rec = by_h[str(h)][name]
                rec["ae"].append(float(ae.mean()))
                rec["ape"].append(float(ape.mean()))
                rec["ret_ae"].append(float(np.mean(np.abs(err))))
                rec["ret_se"].append(float(np.mean(err**2)))

    summary = {
        "by_horizon": {},
        "horizons": list(HORIZONS),
        "n_origins": n_origins,
        "protocol": "Step 11a expanding-window intercept, frozen train-end Phi, observed lags",
    }
    print(f"Origins {n_origins}")
    for h in HORIZONS:
        summary["by_horizon"][str(h)] = {}
        print(f"\n--- horizon {h} ---")
        for name in names:
            rec = by_h[str(h)][name]
            d = {
                "price_mape": float(np.mean(rec["ape"]) * 100),
                "price_mae": float(np.mean(rec["ae"])),
                "ret_mae": float(np.mean(rec["ret_ae"])),
                "ret_rmsfe": float(np.sqrt(np.mean(rec["ret_se"]))),
                "n_origins": n_origins,
            }
            summary["by_horizon"][str(h)][name] = d
            print(
                f"  {name:22s}  MAPE {d['price_mape']:6.2f}%  MAE {d['price_mae']:7.2f}  "
                f"ret MAE {d['ret_mae']*100:5.3f}pp  RMSFE {d['ret_rmsfe']*100:5.3f}pp"
            )

    extra = {name: {"ae": [], "ape": [], "ret_ae": []} for name in names}
    for origin in range(t_train_end, t_price - 1):
        r1 = ret[origin - 1]
        r2 = ret[origin - 2]
        mu = minnesota_intercept(ret[:origin])
        last_px = pmat[origin]
        hats = {
            "LS-LR-NVAR spec": iterate_phi(1, r1, r2, mu, spec_b1, spec_b2)[0],
            "LS-LR-NVAR v2": iterate_phi(1, r1, r2, mu, v2_b1, v2_b2)[0],
            "TVP-LR-NVAR-SV": iterate_phi(1, r1, r2, mu, tvp_b1, tvp_b2)[0],
            "Chen direct": step_chen(r1, r2),
            "BCVAR": step_bcvar(r1, r2, bc),
            "BCVAR-TVP": step_bcvar(r1, r2, bct),
            "Random walk": np.zeros_like(r1),
        }
        for name, rhat in hats.items():
            px = last_px * np.exp(rhat)
            ae = np.abs(px - pmat[origin + 1])
            extra[name]["ae"].append(float(ae.mean()))
            extra[name]["ape"].append(float(np.mean(ae / np.maximum(np.abs(pmat[origin + 1]), 1e-8))))
            extra[name]["ret_ae"].append(float(np.mean(np.abs(rhat - ret[origin]))))
    summary["one_step_all"] = {}
    print(f"\n--- 1-step on all {len(extra['Random walk']['ae'])} holdout days ---")
    for name in names:
        d = {
            "price_mape": float(np.mean(extra[name]["ape"]) * 100),
            "price_mae": float(np.mean(extra[name]["ae"])),
            "ret_mae": float(np.mean(extra[name]["ret_ae"])),
            "n_origins": len(extra[name]["ae"]),
        }
        summary["one_step_all"][name] = d
        print(f"  {name:22s}  MAPE {d['price_mape']:6.2f}%  MAE {d['price_mae']:7.2f}")

    summary["window"] = {
        "train_end": str(prices.index[t_train_end].date()),
        "test_start": str(prices.index[test_start].date()),
        "test_end": str(prices.index[-1].date()),
        "protocol": "Step 11a: frozen Phi, expanding Minnesota intercept, actual lags",
    }
    (DATA_DIR / "rolling_forecast_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_html(summary, DATA_DIR / "rolling_forecast_comparison.html")
    print("Saved data/rolling_forecast_comparison.json / .html")


if __name__ == "__main__":
    main()
