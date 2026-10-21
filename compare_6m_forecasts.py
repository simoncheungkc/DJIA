#!/usr/bin/env python3
"""Score LS-LR-NVAR vs Chen et al. vs Bayesian Compressed VAR, 6-month open-loop."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
HORIZON = 126

MODEL_FILES = [
    ("LS-LR-NVAR (spec Gibbs)", "djia_6m_lslrnvar_mean.csv", "lslrnvar"),
    ("LS-LR-NVAR revised-v2", "djia_6m_lslrnvar_revised_v2_mean.csv", "lslrnvar_v2"),
    ("TVP-LR-NVAR-SV", "djia_6m_tvplrnvarsv_mean.csv", "tvp"),
    ("Chen factor-adjusted TV-VAR", "djia_6m_chen_tvnvar_mean.csv", "chen_fa"),
    ("Chen direct sparse TV-VAR", "djia_6m_chen_direct_mean.csv", "chen_direct"),
    ("BCVAR (Koop et al. 2019)", "djia_6m_bcvar_mean.csv", "bcvar"),
    ("BCVAR-TVP (Koop et al. 2019)", "djia_6m_bcvar_tvp_mean.csv", "bcvar_tvp"),
]
WIN_LABEL = {
    "LS-LR-NVAR (spec Gibbs)": "LS-LR-NVAR",
    "LS-LR-NVAR revised-v2": "LS revised",
    "TVP-LR-NVAR-SV": "TVP-LR-NVAR-SV",
    "Chen factor-adjusted TV-VAR": "Chen FA",
    "Chen direct sparse TV-VAR": "Chen direct",
    "BCVAR (Koop et al. 2019)": "BCVAR",
    "BCVAR-TVP (Koop et al. 2019)": "BCVAR-TVP",
}


def metrics(pred: np.ndarray, actual: np.ndarray) -> dict[str, np.ndarray]:
    mae = np.mean(np.abs(pred - actual), axis=0)
    rmse = np.sqrt(np.mean((pred - actual) ** 2, axis=0))
    mape = np.mean(np.abs(pred - actual) / np.maximum(np.abs(actual), 1e-8), axis=0) * 100
    return {"mae": mae, "rmse": rmse, "mape": mape}


def terminal_return(prices: np.ndarray, last_train: np.ndarray) -> np.ndarray:
    return prices[-1] / last_train - 1.0


def monthly_mape(pred, actual, fc_dates):
    idx = pd.DatetimeIndex(fc_dates)
    df_err = pd.DataFrame(np.abs(pred - actual) / np.maximum(np.abs(actual), 1e-8) * 100, index=idx)
    return {str(p.date()): float(v) for p, v in df_err.mean(axis=1).resample("ME").mean().items()}


def write_html(payload: dict, path: Path) -> None:
    s = payload["summary"]
    wins = payload["wins_mape"]
    rows = []
    for name, d in s.items():
        rows.append(
            f"<tr><td>{name}</td>"
            f"<td class='num'>{d['avg_mape']:.2f}%</td>"
            f"<td class='num'>{d['avg_mae']:.2f}</td>"
            f"<td class='num'>{100*d['terminal_mae_ret']:.1f} pp</td>"
            f"<td class='num'>{100*d['directional_accuracy']:.0f}%</td>"
            f"<td class='num'>{d['spearman_6m_return']:.2f}</td></tr>"
        )
    win_txt = ", ".join(f"{k} {v}" for k, v in wins.items())
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>6-month OOS forecast: TVP-LR-NVAR-SV vs Chen vs BCVAR</title>
<style>
  body {{ margin:0; padding:24px; background:#181818; color:#e6e6e6;
         font:14px/1.5 ui-sans-serif, system-ui, Segoe UI, sans-serif; }}
  .wrap {{ max-width:1100px; margin:0 auto; }}
  h1 {{ font-size:22px; font-weight:600; margin:0 0 8px; }}
  p.lead, p.note {{ color:#9a9a9a; margin:0 0 12px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; border:1px solid #333; }}
  th, td {{ padding:7px 10px; border-bottom:1px solid #333; text-align:left; }}
  th {{ background:#202020; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  tr:nth-child(even) {{ background:#1c1c1c; }}
  .callout {{ border:1px solid #2d5a3d; background:#1a2b20; padding:12px 14px;
              border-radius:8px; margin:16px 0 20px; }}
  .callout strong {{ display:block; color:#3fb950; margin-bottom:4px; }}
</style></head><body><div class="wrap">
<h1>Out-of-sample 6-month price paths</h1>
<p class="lead">
  Holdout {payload['window']['test_start']} to {payload['window']['test_end']}
  ({payload['window']['horizon_days']} trading days, {payload['window']['n_stocks']} DJIA names).
  Open-loop prices trained through {payload['window']['train_end']}.
  LS-LR-NVAR uses specification Gibbs Steps 0–10 and Step 11 predictive simulation on log-returns.
  TVP-LR-NVAR-SV iterates log-prices; Chen and BCVAR iterate daily log-returns then exponentiate.
</p>
<div class="callout">
  <strong>Models</strong>
  LS-LR-NVAR and TVP-LR-NVAR-SV use a Minnesota predictive mean: 5% annual common
  premium, one-third of historical name-mean deviations, and the frozen origin
  network VAR. That replaces unconstrained random-walk intercepts (spec) and
  fading 21-day momentum (TVP v2), which over- or under-shot six-month prices.
  Chen factor-adjusted is the paper's recommended equity spec (PCA k=3 then sparse TV-VAR).
  Chen direct is sparse TV-VAR on raw returns.
  BCVAR / BCVAR-TVP are Koop, Korobilis and Pettenuzzo (JoE 2019).
</div>
<table>
<thead><tr><th>Specification</th><th class="num">Path MAPE</th><th class="num">Price MAE</th>
<th class="num">6m-return MAE</th><th class="num">Direction</th><th class="num">Spearman</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody></table>
<p class="note">Lowest path MAPE by name: {win_txt}. Source: compare_6m_forecasts.py.</p>
</div></body></html>
"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    prices = pd.read_csv(DATA_DIR / "djia_adj_close_wide_10y.csv", parse_dates=["Date"], index_col="Date")
    prices = prices.sort_index().ffill().dropna(how="all")
    tickers = prices.columns.tolist()
    train_prices = prices.iloc[:-HORIZON]
    test_prices = prices.iloc[-HORIZON:]
    last_train_price = train_prices.iloc[-1].values
    actual = test_prices.values
    fc_dates = test_prices.index

    models = {}
    for name, fname, _key in MODEL_FILES:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            print(f"SKIP missing {fname}")
            continue
        df = pd.read_csv(fpath, parse_dates=["Date"], index_col="Date")
        models[name] = df.reindex(fc_dates)[tickers].values

    actual_term = terminal_return(actual, last_train_price)
    summary = {}
    scored = {}
    terms = {}
    print(f"Window {fc_dates[0].date()} to {fc_dates[-1].date()}  n={len(tickers)}")
    for name, pred in models.items():
        m = metrics(pred, actual)
        pred_term = terminal_return(pred, last_train_price)
        err_term = pred_term - actual_term
        dir_acc = float(np.mean(np.sign(pred_term) == np.sign(actual_term)))
        spearman = float(pd.Series(pred_term).rank().corr(pd.Series(actual_term).rank()))
        summary[name] = {
            "avg_mae": float(m["mae"].mean()),
            "avg_rmse": float(m["rmse"].mean()),
            "avg_mape": float(m["mape"].mean()),
            "median_mape": float(np.median(m["mape"])),
            "terminal_mae_ret": float(np.mean(np.abs(err_term))),
            "terminal_rmse_ret": float(np.sqrt(np.mean(err_term**2))),
            "directional_accuracy": dir_acc,
            "spearman_6m_return": spearman,
        }
        scored[name] = m
        terms[name] = pred_term
        print(
            f"{name:36s}  MAPE {m['mape'].mean():6.2f}%  MAE {m['mae'].mean():7.2f}  "
            f"6m-ret MAE {np.mean(np.abs(err_term))*100:5.2f}pp  "
            f"dir {dir_acc*100:5.1f}%  Spearman {spearman:5.3f}"
        )

    per = []
    wins = {WIN_LABEL[n]: 0 for n in models}
    for j, tkr in enumerate(tickers):
        mapes = {WIN_LABEL[n]: scored[n]["mape"][j] for n in models}
        winner = min(mapes, key=mapes.get)
        wins[winner] += 1
        row = {
            "ticker": tkr,
            "actual_6m_ret": float(actual_term[j]),
            "winner_mape": winner,
        }
        for name, key in [(a, b) for a, _, b in MODEL_FILES if a in models]:
            row[f"{key}_mape"] = float(scored[name]["mape"][j])
            row[f"{key}_mae"] = float(scored[name]["mae"][j])
            row[f"{key}_6m_ret"] = float(terms[name][j])
        per.append(row)

    movers = np.argsort(np.abs(actual_term))[::-1][:4]
    chart_names = list(dict.fromkeys(["AAPL", "NVDA", "MSFT"] + [tickers[i] for i in movers]))
    path_series = {}
    for tkr in chart_names:
        j = tickers.index(tkr)
        rec = {"dates": [d.strftime("%Y-%m-%d") for d in fc_dates], "actual": actual[:, j].tolist()}
        for name, _f, key in MODEL_FILES:
            if name in models:
                rec[key] = models[name][:, j].tolist()
        path_series[tkr] = rec

    pd.DataFrame(per).to_csv(DATA_DIR / "forecast_comparison_6m.csv", index=False)
    payload = {
        "window": {
            "train_end": str(train_prices.index[-1].date()),
            "test_start": str(fc_dates[0].date()),
            "test_end": str(fc_dates[-1].date()),
            "horizon_days": HORIZON,
            "n_stocks": len(tickers),
        },
        "summary": summary,
        "per_ticker": per,
        "wins_mape": wins,
        "path_mape_monthly": {name: monthly_mape(pred, actual, fc_dates) for name, pred in models.items()},
        "path_series": path_series,
        "actual_term_ret": {t: float(actual_term[i]) for i, t in enumerate(tickers)},
        "pred_term_ret": {
            key: {t: float(terms[name][i]) for i, t in enumerate(tickers)}
            for name, _f, key in MODEL_FILES
            if name in models
        },
    }
    (DATA_DIR / "forecast_comparison_6m.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_html(payload, DATA_DIR / "forecast_comparison_6m.html")
    print("MAPE winners  " + "  ".join(f"{k} {v}" for k, v in wins.items()))
    print("Saved forecast_comparison_6m.csv / .json / .html")


if __name__ == "__main__":
    main()
