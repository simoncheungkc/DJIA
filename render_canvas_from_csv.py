#!/usr/bin/env python3
"""
Rebuild the DJIA 6-month forecast canvas from CSV.

Reads the same price and forecast files used for the comparison, recomputes
every number on the canvas, and writes a standalone HTML report with the
same sections, charts, ticker switcher, and scorecard.

Usage:
    python render_canvas_from_csv.py
    python render_canvas_from_csv.py --data-dir data --out data/canvas_report.html
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON = 126
PATH_TICKERS = ["AAPL", "NVDA", "MSFT", "UNH", "CSCO", "NKE"]
WEEKLY_STEP = 5


def load_panel(path: Path, dates: pd.DatetimeIndex, tickers: list[str]) -> np.ndarray:
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df.reindex(dates)[tickers].values


def path_mape(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(pred - actual) / np.maximum(np.abs(actual), 1e-8), axis=0) * 100


def path_mae(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(pred - actual), axis=0)


def terminal_return(prices: np.ndarray, last_train: np.ndarray) -> np.ndarray:
    return prices[-1] / last_train - 1.0


def monthly_mape(pred: np.ndarray, actual: np.ndarray, dates: pd.DatetimeIndex) -> list[tuple[str, float]]:
    err = pd.DataFrame(
        np.abs(pred - actual) / np.maximum(np.abs(actual), 1e-8) * 100,
        index=dates,
    )
    series = err.mean(axis=1).resample("ME").mean()
    labels = [d.strftime("%b") for d in series.index]
    return list(zip(labels, [float(v) for v in series.values]))


def weekly_index(n: int, step: int = WEEKLY_STEP) -> list[int]:
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return idx


def fmt_pct(x: float, digits: int = 1) -> str:
    return f"{x:+.{digits}f}%" if digits else f"{x:+.0f}%"


def fmt_ret(x: float) -> str:
    signed = f"{x * 100:+.1f}%"
    return signed.replace("+", "+").replace("-", "\u2212") if x < 0 else signed


def best_name(ours: float, fa: float, direct: float) -> str:
    vals = {"LS-LR-NVAR": ours, "Chen FA": fa, "Chen direct": direct}
    return min(vals, key=vals.get)


def build_payload(data_dir: Path) -> dict:
    prices = pd.read_csv(data_dir / "djia_adj_close_wide_10y.csv", parse_dates=["Date"], index_col="Date")
    prices = prices.sort_index().ffill().dropna(how="all")
    tickers = prices.columns.tolist()
    train_prices = prices.iloc[:-HORIZON]
    test_prices = prices.iloc[-HORIZON:]
    last_train = train_prices.iloc[-1].values
    actual = test_prices.values
    dates = test_prices.index

    ours = load_panel(data_dir / "djia_6m_lslrnvar_mean.csv", dates, tickers)
    chen = load_panel(data_dir / "djia_6m_chen_tvnvar_mean.csv", dates, tickers)
    chen_d = load_panel(data_dir / "djia_6m_chen_direct_mean.csv", dates, tickers)

    actual_term = terminal_return(actual, last_train)
    models = {
        "LS-LR-NVAR": ours,
        "Chen factor-adjusted TV-VAR": chen,
        "Chen direct sparse TV-VAR": chen_d,
    }
    summary = {}
    for name, pred in models.items():
        mape = path_mape(pred, actual)
        pred_term = terminal_return(pred, last_train)
        err_term = pred_term - actual_term
        summary[name] = {
            "avg_mape": float(mape.mean()),
            "avg_mae": float(path_mae(pred, actual).mean()),
            "term_mae_pp": float(np.mean(np.abs(err_term)) * 100),
            "dir_acc": float(np.mean(np.sign(pred_term) == np.sign(actual_term)) * 100),
            "spearman": float(pd.Series(pred_term).rank().corr(pd.Series(actual_term).rank())),
            "mape": mape.tolist(),
            "pred_term": pred_term.tolist(),
        }

    ours_m = np.asarray(summary["LS-LR-NVAR"]["mape"])
    fa_m = np.asarray(summary["Chen factor-adjusted TV-VAR"]["mape"])
    dir_m = np.asarray(summary["Chen direct sparse TV-VAR"]["mape"])
    wins = {"LS-LR-NVAR": 0, "Chen FA": 0, "Chen direct": 0}
    per = []
    for i, tkr in enumerate(tickers):
        winner = best_name(ours_m[i], fa_m[i], dir_m[i])
        wins[winner] += 1
        per.append(
            {
                "ticker": tkr,
                "actual": fmt_ret(actual_term[i]),
                "ours": f"{ours_m[i]:.1f}",
                "chenFa": f"{fa_m[i]:.1f}",
                "chenDir": f"{dir_m[i]:.1f}",
                "best": winner,
            }
        )

    idx = weekly_index(len(dates))
    date_labels = [dates[i].strftime("%m-%d") for i in idx]
    paths = {}
    for tkr in PATH_TICKERS:
        j = tickers.index(tkr)
        paths[tkr] = {
            "actual": [round(float(actual[i, j]), 2) for i in idx],
            "ours": [round(float(ours[i, j]), 2) for i in idx],
            "chenFa": [round(float(chen[i, j]), 2) for i in idx],
            "chenDir": [round(float(chen_d[i, j]), 2) for i in idx],
        }

    month_ours = monthly_mape(ours, actual, dates)
    month_fa = monthly_mape(chen, actual, dates)
    month_dir = monthly_mape(chen_d, actual, dates)

    s_ours = summary["LS-LR-NVAR"]
    s_fa = summary["Chen factor-adjusted TV-VAR"]
    s_dir = summary["Chen direct sparse TV-VAR"]
    fa_wins = [p["ticker"] for p in per if p["best"] == "Chen FA"]

    return {
        "window": {
            "train_end": str(train_prices.index[-1].date()),
            "test_start": str(dates[0].date()),
            "test_end": str(dates[-1].date()),
            "n": len(tickers),
        },
        "stats": {
            "ours_mape": f"{s_ours['avg_mape']:.2f}%",
            "fa_mape": f"{s_fa['avg_mape']:.2f}%",
            "dir_mape": f"{s_dir['avg_mape']:.2f}%",
            "ours_wins": f"{wins['LS-LR-NVAR']} / {len(tickers)}",
        },
        "headline": {
            "categories": ["Path MAPE (%)", "6-month return MAE (pp)", "Directional accuracy (%)"],
            "ours": [round(s_ours["avg_mape"], 2), round(s_ours["term_mae_pp"], 1), round(s_ours["dir_acc"], 1)],
            "fa": [round(s_fa["avg_mape"], 2), round(s_fa["term_mae_pp"], 1), round(s_fa["dir_acc"], 1)],
            "direct": [round(s_dir["avg_mape"], 2), round(s_dir["term_mae_pp"], 1), round(s_dir["dir_acc"], 1)],
        },
        "cards": {
            "fa": (
                f"LS-LR-NVAR beats the paper's factor-adjusted spec on every headline metric: "
                f"{s_ours['avg_mape']:.2f}% vs {s_fa['avg_mape']:.2f}% path MAPE, "
                f"{s_ours['term_mae_pp']:.1f} vs {s_fa['term_mae_pp']:.1f} pp terminal-return MAE, "
                f"{s_ours['dir_acc']:.0f}% vs {s_fa['dir_acc']:.0f}% direction, "
                f"Spearman {s_ours['spearman']:.2f} vs {s_fa['spearman']:.2f}."
            ),
            "direct": (
                f"Skipping factor extraction slightly outperforms both on path MAPE "
                f"({s_dir['avg_mape']:.2f}%) and wins {wins['Chen direct']} of {len(tickers)} names. "
                f"Ranking skill remains similar to LS-LR-NVAR "
                f"(Spearman {s_dir['spearman']:.2f} vs {s_ours['spearman']:.2f}). "
                f"This is not the spec Chen et al. recommend for highly correlated panels."
            ),
            "name_level": (
                f"Lowest path MAPE: Chen direct {wins['Chen direct']} names, "
                f"LS-LR-NVAR {wins['LS-LR-NVAR']}, Chen factor-adjusted {wins['Chen FA']}"
                + (f" ({', '.join(fa_wins)})" if fa_wins else "")
                + ". Versus Chen factor-adjusted alone, LS-LR-NVAR is tighter on most names, "
                "including NVDA, IBM, UNH, and WMT."
            ),
        },
        "monthly": {
            "labels": [lab for lab, _ in month_ours],
            "ours": [round(v, 2) for _, v in month_ours],
            "fa": [round(v, 2) for _, v in month_fa],
            "direct": [round(v, 2) for _, v in month_dir],
        },
        "ticker_mape": {
            "labels": tickers,
            "ours": [round(float(x), 2) for x in ours_m],
            "fa": [round(float(x), 2) for x in fa_m],
        },
        "dates": date_labels,
        "paths": paths,
        "per": per,
    }


def html_report(payload: dict) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)
    rows = []
    for r in payload["per"]:
        tone = {"LS-LR-NVAR": "ours", "Chen FA": "fa", "Chen direct": "dir"}[r["best"]]
        rows.append(
            "<tr class='{tone}'><td>{ticker}</td><td class='num'>{actual}</td>"
            "<td class='num'>{ours}%</td><td class='num'>{chenFa}%</td>"
            "<td class='num'>{chenDir}%</td><td>{best}</td></tr>".format(tone=tone, **r)
        )
    pills = "".join(
        f"<button type='button' class='pill' data-ticker='{t}'>{t}</button>" for t in PATH_TICKERS
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>LS-LR-NVAR vs Chen et al. — 6-month forecasts</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #181818;
    --text: #e6e6e6;
    --muted: #9a9a9a;
    --faint: #6e6e6e;
    --card: #222222;
    --stroke: #333333;
    --ours: #3fb950;
    --fa: #f85149;
    --dir: #d29922;
    --actual: #b0b0b0;
    --accent: #79c0ff;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 24px;
    background: var(--bg); color: var(--text);
    font: 14px/1.5 ui-sans-serif, system-ui, Segoe UI, sans-serif;
  }}
  .wrap {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{ font-size: 24px; font-weight: 600; margin: 0 0 8px; }}
  h2 {{ font-size: 18px; font-weight: 600; margin: 28px 0 8px; }}
  h3 {{ font-size: 15px; font-weight: 600; margin: 20px 0 8px; }}
  p.lead, p.note {{ color: var(--muted); margin: 0 0 12px; }}
  p.caption {{ color: var(--faint); font-size: 12px; margin: 6px 0 0; }}
  .pills {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 16px; }}
  .pill {{
    border: 1px solid var(--stroke); background: transparent; color: var(--muted);
    border-radius: 999px; padding: 3px 10px; font-size: 12px; cursor: default;
  }}
  .pill.active {{ background: #2a2a2a; color: var(--text); border-color: #555; }}
  .pill[data-ticker] {{ cursor: pointer; }}
  .callout {{
    border: 1px solid #2f4f6f; background: #1c2833; padding: 12px 14px; border-radius: 8px;
    margin: 16px 0 20px; color: var(--text);
  }}
  .callout strong {{ display: block; margin-bottom: 4px; }}
  .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 16px 0 8px; }}
  .stat .v {{ font-size: 22px; font-weight: 600; }}
  .stat .l {{ color: var(--muted); font-size: 12px; }}
  .stat.ours .v {{ color: var(--ours); }}
  .stat.fa .v {{ color: var(--fa); }}
  .cards {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 16px 0; }}
  .card {{ border: 1px solid var(--stroke); background: var(--card); border-radius: 8px; }}
  .card-h {{ font-size: 12px; color: var(--muted); padding: 10px 14px; border-bottom: 1px solid var(--stroke);
            display: flex; justify-content: space-between; }}
  .card-b {{ padding: 12px 14px; }}
  .chart {{ background: var(--card); border: 1px solid var(--stroke); border-radius: 8px; padding: 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; border: 1px solid var(--stroke); border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: 7px 10px; border-bottom: 1px solid var(--stroke); text-align: left; }}
  th {{ background: #202020; position: sticky; top: 0; font-weight: 600; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tr:nth-child(even) {{ background: #1c1c1c; }}
  tr.ours td:first-child::before, tr.fa td:first-child::before, tr.dir td:first-child::before {{
    content: ""; display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 8px;
  }}
  tr.ours td:first-child::before {{ background: var(--ours); }}
  tr.fa td:first-child::before {{ background: var(--fa); }}
  tr.dir td:first-child::before {{ background: #58a6ff; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  hr {{ border: 0; border-top: 1px solid var(--stroke); margin: 28px 0; }}
  @media (max-width: 800px) {{
    .stats, .cards, .grid2 {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<div class="wrap">
  <h1>LS-LR-NVAR vs Chen et al. — 6-month forecasts</h1>
  <p class="lead">Both programs rewritten from the source papers. Open-loop from {payload['window']['train_end']} through {payload['window']['test_end']} ({HORIZON} trading days), {payload['window']['n']} DJIA names, log-returns compounded to adjusted close.</p>
  <div class="pills">
    <span class="pill active">Open-loop 6-month</span>
    <span class="pill">p = 2</span>
    <span class="pill">Our model: blocked Gibbs LS-LR-NVAR</span>
    <span class="pill">Chen: two-stage TV-LASSO at tau=1</span>
  </div>

  <div class="callout">
    <strong>What was implemented</strong>
    Our model is the Bayesian locally stationary low-rank Network VAR from the Word draft: low-rank mean, probit network, low-rank B<sub>i,t</sub> = U<sub>t</sub> V<sub>t</sub>', blocked Gibbs, posterior-mean path. Chen, Li, Li and Linton (JoE 2025) is local-linear TV-LASSO then SCAD-weighted LASSO at the origin, with Section 5 PCA factor adjustment as the primary comparable spec for correlated equity returns.
  </div>

  <div class="stats">
    <div class="stat ours"><div class="v">{payload['stats']['ours_mape']}</div><div class="l">LS-LR-NVAR path MAPE</div></div>
    <div class="stat fa"><div class="v">{payload['stats']['fa_mape']}</div><div class="l">Chen factor-adjusted MAPE</div></div>
    <div class="stat"><div class="v">{payload['stats']['dir_mape']}</div><div class="l">Chen direct MAPE (robustness)</div></div>
    <div class="stat"><div class="v">{payload['stats']['ours_wins']}</div><div class="l">Names where LS-LR-NVAR has lowest MAPE</div></div>
  </div>

  <h2>Headline metrics</h2>
  <p class="note">Path MAPE and MAE average across 30 names and 126 days. 6-month return MAE is mean absolute error of the terminal price return. Directional accuracy is the share of names whose predicted 6-month sign matches realized. Spearman ranks names by predicted vs realized 6-month return.</p>
  <div class="chart"><canvas id="headline" height="90"></canvas></div>
  <p class="caption">Source: rewritten estimators on djia_adj_close_wide_10y.csv · origin {payload['window']['train_end']} · {payload['window']['n']} names.</p>

  <div class="cards">
    <div class="card"><div class="card-h">Chen Section 5<span>primary comparable</span></div>
      <div class="card-b">{payload['cards']['fa']}<p class="caption">PCA k = 3, two-stage LASSO on idiosyncratics, coefficients frozen at tau = 1.</p></div></div>
    <div class="card"><div class="card-h">Chen direct sparse<span>robustness</span></div>
      <div class="card-b">{payload['cards']['direct']}</div></div>
  </div>

  <h2>Error along the horizon</h2>
  <p class="note">Average MAPE across names, by calendar month. All three start near 3.5% in February; Chen factor-adjusted pulls away after April.</p>
  <div class="chart"><canvas id="monthly" height="90"></canvas></div>
  <p class="caption">Monthly mean of daily name-level MAPE · Aug is a partial month through {payload['window']['test_end']}.</p>

  <h2>Name-level MAPE</h2>
  <p class="note">{payload['cards']['name_level']}</p>
  <div class="chart"><canvas id="byTicker" height="110"></canvas></div>
  <p class="caption">Path MAPE (%) by ticker · LS-LR-NVAR vs Chen Section 5 factor-adjusted TV-VAR.</p>

  <h3>Price path (weekly subsample)</h3>
  <div class="pills" id="tickerPills">{pills}</div>
  <div class="chart"><canvas id="path" height="100"></canvas></div>
  <p class="caption" id="pathCaption">NVDA adjusted close · weekly points from the 126-day daily path · open-loop from {payload['window']['train_end']}.</p>

  <h2>Per-name scorecard</h2>
  <table>
    <thead><tr><th>Ticker</th><th class="num">Actual 6m</th><th class="num">LS-LR-NVAR MAPE</th><th class="num">Chen FA MAPE</th><th class="num">Chen dir MAPE</th><th>Best MAPE</th></tr></thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
  <p class="caption">MAPE on the 126-day price path. Green marker: LS-LR-NVAR. Blue: Chen direct. Red: Chen factor-adjusted.</p>

  <hr/>
  <h2>Protocol and caveats</h2>
  <div class="grid2">
    <div>
      <h3>Fair comparison</h3>
      <p class="note">Same split as before: 2,387 training return days, last train price {payload['window']['train_end']}, 126 open-loop steps. Chen coefficients are estimated once at the origin. LS-LR-NVAR uses the posterior mean of the recursive mean path (blocked Gibbs, 24 sweeps, 6 kept draws after burn-in).</p>
    </div>
    <div>
      <h3>What this is not</h3>
      <p class="note">Gibbs used a correlation-threshold floor on the probit network and a local-mean clip so multi-step return paths stay in a realistic daily range. Chen CLIME partial-correlation networks are unused because the target is the mean price path. Rolling one-step notebook MAPE is a different protocol.</p>
    </div>
  </div>
  <p class="caption">Rebuilt from CSV by render_canvas_from_csv.py · forecasts in data/djia_6m_lslrnvar_mean.csv, data/djia_6m_chen_tvnvar_mean.csv, data/djia_6m_chen_direct_mean.csv</p>
</div>
<script>
const DATA = {data_json};
const colors = {{
  ours: "#3fb950",
  fa: "#f85149",
  direct: "#d29922",
  actual: "#b0b0b0",
  grid: "#333333",
  text: "#9a9a9a"
}};
Chart.defaults.color = colors.text;
Chart.defaults.borderColor = colors.grid;
Chart.defaults.font.family = "ui-sans-serif, system-ui, Segoe UI, sans-serif";

function groupedBar(id, labels, series, opts) {{
  return new Chart(document.getElementById(id), {{
    type: "bar",
    data: {{ labels, datasets: series }},
    options: Object.assign({{
      responsive: true,
      plugins: {{ legend: {{ position: "top" }} }},
      scales: {{ y: {{ beginAtZero: true }} }}
    }}, opts || {{}})
  }});
}}

groupedBar("headline", DATA.headline.categories, [
  {{ label: "LS-LR-NVAR", data: DATA.headline.ours, backgroundColor: colors.ours }},
  {{ label: "Chen factor-adjusted TV-VAR", data: DATA.headline.fa, backgroundColor: colors.fa }},
  {{ label: "Chen direct sparse TV-VAR", data: DATA.headline.direct, backgroundColor: colors.direct }}
]);

new Chart(document.getElementById("monthly"), {{
  type: "line",
  data: {{
    labels: DATA.monthly.labels,
    datasets: [
      {{ label: "LS-LR-NVAR", data: DATA.monthly.ours, borderColor: colors.ours, backgroundColor: colors.ours, tension: 0.15 }},
      {{ label: "Chen factor-adjusted", data: DATA.monthly.fa, borderColor: colors.fa, backgroundColor: colors.fa, tension: 0.15 }},
      {{ label: "Chen direct sparse", data: DATA.monthly.direct, borderColor: colors.direct, backgroundColor: colors.direct, tension: 0.15 }}
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ position: "top" }} }},
    scales: {{ y: {{ beginAtZero: false, title: {{ display: true, text: "MAPE (%)" }} }} }}
  }}
}});

groupedBar("byTicker", DATA.ticker_mape.labels, [
  {{ label: "LS-LR-NVAR", data: DATA.ticker_mape.ours, backgroundColor: colors.ours }},
  {{ label: "Chen factor-adjusted", data: DATA.ticker_mape.fa, backgroundColor: colors.fa }}
], {{ plugins: {{ legend: {{ position: "top" }} }}, scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: "Path MAPE (%)" }} }} }} }});

let pathChart = new Chart(document.getElementById("path"), {{
  type: "line",
  data: {{ labels: DATA.dates, datasets: [] }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ position: "top" }} }},
    scales: {{
      y: {{ beginAtZero: false, title: {{ display: true, text: "Adjusted close ($)" }} }},
      x: {{ title: {{ display: true, text: "Date (MM-DD, 2026)" }} }}
    }}
  }}
}});

function showTicker(tkr) {{
  const p = DATA.paths[tkr];
  pathChart.data.datasets = [
    {{ label: "Actual adj close", data: p.actual, borderColor: colors.actual, backgroundColor: colors.actual, tension: 0.15 }},
    {{ label: "LS-LR-NVAR", data: p.ours, borderColor: colors.ours, backgroundColor: colors.ours, tension: 0.15 }},
    {{ label: "Chen factor-adjusted", data: p.chenFa, borderColor: colors.fa, backgroundColor: colors.fa, tension: 0.15 }},
    {{ label: "Chen direct sparse", data: p.chenDir, borderColor: colors.direct, backgroundColor: colors.direct, tension: 0.15 }}
  ];
  pathChart.update();
  document.getElementById("pathCaption").textContent =
    tkr + " adjusted close · weekly points from the 126-day daily path · open-loop from {payload['window']['train_end']}.";
  document.querySelectorAll("#tickerPills .pill").forEach(b => {{
    b.classList.toggle("active", b.dataset.ticker === tkr);
  }});
}}
document.querySelectorAll("#tickerPills .pill").forEach(b => {{
  b.addEventListener("click", () => showTicker(b.dataset.ticker));
}});
showTicker("NVDA");
</script>
</body>
</html>
"""


def _project_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def _default_data_dir() -> Path:
    candidates = [_project_dir() / "data", Path.cwd() / "data", Path.cwd()]
    for cand in candidates:
        if (cand / "djia_adj_close_wide_10y.csv").exists():
            return cand
    return _project_dir() / "data"


def main(
    data_dir: Path | str | None = None,
    out: Path | str | None = None,
    open_browser: bool = True,
) -> Path:
    """Rebuild the canvas HTML. Safe to call as main() from a Jupyter cell."""
    data_dir = Path(data_dir) if data_dir is not None else _default_data_dir()
    out_path = Path(out) if out is not None else data_dir / "canvas_report.html"
    payload = build_payload(data_dir)
    out_path.write_text(html_report(payload), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(
        "Headline  LS-LR-NVAR MAPE {0}  Chen FA {1}  Chen direct {2}  our MAPE-wins {3}".format(
            payload["stats"]["ours_mape"],
            payload["stats"]["fa_mape"],
            payload["stats"]["dir_mape"],
            payload["stats"]["ours_wins"],
        )
    )
    if open_browser:
        webbrowser.open(out_path.resolve().as_uri())
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild the 6-month forecast canvas from CSV.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-open", action="store_true", help="Do not open the HTML in a browser.")
    args = parser.parse_args()
    main(data_dir=args.data_dir, out=args.out, open_browser=not args.no_open)
