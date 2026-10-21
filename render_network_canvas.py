#!/usr/bin/env python3
"""
Rebuild the network-measures canvas as a standalone HTML report.

Reads data/network_measures_over_time.json (from compare_networks_over_time.py)
and writes the same sections and charts as DJIA-network-measures-over-time.canvas.tsx.

Usage:
    python render_network_canvas.py
    python render_network_canvas.py --data-dir data --out data/network_measures_report.html
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from datetime import datetime
from pathlib import Path


def _nearest(dates: list[datetime], values: list[float], t: datetime) -> float:
    i = min(range(len(dates)), key=lambda k: abs((dates[k] - t).days))
    return float(values[i])


def _pct_series(xs: list[float], scale: float = 100.0) -> list[float]:
    return [round(float(v) * scale, 2) for v in xs]


def _short_label(iso: str) -> str:
    d = datetime.fromisoformat(iso)
    return d.strftime("%b %y")


def build_payload(data_dir: Path) -> dict:
    src = data_dir / "network_measures_over_time.json"
    raw = json.loads(src.read_text(encoding="utf-8"))
    ours_dates = [datetime.fromisoformat(d) for d in raw["ours"]["dates"]]
    chen_dates = [datetime.fromisoformat(d) for d in raw["chen_fa"]["dates"]]
    labels = [_short_label(d) for d in raw["chen_fa"]["dates"]]

    def align(key: str) -> list[float]:
        return [_nearest(ours_dates, raw["ours"][key], t) for t in chen_dates]

    s = raw["summary"]
    return {
        "window": raw["window"],
        "summary": s,
        "labels": labels,
        "density": {
            "ours": _pct_series(align("density")),
            "fa": _pct_series(raw["chen_fa"]["density"]),
            "direct": _pct_series(raw["chen_dir"]["density"]),
        },
        "reciprocity": {
            "ours": _pct_series(align("reciprocity")),
            "fa": _pct_series(raw["chen_fa"]["reciprocity"]),
            "direct": _pct_series(raw["chen_dir"]["reciprocity"]),
        },
        "clustering": {
            "ours": _pct_series(align("clustering")),
            "fa": _pct_series(raw["chen_fa"]["clustering"]),
            "direct": _pct_series(raw["chen_dir"]["clustering"]),
        },
        "gini": {
            "ours": [round(v, 3) for v in align("out_degree_gini")],
            "fa": [round(float(v), 3) for v in raw["chen_fa"]["out_degree_gini"]],
            "direct": [round(float(v), 3) for v in raw["chen_dir"]["out_degree_gini"]],
        },
        "lcc": {
            "ours": _pct_series(align("largest_component_share")),
            "fa": _pct_series(raw["chen_fa"]["largest_component_share"]),
            "direct": _pct_series(raw["chen_dir"]["largest_component_share"]),
        },
        "means": {
            "ours": [
                round(s["LS-LR-NVAR"]["mean_density"] * 100, 1),
                round(s["LS-LR-NVAR"]["mean_reciprocity"] * 100, 1),
                round(s["LS-LR-NVAR"]["mean_clustering"] * 100, 1),
                round(s["LS-LR-NVAR"]["mean_gini"] * 100, 1),
                round(s["LS-LR-NVAR"]["mean_lcc"] * 100, 1),
            ],
            "fa": [
                round(s["Chen factor-adjusted LASSO"]["mean_density"] * 100, 1),
                round(s["Chen factor-adjusted LASSO"]["mean_reciprocity"] * 100, 1),
                round(s["Chen factor-adjusted LASSO"]["mean_clustering"] * 100, 1),
                round(s["Chen factor-adjusted LASSO"]["mean_gini"] * 100, 1),
                round(s["Chen factor-adjusted LASSO"]["mean_lcc"] * 100, 1),
            ],
            "direct": [
                round(s["Chen direct LASSO"]["mean_density"] * 100, 1),
                round(s["Chen direct LASSO"]["mean_reciprocity"] * 100, 1),
                round(s["Chen direct LASSO"]["mean_clustering"] * 100, 1),
                round(s["Chen direct LASSO"]["mean_gini"] * 100, 1),
                round(s["Chen direct LASSO"]["mean_lcc"] * 100, 1),
            ],
        },
        "tops": {
            "ours": [raw["ours"]["top_start"], raw["ours"]["top_mid"], raw["ours"]["top_end"]],
            "fa": [
                raw["chen_fa"]["top_start"] or "empty graph",
                raw["chen_fa"]["top_mid"],
                raw["chen_fa"]["top_end"],
            ],
            "direct": [raw["chen_dir"]["top_start"], raw["chen_dir"]["top_mid"], raw["chen_dir"]["top_end"]],
        },
        "stats": {
            "ours_dens": f"{s['LS-LR-NVAR']['mean_density'] * 100:.1f}%",
            "fa_dens": f"{s['Chen factor-adjusted LASSO']['mean_density'] * 100:.1f}%",
            "dir_dens": f"{s['Chen direct LASSO']['mean_density'] * 100:.1f}%",
            "ours_recip": f"{s['LS-LR-NVAR']['mean_reciprocity'] * 100:.1f}%",
        },
    }


def html_report(payload: dict) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)
    w = payload["window"]
    t = payload["tops"]
    explained = f"{w['factor_var_explained'] * 100:.1f}%"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Network measures over time — LS-LR-NVAR vs Chen TV-LASSO</title>
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
    border-radius: 999px; padding: 3px 10px; font-size: 12px;
  }}
  .pill.active {{ background: #2a2a2a; color: var(--text); border-color: #555; }}
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
  .stat.dir .v {{ color: var(--dir); }}
  .cards {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin: 16px 0; }}
  .card {{ border: 1px solid var(--stroke); background: var(--card); border-radius: 8px; }}
  .card-h {{ font-size: 12px; color: var(--muted); padding: 10px 14px; border-bottom: 1px solid var(--stroke);
            display: flex; justify-content: space-between; }}
  .card-b {{ padding: 12px 14px; }}
  .chart {{ background: var(--card); border: 1px solid var(--stroke); border-radius: 8px; padding: 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; border: 1px solid var(--stroke); border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: 7px 10px; border-bottom: 1px solid var(--stroke); text-align: left; }}
  th {{ background: #202020; position: sticky; top: 0; font-weight: 600; }}
  tr:nth-child(even) {{ background: #1c1c1c; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  hr {{ border: 0; border-top: 1px solid var(--stroke); margin: 28px 0; }}
  @media (max-width: 900px) {{
    .stats, .cards, .grid2 {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Network measures over time — LS-LR-NVAR vs Chen TV-LASSO</h1>
  <p class="lead">Directed graphs on the 30 current DJIA names, training window {w['start']} to {w['end']} (2,387 return days). LS-LR-NVAR is the posterior-mean probit adjacency (edge if mean probability ≥ 0.5). Chen et al. is a local-linear LASSO Granger graph: j → i if any lag of j is nonzero in equation i. Primary Chen spec is Section 5 factor-adjusted idiosyncratics (PCA k = 3, {explained} of return variance).</p>
  <div class="pills">
    <span class="pill active">Train through {w['end']}</span>
    <span class="pill">Ours: month-end of daily A<sub>t</sub></span>
    <span class="pill">Chen: 36 local graphs, tau in [0.18, 1]</span>
  </div>

  <div class="stats">
    <div class="stat ours"><div class="v">{payload['stats']['ours_dens']}</div><div class="l">LS-LR-NVAR mean density</div></div>
    <div class="stat fa"><div class="v">{payload['stats']['fa_dens']}</div><div class="l">Chen factor-adjusted mean density</div></div>
    <div class="stat dir"><div class="v">{payload['stats']['dir_dens']}</div><div class="l">Chen direct LASSO mean density</div></div>
    <div class="stat"><div class="v">{payload['stats']['ours_recip']}</div><div class="l">LS-LR-NVAR mean reciprocity</div></div>
  </div>

  <div class="callout">
    <strong>These are not the same kind of edge</strong>
    LS-LR-NVAR estimates a contemporaneous association network (probit latents plus a |corr| ≥ 0.35 floor). Chen estimates time-varying Granger links after L1 shrinkage. A denser LS-LR-NVAR graph is expected; the interesting comparison is how topology and time variation differ once that is granted.
  </div>

  <h2>Average topology</h2>
  <p class="note">Means over the aligned 2018-04 to 2026-02 grid (36 Chen dates; LS-LR-NVAR taken at the nearest month-end). Density is directed (870 possible edges). Clustering is undirected transitivity. Gini is out-degree inequality (0 = equal, 1 = one hub).</p>
  <div class="chart"><canvas id="means" height="90"></canvas></div>
  <p class="caption">Source: compare_networks_over_time.py · djia_adj_close_wide_10y.csv · n = 30.</p>

  <div class="cards">
    <div class="card"><div class="card-h">LS-LR-NVAR<span>association</span></div>
      <div class="card-b">Almost complete, nearly undirected, and egalitarian: ~684 of 870 directed edges, reciprocity 99.7%, clustering 86.5%, Gini 0.12, one connected component. Highest out-degree names stay HD, CSCO, HON, SHW.
      <p class="caption">Density moves only 78.3–79.0% over nine years. The correlation floor and a tightly clipped RW step on the dynamic latents keep the graph nearly static.</p></div></div>
    <div class="card"><div class="card-h">Factor-adjusted LASSO<span>primary Chen spec</span></div>
      <div class="card-b">After removing three return factors, residual Granger structure is thin: mean 9 edges, density 1.1%, reciprocity 1.3%. The graph is often empty or a few spokes. Mean largest-component share is 24%.
      <p class="caption">Two mild density bumps: 2019–2020 (peak 3.1%) and late 2025–early 2026 (end 2.2%, hubs MMM and AMZN).</p></div></div>
    <div class="card"><div class="card-h">Direct LASSO<span>robustness</span></div>
      <div class="card-b">Raw-return Granger graphs are denser and more cyclic: mean density 9.8%, clustering 28.6%, largest component 76%. Reciprocity is 36% around 2020 and zero after 2022.
      <p class="caption">Common factors create many contemporaneous-looking lags. End-sample hubs are GS (17), AXP (13), JPM (11).</p></div></div>
  </div>

  <h2>Density over time</h2>
  <p class="note">LS-LR-NVAR stays near 79% on this scale, so the two LASSO series sit on the axis. The next chart zooms those two.</p>
  <div class="chart"><canvas id="densityAll" height="90"></canvas></div>
  <p class="caption">Directed density (%) · LS-LR-NVAR month-end nearest to each Chen tau · 2018-04-24 to {w['end']}.</p>

  <h3>LASSO density only</h3>
  <p class="note">Direct TV-LASSO rises through 2019, peaks at 23.7% in July 2020, collapses through 2022 (2.3% in January 2022), then rebuilds to 8.9% by February 2026. Factor-adjusted stays below 3.1% throughout.</p>
  <div class="chart"><canvas id="densityLasso" height="90"></canvas></div>
  <p class="caption">Directed density (%) · local-linear LASSO Granger graphs, same penalty as LassoCV at tau = 1.</p>

  <h2>Reciprocity and clustering</h2>
  <div class="grid2">
    <div>
      <h3>Reciprocity</h3>
      <div class="chart"><canvas id="recip" height="140"></canvas></div>
      <p class="caption">Share of directed edges that have a reverse edge. LS-LR-NVAR is essentially undirected. Direct LASSO is mutual only in the 2019–2021 dense spell.</p>
    </div>
    <div>
      <h3>Undirected clustering</h3>
      <div class="chart"><canvas id="clust" height="140"></canvas></div>
      <p class="caption">Global transitivity on the symmetrized graph. Direct LASSO triangles peak with COVID-era density; factor-adjusted triangles are rare.</p>
    </div>
  </div>

  <h2>Concentration and connectivity</h2>
  <div class="grid2">
    <div>
      <h3>Out-degree Gini</h3>
      <div class="chart"><canvas id="gini" height="140"></canvas></div>
      <p class="caption">LS-LR-NVAR degrees are nearly equal (Gini ~0.12). LASSO graphs are hub-and-spoke (Gini 0.6–0.9). Factor-adjusted Gini is 0 when the graph is empty.</p>
    </div>
    <div>
      <h3>Largest weakly connected component</h3>
      <div class="chart"><canvas id="lcc" height="140"></canvas></div>
      <p class="caption">LS-LR-NVAR is always one component. Direct LASSO is fully connected in 2019–mid-2021, then fragments. Factor-adjusted never exceeds 63% of names.</p>
    </div>
  </div>

  <h2>Who is central</h2>
  <p class="note">Top five out-degree names at the start, middle, and end of each series. Counts are directed out-degree (number of names that name Granger-causes, or, for LS-LR-NVAR, number of association neighbors).</p>
  <table>
    <thead><tr><th>Model</th><th>Start (2018-04 / 2016-08)</th><th>Mid-sample</th><th>End ({w['end']})</th></tr></thead>
    <tbody>
      <tr><td>LS-LR-NVAR</td><td>{t['ours'][0]}</td><td>{t['ours'][1]}</td><td>{t['ours'][2]}</td></tr>
      <tr><td>Chen factor-adjusted</td><td>{t['fa'][0]}</td><td>{t['fa'][1]}</td><td>{t['fa'][2]}</td></tr>
      <tr><td>Chen direct LASSO</td><td>{t['direct'][0]}</td><td>{t['direct'][1]}</td><td>{t['direct'][2]}</td></tr>
    </tbody>
  </table>
  <p class="caption">LS-LR-NVAR hubs barely change. Chen direct concentrates Granger influence in GS, AXP, and JPM by the forecast origin.</p>

  <hr/>
  <h2>Protocol</h2>
  <div class="grid2">
    <div>
      <h3>LS-LR-NVAR</h3>
      <p class="note">{w['ours_resolution']}. Blocked Gibbs, 24 sweeps, 12 burn-in, thin 2 (6 kept draws). Topology at probability 0.5. Same correlation floor and tau clip as the 6-month forecast program.</p>
    </div>
    <div>
      <h3>Chen TV-LASSO</h3>
      <p class="note">{w['chen_resolution']}. Equation-wise LassoCV once at tau = 1, then that lambda reused on the grid with local-linear Epanechnikov weights (h = 0.20, p = 2). Own lags are not counted as Granger edges.</p>
    </div>
  </div>
  <p class="caption">Rebuilt from JSON by render_network_canvas.py · data/network_measures_over_time.json · compare_networks_over_time.py</p>
</div>
<script>
const DATA = {data_json};
const colors = {{
  ours: "#3fb950",
  fa: "#f85149",
  direct: "#d29922",
  grid: "#333333",
  text: "#9a9a9a"
}};
Chart.defaults.color = colors.text;
Chart.defaults.borderColor = colors.grid;
Chart.defaults.font.family = "ui-sans-serif, system-ui, Segoe UI, sans-serif";

function lineSet(label, data, color) {{
  return {{
    label, data,
    borderColor: color, backgroundColor: color,
    tension: 0.15, pointRadius: 2, pointHoverRadius: 4, borderWidth: 2
  }};
}}

function lineChart(id, datasets, yLabel, beginAtZero) {{
  return new Chart(document.getElementById(id), {{
    type: "line",
    data: {{ labels: DATA.labels, datasets }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ position: "top" }} }},
      scales: {{
        y: {{ beginAtZero: beginAtZero !== false, title: {{ display: true, text: yLabel }} }},
        x: {{ ticks: {{ maxRotation: 60, minRotation: 45, autoSkip: true, maxTicksLimit: 18 }} }}
      }}
    }}
  }});
}}

new Chart(document.getElementById("means"), {{
  type: "bar",
  data: {{
    labels: ["Density (%)", "Reciprocity (%)", "Clustering (%)", "Out-degree Gini × 100", "Largest component (%)"],
    datasets: [
      {{ label: "LS-LR-NVAR", data: DATA.means.ours, backgroundColor: colors.ours }},
      {{ label: "Chen factor-adjusted LASSO", data: DATA.means.fa, backgroundColor: colors.fa }},
      {{ label: "Chen direct LASSO", data: DATA.means.direct, backgroundColor: colors.direct }}
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ position: "top" }} }},
    scales: {{ y: {{ beginAtZero: true }} }}
  }}
}});

lineChart("densityAll", [
  lineSet("LS-LR-NVAR", DATA.density.ours, colors.ours),
  lineSet("Chen factor-adjusted LASSO", DATA.density.fa, colors.fa),
  lineSet("Chen direct LASSO", DATA.density.direct, colors.direct)
], "Directed density (%)");

lineChart("densityLasso", [
  lineSet("Chen factor-adjusted LASSO", DATA.density.fa, colors.fa),
  lineSet("Chen direct LASSO", DATA.density.direct, colors.direct)
], "Directed density (%)");

lineChart("recip", [
  lineSet("LS-LR-NVAR", DATA.reciprocity.ours, colors.ours),
  lineSet("Chen factor-adjusted LASSO", DATA.reciprocity.fa, colors.fa),
  lineSet("Chen direct LASSO", DATA.reciprocity.direct, colors.direct)
], "Reciprocity (%)");

lineChart("clust", [
  lineSet("LS-LR-NVAR", DATA.clustering.ours, colors.ours),
  lineSet("Chen factor-adjusted LASSO", DATA.clustering.fa, colors.fa),
  lineSet("Chen direct LASSO", DATA.clustering.direct, colors.direct)
], "Clustering (%)");

lineChart("gini", [
  lineSet("LS-LR-NVAR", DATA.gini.ours, colors.ours),
  lineSet("Chen factor-adjusted LASSO", DATA.gini.fa, colors.fa),
  lineSet("Chen direct LASSO", DATA.gini.direct, colors.direct)
], "Out-degree Gini", false);

lineChart("lcc", [
  lineSet("LS-LR-NVAR", DATA.lcc.ours, colors.ours),
  lineSet("Chen factor-adjusted LASSO", DATA.lcc.fa, colors.fa),
  lineSet("Chen direct LASSO", DATA.lcc.direct, colors.direct)
], "Largest component (%)");
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
        if (cand / "network_measures_over_time.json").exists():
            return cand
    return _project_dir() / "data"


def main(
    data_dir: Path | str | None = None,
    out: Path | str | None = None,
    open_browser: bool = True,
) -> Path:
    """Rebuild the network canvas HTML. Safe to call as main() from a Jupyter cell."""
    data_dir = Path(data_dir) if data_dir is not None else _default_data_dir()
    out_path = Path(out) if out is not None else data_dir / "network_measures_report.html"
    payload = build_payload(data_dir)
    out_path.write_text(html_report(payload), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(
        "Mean density  LS-LR-NVAR {0}  Chen FA {1}  Chen direct {2}".format(
            payload["stats"]["ours_dens"],
            payload["stats"]["fa_dens"],
            payload["stats"]["dir_dens"],
        )
    )
    if open_browser:
        webbrowser.open(out_path.resolve().as_uri())
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild the network-measures canvas as HTML.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-open", action="store_true", help="Do not open the HTML in a browser.")
    args = parser.parse_args()
    main(data_dir=args.data_dir, out=args.out, open_browser=not args.no_open)
