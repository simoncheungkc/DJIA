#!/usr/bin/env python3
"""Replace the LS-LR-NVAR block of network_measures_over_time.* with revised-Gibbs A_mean."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_networks_over_time import monthly_from_daily

DATA_DIR = Path(__file__).resolve().parent / "data"


def main() -> None:
    npz = DATA_DIR / "lslrnvar_A_mean_train_v2.npz"
    if not npz.exists():
        raise SystemExit("missing data/lslrnvar_A_mean_train_v2.npz — run ls_lr_nvar_6m_forecast.py first")
    z = np.load(npz, allow_pickle=True)
    A_mean = z["A_mean"]
    prices = pd.read_csv(DATA_DIR / "djia_adj_close_wide_10y.csv", parse_dates=["Date"], index_col="Date")
    fallback = prices.sort_index().index[1:-126]
    if "dates" in z.files:
        try:
            dates = pd.to_datetime(z["dates"])
        except (ValueError, TypeError):
            dates = fallback
        if len(dates) != A_mean.shape[0]:
            dates = fallback
    else:
        dates = fallback
    prices = pd.read_csv(DATA_DIR / "djia_adj_close_wide_10y.csv", parse_dates=["Date"], index_col="Date")
    tickers = prices.sort_index().ffill().dropna(how="all").columns.tolist()
    ours = monthly_from_daily(pd.DatetimeIndex(dates), A_mean, tickers)
    ours.insert(0, "model", "LS-LR-NVAR")

    csv = DATA_DIR / "network_measures_over_time.csv"
    if csv.exists():
        old = pd.read_csv(csv)
        other = old[old["model"] != "LS-LR-NVAR"]
        long = pd.concat([ours, other], ignore_index=True)
    else:
        long = ours
    long.to_csv(csv, index=False)

    js = DATA_DIR / "network_measures_over_time.json"
    payload = json.loads(js.read_text(encoding="utf-8")) if js.exists() else {"window": {}, "summary": {}}
    metric_cols = [
        "density", "expected_density", "n_edges", "mean_out_degree", "reciprocity",
        "clustering", "largest_component_share", "out_degree_gini", "n_components", "max_out_degree",
    ]
    payload["ours"] = {
        "dates": ours["date"].astype(str).tolist(),
        **{c: [round(float(v), 4) for v in ours[c].tolist()] for c in metric_cols if c in ours},
        "top_start": ours.iloc[0]["top_out_degree"],
        "top_mid": ours.iloc[len(ours) // 2]["top_out_degree"],
        "top_end": ours.iloc[-1]["top_out_degree"],
    }
    payload["summary"]["LS-LR-NVAR"] = {
        "mean_density": float(ours["density"].mean()),
        "end_density": float(ours["density"].iloc[-1]),
        "mean_reciprocity": float(ours["reciprocity"].mean()),
        "mean_clustering": float(ours["clustering"].mean()),
        "mean_gini": float(ours["out_degree_gini"].mean()),
        "mean_lcc": float(ours["largest_component_share"].mean()),
        "mean_expected_density": float(ours["expected_density"].mean()),
    }
    payload["window"]["ours_resolution"] = (
        "month-end of daily posterior-mean probit probabilities from revised Gibbs "
        "(soft W, ridge δ=1, AR(1) ρ=0.98; topology threshold 0.5)"
    )
    js.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    s = payload["summary"]["LS-LR-NVAR"]
    print(
        f"Revised LS-LR-NVAR  mean density {s['mean_density']:.3f}  "
        f"end {s['end_density']:.3f}  expected {s.get('mean_expected_density', float('nan')):.3f}"
    )
    print("Updated network_measures_over_time.csv / .json")


if __name__ == "__main__":
    main()
