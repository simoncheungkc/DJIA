"""Reprint stored DJIA forecast summaries used in Table 3.

Does not re-run the Gibbs sampler. Table C.3 firm rows are the manuscript
constituent table; their unweighted panel means are not these column means.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRICES = ROOT / "data" / "djia_adj_close_wide_10y.csv"
FORECAST = ROOT / "data" / "forecast_comparison_6m.csv"


def main() -> None:
    prices = pd.read_csv(PRICES, parse_dates=["Date"])
    print(f"DJIA panel: {prices.shape[0]} rows, {prices.shape[1] - 1} names, {prices['Date'].min().date()} to {prices['Date'].max().date()}")
    forecast = pd.read_csv(FORECAST)
    print(f"Stored open-loop file: {len(forecast)} names")
    print(f"TVP path MAPE column mean (Table 3 portfolio object): {forecast['tvp_mape'].mean():.2f}")
    print(f"TVP price MAE column mean: {forecast['tvp_mae'].mean():.2f}")
    print(f"Chen direct path MAPE column mean: {forecast['chen_direct_mape'].mean():.2f}")
    print("Table C.3 Panel Mean is the unweighted average of the manuscript firm rows, not this column mean.")


if __name__ == "__main__":
    main()
