"""Download 10 years of daily prices for current DJIA components."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

# Current DJIA components (as of Nov 2024 reconstitution: NVDA & SHW replaced INTC & DOW)
DJIA_TICKERS = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA",
    "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "JNJ",
    "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW",
    "TRV", "UNH", "V", "VZ", "WMT",
]

OUT_DIR = Path(__file__).resolve().parent / "data"
PER_TICKER_DIR = OUT_DIR / "by_ticker"


def main() -> None:
    end = datetime.now().date()
    start = end - timedelta(days=365 * 10 + 5)  # slight buffer for leap years

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PER_TICKER_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(DJIA_TICKERS)} DJIA tickers")
    print(f"Range: {start} -> {end}")

    raw = yf.download(
        DJIA_TICKERS,
        start=start.isoformat(),
        end=end.isoformat(),
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=True,
    )

    frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for ticker in DJIA_TICKERS:
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker not in raw.columns.get_level_values(0):
                print(f"WARNING: missing ticker {ticker}")
                continue
            df = raw[ticker].dropna(how="all").copy()
        else:
            # Single-ticker fallback shape
            df = raw.dropna(how="all").copy()

        if df.empty:
            print(f"WARNING: no rows for {ticker}")
            continue

        df = df.reset_index()
        # Normalize column names
        df.columns = [str(c).strip() for c in df.columns]
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        df = df.rename(columns={date_col: "Date"})
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
        df.insert(0, "Ticker", ticker)

        keep = [
            c
            for c in ["Ticker", "Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
            if c in df.columns
        ]
        df = df[keep].sort_values("Date")

        per_path = PER_TICKER_DIR / f"{ticker}.csv"
        df.drop(columns=["Ticker"]).to_csv(per_path, index=False)

        frames.append(df)
        summary_rows.append(
            {
                "Ticker": ticker,
                "Rows": len(df),
                "Start": df["Date"].min().date().isoformat(),
                "End": df["Date"].max().date().isoformat(),
                "File": str(per_path.relative_to(OUT_DIR.parent)),
            }
        )
        print(f"  {ticker}: {len(df)} rows ({summary_rows[-1]['Start']} .. {summary_rows[-1]['End']})")

    if not frames:
        raise SystemExit("No data downloaded.")

    combined = pd.concat(frames, ignore_index=True)
    combined_path = OUT_DIR / "djia_components_daily_10y.csv"
    combined.to_csv(combined_path, index=False)

    # Wide Adj Close panel for convenience
    if "Adj Close" in combined.columns:
        adj = combined.pivot(index="Date", columns="Ticker", values="Adj Close").sort_index()
        adj_path = OUT_DIR / "djia_adj_close_wide_10y.csv"
        adj.to_csv(adj_path)
    else:
        adj_path = None

    summary = pd.DataFrame(summary_rows).sort_values("Ticker")
    summary_path = OUT_DIR / "download_summary.csv"
    summary.to_csv(summary_path, index=False)

    print()
    print(f"Combined long CSV : {combined_path} ({len(combined):,} rows)")
    if adj_path is not None:
        print(f"Wide Adj Close    : {adj_path} ({adj.shape[0]} dates x {adj.shape[1]} tickers)")
    print(f"Per-ticker CSVs   : {PER_TICKER_DIR}")
    print(f"Summary           : {summary_path}")


if __name__ == "__main__":
    main()
