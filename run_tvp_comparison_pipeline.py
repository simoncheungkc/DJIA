#!/usr/bin/env python3
"""Post-fit pipeline: scoreboard, networks, SBM study, reports."""
from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(mod: str) -> None:
    print(f"\n======== {mod} ========")
    runpy.run_path(str(ROOT / mod), run_name="__main__")


def main() -> None:
    pred = ROOT / "data" / "djia_6m_tvplrnvarsv_mean.csv"
    if not pred.exists():
        raise SystemExit("missing data/djia_6m_tvplrnvarsv_mean.csv — run tvp_lr_nvar_sv_6m_forecast.py first")
    run("compare_6m_forecasts.py")
    run("refresh_tvp_network_measures.py")
    run("rebuild_network_rankings.py")
    run("compare_sim_three_models.py")
    run("reports/build_tvp_empirical_report.py")
    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
