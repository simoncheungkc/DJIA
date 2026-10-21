#!/usr/bin/env python3
"""Hierarchical open-loop return forecast shared by LS-LR-NVAR and TVP-LR-NVAR-SV.

Point paths use the frozen origin network VAR (Phi_1, Phi_2), a Minnesota
intercept on daily log-returns, a companion-radius cap, and no future shocks.
The common intercept is a 5% annual equity premium (not the 10-year sample
mean). Name effects keep one-third of historical mean deviations.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
HORIZON = 126
LAG_P = 2
RET_CLIP = 0.03
COMPANION_CAP = 0.95

ANNUAL_PREMIUM = 0.05  # Minnesota prior mean for expected equity return
NAME_JS = 1.0 / 3.0  # fraction of historical name deviation kept
FORECAST_KAPPA = 1.0
FORECAST_CAP = 0.85


def companion_radius(B1: np.ndarray, B2: np.ndarray) -> float:
    n = B1.shape[0]
    C = np.zeros((2 * n, 2 * n))
    C[:n, :n] = B1
    C[:n, n:] = B2
    C[n:, :n] = np.eye(n)
    try:
        return float(np.max(np.abs(np.linalg.eigvals(C))))
    except np.linalg.LinAlgError:
        return 10.0


def scale_companion(B1: np.ndarray, B2: np.ndarray, cap: float) -> tuple[np.ndarray, np.ndarray]:
    B1 = np.asarray(B1, dtype=float).copy()
    B2 = np.asarray(B2, dtype=float).copy()
    rad = companion_radius(B1, B2)
    if rad > cap:
        s = cap / (rad + 1e-12)
        B1 = s * B1
        B2 = s * B2
    return B1, B2


def iterate_returns(
    mu: np.ndarray,
    B1: np.ndarray,
    B2: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    h: int,
    clip: float = RET_CLIP,
    kappa: float = 1.0,
) -> np.ndarray:
    """y_t = mu_t + B1 (y_{t-1}-mu_t) + B2 (y_{t-2}-mu_t), with mu_t = kappa^{t} mu."""
    out = np.zeros((h, mu.shape[0]))
    lag1, lag2 = y1.copy(), y2.copy()
    scale = 1.0
    k = float(kappa)
    for t in range(h):
        mu_t = scale * mu
        nxt = mu_t + B1 @ (lag1 - mu_t) + B2 @ (lag2 - mu_t)
        nxt = np.clip(nxt, -clip, clip)
        out[t] = nxt
        lag2, lag1 = lag1, nxt
        scale *= k
    return out


def price_path(last_price: np.ndarray, ret: np.ndarray) -> np.ndarray:
    return last_price * np.exp(np.cumsum(ret, axis=0))


def path_mape(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - actual) / np.maximum(np.abs(actual), 1e-8)) * 100)


def term_mae_ret(pred: np.ndarray, actual: np.ndarray, last: np.ndarray) -> float:
    return float(np.mean(np.abs(pred[-1] / last - actual[-1] / last)))


def minnesota_intercept(train_ret: np.ndarray) -> np.ndarray:
    """5% annual common premium plus one-third of historical name deviations.

    The common term is a Minnesota prior mean for daily equity returns (not the
    10-year realised mean, which overstates a six-month expected return). Name
    deviations are James-Stein / Minnesota-shrunk so idiosyncratic drift cannot
    compound at the sample MLE.
    """
    name_mean = np.asarray(train_ret, dtype=float).mean(axis=0)
    grand = float(name_mean.mean())
    return (ANNUAL_PREMIUM / 252.0) + NAME_JS * (name_mean - grand)


def forecast_from_phi(
    train_ret: np.ndarray,
    last_price: np.ndarray,
    B1: np.ndarray,
    B2: np.ndarray,
    h: int,
    cap: float = FORECAST_CAP,
    kappa: float = FORECAST_KAPPA,
) -> np.ndarray:
    mu = minnesota_intercept(train_ret)
    B1, B2 = scale_companion(B1, B2, cap)
    rets = iterate_returns(mu, B1, B2, train_ret[-1], train_ret[-2], h, kappa=kappa)
    return price_path(last_price, rets)


def last_operators(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(npz_path, allow_pickle=True)
    return np.asarray(z["Phi1"][-1], dtype=float), np.asarray(z["Phi2"][-1], dtype=float)


def main() -> None:
    """Rebuild the three network-model 6m paths from saved origin operators."""
    prices = pd.read_csv(DATA_DIR / "djia_adj_close_wide_10y.csv", parse_dates=["Date"], index_col="Date")
    prices = prices.sort_index().ffill().dropna(how="all")
    tickers = prices.columns.tolist()
    log_ret = np.diff(np.log(prices.values), axis=0)
    train_ret = log_ret[:-HORIZON]
    train_prices = prices.iloc[:-HORIZON].values
    test_prices = prices.iloc[-HORIZON:]
    last_train_price = train_prices[-1]
    fc_dates = test_prices.index
    actual = test_prices.values

    specs = [
        ("LS-LR-NVAR spec", DATA_DIR / "lslrnvar_A_mean_train_spec.npz", DATA_DIR / "djia_6m_lslrnvar_mean.csv"),
        ("LS-LR-NVAR v2", DATA_DIR / "lslrnvar_A_mean_train_v2.npz", DATA_DIR / "djia_6m_lslrnvar_revised_v2_mean.csv"),
        ("TVP-LR-NVAR-SV", DATA_DIR / "tvplrnvarsv_A_mean_train.npz", DATA_DIR / "djia_6m_tvplrnvarsv_mean.csv"),
    ]
    meta = {}
    for name, npz, out_csv in specs:
        if not npz.exists():
            print(f"SKIP missing {npz.name}")
            continue
        B1, B2 = last_operators(npz)
        pred = forecast_from_phi(train_ret, last_train_price, B1, B2, HORIZON)
        pd.DataFrame(pred, index=fc_dates, columns=tickers).rename_axis("Date").to_csv(out_csv)
        mape = path_mape(pred, actual)
        mae = float(np.mean(np.abs(pred - actual)))
        pred_term = pred[-1] / last_train_price - 1.0
        act_term = actual[-1] / last_train_price - 1.0
        dir_acc = float(np.mean(np.sign(pred_term) == np.sign(act_term)))
        sp = float(pd.Series(pred_term).rank().corr(pd.Series(act_term).rank()))
        rmae = float(np.mean(np.abs(pred_term - act_term)))
        print(
            f"{name:22s}  Minnesota {100*ANNUAL_PREMIUM:.0f}% + {NAME_JS:.2f} name  "
            f"OOS MAPE {mape:.2f}%  MAE {mae:.2f}  "
            f"6m {100*rmae:.2f}pp  dir {100*dir_acc:.0f}%  sp {sp:.3f}"
        )
        meta[name] = {
            "annual_premium": ANNUAL_PREMIUM,
            "name_js": NAME_JS,
            "cap": FORECAST_CAP,
            "kappa": FORECAST_KAPPA,
            "oos_mape": mape,
            "oos_mae": mae,
            "oos_term_mae": rmae,
            "oos_dir": dir_acc,
            "oos_spearman": sp,
            "file": str(out_csv.name),
        }
    import json as _json

    (DATA_DIR / "hierarchical_forecast_meta.json").write_text(_json.dumps(meta, indent=2), encoding="utf-8")
    print("Wrote hierarchical_forecast_meta.json")


if __name__ == "__main__":
    main()
