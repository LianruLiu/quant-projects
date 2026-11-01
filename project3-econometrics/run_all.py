"""End-to-end run: download -> analyse -> forecasts -> results.json + charts.

Usage:  python3 run_all.py
Outputs: data/sp500_daily.csv, output/results.json, output/*.png
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from analysis import (  # noqa: E402
    TEST_DAYS,
    adf_report,
    diagnostics,
    fit_arima_grid,
    fit_garch,
    garch_summary,
    log_returns,
    mincer_zarnowitz,
    naive_forecasts,
    rolling_garch_forecasts,
    score_forecasts,
)
from data import load_prices  # noqa: E402
from plots import (  # noqa: E402
    chart_price_returns,
    chart_squared_acf,
    chart_vol_comparison,
)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output"
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)


def main() -> dict:
    prices = load_prices()
    prices.to_csv(DATA_DIR / "sp500_daily.csv")
    returns = log_returns(prices)

    train = returns.iloc[:-TEST_DAYS]
    test = returns.iloc[-TEST_DAYS:]

    results: dict = {
        "source": prices.attrs.get("source", "unknown"),
        "ticker": "^GSPC",
        "sample_start": str(prices.index[0].date()),
        "sample_end": str(prices.index[-1].date()),
        "n_prices": int(len(prices)),
        "n_returns": int(len(returns)),
        "train_start": str(train.index[0].date()),
        "train_end": str(train.index[-1].date()),
        "test_start": str(test.index[0].date()),
        "test_end": str(test.index[-1].date()),
        "test_days": int(len(test)),
        "close_first": float(prices["Close"].iloc[0]),
        "close_last": float(prices["Close"].iloc[-1]),
        "returns_mean_daily": float(returns.mean()),
        "returns_std_daily": float(returns.std()),
        "returns_mean_annualised_pct": float(returns.mean() * 252 * 100),
        "returns_vol_annualised_pct": float(returns.std() * np.sqrt(252) * 100),
        "returns_min": float(returns.min()),
        "returns_max": float(returns.max()),
        "returns_min_date": str(returns.idxmin().date()),
        "returns_max_date": str(returns.idxmax().date()),
    }

    # Stationarity
    results["adf_close"] = adf_report(prices["Close"], "close level")
    results["adf_returns"] = adf_report(returns, "log returns")

    # ARIMA baseline on full-sample returns (mean model, for reference)
    results["arima_full"] = fit_arima_grid(returns)

    # GARCH(1,1) on the training window (in-sample fit)
    garch_fit = fit_garch(train)
    results["garch_train"] = garch_summary(garch_fit)
    results["diagnostics_train"] = diagnostics(train, garch_fit)

    # Out-of-sample forecasts
    garch_df = rolling_garch_forecasts(returns)
    naive_df = naive_forecasts(returns)
    garch_df.to_csv(OUT_DIR / "garch_forecasts.csv")
    naive_df.to_csv(OUT_DIR / "naive_forecasts.csv")

    results["scores"] = {
        "garch": score_forecasts(
            garch_df, "garch_var_forecast", "garch_vol_forecast"
        ),
        "constant": score_forecasts(
            naive_df, "const_var_forecast", "const_vol_forecast"
        ),
        "naive_21d": score_forecasts(
            naive_df, "naive_var_forecast", "naive_vol_forecast"
        ),
    }
    results["mincer_zarnowitz"] = {
        "garch": mincer_zarnowitz(garch_df, "garch_var_forecast"),
        "constant": mincer_zarnowitz(naive_df, "const_var_forecast"),
        "naive_21d": mincer_zarnowitz(naive_df, "naive_var_forecast"),
    }
    # Headline realised-vol summary for the test window
    results["test_realised_vol_daily_mean"] = float(test.std())
    results["test_realised_vol_annualised_pct"] = float(test.std() * np.sqrt(252) * 100)
    results["test_mean_abs_return"] = float(test.abs().mean())

    with open(OUT_DIR / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    chart_price_returns(prices, returns, str(OUT_DIR / "price_returns.png"))
    chart_vol_comparison(garch_df, naive_df, str(OUT_DIR / "vol_comparison.png"))
    chart_squared_acf(returns, str(OUT_DIR / "squared_acf.png"))

    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
