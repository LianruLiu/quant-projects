"""Charts for the volatility-forecasting project (saved to output/)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def chart_price_returns(prices: pd.DataFrame, returns: pd.Series, path: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(prices.index, prices["Close"], lw=0.8)
    axes[0].set_title("S&P 500 daily close")
    axes[0].set_ylabel("Index level")
    axes[1].plot(returns.index, returns.values, lw=0.5)
    axes[1].set_title("Daily log returns")
    axes[1].set_ylabel("Log return")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_vol_comparison(
    garch_df: pd.DataFrame, naive_df: pd.DataFrame, path: str
) -> None:
    realised_vol_21 = garch_df["return"].rolling(21).std()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        realised_vol_21.index,
        realised_vol_21.values,
        lw=0.8,
        label="Realised vol (trailing 21-day std)",
    )
    ax.plot(
        garch_df.index,
        garch_df["garch_vol_forecast"].values,
        lw=0.8,
        label="GARCH(1,1) 1-step-ahead vol forecast",
    )
    ax.plot(
        naive_df.index,
        naive_df["naive_vol_forecast"].values,
        lw=0.8,
        alpha=0.7,
        label="Naive trailing-21-day vol forecast",
    )
    ax.set_title("Out-of-sample volatility: forecasts vs realised (test window)")
    ax.set_ylabel("Daily volatility")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_squared_acf(returns: pd.Series, path: str) -> None:
    from statsmodels.graphics.tsaplots import plot_acf

    fig, ax = plt.subplots(figsize=(12, 4))
    plot_acf(returns.values**2, lags=30, ax=ax)
    ax.set_title("ACF of squared log returns (volatility clustering)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
