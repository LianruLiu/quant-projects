"""Download daily S&P 500 (^GSPC) closing prices.

Primary source: yfinance. Fallback: Yahoo Finance public chart API
(https://query1.finance.yahoo.com/v8/finance/chart/^GSPC), which needs no
API key. The fallback exists because yfinance occasionally fails in
restricted environments; both sources serve the same Yahoo price series.
"""

import datetime as _dt
import time

import pandas as pd
import requests

TICKER = "^GSPC"
DEFAULT_START = "2016-01-01"


def download_yfinance(start: str, end: str | None = None) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        TICKER, start=start, end=end, auto_adjust=False, progress=False
    )
    if df.empty:
        raise ValueError("yfinance returned no rows")
    # yfinance >= 0.2 returns MultiIndex columns for single tickers
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def download_chart_api(
    start: str, end: str | None = None, retries: int = 3
) -> pd.DataFrame:
    """Fetch daily bars from Yahoo's public chart endpoint."""
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(
        pd.Timestamp(end).timestamp()
        if end
        else _dt.datetime.now(_dt.timezone.utc).timestamp()
    )
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{TICKER}"
        f"?period1={start_ts}&period2={end_ts}&interval=1d"
    )
    last_err = None
    for _ in range(retries):
        try:
            resp = requests.get(
                url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30
            )
            resp.raise_for_status()
            payload = resp.json()["chart"]["result"][0]
            break
        except Exception as exc:  # noqa: BLE001 - retry then surface
            last_err = exc
            time.sleep(2)
    else:
        raise RuntimeError(f"chart API failed after {retries} tries: {last_err}")

    ts = payload["timestamp"]
    quote = payload["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "Open": quote["open"],
            "High": quote["high"],
            "Low": quote["low"],
            "Close": quote["close"],
            "Volume": quote["volume"],
        },
        index=pd.to_datetime(ts, unit="s", utc=True).tz_convert(None),
    )
    return df.dropna()


def load_prices(start: str = DEFAULT_START, end: str | None = None) -> pd.DataFrame:
    """Return daily OHLCV, preferring yfinance, falling back to chart API."""
    try:
        df = download_yfinance(start, end)
        source = "yfinance (^GSPC)"
    except Exception as exc:  # noqa: BLE001 - fallback is the point
        print(f"yfinance failed ({exc}); falling back to Yahoo chart API")
        df = download_chart_api(start, end)
        source = "Yahoo chart API (^GSPC)"
    df.index.name = "Date"
    df.attrs["source"] = source
    return df


if __name__ == "__main__":
    prices = load_prices()
    print(prices.attrs["source"])
    print(prices.head())
    print(prices.tail())
    print(f"rows={len(prices)}, {prices.index[0].date()} -> {prices.index[-1].date()}")
