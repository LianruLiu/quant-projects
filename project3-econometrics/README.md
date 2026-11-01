# Volatility Forecasting with ARIMA and GARCH(1,1)

Econometric research project: can a GARCH(1,1) model beat simple
volatility benchmarks out of sample on the S&P 500?

## What it does

1. Downloads daily S&P 500 (`^GSPC`) closes from Yahoo Finance
   (yfinance, with a fallback to Yahoo's public chart API).
2. Computes log returns and tests stationarity (Augmented Dickey-Fuller).
3. Fits an ARIMA mean model (order chosen by AIC grid search) as a baseline.
4. Fits a GARCH(1,1) model (constant mean, normal errors, `arch` package).
5. Produces **true one-step-ahead** GARCH volatility forecasts over the
   last 252 trading days, refitting daily on an expanding window, and
   scores them against two baselines (constant variance, trailing-21-day
   realised volatility) using RMSE/MAE and Mincer-Zarnowitz regressions.

## Quick start

```bash
pip install -r requirements.txt
python3 run_all.py
```

Outputs land in `output/`:

| File | Contents |
|---|---|
| `results.json` | Every number reported in REPORT.md |
| `garch_forecasts.csv` | Daily 1-step-ahead GARCH forecasts, test window |
| `naive_forecasts.csv` | Baseline forecasts, test window |
| `price_returns.png` | Price level and log returns |
| `vol_comparison.png` | Forecasts vs realised volatility |
| `squared_acf.png` | ACF of squared returns (volatility clustering) |

Downloaded prices are cached in `data/sp500_daily.csv`.

## Headline result (sample 2016-01-04 to 2026-10-30)

GARCH(1,1) beats the constant-variance benchmark on every accuracy
metric and roughly ties the trailing-21-day naive forecast: variance RMSE
0.000109 (GARCH) vs 0.000127 (constant) vs 0.000110 (naive). Its
Mincer-Zarnowitz slope of 0.37 (R² = 0.014) is the only one of the three
that is close to informative; all three R² values are tiny, which is
normal for daily squared returns — they are an extremely noisy proxy for
latent volatility. See REPORT.md for the full analysis and limitations.

## Method notes worth knowing

- Returns are scaled by 100 before GARCH estimation because the `arch`
  optimiser is unstable on raw daily returns (scale ~1e-2); unscaled fits
  produced a spurious mean of about -0.10 on one window. Reported mu and
  omega are converted back to raw-return units; alpha, beta and
  persistence are scale-invariant.
- Realised volatility is proxied by squared / absolute daily returns.
  This is the standard, very noisy proxy; it penalises all models and
  compresses differences between them. Do not read the low MZ R² values
  as "the models are useless" — read them as "daily realised proxies
  are mostly noise".
- This is a research exercise, not a trading strategy. No position,
  return, or risk recommendation follows from it.

## Layout

```
run_all.py            # end-to-end pipeline
src/
  data.py             # download (yfinance + chart-API fallback)
  analysis.py         # returns, ADF, ARIMA, GARCH, forecasts, scoring
  plots.py            # charts
data/                 # downloaded prices (generated)
output/               # results, forecasts, charts (generated)
REPORT.md             # full research report
```
