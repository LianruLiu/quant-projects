# Quant Projects

Three small, fully reproducible quantitative finance projects. Every number in
these READMEs comes from actually running the code on real data — no invented
Sharpe ratios, no backfilled results.

## Projects

### 1. Systematic Trading Strategy Backtesting (`project1-backtesting/`)

Momentum (top-5, 12-1 month, monthly rebalance) and MA-crossover (SPY 50/200)
strategies on real daily data, 2015-01-02 to 2026-10-30, with 10 bps
transaction costs and an in-sample (2015–2019) / out-of-sample (2020–2026) split.

- MA crossover, out-of-sample: 9.67% annualized, Sharpe 0.577 — underperformed
  SPY buy-and-hold (14.94%, 0.749) but with lower volatility.
- Momentum top-5, out-of-sample: 35.75% annualized, Sharpe 1.304 — **read the
  caveat in that project's README before quoting this**: the universe is
  today's large-cap winners, so the number carries survivorship bias.

### 2. Options Pricing & Risk Management Engine (`project2-options-engine/`)

Black-Scholes, Cox-Ross-Rubinstein binomial trees (European + American),
Monte Carlo with antithetic variates, analytic Greeks, an implied-volatility
solver, and a volatility-surface builder. Three methods agree to within
~0.005 (binomial) / ~1 standard error (Monte Carlo) on the same inputs;
BS ATM call = 10.4506 for S=100, K=100, T=1, r=5%, σ=20%.

Note: the IV-surface demo uses clearly labelled illustrative inputs, because
no live option chain could be fetched from the build machine. `fetch_market.py`
runs the same solver on a real chain elsewhere.

### 3. Econometric Research: GARCH Volatility Forecasting (`project3-econometrics/`)

S&P 500 daily closes, 2016-01-04 to 2026-10-30 (2,722 log returns). ADF tests,
ARIMA(0,0,3) baseline, GARCH(1,1) with daily-refit genuine one-step-ahead
forecasts over a 252-day test year. Full report in `REPORT.md`.

- GARCH variance RMSE 0.000109 vs constant-variance 0.000127 vs naive 21-day
  window 0.000110: GARCH beats the constant benchmark but essentially ties the
  naive window — the honest headline, documented in the report.

## How to run

Each project has its own `requirements.txt` and README. General pattern:

```bash
python3 -m venv .venv
.venv/bin/pip install -r <project>/requirements.txt
.venv/bin/python <project>/backtest.py   # or demo.py / run_all.py
```

## Limitations, stated up front

- Backtests: survivorship bias in the momentum universe, flat cost assumption,
  no market impact. Past performance does not predict future returns.
- Options engine: constant volatility, no dividends, constant rate.
- GARCH study: single index, noisy squared-return volatility proxy, no trading
  conclusion. See each project's README / report for details.
