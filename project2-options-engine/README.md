# Options Pricing & Risk Management Engine

A small, dependency-light Python engine for European option pricing and
risk, built as a portfolio project. It implements three pricing methods,
analytic Greeks, an implied-volatility solver, and a volatility-surface
builder — and it validates them against each other.

## What it does

| Module | Contents |
|---|---|
| `options_engine/black_scholes.py` | Black-Scholes call/put pricing, analytic Greeks (Delta, Gamma, Vega, Theta, Rho), implied-volatility solver (Newton-Raphson with bisection fallback) |
| `options_engine/binomial.py` | Cox-Ross-Rubinstein tree, European and American exercise |
| `options_engine/monte_carlo.py` | Geometric-Brownian-motion Monte Carlo with antithetic variates and a reported standard error |
| `options_engine/surface.py` | Implied-volatility surface builder + 3-D surface / smile-slice plots |

## How to run

```bash
pip install -r requirements.txt
python3 demo.py
```

`demo.py` prints a pricing-method comparison, American-vs-European put
values, Greeks, an implied-volatility round-trip, and the volatility
surface, and saves `iv_surface.png`. It runs with the base scientific
stack only (numpy/scipy/pandas/matplotlib); yfinance is needed only for
`fetch_market.py`.

To try a live option chain on a machine where Yahoo Finance is reachable:

```bash
python3 fetch_market.py
```

## Data honesty note (read this before citing any demo number)

**No live market data is used in `demo.py`.** A live SPY chain was
attempted during development and could not be fetched from the build
machine: yfinance timed out, the raw Yahoo quote API returned
`401 Invalid Crumb`, and Stooq returned a bot-check page. The IV and
surface demos therefore use clearly-labelled **illustrative** inputs.
Every price, Greek, and IV in `VALIDATION.md` was produced by running
this code on those inputs — none of it is presented as a market quote.
`fetch_market.py` exists so the same IV solver can be run on a real
chain elsewhere.

## Formulas used

Black-Scholes (European, no dividends, constant `r` and `σ`):

```
d1 = [ln(S/K) + (r + σ²/2)·T] / (σ·√T),   d2 = d1 − σ·√T
Call = S·N(d1) − K·e^(−rT)·N(d2)
Put  = K·e^(−rT)·N(−d2) − S·N(−d1)
```

Greeks are the standard analytic partial derivatives. Theta is reported
per year (divide by 365 for per-calendar-day); Vega is per 1.00 change
in volatility (divide by 100 for per-vol-point).

Binomial tree (CRR): `u = e^(σ√dt)`, `d = 1/u`,
`p = (e^(r·dt) − d) / (u − d)`, discounted backward induction, with an
early-exercise check at every node for American options.

Monte Carlo: `S_T = S·exp((r − σ²/2)·T + σ·√T·Z)` with antithetic
variates (each `Z` paired with `−Z`).

Implied volatility: Newton-Raphson on Vega from a 20% starting guess,
falling back to bisection on [0, 500%]; prices below intrinsic value
are rejected, since no IV exists for them.

## Limitations (honest)

- **Constant volatility.** Black-Scholes assumes one `σ` for all strikes
  and maturities. Real markets show a smile/skew — that is exactly why
  the surface builder exists, and why BS prices are a model, not a quote.
- **European exercise** for Black-Scholes and Monte Carlo. Only the
  binomial tree prices American exercise.
- **No dividends.** Discrete or continuous dividend yields are not
  modelled; a dividend-paying underlying needs a yield adjustment.
- **Constant interest rate**, no stochastic rates, no jumps, no
  transaction costs or bid/ask spread.
- **Monte Carlo error is real.** Reported standard errors are ~0.01–0.04
  at 200,000 paths; for tight pricing, raise `n_paths` or use the
  analytic/tree methods.
- **Binomial convergence is slow and oscillatory** in the number of
  steps; 500 steps is a reasonable default, not a guarantee.
- **IV inversion assumes the BS model.** An IV backed out of a market
  price inherits every limitation above; it is a quoting convention,
  not a physical parameter.
