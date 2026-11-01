"""Black-Scholes pricing, analytic Greeks, and implied-volatility inversion.

Formulas (European, no dividends, constant r and sigma):

    d1 = [ln(S/K) + (r + sigma^2 / 2) * T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    Call = S * N(d1) - K * exp(-rT) * N(d2)
    Put  = K * exp(-rT) * N(-d2) - S * N(-d1)

Greeks are the standard analytic partial derivatives. Theta is reported
per year (divide by 365 for a per-calendar-day figure).
"""

import math

from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma):
    if T <= 0:
        raise ValueError("T must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return d1, d1 - sigma * math.sqrt(T)


def bs_price(S, K, T, r, sigma, option="call"):
    """European Black-Scholes price. option in {"call", "put"}."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    disc = math.exp(-r * T)
    if option == "call":
        return S * norm.cdf(d1) - K * disc * norm.cdf(d2)
    if option == "put":
        return K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)
    raise ValueError("option must be 'call' or 'put'")


def bs_greeks(S, K, T, r, sigma, option="call"):
    """Analytic Greeks. Returns a dict: delta, gamma, vega, theta, rho.

    Vega is per 1.00 (100%) change in volatility; divide by 100 for the
    usual "per 1 vol point" quote. Rho is per 1.00 (100%) change in rate.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    disc = math.exp(-r * T)
    pdf_d1 = norm.pdf(d1)
    gamma = pdf_d1 / (S * sigma * math.sqrt(T))
    vega = S * pdf_d1 * math.sqrt(T)
    if option == "call":
        delta = norm.cdf(d1)
        theta = (-(S * pdf_d1 * sigma) / (2 * math.sqrt(T))
                 - r * K * disc * norm.cdf(d2))
        rho = K * T * disc * norm.cdf(d2)
    elif option == "put":
        delta = norm.cdf(d1) - 1.0
        theta = (-(S * pdf_d1 * sigma) / (2 * math.sqrt(T))
                 + r * K * disc * norm.cdf(-d2))
        rho = -K * T * disc * norm.cdf(-d2)
    else:
        raise ValueError("option must be 'call' or 'put'")
    return {"delta": delta, "gamma": gamma, "vega": vega,
            "theta": theta, "rho": rho}


def _bs_vega(S, K, T, r, sigma):
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * math.sqrt(T)


def implied_volatility(market_price, S, K, T, r, option="call",
                       tol=1e-8, max_iter=100):
    """Implied volatility from a market price.

    Newton-Raphson on vega, with a bisection fallback. Raises ValueError
    if the price violates no-arbitrage bounds (e.g. below intrinsic value).
    """
    if T <= 0:
        raise ValueError("T must be positive")
    disc = math.exp(-r * T)
    if option == "call":
        intrinsic = max(S - K * disc, 0.0)
    elif option == "put":
        intrinsic = max(K * disc - S, 0.0)
    else:
        raise ValueError("option must be 'call' or 'put'")
    if market_price < intrinsic - 1e-12:
        raise ValueError(
            f"market price {market_price:.6f} below intrinsic value "
            f"{intrinsic:.6f}; no implied volatility exists")
    if abs(market_price - intrinsic) < 1e-12:
        return 0.0

    def f(sigma):
        return bs_price(S, K, T, r, sigma, option) - market_price

    # Newton-Raphson from a 20% starting guess.
    sigma = 0.20
    for _ in range(max_iter):
        price_err = f(sigma)
        if abs(price_err) < tol:
            return sigma
        vega = _bs_vega(S, K, T, r, sigma)
        if vega < 1e-12:
            break
        step = price_err / vega
        sigma -= step
        if sigma <= 0:
            sigma = 1e-4
            break
    # Bisection fallback on [1e-6, 5.0].
    lo, hi = 1e-6, 5.0
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        raise ValueError("could not bracket implied volatility in [0, 500%]")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if abs(f(mid)) < tol or (hi - lo) < 1e-10:
            return mid
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
