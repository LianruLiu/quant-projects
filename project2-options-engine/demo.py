"""End-to-end demo: pricing-method comparison, Greeks, IV inversion,
and the volatility-surface pipeline.

All numbers printed here are produced by running this file. The IV and
surface demos use ILLUSTRATIVE inputs (see README): live Yahoo Finance
option data could not be fetched from this machine (yfinance timed out,
the raw Yahoo quote API returned 401 Invalid Crumb, Stooq returned a
bot-check page), so no market prices are claimed anywhere in this demo.
Run fetch_market.py on a machine with Yahoo access to pull a real chain.
"""

import numpy as np

from options_engine import bs_price, bs_greeks, implied_volatility, binomial_price, mc_price
from options_engine.surface import build_surface, plot_surface

# Classic textbook case (Hull-style): ATM, 1 year.
S, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.20


def main():
    print("=== 1. Method comparison (same inputs, European) ===")
    print(f"Inputs: S={S}, K={K}, T={T}, r={R}, sigma={SIGMA}")
    cases = [
        ("call ATM ", S, K, T),
        ("put  ATM ", S, K, T),
        ("call ITM ", S, 90.0, T),
        ("put  ITM ", S, 110.0, T),
        ("call OTM ", S, 110.0, T),
        ("put  OTM ", S, 90.0, T),
        ("call ATM, T=0.25", S, K, 0.25),
    ]
    print(f"{'case':<18}{'BS':>10}{'Binom(500)':>12}{'MC':>10}{'MC s.e.':>10}")
    for name, s, k, t in cases:
        for opt in (["call"] if name.startswith("call") else ["put"]):
            bs = bs_price(s, k, t, R, SIGMA, opt)
            bn = binomial_price(s, k, t, R, SIGMA, opt, steps=500)
            mc, se = mc_price(s, k, t, R, SIGMA, opt, n_paths=200_000, seed=42)
            print(f"{name:<18}{bs:>10.4f}{bn:>12.4f}{mc:>10.4f}{se:>10.4f}")

    print("\n=== 2. American vs European put (early exercise premium) ===")
    eu = binomial_price(S, 110.0, T, R, SIGMA, "put", steps=500, american=False)
    am = binomial_price(S, 110.0, T, R, SIGMA, "put", steps=500, american=True)
    print(f"European ITM put (K=110): {eu:.4f}")
    print(f"American  ITM put (K=110): {am:.4f}")
    print(f"Early-exercise premium:   {am - eu:.4f}")

    print("\n=== 3. Greeks (ATM call and put, per-year theta) ===")
    for opt in ("call", "put"):
        g = bs_greeks(S, K, T, R, SIGMA, opt)
        print(f"{opt}: " + ", ".join(f"{k}={v:.4f}" for k, v in g.items()))
        print(f"   theta per calendar day: {g['theta'] / 365:.4f}, "
              f"vega per 1 vol point: {g['vega'] / 100:.4f}")

    print("\n=== 4. Implied volatility round-trip (ILLUSTRATIVE inputs) ===")
    print("Market prices below are ILLUSTRATIVE, not real quotes.")
    for opt, true_sigma in (("call", 0.25), ("put", 0.18), ("call", 0.35)):
        mkt = bs_price(S, K, T, R, true_sigma, opt)
        iv = implied_volatility(mkt, S, K, T, R, opt)
        print(f"{opt} true sigma={true_sigma:.2f} -> price={mkt:.4f} "
              f"-> recovered IV={iv:.6f} (error {abs(iv - true_sigma):.2e})")

    print("\n=== 5. Volatility surface (ILLUSTRATIVE smile pipeline) ===")
    strikes = [80, 90, 100, 110, 120]
    maturities = [0.25, 0.5, 1.0, 2.0]
    iv_grid, _ = build_surface(S, strikes, maturities, R, "call")
    print("Recovered IV grid (rows = maturity, cols = strike):")
    print("        " + "".join(f"{k:>8}" for k in strikes))
    for t, row in zip(maturities, iv_grid):
        print(f"T={t:<4}" + "".join(f"{v:>8.4f}" for v in row))
    plot_surface(strikes, maturities, iv_grid, "iv_surface.png")
    print("Saved plot to iv_surface.png")


if __name__ == "__main__":
    main()
