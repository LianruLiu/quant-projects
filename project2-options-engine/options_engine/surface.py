"""Volatility-surface builder.

Pipeline (all inputs clearly labelled illustrative unless they come from a
market feed): for each (strike, maturity) pair, a market option price is
taken, Black-Scholes is inverted to an implied volatility, and the
resulting IV grid is plotted as a surface plus smile slices.

The demo in demo.py feeds this pipeline with prices generated from a
parametric smile, so the recovered surface can be checked against the
smile that generated it. That is a solver test, not a market observation.
"""

import numpy as np

from .black_scholes import bs_price, implied_volatility


def illustrative_smile_iv(strike, spot, maturity):
    """Parametric smile used ONLY for the demo (not market data).

    IV rises with |log-moneyness| (the classic smile) and slightly with
    maturity (term-structure tilt).
    """
    moneyness = np.log(strike / spot)
    return 0.20 + 0.10 * moneyness ** 2 + 0.02 * np.sqrt(maturity)


def build_surface(spot, strikes, maturities, rate, option="call"):
    """Return (iv_grid, price_grid) for the illustrative smile pipeline.

    For each (K, T): price the option with the smile IV, then invert that
    price back to an IV. A correct solver recovers the smile IV.
    """
    iv_grid = np.zeros((len(maturities), len(strikes)))
    price_grid = np.zeros((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            sigma_in = float(illustrative_smile_iv(K, spot, T))
            price = bs_price(spot, K, T, rate, sigma_in, option)
            sigma_out = implied_volatility(price, spot, K, T, rate, option)
            iv_grid[i, j] = sigma_out
            price_grid[i, j] = price
    return iv_grid, price_grid


def plot_surface(strikes, maturities, iv_grid, path):
    """Save a 3-D IV surface plot plus 2-D smile slices."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-D)

    K, T = np.meshgrid(strikes, maturities)
    fig = plt.figure(figsize=(12, 5))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(K, T, iv_grid, cmap="viridis")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity (years)")
    ax.set_zlabel("Implied volatility")
    ax.set_title("Implied volatility surface (illustrative inputs)")

    ax2 = fig.add_subplot(1, 2, 2)
    for i, t in enumerate(maturities):
        ax2.plot(strikes, iv_grid[i], marker="o", label=f"T={t:.2f}y")
    ax2.set_xlabel("Strike")
    ax2.set_ylabel("Implied volatility")
    ax2.set_title("Volatility smiles by maturity (illustrative inputs)")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
