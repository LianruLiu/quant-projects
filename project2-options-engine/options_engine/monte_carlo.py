"""Monte Carlo pricing under geometric Brownian motion.

    S_T = S * exp((r - sigma^2 / 2) * T + sigma * sqrt(T) * Z)

Antithetic variates (each Z is paired with -Z) halve the variance of the
payoff estimate. The reported standard error is the standard error of the
discounted mean payoff. A fixed seed makes runs reproducible.
"""

import math

import numpy as np


def mc_price(S, K, T, r, sigma, option="call", n_paths=200_000, seed=42):
    """Monte Carlo European price. Returns (price, standard_error).

    n_paths must be even (paths are generated in antithetic pairs).
    """
    if T <= 0:
        raise ValueError("T must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if n_paths < 2 or n_paths % 2 != 0:
        raise ValueError("n_paths must be even and >= 2")
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")
    rng = np.random.default_rng(seed)
    half = n_paths // 2
    z = rng.standard_normal(half)
    z = np.concatenate([z, -z])
    drift = (r - 0.5 * sigma * sigma) * T
    diffusion = sigma * math.sqrt(T) * z
    s_t = S * np.exp(drift + diffusion)
    if option == "call":
        payoffs = np.maximum(s_t - K, 0.0)
    else:
        payoffs = np.maximum(K - s_t, 0.0)
    discounted = math.exp(-r * T) * payoffs
    price = float(discounted.mean())
    se = float(discounted.std(ddof=1) / math.sqrt(n_paths))
    return price, se
