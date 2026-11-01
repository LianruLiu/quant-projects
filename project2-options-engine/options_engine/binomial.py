"""Cox-Ross-Rubinstein binomial tree.

European or American exercise. American early exercise is checked at every
node (this is what makes the tree useful for American puts, where
Black-Scholes does not apply).
"""

import math


def binomial_price(S, K, T, r, sigma, option="call", steps=500,
                   american=False):
    """CRR binomial price.

    u = exp(sigma * sqrt(dt)), d = 1/u,
    p = (exp(r * dt) - d) / (u - d).
    """
    if T <= 0:
        raise ValueError("T must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)

    def payoff(price):
        return max(price - K, 0.0) if option == "call" else max(K - price, 0.0)

    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")

    # Terminal values.
    values = [payoff(S * (u ** j) * (d ** (steps - j))) for j in range(steps + 1)]
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            values[j] = disc * (p * values[j + 1] + (1 - p) * values[j])
            if american:
                spot = S * (u ** j) * (d ** (i - j))
                values[j] = max(values[j], payoff(spot))
    return values[0]
