"""Attempt a live SPY option-chain fetch via yfinance.

This is a convenience script, not part of the demo: on machines where
Yahoo Finance is reachable it prints the nearest-expiry ATM call/put so
their prices can be fed into implied_volatility(). On this build machine
it failed (yfinance timed out; raw Yahoo quote API: 401 Invalid Crumb),
which is why demo.py uses clearly-labelled illustrative inputs instead
of claiming market data.
"""

import sys


def main():
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance is not installed. Install it with:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    try:
        ticker = yf.Ticker("SPY")
        expirations = ticker.options
        if not expirations:
            print("No expirations returned (Yahoo may be unreachable).")
            sys.exit(1)
        spot = float(ticker.history(period="1d")["Close"].iloc[-1])
        chain = ticker.option_chain(expirations[0])
        print(f"SPY spot (last close): {spot:.2f}")
        print(f"Nearest expiry: {expirations[0]}")
        for side, frame in (("call", chain.calls), ("put", chain.puts)):
            atm = frame.iloc[(frame["strike"] - spot).abs().argsort()[:3]]
            print(f"\nNearest-the-money {side}s:")
            print(atm[["strike", "lastPrice", "bid", "ask",
                        "impliedVolatility"]].to_string(index=False))
    except Exception as exc:  # network/auth failures land here
        print(f"Live fetch failed: {exc}")
        print("Use the illustrative inputs in demo.py instead; "
              "do not present them as market data.")
        sys.exit(1)


if __name__ == "__main__":
    main()
