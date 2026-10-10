"""Download a real option chain with yfinance (optional dependency).

    pip install yfinance
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


def load_chain(ticker: str = "SPY", target_days: int = 30):
    """Return (chain, spot, T_years, expiry) for the listed expiry closest to target_days out.

    chain has columns strike, price (bid/ask mid), option ('call'/'put').
    About a month out avoids the noisy front weeklies while staying liquid.
    """
    import yfinance as yf

    tk = yf.Ticker(ticker)
    today = datetime.now(timezone.utc).date()
    expiry = min(tk.options, key=lambda e: abs((datetime.strptime(e, "%Y-%m-%d").date() - today).days - target_days))
    oc = tk.option_chain(expiry)
    spot = float(tk.history(period="1d")["Close"].iloc[-1])

    frames = []
    for df, kind in ((oc.calls, "call"), (oc.puts, "put")):
        f = df[(df.bid > 0) & (df.ask > 0)][["strike", "bid", "ask"]].copy()
        f["price"] = 0.5 * (f.bid + f.ask)
        f["option"] = kind
        frames.append(f[["strike", "price", "option"]])
    chain = pd.concat(frames, ignore_index=True)

    exp_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(hour=21, tzinfo=timezone.utc)
    T = max((exp_dt - datetime.now(timezone.utc)).total_seconds() / (365.25 * 24 * 3600), 1 / 365)
    return chain, spot, T, expiry
