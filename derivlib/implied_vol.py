"""Implied volatility: invert Black-Scholes for sigma given a market price.

Newton-Raphson (fast, uses vega) with a Brent bracketing fallback (robust when
vega is tiny, e.g. deep in/out of the money or very short expiries).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .black_scholes import bs_greeks, bs_price


def implied_vol(price, S, K, T, r, option="call", q=0.0, tol=1e-8, max_iter=50,
                lo=1e-4, hi=5.0) -> float:
    """Return the Black-Scholes implied volatility, or NaN if no solution exists."""
    disc_S, disc_K = S * np.exp(-q * T), K * np.exp(-r * T)
    intrinsic = max(disc_S - disc_K, 0) if option == "call" else max(disc_K - disc_S, 0)
    upper = disc_S if option == "call" else disc_K
    if not (intrinsic - 1e-12 <= price < upper):
        return float("nan")  # violates no-arbitrage bounds

    sigma = 0.2
    for _ in range(max_iter):
        diff = bs_price(S, K, T, r, sigma, option, q) - price
        if abs(diff) < tol:
            return float(sigma)
        vega = bs_greeks(S, K, T, r, sigma, option, q).vega
        if vega < 1e-8:
            break
        sigma -= diff / vega
        if not lo < sigma < hi:
            break
    f = lambda s: bs_price(S, K, T, r, s, option, q) - price  # noqa: E731
    try:
        return float(brentq(f, lo, hi, xtol=tol))
    except ValueError:
        return float("nan")


def smile_from_chain(chain: pd.DataFrame, S: float, T: float, r: float, q: float = 0.0,
                     min_price: float = 0.05) -> pd.DataFrame:
    """Add an `iv` column to an option chain.

    Expects columns: strike, price (use the bid/ask mid), option ('call'/'put').
    Uses out-of-the-money options only (they are the liquid ones), so the smile
    is puts below spot and calls above.
    """
    df = chain.copy()
    otm = ((df.option == "put") & (df.strike < S)) | ((df.option == "call") & (df.strike >= S))
    df = df[otm & (df.price > min_price)].copy()
    df["iv"] = [implied_vol(p, S, k, T, r, o, q) for p, k, o in zip(df.price, df.strike, df.option)]
    df["moneyness"] = df.strike / S
    return df.dropna(subset=["iv"]).sort_values("strike").reset_index(drop=True)
