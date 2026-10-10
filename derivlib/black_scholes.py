"""Closed-form Black-Scholes prices and Greeks for European options.

Model: under the risk-neutral measure the stock follows geometric Brownian motion
    dS = (r - q) S dt + sigma S dW
with constant rate r, dividend yield q and volatility sigma.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q=0.0):
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def bs_price(S, K, T, r, sigma, option="call", q=0.0):
    """Black-Scholes price of a European call or put (vectorised over inputs)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_S = S * np.exp(-q * T)
    disc_K = K * np.exp(-r * T)
    if option == "call":
        return disc_S * norm.cdf(d1) - disc_K * norm.cdf(d2)
    if option == "put":
        return disc_K * norm.cdf(-d2) - disc_S * norm.cdf(-d1)
    raise ValueError("option must be 'call' or 'put'")


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    vega: float   # per 1.00 change in vol (divide by 100 for per vol-point)
    theta: float  # per year (divide by 365 for per calendar day)
    rho: float    # per 1.00 change in rate


def bs_greeks(S, K, T, r, sigma, option="call", q=0.0) -> Greeks:
    """Analytic Black-Scholes Greeks."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    pdf_d1 = norm.pdf(d1)
    eq, er = np.exp(-q * T), np.exp(-r * T)

    gamma = eq * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * eq * pdf_d1 * np.sqrt(T)
    common_theta = -S * eq * pdf_d1 * sigma / (2 * np.sqrt(T))

    if option == "call":
        delta = eq * norm.cdf(d1)
        theta = common_theta - r * K * er * norm.cdf(d2) + q * S * eq * norm.cdf(d1)
        rho = K * T * er * norm.cdf(d2)
    elif option == "put":
        delta = -eq * norm.cdf(-d1)
        theta = common_theta + r * K * er * norm.cdf(-d2) - q * S * eq * norm.cdf(-d1)
        rho = -K * T * er * norm.cdf(-d2)
    else:
        raise ValueError("option must be 'call' or 'put'")
    return Greeks(float(delta), float(gamma), float(vega), float(theta), float(rho))


def geometric_asian_call(S, K, T, r, sigma, n_steps, q=0.0):
    """Closed-form price of a discretely monitored geometric-average Asian call.

    The geometric average of lognormal prices is itself lognormal, so the price is
    Black-Scholes-like with an adjusted drift and volatility. Used as the control
    variate for the arithmetic Asian option in the Monte Carlo engine.
    Averaging dates are t_i = i*T/n for i = 1..n.
    """
    n = n_steps
    sigma_g2 = sigma**2 * (n + 1) * (2 * n + 1) / (6 * n**2)
    mu_g = (r - q - 0.5 * sigma**2) * (n + 1) / (2 * n) + 0.5 * sigma_g2
    sig_g = np.sqrt(sigma_g2)
    d1 = (np.log(S / K) + (mu_g + 0.5 * sigma_g2) * T) / (sig_g * np.sqrt(T))
    d2 = d1 - sig_g * np.sqrt(T)
    return np.exp(-r * T) * (S * np.exp(mu_g * T) * norm.cdf(d1) - K * norm.cdf(d2))
