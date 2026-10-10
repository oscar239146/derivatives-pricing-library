"""Discrete delta-hedging simulation.

Sell a European call at its Black-Scholes price, then hold delta shares,
rebalancing n times over the option's life, financing at rate r. In continuous
time the hedged P&L is exactly zero; with discrete rebalancing it is a random
variable with mean ~0 and standard deviation shrinking like 1/sqrt(n).
The error is driven by gamma: between rebalances the option's convexity
is unhedged.
"""
from __future__ import annotations

import numpy as np

from .black_scholes import bs_greeks, bs_price
from .monte_carlo import simulate_gbm


def delta_hedge_pnl(S0, K, T, r, sigma, n_rebalance, n_paths=10_000, mu=None,
                    hedge_sigma=None, seed=None) -> np.ndarray:
    """Final P&L of a short call that is delta-hedged `n_rebalance` times.

    mu:          real-world drift of the stock (defaults to r). The hedged P&L
                 should not depend on it, which is the point of hedging.
    hedge_sigma: volatility used to price/hedge (defaults to sigma). Set it
                 different from the true sigma to see the cost of a wrong vol.
    """
    mu = r if mu is None else mu
    hs = sigma if hedge_sigma is None else hedge_sigma
    dt = T / n_rebalance
    paths = simulate_gbm(S0, T, mu, sigma, n_paths, n_rebalance, seed=seed)
    S = np.hstack([np.full((n_paths, 1), S0), paths])

    premium = bs_price(S0, K, T, r, hs, "call")
    delta = bs_greeks(S0, K, T, r, hs, "call").delta
    cash = premium - delta * S0
    shares = np.full(n_paths, delta)
    for i in range(1, n_rebalance):
        tau = T - i * dt
        cash *= np.exp(r * dt)
        new_delta = _vec_delta(S[:, i], K, tau, r, hs)
        cash -= (new_delta - shares) * S[:, i]
        shares = new_delta
    cash *= np.exp(r * dt)
    payoff = np.maximum(S[:, -1] - K, 0)
    return cash + shares * S[:, -1] - payoff


def _vec_delta(S, K, tau, r, sigma):
    from scipy.stats import norm
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)
