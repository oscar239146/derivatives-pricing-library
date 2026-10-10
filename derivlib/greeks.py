"""Model-agnostic Greeks by bump-and-revalue (finite differences on any pricer).

Works with any pricing function, which is how risk systems compute sensitivities
for products without closed forms. For Monte Carlo pricers, pass a fixed seed so
the bumped and base prices use common random numbers; otherwise the noise in
each price swamps the small difference you are trying to measure.
"""
from __future__ import annotations

from typing import Callable


def bump_greeks(pricer: Callable[..., float], params: dict, rel_bump_S=1e-3,
                bump_sigma=1e-3, bump_r=1e-4, bump_T=1 / 365) -> dict:
    """Central-difference delta, gamma, vega, rho and forward-difference theta.

    `pricer(**params)` must return a price; params must include S0/S, sigma, r, T.
    """
    s_key = "S0" if "S0" in params else "S"
    p = dict(params)

    def price(**changes):
        return float(pricer(**{**p, **changes}))

    S = p[s_key]
    h = S * rel_bump_S
    base = price()
    up, down = price(**{s_key: S + h}), price(**{s_key: S - h})
    vega = (price(sigma=p["sigma"] + bump_sigma) - price(sigma=p["sigma"] - bump_sigma)) / (2 * bump_sigma)
    rho = (price(r=p["r"] + bump_r) - price(r=p["r"] - bump_r)) / (2 * bump_r)
    theta = (price(T=p["T"] - bump_T) - base) / bump_T  # value change as time passes
    return {
        "delta": (up - down) / (2 * h),
        "gamma": (up - 2 * base + down) / h**2,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }
