"""Monte Carlo pricing under geometric Brownian motion, with variance reduction.

Techniques:
  * antithetic variates  - pair each normal draw Z with -Z
  * control variates     - European: discounted terminal stock (known mean S0*e^{-qT})
                           Asian:    geometric-average Asian (closed form)
  * pathwise Greeks      - differentiate the payoff along each path
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .black_scholes import geometric_asian_call


@dataclass(frozen=True)
class MCResult:
    price: float
    std_error: float
    n_paths: int

    @property
    def ci95(self) -> tuple[float, float]:
        return self.price - 1.96 * self.std_error, self.price + 1.96 * self.std_error


def _normals(rng, n_paths, n_steps, antithetic):
    if antithetic:
        half = rng.standard_normal((n_paths // 2, n_steps))
        return np.vstack([half, -half])
    return rng.standard_normal((n_paths, n_steps))


def simulate_gbm(S0, T, r, sigma, n_paths, n_steps, q=0.0, antithetic=False, seed=None):
    """Exact GBM paths at t_i = i*T/n_steps, shape (n_paths, n_steps) (S0 excluded)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = _normals(rng, n_paths, n_steps, antithetic)
    log_inc = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    return S0 * np.exp(np.cumsum(log_inc, axis=1))


def _summarise(samples: np.ndarray, antithetic: bool) -> MCResult:
    if antithetic:  # average each (Z, -Z) pair so samples are independent
        half = len(samples) // 2
        samples = 0.5 * (samples[:half] + samples[half:])
    return MCResult(float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(len(samples))),
                    len(samples) * (2 if antithetic else 1))


def _control_adjust(y: np.ndarray, x: np.ndarray, x_mean: float) -> np.ndarray:
    """Return y - b*(x - E[x]) with the variance-minimising b = Cov(x,y)/Var(x)."""
    b = np.cov(x, y)[0, 1] / np.var(x, ddof=1)
    return y - b * (x - x_mean)


def mc_european(S0, K, T, r, sigma, option="call", n_paths=100_000, q=0.0,
                antithetic=False, control_variate=False, seed=None) -> MCResult:
    """Price a European option by simulating the terminal stock price."""
    ST = simulate_gbm(S0, T, r, sigma, n_paths, 1, q, antithetic, seed)[:, -1]
    payoff = np.maximum(ST - K, 0) if option == "call" else np.maximum(K - ST, 0)
    disc = np.exp(-r * T)
    y = disc * payoff
    if control_variate:
        y = _control_adjust(y, disc * ST, S0 * np.exp(-q * T))
    return _summarise(y, antithetic)


def mc_asian_call(S0, K, T, r, sigma, n_steps=50, n_paths=100_000, q=0.0,
                  antithetic=False, control_variate=False, seed=None) -> MCResult:
    """Arithmetic-average Asian call (discrete monitoring), optional geometric CV."""
    paths = simulate_gbm(S0, T, r, sigma, n_paths, n_steps, q, antithetic, seed)
    disc = np.exp(-r * T)
    y = disc * np.maximum(paths.mean(axis=1) - K, 0)
    if control_variate:
        geo = np.exp(np.log(paths).mean(axis=1))
        x = disc * np.maximum(geo - K, 0)
        y = _control_adjust(y, x, geometric_asian_call(S0, K, T, r, sigma, n_steps, q))
    return _summarise(y, antithetic)


def mc_pathwise_greeks(S0, K, T, r, sigma, option="call", n_paths=200_000, q=0.0, seed=None):
    """Pathwise delta and vega for a European option.

    For S_T = S0 exp((r-q-sigma^2/2)T + sigma sqrt(T) Z):
        dS_T/dS0    = S_T / S0
        dS_T/dsigma = S_T (sqrt(T) Z - sigma T)
    and d(payoff)/dS_T = 1{S_T > K} (call) or -1{S_T < K} (put).
    Valid because the payoff is Lipschitz; it fails for digitals (use likelihood ratio).
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    ST = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    dpay = (ST > K).astype(float) if option == "call" else -(ST < K).astype(float)
    disc = np.exp(-r * T)
    delta = disc * dpay * ST / S0
    vega = disc * dpay * ST * (np.sqrt(T) * Z - sigma * T)
    se = lambda a: float(a.std(ddof=1) / np.sqrt(n_paths))  # noqa: E731
    return {"delta": (float(delta.mean()), se(delta)), "vega": (float(vega.mean()), se(vega))}
