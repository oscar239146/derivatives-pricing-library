"""Finite-difference PDE solver for the Black-Scholes equation.

Solves  dV/dtau = 1/2 sigma^2 S^2 V_SS + (r - q) S V_S - r V,   tau = T - t
on a uniform grid S_i = i*dS, i = 0..M, with a theta-scheme in time:
    theta = 0.5  -> Crank-Nicolson (second order in dt)
    theta = 1.0  -> fully implicit (first order, but damps oscillations)

Rannacher smoothing: the first few steps use fully implicit half-steps, which
removes the spurious oscillations Crank-Nicolson produces near the strike kink
(they otherwise pollute gamma).

American puts use the Brennan-Schwartz algorithm: a direct tridiagonal solve in
which the early-exercise constraint V >= K - S is applied during back-substitution.
It is exact for puts (single exercise boundary) and O(M) per step, so no iterative
PSOR is needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FDResult:
    price: float
    delta: float
    gamma: float
    grid_S: np.ndarray
    grid_V: np.ndarray


def _operator_coeffs(M, r, q, sigma):
    i = np.arange(1, M)
    alpha = 0.5 * (sigma**2 * i**2 - (r - q) * i)   # coefficient on V_{i-1}
    beta = -(sigma**2 * i**2 + r)                   # coefficient on V_i
    gamma = 0.5 * (sigma**2 * i**2 + (r - q) * i)   # coefficient on V_{i+1}
    return alpha, beta, gamma


def _boundaries(option, american, K, S_max, tau, r, q):
    if option == "call":
        return 0.0, S_max * np.exp(-q * tau) - K * np.exp(-r * tau)
    lower = K if american else K * np.exp(-r * tau)
    return lower, 0.0


def _solve_step(l, d, u, rhs, payoff_inner, american):
    """Solve the tridiagonal system l*V_{i-1} + d*V_i + u*V_{i+1} = rhs.

    Eliminates the upper diagonal from the top, then substitutes upward from
    the bottom; for American puts, takes max(payoff, continuation) as it goes.
    """
    n = len(d)
    d_p, r_p = d.copy(), rhs.copy()
    for i in range(n - 2, -1, -1):
        f = u[i] / d_p[i + 1]
        d_p[i] -= f * l[i + 1]
        r_p[i] -= f * r_p[i + 1]
    V = np.empty(n)
    V[0] = r_p[0] / d_p[0]
    if american:
        V[0] = max(V[0], payoff_inner[0])
    for i in range(1, n):
        V[i] = (r_p[i] - l[i] * V[i - 1]) / d_p[i]
        if american:
            V[i] = max(V[i], payoff_inner[i])
    return V


def fd_price(S0, K, T, r, sigma, option="put", american=False, q=0.0,
             M=800, N=400, S_max_mult=4.0, theta=0.5, rannacher_steps=2) -> FDResult:
    """Price a European call/put or American put on a finite-difference grid."""
    if american and option == "call" and q == 0.0:
        american = False  # never optimal to exercise a call early without dividends
    if american and option == "call":
        raise NotImplementedError("American calls with dividends: use the binomial tree")

    S_max = S_max_mult * max(S0, K)
    S = np.linspace(0.0, S_max, M + 1)
    payoff = np.maximum(S - K, 0) if option == "call" else np.maximum(K - S, 0)
    V = payoff.copy()
    alpha, beta, gamma = _operator_coeffs(M, r, q, sigma)

    # time steps (in tau): optional Rannacher implicit half-steps, then theta-scheme
    dt = T / N
    steps = [(dt / 2, 1.0)] * (2 * rannacher_steps) + [(dt, theta)] * (N - rannacher_steps)
    tau = 0.0
    for h, th in steps:
        tau_new = tau + h
        lo_old, hi_old = _boundaries(option, american, K, S_max, tau, r, q)
        lo_new, hi_new = _boundaries(option, american, K, S_max, tau_new, r, q)
        # explicit part: (I + (1-th) h L) V_old
        e = (1 - th) * h
        rhs = V[1:-1] + e * (alpha * V[:-2] + beta * V[1:-1] + gamma * V[2:])
        # implicit part: (I - th h L) V_new
        im = th * h
        l, d, u = -im * alpha, 1 - im * beta, -im * gamma
        rhs[0] -= l[0] * lo_new
        rhs[-1] -= u[-1] * hi_new
        V_inner = _solve_step(l, d, u, rhs, payoff[1:-1], american)
        V = np.concatenate([[lo_new], V_inner, [hi_new]])
        tau = tau_new

    dS = S[1] - S[0]
    price = float(np.interp(S0, S, V))
    j = int(np.clip(round(S0 / dS), 1, M - 1))
    delta = float((V[j + 1] - V[j - 1]) / (2 * dS))
    gamma_ = float((V[j + 1] - 2 * V[j] + V[j - 1]) / dS**2)
    return FDResult(price, delta, gamma_, S, V)


def binomial_price(S0, K, T, r, sigma, option="put", american=True, q=0.0, n_steps=2000):
    """Cox-Ross-Rubinstein binomial tree (independent check on the PDE solver)."""
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    j = np.arange(n_steps + 1)
    ST = S0 * u ** (n_steps - j) * d**j
    V = np.maximum(ST - K, 0) if option == "call" else np.maximum(K - ST, 0)
    for n in range(n_steps - 1, -1, -1):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        if american:
            S_n = S0 * u ** (n - np.arange(n + 1)) * d ** np.arange(n + 1)
            ex = np.maximum(S_n - K, 0) if option == "call" else np.maximum(K - S_n, 0)
            V = np.maximum(V, ex)
    return float(V[0])
