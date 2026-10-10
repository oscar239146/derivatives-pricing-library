import numpy as np
import pytest

from derivlib import (bs_greeks, bs_price, binomial_price, bump_greeks, delta_hedge_pnl,
                      fd_price, geometric_asian_call, implied_vol, mc_asian_call,
                      mc_european, mc_pathwise_greeks)

P = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)


def bs(**kw):
    a = {**P, **kw}
    return bs_price(a["S0"], a["K"], a["T"], a["r"], a["sigma"], a.get("option", "call"))


# --- Black-Scholes -------------------------------------------------------------
def test_bs_textbook_value():
    assert bs() == pytest.approx(10.4506, abs=1e-4)          # Hull's standard example
    assert bs(option="put") == pytest.approx(5.5735, abs=1e-4)


def test_put_call_parity():
    for K in (80, 100, 120):
        c, p = bs(K=K), bs(K=K, option="put")
        assert c - p == pytest.approx(P["S0"] - K * np.exp(-P["r"] * P["T"]), abs=1e-10)


def test_analytic_greeks_match_bumps():
    pricer = lambda S0, K, T, r, sigma: bs_price(S0, K, T, r, sigma, "call")  # noqa: E731
    g = bs_greeks(P["S0"], P["K"], P["T"], P["r"], P["sigma"], "call")
    b = bump_greeks(pricer, P)
    for name in ("delta", "gamma", "vega", "rho", "theta"):
        assert getattr(g, name) == pytest.approx(b[name], rel=5e-3), name


# --- Monte Carlo -----------------------------------------------------------------
@pytest.mark.parametrize("anti,cv", [(False, False), (True, False), (False, True), (True, True)])
def test_mc_european_within_3se(anti, cv):
    res = mc_european(**P, n_paths=200_000, antithetic=anti, control_variate=cv, seed=1)
    assert abs(res.price - bs()) < 3 * res.std_error


def test_variance_reduction_reduces_error():
    plain = mc_european(**P, n_paths=100_000, seed=2).std_error
    cv = mc_european(**P, n_paths=100_000, control_variate=True, seed=2).std_error
    assert cv < plain / 2


def test_asian_control_variate():
    plain = mc_asian_call(**P, n_steps=50, n_paths=100_000, seed=3)
    cv = mc_asian_call(**P, n_steps=50, n_paths=100_000, control_variate=True, seed=3)
    assert abs(plain.price - cv.price) < 3 * plain.std_error
    assert cv.std_error < plain.std_error / 10
    # arithmetic average >= geometric average, so the arithmetic Asian is worth more
    assert cv.price > geometric_asian_call(100.0, 100.0, 1.0, 0.05, 0.2, n_steps=50)


def test_pathwise_greeks():
    g = bs_greeks(P["S0"], P["K"], P["T"], P["r"], P["sigma"])
    pw = mc_pathwise_greeks(**P, n_paths=400_000, seed=4)
    assert abs(pw["delta"][0] - g.delta) < 3 * pw["delta"][1]
    assert abs(pw["vega"][0] - g.vega) < 3 * pw["vega"][1]


# --- PDE / trees -----------------------------------------------------------------
@pytest.mark.parametrize("option", ["call", "put"])
def test_crank_nicolson_european(option):
    fd = fd_price(**P, option=option)
    assert fd.price == pytest.approx(bs(option=option), abs=1e-3)
    g = bs_greeks(P["S0"], P["K"], P["T"], P["r"], P["sigma"], option)
    assert fd.delta == pytest.approx(g.delta, abs=1e-3)
    assert fd.gamma == pytest.approx(g.gamma, abs=1e-3)


def test_american_put_pde_vs_tree():
    fd = fd_price(**P, option="put", american=True).price
    tree = binomial_price(**P, option="put", american=True, n_steps=2000)
    assert fd == pytest.approx(tree, abs=5e-3)
    assert fd > bs(option="put")  # early-exercise premium is positive


# --- Implied vol & hedging -----------------------------------------------------------
@pytest.mark.parametrize("K,option", [(70, "put"), (100, "call"), (130, "call"), (90, "put")])
def test_implied_vol_round_trip(K, option):
    for sig in (0.1, 0.3, 0.8):
        price = bs(K=K, sigma=sig, option=option)
        assert implied_vol(price, P["S0"], K, P["T"], P["r"], option) == pytest.approx(sig, abs=1e-6)


def test_implied_vol_rejects_arbitrage():
    assert np.isnan(implied_vol(0.0001, 100, 50, 1, 0.05, "call"))  # below intrinsic


def test_hedging_error_shrinks_like_sqrt_n():
    sd = [delta_hedge_pnl(**P, n_rebalance=n, n_paths=20_000, seed=5).std() for n in (25, 100)]
    assert sd[1] < sd[0] * 0.6        # 4x more rebalancing -> ~half the error
    pnl = delta_hedge_pnl(**P, n_rebalance=252, n_paths=20_000, mu=0.15, seed=6)
    assert abs(pnl.mean()) < 0.05      # drift does not matter once hedged
