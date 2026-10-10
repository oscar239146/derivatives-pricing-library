"""Reproduce every figure and number in the README.

    python examples/make_figures.py              # synthetic option chain for the smile
    python examples/make_figures.py --ticker SPY # live chain via yfinance
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from derivlib import (bs_greeks, bs_price, binomial_price, bump_greeks, delta_hedge_pnl,  # noqa: E402
                      fd_price, mc_asian_call, mc_european, mc_pathwise_greeks, smile_from_chain)

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "font.size": 10, "legend.frameon": False,
})
P = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
lines: list[str] = []


def log(s=""):
    print(s)
    lines.append(s)


def save(fig, name):
    from matplotlib.ticker import FuncFormatter
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if axis.get_scale() == "log":
                axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                axis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


# 1. Monte Carlo convergence and variance reduction -----------------------------------
def monte_carlo():
    bs = bs_price(100, 100, 1, 0.05, 0.2)
    Ns = np.array([1_000, 4_000, 16_000, 64_000, 256_000])
    variants = {"Plain": {}, "Antithetic": {"antithetic": True},
                "Control variate": {"control_variate": True},
                "Antithetic + control": {"antithetic": True, "control_variate": True}}
    fig, ax = plt.subplots(figsize=(6.4, 4))
    log("## Monte Carlo: European call (S=K=100, T=1, r=5%, vol=20%)")
    log(f"Black-Scholes price: {bs:.4f}\n")
    log("| Method | Price (256k paths) | Std error | Variance reduction |")
    log("|---|---|---|---|")
    base_se = None
    for (name, kw), col in zip(variants.items(), C):
        se = [mc_european(**P, n_paths=int(n), seed=7, **kw).std_error for n in Ns]
        res = mc_european(**P, n_paths=int(Ns[-1]), seed=7, **kw)
        base_se = base_se or res.std_error
        log(f"| {name} | {res.price:.4f} | {res.std_error:.4f} | {(base_se / res.std_error) ** 2:.1f}x |")
        ax.loglog(Ns, se, "o-", color=col, label=name, markersize=5)
    ax.loglog(Ns, base_se * np.sqrt(Ns[-1] / Ns), "--",
              color=MUTED, linewidth=1, label=r"$\propto 1/\sqrt{N}$")
    ax.set(xlabel="Number of paths", ylabel="Standard error of price",
           title="Monte Carlo error falls as 1/√N; variance reduction shifts the line down")
    ax.legend()
    save(fig, "mc_convergence.png")

    plain = mc_asian_call(**P, n_steps=52, n_paths=100_000, seed=8)
    cv = mc_asian_call(**P, n_steps=52, n_paths=100_000, control_variate=True, seed=8)
    log("\n## Arithmetic Asian call (52 weekly fixings, 100k paths)")
    log("| Method | Price | Std error |")
    log("|---|---|---|")
    log(f"| Plain MC | {plain.price:.4f} | {plain.std_error:.4f} |")
    log(f"| Geometric-Asian control variate | {cv.price:.4f} | {cv.std_error:.5f} |")
    log(f"\nControl variate cuts the standard error {plain.std_error / cv.std_error:.0f}x, "
        f"equivalent to {(plain.std_error / cv.std_error) ** 2:,.0f}x more paths.")


# 2. PDE convergence + American put --------------------------------------------------
def pde():
    bs = bs_price(100, 100, 1, 0.05, 0.2, "put")
    Ms = [100, 200, 400, 800, 1600]
    errs = [abs(fd_price(**P, option="put", M=m, N=400).price - bs) for m in Ms]
    dS = [400 / m for m in Ms]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.loglog(dS, errs, "o-", color=C[0], label="Crank–Nicolson error", markersize=5)
    ax.loglog(dS, errs[-1] * (np.array(dS) / dS[-1]) ** 2, "--", color=MUTED, linewidth=1,
              label=r"$\propto \Delta S^2$")
    ax.set(xlabel="Grid spacing ΔS", ylabel="|PDE price − Black–Scholes|",
           title="Crank–Nicolson converges at second order in ΔS")
    ax.legend()
    save(fig, "pde_convergence.png")
    log("\n## Crank–Nicolson European put vs Black–Scholes")
    log("| Grid points M | ΔS | Abs error |")
    log("|---|---|---|")
    for m, d, e in zip(Ms, dS, errs):
        log(f"| {m} | {d:.2f} | {e:.2e} |")

    t0 = time.perf_counter()
    am = fd_price(**P, option="put", american=True)
    t_fd = time.perf_counter() - t0
    tree = binomial_price(**P, option="put", american=True, n_steps=5000)
    log("\n## American put (S=K=100)")
    log(f"PDE (Brennan–Schwartz, 800x400 grid, {t_fd:.2f}s): {am.price:.4f}  |  "
        f"CRR tree (5000 steps): {tree:.4f}  |  European: {bs:.4f}  |  "
        f"early-exercise premium: {am.price - bs:.4f}")

    S = am.grid_S
    eu = fd_price(**P, option="put")
    mask = (S > 50) & (S < 150)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(S[mask], np.maximum(100 - S[mask], 0), color=MUTED, linewidth=1, linestyle=":",
            label="Payoff max(K−S, 0)")
    ax.plot(S[mask], eu.grid_V[mask], color=C[0], label="European put")
    ax.plot(S[mask], am.grid_V[mask], color=C[1], label="American put")
    ex = S[(am.grid_V - np.maximum(100 - S, 0) < 1e-6) & (S < 100) & (S > 0)]
    if len(ex):
        ax.axvline(ex.max(), color=C[1], linewidth=1, linestyle="--")
        ax.annotate(f"exercise boundary ≈ {ex.max():.1f}", (ex.max(), 30),
                    xytext=(6, 0), textcoords="offset points", color=INK, fontsize=9)
    ax.set(xlabel="Spot S", ylabel="Option value", title="American vs European put")
    ax.legend()
    save(fig, "american_put.png")


# 3. Greeks three ways --------------------------------------------------------------
def greeks():
    g = bs_greeks(100, 100, 1, 0.05, 0.2)
    pricer = lambda S0, K, T, r, sigma: bs_price(S0, K, T, r, sigma)  # noqa: E731
    b = bump_greeks(pricer, P)
    mc_pricer = lambda S0, K, T, r, sigma: mc_european(  # noqa: E731
        S0, K, T, r, sigma, n_paths=200_000, seed=11).price
    b_mc = bump_greeks(mc_pricer, P, rel_bump_S=1e-2, bump_sigma=1e-2)
    pw = mc_pathwise_greeks(**P, n_paths=200_000, seed=12)
    log("\n## Call Greeks three ways (S=K=100)")
    log("| Greek | Analytic | Bump (on BS) | Bump (on MC, common random numbers) | Pathwise MC |")
    log("|---|---|---|---|---|")
    log(f"| Delta | {g.delta:.4f} | {b['delta']:.4f} | {b_mc['delta']:.4f} | "
        f"{pw['delta'][0]:.4f} ± {pw['delta'][1]:.4f} |")
    log(f"| Gamma | {g.gamma:.4f} | {b['gamma']:.4f} | {b_mc['gamma']:.4f} | n/a (payoff kink) |")
    log(f"| Vega | {g.vega:.3f} | {b['vega']:.3f} | {b_mc['vega']:.3f} | "
        f"{pw['vega'][0]:.3f} ± {pw['vega'][1]:.3f} |")


# 4. Implied volatility smile -------------------------------------------------------
def synthetic_chain(S=100.0, T=0.25, r=0.04):
    """Prices generated from a known skewed vol curve, so recovery can be checked."""
    strikes = np.arange(70, 131, 2.5)
    k = np.log(strikes / S)
    true_vol = 0.20 - 0.25 * k + 0.6 * k**2
    rows = []
    for K, v in zip(strikes, true_vol):
        for opt in ("call", "put"):
            rows.append({"strike": K, "option": opt, "price": bs_price(S, K, T, r, v, opt),
                         "true_vol": v})
    return pd.DataFrame(rows), S, T, r


def smile(ticker: str | None):
    if ticker:
        from derivlib.market_data import load_chain
        chain, S, T, expiry = load_chain(ticker)
        r, label = 0.04, f"{ticker} options expiring {expiry} (bid/ask mid)"
    else:
        chain, S, T, r = synthetic_chain()
        label = "Synthetic chain with known skew (run with --ticker SPY for live data)"
    df = smile_from_chain(chain, S, T, r)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for opt, col in (("put", C[0]), ("call", C[1])):
        d = df[df.option == opt]
        ax.plot(d.moneyness, d.iv * 100, "o", color=col, markersize=5, label=f"OTM {opt}s")
    if "true_vol" in df:
        ax.plot(df.moneyness, df.true_vol * 100, color=MUTED, linewidth=1, label="True vol")
        err = (df.iv - df.true_vol).abs().max()
        log(f"\n## Implied vol recovery\nMax |recovered − true| vol on synthetic chain: {err:.1e}")
    ax.set(xlabel="Moneyness K / S", ylabel="Implied volatility (%)", title=label)
    ax.legend()
    save(fig, "iv_smile.png")


# 5. Delta hedging -----------------------------------------------------------------
def hedging():
    freqs = [12, 52, 252]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4))
    log("\n## Delta-hedged short call: P&L vs rebalancing frequency (10k paths, real drift 10%)")
    log("| Rebalances / year | Mean P&L | Std dev P&L |")
    log("|---|---|---|")
    bins = np.linspace(-4, 4, 81)
    for n, col in zip(freqs, C):
        pnl = delta_hedge_pnl(**P, n_rebalance=n, n_paths=10_000, mu=0.10, seed=n)
        log(f"| {n} | {pnl.mean():+.3f} | {pnl.std():.3f} |")
        a1.hist(pnl, bins=bins, histtype="step", linewidth=2, color=col, label=f"{n}× / year")
    a1.set(xlabel="Hedged P&L at expiry", ylabel="Paths", title="Hedging error distribution")
    a1.legend()
    ns = np.array([4, 12, 26, 52, 126, 252, 504])
    sd = [delta_hedge_pnl(**P, n_rebalance=int(n), n_paths=10_000, seed=99).std() for n in ns]
    a2.loglog(ns, sd, "o-", color=C[0], markersize=5, label="Std dev of P&L")
    a2.loglog(ns, sd[0] * np.sqrt(ns[0] / ns), "--", color=MUTED, linewidth=1,
              label=r"$\propto 1/\sqrt{n}$")
    a2.set(xlabel="Rebalances per year", ylabel="Std dev of hedged P&L",
           title="Error halves for every 4× more rebalancing")
    a2.legend()
    save(fig, "delta_hedging.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=None)
    args = ap.parse_args()
    monte_carlo()
    pde()
    greeks()
    smile(args.ticker)
    hedging()
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
