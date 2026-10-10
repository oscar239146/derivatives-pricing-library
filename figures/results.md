## Monte Carlo: European call (S=K=100, T=1, r=5%, vol=20%)
Black-Scholes price: 10.4506

| Method | Price (256k paths) | Std error | Variance reduction |
|---|---|---|---|
| Plain | 10.4526 | 0.0291 | 1.0x |
| Antithetic | 10.4376 | 0.0206 | 2.0x |
| Control variate | 10.4414 | 0.0111 | 6.9x |
| Antithetic + control | 10.4397 | 0.0155 | 3.5x |

## Arithmetic Asian call (52 weekly fixings, 100k paths)
| Method | Price | Std error |
|---|---|---|
| Plain MC | 5.8441 | 0.0255 |
| Geometric-Asian control variate | 5.8529 | 0.00070 |

Control variate cuts the standard error 36x, equivalent to 1,312x more paths.

## Crank–Nicolson European put vs Black–Scholes
| Grid points M | ΔS | Abs error |
|---|---|---|
| 100 | 4.00 | 3.98e-02 |
| 200 | 2.00 | 9.90e-03 |
| 400 | 1.00 | 2.48e-03 |
| 800 | 0.50 | 6.22e-04 |
| 1600 | 0.25 | 1.59e-04 |

## American put (S=K=100)
PDE (Brennan–Schwartz, 800x400 grid, 0.71s): 6.0895  |  CRR tree (5000 steps): 6.0902  |  European: 5.5735  |  early-exercise premium: 0.5160

## Call Greeks three ways (S=K=100)
| Greek | Analytic | Bump (on BS) | Bump (on MC, common random numbers) | Pathwise MC |
|---|---|---|---|---|
| Delta | 0.6368 | 0.6368 | 0.6346 | 0.6370 ± 0.0013 |
| Gamma | 0.0188 | 0.0188 | 0.0189 | n/a (payoff kink) |
| Vega | 37.524 | 37.524 | 37.298 | 37.457 ± 0.169 |

## Delta-hedged short call: P&L vs rebalancing frequency (10k paths, real drift 10%)
| Rebalances / year | Mean P&L | Std dev P&L |
|---|---|---|
| 12 | -0.018 | 1.871 |
| 52 | -0.013 | 0.931 |
| 252 | +0.001 | 0.426 |
