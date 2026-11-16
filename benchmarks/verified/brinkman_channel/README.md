# `brinkman_channel` — Darcy/Brinkman drag closure vs the exact solution

Wall-bounded porous-medium flow: a plane channel of half-width `h` filled with a
homogeneous porous medium of permeability `K`, driven by a body acceleration `g`.
The volume-averaged momentum balance is the Brinkman equation
`nu u'' - (nu/K) u + g = 0`, `u(±h) = 0`, whose closed form is

```
u(y)  = (gK/nu) [1 - cosh(y/sqrt K)/cosh(beta)],   beta = h/sqrt K
u_bar = (gK/nu) [1 - tanh(beta)/beta] = u_D * R(beta),  u_D = gK/nu
tau_w = g sqrt(K) tanh(beta)
```

| | |
|---|---|
| module | `src/tensorlbm/brinkman_channel.py` (drag closure `a = g - (nu/K) u`, D2Q9 BGK + zero-sum Guo forcing, half-way bounce-back walls) |
| reference | closed form above, re-derived in-repo (`verify_exact_solution`: ODE residual, BCs, Darcy/Stokes limits) |
| preregistered | `prereg.json` written before the runs; thresholds in `run.py:THRESHOLDS` |
| acceptance | `eps_inf <= 3 %` for every Da; median `<= 1 %`; shape function `<= 1 %`; `mass_drift <= 1e-9`; observed order `>= 1.5` |
| verdict | **PASS** — worst `eps_inf = 1.903 %` (Da=1e-3, β=15.8, i.e. a 2-cell Brinkman layer), median `0.250 %` |

## Results (`result.json`, `figs/`)

Da sweep at `H = 64`, `tau = 0.8`, `g = 1e-5`:

| Da | beta | eps_inf % | eps_2 % | shape fn % | amp err % | mass drift | steps |
|---|---|---|---|---|---|---|---|
| 1e-03 | 15.81 | 1.903 | 0.381 | 0.0082 | +0.0082 | 1.2e-13 | 2 000 |
| 3e-03 | 9.13 | 0.743 | 0.177 | 0.0050 | +0.0050 | 2.2e-13 | 3 500 |
| 1e-02 | 5.00 | 0.250 | 0.079 | 0.0030 | +0.0030 | 6.5e-13 | 10 000 |
| 3e-02 | 2.89 | 0.098 | 0.041 | 0.0009 | +0.0009 | 1.5e-12 | 24 000 |
| 1e-01 | 1.58 | 0.044 | 0.025 | 0.0054 | -0.0054 | 3.2e-12 | 49 500 |

Self-similar refinement at `Da = 1e-2` (`K ~ H^2`, `g ~ H^-2`, so Re/Ma fixed):
`eps_inf = 0.885 % -> 0.250 % -> 0.067 %` for `H = 32/64/128`, observed order
**1.82 / 1.91**.

Controls: drag off (`K -> inf`) reproduces the Poiseuille parabola to
`eps_inf = 0.022 %`; the truncated-Guo negative control leaks mass at
`2.3e-5` over 4 000 steps and is therefore *detected* (the shipped source is
zero-sum, `|sum_i S_i| < 1e-18`).

## Reproduce

```bash
export PYTHONPATH=src
python benchmarks/verified/brinkman_channel/run.py prereg
python benchmarks/verified/brinkman_channel/run.py swing     # Da sweep  (~15 s)
python benchmarks/verified/brinkman_channel/run.py ladder    # H ladder   (~15 s)
python benchmarks/verified/brinkman_channel/run.py control   # limits     (~15 s)
python benchmarks/verified/brinkman_channel/run.py merge     # verdict + figures
python -m pytest tests/test_brinkman_channel.py -q           # 11 tests, ~8 s
```

## Limitations (honest)

* The drag is **explicit**, so `lam = nu/K << 1` is required. The Darcy limit
  `beta -> infinity` (sub-grid Brinkman layer, `K` of order `nu` or less) is out
  of scope and is **not** claimed; reaching it needs an implicit penalisation.
* `mu_eff = mu` is assumed (the Brinkman effective viscosity is not a free
  parameter here); differentiating `mu_eff/mu` is a separate case.
* Planar walls only (no curvature): this case isolates the *drag closure*, it
  says nothing about curved-wall staircase error.
