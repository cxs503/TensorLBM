# XVAL: free-energy binary LBM equilibrium pressure (sailfish semantics transplant)

Anchor `XVAL-LBM-FREE-ENERGY-BINARY-EQUILIBRIUM-PRESSURE`, repo TensorLBM, mission m17879 (round 2).
Acceptance: owner band `< 15 %` per KPI, declared before the run in
`xval/XVAL-LBM-FREE-ENERGY-BINARY-EQUILIBRIUM-PRESSURE/predeclared.json`.

## 1. Problem setting and reference

The previous round (m17761) reached no verifiable number (no finish, no run). This round starts from the
**pressure-tensor** route of the free-energy binary model instead of the `-phi grad(mu)` Korteweg-force
route, because only the former gives a *measurable* pressure field from the populations, and therefore a
machine-checkable equilibrium: `d P_NN / dx = 0` on a flat interface and `p_in - p_out = sigma / R` on a
droplet.

Reference (external anchor, cited line by line, not hand-copied):

| what | where |
|---|---|
| fluid equilibrium, `pb = rho/3 + A(-phi^2/2 + 3 phi^4/4)`, `-kappa phi lap(phi)` in the bulk term, `f_0 = rho - sum` | `sailfish-team/sailfish` `sym_equilibrium.py:26-48` |
| order-parameter equilibrium, `mu = A(-phi+phi^3) - kappa lap(phi)`, `g_0 = phi - sum` | `sym_equilibrium.py:50-70` |
| D2Q9 `wi` (axis 1/3, diag 1/12) and anisotropic `wxx/wyy/wxy` = 1/3, -1/6, -1/24, `ex ey/4` | `lb_binary.py:185-268` |
| nine-point isotropic gradient/Laplacian | `templates/finite_difference_optimized.mako:46-48` |
| method lineage | Pooley, Furtado & Carr, *Phys. Rev. E* **78**, 056709 (2008) |

**Comparison limit (declared, not upgraded):** sailfish is CUDA/JIT (sympy codegen + pyopencl/pycuda) and
cannot execute in this environment, so the comparison arm is a semantics-exact transplant of its
equilibrium, coefficients and FD operators into
`src/tensorlbm/phasefield/free_energy_pressure.py`, not a run of the OSS binary.

Analytic anchors re-derived in the module docstring from `f = A(-phi^2/2 + phi^4/4) + kappa|grad phi|^2/2`:

* interface width `xi = sqrt(2 kappa / A)`
* surface tension `sigma = (2/3) sqrt(2 kappa A) = (2/3) A xi`
* profile `phi = tanh((x - x0)/xi)`, and `kappa phi'' = A(phi^3 - phi)` implies `d P_NN/dx = 0` exactly

## 2. Method

`src/tensorlbm/phasefield/free_energy_pressure.py`:

* `equilibrium_f` / `equilibrium_g` — the two sailfish equilibria above (BGK, `lambda = 0` on D2Q9);
* `grad_lap` — the nine-point isotropic operators of `finite_difference_optimized.mako:46-48`;
* `collide_and_stream` — BGK collide, then pull streaming on the periodic grid (`sum_i f_i c_i / rho` is
  the velocity; **no Guo forcing**, the pressure tensor is inside the equilibrium);
* `pressure_tensor_measured` — `P_ab = sum_i f_i c_ia c_ib` from the populations (never from the analytic
  formula); `pressure_tensor_analytic` — the same tensor from `phi` for the closure gate.

Runner `benchmarks/xval_free_energy_binary_equilibrium/run_xval.py` (arms declared in `ARMS`), aggregation
`make_report.py` -> `artifacts/xval_free_energy_binary_equilibrium/results.json` + figures in
`docs/figures/xval_fe_pressure_*.png`. Tests `tests/test_free_energy_pressure_xval.py`.

## 3. Parameters

| A | xi | kappa = A xi^2/2 | Gamma | tau_phi | M = Gamma(tau_phi-1/2)cs^2 | tau_f | nu | lattice |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 3 (flat), 2 (R=16) | 0.45, 0.2 | 0.2 | 1.5 | 0.0667 | 1.0 | 1/6 | D2Q9, float64, periodic |

Grids: flat band 64x8 with a slab of width 16 (two interfaces); droplet R=16 in 96x96, step budgets 4000 / 8000.
Screening note (reported, not used to set thresholds): at `Gamma = 1.0` with `tau_phi = 1.5` the same
discretisation goes non-finite at step ~200 on a 64^2 R=10 droplet, while `Gamma <= 0.5` is finite over
1200 steps -- so the declared point keeps `M` at 1/15.

## 4. Results (numbers from `results.json`, none hand-typed)

| KPI | reference | got | error | band |
|---|---|---|---|---|
| closure gate `max_ab |sum f^eq cc - (P+rho uu)|/max|P|` | 0 | 3.10e-16 (flat), 4.57e-16 (droplet) | machine precision | 1e-12 |
| `sigma` flat band (per interface) | 0.20000 | 0.195981 | **2.01 %** | 15 % |
| `P_NN` uniformity across the band | 0 | 1.48e-5 rel | **0.0015 %** | 15 % |
| Laplace `dp` at R=16 | 8.3333e-3 (`sigma/R`) | 8.3181e-3 | **0.18 %** | 15 % |
| Laplace from the ray-integrated tensor `sigma_ray` | `dp = sigma_ray/R`, `sigma_ray` = 0.12517 | ratio 1.0632 | **6.32 %** | 15 % |
| spurious `max|u|` | 0 | 4.3e-6 (flat), 5.2e-4 (droplet) | -- | 1e-2 |
| mass drift `|sum rho - N|/N` | 0 | 2.9e-15 | -- | 1e-3 |

Verdict from the predeclared gate: **PASS** for the two arms that ran (flat band, dropR16);
`artifacts/xval_free_energy_binary_equilibrium/results.json` carries the per-check booleans.

`sigma_ray` (curved interface, 16 rays, 0.5 dx sampling) sits 6.4 % below the analytic `sigma` while the
flat-band tensor integral is 2.0 % below: the ray integral is the least accurate route (nearest-neighbour
radial sampling), which is why the Laplace KPI is declared against the analytic `sigma/R` and the ray route
is reported as an independent cross-check.

## 5. Convergence / sensitivity

* interface width: `sigma` from the tensor integral is 0.19598 at `xi = 3` (64x8 band) vs `sigma_analytic =
  0.2`; the 2 % deficit is a discretisation effect that decreases with `xi` (O(h^2) interface resolution).
* curvature: Laplace ratio 0.9982 at R=16 (`R/xi = 8`). The two further declared arms (R=24 at `R/xi = 12`,
  R=36 at `xi = 3`, `R/xi = 12`) were **submitted but not completed** before the call budget ran out — they
  are the outstanding part of this round and must be re-run (`--arm dropR24 --arm dropR36`) to state the
  curvature/resolution trend. This is reported, not glossed.
* mobility/viscosity screening: `Gamma = 1.0` (M = 1/6) is unstable at `nu = 1/6` on a static droplet;
  `Gamma <= 0.5` is stable for `nu` in [1/6, 7/6] at fixed `tau_phi = 1.5`. The instability is a real
  parameter boundary of the explicit CH + pressure-tensor scheme, not a transplant bug (the equilibrium
  closure holds at 1e-16 and `sigma`/`dp` are correct in the stable region).

## 6. Failure modes and artefacts

* `reply_cmd`: `python benchmarks/xval_free_energy_binary_equilibrium/run_xval.py --arm all --threads 8`
  then `python benchmarks/xval_free_energy_binary_equilibrium/make_report.py`.
* tests: `python -m pytest tests/test_free_energy_pressure_xval.py -q` (log `/tmp/pytest_xval.log`)
* artefacts: `artifacts/xval_free_energy_binary_equilibrium/{flat,dropR16,results}.json`;
  `docs/figures/xval_fe_pressure_{flat_band,dropR16,laplace}.png`.
* cluster: job `cj1704` (same command, stage_dir = this worktree) was still queued at submission+5 min;
  the local arms above are the ones that produced the numbers.
* Known trap found and recorded: a **single** flat interface cannot exist in a periodic domain (the tanh
  wrap creates a second, infinitely sharp interface and the run diverges within 10 steps). The flat arm
  therefore uses a periodic *band* (two interfaces) and `sigma` is halved once; including that factor twice
  was the first bug in this round's analysis and is now a named, tested factor.
* Limitation: `sum_i wi_i wxx_i = 7/72 != 0`, so the kappa term contributes `kappa|grad phi|^2/3` to
  `sum_i f_i`; it is absorbed by `f_0 = rho - sum`, which is exactly what sailfish does, and it is now
  asserted in the tests instead of assumed to vanish.
