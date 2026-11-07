# hadamard_rybczynski_droplet — REVIVED (campaign node IN26-L2)

Status: **revived** (was `dead` per
`../pending/hadamard_rybczynski_droplet/terminal_verdict_m155.md`).

Original death certificate (m155): *"G2 — no module separates viscosity ratio from
density ratio with acceptable accuracy … This kills not just HR but every
buoyant-drop/bubble benchmark class."*  The benchmark was un-runnable, not
un-implemented.

## What was closed

| gap | closure | evidence |
|---|---|---|
| **G2** viscosity ratio slaved to density ratio | new `src/tensorlbm/two_phase_viscosity.py`: per-cell `tau(x) = 1/2 + mu(phi(x))/(rho(x) c_s^2)` inverts the BGK relation, so `mu_d/mu_c = lambda` is imposed independently of `rho_d/rho_c` | realised `lambda` exact to 9e-16 over a 4x4 (rho-ratio, lambda) grid; lattice probe (D2Q9 shear wave) recovers `nu = c_s^2(tau-1/2)` to 0.4%, and at `rho_d/rho_c = 2` the realised mu-ratio is 1.0002 / 3.9965 for targets 1 / 4 |
| **HR drag law never verified** | new `src/tensorlbm/hadamard_rybczynski.py`: analytic law + endpoints + an axisymmetric Stokes drop-drag solver that solves the *same* boundary-value problem HR solved | 8-point lambda ladder: max error 0.80% (window A) / 2.00% (window B); bubble endpoint +1.19%; Stokes endpoint −0.49%; observed grid order 1.70 |

Not closed (unchanged from m155): **G3** (3D two-phase total-density-contrast
buoyant droplet in the LBM engine) and **G4** (Allen–Cahn phase dynamics).  The
LBM two-phase engine still cannot *directly* time-march a buoyant droplet; what
this node restores is (i) the missing viscosity-ratio knob and (ii) a
verification-grade reference for the drag law that any future engine must
reproduce.

## Result table (from `artifacts/.../evidence.json`)

| lambda | F_HR | F_inf (H/R=10,20,40) | error | F_inf (H/R=20,40,80) | error |
|---|---|---|---|---|---|
| 0.01 | 12.6286 | 12.7164 | +0.70% | 12.8363 | +1.64% |
| 0.03728 | 12.7922 | 12.8848 | +0.72% | 13.0085 | +1.69% |
| 0.1389 | 13.3329 | 13.4386 | +0.79% | 13.5756 | +1.82% |
| 0.5179 | 14.7103 | 14.8282 | +0.80% | 15.0048 | +2.00% |
| 1.931 | 16.7056 | 16.7861 | +0.48% | 17.0367 | +1.98% |
| 7.197 | 18.0830 | 18.0982 | +0.08% | 18.4130 | +1.82% |
| 26.83 | 18.6238 | 18.6043 | −0.10% | 18.9475 | +1.74% |
| 100 | 18.7873 | 18.7564 | −0.16% | 19.1085 | +1.71% |

Endpoints: `lambda -> 0` gives 12.7164 vs `4 pi mu_c R U = 12.5664` (+1.19%,
criterion <5%); `lambda -> inf` (lambda = 1e6 rigid check) gives 18.8141 vs
`6 pi mu_c R U = 18.8496` (−0.19%, criterion <3%).

## Reproduce

```bash
PYTHONPATH=src python benchmarks/hadamard_rybczynski_droplet_revival/run_revival.py     # ~36 s
PYTHONPATH=src python benchmarks/hadamard_rybczynski_droplet_revival/make_report_md.py
PYTHONPATH=src pytest tests/test_hadamard_rybczynski_droplet.py -q                      # 28 tests, ~3 s
```

Artifacts: `artifacts/hadamard_rybczynski_droplet_revival/evidence.json`,
`artifacts/hadamard_rybczynski_droplet_revival/figures/fig[1-5]*.png`.
Paper-style report: `docs/benchmarks/hadamard_rybczynski_droplet_revival_2026-11.md`.

## Four traps that cost the most time (read before touching the solver)

1. **Axis singularity.** With unknowns `(u_r, u_theta)` the `cot(theta)` term
   amplifies bilinear interpolation error at the Gauss points nearest the axis;
   the divergence penalty then dominates the functional and the minimiser
   distorts the whole field — measured as a **3.6x** drag error. Use
   `v = u_theta / sin(theta)`.
2. **Pointwise divergence penalty is unusable.** Use the *element-mean*
   (reduced integration) divergence: `Lambda (int_e div u dV)^2 / V_e`.  With it
   the answer is insensitive to `Lambda` over 1e4..1e7.
3. **This is a Stokes solver, not an LBM time stepper.** HR is the `Re -> 0`
   asymptote; a finite-Re LBM run cannot reach 3% error, by construction.
4. **The container must be extrapolated.** At `R/H = 0.1` confinement inflates
   the drag by ~19% (lambda = 0.01) and ~31% (rigid limit). A three-container
   linear fit in `R/H` is mandatory, and the fit residual (0.3-0.6%) is the
   honest error bar.