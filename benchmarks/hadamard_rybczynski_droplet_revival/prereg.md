# Prereg — hadamard_rybczynski_droplet revival (campaign node IN26-L2)

Locked before the numbers were computed (criteria come from the campaign node and
from the analytic law, not from what the solver happened to produce).

## Analytic target (locked)

F(lambda) = 2*pi*mu_c*R*U*(2+3*lambda)/(1+lambda); U(lambda) from the force balance
(4/3)pi R^3 drho g = F.  Limits: lambda->0 -> 4*pi*mu_c*R*U, lambda->inf -> 6*pi*mu_c*R*U,
lambda=1 -> 5*pi*mu_c*R*U.

## Acceptance criteria (machine-adjudicated by run_revival.py)

| # | criterion | threshold |
|---|---|---|
| 1 | lambda settable independently of rho_d/rho_c (operator level) | == target to machine precision |
| 2 | lambda realised on the lattice at rho_d/rho_c = 2 | < 1% |
| 3 | drag error over 8 log-spaced lambda in [0.01, 100] | < 3% each |
| 4 | bubble endpoint (lambda -> 0) vs 4*pi*mu_c*R*U | < 5% |
| 5 | Stokes endpoint (lambda -> inf) vs 6*pi*mu_c*R*U | < 3% |
| 6 | observed grid-convergence order of the drag | >= 1 |
| 7 | test suite for the module | >= 10 tests, all green |

## Method locks (chosen before the runs, no post-hoc tuning)

* Interface: sharp, on the mesh line r = R.  No diffuse-interface smearing of lambda.
* Frame: drop rest frame, outer spherical wall translating at U e_z.
* Drag: dissipation identity F*U = int 2 mu eps:eps dV (no traction integral).
* Container: linear-in-(R/H) extrapolation over three domains, reported with its residual.
* Production window A: H/R = 10, 20, 40; audit window B: 20, 40, 80 (declared up front
  as the robustness check, not as a fallback).
* Resolution: h_wall/R = 0.03, growth 1.35, penalty 1e6 (insensitivity tested).
* Honest-FAIL rule: if a criterion misses, report it; no criterion was relaxed
  after seeing results.  Residual error floors (container extrapolation model,
  interface-deformation and finite-Re physics) are listed in the report.
