# REVIVED — see ../../hadamard_rybczynski_droplet_revival/

The verdict in `terminal_verdict_m155.md` (gap **G2**: "no module separates
viscosity ratio from density ratio with acceptable accuracy") is superseded.

* library fix: `src/tensorlbm/two_phase_viscosity.py` — per-cell
  `tau(x) = 1/2 + mu(phi(x))/(rho(x) c_s^2)` decouples `lambda = mu_d/mu_c` from
  `rho_d/rho_c` (realised exactly; the legacy constant-tau lock is reproduced by
  `legacy_uniform_relaxation` for contrast).
* verification: `src/tensorlbm/hadamard_rybczynski.py` — analytic HR law + an
  axisymmetric Stokes drop-drag solver; 8-point lambda ladder within 0.80% of the
  law, bubble endpoint +1.19%, Stokes endpoint -0.19%, grid order 1.70.
* evidence: `artifacts/hadamard_rybczynski_droplet_revival/evidence.json`,
  report `docs/benchmarks/hadamard_rybczynski_droplet_revival_2026-11.md`.

Still open from m155: **G3** (3D two-phase total-density-contrast buoyant droplet
in the LBM engine) and **G4** (Allen-Cahn phase dynamics).  This note does not
claim them.
