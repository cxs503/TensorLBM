# PREREG — XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTION

Anchor: `XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTION`
Repo: `TensorLBM` (branch `agent/m18793`)
Written **before** any validation number was produced (round 2, supersedes the
round-1 honest-FAIL of `m18664`, whose `fail_stage`/`failure_reason` were left
empty on the board — see `REPORT.md` §① ).

## Decision the experiment must support

Is the D2Q9 Zou/He inlet/outlet population **reconstruction** implemented in
`src/tensorlbm/boundary/boundaries.py` able to reproduce the analytic plane
Poiseuille solution of a 2-D channel to better than the owner's 15 % accuracy
standard (stretch: 3 %)?

## Competing alternatives

* **A1 (repo arm)** — repo functions `zou_he_inlet_velocity` /
  `zou_he_outlet_pressure` / `apply_zou_he_channel_boundaries` driven by the repo
  D2Q9 solver kernels (`tensorlbm.solver.collide_bgk`, `tensorlbm.solver.stream`,
  `tensorlbm.d2q9.equilibrium/macroscopic`), full-way bounce-back walls.
* **A0 (analytic anchor)** — steady plane Poiseuille solution of the
  incompressible Navier–Stokes equations (no fitting of the shape).
* **A2 (OSS horizontal)** — `pylbm` / `Maarten-vd-Sande/lbm` from `~/oss-cache`:
  surveyed for a comparable reconstruction; used only as method-spectrum input,
  not as a numeric reference (result of the survey is recorded in `OSS_DIFF.md`).

## Hypotheses

* H1 (mechanics): the reconstruction is an **exact** algebraic closure — after
  applying the BC to an arbitrary (non-equilibrium) population field, the
  reconstructed column reproduces the prescribed macroscopic quantities to
  round-off. Falsified if any closure residual > 1e-12 (float64).
* H2 (physics, velocity inlet): a channel driven by a prescribed inlet velocity
  and a Zou/He pressure outlet develops the analytic parabola in the downstream
  half of the channel.
* H3 (physics, pressure driven): a channel driven by Zou/He **pressure inlet +
  pressure outlet** reproduces the analytic parabola with an amplitude given
  *a priori* by `u_max = G H² / (8 ν)`, `G = Δρ / (3 L)` — no amplitude fitting.

## Metrics (predeclared, primary metric first)

For each physics arm, on the developed profile at the measurement column:

| symbol | definition | threshold |
|---|---|---|
| `dp_max` (**primary**) | `max_y |ux(y) − u_ana(y)| / max_y u_ana` over fluid rows | **< 0.15** PASS, **< 0.03** stretch |
| `dp_l2` | `sqrt(mean_y (ux − u_ana)²) / max u_ana` | reported, not gating |
| `flux_err` | `|Q_out − Q_in| / |Q_in|`, `Q = Σ_fluid ρ ux` | < 0.01 |
| `closure_err` | max rel. residual of `ρ`, `ρux`, `ρuy` after BC | < 1e-12 |
| `drift` | `max|ux(t) − ux(t−1000)| / max ux` | < 1e-6 (steady state) |

**Stopping rule / no post-hoc switching**: `dp_max` at the measurement column
is the gating number; `dp_l2`, amplitude error and flux error are reported but
may not replace `dp_max`. If `dp_max ≥ 0.15` the verdict is honest-FAIL.
Channel length is chosen (not tuned after seeing results) so that
`L/H ≥ 6`, which is the standard developing-length margin for the laminar
channel inlet (Schlichting, *Boundary-Layer Theory*, §5: entrance length
`L_e/H ≈ 0.05 Re`; here `Re = U H/ν ≤ 10 ⇒ L_e/H ≤ 0.5`).

## Cases (fixed now)

| case | nx × ny | τ (ν) | driving | steps |
|---|---|---|---|---|
| C1 velocity inlet, coarse | 120 × 18 | 0.8 (0.1) | `u_in = 0.05` | ≤ 40 000, steady-stop |
| C2 velocity inlet, fine | 220 × 34 | 0.8 (0.1) | `u_in = 0.05` | ≤ 40 000 |
| C3 pressure driven, coarse | 120 × 18 | 0.8 (0.1) | `ρ_in = 1.002`, `ρ_out = 0.998` | ≤ 40 000 |
| C4 pressure driven, fine | 220 × 34 | 0.8 (0.1) | `ρ_in = 1.002`, `ρ_out = 0.998` | ≤ 40 000 |
| C5 pressure driven, Δρ sweep | 120 × 18 | 0.8 (0.1) | `Δρ ∈ {0.001, 0.002, 0.004}` | ≤ 40 000 |

Dimensionless group check (`C3`, `H = 17`): `G = 0.004/(3·120) = 1.111e-5`,
`u_max = G H²/(8ν) = 4.01e-3`, `Ma = u_max√3 = 6.9e-3` (incompressible regime),
`Re = u_max H/ν = 0.68`. Wall position convention: full-way bounce-back on the
solid node ⇒ no-slip plane at the wall-node centre ⇒ `H = ny − 1`.

Two resolutions (C1/C3 vs C2/C4) and two-to-three driving levels (C5) give the
required resolution and parameter sensitivity statements.

## Environment / reproducibility

CPU, `torch` float64, single process, `OMP_NUM_THREADS ≤ 4`, deterministic
(no randomness except the seeded closure test, `torch.manual_seed(19970317)`).
`repro_cmd` recorded in `REPORT.md` §⑥; raw JSON in `artifacts/`.

## Interpretation limits (declared up front)

* Single-relaxation-time BGK only; MRT/TRT and 3-D Zou/He are out of scope.
* Laminar only (`Re ≤ 700` for the finest case with `u_in = 0.1`).
* The residual difference to the analytic parabola contains both BC error and
  the intrinsic BGK wall-offset error; the two are separated in `REPORT.md` §⑤
  by the wall-offset diagnostic rather than merged silently.