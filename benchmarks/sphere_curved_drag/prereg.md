# Pre-registration — curved-boundary (BFL) sphere drag ladder for `GAP-LBM-SPHERE`

`mission m16085 · repo TensorLBM · branch agent/m16085 · anchor GAP-LBM-SPHERE`

Written **before** the production runs. Anything measured afterwards that contradicts
this text is reported as a result, not redefined.

## 1. What is already on the books (and what is not)

| source | what it establishes | what it leaves open |
|---|---|---|
| `benchmarks/pending/sphere_drag_closure` (m10330) | voxel (staircase) sphere, D3Q19/BGK, periodic box, velocity-pinned Guo forcing; drag from momentum balance `(1-1/(2τ)) g V_f` and from wall momentum exchange agree to 0.35 % | **C_D vs Schiller–Naumann is +14.8 % at Re=10, +16.5 % at Re=1** — anchor criterion (`< 8 %`) not met; confinement ladder was incomplete (one λ per Re) |
| `docs/P399W2_N04_IBM_CURVED_FORCE.md` | IBM marker force on a sphere, Re=100 | Cd −26.9 % in a 15 D box, −7.9 % in 24 D → wake recycling, not force integration; **honest FAIL** |
| `docs/SPHERICAL_FORCE_METHOD.md`, `src/lbm/boundary/spherical_force.py` | D3Q27 BFL spherical force formulation | cylinder/sphere *cylinder* drag C1, C3 are honest FAIL there too |
| `src/tensorlbm/interpolated_bc.py` + `src/tensorlbm/bfl_d3q19.py` | D3Q19 BFL interpolated bounce-back **with a force ledger** (`return_force`, `BFLForceDecomposition`) | never driven on a sphere flow against an **absolute analytic** reference |

The anchor asks for the **curved boundary force method** (曲面边界力法) and `C_D` vs the
Schiller–Naumann correlation inside 8 % over `0.1 < Re < 1000`. Neither half is on record:
the only certified sphere drag in this repository is *staircase* geometry, and the only
BFL-on-a-curved-body results are the honest FAILs above.

## 2. What this node measures

One **analytic-radius** sphere (SDF, not voxel), fixed at the centre of a periodic cube of
side `L`, driven by a uniform body force whose strength is adjusted by a
velocity-pinning controller so that the mean fluid velocity equals `u_char`. The wall is
imposed with the repository's D3Q19 BFL interpolated bounce-back
(`compute_q_sphere` + `bouzidi_bounce_back_d3q19`), which places the no-slip surface at
exactly `R` — the geometry the correlation is defined on.

Two independent force instruments, both recorded for every case:

| # | estimator | formula | role |
|---|---|---|---|
| 1 | global momentum balance | `F_mb = (1 - 1/(2τ)) g_tail V_fluid` | **verdict** (exact for a periodic cell at steady state) |
| 2 | BFL momentum exchange | `F_bfl` = `return_force` ledger of `bouzidi_bounce_back_d3q19` | consistency check (§3 S2) |

`C_D = 2 F / (ρ u² π R²)`, `Re = ρ u (2R) / μ`, both from the **measured** `u` and the
**geometric** `R` (no volume-equivalent fudge factor — that substitution is what the
staircase cases need and is precisely what the curved wall removes).

## 3. Frozen criteria

| id | criterion | verdict rule |
|---|---|---|
| **S1** | Stokes limit, absolute: `C_D = 24/Re` for a sphere in a simple-cubic array, Hasimoto (1959) `K(λ) = (1 − 2.837λ + 4.19λ³ − 27.4λ⁵)⁻¹`, `λ = 2R/L` | `Re = 0.1`, `λ = 1/6, 1/8`: `|C_D,meas / K(λ) − 240| / 240 < 3 %` **in each box** → the curved-boundary instrument + force bookkeeping is validated against an exact analytic reference |
| **S2** | force-estimator consistency | `|F_mb − F_bfl| / F_mb < 2 %` in every steady case |
| **S3** | staircase vs curved A/B, identical `D=12`, `L=48`, `Re=10` | curved `|C_D − C_D^SN|` strictly smaller than the staircase value of `benchmarks/pending/sphere_drag_closure` (`conf_Re10_L48`) |
| **S4** | anchor criterion: `C_D(Re=10)` vs Schiller–Naumann `24/Re (1 + 0.15 Re^0.687)` | λ-ladder `L = 36, 48, 60`, linear fit of `C_D` vs `λ`, `R² ≥ 0.9`; `|C_D(λ→0) − C_D^SN| / C_D^SN < 8 %` |
| **S5** | negative control | `g = 0`, `Re=10` box, 400 steps: `|F| < 1e-6` and `C_D` reported as zero/undefined |
| **S6** | deliverables | `results.json` with per-case records + sha256 provenance, `pytest ≥ 8` cases in `tests/test_sphere_curved_drag.py`, report `docs/GAP-LBM-SPHERE_curved_boundary_verification.md` with figures |

A criterion that fails is recorded as FAIL with its number. No criterion is relaxed after
the fact; if a run is statistically unconverged it is reported `steady=false` and excluded
from the fit, never silently averaged in.

## 4. Frozen numerical protocol

| item | value | reason |
|---|---|---|
| lattice / collision | D3Q19 / BGK | repository default (`tensorlbm.solver3d`) |
| geometry | analytic sphere `R = 6` (`D = 12`), SDF wall via `compute_q_sphere` | the curved-wall path under test |
| box | periodic cube, `L = 36, 48, 60` (`λ = 1/6, 1/8, 1/10`) | confinement ladder |
| `u_char` | `0.05` lattice (`Ma = 0.087`) | compressibility ≤ 1 % of the dynamic head |
| `ν` | `u_char · D / Re` (`Re = 0.1, 10`) | pins Re |
| `τ` | `0.5 + 3ν` → `18.5` at `Re=0.1`, `0.68` at `Re=10` | BGK regime, both stable |
| forcing | Guo (2002) second-order source, prefactor `(1−1/(2τ))`, applied in fluid cells only | the remembered momentum is exactly what estimator 1 measures |
| controller | drag feed-forward `g ← g·(u_target/u_meas)^k` with relaxation, gain capped | a prescribed `g` needs O(10⁴) spin-up steps |
| steady gate | tail-half relative drift of `u` and of `g` `< 1 %` (tail = last 50 samples, every 20 steps) | else `steady=false` |
| steps | `Re=10`: 8000; `Re=0.1`: 4000 (cap 20000) | prior staircase cases needed 4800–8160 |

## 5. Honest limitations fixed in advance

* `Re ≤ 10` only. The anchor spans `Re < 1000`; splitting off the wake-dominated regime
  (`Re ≥ 100`, where the sphere wake is still steady but the box must be ≥ 24 D) is a
  separate node, because that is where the P399 IBM attempt failed for a *wake-recycling*
  reason that no force-method work fixes.
* The λ-extrapolation assumes a single linear confinement slope over `λ ∈ [1/10, 1/6]`;
  the Stokes-limit ladder (S1) is what justifies using the same fit family at `Re=10`.
* `uv.lock`/CPU only; loong64 cluster runs are a different architecture and are noted if used.