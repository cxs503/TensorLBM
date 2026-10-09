# propeller_openwater — DTMB 4119 open-water KT/KQ (PENDING / BLOCKED)

**Status: NOT verified — honest structural block documented below.**

Goal: single-propeller open-water curve `KT(J)`, `10KQ(J)` using the repo's
propeller capability, against the DTMB 4119 open-water experiment (2+ J points,
acceptance < 3%).

## Verdict

The benchmark was built and executed end-to-end (2 J points, real SDAA runs),
but it cannot be accepted. Three **independent, quantified** blockers each
already preclude a < 3 % KT/KQ match; none is a tuning knob.

### B1 — Geometry infidelity (solver-independent)
`tensorlbm.propeller_cad` accepts only **scalar** propeller parameters and builds
an analytic *elliptic chord × NACA4-thickness* blade. It cannot ingest the real
DTMB 4119 section tables (`OpenProp Prop4119_input.m`: `XCoD` peak 0.4622 D,
NACA a=0.8 meanline, DTRC-modified NACA66 thickness, zero skew/rake). Quantified
against the real table (`artifacts/geometry_gap_dtmb4119.json`):

| quantity | real 4119 | CAD delivers | error |
|---|---|---|---|
| peak c/D @ 0.7R | 0.4622 | 0.3142 | **−32 %** |
| expanded area ratio A_E/A_0 | 0.599 (nominal 0.60) | 0.454 | **−24 %** |
| t/c @ 0.7R | 0.054 | 0.27 (D=48) … 0.40 (D=32) | **5–7× too thick** |
| tip chord c/D | 0.002 (tapers) | 0.291 (open elliptic tail) | spurious tip load |
| camber | NACA a=0.8 meanline | none (symmetric shaping) | missing |

A 24 % blade-area deficit and 5–7× too-thick sections cannot reproduce 4119
loading to 3 %.

### B2 — Reynolds mismatch (lattice stability)
The experimental DTMB 4119 at model scale runs at `Re_D = n D²/ν ≈ 1.4e6`
(n = 15 rps, D = 0.3048 m, ν = 1e-6 m²/s). LBM stable relaxation caps
`ν_lu ≳ 5e-4` (τ ≥ ~0.5015). The rotating-mask rotation rate is fixed by the
temporal contract (`rpm = 1/steps_per_rev`), and the practical LBM velocity cap
(tip Ma ≲ 0.2) bounds `rpm·D² ≲ 0.037·D`, so the **achievable lattice Reynolds
number is `Re_D ≲ 73·D`** — ≈ 2.3e3 at D = 32, i.e. **~600× below experiment**.

The repo's own mapping, `propeller_benchmark.map_physical_propeller_refinement`,
run on the real 4119 physical case, returns **`status = fail_closed`**
(`artifacts/propeller_owt/physical_lattice_refinement.json`): matching the
experimental Re_D and J simultaneously forces `ν_lu ≈ 7e-7` (τ = 0.5000022,
below the stability floor).

### B3 — Low-Mach gate vs. temporal resolution
The moving-wall bounce-back is qualified only for tip Ma < 0.004
(`propeller_benchmark.LOW_MACH_TIP_GATE`). To keep a rotating voxel mask from
teleporting (>1 cell/step at the tip), `steps_per_rev` must be O(10²–10³), which
forces tip Ma ≈ 0.17 at the best feasible setting — **~42× above the gate**. The
low-Mach gate and a temporally resolvable rotation are mutually exclusive at any
blade-resolved grid.

Additional (secondary) contributors: domain blockage = πR²/(ny·nz) = **12.6 %**
(open-water tanks target < 5 %), and the legacy static ME torque estimator is
non-comparable (see below).

## What was executed (real runs, not fabricated)

Runner: [`run.py`](run.py). Rotation = Ladd moving-wall bounce-back with the mask
**re-voxelised every lattice update** at the current azimuth. Loads taken from the
**same moving-wall operator reaction** (`moving_wall_bounce_back_3d_with_reaction`)
and converted with the repo caliber `report_propeller_linkwise_loads`.
Legacy static momentum-exchange thrust is recorded as a cross-check.

* Geometry: Z=3, P/D(0.7R)=1.084, A_E/A_0=0.60, hub 0.20, D=32 lu
* Domain 160×80×80 lu; τ=0.503 (ν=0.001), Smagorinsky Cs=0.1
* rpm = 1/1024 (1024 steps/rev), tip Ma = 0.170, Re_D = 1000
* 1 rev warm-up + 1.5 rev sampling per J, device SDAA

<!-- RESULTS_TABLE_START -->
<!-- RESULTS_TABLE_END -->

## Why the numbers are not a solver bug to "fix"

* Moving-wall reaction thrust and the legacy static ME thrust **agree** at every
  sampled step — the thrust caliber is consistent; the offset is physical/geometric.
* The legacy static ME **torque** is `~O(10²)` off the reaction torque and is
  flagged non-comparable by the module itself — only the reaction torque is used.
* The offset is *larger* than geometry alone would predict (the CAD is
  under-area, a thinner propeller would give *lower* KT), consistent with the
  additional Re/blockage/Mach corrections all pushing the same way.

## Files

| file | content |
|---|---|
| `run.py` | open-water runner (smoke / single / full modes) |
| `result.json` | machine-readable verdict (`verified: false`) |
| `run_result.json` | raw per-J output from the executed sweep |
| `artifacts/geometry_gap_dtmb4119.json` | CAD-vs-real geometry quantification |
| `artifacts/propeller_owt/physical_lattice_refinement.json` | repo's own `fail_closed` proof |

## Reference

DTMB 4119 open-water experiment. Primary tabulation (standard open-water table
used across the 4119 CFD literature):

| J | KT | 10KQ | η_O |
|---|---|---|---|
| 0.50 | 0.2853 | 0.4641 | 0.489 |
| 0.60 | 0.2497 | 0.4269 | 0.559 |
| 0.70 | 0.2120 | 0.3893 | 0.607 |
| 0.80 | 0.1700 | 0.3482 | 0.622 |
| 0.833 | 0.1525 | 0.3319 | 0.609 |
| 0.90 | 0.1210 | 0.2959 | 0.586 |

Independent cross-check (2nd source): the OpenProp DTMB 4119 *replica design from
performance inputs* targets **KT = 0.15 at J = 0.833**, matching the tabulated
0.1525 to ≈ 1.6 % — the reference curve is consistent across sources.

Note on J vs Re: KT/KQ of a fixed geometry are functions of J **and** Re. The
table above is model-scale Re ≈ 1.4e6; the lattice runs at Re ≈ 1e3, a different
branch of that surface.