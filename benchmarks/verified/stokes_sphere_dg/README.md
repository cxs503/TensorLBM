# stokes_sphere_dg — periodic-cell Stokes drag on a single sphere  ✅ VERIFIED

Status: **verified** (2026-10-09, Wave-11 track B; replaces the 2026-08-18 B10
pending archive — see History). All 12 tiers within 3% of the locked reference,
per-φ |err| strictly decreasing with resolution, all steady-state gates and the
budget gate pass. Controller independently recomputed every judged quantity
from the raw archived series (12/12 exact match).

## Physics

Stokes (Re → 0) flow in a periodic cell containing one no-slip sphere, driven
by a uniform body force g_x. The measured quantity is the dimensionless drag
correction

    K = F_ledger / (6 π μ a U_sup)

where U_sup is the volume-averaged superposed fluid velocity (the cell relaxes
to the periodic Stokes solution) and F_ledger = g_x·N_fluid·ΔV is the exactly
constant momentum ledger (total force balance: the body force input equals the
sphere drag in the steady state). A second, independent boundary measurement
F_ME (BFL momentum-exchange ledger) cross-checks it (gate S3). The single-sphere
dilute limit is K → 1; finite φ raises K above 1.

This is a *new problem class* for the benchmark suite: body-force-driven
periodic Stokes flow (no inlet, no walls), complementing the existing
inflow-driven cases.

## Reference (locked pre-run, double-derived)

K_ref(φ) was locked before any formal run (reference_table.json,
md5 af32757f9398e3f85b2bd0022c7a8efd) from two independent routes:

* **K_ref_bie_P5** — self-produced periodic boundary-integral Stokes solver,
  Hasimoto-periodic Green function, P5 panel resolution (P-scan {3,4,5}
  converged);
* **K_series_invE** — Ewald-split inverse-series evaluation of the same
  periodic problem.

Route consistency ≤ 9.12e-04% on every tier (gate: ≤0.5%). Literature anchors
(transcribed verbatim, phase-0 audit): Hasimoto dilute-limit series and
Zick & Homsy Table 2 (φ = 0.027/0.064/0.125 → K = 2.008/2.810/4.292).
Amendment A1 changed only the drive calibration (U_sup_target = 0.05/(6R) so
that Re_a = 0.05 exactly on every tier, isolating discretization error); the K
columns were bitwise unchanged.

## Results (all 12 tiers, machine values from result.json)

| φ | R (cells) | L (cells) | steps | K_sim | K_ref | err |
|---|---|---|---|---|---|---|
| 0.005 | 8 | 75 | 79,491 | 1.4090 | 1.4237 | -1.03% |
| 0.005 | 12 | 113 | 181,606 | 1.4101 | 1.4211 | -0.78% |
| 0.005 | 16 | 151 | 325,290 | 1.4109 | 1.4198 | -0.63% |
| 0.005 | 24 | 226 | 726,417 | 1.4131 | 1.4211 | -0.57% |
| 0.010 | 8 | 60 | 36,421 | 1.5543 | 1.5836 | -1.85% |
| 0.010 | 12 | 90 | 81,943 | 1.5620 | 1.5836 | -1.36% |
| 0.010 | 16 | 120 | 145,665 | 1.5649 | 1.5836 | -1.18% |
| 0.010 | 24 | 180 | 327,738 | 1.5668 | 1.5836 | -1.06% |
| 0.020 | 8 | 48 | 15,973 | 1.7786 | 1.8314 | -2.88% |
| 0.020 | 12 | 71 | 34,054 | 1.8080 | 1.8513 | -2.34% |
| 0.020 | 16 | 95 | 61,350 | 1.8065 | 1.8463 | -2.15% |
| 0.020 | 24 | 143 | 139,922 | 1.8039 | 1.8413 | -2.03% |

* main gate |err| ≤ 3%: **pass on 12/12** (worst 2.88%)
* convergence gate, |err| strictly decreasing per φ (R = 8 → 12 → 16 → 24):

| φ | \|err\| sequence |
|---|---|
| 0.005 | 1.03% → 0.78% → 0.63% → 0.57% |
| 0.01 | 1.85% → 1.36% → 1.18% → 1.06% |
| 0.02 | 2.88% → 2.34% → 2.15% → 2.03% |

* steady-state gates: S0 mass drift ≤ 1.3e-07 (gate 1e-6), S1 swing ≤ 3.1e-04
  (1e-3), S2 tail convergence ≤ 5.7e-04 (1e-3), S3 dual-estimator ≤ 4.6e-04 (1%)
* budget: 40.12 h wall ≤ 46 h limit

### Reading notes (disclosed)

* prereg par.5 says "err strictly monotonically decreasing"; all errors are
  negative and rise toward zero with resolution, so the magnitude reading is
  used — the same reading the "finest tier err ≤ 3%" clause itself requires.
  Signed sequences are archived in result.json.
* φ = 0.02, R = 8: the formula tail window (3194 steps) is shorter than the
  5000-step sliding window, so S1 degenerates to a single window (swing not
  measurable, recorded 0.0 with s1_windows = 1); S2
  (5.75e-04) and S3 still bind the tail there.

## Configuration

D3Q19, float64, BGK with τ = 1.0 (Stokes linearity), Guo (2002) body-force
collision (relaxation toward feq(u*), u* = u_raw + F/(2ρ) — the naive
collide-at-raw-u halves the force at τ = 1), sphere radius R cells centred in
an L³ periodic cell with φ_gate = N_solid/L³ matched to the reference φ per
tier (reference_table.json), BFL link-based no-slip with exact ray-sphere q
(sparse kernels), uniform body force g_x calibrated per tier so that
Re_a = 6·U_sup·R = 0.05 throughout, initialised at feq(ρ=1, u=(U_sup_target,0,0))
everywhere (Amendment A2 init change; the pre-amendment halted run-1 is
archived as halt evidence in the campaign staging). steps =
ceil(6.5 · τ_spindown · 1.0426) per tier (spindown-fit receipt in the campaign
md5 chain). Sampling every 100 steps; judged window = tail 20%.

## History

The 2026-08-18 pending archive (B10) coupled a spherical-shell DG layer to a
Cartesian LBM far field for Stokes flow at Re = 0.1; its single grid landed
2.15% but refinement moved it 53% — an honest FAIL recorded then. This remake
is a different, cleaner problem class (fully periodic, single-method, exact
ledger); the old archive is retired with this promotion.

## Reproduce

    python verify.py                 # independent re-judge from case_*.json
    python verify.py --write         # additionally embed the recompute into result.json
    python run.py --only 0.005:8     # rerun a tier (writes results_stokes_sphere_dg/)

Artifacts: case_<run_id>.json × 12 (full per-tier machine results including
the complete sample series), reference_table.json (locked reference + per-tier
drive calibration) and prereg.md (frozen gate protocol; Amendments A1/A2 in
the appendix) — both byte-identical to the campaign-locked files (md5 pins
asserted by run.py), result.json (aggregate verdict + provenance chain),
verify.py (independent checker), run.py (formal runner; see
result.json:provenance.in_repo_run_py_note for the portability port).
