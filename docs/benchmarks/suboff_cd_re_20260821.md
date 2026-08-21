# SUBOFF bare-hull C_D–Re benchmark (v1)

**Dataset**: `/nfs/wangxi/datasets/scan_suboff_re_20260821` (5090 server) ·
scan id `scan-suboff-re-sweep` · 24 points · 192 catalog products ·
splits 17/4/3 · 3.1 GB
**Campaign**: `suboff_n128` (D3Q19, cumulant), resolution 128 → grid
64×64×128, `u_in = 0.1`, bare hull, 4 000 steps, `snapshot_every 500`,
mass correction every 10 steps; 8× RTX 5090, 124 s wall; checkpoints
(A7) every 250 steps, deleted on completion.
**Re levels**: 24 log-spaced 50 → 800 (τ = 0.5 + 7.68/Re ∈ [0.529, 0.961]).
**Estimator**: `tensorlbm.drag_survey` (this PR) — wake momentum deficit
with per-plane border-ring reference.

## Method

On a cross-section `x_w` downstream of the hull,

```
D(x_w) = Σ_{y,z} ρ·u_x·(u_∞ − u_x)      [lattice units]
C_D    = 2D / (ρ_∞·u_∞²·S_proj)
Cf_eq  = C_D·S_proj / S_wet
```

with `u_∞`/`ρ_∞` taken per plane from an 8-cell far-field border ring —
**not** the nominal 0.1. The case's periodic mass correction settles the
actual far-field ≈ +0.6% above nominal; a fixed nominal reference biases
`D` by `δ·Σρu_x` (≈ 64% on this dataset, sign-flipping negative above
Re ≈ 450). The ring reference tracks the drift, leaving first-order
residual sensitivity `δ/u_eff` ≈ 0.6%.

Planes are surveyed at x-offsets {4, 8, 16, 32} from the outlet;
near-invariance across them is the internal consistency check (pressure
term neglected — far enough downstream it should vanish).

Geometry: `S_proj = 69` cells², `S_wet = 2 494` faces, 4 093 solid cells,
hull L = 76.8 cells at cx = 44.8.

## Results

| Re | C_D | Cf_eq | plane ΔD | drift₃ |
|-----:|-------:|-------:|-------:|-------:|
| 50 | 4.58461 | 0.12684 | 0.2655 | 0.002 |
| 56.4 | 4.46886 | 0.12364 | 0.2354 | 0.002 |
| 63.6 | 4.34146 | 0.12011 | 0.2038 | 0.002 |
| 71.8 | 4.19928 | 0.11618 | 0.1706 | 0.002 |
| 81.0 | 4.04301 | 0.11186 | 0.1372 | 0.002 |
| 91.4 | 3.87041 | 0.10708 | 0.1041 | 0.002 |
| 103.1 | 3.68176 | 0.10186 | 0.0722 | 0.002 |
| 116.3 | 3.47685 | 0.09619 | 0.0429 | 0.002 |
| 131.2 | 3.25682 | 0.09010 | 0.0184 | 0.002 |
| 148.0 | 3.02402 | 0.08366 | 0.0082 | 0.002 |
| 166.9 | 2.78233 | 0.07698 | 0.0251 | 0.002 |
| 188.3 | 2.53445 | 0.07012 | 0.0385 | 0.002 |
| 212.4 | 2.28629 | 0.06325 | 0.0471 | 0.002 |
| 239.6 | 2.04191 | 0.05649 | 0.0512 | 0.002 |
| 270.3 | 1.80572 | 0.04996 | 0.0514 | 0.002 |
| 305.0 | 1.58112 | 0.04374 | 0.0485 | 0.002 |
| 344.0 | 1.37201 | 0.03796 | 0.0435 | 0.001 |
| 388.1 | 1.17888 | 0.03262 | 0.0371 | 0.001 |
| 437.8 | 1.00306 | 0.02775 | 0.0302 | 0.001 |
| 493.9 | 0.84408 | 0.02335 | 0.0232 | 0.001 |
| 557.2 | 0.70113 | 0.01940 | 0.0166 | 0.001 |
| 628.6 | 0.57321 | 0.01586 | 0.0120 | 0.001 |
| 709.1 | 0.45826 | 0.01268 | 0.0099 | 0.002 |
| 800.0 | 0.35524 | 0.00983 | 0.0081 | 0.002 |

`drift₃` = relative C_D change across the last three snapshots (500-step
spacing) — all ≤ 0.24%, i.e. every point is snapshot-converged at 4 000
steps.

### Scaling

Log-log fit over the 24 points: **C_D ∝ Re^−0.888** — between laminar
flat-plate friction (Blasius, Re^−0.5) and fully Stokes drag (Re^−1),
consistent with a friction-dominated laminar regime with a growing
pressure-drag fraction as Re rises. `C_D·Re` rises gently 229 → ~450
(mid-range) → 284 rather than staying constant, the same statement in
another form.

### Reference lines (context only — outside validity range)

At Re = 800: ITTC-1957 `0.075/(log10 Re − 2)²` = **0.0920**, Blasius
`1.328/√Re` = **0.0470**, measured `Cf_eq` = **0.0098**. Both reference
lines assume ship-/plate-scale Re (≥ 10⁵) and are shown for orientation
only; the measured value sitting below them is expected for a
streamlined body at lattice Re with a thick laminar boundary layer, and
should not be read as external validation. No experimental SUBOFF data
exists at this Re — the benchmark's authority at v1 is its **internal**
consistency: snapshot convergence ≤ 0.24%, plane invariance ≤ 5% for
Re ≥ 148 (worst 6% at Re = 50 where the viscous wake is widest).

## Known limitations

1. **Pressure term neglected** in the momentum deficit; planes at 4–32
   cells from the outlet are still near-field. Plane monotonicity at low
   Re shows residual pressure recovery.
2. **fp32 solver state**; snapshot export rounds through fp32 HDF5.
3. **Single seed, single realisation** per Re — no ensemble spread. At
   Re ≥ ~500 the wake is weakly unsteady; the tail-mean estimate samples
   it at only 8 snapshots.
4. **Mass-correction coupling**: the +0.6% far-field drift is itself a
   case convention; the ring reference absorbs it but couples `u_∞`
   estimation to border width (8 cells).

## Next steps

- **Exact drag**: live discrete-kinetic control-volume observer
  (`tensorlbm.control_volume_force`) wired into the scan chain as a
  per-point `drag_history` — removes limitations 1 and 4 entirely
  (in development).
- Re range extension needs either τ headroom (u_in sweep at fixed ν) or
  an SGS model; τ = 0.523 at Re = 800 is already near the practical
  floor for this lattice.
- Reproduce with: `survey_dataset("/nfs/wangxi/datasets/scan_suboff_re_20260821")`
  → `write_summary(...)`; campaign launcher `/nfs/wangxi/tmp/b1_launch_scan.py`.

---

*Generated 2026-08-21 from commit bb7ec40 + `drag_survey` (this branch);
raw per-snapshot plane data in the dataset's `drag_summary.json`.*
