# Differentiable SUBOFF voxelization: parameters -> SDF -> masks -> dC_D/dshape (2026-08-25)

First slice of the Phase-3 ship-design-surrogate roadmap item "可微体素化反传":
given `SuboffConfig` design parameters, compute **∂(log10 C_D)/∂θ end to end**
through the B4 drag-surrogate conditioning chain, so that future
gradient-guided design loops have an analytic adjoint of the voxelizer instead
of noisy design-space finite differences.

Module: [`src/tensorlbm/diff_voxelize.py`](../src/tensorlbm/diff_voxelize.py)
(tests: [`tests/test_diff_voxelize.py`](../tests/test_diff_voxelize.py),
validation run report: `/nfs/wangxi/runs/diff_vox_20260825/report.md` on the
dev server). New files only — no existing module was modified.

## 1. What is differentiated

```
θ (SuboffConfig axes)  ──►  analytic component SDFs (hull / sail / fins, ft)
                        ──►  occupancy masks   (soft sigmoid or STE)
                        ──►  geometry channels (log_aproj_ratio, sail_frac,
                                fin_frac, solid_frac — same recipe as the
                                numpy training pipeline)
                        ──►  condition_v3 row (8,)
                        ──►  CondFNODrag ensemble (5 serve seeds)
                        ──►  log10 C_D      ⟹  d/dθ by autograd
```

Differentiable axes (`DIFF_PARAM_NAMES`): `r_over_l`, `sail_scale`,
`fin_scale`, `l_over_d_mult`, `nose_len_mult`, `stern_len_mult`,
`sail_x_mult` — all multiplicative scales, so finite-difference steps in the
validation are *relative* steps.

### Geometry model

Each component reproduces the exact DARPA predicates of
`tensorlbm.suboff_cad` as a signed distance-like field in feet (negative
inside):

| component | field |
|---|---|
| hull | smooth-max of the radial distance `r − R(ξ)` (4-segment profile, torch port of `suboff_radius_profile`) and the **true signed** axial-interval distance of the surface of revolution |
| sail | 2-D cross-section SDF (exact box + scaled-ellipse cap approximation) swept by the footprint half-width `h(x)`, intersected with the axial footprint interval |
| fins | smooth-max of thickness / span / chord-end distances per cruciform pair (NACA 4-digit thickness with SUBOFF coefficients), smooth-min'ed between pairs |

Unions/intersections use the polynomial smooth-min/-max (Quilez);
`_interval_dist` returns the *signed* distance (a clamped version would let
the smooth-max bump displace the whole end-surface shell — that bug cost
368 cells / IoU 0.911 before the fix).

### Masking

* `soft_mask(sdf, tau)` — sigmoid occupancy `σ(−sdf/τ)`, band half-width `tau` ft.
* `straight_through_mask(sdf, tau)` — **hard forward (`sdf <= 0`), soft
  backward**: forward values are bit-identical to the CAD voxel masks, so the
  *forward* chain reproduces the training pipeline exactly, while the
  *backward* chain is the sigmoid slope (STE).

### Channel recipe

`mask_channels` mirrors `tensorlbm.ai.drag_cond.suboff_geometry_features`:
disjoint decomposition `v_sail = Σ m_sail(1−m_hull)`,
`v_fin = Σ m_fin(1−m_hull)(1−m_sail)`, column-OR projection
`aproj = Σ_zy [1 − Π_x (1−m)]`, and the four-channel vector
`[log10(aproj/aproj_bare), v_sail/v_solid, v_fin/v_solid, v_solid]`.
`condition_vector_diff` assembles the 8-column `condition_v3` row with
tensor-valued logs. `DiffDragEnsemble` re-implements the
`ModelEnsembleBackend.predict` arithmetic without `torch.no_grad`; the
differentiated scalar is `log10` of the **linear-space** ensemble mean C_D.

## 2. Fidelity knobs (explicit, documented)

| knob | default | meaning | measured effect |
|---|---|---|---|
| `smooth_k` | 0.01 ft (~1/20 cell) | smooth-min/-max blending radius | hard-mask IoU vs `build_suboff_mask` = **1.0000** (0 differing cells) on mother + 3 variants; larger `smooth_k` rounds component corners (gradients across joins) at O(smooth_k) level-set displacement |
| `tau` | 0.02 ft (~1/9 cell) | occupancy temperature, boundary-band half-width | estimator agreement; see §3.4 — at 0.05 ft the STE appendage sensitivities flip sign against every independent estimate, so 0.02 is the calibrated default |

Both are parameters of `suboff_component_sdfs` / `mask_channels` /
`drag_forward` / `drag_gradients`; nothing is hidden.

## 3. Validation summary (all numbers from the run report)

### 3.1 Hard-mask parity vs the CAD voxel builder

`IoU(soft-union hard mask, build_suboff_mask)` on PRODUCTION_GRID
(128×64×64), smooth_k = 0.01 ft:

| variant | IoU | differing cells |
|---|---|---|
| mother | 1.0000 | 0 |
| appendage (sail 1.35, fin 1.3) | 1.0000 | 0 |
| hull form (l/d 1.15, nose 1.2) | 1.0000 | 0 |
| hull form blunt (l/d 0.85, stern 1.3, sail-x 1.05) | 1.0000 | 0 |
| per hull_type (bare_hull / with_sail / full) | 1.0000 | 0 |

The torch radius profile matches the numpy DARPA profile to < 1e-8 in the
interior (the two exact tips carry a documented root-softening bias < 1e-4).

### 3.2 Channel parity vs the numpy training pipeline

`mask_channels` vs `suboff_geometry_features` over 8 designs (full/with_sail/
bare_hull × sail 0.7–1.5 × fin 0.8–1.3): **max_abs = 6.9e-18** (float64
round-off), and the hard voxel counts `v_bare / v_sail / v_fin / v_solid` are
exactly equal on every design. The STE forward of the whole chain reproduces
the reference pipeline `log10 C_D` to **0.0** at the mother design.

### 3.3 End-to-end gradients vs finite differences

Full table (3 designs × 3 params × {STE autograd, soft autograd, hard FD at
h=0.05 and h=0.10, soft FD at h=1e-3 and 2e-3, numpy-reference FD}) lives in
`/nfs/wangxi/runs/diff_vox_20260825/report.md` §3; per-seed components in §4.
Headlines (d(log10 C_D)/dθ, tau = 0.02 ft):

| design | param | STE | soft | hard FD h=.05 | hard FD h=.10 |
|---|---|---|---|---|---|
| mother | sail_scale | +0.0079 | +0.0041 | +0.0136 | +0.0184 |
| mother | fin_scale | −0.0071 | −0.0015 | +0.0118 | +0.0116 |
| mother | l_over_d_mult | −0.1206 | −0.0890 | +0.0039 | −0.0192 |
| slender | sail_scale | +0.0025 | +0.0075 | +0.0258 | +0.0196 |
| slender | l_over_d_mult | −3.6373 | −0.0844 | −0.0522 | −0.1530 |
| blunt | sail_scale | +0.0105 | +0.0086 | +0.0131 | +0.0136 |
| blunt | l_over_d_mult | −0.4648 | +0.0070 | −0.0082 | −0.0442 |

* **Machinery gate**: soft autograd vs soft FD (two steps) ≤ 5% on every
  axis/design (mostly ≤ 0.5%) — the autograd graph is correct.
* **Appendage axes** (in training support): STE, soft and both hard-FD steps
  share the sign on the sail axis at all three designs, magnitudes within a
  factor ~2–4 of the measured hard slope.
* **Hard FD is a staircase**: its two step sizes disagree by up to 123%
  (step-stability column) wherever the window straddles few voxel jumps —
  no single h is "the" derivative there.

### 3.4 tau calibration (report §3.1)

| tau (ft) | estimator | sail | fin | l_over_d |
|---|---|---|---|---|
| 0.02 (default) | STE / soft | +0.008 / +0.004 | −0.007 / −0.0015 | −0.121 / −0.089 |
| 0.05 | STE / soft | −0.150 / −0.039 | −0.154 / +0.019 | −0.970 / −0.044 |
| 0.10 | STE / soft | −0.359 / −0.029 | −0.337 / +0.021 | −1.376 / −0.055 |
| — | hard FD h=.05 | +0.014 | +0.012 | +0.004 |

STE magnitude grows ~linearly with tau (junction-band artifact: the STE
linearisation freezes the *hard* neighbour occupancy, so overlapping
hull-sail / hull-fin junction bands are double-counted relative to the soft
estimator). The default 0.02 ft is where the estimators converge and the
sail-axis direction matches the hard FD.

### 3.5 Test suite and CI gates

* `tests/test_diff_voxelize.py`: **25 tests** — tensorlbm venv (CUDA): 25
  passed; ci-cpu venv (OMP_NUM_THREADS=1, no CUDA): 24 passed, 1 skipped
  (CUDA smoke). Coverage: mask IoU, STE==soft identity for occupancy sums,
  STE vs soft-FD direction, channel parity vs numpy, smooth-k monotonicity,
  end-to-end STE vs FD (machinery ≤ 5%, direction as in §3.3), CUDA vs CPU.
* `ruff check src tests` clean; `ruff format --check src tests examples
  benchmarks` clean.
* `mypy src/tensorlbm --ignore-missing-imports` (CI advisory gate):
  2761 errors with the new module and 2761 with it removed (removal method)
  — net contribution **0**, and the module itself produces no mypy errors.

## 4. Honest deviations and limitations

1. **STE bias (dominant).** The delivered gradient is the slope of the soft
   occupancy at the hard configuration — not the derivative of the integer
   voxel counts, which is a.e. zero with jumps. Hard FD at a fixed h is
   design- and h-dependent (staircase); the report carries both steps and a
   step-stability column instead of a single "deviation" number.
2. **STE magnitude depends on tau** (~linear; §3.4). The default is
   calibrated against hard FD on the appendage axes; raising tau without
   re-checking directions is unsafe.
3. **fin_scale is a near-cancellation** in the trained response: STE ~1e-2
   magnitudes whose sign can disagree with soft/hard estimates at
   off-mother designs (measured at sail 1.2/fin 1.15: STE +0.032 vs soft
   −0.024, hard FD −0.10/−0.04). The fin channel recipe is faithful; the
   ensemble's fin_frac vs solid_frac response is what nearly cancels.
4. **Smooth-union displacement.** smooth_k = 0.01 ft displaces the zero
   level set by O(0.01 ft) at component joins only; away from joins the
   predicates are exact (IoU 1.0).
5. **Ellipse cap and swept-fin distances are approximations** (scaled
   ellipse bound; local frame distances for the swept fin — obliquity
   ignored in magnitude, not in sign).
6. **Frozen flow channels.** FNO field channels 0–3 (ux, uy, uz, rho) are
   frozen at one corpus reference row: the *flow response* to shape change
   is NOT differentiated. Gradient flows only through the solid-mask slice
   (channel 4) and the 8-dim condition vector.
7. **Field mask provenance.** Channel 4 here is the CAD mask slice; training
   used the simulation solid mask (cache `mask_bit_eq` false for most rows) —
   a systematic offset shared with the reference forward, so parity checks
   are unaffected but absolute predictions inherit it.
8. **Out-of-support hull-form axes.** `l_over_d_mult` / `nose_len_mult` /
   `stern_len_mult` / `sail_x_mult` were not swept in the B4 v4 corpus; the
   numpy channel reference has no such axis by construction. Gradients along
   them (e.g. slender l_over_d STE −3.64) are extrapolations of the trained
   channel recipe — machinery validation, not calibrated design
   sensitivities.

## 5. API (new code only)

```python
from tensorlbm.diff_voxelize import (
    DiffParams, suboff_component_sdfs,          # θ -> component SDFs
    soft_mask, straight_through_mask,           # SDF -> occupancy
    mask_channels,                              # -> channels + counts
    DiffDragEnsemble, drag_forward,             # -> log10 C_D (grad-capable)
    drag_gradients,                             # -> d(log10 C_D)/dθ + per-seed
    drag_finite_difference,                     # FD oracle of the same forward
    reference_drag_forward,                     # numpy training-pipeline oracle
)

out = drag_gradients(
    {"sail_scale": 1.2, "fin_scale": 1.0, "l_over_d_mult": 1.05},
    ckpt_paths, field_row, re=200.0, device="cuda",
)
out["grads"]["sail_scale"]        # ensemble STE gradient
out["member_grads"]["sail_scale"] # per-seed components
```

`ste=False` everywhere switches to the pure soft surrogate (the estimator
whose finite differences match autograd to ≤ 5%); use it for sensitivity
studies where forward parity is not needed.
