# Sub-cell wall position of the 2-point interpolated bounce-back: harness probe (m22892)

**Verdict: WITHHELD — no capability claim, no benchmark promotion.** The probe built here
does reproduce a quantitative deviation, but its own reference/forcing bookkeeping is
not yet calibrated, so the deviation cannot be attributed to the boundary operator.
The honest result is the calibrated next step, not a pass.

Anchor: `KPI-LBM-VERIFIED-CAPS` (count of capabilities that passed independent physical
validation). This mission does **not** add one; it records a reproducible negative.

## 1. Why this probe

`docs/STAIRCASE_WALL_OFFSET_20261111.md` and `xval/XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME/REPORT_V4.md`
leave three named items undone for the interpolated bounce-back (IBB) family:

* no resolution study (V4 §5: "resolution study 未做（N=16/32）");
* the sub-cell wall position `w_bot` was only an *upper bound* at `δ→0.9`;
* a 4.4 % dependence on the initial content of the solid slots (the "mass accounting
  contract") is still an honest FAIL.

All three need one thing that does not exist yet: a **staircase-free** configuration in
which the no-slip plane sits at a known fractional distance from the node, so that the
reference is the closed form and nothing else. `examples/ibb_subcell_probe.py` is that
configuration.

## 2. Configuration (no staircase anywhere)

Straight 2D channel, `N` fluid rows `y = 0 … N-1`, one solid row on each side. The
signed-distance field encodes walls at `y = -q0` and `y = N-1+q1` with `q0,q1 ∈ (0,1)`.
The exact solution is therefore a plane Poiseuille channel of half-width

```
H = (N - 1 + q0 + q1) / 2 ,   y_c = (-q0 + (N-1) + q1) / 2
u(y) = g (H^2 - (y - y_c)^2) / (2 nu) ,   nu = (tau - 1/2)/3
```

driven by a constant body force `g` (momentum source `s_i = 3 w_i (c_i·F)`), `tau = 0.8`,
`g = 1e-5`, D2Q9 BGK, doubly periodic in `x`. Initial field = the analytic profile, so
the run measures the steady closure error and nothing else.

Arms: `halfway` (`boundaries.bounce_back_cells`, the production operator),
`ibb_default` (`interpolated_bounce_back_2d(..., alpha=None)` → `alpha_bfl(q) = 4/(1+2q)`),
`ibb_two_branch` (`alpha = 2` for `q<1/2`, `1/q` for `q>=1/2`, i.e. the exact two-branch
Bouzidi–Firdaouss–Lallemand rule the module docstring describes).

Run: `PYTHONPATH=src python examples/ibb_subcell_probe.py` (≈100 s, single core).
Raw record: `outputs/ibb_subcell/results.json`, table: `outputs/ibb_subcell/table.txt`.

## 3. Measured numbers (programmatically injected, nothing hand-copied)

| N | q0=q1 | scheme | θ (max profile err, %) | Q/Q_ana − 1 (%) | u_max err (%) | ρ̄ |
|---|---|---|---|---|---|---|
| 8 | 0.25 | halfway | 37.17 | +58.24 | +37.14 | 1.000000 |
| 8 | 0.25 | ibb_default | 35.42 | +55.51 | +35.37 | 1.000000 |
| 8 | 0.25 | ibb_two_branch | 35.42 | +55.51 | +35.37 | 1.000000 |
| 8 | 0.5 | halfway | 20.30 | +29.68 | +20.24 | 1.000000 |
| 8 | 0.5 | ibb_default | 28.95 | +42.31 | +28.83 | 1.000000 |
| 8 | 0.5 | ibb_two_branch | 28.95 | +42.31 | +28.83 | 1.000000 |
| 8 | 0.75 | halfway | 6.37 | +8.81 | +6.32 | 1.000000 |
| 8 | 0.75 | ibb_default | 38.12 | +52.91 | +38.02 | 1.000000 |
| 8 | 0.75 | ibb_two_branch | 38.12 | +52.91 | +38.02 | 1.000000 |
| 16 | 0.25 | halfway | 16.98 | +25.43 | +16.02 | 1.000001 |
| 16 | 0.25 | ibb_default | 14.50 | +21.75 | +13.72 | 1.000000 |
| 16 | 0.25 | ibb_two_branch | 14.50 | +21.75 | +13.72 | 1.000000 |
| 16 | 0.5 | halfway | 9.93 | +14.37 | +9.34 | 1.000000 |
| 16 | 0.5 | ibb_default | 13.00 | +18.81 | +12.22 | 1.000000 |
| 16 | 0.5 | ibb_two_branch | 13.00 | +18.81 | +12.22 | 1.000000 |
| 16 | 0.75 | halfway | 3.50 | +4.90 | +3.25 | 1.000000 |
| 16 | 0.75 | ibb_default | 17.94 | +25.08 | +16.67 | 1.000004 |
| 16 | 0.75 | ibb_two_branch | 17.94 | +25.08 | +16.67 | 1.000004 |

### What this table can and cannot be read as

**(a) It cannot be read as an operator verdict.** Take the *geometry-exact* cell: `q0=q1=0.5`
puts the halfway wall exactly at `y = -0.5`, `y = N-0.5`, i.e. `H = N/2`, which is exactly
the `H` used in the reference. There is no staircase and no sub-cell ambiguity, yet the
production operator misses the closed form by **+20.2 % (N=8)** and **+9.3 % (N=16)**.
A 20 % residual at a lattice-aligned wall is a *harness* error (reference or forcing
convention), not a boundary-closure error — every published verification of the same
operator family on the same operator is at the `0.1 %` level
(`benchmarks/verified/poiseuille_2d_bodyforce`).

**(b) The residual scales like a fixed length, not like a discretisation order.**
Excess `u_max`: +20.2 % → +9.3 % when `N` doubles; `(1+err) ∝ (H+δ)²` gives
`δ ≈ 0.39 / 0.37` cells at `N = 8 / 16`. A constant ~0.37-cell offset with **no trend in
the number of nodes** is the signature of a *convention* offset, not of a mesh error.

**(c) The most probable convention candidate (not yet proven).** This probe injects the
body force as a plain post-collision momentum source `s_i = 3 w_i (c_i·F)`. The repo's
verified body-force benchmark instead uses the Guo/`(1 - ω/2)` form
(`run.py`: `ux_eq = ux + 0.5*force`, `cst = (1.0 - 0.5*omega)*3.0*force`). A plain source
leaves an `O(τ)` amplitude offset in the steady state; at `τ = 0.8` that is
`1/0.8 = 1.25`, i.e. the same size as the observed residual at `N = 8`
(measured/reference = 1.202 → `3.8 %` from `1/τ` exactly). **This is a hypothesis, not a
measurement**: at `N=16` the same division leaves `12.6 %` unexplained, so the factor is
not a pure `1/τ`.

**(d) `alpha` has no effect in this configuration — an unexplained anomaly.**
`ibb_default` (`alpha_bfl`) and `ibb_two_branch` agree to *every printed digit* in all six
cells, although `alpha_bfl(0.25) = 2.667 ≠ 2` and `alpha_bfl(0.75) = 1.6 ≠ 1.333`. With
cut links present on the axis **and** both diagonals of both walls (the solid rows are one
cell thick), `alpha` must change the reflected value at `q ≠ 0.5`. Either the IBB arm is
not reaching the operator in this harness, or the closure never enters the measured
profile. This was not resolved within the mission budget and is the first thing the next
mission should chase — it is a one-command check
(`python -c "from tensorlbm.boundary.interpolated_bounce_back import *"` plus a
two-run diff of the raw `f` arrays after one step).

**(e) Mass accounting is clean here.** `ρ̄ − 1 ≤ 4e-6` in all 18 arms, including the
first-step transient, so the V4 "solid-slot initialization" 4.4 % defect does **not**
reproduce in a channel with one solid row per wall on a non-wrapping axis. Consistent
with V4's own attribution to the initialization contract rather than to the closure.

## 4. Error judgement (explicit)

| quantity | reference | measured | < 3 % ? |
|---|---|---|---|
| sub-cell wall position `w(q)`, production `halfway` | `w = 0.5` (geometry-exact at `q=0.5`) | profile err 20.2 % / 9.3 % (N=8/16) | **No — and not interpretable as an operator error (§3a)** |
| sub-cell wall position, IBB (default / two-branch) | `w = q` | 28.9 % / 17.9 % | **No** |
| resolution trend | `O(1/N)` for a discretisation error | constant ≈0.37-cell offset | **No** (wrong signature) |
| mass drift | 0 | ≤4e-6 | **Yes** |
| capability claim for `KPI-LBM-VERIFIED-CAPS` | — | — | **WITHHELD** |

## 5. What this buys the next mission

1. A staircase-free, closed-form-referenced harness for the sub-cell wall question
   (`examples/ibb_subcell_probe.py`, ~100 s single core, deterministic).
2. A *calibration gate* that must be passed before any operator verdict is quotable:
   at `q0=q1=0.5` the production `halfway` operator must return the closed form to
   `<1 %`. Until then, any `w(q)` fit is measuring the harness.
3. One concrete unexplained anomaly (§3d, `alpha`-insensitivity) that is cheap to settle
   and would either close or reopen the α-family question of
   `xval/XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME`.

## 6. Reproduce

```bash
PYTHONPATH=src python examples/ibb_subcell_probe.py        # writes outputs/ibb_subcell/
python -m pytest tests/test_ibb_subcell_probe.py -q         # artifact + algebra invariants
```

## 7. Limitations

* 2D, straight walls only; `τ = 0.8` only (`ν = 0.1`); one initial condition.
* No resolution study beyond `N = 8/16` (32 was cut by the wall-clock budget).
* The reference is a closed form, but the *forcing convention* of the harness is not yet
  anchored to the repo's verified body-force benchmark — this is the blocking gap.
* No conclusion is drawn about the IBB operator, the α-family, or the staircase debt.