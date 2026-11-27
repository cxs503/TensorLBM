# XVAL-LBM-PESKIN-TENSOR-PRODUCT-STENCIL-MDF — round 2 (partial, honest): solid-interior treatment

**Anchor** `XVAL-LBM-PESKIN-TENSOR-PRODUCT-STENCIL-MDF` (round 2) · repo `TensorLBM` · mission 20490.

## Status: hypothesis made testable; the decisive matrix did NOT complete

Round 1 of this slug closed the **operator** level (kernel/stencil identity, MDF
fixed point, achieved no-slip) and left an honest FAIL end to end: the cylinder
at Re = 20 gave `Cd = 25.1…25.6` against the literature `2.05` (Dennis & Chang
1970). Round 1 blamed the shared driver's *solid-interior penalty* force and
deferred the fix. This round makes that hypothesis testable and ships the fix
candidate, but does **not** produce a valid drag number.

## Delivered

1. `src/tensorlbm/ibm_force_transfer.py` — new shipped helper
   `solid_interior_reset(f, mask, feq_at_rest, rho=None)`: the *hard* solid-node
   treatment — solid cells are overwritten with the equilibrium of a fluid at
   rest instead of being penalised by a booked body force. Rationale (in its
   docstring): a booked interior force is a real momentum source/sink on the
   Eulerian grid that must be balanced by an extra marker force, and a single
   pass never removes the whole interior deficit; an overwrite keeps
   `F_body = −Σ_k F_k` (`momentum_audit`) exact so the markers are the only drag
   carrier. Functional check, float64, 4×5 lattice, mask 2×3:
   outside cells bit-identical (`0.0e+00`),
   `max|u_x|` inside after reset `0.0e+00`,
   outside `u_x` preserved (`0.0e+00`),
   interior density `1.000000000000000`.
2. `xval/XVAL-LBM-PESKIN-TENSOR-PRODUCT-STENCIL-MDF/r2_interior/run_l4_interior.py`
   — arm-parametrised L4 driver (`penalty` / `none` / `hard` / `hard_rho1`) with a
   separate Eulerian force budget (surface vs interior), interface slip and
   solid-permeability traces, and a resolution/Reynolds ladder stage.
3. `make_figures.py` / `make_report.py` — artefacts-driven figure and report
   generators (numbers are read back from JSON, never hand-typed).

## What did not run, and why (honest FAIL — not passed)

* Screen `cj2740` is **numerically invalid**. The marker force array is stacked
  `(f_y, f_x)` because the grid is stacked `(u_y, u_x)`; the version that ran read
  component 0 — the *lift* — as the drag. It reported `Cd = −0.0025 ± 0.001`
  with a 0.1 % interface slip for **all four** arms: a plausible-looking pass that
  was pure bookkeeping. Fixed in the driver (explicit `(f_x, f_y)` re-order plus a
  lift trace) and documented in the code.
* The corrected re-run `cj2753` (`--stage full`) exited with `unknown stage full`:
  the combined stage was not present in the snapshot the scheduler staged, so only
  the old screen ran again. **No valid end-to-end `Cd` was obtained.**
* Therefore `Cd` vs literature is **WITHHELD** — explicitly *not* "<3 %" and *not*
  "<15 %". Round 1's honest FAIL remains **unresolved** by this mission.

## Why there is only one figure

It shows the trap itself (drag history read from the lift component). A
"variant comparison" plot built from the invalid screen would have been
misleading, so it is deliberately absent.

## Concrete next step

```
cd xval/XVAL-LBM-PESKIN-TENSOR-PRODUCT-STENCIL-MDF/r2_interior
python run_l4_interior.py --stage screen   # 4 arms, Re=20, D=8, h=0.5, 3000 steps (~5 min)
python run_l4_interior.py --stage ladder   # D=8/16 + Re=40 with the hard arm (~30 min)
python make_figures.py && python make_report.py
```
Acceptance: hard-interior `Cd(Re=20)` within 15 % of 2.05 and `Cd(Re=40)` within
15 % of 1.54, with `max|u|` inside the body < 2 % of U; the penalty arm is
expected to remain order-10× high.
