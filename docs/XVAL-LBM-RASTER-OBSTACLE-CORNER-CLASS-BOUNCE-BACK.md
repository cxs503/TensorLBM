# XVAL-LBM-RASTER-OBSTACLE-CORNER-CLASS-BOUNCE-BACK

**Owner precision standard:** |error| < 15 %.

* **Primary metric (permeability K): PASS** — largest disagreement 2.04 % (case C3);
  all 5 completed cases are below the 15 % bar and the empty-channel control reproduces the discrete
  Poiseuille solution to 0.072 %.
* **Secondary metric (obstacle drag F_x): FAIL** — 17.80 %
  at worst (case C4).
  The eight spurious band-end links are only ~5.41 % of the
  bounced links, yet they carry a disproportionate share of the momentum exchange, so a local force metric is
  far more sensitive to the corner-class over-approximation than the volume-averaged permeability. This is the
  measured reason not to transplant the table (the drag is what a ship/ice-load user reads off the solver).

## 1. Problem and reference source

Anchor tag `raster-obstacle-corner-class-bounce-back` of the platform OSS inventory
(`docs/oss_inventory/OSS-INV-Fingering_dynamics.md`, slug `NEW:CAP-LBM-POROUS-OBSTACLE-STAGGER`),
repository **ebinan92/Fingering_dynamics** (checkout `150a573`, read-only copy at `/tmp/Fingering_dynamics`).
The reference implementation turns *arbitrary raster geometry* into a no-slip wall by

* rasterising rectangles into a boolean mask plus a **corner list** of bounding boxes
  (`lattice_boltzmann/create_block.py` `getRectangleblock`/`setblock`, `create_block.py:395`);
* painting **eight barrier classes** (N/S/E/W fluid bands hugging each rectangle and the four
  diagonal corner cells) and applying a **hand-written reflection table** per class
  (`lattice_boltzmann/bounce_back.py:25-86`, `halfway_bounceback_rec`);
* streaming by periodic `np.roll` (`lattice_boltzmann/fingering.py:393`) with the pre-stream copy
  `f_behind = copy.deepcopy(f)` (`fingering.py:551-560`) as the reflection source;
* channel walls by `bottom_top_wall` (`fingering.py:432`: row 0 `f2<-f4, f5<-f7, f6<-f8`,
  row `ny-1` `f4<-f2, f7<-f5, f8<-f6`).

The repository has **no LICENSE file**, so nothing is vendored: the index arithmetic above was
re-derived from the cited lines (`xval/.../raster_case.py`).

**Question.** Would transplanting that operator change TensorLBM's raster-obstacle physics by more
than the owner's 15 % bar, compared with TensorLBM's own per-direction closed-link mask
(`src/tensorlbm/bitflag_neighbour_mask.py`: bit `i-1` set iff the upstream cell `x - c_i` is solid)?

## 2. Method

Both arms share everything except the bounce-back link set: same raster mask (same rectangle corner
list), same D2Q9 BGK collision, same Guo second-order body force (single-phase, no free energy — the
anchor's phase-field physics was XVAL'd separately and produced no usable OSS number), same rest
seed, same step budget, same measurement code. The flow is a body-force driven channel, periodic in
x, with wall rows at `y = 0` and `y = ny-1` (half-way walls; the mask arm closes dirs 2/5/6 at the
bottom row and 4/7/8 at the top row, which is exactly the `bottom_top_wall` assignment).

Measured quantities: lattice permeability `K = nu * u_bar / g` (pore mean of `u_x`), momentum-exchange
force on the obstacles `F_x = sum over that arm's bounced obstacle links of 2 c_ix f_opp`, and the
field difference over fluid nodes. `predeclared.json` fixes the ladder, the thresholds, the stopping
rule and the analysis before the runs.

Operator-level (machine-checked, `tests/test_xval_raster_corner_class.py`, 10 tests): the table never
misses an exact solid link, adds exactly 8 links per rectangle, and one streaming step from the same
random population field differs **only** at those 8 sites, in the matching diagonal direction.

### Difference list, OSS vs this repo

| item | OSS `Fingering_dynamics` | TensorLBM `bitflag_neighbour_mask` |
|---|---|---|
| geometry input | rectangle corner list -> painted band/corner masks | raster mask -> per-direction closed-link bits |
| wall operator | hand-written 8-class table (148 links per 12x12 block) | exact closed-link mask (140 links per block) |
| band ends | diagonal bounce applied over the whole band, including the two end nodes whose diagonal upstream is **fluid** (8 spurious links/block) | diagonal bounced iff its own upstream cell is solid |
| storage / decision cost | 8 boolean masks, 9 in-place assignments per step | one `uint8` per node, branch-free interior, set-bit loop at walls |
| streaming | `np.roll` over the whole array (periodic in x **and** y) | index-table pull with periodic x and closed y rows |
| corner cells | four separate single-cell masks, diagonal only | bit pattern (identical link set) |
| walls | `bottom_top_wall`, applied to `f_behind[:,1:-1]` (interior columns only) | wall bits on every column (this run is periodic in x) |
| collision | non-Newtonian free-energy D2Q9 | single-phase BGK + Guo force |

**Porting trade-off.** The link-level structure of the OSS table is *correct* for the eight classes it
intends to describe (missed links = 0); what it cannot express
is "this particular diagonal link of the band end is open". TensorLBM's mask expresses it by
construction, so porting the table would be a regression in geometry fidelity at the band ends while
adding a second, hand-maintained data structure.

Link-level comparison (geometry only, predeclared enumeration):

| case | blocks | exact links | table links | spurious links | missed |
|---|---|---|---|---|---|
| C1 | 12 | 1680 | 1776 | 96 (8/block, 96 sites) | 0 |
| C5 | 12 | 3408 | 3504 | 96 (8/block, 96 sites) | 0 |

## 3. Parameters (all lattice units, dimensionless groups complete)

* domain 132 x 144 (C0-C4) and 264 x 288 (C5); 12 staggered blocks (9.09 % solid, staircase)
* `tau in {2.0, 1.5}` -> `nu = (tau-1/2)/3 in {0.5, 0.3333}`
* body force `g in {2.5e-4, 5e-4, 1e-3}` (C0 at `1e-6`), Guo second-order forcing
* derived pore Reynolds number `Re_b = u_bar b / nu` and Mach `Ma = u_bar/c_s`, `c_s = 1/sqrt(3)` — per case in the table
* resolution pair: `b = 12` vs `b = 24` at identical porosity/block count, `g` scaled by `1/b^2` so `Ma` matches

## 4. Results (reference vs got, error marked)

| arm | K_sim | K_analytic = ny^2/12 + 1/24 | err K | err u-profile | steps |
|---|---|---|---|---|---|
| oss | 1729.2912 | 1728.0417 | 0.072 % | 0.053 % | 40000 |
| mask | 1729.2912 | 1728.0417 | 0.072 % | 0.053 % | 40000 |

C0 anchor (no obstacles) is the control: both arms reproduce the discrete Poiseuille solution, so the
collision/forcing/measurement chain is sound before any obstacle difference is interpreted.

| case | blocks | b | tau | g | u_bar (mask) | Re_b | Ma | K_OSS | K_TLBM | err K | err F_x | field L2 | K/b^2 (OSS/TLBM) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 0 | 12 | 2 | 1e-06 | 3.4586e-03 | 0.083 | 0.0060 | 1729.291182 | 1729.291182 | **0.00 %** | n/a % | 0.00 % | 12.00897 / 12.00897 | PASS (<15 %) |
| C1 | 12 | 12 | 2 | 0.0005 | 4.1316e-02 | 0.992 | 0.0716 | 40.490344 | 41.316271 | **2.04 %** | 17.80 % | 1.88 % | 0.28118 / 0.28692 | PASS (<15 %) |
| C2 | 12 | 12 | 1.5 | 0.0005 | 5.5755e-02 | 2.007 | 0.0966 | 36.594167 | 37.170235 | **1.57 %** | 17.37 % | 1.44 % | 0.25413 / 0.25813 | PASS (<15 %) |
| C3 | 12 | 12 | 2 | 0.001 | 8.2612e-02 | 1.983 | 0.1431 | 40.479175 | 41.306205 | **2.04 %** | 17.77 % | 1.89 % | 0.28111 / 0.28685 | PASS (<15 %) |
| C4 | 12 | 12 | 2 | 0.00025 | 2.0659e-02 | 0.496 | 0.0358 | 40.492972 | 41.318610 | **2.04 %** | 17.80 % | 1.87 % | 0.28120 / 0.28693 | PASS (<15 %) |

Missing cases (not run/failed): C5.

`err K` = |K_TLBM - K_OSS| / K_OSS · 100 (primary), `err F_x` the same for the obstacle drag, `field L2`
the relative L2 of `u_x` over fluid nodes. Case C1 detailed difference: see `figures/fig1_fields.png`
(fields) and `figures/fig2_corner_links.png` (the 8 spurious links of one block).

## 5. Convergence and sensitivity

* `figures/fig3_convergence.png` (left): permeability history of both arms, C1 — both settle to their
  plateaus; the reported values are taken at the predeclared stopping rule (relative change < 1e-7 per
  1000 steps, or the fixed budget).
* stop reasons: C0: OSS 40000 / TLBM 40000 steps (converged_at 40000 / 40000), C1: OSS 40000 / TLBM 40000 steps (converged_at None / None), C2: OSS 40000 / TLBM 40000 steps (converged_at None / None), C3: OSS 40000 / TLBM 40000 steps (converged_at None / None), C4: OSS 23000 / TLBM 22000 steps (converged_at 23000 / 22000)
* sensitivity point 1 (viscosity / Re): C1 vs C2 (`tau = 2.0` vs `1.5`).
* sensitivity point 2 (forcing / Mach): C1 vs C3 vs C4 (`g = 5e-4 / 1e-3 / 2.5e-4`).
* sensitivity point 3 (lattice resolution at fixed porosity): C1 vs C5 (`b = 12` vs `24`).

## 6. Failure modes and artifacts

* the OSS band painting has **no out-of-range guard**: a rectangle touching a border writes a corner
  cell at index `-1`/`nx`, which NumPy wraps silently. Our geometry keeps a full block-width margin, so
  the effect is out of scope here — it is a real porting hazard for arbitrary raster geometry.
* the OSS wall routine is applied to `f_behind[:,1:-1]` because the main case uses the first/last column
  as Zou-He inlet/outlet; in a periodic-x comparison the routine must be applied to every column
  (documented deviation).
* the OSS arm carries no free energy here: only its bounce-back operator is under test.
* artifacts: `runs/<case>/fields.npz` (u_x, rho, mask, solid, history), `runs/<case>/metrics.json`,
  `results.json`, `runs/summary.csv`, `figures/*.png`, `predeclared.json`.

**Honest scope statement.** The disagreement measured here is the disagreement of the *wall operator*
on one geometry family (staircase rectangle arrays at 9.09 % solid). It is not a statement about the
OSS phase-field physics, and the drag error is a derived quantity whose sensitivity to the link set is
geometry-dependent.
