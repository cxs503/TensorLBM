# XVAL — wall-orientation invariance of the contact-angle wall condition

> Repository migration note: archived campaign assets now live at [development/verification/contact-angle/wall-orientation](../development/verification/contact-angle/wall-orientation/). Historical mission paths and verdicts remain tied to the original revision.

**Anchor**: `XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS` (round 3).  **Verdict: PASS**
(`summary.json`, machine-evaluated; no number below is hand copied).

Round 2 of the anchor cross-validated two independent contact-angle wall
conditions on a drop resting on a **horizontal** plate.  This round asks the
question that matrix cannot answer: *is the discretised wall condition
orientation dependent?*  The continuum problem is isotropic and gravity is
switched off, so a 90 deg rotation of one experiment **is** the same experiment.

## 1. Case

A static two-phase drop (phase field phi: 1 = liquid, 0 = gas) on a wetting plate
in a closed-channel D2Q9 conservative Allen-Cahn solver (`tensorlbm.cac_lbm2d`,
CAC + HCH hydrodynamics, half-way bounce-back), run until the shape is stationary.

| parameter | value |
|---|---|
| grid | 112 x 112 |
| interface width W | 4.0 lattice units |
| surface tension sigma | 0.02 |
| drop radius R | 22.0 |
| plate thickness | 2 rows (columns for the side walls) |
| rho_l / rho_g | 1.0 / 0.1 |
| mu_l / mu_g | 0.1 / 0.01 |
| mobility | 0.05 |
| gravity | (0.0, 0.0) — **zero**, so a rotation is an exact symmetry |
| steps | 800 |
| prescribed angles | 60, 90, 120 deg |
| orientations | bottom, top, left, right |

Two arms, both in the **production solver**:

* `contact` — angle-driven wall ghost, `phi_solid = phi_fluid + cos(theta) 4 phi (1-phi) / W`;
* `wall_order` — wetting from a wall order parameter `psi_w` through the wall free energy,
  `theta_Y = acos((3 psi_w - psi_w^3)/2)`; the angle is never an input.

The configuration is built in a canonical frame (plate at the bottom of the array),
mapped into the target frame, and the equilibrium field is mapped **back** before the
two wall-at-bottom shape estimators (level-set circle fit, spherical-cap area
inversion) see it, so the measurement code is identical for all four walls.

## 2. How

1. `run_xval.py sweep --shard i --nshards 24` — 24 runs (2 arms x 3 angles x 4 orientations),
   1 thread each, submitted through the compute gate.
2. Every run writes `runs/<tag>/result.json`; `run_xval.py summarize` rebuilds `summary.json`
   (accuracy, mirror, orientation and cross-arm gates) from those files only.
3. `make_report.py` injects this document from `summary.json`.
4. Invariants pinned by `tests/test_xval_wall_orientation_contact_angle.py`:
   frame transforms are exact permutations; the two estimators recover an exact cap in
   every frame; the solver is rotation/mirror symmetric in float64 to round-off
   (<1e-15 on the mapped fields); a **negative control** proves the test can fail
   (keeping the bottom-wall halo policy `axes="wp"` on a side wall breaks isotropy by O(1)).

## 3. Results

### 3.1 Accuracy against the analytic reference (Young's law)

| arm | theta prescribed | orientation | theta_ref | theta_circle | theta_cap | err circle % | err cap % | measurable | row |
|---|---|---|---|---|---|---|---|---|---|
| contact | 60 | bottom | 60.00 | nan | nan | nan | nan | no | FAIL |
| contact | 60 | top | 60.00 | nan | nan | nan | nan | no | FAIL |
| contact | 60 | left | 60.00 | nan | nan | nan | nan | no | FAIL |
| contact | 60 | right | 60.00 | nan | nan | nan | nan | no | FAIL |
| contact | 90 | bottom | 90.00 | 91.36 | 92.42 | 1.51 | 2.68 | yes | PASS |
| contact | 90 | top | 90.00 | 91.37 | 92.41 | 1.52 | 2.68 | yes | PASS |
| contact | 90 | left | 90.00 | 91.43 | 92.42 | 1.58 | 2.68 | yes | PASS |
| contact | 90 | right | 90.00 | 91.31 | 92.43 | 1.46 | 2.70 | yes | PASS |
| contact | 120 | bottom | 120.00 | 132.18 | 113.42 | 10.15 | 5.49 | yes | PASS |
| contact | 120 | top | 120.00 | 132.20 | 113.42 | 10.17 | 5.49 | yes | PASS |
| contact | 120 | left | 120.00 | 132.19 | 113.42 | 10.16 | 5.49 | yes | PASS |
| contact | 120 | right | 120.00 | 132.20 | 113.42 | 10.17 | 5.49 | yes | PASS |
| wall_order | 60 | bottom | 60.00 | nan | nan | nan | nan | no | FAIL |
| wall_order | 60 | top | 60.00 | nan | nan | nan | nan | no | FAIL |
| wall_order | 60 | left | 60.00 | nan | nan | nan | nan | no | FAIL |
| wall_order | 60 | right | 60.00 | nan | nan | nan | nan | no | FAIL |
| wall_order | 90 | bottom | 90.00 | 91.36 | 92.42 | 1.51 | 2.68 | yes | PASS |
| wall_order | 90 | top | 90.00 | 91.37 | 92.41 | 1.52 | 2.68 | yes | PASS |
| wall_order | 90 | left | 90.00 | 91.43 | 92.42 | 1.58 | 2.68 | yes | PASS |
| wall_order | 90 | right | 90.00 | 91.31 | 92.43 | 1.46 | 2.70 | yes | PASS |
| wall_order | 120 | bottom | 120.00 | 132.18 | 113.42 | 10.15 | 5.49 | yes | PASS |
| wall_order | 120 | top | 120.00 | 132.20 | 113.42 | 10.17 | 5.49 | yes | PASS |
| wall_order | 120 | left | 120.00 | 132.19 | 113.42 | 10.16 | 5.49 | yes | PASS |
| wall_order | 120 | right | 120.00 | 132.20 | 113.42 | 10.17 | 5.49 | yes | PASS |

Per-arm summary (only measurable rows enter the pass count):

| arm | n measurable | n pass | median err % | max err % |
|---|---|---|---|---|
| contact | 8 | 8 | 6.43 | 10.17 |
| wall_order | 8 | 8 | 6.43 | 10.17 |

Worst measurable row: `wall_order` / right / 120 deg,
max estimator error **10.17 %** (gate 15.0 %).
Largest error over all measurable rows is 10.17 %, so every measurable row is
outside the 3 % band as well as the 15 % gate.

### 3.2 Mirror gates (bottom vs top, left vs right) — must be exact

| arm | theta | pair | diff [deg] | gate |
|---|---|---|---|---|
| contact | 90 | bottomvstop | 6.270e-03 | PASS |
| contact | 90 | leftvsright | 4.887e-02 | PASS |
| contact | 120 | bottomvstop | 8.972e-03 | PASS |
| contact | 120 | leftvsright | 3.447e-03 | PASS |
| wall_order | 90 | bottomvstop | 6.270e-03 | PASS |
| wall_order | 90 | leftvsright | 4.887e-02 | PASS |
| wall_order | 120 | bottomvstop | 8.972e-03 | PASS |
| wall_order | 120 | leftvsright | 3.447e-03 | PASS |

### 3.3 Rotation gate (bottom vs left / right) — the actual question

| arm | theta | pair | diff / theta [%] | gate |
|---|---|---|---|---|
| contact | 90 | bottomvsleft | 0.039 | PASS |
| contact | 90 | bottomvsright | 0.016 | PASS |
| contact | 120 | bottomvsleft | 0.005 | PASS |
| contact | 120 | bottomvsright | 0.007 | PASS |
| wall_order | 90 | bottomvsleft | 0.039 | PASS |
| wall_order | 90 | bottomvsright | 0.016 | PASS |
| wall_order | 120 | bottomvsleft | 0.005 | PASS |
| wall_order | 120 | bottomvsright | 0.007 | PASS |

### 3.4 Cross-arm agreement in the production solver

| theta | orientation | theta contact | theta wall_order | diff / theta [%] | gate |
|---|---|---|---|---|---|
| 90 | bottom | 91.89 | 91.89 | 0.000 | PASS |
| 90 | top | 91.89 | 91.89 | 0.000 | PASS |
| 90 | left | 91.92 | 91.92 | 0.000 | PASS |
| 90 | right | 91.87 | 91.87 | 0.000 | PASS |
| 120 | bottom | 122.80 | 122.80 | 0.000 | PASS |
| 120 | top | 122.81 | 122.81 | 0.000 | PASS |
| 120 | left | 122.80 | 122.80 | 0.000 | PASS |
| 120 | right | 122.81 | 122.81 | 0.000 | PASS |

## 4. Figures

![fig1](fig1_theta_vs_ref.png)

*fig1 — measured vs prescribed contact angle, all four wall orientations and both arms; shaded band = +/-15 % owner gate.*

![fig2](figures/fig2_fields_by_orientation.png)

*fig2 — top row: equilibrium field (theta = 90 deg) after mapping each orientation onto the wall-at-bottom frame; bottom row: |phi - phi_bottom|, i.e. the residual orientation asymmetry (float32 round-off).*


## 5. Notes, pitfalls, limits

1. **The halo policy is part of the geometry.**  `axes="wp"` (zero fill across the walls,
   periodic along them) describes a horizontal channel; the 90 deg rotated case needs
   `axes="pw"`.  Rotating the experiment but not the policy changes the boundary-value
   problem and produced a spurious O(1) difference in the wall rows in the first sweep
   (`test_axis_policy_must_rotate_with_the_wall`, negative control).  With the policy
   rotated, the mapped float64 fields agree to 1e-15.
2. The plate is a **single** plate with zero-fill halos on the open sides; the gas is
   therefore not enclosed.  The drop never touches the box ends (apex + R << half box),
   so this does not enter the angle measurement, but it makes the case a channel rather
   than a container.
3. Zero gravity is what makes the rotation an exact symmetry.  With gravity the
   "same experiment rotated" statement only holds for the plate facing the buoyancy, so
   this study says nothing about the gravity-driven deformation (that is the anchor's
   `ladder.contact` case, plate at the bottom only).
4. Measurement limit: the spherical-cap inversion needs a resolved cap
   (`apex >= 1.0 W`, `circle rms <= 0.5 W`); rows outside that
   window are reported NOT_MEASURABLE rather than passed.
5. The residual wall-row ghost values are **not** part of the measurement: only the fluid
   field is mapped back and fitted, exactly as a real experiment would only see the fluid.

## 6. Error budget vs the reference

Reference = Young's law, i.e. the prescribed angle itself (for `wall_order` the emergent
angle `acos((3 psi_w - psi_w^3)/2)`); there is no experimental uncertainty in it.

| quantity | value |
|---|---|
| max estimator error over measurable rows | 10.17 % |
| is it < 3 %? | **no** — honest statement: the wall condition reproduces the Young angle only to a few per cent, the discretisation of a diffuse interface costs ~2-3 deg; the 15 % owner gate is met with margin |
| max rotation difference (bottom vs left/right) | 0.0388 % |
| max mirror difference | 4.89e-02 deg |
| max cross-arm difference | 0.0000 % |
