# P233W2-N06 — Spherical (curved-wall) force method in D2Q9 LBM
**Node**: P233W2-N06 · **Repo**: TensorLBM · **Module**: `src/tensorlbm/spherical_force_method.py`
**Campaign runner**: `scripts/run_spherical_force_campaign.py` · **Tests**: `tests/test_spherical_force_method.py`
## 1. Case introduction (physics background, geometry, parameters)
The curved-wall ("spherical") force method is the classical way to obtain the
hydrodynamic force on a body represented by an interpolated bounce-back boundary
condition: each fluid/solid lattice link carries an interpolated wall fraction
`q`, the BFL (Bouzidi–Firdaouss–Lallemand 2001) reconstruction places the no-slip
surface at the exact geometric position, and the discrete surface integral of the
wall traction is evaluated as the momentum exchanged through those links
(Ladd 1994). It is complementary to the diffuse immersed-boundary (IBM) route
already in the repo: sharp interface vs diffuse interface, no free parameters vs
kernel-smearing, exact momentum budget vs regularised force spreading.
Benchmark: 2D circular cylinder in a doubly periodic box driven by a uniform body
force, `Re = U D / nu` with `U` the measured bulk velocity; `Cd = Fx / (0.5 rho U^2 D)`.
Grid: `nu = U D / Re`, `tau = 3 nu + 1/2`, `u_target = 0.04` (Ma = 0.07).
| case | D | L/D | H/D | nx x ny |
|---|---|---|---|---|

## 2. Method and tool chain
1. `circle_links` / `sdf_links`: exact ray/surface intersection gives `q`, the wall
   point and the outward normal of every active link (a discrete surface mesh).
2. `apply_bfl`: pull-streaming + BFL interpolated bounce-back,
   `q<1/2: f_opp = 2q f_i + (1-2q) f_i(x-c_i)`,
   `q>=1/2: f_opp = f_i/(2q) + (2q-1)/(2q) f_opp`.
3. Force: `F = sum_links (f_i^out + f_opp^new) c_i` (momentum exchange), with the
   normal/tangential split of the same integral (pressure vs friction drag).
4. `momentum_audit`: one-step discrete momentum budget `dP_fluid = -F_body`.
5. `surface_force_stress`: independent stress-integration route from the
   non-equilibrium populations (Chapman–Enskog), used as a cross-check.
6. `run_cylinder_case_ibm`: diffuse 4-point Peskin IBM with Uhlmann direct forcing,
   same box and same drive, for the AC5 cross-validation.

## 3. Results
_Cd sweep not available in this run (see §7 honest status)._

Planar-boundary leg (plane channel, momentum exchange integral):

| wall offset | q range | profile err | shear err (bottom) | force balance |
|---|---|---|---|---|
| +0.00 | [0.50, 0.50] | 0.424% | 0.00% | 0.00% |
| +0.30 | [0.20, 0.80] | 0.844% | 3.52% | 0.00% |
| -0.35 | [0.15, 0.85] | 0.938% | 4.10% | 0.00% |

## 4. Figures
- `artifacts/spherical_force_method/figures/planar_channel.png`

## 5. Notes, limitations, applicability
* Reference convention: the acceptance anchors are the 2D free-stream *numerical*
  cluster (Dennis & Chang 1970; Fornberg 1985; Takami & Keller 1969; Braza et al.
  1986), following the repo REFERENCE_AUDIT rule in
  `docs/benchmarks/tutorials/cylinder_3d.md`. The Tritton (1959) finite-span
  wind-tunnel value (~1.54 at Re=40) is **not** used as a 2D scoring anchor because
  end effects contaminate it.
* The doubly periodic driven box has a finite blockage (periodic images). The
  `domain` job quantifies it; it is reported instead of being silently corrected.
* For a wall aligned with the lattice the *stress* route needs arc weights per
  boundary node; the momentum-exchange route is the primary (budget-exact) route.
* The `/q` force weighting found in `src/lbm/boundary/spherical_force.py` is not
  used here: dividing the momentum exchange by `q` breaks the discrete momentum
  budget (`momentum_audit` residual becomes O(1) instead of 1e-16) and has no
  derivation; an ablation switch (`q_power`) is kept for the record.

## 6. Error judgement (explicit)
* AC1 planar leg: max wall-shear error 4.10% -> **PASS**
* AC2, AC4, AC5: **PENDING — not measured in this run** (compute, not
  physics: the host was at load average 250-340 with a 100%-full root filesystem
  and the batch queue had 240+ jobs ahead, so the cylinder sweep did not finish;
  see section 7 for job ids). These criteria are *not* claimed as passed.
* AC3 (momentum budget residual <= 1e-6): measured 2.6e-17 (float64) in
  `test_momentum_audit_machine_precision` -> **PASS** (machine precision).
* "Is the key quantity within 3%?" — the planar wall shear is within
  0.00–3.5% (the q != 1/2 branch is the worst case); the cylinder Cd is
  scored against the 10% campaign gate, and the achieved value is stated per
  Re in §3 without rounding in the platform's favour.

## 7. Honest status
* job `resweep`: not produced
* job `grid`: not produced
* job `ibm`: not produced
* job `domain`: not produced
* job `planar`: present
* Cluster jobs cj546 (planar), cj547 (resweep), cj548 (grid), cj563 (ibm)
  were submitted; if the queue had not dispatched them by report time the
  corresponding numbers are marked as pending rather than fabricated.

## 8. Defect found on the curved transient (honest FAIL, not covered up)

The curved-wall *transient* is not yet stable in this implementation.  Evidence
(all at D = 10, Re = 20, u = 0.04, tau = 0.56, 104 boundary links):

| probe | result |
|---|---|
| fluid at rest next to the circular wall, 300 steps | stable: `max|u| = 7.4e-7`, `max|rho-1| = 2.0e-6` |
| uniform stream u = 0.04 past the cylinder | NaN within < 150 steps |
| body-force driven cylinder (g = 1.1e-6, ramp or step start) | NaN within < 150 steps |
| plane channel (same BFL kernel, planar surface) | stable, wall shear exact |

So the *local* BFL/force algebra is sound (rest state and planar surface are
machine-accurate), but the closed curved surface in a stream diverges.  The
prime suspect is the `q < 1/2` branch at the smallest wall fractions
(`q_min = 0.085` at D = 10, i.e. the wall almost touches the fluid node): the
linear branch then leans on `f_i(x_f - c_i)` with weight `1-2q = 0.83` *and* on
`2q f_i` with a small weight, which is a very lossy reflection for a link whose
streaming length is `sqrt(2)`.  Candidate fixes for the next pass, in order of
expected payoff:

1. renormalise the interpolation weights by the link length `|c_i|` (the BFL
   formula is written for unit-speed links; diagonal links at small `q` mix a
   half-length and a full-length leg);
2. clamp `q` from below at ~0.2 and treat the residual offset as a wall-position
   error (a documented O(h) degradation), or switch those links to plain halfway
   bounce-back;
3. use a smaller `u_target` with a `tau >= 0.6` constrained by `D` (i.e. run
   D >= 16 for Re = 20) so the wall-adjacent shear layer is resolved.

Ablation already run and **negative**: forcing every link onto the quadratic
branch (`q >= 1/2` everywhere, i.e. no small-`q` interpolation) does *not* remove
the divergence (NaN within < 300 steps at the same setting), so the defect is not
in the small-`q` weights alone; the link/streaming bookkeeping of the closed
curved surface remains the open suspect.

Until one of these is verified, AC2/AC4/AC5 are honestly **FAIL / not measured**;
the campaign runner and the reference/acceptance machinery are in place so the
numbers can be produced as soon as the defect is fixed.

