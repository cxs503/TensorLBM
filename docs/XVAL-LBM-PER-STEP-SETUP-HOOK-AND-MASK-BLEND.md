# XVAL-LBM-PER-STEP-SETUP-HOOK-AND-MASK-BLEND

**Round 1 · open-source cross-validation (XVAL)** · mission 18130 · generated 2026-11-13T12:17:57Z · commit `7d6cd8c7` by `benchmarks/xval_per_step_hook_mask_blend/run_xval.py` (every number below is read from `artifacts/xval_per_step_hook_mask_blend/result.json`).

**Verdict: PASS** — owner gate (15 %): PASS; internal target (3 %): PASS. The upstream-faithful moving-lid arm fails by design (measured flow 0.000 % of the imposed wall speed) — that failure *is* the evidence for the one porting change made.

## 1. Problem and reference

The OSS implementation organises a lattice-Boltzmann run in two idioms that TensorLBM did not have:

1. a **driver with per-step setup hooks** — `Domain.Solve(sess, Tf, initialize_step, setup_step, save_step, save_interval)` runs `initialize` once, then per step `save (on the save tick) → setup_step → collide → stream`. Case-specific physics (moving lid, inlet value, output cadence) is injected as a callback instead of being written into the solver.
2. a **mask-blend boundary condition** — `Domain.CollideSC` applies the wall as one tensor expression over a float mask field, `f = f_no_boundary + f_boundary` with `f_no_boundary = f*(1-boundary)` and `f_boundary = conv1x1(f*boundary, BOUNCE)`, so an arbitrary geometry costs no branching code.

Machine-extracted provenance (no hand transcription):

| item | value |
|---|---|
| repository | `loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow` |
| tarball | `https://codeload.github.com/loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow/tar.gz/refs/heads/master` |
| tarball sha256 | `7e4e3459730aecd57daef1cc5695e10368424b497a23849b3c10d095c4e23f10` |
| `LatFlow/Domain.py` sha256 | `d0c21f76df0e79ef511fc40581586b6a0a1bfff4dba1452fdbecd68525475522` |
| `lid_cavity.py` sha256 | `89ec666967e820eefa14c55b5162d24f09c4428faa135a10eeb16608f31a2026` |
| checkout used at run time | `/tmp/oss_lbm/Lattice-Boltzmann-fluid-flow-in-Tensorflow-master` |
| extraction | extracted verbatim from the checkout; per-file sha256 recorded |
| matches previously recorded provenance | True |

Upstream text the port is measured against (`artifacts/xval_per_step_hook_mask_blend/oss_ref/evidence.json`):

| idiom | file:line | text |
|---|---|---|
| `solve_signature` | `LatFlow/Domain.py:225` | `def Solve(self, sess, Tf, initialize_step, setup_step, save_step, save_interval):` |
| `blend_line_split` | `LatFlow/Domain.py:131` | `f_no_boundary = tf.multiply(f, (1.0-self.boundary))` |
| `blend_line_sum` | `LatFlow/Domain.py:132` | `f = f_no_boundary + f_boundary` |
| `bounce_piece` | `LatFlow/Domain.py:101` | `f_boundary = simple_conv(f_boundary, self.Op)` |
| `mask_constant` | `LatFlow/Domain.py:60` | `self.boundary = tf.constant(boundary)` |
| `velocity_recompute` | `LatFlow/Domain.py:184` | `Vel = simple_conv(f_pad, self.C)` |
| `lid_setup_signature` | `lid_cavity.py:44` | `def lid_setup_step(domain, value=0.001):` |

`Domain.Solve` verbatim, `LatFlow/Domain.py:225`:

```python
def Solve(self, sess, Tf, initialize_step, setup_step, save_step, save_interval):
    # make steps
    assign_step = self.Initialize()
    stream_step = self.StreamSC() 
    collide_step = self.CollideSC()

    # run solver
    sess.run(assign_step)
    sess.run(initialize_step)
    sess.run(stream_step)
    num_steps = int(Tf/self.dt)
    for i in tqdm(xrange(num_steps)):
      if int(self.time/save_interval) > int((self.time-self.dt)/save_interval):
        save_step(self, sess)
      sess.run(setup_step) 
      sess.run(collide_step)
```

The moving-lid hook, `lid_cavity.py:44`:

```python
def lid_setup_step(domain, value=0.001):
  # inputing top velocity 
  vel = domain.Vel[0]
  vel_out  = vel[:,1:]
  vel_edge = vel[:,:1]
  vel_edge = tf.split(vel_edge, 3, axis=3)
  vel_edge[0] = vel_edge[0]+value
  vel_edge = tf.concat(vel_edge, axis=3)
  vel = tf.concat([vel_edge,vel_out],axis=1)

  # make steps
  vel_step = domain.Vel[0].assign(vel)
  return vel_step

def lid_save(domain, sess):
  frame = sess.run(domain.Vel[0])
```

## 2. Method and implementation

The port is library code, `src/tensorlbm/step_hooks.py` (not bench-local):

* `solve_with_hooks(state, n_steps, operator=..., initialize_step=..., setup_step=..., save_step=..., save_interval=...)` — upstream `Domain.Solve`'s loop shape. `state` is a dict, hooks may mutate it in place; the save tick precedes the setup hook of that same step, so the record `HookLog(setup=[0..n-1], save=[0, k, 2k, ..., n])` is assertable.
* `compose_masked_boundaries(f_post, f_pre, layers)` — `f = f_post*(1 - M) + Σ_k m_k f_bc_k`, `M = Σ_k m_k`, with the partition enforced (`M ≤ 1`, else `ValueError`). Upstream has a single mask and a single BC family; the generalised form lets a bounce wall, a moving wall and an inlet/outlet coexist on arbitrary geometry.
* three layer families: `stationary_bounce_layer` (upstream `CollideSC`), `moving_bounce_layer` (`+ 6 w_i (c_i·u_wall)`, the one addition), `equilibrium_layer` (Dirichlet velocity by equilibrium substitution). A layer's velocity may be a callable of the hook state, which is the torch analogue of upstream's `setup_step` writing `domain.Vel`.

Arms (one changed variable per comparison):

| arm | transport | collision | loop | boundary |
|---|---|---|---|---|
| **A** explicit baseline | `tensorlbm.solver.stream` (gather) | `collide_bgk` | hand-written `for` | explicit index assignment `f[:, mask] = f_pre[opp][:, mask]`, moving-wall term added inline |
| **B** ported (chosen) | same | same | `solve_with_hooks` | `compose_masked_boundaries` over a disjoint mask partition (bounce / moving bounce / equilibrium) |
| **D** upstream-faithful lid | same | same | `solve_with_hooks` | single stationary bounce mask; lid speed only written into a `vel` variable by the setup hook and recomputed from `f` each step (upstream `lid_cavity.py`) |

Equivalence checks (implementation question) are separate from the physics checks: one step from an identical state, explicit BC vs mask blend, on (a) a single stationary mask, (b) a disjoint two-layer mask (blob + moving patch), plus a 500-step closed box for mass drift.

## 3. Non-dimensional parameter table

| quantity | symbol | value |
|---|---|---|
| relaxation time | tau | 0.800 |
| viscosity | nu = (tau-1/2)/3 | 0.1 |
| lid / wall velocity | U_wall | 0.02 |
| Mach number | Ma = sqrt(3) U | 0.034641 |
| channel height (lattice cells) | Ny | 32 |
| wall spacing, half-way convention | H = Ny - 2 | 30 |
| shear Reynolds number | Re_H = U H / nu | 6 |
| steps (Couette) | N_t | 11142 |
| lid ramp | ramp_steps | 200 |
| grid | nx | 8 |

Poiseuille arm:

| quantity | value |
|---|---|
| Ny | 24 |
| H = Ny - 2 | 22 |
| body force g = 8 nu u_peak / H^2 | 3.3057851e-05 |
| target / measured peak | 0.02 |
| Re_H = u_peak H / nu | 4.4 |
| steps | 16939 |

Composite channel (invariants only):

Ny = 16, nx = 32, tau = 0.800, u_in = 0.02, Ma = 0.034641, Re = 2.8, steps = 6000.

## 4. Reference vs got

### 4.1 Plane Couette (the mask blend *is* the physics)

| Ny | arm | profile L2 % | max err % | peak u vs U(H-0.5)/H % | slip lower % | slip lid % | 15% | 3% |
|---|---|---|---|---|---|---|---|---|
| 16 | A_explicit | 0.0003 | 0.0002 | 0.000 | 0.000 | 0.000 | ✓ | ✓ |
| 16 | B_hook_blend | 0.0003 | 0.0002 | 0.000 | 0.000 | 0.000 | ✓ | ✓ |
| 16 | D_upstream_lid | 100.0000 | 96.4286 | 100.000 | 0.000 | 100.000 | n/a (declared fail) | n/a |
| 24 | A_explicit | 0.0004 | 0.0003 | 0.000 | 0.000 | 0.000 | ✓ | ✓ |
| 24 | B_hook_blend | 0.0004 | 0.0003 | 0.000 | 0.000 | 0.000 | ✓ | ✓ |
| 24 | D_upstream_lid | 100.0000 | 97.7273 | 100.000 | 0.000 | 100.000 | n/a (declared fail) | n/a |
| 32 | A_explicit | 0.0004 | 0.0003 | 0.000 | 0.000 | 0.000 | ✓ | ✓ |
| 32 | B_hook_blend | 0.0004 | 0.0003 | 0.000 | 0.000 | 0.000 | ✓ | ✓ |
| 32 | D_upstream_lid | 100.0000 | 98.3333 | 100.000 | 0.000 | 100.000 | n/a (declared fail) | n/a |

### 4.2 Plane Poiseuille (curvature + body force, two mask layers)

| Ny | arm | profile L2 % | peak err % | <15% | <3% |
|---|---|---|---|---|---|
| 16 | A_explicit | 0.196 | 0.144 | ✓ | ✓ |
| 16 | B_hook_blend | 0.196 | 0.144 | ✓ | ✓ |
| 24 | A_explicit | 0.079 | 0.058 | ✓ | ✓ |
| 24 | B_hook_blend | 0.079 | 0.058 | ✓ | ✓ |

Population-state digests, arm A vs arm B: couette Ny=16: identical, couette Ny=24: identical, couette Ny=32: identical, poiseuille Ny=16: identical, poiseuille Ny=24: identical. The two arms therefore do not merely agree to a tolerance — they produce the same float64 state (the moving-wall momentum term is applied to the identical pre-collision populations).

### 4.3 Operator equivalence and invariants (arbitrary geometry)

| check | value | verdict |
|---|---|---|
| mask-blend vs explicit index BC, one step, disjoint blob + patch masks (max rel) | 0 | ✓ |
| upstream 1x1 bounce convolution (`streaming_conv.bounce_blend`) vs explicit (max rel) | 0 | ✓ |
| closed box, 500 steps: mass drift (rel) | 3.37e-14 | ✓ |
| wall-cell speed / peak |u| (diagnostic only, see §6.6) | 0.929 | n/a |
| composite channel: |Q_out - Q_in| / Q_in | 0.000 % | ✓ |
| composite channel: normal velocity in the wall-adjacent fluid cell / peak |u| | 9.429 % | ✗ |

The composite channel has no independent reference solution (developing flow, first-order equilibrium inlet): only the invariants above are claimed, no accuracy number.

## 5. Convergence and parameter sensitivity

| axis | configurations | result | reading |
|---|---|---|---|
| grid, Couette | Ny ∈ [16, 24, 32] | Ny=16: 0.000 %, Ny=24: 0.000 %, Ny=32: 0.000 % | linear analytic profile, so the residual is the wall-location convention, not a discretisation error |
| grid, Poiseuille | Ny ∈ [16, 24] | Ny=16: 0.196 %, Ny=24: 0.079 % | decreasing with resolution: second-order curvature error of half-way bounce |
| wall-position convention | half-way (H = Ny-2) vs node walls (H = Ny-1) | 4.545 % vs 0.000 % | the node convention is wrong by more than the 3 % target: the wall location must be validated before any accuracy statement, exactly like Cb/L/B for a hull |
| hook cadence | save_interval ∈ {50, 500}, identity setup | final state sha256 equal: True | hook plumbing is physics-neutral |
| lid onset | ramp_steps ∈ {1, 200} | L2 delta 0.000 pp (0.000 % vs 0.000 %) | the transient can be shaped entirely from the setup hook, the steady state is independent of it |

Predeclared stopping rule: step counts are fixed before the run from the analytic viscous time scale: Couette steps = 12*H^2/(pi^2*nu) + 200 (12 start-up e-folding times of the exp(-pi^2 nu t/H^2) transient, residual <1e-5), Poiseuille steps = 3.5*H^2/nu (viscous relaxation), closed-box invariance 500 steps. No arm is extended or truncated after looking at a KPI.

## 6. Failure modes, traps and artefacts

1. **The mask blend alone cannot move a wall.** Upstream's reflected population carries no wall momentum, so a lid can only be *declared*. Arm D reproduces upstream's mechanism: the setup hook writes the lid speed into the velocity variable and the step recomputes that variable from the populations, so the hook changes 0 of 2304 population entries and the measured flow is 0.0000 % of the imposed wall speed (profile L2 100.000 % against the analytic line). The port adds `6 w_i (c_i·u_wall)` and the same configuration reaches 0.000 %.
2. **Library weight tables are float32.** `d2q9.W` is `torch.float32`; used directly in a float64 field the moving-wall momentum carries the float32 rounding of `1/9` (~7.5e-9 relative). `tables_for()` therefore switches to the exact weights (`d2q9.W_EXACT`, `d3q19.W_EXACT64`) for float64 fields — without that switch the bit-wise equivalence check fails at 1e-8 and looks like a formulation error.
3. **Overlapping masks are a silent double-BC.** Upstream has one mask so this cannot happen; with a layer list it can, so `compose_masked_boundaries` raises unless `strict=False`. The composite-channel case selects wall cells out of the inlet/outlet columns for that reason.
4. **The save tick is defined differently.** Upstream saves when `int(t/save_interval)` crosses an integer (float time comparison); the port uses `i % save_interval == 0` plus one final save, which is exact for integer `dt`. Same cadence, different off-by-one at the tail.
5. **First-order inlet.** `equilibrium_layer` substitutes a full equilibrium on the boundary cell (no Zou–He reconstruction), so the composite channel's accuracy is first order — which is why no accuracy number is claimed for it.
6. **A wall cell is not a zero-velocity cell.** Reading the no-slip condition off the macroscopic velocity carried by the *masked* cells is wrong: bounce-back enforces no-slip through the population exchange, and those cells keep a non-zero |u| (92.9 % of the peak in the closed box). The slip KPIs in §4.1 are therefore measured by extrapolating the *fluid* profile to the wall location, which is the only place the boundary condition is actually defined.

Artefacts:

| path | content |
|---|---|
| `artifacts/xval_per_step_hook_mask_blend/result.json` | every scalar KPI, verdict and predeclared design |
| `artifacts/xval_per_step_hook_mask_blend/table.csv` | arm × configuration table with profile errors and state digests |
| `artifacts/xval_per_step_hook_mask_blend/profiles.npz` | velocity profiles and the composite-channel field |
| `artifacts/xval_per_step_hook_mask_blend/figures/*.png` | the five figures below |
| `artifacts/xval_per_step_hook_mask_blend/oss_ref/evidence.json` | machine-extracted upstream snippets + sha256 |
| `src/tensorlbm/step_hooks.py`, `tests/test_step_hooks.py` | the port and its 14 unit tests |

Timing: 1.193 ms/step (hand-written loop) vs 1.027 ms/step (hook + mask blend), -13.9 % overhead; whole benchmark 176.9 s.

## 7. Difference list: TensorLBM vs OSS upstream, and the porting trade-offs

| dimension | upstream (TF 1.x) | TensorLBM port | porting decision |
|---|---|---|---|
| hook mechanics | `tf.Variable.assign` ops run by `sess.run(setup_step)`; hooks mutate graph variables | plain Python callables over a `state` dict | kept (same per-step position), because the hook is what lets a case own its time-dependent BC |
| where the BC lives | inside the collision op (`CollideSC`) | inside the step operator (`masked_step` → `compose_masked_boundaries`), also after collision | kept: the blend needs the pre-collision populations, which only exist between `stream` and the BC application |
| mask cardinality | one constant mask per domain | a partition of masks, `Σ m_k ≤ 1` enforced | generalised: free, and it is what makes a composite inlet/wall/outlet geometry expressible |
| moving wall | impossible in the blend; `lid_cavity.py` only writes `Vel` | `moving_bounce_layer` adds `6 w_i (c_i·u_wall)` | added — measured: without it 0.0000 % of U_wall, with it 0.000 % profile error |
| boundary location | implicitly the solid cell face (half-way) | explicit `H = Ny-2` convention, asserted by the analytic case | documented + tested, because the wrong convention alone costs 4.545 % here |
| precision | float32 GPU graph | float64 CPU, exact weight tables | adopted: the equivalence and mass KPIs need round-off, not float32 noise |
| output cadence | float time comparison, video writer in the example | integer step modulo, `save_step(state, t, i)` | adapted (same cadence, exact integer semantics) |
| lattice coverage | D2Q9 / D3Q15 / D3Q19 with hand-written tables | tables derived from the velocity set on import | kept from the in-house side (the earlier CONV-AS-STREAMING XVAL found upstream's D3Q19 `q=3` STREAM tap is wrong, so its tables are not copied) |
| not ported | LES-with-graph-unroll, multi-component MCMP `tau` mixing, video/session plumbing | — | out of scope here: those are separate inventoried features of the same upstream repo (`oss_solutions` rows 34-40) |

Predeclared design (decision, alternatives, thresholds, confounders, stopping rule):

* decision: Adopt upstream's per-step setup hook and mask-blend boundary composition into TensorLBM, and decide whether the mask blend must carry a wall-momentum term (moving walls) or can be transplanted as-is (stationary walls only).
* hypothesis: H1: the hook-driven, mask-blended formulation reproduces the analytic plane-Couette linear profile and the plane-Poiseuille parabola to <3% (owner gate 15%) while being bit-wise identical to an explicit per-cell BC implementation; H2: upstream's stationary-bounce mask blend cannot carry a moving wall, so the velocity-overwrite hook of lid_cavity.py leaves the population state unchanged (0 bits) and the measured flow ~0. H0 falsifier: any predeclared KPI deviates by >=15%, or the mask-blend operator is not equivalent to the explicit one.
* sensitivity axes: channel height Ny in {16, 24, 32} for Couette (discretisation; the analytic profile is linear so the half-way-wall convention is tested exactly); wall-position convention: half-way walls H = Ny-2 (adopted) vs node walls H = Ny-1 (upstream's implicit reading of 'boundary row = solid'), reported as the geometry trap; save_interval in {50, 500} with an identity setup hook (hook cadence must not change physics); lid ramp length in {1, 200} steps (instant onset vs ramped onset -- the hook makes time-dependent BC trivial); channel height Ny in {16, 24} for the forced Poiseuille arm (curvature-sensitive)
* confounders: upstream is TensorFlow 1.x / Python 2 / GPU float32 and cannot execute here, so the comparison is formulation-level against the analytic solution, not against upstream binaries; upstream's own examples are lid-driven cavity / car drag; neither has a closed-form reference, which is why plane Couette and plane Poiseuille are used as the discriminating cases; the moving-wall momentum term changes the BC itself, so arm B_plus_momentum is not a pure reformulation of A: the equivalence KPI is therefore measured on the stationary-wall configuration (where both arms apply a stationary bounce) and on a disjoint two-layer mask benchmark; the composite inlet/outlet channel has no independent reference; only invariants (mass imbalance, no-penetration) are claimed there

## 8. Figures

![fig1_couette_profiles.png](figures/fig1_couette_profiles.png)

*Plane Couette velocity profiles against the analytic line (final step).*

![fig2_grid_convergence.png](figures/fig2_grid_convergence.png)

*Profile L2 error against the analytic solutions versus channel height.*

![fig3_upstream_lid_gap.png](figures/fig3_upstream_lid_gap.png)

*Flow driven by upstream's velocity-overwrite hook versus the momentum-carrying mask blend.*

![fig4_composite_channel_ux.png](figures/fig4_composite_channel_ux.png)

*Streamwise velocity field from four mask layers (two bounce walls, equilibrium inlet, hook-driven outlet).*

![fig5_poiseuille_profiles.png](figures/fig5_poiseuille_profiles.png)

*Plane Poiseuille profile: explicit BC arm, hook+mask-blend arm, analytic parabola.*

Environment: python 3.11.6, torch 2.7.1, numpy 2.4.6, float64, 4 threads, CUDA available = False.
