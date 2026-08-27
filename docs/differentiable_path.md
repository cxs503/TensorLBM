# The Differentiable Reference Path (eager solver + autograd)

TensorLBM ships **two solver paths with deliberately different jobs**:

| | eager path (`solver.py` / `solver3d.py`) | Triton fused path |
|---|---|---|
| Kernels | plain PyTorch tensor ops (gather / `torch.roll` streaming, tensorised collision) | fused single-pass Triton kernels |
| Performance | reference implementation (slowest tier) | production tier (8×5090 周期场 ~69 GLUPS) |
| **Autograd** | **differentiable end to end** | **not differentiable** (custom kernels, no backward) |
| Role | teaching, verification, **inverse problems, learned collision operators, solver-in-the-loop training** | production runs (SUBOFF / KVLCC2 / icing / acoustics) |

This is the same design lettuce uses: its main line is differentiable
PyTorch primitives, and `cuda_native` is an *opt-in* fast path that trades
autograd for fused CUDA C++ kernels (Bedrunka et al., *Lettuce: PyTorch-based
Lattice Boltzmann Framework*, ISC High Performance 2021,
arXiv:2106.12929, MIT license). XLB makes the same statement from the other
side: its JAX backend is differentiable while its Warp backend has no adjoint
(`wp.Tape` gradients are zero, verified by their `test_stepper_autodiff.py`;
Ataei & Salehipour, arXiv:2311.16080, CPC 300:109187, 2024).

## What exactly is differentiable (audit 2026-08-20)

Every operator on the collide→stream chain was audited line by line and
verified empirically (gradients exist, are finite, and agree with finite
differences — see `tests/test_autograd.py`):

| Operator | File | Differentiable | Evidence / notes |
|---|---|---|---|
| `equilibrium`, `equilibrium3d` | `d2q9.py`, `d3q19.py` | yes | pure arithmetic; the optional `out=` buffer path stays in the graph (`copy_` records a node) |
| `macroscopic`, `macroscopic3d` | `d2q9.py`, `d3q19.py` | yes | `clamp(rho, min=1e-12)` is inactive at physical densities (gradient passthrough) |
| `stream` (gather), `stream3d` | `solver.py`, `solver3d.py` | yes | advanced-indexing gather; index tensors are constants cached per shape |
| `stream3d_roll` | `solver3d.py` | yes | per-direction `torch.roll` into a fresh tensor; in-place `out[q]=` records `index_put_` nodes |
| `collide_bgk(_3d)`, `collide_bgk_fused`, `collide_bgk_matmul` | `solver.py`, `solver3d.py` | yes | τ may be a 0-dim tensor; all scalar coefficients broadcast |
| `collide_trt(_3d)`, `collide_rlbm(_3d)` | `solver.py`, `solver3d.py` | yes | same |
| `collide_mrt`, `collide_mrt3d` | `solver.py`, `solver3d.py` | yes **after fix** | relaxation vector used to be built with `torch.tensor([...])`, which *silently detached* a tensor τ; now graph-preserving for tensor τ, bit-identical for float τ |
| `correct_mass(_3d)` | both | yes (caveat) | `if current.abs() < 1e-30` is data-dependent control flow (host sync); branch is not taken in practice |

Not part of the differentiable path (documented non-goals):

* `torch.compile` / `route_step` compile modes used by benchmarks — compiled
  steps are not audited for autograd; use the plain eager functions.
* Triton fused kernels (`triton_fused*`) — no backward, by design.
* Boundary-condition helpers that overwrite boundary rows with fixed states
  (in-place `index`/slice assignment of non-derived values) are
  gradient-hostile at the boundary; periodic streaming (used by every
  differentiable case so far) is clean.
* `collide_mrt3d_low_memory` targets peak-RAM production runs (in-place
  moment relaxation); use `collide_mrt3d` when you need gradients.

## Using it

The pattern (full code in `examples/differentiable_lbm.py`, physics carriers
from `benchmarks/verified/{shear_wave_decay,taylor_green_2d}`):

```python
import torch
from tensorlbm.solver import collide_bgk, stream
from tensorlbm.d2q9 import equilibrium, macroscopic

tau = torch.tensor(0.8, requires_grad=True)          # 0-dim tensor parameter
f0 = equilibrium(rho, ux0, uy0).requires_grad_(True)  # or a whole field of parameters

f = f0
for _ in range(n_steps):                              # plain eager loop, no compile
    f = collide_bgk(stream(f), tau)

loss = ((macroscopic(f)[1] - target_ux) ** 2).mean()
loss.backward()                                       # gradients through n_steps of solver
```

### Multi-step memory cost and checkpointing

Backprop-through-time stores every step's intermediates. Measured on one
RTX 5090, 256×256 D2Q9 fp32, full forward+backward (`--measure-memory`):

| steps N | plain autograd | per-step `torch.utils.checkpoint(use_reentrant=False)` |
|---|---|---|
| 10 | 138.4 MiB | 54.4 MiB (2.5× less) |
| 50 | 648.4 MiB | 144.4 MiB (4.5× less) |
| 200 | 2560.9 MiB | 481.9 MiB (5.3× less) |

Plain autograd grows **linearly** (~12.8 MiB/step at 256²); per-step
checkpointing keeps a near-constant floor plus a small per-segment term and
returns bit-identical gradients. Note that `torch.func` transforms
(`grad`, `grad_and_value`) do not support checkpoint's saved-tensor hooks —
combine checkpointing with standard `loss.backward()` / `torch.autograd.grad`
(the example switches automatically). This is the transparent, controllable
alternative to rematerialisation used by XLB's out-of-core adjoint
(checkpoint every 16 steps, replay on backward).

## Relation to `adjoint.py` (frozen-field surrogate)

`tensorlbm/adjoint.py` computes shape sensitivities by running autograd
**through the objective function only**, on a *frozen* flow field produced by
a normal (forward-only) simulation; the time-stepping operator is not part of
the graph. That is a deliberate surrogate — cheap, mesh-agnostic, usable with
the Triton production path — but it cannot answer "how would the flow have
evolved if the parameter changed".

The differentiable reference path closes exactly that gap: gradients flow
through the discrete dynamics itself, so ∂loss/∂τ, ∂loss/∂f₀ and (future)
∂loss/∂θ for a learned collision operator are *exact* derivatives of the
simulated quantity. The two are complementary:

* frozen-field surrogate (`adjoint.py`): large production grids, objective-only sensitivity, no rollout memory;
* differentiable path (this document): small/medium grids, exact through-dynamics gradients, inverse problems and training.

## Entry points

* `tests/test_autograd.py` — gradient existence (2-D BGK/MRT/TRT ×{fused, matmul}, 3-D gather/roll streaming), finite-difference cross-checks, SGD τ-recovery convergence, checkpoint-vs-plain gradient equality.
* `examples/differentiable_lbm.py` — τ recovery (viscosity inverse problem) and initial-condition shaping; adapted from XLB `examples/cfd/differentiable_lbm.py` (Apache-2.0, changes listed in its header); includes the memory measurement.

## Roadmap hooks enabled by this path

1. **Learned collision operators** — any `nn.Module` that maps (f, macroscopic) → f can be dropped into the collide slot and trained end to end against downstream losses (lettuce's `LearnedMRT` pattern; neural bulk-viscosity C&F 2024).
2. **Solver-in-the-loop correction training** — train a coarse-grid solver + NN correction against fine-grid references by backprop through the solver (Um et al., NeurIPS 2020; XLB paper §6.1), instead of the current offline FNO pipeline in `apps/neural_operator_fno.py`.
3. **Parameter inverse problems** — τ/ω fields, initial conditions, force terms identified from observations, as demonstrated by the example.

## Implementation progress (2026-08-22): packaged step chain + solver-in-the-loop demos

Roadmap item **A6** turns the audited property above into a packaged,
importable composition contract: `tensorlbm.autograd_path` (exported via
`tensorlbm.api`). It is the TensorLBM counterpart of XLB's
solver-in-the-loop demos (Um et al., NeurIPS 2020; XLB, Apache-2.0 —
paradigm only, no XLB code).

### API

| Function | Signature | Role |
|---|---|---|
| `differentiable_step` | `(f, tau=0.9, mask=None, *, collide=None, return_probe=False, inlet=None, outlet=None, walls=None, outlet_prev=None)` | one autograd-clean D3Q19 step: collision skipped inside the solid (NoDynamics, `torch.where`) → periodic gather streaming (`solver3d.stream3d`) → boundary conditions (lateral walls → inlet → outlet, post-stream / pre-bounce-back) → full-way bounce-back (`where(mask, f[opp], f)`) |
| `rollout` | `(f, n_steps, tau=0.9, mask=None, *, collide=None, checkpoint=False, return_probes=False, inlet=None, outlet=None, walls=None)` | unrolls the step, optionally per-step gradient-checkpointed (`use_reentrant=False`, identical gradients, near-flat activation memory); chains the convective outlet's face history internally |
| `obstacle_force` | `(f_probe, mask)` | differentiable Ladd wet-node momentum-exchange force `F_α = 2·Σ_{x∈solid} Σ_q c_{qα} f[q,x]` on the post-stream / pre-bounce-back probe |

Boundary specs (all fields accept floats or graph-connected 0-dim tensors;
all default `None` = fully periodic, bit-for-bit the original chain):

* `InletSpec(ux, uy, uz, rho0, method="equilibrium"|"zouhe")` — velocity
  inlet on x = 0: whole-plane Dirichlet equilibrium, or Zou/He
  non-equilibrium closure of the five unknown (c_x = +1) populations with
  the plane density from the known streamed moments (pins the *normal*
  velocity; requires u_x < 1 in lattice units).
* `OutletSpec(method="copy"|"convective", u_conv=None)` — outlet on
  x = nx-1 acting on the five unknown (c_x = -1) populations:
  `"copy"` = zero gradient from x = nx-2; `"convective"` = first-order
  upwind `f_out^{n+1} = f_out^n + U_c (f_{out-1}^n - f_out^n)` (a convex
  combination, stable for 0 < U_c < 1, the upwind CFL bound; U_c = 1
  degenerates to the copy). `u_conv=None` derives U_c from `inlet.ux`;
  an explicit tensor makes the Courant number itself learnable. The
  recursion on the previous outlet face is chained automatically inside
  `rollout` (seeded from the initial condition; single steps take
  `outlet_prev`).
* `WallSpec(method="periodic"|"free-slip"|"freestream", rho0, ux, uy, uz,
  overrides=None)` — the spec's own method drives every lateral face that is
  not overridden (all four by default). `"free-slip"` is on-node specular
  reflection: the unknown populations on each face take their mirror
  partner (`f[q] = f[FLIP[q]]`, index-level swap) — wall-normal velocity
  cancels pairwise to machine precision, tangential momentum untouched,
  no closure arithmetic. `"freestream"` resets whole faces to
  `f_eq(rho0, u_inf)` (the equilibrium-inlet construction). Faces close in
  the order y=0, y=ny-1, z=0, z=nz-1 *before* the inlet/outlet; on
  edge/corner lines the later closure wins (last write wins).

  Per-face control (A6+++): `overrides` maps face keys to their own
  `WallSpec`; unlisted faces keep the spec itself as their default closure.
  Face keys are the outward normals of the lateral box:

  | key | plane | | key | plane |
  |---|---|---|---|---|
  | `"-y"` | y = 0 | | `"-z"` | z = 0 |
  | `"+y"` | y = ny-1 | | `"+z"` | z = nz-1 |

  Typical layouts: `WallSpec(method="free-slip",
  overrides={"+z": WallSpec(method="freestream", ux=u_inf)})` — slip walls
  with a far-field top (wind-tunnel floor); `WallSpec(method="periodic",
  overrides={"+y": WallSpec(...)})` single-face patches; asymmetric
  top/bottom closures (`"-z"` / `"+z"` with different methods or
  free-stream values). Override
  specs may not nest `overrides` (fails at construction); a face overridden
  to `"periodic"` is a no-op like the shared default. The face order,
  edge/corner last-write-wins policy and the inlet/outlet phase are
  unchanged; `overrides=None` (or absent) keeps the A6++ shared-closure
  operator **bit-for-bit** (verified `torch.equal` against base 79b17f3 on
  18 fp64/fp32 configs: plain rollouts, uniform free-slip/free-stream and
  the full bounded box incl. checkpointing and probes).

  `WallSpec.to_dict()` / `WallSpec.from_dict()` serialise the spec
  (tensor fields flattened to their numeric value — the graph is not
  serialisable); payloads without an `"overrides"` key (pre-A6+++)
  load unchanged, missing numeric fields fall back to the defaults, and
  unknown keys are ignored.

`tau` may be a graph-connected 0-dim tensor; `collide` is a slot — the
default is single-component BGK (`collide_bgk3d`), and any differentiable
`f, tau -> f` callable drops in (e.g.
`functools.partial(collide_smagorinsky_bgk3d, C_s=cs)` to make the
Smagorinsky constant learnable). Layout stays `(19, nz, ny, nx)`, fp32/fp64.

### Gradient cross-checks (measured, `tests/test_autograd_path.py`)

12-step masked rollout (10×12×16 grid, centred sphere r=2.5), loss = MSE of
final `ux` on fluid cells vs a reference rollout, gradient w.r.t. `tau`
compared to the central difference of the same discrete loss:

| dtype | FD ε | relative error |
|---|---|---|
| float64 | 1e-5 | **8.7e-10** |
| float32 | 5e-3 | **2.2e-4** |

Element-wise `dLoss/df0` (obstacle-adjacent, far-fluid and inside-sphere
entries) and `d(Σ_k F_x,k)/dtau` (drag probe accumulated over 8 steps) agree
with central differences to <1e-6 in float64. Checkpointed rollouts return
bit-identical gradients (rtol 1e-10). CPU↔CUDA fp32 parity: rollout loss and
`dLoss/dtau` agree to <1e-5 / <1e-3.

### Solver-in-the-loop identification (12×16×24 grid, sphere r=3.5, K=15 steps, Adam + cosine lr decay, fp64; `examples/solver_in_the_loop.py`)

| mode | observable | initial loss | final loss | reduction | parameter error |
|---|---|---|---|---|---|
| τ (BGK), truth 0.85, guess 0.60 | final `ux` field | 1.94e-4 | 2.29e-9 | ×8.5e4 | **1.0e-3** |
| τ (BGK) | accumulated obstacle drag | 5.39e-2 | 2.47e-7 | ×2.2e5 | **5.9e-4** |
| C_s (Smagorinsky BGK), truth 0.12, guess 0.03 | final `ux` field | 2.75e-8 | 1.13e-13 | ×2.4e5 | **1.2e-4** |
| C_s (Smagorinsky BGK) | accumulated obstacle drag | 2.94e-3 | 6.99e-9 | ×4.2e5 | **9.0e-5** |

Both scalar parameters are recovered to ≤1e-3 absolute through the discrete
dynamics; the eddy viscosity of the LES closure is identified from the flow
itself because the sub-grid model is inside the backward graph.

### Bounded box (A6+, merged; A6++ lateral walls + convective outlet): measured gradients

Full box = Zou/He inlet (`u_in` = 0.08) + convective outlet (`U_c` derived
from `u_in`) + free-slip walls, 8×10×20 grid, sphere r=2, 10-step rollout,
loss = MSE of final `ux` on fluid cells, fp64, central differences
(ε = 1e-5, `tests/test_autograd_path.py` §8):

| gradient | autograd | finite difference | relative error |
|---|---|---|---|
| dLoss/dτ | -6.115089e-05 | -6.115089e-05 | **9.6e-10** |
| dLoss/du_in (U_c derived from u_in) | -3.167887e-03 | -3.167887e-03 | **7.3e-09** |
| dLoss/dU_c (explicit learnable Courant number) | -7.515991e-05 | -7.515991e-05 | **3.3e-09** |

Element-wise `dLoss/df0` matches central differences (<1e-6 relative, or
<1e-12 absolute for the ~1e-8-gradient corner-line entry) at interior,
wall-plane (mirrored), corner-line and outlet-plane (convective seed)
entries. Checkpointed rollouts reproduce the plain gradients exactly with
the convective history chained; CPU↔CUDA fp32 parity holds for the full box.

Physics sanity (300 steps, fp64, same grid, τ = 0.55): all populations
finite, ρ ∈ [0.94, 1.07], mean `ux` = 0.087 sustained, drag window-stable
(0.287 → 0.289 over windows [50,100) → [250,300)). Free-slip faces carry
|u_n| ≤ 2.5e-13 (machine precision, pairwise cancellation of the mirrored
pairs); the summed mean |u_n| over the four faces is **2.0e-17** with
free-slip walls vs **6.6e-3** with the periodic wrap — the side pollution
of the periodic baseline, quantitatively.

Per-face walls (A6+++, same grid/loss, walls = free-slip with a
`"+z"` free-stream override carrying the learnable far-field speed
`u_inf` = 0.06 and a periodic `"-z"` face, `tests/test_autograd_path.py`
§9): explicit per-face replicas of one closure reproduce the shared-spec
operator `torch.equal`-exactly; the mixed box matches an independent
face-by-face replay built from the hand-derived mirror tables; on an
asymmetric initial field (uy(z), uz(y), phase-shifted so no mirror symmetry
can fake a pass) the slip faces hold |u_n| < 1e-14 with tangential momentum
retained at its O(amplitude) level, the free-stream face sits at its own
`(rho0, u_inf)` to 1e-12 and the periodic face keeps the wrap (equal to the
fully periodic chain off the edge lines). Gradient through the override:

| gradient | autograd | finite difference | relative error |
|---|---|---|---|
| dLoss/du_inf (free-stream `"+z"` override) | 2.626131e-04 | 2.626131e-04 | **1.4e-09** |

Checkpointed rollouts reproduce the plain gradients with the mixed walls
active, and `to_dict`/`from_dict` round-trip the spec (including nested
per-face overrides) while pre-A6+++ payloads load unchanged.

### Known limits of this module

* **Stationary** mask only; moving obstacles and BFL interpolated
  bounce-back (`bfl_boundary.py`) are not wired for autograd. In-place
  overwrite boundary helpers stay out of the chain (gradient-hostile, see
  audit above); NSCBC / sponge pressure relaxation is not differentiable
  here.
* The bounded box is closed but minimal: the inlet pins the **normal**
  velocity only (no Zou/He tangential reconstruction, no turbulent /
  synthetic inflow); the convective outlet uses one uniform `U_c` on the
  outlet plane alone (first-order upwind, 0 < U_c < 1; no sponge, no
  per-direction convective speeds); per-face wall control (A6+++) lets each
  lateral face pick one of the three existing closures independently — no
  new physics (no moving wall/lid method, no per-face parameters beyond
  the free-stream values); on the
  edge/corner lines the later-applied closure wins on doubly-unknown
  directions (last write wins, measured corner-line |u_n| ~ 1e-3 vs
  machine-zero face interiors); free-slip walls are on-node (wall on the
  plane nodes, first-order wall placement — the slip analogue of on-node
  bounce-back).
* The Smagorinsky path keeps main's *absolute* `tau_eff` clamp semantics
  (`torch.clamp(tau_eff, 0.5001, 1.0)` in `turbulence.py`); gradient is zero
  in clamped cells (a.e. differentiability of `clamp`).
* Full-way bounce-back places the effective wall at the solid-cell centre
  (first-order geometry), consistent with the production SUBOFF convention.
* Multi-component, free-surface and distributed paths, and memory-format
  optimisations are deliberate non-goals (A2/A3 territory).

### Entry points (new)

* `tests/test_autograd_path.py` — 49 cases: value contract vs manual
  composition, gradient existence through the masked chain, per-dtype FD
  cross-checks (τ, f0 entries, drag probe), τ and C_s solver-in-the-loop
  recovery, checkpoint-vs-plain gradient equality, CPU↔CUDA parity, the
  bounded inlet/outlet contract of A6+ (§7), the A6++ full-box contract
  (§8: default-path bitwise identity, wall/outlet value contracts, FD
  through all six faces incl. dU_c, checkpoint/CUDA consistency, and the
  300-step sphere physics vs the periodic-sides baseline), and the A6+++
  per-face wall contract (§9: uniform-override bitwise identity, mixed
  value/physics contracts on an asymmetric field, dU_inf FD cross-check,
  checkpoint equality, to_dict/from_dict round-trip incl. pre-A6+++ payloads
  and spec validation).
* `examples/solver_in_the_loop.py` — the four identification runs above
  (`--mode {tau,cs}`, `--observable {field,drag}`).

## Inverse design (2026-08-23): soft solids, geometry in the autograd graph

The one thing the masked chain above cannot do is differentiate a loss with
respect to **the shape itself** — `sphere_mask` is a boolean grid, a
discrete object with no gradient. The opt-in soft-solid path
(`src/tensorlbm/soft_geometry.py`) closes that gap:

```python
from tensorlbm import SoftGeometry                       # exported in api.__all__
from tensorlbm.autograd_path import differentiable_step, rollout, obstacle_force

radius = torch.tensor(2.9, dtype=torch.float64, requires_grad=True)
geom = SoftGeometry(kind="sphere",                        # or "ellipsoid", "box"
                    center=(7.0, 8.0, 6.0),
                    size=(radius,),                       # any entry may be a
                    epsilon=0.25)                         # 0-dim graph tensor
f, probes = rollout(f0, K, tau, None, soft=geom, inlet=..., outlet=...,
                    return_probes=True, probe_start=...)
solid = geom.solid_weight(nz, ny, nx, ...)                # 1 - w
loss = sum(obstacle_force(p, solid)[0] for p in probes)   # soft drag
loss.backward()                                           # d(loss)/d(radius)
```

`SoftGeometry` is a frozen dataclass: an analytic signed distance function
`phi(x; params)` (negative inside the solid) blurred by a temperature
`epsilon` into the fluid weight `w(x) = sigmoid(phi/epsilon)` — 1 in the
fluid, 0 in the solid, transition band `~6*epsilon` wide. Three shapes:
`sphere` (exact SDF `|p-c| - r`), `box` (the standard axis-aligned SDF
`length(max(q,0)) + min(max(qx,qy,qz),0)`, `q = |p-c| - h`), and
`ellipsoid` (the normalised-radius surrogate `(t-1)*min(a,b,c)` with
`t = sqrt(sum ((p_i-c_i)/a_i)^2)` — exact zero set, distance measured in
units of the shortest semi-axis; the exact ellipsoid SDF needs an iterative
closest-point projection and is deliberately not reproduced).

### The three soft operators (and their hard-limit contract)

Each mask-dependent operator of the step chain becomes a convex homotopy in
`w` (`s = 1 - w` is the solid weight):

| phase | hard-mask operator | soft operator |
|-------|--------------------|---------------|
| collision (NoDynamics skip) | `where(mask, f, collide(f))` | `w*collide(f) + (1-w)*f` |
| full-way bounce-back | `where(mask, f_str[opp], f_str)` | `w*f_str + (1-w)*f_str[opp]` |
| momentum-exchange force | `2 sum_{x in solid} sum_q c_q f` | `2 sum_x s(x) sum_q c_q f` |

Full-way bounce-back reflects populations **on the wet node itself**, so the
weight is sampled at the same node — no link-wise mask shift (a half-way
link scheme would need the weight at `x + c_q`; the existing hard
implementation needs none either, for the same reason).

The soft force is *derived*, not asserted: per node the soft bounce-back
changes the field momentum by `sum_q c_q (f_post - f_str) = -2 s sum_q c_q
f_str` (relabel `q -> opp` with `c_opp = -c_q`), so by action-reaction the
momentum handed to the body is exactly `2 s sum_q c_q f_probe` — the soft
force above. Assumption: `s(x)` is read as the node's solid occupancy.
`obstacle_force` needs no signature change: passing the float field
`1 - w` as its mask argument realises the soft form.

**Bitwise hard limit.** Beyond `|phi/epsilon| = 30` the weight is clipped to
*exactly* 0/1 (the logistic tail there is ~1e-13 — no physics change, but a
blend weight of 1e-28 is not zero: the homotopies would collide/bounce
deep-solid cells at O(1) instead of freezing them). At black/white weights
every blend collapses onto its `torch.where` selection, so at a saturating
temperature (e.g. `epsilon = 1e-3` for a sphere whose closest node is
~0.064 from the surface) the **whole chain** — final state, every probe,
accumulated drag — is `torch.equal` to the hard-mask chain of the same
parameters (`test_soft_chain_hard_limit_bitwise`). The default `soft=None`
path is untouched: `test_soft_none_default_path_bitwise_unchanged` replays
the original chain by hand, and the env-gated
`test_default_path_bitwise_vs_baseline_artifacts` compares against the
frozen `baseline.pt` artefacts generated at the pre-feature commit 02275f55
(`scripts/gen_autograd_inverse_baseline.py --check /nfs/wangxi/tmp/inv_base`,
sha256 3e47242b71c64ae4...). The action-reaction identity itself is verified
to machine precision in a periodic box (`F = -d(momentum)/dt`, rel 5.6e-15).

### Measured gradient accuracy (fp64, 5-step rollout, Zou/He inlet + convective outlet, 8x10x20)

| derivative | autograd | central FD | rel error |
|------------|----------|------------|-----------|
| d(drag)/d(radius), sphere r=2.3 | +1.465125260e+01 | +1.465125260e+01 | 3.4e-11 |
| d(drag)/d(b), ellipsoid (2.3, b, 1.5), b=1.8 | +5.423418179e+00 | +5.423418179e+00 | 1.0e-10 |
| d(drag)/d(hx), box (hx, 1.7, 1.4), hx=2.2 | +3.804437524e+00 | +3.804437524e+00 | 7.3e-11 |
| d(drag)/d(cx), sphere, cx=6.3 | +2.313765523e-01 | +2.313765357e-01 | 7.2e-08 |
| d(drag)/d(epsilon), sphere, eps=0.5 | +4.020864986e+01 | +4.020864986e+01 | 8.4e-11 |

(FD step 1e-5; the centre derivative uses 1e-4 — its signal is smaller, so
the quotient is round-off dominated there. All five are pinned in
`tests/test_autograd_inverse.py` at rel < 1e-6 / 1e-5.) Note the same fd64
lesson as the tau audits: keep geometry parameters in fp64 when learning
them — the SDF divides by `epsilon`, and fp32 blends show up at the 1e-3
level in the drag.

### Soft -> hard C_D convergence (12x16x26, sphere r=2.3, u_in=0.08, tau=0.55, 400 steps, window [300,400), fp64)

| epsilon | C_D(soft) | C_D(hard) = 4.511620 | rel err |
|--------:|----------:|--------------------:|--------:|
| 0.5     | 17.072033 |               4.511620 | +2.78   |
| 0.25    |  6.869165 |               4.511620 | +0.522  |
| 0.125   |  5.000857 |               4.511620 | +0.108  |
| 0.0625  |  4.672132 |               4.511620 | +0.0356 |
| 0.02    |  4.502984 |               4.511620 | -0.0019 |

Monotone convergence (`test_soft_to_hard_cd_convergence` also pins this);
`epsilon <= 0.02` lattice units lands within 1%. The epsilon bias is
**window-dependent**: it is a boundary-layer artefact of the smeared
interface, and during the startup transient the soft sphere reads
substantially "fatter" than the hard one (measured on the same grid: the
soft campaign overestimates the window drag of the same sphere by ~1.6x in
the window [30,60), versus +0.5% in the steady window [300,400)).

### Solver-in-the-loop demos (`examples/inverse_design.py`)

Bounded box, equilibrium inlet `u_in = 0.08`, zero-gradient outlet, Adam +
cosine lr decay (5% floor), normalised loss `((drag - drag*)/drag*)^2`,
window-mean momentum-exchange drag over the probe window, fp64:

* **sphere radius** (12x16x26, truth r* = 2.3 hidden, guess 2.9, eps 0.25,
  K = 60 with window [30,60), 120 iters, lr 0.02): loss 3.287e-01 ->
  4.630e-06, recovered radius 2.2973 (abs err 2.7e-03), endpoint C_D error
  **2.1e-03**. The test-suite version (`test_inverse_design_radius_recovery`,
  10x13x22, r* = 2.0, guess 2.6, 150 iters) recovers radius 2.000753,
  endpoint C_D error 6.2e-04.
* **ellipsoid semi-axis** (same grid, learn b of (2.0, b, 1.6), truth
  b* = 2.2, guess 2.6): recovered b = 2.2015 (abs err 1.5e-03), endpoint
  C_D error 7.5e-04. The window-drag observable is **multi-modal** in b —
  it has an interior minimum near b = c (the round cross-section, ~1.6
  here), and b = 0.8 and b = 2.2 give the same window drag to 7e-4 — so the
  guess must start on the truth branch; descending from below the minimum
  finds the twin solution on the slender branch instead.
* **hard reference** (`--target hard --iters 400 --lr 0.05`): the target
  drag comes from the hard-mask campaign of the same sphere; the optimiser
  matches the target C_D to **1.5e-07** — but at radius 1.6901 vs truth
  2.3, i.e. shifted down by the exact amount that compensates the
  startup-window epsilon bias above (~1.6x drag at the same radius).  The
  visible cost of fitting a hard-sphere target with an eps = 0.25 model in
  the transient regime; the default self-consistent soft target has no such
  bias.

### Known limits of the inverse-design path

* **Analytic SDFs only**: sphere / ellipsoid / box with centre, size and
  epsilon parameters. No voxelised/imported geometry, no level-set advection
  — a generic mask has no gradient by construction. The ellipsoid SDF is a
  surrogate (distance in units of the shortest semi-axis), so `epsilon` is
  exactly lattice-units only along the shortest axis.
* **Epsilon boundary-layer bias**: the smeared interface thickens the
  obstacle by `O(epsilon)`; C_D converges to the hard value monotonically
  but only reaches 1% at `epsilon <= 0.02`, while gradients need the band
  wide enough to cover nodes (`~0.25` is the practical optimum). Fitting a
  *hard*-reference observable with a soft model biases the recovered
  parameter; self-consistent soft targets (the default demo) do not.
* **Multi-modal observables**: drag-vs-size is monotone for the sphere but
  not for the ellipsoid semi-axis (interior minimum near the round
  cross-section); gradient descent finds the solution on the branch it
  starts from.
* **Static shapes**: the weight is recomputed from the SDF every step, but
  the parameters are constant across the rollout — no moving/shape-morphing
  control within one optimisation iteration.
* sqrt arguments are clamped at 1e-30 in the sphere/ellipsoid SDFs: values
  are unchanged to ~1e-15 but the (meaningless) gradient at the exact
  centre node is zeroed — inf * 0 = NaN otherwise.

### Entry points (inverse design)

* `src/tensorlbm/soft_geometry.py` — `SoftGeometry` (sdf / fluid_weight /
  solid_weight / hard_mask).
* `tensorlbm.autograd_path.differentiable_step(..., soft=geom)` /
  `rollout(..., soft=geom)` — opt-in soft chain (mutually exclusive with
  `mask`; default `None` keeps the hard chain bit-for-bit).
* `tests/test_autograd_inverse.py` — 19 cases: SDF/weight value contracts,
  soft-step manual replay (BCs incl. per-face walls), default-path bitwise
  identity (in-test + env-gated vs `baseline.pt`), saturated-eps bitwise
  hard limit (state/probes/drag), action-reaction balance, FD gradient
  cross-checks for all five parameters, checkpoint equality, CUDA parity,
  the C_D convergence table and the radius-recovery optimisation.
* `scripts/gen_autograd_inverse_baseline.py` — baseline artefact
  generator/checker (5 configs, CPU-deterministic, `torch.equal`).
* `examples/inverse_design.py` — the demos above
  (`--param {radius,semi-axis}`, `--target {soft,hard}`, `--eps`, `--iters`).
