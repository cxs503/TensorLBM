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
| `differentiable_step` | `(f, tau=0.9, mask=None, *, collide=None, return_probe=False)` | one autograd-clean D3Q19 step: collision skipped inside the solid (NoDynamics, `torch.where`) → periodic gather streaming (`solver3d.stream3d`) → full-way bounce-back (`where(mask, f[opp], f)`) |
| `rollout` | `(f, n_steps, tau=0.9, mask=None, *, collide=None, checkpoint=False, return_probes=False)` | unrolls the step, optionally per-step gradient-checkpointed (`use_reentrant=False`, identical gradients, near-flat activation memory) |
| `obstacle_force` | `(f_probe, mask)` | differentiable Ladd wet-node momentum-exchange force `F_α = 2·Σ_{x∈solid} Σ_q c_{qα} f[q,x]` on the post-stream / pre-bounce-back probe |

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

### Known limits of this module

* Periodic lattice + **stationary** mask only. Zou/He, NSCBC and
  overwrite-style boundary helpers are not part of the chain (gradient-hostile
  in-place writes, see audit above); moving obstacles and BFL interpolated
  bounce-back (`bfl_boundary.py`) are not wired for autograd.
* The Smagorinsky path keeps main's *absolute* `tau_eff` clamp semantics
  (`torch.clamp(tau_eff, 0.5001, 1.0)` in `turbulence.py`); gradient is zero
  in clamped cells (a.e. differentiability of `clamp`).
* Full-way bounce-back places the effective wall at the solid-cell centre
  (first-order geometry), consistent with the production SUBOFF convention.
* Multi-component, free-surface and distributed paths, and memory-format
  optimisations are deliberate non-goals (A2/A3 territory).

### Entry points (new)

* `tests/test_autograd_path.py` — 13 cases: value contract vs manual
  composition, gradient existence through the masked chain, per-dtype FD
  cross-checks (τ, f0 entries, drag probe), τ and C_s solver-in-the-loop
  recovery, checkpoint-vs-plain gradient equality, CPU↔CUDA parity.
* `examples/solver_in_the_loop.py` — the four identification runs above
  (`--mode {tau,cs}`, `--observable {field,drag}`).
