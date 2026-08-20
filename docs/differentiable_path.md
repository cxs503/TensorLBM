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
