# Triton Multi-GPU Distributed Notes

Engineering notes for the Triton multi-GPU (slab decomposition + NCCL halo
exchange) modules in TensorLBM.  Written 2026-08-19 after several
multi-hour debugging sessions on the 8 × RTX 5090 box; §2 is a checklist
to read **before** touching the distributed runner or the fused kernel.

## 1) Architecture

### 1.1 Slab decomposition (z axis)

`tensorlbm.triton_fused_distributed.DistributedTritonFusedSolver3D` splits
the global D3Q19 tensor `(Q=19, nz_global, ny, nx)` along **z** into
`world_size` contiguous slabs:

```
rank r owns z in [r * nz_local, (r + 1) * nz_local),  nz_local = nz_global // world_size
```

`nz_global` must be divisible by `world_size` (checked in `__init__`).
Each rank holds one buffer of shape `(19, nz_local + 2, ny, nx)` — its
owned planes at local indices `[1, nz_local + 1)`, plus **two ghost
planes**: index 0 (received from rank `r-1`) and index `nz_local + 1`
(received from rank `r+1`).  Neighbours wrap periodically
(`left = (r-1) % world`, `right = (r+1) % world`); a non-periodic slab
boundary would require overriding `partition` for an asymmetric halo.

Note the layout convention carried over from the single-GPU modules: the
tensor layout is `(Q, nz, ny, nx)` and **x (axis 2) is the streamwise
axis**; z is the slab axis.

### 1.2 Ping-pong buffers

The class owns a single scratch buffer `self._buf` sized like the input.
`step(f_local)`:

1. waits for the *previous* step's async halo handles (so `f_local`'s
   ghost planes are fresh),
2. runs the fused collide+stream kernel `f_local -> self._buf` on the
   full local buffer (`nz = nz_local + 2`, ghosts included — ghost-plane
   writes happen but are discarded by the next exchange),
3. **starts** the halo exchange on the post-step buffer,
4. returns `self._buf`; the caller passes it back as the next call's
   `f_local`.

So the caller's buffer and `self._buf` are the two sides of the
ping-pong.  In the SUBOFF wrapper (`TritonSuboffDistributedRunner`) the
same pattern holds: `self._buf` (produced by `from_global`) and
`self._dist._buf` alternate, and `self._dist._buf` is re-used lazily as
the kernel output buffer.

### 1.3 NCCL halo exchange

`_start_halo_exchange(f)` posts one `dist.batch_isend_irecv` batch of
four `P2POp`s:

| op  | plane                              | neighbour |
|-----|------------------------------------|-----------|
| isend | owned plane `1`                  | left  (becomes their right ghost) |
| isend | owned plane `nz_local`           | right (becomes their left ghost)  |
| irecv | ghost `nz_local + 1`             | right |
| irecv | ghost `0`                        | left  |

The returned work handles are stored in `self._halo_handles` and awaited
at the top of the *next* `step()` (or via `synchronize()`), so the
exchange overlaps with everything the caller does between steps (BC
writes, force reduction, I/O).  Single-rank fallback: ghosts are filled
by copying the nearest owned plane, keeping the code path identical.

The periodic z-wrap needs **no kernel changes** vs the single-GPU
version: the kernel treats `nz_local + 2` as the periodic length and the
ghost planes sit exactly where the wrap lands.

### 1.4 Relationship to the single-GPU kernel: Q=19 vs `_Q_PAD=32`

- The halo exchange, all host-side buffers, and production tensors are
  **Q=19** (physical D3Q19).
- The Triton kernels use `Q_PAD = 32` internally (`_Q_PAD = 32` in
  `triton_fused.py`, "next power of two >= 19") because `tl.arange`
  requires a power-of-2 range.  In the V2 obstacle kernel
  (`_fused_v2_kernel_xfar_les` / `triton_fused_obstacle_xfar_les`) the
  padding is **inside the kernel**: the host passes Q=19 tensors, the
  kernel builds a `(Q=32 arange, BZ, BY, BX)` tile in registers.
- The *legacy* entry point `triton_fused_obstacle_les` still requires
  **Q=32 host tensors** and raises
  `ValueError: Expected Q=32 (padded D3Q19), got Q=19` otherwise.  That
  path is where the historical `_out_pad` mirror dance (§2.2) came
  from.

Measured on 8 × RTX 5090 at `nz_local=48, ny=nx=384` (global
384 × 384 × 384): weak-scaling efficiency ~100%, ~69 GLUPS aggregate —
i.e. the per-rank kernel runs at full single-GPU speed.

### 1.5 SUBOFF layering

`triton_suboff_step_distributed.TritonSuboffDistributedRunner` wraps the
slab solver and adds, per step:

1. `synchronize()` on the in-flight halo exchange,
2. one call to `triton_fused_obstacle_xfar_les` (fused collide + stream
   + bounce-back + LES; optional `tau_eff` for external WALE/Vreman),
   with fused Ladd force accumulation via `tl.atomic_add` into
   `fx_buf/fy_buf/fz_buf`,
3. async halo roll on the new buffer,
4. far-field BC on the **owned** planes (`apply_far_field_bc_6face`),
5. mass correction every 10 steps against the per-rank initial mass.

The geometry build (`build_suboff_mask`) is **collective**: every rank
builds the full global mask and slices its own slab out of it.  The
obstacle is padded with two zero (fluid) ghost planes to match the
slab+halo f-buffer shape.  Global forces come from an `all_reduce`
performed by the upstream runner (§2.3).

## 2) Gotchas checklist

Eight items, each: **Symptom → Root cause → Fix**.  All of these were
found the hard way; check them first.

### 2.1 NCCL P2P slices must be `.contiguous()`

- **Symptom:** `dist.batch_isend_irecv` rejects the operands (or halos
  come back garbage); the failure only appears on the multi-rank path.
- **Root cause:** a z-plane slice of a `(Q, nz+2, ny, nx)` tensor,
  e.g. `f[:, 1:2, :, :]`, is **not dense** — the Q-dim stride is far
  larger than the slice width.  NCCL point-to-point ops require dense,
  contiguous buffers.
- **Fix:** call `.contiguous()` on **every** send/recv operand.  This is
  already done inside `_start_halo_exchange` (all four `P2POp`s); any
  hand-written exchange must do the same.

### 2.2 Q=19/32 padding and the persistent buffers

- **Symptom:** `ValueError: Expected Q=32 (padded D3Q19), got Q=19` from
  the legacy kernel entry; or (historically) per-step `torch.empty`
  allocations of a 32-channel buffer eating the step time.
- **Root cause:** the halo exchange operates on **Q=19**, but kernels
  historically wanted **`_Q_PAD=32`** channels on the host side.
- **Fix:** on the current **V2** entry (`triton_fused_obstacle_xfar_les`)
  the padding is internal — keep every host-side buffer Q=19
  (`self._dist._buf`, `self._buf`, the NCCL slices).  If you use the
  legacy padded entry, keep a **persistent** `_out_pad` (32 channels)
  plus the Q=19 exchange buffer `self._dist._buf`; mirror 19→32 before
  the kernel, copy 32→19 back after, and pre-allocate **once in
  `__init__`** — never per step.

### 2.3 Force all-reduce is mandatory (`C_t` uses the global mask)

- **Symptom:** `C_t` per rank is wrong (each rank reports only its own
  slab's force, values differ across ranks, and none matches the
  single-GPU run).
- **Root cause:** the geometry build is collective — **every rank builds
  the full mask**, so `dynamic_pressure` (from
  `_voxel_wetted_area(solid)`) is a **global** quantity on every rank.
  The fused kernel's `tl.atomic_add` force buffers, however, only
  accumulate wall cells inside the rank's own slab (ghost planes have
  `obstacle=0`, so they contribute zero).
- **Fix:** after the step, when `world_size > 1`:

  ```python
  dist.all_reduce(fx_t, op=dist.ReduceOp.SUM)
  dist.all_reduce(fy_t, op=dist.ReduceOp.SUM)
  dist.all_reduce(fz_t, op=dist.ReduceOp.SUM)
  ```

  This lives in the runner's distributed branch
  (`suboff_cmk_kbc_runner.py`), not in the wrapper's `step()`.

### 2.4 Obstacle/buffer padding slices: `f[:, 1:nz_local+1]`, `obstacle[1:-1]`

- **Symptom:** forces or mass-correction numbers off (ghost cells
  included / double-counted), or shape-mismatch errors when handing
  buffers to the kernel or BC helpers.
- **Root cause:** slab buffers carry 2 ghost planes — f is
  `(Q, nz_local+2, ny, nx)` while the obstacle's owned region is
  `(nz_local, ny, nx)`.  Owned data sits at `f[:, 1:nz_local+1]` and
  `obstacle[1:-1]`.
- **Fix:** slice explicitly for any owned-plane operation:
  `owned = out_local[:, 1:self.nz_local + 1, :, :]`,
  `self.solid_int8_local[1:-1] = solid_local`.  Ghost planes of the
  obstacle are zeros (always fluid).

### 2.5 Ladd force phase: sample `f_in` (pre-bounce-back), never `f_eff`

- **Symptom:** drag force inflated ~**100×** versus the PyTorch
  reference.
- **Root cause:** inside the kernel, bounce-back produces a new register
  `f_eff`; sampling `f_eff` at wall cells does not match production's
  `compute_obstacle_forces_3d`, which samples **post-stream,
  pre-bounce-back** populations.
- **Fix:** the kernel preserves `f_in` and computes
  `F_α = 2 · Σ_q c_{q,α} · f_in[q]` masked by `own_is_wall` — exactly
  the populations the PyTorch path would see, at zero extra reads.  If
  you rewrite the force block, keep sampling `f_in`.

### 2.6 `compute_force` toggle and the fp32 placeholder pointer

- **Symptom:** crash (or garbage) when running with force computation
  disabled; typically from passing `obstacle` (int8) as the placeholder
  for the force pointers.
- **Root cause:** Triton requires **non-null pointers even for
  constexpr-dead code**, and `tl.atomic_add` requires **fp32** pointers.
- **Fix:** when `fx_buf/fy_buf/fz_buf` are not supplied, the wrapper
  passes the **fp32 `out` buffer** as the placeholder for all three;
  the kernel sees `COMPUTE_FORCE=False` and prunes the whole force
  block.  This saves **~8% of kernel time at n=128**.  Runner knob:
  `SuboffCmkKbcConfig(compute_force=...)` (default `True`; when `False`
  the returned `(fx, fy, fz)` are zeros — use it for visualisation-only
  or LES-validation runs).

### 2.7 `isfinite_check_period` — never check every step in production

- **Symptom:** post-fusion wall time barely improves; profile shows a
  device sync dominating each step.
- **Root cause:** `torch.isfinite(f).all().item()` is a full-grid
  reduction + host sync: **~0.6 ms per step at n=128**, which was
  **57%** of the post-fusion wall time.
- **Fix:** `SuboffCmkKbcConfig.isfinite_check_period` (default **10**)
  runs the check every 10 steps plus the final step.  Set it to `1`
  only when actively debugging a NaN.

### 2.8 rsync BOTH `triton_fused*` locations

- **Symptom:** you fix a bug, re-run, and the old behaviour persists —
  or the import resolves to a stale module.
- **Root cause:** the companion Triton modules exist in **two** live
  locations on the box, and the wrapper imports whichever comes first on
  `sys.path` (benchmarks do `sys.path.insert(0, '/nfs/wangxi')`, so the
  top-level copies usually win):
  - `/nfs/wangxi/tensorlbm_triton_fused_distributed.py`,
    `/nfs/wangxi/tensorlbm_triton_fused_obstacle.py`
  - `/nfs/wangxi/TensorLBM/src/tensorlbm/triton_fused_distributed.py`,
    `/nfs/wangxi/TensorLBM/src/tensorlbm/triton_fused*.py`
- **Fix:** always rsync **both** trees in the same sync
  (`triton_fused*` in `/nfs/wangxi/` **and** in
  `TensorLBM/src/tensorlbm/`); also refresh any copies under
  `/tmp/suboff_test/` that benchmarks import.

## 3) Quickstart

The 5090 box's `torchrun` is **not** on the default `PATH`; use
`/home/LBM/bin/torchrun` (it wraps the venv python).  All commands below
are single-node multi-GPU.

### 3.1 Smoke script

Save as e.g. `/tmp/suboff_test/dist_smoke.py`:

```python
import os, sys, types

sys.path.insert(0, '/nfs/wangxi')
sys.path.insert(0, '/nfs/wangxi/TensorLBM/src')

if 'torch_sdaa' not in sys.modules:
    sys.modules['torch_sdaa'] = types.ModuleType('torch_sdaa')
import numpy as np
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

import torch
from tensorlbm.suboff_cmk_kbc_runner import SuboffCmkKbcConfig, run_suboff_cmk_kbc

rank = int(os.environ['RANK'])
world = int(os.environ['WORLD_SIZE'])

cfg = SuboffCmkKbcConfig(
    re=2_000_000.0, collision='BGK', turbulence_model='smagorinsky',
    nx=128, ny=128, nz=128,          # nz MUST be divisible by world_size
    n_steps=200, u_in=0.05, hull_length=0.6 * 128,
    device=f'cuda:{rank}',
    use_triton_step=True,
    use_triton_distributed=True,     # requires world_size > 1
    world_size=world, rank=rank,
    isfinite_check_period=10,        # see §2.7 — do not set 1 in production
    compute_force=True,
)
run_suboff_cmk_kbc(cfg)
```

### 3.2 Launch commands

```bash
# 2 GPUs
/home/LBM/bin/torchrun --standalone --nproc_per_node=2 /tmp/suboff_test/dist_smoke.py

# 4 GPUs
/home/LBM/bin/torchrun --standalone --nproc_per_node=4 /tmp/suboff_test/dist_smoke.py

# all 8 GPUs
/home/LBM/bin/torchrun --standalone --nproc_per_node=8 /tmp/suboff_test/dist_smoke.py
```

`--standalone` makes torchrun pick a free rendezvous port on localhost
(no external master needed).  If your torchrun predates `--standalone`,
use `--master_addr=127.0.0.1 --master_port=29500` instead.

The process group itself is initialised lazily:
`init_distributed()` reads `RANK`/`WORLD_SIZE` from the environment and
picks the `nccl` backend when CUDA is visible
(`triton_suboff_step_distributed` also calls it defensively — it is a
no-op if the group already exists).

### 3.3 Regression tests

The slab solver's own test module (single-GPU fallback when run without
torchrun; NCCL path when run with it):

```bash
cd /nfs/wangxi/TensorLBM
PYTHONPATH=src /home/LBM/bin/torchrun --standalone --nproc_per_node=8 \
    -m pytest tests/test_triton_fused_distributed.py -v
```

## 4) References

| Module | Role |
|---|---|
| `/nfs/wangxi/TensorLBM/src/tensorlbm/triton_fused_distributed.py` | Slab decomposition + NCCL halo exchange (`DistributedTritonFusedSolver3D`, `init_distributed`) |
| `/nfs/wangxi/TensorLBM/src/tensorlbm/triton_suboff_step_distributed.py` | Distributed SUBOFF step (`TritonSuboffDistributedRunner`): BC, mass correction, fused force, tau_eff plumbing |
| `/nfs/wangxi/TensorLBM/src/tensorlbm/triton_fused.py` | Single-GPU fused kernel; `_Q_PAD = 32`; `DEFAULT_BLOCK_*` |
| `/nfs/wangxi/TensorLBM/src/tensorlbm/triton_fused_obstacle.py` | Obstacle/LES kernels: V2 `triton_fused_obstacle_xfar_les` (Q=19), legacy `triton_fused_obstacle_les` (Q=32), `apply_far_field_bc_6face`, `apply_mass_correction` |
| `/nfs/wangxi/tensorlbm_triton_fused_distributed.py`, `/nfs/wangxi/tensorlbm_triton_fused_obstacle.py` | Top-level companion copies imported first by the benchmarks — keep in sync (§2.8) |
| `/nfs/wangxi/TensorLBM/src/tensorlbm/suboff_cmk_kbc_runner.py` | Production runner: `use_triton_step`, `use_triton_distributed`, `isfinite_check_period`, `compute_force`, force `all_reduce` |
| `/nfs/wangxi/TensorLBM/src/tensorlbm/suboff_resistance.py` | `_voxel_wetted_area` (global mask → `dynamic_pressure`; why the all-reduce in §2.3 is needed) |
| `/nfs/wangxi/TensorLBM/tests/test_triton_fused.py` | Single-GPU fused-kernel tests |
