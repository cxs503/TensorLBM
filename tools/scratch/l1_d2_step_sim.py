"""In-process CPU simulation of the L1 d_max=2 root step with a fake dist.

Replicates octree_integrated_validate.py --l1-block main-loop step 1-6 with a
monkey-patched torch.distributed so every collective is a local no-op that
keeps the sum-trick consistent. Purpose: find where the d2 path stalls / what
sizes are involved, WITHOUT any GPU or real TCCL.

The fake all_gather assumes all ranks' pieces are identical content (we run a
single logical rank view), so sums are just piece * 1.
"""
import sys
sys.path.insert(0, "/root/TensorLBM_feat2/src")
import types
import time

import torch

# ---------------- fake dist ----------------
_fake_rank = 0
_fake_world = 16


def _ag(gathered, piece):
    for i in range(len(gathered)):
        gathered[i] = piece.clone()


def _bcast(t, src=0):
    pass


def _ar(t, op=None):
    pass


fake_dist = types.ModuleType("dist")
fake_dist.all_gather = _ag
fake_dist.broadcast = _bcast
fake_dist.all_reduce = _ar
fake_dist.get_rank = lambda: _fake_rank
fake_dist.get_world_size = lambda: _fake_world
fake_dist.ReduceOp = types.SimpleNamespace(SUM="SUM")
fake_dist.init_process_group = lambda *a, **k: None
fake_dist.destroy_process_group = lambda: None
sys.modules["torch.distributed"] = fake_dist
# `import torch.distributed as dist` resolves via the torch package attribute,
# so override that too (the sys.modules entry alone is bypassed for a.b imports)
torch.distributed = fake_dist
# distributed_stepping does `import torch.distributed as dist`
import torch.distributed as dist  # noqa: E402

# ---------------- geometry (identical to the example) ----------------
from tensorlbm.octree_boundary.geometry import (  # noqa: E402
    build_octree_shell, sphere_distance_field,
)
from tensorlbm.amr_shell_planning import plan_body_shell_box  # noqa: E402

nx, ny, nz = 96, 64, 64
radius = 6.0
center = (nx * 0.5, ny * 0.5, nz * 0.5)
bl = max(2.0, round(radius / 2.0))
dev = torch.device("cpu")

t0 = time.time()
solid_coarse = sphere_distance_field((nz, ny, nx), center, radius, dev) <= 0.0
plan = plan_body_shell_box(solid_coarse, shell_margin=6, wake_cells=32, pad=8)
box = plan.box
l1_shape = ((box.z1 - box.z0) * 2, (box.y1 - box.y0) * 2, (box.x1 - box.x0) * 2)
center_l1 = (center[0] * 2.0 - box.x0 * 2, center[1] * 2.0 - box.y0 * 2,
             center[2] * 2.0 - box.z0 * 2)
D_MAX = 2
octree = build_octree_shell(
    l1_shape, center=center_l1, radius=radius * 2.0,
    bl_thickness_cells=bl, d_max=D_MAX, lattice="D3Q27", device=dev,
)
print(f"[sim] octree built in {time.time()-t0:.1f}s n_leaf={octree.n_leaf}")

q = 27
u_in = 0.06
world_size = 16
rank = 0
nx_local = nx // world_size
lo = rank * nx_local
hi = lo + nx_local

# coarse slab
coarse_f = torch.zeros(q, nz, ny, nx_local + 2, device=dev)
from tensorlbm.d3q27 import equilibrium27
eq = equilibrium27(
    torch.ones(nz, ny, nx_local, device=dev),
    torch.full((nz, ny, nx_local), u_in, device=dev),
    torch.zeros(nz, ny, nx_local, device=dev),
    torch.zeros(nz, ny, nx_local, device=dev),
)
coarse_f[:, :, :, 1:-1] = eq
coarse_f[:, :, :, 0:1] = eq[:, :, :, 0:1]
coarse_f[:, :, :, -1:] = eq[:, :, :, -1:]

# leaf shard (interleaved)
from tensorlbm.octree_boundary.distributed_stepping import interleaved_leaf_indices
lidx = interleaved_leaf_indices(octree.n_leaf, world_size, rank)
n_local = lidx.shape[0]
oshape = tuple(octree.meta["shape"])
eq_global = equilibrium27(
    torch.ones(oshape, device=dev),
    torch.full(oshape, u_in, device=dev),
    torch.zeros(oshape, device=dev),
    torch.zeros(oshape, device=dev),
)
octree.f_leaf = eq_global[:, octree.leaf_host_cell[lidx, 0],
                          octree.leaf_host_cell[lidx, 1],
                          octree.leaf_host_cell[lidx, 2]].clone()
print(f"[sim] shard ready n_local={n_local}")

# tau chain
from tensorlbm.octree_boundary.stepping import _tau_chain
tau_coarse = 0.5 + 3.0 * (u_in * radius / 100.0)
taus = _tau_chain(tau_coarse, octree.d_max)
tau_l1 = taus[1]

# operators (copied from the example)
from tensorlbm.cumulant import collide_cumulant_d3q27
from tensorlbm.d3q27 import C as C27
S27 = [(int(C27[d, 0]), int(C27[d, 1]), int(C27[d, 2])) for d in range(27)]


def collide_coarse(f, tau):
    f4 = f.view(q, 1, 1, -1)
    return collide_cumulant_d3q27(f4, tau, C_s=0.0).view_as(f)


def stream27_roll(f):
    out = torch.empty_like(f)
    for d in range(27):
        sx, sy, sz = S27[d]
        out[d] = torch.roll(f[d], shifts=(sz, sy, sx), dims=(0, 1, 2))
    return out


def halo_exchange(f_local):
    left_interior = f_local[:, :, :, 1:2].contiguous()
    right_interior = f_local[:, :, :, -2:-1].contiguous()
    right_gather = [torch.empty_like(right_interior) for _ in range(world_size)]
    dist.all_gather(right_gather, right_interior)
    left_halo = right_gather[(rank - 1) % world_size]
    left_gather = [torch.empty_like(left_interior) for _ in range(world_size)]
    dist.all_gather(left_gather, left_interior)
    right_halo = left_gather[(rank + 1) % world_size]
    f_local[:, :, :, 0:1] = left_halo
    f_local[:, :, :, -1:] = right_halo


def advance_l1(f, tau):
    f4 = f.view(q, 1, 1, -1)
    return collide_cumulant_d3q27(f4, tau, C_s=0.0).view_as(f)


def advance_shell(f, tau, level, substep):
    f4 = f.view(q, 1, 1, -1)
    return collide_cumulant_d3q27(f4, tau, C_s=0.0).view_as(f)


# L1 block
from tensorlbm.octree_boundary.l1_block import (  # noqa: E402
    L1BlockDistributed, gather_window_chunked, restrict_l1_block_to_coarse,
    step_l1_block_distributed, write_window_back,
)
l1_block = L1BlockDistributed(
    box, (nz, ny, nx), tau_coarse, q=q, ratio=2, ghost=1,
    device=dev, solid_l1=octree._solid,
    collide_fn=advance_l1, stream_fn=stream27_roll,
)
l1_block.initialize_uniform(u_in)
win = l1_block.win
print(f"[sim] L1 block ready shape={l1_block.l1_shape} n_win={win.cells.shape[0]}")

from tensorlbm.octree_boundary.bfl import bfl_apply_gather, leaf_force_weights
leaf_weights = leaf_force_weights(octree).to(dev)[lidx]


def bfl_fn(octree_, out, post, gplan, ghost_vals, *, substep):
    return bfl_apply_gather(
        octree_, out, post, ghost_plan=gplan, ghost_vals=ghost_vals,
        force_weights=leaf_weights, return_force=True, q_min=None,
    )


from tensorlbm.octree_boundary.distributed_stepping import step_octree_shell_distributed

# ---------------- one root step ----------------
print("[sim] step 1 start")
t0 = time.time()
coarse_old = coarse_f.clone()
t1 = time.time()
post = collide_coarse(coarse_f, tau_coarse)
print(f"[sim] collide_coarse {time.time()-t1:.2f}s")
t1 = time.time()
halo_exchange(post)
print(f"[sim] halo_exchange {time.time()-t1:.2f}s")
t1 = time.time()
streamed = stream27_roll(post)
coarse_f = streamed  # skip far-field for the sim
print(f"[sim] stream {time.time()-t1:.2f}s")

t1 = time.time()
cw_old, _ = gather_window_chunked(coarse_old, win, lo, hi, rank=rank, world_size=world_size)
print(f"[sim] cw_old gather {time.time()-t1:.2f}s")
t1 = time.time()
cw_new, in_slab = gather_window_chunked(coarse_f, win, lo, hi, rank=rank, world_size=world_size)
print(f"[sim] cw_new gather {time.time()-t1:.2f}s")
t1 = time.time()
cw_post, _ = gather_window_chunked(post, win, lo, hi, rank=rank, world_size=world_size)
print(f"[sim] cw_post gather {time.time()-t1:.2f}s")

t1 = time.time()
l1_phys_pre, l1_posts_phys, _pg = step_l1_block_distributed(l1_block, cw_old, cw_new)
print(f"[sim] L1 stage {time.time()-t1:.2f}s")
t1 = time.time()
l1_f_phys = l1_block.physical_copy()
_ledger, local_mem, restricted, cells = step_octree_shell_distributed(
    octree, advance_shell, l1_phys_pre, l1_f_phys,
    tau_coarse=l1_block.tau_l1, l1_post=l1_posts_phys,
    ghost_plan=None, bfl_fn=bfl_fn, rank=rank, world_size=world_size,
    reflux=True, interleave=True,
)
print(f"[sim] shell stage {time.time()-t1:.2f}s")
l1_block.set_physical(l1_f_phys)
t1 = time.time()
ledger = restrict_l1_block_to_coarse(l1_block, cw_new, cw_post)
print(f"[sim] L1->coarse restrict+reflux {time.time()-t1:.2f}s")
write_window_back(coarse_f, cw_new, win, in_slab, lo)
print(f"[sim] root step total {time.time()-t0:.2f}s — DONE, ledger residual={ledger.mass_residual:.3e}")
