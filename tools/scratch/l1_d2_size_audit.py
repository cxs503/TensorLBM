"""Static analysis: exact sizes for the L1-block path (sphere 96x64x64, R6).
Runs ONLY on CPU (no GPU). Mirrors octree_integrated_validate.py --l1-block.
"""
import sys
sys.path.insert(0, "/root/TensorLBM_feat2/src")
import torch
from tensorlbm.octree_boundary.geometry import build_octree_shell, sphere_distance_field
from tensorlbm.octree_boundary.geometry_adapters import sphere_inside_fn
from tensorlbm.amr_shell_planning import plan_body_shell_box
from tensorlbm.octree_boundary.l1_block import L1BlockDistributed, build_window_indices
from tensorlbm.octree_boundary.stepping import build_ghost_plan, restrict_shell_to_block
from tensorlbm.octree_boundary.distributed_stepping import (
    interleaved_leaf_indices, _slice_ghost_plan_by_indices,
)
from tensorlbm.octree_boundary.sharding import _slice_ghost_plan
from tensorlbm.refinement import BoxRegion

nx, ny, nz = 96, 64, 64
radius = 6.0
center = (nx * 0.5, ny * 0.5, nz * 0.5)
bl = max(2.0, round(radius / 2.0))
dev = torch.device("cpu")

solid_coarse = sphere_distance_field((nz, ny, nx), center, radius, dev) <= 0.0
print(f"bl={bl} solid_coarse cells={int(solid_coarse.sum())}")

plan = plan_body_shell_box(solid_coarse, shell_margin=6, wake_cells=32, pad=8)
box = plan.box
print(f"box z:[{box.z0},{box.z1}) y:[{box.y0},{box.y1}) x:[{box.x0},{box.x1})")
l1_shape = ((box.z1 - box.z0) * 2, (box.y1 - box.y0) * 2, (box.x1 - box.x0) * 2)
center_l1 = (center[0] * 2.0 - box.x0 * 2, center[1] * 2.0 - box.y0 * 2,
             center[2] * 2.0 - box.z0 * 2)
radius_l1 = radius * 2.0
print(f"l1_shape={l1_shape} center_l1={center_l1} radius_l1={radius_l1}")

# window (box + 1-ring), d_max-independent
win = build_window_indices((nz, ny, nx), box, dev)
n_win = win.cells.shape[0]
print(f"n_win (box+ring cells) = {n_win}  window_shape={win.shape}")

for d_max in (1, 2):
    print(f"\n================ d_max={d_max} ================")
    octree = build_octree_shell(
        l1_shape, center=center_l1, radius=radius_l1,
        bl_thickness_cells=bl, d_max=d_max, lattice="D3Q27", device=dev,
    )
    n_leaf = octree.n_leaf
    n_link = int(octree.interface_links.shape[0])
    stats = octree.stats
    print(f"n_leaf={n_leaf} stats={ {k: stats.get(k) for k in ('n_leaf_l1','n_leaf_l2','n_interface_links')} }")
    print(f"interface_links (n_ghost) = {n_link}  (27*{n_leaf} = {27*n_leaf})")
    lev = octree.leaf_level
    print(f"leaf_level: depth1={int((lev==1).sum())} depth2={int((lev==2).sum())} "
          f"(n_leaf=={int(lev.numel())})")
    host = octree.leaf_host_cell
    n_cells = int(torch.unique(host, dim=0).shape[0])
    print(f"covered L1 cells n_cells = {n_cells}")
    # ghost plan sizes
    gp = build_ghost_plan(octree, tuple(octree.meta["shape"]))
    print(f"ghost plan: n_ghost={gp.n_ghost} slot={tuple(gp.slot.shape)}")
    for ws in (8, 16):
        n_local = (n_leaf + ws - 1) // ws
        lidx = interleaved_leaf_indices(n_leaf, ws, 0)
        gp_loc, grows = _slice_ghost_plan_by_indices(gp, lidx, lidx.shape[0])
        print(f"  ws={ws}: rank0 local leaves={lidx.shape[0]} local ghost rows={gp_loc.n_ghost} "
              f"slot={tuple(gp_loc.slot.shape)} rows_kept_frac={gp_loc.n_ghost/max(n_link,1):.4f}")
    # restriction sizes
    q = octree.Q
    f_leaf = torch.ones(q, n_leaf, dtype=torch.float32)
    from tensorlbm.octree_boundary.stepping import _tau_chain
    taus = _tau_chain(0.55, d_max)
    restricted, cells = restrict_shell_to_block(octree, f_leaf, taus)
    print(f"restricted={(q, cells.shape[0])} cells={tuple(cells.shape)} "
          f"cells_bytes={cells.numel()*cells.element_size()/1e6:.3f}MB "
          f"restricted_bytes={restricted.numel()*restricted.element_size()/1e6:.3f}MB")
    # message-size table
    qb = q * 4  # float32
    print(f"--- msg sizes (float32, Q={q}) ---")
    print(f"  substep all_gather (Q,n_leaf) total = {q*n_leaf*4/1e6:.2f}MB "
          f"chunk_cols={3*1024*1024//qb} nchunks={-(-n_leaf//(3*1024*1024//qb))} "
          f"per_msg={qb*(3*1024*1024//qb)/1e6:.3f}MB")
    print(f"  window gather per chunk = {qb*(3*1024*1024//qb)/1e6:.3f}MB "
          f"nchunks(win)={-(-n_win//(3*1024*1024//qb))} x3 gathers")
    print(f"  cells broadcast UNCHUNKED = {cells.numel()*8/1e6:.3f}MB  (>4MB? {cells.numel()*8 > 4*1024*1024})")
    print(f"  restricted broadcast total = {restricted.numel()*4/1e6:.3f}MB nchunks={-(-cells.shape[0]//(3*1024*1024//qb))}")
    l1_phys = octree.meta["shape"]
    l1_elems = q * l1_phys[0] * l1_phys[1] * l1_phys[2]
    print(f"  l1_f broadcast total = {l1_elems*4/1e6:.3f}MB nchunks={-(-l1_elems//(3*1024*1024//4))}")
