"""Unsharded reference stepper on CPU for d2 (no dist) — check for NaN."""
import sys
sys.path.insert(0, "/root/TensorLBM_feat2/src")
import time
import torch
from tensorlbm.octree_boundary.geometry import build_octree_shell, sphere_distance_field
from tensorlbm.amr_shell_planning import plan_body_shell_box
from tensorlbm.octree_boundary.stepping import step_octree_shell, _tau_chain
from tensorlbm.cumulant import collide_cumulant_d3q27
from tensorlbm.d3q27 import equilibrium27, C as C27

nx, ny, nz = 96, 64, 64
radius = 6.0
center = (nx * 0.5, ny * 0.5, nz * 0.5)
bl = max(2.0, round(radius / 2.0))
dev = torch.device("cpu")
u_in = 0.06
q = 27

solid_coarse = sphere_distance_field((nz, ny, nx), center, radius, dev) <= 0.0
plan = plan_body_shell_box(solid_coarse, shell_margin=6, wake_cells=32, pad=8)
box = plan.box
l1_shape = ((box.z1 - box.z0) * 2, (box.y1 - box.y0) * 2, (box.x1 - box.x0) * 2)
center_l1 = (center[0] * 2.0 - box.x0 * 2, center[1] * 2.0 - box.y0 * 2,
             center[2] * 2.0 - box.z0 * 2)

for D_MAX in (1, 2):
    t0 = time.time()
    octree = build_octree_shell(
        l1_shape, center=center_l1, radius=radius * 2.0,
        bl_thickness_cells=bl, d_max=D_MAX, lattice="D3Q27", device=dev,
    )
    print(f"\n==== d_max={D_MAX} n_leaf={octree.n_leaf} (build {time.time()-t0:.0f}s) ====")
    oshape = tuple(octree.meta["shape"])
    l1_f = equilibrium27(
        torch.ones(oshape, device=dev),
        torch.full(oshape, u_in, device=dev),
        torch.zeros(oshape, device=dev),
        torch.zeros(oshape, device=dev),
    )
    l1_old = l1_f.clone()
    octree.f_leaf = l1_f[:, octree.leaf_host_cell[:, 0],
                         octree.leaf_host_cell[:, 1],
                         octree.leaf_host_cell[:, 2]].clone()
    tau_coarse = 0.5 + 3.0 * (u_in * radius / 100.0)
    taus = _tau_chain(tau_coarse, octree.d_max)

    def advance_shell(f, tau, level, substep):
        f4 = f.view(q, 1, 1, -1)
        return collide_cumulant_d3q27(f4, tau, C_s=0.0).view_as(f)

    for step in range(1, 4):
        t1 = time.time()
        l1_post = l1_f.clone()  # post-collision placeholder (uniform)
        ledger = step_octree_shell(
            octree, advance_shell, l1_old, l1_f,
            tau_coarse=tau_coarse, l1_post=l1_post,
            shell_level=1, reflux=True,
            bfl_fn=None,
        )
        fin = bool(torch.isfinite(octree.f_leaf).all())
        print(f"  step {step}: {time.time()-t1:.1f}s f_leaf finite={fin} "
              f"sum={float(octree.f_leaf.sum()):.6g} residual={ledger.mass_residual:.3e}")
        l1_old = l1_f.clone()
        if not fin:
            break
