#!/usr/bin/env python3
"""CPU quick check: SUBOFF (inside_fn) level-2 leaves + sphere regression.

Mirrors examples/octree_integrated_validate.py SUBOFF L1-block path but on
CPU and without MPI.  Prints n_leaf_l1 / n_leaf_l2 for both geometries.
"""
import time

import torch

from tensorlbm.amr_shell_planning import plan_body_shell_box
from tensorlbm.octree_boundary.geometry import build_octree_shell
from tensorlbm.octree_boundary.geometry_adapters import solid_mask_inside_fn
from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask


def run_sphere():
    shape = (48, 48, 48)
    center = (24.0, 24.0, 24.0)
    radius = 6.0
    t0 = time.time()
    grid = build_octree_shell(
        shape, center=center, radius=radius,
        bl_thickness_cells=4.0, d_max=2, lattice="D3Q27",
        device=torch.device("cpu"),
    )
    dt = time.time() - t0
    print(
        f"[sphere] R={radius} d_max=2 -> "
        f"n_leaf_l1={grid.stats['n_leaf_l1']} "
        f"n_leaf_l2={grid.stats['n_leaf_l2']} "
        f"n_leaf={grid.n_leaf}  ({dt:.2f}s)",
        flush=True,
    )
    return grid


def run_suboff():
    nx, ny, nz = 96, 48, 48
    config = SuboffConfig()
    L = 40.0
    srad = config.r_over_l * L
    solid_cpu, _ = build_suboff_mask(
        hull_type="bare_hull", nx=nx, ny=ny, nz=nz,
        cx=nx * 0.5, cy=ny * 0.5, cz=nz * 0.5,
        length=L, radius=srad, config=config, device="cpu",
    )
    bl = max(2.0, round(srad / 2.0))
    solid_coarse = solid_cpu.bool()
    plan = plan_body_shell_box(
        solid_coarse, shell_margin=6, wake_cells=24, pad=8,
    )
    box = plan.box
    l1_shape = (
        (box.z1 - box.z0) * 2, (box.y1 - box.y0) * 2,
        (box.x1 - box.x0) * 2,
    )
    radius_l1 = max(srad, bl * 2) * 2.0  # dummy sphere params (unused)

    def _suboff_inside_l1(centers):
        # Leaf centres are in L1-local world units; map back to the coarse
        # mask frame (mask index = box origin + local/2).
        coarse = 0.5 * centers + torch.tensor(
            (box.x0, box.y0, box.z0), dtype=torch.float64,
            device=centers.device,
        )
        return solid_mask_inside_fn(solid_cpu.bool(), device=torch.device("cpu"))(coarse)

    t0 = time.time()
    octree = build_octree_shell(
        l1_shape, center=(0.0, 0.0, 0.0), radius=radius_l1,
        bl_thickness_cells=bl, d_max=2, lattice="D3Q27",
        device=torch.device("cpu"),
        inside_fn=_suboff_inside_l1,
    )
    dt = time.time() - t0
    print(
        f"[suboff] l1_shape={l1_shape} bl={bl} srad={srad:.2f} -> "
        f"n_leaf_l1={octree.stats['n_leaf_l1']} "
        f"n_leaf_l2={octree.stats['n_leaf_l2']} "
        f"n_leaf={octree.n_leaf}  ({dt:.2f}s)",
        flush=True,
    )
    return octree


if __name__ == "__main__":
    run_sphere()
    run_suboff()
