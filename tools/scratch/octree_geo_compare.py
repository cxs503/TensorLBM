#!/usr/bin/env python3
"""CPU geometry comparison: single-card AMR octree (bl=6) vs integrated L1 octree (bl=3).

Reproduces both build paths exactly:
  single-card: examples/octree_sphere_validate.py  (sphere_mask coarse solid,
               plan_body_shell_box -> box1, build_fine_block_geometry -> s1/fc1/radius1,
               phys_center = fc1 - GHOST, bl_cells = max(2, round(radius_l1/2)) = 6)
  integrated : examples/octree_integrated_validate.py --l1-block
               (sphere_distance_field<=0 coarse solid, same plan_body_shell_box,
               l1_shape = box*2, center_l1 = center*2 - origin*2, bl = 3)
Isolates the bl effect by cross-building bl=3 / bl=6 on each grid.
"""
import sys, json
sys.path.insert(0, "/root/TensorLBM_feat2/src")
import torch
from tensorlbm.octree_boundary.geometry import (
    build_octree_shell, sphere_distance_field,
    SHELL_OUTSIDE, SOLID, DOMAIN_OUT, FANOUT,
)
from tensorlbm.amr_shell_planning import plan_body_shell_box
from tensorlbm.sphere_amr_common import build_sphere_geometry, build_fine_block_geometry

torch.manual_seed(0)
dev = torch.device("cpu")
nx, ny, nz = 96, 64, 64
radius = 6.0
center = (nx * 0.5, ny * 0.5, nz * 0.5)
GHOST, RATIO = 1, 2

def summarize(tag, o):
    lev = o.leaf_level
    nt = o.neighbor_table
    n_bfl = int(o.bfl_mask.sum().item())
    stats = o.stats
    n_solid = int((nt == SOLID).sum().item())
    n_out = int((nt == SHELL_OUTSIDE).sum().item())
    n_dom = int((nt == DOMAIN_OUT).sum().item())
    n_fan = int((nt == FANOUT).sum().item())
    host = o.leaf_host_cell
    n_cov = int(torch.unique(host, dim=0).shape[0])
    # BFL links only (wall links, upstream-opp stored in interface_links too)
    print(f"{tag}:")
    print(f"  n_leaf={o.n_leaf}  L1={stats.get('n_leaf_l1')}  L2={stats.get('n_leaf_l2')}")
    print(f"  n_shell_cells={stats.get('n_shell_cells')}  n_cov_L1_cells={n_cov}")
    print(f"  n_bfl_links={n_bfl}  n_interface_links={stats.get('n_interface_links')}")
    print(f"  neigh: solid={n_solid} shell_out={n_out} dom_out={n_dom} fanout={n_fan}")
    print(f"  vol_err={stats.get('volume_error'):.4f}  leaf_vol={stats.get('leaf_volume'):.1f}")
    print(f"  meta: bl={o.meta['bl_thickness_cells']} delta={o.meta['delta_mask']} "
          f"shape={o.meta['shape']} center={o.meta['center']} r={o.meta['radius']}")
    return dict(tag=tag, n_leaf=o.n_leaf, n_l1=stats.get('n_leaf_l1'), n_l2=stats.get('n_leaf_l2'),
                n_shell=stats.get('n_shell_cells'), n_cov=n_cov, n_bfl=n_bfl,
                n_iface=stats.get('n_interface_links'),
                n_solid=n_solid, n_out=n_out, n_dom=n_dom, n_fan=n_fan,
                vol_err=stats.get('volume_error'),
                bl=o.meta['bl_thickness_cells'], delta=o.meta['delta_mask'],
                shape=o.meta['shape'], center=o.meta['center'])

results = {}

# ================= 1. SINGLE-CARD path (octree_sphere_validate.py) ==========
print("========== single-card path (octree_sphere_validate.py) ==========")
solid_coarse_s, _ = build_sphere_geometry(nx, ny, nz, *center, radius, dev, lattice="D3Q27")
print(f"single-card coarse solid cells (sphere_mask) = {int(solid_coarse_s.sum())}")
plan1 = plan_body_shell_box(solid_coarse_s, shell_margin=6, wake_cells=32, pad=8)
box1 = plan1.box
print(f"box1 z:[{box1.z0},{box1.z1}) y:[{box1.y0},{box1.y1}) x:[{box1.x0},{box1.x1})")
s1, fc1, radius1, _l1 = build_fine_block_geometry(box1, center, radius, RATIO, GHOST, dev)
print(f"s1={s1} fc1={tuple(float(v) for v in fc1)} radius1={radius1}")
phys_center = (float(fc1[0] - GHOST), float(fc1[1] - GHOST), float(fc1[2] - GHOST))
print(f"phys_center={phys_center}")
bl_single = max(2.0, round(radius1 / 2.0))
print(f"bl_cells (single-card) = max(2, round({radius1}/2)) = {bl_single}")

for lat in ("D3Q27", "D3Q19"):
    o = build_octree_shell(s1, phys_center, radius1, bl_thickness_cells=bl_single,
                           d_max=2, transition=1, lattice=lat, device=dev)
    results[f"single_bl6_{lat}"] = summarize(f"single-card bl=6 d_max=2 {lat}", o)
    o3 = build_octree_shell(s1, phys_center, radius1, bl_thickness_cells=3.0,
                            d_max=2, transition=1, lattice=lat, device=dev)
    results[f"single_bl3_{lat}"] = summarize(f"single-card bl=3 d_max=2 {lat} (bl isolated)", o3)

# ================= 2. INTEGRATED path (octree_integrated_validate.py L1) ===
print("\n========== integrated path (octree_integrated_validate.py --l1-block) ==========")
solid_coarse_i = sphere_distance_field((nz, ny, nx), center, radius, dev) <= 0.0
print(f"integrated coarse solid cells (dist-field<=0) = {int(solid_coarse_i.sum())}")
plan = plan_body_shell_box(solid_coarse_i, shell_margin=6, wake_cells=32, pad=8)
box = plan.box
print(f"box z:[{box.z0},{box.z1}) y:[{box.y0},{box.y1}) x:[{box.x0},{box.x1})")
l1_shape = ((box.z1 - box.z0) * 2, (box.y1 - box.y0) * 2, (box.x1 - box.x0) * 2)
center_l1 = (center[0] * 2.0 - box.x0 * 2, center[1] * 2.0 - box.y0 * 2,
             center[2] * 2.0 - box.z0 * 2)
radius_l1 = radius * 2.0
bl_int = max(2.0, round(radius / 2.0))
print(f"l1_shape={l1_shape} center_l1={center_l1} radius_l1={radius_l1} bl={bl_int}")

o = build_octree_shell(l1_shape, center=center_l1, radius=radius_l1,
                       bl_thickness_cells=bl_int, d_max=2, lattice="D3Q27", device=dev)
results["integ_bl3"] = summarize("integrated bl=3 d_max=2 D3Q27 (reference)", o)
o6 = build_octree_shell(l1_shape, center=center_l1, radius=radius_l1,
                        bl_thickness_cells=6.0, d_max=2, lattice="D3Q27", device=dev)
results["integ_bl6"] = summarize("integrated bl=6 d_max=2 D3Q27 (bl isolated)", o6)

print("\n==================== bl effect on n_shell_cells (analytic band) ====================")
import math
for bl in (3.0, 6.0):
    delta = bl + 1.0
    # AABB band: centres within [R - sqrt(3)/2, R + delta + sqrt(3)/2]
    r_in = max(0.0, 12.0 - 0.5 * math.sqrt(3.0))
    r_out = 12.0 + delta + 0.5 * math.sqrt(3.0)
    vol = 4.0 / 3.0 * math.pi * (r_out ** 3 - r_in ** 3)
    print(f"bl={bl}: delta_mask={delta} band ~[{r_in:.2f},{r_out:.2f}] "
          f"analytic shell-cell vol ~ {vol:.0f}  (ratio vs bl=3: {vol/(4/3*math.pi*((12+4+0.87)**3-(12-0.87)**3)):.2f})")

with open("/tmp/octree_geo_compare.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nsaved /tmp/octree_geo_compare.json")
