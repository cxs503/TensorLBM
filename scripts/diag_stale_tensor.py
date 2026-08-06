#!/usr/bin/env python3
"""Verify: does the cached l1_fine reference track the module's fine_f after amr.step?"""
import torch
from tensorlbm.amr_shell_planning import plan_body_shell_box
from tensorlbm.cumulant import collide_cumulant_d3q19
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.solver3d import stream3d
from tensorlbm.sphere_amr_common import build_fine_block_geometry, build_sphere_geometry
from tensorlbm.static_block_amr import AMRAdvanceResult, NestedStaticBlockAMR3D, StaticBlockAMRConfig

device = torch.device("cpu")
shape = (64, 64, 96)
cx, cy, cz = 48.0, 32.0, 32.0
solid_coarse, solid_coarse_q = build_sphere_geometry(96, 64, 64, cx, cy, cz, 6.0, device)
plan = plan_body_shell_box(solid_coarse, 6, 32, pad=8)
box1 = plan.box
rho = torch.ones(shape, device=device)
ux = torch.full_like(rho, 0.06)
zero = torch.zeros_like(rho)
coarse_f = equilibrium3d(rho, ux, zero, zero, device=device)
config1 = StaticBlockAMRConfig(box1, tau_coarse=0.6, reflux=True)
amr = NestedStaticBlockAMR3D(coarse_f, (config1,), fine_solids=(None,))

l1_fine = amr.interfaces[0].fine_f   # the cached reference (as in run_case)
print("same object at init:", l1_fine is amr.interfaces[0].fine_f)
l1_fine[0, 5, 5, 5] = 123.0
print("module sees mutation:", amr.interfaces[0].fine_f[0, 5, 5, 5].item())

def advance(f, tau, level, substep):
    if level == 0:
        return AMRAdvanceResult(stream3d(f), f)
    collided = collide_cumulant_d3q19(f, tau, C_s=0.0)
    return AMRAdvanceResult(stream3d(collided), collided)

amr.step(advance)
print("cached is module after step:", l1_fine is amr.interfaces[0].fine_f)
print("cached data_ptr:", l1_fine.data_ptr(), " module data_ptr:", amr.interfaces[0].fine_f.data_ptr())
print("cached[0,5,5,5]:", l1_fine[0, 5, 5, 5].item(), " module[0,5,5,5]:", amr.interfaces[0].fine_f[0, 5, 5, 5].item())
print("cached max abs diff vs module:", float((l1_fine - amr.interfaces[0].fine_f).abs().max().item()))
