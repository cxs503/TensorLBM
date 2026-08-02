"""Debug q computation for a specific boundary link."""

from __future__ import annotations
import math, torch
import numpy as np
from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask, SuboffHullType

nx, ny, nz = 64, 32, 32
cx, cy, cz = nx * 0.35, ny / 2.0, nz / 2.0
hull_length = 0.6 * nx
config = SuboffConfig()

solid, _ = build_suboff_mask(
    SuboffHullType.BARE_HULL,
    nx=nx,
    ny=ny,
    nz=nz,
    cx=cx,
    cy=cy,
    cz=cz,
    length=hull_length,
    device="cpu",
)

# Find a few boundary links
from tensorlbm.d3q19 import C as C19

c = C19.cpu().float()

for d in range(1, 19):
    dcx, dcy, dcz = int(c[d, 0].item()), int(c[d, 1].item()), int(c[d, 2].item())
    nb_solid = torch.roll(solid, shifts=(dcz, dcy, dcx), dims=(0, 1, 2))
    fluid_bdry = ~solid & nb_solid
    if fluid_bdry.any():
        indices = torch.nonzero(fluid_bdry)
        # Print first boundary link
        k, j, i = int(indices[0, 0].item()), int(indices[0, 1].item()), int(indices[0, 2].item())
        print(f"\nDirection d={d}: ({dcx}, {dcy}, {dcz})")
        print(f"Fluid cell: ({i}, {j}, {k}) solid={bool(solid[k, j, i].item())}")
        print(
            f"Neighbor: ({i + dcx}, {j + dcy}, {k + dcz}) solid={bool(solid[k + dcz, j + dcy, i + dcx].item())}"
        )

        # Check my surface distance function
        from tests.test_bfl_suboff import _suboff_surface_distance

        x_bow = cx - hull_length / 2.0

        # Fluid cell
        d_fluid = _suboff_surface_distance(
            float(i), float(j), float(k), cx, cy, cz, hull_length, config
        )
        xi_f = (float(i) - x_bow) / hull_length
        radius = config.r_over_l * hull_length
        r_f = math.sqrt((j - cy) ** 2 + (k - cz) ** 2)
        print(f"  Fluid: xi={xi_f:.6f} r={r_f:.3f} (max={radius:.3f}) dist={d_fluid:.6f}")

        # Neighbor cell
        d_nb = _suboff_surface_distance(
            float(i + dcx), float(j + dcy), float(k + dcz), cx, cy, cz, hull_length, config
        )
        xi_n = (float(i + dcx) - x_bow) / hull_length
        r_n = math.sqrt((j + dcy - cy) ** 2 + (k + dcz - cz) ** 2)
        print(f"  Neighbor: xi={xi_n:.6f} r={r_n:.3f} dist={d_nb:.6f}")

        # Check mask directly: is the neighbor cell center inside the hull?
        # xi at neighbor
        r_norm = config.r_over_l * hull_length
        from tensorlbm.suboff_cad import suboff_radius_profile

        r_norm_needed = suboff_radius_profile(np.array([xi_n]), config)[0]
        r_surf_needed = float(r_norm_needed) * r_norm
        print(
            f"  Neighbor surface radius at xi={xi_n:.6f}: r_surf={r_surf_needed:.6f}, actual r={r_n:.6f}"
        )
        print(f"  Mask says solid: {bool(solid[k + dcz, j + dcy, i + dcx].item())}")
        print(f"  Inside? r={r_n:.6f} <= r_surf={r_surf_needed:.6f}? {r_n <= r_surf_needed}")
        break
    break
