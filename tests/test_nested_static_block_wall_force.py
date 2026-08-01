"""Curved-wall ownership and force closure on the deepest static-AMR level."""
from __future__ import annotations

import torch

from tensorlbm.boundaries3d import sphere_mask
from tensorlbm.control_volume_force import (
    box_control_volume,
    observe_control_volume_force,
)
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.drag_pressure import get_near_wall_3d
from tensorlbm.interpolated_bc import compute_q_sphere
from tensorlbm.refinement import BoxRegion
from tensorlbm.solver3d import collide_mrt3d, stream3d
from tensorlbm.static_block_amr import (
    AMRAdvanceResult,
    NestedStaticBlockAMR3D,
    StaticBlockAMRConfig,
)
from tensorlbm.wall_model import bfl_wall_function_3d


def test_deepest_level_exclusively_owns_curved_wall_and_force() -> None:
    shape = (12, 12, 14)
    rho = torch.ones(shape)
    ux = torch.full(shape, 0.04)
    zero = torch.zeros(shape)
    coarse = equilibrium3d(rho, ux, zero, zero).float()
    outer = StaticBlockAMRConfig(
        BoxRegion(x0=3, x1=11, y0=3, y1=9, z0=3, z1=9),
        tau_coarse=0.56,
    )
    inner = StaticBlockAMRConfig(
        BoxRegion(x0=4, x1=14, y0=4, y1=10, z0=4, z1=10),
        tau_coarse=outer.tau_fine,
    )
    hierarchy = NestedStaticBlockAMR3D(coarse, (outer, inner))

    nz, ny, nx = hierarchy.finest_f.shape[1:]
    cx, cy, cz, radius = nx / 2.0, ny / 2.0, nz / 2.0, 2.5
    solid = sphere_mask(
        nx, ny, nz, cx, cy, cz, radius, device=torch.device("cpu"),
    )
    links, q = compute_q_sphere(
        nx, ny, nz, cx, cy, cz, radius, device=torch.device("cpu"),
    )
    near = get_near_wall_3d(solid)
    control_volume = box_control_volume(
        (nz, ny, nx),
        x0=int(cx - radius) - 2,
        x1=int(cx + radius) + 3,
        y0=int(cy - radius) - 2,
        y1=int(cy + radius) + 3,
        z0=int(cz - radius) - 2,
        z1=int(cz + radius) + 3,
    )
    paired_forces: list[tuple[float, float]] = []
    wall_calls_by_level = [0, 0, 0]

    def advance(
        state: torch.Tensor,
        tau: float,
        level: int,
        substep: int,
    ) -> AMRAdvanceResult:
        del substep
        post_collision = collide_mrt3d(state, tau=tau)
        out = stream3d(post_collision)
        if level == 2:
            wall_calls_by_level[level] += 1
            out, friction, pressure = bfl_wall_function_3d(
                out,
                post_collision,
                solid,
                0.002,
                links,
                q,
                wall_law="musker",
                near_mask=near,
                bfl_wall_mode="wall_model_slip",
                wall_activation=1.0,
            )
            cv_force = float(observe_control_volume_force(
                state,
                out,
                post_collision,
                control_volume,
                solid=solid,
            ).force_on_body[0])
            paired_forces.append((cv_force, pressure + friction))
        return AMRAdvanceResult(out, post_collision)

    maximum_reflux_residual = 0.0
    for _ in range(3):
        ledgers = hierarchy.step(advance)
        maximum_reflux_residual = max(
            maximum_reflux_residual,
            *(float(ledger.residual.abs().max()) for ledger in ledgers),
        )

    assert wall_calls_by_level == [0, 0, 12]
    assert len(paired_forces) == 12
    for cv_force, bfl_plus_stress in paired_forces:
        difference_pct = (
            abs(cv_force - bfl_plus_stress)
            / max(abs(cv_force), 1.0e-30)
            * 100.0
        )
        assert difference_pct < 0.002
    assert maximum_reflux_residual < 2.0e-10

