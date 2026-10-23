from __future__ import annotations

import pytest
import torch

from tensorlbm.boundaries3d import sphere_mask
from tensorlbm.control_volume_force import (
    box_control_volume,
    fluid_momentum_change,
    observe_control_volume_force,
)
from tensorlbm.cumulant import collide_cumulant_d3q19
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.drag_pressure import get_near_wall_3d
from tensorlbm.interpolated_bc import compute_q_sphere
from tensorlbm.solver3d import stream3d
from tensorlbm.wall_model import (
    bfl_wall_function_3d,
    compute_bfl_link_normal,
)


def test_bfl_slip_pressure_plus_guo_shear_closes_control_volume_force() -> None:
    """Independent force observers must close even during one transient step."""
    shape = (28, 28, 36)
    nz, ny, nx = shape
    cx = cy = cz = 14.0
    radius = 4.0
    solid = sphere_mask(
        nx,
        ny,
        nz,
        cx,
        cy,
        cz,
        radius,
        device=torch.device("cpu"),
    )
    bfl_mask, bfl_q = compute_q_sphere(
        nx,
        ny,
        nz,
        cx,
        cy,
        cz,
        radius,
        device=torch.device("cpu"),
    )
    rho = torch.ones(shape, dtype=torch.float64)
    ux = torch.full(shape, 0.04, dtype=torch.float64)
    zero = torch.zeros_like(rho)
    old = equilibrium3d(rho, ux, zero, zero)
    post_collision = old.clone()
    streamed = stream3d(post_collision)
    near = get_near_wall_3d(solid)
    normals = compute_bfl_link_normal(bfl_mask)
    updated, friction, pressure = bfl_wall_function_3d(
        streamed,
        post_collision,
        solid,
        1.0e-4,
        bfl_mask,
        bfl_q,
        wall_law="musker",
        near_mask=near,
        bfl_wall_mode="wall_model_slip",
        wall_activation=1.0,
        wall_normals=normals,
    )
    control_volume = box_control_volume(
        shape,
        x0=7,
        x1=22,
        y0=7,
        y1=22,
        z0=7,
        z1=22,
    )
    cv_force = float(
        observe_control_volume_force(
            old,
            updated,
            post_collision,
            control_volume,
            solid=solid,
        ).force_on_body[0]
    )

    combined = pressure + friction
    assert abs(combined - cv_force) / abs(cv_force) < 1.0e-6


def test_developed_nonequilibrium_wall_force_closes_each_step() -> None:
    """Closure must persist beyond the equilibrium first-step special case."""
    shape = (24, 24, 32)
    nz, ny, nx = shape
    cx, cy, cz, radius = 12.0, 12.0, 12.0, 4.0
    solid = sphere_mask(
        nx,
        ny,
        nz,
        cx,
        cy,
        cz,
        radius,
        device=torch.device("cpu"),
    )
    bfl_mask, bfl_q = compute_q_sphere(
        nx,
        ny,
        nz,
        cx,
        cy,
        cz,
        radius,
        device=torch.device("cpu"),
    )
    bfl_q = bfl_q.to(torch.float64)
    rho = torch.ones(shape, dtype=torch.float64)
    ux = torch.full(shape, 0.04, dtype=torch.float64)
    zero = torch.zeros_like(rho)
    f = equilibrium3d(rho, ux, zero, zero)
    solid_q = solid.unsqueeze(0)
    near = get_near_wall_3d(solid)
    normals = tuple(component.to(torch.float64) for component in compute_bfl_link_normal(bfl_mask))
    control_volume = box_control_volume(
        shape,
        x0=5,
        x1=20,
        y0=5,
        y1=20,
        z0=5,
        z1=20,
    )

    for _ in range(8):
        old = f
        collided = collide_cumulant_d3q19(f, tau=0.53, C_s=0.03)
        post = torch.where(solid_q, old, collided)
        streamed = stream3d(post)
        f, friction, pressure = bfl_wall_function_3d(
            streamed,
            post,
            solid,
            1.0e-4,
            bfl_mask,
            bfl_q,
            wall_law="musker",
            near_mask=near,
            bfl_wall_mode="wall_model_slip",
            wall_activation=1.0,
            wall_normals=normals,
        )
        cv_force = observe_control_volume_force(
            old,
            f,
            post,
            control_volume,
            solid=solid,
        ).force_on_body[0]
        collision_source = fluid_momentum_change(
            old,
            post,
            control_volume,
            solid=solid,
        )[0]
        assert pressure + friction == pytest.approx(
            float(cv_force + collision_source),
            abs=2e-10,
        )
