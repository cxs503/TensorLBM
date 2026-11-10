"""Contract and physics tests for the moving-body immersed-boundary solver.

Coverage (AC1 of P233W2-N02): static and moving boundaries, plane and curved
geometry, single and multiple bodies, kernel algebra, and the momentum-exchange
identity that the ice-blade load transfer relies on.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.ibm_moving_body import (
    C9,
    W9,
    IBM2D,
    IBM2DConfig,
    MarkerStencil,
    circle_body,
    delta4,
    equilibrium2d,
    free_body_momentum_audit,
    macroscopic2d,
    momentum_exchange_audit,
    rectangle_body,
)

torch.set_num_threads(1)


# --------------------------------------------------------------------------- #
# Kernel algebra (regularised delta / marker stencils)
# --------------------------------------------------------------------------- #


def test_delta4_partition_of_unity() -> None:
    """The 4-point Peskin kernel must sum to unity over its 4-node support."""
    for shift in np.linspace(0.0, 1.0, 9):
        total = float(
            np.sum([delta4(np.array([shift - j])).item() for j in (-1, 0, 1, 2)])
        )
        assert abs(total - 1.0) < 1e-14


def test_delta4_matches_closed_form_at_integers() -> None:
    """phi(0)=0.5, phi(1)=0.25, phi(2)=0 (analytic sanity values)."""
    vals = delta4(np.array([0.0, 1.0, 2.0]))
    assert vals[0] == pytest.approx(0.5, abs=1e-15)
    assert vals[1] == pytest.approx(0.25, abs=1e-15)
    assert vals[2] == pytest.approx(0.0, abs=1e-15)


def test_stencil_partition_of_unity_offscreen_positions() -> None:
    """Stencil weights sum to one for generic (non-lattice) marker positions."""
    rng = np.random.default_rng(7)
    markers = np.stack([rng.uniform(10, 30, 40), rng.uniform(5, 35, 40)], axis=1)
    stencil = MarkerStencil(markers, nx=64, ny=48)
    assert float((stencil.weight_sums() - 1.0).abs().max()) < 1e-13


def test_stencil_wraps_periodically_in_y() -> None:
    """A marker one cell below y=0 must still see the full kernel support."""
    stencil = MarkerStencil(np.array([[20.0, 0.3]]), nx=32, ny=16)
    assert float((stencil.weight_sums() - 1.0).abs().max()) < 1e-13
    # neighbour indices must all be inside the flat array
    assert int(stencil.flat.min()) >= 0
    assert int(stencil.flat.max()) < 32 * 16


def test_circle_marker_measure_sums_to_circumference() -> None:
    """Surface arc lengths are a physical measure: sum(dv) = 2*pi*R."""
    body = circle_body((32.0, 32.0), 8.0, interior_fill=False, spacing=0.5)
    assert float(body.dv.sum()) == pytest.approx(2.0 * math.pi * 8.0, rel=1e-12)


# --------------------------------------------------------------------------- #
# Rigid kinematics
# --------------------------------------------------------------------------- #


def test_marker_velocity_is_rigid_body_rotation() -> None:
    """Marker velocity must be tangential with magnitude omega*r about the centre."""
    body = circle_body((20.0, 20.0), 5.0, omega=0.02, interior_fill=False)
    markers = body.world_markers()
    vel = body.marker_velocity(markers)
    r = markers - body.centre
    assert np.abs(r[:, 0] * vel[:, 0] + r[:, 1] * vel[:, 1]).max() < 1e-12
    speed = np.linalg.norm(vel, axis=1)
    assert np.allclose(speed, 0.02 * np.linalg.norm(r, axis=1), rtol=1e-12)


def test_body_rotation_transform_matches_angle() -> None:
    """A rectangle rotated by 90 degrees maps its long axis onto y."""
    body = rectangle_body((10.0, 10.0), 6.0, 1.0, angle=math.pi / 2.0)
    markers = body.world_markers()
    assert abs(markers[:, 0].max() - markers[:, 0].min() - 2.0) < 1e-9
    assert abs(markers[:, 1].max() - markers[:, 1].min() - 12.0) < 1e-9


# --------------------------------------------------------------------------- #
# Solver-level checks
# --------------------------------------------------------------------------- #


def test_equilibrium_is_a_fixed_point_without_body() -> None:
    """Zero-velocity uniform equilibrium must be preserved exactly."""
    cfg = IBM2DConfig(nx=32, ny=24, u_in=0.0, nu=0.05)
    solver = IBM2D(cfg)
    f0 = solver.f.clone()
    for _ in range(10):
        solver.step([])
    assert torch.allclose(solver.f, f0, atol=1e-13)


def test_uniform_flow_preserved_without_body() -> None:
    """A uniform stream through the channel must not decay without a body."""
    cfg = IBM2DConfig(nx=48, ny=32, u_in=0.02, nu=0.05)
    solver = IBM2D(cfg)
    for _ in range(40):
        solver.step([])
    ux, uy = solver.velocity_field()
    assert np.abs(ux[4:-4, 4:-4] - 0.02).max() < 5e-3
    assert np.abs(uy[4:-4, 4:-4]).max() < 5e-3


def test_static_curved_body_experiences_streamwise_drag() -> None:
    """Curved (circular) static body: the load must be along +x (drag)."""
    body = circle_body((24.0, 16.0), 5.0, interior_fill=False)
    solver = IBM2D(IBM2DConfig(nx=64, ny=32, u_in=0.05, nu=0.05))
    for _ in range(120):
        loads = solver.step([body])
    assert loads["circle"]["fx"] > 0.0
    assert abs(loads["circle"]["fy"]) < 0.25 * loads["circle"]["fx"]


def test_static_plane_body_experiences_drag() -> None:
    """Plane (rectangular) static body normal to the stream: positive drag."""
    body = rectangle_body((24.0, 16.0), 1.0, 6.0, fixed=True)
    solver = IBM2D(IBM2DConfig(nx=64, ny=32, u_in=0.05, nu=0.05))
    for _ in range(120):
        loads = solver.step([body])
    assert loads["rectangle"]["fx"] > 0.0


def test_symmetric_translation_has_no_transverse_force() -> None:
    """Moving boundary in a quiescent box: symmetry must give Fy = 0."""
    body = circle_body((32.0, 24.0), 5.0, velocity=(0.01, 0.0), fixed=True, interior_fill=False)
    solver = IBM2D(IBM2DConfig(nx=64, ny=48, u_in=0.0, nu=0.05, periodic_x=True))
    for _ in range(60):
        loads = solver.step([body])
        body.centre[0] += 0.01
    assert abs(loads["circle"]["fy"]) < 1e-6 * max(abs(loads["circle"]["fx"]), 1e-3)


def test_rotating_body_feels_opposing_torque() -> None:
    """A body spun up in quiescent fluid must feel a resistive (negative) torque."""
    body = circle_body(
        (40.0, 32.0), 6.0, omega=0.01, interior_fill=True, name="rotor"
    )
    solver = IBM2D(IBM2DConfig(nx=80, ny=64, u_in=0.0, nu=0.05, periodic_x=True))
    for _ in range(80):
        loads = solver.step([body])
    assert loads["rotor"]["torque"] < 0.0


def test_multiple_bodies_loads_are_independent() -> None:
    """Two identical discs side by side must feel identical, positive drag."""
    b1 = circle_body((24.0, 8.0), 4.0, interior_fill=False, name="low")
    b2 = circle_body((24.0, 24.0), 4.0, interior_fill=False, name="high")
    solver = IBM2D(IBM2DConfig(nx=72, ny=32, u_in=0.05, nu=0.05))
    for _ in range(150):
        loads = solver.step([b1, b2])
    assert loads["low"]["fx"] > 0.0
    assert loads["high"]["fx"] > 0.0
    # exact mirror symmetry about y = 16 (periodic in y)
    assert loads["low"]["fx"] == pytest.approx(loads["high"]["fx"], rel=1e-10)
    assert abs(loads["low"]["fy"] + loads["high"]["fy"]) < 1e-10


def test_marker_loads_balance_fluid_momentum() -> None:
    """Momentum handed to the fluid equals the load taken off the body."""
    body = circle_body((30.0, 20.0), 5.0, interior_fill=False, name="disc")
    solver = IBM2D(IBM2DConfig(nx=60, ny=40, u_in=0.03, nu=0.05, periodic_x=True))
    p0 = solver.total_momentum()
    total = np.zeros(2)
    for _ in range(100):
        loads = solver.step([body])
        total += np.array([loads["disc"]["fx"], loads["disc"]["fy"]])
    dp = solver.total_momentum() - p0
    assert np.allclose(total, -dp, rtol=1e-10, atol=1e-12)


# --------------------------------------------------------------------------- #
# Momentum audits (AC4)
# --------------------------------------------------------------------------- #


def test_periodic_momentum_exchange_audit() -> None:
    """Prescribed moving body in a periodic box: sum(F) == dP_fluid."""
    res = momentum_exchange_audit(d_cells=8, n_steps=150, u_body=0.02)
    assert res["relative_residual"] < 1e-10


def test_free_body_momentum_conservation() -> None:
    """Free body: fluid + body momentum stays constant through drag exchange."""
    res = free_body_momentum_audit(d_cells=8, n_steps=200, v0=0.02)
    assert res["relative_drift"] < 1e-6