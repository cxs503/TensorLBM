"""Regression tests for drag_friction_integration friction formulas.

Covers:
  - planar-wall exactness: 'standard' == 'faces' (single-face cells)
  - q identities: bfl(q=0.5) == standard, bfl_lagrange(q=0.5) == lagrange
  - BFL wall-distance scaling: bfl with q=0.25 doubles the shear
  - 'faces' on the cylinder staircase: integrates over wall faces, so
    per-layer face count (84 for D=20) exceeds near-cell count (60) —
    this is the curved-surface friction undercount regression guard
  - error paths (missing solid / q_wall)
"""
import math

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.drag_pressure import (
    SurfaceMesh,
    drag_friction_integration,
    get_near_wall_2d,
    get_near_wall_3d,
)


def _cylinder_mask(nx, ny, nz, cx, cy, radius):
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32),
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


@pytest.fixture(scope="module")
def cylinder_setup():
    D = 20
    R = 10.0
    nx = ny = 320
    nz = 40
    cx = cy = 160
    nu = 0.04
    solid = _cylinder_mask(nx, ny, nz, cx, cy, R)
    near = get_near_wall_2d(solid, axis="z")
    mesh = SurfaceMesh.from_cylinder(solid, near, cx, cy, R, axis="z")
    rho = torch.ones((nz, ny, nx))
    ux = torch.full_like(rho, 0.1)
    uy = torch.zeros_like(rho)
    uz = torch.zeros_like(rho)
    f = equilibrium3d(rho, ux, uy, uz)
    return {"solid": solid, "near": near, "mesh": mesh, "f": f, "nu": nu,
            "dpS": 1.0}


def test_planar_wall_standard_equals_faces():
    """On a planar wall (one face per cell) 'faces' == 'standard'."""
    nz, ny, nx = 8, 24, 24
    slab = torch.zeros((nz, ny, nx), dtype=torch.bool)
    slab[:, 10:13, :] = True
    near = get_near_wall_3d(slab)
    mesh = SurfaceMesh.from_gradient(slab, near)
    rho = torch.ones((nz, ny, nx))
    f = equilibrium3d(rho, torch.full_like(rho, 0.1),
                      torch.zeros_like(rho), torch.zeros_like(rho))
    nu, dpS = 0.05, 1.0
    f_std = drag_friction_integration(f, mesh, dpS, nu, formula="standard")
    f_fc = drag_friction_integration(f, mesh, dpS, nu, formula="faces", solid=slab)
    assert f_std[0] == pytest.approx(f_fc[0], rel=1e-6)
    assert f_std[1] == pytest.approx(f_fc[1], rel=1e-6)
    assert f_std[2] == pytest.approx(f_fc[2], rel=1e-6)


def test_bfl_q_half_identities(cylinder_setup):
    """bfl(q=0.5) == standard; bfl_lagrange(q=0.5) == lagrange."""
    s = cylinder_setup
    q_half = torch.full_like(s["solid"], 0.5, dtype=torch.float32)
    f_std = drag_friction_integration(s["f"], s["mesh"], s["dpS"], s["nu"],
                                      formula="standard")
    f_bfl = drag_friction_integration(s["f"], s["mesh"], s["dpS"], s["nu"],
                                      q_wall=q_half, formula="bfl")
    assert f_bfl[0] == pytest.approx(f_std[0], rel=1e-6)
    f_lag = drag_friction_integration(s["f"], s["mesh"], s["dpS"], s["nu"],
                                      formula="lagrange")
    f_blag = drag_friction_integration(s["f"], s["mesh"], s["dpS"], s["nu"],
                                       q_wall=q_half, formula="bfl_lagrange")
    assert f_blag[0] == pytest.approx(f_lag[0], rel=1e-6)


def test_bfl_wall_distance_scaling(cylinder_setup):
    """bfl with q=0.25 doubles the standard (q=0.5) shear."""
    s = cylinder_setup
    f_std = drag_friction_integration(s["f"], s["mesh"], s["dpS"], s["nu"],
                                      formula="standard")
    q_q = torch.full_like(s["solid"], 0.25, dtype=torch.float32)
    f_bfl = drag_friction_integration(s["f"], s["mesh"], s["dpS"], s["nu"],
                                      q_wall=q_q, formula="bfl")
    assert f_bfl[0] == pytest.approx(2.0 * f_std[0], rel=1e-6)


def test_faces_counts_staircase_faces(cylinder_setup):
    """'faces' integrates over wall faces: D=20 cylinder has 84 faces but
    only 60 near-wall cells per layer.  This is the curved-surface friction
    undercount guard: cell-based dA=1 misses the extra faces of the
    staircase inner-corner cells."""
    s = cylinder_setup
    s0 = s["solid"][0]
    f0 = ~s0
    nfy = torch.zeros_like(s0, dtype=torch.int32)
    nfx = torch.zeros_like(s0, dtype=torch.int32)
    nfy[1:-1, :] += (s0[2:, :] & f0[1:-1, :]).int()
    nfy[1:-1, :] += (s0[:-2, :] & f0[1:-1, :]).int()
    nfx[:, 1:-1] += (s0[:, 2:] & f0[:, 1:-1]).int()
    nfx[:, 1:-1] += (s0[:, :-2] & f0[:, 1:-1]).int()
    n_near_layer = int(s["near"][0].sum())
    n_faces_layer = int((nfy + nfx)[s["near"][0]].sum())
    assert n_near_layer == 60
    assert n_faces_layer == 84
    assert n_faces_layer > n_near_layer  # the undercount regression guard

    # uniform-field check: faces fx = 2*nu*u_x * (#y-faces) * nz exactly
    _, ux, _, _ = macroscopic3d(s["f"])
    y_faces_layer = int(nfy[s["near"][0]].sum())
    f_fc = drag_friction_integration(s["f"], s["mesh"], s["dpS"], s["nu"],
                                     formula="faces", solid=s["solid"])
    expected = 2.0 * s["nu"] * 0.1 * y_faces_layer * s["solid"].shape[0]
    assert f_fc[0] == pytest.approx(expected, rel=1e-5)


def test_error_paths(cylinder_setup):
    s = cylinder_setup
    with pytest.raises(ValueError, match="solid"):
        drag_friction_integration(s["f"], s["mesh"], s["dpS"], s["nu"],
                                  formula="faces")
    with pytest.raises(ValueError, match="q_wall"):
        drag_friction_integration(s["f"], s["mesh"], s["dpS"], s["nu"],
                                  formula="bfl")
    with pytest.raises(ValueError, match="q_wall"):
        drag_friction_integration(s["f"], s["mesh"], s["dpS"], s["nu"],
                                  formula="bfl_lagrange")
    with pytest.raises(ValueError, match="formula"):
        drag_friction_integration(s["f"], s["mesh"], s["dpS"], s["nu"],
                                  formula="bogus")


def test_standard_regression_values(cylinder_setup):
    """Guard the historical standard-formula values on the D=20 staircase
    with a uniform field (pure geometry, no flow simulation)."""
    s = cylinder_setup
    f_std = drag_friction_integration(s["f"], s["mesh"], s["dpS"], s["nu"],
                                      formula="standard")
    # expected: 2*nu*0.1 * sum over 60 near cells of (1 - n_x^2)
    n0 = s["near"][0]
    nxc = s["mesh"].nx_n[0][n0]
    expected = 2.0 * s["nu"] * 0.1 * (1.0 - nxc ** 2).sum() * s["solid"].shape[0]
    assert f_std[0] == pytest.approx(expected.item(), rel=1e-5)
