"""Tests for the 3D immersed boundary helpers."""
from __future__ import annotations

import torch

from tensorlbm import equilibrium3d
from tensorlbm.ibm import (
    ibm_apply_body_force_3d,
    ibm_direct_forcing_3d,
    ibm_force_spread_3d,
    ibm_velocity_interpolate_3d,
)


def test_ibm_velocity_interpolate_3d_shape() -> None:
    ux = torch.rand((4, 5, 6))
    uy = torch.rand((4, 5, 6))
    uz = torch.rand((4, 5, 6))
    marker = torch.tensor([1.5, 2.5])
    u_mx, u_my, u_mz = ibm_velocity_interpolate_3d(ux, uy, uz, marker, marker, marker)
    assert u_mx.shape == u_my.shape == u_mz.shape == (2,)


def test_ibm_force_spread_3d_shape() -> None:
    marker = torch.tensor([1.5, 2.5])
    forces = torch.ones(2)
    fx, fy, fz = ibm_force_spread_3d(forces, forces, forces, marker, marker, marker, 4, 5, 6)
    assert fx.shape == fy.shape == fz.shape == (4, 5, 6)


def test_ibm_direct_forcing_3d_shape() -> None:
    ux = torch.rand((4, 5, 6))
    uy = torch.rand((4, 5, 6))
    uz = torch.rand((4, 5, 6))
    marker = torch.tensor([1.5, 2.5])
    target = torch.zeros(2)
    fx, fy, fz = ibm_direct_forcing_3d(ux, uy, uz, marker, marker, marker, target, target, target)
    assert fx.shape == fy.shape == fz.shape == (4, 5, 6)


def test_ibm_apply_body_force_3d_shape() -> None:
    rho = torch.ones((4, 5, 6))
    zeros = torch.zeros_like(rho)
    f = equilibrium3d(rho, zeros, zeros, zeros)
    force = torch.rand((4, 5, 6))
    fout = ibm_apply_body_force_3d(f, force, force, force)
    assert fout.shape == f.shape
