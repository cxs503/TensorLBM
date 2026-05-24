"""Smoke tests for the rotating-cylinder (Magnus) runner."""
from __future__ import annotations

import json

import torch

from tensorlbm import (
    RotatingCylinderConfig,
    moving_wall_bounce_back,
    rotating_wall_velocity,
    run_rotating_cylinder,
)
from tensorlbm.boundaries import cylinder_mask
from tensorlbm.d2q9 import equilibrium


class TestRotatingWallVelocity:
    def test_centre_is_stationary(self) -> None:
        mask = torch.ones((10, 10), dtype=torch.bool)
        ux, uy = rotating_wall_velocity(mask, cx=4.5, cy=4.5, omega=0.1)
        # At the exact rotation centre, ux=uy=0
        # Closest grid point to (4.5,4.5) – check averages over the 4 neighbours
        # The field is exactly antisymmetric, so its sum over a symmetric patch is 0
        assert torch.allclose(ux.sum(), torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(uy.sum(), torch.tensor(0.0), atol=1e-5)

    def test_tangent_to_circle(self) -> None:
        """At any point (x,y) the wall velocity should be tangent: u·r = 0."""
        mask = torch.ones((20, 20), dtype=torch.bool)
        cx, cy = 10.0, 10.0
        ux, uy = rotating_wall_velocity(mask, cx=cx, cy=cy, omega=0.05)
        yy, xx = torch.meshgrid(
            torch.arange(20, dtype=torch.float32),
            torch.arange(20, dtype=torch.float32),
            indexing="ij",
        )
        rx = xx - cx
        ry = yy - cy
        dot = ux * rx + uy * ry
        assert torch.allclose(dot, torch.zeros_like(dot), atol=1e-5)


class TestMovingWallBounceBack:
    def test_reduces_to_bounce_back_when_static(self) -> None:
        """If ω=0 the moving-wall BC should reduce to plain bounce-back."""
        from tensorlbm.boundaries import bounce_back_cells

        ny, nx = 8, 10
        rho = torch.ones((ny, nx))
        ux = torch.full_like(rho, 0.05)
        uy = torch.zeros_like(rho)
        f = equilibrium(rho, ux, uy)
        mask = cylinder_mask(nx, ny, cx=nx / 2, cy=ny / 2, radius=2.0, device=f.device)
        ux_w = torch.zeros_like(rho)
        uy_w = torch.zeros_like(rho)
        f1 = bounce_back_cells(f.clone(), mask)
        f2 = moving_wall_bounce_back(f.clone(), mask, ux_w, uy_w)
        assert torch.allclose(f1, f2, atol=1e-6)


class TestRunRotatingCylinder:
    def test_smoke_run(self, tmp_path) -> None:
        config = RotatingCylinderConfig(
            nx=64,
            ny=24,
            radius=4.0,
            u_in=0.05,
            re=40.0,
            spin_ratio=1.0,
            n_steps=20,
            output_interval=10,
            output_root=tmp_path,
            run_name="smoke",
            overwrite=True,
        )
        run_dir = run_rotating_cylinder(config)
        assert (run_dir / "run_metadata.json").exists()
        meta = json.loads((run_dir / "run_metadata.json").read_text())
        assert "cd_mean" in meta
        assert "cl_mean" in meta
        # Diagnostics list non-empty for at least one output step
        assert len(meta["diagnostics"]) >= 1
        # forces.csv has one row per step + header
        forces_path = run_dir / "forces.csv"
        assert forces_path.exists()
        lines = forces_path.read_text().strip().splitlines()
        assert len(lines) == config.n_steps + 1  # +1 header

    def test_magnus_lift_sign(self, tmp_path) -> None:
        """A positive spin ratio (CCW rotation) in a left-to-right free stream
        should produce a *negative* mean lift coefficient (downward force).

        Brief justification: with ω > 0 the cylinder surface moves downward on
        the upstream side and upward on the downstream side. By Bernoulli the
        pressure is higher on the top and lower on the bottom, giving a
        downward (−y) net force. We use a very small run that still resolves
        the sign of the asymmetry.
        """
        config = RotatingCylinderConfig(
            nx=160,
            ny=60,
            radius=6.0,
            u_in=0.06,
            re=80.0,
            spin_ratio=1.5,
            n_steps=400,
            output_interval=100,
            output_root=tmp_path,
            run_name="magnus",
            overwrite=True,
        )
        run_dir = run_rotating_cylinder(config)
        meta = json.loads((run_dir / "run_metadata.json").read_text())
        cl = float(meta["cl_mean"])
        assert cl < 0.0, f"Expected negative Cl for CCW spin, got {cl:.4f}"
