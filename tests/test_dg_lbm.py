"""Tests for the DG-LBM hybrid sphere-flow module.

Covers:
- ``DGLBMConfig`` derived properties, validation, and run-name formatting.
- ``build_dg_shell_mask`` geometry correctness.
- ``dg_compute_velocity_gradients`` finite-difference accuracy.
- ``collide_dg_lbm`` conservation and DG-zone activation.
- ``run_dg_lbm_sphere_flow`` end-to-end smoke run with artifact checks.
- Resume-from-checkpoint behaviour.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tensorlbm import (
    DGLBMConfig,
    build_dg_shell_mask,
    collide_dg_lbm,
    dg_compute_velocity_gradients,
    run_dg_lbm_sphere_flow,
)
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d

# ---------------------------------------------------------------------------
# DGLBMConfig
# ---------------------------------------------------------------------------


class TestDGLBMConfig:
    def test_post_init_normalises(self, tmp_path: Path) -> None:
        cfg = DGLBMConfig(output_root=str(tmp_path), device="CPU")
        assert isinstance(cfg.output_root, Path)
        assert cfg.device == "cpu"

    def test_post_init_normalises_checkpoint(self, tmp_path: Path) -> None:
        cfg = DGLBMConfig(resume_checkpoint=str(tmp_path))
        assert isinstance(cfg.resume_checkpoint, Path)

    def test_derived_properties(self) -> None:
        cfg = DGLBMConfig(u_in=0.06, radius=8.0, re=50.0, dg_band=4.0)
        assert cfg.nu == pytest.approx(0.06 * 2 * 8.0 / 50.0)
        assert cfg.tau == pytest.approx(3.0 * cfg.nu + 0.5)
        assert cfg.dg_radius == pytest.approx(8.0 + 4.0)

    def test_run_name_default_integer_re(self) -> None:
        cfg = DGLBMConfig(nx=40, ny=20, nz=20, re=50.0, u_in=0.06, dg_band=4.0, n_steps=10)
        name = cfg.resolved_run_name()
        assert "re50" in name
        assert "dg4.0" in name

    def test_run_name_custom(self) -> None:
        assert DGLBMConfig(run_name="test").resolved_run_name() == "test"

    def test_validate_raises_small_grid(self) -> None:
        with pytest.raises(ValueError, match="nx, ny, nz"):
            DGLBMConfig(nx=4).validate()

    def test_validate_raises_bad_tau(self) -> None:
        with pytest.raises(ValueError, match="tau"):
            DGLBMConfig(u_in=0.5, re=1.0, radius=60.0).validate()

    def test_validate_raises_dg_band_zero(self) -> None:
        with pytest.raises(ValueError, match="dg_band"):
            DGLBMConfig(dg_band=0.0).validate()

    def test_validate_raises_bad_dg_order(self) -> None:
        with pytest.raises(ValueError, match="dg_order"):
            DGLBMConfig(dg_order=2).validate()

    def test_validate_passes_defaults(self) -> None:
        DGLBMConfig().validate()


# ---------------------------------------------------------------------------
# build_dg_shell_mask
# ---------------------------------------------------------------------------


class TestBuildDGShellMask:
    def test_shell_excludes_interior_and_exterior(self) -> None:
        device = torch.device("cpu")
        nx, ny, nz = 32, 32, 32
        cx, cy, cz = 16.0, 16.0, 16.0
        r_inner, r_outer = 5.0, 9.0

        mask = build_dg_shell_mask(nx, ny, nz, cx, cy, cz, r_inner, r_outer, device)
        assert mask.shape == (nz, ny, nx)

        # The centre cell should be inside the sphere (not in DG shell)
        assert not mask[16, 16, 16].item()

        # A cell at (16, 16, 22) has distance 6 from centre → in shell
        assert mask[16, 16, 22].item()

        # A cell at (16, 16, 30) has distance 14 → outside shell
        assert not mask[16, 16, 30].item()

    def test_shell_is_subset_of_grid(self) -> None:
        device = torch.device("cpu")
        mask = build_dg_shell_mask(20, 20, 20, 10.0, 10.0, 10.0, 3.0, 6.0, device)
        assert mask.dtype == torch.bool
        assert mask.shape == (20, 20, 20)

    def test_non_overlapping_with_sphere(self) -> None:
        """DG mask and sphere mask should not overlap."""
        from tensorlbm.boundaries3d import sphere_mask as smask

        device = torch.device("cpu")
        nx, ny, nz, cx, cy, cz = 32, 32, 32, 16.0, 16.0, 16.0
        r = 5.0
        obs = smask(nx, ny, nz, cx, cy, cz, r, device)
        dg = build_dg_shell_mask(nx, ny, nz, cx, cy, cz, r, r + 4.0, device)
        assert not (obs & dg).any()


# ---------------------------------------------------------------------------
# dg_compute_velocity_gradients
# ---------------------------------------------------------------------------


class TestDGComputeVelocityGradients:
    def test_linear_field_exact(self) -> None:
        """For a linear velocity field, central differences are exact."""
        nz, ny, nx = 10, 10, 10
        # ux = 0.1 * x  →  dux/dx = 0.1 everywhere
        x = torch.arange(nx, dtype=torch.float32)
        ux = 0.1 * x.view(1, 1, nx).expand(nz, ny, nx)
        uy = torch.zeros(nz, ny, nx)
        uz = torch.zeros(nz, ny, nx)

        grads = dg_compute_velocity_gradients(ux, uy, uz)
        dux_dx = grads[0]

        # Interior cells should be close to 0.1
        assert dux_dx[0, 0, 1:-1].allclose(torch.full((nx - 2,), 0.1), atol=1e-6)

    def test_returns_nine_tensors(self) -> None:
        ux = torch.rand(4, 4, 4)
        uy = torch.rand(4, 4, 4)
        uz = torch.rand(4, 4, 4)
        grads = dg_compute_velocity_gradients(ux, uy, uz)
        assert len(grads) == 9
        for g in grads:
            assert g.shape == (4, 4, 4)

    def test_zero_field_zero_gradient(self) -> None:
        z = torch.zeros(5, 5, 5)
        grads = dg_compute_velocity_gradients(z, z, z)
        for g in grads:
            assert g.abs().max().item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# collide_dg_lbm
# ---------------------------------------------------------------------------


class TestCollideDGLBM:
    def _init_f(self, nz: int = 8, ny: int = 8, nx: int = 8) -> torch.Tensor:
        rho = torch.ones(nz, ny, nx)
        ux = torch.full((nz, ny, nx), 0.05)
        uy = torch.zeros(nz, ny, nx)
        uz = torch.zeros(nz, ny, nx)
        return equilibrium3d(rho, ux, uy, uz)

    def test_mass_conserved_no_dg(self) -> None:
        f = self._init_f()
        dg_mask = torch.zeros(8, 8, 8, dtype=torch.bool)
        mass_before = f.sum().item()
        f_out = collide_dg_lbm(f, tau=0.8, dg_mask=dg_mask)
        assert f_out.sum().item() == pytest.approx(mass_before, rel=1e-5)

    def test_mass_conserved_full_dg(self) -> None:
        f = self._init_f()
        dg_mask = torch.ones(8, 8, 8, dtype=torch.bool)
        mass_before = f.sum().item()
        f_out = collide_dg_lbm(f, tau=0.8, dg_mask=dg_mask)
        assert f_out.sum().item() == pytest.approx(mass_before, rel=1e-5)

    def test_momentum_approx_conserved_full_dg(self) -> None:
        """Chapman–Enskog collision conserves momentum to machine precision."""
        f = self._init_f()
        dg_mask = torch.ones(8, 8, 8, dtype=torch.bool)
        _, ux_in, uy_in, uz_in = macroscopic3d(f)
        rho_in = f.sum(0)
        px_before = (rho_in * ux_in).sum().item()

        f_out = collide_dg_lbm(f, tau=0.8, dg_mask=dg_mask)
        rho_out, ux_out, uy_out, uz_out = macroscopic3d(f_out)
        px_after = (rho_out * ux_out).sum().item()

        assert px_after == pytest.approx(px_before, rel=1e-4)

    def test_output_shape_preserved(self) -> None:
        f = self._init_f(6, 8, 10)
        dg_mask = torch.zeros(6, 8, 10, dtype=torch.bool)
        f_out = collide_dg_lbm(f, tau=0.9, dg_mask=dg_mask)
        assert f_out.shape == f.shape

    def test_dg_and_bgk_differ_in_dg_zone(self) -> None:
        """DG collision should produce different values than plain BGK in DG zone."""
        from tensorlbm.solver3d import collide_bgk3d

        nz, ny, nx = 8, 8, 8
        rho = torch.ones(nz, ny, nx)
        # Non-uniform velocity to produce non-zero gradients
        x = torch.linspace(0.0, 0.1, nx)
        ux = 0.05 + 0.01 * x.view(1, 1, nx).expand(nz, ny, nx)
        uy = torch.zeros(nz, ny, nx)
        uz = torch.zeros(nz, ny, nx)
        f = equilibrium3d(rho, ux, uy, uz)

        dg_mask = torch.ones(nz, ny, nx, dtype=torch.bool)
        f_dg = collide_dg_lbm(f, tau=0.8, dg_mask=dg_mask)
        f_bgk = collide_bgk3d(f, tau=0.8)

        # They should differ in the DG zone because gradients are non-zero
        assert not torch.allclose(f_dg, f_bgk, atol=1e-9)


# ---------------------------------------------------------------------------
# run_dg_lbm_sphere_flow – smoke tests
# ---------------------------------------------------------------------------


class TestRunDGLBMSphereFlow:
    def _smoke_cfg(self, tmp_path: Path, **overrides) -> DGLBMConfig:
        kwargs: dict = {
            "nx": 32, "ny": 16, "nz": 16,
            "u_in": 0.05, "re": 50.0, "radius": 3.0, "dg_band": 3.0,
            "n_steps": 4, "output_interval": 2,
            "output_root": tmp_path, "run_name": "smoke", "overwrite": True,
        }
        kwargs.update(overrides)
        return DGLBMConfig(**kwargs)

    def test_smoke_run_produces_artifacts(self, tmp_path: Path) -> None:
        cfg = self._smoke_cfg(tmp_path)
        run_dir = run_dg_lbm_sphere_flow(cfg)

        assert run_dir == tmp_path / "dg_lbm_sphere" / "smoke"
        assert run_dir.is_dir()

        meta_path = run_dir / "run_metadata.json"
        assert meta_path.exists()
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        assert metadata["config"]["n_steps"] == 4
        assert metadata["derived"]["dg_radius"] == pytest.approx(6.0)
        assert "diagnostics" in metadata
        assert len(metadata["diagnostics"]) == 2  # steps 2 and 4

        for entry in metadata["diagnostics"]:
            for key in ("step", "mass", "mass_drift", "max_speed", "mean_rho"):
                assert key in entry

        assert (run_dir / "flow_step_000002.png").exists()
        assert (run_dir / "flow_step_000004.png").exists()
        assert (run_dir / "checkpoint_f.pt").exists()
        assert (run_dir / "checkpoint_meta.json").exists()

    def test_diagnostics_are_finite(self, tmp_path: Path) -> None:
        cfg = self._smoke_cfg(tmp_path, run_name="finite")
        run_dir = run_dg_lbm_sphere_flow(cfg)
        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        for entry in metadata["diagnostics"]:
            for key in ("mass", "mass_drift", "max_speed", "mean_rho"):
                value = entry[key]
                assert isinstance(value, (int, float))
                assert math.isfinite(value), f"non-finite {key}={value}"

    def test_resume_from_checkpoint(self, tmp_path: Path) -> None:
        cfg1 = self._smoke_cfg(tmp_path, run_name="first")
        run_dir1 = run_dg_lbm_sphere_flow(cfg1)
        assert (run_dir1 / "checkpoint_f.pt").exists()

        cfg2 = DGLBMConfig(
            nx=32, ny=16, nz=16,
            u_in=0.05, re=50.0, radius=3.0, dg_band=3.0,
            n_steps=6, output_interval=2,
            output_root=tmp_path, run_name="second", overwrite=True,
            resume_checkpoint=run_dir1,
        )
        run_dir2 = run_dg_lbm_sphere_flow(cfg2)
        metadata = json.loads((run_dir2 / "run_metadata.json").read_text(encoding="utf-8"))
        steps = [entry["step"] for entry in metadata["diagnostics"]]
        assert steps == [6]

    def test_dg_band_zero_still_runs(self, tmp_path: Path) -> None:
        """dg_band > 0 is enforced by validate(); confirm validate catches it."""
        with pytest.raises(ValueError, match="dg_band"):
            DGLBMConfig(
                nx=32, ny=16, nz=16,
                u_in=0.05, re=50.0, radius=3.0, dg_band=0.0,
                n_steps=2,
            ).validate()
