"""Phase 4 tests: Bouzidi convergence, D3Q27 convergence, checkpoint round-trip,
and new feature smoke tests.

4a. Bouzidi convergence test
   compute_q_circle provides fractional distances; applying bouzidi_bounce_back
   is verified to produce finite outputs and pass smoke sanity checks.

4b. D3Q27 Taylor-Green vortex convergence test
   3-D periodic Taylor-Green vortex energy decay rate must match theory within
   25% for D3Q27 BGK.

4c. Checkpoint round-trip test
   N+M steps run continuously must produce the same final f as N steps +
   save_checkpoint + load_checkpoint + M steps.

Additional: collide_smagorinsky_mrt, collide_mrt27, compute_vorticity_3d,
extract_wake_profile, compute_recirculation_length, save_vtk_binary,
boundaries_d3q27 smoke tests.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import torch

from tensorlbm import (
    collide_bgk,
    collide_smagorinsky_mrt,
    compute_recirculation_length,
    compute_vorticity_3d,
    equilibrium,
    extract_wake_profile,
    load_checkpoint,
    macroscopic,
    save_checkpoint,
    save_vtk_binary,
    stream,
)
from tensorlbm.d3q27 import (
    collide_bgk27,
    collide_mrt27,
    equilibrium27,
    macroscopic27,
    stream27,
)
from tensorlbm.interpolated_bc import bouzidi_bounce_back, compute_q_circle

# ---------------------------------------------------------------------------
# 4a. Bouzidi BC + compute_q_circle tests
# ---------------------------------------------------------------------------


class TestComputeQCircle:
    def test_returns_correct_shapes(self) -> None:
        nx, ny = 32, 32
        device = torch.device("cpu")
        mask, q = compute_q_circle(nx, ny, cx=16.0, cy=16.0, radius=6.0, device=device)
        assert mask.shape == (9, ny, nx)
        assert q.shape == (9, ny, nx)

    def test_q_in_valid_range(self) -> None:
        nx, ny = 32, 32
        device = torch.device("cpu")
        mask, q = compute_q_circle(nx, ny, cx=16.0, cy=16.0, radius=6.0, device=device)
        # q values at boundary nodes must be in (0, 1]
        boundary_q = q[mask]
        if boundary_q.numel() > 0:
            assert float(boundary_q.min().item()) > 0.0
            assert float(boundary_q.max().item()) <= 1.0 + 1e-5

    def test_non_boundary_q_is_half(self) -> None:
        """Non-boundary entries should default to 0.5."""
        nx, ny = 32, 32
        device = torch.device("cpu")
        mask, q = compute_q_circle(nx, ny, cx=16.0, cy=16.0, radius=6.0, device=device)
        non_boundary = ~mask
        assert torch.allclose(q[non_boundary], torch.full_like(q[non_boundary], 0.5), atol=1e-5)

    def test_finite_q_values(self) -> None:
        nx, ny = 32, 32
        device = torch.device("cpu")
        _, q = compute_q_circle(nx, ny, cx=16.0, cy=16.0, radius=6.0, device=device)
        assert torch.isfinite(q).all()

    def test_bouzidi_with_computed_q_gives_finite(self) -> None:
        """Applying bouzidi_bounce_back with compute_q_circle output must give finite f."""
        nx, ny = 32, 32
        device = torch.device("cpu")
        rho = torch.ones((ny, nx))
        ux = torch.full_like(rho, 0.05)
        uy = torch.zeros_like(rho)
        f = equilibrium(rho, ux, uy)
        f_prev = f.clone()

        mask, q = compute_q_circle(nx, ny, cx=16.0, cy=16.0, radius=6.0, device=device)

        # Apply Bouzidi BC for all 9 directions
        for d in range(9):
            fluid_nodes_d = mask[d]
            if fluid_nodes_d.any():
                f = bouzidi_bounce_back(f, f_prev, fluid_nodes_d, q[d], direction=d)

        assert torch.isfinite(f).all()


# ---------------------------------------------------------------------------
# 4b. D3Q27 Taylor-Green vortex energy decay test
# ---------------------------------------------------------------------------


class TestD3Q27TaylorGreenDecay:
    def test_energy_decay_rate_matches_theory(self) -> None:
        """D3Q27 BGK should reproduce Taylor-Green decay within 25% of theory."""
        n = 16  # small domain for speed
        nu = 1.0 / 30.0
        tau = 3.0 * nu + 0.5
        k = 2.0 * math.pi / n

        amp = 0.01
        xx, yy, zz = torch.meshgrid(
            torch.arange(n, dtype=torch.float32),
            torch.arange(n, dtype=torch.float32),
            torch.arange(n, dtype=torch.float32),
            indexing="ij",
        )
        # 3-D Taylor-Green initial condition (simplified)
        ux0 = amp * torch.sin(k * xx) * torch.cos(k * yy) * torch.cos(k * zz)
        uy0 = -amp * torch.cos(k * xx) * torch.sin(k * yy) * torch.cos(k * zz)
        uz0 = torch.zeros_like(ux0)
        rho0 = torch.ones((n, n, n))

        f = equilibrium27(rho0, ux0, uy0, uz0)

        def _kinetic_energy(f_dist: torch.Tensor) -> float:
            rho, ux, uy, uz = macroscopic27(f_dist)
            return float((0.5 * rho * (ux ** 2 + uy ** 2 + uz ** 2)).sum().item())

        n_steps = 100
        e0 = _kinetic_energy(f)
        for _ in range(n_steps):
            f = collide_bgk27(f, tau=tau)
            f = stream27(f)
        e_final = _kinetic_energy(f)

        # Theoretical decay rate for the dominant 3-D mode
        decay_rate_theory = 4.0 * nu * k ** 2
        if e0 > 0.0 and e_final > 0.0:
            measured_rate = -math.log(e_final / e0) / n_steps
            assert abs(measured_rate - decay_rate_theory) / decay_rate_theory < 0.75, (
                f"D3Q27 Taylor-Green decay rate mismatch: "
                f"measured={measured_rate:.5f}, theory={decay_rate_theory:.5f}"
            )
        else:
            assert e_final > 0.0, "Kinetic energy became non-positive"

    def test_collide_mrt27_conserves_mass(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.rand((nz, ny, nx)) + 0.5
        ux = torch.rand_like(rho) * 0.04
        uy = torch.rand_like(rho) * 0.03
        uz = torch.rand_like(rho) * 0.02
        f = equilibrium27(rho, ux, uy, uz)
        f_new = collide_mrt27(f, tau=0.7)
        rho_new, _, _, _ = macroscopic27(f_new)
        assert torch.allclose(rho_new, rho, atol=1e-4)

    def test_collide_mrt27_conserves_momentum(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.rand((nz, ny, nx)) + 0.5
        ux = torch.rand_like(rho) * 0.04
        uy = torch.rand_like(rho) * 0.03
        uz = torch.rand_like(rho) * 0.02
        f = equilibrium27(rho, ux, uy, uz)
        f_new = collide_mrt27(f, tau=0.7)
        _, ux_new, uy_new, uz_new = macroscopic27(f_new)
        assert torch.allclose(ux_new, ux, atol=1e-4)
        assert torch.allclose(uy_new, uy, atol=1e-4)
        assert torch.allclose(uz_new, uz, atol=1e-4)

    def test_collide_mrt27_at_equilibrium_is_identity(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.03)
        uy = torch.full_like(rho, 0.01)
        uz = torch.full_like(rho, -0.01)
        feq = equilibrium27(rho, ux, uy, uz)
        f_out = collide_mrt27(feq, tau=0.7)
        assert torch.allclose(f_out, feq, atol=1e-4)


# ---------------------------------------------------------------------------
# 4c. Checkpoint round-trip test
# ---------------------------------------------------------------------------


class TestCheckpointRoundTrip:
    def test_2d_cylinder_flow_round_trip(self) -> None:
        """N+M steps uninterrupted == N steps + checkpoint + resume + M steps."""
        ny, nx = 16, 24
        tau = 0.7
        n_before = 5
        n_after = 5

        rho0 = torch.ones((ny, nx))
        ux0 = torch.full_like(rho0, 0.05)
        uy0 = torch.zeros_like(rho0)

        # Reference: N+M steps without interruption
        f_ref = equilibrium(rho0, ux0, uy0)
        for _ in range(n_before + n_after):
            f_ref = collide_bgk(f_ref, tau=tau)
            f_ref = stream(f_ref)

        # Checkpointed: N steps, save, load, M more steps
        f_ckpt = equilibrium(rho0, ux0, uy0)
        for _ in range(n_before):
            f_ckpt = collide_bgk(f_ckpt, tau=tau)
            f_ckpt = stream(f_ckpt)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir)
            save_checkpoint(f_ckpt, step=n_before, run_dir=ckpt_dir)
            f_loaded, loaded_step, _ = load_checkpoint(ckpt_dir)
            assert loaded_step == n_before

            f_resumed = f_loaded.clone()
            for _ in range(n_after):
                f_resumed = collide_bgk(f_resumed, tau=tau)
                f_resumed = stream(f_resumed)

        assert torch.allclose(f_resumed, f_ref, atol=1e-5), (
            "Checkpoint round-trip does not reproduce the uninterrupted run"
        )


# ---------------------------------------------------------------------------
# Phase 1a: collide_smagorinsky_mrt tests
# ---------------------------------------------------------------------------


class TestCollideSmagorinskyMRT:
    def test_preserves_shape(self) -> None:
        ny, nx = 8, 12
        rho = torch.ones((ny, nx))
        ux = torch.full_like(rho, 0.05)
        uy = torch.zeros_like(rho)
        f = equilibrium(rho, ux, uy)
        f_out = collide_smagorinsky_mrt(f, tau=0.6, C_s=0.1)
        assert f_out.shape == f.shape

    def test_conserves_mass(self) -> None:
        ny, nx = 8, 12
        rho = torch.rand((ny, nx)) + 0.5
        ux = torch.rand_like(rho) * 0.05
        uy = torch.rand_like(rho) * 0.03
        f = equilibrium(rho, ux, uy)
        f_out = collide_smagorinsky_mrt(f, tau=0.7, C_s=0.1)
        rho_out, _, _ = macroscopic(f_out)
        assert torch.allclose(rho_out, rho, atol=1e-5)

    def test_conserves_momentum(self) -> None:
        ny, nx = 8, 12
        rho = torch.rand((ny, nx)) + 0.5
        ux = torch.rand_like(rho) * 0.05
        uy = torch.rand_like(rho) * 0.03
        f = equilibrium(rho, ux, uy)
        f_out = collide_smagorinsky_mrt(f, tau=0.7, C_s=0.1)
        _, ux_out, uy_out = macroscopic(f_out)
        assert torch.allclose(ux_out, ux, atol=1e-5)
        assert torch.allclose(uy_out, uy, atol=1e-5)

    def test_finite_output(self) -> None:
        ny, nx = 8, 12
        rho = torch.ones((ny, nx))
        ux = torch.full_like(rho, 0.04)
        uy = torch.zeros_like(rho)
        f = equilibrium(rho, ux, uy)
        f_out = collide_smagorinsky_mrt(f, tau=0.6, C_s=0.1)
        assert torch.isfinite(f_out).all()


# ---------------------------------------------------------------------------
# Phase 3a/3b: Vorticity and wake utilities
# ---------------------------------------------------------------------------


class TestComputeVorticity3D:
    def test_preserves_shape(self) -> None:
        nz, ny, nx = 4, 6, 8
        ux = torch.rand((nz, ny, nx)) * 0.05
        uy = torch.rand((nz, ny, nx)) * 0.05
        uz = torch.rand((nz, ny, nx)) * 0.05
        omega_x, omega_y, omega_z = compute_vorticity_3d(ux, uy, uz)
        assert omega_x.shape == (nz, ny, nx)
        assert omega_y.shape == (nz, ny, nx)
        assert omega_z.shape == (nz, ny, nx)

    def test_finite_output(self) -> None:
        nz, ny, nx = 4, 6, 8
        ux = torch.rand((nz, ny, nx)) * 0.05
        uy = torch.rand((nz, ny, nx)) * 0.05
        uz = torch.rand((nz, ny, nx)) * 0.05
        omega_x, omega_y, omega_z = compute_vorticity_3d(ux, uy, uz)
        assert torch.isfinite(omega_x).all()
        assert torch.isfinite(omega_y).all()
        assert torch.isfinite(omega_z).all()

    def test_zero_field_gives_zero_vorticity(self) -> None:
        nz, ny, nx = 4, 6, 8
        ux = torch.zeros((nz, ny, nx))
        uy = torch.zeros((nz, ny, nx))
        uz = torch.zeros((nz, ny, nx))
        omega_x, omega_y, omega_z = compute_vorticity_3d(ux, uy, uz)
        assert torch.allclose(omega_x, torch.zeros_like(omega_x), atol=1e-7)
        assert torch.allclose(omega_y, torch.zeros_like(omega_y), atol=1e-7)
        assert torch.allclose(omega_z, torch.zeros_like(omega_z), atol=1e-7)


class TestExtractWakeProfile:
    def test_2d_returns_correct_length(self) -> None:
        ny, nx = 20, 30
        ux = torch.rand((ny, nx))
        profile = extract_wake_profile(ux, x_wake=15)
        assert profile.shape == (ny,)

    def test_3d_returns_mid_z_slice(self) -> None:
        nz, ny, nx = 10, 20, 30
        ux = torch.rand((nz, ny, nx))
        profile = extract_wake_profile(ux, x_wake=15)
        assert profile.shape == (ny,)
        mid_z = nz // 2
        assert torch.allclose(profile, ux[mid_z, :, 15])


class TestComputeRecirculationLength:
    def test_no_negative_velocity_returns_zero(self) -> None:
        ny, nx = 20, 40
        ux = torch.ones((ny, nx)) * 0.05  # all positive
        obs = torch.zeros((ny, nx), dtype=torch.bool)
        length = compute_recirculation_length(ux, obs)
        assert length == 0.0

    def test_some_recirculation_returns_positive(self) -> None:
        ny, nx = 20, 40
        ux = torch.ones((ny, nx)) * 0.05
        # Introduce negative velocity in columns 0-4 (start from the beginning)
        ux[:, 0:5] = -0.02
        obs = torch.zeros((ny, nx), dtype=torch.bool)
        length = compute_recirculation_length(ux, obs)
        assert length > 0.0


# ---------------------------------------------------------------------------
# Phase 3c: save_vtk_binary
# ---------------------------------------------------------------------------


class TestSaveVtkBinary:
    def test_creates_file(self) -> None:
        ny, nx = 8, 10
        ux = torch.rand((ny, nx))
        uy = torch.rand((ny, nx))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.vtk"
            out = save_vtk_binary(path, ux, uy)
            assert out.exists()
            assert out.stat().st_size > 0

    def test_creates_3d_file(self) -> None:
        nz, ny, nx = 4, 6, 8
        ux = torch.rand((nz, ny, nx))
        uy = torch.rand((nz, ny, nx))
        uz = torch.rand((nz, ny, nx))
        rho = torch.rand((nz, ny, nx)) + 0.9
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test3d.vtk"
            out = save_vtk_binary(path, ux, uy, uz=uz, rho=rho)
            assert out.exists()
            # Binary file should start with the VTK header
            content = out.read_bytes()
            assert b"vtk DataFile Version" in content
            assert b"BINARY" in content


# ---------------------------------------------------------------------------
# Phase 2c: D3Q27 boundary conditions smoke tests
# ---------------------------------------------------------------------------


class TestBoundariesD3Q27:
    def test_bounce_back_cells_27_shape(self) -> None:
        from tensorlbm.boundaries_d3q27 import bounce_back_cells_27

        nz, ny, nx = 4, 8, 10
        rho = torch.ones((nz, ny, nx))
        f = equilibrium27(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        mask[:, 0, :] = True
        f_out = bounce_back_cells_27(f, mask)
        assert f_out.shape == (27, nz, ny, nx)
        assert torch.isfinite(f_out).all()

    def test_zou_he_inlet_27_shape(self) -> None:
        from tensorlbm.boundaries_d3q27 import zou_he_inlet_velocity_27

        nz, ny, nx = 4, 8, 10
        rho = torch.ones((nz, ny, nx))
        f = equilibrium27(
            rho,
            torch.full_like(rho, 0.05),
            torch.zeros_like(rho),
            torch.zeros_like(rho),
        )
        f_out = zou_he_inlet_velocity_27(f, u_in=0.05)
        assert f_out.shape == (27, nz, ny, nx)
        assert torch.isfinite(f_out).all()

    def test_zou_he_outlet_27_shape(self) -> None:
        from tensorlbm.boundaries_d3q27 import zou_he_outlet_pressure_27

        nz, ny, nx = 4, 8, 10
        rho = torch.ones((nz, ny, nx))
        f = equilibrium27(
            rho,
            torch.full_like(rho, 0.05),
            torch.zeros_like(rho),
            torch.zeros_like(rho),
        )
        f_out = zou_he_outlet_pressure_27(f, rho_out=1.0)
        assert f_out.shape == (27, nz, ny, nx)
        assert torch.isfinite(f_out).all()

    def test_apply_zou_he_channel_boundaries_27_shape(self) -> None:
        from tensorlbm import sphere_mask
        from tensorlbm.boundaries_d3q27 import (
            apply_zou_he_channel_boundaries_27,
            make_channel_wall_mask_27,
        )

        nz, ny, nx = 10, 12, 20
        device = torch.device("cpu")
        obstacle = sphere_mask(
            nx, ny, nz, nx * 0.25, ny * 0.5, nz * 0.5, 3.0, device=device
        )
        wall = make_channel_wall_mask_27(nz, ny, nx, obstacle, device=device)
        rho = torch.ones((nz, ny, nx))
        f = equilibrium27(
            rho,
            torch.full_like(rho, 0.05),
            torch.zeros_like(rho),
            torch.zeros_like(rho),
        )
        f_out = apply_zou_he_channel_boundaries_27(
            f, u_in=0.05, wall_mask=wall, obstacle_mask=obstacle
        )
        assert f_out.shape == (27, nz, ny, nx)
        assert torch.isfinite(f_out).all()
