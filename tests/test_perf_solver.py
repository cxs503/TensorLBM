"""TDD contract tests for the optimized solver module (perf_solver.py).

These tests verify that the optimized in-place collide/stream functions
and the OptimizedSolver3D produce **numerically identical** results to
the reference implementations, while using pre-allocated buffers.

Key invariant: allclose(atol=1e-6) between optimized and reference outputs.
"""
from __future__ import annotations

import pytest
import torch

from tensorlbm.perf_buffers import LBMStepBuffer
from tensorlbm.perf_solver import (
    OptimizedSolver3D,
    collide_bgk3d_inplace,
    collide_bgk27_inplace,
    stream3d_inplace,
    stream27_inplace,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_equilibrium(lattice: str, nz=4, ny=6, nx=8, u=0.05):
    from tensorlbm.d3q19 import equilibrium3d
    from tensorlbm.d3q27 import equilibrium27

    rho = torch.ones((nz, ny, nx))
    ux = torch.full((nz, ny, nx), u)
    uy = torch.zeros((nz, ny, nx))
    uz = torch.zeros((nz, ny, nx))
    if lattice == "D3Q19":
        return equilibrium3d(rho, ux, uy, uz)
    return equilibrium27(rho, ux, uy, uz)


# ---------------------------------------------------------------------------
# In-place BGK collision — numerical identity
# ---------------------------------------------------------------------------

class TestCollideBGKInplace:
    """In-place BGK collision must match the reference exactly."""

    @pytest.mark.parametrize("tau", [0.6, 0.8, 1.5])
    def test_d3q19_bgk_inplace_matches_reference(self, tau):
        from tensorlbm import collide_bgk3d

        f_ref = _make_equilibrium("D3Q19")
        f_opt = f_ref.clone()

        # Add perturbation so collision is non-trivial
        f_ref += 0.01 * torch.randn_like(f_ref)
        f_opt = f_ref.clone()

        buf = LBMStepBuffer.for_lattice("D3Q19", 4, 6, 8)
        collide_bgk3d_inplace(f_opt, tau, buf)

        f_ref_out = collide_bgk3d(f_ref, tau)
        assert torch.allclose(f_opt, f_ref_out, atol=1e-6)

    @pytest.mark.parametrize("tau", [0.6, 0.8, 1.5])
    def test_d3q27_bgk_inplace_matches_reference(self, tau):
        from tensorlbm.d3q27 import collide_bgk27

        f_ref = _make_equilibrium("D3Q27")
        f_ref += 0.01 * torch.randn_like(f_ref)
        f_opt = f_ref.clone()

        buf = LBMStepBuffer.for_lattice("D3Q27", 4, 6, 8)
        collide_bgk27_inplace(f_opt, tau, buf)

        f_ref_out = collide_bgk27(f_ref, tau)
        assert torch.allclose(f_opt, f_ref_out, atol=1e-6)

    def test_d3q19_bgk_inplace_writes_macroscopic_to_buffer(self):
        """The in-place collision must populate buf.rho/ux/uy/uz."""
        from tensorlbm.d3q19 import macroscopic3d

        f = _make_equilibrium("D3Q19")
        f += 0.01 * torch.randn_like(f)
        buf = LBMStepBuffer.for_lattice("D3Q19", 4, 6, 8)

        collide_bgk3d_inplace(f, 0.6, buf)

        ref_rho, ref_ux, ref_uy, ref_uz = macroscopic3d(f)
        assert torch.allclose(buf.rho, ref_rho, atol=1e-6)
        assert torch.allclose(buf.ux, ref_ux, atol=1e-6)

    def test_d3q19_bgk_inplace_modifies_f_inplace(self):
        """The in-place collision must modify f in-place (same storage)."""
        f = _make_equilibrium("D3Q19")
        f += 0.01 * torch.randn_like(f)
        ptr_before = f.data_ptr()
        buf = LBMStepBuffer.for_lattice("D3Q19", 4, 6, 8)
        collide_bgk3d_inplace(f, 0.6, buf)
        assert f.data_ptr() == ptr_before, "f storage changed (not in-place)"


# ---------------------------------------------------------------------------
# In-place streaming — numerical identity
# ---------------------------------------------------------------------------

class TestStreamInplace:
    """In-place streaming must match the reference exactly."""

    def test_d3q19_stream_inplace_matches_reference(self):
        from tensorlbm import stream3d

        f_ref = _make_equilibrium("D3Q19")
        f_ref += 0.01 * torch.randn_like(f_ref)

        buf = LBMStepBuffer.for_lattice("D3Q19", 4, 6, 8)
        stream3d_inplace(f_ref, buf)

        f_ref2 = _make_equilibrium("D3Q19")
        f_ref2 += 0.01 * torch.randn_like(f_ref2)
        # Use same data
        f_ref2 = f_ref.clone()  # undo the stream by re-cloning original
        # Actually, let's just compare with reference stream3d on same input
        f_input = _make_equilibrium("D3Q19")
        f_input += 0.01 * torch.randn_like(f_input)

        buf2 = LBMStepBuffer.for_lattice("D3Q19", 4, 6, 8)
        stream3d_inplace(f_input, buf2)
        f_streamed = buf2.f_stream

        ref_streamed = stream3d(f_input.clone())
        assert torch.allclose(f_streamed, ref_streamed, atol=1e-6)

    def test_d3q27_stream_inplace_matches_reference(self):
        from tensorlbm.d3q27 import stream27

        f_input = _make_equilibrium("D3Q27")
        f_input += 0.01 * torch.randn_like(f_input)

        buf = LBMStepBuffer.for_lattice("D3Q27", 4, 6, 8)
        stream27_inplace(f_input, buf)
        f_streamed = buf.f_stream

        ref_streamed = stream27(f_input.clone())
        assert torch.allclose(f_streamed, ref_streamed, atol=1e-6)

    def test_d3q19_stream_inplace_writes_to_f_stream(self):
        """Streaming result must be in buf.f_stream, not overwriting input."""
        f_input = _make_equilibrium("D3Q19")
        f_input += 0.01 * torch.randn_like(f_input)
        f_input_copy = f_input.clone()

        buf = LBMStepBuffer.for_lattice("D3Q19", 4, 6, 8)
        stream3d_inplace(f_input, buf)

        # Input must be unchanged
        assert torch.allclose(f_input, f_input_copy, atol=1e-7)
        # Output must be in buf.f_stream
        assert not torch.allclose(buf.f_stream, f_input, atol=1e-6)


# ---------------------------------------------------------------------------
# OptimizedSolver3D — full step numerical identity
# ---------------------------------------------------------------------------

class TestOptimizedSolver3D:
    """OptimizedSolver3D produces identical results to a reference loop."""

    def test_d3q19_bgk_step_matches_reference(self):
        """One optimized step must match the reference collide→stream→BC loop."""
        from tensorlbm import collide_bgk3d, stream3d, equilibrium3d
        from tensorlbm.boundaries3d import far_field_bc_3d

        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        ux = torch.full((nz, ny, nx), 0.05)
        uy = torch.zeros((nz, ny, nx))
        uz = torch.zeros((nz, ny, nx))
        f_init = equilibrium3d(rho, ux, uy, uz)
        f_init += 0.01 * torch.randn_like(f_init)

        # Reference
        f_ref = f_init.clone()
        f_ref = collide_bgk3d(f_ref, tau=0.6)
        f_ref = stream3d(f_ref)
        f_ref = far_field_bc_3d(f_ref, 0.05)

        # Optimized
        solver = OptimizedSolver3D(
            lattice="D3Q19", nz=nz, ny=ny, nx=nx,
            tau=0.6, device=torch.device("cpu"),
        )
        f_opt = f_init.clone()
        f_opt = solver.step(f_opt, u_in=0.05)

        assert torch.allclose(f_opt, f_ref, atol=1e-6)

    def test_d3q27_bgk_step_matches_reference(self):
        from tensorlbm.d3q27 import collide_bgk27, stream27, equilibrium27
        from tensorlbm.boundaries_d3q27 import far_field_bc_27

        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        ux = torch.full((nz, ny, nx), 0.05)
        uy = torch.zeros((nz, ny, nx))
        uz = torch.zeros((nz, ny, nx))
        f_init = equilibrium27(rho, ux, uy, uz)
        f_init += 0.01 * torch.randn_like(f_init)

        # Reference
        f_ref = f_init.clone()
        f_ref = collide_bgk27(f_ref, tau=0.6)
        f_ref = stream27(f_ref)
        f_ref = far_field_bc_27(f_ref, 0.05)

        # Optimized
        solver = OptimizedSolver3D(
            lattice="D3Q27", nz=nz, ny=ny, nx=nx,
            tau=0.6, device=torch.device("cpu"),
        )
        f_opt = f_init.clone()
        f_opt = solver.step(f_opt, u_in=0.05)

        assert torch.allclose(f_opt, f_ref, atol=1e-6)

    def test_multi_step_stability(self):
        """Multiple optimized steps must stay finite and close to reference."""
        from tensorlbm import collide_bgk3d, stream3d, equilibrium3d
        from tensorlbm.boundaries3d import far_field_bc_3d

        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        ux = torch.full((nz, ny, nx), 0.05)
        uy = torch.zeros((nz, ny, nx))
        uz = torch.zeros((nz, ny, nx))
        f_init = equilibrium3d(rho, ux, uy, uz)

        f_ref = f_init.clone()
        f_opt = f_init.clone()

        solver = OptimizedSolver3D(
            lattice="D3Q19", nz=nz, ny=ny, nx=nx,
            tau=0.6, device=torch.device("cpu"),
        )

        for _ in range(10):
            f_ref = collide_bgk3d(f_ref, tau=0.6)
            f_ref = stream3d(f_ref)
            f_ref = far_field_bc_3d(f_ref, 0.05)

            f_opt = solver.step(f_opt, u_in=0.05)

        assert torch.allclose(f_opt, f_ref, atol=1e-6)
        assert torch.isfinite(f_opt).all()

    def test_accepts_external_collide_fn(self):
        """The solver must accept an external collide function (e.g. MRT)."""
        from tensorlbm import collide_mrt3d, stream3d, equilibrium3d
        from tensorlbm.boundaries3d import far_field_bc_3d

        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        ux = torch.full((nz, ny, nx), 0.05)
        f_init = equilibrium3d(rho, ux, torch.zeros_like(rho), torch.zeros_like(rho))

        f_ref = f_init.clone()
        f_ref = collide_mrt3d(f_ref, tau=0.6)
        f_ref = stream3d(f_ref)
        f_ref = far_field_bc_3d(f_ref, 0.05)

        solver = OptimizedSolver3D(
            lattice="D3Q19", nz=nz, ny=ny, nx=nx,
            tau=0.6, device=torch.device("cpu"),
        )
        f_opt = f_init.clone()
        f_opt = solver.step(f_opt, u_in=0.05, collide_fn=collide_mrt3d)

        assert torch.allclose(f_opt, f_ref, atol=1e-6)

    def test_wall_function_with_precomputed_macroscopic(self):
        """wall_function must accept pre-computed rho/ux/uy/uz from buffer."""
        from tensorlbm import collide_bgk3d, stream3d, equilibrium3d
        from tensorlbm.boundaries3d import far_field_bc_3d, bounce_back_cells_3d
        from tensorlbm.wall_function_common import (
            compute_u_tau, compute_y_plus, wall_function,
        )

        nz, ny, nx = 6, 8, 10
        rho = torch.ones((nz, ny, nx))
        ux = torch.full((nz, ny, nx), 0.1)
        uy = torch.zeros((nz, ny, nx))
        uz = torch.zeros((nz, ny, nx))
        f_init = equilibrium3d(rho, ux, uy, uz)

        mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        mask[0, :, :] = True
        mask[-1, :, :] = True

        # Reference: standard wall_function (computes macroscopic internally)
        f_ref = f_init.clone()
        f_ref = collide_bgk3d(f_ref, tau=0.6)
        f_ref = stream3d(f_ref)
        f_ref = far_field_bc_3d(f_ref, 0.1)
        u_mag_ref = torch.sqrt(
            ux**2 + uy**2 + uz**2
        ).clamp(min=1e-12)
        u_tau_ref = compute_u_tau(u_mag_ref, nu=0.02, y_val=0.5)
        y_plus_ref = compute_y_plus(u_tau_ref, nu=0.02, y_val=0.5)
        f_ref = wall_function(
            f_ref, mask, u_tau_ref, y_plus_ref,
            lattice="D3Q19", nu=0.02,
        )

        # Optimized: pass pre-computed macroscopic
        f_opt = f_init.clone()
        solver = OptimizedSolver3D(
            lattice="D3Q19", nz=nz, ny=ny, nx=nx,
            tau=0.6, device=torch.device("cpu"),
        )
        f_opt = solver.step(
            f_opt, u_in=0.1,
            wall_mask=mask, nu=0.02,
        )

        assert torch.allclose(f_opt, f_ref, atol=1e-6)


# ---------------------------------------------------------------------------
# Buffer reuse across steps — no per-step allocation
# ---------------------------------------------------------------------------

class TestSolverBufferReuse:
    """The solver must reuse the same buffer storage across steps."""

    def test_buffer_storage_persists_across_steps(self):
        from tensorlbm import equilibrium3d

        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        f = equilibrium3d(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))

        solver = OptimizedSolver3D(
            lattice="D3Q19", nz=nz, ny=ny, nx=nx,
            tau=0.6, device=torch.device("cpu"),
        )
        ptr_feq = solver.buf.feq.data_ptr()
        ptr_rho = solver.buf.rho.data_ptr()

        for _ in range(5):
            f = solver.step(f, u_in=0.05)

        assert solver.buf.feq.data_ptr() == ptr_feq
        assert solver.buf.rho.data_ptr() == ptr_rho
