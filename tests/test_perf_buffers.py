"""TDD contract tests for the performance buffer module (perf_buffers.py).

These tests specify the pre-allocated buffer interface before the
implementation is written.  The buffer module must:

* Pre-allocate all temporary tensors needed for one LBM step.
* Support D3Q19 (Q=19) and D3Q27 (Q=27).
* Reuse buffers across steps (no per-step allocation).
* Provide macroscopic field buffers (rho, ux, uy, uz, u_mag).
* Provide wall-function buffers (u_tau, y_plus).
"""
from __future__ import annotations

import pytest
import torch

from tensorlbm.perf_buffers import LBMStepBuffer


# ---------------------------------------------------------------------------
# Construction and shape contracts
# ---------------------------------------------------------------------------

class TestLBMStepBufferConstruction:
    """LBMStepBuffer allocates correctly-shaped tensors for each lattice."""

    @pytest.mark.parametrize("lattice,q", [("D3Q19", 19), ("D3Q27", 27)])
    def test_for_lattice_creates_correct_q(self, lattice, q):
        buf = LBMStepBuffer.for_lattice(lattice, nz=4, ny=6, nx=8)
        assert buf.q == q

    @pytest.mark.parametrize("lattice,q", [("D3Q19", 19), ("D3Q27", 27)])
    def test_distribution_buffers_have_q_shape(self, lattice, q):
        buf = LBMStepBuffer.for_lattice(lattice, nz=4, ny=6, nx=8)
        for name in ("f_post", "feq", "f_stream", "fneq"):
            t = getattr(buf, name)
            assert t.shape == (q, 4, 6, 8), f"{name} shape mismatch"

    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_macroscopic_buffers_have_3d_shape(self, lattice):
        buf = LBMStepBuffer.for_lattice(lattice, nz=4, ny=6, nx=8)
        for name in ("rho", "ux", "uy", "uz", "u_mag"):
            t = getattr(buf, name)
            assert t.shape == (4, 6, 8), f"{name} shape mismatch"

    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_wall_function_buffers_have_3d_shape(self, lattice):
        buf = LBMStepBuffer.for_lattice(lattice, nz=4, ny=6, nx=8)
        for name in ("u_tau", "y_plus"):
            t = getattr(buf, name)
            assert t.shape == (4, 6, 8), f"{name} shape mismatch"

    def test_rejects_unsupported_lattice(self):
        with pytest.raises(ValueError, match="Unsupported lattice"):
            LBMStepBuffer.for_lattice("D2Q9", nz=4, ny=6, nx=8)

    def test_dtype_defaults_to_float32(self):
        buf = LBMStepBuffer.for_lattice("D3Q19", nz=4, ny=6, nx=8)
        assert buf.feq.dtype == torch.float32
        assert buf.rho.dtype == torch.float32

    def test_custom_dtype(self):
        buf = LBMStepBuffer.for_lattice(
            "D3Q19", nz=4, ny=6, nx=8, dtype=torch.float64
        )
        assert buf.feq.dtype == torch.float64

    def test_device(self):
        buf = LBMStepBuffer.for_lattice(
            "D3Q19", nz=4, ny=6, nx=8, device=torch.device("cpu")
        )
        assert buf.feq.device.type == "cpu"


# ---------------------------------------------------------------------------
# Buffer reuse — no per-step allocation
# ---------------------------------------------------------------------------

class TestLBMStepBufferReuse:
    """Buffers are persistent objects that can be written to repeatedly."""

    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_write_twice_same_storage(self, lattice):
        """Writing to a buffer twice must use the same underlying storage."""
        buf = LBMStepBuffer.for_lattice(lattice, nz=4, ny=6, nx=8)
        ptr1 = buf.feq.data_ptr()
        buf.feq.fill_(1.0)
        buf.feq.fill_(2.0)
        ptr2 = buf.feq.data_ptr()
        assert ptr1 == ptr2, "buffer storage changed between writes"

    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_macroscopic_buffers_persist(self, lattice):
        buf = LBMStepBuffer.for_lattice(lattice, nz=4, ny=6, nx=8)
        ptr_rho = buf.rho.data_ptr()
        buf.rho.fill_(3.0)
        assert buf.rho.data_ptr() == ptr_rho


# ---------------------------------------------------------------------------
# Macroscopic computation into pre-allocated buffers
# ---------------------------------------------------------------------------

class TestBufferMacroscopic:
    """compute_macroscopic_into writes rho/ux/uy/uz into buffer fields."""

    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_macroscopic_into_matches_reference(self, lattice):
        from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
        from tensorlbm.d3q27 import equilibrium27, macroscopic27

        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        ux = torch.full((nz, ny, nx), 0.05)
        uy = torch.zeros((nz, ny, nx))
        uz = torch.zeros((nz, ny, nx))

        if lattice == "D3Q19":
            f = equilibrium3d(rho, ux, uy, uz)
            ref_rho, ref_ux, ref_uy, ref_uz = macroscopic3d(f)
        else:
            f = equilibrium27(rho, ux, uy, uz)
            ref_rho, ref_ux, ref_uy, ref_uz = macroscopic27(f)

        buf = LBMStepBuffer.for_lattice(lattice, nz, ny, nx)
        buf.compute_macroscopic_into(f, lattice=lattice)

        assert torch.allclose(buf.rho, ref_rho, atol=1e-6)
        assert torch.allclose(buf.ux, ref_ux, atol=1e-6)
        assert torch.allclose(buf.uy, ref_uy, atol=1e-6)
        assert torch.allclose(buf.uz, ref_uz, atol=1e-6)
