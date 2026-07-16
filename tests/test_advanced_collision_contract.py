"""Contract tests for the advanced collision capability matrix.

These tests verify the audit boundary of ``advanced_collision_contract``:
which collision families are AVAILABLE (callable, validated kernel) versus
WITHHELD (no standalone validated kernel) for each lattice.

CUMULANT registration:
  * D3Q27 — AVAILABLE (``tensorlbm.cumulant.collide_cumulant_d3q27``)
  * D3Q19 — WITHHELD (no general D3Q19 cumulant kernel; the CG-specific
    ``collide_cg_cumulant_3d`` is a regularized-stress alias, not a cumulant
    transform)
"""
from __future__ import annotations

import pytest
import torch

from tensorlbm.advanced_collision_contract import (
    CollisionKernelWithheldError,
    WITHHELD_NO_D3Q19_CUMULANT_KERNEL,
    collide_advanced_3d,
    collision_capability_matrix,
)
from tensorlbm.cumulant import collide_cumulant_d3q27
from tensorlbm.d3q27 import equilibrium27, macroscopic27


# ---------------------------------------------------------------------------
# Matrix structure: CUMULANT present in both lattices
# ---------------------------------------------------------------------------

class TestCumulantMatrixRegistration:
    def test_d3q27_cumulant_is_available(self) -> None:
        cap = collision_capability_matrix()["D3Q27"]["CUMULANT"]
        assert cap.available
        assert cap.status == "AVAILABLE"
        assert cap.entrypoint == "tensorlbm.cumulant.collide_cumulant_d3q27"

    def test_d3q19_cumulant_is_withheld(self) -> None:
        cap = collision_capability_matrix()["D3Q19"]["CUMULANT"]
        assert not cap.available
        assert cap.status == WITHHELD_NO_D3Q19_CUMULANT_KERNEL
        assert cap.entrypoint is None

    def test_d3q19_cumulant_withheld_note_explains_no_general_kernel(self) -> None:
        cap = collision_capability_matrix()["D3Q19"]["CUMULANT"]
        assert "cumulant" in cap.note.lower()

    def test_both_lattices_have_cumulant_key(self) -> None:
        for lattice in ("D3Q19", "D3Q27"):
            assert "CUMULANT" in collision_capability_matrix()[lattice]


# ---------------------------------------------------------------------------
# collide_advanced_3d dispatch: CUMULANT
# ---------------------------------------------------------------------------

class TestCollideAdvanced3dCumulantDispatch:
    def _make_equilibrium(self, nz=4, ny=5, nx=6):
        rho = torch.ones((nz, ny, nx))
        f = equilibrium27(rho, torch.zeros_like(rho), torch.zeros_like(rho),
                          torch.zeros_like(rho))
        return f

    def test_d3q27_cumulant_dispatches_to_kernel(self) -> None:
        f = self._make_equilibrium()
        fout = collide_advanced_3d("D3Q27", "CUMULANT", f, tau=0.7)
        assert fout.shape == f.shape
        assert torch.isfinite(fout).all()

    def test_d3q27_cumulant_alias_cumulant_lowercase(self) -> None:
        f = self._make_equilibrium()
        fout = collide_advanced_3d("d3q27", "cumulant", f, tau=0.7)
        assert fout.shape == f.shape

    def test_d3q19_cumulant_raises_withheld(self) -> None:
        from tensorlbm.d3q19 import equilibrium3d
        rho = torch.ones((4, 5, 6))
        f = equilibrium3d(rho, torch.zeros_like(rho), torch.zeros_like(rho),
                          torch.zeros_like(rho))
        with pytest.raises(CollisionKernelWithheldError,
                           match=WITHHELD_NO_D3Q19_CUMULANT_KERNEL):
            collide_advanced_3d("D3Q19", "CUMULANT", f, tau=0.7)

    def test_d3q27_cumulant_rejects_wrong_population_count(self) -> None:
        # D3Q19 populations (19) passed to D3Q27 cumulant
        from tensorlbm.d3q19 import equilibrium3d
        rho = torch.ones((4, 5, 6))
        f = equilibrium3d(rho, torch.zeros_like(rho), torch.zeros_like(rho),
                          torch.zeros_like(rho))
        with pytest.raises(ValueError, match="27"):
            collide_advanced_3d("D3Q27", "CUMULANT", f, tau=0.7)


# ---------------------------------------------------------------------------
# Contract tests for the cumulant kernel via the contract dispatcher
# (shape, finite, mass, momentum, equilibrium identity)
# ---------------------------------------------------------------------------

class TestCumulantD3Q27ContractViaDispatcher:
    def _make_field(self, nz=4, ny=5, nx=6):
        rho = torch.rand((nz, ny, nx)) + 0.5
        ux = torch.rand_like(rho) * 0.04
        uy = torch.rand_like(rho) * 0.04
        uz = torch.rand_like(rho) * 0.04
        f = equilibrium27(rho, ux, uy, uz)
        return f, rho, ux, uy, uz

    def test_shape(self) -> None:
        f, *_ = self._make_field()
        fout = collide_advanced_3d("D3Q27", "CUMULANT", f, tau=0.7)
        assert fout.shape == f.shape

    def test_output_finite(self) -> None:
        f, *_ = self._make_field()
        fout = collide_advanced_3d("D3Q27", "CUMULANT", f, tau=0.7)
        assert torch.isfinite(fout).all()

    def test_mass_conservation(self) -> None:
        f, rho, *_ = self._make_field()
        fout = collide_advanced_3d("D3Q27", "CUMULANT", f, tau=0.7)
        rho_out, *_ = macroscopic27(fout)
        assert torch.allclose(rho_out, rho, atol=1e-5)

    def test_momentum_conservation(self) -> None:
        f, rho, ux, uy, uz = self._make_field()
        fout = collide_advanced_3d("D3Q27", "CUMULANT", f, tau=0.7)
        rho_out, ux_out, uy_out, uz_out = macroscopic27(fout)
        assert torch.allclose(rho_out, rho, atol=1e-5)
        assert torch.allclose(ux_out, ux, atol=1e-5)
        assert torch.allclose(uy_out, uy, atol=1e-5)
        assert torch.allclose(uz_out, uz, atol=1e-5)

    def test_equilibrium_is_identity(self) -> None:
        """At equilibrium (zero non-equilibrium) collision is identity."""
        rho = torch.ones((4, 5, 6))
        f = equilibrium27(rho, torch.zeros_like(rho), torch.zeros_like(rho),
                          torch.zeros_like(rho))
        fout = collide_advanced_3d("D3Q27", "CUMULANT", f, tau=0.7)
        assert torch.allclose(fout, f, atol=1e-5)


# ---------------------------------------------------------------------------
# Direct kernel contract tests (entrypoint matches the registered callable)
# ---------------------------------------------------------------------------

class TestCumulantD3Q27KernelContract:
    def _make_field(self, nz=4, ny=5, nx=6):
        rho = torch.rand((nz, ny, nx)) + 0.5
        ux = torch.rand_like(rho) * 0.04
        uy = torch.rand_like(rho) * 0.04
        uz = torch.rand_like(rho) * 0.04
        f = equilibrium27(rho, ux, uy, uz)
        return f, rho, ux, uy, uz

    def test_shape(self) -> None:
        f, *_ = self._make_field()
        assert collide_cumulant_d3q27(f, tau=0.7).shape == f.shape

    def test_finite(self) -> None:
        f, *_ = self._make_field()
        assert torch.isfinite(collide_cumulant_d3q27(f, tau=0.7)).all()

    def test_mass_conservation(self) -> None:
        f, rho, *_ = self._make_field()
        fout = collide_cumulant_d3q27(f, tau=0.7)
        rho_out, *_ = macroscopic27(fout)
        assert torch.allclose(rho_out, rho, atol=1e-5)

    def test_momentum_conservation(self) -> None:
        f, rho, ux, uy, uz = self._make_field()
        fout = collide_cumulant_d3q27(f, tau=0.7)
        rho_out, ux_out, uy_out, uz_out = macroscopic27(fout)
        assert torch.allclose(rho_out, rho, atol=1e-5)
        assert torch.allclose(ux_out, ux, atol=1e-5)
        assert torch.allclose(uy_out, uy, atol=1e-5)
        assert torch.allclose(uz_out, uz, atol=1e-5)

    def test_equilibrium_is_identity(self) -> None:
        rho = torch.ones((4, 5, 6))
        f = equilibrium27(rho, torch.zeros_like(rho), torch.zeros_like(rho),
                          torch.zeros_like(rho))
        assert torch.allclose(collide_cumulant_d3q27(f, tau=0.7), f, atol=1e-5)
