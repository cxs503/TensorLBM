"""Contract tests for CG multiphase + SGS turbulence model coupling.

Verifies that the Color-Gradient (CG) D3Q19 collision adapters accept an
optional ``sgs_model`` parameter selecting between Smagorinsky, WALE, and
Vreman sub-grid stress closures.

Default behaviour (no SGS, scalar tau) is preserved when
``sgs_model='smagorinsky'`` and ``C_s=0`` (the established waLBerla/OpenLB
pattern: ``C_s=0`` ⇒ pure collision, no eddy viscosity).

TDD: this file was written RED before the implementation was GREEN.
"""
from __future__ import annotations

import pytest
import torch

from tensorlbm import equilibrium3d, macroscopic3d, stream3d
from tensorlbm.cg_advanced_collision import (
    collide_cg_central_stress_3d,
    collide_cg_regularized_stress_3d,
)
from tensorlbm.turbulence_capability_contract import (
    IMPLEMENTED,
    NO_IMPLEMENTATION,
    VERIFICATION_CONTRACT_TESTED,
    turbulence_capability_matrix,
)

DEVICE = torch.device("cpu")

_SGS_MODELS = ["smagorinsky", "wale", "vreman"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cg_state(
    nz: int = 4, ny: int = 6, nx: int = 8, u_mag: float = 0.03, rho_ratio: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-component CG equilibrium with a density ratio and small velocity."""
    torch.manual_seed(42)
    rho_b = torch.rand((nz, ny, nx)) * 0.3 + 0.5
    rho_r = rho_b * rho_ratio
    ux = torch.randn_like(rho_b) * u_mag
    uy = torch.randn_like(rho_b) * u_mag
    uz = torch.randn_like(rho_b) * u_mag
    return equilibrium3d(rho_r, ux, uy, uz), equilibrium3d(rho_b, ux, uy, uz)


def _make_uniform_cg_state(
    nz: int = 4, ny: int = 6, nx: int = 8, u_mag: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform-density CG equilibrium with uniform velocity (identity test)."""
    rho_r = torch.full((nz, ny, nx), 0.6)
    rho_b = torch.full((nz, ny, nx), 0.4)
    ux = torch.full_like(rho_r, u_mag)
    uy = torch.full_like(rho_r, 0.0)
    uz = torch.full_like(rho_r, 0.0)
    return equilibrium3d(rho_r, ux, uy, uz), equilibrium3d(rho_b, ux, uy, uz)


def _make_droplet_cg_state(
    nz: int = 12, ny: int = 12, nx: int = 12, radius: float = 4.0, rho_ratio: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Spherical droplet of red phase inside blue phase (zero velocity)."""
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz), torch.arange(ny), torch.arange(nx), indexing="ij")
    cx, cy, cz = nz // 2, ny // 2, nx // 2
    r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2)
    inside = r < radius
    rho_r = torch.where(inside, rho_ratio, 0.1)
    rho_b = torch.where(inside, 0.1, 1.0)
    zero = torch.zeros_like(rho_r)
    return equilibrium3d(rho_r, zero, zero, zero), equilibrium3d(rho_b, zero, zero, zero)


# ---------------------------------------------------------------------------
# Parameter acceptance
# ---------------------------------------------------------------------------

class TestParameterAcceptance:
    """The sgs_model parameter must exist on both public CG stress adapters."""

    @pytest.mark.parametrize("func", [
        collide_cg_regularized_stress_3d, collide_cg_central_stress_3d])
    @pytest.mark.parametrize("sgs_model", _SGS_MODELS)
    def test_accepts_sgs_model(self, func, sgs_model):
        f_r, f_b = _make_cg_state()
        kwargs = {"sgs_model": sgs_model}
        if sgs_model == "smagorinsky":
            kwargs["C_s"] = 0.1
        red, blue = func(f_r, f_b, tau=0.8, **kwargs)
        assert red.shape == f_r.shape
        assert blue.shape == f_b.shape

    @pytest.mark.parametrize("func", [
        collide_cg_regularized_stress_3d, collide_cg_central_stress_3d])
    def test_invalid_sgs_model_raises(self, func):
        f_r, f_b = _make_cg_state()
        with pytest.raises(ValueError, match="sgs_model"):
            func(f_r, f_b, tau=0.8, sgs_model="k_omega")


# ---------------------------------------------------------------------------
# Default behaviour unchanged
# ---------------------------------------------------------------------------

class TestDefaultUnchanged:
    """Default call (no sgs_model, C_s=0) must equal the pre-SGS scalar-tau path."""

    @pytest.mark.parametrize("func", [
        collide_cg_regularized_stress_3d, collide_cg_central_stress_3d])
    def test_default_equals_explicit_smagorinsky_zero_cs(self, func):
        f_r, f_b = _make_cg_state()
        default = func(f_r.clone(), f_b.clone(), tau=0.85)
        explicit = func(
            f_r.clone(), f_b.clone(), tau=0.85,
            sgs_model="smagorinsky", C_s=0.0,
        )
        torch.testing.assert_close(default[0], explicit[0])
        torch.testing.assert_close(default[1], explicit[1])

    @pytest.mark.parametrize("func", [
        collide_cg_regularized_stress_3d, collide_cg_central_stress_3d])
    def test_default_equals_sgs_none(self, func):
        """sgs_model='smagorinsky' with C_s=0 is a no-op (scalar tau)."""
        f_r, f_b = _make_cg_state()
        ref = func(f_r.clone(), f_b.clone(), tau=0.85)
        wale_zero = func(
            f_r.clone(), f_b.clone(), tau=0.85,
            sgs_model="wale", C_w=0.0,
        )
        vreman_zero = func(
            f_r.clone(), f_b.clone(), tau=0.85,
            sgs_model="vreman", C_V=0.0,
        )
        torch.testing.assert_close(ref[0], wale_zero[0], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(ref[1], wale_zero[1], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(ref[0], vreman_zero[0], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(ref[1], vreman_zero[1], atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# WALE coupling contract
# ---------------------------------------------------------------------------

class TestWALECoupling:
    def test_shape(self):
        f_r, f_b = _make_cg_state()
        red, blue = collide_cg_regularized_stress_3d(f_r, f_b, tau=0.8, sgs_model="wale")
        assert red.shape == f_r.shape
        assert blue.shape == f_b.shape

    def test_finite(self):
        f_r, f_b = _make_cg_state()
        red, blue = collide_cg_regularized_stress_3d(f_r, f_b, tau=0.8, sgs_model="wale")
        assert torch.isfinite(red).all()
        assert torch.isfinite(blue).all()

    def test_mass_conservation(self):
        f_r, f_b = _make_cg_state()
        mass_before = (f_r + f_b).sum()
        red, blue = collide_cg_regularized_stress_3d(f_r, f_b, tau=0.8, sgs_model="wale")
        assert torch.allclose((red + blue).sum(), mass_before, atol=1e-4)

    def test_momentum_conservation(self):
        f_r, f_b = _make_cg_state()
        _, ux, uy, uz = macroscopic3d(f_r + f_b)
        red, blue = collide_cg_regularized_stress_3d(f_r, f_b, tau=0.8, sgs_model="wale")
        _, ux_o, uy_o, uz_o = macroscopic3d(red + blue)
        assert torch.allclose(ux_o, ux, atol=1e-4)
        assert torch.allclose(uy_o, uy, atol=1e-4)
        assert torch.allclose(uz_o, uz, atol=1e-4)

    def test_equilibrium_identity(self):
        """Uniform velocity → zero gradients → zero nu_t → identity at equilibrium."""
        f_r, f_b = _make_uniform_cg_state()
        red, blue = collide_cg_regularized_stress_3d(f_r, f_b, tau=0.8, sgs_model="wale")
        torch.testing.assert_close(red, f_r, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(blue, f_b, atol=1e-5, rtol=1e-5)

    def test_central_stress_variant(self):
        """Central-stress adapter also accepts WALE."""
        f_r, f_b = _make_cg_state()
        red, blue = collide_cg_central_stress_3d(
            f_r, f_b, tau=0.8, sgs_model="wale", s_bulk=1.3)
        assert torch.isfinite(red).all()
        assert torch.isfinite(blue).all()


# ---------------------------------------------------------------------------
# Vreman coupling contract
# ---------------------------------------------------------------------------

class TestVremanCoupling:
    def test_shape(self):
        f_r, f_b = _make_cg_state()
        red, blue = collide_cg_regularized_stress_3d(f_r, f_b, tau=0.8, sgs_model="vreman")
        assert red.shape == f_r.shape
        assert blue.shape == f_b.shape

    def test_finite(self):
        f_r, f_b = _make_cg_state()
        red, blue = collide_cg_regularized_stress_3d(f_r, f_b, tau=0.8, sgs_model="vreman")
        assert torch.isfinite(red).all()
        assert torch.isfinite(blue).all()

    def test_mass_conservation(self):
        f_r, f_b = _make_cg_state()
        mass_before = (f_r + f_b).sum()
        red, blue = collide_cg_regularized_stress_3d(f_r, f_b, tau=0.8, sgs_model="vreman")
        assert torch.allclose((red + blue).sum(), mass_before, atol=1e-4)

    def test_momentum_conservation(self):
        f_r, f_b = _make_cg_state()
        _, ux, uy, uz = macroscopic3d(f_r + f_b)
        red, blue = collide_cg_regularized_stress_3d(f_r, f_b, tau=0.8, sgs_model="vreman")
        _, ux_o, uy_o, uz_o = macroscopic3d(red + blue)
        assert torch.allclose(ux_o, ux, atol=1e-4)
        assert torch.allclose(uy_o, uy, atol=1e-4)
        assert torch.allclose(uz_o, uz, atol=1e-4)

    def test_equilibrium_identity(self):
        f_r, f_b = _make_uniform_cg_state()
        red, blue = collide_cg_regularized_stress_3d(f_r, f_b, tau=0.8, sgs_model="vreman")
        torch.testing.assert_close(red, f_r, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(blue, f_b, atol=1e-5, rtol=1e-5)

    def test_central_stress_variant(self):
        f_r, f_b = _make_cg_state()
        red, blue = collide_cg_central_stress_3d(
            f_r, f_b, tau=0.8, sgs_model="vreman", s_bulk=1.3)
        assert torch.isfinite(red).all()
        assert torch.isfinite(blue).all()


# ---------------------------------------------------------------------------
# Smagorinsky coupling contract (C_s > 0)
# ---------------------------------------------------------------------------

class TestSmagorinskyCoupling:
    def test_shape(self):
        f_r, f_b = _make_cg_state()
        red, blue = collide_cg_regularized_stress_3d(
            f_r, f_b, tau=0.8, sgs_model="smagorinsky", C_s=0.1)
        assert red.shape == f_r.shape

    def test_finite(self):
        f_r, f_b = _make_cg_state()
        red, blue = collide_cg_regularized_stress_3d(
            f_r, f_b, tau=0.8, sgs_model="smagorinsky", C_s=0.1)
        assert torch.isfinite(red).all()

    def test_mass_conservation(self):
        f_r, f_b = _make_cg_state()
        mass_before = (f_r + f_b).sum()
        red, blue = collide_cg_regularized_stress_3d(
            f_r, f_b, tau=0.8, sgs_model="smagorinsky", C_s=0.1)
        assert torch.allclose((red + blue).sum(), mass_before, atol=1e-4)

    def test_equilibrium_identity(self):
        f_r, f_b = _make_uniform_cg_state()
        red, blue = collide_cg_regularized_stress_3d(
            f_r, f_b, tau=0.8, sgs_model="smagorinsky", C_s=0.1)
        torch.testing.assert_close(red, f_r, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(blue, f_b, atol=1e-5, rtol=1e-5)

    def test_cs_positive_changes_output(self):
        """C_s > 0 must produce a different result from C_s = 0."""
        f_r, f_b = _make_cg_state()
        # Add a non-equilibrium perturbation so the Smagorinsky stress
        # magnitude is non-zero and tau_eff differs from the scalar tau.
        torch.manual_seed(99)
        f_r = f_r + 1e-3 * torch.randn_like(f_r)
        f_b = f_b + 1e-3 * torch.randn_like(f_b)
        ref, _ = collide_cg_regularized_stress_3d(f_r.clone(), f_b.clone(), tau=0.8)
        sgs, _ = collide_cg_regularized_stress_3d(
            f_r.clone(), f_b.clone(), tau=0.8, sgs_model="smagorinsky", C_s=0.15)
        assert not torch.equal(ref, sgs)


# ---------------------------------------------------------------------------
# Stability: static droplet with streaming
# ---------------------------------------------------------------------------

class TestStability:
    @pytest.mark.parametrize("sgs_model", _SGS_MODELS)
    def test_static_droplet_stable(self, sgs_model):
        """Static droplet must remain finite and mass-conserving over 20 steps."""
        f_r, f_b = _make_droplet_cg_state()
        mass0 = (f_r + f_b).sum()
        kwargs = {"sgs_model": sgs_model}
        if sgs_model == "smagorinsky":
            kwargs["C_s"] = 0.1
        for _ in range(20):
            f_r, f_b = collide_cg_regularized_stress_3d(
                f_r, f_b, tau=0.9, A=0.04, beta=0.7, **kwargs)
            f_r = stream3d(f_r)
            f_b = stream3d(f_b)
        assert torch.isfinite(f_r).all()
        assert torch.isfinite(f_b).all()
        assert torch.allclose((f_r + f_b).sum(), mass0, atol=1e-2)

    @pytest.mark.parametrize("sgs_model", ["wale", "vreman"])
    def test_density_ratio_stability(self, sgs_model):
        """High density ratio (10:1) must remain stable with WALE/Vreman SGS."""
        f_r, f_b = _make_droplet_cg_state(rho_ratio=10.0, radius=4.0)
        mass0 = (f_r + f_b).sum()
        for _ in range(15):
            f_r, f_b = collide_cg_regularized_stress_3d(
                f_r, f_b, tau=0.95, A=0.04, beta=0.7, sgs_model=sgs_model)
            f_r = stream3d(f_r)
            f_b = stream3d(f_b)
        assert torch.isfinite(f_r).all()
        assert torch.isfinite(f_b).all()
        assert torch.allclose((f_r + f_b).sum(), mass0, atol=1e-2)


# ---------------------------------------------------------------------------
# Capability contract matrix
# ---------------------------------------------------------------------------

class TestCapabilityContractCGSGS:
    def test_matrix_includes_cg_collision_type(self):
        matrix = turbulence_capability_matrix()
        for family in ("smagorinsky", "wale", "vreman"):
            assert "CG" in matrix[family]["D3Q19"], family

    @pytest.mark.parametrize("sgs_model", _SGS_MODELS)
    def test_cg_sgs_combination_implemented_and_contract_tested(self, sgs_model):
        cap = turbulence_capability_matrix()[sgs_model]["D3Q19"]["CG"]
        assert cap.implementation_status == IMPLEMENTED, sgs_model
        assert cap.verification_level == VERIFICATION_CONTRACT_TESTED, sgs_model
        assert cap.entrypoint is not None
        assert "cg_advanced_collision" in cap.entrypoint
        assert cap.test_evidence is not None

    def test_cg_collision_not_implemented_for_d2q9(self):
        """CG is a D3Q19-only multiphase model; D2Q9 CG entries are withheld."""
        matrix = turbulence_capability_matrix()
        for family in ("smagorinsky", "wale", "vreman"):
            cap = matrix[family]["D2Q9"]["CG"]
            assert cap.implementation_status == NO_IMPLEMENTATION

    def test_cg_collision_not_implemented_for_d3q27(self):
        matrix = turbulence_capability_matrix()
        for family in ("smagorinsky", "wale", "vreman"):
            cap = matrix[family]["D3Q27"]["CG"]
            assert cap.implementation_status == NO_IMPLEMENTATION

    def test_cg_entries_fail_closed(self):
        """No CG+SGS combination is physics-validated."""
        matrix = turbulence_capability_matrix()
        for family in ("smagorinsky", "wale", "vreman"):
            cap = matrix[family]["D3Q19"]["CG"]
            assert not cap.available
            assert cap.status.startswith("WITHHELD_")
