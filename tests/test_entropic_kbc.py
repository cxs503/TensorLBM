"""TDD tests for the complete entropic KBC collision (D3Q19 / D3Q27).

These tests exercise:
  1. Discrete entropy functional H(f) = Σ f_i ln(f_i / w_i)
  2. KBC decomposition f = f_eq + k + s + h (kinetic / shear / higher-order)
  3. Entropy-condition gamma solver (minimise H(f_eq + γ·s + h))
  4. Positivity / admissibility-domain enforcement
  5. Per-cell nonlinear gamma solve (bisection)
  6. collide_kbc_d3q19 / collide_kbc_d3q27 conservation and H-theorem
  7. Contract registration (KBC AVAILABLE for both lattices)
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.d3q19 import C as C19, W as W19, equilibrium3d, macroscopic3d
from tensorlbm.d3q27 import C as C27, W as W27, equilibrium27, macroscopic27


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state_19(seed: int = 19) -> torch.Tensor:
    torch.manual_seed(seed)
    rho = 0.9 + torch.rand(2, 3, 4)
    ux = 0.03 * torch.randn_like(rho)
    uy = 0.03 * torch.randn_like(rho)
    uz = 0.03 * torch.randn_like(rho)
    feq = equilibrium3d(rho, ux, uy, uz)
    return feq + 1.0e-3 * torch.randn_like(feq)


def _state_27(seed: int = 27) -> torch.Tensor:
    torch.manual_seed(seed)
    rho = 0.9 + torch.rand(2, 3, 4)
    ux = 0.03 * torch.randn_like(rho)
    uy = 0.03 * torch.randn_like(rho)
    uz = 0.03 * torch.randn_like(rho)
    feq = equilibrium27(rho, ux, uy, uz)
    return feq + 1.0e-3 * torch.randn_like(feq)


# ---------------------------------------------------------------------------
# 1. Discrete entropy functional
# ---------------------------------------------------------------------------

class TestDiscreteEntropy:
    """H(f) = Σ_i f_i ln(f_i / w_i)."""

    def test_entropy_of_equilibrium_is_finite_and_positive(self):
        from tensorlbm.entropic_kbc import discrete_entropy

        rho = torch.ones(2, 3, 4)
        zero = torch.zeros_like(rho)
        feq = equilibrium3d(rho, zero, zero, zero)
        w = W19.to(feq.device).view(19, 1, 1, 1)
        H = discrete_entropy(feq, w)
        assert torch.isfinite(H).all()
        # For rho=1, u=0: f_eq_i = w_i, so H = Σ w_i ln(1) = 0
        torch.testing.assert_close(H, torch.zeros_like(H), atol=1e-5, rtol=1e-5)

    def test_entropy_decreases_for_more_equilibrium(self):
        from tensorlbm.entropic_kbc import discrete_entropy

        f = _state_19()
        rho, ux, uy, uz = macroscopic3d(f)
        feq = equilibrium3d(rho, ux, uy, uz)
        w = W19.to(f.device).view(19, 1, 1, 1)
        H_f = discrete_entropy(f, w)
        H_feq = discrete_entropy(feq, w)
        # Equilibrium has lower (more negative) entropy than non-equilibrium
        assert (H_feq <= H_f + 1e-6).all()

    def test_entropy_d3q27_equilibrium(self):
        from tensorlbm.entropic_kbc import discrete_entropy

        rho = torch.ones(2, 3, 4)
        zero = torch.zeros_like(rho)
        feq = equilibrium27(rho, zero, zero, zero)
        w = W27.to(feq.device).view(27, 1, 1, 1)
        H = discrete_entropy(feq, w)
        torch.testing.assert_close(H, torch.zeros_like(H), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 2. KBC decomposition: f_neq = k + s + h
# ---------------------------------------------------------------------------

class TestKBCDecomposition:
    """f_neq = k + s + h where s = shear (deviatoric), k = kinetic (bulk), h = higher-order."""

    def test_decomposition_sums_to_fneq_d3q19(self):
        from tensorlbm.entropic_kbc import kbc_decompose_d3q19

        f = _state_19()
        rho, ux, uy, uz = macroscopic3d(f)
        feq = equilibrium3d(rho, ux, uy, uz)
        f_neq = f - feq
        s, k, h = kbc_decompose_d3q19(f_neq)
        torch.testing.assert_close(s + k + h, f_neq, atol=1e-6, rtol=1e-6)

    def test_decomposition_sums_to_fneq_d3q27(self):
        from tensorlbm.entropic_kbc import kbc_decompose_d3q27

        f = _state_27()
        rho, ux, uy, uz = macroscopic27(f)
        feq = equilibrium27(rho, ux, uy, uz)
        f_neq = f - feq
        s, k, h = kbc_decompose_d3q27(f_neq)
        torch.testing.assert_close(s + k + h, f_neq, atol=1e-6, rtol=1e-6)

    def test_shear_is_traceless_second_order_d3q19(self):
        """The shear part s must have zero trace (bulk) in its second-order moment."""
        from tensorlbm.entropic_kbc import kbc_decompose_d3q19

        f = _state_19()
        rho, ux, uy, uz = macroscopic3d(f)
        feq = equilibrium3d(rho, ux, uy, uz)
        f_neq = f - feq
        s, k, h = kbc_decompose_d3q19(f_neq)
        c = C19.to(f.device).float()
        cx, cy, cz = c[:, 0].view(19, 1, 1, 1), c[:, 1].view(19, 1, 1, 1), c[:, 2].view(19, 1, 1, 1)
        # Trace of s's second-order moment should be ~0 (deviatoric)
        trace_s = (cx * cx * s).sum(0) + (cy * cy * s).sum(0) + (cz * cz * s).sum(0)
        torch.testing.assert_close(trace_s, torch.zeros_like(trace_s), atol=1e-6, rtol=1e-6)

    def test_shear_is_traceless_second_order_d3q27(self):
        from tensorlbm.entropic_kbc import kbc_decompose_d3q27

        f = _state_27()
        rho, ux, uy, uz = macroscopic27(f)
        feq = equilibrium27(rho, ux, uy, uz)
        f_neq = f - feq
        s, k, h = kbc_decompose_d3q27(f_neq)
        c = C27.to(f.device).float()
        cx, cy, cz = c[:, 0].view(27, 1, 1, 1), c[:, 1].view(27, 1, 1, 1), c[:, 2].view(27, 1, 1, 1)
        trace_s = (cx * cx * s).sum(0) + (cy * cy * s).sum(0) + (cz * cz * s).sum(0)
        torch.testing.assert_close(trace_s, torch.zeros_like(trace_s), atol=1e-6, rtol=1e-6)

    def test_kinetic_captures_bulk_trace_d3q19(self):
        """k should capture the trace (bulk) part of the second-order stress."""
        from tensorlbm.entropic_kbc import kbc_decompose_d3q19

        f = _state_19()
        rho, ux, uy, uz = macroscopic3d(f)
        feq = equilibrium3d(rho, ux, uy, uz)
        f_neq = f - feq
        s, k, h = kbc_decompose_d3q19(f_neq)
        c = C19.to(f.device).float()
        cx, cy, cz = c[:, 0].view(19, 1, 1, 1), c[:, 1].view(19, 1, 1, 1), c[:, 2].view(19, 1, 1, 1)
        # The trace of f_neq's second-order moment should equal trace of k's
        trace_fneq = (cx * cx * f_neq).sum(0) + (cy * cy * f_neq).sum(0) + (cz * cz * f_neq).sum(0)
        trace_k = (cx * cx * k).sum(0) + (cy * cy * k).sum(0) + (cz * cz * k).sum(0)
        torch.testing.assert_close(trace_k, trace_fneq, atol=1e-6, rtol=1e-6)

    def test_higher_order_has_zero_second_moments_d3q19(self):
        """h should have zero second-order moments (all captured by s + k)."""
        from tensorlbm.entropic_kbc import kbc_decompose_d3q19

        f = _state_19()
        rho, ux, uy, uz = macroscopic3d(f)
        feq = equilibrium3d(rho, ux, uy, uz)
        f_neq = f - feq
        s, k, h = kbc_decompose_d3q19(f_neq)
        c = C19.to(f.device).float()
        cx, cy, cz = c[:, 0].view(19, 1, 1, 1), c[:, 1].view(19, 1, 1, 1), c[:, 2].view(19, 1, 1, 1)
        for ca, cb in [(cx, cx), (cy, cy), (cz, cz), (cx, cy), (cx, cz), (cy, cz)]:
            moment_h = (ca * cb * h).sum(0)
            torch.testing.assert_close(moment_h, torch.zeros_like(moment_h), atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# 3. Gamma solver (entropy minimisation)
# ---------------------------------------------------------------------------

class TestGammaSolver:
    """Per-cell nonlinear gamma solve via bisection."""

    def test_gamma_satisfies_entropy_condition_d3q19(self):
        """dH/dγ ≈ 0 at the solved gamma."""
        from tensorlbm.entropic_kbc import (
            kbc_decompose_d3q19,
            solve_gamma_entropy,
        )

        f = _state_19()
        rho, ux, uy, uz = macroscopic3d(f)
        feq = equilibrium3d(rho, ux, uy, uz)
        f_neq = f - feq
        s, k, h = kbc_decompose_d3q19(f_neq)
        w = W19.to(f.device).view(19, 1, 1, 1)
        gamma_init = torch.full((2, 3, 4), 1.0 - 1.0 / 0.8, device=f.device, dtype=f.dtype)
        gamma = solve_gamma_entropy(feq, s, h, w, gamma_init)
        # Check dH/dgamma ≈ 0
        f_post = feq + gamma.unsqueeze(0) * s + h
        f_safe = torch.clamp(f_post, min=1e-30)
        dH = (s * (1.0 + torch.log(f_safe / w))).sum(0)
        torch.testing.assert_close(dH, torch.zeros_like(dH), atol=1e-4, rtol=1e-4)

    def test_gamma_satisfies_entropy_condition_d3q27(self):
        from tensorlbm.entropic_kbc import (
            kbc_decompose_d3q27,
            solve_gamma_entropy,
        )

        f = _state_27()
        rho, ux, uy, uz = macroscopic27(f)
        feq = equilibrium27(rho, ux, uy, uz)
        f_neq = f - feq
        s, k, h = kbc_decompose_d3q27(f_neq)
        w = W27.to(f.device).view(27, 1, 1, 1)
        gamma_init = torch.full((2, 3, 4), 1.0 - 1.0 / 0.8, device=f.device, dtype=f.dtype)
        gamma = solve_gamma_entropy(feq, s, h, w, gamma_init)
        f_post = feq + gamma.unsqueeze(0) * s + h
        f_safe = torch.clamp(f_post, min=1e-30)
        dH = (s * (1.0 + torch.log(f_safe / w))).sum(0)
        torch.testing.assert_close(dH, torch.zeros_like(dH), atol=1e-4, rtol=1e-4)

    def test_gamma_is_within_admissibility_domain_d3q19(self):
        """Post-collision distribution must be positive (admissibility)."""
        from tensorlbm.entropic_kbc import (
            kbc_decompose_d3q19,
            solve_gamma_entropy,
        )

        f = _state_19()
        rho, ux, uy, uz = macroscopic3d(f)
        feq = equilibrium3d(rho, ux, uy, uz)
        f_neq = f - feq
        s, k, h = kbc_decompose_d3q19(f_neq)
        w = W19.to(f.device).view(19, 1, 1, 1)
        gamma_init = torch.full((2, 3, 4), 1.0 - 1.0 / 0.8, device=f.device, dtype=f.dtype)
        gamma = solve_gamma_entropy(feq, s, h, w, gamma_init)
        f_post = feq + gamma.unsqueeze(0) * s + h
        assert (f_post > 0).all(), "Post-collision distribution must be positive"

    def test_gamma_is_within_admissibility_domain_d3q27(self):
        from tensorlbm.entropic_kbc import (
            kbc_decompose_d3q27,
            solve_gamma_entropy,
        )

        f = _state_27()
        rho, ux, uy, uz = macroscopic27(f)
        feq = equilibrium27(rho, ux, uy, uz)
        f_neq = f - feq
        s, k, h = kbc_decompose_d3q27(f_neq)
        w = W27.to(f.device).view(27, 1, 1, 1)
        gamma_init = torch.full((2, 3, 4), 1.0 - 1.0 / 0.8, device=f.device, dtype=f.dtype)
        gamma = solve_gamma_entropy(feq, s, h, w, gamma_init)
        f_post = feq + gamma.unsqueeze(0) * s + h
        assert (f_post > 0).all()

    def test_gamma_is_finite_and_within_admissibility(self):
        """For moderate non-equilibrium, gamma must be finite and admissible."""
        from tensorlbm.entropic_kbc import (
            kbc_decompose_d3q19,
            solve_gamma_entropy,
        )

        tau = 0.8
        f = _state_19(seed=99)
        rho, ux, uy, uz = macroscopic3d(f)
        feq = equilibrium3d(rho, ux, uy, uz)
        f_neq = f - feq
        s, k, h = kbc_decompose_d3q19(f_neq)
        w = W19.to(feq.device).view(19, 1, 1, 1)
        gamma_init = torch.full((2, 3, 4), 1.0 - 1.0 / tau, device=feq.device, dtype=feq.dtype)
        gamma = solve_gamma_entropy(feq, s, h, w, gamma_init)
        assert torch.isfinite(gamma).all(), "gamma must be finite"
        # Post-collision must be positive (admissibility)
        f_post = feq + gamma.unsqueeze(0) * s + h
        assert (f_post > 0).all(), "post-collision must be positive"


# ---------------------------------------------------------------------------
# 4. collide_kbc_d3q19 / collide_kbc_d3q27
# ---------------------------------------------------------------------------

class TestCollideKBC:
    """Full entropic KBC collision operators."""

    @pytest.mark.parametrize("tau", [0.55, 0.8, 1.0, 1.5])
    def test_kbc_d3q19_mass_momentum_conservation(self, tau):
        from tensorlbm.entropic_kbc import collide_kbc_d3q19

        f = _state_19()
        rho, ux, uy, uz = macroscopic3d(f)
        f_post = collide_kbc_d3q19(f, tau)
        rho_post, ux_post, uy_post, uz_post = macroscopic3d(f_post)
        torch.testing.assert_close(rho_post, rho, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(ux_post, ux, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(uy_post, uy, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(uz_post, uz, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("tau", [0.55, 0.8, 1.0, 1.5])
    def test_kbc_d3q27_mass_momentum_conservation(self, tau):
        from tensorlbm.entropic_kbc import collide_kbc_d3q27

        f = _state_27()
        rho, ux, uy, uz = macroscopic27(f)
        f_post = collide_kbc_d3q27(f, tau)
        rho_post, ux_post, uy_post, uz_post = macroscopic27(f_post)
        torch.testing.assert_close(rho_post, rho, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(ux_post, ux, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(uy_post, uy, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(uz_post, uz, atol=1e-6, rtol=1e-6)

    def test_kbc_d3q19_equilibrium_fixed_point(self):
        """f_eq is a fixed point of the KBC collision."""
        from tensorlbm.entropic_kbc import collide_kbc_d3q19

        rho = torch.ones(2, 3, 4)
        zero = torch.zeros_like(rho)
        feq = equilibrium3d(rho, zero, zero, zero)
        f_post = collide_kbc_d3q19(feq, tau=0.8)
        torch.testing.assert_close(f_post, feq, atol=1e-6, rtol=1e-6)

    def test_kbc_d3q27_equilibrium_fixed_point(self):
        from tensorlbm.entropic_kbc import collide_kbc_d3q27

        rho = torch.ones(2, 3, 4)
        zero = torch.zeros_like(rho)
        feq = equilibrium27(rho, zero, zero, zero)
        f_post = collide_kbc_d3q27(feq, tau=0.8)
        torch.testing.assert_close(f_post, feq, atol=1e-6, rtol=1e-6)

    def test_kbc_d3q19_positivity(self):
        """Post-collision distribution must be positive."""
        from tensorlbm.entropic_kbc import collide_kbc_d3q19

        f = _state_19()
        f_post = collide_kbc_d3q19(f, tau=0.55)
        assert (f_post > 0).all()

    def test_kbc_d3q27_positivity(self):
        from tensorlbm.entropic_kbc import collide_kbc_d3q27

        f = _state_27()
        f_post = collide_kbc_d3q27(f, tau=0.55)
        assert (f_post > 0).all()

    def test_kbc_d3q19_h_theorem(self):
        """H(f*) ≤ H(f): entropy must not increase."""
        from tensorlbm.entropic_kbc import collide_kbc_d3q19, discrete_entropy

        f = _state_19()
        w = W19.to(f.device).view(19, 1, 1, 1)
        H_before = discrete_entropy(f, w)
        f_post = collide_kbc_d3q19(f, tau=0.8)
        H_after = discrete_entropy(f_post, w)
        assert (H_after <= H_before + 1e-6).all(), "H-theorem violated: H increased"

    def test_kbc_d3q27_h_theorem(self):
        from tensorlbm.entropic_kbc import collide_kbc_d3q27, discrete_entropy

        f = _state_27()
        w = W27.to(f.device).view(27, 1, 1, 1)
        H_before = discrete_entropy(f, w)
        f_post = collide_kbc_d3q27(f, tau=0.8)
        H_after = discrete_entropy(f_post, w)
        assert (H_after <= H_before + 1e-6).all(), "H-theorem violated: H increased"

    def test_kbc_d3q19_reduces_non_equilibrium(self):
        """Post-collision should be closer to equilibrium than pre-collision."""
        from tensorlbm.entropic_kbc import collide_kbc_d3q19

        f = _state_19()
        rho, ux, uy, uz = macroscopic3d(f)
        feq = equilibrium3d(rho, ux, uy, uz)
        neq_before = (f - feq).abs().max().item()
        f_post = collide_kbc_d3q19(f, tau=0.8)
        rho_p, ux_p, uy_p, uz_p = macroscopic3d(f_post)
        feq_p = equilibrium3d(rho_p, ux_p, uy_p, uz_p)
        neq_after = (f_post - feq_p).abs().max().item()
        assert neq_after < neq_before, "Non-equilibrium should decrease"


# ---------------------------------------------------------------------------
# 5. Contract registration
# ---------------------------------------------------------------------------

class TestKBCContractRegistration:
    """KBC must be registered as AVAILABLE in the advanced collision contract."""

    def test_d3q19_kbc_is_available(self):
        from tensorlbm.advanced_collision_contract import collision_capability_matrix

        matrix = collision_capability_matrix()
        cap = matrix["D3Q19"]["KBC"]
        assert cap.available, "D3Q19 KBC must be AVAILABLE"
        assert cap.status == "AVAILABLE"
        assert cap.entrypoint is not None
        assert "kbc" in cap.entrypoint.lower()

    def test_d3q27_kbc_is_available(self):
        from tensorlbm.advanced_collision_contract import collision_capability_matrix

        matrix = collision_capability_matrix()
        cap = matrix["D3Q27"]["KBC"]
        assert cap.available, "D3Q27 KBC must be AVAILABLE"
        assert cap.status == "AVAILABLE"
        assert cap.entrypoint is not None
        assert "kbc" in cap.entrypoint.lower()

    def test_collide_advanced_3d_dispatches_kbc_d3q19(self):
        from tensorlbm.advanced_collision_contract import collide_advanced_3d

        rho = torch.ones(1, 2, 2)
        zero = torch.zeros_like(rho)
        f = equilibrium3d(rho, zero, zero, zero)
        out = collide_advanced_3d("D3Q19", "KBC", f, tau=0.8)
        assert out.shape == (19, 1, 2, 2)
        torch.testing.assert_close(out, f, atol=1e-5, rtol=1e-5)

    def test_collide_advanced_3d_dispatches_kbc_d3q27(self):
        from tensorlbm.advanced_collision_contract import collide_advanced_3d

        rho = torch.ones(1, 2, 2)
        zero = torch.zeros_like(rho)
        f = equilibrium27(rho, zero, zero, zero)
        out = collide_advanced_3d("D3Q27", "KBC", f, tau=0.8)
        assert out.shape == (27, 1, 2, 2)
        torch.testing.assert_close(out, f, atol=1e-5, rtol=1e-5)

    def test_collide_advanced_3d_kbc_alias_entropic_kbc(self):
        from tensorlbm.advanced_collision_contract import collide_advanced_3d

        rho = torch.ones(1, 2, 2)
        zero = torch.zeros_like(rho)
        f = equilibrium3d(rho, zero, zero, zero)
        out = collide_advanced_3d("D3Q19", "entropic_kbc", f, tau=0.8)
        assert out.shape == (19, 1, 2, 2)


# ---------------------------------------------------------------------------
# 6. Stability: short cylinder/sphere flow run
# ---------------------------------------------------------------------------

class TestKBCStability:
    """KBC must remain stable for a short flow simulation."""

    def test_kbc_d3q19_short_flow_stable(self):
        """Run a few steps of a simple periodic shear flow and check stability."""
        from tensorlbm.entropic_kbc import collide_kbc_d3q19
        from tensorlbm.solver3d import stream3d

        torch.manual_seed(123)
        nz, ny, nx = 8, 8, 8
        rho = torch.ones(nz, ny, nx)
        ux = 0.05 * torch.sin(torch.linspace(0, math.pi, nx)).view(1, 1, -1).expand(nz, ny, nx)
        uy = torch.zeros_like(rho)
        uz = torch.zeros_like(rho)
        f = equilibrium3d(rho, ux, uy, uz)
        f = f + 1e-4 * torch.randn_like(f)
        tau = 0.6  # low viscosity → stress test
        for _ in range(20):
            f = collide_kbc_d3q19(f, tau)
            f = stream3d(f)
        assert torch.isfinite(f).all(), "Flow diverged (NaN/Inf)"
        assert (f > 0).all(), "Populations became negative"

    def test_kbc_d3q27_short_flow_stable(self):
        from tensorlbm.entropic_kbc import collide_kbc_d3q27
        from tensorlbm.d3q27 import stream27

        torch.manual_seed(123)
        nz, ny, nx = 8, 8, 8
        rho = torch.ones(nz, ny, nx)
        ux = 0.05 * torch.sin(torch.linspace(0, math.pi, nx)).view(1, 1, -1).expand(nz, ny, nx)
        uy = torch.zeros_like(rho)
        uz = torch.zeros_like(rho)
        f = equilibrium27(rho, ux, uy, uz)
        f = f + 1e-4 * torch.randn_like(f)
        tau = 0.6
        for _ in range(20):
            f = collide_kbc_d3q27(f, tau)
            f = stream27(f)
        assert torch.isfinite(f).all()
        assert (f > 0).all()
