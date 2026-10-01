"""TDD tests for KBC collision diagnostics: gamma solve, decomposition, positivity.

These tests probe the root cause of abnormally high Cd in sphere flow when
using the entropic KBC collision operator.  They check:

1. KBC decomposition identity: s + k + h == f_neq
2. Second-order projection recovers the stress tensor
3. Gamma solve stays within the admissibility (positivity) domain
4. Post-collision populations remain non-negative
5. H-theorem: H(f*) <= H(f)
6. Gamma is finite and bounded
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import C as C19
from tensorlbm.d3q19 import W as W19
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.entropic_kbc import (
    _kbc_decompose,
    _lattice_constants,
    collide_kbc_d3q19,
    discrete_entropy,
    kbc_decompose_d3q19,
    solve_gamma_entropy,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_grid():
    """A 4×4×4 grid with a mild non-equilibrium perturbation."""
    nz, ny, nx = 4, 4, 4
    dev = torch.device("cpu")
    rho = torch.ones(nz, ny, nx, device=dev)
    ux = torch.full((nz, ny, nx), 0.05, device=dev)
    uy = torch.zeros(nz, ny, nx, device=dev)
    uz = torch.zeros(nz, ny, nx, device=dev)
    feq = equilibrium3d(rho, ux, uy, uz, device=dev)
    # Add a small non-equilibrium perturbation
    torch.manual_seed(42)
    delta = 0.001 * torch.randn_like(feq)
    # Ensure mass and momentum are conserved (zero sum over Q for each cell)
    delta -= delta.mean(dim=0, keepdim=True)
    f = feq + delta
    return f, feq, rho, ux, uy, uz


@pytest.fixture
def strong_neq_grid():
    """A grid with stronger non-equilibrium (simulates near-obstacle cells)."""
    nz, ny, nx = 4, 4, 4
    dev = torch.device("cpu")
    rho = torch.ones(nz, ny, nx, device=dev)
    ux = torch.full((nz, ny, nx), 0.1, device=dev)
    uy = torch.zeros(nz, ny, nx, device=dev)
    uz = torch.zeros(nz, ny, nx, device=dev)
    feq = equilibrium3d(rho, ux, uy, uz, device=dev)
    torch.manual_seed(123)
    delta = 0.01 * torch.randn_like(feq)
    delta -= delta.mean(dim=0, keepdim=True)
    f = feq + delta
    return f, feq


# ---------------------------------------------------------------------------
# 1. Decomposition identity
# ---------------------------------------------------------------------------


class TestKBCDecomposition:
    """Verify s + k + h == f_neq and projection correctness."""

    def test_decomposition_identity(self, small_grid):
        f, feq, *_ = small_grid
        f_neq = f - feq
        s, k, h = kbc_decompose_d3q19(f_neq)
        reconstructed = s + k + h
        torch.testing.assert_close(reconstructed, f_neq, rtol=1e-5, atol=1e-6)

    def test_decomposition_identity_strong(self, strong_neq_grid):
        f, feq = strong_neq_grid
        f_neq = f - feq
        s, k, h = kbc_decompose_d3q19(f_neq)
        reconstructed = s + k + h
        torch.testing.assert_close(reconstructed, f_neq, rtol=1e-4, atol=1e-5)

    def test_second_order_projection_recovers_stress(self, small_grid):
        """Σ_i c_α c_β f_neq^(2)_i == Π_αβ."""
        f, feq, *_ = small_grid
        f_neq = f - feq
        s, k, h = kbc_decompose_d3q19(f_neq)
        f_neq_2 = s + k  # second-order projection

        p = _lattice_constants(C19, W19, f_neq.device, f_neq.dtype)
        cx, cy, _cz = p["cx"], p["cy"], p["cz"]

        # Original stress
        pi_xx = (cx * cx * f_neq).sum(0)
        pi_xy = (cx * cy * f_neq).sum(0)

        # Projected stress
        pi_xx_proj = (cx * cx * f_neq_2).sum(0)
        pi_xy_proj = (cx * cy * f_neq_2).sum(0)

        torch.testing.assert_close(pi_xx_proj, pi_xx, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(pi_xy_proj, pi_xy, rtol=1e-5, atol=1e-6)

    def test_shear_is_traceless(self, small_grid):
        """The shear projection s should have zero trace: Σ_i |c_i|^2 s_i == 0."""
        f, feq, *_ = small_grid
        f_neq = f - feq
        s, k, h = kbc_decompose_d3q19(f_neq)
        p = _lattice_constants(C19, W19, f_neq.device, f_neq.dtype)
        c_sq = p["c_sq"]
        trace_s = (c_sq * s).sum(0)
        assert trace_s.abs().max().item() < 1e-5, (
            f"Shear projection s is not traceless: max |tr(s)| = {trace_s.abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# 2. Gamma solve: admissibility and convergence
# ---------------------------------------------------------------------------


class TestGammaSolve:
    """Verify gamma stays in admissibility domain and minimises H."""

    def test_gamma_is_finite(self, small_grid):
        f, feq, *_ = small_grid
        f_neq = f - feq
        s, k, h = kbc_decompose_d3q19(f_neq)
        w = W19.view(19, 1, 1, 1)
        gamma_init = torch.full(feq.shape[1:], 0.5, device=f.device, dtype=f.dtype)
        gamma = solve_gamma_entropy(feq, s, h, w, gamma_init)
        assert torch.isfinite(gamma).all(), "Gamma contains non-finite values"

    def test_post_collision_positive(self, small_grid):
        """f* = f_eq + γ·s + h should be non-negative."""
        f, feq, *_ = small_grid
        f_neq = f - feq
        s, k, h = kbc_decompose_d3q19(f_neq)
        w = W19.view(19, 1, 1, 1)
        gamma_init = torch.full(feq.shape[1:], 0.5, device=f.device, dtype=f.dtype)
        gamma = solve_gamma_entropy(feq, s, h, w, gamma_init)
        f_star = feq + gamma.unsqueeze(0) * s + h
        min_val = f_star.min().item()
        assert min_val >= -1e-10, f"Post-collision population is negative: min = {min_val}"

    def test_post_collision_positive_strong(self, strong_neq_grid):
        """f* should be non-negative even with strong non-equilibrium."""
        f, feq = strong_neq_grid
        f_neq = f - feq
        s, k, h = kbc_decompose_d3q19(f_neq)
        w = W19.view(19, 1, 1, 1)
        gamma_init = torch.full(feq.shape[1:], 0.5, device=f.device, dtype=f.dtype)
        gamma = solve_gamma_entropy(feq, s, h, w, gamma_init)
        f_star = feq + gamma.unsqueeze(0) * s + h
        min_val = f_star.min().item()
        assert min_val >= -1e-10, (
            f"Post-collision population is negative with strong neq: min = {min_val}"
        )

    def test_h_theorem_satisfied(self, small_grid):
        """H(f*) <= H(f)."""
        f, feq, *_ = small_grid
        f_neq = f - feq
        s, k, h = kbc_decompose_d3q19(f_neq)
        w = W19.view(19, 1, 1, 1)
        gamma_init = torch.full(feq.shape[1:], 0.5, device=f.device, dtype=f.dtype)
        gamma = solve_gamma_entropy(feq, s, h, w, gamma_init)
        f_star = feq + gamma.unsqueeze(0) * s + h

        H_before = discrete_entropy(f, w)
        H_after = discrete_entropy(f_star, w)
        violation = (H_after > H_before + 1e-10).sum().item()
        assert violation == 0, (
            f"H-theorem violated in {violation} cells: "
            f"max(H* - H) = {(H_after - H_before).max().item():.6e}"
        )

    def test_gamma_within_admissibility(self, small_grid):
        """Gamma should be within the natural admissibility domain, not expanded."""
        f, feq, *_ = small_grid
        f_neq = f - feq
        s, k, h = kbc_decompose_d3q19(f_neq)
        w = W19.view(19, 1, 1, 1)
        gamma_init = torch.full(feq.shape[1:], 0.5, device=f.device, dtype=f.dtype)

        # Compute natural admissibility bounds
        f_base = feq + h
        eps_s = 1e-30
        s_safe = torch.where(s.abs() > eps_s, s, torch.full_like(s, eps_s))
        ratio = -f_base / s_safe

        neg_inf = torch.full_like(gamma_init, -1e6)
        pos_inf = torch.full_like(gamma_init, 1e6)

        pos_mask = s > eps_s
        ratio_pos = torch.where(pos_mask, ratio, neg_inf.unsqueeze(0).expand_as(ratio))
        gamma_lower_natural = ratio_pos.amax(dim=0)

        neg_mask = s < -eps_s
        ratio_neg = torch.where(neg_mask, ratio, pos_inf.unsqueeze(0).expand_as(ratio))
        gamma_upper_natural = ratio_neg.amin(dim=0)

        gamma = solve_gamma_entropy(feq, s, h, w, gamma_init)

        # Check gamma is within natural bounds (allowing small tolerance)
        below = (gamma < gamma_lower_natural - 1e-8).sum().item()
        above = (gamma > gamma_upper_natural + 1e-8).sum().item()
        assert below == 0, f"Gamma below natural admissibility lower bound in {below} cells"
        assert above == 0, f"Gamma above natural admissibility upper bound in {above} cells"


# ---------------------------------------------------------------------------
# 3. Full collision operator
# ---------------------------------------------------------------------------


class TestCollideKBC:
    """Test the full collide_kbc_d3q19 operator."""

    def test_collision_preserves_mass_momentum(self, small_grid):
        """KBC must preserve the mass and momentum of the *input* f."""
        f, feq, rho, ux, uy, uz = small_grid
        tau = 0.8
        # Macros of the actual input f (may differ from original due to perturbation)
        rho_in, ux_in, uy_in, uz_in = macroscopic3d(f)
        f_star = collide_kbc_d3q19(f, tau=tau)

        rho_s, ux_s, uy_s, uz_s = macroscopic3d(f_star)
        torch.testing.assert_close(rho_s, rho_in, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(ux_s, ux_in, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(uy_s, uy_in, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(uz_s, uz_in, rtol=1e-4, atol=1e-5)

    def test_collision_positive(self, small_grid):
        f, feq, *_ = small_grid
        tau = 0.8
        f_star = collide_kbc_d3q19(f, tau=tau)
        min_val = f_star.min().item()
        assert min_val >= -1e-10, f"Post-collision negative: min = {min_val}"

    def test_collision_h_theorem(self, small_grid):
        f, feq, *_ = small_grid
        tau = 0.8
        f_star = collide_kbc_d3q19(f, tau=tau)
        w = W19.view(19, 1, 1, 1)
        H_before = discrete_entropy(f, w)
        H_after = discrete_entropy(f_star, w)
        violation = (H_after > H_before + 1e-8).sum().item()
        assert violation == 0, f"H-theorem violated in {violation} cells"

    def test_collision_positive_strong_neq(self, strong_neq_grid):
        """Even with strong non-equilibrium, populations should stay positive."""
        f, feq = strong_neq_grid
        tau = 0.6  # low tau → more aggressive relaxation
        f_star = collide_kbc_d3q19(f, tau=tau)
        min_val = f_star.min().item()
        assert min_val >= -1e-10, f"Post-collision negative with strong neq: min = {min_val}"


# ---------------------------------------------------------------------------
# 4. Root-cause tests: admissibility domain expansion bug
# ---------------------------------------------------------------------------


class TestAdmissibilityDomainBug:
    """Historical trigger state for the admissibility-domain expansion defect.

    The legacy collide path (solve_gamma_entropy) expands [lower, upper] to
    include gamma_init; when gamma_init is outside the natural admissibility
    domain the bisection searches regions with negative populations.  The
    Karlin-Boesch construction now used by collide_kbc_d3q19 (f* = feq +
    (1-1/tau)*s + beta*h with beta solved on the positivity-restricted
    [0, 1] interval) removes the post-collision symptoms on this same trigger
    state, which the kernel-facing tests below pin:

    1. No negative post-collision populations
    2. No H-theorem violations (H(f*) <= H(f))

    The two remaining tests verify the trigger state itself and the retained
    legacy solver (solve_gamma_entropy is still public API).
    """

    @pytest.fixture
    def strong_neq_case(self):
        """A case where gamma_init falls outside the natural admissibility domain."""
        dev = torch.device("cpu")
        dtype = torch.float64
        nz, ny, nx = 4, 4, 4
        rho = torch.ones(nz, ny, nx, dtype=dtype)
        ux = torch.full((nz, ny, nx), 0.1, dtype=dtype)
        uy = torch.zeros(nz, ny, nx, dtype=dtype)
        uz = torch.zeros(nz, ny, nx, dtype=dtype)
        feq = equilibrium3d(rho, ux, uy, uz, device=dev)
        torch.manual_seed(99)
        delta = 0.02 * torch.randn_like(feq)
        delta -= delta.mean(dim=0, keepdim=True)
        f = feq + delta
        return f, feq, 0.6  # tau

    def test_gamma_init_outside_natural_domain(self, strong_neq_case):
        """Verify that gamma_init can fall outside the natural admissibility domain."""
        f, feq, tau = strong_neq_case
        p = _lattice_constants(C19, W19, f.device, f.dtype)
        p["w"]
        f_neq = f - feq
        s, k, h = _kbc_decompose(f_neq, p)

        # Natural admissibility domain
        f_base = feq + h
        eps_s = 1e-30
        s_safe = torch.where(s.abs() > eps_s, s, torch.full_like(s, eps_s))
        ratio = -f_base / s_safe
        neg_inf = torch.full_like(feq[0], -1e6, dtype=f.dtype)
        pos_inf = torch.full_like(feq[0], 1e6, dtype=f.dtype)
        pos_mask = s > eps_s
        ratio_pos = torch.where(pos_mask, ratio, neg_inf.unsqueeze(0).expand_as(ratio))
        gamma_lower_nat = ratio_pos.amax(dim=0)
        neg_mask = s < -eps_s
        ratio_neg = torch.where(neg_mask, ratio, pos_inf.unsqueeze(0).expand_as(ratio))
        gamma_upper_nat = ratio_neg.amin(dim=0)

        gamma_init = torch.full(feq.shape[1:], 1.0 - 1.0 / tau, dtype=f.dtype)
        outside = ((gamma_init < gamma_lower_nat) | (gamma_init > gamma_upper_nat)).sum().item()
        assert outside > 0, "Test case should have gamma_init outside natural admissibility domain"

    def test_dH_sign_reversal_at_expanded_boundary(self, strong_neq_case):
        """dH/dgamma sign can reverse at expanded boundaries, breaking bisection."""
        f, feq, tau = strong_neq_case
        p = _lattice_constants(C19, W19, f.device, f.dtype)
        w = p["w"]
        f_neq = f - feq
        s, k, h = _kbc_decompose(f_neq, p)

        # Natural domain
        f_base = feq + h
        eps_s = 1e-30
        s_safe = torch.where(s.abs() > eps_s, s, torch.full_like(s, eps_s))
        ratio = -f_base / s_safe
        neg_inf = torch.full_like(feq[0], -1e6, dtype=f.dtype)
        pos_inf = torch.full_like(feq[0], 1e6, dtype=f.dtype)
        pos_mask = s > eps_s
        ratio_pos = torch.where(pos_mask, ratio, neg_inf.unsqueeze(0).expand_as(ratio))
        gamma_lower_nat = ratio_pos.amax(dim=0)
        neg_mask = s < -eps_s
        ratio_neg = torch.where(neg_mask, ratio, pos_inf.unsqueeze(0).expand_as(ratio))
        gamma_upper_nat = ratio_neg.amin(dim=0)

        gamma_init = torch.full(feq.shape[1:], 1.0 - 1.0 / tau, dtype=f.dtype)

        # Expanded domain (what the code does)
        torch.minimum(gamma_lower_nat, gamma_init)
        gamma_upper_exp = torch.maximum(gamma_upper_nat, gamma_init)

        # dH/dgamma at expanded upper
        f_upper = feq + gamma_upper_exp.unsqueeze(0) * s + h
        f_safe = torch.clamp(f_upper, min=1e-30)
        dH_upper = (s * (1.0 + torch.log(f_safe / w))).sum(dim=0)

        # Bisection assumes dH(upper) > 0; check for sign reversal
        wrong_sign = (dH_upper < 0).sum().item()
        assert wrong_sign > 0, (
            f"Expected dH/dgamma sign reversal at expanded upper boundary, "
            f"found {wrong_sign} cells with wrong sign"
        )

    def test_h_theorem_holds_with_strong_neq(self, strong_neq_case):
        """The discrete entropy does not increase on the strong-neq trigger state.

        The beta solve minimizes H along the positivity-restricted h-ray, so the
        historical H(f*) > H(f) symptom of the gamma-domain expansion is gone
        (on this seeded state H drops by at least 0.03 per cell).
        """
        f, feq, tau = strong_neq_case
        w = W19.view(19, 1, 1, 1).to(dtype=f.dtype)
        H_before = discrete_entropy(f, w)
        f_star = collide_kbc_d3q19(f, tau=tau)
        H_after = discrete_entropy(f_star, w)
        violations = (H_after > H_before + 1e-10).sum().item()
        assert violations == 0, (
            f"H-theorem violated in {violations} cells with strong non-equilibrium."
        )

    def test_no_negative_populations_with_strong_neq(self, strong_neq_case):
        """Post-collision populations stay non-negative on the trigger state.

        The legacy gamma expansion produced negative populations here; the beta
        construction intersects the search interval with the positivity domain.
        """
        f, feq, tau = strong_neq_case
        f_star = collide_kbc_d3q19(f, tau=tau)
        neg_count = (f_star < 0).sum().item()
        assert neg_count == 0, (
            f"{neg_count} negative post-collision populations on the strong-neq "
            f"trigger state (positivity domain not enforced)."
        )


# ---------------------------------------------------------------------------
# 5. h-mode relaxation test
# ---------------------------------------------------------------------------


class TestHModeRetention:
    """The higher-order mode h is relaxed, not carried over unchanged.

    The legacy post-collision formula f* = f_eq + gamma*s + h retained h in
    full (a documented design deviation from KBC).  The current construction
    f* = f_eq + (1-1/tau)*s + beta*h relaxes h along the entropy-optimal
    beta ray: on this seeded state the first collision reduces max|h| by more
    than an order of magnitude.  For weak residual h the optimal beta
    approaches 1, so the decay stalls after the first few collisions; the
    discriminating assertion is the first-collision drop.
    """

    def test_h_is_relaxed_by_first_collision(self):
        """max|h| shrinks markedly after one collision (legacy retained h in full)."""
        dev = torch.device("cpu")
        dtype = torch.float64
        nz, ny, nx = 4, 4, 4
        rho = torch.ones(nz, ny, nx, dtype=dtype)
        ux = torch.full((nz, ny, nx), 0.1, dtype=dtype)
        uy = torch.zeros(nz, ny, nx, dtype=dtype)
        uz = torch.zeros(nz, ny, nx, dtype=dtype)
        feq = equilibrium3d(rho, ux, uy, uz, device=dev)
        torch.manual_seed(42)
        delta = 0.01 * torch.randn_like(feq)
        delta -= delta.mean(dim=0, keepdim=True)
        f = feq + delta

        p = _lattice_constants(C19, W19, dev, dtype)
        f_neq = f - feq
        s0, k0, h0 = _kbc_decompose(f_neq, p)
        h_norm_0 = h0.abs().max().item()

        f = collide_kbc_d3q19(f, tau=0.8)

        rho_f, ux_f, uy_f, uz_f = macroscopic3d(f)
        feq_f = equilibrium3d(rho_f, ux_f, uy_f, uz_f, device=dev)
        f_neq_f = f - feq_f
        s_f, k_f, h_f = _kbc_decompose(f_neq_f, p)
        h_norm_f = h_f.abs().max().item()

        ratio = h_norm_f / max(h_norm_0, 1e-30)
        assert ratio < 0.5, (
            f"h_norm stayed at {h_norm_f:.6e} vs {h_norm_0:.6e} (ratio={ratio:.2f}). "
            f"h-mode is not being relaxed, which contradicts the "
            f"f*=feq+(1-1/tau)*s+beta*h construction."
        )


# ---------------------------------------------------------------------------
# 6. Sphere flow diagnostic test
# ---------------------------------------------------------------------------


class TestSphereFlowDiagnostic:
    """Run the KBC diagnostic on a tiny sphere flow and verify findings."""

    def test_kbc_diagnostic_runs_and_produces_report(self):
        """The diagnostic runner should produce a valid report."""
        from tensorlbm.kbc_diagnostic import (
            KBCDiagnosticConfig,
            run_kbc_diagnostic,
        )

        config = KBCDiagnosticConfig(nx=12, ny=12, nz=12, steps=10)
        report = run_kbc_diagnostic(config)
        assert len(report.kbc_steps) == 10
        assert len(report.bgk_steps) == 10
        assert report.reference_Cd > 0
        # KBC should show H-theorem violations (the bug)
        total_violations = sum(s["H_violation_count"] for s in report.kbc_steps)
        assert total_violations > 0, "Expected H-theorem violations in KBC sphere flow diagnostic"

    def test_kbc_h_mode_retained_in_sphere_flow(self):
        """h_norm should remain significant (not relaxed to zero) in sphere flow.

        The KBC post-collision f* = feq + γ·s + h fully retains h.  Even though
        streaming and boundaries can remove some h, it should NOT decay to near-zero
        like it would if it were being relaxed (as in BGK).
        """
        from tensorlbm.kbc_diagnostic import (
            KBCDiagnosticConfig,
            run_kbc_diagnostic,
        )

        config = KBCDiagnosticConfig(nx=16, ny=16, nz=16, steps=20)
        report = run_kbc_diagnostic(config)
        h_norms = [s["h_norm"] for s in report.kbc_steps]
        # After initial transient (step 3+), h_norm should remain significant
        # (not decay to near-zero).  Compare last step to the peak.
        peak = max(h_norms)
        last = h_norms[-1]
        assert last > 0.1 * peak, (
            f"h_norm decayed too much: peak={peak:.6e}, last={last:.6e}, "
            f"ratio={last / peak:.2f}. h-mode is being relaxed, which contradicts "
            f"the f*=feq+γ·s+h formula."
        )
