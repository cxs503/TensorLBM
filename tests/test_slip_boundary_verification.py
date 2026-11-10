"""Machine-criteria tests for TensorLBM slip boundary (p106w2-n02).

Validates:
1. Plane Poiseuille b-inversion ≤ 2%
2. Pipe Poiseuille b-inversion ≤ 5%
3. b(sigma) scaling law R² ≥ 0.999
4. σ=0 → no-slip (deviation ≤ 0.5%)
5. σ→1 → free-slip divergence (no crash)
6. Drag reduction ΔC_f(b⁺) monotone
7. Net benefit ledger computable
8. Maxwell audit: operator vs flow agreement < 5%
"""

import math
import numpy as np
import pytest

from tensorlbm.boundary import partial_slip_wall as ps
from tensorlbm.boundary import slip_wall_calibration as swc


# ── 1. Plane Poiseuille b-inversion ──────────────────────────────────
class TestPlanePoiseuilleInversion:
    """Analytic slip Poiseuille: u = g/(2ν)(H²-y²) + gHb/ν."""

    def test_profile_matches_analytic_within_2pct(self):
        tau, g, ny = 0.8, 1e-5, 33
        nu = ps.nu_from_tau(tau)
        H = (ny - 2) / 2.0
        res = ps.channel_poiseuille(sigma=0.3, ny=ny, tau=tau, g=g, steps=30000, tol=1e-12)
        inv = ps.invert_slip_length(res)
        u_ana = ps.analytic_plane_profile(res.coord, H, nu, g, b=inv["b"])
        rel_err = np.max(np.abs(res.u - u_ana)) / np.max(np.abs(res.u))
        assert rel_err < 0.02, f"Profile rel error {rel_err:.2e} exceeds 2%"

    def test_center_velocity_matches_analytic(self):
        tau, g, ny = 0.8, 1e-5, 33
        nu = ps.nu_from_tau(tau)
        H = (ny - 2) / 2.0
        res = ps.channel_poiseuille(sigma=0.3, ny=ny, tau=tau, g=g, steps=30000, tol=1e-12)
        inv = ps.invert_slip_length(res)
        u_center_ana = g * H**2 / (2*nu) + g * H * inv["b"] / nu
        u_center_sim = float(res.u[len(res.coord)//2])
        rel_err = abs(u_center_sim - u_center_ana) / u_center_ana
        assert rel_err < 0.02, f"Center velocity error {rel_err:.2e} exceeds 2%"

    def test_roundtrip_inversion(self):
        """Inverting b from the analytic profile with b should recover b."""
        tau, g, ny = 0.8, 1e-5, 33
        nu = ps.nu_from_tau(tau)
        H = (ny - 2) / 2.0
        b_target = 0.15
        y = np.arange(1, ny - 1, dtype=float)
        u_ana = ps.analytic_plane_profile(y, H, nu, g, b=b_target)
        inv = ps.slip_length_from_quadratic_fit(y, u_ana, H, nu, g, geometry="plane")
        assert abs(inv["b"] - b_target) / b_target < 0.01


# ── 2. Pipe Poiseuille b-inversion ───────────────────────────────────
class TestPipePoiseuilleInversion:
    """Analytic slip Hagen-Poiseuille: u = g/(4ν)(R²-r²) + gRb/(2ν)."""

    def test_core_profile_matches_analytic_within_5pct(self):
        tau, g, nr = 0.8, 1e-5, 33
        nu = ps.nu_from_tau(tau)
        R = nr - 1.5
        res = ps.pipe_poiseuille(sigma=0.3, nr=nr, tau=tau, g=g, steps=30000, tol=1e-10)
        inv = ps.invert_slip_length(res)
        u_ana = ps.analytic_pipe_profile(res.coord, R, nu, g, b=inv["b"])
        core = np.abs(res.coord) <= 0.6 * R
        rel_err = np.max(np.abs(res.u[core] - u_ana[core])) / np.max(np.abs(res.u[core]))
        assert rel_err < 0.05, f"Pipe core profile error {rel_err:.2e} exceeds 5%"

    def test_noslip_u_max_within_2pct(self):
        tau, g, nr = 0.8, 1e-5, 33
        nu = ps.nu_from_tau(tau)
        R = nr - 1.5
        res = ps.pipe_poiseuille(sigma=0.0, nr=nr, tau=tau, g=g, steps=30000, tol=1e-10)
        u_max_ana = g * R**2 / (4*nu)
        rel_err = abs(res.u.max() - u_max_ana) / u_max_ana
        assert rel_err < 0.02, f"Pipe no-slip u_max error {rel_err:.2e} exceeds 2%"

    def test_curvature_ratio_near_unity(self):
        res = ps.pipe_poiseuille(sigma=0.3, nr=33, tau=0.8, g=1e-5, steps=30000, tol=1e-10)
        inv = ps.invert_slip_length(res)
        assert abs(inv["curvature_ratio"] - 1.0) < 0.01


# ── 3. b(sigma) scaling law ──────────────────────────────────────────
class TestScalingLaw:
    def test_r2_above_0999(self):
        sigmas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        b_vals = []
        for s in sigmas:
            res = ps.channel_poiseuille(sigma=s, ny=33, tau=0.8, g=1e-5, steps=30000, tol=1e-12)
            inv = ps.invert_slip_length(res)
            b_vals.append(inv["b"])
        law = ps.fit_slip_law(sigmas, b_vals)
        assert law.r2 >= 0.999, f"R²={law.r2:.6f} < 0.999"

    def test_monotone(self):
        sigmas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        b_vals = []
        for s in sigmas:
            res = ps.channel_poiseuille(sigma=s, ny=33, tau=0.8, g=1e-5, steps=30000, tol=1e-12)
            inv = ps.invert_slip_length(res)
            b_vals.append(inv["b"])
        law = ps.fit_slip_law(sigmas, b_vals)
        assert law.monotone, "b(sigma) is not monotone"

    def test_slope_near_03(self):
        """The classical partial-slip slope is ~0.3 lattice units."""
        sigmas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        b_vals = []
        for s in sigmas:
            res = ps.channel_poiseuille(sigma=s, ny=33, tau=0.8, g=1e-5, steps=30000, tol=1e-12)
            inv = ps.invert_slip_length(res)
            b_vals.append(inv["b"])
        law = ps.fit_slip_law(sigmas, b_vals)
        assert abs(law.a - 0.3) < 0.02, f"Slope a={law.a:.4f} deviates from 0.3"


# ── 4. Negative control: σ=0 → no-slip ───────────────────────────────
class TestNoSlipControl:
    def test_sigma0_recovers_noslip_within_05pct(self):
        tau, g, ny = 0.8, 1e-5, 33
        nu = ps.nu_from_tau(tau)
        H = (ny - 2) / 2.0
        res = ps.channel_poiseuille(sigma=0.0, ny=ny, tau=tau, g=g, steps=30000, tol=1e-12)
        u_max_ana = g * H**2 / (2*nu)
        rel_err = abs(res.u.max() - u_max_ana) / u_max_ana
        assert rel_err < 0.005, f"σ=0 u_max error {rel_err:.2e} exceeds 0.5%"

    def test_sigma0_b_near_zero(self):
        res = ps.channel_poiseuille(sigma=0.0, ny=33, tau=0.8, g=1e-5, steps=30000, tol=1e-12)
        inv = ps.invert_slip_length(res)
        assert abs(inv["b"]) < 0.01, f"σ=0 b={inv['b']:.6f} should be ~0"


# ── 5. Free-slip limit ───────────────────────────────────────────────
class TestFreeSlipLimit:
    def test_sigma_approaching_1_grows_monotonely(self):
        u_max_vals = []
        for s in [0.9, 0.95, 0.98, 0.99]:
            res = ps.channel_poiseuille(sigma=s, ny=33, tau=0.8, g=1e-5, steps=30000, tol=1e-10)
            u_max_vals.append(float(res.u.max()))
        for i in range(len(u_max_vals) - 1):
            assert u_max_vals[i] < u_max_vals[i+1], "u_max should grow as σ→1"

    def test_sigma_1_does_not_crash(self):
        """σ=1 (pure specular) should not crash; velocity becomes uniform."""
        res = ps.channel_poiseuille(sigma=1.0, ny=33, tau=0.8, g=1e-5, steps=5000, tol=1e-6)
        assert res.u.max() > 0, "σ=1 should produce non-zero velocity"
        assert np.std(res.u) / max(res.u.max(), 1e-30) < 0.01, "σ=1 should give nearly uniform u"


# ── 6. Drag reduction ────────────────────────────────────────────────
class TestDragReduction:
    def test_monotone(self):
        dr = ps.drag_reduction_curve([0.0, 5.0, 10.0, 20.0], re_L=1e8)
        assert dr["monotone"], "ΔC_f(b⁺) must be monotone"

    def test_zero_at_zero(self):
        dr = ps.drag_reduction_curve([0.0, 5.0, 10.0, 20.0], re_L=1e8)
        assert dr["rows"][0]["dCf_local_rel"] == 0.0


# ── 7. Net benefit ledger ────────────────────────────────────────────
class TestNetBenefit:
    def test_ledger_computable(self):
        ledger = ps.net_benefit_ledger([0.0, 5.0, 10.0, 20.0])
        assert ledger["P_friction_W"] > 0
        assert "breakeven" in ledger
        assert "cost_model_status" in ledger
        assert "WITHHELD" in ledger["cost_model_status"]


# ── 8. Maxwell audit ─────────────────────────────────────────────────
class TestMaxwellAudit:
    def test_rule_vs_fit_agreement(self):
        audit = swc.maxwell_audit(sigma=0.3, ny=33, tau=0.8, g=1e-5, steps=30000)
        assert audit["b_rule_vs_fit_rel_err"] < 0.05, (
            f"Maxwell audit: rule vs fit rel err = {audit['b_rule_vs_fit_rel_err']:.4f} > 5%"
        )

    def test_gradient_matches_analytic(self):
        audit = swc.maxwell_audit(sigma=0.3, ny=33, tau=0.8, g=1e-5, steps=30000)
        assert audit["gradient_rel_err"] < 1e-5, (
            f"Gradient rel err {audit['gradient_rel_err']:.2e} should be ~0"
        )
