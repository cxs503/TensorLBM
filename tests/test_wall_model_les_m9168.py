"""Tests for wall-model LES verification (m9168 mission).

Tests the enhanced verification framework:
1. High-Re closure self-consistency (criterion a)
2. Flat plate friction coefficient (criterion c)
3. KVLCC2 resistance decomposition (criterion d friction limb)
4. Wall law implementations (Spalding, log, Reichardt)
5. Non-equilibrium wall model physics
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.wall_model_les.laws import (
    KAPPA_DEFAULT,
    B_DEFAULT,
    law_u_plus_scalar,
    spalding_u_plus,
    spalding_y_plus,
    solve_u_tau_scalar,
    reichardt_u_plus,
)
from tensorlbm.wall_model_les.models import (
    EquilibriumWallModel,
    NonEquilibriumWallModel,
)
from tensorlbm.wall_model_les.flat_plate import (
    run_flat_plate,
    ittc_1957_mean,
    ittc_1957_local,
    prandtl_schlichting_local,
)
from tensorlbm.wall_model_les.closure_channel import (
    solve_channel_closure,
    log_layer_fit,
)
from tensorlbm.wall_model_les.resistance import ittc_1957_cf


# ---------------------------------------------------------------------------
# Wall law unit tests
# ---------------------------------------------------------------------------
class TestWallLaws:
    """Test the wall law implementations."""

    def test_spalding_roundtrip(self):
        """Spalding y+(u+) and u+(y+) are inverses."""
        for u_plus in [1.0, 5.0, 10.0, 15.0, 20.0, 25.0]:
            y_plus = float(spalding_y_plus(u_plus))
            u_back = float(spalding_u_plus(np.array([y_plus]))[0])
            assert abs(u_back - u_plus) < 1e-6, f"Roundtrip failed at u+={u_plus}"

    def test_log_law_asymptote(self):
        """Spalding matches the log law for y+ >> 1."""
        for y_plus in [100.0, 500.0, 1000.0]:
            u_spalding = float(spalding_u_plus(np.array([y_plus]))[0])
            u_log = law_u_plus_scalar(y_plus, "log")
            assert abs(u_spalding - u_log) < 0.3, f"Log asymptote failed at y+={y_plus}"

    def test_viscous_sublayer(self):
        """u+ = y+ for y+ < 5 (viscous sublayer)."""
        for y_plus in [0.1, 1.0, 2.0, 4.0]:
            u_plus = float(spalding_u_plus(np.array([y_plus]))[0])
            assert abs(u_plus - y_plus) < 0.5, f"Sublayer failed at y+={y_plus}"

    def test_reichardt_smooth(self):
        """Reichardt law is smooth and monotone."""
        y = np.logspace(-1, 3, 100)
        u = np.array([float(reichardt_u_plus(yi)) for yi in y])
        assert np.all(np.diff(u) > 0), "Reichardt not monotone"

    def test_solve_u_tau_consistency(self):
        """solve_u_tau recovers u_tau from a known profile."""
        u_tau_true = 0.05
        y = 0.01
        nu = 1e-6
        y_plus = y * u_tau_true / nu
        u_plus = law_u_plus_scalar(y_plus, "spalding")
        u = u_tau_true * u_plus
        u_tau_recovered = solve_u_tau_scalar(u, y, nu, law="spalding")
        assert abs(u_tau_recovered - u_tau_true) / u_tau_true < 1e-4


# ---------------------------------------------------------------------------
# Wall model tests
# ---------------------------------------------------------------------------
class TestWallModels:
    """Test equilibrium and non-equilibrium wall models."""

    def test_equilibrium_recovers_log_law(self):
        """Equilibrium model with Spalding law recovers the log law."""
        model = EquilibriumWallModel(law="spalding")
        nu = 1e-5
        y_exchange = 0.001
        u_exchange = 1.0
        result = model.skin_friction(u_exchange, y_exchange, nu)
        u_tau = float(result.u_tau[0])
        y_plus = y_exchange * u_tau / nu
        u_plus = u_exchange / u_tau
        u_plus_expected = law_u_plus_scalar(y_plus, "spalding")
        assert abs(u_plus - u_plus_expected) / u_plus_expected < 1e-4

    def test_nonequilibrium_reduces_to_equilibrium_at_zero_source(self):
        """Non-equilibrium model with S=0 matches equilibrium."""
        eq = EquilibriumWallModel(law="spalding")
        neq = NonEquilibriumWallModel()
        nu = 1e-5
        y_exchange = 0.001
        u_exchange = 1.0
        r_eq = eq.skin_friction(u_exchange, y_exchange, nu)
        r_neq = neq.skin_friction(u_exchange, y_exchange, nu, source=0.0)
        u_tau_eq = float(r_eq.u_tau[0])
        u_tau_neq = float(r_neq.u_tau[0])
        assert abs(u_tau_neq - u_tau_eq) / u_tau_eq < 0.02

    def test_nonequilibrium_favourable_gradient(self):
        """Favourable pressure gradient (S<0) increases u_tau."""
        neq = NonEquilibriumWallModel()
        nu = 1e-5
        y_exchange = 0.001
        u_exchange = 1.0
        r_zero = neq.skin_friction(u_exchange, y_exchange, nu, source=0.0)
        r_fav = neq.skin_friction(u_exchange, y_exchange, nu, source=-100.0)
        u_tau_zero = float(r_zero.u_tau[0])
        u_tau_fav = float(r_fav.u_tau[0])
        assert u_tau_fav > u_tau_zero


# ---------------------------------------------------------------------------
# Flat plate tests (criterion c)
# ---------------------------------------------------------------------------
class TestFlatPlate:
    """Test the flat plate boundary layer model."""

    def test_ittc_1957_monotone(self):
        """ITTC-1957 line is monotone decreasing with Re."""
        re = np.logspace(6, 9, 100)
        cf = ittc_1957_mean(re)
        assert np.all(np.diff(cf) < 0)

    def test_flat_plate_cf_within_criterion(self):
        """Flat plate c_f error vs ITTC-1957 <= 5% (criterion c)."""
        result = run_flat_plate(
            re_theta_start=1e4,
            re_theta_end=5e4,
            n_steps=40,
            n_profile_points=201,
        )
        re_x_ref = (result.re_theta / 0.0365) ** 1.25
        cf_ittc = ittc_1957_local(re_x_ref)
        rel_error = np.abs(result.c_f - cf_ittc) / cf_ittc
        max_error = float(rel_error.max())
        assert max_error <= 0.05, f"Flat plate c_f error {max_error:.4f} > 5%"

    def test_flat_plate_shape_factor_physical(self):
        """Shape factor H is in the physical range 1.2-1.4 for ZPG BL."""
        result = run_flat_plate(n_steps=20, n_profile_points=101)
        assert np.all(result.shape_factor > 1.15)
        assert np.all(result.shape_factor < 1.45)


# ---------------------------------------------------------------------------
# High-Re closure self-consistency (criterion a)
# ---------------------------------------------------------------------------
class TestHighReClosure:
    """Test the high-Re closure self-consistency for criterion (a)."""

    def test_closure_self_consistency_at_high_re(self):
        """Mixing-length closure at Re_tau=10000 is self-consistent within 3%."""
        sol = solve_channel_closure(10000.0, closure="mixing_length", n_layers=4000)
        fit = log_layer_fit(sol.y_plus, sol.u_plus, band=(30, 300), h_plus=10000.0)
        assert fit["max_rel_error"] <= 0.03, (
            f"Closure self-consistency error {fit['max_rel_error']:.4f} > 3%"
        )

    def test_closure_fitted_kappa_close_to_declared(self):
        """Fitted kappa from closure is within 5% of declared 0.41."""
        sol = solve_channel_closure(10000.0, closure="mixing_length", n_layers=4000)
        fit = log_layer_fit(sol.y_plus, sol.u_plus, band=(30, 300), h_plus=10000.0)
        kappa_rel_error = abs(fit["kappa"] - 0.41) / 0.41
        assert kappa_rel_error <= 0.05, (
            f"Fitted kappa error {kappa_rel_error:.4f} > 5%"
        )

    def test_closure_stress_identity(self):
        """Closure satisfies the stress identity du+/dy+ + uv+ = 1 - y+/h+."""
        sol = solve_channel_closure(1000.0, closure="mixing_length", n_layers=2000)
        max_error = sol.stress_identity_max_abs_error
        assert max_error < 1e-6, f"Stress identity error {max_error:.2e} > 1e-6"


# ---------------------------------------------------------------------------
# KVLCC2 resistance (criterion d friction limb)
# ---------------------------------------------------------------------------
class TestKVLCC2Resistance:
    """Test the KVLCC2 resistance decomposition."""

    def test_ittc_1957_cf_at_re_13e7(self):
        """ITTC-1957 C_F at Re=1.3e7 is in the expected range."""
        cf = ittc_1957_cf(1.3e7)
        assert 0.001 < cf < 0.005, f"C_F={cf:.6f} outside expected range"

    def test_wall_model_cf_close_to_ittc(self):
        """Wall model mean C_F at Re=1.3e7 is within 5% of ITTC-1957."""
        result = run_flat_plate(
            re_theta_start=2e3,
            re_theta_end=2e5,
            n_steps=200,
        )
        re_x = np.asarray(result.re_x)
        re_target = 1.3e7
        if re_x.min() <= re_target <= re_x.max():
            re_theta = float(np.interp(re_target, re_x, result.re_theta))
            cf_mean = 2.0 * re_theta / re_target
            cf_ittc = ittc_1957_cf(re_target)
            rel_error = abs(cf_mean - cf_ittc) / cf_ittc
            assert rel_error <= 0.05, f"Wall model C_F error {rel_error:.4f} > 5%"


# ---------------------------------------------------------------------------
# Integration test: full acceptance record
# ---------------------------------------------------------------------------
class TestAcceptanceRecord:
    """Test that the acceptance record can be generated and is consistent."""

    def test_acceptance_m9168_generates(self, tmp_path):
        """The m9168 acceptance script generates a valid JSON record (needs artifacts)."""
        import subprocess
        # Use the existing artifacts directory which has apriori.json etc.
        artifacts_dir = REPO_ROOT / "artifacts" / "wall_model_les"
        if not (artifacts_dir / "apriori.json").exists():
            pytest.skip("artifacts/wall_model_les/apriori.json not found")
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "benchmarks" / "wall_model_les" / "acceptance_m9168.py"),
             "--out", str(artifacts_dir)],
            capture_output=True, text=True, timeout=120,
            env={**dict(__import__("os").environ), "PYTHONPATH": str(REPO_ROOT / "src")},
        )
        assert result.returncode == 0, f"Acceptance script failed: {result.stderr}"
        record = json.loads((artifacts_dir / "acceptance_m9168.json").read_text())
        assert "criteria" in record
        assert "summary" in record
        assert record["summary"]["n_pass"] >= 2
