"""Tests for the D2Q9 lid-driven cavity solver and its Ghia (1982) reference data.

Reference-data discipline (see ``docs/audit_m15543.md``):
``GHIA_RE5000`` was removed because the constant was fabricated; Re=5000 must
be injected through ``load_reference_json`` with a provenance-bearing
transcription.  The remaining tables are ``UNVERIFIED_OFFLINE`` and are only
used for shape/interpolation tests, never as an acceptance gate.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from tensorlbm.verification.lid_driven_cavity import (
    CavityConfig,
    CavitySolver,
    GHIA_RE100,
    GHIA_RE400,
    GHIA_RE1000,
    LITERATURE_ANCHORS,
    get_ghia_reference,
    load_reference_json,
)
from tensorlbm.verification.lid_driven_cavity.reference_data import (
    AVAILABLE_RE,
    GhiaReference,
    interpolate_ghia,
)
from tensorlbm.verification.lid_driven_cavity.verification import (
    compute_errors,
)


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------

class TestCavityConfig:
    """Test CavityConfig validation and derived properties."""

    def test_default_config(self):
        cfg = CavityConfig()
        assert cfg.n == 129
        assert cfg.re == 100.0
        assert cfg.u_lid == 0.1
        assert cfg.tau > 0.5

    def test_tau_computed_from_re(self):
        """tau should be computed from Re and u_lid."""
        cfg = CavityConfig(n=129, re=100, u_lid=0.1)
        L = cfg.n - 1
        nu = cfg.u_lid * L / cfg.re
        expected_tau = nu / (1.0/3.0) + 0.5
        assert abs(cfg.tau - expected_tau) < 1e-10

    def test_re5000_tau_is_marginal(self):
        """Re=5000 at n=129 gives tau ~ 0.5077, the RLBM use case."""
        cfg = CavityConfig(n=129, re=5000, u_lid=0.1)
        assert 0.5 < cfg.tau < 0.51

    def test_small_grid_rejected(self):
        with pytest.raises(ValueError, match="Grid too small"):
            CavityConfig(n=9)

    def test_high_velocity_rejected(self):
        with pytest.raises(ValueError, match="too high"):
            CavityConfig(u_lid=0.5)

    def test_omega_property(self):
        cfg = CavityConfig()
        assert abs(cfg.omega - 1.0/cfg.tau) < 1e-10


# ---------------------------------------------------------------------------
# Reference data tests
# ---------------------------------------------------------------------------

class TestReferenceData:
    """Test Ghia reference data integrity and provenance discipline."""

    def test_ghia_re100_data(self):
        assert GHIA_RE100.re == 100.0
        assert len(GHIA_RE100.y) == len(GHIA_RE100.u)
        assert len(GHIA_RE100.x) == len(GHIA_RE100.v)
        # u should be negative in lower half (primary vortex)
        assert np.all(GHIA_RE100.u[:7] < 0)
        # v should be negative in left half
        assert np.all(GHIA_RE100.v[1:7] < 0)  # v[0]=0 at left wall

    def test_ghia_re400_data(self):
        assert GHIA_RE400.re == 400.0
        assert len(GHIA_RE400.y) == len(GHIA_RE400.u)

    def test_ghia_re1000_data(self):
        assert GHIA_RE1000.re == 1000.0
        assert len(GHIA_RE1000.y) == len(GHIA_RE1000.u)

    def test_transcribed_tables_carry_unverified_provenance(self):
        """Tables without a paper/sha256 record must not claim verification."""
        for ref in (GHIA_RE100, GHIA_RE400, GHIA_RE1000):
            assert ref.provenance == "UNVERIFIED_OFFLINE"

    def test_re5000_not_available(self):
        """The fabricated Re=5000 table was removed; it must raise, not fall back."""
        assert 5000 not in AVAILABLE_RE
        with pytest.raises(KeyError, match="Re=5000"):
            get_ghia_reference(5000)

    def test_get_ghia_reference_rejects_untabulated_re(self):
        """Re=3200/7500/10000 not transcribed: must raise, not fall back."""
        with pytest.raises(KeyError):
            get_ghia_reference(3200)
        with pytest.raises(KeyError):
            get_ghia_reference(7500)
        with pytest.raises(KeyError):
            get_ghia_reference(10000)

    def test_get_ghia_reference(self):
        for re in (100, 400, 1000):
            assert get_ghia_reference(re).re == float(re)

    def test_interpolate_ghia(self):
        """Interpolation should return values at requested points."""
        y_test = np.array([0.0625, 0.5, 0.99219])
        u_interp = interpolate_ghia(GHIA_RE100, y_test, axis='u')
        assert len(u_interp) == 3
        assert u_interp[0] == pytest.approx(-0.03717, abs=0.001)  # Ghia y=0.0625
        assert u_interp[-1] == pytest.approx(0.08176, abs=0.001)  # Ghia y=0.99219

    def test_literature_anchors(self):
        """Point anchors must be physically ordered and negative."""
        assert LITERATURE_ANCHORS[100]["u_min"] == pytest.approx(-0.21090, abs=1e-6)
        assert LITERATURE_ANCHORS[1000]["u_min"] == pytest.approx(-0.38289, abs=1e-6)
        assert (abs(LITERATURE_ANCHORS[1000]["u_min"])
                > abs(LITERATURE_ANCHORS[100]["u_min"]))


class TestReferenceJsonLoader:
    """load_reference_json is the only provenance-bearing entry point."""

    @staticmethod
    def _write(tmp_path, data):
        path = tmp_path / "ref.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    @staticmethod
    def _payload(**over):
        data = {
            "re": 5000.0,
            "y": [0.0, 0.5, 1.0],
            "u": [0.0, -0.3, 1.0],
            "x": [0.0, 0.5, 1.0],
            "v": [0.0, 0.25, 0.0],
            "source": "Ghia et al. (1982) JCP 48:387-411, Table III/IV, p. 399",
        }
        data.update(over)
        return data

    def test_roundtrip(self, tmp_path):
        ref = load_reference_json(self._write(tmp_path, self._payload()))
        assert isinstance(ref, GhiaReference)
        assert ref.re == 5000.0
        assert ref.provenance == "BOUND"
        assert "Table III/IV" in ref.source
        assert ref.u[1] == pytest.approx(-0.3)

    def test_rejects_lattice_unit_table(self, tmp_path):
        """The fabrication signature: u stored in lattice units (lid row 0.1)."""
        bad = self._payload(u=[0.0, -0.10156, 0.1], v=[0.0, 0.10156, 0.0])
        with pytest.raises(ValueError, match="not normalised"):
            load_reference_json(self._write(tmp_path, bad))

    def test_rejects_non_monotone_coordinates(self, tmp_path):
        bad = self._payload(y=[0.0, 0.6, 0.5])
        with pytest.raises(ValueError, match="strictly increasing"):
            load_reference_json(self._write(tmp_path, bad))

    def test_rejects_missing_keys(self, tmp_path):
        bad = self._payload()
        del bad["v"]
        with pytest.raises(ValueError, match="missing required key"):
            load_reference_json(self._write(tmp_path, bad))

    def test_rejects_length_mismatch(self, tmp_path):
        bad = self._payload(u=[0.0, -0.3])
        with pytest.raises(ValueError, match="length mismatch"):
            load_reference_json(self._write(tmp_path, bad))


# ---------------------------------------------------------------------------
# Solver tests
# ---------------------------------------------------------------------------

class TestCavitySolver:
    """Test CavitySolver basic functionality."""

    def test_solver_initialization(self):
        cfg = CavityConfig(n=33, re=100, max_steps=100)
        solver = CavitySolver(cfg)
        assert solver.f.shape == (9, 33, 33)
        assert np.all(np.isfinite(solver.f.numpy()))

    def test_solver_step_changes_f(self):
        cfg = CavityConfig(n=33, re=100, max_steps=10)
        solver = CavitySolver(cfg)
        f_initial = solver.f.clone()
        solver.step()
        assert not torch.allclose(solver.f, f_initial)

    def test_solver_step_preserves_finite(self):
        cfg = CavityConfig(n=33, re=100, max_steps=100)
        solver = CavitySolver(cfg)
        for _ in range(50):
            solver.step()
        assert np.all(np.isfinite(solver.f.numpy()))

    @pytest.mark.parametrize("re", [100.0, 1000.0, 5000.0])
    def test_no_divergence_regression(self, re):
        """Regression: the 2026-11 boundary-condition rewrite diverged in <40 steps.

        All three Reynolds numbers must stay finite for 300 steps (that is 8x
        the step count at which the broken kernel produced NaN).
        """
        cfg = CavityConfig(n=33, re=re, max_steps=300)
        solver = CavitySolver(cfg)
        for _ in range(300):
            solver.step()
        assert bool(torch.all(torch.isfinite(solver.f))), f"diverged at Re={re}"
        _, ux, uy = solver._macroscopic(solver.f)
        assert float(ux.abs().max()) < 1.0
        assert float(uy.abs().max()) < 1.0

    def test_mass_conservation(self):
        """Total mass should be approximately conserved."""
        cfg = CavityConfig(n=33, re=100, max_steps=100)
        solver = CavitySolver(cfg)
        mass_initial = solver.f.sum().item()

        for _ in range(200):
            solver.step()

        mass_final = solver.f.sum().item()
        # Link bounce-back is exactly mass conserving; allow round-off only.
        rel_change = abs(mass_final - mass_initial) / mass_initial
        assert rel_change < 1e-10, f"mass drift {rel_change:.3e}"

    def test_wall_boundary_values(self):
        """No-slip walls + moving lid must show up in the wall-node fields."""
        cfg = CavityConfig(n=33, re=100, max_steps=50, u_lid=0.1)
        solver = CavitySolver(cfg)
        for _ in range(50):
            solver.step()
        y, u, x, v = solver.get_profiles()
        # bottom wall: no slip
        assert u[0] == pytest.approx(0.0, abs=1e-12)
        # top wall: lid velocity, normalised by u_lid
        assert u[-1] == pytest.approx(1.0, abs=1e-12)
        # normal velocity vanishes on both walls
        assert abs(v[0]) < 1e-12 and abs(v[-1]) < 1e-12

    def test_get_profiles(self):
        cfg = CavityConfig(n=33, re=100, max_steps=500)
        solver = CavitySolver(cfg)
        for _ in range(200):
            solver.step()

        y, u, x, v = solver.get_profiles()
        assert len(y) == 33
        assert len(u) == 33
        assert len(x) == 33
        assert len(v) == 33
        assert np.all(np.isfinite(u))
        assert np.all(np.isfinite(v))


# ---------------------------------------------------------------------------
# Physics tests (cheap, coarse grid)
# ---------------------------------------------------------------------------

class TestCavityPhysics:
    """Coarse-grid smoke tests of the solution structure."""

    @pytest.fixture(scope="class")
    def re100_n33_solver(self):
        cfg = CavityConfig(n=33, re=100, max_steps=2000, u_lid=0.1)
        solver = CavitySolver(cfg)
        for _ in range(2000):
            solver.step()
        return solver

    @pytest.fixture(scope="class")
    def re100_n33(self, re100_n33_solver):
        return re100_n33_solver.get_profiles()

    def test_momentum_is_injected_in_lid_direction(self, re100_n33_solver):
        """Flow must be driven in +x by the lid, and stay sub-lid speed."""
        solver = re100_n33_solver
        _, ux, uy = solver._macroscopic(solver.f)
        fluid = solver.fluid
        assert float(ux[fluid].mean()) > 0.0
        assert float(ux[fluid].abs().max()) <= solver.cfg.u_lid * 1.2
        assert float(uy[fluid].abs().max()) <= solver.cfg.u_lid * 1.2

    def test_primary_vortex_forms(self, re100_n33):
        y, u, x, v = re100_n33
        # Ghia (1982) Re=100: u_min = -0.21090 on the vertical centreline
        assert u.min() < -0.15
        assert u.min() > -0.35
        # u_min sits near mid height
        assert 0.35 < y[np.argmin(u)] < 0.65
        # counter-rotating lower-left region: v_max positive on the horizontal centreline
        assert v.max() > 0.05

    def test_centerline_value_matches_literature_anchor(self, re100_n33):
        """u(y=0.5) vs the literature anchor -0.20581 (10% tolerance, n=33)."""
        y, u, _, _ = re100_n33
        u_center = u[np.argmin(np.abs(y - 0.5))]
        ref = LITERATURE_ANCHORS[100]["u_center"]
        assert abs(u_center - ref) / abs(ref) < 0.10, (
            f"u(0.5)={u_center:.5f} vs anchor {ref}")


# ---------------------------------------------------------------------------
# Verification helper tests
# ---------------------------------------------------------------------------

class TestVerification:
    """Test verification against Ghia reference."""

    def test_compute_errors(self):
        """Error computation should work correctly."""
        y_sim = np.array([0.0, 0.5, 1.0])
        u_sim = np.array([0.0, -0.15, 0.0])
        x_sim = np.array([0.0, 0.5, 1.0])
        v_sim = np.array([0.0, -0.07, 0.0])

        errors = compute_errors(y_sim, u_sim, x_sim, v_sim, GHIA_RE100)
        assert 'l2_u' in errors
        assert 'l2_v' in errors
        assert 'max_u' in errors
        assert 'max_v' in errors
        assert errors['l2_u'] >= 0
        assert errors['l2_v'] >= 0

    def test_compute_errors_zero_on_reference(self):
        """A profile compared against itself must give zero error."""
        ref = GHIA_RE100
        errors = compute_errors(ref.y, ref.u, ref.x, ref.v, ref)
        assert errors['l2_u'] == pytest.approx(0.0, abs=1e-12)
        assert errors['l2_v'] == pytest.approx(0.0, abs=1e-12)
