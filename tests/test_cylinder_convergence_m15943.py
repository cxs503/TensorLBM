"""Convergence validation for Schäfer–Turek cylinder cases (m15943).

Tests that the Richardson-extrapolated Cd values for Re=20 and Re=100
are within 5% of the reference values, using artifacts produced by the
cluster convergence study (cj1208/cj1209).

Reference values:
  Re=20 (2D-1): Cd_ref = 5.57953523384 (spectral solution, Nabh 1998,
    via benchmarks/verified/cylinder/re20_st/result.json)
  Re=100 (2D-2): Cd_ref band = [3.22, 3.24] (Schäfer & Turek 1996, NNFM 48,
    Table 3, case 2D-2)

The convergence study runs D=20/40/80 (Re=20) and D=20/28/40 (Re=100)
with tau=0.8 (Re=20) and u_mean_lb=0.0833 (Re=100), then performs
Richardson extrapolation to estimate Cd at h→0.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.cylinder_st2d1 import REFERENCE as REF_RE20
from tensorlbm.validation.cylinder_dfg import REFERENCE_BANDS, band_mid

ARTIFACT_RE20 = REPO / "artifacts" / "cylinder_convergence_re20" / "convergence_summary.json"
ARTIFACT_RE100 = REPO / "artifacts" / "cylinder_convergence_re100" / "convergence_summary.json"


def _load_summary(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"Convergence artifact not yet produced: {path} (cluster job pending)")
    return json.loads(path.read_text())


def richardson_extrapolation(h_vals, q_vals):
    """Richardson extrapolation: q = q_inf + C*h^p, three-point."""
    h1, h2, h3 = h_vals
    q1, q2, q3 = q_vals
    r = h1 / h2
    if abs(q2 - q3) > 1e-12 and abs(q1 - q2) > 1e-12:
        ratio = (q1 - q2) / (q2 - q3)
        p_obs = math.log(ratio) / math.log(r) if ratio > 0 else float("nan")
    else:
        p_obs = float("nan")
    if not math.isnan(p_obs) and p_obs > 0:
        rp = r ** p_obs
        q_inf = q3 + (q3 - q2) / (rp - 1) if abs(rp - 1) > 1e-12 else float("nan")
    else:
        q_inf = float("nan")
    return q_inf, p_obs


class TestRe20Convergence:
    """Re=20 (2D-1) steady cylinder drag convergence."""

    @pytest.fixture(autouse=True)
    def load_data(self):
        self.summary = _load_summary(ARTIFACT_RE20)
        self.ref_cd = REF_RE20["cd"]

    def test_artifact_exists(self):
        assert ARTIFACT_RE20.exists(), f"Missing {ARTIFACT_RE20}"

    def test_three_grid_levels(self):
        rows = self.summary["re20"]
        assert len(rows) == 3, f"Expected 3 grid levels, got {len(rows)}"
        D_vals = [r["D_cells"] for r in rows]
        assert D_vals == [20, 40, 80], f"Expected D=[20,40,80], got {D_vals}"

    def test_all_stationary(self):
        for r in self.summary["re20"]:
            assert r["stationary"], f"D={r['D_cells']} not stationary (drift={r['drift']:.2e})"

    def test_richardson_extrapolation_within_5pct(self):
        rows = self.summary["re20"]
        h_vals = [r["h"] for r in rows]
        cd_vals = [r["cd"] for r in rows]
        cd_inf, p_obs = richardson_extrapolation(h_vals, cd_vals)

        assert not math.isnan(cd_inf), "Richardson extrapolation failed (NaN)"
        assert 0.5 < p_obs < 3.0, f"Observed order p={p_obs:.3f} outside expected range [0.5, 3.0]"

        err = abs(cd_inf - self.ref_cd) / self.ref_cd * 100.0
        assert err <= 5.0, (
            f"Re=20 Cd_inf={cd_inf:.6f} vs ref={self.ref_cd:.6f} → err={err:+.3f}% > 5% gate. "
            f"p_obs={p_obs:.3f}, grids={[r['D_cells'] for r in rows]}"
        )

    def test_finest_grid_within_5pct(self):
        """The finest grid (D=80) should also be within 5% even without extrapolation."""
        finest = self.summary["re20"][-1]
        err = abs(finest["cd"] - self.ref_cd) / self.ref_cd * 100.0
        assert err <= 5.0, f"D=80 Cd={finest['cd']:.6f} err={err:+.3f}% > 5%"

    def test_richardson_summary_matches(self):
        """The pre-computed Richardson summary should be consistent."""
        rich = self.summary["richardson"]["re20"]
        assert abs(rich["cd_inf"] - self.ref_cd) / self.ref_cd * 100.0 <= 5.0


class TestRe100Convergence:
    """Re=100 (2D-2) unsteady cylinder drag convergence."""

    @pytest.fixture(autouse=True)
    def load_data(self):
        self.summary = _load_summary(ARTIFACT_RE100)
        self.ref_band = REFERENCE_BANDS["2D-2"]["cd"]
        self.ref_mid = band_mid(self.ref_band)

    def test_artifact_exists(self):
        assert ARTIFACT_RE100.exists(), f"Missing {ARTIFACT_RE100}"

    def test_three_grid_levels(self):
        rows = self.summary["re100"]
        assert len(rows) == 3, f"Expected 3 grid levels, got {len(rows)}"
        D_vals = [r["D_cells"] for r in rows]
        assert D_vals == [20, 28, 40], f"Expected D=[20,28,40], got {D_vals}"

    def test_richardson_extrapolation_within_5pct(self):
        rows = self.summary["re100"]
        h_vals = [r["h"] for r in rows]
        cd_vals = [r["cd"] for r in rows]
        cd_inf, p_obs = richardson_extrapolation(h_vals, cd_vals)

        assert not math.isnan(cd_inf), "Richardson extrapolation failed (NaN)"
        assert 0.5 < p_obs < 3.0, f"Observed order p={p_obs:.3f} outside expected range [0.5, 3.0]"

        err = abs(cd_inf - self.ref_mid) / self.ref_mid * 100.0
        assert err <= 5.0, (
            f"Re=100 Cd_inf={cd_inf:.6f} vs ref_mid={self.ref_mid:.4f} → err={err:+.3f}% > 5% gate. "
            f"p_obs={p_obs:.3f}, grids={[r['D_cells'] for r in rows]}"
        )

    def test_strouhal_within_band(self):
        """Strouhal number should be within the reference band [0.295, 0.305]."""
        st_band = REFERENCE_BANDS["2D-2"]["st"]
        for r in self.summary["re100"]:
            st = r["st"]
            assert st_band[0] <= st <= st_band[1], (
                f"D={r['D_cells']} St={st:.4f} outside [{st_band[0]}, {st_band[1]}]"
            )

    def test_richardson_summary_matches(self):
        rich = self.summary["richardson"]["re100"]
        assert abs(rich["cd_inf"] - self.ref_mid) / self.ref_mid * 100.0 <= 5.0
