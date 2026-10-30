"""Diagnostic test for channel_turb_3d HONEST-FAIL verdict (m458/m490).

This test verifies that the channel_turb_3d case correctly reports
HONEST-FAIL: all unit tests pass (library functionality is correct),
but the physics validation criterion (RMS ≤ 3%) is not met.

Historical context:
- m455: FAIL (gate=no-evidence, no worktree changes)
- m458: HONEST-FAIL (18 tests passed, but RMS=5.33% > 3%)
- m490: Terminal verdict confirmed with full diagnostic report
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Work directory root
WORK = Path(__file__).resolve().parent.parent
HERE = WORK / "benchmarks" / "pending" / "channel_turb_3d"


def _load_validation_report() -> dict:
    return json.loads((HERE / "validation_report.json").read_text())


def _load_result(level: str) -> dict:
    return json.loads((HERE / "runs_3d" / f"{level}_result.json").read_text())


# --- Gate verification tests (these all PASS) ---

class TestGate1LibraryEntry:
    """G1: Library entry point exists and is importable."""

    def test_import_wall_resolved_channel3d(self):
        from tensorlbm.wall_resolved_channel3d import (
            WallResolvedChannel3DConfig,
            run_wall_resolved_channel3d,
        )
        assert WallResolvedChannel3DConfig is not None
        assert run_wall_resolved_channel3d is not None

    def test_config_dataclass(self):
        from tensorlbm.wall_resolved_channel3d import WallResolvedChannel3DConfig
        cfg = WallResolvedChannel3DConfig(nx=8, ny=10, nz=8, re_tau=20, u_tau=0.01, steps=1, device="cpu")
        assert cfg.height == 8
        assert cfg.nu > 0
        assert cfg.tau > 0.5


class TestGate2DirectObservable:
    """G2: Direct observable (RMS error) is computed correctly."""

    def test_validation_report_exists(self):
        assert (HERE / "validation_report.json").exists()

    def test_rms_values_programmatic(self):
        vr = _load_validation_report()
        rms_values = [g["rms_y30_100_pct"] for g in vr["grid_levels"]]
        assert len(rms_values) == 3
        assert all(v > 0 for v in rms_values)

    def test_rms_exceeds_threshold(self):
        """The key HONEST-FAIL condition: best RMS > 3%."""
        vr = _load_validation_report()
        best_rms = min(g["rms_y30_100_pct"] for g in vr["grid_levels"])
        assert best_rms > 3.0, f"Best RMS={best_rms:.2f}% should exceed 3% threshold"

    def test_u_tau_systematically_low(self):
        """u_tau is systematically underestimated across all grid levels."""
        vr = _load_validation_report()
        for g in vr["grid_levels"]:
            assert g["u_tau_err_pct"] < 0, f"{g['level']}: u_tau error should be negative"


class TestGate3MonotoneConvergence:
    """G3: Monotone convergence is demonstrated."""

    def test_monotone_decrease(self):
        vr = _load_validation_report()
        rms_values = [g["rms_y30_100_pct"] for g in vr["grid_levels"]]
        assert vr["convergence"]["monotone"] is True
        assert rms_values[0] > rms_values[1] > rms_values[2]

    def test_convergence_rate(self):
        """Convergence rate is reasonable (not diverging)."""
        vr = _load_validation_report()
        rms_values = [g["rms_y30_100_pct"] for g in vr["grid_levels"]]
        improvement_l1_l2 = (rms_values[0] - rms_values[1]) / rms_values[0]
        improvement_l2_l3 = (rms_values[1] - rms_values[2]) / rms_values[1]
        assert improvement_l1_l2 > 0.3  # >30% improvement
        assert improvement_l2_l3 > 0.4  # >40% improvement


class TestGate4ReferenceLocked:
    """G4: Reference data is locked and verified."""

    def test_prereg_exists(self):
        assert (HERE / "runs_3d" / "prereg.json").exists()

    def test_dns_ref_exists(self):
        assert (HERE / "dns_ref" / "mkm180.csv").exists()

    def test_dns_data_valid(self):
        import csv
        dns_path = HERE / "dns_ref" / "mkm180.csv"
        with open(dns_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 50  # MKM1999 has ~65 points
        assert float(rows[0]["y_plus"]) == 0.0
        assert float(rows[-1]["y_plus"]) > 170  # Re_tau=178.12


class TestFigures:
    """Figures exist and are non-trivial."""

    def test_uplus_profile_figure(self):
        fig = HERE / "figures" / "fig_uplus_profile.png"
        assert fig.exists()
        assert fig.stat().st_size > 10000  # >10KB

    def test_convergence_figure(self):
        fig = HERE / "figures" / "fig_convergence.png"
        assert fig.exists()
        assert fig.stat().st_size > 10000

    def test_rms_convergence_figure(self):
        fig = HERE / "figures" / "fig_rms_convergence.png"
        assert fig.exists()
        assert fig.stat().st_size > 10000


class TestReadme:
    """README is comprehensive."""

    def test_readme_exists(self):
        assert (HERE / "README.md").exists()

    def test_readme_substantial(self):
        content = (HERE / "README.md").read_text()
        assert len(content) > 1000  # Substantial documentation

    def test_readme_has_provenance(self):
        content = (HERE / "README.md").read_text()
        assert "PROVENANCE" in content


class TestHonestFailVerdict:
    """The HONEST-FAIL verdict is correctly documented."""

    def test_verdict_is_terminal(self):
        vr = _load_validation_report()
        assert vr["verdict"] == "terminal"

    def test_reason_documented(self):
        vr = _load_validation_report()
        assert "5.33" in vr["reason"]  # Best RMS value mentioned
        assert "3%" in vr["reason"]  # Threshold mentioned

    def test_gates_json_exists(self):
        assert (HERE / "gates.json").exists()

    def test_gates_verified_json_exists(self):
        assert (HERE / "gates_verified.json").exists()

    def test_diagnostic_artifact_exists(self):
        diag = WORK / "artifacts" / "channel_turb_3d" / "diagnostic_m490.json"
        assert diag.exists()

    def test_report_exists(self):
        report = WORK / "docs" / "reports" / "report_channel_turb_3d_m490.md"
        assert report.exists()
        assert report.stat().st_size > 1000
