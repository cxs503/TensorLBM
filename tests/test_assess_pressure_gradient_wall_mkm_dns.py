from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / (
    "assess_pressure_gradient_wall_mkm_dns.py"
)
SPEC = importlib.util.spec_from_file_location("mkm_wall_assessor", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_profile(path: Path) -> None:
    # A compact profile is sufficient to exercise metadata parsing, interval
    # selection, exact pressure-gradient normalization and fail-closed output.
    path.write_text(
        "# Re_tau = 100.0\n"
        "# y y+ Umean\n"
        "0.0 0.0 0.0\n"
        "0.3 30.0 12.0\n"
        "0.6 60.0 14.0\n"
        "0.9 90.0 15.0\n",
        encoding="utf-8",
    )


def test_mkm_assessor_uses_exact_channel_normalization(tmp_path) -> None:
    profile = tmp_path / "means"
    _write_profile(profile)
    result = MODULE.assess(
        profile,
        minimum_y_plus=25.0,
        maximum_y_plus=95.0,
        maximum_mean_error_pct=100.0,
        maximum_rms_error_pct=100.0,
    )
    assert result["normalization"]["nu"] == pytest.approx(0.01)
    assert result["normalization"]["pressure_gradient_acceleration"] == -1.0
    assert result["assessment"]["sample_y_plus"] == [30.0, 60.0, 90.0]
    assert set(result["models"]) == {"van_driest", "duprat"}
    assert len(result["source"]["sha256"]) == 64
    assert result["production_force_changed"] is False


def test_mkm_assessor_rejects_empty_interval(tmp_path) -> None:
    profile = tmp_path / "means"
    _write_profile(profile)
    with pytest.raises(ValueError, match="insufficient"):
        MODULE.assess(profile, minimum_y_plus=1.0, maximum_y_plus=2.0)
