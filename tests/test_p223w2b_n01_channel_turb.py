"""Tests for the turbulent-channel gate audit (p223w2b-n01).

Pure reductions on the in-repo MKM-1999 reference -- no solver, no compute.
The falsification guards (Dean correlation vs exact DNS) are what stops a future
campaign from grading criterion (d) as written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks/p223w2b_n01_channel_turb"))

import audit  # noqa: E402


def test_reference_profiles_load_with_provenance():
    prov = audit.load_provenance()
    assert prov["upstream_sha256"]
    p = audit.load_profile(180)
    assert p.re_tau == pytest.approx(178.12, abs=0.01)
    assert p.y_plus.size == p.u_plus.size > 10
    assert np.all(np.diff(p.y_plus) > 0)


def test_friction_coefficient_definition():
    p = audit.load_profile(180)
    out = audit.cf_audit(p)
    assert out["cf_dns"] == pytest.approx(2.0 / out["bulk_plus"] ** 2)
    assert out["re_bulk"] == pytest.approx(p.re_tau * out["bulk_plus"])


def test_DEAN_CORRELATION_IS_MIS_SPECIFIED_at_channel_re():
    """Exact DNS data fails criterion (d) as written -- by ~24 %.

    This is the falsification guard: if the Dean deviation ever drops below 10 %
    the correlation or the Re convention changed and the report must be re-read.
    """
    for nominal in audit.available_cases():
        out = audit.cf_audit(audit.load_profile(nominal))
        assert out["rel_dev_dean"] > 0.10, (nominal, out["rel_dev_dean"])


def test_log_law_band_is_attainable_on_reference():
    p = audit.load_profile(180)
    err = audit.log_law_error(p.y_plus, p.u_plus, band=(30.0, 100.0))
    assert err["max_rel"] < 0.05, err
    # ... but the headroom is small: this is the documented caveat
    assert err["max_rel"] > 0.02


def test_stress_peak_targets():
    peaks = audit.stress_peaks(audit.load_profile(180))
    uu = peaks["uu_peak"]
    assert 6.5 < uu["value"] < 7.8
    assert 10.0 < uu["y_plus"] < 20.0
    assert peaks["minus_uv"]["value"] > 0.5


def test_spectral_slope_reduction_recovers_minus_5_3():
    k = np.logspace(0, 1, 40)
    good = audit.spectral_slope(k, k ** (-5.0 / 3.0))
    assert good["slope"] == pytest.approx(-5.0 / 3.0, rel=1e-6)
    bad = audit.spectral_slope(k, k ** (-2.0))
    assert abs(bad["rel_err_vs_minus_5_3"]) > 0.1


def test_audit_payload_has_verdicts(tmp_path):
    path = audit.write_audit(tmp_path)
    payload = json.loads(Path(path).read_text())
    assert payload["schema"] == "tensorlbm-turbulent-channel-gate-audit-v1"
    assert set(payload["verdicts"]) == {
        "a_log_law_5pct", "b_stress_10pct", "c_spectrum_slope_10pct",
        "d_dean_5pct", "e_grid_convergence_3pct",
    }
    assert payload["verdicts"]["d_dean_5pct"]["reference_attains"] is False
    assert payload["power_law"]["max_rel_dev"] < 0.01
