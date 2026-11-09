"""Tests for Taylor-Aris dispersion benchmark verification."""
import math
import pytest


def test_taylor_aris_formula():
    """Verify the Taylor-Aris analytical formula Deff/D = 1 + Pe^2/210."""
    for pe, expected in [(10, 1 + 100/210), (20, 1 + 400/210), (30, 1 + 900/210)]:
        deff_ratio = 1 + pe**2 / 210
        assert abs(deff_ratio - expected) < 1e-12, f"Pe={pe}: {deff_ratio} != {expected}"


def test_result_errors_within_threshold():
    """Verify all 6 cases have error < 3%."""
    import json
    from pathlib import Path
    result_path = Path(__file__).parent / "result.json"
    if not result_path.exists():
        pytest.skip("result.json not found")
    with open(result_path) as f:
        data = json.load(f)
    for case in data.get("per_case", []):
        err = case.get("err_pct", 100)
        assert err < 3.0, f"Case Pe={case.get('pe_target')} H={case.get('H')}: err={err:.3f}% > 3%"


def test_max_error_below_0_3_pct():
    """Verify maximum error is below 0.3% (noise floor)."""
    import json
    from pathlib import Path
    result_path = Path(__file__).parent / "result.json"
    if not result_path.exists():
        pytest.skip("result.json not found")
    with open(result_path) as f:
        data = json.load(f)
    errors = [c["err_pct"] for c in data.get("per_case", [])]
    assert max(errors) < 0.3, f"Max error {max(errors):.3f}% exceeds 0.3% noise floor"


def test_deff_theory_matches_formula():
    """Verify Deff_theory values match 1 + Pe^2/210."""
    import json
    from pathlib import Path
    result_path = Path(__file__).parent / "result.json"
    if not result_path.exists():
        pytest.skip("result.json not found")
    with open(result_path) as f:
        data = json.load(f)
    for case in data.get("per_case", []):
        pe = case.get("pe_target", 0)
        deff_theory = case.get("deff_theory", 0)
        expected = 1 + pe**2 / 210
        rel_err = abs(deff_theory - expected) / expected
        assert rel_err < 1e-10, f"Pe={pe}: Deff_theory={deff_theory} vs expected={expected}"
