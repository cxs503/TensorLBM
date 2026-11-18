"""An arm must reach the registered window with finite data before it can pass."""

import json
import math
import sys

import pytest

from benchmarks.pending.phasefield_water_entry import run_s2_wedge as wedge


def linear_arm(cfg):
    ref = math.pi**3 / (8 * math.tan(math.radians(cfg.zeta_deg)) ** 2)
    samples = []
    for step in range(1, cfg.n_measure + 1):
        h = step * cfg.v_lat - cfg.gap
        fy = cfg.nz * cfg.v_lat**3 * ref * step
        if not 0.02 <= h / cfg.half_length <= 0.10:
            fy += 100.0
        samples.append(dict(step=step, t=step, penetration=h, fy=fy, u_max=2 * cfg.v_lat))
    return samples


def test_units_window_and_incomplete_history():
    cfg = wedge.WedgeRun()
    rows = linear_arm(cfg)
    r = wedge.arm_metrics(cfg, {"step": 0}, rows)
    assert r["verdict"] == "DIAGNOSTIC_PASS"
    assert r["fit_window_steps"] == [300, 1100]
    assert r["measured_slope"] == pytest.approx(r["analytic_slope"])
    assert r["analytic_slope_dimensionless"] == pytest.approx(792.6330443651)
    assert r["measured_slope_dimensionless"] == pytest.approx(r["analytic_slope_dimensionless"])
    assert wedge.arm_metrics(cfg, {"step": 0}, rows[:500])["verdict"] == "FAIL"
    assert wedge.arm_metrics(cfg, {"step": 0}, rows[:500] + rows[501:])["verdict"] == "FAIL"


def test_nonfinite_and_speed_failure_do_not_pass():
    cfg = wedge.WedgeRun()
    rows = linear_arm(cfg)
    rows[99]["u_max"] = 3.1 * cfg.v_lat
    assert wedge.arm_metrics(cfg, {"step": 0}, rows)["verdict"] == "FAIL"
    rows[140]["fy"] = float("nan")
    r = wedge.arm_metrics(cfg, {"step": 0}, rows[:141])
    assert r["first_nonfinite_step"] == 141
    assert r["measured_slope"] is None and r["fit_r2"] is None
    assert r["verdict"] == "FAIL"
    json.dumps([r, wedge.json_observation(rows[140])], allow_nan=False)


def test_failed_cli_keeps_summary_and_returns_nonzero(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["run_s2_wedge", "--arm", "zeta4_dw100", "--out-dir", str(tmp_path)]
    )
    mock = wedge.arm_metrics(wedge.WedgeRun(), {"step": 0, "u_max": float("nan")}, [])
    monkeypatch.setattr(
        wedge, "run", lambda cfg, out: dict(arm="zeta4_dw100", config=vars(cfg), **mock)
    )
    assert wedge.main() != 0
    assert json.loads((tmp_path / "s2_summary.json").read_text())[0]["verdict"] == "FAIL"
