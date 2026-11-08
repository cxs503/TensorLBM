"""End-to-end test of the P40-N09 decomposition driver on synthetic arms.

The campaign analysis (``benchmarks/pending/channel_turb_3d/analyze_p40n09.py``)
is a *judging* script: it decides whether an error is statistical or systematic.
A judging script that only ever runs on the real campaign is untested -- so this
test feeds it three hand-built campaign directories with a known answer:

``clean``     two box lengths that agree, a CI-covered profile, an SGS pair that
              separates -> every gate must pass and the verdict must be
              ``statistics_dominated``;
``biased``    the same but with a 15 % profile bias against MKM -> the verdict
              must flip to ``model_or_domain_dominated`` while the CI gate still
              passes (a bias is not a statistics problem);
``short``     averaging window below three eddy turnovers -> the statistical gate
              must be ``withheld``, never passed.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "benchmarks/pending/channel_turb_3d/analyze_p40n09.py"
REF = REPO / "benchmarks/pending/channel_turb_3d/dns_ref/mkm180.csv"

NY = 24
DELTA = 11.0
U_TAU = 0.01
NU = U_TAU * DELTA / 180.0
Y_W = 0.644
HALF = (NY - 2) // 2


def _reference():
    data = np.loadtxt(REF, delimiter=",", skiprows=1)
    return data[:, 1], data[:, 2]


def _profile(scale: float) -> np.ndarray:
    """Full-height mean profile (u, lattice units) matching the folded grid."""
    ref_y_plus, ref_u_plus = _reference()
    y = np.arange(1, 1 + HALF) - Y_W
    u_plus = np.interp(y * U_TAU / NU, ref_y_plus, ref_u_plus) * scale
    profile = np.zeros(NY)
    profile[1 : 1 + HALF] = u_plus * U_TAU
    profile[NY - 1 - np.arange(1, 1 + HALF)] = u_plus * U_TAU
    return profile


def _write_arm(root: Path, case: str, *, scale: float, n_batches: int, turns: float, seed: int):
    rng = np.random.default_rng(seed)
    mean = _profile(scale)
    batches = []
    for _ in range(n_batches):
        noise = rng.normal(scale=0.15 * U_TAU, size=NY)
        noise[0] = noise[-1] = 0.0
        noise[NY - 1 - np.arange(1, 1 + HALF)] = noise[1 : 1 + HALF]
        moments = [[0.0] * NY for _ in range(7)]
        moments[0] = list(mean + noise)
        batches.append(moments)
    result = {
        "schema": "tensorlbm-wall-resolved-channel3d-result-v1",
        "configuration": {
            "nx": 138,
            "ny": NY,
            "nz": 46,
            "u_tau": U_TAU,
            "re_tau": 180.0,
            "nu": NU,
            "smagorinsky_cs": 0.0,
            "collision_model": "natural_kbc",
            "wall_origin": 0.5,
        },
        "derived": {
            "height": NY - 2,
            "nu": NU,
            "tau": 0.5 + 3.0 * NU,
            "body_force_acceleration": 2.0 * U_TAU**2 / (NY - 2),
            "smagorinsky_cs": 0.0,
        },
        "statistics": {
            "profile_samples": n_batches * 10,
            "mean_velocity_profile": list(mean),
            "reynolds_stress_profiles_uu_vv_ww_uv": [[0.0] * NY for _ in range(4)],
            "statistics_steps": int(turns * DELTA / U_TAU),
            "eddy_turnover_steps": DELTA / U_TAU,
            "statistics_eddy_turnovers": 2.0 * turns,
            "stationarity_half_window_drift_fraction": 0.005,
            "recent_crossflow_rms_over_u_tau": 0.6,
            "profile_batch_count": n_batches,
            "profile_batch_counts": [10] * n_batches,
            "profile_batch_moment_means": batches,
        },
        "wall_exchange": {"block_means": [1e-4] * 8, "block_range_fraction": 0.02,
                          "shear_series_stride": 5, "shear_series": [1e-4] * 8},
        "reports": [],
        "collision_execution": {},
        "acceptance": {},
        "wall_units": {},
        "initialization_diagnostics": {},
    }
    meta = {
        "node": "P40-N09",
        "case": case,
        "plan": {
            "case": case,
            "cells": 138 * NY * 46,
            "averaging_eddy_turnovers": turns,
            "eddy_turnover_steps": DELTA / U_TAU,
        },
        "elapsed_s": 3600.0,
        "cell_updates_per_s": 6.0e5,
        "t_int_ledger": {
            "t_int_samples": 60.0,
            "t_int_steps": 300.0,
            "t_int_turnovers": 0.27,
            "n_samples": 1000,
            "n_effective_samples": 8.0,
            "relative_standard_error_of_mean": 0.02,
            "acf_window": 40,
            "clamped_at_uncorrelated_floor": False,
            "observed_steps": int(turns * DELTA / U_TAU),
            "t_obs_turnovers": turns,
            "t_obs_over_t_int": 40.0,
        },
    }
    (root / f"{case}_result.json").write_text(json.dumps(result))
    (root / f"{case}_meta.json").write_text(json.dumps(meta))


def _run_analysis(run_dir: Path, out_dir: Path) -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run-dir",
            str(run_dir),
            "--out-dir",
            str(out_dir),
            "--fig-dir",
            str(out_dir / "figs"),
        ],
        cwd=str(REPO),
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    return json.loads((out_dir / "gates_p40n09.json").read_text())


def _build(root: Path, *, d4_scale: float, d2_scale: float, turns: float, sgs_same: bool = False):
    root.mkdir(parents=True, exist_ok=True)
    _write_arm(root, "D4_cs0", scale=d4_scale, n_batches=24, turns=turns, seed=1)
    _write_arm(root, "D2_cs0", scale=d2_scale, n_batches=24, turns=turns, seed=2)
    _write_arm(root, "SGS_cs0", scale=1.0, n_batches=24, turns=turns, seed=3)
    _write_arm(
        root,
        "SGS_cs01",
        scale=1.0 if sgs_same else 0.8,
        n_batches=24,
        turns=turns,
        seed=4,
    )


def _gate(gates: dict, name: str) -> dict:
    for gate in gates["gates"]:
        if gate["gate"].startswith(name):
            return gate
    raise AssertionError(f"gate {name} missing from {[g['gate'] for g in gates['gates']]}")


def test_clean_campaign_passes_and_attributes_to_statistics(tmp_path):
    _build(tmp_path / "runs", d4_scale=1.0, d2_scale=1.0, turns=12.0)
    gates = _run_analysis(tmp_path / "runs", tmp_path / "out")
    assert _gate(gates, "G_A_steady_state")["status"] == "pass"
    assert _gate(gates, "G_B_statistical_ci")["status"] == "pass"
    assert _gate(gates, "G_C_domain_length")["status"] == "pass"
    verdict_gate = _gate(gates, "G_D_attribution")
    assert verdict_gate["status"] == "pass"
    attribution = json.loads((tmp_path / "out" / "p40n09_numbers.json").read_text())[
        "attribution"
    ]
    assert attribution["verdict"] == "statistics_dominated"
    assert _gate(gates, "G_E_negative_control")["status"] == "pass"
    assert _gate(gates, "G_F_budget")["status"] == "pass"


def test_bias_flips_the_verdict_but_not_the_ci_gate(tmp_path):
    _build(tmp_path / "runs", d4_scale=1.15, d2_scale=1.15, turns=12.0)
    gates = _run_analysis(tmp_path / "runs", tmp_path / "out")
    numbers = json.loads((tmp_path / "out" / "p40n09_numbers.json").read_text())
    attribution = numbers["attribution"]
    assert attribution["observed_relative_l2"] == pytest.approx(0.15, abs=0.02)
    assert attribution["verdict"] == "model_or_domain_dominated"
    assert attribution["residual_beyond_ci"] > attribution["u_sn"]
    # a systematic 15 % bias is not a statistical failure
    assert _gate(gates, "G_B_statistical_ci")["status"] == "pass"


def test_short_averaging_window_is_withheld(tmp_path):
    _build(tmp_path / "runs", d4_scale=1.0, d2_scale=1.0, turns=2.0)
    gates = _run_analysis(tmp_path / "runs", tmp_path / "out")
    assert _gate(gates, "G_A_steady_state")["status"] == "withheld"
    assert _gate(gates, "G_B_statistical_ci")["status"] == "withheld"
    assert gates["verdict"] == "withheld"


def test_missing_arms_are_withheld_not_passed(tmp_path):
    root = tmp_path / "runs"
    root.mkdir(parents=True)
    _write_arm(root, "D4_cs0", scale=1.0, n_batches=24, turns=12.0, seed=1)
    gates = _run_analysis(root, tmp_path / "out")
    assert _gate(gates, "G_C_domain_length")["status"] == "withheld"
    assert _gate(gates, "G_E_negative_control")["status"] == "withheld"
    assert gates["verdict"] == "withheld"
    assert all(not gate["passed"] for gate in gates["gates"] if gate["status"] == "withheld")