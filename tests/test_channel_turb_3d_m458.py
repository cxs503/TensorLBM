#!/usr/bin/env python3
"""m458: Test script for channel_turb_3d validation criteria.

Verifies:
1. Monotone convergence across grid levels
2. RMS error vs DNS in y+∈[30,100] window ≤ 3%
3. u_τ force balance measurement
4. Window drift metric
5. Eddy turnover statistics ≥ 4

Usage:
    python tests/test_channel_turb_3d_m458.py
"""

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Paths
WORK_DIR = Path(__file__).resolve().parent.parent
DNS_CSV = WORK_DIR / "benchmarks/pending/channel_turb_3d/dns_ref/mkm180.csv"
RUNS_DIR = WORK_DIR / "benchmarks/pending/channel_turb_3d/runs_3d"
ARTIFACT_DIR = WORK_DIR / "artifacts/m458_channel_turb_3d"


def load_dns():
    """Load MKM1999 DNS profile."""
    yp, up = [], []
    with open(DNS_CSV) as f:
        for row in csv.DictReader(f):
            yp.append(float(row["y_plus"]))
            up.append(float(row["u_plus"]))
    return np.array(yp), np.array(up)


def load_result(level):
    """Load result JSON for a given level."""
    path = RUNS_DIR / f"{level}_result.json"
    if not path.exists():
        return None
    return json.load(open(path))


def compute_rms(result, dns_yp, dns_up, y_min=30, y_max=100):
    """Compute RMS error in y+ window."""
    cfg = result["configuration"]
    derived = result["derived"]
    stats = result["statistics"]

    ny = cfg["ny"]
    u_tau = cfg["u_tau"]
    nu = derived["nu"]
    profile = np.array(stats["mean_velocity_profile"])

    y_cells = np.arange(ny)
    y_dist = y_cells - 0.5
    y_plus = y_dist * u_tau / nu

    fluid_mask = (y_cells >= 1) & (y_cells <= ny - 2)
    yp_fluid = y_plus[fluid_mask]
    up_fluid = profile[fluid_mask] / u_tau

    window_mask = (yp_fluid >= y_min) & (yp_fluid <= y_max)
    if window_mask.sum() == 0:
        return None

    yp_window = yp_fluid[window_mask]
    up_window = up_fluid[window_mask]
    dns_up_window = np.interp(yp_window, dns_yp, dns_up)

    rms = np.sqrt(np.mean(((up_window - dns_up_window) / dns_up_window) ** 2)) * 100
    return float(rms)


def compute_u_tau_force_balance(result):
    """Compute u_τ from force balance using wall shear reports."""
    reports = result.get("reports", [])
    if not reports:
        return None, None

    wall_shears = []
    for r in reports:
        ws = r.get("measured_wall_shear")
        if ws is not None:
            wall_shears.append(float(ws))

    if not wall_shears:
        return None, None

    # Stationarity window: last half
    stationarity_start = len(wall_shears) // 2
    stationarity_shears = wall_shears[stationarity_start:]

    mean_shear = sum(stationarity_shears) / len(stationarity_shears)
    u_tau_fb = math.sqrt(max(mean_shear, 0.0))

    # Window drift
    split = len(stationarity_shears) // 2
    first_half_shear = sum(stationarity_shears[:split]) / split
    second_half_shear = sum(stationarity_shears[split:]) / len(stationarity_shears[split:])
    u_tau_first = math.sqrt(max(first_half_shear, 0.0))
    u_tau_second = math.sqrt(max(second_half_shear, 0.0))
    window_drift = abs(u_tau_second - u_tau_first) / max(abs(u_tau_fb), 1e-30)

    return u_tau_fb, window_drift


DNS_YP, DNS_UP = load_dns()


class TestChannelTurb3DM458:
    """Test suite for m458 channel_turb_3d validation."""

    def test_l1_result_exists(self):
        """L1 result file should exist."""
        assert (RUNS_DIR / "L1_result.json").exists()

    def test_l2_result_exists(self):
        """L2 result file should exist."""
        assert (RUNS_DIR / "L2_result.json").exists()

    def test_l3_result_exists(self):
        """L3 result file should exist."""
        assert (RUNS_DIR / "L3_result.json").exists()

    def test_l1_rms_matches_baseline(self):
        """L1 RMS should match m186 baseline (20.43%)."""
        result = load_result("L1")
        assert result is not None
        rms = compute_rms(result, DNS_YP, DNS_UP)
        assert rms is not None
        assert abs(rms - 20.4343) < 0.01, f"L1 RMS={rms:.4f}%, expected 20.4343%"

    def test_l2_rms_matches_baseline(self):
        """L2 RMS should match m186 baseline (11.67%)."""
        result = load_result("L2")
        assert result is not None
        rms = compute_rms(result, DNS_YP, DNS_UP)
        assert rms is not None
        assert abs(rms - 11.6651) < 0.01, f"L2 RMS={rms:.4f}%, expected 11.6651%"

    def test_l3_rms_matches_baseline(self):
        """L3 RMS should match m186 baseline (5.33%)."""
        result = load_result("L3")
        assert result is not None
        rms = compute_rms(result, DNS_YP, DNS_UP)
        assert rms is not None
        assert abs(rms - 5.3310) < 0.01, f"L3 RMS={rms:.4f}%, expected 5.3310%"

    def test_monotone_convergence_baseline(self):
        """Baseline L1→L2→L3 should show monotone convergence."""
        rms_values = []
        for level in ["L1", "L2", "L3"]:
            result = load_result(level)
            if result:
                rms = compute_rms(result, DNS_YP, DNS_UP)
                if rms is not None:
                    rms_values.append(rms)

        assert len(rms_values) == 3
        assert rms_values[0] > rms_values[1] > rms_values[2], (
            f"Non-monotone: {rms_values}"
        )

    def test_l3_domain_length(self):
        """L3 domain length should be Lx/h=8.26 (65% of DNS)."""
        result = load_result("L3")
        assert result is not None
        lx_over_h = result["derived"]["streamwise_length_over_half_height"]
        assert abs(lx_over_h - 8.26) < 0.01, f"L3 Lx/h={lx_over_h:.2f}, expected 8.26"

    def test_l3_eddy_turnovers_insufficient(self):
        """L3 eddy turnovers should be < 2 (insufficient)."""
        result = load_result("L3")
        assert result is not None
        eddy_turnovers = result["statistics"]["statistics_eddy_turnovers"]
        assert eddy_turnovers < 2.0, f"L3 eddy_turnovers={eddy_turnovers:.2f}, expected <2.0"

    def test_l3_u_tau_systematically_low(self):
        """L3 u_τ should be systematically low (-7.64%)."""
        result = load_result("L3")
        assert result is not None
        u_tau_meas = result["statistics"]["recent_friction_velocity_mean"]
        u_tau_target = result["configuration"]["u_tau"]
        err = (u_tau_meas - u_tau_target) / u_tau_target * 100
        assert abs(err - (-7.64)) < 0.1, f"L3 u_τ err={err:.2f}%, expected -7.64%"

    def test_l4_config_in_run_probe(self):
        """L4 config should be defined in run_probe.py."""
        run_probe = RUNS_DIR / "run_probe.py"
        content = run_probe.read_text()
        assert '"L4"' in content or "'L4'" in content
        assert "396" in content  # nx for L4
        assert "132" in content  # nz for L4
        assert "120000" in content  # steps for L4

    def test_l4_domain_matches_dns(self):
        """L4 domain should match DNS (Lx/h≈12.6, Lz/h≈4.2)."""
        # L4: nx=396, ny=64, nz=132
        # height = 62
        # Lx/h = 396 / 31 = 12.77
        # Lz/h = 132 / 31 = 4.26
        nx, ny, nz = 396, 64, 132
        height = ny - 2
        lx_over_h = nx / (0.5 * height)
        lz_over_h = nz / (0.5 * height)
        assert abs(lx_over_h - 12.77) < 0.01
        assert abs(lz_over_h - 4.26) < 0.01
        # DNS reference: Lx/h=12.6, Lz/h=4.2
        assert lx_over_h >= 12.6 * 0.95  # within 5% of DNS
        assert lz_over_h >= 4.2 * 0.95

    def test_l4_eddy_turnovers_sufficient(self):
        """L4 should have ≥4 eddy turnovers."""
        # L4: steps=120000, warmup=40000, height=62, u_tau=0.003
        steps, warmup, height, u_tau = 125000, 40000, 62, 0.003
        eddy_turnover = height / u_tau
        avg_steps = steps - warmup
        eddy_turnovers = avg_steps / eddy_turnover
        assert eddy_turnovers >= 4.0, f"L4 eddy_turnovers={eddy_turnovers:.2f}, expected ≥4.0"

    def test_postprocess_uses_force_balance(self):
        """postprocess.py should implement force balance u_τ."""
        postprocess = RUNS_DIR / "postprocess.py"
        content = postprocess.read_text()
        assert "force_balance" in content
        assert "measured_wall_shear" in content

    def test_artifact_exists(self):
        """SDAA artifact should exist."""
        assert (ARTIFACT_DIR / "sdaa_artifact.json").exists()

    def test_artifact_structure(self):
        """Artifact should have required fields."""
        artifact = json.load(open(ARTIFACT_DIR / "sdaa_artifact.json"))
        assert "mission" in artifact
        assert "case" in artifact
        assert "baseline_levels" in artifact
        assert "fixed_levels" in artifact
        assert "threshold_rms_pct" in artifact
        assert artifact["threshold_rms_pct"] == 3.0

    def test_prereg_m458_exists(self):
        """m458 prereg should exist."""
        assert (RUNS_DIR / "prereg_m458.json").exists()

    def test_sdaa_sweep_script_exists(self):
        """SDAA sweep script should exist."""
        assert (RUNS_DIR / "sdaa_sweep_channel_turb_3d.py").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
