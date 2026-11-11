"""Tests for P389W2-N04: 2D dam-break free-surface validation.

Acceptance criteria:
E1: Water front position error ≤ 10% vs Martin-Moyce (t* ≤ 3.0)
E2: Mass conservation drift ≤ 0.5%
E3: Grid convergence GCI ≤ 5%
E4: Negative control: empty domain mass = 0 and stays 0
E5: Zero-hand-copy: experimental data digitized by script, SHA-256 in provenance
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensorlbm.verification.dam_break_2d_validation import (
    DamBreak2DConfig,
    run_dam_break_2d,
    run_dam_break_2d_multires,
    run_negative_control,
    load_martin_moyce_data,
    interpolate_mm_data,
    compute_gci,
)


class TestMartinMoyceData:
    """E5: Experimental data must be digitized by script with SHA-256 provenance."""

    def test_load_data_returns_list_and_sha(self) -> None:
        data, sha = load_martin_moyce_data()
        assert isinstance(data, list)
        assert len(data) > 0
        assert isinstance(sha, str)
        assert len(sha) == 64

    def test_data_has_required_fields(self) -> None:
        data, _ = load_martin_moyce_data()
        for entry in data:
            assert "t_star" in entry
            assert "X_star" in entry

    def test_data_monotonic(self) -> None:
        data, _ = load_martin_moyce_data()
        x_vals = [d["X_star"] for d in data]
        for i in range(len(x_vals) - 1):
            assert x_vals[i] <= x_vals[i + 1] + 1e-9

    def test_interpolation_at_known_points(self) -> None:
        data, _ = load_martin_moyce_data()
        assert abs(interpolate_mm_data(data, 0.0) - 1.0) < 1e-9
        assert abs(interpolate_mm_data(data, 0.5) - 1.45) < 1e-9


class TestGCI:
    def test_gci_basic(self) -> None:
        result = compute_gci(f_fine=3.0, f_medium=2.8, f_coarse=2.5, r21=2.0, r32=2.0, p=2.0)
        assert "gci_fine_pct" in result
        assert result["gci_fine_pct"] >= 0.0

    def test_gci_converged(self) -> None:
        result = compute_gci(f_fine=3.0, f_medium=3.0, f_coarse=3.0, r21=2.0, r32=2.0, p=2.0)
        assert result["gci_fine_pct"] == 0.0


class TestNegativeControl:
    def test_empty_domain_zero_mass(self) -> None:
        result = run_negative_control(nx=32, n_steps=20)
        assert result["passed"] is True


class TestSingleResolution:
    def _fast_config(self, nx: int = 32) -> DamBreak2DConfig:
        return DamBreak2DConfig(
            nx=nx,
            n_steps=30,
            output_interval=10,
        )

    def test_run_returns_dict(self) -> None:
        cfg = self._fast_config()
        result = run_dam_break_2d(cfg)
        assert isinstance(result, dict)
        assert "samples" in result
        assert "metrics" in result
        assert "criteria" in result

    def test_samples_have_required_fields(self) -> None:
        cfg = self._fast_config()
        result = run_dam_break_2d(cfg)
        for s in result["samples"]:
            assert "t_star_mm" in s
            assert "X_star_sim" in s
            assert "X_star_mm" in s
            assert "front_error_pct" in s

    def test_front_advances(self) -> None:
        cfg = self._fast_config(nx=48)
        result = run_dam_break_2d(cfg)
        x_vals = [s["X_star_sim"] for s in result["samples"]]
        for i in range(len(x_vals) - 1):
            assert x_vals[i] <= x_vals[i + 1] + 1e-6


class TestMultiResolution:
    def test_multires_returns_dict(self) -> None:
        result = run_dam_break_2d_multires(
            resolutions=[32, 48],
            n_steps=30,
        )
        assert isinstance(result, dict)
        assert "results" in result
        assert "verdicts" in result

    def test_provenance_included(self) -> None:
        result = run_dam_break_2d_multires(resolutions=[32, 48], n_steps=30)
        assert "provenance" in result
        assert "martin_moyce_sha256" in result["provenance"]


class TestAcceptanceCriteria:
    def test_E4_negative_control(self) -> None:
        neg = run_negative_control(nx=32, n_steps=20)
        assert neg["passed"] is True

    def test_E5_zero_hand_copy(self) -> None:
        result = run_dam_break_2d_multires(resolutions=[32, 48], n_steps=30)
        assert result["verdicts"]["E5_zero_hand_copy"] is True
