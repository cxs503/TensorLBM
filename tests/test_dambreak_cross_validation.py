"""TDD tests for dam-break cross-validation runner (D3Q19/D3Q27 × BGK/MRT × SGS).

Tests the cross-validation runner that compares front-position evolution
across lattice (D3Q19/D3Q27), collision (BGK/MRT), and SGS model
(none/Smagorinsky/WALE/Vreman) combinations on a small grid.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tensorlbm.dambreak_cross_validation import (
    CrossValidationConfig,
    run_single_dambreak,
    run_dambreak_cross_validation,
)


def _small_config(tmp_path, **overrides):
    """Tiny grid for fast tests."""
    kwargs = {
        "nx": 12,
        "ny": 6,
        "nz": 6,
        "dam_width": 4,
        "fill_height": 4,
        "n_steps": 3,
        "tau": 0.8,
        "gravity": 1e-4,
        "output_path": str(tmp_path / "cv.json"),
    }
    kwargs.update(overrides)
    return CrossValidationConfig(**kwargs)


# ---------------------------------------------------------------------------
# Single-run smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lattice", ["d3q19", "d3q27"])
def test_single_dambreak_runs_for_both_lattices(tmp_path, lattice):
    """Both lattices must produce a result dict with required fields."""
    config = _small_config(tmp_path)
    result = run_single_dambreak(lattice, "bgk", "none", config)
    assert result["lattice"] == lattice
    assert result["collision"] == "bgk"
    assert result["sgs_model"] == "none"
    assert isinstance(result["front_position"], float)
    assert isinstance(result["mass_drift"], float)
    assert isinstance(result["finite"], bool)
    assert result["finite"] is True


@pytest.mark.parametrize("collision", ["bgk", "mrt"])
def test_single_dambreak_runs_for_both_collisions(tmp_path, collision):
    """BGK and MRT collision must both run to completion."""
    config = _small_config(tmp_path)
    result = run_single_dambreak("d3q19", collision, "none", config)
    assert result["collision"] == collision
    assert result["finite"] is True


@pytest.mark.parametrize("sgs_model", ["none", "smagorinsky", "wale", "vreman"])
def test_d3q19_bgk_all_sgs_models(tmp_path, sgs_model):
    """D3Q19 BGK must support all four SGS options."""
    config = _small_config(tmp_path)
    result = run_single_dambreak("d3q19", "bgk", sgs_model, config)
    assert result["sgs_model"] == sgs_model
    assert result["finite"] is True


@pytest.mark.parametrize("sgs_model", ["none", "smagorinsky", "wale", "vreman"])
def test_d3q27_bgk_all_sgs_models(tmp_path, sgs_model):
    """D3Q27 BGK must support all four SGS options."""
    config = _small_config(tmp_path)
    result = run_single_dambreak("d3q27", "bgk", sgs_model, config)
    assert result["sgs_model"] == sgs_model
    assert result["finite"] is True


@pytest.mark.parametrize("sgs_model", ["none", "smagorinsky", "wale", "vreman"])
def test_d3q19_mrt_all_sgs_models(tmp_path, sgs_model):
    """D3Q19 MRT must support all four SGS options."""
    config = _small_config(tmp_path)
    result = run_single_dambreak("d3q19", "mrt", sgs_model, config)
    assert result["sgs_model"] == sgs_model
    assert result["finite"] is True


@pytest.mark.parametrize("sgs_model", ["none", "smagorinsky", "wale", "vreman"])
def test_d3q27_mrt_all_sgs_models(tmp_path, sgs_model):
    """D3Q27 MRT must support all four SGS options."""
    config = _small_config(tmp_path)
    result = run_single_dambreak("d3q27", "mrt", sgs_model, config)
    assert result["sgs_model"] == sgs_model
    assert result["finite"] is True


# ---------------------------------------------------------------------------
# Full matrix tests
# ---------------------------------------------------------------------------

def test_cross_validation_produces_full_matrix(tmp_path):
    """Full matrix: 2 lattices × 2 collisions × 4 SGS = 16 entries."""
    config = _small_config(tmp_path)
    result = run_dambreak_cross_validation(config)
    assert result["status"] == "diagnostic_only"
    assert result["physical_validation"] is False
    matrix = result["matrix"]
    assert len(matrix) == 16
    for entry in matrix:
        assert "lattice" in entry
        assert "collision" in entry
        assert "sgs_model" in entry
        assert "front_position" in entry
        assert "mass_drift" in entry
        assert "finite" in entry
        assert isinstance(entry["front_position"], float)
        assert isinstance(entry["mass_drift"], float)
        assert isinstance(entry["finite"], bool)


def test_cross_validation_writes_json_artifact(tmp_path):
    """Artifact must be written as machine-readable JSON."""
    config = _small_config(tmp_path)
    run_dambreak_cross_validation(config)
    artifact_path = Path(config.output_path)
    assert artifact_path.exists()
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert data["status"] == "diagnostic_only"
    assert data["physical_validation"] is False
    assert len(data["matrix"]) == 16
    # Verify each entry has the required fields
    for entry in data["matrix"]:
        for key in ("lattice", "collision", "sgs_model",
                    "front_position", "mass_drift", "finite"):
            assert key in entry, f"missing key {key} in entry {entry}"


def test_matrix_contains_all_combinations(tmp_path):
    """Matrix must contain every lattice × collision × sgs combination."""
    config = _small_config(tmp_path)
    result = run_dambreak_cross_validation(config)
    combos = {
        (e["lattice"], e["collision"], e["sgs_model"])
        for e in result["matrix"]
    }
    expected = {
        (lat, col, sgs)
        for lat in ("d3q19", "d3q27")
        for col in ("bgk", "mrt")
        for sgs in ("none", "smagorinsky", "wale", "vreman")
    }
    assert combos == expected


# ---------------------------------------------------------------------------
# Physical sanity tests (diagnostic, not validation)
# ---------------------------------------------------------------------------

def test_front_position_is_within_domain(tmp_path):
    """Front position must be a valid x-index within [0, nx-1]."""
    config = _small_config(tmp_path)
    result = run_single_dambreak("d3q19", "bgk", "none", config)
    assert 0.0 <= result["front_position"] <= config.nx - 1


def test_none_sgs_gives_same_result_as_c_s_zero(tmp_path):
    """'none' SGS must be equivalent to C_s=0 (no SGS contribution)."""
    config = _small_config(tmp_path)
    r1 = run_single_dambreak("d3q19", "bgk", "none", config)
    # Run with smagorinsky but C_s=0 — should match 'none'
    config_zero = _small_config(tmp_path, C_smag=0.0)
    r2 = run_single_dambreak("d3q19", "bgk", "smagorinsky", config_zero)
    assert r1["front_position"] == r2["front_position"]
    assert abs(r1["mass_drift"] - r2["mass_drift"]) < 1e-10


def test_mass_drift_is_finite_and_small(tmp_path):
    """Mass drift should be finite and not catastrophic for short runs."""
    config = _small_config(tmp_path)
    result = run_single_dambreak("d3q19", "bgk", "none", config)
    assert torch.isfinite(torch.tensor(result["mass_drift"]))
    # For 3 steps on a 12×6×6 grid, drift can be significant due to
    # topology conversion; just check it's not catastrophic (< 50%).
    total_mass = config.dam_width * config.fill_height * config.nz
    assert abs(result["mass_drift"]) < 0.5 * total_mass
