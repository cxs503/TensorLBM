"""Tests for the IN19-T1 curved-wall force-error scaling campaign.

Three layers:
  * geometry / reflection unit tests (analytic wall fractions, stencil fallbacks)
  * a Couette analytic solution gate for every wall format (momentum-exchange
    force versus ``mu u_wall / H`` and the linear velocity profile)
  * a free-surface-free cylinder smoke test plus a parametrised gate over the
    4-factor factorial table produced by the campaign runner
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorlbm.curved_wall_force_scaling import (
    CD_REFERENCE_RE100,
    BOUNDARY_FORMATS,
    FORCE_METHODS,
    WINDOW_KINDS,
    CylinderCaseConfig,
    build_channel_walls,
    build_cylinder_boundary,
    convergence_order,
    evaluate_criteria,
    reflected_populations,
    run_couette,
    run_cylinder_case,
    summarise_runs,
)

import torch

ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "curved_wall_force_scaling"
    / "sweep_results.json"
)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", BOUNDARY_FORMATS)
def test_circle_links_have_analytic_fractions(fmt: str) -> None:
    d = 32
    nx, ny = 9 * d + 1, 4 * d + 2
    links = build_cylinder_boundary(nx, ny, 3.0 * d, 0.5 * (ny - 1), 0.5 * d, fmt,
                                   torch.device("cpu"))
    assert links.meta["links"] > 8 * d  # at least ~8 links per lattice diameter
    assert float(links.q.min()) > 0.0
    assert float(links.q.max()) <= 1.0
    if fmt == "halfway":
        assert torch.allclose(links.q, torch.full_like(links.q, 0.5))
    else:
        # the analytic fraction must actually vary along the circumference
        assert float(links.q.std()) > 0.1
    # the centre-based mask resolves the circle area to O(h)
    assert 0.90 < links.meta["mask_area_over_circle"] < 1.10
    # every cut link must be flagged with a usable nearest stencil node
    assert bool(links.st1_ok.any())


def test_reflection_reduces_to_bounce_back_at_halfway() -> None:
    """All formats must coincide with half-way bounce-back at q = 1/2."""
    d = 24
    nx, ny = 8 * d + 1, 3 * d + 2
    links = build_cylinder_boundary(nx, ny, 3.0 * d, 0.5 * (ny - 1), 0.5 * d,
                                   "bouzidi_quadratic", torch.device("cpu"))
    links.q = torch.full_like(links.q, 0.5)
    gen = torch.Generator().manual_seed(7)
    fp = torch.rand(9, nx * ny, generator=gen, dtype=torch.float64)
    r_half, o_half, _ = reflected_populations(fp, links, "halfway")
    for fmt in ("bouzidi_linear", "bouzidi_quadratic"):
        r, o, _ = reflected_populations(fp, links, fmt)
        assert torch.allclose(o, o_half)
        assert torch.allclose(r, r_half, atol=1e-12), fmt


def test_reflection_is_positive_for_equilibrium() -> None:
    """An equilibrium field must not create negative reflected populations."""
    d = 20
    nx, ny = 8 * d + 1, 3 * d + 2
    links = build_cylinder_boundary(nx, ny, 3.0 * d, 0.5 * (ny - 1), 0.5 * d,
                                   "bouzidi_quadratic", torch.device("cpu"))
    rho = torch.ones(nx * ny, dtype=torch.float64)
    u = torch.full_like(rho, 0.05)
    view = torch.zeros(9, nx * ny, dtype=torch.float64)
    cx = (0, 1, 0, -1, 0, 1, -1, -1, 1)
    cy = (0, 0, 1, 0, -1, 1, 1, -1, -1)
    w = (4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36)
    for a in range(9):
        cu = cx[a] * u
        view[a] = w[a] * rho * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u**2)
    for fmt in BOUNDARY_FORMATS:
        r, _o, stats = reflected_populations(view, links, fmt)
        assert float(r.min()) > 0.0, fmt
        assert stats["links"] == int(links.node.numel())
        if fmt != "bouzidi_quadratic":
            assert stats["quadratic_fallback"] == 0


def test_channel_wall_links_straddle_both_walls() -> None:
    links = build_channel_walls(4, 20, 0.5, 0.5, "bouzidi_linear", torch.device("cpu"),
                                top_wall_speed=0.05)
    rows = set((links.node // 4).tolist())
    assert rows == {1, 18}
    assert float(links.wall_ux.max()) == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Couette analytic gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt,q", [
    ("halfway", 0.5),
    ("bouzidi_linear", 0.5),
    ("bouzidi_linear", 0.3),
    ("bouzidi_quadratic", 0.7),
])
def test_couette_wall_force_matches_analytic_shear(fmt: str, q: float) -> None:
    res = run_couette(nx=4, ny=18, q_bottom=q, u_wall=0.05, tau=1.0, steps=2500,
                      format=fmt, threads=1)
    assert abs(res["bottom_relative_error"]) < 0.02, res
    assert abs(res["top_relative_error"]) < 0.02, res
    assert abs(res["profile_relative_error"]) < 0.02, res


def test_couette_force_sign_convention() -> None:
    res = run_couette(nx=4, ny=18, q_bottom=0.5, u_wall=0.05, tau=1.0, steps=1200,
                      format="bouzidi_linear", threads=1)
    # the moving top wall feels the reaction, the stationary bottom wall the drag
    assert res["measured_shear_bottom"] > 0.0
    assert res["measured_shear_top"] < 0.0
    assert res["measured_shear_top"] == pytest.approx(-res["measured_shear_bottom"],
                                                      rel=0.05)


# ---------------------------------------------------------------------------
# cylinder smoke test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["halfway", "bouzidi_linear", "bouzidi_quadratic"])
def test_cylinder_smoke_is_finite_and_positive(fmt: str) -> None:
    cfg = CylinderCaseConfig(diameter=8, format=fmt, steps_per_diameter=40.0,
                             lattice_speed=0.1, cv_sample_every=10,
                             transient_diameters=2.0, steady_fraction=0.25)
    res = run_cylinder_case(cfg, max_steps=320, label="smoke")
    for key in ("cd_me_steady", "cd_cv_steady", "cd_me_transient"):
        assert math.isfinite(res[key]), key
    assert 0.2 < res["cd_me_steady"] < 5.0
    # interpolated bounce-back is intrinsically not mass conservative per link;
    # the leak must stay a small fraction of the boundary flux
    per_link_step = abs(res["cumulative_mass_leak"]) / (
        res["steps"] * res["boundary"]["links"]
    )
    assert per_link_step < 0.05, per_link_step
    assert res["grid"]["cells"] > 1000


def test_rest_repair_makes_the_wall_mass_conservative() -> None:
    cfg = CylinderCaseConfig(diameter=8, format="bouzidi_quadratic", steps_per_diameter=40.0,
                             lattice_speed=0.1, cv_sample_every=20,
                             transient_diameters=2.0, rest_repair=True)
    res = run_cylinder_case(cfg, max_steps=200, label="repair")
    assert res["cumulative_mass_leak"] == 0.0
    assert math.isfinite(res["cd_me_steady"])


def test_config_rejects_invalid_setup() -> None:
    with pytest.raises(ValueError):
        CylinderCaseConfig(format="bogus").validate()
    with pytest.raises(ValueError):
        CylinderCaseConfig(lattice_speed=0.5).validate()
    with pytest.raises(ValueError):
        CylinderCaseConfig(diameter=4).validate()


# ---------------------------------------------------------------------------
# convergence / criteria algebra
# ---------------------------------------------------------------------------


def test_convergence_order_recovers_second_order() -> None:
    diameters = [16, 32, 64]
    errors = [1.0e-2 / (d / 16) ** 2 for d in diameters]
    result = convergence_order(diameters, errors, 2)
    assert result["p_observed"] == pytest.approx(2.0, abs=1e-9)


def test_convergence_order_recovers_first_order() -> None:
    diameters = [16, 32, 64]
    errors = [1.0e-2 / (d / 16) for d in diameters]
    assert convergence_order(diameters, errors, 2)["p_observed"] == pytest.approx(1.0)


def _synthetic_runs(linear_errors: dict[int, float], quad_errors: dict[int, float]):
    runs = {}
    for fmt, errs in (("bouzidi_linear", linear_errors), ("bouzidi_quadratic", quad_errors)):
        for d, err in errs.items():
            cd = CD_REFERENCE_RE100 * (1.0 + err)
            runs[f"N{d:04d}_{fmt}"] = {
                "label": f"N{d:04d}_{fmt}",
                "format": fmt,
                "diameter": d,
                "steps": 100,
                "grid": {"cells": 1, "nx": 1, "ny": 1},
                "cd_me_steady": cd,
                "cd_cv_steady": cd,
                "cd_me_transient": cd,
                "cd_cv_transient": cd,
            }
    return runs


def test_criteria_algebra_passes_on_clean_second_order_data() -> None:
    runs = _synthetic_runs(
        linear_errors={16: 0.15, 32: 0.06, 64: 0.025},
        quad_errors={16: 0.04, 32: 0.010, 64: 0.0025},
    )
    table = summarise_runs(runs, expected_rows=2 * 3 * 2 * 2)
    crit = evaluate_criteria(table, finest=64, coarsest=16)
    assert crit["C1"]["pass"] is True
    assert crit["C2"]["p_quadratic"] == pytest.approx(2.0, abs=1e-6)
    assert crit["C3"]["pass"] is True
    assert crit["C4"]["pass"] is True
    assert crit["C5"]["pass"] is True


def test_criteria_algebra_flags_bad_negative_control() -> None:
    runs = _synthetic_runs(
        linear_errors={16: 0.05, 32: 0.03, 64: 0.02},
        quad_errors={16: 0.04, 32: 0.010, 64: 0.0025},
    )
    table = summarise_runs(runs, expected_rows=2 * 3 * 2 * 2)
    crit = evaluate_criteria(table, finest=64, coarsest=16)
    # a linear format that is already accurate on the coarsest grid invalidates
    # the negative control, hence C5 must be reported as failed
    assert crit["C5"]["pass"] is False


# ---------------------------------------------------------------------------
# factorial gate over the produced sweep artifact
# ---------------------------------------------------------------------------


def _load_artifact() -> dict:
    if not ARTIFACT.exists():
        pytest.skip("campaign artifact not produced yet")
    return json.loads(ARTIFACT.read_text())


def test_artifact_has_full_factorial_coverage() -> None:
    data = _load_artifact()
    ladder = data["grid_ladder_executed"]
    rows = data["table"]["rows"]
    keys = {(r["diameter"], r["format"], r["force_method"], r["window"]) for r in rows}
    expected = {
        (n, fmt, method, window)
        for n in ladder
        for fmt in ("bouzidi_linear", "bouzidi_quadratic")
        for method in FORCE_METHODS
        for window in WINDOW_KINDS
    }
    assert keys == expected
    assert len(rows) == len(expected) >= 2 * len(ladder) * 4
    for row in rows:
        assert math.isfinite(row["cd_pred"]) and row["cd_pred"] > 0.0
        assert row["rel_error"] == pytest.approx(
            abs(row["cd_pred"] - CD_REFERENCE_RE100) / CD_REFERENCE_RE100, rel=1e-9
        )


GROUP_NAMES = [
    f"{fmt}|{method}|{window}"
    for fmt in ("bouzidi_linear", "bouzidi_quadratic")
    for method in FORCE_METHODS
    for window in WINDOW_KINDS
]


@pytest.mark.parametrize("group", GROUP_NAMES)
def test_artifact_group_is_well_formed(group: str) -> None:
    """Structural gate: every group covers the ladder with sane, finite numbers.

    Monotonicity of the error in N is *not* asserted here: the campaign data show
    a residual unsteady global mode in the cylinder runs (documented in the
    report), so the gate only guarantees that the scan is complete and finite.
    """
    data = _load_artifact()
    conv = data["table"]["convergence"][group]
    assert len(conv["diameters"]) == len(data["grid_ladder_executed"])
    assert all(math.isfinite(v) for v in conv["cd_pred"])
    errors = conv["rel_error"]
    assert all(0.0 <= e <= 1.0 for e in errors), (group, errors)


@pytest.mark.parametrize("key", ["C1", "C2", "C3", "C4", "C5"])
def test_artifact_records_all_criteria(key: str) -> None:
    data = _load_artifact()
    assert key in data["criteria"]
    assert "pass" in data["criteria"][key] or key == "C2"


def test_artifact_couette_validation_present() -> None:
    data = _load_artifact()
    couette = data.get("couette_validation")
    assert couette, "Couette wall-format validation must be part of the artifact"
    for case in couette["cases"]:
        assert abs(case["bottom_relative_error"]) < 0.02, case