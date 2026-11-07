"""Tests for :mod:`tensorlbm.ship_wind_aero` (node IN38-N04).

Fast, deterministic checks of the catalogue geometry, the engineering empirical
models, the hybrid engine and the wind-rose assessment, plus one deliberately
small LBM smoke test (a few thousand cells, ~40 steps) that exercises the
force integration end to end.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from tensorlbm.ship_wind_aero import (
    CD_HANDBOOK, HybridWindEngine, blendermann_coefficients, blendermann_style,
    bulk_carrier_superstructure, calm_water_resistance_estimate, catalogue,
    complexity_index, component_buildup, container_ship_superstructure,
    evaluate_acceptance, export_stl, fit_blendermann_style, projected_areas,
    shadow_factor, sign_conventions, synthetic_wind_rose, tanker_superstructure,
    validate_geometry, wind_rose_annual_resistance,
)

SHIPS = (container_ship_superstructure, tanker_superstructure,
         bulk_carrier_superstructure)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", SHIPS)
def test_geometry_validates(factory):
    ship = factory()
    res = validate_geometry(ship)
    assert res["ok"], res["problems"]
    assert res["hull_checks_ok"]
    assert not res["problems"]["overlap"]
    assert not res["problems"]["symmetry"]


@pytest.mark.parametrize("factory", SHIPS)
def test_projection_areas_are_physical(factory):
    ship = factory()
    ref = projected_areas(ship)
    assert 200.0 < ref["area_frontal_m2"] < 2500.0
    assert 200.0 < ref["area_lateral_m2"] < 4000.0
    assert ref["area_lateral_m2"] > ref["area_frontal_m2"]
    # bounding-box sanity: no projection can exceed the enclosing box
    lx, ly, lz = ship.extent
    assert ref["area_frontal_m2"] <= ly * lz + 1e-6
    assert ref["area_lateral_m2"] <= lx * lz + 1e-6
    assert ref["area_plan_m2"] <= lx * ly + 1e-6


@pytest.mark.parametrize("factory", SHIPS)
def test_hull_particulars_consistent(factory):
    hull = factory().hull
    assert hull.validate() == []
    assert 0.45 <= hull.block_coefficient <= 0.90
    assert 4.0 <= hull.length_pp_m / hull.breadth_m <= 9.0
    # displacement must be a small fraction of the L^3 envelope
    assert 0.001 < hull.displacement_m3 / hull.length_pp_m ** 3 < 0.02


def test_geometry_validation_detects_overlap():
    ship = tanker_superstructure()
    bad = ship.boxes[0].__class__(0.0, 30.0, -5.0, 5.0, 0.0, 10.0, "intruder")
    ship.boxes = list(ship.boxes) + [bad]
    res = validate_geometry(ship)
    assert not res["ok"]
    assert res["problems"]["overlap"]


def test_voxel_area_check_flags_mismatch():
    ship = container_ship_superstructure()
    ref = projected_areas(ship)
    good = validate_geometry(ship, voxel_areas=ref)
    assert good["ok"]
    bad = validate_geometry(ship, voxel_areas={
        "area_frontal_m2": 0.5 * ref["area_frontal_m2"],
        "area_lateral_m2": ref["area_lateral_m2"]})
    assert not bad["ok"]
    assert bad["problems"]["voxel"]


def test_stl_export(tmp_path):
    ship = bulk_carrier_superstructure()
    path = export_stl(ship, str(tmp_path / "bulker.stl"))
    text = open(path, encoding="ascii").read()
    assert text.startswith("solid bulker")
    assert text.rstrip().endswith("endsolid bulker")
    assert text.count("facet normal") == 12 * len(ship.boxes)
    assert text.count("vertex") == 36 * len(ship.boxes)


# ---------------------------------------------------------------------------
# Empirical models
# ---------------------------------------------------------------------------

def test_shadow_factors_bounded():
    ship = container_ship_superstructure()
    for angle in (0.0, 45.0, 90.0, 180.0):
        f = shadow_factor(ship.boxes, math.radians(angle), kappa=0.5)
        assert len(f) == len(ship.boxes)
        assert all(0.0 < v <= 1.0 for v in f)
    # a head wind: the stow blocks stand in front of the house, so the house
    # must be partially shadowed
    f0 = shadow_factor(ship.boxes, 0.0, kappa=0.5)
    names = [b.name for b in ship.boxes]
    assert f0[names.index("deck_house")] < 1.0


def test_component_buildup_symmetry_and_signs():
    ship = tanker_superstructure()
    res = component_buildup(ship, angles=(0, 90, 180))
    rows = {r["angle_deg"]: r for r in res["rows"]}
    assert rows[0.0]["C_X"] > 0.3            # head wind retards
    assert abs(rows[0.0]["C_Y"]) < 1e-9      # no side force head-on
    assert rows[90.0]["C_Y"] > 0.3           # beam wind pushes to starboard +y
    assert abs(rows[90.0]["C_X"]) < 0.05 * rows[0.0]["C_X"]
    assert rows[180.0]["C_X"] < 0             # following wind gives thrust
    assert abs(rows[180.0]["C_X"] + rows[0.0]["C_X"]) < 0.02 * rows[0.0]["C_X"]


def test_component_buildup_uses_handbook_coefficients():
    ship = bulk_carrier_superstructure()
    res = component_buildup(ship, angles=(0,))
    # the a-priori estimate must be of order unity for a bluff superstructure
    assert 0.4 < res["rows"][0]["C_X"] < 2.5
    assert set(CD_HANDBOOK.values())
    for b in ship.boxes:
        assert b.cd in CD_HANDBOOK.values()


def test_blendermann_shape_is_even_odd():
    kw = dict(cx0=0.9, cx90=0.15, cy0=1.1, p=1.5, q=1.3, cx180=-0.7)
    for a in (0.0, 25.0, 60.0, 90.0, 130.0):
        # even in theta for C_X, odd in theta for C_Y (centreline symmetry)
        assert blendermann_style(a, **kw)["C_X"] == pytest.approx(
            blendermann_style(-a, **kw)["C_X"], abs=1e-12)
        assert blendermann_style(a, **kw)["C_Y"] == pytest.approx(
            -blendermann_style(-a, **kw)["C_Y"], abs=1e-12)
    assert blendermann_style(0.0, **kw)["C_X"] == pytest.approx(0.9)
    assert blendermann_style(180.0, **kw)["C_X"] == pytest.approx(-0.7)
    assert blendermann_style(90.0, **kw)["C_X"] == pytest.approx(0.15)
    assert abs(blendermann_style(90.0, **kw)["C_Y"]) == pytest.approx(1.1)


def test_fit_recovers_synthetic_correlation():
    kw = dict(cx0=1.0, cx90=0.2, cy0=1.2, p=1.6, q=1.4, cx180=-0.8)
    samples = []
    for a in (0, 30, 60, 90, 120, 150, 180):
        c = blendermann_style(a, **kw)
        samples.append({"angle_deg": float(a), "C_X": c["C_X"],
                        "C_Y": c["C_Y"], "C_K": 0.5 * c["C_X"],
                        "C_N": 1.5 * c["C_Y"]})
    fit = fit_blendermann_style(samples)
    assert fit["params"]["cx0"] == pytest.approx(1.0, rel=0.02)
    assert fit["params"]["cx90"] == pytest.approx(0.2, abs=0.02)
    assert fit["params"]["cy0"] == pytest.approx(1.2, rel=0.02)
    assert fit["params"]["cx180"] == pytest.approx(-0.8, abs=0.02)
    assert fit["params"]["p"] == pytest.approx(1.6, abs=0.15)
    assert fit["max_rel_err_C_X"] < 0.02
    assert fit["max_rel_err_C_Y"] < 0.02


# ---------------------------------------------------------------------------
# Complexity metric and hybrid routing
# ---------------------------------------------------------------------------

def _synthetic_db(cx0=1.0, cx90=0.2, cy0=1.2):
    db = []
    for a in (0, 30, 60, 90, 120, 150, 180):
        c = blendermann_style(a, cx0=cx0, cx90=cx90, cy0=cy0, p=1.5, q=1.5)
        db.append({"angle_deg": float(a), "C_X": c["C_X"], "C_Y": c["C_Y"],
                   "C_K": 0.4 * c["C_X"], "C_N": 0.8 * c["C_Y"]})
    return db


def test_complexity_separates_ship_types():
    ships = catalogue()
    ci = {k: complexity_index(v) for k, v in ships.items()}
    # the container ship carries a tall deck stow ahead of the house
    assert ci["container"]["distributed_blockage"] > 0.5
    # the tanker and bulk carrier are deckhouse dominated
    assert ci["tanker"]["distributed_blockage"] < 0.10
    assert ci["bulker"]["distributed_blockage"] < ci["container"]["distributed_blockage"]
    assert ci["container"]["n_boxes"] >= ci["tanker"]["n_boxes"]


def test_engine_routes_by_complexity():
    ships = catalogue()
    db = _synthetic_db()
    routes = {k: HybridWindEngine(v, db).routing_report()["branch"]
              for k, v in ships.items()}
    assert routes["container"] == "lbm"
    assert routes["tanker"] == "empirical"
    assert routes["bulker"] in ("blend", "empirical")
    assert set(routes.values()) >= {"lbm", "empirical"}


def test_engine_interpolates_exactly_at_samples():
    ship = tanker_superstructure()
    db = _synthetic_db()
    eng = HybridWindEngine(ship, db, route="lbm")
    for rec in db:
        got = eng.coefficients(rec["angle_deg"])
        assert got["C_X"] == pytest.approx(rec["C_X"], rel=1e-6, abs=1e-9)
        assert got["C_Y"] == pytest.approx(rec["C_Y"], rel=1e-6, abs=1e-9)


def test_engine_table_is_periodic_and_antisymmetric():
    ship = tanker_superstructure()
    eng = HybridWindEngine(ship, _synthetic_db(), route="lbm")
    for a in (0.0, 37.0, 90.0, 143.0, 200.0, 289.0):
        c0 = eng.coefficients(a)
        c1 = eng.coefficients(a + 360.0)
        assert c0["C_X"] == pytest.approx(c1["C_X"], abs=1e-9)
        assert c0["C_Y"] == pytest.approx(c1["C_Y"], abs=1e-9)
        mirror = eng.coefficients(360.0 - a)
        assert mirror["C_X"] == pytest.approx(c0["C_X"], abs=1e-6)
        assert mirror["C_Y"] == pytest.approx(-c0["C_Y"], abs=1e-6)


def test_engine_blend_is_weighted_average():
    ship = bulk_carrier_superstructure()
    db = _synthetic_db(cy0=5.0)
    eng = HybridWindEngine(ship, db, route="auto")
    w = eng.lbm_weight()
    assert 0.0 < w < 1.0
    a = 90.0
    got = eng.coefficients(a)
    emp = blendermann_coefficients(eng.correlation, a)
    lbm = eng._lbm_eval(a)
    assert got["C_X"] == pytest.approx(w * lbm["C_X"] + (1 - w) * emp["C_X"],
                                       rel=1e-6)
    assert got["method"] == "blend"


def test_engine_force_matches_coefficient_definition():
    ship = container_ship_superstructure()
    ship.reference = projected_areas(ship)
    eng = HybridWindEngine(ship, _synthetic_db(), route="empirical")
    f = eng.force(0.0, 15.0)
    ref = ship.reference
    q = 0.5 * 1.225 * 15.0 ** 2
    assert f["q_Pa"] == pytest.approx(q, rel=1e-9)
    assert f["X_N"] == pytest.approx(f["coefficients"]["C_X"] * q
                                     * ref["area_frontal_m2"], rel=1e-9)
    assert f["Y_N"] == pytest.approx(f["coefficients"]["C_Y"] * q
                                     * ref["area_lateral_m2"], rel=1e-9)


def test_engine_without_database_is_empirical_only():
    ship = tanker_superstructure()
    db = _synthetic_db()
    corr = fit_blendermann_style(db)
    eng = HybridWindEngine(ship, [], correlation=corr)
    assert eng.routing_report()["branch"] == "empirical"
    assert eng.coefficients(45.0)["method"] == "empirical"
    with pytest.raises(RuntimeError):
        HybridWindEngine(ship, []).coefficients(45.0)


def test_sign_conventions_side_force_check():
    db = _synthetic_db()
    res = sign_conventions(db)
    assert len(res["checks"]) == 2              # theta = 0 and 180
    assert res["max_residual_side_force_rel"] < 1e-9
    assert res["max_residual_yaw_moment_rel"] < 1e-9
    assert res["C_X_at_0"] > 0
    assert res["C_X_at_180"] < 0


# ---------------------------------------------------------------------------
# Wind rose / annual assessment
# ---------------------------------------------------------------------------

def test_wind_rose_is_normalised_and_physical():
    rose = synthetic_wind_rose()
    assert sum(rose["probability"]) == pytest.approx(1.0, abs=1e-12)
    assert len(rose["sectors_deg_from"]) == len(rose["probability"])
    assert 5.0 < rose["mean_wind_speed_ms"] < 12.0


def test_annual_resistance_positive_headwind_and_thrust_downwind():
    ship = tanker_superstructure()
    ship.reference = projected_areas(ship)
    eng = HybridWindEngine(ship, _synthetic_db(), route="empirical")
    rose = synthetic_wind_rose()
    # the representative rose has prevailing westerlies (sector 270 deg), so a
    # ship steering into them (heading 270) must see more wind resistance than
    # one running with them (heading 90)
    head = wind_rose_annual_resistance(eng, rose, ship_speed_ms=4.0,
                                       heading_deg=270.0)
    down = wind_rose_annual_resistance(eng, rose, ship_speed_ms=4.0,
                                       heading_deg=90.0)
    assert head["mean_wind_added_resistance_N"] > 0
    assert head["mean_wind_added_resistance_N"] > down["mean_wind_added_resistance_N"]
    assert down["thrust_probability"] > head["thrust_probability"]
    for res in (head, down):
        assert 0.0 <= res["thrust_probability"] <= 1.0
        assert res["mean_wind_added_resistance_positive_N"] >= 0.0
        assert res["frontal_area_used_m2"] == pytest.approx(
            ship.reference["area_frontal_m2"])


def test_annual_assessment_with_extra_area_scales_up():
    ship = container_ship_superstructure()
    ship.reference = projected_areas(ship)
    eng = HybridWindEngine(ship, _synthetic_db(), route="empirical")
    rose = synthetic_wind_rose()
    a = wind_rose_annual_resistance(eng, rose, ship_speed_ms=6.0)
    b = wind_rose_annual_resistance(eng, rose, ship_speed_ms=6.0,
                                    extra_frontal_area_m2=1000.0)
    assert b["mean_wind_added_resistance_N"] > a["mean_wind_added_resistance_N"]
    ratio = (b["frontal_area_used_m2"] / a["frontal_area_used_m2"])
    assert b["mean_wind_added_resistance_N"] / a["mean_wind_added_resistance_N"] == \
        pytest.approx(ratio, rel=0.02)


def test_calm_water_estimate_is_plausible():
    for factory in SHIPS:
        hull = factory().hull
        r = calm_water_resistance_estimate(hull, hull.design_speed_ms)
        assert 1e5 < r["R_calm_water_N"] < 1e7
        assert 1e8 < r["Re"] < 1e10
        assert 0.0010 < r["C_F"] < 0.0030


# ---------------------------------------------------------------------------
# Acceptance helper
# ---------------------------------------------------------------------------

def test_evaluate_acceptance_verdicts():
    res = evaluate_acceptance([
        {"id": "a", "description": "pass", "value": 0.05, "threshold": 0.10,
         "operator": "<=", "hard": True},
        {"id": "b", "description": "fail", "value": 0.30, "threshold": 0.10,
         "operator": "<=", "hard": False},
        {"id": "c", "description": "n/a", "value": None, "threshold": 1},
    ])
    verdicts = {r["id"]: r["verdict"] for r in res["criteria"]}
    assert verdicts == {"a": "PASS", "b": "FAIL", "c": "NOT_EVALUATED"}
    assert res["all_hard_passed"] is True
    assert res["n_hard_passed"] == 1 and res["n_soft_passed"] == 0
    assert res["n_not_evaluated"] == 1        # criteria without a value do not gate


# ---------------------------------------------------------------------------
# LBM integration smoke test (small, a few seconds)
# ---------------------------------------------------------------------------

def test_lbm_force_integration_smoke():
    torch = pytest.importorskip("torch")
    from tensorlbm.wind_load_lng import LatticeUnits, WindLoadSolver, Box, voxelise

    torch.set_num_threads(1)
    box = Box(0.0, 16.0, -8.0, 8.0, 0.0, 12.0, "block")
    geom = voxelise([box], 4.0, 0.0, margin_upstream=24.0,
                    margin_downstream=40.0, margin_side=32.0, margin_top=32.0)
    units = LatticeUnits.from_velocity(dx=4.0, u_phys=15.0, u_lat=0.15)
    solver = WindLoadSolver(geom, units,
                            ref_areas=(geom.area_frontal, geom.area_lateral),
                            ref_breadth_height=(geom.breadth, geom.height))
    coeff = solver.run(40, average_from=0)
    assert math.isfinite(coeff["C_X"])
    assert coeff["C_X"] > 0.0                     # drag on a block
    assert abs(coeff["C_Y"]) < 1e-6               # symmetric head wind
    assert coeff["C_X_std"] >= 0.0
    assert coeff["n_samples"] == 40


# ---------------------------------------------------------------------------
# Built artifacts (skipped until the analysis has been run)
# ---------------------------------------------------------------------------

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "artifacts", "in38_n04")


@pytest.mark.skipif(not os.path.exists(os.path.join(ART, "acceptance.json")),
                    reason="IN38-N04 artifacts not built yet")
def test_acceptance_artifact_is_self_consistent():
    acc = json.load(open(os.path.join(ART, "acceptance.json"), encoding="utf-8"))
    assert acc["schema"] == "tensorlbm.wind_acceptance/v1"
    for row in acc["criteria"]:
        if row["value"] is not None:
            assert row["verdict"] in ("PASS", "FAIL")
    assert acc["n_hard"] >= 5


@pytest.mark.skipif(not os.path.exists(os.path.join(ART, "wind_database.json")),
                    reason="IN38-N04 artifacts not built yet")
def test_database_artifact_is_consistent():
    db = json.load(open(os.path.join(ART, "wind_database.json"), encoding="utf-8"))
    assert set(db["database"]) == {"container", "tanker", "bulker"}
    # coverage is reported explicitly; every stored case must be finite
    assert db["coverage"]["n_cases"] == sum(len(v) for v in db["database"].values())
    for ship, rows in db["database"].items():
        for r in rows:
            for k in ("C_X", "C_Y", "C_K", "C_N"):
                assert math.isfinite(r[k])
    # a ship only enters the cross-validation with a complete 7-angle polar
    for ship in db["cross_validation"]["hybrid_auto"]:
        assert len(db["database"][ship]) == 7