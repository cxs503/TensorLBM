"""Metric-geometry invariants for the lid-driven cavity Ghia comparison (m16740).

The Ghia (1982) RMSE is the acceptance metric of every 2-D cavity benchmark in
TensorLBM.  It needs an index->coordinate map, and until m16740 that map was
implicit.  These tests pin down:

* the wall planes that ``run_lid_driven_cavity``'s boundary recipe actually
  creates (bottom/left/right half-way bounce-back, lid on the Zou/He node row),
  which makes the effective cavity ``nx - 2`` wide and ``ny - 1.5`` tall while
  ``Re`` is built from ``L = nx``;
* that the general plane map reduces to the historical ``wall_offset`` map;
* that a *perfect* solver still scores a non-zero RMSE under a mismatched map,
  with the analytic prediction of :func:`ghia_metric_floor` matching the
  measured value on the same synthetic field;
* that ``run_lid_driven_cavity`` now persists the convention, the wall planes
  and the full centreline profiles next to the RMSE.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from tensorlbm.lid_driven_cavity import (
    CAVITY_BOUNDARY_RECIPE,
    GHIA_METRIC_CONVENTIONS,
    GHIA_RE100,
    GHIA_RE400,
    GHIA_RE1000,
    LidDrivenCavityConfig,
    cavity_wall_planes,
    centreline_coordinates,
    centreline_coordinates_between,
    compare_ghia,
    ghia_metric_floor,
    ghia_metric_report,
    run_lid_driven_cavity,
)


def _master(reference: dict[str, list[float]], axis: str):
    """Monotone master profile through one Ghia column (same recipe as the audit)."""
    from scipy.interpolate import PchipInterpolator

    x = np.asarray(reference["y" if axis == "u" else "x"], dtype=float)
    y = np.asarray(reference["u" if axis == "u" else "v"], dtype=float)
    order = np.argsort(x)
    f = PchipInterpolator(x[order], y[order])
    # Same extrapolation policy as tensorlbm's own floor helper: clamp outside
    # the tabulated range instead of extrapolating, so that the synthetic field
    # and the analytic prediction see identical sample values.
    return lambda q: f(np.clip(q, x[order][0], x[order][-1]))


def _perfect_field(reference: dict[str, list[float]], n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic flow field that is *exact* on the nodes the BC recipe creates."""
    planes = cavity_wall_planes(n, n)
    ys = np.asarray(centreline_coordinates_between(n, planes["y_lower"], planes["y_upper"]))
    xs = np.asarray(centreline_coordinates_between(n, planes["x_lower"], planes["x_upper"]))
    u_prof = _master(reference, "u")(ys)
    v_prof = _master(reference, "v")(xs)
    ux = np.repeat(u_prof[:, None], n, axis=1)  # constant along x
    uy = np.repeat(v_prof[None, :], n, axis=0)  # constant along y
    return (
        torch.as_tensor(ux, dtype=torch.float64),
        torch.as_tensor(uy, dtype=torch.float64),
    )


# ---------------------------------------------------------------------------
# Coordinate maps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("w", [-0.5, 0.0, 0.5])
@pytest.mark.parametrize("n", [17, 64, 128])
def test_plane_map_reduces_to_wall_offset(n: int, w: float) -> None:
    """``centreline_coordinates(n, w)`` is the plane map with ``lower=-w``, ``upper=n-1+w``."""
    legacy = centreline_coordinates(n, w)
    general = centreline_coordinates_between(n, -w, n - 1 + w)
    assert np.allclose(legacy, general, atol=1e-15)


def test_plane_map_is_monotone_and_hits_the_walls() -> None:
    coords = np.asarray(centreline_coordinates_between(129, 0.5, 128.0))
    assert np.all(np.diff(coords) > 0.0)
    assert coords[-1] == pytest.approx(1.0)  # lid node sits exactly on the lid plane
    assert coords[0] < 0.0  # bottom solid node sits below the no-slip plane


def test_plane_map_rejects_degenerate_span() -> None:
    with pytest.raises(ValueError):
        centreline_coordinates_between(16, 1.0, 1.0)
    with pytest.raises(ValueError):
        centreline_coordinates_between(1, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Wall geometry implied by the boundary recipe
# ---------------------------------------------------------------------------


def test_wall_planes_match_boundary_recipe() -> None:
    planes = cavity_wall_planes(128)
    assert planes["x_lower"] == 0.5  # bounce-back off the solid left column
    assert planes["x_upper"] == 126.5  # bounce-back off the solid right column
    assert planes["y_lower"] == 0.5  # bounce-back off the solid bottom row
    assert planes["y_upper"] == 127.0  # Zou/He moving-lid node row
    assert planes["h_x"] == pytest.approx(126.0)
    assert planes["h_y"] == pytest.approx(126.5)


def test_cavity_is_not_square_and_re_length_scale_matches_none() -> None:
    """H_y != H_x, and Re uses L = nx: both mismatches must be reported, not hidden."""
    planes = cavity_wall_planes(128)
    assert planes["aspect_ratio_h_y_over_h_x"] == pytest.approx(126.5 / 126.0)
    assert planes["aspect_ratio_h_y_over_h_x"] != pytest.approx(1.0, abs=1e-6)
    assert planes["implied_re_ratio_x"] == pytest.approx(126.0 / 128.0)
    assert planes["implied_re_ratio_y"] == pytest.approx(126.5 / 128.0)
    assert planes["re_length_scale_used"] == 128.0


def test_wall_planes_rejects_tiny_grid() -> None:
    with pytest.raises(ValueError):
        cavity_wall_planes(3)


def test_config_exposes_wall_planes_and_rejects_unknown_convention() -> None:
    cfg = LidDrivenCavityConfig(nx=64, re=100.0, n_steps=10)
    assert cfg.wall_planes["h_x"] == pytest.approx(62.0)
    assert cfg.ghia_metric_convention == "legacy_w0"
    for conv in GHIA_METRIC_CONVENTIONS:
        LidDrivenCavityConfig(nx=64, re=100.0, n_steps=10, ghia_metric_convention=conv).validate()
    with pytest.raises(ValueError):
        LidDrivenCavityConfig(
            nx=64, re=100.0, n_steps=10, ghia_metric_convention="walls_on_nodes"
        ).validate()


# ---------------------------------------------------------------------------
# Metric floors: a perfect solver is not error free under a wrong map
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reference", [GHIA_RE100, GHIA_RE400, GHIA_RE1000])
def test_perfect_field_scores_zero_under_the_recipe_map(reference) -> None:
    n = 128
    ux, uy = _perfect_field(reference, n)
    planes = cavity_wall_planes(n)
    err = compare_ghia(
        ux,
        uy,
        1.0,
        reference,
        y_planes=(planes["y_lower"], planes["y_upper"]),
        x_planes=(planes["x_lower"], planes["x_upper"]),
        exclude_walls=True,
    )
    # Floor left is pure interpolation error of the (sparse) synthetic field.
    assert err["rmse_u"] < 1e-3
    assert err["rmse_v"] < 1e-3


@pytest.mark.parametrize("reference", [GHIA_RE400, GHIA_RE1000])
def test_measured_legacy_floor_matches_analytic_prediction(reference) -> None:
    """The measured metric artefact must reproduce :func:`ghia_metric_floor`."""
    n = 128
    ux, uy = _perfect_field(reference, n)
    floor = ghia_metric_floor(reference, n)
    measured = compare_ghia(ux, uy, 1.0, reference, wall_offset=0.0)
    assert measured["rmse_u"] == pytest.approx(floor["legacy_w0"]["floor_u"], rel=0.05)
    assert measured["rmse_v"] == pytest.approx(floor["legacy_w0"]["floor_v"], rel=0.05)


@pytest.mark.parametrize("reference", [GHIA_RE100, GHIA_RE400, GHIA_RE1000])
def test_recipe_map_beats_symmetric_half_cell_map(reference) -> None:
    """m16339 suggested w=+0.5 for half-way bounce-back walls; that is a trap here.

    This cavity is *not* symmetric half-cell: the lid is a Zou/He node, so
    w=+0.5 pushes the wall rows (y,x = 0,1) outside the node span where
    ``np.interp`` clamps them.  The recipe-consistent map must score strictly
    better than both symmetric alternatives.
    """
    floor = ghia_metric_floor(reference, 128)
    for axis in ("floor_u", "floor_v"):
        assert floor["recipe_planes"][axis] < floor["legacy_w0"][axis]
        assert floor["recipe_planes"][axis] < floor["symmetric_w+0.5"][axis]


def test_metric_floor_is_not_a_high_order_small_quantity() -> None:
    """The floor only halves when h halves: it does not vanish with grid refinement."""
    coarse = ghia_metric_floor(GHIA_RE1000, 64)["legacy_w0"]["floor_u"]
    fine = ghia_metric_floor(GHIA_RE1000, 128)["legacy_w0"]["floor_u"]
    assert coarse > fine > 0.0
    assert 1.3 < coarse / fine < 3.0  # first order, not second


# ---------------------------------------------------------------------------
# Interior-row scoring and back-compatibility
# ---------------------------------------------------------------------------


def test_exclude_walls_drops_the_two_wall_rows() -> None:
    reference = GHIA_RE100
    ux, uy = _perfect_field(reference, 32)
    full = compare_ghia(ux, uy, 1.0, reference, wall_offset=0.5)
    interior = compare_ghia(ux, uy, 1.0, reference, wall_offset=0.5, exclude_walls=True)
    assert full["n_scored_u"] == len(reference["y"])
    assert interior["n_scored_u"] == len(reference["y"]) - 2
    assert interior["n_scored_v"] == len(reference["x"]) - 2


def test_default_call_is_unchanged() -> None:
    """Adding keyword-only args must not move the historical default score."""
    reference = GHIA_RE400
    ux, uy = _perfect_field(reference, 64)
    default = compare_ghia(ux, uy, 1.0, reference)
    explicit = compare_ghia(ux, uy, 1.0, reference, wall_offset=0.0)
    assert default["rmse_u"] == explicit["rmse_u"]
    assert default["rmse_v"] == explicit["rmse_v"]


def test_diverged_field_never_passes() -> None:
    reference = GHIA_RE100
    ux, uy = _perfect_field(reference, 32)
    ux = ux.clone()
    ux[3, ux.shape[1] // 2] = float("nan")  # on the vertical centreline
    report = ghia_metric_report(ux, uy, 1.0, reference, include_centreline=False)
    for scores in report["scores"].values():
        assert math.isnan(scores["rmse_u"])
        assert math.isnan(scores["interior_rmse_u"])


# ---------------------------------------------------------------------------
# Run metadata carries the convention, the geometry and the field
# ---------------------------------------------------------------------------


def test_run_metadata_persists_metric_and_centreline(tmp_path) -> None:
    cfg = LidDrivenCavityConfig(
        nx=32,
        re=100.0,
        n_steps=6,
        output_interval=3,
        output_root=tmp_path,
        run_name="metric_meta",
        overwrite=True,
    )
    run_dir = run_lid_driven_cavity(cfg)
    meta = json.loads((run_dir / "run_metadata.json").read_text())

    assert meta["ghia_metric_convention"] == "legacy_w0"
    metric = meta["ghia_metric"]
    assert metric["boundary_recipe"] == CAVITY_BOUNDARY_RECIPE
    assert metric["wall_planes"]["h_x"] == pytest.approx(30.0)
    assert set(metric["scores"]) == set(GHIA_METRIC_CONVENTIONS)
    assert set(metric["analytic_floor"]) == set(GHIA_METRIC_CONVENTIONS)

    # Headline RMSE is the selected convention, and the field is recoverable.
    assert meta["ghia_errors"]["rmse_u"] == metric["scores"]["legacy_w0"]["rmse_u"]
    centre = metric["centreline"]
    assert len(centre["y"]) == len(centre["u"]) == 32
    assert len(centre["x"]) == len(centre["v"]) == 32
    assert centre["coordinates"] == "recipe_planes"

    # The written CSV uses the recipe wall planes, not linspace(0, 1, n).
    planes = cavity_wall_planes(32)
    csv_y = [float(row.split(",")[0]) for row in (run_dir / "ghia_comparison.csv").read_text().splitlines()[1:]]
    expected = centreline_coordinates_between(32, planes["y_lower"], planes["y_upper"])
    assert np.allclose(csv_y, expected, atol=1e-6)


def test_run_metadata_honours_selected_convention(tmp_path) -> None:
    cfg = LidDrivenCavityConfig(
        nx=32,
        re=100.0,
        n_steps=6,
        output_interval=3,
        output_root=tmp_path,
        run_name="metric_recipe",
        overwrite=True,
        ghia_metric_convention="recipe_planes",
    )
    run_dir = run_lid_driven_cavity(cfg)
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    metric = meta["ghia_metric"]
    assert meta["ghia_errors"]["rmse_u"] == metric["scores"]["recipe_planes"]["rmse_u"]
    assert meta["ghia_errors"]["rmse_v"] == metric["scores"]["recipe_planes"]["rmse_v"]