"""Verifier-of-verifier tests for the Ghia (1982) cavity metric.

Audit finding (m16339): :func:`tensorlbm.lid_driven_cavity.compare_ghia` is the
metric behind every cavity acceptance claim in this repo, yet

1. the index -> physical-coordinate map was hard-wired to ``j/(n-1)`` with no
   way to state which boundary convention the run used, and the reported RMSE
   moves by ~0.2-1.1 % (absolute, normalised by ``u_lid``) between the defensible
   maps -- a material part of the 3 % / 5 % gates;
2. the endpoint ``app/backend/routers/benchmarks.py`` called it with three
   positional arguments (``compare_ghia(ux, uy, ref)``), i.e. ``/benchmarks/ghia``
   raised ``TypeError`` on every use.

These tests pin the repaired contract: explicit ``wall_offset``, quantitative
convention sensitivity, fail-safe handling of NaN, verified orientation, and an
AST guard against call sites that drop an argument.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tensorlbm.lid_driven_cavity import (
    GHIA_RE100,
    GHIA_RE400,
    GHIA_RE1000,
    centreline_coordinates,
    compare_ghia,
    ghia_coordinate_sensitivity,
)

REPO = Path(__file__).resolve().parents[1]
TABLES = {"Re100": GHIA_RE100, "Re400": GHIA_RE400, "Re1000": GHIA_RE1000}


def _field_from_table(reference: dict, n: int, wall_offset: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """A 'perfect solver' field: node j holds the reference value at y_j."""
    from scipy.interpolate import PchipInterpolator

    y = np.asarray(reference["y"])[::-1]
    u = np.asarray(reference["u"])[::-1]
    x = np.asarray(reference["x"])
    v = np.asarray(reference["v"])
    fu = PchipInterpolator(y, u)
    fv = PchipInterpolator(x, v)
    coords = np.asarray(centreline_coordinates(n, wall_offset))
    ux = torch.zeros((n, n), dtype=torch.float64)
    ux[:, n // 2] = torch.from_numpy(fu(coords))
    uy = torch.zeros((n, n), dtype=torch.float64)
    uy[n // 2, :] = torch.from_numpy(fv(coords))
    return ux, uy


@pytest.mark.unit
def test_reference_tables_keep_documented_ordering() -> None:
    """Guards the trap: 'y' is descending, 'x' ascending, both length 17."""
    for name, ref in TABLES.items():
        assert len(ref["y"]) == len(ref["u"]) == len(ref["x"]) == len(ref["v"]) == 17, name
        assert ref["y"][0] == pytest.approx(1.0) and ref["y"][-1] == pytest.approx(0.0), name
        assert ref["x"][0] == pytest.approx(0.0) and ref["x"][-1] == pytest.approx(1.0), name
        assert all(b < a for a, b in zip(ref["y"], ref["y"][1:])), f"{name}: y must be descending"
        assert all(b > a for a, b in zip(ref["x"], ref["x"][1:])), f"{name}: x must be ascending"
        # lid and wall values are exact by construction
        assert ref["u"][0] == pytest.approx(1.0) and ref["u"][-1] == pytest.approx(0.0)
        assert ref["v"][0] == pytest.approx(0.0) and ref["v"][-1] == pytest.approx(0.0)


@pytest.mark.unit
def test_coordinate_maps_are_what_they_claim() -> None:
    n = 8
    legacy = centreline_coordinates(n, 0.0)
    assert legacy[0] == pytest.approx(0.0) and legacy[-1] == pytest.approx(1.0)
    assert legacy == pytest.approx([j / (n - 1) for j in range(n)])

    centred = centreline_coordinates(n, 0.5)
    assert centred == pytest.approx([(j + 0.5) / n for j in range(n)])
    assert centred[0] > 0.0 and centred[-1] < 1.0  # wall planes outside the lattice

    inside = centreline_coordinates(n, -0.5)
    assert inside == pytest.approx([(j - 0.5) / (n - 2) for j in range(n)])

    assert centreline_coordinates(2, 0.0) == [0.0, 1.0]
    with pytest.raises(ValueError):
        centreline_coordinates(1, 0.0)
    with pytest.raises(ValueError):
        centreline_coordinates(4, -2.0)  # degenerate span


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(TABLES))
def test_perfect_field_scores_zero_only_under_its_own_convention(name: str) -> None:
    """A field that reproduces the table exactly at its own nodes scores ~0
    under its own convention; any other convention adds a floor."""
    ref = TABLES[name]
    n = 128
    # legacy-consistent field: node j holds the tabulated value at y = j/(n-1)
    ux, uy = _field_from_table(ref, n, wall_offset=0.0)
    own = compare_ghia(ux, uy, 1.0, ref, wall_offset=0.0)
    assert own["rmse_u"] < 2e-3, own          # interpolation error only
    assert own["rmse_v"] < 2e-3, own

    other = compare_ghia(ux, uy, 1.0, ref, wall_offset=0.5)
    predicted = ghia_coordinate_sensitivity(ref, n)["+0.0"]["bias_u"]
    assert other["rmse_u"] > 5.0 * own["rmse_u"]
    assert other["rmse_u"] > predicted         # convention bias + endpoint clamping


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(TABLES))
def test_cell_centred_field_interior_rows_are_exact(name: str) -> None:
    """Documented caveat: with w=+0.5 the wall rows of the table lie outside the
    node span and ``np.interp`` clamps them, so the whole residual of a perfect
    solver sits on those two rows (and is *not* solver error)."""
    ref = TABLES[name]
    n, w = 128, 0.5
    ux, _uy = _field_from_table(ref, n, wall_offset=w)
    coords = np.asarray(centreline_coordinates(n, w))
    u_field = ux[:, n // 2].numpy()
    ref_y = np.asarray(ref["y"])
    ref_u = np.asarray(ref["u"])
    residual = np.abs(ref_u - np.interp(ref_y, coords, u_field))
    interior = np.abs(ref_y - 0.5) < 0.499           # drop the y=0 and y=1 rows
    interior_max = float(np.max(residual[interior]))
    wall_max = float(np.max(residual[~interior]))
    # interior residual = linear-interpolation error of the metric only
    assert interior_max < 1.5e-2, residual
    assert wall_max > 2.0 * interior_max, residual  # the wall rows dominate
    assert compare_ghia(ux, _uy, 1.0, ref, wall_offset=w)["rmse_u"] < 3e-2


@pytest.mark.unit
def test_convention_bias_shrinks_with_resolution_and_is_material() -> None:
    """The bias term is first order in h, i.e. it does NOT vanish at benchmark
    resolutions: at 192^2/256^2 it is a sizeable share of a 3 % gate."""
    ref = GHIA_RE1000
    b64 = ghia_coordinate_sensitivity(ref, 64)["+0.0"]["bias_u"]
    b256 = ghia_coordinate_sensitivity(ref, 256)["+0.0"]["bias_u"]
    b384 = ghia_coordinate_sensitivity(ref, 384)["+0.0"]["bias_u"]
    assert b64 > b256 > b384
    assert b384 > 1.5 * b256 - 1.0  # slow (≈1/n) decay, not quadratic
    assert 0.003 < b256 < 0.02, b256
    # outward and inward conventions are not mirror images of each other
    spread = ghia_coordinate_sensitivity(ref, 256)
    assert spread["+0.5"]["bias_u"] < spread["+0.0"]["bias_u"] < spread["-0.5"]["bias_u"]


@pytest.mark.unit
def test_nan_field_is_reported_as_nan_never_as_pass() -> None:
    ref = GHIA_RE100
    n = 64
    ux, uy = _field_from_table(ref, n)
    ux[10, :] = float("nan")
    res = compare_ghia(ux, uy, 1.0, ref, wall_offset=0.5)
    assert math.isnan(res["rmse_u"])
    assert not (res["rmse_u"] < 0.03)  # a gate written as ``err < tol`` fails safe


@pytest.mark.unit
def test_transposed_field_is_penalised() -> None:
    """Orientation guard: u belongs to the vertical centreline, v to the horizontal."""
    ref = GHIA_RE1000
    n = 128
    ux, uy = _field_from_table(ref, n, wall_offset=0.5)
    good = compare_ghia(ux, uy, 1.0, ref, wall_offset=0.5)
    bad = compare_ghia(ux.T, uy.T, 1.0, ref, wall_offset=0.5)
    assert bad["rmse_u"] > 10.0 * good["rmse_u"]


@pytest.mark.unit
def test_invalid_inputs_raise() -> None:
    ref = GHIA_RE100
    ux, uy = _field_from_table(ref, 32)
    with pytest.raises(ValueError):
        compare_ghia(ux, uy, 0.0, ref)            # normalising by zero
    with pytest.raises(ValueError):
        compare_ghia(ux[0], uy[0], 1.0, ref)      # 1-D field
    with pytest.raises(ValueError):
        compare_ghia(torch.zeros((32, 16)), torch.zeros((16, 32)), 1.0, ref)  # shape mismatch
    with pytest.raises(ValueError):
        compare_ghia(ux, uy, 1.0, {"y": [0.0], "u": [0.0]})


@pytest.mark.unit
def test_all_compare_ghia_call_sites_pass_four_arguments() -> None:
    """m16339 defect: ``/benchmarks/ghia`` called compare_ghia(ux, uy, ref)."""
    targets = list((REPO / "app").rglob("*.py")) + list((REPO / "src").rglob("*.py")) + list(
        (REPO / "scripts").rglob("*.py")
    )
    checked = 0
    for path in targets:
        src = path.read_text()
        if "def compare_ghia" in src:  # module-local namesake (verify_n07.py)
            continue
        try:
            tree = ast.parse(src, filename=str(path))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if name != "compare_ghia":
                continue
            n_pos = len(node.args)
            n_kw = len(node.keywords)
            # first arg is the function itself when it is a bare Name call
            checked += 1
            assert n_pos + n_kw >= 4, f"{path}:{node.lineno} compare_ghia called with {n_pos} positional + {n_kw} keyword args"
    assert checked >= 2, f"expected at least two compare_ghia call sites, found {checked}"