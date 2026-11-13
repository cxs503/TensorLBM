"""Cross-validation tests for the LIFE ``geometry.config`` convention.

Anchor: ``XVAL-LBM-COMPILETIME-PARAMS-PLUS-RUNTIME-GEOMETRY``.

Every assertion is tied to a line of the source of the practice being
cross-validated (joconnor22/LIFE, GPL-3.0 — format semantics only, no code
copied): ``src/Objects.cpp:geometryReadIn`` (l.324-472),
``src/IBMBody.cpp`` (l.47-130), ``inc/params.h``,
``src/Objects.cpp:removeOverlapMarkers``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.runtime_geometry import (
    CaseParams,
    CircleBody,
    FilamentBody,
    Scene,
    SimulationSetup,
    parse_geometry_config,
    read_geometry_config,
)

# Verbatim copy of LIFE's own in-tree configuration (2 body lines).
LIFE_FILE = """\
# This is the geometry configuration file. Each specific body is input via
# each line within this file. CIRCLE NUMBER STARTID CX CY SPACING_X SPACING_Y RADIUS
CIRCLE 1 0 0.2 0.2 0.0 0.0 0.05
FILAMENT 1 1 0.25 0.2 0.0 0.0 0.35 0.02 0.0 FLEXIBLE 20 CLAMPED 10000.0 1.4e6
"""


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def test_life_in_tree_file_parses_to_two_bodies():
    scene = parse_geometry_config(LIFE_FILE)
    assert [b.keyword for b in scene.bodies] == ["CIRCLE", "FILAMENT"]
    assert scene.ids() == [0, 1]
    circle, filament = scene.bodies
    assert circle.centre == pytest.approx((0.2, 0.2)) and circle.radius == 0.05
    assert filament.length == 0.35 and filament.height == 0.02
    assert filament.flexible is True
    assert filament.n_elements == 20 and filament.bc == "CLAMPED"
    assert filament.rho == 10_000.0 and filament.youngs_modulus == 1.4e6


def test_array_spacing_expands_positions_and_ids():
    """LIFE: ``pos = start + i*space``, ``ID = start_id + i``."""
    scene = parse_geometry_config("CIRCLE 3 4 0.1 0.2 0.05 0.01 0.02\n")
    assert scene.ids() == [4, 5, 6]
    expected = [(0.1, 0.2), (0.15, 0.21), (0.2, 0.22)]
    for body, exp in zip(scene.bodies, expected):
        assert body.centre == pytest.approx(exp)


def test_zero_spacing_is_a_single_body():
    scene = parse_geometry_config("CIRCLE 1 0 0.2 0.2 0.0 0.0 0.05\n")
    assert len(scene) == 1


def test_trailing_comments_and_blank_lines_are_tolerated():
    text = "\n# header\n\nCIRCLE 1 0 0.2 0.2 0 0 0.05  # one cylinder\n\n"
    scene = parse_geometry_config(text)
    assert len(scene) == 1


@pytest.mark.parametrize(
    "bad",
    [
        "TRIANGLE 1 0 0.2 0.2 0 0 0.05",  # unknown keyword (LIFE silently loops)
        "CIRCLE 1 0 0.2 0.2 0 0",  # missing RADIUS
        "CIRCLE 1 0 0.2 0.2 0 0 0.05 0.01",  # extra column
        "FILAMENT 1 1 0.25 0.2 0 0 0.35 0.02 0 BOUNCY 20 CLAMPED 1e4 1.4e6",
        "FILAMENT 1 1 0.25 0.2 0 0 0.35 0.02 0 FLEXIBLE 20 CLAMPED 1e4",  # short
        "CIRCLE 0 0 0.2 0.2 0 0 0.05",  # NUMBER < 1
        "CIRCLE 1 0 a b 0 0 0.05",  # non-numeric
    ],
)
def test_malformed_lines_raise(bad):
    with pytest.raises(ValueError):
        parse_geometry_config(bad)


def test_duplicate_ids_rejected():
    """LIFE errors on duplicate IDs (Objects.cpp, post-read check)."""
    with pytest.raises(ValueError, match="duplicate body id"):
        parse_geometry_config(
            "CIRCLE 1 0 0.2 0.2 0 0 0.05\nCIRCLE 1 0 0.3 0.2 0 0 0.05\n"
        )


def test_file_and_directory_readers(tmp_path):
    case = tmp_path / "case"
    (case / "input").mkdir(parents=True)
    cfg = case / "input" / "geometry.config"
    cfg.write_text("CIRCLE 1 0 0.2 0.2 0 0 0.05\n")
    assert len(read_geometry_config(case)) == 1  # directory → input/geometry.config
    assert len(read_geometry_config(cfg)) == 1
    with pytest.raises(FileNotFoundError):
        read_geometry_config(tmp_path / "missing")


# ---------------------------------------------------------------------------
# markers (LIFE IBMBody.cpp)
# ---------------------------------------------------------------------------
def test_circle_marker_count_is_floor_2pir_over_dx():
    c = CircleBody(body_id=0, centre=(0.2, 0.2), radius=0.05)
    for dx, expected in ((0.01, 31), (0.005, 62), (0.0025, 125)):
        assert c.marker_count(dx) == int(math.floor(2 * math.pi * 0.05 / dx)) == expected


def test_circle_markers_lie_on_the_radius():
    c = CircleBody(body_id=0, centre=(0.2, 0.2), radius=0.05)
    pts = c.markers(0.005)
    r = np.linalg.norm(pts - np.array([0.2, 0.2]), axis=1)
    assert np.allclose(r, 0.05, rtol=0, atol=1e-12)
    assert pts.shape == (62, 2)


def test_filament_marker_count_and_rotation():
    f = FilamentBody(
        body_id=1,
        start=(0.25, 0.2),
        length=0.35,
        height=0.02,
        angle_deg=90.0,
        n_elements=20,
    )
    # floor(L/dx) + 1, evaluated on the same IEEE double as LIFE:              #
    # 0.35/0.001 = 349.99999999999994 in binary → 350 nodes, not 351.
    assert f.marker_count(0.001) == int(math.floor(0.35 / 0.001)) + 1
    pts = f.markers(0.001)
    assert np.allclose(pts[0], [0.25, 0.2])
    assert np.allclose(pts[-1], [0.25, 0.2 + 0.35])  # degrees → radians, R = [[c,-s],[s,c]]
    assert math.isclose(float(np.linalg.norm(pts[-1] - pts[0])), 0.35, rel_tol=1e-12)


def test_filament_marker_spacing_is_not_below_dx():
    """``floor(L/dx)+1`` nodes ⇒ spacing L/(n-1) >= dx (LIFE's own rounding)."""
    f = FilamentBody(body_id=1, start=(0.0, 0.0), length=0.35, height=0.02, angle_deg=0.0)
    dx = 0.001
    pts = f.markers(dx)
    step = float(np.linalg.norm(pts[1] - pts[0]))
    assert step >= dx - 1e-15
    assert step <= 1.02 * dx


# ---------------------------------------------------------------------------
# rasterisation / mask injection
# ---------------------------------------------------------------------------
def test_mask_matches_verified_cylinder_mask_exactly():
    import torch

    from tensorlbm.boundaries import cylinder_mask
    from tensorlbm.cylinder_st2d1 import case_geometry

    for D in (10, 20, 40):
        geo = case_geometry(D)
        nx, ny, dx = int(geo["nx"]), int(geo["ny"]), geo["dx"]
        scene = parse_geometry_config("CIRCLE 1 0 0.20 0.20 0.0 0.0 0.05\n")
        got = scene.mask(nx, ny, dx)
        ref = cylinder_mask(
            nx, ny, float(geo["cx"]), float(geo["cy"]), float(geo["radius"]), torch.device("cpu")
        ).numpy()
        assert got.shape == ref.shape
        assert int(np.sum(got ^ ref)) == 0, f"D={D}: mask mismatch"


def test_rotated_filament_area_and_bbox():
    L, h, dx = 0.20, 0.02, 0.001
    f = FilamentBody(body_id=3, start=(0.1, 0.1), length=L, height=h, angle_deg=45.0)
    nx = ny = 400
    m = f.rasterise(nx, ny, dx, origin=(0.0, 0.0))
    area_exact = L * h
    area_got = m.sum() * dx * dx
    assert abs(area_got - area_exact) / area_exact < 0.05  # staircase error only
    ys, xs = np.nonzero(m)
    # rotated 45°: the occupied box grows by L/√2 in both directions
    assert (xs.max() - xs.min()) * dx >= L / math.sqrt(2) - 2 * dx
    assert (ys.max() - ys.min()) * dx >= L / math.sqrt(2) - 2 * dx


def test_out_of_domain_body_is_rejected():
    scene = parse_geometry_config("CIRCLE 1 0 0.9 0.2 0 0 0.05\n")
    with pytest.raises(ValueError, match="leaves the domain"):
        scene.validate_domain(nx=100, ny=100, dx=0.005, origin=(0.0, 0.0))


# ---------------------------------------------------------------------------
# overlap rules (Objects.cpp:removeOverlapMarkers)
# ---------------------------------------------------------------------------
def test_overlapping_rigid_markers_deduplicated():
    scene = Scene(
        [
            CircleBody(body_id=0, centre=(0.2, 0.2), radius=0.05),
            CircleBody(body_id=1, centre=(0.2001, 0.2), radius=0.05),  # nearly identical
        ]
    )
    pts = scene.markers(0.005)
    assert len(pts) < 62 * 2
    # no two surviving markers closer than 0.5 dx
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1) / 0.005
    np.fill_diagonal(d, np.inf)
    assert d.min() >= 0.5 - 1e-12


def test_two_flexible_bodies_too_close_error():
    scene = Scene(
        [
            FilamentBody(body_id=0, start=(0.2, 0.2), length=0.1, height=0.02, angle_deg=0.0),
            FilamentBody(body_id=1, start=(0.2001, 0.2), length=0.1, height=0.02, angle_deg=0.0),
        ]
    )
    with pytest.raises(ValueError, match="flexible bodies"):
        scene.markers(0.005)


def test_flexible_marker_survives_over_rigid():
    """LIFE drops the rigid marker when a rigid and a flexible one overlap."""
    rigid = CircleBody(body_id=0, centre=(0.2, 0.2), radius=0.05)
    flex = FilamentBody(body_id=1, start=(0.2, 0.2), length=0.1, height=0.02, angle_deg=0.0)
    scene = Scene([rigid, flex])
    pts = scene.markers(0.005)
    # every surviving marker within 0.5 dx of a flexible marker was dropped
    flex_pts = flex.markers(0.005)
    rigid_pts = rigid.markers(0.005)
    kept_rigid = 0
    for p in rigid_pts:
        dmin = np.min(np.linalg.norm(flex_pts - p, axis=1)) / 0.005
        if dmin >= 0.5:
            kept_rigid += 1
    assert len(pts) == kept_rigid + len(flex_pts)


# ---------------------------------------------------------------------------
# compile-time / runtime boundary
# ---------------------------------------------------------------------------
def _params(**kw):
    base = dict(nx=100, ny=50, tau=0.8, steps=10, dx=0.005)
    base.update(kw)
    return CaseParams(**base)


def test_geometry_edits_do_not_change_param_fingerprint():
    scene = parse_geometry_config("CIRCLE 1 0 0.2 0.2 0 0 0.05\n")
    setup = SimulationSetup(params=_params(), scene=scene)
    fp = setup.param_fingerprint
    second = setup.add_body(CircleBody(body_id=9, centre=(0.3, 0.2), radius=0.02))
    assert second.param_fingerprint == fp
    assert second.remove_body(9).param_fingerprint == fp


def test_param_edit_requires_rebuild():
    setup = SimulationSetup(params=_params(), scene=parse_geometry_config("CIRCLE 1 0 0.2 0.2 0 0 0.05\n"))
    with pytest.raises(RuntimeError, match="rebuild"):
        setup.apply_params(tau=0.81)
    rebuilt = setup.apply_params(tau=0.81, rebuild=True)
    assert rebuilt.param_fingerprint != setup.param_fingerprint


def test_case_params_validation():
    with pytest.raises(ValueError):
        _params(tau=0.5)
    with pytest.raises(ValueError):
        _params(dx=0.0)
    with pytest.raises(ValueError):
        _params(nx=0)


def test_fingerprint_is_stable_and_discriminating():
    a, b = _params(), _params()
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != _params(steps=11).fingerprint()
    assert len(a.fingerprint()) == 16


# ---------------------------------------------------------------------------
# solver injection: default path inert, config path reproduces it
# ---------------------------------------------------------------------------
def test_solver_injection_is_bit_identical_for_identical_geometry():
    from tensorlbm.cylinder_st2d1 import case_geometry, run_cylinder_st2d1

    D = 10
    geo = case_geometry(D)
    scene = parse_geometry_config("CIRCLE 1 0 0.20 0.20 0.0 0.0 0.05\n")
    mask = scene.mask(int(geo["nx"]), int(geo["ny"]), geo["dx"])
    # D=10 needs an explicit u_lb: the tau-derived u_mean would be Ma 0.35.
    kw = dict(D_cells=D, steps=40, progress=False, keep_fields=False, sample_every=10, u_lb=0.05)
    base = run_cylinder_st2d1(**kw)
    injected = run_cylinder_st2d1(solid_mask=mask, **kw)
    assert base["geometry_source"] == "hard_coded"
    assert injected["geometry_source"] == "runtime_config"
    assert base["n_solid_cells"] == injected["n_solid_cells"]
    assert base["cd"] == pytest.approx(injected["cd"], rel=0, abs=0)


def test_solver_injection_rejects_wrong_shape():
    from tensorlbm.cylinder_st2d1 import run_cylinder_st2d1

    with pytest.raises(ValueError, match="solid_mask shape"):
        run_cylinder_st2d1(
            D_cells=10, steps=1, progress=False, u_lb=0.05, solid_mask=np.zeros((3, 3), bool)
        )