"""CAD STL round trip for the parametric SUBOFF: params -> STL -> mask.

Pins the post-merge gate promised by ``docs/voxelize_stl_20260824.md``:
a tessellated surface of the SAME analytic description ``suboff_cad``
voxelises, through the generic STL intake, onto the corpus masks.

1. binary STL writer/loader round trip (triangle count + watertight);
2. surface-of-revolution volumes vs closed forms (cylinder, frustum)
   and vs the ``suboff_cad`` quadrature of the real profile;
3. mother-hull round trip at the production 40x40x128 grid: IoU,
   interior exactness, disagreement localized to the boundary band;
4. one family variant (slender) round trip;
5. component visibility: with_sail vs bare differ exactly in the sail
   region (XOR confined to the sail bbox + 1 voxel);
6. the gate runner writes a machine-readable JSON summary.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

# Defensive import guard: skip cleanly when a needed main-branch module
# is absent (all of them exist on the 65cb33c base this gate targets).
if (
    importlib.util.find_spec("tensorlbm.voxelize") is None
    or importlib.util.find_spec("tensorlbm.suboff_cad") is None
    or importlib.util.find_spec("tensorlbm.cad_stl") is None
):
    pytest.skip("tensorlbm.voxelize/suboff_cad/cad_stl unavailable", allow_module_level=True)

from tensorlbm.cad_stl import (  # noqa: E402
    PRODUCTION_SHAPE,
    _loft_closed_rings,
    _mesh_volume,
    _surface_of_revolution,
    roundtrip_mask,
    run_roundtrip_gate,
    suboff_to_stl,
    tessellate_suboff,
)
from tensorlbm.suboff_cad import SuboffConfig, suboff_statistics  # noqa: E402
from tensorlbm.voxelize import (  # noqa: E402
    is_watertight,
    load_stl,
    mask_from_stl,
    place_on_grid,
)

LENGTH = 0.6 * PRODUCTION_SHAPE[2]  # 76.8 lattice units at nx=128
# Corpus masks sample integer lattice nodes; mask_from_stl samples cell
# centres (origin + (i + 0.5) * spacing), so -0.5 per axis evaluates the
# identical points (the documented round-trip sampling alignment).
NODE_ORIGIN = (-0.5, -0.5, -0.5)


# ---------------------------------------------------------------------------
# STL writer / loader round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hull_type", ["full", "bare_hull"])
def test_stl_export_reload_watertight(tmp_path: Path, hull_type: str) -> None:
    report = suboff_to_stl(
        SuboffConfig(), tmp_path / f"suboff_{hull_type}.stl", hull_type=hull_type, length=LENGTH
    )
    mesh = load_stl(report.path)
    assert mesh.vertices.shape == (report.n_triangles, 3, 3)
    assert report.n_triangles == sum(report.components.values())
    assert report.watertight and is_watertight(mesh.vertices)
    # authored frame: bow tip at x=0, stern tip at x=length, axis at y=z=0
    assert report.bbox_min[0] == pytest.approx(0.0)
    assert report.bbox_max[0] == pytest.approx(LENGTH)
    assert report.volume_lu3 > 0.0
    # the params echo carries every geometry-bearing axis
    assert set(report.params) == {
        "r_over_l",
        "sail_scale",
        "fin_scale",
        "l_over_d_mult",
        "nose_len_mult",
        "stern_len_mult",
        "sail_x_mult",
    }


def test_component_tessellations_are_closed_manifolds() -> None:
    comps = tessellate_suboff(SuboffConfig(), hull_type="full", length=LENGTH)
    assert set(comps) == {"hull", "sail", "fins"}
    for key, tris in comps.items():
        assert tris.ndim == 3 and tris.shape[1:] == (3, 3), key
        assert is_watertight(tris), key
        assert _mesh_volume(tris) > 0.0, key  # outward-oriented shells


# ---------------------------------------------------------------------------
# Surface-of-revolution volume vs closed forms
# ---------------------------------------------------------------------------


def test_surface_of_revolution_cylinder() -> None:
    x = np.linspace(0.0, 10.0, 17)
    r = np.full_like(x, 3.0)
    tris = _surface_of_revolution(x, r, 64)
    assert is_watertight(tris)
    exact = np.pi * 3.0**2 * 10.0
    assert abs(_mesh_volume(tris) - exact) / exact < 0.01


def test_surface_of_revolution_frustum() -> None:
    x = np.linspace(0.0, 9.0, 33)
    r = 1.0 + 3.0 * (x / 9.0)
    tris = _surface_of_revolution(x, r, 64)
    assert is_watertight(tris)
    exact = np.pi * 9.0 * (1.0**2 + 1.0 * 4.0 + 4.0**2) / 3.0
    assert abs(_mesh_volume(tris) - exact) / exact < 0.01


def test_hull_volume_matches_suboff_quadrature() -> None:
    cfg = SuboffConfig()
    comps = tessellate_suboff(cfg, hull_type="bare_hull", length=LENGTH)
    exact = float(
        suboff_statistics("bare_hull", LENGTH, cfg.r_over_l * LENGTH, cfg)["displacement_lu3"]
    )
    assert abs(_mesh_volume(comps["hull"]) - exact) / exact < 0.01


def test_loft_closed_rings_prism_is_watertight() -> None:
    # sweeping a square ring must give a closed watertight prism
    n = 12
    square = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
    rings = np.broadcast_to(square, (n, 4, 3)).copy()
    rings[..., 2] = np.linspace(0.0, 5.0, n)[:, None]
    tris = _loft_closed_rings(rings)
    assert is_watertight(tris)
    assert abs(_mesh_volume(tris) - 4.0 * 5.0) / (4.0 * 5.0) < 1e-9


# ---------------------------------------------------------------------------
# Round trips at the production grid
# ---------------------------------------------------------------------------


def test_mother_round_trip_production_grid() -> None:
    report = roundtrip_mask(SuboffConfig(), PRODUCTION_SHAPE, name="mother", hull_type="full")
    assert report.iou > 0.98
    assert report.boundary_disagreement_frac < 0.05
    # disagreement is localized to the tessellated surface, and the
    # interior agrees exactly -- assert the localization, not the count
    assert report.localized_frac >= 0.99
    assert report.interior_exact
    assert 0.97 < report.volume_ratio < 1.03
    comp = report.components
    assert comp["hull"]["iou"] > 0.98
    assert comp["sail"]["iou"] > 0.95
    # fins are ~1 voxel thin at this grid (the PR #235 caveat) yet the
    # tessellated NACA slabs reproduce them cell-for-cell
    assert comp["fins"]["iou"] > 0.90


def test_slender_variant_round_trip() -> None:
    report = roundtrip_mask(
        SuboffConfig(l_over_d_mult=1.25),
        PRODUCTION_SHAPE,
        name="slender",
        hull_type="full",
    )
    assert report.iou > 0.98
    assert report.boundary_disagreement_frac < 0.05
    assert report.interior_exact


def test_sail_visibility_xor_confined_to_sail_bbox() -> None:
    shape = PRODUCTION_SHAPE
    cfg = SuboffConfig()
    bare = tessellate_suboff(cfg, hull_type="bare_hull", length=LENGTH)
    rigged = tessellate_suboff(cfg, hull_type="with_sail", length=LENGTH)
    # canonical placement derived from the hull (the analytic frame),
    # applied unchanged to the sail -- the round-trip convention
    placement = place_on_grid(bare["hull"], shape, scale=1.0)
    offset = placement.tris[0, 0] - bare["hull"][0, 0]
    mask_bare = mask_from_stl(bare["hull"] + offset, shape, origin=NODE_ORIGIN)
    sail = rigged["sail"] + offset
    mask_ws = mask_bare | mask_from_stl(sail, shape, origin=NODE_ORIGIN)
    xor = np.argwhere(mask_ws ^ mask_bare)
    assert xor.size > 0  # the sail is visible through the STL path
    lo = np.floor(sail.reshape(-1, 3).min(axis=0)) - 1.0
    hi = np.ceil(sail.reshape(-1, 3).max(axis=0)) + 1.0
    assert (xor >= lo).all() and (xor <= hi).all()


# ---------------------------------------------------------------------------
# Gate runner
# ---------------------------------------------------------------------------


def test_run_roundtrip_gate_writes_json(tmp_path: Path) -> None:
    out = tmp_path / "gate.json"
    cases = [
        {"name": "mother", "hull_type": "full", "params": {}},
        {"name": "bare", "hull_type": "bare_hull", "params": {}},
    ]
    summary = run_roundtrip_gate(cases, out)
    data = json.loads(out.read_text())
    assert set(data["cases"]) == {"mother", "bare"}
    assert data["all_pass"] is True
    assert summary["all_pass"] is True
    for name in ("mother", "bare"):
        entry = data["cases"][name]
        assert entry["iou"] > 0.98
        assert entry["boundary_disagreement_frac"] < 0.05
        assert entry["interior_exact"] is True
        assert entry["components"], name
