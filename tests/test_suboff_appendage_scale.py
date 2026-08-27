"""Tests for the SUBOFF appendage scale axis (``sail_scale`` / ``fin_scale``).

B1-5 context: the three DARPA configurations (bare_hull / with_sail /
full) voxelise self-similarly at production resolutions because the sail
is sub-cell wide (0.7% of hull solid cells at n128).  ``sail_scale`` /
``fin_scale`` multiply each appendage's own dimensions about fixed DARPA
anchors, opening a real geometry axis.  These tests pin:

- ``scale == 1.0`` reproduces the historical masks **bitwise**
  (``torch.equal``), including against exact solid-cell counts recorded
  from the pre-scale code (base 2f75646);
- the scaled solid grows monotonically and nested (mask(s) ⊇ mask(1));
- stats follow the exact similarity laws (volume ~ s^3, wetted ~ s^2)
  and the voxel appendage counts are internally consistent;
- the continuous point predicates agree bitwise with the voxel mask
  builders at cell centres (the two code paths stay one geometry);
- the ``suboff_n128`` case threads the scales through ``SuboffConfig``
  so the scan chain can sweep them as plain numeric params.
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.cases import get_case
from tensorlbm.suboff_cad import (
    SuboffConfig,
    build_suboff_mask,
    suboff_appendages_contain_points,
    suboff_sail_contains_points,
)

HULLS = ("bare_hull", "with_sail", "full")

# Production suboff_n128 placement (cases/suboff.py): 128 x 64 x 64 grid,
# hull centred at cx = 0.35 nx with length 0.6 nx.
NX, NY, NZ = 128, 64, 64
CX, CY, CZ = NX * 0.35, NY / 2.0, NZ / 2.0
LENGTH = 0.6 * NX

# Small grid for the fast tests.
SNX, SNY, SNZ = 64, 40, 40
SCX, SCY, SCZ = SNX * 0.35, SNY / 2.0, SNZ / 2.0
SLENGTH = 0.6 * SNX

# Exact solid-cell counts of the pre-scale code (base 2f75646) at the
# production grid above — the scale=1 regression anchor.
BASE_SOLID_CELLS = {"bare_hull": 4093, "with_sail": 4121, "full": 4157}


def _mask(hull: str, sail: float = 1.0, fin: float = 1.0, small: bool = False):
    nx, ny, nz = (SNX, SNY, SNZ) if small else (NX, NY, NZ)
    cx, cy, cz = (SCX, SCY, SCZ) if small else (CX, CY, CZ)
    length = SLENGTH if small else LENGTH
    return build_suboff_mask(
        hull,
        nx=nx,
        ny=ny,
        nz=nz,
        cx=cx,
        cy=cy,
        cz=cz,
        length=length,
        config=SuboffConfig(sail_scale=sail, fin_scale=fin),
    )


# ---------------------------------------------------------------------------
# scale = 1.0 regression pins
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hull", HULLS)
def test_scale_one_bitwise_identical_to_default(hull: str) -> None:
    """Explicit scale=1.0 config must equal the default config bitwise."""
    default, _ = build_suboff_mask(hull, nx=NX, ny=NY, nz=NZ, cx=CX, cy=CY, cz=CZ, length=LENGTH)
    explicit, stats = _mask(hull, sail=1.0, fin=1.0)
    assert torch.equal(default, explicit)
    assert stats["solid_cells"] == BASE_SOLID_CELLS[hull]
    assert stats["sail_scale"] == 1.0 and stats["fin_scale"] == 1.0


@pytest.mark.parametrize("hull", HULLS)
def test_scale_one_matches_pre_scale_counts(hull: str) -> None:
    """Solid cells at scale 1 equal the pre-scale (base 2f75646) counts."""
    _, stats = _mask(hull)
    assert stats["solid_cells"] == BASE_SOLID_CELLS[hull]
    assert stats["appendage_solid_cells"] == BASE_SOLID_CELLS[hull] - BASE_SOLID_CELLS["bare_hull"]


@pytest.mark.parametrize("sail,fin", ((2.0, 3.0), (3.0, 2.0), (1.7, 1.7)))
def test_bare_hull_immune_to_scale(sail: float, fin: float) -> None:
    """Scales only touch appendages; the bare hull mask is untouched."""
    base, s0 = _mask("bare_hull", small=True)
    scaled, s1 = _mask("bare_hull", sail=sail, fin=fin, small=True)
    assert torch.equal(base, scaled)
    assert s1["displacement_lu3"] == s0["displacement_lu3"]
    assert s1["wetted_area_lu2"] == s0["wetted_area_lu2"]
    assert s1["appendage_solid_cells"] == 0
    assert "sail_own_volume_lu3" not in s1  # bare hull reports no appendage stats


# ---------------------------------------------------------------------------
# Monotonicity and nesting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hull", ("with_sail", "full"))
def test_solid_cells_monotone_in_scale(hull: str) -> None:
    """More scale, strictly more solid cells (the geometry axis opens)."""
    counts = [
        _mask(hull, sail=s, fin=s, small=True)[1]["solid_cells"] for s in (1.0, 1.5, 2.0, 3.0)
    ]
    assert all(b > a for a, b in zip(counts, counts[1:])), counts


@pytest.mark.parametrize("hull", ("with_sail", "full"))
def test_scaled_mask_nests_the_scale_one_mask(hull: str) -> None:
    """Growing appendages never carve fluid: mask(1) ⊆ mask(2) ⊆ mask(3)."""
    m1, _ = _mask(hull, sail=1.0, fin=1.0, small=True)
    m2, _ = _mask(hull, sail=2.0, fin=2.0, small=True)
    m3, _ = _mask(hull, sail=3.0, fin=3.0, small=True)
    assert int((m1 & ~m2).sum()) == 0
    assert int((m2 & ~m3).sum()) == 0
    assert int((m3 & ~m1).sum()) > 0


def test_appendage_cell_counts_consistent() -> None:
    """appendage_solid_cells == solid - bare, and grows with scale."""
    for scale in (1.0, 2.0, 3.0):
        m, s = _mask("full", sail=scale, fin=scale, small=True)
        assert s["appendage_solid_cells"] == s["solid_cells"] - s["bare_hull_solid_cells"]
        assert int(m.sum()) == s["solid_cells"]


# ---------------------------------------------------------------------------
# Similarity scaling laws in the stats
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key,power", (("sail_own_volume_lu3", 3), ("sail_own_wetted_area_lu2", 2)))
def test_sail_stats_self_similar_laws(key: str, power: int) -> None:
    """Analytic own-dimension quadrature scales exactly as s^power."""
    base = _mask("with_sail", small=True)[1][key]
    for scale in (2.0, 3.0):
        scaled = _mask("with_sail", sail=scale, small=True)[1][key]
        assert scaled == pytest.approx(base * scale**power, rel=1e-3)


@pytest.mark.parametrize("key,power", (("fin_own_volume_lu3", 3), ("fin_own_wetted_area_lu2", 2)))
def test_fin_stats_self_similar_laws(key: str, power: int) -> None:
    base = _mask("full", small=True)[1][key]
    for scale in (2.0, 3.0):
        scaled = _mask("full", fin=scale, small=True)[1][key]
        assert scaled == pytest.approx(base * scale**power, rel=1e-3)


def test_bare_hull_stats_unchanged_by_scale() -> None:
    """Bare-hull displacement / wetted area must not depend on appendage scale."""
    s1 = _mask("full", small=True)[1]
    s3 = _mask("full", sail=3.0, fin=3.0, small=True)[1]
    for key in ("displacement_lu3", "wetted_area_lu2", "prismatic_coefficient"):
        assert s3[key] == s1[key]


# ---------------------------------------------------------------------------
# Anchors: scaled appendages stay put and attached
# ---------------------------------------------------------------------------


def test_sail_axial_anchor_fixed() -> None:
    """The sail footprint grows about its axial centre: bbox centre pinned."""
    centres = []
    for scale in (1.0, 2.0, 3.0):
        bare, _ = _mask("bare_hull", small=True)
        ws, _ = _mask("with_sail", sail=scale, small=True)
        xs = (ws & ~bare).nonzero()[:, 2]
        centres.append(0.5 * (float(xs.min()) + float(xs.max())))
    assert max(centres) - min(centres) <= 1.0, centres


def test_fin_trailing_edge_anchor_fixed() -> None:
    """Fin chord grows forward about the common trailing edge: x_max pinned."""
    xs_max = []
    for scale in (1.0, 2.0, 3.0):
        ws, _ = _mask("with_sail", small=True)
        full, _ = _mask("full", fin=scale, small=True)
        fin_cells = full & ~ws
        xs_max.append(int(fin_cells.nonzero()[:, 2].max()))
    assert max(xs_max) == min(xs_max), xs_max


def test_scaled_appendage_stays_attached_to_hull() -> None:
    """No floating appendage islands at s=3: every appendage connected
    component of the solid reaches the (1-cell dilated) bare hull."""
    bare, _ = _mask("bare_hull", sail=3.0, fin=3.0, small=True)
    full, _ = _mask("full", sail=3.0, fin=3.0, small=True)
    appendage = full & ~bare

    def dilate(m: torch.Tensor) -> torch.Tensor:
        pad = torch.zeros((SNZ + 2, SNY + 2, SNX + 2), dtype=torch.bool)
        pad[1:-1, 1:-1, 1:-1] = m
        return (
            pad[:-2, 1:-1, 1:-1]
            | pad[2:, 1:-1, 1:-1]
            | pad[1:-1, :-2, 1:-1]
            | pad[1:-1, 2:, 1:-1]
            | pad[1:-1, 1:-1, :-2]
            | pad[1:-1, 1:-1, 2:]
        )

    grown = dilate(bare)
    reachable = grown.clone()
    targets = appendage | grown
    for _ in range(4 * SNX):  # generous BFS bound
        nxt = dilate(reachable) & targets
        if torch.equal(nxt, reachable):
            break
        reachable = nxt
    assert int((appendage & ~reachable).sum()) == 0


# ---------------------------------------------------------------------------
# Point predicates vs voxel builders (the two code paths)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sail,fin", ((1.0, 1.0), (2.0, 2.0), (3.0, 1.5)))
def test_point_predicate_matches_voxel_mask(sail: float, fin: float) -> None:
    """Cell-centre predicate union == voxel mask builder, bitwise.

    The continuous predicates and the voxel builders are two code paths
    over one geometry; this pins them together across scales.  Note the
    union contains both sail and fins, so it reproduces the FULL mask
    over the bare hull (the with_sail mask needs the sail predicate
    alone).
    """
    nz, ny, nx = SNZ, SNY, SNX
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32),
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    bare, _ = _mask("bare_hull", sail=sail, fin=fin, small=True)
    ws, _ = _mask("with_sail", sail=sail, fin=fin, small=True)
    full, _ = _mask("full", sail=sail, fin=fin, small=True)
    pts = suboff_appendages_contain_points(
        xx,
        yy,
        zz,
        center=(SCX, SCY, SCZ),
        length=SLENGTH,
        sail_scale=sail,
        fin_scale=fin,
    )
    sail_pts = suboff_sail_contains_points(
        xx, yy, zz, center=(SCX, SCY, SCZ), length=SLENGTH, scale=sail
    )
    assert torch.equal(ws, bare | sail_pts)
    assert torch.equal(full, bare | pts)
    assert torch.equal(full, ws | pts)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", (0.0, -1.0, float("nan"), float("inf")))
def test_config_rejects_nonpositive_scale(bad: float) -> None:
    with pytest.raises(ValueError, match="sail_scale|fin_scale"):
        SuboffConfig(sail_scale=bad)
    with pytest.raises(ValueError, match="sail_scale|fin_scale"):
        SuboffConfig(fin_scale=bad)


# ---------------------------------------------------------------------------
# Case wiring (suboff_n128)
# ---------------------------------------------------------------------------


def test_case_scales_flow_into_mask() -> None:
    base = get_case("suboff_n128", resolution=48, re=100.0, hull_type="full")
    scaled = get_case(
        "suboff_n128", resolution=48, re=100.0, hull_type="full", sail_scale=2.0, fin_scale=2.0
    )
    m0, m1 = base.solid_mask(), scaled.solid_mask()
    assert int(m1.sum()) > int(m0.sum())
    assert int((m0 & ~m1).sum()) == 0  # nesting holds through the case too
    meta = scaled.metadata()
    assert meta["sail_scale"] == 2.0 and meta["fin_scale"] == 2.0


def test_case_default_equals_explicit_scale_one() -> None:
    default = get_case("suboff_n128", resolution=48, re=100.0, hull_type="full")
    explicit = get_case(
        "suboff_n128", resolution=48, re=100.0, hull_type="full", sail_scale=1.0, fin_scale=1.0
    )
    assert torch.equal(default.solid_mask(), explicit.solid_mask())
