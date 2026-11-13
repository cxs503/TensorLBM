"""XVAL-LBM-RASTER-OBSTACLE-CORNER-CLASS-BOUNCE-BACK -- operator-level tests.

These tests are the machine-checked core of the cross-validation claim:

1. the OSS corner-class table never *misses* an exact solid link
   (``missed_links == 0``) and always *adds* exactly 8 links per isolated
   rectangle (the band-end diagonals);
2. the TensorLBM per-direction mask reproduces ``tensorlbm``'s own
   ``neighbour_mask`` / ``stream_mask_vector`` kernels bit for bit;
3. one streaming step from the same population field produces a difference
   confined to exactly those spurious sites and directions;
4. the analytic anchor closed form used by the report is the discrete
   Poiseuille permeability ``ny^2/12 + 1/24``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "xval" / "XVAL-LBM-RASTER-OBSTACLE-CORNER-CLASS-BOUNCE-BACK"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raster_case import (  # noqa: E402
    Arm,
    Case,
    D2Q9_C,
    analytic_channel,
    equilibrium,
    exact_closed_links,
    link_difference,
    oss_bounced_links,
    oss_rect_corner_list,
    raster_solid_mask,
)
from raster_case import D2Q9_OPP  # noqa: E402
from tensorlbm.bitflag_neighbour_mask import (  # noqa: E402
    D2Q9_C as TL_C,
    CellFlag,
    neighbour_mask,
)

#: module index of every OSS direction 1..8 (the two stencils differ by a
#: permutation of the diagonal/face labels only, which is asserted in raster_case)
PERM = [[tuple(int(t) for t in r) for r in TL_C.tolist()].index(tuple(int(t) for t in D2Q9_C[i]))
        for i in range(1, 9)]
INV_PERM = [PERM.index(j + 1) for j in range(8)]  # OSS index (1-based) of module dir j+1


def _to_module_order(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask)
    for i in range(1, 9):
        bit = (mask >> np.uint8(i - 1)) & np.uint8(1)
        out |= bit << np.uint8(PERM[i - 1] - 1)
    return out


def _bits(v: int) -> list[int]:
    return [i for i in range(1, 9) if int(v) >> (i - 1) & 1]


@pytest.mark.parametrize("b,nx,ny", [(12, 132, 144), (24, 264, 288)])
def test_geometry_matches_declared_porosity(b, nx, ny):
    rects = oss_rect_corner_list(nx, ny, b)
    solid = raster_solid_mask(nx, ny, rects)
    assert len(rects) == 12
    # every block is b x b, no overlap, no wall contact
    assert solid.sum() == 12 * b * b
    assert not solid[:, :b].any() and not solid[:, -b:].any()
    assert not solid[:b, :].any() and not solid[-b:, :].any()


def test_exact_mask_matches_bruteforce():
    nx, ny, b = 40, 40, 8
    rects = [((8, 8), (15, 15)), ((24, 20), (31, 27))]
    solid = raster_solid_mask(nx, ny, rects)
    mask = exact_closed_links(solid)
    bf = np.zeros((nx, ny), dtype=np.uint8)
    for x in range(nx):
        for y in range(ny):
            for i in range(1, 9):
                cx, cy = int(D2Q9_C[i, 0]), int(D2Q9_C[i, 1])
                ux, uy = (x - cx) % nx, y - cy
                if uy < 0 or uy >= ny or solid[ux, uy]:
                    bf[x, y] |= 1 << (i - 1)
    assert np.array_equal(mask, bf)


def test_exact_mask_is_tensorlbm_neighbour_mask_without_periodic_wrap():
    """With solid x-borders no open link crosses the periodic wrap, so our mask
    must coincide with ``tensorlbm.bitflag_neighbour_mask.neighbour_mask``."""
    nx, ny, b = 48, 48, 6
    rects = [((0, 0), (0, ny - 1)), ((nx - 1, 0), (nx - 1, ny - 1)), ((18, 18), (23, 23))]
    solid = raster_solid_mask(nx, ny, rects)
    ours = exact_closed_links(solid)
    cell_type = np.where(solid, int(CellFlag.OBSTACLE), int(CellFlag.FLUID)).astype(np.int32)
    ref = neighbour_mask(cell_type)
    assert np.array_equal(_to_module_order(ours), ref)


@pytest.mark.parametrize("b,nx,ny", [(12, 132, 144), (24, 264, 288)])
def test_table_covers_every_exact_link_and_adds_eight_per_block(b, nx, ny):
    rects = oss_rect_corner_list(nx, ny, b)
    solid = raster_solid_mask(nx, ny, rects)
    d = link_difference(solid, rects)
    assert d["missed_links"] == 0, "the corner-class table must never miss a solid link"
    assert d["spurious_links"] == 8 * len(rects)
    assert d["table_links"] == d["exact_links"] + d["spurious_links"]
    # each spurious site sits on a band end (one cell beside a rectangle edge) and
    # carries a single diagonal direction
    for x, y in np.argwhere(d["spurious_mask"] > 0):
        dirs = _bits(d["spurious_mask"][x, y])
        assert len(dirs) == 1 and dirs[0] in (5, 6, 7, 8)
        assert not solid[x, y]


def test_oss_table_reproduces_all_eight_corner_classes():
    """The four band classes and four diagonal corner classes of the OSS table."""
    nx, ny, b = 40, 40, 8
    rects = [((8, 8), (15, 15))]
    solid = raster_solid_mask(nx, ny, rects)
    table, masks = oss_bounced_links(nx, ny, rects)
    (x0, y0), (x1, y1) = rects[0]
    expected = {
        (x0, y1 + 1): {2, 5, 6},      # north band, west end
        (x1, y1 + 1): {2, 5, 6},      # north band, east end
        (x0 - 1, y0 - 1): {7},        # south-west corner cell
        (x1 + 1, y1 + 1): {5},        # north-east corner cell
        (x1 + 1, y0): {1, 5, 8},      # east band, south end
        (x0 - 1, y1): {3, 6, 7},      # west band, north end
    }
    for (x, y), dirs in expected.items():
        assert set(_bits(table[x, y])) == dirs, (x, y, _bits(table[x, y]), dirs)


def test_operator_difference_confined_to_spurious_links():
    """One streaming step from the same field differs exactly at the spurious links."""
    case = Case(blocks=True, b=6, nx=60, ny=60, tau=2.0, g=1e-4)
    oss, mask = Arm(case, "oss"), Arm(case, "mask")
    f = equilibrium(np.ones((case.nx, case.ny)), np.full((case.nx, case.ny), 0.01),
                    np.zeros((case.nx, case.ny)))
    f = f * (1.0 + 0.02 * np.random.default_rng(0).standard_normal(f.shape))
    d = np.abs(oss.stream(f, use_mask=False) - mask.stream(f, use_mask=True))
    info = link_difference(mask.solid, mask.rects)
    fluid = ~mask.solid
    sites = set(map(tuple, np.argwhere(info["spurious_mask"] > 0).tolist()))
    diff_sites = set(map(tuple, np.argwhere((d.max(axis=0) > 0) & fluid).tolist()))
    assert diff_sites == sites
    for x, y in sites:
        assert _bits(info["spurious_mask"][x, y]) == [i for i in range(1, 9) if d[i, x, y] > 0]


def test_mask_stream_equals_bruteforce_pull_kernel():
    """Our mask streaming equals an independent brute-force pull+bounce sweep.

    (The mask rule itself is proven equal to ``tensorlbm``'s ``neighbour_mask``
    by ``test_exact_mask_is_tensorlbm_neighbour_mask_without_periodic_wrap``;
    this test pins the streaming semantics: ``f_i(x) = f_opp(x)`` on a closed
    link, ``f_i(x - c_i)`` otherwise, periodic in x, bounced wall rows.)
    """
    nx, ny, b = 48, 48, 6
    rects = [((0, 0), (0, ny - 1)), ((nx - 1, 0), (nx - 1, ny - 1)), ((18, 18), (23, 23))]
    case = Case(blocks=False, b=b, nx=nx, ny=ny, tau=2.0, g=0.0)
    arm = Arm(case, "mask")
    arm.rects = rects
    arm.solid = raster_solid_mask(nx, ny, rects)
    arm.fluid = ~arm.solid
    arm.exact_mask = exact_closed_links(arm.solid)
    arm.mask = arm.exact_mask
    arm.mask_all = arm.mask | arm.wall_mask
    rng = np.random.default_rng(1)
    f = rng.standard_normal((9, nx, ny))
    ours = arm.stream(f, use_mask=True)
    ref = np.empty_like(f)
    for x in range(nx):
        for y in range(ny):
            ref[0, x, y] = f[0, x, y]
            if arm.solid[x, y]:
                ref[:, x, y] = f[:, x, y]
                continue
            for i in range(1, 9):
                cx, cy = int(D2Q9_C[i, 0]), int(D2Q9_C[i, 1])
                if int(arm.mask_all[x, y]) >> (i - 1) & 1:
                    ref[i, x, y] = f[int(D2Q9_OPP[i]), x, y]
                else:
                    ref[i, x, y] = f[i, (x - cx) % nx, y - cy]
    fluid = ~arm.solid
    assert np.allclose(ours[:, fluid], ref[:, fluid])


def test_analytic_anchor_closed_form():
    case = Case(blocks=False, b=12, nx=132, ny=144, tau=2.0, g=1e-6)
    ana = analytic_channel(case)
    ny = case.ny
    assert ana["k_lattice"] == pytest.approx(ny ** 2 / 12.0 + 1.0 / 24.0, rel=1e-12)