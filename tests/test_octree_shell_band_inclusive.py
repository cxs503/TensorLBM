"""Leaf-BFL band symmetry: inclusive-band regression pins.

The legacy ``inside_fn`` shell band (``dilated & ~solid``) never enumerates
the fluid-classified quarter-children of boundary solid host cells, so every
body face keeps its +normal BFL link layer and loses its -normal layer: the
link set becomes reflection-asymmetric and its correct linkwise-MEM sum
manufactures a spurious net static force (443x per-link pump, diagnosed
2026-10-13 in the suboff AMR campaign; frozen census anchors live in
/nfs/wangxi/runs/linkmech_20261013/probe_d2a2_asymmetry.json).

``build_octree_shell(include_solid_hosts=True)`` restores the enumeration
closure.  These tests pin:

* the legacy defect itself (frozen census numbers, default path) so it can
  never silently change character;
* the inclusive-band closure on three body shapes (centred sphere, offset
  sphere, non-convex snowman);
* the band-mask semantics of both channels;
* the analytic-sphere channel being a leaf-set no-op under the flag;
* the default path being the legacy band bit-exactly.
"""

import pytest
import torch

from tensorlbm.octree_boundary.geometry import (
    build_octree_shell,
    build_shell_cell_mask,
)
from tensorlbm.octree_boundary.geometry_adapters import solid_mask_inside_fn
from tensorlbm.octree_boundary.topology import bfl_link_balance


def _sphere_mask(
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    radius: float,
) -> torch.Tensor:
    """Integer-centre sphere test on the (z, y, x) voxel lattice."""
    nz, ny, nx = shape
    gz, gy, gx = torch.meshgrid(
        torch.arange(nz, dtype=torch.float64),
        torch.arange(ny, dtype=torch.float64),
        torch.arange(nx, dtype=torch.float64),
        indexing="ij",
    )
    cx, cy, cz = center
    m = (gx - cx) ** 2 + (gy - cy) ** 2 + (gz - cz) ** 2 <= radius**2
    return m.contiguous()


def _snowman_mask(
    shape: tuple[int, int, int],
) -> torch.Tensor:
    """Non-convex body: two overlapping spheres of different radii."""
    return _sphere_mask(shape, (16.0, 16.0, 16.0), 7.0) | _sphere_mask(
        shape,
        (24.3, 19.6, 13.4),
        5.0,
    )


# The frozen hybrid sphere arm geometry: analytic L1 sphere mask (38, 38,
# 72), centre (19.2, 18, 18), radius 8, BL 4 + transition 1, d_max 1.
_HYBRID_SHAPE = (38, 38, 72)
_HYBRID_CENTER = (19.2, 18.0, 18.0)
_HYBRID_RADIUS = 8.0


def _hybrid_shell(mask: torch.Tensor, **kw) -> object:
    params = dict(
        shape=tuple(mask.shape),
        center=(19.0, 19.0, 19.0),  # harness call: ignored under inside_fn
        radius=1.0,  # harness call: ignored under inside_fn
        bl_thickness_cells=4.0,
        d_max=1,
        transition=1,
        lattice="D3Q19",
        device="cpu",
        inside_fn=solid_mask_inside_fn(mask),
    )
    params.update(kw)
    return build_octree_shell(**params)


def _legacy_dilated_band(solid: torch.Tensor, passes: int) -> torch.Tensor:
    """Independent reimplementation of the dilation band."""
    import torch.nn.functional as F

    d = solid.float().unsqueeze(0).unsqueeze(0)
    for _ in range(passes):
        d = F.max_pool3d(d, kernel_size=3, stride=1, padding=1)
    return d.squeeze(0).squeeze(0) > 0.5


def test_legacy_band_defect_pinned():
    """Default band reproduces the frozen asymmetric census (defect pin).

    Anchors: probe_d2a2_asymmetry.json (2026-10-13) — 11130 links,
    sum_d n_d w_d c_dx = 211/3 = 70.333..., sum_w_c_y = sum_w_c_z = 74.0
    (= 2664/36 from the frozen per-direction table), 100% mirror orphans,
    and the 3:1 per-direction split (d1 +x 623 vs d2 -x 207).
    """
    mask = _sphere_mask(_HYBRID_SHAPE, _HYBRID_CENTER, _HYBRID_RADIUS)
    shell = _hybrid_shell(mask)
    census = bfl_link_balance(shell, mirror_center=_HYBRID_CENTER)
    assert census["n_links"] == 11130
    assert census["sum_w_c_x"] == pytest.approx(211.0 / 3.0, abs=1.0e-9)
    assert census["sum_w_c_y"] == pytest.approx(74.0, abs=1.0e-9)
    assert census["sum_w_c_z"] == pytest.approx(74.0, abs=1.0e-9)
    assert census["directional_balance_ok"] is False
    assert census["per_direction"][1] == 623  # +x
    assert census["per_direction"][2] == 207  # -x
    # Mirror census at collision-free quarter-lattice resolution (the
    # probe's half-unit keys quantise b=0/b=1 children onto even nodes and
    # collide, reporting 11130 orphans; the finer census still finds the
    # link set overwhelmingly unpaired).
    assert census["mirror_pairing"]["orphan"] == 9290
    assert census["mirror_pairing"]["paired"] == census["n_links"] - 9290


def test_inclusive_band_closes_hybrid_census():
    """Inclusive band: the same geometry closes to machine zero."""
    mask = _sphere_mask(_HYBRID_SHAPE, _HYBRID_CENTER, _HYBRID_RADIUS)
    shell = _hybrid_shell(mask, include_solid_hosts=True)
    census = bfl_link_balance(shell, mirror_center=_HYBRID_CENTER)
    assert census["n_links"] > 0
    assert abs(census["sum_w_c_x"]) <= 1.0e-9
    assert abs(census["sum_w_c_y"]) <= 1.0e-9
    assert abs(census["sum_w_c_z"]) <= 1.0e-9
    assert census["directional_balance_ok"] is True
    # per-direction counts pair up (n_d == n_opp[d])
    from tensorlbm.d3q19 import OPPOSITE

    for d in range(1, 19):
        assert census["per_direction"][d] == census["per_direction"][int(OPPOSITE[d])]


@pytest.mark.parametrize(
    ("shape", "center", "radius", "label"),
    [
        ((32, 32, 32), (16.0, 16.0, 16.0), 8.0, "centred_sphere"),
        ((32, 32, 32), (13.7, 17.1, 15.4), 6.0, "offset_sphere"),
        ((36, 36, 36), None, None, "nonconvex_snowman"),
    ],
)
def test_inclusive_band_balances_three_bodies(shape, center, radius, label):
    """ON -> directional sum == 0 (<= 1e-9) on three body shapes."""
    if label == "nonconvex_snowman":
        mask = _snowman_mask(shape)
    else:
        mask = _sphere_mask(shape, center, radius)
    shell = build_octree_shell(
        shape=shape,
        center=(shape[2] / 2.0, shape[1] / 2.0, shape[0] / 2.0),
        radius=1.0,
        bl_thickness_cells=4.0,
        d_max=1,
        transition=1,
        lattice="D3Q19",
        device="cpu",
        inside_fn=solid_mask_inside_fn(mask),
        include_solid_hosts=True,
    )
    census = bfl_link_balance(shell)
    assert census["n_links"] > 0
    assert census["directional_balance_ok"] is True, census
    # a centred, lattice-aligned body must also mirror-pair exactly
    if label == "centred_sphere":
        census_m = bfl_link_balance(shell, mirror_center=center)
        assert census_m["mirror_pairing"]["orphan"] == 0
        assert census_m["mirror_pairing"]["paired"] == census_m["n_links"]


def test_band_mask_semantics_both_channels():
    """OFF == legacy ``dilated & ~solid``; ON == ``dilated`` (mask channel)."""
    mask = _sphere_mask(_HYBRID_SHAPE, _HYBRID_CENTER, _HYBRID_RADIUS)
    inside = solid_mask_inside_fn(mask)
    solid_off, shell_off, _ = build_shell_cell_mask(
        _HYBRID_SHAPE,
        (19.0, 19.0, 19.0),
        1.0,
        4.0,
        1,
        torch.device("cpu"),
        inside_fn=inside,
    )
    solid_on, shell_on, _ = build_shell_cell_mask(
        _HYBRID_SHAPE,
        (19.0, 19.0, 19.0),
        1.0,
        4.0,
        1,
        torch.device("cpu"),
        inside_fn=inside,
        include_solid_hosts=True,
    )
    dilated = _legacy_dilated_band(solid_off, passes=5)  # round(4.0 + 1)
    assert torch.equal(solid_off, mask.to(solid_off.device))
    assert torch.equal(shell_off, dilated & ~solid_off)
    assert torch.equal(shell_on, dilated)
    assert bool((shell_on & ~shell_off).eq(solid_off & dilated).all())


def test_sphere_channel_flag_is_leaf_set_noop():
    """Analytic-sphere channel: flag ON leaves every topology tensor intact.

    The sphere band already includes cut host cells; the only cells the
    flag adds are wholly inside the sphere, whose every sub-leaf centre is
    inside and is dropped by the leaf centre test.
    """
    common = dict(
        shape=(40, 40, 40),
        center=(20.0, 20.0, 20.0),
        radius=10.0,
        bl_thickness_cells=3.0,
        d_max=2,
        device="cpu",
    )
    off = build_octree_shell(**common)
    on = build_octree_shell(**common, include_solid_hosts=True)
    assert torch.equal(off.leaf_morton, on.leaf_morton)
    assert torch.equal(off.leaf_level, on.leaf_level)
    assert torch.equal(off.leaf_center, on.leaf_center)
    assert torch.equal(off.neighbor_table, on.neighbor_table)
    assert torch.equal(off.bfl_mask, on.bfl_mask)
    assert torch.equal(off.q_field, on.q_field)
    assert torch.equal(off.cross_level_donor, on.cross_level_donor)
    assert torch.equal(off.interface_links, on.interface_links)
    assert torch.equal(off.leaf_host_cell, on.leaf_host_cell)
    assert off.interface_fanout.keys() == on.interface_fanout.keys()
    for k in off.interface_fanout:
        assert off.interface_fanout[k] == on.interface_fanout[k]
    assert on.stats["n_shell_cells"] >= off.stats["n_shell_cells"]
    assert on.meta.get("include_solid_hosts") is True
    assert "include_solid_hosts" not in off.meta
    # both channels carry a balanced census (sphere channel always did)
    for grid in (off, on):
        census = bfl_link_balance(grid)
        assert census["directional_balance_ok"] is True, census


def test_default_path_is_legacy_bit_identical():
    """Default OFF: octree tensors equal the flag explicitly set to False."""
    mask = _sphere_mask(_HYBRID_SHAPE, _HYBRID_CENTER, _HYBRID_RADIUS)
    default = _hybrid_shell(mask)
    explicit = _hybrid_shell(mask, include_solid_hosts=False)
    assert torch.equal(default.leaf_morton, explicit.leaf_morton)
    assert torch.equal(default.neighbor_table, explicit.neighbor_table)
    assert torch.equal(default.bfl_mask, explicit.bfl_mask)
    assert torch.equal(default.q_field, explicit.q_field)
    assert torch.equal(default.interface_links, explicit.interface_links)
    assert default.stats == explicit.stats
    assert default.checks == explicit.checks
    assert "include_solid_hosts" not in default.meta
