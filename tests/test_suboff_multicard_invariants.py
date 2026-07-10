"""Partition-invariance regressions for SUBOFF multicard normalization."""

import torch

from tensorlbm.suboff_resistance import _voxel_wetted_area, voxel_wetted_area_x_slab
from examples.dg_suboff_cumulant_d3q27_multicard import (
    apply_halfway_bounce_back_27,
    pressure_drag_x_27,
    validate_suboff_voxel_resolution,
)


def test_x_slab_wetted_area_matches_global_mask_without_cut_faces():
    """A domain-decomposed hull must not gain wetted area at rank cuts."""
    mask = torch.zeros((3, 5, 12), dtype=torch.bool)
    # Solid crosses both x cuts (x=4 and x=8), so treating slab ends as
    # physical boundaries would spuriously add four exposed faces.
    mask[:, 1:4, 2:10] = True

    global_area = _voxel_wetted_area(mask, 1.0)
    slab_area = sum(
        voxel_wetted_area_x_slab(mask[:, :, start:end], 1.0,
                                 has_left_neighbor=start > 0,
                                 has_right_neighbor=end < mask.shape[2])
        for start, end in ((0, 4), (4, 8), (8, 12))
    )

    assert slab_area == global_area


def test_halfway_bounce_back_reflects_population_from_solid_source():
    """A population pulled from a solid voxel must be reflected at the fluid cell."""
    # x=1 is fluid; its +x-going population is pulled from solid x=0.
    solid = torch.tensor([[[True, False, False]]])
    streamed = torch.zeros((27, 1, 1, 3))
    postcollision = torch.zeros_like(streamed)
    # q=1 is +x and q=2 is its opposite (-x) in D3Q27.
    postcollision[2, 0, 0, 1] = 0.375

    reflected = apply_halfway_bounce_back_27(streamed, postcollision, solid)

    assert reflected[1, 0, 0, 1].item() == 0.375
    assert reflected[1, 0, 0, 2].item() == 0.0


def test_pressure_drag_uses_solid_surface_normal_sign():
    """Higher pressure on a hull's upstream (-x) face gives positive drag."""
    # Solid lies at x=1.  The fluid cells x=0 and x=2 are its -x/+x faces.
    solid = torch.tensor([[[False, True, False]]])
    pressure = torch.tensor([[[2.0, 0.0, 1.0]]])

    # -p*n_x: n_x=-1 upstream and +1 downstream => 2 - 1 = +1.
    assert pressure_drag_x_27(pressure, solid).item() == 1.0


def test_suboff_ct_benchmark_rejects_underresolved_diameter():
    """A 9-cell SUBOFF diameter cannot supply an absolute resistance Ct.

    The previous 192×96×96 / L=80 setup has R=L/(2*8.57)=4.67 cells.
    Its voxel staircase is therefore a dominant form-drag source, rather
    than a discretisation of the smooth AFF-8 reference geometry.  The
    benchmark must fail closed instead of reporting that non-physical Ct as
    a comparison to the experimental smooth-body value.
    """
    try:
        validate_suboff_voxel_resolution(hull_length=80.0)
    except ValueError as exc:
        assert "diameter" in str(exc)
    else:
        raise AssertionError("underresolved SUBOFF geometry was accepted")

    # A 24-cell diameter is the minimum coarse resolution at which the
    # benchmark is permitted to make an absolute-Ct claim.
    validate_suboff_voxel_resolution(hull_length=206.0)
