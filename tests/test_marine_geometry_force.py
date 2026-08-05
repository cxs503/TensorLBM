import pytest
import torch

from tensorlbm.force_observation import ForceObservation
from tensorlbm.marine_geometry import GeometryAsset, compile_d3q19_wall_links


def _asset(mask: torch.Tensor) -> GeometryAsset:
    return GeometryAsset(
        solid_mask=mask,
        body_id="shared-body",
        origin=(1.0, 2.0, 3.0),
        units="lattice",
        source_id="test-fixture",
    )


def test_geometry_asset_is_static_and_records_provenance() -> None:
    mask = torch.zeros((3, 4, 5), dtype=torch.bool)
    mask[1, 2, 3] = True
    asset = _asset(mask)

    mask[1, 2, 3] = False
    assert asset.solid_mask[1, 2, 3].item() is True
    public_mask = asset.solid_mask
    public_mask[1, 2, 3] = False
    assert asset.solid_mask[1, 2, 3].item() is True
    assert compile_d3q19_wall_links(asset).count > 0
    assert asset.origin == (1.0, 2.0, 3.0)
    assert asset.units == "lattice"
    assert asset.source_id == "test-fixture"
    assert len(asset.source_hash) == 64


def test_compile_d3q19_emits_all_18_solid_to_fluid_links_without_axis_swap() -> None:
    mask = torch.zeros((5, 5, 5), dtype=torch.bool)
    mask[2, 2, 2] = True

    links = compile_d3q19_wall_links(_asset(mask))

    assert links.lattice_id == "D3Q19"
    assert links.count == 18
    assert torch.equal(links.owner_zyx, torch.tensor([[2, 2, 2]]).repeat(18, 1))
    assert set(links.direction.tolist()) == set(range(1, 19))
    # q=1 is C=(+x, 0, 0), hence z/y/x neighbour=(2, 2, 3).
    q1 = (links.direction == 1).nonzero(as_tuple=False).item()
    assert tuple(links.neighbor_zyx[q1].tolist()) == (2, 2, 3)
    assert links.body_id == "shared-body"


def test_wall_links_do_not_treat_periodic_wrap_as_a_wall_link() -> None:
    mask = torch.zeros((3, 3, 3), dtype=torch.bool)
    mask[0, 0, 0] = True

    links = compile_d3q19_wall_links(_asset(mask))

    # Only directions whose z/y/x neighbour is in-bounds are valid; no wrap.
    assert links.count == 6
    assert bool((links.neighbor_zyx >= 0).all())
    assert bool((links.neighbor_zyx < 3).all())


def test_only_solid_to_fluid_links_are_owned() -> None:
    mask = torch.zeros((5, 5, 5), dtype=torch.bool)
    mask[2, 2, 2] = True
    mask[2, 2, 3] = True

    links = compile_d3q19_wall_links(_asset(mask))

    assert not bool(((links.neighbor_zyx == torch.tensor([2, 2, 3])).all(dim=1)).any())
    assert bool(mask[tuple(links.owner_zyx.T)].all())
    assert not bool(mask[tuple(links.neighbor_zyx.T)].any())


def test_force_observation_fails_closed_without_link_ownership() -> None:
    common = dict(
        method="momentum_exchange",
        lattice_id="D3Q19",
        sample_phase="post_stream_pre_bounce_back",
        force_on="body",
        origin=(0.0, 0.0, 0.0),
        force=(1.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="link ownership"):
        ForceObservation(status="measured", **common)

    diagnostic = ForceObservation(status="diagnostic_only", **common)
    assert diagnostic.status == "diagnostic_only"
    measured = ForceObservation(status="measured", link_ownership=True, **common)
    assert measured.status == "measured"
