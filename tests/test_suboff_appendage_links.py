from __future__ import annotations

import torch

from tensorlbm.interpolated_bc_suboff import compute_q_suboff
from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask
from tensorlbm.suboff_static_amr import apply_suboff_appendage_halfway_links


def test_full_suboff_appendage_links_receive_audited_halfway_fallback() -> None:
    nx, ny, nz = 180, 70, 70
    center = (60.0, 35.0, 35.0)
    length = 120.0
    config = SuboffConfig()
    solid, _ = build_suboff_mask(
        "full", nx, ny, nz,
        cx=center[0], cy=center[1], cz=center[2],
        length=length, config=config,
    )
    mask, q = compute_q_suboff(
        nx, ny, nz, *center, length,
        hull_type="full", config=config, solid_mask=solid,
    )

    count = apply_suboff_appendage_halfway_links(
        solid, mask, q, center=center, length=length, config=config,
    )

    assert count > 0
    assert int((q == 0.5).sum()) >= count


def test_appendage_link_contract_validates_shapes() -> None:
    solid = torch.zeros((8, 8, 8), dtype=torch.bool)
    mask = torch.zeros((19, 8, 8, 8), dtype=torch.bool)
    q = torch.zeros((19, 8, 8, 7))
    try:
        apply_suboff_appendage_halfway_links(
            solid, mask, q, center=(4.0, 4.0, 4.0), length=4.0,
        )
    except ValueError as error:
        assert "must match" in str(error)
    else:
        raise AssertionError("shape mismatch must fail closed")
