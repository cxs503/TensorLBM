"""RED contract for future rank-local D3Q19 Gloo checkpoint/restart hooks."""

import pytest
import torch

from tensorlbm.multi_gpu import D3Q19GlooTransport


@pytest.mark.xfail(
    strict=True,
    reason="checkpoint/restart is the next milestone; D3Q19GlooTransport has no rank-local hook yet",
)
def test_gloo_transport_checkpoint_restart_hook_roundtrips_owned_state(tmp_path) -> None:
    """Specify the minimal future API without claiming checkpoint support exists."""
    owned = torch.randn(19, 2, 3, 3, dtype=torch.float64)
    checkpoint = tmp_path / "rank-local.pt"
    D3Q19GlooTransport.save_checkpoint(checkpoint, owned, step=2)
    restored, step = D3Q19GlooTransport.load_checkpoint(checkpoint)
    assert step == 2
    assert torch.equal(restored, owned)
