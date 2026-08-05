"""Regression coverage for the D3Q19 x-slab pull-stream ordering."""

import pytest
import torch

from tensorlbm.multi_gpu import DomainDecomposition, MultiGPUSolver3D


# (cx, cy, cz), in the conventional D3Q19 order.  The test deliberately
# populates every population independently: comparing only density could hide
# direction-specific halo mistakes.
D3Q19_C = (
    (0, 0, 0),
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
    (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
    (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1),
)


def _d3q19_pull_stream(f: torch.Tensor) -> torch.Tensor:
    """Periodic pull stream, expressed as per-population tensor rolls."""
    out = torch.empty_like(f)
    for q, (cx, cy, cz) in enumerate(D3Q19_C):
        out[q] = torch.roll(f[q], shifts=(cz, cy, cx), dims=(0, 1, 2))
    return out


@pytest.mark.parametrize("n_steps", (1, 2, 3))
def test_two_x_slabs_match_monolithic_d3q19_for_every_owned_population(n_steps: int) -> None:
    """Halo values must be exchanged before—not after—the local pull stream."""
    torch.manual_seed(20260714)
    initial = torch.randn(19, 3, 4, 8, dtype=torch.float64)
    decomp = DomainDecomposition(devices=["cpu", "cpu"], nx_global=initial.shape[-1])
    solver = MultiGPUSolver3D(initial, decomp)

    expected = initial.clone()
    for _ in range(n_steps):
        expected = _d3q19_pull_stream(expected)
        solver.step(lambda f: f, _d3q19_pull_stream)

    actual = solver.gather()
    mismatch = (actual != expected).sum(dim=(1, 2, 3))
    assert torch.equal(mismatch, torch.zeros(19, dtype=mismatch.dtype)), mismatch.tolist()
