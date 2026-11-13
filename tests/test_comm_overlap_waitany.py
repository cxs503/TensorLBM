"""Schedule contracts for the TCLB wait-any overlap port (virtual-clock arm).

No MPI/process group needed: the modelled backend makes the overlap ratios
deterministic, so the assertions below are exact rather than tolerance-lawndered.
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.comm_overlap import D3Q19OverlapTransport, VirtualClockBackend
from tensorlbm.comm_overlap.virtual_case import run_virtual_case, slab_cuts

MODEL = {
    "latency_us": 5.0,
    "bandwidth_bytes_per_us": 32.0,
    "compute_us": {"collide": 2.0, "stream": 2.0, "pack": 4.0},
}


def collide_stub(f: torch.Tensor) -> torch.Tensor:
    """Pointwise synthetic collision (fixture, not platform physics)."""
    return f * 1.5 - 0.25


def _case(nx: int = 16, world_size: int = 3, nz: int = 2, ny: int = 3) -> torch.Tensor:
    torch.manual_seed(20261113)
    return torch.randn(19, nz, ny, nx, dtype=torch.float64)


@pytest.mark.parametrize("mode", ["blocking", "waitall", "waitany"])
@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_all_arms_are_bit_exact_versus_monolithic_reference(mode: str, world_size: int) -> None:
    result = run_virtual_case(
        _case(nx=16), mode=mode, world_size=world_size, n_steps=4, model=MODEL,
        collide_fn=collide_stub,
    )
    assert result["mismatch"] == 0
    assert result["max_abs_error"] == 0.0
    assert all(row["mismatch"] == 0 for row in result["steps"])


def test_pipelined_prefix_posts_every_request_before_the_interior_compute() -> None:
    """TCLB posts all Irecv/Isend, then runs the interior kernel under them."""
    result = run_virtual_case(_case(), mode="waitany", world_size=3, n_steps=1, model=MODEL,
                              collide_fn=collide_stub)
    assert result["mismatch"] == 0
    backend = VirtualClockBackend(supplier=lambda name, buf: None, **MODEL)
    transport = D3Q19OverlapTransport(backend, mode="waitany")
    ranks = _case()[..., :5].clone().contiguous()
    transport.step(ranks, collide_stub)
    events = backend.stats.events
    kinds = [kind for _t, kind, _n in events]
    last_post = max(i for i, (t, k, _n) in enumerate(events) if k == "post_recv")
    # the interior block is the only 'compute'/'stream' event before the first wait
    first_wait = min(i for i, (t, k, _n) in enumerate(events) if k == "waitany_done")
    interior_events = [i for i, (t, k, n) in enumerate(events) if k == "compute" and n == "stream"]
    assert interior_events, "interior kernel never ran"
    assert last_post < min(interior_events) < first_wait
    assert kinds.count("post_recv") == 2, "buffer count must equal the number of real neighbours"
    assert backend.stats.waitany_calls == 2
    assert backend.stats.waitall_calls == 1  # the send side only


def test_waitall_arm_never_uses_waitany_and_waits_once_for_all_receives() -> None:
    backend = VirtualClockBackend(supplier=lambda name, buf: None, **MODEL)
    transport = D3Q19OverlapTransport(backend, mode="waitall")
    transport.step(_case()[..., :5].clone().contiguous(), collide_stub)
    assert backend.stats.waitany_calls == 0
    assert backend.stats.waitall_calls == 2
    assert backend.stats.posts_recv == 2


def test_blocking_arm_is_the_shipped_waitall_everything_schedule() -> None:
    backend = VirtualClockBackend(supplier=lambda name, buf: None, **MODEL)
    transport = D3Q19OverlapTransport(backend, mode="blocking")
    transport.step(_case()[..., :5].clone().contiguous(), collide_stub)
    assert backend.stats.waitany_calls == 0
    kinds = [k for _t, k, _n in backend.stats.events]
    # no wait happens before the whole-slab collide + pack
    first_wait = kinds.index("post_recv")
    assert kinds.index("compute") < first_wait


@pytest.mark.parametrize("arm", ["waitall", "waitany"])
def test_stall_equals_the_analytic_overlap_bound(arm: str) -> None:
    """Pipelined stall must equal max(0, T_comm - T_interior) exactly."""
    nx, world_size, nz, ny = 12, 3, 2, 3
    payload_bytes = 19 * nz * ny * 8
    t_comm = MODEL["latency_us"] + payload_bytes / MODEL["bandwidth_bytes_per_us"]
    t_interior = (nx // world_size - 2) * MODEL["compute_us"]["stream"]
    expected = max(0.0, t_comm - t_interior)
    result = run_virtual_case(_case(nx=nx, nz=nz, ny=ny), mode=arm, world_size=world_size,
                              n_steps=3, model=MODEL, collide_fn=collide_stub)
    assert result["mismatch"] == 0
    assert result["stall_us"] == pytest.approx(expected, rel=0, abs=1e-9)
    # and the blocked arm is exactly the full exposed transfer
    blocked = run_virtual_case(_case(nx=nx, nz=nz, ny=ny), mode="blocking",
                               world_size=world_size, n_steps=3, model=MODEL,
                               collide_fn=collide_stub)
    assert blocked["stall_us"] == pytest.approx(t_comm, rel=0, abs=1e-9)
    assert blocked["stall_us"] > result["stall_us"]


@pytest.mark.parametrize("jitter_us,expected", [(4.0, 2.0), (12.0, 2.0), (1.0, 1.0)])
def test_waitany_consumes_an_early_arrival_while_a_late_one_is_in_flight(
    jitter_us: float, expected: float
) -> None:
    """Gain over ``waitall`` is min(arrival spread, per-buffer consume cost)."""
    model = dict(MODEL)
    model["jitter_us"] = {"left": 0.0, "right": jitter_us}
    field = _case(nx=12)  # uniform 4/4/4 slabs: every rank sees the same timeline
    waitany = run_virtual_case(field, mode="waitany", world_size=3, n_steps=3, model=model,
                               collide_fn=collide_stub)
    waitall = run_virtual_case(field, mode="waitall", world_size=3, n_steps=3, model=model,
                               collide_fn=collide_stub)
    assert waitany["mismatch"] == waitall["mismatch"] == 0
    benefit = waitall["stall_us"] - waitany["stall_us"]
    assert benefit == pytest.approx(expected, rel=0, abs=1e-9)
    # the schedule, not the network, is what changed: identical compute bill
    assert waitany["compute_us"] == waitall["compute_us"] == waitany["compute_us"]


def test_slab_cuts_are_non_uniform_and_cover_the_domain() -> None:
    cuts = slab_cuts(16, 3)
    assert cuts == [0, 6, 11, 16]
    assert all(cuts[i + 1] - cuts[i] >= 3 for i in range(3))


def test_narrow_slabs_are_rejected_with_guidance() -> None:
    backend = VirtualClockBackend(supplier=lambda name, buf: None, **MODEL)
    transport = D3Q19OverlapTransport(backend, mode="waitany")
    with pytest.raises(ValueError, match="slab width"):
        transport.step(torch.zeros(19, 2, 2, 2), collide_stub)