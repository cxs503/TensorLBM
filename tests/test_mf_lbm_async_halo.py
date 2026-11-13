"""XVAL-LBM-OPENACC-ASYNC-HALO-OVERLAP: schedule equivalence + overlap claims.

Three properties, all machine-checked here (the physics-accurate comparison
against the Taylor-Green analytic solution lives in
``benchmarks/xval_mf_lbm_async_halo/run_xval.py`` and is too slow for the suite):

1. ``blocking`` / ``cpu_order`` / ``gpu_order`` produce the *same* field to machine
   precision -- the order swap moves work in time, it must not move numbers.
2. ``strip_lanes="split"`` (one async queue per face, ``Module.F90:159-164``) never
   costs more than ``"shared"`` (single queue).
3. the overlapped arms hide a predeclared >= 50 % of the exchange time that the
   halo-before-compute arm exposes, at a modelled latency of 25 us on a slab whose per-rank compute exceeds it.
"""

from __future__ import annotations

import torch

from tensorlbm.mf_lbm_async_halo import (
    MF_LBM_MODES,
    bgk_collide,
    run_mf_virtual_case,
)

TAU = 0.56
MODEL = {
    "latency_us": 25.0,
    "bandwidth_bytes_per_us": 8000.0,
    "compute_us": {"collide": 1.5e-3, "stream": 1.5e-3, "pack": 1.0e-4},
    "jitter_us": {},
}


def _field(nx: int = 16, ny: int = 8, nz: int = 8) -> torch.Tensor:
    torch.manual_seed(7)
    f = torch.rand(19, nz, ny, nx, dtype=torch.float64) + 1.0
    return f


def test_all_schedules_agree_to_machine_precision() -> None:
    full = _field()
    collide = lambda b: bgk_collide(b, TAU)  # noqa: E731
    fields = {}
    for mode in MF_LBM_MODES:
        for lanes in ("split", "shared"):
            res = run_mf_virtual_case(
                full, mode=mode, world_size=4, n_steps=5, model=MODEL,
                collide_fn=collide, strip_width=2, strip_lanes=lanes,
            )
            fields[(mode, lanes)] = torch.cat(res["final"], dim=-1)
            assert res["max_abs_error"] <= 1e-12, (mode, lanes, res["max_abs_error"])
    ref = fields[("blocking", "split")]
    for key, got in fields.items():
        assert float((got - ref).abs().max()) <= 1e-12, key


def test_split_lanes_never_slower_than_shared() -> None:
    full = _field()
    collide = lambda b: bgk_collide(b, TAU)  # noqa: E731
    times = {}
    for lanes in ("split", "shared"):
        res = run_mf_virtual_case(
            full, mode="gpu_order", world_size=4, n_steps=5, model=MODEL,
            collide_fn=collide, strip_width=1, strip_lanes=lanes,
        )
        times[lanes] = res["clock_us"]
    assert times["split"] <= times["shared"] + 1e-9


def test_overlap_hides_half_the_exposed_exchange() -> None:
    full = _field(nx=32, ny=64, nz=32)  # per-rank compute > modelled latency
    collide = lambda b: bgk_collide(b, TAU)  # noqa: E731
    rows = {}
    for mode in ("blocking", "cpu_order", "gpu_order"):
        rows[mode] = run_mf_virtual_case(
            full, mode=mode, world_size=4, n_steps=5, model=MODEL,
            collide_fn=collide, strip_width=1, strip_lanes="split",
        )
    exposed = rows["blocking"]["clock_us"] - rows["blocking"]["compute_us"]
    assert exposed > 0.0
    for mode in ("cpu_order", "gpu_order"):
        hidden = (rows["blocking"]["clock_us"] - rows[mode]["clock_us"]) / exposed
        assert hidden >= 0.5, (mode, hidden)


def test_rejects_unknown_mode() -> None:
    from tensorlbm.comm_overlap.backend import VirtualClockBackend

    backend = VirtualClockBackend(supplier=lambda name, buf: None)
    try:
        __import__("tensorlbm.mf_lbm_async_halo", fromlist=["x"]).D3Q19MFAsyncTransport(
            backend, mode="overlap"
        )
    except ValueError:
        return
    raise AssertionError("unknown mode must raise")