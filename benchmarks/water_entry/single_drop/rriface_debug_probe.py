"""DE-X-RELX3-RRIFACE: debug probe module — wraps the real probe and dumps the
mid-plane fill/flags at a target step (for the cavity-connectivity check)."""

from __future__ import annotations

import numpy as np
import torch

import benchmarks.water_entry.single_drop.free_surface_probe as _real

TARGET = 886  # rr_iface peak step (10.774ms / dt)


def probe(*args, **kwargs):
    captured = {}
    orig = _real.free_surface_step
    calls = {"n": 0}

    def spy(f, fill, flags, solid, **kw):
        calls["n"] += 1
        if calls["n"] == TARGET:
            captured["fill"] = fill.clone().cpu().numpy()
            captured["flags"] = flags.clone().cpu().numpy()
        return orig(f, fill, flags, solid, **kw)

    _real.free_surface_step = spy
    try:
        result = _real.probe(*args, **kwargs)
    finally:
        _real.free_surface_step = orig

    if captured:
        np.savez(
            "/out/plane_dump.npz",
            fill=captured["fill"],
            flags=captured["flags"],
            step=TARGET,
        )
        print(f"PLANE_DUMP step={TARGET} saved", flush=True)
    return result
