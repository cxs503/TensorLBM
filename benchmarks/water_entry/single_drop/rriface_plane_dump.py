"""DE-X-RELX3-RRIFACE: dump the mid-plane fill at a late step to inspect the
cavity connectivity (debug for the cavity cross-section probe)."""

import numpy as np
import torch

import benchmarks.water_entry.single_drop.free_surface_probe as P

torch.set_num_threads(7)
torch.manual_seed(0)

TARGET = 1100
captured = {}
orig = P.free_surface_step
calls = {"n": 0}


def spy(f, fill, flags, solid, **kw):
    calls["n"] += 1
    if calls["n"] == TARGET:
        captured["fill"] = fill.clone()
        captured["flags"] = flags.clone()
    return orig(f, fill, flags, solid, **kw)


P.free_surface_step = spy
try:
    P.probe(
        8,
        0.06,
        TARGET,
        0.18,
        capacity=True,
        nearest=False,
        pilot=False,
        span=4.0,
        pool_depth=3.0,
        simple=True,
        release=1.53,
        planes_path=None,
        silhouette_path=None,
        ledger=False,
        diag_path=None,
        u_limit=0.15,
        cav_clos_alpha=0.0,
        rr_tau=0.0,
        rr_iface=True,
    )
finally:
    P.free_surface_step = orig

fill = captured["fill"].numpy()
flags = captured["flags"].numpy()
field = fill[1:-1]
plane = field[:, 16, :]
print("plane shape:", plane.shape)
# ASCII render: # = liquid (fill>0.5), . = gas, o = interface
for k in range(0, plane.shape[0], 2):
    row = ""
    for j in range(0, plane.shape[1], 1):
        v = plane[k, j]
        row += "#" if v > 0.5 else ("o" if v > 0.0 else ".")
    print(f"z={k + 1:3d} {row}")
