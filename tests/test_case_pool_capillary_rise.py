"""Smoke tests for the CASE-MCMP-CAPILLARY-RISE case-pool driver.

No solver run here (the case driver is exercised end-to-end by
``benchmarks/case_pool/capillary_rise/sc_closed_box.py``): these tests pin the
geometry/mask/measurement chain that produced the reported numbers.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "benchmarks" / "case_pool" / "capillary_rise"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_spec = importlib.util.spec_from_file_location(
    "case_capillary_rise", REPO / "benchmarks" / "case_pool" / "capillary_rise" / "sc_closed_box.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)  # type: ignore[union-attr]


def test_geometry_is_closed_and_channel_is_open_at_top():
    solid, bands = M.geometry(*M.CFG["shape"])
    ny, nx = solid.shape
    # closed box: all four rims are solid
    assert solid[0].all() and solid[-1].all() and solid[:, 0].all() and solid[:, -1].all()
    lo, hi = bands["channel_inner"]
    assert hi - lo == M.CFG["gap"]
    # the channel columns are fluid all the way to the top plate (gas can pass)
    top_wall = ny - M.CFG["wall"]
    assert not solid[top_wall - 1, (lo + hi) // 2]
    # the baffles stop short of the top plate
    xl = lo - M.CFG["baffle"]
    xr = hi
    assert solid[M.CFG["pool"], xl] and not solid[top_wall - 1, xl]
    assert solid[M.CFG["pool"], xr] and not solid[top_wall - 1, xr]
    # pool measurement bands are outside the channel
    for lo_b, hi_b in bands["pool_bands"]:
        assert hi_b <= xl or lo_b >= xr + M.CFG["baffle"]


def test_initial_rho_has_liquid_below_pool_and_gas_above():
    solid, _ = M.geometry(*M.CFG["shape"])
    rho = M.initial_rho(solid, M.CFG["pool"], 1.957, 0.1596)
    fluid = ~solid
    below = rho[fluid][: 4]  # rows 2..5 are liquid
    assert float(below.mean()) > 1.9
    high = rho[-(M.CFG["wall"] + 6):-M.CFG["wall"]][fluid[-(M.CFG["wall"] + 6):-M.CFG["wall"]]]
    assert float(high.mean()) < 0.17


def test_meniscus_height_interpolates_between_nodes():
    from tensorlbm.wetting.metrics import meniscus_height

    nh = np.zeros((40, 8))
    nh[:20, :] = 1.0
    nh[20, :] = 0.75          # linear crossing between 0.75 and 0.0 -> 1/3 below
    h, sd = meniscus_height(nh, 1, 7, 0.5)
    assert abs(h - (20 + (0.75 - 0.5) / 0.75)) < 1e-9
    assert sd == 0.0


def test_jurin_reference_scales_as_inverse_gravity_and_width():
    sigma, drho, w = 0.05611, 1.797, 16.0
    h = lambda g: 2.0 * sigma * math.cos(math.radians(60.0)) / (drho * g * w)
    assert abs(h(1e-4) / h(2e-4) - 2.0) < 1e-12          # h ~ 1/g
    assert abs(h(1e-4) * 16.0 / (h(1e-4) * 8.0) - 2.0) < 1e-12   # h ~ 1/w
