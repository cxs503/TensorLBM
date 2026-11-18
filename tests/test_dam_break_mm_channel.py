"""Published clocks, coarse/fine eligibility and free-surface MRT force."""

import math

import numpy as np
import pytest
import torch

from benchmarks.pending.dam_break_mm_channel.run import measure
from benchmarks.pending.dam_break_mm_channel.score import ladder, read_reference, score_run
from tensorlbm.d3q19 import W_EXACT64, C, equilibrium3d, macroscopic3d
from tensorlbm.free_surface_lbm import LIQUID, free_surface_step
from tensorlbm.solver3d import _get_d3q19_mrt_matrices


@pytest.mark.parametrize("tau,cs", [(0.5001, 0.0), (1.0, 0.0), (0.7, 0.12)])
def test_uniform_mrt_impulse(tau, cs):
    rho = torch.ones((4, 5, 6), dtype=torch.float64)
    u = [rho * v for v in (0.012, -0.018, 0.006)]
    accel = (1e-5, -2e-5, 3e-5)
    f = equilibrium3d(rho, *u)
    actual, *_ = free_surface_step(
        f,
        torch.ones_like(rho),
        torch.full_like(rho, LIQUID, dtype=torch.int8),
        torch.zeros_like(rho, dtype=torch.bool),
        mass=rho.clone(),
        tau=tau,
        gx=accel[0],
        gy=accel[1],
        gz=accel[2],
        collision="mrt",
        C_s=cs,
    )
    current, *velocity = macroscopic3d(actual)
    torch.testing.assert_close(current, rho, rtol=0, atol=2e-13)
    for before, after, g in zip(u, velocity, accel):
        torch.testing.assert_close(after, before + g, rtol=0, atol=2e-13)
    if cs == 0:
        # Direct (I-S/2) moment-space reference checks higher moments too.
        m, inv = _get_d3q19_mrt_matrices(rho.device, rho.dtype)
        rates = rho.new_tensor(
            [0, 1.19, 1.4, 0, 1.2, 0, 1.2, 0, 1.2, *([1 / tau] * 5), 1.19, 1.19, 1, 1, 1]
        )
        c = C.to(rho)
        um = torch.stack([v[0, 0, 0] + g / 2 for v, g in zip(u, accel)])
        force = rho.new_tensor(accel)
        phi = W_EXACT64 * (3 * ((c - um) @ force) + 9 * (c @ um) * (c @ force))
        pop = f[:, 0, 0, 0]
        eq = equilibrium3d(rho, *(v + g / 2 for v, g in zip(u, accel)))[:, 0, 0, 0]
        expected = inv @ ((m @ pop) - rates * (m @ (pop - eq)) + (1 - rates / 2) * (m @ phi))
        torch.testing.assert_close(actual[:, 0, 0, 0], expected, rtol=0, atol=2e-13)


def test_connected_front_and_crest_exclude_detached_droplet():
    a = np.zeros((4, 12, 20))
    a[:, :8, :4] = 1
    a[:, 0, 4] = 0.25
    a[0, 9, 15] = 1
    front, crest = measure(a, 4)
    assert front == pytest.approx((3.5 + 0.5 / 0.75) / 4)
    assert crest == 1


def test_published_clocks_anchor_and_no_extrapolation():
    ref = read_reference()
    assert sum(k == "front" for k, _, _ in ref) == 25
    assert sum(k == "crest" for k, _, _ in ref) == 17
    keys = "step,T,Z,H,mass_drift,volume_drift,mach,rho_min"
    data = np.zeros(301, dtype=[(k, float) for k in keys.split(",")])
    data["step"] = np.arange(len(data))
    data["T"] = 0.01 * data["step"]
    data["Z"], data["H"] = 1 + 0.4 * data["T"], 1 - 0.1 * data["T"]
    data["rho_min"] = 1
    meta = dict(last_step=300, c0=0.06, n=6, span="channel", collision="mrt", stop="partial")
    row = score_run(meta, data, ref)
    assert row["first_T_at_Z144"] == pytest.approx(1.1)
    assert row["front_time_shift"] == pytest.approx(0.09)
    assert row["curves"]["front"]["required"] == 24
    assert row["points"][2]["anchor"]
    assert row["points"][25]["prediction"] == pytest.approx(1 - 0.1 * 0.56 * math.sqrt(2))
    assert row["points"][-1]["prediction"] is None
    assert row["curves"]["front"]["covered"] < 24
    assert not ladder({"partial": row}, "mrt")["passed"]


def test_admission_requires_both_grids_conservation_and_all_points():
    import copy

    def candidate(n, error):
        return {
            "config": {"n": n, "collision": "mrt", "c0": 0.06, "span": "channel"},
            "points": [{"anchor": False, "error_pct": error} for _ in range(41)],
            "curves": {
                k: {"within_3pct": count, "required": count}
                for k, count in (("front", 24), ("crest", 17))
            },
            "health_pass": True,
            "conservation_pass": True,
        }

    runs = {"coarse": candidate(8, 2.0), "fine": candidate(12, 1.0)}
    assert ladder(runs, "mrt")["passed"]
    for key in ("health_pass", "conservation_pass"):
        bad = copy.deepcopy(runs)
        bad["fine"][key] = False
        assert not ladder(bad, "mrt")["passed"]
    bad = copy.deepcopy(runs)
    bad["coarse"]["curves"]["front"]["within_3pct"] -= 1
    assert not ladder(bad, "mrt")["passed"]
    runs["fine"]["points"][-1]["error_pct"] = None
    assert not ladder(runs, "mrt")["passed"]
