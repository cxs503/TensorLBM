"""Tests for the quasi-static pc ladder + checkpoint protocol (pc_ladder)."""
from __future__ import annotations

import math
import os

import pytest
import torch

from tensorlbm import cac_lbm2d as cac
from tensorlbm import pc_ladder as P

torch.set_num_threads(1)

CFG = dict(rho_l=1.0, rho_g=0.1, mu_l=0.2, mu_g=0.02, sigma=0.05, width=3.0,
           mobility=0.05, mu_law="harmonic", p0=0.01, gravity=(0.0, 0.0),
           gravity_mode="rho", axes="pp")


def test_schedule_oss_step_equals_entry_pressure():
    """Upstream step dRho = 6 sigma cos(theta)/r_min with p = rho/3 -> dP = 2 sigma cos/r."""
    p3 = P.pc_stair_schedule(sigma=0.1, cos_theta=0.5, r_min=4.0, num_pc_steps=3, dim=3, span="oss")
    assert p3 == pytest.approx([0.025, 0.05, 0.075])
    # 2D single-curvature half of it, same total range, subdivided
    p2 = P.pc_stair_schedule(sigma=0.1, cos_theta=0.5, r_min=4.0, num_pc_steps=3, dim=2, span="full")
    assert p2 == pytest.approx([0.0125 / 3, 0.025 / 3, 0.0375 / 3])
    with pytest.raises(ValueError):
        P.pc_stair_schedule(sigma=0.1, cos_theta=0.5, r_min=0.0, num_pc_steps=2)
    with pytest.raises(ValueError):
        P.pc_stair_schedule(sigma=0.1, cos_theta=0.5, r_min=1.0, num_pc_steps=2, span="nope")


def test_curvature_of_static_droplet():
    """kappa = -div(grad phi/|grad phi|) must recover 1/R of a diffuse drop."""
    shape = (48, 48)
    R0 = 14.0
    sim = cac.CAC2D(shape, **CFG)
    sim.initialize(cac.init_phi_droplet(shape, R0, CFG["width"], center=(23.5, 23.5)))
    cac.run(sim, 200, sample_every=0)
    phi = sim.f.sum(dim=0)
    kap = P.curvature_field(phi)
    band = (phi > 0.3) & (phi < 0.7)
    k = float(kap[band].mean())
    Req = math.sqrt(float((phi > 0.5).sum().item()) / math.pi)
    assert abs(k - 1.0 / Req) / (1.0 / Req) < 0.05


def test_taper_has_horizontal_faces_only():
    ny, nx = 44, 128
    mask, r_rows = P.tapered_channel(ny, nx, wall_t=2, r_wide=20.0, r_min=8.0,
                                     throat_center=78.0, throat_half_length=16.0)
    assert int(r_rows.min()) == 8 and int(r_rows.max()) == 20
    # the quantised taper has vertical faces ONLY as unit staircase steps:
    # no column may differ from its neighbour by more than one row of half-width
    dr = (r_rows[1:] - r_rows[:-1]).abs()
    assert int(dr.max().item()) <= 1
    solid, fluid = mask, ~mask
    x_face = solid[:, :-1] & fluid[:, 1:]
    wall_cells = int(mask.sum().item())
    assert int(x_face.sum().item()) < 0.1 * wall_cells


def test_ladder_checkpoint_roundtrip_and_resume(tmp_path):
    """Complete-state checkpoint: resume reproduces the continuous trajectory exactly."""
    shape = (44, 64)
    mask, _ = P.tapered_channel(44, 64, wall_t=2, r_wide=16.0, r_min=9.0,
                                throat_center=40.0, throat_half_length=8.0)

    def build():
        sim = cac.CAC2D(shape, wall_mask=mask, wall_phi_policy="neumann", **CFG)
        sim.initialize(P.plug_phi(shape, width=CFG["width"], x_lo=8.0, x_hi=28.0))
        return sim

    params = P.LadderParams(conv=1e-12, it_conv=5, it_max=5, min_iters=0)
    pressures = [1e-3, 2e-3]
    ck = tmp_path / "ck"
    recs = P.run_ladder(build, pressures=pressures, delta_rho=0.9, plug_length=20.0,
                        params=params, checkpoint_dir=ck)
    assert [r.rung for r in recs] == [0, 1, 2]
    assert len(recs[1].checkpoint["state_sha256"]) == 64

    sim = build()
    meta = P.load_ladder_state(ck / "rung_001", sim)
    assert meta["rung"] == 1
    # tamper detection
    pt = sorted((ck / "rung_001").glob("*.pt"))[0]
    t = torch.load(pt, map_location="cpu")
    t[0] = t[0] + 1.0
    torch.save(t, pt)
    with pytest.raises(ValueError):
        P.load_ladder_state(ck / "rung_001", build())
