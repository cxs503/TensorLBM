"""Gate tests: XVAL-LBM-MOVING-WALL-WALL-SHEAR-ME-D2Q9-BGK.

What is pinned here (fast subset, <60 s):

1. **Driver equivalence** — the force-instrumented step is *bit-identical* to the
   merged reference closure ``benchmarks/cross_validation/moving_wall_halfway_bb_d2q9_bgk/mwbb.py::make_step``.
   Without this the whole study would verify a re-implementation, not the
   adjudicated closure.
2. **Impulse identity** (M7) — ``sum_{c_q.n>0} c_q inj_q == rho_w u_w`` cell-wise
   on random states; pins the lattice structure/weights of the Ladd term.
3. **ME bookkeeping identity** (M6) — ``F_bot + F_top + dP_fluid == 0`` to machine
   precision on a live run; catches factor-2 / normal / sampling errors in the
   momentum-exchange formula.
4. **Sign pin** — for a uniform flow between static walls the wall must *gain*
   +x momentum from the fluid (force on wall > 0), which fixes the sign
   convention used everywhere else in the study.
5. **Physics gates at reduced cost** — Couette wall shear (exact N-S value),
   Stokes-2nd force amplitude/phase, Rayleigh start-up decay, all against the
   predeclared 15 % / 15 deg standards.

Run: PYTHONPATH=src python3 -m pytest tests/test_xval_moving_wall_wall_shear_me.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

XV = Path(__file__).resolve().parents[1] / "benchmarks/cross_validation"
sys.path.insert(0, str(XV / "moving_wall_wall_shear_me_d2q9_bgk"))
sys.path.insert(0, str(XV / "moving_wall_halfway_bb_d2q9_bgk"))

import mwbb  # noqa: E402
import mwforce as mf  # noqa: E402


def test_step_is_bit_identical_to_merged_closure():
    """The instrumented step must not change the physics of the reference step."""
    ny, nx, tau = 18, 4, 0.8
    ref = mwbb.make_step(ny, nx, tau, "ladd_pre_local")
    got = mf.make_step_flux(ny, nx, tau, "ladd_pre_local")
    y = np.arange(ny, dtype=float)
    f_ref = mwbb.init_field(ny, nx, 0.05 * np.clip((y - 0.5) / (ny - 2.0), 0.0, 1.0))
    f_got = f_ref.clone()
    uw_s = mwbb.Wall.uniform(ny, nx, 0.0, 0.05).u_wall
    uw_v = mf._wall_field(ny, nx, (0.0, 0.0), (0.05, 0.0))
    for _ in range(7):
        f_ref = ref(f_ref, uw_s)
        f_got = got(f_got, uw_v)[0]
    assert torch.equal(f_ref, f_got)


def test_ladd_impulse_identity():
    """M7: the Ladd term transfers exactly rho_w u_w per wall cell."""
    ident = mf.impulse_identity()
    assert ident["rel_res"] <= 1e-14
    assert ident["all_dirs_rel_res"] <= 1e-14
    # and the vector version reduces to mwbb._inject for a purely tangential wall
    ny, nx = 6, 3
    rho = torch.ones((ny, nx), dtype=torch.float64)
    uw = torch.zeros((2, ny, nx), dtype=torch.float64)
    uw[0] = 0.037
    a = mf.inject(rho, uw)
    b = mwbb._inject(uw[0], rho)
    assert torch.allclose(a, b, atol=0.0, rtol=1e-15)


def test_me_identity_and_sign_on_live_run():
    """M6 identity + sign pin for the momentum-exchange count."""
    ny, nx, tau = 12, 4, 0.8
    step = mf.make_step_flux(ny, nx, tau)
    # (a) uniform flow, static walls -> fluid loses +x momentum, wall gains it
    f = mwbb.init_field(ny, nx, np.full(ny, 0.02))
    uw0 = mf._wall_field(ny, nx, (0.0, 0.0), (0.0, 0.0))
    f, Fb, Ft, dP, _ = step(f, uw0)
    assert float(Fb[0]) > 0.0 and float(Ft[0]) > 0.0
    assert float(dP[0]) < 0.0
    scale = max(abs(float(Fb[0])), abs(float(Ft[0])))
    assert float((Fb + Ft + dP).abs().max()) <= 1e-12 * scale
    # (b) Couette driven by the top wall: reaction on the moving wall is -x
    f = mwbb.init_field(ny, nx, 0.05 * np.clip((np.arange(ny) - 0.5) / (ny - 2.0), 0, 1))
    uw = mf._wall_field(ny, nx, (0.0, 0.0), (0.05, 0.0))
    for _ in range(200):
        f, Fb, Ft, dP, _ = step(f, uw)
        assert float((Fb + Ft + dP).abs().max()) <= 1e-12 * max(abs(float(Ft[0])), 1e-30)
    assert float(Ft[0]) < 0.0 < float(Fb[0])


@pytest.fixture(scope="module")
def couette():
    return mf.run_couette_shear(ny=18, nx=4, tau=0.8, u0=0.05, n_steps=20000, avg_steps=100)


@pytest.fixture(scope="module")
def stokes():
    return mf.run_stokes2_shear(ny=26, nx=4, tau=0.8, u0=0.02, delta_bl=4.0, n_periods=2)


@pytest.fixture(scope="module")
def rayleigh():
    return mf.run_rayleigh_shear(ny=34, nx=4, tau=0.8, u0=0.02, n_steps=600, win=(200, 500))


def test_couette_wall_shear_matches_exact_solution(couette):
    assert couette["steady"] is True
    assert abs(couette["tau_err_pct"]) <= 15.0
    assert abs(couette["tau_bot_err_pct"]) <= 15.0
    assert couette["force_balance_rel"] <= 1e-6
    # the discrete ME force is exact for the linear profile: demand much better
    assert abs(couette["tau_err_pct"]) <= 1e-3


def test_stokes2_wall_force_amplitude_and_phase(stokes):
    assert abs(stokes["amp_err_pct"]) <= 15.0
    assert abs(stokes["phase_err_deg"]) <= 15.0
    # 2 periods at delta=4 is a coarse transient; still far inside the gates
    assert abs(stokes["amp_err_pct"]) <= 5.0
    assert abs(stokes["phase_err_deg"]) <= 5.0
    # the analytic amplitude used here is the textbook rho U0 sqrt(nu omega)
    assert stokes["amp_num"] == pytest.approx(stokes["amp_closed_form"], rel=0.05)


def test_rayleigh_wall_shear_decay(rayleigh):
    assert rayleigh["err_pct_max_in_win"] <= 15.0


def test_normal_wall_velocity_is_out_of_contract():
    """D1 is a documented limitation: a normal wall velocity leaks fluid mass."""
    ctrl = mf.run_normal_wall_drift(ny=20, nx=4, tau=0.8, uw_n=0.0, n_steps=120)
    assert abs(ctrl["dm_fluid"]) <= 1e-12
    bad = mf.run_normal_wall_drift(ny=20, nx=4, tau=0.8, uw_n=0.01, n_steps=120)
    assert bad["dm_fluid"] < 0.0  # fluid mass **decreases**: not physical for a solid wall
    assert abs(bad["dm_fluid_rel_pct"]) > 0.1