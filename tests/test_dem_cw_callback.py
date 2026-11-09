"""TensorDEM ice/water drag closure callback (mission P103W2C-N01, criterion C6).

The criterion is end-to-end: the DEM solver must accept the roughness-dependent
water-drag coefficient ``C_w`` through a callback, the same particle set must be
runnable with the constant entry, the closure and the classical
Schiller-Naumann model by switching one configuration field, and the momentum
the solver takes from the water must equal the momentum it gives to the
particles to better than ``1e-10`` relative.

The momentum check is deliberately *not* trivially zero: the water reaction is
recomputed by :class:`tensorlbm.dem.cw_coupling.MomentumExchange` from the
recorded drag states, so corrupting one recorded ``C_w`` by 1 % must raise the
residual by ~1e-2 (negative control below).
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.dem import DEMConfig, DEMSolver, IceParticle
from tensorlbm.dem import cw_coupling as cc
from tensorlbm.ice_roughness import propagation as pg
from tensorlbm.ice_roughness import regime_law as rl


def _particles():
    zero = torch.zeros(3, dtype=torch.float64)
    return [
        IceParticle(position=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
                    velocity=torch.tensor([0.10, 0.00, 0.00], dtype=torch.float64),
                    angular_velocity=zero.clone(), radius=0.05, density=917.0),
        IceParticle(position=torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64),
                    velocity=torch.tensor([-0.05, 0.02, 0.0], dtype=torch.float64),
                    angular_velocity=zero.clone(), radius=0.05, density=917.0),
    ]


def _run(drag_model="cw_callback", callback=None, *, k_s_m=0.35, steps=5):
    cfg = DEMConfig(
        time_step=1e-3, gravity=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
        drag_model=drag_model, fluid_density=1025.0, fluid_viscosity=1.83e-3,
        enable_wall_contacts=False, cw_callback=callback,
        ks_m=k_s_m, lambda_over_ks=5.0)
    solver = DEMSolver(cfg)
    solver.exchange = cc.MomentumExchange(rho_water=1025.0)
    solver.initialize(_particles())
    fluid = torch.zeros((2, 3), dtype=torch.float64)
    for _ in range(steps):
        solver.step(fluid)
    return solver


def _law(k0=3.6, lam=5.0):
    return rl.ThreeZoneLaw(lambda_over_ks=lam, k0=k0, k0_source="test")


# ---------------------------------------------------------------------------
# callback implementations
# ---------------------------------------------------------------------------

def test_constant_callback_reproduces_the_m5682_entry():
    cb = cc.ConstantCwCallback(pg.M5682_CONSTANT_CW)
    res = cb(cc.IceWaterDragState(relative_speed=0.1, area=1.0))
    assert res.c_w == pytest.approx(5.5e-3)
    assert len(cb.calls) == 1


def test_roughness_callback_is_velocity_independent_in_the_fully_rough_zone():
    """In the fully rough limit ``C_w`` must not depend on the drift speed.

    ``dU^+ = (1/kappa) ln(k_s^+/k0)`` cancels the ``ln z^+`` of the log law, so
    the roughness closure becomes a pure roughness law: ``C_w`` is set by
    ``k_s`` alone (this is exactly why a roughness closure is usable in a drift
    model).  Only the finite ``ln(1 + k_s^+/k0)`` blend leaves a weak residual
    velocity dependence.
    """
    cb = cc.RoughnessCwCallback(_law(), k_s_m=0.35)
    slow = cb(cc.IceWaterDragState(relative_speed=0.05, area=1.0))
    fast = cb(cc.IceWaterDragState(relative_speed=0.30, area=1.0))
    assert slow.regime == fast.regime == "fully_rough"
    assert fast.c_w == pytest.approx(slow.c_w, rel=0.02)
    assert fast.ks_plus > slow.ks_plus
    assert slow.u_tau < fast.u_tau


def test_rougher_underside_gives_more_drag_at_fixed_speed():
    smooth = cc.RoughnessCwCallback(_law(), k_s_m=1e-4)
    rough = cc.RoughnessCwCallback(_law(), k_s_m=0.35)
    state = cc.IceWaterDragState(relative_speed=0.15, area=1.0)
    assert rough(state).c_w > smooth(state).c_w


def test_roughness_callback_requires_a_positive_roughness():
    with pytest.raises(ValueError):
        cc.RoughnessCwCallback(_law(), k_s_m=0.0)


def test_tabulated_callback_interpolates_the_table_nodes():
    laws = {2.0: _law(lam=2.0), 5.0: _law(lam=5.0), 10.0: _law(lam=10.0)}
    table = pg.cw_table(laws, ks_m=(0.05, 0.20, 0.50), lambda_over_ks=(5.0,))
    cb = cc.TabulatedCwCallback(table, k_s_m=0.20, lambda_over_ks=5.0)
    res = cb(cc.IceWaterDragState(relative_speed=0.10, area=1.0))
    assert res.c_w > 0.0
    assert res.source == "tabulated"
    # the self-consistent k_s^+ of the callback must be a fixed point of u_tau
    assert res.u_tau == pytest.approx(math.sqrt(res.c_w) * 0.10, rel=1e-6)
    assert 0.0 < res.ks_plus < 1e5


def test_tabulated_callback_flags_extrapolation_outside_the_table():
    law = _law()
    table = pg.cw_table({5.0: law}, ks_m=(0.50,), lambda_over_ks=(5.0,))
    # a huge relative speed drives k_s^+ far above the tabulated node
    cb = cc.TabulatedCwCallback(table, k_s_m=5.0, lambda_over_ks=5.0)
    cb(cc.IceWaterDragState(relative_speed=5.0, area=1.0))
    assert cb.calls[-1]["extrapolated"] is True


def test_empty_table_is_rejected():
    with pytest.raises(ValueError):
        cc.TabulatedCwCallback([], k_s_m=0.1)


# ---------------------------------------------------------------------------
# solver integration
# ---------------------------------------------------------------------------

def test_solver_requires_a_callback_for_the_cw_model():
    solver = _run.__wrapped__ if hasattr(_run, "__wrapped__") else None
    assert solver is None  # keep the helper opaque; the check is below
    cfg = DEMConfig(drag_model="cw_callback", enable_wall_contacts=False)
    s = DEMSolver(cfg)
    s.initialize(_particles())
    with pytest.raises(ValueError):
        s.step(torch.zeros((2, 3), dtype=torch.float64))


def test_solver_is_switchable_between_drag_models():
    constant = _run(callback=cc.ConstantCwCallback(5.5e-3))
    closure = _run(callback=cc.RoughnessCwCallback(_law(), k_s_m=0.35))
    naumann = _run(drag_model="schiller_naumann")
    v_const = [tuple(float(c) for c in p.velocity) for p in constant.system]
    v_close = [tuple(float(c) for c in p.velocity) for p in closure.system]
    v_naum = [tuple(float(c) for c in p.velocity) for p in naumann.system]
    assert v_const != v_close
    assert v_const != v_naum
    assert len(constant.drag_events) == len(closure.drag_events) == 2 * 5
    assert naumann.drag_events == []      # only the callback model records states


def test_momentum_exchange_residual_is_at_roundoff():
    solver = _run(callback=cc.RoughnessCwCallback(_law(), k_s_m=0.35))
    res = solver.exchange.residual()
    assert res["n_events"] == 10
    assert res["relative_residual"] < 1e-10
    assert res["scale"] > 0.0


def test_momentum_residual_detects_a_corrupted_exchange():
    """Negative control: the metric must be sensitive to a 1 % bookkeeping error."""
    solver = _run(callback=cc.RoughnessCwCallback(_law(), k_s_m=0.35), steps=1)
    ledger = solver.exchange
    ledger._states[0]["c_w"] *= 1.01          # corrupt one recorded drag coefficient
    res = ledger.residual()
    assert res["relative_residual"] > 1e-3


def test_tabulated_callback_through_the_solver():
    laws = {2.0: _law(lam=2.0), 5.0: _law(lam=5.0), 10.0: _law(lam=10.0)}
    table = pg.cw_table(laws, ks_m=(0.05, 0.20, 0.50), lambda_over_ks=(5.0,))
    cb = cc.TabulatedCwCallback(table, k_s_m=0.20, lambda_over_ks=5.0)
    solver = _run(callback=cb)
    assert len(cb.calls) == 10
    assert solver.exchange.residual()["relative_residual"] < 1e-10


def test_drag_force_form_matches_the_other_models():
    """``cw_callback`` uses the same ``0.5 rho C A |u| u`` form as the built-ins."""
    c_w = 8.0e-3
    solver = _run(callback=cc.ConstantCwCallback(c_w), steps=1)
    ev = solver.drag_events[0]
    rho, area = 1025.0, ev["area"]
    u = ev["relative_velocity"]
    speed = math.sqrt(sum(v * v for v in u))
    expected = [0.5 * c_w * rho * area * speed * v for v in u]
    assert ev["force"] == pytest.approx(expected, rel=1e-12)


def test_ks_and_lambda_reach_the_callback_from_the_config():
    seen = []

    def cb(state):
        seen.append((state.ks_m, state.lambda_over_ks, state.area))
        return 5.0e-3

    _run(callback=cb, k_s_m=0.42, steps=1)
    assert seen
    assert all(k == pytest.approx(0.42) and lam == pytest.approx(5.0) for k, lam, _ in seen)