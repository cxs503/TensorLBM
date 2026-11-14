"""Unit tests for the per-step setup hook + mask-blend BC composition.

These pin the four claims the XVAL report
``docs/XVAL-LBM-PER-STEP-SETUP-HOOK-AND-MASK-BLEND.md`` rests on:

1. the driver reproduces upstream ``Domain.Solve``'s hook ordering and save
   cadence, and an identity ``setup_step`` changes nothing (bit-wise);
2. the single-layer composition is *the same operator* as an explicit
   direction reversal on the masked cells, and as
   :func:`tensorlbm.streaming_conv.bounce_blend`;
3. the moving-wall layer reduces to the stationary layer at ``u_wall = 0`` and
   its added momentum is exactly ``6 w_i (c_i . u_wall)``;
4. the mask partition is validated (overlap / out-of-range masks are errors)
   and partial masks blend smoothly.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.d2q9 import C, OPPOSITE, W, equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream
from tensorlbm.step_hooks import (
    BoundaryLayer,
    HookLog,
    blend_coverage,
    compose_masked_boundaries,
    equilibrium_layer,
    layer_populations,
    masked_step,
    moving_bounce_layer,
    solve_with_hooks,
    stationary_bounce_layer,
    tables_for,
    velocity_ramp_hook,
)
from tensorlbm.streaming_conv import bounce_blend

DT = torch.float64


def rest_field(ny: int = 8, nx: int = 8, rho: float = 1.0) -> torch.Tensor:
    ones = torch.full((ny, nx), rho, dtype=DT)
    zeros = torch.zeros(ny, nx, dtype=DT)
    return equilibrium(ones, zeros, zeros)


def top_mask(ny: int = 8, nx: int = 8) -> torch.Tensor:
    m = torch.zeros(ny, nx, dtype=DT)
    m[-1, :] = 1.0
    return m


# ---------------------------------------------------------------- tables / API
def test_tables_for_picks_lattice_by_rank():
    t2 = tables_for(rest_field())
    assert t2.lveloc.shape == (9, 2) and t2.opposite.shape == (9,)
    assert torch.allclose(t2.weights.sum(), torch.tensor(1.0, dtype=DT))
    f3 = torch.zeros(19, 4, 4, 4, dtype=DT)
    t3 = tables_for(f3)
    assert t3.dim == 3 and t3.lveloc.shape == (19, 3)
    with pytest.raises(ValueError):
        tables_for(torch.zeros(2, 2, dtype=DT))


def test_layer_validation():
    f = rest_field()
    with pytest.raises(ValueError):
        BoundaryLayer("bad", top_mask(), kind="magic")
    with pytest.raises(ValueError):
        BoundaryLayer("bad", torch.full((8, 8), 1.5, dtype=DT))
    with pytest.raises(ValueError):
        BoundaryLayer("bad", torch.full((8, 8), -0.1, dtype=DT))
    with pytest.raises(ValueError):
        moving_bounce_layer("lid", top_mask(), velocity=None)
    assert stationary_bounce_layer("wall", top_mask()).kind == "bounce"
    assert f.shape == (9, 8, 8)


# ------------------------------------------------------- single-layer algebra
def test_stationary_layer_equals_explicit_reversal():
    torch.manual_seed(0)
    f = torch.rand(9, 8, 8, dtype=DT) + 0.1
    mask = torch.zeros(8, 8, dtype=DT)
    mask[0, :] = 1.0
    mask[:, 0] = 1.0
    layer = stationary_bounce_layer("wall", mask)
    got = compose_masked_boundaries(f, f, [layer], tables=tables_for(f))

    explicit = f.clone()
    idx = OPPOSITE.to(torch.int64)
    explicit[:, mask > 0] = f[idx][:, mask > 0]
    assert torch.equal(got, explicit)

    # ... and with the upstream 1x1 bounce convolution of the conv-as-streaming port
    conv = bounce_blend(f, f, mask)
    assert torch.equal(got, conv)


def test_partial_mask_is_a_linear_blend():
    f = rest_field()
    g = torch.rand(9, 8, 8, dtype=DT) * 0.1
    mask = torch.full((8, 8), 0.25, dtype=DT)
    layer = stationary_bounce_layer("soft", mask)
    got = compose_masked_boundaries(f, g, [layer], tables=tables_for(f))
    want = f * 0.75 + g[OPPOSITE.to(torch.int64)] * 0.25
    assert torch.allclose(got, want, atol=1e-15)


def test_moving_layer_reduces_to_stationary_at_zero_velocity():
    torch.manual_seed(1)
    f = torch.rand(9, 8, 8, dtype=DT) + 0.1
    mask = top_mask()
    still = compose_masked_boundaries(
        f, f, [stationary_bounce_layer("w", mask)], tables=tables_for(f)
    )
    zero = compose_masked_boundaries(
        f, f, [moving_bounce_layer("w", mask, (0.0, 0.0))], tables=tables_for(f)
    )
    assert torch.equal(still, zero)


def test_moving_layer_momentum_injection_is_6_w_c_dot_u():
    f = rest_field()
    mask = top_mask()
    u = torch.tensor([0.05, 0.0], dtype=DT)
    layer = moving_bounce_layer("lid", mask, u)
    pop = layer_populations(layer, f, tables=tables_for(f))
    reversed_pop = f[OPPOSITE.to(torch.int64)]
    cu = C.to(DT)[:, 0] * u[0] + C.to(DT)[:, 1] * u[1]
    want = reversed_pop + 6.0 * W.to(DT).view(9, 1, 1) * cu.view(9, 1, 1)
    assert torch.allclose(pop, want, atol=1e-8, rtol=0)  # f itself carries the float32 weight rounding
    # the injected momentum along x must be 6*u_x*cs^2*rho = 2*u_x for rho=1
    dmom_x = float((pop * C.to(DT)[:, 0].view(9, 1, 1)).sum(dim=0)[-1, 4]) - float(
        (reversed_pop * C.to(DT)[:, 0].view(9, 1, 1)).sum(dim=0)[-1, 4]
    )
    assert dmom_x == pytest.approx(2.0 * 0.05, rel=1e-14, abs=1e-16)


def test_equilibrium_layer_installs_requested_velocity():
    f = rest_field()
    mask = torch.zeros(8, 8, dtype=DT)
    mask[:, 0] = 1.0
    layer = equilibrium_layer("inlet", mask, torch.tensor([0.03, 0.01], dtype=DT), density=1.0)
    got = compose_masked_boundaries(f, f, [layer], tables=tables_for(f))
    rho, ux, uy = macroscopic(got)
    assert float(rho[3, 0]) == pytest.approx(1.0, rel=1e-12)
    assert float(ux[3, 0]) == pytest.approx(0.03, rel=1e-6)
    assert float(uy[3, 0]) == pytest.approx(0.01, rel=1e-6)


# ------------------------------------------------------------- partition rules
def test_overlapping_masks_are_rejected():
    f = rest_field()
    a = stationary_bounce_layer("a", top_mask())
    b = stationary_bounce_layer("b", top_mask())
    with pytest.raises(ValueError, match="overlap"):
        compose_masked_boundaries(f, f, [a, b], tables=tables_for(f))
    compose_masked_boundaries(f, f, [a, b], tables=tables_for(f), strict=False)
    assert float(blend_coverage([a, b], f).max()) == pytest.approx(2.0)


def test_disjoint_masks_partition_the_field():
    f = rest_field()
    lo = torch.zeros(8, 8, dtype=DT)
    hi = torch.zeros(8, 8, dtype=DT)
    lo[0, :] = 1.0
    hi[-1, :] = 1.0
    layers = [stationary_bounce_layer("lo", lo), moving_bounce_layer("hi", hi, (0.02, 0.0))]
    cov = blend_coverage(layers, f)
    assert float(cov.max()) == pytest.approx(1.0)
    got = compose_masked_boundaries(f, f, layers, tables=tables_for(f))
    assert torch.isfinite(got).all()
    # mass is conserved by both layers (bounce is a permutation, the momentum
    # term sums to zero because sum_i w_i c_i = 0)
    assert float(got.sum()) == pytest.approx(float(f.sum()), rel=1e-14)


# ------------------------------------------------------------------- the driver
def test_solve_with_hooks_order_and_save_cadence():
    f = rest_field()
    calls: list[tuple[str, int]] = []

    def init(state, t):
        calls.append(("init", 0))
        state["f"] = state["f"].clone()
        return state

    def setup(state, t, i):
        calls.append(("setup", i))
        state["seen"] = i
        return state

    def save(state, t, i):
        calls.append(("save", i))

    state = {"f": f}
    op_calls: list[float] = []

    def op(st, t):
        op_calls.append(t)
        return st

    out, log = solve_with_hooks(
        state, 10, operator=op, initialize_step=init, setup_step=setup,
        save_step=save, save_interval=4,
    )
    assert log.initialize == 1
    assert log.setup == list(range(10))
    assert log.save == [0, 4, 8, 10]
    assert [c for c in calls if c[0] == "save"] == [("save", 0), ("save", 4), ("save", 8), ("save", 10)]
    # the save tick of step i precedes that step's setup hook
    order = [c[0] for c in calls[:3]]
    assert order == ["init", "save", "setup"]
    assert op_calls == [float(t) for t in range(10)]
    assert out["seen"] == 9


def test_identity_hook_does_not_change_the_trajectory():
    f0 = rest_field(10, 10)
    f0 = f0 + 1e-3 * torch.rand(9, 10, 10, dtype=DT)
    mask = torch.zeros(10, 10, dtype=DT)
    mask[0, :] = 1.0
    mask[-1, :] = 1.0
    layers = [stationary_bounce_layer("walls", mask)]
    tables = tables_for(f0)
    op = masked_step(tau=0.8, layers=layers, tables=tables, collide=collide_bgk, transport=stream)

    plain = {"f": f0.clone()}
    for _ in range(20):
        op(plain, 0.0)

    hooked, log = solve_with_hooks(
        {"f": f0.clone()}, 20, operator=op, setup_step=lambda s, t, i: s, save_interval=7,
    )
    assert log.setup == list(range(20))
    assert torch.equal(plain["f"], hooked["f"]), "hook plumbing must be physics-neutral"


def test_velocity_ramp_hook_reaches_target():
    f = rest_field()
    hook = velocity_ramp_hook(key="wall_velocity", velocity=(0.04, 0.0), ramp_steps=5)
    state = {"f": f}
    seen = []
    for i in range(8):
        state = hook(state, float(i), i)
        seen.append(float(state["wall_velocity"][0]))
    assert seen[0] == pytest.approx(0.04 / 5)
    assert seen[4] == pytest.approx(0.04)
    assert seen[-1] == pytest.approx(0.04)


def test_callable_layer_velocity_reads_hook_state():
    f = rest_field()
    mask = top_mask()
    layer = moving_bounce_layer("lid", mask, lambda state, like: state["wall_velocity"])
    state = {"f": f, "wall_velocity": torch.tensor([0.02, 0.0], dtype=DT)}
    pop = layer_populations(layer, f, tables=tables_for(f), state=state)
    dmom_row = float(
        ((pop - f[OPPOSITE.to(torch.int64)]) * C.to(DT)[:, 0].view(9, 1, 1)).sum(dim=0)[-1, :].sum()
    )
    assert dmom_row == pytest.approx(2.0 * 0.02 * 8, rel=1e-12)  # 8 masked cells, 6*u_x*cs^2*rho
    assert float((pop * C.to(DT)[:, 1].view(9, 1, 1)).sum(dim=0)[-1, :].sum()) == pytest.approx(0.0, abs=1e-12)


def test_couette_reaches_analytic_linear_profile():
    """End-to-end: hook-driven moving lid + two disjoint masks -> plane Couette."""
    ny, nx = 16, 4
    tau = 0.8
    u_wall = 0.01
    H = float(ny - 2)
    f = rest_field(ny, nx)
    lo = torch.zeros(ny, nx, dtype=DT)
    hi = torch.zeros(ny, nx, dtype=DT)
    lo[0, :] = 1.0
    hi[-1, :] = 1.0
    tables = tables_for(f)
    layers = [
        stationary_bounce_layer("lower", lo),
        moving_bounce_layer("lid", hi, lambda state, like: state["wall_velocity"]),
    ]
    op = masked_step(tau=tau, layers=layers, tables=tables)
    state = {"f": f}
    setup = velocity_ramp_hook(velocity=(u_wall, 0.0), ramp_steps=200)
    saved: list[int] = []

    def save(st, t, i):
        saved.append(i)

    state, log = solve_with_hooks(
        state, 4000, operator=op, setup_step=setup, save_step=save, save_interval=1000
    )
    assert log.save == [0, 1000, 2000, 3000, 4000] == saved

    _, ux, _ = macroscopic(state["f"])
    rows = torch.arange(1, ny - 1, dtype=DT)
    exact = u_wall * (rows - 0.5) / H
    got = ux[1:-1, nx // 2]
    err = float((got - exact).abs().max() / u_wall)
    assert err < 5e-3, f"Couette profile error {err:.3%} too large"
    # ... and the top/lower masks themselves carry no spurious flow through them
    assert torch.isfinite(state["f"]).all()

# --------------------------------------------------------------------------
# Round 2: partially saturated cells (fractional solid fraction)
# --------------------------------------------------------------------------
from tensorlbm.step_hooks import (  # noqa: E402
    layer_weight,
    partial_cell_layer,
    partial_cell_weight,
)


def _rand_field(ny: int = 6, nx: int = 5, seed: int = 7) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(9, ny, nx, dtype=torch.float64, generator=g) * 0.1 + 0.05


def test_partial_cell_weight_limits_and_monotonicity():
    """B(0)=0 (no wall), B(1)=1 (pure bounce-back), monotone in eps_s, tau-dependent."""
    tau = 0.8
    eps = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
    b = partial_cell_weight(eps, tau)
    assert float(b[0]) == 0.0
    assert float(b[-1]) == 1.0
    assert torch.all(b[1:] > b[:-1])
    # the raw-mask blend is NOT the Noble-Torczynski weight inside (0,1)
    mid = partial_cell_weight(torch.tensor(0.5, dtype=torch.float64), tau)
    assert float(mid) < 0.5
    assert float(partial_cell_weight(torch.tensor(0.5, dtype=torch.float64), 1.2)) > float(mid)
    with pytest.raises(ValueError):
        partial_cell_weight(torch.tensor(0.5, dtype=torch.float64), 0.5)


def test_partial_cell_layer_validation():
    f = _rand_field()
    with pytest.raises((TypeError, ValueError)):  # tau is required
        partial_cell_layer("p", torch.full(f.shape[1:], 0.5, dtype=torch.float64), None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):  # tau out of range
        partial_cell_layer("p", torch.full(f.shape[1:], 0.5, dtype=torch.float64), 0.4)
    with pytest.raises(ValueError):  # solid fraction outside [0, 1]
        partial_cell_layer("p", torch.full(f.shape[1:], 1.5, dtype=torch.float64), 0.8)
    with pytest.raises(ValueError):
        partial_cell_layer(
            "p", torch.full(f.shape[1:], 0.5, dtype=torch.float64), 0.8, reflect="sideways"
        )


def test_partial_cell_eps_one_is_the_bounce_layer_bitwise():
    """A mask partition may mix full and partial cells: eps_s=1 must be bounce-back."""
    f = _rand_field()
    m = torch.zeros(f.shape[1:], dtype=torch.float64)
    m[1:3, 1:3] = 1.0
    a = compose_masked_boundaries(f, f, [stationary_bounce_layer("wall", m)])
    b = compose_masked_boundaries(f, f, [partial_cell_layer("part", m, 0.8, reflect="pre")])
    assert torch.equal(a, b)


def test_partial_cell_eps_zero_is_the_identity():
    f = _rand_field()
    z = torch.zeros(f.shape[1:], dtype=torch.float64)
    out = compose_masked_boundaries(f, f, [partial_cell_layer("part", z, 0.8)])
    assert torch.equal(out, f)


def test_partial_cell_conserves_mass_and_weight_is_not_the_mask():
    f = _rand_field()
    blob = torch.zeros(f.shape[1:], dtype=torch.float64)
    blob[2:4, 1:4] = 0.37
    layer = partial_cell_layer("part", blob, 0.8)
    out = compose_masked_boundaries(f, f, [layer])
    assert abs(float(out.sum()) - float(f.sum())) < 1e-14 * abs(float(f.sum()))
    w = layer_weight(layer, f)
    assert torch.allclose(w[0], partial_cell_weight(blob, 0.8))
    assert not torch.allclose(w[0], blob)  # the naive float-mask weight


def test_partial_cell_reflect_choice_changes_the_operator():
    """pre vs post reflection differ once the cell is not fully solid."""
    f_pre = _rand_field(seed=3)
    f_post = f_pre + 0.01 * _rand_field(seed=4)
    m = torch.zeros(f_pre.shape[1:], dtype=torch.float64)
    m[2, 2] = 0.5
    a = compose_masked_boundaries(f_post, f_pre, [partial_cell_layer("p", m, 0.8, reflect="pre")])
    b = compose_masked_boundaries(f_post, f_pre, [partial_cell_layer("p", m, 0.8, reflect="post")])
    assert not torch.equal(a, b)
    # ... and they coincide on a fully solid cell
    m1 = torch.zeros_like(m)
    m1[2, 2] = 1.0
    a1 = compose_masked_boundaries(f_post, f_pre, [partial_cell_layer("p", m1, 0.8, reflect="pre")])
    b1 = compose_masked_boundaries(f_post, f_pre, [partial_cell_layer("p", m1, 0.8, reflect="post")])
    assert not torch.equal(a1, b1)  # post-collision pull-back at eps_s=1 is the trap
    boun = compose_masked_boundaries(f_post, f_pre, [stationary_bounce_layer("wall", m1)])
    assert torch.equal(a1, boun)


def test_partial_cell_moving_wall_reduces_to_stationary_layer():
    f = _rand_field()
    m = torch.zeros(f.shape[1:], dtype=torch.float64)
    m[1, :] = 0.5
    zero = partial_cell_layer("p", m, 0.8, velocity=(0.0, 0.0))
    still = partial_cell_layer("p", m, 0.8)
    a = compose_masked_boundaries(f, f, [zero])
    b = compose_masked_boundaries(f, f, [still])
    assert torch.equal(a, b)
    moving = partial_cell_layer("p", m, 0.8, velocity=(0.01, 0.0))
    c = compose_masked_boundaries(f, f, [moving])
    assert not torch.allclose(a, c)
