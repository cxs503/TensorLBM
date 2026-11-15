"""Physics regression for the *moving* sub-cell wall and the per-step setup hook.

R3 (``docs/XVAL-LBM-PER-STEP-SETUP-HOOK-AND-MASK-BLEND-R3-FRACTIONAL-MASK.md``)
left the moving partially saturated cell unmeasured; the only test was a smoke
assertion that "moving differs from stationary".  R4
(``docs/XVAL-LBM-PER-STEP-SETUP-HOOK-AND-MASK-BLEND-R4-MOVING-SUBCELL-WALL.md``)
measures it against closed-form plane Couette (bottom sub-cell wall moving at U,
top at rest, ``u(y) = U (y_t - y)/H``).  These tests keep both halves of the
result under pytest:

1. the features that *do* work: a per-step ``setup_step`` hook drives the
   sub-cell wall (bit-wise equal to passing the velocity directly, and a zero
   hook leaves the field at rest); the solid limit ``eps_s = 1`` is bit-wise the
   binary layer (``moving_bounce_layer`` / ``stationary_bounce_layer``);
2. the measured limitation, pinned with numbers: for ``0 < eps_s < 1`` the NT
   momentum term is scaled by ``B``, so the wall velocity is *under-driven* and
   the analytic profile is missed by far more than the 3 % internal target.
   A future operator fix must update these numbers, not silently pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "xval_per_step_hook_mask_blend"
sys.path.insert(0, str(BENCH))

import run_r4_moving as M  # noqa: E402

NY, NX, TAU, U = 16, 8, 0.8, 0.02


@pytest.fixture(scope="module")
def wall_eps_025() -> dict:
    return M.run_moving_wall(NY, NX, TAU, 0.25, "nt_post")


@pytest.fixture(scope="module")
def wall_eps_050() -> dict:
    return M.run_moving_wall(NY, NX, TAU, 0.5, "nt_post")


def test_solid_limit_is_bitwise_the_binary_moving_wall():
    """eps_s = 1 must collapse to moving_bounce_layer, bit-wise.

    Before the R4 solid-limit fix the default ``reflect="post"`` reflected the
    post-collision state of a fully solid cell and displaced the interface by
    0.40 dx (measured in R3 as an xfail).
    """
    lim = M.solid_limit(NY, NX, TAU, moving=True)
    assert lim["bitwise"], (
        f"eps_s=1 moving wall differs from moving_bounce_layer by {lim['max_rel_vs_binary']:.3e} "
        "(expected bit-wise)"
    )
    lim_s = M.solid_limit(NY, NX, TAU, moving=False)
    assert lim_s["bitwise"], (
        f"eps_s=1 stationary wall differs from stationary_bounce_layer by "
        f"{lim_s['max_rel_vs_binary']:.3e} (expected bit-wise)"
    )


def test_per_step_hook_drives_a_sub_cell_wall():
    """setup_step -> callable velocity must reach the mask blend bit-wise."""
    import tensorlbm.step_hooks as SH

    def make(value: float):
        def setup(state, t, i):
            state["wall_velocity"] = torch.tensor([value, 0.0], dtype=state["f"].dtype)
            return state

        return setup

    direct = M.run_moving_wall(NY, NX, TAU, 0.5, "nt_post")
    hooked = M.run_moving_wall(NY, NX, TAU, 0.5, "nt_post", setup_step=make(U))
    zero = M.run_moving_wall(NY, NX, TAU, 0.5, "nt_post", setup_step=make(0.0))
    assert hooked["hooks_setup"] == direct["steps"]
    assert np.max(np.abs(np.array(hooked["profile"]) - np.array(direct["profile"]))) <= 1e-12 * U, (
        "the hook-written wall velocity does not reproduce the direct argument"
    )
    assert np.max(np.abs(zero["profile"])) <= 1e-9 * U, (
        "a hook writing u_wall = 0 must leave the channel at rest"
    )
    assert SH.velocity_ramp_hook is not None  # the hook family used by the docs


def test_hook_ramp_reaches_the_same_steady_state():
    """A ramped hook must converge to the constant-velocity steady state."""
    ramp = M.hook_cases(NY, NX, TAU, 0.5)
    assert ramp["ramp_vs_constant_l2_pct"] <= 1.0, (
        f"ramped run differs from constant-U by {ramp['ramp_vs_constant_l2_pct']:.3f} % L2"
    )


def test_moving_sub_cell_wall_underdrives_the_wall_velocity(wall_eps_025):
    """Pinned limitation: for eps_s < 1 the momentum term is scaled by B."""
    assert wall_eps_025["finite"]
    assert wall_eps_025["u_wall_recovered"] < 0.75, (
        "the NT momentum term used to under-drive the wall velocity badly at eps_s=0.25; "
        f"measured u(y_b)/U = {wall_eps_025['u_wall_recovered']:.4f}. If the operator was "
        "fixed, update docs/XVAL-...-R4-MOVING-SUBCELL-WALL.md and this test together"
    )
    assert wall_eps_025["profile_l2_pct"] > 15.0, (
        f"profile L2 {wall_eps_025['profile_l2_pct']:.3f} % is expected to exceed the owner "
        "ceiling for a moving sub-cell wall -- this is the documented R4 finding"
    )


def test_binary_moving_wall_is_still_exact():
    """The staircase (binary) moving wall is the reference: it must stay accurate.

    Guards against 'fixing' the partial operator by breaking the binary one.
    """
    y_b, y_t = 1.5, NY - 1.5
    H = y_t - y_b
    solid = torch.zeros(NY, NX, dtype=torch.float64)
    solid[0, :] = 1.0
    row1 = torch.zeros(NY, NX, dtype=torch.float64)
    row1[1, :] = 1.0
    top = torch.zeros(NY, NX, dtype=torch.float64)
    top[-1, :] = 1.0
    import tensorlbm.step_hooks as SH
    import run_xval as R

    f0 = R.rest_field(NY, NX)
    op = SH.masked_step(
        tau=TAU,
        layers=[
            SH.stationary_bounce_layer("s0", solid),
            SH.moving_bounce_layer("lid", row1, (U, 0.0)),
            SH.stationary_bounce_layer("top", top),
        ],
        tables=SH.tables_for(f0),
    )
    with torch.no_grad():
        state, _ = SH.solve_with_hooks(
            {"f": f0.clone()}, R.steps_for_couette(H, (TAU - 0.5) / 3.0), operator=op
        )
    _rho, ux, _ = R.macroscopic(state["f"])
    u = ux.numpy()[:, 0]
    ys = np.arange(2, NY - 1, dtype=float)
    ref = U * (y_t - ys) / H
    l2 = 100.0 * float(np.linalg.norm(u[2:NY - 1] - ref) / np.linalg.norm(ref))
    assert l2 < 1.0, f"binary moving wall L2 {l2:.4f} % >= 1 %"
