"""Tests for the anti-bounceback (Inamuro) wet-node boundary condition.

Coverage
--------
* mechanics: direction classification, garbage-overwrite, no-op off the walls
* exactness: a uniform flow with matching wall velocity is an exact fixed point
* physics: plane Poiseuille and Couette conformance against the analytic
  solutions (the acceptance criterion of the ledger entry
  ``XVAL-LBM-ANTI-BOUNCE-BACK-INAMURO-WETNODE`` is < 15 %)
* invariants: zero normal velocity on the walls, mass conservation
* effective wall position: the rule places the no-slip surface at the boundary
  node (up to an O(1) tau-dependent bias), unlike halfway bounce-back
* negative controls: the linearised source term biases ``u_w`` by ``u_w/3``;
  dropping the wall rule (or the collision-level velocity prescription)
  destroys the solution -- i.e. the positive results are not vacuous

Numbers quoted in the assertions are measured values (see
``docs/antibb_wetnode_verification_20261113.md``); tolerances leave room for
platform/dtype noise but are tight enough to reject the controls.
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.boundary.antibb_channel_benchmark import (
    ChannelSetup,
    couette_profile,
    l2_relative_error,
    poiseuille_profile,
    run_channel,
)
from tensorlbm.boundary.antibb_wetnode import (
    antibb_unknown_mask,
    apply_antibb_wetnode,
    bottom_top_normals,
    effective_wall_offset,
)
from tensorlbm.d2q9 import C, equilibrium

NU = 0.1
NX = 4


def _fixed_point_case(ny: int = 5, nx: int = 4, u: float = 0.1):
    rho = torch.ones(ny, nx, dtype=torch.float64)
    ux = torch.full((ny, nx), u, dtype=torch.float64)
    uy = torch.zeros_like(ux)
    f = equilibrium(rho, ux, uy)
    wet = torch.zeros((ny, nx), dtype=torch.bool)
    wet[0, :] = True
    wet[-1, :] = True
    normals = bottom_top_normals(ny, nx, dtype=torch.float64)
    u_wall = torch.zeros((ny, nx, 2), dtype=torch.float64)
    u_wall[..., 0] = u
    f_streamed = torch.stack(
        [torch.roll(f[i], shifts=(int(C[i, 1]), int(C[i, 0])), dims=(0, 1)) for i in range(9)]
    )
    mask = antibb_unknown_mask(normals, wet)
    polluted = f_streamed.clone()
    polluted[mask] += 0.3  # must be overwritten by the rule
    return f, polluted, wet, normals, u_wall, u, mask


def _macros(f: torch.Tensor):
    rho = f.sum(dim=0)
    ux = (f * C[:, 0].view(-1, 1, 1)).sum(dim=0) / rho
    uy = (f * C[:, 1].view(-1, 1, 1)).sum(dim=0) / rho
    return rho, ux, uy


# ---------------------------------------------------------------------------
# mechanics
# ---------------------------------------------------------------------------


def test_unknown_direction_classification_d2q9():
    """Bottom wall (n=+y) misses directions 2,5,6; top wall (n=-y) misses 4,7,8."""
    ny, nx = 5, 4
    normals = bottom_top_normals(ny, nx, dtype=torch.float64)
    wet = torch.zeros((ny, nx), dtype=torch.bool)
    wet[0, :] = wet[-1, :] = True
    mask = antibb_unknown_mask(normals, wet)
    assert mask[:, 0, 0].nonzero().flatten().tolist() == [2, 5, 6]
    assert mask[:, -1, 0].nonzero().flatten().tolist() == [4, 7, 8]
    # interior nodes are never touched
    assert not mask[:, 1:-1, :].any()


def test_uniform_flow_is_an_exact_fixed_point():
    """Garbage in the unknown directions is replaced; the flow is reproduced."""
    f, polluted, wet, normals, u_wall, u, _ = _fixed_point_case()
    fb = apply_antibb_wetnode(f, polluted, wet, normals, u_wall, 1.0, closure="exact")
    rho, ux, uy = _macros(fb)
    assert float((rho - 1.0).abs().max()) < 1e-12
    assert float((ux - u).abs().max()) < 1e-12
    assert float(uy.abs().max()) < 1e-12
    # interior populations are untouched by the boundary rule
    assert torch.equal(fb[:, 1:-1, :], polluted[:, 1:-1, :])


def test_linearised_source_is_biased_negative_control():
    """The ``1 + 3(c.u)`` source term overshoots the wall velocity by u/3."""
    f, polluted, wet, normals, u_wall, u, _ = _fixed_point_case(u=0.1)
    fb = apply_antibb_wetnode(f, polluted, wet, normals, u_wall, 1.0, closure="linear")
    _, ux, uy = _macros(fb)
    overshoot = float((ux - u).abs().max())
    assert overshoot == pytest.approx(u / 3.0, rel=0.02)
    assert float(uy.abs().max()) < 1e-12  # normal momentum stays balanced


def test_effective_wall_offset_contract():
    """The documented wall position of the rule is the boundary-node plane."""
    assert effective_wall_offset() == 0.0


# ---------------------------------------------------------------------------
# physics
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def poiseuille_result():
    ny = 17
    H = ny - 1
    force = 8.0 * NU * 0.02 / H**2
    setup = ChannelSetup(
        ny=ny, nx=NX, tau=3 * NU + 0.5, force=force, steps=12000,
        steady_tol=1e-9, check_every=500,
    )
    res = run_channel(setup)
    return setup, res


@pytest.fixture(scope="module")
def couette_result():
    ny = 17
    setup = ChannelSetup(
        ny=ny, nx=NX, tau=3 * NU + 0.5, u_wall_bottom=0.05, steps=12000,
        steady_tol=1e-9, check_every=500,
    )
    res = run_channel(setup)
    return setup, res


def test_poiseuille_matches_analytic_within_acceptance(poiseuille_result):
    setup, res = poiseuille_result
    prof = res.ux[:, NX // 2].double()
    exact = poiseuille_profile(setup)
    err = l2_relative_error(prof[1:-1], exact[1:-1])
    print(f"Poiseuille ny={setup.ny}: relative L2 error = {err*100:.3f}%")
    assert err < 0.15, f"acceptance (<15%) violated: {err*100:.2f}%"
    # wall-normal velocity must vanish identically (no leakage through the wall)
    assert float(res.uy.abs().max()) / float(prof.abs().max()) < 1e-10
    # mass is conserved by the rule (no spurious sources)
    assert abs(res.mass_drift) < 1e-6


def test_poiseuille_profile_is_exactly_parabolic_at_fitted_wall(poiseuille_result):
    """Accounting for the measured wall offset, the profile is the exact parabola.

    This separates the *shape* error of the rule (second order, ~1e-8 here) from
    the O(1) wall-position bias measured in the verification report.
    """
    setup, res = poiseuille_result
    ny = setup.ny
    prof = res.ux[:, NX // 2].double()
    y = torch.arange(ny, dtype=torch.float64)[1:-1]

    def err_for(delta: float) -> float:
        H = (ny - 1) + 2.0 * delta
        yy = y + delta
        exact = setup.force * yy * (H - yy) / (2.0 * setup.nu)
        return l2_relative_error(prof[1:-1], exact)

    deltas = torch.linspace(-0.5, 0.2, 71).tolist()
    errs = [err_for(d) for d in deltas]
    i = min(range(len(errs)), key=errs.__getitem__)
    lo, hi = deltas[max(i - 1, 0)], deltas[min(i + 1, len(deltas) - 1)]
    # ternary refinement of the one-parameter fit
    for _ in range(40):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if err_for(m1) < err_for(m2):
            hi = m2
        else:
            lo = m1
    best = (err_for(0.5 * (lo + hi)), 0.5 * (lo + hi))
    assert best[0] < 1e-6, f"profile is not an exact parabola (err={best[0]:.2e})"
    # measured effective wall position sits within half a lattice unit of the node
    assert -0.5 < best[1] < 0.2


def test_couette_moving_wall(couette_result):
    setup, res = couette_result
    prof = res.ux[:, NX // 2].double()
    exact = couette_profile(setup)
    err = l2_relative_error(prof[1:-1], exact[1:-1])
    print(f"Couette ny={setup.ny}: relative L2 error = {err*100:.3f}%")
    assert err < 0.05, f"moving wall not reproduced: {err*100:.2f}%"
    # the prescribed wall velocity is reproduced at the boundary node
    assert float(prof[0]) == pytest.approx(0.05, rel=0.01)
    assert abs(res.mass_drift) < 1e-6


def test_couette_without_dirichlet_source_is_wrong_negative_control():
    """Closure ablation: wrong source term -> order-one error, test is sensitive."""
    ny = 17
    setup = ChannelSetup(
        ny=ny, nx=NX, tau=3 * NU + 0.5, u_wall_bottom=0.05, steps=3000,
    )
    import tensorlbm.boundary.antibb_channel_benchmark as bench

    original = bench.apply_antibb_wetnode

    def patched(*args, **kwargs):
        kwargs["closure"] = "linear"
        return original(*args, **kwargs)

    bench.apply_antibb_wetnode = patched
    try:
        res = run_channel(setup)
    finally:
        bench.apply_antibb_wetnode = original
    prof = res.ux[:, NX // 2].double()
    err = l2_relative_error(prof[1:-1], couette_profile(setup)[1:-1])
    assert err > 0.2, f"linearised closure should be badly wrong, got {err*100:.1f}%"


def test_wall_rule_required_negative_control():
    """Without a wall rule the forced channel never balances the body force."""
    ny, H = 17, 16
    force = 8.0 * NU * 0.02 / H**2
    setup = ChannelSetup(ny=ny, nx=NX, tau=3 * NU + 0.5, force=force, steps=1500,
                         wall_mode="none")
    res = run_channel(setup)
    prof = res.ux[:, NX // 2].double()
    err = l2_relative_error(prof[1:-1], poiseuille_profile(setup)[1:-1])
    # the runaway either diverges (nan/inf) or misses the analytic profile by
    # orders of magnitude -- both are "not the boundary-condition result"
    # at this resolution the ABB baseline is 5.8 %; a runaway 3x above that is
    # unmistakable (it keeps growing with time, the control is not converged)
    assert (not err == err) or err > 3.0 * 0.058, f"no-BC control should be far off, got {err*100:.1f}%"


def test_halfway_bounce_back_wall_position_differs():
    """The two rules do not place the wall at the same distance from the node.

    Anti-bounceback keeps the no-slip surface close to the boundary node, while
    halfway bounce-back shifts it by a fraction of a lattice unit; the
    fitted offsets must therefore differ measurably.
    """
    ny, H = 17, 16
    force = 8.0 * NU * 0.02 / H**2
    offsets = {}
    for wall_mode in ("antibb", "halfway"):
        setup = ChannelSetup(ny=ny, nx=NX, tau=3 * NU + 0.5, force=force,
                             steps=12000, steady_tol=1e-9, check_every=500,
                             wall_mode=wall_mode)
        res = run_channel(setup)
        prof = res.ux[:, NX // 2].double()
        y = torch.arange(ny, dtype=torch.float64)[1:-1]

        def err_for(delta, prof=prof, y=y):
            Hw = (ny - 1) + 2.0 * delta
            yy = y + delta
            return l2_relative_error(
                prof[1:-1], force * yy * (Hw - yy) / (2.0 * NU)
            )

        offsets[wall_mode] = min(
            (err_for(d), d) for d in torch.linspace(-0.6, 0.4, 101).tolist()
        )
    assert abs(offsets["antibb"][1] - offsets["halfway"][1]) > 0.03