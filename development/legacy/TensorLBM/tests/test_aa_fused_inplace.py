"""Verification suite for the strict single-field (A-A) fused push-pull kernel.

Anchor: ``XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN``
(``lbm-principles-practice/code``, fused push-pull A-A pattern).

The A-A pattern is a *storage* discipline on top of the fused push-pull scheme:
one ``Q x N`` field is advanced in place.  These tests pin both halves:

* **storage**: the step allocates no second ``Q x N`` field (probed by
  intercepting ``torch.empty*``; the fused A-B kernel is used as the positive
  control that proves the probe bites);
* **scheme**: the in-place pass agrees with the fused A-B reference for the
  periodic, walled, moving-wall and Guo-forced cases to round-off, conserves
  mass in a closed box, and reproduces plane Couette / plane Poiseuille against
  their analytic solutions (owner standard: < 3 %).
"""

from __future__ import annotations

import torch

import tensorlbm.aa_fused_push_pull as AB
import tensorlbm.aa_fused_push_pull_inplace as AA

FIELD_GATE = 1e-13     # round-off gate for fused vs in-place over 200 steps
MASS_GATE = 1e-12
PRECISION_GATE = 0.03  # owner standard: 3 %


def rest_state(nx: int, ny: int, perturb: bool = False) -> torch.Tensor:
    return AA.rest_equilibrium(ny, nx, perturb=perturb)


def test_in_place_contract():
    """The step mutates and returns the very same buffer (no second field)."""
    f = rest_state(5, 4, perturb=True)
    ptr = f.data_ptr()
    out = AA.aa_fused_push_pull_step(f, 0.9)
    assert out is f
    assert f.data_ptr() == ptr


def test_no_second_field_is_allocated():
    """Intercept ``torch.empty*``: no ``(9, ny, nx)`` allocation in the A-A step."""
    nx, ny = 12, 10
    solid = AA.channel_masks(nx, ny, 1)
    wall = torch.zeros(2, ny, nx, dtype=torch.float64)
    wall[0, -1, :] = 0.01
    force = torch.zeros(2, ny, nx, dtype=torch.float64)
    force[0] = 1e-6

    seen: list[tuple[int, ...]] = []
    orig_empty = torch.empty
    orig_empty_like = torch.empty_like
    orig_zeros = torch.zeros
    orig_zeros_like = torch.zeros_like

    def spy(fn):
        def wrapped(*a, **kw):
            t = fn(*a, **kw)
            if hasattr(t, "shape"):
                seen.append(tuple(t.shape))
            return t
        return wrapped

    torch.empty = spy(orig_empty)
    torch.empty_like = spy(orig_empty_like)
    torch.zeros = spy(orig_zeros)
    torch.zeros_like = spy(orig_zeros_like)
    try:
        f = rest_state(nx, ny)
        seen.clear()
        AA.aa_fused_push_pull_step(f, 0.9, solid=solid, wall_u=wall, force=force)
        aa_shapes = list(seen)

        seen.clear()
        AB.fused_push_pull_step(f.clone(), 0.9, solid=solid, wall_u=wall, force=force)
        ab_shapes = list(seen)
    finally:
        torch.empty = orig_empty
        torch.empty_like = orig_empty_like
        torch.zeros = orig_zeros
        torch.zeros_like = orig_zeros_like

    full = (9, ny, nx)
    assert full not in aa_shapes, f"A-A step allocated a full field: {aa_shapes}"
    # positive control: the fused A-B kernel does allocate one, so the probe bites
    assert full in ab_shapes, f"probe missed the known A-B allocation: {ab_shapes}"


def _run_pair(nx, ny, steps, tau, solid=None, wall_u=None, force=None, perturb=True):
    f_ab = rest_state(nx, ny, perturb=perturb)
    f_aa = f_ab.clone()
    for _ in range(steps):
        f_ab = AB.fused_push_pull_step(f_ab, tau, solid=solid, wall_u=wall_u, force=force)
        AA.aa_fused_push_pull_step(f_aa, tau, solid=solid, wall_u=wall_u, force=force)
    return f_ab, f_aa


def test_equivalence_periodic():
    f_ab, f_aa = _run_pair(8, 7, 200, 0.9)
    assert float((f_ab - f_aa).abs().max()) < FIELD_GATE


def test_equivalence_walled_channel():
    solid = AA.channel_masks(4, 10, 1)
    f_ab, f_aa = _run_pair(4, 10, 200, 0.9, solid=solid)
    diff = float((f_ab - f_aa).abs().max())
    assert diff < FIELD_GATE, diff


def test_equivalence_moving_wall():
    nx, ny = 4, 10
    solid = AA.channel_masks(nx, ny, 1)
    wall = torch.zeros(2, ny, nx, dtype=torch.float64)
    wall[0, -1, :] = 0.02
    f_ab, f_aa = _run_pair(nx, ny, 200, 0.9, solid=solid, wall_u=wall)
    assert float((f_ab - f_aa).abs().max()) < FIELD_GATE


def test_equivalence_guo_forced():
    nx, ny, nu = 4, 10, 0.05
    tau = 3.0 * nu + 0.5
    solid = AA.channel_masks(nx, ny, 1)
    force = torch.zeros(2, ny, nx, dtype=torch.float64)
    force[0] = 1e-6
    f_ab, f_aa = _run_pair(nx, ny, 200, tau, solid=solid, force=force, perturb=False)
    diff = float((f_ab - f_aa).abs().max())
    assert diff < FIELD_GATE, diff
    _, u_ab, _ = AB.macroscopic(f_ab, force)
    _, u_aa, _ = AA.macroscopic(f_aa, force)
    assert float((u_ab - u_aa).abs().max()) < 1e-12


def test_mass_conservation_closed_box():
    """Closed walled box: bounce-back inside the pull pass conserves mass."""
    nx = ny = 16
    solid = torch.zeros(ny, nx, dtype=torch.bool)
    solid[0, :] = solid[-1, :] = True
    solid[:, 0] = solid[:, -1] = True
    f = rest_state(nx, ny, perturb=True)
    fluid = ~solid
    m0 = float(f[:, fluid].sum())
    for step in range(500):
        AA.aa_fused_push_pull_step(f, 0.9, solid=solid)
        if step in (0, 99, 499):
            drift = abs(float(f[:, fluid].sum()) - m0) / m0
            assert drift < MASS_GATE, (step, drift)


def test_couette_analytic():
    """Moving-wall halfway bounce-back reproduces the analytic linear profile."""
    n_fluid, nx, steps, nu, u_wall = 8, 4, 4000, 0.05, 0.02
    tau = 3.0 * nu + 0.5
    ny = n_fluid + 2
    solid = AA.channel_masks(nx, ny, 1)
    fluid = ~solid
    wall = torch.zeros(2, ny, nx, dtype=torch.float64)
    wall[0, -1, :] = u_wall

    f = rest_state(nx, ny)
    for _ in range(steps):
        AA.aa_fused_push_pull_step(f, tau, solid=solid, wall_u=wall)
    _, ux, _ = AA.macroscopic(f)
    prof = ux[fluid].reshape(n_fluid, nx).mean(dim=1)
    err = float((prof - AA.analytic_couette(n_fluid, u_wall)).abs().max() / u_wall)
    assert err < 1e-6, err
    assert err < PRECISION_GATE


def test_poiseuille_analytic_under_3_percent():
    n_fluid, nx, g, nu, steps = 16, 4, 1e-6, 0.05, 4000
    tau = 3.0 * nu + 0.5
    ny = n_fluid + 2
    solid = AA.channel_masks(nx, ny, 1)
    fluid = ~solid
    force = torch.zeros(2, ny, nx, dtype=torch.float64)
    force[0] = g

    f = rest_state(nx, ny)
    for _ in range(steps):
        AA.aa_fused_push_pull_step(f, tau, solid=solid, force=force)
    _, ux, uy = AA.macroscopic(f, force)
    prof = ux[fluid].reshape(n_fluid, nx).mean(dim=1)
    analytic, u_max = AA.analytic_poiseuille(n_fluid, g, nu)

    rel_umax = float(abs(prof.max() - u_max) / u_max)
    l2 = float((prof - analytic).norm() / analytic.norm())
    assert rel_umax < PRECISION_GATE, rel_umax
    assert l2 < PRECISION_GATE, l2
    assert float(uy[fluid].abs().max()) < 1e-6 * u_max


def test_rejects_bad_shape():
    import pytest

    with pytest.raises(ValueError):
        AA.aa_fused_push_pull_step(torch.zeros(5, 4, 4, dtype=torch.float64), 0.8)


def test_macroscopic_matches_reference():
    """The compact (no velocity-tensor) moment evaluation matches the reference."""
    f = rest_state(9, 6, perturb=True)
    f[1] += 1e-3
    f[3] -= 2e-3
    f[2] += 5e-4
    r1, u1, v1 = AB.macroscopic(f)
    r2, u2, v2 = AA.macroscopic(f)
    assert abs(float((r1 - r2).abs().max())) == 0.0
    assert float((u1 - u2).abs().max()) < 1e-15
    assert float((v1 - v2).abs().max()) < 1e-15