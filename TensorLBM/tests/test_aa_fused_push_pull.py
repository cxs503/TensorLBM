"""Verification suite for the AA-pattern fused push-pull kernel.

Anchor: XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN (lbm-principles-practice/code).

The tests pin the *physics* invariants that the fused (pull, no post-collision
array) formulation must satisfy, not merely that the code runs:

* equivalence with the canonical collide-then-stream push scheme -- bit for bit --
  for a periodic domain **and** for a wall-bounded channel with halfway
  bounce-back (a destination-colliding pull scheme is only equivalent in the
  periodic case, see ``test_legacy_kernel_leaks_mass_with_walls``);
* exact mass conservation in a closed walled box;
* plane Couette and plane Poiseuille against their analytic solutions, with the
  owner precision standard (error < 3 %) as the gate.
"""

from __future__ import annotations

import torch

import tensorlbm.aa_fused_push_pull as M


def rest_state(nx: int, ny: int, perturb: bool = False) -> torch.Tensor:
    rho = torch.ones(ny, nx, dtype=torch.float64)
    if perturb:
        rho[ny // 2, nx // 3] += 0.02
    zero = torch.zeros_like(rho)
    return M.equilibrium(rho, zero, zero).clone()


def test_d2q9_lattice_constants():
    assert abs(sum(M.W9) - 1.0) < 1e-15
    assert M.W9[0] == 4.0 / 9.0
    for i in range(9):
        j = M.OPP[i]
        assert M.OPP[j] == i
        assert M.CX[i] == -M.CX[j] and M.CY[i] == -M.CY[j]
        assert M.W9[i] == M.W9[j]
        assert M.CX[i] ** 2 + M.CY[i] ** 2 in (0, 1, 2)


def test_equilibrium_moments():
    rho = torch.full((6, 5), 1.3, dtype=torch.float64)
    ux = torch.full_like(rho, 0.02)
    uy = torch.full_like(rho, -0.03)
    feq = M.equilibrium(rho, ux, uy)
    r, u, v = M.macroscopic(feq)
    assert torch.allclose(r, rho, atol=1e-14)
    assert torch.allclose(u, ux, atol=1e-14)
    assert torch.allclose(v, uy, atol=1e-14)


def test_fused_equals_push_periodic():
    """Periodic domain: the fused pull must reproduce collide-then-stream exactly."""
    f_fused = rest_state(16, 16, perturb=True)
    f_push = f_fused.clone()
    for _ in range(100):
        f_fused = M.fused_push_pull_step(f_fused, 0.8)
        f_push = M.collide_stream_step(f_push, 0.8)
    assert float((f_fused - f_push).abs().max()) < 1e-14


def test_fused_equals_push_channel_halfway_bb():
    """Walled channel: inline halfway bounce-back must match the push reference."""
    nx, ny = 8, 18
    solid = M.channel_masks(nx, ny, 1)
    wall = torch.zeros(2, ny, nx, dtype=torch.float64)
    wall[0, -1, :] = 0.02
    f_fused = rest_state(nx, ny, perturb=True)
    f_push = f_fused.clone()
    for _ in range(300):
        f_fused = M.fused_push_pull_step(f_fused, 0.8, solid=solid, wall_u=wall)
        f_push = M.collide_stream_step(f_push, 0.8, solid=solid, wall_u=wall)
    assert float((f_fused - f_push).abs().max()) < 1e-14


def test_mass_conservation_closed_box():
    """Halfway bounce-back conserves fluid mass to round-off in a closed box."""
    nx, ny, steps = 16, 16, 500
    solid = torch.zeros(ny, nx, dtype=torch.bool)
    solid[0, :] = solid[-1, :] = True
    solid[:, 0] = solid[:, -1] = True
    fluid = ~solid

    for step in (M.fused_push_pull_step, M.collide_stream_step):
        f = rest_state(nx, ny, perturb=True)
        m0 = float(f[:, fluid].sum())
        for _ in range(steps):
            f = step(f, 0.8, solid=solid)
        drift = abs(float(f[:, fluid].sum()) - m0) / m0
        assert drift < 1e-12, (step.__name__, drift)


def test_legacy_kernel_leaks_mass_with_walls():
    """Characterisation of the pre-existing destination-colliding kernel.

    ``aa_fused_stream_collide.aa_fused_step`` collides at the *destination* and
    applies bounce-back on *solid* nodes; with walls present it does not
    conserve mass, so its wall-bounded results cannot be used as evidence. This
    test documents the defect and fails loudly if the old kernel is ever
    "fixed" without being re-validated.
    """
    from tensorlbm.aa_fused_stream_collide import aa_fused_step

    nx, ny, steps = 16, 16, 500
    solid = torch.zeros(ny, nx, dtype=torch.bool)
    solid[0, :] = solid[-1, :] = True
    solid[:, 0] = solid[:, -1] = True
    fluid = ~solid

    f = rest_state(nx, ny, perturb=True)
    m0 = float(f[:, fluid].sum())
    for _ in range(steps):
        f = aa_fused_step(f, 0.8, solid_mask=solid)
    drift = abs(float(f[:, fluid].sum()) - m0) / m0
    assert drift > 1e-8, f"legacy kernel now conserves mass (drift={drift:.2e}); re-validate its BCs"


def test_couette_analytic():
    """Moving-wall halfway bounce-back reproduces the analytic linear profile."""
    n_fluid, nx, steps = 8, 4, 4000
    nu, u_wall = 0.05, 0.02
    tau = 3.0 * nu + 0.5
    ny = n_fluid + 2
    solid = M.channel_masks(nx, ny, 1)
    fluid = ~solid
    wall = torch.zeros(2, ny, nx, dtype=torch.float64)
    wall[0, -1, :] = u_wall

    f = rest_state(nx, ny)
    for _ in range(steps):
        f = M.fused_push_pull_step(f, tau, solid=solid, wall_u=wall)
    _, ux, _ = M.macroscopic(f)
    prof = ux[fluid].reshape(n_fluid, nx).mean(dim=1)
    analytic = M.analytic_couette(n_fluid, u_wall)
    err = float((prof - analytic).abs().max() / u_wall)
    assert err < 1e-6, err


def test_poiseuille_analytic_under_3_percent():
    """Guo-forced plane Poiseuille vs u_max = g H^2 / (8 nu) (owner 3 % gate)."""
    n_fluid, nx, g, nu, steps = 16, 4, 1e-6, 0.05, 4000
    tau = 3.0 * nu + 0.5
    ny = n_fluid + 2
    solid = M.channel_masks(nx, ny, 1)
    fluid = ~solid
    force = torch.zeros(2, ny, nx, dtype=torch.float64)
    force[0] = g

    f = rest_state(nx, ny)
    for _ in range(steps):
        f = M.fused_push_pull_step(f, tau, solid=solid, force=force)
    _, ux, uy = M.macroscopic(f, force)
    prof = ux[fluid].reshape(n_fluid, nx).mean(dim=1)
    analytic, u_max = M.analytic_poiseuille(n_fluid, g, nu)

    rel_umax = float(abs(prof.max() - u_max) / u_max)
    l2 = float((prof - analytic).norm() / analytic.norm())
    assert rel_umax < 0.03, rel_umax
    assert l2 < 0.03, l2
    # transverse velocity must stay negligible (incompressible, fully developed)
    assert float(uy[fluid].abs().max()) < 1e-6 * u_max
    # no net streamwise momentum drift in a periodic-in-x channel
    assert abs(float(ux[fluid].mean()) - float(prof.mean())) < 1e-15


def test_fused_matches_push_poiseuille():
    """The fused pass and the push oracle give identical forced-flow results."""
    n_fluid, nx, g, nu, steps = 16, 4, 1e-6, 0.05, 2000
    tau = 3.0 * nu + 0.5
    ny = n_fluid + 2
    solid = M.channel_masks(nx, ny, 1)
    force = torch.zeros(2, ny, nx, dtype=torch.float64)
    force[0] = g

    f = rest_state(nx, ny)
    for _ in range(steps):
        f = M.fused_push_pull_step(f, tau, solid=solid, force=force)
    f_ref = rest_state(nx, ny)
    for _ in range(steps):
        f_ref = M.collide_stream_step(f_ref, tau, solid=solid, force=force)
    du = float((M.macroscopic(f, force)[1] - M.macroscopic(f_ref, force)[1]).abs().max())
    assert du < 1e-14, du


def test_rejects_bad_shape():
    import pytest

    with pytest.raises(ValueError):
        M.fused_push_pull_step(torch.zeros(5, 4, 4, dtype=torch.float64), 0.8)
    with pytest.raises(ValueError):
        M.collide_stream_step(torch.zeros(9, 4, dtype=torch.float64), 0.8)