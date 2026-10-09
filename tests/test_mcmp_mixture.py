"""W5-A discriminative tests: MCMP mixture-momentum equilibrium (u_eq="mixture").

Four discriminating properties of the Shan-Doolen mixture equilibrium versus
the legacy per-component ("self") equilibrium in
``tensorlbm.multiphase.collide_sc_two_component`` (library package):

T1  closed-domain F=0 momentum behaviour (uniform periodic field):
    total momentum is conserved by both conventions (exact D2Q9 moment
    identity), but only "mixture" equilibrates the *relative* component
    velocity - "self" leaves it bit-identical forever (f = feq(u_sigma) is a
    fixed point).  With unequal relaxation times the mixture exchange
    transiently biases total momentum by O(d(1/tau) * |u1-u2|); quantified.
T2  rho2 -> 0 single-component limit: both conventions converge, deviation
    shrinks monotonically with rho2.
T3  Galilean boost invariance of an interface (periodic, no walls):
    |u_B - u0 - u_A| stays small for "mixture".  Run at the mixture-stable
    working point G_12 = -1.0 (explore finding: unequal-tau mixture diverges
    for G <= -1.5); thresholds locked by explore E2.
T4  two-phase Poiseuille component-slip collapse (H64, G_12 = -1.0):
    mixture collapses the inter-component slip u1-u2 to a few per-mille of
    u_max while "self" retains an O(u_max) slip - the direct signature of
    the restored inter-component viscous coupling.  NOTE (REV-C): the
    originally pre-registered expectation "centre error drops sharply under
    mixture" was falsified by explore E/P series (mixture NaNs at the
    protocol coupling G=-2.5; at stable couplings the profile error does
    not drop) - errors for both modes are therefore recorded, not asserted.
"""

from __future__ import annotations

import torch

from tensorlbm.boundaries import bounce_back_cells  # noqa: E402
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.multiphase import collide_sc_two_component  # noqa: E402
from tensorlbm.porous_media import _two_phase_poiseuille_analytical  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402

DEV = torch.device("cuda:5") if torch.cuda.is_available() else torch.device("cpu")
G12 = -2.5  # library sign convention: segregation working point (W4-B record)
G12_STABLE = -1.0  # mixture-stable working point for unequal taus (explore E3)


# ------------------------------------------------------------------ helpers


def _uniform_pair(ny, nx, rho1, rho2, u1x, u2x):
    r1 = torch.full((ny, nx), rho1, device=DEV)
    r2 = torch.full((ny, nx), rho2, device=DEV)
    u1 = torch.full((ny, nx), u1x, device=DEV)
    u2 = torch.full((ny, nx), u2x, device=DEV)
    z = torch.zeros_like(u1)
    return equilibrium(r1, u1, z), equilibrium(r2, u2, z)


def _step(f1, f2, tau1, tau2, u_eq, gx=0.0, gy=0.0, g12=G12):
    f1, f2 = collide_sc_two_component(
        f1, f2, G_12=g12, tau1=tau1, tau2=tau2, gx=gx, gy=gy, u_eq=u_eq
    )
    return stream(f1), stream(f2)


def _moments(f1, f2):
    r1, ux1, _ = macroscopic(f1)
    r2, ux2, _ = macroscopic(f2)
    p_tot = float((r1 * ux1 + r2 * ux2).sum())
    u_rel = float((ux1 - ux2).abs().max())
    # roundoff reference: grid-summed absolute momentum magnitude
    p_scale = float((r1 * ux1).abs().sum() + (r2 * ux2).abs().sum())
    return p_tot, u_rel, p_scale


# ------------------------------------------------------------------ T1


def test_t1_closed_domain_momentum_and_relative_velocity_damping():
    """F=0 closed (periodic) domain, uniform densities -> SC force vanishes.

    Equal taus: mixture conserves total momentum *and* damps the relative
    velocity; self conserves total momentum but the relative velocity is a
    fixed point (missing inter-component momentum coupling)."""
    ny = nx = 24
    rho1, rho2 = 0.7, 0.3
    u1x, u2x = 0.05, -0.05 * rho1 / rho2  # zero total momentum
    u_rel0 = abs(u1x - u2x)
    m1_scale = rho1 * abs(u1x) * ny * nx  # grid-summed water-momentum scale
    # roundoff reference fixed at the INITIAL momentum scale: the mixture
    # leg damps the velocities, so a final-state reference degenerates
    p_scale0 = (rho1 * abs(u1x) + rho2 * abs(u2x)) * ny * nx

    for u_eq, expect_damped in (("mixture", True), ("self", False)):
        f1, f2 = _uniform_pair(ny, nx, rho1, rho2, u1x, u2x)
        for _ in range(200):
            f1, f2 = _step(f1, f2, 1.0, 1.0, u_eq)
        p_tot, u_rel, p_scale = _moments(f1, f2)
        # total momentum: conserved in both conventions to fp32 roundoff of
        # the grid-summed momentum scale (REV-D: absolute 1e-6 was below
        # the roundoff floor; measured self equal-tau |P| ~ 1e-7 relative)
        assert abs(p_tot) < 1e-5 * p_scale0, (u_eq, p_tot, p_scale0)
        if expect_damped:
            assert u_rel < 1e-3 * u_rel0, (u_eq, u_rel, u_rel0)
        else:
            assert u_rel > 0.999 * u_rel0, (u_eq, u_rel, u_rel0)

    # Unequal taus: self still never damps and still conserves; mixture damps
    # with a quantified transient total-momentum bias O(d(1/tau)*|u1-u2|).
    for u_eq, expect_damped in (("mixture", True), ("self", False)):
        f1, f2 = _uniform_pair(ny, nx, rho1, rho2, u1x, u2x)
        for _ in range(200):
            f1, f2 = _step(f1, f2, 1.0, 0.75, u_eq)
        p_tot, u_rel, p_scale = _moments(f1, f2)
        if expect_damped:
            assert u_rel < 0.05 * u_rel0, (u_eq, u_rel, u_rel0)
            assert abs(p_tot) < 0.5 * m1_scale, (u_eq, p_tot, m1_scale)
        else:
            assert u_rel > 0.999 * u_rel0, (u_eq, u_rel, u_rel0)
            assert abs(p_tot) < 1e-5 * p_scale0, (u_eq, p_tot, p_scale0)


# ------------------------------------------------------------------ T2


def test_t2_single_component_limit_agreement():
    """rho2 -> 0: mixture and self converge (deviation shrinks with rho2)."""
    ny = nx = 24
    devs = {}
    for rho2 in (1e-2, 1e-4):
        finals = {}
        for u_eq in ("mixture", "self"):
            f1, f2 = _uniform_pair(ny, nx, 1.0 - rho2, rho2, 0.03, 0.02)
            for _ in range(300):
                f1, f2 = _step(f1, f2, 1.0, 0.75, u_eq, gx=1e-4)
            r1, ux1, _ = macroscopic(f1)
            finals[u_eq] = float(ux1.max())
        devs[rho2] = abs(finals["mixture"] - finals["self"])
    uscale = 0.03
    assert devs[1e-2] < 1e-2 * uscale, devs
    assert devs[1e-4] < devs[1e-2] / 3.0, devs


# ------------------------------------------------------------------ T3


def _galilean_dev(u_eq, u0=0.05, ny=48, nx=12, steps=800):
    """Interface under boost: returns (max|u_B - u0 - u_A|, max|rho_w,B - rho_w,A|)."""

    def run(boost):
        ys = torch.arange(ny, dtype=torch.float32, device=DEV).view(ny, 1)
        ys = ys.expand(ny, nx)
        alpha = 0.5 * (1.0 - torch.tanh((ys - ny / 2.0) / 3.0))
        rw = 0.7 * alpha + 0.3 * (1 - alpha)
        rg = 0.3 * alpha + 0.7 * (1 - alpha)
        ux = torch.full((ny, nx), boost, device=DEV)
        z = torch.zeros_like(ux)
        f1, f2 = equilibrium(rw, ux, z), equilibrium(rg, ux, z)
        for _ in range(steps):
            f1, f2 = _step(f1, f2, 1.0, 0.75, u_eq, g12=G12_STABLE)
        r1, ux1, _ = macroscopic(f1)
        r2, ux2, _ = macroscopic(f2)
        rt = (r1 + r2).clamp(min=1e-12)
        return (r1 * ux1 + r2 * ux2) / rt, r1

    uxA, rwA = run(0.0)
    uxB, rwB = run(u0)
    return float((uxB - u0 - uxA).abs().max()), float((rwB - rwA).abs().max())


def test_t3_galilean_boost_invariance_interface():
    """Boost by u0 at the mixture-stable working point G_12=-1.0: mixture
    physics is invariant up to the lattice-velocity truncation error
    (thresholds locked by W5-A explore E2: measured 3.45e-4 / 6.2e-6)."""
    dev_u, dev_rho = _galilean_dev("mixture")
    dev_u_self, _ = _galilean_dev("self")
    assert dev_u < 1.0e-3, (dev_u, dev_u_self)
    assert dev_rho < 2.0e-5, dev_rho


# ------------------------------------------------------------------ T4


def _poiseuille_slip_err(u_eq, ny=65, nx=8, steps=12000):
    """Quick H64 rung at the mixture-stable coupling G_12=-1.0.

    Returns (component-slip max over interior, u_max, centre max rel err %)
    of the mixture-velocity profile vs the nominal-M analytical solution."""
    half = ny // 2
    ys = torch.arange(ny, dtype=torch.float32, device=DEV).view(ny, 1)
    ys = ys.expand(ny, nx)
    alpha = 0.5 * (1.0 - torch.tanh((ys - half) / 3.0))
    rw = 0.7 * alpha + 0.3 * (1 - alpha)
    rg = 0.3 * alpha + 0.7 * (1 - alpha)
    z = torch.zeros_like(rw)
    f1, f2 = equilibrium(rw, z, z), equilibrium(rg, z, z)
    wall = torch.zeros((ny, nx), dtype=torch.bool, device=DEV)
    wall[0, :] = True
    wall[-1, :] = True
    for _ in range(steps):
        f1, f2 = collide_sc_two_component(
            f1, f2, G_12=G12_STABLE, tau1=1.0, tau2=0.75, gx=5e-6, solid_mask=wall, u_eq=u_eq
        )
        f1, f2 = stream(f1), stream(f2)
        f1 = bounce_back_cells(f1, wall)
        f2 = bounce_back_cells(f2, wall)
    r1, ux1, _ = macroscopic(f1)
    r2, ux2, _ = macroscopic(f2)
    rt = (r1 + r2).clamp(min=1e-12)
    slip = float((ux1 - ux2)[1:-1, :].abs().max())
    ux = ((r1 * ux1 + r2 * ux2) / rt)[:, nx // 2]
    nu_w, nu_g = 1.0 / 6.0, 1.0 / 12.0
    ana = torch.tensor(
        _two_phase_poiseuille_analytical(ny, half, 5e-6, nu_w, nu_g, nu_w / nu_g), device=DEV
    )
    umax = float(ux.max())
    m = ux.abs() > 0.2 * umax
    e = ((ux[m] - ana[m]) / ana[m]).abs()
    return slip, umax, float(e.max()) * 100


def test_t4_poiseuille_component_slip_collapses_under_mixture():
    """H64/12k rung at G_12=-1.0: mixture collapses the inter-component slip
    (<5% of u_max) while self retains a large slip (>20% of u_max).

    REV-C: the pre-registered "mixture error drops sharply" expectation was
    falsified by the explore series; the profile errors of both modes are
    reported in the assertion messages but deliberately not asserted."""
    slip_m, umax_m, err_m = _poiseuille_slip_err("mixture")
    slip_s, umax_s, err_s = _poiseuille_slip_err("self")
    assert slip_m < 0.05 * umax_m, (slip_m, umax_m, err_m)
    assert slip_s > 0.20 * umax_s, (slip_s, umax_s, err_s)
