"""Verification tests for TensorLBM's viscoelastic (Toms) branch.

Covers the machine-checkable acceptance criteria of the Toms node:

* exact Oldroyd-B / FENE-P steady plane-Poiseuille anchors (velocity, ``C``, ``N1``);
* Newtonian limit ``lam -> 0``;
* realizability ``tr C >= 3`` (with a regression guard on the *wrong* ``C >= I``
  constraint);
* energy budget ``P_in = Phi_s + Phi_p + dE/dt``;
* Virk/Prandtl-Karman asymptotes and the ``Delta B(Wi) <= 11.7`` envelope;
* grid doubling convergence;
* three negative controls: ``beta = 1``, upper-convected term off, ``Wi`` beyond
  the stability limit (explicit :class:`ViscoelasticDivergenceError`).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.tensorlbm.viscoelastic import (
    FeneP,
    OldroydB,
    ViscoelasticChannel2D,
    ViscoelasticDivergenceError,
    analytic_channel,
    energy_budget,
    min_eigenvalue,
    project_implementable,
    trace3,
)
from src.tensorlbm.viscoelastic import virk

U_MAX = 0.04
RE = 10.0
BETA = 0.5


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_solver(ny=18, wi=1.0, model="oldroyd_b", L2=None, beta=BETA, lam=None, **kw):
    H = 0.5 * (ny - 2)
    nu0 = U_MAX * H / RE
    if lam is None:
        lam = wi * H / (2.0 * U_MAX)
    kw.setdefault("tol", 1e-8)
    kw.setdefault("n_substeps", 2)
    kw.setdefault("check_every", 100)
    kw.setdefault("max_steps", 200_000)
    return ViscoelasticChannel2D(
        ny=ny, nu0=nu0, beta=beta, lam=lam, model=model, L2=L2,
        u_max_target=U_MAX, **kw
    )


# --------------------------------------------------------------------------- #
# 1. exact constitutive anchors
# --------------------------------------------------------------------------- #
def test_analytic_oldshear_oldroyd_b_is_exact():
    """Exact steady simple shear: C_xx = 1+2Wi^2, C_xy = lam*gamma_dot, trC = 3+2Wi^2."""
    lam, G, nu0, H = 50.0, 2e-3, 0.1, 4.0
    y = np.array([1.0, 2.0, 3.0])
    out = analytic_channel(y, H, G, nu0, lam, beta=0.5, model="oldroyd_b")
    gd = out["gamma_dot"]
    wi = lam * np.abs(gd)
    assert np.allclose(out["C"][:, 0, 0], 1.0 + 2.0 * wi ** 2, rtol=1e-14)
    assert np.allclose(out["C"][:, 1, 1], 1.0, rtol=1e-14)
    assert np.allclose(out["C"][:, 2, 2], 1.0, rtol=1e-14)
    assert np.allclose(out["C"][:, 0, 1], lam * gd, rtol=1e-14)
    assert np.allclose(out["trC"], 3.0 + 2.0 * wi ** 2, rtol=1e-14)
    # first normal-stress difference N1 = 2 eta_p lam gamma_dot^2
    eta_p = 0.5 * nu0
    assert np.allclose(out["N1"], 2.0 * eta_p * lam * gd ** 2, rtol=1e-12)
    # symmetry of the conformation tensor
    assert np.allclose(out["C"], np.swapaxes(out["C"], -1, -2))


def test_analytic_shear_fene_p_solves_the_steady_constitutive_equation():
    """FENE-P steady shear: the closed form is an exact fixed point of the tensor ODE.

    The self-consistent branch is ``trC = a (3 + 2Wi^2/f^2)/f`` with
    ``f = L2/(L2-trC)``; inserting it into
    ``L C + C L^T - (f C - a I)/lam = 0`` must give a machine-zero residual.
    """
    lam, G, nu0, H, L2 = 50.0, 2e-3, 0.1, 4.0, 20.0
    y = np.linspace(-3.99, 3.99, 21)
    y = y[np.abs(y) > 1e-6]          # y = 0 is the trivial zero-shear point
    out = analytic_channel(y, H, G, nu0, lam, beta=0.5, model="fene_p", L2=L2)
    a = L2 / (L2 - 3.0)
    worst = 0.0
    for k in range(y.size):
        C = out["C"][k]
        gd = out["gamma_dot"][k]
        L = np.zeros((3, 3))
        L[0, 1] = gd
        f = L2 / (L2 - out["trC"][k])
        rhs = (f * C - a * np.eye(3)) / lam
        resid = L @ C + C @ L.T - rhs
        scale = np.max(np.abs(rhs))
        assert scale > 0.0
        worst = max(worst, np.max(np.abs(resid)) / scale)
    assert worst < 1e-12, f"FENE-P closed form residual {worst:.3e}"
    assert np.all(out["trC"] < L2), "FENE-P trace must stay below the extensibility bound"
    assert np.allclose(out["C"][:, 0, 1], a * lam * out["gamma_dot"] / (1 + 0.0)
                       * (L2 / (L2 - out["trC"])) ** -2, rtol=1e-12)


def test_fene_p_is_shear_thinning():
    """FENE-P has a shear-dependent viscosity, so its channel profile is NOT parabolic.

    This is the physics that an initially wrong reference solution missed: the
    polymer viscosity is ``eta_p a/f(trC)`` with ``f > 1``, hence the total
    viscosity drops where the shear rate is largest (near the wall).
    """
    H, G, nu0, lam, L2 = 4.0, 1e-3, 0.1, 100.0, 20.0
    y = np.linspace(-3.5, 3.5, 15)
    out = analytic_channel(y, H, G, nu0, lam, beta=0.5, model="fene_p", L2=L2)
    eta = out["eta_local"]
    assert np.all(eta[7:] <= eta[7]), "viscosity must decrease away from the centre line"
    assert eta.max() / eta.min() > 1.02
    u_parabola = G * (H ** 2 - y ** 2) / (2.0 * nu0)
    assert np.max(np.abs(out["u"] - u_parabola)) / u_parabola.max() > 0.05


def test_fene_p_reduces_to_oldroyd_b_as_L2_to_infinity():
    lam, G, nu0, H = 50.0, 2e-3, 0.1, 4.0
    y = np.linspace(0.25, 3.75, 9)
    ob = analytic_channel(y, H, G, nu0, lam, beta=0.5, model="oldroyd_b")
    fp = analytic_channel(y, H, G, nu0, lam, beta=0.5, model="fene_p", L2=1e7)
    assert np.max(np.abs(fp["C"] - ob["C"])) / np.max(np.abs(ob["C"])) < 1e-5
    assert np.max(np.abs(fp["N1"] - ob["N1"])) / np.max(np.abs(ob["N1"])) < 1e-5


def test_analytic_poiseuille_momentum_balance():
    """G H = nu0 gamma_dot_wall and Q = (2/3) u_max * 2H for the exact solution."""
    G, nu0, H, lam = 1e-3, 0.05, 8.0, 100.0
    out = analytic_channel(np.linspace(-H, H, 2001), H, G, nu0, lam, beta=0.5)
    u_max = out["u_max"]
    assert pytest.approx(u_max, rel=1e-12) == G * H ** 2 / (2.0 * nu0)
    gamma_wall = abs(out["gamma_dot"][0])
    assert pytest.approx(G * H, rel=1e-10) == nu0 * gamma_wall
    Q = np.trapezoid(out["u"], np.linspace(-H, H, 2001))
    assert pytest.approx(Q, rel=1e-6) == (2.0 / 3.0) * u_max * 2.0 * H


# --------------------------------------------------------------------------- #
# 2. realizability / projection
# --------------------------------------------------------------------------- #
def test_min_eigenvalue_of_exact_solution_is_below_one():
    """Regression guard: the exact Oldroyd-B solution violates C >= I.

    The exact steady-shear tensor [[1+2Wi^2, Wi],[Wi, 1]] has smallest eigenvalue
    1 + Wi^2 - Wi sqrt(1+Wi^2) = 0.586 at Wi = 1.  An earlier implementation
    clipped the spectrum at 1, which silently destroyed this exact solution --
    the projection must only enforce (i) PSD and (ii) tr C >= 3.
    """
    out = analytic_channel(np.array([1.0]), 4.0, 1e-3, 0.1, 1000.0, beta=0.5)
    # lam*gamma_dot = 1000 * (1e-3*1/0.1) = 10 -> Wi = 10 here; check the formula
    wi = 10.0
    expected = 1.0 + wi ** 2 - wi * np.sqrt(1.0 + wi ** 2)
    assert pytest.approx(min_eigenvalue(out["C"])[0], rel=1e-10) == expected
    assert expected < 1.0


def test_projection_restores_positive_semidefiniteness():
    bad = np.zeros((1, 1, 3, 3))
    bad[0, 0] = np.array([[2.0, 3.0, 0.0], [3.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    C, n = project_implementable(bad)
    assert n > 0
    assert min_eigenvalue(C) >= -1e-12
    assert trace3(C)[0, 0] >= 3.0 - 1e-12


def test_projection_enforces_trace_floor():
    C0 = np.broadcast_to(0.5 * np.eye(3), (4, 1, 3, 3)).copy()
    C, n = project_implementable(C0)
    assert n == 4
    assert np.allclose(trace3(C), 3.0, atol=1e-12)
    assert np.allclose(min_eigenvalue(C), 1.0, atol=1e-12)


def test_projection_is_inert_on_the_exact_solution():
    """The projection must not alter the exact (already realizable) field."""
    out = analytic_channel(np.linspace(-3.5, 3.5, 15), 4.0, 1e-3, 0.05, 200.0, beta=0.5)
    C = out["C"][:, None, :, :]
    Cp, n = project_implementable(C)
    assert n == 0
    assert np.allclose(Cp, C, atol=0.0)


# --------------------------------------------------------------------------- #
# 3. energy budget
# --------------------------------------------------------------------------- #
def test_energy_identity_on_exact_shear():
    """Steady state: tau_p : grad u = Phi_p (elastic energy stationary)."""
    lam, nu0, eta_p = 100.0, 0.1, 0.05
    y = np.linspace(-4.0, 4.0, 2001)
    G, H = 1e-3, 4.0
    out = analytic_channel(y, H, G, nu0, lam, beta=0.5)
    model = OldroydB(lam)
    bud = energy_budget(out["u"], out["gamma_dot"], out["C"], model, eta_p,
                        nu_s=0.5 * nu0, G=G, y=y, pad_to_walls=False)
    assert bud["identity_residual_rel"] < 1e-12
    assert pytest.approx(bud["polymer_work"], rel=1e-10) == bud["Phi_p"]
    # pump power = total (solvent + polymer) dissipation
    assert bud["residual_rel"] < 1e-6
    assert bud["Phi_p"] > 0.0 and bud["Phi_s"] > 0.0


def test_energy_budget_fene_p_identity():
    lam, nu0, eta_p, L2 = 60.0, 0.1, 0.05, 30.0
    y = np.linspace(-4.0, 4.0, 2001)
    out = analytic_channel(y, 4.0, 1e-3, nu0, lam, beta=0.5, model="fene_p", L2=L2)
    bud = energy_budget(out["u"], out["gamma_dot"], out["C"], FeneP(lam, L2), eta_p,
                        nu_s=0.5 * nu0, G=1e-3, y=y, pad_to_walls=False)
    assert bud["identity_residual_rel"] < 1e-12
    assert bud["residual_rel"] < 1e-6
    assert bud["Phi_p"] >= 0.0


# --------------------------------------------------------------------------- #
# 4. coupled LBM solver against the exact anchors
# --------------------------------------------------------------------------- #
def test_lbm_newtonian_poiseuille():
    """beta = 1 recovers the exact parabolic profile to well under 2%."""
    res = make_solver(ny=18, wi=1.0, beta=1.0).run()
    an = analytic_channel(res.y[1:-1], res.H, res.G, res.nu0, res.lam, beta=1.0)
    err = np.max(np.abs(res.u[1:-1, 0] - an["u"])) / np.max(np.abs(an["u"]))
    assert err < 0.02
    assert res.converged


def test_lbm_viscoelastic_poiseuille_velocity_and_n1():
    """Oldroyd-B: velocity <= 2% pointwise and N1 <= 3% against the exact solution."""
    res = make_solver(ny=18, wi=1.0).run()
    an = analytic_channel(res.y[1:-1], res.H, res.G, res.nu0, res.lam, beta=BETA)
    u_err = np.max(np.abs(res.u[1:-1, 0] - an["u"])) / np.max(np.abs(an["u"]))
    n1_err = np.max(np.abs(res.N1[1:-1, 0] - an["N1"])) / np.max(np.abs(an["N1"]))
    assert u_err <= 0.02, f"velocity error {u_err:.3e}"
    assert n1_err <= 0.03, f"N1 error {n1_err:.3e}"


def test_lbm_fene_p_poiseuille_and_extensibility():
    res = make_solver(ny=18, wi=1.0, model="fene_p", L2=100.0).run()
    an = analytic_channel(res.y[1:-1], res.H, res.G, res.nu0, res.lam, beta=BETA,
                          model="fene_p", L2=100.0)
    u_err = np.max(np.abs(res.u[1:-1, 0] - an["u"])) / np.max(np.abs(an["u"]))
    n1_err = np.max(np.abs(res.N1[1:-1, 0] - an["N1"])) / np.max(np.abs(an["N1"]))
    assert u_err <= 0.02
    assert n1_err <= 0.03
    assert np.all(res.trC < 100.0)


def test_realizability_trace_floor_after_lbm_run():
    res = make_solver(ny=18, wi=2.0).run()
    assert np.min(trace3(res.C)) >= 3.0 - 1e-9


def test_newtonian_limit_stress_law_degenerates():
    """lam -> 0: C_xy / lam must reproduce the Newtonian stress law C_xy/lam = gamma_dot."""
    lam_tiny = 1e-6
    res = make_solver(ny=18, wi=1.0, lam=lam_tiny).run()
    g = res.gamma_dot[1:-1, 0]
    c_xy = res.C[1:-1, 0, 0, 1]
    mask = np.abs(g) > 1e-8
    assert np.max(np.abs(c_xy[mask] / lam_tiny - g[mask])) / np.max(np.abs(g[mask])) <= 1e-6


def test_grid_doubling_second_order():
    """Refinement at fixed (Re, Wi): the L2 error must drop by >= 3.2 per doubling."""
    errs = []
    for ny in (10, 18):
        res = make_solver(ny=ny, wi=1.0).run()
        an = analytic_channel(res.y[1:-1], res.H, res.G, res.nu0, res.lam, beta=BETA)
        errs.append(np.linalg.norm(res.u[1:-1, 0] - an["u"]) / np.linalg.norm(an["u"]))
    assert errs[0] / errs[1] >= 3.2, f"ratio {errs[0] / errs[1]:.2f} (errors {errs})"


# --------------------------------------------------------------------------- #
# 5. negative controls
# --------------------------------------------------------------------------- #
def test_negative_control_beta_one_has_no_polymer():
    """NEGATIVE CONTROL 1: beta = 1 -> tau_p == 0, C == I, zero polymer drag."""
    res = make_solver(ny=18, wi=1.0, beta=1.0).run()
    assert np.max(np.abs(res.tau_p)) == 0.0
    assert np.allclose(res.trC, 3.0, atol=1e-12)
    assert res.tau_p[1, 0, 0, 1] == 0.0
    # no polymer contribution to the wall shear -> drag reduction is identically zero
    assert abs(res.nu_s * res.gamma_dot[1, 0]) == pytest.approx(
        abs(res.nu_s * res.gamma_dot[1, 0] + res.tau_p[1, 0, 0, 1]), rel=0.0, abs=0.0
    )


def test_negative_control_without_upper_convective_term():
    """NEGATIVE CONTROL 2: dropping L C + C L^T collapses N1 by far more than 20%."""
    res = make_solver(ny=18, wi=1.0, upper_convective=False).run()
    an = analytic_channel(res.y[1:-1], res.H, res.G, res.nu0, res.lam, beta=BETA)
    n1_err = np.max(np.abs(res.N1[1:-1, 0] - an["N1"])) / np.max(np.abs(an["N1"]))
    assert n1_err > 0.20, f"N1 deviation only {n1_err:.3f}"
    assert np.allclose(res.trC, 3.0, atol=1e-10)


def test_negative_control_wi_beyond_limit_raises():
    """NEGATIVE CONTROL 3: Wi beyond the stability envelope fails explicitly."""
    ny, u_max, re, beta = 6, 0.15, 20.0, 0.2
    H = 0.5 * (ny - 2)
    nu0 = u_max * H / re
    lam = 256.0 * H / (2.0 * u_max)
    solver = ViscoelasticChannel2D(
        ny=ny, nu0=nu0, beta=beta, lam=lam, u_max_target=u_max,
        n_substeps=2, check_every=100, max_steps=4000, tol=1e-8,
    )
    with pytest.raises(ViscoelasticDivergenceError):
        solver.run()
    # and the declared-envelope guard refuses *before* integration
    guarded = ViscoelasticChannel2D(
        ny=ny, nu0=nu0, beta=beta, lam=lam, u_max_target=u_max,
        n_substeps=2, check_every=100, max_steps=100, tol=1e-8, wi_limit=50.0,
    )
    with pytest.raises(ViscoelasticDivergenceError, match="wi_limit"):
        guarded.run()


def test_stability_envelope_is_measured_not_assumed():
    """Wi = 64 is stable in the strongly coupled probe, Wi = 256 is not."""
    ny, u_max, re, beta = 6, 0.15, 20.0, 0.2
    H = 0.5 * (ny - 2)
    nu0 = u_max * H / re
    lam = 64.0 * H / (2.0 * u_max)
    res = ViscoelasticChannel2D(
        ny=ny, nu0=nu0, beta=beta, lam=lam, u_max_target=u_max,
        n_substeps=2, check_every=200, max_steps=8000, tol=1e-8,
    ).run()
    assert np.isfinite(res.u).all()
    assert np.max(np.abs(res.u)) < 0.25


# --------------------------------------------------------------------------- #
# 6. Virk asymptotes and the Delta B envelope
# --------------------------------------------------------------------------- #
def test_virk_newtonian_friction_factor():
    f = virk.friction_newtonian(np.array([1e4]))[0]
    y = 1.0 / np.sqrt(f)
    assert pytest.approx(4.0 * np.log10(1e4 / y) - 0.4, rel=1e-10) == y
    assert 0.0070 < f < 0.0085   # smooth-pipe value at Re = 1e4


def test_virk_mdr_below_newtonian():
    re = np.array([1e4, 1e5])
    fn = virk.friction_newtonian(re)
    fm = virk.friction_mdr(re)
    assert np.all(fm < fn)
    dr = virk.drag_reduction(re, fm)
    assert np.all(dr > 50.0)
    for r, f in zip(re, fm):
        y = 1.0 / np.sqrt(f)
        assert pytest.approx(19.0 * np.log10(r / y) - 32.4, rel=1e-10) == y


def test_delta_B_roundtrip():
    for re, db in ((1e4, 5.0), (1e5, 12.0), (3e4, 0.0)):
        f = virk.friction_from_delta_B(np.array([re]), np.array([db]))
        back = virk.delta_B_from_friction(np.array([re]), f)
        assert float(back[0]) == pytest.approx(db, abs=1e-9)
    # Newtonian asymptote has Delta B = 0 by construction
    re = np.array([1e4, 1e5])
    assert np.allclose(virk.delta_B_from_friction(re, virk.friction_newtonian(re)), 0.0,
                       atol=1e-9)


def test_delta_B_polymer_model_envelope():
    wi = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0])
    db = virk.delta_B_polymer_model(wi)
    assert db[0] == 0.0                                  # Newtonian limit
    assert np.all(np.diff(db) > 0.0)                     # monotone increasing
    assert np.all(db < virk.DELTA_B_MDR)                 # strictly below the MDR bound
    assert pytest.approx(virk.delta_B_polymer_model(1e9), rel=1e-6) == virk.DELTA_B_MDR


def test_mdr_delta_B_reaches_virk_reference_value():
    """Virk's Delta B = 11.7 is attained on the MDR line at a well-defined Re."""
    re = np.logspace(3.0, 7.0, 2001)
    db = virk.delta_B_from_friction(re, virk.friction_mdr(re))
    assert np.min(db) < virk.DELTA_B_MDR < np.max(db)
    idx = int(np.argmin(np.abs(db - virk.DELTA_B_MDR)))
    assert 1e4 < re[idx] < 1e5