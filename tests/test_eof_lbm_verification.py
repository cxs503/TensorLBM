"""Verification tests for the electro-osmotic flow (EOF) capability.

These lock two things that were previously broken and would have silently
passed:

1. ``pnp_solver.run_electroosmosis`` did not solve any flow equation - the
   "numerical" velocity was the analytical Helmholtz-Smoluchowski identity
   evaluated and then compared against itself (gate C3 was circular).
2. The LBM kernel first written for the real solve used a *specular* (free
   slip) wall - reflecting only the wall-normal population component - which
   makes a uniform body force accelerate the whole profile as ``a*t`` without
   ever generating wall shear.  Fixed by full population reversal; locked by
   ``test_wall_is_noslip_not_specular``.
"""

import numpy as np
import pytest

from tensorlbm.verification.cdr_pnp.eof_lbm import (
    charge_density_from_potential,
    debye_huckel_psi,
    eof_exact_velocity,
    eof_profile_errors,
    run_eof_benchmark,
    solve_stokes_fd,
    solve_stokes_lbm,
)
from tensorlbm.verification.cdr_pnp.gates import (
    GATE_THRESHOLDS,
    check_c3_electroosmosis,
)
from tensorlbm.verification.cdr_pnp.pnp_solver import PNPConfig, run_electroosmosis

EPS = 78.5 * 8.854e-12  # water permittivity [F/m]


# ---------------------------------------------------------------------------
# Analytical reference
# ---------------------------------------------------------------------------
def test_poisson_consistency_of_charge_density():
    """rho_e = -eps psi'' must hold exactly for the Debye-Huckel profile.

    For ``psi = zeta cosh(xi/lam)/cosh(H/2lam)`` the second derivative is
    ``psi/lam**2`` analytically, so the identity can be checked without a
    difference stencil; a separate interior-stencil check guards consistency.
    """
    ny, kappa_H = 129, 20.0
    H = float(ny - 2)
    lam = H / kappa_H
    y = np.arange(1, ny - 1) - 0.5
    psi = debye_huckel_psi(y, H, 1.0, lam)
    rho = charge_density_from_potential(psi, EPS, lam)

    psi_pp_analytic = psi / lam**2
    assert np.max(np.abs(EPS * psi_pp_analytic + rho)) / np.max(np.abs(rho)) < 1e-14

    # interior 3-point stencil is 2nd order -> O(dy^2/lambda^2) residual
    d2 = (psi[2:] - 2.0 * psi[1:-1] + psi[:-2])
    resid = EPS * d2 + rho[1:-1]
    assert np.max(np.abs(resid)) / np.max(np.abs(rho)) < 0.05


def test_exact_profile_satisfies_stokes_with_noslip():
    """u = u_hs(1 - psi/zeta) is the exact EOF solution with two no-slip walls."""
    ny, kappa_H, u_hs = 257, 20.0, 0.02
    H = float(ny - 2)
    lam = H / kappa_H
    y = np.arange(1, ny - 1) - 0.5
    psi = debye_huckel_psi(y, H, 1.0, lam)
    u = eof_exact_velocity(psi, 1.0, u_hs)

    # (a) no-slip at both walls (walls sit half a cell beyond the end nodes)
    wall = debye_huckel_psi(np.array([0.0, H]), H, 1.0, lam)
    assert np.max(np.abs(eof_exact_velocity(wall, 1.0, u_hs))) < 1e-15
    # (b) nu u'' = -a(y) with a(y) = a0 psi/zeta and a0 = u_hs nu/lam**2
    nu = 1.0 / 6.0
    a0 = u_hs * nu / lam**2
    u_pp_analytic = -u_hs * psi / lam**2
    assert np.max(np.abs(nu * u_pp_analytic + a0 * psi)) / (a0 * np.max(np.abs(psi))) < 1e-13
    # (c) interior stencil consistency (2nd order)
    d2 = u[2:] - 2.0 * u[1:-1] + u[:-2]
    assert np.max(np.abs(nu * d2 + a0 * psi[1:-1])) / (a0 * np.max(np.abs(psi))) < 0.05


# ---------------------------------------------------------------------------
# Independent discretisations
# ---------------------------------------------------------------------------
def test_fd_solver_is_second_order_accurate():
    e65 = run_eof_benchmark(ny=65, kappa_H=10.0, solver="fd")["errors"]["max_rel"]
    e129 = run_eof_benchmark(ny=129, kappa_H=10.0, solver="fd")["errors"]["max_rel"]
    assert e65 < 0.03 and e129 < 0.03
    assert e129 < e65  # grid convergence


def test_lbm_matches_analytic_and_converges():
    """Real LBM solve agrees with the exact profile, with grid convergence."""
    r33 = run_eof_benchmark(ny=33, kappa_H=10.0, solver="lbm",
                            max_steps=60_000, tol=1e-12)
    r65 = run_eof_benchmark(ny=65, kappa_H=10.0, solver="lbm",
                            max_steps=120_000, tol=1e-12)
    assert r33["converged"] and r65["converged"]
    assert r33["errors"]["max_rel"] < 0.03
    assert r65["errors"]["max_rel"] < 0.03
    assert r65["errors"]["max_rel"] < r33["errors"]["max_rel"]
    # independent discretisation (FD) must corroborate the LBM result
    fd = run_eof_benchmark(ny=65, kappa_H=10.0, solver="fd")
    cross = eof_profile_errors(r65["u"], fd["u"], r65["u_hs"])["max_rel"]
    assert cross < 0.03


def test_lbm_from_rest_reaches_the_same_steady_state():
    """The steady state is not an artefact of initialising at the answer."""
    a = solve_stokes_lbm(ny=33, kappa_H=10.0, u_hs=0.02, init="exact",
                         max_steps=60_000, tol=1e-12)
    b = solve_stokes_lbm(ny=33, kappa_H=10.0, u_hs=0.02, init="rest",
                         max_steps=60_000, tol=1e-12)
    assert b["converged"]
    d = np.max(np.abs(a["u"] - b["u"])) / 0.02
    assert d < 0.01


def test_wall_is_noslip_not_specular():
    """Regression: a specular wall lets a driven channel accelerate as a*t.

    With free-slip walls the momentum injected by the body force is never
    absorbed by wall shear, so |u| grows linearly in time and far exceeds the
    analytic steady value.  The no-slip half-way bounce-back bounds it.
    """
    r = solve_stokes_lbm(ny=33, kappa_H=10.0, u_hs=0.02, init="rest",
                         max_steps=20_000, tol=0.0, check_every=20_000)
    assert np.max(np.abs(r["u"])) < 3.0 * 0.02  # bounded by ~u_hs
    # no-slip signature: the wall-adjacent node moves less than the core
    assert abs(r["u"][0]) < abs(r["u"][len(r["u"]) // 2])


def test_plug_flow_core_velocity_reaches_helmholtz_smoluchowski():
    """Thin-EDL limit: the core velocity must approach u_hs (physical check)."""
    r = run_eof_benchmark(ny=129, kappa_H=20.0, solver="fd")
    assert abs(r["flatness"]) < 0.02
    core_vel = np.mean(r["u"][np.abs(r["y_phys"] - 0.5 * r["H_eff"]) < 0.25 * r["H_eff"]])
    assert abs(core_vel / r["u_hs"] - 1.0) < 0.02


# ---------------------------------------------------------------------------
# Gate C3
# ---------------------------------------------------------------------------
def test_gate_c3_uses_a_real_solve_and_passes():
    res = check_c3_electroosmosis(ny=65)
    assert res["passed"], res
    assert res["u_error"] < GATE_THRESHOLDS["C3_electroosmosis_rel_error"]
    assert res["flatness"] < GATE_THRESHOLDS["C3_electroosmosis_flatness"]
    assert res["nodes_per_debye"] >= GATE_THRESHOLDS["C3_min_nodes_per_debye"]
    # non-circular: two independent discretisations must not agree bitwise
    assert res["lbm_vs_fd"] > 0.0
    assert res["u_error_fd"] > 0.0


def test_gate_c3_negative_controls_reject_wrong_physics():
    res = check_c3_electroosmosis(ny=65)
    nc = res["negative_controls"]
    assert set(nc) == {"force_0.8x", "field_reversed", "no_field"}
    for name, c in nc.items():
        assert c["detected"], (name, c)
        assert c["error"] >= GATE_THRESHOLDS["C3_negative_control_min_error"]
    assert res["passed_negative_controls"]


def test_run_electroosmosis_is_not_the_analytic_identity():
    """The returned profile must come from a solver, not from the formula."""
    cfg = PNPConfig(ny=65, H=20.0 * PNPConfig().kappa_inv)
    r = run_electroosmosis(cfg, method="lbm")
    assert r["method"] == "lbm"
    assert r["u_error"] > 0.0                     # a real solve never matches exactly
    assert np.max(np.abs(r["u"] - r["u_analytic"])) > 0.0
    assert r["lbm_vs_fd"] > 0.0                   # FD is a genuinely different code path
    assert r["converged"]
    # physical units: Helmholtz-Smoluchowski velocity
    assert r["u_hs"] == pytest.approx(3.4752e-4, rel=1e-3)


def test_flatness_metric_is_not_vacuous_in_the_gate_configuration():
    """Historical default H=1um gave kappa_H=3.3 -> no plug core -> flatness 0/NaN."""
    thin = PNPConfig(ny=65, H=1e-6)               # kappa_H ~ 3.3
    assert thin.kappa_H < 5.0
    r = run_electroosmosis(thin, method="fd")
    assert not np.isfinite(r["flatness"])          # no core: metric must be NaN, not 0.0
    good = PNPConfig(ny=65, H=20.0 * PNPConfig().kappa_inv)
    assert np.isfinite(run_electroosmosis(good, method="fd")["flatness"])