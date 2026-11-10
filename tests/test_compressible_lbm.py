"""IN50-W4-N04 -- compressible-flow system verification unit tests (TensorLBM).

These tests cover the *verification metric layer*
(:mod:`tensorlbm.compressible.convergence`) plus the end-to-end wiring of the
compressible solvers used by the node driver
``scripts/run_in50_w4_n04.py``:

* geometry of the NACA0012 test article (analytic 4-digit anchors);
* Richardson extrapolation / observed order / Roache GCI, including the
  non-monotone branch that made the earlier IN421-W3 triplet unusable;
* Rankine-Hugoniot and isentropic relations (analytic anchors);
* terminal-shock strength extraction from a surface ``Cp`` distribution
  (synthetic exact-shock state -> measured ratio must reproduce RH to machine
  precision, which pins the sign/plateau logic);
* N-wave source-signature descriptors;
* Ackeret linear theory and its truncation error vs the exact oblique shock;
* solver smoke tests (small grids, few iterations -- the node's production runs
  live in ``artifacts/compressible_lbm/`` and are not re-run here);
* consistency check of the produced gate record, when present.

Run: ``python -m pytest tests/test_compressible_lbm.py -q``
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tensorlbm.compressible import convergence as cv          # noqa: E402
from tensorlbm.compressible.kinetic_flux import (            # noqa: E402
    contour_loop, naca0012_half_thickness, build_annulus_mesh,
    radial_levels_from_surface, integrate_forces, mesh_quality,
    KineticFluxSolver, SolverConfig,
)
from tensorlbm.compressible.transonic_aero import (           # noqa: E402
    ackeret_pressure_ratio, oblique_shock_pressure_ratio as exact_oblique_ratio,
    surface_mach_from_cp, lift_slope, prandtl_glauert_lift_slope,
    run_naca_case, run_wedge_case,
)

GAMMA = 1.4
ART = os.path.join(ROOT, "artifacts", "compressible_lbm")


def _cp_from_pressure(p, mach_inf, gamma=GAMMA):
    """Cp from a static-pressure ratio ``p/p_inf`` (p_inf = rho_inf = 1)."""
    q = 0.5 * gamma * mach_inf ** 2
    return (p - 1.0) / q


def _p_isentropic(mach, mach_inf, gamma=GAMMA):
    """Static p/p_inf on an isentropic streamline from the freestream state."""
    stag = (1.0 + 0.5 * (gamma - 1.0) * mach_inf ** 2) ** (gamma / (gamma - 1.0))
    return stag * (1.0 + 0.5 * (gamma - 1.0) * mach ** 2) ** (-gamma / (gamma - 1.0))


# ---------------------------------------------------------------------------
# 1. test-article geometry
# ---------------------------------------------------------------------------
def test_naca0012_geometry_analytic_anchors():
    """Chord, thickness and nose radius must match the 4-digit formulae."""
    x, y = contour_loop(naca0012_half_thickness, n_points=4000)
    chord = float(x.max() - x.min())
    t_max = float(y.max() - y.min())
    assert abs(chord - 1.0) < 1e-3
    assert abs(t_max - 0.12) < 1e-4
    # leading-edge radius of a NACA 4-digit section: r = 1.1019 t^2
    nose = np.argsort(x)[:6]
    p = np.array([x[nose], y[nose]]).T
    A = np.hstack([p, np.ones((p.shape[0], 1))])
    b = (p ** 2).sum(axis=1)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    r_le = math.sqrt(sol[2] + (sol[0] / 2) ** 2 + (sol[1] / 2) ** 2)
    assert abs(r_le - 1.1019 * 0.12 ** 2) / (1.1019 * 0.12 ** 2) < 0.05
    # maximum thickness station of a 4-digit section is at x = 0.30 c
    assert abs(float(x[np.argmax(y)]) - 0.30) < 1e-2


def test_mesh_family_is_uniformly_refined():
    """Grid-family helpers must return the refinement ratio actually used."""
    h1 = cv.mesh_size(10080)
    h2 = cv.mesh_size(10080 * 2.25)
    assert cv.refinement_ratio(10080, 10080 * 2.25) == pytest.approx(h1 / h2)
    assert h1 / h2 == pytest.approx(1.5, rel=1e-12)


# ---------------------------------------------------------------------------
# 2. Richardson / GCI
# ---------------------------------------------------------------------------
def test_richardson_recovers_second_order():
    """Synthetic second-order data must give p = 2 and the exact limit."""
    exact, coef, r = 0.2513, 0.041, 1.5
    h = np.array([1.0, 1.0 / r, 1.0 / r ** 2])
    f = exact + coef * h ** 2                       # coarse -> medium -> fine
    out = cv.richardson_3grid(f[0], f[1], f[2], r)
    assert out["monotone"] is True
    assert out["observed_order"] == pytest.approx(2.0, abs=1e-9)
    assert out["extrapolated"] == pytest.approx(exact, rel=1e-9)
    assert out["gci_fine"] > 0.0
    assert out["gci_fine"] < 0.05
    assert out["status"] == "CONVERGED"


def test_richardson_flags_non_monotone_family():
    """A non-monotone triplet must be reported, not silently extrapolated."""
    out = cv.richardson_3grid(1.0, 2.0, 1.5, 1.5)
    assert out["monotone"] is False
    assert out["observed_order"] is None
    assert out["gci_fine"] is None
    assert out["status"] == "NON_MONOTONE"


def test_richardson_rejects_bad_ratio():
    with pytest.raises(ValueError):
        cv.richardson_3grid(1.0, 0.9, 0.85, 1.0)


# ---------------------------------------------------------------------------
# 3. analytic anchors
# ---------------------------------------------------------------------------
def test_normal_shock_rankine_hugoniot():
    """p2/p1 = 4.5 and M2 = 0.577 at M1 = 2 (gamma = 1.4)."""
    assert cv.normal_shock_pressure_ratio(2.0) == pytest.approx(4.5, rel=1e-12)
    assert cv.normal_shock_downstream_mach(2.0) == pytest.approx(0.57735, rel=1e-4)
    # consistency with the solver-side R-H implementation
    from tensorlbm.compressible.solver import normal_shock_relations
    rh = normal_shock_relations(2.0)
    assert cv.normal_shock_pressure_ratio(2.0) == pytest.approx(rh.p_ratio, rel=1e-12)
    assert cv.normal_shock_downstream_mach(2.0) == pytest.approx(rh.M2, rel=1e-12)


def test_isentropic_map_inverts():
    """Cp -> M -> Cp round trip on the isentropic branch."""
    mach_inf = 0.85
    for m in (0.2, 0.6, 0.85, 1.2):
        cp = _cp_from_pressure(_p_isentropic(m, mach_inf), mach_inf)
        assert cv.isentropic_mach_from_cp_local(cp, mach_inf) == pytest.approx(m, rel=1e-8)
        assert surface_mach_from_cp(np.array([cp]), mach_inf)[0] == pytest.approx(m, rel=1e-8)


def test_isentropic_helper_returns_nan_outside_map():
    """Unphysically low pressure must map to NaN, never raise."""
    assert math.isnan(cv.isentropic_mach_from_cp_local(-50.0, 0.85))
    assert math.isnan(cv.isentropic_mach_from_cp_local(float("nan"), 0.85))


def test_ackeret_matches_transonic_module_and_truncation_error():
    """Ackeret vs exact oblique shock: error must shrink with the wedge angle."""
    ack = cv.ackeret_dp_over_p(2.0, 10.0)
    assert ack == pytest.approx(ackeret_pressure_ratio(2.0, 10.0), rel=1e-12)
    exact = cv.oblique_shock_pressure_ratio(2.0, 10.0) - 1.0
    assert exact == pytest.approx(exact_oblique_ratio(2.0, 10.0) - 1.0, rel=1e-12)
    err_10 = abs(ack - exact) / exact
    ack2 = cv.ackeret_dp_over_p(2.0, 2.0)
    exact2 = cv.oblique_shock_pressure_ratio(2.0, 2.0) - 1.0
    err_2 = abs(ack2 - exact2) / exact2
    assert err_10 > err_2 > 0.0          # O(theta^2) truncation
    assert 0.15 < err_10 < 0.30          # ~-20 % at 10 deg, Ma = 2
    with pytest.raises(ValueError):
        cv.ackeret_dp_over_p(0.8, 5.0)


# ---------------------------------------------------------------------------
# 4. shock strength extraction
# ---------------------------------------------------------------------------
def test_shock_strength_exact_synthetic_shock():
    """An exact R-H jump on the surface must be measured back to round-off."""
    mach_inf, m1 = 0.85, 1.25
    p1 = _p_isentropic(m1, mach_inf)
    p2 = p1 * cv.normal_shock_pressure_ratio(m1)
    cp1, cp2 = _cp_from_pressure(p1, mach_inf), _cp_from_pressure(p2, mach_inf)
    x = np.concatenate([np.linspace(0.0, 0.58, 60), np.linspace(0.62, 0.99, 60)])
    cp = np.concatenate([np.full(60, cp1), np.full(60, cp2)])
    out = cv.shock_strength(x, cp, mach_inf)
    assert out["status"] == "SUPERCRITICAL"
    assert out["x_shock"] == pytest.approx(0.58, abs=1e-9)
    assert out["pre_shock_mach_plateau"] == pytest.approx(m1, rel=1e-6)
    assert out["measured_p_ratio"] == pytest.approx(
        cv.normal_shock_pressure_ratio(m1), rel=1e-9)
    assert abs(out["strength_rel_error"]) < 1e-8
    assert out["pre_plateau_cp_std"] < 1e-12


def test_shock_strength_subcritical_flow():
    """No supersonic pocket -> honest SUBCRITICAL, no fabricated shock."""
    mach_inf = 0.3
    x = np.linspace(0.0, 1.0, 50)
    cp = np.full(50, _cp_from_pressure(_p_isentropic(0.35, mach_inf), mach_inf))
    out = cv.shock_strength(x, cp, mach_inf)
    assert out["status"] == "SUBCRITICAL"
    assert out["max_surface_mach"] < 1.0


# ---------------------------------------------------------------------------
# 5. N-wave signature
# ---------------------------------------------------------------------------
def test_nwave_metrics_on_synthetic_signature():
    x = np.array([0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0])
    p = np.array([1.6, 1.6, 1.6, 0.5, 0.5, 0.5, 1.0])
    out = cv.nwave_metrics(x, p)
    assert out["status"] == "OK"
    assert out["front_dp_over_p"] == pytest.approx(0.6, abs=1e-12)
    assert out["rear_dp_over_p"] == pytest.approx(-0.5, abs=1e-12)
    assert out["asymmetry"] == pytest.approx(0.5 / 0.6, rel=1e-12)
    # impulse = signed area under (p/p_inf - 1) over the chord
    assert out["impulse"] == pytest.approx(
        float(np.sum(0.5 * (np.diff(x)) * ((p - 1)[:-1] + (p - 1)[1:]))), rel=1e-12)
    # zero crossing lies between the last compression and the first expansion sample
    assert 0.35 < out["zero_crossing_x"] < 0.5


def test_nwave_metrics_insufficient_data():
    assert cv.nwave_metrics([0.0, 1.0], [1.5, 0.5])["status"] == "INSUFFICIENT_DATA"


# ---------------------------------------------------------------------------
# 6. lift-slope / compressibility theory
# ---------------------------------------------------------------------------
def test_lift_slope_and_prandtl_glauert():
    alphas = [0, 2, 4, 6, 8, 10]
    slope_true = 6.9
    cl = [slope_true * math.radians(a) for a in alphas]
    out = lift_slope(alphas, cl, alpha_max=6.0)
    assert out["dcl_dalpha_per_rad"] == pytest.approx(slope_true, rel=1e-9)
    assert abs(out["zero_lift_offset_deg"]) < 1e-9
    assert prandtl_glauert_lift_slope(0.0) == pytest.approx(2 * math.pi, rel=1e-12)
    assert prandtl_glauert_lift_slope(0.6) > 2 * math.pi


# ---------------------------------------------------------------------------
# 7. solver smoke tests (tiny grids -- production runs are in artifacts/)
# ---------------------------------------------------------------------------
def test_mesh_wall_is_the_airfoil_surface():
    """Wall faces must lie on the body surface (IN421 defect-1 regression).

    ``wall_x/wall_y`` are the midpoints of the wall faces (the wall pressure is
    cell centred in ``i``), so they are compared against the analytic profile,
    not against the node list.  The pre-IN421 behaviour extruded an *offset
    curve* ``r0`` away from the body, which this test would fail.
    """
    x, y = contour_loop(naca0012_half_thickness, n_points=80)
    mesh = build_annulus_mesh(x, y, radial_levels_from_surface(24, 6.0, first_cell=0.002))
    dev = np.abs(np.abs(mesh.wall_y) - naca0012_half_thickness(mesh.wall_x))
    # wall-face midpoints deviate from the analytic profile by the local
    # curvature error of one face (~1.5e-3 at the nose with 80 points), whereas
    # the pre-IN421 offset curve sat r0 = 0.02 c away from the body.
    assert dev.max() < 3e-3               # on the surface, not an offset curve
    assert mesh.wall_x.min() < 2e-3       # the leading edge is reached
    assert mesh.wall_x.max() > 0.998      # and so is the trailing edge


def test_contour_loop_has_no_repeated_node():
    """Closed loop: nose and trailing edge present exactly once (IN50 defect-1)."""
    x, y = contour_loop(naca0012_half_thickness, n_points=80)
    gap = float(np.hypot(x[0] - x[-1], y[0] - y[-1]))
    assert gap > 1e-4                    # not a duplicated node
    assert abs(float(x.min())) < 1e-12    # exact leading-edge node
    assert abs(float(x.max()) - 1.0) < 1e-12   # exact trailing-edge node
    mesh = build_annulus_mesh(x, y, radial_levels_from_surface(24, 6.0, first_cell=0.002))
    from tensorlbm.compressible.kinetic_flux import mesh_quality
    assert mesh_quality(mesh)["n_folded"] == 0


@pytest.mark.parametrize("n_points,nj", [(176, 64), (192, 72)])
def test_mesh_family_is_stable(n_points, nj):
    """Refinement levels that used to blow up must stay subsonic and bounded."""
    x, y = contour_loop(naca0012_half_thickness, n_points=n_points)
    mesh = build_annulus_mesh(x, y, radial_levels_from_surface(nj, 12.0, first_cell=0.002))
    cfg = SolverConfig(mach=0.30, alpha_deg=2.0, cfl=0.4)
    sol = KineticFluxSolver(mesh, cfg)
    sol.run(300)
    assert np.isfinite(sol.W[0]).all() and np.isfinite(sol.W[3]).all()
    assert float(sol.mach_field().max()) < 1.0        # no spurious blow-up
    assert sol.density_residual_norm() < 1e-2


def test_airfoil_solver_smoke_subsonic():
    """Subsonic Euler solve stays finite and produces the expected record."""
    r = run_naca_case(0.30, 2.0, n_points=64, nj=24, n_iter=250, cfl=0.4)
    assert math.isfinite(r["forces"]["cl"]) and math.isfinite(r["forces"]["cd"])
    assert r["forces"]["cl"] > 0.0                      # positive incidence
    assert math.isfinite(r["field"]["max_local_mach"])
    assert r["field"]["max_local_mach"] > 0.0
    for key in ("max_rel_dev_subsonic", "p99_rel_dev_subsonic"):
        assert key in r["isentropic"]
    assert len(r["surface"]["upper_cp"]) > 10


def test_wedge_solver_smoke_nwave():
    """Supersonic double wedge: compression front, expansion rear."""
    r = run_wedge_case(2.0, 10.0, n_points=200, nj=80, n_iter=400, cfl=0.35)
    assert r["front_wedge"]["dp_over_p"] > 0.1          # front wedge compresses
    assert math.isfinite(r["rear_wedge"]["dp_over_p"])
    assert math.isfinite(r["error_vs_exact_oblique_pct"])


# ---------------------------------------------------------------------------
# 8. node artifact record
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(os.path.join(ART, "gates_verified.json")),
                    reason="node gate record not generated in this checkout")
def test_gate_record_is_well_formed():
    with open(os.path.join(ART, "gates_verified.json"), encoding="utf-8") as fh:
        rec = json.load(fh)
    assert rec["node"] == "IN50-W4-N04"
    assert rec["n_gates"] == len(rec["gates"])
    n_pass = sum(1 for g in rec["gates"] if g["pass"])
    assert rec["n_pass"] == n_pass
    assert rec["all_pass"] == (n_pass == len(rec["gates"]))
    for g in rec["gates"]:
        assert set(("id", "criterion", "reference", "threshold", "pass")) <= set(g)
        if g["measured"] is not None:
            assert math.isfinite(float(g["measured"]))


@pytest.mark.skipif(not os.path.exists(os.path.join(ART, "result.json")),
                    reason="node result record not generated in this checkout")
def test_result_record_has_paper_requirements():
    with open(os.path.join(ART, "result.json"), encoding="utf-8") as fh:
        res = json.load(fh)
    for key in ("geometry", "lift", "transonic", "convergence", "boom",
                "gates", "gate_summary", "experimental_anchors"):
        assert key in res
    # geometry must be verified before any physics is quoted
    assert abs(res["geometry"]["chord_error_pct"]) < 0.1
    assert abs(res["geometry"]["thickness_error_pct"]) < 0.1
    # experimental anchors are honestly WITHHELD, never fabricated
    assert res["experimental_anchors"]["agard_cp_303"]["status"] in ("WITHHELD", "AVAILABLE")
    assert "WITHHELD" in res["experimental_anchors"]["verdict"] or \
        res["experimental_anchors"]["agard_cp_303"]["status"] == "AVAILABLE"
    # every quoted number must be finite
    for k, L in res["lift"].items():
        assert all(math.isfinite(v) for v in L["cl"])
        assert math.isfinite(L["lift_slope"]["dcl_dalpha_per_rad"])


@pytest.mark.skipif(not os.path.exists(os.path.join(ART, "raw")),
                    reason="raw case records not present")
def test_raw_shock_strength_recomputes():
    """Recompute the transonic shock from a stored raw surface distribution."""
    path = os.path.join(ART, "raw", "trans_a0p0_ma0p85.json")
    if not os.path.exists(path):
        pytest.skip("transonic alpha=0 raw record absent")
    with open(path, encoding="utf-8") as fh:
        rec = json.load(fh)
    up = rec["surface"]["upper"]
    out = cv.shock_strength(np.array(up["x"]), np.array(up["cp"]), 0.85)
    assert out["status"] == "SUPERCRITICAL"
    assert 0.0 < out["x_shock"] < 1.0
    assert out["max_surface_mach"] > 1.0
    assert math.isfinite(out["measured_p_ratio"])