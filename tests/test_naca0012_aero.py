"""Verification tests for the NACA0012 external-aerodynamics surface.

Two layers are tested:

1. **Reference side** (:mod:`tensorlbm.aero_reference`) -- the panel method is
   validated against the exact analytic cylinder solution and against the
   ``2*pi`` thin-airfoil limit; the integral boundary-layer polar is validated
   against the Blasius flat-plate solution.  These are hard analytic anchors,
   so the reference used to judge the lattice-Boltzmann results is itself
   verified.
2. **Lattice side** (:mod:`tensorlbm.airfoil_polar`) -- link geometry of the
   interpolated bounce-back, equivalence with half-way bounce-back at
   ``q = 1/2``, GCI/polar-fit utilities, and a small end-to-end LBM case that
   must produce finite, physically signed coefficients.

The three-grid polar itself is a cluster run (``scripts/run_polar_sweep.py``);
if its summary is present the test-suite also checks that every acceptance
criterion is recorded with a finite number or an explicit WITHHELD status.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorlbm.aero_reference import (
    THIN_AIRFOIL_CL_SLOPE,
    cylinder_nodes,
    flat_plate_cd_blasius,
    naca0012_half_thickness,
    naca0012_surface,
    solve_vortex_panel,
    squire_young_polar,
    thin_airfoil_cl,
)
from tensorlbm.airfoil_polar import (
    AirfoilPolarConfig,
    apply_wall,
    build_airfoil_grid,
    drag_polar_fit,
    gci_analysis,
    lift_slope,
    run_polar_case,
    wall_resolution_budget,
)

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def test_naca0012_geometry():
    pts = naca0012_surface(200, 0.12, spacing="cosine")
    assert pts.shape == (201, 2)
    assert np.allclose(pts[0], pts[-1]), "contour must be closed"
    xi = np.linspace(0, 1, 4001)
    yt = naca0012_half_thickness(xi, 0.12)
    assert abs(yt.max() - 0.06) < 5e-4, "max half thickness must be 0.06 c (t/c = 0.12)"
    # x stations are monotone on each surface and stay inside [0, 1]
    assert pts[:, 0].min() >= 0.0 and pts[:, 0].max() <= 1.0 + 1e-9
    # both spacing laws cluster panels at the leading edge, but only the
    # cosine law also clusters them at the trailing edge
    def _te_panel(pts):
        # contour starts at the trailing edge
        return abs(pts[0, 0] - pts[1, 0])

    def _le_panel(pts):
        x = pts[:, 0]
        i_le = int(np.argmin(x))
        return abs(x[i_le + 1] - x[i_le])

    pts_quad = naca0012_surface(200, 0.12, spacing="quadratic")
    assert _le_panel(pts) < 1e-3 and _le_panel(pts_quad) < 1e-3
    assert _te_panel(pts) < 1e-3 < _te_panel(pts_quad)


# ---------------------------------------------------------------------------
# panel method: analytic anchors
# ---------------------------------------------------------------------------

def test_panel_cylinder_matches_analytic_cp():
    """Uniform flow past a circle: C_p = 1 - 4 sin^2(theta) exactly."""
    nodes = cylinder_nodes(radius=0.5, n_points=201)
    sol = solve_vortex_panel(alpha_deg=0.0, nodes=nodes)
    beta = np.arctan2(sol.yc, sol.xc)
    cp_exact = 1.0 - 4.0 * np.sin(beta) ** 2
    assert np.max(np.abs(sol.cp - cp_exact)) < 0.06
    assert abs(sol.cl) < 1e-6


def test_panel_thin_airfoil_slope_is_2pi():
    rows = [(a, solve_vortex_panel(200, a, 0.001).cl) for a in (-2.0, 0.0, 2.0, 4.0, 6.0)]
    a = np.radians([r[0] for r in rows])
    cl = np.array([r[1] for r in rows])
    slope = float(np.sum(a * cl) / np.sum(a * a))
    assert abs(slope / THIN_AIRFOIL_CL_SLOPE - 1.0) < 0.015, f"slope={slope}"


def test_panel_thickness_correction():
    """t/c = 0.12 raises the lift slope to ~2 pi (1 + 0.77 t/c) = 6.9."""
    rows = [(a, solve_vortex_panel(200, a, 0.12).cl) for a in (0.0, 2.0, 4.0, 6.0)]
    a = np.radians([r[0] for r in rows])
    cl = np.array([r[1] for r in rows])
    slope = float(np.sum(a * cl) / np.sum(a * a))
    theory = THIN_AIRFOIL_CL_SLOPE * (1.0 + 0.77 * 0.12)
    assert abs(slope / theory - 1.0) < 0.03, f"slope={slope:.4f} theory={theory:.4f}"


def test_panel_zero_lift_and_antisymmetry():
    p0 = solve_vortex_panel(200, 0.0, 0.12)
    assert abs(p0.cl) < 1e-9
    pu = solve_vortex_panel(200, 4.0, 0.12)
    pl = solve_vortex_panel(200, -4.0, 0.12)
    assert abs(pu.cl + pl.cl) < 1e-6, "C_L must be odd in alpha for a symmetric section"
    # the upper surface at +alpha carries the suction peak
    assert pu.cp_at(np.array([0.3]), True)[0] < pu.cp_at(np.array([0.3]), False)[0]
    assert pu.cp_at(np.array([0.3]), True)[0] == pytest.approx(
        pl.cp_at(np.array([0.3]), False)[0], abs=1e-6)


def test_panel_pressure_and_circulation_agree():
    for a in (-2.0, 2.0, 6.0):
        sol = solve_vortex_panel(200, a, 0.12)
        assert abs(sol.cl - sol.cl_pressure) / abs(sol.cl) < 0.02, (
            f"alpha={a}: cl(circ)={sol.cl:.4f} cl(pressure)={sol.cl_pressure:.4f}")
        assert sol.cl == pytest.approx(thin_airfoil_cl(a), rel=0.12)


# ---------------------------------------------------------------------------
# integral boundary layer
# ---------------------------------------------------------------------------

def test_squire_young_flat_plate_matches_blasius():
    for re_v in (1e3, 1e4, 1e5):
        panel = solve_vortex_panel(160, 0.0, 0.001)
        res = squire_young_polar(panel, re_v, 0.001)
        blasius = flat_plate_cd_blasius(re_v)
        assert abs(res.cd / blasius - 1.0) < 0.02, (
            f"Re={re_v:.0e}: cd={res.cd:.6f} blasius={blasius:.6f}")


def test_squire_young_polar_is_finite_and_symmetric():
    p_pos = solve_vortex_panel(160, 4.0, 0.12)
    p_neg = solve_vortex_panel(160, -4.0, 0.12)
    r_pos = squire_young_polar(p_pos, 1e4)
    r_neg = squire_young_polar(p_neg, 1e4)
    assert math.isfinite(r_pos.cd) and r_pos.cd > 0.0
    # the branch split is index based, so +-alpha agree only to the panel
    # discretisation level (~1 %); the value must still be symmetric to that
    assert r_pos.cd == pytest.approx(r_neg.cd, rel=0.02), "symmetric section -> symmetric polar"


# ---------------------------------------------------------------------------
# lattice geometry / bounce-back
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_cfg() -> AirfoilPolarConfig:
    return AirfoilPolarConfig(chord=24, alpha_deg=3.0, re=200.0, u_in=0.1,
                              upstream_c=2.0, downstream_c=4.0, halfheight_c=2.5,
                              sponge_c=2.0, n_steps=1)


def test_bouzidi_link_geometry(small_cfg):
    grid = build_airfoil_grid(small_cfg)
    solid = grid.solid.numpy()
    nx = grid.nx
    from tensorlbm.d2q9 import C as LATTICE_C

    dirs = {i: (int(LATTICE_C[i, 0]), int(LATTICE_C[i, 1])) for i in range(1, 9)}
    assert grid.n_boundary > 0
    for key, link in grid.links.items():
        if not link:
            continue
        cx, cy = dirs[int(key)]
        flat = link["flat"].numpy()
        by, bx = np.divmod(flat, nx)
        assert solid[by + cy, bx + cx].all(), f"dir {key}: neighbour must be solid"
        assert not solid[by, bx].any(), f"dir {key}: link origin must be fluid"
        q = link["q"].numpy()
        assert (q >= 0.05).all() and (q <= 0.95).all()
        bvalid = link["back_valid"].numpy()
        back = link["back"].numpy()
        bby, bbx = np.divmod(back, nx)
        assert ((bby == by - cy) & (bbx == bx - cx)).all()
        assert not solid[bby[bvalid], bbx[bvalid]].any()


def test_interpolated_equals_halfway_bounce_back_at_q_half(small_cfg):
    """At q = 1/2 the Bouzidi scheme must reduce to half-way bounce-back."""
    import torch
    from tensorlbm.d2q9 import equilibrium

    grid = build_airfoil_grid(small_cfg)
    for link in grid.links.values():
        if link:
            link["q"] = torch.full_like(link["q"], 0.5)
            link["lo"] = torch.zeros_like(link["lo"])
    a = math.radians(small_cfg.alpha_deg)
    rho = torch.ones((grid.ny, grid.nx))
    f = equilibrium(rho, torch.full_like(rho, small_cfg.u_in * math.cos(a)),
                    torch.full_like(rho, small_cfg.u_in * math.sin(a))).to(torch.float32)
    fx_i, fy_i = apply_wall(f.clone(), f.clone(), grid, scheme="interpolated")
    fx_s, fy_s = apply_wall(f.clone(), f.clone(), grid, scheme="staircase")
    assert fx_i == pytest.approx(fx_s, rel=1e-6)
    assert fy_i == pytest.approx(fy_s, rel=1e-6)


# ---------------------------------------------------------------------------
# analysis utilities
# ---------------------------------------------------------------------------

def test_gci_cubic_convergence():
    # f(h) = 1 + h^2 with h = 1, 1/1.5, 1/2.25 -> observed order p = 2
    r = 1.5
    f_coarse, f_mid, f_fine = 1.0 + 1.0, 1.0 + 1 / r**2, 1.0 + 1 / r**4
    out = gci_analysis(f_fine, f_mid, f_coarse, r=r)
    assert out["p"] == pytest.approx(2.0, abs=0.05)
    assert out["gci_fine_pct"] > 0.0
    # Richardson extrapolation of a quadratic sequence recovers the h -> 0 value
    assert out["extrapolated"] == pytest.approx(1.0, abs=1e-9)


def test_gci_flags_non_monotone_data():
    out = gci_analysis(1.0, 1.1, 0.95, r=1.5)
    assert out["p"] is None
    assert "non-monotone" in out["note"]


def test_lift_slope_and_polar_fit():
    alphas = np.array([-2.0, 0.0, 2.0, 4.0, 6.0])
    cl = 6.0 * np.radians(alphas)
    fit = lift_slope(alphas, cl)
    assert fit["slope_per_rad"] == pytest.approx(6.0, rel=1e-6)
    assert fit["err_pct"] == pytest.approx(100 * (6.0 / (2 * math.pi) - 1), abs=1e-9)
    assert fit["n_points"] == 4, "only alpha >= 0 is inside the [0, 6] window"
    cd = 0.01 + 0.05 * cl**2
    polar = drag_polar_fit(cl, cd)
    assert polar["cd0"] == pytest.approx(0.01, rel=1e-6)
    assert polar["k"] == pytest.approx(0.05, rel=1e-6)
    assert polar["r2"] == pytest.approx(1.0, abs=1e-9)


def test_wall_resolution_budget_scaling():
    b1 = wall_resolution_budget(1e4, 128, 0.12)
    b2 = wall_resolution_budget(1e6, 128, 0.12)
    assert b2["y_plus_at_chord"] > b1["y_plus_at_chord"]
    assert b2["chord_required_for_y_plus_1"] > b1["chord_required_for_y_plus_1"]
    assert b2["chord_required_for_y_plus_1"] > 1e3   # Re = 6e6 is out of reach


# ---------------------------------------------------------------------------
# end-to-end lattice case (small, fast)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_lbm_case_is_stable_and_signed():
    """End-to-end run at a resolution where the section actually lifts.

    A 24-cell chord is not enough: at Re <= 1e3 the laminar boundary layer is
    ~16 % of the chord, the section becomes a bluff body and C_L is
    indistinguishable from zero (that itself is reported by the polar study).
    Here we only require the solver to produce a positive, finite lift at
    positive incidence on a configuration where the potential-flow start-up
    has already established the circulation.
    """
    cfg = AirfoilPolarConfig(chord=40, alpha_deg=4.0, re=2000.0, u_in=0.12,
                             n_steps=1200, warmup_frac=0.5, sample_interval=10,
                             upstream_c=2.5, downstream_c=6.0, halfheight_c=3.0,
                             sponge_c=3.0, init="potential")
    res = run_polar_case(cfg)
    assert not res["diverged"]
    assert math.isfinite(res["cl"]) and math.isfinite(res["cd"])
    assert res["cl"] > 0.02, "positive incidence must give positive lift"
    assert res["cd"] > 0.0
    assert len(res["cp"]["xc"]) == res["n_boundary_nodes"]
    assert abs(res["cd"]) < 5.0 and abs(res["cl"]) < 5.0
    # the reported relaxation time must be stable
    assert res["config"]["tau"] > 0.5


@pytest.mark.slow
def test_lbm_zero_incidence_has_no_lift(small_cfg):
    cfg = AirfoilPolarConfig(**{**small_cfg.__dict__, "alpha_deg": 0.0,
                                "n_steps": 900, "warmup_frac": 0.5,
                                "sample_interval": 10, "init": "uniform"})
    res = run_polar_case(cfg)
    assert not res["diverged"]
    assert abs(res["cl"]) < 0.05, f"CL(alpha=0) = {res['cl']:.4f} for a symmetric section"


# ---------------------------------------------------------------------------
# artefact cross-check (only when the cluster summary is present)
# ---------------------------------------------------------------------------

SUMMARY = REPO / "artifacts" / "naca0012_polar" / "polar_summary.json"


@pytest.mark.slow
@pytest.mark.skipif(not SUMMARY.exists(), reason="cluster polar summary not present")
def test_polar_summary_records_every_criterion():
    data = json.loads(SUMMARY.read_text())
    crit = data["criteria"]
    for key in ("C1_lift_slope_vs_2pi_le_5pct", "C2_drag_polar_vs_xfoil_le_10pct",
                "C3_cp_rms_vs_experiment_le_0.05", "C4_gci_cl_le_2pct"):
        assert key in crit, f"missing criterion {key}"
    c1 = crit["C1_lift_slope_vs_2pi_le_5pct"]
    if c1["value_per_rad"] is not None:
        # only the physical range is asserted here; whether the value passes
        # the campaign threshold is recorded in c1["pass"], not enforced by the
        # test (a test must not encode the target)
        assert math.isfinite(c1["value_per_rad"])
        assert 0.0 < c1["value_per_rad"] < 10.0
        assert c1["err_pct"] is not None
    assert crit["C2_drag_polar_vs_xfoil_le_10pct"]["status"] in ("WITHHELD", "PASS", "FAIL")
    assert data["grid_convergence"]["rows"], "grid convergence rows must be recorded"
    for fig in ("cl_alpha.png", "cd_cl2.png"):
        if data["criteria"]["C6_figures"].get("pass"):
            assert (Path(data["criteria"]["C6_figures"]["dir"]) / fig).exists()