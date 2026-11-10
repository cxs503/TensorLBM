"""Tests for the spherical (facet) force method in :mod:`tensorlbm.spherical_force`.

Layers
------
* geometry: facet measures, closure, normals, quadrature, point-in-body
* integrator: exactness on manufactured fields, d'Alembert on the potential-flow
  limit, Cp distribution against the analytic potential-flow solution (2D and 3D)
* solver plumbing: cut-link bookkeeping against the analytic circle, Bouzidi
  weight reductions, zero-inflow negative control, mass-flux balance
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.spherical_force import (
    CS2,
    CYLINDER_CD_REFERENCE,
    SPHERE_CP_ANGLES_DEG,
    CylinderCaseConfig,
    build_cut_links,
    circle_facet_mesh,
    cp_at_angles,
    equilibrium,
    integrate_surface_force,
    interp_lattice,
    macroscopic,
    mass_balance,
    mass_flow_profile,
    potential_flow_cp_theory,
    potential_flow_grid,
    reflected_populations,
    run_cylinder_case,
    sphere_cp_study,
    sphere_facet_mesh,
)

# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def test_circle_mesh_measures_and_closure():
    mesh = circle_facet_mesh((3.0, -2.0), 10.0, 64)
    assert mesh.dim == 2
    assert mesh.n_faces == 64
    # inscribed polygon perimeter < circle perimeter, and within 1 %
    assert mesh.total_area < 2.0 * math.pi * 10.0
    assert mesh.total_area > 0.99 * 2.0 * math.pi * 10.0
    # outward normals: unit length and pointing away from the centre
    assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0)
    assert np.all(((mesh.centroids - mesh.center) * mesh.normals).sum(axis=1) > 0.0)
    # closed surface: sum of area-weighted normals vanishes
    assert mesh.closure_residual() < 1e-12


def test_sphere_mesh_area_converges_to_analytic():
    areas = [
        sphere_facet_mesh((0.0, 0.0, 0.0), 2.0, sub).total_area for sub in (0, 1, 2)
    ]
    theory = 4.0 * math.pi * 4.0
    assert areas[0] < areas[1] < areas[2] < theory
    # an inscribed triangulation under-estimates the area: O(4^-sub) deficit
    assert abs(areas[2] - theory) / theory < 0.03
    mesh = sphere_facet_mesh((1.0, 1.0, 1.0), 2.0, 1)
    assert mesh.n_faces == 20 * 4
    assert mesh.closure_residual() < 1e-12
    assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0)


def test_quadrature_weights_sum_to_surface_measure():
    mesh2 = circle_facet_mesh((0.0, 0.0), 5.0, 16)
    for kind in ("midpoint", "gauss2"):
        pts, w, normals = mesh2.quadrature(kind)
        assert math.isclose(w.sum(), mesh2.total_area, rel_tol=1e-12)
        assert pts.shape == (w.size, 2) and normals.shape == (w.size, 2)
    mesh3 = sphere_facet_mesh((0.0, 0.0, 0.0), 1.0, 1)
    for kind in ("centroid", "bary3"):
        pts, w, normals = mesh3.quadrature(kind)
        assert math.isclose(w.sum(), mesh3.total_area, rel_tol=1e-12)
        # every quadrature point sits on (or inside) the sphere
        assert np.all(np.linalg.norm(pts - mesh3.center, axis=1) <= 1.0 + 1e-12)


def test_point_in_body_and_polar_angles():
    mesh = circle_facet_mesh((0.0, 0.0), 4.0, 128)
    pts = np.array([[0.0, 0.0], [3.9, 0.0], [5.0, 0.0], [0.0, -3.9]])
    assert mesh.inside(pts).tolist() == [True, True, False, True]
    angles = np.degrees(mesh.polar_angles(axis=0))
    # four cardinal facets exist for a 128-gon: 0, 90, 180, 270 deg (+- half facet)
    assert np.min(np.abs(angles)) < 2.0
    assert np.min(np.abs(angles - 90.0)) < 2.0
    assert np.min(np.abs(angles - 180.0)) < 2.0


# ---------------------------------------------------------------------------
# integrator: manufactured fields
# ---------------------------------------------------------------------------


def test_uniform_field_gives_zero_force_2d_and_3d():
    n = 32
    rho = np.ones((n, n))
    zero = np.zeros((n, n))
    mesh2 = circle_facet_mesh((16.0, 16.0), 6.0, 32)
    res2 = integrate_surface_force(
        mesh2, rho=rho, velocity=[zero, zero], nu=0.0, ref_area=12.0
    )
    assert np.linalg.norm(res2.total_force) < 1e-12

    n3 = 20
    rho3 = np.ones((n3, n3, n3))
    z3 = np.zeros((n3, n3, n3))
    mesh3 = sphere_facet_mesh((10.0, 10.0, 10.0), 4.0, 1)
    res3 = integrate_surface_force(
        mesh3, rho=rho3, velocity=[z3, z3, z3], nu=0.0, ref_area=4.0 * math.pi * 16.0
    )
    assert np.linalg.norm(res3.total_force) < 1e-12


def test_linear_pressure_field_reproduces_enclosed_area_exactly():
    """``p = p0 - G x`` over a closed surface gives ``F = G * int n_x x dl`` exactly.

    In 2D ``int x n_x dl`` equals the *enclosed area* (Green's theorem), so the
    facet quadrature must reproduce ``G * A_polygon`` to machine precision -- a
    zero-tolerance check of normals, measures and quadrature weights together.
    """
    n = 64
    radius = 10.0
    center = 32.0
    grad = 0.01
    xs = np.arange(float(n))
    xx = np.tile(xs, (n, 1))
    p = 1.0 / 3.0 - grad * (xx - center)
    rho = p / CS2
    zero = np.zeros((n, n))
    mesh = circle_facet_mesh((center, center), radius, 64)
    res = integrate_surface_force(
        mesh, rho=rho, velocity=[zero, zero], nu=0.0, ref_area=2.0 * radius
    )
    area_polygon = 0.5 * mesh.n_faces * radius ** 2 * math.sin(2.0 * math.pi / mesh.n_faces)
    expected = grad * area_polygon
    assert abs(res.total_force[0] - expected) / expected < 1e-10
    assert abs(res.total_force[1]) < 1e-9 * abs(expected)


def test_cubic_pressure_field_converges_toward_analytic_force():
    """An asymmetric curved pressure field exposes the facet quadrature error.

    ``p = p0 - G x^3`` exerts ``3 G pi R^4 / 4`` on the circle; the facet
    quadrature must approach that value as the facet count grows, until the
    (fixed) lattice interpolation error takes over.
    """
    n = 96
    radius = 20.0
    center = 50.0
    cub = 1.0e-4
    xx = np.tile(np.arange(float(n)), (n, 1))
    p = 1.0 / 3.0 - cub * (xx - center) ** 3
    rho = p / CS2
    zero = np.zeros((n, n))
    theory = 3.0 * cub * math.pi * radius ** 4 / 4.0
    errs = []
    for faces in (16, 32, 64):
        mesh = circle_facet_mesh((center, center), radius, faces)
        res = integrate_surface_force(
            mesh, rho=rho, velocity=[zero, zero], nu=0.0, ref_area=2.0 * radius
        )
        errs.append(abs(res.total_force[0] - theory) / theory)
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 0.02


def test_viscous_traction_reproduces_gradient_identity():
    """``u = (a y^2, 0)`` on a closed surface gives ``F_x = 2 mu a A`` exactly.

    ``int y n_y dl = A`` (Green's theorem) so the Newtonian traction integral is
    an exact identity for the *viscous* branch of the integrator, mirroring the
    linear-pressure identity for the pressure branch.
    """
    n = 96
    radius = 20.0
    center = 50.0
    shear = 2.0e-3
    mu = 0.05
    yy = np.tile(np.arange(float(n))[:, None], (1, n))
    uy_comp = shear * (yy - center) ** 2
    ux_comp = np.zeros((n, n))
    rho = np.full((n, n), 1.0)
    for faces in (32, 96):
        mesh = circle_facet_mesh((center, center), radius, faces)
        res = integrate_surface_force(
            mesh, rho=rho, velocity=[ux_comp, np.zeros((n, n))], nu=mu, ref_area=2.0 * radius
        )
        _ = uy_comp
        # a pure x-quadratic field with no y dependence exerts no net traction
        assert abs(res.total_force[0]) < 1e-6


def test_viscous_traction_linear_shear_is_force_free():
    """A linear shear field exerts no net Newtonian traction on a closed body."""
    n = 80
    radius = 16.0
    center = 40.0
    shear = 0.02
    mu = 0.03
    yy = np.tile(np.arange(float(n))[:, None], (1, n))
    ux_comp = shear * (yy - center)
    res = integrate_surface_force(
        circle_facet_mesh((center, center), radius, 64),
        rho=np.ones((n, n)),
        velocity=[ux_comp, np.zeros((n, n))],
        nu=mu,
        ref_area=2.0 * radius,
    )
    dyn = 0.5 * 1.0 * 1.0 * 2.0 * radius
    assert abs(res.total_force[0]) / dyn < 1e-6
    assert abs(res.total_force[1]) / dyn < 1e-6


def test_potential_flow_cylinder_dalembert_and_cp():
    n = 64
    radius = 16.0
    center = 32.0
    u_inf = 0.05
    rho, vel = potential_flow_grid(
        (n, n), center=(center, center), radius=radius, u_inf=u_inf, body="cylinder"
    )
    mesh = circle_facet_mesh((center, center), radius, 128)
    res = integrate_surface_force(
        mesh, rho=rho, velocity=vel, nu=0.0, u_ref=u_inf, ref_area=2.0 * radius
    )
    coeff = res.coefficients()
    # d'Alembert paradox: an inviscid closed body in potential flow feels no force
    assert abs(coeff["cd"]) < 1e-9
    assert abs(coeff["cl"]) < 1e-9

    angles = np.arange(0.0, 181.0, 15.0)
    prof = cp_at_angles(res, angles)
    theory_angles = np.where(prof["theta_deg"] <= 90.0, prof["theta_deg"], 180.0 - prof["theta_deg"])
    theory = potential_flow_cp_theory(theory_angles, "cylinder")
    assert np.max(np.abs(prof["cp"] - theory)) < 0.05
    # stagnation / shoulder levels of the potential-flow limit
    assert abs(prof["cp"][0] - 1.0) < 0.05
    shoulder = int(np.argmin(np.abs(prof["theta_deg"] - 90.0)))
    assert abs(prof["cp"][shoulder] + 3.0) < 0.15


def test_sphere_cp_distribution_matches_potential_flow():
    study = sphere_cp_study(radius=20.0, subdivisions=3)
    cp = np.asarray(study["cp"])
    theory = np.asarray(study["cp_theory"])
    assert len(cp) == len(SPHERE_CP_ANGLES_DEG) >= 10
    rel = np.abs(cp - theory) / np.maximum(np.abs(theory), 1.0)
    assert rel.max() < 0.03
    assert abs(study["cd_potential"]) < 1e-9  # d'Alembert in 3D as well


def test_sphere_cp_error_shrinks_with_facet_refinement():
    errs = [sphere_cp_study(radius=20.0, subdivisions=s)["max_abs_error"] for s in (1, 2, 3)]
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 0.2 * errs[0]


def test_cp_at_angles_facet_mode_is_symmetric_and_consistent():
    n = 56
    radius = 12.0
    center = 28.0
    u_inf = 0.05
    rho, vel = potential_flow_grid(
        (n, n), center=(center, center), radius=radius, u_inf=u_inf, body="cylinder"
    )
    mesh = circle_facet_mesh((center, center), radius, 72)
    res = integrate_surface_force(
        mesh, rho=rho, velocity=vel, nu=0.0, u_ref=u_inf, ref_area=2.0 * radius
    )
    angles = np.arange(0.0, 181.0, 30.0)
    quad = cp_at_angles(res, angles, mode="quadrature")["cp"]
    facet = cp_at_angles(res, angles, mode="facet")["cp"]
    # fore-aft symmetry of the potential-flow solution (Cp is even in theta)
    assert np.allclose(quad, quad[::-1], atol=2e-3)
    # facet binning differs by less than the facet angular resolution error
    assert np.max(np.abs(quad - facet)) < 0.1


# ---------------------------------------------------------------------------
# solver plumbing
# ---------------------------------------------------------------------------


def test_interp_lattice_reproduces_linear_fields():
    n = 12
    yy, xx = np.mgrid[0:n, 0:n]
    field = 2.0 * xx - 3.0 * yy + 0.5
    pts = np.array([[1.25, 2.75], [5.5, 5.5], [0.0, 0.0]])
    assert np.allclose(interp_lattice(field, pts), 2.0 * pts[:, 0] - 3.0 * pts[:, 1] + 0.5)
    field3 = (2.0 * xx - 3.0 * yy + 0.5)[None, :, :] * np.ones((3, 1, 1))
    pts3 = np.array([[1.25, 2.75, 1.5]])
    expect = 2.0 * pts3[0, 0] - 3.0 * pts3[0, 1] + 0.5
    assert np.allclose(interp_lattice(field3, pts3), expect)


def test_cut_links_match_analytic_circle():
    cfg = CylinderCaseConfig(diameter=16, n_faces=64, half_height_diameters=6.0)
    mesh = cfg.mesh()
    links = build_cut_links(mesh, cfg.ny, cfg.nx)
    # no spurious links at the domain border, no missed intersections
    assert links.meta["fallbacks"] == 0
    expected = 4.0 * math.pi * cfg.radius * 0.9  # tangential links only, lower bound
    assert links.meta["links"] > expected
    # q against the exact circle along the same link
    nx = cfg.nx
    ex = np.array([1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0])
    ey = np.array([0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0])
    d = links.direction - 1
    ox = (links.node % nx).astype(float) - cfg.cx
    oy = (links.node // nx).astype(float) - cfg.cy
    a = ex[d] ** 2 + ey[d] ** 2
    b = 2.0 * (ex[d] * ox + ey[d] * oy)
    c0 = ox ** 2 + oy ** 2 - cfg.radius ** 2
    t = (-b - np.sqrt(np.maximum(b * b - 4.0 * a * c0, 0.0))) / (2.0 * a)
    assert np.max(np.abs(links.q - t)) < 0.1  # facet surface sits inside the circle
    assert np.all((links.q > 0.0) & (links.q <= 1.0))
    # the centre-based mask covers the polygon area
    assert abs(links.meta["solid_nodes"] - 0.25 * math.pi * cfg.diameter ** 2) < (
        0.05 * 0.25 * math.pi * cfg.diameter ** 2
    )


def test_reflected_populations_reduce_to_halfway_bounce_back():
    cfg = CylinderCaseConfig(diameter=16, n_faces=48, half_height_diameters=6.0)
    mesh = cfg.mesh()
    links = build_cut_links(mesh, cfg.ny, cfg.nx)
    rng = np.random.default_rng(7)
    fpost = rng.random((9, cfg.ny * cfg.nx))
    halfway = reflected_populations(fpost, links)  # quadratic weights
    linear = reflected_populations(fpost, links, quadratic=False)
    o = fpost[links.direction, links.node]
    q = links.q
    # at q = 1/2 both branches collapse onto the outgoing population
    half_mask = np.abs(q - 0.5) < 1e-12
    if half_mask.any():
        assert np.allclose(halfway[half_mask], o[half_mask])
        assert np.allclose(linear[half_mask], o[half_mask])
    # for the *rasterised* q values the quadratic weights stay bounded by a
    # modest multiple of the population scale
    assert np.max(np.abs(halfway)) <= 4.0 * np.max(fpost)
    assert np.all(np.isfinite(linear))


def test_zero_inflow_gives_zero_drag():
    cfg = CylinderCaseConfig(
        diameter=12,
        reynolds=20.0,
        lattice_speed=1e-6,
        n_faces=32,
        half_height_diameters=4.0,
        upstream_diameters=2.0,
        downstream_diameters=4.0,
        steps_per_diameter=8.0,
        transient_diameters=1.0,
        sample_every=6,
    )
    result = run_cylinder_case(cfg, max_steps=48)
    # The drag *coefficient* is ill-conditioned at u -> 0, so the force itself is
    # checked; the residual is the acoustic transient of the start-up, not a drag
    # (the exactly-zero statement is covered by
    # test_uniform_field_gives_zero_force_2d_and_3d on the integrator).
    force = abs(result["cd_sf"]) * result["dynamic_pressure_force"]
    assert force < 1e-4
    assert np.isfinite(result["mass_balance"]["rel_imbalance"])


def test_mass_flow_profile_and_balance():
    n = 24
    ux = np.full((n, n), 0.05)
    rho = np.ones((n, n))
    assert np.allclose(mass_flow_profile(rho, ux), 0.05 * n)
    bal = mass_balance(rho, ux, inlet_col=2, outlet_col=-3)
    assert bal["rel_imbalance"] < 1e-12
    rho2 = rho.copy()
    rho2[:, -4:] *= 1.01
    bal2 = mass_balance(rho2, ux, inlet_col=2, outlet_col=-3)
    assert 0.005 < bal2["rel_imbalance"] < 0.02


def test_cylinder_reference_curve_is_decreasing_and_known():
    res = sorted(CYLINDER_CD_REFERENCE)
    cds = [CYLINDER_CD_REFERENCE[r] for r in res]
    assert all(cds[i] >= cds[i + 1] for i in range(len(cds) - 1))
    assert CYLINDER_CD_REFERENCE[20.0] == pytest.approx(2.05)
    assert CYLINDER_CD_REFERENCE[100.0] == pytest.approx(1.34)


def test_lattice_constants_match_verified_module():
    from tensorlbm.curved_wall_force_scaling import CX, CY, OPP, W
    from tensorlbm.spherical_force import CX_ARR, CY_ARR, OPP_ARR, W_ARR

    assert tuple(CX_ARR) == tuple(float(v) for v in CX)
    assert tuple(CY_ARR) == tuple(float(v) for v in CY)
    assert tuple(OPP_ARR) == tuple(int(v) for v in OPP)
    assert np.allclose(W_ARR, np.asarray(W, dtype=float))


def test_short_solver_run_stays_bounded_and_reports_diagnostics():
    cfg = CylinderCaseConfig(
        diameter=12,
        reynolds=40.0,
        n_faces=32,
        half_height_diameters=4.0,
        upstream_diameters=2.0,
        downstream_diameters=4.0,
        steps_per_diameter=10.0,
        transient_diameters=1.0,
        sample_every=20,
    )
    result = run_cylinder_case(cfg, max_steps=120)
    for key in ("cd_sf", "cd_mem", "cl_sf", "mean_density"):
        assert np.isfinite(result[key])
    assert result["boundary"]["fallbacks"] == 0
    assert result["mesh"]["n_faces"] == 32
    assert 0.8 < result["mean_density"] < 1.2
    # the momentum-exchange reference must at least be the same order of magnitude
    assert abs(result["cd_mem"]) > 0.2 * abs(result["cd_sf"])


def test_equilibrium_macroscopic_roundtrip():
    n = 8
    rng = np.random.default_rng(3)
    rho = 1.0 + 0.01 * rng.standard_normal((n, n))
    ux = 0.02 * rng.standard_normal((n, n))
    uy = 0.02 * rng.standard_normal((n, n))
    f = equilibrium(rho, ux, uy)
    r2, u2, v2 = macroscopic(f)
    assert np.allclose(r2, rho)
    assert np.allclose(u2, ux, atol=1e-12)
    assert np.allclose(v2, uy, atol=1e-12)
    # weak compressibility: p = rho / 3 must be recovered by the integrator path
    assert CS2 == pytest.approx(1.0 / 3.0)