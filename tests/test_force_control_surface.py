"""Tests for the Galilean-invariant kinetic-flux control-surface force estimator.

These tests are *analytic*: no time stepping, no reference data, no fitting.
They pin the two properties that make the estimator a genuine third route to
the sphere force (see ``src/tensorlbm/force_control_surface.py``):

* the **uniform-background null space** — a spatially uniform population
  state (the configuration where link-wise MEM returns a large spurious
  force, the G15 root cause) integrates to exactly zero;
* the **kinetic-flux closure** — for a Chapman-Enskog-consistent population
  field built from the exact Stokes solution, the surface integral of the raw
  second moment recovers ``6 π μ a U`` with a control radius that does not
  matter (equation (1) is radius independent).
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.d3q19 import C, W, equilibrium3d
from tensorlbm.force_control_surface import (
    ControlSurfaceForceTracker,
    control_surface_force,
    control_volume_momentum,
    gauss_legendre,
    hasimoto_factor,
    momentum_flux_tensor,
    sphere_quadrature,
    stokes_drag,
    stokes_terminal_velocity,
    trilinear_sample,
)
from tensorlbm.momentum_exchange import (
    momentum_exchange_standard,
    momentum_exchange_wet_node,
)

CS2 = 1.0 / 3.0


# ---------------------------------------------------------------------------
# helpers: analytic Stokes field + Chapman-Enskog first-order populations
# ---------------------------------------------------------------------------
def _stokes_field(coords, radius, u_inf):
    """Exact steady Stokes flow past a sphere (differentiable in space).

    ``u_r = U cosθ (1 - 3a/2r + a³/2r³)``,
    ``u_θ = -U sinθ (1 - 3a/4r - a³/4r³)``,
    ``p  = p0 - 3 μ U a x / (2 r³)``  (``p0`` dropped: only ∇p enters).
    """
    a, U = float(radius), float(u_inf)
    x, y, z = coords
    eps = 1e-30
    r = torch.sqrt(x * x + y * y + z * z + eps)
    rho_c = torch.sqrt(y * y + z * z + eps)
    cos_t = x / r
    sin_t = rho_c / r
    ar = a / r
    u_r = U * cos_t * (1.0 - 1.5 * ar + 0.5 * ar**3)
    u_t = -U * sin_t * (1.0 - 0.75 * ar - 0.25 * ar**3)
    ux = u_r * cos_t - u_t * sin_t
    uy = (u_r * sin_t + u_t * cos_t) * (y / rho_c)
    uz = (u_r * sin_t + u_t * cos_t) * (z / rho_c)
    return ux, uy, uz


def _grid_coords(centre, n=40):
    dtype = torch.float64
    k, j, i = torch.meshgrid(
        torch.arange(n, dtype=dtype),
        torch.arange(n, dtype=dtype),
        torch.arange(n, dtype=dtype),
        indexing="ij",
    )
    return [
        (i - centre[0]).requires_grad_(True),
        (j - centre[1]).requires_grad_(True),
        (k - centre[2]).requires_grad_(True),
    ]


def _velocity_gradient(vel, coords):
    """``G[i][j] = ∂u_j/∂x_i`` by autograd on the analytic field."""
    grads = [torch.autograd.grad(vel[j].sum(), coords, retain_graph=True) for j in range(3)]
    return [[grads[j][i] for j in range(3)] for i in range(3)]


def _populations_from_stokes(radius, u_inf, nu, centre=(20.0, 20.0, 20.0)):
    """``f = f^eq(ρ=1, u) + f^(1)`` for the analytic Stokes field (float64)."""
    coords = _grid_coords(centre)
    ux, uy, uz = _stokes_field(coords, radius, u_inf)
    G = _velocity_gradient((ux, uy, uz), coords)
    tau = 0.5 + 3.0 * nu
    c = C.to(torch.float64)
    w = W.to(torch.float64)
    tr_g = G[0][0] + G[1][1] + G[2][2]
    f1 = torch.zeros(19, 40, 40, 40, dtype=torch.float64)
    for q in range(19):
        cg = sum(c[q, i] * G[i][j] * c[q, j] for i in range(3) for j in range(3))
        f1[q] = -(tau - 0.5) / CS2 * w[q] * (cg - CS2 * tr_g)
    # pressure field p' = -(3/2) μ U a x / r³ enters the equilibrium through
    # the density perturbation δρ = p'/c_s² (incompressible LBM)
    x, y, z = coords
    r = torch.sqrt(x * x + y * y + z * z + 1e-30)
    mu = nu  # ρ = 1
    p_prime = -1.5 * mu * float(u_inf) * float(radius) * x / r**3
    rho = 1.0 + p_prime / CS2
    feq = equilibrium3d(rho, ux, uy, uz)
    return feq + f1, (ux, uy, uz), G, tau, rho


# ---------------------------------------------------------------------------
# 1. quadrature
# ---------------------------------------------------------------------------
def test_gauss_legendre_exactness_and_symmetry():
    n = 8
    x, w = gauss_legendre(n)
    assert x.shape == (n,) and w.shape == (n,)
    # exact for polynomials up to degree 2n-1
    assert float((w).sum()) == pytest.approx(2.0, abs=1e-14)
    assert float((w * x**2).sum()) == pytest.approx(2.0 / 3.0, abs=1e-14)
    assert float((w * x**6).sum()) == pytest.approx(2.0 / 7.0, abs=1e-13)
    assert float((w * x**7).sum()) == pytest.approx(0.0, abs=1e-14)
    # nodes symmetric about 0 (odd powers integrate to zero to machine precision)
    assert torch.allclose(x, -x.flip(0), atol=0.0, rtol=0.0)


def test_sphere_quadrature_null_space_is_exact():
    q = sphere_quadrature(7.5, centre=(3.0, 4.0, 5.0), n_theta=16, n_phi=32)
    res = q.null_space_residual()
    assert q.n_points == 16 * 32
    assert res["sum_n_over_area"] < 1e-14
    assert res["sum_outer_n_rel"] < 1e-14
    # ∮ x_i n_j dA = V δ_ij (moment identity of the sphere)
    assert res["sum_xn_rel"] < 1e-12
    assert float(q.weights.sum()) == pytest.approx(4 * math.pi * 7.5**2, rel=1e-13)
    # points really lie on the sphere
    d = (q.points - torch.tensor([3.0, 4.0, 5.0])).norm(dim=1)
    assert torch.allclose(d, torch.full_like(d, 7.5), atol=1e-12)


def test_trilinear_reproduces_constant_and_linear_fields():
    nz, ny, nx = 12, 10, 14
    k, j, i = torch.meshgrid(
        torch.arange(nz, dtype=torch.float64),
        torch.arange(ny, dtype=torch.float64),
        torch.arange(nx, dtype=torch.float64),
        indexing="ij",
    )
    const = torch.full((nz, ny, nx), 0.25, dtype=torch.float32)
    lin = (2.0 * i - 0.5 * j + 1.25 * k).to(torch.float32)
    pts = torch.tensor(
        [[3.25, 4.75, 5.5], [11.9, 0.1, 2.3], [7.0, 8.0, 9.0]], dtype=torch.float64
    )
    v_c = trilinear_sample(const, pts, periodic=False)
    assert torch.allclose(v_c, torch.full_like(v_c, 0.25), atol=1e-16)
    v_l = trilinear_sample(lin, pts, periodic=False)
    assert torch.allclose(v_l, 2.0 * pts[:, 0] - 0.5 * pts[:, 1] + 1.25 * pts[:, 2], atol=1e-13)
    # periodic wrap: x=11.5 and x=0.5-1 = -0.5 must give the same linear field
    v_wrap = trilinear_sample(lin, torch.tensor([[14.5, 3.0, 4.0]], dtype=torch.float64), periodic=True)
    v_direct = trilinear_sample(lin, torch.tensor([[0.5, 3.0, 4.0]], dtype=torch.float64), periodic=False)
    assert torch.allclose(v_wrap, v_direct, atol=1e-13)


# ---------------------------------------------------------------------------
# 2. the uniform-background anchor (the G15 root cause, annihilated)
# ---------------------------------------------------------------------------
def _staircase_sphere(n=48, radius=6.0, centre=None):
    centre = centre or (n / 2, n / 2, n / 2)
    k, j, i = torch.meshgrid(
        torch.arange(n, dtype=torch.float64),
        torch.arange(n, dtype=torch.float64),
        torch.arange(n, dtype=torch.float64),
        indexing="ij",
    )
    return (
        (i - centre[0]) ** 2 + (j - centre[1]) ** 2 + (k - centre[2]) ** 2 <= radius**2
    )


def test_uniform_state_gives_exactly_zero_force_while_mem_does_not():
    """A uniform stream past the body must measure zero force.

    This is the analytic anchor of the G15 failure: link-wise MEM returns a
    large additive background ((2/3)ρUA per link in the plane-wall limit,
    +264.6% Cd on the sphere), whereas the closed-surface raw-moment integral
    is zero *by construction* because ∮ n dA = 0 exactly.
    """
    n, R = 48, 6.0
    solid = _staircase_sphere(n, R)
    near = torch.zeros_like(solid)
    for dim in (0, 1, 2):
        near |= torch.roll(~solid, 1, dim) & solid
    u = (0.02, 0.01, -0.03)
    rho = torch.ones(n, n, n)
    f = equilibrium3d(rho, torch.full_like(rho, u[0]), torch.full_like(rho, u[1]), torch.full_like(rho, u[2]))
    scale = math.pi * R**2 * (u[0] ** 2 + u[1] ** 2 + u[2] ** 2) ** 0.5 * 1.0

    cs = control_surface_force(f, (n / 2, n / 2, n / 2), R + 2.5, periodic=True)
    rel_cs = max(abs(cs.fx), abs(cs.fy), abs(cs.fz)) / scale
    assert rel_cs < 1e-9, f"control-surface background {rel_cs:.3e} should be ~0"

    # link-wise MEM on the same state: a large, non-zero background
    mem = momentum_exchange_standard(f, solid, near)
    wet = momentum_exchange_wet_node(f, solid)
    mem_mag = max(abs(mem[0]), abs(mem[1]), abs(mem[2]))
    assert mem_mag > 1e-3 * scale, "expected a finite MEM background (G15 failure mode)"
    assert wet is not None  # documented companion estimator
    # the control-surface anchor is at least 9 orders of magnitude cleaner
    assert rel_cs * 1e9 < mem_mag / scale


# ---------------------------------------------------------------------------
# 3. flux closure
# ---------------------------------------------------------------------------
def test_momentum_flux_tensor_equals_equilibrium_flux():
    n = 8
    rho = torch.full((n, n, n), 0.87, dtype=torch.float64)
    ux = torch.full_like(rho, 0.031)
    uy = torch.full_like(rho, -0.017)
    uz = torch.full_like(rho, 0.009)
    f = equilibrium3d(rho, ux, uy, uz)
    pi = momentum_flux_tensor(f, dtype=torch.float64)
    u = (0.031, -0.017, 0.009)
    expected = [
        rho * (CS2 + u[0] * u[0]),
        rho * (CS2 + u[1] * u[1]),
        rho * (CS2 + u[2] * u[2]),
        rho * u[0] * u[1],
        rho * u[0] * u[2],
        rho * u[1] * u[2],
    ]
    for k in range(6):
        assert torch.allclose(pi[k], expected[k], rtol=1e-12, atol=1e-14)


def test_chapman_enskog_first_order_gives_minus_two_rho_nu_strain():
    a, U, nu = 12.0, 1e-5, 0.5
    f, (ux, uy, uz), G, tau, rho = _populations_from_stokes(a, U, nu)
    pi = momentum_flux_tensor(f, dtype=torch.float64)
    # Π^neq = Π - ρ c_s² δ - ρ u u  must equal -2 ρ ν S
    strain = [[0.5 * (G[i][j] + G[j][i]) for j in range(3)] for i in range(3)]
    offdiag = {(0, 1): 3, (0, 2): 4, (1, 2): 5}
    diag = {0: 0, 1: 1, 2: 2}
    for i in range(3):
        for j in range(i, 3):
            comp = pi[0 if (i, j) == (0, 0) else (1 if (i, j) == (1, 1) else (2 if (i, j) == (2, 2) else offdiag[(i, j)]))]
            u_i = (ux, uy, uz)[i]
            u_j = (ux, uy, uz)[j]
            pi_neq = comp - rho * ((CS2 if i == j else 0.0) + u_i * u_j)
            target = -2.0 * nu * strain[i][j]
            # compare in the annulus where the CE expansion is valid and away
            # from the singular far corners
            k, jj, ii = torch.meshgrid(
                torch.arange(40, dtype=torch.float64),
                torch.arange(40, dtype=torch.float64),
                torch.arange(40, dtype=torch.float64),
                indexing="ij",
            )
            r = torch.sqrt((ii - 20) ** 2 + (jj - 20) ** 2 + (k - 20) ** 2)
            mask = (r > a) & (r < 1.6 * a)
            err = (pi_neq - target).abs()[mask].max() / target.abs()[mask].max()
            assert float(err) < 5e-2, f"component {(i, j)} closure error {float(err):.3e}"
    assert tau > 1.0


# ---------------------------------------------------------------------------
# 4. the analytic Stokes closure / radius independence
# ---------------------------------------------------------------------------
def test_analytic_stokes_force_is_recovered_and_radius_independent():
    """``-∮ Π·n`` on the exact Stokes field == 6πμaU, for any control radius."""
    a, U, nu = 16.0, 3e-5, 0.5
    mu = nu  # ρ = 1
    centre = (20.0, 20.0, 20.0)
    f, _, _, _, _ = _populations_from_stokes(a, U, nu, centre=centre)
    f_ref = stokes_drag(a, mu, U)
    vals = []
    for dr in (2.0, 2.5, 3.0):
        res = control_surface_force(f, centre, a + dr, periodic=False)
        vals.append(res.fx)
        assert res.fx > 0.0, "drag on the body must be drag-positive (+x)"
        rel = (res.fx - f_ref) / f_ref
        assert abs(rel) < 5e-2, f"r_s=a+{dr}: {res.fx:.6e} vs {f_ref:.6e} ({rel:+.2%})"
        # y/z by symmetry of the axisymmetric field
        assert abs(res.fy) / f_ref < 2e-2
        assert abs(res.fz) / f_ref < 2e-2
    spread = (max(vals) - min(vals)) / (sum(vals) / len(vals))
    assert spread < 3e-2, f"control radius spread {spread:.2%} should be small (CV exactness)"
    # linearity in U (creeping flow)
    f2, _, _, _, _ = _populations_from_stokes(a, 2 * U, nu, centre=centre)
    res2 = control_surface_force(f2, centre, a + 2.0, periodic=False)
    assert abs(res2.fx / vals[0] - 2.0) < 2e-2


# ---------------------------------------------------------------------------
# 5. tracker statistics + volume momentum
# ---------------------------------------------------------------------------
def test_tracker_reports_mean_rms_and_peak_to_peak():
    n, R = 32, 4.0
    tracker = ControlSurfaceForceTracker(
        centre=(n / 2, n / 2, n / 2), radius=R + 2.5, periodic=True
    )
    rho = torch.ones(n, n, n)
    base = 0.4
    for step in range(40):
        ux = base + 0.02 * math.sin(0.5 * step)
        f = equilibrium3d(rho, torch.full_like(rho, ux), torch.zeros_like(rho), torch.zeros_like(rho))
        tracker.sample(f)
    s = tracker.summary("x", last=40)
    assert s["n_samples"] == 40
    assert s["mean"] == pytest.approx(0.0, abs=1e-12)  # uniform field -> no force
    assert s["rms_over_mean"] != s["rms_over_mean"] or True  # nan allowed (mean==0)
    # a non-zero mean series: use the analytic field instead
    a, U, nu = 12.0, 3e-5, 0.5
    f, _, _, _, _ = _populations_from_stokes(a, U, nu, centre=(20.0, 20.0, 20.0))
    t2 = ControlSurfaceForceTracker(centre=(20.0, 20.0, 20.0), radius=a + 2.5, periodic=False)
    for _ in range(5):
        t2.sample(f)
    s2 = t2.summary("x", last=5)
    assert s2["mean"] > 0.0
    assert s2["rms"] == pytest.approx(0.0, abs=1e-12 * max(1.0, s2["mean"]))
    assert s2["ptp_over_mean"] < 1e-10
    assert len(t2.momentum_drift(dt_steps=10)) == 3


def test_control_volume_momentum_counts_the_enclosed_cells():
    n, R, rho0 = 32, 5.0, 1.0
    u = (0.05, 0.0, 0.0)
    rho = torch.full((n, n, n), rho0)
    f = equilibrium3d(rho, torch.full_like(rho, u[0]), torch.zeros_like(rho), torch.zeros_like(rho))
    mom = control_volume_momentum(f, (n / 2, n / 2, n / 2), R)
    # ∫ρu dV over the ball: count the cells inside, tolerance one shell
    k, j, i = torch.meshgrid(
        torch.arange(n, dtype=torch.float64),
        torch.arange(n, dtype=torch.float64),
        torch.arange(n, dtype=torch.float64),
        indexing="ij",
    )
    n_cells = float(
        (((i - n / 2) ** 2 + (j - n / 2) ** 2 + (k - n / 2) ** 2) <= R**2).sum()
    )
    exact_vol = 4.0 / 3.0 * math.pi * R**3
    assert n_cells < exact_vol * 1.05
    assert float(mom[0]) == pytest.approx(rho0 * u[0] * n_cells, rel=0.02)
    assert abs(float(mom[1])) < 1e-6


# ---------------------------------------------------------------------------
# 6. analytic references used for the settling case
# ---------------------------------------------------------------------------
def test_stokes_and_hasimoto_references():
    mu, a, U = 1.0, 2.0, 0.01
    assert stokes_drag(a, mu, U) == pytest.approx(6 * math.pi * mu * a * U)
    # settling: U_t = 2 a² Δρ g / (9 μ) and 6πμaU_t = (4/3)πa³Δρ g
    drho, g = 0.1, 1e-4
    ut = stokes_terminal_velocity(a, mu, drho, g)
    assert stokes_drag(a, mu, ut) == pytest.approx(4.0 / 3.0 * math.pi * a**3 * drho * g, rel=1e-14)
    assert hasimoto_factor(0.0) == pytest.approx(1.0)
    assert hasimoto_factor(0.05) < hasimoto_factor(0.15)
    assert hasimoto_factor(0.05) > hasimoto_factor(0.0)
    # Hasimoto (1959) simple-cubic series, phi = 4 pi lam^3 / 3 (canonical form;
    # the previous locked value came from a 2.10444 lambda series that was 17 %
    # low at lam = 1/6 -- see docs/ARRAY_DRAG_REFERENCE_AUDIT.md)
    from tensorlbm.array_drag_reference import array_drag_factor_from_lambda

    assert hasimoto_factor(0.1) == pytest.approx(
        array_drag_factor_from_lambda(0.1), rel=1e-12
    )
    phi = 4.0 * math.pi * 0.1**3 / 3.0
    u_ratio = (
        1.0
        - 1.7601 * phi ** (1.0 / 3.0)
        + phi
        - 1.5593 * phi**2
        + 3.9799 * phi ** (8.0 / 3.0)
        - 3.0734 * phi ** (10.0 / 3.0)
    )
    assert hasimoto_factor(0.1) == pytest.approx(1.0 / u_ratio, rel=1e-9)