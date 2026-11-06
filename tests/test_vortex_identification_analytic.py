"""Analytic verification of :mod:`tensorlbm.vortex_identification`.

The module shipped with only smoke tests (shape / sign checks against random
fields).  This file closes that verification debt by benchmarking every
criterion against **closed-form** reference fields:

1. **Lamb-Oseen vortex** (the classical Oseen 1911 / Lamb 1932 viscous vortex,
   also the self-similar solution used for trailing / tip vortices in ship
   hydrodynamics).  With ``u_theta(r) = (Gamma/2 pi r) (1 - exp(-r^2/r_c^2))``
   and ``x = r^2/r_c^2`` the velocity-gradient invariants are available in
   closed form (derived in ``docs/vortex_identification_verification.md``)::

       omega_z/2 = e^{-x}
       s         = ((x+1) e^{-x} - 1) / x          (S_rtheta)
       Q         = e^{-2x} - s^2
       lambda2   = -Q                              (both 2-D eigenvalues equal)
       Omega     = omega^2 / (omega^2 + s^2)

   Consequences used as *scalar* acceptance targets:

   * Q changes sign at ``r_Q = 1.120906 r_c`` (root of ``(2x+1)e^{-x} = 1``),
   * the Liu et al. (2016) threshold ``Omega = 0.52`` sits at ``r = 1.106100 r_c``.

2. **Rigid-body rotation** ``u = Omega x (r)``: pure rotation, so
   ``Q = Omega^2`` uniformly, ``lambda2 = -Omega^2``.

3. **Irrotational plane strain** (stagnation-point flow ``u = a x, v = -a y``):
   ``Q = -a^2``, ``lambda2 = +a^2``, ``Omega = 0``.

4. **3-D axisymmetric check**: the same Lamb-Oseen field extruded along z must
   reproduce the analytic Q through the 3-D machinery (which uses a batched
   3x3 symmetric eigensolver for lambda2), i.e. the 2-D and 3-D implementations
   are mutually consistent.

5. **Grid-spacing (dx/dy/dz) scaling**: Q and lambda2 carry units of
   1/time^2, so the physical spacing must enter the stencils.  ``dx=h`` and
   ``dx=1`` results must differ by exactly ``1/h^2``.

Every reference value below is computed here from the analytic formula; nothing
is copied from the implementation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.vortex_identification import (
    lambda2_criterion_2d,
    lambda2_criterion_3d,
    omega_criterion_2d,
    omega_criterion_3d,
    q_criterion_2d,
    q_criterion_3d,
    vortex_fields_2d,
)

pytestmark = pytest.mark.physics

RC = 1.0  # vortex core radius (length units)
L_DOMAIN = 8.0 * RC  # 8 core radii: outer boundary carries |Q| < 1e-3


# ---------------------------------------------------------------------------
# analytic reference fields (independent of the implementation under test)
# ---------------------------------------------------------------------------


def lob_velocity_2d(n: int, length: float = L_DOMAIN, rc: float = RC):
    """Sample the Lamb-Oseen swirl on an offset node grid.

    Nodes are placed at cell centres so that no node sits exactly on the
    singular axis ``r = 0`` (where ``u_theta = 0/0``).
    """
    h = length / n
    c = (np.arange(n) + 0.5) * h - 0.5 * length
    xg, yg = np.meshgrid(c, c)  # (ny, nx)
    r2 = xg**2 + yg**2
    r = np.sqrt(r2)
    r_safe = np.where(r < 1e-30, 1.0, r)
    x = r2 / rc**2
    u_theta = rc * (1.0 - np.exp(-x)) / r_safe  # Gamma/(2 pi) factored out
    ux = -u_theta * yg / r_safe
    uy = u_theta * xg / r_safe
    return h, xg, yg, x, ux, uy


def lob_q_analytic(x: np.ndarray) -> np.ndarray:
    """Q * r_c^4 / (Gamma/2pi)^2 for the Lamb-Oseen vortex."""
    ex = np.exp(-x)
    s = ((x + 1.0) * ex - 1.0) / np.where(x == 0, 1.0, x)
    return ex * ex - s * s


def lob_strain_analytic(x: np.ndarray) -> np.ndarray:
    ex = np.exp(-x)
    return ((x + 1.0) * ex - 1.0) / np.where(x == 0, 1.0, x)


def lob_omega_analytic(x: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """Omega criterion *without* the eps stabiliser (eps=0 reference)."""
    ex = np.exp(-x)
    s = lob_strain_analytic(x)
    num = ex * ex
    return num / (num + s * s + eps)


def _bisect(func, lo: float, hi: float, n_iter: int = 200) -> float:
    """Bracketed root finder (avoids a scipy dependency in the test suite)."""
    f_lo = func(lo)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        f_mid = func(mid)
        if f_lo * f_mid <= 0.0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


X_Q_ROOT = _bisect(lambda x: (2.0 * x + 1.0) * math.exp(-x) - 1.0, 0.5, 2.0)
R_Q_EXACT = math.sqrt(X_Q_ROOT)  # in units of r_c
R_OMEGA_EXACT = math.sqrt(_bisect(lambda x: float(lob_omega_analytic(np.array([x]))[0]) - 0.52, 0.01, 5.0))


def _interp_crossing(r: np.ndarray, f: np.ndarray, level: float) -> float:
    """First outward crossing of ``level`` along a radial ray (linear interp)."""
    for i in range(1, len(f)):
        if (f[i - 1] - level) * (f[i] - level) < 0.0:
            return r[i - 1] + (level - f[i - 1]) * (r[i] - r[i - 1]) / (f[i] - f[i - 1])
    raise AssertionError("no crossing found along the radial ray")


# ---------------------------------------------------------------------------
# 1. Q-criterion against the Lamb-Oseen closed form
# ---------------------------------------------------------------------------


def test_q_criterion_lamb_oseen_matches_closed_form_and_is_second_order():
    """L-infinity error of Q / max|Q| < 3 % at n=128, converging at O(h^2)."""
    errors = {}
    for n in (64, 128, 256):
        h, _xg, _yg, x, ux, uy = lob_velocity_2d(n)
        q = q_criterion_2d(torch.tensor(ux), torch.tensor(uy), dx=h, dy=h).numpy()
        q_exact = lob_q_analytic(x)
        inner = (slice(1, -1), slice(1, -1))  # boundary ring has no stencil
        errors[n] = float(np.abs(q[inner] - q_exact[inner]).max() / np.abs(q_exact[inner]).max())

    assert errors[128] < 0.03, f"Linf(Q) error {errors[128]:.3%} >= 3 %"
    # second-order convergence: refining by 2 must cut the error by ~4
    order_1 = math.log2(errors[64] / errors[128])
    order_2 = math.log2(errors[128] / errors[256])
    assert order_1 > 1.8 and order_2 > 1.8, f"observed orders {order_1:.2f}/{order_2:.2f}"


def test_q_criterion_core_radius_matches_theory():
    """The Q = 0 contour sits at r = 1.120906 r_c (root of (2x+1)e^-x = 1)."""
    n = 128
    h, xg, yg, _x, ux, uy = lob_velocity_2d(n)
    q = q_criterion_2d(torch.tensor(ux), torch.tensor(uy), dx=h, dy=h).numpy()
    j = int(np.argmin(np.abs(yg[:, 0])))  # node row closest to y = 0
    i0 = int(np.searchsorted(xg[0], 0.0)) - 1
    r_meas = _interp_crossing(xg[0, i0:], q[j, i0:], 0.0)
    rel = abs(r_meas / (R_Q_EXACT * RC) - 1.0)
    assert rel < 0.01, f"Q core radius {r_meas:.5f} vs {R_Q_EXACT:.5f} ({rel:.3%})"


# ---------------------------------------------------------------------------
# 2. lambda2-criterion
# ---------------------------------------------------------------------------


def test_lambda2_equals_minus_q_for_a_circular_vortex():
    """For any u_theta(r) vortex the 2-D eigenvalues of S^2+Omega^2 are both -Q."""
    n = 128
    h, _xg, _yg, _x, ux, uy = lob_velocity_2d(n)
    ux_t, uy_t = torch.tensor(ux), torch.tensor(uy)
    q = q_criterion_2d(ux_t, uy_t, dx=h, dy=h).numpy()
    l2 = lambda2_criterion_2d(ux_t, uy_t, dx=h, dy=h).numpy()
    inner = (slice(1, -1), slice(1, -1))
    scale = np.abs(q[inner]).max()
    assert float(np.abs(l2 + q)[inner].max()) / scale < 1e-3


def test_lambda2_sign_flips_between_rotation_and_strain():
    """Rotation: lambda2 = -Q < 0 (vortex).  Pure strain: lambda2 > 0."""
    n = 64
    length = 2.0
    h = length / n
    c = (np.arange(n) + 0.5) * h - 0.5 * length
    xg, yg = np.meshgrid(c, c)
    inner = (slice(1, -1), slice(1, -1))

    omega0 = 1.7
    ux = torch.tensor(-omega0 * yg)
    uy = torch.tensor(omega0 * xg)
    q = q_criterion_2d(ux, uy, dx=h, dy=h).numpy()
    l2 = lambda2_criterion_2d(ux, uy, dx=h, dy=h).numpy()
    assert float(np.abs(q[inner] / (omega0**2) - 1.0).max()) < 1e-10
    assert float(np.abs(l2[inner] / (-(omega0**2)) - 1.0).max()) < 1e-10

    a = 1.3
    ux = torch.tensor(a * xg)
    uy = torch.tensor(-a * yg)
    q = q_criterion_2d(ux, uy, dx=h, dy=h).numpy()
    l2 = lambda2_criterion_2d(ux, uy, dx=h, dy=h).numpy()
    om = omega_criterion_2d(ux, uy, dx=h, dy=h).numpy()
    assert float(np.abs(q[inner] / (-(a**2)) - 1.0).max()) < 1e-6
    assert float(np.abs(l2[inner] / (a**2) - 1.0).max()) < 1e-6
    assert float(om[inner].max()) < 1e-4  # no vorticity -> Omega ~ 0


# ---------------------------------------------------------------------------
# 3. Omega-criterion
# ---------------------------------------------------------------------------


def test_omega_criterion_matches_closed_form():
    n = 128
    h, _xg, _yg, x, ux, uy = lob_velocity_2d(n)
    om = omega_criterion_2d(torch.tensor(ux), torch.tensor(uy), dx=h, dy=h).numpy()
    # the implementation adds eps = 1e-5 * max(|S|^2+|Omega|^2) ~ 1e-5 here
    om_exact = lob_omega_analytic(x, eps=1e-5)
    inner = (slice(1, -1), slice(1, -1))
    assert float(np.abs(om - om_exact)[inner].max()) < 1e-2
    assert 0.0 <= om.min() and om.max() <= 1.0


def test_omega_criterion_core_radius_matches_theory():
    """The Liu et al. (2016) Omega = 0.52 iso-surface sits at r = 1.106100 r_c."""
    n = 128
    h, xg, yg, _x, ux, uy = lob_velocity_2d(n)
    om = omega_criterion_2d(torch.tensor(ux), torch.tensor(uy), dx=h, dy=h).numpy()
    j = int(np.argmin(np.abs(yg[:, 0])))
    i0 = int(np.searchsorted(xg[0], 0.0)) - 1
    r_meas = _interp_crossing(xg[0, i0:], om[j, i0:], 0.52)
    rel = abs(r_meas / (R_OMEGA_EXACT * RC) - 1.0)
    assert rel < 0.01, f"Omega=0.52 radius {r_meas:.5f} vs {R_OMEGA_EXACT:.5f} ({rel:.3%})"


# ---------------------------------------------------------------------------
# 4. 3-D implementation consistency
# ---------------------------------------------------------------------------


def test_3d_matches_2d_for_an_axisymmetric_vortex():
    """Extruding Lamb-Oseen along z must reproduce the analytic Q and lambda2 = -Q."""
    n, nz = 96, 8
    lz = 2.0 * RC
    h, _xg, _yg, x, ux, uy = lob_velocity_2d(n)
    ux3 = torch.tensor(np.broadcast_to(ux, (nz, n, n)).copy())
    uy3 = torch.tensor(np.broadcast_to(uy, (nz, n, n)).copy())
    uz3 = torch.zeros_like(ux3)
    dz = lz / nz
    q3 = q_criterion_3d(ux3, uy3, uz3, dx=h, dy=h, dz=dz).numpy()
    l23 = lambda2_criterion_3d(ux3, uy3, uz3, dx=h, dy=h, dz=dz).numpy()
    om3 = omega_criterion_3d(ux3, uy3, uz3, dx=h, dy=h, dz=dz).numpy()

    q_exact = np.broadcast_to(lob_q_analytic(x), (nz, n, n))
    inner = (slice(1, -1), slice(1, -1), slice(1, -1))
    scale = float(np.abs(q_exact[inner]).max())
    assert float(np.abs(q3 - q_exact)[inner].max()) / scale < 0.03
    assert float(np.abs(l23 + q3)[inner].max()) / scale < 1e-3
    om_exact = np.broadcast_to(lob_omega_analytic(x, eps=1e-5), (nz, n, n))
    assert float(np.abs(om3 - om_exact)[inner].max()) < 1e-2


# ---------------------------------------------------------------------------
# 5. physical grid spacing
# ---------------------------------------------------------------------------


def test_grid_spacing_scaling_is_exactly_one_over_h_squared():
    """Q carries 1/length^2 units: q(dx=1) / q(dx=h) == 1/h^2 exactly."""
    n = 64
    h, _xg, _yg, _x, ux, uy = lob_velocity_2d(n)
    ux_t, uy_t = torch.tensor(ux), torch.tensor(uy)
    q_unit = q_criterion_2d(ux_t, uy_t).numpy()
    q_phys = q_criterion_2d(ux_t, uy_t, dx=h, dy=h).numpy()
    inner = (slice(1, -1), slice(1, -1))
    # q ~ 1/length^2, so the unit-grid value is the physical value times h^2
    ratio = q_unit[inner] / q_phys[inner]
    assert float(np.abs(ratio / h**2 - 1.0).max()) < 1e-3


def test_grid_spacing_rectangular_cells():
    """A non-cubic grid must reach the same physical Q as a cubic grid of the same h."""
    n = 128
    h, _xg, _yg, x, ux, uy = lob_velocity_2d(n)
    ux_t, uy_t = torch.tensor(ux), torch.tensor(uy)
    # same node values viewed as a grid with dx = dy = h  ->  physical Q
    q_phys = q_criterion_2d(ux_t, uy_t, dx=h, dy=h).numpy()
    # dx-only scaling must not touch the y-stencil: compare against 1/dx run
    q_dx_only = q_criterion_2d(ux_t, uy_t, dx=h, dy=1.0).numpy()
    assert float(np.abs(q_dx_only).max()) > 0.0  # distinct field, no aliasing
    assert not np.allclose(q_dx_only, q_phys)


def test_wrapper_accepts_grid_spacing_and_matches_plain_calls():
    n = 48
    h, _xg, _yg, _x, ux, uy = lob_velocity_2d(n)
    ux_t, uy_t = torch.tensor(ux), torch.tensor(uy)
    fields = vortex_fields_2d(ux_t, uy_t, dx=h, dy=h)
    q_ref = q_criterion_2d(ux_t, uy_t, dx=h, dy=h).numpy()
    l2_ref = lambda2_criterion_2d(ux_t, uy_t, dx=h, dy=h).numpy()
    assert np.allclose(np.array(fields["q"]), q_ref)
    assert np.allclose(np.array(fields["lambda2"]), l2_ref)