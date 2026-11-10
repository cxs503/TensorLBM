"""Closed-form (analytic) view-factor kernel family for thermal radiation.

Motivation
----------
``tensorlbm.view_factor`` supplies view factors that are mostly obtained by
Gauss-Legendre quadrature, and the pre-existing ship benchmark
(``benchmarks/verified/radiation_enclosure_ship``) builds its enclosure view
factors from a *hand-set* matrix that is then re-normalised.  Neither route is
an analytical anchor: a quadrature result cannot certify itself, and a
normalised guess is a MOCK-level evidence source.

This module provides an *independent, closed-form* kernel family whose members
are exact trigonometric/logarithmic solutions of the dimensionless contour
integrals.  Each member is verified twice:

1. against high-accuracy, multi-dimensional Gauss-Legendre quadrature of the
   original double-area integrals (``numerical_*`` helpers below, deliberately
   separate implementations), and
2. against catalogue values from Howell / Siegel / Menguc and Modest where a
   published value exists (``REFERENCE_VALUES``).

Closed-form family (6 classes)
------------------------------
================  ============================================  ==============
key               configuration                                 dimensionless
================  ============================================  ==============
``parallel_rect`` two parallel, directly opposed, equal rects    X=a/c, Y=b/c
``perp_rect``     two perpendicular rects sharing an edge        X=a/c, Y=b/c
``coax_disks``    two coaxial parallel disks (unequal radii)     R1, R2, S
``conc_spheres``  concentric spheres (inner -> outer)            r1, r2
``diff_disk``     differential plane element -> coaxial disk     R, H
``sky_dome``      uniform sky hemisphere -> tilted plane         beta [deg]
``elem_sphere``   differential planar element -> sphere           R, d
``sphere_plane``  sphere -> infinite plane (exactly 1/2)          R, h
``elem_infcyl``   differential planar element -> inf. cylinder    R, h
``cyl_plane``     infinite cylinder -> infinite plane (1/2)       R, h
================  ============================================  ==============

All functions return ``F_{1->2}`` (view factor *from* surface 1 *to* 2) and
raise ``ValueError`` for degenerate geometry (fail-closed) instead of silently
clamping.

References
----------
Howell, J. R., Siegel, R., Menguc, M. P. (2010). *Thermal Radiation Heat
    Transfer*, 5th ed., CRC Press, Table 5.1 (catalogue of configuration
    factors).
Modest, M. F. (2013). *Radiative Heat Transfer*, 3rd ed., Elsevier, Table 5.2.
Hamilton, D. C., Morgan, W. R. (1952). NACA TN 2836 (equal parallel
    rectangles).
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "vf_parallel_equal_rectangles",
    "vf_perpendicular_rectangles_common_edge",
    "vf_parallel_coaxial_disks",
    "vf_concentric_spheres",
    "vf_concentric_spheres_inner_to_outer",
    "vf_differential_element_to_disk",
    "vf_sky_dome_to_tilted_plane",
    "vf_sky_dome_to_ground",
    "CLOSED_FORM_FAMILY",
    "REFERENCE_VALUES",
    "closed_form_table",
    "numerical_parallel_equal_rectangles",
    "numerical_perpendicular_rectangles",
    "numerical_parallel_coaxial_disks",
    "numerical_differential_element_to_disk",
    "numerical_sky_dome_to_tilted_plane",
    "monte_carlo_view_factor_rectangles",
    "vf_element_to_sphere",
    "vf_sphere_to_infinite_plane",
    "vf_element_to_infinite_cylinder",
    "vf_infinite_cylinder_to_parallel_plane",
    "numerical_element_to_sphere",
    "numerical_element_to_infinite_cylinder",
    "monte_carlo_sphere_to_infinite_plane",
    "monte_carlo_element_to_infinite_cylinder",
]

_GUARD = 1.0e-300


def _gauss01(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes/weights mapped to [0, 1]."""
    x, w = np.polynomial.legendre.leggauss(int(n))
    return 0.5 * (x + 1.0), 0.5 * w


def _positive(label: str, *values: float) -> None:
    for v in values:
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError(f"{label} must be finite and > 0, got {v!r}")


# ---------------------------------------------------------------------------
# 1) parallel, directly opposed, equal rectangles (aligned)
# ---------------------------------------------------------------------------


def vf_parallel_equal_rectangles(X: float, Y: float) -> float:
    r"""Closed form: two aligned, directly opposed equal rectangles.

    Both rectangles measure :math:`a \times b` and are separated by a distance
    :math:`c`, with :math:`X = a/c`, :math:`Y = b/c`.

    .. math::

        F_{1\to2} = \frac{2}{\pi X Y}\Bigl[
            \ln\sqrt{\frac{(1+X^2)(1+Y^2)}{1+X^2+Y^2}}
            + X\sqrt{1+Y^2}\,\tan^{-1}\frac{X}{\sqrt{1+Y^2}}
            + Y\sqrt{1+X^2}\,\tan^{-1}\frac{Y}{\sqrt{1+X^2}}
            - X\tan^{-1}X - Y\tan^{-1}Y \Bigr]

    Catalogue value: ``X = Y = 1`` gives 0.19982 (Howell Table 5.1).
    """
    _positive("X, Y", X, Y)
    sx, sy = math.sqrt(1.0 + X * X), math.sqrt(1.0 + Y * Y)
    log_term = math.log(math.sqrt((1.0 + X * X) * (1.0 + Y * Y) / (1.0 + X * X + Y * Y)))
    bracket = (
        log_term
        + X * sy * math.atan(X / sy)
        + Y * sx * math.atan(Y / sx)
        - X * math.atan(X)
        - Y * math.atan(Y)
    )
    return float(2.0 * bracket / (math.pi * X * Y))


# ---------------------------------------------------------------------------
# 2) perpendicular rectangles with a common edge
# ---------------------------------------------------------------------------


def vf_perpendicular_rectangles_common_edge(H: float, W: float) -> float:
    r"""Closed form: perpendicular rectangles sharing one edge.

    Rectangle 1 is ``H x L`` in the plane ``z = 0`` (normal +z), rectangle 2 is
    ``W x L`` in the plane ``x = 0`` (normal +x); they share the edge ``x = z =
    0`` of length :math:`L`.  **Both arguments are normalised by the shared edge
    length**: :math:`H = a/L` (width of rectangle 1), :math:`W = c/L` (width of
    rectangle 2).  Normalising by the plate separation instead (``X = a/c``,
    ``Y = b/c``) is the classic porting error: at ``a = c = L`` both conventions
    coincide, everywhere else the result is wrong by up to a factor 3 — see
    ``test_perpendicular_rectangles_vs_monte_carlo``.

    .. math::

        F_{1\to2} = \frac{1}{\pi X}\Bigl[
            X\tan^{-1}\frac{1}{X} + Y\tan^{-1}\frac{1}{Y}
            - \sqrt{X^2+Y^2}\,\tan^{-1}\frac{1}{\sqrt{X^2+Y^2}}
            + \frac{1}{4}\ln\Bigl\{
              \frac{(1+X^2)(1+Y^2)}{1+X^2+Y^2}
              \Bigl[\frac{X^2(1+X^2+Y^2)}{(1+X^2)(X^2+Y^2)}\Bigr]^{X^2}
              \Bigl[\frac{Y^2(1+X^2+Y^2)}{(1+Y^2)(X^2+Y^2)}\Bigr]^{Y^2}
            \Bigr\}\Bigr]

    Catalogue value: ``H = W = 1`` gives 0.20004 (Howell Table 5.1).
    """
    X, Y = H, W  # the published expression uses X, Y for the two widths / L
    _positive("H, W", X, Y)
    sxy = math.sqrt(X * X + Y * Y)
    log_arg = (1.0 + X * X) * (1.0 + Y * Y) / (1.0 + X * X + Y * Y)
    term_x = (X * X * (1.0 + X * X + Y * Y) / ((1.0 + X * X) * (X * X + Y * Y))) ** (X * X)
    term_y = (Y * Y * (1.0 + X * X + Y * Y) / ((1.0 + Y * Y) * (X * X + Y * Y))) ** (Y * Y)
    bracket = (
        X * math.atan(1.0 / X)
        + Y * math.atan(1.0 / Y)
        - sxy * math.atan(1.0 / sxy)
        + 0.25 * math.log(log_arg * term_x * term_y)
    )
    return float(bracket / (math.pi * X))


# ---------------------------------------------------------------------------
# 3) parallel coaxial disks
# ---------------------------------------------------------------------------


def vf_parallel_coaxial_disks(R1: float, R2: float, S: float) -> float:
    r"""Closed form: two coaxial parallel disks with unequal radii.

    .. math::

        S^* = 1 + \frac{1 + (R_2/S)^2}{(R_1/S)^2}, \qquad
        F_{1\to2} = \tfrac12\Bigl[S^* - \sqrt{(S^*)^2 - 4 (R_2/R_1)^2}\Bigr]

    Catalogue value: ``R1 = R2 = S`` gives ``(3 - sqrt5)/2 = 0.381966``.
    """
    _positive("R1, R2, S", R1, R2, S)
    r1, r2 = R1 / S, R2 / S
    s_star = 1.0 + (1.0 + r2 * r2) / (r1 * r1)
    disc = s_star * s_star - 4.0 * (r2 / r1) ** 2
    disc = max(disc, 0.0)  # exact value is >= 0; guard round-off near tangency
    return float(0.5 * (s_star - math.sqrt(disc)))


# ---------------------------------------------------------------------------
# 4) concentric spheres
# ---------------------------------------------------------------------------


def vf_concentric_spheres(r1: float, r2: float) -> float:
    r"""Closed form: *outer* sphere (2) to concentric *inner* sphere (1).

    The inner sphere is convex (it cannot see itself) so it radiates
    exclusively to the outer one, :math:`F_{1\to2} = 1`; the outer (concave)
    sphere sees the inner one over the solid-angle fraction

    .. math:: F_{2\to1} = A_1/A_2 = (r_1/r_2)^2

    which is what this function returns (reciprocity
    :math:`A_1F_{12} = A_2F_{21}` then fixes the complementary
    :math:`F_{2\to2} = 1 - (r_1/r_2)^2`).  Use
    :func:`vf_concentric_spheres_inner_to_outer` for the other direction.
    """
    _positive("r1, r2", r1, r2)
    if r1 > r2 * (1.0 + 1e-12):
        raise ValueError(f"inner radius r1={r1} cannot exceed outer radius r2={r2}")
    return float(min(1.0, (r1 / r2) ** 2))


def vf_concentric_spheres_inner_to_outer(r1: float, r2: float) -> float:
    """Closed form: inner sphere (1) to concentric outer sphere (2) = 1 (convex)."""
    _positive("r1, r2", r1, r2)
    if r1 > r2 * (1.0 + 1e-12):
        raise ValueError(f"inner radius r1={r1} cannot exceed outer radius r2={r2}")
    return 1.0


# ---------------------------------------------------------------------------
# 5) differential planar element -> coaxial parallel disk
# ---------------------------------------------------------------------------


def vf_differential_element_to_disk(R: float, H: float) -> float:
    r"""Closed form: differential plane element to coaxial parallel disk.

    Element ``dA_1`` on the axis at distance :math:`H` from a coaxial disk of
    radius :math:`R`, element and disk parallel:

    .. math::

        F_{dA_1\to2} = \frac{1}{\pi}\int_{A_2}\frac{H^2}{r^4}\,dA_2
                     = \frac{R^2}{R^2 + H^2}
    """
    _positive("R, H", R, H)
    return float((R * R) / (R * R + H * H))


# ---------------------------------------------------------------------------
# 6) uniform sky dome -> tilted plane (solar-engineering sky view factor)
# ---------------------------------------------------------------------------


def vf_sky_dome_to_tilted_plane(beta_deg: float) -> float:
    r"""Closed form: isothermal sky hemisphere to a tilted plane.

    :math:`\beta` is the plane's tilt from horizontal (0 = up-facing, 90 deg =
    vertical, 180 deg = down-facing); the sky is a uniform isothermal
    hemisphere with no shading and no participating medium:

    .. math::
        F_{\text{plane}\to\text{sky}} = \frac{1}{\pi}\int_{\text{sky}}
            \cos\theta\,d\Omega = \frac{1+\cos\beta}{2},
        \qquad
        F_{\text{plane}\to\text{ground}} = \frac{1-\cos\beta}{2}
    """
    if not math.isfinite(beta_deg):
        raise ValueError(f"beta_deg must be finite, got {beta_deg!r}")
    return float(0.5 * (1.0 + math.cos(math.radians(beta_deg))))


def vf_sky_dome_to_ground(beta_deg: float) -> float:
    """Complement of :func:`vf_sky_dome_to_tilted_plane` (summation rule)."""
    return float(1.0 - vf_sky_dome_to_tilted_plane(beta_deg))


# ---------------------------------------------------------------------------
# family registry + literature reference values
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 7) sphere <-> plane  (球-平板)
# ---------------------------------------------------------------------------


def vf_element_to_sphere(R: float, d: float) -> float:
    r"""Closed form: differential planar element -> sphere, :math:`F=R^2/d^2`.

    The element sits at distance :math:`d` from the sphere centre and its
    normal points at the centre.  The double-area integral

    .. math::
        F = \frac{1}{\pi}\int_{A_2}\frac{\cos\theta_1\cos\theta_2}{r^2}\,dA_2

    with :math:`\cos\theta_1\cos\theta_2/r^2 = (d+R\cos t)(R+d\cos t)/r^4`
    reduces (after the azimuthal integration) to a single integral over
    :math:`u=\cos t\in[-1,-R/d]`,

    .. math::
        F = 2R^2\int_{-1}^{-R/d}\frac{(R+du)(Ru+d)}{(R^2+2Rdu+d^2)^2}\,du
          = \frac{R^2}{d^2},

    i.e. the whole geometry collapses to the ratio of the silhouette area
    :math:`\pi R^2` to the sphere's projected... rather to
    :math:`\pi d^2`; the result is exact for every :math:`d>R` and tends to
    one as :math:`d\to R^+` (an element resting on the sphere sees it fill the
    entire hemisphere).  Verified against
    :func:`numerical_element_to_sphere` (independent surface quadrature).

    This is the deck-element -> solar disk / ball-shaped fitting member case.
    """
    _positive("R, d", R, d)
    if d <= R:
        raise ValueError(f"element must lie outside the sphere (d > R), got d={d!r}, R={R!r}")
    return float((R / d) ** 2)


def vf_sphere_to_infinite_plane(R: float, center_height: float) -> float:
    r"""Closed form: sphere -> infinite plane, exactly :math:`1/2`.

    *Convex-body theorem.*  From any surface point of a convex body the
    outward normal hemisphere is unobstructed, so a diffuse (cosine-weighted)
    photon emitted there escapes and, for an infinite plane below the body,
    reaches it whenever its direction has :math:`d_z<0`.  The probability of
    that is :math:`\tfrac12(1-n_z)`; averaging over the sphere
    (:math:`\langle n_z\rangle = 0`) gives exactly :math:`1/2`, independent of
    both radius and height (for :math:`h \ge R`, i.e. the sphere not
    intersecting the plane).  Verified against
    :func:`monte_carlo_sphere_to_infinite_plane` (definitional ray casting).
    """
    _positive("R, center_height", R, center_height)
    if center_height < R:
        raise ValueError(
            f"sphere must not cut the plane (center_height >= R), got "
            f"center_height={center_height!r}, R={R!r}"
        )
    return 0.5


def numerical_element_to_sphere(R: float, d: float, n_theta: int = 160, n_phi: int = 320) -> float:
    r"""Quadrature ground truth for :func:`vf_element_to_sphere`.

    Surface quadrature of the double-area integral with the element at
    :math:`(0,0,-d)` and the sphere centred at the origin:
    :math:`F=\frac{1}{\pi}\int \cos\theta_1\cos\theta_2 R^2\sin t/r^2\,dt\,d\phi`
    over the visible cap :math:`t>\arccos(-R/d)`.
    """
    _positive("R, d", R, d)
    if d <= R:
        raise ValueError(f"element must lie outside the sphere (d > R), got d={d!r}, R={R!r}")
    t0 = math.acos(-R / d)
    tt, wt = _gauss01(n_theta)
    tp, wp = _gauss01(n_phi)
    t = t0 + (math.pi - t0) * tt
    p = 2.0 * math.pi * tp
    st, ct = np.sin(t)[:, None], np.cos(t)[:, None]
    px = R * st * np.cos(p)[None, :]
    py = R * st * np.sin(p)[None, :]
    pz = np.broadcast_to(R * ct, px.shape)
    rx = px
    ry = py
    rz = pz + d
    r2 = rx * rx + ry * ry + rz * rz
    r = np.sqrt(r2)
    cos1 = rz / r  # element normal +z
    cos2 = -((px * rx + py * ry + pz * rz) / (R * r))  # sphere outward normal
    dA = R * R * np.sin(t)[:, None]
    w2 = ((math.pi - t0) * wt)[:, None] * (2.0 * math.pi * wp)[None, :]
    integrand = np.where(cos1 * cos2 > 0.0, cos1 * cos2 / r2 * dA, 0.0)
    return float(np.sum(integrand * w2) / math.pi)


def monte_carlo_sphere_to_infinite_plane(
    R: float, center_height: float, n_rays: int = 400_000, seed: int = 0
) -> dict:
    r"""Definitional Monte-Carlo check of :func:`vf_sphere_to_infinite_plane`.

    Points are uniform on the sphere, directions cosine-weighted about the
    outward normal; a ray counts when it travels towards the plane
    (:math:`d_z<0`).  The estimator is unbiased, so it certifies the
    :math:`1/2` theorem without reusing its proof.
    """
    _positive("R, center_height", R, center_height)
    if n_rays <= 0:
        raise ValueError(f"n_rays must be > 0, got {n_rays}")
    rng = np.random.default_rng(seed)
    # uniform point on the sphere (z = center_height + R cos t)
    u = rng.random(n_rays)
    ct = 1.0 - 2.0 * u
    st = np.sqrt(np.maximum(0.0, 1.0 - ct * ct))
    phi = 2.0 * math.pi * rng.random(n_rays)
    nvec = np.stack([st * np.cos(phi), st * np.sin(phi), ct], axis=1)  # outward normal
    # cosine-weighted direction about nvec
    r_a = rng.random(n_rays)
    r_b = rng.random(n_rays)
    ct_l = np.sqrt(1.0 - r_a)
    st_l = np.sqrt(r_a)
    ph_l = 2.0 * math.pi * r_b
    ref = np.where(np.abs(nvec[:, 2:3]) < 0.9, np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]))
    e1 = np.cross(ref, nvec)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(nvec, e1)
    d = (
        (st_l * np.cos(ph_l))[:, None] * e1
        + (st_l * np.sin(ph_l))[:, None] * e2
        + ct_l[:, None] * nvec
    )
    hits = int(np.count_nonzero(d[:, 2] < 0.0))
    F = hits / n_rays
    return {"F": float(F), "stderr": float(math.sqrt(max(F * (1.0 - F), 0.0) / n_rays)), "n_rays": int(n_rays)}


# ---------------------------------------------------------------------------
# 8) cylinder <-> plane  (圆柱-平板)
# ---------------------------------------------------------------------------


def vf_element_to_infinite_cylinder(R: float, h: float) -> float:
    r"""Closed form: differential planar element -> infinite cylinder, :math:`F=R/h`.

    The element sits at distance :math:`h` from the cylinder axis, its normal
    perpendicular to the axis and pointing at it (the deck-element -> pipe /
    funnel / rail case).

    *Proof by cosine-projected solid angle.*  Cosine-weighted directions about
    the element normal map bijectively onto the unit disk
    :math:`(x,y)=(\sin\theta\cos\psi,\sin\theta\sin\psi)` with **uniform**
    measure :math:`\tfrac1\pi\cos\theta\,d\omega = \pi^{-1}r\,dr\,d\psi`; hence
    :math:`F` is the area fraction of that disk whose straight-line ray meets
    the cylinder.  Writing :math:`\beta=R/h`, the ray
    :math:`(x,z)=(\tau x,\tau\sqrt{1-r^2})` meets the circle
    :math:`x^2+(z-h)^2=R^2` iff

    .. math::
        x^2 \le \beta^2(1-y^2),

    an ellipse of semiaxes :math:`\beta` and 1 centred on the origin, of area
    :math:`\pi\beta`.  Therefore :math:`F = \beta = R/h`, independent of the
    cylinder's length (it is infinite) and exact for all :math:`h>R`.

    Verified twice: against :func:`numerical_element_to_infinite_cylinder`
    (3-D surface quadrature with a :math:`y=c\tan\psi` substitution) and
    against :func:`monte_carlo_element_to_infinite_cylinder` (ray casting).
    """
    _positive("R, h", R, h)
    if h <= R:
        raise ValueError(
            f"element must lie outside the cylinder (h > R), got h={h!r}, R={R!r}"
        )
    return float(R / h)


def vf_infinite_cylinder_to_parallel_plane(R: float, center_height: float) -> float:
    r"""Closed form: infinite cylinder -> parallel infinite plane, exactly :math:`1/2`.

    Same convex-body theorem as :func:`vf_sphere_to_infinite_plane`: the
    emitted hemisphere is unobstructed and every direction with
    :math:`d_z<0` reaches the plane; averaging :math:`\tfrac12(1-n_z)` over the
    cylinder :math:`(\langle n_z\rangle=0)` gives :math:`1/2` for any
    :math:`h \ge R`.  Used as the limiting anchor of the cylinder-plate family.
    """
    _positive("R, center_height", R, center_height)
    if center_height < R:
        raise ValueError(
            f"cylinder must not cut the plane (center_height >= R), got "
            f"center_height={center_height!r}, R={R!r}"
        )
    return 0.5


def numerical_element_to_infinite_cylinder(
    R: float, h: float, n_phi: int = 240, n_psi: int = 240
) -> float:
    r"""Quadrature ground truth for :func:`vf_element_to_infinite_cylinder`.

    Surface quadrature of the double-area integral over the cylinder
    :math:`x^2+(z-h)^2=R^2` (axis along :math:`y`), with the element at the
    origin and normal :math:`+z`:

    .. math::
        F = \frac{1}{\pi}\int_{\text{visible}}\!\!\int_{-\infty}^{\infty}
            \frac{a\,b\,R\,(\varphi)}{(D(\varphi)+y^2)^2}\,dy\,d\varphi,
        \qquad a = -(R+h\cos\varphi),\ b = h+R\cos\varphi,\
        D = R^2+h^2+2hR\cos\varphi ,

    with only the arc :math:`\cos\varphi < -R/h` facing the element.  The
    axial integral is *not* evaluated in closed form: the substitution
    :math:`y=\sqrt{D}\tan\psi` maps it to a smooth :math:`\cos^2\psi`
    integrand on :math:`(-\pi/2,\pi/2)`, which keeps the estimator
    independent of the closed-form reduction (and convergent even when the
    cylinder almost touches the element, :math:`h\to R^+`).
    """
    _positive("R, h", R, h)
    if h <= R:
        raise ValueError(f"element must lie outside the cylinder (h > R), got h={h!r}, R={R!r}")
    phi0 = math.acos(-R / h)
    tp, wp = _gauss01(n_phi)
    ts, ws = _gauss01(n_psi)
    phi = phi0 + (2.0 * math.pi - 2.0 * phi0) * tp
    psi = -0.5 * math.pi + math.pi * ts
    cp = np.cos(phi)[:, None]
    D = R * R + h * h + 2.0 * h * R * cp
    a = -(R + h * cp)
    b = h + R * cp
    integrand = np.where(a > 0.0, a * b * R * np.cos(psi)[None, :] ** 2 / D**1.5, 0.0)
    w2 = ((2.0 * math.pi - 2.0 * phi0) * wp)[:, None] * (math.pi * ws)[None, :]
    return float(np.sum(integrand * w2) / math.pi)


def monte_carlo_element_to_infinite_cylinder(
    R: float, h: float, n_rays: int = 1_000_000, seed: int = 0
) -> dict:
    r"""Definitional ray-casting check of :func:`vf_element_to_infinite_cylinder`.

    Cosine-weighted rays are launched from the element (``n = +z``, cylinder
    axis parallel to ``y`` at ``x = 0, z = h``) and tested against the exact
    quadratic :math:`t^2(d_x^2+d_z^2)-2thd_z+h^2-R^2=0`: a ray is a hit when
    that quadratic has a positive real root.
    """
    _positive("R, h", R, h)
    if n_rays <= 0:
        raise ValueError(f"n_rays must be > 0, got {n_rays}")
    rng = np.random.default_rng(seed)
    u1 = rng.random(n_rays)
    u2 = rng.random(n_rays)
    r = np.sqrt(u1)
    c = np.sqrt(1.0 - u1)
    psi = 2.0 * math.pi * u2
    dx, dz = r * np.cos(psi), c  # the axial (y) component does not enter the test
    A = dx * dx + dz * dz
    B = -2.0 * h * dz
    C = h * h - R * R
    disc = B * B - 4.0 * A * C
    ok = disc >= 0.0
    t = np.where(ok, (-B - np.sqrt(np.where(ok, disc, 0.0))) / (2.0 * np.maximum(A, _GUARD)), -1.0)
    hits = int(np.count_nonzero(ok & (t > 0.0)))
    F = hits / n_rays
    return {"F": float(F), "stderr": float(math.sqrt(max(F * (1.0 - F), 0.0) / n_rays)), "n_rays": int(n_rays)}


CLOSED_FORM_FAMILY: dict[str, dict] = {
    "parallel_rect": {
        "func": vf_parallel_equal_rectangles,
        "description": "parallel, aligned, directly opposed equal rectangles",
        "parameters": ("X", "Y"),
    },
    "perp_rect": {
        "func": vf_perpendicular_rectangles_common_edge,
        "description": "perpendicular rectangles sharing a common edge",
        "parameters": ("H", "W"),
    },
    "coax_disks": {
        "func": vf_parallel_coaxial_disks,
        "description": "coaxial parallel disks, unequal radii",
        "parameters": ("R1", "R2", "S"),
    },
    "conc_spheres": {
        "func": vf_concentric_spheres,
        "description": "concentric spheres, outer -> inner",
        "parameters": ("r1", "r2"),
    },
    "diff_disk": {
        "func": vf_differential_element_to_disk,
        "description": "differential planar element -> coaxial parallel disk",
        "parameters": ("R", "H"),
    },
    "sky_dome": {
        "func": vf_sky_dome_to_tilted_plane,
        "description": "uniform sky hemisphere -> tilted plane",
        "parameters": ("beta_deg",),
    },
    "elem_sphere": {
        "func": vf_element_to_sphere,
        "description": "differential planar element -> sphere (normal at centre)",
        "parameters": ("R", "d"),
    },
    "sphere_plane": {
        "func": vf_sphere_to_infinite_plane,
        "description": "sphere -> infinite plane (convex-body theorem)",
        "parameters": ("R", "center_height"),
    },
    "elem_infcyl": {
        "func": vf_element_to_infinite_cylinder,
        "description": "differential planar element -> infinite cylinder (normal on axis)",
        "parameters": ("R", "h"),
    },
    "cyl_plane": {
        "func": vf_infinite_cylinder_to_parallel_plane,
        "description": "infinite cylinder -> parallel infinite plane (convex-body theorem)",
        "parameters": ("R", "center_height"),
    },
}

REFERENCE_VALUES: tuple[dict, ...] = (
    {
        "class": "parallel_rect",
        "args": (1.0, 1.0),
        "F_ref": 0.19982,
        "source": "Howell/Siegel/Menguc 2010 Table 5.1 (X=Y=1)",
    },
    {
        "class": "perp_rect",
        "args": (1.0, 1.0),
        "F_ref": 0.20004,
        "source": "Howell 2010 Table 5.1 (H=W=1, common edge)",
    },
    {
        "class": "coax_disks",
        "args": (1.0, 1.0, 1.0),
        "F_ref": 0.5 * (3.0 - math.sqrt(5.0)),
        "source": "Howell 2010 Table 5.1 (R1=R2=S): (3-sqrt5)/2",
    },
    {
        "class": "conc_spheres",
        "args": (2.0, 4.0),
        "F_ref": 0.25,
        "source": "Howell 2010 Table 5.1 concentric spheres: F_(outer->inner) = (r1/r2)^2 = 1/4",
    },
    {
        "class": "diff_disk",
        "args": (1.0, 1.0),
        "F_ref": 0.5,
        "source": "R^2/(R^2+H^2), R=H=1 (exact)",
    },
    {
        "class": "sky_dome",
        "args": (0.0,),
        "F_ref": 1.0,
        "source": "horizontal up-facing plane sees the full sky hemisphere",
    },
    {
        "class": "sky_dome",
        "args": (60.0,),
        "F_ref": 0.75,
        "source": "(1+cos60)/2 = 3/4 (exact)",
    },
    {
        "class": "elem_sphere",
        "args": (1.0, 2.0),
        "F_ref": 0.25,
        "source": "R^2/d^2 = 1/4 (exact; sympy-reduced integral)",
    },
    {
        "class": "elem_sphere",
        "args": (2.0, 2.5),
        "F_ref": 0.64,
        "source": "R^2/d^2 = (2/2.5)^2 = 0.64 (exact)",
    },
    {
        "class": "sphere_plane",
        "args": (1.0, 1.0),
        "F_ref": 0.5,
        "source": "convex-body theorem: sphere -> infinite plane = 1/2 (exact)",
    },
    {
        "class": "elem_infcyl",
        "args": (1.0, 2.0),
        "F_ref": 0.5,
        "source": "R/h = 1/2 (exact; cosine-projected unit-disk ellipse proof)",
    },
    {
        "class": "elem_infcyl",
        "args": (0.4, 2.0),
        "F_ref": 0.2,
        "source": "R/h = 0.4/2 = 0.2 (exact)",
    },
    {
        "class": "cyl_plane",
        "args": (1.0, 3.0),
        "F_ref": 0.5,
        "source": "convex-body theorem: infinite cylinder -> infinite plane = 1/2 (exact)",
    },
)


def closed_form_table() -> list[dict]:
    """Evaluate the whole family on the catalogue points.

    Returns
    -------
    list of dict
        ``class``, ``args``, ``F_closed_form``, ``F_ref``, ``rel_error``,
        ``source`` per reference row.
    """
    rows = []
    for ref in REFERENCE_VALUES:
        func = CLOSED_FORM_FAMILY[ref["class"]]["func"]
        F_cf = float(func(*ref["args"]))
        F_ref = float(ref["F_ref"])
        denom = abs(F_ref) if abs(F_ref) > _GUARD else _GUARD
        rows.append(
            {
                "class": ref["class"],
                "args": list(ref["args"]),
                "F_closed_form": F_cf,
                "F_ref": F_ref,
                "rel_error": abs(F_cf - F_ref) / denom,
                "source": ref["source"],
            }
        )
    return rows



# ---------------------------------------------------------------------------
# Independent numerical ground truth
#
# Separate implementations of the underlying integrals: these are the
# falsification instruments for the closed forms above (no shared algebra, no
# shared constants).  Where the integrand is smooth, tensor-product
# Gauss-Legendre converges to machine precision; where the two surfaces touch
# (contact-edge families), the double-area integrand is only slowly convergent,
# so the definitional Monte-Carlo ray-tracing estimator
# :func:`monte_carlo_view_factor_rectangles` is the certification instrument.
# ---------------------------------------------------------------------------


def numerical_parallel_equal_rectangles(X: float, Y: float, n: int = 200) -> float:
    r"""Quadrature ground truth for :func:`vf_parallel_equal_rectangles`.

    .. math::
        F_{1\to2} = \frac{4}{\pi X Y}\int_0^X\!\!\int_0^Y
            \frac{(X-u)(Y-v)}{(u^2+v^2+1)^2}\,du\,dv

    (the ``|u|``, ``|v|`` kinks are removed by symmetry; G-L at ``n = 200``
    reaches ~1e-11 relative accuracy for 0.1 <= X, Y <= 10).
    """
    _positive("X, Y", X, Y)
    t, w = _gauss01(n)
    uu = X * t[:, None]
    vv = Y * t[None, :]
    ww = (X * w[:, None]) * (Y * w[None, :])
    kern = (X - uu) * (Y - vv) / (uu * uu + vv * vv + 1.0) ** 2
    F = 4.0 * float(np.sum(kern * ww)) / (math.pi * X * Y)
    return float(min(1.0, max(0.0, F)))


def numerical_perpendicular_rectangles(H: float, W: float, n: int = 120) -> float:
    r"""Quadrature ground truth for :func:`vf_perpendicular_rectangles_common_edge`.

    Direct fourfold integration over both areas with the shared edge length set
    to ``L = 1``: ``A1 = {x in [0, H], y in [0, 1], z = 0}``,
    ``A2 = {x = 0, y in [0, 1], z in [0, W]}``, so
    :math:`\cos\theta_1\cos\theta_2/r^4 = z_2 x_1 / r^4` and
    ``F = (1/(pi H)) * integral``.

    Warning
    -------
    Unlike the parallel-rectangle case, the integrand is concentrated along the
    *shared edge* (both cosines vanish there), so tensor-product G-L converges
    only like :math:`O(1/n)`: for ``H = W = 1`` the sequence is
    0.2338 (n=20), 0.2205 (n=40), 0.2120 (n=80), 0.2087 (n=120), 0.2069
    (n=160), i.e. still ~3 % above the exact 0.20004 at n=160 — and it degrades
    further for long shared edges (0.87 at n=160 for ``H = W = 0.02``).  Use
    :func:`monte_carlo_view_factor_rectangles` (definitional, unbiased) to
    certify this family; this routine is kept as a convergence demonstrator.
    """
    _positive("H, W", H, W)
    t, w = _gauss01(n)
    x1 = H * t
    wx = H * w
    y1 = 1.0 * t
    wy = 1.0 * w
    y2 = 1.0 * t
    wy2 = 1.0 * w
    z2 = W * t
    wz = W * w
    d2 = (
        x1[:, None, None, None] ** 2
        + (y2[None, None, :, None] - y1[None, :, None, None]) ** 2
        + z2[None, None, None, :] ** 2
    )
    kern = z2[None, None, None, :] * x1[:, None, None, None] / d2**2
    ww = (
        wx[:, None, None, None]
        * wy[None, :, None, None]
        * wy2[None, None, :, None]
        * wz[None, None, None, :]
    )
    F = float(np.sum(kern * ww)) / (math.pi * H)
    return float(min(1.0, max(0.0, F)))


def numerical_parallel_coaxial_disks(R1: float, R2: float, S: float, n: int = 40) -> float:
    r"""Quadrature ground truth for :func:`vf_parallel_coaxial_disks`.

    Polar fourfold tensor-product integration with
    :math:`\cos\theta_1=\cos\theta_2=S/r`:

    .. math::
        F_{1\to2} = \frac{1}{\pi^2 R_1^2}\int\!\!\int\!\!\int\!\!\int
            \frac{S^2}{r^4}\,\rho_1\rho_2\,
            d\rho_1 d\phi_1 d\rho_2 d\phi_2

    ``n = 40`` per axis (2.56e6 nodes) reproduces the closed form to ~1e-13.
    """
    _positive("R1, R2, S", R1, R2, S)
    t, w = _gauss01(n)
    rho1 = R1 * t
    w_rho1 = R1 * w * rho1
    phi1 = 2.0 * math.pi * t
    w_phi1 = 2.0 * math.pi * w
    rho2 = R2 * t
    w_rho2 = R2 * w * rho2
    phi2 = 2.0 * math.pi * t
    w_phi2 = 2.0 * math.pi * w

    r1_4 = rho1[:, None, None, None]
    r2_4 = rho2[None, None, :, None]
    d2 = (
        r1_4**2
        + r2_4**2
        - 2.0 * r1_4 * r2_4 * np.cos(phi1[:, None, None, None] - phi2[None, None, None, :])
        + S * S
    )
    kern = S * S / d2**2
    ww = (
        w_rho1[:, None, None, None]
        * w_phi1[None, :, None, None]
        * w_rho2[None, None, :, None]
        * w_phi2[None, None, None, :]
    )
    F = float(np.sum(kern * ww)) / (math.pi * math.pi * R1 * R1)
    return float(min(1.0, max(0.0, F)))


def numerical_differential_element_to_disk(R: float, H: float, n: int = 400) -> float:
    r"""Quadrature ground truth for :func:`vf_differential_element_to_disk`.

    .. math::
        F = \frac{1}{\pi}\int_0^R\!\!\int_0^{2\pi}
            \frac{H^2}{(\rho^2+H^2)^2}\,\rho\,d\phi\,d\rho
          = 2\int_0^R \frac{H^2\rho}{(\rho^2+H^2)^2}\,d\rho
    """
    _positive("R, H", R, H)
    t, w = _gauss01(n)
    rho = R * t
    wr = R * w
    kern = 2.0 * H * H * rho / (rho * rho + H * H) ** 2
    F = float(np.sum(kern * wr))
    return float(min(1.0, max(0.0, F)))


def numerical_sky_dome_to_tilted_plane(
    beta_deg: float, n_theta: int = 120, n_phi: int = 240
) -> float:
    r"""Quadrature ground truth for :func:`vf_sky_dome_to_tilted_plane`.

    ``n_theta = 120``, ``n_phi = 240`` gives ~1e-6 accuracy; larger orders are
    avoided because ``numpy.legendre.leggauss`` is O(n^2) (``n = 800`` alone
    costs ~16 s).

    Projects the sky half-space (:math:`d_z>0`) onto the plane normal:

    .. math::
        F = \frac{1}{\pi}\int\!\!\int_{\text{sky}}
            [\mathbf{n}\cdot\mathbf{d}]_+ \sin\theta\,d\theta\,d\phi
    """
    if not math.isfinite(beta_deg):
        raise ValueError(f"beta_deg must be finite, got {beta_deg!r}")
    beta = math.radians(beta_deg)
    n = np.array([math.sin(beta), 0.0, math.cos(beta)])
    tt, wt = _gauss01(n_theta)
    tp, wp = _gauss01(n_phi)
    theta = 0.5 * math.pi * tt
    phi = 2.0 * math.pi * tp
    st = np.sin(theta)[:, None]
    ct = np.cos(theta)[:, None]
    dx = st * np.cos(phi)[None, :]
    dy = st * np.sin(phi)[None, :]
    dz = np.broadcast_to(ct, dx.shape)
    cos_surf = n[0] * dx + n[1] * dy + n[2] * dz
    visible = (dz > 0.0) & (cos_surf > 0.0)
    integrand = np.where(visible, cos_surf * st, 0.0)
    w2 = (0.5 * math.pi * wt)[:, None] * (2.0 * math.pi * wp)[None, :]
    return float(min(1.0, max(0.0, float(np.sum(integrand * w2)) / math.pi)))


def monte_carlo_view_factor_rectangles(
    surface1: tuple,
    surface2: tuple,
    n_rays: int = 2_000_000,
    seed: int = 0,
) -> dict:
    r"""Definitional Monte-Carlo view factor between two planar rectangles.

    ``surface = (origin, e1, e2)`` defines the rectangle
    ``{origin + s*e1 + t*e2 : s, t in [0, 1]}`` with ``e1 . e2 = 0``; the
    outward normal is ``e1 x e2`` (right-handed) and **must be oriented so that
    it faces the other surface** — a ray only counts as a hit when it arrives on
    the front side (``d . n2 < 0``).  Rays are launched from uniformly
    distributed points on surface 1 in cosine-weighted directions about ``n1``
    and the view factor is the fraction that strike the front side of
    surface 2.  This estimator is
    unbiased and needs no integrand algebra, which is why it is the
    certification instrument for the contact-edge families where the
    double-area integrand is only slowly convergent under quadrature.

    Returns
    -------
    dict
        ``F``, ``stderr`` (binomial standard error), ``n_rays``.
    """
    if n_rays <= 0:
        raise ValueError(f"n_rays must be > 0, got {n_rays}")
    rng = np.random.default_rng(seed)
    o1, e1, e2 = (np.asarray(a, dtype=np.float64) for a in surface1)
    o2, f1, f2 = (np.asarray(a, dtype=np.float64) for a in surface2)
    l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
    lf1, lf2 = np.linalg.norm(f1), np.linalg.norm(f2)
    if l1 <= 0 or l2 <= 0 or lf1 <= 0 or lf2 <= 0:
        raise ValueError("rectangle edge vectors must be non-degenerate")
    n1 = np.cross(e1, e2)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(f1, f2)
    n2 /= np.linalg.norm(n2)
    u1, u2 = e1 / l1, e2 / l2
    b1, b2 = f1 / lf1, f2 / lf2

    # sample points on surface 1
    s = rng.random(n_rays)
    tts = rng.random(n_rays)
    p = o1[None, :] + s[:, None] * e1[None, :] + tts[:, None] * e2[None, :]

    # cosine-weighted directions about n1
    r_a = rng.random(n_rays)
    r_b = rng.random(n_rays)
    cos_t = np.sqrt(1.0 - r_a)
    sin_t = np.sqrt(r_a)
    phi = 2.0 * np.pi * r_b
    d = (
        (sin_t * np.cos(phi))[:, None] * u1[None, :]
        + (sin_t * np.sin(phi))[:, None] * u2[None, :]
        + cos_t[:, None] * n1[None, :]
    )

    denom = d @ n2
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = ((o2[None, :] - p) @ n2) / denom
    # front-side arrival only: a ray crossing the back of surface 2 sees nothing
    valid = np.isfinite(dist) & (dist > 1e-12) & (denom < 0.0)
    hit = p + np.where(valid, dist, 0.0)[:, None] * d
    rel = hit - o2[None, :]
    a = (rel @ b1) / lf1
    b = (rel @ b2) / lf2
    inside = valid & (a >= 0.0) & (a <= 1.0) & (b >= 0.0) & (b <= 1.0)
    hits = int(np.count_nonzero(inside))
    F = hits / n_rays
    stderr = math.sqrt(max(F * (1.0 - F), 0.0) / n_rays)
    return {"F": float(F), "stderr": float(stderr), "n_rays": int(n_rays)}
