"""Continuum Surface Force (CSF) surface tension for the free-surface framework.

Physics
-------
Brackbill, Kothe & Zemach (1992) replace the capillary pressure jump with an
equivalent body force concentrated on the interface::

    f_s(x) = sigma * kappa(x) * n_hat(x) * delta_s(x)
           = sigma * kappa(x) * grad(phi)(x)                        (1)

because the surface delta function of a smooth indicator ``phi`` is
``delta_s = |grad(phi)|`` and ``n_hat = grad(phi)/|grad(phi)|``, so that
``n_hat * delta_s = grad(phi)``.  The curvature is the *sum of the principal
curvatures*::

    kappa = -div(n_hat)                                            (2)

so that a convex liquid droplet has ``kappa > 0`` and (1) points inward.  In 2-D
``kappa = 1/R`` for a circle, in 3-D ``kappa = 2/R`` for a sphere, hence the
Laplace pressure jump

    Delta p = p_in - p_out = sigma * kappa                          (3)

reduces to ``sigma/R`` (2-D) and ``2 sigma/R`` (3-D).

Discretisation
--------------
Gradients / divergences are isotropic second-order central differences on the
lattice (the same stencil the LBM pressure gradient effectively uses), which is
what makes the *balanced-force* property possible: (1) is evaluated with the
very operator family that transports ``p`` in the momentum equation, so the
force/``grad p`` residual is a truncation error ``O(h^2)`` rather than an
``O(h)`` mismatch.  This is the mechanism behind the ``|u_spurious| ~ h^2``
result measured in :mod:`tensorlbm.free_surface_tension.verification`.

Marangoni (thermo-capillary) drive
----------------------------------
A surface-tension *gradient* adds a tangential stress.  Only the surface
gradient of ``sigma`` acts (a constant ``sigma`` contributes nothing, and the
normal component is the Laplace term already in (1))::

    f_M = grad_s(sigma) * delta_s,   grad_s = (I - n_hat n_hat) . grad(sigma)
                                                                   (4)

which for a flat interface reduces to ``f_x = d(sigma)/dx * delta_s``, i.e. a
tangential stress of magnitude ``d(sigma)/dx`` exerted on the liquid.

Nothing in this module is problem specific: it only needs ``phi``.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

__all__ = [
    "gradient",
    "divergence",
    "interface_normal",
    "interface_delta",
    "curvature",
    "smooth_field",
    "csf_force",
    "marangoni_force",
    "total_force",
    "laplace_pressure",
    "rayleigh_period",
    "rayleigh_period_2fluid",
    "tanh_droplet_phi",
    "tanh_ellipse_phi",
    "tanh_sphere_phi",
    "tanh_flat_phi",
]


# --------------------------------------------------------------------------- #
# Differential operators (periodic, isotropic, second order)                  #
# --------------------------------------------------------------------------- #
def _axis(field: torch.Tensor, comp: int) -> int:
    """Tensor axis of the ``comp``-th physical component (0=x, 1=y, 2=z)."""
    return field.dim() - 1 - comp


def gradient(field: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """``(d/dx, d/dy[, d/dz])`` by second-order central differences (periodic)."""
    out = []
    for comp in range(field.dim()):
        ax = _axis(field, comp)
        out.append(0.5 * (torch.roll(field, -1, ax) - torch.roll(field, 1, ax)))
    return tuple(out)


def divergence(components: Sequence[torch.Tensor]) -> torch.Tensor:
    """``div(v)`` of a vector stored as ``(ndim, *spatial)`` / tuple of fields."""
    div = torch.zeros_like(components[0])
    for comp, v in enumerate(components):
        ax = _axis(v, comp)
        div = div + 0.5 * (torch.roll(v, -1, ax) - torch.roll(v, 1, ax))
    return div


def interface_normal(phi: torch.Tensor, eps: float = 1e-12) -> tuple[torch.Tensor, ...]:
    """Unit normal ``n_hat = grad(phi)/|grad(phi)|`` (components 0=x, 1=y[, 2=z])."""
    g = gradient(phi)
    mag = torch.sqrt(sum(gi * gi for gi in g) + eps * eps)
    return tuple(gi / mag for gi in g)


def interface_delta(phi: torch.Tensor) -> torch.Tensor:
    """Surface delta function ``delta_s = |grad(phi)|`` (Brackbill et al. 1992)."""
    g = gradient(phi)
    return torch.sqrt(sum(gi * gi for gi in g))


def smooth_field(field: torch.Tensor, passes: int = 1) -> torch.Tensor:
    """Binomial ``(1/4, 1/2, 1/4)`` low-pass filter, ``passes`` sweeps per axis.

    Applied once along every axis this is the separable 3x3 binomial kernel,
    i.e. the smallest isotropic-ish smoother on the lattice.  It exists to
    suppress the *lattice-anisotropy* component of the discrete curvature: the
    CSF force needs ``kappa(phi)`` pointwise, and on a square lattice the
    central-difference estimate of ``-div(n_hat)`` carries an ``O(h^2)``
    four-fold (``cos 4 theta``) error whose amplitude can exceed the shape
    perturbation one is trying to resolve.  Measured on a
    ``r(0) = R (1 + eps cos 2 theta)`` ellipse with ``eps = 0.02`` the
    ``cos 2 theta`` amplitude of ``kappa`` is ``0.70 x`` the analytic value
    without the filter and ``0.96 x`` with one pass (``R = 16``, ``w = 2``);
    see ``docs/P284W3_N02_SURFACE_TENSION_V2.md``.

    Args:
        field: Any scalar lattice field.
        passes: Number of sweeps (``0`` returns the input unchanged).

    Returns:
        The filtered field (same shape/dtype/device).
    """
    out = field
    for _ in range(int(passes)):
        for ax in range(out.dim()):
            out = 0.25 * torch.roll(out, 1, ax) + 0.5 * out + 0.25 * torch.roll(out, -1, ax)
    return out


def curvature(phi: torch.Tensor, smooth: int = 0) -> torch.Tensor:
    """Interface curvature ``kappa = -div(n_hat)`` (sum of principal curvatures).

    Positive for a convex *liquid* droplet (``phi = 1`` inside), negative for a
    bubble.  Computed with the same central-difference family as
    :func:`interface_normal`; the accuracy on a tanh-profiled circle/sphere is
    ``O(h^2)`` (see ``tests/test_free_surface_surface_tension.py``).
    """
    if smooth:
        phi = smooth_field(phi, smooth)
    n = interface_normal(phi)
    return -divergence(n)


# --------------------------------------------------------------------------- #
# Forces                                                                      #
# --------------------------------------------------------------------------- #
def csf_force(
    phi: torch.Tensor,
    sigma: float,
    kappa: torch.Tensor | None = None,
    smooth: int = 0,
) -> torch.Tensor:
    """CSF surface-tension force ``f = sigma * kappa * grad(phi)``.

    Args:
        phi: Interface indicator (liquid = 1, gas = 0), shape ``(*spatial)``.
        sigma: Surface-tension coefficient in lattice units.
        kappa: Optional pre-computed curvature (avoids recomputation when the
            caller already needs it, e.g. for reporting).  When given, ``smooth``
            is ignored.
        smooth: Binomial pre-filter passes applied to ``phi`` *before* the
            curvature is evaluated (see :func:`smooth_field`).  The force
            amplitude still uses the unfiltered ``grad(phi)`` so that the
            interface delta function keeps its sharp profile.

    Returns:
        Force density ``(ndim, *spatial)`` (component 0 = x).
    """
    if sigma == 0.0:
        return torch.zeros((phi.dim(), *phi.shape), dtype=phi.dtype, device=phi.device)
    k = curvature(phi, smooth=smooth) if kappa is None else kappa
    g = gradient(phi)
    coef = sigma * k
    return torch.stack([coef * gi for gi in g], dim=0)


def marangoni_force(
    phi: torch.Tensor,
    grad_sigma: Sequence[float],
) -> torch.Tensor:
    """Tangential Marangoni force ``f_M = grad_s(sigma) * |grad(phi)|``.

    Args:
        phi: Interface indicator, shape ``(*spatial)``.
        grad_sigma: Constant surface-tension gradient, ``(d/dx, d/dy[, d/dz])``.

    Returns:
        Force density ``(ndim, *spatial)``.
    """
    n = interface_normal(phi)
    delta = interface_delta(phi)
    gs = torch.tensor(list(grad_sigma), dtype=phi.dtype, device=phi.device)
    # tangential projection: grad_s = grad - n (n . grad)
    n_dot_grad = sum(n[a] * gs[a] for a in range(len(n)))
    out = []
    for a in range(len(n)):
        out.append((gs[a] - n[a] * n_dot_grad) * delta)
    return torch.stack(out, dim=0)


def total_force(
    phi: torch.Tensor,
    sigma: float,
    grad_sigma: Sequence[float] | None = None,
    kappa: torch.Tensor | None = None,
    smooth: int = 0,
) -> torch.Tensor:
    """Laplace (CSF) + optional Marangoni force."""
    f = csf_force(phi, sigma, kappa=kappa, smooth=smooth)
    if grad_sigma is not None and any(v != 0.0 for v in grad_sigma):
        f = f + marangoni_force(phi, grad_sigma)
    return f


# --------------------------------------------------------------------------- #
# Analytic anchors                                                            #
# --------------------------------------------------------------------------- #
def laplace_pressure(sigma: float, radius: float, ndim: int = 2) -> float:
    """Young-Laplace jump ``sigma/R`` (2-D circle) or ``2 sigma/R`` (3-D sphere)."""
    if ndim == 2:
        return sigma / radius
    if ndim == 3:
        return 2.0 * sigma / radius
    raise ValueError("ndim must be 2 or 3")


def rayleigh_period(
    sigma: float,
    radius: float,
    rho: float = 1.0,
    mode: int = 2,
) -> float:
    """Rayleigh (1879) linear oscillation period of an inviscid droplet.

    ``T = 2 pi sqrt( rho R^3 / ( n (n-1)(n+2) sigma ) )`` with ``n`` the
    spherical-harmonic mode (``n = 2`` -> ``n(n-1)(n+2) = 8``).
    """
    n = int(mode)
    denom = n * (n - 1) * (n + 2) * sigma
    return 2.0 * math.pi * math.sqrt(rho * radius**3 / denom)


def rayleigh_period_2fluid(
    sigma: float,
    radius: float,
    rho_inner: float = 1.0,
    rho_outer: float = 1.0,
    mode: int = 2,
    ndim: int = 2,
) -> float:
    """Rayleigh period of a drop surrounded by a fluid of finite density.

    The classic Rayleigh (1879) formula quoted for a "droplet" is the
    **drop-in-vacuum** result; for a drop embedded in a fluid of density
    ``rho_outer`` the linearised normal-stress balance at the interface
    (Lamb, *Hydrodynamics*, art. 275) replaces ``rho`` by ``rho_inner +
    rho_outer``::

        omega^2 = n (n^2 - 1) sigma / ((rho_i + rho_o) a^3)          (2-D)
        omega^2 = n (n-1)(n+2) sigma / ((rho_i + rho_o) a^3)         (3-D)

    This distinction is *not* cosmetic: a single-phase free-surface LBM has
    one uniform density on both sides of the ``phi`` interface (it solves one
    fluid and has no density jump), so its physically matching anchor is the
    ``rho_i = rho_o`` limit, which is ``sqrt(2) = 41 %`` longer in period than
    the vacuum formula.

    Args:
        sigma: Surface-tension coefficient.
        radius: Equilibrium radius ``a``.
        rho_inner: Density inside the drop.
        rho_outer: Density of the surrounding fluid.
        mode: Spherical/cylindrical harmonic mode ``n`` (``2`` = ellipse).
        ndim: ``2`` (circular cylinder) or ``3`` (sphere).

    Returns:
        The linear oscillation period ``2 pi / omega``.
    """
    n = int(mode)
    if ndim == 2:
        factor = n * (n * n - 1)
    elif ndim == 3:
        factor = n * (n - 1) * (n + 2)
    else:
        raise ValueError("ndim must be 2 or 3")
    omega_sq = factor * sigma / ((rho_inner + rho_outer) * radius**3)
    return 2.0 * math.pi / math.sqrt(omega_sq)


# --------------------------------------------------------------------------- #
# Interface indicator constructions (exact hyperbolic-tangent profiles)       #
# --------------------------------------------------------------------------- #
def _coords(shape: Sequence[int], device, dtype) -> tuple[torch.Tensor, ...]:
    """Cell-centre coordinate grids in ``(x, y[, z])`` order."""
    axes = [torch.arange(n, device=device, dtype=dtype) + 0.5 for n in reversed(tuple(shape))]
    grids = torch.meshgrid(*reversed(axes), indexing="ij")
    return tuple(reversed(grids))


def tanh_droplet_phi(
    shape: Sequence[int],
    center: Sequence[float],
    radius: float,
    width: float = 3.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Circular (2-D) / spherical (3-D) droplet indicator ``phi = 0.5(1-tanh(2(r-R)/w))``."""
    coords = _coords(shape, device, dtype)
    r = torch.sqrt(sum((c - c0) ** 2 for c, c0 in zip(coords, center)) + 1e-30)
    return 0.5 * (1.0 - torch.tanh(2.0 * (r - radius) / width))


def tanh_ellipse_phi(
    shape: Sequence[int],
    center: Sequence[float],
    semi_axes: Sequence[float],
    width: float = 3.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Ellipse indicator; ``width`` is the interface thickness of the tanh profile."""
    coords = _coords(shape, device, dtype)
    # normalised radius -> phi = 0.5(1 - tanh(2 (rho-1) R_eff / w))
    rho_n = torch.sqrt(
        sum(((c - c0) / a) ** 2 for c, c0, a in zip(coords, center, semi_axes)) + 1e-30
    )
    r_eff = sum(float(a) for a in semi_axes) / len(semi_axes)
    return 0.5 * (1.0 - torch.tanh(2.0 * (rho_n - 1.0) * r_eff / width))


def tanh_sphere_phi(
    shape: Sequence[int],
    center: Sequence[float],
    radius: float,
    width: float = 3.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Alias of :func:`tanh_droplet_phi` kept for 3-D readability."""
    return tanh_droplet_phi(shape, center, radius, width=width, dtype=dtype, device=device)


def tanh_flat_phi(
    shape: Sequence[int],
    y0: float,
    width: float = 3.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Flat interface normal to ``y``: ``phi = 1`` for ``y < y0`` (liquid below)."""
    coords = _coords(shape, device, dtype)
    y = coords[1]
    return 0.5 * (1.0 - torch.tanh(2.0 * (y - y0) / width))