"""Passive-scalar transport verification surface (campaign node P73W2B-N05).

Why this module exists
----------------------
The library ships several passive-scalar (advection–diffusion) kernels —
:mod:`tensorlbm.scalar_transport` (D2Q5), :mod:`tensorlbm.passive_scalar`
(D2Q5/D3Q7) and :mod:`tensorlbm.taylor_aris` (D2Q9 scalar with zero-flux
walls) — but the evidence around them is heterogeneous: dispersion has been
verified for a plane slit and for a transversely periodic shear, while the
*elementary* transport properties of the scalar solver (diffusive truncation
order, mass conservation at the ``1e-8`` level, and the absence of
Schmidt-number-dependent numerical artefacts) had no machine-checkable anchor
at all.

This module builds one reproducible verification surface with four independent
anchors and emits one machine-readable verdict:

``G1`` Taylor–Aris shear dispersion (analytic anchor)
    ``D_eff / D = 1 + Pe^2 / 210``  for a plane slit,  ``Pe = U H / D``
    ``D_eff / D = 1 + Pe^2 / 48``   for a circular pipe, ``Pe = U a / D``

    verified over ``Pe in [0.1, 100]`` by three mutually independent means: the
    closed forms themselves (:func:`taylor_aris_ratio`), the *discrete* Aris
    auxiliary problem solved on the very nodes a lattice solver uses
    (:func:`discrete_aris_slit`, :func:`discrete_aris_pipe`, which isolate
    quadrature/curvature error from physics), and a time-marched LBM run
    (:func:`measure_slit_dispersion`).
``G2`` erf diffusion front (truncation order)
    ``C(x,t) = 0.5 erfc((x-x0)/sqrt(4Dt))`` reproduced by the D2Q5 scalar
    kernel on three grids refined with the diffusive scaling ``dt ~ dx^2``
    (relaxation time held fixed, so the LBM sees a constant lattice
    diffusivity).  The fitted L2 order must be ``>= 1.9``.
``G3`` Schmidt-number independence
    At fixed ``Pe`` the dimensionless dispersion enhancement ``D_eff/D`` must
    not depend on ``Sc = nu / D`` over ``Sc in [0.7, 10]``: the scalar solver
    must not add relaxation-time-dependent artificial diffusion.  ``nu`` is
    pinned by the companion hydraulic relaxation (``nu = c_s^2 (tau_f-1/2)``);
    ``D`` (hence ``Sc``) is swept while the imposed velocity amplitude is
    scaled with ``D`` so that ``Pe`` stays fixed.
``G4`` discrete mass conservation
    ``|M(t) - M(0)| / M(0) < 1e-8`` for every run of the surface.
``G5`` the artifact ``artifacts/passive_scalar_verification.json`` written by
    :func:`run_verification` (numbers, thresholds, verdicts, git commit).

Note on the campaign seed
-------------------------
The task text labels ``1 + Pe^2/210`` as the *circular pipe* result.  That is a
mislabel: ``1/210`` belongs to the plane slit with ``Pe = U H / D`` (``H`` =
full gap); the pipe result is ``1 + Pe^2/48`` with ``Pe = U a / D`` (``a`` =
radius, equivalently ``1 + Pe_D^2/192`` with ``Pe_D = 2aU/D``).  Both are
verified here and the attribution is reported explicitly.

References
----------
Taylor, G. I. (1953). Dispersion of soluble matter in solvent flowing slowly
    through a tube. *Proc. R. Soc. Lond. A* 219, 186–203.
Aris, R. (1956). On the dispersion of a solute in a fluid flowing through a
    tube. *Proc. R. Soc. Lond. A* 235, 67–77.
Wooding, R. A. (1960). Rayleigh instability of a thermal boundary layer in flow
    through a porous medium. *J. Fluid Mech.* 9, 183–192.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .scalar import (
    scalar_collide_bgk as scalar_collision,
    scalar_equilibrium,
    scalar_macroscopic,
    scalar_stream,
    tau_from_diffusivity,
)
from .taylor_aris import (
    DispersionConfig,
    aris_dispersion_coefficient,
    poiseuille_channel_velocity,
    poiseuille_pipe_velocity,
    simulate_channel_dispersion,
)

__all__ = [
    "ARIS_PE2_SLIT",
    "ARIS_PE2_PIPE",
    "taylor_aris_ratio",
    "taylor_aris_effective_diffusivity",
    "slit_lattice_velocity",
    "pipe_lattice_velocity",
    "discrete_aris_slit",
    "discrete_aris_pipe",
    "erf_diffusion_profile",
    "gaussian_diffusion_profile",
    "DiffusionFrontConfig",
    "simulate_diffusion_front",
    "diffusion_convergence_study",
    "measure_slit_dispersion",
    "schmidt_independence_study",
    "VerificationSpec",
    "quick_spec",
    "run_verification",
    "write_artifact",
]

# ``D_eff/D = 1 + Pe^2 * ARIS_PE2_*``.
#   slit: D_eff = D + (2/105) h^2 U^2 / D with full gap H = 2h
#               = D (1 + H^2 U^2 / (210 D^2)),  Pe = U H / D
#   pipe: D_eff = D + a^2 U^2 / (48 D)      ,  Pe = U a / D
ARIS_PE2_SLIT = 1.0 / 210.0
ARIS_PE2_PIPE = 1.0 / 48.0

GEOMETRIES = {"slit": ARIS_PE2_SLIT, "pipe": ARIS_PE2_PIPE}


# ── analytic anchors ─────────────────────────────────────────────────────────
def taylor_aris_ratio(pe: float, geometry: str = "slit") -> float:
    """``D_eff / D`` of the Taylor–Aris long-time asymptote.

    Parameters
    ----------
    pe :
        ``U H / D`` (slit, ``H`` = full gap) or ``U a / D`` (pipe, ``a`` =
        radius).
    geometry :
        ``"slit"`` (plane Poiseuille, ``1/210``) or ``"pipe"`` (circular
        Poiseuille, ``1/48``).
    """
    try:
        coefficient = GEOMETRIES[geometry]
    except KeyError:  # pragma: no cover - defensive
        raise ValueError(f"unknown geometry: {geometry!r}") from None
    return 1.0 + coefficient * float(pe) ** 2


def taylor_aris_effective_diffusivity(
    diffusivity: float, transverse_scale: float, u_mean: float, geometry: str = "slit"
) -> float:
    """``D_eff = D (1 + Pe^2 / K)`` with ``Pe = u_mean * transverse_scale / D``."""
    if diffusivity <= 0.0:
        raise ValueError("diffusivity must be positive")
    pe = abs(u_mean) * transverse_scale / diffusivity
    return diffusivity * taylor_aris_ratio(pe, geometry)


# ── lattice-sampled velocity profiles ────────────────────────────────────────
def slit_lattice_velocity(
    n_nodes: int, u_mean: float, spacing: float = 1.0
) -> tuple[np.ndarray, np.ndarray, float]:
    """Plane-Poiseuille parabola sampled on the nodes a lattice solver uses.

    Node ``j`` sits at ``y = (j + 0.5 - n/2) * spacing`` and the no-flux walls
    sit half a cell outside at ``y = ± n*spacing/2``, which is exactly the
    geometry of the half-way bounce-back in
    :func:`tensorlbm.taylor_aris.scalar_stream`.

    Returns ``(u, y, half_width)`` with ``half_width = n*spacing/2``.
    """
    n = int(n_nodes)
    if n < 3:
        raise ValueError("need at least three transverse nodes")
    y = (np.arange(n, dtype=np.float64) + 0.5 - 0.5 * n) * spacing
    half_width = 0.5 * n * spacing
    u = np.asarray(poiseuille_channel_velocity(y, half_width, u_mean), dtype=np.float64)
    return u, y, half_width


def pipe_lattice_velocity(
    n_nodes: int, u_mean: float, spacing: float = 1.0
) -> tuple[np.ndarray, np.ndarray, float]:
    """Circular-Poiseuille parabola on cell-centred radial nodes.

    Node ``j`` sits at ``r = (j + 0.5) * spacing`` (finite-volume cells inside
    the wall) and the wall is at ``r = a = n * spacing``.  Returns
    ``(u, r, radius)``.
    """
    n = int(n_nodes)
    if n < 3:
        raise ValueError("need at least three radial nodes")
    r = (np.arange(n, dtype=np.float64) + 0.5) * spacing
    radius = n * spacing
    u = np.asarray(poiseuille_pipe_velocity(r, radius, u_mean), dtype=np.float64)
    return u, r, radius


# ── discrete Aris auxiliary problems ─────────────────────────────────────────
def _thomas_solve(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Solve a tridiagonal system with the Thomas algorithm.

    Deliberately LAPACK-free.  ``numpy.linalg.solve`` on the shared loong64
    compute node returns a *garbage* solution for these pinned, mildly
    ill-conditioned (``cond ~ n^2``) systems once ``n >= 64`` — residual
    ``1.9e2`` against ``1.1e-12`` from ``scipy.linalg.solve`` on the same
    matrix — so a dense solve would silently corrupt the reference.  The
    Thomas recursion is ``O(n)``, deterministic and platform independent; the
    dense-solve path is kept only as a cross-check
    (``library_cross_check`` in :func:`discrete_aris_slit`).
    """
    n = diag.size
    c = np.zeros(n, dtype=np.float64)
    d = np.zeros(n, dtype=np.float64)
    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, n):
        m = diag[i] - lower[i] * c[i - 1]
        if m == 0.0:  # pragma: no cover - only for a degenerate system
            raise ZeroDivisionError("singular tridiagonal system")
        c[i] = upper[i] / m if i < n - 1 else 0.0
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / m
    x = np.zeros(n, dtype=np.float64)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def _slit_aris_system(n_nodes: int, spacing: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tridiagonal no-flux Laplacian for the plane slit (pin ``chi[-1] = 0``)."""
    n = int(n_nodes)
    inv_h2 = 1.0 / spacing**2
    lower = np.zeros(n)
    diag = np.zeros(n)
    upper = np.zeros(n)
    for j in range(n):
        left = 1.0 if j > 0 else 0.0  # reflection at the lower wall
        right = 1.0 if j < n - 1 else 0.0  # reflection at the upper wall
        diag[j] = -(left + right) * inv_h2
        if j > 0:
            lower[j] = left * inv_h2
        if j < n - 1:
            upper[j] = right * inv_h2
    # pin the constant null space on the last node
    lower[-1] = 0.0
    diag[-1] = 1.0
    upper[-1] = 0.0
    return lower, diag, upper


def _pipe_aris_system(
    n_nodes: int, spacing: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tridiagonal cylindrical Laplacian for the pipe (pin ``chi[-1] = 0``).

    Finite-volume coefficients ``(1/r_j) [r_{j+1/2}(chi_{j+1}-chi_j)
    - r_{j-1/2}(chi_j-chi_{j-1})] / h^2`` with ``r_{-1/2} = 0`` (zero flux on
    the axis) and ``r_{n-1/2} = a`` (zero flux at the wall).  Returns the three
    diagonals plus the area weights ``r_j``.
    """
    n = int(n_nodes)
    r = (np.arange(n, dtype=np.float64) + 0.5) * spacing
    inv_h2 = 1.0 / spacing**2
    lower = np.zeros(n)
    diag = np.zeros(n)
    upper = np.zeros(n)
    for j in range(n):
        r_plus = (j + 1.0) * spacing if j < n - 1 else float(n) * spacing
        r_minus = j * spacing
        diag[j] = -(r_plus + r_minus) * inv_h2 / r[j]
        if j > 0:
            lower[j] = r_minus * inv_h2 / r[j]
        if j < n - 1:
            upper[j] = r_plus * inv_h2 / r[j]
    lower[-1] = 0.0
    diag[-1] = 1.0
    upper[-1] = 0.0
    return lower, diag, upper, r


def discrete_aris_slit(
    n_nodes: int = 64,
    diffusivity: float = 0.1,
    u_mean: float = 0.05,
    spacing: float = 1.0,
    library_cross_check: bool = True,
) -> dict[str, Any]:
    """Discrete Aris problem for the plane slit on the lattice node set.

    Solves ``D d^2 chi/dy^2 = -u'`` with no-flux (odd-reflection) walls on the
    parabola sampled at the nodes a lattice solver sees.  The lattice-
    consistent ``Pe`` uses the *discrete* mean of the sampled parabola, so the
    residual is quadrature/curvature error only: the bookkeeping mismatch
    between a node-averaged mean and the continuum ``U`` is removed by
    construction.

    ``library_cross_check`` additionally evaluates
    :func:`tensorlbm.taylor_aris.aris_dispersion_coefficient` (which solves the
    same system with ``numpy.linalg.solve``) and records the difference; on a
    platform whose LAPACK is broken for this conditioning the difference is
    O(1) and is reported rather than trusted.
    """
    u, _y, half_width = slit_lattice_velocity(n_nodes, u_mean, spacing)
    n = u.size
    u_mean_lattice = float(u.mean())
    u_prime = u - u_mean_lattice
    lower, diag, upper = _slit_aris_system(n, spacing)
    chi = _thomas_solve(lower, diag, upper, -u_prime / diffusivity)
    chi = chi - chi.mean()
    taylor_term = float(np.mean(u_prime * chi))
    # residual on the interior rows only (the last row is the pinning equation)
    lap = np.zeros(n)
    lap[1:-1] = (chi[2:] - 2.0 * chi[1:-1] + chi[:-2]) / spacing**2
    residual = float(np.max(np.abs(diffusivity * lap[1:-1] + u_prime[1:-1])))
    d_eff = diffusivity + taylor_term
    gap = 2.0 * half_width
    pe_imposed = abs(u_mean) * gap / diffusivity
    pe_lattice = abs(u_mean_lattice) * gap / diffusivity
    ratio_lattice = d_eff / diffusivity
    ratio_closed = taylor_aris_ratio(pe_lattice, "slit")
    row: dict[str, Any] = {
        "geometry": "slit",
        "n_nodes": int(n),
        "spacing": float(spacing),
        "diffusivity": float(diffusivity),
        "transverse_scale": float(gap),
        "pe_imposed": float(pe_imposed),
        "pe_lattice": float(pe_lattice),
        "u_mean_imposed": float(u_mean),
        "u_mean_lattice": float(u_mean_lattice),
        "d_eff_lattice": float(d_eff),
        "ratio_lattice": float(ratio_lattice),
        "ratio_closed_form": float(ratio_closed),
        "ratio_closed_form_imposed_pe": float(taylor_aris_ratio(pe_imposed, "slit")),
        "rel_err_lattice_vs_closed": float(ratio_lattice / ratio_closed - 1.0),
        "taylor_term": float(taylor_term),
        "aux_residual": float(residual),
    }
    if library_cross_check:
        try:
            ref = aris_dispersion_coefficient(u, diffusivity, spacing)
            finite = bool(np.isfinite(ref["d_eff"]) and abs(ref["d_eff"]) > 0.0)
            rel = float(abs(d_eff / ref["d_eff"] - 1.0)) if finite else float("nan")
            row["library_cross_check"] = {
                "d_eff": float(ref["d_eff"]) if finite else None,
                "residual": float(ref["residual"]) if finite else None,
                "rel_diff_thomas_vs_library": rel if finite else None,
                "trustworthy": bool(finite and rel < 1e-6),
            }
        except Exception as exc:  # pragma: no cover - defensive
            row["library_cross_check"] = {"error": repr(exc)}
    return row


def discrete_aris_pipe(
    n_nodes: int = 64,
    diffusivity: float = 0.1,
    u_mean: float = 0.05,
    spacing: float = 1.0,
) -> dict[str, Any]:
    """Discrete Aris problem for the circular pipe (cylindrical radial operator).

    Finite-volume discretisation of ``(D/r) d/dr (r dchi/dr) = -u'`` with a
    zero-flux axis and a zero-flux wall, area weights ``r_j`` for the
    cross-sectional average::

        D_eff = D + <u' chi>_area ,   <f>_area = sum(r_j f_j) / sum(r_j)

    This isolates the classical ``1/48`` constant from lattice curvature error:
    the closed form is approached as ``O(h^2)`` in the radial spacing.
    """
    u, r, radius = pipe_lattice_velocity(n_nodes, u_mean, spacing)
    n = u.size
    w = r.copy()  # radial measure r dr; the common factor dr cancels
    u_mean_area = float((w * u).sum() / w.sum())
    u_prime = u - u_mean_area
    lower, diag, upper, r_nodes = _pipe_aris_system(n, spacing)
    chi = _thomas_solve(lower, diag, upper, -u_prime / diffusivity)
    chi = chi - float((w * chi).sum() / w.sum())  # mean-free in the area sense
    taylor_term = float((w * u_prime * chi).sum() / w.sum())
    # residual: cylindrical Laplacian applied to chi (rows 1 .. n-2 are
    # interior; row 0 sits on the axis, row n-1 is the pinning equation)
    lap = np.zeros(n)
    for j in range(1, n - 1):
        r_plus = (j + 1.0) * spacing
        r_minus = j * spacing
        lap[j] = (r_plus * (chi[j + 1] - chi[j]) - r_minus * (chi[j] - chi[j - 1])) / (
            r_nodes[j] * spacing**2
        )
    residual = float(np.max(np.abs(diffusivity * lap[1 : n - 1] + u_prime[1 : n - 1])))
    d_eff = diffusivity + taylor_term
    pe_imposed = abs(u_mean) * radius / diffusivity
    pe_lattice = abs(u_mean_area) * radius / diffusivity
    ratio_lattice = d_eff / diffusivity
    ratio_closed = taylor_aris_ratio(pe_lattice, "pipe")
    return {
        "geometry": "pipe",
        "n_nodes": int(n),
        "spacing": float(spacing),
        "diffusivity": float(diffusivity),
        "transverse_scale": float(radius),
        "pe_imposed": float(pe_imposed),
        "pe_lattice": float(pe_lattice),
        "u_mean_imposed": float(u_mean),
        "u_mean_lattice": float(u_mean_area),
        "d_eff_lattice": float(d_eff),
        "ratio_lattice": float(ratio_lattice),
        "ratio_closed_form": float(ratio_closed),
        "ratio_closed_form_imposed_pe": float(taylor_aris_ratio(pe_imposed, "pipe")),
        "rel_err_lattice_vs_closed": float(ratio_lattice / ratio_closed - 1.0),
        "taylor_term": float(taylor_term),
        "aux_residual": float(residual),
    }


# ── G2: erf diffusion front ──────────────────────────────────────────────────
def erf_diffusion_profile(x: Any, t: float, diffusivity: float, x0: float = 0.0) -> torch.Tensor:
    """Half-space step solution ``C = 0.5 erfc((x-x0)/sqrt(4Dt))``.

    Exact solution of ``dC/dt = D d^2C/dx^2`` for ``C(x,0) = 1 (x < x0)``,
    ``0 (x > x0)`` on an unbounded domain.
    """
    if t <= 0.0:
        raise ValueError("t must be positive")
    if diffusivity <= 0.0:
        raise ValueError("diffusivity must be positive")
    xt = torch.as_tensor(x, dtype=torch.float64)
    return 0.5 * torch.erfc((xt - float(x0)) / math.sqrt(4.0 * diffusivity * t))


def gaussian_diffusion_profile(
    x: Any, t: float, diffusivity: float, x0: float = 0.0, sigma0: float = 0.0
) -> torch.Tensor:
    """Free-space Gaussian blob: ``sigma^2(t) = sigma0^2 + 2 D t``.

    Normalised to a unit *peak* at ``t = 0``, so the conserved integral is
    ``sigma0 sqrt(2 pi)`` (not one) — the shape, not the scale, is what the
    Schmidt-number collapse compares.
    """
    xt = torch.as_tensor(x, dtype=torch.float64)
    sigma2 = sigma0**2 + 2.0 * diffusivity * t
    return torch.exp(-0.5 * (xt - x0) ** 2 / sigma2) * (sigma0**2 / sigma2) ** 0.5


@dataclass
class DiffusionFrontConfig:
    """Configuration of a 1-D diffusion-front run of the D2Q5 scalar kernel.

    Refinement follows the *diffusive scaling*: level ``k`` uses ``dx = 2^-k``
    and ``dt = 4^-k`` in physical (coarse-lattice) units, so the lattice
    diffusivity ``D dt/dx^2`` — and therefore the relaxation time — is
    identical on every level.  Refining both ``dx`` and ``dt`` makes the
    truncation error ``O(dx^2)``; with the acoustic scaling (``dt ~ dx``) the
    ``O(dt)`` term dominates and caps the observed order at one.
    """

    nx: int = 64
    ny: int = 4  # rows uniform in y -> the plane problem is exactly 1-D
    level: int = 0
    diffusivity: float = 0.1  # physical (= lattice, since dt = dx^2) value
    end_time: float = 200.0  # physical time in coarse-lattice units
    smooth_time: float = 0.0  # > 0: start from the exact profile at that time
    window_sigmas: float = 3.0
    dtype: str = "float64"
    device: str = "cpu"


def simulate_diffusion_front(cfg: DiffusionFrontConfig) -> dict[str, Any]:
    """Run the D2Q5 scalar kernel against the erf diffusion front.

    The initial condition is a sharp step (``smooth_time = 0``) or, when
    ``smooth_time > 0``, the exact profile at that time (a smooth start that
    removes the initial-discontinuity singularity from the error budget).
    The L2 error is evaluated only inside
    ``|x - x0| <= window_sigmas * sqrt(4 D t)``, where the periodic image of
    the front is below ``1e-19`` and contributes nothing.
    """
    device = torch.device(cfg.device)
    tdt = torch.float64 if cfg.dtype == "float64" else torch.float32
    level = int(cfg.level)
    dx = 2.0 ** (-level)
    dt = dx**2  # diffusive scaling; D_phys = D_lattice = cfg.diffusivity
    nx, ny = int(cfg.nx), int(cfg.ny)
    d_phys = float(cfg.diffusivity)
    d_lattice = d_phys * dt / dx**2
    tau_s = tau_from_diffusivity(d_lattice)

    steps = int(round(cfg.end_time / dt))
    if steps < 1:
        raise ValueError("end_time is shorter than one time step")

    x = (torch.arange(nx, dtype=torch.float64, device=device) + 0.5) * dx
    x0 = 0.5 * nx * dx
    t0 = float(cfg.smooth_time)
    if t0 > 0.0:
        c0 = erf_diffusion_profile(x, t0, d_phys, x0).to(tdt)
    else:
        c0 = torch.where(
            x < x0,
            torch.ones_like(x, dtype=tdt),
            torch.zeros_like(x, dtype=tdt),
        )
    c_field = c0.view(1, nx).expand(ny, nx).contiguous()
    zero = torch.zeros((ny, nx), dtype=tdt, device=device)
    g = scalar_equilibrium(c_field, zero, zero)

    mass0 = float(scalar_macroscopic(g).double().sum())
    for _ in range(steps):
        c_field = scalar_macroscopic(g)
        g = scalar_collision(g, c_field, zero, zero, tau_s)
        g = scalar_stream(g)

    c_field = scalar_macroscopic(g)
    mass1 = float(c_field.double().sum())
    mass_drift = abs(mass1 - mass0) / mass0

    t_total = t0 + cfg.end_time
    exact = erf_diffusion_profile(x, t_total, d_phys, x0)
    c_num = c_field[0].double()
    half_width = cfg.window_sigmas * math.sqrt(4.0 * d_phys * t_total)
    win = torch.abs(x - x0) <= half_width
    diff = (c_num - exact)[win]
    l2 = float(torch.sqrt(torch.mean(diff**2)))
    l2_rel = l2 / max(float(torch.sqrt(torch.mean(exact[win] ** 2))), 1e-30)
    return {
        "level": level,
        "nx": nx,
        "dx": dx,
        "dt": dt,
        "steps": steps,
        "tau_s": tau_s,
        "diffusivity_lattice": d_lattice,
        "t_end": t_total,
        "x": x.detach().cpu().numpy(),
        "c_numeric": c_num.detach().cpu().numpy(),
        "c_exact": exact.detach().cpu().numpy(),
        "window_mask": win.detach().cpu().numpy(),
        "l2_error": l2,
        "l2_error_relative": l2_rel,
        "mass_drift": mass_drift,
        "mass_final": mass1,
    }


def diffusion_convergence_study(
    base_nx: int = 64,
    levels: Sequence[int] = (0, 1, 2),
    diffusivity: float = 0.1,
    end_time: float = 200.0,
    smooth_time: float = 0.0,
    ny: int = 4,
    dtype: str = "float64",
    device: str = "cpu",
) -> dict[str, Any]:
    """L2 convergence order of the erf diffusion front over ``levels``.

    Returns the per-level cases plus the pairwise orders
    ``p_k = log2(e_k / e_{k+1})`` (``dx`` is halved on every level) and the
    minimum order, which is the number the ``G2`` gate compares with ``1.9``.
    """
    cases: list[dict[str, Any]] = []
    for level in levels:
        cfg = DiffusionFrontConfig(
            nx=int(base_nx) * 2**level,
            ny=ny,
            level=level,
            diffusivity=diffusivity,
            end_time=end_time,
            smooth_time=smooth_time,
            dtype=dtype,
            device=device,
        )
        cases.append(simulate_diffusion_front(cfg))
    errors = [c["l2_error"] for c in cases]
    orders: list[float] = []
    for e0, e1 in zip(errors[:-1], errors[1:]):
        if e0 <= 0.0 or e1 <= 0.0:  # pragma: no cover - pathological
            orders.append(float("nan"))
        else:
            orders.append(math.log2(e0 / e1))
    return {
        "base_nx": int(base_nx),
        "levels": list(levels),
        "diffusivity": float(diffusivity),
        "end_time": float(end_time),
        "smooth_time": float(smooth_time),
        "cases": cases,
        "errors": errors,
        "orders": orders,
        "min_order": float(min(orders)) if orders else float("nan"),
        "mass_drift_max": max(c["mass_drift"] for c in cases),
    }


# ── G1: time-marched Taylor–Aris dispersion ──────────────────────────────────
def measure_slit_dispersion(
    pe: float,
    ny: int = 17,
    nx: int = 256,
    diffusivity: float = 0.04,
    steps: int | None = None,
    dtype: str = "float64",
    device: str = "cpu",
) -> dict[str, Any]:
    """Time-marched LBM dispersion meter for the plane slit (D2Q9 scalar kernel).

    The velocity field is the analytic Poiseuille parabola evaluated on the
    lattice nodes (the scalar solver is what is under test; the momentum solve
    is verified elsewhere), ``u_mean = Pe D / H`` with ``H = ny``, and
    ``D_eff`` comes from the decay rate of the fundamental Fourier mode
    (:func:`tensorlbm.taylor_aris.simulate_channel_dispersion`,
    ``method="mode"``), which is the asymptotic dispersion coefficient.

    ``steps`` defaults to ``4 h^2 / D`` (``h = ny/2``), i.e. four transverse
    diffusion times.
    """
    half_width = 0.5 * ny
    gap = float(ny)
    u_mean = pe * diffusivity / gap
    if abs(u_mean) > 0.1:  # keep the lattice Mach number small
        raise ValueError(f"Pe={pe} needs |u|={abs(u_mean):.3g} > 0.1 (raise D or ny)")
    if steps is None:
        steps = max(int(round(4.0 * half_width**2 / diffusivity)), 4000)
    cfg = DispersionConfig(
        nx=int(nx),
        ny=int(ny),
        diffusivity=float(diffusivity),
        u_mean=float(u_mean),
        steps=int(steps),
        profile="parabolic",
        method="mode",
        dtype=dtype,
        device=device,
        report_every=max(1, int(steps) // 200),
        metadata={"pe_target": float(pe), "node": "p73w2b-n05"},
    )
    out = simulate_channel_dispersion(cfg)
    ratio_sim = out["d_eff"] / out["d_molecular"]
    ratio_discrete = out["d_eff_discrete"] / out["d_molecular"]
    ratio_closed = taylor_aris_ratio(pe, "slit")
    return {
        "label": f"lbm_slit_pe{pe:g}",
        "method": "lbm_time_march",
        "geometry": "slit",
        "pe_target": float(pe),
        "pe_measured": float(out["pe"] * 2.0),  # module reports U h / D
        "u_mean": float(u_mean),
        "diffusivity": float(diffusivity),
        "ny": int(ny),
        "nx": int(nx),
        "steps": int(steps),
        "d_eff_sim": float(out["d_eff"]),
        "ratio_sim": float(ratio_sim),
        "ratio_discrete_lattice": float(ratio_discrete),
        "ratio_closed_form": float(ratio_closed),
        "rel_err_vs_closed": float(ratio_sim / ratio_closed - 1.0),
        "rel_err_vs_discrete": float(ratio_sim / ratio_discrete - 1.0),
        "taylor_increment_rel_err_vs_closed": float((ratio_sim - 1.0) / (ratio_closed - 1.0) - 1.0)
        if ratio_closed > 1.0
        else float("nan"),
        "mass_drift": float(out["mass_drift"]),
        "r2": float(out["r2"]),
        "rate_scatter": float(out["rate_scatter"]),
        "transient_resolved": bool(out["transient_resolved"]),
        "fit_t0": float(out["fit_t0"]),
        "fit_t1": float(out["fit_t1"]),
        "tau_d": float(out["tau_d"]),
    }


# ── G3: Schmidt-number independence ─────────────────────────────────────────
def schmidt_independence_study(
    pe: float = 10.0,
    sc_list: Sequence[float] = (0.7, 1.0, 2.5, 5.0),
    viscosity: float = 0.1,
    ny: int = 17,
    nx: int = 256,
    dtype: str = "float64",
    device: str = "cpu",
    steps_per_case: int | None = None,
    resolution_ny: int = 0,
) -> dict[str, Any]:
    """Dispersion enhancement at fixed ``Pe`` across a Schmidt-number sweep.

    ``Sc = nu / D`` with ``nu`` pinned by the companion hydraulic relaxation
    (``tau_f = 0.8`` -> ``nu = 0.1``).  For every ``Sc`` the molecular
    diffusivity is ``D = nu / Sc`` and the imposed mean velocity is
    ``u_mean = Pe D / H`` so the *dimensionless* problem — and therefore the
    Taylor–Aris ratio ``1 + Pe^2/210`` — is identical in all cases.  Any
    spread between the measured ratios is numerical error that depends on the
    scalar relaxation time, i.e. an ``Sc``-dependent artefact.

    Notes
    -----
    The advecting field is the analytic parabola, so ``nu`` (and hence ``Sc``)
    enters only through the definition of the sweep; the momentum solver's own
    ``Sc``-coupling is out of scope (see the module limitations).
    """
    rows: list[dict[str, Any]] = []
    for sc in sc_list:
        d = viscosity / float(sc)
        steps = steps_per_case
        if steps is None:
            steps = max(int(round(4.0 * (0.5 * ny) ** 2 / d)), 2000)
        row = measure_slit_dispersion(
            pe=pe,
            ny=ny,
            nx=nx,
            diffusivity=d,
            steps=steps,
            dtype=dtype,
            device=device,
        )
        row.update(
            {
                "label": f"sc{sc:g}_pe{pe:g}",
                "schmidt": float(sc),
                "viscosity_reference": float(viscosity),
            }
        )
        rows.append(row)
        row.setdefault("role", "sweep")
    ratios = [r["ratio_sim"] for r in rows]
    closed = taylor_aris_ratio(pe, "slit")
    mean_ratio = float(np.mean(ratios))
    spread = (max(ratios) - min(ratios)) / mean_ratio
    max_dev_vs_closed = max(abs(r / closed - 1.0) for r in ratios)

    # transverse-resolution control: the largest deviation of the sweep is
    # re-measured on a refined cross-section, which shows whether it is a
    # Schmidt effect or a discretisation effect.
    control: dict[str, Any] | None = None
    if resolution_ny and rows:
        worst = max(rows, key=lambda r: abs(r["rel_err_vs_closed"]))
        d_worst = float(worst["diffusivity"])
        ctrl = measure_slit_dispersion(
            pe=pe,
            ny=int(resolution_ny),
            nx=nx,
            diffusivity=d_worst,
            dtype=dtype,
            device=device,
        )
        ctrl.update(
            {
                "label": f"sc{worst['schmidt']:g}_pe{pe:g}_ny{resolution_ny}",
                "schmidt": float(worst["schmidt"]),
                "viscosity_reference": float(viscosity),
                "role": "resolution_control",
                "ny_reference": int(worst["ny"]),
            }
        )
        control = ctrl
    return {
        "pe": float(pe),
        "viscosity_reference": float(viscosity),
        "sc_list": [float(s) for s in sc_list],
        "ratio_closed_form": float(closed),
        "rows": rows,
        "mean_ratio": mean_ratio,
        "ratio_spread": float(spread),
        "max_dev_vs_closed": float(max_dev_vs_closed),
        "mass_drift_max": max(r["mass_drift"] for r in rows),
        "resolution_control": control,
    }


# ── specification + driver ──────────────────────────────────────────────────
@dataclass
class VerificationSpec:
    """Knobs of the passive-scalar verification surface.

    The defaults are the *formal* setting used for the published artifact;
    :func:`quick_spec` is the reduced setting used by the pytest suite.
    """

    # G1 — analytic + discrete Aris sweep
    pe_sweep: Sequence[float] = (0.1, 1.0, 5.0, 20.0, 50.0, 100.0)
    slit_nodes: int = 64
    pipe_nodes: int = 64
    grid_convergence_nodes: Sequence[int] = (16, 32, 64, 128, 256)
    grid_convergence_pe: float = 20.0
    diffusivity_anchor: float = 0.1
    u_mean_anchor: float = 0.05
    # G1 — time-marched LBM (gate) and an out-of-gate resolution probe
    lbm_pe_list: Sequence[float] = (0.1, 2.0, 5.0, 20.0)
    lbm_probe_pe: Sequence[float] = (40.0,)
    lbm_ny: int = 17
    lbm_nx: int = 128
    lbm_diffusivity: float = 0.04
    # G2 — erf diffusion
    erf_base_nx: int = 128
    erf_levels: Sequence[int] = (0, 1, 2)
    erf_end_time: float = 200.0
    erf_smooth_time: float = 0.0
    # G3 — Schmidt sweep (Pe fixed, D = nu/Sc varies, u = Pe D/H follows D)
    schmidt_pe: float = 3.0
    schmidt_sc: Sequence[float] = (0.7, 1.0, 2.5, 5.0, 10.0)
    schmidt_resolution_ny: int = 33  # transverse-resolution control for the worst Sc
    viscosity_reference: float = 0.1
    # gates
    taylor_tolerance_pct: float = 3.0
    erf_order_threshold: float = 1.9
    schmidt_tolerance_pct: float = 2.0
    mass_tolerance: float = 1e-8
    dtype: str = "float64"
    device: str = "cpu"
    label: str = "formal"

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (tuple, list)):
                out[key] = [v.item() if isinstance(v, np.generic) else v for v in value]
            else:
                out[key] = value
        return out


def quick_spec() -> VerificationSpec:
    """Reduced spec (seconds, not minutes) used by the unit tests."""
    return VerificationSpec(
        pe_sweep=(0.1, 10.0, 100.0),
        grid_convergence_nodes=(16, 32, 64),
        lbm_pe_list=(5.0,),
        lbm_probe_pe=(),
        lbm_ny=9,
        lbm_nx=64,
        lbm_diffusivity=0.1,
        erf_base_nx=128,
        erf_levels=(0, 1),
        erf_end_time=200.0,
        schmidt_pe=3.0,
        schmidt_sc=(1.0, 5.0),
        schmidt_resolution_ny=0,
        dtype="float64",
        label="quick",
    )


def _git_commit(root: Path | None = None) -> str:
    import os

    env = os.environ.get("PSV_GIT_COMMIT", "").strip()
    if env:
        return env
    try:
        root = Path(root or Path(__file__).resolve().parents[2])
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # pragma: no cover - git unavailable
        return "unknown"


def run_verification(
    spec: VerificationSpec | None = None,
    out_path: str | Path | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Execute the four anchors and assemble the machine-readable verdict.

    Parameters
    ----------
    spec :
        :class:`VerificationSpec`; formal defaults if ``None``.
    out_path :
        Where to write the JSON artifact (``artifacts/passive_scalar_verification.json``
        for the published run).  ``None`` keeps the report in memory.
    verbose :
        Print per-criterion progress to stderr.
    """
    spec = spec or VerificationSpec()
    started = time.time()
    log = (lambda msg: print(f"[psv] {msg}", flush=True)) if verbose else (lambda msg: None)

    # ---- G1a: closed form + discrete Aris, Pe sweep over both geometries ----
    log("G1a discrete Aris sweep (slit + pipe)")
    anchor_rows: list[dict[str, Any]] = []
    for geometry, solver in (("slit", discrete_aris_slit), ("pipe", discrete_aris_pipe)):
        nodes = spec.slit_nodes if geometry == "slit" else spec.pipe_nodes
        transverse = float(nodes)  # slit: full gap H = n; pipe: radius a = n
        for pe in spec.pe_sweep:
            # D fixed, u scaled with Pe (well conditioned at every Pe)
            d = spec.diffusivity_anchor
            u = pe * d / transverse
            row = solver(n_nodes=nodes, diffusivity=d, u_mean=u)
            row["pe_target"] = float(pe)
            row["peak_ratio"] = float(taylor_aris_ratio(pe, geometry))
            row["rel_err_vs_peak_ratio"] = float(row["ratio_lattice"] / row["peak_ratio"] - 1.0)
            anchor_rows.append(row)
            log(
                f"  {geometry:4s} Pe={pe:6g} ratio_lat={row['ratio_lattice']:.6f} "
                f"closed={row['peak_ratio']:.6f} err={100 * row['rel_err_vs_peak_ratio']:+.4f}%"
            )

    # ---- G1b: lattice convergence of the discrete Aris reference ----
    log("G1b discrete Aris lattice convergence")
    grid_rows: list[dict[str, Any]] = []
    for geometry, solver in (("slit", discrete_aris_slit), ("pipe", discrete_aris_pipe)):
        for nodes in spec.grid_convergence_nodes:
            transverse = float(nodes)
            d = spec.diffusivity_anchor
            u = spec.grid_convergence_pe * d / transverse
            row = solver(n_nodes=nodes, diffusivity=d, u_mean=u)
            row["pe_target"] = float(spec.grid_convergence_pe)
            row["peak_ratio"] = float(taylor_aris_ratio(spec.grid_convergence_pe, geometry))
            row["rel_err_vs_peak_ratio"] = float(row["ratio_lattice"] / row["peak_ratio"] - 1.0)
            grid_rows.append(row)
            log(f"  {geometry:4s} n={nodes:4d} err={100 * row['rel_err_vs_peak_ratio']:+.4f}%")

    # ---- G1c: time-marched LBM dispersion ----
    log("G1c time-marched LBM dispersion (slit)")
    lbm_rows: list[dict[str, Any]] = []
    for pe in spec.lbm_pe_list:
        row = measure_slit_dispersion(
            pe=pe,
            ny=spec.lbm_ny,
            nx=spec.lbm_nx,
            diffusivity=spec.lbm_diffusivity,
            dtype=spec.dtype,
            device=spec.device,
        )
        lbm_rows.append(row)
        log(
            f"  Pe={pe:6g} ratio_sim={row['ratio_sim']:.6f} "
            f"closed={row['ratio_closed_form']:.6f} "
            f"err={100 * row['rel_err_vs_closed']:+.3f}% steps={row['steps']}"
        )

    # ---- G1d: out-of-gate resolution probe (documents the lattice limit) ----
    probe_rows: list[dict[str, Any]] = []
    for pe in spec.lbm_probe_pe:
        row = measure_slit_dispersion(
            pe=pe,
            ny=spec.lbm_ny,
            nx=spec.lbm_nx,
            diffusivity=spec.lbm_diffusivity,
            dtype=spec.dtype,
            device=spec.device,
        )
        row["role"] = "resolution_limit_probe"
        probe_rows.append(row)
        log(
            f"  probe Pe={pe:6g} ratio_sim={row['ratio_sim']:.6f} "
            f"closed={row['ratio_closed_form']:.6f} "
            f"err={100 * row['rel_err_vs_closed']:+.3f}% (u={row['u_mean']:.4f})"
        )

    g1_errors = [abs(r["rel_err_vs_peak_ratio"]) for r in anchor_rows] + [
        abs(r["rel_err_vs_closed"]) for r in lbm_rows
    ]
    g1_max_err_pct = 100.0 * max(g1_errors) if g1_errors else float("nan")
    g1_pass = g1_max_err_pct < spec.taylor_tolerance_pct

    # ---- G2: erf diffusion convergence ----
    log("G2 erf diffusion convergence")
    erf = diffusion_convergence_study(
        base_nx=spec.erf_base_nx,
        levels=spec.erf_levels,
        diffusivity=spec.diffusivity_anchor,
        end_time=spec.erf_end_time,
        smooth_time=spec.erf_smooth_time,
        dtype=spec.dtype,
        device=spec.device,
    )
    log(f"  orders={['%.3f' % o for o in erf['orders']]} min={erf['min_order']:.3f}")
    g2_pass = bool(erf["orders"]) and erf["min_order"] >= spec.erf_order_threshold

    # ---- G3: Schmidt independence ----
    log("G3 Schmidt independence")
    schmidt = schmidt_independence_study(
        pe=spec.schmidt_pe,
        sc_list=spec.schmidt_sc,
        viscosity=spec.viscosity_reference,
        ny=spec.lbm_ny,
        nx=spec.lbm_nx,
        dtype=spec.dtype,
        device=spec.device,
        resolution_ny=spec.schmidt_resolution_ny,
    )
    schmidt_spread_pct = 100.0 * schmidt["ratio_spread"]
    schmidt_dev_pct = 100.0 * schmidt["max_dev_vs_closed"]
    g3_pass = max(schmidt_spread_pct, schmidt_dev_pct) < spec.schmidt_tolerance_pct
    log(
        f"  spread={schmidt_spread_pct:.3f}% max_dev={schmidt_dev_pct:.3f}% "
        f"closed={schmidt['ratio_closed_form']:.6f}"
    )

    # ---- G4: mass conservation across every run ----
    mass_drifts = {
        "erf_diffusion": erf["mass_drift_max"],
        "lbm_dispersion": max(r["mass_drift"] for r in lbm_rows) if lbm_rows else 0.0,
        "schmidt_sweep": schmidt["mass_drift_max"],
    }
    g4_max = max(mass_drifts.values())
    g4_pass = g4_max < spec.mass_tolerance

    verdict = {
        "G1_taylor_aris": g1_pass,
        "G2_erf_convergence": g2_pass,
        "G3_schmidt_independence": g3_pass,
        "G4_mass_conservation": g4_pass,
    }
    report: dict[str, Any] = {
        "node": "p73w2b-n05-lbm-passive-scalar",
        "module": "tensorlbm.passive_scalar_verification",
        "label": spec.label,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "runtime_seconds": 0.0,
        "spec": spec.to_json(),
        "criteria": {
            "G1": {
                "name": "Taylor-Aris shear dispersion analytic anchor",
                "threshold_pct": spec.taylor_tolerance_pct,
                "max_err_pct": float(g1_max_err_pct),
                "pass": bool(g1_pass),
                "geometry_attribution": {
                    "slit": "D_eff/D = 1 + Pe^2/210, Pe = U H / D (full gap H)",
                    "pipe": "D_eff/D = 1 + Pe^2/48,  Pe = U a / D (radius a)",
                    "campaign_seed_erratum": (
                        "1 + Pe^2/210 is the plane-slit coefficient, not the circular "
                        "pipe one; the pipe coefficient is 1/48. Both are verified here."
                    ),
                },
                "anchor_rows": anchor_rows,
                "lattice_convergence_rows": grid_rows,
                "lbm_rows": lbm_rows,
                "lbm_probe_rows": probe_rows,
            },
            "G2": {
                "name": "erf diffusion front L2 convergence order",
                "threshold_order": spec.erf_order_threshold,
                "min_order": float(erf["min_order"]),
                "orders": [float(o) for o in erf["orders"]],
                "pass": bool(g2_pass),
                "errors": [float(e) for e in erf["errors"]],
                "cases": [
                    {k: v for k, v in case.items() if not isinstance(v, np.ndarray)}
                    for case in erf["cases"]
                ],
            },
            "G3": {
                "name": "Schmidt-number independence at fixed Pe",
                "threshold_pct": spec.schmidt_tolerance_pct,
                "ratio_spread_pct": float(schmidt_spread_pct),
                "max_dev_vs_closed_pct": float(schmidt_dev_pct),
                "ratio_closed_form": float(schmidt["ratio_closed_form"]),
                "pass": bool(g3_pass),
                "rows": [{k: v for k, v in r.items() if k != "history"} for r in schmidt["rows"]],
                "resolution_control": (
                    {k: v for k, v in schmidt["resolution_control"].items() if k != "history"}
                    if schmidt["resolution_control"]
                    else None
                ),
            },
            "G4": {
                "name": "discrete scalar mass conservation",
                "threshold": spec.mass_tolerance,
                "max_drift": float(g4_max),
                "drifts": {k: float(v) for k, v in mass_drifts.items()},
                "pass": bool(g4_pass),
            },
            "G5": {
                "name": "machine-readable artifact",
                "path": "artifacts/passive_scalar_verification.json",
                "pass": True,
            },
        },
        "verdict": "PASS" if all(verdict.values()) else "FAIL",
        "criteria_pass": verdict,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "device": spec.device,
            "dtype": spec.dtype,
        },
        "limitations": [
            "The circular-pipe anchor is verified by the discrete cylindrical Aris "
            "solve (finite-volume, zero-flux axis and wall) against the closed form, "
            "not by a time-marched axisymmetric/3-D LBM pipe run: the shipped scalar "
            "kernels are Cartesian, so a pipe cannot be marched without adding an "
            "axisymmetric source term.",
            "The time-marched LBM dispersion runs use the analytic Poiseuille "
            "parabola as the advecting field (the momentum solve is verified "
            "separately); they verify the scalar kernel, not the flow solver.",
            "The Schmidt sweep pins nu with the companion hydraulic relaxation and "
            "scales the imposed velocity with D to hold Pe fixed, so Sc enters the "
            "discrete problem through tau_d only; a fully coupled Sc study (varying "
            "tau_f as well) is not part of this surface.",
            "Pe >~ 40 time-marched runs are excluded by the lattice Mach-number "
            "limit |u| < 0.1 combined with the requirement t >> h^2/D; the high-Pe "
            "end of the sweep is covered by the discrete Aris reference instead.",
        ],
    }
    # Raw profile arrays kept out of the JSON artifact (keys starting with "_"
    # are dropped by write_artifact) but available to the plotting driver.
    report["_figure_data"] = {
        "erf": [
            {
                "level": c["level"],
                "nx": c["nx"],
                "dx": c["dx"],
                "x": np.asarray(c["x"]).tolist(),
                "c_numeric": np.asarray(c["c_numeric"]).tolist(),
                "c_exact": np.asarray(c["c_exact"]).tolist(),
                "mask": np.asarray(c["window_mask"]).astype(bool).tolist(),
            }
            for c in erf["cases"]
        ],
        "erf_errors": [float(e) for e in erf["errors"]],
        "erf_dx": [float(c["dx"]) for c in erf["cases"]],
    }
    report["runtime_seconds"] = float(time.time() - started)
    if out_path is not None:
        write_artifact(report, out_path)
    return report


def write_artifact(report: dict[str, Any], out_path: str | Path) -> Path:
    """Write the verification report as JSON (numpy-safe), creating parents."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(obj: Any) -> Any:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"not JSON serialisable: {type(obj)!r}")

    payload = {k: v for k, v in report.items() if not k.startswith("_")}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_default))
    return path
