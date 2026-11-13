"""Two-PDF (partial-density) D2Q9 colour-gradient multiphase model.

What this module is
-------------------
The **two-PDF partial-density colour-gradient** model carries *two* complete
distribution functions ``f`` and ``g``, each holding the **partial density**
of one fluid species rather than a full density:

    rho_A = sum_i f_i ,   rho_B = sum_i g_i ,   rho = rho_A + rho_B
    phi   = (rho_A - rho_B) / rho   in [-1, 1]      (colour / order parameter)
    u     = sum_i c_i (f_i + g_i) / rho (+ Guo half-step force shift)

In a pure region ``phi = +-1`` and exactly one partial density carries the
whole mass (``rho = 1``, ``rho_A = 1, rho_B = 0`` for ``phi = +1``).  Mass of
each species is therefore tracked directly by its own PDF -- that is the
"partial density" bookkeeping (as opposed to the classical Rothman-Keller
two-component convention where each colour carries ``rho/2`` everywhere).

Algorithm (one lattice step)
----------------------------
1. macroscopic fields ``rho_A, rho_B, rho, phi, u`` (:func:`two_pdf_macros`);
2. **viscosity from the order parameter**, harmonic interpolation
   ``1/nu(phi) = (1+phi)/(2 nu_A) + (1-phi)/(2 nu_B)``  ->
   ``tau = 3 nu + 1/2`` -> ``omega = 1/tau`` (single relaxation time per node,
   shared by both partial PDFs: the physical viscosity of the *mixture*);
3. BGK collision per species toward ``f_i^eq(rho_A, u)`` / ``g_i^eq(rho_B, u)``
   plus the full three-term Guo et al. (2002) body-force source with
   ``F_sigma = rho_sigma * g`` (exact discrete momentum balance per species);
4. **surface tension** as the anisotropic Latva-Kokko/Rothman colour-gradient
   perturbation on the *total* PDF,
   ``F_i = (A/2) |grad phi| w_i ((c_i . n)^2 - 1/3)``
   (mass-free and momentum-free on the symmetric D2Q9 stencil);
5. **recolouring / segregation**: split the post-collision total PDF back into
   the two partial densities with the MF-LBM weight stencil (main axes
   ``w = 4/9``, diagonals ``w = 1/(36 sqrt 2)``)
   ``f_i = (rho_A/rho) f_i^tot + beta (rho_A rho_B/rho) w_i (c_i . n)``
   ``g_i = (rho_B/rho) f_i^tot - beta (rho_A rho_B/rho) w_i (c_i . n)``
   with the isotropic D2Q9 weights ``w_i`` (``recolor="iso"``, default); the
   literal MF-LBM stencil (``4/9`` axes, ``1/(36 sqrt 2)`` diagonals) is kept
   as ``recolor="mf"`` but is *not* stable at this amplitude scaling.
   which leaves ``rho_A`` and ``rho_B`` bit-exactly unchanged because
   ``sum_i w_i c_i = 0`` on the symmetric stencil.

Lineage / external anchors
--------------------------
* Rothman & Keller (1988) J. Stat. Phys. 52, 1119 -- immiscible lattice gas.
* Latva-Kokko & Rothman (2005) PRE 71, 056702 -- recolouring / segregation
  parameter ``beta`` and the ``(c_i . n)^2`` colour-gradient perturbation.
* Guo, Zheng & Shi (2002) PRE 65, 046308 -- second-order forcing.
* Leclaire, Reggio & Trepanier (2012) J. Comput. Phys. 231, 6294 and
  Leclaire et al. (2013) J. Comput. Phys. 246, 318 -- conservative
  (mass-conserving) recolouring and partial-density equilibria.
* LANL ``MF-LBM`` (Zhang et al., Comput. Fluids 2019) -- reference
  implementation of the two-species partial-density colour gradient
  (``Module.F90``, ``Kernel_multiphase.F90``): total density from the two
  partial PDFs, recolour weights ``4/9`` / ``1/(36 sqrt 2)``, viscosity
  interpolated from ``phi``.

Additive: nothing in ``multiphase`` / ``color_gradient2d`` is modified.
"""

from __future__ import annotations

import torch

from .d2q9 import C

__all__ = [
    "colorgradient_two_pdf_step",
    "two_pdf_macros",
    "viscosity_from_phi",
    "tau_from_phi",
    "dissipation_like",
    "recolor_weights",
    "max_velocity",
    "pressure_jump",
    "relative_l2",
]

_CS2 = 1.0 / 3.0
_REC_MAIN = 4.0 / 9.0           # MF-LBM w_equ_1 (main-axis recolour weight)
_REC_DIAG = 1.0 / (36.0 * 2.0**0.5)  # MF-LBM diagonal recolour weight

_W_D2Q9 = (4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
           1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0)


def _weights_like(f: torch.Tensor) -> torch.Tensor:
    """D2Q9 weights in the working precision of *f* (no float32 round-off)."""
    return torch.tensor(_W_D2Q9, dtype=f.dtype, device=f.device)


def recolor_weights(f: torch.Tensor) -> torch.Tensor:
    """MF-LBM recolouring stencil, shape ``(9,)`` in *f*'s dtype/device.

    ``w = 0`` for the rest direction, ``4/9`` for the four main axes and
    ``1/(36 sqrt 2)`` for the four diagonals.
    """
    c = C.to(f.device).to(f.dtype)
    norm2 = (c * c).sum(dim=1)
    main = torch.full_like(norm2, _REC_MAIN)
    diag = torch.full_like(norm2, _REC_DIAG)
    rest = torch.zeros_like(norm2)
    return torch.where(norm2 < 0.5, rest, torch.where(norm2 < 1.5, main, diag))


def viscosity_from_phi(
    phi: torch.Tensor, nu_a: float, nu_b: float
) -> torch.Tensor:
    """Harmonic viscosity interpolation in the colour field.

    ``1/nu = (1+phi)/(2 nu_a) + (1-phi)/(2 nu_b)``  =>  ``nu(phi=+1) = nu_a``,
    ``nu(phi=-1) = nu_b``, ``nu(phi=0)`` = harmonic mean of the two.

    This is the ``omega = 1/(6/((1+phi)/nu_1 + (1-phi)/nu_2) + 1/2)`` rule of
    the MF-LBM two-species kernel rewritten as a viscosity law.
    """
    if nu_a <= 0.0 or nu_b <= 0.0:
        raise ValueError(f"viscosities must be positive (got nu_a={nu_a}, nu_b={nu_b})")
    inv = (1.0 + phi) / (2.0 * nu_a) + (1.0 - phi) / (2.0 * nu_b)
    return 1.0 / inv.clamp(min=1e-30)


def tau_from_phi(phi: torch.Tensor, nu_a: float, nu_b: float) -> torch.Tensor:
    """Relaxation time ``tau = 3 nu(phi) + 1/2``."""
    return 3.0 * viscosity_from_phi(phi, nu_a, nu_b) + 0.5


def two_pdf_macros(
    f: torch.Tensor, g: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """Return ``(rho_a, rho_b, rho, phi, ux, uy)`` from the two partial PDFs.

    ``ux, uy`` are the *bare* (pre-force-shift) moments ``sum_i c_i (f_i+g_i)/rho``.
    """
    if f.shape != g.shape:
        raise ValueError(f"shape mismatch: f {tuple(f.shape)} vs g {tuple(g.shape)}")
    rho_a = f.sum(dim=0)
    rho_b = g.sum(dim=0)
    rho = rho_a + rho_b
    rho_s = rho.clamp(min=1e-12)
    phi = (rho_a - rho_b) / rho_s
    c = C.to(f.device).to(f.dtype)
    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)
    f_tot = f + g
    ux = (f_tot * cx).sum(dim=0) / rho_s
    uy = (f_tot * cy).sum(dim=0) / rho_s
    return rho_a, rho_b, rho, phi, ux, uy


def _color_normal(
    rho_a: torch.Tensor,
    rho_b: torch.Tensor,
    solid_mask: torch.Tensor | None,
    phi_solid: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Colour field, unit normal and gradient magnitude (central differences).

    Solid cells are masked (or filled with the prescribed wall colour
    ``phi_solid``) so that no spurious colour gradient is generated at walls.
    """
    rho = rho_a + rho_b
    if solid_mask is None:
        ra, rb = rho_a, rho_b
    elif phi_solid is None:
        ra = rho_a.masked_fill(solid_mask, 0.0)
        rb = rho_b.masked_fill(solid_mask, 0.0)
    else:
        if phi_solid.shape != solid_mask.shape:
            raise ValueError(
                f"phi_solid shape {tuple(phi_solid.shape)} != solid_mask {tuple(solid_mask.shape)}"
            )
        half = 0.5 * solid_mask.to(rho_a.dtype) * rho
        ra = torch.where(solid_mask, half * (1.0 + phi_solid.to(rho_a.dtype)), rho_a)
        rb = torch.where(solid_mask, half * (1.0 - phi_solid.to(rho_a.dtype)), rho_b)
    phi = (ra - rb) / (ra + rb).clamp(min=1e-12)
    dphidx = 0.5 * (torch.roll(phi, -1, dims=-1) - torch.roll(phi, 1, dims=-1))
    dphidy = 0.5 * (torch.roll(phi, -1, dims=-2) - torch.roll(phi, 1, dims=-2))
    mag = torch.sqrt(dphidx**2 + dphidy**2)
    mag_safe = mag.clamp(min=1e-12)
    return phi, dphidx / mag_safe, dphidy / mag_safe, mag


def colorgradient_two_pdf_step(
    f: torch.Tensor,
    g: torch.Tensor,
    *,
    nu_a: float = 1.0 / 6.0,
    nu_b: float = 1.0 / 6.0,
    A: float = 0.04,
    beta: float = 0.9,
    gx: float = 0.0,
    gy: float = 0.0,
    solid_mask: torch.Tensor | None = None,
    phi_solid: torch.Tensor | None = None,
    recolor: str = "iso",
) -> tuple[torch.Tensor, torch.Tensor]:
    """One two-PDF partial-density colour-gradient collision step.

    Args:
        f: species-A partial-density PDF, shape ``(9, ny, nx)``.
        g: species-B partial-density PDF, shape ``(9, ny, nx)``.
        nu_a, nu_b: kinematic viscosities of the ``phi=+1`` / ``phi=-1`` phases.
        A: surface-tension (colour-gradient) amplitude.
        beta: segregation (recolouring) parameter, recommended 0.9-0.95.
        gx, gy: body acceleration (per unit mass, identical for both species).
        solid_mask: optional ``(ny, nx)`` bool mask of solid cells; the step
            leaves their populations untouched (bounce-back is applied by the
            caller after streaming).
        phi_solid: optional wall colour field used to mask the colour gradient.
        recolor: ``"iso"`` (default; isotropic D2Q9 weights ``w_i``, the
            Latva-Kokko/Rothman form) or ``"mf"`` (MF-LBM literal stencil:
            main axes ``4/9``, diagonals ``1/(36 sqrt 2)``).  Both are exactly
            mass conserving, but ``"mf"`` at this amplitude scaling drives the
            minority partial density negative in the interface band -- see
            ``xval/XVAL-LBM-TWO-PDF-PARTIAL-DENSITY-COLORGRADIENT/REPORT.md``
            section "recolour stencil selection"; use ``"iso"`` for production.

    Returns:
        ``(f_out, g_out)`` post-collision, pre-streaming partial PDFs.
    """
    if f.shape != g.shape or f.shape[0] != 9:
        raise ValueError(f"expect matching (9, ny, nx) PDFs, got {tuple(f.shape)}, {tuple(g.shape)}")
    if beta < 0.0 or beta > 1.0:
        raise ValueError(f"beta must be in [0, 1] (got {beta})")
    if recolor not in ("mf", "iso"):
        raise ValueError(f"recolor must be 'mf' or 'iso' (got {recolor!r})")

    device = f.device
    c = C.to(device).to(f.dtype)
    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)
    w = _weights_like(f).view(9, 1, 1)

    # --- (1) macros ------------------------------------------------------
    rho_a, rho_b, rho, phi, ux, uy = two_pdf_macros(f, g)
    rho_s = rho.clamp(min=1e-12)

    # --- (2) viscosity from the colour field -----------------------------
    tau = tau_from_phi(phi, nu_a, nu_b)

    # Guo half-step shifted equilibrium velocity (per unit mass force is
    # colour independent, so the shift is common to both species).
    ux_eq = ux + 0.5 * gx
    uy_eq = uy + 0.5 * gy
    cu = cx * ux_eq.unsqueeze(0) + cy * uy_eq.unsqueeze(0)
    u_sq = (ux_eq**2 + uy_eq**2).unsqueeze(0)
    theta = 1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq  # common eq factor

    def _guo(rho_sigma: torch.Tensor) -> torch.Tensor:
        fx = rho_sigma * gx
        fy = rho_sigma * gy
        ci_f = cx * fx.unsqueeze(0) + cy * fy.unsqueeze(0)
        u_f = (ux_eq * fx + uy_eq * fy).unsqueeze(0)
        src = w * (3.0 * ci_f + 9.0 * cu * ci_f - 3.0 * u_f)
        return (1.0 - 0.5 / tau).unsqueeze(0) * src

    # --- (3) BGK collision + Guo forcing, per partial density -------------
    f_post = f - (f - w * rho_a.unsqueeze(0) * theta) / tau.unsqueeze(0) + _guo(rho_a)
    g_post = g - (g - w * rho_b.unsqueeze(0) * theta) / tau.unsqueeze(0) + _guo(rho_b)
    f_tot = f_post + g_post

    # --- (4) surface tension: colour-gradient perturbation on total ------
    _phi, nx_, ny_, mag = _color_normal(rho_a, rho_b, solid_mask, phi_solid)
    ci_n = cx * nx_.unsqueeze(0) + cy * ny_.unsqueeze(0)
    f_tot = f_tot + (A / 2.0) * mag.unsqueeze(0) * w * (ci_n**2 - _CS2)

    # --- (5) recolouring (partial-density segregation) -------------------
    if recolor == "mf":
        w_rec = recolor_weights(f).view(9, 1, 1)
    else:
        w_rec = w
    amp = beta * (rho_a * rho_b / rho_s).unsqueeze(0) * w_rec * ci_n
    wa = (rho_a / rho_s).unsqueeze(0)
    wb = (rho_b / rho_s).unsqueeze(0)
    f_out = wa * f_tot + amp
    g_out = wb * f_tot - amp

    if solid_mask is not None:
        m3 = solid_mask.unsqueeze(0)
        f_out = torch.where(m3, f, f_out)
        g_out = torch.where(m3, g, g_out)
    return f_out, g_out


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def max_velocity(f: torch.Tensor, g: torch.Tensor) -> float:
    """Maximum ``|u|`` of the mixture over all nodes."""
    rho_a, rho_b, rho, _phi, ux, uy = two_pdf_macros(f, g)
    return float(torch.sqrt(ux**2 + uy**2).max().item())


def pressure_jump(
    f: torch.Tensor, g: torch.Tensor, radius: float
) -> tuple[float, float, float]:
    """``(p_in, p_out, dp)`` around the domain centre, ``p = rho/3``.

    Inside band ``r <= 0.5 R``, outside band ``r >= 1.5 R`` -- the convention
    of ``multiphase_benchmarks._measure_pressure_jump``.
    """
    rho = f.sum(dim=0) + g.sum(dim=0)
    ny, nx = rho.shape
    ys = torch.arange(ny, dtype=torch.float64, device=rho.device)
    xs = torch.arange(nx, dtype=torch.float64, device=rho.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    rr = torch.sqrt((xx - nx / 2.0) ** 2 + (yy - ny / 2.0) ** 2)
    inside = rr <= 0.5 * radius
    outside = rr >= 1.5 * radius
    p_in = float((_CS2 * rho[inside].to(torch.float64)).mean().item())
    p_out = float((_CS2 * rho[outside].to(torch.float64)).mean().item())
    return p_in, p_out, p_in - p_out


def relative_l2(num: torch.Tensor, ref: torch.Tensor) -> float:
    """``||num - ref||_2 / ||ref||_2`` over the whole tensor."""
    num64 = num.to(torch.float64)
    ref64 = ref.to(torch.float64)
    return float((num64 - ref64).norm() / ref64.norm())


def dissipation_like(f: torch.Tensor, g: torch.Tensor) -> float:
    """Mean mixture kinetic energy ``<rho u^2> / 2`` (spurious-current proxy)."""
    _ra, _rb, rho, _phi, ux, uy = two_pdf_macros(f, g)
    return float(0.5 * (rho * (ux * ux + uy * uy)).mean().item())