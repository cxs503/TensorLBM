"""Two-relaxation-time D2Q9 Color-Gradient multiphase model.

Lineage
-------
Rothman & Keller (1988) immiscible lattice gas; Latva-Kokko & Rothman (2005)
lattice-Boltzmann color gradient with segregation (recolor) parameter beta.
The surface-tension perturbation and recolor forms follow the library's
existing single-tau ``multiphase.color_gradient_step`` (anisotropic
A-perturbation, simplified Latva-Kokko recolor), which remains untouched and
bit-identical in default behaviour.  This module is purely additive.

What this module adds over ``multiphase.color_gradient_step``:

1. **Per-component relaxation times** ``tau_r`` / ``tau_b`` -> kinematic
   viscosity contrast ``nu_sigma = c_s^2 (tau_sigma - 1/2)`` between the
   phases.  The stock step shares a single ``tau`` (viscosity ratio pinned
   to 1; documented limitation in ``multiphase_benchmarks``).
2. **Full three-term Guo et al. (2002) body force** per component, with the
   physical (half-step) velocity shift ``u_eq = u + g/2`` and equal
   per-mass acceleration ``g`` for both colors.  Exact discrete momentum
   balance per color: ``sum_i c_i Delta f_i^sigma = rho_sigma g`` per step.
   The stock step shifts the total-distribution velocity by ``tau g``
   (first-order).
3. **Diagnostics helpers** sharing the measurement conventions of
   ``multiphase_benchmarks`` (lattice pressure ``p = rho/3``; inside/outside
   bands ``0.5 R`` / ``1.5 R``).

Structural conservation properties (unit-tested in
``tests/test_color_gradient2d.py``):

* **Per-color mass**: collision (``sum_i f_eq^sigma = rho_sigma``),
  perturbation (``sum_i w_i [(c_i . n)^2 - 1/3] = 0``), recolor
  (``sum_i w_i (c_i . n) = 0``) and the Guo source (``sum_i S_i = 0``) are
  all mass-free -> per-color mass is conserved to round-off.
* **Total momentum**: both colors relax towards the *same* mixture velocity
  (``f_eq`` is linear in ``rho`` at fixed ``u``, so the two-tau collision
  with ``tau_r = tau_b`` reduces algebraically to the stock total-distribution
  collision); the perturbation is third-moment-free on the symmetric D2Q9
  lattice; the recolor amplitude enters with opposite signs for the two
  colors -> total momentum changes only through the body force.
"""

from __future__ import annotations

import torch

from .d2q9 import C

__all__ = [
    "color_gradient_two_tau_step",
    "cg_phase_field",
    "cg_phase_gradient",
    "cg_pressure_jump",
    "cg_max_velocity",
    "cg_interface_width_cut",
]

_CS2 = 1.0 / 3.0


def _weights_like(f: torch.Tensor) -> torch.Tensor:
    """D2Q9 weights rebuilt in the working precision of *f*.

    ``d2q9.W`` stores float32 constants; for float64 runs the exact binary
    expressions (``4/9``, ``1/9``, ``1/36``) avoid injecting float32
    round-off into fp64 arithmetic.
    """
    return torch.tensor(
        [
            4.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
        ],
        dtype=f.dtype,
        device=f.device,
    )


def cg_phase_field(
    f_r: torch.Tensor,
    f_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(rho_r, rho_b, rho, phi)`` with ``phi = (rho_r - rho_b)/rho``."""
    rho_r = f_r.sum(dim=0)
    rho_b = f_b.sum(dim=0)
    rho = rho_r + rho_b
    phi = (rho_r - rho_b) / rho.clamp(min=1e-12)
    return rho_r, rho_b, rho, phi


def cg_phase_gradient(
    rho_r: torch.Tensor,
    rho_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Color-field gradient via second-order central differences (periodic).

    Identical convention to ``multiphase._grad_phase_field``; returns
    ``(phi, n_hat_x, n_hat_y, mag)``.
    """
    n = rho_r + rho_b
    n_safe = torch.clamp(n, min=1e-12)
    phi = (rho_r - rho_b) / n_safe
    dphi_dx = 0.5 * (torch.roll(phi, -1, dims=-1) - torch.roll(phi, 1, dims=-1))
    dphi_dy = 0.5 * (torch.roll(phi, -1, dims=-2) - torch.roll(phi, 1, dims=-2))
    mag = torch.sqrt(dphi_dx**2 + dphi_dy**2)
    mag_safe = torch.clamp(mag, min=1e-12)
    return phi, dphi_dx / mag_safe, dphi_dy / mag_safe, mag


def color_gradient_two_tau_step(
    f_r: torch.Tensor,
    f_b: torch.Tensor,
    tau_r: float = 1.0,
    tau_b: float = 1.0,
    A: float = 0.04,
    beta: float = 0.7,
    gx: float = 0.0,
    gy: float = 0.0,
    solid_mask: torch.Tensor | None = None,
    visc_mode: str = "mixture",
) -> tuple[torch.Tensor, torch.Tensor]:
    """One Color-Gradient iteration (collision only; streaming is external).

    Sequence per iteration:

    (a) macroscopic color densities and mixture velocity ``u = j / rho``;
    (b) collision towards the shared mixture velocity at the Guo half-step
        shift ``u_eq = u + g/2`` plus the three-term Guo source with
        ``F = rho g`` (exact discrete momentum input ``rho g`` per step),
        in one of two viscosity-blend modes (``visc_mode``):
        * ``"mixture"`` (default): collide the *total* distribution with the
          mass-weighted relaxation-time field ``tau_eff = (rho_r tau_r +
          rho_b tau_b) / rho``.  The emergent kinematic viscosity is the
          arithmetic blend ``nu = sum (rho_sigma/rho) nu_sigma`` (the
          continuum-correct dynamic-viscosity mixture rule), pure-layer
          values ``nu_sigma = cs^2 (tau_sigma - 1/2)`` in pure regions, and
          for ``tau_r == tau_b`` the step reduces algebraically to the stock
          ``multiphase.color_gradient_step`` (minus its first-order force
          shift, which the full Guo source replaces).
        * ``"per_color"``: each color collides separately with its own
          ``tau_sigma``.  A homogeneous mixture then realizes the
          *harmonic*-tau blend ``nu = cs^2 (2/(1/tau_r + 1/tau_b) - 1/2)``
          (measured by shear-wave decay, NOTES 2026-10-04), which is why
          ``"mixture"`` is the default.
    (c) surface-tension perturbation on the total post-collision
        distribution, ``Delta f_i = (A/2) |grad phi| w_i [(c_i . n)^2 - 1/3]``
        (identical form to the stock step);
    (d) recoloring (identical form to the stock step, Latva-Kokko &
        Rothman 2005): ``f_r^out = (rho_r/rho) f** + amp``,
        ``f_b^out = (rho_b/rho) f** - amp`` with
        ``amp_i = beta (rho_r rho_b / rho) w_i (c_i . n_hat)``.

    Args:
        f_r:         Red (heavy) distribution ``(9, ny, nx)``.
        f_b:         Blue (light) distribution ``(9, ny, nx)``.
        tau_r:       Red relaxation time (``> 0.5``; ``nu_r = (tau_r - .5)/3``).
        tau_b:       Blue relaxation time (``> 0.5``; ``nu_b = (tau_b - .5)/3``).
        A:           Surface-tension amplitude (stock default ``0.04``).
        beta:        Recolor/segregation parameter in ``(0, 1]``.
        gx, gy:      Body-force acceleration per unit mass (same for both colors).
        solid_mask:  Optional ``(ny, nx)`` boolean mask; solid cells skip the
                     whole step (stock convention: pre-step distributions are
                     preserved; the color gradient is computed on masked
                     densities to avoid spurious wall tension).
        visc_mode:   ``"mixture"`` (arithmetic tau blend on the total
                     distribution) or ``"per_color"`` (separate per-color
                     relaxation).

    Returns:
        Post-collision ``(f_r_out, f_b_out)``.
    """
    if f_r.shape != f_b.shape:
        raise ValueError(f"shape mismatch: f_r {tuple(f_r.shape)} vs f_b {tuple(f_b.shape)}")
    if tau_r <= 0.5 or tau_b <= 0.5:
        raise ValueError(f"tau must exceed 0.5 (tau_r={tau_r}, tau_b={tau_b})")
    if visc_mode not in ("mixture", "per_color"):
        raise ValueError(f"visc_mode must be 'mixture' or 'per_color', got {visc_mode!r}")

    device = f_r.device
    c = C.to(device=device)
    cx = c[:, 0].to(f_r.dtype).view(9, 1, 1)
    cy = c[:, 1].to(f_r.dtype).view(9, 1, 1)
    w_v = _weights_like(f_r).view(9, 1, 1)

    # --- (a) macroscopic color densities and mixture velocity -------------
    rho_r = f_r.sum(dim=0)
    rho_b = f_b.sum(dim=0)
    rho = rho_r + rho_b
    rho_s = rho.clamp(min=1e-12)
    jx = ((f_r + f_b) * cx).sum(dim=0)
    jy = ((f_r + f_b) * cy).sum(dim=0)
    ux = jx / rho_s
    uy = jy / rho_s

    # Guo physical velocity: half-step force shift, identical for both colors
    # because the acceleration g (per unit mass) is color-independent.
    ux_eq = ux + 0.5 * gx
    uy_eq = uy + 0.5 * gy

    # --- (b) collision + Guo force -----------------------------------------
    cu = cx * ux_eq.unsqueeze(0) + cy * uy_eq.unsqueeze(0)  # (9, ny, nx)
    u_sq = (ux_eq**2 + uy_eq**2).unsqueeze(0)

    def _feq(rho_field: torch.Tensor) -> torch.Tensor:
        return w_v * rho_field.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq)

    def _guo(rho_field: torch.Tensor, tau_field: torch.Tensor | float) -> torch.Tensor:
        fx = rho_field * gx
        fy = rho_field * gy
        ci_f = cx * fx.unsqueeze(0) + cy * fy.unsqueeze(0)
        u_f = (ux_eq * fx + uy_eq * fy).unsqueeze(0)
        src = w_v * (3.0 * ci_f + 9.0 * cu * ci_f - 3.0 * u_f)
        return (1.0 - 0.5 / tau_field) * src

    if visc_mode == "mixture":
        tau_eff = (rho_r * tau_r + rho_b * tau_b) / rho_s
        f_post = (f_r + f_b) - ((f_r + f_b) - _feq(rho)) / tau_eff + _guo(rho, tau_eff)
    else:

        def _collide(f_sig, rho_sig, tau):
            return f_sig - (f_sig - _feq(rho_sig)) / tau + _guo(rho_sig, tau)

        f_post = _collide(f_r, rho_r, tau_r) + _collide(f_b, rho_b, tau_b)

    # --- (c) surface-tension perturbation ---------------------------------
    # Masked densities avoid spurious color gradients at solid boundaries
    # (stock convention).
    rho_r_safe = rho_r if solid_mask is None else rho_r.masked_fill(solid_mask, 0.0)
    rho_b_safe = rho_b if solid_mask is None else rho_b.masked_fill(solid_mask, 0.0)
    _phi, nhat_x, nhat_y, mag = cg_phase_gradient(rho_r_safe, rho_b_safe)

    ci_dot_n = cx * nhat_x.unsqueeze(0) + cy * nhat_y.unsqueeze(0)
    perturbation = (A / 2.0) * mag.unsqueeze(0) * w_v * (ci_dot_n**2 - _CS2)
    f_post = f_post + perturbation

    # --- (d) recoloring (segregation) --------------------------------------
    recolor_amp = beta * (rho_r * rho_b / rho_s).unsqueeze(0) * ci_dot_n * w_v
    f_r_out = (rho_r / rho_s).unsqueeze(0) * f_post + recolor_amp
    f_b_out = (rho_b / rho_s).unsqueeze(0) * f_post - recolor_amp

    # Solid cells keep their pre-step distributions (bounce-back is applied
    # by the caller after streaming, stock convention).
    if solid_mask is not None:
        mask_3d = solid_mask.unsqueeze(0)
        f_r_out = torch.where(mask_3d, f_r, f_r_out)
        f_b_out = torch.where(mask_3d, f_b, f_b_out)

    return f_r_out, f_b_out


# ---------------------------------------------------------------------------
# Measurement helpers (conventions shared with multiphase_benchmarks)
# ---------------------------------------------------------------------------


def cg_pressure_jump(f_total: torch.Tensor, r: float) -> tuple[float, float, float]:
    """``(p_inside, p_outside, delta_p)`` with ``p = cs^2 rho = rho/3``.

    Convention copy of ``multiphase_benchmarks._measure_pressure_jump``:
    inside band ``r <= 0.5 R``, outside band ``r >= 1.5 R`` around the
    domain centre.
    """
    rho = f_total.sum(dim=0)
    ny, nx = rho.shape
    ys = torch.arange(ny, dtype=torch.float64, device=rho.device)
    xs = torch.arange(nx, dtype=torch.float64, device=rho.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    cy, cx = ny / 2.0, nx / 2.0
    r_field = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    inside = r_field <= r * 0.5
    outside = r_field >= r * 1.5
    p_in = (
        float((_CS2 * rho[inside].to(torch.float64)).mean().item())
        if inside.any()
        else float("nan")
    )
    p_out = (
        float((_CS2 * rho[outside].to(torch.float64)).mean().item())
        if outside.any()
        else float("nan")
    )
    return p_in, p_out, p_in - p_out


def cg_max_velocity(f_total: torch.Tensor) -> float:
    """Maximum ``|u|`` over all nodes (convention copy of ``_max_velocity``)."""
    rho = f_total.sum(dim=0).clamp(min=1e-12)
    c = C.to(f_total.device).to(f_total.dtype)
    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)
    ux = (f_total * cx).sum(dim=0) / rho
    uy = (f_total * cy).sum(dim=0) / rho
    return float(torch.sqrt(ux**2 + uy**2).max().item())


def cg_interface_width_cut(phi: torch.Tensor, row: int | None = None) -> int:
    """Count nodes with ``|phi| <= 0.9`` along a horizontal cut.

    Coarse interface-width proxy for stability diagnostics (droplet runs use
    the centre row; the absolute value maps a tanh profile of parameter w to
    ``2 * atanh(0.9) * w`` nodes).
    """
    line = phi if row is None else phi[row, :]
    return int((line.abs() <= 0.9).sum().item())
