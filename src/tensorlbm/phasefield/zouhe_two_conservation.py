"""Two-conservation Zou-He boundaries for free-energy two-distribution LBM.

The free-energy (Cahn-Hilliard) lattice-Boltzmann model carries **two**
conserved quantities on **two** D2Q9 populations:

* ``f`` -- fluid mass ``rho = sum_i f_i`` and momentum ``rho u = sum_i c_i f_i + F/2``
  (Guo, Zheng & Shi 2002 half-force convention),
* ``g`` -- order parameter ``psi = sum_i g_i`` with advective flux ``sum_i c_i g_i = psi u``;
  the chemical potential ``mu`` enters the equilibrium (``B_i`` terms) and drives the
  Cahn-Hilliard diffusive flux through relaxation.

The classical Zou & He (1997) reconstruction assumes ``sum_i c_i f_i = rho u`` and has
no notion of ``psi``.  In the presence of the capillary/chemical-potential force
``F = mu grad(psi)`` that assumption is violated: the reconstructed density and the
transverse velocity are off by ``O(F/2)`` at the open boundary, and the order parameter
has no boundary closure at all (its incoming populations are simply left as the wrapped
streaming values).

This module implements the force-consistent two-conservation reconstruction used by
``ebinan92/Fingering_dynamics`` (``lattice_boltzmann/fingering.py:303,304,322,369``,
catalogued as ``oss_solutions`` id=191, tag
``two-conservation-zouhe-with-chemical-potential-flux``):

Inlet (x = 0), flow in +x, unknown incoming directions {1, 5, 8}::

    rho = (Kf + f3 + f6 + f7 - Fx/2) / (1 - u_x)          # Kf = sum of known f
    f1  = f3 + (2/3) rho u_x                              # non-equilibrium bounce-back
    f5  = (S + D)/2,  f8 = (S - D)/2
    S = rho u_x - Fx/2 + f3 + f6 + f7 - f1
    D = rho u_y - Fy/2 - f2 - f6 + f4 + f7

so that, exactly (machine precision),

    sum_i f_i          = rho
    sum_i c_i f_i + F/2 = rho u

For the order parameter the same algebra on ``g`` with the *flux-consistent* closure
gives ``psi = (Kg + g3 + g6 + g7) / (1 - u_x)`` and
``sum_i g_i = psi``, ``sum_i c_i g_i = psi u``.  The upstream "deficit distribution"
variant (``psi`` prescribed, unknown populations split in proportion to
``w1 : w5 : w8``) is available as ``psi_mode="impose"`` for comparison.

References
----------
* Zou, Q. & He, X. (1997) *On pressure and velocity boundary conditions for the
  lattice Boltzmann BGK model*, Phys. Fluids 9, 1591.
* Guo, Z., Zheng, C. & Shi, B. (2002) *Discrete lattice effects on the forcing term in
  the lattice Boltzmann method*, Phys. Rev. E 65, 046308.
* Swift, M., Osborn, W. & Yeomans, J. (1995) *Lattice Boltzmann simulation of
  nonideal fluids*, Phys. Rev. Lett. 75, 830 (free-energy two-distribution model).
* Kruger et al. (2017) *The Lattice Boltzmann Method: Principles and Practice*,
  Springer, Sec. 9.4 (force-consistent velocity/pressure boundaries).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# D2Q9 layout identical to ``tensorlbm.d2q9`` (kept local as float64 tensors are
# created per call to avoid device/dtype promotion surprises in the BC hot path).
_CX: tuple[int, ...] = (0, 1, 0, -1, 0, 1, -1, -1, 1)
_CY: tuple[int, ...] = (0, 0, 1, 0, -1, 1, 1, -1, -1)
_W: tuple[float, ...] = (4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36)

#: unknown incoming directions at the x = 0 column (cx > 0)
INLET_UNKNOWN = (1, 5, 8)
#: unknown outgoing directions at the x = nx-1 column (cx < 0)
OUTLET_UNKNOWN = (3, 6, 7)


def _zeros(ref: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(ref)


def _as_column(value: float | torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Broadcast a scalar / 1-D column / (ny, 1) tensor to the column shape of ``ref``."""
    ref = ref.reshape(-1)
    if isinstance(value, torch.Tensor):
        col = value.to(device=ref.device, dtype=ref.dtype).reshape(-1)
        if col.numel() == 1:
            return col.expand_as(ref)
        if col.shape != ref.shape:
            msg = f"boundary profile must have shape {tuple(ref.shape)}, got {tuple(col.shape)}"
            raise ValueError(msg)
        return col
    return torch.full_like(ref, float(value))


@dataclass
class ColumnState:
    """Invariants of one open-boundary column after reconstruction."""

    rho: torch.Tensor
    ux: torch.Tensor
    uy: torch.Tensor
    psi: torch.Tensor
    mass_flux: torch.Tensor  # sum_i c_{i,x} f_i  (+ F_x/2 when force consistent)
    psi_flux: torch.Tensor  # sum_i c_{i,x} g_i


def column_invariants(
    f_col: torch.Tensor,
    g_col: torch.Tensor,
    force: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
    *,
    force_correction: bool = True,
) -> ColumnState:
    """Measure ``(rho, u, psi)`` and the x-fluxes of a column of populations.

    ``force`` is the (Fx, Fy) chemical-potential force density on that column.  With
    ``force_correction=True`` the Guo half-force velocity ``u = (j + F/2)/rho`` is used.
    """
    cx = torch.tensor(_CX, dtype=f_col.dtype, device=f_col.device).reshape(-1, 1)
    cy = torch.tensor(_CY, dtype=f_col.dtype, device=f_col.device).reshape(-1, 1)
    fx = _zeros(f_col[:, 0]) if force is None or force[0] is None else force[0]
    fy = _zeros(f_col[:, 0]) if force is None or force[1] is None else force[1]

    rho = f_col.sum(0)
    jx = (f_col * cx).sum(0)
    jy = (f_col * cy).sum(0)
    psi = g_col.sum(0)
    psi_flux = (g_col * cx).sum(0)
    half = 0.5 if force_correction else 0.0
    ux = (jx + half * fx) / rho
    uy = (jy + half * fy) / rho
    return ColumnState(
        rho=rho,
        ux=ux,
        uy=uy,
        psi=psi,
        mass_flux=jx + half * fx,
        psi_flux=psi_flux,
    )


def reconstruct_inlet(
    f_col: torch.Tensor,
    g_col: torch.Tensor,
    u_in: float | torch.Tensor,
    force: tuple[float | torch.Tensor, float | torch.Tensor] | None = None,
    *,
    psi_mode: str = "flux",
    psi_target: float | torch.Tensor | None = None,
    force_correction: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, ColumnState]:
    """Force-consistent two-conservation Zou-He inlet at a left column.

    Args:
        f_col: ``(9, ny)`` fluid populations at the inlet column (after streaming).
        g_col: ``(9, ny)`` order-parameter populations at the inlet column.
        u_in: prescribed streamwise velocity ``u_x``.
        force: ``(Fx, Fy)`` chemical-potential force density on the column.
        psi_mode: ``"flux"`` (flux-consistent psi, conserves ``psi`` and ``psi u``) or
            ``"impose"`` (prescribe ``psi_target`` and split the deficit over the unknown
            directions with weights ``w1 : w5 : w8`` -- the upstream variant).
        psi_target: target order parameter (required for ``psi_mode="impose"``).
        force_correction: ``False`` reproduces the naive Zou-He reconstruction that
            ignores the half-force term (used as the negative control).

    Returns:
        ``(f_new, g_new, state)`` in the same layout as the inputs.
    """
    ref = f_col[0]
    ux = _as_column(u_in, ref)
    if force is None:
        fx = torch.zeros_like(ref)
        fy = torch.zeros_like(ref)
    else:
        fx = _as_column(force[0], ref)
        fy = _as_column(force[1], ref)
    half = 0.5 if force_correction else 0.0

    f0, f2, f3, f4, f6, f7 = f_col[0], f_col[2], f_col[3], f_col[4], f_col[6], f_col[7]
    known_f = f0 + f2 + f4
    rho = (known_f + 2.0 * (f3 + f6 + f7) - half * fx) / (1.0 - ux)

    f1 = f3 + (2.0 / 3.0) * rho * ux
    s58 = rho * ux - half * fx + (f3 + f6 + f7) - f1
    d58 = -half * fy - f2 - f6 + f4 + f7
    f5 = 0.5 * (s58 + d58)
    f8 = 0.5 * (s58 - d58)

    f_new = f_col.clone()
    f_new[1], f_new[5], f_new[8] = f1, f5, f8

    g0, g2, g3, g4, g6, g7 = g_col[0], g_col[2], g_col[3], g_col[4], g_col[6], g_col[7]
    known_g = g0 + g2 + g4
    psi_flux = (known_g + 2.0 * (g3 + g6 + g7)) / (1.0 - ux)

    if psi_mode == "flux":
        psi = psi_flux
        g1 = g3 + (2.0 / 3.0) * psi * ux
    elif psi_mode == "impose":
        # Upstream (ebinan92/Fingering_dynamics) variant: psi is prescribed and the
        # population deficit is split over the unknown directions in proportion to
        # w1 : w5 : w8.  Sigma g = psi exactly, but the advective psi-flux is not
        # enforced -- this is the negative control in the XVAL comparison.
        if psi_target is None:
            msg = "psi_mode='impose' requires psi_target"
            raise ValueError(msg)
        psi = _as_column(psi_target, ref)
        deficit = psi - (known_g + g3 + g6 + g7)
        total = _W[1] + _W[5] + _W[8]
        g_new_imp = g_col.clone()
        g_new_imp[1] = deficit * (_W[1] / total)
        g_new_imp[5] = deficit * (_W[5] / total)
        g_new_imp[8] = deficit * (_W[8] / total)
        state = column_invariants(f_new, g_new_imp, (fx, fy), force_correction=force_correction)
        return f_new, g_new_imp, state
    else:
        msg = f"psi_mode must be 'flux' or 'impose', got {psi_mode!r}"
        raise ValueError(msg)

    s58g = psi - (known_g + g3 + g6 + g7) - g1
    d58g = -g2 - g6 + g4 + g7
    g5 = 0.5 * (s58g + d58g)
    g8 = 0.5 * (s58g - d58g)

    g_new = g_col.clone()
    g_new[1], g_new[5], g_new[8] = g1, g5, g8
    state = column_invariants(f_new, g_new, (fx, fy), force_correction=force_correction)
    return f_new, g_new, state


def reconstruct_outlet(
    f_col: torch.Tensor,
    g_col: torch.Tensor,
    rho_out: float | torch.Tensor = 1.0,
    force: tuple[float | torch.Tensor, float | torch.Tensor] | None = None,
    *,
    psi_out: float | torch.Tensor | None = None,
    force_correction: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, ColumnState]:
    """Pressure-type two-conservation Zou-He outlet at a right column.

    ``rho`` is prescribed (pressure boundary).  ``psi`` is taken from ``psi_out``
    (normally the upstream column, i.e. a zero-gradient / transmissive closure); when
    ``None`` the flux-consistent value implied by the outgoing populations is used.
    Both ``rho`` and ``psi`` are conserved exactly by construction.
    """
    ref = f_col[0]
    rho = _as_column(rho_out, ref)
    if force is None:
        fx = torch.zeros_like(ref)
        fy = torch.zeros_like(ref)
    else:
        fx = _as_column(force[0], ref)
        fy = _as_column(force[1], ref)
    half = 0.5 if force_correction else 0.0

    f0, f1, f2, f4, f5, f8 = f_col[0], f_col[1], f_col[2], f_col[4], f_col[5], f_col[8]
    known_f = f0 + f1 + f2 + f4 + f5 + f8
    ux = (half * fx + (f1 + f5 + f8) - rho + known_f) / rho
    f3 = f1 - (2.0 / 3.0) * rho * ux
    s67 = rho - known_f - f3
    d67 = -half * fy - f2 - f5 + f4 + f8
    f6 = 0.5 * (s67 + d67)
    f7 = 0.5 * (s67 - d67)
    f_new = f_col.clone()
    f_new[3], f_new[6], f_new[7] = f3, f6, f7

    g0, g1, g2, g4, g5, g8 = g_col[0], g_col[1], g_col[2], g_col[4], g_col[5], g_col[8]
    known_g = g0 + g1 + g2 + g4 + g5 + g8
    if psi_out is None:
        # flux-consistent psi: psi (1 + u_x) = (g0+g2+g4) + 2 (g1+g5+g8)
        psi = ((g0 + g2 + g4) + 2.0 * (g1 + g5 + g8)) / (1.0 + ux)
    else:
        psi = _as_column(psi_out, ref)
    g3 = g1 - (2.0 / 3.0) * psi * ux
    s67g = psi - known_g - g3
    d67g = -g2 - g5 + g4 + g8
    g6 = 0.5 * (s67g + d67g)
    g7 = 0.5 * (s67g - d67g)
    g_new = g_col.clone()
    g_new[3], g_new[6], g_new[7] = g3, g6, g7
    state = column_invariants(f_new, g_new, (fx, fy), force_correction=force_correction)
    return f_new, g_new, state


@dataclass
class TwoConservationBudget:
    """Discrete conservation ledger for one time step (per full domain)."""

    mass_drift: float
    psi_drift: float
    mass_flux_in: float
    mass_flux_out: float
    psi_flux_in: float
    psi_flux_out: float
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def mass_flux_residual(self) -> float:
        denom = max(abs(self.mass_flux_in), abs(self.mass_flux_out), 1e-30)
        return (self.mass_flux_out - self.mass_flux_in) / denom

    @property
    def psi_flux_residual(self) -> float:
        denom = max(abs(self.psi_flux_in), abs(self.psi_flux_out), 1e-30)
        return (self.psi_flux_out - self.psi_flux_in) / denom


__all__ = [
    "INLET_UNKNOWN",
    "OUTLET_UNKNOWN",
    "ColumnState",
    "TwoConservationBudget",
    "column_invariants",
    "reconstruct_inlet",
    "reconstruct_outlet",
]