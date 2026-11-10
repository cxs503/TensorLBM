"""LBM free-surface fill-transport Zalesak rotation benchmark (D2Q9 / D3Q19).

Why this module exists
----------------------
:mod:`tensorlbm.free_surface_zalesak` grades the *finite-volume* face-flux VOF
operator (4 faces in 2-D, 6 in 3-D, MUSCL reconstruction).  A production LBM
free-surface code does **not** transport the liquid fraction that way: the fill
level of a GAS/LIQUID/INTERFACE cell is carried by the *lattice populations* on
the full D2Q9 / D3Q19 velocity set (Körner et al. 2005).  This module
implements that second, structurally different transport path and grades it on
the same classical benchmark, so the two families can be compared on identical
geometry, identical metrics and identical velocity fields.

Physics
-------
Zalesak's (1979) slotted cylinder is placed off-centre in a periodic unit box
and carried by the prescribed, analytically divergence-free rigid rotation

    u(x) = Omega * ( -(y - y0), x - x0 )              (2-D, D2Q9)
    u(x) = Omega * ( -(z - z0), 0, x - x0 )           (3-D, D3Q19, x-z plane)

After one full revolution ``t = 2*pi/Omega`` the exact solution is the initial
condition again, so every deviation is a discretisation error of the
fill-transport operator.  The velocity is prescribed rather than solved because
the benchmark is deliberately kinematic -- it isolates the interface operator
(Zalesak 1979; Rider & Kothe 1998) -- and because a *free* liquid blob spun up
at zero gravity cannot hold solid-body rotation at all (that would require
``p < p_atm`` inside the blob), so a fully coupled run has no known exact
solution to grade against.  The prescribed field is exactly divergence-free on
the lattice (checked in :func:`lbm_case_metrics`), hence every volume error
reported is an interface-operator error, not a compressibility artefact.

Transport operator
------------------
State: ``(phi, g)`` -- ``phi`` the liquid fraction (0 = gas, 1 = liquid, sub-cell
values live in INTERFACE cells) and ``g`` the ``Q`` lattice populations of that
fraction (``Q = 9`` D2Q9, ``19`` D3Q19) with ``sum_q g_q = phi``.

One step is a scalar-lattice BGK update written in flux form,

    g_q^eq  = w_q phi (1 + 3 c_q . u)
    g_q^H   = 2 g_q^eq - g_q                     (tau = 1/2: zero diffusivity)
    phi^{n+1}(i) = sum_q g_q^H(i - c_q)

with the pull-index streaming of the D2Q9 / D3Q19 stencil.  ``tau = 1/2`` is the
zero-diffusivity point of the scalar lattice Boltzmann equation
(``nu = c_s^2 (tau - 1/2)``), i.e. the fill level is transported by pure lattice
advection -- the LBM counterpart of the anti-diffusive part of an FCT scheme.
Because every population is a *single-valued* link flux, summing the update
over the box telescopes and ``sum(phi)`` is conserved to round-off.

Pure lattice advection disperses at a jump (Lax-Wendroff-type oscillations), so
the free-surface mass ledger is applied afterwards
(:func:`bound_ledger`): a cell that overfills hands the surplus to neighbours
with spare capacity, a cell that over-empties draws the deficit from neighbours
that still carry fill -- the continuous analogue of the Körner (2005)
excess-mass redistribution on I -> L / I -> G conversion, and never silently
discarding liquid.  The ledger conserves ``sum(phi)`` exactly and returns
``phi in [0, 1]``.

The following schemes are available and compared by the campaign driver:

``"lbm"``   BGK LBM flux (default tau = 0.504, i.e. nu = 1.33e-3 lattice units --
            the usable band sits just above the tau = 1/2 stability boundary)
            plus the conservative bound ledger (**proposed**),
``"high"``  tau = 1/2 LBM flux, unlimited (sharpest, oscillatory control: it is
            only marginally stable on a discontinuous fill field and diverges
            over one revolution -- kept as the negative control),
``"low"``   first-order equilibrium flux ``g^eq`` only (diffusive control),
``"fct"``   tau = 1/2 flux with a Zalesak (1979) capacity limiter applied to the
            anti-diffusive link flux ``A_q = g_q^eq - g_q`` (bounded control),

Interface bookkeeping
---------------------
The free-surface flag layer (``GAS / LIQUID / INTERFACE``) is derived from the
fill level with the Körner thresholds; the number of INTERFACE cells per unit
interface length is the *band width in cells* of the free-surface
representation, reported alongside the continuous 10-90 % thickness of the
reconstructed ``phi`` field.

References
----------
Zalesak, S. T. (1979). Fully multidimensional flux-corrected transport
    algorithms for fluids. J. Comput. Phys. 31, 335-362.
Körner, C., Thies, M., Hofmann, T., Thürey, N., & Rüde, U. (2005). Lattice
    Boltzmann model for free surface flow for modeling foaming.
    J. Comput. Phys. 202, 68-84.
Rider, W. J., & Kothe, D. B. (1998). Reconstructing volume tracking.
    J. Comput. Phys. 141, 112-152.
Roache, P. J. (1994). Perspective: a method for uniform reporting of grid
    refinement studies. J. Fluids Eng. 116, 405-413.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from .d2q9 import OPPOSITE as _OPP2
from .d2q9 import W_EXACT as _W2
from .d2q9 import C as _C2
from .d3q19 import OPPOSITE as _OPP3
from .d3q19 import W_EXACT64 as _W3
from .d3q19 import C as _C3
from .free_surface_zalesak import (
    convergence_order,
    gci,
    hausdorff_distance,
    interface_thickness,
    l1_shape_error,
    liquid_volume,
    rigid_rotation_field,
    shape_overlap,
    slotted_cylinder_phi_2d,
    slotted_sphere_phi_3d,
)

__all__ = [
    "GAS",
    "LIQUID",
    "INTERFACE",
    "LATTICE_NAMES",
    "SCHEMES",
    "lattice",
    "cell_velocity",
    "equilibrium_fill",
    "stream_sum",
    "bound_ledger",
    "fill_transport_step",
    "flags_from_fill",
    "interface_band_cells",
    "initial_fill",
    "lbm_case_metrics",
    "run_lbm_zalesak_case",
    "convergence_order",
    "gci",
]

# Free-surface cell types (Körner et al. 2005).
GAS = 0
LIQUID = 1
INTERFACE = 2

LATTICE_NAMES = {2: "D2Q9", 3: "D3Q19"}
SCHEMES = ("lbm", "high", "low", "fct")

# Exact double-precision weights, keyed by dimension.
_W_EXACT = {2: tuple(float(x) for x in _W2), 3: tuple(float(x) for x in _W3.tolist())}


# --------------------------------------------------------------------------- #
# Lattice helpers                                                             #
# --------------------------------------------------------------------------- #


def lattice(dim: int, *, device="cpu", dtype=torch.float64):
    """Return ``(name, C, w, opposite)`` for the ``dim``-dimensional lattice.

    ``C`` has shape ``(Q, dim)`` with components in ``(x, y[, z])`` order -- the
    convention of :mod:`tensorlbm.d2q9` / :mod:`tensorlbm.d3q19` -- while fields
    are stored ``(..., y, x)`` / ``(..., z, y, x)``, so the component driving
    field axis ``d`` is ``C[:, dim-1-d]``.
    """
    if dim not in LATTICE_NAMES:
        raise ValueError(f"dim must be 2 or 3, got {dim}")
    device = torch.device(device)
    c_src, opp = (_C2, _OPP2) if dim == 2 else (_C3, _OPP3)
    C = c_src.to(device=device, dtype=dtype)
    w = torch.tensor(_W_EXACT[dim], device=device, dtype=dtype)
    return LATTICE_NAMES[dim], C, w, opp.to(device)


def _shifts(C: torch.Tensor, dim: int) -> tuple[tuple[int, ...], ...]:
    """Pull-source roll shifts per direction, in field axis order."""
    return tuple(tuple(reversed(c[:dim])) for c in C.to(torch.int64).tolist())


def _axes(dim: int) -> tuple[int, ...]:
    return tuple(range(dim))


def stream_sum(field: torch.Tensor, shifts: Sequence[Sequence[int]]) -> torch.Tensor:
    """``sum_q roll(field[q], c_q)`` -- the population mass arriving per cell."""
    dim = field.dim() - 1
    axes = _axes(dim)
    out = torch.zeros_like(field[0])
    for q, s in enumerate(shifts):
        out = out + torch.roll(field[q], shifts=s, dims=axes)
    return out


def _stream(field: torch.Tensor, shifts: Sequence[Sequence[int]]) -> torch.Tensor:
    dim = field.dim() - 1
    axes = _axes(dim)
    return torch.stack(
        [torch.roll(field[q], shifts=s, dims=axes) for q, s in enumerate(shifts)]
    )


def cell_velocity(
    n: int,
    *,
    dim: int = 2,
    omega: float = 1.0,
    centre: Sequence[float] = (0.5, 0.5),
    device="cpu",
    dtype=torch.float64,
) -> list[torch.Tensor]:
    """Analytic rigid-rotation velocity in *array axis* order.

    Wraps :func:`tensorlbm.free_surface_zalesak.rigid_rotation_field` and returns
    the components ordered so that ``u[d]`` multiplies array axis ``d``
    (``[u_y, u_x]`` in 2-D, ``[u_z, u_y, u_x]`` in 3-D).
    """
    comps = rigid_rotation_field(
        n, dim=dim, centre=centre, omega=omega, device=device, dtype=dtype
    )
    return list(reversed(comps))


def equilibrium_fill(
    phi: torch.Tensor, u: Sequence[torch.Tensor], C: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    """Scalar-lattice equilibrium ``g_q^eq = w_q phi (1 + 3 c_q . u)``."""
    dim = phi.dim()
    shape = (-1,) + (1,) * dim
    cu = torch.zeros_like(phi)
    for d in range(dim):
        cu = cu + C[:, dim - 1 - d].view(shape) * u[d]
    return w.view(shape) * phi.unsqueeze(0) * (1.0 + 3.0 * cu)


# --------------------------------------------------------------------------- #
# Conservative bound ledger (Körner excess-mass redistribution)               #
# --------------------------------------------------------------------------- #


def bound_ledger(
    phi: torch.Tensor, *, iters: int = 8, relax: float = 0.5
) -> tuple[torch.Tensor, dict]:
    """Push ``phi`` back into ``[0, 1]`` while conserving ``sum(phi)`` exactly.

    Surplus (``phi > 1``) is handed to neighbouring cells with spare capacity and
    deficit (``phi < 0``) is drawn from neighbouring cells that still carry
    fill; the surplus/deficit field is diffused over the ``2*dim`` axis
    neighbours, so the transfer stays local.  Whatever does not fit inside the
    box after ``iters`` passes is spread over the remaining capacity of the whole
    box, so the total liquid volume is never silently discarded.

    Returns ``(phi_bounded, info)``; ``info["moved_fraction"]`` is the moved
    volume as a fraction of the total, ``info["global_correction"]`` the
    residual spread over the whole box (both should be tiny for a resolved run).
    """
    lim = phi.clamp(0.0, 1.0)
    resid = phi - lim
    dim = phi.dim()
    scale = float(lim.sum().item())
    moved = 0.0
    global_corr = 0.0
    for sign in (1.0, -1.0):
        r = (sign * resid).clamp(min=0.0)
        if float(r.max().item()) <= 0.0:
            continue
        for _ in range(iters):
            if float(r.max().item()) <= 1e-15:
                break
            spread = torch.zeros_like(r)
            for d in range(dim):
                for sh in (1, -1):
                    spread = spread + torch.roll(r, shifts=sh, dims=d)
            spread = spread / (2.0 * dim)
            r = r * (1.0 - relax) + relax * spread
            capacity = (1.0 - lim).clamp(min=0.0) if sign > 0 else lim.clamp(min=0.0)
            absorb = torch.minimum(r, capacity)
            lim = lim + absorb if sign > 0 else lim - absorb
            r = (r - absorb).clamp(min=0.0)
            moved += float(absorb.sum().item())
        rest = float(r.sum().item())
        if rest > 1e-14:
            global_corr += rest
            capacity = (1.0 - lim).clamp(min=0.0) if sign > 0 else lim.clamp(min=0.0)
            denom = float(capacity.sum().item())
            if denom > 0.0:
                delta = rest * capacity / denom
                lim = lim + delta if sign > 0 else lim - delta
            else:  # pragma: no cover - degenerate all-gas / all-liquid box
                delta = rest / float(lim.numel())
                lim = lim + delta if sign > 0 else lim - delta
    info = {
        "moved_fraction": moved / scale if scale > 0 else float("nan"),
        "global_correction": global_corr / scale if scale > 0 else float("nan"),
    }
    return lim, info


# --------------------------------------------------------------------------- #
# Fill-fraction LBM step                                                      #
# --------------------------------------------------------------------------- #


def fill_transport_step(
    phi: torch.Tensor,
    g: torch.Tensor,
    u: Sequence[torch.Tensor],
    *,
    dim: int,
    C: torch.Tensor,
    w: torch.Tensor,
    shifts: Sequence[Sequence[int]],
    scheme: str = "lbm",
    tau: float = 0.504,
    eps: float = 1e-300,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """One fill-fraction LBM step.

    Args:
        phi: Liquid fraction, shape ``(n,)*dim``.
        g: Populations of the fraction, shape ``(Q,) + (n,)*dim``, with
            ``g.sum(0) == phi`` (re-imposed by the isotropic correction below).
        u: Velocity components in array axis order.
        scheme: One of :data:`SCHEMES`.

    Returns:
        ``(phi_new, g_new, info)`` with ``g_new.sum(0) == phi_new`` and the total
        volume conserved to round-off for every scheme.
    """
    geq = equilibrium_fill(phi, u, C, w)
    info: dict = {}

    if scheme == "low":
        g_new = _stream(geq, shifts)
        return g_new.sum(0), g_new, info

    # BGK collision.  ``relax = 1 - 1/tau`` multiplies the non-equilibrium part;
    # relax = -1 is tau = 1/2 (zero diffusivity, marginal stability), relax = 0
    # is tau = 1 (nu = c_s^2/3, the diffusive control).
    # g^post = g + (g^eq - g)/tau  =>  g^post = geq + A,  A = (1 - 1/tau)(g - geq).
    relax = float(1.0 - 1.0 / tau)
    A = relax * (g - geq)
    if scheme == "high":
        g_new = _stream(geq + A, shifts)
        return g_new.sum(0), g_new, info

    if scheme == "fct":
        # Zalesak (1979) capacity limiter on the anti-diffusive link flux.  The
        # anti-diffusive part sums to zero over q at every cell, so it can be
        # written as an exact divergence of a single-valued link flux.
        phi_low = stream_sum(geq, shifts)
        out_tot = A.clamp(min=0.0).sum(0)
        gain = torch.zeros_like(phi)
        for q, s in enumerate(shifts):
            gain = gain + torch.roll(A[q].clamp(min=0.0), shifts=s, dims=_axes(dim))
        alpha = torch.where(
            out_tot > eps,
            (phi_low / out_tot.clamp(min=eps)).clamp(max=1.0),
            torch.ones_like(phi_low),
        )
        beta = torch.where(
            gain > eps,
            ((1.0 - phi_low) / gain.clamp(min=eps)).clamp(min=0.0, max=1.0),
            torch.ones_like(phi_low),
        )
        omega = torch.zeros_like(A)
        for q, s in enumerate(shifts):
            # The anti-diffusive link flux changes sign, so the donor / acceptor
            # roles must follow the sign of A_q(i).
            alpha_don = torch.where(
                A[q] > 0.0, alpha, torch.roll(alpha, shifts=tuple(-x for x in s), dims=_axes(dim))
            )
            beta_acc = torch.where(
                A[q] > 0.0, torch.roll(beta, shifts=tuple(-x for x in s), dims=_axes(dim)), beta
            )
            omega[q] = torch.where(A[q] != 0.0, torch.minimum(alpha_don, beta_acc), torch.zeros_like(A[q]))
        abar = A * omega
        phi_new = phi_low - (abar.sum(0) - stream_sum(abar, shifts))
        g_new = _renormalise(_stream(geq + abar, shifts), phi_new, w)
        return phi_new, g_new, info

    if scheme != "lbm":
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")

    # ---- proposed: BGK LBM flux + conservative bound ledger ----------------
    g_raw = _stream(geq + A, shifts)
    phi_raw = g_raw.sum(0)
    phi_new, info = bound_ledger(phi_raw)
    g_new = _renormalise(g_raw, phi_new, w)
    return phi_new, g_new, info


def _renormalise(g: torch.Tensor, phi: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Re-impose ``sum_q g_q = phi`` with the isotropic (weight-shaped) part.

    The bound ledger changes the cell content; the missing amount is added to
    every population in proportion to its lattice weight, which is the
    isotropic (velocity-independent) correction and therefore does not bias the
    transported flux direction.
    """
    dim = phi.dim()
    shape = (-1,) + (1,) * dim
    delta = phi - g.sum(0)
    return g + delta.unsqueeze(0) * w.view(shape)


# --------------------------------------------------------------------------- #
# Free-surface flag layer                                                     #
# --------------------------------------------------------------------------- #


def flags_from_fill(phi: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Körner cell types from the fill level: GAS / LIQUID / INTERFACE."""
    gas = phi <= eps
    full = phi >= 1.0 - eps
    flags = torch.full(phi.shape, INTERFACE, dtype=torch.int64, device=phi.device)
    flags = torch.where(gas, torch.full_like(flags, GAS), flags)
    return torch.where(full, torch.full_like(flags, LIQUID), flags)


def interface_band_cells(phi: torch.Tensor, *, thr: float = 0.5, dx: float | None = None) -> dict:
    """Width of the free-surface interface band, in grid cells.

    The Körner flag layer carries the interface in the cells whose fill is
    strictly between 0 and 1.  Counting those cells and dividing by the length
    of the ``phi = thr`` contour gives the *band width in cells* -- the
    free-surface analogue of the 10-90 % VOF thickness, but a topological
    (flag-based) quantity instead of a gradient-based one.
    """
    n = phi.shape[0]
    if dx is None:
        dx = 1.0 / n
    flags = flags_from_fill(phi)
    n_iface = int((flags == INTERFACE).sum().item())
    base = phi > thr
    length = 0.0
    for d in range(phi.dim()):
        cross = base ^ (torch.roll(phi, -1, dims=d) > thr)
        length += float(cross.sum().item()) * dx ** (phi.dim() - 1)
    # interface-cell area divided by contour length, converted to grid cells
    band = (
        n_iface * dx ** (phi.dim() - 1) / length if length > 1e-12 else float("nan")
    )
    return {"band_cells": band, "n_interface_cells": n_iface, "contour_length": length}


# --------------------------------------------------------------------------- #
# Initial condition                                                           #
# --------------------------------------------------------------------------- #


def initial_fill(
    n: int,
    *,
    dim: int = 2,
    cylindrical_slot: bool = False,
    extruded: bool = False,
    sub: int | None = None,
    device="cpu",
    dtype=torch.float64,
) -> torch.Tensor:
    """Initial (= exact final) fill field, reusing the shared geometry.

    ``extruded`` (3-D only) is the 2-D slotted disk in the x-z plane extruded
    through y: the *kernel-equivalence* configuration.  A D3Q19 run on it must
    keep the field y-invariant (the flow has no out-of-plane component), but it
    is **not** required to reproduce the D2Q9 cross-section to round-off: the
    transversal populations change the effective numerical diffusion of the
    lattice transport even when the flux through them vanishes by symmetry.  The
    measured cross-section deviation is reported by the campaign driver
    (``benchmarks/zalesak_rotation/run_zalesak_lbm_campaign.py``, stage ``3d``)
    and documented in ``docs/P190_N01_ZALESAK_LBM.md``.
    """
    device = torch.device(device)
    if dim == 2:
        return slotted_cylinder_phi_2d(
            n, sub=8 if sub is None else sub, device=device, dtype=dtype
        )
    if extruded:
        from .free_surface_zalesak import slotted_cylinder_extruded_phi_3d

        return slotted_cylinder_extruded_phi_3d(
            n, sub=4 if sub is None else sub, device=device, dtype=dtype
        )
    return slotted_sphere_phi_3d(
        n, cylindrical_slot=cylindrical_slot, sub=4 if sub is None else sub,
        device=device, dtype=dtype,
    )


def dirichlet_inflow_max(phi: torch.Tensor, u: Sequence[torch.Tensor], dim: int) -> float:
    """max |sum_d d_d u_d| on cell faces -- the divergence the transport sees."""
    total = torch.zeros_like(phi)
    for d in range(dim):
        face = 0.5 * (u[d] + torch.roll(u[d], -1, dims=d))
        total = total + (face - torch.roll(face, 1, dims=d))
    return float(total.abs().max().item())


# --------------------------------------------------------------------------- #
# Metrics + case runner                                                       #
# --------------------------------------------------------------------------- #


def lbm_case_metrics(phi: torch.Tensor, phi0: torch.Tensor, *, dim: int, history=None) -> dict:
    """All Zalesak metrics of one finished run."""
    dx = 1.0 / phi.shape[0]
    thick = interface_thickness(phi)
    band = interface_band_cells(phi, dx=dx)
    vol0 = liquid_volume(phi0)
    volT = liquid_volume(phi)
    return {
        "lattice": LATTICE_NAMES[dim],
        "n": int(phi.shape[0]),
        "volume_initial": vol0,
        "volume_final": volT,
        "volume_rel_change": (volT - vol0) / vol0,
        "volume_abs_rel_change": abs(volT - vol0) / vol0,
        "l1_shape_error": l1_shape_error(phi, phi0),
        "shape_overlap": shape_overlap(phi, phi0),
        "overlap_deficit": 1.0 - shape_overlap(phi, phi0),
        "hausdorff_distance": hausdorff_distance(phi, phi0),
        "interface_thickness_mean_cells": thick["mean"],
        "interface_thickness_p95_cells": thick["p95"],
        "interface_thickness_max_cells": thick["max"],
        "interface_band_cells": band["band_cells"],
        "n_interface_cells": band["n_interface_cells"],
        "fill_min": float(phi.min().item()),
        "fill_max": float(phi.max().item()),
        "bound_violation": float(max(0.0, -phi.min().item(), phi.max().item() - 1.0)),
        **({"history": history} if history else {}),
    }


def run_lbm_zalesak_case(
    n: int,
    *,
    dim: int = 2,
    scheme: str = "lbm",
    tau: float = 0.504,
    u_max_lat: float = 0.1,
    steps: int | None = None,
    cylindrical_slot: bool = False,
    extruded: bool = False,
    phi_init: torch.Tensor | None = None,
    sample_every: int | None = None,
    record_fields: bool = False,
    dtype=torch.float64,
    device="cpu",
    verbose: bool = False,
) -> dict:
    """Run one full revolution (``2*pi``) of the LBM fill-transport Zalesak case.

    The step count defaults to ``round(2*pi*R_max / (u_max_lat*dx))`` with
    ``R_max`` the largest in-plane distance from the rotation centre to liquid,
    and the lattice angular rate is ``omega = 2*pi/(steps*dx)``, so the run
    covers exactly ``2*pi`` of rotation with no end-of-run phase error.
    """
    device = torch.device(device)
    name, C, w, _opp = lattice(dim, device=device, dtype=dtype)
    shifts = _shifts(C, dim)
    dx = 1.0 / n

    phi0 = (
        initial_fill(
            n, dim=dim, cylindrical_slot=cylindrical_slot, extruded=extruded,
            device=device, dtype=dtype,
        )
        if phi_init is None
        else phi_init.to(device=device, dtype=dtype).clone()
    ).contiguous()

    idx = (phi0 > 0.5).nonzero().to(dtype)
    offsets = (idx[:, [0, dim - 1]] + 0.5) * dx - 0.5
    r_max = float(offsets.pow(2).sum(dim=1).sqrt().max().item())
    if steps is None:
        steps = max(8, int(round(2.0 * math.pi * r_max / (u_max_lat * dx))))
    omega = 2.0 * math.pi / (steps * dx)
    u = cell_velocity(n, dim=dim, omega=omega, centre=(0.5, 0.5), device=device, dtype=dtype)
    div_u = dirichlet_inflow_max(phi0, u, dim)

    phi = phi0.clone()
    g = equilibrium_fill(phi, u, C, w)
    m0 = float(phi.sum().item())
    if sample_every is None:
        sample_every = max(1, steps // 20)
    history = {"step": [], "mass_rel_drift": [], "l1": [], "overlap": [], "moved_fraction": []}
    max_step_drift = 0.0
    moved_sum = 0.0
    prev_m = m0
    for it in range(steps):
        phi, g, info = fill_transport_step(
            phi, g, u, dim=dim, C=C, w=w, shifts=shifts, scheme=scheme, tau=tau
        )
        moved_sum += info.get("moved_fraction", 0.0)
        m = float(phi.sum().item())
        max_step_drift = max(max_step_drift, abs(m - prev_m) / m0)
        prev_m = m
        if (it + 1) % sample_every == 0 or it == steps - 1:
            history["step"].append(it + 1)
            history["mass_rel_drift"].append((m - m0) / m0)
            history["l1"].append(l1_shape_error(phi, phi0))
            history["overlap"].append(shape_overlap(phi, phi0))
            history["moved_fraction"].append(info.get("moved_fraction", 0.0))
        if verbose and (it + 1) % max(1, steps // 10) == 0:
            print(
                f"  [{name} n={n} {scheme}] {it + 1}/{steps} "
                f"mass {(m - m0) / m0:+.2e} l1 {l1_shape_error(phi, phi0):.3e} "
                f"fill [{float(phi.min()):+.3f},{float(phi.max()):.3f}]"
            )

    metrics = lbm_case_metrics(phi, phi0, dim=dim, history=history)
    metrics.update(
        {
            "scheme": scheme,
            "tau": tau,
            "dim": dim,
            "steps": steps,
            "u_max_lat": u_max_lat,
            "omega_lat": omega,
            "r_max": r_max,
            "max_stepwise_mass_drift": max_step_drift,
            "mean_moved_fraction_per_step": moved_sum / steps,
            "div_u_max": div_u,
            "cylindrical_slot": cylindrical_slot,
            "extruded": extruded,
            "dtype": str(dtype).replace("torch.", ""),
        }
    )
    if record_fields:
        metrics["phi_final"] = phi.cpu()
        metrics["phi_initial"] = phi0.cpu()
    return metrics