"""Shared D2Q9 kernels for the gas-layer campaign.

These are the *exact kernels* used by both the Laplace gate and the gas-layer
channel, so that the interface physics verified by the Laplace gate is the same
code that runs in the channel.  They are thin, explicit implementations of

* the Shan-Chen pseudopotential force in divergence (pressure-tensor) form,
  evaluated on a rectangular grid with either periodic (``wrap``) or open
  (``replicate``) streamwise padding;
* BGK collision with Guo (2002) forcing, mass-conserving source term;
* streaming with half-way bounce-back walls, returning the momentum exchange
  into the walls and (in the open case) the mass/momentum leaving the outlet so
  that the mass ledger closes by construction.

Nothing here is model-specific beyond Shan-Chen + D2Q9, and the routines are
pure (no hidden state) so the tests can check conservation identities directly.
"""

from __future__ import annotations

from typing import Callable

import torch

from ..d2q9 import OPPOSITE, C, W

CS2 = 1.0 / 3.0

_C = C
#: Lattice weights renormalised in float64.  The tabulated ones are float32, and
#: their float64 sum is 1 + O(1e-8); using them as-is leaks ~1e-8 of the mass per
#: collision step, which is fatal for a 1e-6 mass-drift gate.  Renormalising
#: makes ``sum_i w_i = 1`` exact to machine precision.
_W = W.to(torch.float64)
_W = _W / _W.sum()
_OPP = OPPOSITE

#: Directions with positive streamwise component (leave through an open outlet).
_OUT_X = [i for i in range(9) if int(_C[i, 0]) > 0]
_DOWN = [i for i in range(9) if int(_C[i, 1]) < 0]
_UP = [i for i in range(9) if int(_C[i, 1]) > 0]


def _pad_neighbours(field: torch.Tensor, periodic_x: bool) -> torch.Tensor:
    """One-cell halo: ``wrap`` (periodic) or ``replicate`` (open) in x, replicate in y."""
    if periodic_x:
        out = torch.cat([field[:, -1:], field, field[:, :1]], dim=1)
    else:
        out = torch.cat([field[:, :1], field, field[:, -1:]], dim=1)
    return torch.cat([out[:1, :], out, out[-1:, :]], dim=0)


def neighbour_pull(field: torch.Tensor, periodic_x: bool = True) -> list[torch.Tensor]:
    """Return ``[field(x - c_i)]`` for the nine D2Q9 directions."""
    ny, nx = field.shape
    padded = _pad_neighbours(field, periodic_x)
    out = []
    for i in range(9):
        cx, cy = int(_C[i, 0]), int(_C[i, 1])
        out.append(padded[1 - cy : 1 - cy + ny, 1 - cx : 1 - cx + nx])
    return out


def sc_force(
    rho: torch.Tensor,
    psi: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
    G: float = 1.0,
    gx: float = 0.0,
    gy: float = 0.0,
    solid: torch.Tensor | None = None,
    wall_psi: float | None = None,
    periodic_x: bool = True,
    form: str = "standard",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shan-Chen interaction force.

    ``form="standard"`` (default, the form validated in this repository for the
    EOS pseudopotentials): ``F = -G psi(x) sum_i w_i c_i psi(x - c_i) + rho g``.

    ``form="pressure_tensor"``: the divergence form
    ``F = -(G/2) sum_i w_i c_i psi^2(x - c_i) + rho g`` (Ramshaw-Phathanapirom);
    exactly momentum conserving on a periodic box, but *not* interchangeable
    with the standard form at high density contrast (it changes the effective
    coupling), and it was found unstable for the present equation of state.

    ``G > 0`` is attractive in this library's backward-gather convention.
    """
    if psi is None or G == 0.0:
        return rho * gx, rho * gy
    psi_field = psi(rho) if callable(psi) else psi
    if solid is not None and wall_psi is not None:
        psi_field = torch.where(solid, torch.full_like(psi_field, float(wall_psi)), psi_field)
    if form == "standard":
        pulls = neighbour_pull(psi_field, periodic_x=periodic_x)
        sum_x = torch.zeros_like(rho)
        sum_y = torch.zeros_like(rho)
        for i in range(9):
            sum_x = sum_x + float(_W[i]) * float(_C[i, 0]) * pulls[i]
            sum_y = sum_y + float(_W[i]) * float(_C[i, 1]) * pulls[i]
        fx = -G * psi_field * sum_x + rho * gx
        fy = -G * psi_field * sum_y + rho * gy
    elif form == "pressure_tensor":
        pulls = neighbour_pull(psi_field * psi_field, periodic_x=periodic_x)
        sumsq_x = torch.zeros_like(rho)
        sumsq_y = torch.zeros_like(rho)
        for i in range(9):
            sumsq_x = sumsq_x + float(_W[i]) * float(_C[i, 0]) * pulls[i]
            sumsq_y = sumsq_y + float(_W[i]) * float(_C[i, 1]) * pulls[i]
        fx = -0.5 * G * sumsq_x + rho * gx
        fy = -0.5 * G * sumsq_y + rho * gy
    else:
        raise ValueError(f"unknown force form {form!r}")
    return fx, fy


def macroscopic(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lattice density and velocity moment of ``f`` (9, ny, nx)."""
    rho = f.sum(dim=0)
    ux = (f * _C[:, 0].view(9, 1, 1)).sum(dim=0) / rho
    uy = (f * _C[:, 1].view(9, 1, 1)).sum(dim=0) / rho
    return rho, ux, uy


def equilibrium(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """D2Q9 equilibrium, stacked as (9, ny, nx)."""
    cu = _C[:, 0].view(9, 1, 1) * ux + _C[:, 1].view(9, 1, 1) * uy
    usq = ux * ux + uy * uy
    return (
        _W.view(9, 1, 1)
        * rho
        * (1.0 + cu / CS2 + 0.5 * cu * cu / (CS2 * CS2) - 0.5 * usq / CS2)
    )


def _guo_source(fx: torch.Tensor, fy: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor,
                tau: float) -> torch.Tensor:
    cf = _C[:, 0].view(9, 1, 1) * fx + _C[:, 1].view(9, 1, 1) * fy
    cu = _C[:, 0].view(9, 1, 1) * ux + _C[:, 1].view(9, 1, 1) * uy
    uf = ux * fx + uy * fy
    return _W.view(9, 1, 1) * (1.0 - 0.5 / tau) * (
        (cf - uf.unsqueeze(0)) / CS2 + cu * cf / CS2**2
    )


def collide_scp(
    f: torch.Tensor,
    tau: float,
    psi: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
    G: float = 1.0,
    gx: float = 0.0,
    gy: float = 0.0,
    solid: torch.Tensor | None = None,
    wall_psi: float | None = None,
    periodic_x: bool = True,
    forcing: str = "edm",
    form: str = "standard",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BGK collision + Shan-Chen force.

    ``forcing="edm"`` (default) is the exact-difference method
    (``feq(rho, u + F/rho) - feq(rho, u)`` added to the BGK step): momentum exact,
    mass conserving, and the scheme that stays stable at the high density
    contrast used here.  ``forcing="guo"`` is the Guo (2002) source-term form,
    kept for comparison.

    Returns ``(f_post, fx, fy)``; solid nodes are left at zero.
    """
    rho, ux, uy = macroscopic(f)
    fx, fy = sc_force(rho, psi, G, gx, gy, solid, wall_psi, periodic_x, form=form)
    rho_s = torch.clamp(rho, min=1e-12)
    if forcing == "edm":
        feq_u = equilibrium(rho, ux, uy)
        feq_f = equilibrium(rho, ux + fx / rho_s, uy + fy / rho_s)
        f_out = f - (f - feq_u) / tau + (feq_f - feq_u)
    elif forcing == "guo":
        uxp = ux + 0.5 * fx / rho_s
        uyp = uy + 0.5 * fy / rho_s
        feq = equilibrium(rho, uxp, uyp)
        f_out = f - (f - feq) / tau + _guo_source(fx, fy, uxp, uyp, tau)
    else:
        raise ValueError(f"unknown forcing {forcing!r}")
    if solid is not None:
        f_out = torch.where(solid.unsqueeze(0), torch.zeros_like(f), f_out)
    return f_out, fx, fy


def collide_bgk(
    f: torch.Tensor,
    tau: float,
    gx: float = 0.0,
    gy: float = 0.0,
) -> torch.Tensor:
    """Single-phase BGK + Guo body force (used for the 1ph reference run)."""
    rho, ux, uy = macroscopic(f)
    uxp = ux + 0.5 * gx  # u_phys = sum(c f)/rho + F/(2 rho)
    uyp = uy + 0.5 * gy
    feq = equilibrium(rho, uxp, uyp)
    fx = rho * gx
    fy = rho * gy
    return f - (f - feq) / tau + _guo_source(fx, fy, uxp, uyp, tau)


def _shift_x_push(field: torch.Tensor, cx: int, periodic: bool) -> torch.Tensor:
    """Push ``field`` by ``cx`` cells along x (``out[x] = field[x - cx]``)."""
    if cx == 0:
        return field
    out = torch.zeros_like(field)
    nx = field.shape[1]
    if periodic:
        return torch.roll(field, shifts=cx, dims=1)
    if cx > 0:
        out[:, cx:] = field[:, : nx - cx]
    else:
        out[:, :cx] = field[:, -cx:]
    return out


def stream(
    f_post: torch.Tensor,
    solid: torch.Tensor | None = None,
    periodic_x: bool = True,
    periodic_y: bool = False,
) -> dict[str, object]:
    """Push-stream with half-way bounce-back on the top/bottom solid rows.

    Fluid rows are ``1 .. ny-2``; rows ``0`` and ``ny-1`` are solid.  Each
    population is pushed by one lattice vector, and every population whose
    target falls in a solid row is reflected into the opposite direction at the
    source node (exact half-way bounce-back, mass conserving).

    Returns a dict with

    ``f``             the streamed populations,
    ``wall_force_x``  ``(2, nx)`` streamwise momentum transfer into the bottom
                      (index 0) and top (index 1) walls (Ladd momentum exchange),
    ``mass_out``      mass that left through the open outlet (0 when periodic),
    ``momentum_out``  streamwise momentum that left through the outlet.
    """
    ny, nx = f_post.shape[1], f_post.shape[2]
    if periodic_y:
        # fully periodic box (bulk two-phase diagnostics, no walls)
        f_new = torch.zeros_like(f_post)
        for i in range(9):
            cy, cx = int(_C[i, 1]), int(_C[i, 0])
            out = f_post[i]
            if cy or cx:
                out = torch.roll(out, shifts=(cy, cx), dims=(0, 1))
            f_new[i] = out
        zero = torch.zeros(2, nx, dtype=f_post.dtype, device=f_post.device)
        zero_out = torch.zeros((), dtype=f_post.dtype, device=f_post.device)
        return {
            "f": f_new,
            "wall_force_x": zero,
            "mass_out": zero_out,
            "momentum_out": zero_out,
        }
    f_new = torch.zeros_like(f_post)
    f_new[0] = f_post[0]
    wall_force = torch.zeros(2, nx, dtype=f_post.dtype, device=f_post.device)
    for i in range(1, 9):
        cx, cy = int(_C[i, 0]), int(_C[i, 1])
        src = f_post[i]
        j = int(_OPP[i])
        if cy == 0:
            f_new[i] += _shift_x_push(src, cx, periodic_x)
        elif cy == 1:
            # sources y = 1..ny-3 flow to y = 2..ny-2; y = ny-2 hits the wall
            moved = _shift_x_push(src[1 : ny - 2, :], cx, periodic_x)
            f_new[i][2 : ny - 1, :] += moved
            f_new[j][ny - 2, :] += src[ny - 2, :]
        else:
            # sources y = 2..ny-2 flow to y = 1..ny-3; y = 1 hits the wall
            moved = _shift_x_push(src[2 : ny - 1, :], cx, periodic_x)
            f_new[i][1 : ny - 2, :] += moved
            f_new[j][1, :] += src[1, :]
    for i in _DOWN:
        wall_force[0] += 2.0 * float(_C[i, 0]) * f_post[i][1, :]
    for i in _UP:
        wall_force[1] += 2.0 * float(_C[i, 0]) * f_post[i][ny - 2, :]
    mass_out = torch.zeros((), dtype=f_post.dtype, device=f_post.device)
    momentum_out = torch.zeros((), dtype=f_post.dtype, device=f_post.device)
    if not periodic_x:
        # Mass *flux* through the outlet plane: sum_i c_ix f_i.  This is the
        # advective mass flow rho u_x (not the gross population crossing budget),
        # so the ledger dM = inflow - outflow + injection closes.
        for i in _OUT_X:
            cx = int(_C[i, 0])
            plane = f_post[i][1 : ny - 1, nx - cx :]
            mass_out = mass_out + float(cx) * plane.sum()
            momentum_out = momentum_out + float(cx * cx) * plane.sum()
    if solid is not None:
        f_new = torch.where(solid.unsqueeze(0), torch.zeros_like(f_new), f_new)
    return {
        "f": f_new,
        "wall_force_x": wall_force,
        "mass_out": mass_out,
        "momentum_out": momentum_out,
    }


def mechanical_pressure(
    rho: torch.Tensor,
    psi: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
    G: float = 1.0,
) -> torch.Tensor:
    """Shan-Chen mechanical pressure ``p = rho cs^2 - (G/2) cs^2 psi^2``."""
    psi_field = psi(rho) if callable(psi) else psi
    return rho * CS2 - 0.5 * G * CS2 * psi_field * psi_field