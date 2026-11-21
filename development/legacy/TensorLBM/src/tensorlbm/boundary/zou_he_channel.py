"""Zou-He (1997) inlet/outlet reconstruction for open plane channels.

The OSS reference ``loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow``
closes a channel with a *Zou-He reconstruction* at both ends: a velocity inlet
where the prescribed velocity fixes the unknown incoming populations and a
pressure (density) outlet where the prescribed density does.  The in-repo
``boundary.solver2d.zou_he_moving_lid`` only covers the moving-lid (north wall)
case, so an open channel could not be closed without re-deriving the algebra
inline.  This module gives the four walls, both closure modes, as a single
vectorised torch implementation usable as a boundary condition in any D2Q9
solver that stores populations as ``f[direction, y, x]``.

Derivation (D2Q9, BGK, lattice units)
-------------------------------------
For a wall on the *west* side the unknown populations are the ones entering the
domain, ``f1, f5, f7`` (``c_x = +1``); ``f0, f2, f3, f4, f6, f8`` are known
after streaming.  Mass and normal momentum combine into

    rho (1 - ux) = f0 + f3 + f4 + 2 (f2 + f6 + f8)          (west)

which yields ``rho`` when ``ux`` is prescribed (velocity mode) or ``ux`` when
``rho`` is prescribed (pressure mode).  The transverse momentum equation

    rho uy = (f3 - f4) + (f5 + f6) - (f7 + f8)

supplies the *difference* of the two unknown diagonals, mass supplies their
*sum*, and the normal population is closed by bounce-back of the
non-equilibrium part along the normal link (``f1 - f2 = 2 w1 rho 3 ux``), i.e.

    f1 = f2 + (2/3) rho ux
    f5 = f8 + (1/6) rho ux - 0.5 (f3 - f4) + 0.5 rho uy
    f7 = f6 + (1/6) rho ux + 0.5 (f3 - f4) - 0.5 rho uy

Note that each unknown diagonal is rebuilt from *its own opposite* known
population (west: ``f5`` from ``f8`` and ``f7`` from ``f6``; east: ``f6`` from
``f7`` and ``f8`` from ``f5``).  Pairing them the other way round -- an easy
slip, it is what the D2Q9 index table suggests if one forgets that ``5<->8`` and
``6<->7`` are the actual opposite pairs -- leaves the reconstructed *density*
exactly right while silently corrupting the transverse momentum.  That is why
``tests/test_zou_he_channel.py::test_moment_exactness_*`` asserts the moments of
the reconstructed layer instead of only checking that the code runs; the slip
was caught by exactly that test during development.

The east/north/south closures are the mirror images of the above (mirror
``x -> -x`` swaps ``1 <-> 2``, ``5 <-> 6``, ``7 <-> 8`` and flips ``ux``;
mirror ``y -> -y`` swaps ``3 <-> 4``, ``5 <-> 7``, ``6 <-> 8`` and flips ``uy``).

Mass conservation
-----------------
Velocity mode is exactly mass conserving (the inflow is imposed and equals the
reconstructed momentum).  Pressure mode leaves one degree of freedom -- the
transverse velocity at the boundary -- which is *assumed* here (default 0); the
resulting mass defect is O(u^2) and is reported by the verification case as a
diagnostic, not silently ignored.
"""

from __future__ import annotations

from typing import Optional

import torch

from .d2q9 import C2Q9, W2Q9, OPPOSITE_2Q9, CS2_2Q9  # noqa: F401  (re-exported)

__all__ = [
    "ZOU_HE_SIDES",
    "zou_he_wall",
    "zou_he_channel",
    "zou_he_column_density_momentum",
]

ZOU_HE_SIDES = ("west", "east", "north", "south")


def _line(f: torch.Tensor, side: str) -> torch.Tensor:
    """View of the boundary layer, shape (9, N) -- writing through it updates f."""
    if side == "west":
        return f[:, :, 0]
    if side == "east":
        return f[:, :, -1]
    if side == "north":
        return f[:, -1, :]
    if side == "south":
        return f[:, 0, :]
    raise ValueError(f"unknown side {side!r}; expected one of {ZOU_HE_SIDES}")


def _as_line(value, ref: torch.Tensor) -> torch.Tensor:
    """Broadcast a scalar / array BC value to the boundary line of ``ref``."""
    if isinstance(value, torch.Tensor):
        return value.to(device=ref.device, dtype=ref.dtype)
    return torch.full_like(ref, float(value))


def zou_he_wall(
    f: torch.Tensor,
    side: str,
    *,
    mode: str = "velocity",
    ux=None,
    uy=None,
    rho=None,
) -> torch.Tensor:
    """Reconstruct the unknown populations on one Zou-He boundary layer.

    Parameters
    ----------
    f : torch.Tensor, shape (9, ny, nx)
        Populations *after* streaming at the boundary layer.  Modified in place
        and returned.
    side : {"west", "east", "north", "south"}
    mode : {"velocity", "pressure"}
        ``"velocity"``: ``ux``/``uy`` prescribed, density reconstructed.
        ``"pressure"``: ``rho`` prescribed, normal velocity reconstructed; the
        transverse velocity is taken from ``ux``/``uy`` (default 0).
    ux, uy : float or tensor, optional
        Velocity in lattice units.  Tensors must broadcast to the boundary line
        length (``ny`` for west/east, ``nx`` for north/south), which is how a
        parabolic inlet profile is imposed.
    rho : float or tensor, optional
        Density (lattice units) for ``mode="pressure"``.

    Returns
    -------
    f : torch.Tensor
        Same object as the input, with the unknown populations overwritten.
    """
    if side not in ZOU_HE_SIDES:
        raise ValueError(f"unknown side {side!r}; expected one of {ZOU_HE_SIDES}")
    if mode not in ("velocity", "pressure"):
        raise ValueError(f"unknown mode {mode!r}; expected 'velocity' or 'pressure'")
    if f.shape[0] != 9:
        raise ValueError(f"expected 9 populations on axis 0, got {f.shape[0]}")
    if mode == "velocity":
        # The *normal* component is what the reconstruction solves for; the
        # transverse one defaults to 0.  Without this guard a caller who omits
        # the normal component gets a TypeError from arithmetic on ``None``
        # several lines below instead of a contract error.
        normal = "ux" if side in ("west", "east") else "uy"
        if (ux if normal == "ux" else uy) is None:
            raise ValueError(
                f"velocity mode on side {side!r} needs the normal component "
                f"{normal!r} (the transverse one defaults to 0)"
            )
    if mode == "pressure" and rho is None:
        raise ValueError("pressure mode needs rho")

    line = _line(f, side)
    zero = torch.zeros_like(line[0])
    ux_l = _as_line(ux, zero) if ux is not None else None
    uy_l = _as_line(uy, zero) if uy is not None else None

    f0, f1, f2, f3, f4, f5, f6, f7, f8 = (line[i] for i in range(9))

    if side == "west":
        rho_in = f0 + f3 + f4 + 2.0 * (f2 + f6 + f8)
        if mode == "velocity":
            rho_l = rho_in / (1.0 - ux_l)
            ux_l = _as_line(ux, zero)
            uy_l = _as_line(uy, zero) if uy is not None else zero
        else:
            rho_l = _as_line(rho, zero)
            ux_l = 1.0 - rho_in / rho_l
            uy_l = uy_l if uy is not None else zero
        line[1] = f2 + (2.0 / 3.0) * rho_l * ux_l
        line[5] = f8 + rho_l * ux_l / 6.0 - 0.5 * (f3 - f4) + 0.5 * rho_l * uy_l
        line[7] = f6 + rho_l * ux_l / 6.0 + 0.5 * (f3 - f4) - 0.5 * rho_l * uy_l

    elif side == "east":
        rho_in = f0 + f3 + f4 + 2.0 * (f1 + f5 + f7)
        if mode == "velocity":
            rho_l = rho_in / (1.0 + ux_l)
            ux_l = _as_line(ux, zero)
            uy_l = _as_line(uy, zero) if uy is not None else zero
        else:
            rho_l = _as_line(rho, zero)
            ux_l = rho_in / rho_l - 1.0
            uy_l = uy_l if uy is not None else zero
        line[2] = f1 - (2.0 / 3.0) * rho_l * ux_l
        line[6] = f7 - rho_l * ux_l / 6.0 - 0.5 * (f3 - f4) + 0.5 * rho_l * uy_l
        line[8] = f5 - rho_l * ux_l / 6.0 + 0.5 * (f3 - f4) - 0.5 * rho_l * uy_l

    elif side == "north":
        rho_in = f0 + f1 + f2 + 2.0 * (f3 + f5 + f6)
        if mode == "velocity":
            rho_l = rho_in / (1.0 + uy_l)
            uy_l = _as_line(uy, zero)
            ux_l = _as_line(ux, zero) if ux is not None else zero
        else:
            rho_l = _as_line(rho, zero)
            uy_l = rho_in / rho_l - 1.0
            ux_l = ux_l if ux is not None else zero
        line[4] = f3 - (2.0 / 3.0) * rho_l * uy_l
        line[7] = f6 - rho_l * uy_l / 6.0 + 0.5 * rho_l * ux_l - 0.5 * (f1 - f2)
        line[8] = f5 - rho_l * uy_l / 6.0 - 0.5 * rho_l * ux_l + 0.5 * (f1 - f2)

    else:  # south
        rho_in = f0 + f1 + f2 + 2.0 * (f4 + f7 + f8)
        if mode == "velocity":
            rho_l = rho_in / (1.0 - uy_l)
            uy_l = _as_line(uy, zero)
            ux_l = _as_line(ux, zero) if ux is not None else zero
        else:
            rho_l = _as_line(rho, zero)
            uy_l = 1.0 - rho_in / rho_l
            ux_l = ux_l if ux is not None else zero
        line[3] = f4 + (2.0 / 3.0) * rho_l * uy_l
        line[5] = f8 + rho_l * uy_l / 6.0 + 0.5 * rho_l * ux_l - 0.5 * (f1 - f2)
        line[6] = f7 + rho_l * uy_l / 6.0 - 0.5 * rho_l * ux_l + 0.5 * (f1 - f2)

    return f


def zou_he_channel(
    f: torch.Tensor,
    *,
    inlet: str = "velocity",
    outlet: str = "pressure",
    inlet_ux=None,
    inlet_uy=None,
    inlet_rho=None,
    outlet_ux=None,
    outlet_uy=None,
    outlet_rho=None,
) -> torch.Tensor:
    """Close a streamwise channel: Zou-He reconstruction at ``x=0`` and ``x=-1``.

    This is the OSS ``zou-he-channel-inlet-outlet-reconstruction`` closure: by
    default a *velocity* inlet (prescribed ``inlet_ux``, possibly a profile) and
    a *pressure* outlet (prescribed ``outlet_rho``), i.e. the configuration for
    which ``dp/dx`` is set by the density and the flow rate results from the
    momentum balance.  Passing both sides as ``"pressure"`` gives the fully
    pressure-driven channel used by the verification case.

    Returns ``f`` (modified in place).
    """
    zou_he_wall(
        f, "west", mode=inlet, ux=inlet_ux, uy=inlet_uy, rho=inlet_rho
    )
    zou_he_wall(
        f, "east", mode=outlet, ux=outlet_ux, uy=outlet_uy, rho=outlet_rho
    )
    return f


def zou_he_column_density_momentum(f: torch.Tensor, side: str):
    """Macroscopic density and velocity of one boundary layer.

    Returns ``(rho, ux, uy)`` from the populations *in place* (no reconstruction),
    so a test can check what a boundary condition actually produced.
    """
    line = _line(f, side)
    c = C2Q9.to(device=f.device)
    rho = line.sum(0)
    ux = (line * c[:, 0:1].to(f.dtype)).sum(0) / rho
    uy = (line * c[:, 1:2].to(f.dtype)).sum(0) / rho
    return rho, ux, uy