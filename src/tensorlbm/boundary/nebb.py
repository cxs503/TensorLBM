"""Force-corrected NEBB walls + body-force driver (D2Q9 / D3Q19 channel flows).

What this module is
-------------------
A *wall-node* no-slip boundary condition ("NEBB" = non-equilibrium
bounce-back, a.k.a. Zou/He-style reconstruction of the non-equilibrium part)
combined with a body-force driver, in the form that makes the pair mutually
consistent: the body force is applied with the Guo scheme (equilibrium at the
force-corrected velocity ``u* = u + F/(2 rho)`` plus an explicit source term),
so the *wall* equilibrium must be evaluated at the same force-corrected wall
velocity ``u_w - F/(2 rho_w)``; otherwise the reconstructed wall layer carries
an O(F) slip that contaminates the mean profile.

Lineage (read from source, no hand-copied tables)
------------------------------------------------
* ``haimingz/vivsim`` — ``vivsim/lbm/boundary/nebb.py`` (``boundary_nebb``,
  shear-reconstruction form), ``vivsim/lbm/boundary/_helpers.py``
  (``get_corrected_wall_velocity``, ``get_rho_wall_from_velocity``),
  ``vivsim/lbm/forcing/{edm,guo}.py``, ``vivsim/lbm/basic.py``
  (``get_velocity_correction = F/(2 rho)``).
  The implemented forms are the standard Guo (2002) forcing + the standard
  non-equilibrium bounce-back; the vivsim port of the same algorithm lives in
  ``xval/xval_nebb_bodyforce_driver.py`` (cross-benchmark arm A) and is used as
  an independent implementation arm by the benchmark entry point.
* TensorLBM reuse: lattice constants from :mod:`tensorlbm.boundary.d2q9`
  (``C2Q9``/``W2Q9``/``OPPOSITE_2Q9``) and the solver primitives from
  :mod:`tensorlbm.boundary.solver2d` (``equilibrium2d``/``macroscopic2d``/
  ``stream2d``).  The wall direction sets are *derived from the lattice
  vectors* (sign of the wall-normal component), never from hard-coded index
  tables — the class of error recorded after a hand-transcribed sign mistake
  in :mod:`tensorlbm.triton_fused`.

Conventions
-----------
``f`` has shape ``(q, ..., ny, nx)``; the wall normal is the ``ny`` axis.
Wall nodes are the first (``row=0``, outward normal ``-y``) and last
(``row=ny-1``, outward normal ``+y``) planes, so the no-slip planes coincide
with the wall-node planes and the channel height is ``H = ny - 1`` in lattice
units (the convention of the closed-form Poiseuille solution
``ux(y) = Gx/(2 nu) y (H - y)``).  Fluid rows are ``1 .. ny-2``.

Public API
----------
``wall_directions(C, normal_sign)``      : (in_dirs, out_dirs, tan_dirs)
``wall_density(f, row, dirs)``           : rho_w from the known populations
``force_corrected_velocity(u, force, rho)`` : ``u - F/(2 rho)`` (wall) and
                                             ``u + F/(2 rho)`` (bulk)
``nebb_wall_plane(...)``                 : reconstruct one wall plane in place
``guo_source(...)`` / ``collide_guo(...)`` / ``collide_edm(...)``
``NebbPoiseuilleChannel``                : body-force-driven channel driver
                                           (input -> compute -> output)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from .d2q9 import C2Q9, OPPOSITE_2Q9, W2Q9
from .solver2d import equilibrium2d, macroscopic2d, stream2d

__all__ = [
    "wall_directions",
    "wall_density",
    "force_corrected_velocity",
    "nebb_wall_plane",
    "guo_source",
    "collide_guo",
    "collide_edm",
    "NebbChannelConfig",
    "NebbPoiseuilleChannel",
    "analytic_poiseuille",
]

# --------------------------------------------------------------------------- #
# Lattice-derived wall direction sets
# --------------------------------------------------------------------------- #
def wall_directions(
    C: torch.Tensor, normal_sign: int, axis: int = 1
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Directions for a wall whose *outward* normal is ``normal_sign * e_axis``.

    Parameters
    ----------
    C : (q, dim) integer lattice velocities (dim 2 for D2Q9, 3 for D3Q19).
    normal_sign : ``+1`` for the top wall (outward ``+y``), ``-1`` for the
        bottom wall (outward ``-y``).
    axis : index of the wall-normal lattice component (default ``1`` = y).

    Returns
    -------
    in_dirs : directions **reconstructed by the BC** — the populations that
        arrive from outside the domain after streaming.  Note the sign: the
        repo's :func:`tensorlbm.boundary.solver2d.stream2d` advects
        ``f_k`` along ``-c_k`` (``torch.roll(shifts=-c)``), so at a wall with
        outward normal ``normal_sign * e_axis`` the outside-borne populations
        are those with ``c_axis * normal_sign > 0``.  Verified by a one-hot
        streaming probe in ``tests/test_nebb_bodyforce_force_corrected.py``.
    out_dirs : the opposite (known) directions — used by :func:`wall_density`.
    tan_dirs : wall-tangential directions (zero normal component, non-rest).
    """
    if normal_sign not in (-1, 1):
        raise ValueError(f"normal_sign must be +-1, got {normal_sign!r}")
    comp = C[:, axis].tolist()
    other = [i for i in range(C.shape[1]) if i != axis]
    in_dirs = tuple(i for i, c in enumerate(comp) if c * normal_sign > 0)
    out_dirs = tuple(i for i, c in enumerate(comp) if c * normal_sign < 0)
    tan_dirs = tuple(
        i
        for i, c in enumerate(comp)
        if c == 0 and any(int(C[i, j]) != 0 for j in other)
    )
    if not in_dirs or len(in_dirs) != len(out_dirs):
        raise ValueError("degenerate lattice for the requested wall normal")
    return in_dirs, out_dirs, tan_dirs


# --------------------------------------------------------------------------- #
# Wall density and force-corrected wall velocity
# --------------------------------------------------------------------------- #
def wall_density(
    f: torch.Tensor,
    row: int,
    dirs: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    un: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    """Density at a wall node from the locally known populations.

    ``rho_w = (sum_known + 2 sum_out) / (1 - u_n)`` with ``sum_known`` over the
    rest + tangential populations and ``u_n`` the wall-normal velocity
    (positive *into* the fluid).  For the no-slip body-force channel
    ``u_n = 0`` and this reduces to the vivsim ``_get_rho_wall_numerator``.
    """
    in_dirs, out_dirs, tan_dirs = dirs
    sel = lambda idx: f[idx, ..., row, :]  # noqa: E731 - (..., nx) slices
    known = f[0, ..., row, :].clone()
    for i in tan_dirs:
        known = known + sel(i)
    num = known
    for i in out_dirs:
        num = num + 2.0 * sel(i)
    denom = 1.0 - torch.as_tensor(un, dtype=num.dtype)
    return num / denom


def force_corrected_velocity(
    u: torch.Tensor, force: torch.Tensor | float, rho: torch.Tensor, sign: float = -1.0
) -> torch.Tensor:
    """``u + sign * F/(2 rho)`` — the force-corrected velocity.

    ``sign=-1`` gives the *wall* velocity used inside the BC equilibrium
    (``u_wall - F/(2 rho_w)``, vivsim ``get_corrected_wall_velocity``);
    ``sign=+1`` gives the physical bulk velocity
    (``u + F/(2 rho)``, vivsim ``get_velocity_correction``).
    """
    return u + sign * torch.as_tensor(force, dtype=u.dtype, device=u.device) / (
        2.0 * rho
    )


# --------------------------------------------------------------------------- #
# NEBB reconstruction of one wall plane
# --------------------------------------------------------------------------- #
def nebb_wall_plane(
    f: torch.Tensor,
    row: int,
    normal_sign: int,
    rho_w: torch.Tensor,
    ux_w: torch.Tensor,
    uy_w: torch.Tensor,
    C: torch.Tensor | None = None,
    W: torch.Tensor | None = None,
    OPP: torch.Tensor | None = None,
) -> torch.Tensor:
    """Non-equilibrium bounce-back on one wall plane, in place.

    ``f[k] = feq[k](rho_w, u_w) - feq[opp k](rho_w, u_w) + f[opp k]`` for every
    ``k`` pointing into the fluid, i.e. the equilibrium part is set by the wall
    state and the non-equilibrium part is mirrored from the opposite
    (known) population.  Works for any lattice passed through ``C/W/OPP``.
    """
    if C is None:
        C, W, OPP = C2Q9, W2Q9, OPPOSITE_2Q9
    in_dirs, _out_dirs, _tan = wall_directions(C, normal_sign)

    w = W.to(dtype=rho_w.dtype, device=rho_w.device)
    c = C.to(dtype=rho_w.dtype, device=rho_w.device)
    f_eq = equilibrium_lattice(rho_w, ux_w, uy_w, c, w)
    for k in in_dirs:
        kk = int(OPP[k])
        f[k, ..., row, :] = f_eq[k] - f_eq[kk] + f[kk, ..., row, :]
    return f


def equilibrium_lattice(
    rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor, c: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    """Second-order equilibrium for an arbitrary 2-D lattice (q, ..., ny, nx)."""
    shape = (c.shape[0],) + (1,) * rho.dim()
    cx = c[:, 0].view(shape)
    cy = c[:, 1].view(shape)
    cc = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0)
    usq = (ux * ux + uy * uy).unsqueeze(0)
    return w.view(shape) * rho.unsqueeze(0) * (1.0 + 3.0 * cc + 4.5 * cc * cc - 1.5 * usq)


# --------------------------------------------------------------------------- #
# Body-force driver: Guo (explicit source) and EDM (equilibrium-shift) schemes
# --------------------------------------------------------------------------- #
def guo_source(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    gx: float,
    gy: float,
    omega: float,
    C: torch.Tensor | None = None,
    W: torch.Tensor | None = None,
) -> torch.Tensor:
    """Guo (2002) discrete force term ``(1 - 1/(2 tau)) S_i`` at the *actual* u."""
    if C is None:
        C, W = C2Q9, W2Q9
    c = C.to(dtype=rho.dtype, device=rho.device)
    w = W.to(dtype=rho.dtype, device=rho.device)
    shape = (c.shape[0],) + (1,) * rho.dim()
    fx = gx * rho
    fy = gy * rho
    cx = c[:, 0].view(shape)
    cy = c[:, 1].view(shape)
    cf = cx * fx.unsqueeze(0) + cy * fy.unsqueeze(0)
    cu = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0)
    uf = (ux * fx + uy * fy).unsqueeze(0)
    return (1.0 - 0.5 * omega) * w.view(shape) * (3.0 * cf + 9.0 * cu * cf - 3.0 * uf)


def collide_guo(
    f: torch.Tensor,
    tau: float,
    gx: float = 0.0,
    gy: float = 0.0,
    C: torch.Tensor | None = None,
    W: torch.Tensor | None = None,
) -> torch.Tensor:
    """BGK collision + Guo body force; equilibrium at ``u* = u + F/(2 rho)``."""
    if C is None:
        C, W = C2Q9, W2Q9
    omega = 1.0 / tau
    rho, ux, uy = macroscopic2d(f)
    inv_rho = 1.0 / rho.clamp_min(1e-30)
    uxs = ux + 0.5 * gx * rho * inv_rho
    uys = uy + 0.5 * gy * rho * inv_rho
    feq = equilibrium2d(rho, uxs, uys)
    f_post = f - omega * (f - feq)
    if gx != 0.0 or gy != 0.0:
        f_post = f_post + guo_source(rho, ux, uy, gx, gy, omega, C=C, W=W)
    return f_post


def collide_edm(
    f: torch.Tensor,
    tau: float,
    gx: float = 0.0,
    gy: float = 0.0,
    C: torch.Tensor | None = None,
    W: torch.Tensor | None = None,
) -> torch.Tensor:
    """EDM-style collision: equilibrium evaluated at the corrected velocity only.

    No explicit source term is added — the whole force enters through
    ``feq(u + F/(2 rho))`` (vivsim ``forcing/edm.py``).  Provided so the two
    forcing families can be compared on identical walls.
    """
    omega = 1.0 / tau
    rho, ux, uy = macroscopic2d(f)
    inv_rho = 1.0 / rho.clamp_min(1e-30)
    feq = equilibrium2d(rho, ux + 0.5 * gx * rho * inv_rho, uy + 0.5 * gy * rho * inv_rho)
    return f - omega * (f - feq)


# --------------------------------------------------------------------------- #
# Analytic reference
# --------------------------------------------------------------------------- #
def analytic_poiseuille(ny: int, gx: float, nu: float, nx: int = 1) -> torch.Tensor:
    """Closed-form steady profile ``ux(y) = Gx/(2 nu) y (H - y)``, ``H = ny-1``.

    Returns shape ``(ny,)`` (broadcast against the ``ny``-axis profile).
    """
    H = float(ny - 1)
    y = torch.arange(ny, dtype=torch.float64)
    prof = gx / (2.0 * nu) * y * (H - y)
    return prof if nx == 1 else prof.unsqueeze(1).expand(ny, nx)


# --------------------------------------------------------------------------- #
# Channel driver
# --------------------------------------------------------------------------- #
@dataclass
class NebbChannelConfig:
    """Configuration of the body-force-driven D2Q9 channel."""

    nx: int = 4
    ny: int = 10
    nu: float = 0.2
    gx: float = 1.0e-3
    gy: float = 0.0
    steps: int = 200_000
    scheme: str = "guo"  # {"guo", "edm"}
    wall: str = "nebb"  # {"nebb", "halfway"}
    force_corrected_wall: bool = True
    steady_tol: float = 1.0e-13
    check_every: int = 50
    dtype: torch.dtype = field(default=torch.float64)

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5

    def validate(self) -> None:
        if self.ny < 4:
            raise ValueError("ny must be >= 4 (two wall rows + >= 2 fluid rows)")
        if self.nu <= 0.0:
            raise ValueError("nu must be positive")
        if self.scheme not in ("guo", "edm"):
            raise ValueError(f"scheme must be 'guo' or 'edm', got {self.scheme!r}")
        if self.wall not in ("nebb", "halfway"):
            raise ValueError(f"wall must be 'nebb' or 'halfway', got {self.wall!r}")


class NebbPoiseuilleChannel:
    """Body-force-driven 2-D channel: input -> compute -> output.

    One :meth:`step` is collide(body force) -> stream -> walls.  With
    ``wall="nebb"`` the two wall-node planes are rebuilt with
    :func:`nebb_wall_plane` using the force-corrected wall velocity
    (``-Gx/(2 rho_w)``) and the wall density recovered from the field; with
    ``wall="halfway"`` the classic half-way bounce-back is used instead
    (control arm).  ``force_corrected_wall=False`` drops the ``-Gx/(2 rho_w)``
    shift, i.e. the *pre-fix* formulation, as a negative control.
    """

    def __init__(self, cfg: NebbChannelConfig, device: str | torch.device = "cpu"):
        cfg.validate()
        self.cfg = cfg
        self.device = torch.device(device)
        self.C = C2Q9.to(self.device)
        self.W = W2Q9.to(dtype=cfg.dtype, device=self.device)
        self.OPP = OPPOSITE_2Q9.to(self.device)
        self._bot = wall_directions(self.C, -1)
        self._top = wall_directions(self.C, +1)
        rho = torch.ones((cfg.ny, cfg.nx), dtype=cfg.dtype, device=self.device)
        zero = torch.zeros_like(rho)
        self.f = equilibrium2d(rho, zero, zero)
        self.steps_done = 0
        self.residual = float("inf")

    # -- one time step ----------------------------------------------------- #
    def step(self) -> torch.Tensor:
        cfg = self.cfg
        collide = collide_guo if cfg.scheme == "guo" else collide_edm
        f_post = collide(self.f, cfg.tau, cfg.gx, cfg.gy, C=self.C, W=self.W)
        f_new = stream2d(f_post)
        if cfg.wall == "nebb":
            for row, sign, dirs in ((0, -1, self._bot), (cfg.ny - 1, +1, self._top)):
                rho_w = wall_density(f_new, row, dirs)
                uxw = torch.zeros_like(rho_w)
                uyw = torch.zeros_like(rho_w)
                if cfg.force_corrected_wall:
                    # u_wall - F/(2 rho_w); u_wall = 0 (no-slip)
                    uxw = force_corrected_velocity(
                        uxw, cfg.gx, rho_w, sign=-1.0
                    )
                    uyw = force_corrected_velocity(uyw, cfg.gy, rho_w, sign=-1.0)
                f_new = nebb_wall_plane(
                    f_new, row, sign, rho_w, uxw, uyw,
                    C=self.C, W=self.W, OPP=self.OPP,
                )
        else:  # half-way bounce-back on the fluid rows adjacent to the walls
            in_bot = self._bot[0]
            for k in in_bot:
                f_new[k, ..., 1, :] = f_post[int(self.OPP[k]), ..., 1, :]
            for k in self._top[0]:
                f_new[k, ..., cfg.ny - 2, :] = f_post[int(self.OPP[k]), ..., cfg.ny - 2, :]
        self.f = f_new
        self.steps_done += 1
        return self.f

    # -- fields ------------------------------------------------------------ #
    def velocity(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Physical (force-corrected) velocity and density."""
        rho, ux, uy = macroscopic2d(self.f)
        if self.cfg.scheme == "guo" or self.cfg.scheme == "edm":
            inv_rho = 1.0 / rho.clamp_min(1e-30)
            ux = ux + 0.5 * self.cfg.gx * inv_rho
            uy = uy + 0.5 * self.cfg.gy * inv_rho
            ux = torch.where(rho > 0, ux, torch.zeros_like(ux))
        return ux, uy, rho

    def run(self, steps: int | None = None) -> dict:
        """Advance to steady state (or ``steps``) and return the summary."""
        cfg = self.cfg
        n = cfg.steps if steps is None else steps
        prev = None
        for it in range(n):
            self.step()
            if prev is not None and (it % cfg.check_every == 0):
                ux, _uy, _rho = self.velocity()
                res = float((ux - prev).abs().max())
                self.residual = res
                if res < cfg.steady_tol:
                    break
                prev = ux
            elif prev is None:
                prev = self.velocity()[0]
        ux, uy, rho = self.velocity()
        prof = ux.mean(dim=1)  # y-profile averaged over the periodic x direction
        ref = analytic_poiseuille(cfg.ny, cfg.gx, cfg.nu)
        scale = float(ref.abs().max().clamp_min(1e-30))
        err = float((prof - ref).abs().max()) / scale
        umax = float(prof.abs().max())
        return {
            "cfg": {
                "nx": cfg.nx,
                "ny": cfg.ny,
                "nu": cfg.nu,
                "gx": cfg.gx,
                "gy": cfg.gy,
                "tau": cfg.tau,
                "scheme": cfg.scheme,
                "wall": cfg.wall,
                "force_corrected_wall": cfg.force_corrected_wall,
                "dtype": str(cfg.dtype),
            },
            "steps_done": self.steps_done,
            "steady_residual": self.residual,
            "profile_ux": prof.tolist(),
            "analytic_ux": ref.tolist(),
            "umax": umax,
            "umax_analytic": scale,
            "umax_rel_err": abs(umax - scale) / scale,
            "err_global_vs_analytic": err,
            "mass_drift": float((rho - 1.0).abs().max()),
            "wall_node_ux": [float(prof[0]), float(prof[-1])],
        }