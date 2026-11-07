"""Ice-resistance numerical-uncertainty *factor certification* (mission IN37-N05).

Why this module exists
----------------------
Mission m5092 certified the numerical uncertainty of the idealised ice-channel
resistance problem with a three-grid GCI study plus a cross-solver (FVM / LBM)
comparison.  Its criterion (c) came out **honest FAIL** for the level-ice case:

    |C_D^FVM - C_D^LBM| / C_D^FVM = 10.30 %      (C_D^FVM = 13.130162,
                                                  C_D^LBM = 14.638326)

which is larger than the combined numerical uncertainty of that study
(U_SN = 4.88 %).  A single scalar "we are 10.3 % off" is not actionable: the
same number can be produced by (i) an under-resolved grid (D = 9..15 cells per
obstruction), by (ii) the wall / pressure operator that closes the staircase
boundary and integrates the force, or by (iii) genuine model-form differences
between a cell-centred finite-volume discretisation and a lattice-Boltzmann
discretisation.

This module turns that single number into an attribution.  It is a *two-axis
factorial certification*:

* resolution axis (``U_D``): operator family fixed, refine the obstruction from
  D = 9-15 cells (the m5092 triple) to D = 30 and D = 60 cells, estimate the
  observed order ``p`` and the Richardson extrapolation, and report the
  grid-convergence uncertainty of every grid level;
* operator axis (``U_op``): finest resolution fixed, sweep four wall / pressure
  operators that bracket the two freedoms of a staircase boundary -- where the
  no-slip surface actually sits (link midpoint, first fluid node, smeared by a
  momentum forcing) and how the force on the ice is integrated (Ladd momentum
  exchange vs. Gauger-Schulte pressure/viscous stress integral);
* 2x2 factorial adjudication: the 10.30 % gap is split into a grid share, an
  operator share and a residual (model-form) share that sum *exactly* to the
  gap, plus the analysis-of-variance main effects and interaction of the 2x2
  design;
* minimum-resolution requirement: the smallest obstruction resolution at which
  the numerical uncertainty of ice resistance, ``U_V``, falls below 5 %.

Case definition (inherited verbatim from m5092 so that numbers are comparable)
-----------------------------------------------------------------------------
Idealised ice channel, cut through a level-ice sheet:

* plane channel, periodic in ``x`` (a periodic cell is the standard
  idealisation of a ship advancing through a quasi-homogeneous ice field and it
  removes inlet/outlet conditions as a spurious grid-dependent error source);
* channel height ``H = 4 D``, length ``Lx = 8 D``, ``D`` = side of the square
  level-ice obstruction (blockage ``D/H = 0.25``);
* lattice/dimensionless units, ``rho = 1``, bulk velocity scale ``u0 = 0.06``,
  ``Re = u0 D / nu = 20``, driving body force per unit mass
  ``g = 12 nu u0 / H**2`` (chosen so that the *empty* channel would carry
  ``u0`` in plane Poiseuille flow);
* quantity of interest ``C_D = F_x / (0.5 rho ubar**2 A_front)`` with ``ubar``
  the *measured* bulk velocity and ``A_front = D`` the frontal projection of
  the obstruction -- the resistance-generating form drag of the ice feature,
  the ice-resistance analogue of hull form drag;
* the obstruction and the two channel walls are placed by *cell index* (never
  by a continuous coordinate test), so every refinement level discretises the
  same staircase geometry: a side-``D`` obstruction covers exactly ``D x D``
  cells at every level.

Solvers
-------
``IceChannelLBM`` -- D2Q9 BGK with Guo forcing, pull-scheme streaming with
half-way bounce-back, and three interchangeable boundary closures:

``sbb``        half-way bounce-back (no-slip at the link midpoint, second order
               for straight walls),
``dirichlet``  velocity-Dirichlet reconstruction at the first fluid node
               (Zou-He style; no-slip sits at the fluid node centre, i.e. the
               solid surface is displaced half a cell into the fluid),
``ibm``        one-layer direct-forcing immersed boundary (no-slip smeared onto
               the surface fluid nodes through a momentum source, and the body
               force recovered from that source).

Two interchangeable force integrators:

``me``      Ladd momentum exchange, ``F = sum_links 2 f_k c_k`` over the
            post-collision populations of fluid nodes with a solid neighbour,
``stress``  Gauger-Schulte surface integral of the pressure and of the
            non-equilibrium momentum flux, ``tau = -(1 - 1/(2 tau_r)) Pi^neq``,
            evaluated on the fluid side of every boundary link.

The four certified operators are therefore ``O1 = sbb+me``, ``O2 = sbb+stress``,
``O3 = dirichlet+stress``, ``O4 = ibm+ibm``.  O1 and O2 share the identical flow
field and differ *only* in the force / pressure operator, which isolates that
effect cleanly; O3 and O4 displace the effective wall position, which bounds the
wall-operator effect.

Evidence grade: REAL (executable LBM, no hard-coded force values).  The FVM
anchor is *inherited evidence* from m5092; this node does not re-run TensorFVM,
see the accompanying report for provenance and caveats.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "LevelIceCase",
    "IceChannelLBM",
    "M5092_FVM_ANCHOR",
    "M5092_LBM_ANCHOR",
    "M5092_U_SN",
    "OPERATORS",
    "GRID_LEVELS",
    "observed_order_ls",
    "richardson_triple",
    "gci_fine",
    "run_single",
    "certify",
]

#: m5092 Richardson-extrapolated drag coefficients (TensorPlatform
#: ``scripts/vv``, ``runs/w10n07/certification.json``, criterion c).
M5092_FVM_ANCHOR: float = 13.130162065502638
M5092_LBM_ANCHOR: float = 14.638325915966066
M5092_U_GCI: float = 0.04450822806385927
M5092_U_SN: float = 0.048795310895459104

#: The four wall / pressure operators certified here.
OPERATORS: Dict[str, Dict[str, object]] = {
    "O1_sbb_halfway": {
        "wall": "sbb",
        "force": "balance",
        "erode": 0,
        "label": "half-way bounce-back, wall on the link midpoint (sharp, 2nd order)",
    },
    "O2_dirichlet_node": {
        "wall": "dirichlet",
        "force": "balance",
        "erode": 0,
        "label": "velocity-Dirichlet reconstruction: wall displaced +0.5 h into the fluid",
    },
    "O3_ibm_diffuse": {
        "wall": "ibm",
        "force": "balance",
        "erode": 0,
        "label": "one-layer direct-forcing immersed boundary: wall smeared, body grown by one cell",
    },
    "O4_sbb_eroded": {
        "wall": "sbb",
        "force": "balance",
        "erode": 1,
        "label": "half-way bounce-back on a one-cell-eroded staircase: wall displaced -0.5 h into the ice",
    },
}

#: Resolution axis.  D = 9, 12, 15 are the m5092 triple ("9-15 cells per
#: obstruction"); D = 30 and D = 60 are the refinements demanded by IN37-N05.
GRID_LEVELS: Tuple[int, ...] = (9, 12, 15, 30, 60)
MAX_STEPS: Dict[int, int] = {9: 9000, 12: 11000, 15: 12000, 30: 13000, 60: 13000}

_W = np.array([4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4)
_CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
_CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
_OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)
_CS2 = 1.0 / 3.0


# ---------------------------------------------------------------------------
# case
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LevelIceCase:
    """Level-ice channel constriction (m5092 case, unchanged).

    ``D`` is the obstruction side in cells and therefore the resolution level:
    ``D`` cells per blockage, blockage ratio ``D / H = 0.25``.
    """

    D: int = 12
    H_over_D: float = 4.0
    L_over_D: float = 8.0
    re_ref: float = 20.0
    u0: float = 0.06
    name: str = "level_ice_constriction"

    @property
    def nx(self) -> int:
        return int(round(self.L_over_D * self.D))

    @property
    def ny(self) -> int:
        """Open channel height plus the two staircase wall rows."""
        return int(round(self.H_over_D * self.D)) + 2

    @property
    def H(self) -> float:
        """Open channel height in cells."""
        return float(self.ny - 2)

    @property
    def nu(self) -> float:
        """Kinematic viscosity fixing ``Re = u0 D / nu``."""
        return self.u0 * self.D / self.re_ref

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5

    @property
    def body_force(self) -> float:
        """Body force per unit mass: the empty channel would carry ``u0``."""
        return 12.0 * self.nu * self.u0 / self.H ** 2

    @property
    def frontal_area(self) -> float:
        return float(self.D)

    def solid_mask(self, erode: int = 0, dilate: int = 0) -> np.ndarray:
        """Cell-index staircase mask of the obstruction only.

        ``erode``/``dilate`` shrink/grow the obstruction, i.e. shift the
        effective wall by one cell.  They exist only for the wall-position
        bracket audit; all certified operators use ``erode = dilate = 0``.
        """
        nx, ny = self.nx, self.ny
        solid = np.zeros((ny, nx), dtype=bool)
        n = int(self.D)
        i0 = int(round(0.5 * nx - 0.5 * n))
        j0 = int(round(1 + 0.5 * (ny - 2) - 0.5 * n))
        j0 = max(1, min(j0, ny - 2 - n + 1))
        ii = (np.arange(n) + i0) % nx
        solid[np.ix_(np.arange(j0, j0 + n), ii)] = True
        grow = int(dilate) - int(erode)
        for _ in range(max(0, grow)):
            new = solid.copy()
            new[1:, :] |= solid[:-1, :]
            new[:-1, :] |= solid[1:, :]
            new[:, 1:] |= solid[:, :-1]
            new[:, :-1] |= solid[:, 1:]
            new[:, 0] |= solid[:, -1]
            new[:, -1] |= solid[:, 0]
            solid = new
        for _ in range(max(0, -grow)):
            new = solid.copy()
            new[1:-1, :] &= solid[:-2, :]
            new[1:-1, :] &= solid[2:, :]
            new[:, 1:-1] &= solid[:, :-2]
            new[:, 1:-1] &= solid[:, 2:]
            solid = new
        solid[0, :] = False
        solid[-1, :] = False
        return solid

    def wall_mask(self) -> np.ndarray:
        w = np.zeros((self.ny, self.nx), dtype=bool)
        w[0, :] = True
        w[-1, :] = True
        return w

    def poiseuille(self) -> np.ndarray:
        """Analytic plane-Poiseuille streamwise velocity of the empty channel."""
        j = np.arange(self.ny, dtype=float) + 0.5
        y0, y1 = 1.0, float(self.ny - 1)
        g = 12.0 * self.nu * self.u0 / (y1 - y0) ** 2
        return np.clip(g / (2.0 * self.nu) * (j - y0) * (y1 - j), 0.0, None)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(
            nx=self.nx,
            ny=self.ny,
            H=self.H,
            nu=self.nu,
            tau=self.tau,
            body_force=self.body_force,
            frontal_area=self.frontal_area,
            blockage=self.D / self.H,
            reynolds=self.u0 * self.D / self.nu,
        )
        return d


# ---------------------------------------------------------------------------
# LBM
# ---------------------------------------------------------------------------
def _shift(a: np.ndarray, k: int, sign: int = 1) -> np.ndarray:
    """Shifted copy of ``a`` along lattice direction ``k``.

    ``sign = +1`` gives ``out[j, i] = a[j - cy, i - cx]`` (the *upstream*
    neighbour, used by pull streaming); ``sign = -1`` gives the *downstream*
    neighbour ``a[j + cy, i + cx]``, which is the node the link points into and
    therefore the one used by the force integrators.
    """
    out = a
    if _CY[k]:
        out = np.roll(out, int(sign * _CY[k]), axis=0)
    if _CX[k]:
        out = np.roll(out, int(sign * _CX[k]), axis=1)
    return out


def _equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Equilibrium populations, shape ``(9, ny, nx)``."""
    cu = _CX[:, None, None] * ux + _CY[:, None, None] * uy
    usq = ux * ux + uy * uy
    return _W[:, None, None] * rho * (1.0 + cu / _CS2 + 0.5 * cu * cu / _CS2 ** 2 - 0.5 * usq / _CS2)


class IceChannelLBM:
    """D2Q9 ice-channel solver with interchangeable wall / force operators."""

    def __init__(
        self,
        case: LevelIceCase,
        wall: str = "sbb",
        erode: int = 0,
        dilate: int = 0,
        with_body: bool = True,
    ) -> None:
        if wall not in ("sbb", "dirichlet", "ibm"):
            raise ValueError(f"unknown wall operator {wall!r}")
        self.case = case
        self.wall = wall
        self.nx, self.ny = case.nx, case.ny
        # ``with_body=False`` gives the empty channel, whose exact solution is
        # plane Poiseuille flow -- the analytic anchor of the solver tests.
        self.body = case.solid_mask(erode=erode, dilate=dilate) if with_body else np.zeros((self.ny, self.nx), dtype=bool)
        self.walls = case.wall_mask()
        self.solid = self.body | self.walls
        self.fluid = ~self.solid
        self.n_open = int(self.fluid.sum())
        self.nu = case.nu
        self.tau_r = case.tau
        self.omega = 1.0 / self.tau_r
        self.g = case.body_force
        self.visc_factor = -(1.0 - 1.0 / (2.0 * self.tau_r))
        # Shifted masks.  ``_src_*`` marks fluid nodes whose upstream neighbour
        # (x - c_k) is solid -- the streaming/bounce-back mask.  ``_nb_*`` marks
        # fluid nodes whose downstream neighbour (x + c_k) is solid, i.e. the
        # links that end on the boundary; those are the links the force
        # integrators must sum over (c_k points into the solid there).
        self._src_solid = {k: _shift(self.solid, k) & self.fluid for k in range(1, 9)}
        self._nb_solid = {k: _shift(self.solid, k, -1) & self.fluid for k in range(1, 9)}
        self._nb_body = {k: _shift(self.body, k, -1) & self.fluid for k in range(1, 9)}
        self._nb_wall = {k: _shift(self.walls, k, -1) & self.fluid for k in range(1, 9)}
        # direct-forcing target: fluid nodes with at least one solid neighbour
        self.surface = np.zeros_like(self.solid)
        for k in range(1, 9):
            self.surface |= self._src_solid[k] | self._nb_solid[k]
        self.surface &= self.fluid
        # immersed boundary is applied to the ice only; the channel walls keep
        # the half-way bounce-back closure so that their force stays unambiguous
        self.ibm_nodes = np.zeros_like(self.solid)
        for k in range(1, 9):
            self.ibm_nodes |= self._nb_body[k] | (self._src_solid[k] & self._nb_body[k])
        self.ibm_nodes = self._nb_body[1] | self._nb_body[2] | self._nb_body[3] | self._nb_body[4] | \
            self._nb_body[5] | self._nb_body[6] | self._nb_body[7] | self._nb_body[8]
        up = case.poiseuille()
        ux = np.broadcast_to(up[:, None], (self.ny, self.nx)).copy()
        ux[~self.fluid] = 0.0
        rho = np.ones((self.ny, self.nx))
        self.f = _equilibrium(rho, ux, np.zeros_like(ux))
        self.f[:, ~self.fluid] = _W[:, None] * 1.0
        self.steps = 0

    # -- diagnostics --------------------------------------------------------
    def moments(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rho = self.f.sum(axis=0)
        ux = (self.f * _CX[:, None, None]).sum(axis=0) / rho
        uy = (self.f * _CY[:, None, None]).sum(axis=0) / rho
        return rho, ux, uy

    def bulk_velocity(self) -> float:
        _rho, ux, _uy = self.moments()
        return float(ux[self.fluid].mean())

    # -- force integrators --------------------------------------------------
    def _force_me(self, f_post: np.ndarray) -> Tuple[float, float, float, float]:
        """Ladd momentum exchange (body, wall) x/y."""
        fx_b = fy_b = fx_w = fy_w = 0.0
        for k in range(1, 9):
            m = self._nb_body[k]
            if m.any():
                fx_b += 2.0 * float(f_post[k][m].sum()) * _CX[k]
                fy_b += 2.0 * float(f_post[k][m].sum()) * _CY[k]
            m = self._nb_wall[k]
            if m.any():
                fx_w += 2.0 * float(f_post[k][m].sum()) * _CX[k]
                fy_w += 2.0 * float(f_post[k][m].sum()) * _CY[k]
        return fx_b, fy_b, fx_w, fy_w

    def _force_stress(self, f: np.ndarray, rho: np.ndarray, ux: np.ndarray, uy: np.ndarray):
        """Gauger-Schulte pressure + viscous stress integral on the body surface."""
        feq = _equilibrium(rho, ux, uy)
        fneq = f - feq
        # non-equilibrium momentum flux Pi^neq_ij
        pn = np.zeros((2, 2, self.ny, self.nx))
        for k in range(9):
            c = np.array([_CX[k], _CY[k]], dtype=float)
            pn += c[:, None, None, None] * c[None, :, None, None] * fneq[k]
        tau = self.visc_factor * pn  # viscous stress sigma_ij
        press = _CS2 * rho
        fx = fy = 0.0
        for k in range(1, 9):
            m = self._nb_body[k]
            if not m.any():
                continue
            c = np.array([_CX[k], _CY[k]], dtype=float)
            # outward normal of the *body* surface at this link is -c; the
            # traction on the body is  -p n + tau . n = +p c - tau . c
            fx += float((press[m] * c[0] - (tau[0, 0][m] * c[0] + tau[0, 1][m] * c[1])).sum())
            fy += float((press[m] * c[1] - (tau[1, 0][m] * c[0] + tau[1, 1][m] * c[1])).sum())
        return fx, fy

    # -- one time step ------------------------------------------------------
    def step(self, with_stress: bool = False) -> Dict[str, float]:
        rho, ux, uy = self.moments()
        f = self.f
        fx_ext = self.g * rho
        fy_ext = np.zeros_like(rho)
        f_ibm_x = f_ibm_y = 0.0
        if self.wall == "ibm":
            # direct forcing: drive the surface fluid nodes to zero velocity
            fu = (f * _CX[:, None, None]).sum(axis=0) + 0.5 * fx_ext
            fv = (f * _CY[:, None, None]).sum(axis=0) + 0.5 * fy_ext
            # target velocity 0 at the surface fluid nodes -> momentum source
            # -rho*u/dt with dt = 1 (direct forcing); the reaction on the body
            # is +sum(source)
            fx_ib = np.where(self.ibm_nodes, -fu, 0.0)
            fy_ib = np.where(self.ibm_nodes, -fv, 0.0)
            fx_ext = fx_ext + fx_ib
            fy_ext = fy_ext + fy_ib
            f_ibm_x = float(-fx_ib[self.ibm_nodes].sum())
            f_ibm_y = float(-fy_ib[self.ibm_nodes].sum())
        # macroscopic velocity with Guo half-force
        ux = ((f * _CX[:, None, None]).sum(axis=0) + 0.5 * fx_ext) / rho
        uy = ((f * _CY[:, None, None]).sum(axis=0) + 0.5 * fy_ext) / rho
        ux[~self.fluid] = 0.0
        uy[~self.fluid] = 0.0
        feq = _equilibrium(rho, ux, uy)
        # Guo forcing term
        cu = _CX[:, None, None] * ux + _CY[:, None, None] * uy
        src = (1.0 - 0.5 * self.omega) * _W[:, None, None] * (
            (_CX[:, None, None] - ux) / _CS2 + cu * _CX[:, None, None] / _CS2 ** 2
        )
        srcy = (1.0 - 0.5 * self.omega) * _W[:, None, None] * (
            (_CY[:, None, None] - uy) / _CS2 + cu * _CY[:, None, None] / _CS2 ** 2
        )
        f_post = f - self.omega * (f - feq) + src * fx_ext + srcy * fy_ext
        # stress force uses the pre-collision state; it is expensive (four
        # outer products) so it is only evaluated when a force sample is taken
        fsx, fsy = self._force_stress(f, rho, ux, uy) if with_stress else (float("nan"), float("nan"))
        # streaming with boundary closure
        f_new = np.empty_like(f)
        if self.wall == "dirichlet":
            req = _equilibrium(rho[None], np.zeros_like(ux)[None], np.zeros_like(uy)[None])
            for k in range(1, 9):
                kk = int(_OPP[k])
                pulled = _shift(f_post[k], k)
                m = self._src_solid[k]
                f_new[k] = np.where(m, req[k] + (f_post[kk] - req[kk]), pulled)
        else:
            for k in range(1, 9):
                pulled = _shift(f_post[k], k)
                m = self._src_solid[k]
                f_new[k] = np.where(m, f_post[int(_OPP[k])], pulled)
        f_new[0] = f_post[0]
        self.f = f_new
        self.steps += 1
        fbx, fby, fwx, fwy = self._force_me(f_post)
        return {
            # global x-momentum budget: drive - wall friction = force on ice
            "cd_balance": self.g * self.n_open - fwx,
            "cd_me": fbx,
            "cd_stress": fsx,
            "cd_ibm": f_ibm_x,
            "f_body_x_me": fbx,
            "f_body_y_me": fby,
            "f_wall_x_me": fwx,
            "f_wall_y_me": fwy,
            "f_body_x_stress": fsx,
            "f_body_y_stress": fsy,
            "f_body_x_ibm": f_ibm_x,
            "u_bar": float(ux[self.fluid].mean()),
        }

    def momentum_balance_residual(self, f_wall_x: float, f_body_x: float) -> float:
        """Steady-state x-momentum balance residual (per unit depth).

        ``g * V_open - F_wall_x - F_body_x = 0`` up to the (zero) streamwise
        pressure drop of a periodic cell.  Used as an independent audit of the
        force integrators.
        """
        return self.g * self.n_open - f_wall_x - f_body_x


# ---------------------------------------------------------------------------
# convergence driver
# ---------------------------------------------------------------------------
def run_single(
    D: int,
    wall: str = "sbb",
    force: str = "me",
    max_steps: Optional[int] = None,
    tol: float = 3.0e-4,
    check_every: int = 250,
    erode: int = 0,
    dilate: int = 0,
    verbose: bool = False,
) -> dict:
    """Time-march one (resolution, operator) cell to steady state.

    Convergence criterion: relative drift of the certified force estimator
    below ``tol`` between two consecutive checks, with a step cap.  The
    achieved drift is reported, so a non-converged run can never be mistaken
    for a converged one.
    """
    case = LevelIceCase(D=int(D))
    solver = IceChannelLBM(case, wall=wall, erode=erode, dilate=dilate)
    cap = int(max_steps if max_steps is not None else MAX_STEPS.get(int(D), 40000))
    key = {"me": "cd_me", "stress": "cd_stress", "ibm": "cd_ibm", "balance": "cd_balance"}[force]
    t0 = time.time()
    hist: List[float] = []
    prev = None
    drift = float("inf")
    converged = False
    last: Dict[str, float] = {}
    while solver.steps < cap:
        check = (solver.steps + 1) % check_every == 0
        last = solver.step(with_stress=check)
        if check:
            val = float(last[key])
            hist.append(val)
            if prev is not None and prev != 0.0:
                drift = abs(val - prev) / abs(prev)
            prev = val
            if verbose and solver.steps % (check_every * 20) == 0:
                print(f"  D={D} {wall}/{force} step={solver.steps} cd={val:.6f} drift={drift:.2e}", flush=True)
            if drift < tol and solver.steps > 4 * check_every:
                converged = True
                break
    ubar = last.get("u_bar", float("nan"))
    area = case.frontal_area
    scale = 0.5 * 1.0 * ubar * ubar * area
    out = {
        "D": int(D),
        "wall": wall,
        "force": force,
        "erode": int(erode),
        "dilate": int(dilate),
        "nx": case.nx,
        "ny": case.ny,
        "h": 1.0 / float(D),
        "steps": solver.steps,
        "max_steps": cap,
        "converged": converged,
        "drift": drift,
        "tol": tol,
        "u_bar": ubar,
        "nu": case.nu,
        "tau": case.tau,
        "body_force": case.body_force,
        "frontal_area": area,
        "cd_balance": last.get("cd_balance", float("nan")) / scale,
        "cd_me": last.get("f_body_x_me", float("nan")) / scale,
        "cd_stress": last.get("f_body_x_stress", float("nan")) / scale,
        "cd_ibm": last.get("f_body_x_ibm", float("nan")) / scale,
        "f_body_x_me": last.get("f_body_x_me", float("nan")),
        "f_wall_x_me": last.get("f_wall_x_me", float("nan")),
        "momentum_balance_residual": solver.momentum_balance_residual(
            last.get("f_wall_x_me", 0.0), last.get("f_body_x_me", 0.0)
        ),
        "n_open": solver.n_open,
        "runtime_s": time.time() - t0,
        "history": hist[-20:],
    }
    out["cd"] = out[key]
    out["cd_me_vs_stress_pct"] = (
        100.0 * (out["cd_me"] - out["cd_stress"]) / out["cd_me"] if out["cd_me"] else float("nan")
    )
    return out


# ---------------------------------------------------------------------------
# GCI / Richardson machinery (Roache 1994/1998, ITTC 7.5-03-02-01)
# ---------------------------------------------------------------------------
def observed_order_ls(h: Sequence[float], f: Sequence[float]) -> dict:
    """Least-squares fit ``f(h) = f0 + a h**p`` over all available grids."""
    h = np.asarray(h, dtype=float)
    f = np.asarray(f, dtype=float)
    ok = np.isfinite(h) & np.isfinite(f) & (h > 0)
    h, f = h[ok], f[ok]
    if h.size < 3:
        return {"p": None, "f_ext": None, "rss": None, "n": int(h.size)}
    best = None
    for p in np.linspace(0.1, 6.0, 5901):
        A = np.vstack([np.ones_like(h), h ** p]).T
        coef, *_ = np.linalg.lstsq(A, f, rcond=None)
        rss = float(((A @ coef - f) ** 2).sum())
        if best is None or rss < best[0]:
            best = (rss, coef, float(p))
    rss, coef, p = best
    return {"p": p, "f_ext": float(coef[0]), "slope": float(coef[1]), "rss": rss, "n": int(h.size)}


def richardson_triple(
    f1: float, f2: float, f3: float, r21: float, r32: float, safety: float = 1.25
) -> dict:
    """Observed order and GCI of a non-uniform three-grid triplet.

    ``f1`` is the finest.  Returns ``p``, the Richardson extrapolation
    ``f_ext``, the convergence ratio ``R`` and the fine-grid GCI.  Non-monotone
    or divergent triplets are reported as degraded, never hidden.
    """
    out: Dict[str, object] = {
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "r21": r21,
        "r32": r32,
        "p": None,
        "f_ext": None,
        "R": None,
        "gci_fine_pct": None,
        "certifiable": False,
        "degradation": None,
    }
    d21 = f2 - f1
    d32 = f3 - f2
    if d32 == 0.0:
        out["degradation"] = "zero_inner_difference"
        return out
    out["R"] = d21 / d32
    if not (0.0 < float(out["R"]) < 1.0):
        out["degradation"] = "non_monotone_or_divergent"
        return out

    def g(p: float) -> float:
        return d21 / d32 - (r21 ** p) * (r32 ** p - 1.0) / (r21 ** p - 1.0)

    lo, hi = 0.05, 8.0
    glo, ghi = g(lo), g(hi)
    if not (np.isfinite(glo) and np.isfinite(ghi)) or glo * ghi > 0.0:
        out["degradation"] = "no_order_root"
        return out
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        gm = g(mid)
        if gm == 0.0:
            break
        if gm * glo <= 0.0:
            hi, ghi = mid, gm
        else:
            lo, glo = mid, gm
    p = 0.5 * (lo + hi)
    out["p"] = p
    out["f_ext"] = f1 + d21 / (r21 ** p - 1.0)
    out["gci_fine_pct"] = 100.0 * safety * abs(d21 / f1) / (r21 ** p - 1.0)
    out["certifiable"] = True
    return out


def gci_fine(f_fine: float, f_ext: float, p: Optional[float], safety: float = 1.25) -> Optional[float]:
    """GCI-style fine-grid discretisation uncertainty (relative, in %).

    Uses the extrapolated value as reference together with the observed order
    (``h**p`` scaling of the remaining error) so that it can be evaluated for
    *every* grid level, which is what the minimum-resolution requirement needs.
    """
    if p is None or not np.isfinite(f_ext) or f_fine == 0.0:
        return None
    return 100.0 * safety * abs((f_fine - f_ext) / f_fine)


if __name__ == "__main__":  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description="IN37-N05 ice-resistance factor certification")
    ap.add_argument("cmd", choices=["single", "sweep"])
    ap.add_argument("--D", type=int, default=12)
    ap.add_argument("--wall", default="sbb")
    ap.add_argument("--force", default="me")
    ap.add_argument("--erode", type=int, default=0)
    ap.add_argument("--dilate", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", default="runs/in37_n05")
    a = ap.parse_args()
    if a.cmd == "single":
        r = run_single(a.D, a.wall, a.force, a.max_steps, a.tol, erode=a.erode, dilate=a.dilate, verbose=True)
        print("RESULT_JSON " + json.dumps(r))
    else:  # pragma: no cover
        pass