"""Oscillating-foil (flapping-foil) propulsion: LBM solver for a verification face.

This module is the *solver side* of the flapping-foil verification face.  The
analytic side (Theodorsen / Garrick closed forms, thrust-reversal line, GCI,
energy ledger) lives in :mod:`tensorlbm.flapping_propulsion`.

Physics
-------
A NACA 0012 section with chord ``c`` is immersed in a uniform stream ``U`` and
prescribed a combined pitching/plunging motion

.. math::

    h(t) = h_0 \\sin(\\omega t), \\qquad
    \\alpha(t) = \\alpha_0 \\sin(\\omega t + \\varphi)

with the pitch axis at ``x = 0.25 c`` from the leading edge (``a = -1/2`` in
Theodorsen's semi-chord notation).  Three motion modes are supported:

``coupled``    heave + pitch (both amplitudes non-zero)
``pitch``      pure pitching (``h_0 = 0``): thrust is generated purely by the
               leading-edge suction of the pitching section
``heave``      pure plunging (``alpha_0 = 0``)

Reduced frequency ``k = omega c / (2 U)``, Strouhal number ``St = f A / U`` with
``A = 2 h_0`` the peak-to-peak heave excursion (so ``St = 2 k (h_0/c) / pi``).

Numerics
--------
D2Q9 BGK with Guo forcing, multi-direct-forcing immersed boundary, 4-point
Peskin kernel, Lagrangian markers at ``<= 0.5`` lattice spacing, and — this is
the important part relative to a pure surface-only IBM — the *interior* of the
section is filled with forcing points held at the rigid-body velocity.  Without
the interior fill a thin section leaks, the effective body becomes rough, and
the spurious drag swamps the propulsive force signal.

Diagnostics per step:

``Fx_fluid, Fy_fluid``  total IBM force *on the fluid* (so the force on the foil
                        is the negative of it; ``Fx_fluid > 0`` means thrust)
``body_work``           rate of work done by the foil on the fluid
                        ``sum_marker F_m . v_body(m)`` = input power ``P_in``
``ke``                  total kinetic energy (lattice units)
``phi``                 continuum viscous dissipation ``2 nu S:S``
``phi_neq``             discrete dissipation ``-Pi_neq : grad u``
``flux``                advective/diffusive kinetic-energy flux across the
                        open boundaries (zero for a periodic box)

References
----------
Theodorsen, T. (1935). General theory of aerodynamic instability and the
    mechanism of flutter. *NACA TR 496*.
Garrick, I. E. (1936). Propulsion of a flapping and oscillating airfoil.
    *NACA TR 567*.
Peskin, C. S. (2002). The immersed boundary method. *Acta Numerica* 11, 479.
Guo, Z., Zheng, C. & Shi, B. (2002). Discrete lattice effects on the forcing
    term in the lattice Boltzmann method. *Phys. Rev. E* 65, 046308.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .flapping_lbm_2d import (
    _CX,
    _CY,
    _W,
    D2Q9_C,
    D2Q9_MIRROR_Y,
    equilibrium,
    momenta,
    peskin4,
)

__all__ = [
    "FoilMotion",
    "OscillatingFoilConfig",
    "OscillatingFoil2D",
    "naca0012_half_thickness",
    "naca0012_markers",
    "cycle_statistics",
    "dissipation_rate",
    "dissipation_neq_rate",
    "step_series",
    "strouhal_from_k",
    "k_from_strouhal",
]

_CS2 = 1.0 / 3.0


# ---------------------------------------------------------------------------
# kinematics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoilMotion:
    """Prescribed rigid-body motion of the section (lattice units)."""

    c: float = 40.0
    U: float = 0.1
    h0: float = 0.0
    alpha0: float = 0.0
    k: float = 1.0
    phi: float = math.pi / 2.0
    phi_h: float = 0.0  # heave phase (180 deg flips h -> -h, control C7c)
    x_axis_frac: float = 0.25  # pitch axis location from the leading edge

    @property
    def omega(self) -> float:
        """Angular frequency ``omega = 2 k U / c``."""
        return 2.0 * self.k * self.U / self.c

    @property
    def period(self) -> float:
        return 2.0 * math.pi / self.omega

    @property
    def strouhal(self) -> float:
        return 2.0 * self.k * (self.h0 / self.c) / math.pi

    @property
    def a(self) -> float:
        """Theodorsen pitch-axis coordinate (semi-chords from mid-chord)."""
        return 2.0 * (self.x_axis_frac - 0.5)

    def state(self, t: float) -> dict[str, float]:
        """Instantaneous kinematic state at time ``t``."""
        wt = self.omega * t
        h = self.h0 * math.sin(wt + self.phi_h)
        hd = self.h0 * self.omega * math.cos(wt + self.phi_h)
        al = self.alpha0 * math.sin(wt + self.phi)
        ald = self.alpha0 * self.omega * math.cos(wt + self.phi)
        return {"h": h, "h_dot": hd, "alpha": al, "alpha_dot": ald}


def strouhal_from_k(k: float, h0_over_c: float) -> float:
    """``St = f A / U`` with ``A = 2 h0`` (peak-to-peak heave)."""
    return 2.0 * k * h0_over_c / math.pi


def k_from_strouhal(st: float, h0_over_c: float) -> float:
    """Inverse of :func:`strouhal_from_k`."""
    return 0.5 * math.pi * st / h0_over_c


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def naca0012_half_thickness(xi: np.ndarray | float) -> np.ndarray:
    """NACA 0012 half-thickness (closed trailing edge) for ``xi = x/c``."""
    x = np.asarray(xi, dtype=float)
    return 5.0 * 0.12 * (
        0.2969 * np.sqrt(np.clip(x, 0.0, None))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )


def naca0012_markers(c: float, spacing: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Closed NACA 0012 contour in the body frame (LE at ``x=0``, chord along x).

    The contour is traversed from the trailing edge along the upper surface to
    the leading edge and back along the lower surface, with arc length spacing
    not exceeding ``spacing`` (lattice cells).
    """
    # dense geometric sampling, then resample by arc length
    n_dense = max(400, int(20 * c))
    beta = np.linspace(0.0, math.pi, n_dense)  # 0 -> LE, pi -> TE
    xi = 0.5 * (1.0 - np.cos(beta))
    yt = naca0012_half_thickness(xi)
    xs = np.concatenate([xi[::-1], xi[1:]]) * c
    ys = np.concatenate([yt[::-1], -yt[1:]]) * c
    ds = np.hypot(np.diff(xs), np.diff(ys))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    total = s[-1]
    n_mark = max(16, int(math.ceil(total / spacing)))
    su = np.linspace(0.0, total, n_mark, endpoint=False)
    mx = np.interp(su, s, xs)
    my = np.interp(su, s, ys)
    return mx, my


@dataclass
class OscillatingFoilConfig:
    """Case definition; all lengths in lattice units."""

    chord: int = 40
    re: float = 500.0
    U: float = 0.1
    h0_over_c: float = 0.25
    alpha0_deg: float = 0.0
    k: float = 1.0
    phi_deg: float = 90.0
    phi_h_deg: float = 0.0
    mode: str = "coupled"  # coupled | pitch | heave | locked
    domain_x_over_c: float = 12.0
    domain_y_over_c: float = 8.0
    upstream_c: float = 4.0  # leading-edge distance from the inlet
    marker_spacing: float = 0.5
    n_forcing_passes: int = 2
    solid_interior: bool = True
    kernel: str = "4pt"
    n_cycles: int = 8
    warmup_cycles: int = 2
    periodic_x: bool = False
    periodic_y: bool = False
    sponge_frac: float = 0.06
    diag_stride: int = 1
    track_neq: bool = False
    seed: int = 0
    run_name: str = "oscillating_foil"

    def __post_init__(self) -> None:
        if self.mode not in {"coupled", "pitch", "heave", "locked"}:
            raise ValueError(f"unknown mode {self.mode!r}")
        if self.kernel not in {"hat", "4pt"}:
            raise ValueError(f"unknown kernel {self.kernel!r}")

    @property
    def nu(self) -> float:
        return self.U * self.chord / self.re

    @property
    def tau(self) -> float:
        return 0.5 + 3.0 * self.nu

    @property
    def motion(self) -> FoilMotion:
        h0 = 0.0 if self.mode in {"pitch", "locked"} else self.h0_over_c * self.chord
        a0 = 0.0 if self.mode in {"heave", "locked"} else math.radians(self.alpha0_deg)
        return FoilMotion(
            c=float(self.chord),
            U=self.U,
            h0=h0,
            alpha0=a0,
            k=self.k,
            phi=math.radians(self.phi_deg),
            phi_h=math.radians(self.phi_h_deg),
        )

    def dims(self) -> tuple[int, int]:
        nx = int(round(self.domain_x_over_c * self.chord))
        ny = int(round(self.domain_y_over_c * self.chord))
        return nx, ny

    def steps_per_cycle(self) -> int:
        return max(8, int(round(self.motion.period)))

    def total_steps(self) -> int:
        return (self.n_cycles + self.warmup_cycles) * self.steps_per_cycle()


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------


class OscillatingFoil2D:
    """D2Q9 BGK + Guo forcing + multi-direct-forcing IBM around a moving section."""

    def __init__(self, cfg: OscillatingFoilConfig) -> None:
        self.cfg = cfg
        self.nx, self.ny = cfg.dims()
        self.tau = cfg.tau
        self.nu = cfg.nu
        self.motion = cfg.motion
        self.step_count = 0
        self.time = 0.0

        rho0 = np.ones((self.ny, self.nx))
        ux0 = np.full((self.ny, self.nx), cfg.U)
        uy0 = np.zeros((self.ny, self.nx))
        self.f = equilibrium(rho0, ux0, uy0)

        # body reference point: leading edge
        self.x_le = cfg.upstream_c * cfg.chord
        self.y_c = 0.5 * self.ny
        self.xa = self.motion.x_axis_frac * cfg.chord  # pitch axis in body frame

        mx, my = naca0012_markers(float(cfg.chord), cfg.marker_spacing)
        self._bx = mx  # body-frame marker coordinates (LE at origin)
        self._by = my

        # outlet sponge ramp (skipped for a periodic domain)
        n_sp = 0 if cfg.periodic_x else int(round(cfg.sponge_frac * self.nx))
        self.n_sponge = n_sp
        if n_sp > 0:
            self._sp_x = np.arange(self.nx - n_sp, self.nx)
            self._sp_w = np.linspace(0.0, 0.08, n_sp)

        # control-volume planes for the independent momentum balance: the
        # downstream plane is kept *outside* the outlet sponge, otherwise the
        # sponge relaxation biases the flux estimate by ~10%.
        self.cv_planes = (2, max(4, self.nx - 3 - n_sp))

        self._feq_in = None
        self._spread_index = None
        self.ledger = {
            "work": 0.0,
            "flux": 0.0,
            "phi": 0.0,
            "phi_neq": 0.0,
            "ke": [],
        }

    # -- geometry ----------------------------------------------------------
    def body_points(self, t: float):
        """World positions/velocities of the surface markers at time ``t``."""
        st = self.motion.state(t)
        al, ald = st["alpha"], st["alpha_dot"]
        ca, sa = math.cos(al), math.sin(al)
        xr = self._bx - self.xa
        yr = self._by
        px = self.x_le + self.xa + ca * xr - sa * yr
        py = self.y_c + st["h"] + sa * xr + ca * yr
        px_pivot = self.x_le + self.xa
        py_pivot = self.y_c + st["h"]
        vx = -ald * (py - py_pivot)
        vy = st["h_dot"] + ald * (px - px_pivot)
        return px, py, vx, vy, st

    def interior_nodes(self, t: float):
        """Eulerian nodes strictly inside the section (body-frame test)."""
        if not self.cfg.solid_interior:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0), np.empty(0)
        st = self.motion.state(t)
        al = st["alpha"]
        ca, sa = math.cos(al), math.sin(al)
        c = float(self.cfg.chord)
        margin = 0.6
        x0 = int(max(1, math.floor(self.x_le - margin)))
        x1 = int(min(self.nx - 1, math.ceil(self.x_le + c + margin)))
        y0 = int(max(1, math.floor(self.y_c + st["h"] - 0.1 * c - margin)))
        y1 = int(min(self.ny - 1, math.ceil(self.y_c + st["h"] + 0.1 * c + margin)))
        if y1 <= y0 or x1 <= x0:
            # the section does not fit in the computational box (body-free sets)
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0),
                np.empty(0),
            )
        jj, ii = np.mgrid[y0:y1, x0:x1]
        dx = ii + 0.5 - (self.x_le + self.xa)
        dy = jj + 0.5 - (self.y_c + st["h"])
        # rotate into the body frame about the pitch axis
        xb = ca * dx + sa * dy + self.xa
        yb = -sa * dx + ca * dy
        xi = xb / c
        inside = (xi > 0.0) & (xi < 1.0)
        yt = np.zeros_like(xi)
        yt[inside] = naca0012_half_thickness(xi[inside]) * c
        inside &= np.abs(yb) <= yt
        jj = jj[inside]
        ii = ii[inside]
        ux_b = -st["alpha_dot"] * (dy[inside])
        uy_b = st["h_dot"] + st["alpha_dot"] * (dx[inside])
        return jj, ii, ux_b, uy_b

    # -- stencil -----------------------------------------------------------
    def _stencil(self, mx: np.ndarray, my: np.ndarray):
        """Sparse 4-point (or 2-point) Peskin stencil indices + weights."""
        support = 4 if self.cfg.kernel == "4pt" else 2
        half = support // 2
        ix0 = np.floor(mx).astype(np.int64) - half + 1
        iy0 = np.floor(my).astype(np.int64) - half + 1
        di = np.arange(support)
        rx = ix0[:, None] + di[None, :] - mx[:, None]
        ry = iy0[:, None] + di[None, :] - my[:, None]
        wx = peskin4(rx)
        wy = peskin4(ry)
        ii = ix0[:, None, None] + di[None, None, :] + np.zeros((1, support, 1), dtype=np.int64)
        jj = iy0[:, None, None] + di[None, None, :] + np.zeros((1, 1, support), dtype=np.int64)
        if self.cfg.periodic_x:
            ii = np.mod(ii, self.nx)
        else:
            ii = np.clip(ii, 0, self.nx - 1)
        if self.cfg.periodic_y:
            jj = np.mod(jj, self.ny)
        else:
            jj = np.clip(jj, 0, self.ny - 1)
        wts = wy[:, :, None] * wx[:, None, :]
        return ii, jj, wts

    @staticmethod
    def _gather(field: np.ndarray, idx: np.ndarray, w: np.ndarray, nmarks: int) -> np.ndarray:
        return (field[idx].reshape(nmarks, -1) * w).sum(axis=1)

    # -- one step ----------------------------------------------------------
    def step(self, diagnostics: bool = True) -> dict:
        cfg = self.cfg
        nx, ny = self.nx, self.ny
        rho, ux, uy = momenta(self.f)

        t_mid = self.time + 0.5
        mx, my, vx_t, vy_t, _st = self.body_points(t_mid)
        nm = mx.size
        if nm:
            ii, jj, wts = self._stencil(mx, my)
            idx = (jj * nx + ii).reshape(nm, -1)
            w = wts.reshape(nm, -1)
            wsum = w.sum(axis=1, keepdims=True)
        else:  # body-free control run
            idx = w = wsum = None

        fgrid_x = np.zeros(ny * nx)
        fgrid_y = np.zeros(ny * nx)
        flx = np.zeros(nm)
        fly = np.zeros(nm)
        gain = 1.0 / cfg.n_forcing_passes
        inv_rho = 1.0 / rho
        for _ in range(cfg.n_forcing_passes if nm else 0):
            usx = (ux + 0.5 * fgrid_x.reshape(ny, nx) * inv_rho).ravel()
            usy = (uy + 0.5 * fgrid_y.reshape(ny, nx) * inv_rho).ravel()
            umx = self._gather(usx, idx, w, nm) / wsum[:, 0]
            umy = self._gather(usy, idx, w, nm) / wsum[:, 0]
            dux = gain * (vx_t - umx)
            duy = gain * (vy_t - umy)
            flx += dux
            fly += duy
            fgrid_x += np.bincount(idx.ravel(), weights=(w * dux[:, None]).ravel(), minlength=ny * nx)
            fgrid_y += np.bincount(idx.ravel(), weights=(w * duy[:, None]).ravel(), minlength=ny * nx)

        # interior nodes: hold the rigid-body velocity (no fluid inside the foil)
        jj_i, ii_i, uxb, uyb = self.interior_nodes(t_mid)
        fx_grid = fgrid_x.reshape(ny, nx)
        fy_grid = fgrid_y.reshape(ny, nx)
        n_int = jj_i.size
        if n_int:
            fx_grid[jj_i, ii_i] += rho[jj_i, ii_i] * (uxb - ux[jj_i, ii_i])
            fy_grid[jj_i, ii_i] += rho[jj_i, ii_i] * (uyb - uy[jj_i, ii_i])

        Fx_fluid = float(fx_grid.sum())
        Fy_fluid = float(fy_grid.sum())
        if n_int:
            Fx_interior = float(fx_grid[jj_i, ii_i].sum())
            Fy_interior = float(fy_grid[jj_i, ii_i].sum())
        else:
            Fx_interior = Fy_interior = 0.0
        # mechanical power of the actuator: force on the (idealised) rigid
        # surface times its prescribed velocity
        body_work = float(np.sum(flx * vx_t + fly * vy_t))
        # work actually delivered to the *fluid*: the discrete change of the
        # macroscopic kinetic energy caused by the momentum source F is
        # F.u + |F|^2/(2 rho); this is the term that enters the energy ledger.
        work_fluid = float(
            np.sum(fx_grid * ux + fy_grid * uy)
            + np.sum((fx_grid**2 + fy_grid**2) / (2.0 * rho))
        )
        interior_work = (
            float(np.sum(fx_grid[jj_i, ii_i] * uxb + fy_grid[jj_i, ii_i] * uyb)) if n_int else 0.0
        )

        # Guo-forced BGK: u* = u + F/(2 rho), feq(u*) + source
        unsx = ux + 0.5 * fx_grid * inv_rho
        unsy = uy + 0.5 * fy_grid * inv_rho
        feq = equilibrium(rho, unsx, unsy)
        guo = 1.0 - 0.5 / self.tau
        duf = unsx * fx_grid + unsy * fy_grid
        for i in range(9):
            cu = _CX[i] * unsx + _CY[i] * unsy
            cf = _CX[i] * fx_grid + _CY[i] * fy_grid
            feq[i] += guo * _W[i] * ((cf - duf) / _CS2 + cu * cf / (_CS2 * _CS2))

        it = 1.0 / self.tau
        fpost = self.f * (1.0 - it) + feq * it

        f_new = np.empty_like(fpost)
        for i in range(9):
            cx, cy = D2Q9_C[i]
            arr = fpost[i]
            if cx:
                arr = np.roll(arr, cx, axis=1)
            if cy:
                arr = np.roll(arr, cy, axis=0)
            f_new[i] = arr

        # inlet: equilibrium free stream; outlet: zero-gradient + sponge
        if not cfg.periodic_x:
            if self._feq_in is None:
                self._feq_in = equilibrium(
                    np.ones(ny), np.full(ny, cfg.U), np.zeros(ny)
                )
            f_new[:, :, 0] = self._feq_in
            f_new[:, :, 1] = 0.5 * (f_new[:, :, 1] + self._feq_in)
            if self.n_sponge > 0:
                fs = f_new[:, :, self._sp_x]
                rho_s = fs[0] + fs[1] + fs[2] + fs[3] + fs[4] + fs[5] + fs[6] + fs[7] + fs[8]
                uxs = (fs[1] - fs[3] + fs[5] - fs[6] - fs[7] + fs[8]) / rho_s
                uys = (fs[2] - fs[4] + fs[5] + fs[6] - fs[7] - fs[8]) / rho_s
                feq_s = equilibrium(rho_s, uxs, uys)
                wv = self._sp_w[None, None, :]
                f_new[:, :, self._sp_x] = (1.0 - wv) * fs + wv * feq_s
        if not cfg.periodic_y:
            for i in range(9):
                cy = D2Q9_C[i, 1]
                if cy < 0:
                    f_new[i, 0, :] = f_new[D2Q9_MIRROR_Y[i], 0, :]
                elif cy > 0:
                    f_new[i, -1, :] = f_new[D2Q9_MIRROR_Y[i], -1, :]

        self.f = f_new
        self.step_count += 1
        self.time += 1.0

        out = {
            "step": self.step_count,
            "t": self.time,
            "Fx_fluid": Fx_fluid,
            "Fy_fluid": Fy_fluid,
            "body_work": body_work,
            "work_fluid": work_fluid,
            "interior_work": interior_work,
            "Fx_interior": Fx_interior,
            "Fy_interior": Fy_interior,
            "n_markers": nm,
            "n_interior": n_int,
        }
        if diagnostics:
            rho2, ux2, uy2 = momenta(self.f)
            ke = 0.5 * float(np.sum(rho2 * (ux2 * ux2 + uy2 * uy2)))
            phi = dissipation_rate(self.nu, ux2, uy2)
            out["ke"] = ke
            out["phi"] = phi
            out["flux"] = self._boundary_flux()
            # control-volume momentum flux through two x-planes (independent
            # estimate of the mean streamwise force on the section)
            p2 = rho2 / 3.0
            x1, x2 = self.cv_planes
            out["cv_in"] = float(
                np.sum(rho2[:, x1] * ux2[:, x1] ** 2 + p2[:, x1])
            )
            out["cv_out"] = float(
                np.sum(rho2[:, x2] * ux2[:, x2] ** 2 + p2[:, x2])
            )
            if cfg.track_neq:
                phi_neq = dissipation_neq_rate(self.f, self.tau)
                out["phi_neq"] = phi_neq
                self.ledger["phi_neq"] += phi_neq
            self.ledger["work"] += work_fluid
            self.ledger["flux"] += out["flux"]
            self.ledger["phi"] += phi
            self.ledger["ke"].append(ke)
        return out

    def _boundary_flux(self) -> float:
        """Net kinetic-energy flux through the open boundaries (per step)."""
        if self.cfg.periodic_x and self.cfg.periodic_y:
            return 0.0
        rho, ux, uy = momenta(self.f)
        # kinetic-energy flux density: (u^2/2 + p/rho) * u_x     (p = rho/3)
        p = rho / _CS2 / 3.0
        fx = 0.5 * (ux * ux + uy * uy) + p / rho
        out = float(np.sum(fx[:, -1] * ux[:, -1] * rho[:, -1]) - np.sum(fx[:, 0] * ux[:, 0] * rho[:, 0]))
        return out


def _grad4(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """4th-order central-difference gradients on the interior ``[2:-2, 2:-2]``.

    The viscous dissipation is quadratic in the velocity gradient, so a 2nd-order
    difference on a ``k dx ~ 0.1`` field underestimates ``2 nu S:S`` by ~6-12 %.
    The 4th-order stencil reduces that error to ``O((k dx)^4)`` and makes the
    energy ledger (criterion C4) meaningful.
    """
    gx = (-field[:, 4:] + 8.0 * field[:, 3:-1] - 8.0 * field[:, 1:-3] + field[:, :-4]) / 12.0
    gy = (-field[4:, :] + 8.0 * field[3:-1, :] - 8.0 * field[1:-3, :] + field[:-4, :]) / 12.0
    return gx[2:-2, :], gy[:, 2:-2]


def _strain_sq(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """``S_ij S_ij`` with 4th-order gradients, on the interior sub-grid."""
    ux_x, ux_y = _grad4(ux)
    uy_x, uy_y = _grad4(uy)
    sxy = 0.5 * (ux_y + uy_x)
    return ux_x**2 + uy_y**2 + 2.0 * sxy**2


def dissipation_neq_rate(f: np.ndarray, tau: float) -> float:
    """Discrete viscous dissipation ``-Pi_neq : grad u`` of a D2Q9 state.

    The non-equilibrium momentum flux ``Pi_neq_ab = sum_i (f_i - f_i^eq) c_ia c_ib``
    carries the viscous stress of the BGK model: ``sigma = -(1 - 1/(2 tau)) Pi_neq``.
    The viscous sink of kinetic energy is therefore

        Phi = sigma : grad u = -(tau - 1/2)/tau * Pi_neq : grad u

    which is the discrete counterpart of the continuum ``2 nu S:S``
    (``nu = cs^2 (tau - 1/2)``).  Gradients are 4th-order accurate.
    """
    rho, ux, uy = momenta(f)
    feq = equilibrium(rho, ux, uy)
    fneq = f - feq
    pxx = (_CX[:, None, None] ** 2 * fneq).sum(axis=0)
    pyy = (_CY[:, None, None] ** 2 * fneq).sum(axis=0)
    pxy = (_CX[:, None, None] * _CY[:, None, None] * fneq).sum(axis=0)
    ux_x, ux_y = _grad4(ux)
    uy_x, uy_y = _grad4(uy)
    sxy = 0.5 * (ux_y + uy_x)
    work = float(
        np.sum(pxx[2:-2, 2:-2] * ux_x)
        + np.sum(pyy[2:-2, 2:-2] * uy_y)
        + 2.0 * np.sum(pxy[2:-2, 2:-2] * sxy)
    )
    return -(1.0 - 0.5 / tau) * work


def dissipation_rate(nu: float, ux: np.ndarray, uy: np.ndarray) -> float:
    """Continuum viscous dissipation ``2 nu S:S`` (lattice units per step).

    4th-order velocity gradients (see :func:`_grad4`).
    """
    return 2.0 * nu * float(np.sum(_strain_sq(ux, uy)))


# ---------------------------------------------------------------------------
# post-processing
# ---------------------------------------------------------------------------


def cycle_statistics(
    rows: list[dict],
    cfg: OscillatingFoilConfig,
    n_skip_cycles: float = 0.0,
) -> dict:
    """Cycle-resolved and time-mean force/power coefficients.

    Coefficients use ``q = 0.5 rho U^2 c`` (force) and ``q_U = 0.5 rho U^3 c``
    (power), so ``C_T = <F_on_foil,x>/q`` with ``F_on_foil = -F_fluid``.
    """
    q = 0.5 * cfg.U**2 * cfg.chord
    qp = 0.5 * cfg.U**3 * cfg.chord
    per = cfg.steps_per_cycle()
    skip = int(round(n_skip_cycles * per))
    use = [r for r in rows if r["step"] > skip]
    if not use:
        raise ValueError("no samples in averaging window")

    n_cycles = len(use) / per
    # F_fluid is the force the foil applies to the fluid; the force on the foil
    # is its negative.  Thrust pushes the foil upstream (-x), hence
    # C_T = (+F_fluid,x)/q  (positive = thrust), C_L = (-F_fluid,y)/q.
    Fx = np.array([r["Fx_fluid"] for r in use]) / q
    Fy = np.array([-r["Fy_fluid"] for r in use]) / q
    P = np.array([r["body_work"] for r in use]) / qp

    # per-cycle means -> statistical uncertainty of the mean
    n_full = int(len(use) // per)
    cyc_T = np.array([Fx[i * per : (i + 1) * per].mean() for i in range(n_full)])
    cyc_P = np.array([P[i * per : (i + 1) * per].mean() for i in range(n_full)])
    out = {
        "C_T": float(Fx.mean()),
        "C_L": float(Fy.mean()),
        "C_P": float(P.mean()),
        "C_T_rms": float(np.sqrt(np.mean(Fx**2))),
        "C_T_cycle_std": float(cyc_T.std(ddof=1)) if n_full > 1 else 0.0,
        "C_P_cycle_std": float(cyc_P.std(ddof=1)) if n_full > 1 else 0.0,
        "eta": float(Fx.mean() / P.mean()) if P.mean() > 0 else float("nan"),
        "n_cycles_averaged": n_cycles,
        "n_cycles_full": n_full,
        "n_samples": len(use),
        "strouhal": cfg.motion.strouhal,
        "reduced_frequency": cfg.k,
    }
    return out


def step_series(rows: list[dict]) -> dict:
    """Stack per-step diagnostics into arrays (for CSV/figures)."""
    return {
        "step": np.array([r["step"] for r in rows]),
        "Fx_fluid": np.array([r["Fx_fluid"] for r in rows]),
        "Fy_fluid": np.array([r["Fy_fluid"] for r in rows]),
        "body_work": np.array([r["body_work"] for r in rows]),
        "ke": np.array([r.get("ke", np.nan) for r in rows]),
        "phi": np.array([r.get("phi", np.nan) for r in rows]),
        "flux": np.array([r.get("flux", np.nan) for r in rows]),
    }