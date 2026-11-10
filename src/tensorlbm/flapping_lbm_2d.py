"""2-D LBM + immersed-boundary solver for oscillating-foil propulsion.

Verification-face solver for node ``P162W1-N07`` (flapping-foil propulsion).
D2Q9 BGK with Guo forcing, direct-forcing immersed boundary around a thin
(lattice-thickness) foil, prescribed harmonic heave+pitch kinematics.

The solver is deliberately small and dependency-light (numpy float64 only) so
that

* it is *bit-wise reproducible* (no threading, no random numbers, fixed
  summation order) -- acceptance criterion C8;
* the force/power accounting is exact at the discrete level: the IBM spreading
  operator is the adjoint of the interpolation operator, so the reaction force
  ``F_body = -sum(F_E)`` is an identity of discrete momentum conservation and
  the body work ``sum(F_E . u_E)`` is the work actually injected in the fluid.

Layout
------
``f`` has shape ``(9, ny, nx)``; ``+x`` streamwise, ``+y`` up.  Freestream
``U`` enters at ``x = 0`` (equilibrium inlet), leaves through a damped sponge
at ``x = nx-1``; lateral walls are free-slip.

Kinematics (matching :mod:`tensorlbm.flapping_propulsion`)
---------------------------------------------------------
``h(t) = h0*sin(omega*t)`` positive *downward*, so the lattice offset is
``y = -h``.  ``alpha(t) = alpha0*sin(omega*t+phi)`` positive nose-up, i.e. a
counter-clockwise rotation of the markers about the pitch axis ``a*b`` from
mid-chord.

Force convention: thrust ``T = -sum(F_E,x)`` (force *on* the foil, forward),
lift ``L = -sum(F_E,y)`` (up positive), input power
``P_in = +sum(F_E . u_E)`` (work done by the foil on the fluid).
"""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import asdict, dataclass

import numpy as np

from .flapping_propulsion import FlappingKinematics

__all__ = [
    "LBMConfig",
    "FlappingFoil2D",
    "build_markers",
    "peskin4",
    "cycle_average",
    "dissipation",
    "file_sha256",
    "D2Q9_C",
    "D2Q9_W",
    "D2Q9_OPP",
]

# ---------------------------------------------------------------------------
# lattice
# ---------------------------------------------------------------------------
D2Q9_C = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=np.int64,
)
D2Q9_W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
D2Q9_OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)
D2Q9_MIRROR_Y = np.array([0, 1, 4, 3, 2, 8, 7, 6, 5], dtype=np.int64)
_CS2 = 1.0 / 3.0

_CX = D2Q9_C[:, 0].astype(float)
_CY = D2Q9_C[:, 1].astype(float)
_W = D2Q9_W.astype(float)
_CX3 = _CX[:, None, None]
_CY3 = _CY[:, None, None]
_W3 = _W[:, None, None]


def peskin4(r: np.ndarray) -> np.ndarray:
    """1-D 4-point Peskin delta (support ``|r| <= 2``)."""
    a = np.abs(np.asarray(r, dtype=float))
    w = np.zeros_like(a)
    m1 = a <= 1.0
    w[m1] = (3.0 - 2.0 * a[m1] + np.sqrt(1.0 + 4.0 * a[m1] - 4.0 * a[m1] ** 2)) / 8.0
    m2 = (a > 1.0) & (a <= 2.0)
    w[m2] = (5.0 - 2.0 * a[m2] - np.sqrt(-7.0 + 12.0 * a[m2] - 4.0 * a[m2] ** 2)) / 8.0
    return w


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """D2Q9 equilibrium, shape ``(9, ny, nx)``.

    Uses the explicit direction sums (``cu`` for the 8 non-rest directions are
    simple +/- combinations of ``ux``, ``uy``) to avoid temporaries.
    """
    f = np.empty((9,) + rho.shape)
    usq = 1.5 * (ux * ux + uy * uy)
    base = 1.0 - usq
    # rest
    f[0] = (4.0 / 9.0) * rho * base
    # axis directions, weight 1/9
    w1 = (1.0 / 9.0) * rho
    f[1] = w1 * (base + 3.0 * ux + 4.5 * ux * ux)
    f[3] = w1 * (base - 3.0 * ux + 4.5 * ux * ux)
    f[2] = w1 * (base + 3.0 * uy + 4.5 * uy * uy)
    f[4] = w1 * (base - 3.0 * uy + 4.5 * uy * uy)
    # diagonals, weight 1/36
    w2 = (1.0 / 36.0) * rho
    cu5 = ux + uy
    cu6 = -ux + uy
    cu7 = -ux - uy
    cu8 = ux - uy
    f[5] = w2 * (base + 3.0 * cu5 + 4.5 * cu5 * cu5)
    f[6] = w2 * (base + 3.0 * cu6 + 4.5 * cu6 * cu6)
    f[7] = w2 * (base + 3.0 * cu7 + 4.5 * cu7 * cu7)
    f[8] = w2 * (base + 3.0 * cu8 + 4.5 * cu8 * cu8)
    return f


def momenta(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Density and velocity (fast, explicit direction sums)."""
    rho = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8]
    mx = rho.copy()
    mx[:] = f[1]
    mx -= f[3]
    mx += f[5]
    mx -= f[6]
    mx -= f[7]
    mx += f[8]
    my = f[2] - f[4] + f[5] + f[6] - f[7] - f[8]
    return rho, mx / rho, my / rho


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class LBMConfig:
    """Case definition (all lengths in lattice units)."""

    chord: int = 64               # chord in lattice cells
    re: float = 500.0             # Reynolds number U*c/nu
    U: float = 0.1                # freestream (inlet) velocity
    h0_over_c: float = 0.02       # heave amplitude / chord
    alpha0_deg: float = 2.0       # pitch amplitude [deg]
    k: float = 1.0                # reduced frequency omega*c/(2U)
    phi_deg: float = 90.0         # pitch-heave phase lead [deg]
    pitch_axis_a: float = -0.5    # pitch axis, semi-chords from mid-chord
    n_cycles: int = 8             # averaging window (>= 8 per acceptance)
    warmup_cycles: int = 2        # discarded transient
    domain_x_over_c: float = 10.0
    domain_y_over_c: float = 6.0
    marker_spacing: float = 0.5   # Lagrangian spacing (cells)
    foil_thickness_over_c: float = 0.04
    sponge_cells: int = 0         # default: 5% of nx
    n_forcing_passes: int = 2     # multi-direct-forcing iterations
    n_samples_per_cycle: int = 64 # CSV / diagnostics resolution
    csv_path: str | None = None
    locked: bool = False          # negative control C7a: motion locked
    ibm: bool = True              # negative control C7c: IBM body force off
    phase_offset_deg: float = 0.0 # negative control C7b: phase shift
    max_force: float = 0.5        # per-marker force clip (stability guard)
    periodic_x: bool = False      # closed domain in x (energy-budget test, C4)

    @property
    def nu(self) -> float:
        return self.U * self.chord / self.re

    @property
    def tau(self) -> float:
        return 0.5 + 3.0 * self.nu

    @property
    def kinematic(self) -> FlappingKinematics:
        return FlappingKinematics(
            U=self.U,
            c=float(self.chord),
            h0=0.0 if self.locked else self.h0_over_c * self.chord,
            alpha0=0.0 if self.locked else math.radians(self.alpha0_deg),
            phi=math.radians(self.phi_deg + self.phase_offset_deg),
            a=self.pitch_axis_a,
        )

    def dims(self) -> tuple[int, int]:
        nx = int(round(self.domain_x_over_c * self.chord))
        ny = int(round(self.domain_y_over_c * self.chord))
        return nx, ny

    def steps_per_cycle(self) -> int:
        return int(round(2.0 * math.pi / self.kinematic.omega(self.k)))

    def period(self) -> float:
        return 2.0 * math.pi / self.kinematic.omega(self.k)

    def total_steps(self) -> int:
        return (self.n_cycles + self.warmup_cycles) * self.steps_per_cycle()


def build_markers(cfg: LBMConfig, t: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Marker positions and rigid-body velocities at time ``t``.

    Thin ellipse (thickness ``foil_thickness_over_c*c``) centred at ``(4c, ny/2)``
    in the body frame, rotated by ``alpha`` (nose-up) and translated by ``-h``
    (heave positive downward).
    """
    c = float(cfg.chord)
    nx, ny = cfg.dims()
    xc = 4.0 * c
    yc = 0.5 * ny
    t_over_c = cfg.foil_thickness_over_c
    n_mark = max(8, int(round(math.pi * c * (1.0 + t_over_c) / cfg.marker_spacing)))

    theta = np.linspace(0.0, 2.0 * math.pi, n_mark, endpoint=False)
    xb = 0.5 * c * np.cos(theta)
    yb = 0.5 * t_over_c * c * np.sin(theta)

    kin = cfg.kinematic
    k = cfg.k
    w = kin.omega(k)
    if cfg.locked:
        alpha = 0.0
        h = 0.0
        adot = 0.0
        hdot = 0.0
    else:
        alpha = kin.alpha0 * math.sin(w * t + kin.phi)
        h = kin.h0 * math.sin(w * t)
        adot = kin.alpha0 * w * math.cos(w * t + kin.phi)
        hdot = kin.h0 * w * math.cos(w * t)

    ca, sa = math.cos(alpha), math.sin(alpha)
    x_axis = kin.a * kin.b
    xr = x_axis + ca * (xb - x_axis) - sa * yb
    yr = sa * (xb - x_axis) + ca * yb - h
    mx = xc + xr
    my = yc + yr

    rx = mx - (xc + x_axis)
    ry = my - yc
    vx = -adot * ry
    vy = -hdot + adot * rx
    return np.vstack([mx, my]), np.vstack([vx, vy])


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------
class FlappingFoil2D:
    """D2Q9 BGK + direct-forcing IBM around a prescribed-motion thin foil."""

    def __init__(self, cfg: LBMConfig):
        self.cfg = cfg
        self.nx, self.ny = cfg.dims()
        self.tau = cfg.tau
        self.nu = cfg.nu

        rho0 = np.ones((self.ny, self.nx))
        ux0 = np.full((self.ny, self.nx), cfg.U)
        uy0 = np.zeros((self.ny, self.nx))
        self.f = equilibrium(rho0, ux0, uy0)

        self.n_sponge = cfg.sponge_cells or max(8, int(0.05 * self.nx))
        self._sponge_x = np.arange(self.nx - self.n_sponge, self.nx)
        self._sponge_w = np.linspace(0.0, 0.2, self.n_sponge)

        self.step_count = 0
        self.time = 0.0
        self._stencil_cache = None

    # -- macros --------------------------------------------------------------
    def _boundary_energy_flux(self) -> float:
        """Mechanical energy flux out of the open boundaries: ``int (E+p) u.n``."""
        if self.cfg.periodic_x:
            return 0.0      # closed box: no boundary flux, the budget is exact
        rho, ux, uy = momenta(self.f)
        e = 0.5 * rho * (ux * ux + uy * uy) + (rho - 1.0) / 3.0
        return float(np.sum(e[-2, :] * ux[-2, :]) - np.sum(e[:, 1] * ux[:, 1]))

    def macros(self):
        return momenta(self.f)

    # -- IBM stencil ---------------------------------------------------------
    def _stencil(self, mx: np.ndarray, my: np.ndarray):
        i0 = np.floor(mx).astype(np.int64)
        j0 = np.floor(my).astype(np.int64)
        off = np.arange(-1, 3)
        ii = i0[:, None] + off[None, :]
        jj = j0[:, None] + off[None, :]
        wx = peskin4(mx[:, None] - ii)
        wy = peskin4(my[:, None] - jj)
        w = wy[:, :, None] * wx[:, None, :]
        ii = np.clip(ii, 0, self.nx - 1)
        jj = np.clip(jj, 0, self.ny - 1)
        return ii, jj, w

    @staticmethod
    def _interp(field: np.ndarray, ii: np.ndarray, jj: np.ndarray, w: np.ndarray) -> np.ndarray:
        vals = field[jj[:, :, None], ii[:, None, :]]
        return np.einsum("nij,nij->n", vals, w)

    @staticmethod
    def _spread(vals, ii, jj, w, ny, nx) -> np.ndarray:
        g = np.zeros((ny, nx))
        np.add.at(g, (jj[:, :, None], ii[:, None, :]), vals[:, None, None] * w)
        return g

    # -- one time step -------------------------------------------------------
    def step(self, record_fields: bool = False) -> dict:
        cfg = self.cfg
        rho, ux, uy = momenta(self.f)
        ux_f = ux
        uy_f = uy

        marker_force_sum = np.zeros(2)
        body_work = 0.0
        body_work_fluid = 0.0
        fx_grid = fy_grid = None

        if cfg.ibm:
            pos, vel = build_markers(cfg, self.time + 0.5)
            mx, my = pos
            vx_t, vy_t = vel
            fx_grid = np.zeros((self.ny, self.nx))
            fy_grid = np.zeros((self.ny, self.nx))
            ii, jj, wts = self._stencil(mx, my)
            inv_rho = 1.0 / rho
            gain = 1.0 / cfg.n_forcing_passes
            flx_tot = np.zeros(mx.size)
            fly_tot = np.zeros(mx.size)
            for _pass in range(cfg.n_forcing_passes):
                u_mx = self._interp(ux_f + 0.5 * fx_grid * inv_rho, ii, jj, wts)
                u_my = self._interp(uy_f + 0.5 * fy_grid * inv_rho, ii, jj, wts)
                dux = gain * (vx_t - u_mx)
                duy = gain * (vy_t - u_my)
                if cfg.max_force:
                    np.clip(dux, -cfg.max_force, cfg.max_force, out=dux)
                    np.clip(duy, -cfg.max_force, cfg.max_force, out=duy)
                fx_grid += self._spread(dux, ii, jj, wts, self.ny, self.nx)
                fy_grid += self._spread(duy, ii, jj, wts, self.ny, self.nx)
                flx_tot += dux
                fly_tot += duy
            marker_force_sum[0] = fx_grid.sum()
            marker_force_sum[1] = fy_grid.sum()
            # mechanical power delivered by the foil = sum(F_marker . v_body);
            # the marker force equals the force the body applies to the fluid,
            # so this is exactly the work injected in the fluid.
            body_work = float(np.sum(flx_tot * vx_t + fly_tot * vy_t))
            body_work_fluid = float(np.sum(fx_grid * ux_f + fy_grid * uy_f))

            ux_s = ux_f + 0.5 * fx_grid * inv_rho
            uy_s = uy_f + 0.5 * fy_grid * inv_rho
            feq = equilibrium(rho, ux_s, uy_s)
            guo = 1.0 - 0.5 / self.tau
            for i in range(9):
                cu = _CX[i] * ux_s + _CY[i] * uy_s
                cf = _CX[i] * fx_grid + _CY[i] * fy_grid
                src = guo * _W[i] * (
                    (cf - (ux_s * fx_grid + uy_s * fy_grid)) / _CS2 + cu * cf / (_CS2**2)
                )
                feq[i] += src
        else:
            feq = equilibrium(rho, ux_f, uy_f)

        # collision (in place on a copy) + streaming
        it = 1.0 / self.tau
        f_post = self.f.copy()
        np.multiply(f_post, 1.0 - it, out=f_post)
        f_post += feq * it

        f_new = np.empty_like(f_post)
        for i in range(9):
            cx, cy = D2Q9_C[i]
            nx_i = f_post[i]
            if cx:
                nx_i = np.roll(nx_i, cx, axis=1)
            if cy:
                nx_i = np.roll(nx_i, cy, axis=0)
            f_new[i] = nx_i

        # inlet (equilibrium at U) -- skipped for the closed periodic box, where
        # streaming already wraps around and no energy crosses x = const
        if not cfg.periodic_x:
            feq_in = equilibrium(
                np.ones(self.ny), np.full(self.ny, cfg.U), np.zeros(self.ny)
            )
            f_new[:, :, 0] = feq_in

        # free-slip lateral walls
        for i in range(9):
            cy = D2Q9_C[i, 1]
            if cy < 0:
                f_new[i, 0, :] = f_new[D2Q9_MIRROR_Y[i], 0, :]
            elif cy > 0:
                f_new[i, -1, :] = f_new[D2Q9_MIRROR_Y[i], -1, :]

        # outlet sponge
        if self.n_sponge > 0 and not cfg.periodic_x:
            fs = f_new[:, :, self._sponge_x]
            rho_s = fs.sum(axis=0)
            uxs = (
                fs[1] - fs[3] + fs[5] - fs[6] - fs[7] + fs[8]
            ) / rho_s
            uys = (fs[2] - fs[4] + fs[5] + fs[6] - fs[7] - fs[8]) / rho_s
            feq_s = equilibrium(rho_s, uxs, uys)
            w = self._sponge_w[None, None, :]
            f_new[:, :, self._sponge_x] = (
                (1.0 - w) * fs + w * feq_s
            )

        self.f = f_new
        self.step_count += 1
        self.time += 1.0

        ke, phi = ke_and_dissipation(self.nu, self.f)
        phi_neq = dissipation_neq(self.f, self.tau)
        flux = self._boundary_energy_flux()
        out = {
            "step": self.step_count,
            "Fx_fluid": float(marker_force_sum[0]),
            "Fy_fluid": float(marker_force_sum[1]),
            "body_work": body_work,
            "body_work_fluid": body_work_fluid,
            "ke": ke,
            "phi": phi,
            "phi_neq": phi_neq,
            "flux": flux,
        }
        if record_fields:
            rho2, ux2, uy2 = momenta(self.f)
            out.update({"ux": ux2, "uy": uy2, "rho": rho2,
                        "fx_body": fx_grid, "fy_body": fy_grid})
        return out

    # -- run -----------------------------------------------------------------
    def run(
        self,
        n_steps: int | None = None,
        csv_path: str | None = None,
        record_every: int | None = None,
        collect_fields_every: int = 0,
    ) -> dict:
        cfg = self.cfg
        if n_steps is None:
            n_steps = cfg.total_steps()
        period = cfg.period()
        if record_every is None:
            record_every = max(1, cfg.steps_per_cycle() // cfg.n_samples_per_cycle)

        rows: list[dict] = []
        snapshots: list[dict] = []
        phi_visc_total = 0.0
        phi_visc_window = 0.0
        writer = None
        fh = None
        if csv_path:
            fh = open(csv_path, "w", newline="")
            writer = csv.writer(fh)
            writer.writerow(
                ["step", "t_over_T", "Fx_fluid", "Fy_fluid", "body_work", "ke", "phi_visc"]
            )

        t_warm = cfg.warmup_cycles * period
        for _ in range(n_steps):
            want_fields = collect_fields_every > 0 and (
                self.step_count % collect_fields_every == 0
            )
            d = self.step(record_fields=want_fields)
            phi = d["phi"]
            phi_visc_total += phi
            if want_fields:
                snapshots.append(
                    {
                        "t_over_T": (self.time) / period,
                        "ux": d["ux"].copy(),
                        "uy": d["uy"].copy(),
                        "fx_body": None if d["fx_body"] is None else d["fx_body"].copy(),
                        "fy_body": None if d["fy_body"] is None else d["fy_body"].copy(),
                    }
                )
            if writer is not None and (self.step_count % record_every == 0):
                writer.writerow(
                    [
                        self.step_count,
                        f"{self.time / period:.6f}",
                        f"{d['Fx_fluid']:.10e}",
                        f"{d['Fy_fluid']:.10e}",
                        f"{d['body_work']:.10e}",
                        f"{d['ke']:.10e}",
                        f"{phi:.10e}",
                    ]
                )
            if self.time >= t_warm:
                phi_visc_window += phi
                rows.append(
                    {
                        "step": self.step_count,
                        "t": self.time,
                        # thrust: force on the foil along the heading.  With the
                        # freestream in +x the foil heads in -x, so a propulsive
                        # force is -x on the body = +F_fluid,x.
                        "Fx": d["Fx_fluid"],
                        "Fy": -d["Fy_fluid"],
                        "W": d["body_work"],
                        "ke": d["ke"],
                        "phi": phi,
                        "phi_neq": d["phi_neq"],
                        "flux": d["flux"],
                    }
                )
        if fh:
            fh.close()

        return {
            "config": asdict(cfg),
            "tau": self.tau,
            "nu": self.nu,
            "dims": [self.nx, self.ny],
            "rows": rows,
            "snapshots": snapshots,
            "phi_visc_total": phi_visc_total,
            "phi_visc_window": phi_visc_window,
            "period": period,
            "steps": n_steps,
        }


def dissipation_neq(f: np.ndarray, tau: float) -> float:
    """Discrete viscous dissipation ``-<Pi_neq : grad u>`` (lattice units).

    The non-equilibrium momentum flux ``Pi_neq_ab = sum_i (f_i-f_i^eq) c_ia c_ib``
    carries the viscous stress of the BGK model, so ``-Pi_neq : grad u`` is the
    actual sink of kinetic energy implied by the discrete momentum equation.  It
    differs from the continuum estimate ``2*nu*S:S`` by a factor
    ``tau/(tau-1/2)``, which matters for the energy budget.
    """
    rho, ux, uy = momenta(f)
    feq = equilibrium(rho, ux, uy)
    fneq = f - feq
    pxx = np.tensordot(_CX * _CX, fneq, axes=1)
    pyy = np.tensordot(_CY * _CY, fneq, axes=1)
    pxy = np.tensordot(_CX * _CY, fneq, axes=1)
    dux_dx = 0.5 * (ux[1:-1, 2:] - ux[1:-1, :-2])
    duy_dy = 0.5 * (uy[2:, 1:-1] - uy[:-2, 1:-1])
    dux_dy = 0.5 * (ux[2:, 1:-1] - ux[:-2, 1:-1])
    duy_dx = 0.5 * (uy[1:-1, 2:] - uy[1:-1, :-2])
    sxy = 0.5 * (dux_dy + duy_dx)
    work = (
        np.einsum("ij,ij->", pxx[1:-1, 1:-1], dux_dx)
        + np.einsum("ij,ij->", pyy[1:-1, 1:-1], duy_dy)
        + np.einsum("ij,ij->", pxy[1:-1, 1:-1], sxy)
        + np.einsum("ij,ij->", pxy[1:-1, 1:-1], sxy)
    )
    return -float(work)


def ke_and_dissipation(nu: float, f: np.ndarray, tau: float | None = None) -> tuple[float, float]:
    """Kinetic energy and viscous dissipation of a D2Q9 state (single pass)."""
    rho, ux, uy = momenta(f)
    ke = 0.5 * float(np.einsum("ij,ij->", rho, ux * ux + uy * uy))
    dux_dy = 0.5 * (ux[2:, 1:-1] - ux[:-2, 1:-1])
    dux_dx = 0.5 * (ux[1:-1, 2:] - ux[1:-1, :-2])
    duy_dy = 0.5 * (uy[2:, 1:-1] - uy[:-2, 1:-1])
    duy_dx = 0.5 * (uy[1:-1, 2:] - uy[1:-1, :-2])
    sxy = 0.5 * (dux_dy + duy_dx)
    phi = 2.0 * nu * float(np.einsum("ij,ij->", dux_dx, dux_dx)
                           + np.einsum("ij,ij->", duy_dy, duy_dy)
                           + 2.0 * np.einsum("ij,ij->", sxy, sxy))
    return ke, phi


def ke_phi_all(nu: float, f: np.ndarray, tau: float) -> tuple[float, float, float]:
    """KE, continuum dissipation ``2*nu*S:S`` and discrete ``-Pi_neq:grad u``."""
    ke, phi_c = ke_and_dissipation(nu, f)
    return ke, phi_c, dissipation_neq(f, tau)


def dissipation(nu: float, ux: np.ndarray, uy: np.ndarray) -> float:
    """``Phi = sum 2*nu*S_ij*S_ij`` with central differences (per time step)."""
    dux_dy = 0.5 * (ux[2:, 1:-1] - ux[:-2, 1:-1])
    dux_dx = 0.5 * (ux[1:-1, 2:] - ux[1:-1, :-2])
    duy_dy = 0.5 * (uy[2:, 1:-1] - uy[:-2, 1:-1])
    duy_dx = 0.5 * (uy[1:-1, 2:] - uy[1:-1, :-2])
    sxy = 0.5 * (dux_dy + duy_dx)
    return float(
        2.0 * nu * np.sum(dux_dx**2 + duy_dy**2 + 2.0 * sxy**2)
    )


# ---------------------------------------------------------------------------
# post-processing
# ---------------------------------------------------------------------------
def cycle_average(res: dict, U: float, chord: float, skip_cycles: float = 0.0) -> dict:
    """Time means and force/power coefficients over the averaging window.

    ``C_T = <F_fluid,x>/(0.5*rho*U^2*c)``  (positive = thrust)
    ``C_L = <-F_fluid,y>/(0.5*rho*U^2*c)`` (positive = lift up)
    ``C_P = <W_body>/(0.5*rho*U^3*c)``     (positive = foil feeds the fluid)
    """
    rows = res["rows"]
    if not rows:
        raise RuntimeError("no samples in the averaging window")
    period = res["period"]
    t0 = rows[0]["t"] + skip_cycles * period
    sel = [r for r in rows if r["t"] >= t0] or rows
    Fx = np.array([r["Fx"] for r in sel])
    Fy = np.array([r["Fy"] for r in sel])
    W = np.array([r["W"] for r in sel])
    ke = np.array([r["ke"] for r in sel])
    q = 0.5 * U * U * chord
    qp = 0.5 * U**3 * chord
    n_cycles = (sel[-1]["t"] - sel[0]["t"]) / period
    return {
        "n_samples": len(sel),
        "n_cycles": n_cycles,
        "Fx_mean": float(Fx.mean()),
        "Fy_mean": float(Fy.mean()),
        "W_mean": float(W.mean()),
        "C_T": float(Fx.mean() / q),
        "C_L": float(Fy.mean() / q),
        "C_P": float(W.mean() / qp),
        "eta": float(Fx.mean() / W.mean()) if W.mean() != 0 else float("nan"),
        "ke_start": float(ke[0]),
        "ke_end": float(ke[-1]),
    }


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()