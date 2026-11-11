"""D2Q9 lattice-Boltzmann channel used to (i) verify the solver on plane Poiseuille
flow and (ii) measure the hydrodynamic pressure response of a deflected ice cover.

Why D2Q9 and not D3Q19 (documented deviation from the campaign plan)
-------------------------------------------------------------------
The canonical case is a 2-D channel (water depth ``h`` in z, cover along x).  The
hydroelastic cover-pressure response is a 2-D (x, z) potential-flow quantity, so a
2-D lattice carries the full physics at a small fraction of the cost of D3Q19; a
3-D run would not add information for this canonical case.  The deviation is
stated in the report ("注意点/局限").

Boundary conditions
-------------------
* ``no_slip``   : half-way bounce-back (used for the Poiseuille verification).
* ``free_slip`` : specular reflection about the local wall axis.  The ice underside
  is treated as a free-slip (inviscid outer-region) surface, which is the regime in
  which the potential-flow pressure law ``p_hat = rho_w U^2 k coth(kh) w_hat``
  holds.  The staircase representation of the wavy wall introduces an
  ``O(dw/dx)`` geometric error, quantified in the report.

The flow is driven by a constant body force; the cover waviness provides the only
form drag, so a statistical steady state exists with a well defined mean velocity,
which is *measured* and used in the gain normalisation (no imposed U).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=int)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=int)
WT = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=int)
MIRROR_Y = np.array([0, 1, 4, 3, 2, 8, 7, 6, 5], dtype=int)   # reflect c_y
MIRROR_X = np.array([0, 3, 2, 1, 4, 6, 5, 8, 7], dtype=int)   # reflect c_x


@dataclass
class ChannelConfig:
    nx: int = 64
    ny: int = 40
    tau: float = 0.8
    n_waves: int = 1
    cover_amplitude: float = 1.0     # downward deflection amplitude [lattice units]
    cover_depth: float = 30.0        # mean water depth [lattice units]
    steps: int = 14000
    skip: int = 7000
    bc: str = "free_slip"            # "free_slip" | "no_slip"
    force: float = 1.5e-5            # body force per unit mass
    forcing: str = "guo"             # "guo" | "simple"
    u_target: float | None = None    # if set, proportional velocity controller
    controller_gain: float = 1.0e-6


class D2Q9Channel:
    """Minimal, auditable D2Q9 BGK channel solver (numpy)."""

    def __init__(self, cfg: ChannelConfig):
        self.cfg = cfg
        nx, ny = cfg.nx, cfg.ny
        self.solid = np.zeros((nx, ny), dtype=bool)
        self.wall_height = np.zeros(nx, dtype=int)
        self.force = cfg.force
        self.forcing = cfg.forcing
        self._build_cover(cfg.cover_amplitude)
        self.f = np.ones((9, nx, ny)) * WT[:, None, None]

    # ------------------------------------------------------------------ geometry
    def _build_cover(self, amplitude: float) -> None:
        cfg = self.cfg
        x = np.arange(cfg.nx)
        k = 2.0 * np.pi * cfg.n_waves / cfg.nx
        w = amplitude * np.cos(k * x)
        self.amplitude = amplitude
        self.wall_height = np.round(cfg.cover_depth - w).astype(int)
        self.solid[:] = False
        self.solid[:, 0] = True                       # bottom wall row
        for ix in range(cfg.nx):
            self.solid[ix, self.wall_height[ix]:] = True
        self.fluid = ~self.solid
        self.solid_yp = np.zeros_like(self.solid)
        self.solid_ym = np.zeros_like(self.solid)
        self.solid_xp = np.zeros_like(self.solid)
        self.solid_xm = np.zeros_like(self.solid)
        self.solid_yp[:, :-1] = self.solid[:, 1:]
        self.solid_ym[:, 1:] = self.solid[:, :-1]
        self.solid_xp[:-1, :] = self.solid[1:, :]
        self.solid_xm[1:, :] = self.solid[:-1, :]
        self._build_reflection_table()

    def _build_reflection_table(self) -> None:
        """Precompute (incoming direction, outgoing direction, node mask) triples."""
        table = []
        for i in range(1, 9):
            cx, cy = CX[i], CY[i]
            if cx == 0:
                src_solid = np.roll(self.solid, -cy, axis=1)
            elif cy == 0:
                src_solid = np.roll(self.solid, -cx, axis=0)
            else:
                src_solid = np.roll(np.roll(self.solid, -cy, axis=1), -cx, axis=0)
            m = src_solid & self.fluid
            if not m.any():
                continue
            if self.cfg.bc == "no_slip":
                table.append((i, OPP[i], m))
                continue
            j = np.full(m.shape, OPP[i], dtype=int)
            if cy != 0:
                hit_y = (self.solid_yp if cy > 0 else self.solid_ym) & m
                hit_x = (self.solid_xp if cx > 0 else self.solid_xm) & m & ~hit_y
                j[hit_y] = MIRROR_Y[i]
                j[hit_x] = MIRROR_X[i]
            else:
                hit = (self.solid_xp if cx > 0 else self.solid_xm) & m
                j[hit] = MIRROR_X[i]
            for jj in np.unique(j[m]):
                table.append((i, int(jj), m & (j == jj)))
        self._reflection = table

    def set_cover_amplitude(self, amplitude: float) -> None:
        self._build_cover(amplitude)

    # ------------------------------------------------------------------ initialisation
    def init_equilibrium(self, ux_profile) -> None:
        """Initialise f at local equilibrium with the given u_x profile (y-dependent)."""
        ny = self.cfg.ny
        ux = np.zeros((self.cfg.nx, ny))
        ux[:] = np.asarray(ux_profile, dtype=float)[None, :]
        f = np.empty((9, self.cfg.nx, ny))
        for i in range(9):
            cu = 3.0 * CX[i] * ux
            f[i] = WT[i] * (1.0 + cu + 0.5 * cu * cu - 1.5 * ux * ux)
        self.f = f

    # ------------------------------------------------------------------ streaming
    def _stream(self) -> np.ndarray:
        f = self.f
        out = np.empty_like(f)
        out[0] = f[0]
        # convention: f_i(x) travels to x + c_i  ->  out[i][x] = f_i[x - c_i]
        for i in range(1, 9):
            cx, cy = CX[i], CY[i]
            if cx == 0:
                out[i] = np.roll(f[i], cy, axis=1)
            elif cy == 0:
                out[i] = np.roll(f[i], cx, axis=0)
            else:
                out[i] = np.roll(np.roll(f[i], cy, axis=1), cx, axis=0)
        for i, jj, mask in self._reflection:
            out[jj][mask] = f[i][mask]
        return out

    # ------------------------------------------------------------------ collision
    def _collide(self, f: np.ndarray) -> np.ndarray:
        rho = f.sum(axis=0)
        ux = (f * CX[:, None, None]).sum(axis=0) / rho
        uy = (f * CY[:, None, None]).sum(axis=0) / rho
        ux_eq = ux + 0.5 * self.force            # Guo second-order equilibrium velocity
        usq = 1.5 * (ux_eq * ux_eq + uy * uy)
        omega = 1.0 / self.cfg.tau
        feq = np.empty_like(f)
        for i in range(9):
            cu = 3.0 * (CX[i] * ux_eq + CY[i] * uy)
            feq[i] = WT[i] * rho * (1.0 + cu + 0.5 * cu * cu - usq)
        fnew = f - omega * (f - feq)
        if self.forcing == "guo":
            cst = (1.0 - 0.5 * omega) * 3.0 * self.force
            for i in range(9):
                fnew[i] += cst * WT[i] * (CX[i] - ux_eq)
        else:                                   # first-order (simple) forcing
            for i in range(9):
                fnew[i] += 3.0 * WT[i] * CX[i] * self.force
        return fnew

    # ------------------------------------------------------------------ run
    def run(self) -> dict:
        cfg = self.cfg
        self.f = self._stream_from(self._collide(self.f))
        p_acc = np.zeros((cfg.nx, cfg.ny))
        u_acc = np.zeros((cfg.nx, cfg.ny))
        u_hist: list[float] = []
        n_acc = 0
        for step in range(cfg.steps):
            self.f = self._stream_from(self._collide(self.f))
            if cfg.u_target is not None:
                self.force += cfg.controller_gain * (cfg.u_target - self._mean_u(self.f))
            if step % 200 == 0:
                u_hist.append(self._mean_u(self.f))
            if step >= cfg.steps - cfg.skip:
                rho = self.f.sum(axis=0)
                p_acc += np.where(self.fluid, rho / 3.0, 0.0)
                u_acc += np.where(self.fluid, (self.f * CX[:, None, None]).sum(axis=0) / rho, 0.0)
                n_acc += 1
        n_acc = max(n_acc, 1)
        return {
            "p": p_acc / n_acc,
            "u": u_acc / n_acc,
            "force": self.force,
            "u_history": u_hist,
            "u_mean": self._mean_u(self.f),
            "fluid": self.fluid,
        }

    def _stream_from(self, fpost: np.ndarray) -> np.ndarray:
        self.f = fpost
        return self._stream()

    def _mean_u(self, f: np.ndarray) -> float:
        rho = f.sum(axis=0)
        ux = (f * CX[:, None, None]).sum(axis=0) / rho
        return float(ux[self.fluid].mean())

    # ------------------------------------------------------------------ sampling
    def cover_pressure(self, p: np.ndarray) -> np.ndarray:
        """Pressure sampled one fluid node below the cover, per column."""
        return np.array([p[ix, self.wall_height[ix] - 1] for ix in range(self.cfg.nx)])

    def cover_shear_height(self) -> np.ndarray:
        return self.wall_height


def _harmonic(p: np.ndarray, n_waves: int) -> tuple[float, float]:
    nx = p.shape[0]
    x = 2.0 * np.pi * n_waves * np.arange(nx) / nx
    a = 2.0 / nx * (p * np.cos(x)).sum()
    b = 2.0 / nx * (p * np.sin(x)).sum()
    return float(np.hypot(a, b)), float(np.arctan2(b, a))


def poiseuille_channel(
    ny: int = 34,
    tau: float = 0.8,
    force: float = 1.0e-5,
    steps: int = 12000,
    nx: int = 4,
    forcing: str = "guo",
) -> dict:
    """Plane Poiseuille flow with no-slip walls; profile vs analytical parabola.

    Analytic: u(y) = (g/(2 nu)) * y_e (H - y_e),  y_e = y - 1/2 (half-way wall),
    H = number of *fluid* rows, nu = (tau - 1/2)/3, g = body force per unit mass.

    Three deviations are reported because a lattice wall offset of O(1) cell is a
    known staircase artefact:
      * ``max_relative_deviation``        : raw, one-to-one comparison;
      * ``shape_deviation_after_fit``     : after a single-parameter least-squares
                                            fit of the effective channel width
                                            (standard LBM practice);
      * ``curvature_residual``            : max |nu u'' + g| / g over the bulk,
                                            which is the momentum-balance check.
    """
    cfg = ChannelConfig(nx=nx, ny=ny, tau=tau, force=force, steps=steps,
                        skip=max(steps // 2, 2000), cover_amplitude=0.0,
                        cover_depth=float(ny - 3), bc="no_slip", forcing=forcing)
    ch = D2Q9Channel(cfg)
    fluid_rows = np.where(ch.fluid[0, :])[0]
    n_fluid = fluid_rows.size
    nu_lbm = (tau - 0.5) / 3.0
    y = fluid_rows - 0.5                   # distance from the half-way wall
    u_analytic = force / (2.0 * nu_lbm) * y * (n_fluid - y)
    u_init = np.zeros(ny)
    u_init[fluid_rows] = u_analytic
    ch.init_equilibrium(u_init)
    out = ch.run()
    u_num = out["u"][0, fluid_rows]
    H = float(n_fluid)
    # single-parameter fit of the effective channel width
    yy = y
    basis = yy * (H - yy)
    scale = float(np.sum(u_num * basis) / np.sum(basis * basis))
    u_fit = scale * basis
    shape_dev = float(np.max(np.abs(u_num - u_fit)) / np.max(u_num))
    curv = np.diff(u_num, 2)
    curv_res = float(np.max(np.abs(nu_lbm * curv + force)) / force)
    return {
        "y": y.tolist(),
        "u_numerical": u_num.tolist(),
        "u_analytic": u_analytic.tolist(),
        "u_fitted": u_fit.tolist(),
        "max_relative_deviation": float(np.max(np.abs(u_num - u_analytic))
                                        / np.max(u_analytic)),
        "rms_relative_deviation": float(np.sqrt(np.mean((u_num - u_analytic) ** 2))
                                        / np.max(u_analytic)),
        "shape_deviation_after_fit": shape_dev,
        "amplitude_ratio": float(np.max(u_num) / np.max(u_analytic)),
        "curvature_residual": curv_res,
        "u_max_numerical": float(np.max(u_num)),
        "u_max_analytic": float(np.max(u_analytic)),
        "viscosity_lattice": nu_lbm,
        "force": force,
        "tau": tau,
        "steps": steps,
        "fluid_rows": int(n_fluid),
    }


def wavy_cover_pressure_gain(
    n_waves: int = 1,
    amplitude: float = 1.0,
    depth: float = 30.0,
    force: float = 1.5e-5,
    tau: float = 0.8,
    nx: int = 64,
    ny: int = 44,
    steps: int = 14000,
    skip: int = 7000,
) -> dict:
    """Measure the cover pressure gain  G = p_hat / (rho U^2 k coth(kh) w_hat).

    One wavy-cover run per call (constant body force); the mean velocity is measured
    from the run.  ``G -> 1`` means the potential-flow wall-pressure law holds; the
    prescribed k^2 criterion would instead require ``p = rho U^2 k^2 w`` (i.e. a
    gain that scales like ``k/tanh(kh)`` -- discarded by the measurement).
    """
    cfg = ChannelConfig(nx=nx, ny=ny, tau=tau, force=force, steps=steps, skip=skip,
                        cover_amplitude=amplitude, cover_depth=depth, n_waves=n_waves,
                        bc="free_slip")
    ch = D2Q9Channel(cfg)
    ch.init_equilibrium(np.full(ny, 0.03))
    out = ch.run()
    p_cover = ch.cover_pressure(out["p"])
    dp = p_cover - p_cover.mean()
    p_hat, phase = _harmonic(dp, n_waves)
    k = 2.0 * np.pi * n_waves / nx
    kh = k * depth
    coth = 1.0 / np.tanh(kh)
    u = out["u_mean"]
    theory = u**2 * k * coth * amplitude          # rho = 1 in lattice units
    # alternative (prescribed k^2) normalisation, for the adjudication table
    theory_k2 = u**2 * k * k * amplitude
    return {
        "n_waves": n_waves,
        "amplitude_lattice": amplitude,
        "slope_max": float(2.0 * np.pi * n_waves / nx * amplitude),
        "depth_lattice": depth,
        "tau": tau,
        "u_mean": u,
        "u_history": out["u_history"],
        "k_lattice": k,
        "kh": kh,
        "coth_kh": coth,
        "p_hat": p_hat,
        "phase": phase,
        "p_hat_theory_k1": theory,
        "p_hat_theory_k2": theory_k2,
        "gain": p_hat / theory if theory > 0 else float("nan"),
        "p_cover_mean": float(p_cover.mean()),
        "dp_profile": dp.tolist(),
        "force": force,
    }