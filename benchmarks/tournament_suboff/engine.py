"""Lean D3Q19 LBM engine for the SUBOFF resistance tournament (numpy, CPU).

Self-contained (no torch) so it runs on the 702 loong64 cluster.  Physics
conventions are inherited 1:1 from the frozen verified driver
(``benchmarks/verified/suboff_sail_resistance/run.py``):

* D3Q19, float64, step order: collision -> NoDynamics restore -> half-way
  bounce-back (before streaming) -> streaming -> far-field BC -> mass
  correction every 200 steps (``lbm_step_correct`` BB mode);
* far-field BC (``far_field_bc_3d`` 3D-hull convention): inlet x=0 and all
  four lateral faces get the uniform free-stream equilibrium
  feq(rho=1, u=(u_in,0,0)); outlet x=nx-1 is zero-gradient;
* mass correction = global rescale to the initial total mass
  (``correct_mass3d``);
* MRT moment matrix ported verbatim from
  ``src/tensorlbm/solver3d.py::_build_d3q19_mrt_matrices`` (polynomial
  moment construction, rank-19 asserted);
* MRT rate vector ``[0, s_e, s_eps, 0, s_q, 0, s_q, 0, s_q, s_pi, s_pi,
  1/tau x5, 1, 1, 1]`` with repo defaults s_e=1.19, s_eps=1.4, s_q=1.2;
* force = momentum exchange (MEM) with selectable sampling phase;
* wall treatment: no-slip BB or log-law slip (Guo first-order correction)
  with y+ threshold 11.6 (repo wall-function convention).

Turbulence treatments (turbulence family):
* ``none`` — resolved (laminar)
* ``smag`` — Smagorinsky: nu_sgs = (C_s*delta)^2 * sqrt(2 S:S)
* ``wale`` — WALE (C_w=0.32), standard form

The engine is deterministic for a fixed (numpy version, thread count);
cross-node determinism is a gen-2 verification seat, not an assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- #
# D3Q19 lattice
# --------------------------------------------------------------------------- #
_C3D = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
        [1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, -1.0],
        [0.0, 1.0, 1.0], [0.0, 1.0, -1.0], [0.0, -1.0, 1.0], [0.0, -1.0, -1.0],
    ],
    dtype=np.float64,
)
_W3D = np.array(
    [1.0 / 3.0] + [1.0 / 18.0] * 6 + [1.0 / 36.0] * 12, dtype=np.float64
)
_OPPOSITE = np.array(
    [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17], dtype=np.int64
)


def _build_mrt_matrices() -> tuple[np.ndarray, np.ndarray]:
    """D3Q19 MRT (M, M_inv) — port of solver3d._build_d3q19_mrt_matrices."""
    cx, cy, cz = _C3D[:, 0], _C3D[:, 1], _C3D[:, 2]
    e2 = cx**2 + cy**2 + cz**2
    e4 = e2**2
    m = np.array(
        [
            np.ones(19),
            19.0 * e2 - 30.0,
            (21.0 * e4 - 53.0 * e2 + 24.0) / 2.0,
            cx,
            cx * e2 * (5.0 * e2 - 9.0) / 2.0,
            cy,
            cy * e2 * (5.0 * e2 - 9.0) / 2.0,
            cz,
            cz * e2 * (5.0 * e2 - 9.0) / 2.0,
            3.0 * cx**2 - e2,
            cx**2 - cy**2,
            cx * cy,
            cx * cz,
            cy * cz,
            (3.0 * e2 - 5.0) * (3.0 * cx**2 - e2) / 2.0,
            (3.0 * e2 - 5.0) * (cx**2 - cy**2) / 2.0,
            cx**2 * cy,
            cx**2 * cz,
            cy**2 * cx,
        ]
    )
    if np.linalg.matrix_rank(m) != 19:
        raise RuntimeError("D3Q19 MRT matrix rank-deficient")
    return m, np.linalg.inv(m)


_M, _M_INV = _build_mrt_matrices()


def mrt_matrix_self_test() -> dict:
    """M @ M_inv = I; conserved moments exact."""
    c: dict[str, bool] = {}
    c["m_minv_identity"] = bool(np.allclose(_M @ _M_INV, np.eye(19), atol=1e-12))
    f = np.random.default_rng(0).random((19, 64, 32, 32))
    rho = f.sum(0)
    mom = np.stack([(_C3D[:, k][:, None, None, None] * f).sum(0) for k in range(3)])
    m = _M @ f.reshape(19, -1)
    c["rho_moment"] = bool(np.allclose(m[0], rho.reshape(-1), atol=1e-12))
    c["momentum_moments"] = bool(
        np.allclose(m[[3, 5, 7]], mom.reshape(3, -1), atol=1e-12)
    )
    return c


# --------------------------------------------------------------------------- #
# geometry: SUBOFF bare hull (AFF-1), axisymmetric
# --------------------------------------------------------------------------- #
def suboff_radius_profile(xi: np.ndarray) -> np.ndarray:
    """Normalised hull radius r(xi)/R_max, 4-segment Groves-1989 formula.

    Port of ``src/tensorlbm/suboff_cad.py::suboff_radius_profile`` (same
    coefficients, same segment breaks).
    """
    xi = np.asarray(xi, dtype=np.float64)
    r = np.zeros_like(xi)
    L_ft = 14.291667
    BOW_END = 3.333333 / L_ft
    MID_END = 10.645833 / L_ft
    STERN_END = 13.979167 / L_ft

    bow = (xi >= 0.0) & (xi < BOW_END)
    if np.any(bow):
        x_ft = xi[bow] * L_ft
        tmp = 0.3 * x_ft - 1.0
        tmp2 = tmp * tmp
        tmp4 = tmp2 * tmp2
        a1 = (
            1.126395101 * x_ft * tmp4
            + 0.442874707 * x_ft * x_ft * (tmp2 * tmp)
            + 1.0
            - tmp4 * (1.2 * x_ft + 1.0)
        )
        a1 = np.maximum(a1, 0.0)
        r[bow] = np.power(a1, 1.0 / 2.1)

    mid = (xi >= BOW_END) & (xi <= MID_END)
    r[mid] = 1.0

    taper = (xi > MID_END) & (xi <= STERN_END)
    if np.any(taper):
        r1 = 0.1175
        k0 = 10.0
        k1 = 44.6244
        x_ft = xi[taper] * L_ft
        ksi = (13.979167 - x_ft) / 3.333333
        ksi2 = ksi * ksi
        ksi3 = ksi2 * ksi
        ksi4 = ksi3 * ksi
        ksi5 = ksi4 * ksi
        ksi6 = ksi5 * ksi
        a3 = (
            r1 * r1
            + r1 * k0 * ksi2
            + (20.0 - 20.0 * r1 * r1 - 4.0 * r1 * k0 - k1 / 3.0) * ksi3
            + (-45.0 + 45.0 * r1 * r1 + 6.0 * r1 * k0 + k1) * ksi4
            + (36.0 - 36.0 * r1 * r1 - 4.0 * r1 * k0 - k1) * ksi5
            + (-10.0 + 10.0 * r1 * r1 + r1 * k0 + k1 / 3.0) * ksi6
        )
        a3 = np.maximum(a3, 0.0)
        r[taper] = np.sqrt(a3)

    tip = (xi > STERN_END) & (xi <= 1.0)
    if np.any(tip):
        x_ft = xi[tip] * L_ft
        val = np.maximum(1.0 - (3.2 * x_ft - 44.733333) ** 2, 0.0)
        r[tip] = np.sqrt(val) * 0.1175
    return np.clip(r, 0.0, 1.0)


def hull_mask(nx: int, ny: int, nz: int, L: float, x0: float, r_max: float) -> np.ndarray:
    """Solid mask: axisymmetric SUBOFF bare hull centred on the (y,z) midline."""
    x = np.arange(nx)
    xi = (x - x0) / L
    prof = suboff_radius_profile(np.clip(xi, 0.0, 1.0))
    prof[(xi < 0.0) | (xi > 1.0)] = 0.0
    r_phys = prof * r_max
    y = np.arange(ny) - (ny - 1) / 2.0
    z = np.arange(nz) - (nz - 1) / 2.0
    yy, zz = np.meshgrid(y, z, indexing="ij")
    rad = np.sqrt(yy**2 + zz**2)  # (ny, nz)
    solid = np.zeros((nz, ny, nx), dtype=bool)
    for i in range(nx):
        if r_phys[i] > 0.0:
            solid[:, :, i] = rad <= r_phys[i]
    return solid


def wetted_area(L: float, r_max: float, n: int = 4096) -> float:
    """Wetted area S = 2*pi*int R dx (full hull)."""
    xi = np.linspace(0.0, 1.0, n)
    r = suboff_radius_profile(xi) * r_max
    return float(2.0 * np.pi * np.trapezoid(r, xi) * L)


# --------------------------------------------------------------------------- #
# case configuration
# --------------------------------------------------------------------------- #
@dataclass
class CaseConfig:
    """One tournament seat's physics/numerics configuration."""

    # geometry / domain
    L: float = 80.0  # hull length in lattice units
    domain_factor: tuple = (4.0, 12.0, 12.0)  # (x, y, z) multiples of (L, D, D)
    x0_frac: float = 0.5  # bow position as fraction of domain length
    # physics
    re: float = 1000.0
    u_in: float = 0.05
    tau: float | None = None  # None -> physical tau = 3*nu + 0.5
    # numerics
    collision: str = "bgk"  # bgk | mrt
    s_e: float = 1.19
    s_eps: float = 1.4
    s_q: float = 1.2
    s_pi: float | None = None
    sgs: str = "none"  # none | smag | wale
    cs: float = 0.1
    cw: float = 0.32
    wall: str = "bb"  # bb | loglaw
    yplus_min: float = 11.6
    kappa: float = 0.41
    b_log: float = 5.0
    force_phase: str = "post_stream"  # pre_collision|post_collision|post_stream|post_bb
    mass_interval: int = 200
    steps: int = 3000
    sample_interval: int = 10
    avg_frac: float = 0.2  # average last 20% of force samples
    seed: int = 20261107

    def derived(self) -> dict:
        nu = self.u_in * self.L / self.re
        tau = self.tau if self.tau is not None else 3.0 * nu + 0.5
        d = 2.0 * (0.254 / 4.356) * self.L  # D in lattice units
        fx, fy, fz = self.domain_factor
        return {
            "nu": nu,
            "tau": tau,
            "domain": (int(round(fx * self.L)), int(round(fy * d)), int(round(fz * d))),
        }


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #
class D3Q19Engine:
    def __init__(self, cfg: CaseConfig):
        self.cfg = cfg
        self.nx, self.ny, self.nz = cfg.derived()["domain"]
        self.solid = hull_mask(
            self.nx, self.ny, self.nz, cfg.L, cfg.x0_frac * self.nx,
            (0.254 / 4.356) * cfg.L,
        )
        self.n_solid = int(self.solid.sum())
        self._tau = cfg.derived()["tau"]
        self._nu = cfg.derived()["nu"]
        self._s_vec = self._build_s_vec()
        self.f = self._equilibrium(
            np.ones((self.nz, self.ny, self.nx)),
            np.full((self.nz, self.ny, self.nx), cfg.u_in),
            np.zeros((self.nz, self.ny, self.nx)),
            np.zeros((self.nz, self.ny, self.nx)),
        )
        self._mass0 = float(self.f.sum())
        self.n_done = 0
        self.force_hist: list[float] = []
        self.diverged = False

    def _build_s_vec(self) -> np.ndarray:
        cfg = self.cfg
        s_pi = cfg.s_pi if cfg.s_pi is not None else cfg.s_e
        return np.array(
            [0.0, cfg.s_e, cfg.s_eps, 0.0, cfg.s_q, 0.0, cfg.s_q, 0.0, cfg.s_q,
             s_pi, s_pi, 1.0 / self._tau, 1.0 / self._tau, 1.0 / self._tau,
             1.0 / self._tau, 1.0 / self._tau, 1.0, 1.0, 1.0],
            dtype=np.float64,
        )

    def _equilibrium(self, rho, ux, uy, uz) -> np.ndarray:
        cu = (
            _C3D[:, 0][:, None, None, None] * ux[None]
            + _C3D[:, 1][:, None, None, None] * uy[None]
            + _C3D[:, 2][:, None, None, None] * uz[None]
        )
        u2 = ux**2 + uy**2 + uz**2
        return _W3D[:, None, None, None] * rho[None] * (
            1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u2[None]
        )

    def _macroscopic(self, f):
        rho = f.sum(0)
        rho_s = np.where(rho > 1e-12, rho, 1.0)
        ux = (f * _C3D[:, 0][:, None, None, None]).sum(0) / rho_s
        uy = (f * _C3D[:, 1][:, None, None, None]).sum(0) / rho_s
        uz = (f * _C3D[:, 2][:, None, None, None]).sum(0) / rho_s
        return rho, ux, uy, uz

    # -- collision ---------------------------------------------------------- #
    def _collide(self, f):
        cfg = self.cfg
        rho, ux, uy, uz = self._macroscopic(f)
        feq = self._equilibrium(rho, ux, uy, uz)
        if cfg.collision == "bgk":
            out = f + (feq - f) / self._tau
        elif cfg.collision == "mrt":
            m = _M @ f.reshape(19, -1)
            me = _M @ feq.reshape(19, -1)
            m -= self._s_vec[:, None] * (m - me)
            out = (_M_INV @ m).reshape(19, self.nz, self.ny, self.nx)
        else:
            raise ValueError(f"unknown collision {cfg.collision}")
        if cfg.sgs != "none":
            out = self._apply_sgs(out, feq, rho, ux, uy, uz)
        return out

    def _apply_sgs(self, f, feq, rho, ux, uy, uz):
        """Subgrid viscosity: extra relaxation of the non-equilibrium part.

        tau_eff = 3*(nu + nu_sgs) + 0.5; the non-equilibrium part is relaxed
        with 1/tau_eff instead of 1/tau (stress-carrying modes dominate the
        non-equilibrium part at low Ma, so this is a standard Sgs proxy).
        """
        cfg = self.cfg
        du = np.empty((3, 3, self.nz, self.ny, self.nx))
        for a in range(3):
            for b in range(3):
                du[a, b] = np.gradient((ux, uy, uz)[b], axis=a)
        s = 0.5 * (du + np.transpose(du, (1, 0, 2, 3, 4)))
        s2 = np.einsum("abijk,abijk->ijk", s, s)
        delta = 1.0
        if cfg.sgs == "smag":
            nu_sgs = (cfg.cs * delta) ** 2 * np.sqrt(2.0 * s2)
        elif cfg.sgs == "wale":
            sd = np.einsum("abijk,abijk->ijk", s, du)
            nu_sgs = (
                (cfg.cw * np.minimum(np.sqrt(s2), np.sqrt(5.0 * np.maximum(sd, 0.0)))) ** 3
                / np.maximum(sd, 1e-12) ** 1.5
            )
        else:
            raise ValueError(cfg.sgs)
        tau_eff = 3.0 * (self._nu + nu_sgs) + 0.5
        f_neq = f - feq
        return f + f_neq * (1.0 / self._tau - 1.0 / tau_eff)

    # -- step --------------------------------------------------------------- #
    def _do_step(self):
        cfg = self.cfg
        f = self.f
        f_pre = f  # phase snapshot: pre_collision (alias)
        f1 = self._collide(f)
        f1[:, self.solid] = f[:, self.solid]  # NoDynamics
        f_coll = f1  # phase snapshot: post_collision
        f2 = f1.copy()
        f2[:, self.solid] = f1[:, self.solid][_OPPOSITE]  # half-way BB
        f3 = self._stream(f2)
        f_stream = f3  # phase snapshot: post_stream
        if cfg.wall == "loglaw":
            f3 = self._loglaw_wall(f3)
        self._far_field(f3)
        if cfg.mass_interval > 0 and self.n_done % cfg.mass_interval == 0:
            cur = float(f3.sum())
            if cur > 1e-30:
                f3 *= self._mass0 / cur
        self._sample_force(f_pre, f_coll, f_stream, f3)
        if not np.isfinite(f3).all():
            self.diverged = True
        self.f = f3
        self.n_done += 1

    def _stream(self, f):
        out = np.empty_like(f)
        for q in range(19):
            cx, cy, cz = (int(v) for v in _C3D[q])
            if cx == 0 and cy == 0 and cz == 0:
                out[q] = f[q]
                continue
            o, src = out[q], f[q]
            if cx > 0:
                o[:, :, 1:] = src[:, :, :-1]
                o[:, :, 0] = src[:, :, 0]
            elif cx < 0:
                o[:, :, :-1] = src[:, :, 1:]
                o[:, :, -1] = src[:, :, -1]
            if cy > 0:
                o[1:, :, :] = src[:-1, :, :]
                o[0, :, :] = src[0, :, :]
            elif cy < 0:
                o[:-1, :, :] = src[1:, :, :]
                o[-1, :, :] = src[-1, :, :]
            if cz > 0:
                o[:, 1:, :] = src[:, :-1, :]
                o[:, 0, :] = src[:, 0, :]
            elif cz < 0:
                o[:, :-1, :] = src[:, 1:, :]
                o[:, -1, :] = src[:, -1, :]
        return out

    def _far_field(self, f):
        cfg = self.cfg
        feq = self._equilibrium(
            np.ones((1, 1, 1)),
            np.full((1, 1, 1), cfg.u_in),
            np.zeros((1, 1, 1)),
            np.zeros((1, 1, 1)),
        )  # (19,1,1,1)
        feq0 = feq[:, 0, 0, 0]  # (19,)
        f[:, :, :, 0] = feq0[:, None, None]  # inlet
        f[:, :, :, -1] = f[:, :, :, -2]  # outlet zero-gradient
        f[:, 0, :, :] = feq0[:, None, None]  # y-
        f[:, -1, :, :] = feq0[:, None, None]  # y+
        f[:, :, 0, :] = feq0[:, None, None]  # z-
        f[:, :, -1, :] = feq0[:, None, None]  # z+

    def _loglaw_wall(self, f):
        """Log-law slip at solid cells (y+ >= yplus_min), Guo first order.

        u* from the local resolved shear proxy |grad u_x|; u_slip along the
        wall normal; correction 6*w_q*rho*(c_q . u_slip) added to the
        bounced distribution (standard first-order Guo forcing form).
        """
        cfg = self.cfg
        rho, ux, uy, uz = self._macroscopic(f)
        n = np.zeros((3, self.nz, self.ny, self.nx))
        fluid = ~self.solid
        for a in range(3):
            for s in (-1, 1):
                sh = [0, 0, 0]
                sh[a] = s
                shifted = np.roll(fluid, shift=sh, axis=(2, 1, 0))
                n[a] += np.where(self.solid & shifted, s, 0.0)
        nrm = np.sqrt((n**2).sum(0))
        nrm = np.where(nrm > 0, nrm, 1.0)
        n = n / nrm[None]
        gu = np.sqrt(
            np.gradient(ux, axis=0) ** 2
            + np.gradient(ux, axis=1) ** 2
            + np.gradient(ux, axis=2) ** 2
        )
        u_star = np.sqrt(self._nu * gu)
        y_plus = np.maximum(nrm * u_star / np.maximum(self._nu, 1e-12), 1.0)
        u_slip = np.where(
            y_plus >= cfg.yplus_min,
            u_star * (np.log(y_plus) / cfg.kappa + cfg.b_log),
            0.0,
        )
        usv = (u_slip * n[0], u_slip * n[1], u_slip * n[2])
        for q in range(1, 19):
            c_dot_u = _C3D[q, 0] * usv[0] + _C3D[q, 1] * usv[1] + _C3D[q, 2] * usv[2]
            f[q, self.solid] += 6.0 * _W3D[q] * rho[self.solid] * c_dot_u
        return f

    def _sample_force(self, f_pre, f_coll, f_stream, f_final):
        cfg = self.cfg
        phase = cfg.force_phase
        if phase == "pre_collision":
            f_in = f_pre
        elif phase == "post_collision":
            f_in = f_coll
        elif phase == "post_stream":
            f_in = f_stream
        elif phase == "post_bb":
            f_in = f_final
        else:
            raise ValueError(cfg.force_phase)
        # MEM: F = sum_{q, x in solid} (f_in[q,x] - f_bb[q,x]) . c_q
        # f_bb = the value the half-way bounce-back produces from the
        # streamed (arriving) distribution: f_stream[opp].
        f_bb = f_stream[:, self.solid][_OPPOSITE]
        f_in_s = f_in[:, self.solid]
        fx = float(np.einsum("q,qn->", _C3D[:, 0], f_in_s - f_bb))
        self.force_hist.append(fx)

    def run(self, log_every: int = 0) -> dict:
        cfg = self.cfg
        for _ in range(cfg.steps):
            self._do_step()
            if self.diverged:
                break
            if log_every and self.n_done % log_every == 0:
                rho = self.f.sum(0)
                print(
                    f"[step {self.n_done}] rho_mean={rho[~self.solid].mean():.6f} "
                    f"Fx={self.force_hist[-1]:.6e}",
                    flush=True,
                )
        return self.result()

    def result(self) -> dict:
        cfg = self.cfg
        hist = np.array(self.force_hist) if self.force_hist else np.array([np.nan])
        n_avg = max(1, int(len(hist) * cfg.avg_frac))
        finite = bool(np.all(np.isfinite(hist)))
        fx = float(hist[-n_avg:].mean()) if finite else float("nan")
        s_wet = wetted_area(cfg.L, (0.254 / 4.356) * cfg.L)
        cd = fx / (0.5 * 1.0 * cfg.u_in**2 * s_wet)
        return {
            "status": "diverged" if (self.diverged or not np.isfinite(cd)) else "ok",
            "cd_total": float(cd),
            "fx_mean": fx,
            "fx_std": float(hist[-n_avg:].std()) if finite else None,
            "n_samples": int(len(hist)),
            "n_solid_cells": self.n_solid,
            "domain": [self.nx, self.ny, self.nz],
            "wetted_area_lu2": s_wet,
            "tau": self._tau,
            "nu": self._nu,
        }


def run_case(cfg: CaseConfig) -> dict:
    """Run one seat end-to-end; divergence is a scored outcome, not a crash."""
    try:
        eng = D3Q19Engine(cfg)
        res = eng.run()
    except MemoryError:
        return {"status": "error", "error": "OOM", "cd_total": float("nan")}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e), "cd_total": float("nan")}
    if not np.isfinite(res.get("cd_total", float("nan"))):
        res["status"] = "diverged"
    return res
