"""3D (D3Q19) pulsatile circular-pipe driver -- Womersley tube verification.

This is the geometry of the case reference (Womersley 1955, J. Physiol.
127:553): a straight circular tube of radius ``R``, periodic along its axis,
driven by the uniform time-harmonic body force ``a(t) = a0 cos(omega t)``
(equivalent to ``dp/dx = -rho a``).  The velocity field is axisymmetric and
axial, so a short periodic box (``nx`` cells) is sufficient and only the
radial direction is resolved.

Numerics
--------
D3Q19 BGK with the second-order Guo (2002) body-force source
``(1-1/(2tau)) w_i [3 (c_ix-u_x) + 9 (c_i.u) c_ix] a`` and the equilibrium
velocity shifted by ``a/2`` -- mathematically identical to the library's
verified body-force channel scheme
``tensorlbm.ice_hydro.lbm_channel._collide`` (forcing="guo"), transferred to
D3Q19 (the library ships no numpy D3Q19 Guo entry).

Curved no-slip wall: the tube surface is resolved with the Bouzidi-Firdaouss-
Lallemand (BFL 2001) interpolated bounce-back for every lattice link that ends
inside the solid.  The wall distance fraction ``q`` is computed analytically
from the true circle (not from the staircase), and the reflected population is

    q <  1/2 :  f_opp(A) = 2q f_i(A) + (1 - 2q) f_i(A - e_i)
    q >= 1/2 :  f_opp(A) = (1/2q) f_i(A) + (1 - 1/2q) f_opp(A)

with ``f`` the post-collision state and ``A`` the last fluid node on the link.
Structure transferred from the OSS inventory implementations of the same idea
(pylbm ``BFLBounceBack``, ``pylbm/boundary.py``; lbmpy ``NoSlipLinearBouzidi``;
palabos ``partialBBdynamics``) -- see
``docs/benchmarks/womersley_pulsatile_alpha5_report.md`` section 2.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from ...d3q19 import C, OPPOSITE, W
from . import analytical
from .gates import evaluate_case
from .metrics import fit_phasor, fold_phase, harmonic_metrics, waveform_error

_C = np.asarray(C.cpu(), dtype=np.int64)
_W = np.asarray(W.cpu(), dtype=float)
_OPP = np.asarray(OPPOSITE.cpu(), dtype=np.int64)


@dataclass(frozen=True)
class PipePulsatileConfig:
    """Configuration of one pulsatile pipe run."""

    R: float = 20.5
    alpha: float = 5.0
    tau: float = 0.8
    u_peak: float = 0.03
    nx: int = 4
    measure_periods: int = 4
    n_bins: int = 64
    skip_periods: int | None = None
    steady: bool = False
    seed: int = 0

    def validate(self) -> None:
        if self.R < 4.0:
            raise ValueError("R must be >= 4 lattice units")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be > 0")
        if not 0.5 < self.tau < 3.0:
            raise ValueError("tau must be in (0.5, 3)")
        if self.u_peak <= 0.0 or self.u_peak > 0.05:
            raise ValueError("u_peak must be in (0, 0.05]")
        if self.nx < 2:
            raise ValueError("nx must be >= 2")


class PipePulsatileSolver:
    """D3Q19 BGK pulsatile pipe flow with BFL curved-wall bounce-back."""

    def __init__(self, cfg: PipePulsatileConfig):
        cfg.validate()
        self.cfg = cfg
        self.R = float(cfg.R)
        # odd cross-section with a centre node and at least one solid ring
        self.n = 2 * int(math.ceil(self.R)) + 1
        self.nx = cfg.nx
        self.nz = self.ny = self.n
        self.yc = self.zc = (self.n - 1) / 2.0
        self.nu = (cfg.tau - 0.5) / 3.0
        self.tau = cfg.tau
        self.omega = analytical.omega_from_alpha("pipe", cfg.alpha, self.nu, self.R)
        self.period = 2.0 * math.pi / self.omega
        self.period_int = int(round(self.period))
        self.alpha_effective = analytical.alpha_from_omega(
            "pipe", self.omega, self.nu, self.R
        )
        self.a0 = analytical.force_amplitude_for_peak(
            "pipe", self.R, cfg.alpha, self.omega, cfg.u_peak
        )
        self.u_peak_ana = self.a0 * analytical.centerline_amplitude_unit(
            "pipe", cfg.alpha
        ) / self.omega
        self.ma = cfg.u_peak / math.sqrt(1.0 / 3.0)

        tau_diff = self.R * self.R / (math.pi**2 * self.nu)
        auto_skip = max(5, math.ceil(6.5 * tau_diff / self.period))
        self.skip_periods = cfg.skip_periods if cfg.skip_periods is not None else auto_skip
        self.skip_steps = self.skip_periods * self.period_int
        self.n_meas = cfg.measure_periods * self.period_int
        self.total_steps = self.skip_steps + self.n_meas

        # --- geometry: circular solid mask -------------------------------
        jj, kk = np.meshgrid(np.arange(self.ny), np.arange(self.nz), indexing="ij")
        self.r2 = (jj - self.yc) ** 2 + (kk - self.zc) ** 2
        self.fluid2d = self.r2 <= self.R * self.R
        self.solid2d = ~self.fluid2d
        self.fluid3d = np.broadcast_to(
            self.fluid2d[:, :, None], (self.nz, self.ny, self.nx)
        )
        self._build_bfl_tables()

        # --- initial state: rest -----------------------------------------
        self.f = np.zeros((19, self.nz, self.ny, self.nx))
        u0 = np.zeros((self.nz, self.ny, self.nx))
        self.f = self._equilibrium(np.ones_like(u0), u0, u0, u0)
        self.initial_mass = float(self.f.sum())

        # radial probes (y through the centreline)
        self.y_phys = np.arange(self.ny) - self.yc  # signed distance from axis
        self.center_k = int(round(self.zc))
        self.center_j = int(round(self.yc))
        self.radial_mask = self.fluid2d[self.center_k, :]
        self.r_phys = np.abs(self.y_phys)

        self.z_ana_profile = analytical.phasor(
            "pipe", self.r_phys, self.R, cfg.alpha, self.a0, self.omega
        )
        self.z_ana_center = analytical.centerline_phasor(
            "pipe", self.R, cfg.alpha, self.a0, self.omega
        )

        self.n_bins = cfg.n_bins
        self.prof_sum = np.zeros((cfg.n_bins, self.ny))
        self.center_sum = np.zeros(cfg.n_bins)
        self.counts = np.zeros(cfg.n_bins, dtype=np.int64)

    # -- curved-wall tables ------------------------------------------------
    def _build_bfl_tables(self) -> None:
        tables = []
        p = np.stack([(np.arange(self.ny) - self.yc), (np.arange(self.nz) - self.zc)])
        for i in range(1, 19):
            cx, cy, cz = (int(v) for v in _C[i])
            if cy == 0 and cz == 0:
                continue  # wall is invariant along x: axial links never cut the wall
            nb_solid = np.roll(self.solid2d, (-cz, -cy), axis=(0, 1))
            mask2d = self.fluid2d & nb_solid
            if not mask2d.any():
                continue
            kk, jj = np.nonzero(mask2d)  # axis0 = z (k), axis1 = y (j)
            py = jj - self.yc
            pz = kk - self.zc
            a = float(cy * cy + cz * cz)
            b = 2.0 * (py * cy + pz * cz)
            c = py * py + pz * pz - self.R * self.R
            disc = np.maximum(b * b - 4.0 * a * c, 0.0)
            q = (-b + np.sqrt(disc)) / (2.0 * a)
            q = np.clip(q, 1e-9, 1.0).astype(float)
            q2d = np.zeros((self.nz, self.ny))
            q2d[kk, jj] = q
            mask3 = np.broadcast_to(mask2d[:, :, None], (self.nz, self.ny, self.nx))
            q3 = np.broadcast_to(q2d[:, :, None], (self.nz, self.ny, self.nx))
            tables.append(
                {
                    "i": i,
                    "opp": int(_OPP[i]),
                    "shift": (cz, cy, cx),
                    "mask": mask3,
                    "q": q3,
                    "min_q": float(q.min()),
                    "count": int(q.size),
                }
            )
        self._bfl = tables

    # -- lattice kernels ---------------------------------------------------
    def _macroscopic(self, f: np.ndarray):
        rho = f.sum(axis=0)
        ux = np.einsum("i,izyx->zyx", _C[:, 0].astype(float), f) / rho
        uy = np.einsum("i,izyx->zyx", _C[:, 1].astype(float), f) / rho
        uz = np.einsum("i,izyx->zyx", _C[:, 2].astype(float), f) / rho
        return rho, ux, uy, uz

    def _equilibrium(self, rho, ux, uy, uz):
        u2 = ux * ux + uy * uy + uz * uz
        feq = np.empty((19,) + rho.shape)
        for i in range(19):
            cu = 3.0 * (_C[i, 0] * ux + _C[i, 1] * uy + _C[i, 2] * uz)
            feq[i] = _W[i] * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * u2)
        return feq

    def _collide(self, a: float) -> np.ndarray:
        """BGK + second-order Guo (2002) body force in +x (see module docstring)."""
        rho, ux, uy, uz = self._macroscopic(self.f)
        ux_eq = ux + 0.5 * a                      # Guo shifted equilibrium velocity
        feq = self._equilibrium(rho, ux_eq, uy, uz)
        f_post = self.f - (self.f - feq) / self.tau
        if a != 0.0:
            cst = (1.0 - 0.5 / self.tau) * a
            for i in range(19):
                if _C[i, 0] != 0:
                    cu = _C[i, 0] * ux + _C[i, 1] * uy + _C[i, 2] * uz
                    f_post[i] += (
                        cst * _W[i] * (3.0 * (_C[i, 0] - ux) + 9.0 * cu * _C[i, 0])
                    )
        return f_post

    def _stream(self, f_post: np.ndarray) -> np.ndarray:
        out = np.empty_like(f_post)
        out[0] = f_post[0]
        for i in range(1, 19):
            cx, cy, cz = (int(v) for v in _C[i])
            out[i] = np.roll(f_post[i], (cz, cy, cx), axis=(0, 1, 2))
        # curved no-slip wall: BFL interpolated bounce-back
        for tab in self._bfl:
            i, opp = tab["i"], tab["opp"]
            shift = tab["shift"]
            mask, q = tab["mask"], tab["q"]
            fi = f_post[i]
            fi_prev = np.roll(fi, (shift[0], shift[1], shift[2]), axis=(0, 1, 2))
            qv = q[mask]
            near = qv < 0.5
            fopp = np.where(
                near,
                2.0 * qv * fi[mask] + (1.0 - 2.0 * qv) * fi_prev[mask],
                (1.0 / (2.0 * qv)) * fi[mask]
                + (1.0 - 1.0 / (2.0 * qv)) * f_post[opp][mask],
            )
            out[opp][mask] = fopp
        return out

    def step_force(self, a: float) -> None:
        """Advance one step with the explicitly given force amplitude."""
        self.f = self._stream(self._collide(a))

    def step(self, step_index: int = 0) -> None:
        a = 0.0 if self.cfg.steady else self.a0 * math.cos(self.omega * step_index)
        self.step_force(a)

    # -- measurement -------------------------------------------------------
    def centerline_velocity(self) -> float:
        _, ux, _, _ = self._macroscopic(self.f)
        return float(ux[self.center_k, self.center_j, 0])

    def radial_profile(self) -> np.ndarray:
        _, ux, _, _ = self._macroscopic(self.f)
        return ux[self.center_k, :, 0].copy()

    def _accumulate(self, step_index: int) -> None:
        _, ux, _, _ = self._macroscopic(self.f)
        prof = ux[self.center_k, :, 0]
        bin_idx = int(
            math.floor(
                (self.omega * step_index % (2.0 * math.pi))
                / (2.0 * math.pi)
                * self.n_bins
            )
        )
        bin_idx = min(bin_idx, self.n_bins - 1)
        self.prof_sum[bin_idx] += prof
        self.center_sum[bin_idx] += float(prof[self.center_j])
        self.counts[bin_idx] += 1

    def run_steady(self, u_center_target: float = 0.02) -> dict:
        """Steady Poiseuille check (constant force) -- wall-treatment audit."""
        t0 = time.time()
        a0 = u_center_target * 4.0 * self.nu / (self.R * self.R)
        steps = max(5 * self.period_int, int(8 * self.R * self.R / (math.pi**2 * self.nu)))
        for step in range(1, steps + 1):
            self.step_force(a0)
        elapsed = time.time() - t0
        u_num = self.radial_profile()
        u_ana = analytical.poiseuille_profile(
            "pipe", self.r_phys, self.R, a0, self.nu
        )
        mask = self.radial_mask
        u_c_num = float(u_num[self.center_j])
        u_c_ana = analytical.poiseuille_centerline("pipe", self.R, a0, self.nu)
        return {
            "mode": "steady_poiseuille",
            "geometry": "pipe",
            "R": self.R,
            "n": self.n,
            "tau": self.tau,
            "nu": self.nu,
            "a0": a0,
            "u_center_target": u_center_target,
            "steps": steps,
            "u_center_num": u_c_num,
            "u_center_ana": u_c_ana,
            "finite": bool(np.isfinite(self.f).all()),
            "u_center_rel_pct": abs(u_c_num / u_c_ana - 1.0) * 100.0,
            "profile_l2_pct": float(
                np.sqrt(np.mean((u_num[mask] - u_ana[mask]) ** 2)) / u_c_ana * 100.0
            ),
            "profile_max_pct": float(np.max(np.abs(u_num[mask] - u_ana[mask])) / u_c_ana * 100.0),
            "r": self.r_phys[mask].tolist(),
            "u_num": u_num[mask].tolist(),
            "u_ana": u_ana[mask].tolist(),
            "elapsed_s": round(elapsed, 2),
        }

    def run(self) -> dict:
        t0 = time.time()
        for step in range(1, self.total_steps + 1):
            self.step(step)
            if step > self.skip_steps:
                self._accumulate(step)
        elapsed = time.time() - t0

        good = self.counts > 0
        phase = (np.arange(self.n_bins) + 0.5) * (2.0 * math.pi / self.n_bins)
        center_mean = np.full(self.n_bins, np.nan)
        center_mean[good] = self.center_sum[good] / self.counts[good]
        prof_mean = np.full((self.ny, self.n_bins), np.nan)
        prof_mean[:, good] = (self.prof_sum[good] / self.counts[good][:, None]).T

        fit = fit_phasor(phase / self.omega, center_mean[good], self.omega)
        z_num_center = fit["Z"]
        metrics = harmonic_metrics(z_num_center, self.z_ana_center)

        design = np.column_stack([np.cos(phase[good]), np.sin(phase[good])])
        coef, *_ = np.linalg.lstsq(design, prof_mean[:, good].T, rcond=None)
        z_num_profile = coef[0] - 1j * coef[1]
        mask = self.radial_mask
        row_rel = np.abs(z_num_profile / self.z_ana_profile - 1.0) * 100.0
        row_amp = np.abs(z_num_profile) / np.abs(self.z_ana_profile)

        u_ana_wave = np.real(
            self.z_ana_profile[:, None] * np.exp(1j * phase[good])[None, :]
        )
        err = waveform_error(
            phase[good],
            prof_mean[mask][:, good],
            u_ana_wave[mask],
            scale=abs(self.z_ana_center),
        )
        inner = mask & (self.r_phys / self.R <= 0.8)
        err_inner = waveform_error(
            phase[good],
            prof_mean[inner][:, good],
            u_ana_wave[inner],
            scale=abs(self.z_ana_center),
        )

        mass_drift_pct = (float(self.f.sum()) - self.initial_mass) / self.initial_mass * 100.0
        result = {
            "geometry": "pipe",
            "case": "womersley_pulsatile_pipe_3d",
            "R": self.R,
            "resolution": self.R,
            "n_cross": self.n,
            "nx": self.nx,
            "alpha": self.cfg.alpha,
            "alpha_effective": self.alpha_effective,
            "tau": self.tau,
            "nu": self.nu,
            "omega": self.omega,
            "period_steps": self.period,
            "period_steps_int": self.period_int,
            "a0": self.a0,
            "u_peak_ana": self.u_peak_ana,
            "u_peak_target": self.cfg.u_peak,
            "Ma": self.ma,
            "stokes_layer": self.R / self.cfg.alpha,
            "skip_periods": self.skip_periods,
            "skip_steps": self.skip_steps,
            "measure_periods": self.cfg.measure_periods,
            "n_meas_steps": self.n_meas,
            "total_steps": self.total_steps,
            "n_phase_bins": self.n_bins,
            "samples_per_bin": self.counts.tolist(),
            "transport": "BGK (SRT) D3Q19",
            "wall": "BFL (Bouzidi-Firdaouss-Lallemand) interpolated bounce-back on the true circle; "
            f"{len(self._bfl)} cut-link families, min q={min(t['min_q'] for t in self._bfl):.4f}",
            "driving": "uniform body force a(t)=a0 cos(omega t), second-order Guo (2002) source",
            "centerline": {
                "z_num": [z_num_center.real, z_num_center.imag],
                "z_ana": [self.z_ana_center.real, self.z_ana_center.imag],
                **metrics,
            },
            "metrics": {
                "amplitude_rel_pct": metrics["amplitude_rel_pct"],
                "phase_rel_pct": metrics["phase_rel_pct"],
                "phase_diff_deg": metrics["phase_diff_deg"],
                "profile_l2_pct": err["l2_pct"],
                "profile_l2_inner_pct": err_inner["l2_pct"],
                "profile_max_pct": err["max_pct"],
                "profile_row_complex_rel_max_pct": float(row_rel[mask].max()),
            },
            "phase_fit": {"offset": fit["offset"], "rms": fit["rms"]},
            "profile": {
                "r": self.r_phys[mask].tolist(),
                "amp_ratio_masked": row_amp[mask].tolist(),
                "phase_lag_num_deg_masked": (-np.degrees(np.angle(z_num_profile)))[mask].tolist(),
                "phase_lag_ana_deg_masked": (-np.degrees(np.angle(self.z_ana_profile)))[mask].tolist(),
                "complex_rel_pct_masked": row_rel[mask].tolist(),
                "phase_centers": phase.tolist(),
                "centerline_phase_averaged": center_mean.tolist(),
                "centerline_ana_waveform": np.real(
                    self.z_ana_center * np.exp(1j * phase)
                ).tolist(),
                "radial_profile_phase_averaged": prof_mean[mask].tolist(),
                "radial_ana_phasor_re": self.z_ana_profile[mask].real.tolist(),
                "radial_ana_phasor_im": self.z_ana_profile[mask].imag.tolist(),
                "radial_num_phasor_re": np.real(z_num_profile[mask]).tolist(),
                "radial_num_phasor_im": np.imag(z_num_profile[mask]).tolist(),
            },
            "mass_drift_pct": mass_drift_pct,
            "finite": bool(np.isfinite(self.f).all()),
            "elapsed_s": round(elapsed, 2),
            "steps_per_s": round(self.total_steps / max(elapsed, 1e-9), 1),
        }
        result["gate"] = evaluate_case(result["metrics"])
        return result


def run_case(cfg: PipePulsatileConfig | None = None, **kwargs) -> dict:
    cfg = cfg or PipePulsatileConfig(**kwargs)
    solver = PipePulsatileSolver(cfg)
    if cfg.steady:
        return solver.run_steady()
    return solver.run()