"""2D (D2Q9) pulsatile channel-flow driver for the Womersley verification case.

Physics / numerics
------------------
Plane channel with walls at ``y = 0.5`` and ``y = ny - 1.5`` (pre-streaming
half-way bounce-back on the wall rows, the repo-validated analytic-channel
wall pattern also used by ``benchmarks/verified/womersley`` and the poiseuille
/ couette cases), periodic in ``x``, gap ``H_eff = ny - 2``, driven by the
library body-force entry ``tensorlbm.turbulent_channel._apply_body_force_2d``
with the instantaneous force ``a(t) = a0 cos(omega t)``.

Library entries used: ``tensorlbm.d2q9.stream / OPPOSITE / equilibrium /
macroscopic``, and the second-order Guo (2002) body-force collision whose
source term is transferred verbatim from the library's own channel solver
``tensorlbm.ice_hydro.lbm_channel._collide`` (forcing="guo"): the equilibrium
velocity is shifted by ``F/2`` and the zero-sum source
``(1-1/(2tau)) 3 w_i a [ (c_ix - u_x) + 3 (c_i.u) c_ix ]`` is added.
Recorded gap (not hidden): the D2Q9 torch body-force helper
``turbulent_channel._apply_body_force_2d`` used by the older
``benchmarks/verified/womersley`` no longer exists in the library, and no
public D2Q9 torch body-force entry replaced it -- hence this transfer.

Known composition gap (recorded, not hidden): ``tensorlbm.boundaries.
bounce_back_cells`` implements the post-streaming full-way variant (wall on
the nodes); with that composition the effective wall sits one half cell too
far out for these analytic profiles (first-row phasor error 9-12 % at alpha=8,
non-convergent under H -> 2H -- see benchmarks/verified/womersley/README.md).
The pre-streaming half-way pattern is therefore used, which is the library's
own verified analytic-channel convention.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from ...d2q9 import OPPOSITE, _c_on, _w_for, equilibrium, macroscopic
from . import analytical
from .gates import evaluate_case
from .metrics import fit_phasor, fold_phase, harmonic_metrics, waveform_error


def stream(f: torch.Tensor) -> torch.Tensor:
    """Stream D2Q9 distributions, periodic in both axes.

    Local to this verification package: the shipped ``tensorlbm.d2q9`` module
    exposes the lattice tables, ``equilibrium`` and ``macroscopic`` but no
    streaming kernel (the pre-2026 ``stream``/``collide_bgk`` helpers were
    removed in the kernel refactor), so the five-line roll is carried here
    instead of reaching into a removed public entry.
    """
    c = _c_on(f.device)
    out = torch.empty_like(f)
    for i in range(f.shape[0]):
        cx, cy = int(c[i, 0]), int(c[i, 1])  # c[:, 0] is the x-component
        out[i] = f[i] if (cx == 0 and cy == 0) else torch.roll(
            f[i], shifts=(-cy, -cx), dims=(0, 1)
        )
    return out



@dataclass(frozen=True)
class ChannelPulsatileConfig:
    """Configuration of one pulsatile channel run."""

    H: int = 59
    alpha: float = 5.0
    tau: float = 0.8
    u_peak: float = 0.03
    nx: int = 8
    measure_periods: int = 4
    n_bins: int = 64
    skip_periods: int | None = None
    device: str = "cpu"
    seed: int = 0

    def validate(self) -> None:
        if self.H < 5 or self.H % 2 != 1:
            raise ValueError("H must be odd and >= 5 (centreline is then a fluid row)")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be > 0")
        if not 0.5 < self.tau < 3.0:
            raise ValueError("tau must be in (0.5, 3)")
        if self.u_peak <= 0.0 or self.u_peak > 0.1:
            raise ValueError("u_peak must be in (0, 0.1] (incompressible LBM)")


class ChannelPulsatileSolver:
    """D2Q9 BGK pulsatile channel flow with phase-averaged measurement."""

    def __init__(self, cfg: ChannelPulsatileConfig):
        cfg.validate()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.ny = cfg.H + 2
        self.H = float(self.ny - 2)  # half-way walls at 0.5 / ny-1.5
        self.nu = (cfg.tau - 0.5) / 3.0
        self.tau = cfg.tau
        self.omega = analytical.omega_from_alpha("channel", cfg.alpha, self.nu, self.H)
        self.period = 2.0 * math.pi / self.omega
        self.period_int = int(round(self.period))
        self.alpha_effective = analytical.alpha_from_omega(
            "channel", self.omega, self.nu, self.H
        )
        self.a0 = analytical.force_amplitude_for_peak(
            "channel", self.H, cfg.alpha, self.omega, cfg.u_peak
        )
        self.u_peak_ana = self.a0 * analytical.centerline_amplitude_unit(
            "channel", cfg.alpha
        ) / self.omega
        self.ma = cfg.u_peak / math.sqrt(1.0 / 3.0)

        tau_diff = self.H * self.H / (math.pi**2 * self.nu)
        auto_skip = max(5, math.ceil(6.5 * tau_diff / self.period))
        self.skip_periods = cfg.skip_periods if cfg.skip_periods is not None else auto_skip
        self.skip_steps = self.skip_periods * self.period_int
        self.n_meas = cfg.measure_periods * self.period_int
        self.total_steps = self.skip_steps + self.n_meas

        rows = np.arange(1, self.ny - 1)
        self.y_phys = rows - 0.5  # physical ordinate of the fluid rows
        self.z_ana_profile = analytical.phasor(
            "channel", self.y_phys, self.H, cfg.alpha, self.a0, self.omega
        )
        self.z_ana_center = analytical.centerline_phasor(
            "channel", self.H, cfg.alpha, self.a0, self.omega
        )

        wall = torch.zeros((self.ny, cfg.nx), dtype=torch.bool, device=self.device)
        wall[0, :] = True
        wall[-1, :] = True
        self.wall = wall
        self.opp = OPPOSITE.to(self.device)
        self.cx = _c_on(self.device)[:, 0].float()
        self.cy = _c_on(self.device)[:, 1].float()
        self.w = _w_for(torch.float32, self.device).view(9, 1, 1)

        torch.manual_seed(cfg.seed)
        rho0 = torch.ones((self.ny, cfg.nx), device=self.device)
        u0 = torch.zeros((self.ny, cfg.nx), device=self.device)
        self.f = equilibrium(rho0, u0, u0)
        self.initial_mass = float(self.f.sum().item())

        # phase-fold accumulators
        self.n_bins = cfg.n_bins
        self.prof_sum = np.zeros((cfg.n_bins, self.ny - 2))
        self.center_sum = np.zeros(cfg.n_bins)
        self.counts = np.zeros(cfg.n_bins, dtype=np.int64)

    # -- one lattice step ---------------------------------------------------
    def _collide_guo(self, f: torch.Tensor, a: float) -> torch.Tensor:
        """BGK collision + second-order Guo (2002) body force in +x.

        Verbatim structure of ``tensorlbm.ice_hydro.lbm_channel._collide``
        (forcing="guo"), the library's verified body-force channel scheme.
        """
        rho, ux, uy = macroscopic(f)
        ux_eq = ux + 0.5 * a                     # Guo shifted equilibrium velocity
        feq = equilibrium(rho, ux_eq, uy)
        f_post = f - (f - feq) / self.tau
        cst = (1.0 - 0.5 / self.tau) * 3.0 * a
        for i in range(9):
            cu = self.cx[i] * ux + self.cy[i] * uy
            f_post[i] += cst * self.w[i] * (self.cx[i] - ux + 3.0 * cu * self.cx[i])
        return f_post

    def step(self, step_index: int) -> None:
        """Advance one step with force amplitude ``a0 cos(omega * step_index)``."""
        a = self.a0 * math.cos(self.omega * step_index)
        f_pre = self.f
        f = self._collide_guo(f_pre, a)
        # pre-streaming half-way bounce-back on the wall rows
        f = torch.where(self.wall.unsqueeze(0), f_pre[self.opp], f)
        self.f = stream(f)

    def _accumulate(self, step_index: int) -> None:
        _, ux, _ = macroscopic(self.f)
        profile = ux[1:-1, :].mean(dim=1).detach().cpu().numpy()
        bin_idx = int(
            math.floor(
                (self.omega * step_index % (2.0 * math.pi))
                / (2.0 * math.pi)
                * self.n_bins
            )
        )
        bin_idx = min(bin_idx, self.n_bins - 1)
        self.prof_sum[bin_idx] += profile
        self.center_sum[bin_idx] += float(profile[self.ny // 2 - 1])
        self.counts[bin_idx] += 1

    # -- run + measurement --------------------------------------------------
    def run(self, progress: bool = False) -> dict:
        t0 = time.time()
        for step in range(1, self.total_steps + 1):
            self.step(step)
            if step > self.skip_steps:
                self._accumulate(step)
            if progress and step % max(1, self.total_steps // 5) == 0:
                print(f"  channel step {step}/{self.total_steps}", flush=True)
        elapsed = time.time() - t0

        good = self.counts > 0
        phase = (np.arange(self.n_bins) + 0.5) * (2.0 * math.pi / self.n_bins)
        center_mean = np.full(self.n_bins, np.nan)
        center_mean[good] = self.center_sum[good] / self.counts[good]
        prof_mean = np.full((self.ny - 2, self.n_bins), np.nan)
        prof_mean[:, good] = (
            self.prof_sum[good] / self.counts[good][:, None]
        ).T

        # centreline phasor from the whole-period phase-averaged signal
        fit = fit_phasor(phase / self.omega, center_mean[good], self.omega)
        z_num_center = fit["Z"]
        metrics = harmonic_metrics(z_num_center, self.z_ana_center)

        # per-row complex amplitude from the phase-averaged profiles
        design = np.column_stack([np.cos(phase[good]), np.sin(phase[good])])
        coef, *_ = np.linalg.lstsq(design, prof_mean[:, good].T, rcond=None)
        z_num_profile = coef[0] - 1j * coef[1]
        weight = np.abs(self.z_ana_profile)
        mask = weight >= 0.1 * weight.max()
        row_rel = np.abs(z_num_profile / self.z_ana_profile - 1.0) * 100.0
        row_amp_ratio = np.abs(z_num_profile) / np.abs(self.z_ana_profile)
        row_lag_num = -np.degrees(np.angle(z_num_profile))
        row_lag_ana = -np.degrees(np.angle(self.z_ana_profile))

        # phase-averaged waveform error over the whole profile
        u_ana_wave = np.real(
            self.z_ana_profile[:, None] * np.exp(1j * phase[good])[None, :]
        )
        err = waveform_error(
            phase[good], prof_mean[:, good], u_ana_wave, scale=abs(self.z_ana_center)
        )
        inner = np.abs(self.y_phys / self.H - 0.5) <= 0.4
        err_inner = waveform_error(
            phase[good],
            prof_mean[inner][:, good],
            u_ana_wave[inner],
            scale=abs(self.z_ana_center),
        )

        mass_drift_pct = (
            (float(self.f.sum().item()) - self.initial_mass) / self.initial_mass * 100.0
        )
        result = {
            "geometry": "channel",
            "case": "womersley_pulsatile_channel_2d",
            "H": self.cfg.H,
            "H_eff": self.H,
            "resolution": self.cfg.H,
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
            "nx": self.cfg.nx,
            "skip_periods": self.skip_periods,
            "skip_steps": self.skip_steps,
            "measure_periods": self.cfg.measure_periods,
            "n_meas_steps": self.n_meas,
            "total_steps": self.total_steps,
            "n_phase_bins": self.n_bins,
            "samples_per_bin": self.counts.tolist(),
            "transport": "BGK collision, local rolled-stream kernel",
            "wall": "pre-streaming half-way bounce-back (walls at y=0.5, ny-1.5)",
            "driving": "second-order Guo (2002) body force, a(t)=a0 cos(omega t), source transferred from ice_hydro.lbm_channel",
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
            "phase_fit": {
                "offset": fit["offset"],
                "rms": fit["rms"],
            },
            "profile": {
                "y": self.y_phys.tolist(),
                "amp_ratio": row_amp_ratio.tolist(),
                "phase_lag_num_deg": row_lag_num.tolist(),
                "phase_lag_ana_deg": row_lag_ana.tolist(),
                "complex_rel_pct": row_rel.tolist(),
                "mask": mask.tolist(),
                "phase_centers": phase.tolist(),
                "centerline_phase_averaged": center_mean.tolist(),
                "centerline_ana_waveform": np.real(
                    self.z_ana_center * np.exp(1j * phase)
                ).tolist(),
                "profile_phase_averaged": prof_mean.tolist(),
                "profile_ana_phasor_re": self.z_ana_profile.real.tolist(),
                "profile_ana_phasor_im": self.z_ana_profile.imag.tolist(),
                "profile_num_phasor_re": np.real(z_num_profile).tolist(),
                "profile_num_phasor_im": np.imag(z_num_profile).tolist(),
            },
            "mass_drift_pct": mass_drift_pct,
            "finite": bool(torch.isfinite(self.f).all().item()),
            "elapsed_s": round(elapsed, 2),
            "steps_per_s": round(self.total_steps / max(elapsed, 1e-9), 1),
        }
        result["gate"] = evaluate_case(result["metrics"])
        return result


def run_case(cfg: ChannelPulsatileConfig | None = None, **kwargs) -> dict:
    """Convenience wrapper: build the solver and run one case."""
    cfg = cfg or ChannelPulsatileConfig(**kwargs)
    return ChannelPulsatileSolver(cfg).run()