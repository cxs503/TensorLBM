"""Armaly-caliber backward-facing step (BFS) benchmark -- reusable library API.

Why this module exists
----------------------
:mod:`tensorlbm.backward_facing_step` already ships a BFS driver, but its caliber
is *not* the one used by the Armaly et al. (1983) experiment and its reattachment
diagnostic is a single-row velocity sign test at ``y = 1``:

* inlet velocity **uniform** (experiment: fully developed parabolic profile),
* Reynolds number ``Re = u_in * step_h / nu`` (experiment: ``Re = u_bar * D_h / nu``
  with ``D_h = 2 * inlet height`` of the *inlet* channel),
* expansion ratio fixed at ``H / h_in = 2`` (experiment: 10.1 mm / 5.2 mm = 1.9423),
* reattachment from ``ux`` at one row only -- sensitive to the step-corner
  singularity, which produces spurious early sign changes (recorded in
  ``docs/CASE_BACKWARD_STEP_LBM.md`` of the first round: Re=400 gave X1/S = 0.34).

This module provides the calibrated variant plus a robust wall-shear
reattachment estimator.  Everything is built from the ordinary library kernels
(:mod:`tensorlbm.d2q9`, :mod:`tensorlbm.solver`, :mod:`tensorlbm.boundary`) -- no
new solver is introduced.

Caliber (Armaly et al. 1983, J. Fluid Mech. 127:473-496)
-------------------------------------------------------
======================  ====================================================
quantity                definition used here
======================  ====================================================
geometry                inlet channel ``h_in`` cells, step ``s`` cells,
                        downstream ``H = h_in + s``; default ``h_in=52, s=49``
                        gives ``H/h_in = 101/52 = 1.9423077`` exactly the
                        experimental 10.1/5.2 (deviation 0.000 %)
inlet                   fully developed plane-Poiseuille profile, row mean
                        exactly ``u_bar``
Reynolds number         ``Re = u_bar * D_h / nu``, ``D_h = 2 * h_in``
walls                   half-way bounce-back (pre-stream formulation)
inlet BC                Zou/He velocity, applied to the inlet rows only
outlet BC               Zou/He pressure + non-equilibrium sponge layer
collision               TRT with magic parameter Lambda = 3/16 (removes the
                        half-way bounce-back wall-position error), BGK optional
reattachment           zero crossing of the time-averaged **wall shear**
                        ``tau_w(x) ~ du_x/dy|_wall = -2 u_1 + 3 u_2 - u_3``
                        (exact for a quadratic near-wall profile), located by
                        linear interpolation between columns
======================  ====================================================

The measured observable is ``X1 / s`` (step-height units), the primary lower-wall
reattachment length.  Upper-wall separation/detachment (present for Re >= 400) is
also reported when found.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .boundary.boundaries import (
    make_sponge_strength,
    zou_he_inlet_velocity,
    zou_he_outlet_pressure,
)
from .d2q9 import OPPOSITE, equilibrium, macroscopic
from .solver import collide_bgk, collide_trt, stream

__all__ = [
    "ARMALY_EXPANSION_RATIO",
    "ArmalyConfig",
    "armaly_geometry",
    "parabolic_inlet_profile",
    "wall_shear_from_near_wall",
    "zero_crossing",
    "reattachment_from_wall_shear",
    "make_armaly_solid_mask",
    "run_armaly_case",
]

#: Armaly et al. (1983) measured expansion ratio: outlet 10.1 mm / inlet 5.2 mm.
ARMALY_EXPANSION_RATIO = 10.1 / 5.2  # 1.9423077


def armaly_geometry(scale: int = 1) -> tuple[int, int]:
    """Return ``(h_in, s)`` in lattice cells for the Armaly expansion ratio.

    ``h_in = 52 * scale`` and ``s = 49 * scale`` reproduce ``H / h_in =
    101 / 52 = 10.1 / 5.2`` exactly, so the only geometric deviation from the
    experiment is the staircase representation of the step corner.
    """
    if scale < 1:
        raise ValueError("scale must be >= 1")
    return 52 * scale, 49 * scale


def parabolic_inlet_profile(n_rows: int, u_bar: float) -> torch.Tensor:
    """Fully developed plane-Poiseuille profile with exact row mean ``u_bar``.

    Values are placed at fluid-node centres ``y = i + 1/2`` between the two
    channel walls (``y = 0`` and ``y = n_rows``) and renormalised so that the
    discrete row mean is exactly ``u_bar`` (zero net mass-flux bias).
    """
    i = np.arange(n_rows, dtype=np.float64) + 0.5
    h = float(n_rows)
    prof = 4.0 * i * (h - i) / (h * h)  # parabola, max 1.0 at centre
    prof *= u_bar / prof.mean()
    return torch.tensor(prof, dtype=torch.float32)


def wall_shear_from_near_wall(u1: np.ndarray, u2: np.ndarray, u3: np.ndarray) -> np.ndarray:
    """Wall-normal derivative ``du_x/dy`` at the wall (half-cell below row 1).

    Three-point one-sided Lagrange derivative for nodes at ``y = 1, 2, 3``
    evaluated at the wall ``y = 1/2``: ``f'(1/2) = -2 f(1) + 3 f(2) - f(3)``.
    Exact for quadratic profiles, i.e. the leading wall-shear error is
    ``O(dy^3)`` and no fitting window has to be chosen.
    """
    return -2.0 * np.asarray(u1) + 3.0 * np.asarray(u2) - np.asarray(u3)


def zero_crossing(values: np.ndarray, start: int = 0) -> float:
    """First ``- -> +`` zero crossing index (linear interpolation), or NaN."""
    v = np.asarray(values, dtype=np.float64)
    for i in range(max(start, 0), len(v) - 1):
        a, b = v[i], v[i + 1]
        if a <= 0.0 < b:
            return float(i + (0.0 - a) / (b - a))
        if a < 0.0 and b == 0.0:
            return float(i + 1)
    return float("nan")


def reattachment_from_wall_shear(
    tau_w: np.ndarray, x_step: int, s: int
) -> float:
    """Primary reattachment length ``X1/s`` from a wall-shear array.

    ``tau_w`` is indexed by column and covers the whole domain; the search starts
    at the step face ``x_step``.  Returns NaN when no reattachment is found inside
    the domain (the honest answer: the recirculation is not closed).
    """
    idx = zero_crossing(tau_w, start=x_step)
    if not math.isfinite(idx):
        return float("nan")
    return (idx - (x_step - 0.5)) / float(s)


def make_armaly_solid_mask(ny: int, nx: int, h_in: int, s: int, x_step: int) -> torch.Tensor:
    """Boolean solid mask ``(ny, nx)`` for the Armaly geometry.

    Fluid rows downstream of the step are ``1 .. ny-2`` (height ``H = h_in + s``),
    fluid rows in the inlet channel are ``s+1 .. ny-2`` (height ``h_in``); the
    walls therefore sit at half-cell positions ``y = 0.5`` and ``y = ny-1.5`` and
    the expansion ratio is exactly ``H / h_in``.
    """
    solid = torch.zeros((ny, nx), dtype=torch.bool)
    solid[-1, :] = True          # top wall
    solid[0, x_step:] = True     # downstream lower wall
    solid[0 : s + 1, :x_step] = True  # step block + upstream lower wall
    return solid


@dataclass
class ArmalyConfig:
    """Configuration of one Armaly-caliber BFS run."""

    re: float
    hi: int = 52                 # inlet channel height (cells)
    s: int = 49                  # step height (cells)
    u_bar: float = 0.1           # mean inlet velocity (lattice units)
    l_over_h: float = 14.0       # downstream length / outlet height
    up_over_hi: float = 4.0      # upstream length / inlet height
    n_steps: int = 90000         # hard step budget
    avg_from: float = 0.45       # start of the averaging window (fraction of budget)
    avg_stride: int = 200        # sampling stride inside the averaging window
    check_interval: int = 4000   # steadiness check interval
    steady_tol: float = 5e-3     # relative X1 drift below which the run stops
    steady_hits: int = 3
    collision: str = "trt"       # "trt" (magic Lambda = 3/16) or "bgk"
    sponge_width_h: float = 3.0  # sponge width in outlet heights
    seed: int = 0
    tag: str = ""

    def validate(self) -> None:
        if self.re <= 0 or self.u_bar <= 0:
            raise ValueError("re and u_bar must be > 0")
        if self.hi < 4 or self.s < 2:
            raise ValueError("hi >= 4 and s >= 2 required")
        if self.collision not in {"trt", "bgk"}:
            raise ValueError(f"unknown collision {self.collision!r}")
        if self.tau <= 0.5:
            raise ValueError(f"tau={self.tau:.4f} <= 0.5: lower u_bar or raise hi")

    @property
    def H(self) -> int:
        return self.hi + self.s

    @property
    def expansion_ratio(self) -> float:
        return self.H / float(self.hi)

    @property
    def nu(self) -> float:
        """Kinematic viscosity from Re = u_bar * D_h / nu, D_h = 2 * h_in."""
        return self.u_bar * 2.0 * self.hi / self.re

    @property
    def tau(self) -> float:
        return 0.5 + 3.0 * self.nu

    @property
    def x_step(self) -> int:
        return int(round(self.up_over_hi * self.hi))

    @property
    def nx(self) -> int:
        return self.x_step + int(round(self.l_over_h * self.H))

    @property
    def ny(self) -> int:
        return self.H + 2

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny

    def derived(self) -> dict:
        return {
            "H": self.H,
            "nx": self.nx,
            "ny": self.ny,
            "n_cells": self.n_cells,
            "x_step": self.x_step,
            "expansion_ratio": self.expansion_ratio,
            "expansion_ratio_deviation_pct": 100.0 * (self.expansion_ratio / ARMALY_EXPANSION_RATIO - 1.0),
            "nu": self.nu,
            "tau": self.tau,
            "d_h_inlet": 2 * self.hi,
            "u_max_inlet": 1.5 * self.u_bar,
            "mach_inlet": 1.5 * self.u_bar / math.sqrt(1.0 / 3.0),
            "reynolds": self.re,
        }


def _inlet_column_mask(ny: int, nx: int, s: int) -> torch.Tensor:
    mask = torch.zeros((9, ny, nx), dtype=torch.bool)
    mask[:, s + 1 : ny - 1, 0] = True
    return mask


def run_armaly_case(cfg: ArmalyConfig, out_dir: Path | None = None) -> dict:
    """Run one Armaly-caliber BFS case and return the measured result.

    The returned dictionary is JSON-serialisable and contains the configuration,
    the derived dimensionless table, the measured observable ``X1/s`` (wall-shear
    criterion), cross-check observables, convergence history and diagnostics.
    When ``out_dir`` is given, ``result.json``, ``wall_shear.csv`` and
    ``field.npz`` (time-averaged ``ux``) are written there.
    """
    cfg.validate()
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    nx, ny, s, hi = cfg.nx, cfg.ny, cfg.s, cfg.hi
    x_step = cfg.x_step
    solid = make_armaly_solid_mask(ny, nx, hi, s, x_step)
    solid3 = solid.unsqueeze(0)
    opp = OPPOSITE

    prof = parabolic_inlet_profile(hi, cfg.u_bar)
    inlet_full = torch.zeros(ny, dtype=torch.float32)
    inlet_full[s + 1 : ny - 1] = prof
    inlet_mask = _inlet_column_mask(ny, nx, s)

    rho = torch.ones((ny, nx))
    ux = torch.zeros((ny, nx))
    ux[s + 1 : ny - 1, :x_step] = prof.view(hi, 1)
    ux[1:ny - 1, x_step:] = cfg.u_bar * hi / float(cfg.H) * 1.5
    ux[solid] = 0.0
    uy = torch.zeros((ny, nx))
    f = equilibrium(rho, ux, uy)

    sw = max(4, int(round(cfg.sponge_width_h * cfg.H)))
    sigma = make_sponge_strength(ny, nx, nx - sw, sw, power=2.0,
                                 device=torch.device("cpu")).unsqueeze(0)

    if cfg.collision == "trt":
        collide = lambda f, tau: collide_trt(f, tau_plus=tau)  # noqa: E731
    else:
        collide = lambda f, tau: collide_bgk(f, tau=tau)  # noqa: E731
    tau = cfg.tau

    n_avg_start = int(cfg.avg_from * cfg.n_steps)
    acc = np.zeros((3, nx), dtype=np.float64)   # near-wall rows 1,2,3
    acc_field = np.zeros((ny, nx), dtype=np.float64)
    n_acc = 0
    history: list[dict] = []
    x1_check: list[float] = []
    steady_at = None
    t0 = time.time()
    step_done = 0

    for step in range(1, cfg.n_steps + 1):
        step_done = step
        fp = f
        fc = collide(f, tau=tau)
        fc = torch.where(solid3, fp[opp], fc)
        f = stream(fc)
        f = torch.where(inlet_mask, zou_he_inlet_velocity(f, inlet_full, 0.0), f)
        f = zou_he_outlet_pressure(f, 1.0)
        if n_acc > 0 or step > n_avg_start:
            rho_f, ux_f, uy_f = macroscopic(f)
            feq = equilibrium(rho_f, ux_f, uy_f)
            f = f - sigma * (f - feq)

        if step >= n_avg_start and step % cfg.avg_stride == 0:
            _, ux_now, _ = macroscopic(f)
            ux_now = ux_now.masked_fill(solid, 0.0)
            acc[0] += ux_now[1].numpy()
            acc[1] += ux_now[2].numpy()
            acc[2] += ux_now[3].numpy()
            acc_field += ux_now.numpy()
            n_acc += 1

        if step % cfg.check_interval == 0 or step == cfg.n_steps:
            _, ux_now, _ = macroscopic(f)
            ux_np = ux_now.masked_fill(solid, 0.0).numpy()
            tau_inst = wall_shear_from_near_wall(ux_np[1], ux_np[2], ux_np[3])
            x1_inst = reattachment_from_wall_shear(tau_inst, x_step, s)
            rec = {
                "step": step,
                "max_speed": float(torch.sqrt(ux_now.pow(2).sum()).max()),
                "mean_rho": float(macroscopic(f)[0].mean()),
                "x1_over_s_instant": x1_inst,
                "wall_time_s": time.time() - t0,
            }
            history.append(rec)
            if step > n_avg_start:
                x1_check.append(x1_inst)
                if len(x1_check) > cfg.steady_hits:
                    recent = x1_check[-cfg.steady_hits:]
                    if all(math.isfinite(v) for v in recent) and recent[0] > 0:
                        drift = max(abs(v - recent[0]) for v in recent) / recent[0]
                        if drift < cfg.steady_tol:
                            steady_at = step
                            break

    if n_acc == 0:
        raise RuntimeError("no averaging samples collected -- increase n_steps")

    ux_mean = acc_field / n_acc
    u1, u2, u3 = acc[0] / n_acc, acc[1] / n_acc, acc[2] / n_acc
    tau_mean = wall_shear_from_near_wall(u1, u2, u3)
    x1 = reattachment_from_wall_shear(tau_mean, x_step, s)
    x1_row1 = float("nan")
    zc = zero_crossing(u1, start=x_step)
    if math.isfinite(zc):
        x1_row1 = (zc - (x_step - 0.5)) / float(s)

    # upper-wall secondary bubble: shear at the top wall, sign flipped
    tau_top = np.zeros(nx)
    tau_top = -wall_shear_from_near_wall(ux_mean[ny - 2], ux_mean[ny - 3], ux_mean[ny - 4])

    _, ux_final, _ = macroscopic(f)
    rho_final = macroscopic(f)[0]

    # mass-flux conservation check over the *fluid* rows only (solid rows carry
    # bounce-back populations whose macroscopic velocity is not a flux).
    _rho_f, _ux_f, _ = macroscopic(f)

    def _flux(col: int, rows) -> float:
        return float((_rho_f[rows, col] * _ux_f[rows, col]).sum())

    out = {
        "case": "CASE-BACKWARD-STEP-LBM",
        "caliber": "Armaly et al. 1983: parabolic inlet, Re = u_bar*D_h/nu (D_h = 2*h_in), ER = H/h_in",
        "config": asdict(cfg),
        "derived": cfg.derived(),
        "result": {
            "x1_over_s": x1,
            "x1_over_s_row1": x1_row1,
            "x1_over_H": x1 / cfg.H if math.isfinite(x1) else float("nan"),
            "n_averaged_samples": n_acc,
            "average_window_steps": [n_avg_start, step_done],
            "steps_run": step_done,
            "steady_at_step": steady_at,
            "max_speed_final": float(torch.sqrt(ux_final.pow(2).sum()).max()),
            "rho_min": float(rho_final.min()),
            "rho_max": float(rho_final.max()),
            "mass_flux_inlet": _flux(0, slice(s + 1, ny - 1)),
            "mass_flux_outlet": _flux(nx - 1, slice(1, ny - 1)),
            "mass_flux_target": cfg.u_bar * hi,
            "mass_flux_rel_error": abs(_flux(0, slice(s + 1, ny - 1)) - _flux(nx - 1, slice(1, ny - 1)))
                                    / max(abs(cfg.u_bar * hi), 1e-30),
            "min_tau_w": float(np.nanmin(tau_mean[x_step:])),
            "max_tau_w": float(np.nanmax(tau_mean[x_step:])),
        },
        "convergence_history": history,
        "wall_time_s": time.time() - t0,
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.json").write_text(json.dumps(out, indent=2, default=str))
        xs = np.arange(x_step, nx)
        np.savetxt(
            out_dir / "wall_shear.csv",
            np.column_stack([xs, tau_mean[x_step:], tau_top[x_step:], u1[x_step:]]),
            delimiter=",", header="x,tau_w_bottom,tau_w_top,u1", comments="",
        )
        np.savez_compressed(out_dir / "field.npz", ux_mean=ux_mean.astype(np.float32),
                            solid=solid.numpy(), x_step=x_step, s=s, hi=hi)
    return out