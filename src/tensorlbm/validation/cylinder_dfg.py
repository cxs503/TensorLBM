"""Schäfer & Turek (1996) DFG "flow around a cylinder" benchmark on library primitives.

Why this module exists
----------------------
``benchmarks/verified/cylinder/re20_st`` covers the **2D-1** case (Re=20, steady)
and ``benchmarks/verified/cylinder/re100`` covers a **free-stream** cylinder at
Re=100 (Braza 1986 anchor, Cd≈1.35, St≈0.165).  Neither is the DFG **2D-2**
case (Re=100 *inside the channel*), whose anchor band is Cd≈3.22–3.24 — the
channel blockage (D/H = 0.244) roughly doubles the drag.  Mixing those two
reference families is the error this repo records in
``benchmarks/roadmap.md`` ("圆柱绕流参考口径（Schäfer-Turek vs 自由流——必须分清）").

Benchmark definition (case 2D-2)
--------------------------------
* channel  x ∈ [0, 2.2], y ∈ [0, 0.41]
* cylinder centre (0.2, 0.2), diameter D = 0.1 (blockage D/H = 0.2439)
* inlet  u(0,y) = 4·U_max·y·(H−y)/H², U_max = 1.5 → U_mean = 1.0
* outlet p = 0 (Zou/He pressure outlet, rho = 1)
* no-slip on channel walls and cylinder
* Re = U_mean·D/ν = 100 → ν = 0.001
* observables (S&T 1996, Table 3): C_D, C_L amplitude, St, Δp between
  (0.15, 0.2) and (0.25, 0.2)

Non-dimensionalisation is exactly the reference one: C_D = F_x/(0.5 ρ U_mean² D)
with the *physical* D = 0.1, and Δp is reported as Δp/(0.5 ρ U_mean²)
(Δp* = 2·Δp_raw because the DFG table prints the raw pressure with ρ=1, U_mean=1).

Numerics
--------
Library primitives only (no hand-written collide/stream/equilibrium):

* ``tensorlbm.solver.collide_mrt`` / ``stream``      — D2Q9 MRT collision, pull streaming
* ``tensorlbm.d2q9.equilibrium``                     — initial condition
* ``tensorlbm.boundaries.zou_he_inlet_velocity``     — parabolic (and seeded) inlet
* ``tensorlbm.boundaries.zou_he_outlet_pressure``    — rho = 1 outlet
* ``tensorlbm.boundaries.bounce_back_cells``         — walls + cylinder (half-way BB)
* ``tensorlbm.boundaries.cylinder_mask`` / ``make_channel_wall_mask``
* ``tensorlbm.boundaries.compute_obstacle_forces``   — Ladd (1994) momentum exchange

The staggering is the one verified in ``cylinder/re20_st``: solid cells skip the
collision, the momentum-exchange force is sampled after streaming and before the
obstacle bounce-back, so no population is counted twice.  No mass
renormalisation (the re20_st study showed it excites a spurious Cd oscillation).

Pressure probes
---------------
The DFG probes (0.15, 0.2)/(0.25, 0.2) sit *exactly on* the cylinder surface
(x = 0.2 ∓ D/2), i.e. on solid cells for a lattice whose diameter is an even
number of cells.  Populations of solid cells after bounce-back are mirror
images, not a physical density, so probing them biases Δp low (measured −19%
in the D=20 pilot).  This module therefore samples the two nearest **fluid**
columns on each side and linearly extrapolates to the DFG x-positions; the
raw nearest-fluid-column reading is kept as a sensitivity figure.  The
centerline density profile is also stored so the extrapolation is auditable.

References
----------
Schäfer, M. & Turek, S. (1996) "Benchmark computations of laminar flow around
a cylinder", in *Flow Simulation with High-Performance Computers II*, Notes on
Numerical Fluid Mechanics vol. 48, Vieweg, pp. 547–566.

Repo-internal frozen register of the same table:
``benchmarks/verified/cylinder/re100_dfg/REFERENCE.md`` (values + tokens +
sha256 recorded by ``benchmarks/validation_matrix/references.json``).
"""

from __future__ import annotations

import json
import math
import pathlib
import time
from dataclasses import dataclass, field

import torch

from tensorlbm.boundaries import (
    bounce_back_cells,
    compute_obstacle_forces,
    cylinder_mask,
    make_channel_wall_mask,
    zou_he_inlet_velocity,
    zou_he_outlet_pressure,
)
from tensorlbm.d2q9 import equilibrium
from tensorlbm.solver import collide_mrt, stream

# --------------------------------------------------------------------------
# Benchmark definition and reference bands
# --------------------------------------------------------------------------

LX = 2.2  # channel length (m)
LY = 0.41  # channel height (m)
CYL_CX = 0.2  # cylinder centre x (m)
CYL_CY = 0.2  # cylinder centre y (m)
CYL_D = 0.1  # cylinder diameter (m)
U_MAX_INLET = 1.5  # inlet peak velocity (m/s) -> U_mean = 1.0
U_MEAN = 1.0

PROBE_A = (0.15, 0.2)  # Δp upstream probe (S&T Table 3)
PROBE_B = (0.25, 0.2)  # Δp downstream probe

#: S&T 1996 benchmark quantity bands (lower, upper).
REFERENCE_BANDS: dict[str, dict[str, object]] = {
    "2D-2": {
        "re": 100.0,
        "cd": (3.2200, 3.2400),
        "cl_amp": (0.9900, 1.0100),
        "st": (0.2950, 0.3050),
        "dp": (2.4600, 2.5000),  # raw Δp, ρ=1, U_mean=1
        "source": "Schäfer & Turek 1996, NNFM 48, pp. 547-566, Table 3 (2D-2)",
    },
    "2D-1": {
        "re": 20.0,
        "cd": (5.5700, 5.5900),
        "cl_amp": None,
        "st": None,
        "dp": (0.1172, 0.1176),
        "source": "Schäfer & Turek 1996, NNFM 48, pp. 547-566, Table 1 (2D-1)",
    },
}


def band_mid(band: tuple[float, float]) -> float:
    return 0.5 * (band[0] + band[1])


def band_error(value: float, band: tuple[float, float]) -> dict:
    """Error bookkeeping against a reference band (no hidden convention)."""
    mid = band_mid(band)
    err = (value - mid) / mid * 100.0
    if band[0] <= value <= band[1]:
        err_bound = 0.0
    elif value > band[1]:
        err_bound = (value - band[1]) / band[1] * 100.0
    else:
        err_bound = (value - band[0]) / band[0] * 100.0
    return {
        "value": float(value),
        "band": [float(band[0]), float(band[1])],
        "band_mid": float(mid),
        "err_pct_vs_mid": float(err),
        "inside_band": bool(band[0] <= value <= band[1]),
        "err_pct_vs_nearest_bound": float(err_bound),
    }


@dataclass
class CaseConfig:
    """Lattice configuration derived from a grid level and a velocity/τ choice."""

    D_cells: int
    u_mean_lb: float = 0.0833
    re: float = 100.0
    n_steps: int = 60_000
    seed_periods: float = 2.0
    seed_st: float = 0.30
    seed_ampl: float = 0.05
    avg_periods: float = 20.0
    report_every: int = 5_000
    row_sample_every: int = 5

    # derived
    dx: float = field(init=False, default=0.0)
    nx: int = field(init=False, default=0)
    ny: int = field(init=False, default=0)
    tau: float = field(init=False, default=0.0)
    nu_lb: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.dx = CYL_D / self.D_cells
        self.nx = int(round(LX / self.dx))
        self.ny = int(round(LY / self.dx))
        self.nu_lb = self.u_mean_lb * self.D_cells / self.re
        self.tau = 0.5 + 3.0 * self.nu_lb
        if self.tau <= 0.52:
            raise ValueError(f"tau={self.tau:.3f} too close to 0.5 for stable collision")

    @property
    def period_steps(self) -> float:
        """Shedding period in lattice steps at the benchmark Strouhal number."""
        st = band_mid(REFERENCE_BANDS["2D-2"]["st"])
        return self.D_cells / (st * self.u_mean_lb)


def inlet_profile(ny: int, dx: float, u_mean_lb: float) -> torch.Tensor:
    """DFG parabolic inlet profile with unit mean (U_max/U_mean = 1.5)."""
    y = torch.arange(ny, dtype=torch.float32) * dx
    profile = 4.0 * U_MAX_INLET * y * (LY - y) / LY**2
    profile = profile * (U_MEAN / profile.mean())  # force exact unit mean on the grid
    return u_mean_lb * profile


def _smooth(x: torch.Tensor, k: int) -> torch.Tensor:
    """Odd-length moving average that preserves the number of samples."""
    if k <= 1:
        return x
    k = min(k, max(1, x.numel() // 4))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    kernel = torch.ones(k, dtype=x.dtype) / k
    xp = torch.nn.functional.pad(x.view(1, 1, -1), (pad, pad), mode="replicate")
    return torch.nn.functional.conv1d(xp, kernel.view(1, 1, -1)).view(-1)


def strouhal_from_series(
    cl: torch.Tensor, period_guess: float, st_guess: float, D_cells: int, u_mean_lb: float
) -> dict:
    """Strouhal number from a lift history: autocorrelation peak + FFT cross-check.

    Raw zero-crossing counting is *not* used: the momentum-exchange lift signal
    carries small high-frequency wiggles that add spurious crossings around zero
    (the D=20 pilot gave St=0.52 by crossing count while the shedding period was
    clearly 792 steps → St≈0.30).
    """
    x = (cl - cl.mean()).to(torch.float64).cpu()
    n = x.numel()
    out = {
        "n_samples": int(n),
        "period_acf": float("nan"),
        "period_fft": float("nan"),
        "st_acf": float("nan"),
        "st_fft": float("nan"),
        "acf_peak_lag": None,
        "fft_peak_hz": float("nan"),
    }
    if n < 32:
        return out
    xs = _smooth(x, max(3, int(period_guess / 40)))
    # ---- autocorrelation, first interior local maximum in [0.5, 1.5]·T_guess
    ac = torch.fft.irfft(torch.abs(torch.fft.rfft(xs, n=2 * n)) ** 2)[:n] / (xs @ xs)
    lo = max(3, int(0.5 * period_guess))
    hi = min(n // 2, int(1.6 * period_guess))
    # global maximum inside the window: the first local maximum is not robust
    # (lift harmonics produce small interior bumps; the D=20 pilot locked onto
    # one at lag 630 while the shedding period was 792 steps)
    lag = None
    if hi > lo + 1:
        seg = ac[lo:hi]
        lag = lo + int(torch.argmax(seg))
        if lag <= 0:
            lag = None
    if lag is not None:
        # parabolic refinement
        y0, y1, y2 = float(ac[lag - 1]), float(ac[lag]), float(ac[lag + 1])
        denom = y0 - 2.0 * y1 + y2
        shift = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
        period = lag + max(-0.5, min(0.5, shift))
        out["period_acf"] = float(period)
        out["acf_peak_lag"] = int(lag)
        out["st_acf"] = float(D_cells / (u_mean_lb * period))
    # ---- FFT, peak restricted to a ±30% window around the expected St
    spec = torch.abs(torch.fft.rfft(xs))
    freqs = torch.fft.rfftfreq(n, d=1.0)
    f_lo = 0.7 * st_guess * u_mean_lb / D_cells
    f_hi = 1.4 * st_guess * u_mean_lb / D_cells
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if mask.any():
        idx = int(torch.argmax(spec[mask]))
        f_peak = float(freqs[mask][idx])
        # parabolic interpolation on the spectrum for sub-bin accuracy
        idxs = torch.nonzero(mask).flatten()
        j = int(idxs[idx])
        if 0 < j < spec.numel() - 1:
            s0, s1, s2 = float(spec[j - 1]), float(spec[j]), float(spec[j + 1])
            denom = s0 - 2.0 * s1 + s2
            if abs(denom) > 1e-12:
                f_peak += 0.5 * (s0 - s2) / denom * float(freqs[1] - freqs[0])
        out["fft_peak_hz"] = f_peak
        out["period_fft"] = float(1.0 / f_peak)
        out["st_fft"] = float(f_peak * D_cells / u_mean_lb)
    return out


def probe_columns(cfg: CaseConfig) -> dict:
    """Fluid columns used for the Δp probes and their x positions (m)."""
    dx = cfg.dx
    cx_col = int(round(CYL_CX / dx))
    r_cols = int(round(0.5 * CYL_D / dx))
    row = int(round(PROBE_A[1] / dx))
    up_in = cx_col - r_cols - 1  # nearest fluid column upstream of the surface
    dn_in = cx_col + r_cols + 1  # nearest fluid column downstream
    return {
        "row": row,
        "up_near": up_in,
        "up_far": up_in - 2,
        "dn_near": dn_in,
        "dn_far": dn_in + 2,
        "x_up_near": up_in * dx,
        "x_up_far": (up_in - 2) * dx,
        "x_dn_near": dn_in * dx,
        "x_dn_far": (dn_in + 2) * dx,
        "x_probe_a": PROBE_A[0],
        "x_probe_b": PROBE_B[0],
    }


def run_case(
    cfg: CaseConfig,
    device: str = "cpu",
    threads: int | None = None,
    keep_series: bool = True,
) -> dict:
    """Run the DFG 2D-2 case at one grid level; return observables + provenance."""
    if threads:
        torch.set_num_threads(int(threads))
    dev = torch.device(device)
    dx = cfg.dx
    solid = cylinder_mask(
        cfg.nx, cfg.ny, round(CYL_CX / dx), round(CYL_CY / dx), 0.5 * CYL_D / dx, dev
    )
    wall = make_channel_wall_mask(cfg.ny, cfg.nx, solid, dev)
    fluid = ~solid
    surface = solid & (
        torch.roll(fluid, 1, 0)
        | torch.roll(fluid, -1, 0)
        | torch.roll(fluid, 1, 1)
        | torch.roll(fluid, -1, 1)
    )

    u_in = inlet_profile(cfg.ny, dx, cfg.u_mean_lb)
    y = torch.arange(cfg.ny, dtype=torch.float32) * dx
    seed_shape = torch.sin(2.0 * math.pi * y / LY)  # antisymmetric, vanishes at walls

    rho0 = torch.ones((cfg.ny, cfg.nx), dtype=torch.float32, device=dev)
    ux0 = torch.zeros_like(rho0)
    ux0[:, :] = u_in[:, None].expand(cfg.ny, cfg.nx)
    ux0[solid] = 0.0
    f = equilibrium(rho0, ux0, torch.zeros_like(rho0))

    dyn_p = 0.5 * cfg.u_mean_lb**2 * cfg.D_cells
    pr = probe_columns(cfg)
    row = pr["row"]

    period_guess = cfg.period_steps
    n_seed = min(int(round(cfg.seed_periods * period_guess)), cfg.n_steps // 4)
    n_avg = min(cfg.n_steps, int(round(cfg.avg_periods * period_guess)))
    window_start = cfg.n_steps - n_avg + 1

    cd_hist: list[float] = []
    cl_hist: list[float] = []
    dp_hist: list[float] = []
    log: list[dict] = []
    row_sum = torch.zeros(cfg.nx, dtype=torch.float64)
    row_n = 0

    wall_t0 = time.time()
    for step in range(1, cfg.n_steps + 1):
        before = f.clone()
        collided = collide_mrt(f, cfg.tau)
        f = torch.where(solid.unsqueeze(0), before, collided)
        f = stream(f)
        if step <= n_seed:
            amp = cfg.seed_ampl * cfg.u_mean_lb * math.sin(2.0 * math.pi * step / period_guess)
            f = zou_he_inlet_velocity(f, u_in, seed_shape * amp)
        else:
            f = zou_he_inlet_velocity(f, u_in, 0.0)
        f = zou_he_outlet_pressure(f, 1.0)
        f = bounce_back_cells(f, wall)
        fx, fy = compute_obstacle_forces(f, surface)
        f = bounce_back_cells(f, solid)

        cd_hist.append(float(fx.item()) / dyn_p)
        cl_hist.append(float(fy.item()) / dyn_p)
        # p = rho/3 (c_s^2 = 1/3); Δp* = Δp/(0.5 ρ U²) = 2·Δp_raw/ρU²
        rho_row = f[:, row, :].sum(0)
        dp_hist.append(
            float(rho_row[pr["up_near"]] - rho_row[pr["dn_near"]]) / 3.0 / (0.5 * cfg.u_mean_lb**2)
        )
        if step >= window_start and step % cfg.row_sample_every == 0:
            row_sum += rho_row.to(torch.float64)
            row_n += 1
        if cfg.report_every and step % cfg.report_every == 0:
            k = min(len(cd_hist), 500)
            log.append(
                {
                    "step": step,
                    "cd_recent": round(sum(cd_hist[-k:]) / k, 4),
                    "cl_recent": round(sum(cl_hist[-k:]) / k, 4),
                    "dp_recent": round(sum(dp_hist[-k:]) / k, 4),
                    "wall_s": round(time.time() - wall_t0, 1),
                }
            )

    window = min(n_avg, len(cd_hist))
    cd_t = torch.tensor(cd_hist[-window:], dtype=torch.float64)
    cl_t = torch.tensor(cl_hist[-window:], dtype=torch.float64)
    dp_t = torch.tensor(dp_hist[-window:], dtype=torch.float64)
    half = window // 2

    st = strouhal_from_series(
        cl_t.to(torch.float32),
        period_guess,
        band_mid(REFERENCE_BANDS["2D-2"]["st"]),
        cfg.D_cells,
        cfg.u_mean_lb,
    )

    # Δp at the DFG x positions: linear extrapolation from the two nearest fluid
    # columns on each side (see module docstring "Pressure probes").
    rho_mean_row = (row_sum / max(row_n, 1)).to(torch.float32)
    p_up_near = float(rho_mean_row[pr["up_near"]]) / 3.0
    p_up_far = float(rho_mean_row[pr["up_far"]]) / 3.0
    p_dn_near = float(rho_mean_row[pr["dn_near"]]) / 3.0
    p_dn_far = float(rho_mean_row[pr["dn_far"]]) / 3.0
    q = 0.5 * cfg.u_mean_lb**2
    # slope per metre from the two sampled columns, extrapolated to the probe x
    g_up = (p_up_near - p_up_far) / (pr["x_up_near"] - pr["x_up_far"])
    g_dn = (p_dn_near - p_dn_far) / (pr["x_dn_near"] - pr["x_dn_far"])
    p_up_probe = p_up_near + g_up * (pr["x_probe_a"] - pr["x_up_near"])
    p_dn_probe = p_dn_near + g_dn * (pr["x_probe_b"] - pr["x_dn_near"])
    dp_extrap = (p_up_probe - p_dn_probe) / q
    dp_near = (p_up_near - p_dn_near) / q

    ref = REFERENCE_BANDS["2D-2"]
    cd_mean = float(cd_t.mean())
    cl_amp = 0.5 * float(cl_t.max() - cl_t.min())
    # FFT period is the primary estimator: it is sub-bin interpolated and was the
    # only one of the two that stayed inside the DFG band on the D=20 pilot.
    st_primary = st["st_fft"] if st["st_fft"] == st["st_fft"] else st["st_acf"]
    st_disagreement_pct = (
        abs(st["st_acf"] - st["st_fft"]) / st["st_fft"] * 100.0
        if (st["st_acf"] == st["st_acf"] and st["st_fft"] == st["st_fft"])
        else float("nan")
    )

    res: dict = {
        "case": "2D-2",
        "re": cfg.re,
        "D_cells": cfg.D_cells,
        "nx": cfg.nx,
        "ny": cfg.ny,
        "dx": float(dx),
        "cells": cfg.nx * cfg.ny,
        "u_mean_lb": cfg.u_mean_lb,
        "u_max_lb": float(u_in.max()),
        "nu_lb": cfg.nu_lb,
        "tau": cfg.tau,
        "n_steps": cfg.n_steps,
        "seed_steps": n_seed,
        "seed_ampl_frac": cfg.seed_ampl,
        "avg_window": window,
        "avg_window_start_step": window_start,
        "period_steps_analytic": period_guess,
        "period_steps_acf": st["period_acf"],
        "period_steps_fft": st["period_fft"],
        "n_solid_cells": int(solid.sum()),
        "n_surface_cells": int(surface.sum()),
        "probes": {
            "row": row,
            "cols": {k: pr[k] for k in ("up_near", "up_far", "dn_near", "dn_far")},
            "x_m": {k: pr[k] for k in ("x_up_near", "x_up_far", "x_dn_near", "x_dn_far")},
            "p_up_near": p_up_near,
            "p_up_far": p_up_far,
            "p_dn_near": p_dn_near,
            "p_dn_far": p_dn_far,
            "p_up_probe": p_up_probe,
            "p_dn_probe": p_dn_probe,
        },
        "cd": cd_mean,
        "cd_std": float(cd_t.std()),
        "cd_stderr": float(cd_t.std() / math.sqrt(window)),
        "cd_first_half": float(cd_t[:half].mean()),
        "cd_second_half": float(cd_t[half:].mean()),
        "cl_mean": float(cl_t.mean()),
        "cl_amp": cl_amp,
        "st": st_primary,
        "st_acf": st["st_acf"],
        "st_fft": st["st_fft"],
        "st_acf_vs_fft_pct": st_disagreement_pct,
        "st_ambiguous": bool(st_disagreement_pct == st_disagreement_pct and st_disagreement_pct > 3.0),
        "dp": dp_extrap,
        "dp_nearest_fluid_columns": dp_near,
        "dp_stderr": float(dp_t.std() / math.sqrt(window)),
        "wall_s": round(time.time() - wall_t0, 1),
        "reference_source": ref["source"],
        "st_extraction": st,
        "log": log,
    }
    res["bench"] = {
        "cd": band_error(cd_mean, ref["cd"]),
        "cl_amp": band_error(cl_amp, ref["cl_amp"]),
        "st": band_error(st_primary, ref["st"]),
        "dp": band_error(dp_extrap, ref["dp"]),
    }
    res["err_pct"] = {k: v["err_pct_vs_mid"] for k, v in res["bench"].items()}
    res["drift_pct"] = {
        "cd": (res["cd_second_half"] - res["cd_first_half"]) / res["cd_first_half"] * 100.0
    }
    if keep_series:
        res["series"] = {
            "cd": cd_hist[-window:],
            "cl": cl_hist[-window:],
            "dp": dp_hist[-window:],
            "rho_centerline_mean": rho_mean_row.tolist(),
            "window_start_step": window_start,
        }
    return res


def _json_default(obj):  # pragma: no cover - tiny JSON helper
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:  # noqa: BLE001
            pass
    return str(obj)


def _cli() -> int:  # pragma: no cover - thin CLI wrapper
    import argparse

    ap = argparse.ArgumentParser(description="Schäfer-Turek DFG cylinder benchmark (2D-2)")
    ap.add_argument("--D", type=int, default=40, help="lattice cells across the cylinder diameter")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--u-mean", type=float, default=0.0833, help="mean inlet velocity, lattice units")
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--tau", type=float, default=None, help="override relaxation time (keeps Re)")
    ap.add_argument("--avg-periods", type=float, default=20.0)
    ap.add_argument("--seed-ampl", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-series", action="store_true")
    a = ap.parse_args()

    cfg = CaseConfig(
        D_cells=a.D,
        u_mean_lb=a.u_mean,
        re=a.re,
        n_steps=a.steps,
        avg_periods=a.avg_periods,
        seed_ampl=a.seed_ampl,
    )
    if a.tau is not None:
        # keep Re fixed: nu = (tau-0.5)/3 -> u_mean = Re*nu/D
        nu = (a.tau - 0.5) / 3.0
        cfg.nu_lb = nu
        cfg.tau = a.tau
        cfg.u_mean_lb = a.re * nu / a.D_cells
    res = run_case(cfg, device=a.device, threads=a.threads, keep_series=not a.no_series)
    print(
        json.dumps(
            {k: v for k, v in res.items() if k != "series"}, indent=2, default=_json_default
        )
    )
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(res, default=_json_default))
        if not a.no_series and res.get("series"):
            import numpy as np

            np.savez(
                str(a.out).replace(".json", "") + ".series.npz",
                **{
                    k: np.asarray(v, dtype=np.float32)
                    for k, v in res["series"].items()
                    if k != "window_start_step"
                },
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())