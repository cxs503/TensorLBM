"""D3Q27 LBM run + comparison harness for the *classic* (z-dependent) 3-D
Taylor–Green vortex against the exact time-Taylor series reference.

This module is the measuring instrument of ``CASE-TAYLOR-GREEN-3D`` round 3:
the reference comes from :mod:`tensorlbm.tg3d_series` (a polynomial in ``t``
with machine-precision coefficients), the solver is the plain periodic D3Q27
BGK path already exercised by :mod:`tensorlbm.d3q27_validation`, and every
reported number is produced by :func:`compare_to_series` — no hand-typed
values.

Parameterisation (all in lattice units, ``Δx = Δt = 1``)
-------------------------------------------------------
The continuum problem is fixed by two dimensionless groups so that the *same*
flow is solved on every grid:

* ``T(t) = A k t`` with ``k = 2π/N`` — eddy-turnover time (``A`` = initial
  velocity amplitude);
* ``Re = A N / ν`` — Reynolds number based on the box length ``N``.

For a given ``N``, fixing ``Re`` and ``A`` fixes ``ν = A N / Re`` and the step
count ``t_end = T_end / (A k)``.

Honest notes
------------
* The initial distribution is the *equilibrium* of the analytic velocity field,
  so the first steps carry an O(Ma²) acoustic transient; it is part of the
  measured error, not subtracted.
* BGK only (no MRT/cumulant); one τ per grid.
* Everything is ``float64``; mass drift is monitored and reported.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from .d3q27 import collide_bgk27, equilibrium27, macroscopic27, stream27
from .tg3d_series import (  # noqa: F401  (re-exported for convenience)
    TG3DSeries,
    taylor_green_classic,
    taylor_green_extruded,
    vorticity_spectral,
)

__all__ = [
    "compare_to_series",
    "lattice_ic_fields",
    "run_tg3d_lbm",
    "run_tg3d_lbm_ladder",
    "series_reference",
    "tg3d_parameters",
]


def tg3d_parameters(n: int, *, amplitude: float = 0.05, reynolds: float = 160.0) -> dict[str, float]:
    """Lattice parameters for grid ``n`` at fixed ``(amplitude, Re)``."""
    k = 2.0 * math.pi / n
    nu = amplitude * n / reynolds
    return {
        "n": float(n),
        "amplitude": float(amplitude),
        "reynolds": float(reynolds),
        "k": k,
        "nu": nu,
        "tau": 3.0 * nu + 0.5,
        "eddy_time_per_step": amplitude * k,   # dT/dt
        "ma": amplitude * math.sqrt(3.0),
    }


def lattice_ic_fields(n: int, amplitude: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classic Taylor–Green initial velocity on the ``n³`` lattice (z, y, x)."""
    xi = 2.0 * math.pi * np.arange(n) / n
    zz, yy, xx = np.meshgrid(xi, xi, xi, indexing="ij")
    u = amplitude * np.sin(xx) * np.cos(yy) * np.cos(zz)
    v = -amplitude * np.cos(xx) * np.sin(yy) * np.cos(zz)
    w = np.zeros_like(u)
    return u, v, w


def series_reference(
    n: int,
    *,
    amplitude: float = 0.05,
    reynolds: float = 160.0,
    order: int = 16,
    nfft: int | None = None,
) -> TG3DSeries:
    """Build the exact series reference for the lattice problem of size ``n``."""
    p = tg3d_parameters(n, amplitude=amplitude, reynolds=reynolds)
    return TG3DSeries(
        taylor_green_classic(amplitude),
        box=float(n),
        nu=p["nu"],
        order=order,
        nfft=nfft,
    )


def _energy(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
    return 0.5 * float(np.mean(u * u + v * v + w * w))


def _enstrophy(u: np.ndarray, v: np.ndarray, w: np.ndarray, box: float) -> float:
    wx, wy, wz = vorticity_spectral(u, v, w, box)
    return 0.5 * float(np.mean(wx * wx + wy * wy + wz * wz))


def run_tg3d_lbm(
    n: int,
    *,
    amplitude: float = 0.05,
    reynolds: float = 160.0,
    eddy_times: float = 1.0,
    sample_every: int | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Any]:
    """Run the classic Taylor–Green vortex on a periodic ``n³`` D3Q27 lattice."""
    p = tg3d_parameters(n, amplitude=amplitude, reynolds=reynolds)
    steps = max(2, int(round(eddy_times / p["eddy_time_per_step"])))
    sample_every = sample_every or max(1, steps // 10)
    tau = p["tau"]

    u0, v0, w0 = lattice_ic_fields(n, amplitude)
    t = {k: torch.tensor(a, dtype=dtype) for k, a in (("u", u0), ("v", v0), ("w", w0))}
    rho = torch.ones((n, n, n), dtype=dtype)
    f = equilibrium27(rho, t["u"], t["v"], t["w"])

    def sample(step: int) -> dict[str, float]:
        r, vx, vy, vz = macroscopic27(f)
        uu, vv, ww = vx.numpy(), vy.numpy(), vz.numpy()
        return {
            "step": float(step),
            "time": float(step),
            "eddy_time": float(step * p["eddy_time_per_step"]),
            "energy": _energy(uu, vv, ww),
            "enstrophy": _enstrophy(uu, vv, ww, float(n)),
            "rho_min": float(r.min()),
            "rho_max": float(r.max()),
            "u_rms": float(np.sqrt(np.mean(uu * uu + vv * vv + ww * ww))),
        }

    history = [sample(0)]
    for step in range(1, steps + 1):
        f = collide_bgk27(f, tau)
        f = stream27(f)
        if step % sample_every == 0 or step == steps:
            history.append(sample(step))

    _, vx, vy, vz = macroscopic27(f)
    return {
        "n": n,
        "steps": steps,
        "tau": tau,
        "nu": p["nu"],
        "ma": p["ma"],
        "reynolds": p["reynolds"],
        "amplitude": amplitude,
        "eddy_times": eddy_times,
        "sample_every": sample_every,
        "history": history,
        "final": {"u": vx.numpy(), "v": vy.numpy(), "w": vz.numpy()},
    }


def compare_to_series(
    run: dict[str, Any],
    ser: TG3DSeries,
    *,
    order: int | None = None,
    band_limit: bool = False,
) -> list[dict[str, float]]:
    """Per-sample comparison of the LBM run against the exact series.

    Reported per sample: kinetic energy and enstrophy ratios versus the exact
    values, the volume-averaged ``L2`` velocity error normalised by the exact
    rms velocity, and the *vorticity* L2 error (the quantity that carries the
    3-D vortex-stretching signal).
    """
    n = int(run["n"])
    # energies are modal sums (no grid involved) -> full series order is fine
    rows: list[dict[str, float]] = []
    for h in run["history"]:
        t = h["time"]
        e_ex = ser.energy(t)
        o_ex = ser.enstrophy(t)
        row = {
            "step": h["step"],
            "time": t,
            "eddy_time": h["eddy_time"],
            "energy_lbm": h["energy"],
            "energy_series": e_ex,
            "energy_ratio_lbm": h["energy"] / run["history"][0]["energy"],
            "energy_ratio_series": e_ex / ser.energy(0.0),
            "energy_rel_err_pct": 100.0 * (h["energy"] / e_ex - 1.0),
            "enstrophy_lbm": h["enstrophy"],
            "enstrophy_series": o_ex,
            "enstrophy_ratio_lbm": h["enstrophy"] / run["history"][0]["enstrophy"],
            "enstrophy_ratio_series": o_ex / ser.enstrophy(0.0),
            "enstrophy_rel_err_pct": 100.0 * (h["enstrophy"] / o_ex - 1.0),
        }
        rows.append(row)
    # field-level comparison at the final state only (needs the fields)
    t_end = run["history"][-1]["time"]
    # the field is sampled on the n³ lattice: modes above Nyquist would alias,
    # so the truncation order is capped (truncation level is reported)
    field_order = None if order is None else int(order)
    ue, ve, we = ser.velocity_field(t_end, n=n, order=field_order,
                                    band_limit=band_limit)
    field_order = ser.last_field_order
    ub = run["final"]["u"]
    vb = run["final"]["v"]
    wb = run["final"]["w"]
    num = math.sqrt(float(np.mean((ub - ue) ** 2 + (vb - ve) ** 2 + (wb - we) ** 2)))
    den = math.sqrt(float(np.mean(ue * ue + ve * ve + we * we)))
    wxr = vorticity_spectral(ue, ve, we, float(n))
    wxl = vorticity_spectral(ub, vb, wb, float(n))
    num_w = math.sqrt(
        float(np.mean(sum((wxl[i] - wxr[i]) ** 2 for i in range(3))))
    )
    den_w = math.sqrt(float(np.mean(sum(wxr[i] ** 2 for i in range(3)))))
    rows[-1].update(
        {
            "field_order": float(field_order),
            "u_l2_rel_err_pct": 100.0 * num / den if den else float("nan"),
            "vorticity_l2_rel_err_pct": 100.0 * num_w / den_w if den_w else float("nan"),
            "u_l2_abs": num,
            "u_l2_ref": den,
        }
    )
    return rows


def run_tg3d_lbm_ladder(
    sizes: tuple[int, ...] = (24, 32, 48),
    *,
    amplitude: float = 0.05,
    reynolds: float = 160.0,
    eddy_times: float = 1.0,
) -> dict[str, Any]:
    """Same continuum problem on several grids (fixed ``Re``, ``A``, ``T_end``)."""
    out: dict[str, Any] = {"sizes": list(sizes), "amplitude": amplitude, "reynolds": reynolds,
                           "eddy_times": eddy_times, "grids": {}}
    for n in sizes:
        ser = series_reference(n, amplitude=amplitude, reynolds=reynolds)
        run = run_tg3d_lbm(n, amplitude=amplitude, reynolds=reynolds, eddy_times=eddy_times)
        rows = compare_to_series(run, ser)
        out["grids"][str(n)] = {
            "n": n,
            "parameters": tg3d_parameters(n, amplitude=amplitude, reynolds=reynolds),
            "steps": run["steps"],
            "series_order": ser.order,
            "series_nfft": ser.nfft,
            "series_truncation": ser.relative_tail(run["history"][-1]["time"]),
            "rows": rows,
        }
    return out