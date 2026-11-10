"""Standalone verification of the two numerical ingredients of the MRT face.

Two experiments, both cheap and both reported as numbers in the campaign JSON:

``shear_wave``
    A transverse shear wave in the (torch) MRT implementation decays at
    ``exp(-nu k^2 t)``.  The measured decay rate is compared with
    ``nu = cs2 (tau - 1/2)``; this is the definition-level check that the
    stress-mode relaxation rate really is the shear viscosity.
``poiseuille``
    The native D2Q9 kernel is driven by a constant body force in a periodic
    channel; the steady state is parabolic, so ``nu_eff = g h^2/(2 u_max)``
    measures the viscosity the kernel actually uses.  Both the peak-value route
    and a whole-profile least-squares fit are reported; the two converge at the
    same O(h^2) rate, which is what proves the residual is discretisation and
    not a wrong relaxation rate (measured: 0.11-0.40% at h = 11).  This is the check that the C kernel and the reference
    Python operator agree.

Nothing here is a fit to a reference: both targets are analytic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import native
from .mrt import MRTModel

__all__ = ["shear_wave_decay", "poiseuille_viscosity", "run_verification"]


def shear_wave_decay(tau: float, *, ny: int = 16, nx: int = 2, steps: int = 600,
                     amplitude: float = 1e-4, s_energy: float = 1.2,
                     s_ghost: float = 1.0) -> dict:
    """Measure the decay rate of a shear wave evolved with the MRT operator."""
    import torch

    from .lattice import D2Q9

    lat = D2Q9
    k = 2.0 * np.pi / ny
    y = np.arange(ny)
    u = np.zeros((ny, nx, 2))
    u[:, :, 0] = amplitude * np.sin(k * y)[:, None]
    f = torch.as_tensor(lat.equilibrium(np.ones((ny, nx)), u))
    model = MRTModel("D2Q9", tau=tau, s_energy=s_energy, s_ghost=s_ghost)
    cx = torch.as_tensor(lat.c[:, 0]).view(9, 1, 1)
    times, amps = [], []
    for n in range(steps):
        f = model.collide(f)
        f = model.stream(f)
        if n % 20 == 0:
            rho = f.sum(0)
            ux = (cx * f).sum(0) / rho
            times.append(float(n))
            amps.append(float(abs(np.sum(ux[:, 0].numpy() * np.sin(k * y))) * 2.0 / ny))
    times = np.asarray(times)
    amps = np.asarray(amps)
    # keep the window where the signal is well above round-off
    keep = amps > 1e-3 * amps[0]
    slope = float(np.polyfit(times[keep], np.log(amps[keep]), 1)[0])
    nu_measured = -slope / k**2
    nu_exact = (tau - 0.5) / 3.0
    return {
        "tau": tau,
        "nu_exact": nu_exact,
        "nu_measured": nu_measured,
        "rel_error": abs(nu_measured - nu_exact) / nu_exact,
        "amplitude_initial": float(amps[0]),
        "amplitude_final": float(amps[-1]),
        "window_steps": int(times[keep].max()),
    }


def poiseuille_viscosity(tau: float, *, ny: int = 24, nx: int = 4, u_peak: float = 0.02,
                         threads: int | None = None, build_dir=None) -> dict:
    """Measure the native kernel viscosity from a forced periodic channel."""
    solid = np.zeros((ny, nx), dtype=np.int32)
    solid[0, :] = 1
    solid[-1, :] = 1
    no_body = np.zeros((ny, nx), dtype=np.int32)
    h = (ny - 2) / 2.0
    nu = (tau - 0.5) / 3.0
    g = u_peak * 2.0 * nu / h**2
    steps = int(max(6000, 6 * h * h / nu))
    prefix = Path(native._build_dir(build_dir)) / f"poiseuille_{tau}"
    native.run_d2q9(
        out_prefix=prefix, nx=nx, ny=ny, solid=solid, body=no_body,
        steps=steps, warmup=steps - 2000, u_in=0.0, tau=tau, body_force=g,
        periodic_x=True, sample_every=100, threads=threads, build_dir=build_dir,
        force_every=max(steps - 1000, 1),
    )
    mean = native.read_mean_field(prefix, nx, ny)
    ux = mean["ux"][:, nx // 2]
    u_max = float(np.nanmax(ux))
    # Unbiased estimator: least-squares fit of the *whole* discrete solution
    # u(y) = A (y - y_w)(y_1 - y) -> nu = g / (2 A).  Using only the peak cell
    # value is biased by ~2% because the maximum falls between cell centres.
    yw, y1 = 0.5, ny - 1.5
    yy = np.arange(1, ny - 1, dtype=float)
    basis = (yy - yw) * (y1 - yy)
    good = np.isfinite(ux[: basis.size])
    A = float(np.dot(basis[good], ux[: basis.size][good]) / np.dot(basis[good], basis[good]))
    nu_fit = g / (2.0 * A) if A > 0 else float("nan")
    nu_eff = g * h**2 / (2.0 * u_max)
    return {
        "tau": tau,
        "nu_exact": nu,
        "nu_measured": nu_eff,
        "nu_measured_profile_fit": nu_fit,
        "rel_error": abs(nu_eff - nu) / nu,
        "rel_error_profile_fit": abs(nu_fit - nu) / nu,
        "u_peak_target": u_peak,
        "u_peak_measured": u_max,
        "steps": steps,
    }


def run_verification(*, threads: int | None = None, build_dir=None) -> dict:
    taus = (0.53, 0.60, 0.80, 1.05)
    shear = [shear_wave_decay(t) for t in (0.55, 0.60, 0.80, 1.05)]
    poiseuille = [poiseuille_viscosity(t, threads=threads, build_dir=build_dir) for t in taus]
    return {
        "shear_wave": shear,
        "poiseuille": poiseuille,
        "max_rel_error_shear_wave": max(s["rel_error"] for s in shear),
        "max_rel_error_poiseuille": max(p["rel_error"] for p in poiseuille),
    }


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run_verification(), indent=2))