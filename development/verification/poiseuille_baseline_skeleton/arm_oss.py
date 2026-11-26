"""OSS arm of ``XVAL-LBM-SINGLE-PHASE-POISEUILLE-BASELINE-SKELETON``.

Executes the **reference implementation itself** (lettuce ``PoiseuilleFlow2D``
baseline skeleton, the object behind ``examples/simple_flows/Poiseuille.ipynb``)
on CPU, in double precision, with no reporter/VTK output, and extracts the
developed cross-section profile.

Provenance of every number that is *not* written here but taken from the OSS
code (paths are the checkout used for the run, recorded in the result JSON):

* ``lettuce/ext/_flows/poiseuille.py``  ``make_units``        -> L_lu = resolution-1
* ``lettuce/_unit.py``                  ``UnitConversion``    -> nu_lu, tau_lu, a_lu
* ``lettuce/ext/_flow.py`` / ``_simulation.py``               -> half-way bounce-back
  walls on axis 1 (``post_boundaries`` mask ``[:, [0, -1]]``), periodic in the
  streamwise direction, ``ShanChen`` force (``u_eq = u + tau*a/rho``).
* ``lettuce/ext/_flows/poiseuille.py``  ``analytic_solution``  -> the OSS internal
  reference; used only as a cross-check, the primary reference is the in-repo
  independent parabola.

Nothing is re-derived from memory: the lattice velocity/viscosity/acceleration are
read back from ``flow.units`` at run time and written into the result JSON.
"""

from __future__ import annotations

import os
import time
from typing import Dict

import numpy as np

from tensorlbm.verification.xval.poiseuille_skeleton import (
    analytic_profile,
    fit_effective_height,
    profile_kpis,
    wall_normal_coordinates,
)

CS_LU = 1.0 / np.sqrt(3.0)


def oss_path() -> str:
    return os.environ.get("LETTUCE_PATH", "/tmp/up/lettuce")


def available() -> bool:
    import sys

    p = oss_path()
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
    try:
        import lettuce  # noqa: F401
    except Exception:
        return False
    return True


def version() -> str:
    try:
        import lettuce

        return getattr(lettuce, "__version__", "unknown")
    except Exception:
        return "unavailable"


def run(spec: dict, steady_tol: float = 1e-6, window: int = 200,
        wall_cap_s: float = 900.0) -> Dict:
    import sys

    from cases import dimensionless  # noqa: WPS433

    p = oss_path()
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
    import torch
    import lettuce as lt

    res = int(spec["resolution"])
    nu_target = float(spec["viscosity_lu"])
    g_target = float(spec["acceleration_lu"])
    H = float(res - 2)
    y, _ = wall_normal_coordinates(res)

    dtype = torch.float32 if spec.get("dtype") == "float32" else torch.float64
    context = lt.Context(device=torch.device("cpu"), dtype=dtype)

    # The OSS skeleton fixes Re and Ma and lets nu follow from the resolution.
    # To reach this arm's *target* nu_lu at fixed (res, Ma) we invert the same
    # relation (nu_lu = (res-1) cs Ma / Re) instead of overriding internals.
    mach = float(spec["mach_number"])
    re_eff = (res - 1) * CS_LU * mach / nu_target
    flow = lt.PoiseuilleFlow2D(
        context=context,
        resolution=res,
        reynolds_number=re_eff,
        mach_number=mach,
    )
    nu_lu = float(flow.units.viscosity_lu)
    if abs(nu_lu - nu_target) > 1e-12 * abs(nu_target):
        raise RuntimeError(f"OSS unit inversion failed: nu={nu_lu!r} != {nu_target!r}")

    acc_raw = float(flow.units.convert_acceleration_to_lu(flow.acceleration)[0])
    force_scale = g_target / acc_raw
    acc_lu = context.convert_to_tensor([g_target, 0.0])
    force = lt.ShanChen(flow=flow, tau=flow.units.relaxation_parameter_lu, acceleration=acc_lu)
    collision = lt.BGKCollision(tau=flow.units.relaxation_parameter_lu, force=force)
    sim = lt.Simulation(flow=flow, collision=collision, reporter=[])
    u_scale = float(flow.units.characteristic_velocity_lu)

    def to_lu(u_pu):
        """OSS unit conversion (lettuce UnitConversion.convert_velocity_to_lu)."""
        return np.asarray(flow.units.convert_velocity_to_lu(u_pu), dtype=float)

    def profile_lu() -> np.ndarray:
        # u_pu has shape (2, x, y); the wall-normal axis is 1 (walls at [:,0]/[:,-1]).
        # The wall rows are dropped: they belong to the bounce-back boundary, not to
        # the discrete solution of the ODE (see predeclared.json "exclusion").
        u_pu = context.convert_to_ndarray(flow.u_pu)[0]
        return to_lu(u_pu[:, 1:-1]).mean(axis=0)             # -> lattice units

    t0 = time.time()
    prev = profile_lu()
    steps = 0
    converged = False
    truncated = False
    change = float("inf")
    while steps < int(spec["max_steps"]):
        sim(window)
        steps += window
        prof = profile_lu()
        change = float(np.max(np.abs(prof - prev)) / max(np.max(np.abs(prof)), 1e-300))
        prev = prof.copy()
        if change < steady_tol:
            converged = True
            break
        if time.time() - t0 > wall_cap_s:
            truncated = True
            break
    wall = time.time() - t0

    prof = profile_lu()
    u_field = to_lu(context.convert_to_ndarray(flow.u_pu)[0])
    rho = np.asarray(context.convert_to_ndarray(flow.rho()))
    if rho.ndim == 3:                     # lettuce may return (1, x, y)
        rho = rho[0]
    u_ex = analytic_profile(y, H, g_target, nu_target)
    kpis = profile_kpis(y, prof, u_ex)
    h_eff = fit_effective_height(y, prof, g_target, nu_target)

    # OSS internal reference (its own analytic_solution), for cross-check only.
    try:
        _p, u_ana = flow.analytic_solution()
        oss_ana_pu = context.convert_to_ndarray(u_ana.double() if dtype == torch.float32 else u_ana)[0]
        oss_ana_lu = to_lu(oss_ana_pu).mean(axis=0)
        oss_ana_err = float(np.max(np.abs(oss_ana_lu[1:-1] - u_ex)) / np.max(np.abs(u_ex)))
    except Exception as exc:      # pragma: no cover - diagnostic only
        oss_ana_err = float("nan")
        prof_note = str(exc)
    else:
        prof_note = ""

    return {
        "code": "lettuce",
        "label": spec["label"],
        "family": spec["family"],
        "resolution": res,
        "steps": steps,
        "converged": bool(converged),
        "wall_cap_hit": bool(truncated),
        "wall_s": round(wall, 2),
        "steady_change": change,
        "y": y.tolist(),
        "u_profile": prof.tolist(),
        "u_field": u_field.tolist(),
        "rho_min": float(rho[:, 1:-1].min()),
        "rho_max": float(rho[:, 1:-1].max()),
        "u_centerline": float(prof[len(prof) // 2]),
        "h_eff_mean": float(np.mean(h_eff)),
        "h_eff_max_dev": float(np.max(np.abs(h_eff - H))),
        "kpis": kpis,
        "dimensionless": dimensionless(spec),
        "params": dict(spec),
        "q_continuum": float(g_target * H ** 3 / (12.0 * nu_target)),
        "oss_meta": {
            "lettuce_version": getattr(lt, "__version__", "unknown"),
            "checkout": p,
            "dtype": str(dtype).replace("torch.", ""),
            "reynolds_number_effective": re_eff,
            "nu_lu": nu_lu,
            "tau_lu": float(flow.units.relaxation_parameter_lu),
            "acceleration_lu": acc_lu.tolist(),
            "acceleration_lu_reported_by_oss": acc_raw,
            "force_scale_effective": force_scale,
            "characteristic_velocity_lu": u_scale,
            "characteristic_length_lu": float(flow.units.characteristic_length_lu),
            "u_pu_peak": float(context.convert_to_ndarray(flow.u_pu)[0].max()),
            "oss_internal_analytic_err": oss_ana_err,
            "note": prof_note,
        },
    }