"""IN32-L01 wetting verification ladder: case builders + acceptance gates.

Six cases, all on the D2Q9 CAC phase-field solver (:mod:`tensorlbm.cac_lbm2d`):

===============  ===========================================================
case             analytic anchor
===============  ===========================================================
``sigma_flat``   planar interface tension ``sigma = int(f + k/2 |grad phi|^2)``
``laplace``      Young-Laplace  ``dp = sigma / R``  (2D drop, 3% gate)
``contact``      Young's law    ``cos(theta_Y) = h / sigma_lg``  (3 deg gate)
``jurin``        Jurin          ``h = 2 sigma cos(theta) / (drho g w)`` (5%)
``lamb``         Lamb/Rayleigh  ``omega^2 = m(m^2-1) sigma / ((rho_l+rho_g) R^3)``
``spurious``     spurious-current gate ``max|u| < 1e-4 U_ref``
``cox_voinov``   ``theta_d^3 = theta_e^3 + 9 Ca ln(L/Ls)`` (Jurin transient)
===============  ===========================================================

Every case returns a plain dict that is JSON-serialisable and carries its own
inputs, raw measurements, the gate value and the pass flag, so the report can
inject numbers without hand copying.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .. import cac_lbm2d as cac
from . import metrics as M
from .wall_order_parameter_2d import analytic_contact_angle, young_cos_theta

# ---------------------------------------------------------------------------
# Frozen base parameters (one place; the report reads them back from the JSON)
# ---------------------------------------------------------------------------

BASE = {
    "rho_l": 1.0,
    "rho_g": 0.1,
    "mu_l": 0.1,
    "mu_g": 0.01,
    "width": 4.0,
    "mobility": 0.05,
    "mu_law": "harmonic",
    "p0": 0.01,
    "dtype": "float32",
    "threads": 1,
}

GATES = {
    "laplace_pct": 3.0,
    "contact_deg": 3.0,
    "jurin_pct": 5.0,
    "lamb_pct": 3.0,
    "spurious_ratio": 1e-4,
    "n_pytest": 15,
}


def _sim(shape, *, sigma, wall_mask=None, policy="freeze", angle=None, gravity=(0.0, 0.0),
         gravity_mode="uniform", axes="pp", mu_l=None, mu_g=None, **over):
    kw = dict(
        rho_l=BASE["rho_l"], rho_g=BASE["rho_g"],
        mu_l=BASE["mu_l"] if mu_l is None else mu_l,
        mu_g=BASE["mu_g"] if mu_g is None else mu_g,
        sigma=sigma, width=BASE["width"], mobility=BASE["mobility"],
        mu_law=BASE["mu_law"], p0=BASE["p0"], gravity=gravity, gravity_mode=gravity_mode,
        axes=axes, wall_mask=wall_mask, wall_phi_policy=policy, contact_angle_deg=angle,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kw.update(over)
    return cac.CAC2D(shape, **kw)


def _numpy(sim):
    return cac.macro(sim)


# ---------------------------------------------------------------------------
# Case 1: planar interface tension
# ---------------------------------------------------------------------------


def sigma_flat(sigma: float = 0.02, shape=(64, 8), steps: int = 4000) -> dict:
    """Planar interface tension: a periodic liquid slab carries two flat
    interfaces, so ``sigma = 0.5 * int (f + k/2 |grad phi|^2) dy``."""
    t0 = time.time()
    sim = _sim(shape, sigma=sigma)
    ny = shape[0]
    sim.initialize(cac.init_phi_slab(shape, BASE["width"], y_lo=ny * 0.25, y_hi=ny * 0.75))
    diags = cac.run(sim, steps, sample_every=max(1, steps // 4))
    m = M.flat_interface_tension(sim)
    for key in ("sigma_integral", "sigma_from_well", "sigma_from_gradient"):
        m[key] = 0.5 * m[key]  # two interfaces per periodic slab
    return {
        "case": "sigma_flat",
        "inputs": {"shape": list(shape), "steps": steps, "sigma_input": sigma, **BASE},
        "meas": m,
        "sigma_meas": m["sigma_integral"],
        "ratio_to_input": m["sigma_integral"] / sigma,
        "u_max": diags[-1]["u_max_fluid"],
        "wall_time_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Case 2: Young-Laplace
# ---------------------------------------------------------------------------


def laplace(sigma: float = 0.02, shape=(128, 128), radius: float = 30.0, steps: int = 4000) -> dict:
    t0 = time.time()
    sim = _sim(shape, sigma=sigma)
    sim.initialize(cac.init_phi_droplet(shape, radius, BASE["width"], center=(shape[0] / 2.0, shape[1] / 2.0)))
    traj = []
    for k in range(0, steps, max(1, steps // 4)):
        cac.run(sim, max(1, steps // 4), sample_every=0)
        m = M.laplace_drop(sim)
        m["step"] = sim.step_count
        traj.append({"step": sim.step_count, "dp": m["dp"], "R_eq": m["R_eq"]})
    m = M.laplace_drop(sim)
    return {
        "case": "laplace",
        "inputs": {"shape": list(shape), "steps": steps, "R_init": radius, "sigma_input": sigma, **BASE},
        "meas": m,
        "traj": traj,
        "wall_time_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Case 3: static contact angle (Young)
# ---------------------------------------------------------------------------


def contact(
    theta_deg: float,
    sigma: float = 0.02,
    shape=(112, 112),
    radius: float = 22.0,
    steps: int = 3500,
    wall_thickness: int = 2,
    gravity: float = 2.0e-6,
) -> dict:
    """Drop on a flat plate with a prescribed Young angle.

    The plate is the bottom ``wall_thickness`` rows; the top of the box is a
    second plate so the (incompressible) gas is enclosed but never needs to
    move: a drop reshaping on a plate conserves liquid area.  A very weak
    gravity (``gravity``, capillary length >> R) holds a non-wetting drop on
    the plate; the initial condition is already the geometric cap of radius
    ``radius`` and angle ``theta_deg`` (centre at ``y_wall - R cos(theta)``, i.e. a small cap above the plate).
    """
    t0 = time.time()
    ny, nx = shape
    mask = cac.solid_plate(shape, wall_thickness, "bottom")
    mask |= cac.solid_plate(shape, wall_thickness, "top")
    y_wall = wall_thickness - 0.5
    sim = _sim(shape, sigma=sigma, wall_mask=mask, policy="contact", angle=theta_deg, axes="wp",
               gravity=(0.0, -gravity), gravity_mode="rho")
    y0 = y_wall - radius * math.cos(math.radians(theta_deg))
    sim.initialize(cac.init_phi_droplet(shape, radius, BASE["width"], center=(y0, nx / 2.0)))
    diags = cac.run(sim, steps, sample_every=max(1, steps // 4))
    phi = _numpy(sim)["phi"].numpy()
    fit = M.contact_angle_circle_fit(phi, y_wall)
    cap = M.spherical_cap_inversion(phi, y_wall)
    flux = cac.wall_contact_flux_from_angle(theta_deg, BASE["width"])
    k_coef = 1.5 * sigma * BASE["width"]
    h_wall = k_coef * flux  # = sigma_input cos(theta_input)
    return {
        "case": "contact",
        "theta_input_deg": float(theta_deg),
        "inputs": {"shape": list(shape), "steps": steps, "R_init": radius, "sigma_input": sigma,
                   "wall_thickness": wall_thickness, "wall_flux": flux, "h_wall": h_wall, **BASE},
        "meas": {"circle_fit": fit, "cap_inversion": cap},
        "u_max": diags[-1]["u_max_fluid"],
        "wall_time_s": time.time() - t0,
    }


def contact_wall_order(
    psi_wall: float,
    psi_wall_top: float = -1.0,
    sigma: float = 0.02,
    shape=(112, 112),
    radius: float = 22.0,
    steps: int = 3500,
    wall_thickness: int = 2,
    gravity: float = 2.0e-6,
    sample_every: int | None = None,
) -> dict:
    """Drop on a flat plate whose wetting is set by a *wall order parameter*.

    Integration arm of ``XVAL-LBM-DIFFUSE-INTERFACE-WETTING-VIA-WALL-ORDER-PARAMETER``:
    same geometry as :func:`contact`, but the caller prescribes ``psi_w`` instead
    of a contact angle and the *production* solver
    (:class:`tensorlbm.cac_lbm2d.CAC2D`, ``wall_phi_policy="wall_order"``) derives
    the wall flux from the emergent Young angle
    ``cos(theta) = (3 psi_w - psi_w^3)/2``.

    Two independent estimators are reported per run (circle fit of the level set
    and spherical-cap area inversion, both from :mod:`tensorlbm.wetting.metrics`,
    which never see the solver's internal angle), plus the analytic angle.
    """
    t0 = time.time()
    ny, nx = shape
    mask = cac.solid_plate(shape, wall_thickness, "bottom")
    mask |= cac.solid_plate(shape, wall_thickness, "top")
    y_wall = wall_thickness - 0.5
    sim = _sim(shape, sigma=sigma, wall_mask=mask, policy="wall_order", angle=None, axes="wp",
               gravity=(0.0, -gravity), gravity_mode="rho", wall_order_parameter=float(psi_wall),
               wall_order_parameter_top=float(psi_wall_top))
    theta_young = math.degrees(analytic_contact_angle(psi_wall))
    y0 = y_wall - radius * math.cos(math.radians(theta_young))
    sim.initialize(cac.init_phi_droplet(shape, radius, BASE["width"], center=(y0, nx / 2.0)))
    diags = cac.run(sim, steps, sample_every=sample_every or max(1, steps // 4))
    phi = _numpy(sim)["phi"].numpy()
    fit = M.contact_angle_circle_fit(phi, y_wall)
    cap = M.spherical_cap_inversion(phi, y_wall)
    measured = [fit.get("theta_deg") if isinstance(fit, dict) else None,
                cap.get("theta_deg") if isinstance(cap, dict) else None]
    measured = [float(m) for m in measured if m is not None and math.isfinite(float(m))]
    theta_meas = float(np.median(measured)) if measured else float("nan")
    return {
        "case": "contact_wall_order",
        "psi_wall": float(psi_wall),
        "psi_wall_top": float(psi_wall_top),
        "cos_young": float(young_cos_theta(psi_wall)),
        "theta_young_deg": float(theta_young),
        "theta_measured_deg": theta_meas,
        "err_pct": (theta_meas - theta_young) / theta_young * 100.0 if theta_young > 0 else float("nan"),
        "inputs": {"shape": list(shape), "steps": steps, "R_init": radius, "sigma_input": sigma,
                   "wall_thickness": wall_thickness, "wall_phi_policy": "wall_order", **BASE},
        "meas": {"circle_fit": fit, "cap_inversion": cap, "n_estimators": len(measured)},
        "u_max": diags[-1]["u_max_fluid"],
        "wall_time_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Case 4: Jurin capillary rise
# ---------------------------------------------------------------------------


def jurin(
    theta_deg: float = 60.0,
    sigma: float = 0.03,
    shape=(160, 88),
    gap: int = 20,
    baffle: int = 2,
    pool: int = 20,
    gravity: float = 2.0e-4,
    steps: int = 9000,
    sample_every: int = 50,
    mu_l: float = 0.05,
    mu_g: float = 0.005,
) -> dict:
    """Capillary rise between two vertical baffles standing in a pool.

    Geometry (rows grow upward): bottom plate ``rows 0-1``, top plate at the
    last two rows, one baffle pair of thickness ``baffle`` separated by
    ``gap`` fluid columns, spanning rows ``pool .. ny-3``.  The liquid pool
    starts at row ``pool`` across the whole width, so the gas displaced by the
    rising column escapes over the baffle tops (a closed tube would block the
    rise in an incompressible formulation).

    ``h = 2 sigma cos(theta) / (drho g * gap)`` with ``drho = rho_l - rho_g``.
    """
    t0 = time.time()
    ny, nx = shape
    mask = cac.solid_plate(shape, 2, "bottom") | cac.solid_plate(shape, 2, "top")
    x_c = nx // 2
    x_left = x_c - gap // 2 - baffle
    x_right = x_c + (gap - gap // 2)
    mask[pool : ny - 2, x_left : x_left + baffle] = True
    mask[pool : ny - 2, x_right : x_right + baffle] = True
    sim = _sim(
        shape, sigma=sigma, wall_mask=mask, policy="contact", angle=theta_deg, axes="wp",
        gravity=(0.0, -gravity), gravity_mode="rho", mu_l=mu_l, mu_g=mu_g,
    )
    sim.initialize(cac.init_phi_layers(shape, BASE["width"], y0=float(pool)))
    ch_lo, ch_hi = x_left + baffle + 1, x_right - 1
    res_lo, res_hi = 4, x_left - 2
    traj = []
    for k in range(0, steps, sample_every):
        cac.run(sim, sample_every, sample_every=0)
        phi = _numpy(sim)["phi"].numpy()
        h_ch, h_ch_std = M.meniscus_height(phi, ch_lo, ch_hi)
        h_res, h_res_std = M.meniscus_height(phi, res_lo, res_hi)
        u = cac.macro(sim)["u"]
        speed = (u[0] ** 2 + u[1] ** 2).sqrt()
        traj.append({
            "step": sim.step_count,
            "y_channel": h_ch,
            "y_channel_std": h_ch_std,
            "y_reservoir": h_res,
            "h": h_ch - h_res,
            "u_max": float(speed.max().item()),
        })
    u = cac.macro(sim)["u"]
    speed = float((u[0] ** 2 + u[1] ** 2).sqrt().max().item())
    return {
        "case": "jurin",
        "theta_input_deg": float(theta_deg),
        "inputs": {"shape": list(shape), "steps": steps, "gap": gap, "baffle": baffle, "pool": pool,
                   "gravity": gravity, "sigma_input": sigma, "x_channel": [ch_lo, ch_hi],
                   "x_reservoir": [res_lo, res_hi], "mu_l": mu_l, "mu_g": mu_g, **BASE},
        "traj": traj,
        "u_max": speed,
        "wall_time_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Case 5: Lamb / Rayleigh drop oscillation
# ---------------------------------------------------------------------------


def lamb(
    sigma: float = 0.06,
    shape=(112, 112),
    radius: float = 24.0,
    amp: float = 0.05,
    steps: int = 5000,
    sample_every: int = 5,
    mu_l: float = 0.03,
    mu_g: float = 0.003,
) -> dict:
    """m=2 capillary oscillation of an elliptical drop (periodic box)."""
    t0 = time.time()
    sim = _sim(shape, sigma=sigma, mu_l=mu_l, mu_g=mu_g)
    sim.initialize(
        cac.init_phi_ellipse(
            shape, radius * (1.0 + amp), radius / (1.0 + amp), BASE["width"],
            center=(shape[0] / 2.0, shape[1] / 2.0),
        )
    )
    ts, sig, req_log = [], [], []
    for k in range(0, steps, sample_every):
        cac.run(sim, sample_every, sample_every=0)
        phi = _numpy(sim)["phi"].numpy()
        rx, ry, req = M.radial_extents(phi)
        ts.append(sim.step_count)
        sig.append(rx - ry)
        req_log.append(req)
    ts = np.array(ts, dtype=float)
    sig = np.array(sig, dtype=float)
    req = float(np.mean(req_log))
    rho_sum = BASE["rho_l"] + BASE["rho_g"]
    omega_th = math.sqrt(6.0 * sigma / (rho_sum * req**3))
    # crude period guess for the scan centre: 2*pi/omega_th
    fit = M.damped_sine_fit(ts, sig, 2.0 * math.pi / (2.0 * math.pi / omega_th))
    return {
        "case": "lamb",
        "inputs": {"shape": list(shape), "steps": steps, "R_init": radius, "amp": amp,
                   "sigma_input": sigma, "sample_every": sample_every, "mu_l": mu_l, "mu_g": mu_g,
                   "rho_sum": rho_sum, **BASE},
        "R_eq": req,
        "omega_theory": omega_th,
        "period_theory": 2.0 * math.pi / omega_th,
        "fit": fit,
        "traj": {"t": ts.tolist(), "RxRy": sig.tolist()},
        "wall_time_s": time.time() - t0,
    }


def save(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))
    print(f"[ladder] wrote {path}", flush=True)