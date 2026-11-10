"""Machine-checkable validation gates for the air-layer two-phase solver.

Four gates (see campaign IN30-N03):

``single_phase_cf``   : Poiseuille channel, Cf = 2 tau_w/(rho u_b^2) vs 12/Re.
``surface_tension``   : flat interface, sigma from the anisotropy of the LBM
                        second moment  sigma = int (Pi_yy - Pi_xx) dy.
``laplace``           : static bubbles, dP vs sigma (1/R1 + 1/R2).
``mass_conservation`` : closed periodic box, sum(phi) and total mass drift.

Every function returns plain JSON-serialisable dicts so the campaign report can
quote the numbers without hand copying.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .core import (
    CS2,
    PhaseFieldChannel,
    PhaseFieldParams,
    SinglePhaseChannel,
    circular_bubble,
    laplace_pressure,
)


# ---------------------------------------------------------------------------
# gate (a): single-phase friction
# ---------------------------------------------------------------------------
def single_phase_cf(*, nx: int = 8, ny: int = 48, nu: float = 0.05,
                    body_force_x: float = 1e-5, steps: int = 60000,
                    rho: float = 1.0) -> dict:
    ch = SinglePhaseChannel(nx, ny, rho=rho, nu=nu, body_force_x=body_force_x)
    ch.step(steps)
    ub = ch.bulk_velocity()
    cf_sim = ch.simulated_cf()
    cf_ana = ch.analytic_cf()
    tau_w = ch.wall_shear_mean()
    tau_balance = body_force_x * ch.height / 2.0
    u_max = float(np.max(np.abs(ch.ux)))
    return {
        "gate": "a_single_phase_friction",
        "grid": [nx, ny],
        "nu": nu,
        "body_force_fx": body_force_x,
        "steps": steps,
        "bulk_velocity": ub,
        "u_max": u_max,
        "reynolds": rho * ub * ch.height / nu,
        "cf_simulated": cf_sim,
        "cf_analytic": cf_ana,
        "cf_rel_error": abs(cf_sim - cf_ana) / cf_ana,
        "wall_shear_mem": tau_w,
        "wall_shear_momentum_balance": tau_balance,
        "momentum_balance_rel_error": abs(tau_w - tau_balance) / tau_balance,
        "profile_rel_l2_error": ch.profile_error(),
        "pass_3pct": bool(abs(cf_sim - cf_ana) / cf_ana <= 0.03),
    }


# ---------------------------------------------------------------------------
# gate (b): surface tension from the flat interface (independent calibration)
# ---------------------------------------------------------------------------
def flat_interface_phi(ny: int, nx: int, band: tuple[float, float],
                       delta: float) -> np.ndarray:
    """Periodic gas band ``band``=(y0, y1) in a domain of height ``ny``."""
    y0, y1 = band
    mid, half = 0.5 * (y0 + y1), 0.5 * (y1 - y0)
    y = np.arange(ny)
    d = np.abs(((y - mid + 0.5 * ny) % ny) - 0.5 * ny)   # periodic distance to mid
    phi = -np.tanh((half - d) / delta)
    return np.repeat(phi[:, None], nx, axis=1)


def surface_tension(*, params: PhaseFieldParams | None = None, ny: int = 64,
                    nx: int = 32, steps: int = 4000) -> dict:
    p = params or PhaseFieldParams(interface_width=6)
    band = (0.25 * ny, 0.75 * ny)
    phi0 = flat_interface_phi(ny, nx, band, p.delta)
    ch = PhaseFieldChannel(nx, ny, p, phi0, walls=False)
    ch.step(steps)
    pxx, pyy, _ = ch.stress_tensor()
    diff = (pyy - pxx).mean(axis=1)
    sigma = float(np.trapezoid(diff, np.arange(ny)))
    return {
        "gate": "b0_surface_tension_flat",
        "sigma_measured": sigma,
        "sigma_nominal": p.sigma,
        "ratio_measured_over_nominal": sigma / p.sigma,
        "u_max_spurious": float(np.max(np.abs(ch.ux))),
        "steps": steps,
        "grid": [nx, ny],
        "interface_width": p.interface_width,
    }


# ---------------------------------------------------------------------------
# gate (b): Laplace law
# ---------------------------------------------------------------------------
def laplace_single(params: PhaseFieldParams, radius: float, *, box: int = 96,
                   steps: int = 5000) -> dict:
    phi0 = circular_bubble(box, box, radius, delta=params.delta)
    p_init = laplace_pressure(box, box, radius, params.sigma, delta=params.delta)
    ch = PhaseFieldChannel(box, box, params, phi0, walls=False, p_init=p_init)
    ch.step(steps)
    x = np.arange(box)[None, :] - 0.5 * box
    y = np.arange(box)[:, None] - 0.5 * box
    r = np.sqrt(x ** 2 + y ** 2)
    inside = r < radius - 3.0 * params.delta
    outside = r > radius + 3.0 * params.delta
    pr = ch.pressure()
    dp = float(pr[inside].mean() - pr[outside].mean())
    return {
        "radius": radius,
        "dp_measured": dp,
        "dp_laplace_nominal": params.sigma / radius,
        "rel_error_nominal": (dp - params.sigma / radius) / (params.sigma / radius),
        "sigma_effective": dp * radius,
        "u_max_spurious": float(np.max(np.abs(ch.ux))),
        "phi_drift_rel": float((ch.phi.sum() - phi0.sum()) / abs(phi0.sum())),
    }


def laplace_gate(params: PhaseFieldParams | None = None, radii=(16.0, 24.0, 32.0),
                 box: int = 96, steps: int = 5000,
                 sigma_ref: float | None = None) -> dict:
    p = params or PhaseFieldParams(interface_width=6)
    runs = [laplace_single(p, r, box=box, steps=steps) for r in radii]
    rr = np.array([r["radius"] for r in runs])
    dd = np.array([r["dp_measured"] for r in runs])
    # dP = sigma (1/R); least squares through the origin
    sigma_fit = float(np.sum(dd * (1.0 / rr)) / np.sum((1.0 / rr) ** 2))
    ref = p.sigma if sigma_ref is None else sigma_ref
    errs = [abs(d - ref / r) / (ref / r) for d, r in zip(dd, rr)]
    return {
        "gate": "b_laplace_law",
        "radius_list": list(radii),
        "sigma_nominal": p.sigma,
        "sigma_fit_from_slope": sigma_fit,
        "reference_sigma": ref,
        "runs": runs,
        "max_rel_error_vs_reference": float(max(errs)),
        "rel_errors_vs_reference": errs,
        "pass_5pct": bool(max(errs) <= 0.05),
    }


# ---------------------------------------------------------------------------
# gate (c): mass / phase conservation in a closed domain
# ---------------------------------------------------------------------------
def mass_conservation_gate(params: PhaseFieldParams | None = None, *, box: int = 64,
                           steps: int = 4000, n_samples: int = 8,
                           radius: float = 16.0, with_injection: bool = False) -> dict:
    p = params or PhaseFieldParams(interface_width=6)
    phi0 = circular_bubble(box, box, radius, delta=p.delta)
    ch = PhaseFieldChannel(box, box, p, phi0, walls=False,
                           p_init=laplace_pressure(box, box, radius, p.sigma, delta=p.delta))
    m0 = ch.total_mass()
    s0 = ch.phi.sum()
    mass_hist, phi_hist = [m0], [s0]
    inj_mask = np.zeros((box, box), dtype=bool)
    inj_mask[1:4, :] = True
    n_steps = max(1, steps // n_samples)
    for _ in range(n_samples):
        if with_injection:
            ch.step(n_steps, inject=(inj_mask, 1e-3))
        else:
            ch.step(n_steps)
        mass_hist.append(ch.total_mass())
        phi_hist.append(ch.phi.sum())
    drift_mass = max(abs(m - m0) for m in mass_hist) / abs(m0)
    drift_phi = max(abs(s - s0) for s in phi_hist) / max(abs(s0), 1.0)
    ledger = {
        "injected_volume": ch.injected_volume,
        "vented_volume": ch.vented_volume,
        "gas_volume_now": ch.total_gas_volume(),
        "gas_volume_initial": float(np.sum(0.5 * (1 - phi0))),
    }
    ledger["volume_gain"] = ledger["gas_volume_now"] - ledger["gas_volume_initial"]
    ledger["ledger_residual"] = (ledger["volume_gain"]
                                 - (ledger["injected_volume"] - ledger["vented_volume"]))
    ledger["ledger_rel_residual"] = abs(ledger["ledger_residual"]) / max(
        abs(ledger["volume_gain"]), abs(ledger["injected_volume"]), 1e-30)
    return {
        "gate": "c_mass_conservation",
        "grid": [box, box],
        "steps": steps,
        "with_injection": with_injection,
        "mass_rel_drift": drift_mass,
        "phi_rel_drift": drift_phi,
        "pass_1e-6": bool(drift_mass <= 1e-6 and drift_phi <= 1e-6),
        "ledger": ledger,
    }