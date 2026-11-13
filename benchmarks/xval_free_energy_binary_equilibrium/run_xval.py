"""XVAL: free-energy binary LBM equilibrium pressure (sailfish semantics transplant).

Anchor: ``XVAL-LBM-FREE-ENERGY-BINARY-EQUILIBRIUM-PRESSURE``.
Acceptance band: owner's ``< 15 %`` on every physical KPI, declared before the
run in ``xval/predeclared.json``.

Arms
----
``flat``   flat (single-normal) interface as a periodic *band*: two interfaces,
           normal along x, no droplet curvature.  Measures the pressure-tensor
           surface tension ``sigma = integral (P_NN - P_TT) dx`` and the
           mechanical-equilibrium residual ``d P_NN / dx``.
``dropR16``/``dropR24``/``dropR36``
           circular droplet, three (R, xi) pairs.  Measures the Laplace jump
           ``p_in - p_out`` from the measured pressure fields and compares it
           with ``sigma / R``.

Every number written to ``artifacts/xval_free_energy_binary_equilibrium/*.json``
is computed here from the populations; nothing is hand-typed.

Usage::

    python benchmarks/xval_free_energy_binary_equilibrium/run_xval.py --arm flat
    python benchmarks/xval_free_energy_binary_equilibrium/run_xval.py --arm all
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from tensorlbm.phasefield.free_energy_pressure import (  # noqa: E402
    CX,
    CY,
    FreeEnergyBinaryParams,
    chemical_potential,
    collide_and_stream,
    equilibrium_f,
    grad_lap,
    initial_condition,
    macroscopic,
    mean_pressure,
    periodic_distance,
    pressure_tensor_analytic,
    pressure_tensor_measured,
    spurious_velocity,
    tanh_profile,
)

OUT_DIR = os.path.join(
    _ROOT, "artifacts", "xval_free_energy_binary_equilibrium"
)

#: declared arm table (resolution analogy: interface width xi in lattice units)
ARMS: dict[str, dict] = {
    "flat": dict(kind="flat", A=0.1, xi=3.0, Gamma=0.2, tau_f=1.0, tau_phi=1.5,
                 Lx=64, Ly=8, band=16, steps=4000),
    "dropR16": dict(kind="drop", A=0.1, xi=2.0, Gamma=0.2, tau_f=1.0, tau_phi=1.5,
                    L=96, R=16, steps=8000),
    "dropR24": dict(kind="drop", A=0.1, xi=2.0, Gamma=0.2, tau_f=1.0, tau_phi=1.5,
                    L=128, R=24, steps=10000),
    "dropR36": dict(kind="drop", A=0.1, xi=3.0, Gamma=0.2, tau_f=1.0, tau_phi=1.5,
                    L=176, R=36, steps=12000),
}


def params_of(cfg: dict) -> FreeEnergyBinaryParams:
    """``kappa`` is fixed by the declared interface width: ``kappa = A xi^2 / 2``."""
    return FreeEnergyBinaryParams(
        A=cfg["A"],
        kappa=0.5 * cfg["A"] * cfg["xi"] ** 2,
        Gamma=cfg["Gamma"],
        tau_f=cfg["tau_f"],
        tau_phi=cfg["tau_phi"],
    )


# --------------------------------------------------------------------------- #
# initial conditions
# --------------------------------------------------------------------------- #
def flat_band_phi(nx: int, ny: int, band: int, xi: float) -> torch.Tensor:
    """Periodic band: ``phi = +1`` inside a slab of width ``band`` centred in x."""
    x = torch.arange(nx, dtype=torch.float64).view(1, -1)
    signed = (band / 2.0) - (x - nx / 2.0).abs()
    return tanh_profile(signed, 0.0, xi).expand(ny, nx).contiguous()


def droplet_phi(n: int, radius: float, xi: float) -> torch.Tensor:
    """Circular droplet: ``phi = -tanh((r - R) / xi)`` (+1 inside)."""
    x = torch.arange(n, dtype=torch.float64).view(1, -1)
    y = torch.arange(n, dtype=torch.float64).view(-1, 1)
    dx = periodic_distance(x.expand(n, n).contiguous(), n / 2.0, n)
    dy = periodic_distance(y.expand(n, n).contiguous(), n / 2.0, n)
    r = torch.sqrt(dx * dx + dy * dy)
    return tanh_profile(r, radius, xi) * -1.0


# --------------------------------------------------------------------------- #
# measurement helpers
# --------------------------------------------------------------------------- #
def closure_residual(
    f: torch.Tensor, rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor,
    phi: torch.Tensor, p: FreeEnergyBinaryParams,
) -> dict[str, float]:
    """``|| sum_i f_i^eq c_ia c_ib - (P_ab + rho u_a u_b) ||`` relative to ``max|P|``.

    ``P`` here is the analytic free-energy pressure tensor *computed with the
    same nine-point operators* used by the model, so a non-zero residual is a
    genuine algebra mismatch, not a discretisation error.
    """
    gx, gy, lap = grad_lap(phi)
    feq = equilibrium_f(rho, ux, uy, phi, gx, gy, lap, p.A, p.kappa)
    Pm = pressure_tensor_measured(feq)
    Pa = pressure_tensor_analytic(rho, phi, gx, gy, lap, p.A, p.kappa)
    target = (rho * ux * ux + Pa[0], rho * uy * uy + Pa[1], rho * ux * uy + Pa[2])
    scale = max(float(t.abs().max()) for t in target)
    resid = max(float((Pm[i] - target[i]).abs().max()) for i in range(3))
    return {"abs": resid, "scale": scale, "rel": resid / scale}


def ring_mean(field: torch.Tensor, r: torch.Tensor, r0: float, r1: float) -> float:
    mask = (r >= r0) & (r <= r1)
    if not bool(mask.any()):
        return float("nan")
    return float(field[mask].mean())


def radial_profile(
    field: torch.Tensor, r: torch.Tensor, width: float = 1.0, rmax: float | None = None
) -> dict[str, list[float]]:
    rmax = float(r.max()) if rmax is None else rmax
    nb = int(math.floor(rmax / width))
    centres, means = [], []
    for b in range(nb):
        r0, r1 = b * width, (b + 1) * width
        mask = (r >= r0) & (r < r1)
        if bool(mask.any()):
            centres.append(0.5 * (r0 + r1))
            means.append(float(field[mask].mean()))
    return {"r": centres, "value": means}


def radial_pressure_tensor(
    Pxx: torch.Tensor, Pyy: torch.Tensor, Pxy: torch.Tensor,
    dx: torch.Tensor, dy: torch.Tensor, r: torch.Tensor, eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(P_rr, P_tt)`` from the Cartesian tensor and the radial direction."""
    nx = dx / (r + eps)
    ny = dy / (r + eps)
    P_rr = nx * nx * Pxx + 2.0 * nx * ny * Pxy + ny * ny * Pyy
    P_tt = ny * ny * Pxx - 2.0 * nx * ny * Pxy + nx * nx * Pyy
    return P_rr, P_tt


def ray_sigma(
    P_rr: torch.Tensor, P_tt: torch.Tensor, r: torch.Tensor,
    angles: int = 16, dr: float = 0.5,
) -> dict[str, float]:
    """``sigma = integral (P_NN - P_TT) dr`` averaged over rays through the centre.

    Values are taken by nearest-neighbour sampling on a radial grid, which is
    adequate for smooth fields with interface width >= 2 lattice spacings.
    """
    n = P_rr.shape[0]
    c = n / 2.0
    grid = torch.arange(n, dtype=torch.float64)
    yy = (grid.view(-1, 1) - c).expand(n, n)
    xx = (grid.view(1, -1) - c).expand(n, n)
    sigmas = []
    for a in range(angles):
        th = 2.0 * math.pi * a / angles
        rs = torch.arange(dr, n / 2.0 - 1.0, dr, dtype=torch.float64)
        ix = torch.clamp((c + rs * math.cos(th)).round().long(), 0, n - 1)
        iy = torch.clamp((c + rs * math.sin(th)).round().long(), 0, n - 1)
        diff = (P_rr[iy, ix] - P_tt[iy, ix]).sum().item() * dr
        sigmas.append(diff)
    t = torch.tensor(sigmas, dtype=torch.float64)
    return {"mean": float(t.mean()), "min": float(t.min()), "max": float(t.max()),
            "n_rays": angles, "dr": dr,
            "ray_spread_rel": float((t.max() - t.min()) / abs(t.mean())) if t.mean() != 0 else float("nan")}


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #
def run_flat(cfg: dict, log=print) -> dict:
    p = params_of(cfg)
    nx, ny = cfg["Lx"], cfg["Ly"]
    phi0 = flat_band_phi(nx, ny, cfg["band"], cfg["xi"])
    f, g = initial_condition(phi0, p)

    rho, ux, uy, phi = macroscopic(f, g)
    closure0 = closure_residual(f, rho, ux, uy, phi, p)

    history = []
    t0 = time.time()
    for n in range(1, cfg["steps"] + 1):
        f, g = collide_and_stream(f, g, p)
        if n % 500 == 0 or n == cfg["steps"]:
            r, u, v, ph = macroscopic(f, g)
            Pxx, Pyy, _ = pressure_tensor_measured(f)
            sigma = float((Pxx - Pyy).mean(dim=0).sum()) / 2.0  # two interfaces
            prof = Pxx.mean(dim=0)
            history.append({
                "step": n,
                "sigma": sigma,
                "u_max": float((u * u + v * v).sqrt().max()),
                "rho_min": float(r.min()), "rho_max": float(r.max()),
                "phi_err_inf": float((ph - phi0).abs().max()),
                "Pnn_dev_rel": float((prof - prof.mean()).abs().max() / prof.mean()),
                "mass_drift": float((r.sum() - r.numel()).abs() / r.numel()),
            })
            log(f"  {n:6d} sigma={sigma:.6f} umax={history[-1]['u_max']:.3e} "
                f"rho=[{history[-1]['rho_min']:.6f},{history[-1]['rho_max']:.6f}] "
                f"Pnn_dev={history[-1]['Pnn_dev_rel']:.3e}")
    elapsed = time.time() - t0

    rho, ux, uy, phi = macroscopic(f, g)
    Pxx, Pyy, Pxy = pressure_tensor_measured(f)
    prof = Pxx.mean(dim=0)
    sigma_band_total = float((Pxx - Pyy).mean(dim=0).sum())
    sigma_meas = sigma_band_total / 2.0  # the periodic band carries two interfaces
    closure = closure_residual(f, rho, ux, uy, phi, p)
    sp = spurious_velocity(ux, uy)

    x = torch.arange(nx, dtype=torch.float64)
    gx, gy, lap = grad_lap(phi)
    Pa = pressure_tensor_analytic(rho, phi, gx, gy, lap, p.A, p.kappa)

    return {
        "arm": "flat",
        "config": {**cfg, "kappa": p.kappa, "sigma_analytic": p.sigma,
                   "xi_analytic": p.xi, "tau_f": p.tau_f, "tau_phi": p.tau_phi,
                   "Gamma": p.Gamma, "viscosity": p.viscosity, "mobility": p.mobility},
        "elapsed_s": elapsed,
        "closure_initial": closure0,
        "closure_final": closure,
        "sigma_band_total": sigma_band_total,
        "sigma_measured": sigma_meas,
        "sigma_analytic": p.sigma,
        "sigma_rel_error": abs(sigma_meas - p.sigma) / p.sigma,
        "Pnn_uniformity_rel": float((prof - prof.mean()).abs().max() / prof.mean()),
        "Pnn_mean": float(prof.mean()),
        "spurious_velocity": sp,
        "rho_range": [float(rho.min()), float(rho.max())],
        "phi_err_inf_initial": float((phi - phi0).abs().max()),
        "pressure_tensor_energy_residual": float(
            (Pxx - Pa[0]).abs().max() / Pa[0].abs().max()),
        "profiles": {
            "x": x.tolist(),
            "phi": phi.mean(dim=0).tolist(),
            "phi_analytic": phi0.mean(dim=0).tolist(),
            "Pxx_measured": prof.tolist(),
            "Pyy_measured": Pyy.mean(dim=0).tolist(),
            "Pxx_analytic": Pa[0].mean(dim=0).tolist(),
            "rho": rho.mean(dim=0).tolist(),
            "ux": ux.mean(dim=0).tolist(),
        },
        "history": history,
    }


def run_droplet(cfg: dict, log=print) -> dict:
    p = params_of(cfg)
    n, R = cfg["L"], cfg["R"]
    phi0 = droplet_phi(n, R, cfg["xi"])
    f, g = initial_condition(phi0, p)

    grid = torch.arange(n, dtype=torch.float64)
    c = n / 2.0
    dx = periodic_distance(grid.view(1, -1).expand(n, n).contiguous(), c, n)
    dy = periodic_distance(grid.view(-1, 1).expand(n, n).contiguous(), c, n)
    r = torch.sqrt(dx * dx + dy * dy)

    rho, ux, uy, phi = macroscopic(f, g)
    closure0 = closure_residual(f, rho, ux, uy, phi, p)

    history = []
    t0 = time.time()
    dp_prev = None
    stop_reason = "step budget"
    for step in range(1, cfg["steps"] + 1):
        f, g = collide_and_stream(f, g, p)
        if step % 500 == 0 or step == cfg["steps"]:
            rr, u, v, ph = macroscopic(f, g)
            pr = mean_pressure(f)
            p_in = ring_mean(pr, r, 0.0, max(1.0, R - 3.0 * cfg["xi"]))
            p_out = ring_mean(pr, r, R + 3.0 * cfg["xi"], n / 2.0 - 1.0)
            dp = p_in - p_out
            history.append({
                "step": step, "p_in": p_in, "p_out": p_out, "dp": dp,
                "u_max": float((u * u + v * v).sqrt().max()),
                "rho_min": float(rr.min()), "rho_max": float(rr.max()),
            })
            log(f"  {step:6d} dp={dp:.6e} pin={p_in:.6f} pout={p_out:.6f} "
                f"umax={history[-1]['u_max']:.3e}")
            if dp_prev is not None and abs(dp) > 0:
                drift = abs(dp - dp_prev) / abs(dp)
                if drift < 1e-3:
                    stop_reason = f"laplace jump settled (drift {drift:.2e} < 1e-3 at step {step})"
                    break
            dp_prev = dp
    elapsed = time.time() - t0

    rho, ux, uy, phi = macroscopic(f, g)
    Pxx, Pyy, Pxy = pressure_tensor_measured(f)
    pr = 0.5 * (Pxx + Pyy)
    P_rr, P_tt = radial_pressure_tensor(Pxx, Pyy, Pxy, dx, dy, r)
    sigma_ray = ray_sigma(P_rr, P_tt, r)

    p_in = ring_mean(pr, r, 0.0, max(1.0, R - 3.0 * cfg["xi"]))
    p_out = ring_mean(pr, r, R + 3.0 * cfg["xi"], n / 2.0 - 1.0)
    dp = p_in - p_out
    closure = closure_residual(f, rho, ux, uy, phi, p)
    sp = spurious_velocity(ux, uy)
    dp_analytic = p.sigma / R

    return {
        "arm": cfg.get("name", "drop"),
        "config": {**cfg, "kappa": p.kappa, "sigma_analytic": p.sigma,
                   "xi_analytic": p.xi, "viscosity": p.viscosity,
                   "mobility": p.mobility},
        "elapsed_s": elapsed,
        "stop_reason": stop_reason,
        "steps_run": history[-1]["step"] if history else 0,
        "closure_initial": closure0,
        "closure_final": closure,
        "R": R,
        "p_in": p_in, "p_out": p_out, "dp": dp,
        "dp_analytic": dp_analytic,
        "laplace_ratio_analytic": dp / dp_analytic,
        "laplace_rel_error_analytic": abs(dp - dp_analytic) / dp_analytic,
        "sigma_ray": sigma_ray,
        "laplace_ratio_ray": dp / sigma_ray["mean"] * R if sigma_ray["mean"] else float("nan"),
        "laplace_rel_error_ray": (
            abs(dp - sigma_ray["mean"] / R) / (sigma_ray["mean"] / R)
            if sigma_ray["mean"] else float("nan")),
        "spurious_velocity": sp,
        "rho_range": [float(rho.min()), float(rho.max())],
        "radial": {
            "r": radial_profile(phi, r)["r"],
            "phi": radial_profile(phi, r)["value"],
            "phi_analytic": radial_profile(phi0, r)["value"],
            "p": radial_profile(pr, r)["value"],
            "P_rr": radial_profile(P_rr, r)["value"],
            "P_tt": radial_profile(P_tt, r)["value"],
            "u_mag": radial_profile(torch.sqrt(ux * ux + uy * uy), r)["value"],
        },
        "history": history,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="all",
                    help="one of " + ", ".join(ARMS) + ", or 'all'")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="step-count scale (smoke runs only; 1.0 = predeclared)")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)

    names = list(ARMS) if args.arm == "all" else [args.arm]
    for name in names:
        if name not in ARMS:
            raise SystemExit(f"unknown arm {name!r}")
        cfg = dict(ARMS[name])
        cfg["steps"] = int(round(cfg["steps"] * args.scale))
        cfg["name"] = name
        runner = run_flat if cfg["kind"] == "flat" else run_droplet
        print(f"[xval] arm {name}: {cfg}", flush=True)
        result = runner(cfg, log=lambda s: print(s, flush=True))
        path = os.path.join(args.out, f"{name}.json")
        with open(path, "w") as fh:
            json.dump(result, fh, indent=1)
        print(f"[xval] wrote {path} ({os.path.getsize(path)} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())