#!/usr/bin/env python
"""SC-LBM interface-tension campaign driver (node P143W2-N01).

Produces ``artifacts/sc_surface_tension/*.json`` + figures for

  (a) the three-radius Laplace gate  ``Delta p * R`` vs ``sigma_K``  (<3 %),
  (b) the directional-artefact gate  angular spread of ``Delta p(theta)`` (<1 %),
  (c) the contact-angle gate         ``theta`` set vs measured (<3 deg),
  (d) the negative controls          ``G_eff = 0`` and ``kappa3 = 0`` (spectral).

Examples::

    python benchmarks/sc_surface_tension/run.py --task laplace --radii 20,40,60 \
        --stencils d2q9,shell4 --steps 24000 --out artifacts/sc_surface_tension
    python benchmarks/sc_surface_tension/run.py --task contact --thetas 60,90,120
    python benchmarks/sc_surface_tension/run.py --task negative
    python benchmarks/sc_surface_tension/run.py --task gate      # assemble + figures
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.sc_surface_tension import (  # noqa: E402
    CS2,
    G_EFF,
    MID_RHO,
    RHO_L,
    RHO_V,
    TAU,
    angular_pressure_jump,
    build_stencil,
    contact_angle_from_rho,
    contact_angle_from_wall_psi,
    droplet_rho,
    equivalent_radius,
    pressure_eos,
    pressure_flux,
    pressure_kinetic,
    pressure_tensor,
    psi_exp,
    radial_profile,
    radius_field,
    sc_force,
    sessile_rho,
    sigma_korteweg,
    step_sc,
    wall_psi_from_contact_angle,
)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (torch.Tensor,)):
        return obj.tolist()
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    return str(obj)


def box_for(R: float) -> int:
    """Smallest even box keeping every measurement band at least ``0.4 R`` off the wrap."""
    return int(4 * R + 16)


def measure_droplet(f: torch.Tensor, R0: float, stencil: dict, n_wedges: int = 36) -> dict:
    """Full measurement of one converged static droplet."""
    L = f.shape[-1]
    rho = f.sum(dim=0)
    _, ux, uy = macroscopic(f)
    umag = torch.sqrt(ux * ux + uy * uy)
    p_flux = pressure_flux(f)
    p_eos = pressure_eos(rho)
    p_kin = pressure_kinetic(f)
    lxx, lyy, lxy = pressure_tensor(f)
    r = radius_field(L, f.device)
    R = equivalent_radius(rho)
    inn = r < 0.5 * R
    out = r > 1.5 * R
    dp_flux = float(p_flux[inn].mean() - p_flux[out].mean())
    dp_eos = float(p_eos[inn].mean() - p_eos[out].mean())
    dp_kin = float(p_kin[inn].mean() - p_kin[out].mean())
    sigma_L = dp_flux * R  # 2-D cylinder: Delta p * R = sigma
    ang_flux = angular_pressure_jump(p_flux, R, n_wedges)
    ang_eos = angular_pressure_jump(p_eos, R, n_wedges)
    sk = sigma_korteweg(rho, L, stencil)
    prof = radial_profile(psi_exp(rho), L, 200)
    return {
        "L": L,
        "R_init": R0,
        "R_eq": R,
        "rho_in": float(rho[inn].mean()),
        "rho_out": float(rho[out].mean()),
        "dp_flux": dp_flux,
        "dp_eos": dp_eos,
        "dp_kinetic": dp_kin,
        "sigma_laplace": sigma_L,
        "sigma_eos": dp_eos * R,
        "sigma_K": sk["sigma_K"],
        "korteweg_coeff": sk["korteweg_coeff"],
        "kappa3": sk["kappa3"],
        "kappa3_over_integral": sk["sigma_K"] / max(sk["integral_dpsi2"], 1e-12),
        "integral_dpsi2": sk["integral_dpsi2"],
        "u_max": float(umag.max()),
        "u_rms": float(torch.sqrt((umag ** 2).mean())),
        "tau_xy_max": float(lxy.abs().max()),
        "shear_aniso_max": float(((lxx - lyy).abs() * 0.5).max()),
        "spread_flux_pct": ang_flux["spread_frac"] * 100.0,
        "maxdev_flux_pct": ang_flux["max_dev_frac"] * 100.0,
        "spread_eos_pct": ang_eos["spread_frac"] * 100.0,
        "dp_theta": ang_flux["dp_theta"],
        "n_wedges_used": ang_flux["n_wedges_used"],
        "profile_r": prof["r"],
        "profile_psi": prof["mean"],
    }


def run_droplet(
    R: float,
    stencil_name: str = "d2q9",
    mode: str = "stencil",
    G_eff: float = G_EFF,
    steps: int = 24000,
    min_steps: int = 4000,
    sample: int = 1000,
    conv_tol: float = 2e-3,
    L: int | None = None,
    seed_noise: float = 0.0,
    width: float = 2.0,
    verbose: bool = True,
) -> dict:
    """Relax a droplet and return the converged measurement + history."""
    L = int(L or box_for(R))
    stencil = build_stencil(stencil_name)
    rho0 = droplet_rho(L, R, width=width)
    if seed_noise:
        g = torch.Generator().manual_seed(1234)
        rho0 = rho0 + seed_noise * torch.randn(rho0.shape, generator=g)
    zero = torch.zeros_like(rho0)
    f = equilibrium(rho0, zero, zero)
    mass0 = float(rho0.sum())
    t0 = time.perf_counter()
    hist: list[dict] = []
    converged = False
    step = 0
    for step in range(1, steps + 1):
        f = step_sc(f, G_eff=G_eff, tau=TAU, stencil=stencil, mode=mode)
        if step % sample == 0:
            rho = f.sum(dim=0)
            if not torch.isfinite(rho).all():
                raise RuntimeError(f"divergence at step {step} (R={R}, {stencil_name})")
            R_eq = equivalent_radius(rho)
            r = radius_field(L)
            m = float(pressure_flux(f)[r < 0.5 * R_eq].mean() - pressure_flux(f)[r > 1.5 * R_eq].mean())
            hist.append({"step": step, "dp": m, "R_eq": R_eq, "rho_min": float(rho.min()),
                         "rho_max": float(rho.max())})
            if verbose:
                print(f"  [{stencil_name} R={R}] step {step} dp={m:.6f} R_eq={R_eq:.3f}", flush=True)
            if len(hist) >= 3 and step >= min_steps:
                d1 = abs(hist[-1]["dp"] - hist[-2]["dp"]) / max(abs(hist[-2]["dp"]), 1e-12)
                d2 = abs(hist[-2]["dp"] - hist[-3]["dp"]) / max(abs(hist[-3]["dp"]), 1e-12)
                if d1 < conv_tol and d2 < conv_tol:
                    converged = True
                    break
    res = measure_droplet(f, R, stencil)
    res.update({
        "stencil": stencil_name,
        "gradient_mode": mode,
        "G_eff": G_eff,
        "steps": step,
        "converged": converged,
        "mass_drift": abs(float(f.sum()) - mass0) / mass0,
        "wall_s": time.perf_counter() - t0,
        "history": hist,
    })
    return res


# ---------------------------------------------------------------------------
# Contact angle
# ---------------------------------------------------------------------------


def run_contact_angle(
    theta_set: float,
    R0: float = 18.0,
    ny: int = 72,
    nx: int = 112,
    steps: int = 16000,
    sample: int = 1000,
    conv_tol: float = 1e-3,
    wall_rows: int = 2,
    G_eff: float = G_EFF,
    stencil_name: str = "d2q9",
    mode: str = "stencil",
    verbose: bool = True,
    wall_override: float | None = None,
) -> dict:
    """Relax a sessile drop and measure its contact angle.

    ``wall_override`` sets the wall pseudopotential directly (used by the
    calibration sweep, which needs ``psi_w`` as the control variable rather than
    the nominal angle).
    """
    stencil = build_stencil(stencil_name)
    solid = torch.zeros((ny, nx), dtype=torch.bool)
    solid[:wall_rows, :] = True
    solid[ny - wall_rows:, :] = True
    y_wall = float(wall_rows - 1)
    wall_psi = (float(wall_override) if wall_override is not None
                else wall_psi_from_contact_angle(theta_set))
    area = math.pi * R0 * R0
    rho0 = sessile_rho(ny, nx, y_wall, theta_set, area)
    rho0 = torch.where(solid, torch.full_like(rho0, RHO_L), rho0)
    zero = torch.zeros_like(rho0)
    f = equilibrium(rho0, zero, zero)
    f = torch.where(solid.unsqueeze(0), equilibrium(
        torch.full_like(rho0, RHO_L), zero, zero), f)
    mass0 = float((rho0 * (~solid)).sum())
    hist: list[dict] = []
    t0 = time.perf_counter()
    step = 0
    converged = False
    for step in range(1, steps + 1):
        f = step_sc(f, G_eff=G_eff, tau=TAU, stencil=stencil, mode=mode,
                    solid=solid, wall_psi=wall_psi, bounce_back=True)
        if step % sample == 0:
            rho = f.sum(dim=0)
            if not torch.isfinite(rho).all():
                raise RuntimeError(f"contact-angle run diverged (theta={theta_set})")
            meas = contact_angle_from_rho(rho, solid)
            hist.append({"step": step, "theta_deg": meas["theta_deg"],
                         "cos_theta": meas["cos_theta"], "R_fit": meas["R_fit"],
                         "z_c": meas["z_c"], "n_contour": meas["n_contour"]})
            if verbose:
                print(f"  [theta={theta_set}] step {step} theta_meas={meas['theta_deg']:.2f} "
                      f"R_fit={meas['R_fit']:.2f}", flush=True)
            if len(hist) >= 3 and step >= 4000:
                d1 = abs(hist[-1]["theta_deg"] - hist[-2]["theta_deg"])
                d2 = abs(hist[-2]["theta_deg"] - hist[-3]["theta_deg"])
                if d1 < conv_tol * 180.0 and d2 < conv_tol * 180.0:
                    converged = True
                    break
    rho = f.sum(dim=0)
    meas = contact_angle_from_rho(rho, solid)
    fluid = ~solid
    return {
        "theta_set": theta_set,
        "theta_measured": meas["theta_deg"],
        "theta_error_deg": meas["theta_deg"] - theta_set,
        "cos_theta_measured": meas["cos_theta"],
        "R_fit": meas["R_fit"],
        "z_c": meas["z_c"],
        "y_wall": meas["y_wall"],
        "n_contour": meas["n_contour"],
        "wall_psi": wall_psi,
        "theta_from_wall_psi": contact_angle_from_wall_psi(wall_psi),
        "steps": step,
        "converged": converged,
        "mass_drift": abs(float((rho * fluid).sum()) - mass0) / mass0,
        "rho_liq_max": float(rho[fluid].max()),
        "steps_per_s": step / max(time.perf_counter() - t0, 1e-9),
        "history": hist,
        "rho_field": rho.tolist(),
        "status": meas.get("status", "ok"),
    }


# ---------------------------------------------------------------------------
# Gate assembly
# ---------------------------------------------------------------------------


def laplace_gate(rows: list[dict], tol_pct: float = 3.0) -> dict:
    """Three-radius self-consistency of ``sigma = Delta p * R`` + Korteweg anchor."""
    sig = [r["sigma_laplace"] for r in rows]
    mean = sum(sig) / len(sig)
    devs = [abs(s - mean) / abs(mean) * 100.0 for s in sig]
    per_r = []
    for r in rows:
        per_r.append({
            "stencil": r["stencil"],
            "gradient_mode": r.get("gradient_mode", "stencil"),
            "R_init": r["R_init"],
            "R_eq": r["R_eq"],
            "dp_flux": r["dp_flux"],
            "dp_eos": r["dp_eos"],
            "dp_kinetic": r["dp_kinetic"],
            "sigma_laplace": r["sigma_laplace"],
            "sigma_eos": r["sigma_eos"],
            "sigma_K": r["sigma_K"],
            "dev_from_mean_pct": abs(r["sigma_laplace"] - mean) / abs(mean) * 100.0,
            "ratio_sigma_laplace_to_K": r["sigma_laplace"] / r["sigma_K"],
            "ratio_2sigmaR_check": (r["sigma_laplace"] / (2.0 * r["sigma_K"])) * 2.0,
            "u_max": r["u_max"],
            "rho_in": r["rho_in"],
            "rho_out": r["rho_out"],
            "steps": r["steps"],
            "converged": r["converged"],
            "spread_flux_pct": r["spread_flux_pct"],
            "mass_drift": r["mass_drift"],
        })
    sk = [r["sigma_K"] for r in rows]
    sk_mean = sum(sk) / len(sk)
    ratio_k = [r["sigma_laplace"] / r["sigma_K"] for r in rows]
    ratio_mean = sum(ratio_k) / len(ratio_k)
    return {
        "n_radii": len(rows),
        "sigma_mean": mean,
        "max_dev_pct": max(devs),
        "per_radius": per_r,
        "sigma_K_mean": sk_mean,
        "sigma_K_spread_pct": (max(sk) - min(sk)) / abs(sk_mean) * 100.0 if len(sk) > 1 else 0.0,
        "ratio_laplace_over_K": ratio_mean,
        "ratio_laplace_over_K_dev_pct": max(abs(v - ratio_mean) / abs(ratio_mean) * 100.0 for v in ratio_k),
        "pass_three_radius": max(devs) < tol_pct,
        "pass_korteweg_anchor": max(abs(v - ratio_mean) / abs(ratio_mean) * 100.0 for v in ratio_k) < tol_pct,
        "tol_pct": tol_pct,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="laplace",
                    choices=["laplace", "contact", "negative", "gate", "anisotropy"])
    ap.add_argument("--radii", default="20,40,60")
    ap.add_argument("--stencils", default="d2q9")
    ap.add_argument("--thetas", default="60,90,120")
    ap.add_argument("--steps", type=int, default=24000)
    ap.add_argument("--min-steps", type=int, default=4000)
    ap.add_argument("--sample", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=str(REPO / "artifacts" / "sc_surface_tension"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.task in ("laplace", "anisotropy"):
        rows = []
        for st in args.stencils.split(","):
            for R in [float(v) for v in args.radii.split(",")]:
                res = run_droplet(R, stencil_name=st, steps=args.steps,
                                  min_steps=args.min_steps, sample=args.sample)
                name = f"laplace_{st}_R{int(R)}{args.tag}.json"
                (out / name).write_text(json.dumps(_jsonable(res), indent=1))
                rows.append(res)
                print(f"[done] {st} R={R} sigma_L={res['sigma_laplace']:.5f} "
                      f"sigma_K={res['sigma_K']:.5f} spread={res['spread_flux_pct']:.2f}%", flush=True)
        (out / f"laplace_rows{args.tag}.json").write_text(json.dumps(_jsonable(rows), indent=1))
        print(json.dumps(_jsonable(laplace_gate(rows)), indent=1))

    if args.task == "contact":
        res = []
        for th in [float(v) for v in args.thetas.split(",")]:
            r = run_contact_angle(th, steps=args.steps)
            (out / f"contact_theta{int(th)}{args.tag}.json").write_text(
                json.dumps(_jsonable(r), indent=1))
            res.append({k: v for k, v in r.items() if k not in ("rho_field", "history")})
            print(f"[done] theta_set={th} measured={r['theta_measured']:.2f} "
                  f"err={r['theta_error_deg']:+.2f} deg", flush=True)
        (out / f"contact_rows{args.tag}.json").write_text(json.dumps(_jsonable(res), indent=1))

    if args.task == "negative":
        cases = {
            # (i) no interaction at all: a *smooth* density bump (width 8 ke)
            # is used because without the interaction pressure the sharp
            # (width 2) interface of the multiphase state is not an equilibrium
            # of the single-phase solver and would ring.
            "G_eff_zero": dict(R=20.0, stencil_name="d2q9", G_eff=0.0, steps=5000,
                               min_steps=5000, width=8.0),
            # (ii) the faithful kappa -> 0 control: phase separation kept
            # (G_eff = -5), capillary stress removed by a spectrally exact
            # gradient (kappa3 = 0 in eq. 4 of the module).
            "kappa3_zero_spectral": dict(R=20.0, stencil_name="d2q9", mode="spectral",
                                         G_eff=G_EFF, steps=8000, min_steps=8000),
        }
        out_rows = {}
        for name, kw in cases.items():
            res = run_droplet(kw.pop("R"), steps=kw.pop("steps"), min_steps=kw.pop("min_steps"),
                              sample=2000, **kw)
            out_rows[name] = {k: v for k, v in res.items() if k not in ("profile_r", "profile_psi", "dp_theta", "history")}
            (out / f"negative_{name}.json").write_text(json.dumps(_jsonable(res), indent=1))
            print(f"[done] {name} dp_flux={res['dp_flux']:.6f} dp_eos={res['dp_eos']:.6f} "
                  f"rho={res['rho_in']:.3f}/{res['rho_out']:.3f}", flush=True)
        (out / "negative_rows.json").write_text(json.dumps(_jsonable(out_rows), indent=1))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())