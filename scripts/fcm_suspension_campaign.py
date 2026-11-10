#!/usr/bin/env python
"""FCM particle-suspension verification campaign (P167W2-N01).

Four verification lanes, all built on :mod:`tensorlbm.fcm` and compared with the
analytic / literature references in :mod:`tensorlbm.particle_suspension`:

``drag``         single sphere in a body-force driven periodic box
                 (Re = 0.1 ... 10): C_D vs Stokes / Schiller-Naumann, plus a
                 resolution study (d = 8, 12) and a domain study (L/d = 4, 6, 8).
                 The measured drag also yields the FCM *effective radius*
                 ``a_eff`` used to calibrate the suspension volume fraction.
``settling``     free settling sphere (rho_p/rho_f = 2, gravity + buoyancy):
                 terminal velocity vs Stokes x f(Re).
``lubrication``  two spheres approaching along their line of centres with a
                 prescribed relative velocity: FCM force vs the leading-order
                 lubrication asymptote and a Nguyen-Ladd near-field correction.
``tg``           Taylor-Green vortex laden with free spheres: viscous decay rate
                 of the fluid kinetic energy, gamma/gamma_0 vs mu_eff/mu from
                 Batchelor-Green.

Usage::

    python scripts/fcm_suspension_campaign.py --case drag --out artifacts/fcm
    python scripts/fcm_suspension_campaign.py --case tg --out artifacts/fcm

Every result is written as JSON with the exact configuration, so the report can
quote programmatically injected numbers only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch

from tensorlbm import particle_suspension as ps
from tensorlbm.fcm import FCMSimulation, ParticleSet, sphere_particle


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _threads(n: int) -> None:
    if n > 0:
        torch.set_num_threads(n)


def _sphere_mask(shape, centers, radii, device, dtype=torch.float32):
    """Geometric mask of nodes inside any sphere (used to mask body forces)."""
    nz, ny, nx = shape
    z = torch.arange(nz, device=device, dtype=dtype).view(nz, 1, 1)
    y = torch.arange(ny, device=device, dtype=dtype).view(1, ny, 1)
    x = torch.arange(nx, device=device, dtype=dtype).view(1, 1, nx)
    inside = torch.zeros(shape, dtype=torch.bool, device=device)
    for (cx, cy, cz), a in zip(centers, radii):
        d2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
        inside |= d2 <= a * a
    return inside


def _mean(vec) -> float:
    return float(np.mean(np.asarray(vec, dtype=float)))


def _slope(t, y):
    """Least-squares slope of y(t) (y already log-transformed by caller)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.vstack([t, np.ones_like(t)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


def _tg_mode_amplitude(sim, k: float) -> float:
    """Projection of the fluid velocity onto the Taylor-Green mode.

    ``A = sum_x (u_x sin kx cos ky - u_y cos kx sin ky) / sum_x sin^2 kx cos^2 ky``
    is insensitive to the particle-scale perturbation field, so it isolates the
    macroscopic (box-scale) mode whose decay defines ``nu_eff``.
    """
    _, ux, uy, _ = sim.velocity()
    L = sim.shape[-1]
    xi = torch.arange(L, device=ux.device, dtype=ux.dtype)
    sx = torch.sin(k * xi)
    cx = torch.cos(k * xi)
    sx = sx.view(1, 1, L)
    cx = cx.view(1, 1, L)
    sy = torch.sin(k * xi).view(1, L, 1)
    cy = torch.cos(k * xi).view(1, L, 1)
    num = float((ux * sx * cy - uy * cx * sy).mean())
    den = float(((sx * cy) ** 2).mean())
    return num / den


# --------------------------------------------------------------------------- #
# Lane 1: single-sphere drag in a force-driven periodic box
# --------------------------------------------------------------------------- #


@dataclass
class DragConfig:
    L: int = 48
    d: float = 8.0
    nu: float = 0.1
    sigma: float = 1.5
    marker_spacing: float = 1.0
    re_targets: tuple = (0.1, 1.0, 10.0)
    passes: int = 2
    settle_steps: int = 3000
    ramp_steps: int = 1500
    avg_steps: int = 500
    seed: int = 0


def run_drag(cfg: DragConfig, device="cpu") -> dict:
    """Single fixed sphere in a body-force driven periodic box.

    The body force is applied only on the fluid nodes outside the geometric
    sphere, so in the steady state momentum conservation gives

        F_drag = rho * g * N_fluid      (drive == momentum sink)

    which is an unambiguous drag measurement: the FCM only has to deliver the
    correct velocity response.  Two outer passes rescale the drive by
    ``(U_target/U_measured)^1.6`` so that each requested Reynolds number is hit.
    """
    L, d = int(cfg.L), float(cfg.d)
    a = 0.5 * d
    tau = 0.5 + 3.0 * cfg.nu
    mu = cfg.nu  # rho0 = 1
    shape = (L, L, L)
    sim = FCMSimulation(shape, tau, sigma=cfg.sigma, marker_spacing=cfg.marker_spacing,
                        device=device)
    center = (L / 2.0, L / 2.0, L / 2.0)
    part = sphere_particle(center, a, fixed=True, device=device)
    sim.initialize(part)
    mask = _sphere_mask(shape, [center], [a], device)
    fluid = (~mask).to(torch.float32)
    n_fluid = float(fluid.sum())
    box_vol = float(L**3)
    phi_box = (4.0 / 3.0) * math.pi * a**3 / box_vol
    corr = ps.hasimoto_drag_correction(phi_box)
    bf = torch.zeros(3, L, L, L, device=device)
    area = 0.25 * math.pi * d * d
    rows = []
    steps_total = 0
    t_start = time.time()
    for re_target in cfg.re_targets:
        u_target = re_target * cfg.nu / d
        f_target = ps.schiller_naumann_cd(re_target) * 0.5 * u_target**2 * area * corr
        drive = f_target  # total force to be injected by the body force
        row = None
        for outer in range(cfg.passes):
            n_steps = cfg.settle_steps if outer == 0 else cfg.ramp_steps
            bf[0] = (drive / n_fluid) * fluid
            u_hist, f_hist, res_hist = [], [], []
            p_prev = sim.momentum()
            for it in range(n_steps + cfg.avg_steps):
                fp = sim.step(part, body_force=bf, update_particles=False)
                steps_total += 1
                if it >= n_steps:
                    u_hist.append(float(sim.mean_velocity()[0]))
                    f_hist.append(-float(fp[0, 0]))
                    if it % 50 == 0:
                        p_now = sim.momentum()
                        res_hist.append(float((p_now - p_prev)[0]) / n_fluid)
                        p_prev = p_now
            u_mean = _mean(u_hist)
            f_drag = _mean(f_hist)
            re_meas = u_mean * d / cfg.nu
            cd = f_drag / (0.5 * u_mean**2 * area)
            cd_drive = drive / (0.5 * u_mean**2 * area)
            row = {
                "re_target": re_target,
                "re_measured": re_meas,
                "u_mean": u_mean,
                "u_target": u_target,
                "F_drag": f_drag,
                "F_drive": drive,
                "momentum_balance": f_drag / drive,
                "dP_dt_residual": _mean(res_hist) if res_hist else float("nan"),
                "C_D": cd,
                "C_D_from_drive": cd_drive,
                "C_D_schiller_naumann": ps.schiller_naumann_cd(max(re_meas, 1e-9)),
                "C_D_stokes": ps.stokes_cd(max(re_meas, 1e-9)),
                "hasimoto_factor": corr,
                "C_D_from_drive_over_SN": cd_drive / ps.schiller_naumann_cd(max(re_meas, 1e-9)),
                "C_D_from_drive_over_SN_hasimoto": cd_drive
                / (ps.schiller_naumann_cd(max(re_meas, 1e-9)) * corr),
                "a_eff_from_drive": drive / (6.0 * math.pi * mu * u_mean),
                "a_eff_over_a": drive / (6.0 * math.pi * mu * u_mean) / a,
                "passes": outer + 1,
            }
            # rescale the drive for the next pass (drag ~ U^2)
            drive = drive * float(np.clip((u_target / max(u_mean, 1e-12)) ** 1.6, 0.4, 2.5))
        rows.append(row)
        print(
            f"[drag] L={L} d={d} Re_t={re_target:<5} Re={row['re_measured']:8.4f} "
            f"U={row['u_mean']:.6e} C_D_drive/SN_H={row['C_D_from_drive_over_SN_hasimoto']:6.3f} "
            f"bal={row['momentum_balance']:6.3f} a_eff/a={row['a_eff_over_a']:5.3f}",
            flush=True,
        )
    return {
        "lane": "drag",
        "config": {**asdict(cfg), "re_targets": list(cfg.re_targets)},
        "phi_box": phi_box,
        "n_fluid_nodes": n_fluid,
        "steps": steps_total,
        "wall_seconds": time.time() - t_start,
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# Lane 2: free settling sphere
# --------------------------------------------------------------------------- #


@dataclass
class SettleConfig:
    L: int = 32
    d: float = 8.0
    nu: float = 0.1
    density_ratio: float = 2.0
    gravity: float = 2.0e-4
    sigma: float = 1.5
    marker_spacing: float = 1.0
    steps: int = 12000
    avg_steps: int = 1500


def run_settling(cfg: SettleConfig, device="cpu") -> dict:
    L, d = int(cfg.L), float(cfg.d)
    a = 0.5 * d
    tau = 0.5 + 3.0 * cfg.nu
    mu = cfg.nu
    rho_p = cfg.density_ratio
    shape = (L, L, L)
    sim = FCMSimulation(shape, tau, sigma=cfg.sigma, marker_spacing=cfg.marker_spacing, device=device)
    center = (L / 2.0, L / 2.0, L / 2.0)
    part = sphere_particle(center, a, density_ratio=rho_p, device=device)
    sim.initialize(part)
    vol = 4.0 / 3.0 * math.pi * a**3
    # gravity acts along -z, buoyancy included through (rho_p - rho_f)
    f_ext = torch.zeros(1, 3, device=device)
    f_ext[0, 2] = -(rho_p - 1.0) * vol * cfg.gravity
    vz_hist, t_hist = [], []
    for it in range(cfg.steps):
        sim.step(part, external_force=f_ext, remove_mean_momentum=True)
        if it >= cfg.steps - cfg.avg_steps:
            vz_hist.append(float(-part.velocities[0, 2]))
            t_hist.append(it)
    v_terminal = _mean(vz_hist)
    re_meas = v_terminal * d / cfg.nu
    v_stokes = ps.stokes_terminal_velocity(d, rho_p, 1.0, mu, cfg.gravity)
    v_finite = ps.finite_re_terminal_velocity(d, rho_p, 1.0, mu, cfg.gravity)
    row = {
        "v_terminal": v_terminal,
        "v_terminal_std": float(np.std(np.asarray(vz_hist, dtype=float))),
        "Re_measured": re_meas,
        "v_stokes": v_stokes,
        "v_finite_re": v_finite,
        "ratio_to_stokes": v_terminal / v_stokes,
        "ratio_to_finite_re": v_terminal / v_finite,
        "F_drag_measured": float(-sim.last_forces[0, 2]),
        "F_buoyancy": float(-f_ext[0, 2]),
        "stokes_drag_force": ps.stokes_drag_force(mu, a, v_terminal),
    }
    print(f"[settle] {row}", flush=True)
    return {
        "lane": "settling",
        "config": asdict(cfg),
        "rows": [row],
        "steps": cfg.steps,
    }


# --------------------------------------------------------------------------- #
# Lane 3: two-sphere lubrication
# --------------------------------------------------------------------------- #


@dataclass
class LubConfig:
    L: int = 40
    Lz: int = 64
    d: float = 16.0
    nu: float = 0.1
    sigma: float = 1.5
    marker_spacing: float = 1.0
    gaps_over_d: tuple = (0.4, 0.2, 0.1, 0.05, 0.02, 0.01)
    u_rel: float = 2.0e-4
    steps: int = 4000
    avg_steps: int = 800
    h_cut: float = 2.0


def run_lubrication(cfg: LubConfig, device="cpu") -> dict:
    L, Lz, d = int(cfg.L), int(cfg.Lz), float(cfg.d)
    a = 0.5 * d
    tau = 0.5 + 3.0 * cfg.nu
    mu = cfg.nu
    shape = (Lz, L, L)
    rows = []
    for gap_ratio in cfg.gaps_over_d:
        h = gap_ratio * d
        z1 = Lz / 2.0 - (a + h / 2.0)
        z2 = Lz / 2.0 + (a + h / 2.0)
        sim = FCMSimulation(shape, tau, sigma=cfg.sigma, marker_spacing=cfg.marker_spacing, device=device)
        pos = torch.tensor([[L / 2.0, L / 2.0, z1], [L / 2.0, L / 2.0, z2]], device=device)
        rad = torch.full((2,), a, device=device)
        vel = torch.tensor(
            [[0.0, 0.0, +cfg.u_rel / 2.0], [0.0, 0.0, -cfg.u_rel / 2.0]], device=device
        )
        part = ParticleSet(
            positions=pos,
            radii=rad,
            velocities=vel,
            density_ratio=torch.ones(2, device=device),
            fixed=torch.ones(2, dtype=torch.bool, device=device),
        )
        sim.initialize(part)
        f_hist = []
        for it in range(cfg.steps):
            fp = sim.step(part, update_particles=False)
            if it >= cfg.steps - cfg.avg_steps:
                # reaction on particle 1 (pushed in -z) is -fp
                f_hist.append(float(fp[0, 2]))
        f_fcm = _mean(f_hist)
        f_exact = ps.lubrication_force_sphere_sphere(a, a, h, cfg.u_rel, mu)
        a_eff = a * a / (2.0 * a)
        f_lub = 6.0 * math.pi * mu * a_eff**2 * cfg.u_rel * max(0.0, 1.0 / h - 1.0 / cfg.h_cut)
        f_corr = f_fcm + f_lub
        rows.append(
            {
                "gap_over_d": gap_ratio,
                "gap_lattice": h,
                "F_FCM": f_fcm,
                "F_lubrication_asymptote": f_exact,
                "F_lub_correction": f_lub,
                "F_corrected": f_corr,
                "ratio_raw": f_fcm / f_exact,
                "ratio_corrected": f_corr / f_exact,
                "h_over_a": h / a,
            }
        )
        print(
            f"[lub] h/d={gap_ratio:<5} h={h:5.2f} F_FCM={f_fcm:.6e} "
            f"F_exact={f_exact:.6e} raw={f_fcm / f_exact:6.3f} "
            f"corr={f_corr / f_exact:6.3f}",
            flush=True,
        )
    return {
        "lane": "lubrication",
        "config": {**asdict(cfg), "gaps_over_d": list(cfg.gaps_over_d)},
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# Lane 4: Taylor-Green decay with suspended spheres
# --------------------------------------------------------------------------- #


def _place_spheres(n, L, a, rng, min_sep):
    pts = []
    tries = 0
    while len(pts) < n and tries < 200000:
        tries += 1
        p = rng.uniform(0, L, size=3)
        ok = True
        for q in pts:
            dxyz = np.abs(p - q)
            dxyz = np.minimum(dxyz, L - dxyz)  # periodic
            if np.linalg.norm(dxyz) < min_sep:
                ok = False
                break
        if ok:
            pts.append(p)
    if len(pts) < n:
        raise RuntimeError(f"could only place {len(pts)}/{n} spheres")
    return np.asarray(pts)


@dataclass
class TGConfig:
    L: int = 32
    d: float = 8.0
    nu: float = 0.1
    u0: float = 0.02
    sigma: float = 1.5
    marker_spacing: float = 1.0
    phis: tuple = (0.0, 0.05, 0.1, 0.2)
    seeds: tuple = (0, 1, 2, 3)
    steps: int = 900
    fit_from: int = 100
    fit_to: int = 800


def run_tg(cfg: TGConfig, device="cpu") -> dict:
    L = int(cfg.L)
    a = 0.5 * float(cfg.d)
    tau = 0.5 + 3.0 * cfg.nu
    k = 2.0 * math.pi / L
    gamma0 = ps.taylor_green_decay_rate_2d(cfg.nu, k)
    shape = (L, L, L)
    rows = []
    # --- particle-free baseline
    sim = FCMSimulation(shape, tau, device=device)
    sim.initialize()
    xi = torch.arange(L, device=device, dtype=torch.float32)
    X = xi.view(1, 1, L)
    Y = xi.view(1, L, 1)
    ux0 = cfg.u0 * torch.sin(k * X) * torch.cos(k * Y)
    uy0 = -cfg.u0 * torch.cos(k * X) * torch.sin(k * Y)
    from tensorlbm.d3q19 import equilibrium3d

    rho = torch.ones(shape, device=device)
    sim.f = equilibrium3d(rho, ux0.expand(shape).contiguous(), uy0.expand(shape).contiguous(),
                          torch.zeros(shape, device=device))
    e_hist, a_hist = [], []
    for it in range(cfg.steps):
        sim.step(update_particles=False)
        e_hist.append(sim.kinetic_energy())
        a_hist.append(_tg_mode_amplitude(sim, k))
    t = np.arange(cfg.steps)
    sel = (t >= cfg.fit_from) & (t <= cfg.fit_to)
    slope0 = _slope(t[sel], np.log(np.maximum(np.asarray(a_hist)[sel], 1e-30)))
    gamma_meas0 = -slope0
    rows.append(
        {
            "phi": 0.0,
            "seed": None,
            "n_particles": 0,
            "gamma_measured": gamma_meas0,
            "gamma_theory": gamma0,
            "ratio": gamma_meas0 / gamma0,
            "mu_eff_ratio_measured": gamma_meas0 / gamma0,
            "mu_eff_ratio_batchelor_green": 1.0,
            "E0": float(e_hist[0]),
            "E_end": float(e_hist[-1]),
        }
    )
    print(f"[tg] phi=0 gamma={gamma_meas0:.6e} theory={gamma0:.6e} "
          f"ratio={gamma_meas0 / gamma0:.4f}", flush=True)
    for phi in cfg.phis:
        if phi <= 0:
            continue
        n_part = int(round(phi * L**3 / ((4.0 / 3.0) * math.pi * a**3)))
        for seed in cfg.seeds:
            rng = np.random.default_rng(1000 * int(round(phi * 100)) + seed)
            pts = _place_spheres(n_part, L, a, rng, min_sep=2.4 * a)
            sim = FCMSimulation(shape, tau, sigma=cfg.sigma, marker_spacing=cfg.marker_spacing, device=device)
            part = ParticleSet(
                positions=torch.tensor(pts, dtype=torch.float32, device=device),
                radii=torch.full((n_part,), a, device=device),
                velocities=torch.zeros(n_part, 3, device=device),
                density_ratio=torch.ones(n_part, device=device),
                fixed=torch.zeros(n_part, dtype=torch.bool, device=device),
            )
            sim.initialize(part)
            sim.f = equilibrium3d(
                torch.ones(shape, device=device),
                ux0.expand(shape).contiguous(),
                uy0.expand(shape).contiguous(),
                torch.zeros(shape, device=device),
            )
            e_hist, a_hist = [], []
            for it in range(cfg.steps):
                sim.step(part)
                e_hist.append(sim.kinetic_energy()
                              + 0.5 * float((part.masses(sim.rho0).view(-1, 1)
                                             * part.velocities**2).sum()))
                a_hist.append(_tg_mode_amplitude(sim, k))
            sel = (t >= cfg.fit_from) & (t <= cfg.fit_to)
            slope = _slope(t[sel], np.log(np.maximum(np.asarray(a_hist)[sel], 1e-30)))
            gamma_meas = -slope
            phi_real = ps.volume_fraction([a] * n_part, float(L**3))
            rows.append(
                {
                    "phi": phi,
                    "phi_realised": phi_real,
                    "seed": seed,
                    "n_particles": n_part,
                    "gamma_measured": gamma_meas,
                    "gamma_theory": gamma0,
                    "ratio": gamma_meas / gamma0,
                    "mu_eff_ratio_measured": gamma_meas / gamma0,
                    "mu_eff_ratio_batchelor_green": ps.batchelor_green_viscosity_ratio(phi),
                    "E0": float(e_hist[0]),
                    "E_end": float(e_hist[-1]),
                }
            )
            print(f"[tg] phi={phi} seed={seed} n={n_part} "
                  f"ratio={gamma_meas / gamma0:.4f} "
                  f"BG={ps.batchelor_green_viscosity_ratio(phi):.4f}", flush=True)
    return {"lane": "tg", "config": {**asdict(cfg), "phis": list(cfg.phis),
                                     "seeds": list(cfg.seeds)},
            "k": k, "gamma0": gamma0, "rows": rows}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", required=True,
                    choices=["drag", "settling", "lubrication", "tg"])
    ap.add_argument("--out", default="artifacts/fcm_suspension")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--L", type=int, default=None)
    ap.add_argument("--d", type=float, default=None)
    ap.add_argument("--nu", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--passes", type=int, default=None)
    ap.add_argument("--re-targets", type=str, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    _threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    if args.case == "drag":
        cfg = DragConfig()
        if args.L:
            cfg.L = args.L
        if args.d:
            cfg.d = args.d
        if args.nu:
            cfg.nu = args.nu
        if args.steps:
            cfg.settle_steps = args.steps
            cfg.ramp_steps = args.steps
        if args.sigma:
            cfg.sigma = args.sigma
        if args.passes:
            cfg.passes = args.passes
        if args.re_targets:
            cfg.re_targets = tuple(float(x) for x in args.re_targets.split(","))
        res = run_drag(cfg)
        name = f"drag_L{cfg.L}_d{cfg.d:g}_nu{cfg.nu:g}_s{cfg.sigma:g}"
    elif args.case == "settling":
        cfg = SettleConfig()
        if args.L:
            cfg.L = args.L
        if args.d:
            cfg.d = args.d
        if args.nu:
            cfg.nu = args.nu
        if args.steps:
            cfg.steps = args.steps
        if args.sigma:
            cfg.sigma = args.sigma
        res = run_settling(cfg)
        name = f"settling_L{cfg.L}_d{cfg.d:g}_nu{cfg.nu:g}_s{cfg.sigma:g}"
    elif args.case == "lubrication":
        cfg = LubConfig()
        if args.d:
            cfg.d = args.d
        if args.nu:
            cfg.nu = args.nu
        if args.steps:
            cfg.steps = args.steps
        if args.sigma:
            cfg.sigma = args.sigma
        res = run_lubrication(cfg)
        name = f"lubrication_d{cfg.d:g}_nu{cfg.nu:g}_s{cfg.sigma:g}"
    else:
        cfg = TGConfig()
        if args.L:
            cfg.L = args.L
        if args.d:
            cfg.d = args.d
        if args.nu:
            cfg.nu = args.nu
        if args.steps:
            cfg.steps = args.steps
        if args.sigma:
            cfg.sigma = args.sigma
        res = run_tg(cfg)
        name = f"tg_L{cfg.L}_d{cfg.d:g}_nu{cfg.nu:g}_s{cfg.sigma:g}"
    res["wall_seconds"] = time.time() - t0
    if args.tag:
        name += "_" + args.tag
    path = os.path.join(args.out, name + ".json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"wrote {path}  ({res['wall_seconds']:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())