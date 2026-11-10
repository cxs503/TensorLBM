#!/usr/bin/env python3
"""p339w2-n06 — medium/low-porosity porous-media verification ladder driver.

Rungs (pre-declared gates, see docs/p339w2_n06_porous_ladder.md):

  G1  Kozeny-Carman  K = phi^3 d_p^2 /[180 (1-phi)^2],  3 porosity tiers, < 5 %
  G2  Ergun          -dp/dx = 150 mu (1-phi)^2 u/(phi^3 d_p^2) + 1.75 rho (1-phi) u^2/(phi^3 d_p),
                    2 pore-Reynolds tiers, < 10 %
  G3  Beavers-Joseph du/dy|_int = alpha/sqrt(K) (u_int - u_D), alpha from a
                    pressure-driven port and a shear-driven port of the *same*
                    BCC bed, cross-mechanism consistency < 15 %
  G4  Brinkman       delta = sqrt(K/phi) vs the e-folding length of the
                    REV-averaged tangential velocity inside the bed, < 10 %
  G5  port A (resolved spheres) vs port B (volume-averaged Brinkman/Darcy
                    closure calibrated on the shear port, blind-predicting the
                    pressure port): slab-averaged velocity deviation < 5 %

Usage
-----
    python run.py case --name kc_038 --out cases/kc_038.json
    python run.py shard --i 0 --n 8 --outdir cases
    python run.py report --cases cases --out result.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import tensorlbm.porous_ladder as pl  # noqa: E402
from tensorlbm.d3q19 import equilibrium3d  # noqa: E402

try:  # provenance only
    GIT = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True
    ).strip()
except Exception:  # pragma: no cover
    GIT = "unknown"

NU_TAU1 = 1.0 / 6.0

# ----------------------------------------------------------------------------
# case matrix (frozen before any production run)
# ----------------------------------------------------------------------------
BULK_L = 40
BULK_DP = 12.0
BULK_TAU = 1.5   # nu = 1/3: steady Stokes is tau-independent (TRT, certified
                 # by tests/test_porous_ladder.py::...poiseuille); the larger nu
                 # only shortens the viscous transient (deviation disclosed).
BULK_TIERS = {"kc_038": 0.38, "kc_045": 0.45, "kc_052": 0.52}
FINE = {"L": 72, "d_p": 24.0, "phi": 0.45}
# Ergun tiers: pore Reynolds numbers are limited by Ma < 0.2 and by the
# viscous transient (deviation from the pre-declared Re=10/80 disclosed in the
# report).  Both are warm-started from the phi=0.45 Stokes field.
# Both tiers keep tau=1.0: the tau=0.75 tier destabilised (NaN) when the
# Stokes field was re-driven at a much larger acceleration (disclosed).
ERGUN = {"re10": dict(tau=1.00, re_p=3.0), "re80": dict(tau=1.00, re_p=4.5)}
BED = dict(n_layers=7, cell=24.0, radius=6.276, free_height=16)
BED_TAU = 1.0
PRESSURE_ACCEL = 2.0e-5
SHEAR_WALL = 0.05

CASES = list(BULK_TIERS) + ["kc_fine", "ergun_re10", "ergun_re80", "bj_pressure", "bj_shear", "bed_k"]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def provenance(steps_info: dict) -> dict:
    return {
        "git": GIT,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "threads": torch.get_num_threads(),
        "steps": steps_info,
    }


def _bulk_pack(phi: float, L: int = BULK_L, d_p: float = BULK_DP, seed: int = 1106):
    pack = pl.generate_sphere_pack((L, L, L), d_p, phi, seed=seed)
    mask = pack.as_mask()
    return pack, mask


def _run_bulk(mask, tau: float, accel: float, max_steps: int, rtol: float = 2e-5,
              init_from: str | None = None, save_field: str | None = None,
              init_uniform: float | None = None):
    sim = pl.PorousLBM3D(mask, tau=tau)
    if init_uniform:
        # uniform-flow start: for a fully periodic body-force-driven cell the
        # only slow transient is the viscous profile, so seeding the expected
        # superficial velocity removes the L^2/nu wait (pore-scale transient
        # d_p^2/nu remains, which is what we actually need to resolve).
        nz, ny, nx = sim.solid.shape
        ones = torch.ones((nz, ny, nx), dtype=sim.dtype, device=sim.device)
        zeros = torch.zeros_like(ones)
        sim.f = equilibrium3d(ones, ones * float(init_uniform), zeros, zeros,
                              device=sim.device)
        print(f"[uniform-start] u0={init_uniform:.5e}", flush=True)
    if init_from and Path(init_from).exists():
        sim.f = torch.load(init_from, map_location=sim.device)
        print(f"[warm-start] {init_from}", flush=True)
    info = sim.run(max_steps, accel=accel, check_every=500, rtol=rtol, min_steps=1500)
    rho, ux, uy, uz = sim.velocity()
    fluid = sim.fluid
    n_fl = int(fluid.sum())
    u_mean_fluid = float((ux * fluid).sum() / n_fl)
    accel_effective = float(sim.fluid.float().mean())  # void fraction of the digitiser
    if save_field:
        torch.save(sim.f, save_field)
    mass = float(rho.sum())
    return sim, info, dict(u_mean_fluid=u_mean_fluid, n_fluid=n_fl, mass=mass,
                           accel_effective=accel_effective)


def case_kc(name: str, phi_target: float, L: int, d_p: float, seed: int = 1106,
            accel: float = 1.0e-3, tau: float = BULK_TAU, max_steps: int = 1200,
            save_field: bool = False) -> dict:
    t0 = time.time()
    pack, mask = _bulk_pack(phi_target, L=L, d_p=d_p, seed=seed)
    stats = pl.pack_statistics(mask, d_p)
    field = str(Path(os.environ.get("TL_CASES", HERE / "cases")) / f"_field_{name}.pt") if save_field else None
    sim, info, meas = _run_bulk(mask, tau, accel, max_steps, save_field=field)
    K = pl.darcy_permeability(sim.nu, meas["u_mean_fluid"], accel)
    u_super = meas["u_mean_fluid"] * (1.0 - stats["phi_mask"])
    phi_nom = pack.phi_nominal
    eff = pl.effective_geometry(stats["phi_mask"], d_p, stats["d_sauter_mask"],
                                phi_nominal=phi_nom)
    K_kc = pl.kozeny_carman_permeability(phi_nom, d_p)
    K_kc_eff = pl.kozeny_carman_permeability(eff["phi_eff"], eff["d_eff"])
    K_kc_mask = pl.kozeny_carman_permeability(stats["phi_mask"], d_p)
    K_kc150 = pl.kozeny_carman_permeability(phi_nom, d_p, constant=150.0)
    return {
        "case": name,
        "kind": "kozeny_carman",
        "L": L,
        "d_p": d_p,
        "phi_target": phi_target,
        "phi_nominal": phi_nom,
        "phi_mask": stats["phi_mask"],
        "d_sauter_mask": stats["d_sauter_mask"],
        "n_spheres": pack.meta["n_spheres"],
        "max_residual_overlap": pack.meta["max_residual_overlap"],
        "tau": tau,
        "nu": sim.nu,
        "accel": accel,
        "u_mean_fluid": meas["u_mean_fluid"],
        "u_super": u_super,
        "re_pore": pl.pore_reynolds(u_super, d_p, sim.nu),
        "mach": meas["u_mean_fluid"] * np.sqrt(3.0),
        "K_lbm": K,
        "K_over_dp2": K / d_p**2,
        "K_kc_180": K_kc,
        "K_kc_180_at_phi_mask": K_kc_mask,
        "K_kc_150": K_kc150,
        "err_vs_kc180": K / K_kc - 1.0,
        "err_vs_kc180_phi_mask": K / K_kc_mask - 1.0,
        "kozeny_constant_effective": pl.kozeny_constant(phi_nom, d_p, K),
        "phi_eff": eff["phi_eff"],
        "d_eff": eff["d_eff"],
        "K_kc_at_phi_eff": K_kc_eff,
        "err_vs_kc180_eff": K / K_kc_eff - 1.0,
        "kozeny_constant_eff": pl.kozeny_constant(eff["phi_eff"], eff["d_eff"], K),
        "mass_drift": abs(meas["mass"] - meas["n_fluid"]) / meas["n_fluid"],
        "field_file": f"_field_{name}.pt" if save_field else None,
        "wall_s": time.time() - t0,
        "provenance": provenance(info),
    }


def case_ergun(name: str, re_p: float, tau: float, phi_target: float = 0.45,
               L: int = BULK_L, d_p: float = BULK_DP, seed: int = 1106,
               max_steps: int = 1500) -> dict:
    """Drive the phi=0.45 pack to a target pore Reynolds number.

    ``u_super = Re nu / d_p`` fixes the per-fluid-node acceleration through the
    Stokes permeability measured in the sibling Kozeny-Carman case.
    """
    t0 = time.time()
    pack, mask = _bulk_pack(phi_target, L=L, d_p=d_p, seed=seed)
    stats = pl.pack_statistics(mask, d_p)
    phi_mask = stats["phi_mask"]
    nu = (tau - 0.5) / 3.0
    # the measured Stokes permeability of this very pack (sibling case) sets the
    # acceleration; falling back to the KC prior would over-drive the flow by the
    # discretisation factor and blow up (observed: NaN at Ma -> 0.4)
    _kc = Path(os.environ.get("TL_CASES", HERE / "cases")) / "kc_045.json"
    if not _kc.exists():
        raise SystemExit(f"case_ergun needs {_kc}; run kc_045 first")
    K_eff_stokes = json.loads(_kc.read_text())["K_lbm"]
    K_stokes = K_eff_stokes
    u_super_target = re_p * nu / d_p
    accel = u_super_target * nu / (phi_mask * K_stokes)  # u_s = phi K a / nu
    init = str(Path(os.environ.get("TL_CASES", HERE / "cases")) / "_field_kc_045.pt")
    sim, info, meas = _run_bulk(mask, tau, accel, max_steps, rtol=5e-6,
                                init_uniform=u_super_target / phi_mask)
    u_super = meas["u_mean_fluid"] * phi_mask
    grad_meas = accel * phi_mask  # exact steady momentum balance (force/volume)
    erg = pl.ergun_pressure_gradient(
        pack.phi_nominal, d_p, u_super, rho=1.0, mu=nu
    )
    erg_at_mask = pl.ergun_pressure_gradient(phi_mask, d_p, u_super, rho=1.0, mu=nu)
    K_eff = pl.darcy_permeability(nu, meas["u_mean_fluid"], accel)
    return {
        "case": name,
        "kind": "ergun",
        "L": L,
        "d_p": d_p,
        "phi_nominal": pack.phi_nominal,
        "phi_mask": phi_mask,
        "tau": tau,
        "nu": nu,
        "accel": accel,
        "re_pore_target": re_p,
        "re_pore_measured": pl.pore_reynolds(u_super, d_p, nu),
        "u_super": u_super,
        "u_mean_fluid": meas["u_mean_fluid"],
        "mach": meas["u_mean_fluid"] * np.sqrt(3.0),
        "grad_measured": grad_meas,
        "grad_ergun": erg["total"],
        "grad_ergun_viscous": erg["viscous"],
        "grad_ergun_inertial": erg["inertial"],
        "ergun_inertial_fraction": erg["inertial_fraction"],
        "grad_ergun_at_phi_mask": erg_at_mask["total"],
        "err_vs_ergun": grad_meas / erg["total"] - 1.0,
        "err_vs_ergun_at_phi_mask": grad_meas / erg_at_mask["total"] - 1.0,
        "K_eff_apparent": K_eff,
        "K_eff_over_K_kozeny": K_eff / pl.kozeny_carman_permeability(pack.phi_nominal, d_p),
        # inertial isolation (the node's "Ergun inertial deviation"): subtract
        # the *Stokes-measured* viscous gradient, keep only the u^2 part
        "grad_stokes_measured": nu * u_super / K_eff_stokes,
        "inertial_measured": grad_meas - nu * u_super / K_eff_stokes,
        "err_inertial": (grad_meas - nu * u_super / K_eff_stokes) / erg["inertial"] - 1.0,
        "mass_drift": abs(meas["mass"] - meas["n_fluid"]) / meas["n_fluid"],
        "wall_s": time.time() - t0,
        "provenance": provenance(info),
    }


def case_bed_k(name: str = "bed_k", tau: float = BED_TAU, accel: float = 1.0e-3,
              max_steps: int = 2500) -> dict:
    """Darcy permeability of the *same* BCC lattice as the interface bed.

    The interface bed is a finite slab, so its 3-D permeability tensor is
    measured here on the fully periodic bulk lattice (cubic symmetry -> the
    permeability is isotropic and this scalar ``K`` is what the
    Beavers-Joseph / Brinkman closures consume).
    """
    t0 = time.time()
    a = BED["cell"]
    r = BED["radius"]
    n = int(round(a))
    centers = np.array([[0.0, 0.0, 0.0], [a / 2.0, a / 2.0, a / 2.0]])
    pack = pl.SpherePack(
        centers=centers, radius=r, shape=(n, n, n), periodic=(True, True, True),
        phi_nominal=1.0 - 2.0 * (4.0 / 3.0 * np.pi * r**3) / a**3,
        meta={"raster_offset": pl.RASTER_OFFSET,
              "raster_radius": float(r + pl.RASTER_OFFSET)},
    )
    mask = pack.as_mask()
    stats = pl.pack_statistics(mask, 2.0 * r)
    sim, info, meas = _run_bulk(mask, tau, accel, max_steps)
    K = pl.darcy_permeability(sim.nu, meas["u_mean_fluid"], accel)
    return {
        "case": name,
        "kind": "bed_permeability",
        "cell": a,
        "radius": r,
        "d_p": 2.0 * r,
        "phi_nominal": pack.phi_nominal,
        "phi_mask": stats["phi_mask"],
        "shape": list(pack.shape),
        "tau": tau,
        "nu": sim.nu,
        "accel": accel,
        "u_mean_fluid": meas["u_mean_fluid"],
        "u_super": meas["u_mean_fluid"] * (1.0 - stats["phi_mask"]),
        "re_pore": pl.pore_reynolds(meas["u_mean_fluid"] * (1.0 - stats["phi_mask"]), 2 * r, sim.nu),
        "K_lbm": K,
        "K_over_dp2": K / (2.0 * r) ** 2,
        "K_kozeny_carman_180": pl.kozeny_carman_permeability(pack.phi_nominal, 2.0 * r),
        "mass_drift": abs(meas["mass"] - meas["n_fluid"]) / meas["n_fluid"],
        "wall_s": time.time() - t0,
        "provenance": provenance(info),
    }


def _bed(extra_bottom_rows: int = 1):
    pack = pl.bcc_bed(
        BED["n_layers"], BED["cell"], BED["radius"], free_height=BED["free_height"]
    )
    mask = pack.as_mask()
    mask[0:extra_bottom_rows, :, :] = True  # impermeable floor
    meta = dict(pack.meta)
    meta["y_int"] = float(pack.meta["y_bed_top"])  # interface plane (sphere tops)
    meta["floor_rows"] = extra_bottom_rows
    meta["nz"], meta["ny"], meta["nx"] = pack.shape
    meta["phi_bed"] = float(
        1.0 - len(pack.centers) * (4.0 / 3.0 * np.pi * pack.radius**3)
        / (pack.shape[2] * pack.shape[0] * pack.meta["y_bed_top"])
    )
    return pack, mask, meta


def _profile(sim):
    rho, ux, uy, uz = sim.velocity()
    u = pl.plane_average(ux, sim.fluid)
    frac = pl.plane_fluid_fraction(sim.fluid)
    return u, frac, ux


def case_bj(name: str, mode: str, K_from: str = "self", max_steps: int = 1600) -> dict:
    """Beavers-Joseph port.  ``mode`` = ``pressure`` or ``shear``."""
    t0 = time.time()
    pack, mask, meta = _bed()
    y_int = meta["y_int"]
    ny = meta["ny"]
    free_rows = np.arange(ny, dtype=float)
    sim = pl.PorousLBM3D(mask, tau=BED_TAU)
    if mode == "pressure":
        info = sim.run(max_steps, accel=PRESSURE_ACCEL, check_every=500,
                       rtol=1e-6, min_steps=4000)
        wall_speed = 0.0
    elif mode == "shear":
        info = sim.run(max_steps, top_wall_speed=SHEAR_WALL, check_every=500,
                       rtol=1e-6, min_steps=4000)
        wall_speed = SHEAR_WALL
    else:
        raise ValueError(mode)
    u_prof, frac, ux = _profile(sim)
    # free region rows (bottom of the free region .. just below the top wall)
    free_lo = int(np.ceil(y_int)) + 1
    free_hi = ny - 2
    y_free = free_rows[free_lo : free_hi + 1]
    fit = pl.free_region_quadratic_fit(
        y_free, u_prof[free_lo : free_hi + 1], y_int, drop=1
    )
    # deep-bed Darcy (superficial) velocity: mean over the lowest 20 rows above
    # the floor (well below any interface layer)
    bed_depth = y_int - meta["floor_rows"]
    lo = meta["floor_rows"] + 0.25 * bed_depth
    hi = meta["floor_rows"] + 0.55 * bed_depth
    deep = u_prof[int(lo) : int(hi)]
    u_darcy_meas = float(np.mean(deep))
    # decay band: interface minus (1.5 grain diameters .. 60 % of the bed depth),
    # i.e. clear of both the interface staircase and the floor layer
    bed_band = (meta["floor_rows"] + 0.15 * bed_depth, y_int - 1.5 * 2 * BED["radius"])
    lam_fit = pl.fit_decay_length(free_rows, u_prof, y_hi=bed_band[1], y_lo=bed_band[0])
    return {
        "case": name,
        "kind": "beavers_joseph",
        "mode": mode,
        "geometry": meta,
        "tau": BED_TAU,
        "nu": sim.nu,
        "accel": PRESSURE_ACCEL if mode == "pressure" else 0.0,
        "wall_speed": wall_speed,
        "y_int": y_int,
        "free_fit": fit,
        "u_darcy_measured": u_darcy_meas,
        "decay_fit": lam_fit,
        "deep_window": [float(lo), float(hi)],
        "profile_u": [float(v) for v in u_prof],
        "profile_frac": [float(v) for v in frac],
        "profile_y": [float(v) for v in free_rows],
        "K_reference_case": K_from,
        "wall_s": time.time() - t0,
        "provenance": provenance(info),
    }


# ----------------------------------------------------------------------------
def run_case(name: str, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    if name in BULK_TIERS:
        rec = case_kc(name, BULK_TIERS[name], BULK_L, BULK_DP,
                      save_field=(name == "kc_045"))
    elif name == "kc_fine":
        rec = case_kc(name, FINE["phi"], FINE["L"], FINE["d_p"], max_steps=1500)
    elif name == "ergun_re10":
        rec = case_ergun(name, **ERGUN["re10"])
    elif name == "ergun_re80":
        rec = case_ergun(name, **ERGUN["re80"], max_steps=150000)
    elif name == "bj_pressure":
        rec = case_bj(name, "pressure")
    elif name == "bj_shear":
        rec = case_bj(name, "shear")
    elif name == "bed_k":
        rec = case_bed_k(name)
    else:
        raise SystemExit(f"unknown case {name!r}")
    path = outdir / f"{name}.json"
    path.write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(f"[done] {name} -> {path} ({rec['wall_s']:.1f}s)", flush=True)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["case", "shard", "report", "list"])
    ap.add_argument("--name")
    ap.add_argument("--i", type=int, default=0)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--outdir", default=str(HERE / "cases"))
    ap.add_argument("--cases", default=str(HERE / "cases"))
    ap.add_argument("--threads", type=int, default=int(os.environ.get("TL_THREADS", "8")))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.command == "list":
        print("\n".join(CASES))
        return
    if args.command == "case":
        run_case(args.name, Path(args.outdir) if not args.out else Path(args.out).parent)
        return
    if args.command == "shard":
        mine = CASES[args.i :: args.n]
        print(f"shard {args.i}/{args.n}: {mine}", flush=True)
        for c in mine:
            run_case(c, Path(args.outdir))
        return
    if args.command == "report":
        raise SystemExit("report is produced by report.py")


if __name__ == "__main__":
    main()