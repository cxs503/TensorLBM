#!/usr/bin/env python3
"""Sliding-mesh KVLCC2 self-propulsion driver (IN62-N02).

Stages (each writes its own JSON into ``artifacts/sliding_mesh_kvlcc2/``):

  verification  analytic Taylor--Couette + interface transparency/conservation
  openwater     sliding-mesh propeller open-water curve at several J
  behindhull    propeller driven by the measured hull wake (thrust identity)
  hull          KVLCC2 static-zone run (towed, and propeller-on disk)
  assemble      decomposition + self-propulsion point + adjudication
  figures       publication figures from the stage JSONs

Example
-------
  python scripts/sliding_mesh_kvlcc2_selfpropulsion.py verification
  python scripts/sliding_mesh_kvlcc2_selfpropulsion.py openwater --js 0.5 0.7 0.9
  python scripts/sliding_mesh_kvlcc2_selfpropulsion.py hull --steps 6000
  python scripts/sliding_mesh_kvlcc2_selfpropulsion.py assemble
  python scripts/sliding_mesh_kvlcc2_selfpropulsion.py figures
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.kvlcc2_selfpropulsion import (  # noqa: E402
    KVLCC2_MODEL,
    KVLCC2_REFERENCE_BANDS,
    b4_70_geometry,
    disk_plane_wake,
    empirical_cross_check,
    model_scale_resistance,
    radial_loading,
    reynolds_wake_scaling,
    run_hull_zone,
    solve_self_propulsion,
    thrust_deduction,
    thrust_identity_wake,
)
from tensorlbm.selfpropulsion import (  # noqa: E402
    KP505Geometry,
    SectionPolar,
    blade_element_momentum,
)
from tensorlbm.sliding_mesh_propulsion import (  # noqa: E402
    RotatingZoneConfig,
    SlidingMeshPropeller,
    run_taylor_couette,
)

ART = ROOT / "artifacts" / "sliding_mesh_kvlcc2"
DEFAULT_STEPS = 2400
DEFAULT_WARMUP = 1200


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def _write(name: str, payload: dict) -> Path:
    ART.mkdir(parents=True, exist_ok=True)
    path = ART / name
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=False))
    print(f"[write] {path}")
    return path


def _read(name: str) -> dict:
    return json.loads((ART / name).read_text())


def b4_70_bem_geometry(diameter_m: float) -> KP505Geometry:
    """B-series (B4-70) blade-element input: chord distribution + pitch + EAR.

    The chord distribution is the standard B-series shape for ``A_E/A_0 = 0.70``
    (tabulated ``c/D`` at nine radii, MODEL grade); the geometry is normalised
    internally to the declared area ratio, so only the *shape* enters.
    """
    rr = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.97)
    cd = (0.150, 0.235, 0.285, 0.315, 0.325, 0.315, 0.285, 0.225, 0.110)
    pd = tuple(0.721 for _ in rr)
    return KP505Geometry(
        n_blades=4,
        diameter=float(diameter_m),
        hub_ratio=0.155,
        expanded_area_ratio=0.70,
        r_over_R=rr,
        chord_over_D=cd,
        pitch_over_D=pd,
    )


def bem_open_water(model=KVLCC2_MODEL, js=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)) -> list[dict]:
    geom = b4_70_bem_geometry(model.propeller_diameter)
    polar = SectionPolar()
    n_rps = model.speed / (0.7 * model.propeller_diameter)
    out = []
    for j in js:
        sol = blade_element_momentum(geom, polar, j, n_rps=n_rps, rho=model.rho)
        out.append(
            {
                "j": float(j),
                "kt": float(sol["kt"]),
                "kq": float(sol["kq"]),
                "eta0": float(sol["eta0"]),
                "thrust": float(sol["thrust"]),
                "torque": float(sol["torque"]),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# stage: verification
# --------------------------------------------------------------------------- #


def stage_verification(args) -> None:
    torch.set_num_threads(args.threads)
    t0 = time.time()
    tc = run_taylor_couette(
        n=args.tc_n,
        r_inner=args.tc_ri,
        r_outer=args.tc_ro,
        zone_radius=args.tc_rz,
        omega=args.tc_omega,
        tau=args.tc_tau,
        steps=args.tc_steps,
        n_theta=args.tc_ntheta,
    )
    tc["wall_time_s"] = time.time() - t0

    # interface transparency: uniform stream through an empty rotating zone
    cfg = RotatingZoneConfig(
        nx=40,
        ny=40,
        nz=40,
        inflow=0.05,
        omega=0.0,
        tau=0.51,
        zone_radius=12.0,
        n_theta=48,
    )
    s = SlidingMeshPropeller(cfg, None, center=(20, 20, 20))
    s.run(200, warmup=0)
    rho, ux, uy, uz = s.macroscopic()
    transparency = {
        "u_mean": float(ux.mean().item()),
        "max_deviation": float((ux - cfg.inflow).abs().max().item()),
        "transverse_max": float(torch.maximum(uy.abs(), uz.abs()).max().item()),
        "mass_drift_rel": (s.mass() - s._mass0) / s._mass0,
        "steps": 200,
    }

    # rotating-frame solid-body rotation: no blades, whole domain in the zone
    cfg2 = RotatingZoneConfig(
        nx=40,
        ny=40,
        nz=40,
        inflow=0.0,
        omega=2.0e-4,
        tau=0.7,
        zone_radius=19.0,
        n_theta=64,
        streamwise="closed",
        sides="solid",
    )
    s2 = SlidingMeshPropeller(cfg2, None, center=(20, 20, 20))
    s2.run(1200, warmup=0)
    _, ux2, uy2, uz2 = s2.macroscopic()
    # in the rotating frame the exact solution is u_rel = 0 (solid-body rotation)
    u_rel = torch.sqrt(ux2 * ux2 + uy2 * uy2 + uz2 * uz2)
    solid_body = {
        "u_rel_max": float(u_rel.max().item()),
        "u_rel_mean": float(u_rel.mean().item()),
        "u_theta_lab_centre_r10": None,
        "omega": cfg2.omega,
        "steps": 1200,
    }

    payload = {
        "taylor_couette": tc,
        "interface_transparency": transparency,
        "solid_body_rotation": solid_body,
        "criteria": {
            "taylor_couette_max_rel_error": tc["max_rel_error"],
            "transparency_max_deviation": transparency["max_deviation"],
        },
    }
    _write("verification.json", payload)


# --------------------------------------------------------------------------- #
# stage: open water
# --------------------------------------------------------------------------- #


def _propeller_cfg(args, geometry) -> RotatingZoneConfig:
    return RotatingZoneConfig(
        nx=args.nx,
        ny=args.tank,
        nz=args.tank,
        inflow=args.inflow,
        omega=0.0,
        tau=args.tau,
        smagorinsky=args.smagorinsky,
        zone_radius=args.d * 0.62,
        n_theta=args.ntheta,
        interface_width=1.5,
        sides="free_slip",
        streamwise="inlet_outlet",
        slip_planes=("y_min", "y_max", "z_min", "z_max"),
    )


def run_open_water_point(args, j: float, geometry, *, label: str = "") -> dict:
    cfg = _propeller_cfg(args, geometry)
    d = geometry.diameter
    n_rps = cfg.inflow / (j * d)
    omega = args.omega_sign * 2.0 * math.pi * n_rps
    cfg = RotatingZoneConfig(**{**cfg.__dict__, "omega": omega})
    solver = SlidingMeshPropeller(
        cfg, geometry, center=(args.nx * 0.5, args.tank * 0.5, args.tank * 0.5)
    )
    t0 = time.time()
    hist = []
    res = None
    chunk = max(1, args.steps // 8)
    done = 0
    while done < args.steps:
        n_now = min(chunk, args.steps - done)
        res = solver.run(
            n_now,
            warmup=max(0, args.warmup - done) if done < args.warmup else 0,
            measure_every=args.measure_every,
        )
        done += n_now
        if res["samples"]:
            hist.append(
                {
                    "step": solver.time,
                    "kt": res["thrust_lattice"] / (1.0 * n_rps**2 * d**4),
                    "kq": res["torque_lattice"] / (1.0 * n_rps**2 * d**5),
                }
            )
    thrust = res["thrust_lattice"]
    torque = res["torque_lattice"]
    kt = thrust / (n_rps**2 * d**4)
    kq = torque / (n_rps**2 * d**5)
    eta0 = j * kt / (2.0 * math.pi * kq) if kq > 0 else float("nan")
    loads = radial_loading(solver, radius=d, nbins=8)
    wake = disk_plane_wake(solver, int(round(args.nx * 0.5)), nbins=16)
    return {
        "label": label or f"J{j:.2f}",
        "j": j,
        "n_rps": n_rps,
        "omega": cfg.omega,
        "omega_sign": args.omega_sign,
        "u_inflow": cfg.inflow,
        "diameter": d,
        "thrust_lattice": thrust,
        "torque_lattice": torque,
        "kt": kt,
        "kq": kq,
        "eta0": eta0,
        "radial_loading": loads,
        "disk_wake_profile": wake["profile"],
        "history": hist,
        "steps": args.steps,
        "warmup": args.warmup,
        "samples": res["samples"],
        "wall_time_s": time.time() - t0,
        "cfg": {
            "nx": cfg.nx,
            "ny": cfg.ny,
            "nz": cfg.nz,
            "tau": cfg.tau,
            "nu": cfg.nu,
            "smagorinsky": cfg.smagorinsky,
            "zone_radius": cfg.zone_radius,
            "n_theta": cfg.n_theta,
            "tip_mach": cfg.omega * d / 2.0,
            "re_d": cfg.inflow * d / cfg.nu,
        },
    }


def stage_openwater(args) -> None:
    torch.set_num_threads(args.threads)
    geometry = b4_70_geometry(args.d)
    pts = [run_open_water_point(args, float(j), geometry) for j in args.js]
    payload = {
        "points": pts,
        "bem_b4_70": bem_open_water(),
        "case": {
            "propeller": "Wageningen B4-70 (Z=4, AE/A0=0.70, P/D=0.721)",
            "diameter_lattice": geometry.diameter,
            "tank": [args.tank, args.tank, args.nx],
            "blockage": geometry.diameter / args.tank,
        },
    }
    _write(args.out or "openwater.json", payload)


# --------------------------------------------------------------------------- #
# stage: hull zone
# --------------------------------------------------------------------------- #


def stage_hull(args) -> None:
    """Towed and propeller-on KVLCC2 runs in the static zone.

    The propeller-on case injects the momentum-conserving actuator-disk loading
    of :func:`tensorlbm.kvlcc2_selfpropulsion.disk_body_force` (MODEL propulsor):
    the axial loading is set to ``R_T^tow / (1 - t_ref)`` so that the run sits at
    the expected self-propulsion thrust, and the measured resistance increment
    gives a *resolved* thrust deduction ``t = (R_SP - R_tow) / T``.  The mild
    circularity (the disk thrust uses a reference ``t``) only sets the operating
    point; ``t`` itself is measured.
    """
    torch.set_num_threads(args.threads)
    towed = run_hull_zone(
        nx=args.hull_nx,
        ny=args.hull_ny,
        nz=args.hull_nz,
        inflow=args.inflow,
        tau=args.hull_tau,
        smagorinsky=args.smagorinsky,
        steps=args.steps,
        warmup=args.warmup,
    )
    r_lat = towed["resistance_lattice"]
    t_ref = KVLCC2_REFERENCE_BANDS["thrust_deduction"]["value"]
    d_over_l = KVLCC2_MODEL.propeller_diameter / KVLCC2_MODEL.length
    d_lat = towed["lattice"]["length"] * d_over_l
    thrust_lat = r_lat / (1.0 - t_ref)
    # shaft torque from the B-series ratio K_Q/K_T at J ~ 0.7, eta_0 ~ 0.65
    torque_lat = 0.171 * thrust_lat * d_lat
    disk = {
        "x_over_l": KVLCC2_MODEL.propeller_x_over_l,
        "diameter_over_l": d_over_l,
        "thrust": thrust_lat,
        "torque": torque_lat,
    }
    propelled = run_hull_zone(
        nx=args.hull_nx,
        ny=args.hull_ny,
        nz=args.hull_nz,
        inflow=args.inflow,
        tau=args.hull_tau,
        smagorinsky=args.smagorinsky,
        steps=args.steps,
        warmup=args.warmup,
        disk=disk,
    )
    t_res = thrust_deduction(
        resistance_towed=r_lat,
        resistance_propelled=propelled["resistance_lattice"],
        thrust=thrust_lat,
    )
    t_res["grade"] = "MODEL (resolved R_SP - R_tow with an actuator-disk propulsor)"
    t_res["disk"] = disk
    t_res["propeller_cells"] = d_lat
    payload = {
        "towed": towed,
        "propelled": propelled,
        "lattice": towed["lattice"],
        "wake_profile": towed["wake"]["profile"],
        "wake_field": towed["wake"]["field"],
        "wake_field_propelled": propelled["wake"]["field"],
        "thrust_deduction_resolved": t_res,
        "geometry": {
            "cb_numerical": towed["cb_numerical"],
            "geometry_ok": towed["geometry_ok"],
            "hull_stats": towed["hull_stats"],
            "solid_cells": towed["solid_cells"],
        },
    }
    _write("hull.json", payload)


# --------------------------------------------------------------------------- #
# stage: behind hull
# --------------------------------------------------------------------------- #


def _lifted_wake_map(hull: dict, args) -> tuple[np.ndarray, dict]:
    """Radial hull wake profile -> propeller-tank inflow map (MODEL lifted)."""
    profile = hull["wake_profile"]
    model = KVLCC2_MODEL
    scaling = reynolds_wake_scaling(
        re_lattice=hull["lattice"]["re_lattice"], re_model=model.reynolds
    )
    # radial interpolation of the deficit, then Re lifting
    r_nodes = np.array([p["r"] for p in profile])
    w_nodes = 1.0 - np.array([p["u_axial"] for p in profile]) / args.inflow
    scaling["wake_lattice_max"] = float(w_nodes.max())
    scaling["wake_lifted_max"] = float(w_nodes.max() * scaling["factor"])

    d = args.d
    yy, zz = np.meshgrid(
        np.arange(args.tank) - args.tank / 2.0,
        np.arange(args.tank) - args.tank / 2.0,
        indexing="ij",
    )
    r = np.sqrt(yy**2 + zz**2) * (d / (2.0 * args.d))  # in units of propeller radius
    r_over_R = r
    w_map = np.interp(r_over_R * r_nodes.max(), r_nodes, w_nodes) * scaling["factor"]
    return w_map, scaling


def run_behind_hull_point(args, geometry, hull: dict) -> dict:
    w_map, scaling = _lifted_wake_map(hull, args)
    profile = args.inflow * (1.0 - w_map)
    cfg = _propeller_cfg(args, geometry)
    d = geometry.diameter
    w_mean_disk = float(
        w_map[
            np.sqrt(
                (np.arange(args.tank)[:, None] - args.tank / 2) ** 2
                + (np.arange(args.tank)[None, :] - args.tank / 2) ** 2
            )
            <= d / 2
        ].mean()
    )
    j_nom = args.j_behind
    n_rps = cfg.inflow * (1.0 - w_mean_disk) / (j_nom * d)
    cfg = RotatingZoneConfig(**{**cfg.__dict__, "omega": args.omega_sign * 2.0 * math.pi * n_rps})
    solver = SlidingMeshPropeller(
        cfg, geometry, center=(args.nx * 0.5, args.tank * 0.5, args.tank * 0.5)
    )
    solver.set_inflow_profile(torch.from_numpy(profile).float())
    t0 = time.time()
    res = solver.run(args.steps, warmup=args.warmup, measure_every=args.measure_every)
    thrust = res["thrust_lattice"]
    torque = res["torque_lattice"]
    kt = thrust / (n_rps**2 * d**4)
    kq = torque / (n_rps**2 * d**5)
    return {
        "j_nominal": j_nom,
        "n_rps": n_rps,
        "wake_mean_disk": w_mean_disk,
        "thrust_lattice": thrust,
        "torque_lattice": torque,
        "kt": kt,
        "kq": kq,
        "eta0": j_nom * kt / (2.0 * math.pi * kq) if kq > 0 else float("nan"),
        "re_scaling": scaling,
        "steps": args.steps,
        "warmup": args.warmup,
        "samples": res["samples"],
        "wall_time_s": time.time() - t0,
        "cfg": {
            "nx": cfg.nx,
            "ny": cfg.ny,
            "nz": cfg.nz,
            "tau": cfg.tau,
            "nu": cfg.nu,
            "smagorinsky": cfg.smagorinsky,
            "inflow": cfg.inflow,
            "tip_mach": cfg.omega * d / 2.0,
        },
    }


def stage_behindhull(args) -> None:
    torch.set_num_threads(args.threads)
    hull = _read("hull.json")
    geometry = b4_70_geometry(args.d)
    pt = run_behind_hull_point(args, geometry, hull)
    _write("behindhull.json", {"point": pt})


# --------------------------------------------------------------------------- #
# stage: assemble
# --------------------------------------------------------------------------- #


def stage_assemble(args) -> None:
    model = KVLCC2_MODEL
    resistance = model_scale_resistance(model)
    r_t = resistance["R_T"]

    open_water = _read(args.openwater)
    pts = sorted(open_water["points"], key=lambda p: p["j"])
    js = [p["j"] for p in pts]
    kts = [p["kt"] for p in pts]
    kqs = [p["kq"] for p in pts]

    behind = None
    try:
        behind = _read("behindhull.json")["point"]
    except FileNotFoundError:
        pass

    # --- wake fraction -----------------------------------------------------
    w_emp = empirical_cross_check(model.cb, 0.35)
    decomposition = {"steps": []}
    w_used = KVLCC2_REFERENCE_BANDS["wake_fraction"]["value"]
    if behind is not None:
        d_lat = b4_70_geometry(args.d).diameter
        ident = thrust_identity_wake(
            thrust=behind["thrust_lattice"],
            n_rps=behind["n_rps"],
            diameter=d_lat,
            speed=behind["cfg"].get("inflow", args.inflow),
            rho=1.0,
            j_table=js,
            kt_table=kts,
        )
        ident["j_nominal"] = behind["j_nominal"]
        ident["wake_imposed_mean"] = behind["wake_mean_disk"]
        ident["consistency_error"] = (
            ident["wake_fraction_identity"] - behind["wake_mean_disk"]
        ) / behind["wake_mean_disk"]
        decomposition["thrust_identity"] = ident
        if 0.0 < ident["wake_fraction_identity"] < 0.8:
            w_used = ident["wake_fraction_identity"]
        decomposition["steps"].append(
            "thrust identity from the resolved propeller in the measured hull wake"
        )

    t_band = KVLCC2_REFERENCE_BANDS["thrust_deduction"]
    t_used = t_band["value"]
    decomposition["thrust_deduction"] = {
        "value_used": t_used,
        "grade": "reference band (no resolved t available)",
        "heckscher_with_w": empirical_cross_check(model.cb, w_used)["thrust_deduction_heckscher"],
    }
    try:
        hull = _read("hull.json")
    except FileNotFoundError:
        hull = None
    if hull is not None and "thrust_deduction_resolved" in hull:
        resolved = hull["thrust_deduction_resolved"]
        t_res = resolved["thrust_deduction"]
        decomposition["thrust_deduction"] = {
            "value_used": t_res,
            "value_resolved": t_res,
            "grade": resolved["grade"],
            "delta_resistance_lattice": resolved["delta_resistance"],
            "resistance_towed_lattice": resolved["resistance_towed"],
            "resistance_propelled_lattice": resolved["resistance_propelled"],
            "thrust_lattice": resolved["thrust"],
            "propeller_cells": resolved.get("propeller_cells"),
            "heckscher_with_w": empirical_cross_check(model.cb, w_used)[
                "thrust_deduction_heckscher"
            ],
            "reference_value": t_band["value"],
        }
        if math.isfinite(t_res) and -0.2 < t_res < 0.8:
            t_used = t_res
        decomposition["thrust_deduction"]["value_used"] = t_used
        decomposition["steps"].append(
            "thrust deduction measured from the towed/propeller-on resistance difference"
        )

    eta_R = 1.0
    if behind is not None:
        jt = decomposition.get("thrust_identity", {}).get("j_thrust_identity", 0.7)
        kq_ow = float(np.interp(jt, js, kqs))
        if behind["kq"] > 0 and kq_ow > 0:
            eta_R = kq_ow / behind["kq"]
            decomposition["kq_open_water_at_jt"] = kq_ow
            decomposition["kq_behind_hull"] = behind["kq"]
    decomposition["eta_R"] = eta_R

    sol = solve_self_propulsion(
        resistance=r_t,
        speed=model.speed,
        diameter=model.propeller_diameter,
        rho=model.rho,
        wake_fraction=w_used,
        thrust_deduction_value=t_used,
        j_table=js,
        kt_table=kts,
        kq_table=kqs,
        eta_R=eta_R,
    )

    ref_w = KVLCC2_REFERENCE_BANDS["wake_fraction"]
    ref_t = KVLCC2_REFERENCE_BANDS["thrust_deduction"]
    errors = {
        "wake_fraction": {
            "value": w_used,
            "reference": ref_w["value"],
            "rel_error": (w_used - ref_w["value"]) / ref_w["value"],
            "within_band": ref_w["band"][0] <= w_used <= ref_w["band"][1],
            "grade": ref_w["grade"],
        },
        "thrust_deduction": {
            "value": t_used,
            "reference": ref_t["value"],
            "rel_error": (t_used - ref_t["value"]) / ref_t["value"],
            "within_band": ref_t["band"][0] <= t_used <= ref_t["band"][1],
            "grade": ref_t["grade"],
        },
        "momentum_identity_residual": {
            "value": sol["residual_rel"],
            "reference": 0.0,
            "note": "T(1-t) - R_T normalised by R_T (solver tolerance 1e-12)",
        },
    }

    payload = {
        "case": {
            "hull": "KVLCC2 (Cb=0.8098), model lambda=58, Lpp=5.5172 m",
            "propeller": "Wageningen B4-70 (Z=4, AE/A0=0.70, P/D=0.721), D=0.17 m",
            "speed_m_s": model.speed,
            "fr": model.foude,
            "re": model.reynolds,
        },
        "resistance": resistance,
        "open_water": {"j": js, "kt": kts, "kq": kqs},
        "bem": open_water.get("bem_b4_70"),
        "decomposition": decomposition,
        "self_propulsion": sol,
        "errors": errors,
        "reference_bands": KVLCC2_REFERENCE_BANDS,
        "empirical": w_emp,
        "evidence": {
            "REAL": [
                "sliding-mesh rotating-zone LBM (interface transparency/conservation, Taylor-Couette)",
                "blade momentum-exchange loads, open-water K_T/K_Q sweep",
                "hull momentum-exchange resistance, disk-plane wake field",
                "thrust identity, self-propulsion root find, power-chain arithmetic",
            ],
            "MODEL": [
                "ITTC-78 resistance split (C_R prescribed)",
                "Reynolds lifting of the lattice wake fraction (delta* ~ Re^-0.2)",
                "thrust deduction from the Heckscher regression (resolved two-zone t out of reach)",
                "BEM B4-70 chord shape (B-series c/D table, area normalised)",
            ],
            "WITHHELD": [
                "EFD-grade adjudication of w and t (KRISO model-test table not in repo)",
                "resolved hull-propeller thrust deduction (grid requirement Lpp/D >~ 500 cells)",
            ],
        },
    }
    _write("selfpropulsion.json", payload)


# --------------------------------------------------------------------------- #
# stage: figures
# --------------------------------------------------------------------------- #


def stage_figures(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ART.mkdir(parents=True, exist_ok=True)
    ver = _read("verification.json")
    ow = _read("openwater.json")
    sp = _read("selfpropulsion.json")

    # ---- figure 1: verification -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    tc = ver["taylor_couette"]
    rs = [s["r"] for s in tc["samples"]]
    ax = axes[0]
    r_exact = np.linspace(rs[0] * 0.9, rs[-1] * 1.05, 100)
    from tensorlbm.sliding_mesh_propulsion import taylor_couette_velocity

    c = tc["config"]
    ax.plot(
        r_exact,
        taylor_couette_velocity(r_exact, c["r_inner"], c["r_outer"], c["omega"]),
        "k-",
        label="exact $v_\\theta = Ar+B/r$",
    )
    ax.plot(rs, [s["v_theta_lbm"] for s in tc["samples"]], "o", ms=6, label="sliding-mesh LBM")
    ax.axvline(c["zone_radius"], color="0.6", ls="--", lw=1)
    ax.text(c["zone_radius"], ax.get_ylim()[0], " sliding interface", fontsize=8, va="bottom")
    ax.set_xlabel("r [lattice units]")
    ax.set_ylabel("$v_\\theta$ [lattice units]")
    ax.set_title(f"Taylor-Couette vs analytic (max err {tc['max_rel_error'] * 100:.2f}%)")
    ax.legend(fontsize=8)
    ax = axes[1]
    rr = [s["r"] for s in tc["samples"]]
    ax.bar(rr, [100 * s["rel_error"] for s in tc["samples"]], width=0.9)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("r [lattice units]")
    ax.set_ylabel("relative error [%]")
    ax.set_title("interface error budget")
    fig.tight_layout()
    fig.savefig(ART / "fig_verification.png", dpi=150)
    plt.close(fig)

    # ---- figure 2: open water ---------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    pts = sorted(ow["points"], key=lambda p: p["j"])
    js = [p["j"] for p in pts]
    bem = ow["bem_b4_70"]
    axes[0].plot(js, [p["kt"] for p in pts], "o-", label="sliding-mesh LBM")
    axes[0].plot([b["j"] for b in bem], [b["kt"] for b in bem], "s--", label="BEM (MODEL)")
    axes[0].set_xlabel("J")
    axes[0].set_ylabel("$K_T$")
    axes[0].legend(fontsize=8)
    axes[1].plot(js, [p["kq"] for p in pts], "o-", label="sliding-mesh LBM")
    axes[1].plot([b["j"] for b in bem], [b["kq"] for b in bem], "s--", label="BEM (MODEL)")
    axes[1].set_xlabel("J")
    axes[1].set_ylabel("$K_Q$")
    axes[1].legend(fontsize=8)
    axes[2].plot(js, [p["eta0"] for p in pts], "o-", label="sliding-mesh LBM")
    axes[2].plot([b["j"] for b in bem], [b["eta0"] for b in bem], "s--", label="BEM (MODEL)")
    axes[2].set_xlabel("J")
    axes[2].set_ylabel("$\\eta_0$")
    axes[2].legend(fontsize=8)
    fig.suptitle(
        "Wageningen B4-70 open water (D=%.0f lattice, Re_D=%.0f)"
        % (pts[0]["diameter"], pts[0]["cfg"]["re_d"])
    )
    fig.tight_layout()
    fig.savefig(ART / "fig_openwater.png", dpi=150)
    plt.close(fig)

    # ---- figure 3: wake field + radial loading ----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    p0 = pts[len(pts) // 2]
    axes[0].plot(
        [q["r"] for q in p0["disk_wake_profile"]],
        [q["u_axial"] for q in p0["disk_wake_profile"]],
        "o-",
    )
    axes[0].axhline(p0["u_inflow"], color="k", ls=":", lw=1, label="U (undisturbed)")
    axes[0].set_xlabel("r [lattice units]")
    axes[0].set_ylabel("$\\bar u_x$")
    axes[0].set_title(f"disk-plane axial velocity (J={p0['j']:.2f})")
    axes[0].legend(fontsize=8)
    axes[1].plot(
        [q["r"] for q in p0["radial_loading"]],
        [q["dT_dr"] for q in p0["radial_loading"]],
        "o-",
        label="$dT/dr$",
    )
    axes[1].plot(
        [q["r"] for q in p0["radial_loading"]],
        [q["dQ_dr"] for q in p0["radial_loading"]],
        "s-",
        label="$dQ/dr$",
    )
    axes[1].set_xlabel("r [lattice units]")
    axes[1].set_ylabel("load density")
    axes[1].set_title("resolved radial loading")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ART / "fig_disk.png", dpi=150)
    plt.close(fig)

    # ---- figure 4: self propulsion ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sol = sp["self_propulsion"]
    hist = sol["history"]
    ns = [h["n"] for h in hist]
    ts = [h["thrust"] for h in hist]
    axes[0].semilogx(ns, ts, "o-", ms=3, label="T(n) from measured $K_T$(J)")
    axes[0].axhline(
        sol["required_thrust"],
        color="r",
        ls="--",
        label="$R_T/(1-t)$ = %.2f N" % sol["required_thrust"],
    )
    axes[0].axvline(sol["n_rps"], color="0.4", ls=":", lw=1)
    axes[0].set_xlabel("n [rps]")
    axes[0].set_ylabel("thrust [N]")
    axes[0].set_title("self-propulsion point (Fr=%.3f)" % sp["case"]["fr"])
    axes[0].legend(fontsize=8)
    pc = sol["power_chain"]
    axes[1].bar(["$P_E$", "$P_T$", "$P_D$"], [pc["P_E"], pc["P_T"], pc["P_D"]])
    for i, (k, v) in enumerate([("$P_E$", pc["P_E"]), ("$P_T$", pc["P_T"]), ("$P_D$", pc["P_D"])]):
        axes[1].text(i, v, f"{v:.1f} W", ha="center", va="bottom", fontsize=8)
    axes[1].set_ylabel("power [W]")
    axes[1].set_title(
        "$\\eta_D$=%.3f  $\\eta_H$=%.3f  $\\eta_0$=%.3f  $\\eta_R$=%.3f"
        % (pc["eta_D"], pc["eta_H"], pc["eta_0"], pc["eta_R"])
    )
    fig.tight_layout()
    fig.savefig(ART / "fig_selfpropulsion.png", dpi=150)
    plt.close(fig)

    # ---- figure 5: decomposition / adjudication --------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    items = [("w", sp["errors"]["wake_fraction"]), ("t", sp["errors"]["thrust_deduction"])]
    xs = np.arange(len(items))
    vals = [it[1]["value"] for it in items]
    refs = [it[1]["reference"] for it in items]
    bands = [
        sp["reference_bands"]["wake_fraction"]["band"],
        sp["reference_bands"]["thrust_deduction"]["band"],
    ]
    ax.bar(xs - 0.18, vals, width=0.34, label="this work")
    ax.bar(xs + 0.18, refs, width=0.34, label="literature value (EFD band)")
    for i, b in enumerate(bands):
        ax.fill_between(
            [i - 0.45, i + 0.45],
            b[0],
            b[1],
            color="0.8",
            zorder=0,
            label="EFD band" if i == 0 else None,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([it[0] for it in items])
    ax.set_ylabel("value")
    ax.set_title("wake fraction / thrust deduction vs KVLCC2 reference")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ART / "fig_adjudication.png", dpi=150)
    plt.close(fig)

    # ---- figure 6: hull wake field, towed vs propeller-on -----------------
    try:
        hull = _read("hull.json")
    except FileNotFoundError:
        hull = None
    if hull is not None:
        field_t = np.asarray(hull["wake_field"], dtype=float)
        field_p = np.asarray(hull.get("wake_field_propelled", field_t), dtype=float)
        vmax = float(max(np.abs(field_t).max(), np.abs(field_p).max(), 1e-12))
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, fld, title in (
            (axes[0], field_t, "towed"),
            (axes[1], field_p, "propeller-on"),
            (axes[2], field_p - field_t, "increment"),
        ):
            lim = vmax if title != "increment" else None
            im = ax.imshow(
                fld, origin="lower", cmap="viridis", vmin=(None if lim is None else 0.0), vmax=lim
            )
            ax.set_title(f"axial velocity, {title}")
            ax.set_xlabel("y")
            ax.set_ylabel("z")
            fig.colorbar(im, ax=ax, fraction=0.046)
        A = hull["thrust_deduction_resolved"]
        fig.suptitle(
            "KVLCC2 propeller-disc plane wake (lattice units)  t_measured=%.4f"
            % A["thrust_deduction"]
        )
        fig.tight_layout()
        fig.savefig(ART / "fig_wake.png", dpi=150)
        plt.close(fig)
    print("[figures] written to", ART)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "stage", choices=["verification", "openwater", "behindhull", "hull", "assemble", "figures"]
    )
    ap.add_argument("--threads", type=int, default=int(os.environ.get("LBM_THREADS", "8")))
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--openwater", type=str, default="openwater.json")
    # propeller tank
    ap.add_argument("--d", type=float, default=40.0)
    ap.add_argument("--nx", type=int, default=120)
    ap.add_argument("--tank", type=int, default=96)
    ap.add_argument("--tau", type=float, default=0.502)
    ap.add_argument("--inflow", type=float, default=0.05)
    ap.add_argument("--smagorinsky", type=float, default=0.12)
    ap.add_argument("--ntheta", type=int, default=72)
    ap.add_argument("--js", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    ap.add_argument("--j-behind", type=float, default=0.7)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--measure-every", type=int, default=4)
    ap.add_argument(
        "--omega-sign",
        type=float,
        default=-1.0,
        help="rotation sense: with this lattice/B4-70 convention omega<0 accelerates the "
        "flow downstream (propulsive); omega>0 makes it a turbine",
    )
    # hull zone
    ap.add_argument("--hull-nx", type=int, default=140)
    ap.add_argument("--hull-ny", type=int, default=56)
    ap.add_argument("--hull-nz", type=int, default=36)
    ap.add_argument("--hull-tau", type=float, default=0.502)
    # Taylor-Couette
    ap.add_argument("--tc-n", type=int, default=40)
    ap.add_argument("--tc-ri", type=float, default=7.0)
    ap.add_argument("--tc-ro", type=float, default=17.0)
    ap.add_argument("--tc-rz", type=float, default=12.0)
    ap.add_argument("--tc-omega", type=float, default=3e-4)
    ap.add_argument("--tc-tau", type=float, default=0.8)
    ap.add_argument("--tc-steps", type=int, default=4000)
    ap.add_argument("--tc-ntheta", type=int, default=64)
    args = ap.parse_args()

    if args.stage == "verification":
        stage_verification(args)
    elif args.stage == "openwater":
        stage_openwater(args)
    elif args.stage == "behindhull":
        stage_behindhull(args)
    elif args.stage == "hull":
        stage_hull(args)
    elif args.stage == "assemble":
        stage_assemble(args)
    elif args.stage == "figures":
        stage_figures(args)


if __name__ == "__main__":
    main()
