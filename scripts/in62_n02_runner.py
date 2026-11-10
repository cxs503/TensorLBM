#!/usr/bin/env python3
"""Resumable chunked runner for the IN62-N02 sliding-mesh campaign.

The two-zone solver costs ~0.2 s/step on a control-plane CPU regardless of grid
size (per-step Python overhead dominates), so a 600-step propeller point does not
fit into a single interactive command, and the shared cluster queue can be
saturated for long periods.  This runner therefore advances a stage for a wall
clock budget, checkpoints the lattice state to disk, and continues on the next
invocation::

    python scripts/in62_n02_runner.py openwater --j 0.7 --tag ow07
    python scripts/in62_n02_runner.py openwater --j 0.7 --tag ow07   # continue
    python scripts/in62_n02_runner.py hull --mode towed
    python scripts/in62_n02_runner.py hull --mode propelled
    python scripts/in62_n02_runner.py behindhull --j 0.7

Each stage writes the same JSON structure as
``scripts/sliding_mesh_kvlcc2_selfpropulsion.py`` (``ow_<tag>.json``, ``hull.json``,
``behindhull.json``) so that ``assemble`` / ``figures`` / ``in62_n02_summary.py``
consume them unchanged.  The checkpoint file is removed when a stage finishes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.kvlcc2_selfpropulsion import (  # noqa: E402
    KVLCC2_MODEL,
    _hull_layout,
    b4_70_geometry,
    disk_body_force,
    disk_plane_wake,
    radial_loading,
)
from tensorlbm.ship_cad import build_hull_mask  # noqa: E402
from tensorlbm.sliding_mesh_propulsion import (  # noqa: E402
    RotatingZoneConfig,
    SlidingMeshPropeller,
)

ART = ROOT / "artifacts" / "sliding_mesh_kvlcc2"
ART.mkdir(parents=True, exist_ok=True)


def _state_path(tag: str) -> Path:
    return ART / f"_state_{tag}.pt"


def run_chunked(
    solver: SlidingMeshPropeller,
    tag: str,
    *,
    steps: int,
    warmup: int,
    budget_s: float,
    measure_every: int = 4,
    chunk: int = 20,
) -> dict:
    """Advance ``solver`` towards ``steps`` within ``budget_s``, checkpointing."""
    path = _state_path(tag)
    st = {"done": 0, "t_sum": 0.0, "q_sum": 0.0, "samples": 0, "history": []}
    if path.exists():
        st = torch.load(path, weights_only=False)
        solver.f = st["f"]
        solver.theta = st["theta"]
        solver.time = st["time"]

    t0 = time.time()
    while st["done"] < steps:
        if time.time() - t0 > budget_s:
            break
        n = min(chunk, steps - st["done"])
        res = solver.run(
            n,
            warmup=0 if st["done"] >= warmup else max(0, warmup - st["done"]),
            measure_every=measure_every,
        )
        st["done"] += n
        if res["samples"]:
            st["t_sum"] += res["thrust_lattice"] * res["samples"]
            st["q_sum"] += res["torque_lattice"] * res["samples"]
            st["samples"] += res["samples"]
            st["history"].append(
                {
                    "step": st["done"],
                    "thrust": st["t_sum"] / st["samples"],
                    "torque": st["q_sum"] / st["samples"],
                }
            )
        st["f"] = solver.f
        st["theta"] = solver.theta
        st["time"] = solver.time
        torch.save(st, path)

    out = {
        "steps_done": st["done"],
        "steps_target": steps,
        "samples": st["samples"],
        "thrust": st["t_sum"] / max(st["samples"], 1),
        "torque": st["q_sum"] / max(st["samples"], 1),
        "history": st["history"],
        "complete": st["done"] >= steps,
        "theta": solver.theta,
    }
    if out["complete"] and path.exists():
        path.unlink()
    return out


def _progress(tag: str, out: dict) -> None:
    print(
        f"[{tag}] {out['steps_done']}/{out['steps_target']} steps "
        f"T={out['thrust']:+.4f} Q={out['torque']:+.4f} samples={out['samples']} "
        f"complete={out['complete']}",
        flush=True,
    )


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #


def stage_openwater(args) -> None:
    d = args.d
    geometry = b4_70_geometry(d)
    n_rps = args.inflow / (args.j * d)
    omega = args.omega_sign * 2.0 * math.pi * n_rps
    cfg = RotatingZoneConfig(
        nx=args.nx,
        ny=args.tank,
        nz=args.tank,
        inflow=args.inflow,
        omega=omega,
        tau=args.tau,
        smagorinsky=args.smagorinsky,
        zone_radius=d * 0.62,
        n_theta=args.ntheta,
        sides="free_slip",
        streamwise="inlet_outlet",
        slip_planes=("y_min", "y_max", "z_min", "z_max"),
    )
    solver = SlidingMeshPropeller(
        cfg, geometry, center=(args.nx * 0.5, args.tank * 0.5, args.tank * 0.5)
    )
    tag = f"ow_{args.tag}"
    out = run_chunked(
        solver,
        tag,
        steps=args.steps,
        warmup=args.warmup,
        budget_s=args.budget,
        measure_every=args.measure_every,
    )
    _progress(tag, out)
    if not out["complete"]:
        return
    thrust, torque = out["thrust"], out["torque"]
    kt = thrust / (n_rps**2 * d**4)
    kq = torque / (n_rps**2 * d**5)
    payload = {
        "label": f"J{args.j:.2f}",
        "j": args.j,
        "n_rps": n_rps,
        "omega": omega,
        "omega_sign": args.omega_sign,
        "u_inflow": args.inflow,
        "diameter": d,
        "thrust_lattice": thrust,
        "torque_lattice": torque,
        "kt": kt,
        "kq": kq,
        "eta0": args.j * kt / (2.0 * math.pi * kq) if kq > 0 else float("nan"),
        "radial_loading": radial_loading(solver, radius=d, nbins=8),
        "disk_wake_profile": disk_plane_wake(solver, int(round(args.nx * 0.5)), nbins=16)[
            "profile"
        ],
        "history": [
            {
                "step": h["step"],
                "kt": h["thrust"] / (n_rps**2 * d**4),
                "kq": h["torque"] / (n_rps**2 * d**5),
            }
            for h in out["history"]
        ],
        "steps": args.steps,
        "warmup": args.warmup,
        "samples": out["samples"],
        "cfg": {
            "nx": cfg.nx,
            "ny": cfg.ny,
            "nz": cfg.nz,
            "tau": cfg.tau,
            "nu": cfg.nu,
            "smagorinsky": cfg.smagorinsky,
            "zone_radius": cfg.zone_radius,
            "n_theta": cfg.n_theta,
            "tip_mach": abs(omega) * d / 2.0,
            "re_d": args.inflow * d / cfg.nu,
        },
    }
    (ART / f"ow_{args.tag}.json").write_text(json.dumps(payload, indent=1))
    print(
        f"[{tag}] wrote ow_{args.tag}.json  K_T={kt:+.4f} 10K_Q={10 * kq:+.4f} eta0={payload['eta0']:+.3f}"
    )


def _hull_solver(args, disk: dict | None):
    nx, ny, nz = args.hull_nx, args.hull_ny, args.hull_nz
    length, beam, draft, cx, cy, cz_keel = _hull_layout(nx, ny, nz)
    mask, stats = build_hull_mask(
        "kvlcc2",
        nx,
        ny,
        nz,
        cx=cx,
        cy=cy,
        cz_keel=cz_keel,
        length=length,
        beam=beam,
        draft=draft,
    )
    cfg = RotatingZoneConfig(
        nx=nx,
        ny=ny,
        nz=nz,
        inflow=args.inflow,
        omega=0.0,
        tau=args.tau,
        smagorinsky=args.smagorinsky,
        zone_radius=1.0,
        n_theta=8,
        sides="free_slip",
        streamwise="inlet_outlet",
        slip_planes=("z_min",),
    )
    solver = SlidingMeshPropeller(cfg, None, center=(cx, cy, nz * 0.5), outer_solid=mask)
    if disk is not None:
        fx, fy, fz, region = disk_body_force(solver, **disk)
        solver.set_body_force(region, fx, fy, fz)
    return solver, stats, (length, beam, draft, cx, cy)


def stage_hull(args) -> None:
    model = KVLCC2_MODEL
    towed_solver, stats, geom = _hull_solver(args, None)
    res_t = run_chunked(
        towed_solver,
        "hull_tow",
        steps=args.steps,
        warmup=args.warmup,
        budget_s=args.budget,
        measure_every=args.measure_every,
    )
    _progress("hull_tow", res_t)
    if not res_t["complete"]:
        return
    r_lat = -res_t["thrust"]  # drag = -force on the hull along the flow
    t_ref = args.t_ref
    d_over_l = model.propeller_diameter / model.length
    length = geom[0]
    d_lat = length * d_over_l
    thrust_lat = r_lat / (1.0 - t_ref)
    torque_lat = 0.171 * thrust_lat * d_lat
    disk = {
        "x_over_l": model.propeller_x_over_l,
        "diameter_over_l": d_over_l,
        "thrust": thrust_lat,
        "torque": torque_lat,
    }
    prop_solver, _, _ = _hull_solver(args, disk)
    res_p = run_chunked(
        prop_solver,
        "hull_prop",
        steps=args.steps,
        warmup=args.warmup,
        budget_s=args.budget,
        measure_every=args.measure_every,
    )
    _progress("hull_prop", res_p)
    if not res_p["complete"]:
        return
    r_sp = -res_p["thrust"]
    t_meas = (r_sp - r_lat) / thrust_lat if thrust_lat else float("nan")
    wake_t = disk_plane_wake(towed_solver, int(round(args.hull_nx * 0.9825)), nbins=16)
    wake_p = disk_plane_wake(prop_solver, int(round(args.hull_nx * 0.9825)), nbins=16)
    payload = {
        "towed": {
            "resistance_lattice": r_lat,
            "wake": wake_t,
            "steps": args.steps,
            "warmup": args.warmup,
            "cb_numerical": stats["Cb_numerical"],
            "geometry_ok": abs(stats["Cb_numerical"] - 0.8098) < 0.02,
            "hull_stats": stats,
            "solid_cells": stats["solid_cells"],
            "lattice": {
                "length": geom[0],
                "beam": geom[1],
                "draft": geom[2],
                "re_lattice": args.inflow * geom[0] / towed_solver.cfg.nu,
            },
            "grid": {"nx": args.hull_nx, "ny": args.hull_ny, "nz": args.hull_nz},
        },
        "propelled": {
            "resistance_lattice": r_sp,
            "wake": wake_p,
            "steps": args.steps,
            "warmup": args.warmup,
            "solid_cells": stats["solid_cells"],
        },
        "lattice": {
            "length": geom[0],
            "beam": geom[1],
            "draft": geom[2],
            "re_lattice": args.inflow * geom[0] / towed_solver.cfg.nu,
        },
        "wake_profile": wake_t["profile"],
        "wake_field": wake_t["field"],
        "wake_field_propelled": wake_p["field"],
        "thrust_deduction_resolved": {
            "resistance_towed": r_lat,
            "resistance_propelled": r_sp,
            "delta_resistance": r_sp - r_lat,
            "thrust": thrust_lat,
            "thrust_deduction": t_meas,
            "grade": "MODEL (resolved R_SP - R_tow with an actuator-disk propulsor)",
            "disk": disk,
            "propeller_cells": d_lat,
        },
        "geometry": {
            "cb_numerical": stats["Cb_numerical"],
            "geometry_ok": abs(stats["Cb_numerical"] - 0.8098) < 0.02,
            "hull_stats": stats,
            "solid_cells": stats["solid_cells"],
        },
    }
    (ART / "hull.json").write_text(json.dumps(payload, indent=1))
    print(f"[hull] wrote hull.json  R_tow={r_lat:.4f} R_SP={r_sp:.4f} t={t_meas:+.4f}")


def stage_behindhull(args) -> None:
    hull = json.loads((ART / "hull.json").read_text())
    profile_rows = hull["wake_profile"]
    lattice_re = hull["lattice"]["re_lattice"]
    model = KVLCC2_MODEL
    factor = (model.reynolds / lattice_re) ** -0.2
    r_nodes = [p["r"] for p in profile_rows]
    w_nodes = [1.0 - p["u_axial"] / args.inflow for p in profile_rows]
    import numpy as np

    r_nodes_a = np.asarray(r_nodes)
    w_map = (
        np.interp(
            np.arange(args.tank) * 0.0 + np.linspace(0.0, r_nodes_a.max(), args.tank),
            r_nodes_a,
            np.asarray(w_nodes),
        )
        * factor
    )
    yy, zz = np.meshgrid(
        np.arange(args.tank) - args.tank / 2.0,
        np.arange(args.tank) - args.tank / 2.0,
        indexing="ij",
    )
    r_disk = np.sqrt(yy**2 + zz**2)
    w_two_d = np.interp(r_disk, np.linspace(0.0, r_nodes_a.max(), args.tank), w_map)
    d = args.d
    disk_mask = r_disk <= d / 2.0
    w_mean = float(w_two_d[disk_mask].mean()) if disk_mask.any() else 0.0
    profile = args.inflow * (1.0 - w_two_d)
    geometry = b4_70_geometry(d)
    n_rps = args.inflow * (1.0 - w_mean) / (args.j * d)
    omega = args.omega_sign * 2.0 * math.pi * n_rps
    cfg = RotatingZoneConfig(
        nx=args.nx,
        ny=args.tank,
        nz=args.tank,
        inflow=args.inflow,
        omega=omega,
        tau=args.tau,
        smagorinsky=args.smagorinsky,
        zone_radius=d * 0.62,
        n_theta=args.ntheta,
        sides="free_slip",
        streamwise="inlet_outlet",
        slip_planes=("y_min", "y_max", "z_min", "z_max"),
    )
    solver = SlidingMeshPropeller(
        cfg, geometry, center=(args.nx * 0.5, args.tank * 0.5, args.tank * 0.5)
    )
    solver.set_inflow_profile(torch.from_numpy(profile).float())
    out = run_chunked(
        solver,
        "behind",
        steps=args.steps,
        warmup=args.warmup,
        budget_s=args.budget,
        measure_every=args.measure_every,
    )
    _progress("behind", out)
    if not out["complete"]:
        return
    kt = out["thrust"] / (n_rps**2 * d**4)
    kq = out["torque"] / (n_rps**2 * d**5)
    payload = {
        "point": {
            "j_nominal": args.j,
            "n_rps": n_rps,
            "wake_mean_disk": w_mean,
            "thrust_lattice": out["thrust"],
            "torque_lattice": out["torque"],
            "kt": kt,
            "kq": kq,
            "eta0": args.j * kt / (2.0 * math.pi * kq) if kq > 0 else float("nan"),
            "re_scaling": {
                "re_lattice": lattice_re,
                "re_model": model.reynolds,
                "factor": float(factor),
                "wake_lattice_max": float(np.max(w_nodes)),
                "wake_lifted_max": float(np.max(w_nodes) * factor),
            },
            "steps": args.steps,
            "warmup": args.warmup,
            "samples": out["samples"],
            "cfg": {
                "nx": cfg.nx,
                "ny": cfg.ny,
                "nz": cfg.nz,
                "tau": cfg.tau,
                "nu": cfg.nu,
                "smagorinsky": cfg.smagorinsky,
                "inflow": cfg.inflow,
                "tip_mach": abs(omega) * d / 2.0,
            },
        }
    }
    (ART / "behindhull.json").write_text(json.dumps(payload, indent=1))
    print(
        f"[behind] wrote behindhull.json  K_T^B={kt:+.4f} 10K_Q^B={10 * kq:+.4f} w_disk={w_mean:.4f}"
    )


def stage_modelchain(args) -> None:
    """MODEL-grade fallback chain when the lattice campaign cannot get compute.

    Open-water ``K_T(J)/K_Q(J)`` from the in-repo blade-element momentum theory
    (BEMT) for the same B4-70 geometry, ITTC-78 model resistance, and the same
    ``T(1-t) = R_T`` root find that ``assemble`` uses.  Nothing here is a lattice
    result; the report grades it MODEL and keeps the resolved decomposition
    withheld.
    """
    import sys as _sys

    _sys.path.insert(0, str(ROOT / "scripts"))
    from sliding_mesh_kvlcc2_selfpropulsion import bem_open_water  # noqa: E402

    from tensorlbm.kvlcc2_selfpropulsion import (  # noqa: E402
        KVLCC2_REFERENCE_BANDS,
        empirical_cross_check,
        model_scale_resistance,
        solve_self_propulsion,
    )

    model = KVLCC2_MODEL
    resistance = model_scale_resistance(model)
    r_t = resistance["R_T"]
    bem = [b for b in bem_open_water() if b["j"] <= 1.0]
    js = [b["j"] for b in bem]
    kts = [b["kt"] for b in bem]
    kqs = [b["kq"] for b in bem]
    w = KVLCC2_REFERENCE_BANDS["wake_fraction"]["value"]
    t = KVLCC2_REFERENCE_BANDS["thrust_deduction"]["value"]
    sol = solve_self_propulsion(
        resistance=r_t,
        speed=model.speed,
        diameter=model.propeller_diameter,
        rho=model.rho,
        wake_fraction=w,
        thrust_deduction_value=t,
        j_table=js,
        kt_table=kts,
        kq_table=kqs,
    )
    payload = {
        "grade": "MODEL (BEMT open water + ITTC-78 resistance + reference w/t)",
        "case": {
            "speed_m_s": model.speed,
            "fr": model.foude,
            "re": model.reynolds,
            "diameter_m": model.propeller_diameter,
        },
        "resistance": resistance,
        "open_water_bemt": bem,
        "wake_fraction_used": w,
        "thrust_deduction_used": t,
        "self_propulsion": sol,
        "empirical": empirical_cross_check(model.cb, w),
    }
    (ART / "model_chain.json").write_text(json.dumps(payload, indent=1))
    print(
        "[modelchain] n_sp=%.3f rps  J=%.3f  eta0=%.3f  etaD=%.3f  residual=%.2e"
        % (sol["n_rps"], sol["j"], sol["eta0"], sol["power_chain"]["eta_D"], sol["residual_rel"])
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["openwater", "hull", "behindhull", "modelchain"])
    ap.add_argument("--tag", default="j07")
    ap.add_argument("--j", type=float, default=0.7)
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--tank", type=int, default=48)
    ap.add_argument("--d", type=float, default=20.0)
    ap.add_argument("--tau", type=float, default=0.51)
    ap.add_argument("--smagorinsky", type=float, default=0.16)
    ap.add_argument("--inflow", type=float, default=0.05)
    ap.add_argument("--ntheta", type=int, default=48)
    ap.add_argument("--omega-sign", type=float, default=-1.0)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--measure-every", type=int, default=4)
    ap.add_argument("--budget", type=float, default=95.0)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--hull-nx", type=int, default=140)
    ap.add_argument("--hull-ny", type=int, default=44)
    ap.add_argument("--hull-nz", type=int, default=30)
    ap.add_argument("--t-ref", type=float, default=0.20)
    ap.add_argument("--mode", default="towed")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.stage == "openwater":
        stage_openwater(args)
    elif args.stage == "hull":
        stage_hull(args)
    elif args.stage == "modelchain":
        stage_modelchain(args)
    else:
        stage_behindhull(args)


if __name__ == "__main__":
    main()
