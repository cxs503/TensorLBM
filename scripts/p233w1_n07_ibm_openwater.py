#!/usr/bin/env python
"""Campaign ``p233w1-N07`` driver: IBM open-water propeller + cylinder drag gate.

Tasks
-----
``audit``      AC3 momentum-balance audit in a closed periodic box.
``cylinder``   AC1 2-D cylinder drag sweep ``Re = 20..200`` (IBM vs literature).
``openwater``  AC2 one open-water mesh (D) swept over ``J`` -> ``K_T/K_Q/eta_0``.
``sweep``      AC4 grid-convergence sweep over three rotor diameters.
``figure``     AC6 open-water curve + cylinder drag figures (matplotlib).
``report``     assemble ``report.json`` (criteria numbers + artefact paths).

Every task writes a JSON artefact so that no number in the report is typed by
hand: the report generator reads the artefacts back.

Example
-------
    python scripts/p233w1_n07_ibm_openwater.py --task openwater --diameter 24 \
        --out artifacts/p233w1_n07_ibm_openwater/openwater_d24.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.ibm_openwater import (  # noqa: E402
    CYLINDER_CD_2D,
    CylinderConfig,
    OpenWaterConfig,
    PropellerGeometry,
    cylinder_drag_2d,
    eta_error_is_monotone,
    momentum_balance_audit,
    run_open_water_sweep,
)

OUT_ROOT = Path("artifacts/p233w1_n07_ibm_openwater")


def _dump(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=str)
    print(f"[write] {path}")
    return path


def _threads(n: int) -> None:
    if n > 0:
        torch.set_num_threads(n)


# --------------------------------------------------------------------------- #
# tasks
# --------------------------------------------------------------------------- #


def task_audit(args) -> dict:
    """AC3: IBM force integral == fluid momentum change (closed periodic box)."""
    payload = momentum_balance_audit(
        shape=(args.audit_n,) * 2 + (2 * args.audit_n,),
        tau=args.tau_audit,
        diameter=args.audit_n / 2.0,
        n_steps=args.audit_steps,
    )
    payload["criterion"] = "rel_residual = max|dP - F| / peak|F| <= 1e-2"
    payload["passed"] = bool(payload["rel_residual"] <= 1e-2)
    return payload


def task_cylinder(args) -> dict:
    """AC1: 2-D cylinder drag from IBM."""
    cfg = CylinderConfig(
        diameter=args.diameter_cyl,
        reynolds=tuple(args.reynolds),
        inflow=args.inflow,
        n_steps=args.steps_cyl,
    )
    cases = []
    for re in cfg.reynolds:
        t0 = time.time()
        res = cylinder_drag_2d(re, cfg, verbose=False)
        res["wall_seconds"] = time.time() - t0
        cases.append(res)
        print(
            f"  Re={re:6.0f}  Cd={res['Cd']:+.4f}  ref={res['Cd_reference']:.4f}  "
            f"err={100 * res['Cd_rel_error']:+.2f}%  fluct={res['Cd_fluctuation']:.4f}",
            flush=True,
        )
    errs = [abs(c["Cd_rel_error"]) for c in cases if math.isfinite(c["Cd"])]
    payload = {
        "cases": cases,
        "max_abs_rel_error": max(errs) if errs else float("inf"),
        "mean_abs_rel_error": sum(errs) / len(errs) if errs else float("inf"),
        "diverged": any(c["diverged"] for c in cases),
        "criterion": "max |Cd - Cd_ref| / Cd_ref <= 0.10 for Re in 20..200",
        "configuration": cfg.__dict__ | {"reynolds": list(cfg.reynolds)},
    }
    payload["passed"] = bool(payload["max_abs_rel_error"] <= 0.10 and not payload["diverged"])
    return payload


def _openwater_cfg(args, diameter: float) -> OpenWaterConfig:
    return OpenWaterConfig(
        advance_ratios=tuple(args.j),
        tip_speed=args.tip_speed,
        tau=args.tau,
        steps_per_rev=args.steps_per_rev,
        warmup_revs=args.warmup_revs,
        measure_revs=args.measure_revs,
        lateral_factor=args.lateral,
        kernel=args.kernel,
        collision=args.collision,
        max_steps=args.max_steps,
    )


def task_openwater(args) -> dict:
    """AC2: open-water curve for one rotor resolution."""
    geom = PropellerGeometry(diameter=args.diameter, radial_stations=args.radial_stations)
    cfg = _openwater_cfg(args, args.diameter)
    t0 = time.time()
    payload = run_open_water_sweep(geom, cfg, verbose=True)
    payload["wall_seconds"] = time.time() - t0
    payload["configuration"] = {
        "diameter": args.diameter,
        "advance_ratios": list(args.j),
        "tip_speed": args.tip_speed,
        "tau": args.tau,
        "nu": cfg.nu,
        "steps_per_rev": args.steps_per_rev,
        "warmup_revs": args.warmup_revs,
        "measure_revs": args.measure_revs,
        "lateral_factor": args.lateral,
        "kernel": args.kernel,
        "collision": args.collision,
        "radial_stations": args.radial_stations,
        "n_blades": geom.n_blades,
        "pitch_ratio_07": geom.pitch_ratio_07,
        "blade_area_ratio": geom.blade_area_ratio,
    }
    payload["criterion"] = "max |eta_0 - eta_ref| / eta_ref <= 0.15 for J in 0.2..1.0"
    payload["passed"] = bool(
        payload["max_abs_eta_error"] <= 0.15 and not payload["diverged"]
    )
    return payload


def task_sweep(args) -> dict:
    """AC4: grid convergence of the open-water curve over three meshes."""
    meshes = []
    for d in args.diameters:
        geom = PropellerGeometry(diameter=float(d), radial_stations=args.radial_stations)
        cfg = _openwater_cfg(args, float(d))
        t0 = time.time()
        res = run_open_water_sweep(geom, cfg, verbose=True)
        res["wall_seconds"] = time.time() - t0
        res["configuration"] = {
            "diameter": float(d),
            "advance_ratios": list(args.j),
            "tip_speed": args.tip_speed,
            "tau": args.tau,
            "steps_per_rev": args.steps_per_rev,
            "lateral_factor": args.lateral,
            "kernel": args.kernel,
            "collision": args.collision,
            "radial_stations": args.radial_stations,
        }
        meshes.append(res)
        print(
            f"  D={d}: max|deta|={100 * res['max_abs_eta_error']:.2f}%  "
            f"rms={100 * res['rms_abs_eta_error']:.2f}%",
            flush=True,
        )
    errors = [m["max_abs_eta_error"] for m in meshes]
    payload = {
        "meshes": meshes,
        "eta_errors_coarse_to_fine": errors,
        "monotone": eta_error_is_monotone(errors),
        "criterion": "max |eta error| decreases monotonically over 3 refined meshes",
        "diverged": any(m["diverged"] for m in meshes),
    }
    payload["passed"] = bool(payload["monotone"] and not payload["diverged"])
    return payload


def task_figure(args) -> dict:
    """AC6: figures from the JSON artefacts (no hand-typed numbers)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(args.root)
    cand = sorted(root.glob("openwater_d*.json"))
    if not cand:
        raise SystemExit("no open-water sweep artefact found - run --kind aggregate first")
    cand.sort(key=lambda p: len(json.loads(p.read_text())["points"]), reverse=True)
    ow = json.loads(cand[0].read_text())
    cyl = json.loads((root / "cylinder.json").read_text())
    fig_dir = Path("figures/p233w1_n07")
    fig_dir.mkdir(parents=True, exist_ok=True)

    produced = []

    # --- figure 1: open-water curve, IBM vs KP505 --------------------------
    pts = ow["points"]
    js = [p["J"] for p in pts]
    kt = [p["KT"] for p in pts]
    kq10 = [p["10KQ"] for p in pts]
    eta = [p["eta_0"] for p in pts]
    refs = [p["reference"] for p in pts]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    axes[0].plot(js, [r["KT"] for r in refs], "k--o", label="KP505 (exp.)")
    axes[0].plot(js, kt, "r-s", label="IBM (D=%g)" % ow["diameter"])
    axes[0].set_xlabel("J"); axes[0].set_ylabel(r"$K_T$"); axes[0].grid(alpha=0.3)
    axes[0].legend(); axes[0].set_title("thrust")
    axes[1].plot(js, [r["10KQ"] for r in refs], "k--o", label="KP505 (exp.)")
    axes[1].plot(js, kq10, "r-s", label="IBM")
    axes[1].set_xlabel("J"); axes[1].set_ylabel(r"$10K_Q$"); axes[1].grid(alpha=0.3)
    axes[1].legend(); axes[1].set_title("torque")
    axes[2].plot(js, [r["eta_0"] for r in refs], "k--o", label="KP505 (exp.)")
    axes[2].plot(js, eta, "r-s", label="IBM")
    axes[2].set_xlabel("J"); axes[2].set_ylabel(r"$\eta_0$"); axes[2].grid(alpha=0.3)
    axes[2].legend(); axes[2].set_title("open-water efficiency")
    fig.suptitle("IBM open-water propeller curve vs KP505 reference")
    fig.tight_layout()
    p1 = fig_dir / "openwater_curve.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    produced.append(str(p1))

    # --- figure 1b: time history of KT/KQ (transient diagnostic) ------------
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for pt in pts:
        h = [x for x in pt["history"] if isinstance(x, dict)]
        if not h:
            continue
        t = [x["step"] for x in h]
        axs[0].plot(t, [x["KT"] for x in h], label=f"J={pt['J']:g}")
        axs[1].plot(t, [10 * x["KQ"] for x in h], label=f"J={pt['J']:g}")
    axs[0].set_xlabel("lattice step"); axs[0].set_ylabel(r"$K_T$")
    axs[1].set_xlabel("lattice step"); axs[1].set_ylabel(r"$10K_Q$")
    for a in axs:
        a.grid(alpha=0.3); a.legend(fontsize=7)
    axs[0].set_title("running thrust coefficient")
    axs[1].set_title("running torque coefficient")
    fig.suptitle("IBM open-water: running averages (statistical-convergence diagnostic)")
    fig.tight_layout()
    p1b = fig_dir / "openwater_history.png"
    fig.savefig(p1b, dpi=150)
    plt.close(fig)
    produced.append(str(p1b))

    # --- figure 2: cylinder drag sweep -------------------------------------
    cases = cyl["cases"]
    res_ = [c["Re"] for c in cases]
    cd = [c["Cd"] for c in cases]
    cd_ref = [c["Cd_reference"] for c in cases]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(res_, cd_ref, "k--o", label="literature")
    ax.plot(res_, cd, "r-s", label="IBM (D2Q9)")
    ax.set_xlabel("Re"); ax.set_ylabel(r"$C_D$"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("2-D cylinder drag: IBM vs literature")
    fig.tight_layout()
    p2 = fig_dir / "cylinder_drag.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    produced.append(str(p2))

    # --- figure 3: grid convergence of the open-water error ----------------
    sweep_path = root / "grid_convergence.json"
    if sweep_path.exists():
        sw = json.loads(sweep_path.read_text())
        ds = [m["diameter"] for m in sw["meshes"]]
        errs = [100 * e for e in sw["eta_errors_coarse_to_fine"]]
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        ax.plot(ds, errs, "b-o")
        ax.set_xlabel("rotor diameter D [lu]"); ax.set_ylabel(r"max $|\Delta\eta_0|$ [%]")
        ax.grid(alpha=0.3); ax.set_title("grid convergence of the open-water curve")
        fig.tight_layout()
        p3 = fig_dir / "grid_convergence.png"
        fig.savefig(p3, dpi=150)
        plt.close(fig)
        produced.append(str(p3))

    # --- figure 4: instantaneous field / disk loading ----------------------
    snap = root / "field_slice.json"
    if snap.exists():
        data = json.loads(snap.read_text())
        import numpy as np

        ux = np.array(data["ux"])
        fig, ax = plt.subplots(figsize=(8.0, 4.0))
        im = ax.imshow(ux, origin="lower", aspect="auto", cmap="RdBu_r")
        ax.set_xlabel("x [lu] (streamwise)"); ax.set_ylabel("y [lu]")
        ax.set_title(f"IBM rotor wake: axial velocity u_x, J={data['J']}")
        fig.colorbar(im, ax=ax, label=r"$u_x$ [lu]")
        fig.tight_layout()
        p4 = fig_dir / "wake_slice.png"
        fig.savefig(p4, dpi=150)
        plt.close(fig)
        produced.append(str(p4))

    return {"figures": produced}


def task_report(args) -> dict:
    """Assemble ``report.json`` (criteria -> numbers + artefact paths)."""
    root = Path(args.root)
    out = {"mission": "p233w1-N07", "title": "IBM ice-propeller Phase 0: immersed boundary + open-water curve"}
    files = {
        "momentum_audit": "momentum_audit.json",
        "cylinder": "cylinder.json",
        "openwater": None,
        "grid": "grid_convergence.json",
    }
    if files["openwater"] is None:
        cand = sorted(root.glob("openwater_d*.json"))
        cand.sort(key=lambda q: len(json.loads(q.read_text())["points"]), reverse=True)
        files["openwater"] = cand[0].name if cand else "openwater_missing.json"
    data = {}
    for key, name in files.items():
        p = root / name
        if p.exists():
            data[key] = json.loads(p.read_text())
            out[f"{key}_path"] = str(p)
        else:
            data[key] = None
    criteria = {}

    if data["momentum_audit"]:
        criteria["AC3"] = {
            "metric": "momentum_balance_max_rel_residual",
            "value": data["momentum_audit"]["rel_residual"],
            "threshold": 1e-2,
            "passed": data["momentum_audit"]["passed"],
            "artifact": f"{root}/momentum_audit.json",
        }
    if data["cylinder"]:
        criteria["AC1"] = {
            "metric": "cylinder_Cd_max_abs_rel_error",
            "value": data["cylinder"]["max_abs_rel_error"],
            "threshold": 0.10,
            "passed": data["cylinder"]["passed"],
            "artifact": f"{root}/cylinder.json",
        }
    if data["openwater"]:
        criteria["AC2"] = {
            "metric": "openwater_eta_max_abs_rel_error",
            "value": data["openwater"]["max_abs_eta_error"],
            "threshold": 0.15,
            "passed": data["openwater"]["passed"],
            "artifact": f"{root}/{files['openwater']}",
        }
    if data["grid"]:
        criteria["AC4"] = {
            "metric": "eta_error_monotone_over_3_meshes",
            "value": data["grid"]["eta_errors_coarse_to_fine"],
            "threshold": "non-increasing",
            "passed": data["grid"]["passed"],
            "artifact": f"{root}/grid_convergence.json",
        }
    out["criteria"] = criteria
    out["all_passed"] = all(bool(c["passed"]) for c in criteria.values()) and len(criteria) == 4
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--task",
        required=True,
        choices=["audit", "cylinder", "openwater", "sweep", "figure", "report"],
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--root", type=Path, default=OUT_ROOT)
    p.add_argument("--threads", type=int, default=int(os.environ.get("TL_THREADS", "8")))
    # shared numerics
    p.add_argument("--tau", type=float, default=0.5005)
    p.add_argument("--tip-speed", type=float, default=0.12)
    p.add_argument("--steps-per-rev", type=int, default=360)
    p.add_argument("--warmup-revs", type=float, default=0.5)
    p.add_argument("--measure-revs", type=float, default=1.5)
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--lateral", type=float, default=1.5)
    p.add_argument("--kernel", default="4pt")
    p.add_argument("--collision", default="mrt", choices=["mrt", "bgk"])
    p.add_argument("--radial-stations", type=int, default=12)
    # single-mesh open-water
    p.add_argument("--diameter", type=float, default=24.0)
    p.add_argument("--j", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8, 1.0])
    # grid sweep
    p.add_argument("--diameters", type=float, nargs="+", default=[16.0, 24.0, 32.0])
    # cylinder
    p.add_argument("--diameter-cyl", type=float, default=10.0)
    p.add_argument("--reynolds", type=int, nargs="+", default=[20, 40, 60, 80, 100, 150, 200])
    p.add_argument("--inflow", type=float, default=0.05)
    p.add_argument("--steps-cyl", type=int, default=24000)
    # audit
    p.add_argument("--audit-n", type=int, default=24)
    p.add_argument("--audit-steps", type=int, default=20)
    p.add_argument("--tau-audit", type=float, default=0.8)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _threads(args.threads)
    task = args.task
    if task == "audit":
        payload = task_audit(args)
        out = args.out or OUT_ROOT / "momentum_audit.json"
    elif task == "cylinder":
        payload = task_cylinder(args)
        out = args.out or OUT_ROOT / "cylinder.json"
    elif task == "openwater":
        payload = task_openwater(args)
        out = args.out or OUT_ROOT / f"openwater_d{int(args.diameter)}.json"
    elif task == "sweep":
        payload = task_sweep(args)
        out = args.out or OUT_ROOT / "grid_convergence.json"
    elif task == "figure":
        payload = task_figure(args)
        out = args.out or OUT_ROOT / "figures.json"
    elif task == "report":
        payload = task_report(args)
        out = args.out or OUT_ROOT / "report.json"
    else:  # pragma: no cover
        raise SystemExit(f"unknown task {task}")
    _dump(payload, Path(out))
    if task in {"audit", "cylinder", "openwater", "sweep", "report"}:
        print(json.dumps({k: v for k, v in payload.items() if k in ("passed", "all_passed")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())