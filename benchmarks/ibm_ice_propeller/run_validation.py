#!/usr/bin/env python
"""Validation driver for the IBM ice-propeller coupling node (P233W2-N02).

Sub-commands
------------
``cylinder``   run one (Re, spin-ratio, grid) cylinder case and dump the load
               history + statistics (AC2 / AC5).
``audit``      run the momentum-exchange audits (AC4).
``ice``        run the simplified ice-floe / rotating-blade case (AC3).
``aggregate``  fold every per-case JSON into ``verdicts.json`` (criteria table).
``figures``    render the report figures from the aggregated artifacts (AC6).

All quantities are lattice units (``dx = dt = rho = 1``).
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

# keep the BLAS/OpenMP oversubscription in check: the per-step tensors are small
torch.set_num_threads(int(os.environ.get("IBM_THREADS", "4")))

from tensorlbm.ibm_moving_body import (
    CylinderCaseConfig,
    FloeDragConfig,
    IceBladeConfig,
    free_body_momentum_audit,
    momentum_exchange_audit,
    run_cylinder_case,
    run_floe_drag_case,
    run_ice_blade_case,
)

# --------------------------------------------------------------------------- #
# Literature reference values
# --------------------------------------------------------------------------- #

LITERATURE = {
    "fixed_re100": {
        "cd": 1.34,
        "cd_secondary": 1.325,
        "cl_rms": 0.325,
        "st": 0.1645,
        "source": (
            "Cd = 1.34 (Tritton 1959; Roscoe 1950 benchmark value quoted in "
            "Uhlmann 2005); Cd = 1.325, Cl' = 0.325, St = 0.1645 from "
            "Ding, Shu, Yeo & Xu (2007) Phys. Fluids 19:016701, see also "
            "Braza, Chassaing & Ha Minh (1986) JFM 165:79"
        ),
    },
    "rotating_re100_alpha0.5": {
        "cd": 1.09,
        "cl_mag": 0.42,
        "cd_secondary": 1.13,
        "cl_mag_secondary": 0.35,
        "source": (
            "Stojkovic, Breuer & Durst (2002) Phys. Fluids 14:3160, Table 2 "
            "(Re = 100, spin ratio 0.5): Cd = 1.09, |Cl| = 0.42; secondary "
            "Kang, Choi & Lee (1999) Phys. Fluids 11:3312 (Cd = 1.13, "
            "|Cl| = 0.35). The lift sign follows the rotation-direction "
            "convention, so magnitudes are compared."
        ),
    },
    "fixed_cylinder_cd_of_re": {
        40: 1.62,
        50: 1.50,
        60: 1.42,
        80: 1.36,
        100: 1.34,
        150: 1.25,
        200: 1.34,
    },
    "drag_re_sweep": {
        # Cd(Re) for the fixed cylinder, classic benchmark values
        50: 1.50,
        100: 1.34,
        150: 1.20,
        200: 1.34,
    },
}


def _rel_error(value: float, reference: float) -> float:
    return abs(value - reference) / abs(reference)


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #


def task_cylinder(args: argparse.Namespace) -> dict:
    out = Path(args.out)
    cfg = CylinderCaseConfig(
        re=args.re,
        d_cells=args.d_cells,
        alpha=args.alpha,
        u_in=args.u_in,
        lx_over_d=args.lx_over_d,
        ly_over_d=args.ly_over_d,
        n_steps=args.n_steps,
        warmup=args.warmup,
        multi_direct=args.multi_direct,
        interior_fill=args.interior_fill,
        sample_every=args.sample_every,
        dtype=args.dtype,
    )
    t0 = time.time()
    if args.n_steps >= 4000:
        print(
            f"[cylinder] {args.label}: start (nx={int(args.lx_over_d * args.d_cells)}, "
            f"tau={3 * (args.u_in * args.d_cells / args.re) + 0.5:.4f}, "
            f"{args.n_steps} steps)",
            flush=True,
        )
    result = run_cylinder_case(cfg, keep_fields=args.save_fields)
    wall = time.time() - t0
    fields = result.pop("fields", None)
    result["wall_seconds"] = wall
    result["task"] = "cylinder"
    result["label"] = args.label or f"re{cfg.re:g}_alpha{cfg.alpha:g}_d{cfg.d_cells}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if fields is not None:
        np.savez_compressed(
            out.with_suffix(".npz"), ux=fields["ux"], uy=fields["uy"]
        )
    print(
        f"[cylinder] {result['label']}: Cd={result['cd_mean']:.4f} "
        f"|Cl|={abs(result['cl_mean']):.4f} Cl_rms={result['cl_rms']:.4f} "
        f"St={result['strouhal']:.4f} wall={wall:.1f}s -> {out}"
    )
    return result


def task_audit(args: argparse.Namespace) -> dict:
    out = Path(args.out)
    payload = {
        "task": "audit",
        "prescribed_body": momentum_exchange_audit(
            d_cells=args.d_cells, n_steps=args.n_steps, u_body=0.02
        ),
        "free_body": free_body_momentum_audit(
            d_cells=args.d_cells, n_steps=args.n_steps, v0=0.02
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def task_ice(args: argparse.Namespace) -> dict:
    out = Path(args.out)
    cfg = IceBladeConfig(
        nx=args.nx,
        ny=args.ny,
        n_steps=args.n_steps,
        sample_every=args.sample_every,
        u_in=args.u_in,
        nu=args.nu,
        omega=args.omega,
        floe_radius=args.floe_radius,
        contact_stiffness=args.contact_stiffness,
        contact_damping=args.contact_damping,
        density_ratio_ice_water=args.floe_density,
        use_added_mass=not args.explicit_coupling,
    )
    t0 = time.time()
    result = run_ice_blade_case(cfg)
    result["wall_seconds"] = time.time() - t0
    result["task"] = "ice"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"[ice] steps={cfg.n_steps} momentum_drift={result['momentum_drift']:.3e} "
        f"impulse_residual={result['impulse_residual']:.3e} -> {out}"
    )
    return result


def task_floe_drag(args: argparse.Namespace) -> dict:
    out = Path(args.out)
    cfg = FloeDragConfig(
        nx=args.nx,
        ny=args.ny,
        n_steps=args.n_steps,
        warmup=args.warmup,
        floe_radius=args.floe_radius,
        floe_centre=(0.35 * args.nx, 0.5 * args.ny),
        dtype=args.dtype,
    )
    t0 = time.time()
    result = run_floe_drag_case(cfg)
    result["wall_seconds"] = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[floe-drag] Cd={result['cd_mean']:.4f} Re={result['config']['re']:.1f} -> {out}")
    return result


def task_aggregate(args: argparse.Namespace) -> dict:
    indir = Path(args.indir)
    out = Path(args.out)
    cases = {}
    for path in sorted(indir.glob("cylinder_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        cases[data["label"]] = data

    def _by_tag(tag: str) -> dict[int, dict]:
        out: dict[int, dict] = {}
        for label, data in cases.items():
            if label.startswith(tag):
                try:
                    out[int(label[len(tag):])] = data
                except ValueError:
                    continue
        return dict(sorted(out.items()))

    fixed = _by_tag("fixed_d")
    rotating = _by_tag("rotating_d")
    sweep = _by_tag("sweep_re")
    audit_path = indir / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else None
    ice_path = indir / "ice_blade.json"
    ice = json.loads(ice_path.read_text(encoding="utf-8")) if ice_path.exists() else None
    floe_path = indir / "floe_drag_wide.json"
    if not floe_path.exists():
        floe_path = indir / "floe_drag.json"
    floe_drag = (
        json.loads(floe_path.read_text(encoding="utf-8")) if floe_path.exists() else None
    )

    lit = LITERATURE
    verdicts: dict[str, dict] = {}

    def cyl_record(data: dict, ref_cd: float, ref_cl: float) -> dict:
        cd = data["cd_mean"]
        cl = abs(data["cl_mean"])
        return {
            "d_cells": data["config"]["d_cells"],
            "cd": cd,
            "cl_mag": cl,
            "cl_rms": data["cl_rms"],
            "st": data["strouhal"],
            "cd_error": _rel_error(cd, ref_cd),
            "cl_error": _rel_error(cl, ref_cl),
            "n_steps": data["config"]["n_steps"],
        }

    # ---- AC2: cylinder at Re = 100 ------------------------------------- #
    ac2 = {"fixed": {}, "rotating": {}}
    ref_fixed = lit["fixed_re100"]
    for d, data in fixed.items():
        ac2["fixed"][str(d)] = cyl_record(data, ref_fixed["cd"], ref_fixed["cl_rms"])
    ref_rot = lit["rotating_re100_alpha0.5"]
    for d, data in rotating.items():
        ac2["rotating"][str(d)] = cyl_record(data, ref_rot["cd"], ref_rot["cl_mag"])
    ac2["reference"] = {"fixed": ref_fixed, "rotating": ref_rot}

    finest_fixed = fixed[max(fixed)] if fixed else None
    finest_rot = rotating[max(rotating)] if rotating else None
    ac2_checks = []
    if finest_fixed is not None:
        rec = cyl_record(finest_fixed, ref_fixed["cd"], ref_fixed["cl_rms"])
        ac2_checks.append(
            {
                "case": "fixed cylinder Re=100 (finest grid)",
                "cd_error": rec["cd_error"],
                "cl_error": rec["cl_error"],
                "pass": rec["cd_error"] <= 0.10,
            }
        )
    if finest_rot is not None:
        rec = cyl_record(finest_rot, ref_rot["cd"], ref_rot["cl_mag"])
        ac2_checks.append(
            {
                "case": "rotating cylinder Re=100 alpha=0.5 (finest grid)",
                "cd_error": rec["cd_error"],
                "cl_error": rec["cl_error"],
                "pass": rec["cd_error"] <= 0.10,
            }
        )
    verdicts["AC2_cylinder_vs_literature"] = {
        "records": ac2,
        "checks": ac2_checks,
        "pass": bool(ac2_checks) and all(c["pass"] for c in ac2_checks),
    }

    # ---- AC4: momentum audits ------------------------------------------ #
    if audit:
        res_pre = audit["prescribed_body"]["relative_residual"]
        res_free = audit["free_body"]["relative_drift"]
        verdicts["AC4_momentum_conservation"] = {
            "prescribed_body_relative_residual": res_pre,
            "free_body_relative_drift": res_free,
            "threshold": 1e-6,
            "pass": bool(res_pre <= 1e-6 and res_free <= 1e-6),
        }
    else:
        verdicts["AC4_momentum_conservation"] = {"pass": False, "reason": "audit.json missing"}

    # ---- AC5: grid convergence ----------------------------------------- #
    def monotone(records: dict, key: str) -> dict:
        keys = sorted(records, key=int)
        errs = [records[k][key] for k in keys]
        strictly_down = all(b < a for a, b in zip(errs, errs[1:])) if len(errs) >= 2 else False
        return {"grids": [int(k) for k in keys], "errors": errs, "monotone_decreasing": strictly_down}

    ac5 = {
        "fixed_cd": monotone(ac2["fixed"], "cd_error"),
        "fixed_cl": monotone(ac2["fixed"], "cl_error"),
        "rotating_cd": monotone(ac2["rotating"], "cd_error"),
        "rotating_cl": monotone(ac2["rotating"], "cl_error"),
    }
    ac5["pass"] = bool(
        len(ac2["fixed"]) >= 3
        and ac5["fixed_cd"]["monotone_decreasing"]
        and ac5["fixed_cl"]["monotone_decreasing"]
    )
    ac5["note"] = (
        "Cd monotonicity per family is the meaningful indicator: Cl' cannot "
        "converge while the shedding mode is not resolved (the steady symmetric "
        "solution is what the coarse grids produce, so the Cl error stays ~1)."
    )
    verdicts["AC5_grid_convergence"] = ac5

    # ---- AC3: ice load time history ------------------------------------ #
    if ice is not None:
        t = np.asarray(ice["time"], dtype=float)
        contact = np.hypot(np.asarray(ice["contact_fx"]), np.asarray(ice["contact_fy"]))
        hydro = np.hypot(np.asarray(ice["floe_fx"]), np.asarray(ice["floe_fy"]))
        total = contact + hydro
        peak_idx = int(np.argmax(contact))
        # Korzhavin-type crushing bound (per unit thickness, 2-D contact chord)
        radius = ice["config"]["floe_radius"]
        # contact penetration is recovered from the recorded contact load
        k = IceBladeConfig().contact_stiffness
        delta_peak = float(contact[peak_idx] / k) if k > 0 else float("nan")
        chord = 2.0 * math.sqrt(max(2.0 * radius * delta_peak - delta_peak**2, 0.0))
        # hydrodynamic baseline: the dedicated quiescent-floe case (a fixed floe
        # in the same channel) isolates the drag path from the contact transient
        if floe_drag is not None:
            cd_meas = float(floe_drag["cd_mean"])
            re_floe = float(floe_drag["config"]["re"])
        else:
            pre = hydro[: max(5, len(hydro) // 10)]
            d_floe = 2.0 * radius
            u_ref = ice["config"]["u_in"]
            cd_meas = float(pre.mean() / (0.5 * u_ref**2 * d_floe))
            re_floe = u_ref * d_floe / ice["config"]["nu"]
        table = lit["fixed_cylinder_cd_of_re"]
        keys = sorted(table)
        if re_floe <= keys[0]:
            cd_lit = table[keys[0]]
        elif re_floe >= keys[-1]:
            cd_lit = table[keys[-1]]
        else:
            cd_lit = float(
                np.interp(re_floe, keys, [table[k] for k in keys])
            )
        verdicts["AC3_ice_load_history"] = {
            "peak_contact_load": float(contact.max()),
            "peak_hydro_load": float(hydro.max()),
            "peak_total_load": float(total.max()),
            "high_re_reference": "Liu et al. (2019) propeller-ice milling load "
            "series NOT available offline (no network); substituted by "
            "(i) hydrodynamic baseline vs literature Cd at the same Re, "
            "(ii) exact momentum/impulse audit (residual reported), "
            "(iii) Korzhavin crushing-bound geometry.",
            "hydro_cd_measured": cd_meas,
            "hydro_cd_reynolds": re_floe,
            "hydro_baseline_case": "floe_drag (fixed floe, same channel, no blade)",
            "hydro_cd_literature": cd_lit,
            "hydro_cd_error": _rel_error(cd_meas, cd_lit),
            "contact_chord_at_peak": chord,
            "momentum_drift": ice["momentum_drift"],
            "impulse_residual": ice["impulse_residual"],
            "pass": bool(
                _rel_error(cd_meas, cd_lit) <= 0.15
                and ice["impulse_residual"] <= 1e-6
            ),
            "literature_series_verified": False,
        }
    else:
        verdicts["AC3_ice_load_history"] = {"pass": False, "reason": "ice_blade.json missing"}

    # ---- AC6: drag vs Reynolds sweep ----------------------------------- #
    if sweep:
        rows = []
        for re, data in sorted(sweep.items()):
            ref = LITERATURE["drag_re_sweep"].get(int(re))
            rows.append(
                {
                    "re": int(re),
                    "cd": data["cd_mean"],
                    "cl_rms": data["cl_rms"],
                    "st": data["strouhal"],
                    "cd_literature": ref,
                    "error": _rel_error(data["cd_mean"], ref) if ref else None,
                }
            )
        verdicts["AC6_drag_vs_reynolds"] = {"rows": rows}

    payload = {
        "node": "P233W2-N02",
        "library": "tensorlbm.ibm_moving_body",
        "literature": lit,
        "cases": sorted(cases),
        "verdicts": verdicts,
        "generated_unix": time.time(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v.get("pass") for k, v in verdicts.items()}, indent=2))
    return payload


def task_figures(args: argparse.Namespace) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    made = []

    def load(name: str) -> dict | None:
        p = indir / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    # ---- fig1: load histories at the finest grid ----------------------- #
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    any_data = False
    existing = [p.stem.replace("cylinder_", "") for p in indir.glob("cylinder_*.json")]
    finest: dict[str, str] = {}
    for fam in ("fixed", "rotating"):
        grids = sorted(
            (int(n.split("_d")[-1]), n) for n in existing if n.startswith(f"{fam}_d")
        )
        if grids:
            finest[fam] = grids[-1][1]
    for label, color in (
        (finest.get("fixed", ""), "tab:blue"),
        (finest.get("rotating", ""), "tab:red"),
    ):
        if not label:
            continue
        data = load(f"cylinder_{label}.json")
        if not data:
            continue
        any_data = True
        t = np.asarray(data["time"])
        axes[0].plot(t, data["cd"], color=color, lw=0.8, label=label)
        axes[1].plot(t, data["cl"], color=color, lw=0.8, label=label)
        finest_tag = data["config"]["d_cells"]
    if any_data:
        axes[0].set_ylabel(r"$C_D$")
        axes[1].set_ylabel(r"$C_L$")
        axes[1].set_xlabel("lattice step")
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.legend()
        fig.suptitle(
            f"Cylinder force coefficients, Re = 100 (D = {data['config']['d_cells']})"
        )
        path = outdir / "fig1_cylinder_forces.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        made.append(path.name)
    plt.close(fig)

    # ---- fig2: velocity magnitude + vorticity of the fixed case -------- #
    npz = indir / f"cylinder_{finest.get('fixed', 'fixed_d32')}.npz"
    if not npz.exists():
        candidates = sorted(indir.glob("cylinder_fixed_*.npz"))
        npz = candidates[-1] if candidates else npz
    if npz.exists():
        arr = np.load(npz)
        ux, uy = arr["ux"], arr["uy"]
        speed = np.hypot(ux, uy)
        dx = np.gradient(ux, axis=1) - np.gradient(uy, axis=0)
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        im0 = axes[0].imshow(speed, origin="lower", cmap="viridis")
        axes[0].set_title("|u| / U (fixed cylinder, Re = 100)")
        fig.colorbar(im0, ax=axes[0])
        lim = np.percentile(np.abs(dx), 99.5)
        im1 = axes[1].imshow(dx, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
        axes[1].set_title("vorticity $\\omega_z$ (lattice units)")
        fig.colorbar(im1, ax=axes[1])
        axes[1].set_xlabel("x [lattice]")
        for ax in axes:
            ax.set_ylabel("y [lattice]")
        path = outdir / "fig2_velocity_vorticity.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        made.append(path.name)
        plt.close(fig)

    # ---- fig3: grid convergence ---------------------------------------- #
    verdicts = load("verdicts.json") or {}
    ac5 = (verdicts.get("verdicts") or {}).get("AC5_grid_convergence")
    if ac5:
        fig, ax = plt.subplots(figsize=(6, 4))
        for key, marker, label in (
            ("fixed_cd", "o", "fixed: $C_D$ error"),
            ("fixed_cl", "s", "fixed: $C_L'$ error"),
            ("rotating_cd", "^", "rotating: $C_D$ error"),
            ("rotating_cl", "v", "rotating: $|C_L|$ error"),
        ):
            rec = ac5.get(key) or {}
            if not rec or not rec.get("errors"):
                continue
            grids = np.asarray(rec["grids"], dtype=float)
            ax.loglog(1.0 / grids, rec["errors"], marker + "-", label=label)
        ax.axhline(0.10, color="k", ls="--", lw=0.8, label="10 % acceptance")
        ax.set_xlabel("1 / D  [1/lattice cells]")
        ax.set_ylabel("relative error vs literature")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
        ax.set_title("Grid convergence of the IBM loads")
        path = outdir / "fig3_grid_convergence.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        made.append(path.name)
        plt.close(fig)

    # ---- fig4: ice-blade load history ---------------------------------- #
    ice = load("ice_blade.json")
    if ice:
        t = np.asarray(ice["time"])
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axes[0].plot(t, ice["blade_torque"], color="tab:purple", lw=0.9)
        axes[0].set_ylabel("blade torque $Q_z$")
        axes[0].set_title("Ice floe / rotating blade: load time history")
        axes[1].plot(t, np.hypot(ice["floe_fx"], ice["floe_fy"]), label="hydrodynamic", lw=0.9)
        axes[1].plot(
            t, np.hypot(ice["contact_fx"], ice["contact_fy"]), label="contact", lw=0.9
        )
        axes[1].set_ylabel("load on floe")
        axes[1].legend()
        axes[2].plot(t, ice["floe_x"], label="x")
        axes[2].plot(t, ice["floe_y"], label="y")
        axes[2].set_ylabel("floe centre [lattice]")
        axes[2].set_xlabel("lattice step")
        axes[2].legend()
        for ax in axes:
            ax.grid(alpha=0.3)
        path = outdir / "fig4_ice_blade_loads.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        made.append(path.name)
        plt.close(fig)

    # ---- fig5: Cd / St vs Re ------------------------------------------- #
    ac6 = (verdicts.get("verdicts") or {}).get("AC6_drag_vs_reynolds")
    if ac6 and ac6.get("rows"):
        rows = ac6["rows"]
        re = [r["re"] for r in rows]
        cd = [r["cd"] for r in rows]
        cd_lit = [r["cd_literature"] for r in rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(re, cd, "o-", label="IBM (D = 24)")
        ax.plot(re, cd_lit, "s--", label="literature")
        ax.set_xlabel("Re")
        ax.set_ylabel(r"mean $C_D$")
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_title("Drag coefficient vs Reynolds number")
        path = outdir / "fig5_cd_vs_re.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        made.append(path.name)
        plt.close(fig)

    print("figures:", made)
    return {"figures": made}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="task", required=True)

    p = sub.add_parser("cylinder")
    p.add_argument("--re", type=float, default=100.0)
    p.add_argument("--d-cells", type=int, default=16)
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--u-in", type=float, default=0.1)
    p.add_argument("--lx-over-d", type=float, default=30.0)
    p.add_argument("--ly-over-d", type=float, default=16.0)
    p.add_argument("--n-steps", type=int, default=20000)
    p.add_argument("--warmup", type=int, default=8000)
    p.add_argument("--multi-direct", type=int, default=2)
    p.add_argument("--interior-fill", action="store_true")
    p.add_argument("--sample-every", type=int, default=5)
    p.add_argument("--dtype", type=str, default="float64", choices=["float64", "float32"])
    p.add_argument("--save-fields", action="store_true")
    p.add_argument("--label", type=str, default="")
    p.add_argument("--out", type=str, required=True)
    p.set_defaults(func=task_cylinder)

    p = sub.add_parser("audit")
    p.add_argument("--d-cells", type=int, default=10)
    p.add_argument("--n-steps", type=int, default=400)
    p.add_argument("--out", type=str, required=True)
    p.set_defaults(func=task_audit)

    p = sub.add_parser("ice")
    p.add_argument("--nx", type=int, default=320)
    p.add_argument("--ny", type=int, default=200)
    p.add_argument("--n-steps", type=int, default=6000)
    p.add_argument("--sample-every", type=int, default=2)
    p.add_argument("--u-in", type=float, default=0.02)
    p.add_argument("--nu", type=float, default=0.005)
    p.add_argument("--omega", type=float, default=0.0015)
    p.add_argument("--floe-radius", type=float, default=12.0)
    p.add_argument("--contact-stiffness", type=float, default=1.0)
    p.add_argument("--contact-damping", type=float, default=0.5)
    p.add_argument("--floe-density", type=float, default=0.895)
    p.add_argument(
        "--explicit-coupling",
        action="store_true",
        help="pure Newton update (mass matrix = solid mass); exact momentum "
        "bookkeeping, stable only for a body heavier than its added mass",
    )
    p.add_argument("--out", type=str, required=True)
    p.set_defaults(func=task_ice)

    p = sub.add_parser("floe-drag")
    p.add_argument("--nx", type=int, default=320)
    p.add_argument("--ny", type=int, default=200)
    p.add_argument("--n-steps", type=int, default=8000)
    p.add_argument("--warmup", type=int, default=4000)
    p.add_argument("--floe-radius", type=float, default=12.0)
    p.add_argument("--dtype", type=str, default="float64", choices=["float64", "float32"])
    p.add_argument("--out", type=str, required=True)
    p.set_defaults(func=task_floe_drag)

    p = sub.add_parser("aggregate")
    p.add_argument("--indir", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.set_defaults(func=task_aggregate)

    p = sub.add_parser("figures")
    p.add_argument("--indir", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.set_defaults(func=task_figures)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())