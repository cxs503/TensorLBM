#!/usr/bin/env python3
"""B6 SUBOFF bare hull Re=1000 — independent verifier.

The case directory ships ``run.py`` (simulation) and ``result.json``
(summaries).  This verifier *recomputes* every judged number from the archived
raw data instead of trusting the summaries, following the repo rule that
geometry must be validated (Cb / L / B) before a drag number is believed:

  gate 1  geometry — the voxel mask is rebuilt from the case config and
          compared against the analytic SUBOFF surface-of-revolution form
          statistics (``suboff_cad.suboff_statistics``): L/D, max radius,
          prismatic coefficient Cp = V /(A_max L) and domain blockage.
  gate 2  forces   — Cd_p / Cd_f are recomputed from the archived
          ``final_field.pt`` with ``drag_pressure_integration`` and
          ``drag_friction_integration`` (mix50 plus the two bounding
          formulas standard / faces), normalised on the wetted area pi*D*L.
  gate 3  history  — window means and the convergence drift (last 20% vs
          previous 20% of force samples) are recomputed from the raw
          ``forces_history.json``; the recomputed primary Cd_tot must match
          ``result.json`` to 1e-9 (arithmetic identity check).
  gate 4  accept   — |err| <= gate (default 3%) vs the locked reference
          Blasius laminar flat-plate Cf = 1.328/sqrt(Re) on the wetted area
          pi*D*L, evaluated on BOTH the window mean and the final-field
          recomputation.

Reference frame note: the repo Re=1000 family uses Blasius Cf = 1.328/sqrt(Re)
= 0.041995 with S = pi*D*L.  The "Ct ~ 0.004 (exp)" number in
``benchmarks/TODO.md`` is the AFF-8 full-scale Re=2e6 total-drag coefficient
and does NOT apply to Re=1000 (see ``src/tensorlbm/suboff_reference_data.py``).

Usage
-----
    python verify.py                       # verify ./ (the case directory)
    python verify.py <out_dir> [...]       # verify one or more run outputs
    python verify.py <out_dir> --write     # also write verify_report.json

Exit code: 0 = PASS, 1 = FAIL, 2 = WITHHELD (archive data missing/insufficient).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tensorlbm.drag_pressure import (  # noqa: E402
    drag_friction_integration,
    drag_pressure_integration,
)
from tensorlbm.general_sim import (  # noqa: E402
    CollisionModel,
    ForceMethod,
    GeneralSimConfig,
    GeneralSimEngine,
    GeometryConfig,
    GeometrySource,
    LatticeModel,
    OutputConfig,
    PhysicsConfig,
    SolverConfig,
    WallTreatment,
)
from tensorlbm.suboff_cad import (  # noqa: E402
    SuboffConfig,
    build_suboff_mask,
    suboff_statistics,
)

SUBOFF_LENGTH_M = 4.356
SUBOFF_RADIUS_M = 0.254
U_PHYS = 1.0e-3
RE_TARGET = 1000.0

# Locked reference (repo Re=1000 family): Blasius laminar flat plate.
CF_REF = 1.328 / math.sqrt(RE_TARGET)

GATE_DRAG = 0.03  # 3% acceptance on the drag coefficient
GATE_GEOM_CP = 0.15  # voxel vs analytic prismatic coefficient
GATE_GEOM_R = 0.20  # voxel max radius vs analytic R (coarse-grid regime)
GATE_BLOCKAGE = 0.10  # domain blockage (mid-body frontal area fraction)
GATE_DRIFT = 1.0  # % window-to-window drift still called "converged"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _fmt_row(name: str, value, ref=None, err_pct=None, ok=None, unit="") -> str:
    line = f"  {name:<34s} {value:>14s}"
    if ref is not None:
        line += f" {ref:>12.6f}"
    if err_pct is not None:
        line += f" {err_pct:>9.2f}%"
    if ok is not None:
        line += f"  {'OK' if ok else 'FAIL'}"
    return line + (f" {unit}" if unit else "")


def window_mean(samples: list[dict], n: int, key: str = "cd_total") -> float:
    """Mean of ``key`` over the last ``n`` samples (n>len -> all samples)."""
    w = samples[-n:]
    return sum(float(s[key]) for s in w) / len(w)


def drift_pct(samples: list[dict], frac: float = 0.2, key: str = "cd_total"):
    """(last-frac mean vs previous-frac mean) drift in % + window sizes."""
    vals = [float(s[key]) for s in samples]
    n = len(vals)
    w = max(1, int(n * frac))
    if n < 2 * w:
        return float("nan"), w, n
    tail = sum(vals[-w:]) / w
    prev = sum(vals[-2 * w : -w]) / w
    return (tail - prev) / prev * 100.0, w, n


def build_engine(resolution: int, friction: str, p0: str, device: str) -> GeneralSimEngine:
    """Rebuild the exact case config (same knobs as run.py) and set up geometry."""
    viscosity = U_PHYS * SUBOFF_LENGTH_M / 1000.0
    config = GeneralSimConfig(
        name=f"verify_suboff_re1000_L{resolution}",
        geometry=GeometryConfig(
            source=GeometrySource.PARAMETRIC_SUBOFF,
            suboff_length=SUBOFF_LENGTH_M,
            suboff_radius=SUBOFF_RADIUS_M,
        ),
        physics=PhysicsConfig(
            density=1000.0,
            viscosity=viscosity,
            inlet_velocity=U_PHYS,
            reference_length=SUBOFF_LENGTH_M,
        ),
        solver=SolverConfig(
            lattice=LatticeModel.D3Q19,
            collision=CollisionModel.MRT,
            resolution=resolution,
            domain_padding=(1.0, 4.0, 1.0, 1.0, 1.0, 1.0),
            max_steps=1,
            snapshot_interval=10_000_000,
            force_sample_interval=10,
            device=device,
            wall_treatment=WallTreatment.AUTO,
            force_method=ForceMethod.PRESSURE_FRICTION,
            pressure_extrap="none",
            p0_method=p0,
            friction_formula=friction,
            mass_correction=True,
            mass_correction_interval=200,
            smagorinsky_cs=0.05,
        ),
        output=OutputConfig(directory="", formats=[], save_macroscopic=False, save_forces=True),
    )
    engine = GeneralSimEngine(config)
    engine.setup()
    return engine


# --------------------------------------------------------------------------
# gate 1: geometry
# --------------------------------------------------------------------------
def check_geometry(engine: GeneralSimEngine, resolution: int, res: dict) -> dict:
    L_lb = float(resolution)
    R_lb = SUBOFF_RADIUS_M / (SUBOFF_LENGTH_M / L_lb)
    # engine (and build_suboff_mask) layout is (nz, ny, nx): hull axis along x
    # = the LAST axis, cross-section = (nz, ny).
    nz, ny, nx = engine.solid.shape

    analytic = suboff_statistics("bare_hull", length=L_lb, radius=R_lb, config=SuboffConfig())
    # Independent voxel rebuild (not the engine's own mask object).
    mask, mstats = build_suboff_mask(
        hull_type="bare_hull",
        nx=nx,
        ny=ny,
        nz=nz,
        cx=nx * 0.25,
        cy=ny * 0.5,
        cz=nz * 0.5,
        length=L_lb,
        radius=R_lb,
        config=SuboffConfig(),
        device="cpu",
    )
    mask = mask.bool()
    n_solid = int(mask.sum().item())
    cy, cz = ny * 0.5, nz * 0.5
    zz, yy = torch.meshgrid(
        torch.arange(nz, dtype=torch.float64),
        torch.arange(ny, dtype=torch.float64),
        indexing="ij",
    )
    r2 = (yy - cy) ** 2 + (zz - cz) ** 2
    section = mask.any(dim=2)  # (nz, ny): cells occupied at some x
    r_max_voxel = float(r2[section].max().sqrt().item())

    # Volume / prismatic coefficient from the voxel count (1 cell = 1 lu^3).
    v_voxel = float(n_solid)
    a_max = math.pi * R_lb**2 * L_lb  # lu^3 of the circumscribed cylinder
    cp_voxel = v_voxel / a_max
    cp_analytic = float(analytic["prismatic_coefficient"])
    l_d_analytic = float(analytic["L_D_ratio"])
    l_d_voxel = L_lb / (2.0 * r_max_voxel)

    # Domain blockage: mid-body frontal-area fraction (max over x of the
    # solid fraction of the cross-section).
    xprof = mask.sum(dim=(0, 1)).float() / float(nz * ny)
    blockage = float(xprof.max().item())

    err_cp = (cp_voxel - cp_analytic) / cp_analytic * 100.0
    err_r = (r_max_voxel - R_lb) / R_lb * 100.0
    ok = (
        abs(err_cp) <= GATE_GEOM_CP * 100.0
        and abs(err_r) <= GATE_GEOM_R * 100.0
        and blockage <= GATE_BLOCKAGE
    )
    return {
        "gate": "geometry",
        "pass": bool(ok),
        "domain_lu": [nx, ny, nz],  # (nx, ny, nz) = streaming, span, vertical
        "L_cells": L_lb,
        "R_lb": R_lb,
        "solid_cells_voxel": n_solid,
        "solid_cells_result_json": res.get("solid_cells")
        or (res.get("primary_config") or {}).get("solid_cells"),
        "solid_cells_match": (
            (res.get("solid_cells") or (res.get("primary_config") or {}).get("solid_cells"))
            in (None, n_solid)
        ),
        "r_max_voxel_lu": r_max_voxel,
        "r_max_err_pct": err_r,
        "L_D_analytic": l_d_analytic,
        "L_D_voxel": l_d_voxel,
        "Cp_voxel": cp_voxel,
        "Cp_analytic": cp_analytic,
        "Cp_err_pct": err_cp,
        "blockage": blockage,
        "wetted_area_analytic_lu2": float(analytic["wetted_area_lu2"]),
        "thresholds": {
            "Cp_err_pct": GATE_GEOM_CP * 100.0,
            "r_max_err_pct": GATE_GEOM_R * 100.0,
            "blockage": GATE_BLOCKAGE,
        },
        "mask_stats_L_D": mstats.get("L_D_ratio") if isinstance(mstats, dict) else None,
        "mask_stats_Cp": mstats.get("prismatic_coefficient") if isinstance(mstats, dict) else None,
        "mask_stats_disp_lu3": mstats.get("displacement_lu3") if isinstance(mstats, dict) else None,
    }


# --------------------------------------------------------------------------
# gate 2: forces recomputed from the archived final field
# --------------------------------------------------------------------------
def check_final_field(
    engine: GeneralSimEngine, out_dir: Path, resolution: int, gate: float = GATE_DRAG
) -> dict:
    field_path = out_dir / "final_field.pt"
    if not field_path.exists():
        return {"gate": "final_field", "pass": None, "reason": "final_field.pt missing"}
    f = torch.load(field_path, map_location="cpu", weights_only=True)
    if tuple(f.shape) != tuple(engine.f.shape):
        return {
            "gate": "final_field",
            "pass": False,
            "reason": f"field shape {tuple(f.shape)} != engine {tuple(engine.f.shape)}",
        }
    device = engine.f.device
    f = f.to(device)

    L_lb = float(resolution)
    R_lb = SUBOFF_RADIUS_M / (SUBOFF_LENGTH_M / L_lb)
    u_lb = engine.uc.u_lb
    nu_lb = engine.uc.nu_lb
    dpS_wet = 0.5 * u_lb**2 * math.pi * (2.0 * R_lb) * L_lb
    mesh, solid = engine.mesh, engine.solid

    rows = {}
    for p0 in ("near_wall", "far_field", "domain_avg", "inlet"):
        fx_p, _, _ = drag_pressure_integration(
            f, mesh, dpS_wet, extrap="none", p0_method=p0, solid=solid
        )
        row = {"cd_p": float(fx_p)}
        for formula in ("standard", "faces", "mix50"):
            fx_f, _, _ = drag_friction_integration(
                f, mesh, dpS_wet, nu_lb, formula=formula, solid=solid
            )
            row[f"cd_f_{formula}"] = float(fx_f)
            row[f"cd_tot_{formula}"] = row["cd_p"] + float(fx_f)
        rows[p0] = row

    primary = rows["near_wall"]
    err = (primary["cd_tot_mix50"] - CF_REF) / CF_REF * 100.0
    return {
        "gate": "final_field",
        "pass": abs(err) <= gate * 100.0,
        "reference": "Blasius Cf=1.328/sqrt(Re), wetted area pi*D*L",
        "Cf_ref": CF_REF,
        "dpS_wet": dpS_wet,
        "rows": rows,
        "primary_p0": "near_wall",
        "cd_p": primary["cd_p"],
        "cd_f_mix50": primary["cd_f_mix50"],
        "cd_tot_mix50": primary["cd_tot_mix50"],
        "err_pct": err,
        "mix50_bounds_ok": (
            primary["cd_f_standard"] <= primary["cd_f_mix50"] <= primary["cd_f_faces"]
        ),
    }


# --------------------------------------------------------------------------
# gate 3: raw force history
# --------------------------------------------------------------------------
def check_history(out_dir: Path, res: dict, gate: float = GATE_DRAG) -> dict:
    hist_path = out_dir / "forces_history.json"
    if not hist_path.exists():
        return {"gate": "history", "pass": None, "reason": "forces_history.json missing"}
    h = json.loads(hist_path.read_text())
    samples = h["samples"]
    rescale = float(h["rescale_frontal_to_wetted"])
    n = len(samples)
    if n < 20:
        return {"gate": "history", "pass": None, "reason": f"only {n} samples"}

    n_win = min(1000, n)
    cd_p = window_mean(samples, n_win, "cd_pressure") * rescale
    cd_f = window_mean(samples, n_win, "cd_friction") * rescale
    cd_tot = cd_p + cd_f
    cd_tot_last500 = window_mean(samples, min(500, n), "cd_total") * rescale

    drift, w, _ = drift_pct(samples, 0.2)
    conv = {}
    for frac in (0.25, 0.5, 0.75, 1.0):
        k = int(n * frac)
        seg = samples[max(0, k - n_win) : k]
        conv[f"{int(frac * 100)}%"] = round(sum(s["cd_total"] for s in seg) / len(seg) * rescale, 6)

    # arithmetic identity check against result.json summaries
    ident = {}
    for key, mine in (
        ("Cd_pressure", cd_p),
        ("Cd_friction", cd_f),
        ("Cd_total", cd_tot),
        ("Cd_total_last5000", cd_tot_last500),
    ):
        ref = res.get(key)
        ident[key] = None if ref is None else abs(float(ref) - mine)
    ident_ok = all(v is None or v < 1e-9 for v in ident.values())

    err = (cd_tot - CF_REF) / CF_REF * 100.0
    converged = (not math.isnan(drift)) and abs(drift) <= GATE_DRIFT
    return {
        "gate": "history",
        "pass": bool(abs(err) <= gate * 100.0 and converged and ident_ok),
        "n_samples": n,
        "sample_interval": h.get("force_sample_interval"),
        "window_samples": n_win,
        "window_steps": n_win * (h.get("force_sample_interval") or 10),
        "cd_p": cd_p,
        "cd_f": cd_f,
        "cd_tot": cd_tot,
        "cd_tot_last500": cd_tot_last500,
        "err_pct": err,
        "drift_pct_last20_vs_prev20": drift,
        "drift_window_samples": w,
        "converged": converged,
        "convergence_windows": conv,
        "identity_max_abs_diff": ident,
        "identity_ok": ident_ok,
    }


# --------------------------------------------------------------------------
# geometry-only sweep (no simulation): voxel vs analytic form per resolution
# --------------------------------------------------------------------------
def geometry_table(resolutions: list[int], device: str = "cpu") -> list[dict]:
    """Mask-vs-analytic SUBOFF form check for each resolution L (cells per hull).

    Cheap (mask build only, no time stepping) — the repo rule "validate the
    geometry (Cb / L / B) before believing a drag number" as a runnable gate.
    """
    rows = []
    for L in resolutions:
        engine = build_engine(L, "mix50", "near_wall", device)
        g = check_geometry(engine, L, {})
        nz, ny, nx = engine.solid.shape
        rows.append(
            {
                "L_cells": L,
                "domain_lu": [nx, ny, nz],
                "n_cells": nx * ny * nz,
                "R_lb_analytic": g["R_lb"],
                "r_max_voxel_lu": g["r_max_voxel_lu"],
                "r_max_err_pct": g["r_max_err_pct"],
                "L_D_analytic": g["L_D_analytic"],
                "L_D_voxel": g["L_D_voxel"],
                "Cp_voxel": g["Cp_voxel"],
                "Cp_analytic": g["Cp_analytic"],
                "Cp_err_pct": g["Cp_err_pct"],
                "blockage": g["blockage"],
                "solid_cells": g["solid_cells_voxel"],
                "pass": g["pass"],
            }
        )
        del engine
    return rows


# --------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------
def verify_one(out_dir: Path, gate: float = GATE_DRAG, device_override: str | None = None) -> dict:
    res_path = out_dir / "result.json"
    if not res_path.exists():
        return {"out_dir": str(out_dir), "status": "WITHHELD", "reason": "result.json missing"}
    res = json.loads(res_path.read_text())
    resolution = int(
        res.get("L_cells")
        or res.get("resolution")
        or (res.get("primary_config") or {}).get("L_cells")
        or 0
    )
    if resolution <= 0:
        return {"out_dir": str(out_dir), "status": "WITHHELD", "reason": "L_cells missing"}
    friction = (res.get("config") or {}).get("friction_formula") or res.get(
        "friction_formula", "mix50"
    )
    p0 = res.get("p0_method", "near_wall")
    device = device_override or res.get("device", "cpu")
    if device.startswith("sdaa") and not (
        getattr(torch, "sdaa", None) and torch.sdaa.is_available()
    ):
        device = "cpu"

    engine = build_engine(resolution, friction, p0, device)
    geom = check_geometry(engine, resolution, res)
    field = check_final_field(engine, out_dir, resolution, gate)
    hist = check_history(out_dir, res, gate)

    # Re covered from the archived kinematics
    u_lb = float(res.get("u_lb") or engine.uc.u_lb)
    nu_lb = float(res.get("nu_lb") or engine.uc.nu_lb)
    re_archived = u_lb * resolution / nu_lb

    gates = [geom, field, hist]
    # Geometry is never a "soft" gate: an invalid mask invalidates the run.
    # The drag gates must be present AND green; absence of both is WITHHELD
    # (never a silent PASS on geometry alone).
    drag_gates = [field, hist]
    decisive = [g for g in drag_gates if g.get("pass") is not None]
    if not geom["pass"]:
        status = "FAIL"
    elif not decisive:
        status = "WITHHELD"
    elif all(g["pass"] for g in decisive):
        status = "PASS"
    else:
        status = "FAIL"
    return {
        "out_dir": str(out_dir),
        "case": "suboff_re1000",
        "status": status,
        "gate_pct": gate * 100.0,
        "reference": {"name": "Blasius Cf=1.328/sqrt(Re)", "Cf_ref": CF_REF, "Re": RE_TARGET},
        "Re_archived": re_archived,
        "Re_ok": abs(re_archived - RE_TARGET) / RE_TARGET < 0.01,
        "live_kinematics": {"u_lb": u_lb, "nu_lb": nu_lb, "tau": engine.uc.tau},
        "gates": {g["gate"]: g for g in gates},
    }


def report(rep: dict) -> None:
    print("=" * 78)
    print(f"suboff_re1000 verify — {rep['out_dir']}")
    print("=" * 78)
    if rep.get("status") == "WITHHELD":
        reason = rep.get("reason")
        if reason:
            print(f"  WITHHELD: {reason}")
        else:
            print("  WITHHELD (nothing decisive to judge):")
            for name, g in (rep.get("gates") or {}).items():
                if g.get("pass") is None:
                    print(f"    - {name}: {g.get('reason')}")
        return
    print(
        f"  Re (archived)      : {rep['Re_archived']:.3f} "
        f"({'OK' if rep['Re_ok'] else 'MISMATCH'})   gate = +/-{rep['gate_pct']:.1f}%"
    )
    print(f"  reference          : {rep['reference']['name']} = {CF_REF:.6f}")

    g = rep["gates"]["geometry"]
    print("\n[gate 1] geometry (voxel mask vs analytic SUBOFF form)")
    print(_fmt_row("L/D analytic", f"{g['L_D_analytic']:.4f}"))
    print(_fmt_row("L/D voxel", f"{g['L_D_voxel']:.4f}"))
    print(_fmt_row("R_max voxel / analytic", f"{g['r_max_voxel_lu']:.4f} / {g['R_lb']:.4f}",
                   err_pct=g["r_max_err_pct"], ok=abs(g["r_max_err_pct"]) <= 20.0))
    print(_fmt_row("Cp (Cb) voxel", f"{g['Cp_voxel']:.5f}", g["Cp_analytic"],
                   g["Cp_err_pct"], abs(g["Cp_err_pct"]) <= 15.0))
    print(_fmt_row("domain blockage", f"{g['blockage']:.5f}", unit="(<=0.10)"))
    print(f"  solid cells        : voxel={g['solid_cells_voxel']} "
          f"result.json={g['solid_cells_result_json']} "
          f"match={g['solid_cells_match']}  -> {'OK' if g['pass'] else 'FAIL'}")

    fld = rep["gates"]["final_field"]
    print("\n[gate 2] forces recomputed from final_field.pt (wetted area pi*D*L)")
    if fld.get("pass") is None:
        print(f"  WITHHELD: {fld.get('reason')}")
    else:
        print(f"  {'p0':<10s} {'Cd_p':>10s} {'Cd_f_std':>10s} {'Cd_f_faces':>11s} "
              f"{'Cd_f_mix50':>11s} {'Cd_tot':>10s} {'err':>8s}")
        for p0, row in fld["rows"].items():
            print(f"  {p0:<10s} {row['cd_p']:>10.6f} {row['cd_f_standard']:>10.6f} "
                  f"{row['cd_f_faces']:>11.6f} {row['cd_f_mix50']:>11.6f} "
                  f"{row['cd_tot_mix50']:>10.6f} "
                  f"{(row['cd_tot_mix50'] - CF_REF) / CF_REF * 100:>+7.2f}%")
        print(_fmt_row("primary (near_wall, mix50)", f"{fld['cd_tot_mix50']:.6f}", CF_REF,
                       fld["err_pct"], abs(fld["err_pct"]) <= rep["gate_pct"]))
        print(f"  standard <= mix50 <= faces : {fld['mix50_bounds_ok']}")

    h = rep["gates"]["history"]
    print("\n[gate 3] raw force history (forces_history.json)")
    if h.get("pass") is None:
        print(f"  WITHHELD: {h.get('reason')}")
    else:
        print(_fmt_row("Cd_total (window mean)", f"{h['cd_tot']:.6f}", CF_REF,
                       h["err_pct"], abs(h["err_pct"]) <= rep["gate_pct"]))
        print(f"  window             : {h['window_samples']} samples "
              f"({h['window_steps']} steps, interval {h['sample_interval']})")
        print(f"  drift last/prev 20%: {h['drift_pct_last20_vs_prev20']:+.4f}% "
              f"(gate |drift|<=1.0%) -> converged={h['converged']}")
        print(f"  windows            : {h['convergence_windows']}")
        print(f"  result.json identity max|diff| = "
              f"{max(v for v in h['identity_max_abs_diff'].values() if v is not None):.3e} "
              f"-> {'OK' if h['identity_ok'] else 'FAIL'}")

    print("\n" + "-" * 78)
    print(f"  STATUS: {rep['status']}")
    if rep["status"] == "WITHHELD":
        print("  (geometry gate evaluated, but no archived drag evidence: "
              "final_field.pt / forces_history.json absent -> drag cannot be judged)")
    print("-" * 78)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify the B6 suboff_re1000 benchmark")
    ap.add_argument("out_dirs", nargs="*", type=Path, default=None,
                    help="run output dir(s); default = this case directory")
    ap.add_argument("--gate", type=float, default=GATE_DRAG, help="acceptance gate (0.03 = 3%%)")
    ap.add_argument("--write", action="store_true", help="write verify_report.json next to each result")
    ap.add_argument("--device", default=None, help="override device (e.g. cpu)")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    ap.add_argument(
        "--geometry-table",
        nargs="+",
        type=int,
        default=None,
        metavar="L",
        help="only run the geometry gate for these resolutions (no time stepping)",
    )
    ap.add_argument("--geometry-out", type=Path, default=None, help="write the geometry table JSON here")
    args = ap.parse_args(argv)

    if args.geometry_table:
        rows = geometry_table(args.geometry_table, args.device or "cpu")
        print(f"{'L':>4s} {'domain':>16s} {'cells':>10s} {'R_lu':>7s} {'R_vox':>7s} "
              f"{'dR%':>7s} {'L/D':>7s} {'L/D_vox':>8s} {'Cp':>7s} {'Cp_vox':>8s} "
              f"{'dCp%':>7s} {'block':>7s}  gate")
        for r in rows:
            print(f"{r['L_cells']:>4d} {'x'.join(str(v) for v in r['domain_lu']):>16s} "
                  f"{r['n_cells']:>10d} {r['R_lb_analytic']:>7.3f} {r['r_max_voxel_lu']:>7.3f} "
                  f"{r['r_max_err_pct']:>+7.2f} {r['L_D_analytic']:>7.3f} {r['L_D_voxel']:>8.3f} "
                  f"{r['Cp_analytic']:>7.4f} {r['Cp_voxel']:>8.4f} {r['Cp_err_pct']:>+7.2f} "
                  f"{r['blockage']:>7.4f}  {'OK' if r['pass'] else 'FAIL'}")
        if args.geometry_out:
            args.geometry_out.parent.mkdir(parents=True, exist_ok=True)
            args.geometry_out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
            print(f"geometry table -> {args.geometry_out}")
        return 0

    out_dirs = args.out_dirs or [_HERE]
    worst = 0
    reports = []
    for d in out_dirs:
        rep = verify_one(Path(d), args.gate, args.device)
        reports.append(rep)
        if args.json:
            print(json.dumps(rep, indent=2, ensure_ascii=False))
        else:
            report(rep)
        if args.write:
            (Path(d) / "verify_report.json").write_text(
                json.dumps(rep, indent=2, ensure_ascii=False)
            )
        worst = max(worst, {"PASS": 0, "FAIL": 1}.get(rep.get("status"), 2))
    return worst


if __name__ == "__main__":
    raise SystemExit(main())