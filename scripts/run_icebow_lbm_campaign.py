#!/usr/bin/env python3
"""Ice-bow resistance campaign (node IN13W2-N06): staged, resumable artifacts.

Stages (each writes one JSON under ``--out`` and is safe to re-run):

``gate``    geometry first-verification gate + parameter read-back for the family
``sdf``     three-spacing SDF volume consistency + GCI
``ladder``  three-grid LBM ladder of the reference member (GCI on ``C_T``)
``angles``  remaining stem angles / families at the middle grid
``sphere``  driver validation against the Schiller-Naumann drag correlation
``re``      Reynolds-number sensitivity of the reference member at the middle grid
``field``   mid-depth velocity slices at the coarsest grid (figure material)
``hybrid``  viscous (LBM) + wave (Michell) sensitivity split at fixed Fn
``all``     every stage above, sequentially

Examples::

    PYTHONPATH=src python3 scripts/run_icebow_lbm_campaign.py --stage gate --out artifacts/icebow_lbm_r2
    PYTHONPATH=src python3 scripts/run_icebow_lbm_campaign.py --stage ladder --grids 64 96 144
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from tensorlbm import icebow_cad as cad
from tensorlbm import icebow_lbm as lbm

REFERENCE_ALPHA = 30.0


def label_of(family: str, alpha: float, cb: float) -> str:
    return f"{family}_a{alpha:g}_cb{cb:g}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dump(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=float))
    return path


def _reference(members: list[cad.IceBowParams]) -> cad.IceBowParams:
    return next(
        m
        for m in members
        if m.family == "straight" and abs(m.alpha_stem_deg - REFERENCE_ALPHA) < 1e-9
    )


# --------------------------------------------------------------------------- #
def stage_gate(members) -> dict:
    rows = []
    for p in members:
        t0 = time.time()
        mesh = cad.build_ice_bow_mesh(p)
        gate = cad.validate_mesh(mesh)
        rev = cad.reverse_parameters(mesh)
        rows.append(
            {
                "label": label_of(p.family, p.alpha_stem_deg, p.cb_target),
                "family": p.family,
                "alpha_stem_deg": p.alpha_stem_deg,
                "beta_ent_deg": p.beta_ent_deg,
                "c_flare": p.c_flare,
                "c_fwd": p.c_fwd,
                "cb_target": p.cb_target,
                "gate": gate,
                "reverse": rev,
                "seconds": round(time.time() - t0, 2),
            }
        )
    return {
        "members": rows,
        "all_passed": all(bool(r["gate"].get("passed")) for r in rows),
        "families": sorted({r["family"] for r in rows}),
        "alpha_angles_deg": sorted({r["alpha_stem_deg"] for r in rows}),
        "cb_range": [min(r["cb_target"] for r in rows), max(r["cb_target"] for r in rows)],
    }


def stage_sdf(members, spacings, mesh_levels) -> dict:
    rows = []
    for p in members:
        resolved = cad.resolve_params(p)
        full = float(cad.design_volume(resolved)["volume_full"])
        per = []
        for h in spacings:
            mesh = cad.build_ice_bow_mesh(p, n_stations=mesh_levels[0], n_levels=mesh_levels[1])
            mask, _lo, spacing = cad.voxelize_mesh(mesh, spacing=h, pad_cells=0)
            per.append({"spacing_L": spacing, "cells": int(mask.sum()), "volume": float(mask.sum()) * spacing**3})
        errs = [100.0 * (v["volume"] - full) / full for v in per]
        row = {
            "label": label_of(p.family, p.alpha_stem_deg, p.cb_target),
            "alpha_stem_deg": p.alpha_stem_deg,
            "volume_full_analytic": full,
            "levels": per,
            "volume_error_percent": errs,
            "max_abs_error_percent": float(max(abs(e) for e in errs)),
        }
        if len(errs) >= 3:
            row["gci"] = cad.gci(errs[-1], errs[-2], errs[-3], r=float(spacings[1] / spacings[2]))
        rows.append(row)
    return {
        "spacings_L": list(spacings),
        "members": rows,
        "worst_abs_error_percent": float(max(r["max_abs_error_percent"] for r in rows)),
        "consistent_under_1_percent": bool(all(r["max_abs_error_percent"] < 1.0 for r in rows)),
    }


def stage_ladder(ref, grids, re, u_in, steps_per_length, drift_tol, out=None) -> dict:
    t0 = time.time()
    out = lbm.run_grid_ladder(
        ref,
        grids=grids,
        re=re,
        target_velocity=u_in,
        steps_per_length=steps_per_length,
        drift_tol=drift_tol,
    )
    out["wall_seconds"] = round(time.time() - t0, 1)
    out["diagnostics"] = _diag(out["runs"])
    return out


def stage_angles(members, ref, mid, re, u_in, steps, drift_tol, reuse=None, out=None) -> dict:
    rows = []
    if reuse is not None:
        rows.append(reuse)
    done = {r["label"] for r in rows}
    for m in members:
        lab = label_of(m.family, m.alpha_stem_deg, m.cb_target)
        if lab in done:
            continue
        t0 = time.time()
        row = lbm.run_case(
            lbm.BowLbmCase(
                params=m,
                cells_per_length=int(mid),
                re=re,
                target_velocity=u_in,
                steps=int(steps),
                min_steps=int(0.5 * steps),
                drift_tol=drift_tol,
            )
        )
        row["wall_seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        print(f"[angles] {row['label']} C_T={row['c_t_lbm']:.4f} ({row['wall_seconds']} s)", flush=True)
        if out is not None:
            _dump(
                out / "lbm_angles.json",
                {
                    "cells_per_length": int(mid),
                    "reynolds": float(re),
                    "u_in": float(u_in),
                    "steps": int(steps),
                    "rows": rows,
                    "diagnostics": _diag(rows),
                },
            )
    return {
        "cells_per_length": int(mid),
        "reynolds": float(re),
        "u_in": float(u_in),
        "steps": int(steps),
        "rows": rows,
        "diagnostics": _diag(rows),
    }


def _diag(rows) -> dict:
    return {
        "max_rel_sem": float(max(r["steady"]["rel_sem"] for r in rows)),
        "max_drift_rel": float(max(r["steady"]["drift_rel"] for r in rows)),
        "max_momentum_budget_percent": float(
            max(abs(r["momentum_budget_percent"]) for r in rows)
        ),
        "u_mean_max_deviation_percent": float(
            max(
                100.0
                * abs(r["setup"]["u_mean_steady"] - r["setup"]["target_velocity"])
                / r["setup"]["target_velocity"]
                for r in rows
            )
        ),
        "all_completed": all(r["status"] == "COMPLETED" for r in rows),
        "all_converged": all(bool(r["converged"]) for r in rows),
    }


def stage_hybrid(angle_rows, fn) -> dict:
    members = cad.family_members()
    wave = cad.resistance_sensitivity(members, fn=fn)
    cw_by_label, info_by_label = {}, {}
    for r in wave["rows"]:
        lab = label_of(r["family"], r["alpha_stem_deg"], round(r["cb"], 2))
        cw_by_label[lab] = float(r["rw_michell_per_density"][-1])
        info_by_label[lab] = {
            "rw_michell_n": float(r["rw_michell_n"]),
            "rw_holtrop_n": float(r["rw_holtrop_n"]),
            "rt_holtrop_n": float(r["rt_holtrop_n"]),
            "rf_n": float(r["rf_n"]),
            "S_m2": float(r["S_m2"]),
            "rw_michell_gci_percent": float(r["rw_michell_gci_percent"]),
            "rt_holtrop_gci_percent": float(r["rt_holtrop_gci_percent"]),
        }
    rows = [r for r in angle_rows if r["label"] in cw_by_label]
    out = lbm.hybrid_sensitivity(
        rows, {"share_wave": wave["share_rw_holtrop"], "cw_by_label": cw_by_label}
    )
    out["fn"] = float(fn)
    out["holtrop_share_wave"] = float(wave["share_rw_holtrop"])
    out["michell_gci_max_percent"] = float(wave["max_rw_michell_gci_percent"])
    out["michell_gci_hybrid_percent"] = (
        float(wave["gci_hybrid_35deg"].get("gci_fine_percent"))
        if wave.get("gci_hybrid_35deg")
        else None
    )
    out["reference_values"] = info_by_label
    out["labels_used"] = sorted(cw_by_label)
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/icebow_lbm_r2")
    # Grid ladder note: the hull draft is 0.08625 L, so N_L = 11.594 * n puts the
    # waterline on a whole voxel layer (n = 4, 6, 9 -> 46, 70, 104 cells per
    # length, draft rounding < 1 %).  These keep the three grids the *same*
    # physical hull, which a GCI ladder requires.
    ap.add_argument(
        "--stage",
        default="all",
        choices=["gate", "sdf", "ladder", "angles", "sphere", "re", "field", "hybrid", "all"],
    )
    ap.add_argument("--grids", type=int, nargs="+", default=[46, 70, 104])
    ap.add_argument("--steps-per-length", type=float, default=45.0)
    ap.add_argument("--alphas", type=float, nargs="+", default=None, help="subset of stem angles for the angles stage")
    ap.add_argument("--drift-tol", type=float, default=0.01)
    ap.add_argument("--re", type=float, default=1000.0)
    ap.add_argument("--u-in", type=float, default=0.08)
    ap.add_argument("--fn", type=float, default=0.20)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("TL_THREADS", "16")))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    import torch

    torch.set_num_threads(max(1, args.threads))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    members = cad.family_members()
    ref = _reference(members)
    grids = args.grids
    mid = grids[len(grids) // 2]
    steps = int(round(args.steps_per_length * mid))

    if args.quick:
        grids = [48, 64, 96]
        mid = grids[len(grids) // 2]
        steps = int(round(args.steps_per_length * mid))
        members = members[:2]

    stage = args.stage
    if stage in ("gate", "all"):
        g = stage_gate(members)
        _dump(out / "geometry_gate.json", g)
        print(f"[gate] {len(members)} members all_passed={g['all_passed']}", flush=True)
    if stage in ("sdf", "all"):
        spacings = [0.04, 0.02] if args.quick else [0.005, 0.0025, 0.00125]
        levels = (41, 31) if args.quick else (201, 121)
        s = stage_sdf(members, spacings, levels)
        _dump(out / "voxel_consistency.json", s)
        print(f"[sdf] worst {s['worst_abs_error_percent']:.3f} %", flush=True)
    if stage in ("ladder", "all"):
        lad = stage_ladder(ref, grids, args.re, args.u_in, args.steps_per_length, args.drift_tol, out)
        _dump(out / "lbm_ladder.json", lad)
        print(f"[ladder] C_T={lad['c_t_per_grid']} gci={lad['gci'].get('gci_fine_percent')}", flush=True)
    if stage in ("angles", "all"):
        reuse = None
        lad_path = out / "lbm_ladder.json"
        if lad_path.exists():
            lad = json.loads(lad_path.read_text())
            if lad["grids"][len(lad["grids"]) // 2] == mid:
                reuse = lad["runs"][len(lad["grids"]) // 2]
        straight = [m for m in members if m.family == "straight"]
        if args.alphas:
            straight = [m for m in straight if any(abs(m.alpha_stem_deg - a) < 1e-9 for a in args.alphas)]
        ang = stage_angles(straight, ref, mid, args.re, args.u_in, steps, args.drift_tol, reuse, out)
        ang["family_members_available"] = [
            label_of(m.family, m.alpha_stem_deg, m.cb_target) for m in cad.family_members()
        ]
        _dump(out / "lbm_angles.json", ang)
        print(f"[angles] {len(ang['rows'])} rows", flush=True)
    if stage in ("sphere", "all"):
        cfg = (
            dict(re=100.0, diameter_cells=6, box=(36, 24, 24), steps=400)
            if args.quick
            else dict(re=100.0, diameter_cells=12, box=(72, 48, 48), steps=4000, sample_every=10)
        )
        sph = lbm.sphere_validation(**cfg)
        _dump(out / "sphere_validation.json", sph)
        print(f"[sphere] Cd={sph['cd_lbm']:.3f} vs {sph['cd_reference']:.3f} ({sph['error_percent']:+.1f} %)", flush=True)
    if stage in ("re", "all"):
        rows = []
        for re_val in (500.0, 1000.0, 2000.0):
            row = lbm.run_case(
                lbm.BowLbmCase(
                    params=ref,
                    cells_per_length=int(mid),
                    re=re_val,
                    target_velocity=args.u_in,
                    steps=steps,
                    min_steps=int(0.5 * steps),
                    drift_tol=args.drift_tol,
                )
            )
            rows.append(row)
            print(f"[re] Re={re_val:g} C_T={row['c_t_lbm']:.4f}", flush=True)
            _dump(
                out / "reynolds_sensitivity.json",
                {
                    "cells_per_length": int(mid),
                    "rows": rows,
                    "c_t_per_re": [r["c_t_lbm"] for r in rows],
                    "relative_spread_percent": float(
                        100.0
                        * (max(r["c_t_lbm"] for r in rows) - min(r["c_t_lbm"] for r in rows))
                        / rows[len(rows) // 2]["c_t_lbm"]
                    ),
                    "diagnostics": _diag(rows),
                },
            )
        spread = (max(r["c_t_lbm"] for r in rows) - min(r["c_t_lbm"] for r in rows)) / rows[
            len(rows) // 2
        ]["c_t_lbm"]
        _dump(
            out / "reynolds_sensitivity.json",
            {
                "cells_per_length": int(mid),
                "rows": rows,
                "c_t_per_re": [r["c_t_lbm"] for r in rows],
                "relative_spread_percent": float(100.0 * spread),
                "diagnostics": _diag(rows),
            },
        )
    if stage in ("field", "all"):
        import numpy as _np

        grid_field = min(grids)
        fields = {}
        for alpha in (20.0, 35.0):
            m = next(x for x in members if x.family == "straight" and abs(x.alpha_stem_deg - alpha) < 1e-9)
            row = lbm.run_case(
                lbm.BowLbmCase(
                    params=m,
                    cells_per_length=int(grid_field),
                    re=args.re,
                    target_velocity=args.u_in,
                    steps=int(round(args.steps_per_length * grid_field)),
                    min_steps=int(0.5 * args.steps_per_length * grid_field),
                    drift_tol=args.drift_tol,
                ),
                keep_field=True,
            )
            fld = row.pop("field")
            fields[f"ux_a{alpha:g}"] = fld["ux"]
            fields[f"uy_a{alpha:g}"] = fld["uy"]
            fields[f"mask_a{alpha:g}"] = fld["mask"]
            _dump(out / f"field_a{alpha:g}.json", {k: v for k, v in row.items() if k != "history"})
            print(f"[field] a={alpha:g} C_T={row['c_t_lbm']:.4f}", flush=True)
        _np.savez_compressed(out / "fields.npz", **fields)
    if stage in ("hybrid", "all"):
        ang_path = out / "lbm_angles.json"
        ang = json.loads(ang_path.read_text())
        hy = stage_hybrid(ang["rows"], args.fn)
        _dump(out / "hybrid.json", hy)
        print(f"[hybrid] monotone={hy['monotone_delta_rt_hybrid']} max={hy['max_abs_delta_rt_hybrid']:.4f}", flush=True)

    # summary + manifest over whatever exists now
    files = {p.name: _sha256(p) for p in sorted(out.glob("*.json")) if p.name not in ("manifest.json",)}
    _dump(
        out / "manifest.json",
        {
            "files": files,
            "python": os.sys.version.split()[0],
            "threads": args.threads,
            "stage": stage,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())