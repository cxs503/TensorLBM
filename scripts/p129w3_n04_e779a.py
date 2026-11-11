#!/usr/bin/env python
"""P129W3-N04 — INSEAN E779A unsteady open-water / cavitation-inception runner.

Modes
-----
``geometry``  geometry gate (expanded area ratio, pitch ratio, hub ratio) from
              the voxel mask at several rotor phases;
``point``     one advance ratio ``J`` (moving-boundary run, JSON + optional
              field snapshot);
``sweep``     a set of ``J`` values, shardable with ``--shard/--nshards``;
``gci``       three-grid grid-convergence study at a fixed ``J``.

Every product is a JSON file under ``--out``; nothing is transcribed by hand —
the report generator reads these files back.

Examples
--------
    python scripts/p129w3_n04_e779a.py geometry --D 48 --out artifacts/p129w3_n04_e779a
    python scripts/p129w3_n04_e779a.py sweep --js 0.1,0.2,0.3 --D 32 --shard 0 --nshards 3
    python scripts/p129w3_n04_e779a.py gci --j 0.6 --grids 24,32,40
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from tensorlbm.insean_e779a import (
    E779A_SPEC,
    OpenWaterConfig,
    geometry_gate,
    run_openwater_point,
)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"not serialisable: {type(obj)}")


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in payload.items() if not k.startswith("_")}
    path.write_text(
        json.dumps(clean, indent=1, sort_keys=True, default=_json_default), encoding="utf-8"
    )
    return path


def _strip_fields(out: dict) -> dict:
    """Keep the field snapshot out of the JSON (saved separately as .npz)."""
    return {k: v for k, v in out.items() if k not in ("fields", "_mask")}


def _save_fields(out: dict, path: Path) -> None:
    fld = out.get("fields")
    if not fld:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        rho=fld["rho"],
        ux=fld["ux"],
        uy=fld["uy"],
        uz=fld["uz"],
        origin=np.asarray(fld["origin"], dtype=np.float64),
        diameter_lu=np.asarray([fld["diameter_lu"]], dtype=np.float64),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["geometry", "point", "sweep", "gci"])
    ap.add_argument("--j", type=float, default=0.6)
    ap.add_argument("--js", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--D", type=int, default=32)
    ap.add_argument("--grids", type=str, default="24,32,40")
    ap.add_argument("--axial", type=float, default=4.0)
    ap.add_argument("--lateral", type=float, default=3.0)
    ap.add_argument("--tip-mach", type=float, default=0.18)
    ap.add_argument("--min-thickness", type=float, default=0.9)
    ap.add_argument("--save-fields", action="store_true")
    ap.add_argument("--tau", type=float, default=0.5008)
    ap.add_argument("--smagorinsky", type=float, default=0.0)
    ap.add_argument("--warmup-revs", type=float, default=4.0)
    ap.add_argument("--avg-revs", type=float, default=2.0)
    ap.add_argument("--steps-per-rev-max", type=int, default=4000)
    ap.add_argument("--strut-diameter-ratio", type=float, default=0.0)
    ap.add_argument("--rotation-sign", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", type=str, default="artifacts/p129w3_n04_e779a")
    args = ap.parse_args(argv)

    import torch

    torch.set_num_threads(args.threads)
    out_root = Path(args.out)

    cfg_kwargs = dict(
        diameter_lu=args.D,
        axial_diameters=args.axial,
        lateral_diameters=args.lateral,
        tip_mach=args.tip_mach,
        tau=args.tau,
        smagorinsky=args.smagorinsky,
        warmup_revs=args.warmup_revs,
        avg_revs=args.avg_revs,
        steps_per_rev_max=args.steps_per_rev_max,
        strut_diameter_ratio=args.strut_diameter_ratio,
        min_thickness_lu=args.min_thickness,
        save_fields=args.save_fields,
        rotation_sign=args.rotation_sign,
    )

    if args.mode == "geometry":
        res = geometry_gate(diameter_lu=args.D, angles_deg=(0.0, 7.5, 13.0, 27.0, 45.0))
        res["observed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        p = _write(out_root / f"geometry_gate_D{args.D}.json", res)
        print(f"[geometry] D={args.D} worst={res['worst_error_pct']} passed={res['passed']} -> {p}")
        return 0

    if args.mode == "point":
        cfg = OpenWaterConfig(**cfg_kwargs)
        out = run_openwater_point(args.j, cfg, verbose=True)
        tag = f"D{args.D}_J{args.j:.2f}"
        if abs(args.min_thickness - 0.9) > 1e-9:
            tag += f"_t{args.min_thickness:.1f}"
        if args.strut_diameter_ratio > 0:
            tag += f"_strut{args.strut_diameter_ratio:.3f}"
        if args.smagorinsky > 0:
            tag += f"_smag{args.smagorinsky:.2f}"
        _save_fields(out, out_root / "fields" / f"fields_{tag}.npz")
        payload = _strip_fields(out)
        payload["wall_time_s"] = None
        p = _write(out_root / "points" / f"point_{tag}.json", payload)
        print(
            f"[point] {tag} kT={out['kT']:+.4f} kQ={out['kQ']:+.4f} eta={out['eta']:+.3f} "
            f"drift={out['convergence']['thrust']['max_block_deviation']:.4f} -> {p}",
            flush=True,
        )
        return 0

    if args.mode == "sweep":
        js = [float(x) for x in args.js.split(",") if x.strip()]
        mine = [j for i, j in enumerate(js) if i % args.nshards == args.shard]
        cfg = OpenWaterConfig(**cfg_kwargs)
        print(f"[sweep] shard {args.shard}/{args.nshards} J={mine}", flush=True)
        for j in mine:
            t0 = time.perf_counter()
            out = run_openwater_point(j, cfg, verbose=False)
            tag = f"D{args.D}_J{j:.2f}"
            if abs(args.min_thickness - 0.9) > 1e-9:
                tag += f"_t{args.min_thickness:.1f}"
            if args.strut_diameter_ratio > 0:
                tag += f"_strut{args.strut_diameter_ratio:.3f}"
            if args.smagorinsky > 0:
                tag += f"_smag{args.smagorinsky:.2f}"
            _save_fields(out, out_root / "fields" / f"fields_{tag}.npz")
            payload = _strip_fields(out)
            payload["wall_time_s"] = time.perf_counter() - t0
            p = _write(out_root / "points" / f"point_{tag}.json", payload)
            print(
                f"[sweep] J={j:.2f} kT={out['kT']:+.4f} kQ={out['kQ']:+.4f} eta={out['eta']:+.3f} "
                f"drift={out['convergence']['thrust']['max_block_deviation']:.4f} "
                f"t={payload['wall_time_s']:.0f}s -> {p}",
                flush=True,
            )
        return 0

    if args.mode == "gci":
        grids = [int(x) for x in args.grids.split(",") if x.strip()]
        for D in grids:
            kw = dict(cfg_kwargs)
            kw["diameter_lu"] = D
            cfg = OpenWaterConfig(**kw)
            t0 = time.perf_counter()
            out = run_openwater_point(args.j, cfg, verbose=False)
            tag = f"GCI_D{D}_J{args.j:.2f}"
            _save_fields(out, out_root / "fields" / f"fields_{tag}.npz")
            payload = _strip_fields(out)
            payload["wall_time_s"] = time.perf_counter() - t0
            p = _write(out_root / "gci" / f"gci_D{D}.json", payload)
            print(
                f"[gci] D={D} kT={out['kT']:+.4f} kQ={out['kQ']:+.4f} "
                f"t={payload['wall_time_s']:.0f}s -> {p}",
                flush=True,
            )
        return 0

    raise SystemExit(f"unknown mode {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())