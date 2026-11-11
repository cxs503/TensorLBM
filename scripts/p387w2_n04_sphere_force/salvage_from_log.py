#!/usr/bin/env python3
"""Salvage a drag/torque case record from a run log (partial statistics).

When a case is still running at the end of the node's wall clock, its record is
reconstructed from the periodic report lines the runner prints, using the
**second half** of the printed samples.  The record is marked
``"partial_from_log": true`` and carries the sample count and step range so the
provenance of every number stays visible; nothing is extrapolated.

    python scripts/p387w2_n04_sphere_force/salvage_from_log.py runs/p387w2_n04/conv_d8.log \
        --layer drag --out artifacts/p387w2_n04/cases
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DRAG_RE = re.compile(
    r"step\s+(\d+)\s+Re=\s*([\d.]+)\s+Cd=\s*([\d.]+)\s+Cd_drive=\s*([\d.]+)\s+g=\s*([\d.eE+-]+)\s+mass=\s*([\d.eE+-]+)"
)
TORQUE_RE = re.compile(
    r"step\s+(\d+)\s+C_M=\s*([\d.]+)\s+\(Stokes\s+([\d.]+)\)\s+C_M\[pressure-only\]=\s*([\d.eE+-]+)\s+mass=\s*([\d.eE+-]+)"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--layer", choices=["drag", "torque"], required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "p387w2_n04" / "cases")
    ap.add_argument("--name", type=str, default=None)
    args = ap.parse_args()

    args.log = args.log.resolve()
    text = args.log.read_text()
    samples = []
    for line in text.splitlines():
        if args.layer == "drag":
            m = DRAG_RE.search(line)
            if m:
                samples.append(
                    {
                        "step": int(m.group(1)),
                        "re": float(m.group(2)),
                        "cd": float(m.group(3)),
                        "cd_drive": float(m.group(4)),
                        "mass": float(m.group(6)),
                    }
                )
        else:
            m = TORQUE_RE.search(line)
            if m:
                samples.append(
                    {
                        "step": int(m.group(1)),
                        "cm": float(m.group(2)),
                        "cm_stokes": float(m.group(3)),
                        "cm_pressure_only": float(m.group(4)),
                        "mass": float(m.group(5)),
                    }
                )
    if len(samples) < 6:
        raise SystemExit(f"not enough samples in {args.log} ({len(samples)})")
    half = samples[len(samples) // 2 :]
    name = args.name or args.log.stem
    out = args.out / f"{name}.json"
    if args.layer == "drag":
        cds = [s["cd"] for s in half]
        cd = statistics.mean(cds)
        rec = {
            "name": name,
            "layer": "drag",
            "case": "drag",
            "partial_from_log": True,
            "log": str(args.log.relative_to(ROOT)),
            "config": {
                "reynolds_target": round(statistics.mean([s["re"] for s in half]), 2),
                "diameter": float(re.search(r"--diameter (\d+)", text).group(1)) if "--diameter" in text else None,
                "steps": samples[-1]["step"],
                "dtype": "float32",
                "collision": "trt" if "trt" in args.log.name else "bgk",
                "lateral_diameters": 5.0,
            },
            "cd_mean": cd,
            "cd_std": statistics.pstdev(cds),
            "cd_drive_mean": statistics.mean([s["cd_drive"] for s in half]),
            "reynolds_measured": statistics.mean([s["re"] for s in half]),
            "max_mass_residual": max(s["mass"] for s in samples),
            "max_closure_residual": float("nan"),
            "samples": len(half),
            "step_range": [half[0]["step"], half[-1]["step"]],
        }
    else:
        cms = [s["cm"] for s in half]
        rec = {
            "name": name,
            "layer": "torque",
            "case": "torque",
            "partial_from_log": True,
            "log": str(args.log.relative_to(ROOT)),
            "config": {
                "re_omega_target": half[-1]["cm_stokes"],
                "diameter": None,
                "box_diameters": 6.0,
                "steps": samples[-1]["step"],
            },
            "cm_mean": statistics.mean(cms),
            "cm_std": statistics.pstdev(cms),
            "cm_pressure_only_mean": statistics.mean([s["cm_pressure_only"] for s in half]),
            "cm_stokes_reference": 8.0 / half[-1]["cm_stokes"],
            "cm_vs_stokes_pct": 100.0 * (statistics.mean(cms) / (8.0 / half[-1]["cm_stokes"]) - 1.0),
            "max_mass_residual": max(s["mass"] for s in samples),
            "samples": len(half),
            "step_range": [half[0]["step"], half[-1]["step"]],
        }
    for key, val in list(rec.items()):
        if isinstance(val, float) and not math.isfinite(val):
            rec[key] = None
    out.write_text(json.dumps(rec, indent=1, default=float))
    print(f"wrote {out} (partial, {len(half)} samples, steps {rec['step_range']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())