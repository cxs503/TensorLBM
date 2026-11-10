#!/usr/bin/env python
"""Run the P482-N05 collision-stability envelope campaign (staged, resumable).

Stage ``planes``   -- von Neumann plane scans, one JSON per tier/variant
Stage ``foils``    -- the two declared MRT foils of the falsifiability leg
Stage ``measure``  -- measured side: exchange law, boundary, negative controls
Stage ``assemble`` -- envelope JSON + figure + manifest (digest verified)

Existing stage outputs are reused, so a failed stage never costs the previous
ones.  Usage::

    PYTHONPATH=src python scripts/run_p482_n05_stability_envelope.py --stage all \
        --out TensorLBM/artifacts/p482_n05
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.stability import envelope as ev  # noqa: E402
from tensorlbm.stability.collision_stability import PREDECLARED  # noqa: E402


def _tier_planes(planes_dir: pathlib.Path, quick: bool, force: bool) -> dict[str, dict]:
    planes: dict[str, dict] = {}
    for tier in ev.TIERS:
        for variant in ev.VARIANTS:
            if tier == "bgk" and variant == "oracle":
                continue
            label = ev.tier_label(tier, variant)
            path = planes_dir / f"{label.replace(':', '_')}.json"
            if force or not path.exists():
                t0 = time.time()
                plane = (ev.scan_plane(tier, variant, n_tau=13, n_ma=13, n_k=11)
                         if quick else ev.scan_plane(tier, variant))
                path.write_text(json.dumps(plane, sort_keys=True), encoding="utf-8")
                print(f"[plane] {label}: cases={plane['n_cases']} "
                      f"stable_area={plane['stable_area_ratio']:.4f} ({time.time() - t0:.0f}s)",
                      flush=True)
            planes[label] = json.loads(path.read_text(encoding="utf-8"))
    return planes


def _foil_planes(planes_dir: pathlib.Path, quick: bool, force: bool) -> dict[str, dict]:
    foils: dict[str, dict] = {}
    for family in PREDECLARED["foil_rates"]:
        path = planes_dir / f"mrt_foil_{family}_canonical.json"
        if force or not path.exists():
            t0 = time.time()
            plane = ev._foil_plane(family)
            path.write_text(json.dumps(plane, sort_keys=True), encoding="utf-8")
            print(f"[foil] {family}: stable_area={plane['stable_area_ratio']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        foils[family] = json.loads(path.read_text(encoding="utf-8"))
    return foils


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "TensorLBM" / "artifacts" / "p482_n05"))
    parser.add_argument("--stage", default="all",
                        choices=("planes", "foils", "measure", "assemble", "all"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true", help="recompute even if cached")
    args = parser.parse_args(argv)

    out = pathlib.Path(args.out)
    planes_dir = out / "planes"
    planes_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    planes = _tier_planes(planes_dir, args.quick, args.force) if args.stage in ("planes", "all", "assemble") else {}
    foils = _foil_planes(planes_dir, args.quick, args.force) if args.stage in ("foils", "all", "assemble") else {}

    meas_path = out / "measurements.json"
    measurements = None
    if args.stage in ("measure", "all"):
        measurements = {
            "exchange_law": ev.exchange_law(ev.TIERS, n_box=16 if args.quick else None),
            "boundary": ev.boundary_table(ev.TIERS, n_stations=5 if args.quick else 9),
            "negative_controls": ev.negative_controls_report(((0.6, 0.2),) if args.quick else None),
            "verdict_agreement": ev.verdict_agreement(ev.TIERS, n_sample=4 if args.quick else 12),
        }
        meas_path.write_text(json.dumps(measurements, sort_keys=True), encoding="utf-8")
        print(f"[measure] written -> {meas_path}", flush=True)
    elif meas_path.exists():
        measurements = json.loads(meas_path.read_text(encoding="utf-8"))

    if args.stage in ("assemble", "all"):
        # honest wall-time estimate: from the oldest cached artefact (planes were
        # produced by an earlier invocation of the same staged runner)
        stamps = [p.stat().st_mtime for p in planes_dir.glob("*.json")]
        if meas_path.exists():
            stamps.append(meas_path.stat().st_mtime)
        campaign_seconds = (time.time() - min(stamps)) if stamps else (time.time() - t_start)
        envelope = ev.assemble_envelope(
            planes, foils, measurements, runtime_seconds=campaign_seconds
        )
        envelope["m4_reproducibility"]["recompute_sample_bgk"] = ev.recompute_check("bgk", "canonical")
        envelope["m4_reproducibility"]["recompute_sample_mrt_b"] = ev.recompute_check("mrt_b", "canonical")
        envelope["digest"] = ev.envelope_digest(envelope)
        ev.write_artifacts(envelope, planes, foils, out)
        print(json.dumps({
            "digest": envelope["digest"],
            "verify": ev.verify_envelope(out / "stability_envelope.json"),
            "runtime_seconds": envelope["runtime_seconds"],
            "stable_area": {k: v["stable_area_ratio"] for k, v in envelope["tiers"].items()},
            "max_stable_ma": {k: v["max_stable_ma_at_tau_min"] for k, v in envelope["tiers"].items()},
            "m1_boundary_dev_box": (envelope["m1_boundary"] or {}).get("max_abs_rel_dev_vs_box_grid"),
            "m1_boundary_dev_fine": (envelope["m1_boundary"] or {}).get("max_abs_rel_dev_vs_fine_grid"),
            "m1_exchange_rel_sigma": (envelope["m1_exchange_law"] or {}).get("max_rel_sigma_dev"),
            "m1_exchange_abs_lambda": (envelope["m1_exchange_law"] or {}).get("max_abs_lambda_dev"),
            "m3_all_red": (envelope["m3_negative_controls"] or {}).get("all_red"),
            "m3_falsification": (envelope["m3_falsification"] or {}).get("verdict"),
        }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
