"""Run an IN101-N01 case table and store one JSON record per case.

The runner is a thin, provenance-preserving wrapper around
:func:`tensorlbm.underice_drag.run_underice_channel_native`: it voxelises the
keel geometry, runs the channel, adds the case identifiers and the analytic
frontal-area reference, and writes the record atomically.  A campaign is a
directory::

    <run_dir>/<case_id>/record.json
    <run_dir>/index.json          # case table + environment + per-case status

which is what :mod:`tensorlbm.ice_keel_drag.criteria` consumes.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..underice_drag import (analytic_frontal_area_ratio, triangular_keel_mask,
                             run_underice_channel, run_underice_channel_native)
from .cases import CaseSpec


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _atomic_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True, allow_nan=True))
    os.replace(tmp, path)


def run_case(case: CaseSpec, *, backend: str = "native") -> dict[str, Any]:
    """Run one case and return its record (geometry + drag accounting)."""
    cfg = case.cfg
    mask = None
    if case.keel is not None:
        mask = triangular_keel_mask(case.keel)
    runner = run_underice_channel_native if backend == "native" else run_underice_channel
    t0 = time.time()
    rec = runner(cfg, mask, spec=case.keel)
    rec["case_id"] = case.case_id
    rec["family"] = case.family
    rec["tags"] = dict(case.tags)
    rec["backend"] = backend
    rec["wall_seconds"] = round(time.time() - t0, 2)
    fr = rec["geometry"]["S_fr_x_over_plan"]
    rec["S_fr_over_plan"] = float(fr)
    rec["analytic_S_fr_over_plan"] = analytic_frontal_area_ratio(
        coverage=rec["geometry"]["coverage_geometric"],
        h_keel=0.0 if case.keel is None else case.keel.h_keel,
        h_eff=rec["h_eff"])
    return rec


def run_campaign(cases: Iterable[CaseSpec], out_dir: str | Path, *,
                 backend: str = "native", skip_existing: bool = True,
                 log=print) -> dict[str, Any]:
    """Run every case in ``cases`` into ``out_dir``; returns the index."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    case_list = list(cases)
    index: dict[str, Any] = {
        "mission": "IN101-N01",
        "backend": backend,
        "git_rev": _git_rev(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cases": {},
    }
    for case in case_list:
        cdir = out / case.case_id
        cdir.mkdir(exist_ok=True)
        rec_path = cdir / "record.json"
        if skip_existing and rec_path.exists():
            index["cases"][case.case_id] = {"status": "cached",
                                           "record": str(rec_path.relative_to(out))}
            log(f"[skip] {case.case_id}")
            continue
        t0 = time.time()
        try:
            rec = run_case(case, backend=backend)
        except Exception as exc:  # keep the campaign going, record honestly
            index["cases"][case.case_id] = {
                "status": "error", "error": f"{type(exc).__name__}: {exc}",
                "wall_seconds": round(time.time() - t0, 2),
                "config": asdict(case.cfg), "tags": dict(case.tags)}
            log(f"[FAIL] {case.case_id}: {exc}")
            continue
        _atomic_json(rec_path, rec)
        index["cases"][case.case_id] = {
            "status": "ok", "record": str(rec_path.relative_to(out)),
            "family": case.family, "tags": dict(case.tags),
            "wall_seconds": rec["wall_seconds"],
            "cf": rec["cf"], "cd_total": rec["cd_total"],
            "cd_form": rec["cd_form"], "cd_skin": rec["cd_skin"],
            "tau_me": rec.get("tau_me"), "tau_flux": rec.get("tau_flux"),
            "tau_skin": rec.get("tau_skin"), "tau_total": rec["tau_total"],
            "tau_form": rec["tau_form"],
            "closure_residual": rec.get("closure_residual"),
            "stationarity_drift": rec.get("stationarity_drift"),
        }
        log(f"[ok] {case.case_id} cf={rec['cf']:.6g} cd={rec['cd_total']:.6g} "
            f"({rec['wall_seconds']:.1f}s)")
    index["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _atomic_json(out / "index.json", index)
    return index


def load_records(run_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Load every ``record.json`` under ``run_dir`` keyed by case id."""
    run = Path(run_dir)
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(run.glob("*/record.json")):
        rec = json.loads(path.read_text())
        out[rec.get("case_id", path.parent.name)] = rec
    return out


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse

    from .cases import keel_sweep, smooth_ladder, turbulent_attempt

    ap = argparse.ArgumentParser(description="IN101-N01 under-ice drag campaign")
    ap.add_argument("--out", default="runs/IN101_N01")
    ap.add_argument("--which", default="all", choices=["all", "smooth", "keel", "turbulent"])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--backend", default="native", choices=["native", "torch"])
    ap.add_argument("--no-skip", action="store_true")
    args = ap.parse_args(argv)

    cases = []
    if args.which in ("all", "smooth"):
        cases += smooth_ladder(threads=args.threads,
                              **({"steps": args.steps, "warmup": args.steps // 4}
                                 if args.steps else {}))
    if args.which in ("all", "keel"):
        cases += keel_sweep(threads=args.threads,
                            **({"steps": args.steps, "warmup": args.steps // 4}
                               if args.steps else {}))
    if args.which in ("all", "turbulent"):
        cases.append(turbulent_attempt(threads=args.threads))
    run_campaign(cases, args.out, backend=args.backend, skip_existing=not args.no_skip)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())