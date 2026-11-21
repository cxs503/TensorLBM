#!/usr/bin/env python3
"""Ref-vs-got benchmark bundle for the Re = 3200 cavity (anchor REPRO-LBM-HOU1995).

Reads the run records in ``out/re3200_n*.json`` (produced by ``run.py``), scores
each of them against the admitted Ghia (1982) Re = 3200 column through the
library scorer (:mod:`tensorlbm.verification.cavity_re3200.scoring`, one
implementation shared with the runner), and writes a self-describing bundle:

``benchmark.json``  machine-readable: per grid, per row ``[coord, ref, got, err%]``,
                    the tolerance, the pass/fail verdict, convergence metadata,
                    the exact ``repro_cmd`` and the sha256 of every input record
``benchmark.csv``   the same rows flattened, for plotting/plot-digitising
``benchmark.md``    a human table with the verdict spelled out

Nothing is typed by hand and nothing is scored silently: a missing record makes
that grid ``WITHHELD``, and a grid that did not converge is flagged
``converged: false`` next to its error so the number cannot be quoted as a
steady-state result by accident.

Usage::

    python benchmarks/verified/cavity/re3200/benchmark.py          # write bundle
    python benchmarks/verified/cavity/re3200/benchmark.py --check   # exit 1 on fail
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.verification.cavity_re3200 import reference as refmod  # noqa: E402
from tensorlbm.verification.cavity_re3200 import scoring  # noqa: E402

OUT = HERE / "out"
GRIDS = (128, 192, 256)


def _load_grids(ref: dict) -> dict:
    grids = {}
    for n in GRIDS:
        path = OUT / f"re3200_n{n}.json"
        if not path.is_file():
            grids[str(n)] = {"status": "WITHHELD", "reason": f"missing {path.name}"}
            continue
        run = json.loads(path.read_text())
        if run.get("run_status") not in ("ok", "not_converged") or not run.get("u_centerline"):
            grids[str(n)] = {"status": run.get("run_status", "VOID"),
                             "reason": run.get("error", "no finite field"),
                             "steps": run.get("steps")}
            continue
        rec = {
            "status": run["run_status"],
            "steps": run["steps"],
            "converged": bool(run.get("converged")),
            "resid_tol": run.get("resid_tol"),
            "final_resid": run.get("final_resid"),
            "tau": run.get("tau"),
            "nu_lattice": run.get("nu_lattice"),
            "elapsed_s": run.get("elapsed_s"),
            "repro_cmd": run.get("repro_cmd"),
            "record": path.name,
            "record_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "u_min": run.get("u_min"), "u_min_y": run.get("u_min_y"),
            "v_max": run.get("v_max"), "v_max_x": run.get("v_max_x"),
            "scores": {conv: scoring.score(run, ref, conv, scoring.TOL_PCT)
                       for conv in scoring.CONVENTIONS},
            "quarantine_sensitivity": scoring.quarantine_sensitivity(run, ref, "wall"),
        }
        grids[str(n)] = rec
    return grids


def _verdict(grids: dict) -> dict:
    scored = {k: v for k, v in grids.items() if "scores" in v}
    if not scored:
        return {"status": "WITHHELD", "reason": "no scored grid"}
    best = max(scored, key=lambda k: int(k))          # finest usable grid rules
    block = scored[best]["scores"]["wall"]
    passed = [k for k, v in scored.items() if v["scores"]["wall"]["pass"]]
    return {
        "status": "PASS" if block["pass"] else "FAIL",
        "finest_grid": int(best),
        "finest_max_abs_err_pct": block["max_abs_err_pct"],
        "tolerance_pct": block["tolerance_pct"],
        "grids_passing": sorted(int(k) for k in passed),
        "grids_scored": sorted(int(k) for k in scored),
        "all_scored_converged": all(v["converged"] for v in scored.values()),
        "note": "verdict is taken on the finest grid in the wall (half-way "
                "bounce-back) coordinate convention; a non-converged grid is "
                "reported with converged=false and its error is a truncation "
                "artefact, not a steady-state result",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 unless the verdict is PASS")
    args = ap.parse_args()

    ref = refmod.load_reference()
    grids = _load_grids(ref)
    verdict = _verdict(grids)
    payload = {
        "case": "cavity_re3200_ghia",
        "anchor": "REPRO-LBM-HOU1995",
        "description": "2-D lid-driven square cavity Re=3200 vs Ghia et al. (1982) Table I/II",
        "reference": {
            "file": Path(ref["_path"]).name,
            "source": ref.get("source"),
            "table": ref.get("table"),
            "content_sha256": ref.get("_sha256_content"),
            "hash_verified": ref.get("_hash_verified"),
            "quarantined": ref.get("provenance", {}).get("quarantined", []),
        },
        "tolerance_pct": scoring.TOL_PCT,
        "verdict": verdict,
        "grid_convergence": scoring.grid_convergence(
            {k: v["scores"]["wall"] for k, v in grids.items() if "scores" in v}),
        "grids": grids,
        "repro_cmd": {
            "run": "python benchmarks/verified/cavity/re3200/run.py --n 128 192 256 "
                   "--steps 150000 --monitor-every 5000 --resid-tol 2e-06",
            "score": "python benchmarks/verified/cavity/re3200/benchmark.py",
            "figures": "python benchmarks/verified/cavity/re3200/analyze.py",
        },
    }
    (HERE / "benchmark.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    rows = []
    for k in sorted(grids, key=int):
        g = grids[k]
        if "scores" not in g:
            rows.append([k, g["status"], "", "", "", "", "", g.get("reason", "")])
            continue
        s = g["scores"]["wall"]
        for kind, block in (("u", s["u_rows"]), ("v", s["v_rows"])):
            for coord, r, got, err in block:
                rows.append([k, kind, f"{coord:.4f}", f"{r:.5f}", f"{got:.5f}",
                             f"{err:+.2f}", "pass" if s["pass"] else "fail",
                             f"steps={g['steps']} converged={g['converged']}"])
    with (HERE / "benchmark.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "profile", "coord", "ref", "got", "err_pct_of_lid",
                    "grid_pass", "notes"])
        w.writerows(rows)

    lines = [
        "# Cavity Re = 3200 -- ref vs got (anchor REPRO-LBM-HOU1995)",
        "",
        f"reference: {payload['reference']['source']} -- {payload['reference']['table']}",
        f"content sha256: `{payload['reference']['content_sha256']}` "
        f"(verified: {payload['reference']['hash_verified']})",
        f"tolerance: {scoring.TOL_PCT:.1f} % of the lid speed, pointwise",
        "",
        f"**verdict: {verdict['status']}** on the finest scored grid "
        f"n={verdict.get('finest_grid')} -> max |err| = "
        f"{verdict.get('finest_max_abs_err_pct', float('nan')):.2f} %",
        "",
        "| n | status | steps | converged | max abs err [%] | rmse [%] | pass |",
        "|---|--------|-------|-----------|-----------------|----------|------|",
    ]
    for k in sorted(grids, key=int):
        g = grids[k]
        if "scores" not in g:
            lines.append(f"| {k} | {g['status']} | - | - | - | - | - |")
            continue
        s = g["scores"]["wall"]
        lines.append(f"| {k} | {g['status']} | {g['steps']} | {g['converged']} | "
                     f"{s['max_abs_err_pct']:.2f} | {s['rmse_pct']:.2f} | "
                     f"{'PASS' if s['pass'] else 'FAIL'} |")
    lines += ["", "## worst rows (finest grid, wall convention)", ""]
    fine = verdict.get("finest_grid")
    if fine is not None:
        s = grids[str(fine)]["scores"]["wall"]
        for kind, block in (("u", s["u_rows"]), ("v", s["v_rows"])):
            worst = max(block, key=lambda r: abs(r[3]))
            lines.append(f"* {kind}: coord {worst[0]:.4f} ref {worst[1]:+.5f} "
                         f"got {worst[2]:+.5f} err {worst[3]:+.2f} %")
    lines += ["", "## how to reproduce", ""]
    for k, v in payload["repro_cmd"].items():
        lines.append(f"* {k}: `{v}`")
    (HERE / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"verdict": verdict, "grids": {
        k: (g.get("scores", {}).get("wall", {}).get("max_abs_err_pct")
            if "scores" in g else g["status"]) for k, g in grids.items()}}, indent=1))
    if args.check and verdict["status"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())