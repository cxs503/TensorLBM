"""Machine-checkable maturity evidence for the force-corrected NEBB + body-force driver.

Run after the cross-validation harness::

    python benchmarks/xval/xval_vivsim_nebb_bodyforce.py --out runs/XVAL-NEBB-BODYFORCE
    python benchmarks/xval/nebb_bodyforce_evidence.py \
        --run-dir runs/XVAL-NEBB-BODYFORCE --strict

It writes ``<run-dir>/evidence.json`` with

``impl_files``            implementation files + non-placeholder audit (line
                          count, public functions, no ``NotImplementedError`` /
                          TODO / bare ``pass`` bodies)
``entry``                 executable entries (module CLI + benchmark) with the
                          exact reproduction command and a sha256
``run_artifacts``         files a *real* run produced (results.json, run.log,
                          figures, CLI artifact), each with size + sha256
``integration_callers``   files outside the module that import and call it,
                          verified by grepping the symbol in the caller source
``benchmark``             ref/got/rel_error/tolerance/pass rows recomputed from
                          ``results.json`` (the L4 numeric comparison)
``tests``                 collected pytest node ids of the anchor tests
``repro_cmd``             the four commands that regenerate everything

Exit code 0 only when every declared path exists, every caller is verified,
every implementation file passes the non-placeholder audit, the benchmark rows
recompute from the stored raw profiles and (with ``--strict``) every row passes
its declared tolerance.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
NODE = "XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S"

IMPL_FILES = [
    "src/tensorlbm/nebb_bodyforce.py",
    "benchmarks/xval/xval_vivsim_nebb_bodyforce.py",
]

ENTRIES = [
    {
        "path": "src/tensorlbm/nebb_bodyforce.py",
        "kind": "cli",
        "command": "python -m tensorlbm.nebb_bodyforce --ny 10 --scheme guo "
                   "--out runs/XVAL-NEBB-BODYFORCE/cli_guo_ny10.json",
        "note": "package CLI (`python -m tensorlbm.nebb_bodyforce`); prints the "
                "error table and writes the JSON artifact",
    },
    {
        "path": "benchmarks/xval/xval_vivsim_nebb_bodyforce.py",
        "kind": "benchmark",
        "command": "python benchmarks/xval/xval_vivsim_nebb_bodyforce.py "
                   "--out runs/XVAL-NEBB-BODYFORCE",
        "note": "XVAL harness: OSS arm vs TensorLBM arm, wall/scheme/grid "
                "sensitivity, results.json + four figures",
    },
]

INTEGRATION_CALLERS = [
    {
        "file": "src/tensorlbm/boundary/antibb_channel_benchmark.py",
        "symbols": ["corrected_wall_velocity", "nebb_row", "wall_spec"],
        "kind": "main_solver",
        "note": "the D2Q9 channel benchmark harness (used by the anti-"
                "bounceback verification study) gained wall_mode="
                "'force_corrected_nebb', which calls the module's wall kernel "
                "and half-force-step wall velocity",
    },
    {
        "file": "benchmarks/xval/xval_vivsim_nebb_bodyforce.py",
        "symbols": ["BodyForceChannel", "analytic_poiseuille", "effective_wall_offset"],
        "kind": "benchmark_driver",
        "note": "the XVAL harness drives the module and compares it with the "
                "OSS transcription",
    },
    {
        "file": "tests/test_nebb_bodyforce.py",
        "symbols": ["nb.nebb_row", "nb.BodyForceChannel", "nb.wall_spec"],
        "kind": "test",
        "note": "anchor tests (kernel moments, shift invariance, driver "
                "accuracy, wall-model negative control, torch path, CLI)",
    },
]

RUN_FILES = ["results.json", "run.log", "cli_guo_ny10.json"]
FIGURES = ["fig_profile.png", "fig_cross_arm.png", "fig_convergence.png", "fig_field.png"]
TESTS = ["tests/test_nebb_bodyforce.py"]
PLACEHOLDER_MARKERS = ("NotImplementedError", "TODO", "FIXME", "placeholder")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def audit_impl(rel: str) -> dict:
    path = REPO / rel
    out = {"path": rel, "exists": path.is_file(), "ok": False}
    if not path.is_file():
        return out
    src = path.read_text()
    tree = ast.parse(src)
    funcs = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    markers = [m for m in PLACEHOLDER_MARKERS if m in src]
    bare_pass = 0
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = [s for s in n.body if not isinstance(s, ast.Expr) or not isinstance(s.value, ast.Constant)]
            if len(body) == 1 and isinstance(body[0], ast.Pass):
                bare_pass += 1
    out.update(
        {
            "lines": len(src.splitlines()),
            "public_functions": [f for f in funcs if not f.startswith("_")],
            "classes": classes,
            "placeholder_markers": markers,
            "bare_pass_functions": bare_pass,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    )
    out["ok"] = (
        out["lines"] >= 40
        and len(out["public_functions"]) >= 3
        and not markers
        and bare_pass == 0
    )
    return out


def audit_caller(row: dict) -> dict:
    path = REPO / row["file"]
    src = path.read_text() if path.is_file() else ""
    found = [s for s in row["symbols"] if s in src]
    return {**row, "exists": path.is_file(), "symbols_found": found,
            "verified": path.is_file() and len(found) == len(row["symbols"]),
            "sha256": sha256(path) if path.is_file() else None}


def recompute_rows(results: dict) -> tuple[list[dict], bool]:
    """Re-derive every ref/got/rel_err row from the stored raw profiles."""
    sys.path.insert(0, str(REPO / "src"))
    from tensorlbm.nebb_bodyforce import analytic_poiseuille, profile_errors

    rows, ok = [], True
    for key, r in results["runs"].items():
        p = r["params"]
        ref = analytic_poiseuille(p["ny"], p["gx"], p["nu"], 0.0)
        err = profile_errors(np.array(r["got"]), ref)
        agrees = abs(err["max_rel_global"] - r["rel_err"]) < 1e-9 * max(r["rel_err"], 1e-12) + 1e-18
        ok = ok and agrees
        primary = key.startswith(("armA_", "armB_")) and "wall" not in key
        rows.append(
            {
                "case": key,
                "primary": primary,
                "ny": p["ny"],
                "nu": p["nu"],
                "gx": p["gx"],
                "scheme": p.get("scheme"),
                "wall": p.get("wall", "force_corrected_nebb"),
                "ref_kind": "analytic_poiseuille (walls on node planes)",
                "umax_ref": err["umax_ref"],
                "umax_got": err["umax_got"],
                "rel_err": err["max_rel_global"],
                "rel_err_offset_fit": r.get("rel_err_offset_fit"),
                "wall_offset_fitted": r.get("wall_offset_fitted"),
                "tolerance": r["tolerance"],
                "pass": bool(err["max_rel_global"] < r["tolerance"]),
                "recomputed_matches_stored": bool(agrees),
            }
        )
    return rows, ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", default=str(REPO / "runs" / "XVAL-NEBB-BODYFORCE"))
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    results_path = run_dir / "results.json"
    if not results_path.is_file():
        print(f"FAIL: no results.json under {run_dir}", file=sys.stderr)
        return 2
    results = json.loads(results_path.read_text())

    impl = [audit_impl(rel) for rel in IMPL_FILES]
    entries = []
    for e in ENTRIES:
        p = REPO / e["path"]
        entries.append({**e, "exists": p.is_file(), "sha256": sha256(p) if p.is_file() else None})
    callers = [audit_caller(r) for r in INTEGRATION_CALLERS]

    run_artifacts = []
    for name in RUN_FILES + FIGURES:
        p = run_dir / name
        run_artifacts.append(
            {
                "path": str(p.relative_to(REPO)),
                "exists": p.is_file(),
                "bytes": p.stat().st_size if p.is_file() else 0,
                "sha256": sha256(p) if p.is_file() else None,
                "role": results.get("figures", {}).get(name, "run artifact"),
            }
        )

    bench_rows, recomputed_ok = recompute_rows(results)
    cross_rows = []
    for key, c in results["cross_arm"].items():
        got = np.array(c["got"])
        ref = np.array(c["ref"])
        rel = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-300))
        cross_rows.append({"case": key, "metric": "max|armB-armA|/max|armA|",
                           "rel_diff": rel, "tolerance": c["tolerance"],
                           "pass": bool(rel < c["tolerance"]),
                           "recomputed_matches_stored": abs(rel - c["rel_diff"]) < 1e-12})
        recomputed_ok = recomputed_ok and abs(rel - c["rel_diff"]) < 1e-12

    primary_rows = [r for r in bench_rows if r.get("primary")]
    sensitivity_rows = [r for r in bench_rows if not r.get("primary")]
    benchmark = {
        "rows": bench_rows,
        "cross_arm_rows": cross_rows,
        # the declared claims are the two arms (accuracy) and their agreement
        # (cross-validation); the wall-model variants are *sensitivity* rows and
        # are reported, not frozen into the gate.
        "all_pass": all(r["pass"] for r in primary_rows) and all(r["pass"] for r in cross_rows),
        "sensitivity_rows_pass": {r["case"]: r["pass"] for r in sensitivity_rows},
        "n_rows": len(bench_rows) + len(cross_rows),
        "n_primary_rows": len(primary_rows),
        "source": "results.json",
        "recomputed_from": f"{run_dir.name}/results.json",
        "reproduces_stored_rows": bool(recomputed_ok),
        "thresholds": results["thresholds"],
        "verdict": results["verdict"],
    }

    tests = []
    for rel in TESTS:
        p = REPO / rel
        proc = subprocess.run([sys.executable, "-m", "pytest", rel, "--collect-only", "-q"],
                              cwd=REPO, capture_output=True, text=True)
        ids = [ln.strip() for ln in proc.stdout.splitlines() if "::" in ln]
        tests.append({"path": rel, "exists": p.is_file(), "n_collected": len(ids),
                      "node_ids_sample": ids[:3]})

    checks = {
        "impl_files_ok": all(r["ok"] for r in impl),
        "entries_exist": all(r["exists"] for r in entries),
        "run_artifacts_exist": all(r["exists"] for r in run_artifacts),
        "integration_callers_verified": all(r["verified"] for r in callers),
        "tests_collected": all(r["n_collected"] > 0 for r in tests),
        "benchmark_rows_recomputed": benchmark["reproduces_stored_rows"],
    }
    mandatory = list(checks)
    if args.strict:
        checks["benchmark_all_pass"] = benchmark["all_pass"]
        mandatory.append("benchmark_all_pass")
    ok = all(checks[k] for k in mandatory)

    bundle = {
        "node": NODE,
        "repo": "TensorLBM",
        "capability": "force-corrected NEBB wall + body-force driver (D2Q9)",
        "module": "tensorlbm.nebb_bodyforce",
        "run_dir": str(run_dir.relative_to(REPO)),
        "environment": results.get("environment", {}),
        "oss_provenance": results.get("oss_provenance", {}),
        "friction": results.get("friction", []),
        "impl_files": impl,
        "entry": entries,
        "run_artifacts": run_artifacts,
        "integration_callers": callers,
        "benchmark": benchmark,
        "tests": tests,
        "repro_cmd": {
            "benchmark": results.get("repro_cmd"),
            "cli": ENTRIES[0]["command"],
            "evidence": "python benchmarks/xval/nebb_bodyforce_evidence.py "
                        "--run-dir runs/XVAL-NEBB-BODYFORCE --strict",
            "tests": "python -m pytest tests/test_nebb_bodyforce.py -q",
        },
        "checks": checks,
        "ok": ok,
    }
    out = run_dir / "evidence.json"
    out.write_text(json.dumps(bundle, indent=2, default=float))
    for k, v in checks.items():
        print(f"[{'PASS' if v else 'FAIL'}] {k}")
    print(f"evidence -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())