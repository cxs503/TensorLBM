#!/usr/bin/env python3
"""Independent board audit of mission m18129
(anchor XVAL-LBM-PER-MARGIN-ARRAYS-POINTER-SHIFT-STREAMIN, TensorLBM).

What this script does — and does *not* trust
--------------------------------------------
It does not trust the shipped ``runs/xval_tclb_pointer_shift/result.json`` and it
does not import the delivered package in place.  Everything is re-derived from
raw behaviour in a throw-away stage:

  A1 stage          assemble a clean tree: this worktree's ``src/tensorlbm`` plus
                    the *frozen* mission module (evidence_snapshot), the frozen
                    benchmark driver, the frozen test file.  Nothing from the
                    mission work tree is executed in place.
  A2 provenance     re-hash every OSS excerpt in ``oss_ref/REF.json`` against the
                    pinned clone ``/tmp/TCLB`` (zero-hand-transcription check) and
                    check the pinned revision against ``.git/refs/heads/master``.
  A3 tests          run the shipped pytest file (does it assert invariants, or
                    merely execute code?).
  A4 mutation       sabotage the plan in three ways (drop a wrapped block, perturb
                    a block's flat_offset, de-periodise the wrap) and check whether
                    the invariant checker *and* the pytest suite notice.  This is
                    the detection-power probe: a test suite that cannot fail
                    proves nothing.
  A5 fresh run      re-execute the shipped driver end-to-end in the stage and diff
                    its headline numbers against the shipped result.json.
  A6 cost probe     independent timing of the two transport arms (warm-up, more
                    reps, interleaved) to test whether the shipped "17x slower"
                    cost claim is a measurement artefact.
  A7 referee        the machine referee row set for the anchor is resolved outside
                    this script (psql) and injected through --referee-json.

Run:  python docs/audit_m18222/audit_run.py [--quick] [--referee-json ref.json]
Writes ``docs/audit_m18222/audit_summary.json`` (+ raw per-probe JSON).
Exit code is always 0: this is evidence collection, the verdict lives in the
summary / AUDIT_m18222.md / the ``tensor_maturity_verdicts`` row.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SNAP = HERE / "evidence_snapshot"
STAGE = Path("/tmp/audit_m18222/stage")
TCLB = Path("/tmp/TCLB")
ANCHOR = "XVAL-LBM-PER-MARGIN-ARRAYS-POINTER-SHIFT-STREAMIN"
ORIG_MISSION = 18129


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------- A1 stage
def stage() -> Path:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "src").mkdir(parents=True)
    shutil.copytree(REPO / "src" / "tensorlbm", STAGE / "src" / "tensorlbm")
    xval = STAGE / "src" / "tensorlbm" / "xval"
    xval.mkdir(exist_ok=True)
    if not (xval / "__init__.py").exists():
        (xval / "__init__.py").write_text('"""Cross-validation modules."""\n')
    shutil.copy(SNAP / "per_margin_pointer_shift.py", xval / "per_margin_pointer_shift.py")
    shutil.copytree(SNAP / "xval_tclb_pointer_shift", STAGE / "benchmarks" / "xval_tclb_pointer_shift")
    (STAGE / "tests").mkdir()
    shutil.copy(SNAP / "test_xval_tclb_pointer_shift.py", STAGE / "tests" / "test_xval_tclb_pointer_shift.py")
    return STAGE


def staged_import():
    sys.path.insert(0, str(STAGE / "src"))
    from tensorlbm import solver, solver3d  # noqa: E402
    from tensorlbm.d2q9 import C as C2D  # noqa: E402
    from tensorlbm.d3q19 import C as C3D  # noqa: E402
    from tensorlbm.xval import per_margin_pointer_shift as PMPS  # noqa: E402

    return solver, solver3d, C2D, C3D, PMPS


# ---------------------------------------------------------------- A2 provenance
def provenance() -> dict:
    ref = json.loads((SNAP / "xval_tclb_pointer_shift" / "oss_ref" / "REF.json").read_text())
    out = {
        "ref_json_meta": {k: ref[k] for k in ref if k not in ("files", "snippets")},
        "files_missing": [],
        "file_sha_mismatch": [],
        "file_sha_ok": 0,
        "snippet_checked": 0,
        "snippet_sha_mismatch": [],
        "snippet_text_mismatch": [],
        "tclb_present": TCLB.is_dir(),
    }
    if not TCLB.is_dir():
        return out
    for rel, meta in ref["files"].items():
        p = TCLB / rel
        if not p.is_file():
            out["files_missing"].append(rel)
            continue
        # .git metadata files are content of the clone itself
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h == meta["sha256"]:
            out["file_sha_ok"] += 1
        else:
            out["file_sha_mismatch"].append({"file": rel, "expected": meta["sha256"], "got": h})
    mask = TCLB / ".git" / "refs" / "heads" / "master"
    out["pinned_rev_matches_clone"] = bool(
        mask.is_file() and mask.read_text().strip() == "ded67cd768cf7e727bd078af139e3ec7895076e5"
    )
    # Zero-hand-transcription check: the delivered excerpt body must occur
    # *verbatim* in the pinned clone, starting on the recorded line, and the
    # body must hash to the recorded excerpt_sha256.  (The extractor cuts at the
    # end-regex position, not at a line end — so the body is located by search,
    # not re-sliced by line numbers.)
    for sn in ref["snippets"]:
        key = sn["key"]
        delivered = SNAP / "xval_tclb_pointer_shift" / "oss_ref" / f"{key}.txt"
        out["snippet_checked"] += 1
        if not delivered.is_file():
            out.setdefault("excerpt_file_missing", []).append(key)
            continue
        raw_lines = delivered.read_text().splitlines(keepends=True)
        body = "".join(ln for ln in raw_lines if not ln.startswith("# "))
        rec = {
            "key": key,
            "loc": f"{sn['rel_path']}:{sn['start_line']}-{sn['end_line']}",
            "sha_matches_record": hashlib.sha256(body.encode()).hexdigest() == sn["excerpt_sha256"],
            "found_in_clone": False,
        }
        p = TCLB / sn["rel_path"]
        if p.is_file():
            text = p.read_text(errors="replace")
            pos = text.find(body)
            if pos < 0:
                pos = text.find(body.strip())
            if pos >= 0:
                rec["found_in_clone"] = True
                rec["found_line"] = text.count("\n", 0, pos) + 1
                rec["line_matches_record"] = rec["found_line"] == sn["start_line"]
                rec["n_lines_in_body"] = body.count("\n") + (0 if body.endswith("\n") else 1)
        out.setdefault("snippets", []).append(rec)
        if not rec["sha_matches_record"]:
            out["snippet_sha_mismatch"].append(rec)
        if rec.get("found_in_clone") and not rec.get("line_matches_record"):
            out["snippet_text_mismatch"].append(rec)
    # can the OSS side actually be executed here?
    out["R_available"] = bool(shutil.which("Rscript") or shutil.which("R"))
    out["nvcc_available"] = bool(shutil.which("nvcc"))
    return out


# --------------------------------------------------------------------- A3 tests
def run_tests() -> dict:
    env = dict(os.environ, PYTHONPATH=str(STAGE / "src"), OMP_NUM_THREADS="8")
    t0 = time.perf_counter()
    cp = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_xval_tclb_pointer_shift.py", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=STAGE, env=env, capture_output=True, text=True, timeout=3600,
    )
    tail = (cp.stdout + cp.stderr).strip().splitlines()
    return {
        "returncode": cp.returncode,
        "wall_s": round(time.perf_counter() - t0, 2),
        "tail": tail[-6:],
        "n_tests": sum(1 for ln in tail if ln.startswith("tests/") and "%" in ln) or None,
    }


# ------------------------------------------------------------------ A4 mutation
def mutation() -> dict:
    solver, solver3d, C2D, C3D, PMPS = staged_import()
    import torch

    torch.manual_seed(0)
    shape = (9, 11)
    f = torch.rand(9, *shape, dtype=torch.float64)
    plan = PMPS.pointer_shift_plan(C2D, shape)
    base = solver.stream(f)

    def probe(name, mutant_plan, expect_gather_equal=False):
        inv = PMPS.check_plan_invariant(f, C2D, plan=mutant_plan)
        got = None
        err = None
        try:
            got = PMPS.stream_pointer_shift(f, C2D, plan=mutant_plan)
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
        diff = float((got - base).abs().max()) if got is not None else None
        return {
            "mutant": name,
            "partition_ok": inv["partition_ok"],
            "holes": inv["holes"],
            "overlaps": inv["overlaps"],
            "max_offset_error": inv["max_offset_error"],
            "max_abs_diff_vs_gather": inv["max_abs_diff_vs_gather"],
            "stream_vs_production_max_abs_diff": diff,
            "stream_raised": err,
            "detected_by_invariant": not (inv["partition_ok"] and inv["max_offset_error"] == 0),
            "detected_by_bitwise_test": (diff is None) or diff > 0.0,
        }

    results = [{
        "mutant": "M0 control (unmodified plan)",
        "partition_ok": PMPS.check_plan_invariant(f, C2D, plan=plan)["partition_ok"],
        "max_offset_error": PMPS.check_plan_invariant(f, C2D, plan=plan)["max_offset_error"],
        "stream_vs_production_max_abs_diff": float((PMPS.stream_pointer_shift(f, C2D, plan=plan) - base).abs().max()),
        "detected_by_invariant": False,
        "detected_by_bitwise_test": False,
    }]

    # M1: drop the wrapped strip of every direction that wraps along x (holes in dst)
    copies = tuple(
        cp for cp in plan.copies
        if not (cp.dst[-1].start == 0 and cp.dst[-1].stop == 1 and cp.flat_offset > 0)
    )
    results.append(probe("M1 drop the wrapped x-strip blocks", dataclasses.replace(plan, copies=copies)))

    # M2: perturb one block's recorded flat_offset (the property the report banks)
    bad = []
    for i, cp in enumerate(plan.copies):
        bad.append(dataclasses.replace(cp, flat_offset=cp.flat_offset + 1) if i == 3 else cp)
    results.append(probe("M2 flat_offset +1 on block #3", dataclasses.replace(plan, copies=tuple(bad))))

    # M3: de-periodise (non-wrapping) transport => wrong physics at the border
    deper = []
    for cp in plan.copies:
        n = plan.shape
        if cp.dst[-1].start == 0 and cp.dst[-1].stop == 1:  # wrapped strip -> copy from itself
            deper.append(dataclasses.replace(cp, src=cp.dst, flat_offset=0))
        else:
            deper.append(cp)
    results.append(probe("M3 de-periodise the wrap (copy src onto itself)", dataclasses.replace(plan, copies=tuple(deper))))

    # 3D control so the probe is not 2D-only
    f3 = torch.rand(19, 6, 7, 8, dtype=torch.float64)
    p3 = PMPS.pointer_shift_plan(C3D, (6, 7, 8))
    inv3 = PMPS.check_plan_invariant(f3, C3D, plan=p3)
    eq3 = float((solver3d.stream3d_gather(f3) - PMPS.stream_pointer_shift(f3, C3D, plan=p3)).abs().max())
    return {
        "mutants": results,
        "detection_power": {
            "n_mutants_detected": sum(1 for r in results[1:] if r["detected_by_invariant"] or r["detected_by_bitwise_test"]),
            "n_mutants": len(results) - 1,
            "control_clean": results[0]["stream_vs_production_max_abs_diff"] == 0.0,
        },
        "control_3d": {"partition_ok": inv3["partition_ok"], "max_offset_error": inv3["max_offset_error"], "gather_max_abs_diff": eq3},
    }


# ---------------------------------------------------------------- A5 fresh run
def fresh_run(quick: bool) -> dict:
    env = dict(os.environ, PYTHONPATH=str(STAGE / "src"), OMP_NUM_THREADS="128")
    t0 = time.perf_counter()
    cp = subprocess.run(
        [sys.executable, "benchmarks/xval_tclb_pointer_shift/run_xval.py"] + (["--quick"] if quick else []),
        cwd=STAGE, env=env, capture_output=True, text=True, timeout=7200,
    )
    wall = round(time.perf_counter() - t0, 1)
    out = {"returncode": cp.returncode, "wall_s": wall, "stdout_tail": cp.stdout.strip().splitlines()[-8:],
           "stderr_tail": cp.stderr.strip().splitlines()[-5:]}
    if cp.returncode != 0:
        return out
    fresh = json.loads((STAGE / "runs" / "xval_tclb_pointer_shift" / "result.json").read_text())
    shipped = json.loads((SNAP / "result.json").read_text())
    out["fresh_decision"] = fresh["decision"]
    out["shipped_decision"] = shipped["decision"]
    out["decision_identical"] = fresh["decision"] == shipped["decision"]

    def flat_phys(res):
        rows = {}
        for r in res["physics"]["tgv"] + res["physics"]["tau_sensitivity"]:
            rows[f"tgv_{r['arm']}_n{r['n']}_tau{r['tau']}_s{r['steps']}"] = r["final_error"]
        for r in res["physics"]["poiseuille"]:
            rows[f"pois_{r['arm']}_umax_err"] = r["umax_rel_err"]
            rows[f"pois_{r['arm']}_rel_l2"] = r["rel_l2_profile"]
        return rows

    fp, sp = flat_phys(fresh), flat_phys(shipped)
    cmp_rows = {}
    for k in sorted(set(fp) & set(sp)):
        cmp_rows[k] = {"shipped": sp[k], "auditor_rerun": fp[k],
                       "abs_delta": abs(fp[k] - sp[k]),
                       "rel_delta": abs(fp[k] - sp[k]) / abs(sp[k]) if sp[k] else None}
    out["physics_compare"] = cmp_rows
    out["physics_max_rel_delta"] = max((r["rel_delta"] or 0.0) for r in cmp_rows.values()) if cmp_rows else None
    out["key_missing_on_rerun"] = sorted(set(sp) - set(fp))
    out["equiv_identical"] = fresh["equivalence"] == shipped["equivalence"]
    out["all_bitwise_equal"] = fresh["equivalence"]["all_bitwise_equal"]
    out["invariants_ok"] = fresh["equivalence"]["invariants_ok"]
    return out


# ----------------------------------------------------------------- A6 cost probe
def cost_probe(reps: int = 9) -> dict:
    solver, solver3d, C2D, C3D, PMPS = staged_import()
    import torch

    torch.set_num_threads(128)
    gen = torch.Generator().manual_seed(11)
    out = {"reps": reps, "rows": {}}
    for n in (128, 256):
        f = torch.rand(9, n, n, generator=gen, dtype=torch.float32)
        plan = PMPS.pointer_shift_plan(C2D, (n, n))
        for _ in range(3):  # warm-up
            solver.stream(f)
            PMPS.stream_pointer_shift(f, C2D, plan=plan)
            PMPS.stream_pointer_shift_flat(f, C2D, plan=plan)
        tb, to, tf = [], [], []
        for _ in range(reps):
            t = time.perf_counter(); solver.stream(f); tb.append(time.perf_counter() - t)
            t = time.perf_counter(); PMPS.stream_pointer_shift(f, C2D, plan=plan); to.append(time.perf_counter() - t)
            t = time.perf_counter(); PMPS.stream_pointer_shift_flat(f, C2D, plan=plan); tf.append(time.perf_counter() - t)
        b, o = statistics.median(tb), statistics.median(to)
        out["rows"][f"2d/{n}"] = {
            "baseline_s": b, "oss_s": o, "oss_flat_s": statistics.median(tf),
            "slowdown": o / b, "slowdown_min_of_reps": min(to) / min(tb),
            "baseline_spread": [min(tb), max(tb)], "oss_spread": [min(to), max(to)],
        }
    for n in (32, 48):
        f = torch.rand(19, n, n, n, generator=gen, dtype=torch.float32)
        plan = PMPS.pointer_shift_plan(C3D, (n, n, n))
        for _ in range(2):
            solver3d.stream3d_gather(f)
            PMPS.stream_pointer_shift(f, C3D, plan=plan)
        tb, to = [], []
        for _ in range(reps):
            t = time.perf_counter(); solver3d.stream3d_gather(f); tb.append(time.perf_counter() - t)
            t = time.perf_counter(); PMPS.stream_pointer_shift(f, C3D, plan=plan); to.append(time.perf_counter() - t)
        b, o = statistics.median(tb), statistics.median(to)
        out["rows"][f"3d/{n}"] = {
            "baseline_s": b, "oss_s": o, "slowdown": o / b,
            "slowdown_min_of_reps": min(to) / min(tb),
            "baseline_spread": [min(tb), max(tb)], "oss_spread": [min(to), max(to)],
        }
    out["max_slowdown"] = max(r["slowdown"] for r in out["rows"].values())
    out["max_slowdown_best_case"] = max(r["slowdown_min_of_reps"] for r in out["rows"].values())
    return out


# ---------------------------------------------------------------- A7 referees
def referee(path: str | None) -> dict:
    if not path:
        return {"provided": False}
    rows = json.loads(Path(path).read_text())
    return {"provided": True, "rows": rows, "n_rows": len(rows) if isinstance(rows, list) else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="driver in quick mode (A5)")
    ap.add_argument("--referee-json", default=None)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    log(f"[stage] {stage()}")
    summary: dict = {"anchor": ANCHOR, "mission_audited": ORIG_MISSION,
                     "auditor_mission": 18222, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    want = args.only or "all"

    def dump(key, obj):
        (HERE / f"{key}.json").write_text(json.dumps(obj, indent=1, default=float) + "\n")
        summary[key] = obj

    if want in ("all", "provenance"):
        log("[A2] provenance …"); dump("provenance", provenance())
    if want in ("all", "tests"):
        log("[A3] pytest …"); dump("tests", run_tests()); log(f"     rc={summary['tests']['returncode']}")
    if want in ("all", "mutation"):
        log("[A4] mutation / detection power …"); dump("mutation", mutation())
        log(f"     detected {summary['mutation']['detection_power']['n_mutants_detected']}"
            f"/{summary['mutation']['detection_power']['n_mutants']}")
    if want in ("all", "cost"):
        log("[A6] cost probe …"); dump("cost", cost_probe())
        log(f"     max slowdown {summary['cost']['max_slowdown']:.2f}x")
    if want in ("all", "fresh"):
        log("[A5] fresh driver run …"); dump("fresh", fresh_run(args.quick))
        log(f"     rc={summary['fresh']['returncode']} wall={summary['fresh'].get('wall_s')}s")
    if want in ("all", "referee"):
        dump("referee", referee(args.referee_json))
    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    prev = HERE / "audit_summary.json"
    if prev.is_file() and want != "all":
        old = json.loads(prev.read_text())
        old.update({k: v for k, v in summary.items() if k not in ("started",)})
        summary = old
    prev.write_text(json.dumps(summary, indent=1, default=float) + "\n")
    log(f"[done] {prev}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())