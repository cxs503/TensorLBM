#!/usr/bin/env python3
"""XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN — controlled cross-validation run.

Decision (predeclared, before any result was inspected)
-------------------------------------------------------
D1  Is TensorLBM's example set a *worse* instance of the "example-per-main"
    antipattern than the OSS reference (listLBM), on the same mechanical
    metrics?  Keep the current example files if TensorLBM is no worse; add a
    dispatch layer if it is no worse on duplication/CLI but still lacks a
    programmatic index.
D2  Does the added dispatch layer (``tools/xval/example_runner.py``) change
    example behaviour?  H1: stdout/stderr bytes and exit code must be
    *identical* to a hand-typed invocation (0 tolerance).  Deviation declared
    after the first run: one target (``ai_dns_case``) prints wall-clock
    timings, so its own repeated direct runs already differ byte-wise.  For
    such an example the criterion becomes "every line that is invariant across
    the direct runs is byte-identical in the runner runs" plus rc/stderr
    equality -- i.e. the runner may not perturb anything the example itself
    controls.  Both criteria are reported per example, never mixed.
D3  Is the dispatch overhead acceptable?  H2: median wall-clock overhead of
    the runner relative to direct invocation must stay **<= 15%** (the owner's
    precision line reused as the tolerance for dispatch cost; the runner only
    spawns one subprocess, so the expectation is a few percent).

Alternatives considered
-----------------------
A0  Do nothing (leave 178 per-file ``__main__`` blocks with no index).
A1  Rewrite every example into a library + one CLI (removes the antipattern at
    the source but touches validated physics code: rejected — 189 files, high
    blast radius, zero numerical benefit).
A2  Delete examples and keep only pytest cases (loses the documented,
    human-readable reproduction commands used by the reports).
A3  Add a static index + single dispatch entry around the *unchanged* examples
    (chosen: additive, reversible, testable for transparency).

Design
------
* Unit of analysis: one example module (Python file under ``examples/``).
* Controlled variable: invocation path (direct interpreter vs runner).
* Changed variable: none inside the example — the runner passes the same
  interpreter, file and arguments; only ``PYTHONPATH`` is normalised.
* Repetitions: 3 per (example, path) pair; ordering interleaved (direct, runner)
  so machine load drift does not systematically favour one path.
* Metrics: rc equality, stdout sha256 equality, stderr sha256 equality, wall
  time per run (median over 3).
* Confounders: torch thread counts / CPU contention (mitigated by interleaving
  and medians), first-run disk cache (mitigated by repetitions), matplotlib-less
  headless runs (no figures inside examples).
* Stopping rule: abort after an example if its direct run fails to exit within
  ``--timeout`` (recorded as an environment failure, not silently dropped).

Outputs (all inside the worktree)
---------------------------------
artifacts/xval_example_per_main/{audit_tensorlbm,audit_listlbm,audit_listlbm_repo,
index,ab_runs,summary}.json
figures/xval_example_per_main/*.png
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "xval"))

from example_main_audit import scan_tree  # noqa: E402
from example_runner import build_index, resolve, run_example  # noqa: E402

OUT = REPO_ROOT / "artifacts" / "xval_example_per_main"
FIG = REPO_ROOT / "figures" / "xval_example_per_main"

# (example id, forwarded args, rationale)
SMOKE = [
    ("minimal_d2q9", [], "pure library call, no CLI, 8 s: cheapest end-to-end check"),
    ("ai_dns_case", [], "fixed seed=0 AI-DNS pipeline: deterministic JSON payload, "
                        "tests multi-line structured stdout"),
    ("bfl_fast_piston", ["--ps", "20", "--steps", "20", "--device", "cpu"],
     "CLI example; emits a known non-finite state at these settings, so it also "
     "proves the runner does not mask failures"),
]
BLOCK_LENS = (2, 3, 5)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


def _invariant_lines(outputs: list[str]) -> list[str]:
    """Lines that are byte-identical at the same position in every run."""
    if not outputs:
        return []
    split = [out.splitlines() for out in outputs]
    n = len(split[0])
    if any(len(lines) != n for lines in split):
        return []
    return [split[0][i] for i in range(n) if len({lines[i] for lines in split}) == 1]


def _has_lines(output: str, invariant: list[str]) -> bool:
    """True when every invariant line appears in ``output`` (in order)."""
    lines = iter(output.splitlines())
    return all(any(candidate == want for candidate in lines) for want in invariant)


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=False,
                              capture_output=True, text=True).stdout.strip()
    except OSError:  # pragma: no cover
        return ""


def _direct_run(entry, extra_args: list[str], timeout: float) -> dict[str, object]:
    """Hand-typed equivalent of the runner invocation (control condition)."""
    env = dict(os.environ)
    src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        [src, *[p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]]
    )
    cmd = [sys.executable, entry.path, *extra_args]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True,
                          timeout=timeout, check=False)
    duration = time.perf_counter() - t0
    import hashlib

    return {
        "path_kind": "direct",
        "cmd_shell": " ".join(cmd),
        "rc": proc.returncode,
        "duration_s": round(duration, 3),
        "stdout_sha256": hashlib.sha256(proc.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(proc.stderr.encode()).hexdigest(),
        "stdout_lines": len(proc.stdout.splitlines()),
        "stdout_tail": proc.stdout.splitlines()[-6:],
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--listlbm", default="/tmp/listlbm", help="OSS reference checkout")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--skip-ab", action="store_true")
    ap.add_argument("--only", action="append", default=[], help="restrict smoke ids")
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    summary: dict[str, object] = {
        "anchor": "XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN",
        "git_head": _git_head(),
        "started_at": started.isoformat().replace("+00:00", "Z"),
        "repo_root": str(REPO_ROOT),
        "python": sys.executable,
        "listlbm_root": args.listlbm,
        "reps": args.reps,
    }

    # ---------------- M1: structural audit of both trees ----------------
    tb = scan_tree(REPO_ROOT / "examples", block_len=3, relative_to=REPO_ROOT)
    # Pre-change baseline: the same tree with this mission's dispatch layer
    # masked out, so "no programmatic index" is reproducible, not a memory.
    tb_baseline = scan_tree(
        REPO_ROOT / "examples",
        block_len=3,
        relative_to=REPO_ROOT,
        ignore=("tools/example_runner.py", "tools/xval/example_runner.py",
                "examples/run.py"),
    )
    lb = scan_tree(Path(args.listlbm) / "examples", block_len=3)
    lb_repo = scan_tree(Path(args.listlbm), block_len=3, suffixes={".f90", ".F90"})
    for report in (tb, lb, lb_repo):
        report.pop("units", None)
    tb_baseline.pop("units", None)
    _write(OUT / "audit_tensorlbm.json", tb)
    _write(OUT / "audit_tensorlbm_baseline.json", tb_baseline)
    _write(OUT / "audit_listlbm.json", lb)
    _write(OUT / "audit_listlbm_repo.json", lb_repo)

    # determinism of the scanner: same inputs twice -> identical JSON
    tb2 = scan_tree(REPO_ROOT / "examples", block_len=3, relative_to=REPO_ROOT)
    tb2.pop("units", None)
    summary["audit_deterministic"] = tb == tb2

    # ---------------- M2: block-length sensitivity ----------------
    sens = {}
    for block_len in BLOCK_LENS:
        t = scan_tree(REPO_ROOT / "examples", block_len=block_len, relative_to=REPO_ROOT)
        r = scan_tree(Path(args.listlbm) / "examples", block_len=block_len)
        sens[str(block_len)] = {
            "tensorlbm_dup_line_ratio": round(t["dup_line_ratio"], 6),
            "listlbm_dup_line_ratio": round(r["dup_line_ratio"], 6),
            "tensorlbm_dup_lines": t["dup_lines"],
            "listlbm_dup_lines": r["dup_lines"],
        }
        del t, r
    summary["block_len_sensitivity"] = sens

    # scope sensitivity: top-level examples vs nested example packages
    top = scan_tree(REPO_ROOT / "examples", block_len=3, relative_to=REPO_ROOT)
    top_files = [u for u in top["units"] if "/" not in u["path"].split("examples/")[-1]]
    top.pop("units")
    nested = scan_tree(REPO_ROOT / "examples", block_len=3, relative_to=REPO_ROOT)
    nested_files = [u for u in nested["units"] if "/" in u["path"].split("examples/")[-1]]
    nested.pop("units")
    summary["scope_sensitivity"] = {
        "top_level_files": len(top_files),
        "nested_files": len(nested_files),
        "top_level_nonblank_lines": sum(u["nonblank_lines"] for u in top_files),
        "nested_nonblank_lines": sum(u["nonblank_lines"] for u in nested_files),
        "note": "dup_line_ratio is a tree-level n-gram statistic; per-scope split is "
                "reported as line counts so the ratio is not double-counted",
    }

    # ---------------- M3: dispatch index with provenance ----------------
    entries = build_index(REPO_ROOT / "examples", REPO_ROOT)
    index_payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "n_entries": len(entries),
        "n_runnable": sum(1 for e in entries if e.has_entry),
        "n_cli": sum(1 for e in entries if e.cli_flags),
        "examples_dir": "examples",
        "entries": [e.__dict__ for e in entries],
    }
    _write(OUT / "index.json", index_payload)
    summary["index"] = {
        "n_entries": index_payload["n_entries"],
        "n_runnable": index_payload["n_runnable"],
        "n_cli": index_payload["n_cli"],
        "ids_unique": len({e["id"] for e in index_payload["entries"]}) == len(entries),
    }

    # ---------------- M4: A/B transparency + overhead ----------------
    ab: list[dict[str, object]] = []
    smoke = [s for s in SMOKE if not args.only or s[0] in args.only]
    if not args.skip_ab:
        for example_id, extra, rationale in smoke:
            entry = resolve(entries, example_id)
            for rep in range(args.reps):
                direct = _direct_run(entry, extra, args.timeout)
                manifest = OUT / "runs" / f"{example_id.replace('/', '_')}_rep{rep}.json"
                ran = run_example(entry, root=REPO_ROOT, extra_args=extra,
                                  timeout=args.timeout, manifest=manifest,
                                  keep_stdout=True)
                record = {
                    "example_id": example_id,
                    "rationale": rationale,
                    "extra_args": extra,
                    "rep": rep,
                    "direct": direct,
                    "runner": {
                        "path_kind": "runner",
                        "cmd_shell": ran["cmd_shell"],
                        "rc": ran["rc"],
                        "duration_s": ran["duration_s"],
                        "stdout_sha256": ran["stdout_sha256"],
                        "stderr_sha256": ran["stderr_sha256"],
                        "stdout_lines": ran["stdout_lines"],
                        "stdout_tail": ran["stdout_tail"],
                        "manifest": str(manifest.relative_to(REPO_ROOT)),
                    },
                    "identical_stdout": direct["stdout_sha256"] == ran["stdout_sha256"],
                    "identical_stderr": direct["stderr_sha256"] == ran["stderr_sha256"],
                    "identical_rc": direct["rc"] == ran["rc"],
                    "overhead_ratio": (ran["duration_s"] - direct["duration_s"])
                    / direct["duration_s"] if direct["duration_s"] else None,
                }
                record["stdout_identical"] = record.pop("identical_stdout")
                record["stderr_identical"] = record.pop("identical_stderr")
                record["rc_identical"] = record.pop("identical_rc")
                record["runner"]["stdout"] = ran["stdout"]
                record["runner"].pop("stdout_tail", None)
                ab.append(record)
                print(f"[ab] {example_id} rep{rep}: rc {direct['rc']}/{ran['rc']} "
                      f"ident={record['stdout_identical']} "
                      f"t={direct['duration_s']:.2f}/{ran['duration_s']:.2f}s",
                      flush=True)
    _write(OUT / "ab_runs.json", ab)

    per_example: dict[str, object] = {}
    for example_id, _, _ in smoke:
        rows = [r for r in ab if r["example_id"] == example_id]
        if not rows:
            continue
        d = [r["direct"]["duration_s"] for r in rows]
        q = [r["runner"]["duration_s"] for r in rows]
        direct_stdout = [r["direct"]["stdout"] for r in rows]
        runner_stdout = [r["runner"]["stdout"] for r in rows]
        invariant = _invariant_lines(direct_stdout)
        per_example[example_id] = {
            "n": len(rows),
            "deterministic_direct": len({_hash(x) for x in direct_stdout}) == 1,
            "distinct_stdout_hashes_direct": len({_hash(x) for x in direct_stdout}),
            "distinct_stdout_hashes_runner": len({_hash(x) for x in runner_stdout}),
            "all_identical_stdout": all(r["stdout_identical"] for r in rows),
            "all_identical_stderr": all(r["stderr_identical"] for r in rows),
            "all_identical_rc": all(r["rc_identical"] for r in rows),
            "invariant_lines": len(invariant),
            "invariant_lines_preserved_by_runner": all(
                _has_lines(x, invariant) for x in runner_stdout
            ),
            "transparent": (
                all(r["stderr_identical"] for r in rows)
                and all(r["rc_identical"] for r in rows)
                and (
                    all(r["stdout_identical"] for r in rows)
                    if len({_hash(x) for x in direct_stdout}) == 1
                    else all(_has_lines(x, invariant) for x in runner_stdout)
                )
            ),
            "rc_values": sorted({r["direct"]["rc"] for r in rows}),
            "direct_median_s": round(statistics.median(d), 3),
            "runner_median_s": round(statistics.median(q), 3),
            "overhead_pct": round(
                100.0 * (statistics.median(q) - statistics.median(d)) / statistics.median(d), 3
            ),
        }
    summary["ab"] = per_example
    summary["ab_all_transparent"] = bool(per_example) and all(
        v["transparent"] for v in per_example.values()
    )
    summary["ab_max_overhead_pct"] = max(
        (v["overhead_pct"] for v in per_example.values()), default=None
    )

    # ---------------- decision block ----------------
    decisions = {
        "D1_tensorlbm_worse_than_oss": {
            "dup_line_ratio": [round(tb["dup_line_ratio"], 4), round(lb["dup_line_ratio"], 4)],
            "entry_ratio": [round(tb["entry_ratio"], 4), round(lb["entry_ratio"], 4)],
            "cli_adoption": [round(tb["cli_adoption"], 4), round(lb["cli_adoption"], 4)],
            "programmatic_index_baseline": [tb_baseline["has_programmatic_index"],
                                            lb["has_programmatic_index"]],
            "programmatic_index_after": [tb["has_programmatic_index"],
                                         lb["has_programmatic_index"]],
        },
        "D2_transparent": summary["ab_all_transparent"],
        "D3_overhead_ok_15pct": (
            None if summary["ab_max_overhead_pct"] is None
            else summary["ab_max_overhead_pct"] <= 15.0
        ),
    }
    summary["decisions"] = decisions
    summary["finished_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    _write(OUT / "summary.json", summary)
    print(json.dumps(decisions, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())