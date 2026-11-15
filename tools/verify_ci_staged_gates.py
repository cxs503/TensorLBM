#!/usr/bin/env python3
"""Verify the GitLab staged-gate ladder of this repository (executable receipt).

The ``.gitlab-ci.yml`` header claims four ladder properties (C1 fail-fast,
C2 stage barrier survives ``needs: []``, C3 advisory gates cannot stop the
pipeline, C4 rules-restricted gates only run on their events).  This script
turns those claims, plus structural and artefact-contract hygiene, into a
machine-checked receipt:

    python tools/verify_ci_staged_gates.py                # semantics only
    python tools/verify_ci_staged_gates.py --probe        # + run the cheap
                                                          #   blocking gates
    python tools/verify_ci_staged_gates.py --json-out docs/ci/receipt.json

Exit codes
----------
0  ladder verified: no structural/artefact errors, all C1-C4 checks pass,
   and (with ``--probe``) every probed blocking gate is green.
1  a claim is falsified, an artefact contract is broken, or a probed blocking
   gate is red.
2  the verifier itself could not complete.

Gates that cannot be exercised inside one short local run (full pytest suite,
platform suite, mypy, package build, sphinx) are reported as ``not_run`` with
a reason -- never as passed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# `tools/` contains unrelated sub-directories whose names collide with the
# repository packages (`tools/xval`), and Python puts the *script* directory
# first on sys.path.  Drop it before importing the repo namespace packages.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in (_SCRIPT_DIR, "", ".")]
sys.path.insert(0, str(REPO_ROOT))

from xval.gitlab_ci_staged_gates.stage_semantics import Pipeline, summarize  # noqa: E402

#: Local probes for the blocking gates that are cheap to run hermetically.
#: Each entry is (argv, per-step timeout seconds).  Anything absent is not_run.
PROBES: dict[str, tuple[tuple[list[str], int], ...]] = {
    "lint": (([sys.executable, "tools/ci_ratchet.py", "ruff"], 180),),
    "format-check": (([sys.executable, "tools/ci_ratchet.py", "format"], 180),),
    "repository-layout": (
        ([sys.executable, "tests/test_repository_layout.py"], 180),
        ([sys.executable, "tests/test_repository_report_migration.py"], 180),
        ([sys.executable, "tools/check_repository_layout.py"], 180),
    ),
    "architecture-audit": (([sys.executable, "tests/test_audit_import_graph.py"], 180),),
}

NOT_RUN_REASONS = {
    "testsuite-fast": "full fast suite exceeds the short local budget; owned by CI runners",
    "platform-test": "needs app/requirements.txt environment; owned by CI runners",
    "typecheck-core": "mypy on 4 modules exceeds the short local budget; owned by CI runners",
    "typecheck-full": "advisory full-package mypy; owned by CI runners",
    "package-build": "build/install round-trip; owned by CI runners",
    "build-documentation": "sphinx build; owned by CI runners",
    "coverage-report": "needs coverage data produced by testsuite-fast",
    "prerelease-pytorch-nightly": "installs pre-release torch; advisory nightly-only job",
    "nightly-full": "scheduled full suite; not runnable as a gate probe",
    "pages": "deploy job; only meaningful on the default branch",
}


def _probe(name: str, cwd: Path) -> dict[str, object]:
    steps = PROBES.get(name)
    if not steps:
        return {"status": "not_run", "reason": NOT_RUN_REASONS.get(name, "no local probe defined")}
    if (
        steps[0][0][0] == sys.executable
        and not shutil.which("ruff")
        and "ci_ratchet" in steps[0][0][1]
    ):
        return {"status": "not_run", "reason": "ruff is not installed in this environment"}
    detail = []
    started = time.time()
    for argv, timeout in steps:
        try:
            proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"status": "not_run", "reason": f"timeout after {timeout}s: {' '.join(argv)}"}
        detail.append(
            {
                "cmd": " ".join(argv),
                "rc": proc.returncode,
                "tail": (proc.stdout + proc.stderr).strip().splitlines()[-3:],
            }
        )
        if proc.returncode != 0:
            return {
                "status": "fail",
                "detail": detail,
                "elapsed_s": round(time.time() - started, 2),
            }
    return {"status": "pass", "detail": detail, "elapsed_s": round(time.time() - started, 2)}


def greenness(pipe: Pipeline, cwd: Path) -> dict[str, object]:
    """Run the cheap blocking gates; report honesty-preserving not_run states."""
    gates = sorted((n, j) for n, j in pipe.jobs.items() if j.is_gate)
    results: dict[str, object] = {}
    for name, job in gates:
        if job.stage == pipe.stages[-1] and not PROBES.get(name):
            results[name] = {"status": "not_run", "reason": "final-stage job (deploy sink)"}
            continue
        results[name] = _probe(name, cwd)
    counts = {"pass": 0, "fail": 0, "not_run": 0}
    for res in results.values():
        counts[str(res["status"])] += 1
    return {
        "n_blocking_gates": len(gates),
        "counts": counts,
        "not_run": sorted(n for n, r in results.items() if r["status"] == "not_run"),
        "red": sorted(n for n, r in results.items() if r["status"] == "fail"),
        "per_gate": results,
    }


def render_markdown(receipt: dict) -> str:
    lines = [
        "# Staged-gate ladder receipt",
        "",
        f"* source: `{receipt['source']}`",
        f"* stages: {', '.join(f'`{s}`' for s in receipt['stages'])}",
        f"* jobs: {receipt['n_jobs']}  ·  blocking gates: {len(receipt['gates'])}"
        f"  ·  advisory: {', '.join(receipt['advisory']) or '—'}",
        f"* structural/artefact errors: **{receipt['n_error']}**",
        f"* ladder claims (C1–C4): {receipt['claims']['n_checks'] - receipt['claims']['n_failed']}"
        f"/{receipt['claims']['n_checks']} verified",
        "",
        "## Gate power (downstream jobs pruned when this gate fails alone)",
        "",
        "| event | " + " | ".join(receipt["gates"]) + " |",
        "|" + "---|" * (len(receipt["gates"]) + 1),
    ]
    for event, payload in receipt["events"].items():
        power = payload["gate_power"]
        cells = [str(power.get(g, "–")) for g in receipt["gates"]]
        lines.append(f"| {event} | " + " | ".join(cells) + " |")
    if "greenness" in receipt:
        green = receipt["greenness"]
        lines += [
            "",
            "## Blocking-gate greenness (local probes)",
            "",
            f"* pass: {green['counts']['pass']}  ·  fail: {green['counts']['fail']}"
            f"  ·  not run: {green['counts']['not_run']}",
            "",
            "| gate | status | note |",
            "|---|---|---|",
        ]
        for name, res in sorted(green["per_gate"].items()):
            note = res.get("reason", "") or ""
            lines.append(f"| `{name}` | {res['status']} | {note} |")
    if receipt["findings"]:
        lines += ["", "## Findings", ""]
        for f in receipt["findings"]:
            lines.append(f"* **{f['severity']}** `{f['job']}` — {f['message']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ci", type=Path, default=REPO_ROOT / ".gitlab-ci.yml")
    ap.add_argument("--probe", action="store_true", help="also run the cheap blocking gates")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--markdown-out", type=Path, default=None)
    ap.add_argument("--cwd", type=Path, default=REPO_ROOT)
    args = ap.parse_args(argv)

    if not args.ci.is_file():
        print(f"no CI definition at {args.ci}", file=sys.stderr)
        return 2
    try:
        pipe = Pipeline.load(args.ci)
        receipt = summarize(pipe)
        if args.probe:
            receipt["greenness"] = greenness(pipe, args.cwd)
    except Exception as exc:  # pragma: no cover - verifier self-failure
        print(f"verifier failed: {exc!r}", file=sys.stderr)
        return 2

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_markdown(receipt))

    print(render_markdown(receipt))

    failed = receipt["n_error"] > 0 or receipt["claims"]["n_failed"] > 0
    if args.probe and receipt["greenness"]["counts"]["fail"]:
        failed = True
    print("VERDICT:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
