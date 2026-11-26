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
    python tools/verify_ci_staged_gates.py --check \
        docs/ci/staged_gate_semantics_receipt.json      # committed receipt
                                                        #   still true?
    python tools/verify_ci_staged_gates.py --check-doc \
        docs/ci/gitlab_ci_staged_gates.md               # generated ladder
                                                        #   table still true?

Exit codes
----------
0  ladder verified: no structural/artefact errors, all C1-C4 checks pass,
   and (with ``--probe``) every probed blocking gate is green.
1  a claim is falsified, an artefact contract is broken, or a probed blocking
   gate is red.
1  (``--check`` / ``--check-doc``) the committed receipt or the generated
   ladder table in the design document disagrees with today's
   ``.gitlab-ci.yml`` -- i.e. the artefact is STALE.
2  the verifier itself could not complete.

Why the stale check exists
--------------------------
A receipt that was true when it was committed proves nothing about today's
pipeline.  The committed receipt therefore carries the sha256 of the exact
``.gitlab-ci.yml`` bytes it was derived from, and ``--check`` diffs the live
re-derivation against it field by field.  The ladder tables in the design
document are generated from the same derivation between explicit markers, so
prose cannot drift from the pipeline either.

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

from development.verification.gitlab_ci_staged_gates.stage_semantics import Pipeline, summarize  # noqa: E402

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


#: Fields that must agree between a committed receipt and a live re-derivation.
#: Environment-dependent fields (``greenness``) are deliberately excluded: they
#: depend on which optional tools are installed on the host, not on the ladder.
DRIFT_FIELDS = (
    "source",
    "source_sha256",
    "stages",
    "ladder",
    "n_jobs",
    "gates",
    "advisory",
    "immediate_jobs",
    "unverifiable_rules",
    "findings",
    "n_error",
    "events",
    "claims",
)

INVENTORY_BEGIN = "<!-- BEGIN:generated-ladder-inventory -->"
INVENTORY_END = "<!-- END:generated-ladder-inventory -->"


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


def _short(item: dict) -> str:
    """One-line human summary of a drift item."""
    if item["kind"] == "absent":
        return "missing from the committed receipt (added by a newer verifier)"
    was, now = item.get("committed"), item.get("live")
    return f"committed={json.dumps(was, sort_keys=True)[:120]} live={json.dumps(now, sort_keys=True)[:120]}"


def _event_label(events: list[str]) -> str:
    return ", ".join(events) if events else "—"


def _inventory_body(receipt: dict) -> str:
    """Ladder inventory as markdown rows, derived from the live CI definition.

    Nothing here is transcribed by hand: stage counts, tier mix and job lists
    all come from ``receipt['ladder']``.
    """
    ladder = receipt.get("ladder") or []
    n_jobs = sum(len(stage["jobs"]) for stage in ladder)
    tiers: dict[str, int] = {"blocking": 0, "advisory": 0, "manual": 0}
    for stage in ladder:
        for job in stage["jobs"]:
            tiers[job["tier"]] = tiers.get(job["tier"], 0) + 1
    empty = [s["stage"] for s in ladder if not s["jobs"]]
    lines = [
        f"{len(ladder)} declared stages · {n_jobs} jobs · "
        f"{tiers['blocking']} blocking · {tiers['advisory']} advisory · "
        f"{tiers['manual']} manual · {len(empty)} declared-but-empty "
        f"({', '.join(empty) or 'none'})",
        "",
        "| stage | active jobs | tier mix | jobs (tier · events) |",
        "|---|---|---|---|",
    ]
    for stage in ladder:
        mix: dict[str, int] = {}
        for job in stage["jobs"]:
            mix[job["tier"]] = mix.get(job["tier"], 0) + 1
        mix_txt = ", ".join(f"{k} {v}" for k, v in sorted(mix.items())) or "—"
        jobs_txt = (
            "<br>".join(
                f"`{j['name']}` ({j['tier']} · {_event_label(j['events'])})" for j in stage["jobs"]
            )
            or "*(declared, no active job)*"
        )
        lines.append(f"| `{stage['stage']}` | {len(stage['jobs'])} | {mix_txt} | {jobs_txt} |")
    return "\n".join(lines)


def render_inventory(receipt: dict) -> str:
    """Marker-delimited block kept in sync inside the design document."""
    provenance = (
        f"<!-- generated by tools/verify_ci_staged_gates.py --write-doc from "
        f"{receipt['source']} sha256:{receipt['source_sha256'][:16]} -->"
    )
    return (
        "\n".join([INVENTORY_BEGIN, provenance, "", _inventory_body(receipt), "", INVENTORY_END])
        + "\n"
    )


def drift_of(committed: dict, live: dict) -> list[dict[str, object]]:
    """Field-level drift between a committed receipt and the live derivation."""
    out: list[dict[str, object]] = []
    for field in DRIFT_FIELDS:
        if field not in committed:
            out.append({"field": field, "kind": "absent", "detail": "not in committed receipt"})
            continue
        was, now = committed[field], live.get(field)
        if was != now:
            out.append({"field": field, "kind": "changed", "committed": was, "live": now})
    return out


def extract_inventory(text: str) -> str | None:
    """Return the generated inventory block of a document, markers included."""
    start = text.find(INVENTORY_BEGIN)
    if start < 0:
        return None
    end = text.find(INVENTORY_END, start)
    if end < 0:
        return None
    return text[start : end + len(INVENTORY_END)] + "\n"


def rewrite_inventory(path: Path, block: str) -> bool:
    """Replace (or insert) the inventory block in ``path``; True if changed."""
    text = path.read_text()
    current = extract_inventory(text)
    if current is not None:
        if current == block:
            return False
        path.write_text(text.replace(current, block))
        return True
    raise ValueError(f"{path} has no {INVENTORY_BEGIN} ... {INVENTORY_END} markers")


def render_markdown(receipt: dict) -> str:
    lines = [
        "# Staged-gate ladder receipt",
        "",
        f"* source: `{receipt['source']}` sha256:`{receipt.get('source_sha256', '')[:16]}`",
        f"* stages: {', '.join(f'`{s}`' for s in receipt['stages'])}",
        f"* jobs: {receipt['n_jobs']}  ·  blocking gates: {len(receipt['gates'])}"
        f"  ·  advisory: {', '.join(receipt['advisory']) or '—'}",
        f"* structural/artefact errors: **{receipt['n_error']}**",
        f"* ladder claims (C1–C4): {receipt['claims']['n_checks'] - receipt['claims']['n_failed']}"
        f"/{receipt['claims']['n_checks']} verified",
        "",
        "## Ladder inventory (generated from the CI definition)",
        "",
        _inventory_body(receipt),
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
    ap.add_argument(
        "--check",
        type=Path,
        default=None,
        metavar="RECEIPT",
        help="fail if this committed receipt no longer matches a live re-derivation",
    )
    ap.add_argument(
        "--check-doc",
        type=Path,
        default=None,
        metavar="DOC",
        help="fail if the generated ladder inventory block in this document is stale",
    )
    ap.add_argument(
        "--write-doc",
        type=Path,
        default=None,
        metavar="DOC",
        help="regenerate the ladder inventory block inside this document",
    )
    args = ap.parse_args(argv)

    if not args.ci.is_file():
        print(f"no CI definition at {args.ci}", file=sys.stderr)
        return 2
    try:
        pipe = Pipeline.load(args.ci, root=REPO_ROOT)
        receipt = summarize(pipe)
        if args.probe:
            receipt["greenness"] = greenness(pipe, args.cwd)
    except Exception as exc:  # pragma: no cover - verifier self-failure
        print(f"verifier failed: {exc!r}", file=sys.stderr)
        return 2

    stale = False
    if args.check is not None:
        if not args.check.is_file():
            print(f"STALE: no committed receipt at {args.check}", file=sys.stderr)
            return 1
        committed = json.loads(args.check.read_text())
        drift = drift_of(committed, receipt)
        if drift:
            stale = True
            print(f"STALE: {args.check} disagrees with today's {receipt['source']}:")
            for item in drift:
                print(f"  - {item['field']}: {_short(item)}")
        else:
            print(
                f"fresh: {args.check} matches {receipt['source']} "
                f"sha256:{receipt['source_sha256'][:16]}"
            )
    if args.write_doc is not None:
        changed = rewrite_inventory(args.write_doc, render_inventory(receipt))
        print(f"{'rewrote' if changed else 'unchanged'}: {args.write_doc}")
    if args.check_doc is not None:
        text = args.check_doc.read_text()
        block = extract_inventory(text)
        if block is None:
            stale = True
            print(
                f"STALE: {args.check_doc} carries no generated ladder inventory block",
                file=sys.stderr,
            )
        elif block != render_inventory(receipt):
            stale = True
            print(
                f"STALE: ladder inventory in {args.check_doc} disagrees with today's "
                f"{receipt['source']} (run --write-doc)",
                file=sys.stderr,
            )

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
    if stale:
        failed = True
    print("VERDICT:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
