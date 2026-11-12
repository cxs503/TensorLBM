"""Validation-matrix runner: one command per case, machine-checked verdicts.

Usage::

    python benchmarks/validation_matrix/runner.py list
    python benchmarks/validation_matrix/runner.py audit          # reference provenance
    python benchmarks/validation_matrix/runner.py check          # judge frozen evidence
    python benchmarks/validation_matrix/runner.py run --case dam_break_ritter_ci
    python benchmarks/validation_matrix/runner.py run --tier ci  # CI-scale cases only
    python benchmarks/validation_matrix/runner.py status --write # three-level table

``run`` executes the case's ``repro.cmd`` (unless ``--no-exec``) and then judges
the artifact it produces; ``check`` only judges the artifacts already on disk.
Neither ever upgrades a status: a missing artifact is ``pending`` with the
reason recorded, a failed criterion is ``failed``, and a declared negative
result (Rayleigh-Taylor, channel) keeps its root cause.

Status files are written into the repository lifecycle folders::

    benchmarks/verified/<case_id>/validation_status.json
    benchmarks/pending/<case_id>/validation_status.json

so the three-level table and the evidence chain live under the same
``verified|pending`` directory convention that the rest of ``benchmarks/`` uses.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
MATRIX = HERE / "matrix.json"
REFS = HERE / "references.json"
STATUS_TABLE = HERE / "STATUS_MATRIX.md"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from criteria import STATUSES, judge_case  # noqa: E402
from reference_audit import audit_all, load_references, sha256_file  # noqa: E402


class MatrixError(RuntimeError):
    pass


def _dotted(obj, path: str):
    cur = obj
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur


def load_matrix() -> dict:
    return json.loads(MATRIX.read_text(encoding="utf-8"))


def find_case(matrix: dict, case_id: str) -> dict:
    for case in matrix["cases"]:
        if case["id"] == case_id:
            return case
    raise MatrixError(f"unknown case id {case_id!r}; known: "
                      + ", ".join(c["id"] for c in matrix["cases"]))


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _base_evidence(case: dict) -> list[dict]:
    """Evidence chain that exists even when the case produced no artifact."""
    chain = [
        {"kind": "matrix", "path": str(MATRIX.relative_to(REPO_ROOT)),
         "sha256": sha256_file(MATRIX)},
        {"kind": "references", "path": str(REFS.relative_to(REPO_ROOT)),
         "sha256": sha256_file(REFS)},
    ]
    for token in ((case.get("repro") or {}).get("cmd") or "").split():
        candidate = REPO_ROOT / token
        if token.endswith(".py") and candidate.exists():
            chain.append({"kind": "runner", "path": token,
                          "sha256": sha256_file(candidate)})
    return chain


def evaluate_case(case: dict, references: dict | None = None) -> dict:
    """Judge one case against the frozen artifact declared in the matrix."""
    references = references if references is not None else load_references()
    measure = case.get("measure") or {}
    artifact_rel = measure.get("artifact")
    artifact_path = REPO_ROOT / artifact_rel if artifact_rel else None

    result: dict = {
        "case": case["id"],
        "slot": case["slot"],
        "title": case["title"],
        "reference_basis": case.get("reference_basis"),
        "scope": case.get("scope"),
        "matrix_version": load_matrix()["version"],
        "generated_utc": _utcnow(),
        "artifact": artifact_rel,
        "repro_cmd": (case.get("repro") or {}).get("cmd"),
    }

    for ref_id in case.get("refs", []):
        ref = references.get(ref_id)
        if ref is None:
            raise MatrixError(f"case {case['id']}: unknown reference {ref_id!r}")
        result.setdefault("references", []).append(
            {
                "id": ref_id,
                "quantity": ref["quantity"],
                "value": ref.get("value"),
                "formula": ref.get("formula"),
                "origin": ref["origin"],
                "source": ref["source"],
                "source_sha256": sha256_file(REPO_ROOT / ref["source"])
                if (REPO_ROOT / ref["source"]).exists()
                else None,
                "token": ref["token"],
            }
        )

    metric_specs = measure.get("metrics", [])
    levels = measure.get("levels", [])

    if artifact_path is None or not artifact_path.exists():
        result.update(
            {
                "status": "pending",
                "machine_pass": False,
                "reason": f"no artifact yet: {artifact_rel or '(none declared)'}"
                + (" — rerun `runner.py run --case " + case["id"] + "`" if artifact_rel else
                   " — " + str(case.get("blocker", "no machine-checkable artifact")[:200])),
                "blocker": case.get("blocker"),
                "metrics": [],
                "convergence": [],
            }
        )
        if case.get("declared_status"):
            result["declared_status"] = case["declared_status"]
            result["status"] = case["declared_status"]
            if case.get("blocker"):
                result["reason"] = case["blocker"]
        result["evidence_chain"] = _base_evidence(case)
        if case.get("notes"):
            result["notes"] = case["notes"]
        return result

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    metric_values: dict[str, list[float]] = {}
    resolved = {}
    for spec in metric_specs:
        vals = []
        ok = True
        for path in spec["paths"]:
            try:
                vals.append(float(_dotted(payload, path)))
            except (KeyError, IndexError, TypeError, ValueError):
                ok = False
                vals.append(float("nan"))
        resolved[spec["name"]] = ok
        metric_values[spec["name"]] = vals

    verdict = judge_case(
        metric_values,
        metric_specs,
        measure.get("convergence") or [],
        levels=levels,
        declared_status=case.get("declared_status"),
        blocker=case.get("blocker"),
    )
    result.update(verdict)
    result["unresolved_metrics"] = [k for k, v in resolved.items() if not v]
    if result["unresolved_metrics"]:
        # A declared-but-absent artifact path is missing evidence, not a pass.
        # A case that already carries a documented negative result keeps it.
        result["machine_pass"] = False
        if case.get("declared_status") == "failed":
            result["status"] = "failed"
            result["reason"] = case.get("blocker") or (
                "declared failed; no machine-checkable metric paths"
            )
        else:
            result["status"] = "pending"
            result["reason"] = (
                "artifact present but metric paths unresolved: "
                + ", ".join(result["unresolved_metrics"])
            )

    result["artifact_sha256"] = sha256_file(artifact_path)
    result["artifact_bytes"] = artifact_path.stat().st_size
    result["evidence_chain"] = [
        {"kind": "artifact", "path": artifact_rel, "sha256": result["artifact_sha256"]}
    ] + _base_evidence(case)
    if case.get("notes"):
        result["notes"] = case["notes"]
    return result


def run_repro(case: dict) -> dict:
    repro = case.get("repro") or {}
    cmd = repro.get("cmd")
    if not cmd:
        raise MatrixError(f"case {case['id']} declares no repro command")
    started = _dt.datetime.now(_dt.timezone.utc)
    proc = subprocess.run(
        cmd, shell=True, cwd=REPO_ROOT, capture_output=True, text=True
    )
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_s": (_dt.datetime.now(_dt.timezone.utc) - started).total_seconds(),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def write_status(result: dict) -> Path:
    status = result["status"]
    if status == "passed":
        folder = REPO_ROOT / "benchmarks" / "verified" / result["case"]
    else:
        folder = REPO_ROOT / "benchmarks" / "pending" / result["case"]
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / "validation_status.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def render_table(results: list[dict]) -> str:
    order = {name: i for i, name in enumerate(("passed", "failed", "pending"))}
    results = sorted(results, key=lambda r: (order.get(r["status"], 9), r["case"]))
    counts = {s: sum(1 for r in results if r["status"] == s) for s in STATUSES}
    lines = [
        "# Validation matrix — three-level status (generated)",
        "",
        f"generated: {_utcnow()} | matrix v{load_matrix()['version']} | "
        f"**passed {counts['passed']} / failed {counts['failed']} / "
        f"pending {counts['pending']}**",
        "",
        "Gate: every metric |err| <= its tolerance at every level AND the level series",
        "converges (span <= 3%, monotone where >= 3 levels exist). `failed` keeps its root",
        "cause; `pending` means evidence is withheld or not yet produced — never a hidden pass.",
        "",
        "| # | slot | case | status | key metric (worst level) | convergence | evidence |",
        "|---|------|------|--------|--------------------------|-------------|----------|",
    ]
    for i, r in enumerate(results, 1):
        worst = "—"
        conv = "—"
        if r.get("metrics"):
            rows = [
                (m["metric"], m.get("max_abs_err_pct"), m.get("tol_pct"))
                for m in r["metrics"]
                if m.get("max_abs_err_pct") is not None
            ]
            if rows:
                name, err, tol = max(rows, key=lambda t: abs(t[1]))
                worst = f"{name} {err:+.2f}% (tol {tol:.0f}%)"
        if r.get("convergence"):
            convs = [
                f"{c['metric']} span {c['span_pct']:.2f}%"
                for c in r["convergence"]
                if "span_pct" in c
            ]
            conv = "; ".join(convs) or "—"
        ev = r.get("artifact") or (r.get("repro_cmd") or "—")
        lines.append(
            f"| {i} | {r['slot']} | `{r['case']}` | **{r['status']}** | {worst} | "
            f"{conv} | `{ev}` |"
        )
    lines += ["", "## Root causes / blockers", ""]
    for r in results:
        if r["status"] == "failed" and r.get("reason"):
            lines.append(f"- `{r['case']}` **failed** — {r['reason']}")
    for r in results:
        if r["status"] == "pending" and r.get("reason"):
            lines.append(f"- `{r['case']}` **pending** — {r['reason']}")
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "python benchmarks/validation_matrix/runner.py audit    # reference provenance (zero hand-copy)",
        "python benchmarks/validation_matrix/runner.py check    # re-judge frozen evidence",
        "python benchmarks/validation_matrix/runner.py run --tier ci   # CI-scale cases end-to-end",
        "python benchmarks/validation_matrix/runner.py status --write  # regenerate this table",
        "```",
        "",
        "Per-case status + sha256 evidence chain: `benchmarks/{verified,pending}/<case>/validation_status.json`.",
        "",
    ]
    return "\n".join(lines)


def cmd_list(args) -> int:
    matrix = load_matrix()
    for case in matrix["cases"]:
        declared = case.get("declared_status", "machine")
        print(f"{case['id']:<28} slot={case['slot']:<15} declared={declared:<8} "
              f"{case['title']}")
    return 0


def cmd_audit(args) -> int:
    report = audit_all()
    print(f"{report['n_passed']}/{report['n_references']} reference anchors audited clean")
    for rec in report["references"]:
        if not rec["pass"]:
            print(f"  FAIL {rec['id']}: " + "; ".join(rec["reasons"]))
    return 0 if report["pass"] else 1


def cmd_check(args) -> int:
    matrix = load_matrix()
    results = [evaluate_case(case) for case in matrix["cases"]]
    for r in results:
        print(f"[{r['status']:<7}] {r['case']:<28} {r.get('reason', '')[:110]}")
    return 0


def cmd_run(args) -> int:
    matrix = load_matrix()
    if args.case:
        cases = [find_case(matrix, args.case)]
    elif args.tier == "ci":
        cases = [c for c in matrix["cases"] if "ci" in c["id"] or "ci" in (
            c.get("repro") or {}).get("cmd", "")]
    else:
        cases = matrix["cases"]

    exit_code = 0
    for case in cases:
        print(f"=== {case['id']} ===", flush=True)
        if not args.no_exec:
            run = run_repro(case)
            print(f"  repro rc={run['returncode']} in {run['elapsed_s']:.1f}s", flush=True)
            if run["returncode"] != 0:
                print("  stderr tail:\n" + run["stderr_tail"], flush=True)
                exit_code = 1
                continue
        result = evaluate_case(case)
        path = write_status(result)
        print(f"  status={result['status']} -> {path.relative_to(REPO_ROOT)}")
        if result["status"] == "failed":
            exit_code = 1
    return exit_code


def cmd_status(args) -> int:
    matrix = load_matrix()
    results = [evaluate_case(case) for case in matrix["cases"]]
    table = render_table(results)
    if args.write:
        STATUS_TABLE.write_text(table, encoding="utf-8")
        for r in results:
            write_status(r)
        shown = STATUS_TABLE.relative_to(REPO_ROOT)
        print(f"wrote {shown} and {len(results)} validation_status.json files")
    else:
        print(table)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list the matrix cases").set_defaults(func=cmd_list)
    sub.add_parser("audit", help="audit reference provenance").set_defaults(func=cmd_audit)
    sub.add_parser("check", help="judge frozen evidence").set_defaults(func=cmd_check)

    p_run = sub.add_parser("run", help="execute a case and judge it")
    p_run.add_argument("--case")
    p_run.add_argument("--tier", choices=["ci", "all"], default="all")
    p_run.add_argument("--no-exec", action="store_true",
                       help="skip the repro command and judge the existing artifact")
    p_run.set_defaults(func=cmd_run)

    p_st = sub.add_parser("status", help="render the three-level status table")
    p_st.add_argument("--write", action="store_true",
                      help="write STATUS_MATRIX.md and the per-case status files")
    p_st.set_defaults(func=cmd_status)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())