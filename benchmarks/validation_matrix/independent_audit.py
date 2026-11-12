"""Independent re-judgement of the validation-matrix statuses and status table.

Why this file exists
--------------------
``criteria.py`` decides the three-level status and ``runner.py`` renders the
published ``STATUS_MATRIX.md``.  A reviewer who only re-runs those two modules
learns that the code agrees with itself, which is the weakest possible form of
verification.  This module re-derives the same verdicts from the frozen JSON
artifacts with its **own** dotted-path resolver, its **own** error/gating math
and its **own** status rule, and imports neither ``criteria`` nor ``runner``.
It then goes one step further and checks the *published* table against those
independent verdicts, because a table is a claim: every ``passed`` row must
display a metric that is (a) gated at the level it quotes and (b) inside the
tolerance it prints.

Checks performed (all machine-readable, exit code 1 on any failure)
-------------------------------------------------------------------
``C0`` artifact/level resolution
    The declared artifact exists and every declared metric path resolves to a
    finite float.  An unresolved path is *missing evidence*, never a pass.
``C1`` per-level gating
    A level with ``tol_pct_by_level[i] is None`` is informational: it is
    reported and enters the convergence trend, but it cannot fail the metric.
    A level with a tolerance fails when ``|err| > tol``.
``C2`` convergence
    ``max|err| - min|err| <= max_span_pct`` and, where the case requires it,
    ``|err|`` is non-increasing from coarse to fine.
``C3`` status rule
    ``declared_status`` may only *downgrade* (``pending``/``failed``); a machine
    FAIL is never upgraded to passed.
``C4`` registry consistency (zero hand-copy)
    A ``kind="value"`` metric carries an inline ``reference`` in ``matrix.json``
    *and* names a registry anchor in ``references.json``.  The two must be the
    same number, otherwise the gate is silently judging against a hand-typed
    copy that the provenance audit never sees.
``C5`` published-table consistency
    For every row of ``STATUS_MATRIX.md``: the status token equals the
    independent status, and a ``passed`` row quotes a gated level whose
    ``|err| <= tol`` as printed.

Usage::

    python benchmarks/validation_matrix/independent_audit.py
    python benchmarks/validation_matrix/independent_audit.py --json out.json

Exit code 0 = every check clean, 1 = at least one finding.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
MATRIX = HERE / "matrix.json"
REFERENCES = HERE / "references.json"
STATUS_TABLE = HERE / "STATUS_MATRIX.md"

STATUSES = ("passed", "failed", "pending")
ROW_RE = re.compile(
    r"^\|\s*\d+\s*\|\s*([^|]+?)\s*\|\s*`([^`]+)`\s*\|\s*\*\*([a-z]+)\*\*\s*\|"
    r"\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*`([^`]*)`\s*\|"
)
KEY_METRIC_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*)\s+([-+]?\d+(?:\.\d+)?)\s*%\s*\(tol\s+([-+]?\d+(?:\.\d+)?)\s*%\s*\)"
)


class AuditError(RuntimeError):
    """Structural problem: the matrix itself cannot be audited as declared."""


def resolve(payload, dotted: str):
    """Resolve ``a.b.0.c`` against nested dict/list JSON (own mini-resolver)."""
    node = payload
    for part in dotted.split("."):
        if isinstance(node, dict):
            node = node[part]
        elif isinstance(node, (list, tuple)):
            node = node[int(part)]
        else:  # pragma: no cover - defensive
            raise KeyError(dotted)
    return node


def signed_error_pct(value: float, reference: float) -> float:
    if reference == 0:
        raise AuditError("percentage error against a zero reference is undefined")
    return (value - reference) / abs(reference) * 100.0


def judge_level(value, reference, tol) -> dict:
    """One level of one metric. ``tol is None`` means *informational*."""
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return {
            "value": None,
            "err_pct": None,
            "tol_pct": None if tol is None else abs(float(tol)),
            "gated": tol is not None,
            "pass": False,
            "note": "unresolved metric path (missing evidence)",
        }
    err = (
        signed_error_pct(float(value), float(reference))
        if reference is not None
        else float(value)
    )
    if tol is None:
        return {
            "value": float(value),
            "err_pct": err,
            "tol_pct": None,
            "gated": False,
            "pass": True,
            "note": "informational level (not gated)",
        }
    return {
        "value": float(value),
        "err_pct": err,
        "tol_pct": abs(float(tol)),
        "gated": True,
        "pass": abs(err) <= abs(float(tol)),
        "note": "",
    }


def trend(errors, max_span_pct: float, require_monotone) -> dict:
    errs = [abs(float(e)) for e in errors]
    if len(errs) < 2:
        raise AuditError("convergence needs two levels")
    span = max(errs) - min(errs)
    monotone = all(b <= a + 1e-12 for a, b in zip(errs, errs[1:]))
    if require_monotone is None:
        require_monotone = len(errs) >= 3
    return {
        "errors_abs_pct": errs,
        "span_pct": span,
        "max_span_pct": abs(float(max_span_pct)),
        "monotone": monotone,
        "require_monotone": bool(require_monotone),
        "pass": span <= abs(float(max_span_pct))
        and ((not require_monotone) or monotone),
    }


def audit_case(case: dict, artifact_root: Path) -> dict:
    measure = case.get("measure") or {}
    metrics_out: list[dict] = []
    conv_out: list[dict] = []
    findings: list[str] = []
    artifact_rel = measure.get("artifact")
    artifact_path = (artifact_root / artifact_rel) if artifact_rel else None
    payload = None
    if artifact_path is not None and artifact_path.exists():
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    levels = list(measure.get("levels") or [])
    for spec in measure.get("metrics", []):
        tol_base = float(spec.get("tol_pct", 3.0))
        tol_by_level = spec.get("tol_pct_by_level")
        rows = []
        for i, path in enumerate(spec.get("paths", [])):
            tol = tol_base if tol_by_level is None else tol_by_level[i]
            if tol is not None:
                tol = float(tol)
            value = None
            if payload is not None:
                try:
                    value = float(resolve(payload, path))
                except (KeyError, IndexError, TypeError, ValueError):
                    value = None
                    findings.append(
                        ("C0", "advisory",
                         f"{spec['name']} path {path!r} unresolved")
                    )
            else:
                findings.append(
                    ("C0", "advisory",
                     f"artifact missing: {artifact_rel or '(none declared)'}")
                )
            row = judge_level(value, spec.get("reference"), tol)
            row["level"] = levels[i] if i < len(levels) else path
            row["path"] = path
            rows.append(row)
        gated = [r for r in rows if r["gated"]]
        metrics_out.append(
            {
                "metric": spec["name"],
                "kind": spec.get("kind", "value"),
                "reference": spec.get("reference"),
                "tol_pct": tol_base,
                "levels": rows,
                "pass": bool(gated) and all(r["pass"] for r in gated),
                "worst_gated": max(
                    (r for r in gated if r["err_pct"] is not None),
                    key=lambda r: abs(r["err_pct"]),
                    default=None,
                ),
                "informational_levels": [r["level"] for r in rows if not r["gated"]],
            }
        )

    for spec in measure.get("convergence") or []:
        target = next((m for m in metrics_out if m["metric"] == spec["metric"]), None)
        if target is None:
            conv_out.append({"metric": spec["metric"], "pass": False,
                             "note": "metric not declared"})
            findings.append(
                ("C2", "blocking", f"{spec['metric']} convergence on undeclared metric")
            )
            continue
        errs = [r["err_pct"] for r in target["levels"] if r["err_pct"] is not None]
        if len(errs) < 2:
            conv_out.append({"metric": spec["metric"], "pass": False,
                             "note": "fewer than two resolved levels"})
            findings.append(
                ("C2", "advisory", f"{spec['metric']} fewer than two resolved levels")
            )
            continue
        t = trend(errs, spec.get("max_span_pct", 3.0), spec.get("require_monotone"))
        t["metric"] = spec["metric"]
        conv_out.append(t)

    machine_pass = bool(metrics_out) and all(m["pass"] for m in metrics_out) and all(
        c["pass"] for c in conv_out
    )
    declared = case.get("declared_status")
    if declared == "pending":
        status = "pending"
    elif declared == "failed" or not machine_pass:
        status = "failed"
    else:
        status = "passed"
    # A missing/unresolved metric path is missing evidence.  It can never hide
    # behind a pass: if the case would otherwise be passed it is blocking.
    if status == "passed" and findings:
        findings = [(c, "blocking", m) for (c, _s, m) in findings]
    return {
        "case": case["id"],
        "slot": case["slot"],
        "declared_status": declared,
        "status": status,
        "machine_pass": machine_pass,
        "metrics": metrics_out,
        "convergence": conv_out,
        "findings": [{"code": c, "severity": s_, "message": m}
                     for (c, s_, m) in findings],
        "artifact": artifact_rel,
    }


def audit_registry(matrix: dict, references: dict, artifact_root: Path) -> dict:
    """C4: inline ``reference`` must equal the registry anchor value."""
    findings = []
    checked = 0
    for case in matrix["cases"]:
        for spec in (case.get("measure") or {}).get("metrics", []):
            inline = spec.get("reference")
            if spec.get("kind", "value") == "error_pct" or inline is None:
                continue
            for ref_id in case.get("refs", []):
                ref = references.get(ref_id)
                if ref is None:
                    findings.append(
                        f"C4 {case['id']}/{spec['name']}: unknown reference {ref_id!r}"
                    )
                    continue
                if ref.get("value") is None:
                    continue
                checked += 1
                if abs(float(ref["value"]) - float(inline)) > 1e-12 * max(
                    1.0, abs(float(ref["value"]))
                ):
                    findings.append(
                        {"code": "C4", "severity": "blocking",
                         "message": f"{case['id']}/{spec['name']}: inline reference "
                                    f"{inline!r} != registry {ref_id} "
                                    f"value {ref['value']!r}"}
                    )
    return {"checked": checked, "findings": findings, "notes": [], "pass": not findings}


def audit_table(table_path: Path, cases: dict) -> dict:
    """C5: every published row must agree with the independent verdict."""
    if not table_path.exists():
        return {"pass": False, "rows": [], "notes": [],
                "findings": [{"code": "C5", "severity": "blocking",
                              "message": f"status table missing: {table_path}"}]}
    findings = []
    rows = []
    for line in table_path.read_text(encoding="utf-8").splitlines():
        m = ROW_RE.match(line.strip())
        if not m:
            continue
        slot, case_id, status, key_cell, _conv, _ev = m.groups()
        rec = {"case": case_id, "slot": slot.strip(), "status": status,
               "key_metric": key_cell, "findings": []}

        def flag(message: str, code: str = "C5") -> None:
            rec["findings"].append(
                {"code": code, "severity": "blocking", "message": message}
            )
        if status not in STATUSES:
            flag(f"unknown status token {status!r}")
        verdict = cases.get(case_id)
        if verdict is None:
            flag("row for a case that is not in matrix.json")
            findings.extend(rec["findings"])
            rows.append(rec)
            continue
        if verdict["status"] != status:
            flag(
                f"status mismatch: table {status!r} vs independent "
                f"{verdict['status']!r}"
            )
        if status == "passed":
            km = KEY_METRIC_RE.match(key_cell)
            if not km:
                flag(f"passed row without a parseable key metric: {key_cell!r}")
            else:
                name, err, tol = km.group(1), float(km.group(2)), float(km.group(3))
                if abs(err) > tol:
                    flag(
                        f"CONTRADICTION: passed row DISPLAYS |err| {err:+.2f}% "
                        f"> tol {tol:.2f}% (metric {name})"
                    )
                metric = next(
                    (m_ for m_ in verdict["metrics"] if m_["metric"] == name), None
                )
                if metric is None:
                    flag(f"key metric {name!r} not declared for this case")
                else:
                    worst = metric["worst_gated"]
                    if worst is None:
                        flag(f"key metric {name!r} has no gated level")
                    elif abs(abs(worst["err_pct"]) - abs(err)) > 5e-3:
                        flag(
                            f"key metric value mismatch: table {err:+.2f}% vs "
                            f"independent gated worst {worst['err_pct']:+.2f}% "
                            f"({worst['level']})"
                        )
                    elif abs(worst["tol_pct"] - tol) > 5e-3:
                        flag(
                            f"tolerance mismatch: table tol {tol:.2f}% vs "
                            f"gated tol {worst['tol_pct']:.2f}%"
                        )
        findings.extend(rec["findings"])
        rows.append(rec)
    return {"pass": not findings, "rows": rows, "findings": findings, "notes": []}


def run(matrix_path: Path = MATRIX, references_path: Path = REFERENCES,
        table_path: Path = STATUS_TABLE, artifact_root: Path = REPO_ROOT) -> dict:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    references = json.loads(references_path.read_text(encoding="utf-8"))["references"]
    case_verdicts = {
        case["id"]: audit_case(case, artifact_root) for case in matrix["cases"]
    }
    registry = audit_registry(matrix, references, artifact_root)
    table = audit_table(table_path, case_verdicts)
    counts = {s: sum(1 for v in case_verdicts.values() if v["status"] == s)
              for s in STATUSES}
    findings: list[dict] = []
    notes: list[dict] = []
    for cid, v in case_verdicts.items():
        for f in v["findings"]:
            rec = {"case": cid, **f}
            (findings if f["severity"] == "blocking" else notes).append(rec)
    findings.extend(registry["findings"])
    findings.extend(table["findings"])
    notes.extend(registry.get("notes", []))
    notes.extend(table.get("notes", []))
    return {
        "kind": "tensorlbm.validation_matrix.independent_audit/v1",
        "matrix_version": matrix.get("version"),
        "matrix_sha256": _sha256(matrix_path),
        "status_table_sha256": _sha256(table_path) if table_path.exists() else None,
        "counts": counts,
        "cases": case_verdicts,
        "registry_consistency": registry,
        "status_table": table,
        "findings": findings,
        "notes": notes,
        "pass": not findings,
    }


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", type=Path, default=None,
                    help="write the full audit record here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    report = run()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    c = report["counts"]
    if not args.quiet:
        print(
            f"independent verdicts: passed {c['passed']} / failed {c['failed']} / "
            f"pending {c['pending']}  | blocking findings "
            f"{len(report['findings'])} | advisory notes {len(report['notes'])}"
        )
        for f in report["findings"]:
            print(f"  FAIL [{f['code']}] {f.get('case', '-')}: {f['message']}")
        for n in report["notes"]:
            print(f"  note [{n['code']}] {n.get('case', '-')}: {n['message']}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())