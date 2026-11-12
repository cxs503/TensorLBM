"""Generate the machine-built reference-vs-got table for the validation matrix.

The matrix README explains *why* a case passes; this module emits the raw
comparison an external reviewer needs in order to check that claim without
trusting prose: for every case, every gated metric, every level, the reference
value (with its in-repo anchor and sha256), the measured value, the signed
error in percent, the tolerance, and the level's pass flag — plus the exact
``repro_cmd`` and the artifact sha256.

Zero hand-copy: every number is read from ``matrix.json`` + ``references.json``
+ the artifact, and judged by the same ``criteria`` code path that decides the
three-level status.  If a case's artifact disappears the row says so instead of
keeping its old numbers.

Outputs::

    benchmarks/validation_matrix/BENCHMARK_REF_VS_GOT.json   machine-readable
    benchmarks/validation_matrix/BENCHMARK_REF_VS_GOT.md     reviewer-facing

Usage::

    python benchmarks/validation_matrix/make_benchmark_table.py [--check]

``--check`` exits non-zero when an emitted table is inconsistent with a fresh
machine verdict (used by CI).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import runner  # noqa: E402

JSON_OUT = HERE / "BENCHMARK_REF_VS_GOT.json"
MD_OUT = HERE / "BENCHMARK_REF_VS_GOT.md"


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if value != value:  # NaN
            return "nan"
        return f"{value:.6g}"
    return str(value)


def build() -> dict:
    matrix = runner.load_matrix()
    cases = []
    for case in matrix["cases"]:
        verdict = runner.evaluate_case(case)
        rows = []
        for metric in verdict.get("metrics", []):
            for level in metric.get("levels", []):
                rows.append(
                    {
                        "metric": metric["metric"],
                        "level": level.get("level"),
                        "reference": level.get("reference", metric.get("reference")),
                        "got": level.get("value"),
                        "err_pct": level.get("err_pct"),
                        "tol_pct": level.get("tol_pct"),
                        "gated": level.get("gated"),
                        "pass": level.get("pass"),
                    }
                )
        cases.append(
            {
                "case": case["id"],
                "slot": case["slot"],
                "title": case["title"],
                "status": verdict.get("status"),
                "machine_pass": verdict.get("machine_pass"),
                "reason": verdict.get("reason"),
                "repro_cmd": verdict.get("repro_cmd"),
                "artifact": verdict.get("artifact"),
                "artifact_sha256": verdict.get("artifact_sha256"),
                "references": [
                    {k: r.get(k) for k in ("id", "quantity", "value", "formula",
                                           "origin", "source", "source_sha256", "token")}
                    for r in verdict.get("references", [])
                ],
                "rows": rows,
                "convergence": verdict.get("convergence", []),
                "unresolved_metrics": verdict.get("unresolved_metrics", []),
            }
        )
    counts = {s: sum(1 for c in cases if c["status"] == s)
              for s in ("passed", "failed", "pending")}
    return {
        "kind": "tensorlbm.validation_matrix.benchmark_table/v1",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "matrix_version": matrix["version"],
        "gate": ("|err| <= tol at every gated level AND the level series converges "
                 "(span <= declared max_span_pct; monotone where required)"),
        "counts": counts,
        "cases": cases,
    }


def render_md(table: dict) -> str:
    c = table["counts"]
    lines = [
        "# Validation matrix — reference vs got (generated, zero hand-copy)",
        "",
        f"generated: {table['generated_utc']} | matrix v{table['matrix_version']} | "
        f"**passed {c['passed']} / failed {c['failed']} / pending {c['pending']}**",
        "",
        f"Gate: {table['gate']}",
        "",
        "Every number below is read from the case artifact and judged by "
        "`criteria.py`; the reference column carries its in-repo anchor and the "
        "sha256 of the file the anchor was read from.",
        "",
        "## Per-level comparison",
        "",
        "| case | slot | metric | level | reference | got | err % | tol % | pass | status |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for case in table["cases"]:
        if not case["rows"]:
            lines.append(
                f"| `{case['case']}` | {case['slot']} | — | — | — | — | — | — | — | "
                f"**{case['status']}** |"
            )
            continue
        for row in case["rows"]:
            lines.append(
                "| `{case}` | {slot} | {metric} | {level} | {ref} | {got} | {err} | "
                "{tol} | {ok} | **{status}** |".format(
                    case=case["case"], slot=case["slot"], metric=row["metric"],
                    level=row["level"], ref=_fmt(row["reference"]), got=_fmt(row["got"]),
                    err=_fmt(row["err_pct"]), tol=_fmt(row["tol_pct"]),
                    ok="yes" if row["pass"] else "NO", status=case["status"],
                )
            )

    lines += ["", "## Convergence (span across levels)", "",
              "| case | metric | errors | span % | max span % | monotone | pass |",
              "|---|---|---|---|---|---|---|"]
    for case in table["cases"]:
        for conv in case["convergence"]:
            lines.append(
                "| `{case}` | {metric} | {errs} | {span} | {maxspan} | {mono} | {ok} |".format(
                    case=case["case"], metric=conv["metric"],
                    errs=" → ".join(_fmt(e) for e in conv.get("errors_abs_pct", [])),
                    span=_fmt(conv.get("span_pct")), maxspan=_fmt(conv.get("max_span_pct")),
                    mono="yes" if conv.get("monotone") else "no",
                    ok="yes" if conv.get("pass") else "NO",
                )
            )
    if not any(c["convergence"] for c in table["cases"]):
        lines.append("| — | — | — | — | — | — | — |")

    lines += ["", "## Reference anchors", "",
              "| case | anchor id | quantity | value / formula | origin | source (sha256) |",
              "|---|---|---|---|---|---|"]
    for case in table["cases"]:
        for ref in case["references"]:
            lines.append(
                "| `{case}` | `{rid}` | {qty} | {val} | {origin} | `{src}` ({sha}) |".format(
                    case=case["case"], rid=ref["id"], qty=ref["quantity"],
                    val=_fmt(ref.get("value")) if ref.get("value") is not None
                    else _fmt(ref.get("formula")),
                    origin=(ref.get("origin") or "")[:90],
                    src=ref.get("source"), sha=(ref.get("source_sha256") or "—")[:16],
                )
            )

    lines += ["", "## Reproduce", ""]
    for case in table["cases"]:
        lines.append(f"- `{case['case']}` ({case['status']}): "
                     f"`{case['repro_cmd'] or 'no repro command declared'}`")
    lines += ["", "Artifacts and sha256: `benchmarks/validation_matrix/BENCHMARK_REF_VS_GOT.json`; "
              "run-level provenance: `artifacts/validation_matrix/runs/index.jsonl`.", ""]
    return "\n".join(lines)


def check_consistency(table: dict) -> list[str]:
    """A `passed` case must have resolvable, in-tolerance gated levels."""
    problems = []
    for case in table["cases"]:
        if case["status"] != "passed":
            continue
        if case["unresolved_metrics"]:
            problems.append(f"{case['case']}: passed but unresolved metrics "
                            f"{case['unresolved_metrics']}")
        if not case["rows"]:
            problems.append(f"{case['case']}: passed but has no comparison rows")
        for row in case["rows"]:
            if row["gated"] and not row["pass"]:
                problems.append(
                    f"{case['case']}/{row['metric']}/{row['level']}: passed but "
                    f"err {row['err_pct']}% > tol {row['tol_pct']}%"
                )
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="also fail if the table contradicts a fresh verdict")
    args = ap.parse_args(argv)

    table = build()
    JSON_OUT.write_text(json.dumps(table, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    MD_OUT.write_text(render_md(table), encoding="utf-8")
    c = table["counts"]
    print(f"wrote {JSON_OUT.relative_to(REPO_ROOT)} and {MD_OUT.relative_to(REPO_ROOT)} "
          f"({len(table['cases'])} cases, passed {c['passed']}/failed {c['failed']}/"
          f"pending {c['pending']})")
    if args.check:
        problems = check_consistency(table)
        for p in problems:
            print("  INCONSISTENT " + p)
        return 0 if not problems else 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())