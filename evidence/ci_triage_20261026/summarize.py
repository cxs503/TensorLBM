"""Summarize locally captured CI triage results; run from the repository root.

Inputs are ignored workspace artifacts under artifacts/runs/ci-triage/. This
script does not run tests, download CI logs, or infer results for absent files.
The report describes this investigation, not a general CI acceptance gate.
"""

from __future__ import annotations

import ast
import collections
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

BASELINE = "ad3d19ae6e5f55f02ac691f6719d4a5ae68f4cdc"
MERGED = "49eabc4abc26bfdfe5805890e95fb41d180c95c5"
RAW = Path("artifacts/runs/ci-triage")
DEST = Path("evidence/ci_triage_20261026/summary.json")
MECHANICAL_FILES = [
    "benchmarks/verified/suboff_sail_resistance/gen_tau_ladder_report.py",
    "benchmarks/verified/suboff_sail_resistance/run_tau_ladder.py",
    "benchmarks/verified/suboff_sail_resistance/tau_anchor_probe.py",
    "tests/test_audit_import_graph.py",
    "tests/test_tau_ladder_instrument.py",
]


class Normalize(ast.NodeTransformer):
    def visit_JoinedStr(self, node):
        node = self.generic_visit(node)
        if all(isinstance(v, ast.Constant) and isinstance(v.value, str) for v in node.values):
            return ast.Constant(value="".join(v.value for v in node.values))
        return node

    def visit_ImportFrom(self, node):
        node.names.sort(key=lambda alias: (alias.name, alias.asname or ""))
        return node

    def visit_Module(self, node):
        node = self.generic_visit(node)
        imports = [n for n in node.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        other = [n for n in node.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        node.body = sorted(imports, key=lambda n: ast.dump(n, include_attributes=False)) + other
        return node


def results(name):
    data = {}
    for case in ET.parse(RAW / name).getroot().iter("testcase"):
        failure, error, skip = case.find("failure"), case.find("error"), case.find("skipped")
        if failure is not None:
            status = "failed"
        elif error is not None:
            status = "error"
        else:
            status = "skipped" if skip is not None else "passed"
        detail = failure if failure is not None else error
        data[case.get("classname") + "::" + case.get("name")] = {
            "status": status,
            "message": detail.get("message", "").splitlines()[0][:350]
            if detail is not None
            else "",
        }
    return data


def category(key, message):
    if any(
        part in key
        for part in (
            "test_frontend_",
            "test_i18n.",
            "test_i18n::",
            "test_platform_basic",
            "TestSuboffI18nKeys",
        )
    ):
        return "frontend_contract_or_missing_build"
    if "trapezoid" in message:
        return "local_numpy_1_compatibility"
    for part, label in [
        ("test_generic_sim_api", "local_torch_compile_backend_cache"),
        ("TestSuboffPostprocessEndpoints", "job_fixture_constructor_contract"),
        ("test_study_compare", "study_metadata_or_result_aggregation"),
        ("TestDoEAPI", "study_metadata_or_result_aggregation"),
        ("TestLiveMetricsAPI", "live_metrics_route_contract"),
        ("test_simulations_generic_run_api", "test_double_to_dict_signature"),
        ("TestBoundaryLayerAtStation", "boundary_layer_yplus_early_return"),
        ("TestSuboffSolveEndpoint", "hull_type_input_validation"),
    ]:
        if part in key:
            return label
    raise ValueError(f"Unclassified failure: {key}")


def lint(name):
    return [
        {
            "file": x["filename"].split("benchmarks/", 1)[-1],
            "rule": x["code"],
            "line": x["location"]["row"],
        }
        for x in json.loads((RAW / name).read_text())
    ]


def main():
    proof = []
    for path in MECHANICAL_FILES:
        before = subprocess.check_output(["git", "show", MERGED + ":" + path]).decode()
        after = Path(path).read_text()
        normalized = [
            ast.dump(Normalize().visit(ast.parse(text)), include_attributes=False)
            for text in (before, after)
        ]
        assert normalized[0] == normalized[1], path
        proof.append({"path": path, "normalized_ast_equal": True})
    baseline = results("platform-baseline.xml")
    current = results("platform-current.xml")
    after = results("platform-after.xml")
    assert {k: v["status"] for k, v in baseline.items()} == {
        k: v["status"] for k, v in current.items()
    }
    fixed = [
        k
        for k in current
        if current[k]["status"] in {"failed", "error"} and after[k]["status"] == "passed"
    ]
    regressions = [
        k for k in current if current[k]["status"] == "passed" and after[k]["status"] != "passed"
    ]
    assert len(fixed) == 7 and not regressions
    remaining = [
        {"node": key, **value, "category": category(key, value["message"])}
        for key, value in after.items()
        if value["status"] in {"failed", "error"}
    ]
    assert lint("ruff-baseline.json") == lint("ruff-current.json")
    summary = {
        "date": "2026-10-26",
        "baseline_commit": BASELINE,
        "merged_commit": MERGED,
        "environment_note": "CPU Python 3.11.2 / torch 2.2.2+cu121 / numpy 1.26.4; "
        "same environment for both revisions and fixes, not exact remote CI.",
        "platform_counts": {
            name: dict(collections.Counter(v["status"] for v in data.values()))
            for name, data in [
                ("baseline", baseline),
                ("merged_before_fix", current),
                ("working_tree_after_fix", after),
            ]
        },
        "baseline_vs_merged_changed_outcomes": [],
        "fixed_nodes": fixed,
        "new_regressions": regressions,
        "remaining_categories": dict(collections.Counter(r["category"] for r in remaining)),
        "remaining_failures": remaining,
        "baseline_and_merged_lint_findings": lint("ruff-current.json"),
        "mechanical_ast_proof": proof,
        "proof_scope": "Import ordering and constant-only f-strings normalized; "
        "not proof of arbitrary import-order side-effect equivalence.",
        "raw_artifacts": [],
    }
    for path in sorted(RAW.glob("*")):
        if path.suffix not in {".log", ".xml", ".json", ".txt"} or path.name.startswith("install"):
            continue
        raw = path.read_bytes()
        summary["raw_artifacts"].append(
            {
                "path": str(path),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "storage": "ignored local workspace output; not committed",
            }
        )
    DEST.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary["platform_counts"], indent=2))
    print(json.dumps(summary["remaining_categories"], indent=2))


if __name__ == "__main__":
    main()
