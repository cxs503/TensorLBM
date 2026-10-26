"""Summarize same-environment JUnit evidence; raw artifacts are not committed."""

from __future__ import annotations

import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RAW = REPO / "artifacts/runs/platform-contract-fixes"
CONTRACT = "test_platform_contract_regressions"


def read_cases(path: Path) -> dict[str, dict[str, str]]:
    cases = {}
    for node in ET.parse(path).iter("testcase"):
        key = node.attrib["classname"] + "::" + node.attrib["name"]
        assert key not in cases, key
        status = "passed"
        message = ""
        for kind in ("error", "failure", "skipped"):
            child = node.find(kind)
            if child is not None:
                status = kind
                message = child.get("message", "")
                break
        cases[key] = {"status": status, "message": message}
    return cases


def counts(cases: dict) -> dict:
    return dict(sorted(Counter(case["status"] for case in cases.values()).items()))


def main() -> None:
    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RAW
    baseline_all = read_cases(raw / "platform-baseline.xml")
    current_all = read_cases(raw / "platform-final.xml")
    baseline = {k: v for k, v in baseline_all.items() if CONTRACT not in k}
    current = {k: v for k, v in current_all.items() if CONTRACT not in k}
    baseline_new = read_cases(raw / "baseline-contract-final.xml")
    current_new = {k: v for k, v in current_all.items() if CONTRACT in k}
    assert baseline.keys() == current.keys(), "Original test inventory changed"
    assert baseline_new.keys() == current_new.keys(), "Contract test inventory changed"
    recovered = [
        k
        for k in baseline
        if baseline[k]["status"] in {"failure", "error"} and current[k]["status"] == "passed"
    ]
    regressions = [
        k
        for k in baseline
        if baseline[k]["status"] == "passed" and current[k]["status"] != "passed"
    ]
    newly_skipped = [
        k
        for k in baseline
        if baseline[k]["status"] != "skipped" and current[k]["status"] == "skipped"
    ]
    remaining = {k: v for k, v in current.items() if v["status"] in {"failure", "error"}}
    files = [
        "platform-baseline.xml",
        "platform-final.xml",
        "baseline-contract-final.xml",
        "platform-baseline.log",
        "platform-final.log",
        "baseline-contract-final.log",
        "environment.txt",
        "governance.xml",
    ]
    result = {
        "baseline_revision": "90332d1bc24b4aeffd58bc40dfd62e4a876562c3",
        "scope": "CPU sandbox, same venv; not remote CI or physical/GPU acceptance",
        "original_case_count": len(baseline),
        "baseline_original": counts(baseline),
        "current_original": counts(current),
        "baseline_new_contract_tests": counts(baseline_new),
        "current_new_contract_tests": counts(current_new),
        "current_full_platform": counts(current_all),
        "recovered_original_cases": sorted(recovered),
        "passed_to_nonpassed": regressions,
        "newly_skipped": newly_skipped,
        "remaining_failures": remaining,
        "raw_artifact_sha256": {
            name: hashlib.sha256((raw / name).read_bytes()).hexdigest() for name in files
        },
    }
    assert not regressions
    assert not newly_skipped
    assert counts(current_new) == {"passed": 61}
    assert len(recovered) == 17
    output = Path(__file__).with_name("summary.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                not in {"remaining_failures", "recovered_original_cases", "raw_artifact_sha256"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
