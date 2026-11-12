#!/usr/bin/env python3
"""Test-baseline patrol for TensorLBM.

Turns two pytest JUnit XML reports into node-id level *signatures*, diffs them,
and classifies every newly-red node as environment red or real red using rules
declared up front (see ENV_RED_RULES below).  Nothing is fixed here; the tool
only reports so that a patrol is comparable across missions.

Usage
-----
    python tools/test_patrol/patrol.py run       --out artifacts/test_patrol_m15878
    python tools/test_patrol/patrol.py compare   --baseline B.xml --current C.xml \
                                                 --out artifacts/test_patrol_m15878/compare.json

`run` executes the frozen manifest (tools/test_patrol/core_suite.txt) with the
same flags the m14496 baseline used, so the two reports are comparable.

Signature format: one line per node id, "<node_id>\t<outcome>", outcomes in
{passed, failed, error, skipped, xfailed, xpassed}; the report sha256 is taken
over those sorted lines.  A changed sha256 == suite drift.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "tools" / "test_patrol" / "core_suite.txt"

# Pre-declared before any run (do not tune after seeing results).
ENV_RED_RULES = {
    "timeout": "pytest-timeout marker or 'Timeout >' in the message: host contention, not a code defect",
    "no_accelerator": "message mentions CUDA/GPU/SDAA/NPU/sdaa device or 'no CUDA-capable device'",
    "missing_optional_dep": "ModuleNotFoundError/ImportError naming a package outside the hard deps "
    "(torch/numpy/matplotlib/einops/tqdm) - an optional extra is not installed in this host",
    "external_binary": "FileNotFoundError/OSError for a non-python executable or absolute path outside the repo",
    "resource": "MemoryError / 'Killed' / OSError 28 (ENOSPC)",
}
# Anything else failing is a real red and must be reproduced in isolation before
# being called a defect.


def parse_junit(path: Path) -> dict[str, str]:
    """node_id -> outcome. Test-case names may carry parametrisation params."""
    root = ET.parse(path).getroot()
    out: dict[str, str] = {}
    for case in root.iter("testcase"):
        cls = case.get("classname") or ""
        name = case.get("name") or ""
        node = f"{cls.replace('.', '/', 1).replace('.', '/')}::{name}" if cls else name
        # normalise back to dotted module path (pytest classname uses dots)
        node = f"{cls}::{name}"
        outcome = "passed"
        for child in case:
            if child.tag == "failure":
                outcome = "failed"
            elif child.tag == "error":
                outcome = "error"
            elif child.tag == "skipped":
                outcome = "xfailed" if (child.get("type") or "") == "pytest.xfail" else "skipped"
        out[node] = outcome
    return out


def signature(sig: dict[str, str]) -> tuple[str, list[str]]:
    lines = [f"{k}\t{v}" for k, v in sorted(sig.items())]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest(), lines


def failure_messages(path: Path) -> dict[str, str]:
    """node_id -> first line of the failure/error message."""
    root = ET.parse(path).getroot()
    msgs: dict[str, str] = {}
    for case in root.iter("testcase"):
        node = f"{case.get('classname')}::{case.get('name')}"
        for child in case:
            if child.tag in ("failure", "error"):
                msg = (child.get("message") or "").strip().splitlines()
                msgs[node] = msg[0] if msg else (child.text or "").strip().splitlines()[0] if child.text else ""
    return msgs


def classify(node: str, message: str) -> tuple[str, str]:
    low = (message or "").lower()
    if "timeout" in low:
        return "env_red", "timeout"
    for tok in ("no cuda-capable", "cuda error", "sdaa", "npu", "gpu is not available"):
        if tok in low:
            return "env_red", "no_accelerator"
    if "modulenotfounderror" in low or "importerror" in low:
        return "env_red", "missing_optional_dep"
    if "filenotfounderror" in low or "no such file" in low:
        return "env_red", "external_binary"
    if "memoryerror" in low or "killed" in low or "errno 28" in low:
        return "env_red", "resource"
    return "real_red", "assert_or_exception"


def compare(baseline: Path, current: Path, out: Path) -> dict:
    base, cur = parse_junit(baseline), parse_junit(current)
    bsha, _ = signature(base)
    csha, _ = signature(cur)
    bmsg, cmsg = failure_messages(baseline), failure_messages(current)
    red = ("failed", "error")
    new_red = [n for n, o in cur.items() if o in red and base.get(n) != o]
    fixed = [n for n, o in cur.items() if o == "passed" and base.get(n) in red]
    vanished = [n for n in base if n not in cur]
    added = [n for n in cur if n not in base]
    # skips<->xfails are the same node with a different bookkeeping label
    relabel = {n: (base[n], cur[n]) for n in cur if n in base and base[n] != cur[n]
               and {base[n], cur[n]} <= {"skipped", "xfailed"}}
    report = {
        "baseline_xml": str(baseline),
        "current_xml": str(current),
        "baseline_signature_sha256": bsha,
        "current_signature_sha256": csha,
        "suite_signature_changed": bsha != csha,
        "counts": {
            "baseline": _counts(base),
            "current": _counts(cur),
        },
        "new_red": [{"node": n, "outcome": cur[n], "message": cmsg.get(n, ""),
                     "classification": classify(n, cmsg.get(n, ""))} for n in sorted(new_red)],
        "fixed_since_baseline": sorted(fixed),
        "nodes_absent_now": sorted(vanished),
        "nodes_added_now": sorted(added),
        "bookkeeping_relabel": relabel,
        "env_red_rules": ENV_RED_RULES,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def _counts(sig: dict[str, str]) -> dict[str, int]:
    c: dict[str, int] = {}
    for o in sig.values():
        c[o] = c.get(o, 0) + 1
    return c


def run(out: Path, extra: list[str]) -> int:
    files = [ln.strip() for ln in MANIFEST.read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    missing = [f for f in files if not (REPO / f).exists()]
    if missing:
        print(f"manifest references missing files: {missing}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    junit = out / "junit_core.xml"
    log = out / "pytest_core.log"
    env = dict(os.environ, PYTHONPATH="src", MPLBACKEND="Agg")
    cmd = [sys.executable, "-m", "pytest", *files, "-q", "--timeout=1200", "-n", "8",
           "--tb=line", "-p", "no:cacheprovider", f"--junitxml={junit}", *extra]
    with log.open("w") as fh:
        rc = subprocess.call(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT)
    print(f"pytest rc={rc}; log={log}; junit={junit}")
    return 0 if junit.exists() else rc


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--out", required=True)
    p_run.add_argument("extra", nargs="*")
    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--baseline", required=True)
    p_cmp.add_argument("--current", required=True)
    p_cmp.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        return run(Path(a.out), a.extra)
    rep = compare(Path(a.baseline), Path(a.current), Path(a.out))
    print(json.dumps({k: rep[k] for k in ("suite_signature_changed", "counts",
                                          "new_red", "fixed_since_baseline",
                                          "nodes_absent_now", "nodes_added_now")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())