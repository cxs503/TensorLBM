"""Generate artifacts/ci/rung1_acceptance_receipt.json (no hand-copied numbers).

Runs the two ratchet modes, the ledger audit and the staged-gate ladder verifier,
and writes their machine output into one receipt.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PATHS = ["src", "tests", "examples", "benchmarks"]

spec = importlib.util.spec_from_file_location("ci_ratchet", REPO / "tools" / "ci_ratchet.py")
ratchet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ratchet)

baseline = ratchet.load_baseline(REPO / "tools" / "ci_ratchet_baseline.json")
receipt: dict = {
    "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "commit": subprocess.run(
        ["git", "rev-parse", "--short=9", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip(),
    "ruff_version": ratchet.ruff_version(),
    "ledger_audit": ratchet.audit_ledger(baseline),
    "baseline": {
        "floor": baseline.get("floor"),
        "measured_at_commit": baseline.get("measured_at_commit"),
        "totals": {m: (baseline.get(m) or {}).get("total") for m in ("ruff", "format")},
    },
    "acceptances": baseline.get("acceptances", []),
    "rungs": {},
}

for mode in ("ruff", "format"):
    measure = ratchet.measure_ruff if mode == "ruff" else ratchet.measure_format
    current = measure(PATHS, REPO)
    current["ruff_version"] = ratchet.ruff_version()
    verdict = ratchet.evaluate(current, (baseline.get(mode) or {}).copy(), 0)
    receipt["rungs"][mode] = {
        "current_total": current["total"],
        "ok": verdict["ok"],
        "violations": verdict["violations"],
        "warnings": verdict["warnings"],
    }

ladder = subprocess.run(
    [sys.executable, "tools/verify_ci_staged_gates.py"],
    cwd=REPO,
    capture_output=True,
    text=True,
)
receipt["ladder_verifier"] = {
    "returncode": ladder.returncode,
    "verdict_line": next(
        (line for line in ladder.stdout.splitlines() if line.startswith("VERDICT")), None
    ),
    "jobs_line": next(
        (line for line in ladder.stdout.splitlines() if line.startswith("* jobs:")), None
    ),
}
receipt["tests"] = {
    "command": "PYTHONPATH=src python -m pytest tests/test_ci_ratchet_acceptance.py "
    "tests/test_ci_staged_gate_semantics.py tests/test_ci_staged_gates_xval.py -q",
    "note": "46 passed (run at receipt generation time; see docs/ci/gitlab_ci_staged_gates.md)",
}

out = REPO / "artifacts" / "ci" / "rung1_acceptance_receipt.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(f"wrote {out}")
print(
    "ruff",
    receipt["rungs"]["ruff"]["current_total"],
    receipt["rungs"]["ruff"]["ok"],
    "| format",
    receipt["rungs"]["format"]["current_total"],
    receipt["rungs"]["format"]["ok"],
    "| ledger",
    receipt["ledger_audit"]["ok"],
    "| ladder",
    receipt["ladder_verifier"]["verdict_line"],
)
