#!/usr/bin/env python3
"""Ratchet gate for repo-wide style debt (XVAL-LBM-GITLAB-CI-STAGED-GATES).

Why a ratchet instead of a plain ``ruff`` gate
----------------------------------------------
The ALT-2 staged:quality rung was declared *blocking* but was red by
construction: ``ruff check src tests examples benchmarks`` reports ~3k
findings and ``ruff format --check`` wants to rewrite ~2k files.  A blocking
first rung that can never pass makes every later rung unreachable, so the
whole ladder is decorative -- the exact anti-pattern the anchor targets.

Two ways out: (a) mass-fix the tree (huge, risky, unrelated diff), or
(b) turn the gate into a *ratchet*: the full rule set keeps its blocking
power ("no new debt"), while the legacy baseline is frozen in
``tools/ci_ratchet_baseline.json`` and may only shrink.  This tool implements
(b) and is what the ``lint`` / ``format-check`` jobs now run.

Semantics
---------
* ``ruff``   -- count of findings from ``ruff check`` over the given paths.
  Both the **total** and every **per-code** count must be ``<=`` the baseline
  (``--max-increase`` default 0), so fixing ``F401`` cannot pay for new
  ``E722``.
* ``format`` -- number of files ``ruff format --check`` would rewrite.  The
  count is ratcheted and the *set* is fingerprinted (sha256 of the sorted
  list): a swapped file with an unchanged count is reported as drift.

The baseline records the ``ruff`` version it was measured with, because rule
sets move between releases.  A version mismatch is surfaced as a warning in
the receipt -- it is never silently ignored.

Usage
-----
    python tools/ci_ratchet.py ruff   --paths src tests examples benchmarks
    python tools/ci_ratchet.py format --paths src tests examples benchmarks
    python tools/ci_ratchet.py ruff --write-baseline    # after a real cleanup

Exit codes: 0 = at or below baseline, 1 = regression, 2 = tool error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_BASELINE = Path(__file__).resolve().parent / "ci_ratchet_baseline.json"
DEFAULT_PATHS = ("src", "tests", "examples", "benchmarks")


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)  # noqa: S603


def ruff_version() -> str:
    exe = shutil.which("ruff")
    if not exe:
        return "absent"
    return _run([exe, "--version"]).stdout.strip() or "unknown"


def measure_ruff(paths: list[str], cwd: Path) -> dict[str, Any]:
    """Count ``ruff check`` findings, overall and per rule code."""
    exe = shutil.which("ruff")
    if not exe:
        raise SystemExit("ruff not found on PATH")
    proc = _run([exe, "check", *paths, "--output-format", "json"], cwd=cwd)
    if proc.returncode not in (0, 1):
        raise SystemExit(f"ruff check failed to run (rc={proc.returncode}): {proc.stderr[:400]}")
    try:
        findings = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"could not parse ruff json output: {exc}") from exc
    codes = Counter(str(f.get("code") or "invalid-syntax") for f in findings)
    return {"kind": "ruff", "total": len(findings), "by_code": dict(sorted(codes.items()))}


def measure_format(paths: list[str], cwd: Path) -> dict[str, Any]:
    """Count files ``ruff format --check`` would reformat, and fingerprint them."""
    exe = shutil.which("ruff")
    if not exe:
        raise SystemExit("ruff not found on PATH")
    proc = _run([exe, "format", "--check", *paths], cwd=cwd)
    # rc=2 also happens when a legacy file cannot be parsed; ruff still lists
    # every file it *would* reformat, so the count stays usable (the parse
    # failures are counted separately as `parse_errors`).
    if proc.returncode not in (0, 1, 2):
        raise SystemExit(f"ruff format failed to run (rc={proc.returncode}): {proc.stderr[:400]}")
    unformatted = sorted(
        line.strip()
        for line in proc.stdout.splitlines()
        if line.strip().startswith("Would reformat:")
    )
    files = [line.split(":", 1)[1].strip() for line in unformatted]
    digest = hashlib.sha256("\n".join(files).encode()).hexdigest()[:16]
    parse_errors = sum(1 for line in proc.stderr.splitlines() if "Failed to parse" in line)
    return {
        "kind": "format",
        "total": len(files),
        "fingerprint": digest,
        "parse_errors": parse_errors,
    }


def load_baseline(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def save_baseline(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def evaluate(
    current: dict[str, Any], baseline: dict[str, Any] | None, max_increase: int = 0
) -> dict[str, Any]:
    """Compare a measurement against its baseline (pure function, unit-tested)."""
    result: dict[str, Any] = {
        "kind": current["kind"],
        "current": current,
        "baseline": baseline,
        "max_increase": max_increase,
        "violations": [],
        "warnings": [],
        "ok": True,
    }
    if not baseline:
        result["warnings"].append("no baseline recorded: first run, nothing to ratchet against")
        return result

    budget = int(baseline.get("total", 0)) + max_increase
    if current["total"] > budget:
        result["violations"].append(
            f"total {current['kind']} debt grew: {current['total']} > baseline {baseline.get('total')} "
            f"(+{current['total'] - int(baseline.get('total', 0))})"
        )
    elif current["total"] < int(baseline.get("total", 0)):
        result["warnings"].append(
            f"{current['kind']} debt shrank to {current['total']}: lower the baseline "
            f"({baseline.get('total')} -> {current['total']}) to lock the improvement in"
        )

    base_codes = baseline.get("by_code") or {}
    if base_codes:
        for code, count in current.get("by_code", {}).items():
            allowed = int(base_codes.get(code, 0)) + max_increase
            if count > allowed:
                result["violations"].append(
                    f"new/increased {current['kind']} violation {code}: {count} > {base_codes.get(code, 0)}"
                )
        for code, count in base_codes.items():
            if int(count) > 0 and current.get("by_code", {}).get(code, 0) == 0:
                result["warnings"].append(f"rule {code} is now clean: drop it from the baseline")

    base_fp = baseline.get("fingerprint")
    if base_fp and current.get("fingerprint") and base_fp != current["fingerprint"]:
        result["warnings"].append(
            "unformatted-file set changed (fingerprint drift) while the count stayed at "
            f"{current['total']}: one file was fixed and another regressed"
        )

    base_version = baseline.get("ruff_version")
    current_version = current.get("ruff_version")
    if base_version and current_version and base_version != current_version:
        result["warnings"].append(
            f"ruff version changed: baseline {base_version} vs current {current_version} -- "
            "rule-set drift can move counts; re-measure before trusting the verdict"
        )

    result["ok"] = not result["violations"]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("ruff", "format"))
    parser.add_argument("--paths", nargs="+", default=list(DEFAULT_PATHS))
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--max-increase", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)

    measure = measure_ruff if args.mode == "ruff" else measure_format
    current = measure(list(args.paths), args.cwd)
    current["ruff_version"] = ruff_version()
    current["paths"] = list(args.paths)

    baseline_doc = load_baseline(args.baseline)
    mode_baseline = (baseline_doc.get(args.mode) or {}).copy()
    if mode_baseline:
        mode_baseline.setdefault("ruff_version", baseline_doc.get("ruff_version"))

    if args.write_baseline:
        baseline_doc["ruff_version"] = current["ruff_version"]
        baseline_doc[args.mode] = current
        save_baseline(args.baseline, baseline_doc)
        print(f"baseline written to {args.baseline}: {args.mode} total={current['total']}")

    result = evaluate(current, mode_baseline, args.max_increase)
    result["baseline_path"] = str(args.baseline)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    label = f"{args.mode} ratchet"
    print(f"[{label}] current={current['total']} baseline={mode_baseline.get('total', '<none>')}")
    for warning in result["warnings"]:
        print(f"[{label}] warn: {warning}")
    for violation in result["violations"]:
        print(f"[{label}] FAIL: {violation}", file=sys.stderr)
    print(f"[{label}] {'PASS' if result['ok'] else 'FAIL'}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
