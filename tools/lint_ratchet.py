#!/usr/bin/env python3
"""Lint/format debt ratchet for TensorLBM.

Why this exists
---------------
``ruff check src tests examples benchmarks`` currently reports a large,
pre-existing backlog (legacy debt accumulated over ~200 merged missions), so a
plain ``ruff check`` gate is red for reasons unrelated to the change under
review and therefore gets ignored.  A ratchet converts "must be zero" into
"must never get worse": the measured count is compared against a checked-in
baseline (``tools/lint_baseline.json``) and the job fails only on regression.
The baseline may only move down (``--update`` refuses to raise it without
``--force``), so the debt is monotonically repaid instead of frozen.

Usage
-----
    python tools/lint_ratchet.py            # same as --check
    python tools/lint_ratchet.py --check    # exit 1 if any metric regressed
    python tools/lint_ratchet.py --update   # record a *lower* baseline
    python tools/lint_ratchet.py --json     # machine-readable report

Exit codes: 0 = within baseline, 1 = regression, 2 = tooling error.

The reported numbers are exactly the ones a human gets from
``ruff check <scope>`` / ``ruff format --check <scope>``: one line per
finding, one line per file that would be reformatted.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "tools" / "lint_baseline.json"

#: Lint/format scope, mirroring the ``lint`` job in .github/workflows/ci.yml.
SCOPE: tuple[str, ...] = ("src", "tests", "examples", "benchmarks")


@dataclass(frozen=True)
class Metric:
    """One ratcheted number: current measurement, baseline and direction."""

    name: str
    current: int
    baseline: int

    @property
    def regressed(self) -> bool:
        return self.current > self.baseline

    @property
    def improvement(self) -> int:
        """How many findings could be shaved off the baseline (>= 0)."""
        return max(self.baseline - self.current, 0)


def _ruff() -> str:
    exe = shutil.which("ruff")
    if exe is None:
        raise SystemExit(
            "lint_ratchet: `ruff` not found on PATH. Install the dev extra "
            "(`pip install -e '.[dev]'`) before running the ratchet."
        )
    return exe


def ruff_version() -> str:
    out = subprocess.run(
        [_ruff(), "--version"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return out.stdout.strip() or "unknown"


def count_lint(scope: tuple[str, ...] = SCOPE) -> int:
    """Number of findings from ``ruff check`` (the ledger metric)."""
    out = subprocess.run(
        [_ruff(), "check", *scope, "--output-format=json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if not out.stdout.strip():
        raise SystemExit(f"lint_ratchet: ruff check produced no output.\n{out.stderr[-2000:]}")
    try:
        findings = json.loads(out.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"lint_ratchet: cannot parse ruff json output: {exc}") from exc
    return len(findings)


def count_syntax_errors(scope: tuple[str, ...] = SCOPE) -> int:
    """Unparsable files reported by ruff (``invalid-syntax``).

    They are *not* ruff-fixable, so they must never be allowed to grow at all:
    an unparsable committed script is dead code that can never run.
    """
    out = subprocess.run(
        [_ruff(), "check", *scope, "--output-format=json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if not out.stdout.strip():
        return 0
    findings = json.loads(out.stdout)
    return sum(
        1
        for f in findings
        if "invalid-syntax" in str(f.get("code", ""))
        or "invalid-syntax" in str(f.get("message", ""))
    )


def count_unformatted(scope: tuple[str, ...] = SCOPE) -> int:
    """Number of files ``ruff format --check`` would rewrite."""
    out = subprocess.run(
        [_ruff(), "format", "--check", *scope],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return sum(1 for line in out.stdout.splitlines() if line.startswith("Would reformat:"))


def load_baseline(path: Path | None = None) -> dict[str, Any]:
    """Read the baseline document (resolved at call time, so it can be relocated)."""
    resolved = BASELINE_PATH if path is None else path
    return json.loads(resolved.read_text(encoding="utf-8"))


def measure(scope: tuple[str, ...] = SCOPE) -> dict[str, int]:
    return {
        "ruff_check_findings": count_lint(scope),
        "ruff_format_unformatted_files": count_unformatted(scope),
        "invalid_syntax_findings": count_syntax_errors(scope),
    }


def evaluate(current: dict[str, int], baseline: dict[str, int]) -> list[Metric]:
    """Compare measurements with baselines; unknown keys default to 0."""
    return [
        Metric(name, current.get(name, 0), int(baseline.get(name, 0))) for name in sorted(baseline)
    ]


def _print_report(metrics: list[Metric], version: str, recorded: str | None) -> None:
    print(f"ruff {version}")
    if recorded and recorded != version:
        print(
            f"NOTE: baseline was recorded with {recorded}; rule coverage differs between "
            "ruff releases, so a large delta may be tool drift rather than new debt."
        )
    print(f"{'metric':<34}{'current':>9}{'baseline':>10}{'delta':>8}  status")
    for m in metrics:
        delta = m.current - m.baseline
        status = "REGRESSED" if m.regressed else ("ok" if delta == 0 else "improved")
        print(f"{m.name:<34}{m.current:>9}{m.baseline:>10}{delta:>+8}  {status}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true", help="fail on regression (default)")
    parser.add_argument(
        "--update", action="store_true", help="write measured counts as the new baseline"
    )
    parser.add_argument(
        "--force", action="store_true", help="with --update: allow raising a baseline"
    )
    parser.add_argument("--json", action="store_true", help="emit a machine-readable report")
    parser.add_argument(
        "--scope",
        nargs="+",
        default=list(SCOPE),
        help=f"paths to lint (default: {' '.join(SCOPE)})",
    )
    args = parser.parse_args(argv)
    scope = tuple(args.scope)

    version = ruff_version()
    baseline_doc = load_baseline() if BASELINE_PATH.exists() else {}
    baseline: dict[str, int] = dict(baseline_doc.get("baseline", {}))
    current = measure(scope)

    if args.update:
        merged = dict(baseline)
        raised = []
        for name, value in current.items():
            previous = int(baseline.get(name, value))
            if value > previous:
                raised.append(f"{name}: {previous} -> {value}")
            merged[name] = value if (args.force or value <= previous) else previous
        if raised and not args.force:
            print("lint_ratchet: refusing to raise baseline (debt must only shrink):")
            for line in raised:
                print(f"  {line}")
            return 1
        baseline_doc["baseline"] = merged
        baseline_doc["measured_with"] = version
        baseline_doc["scope"] = list(scope)
        BASELINE_PATH.write_text(
            json.dumps(baseline_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        try:
            shown = BASELINE_PATH.relative_to(REPO_ROOT)
        except ValueError:  # baseline relocated (tests point it at tmp_path)
            shown = BASELINE_PATH
        print(f"lint_ratchet: baseline updated -> {shown}")
        for name in sorted(merged):
            print(f"  {name} = {merged[name]}")
        return 0

    metrics = evaluate(current, baseline) if baseline else []
    if not metrics:  # pragma: no cover - first run bootstrap
        print("lint_ratchet: no baseline recorded yet; run `--update` to seed it.")
        _print_report([Metric(n, v, v) for n, v in sorted(current.items())], version, None)
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "ruff_version": version,
                    "baseline_recorded_with": baseline_doc.get("measured_with"),
                    "scope": list(scope),
                    "metrics": [
                        {
                            "name": m.name,
                            "current": m.current,
                            "baseline": m.baseline,
                            "delta": m.current - m.baseline,
                            "regressed": m.regressed,
                        }
                        for m in metrics
                    ],
                },
                indent=2,
            )
        )
    else:
        _print_report(metrics, version, baseline_doc.get("measured_with"))

    regressed = [m for m in metrics if m.regressed]
    improved = [m for m in metrics if m.improvement > 0]
    for m in improved:
        print(
            f"ratchet available: {m.name} is {m.improvement} below baseline "
            f"({m.current} vs {m.baseline}); run `--update` to lock the gain in."
        )
    if regressed:
        print(
            "lint_ratchet: FAIL — new lint/format debt was introduced "
            "(fix it, or land the fix together with the change that caused it)."
        )
        return 1
    print("lint_ratchet: OK — no new lint/format debt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
