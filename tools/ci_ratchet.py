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
    python tools/ci_ratchet.py ruff   --accept-drift --ref <sha> --reason "..."
    python tools/ci_ratchet.py format --accept-drift --ref <sha> --reason "..."
    python tools/ci_ratchet.py ruff --audit       # verify the acceptance chain

Acceptance ledger
-----------------
On a shared trunk the aggregate baseline can go red through debt that the
current change did not write.  Raising the baseline silently would ratify
that debt, so an increase is only allowed through ``--accept-drift``, which
re-measures the tree at the baseline commit (``--ref``, a ``git archive``
export, no git state is touched) and refuses the update unless *every*
increase is attributable to a file that did not exist at that commit.
Growth inside a pre-existing file is always refused: fix it instead.
Each accepted increase is appended to ``acceptances`` in the baseline file
with the ref pair, the reason, the per-rule delta, the newly-added files that
carry the debt and the commit that added each of them, so cumulative signed
drift stays auditable (``--audit``) and nameable (each entry is a follow-up
ticket for the owner of the introducing commit).

Exit codes: 0 = at or below baseline, 1 = regression, 2 = tool error.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
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
    per_file: Counter[str] = Counter()
    for finding in findings:
        per_file[_rel(str(finding.get("filename") or ""), cwd)] += 1
    return {
        "kind": "ruff",
        "total": len(findings),
        "by_code": dict(sorted(codes.items())),
        "files": dict(sorted(per_file.items())),
    }


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
        "files": dict(sorted({_rel(f, cwd): 1 for f in files}.items())),
    }


def load_baseline(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def save_baseline(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


BASELINE_KEYS = ("kind", "total", "by_code", "fingerprint", "parse_errors", "paths", "ruff_version")


def _rel(path: str, cwd: Path) -> str:
    """Repo-relative form of a tool-reported path (ruff prints absolute paths)."""
    try:
        return str(Path(path).resolve().relative_to(Path(cwd).resolve()))
    except (ValueError, OSError):
        return path


def strip_for_baseline(measurement: dict[str, Any]) -> dict[str, Any]:
    """Drop the per-file detail: the baseline stores counts, not the file list."""
    return {k: measurement[k] for k in BASELINE_KEYS if k in measurement}


def export_ref(ref: str, cwd: Path, dest: Path) -> Path:
    """Materialise the tree at ``ref`` into ``dest`` without touching git state."""
    proc = subprocess.run(  # noqa: S603
        ["git", "archive", ref], cwd=cwd, capture_output=True
    )
    if proc.returncode != 0:
        raise SystemExit(f"git archive {ref} failed: {proc.stderr.decode()[:300]}")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tar:
        tar.extractall(dest)  # noqa: S202 - archive content is our own repo at a commit
    return dest


def paths_at_ref(ref: str, cwd: Path, prefixes: Iterable[str]) -> set[str]:
    """Every tracked path that exists at ``ref`` under the ratcheted prefixes."""
    proc = _run(["git", "ls-tree", "-r", "--name-only", ref], cwd=cwd)
    if proc.returncode != 0:
        raise SystemExit(f"git ls-tree {ref} failed: {proc.stderr[:300]}")
    wanted = tuple(p.rstrip("/") for p in prefixes)
    return {
        name
        for name in proc.stdout.splitlines()
        if name and any(name == w or name.startswith(w + "/") for w in wanted)
    }


def classify_growth(
    current: dict[str, Any], reference: dict[str, Any], known_paths: set[str]
) -> dict[str, Any]:
    """Partition a debt increase into new-file drift and pre-existing regressions.

    Pure function.  ``known_paths`` are the paths that already existed at the
    baseline commit, so a file absent from the reference measurement but present
    in ``known_paths`` is a pre-existing file whose count went from 0 to n: that
    is a regression, not drift.
    """
    cur_files = current.get("files") or {}
    ref_files = reference.get("files") or {}
    new_debt: dict[str, int] = {}
    regressions: dict[str, dict[str, int]] = {}
    fixed: dict[str, int] = {}
    for name, count in cur_files.items():
        before = int(ref_files.get(name, 0))
        if name not in known_paths:
            new_debt[name] = count
        elif count > before:
            regressions[name] = {"at_ref": before, "now": count}
    for name, count in ref_files.items():
        now = int(cur_files.get(name, 0))
        if now < count:
            fixed[name] = {"at_ref": count, "now": now}
    return {
        "ref_total": reference.get("total"),
        "current_total": current.get("total"),
        "new_debt_files": dict(sorted(new_debt.items())),
        "pre_existing_regressions": dict(sorted(regressions.items())),
        "fixed_files": dict(sorted(fixed.items())),
        "ok": not regressions,
    }


def audit_ledger(doc: dict[str, Any]) -> dict[str, Any]:
    """Verify that the acceptance chain explains every baseline increase (pure)."""
    problems: list[str] = []
    acceptances = doc.get("acceptances") or []
    floor = doc.get("floor") or {}
    summary: dict[str, Any] = {"n_acceptances": len(acceptances), "total_delta": 0, "modes": {}}
    for entry in acceptances:
        for key in (
            "mode",
            "ref_from",
            "ref_to",
            "recorded_at",
            "reason",
            "from_total",
            "to_total",
        ):
            if key not in entry:
                problems.append(f"acceptance entry missing {key}: {entry}")
        if not str(entry.get("reason") or "").strip():
            problems.append(f"acceptance entry has an empty reason: {entry.get('ref_to')}")
        delta = int(entry.get("to_total", 0)) - int(entry.get("from_total", 0))
        if delta < 0:
            problems.append(
                f"acceptance lowered the baseline ({entry.get('mode')}: "
                f"{entry.get('from_total')} -> {entry.get('to_total')}); use --write-baseline"
            )
        summary["total_delta"] += delta
        per_mode = summary["modes"].setdefault(
            entry.get("mode"), {"chain": [], "delta": 0, "floor": floor.get(entry.get("mode"))}
        )
        per_mode["chain"].append([int(entry.get("from_total", 0)), int(entry.get("to_total", 0))])
        per_mode["delta"] += delta
    for mode, payload in summary["modes"].items():
        chain = payload["chain"]
        base_floor = payload["floor"]
        if base_floor is not None and chain and chain[0][0] != int(base_floor):
            problems.append(
                f"{mode}: first acceptance starts at {chain[0][0]}, floor is {base_floor}"
            )
        for prev, nxt in zip(chain, chain[1:]):
            if prev[1] != nxt[0]:
                problems.append(f"{mode}: chain break {prev[1]} -> {nxt[0]}")
        current_baseline = int((doc.get(mode) or {}).get("total", -1))
        if chain and chain[-1][1] != current_baseline:
            problems.append(
                f"{mode}: last accepted total {chain[-1][1]} != baseline {current_baseline}"
            )
        if base_floor is not None and current_baseline >= 0:
            if current_baseline - int(base_floor) != payload["delta"]:
                problems.append(
                    f"{mode}: baseline {current_baseline} - floor {base_floor} != signed drift "
                    f"{payload['delta']}"
                )
    summary["problems"] = problems
    summary["ok"] = not problems
    return summary


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


def _head(cwd: Path) -> str:
    proc = _run(["git", "rev-parse", "--short=9", "HEAD"], cwd=cwd)
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def _introduce_commits(files: Iterable[str], ref: str, cwd: Path) -> dict[str, str]:
    """Map each newly-added file to the commit (and subject) that added it."""
    out: dict[str, str] = {}
    for name in files:
        proc = _run(
            ["git", "log", "--diff-filter=A", "--format=%h %s", "-1", f"{ref}..HEAD", "--", name],
            cwd=cwd,
        )
        out[name] = proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() else "?"
    return out


def _code_delta(before: dict[str, int], now: dict[str, int]) -> dict[str, int]:
    delta = {
        code: int(now.get(code, 0)) - int(before.get(code, 0)) for code in set(before) | set(now)
    }
    return dict(sorted((c, d) for c, d in delta.items() if d))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("ruff", "format"))
    parser.add_argument("--paths", nargs="+", default=list(DEFAULT_PATHS))
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--accept-drift", action="store_true")
    parser.add_argument("--reason", default="", help="why the increase is acceptable")
    parser.add_argument("--ref", default=None, help="baseline commit to re-measure")
    parser.add_argument("--max-accept-increase", type=int, default=50)
    parser.add_argument("--audit", action="store_true", help="verify the acceptance chain")
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

    if args.audit:
        ledger = audit_ledger(baseline_doc)
        for entry_mode, payload in ledger["modes"].items():
            print(
                f"[audit] {entry_mode}: floor={payload['floor']} signed drift={payload['delta']} "
                f"chain={payload['chain']}"
            )
        for problem in ledger["problems"]:
            print(f"[audit] FAIL: {problem}", file=sys.stderr)
        print(
            f"[audit] {ledger['n_acceptances']} acceptances, {ledger['total_delta']:+d} signed"
            f" -> {'PASS' if ledger['ok'] else 'FAIL'}"
        )
        return 0 if ledger["ok"] else 1

    if args.accept_drift:
        ref = args.ref or baseline_doc.get("measured_at_commit")
        if not ref:
            print(
                "[accept] FAIL: no --ref and no measured_at_commit in the baseline", file=sys.stderr
            )
            return 2
        if len(args.reason.strip()) < 10:
            print(
                "[accept] FAIL: --reason is required (>=10 chars) for an accepted increase",
                file=sys.stderr,
            )
            return 2
        baseline_total = int(mode_baseline.get("total", 0))
        delta = int(current["total"]) - baseline_total
        if delta <= 0:
            print(
                f"[accept] nothing to accept: current {current['total']} <= baseline "
                f"{baseline_total}"
            )
            return 0
        if delta > args.max_accept_increase:
            print(
                f"[accept] FAIL: +{delta} exceeds --max-accept-increase {args.max_accept_increase}",
                file=sys.stderr,
            )
            return 1
        with tempfile.TemporaryDirectory(prefix="ci_ratchet_ref_") as tmp:
            ref_tree = export_ref(ref, args.cwd, Path(tmp) / "tree")
            reference = measure(list(args.paths), ref_tree)
            known = paths_at_ref(ref, args.cwd, args.paths)
            attribution = classify_growth(current, reference, known)
            attribution["ref_measured_total"] = reference["total"]
        attribution["introduced_by"] = _introduce_commits(
            attribution["new_debt_files"], ref, args.cwd
        )
        if not attribution["ok"]:
            print(
                "[accept] FAIL: debt grew inside pre-existing files -- fix them first:",
                file=sys.stderr,
            )
            for name, counts in attribution["pre_existing_regressions"].items():
                print(f"[accept]   {name}: {counts['at_ref']} -> {counts['now']}", file=sys.stderr)
            return 1
        entry = {
            "mode": args.mode,
            "ref_from": ref,
            "ref_to": _head(args.cwd),
            "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "reason": args.reason.strip(),
            "from_total": baseline_total,
            "to_total": int(current["total"]),
            "delta": delta,
            "by_code_delta": _code_delta(
                mode_baseline.get("by_code") or {}, current.get("by_code") or {}
            ),
            "new_debt_files": attribution["new_debt_files"],
            "introduced_by": attribution["introduced_by"],
            "ref_measured_total": attribution["ref_measured_total"],
            "fixed_files": attribution["fixed_files"],
        }
        baseline_doc.setdefault("floor", {}).setdefault(args.mode, baseline_total)
        baseline_doc["measured_at_commit"] = ref
        baseline_doc.setdefault("acceptances", []).append(entry)
        baseline_doc["ruff_version"] = current["ruff_version"]
        baseline_doc[args.mode] = strip_for_baseline(current)
        save_baseline(args.baseline, baseline_doc)
        print(
            f"[accept] recorded +{delta} for {args.mode} ({baseline_total} -> "
            f"{current['total']}) over {len(attribution['new_debt_files'])} newly-added files"
        )
        mode_baseline = dict(baseline_doc[args.mode])

    if args.write_baseline:
        baseline_doc["ruff_version"] = current["ruff_version"]
        baseline_doc["measured_at_commit"] = _head(args.cwd)
        baseline_doc[args.mode] = strip_for_baseline(current)
        save_baseline(args.baseline, baseline_doc)
        print(f"baseline written to {args.baseline}: {args.mode} total={current['total']}")

    result = evaluate(current, mode_baseline, args.max_increase)
    result["per_file"] = current.get("files", {})
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
