#!/usr/bin/env python3
"""Adjudicate entries of ``tests/ci_quarantine.txt`` by actually running them.

Why this exists
---------------
``tests/conftest.py`` rewrites every quarantined node id into a non-strict
``xfail``, so a quarantined test that really fails is *reported as xfail*, not
as a failure.  The list therefore only ever grows unless somebody runs the
tests with ``--runxfail`` and records what happens.  This tool performs that
adjudication mechanically and emits a machine-readable ledger:

    python3 tools/quarantine_adjudicate.py --out artifacts/ci_quarantine/adjudication.json

Each entry produces
    nodeid, returncode, verdict, signature, duration_s, timed_out

with ``verdict`` in

    PASS_NOW        the test passes today -> unquarantine candidate (real win)
    EMPTY_EVIDENCE  assertion shows an empty diagnostic/evidence payload
    API_DRIFT       exception type that indicates a moved/renamed interface
    NUMERIC         assertion on a numeric tolerance / magnitude
    THRESHOLD       assertion on a count or ordering requirement (non-numeric)
    ASSERT          any other assertion failure
    ERROR           collection / setup error
    TIMEOUT         exceeded --timeout
    SKIP            collected but skipped

The tool never mutates the quarantine file: adjudication and burn-down are two
separate human-reviewed steps (a test that passes in isolation but fails under
the full suite must not be silently promoted).

Design constraints:
* run one node id per subprocess so an interpreter-level crash cannot take the
  whole sweep down;
* pin torch/OMP threads to 1 so the sweep is a polite neighbour on a shared
  host and so thread-count effects do not masquerade as defects;
* pure functions (``parse_quarantine``, ``classify``, ``summarise``) are unit
  tested in ``tests/test_quarantine_adjudicate.py`` without running pytest in
  pytest.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUARANTINE = REPO_ROOT / "tests" / "ci_quarantine.txt"

PASS_NOW = "PASS_NOW"
EMPTY_EVIDENCE = "EMPTY_EVIDENCE"
API_DRIFT = "API_DRIFT"
NUMERIC = "NUMERIC"
THRESHOLD = "THRESHOLD"
ASSERT = "ASSERT"
ERROR = "ERROR"
TIMEOUT = "TIMEOUT"
SKIP = "SKIP"

_ALWAYS_RUN = (PASS_NOW, EMPTY_EVIDENCE, API_DRIFT, NUMERIC, THRESHOLD, ASSERT, ERROR, TIMEOUT, SKIP)

_NODEID = re.compile(r"^(tests/[^\s:]+\.py)::(\S+)$")
_ASSERTION_LINE = re.compile(r"^E\s{2,}(\w*(?:Error|Exception)?:?.*)$")
_EXC = re.compile(r"^E\s{2,}([A-Za-z_][\w.]*(?:Error|Exception|Exit))\b")


@dataclass(frozen=True)
class Entry:
    """One quarantined node id, with the batch comment that introduced it."""

    nodeid: str
    batch: str
    file: str


@dataclass(frozen=True)
class Verdict:
    nodeid: str
    file: str
    batch: str
    verdict: str
    returncode: int | None
    signature: str
    duration_s: float
    timed_out: bool


def parse_quarantine(path: Path) -> list[Entry]:
    """Return node ids from a quarantine file, each tagged with its batch header.

    The file is free-form: comment blocks describe why a batch is quarantined and
    the node ids follow.  Duplicated ids (the same test listed twice, which has
    happened) are collapsed to the first occurrence.
    """
    entries: list[Entry] = []
    seen: set[str] = set()
    batch = "unlabelled"
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            marker = re.match(r"^#\s*-+\s*(.*?)\s*-+$", line)
            if marker:
                batch = marker.group(1).strip()
            continue
        match = _NODEID.match(line)
        if not match:
            continue
        nodeid = line
        if nodeid in seen:
            continue
        seen.add(nodeid)
        entries.append(Entry(nodeid=nodeid, batch=batch, file=match.group(1)))
    return entries


def classify(returncode: int | None, stdout: str, *, timed_out: bool = False) -> tuple[str, str]:
    """Map one pytest invocation's outcome to (verdict, signature).

    ``signature`` is the most specific line pytest printed, trimmed to 240 chars,
    so the ledger stays human-auditable without the full log.
    """
    if timed_out:
        return TIMEOUT, f"exceeded timeout after {returncode} rc"
    if returncode == 0:
        return PASS_NOW, "passes on today's tree"
    if returncode is None:
        return ERROR, "no return code captured"

    lines = [line.rstrip() for line in stdout.splitlines()]
    detail = ""
    for line in lines:
        if _EXC.match(line):
            detail = _strip_e_prefix(line)
            break
    if not detail:
        for line in lines:
            if _ASSERTION_LINE.match(line):
                detail = _strip_e_prefix(line)
                break
    lowered = stdout.lower()
    if "error collecting" in lowered or "errors during collection" in lowered:
        return ERROR, _trim(detail or "collection error")
    if "skipped" in lowered and "no tests ran" not in lowered and returncode == 0:
        return SKIP, _trim(detail or "skipped")
    if not detail:
        named = re.search(
            r"\b(ImportError|ModuleNotFoundError|AttributeError|KeyError|TypeError|NameError)\b[^\n]*",
            stdout,
        )
        if named:
            return API_DRIFT, _trim(named.group(0))
        if "failed" in lowered and "passed" not in lowered:
            return ASSERT, _trim(_last_meaningful(lines))
        return ERROR, _trim(_last_meaningful(lines))

    detail = _trim(detail)
    if re.search(r"(ImportError|ModuleNotFoundError|AttributeError|KeyError|TypeError|NameError)", detail):
        return API_DRIFT, detail
    assert_text = " ".join(
        line
        for line in lines
        if line.startswith("E ") or line.lstrip().startswith("assert ")
    )
    if re.search(r"(assert\s+0\s*[><=]=?\s*0|assert\s+\(\)|assert\s+\[\]|assert\s+\{\}|assert\s+None)", assert_text):
        return EMPTY_EVIDENCE, detail
    if re.search(r"(approx|allclose|tolerance|rel=|rtol|atol|nan|inf)", assert_text, re.IGNORECASE):
        return NUMERIC, detail
    if re.search(r"(assert\s+\d+\s*[<>]|>\s*0|==\s*0|count)", assert_text):
        return THRESHOLD, detail
    return ASSERT, detail


def _strip_e_prefix(line: str) -> str:
    """Drop pytest's ``E   `` marker so signatures read like the assertion."""
    return line.strip()[1:].strip() if line.strip().startswith("E") else line.strip()


def _trim(text: str, limit: int = 240) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _last_meaningful(lines: Sequence[str]) -> str:
    for line in reversed(lines):
        if line.strip() and not line.startswith("="):
            return line.strip()
    return ""


def summarise(verdicts: Iterable[Verdict]) -> dict[str, object]:
    """Aggregate verdicts: totals, per-verdict counts, per-batch counts."""
    rows = list(verdicts)
    counts = {name: 0 for name in _ALWAYS_RUN}
    per_batch: dict[str, dict[str, int]] = {}
    for row in rows:
        counts[row.verdict] = counts.get(row.verdict, 0) + 1
        bucket = per_batch.setdefault(row.batch, {})
        bucket[row.verdict] = bucket.get(row.verdict, 0) + 1
    return {
        "total": len(rows),
        "counts": counts,
        "per_batch": per_batch,
        "unquarantine_candidates": sorted(row.nodeid for row in rows if row.verdict == PASS_NOW),
        "total_seconds": round(sum(row.duration_s for row in rows), 2),
    }


def run_one(entry: Entry, *, timeout: float, python: str) -> Verdict:
    """Run a single node id with ``--runxfail`` so the quarantine mask is off."""
    env = dict(os.environ)
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TORCH_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    cmd = [
        python,
        "-m",
        "pytest",
        entry.nodeid,
        "-q",
        "--runxfail",
        "-p",
        "no:randomly",
        "-p",
        "no:cacheprovider",
        "--no-header",
    ]
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = proc.stdout + proc.stderr
        verdict, signature = classify(proc.returncode, stdout)
        return Verdict(
            nodeid=entry.nodeid,
            file=entry.file,
            batch=entry.batch,
            verdict=verdict,
            returncode=proc.returncode,
            signature=signature,
            duration_s=round(time.time() - started, 2),
            timed_out=False,
        )
    except subprocess.TimeoutExpired:
        verdict, signature = classify(None, "", timed_out=True)
        return Verdict(
            nodeid=entry.nodeid,
            file=entry.file,
            batch=entry.batch,
            verdict=verdict,
            returncode=None,
            signature=signature,
            duration_s=round(time.time() - started, 2),
            timed_out=True,
        )


def select(entries: Sequence[Entry], *, files: Sequence[str], limit: int | None) -> list[Entry]:
    """Filter by file substring (``--file``) then cap with ``--limit``."""
    chosen = [entry for entry in entries if not files or any(token in entry.file for token in files)]
    return chosen[:limit] if limit else chosen


def render_markdown(payload: dict[str, object]) -> str:
    """Render the ledger as the docs table (kept boring so diffs stay readable)."""
    rows: Sequence[dict[str, object]] = payload["verdicts"]  # type: ignore[assignment]
    summary = payload["summary"]  # type: ignore[assignment]
    out = [
        "| verdict | count |",
        "|---|---|",
    ]
    for name, count in summary["counts"].items():  # type: ignore[index]
        out.append(f"| `{name}` | {count} |")
    out.append("")
    out.append("| node id | verdict | seconds | signature |")
    out.append("|---|---|---|---|")
    for row in rows:
        signature = str(row["signature"]).replace("|", "\\|")
        out.append(
            f"| `{row['nodeid']}` | `{row['verdict']}` | {row['duration_s']} | {signature} |"
        )
    return "\n".join(out) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quarantine", type=Path, default=DEFAULT_QUARANTINE)
    parser.add_argument("--out", type=Path, default=None, help="JSON ledger path")
    parser.add_argument("--markdown", type=Path, default=None, help="Markdown table path")
    parser.add_argument("--file", action="append", default=[], help="only node ids from files matching")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=300.0, help="per-test seconds")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true", help="list the selection and exit")
    args = parser.parse_args(argv)

    entries = select(
        parse_quarantine(args.quarantine), files=args.file, limit=args.limit
    )
    if args.dry_run:
        for entry in entries:
            print(entry.nodeid)
        print(f"# {len(entries)} node ids selected", file=sys.stderr)
        return 0

    verdicts: list[Verdict] = []
    for index, entry in enumerate(entries, start=1):
        result = run_one(entry, timeout=args.timeout, python=args.python)
        verdicts.append(result)
        print(
            f"[{index}/{len(entries)}] {result.verdict:14s} {result.duration_s:7.2f}s {result.nodeid}",
            flush=True,
        )

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "quarantine_file": str(args.quarantine.relative_to(REPO_ROOT))
        if args.quarantine.is_relative_to(REPO_ROOT)
        else str(args.quarantine),
        "selection": {"files": args.file, "limit": args.limit, "timeout_s": args.timeout},
        "summary": summarise(verdicts),
        "verdicts": [asdict(row) for row in verdicts],
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())