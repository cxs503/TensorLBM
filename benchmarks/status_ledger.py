#!/usr/bin/env python
"""Machine-checked status ledger for ``benchmarks/{pending,verified}``.

Why this exists
---------------
The benchmark archive carries two hand-maintained indexes
(``benchmarks/pending/README.md`` and ``benchmarks/STATUS.md``) plus per-case
machine verdicts (``result.json`` / ``validation_status.json``).  Nothing
checked that the three agree, so the archive silently rots: cases get promoted
to ``verified/`` without being removed from the pending table, new case
directories never get registered, and the advertised case counts drift away
from the directory listing.  This tool turns all of that into a machine check.

What it does
------------
1. discovers case directories under ``benchmarks/pending`` and
   ``benchmarks/verified``;
2. parses the pending index table (case link, status legend emoji, notes);
3. extracts a machine verdict from each pending case directory
   (``result.json`` / ``validation_status.json``, well-known keys only —
   anything unrecognised is reported as ``unknown``, never guessed);
4. compares the three views and emits violations with a severity:
   ``hard`` violations make ``--check`` fail (CI gate), ``soft`` violations are
   enumerated debt (legacy rows, claim conflicts, missing machine verdicts);
5. writes ``benchmarks/STATUS_LEDGER.json`` and refreshes the generated
   markdown blocks delimited by ``<!-- STATUS_LEDGER:BEGIN/END -->`` in the
   two index files, so the registration lists cannot go stale again.

Usage
-----
    python benchmarks/status_ledger.py                 # human summary
    python benchmarks/status_ledger.py --check         # exit 1 on hard violations
    python benchmarks/status_ledger.py --write         # refresh JSON + md blocks
    python benchmarks/status_ledger.py --json out.json # dump the ledger

The tool is read-only unless ``--write`` is given; it never invents a verdict
and never deletes a human-written index row.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH = Path("benchmarks")
PENDING_INDEX = BENCH / "pending" / "README.md"
STATUS_MD = BENCH / "STATUS.md"
LEDGER_JSON = BENCH / "STATUS_LEDGER.json"

BLOCK_BEGIN = "<!-- STATUS_LEDGER:BEGIN -->"
BLOCK_END = "<!-- STATUS_LEDGER:END -->"

# Status legend of benchmarks/pending/README.md (emoji -> machine class).
LEGEND_CLASS = {
    "✅": "pass_candidate",
    "🟡": "pass_candidate",
    "🔵": "running",
    "⚠️": "unverified",
    "⛔": "fail",
    "❌": "fail",
    "⏸️": "closed",  # terminal / superseded / archived verdicts
    "📦": "closed",
}

# Keys that carry a machine verdict, in priority order (first hit wins).
VERDICT_KEYS = ("verdict", "status", "passed_3pct_and_converged", "passed", "verified", "ok")
PASS_TOKENS = {"pass", "passed", "verified", "verified_pass", "true", "green", "ok"}
FAIL_TOKENS = {"fail", "failed", "not_verified", "false", "red", "rejected", "terminal"}

HARD = "hard"
SOFT = "soft"


# --------------------------------------------------------------------------- #
# discovery / parsing
# --------------------------------------------------------------------------- #
def case_dirs(root: Path, bucket: str) -> List[str]:
    """Case directory names under benchmarks/<bucket> (sorted)."""
    base = root / BENCH / bucket
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir() and not p.name.startswith((".", "_")))


_ROW_RE = re.compile(r"^\|\s*\[([^\]]+)\]\(([^)]*)\)\s*\|(.*)$")


def parse_pending_index(text: str) -> List[Dict[str, Any]]:
    """Parse the pending index table rows.

    Tolerates the variable number of note columns; only the first two cells
    (case, status) are interpreted.
    """
    rows: List[Dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        m = _ROW_RE.match(line)
        if not m:
            continue
        case, target, rest = m.group(1).strip(), m.group(2).strip(), m.group(3)
        cells = [c.strip() for c in rest.split("|")]
        status_raw = cells[0] if cells else ""
        emojis = [ch for ch in status_raw if ch in LEGEND_CLASS]
        rows.append(
            {
                "case": case,
                "target": target,
                "status_raw": status_raw,
                "class": LEGEND_CLASS[emojis[0]] if emojis else "unknown",
                "lineno": lineno,
            }
        )
    return rows


_VERIFIED_ROW_RE = re.compile(r"^\|\s*`([A-Za-z0-9_.\-]+)`\s*\|")


def parse_status_md_verified(text: str) -> List[str]:
    """Case names listed in the verified table of benchmarks/STATUS.md."""
    out: List[str] = []
    in_verified = False
    for line in text.splitlines():
        if line.startswith("## verified/"):
            in_verified = True
            continue
        if in_verified and line.startswith("## "):
            in_verified = False
            continue
        if in_verified:
            m = _VERIFIED_ROW_RE.match(line)
            if m:
                out.append(m.group(1))
    return out


_BLOCK_ROW_RE = re.compile(r"^\|\s*`([A-Za-z0-9_.\-]+)`\s*\|", re.M)


def block_text(text: str) -> str:
    """Return the generated STATUS_LEDGER block (empty string if absent)."""
    if BLOCK_BEGIN not in text or BLOCK_END not in text:
        return ""
    return text.split(BLOCK_BEGIN, 1)[1].split(BLOCK_END, 1)[0]


def strip_block(text: str) -> str:
    """Drop the generated block so the human tables are parsed alone."""
    if BLOCK_BEGIN not in text or BLOCK_END not in text:
        return text
    return text.split(BLOCK_BEGIN, 1)[0] + text.split(BLOCK_END, 1)[1]


def block_cases(text: str) -> List[str]:
    """Case names registered in the generated STATUS_LEDGER block."""
    return _BLOCK_ROW_RE.findall(block_text(text))


def parse_status_md_counts(text: str) -> Tuple[Optional[int], Optional[int]]:
    """Advertised (verified, pending) counts from the STATUS.md header."""
    m = re.search(r"verified:\s*\*\*(\d+)\*\*\s*\|\s*pending:\s*\*\*(\d+)\*\*", text)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _normalise_verdict(value: Any) -> str:
    if isinstance(value, bool):
        return "pass" if value else "fail"
    if isinstance(value, (int, float)):
        return "pass" if value else "fail"
    if isinstance(value, str):
        tok = value.strip().lower().replace(" ", "_")
        if tok in PASS_TOKENS:
            return "pass"
        if tok in FAIL_TOKENS:
            return "fail"
    return "unknown"


def machine_verdict(case_dir: Path) -> Dict[str, Any]:
    """Extract a machine verdict from a case directory.

    Never guesses: unrecognised payloads yield ``state == "unknown"`` with the
    source recorded, so the ledger can distinguish "no machine verdict on
    disk" from "machine says fail".
    """
    for fname in ("result.json", "validation_status.json"):
        path = case_dir / fname
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # malformed evidence is a finding, not a crash
            return {"state": "unreadable", "source": str(path), "key": None, "error": str(exc)[:200]}
        if not isinstance(payload, dict):
            continue
        for key in VERDICT_KEYS:
            if key in payload:
                state = _normalise_verdict(payload[key])
                if state != "unknown":
                    return {"state": state, "source": str(path), "key": key, "value": repr(payload[key])}
    return {"state": "unknown", "source": None, "key": None, "value": None}


# --------------------------------------------------------------------------- #
# ledger build
# --------------------------------------------------------------------------- #
def build_ledger(root: Path = REPO_ROOT) -> Dict[str, Any]:
    pending = case_dirs(root, "pending")
    verified = case_dirs(root, "verified")
    index_path = root / PENDING_INDEX
    index_text = index_path.read_text(encoding="utf-8") if index_path.is_file() else ""
    rows = parse_pending_index(strip_block(index_text))

    by_case: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(row["case"], []).append(row)

    # A case is "registered" when it has a human index row OR a row in the
    # generated ledger block; the block is rewritten from the directory scan,
    # so registration cannot rot even for cases nobody hand-documented.
    pending_registered = set(by_case) | set(block_cases(index_text))

    cases: List[Dict[str, Any]] = []
    violations: List[Dict[str, Any]] = []

    for case in pending:
        d = root / BENCH / "pending" / case
        verdict = machine_verdict(d)
        row_list = by_case.get(case, [])
        case_class = row_list[0]["class"] if row_list else None
        rec = {
            "case": case,
            "dir": str(d.relative_to(root)),
            "readme": (d / "README.md").is_file(),
            "result_json": (d / "result.json").is_file(),
            "validation_status_json": (d / "validation_status.json").is_file(),
            "index_rows": len(row_list),
            "index_class": case_class,
            "machine": verdict["state"],
            "machine_source": verdict["source"],
            "machine_key": verdict["key"],
        }
        cases.append(rec)

        if case not in pending_registered:
            violations.append(
                {
                    "id": "unregistered_pending_case",
                    "severity": HARD,
                    "case": case,
                    "detail": "pending case directory has no row in benchmarks/pending/README.md",
                }
            )
        if len(row_list) > 1:
            violations.append(
                {
                    "id": "duplicate_index_row",
                    "severity": SOFT,
                    "case": case,
                    "detail": f"{len(row_list)} index rows (lines {[r['lineno'] for r in row_list]})",
                }
            )
        if row_list and verdict["state"] in ("pass", "fail") and case_class in ("pass_candidate", "fail"):
            claimed = "pass" if case_class == "pass_candidate" else "fail"
            if claimed != verdict["state"]:
                violations.append(
                    {
                        "id": "status_claim_conflict",
                        "severity": SOFT,
                        "case": case,
                        "detail": (
                            f"index class {case_class} implies {claimed}, "
                            f"machine verdict {verdict['state']} ({verdict['key']}={verdict['value']} in {verdict['source']})"
                        ),
                    }
                )
        if verdict["state"] == "unknown" and not rec["readme"]:
            violations.append(
                {
                    "id": "missing_machine_verdict",
                    "severity": SOFT,
                    "case": case,
                    "detail": "no README and no machine verdict on disk",
                }
            )

    pending_set, verified_set = set(pending), set(verified)
    for row in rows:
        if row["case"] not in pending_set:
            moved = "moved to verified/" if row["case"] in verified_set else "no such directory"
            violations.append(
                {
                    "id": "stale_index_row",
                    "severity": SOFT,
                    "case": row["case"],
                    "detail": f"index row at line {row['lineno']} points to a non-pending case ({moved})",
                }
            )

    status_path = root / STATUS_MD
    status_text = status_path.read_text(encoding="utf-8") if status_path.is_file() else ""
    listed = parse_status_md_verified(strip_block(status_text))
    verified_registered = set(listed) | set(block_cases(status_text))
    for case in verified:
        if case not in verified_registered:
            violations.append(
                {
                    "id": "unlisted_verified_case",
                    "severity": HARD,
                    "case": case,
                    "detail": "verified case directory missing from the benchmarks/STATUS.md verified table",
                }
            )
    for case in set(listed):
        if listed.count(case) > 1:
            violations.append(
                {
                    "id": "duplicate_verified_row",
                    "severity": SOFT,
                    "case": case,
                    "detail": f"{listed.count(case)} rows in benchmarks/STATUS.md verified table",
                }
            )

    adv_v, adv_p = parse_status_md_counts(status_text)
    if (adv_v, adv_p) != (len(verified), len(pending)):
        violations.append(
            {
                "id": "stale_counts",
                "severity": HARD,
                "case": None,
                "detail": (
                    f"benchmarks/STATUS.md advertises verified={adv_v} pending={adv_p}, "
                    f"directories are verified={len(verified)} pending={len(pending)}"
                ),
            }
        )

    hard = [v for v in violations if v["severity"] == HARD]
    soft = [v for v in violations if v["severity"] == SOFT]
    return {
        "tool": "benchmarks/status_ledger.py",
        "schema": 1,
        "counts": {
            "pending_dirs": len(pending),
            "verified_dirs": len(verified),
            "pending_index_rows": len(rows),
            "status_md_verified_rows": len(listed),
            "advertised_verified": adv_v,
            "advertised_pending": adv_p,
            "pending_with_result_json": sum(c["result_json"] for c in cases),
            "pending_with_readme": sum(c["readme"] for c in cases),
            "pending_machine_pass": sum(c["machine"] == "pass" for c in cases),
            "pending_machine_fail": sum(c["machine"] == "fail" for c in cases),
            "pending_machine_unknown": sum(c["machine"] == "unknown" for c in cases),
        },
        "violations": {"hard": hard, "soft": soft},
        "cases": cases,
        "verified": verified,
        "index_rows": rows,
    }


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _pending_table(ledger: Dict[str, Any]) -> str:
    lines = [
        "| case | README | result.json | machine verdict | index rows | index class |",
        "|---|---|---|---|---|---|",
    ]
    for c in ledger["cases"]:
        lines.append(
            "| `{case}` | {readme} | {result} | {machine}{src} | {rows} | {cls} |".format(
                case=c["case"],
                readme="yes" if c["readme"] else "**no**",
                result="yes" if c["result_json"] else "**no**",
                machine=c["machine"],
                src="" if c["machine"] == "unknown" else f" (`{c['machine_key']}`)",
                rows=c["index_rows"],
                cls=c["index_class"] or "**unregistered**",
            )
        )
    return "\n".join(lines)


def render_pending_block(ledger: Dict[str, Any]) -> str:
    cnt = ledger["counts"]
    hard = ledger["violations"]["hard"]
    soft = ledger["violations"]["soft"]
    head = [
        "### 机器台账（自动生成，勿手改；`python benchmarks/status_ledger.py --write`）",
        "",
        f"pending 目录 **{cnt['pending_dirs']}** · 已登记行 **{cnt['pending_index_rows']}** · "
        f"带 result.json **{cnt['pending_with_result_json']}** · "
        f"机器判决 pass/fail/unknown = **{cnt['pending_machine_pass']}/{cnt['pending_machine_fail']}/{cnt['pending_machine_unknown']}**",
        "",
        f"违规：hard **{len(hard)}** · soft **{len(soft)}**"
        "（hard 阻断 CI，soft 为已披露存量债，逐条列在下表/JSON）",
        "",
    ]
    if hard:
        head += ["**hard 违规**：", ""]
        head += [f"- `{v['id']}` `{v['case']}` — {v['detail']}" for v in hard]
        head.append("")
    if soft:
        head += ["<details><summary>soft 存量债（逐条）</summary>", ""]
        head += [f"- `{v['id']}` `{v['case']}` — {v['detail']}" for v in soft]
        head += ["", "</details>", ""]
    return "\n".join(head + [_pending_table(ledger)]) + "\n"


def render_verified_block(ledger: Dict[str, Any]) -> str:
    """Full verified registry (stable): human-table coverage is a column, not a filter.

    The block intentionally lists *every* verified directory so that the
    registration check (`verified dir is in human table OR in this block`)
    cannot oscillate between "listed as missing" and "listed as present".
    """
    cnt = ledger["counts"]
    listed = set(ledger.get("status_md_verified_listed", []))
    covered = sum(c in listed for c in ledger["verified"])
    lines = [
        "### verified 机器台账（自动生成，勿手改）",
        "",
        f"verified 目录 **{cnt['verified_dirs']}** · STATUS.md 人工表已写 note 的 **{covered}** · "
        f"advertised header verified={cnt['advertised_verified']} pending={cnt['advertised_pending']}",
        "",
        f"人工表未覆盖 **{cnt['verified_dirs'] - covered}** 个目录（notes 待补；下表由目录扫描生成，"
        "覆盖=该目录是否已出现在上方人工表）：",
        "",
        "| verified case | in human table |",
        "|---|---|",
    ]
    lines += [f"| `{c}` | {'yes' if c in listed else '**no**'} |" for c in ledger["verified"]]
    return "\n".join(lines) + "\n"


def update_block(path: Path, block: str) -> bool:
    """Replace (or append) the generated block in a markdown file. Idempotent."""
    text = path.read_text(encoding="utf-8")
    new = f"{BLOCK_BEGIN}\n{block}{BLOCK_END}\n"
    if BLOCK_BEGIN in text and BLOCK_END in text:
        pre = text.split(BLOCK_BEGIN)[0]
        post = text.split(BLOCK_END, 1)[1]
        out = pre + new + post.lstrip("\n")
    else:
        out = text.rstrip("\n") + "\n\n" + new
    if out == text:
        return False
    path.write_text(out, encoding="utf-8")
    return True


_COUNT_RE = re.compile(r"verified:\s*\*\*(\d+)\*\*\s*\|\s*pending:\s*\*\*(\d+)\*\*(\s*\([^)]*\))?")


def refresh_status_counts(path: Path, ledger: Dict[str, Any]) -> bool:
    """Rewrite the advertised case counts in benchmarks/STATUS.md. Idempotent."""
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    cnt = ledger["counts"]
    replacement = (
        f"verified: **{cnt['verified_dirs']}** | pending: **{cnt['pending_dirs']}**  "
        "(counts machine-refreshed by `benchmarks/status_ledger.py --write`)"
    )
    new_text, n = _COUNT_RE.subn(replacement, text, count=1)
    if not n or new_text == text:
        return False
    path.write_text(new_text, encoding="utf-8")
    return True


def _write_once(ledger: Dict[str, Any], root: Path) -> Dict[str, str]:
    payload = dict(ledger)
    status_text = (root / STATUS_MD).read_text(encoding="utf-8") if (root / STATUS_MD).is_file() else ""
    payload["status_md_verified_listed"] = parse_status_md_verified(strip_block(status_text))
    state: Dict[str, str] = {}
    refresh_status_counts(root / STATUS_MD, payload)
    state[str(STATUS_MD)] = "updated" if update_block(root / STATUS_MD, render_verified_block(payload)) else "unchanged"
    state[str(PENDING_INDEX)] = "updated" if update_block(root / PENDING_INDEX, render_pending_block(payload)) else "unchanged"
    return state


def write_artifacts(ledger: Dict[str, Any], root: Path = REPO_ROOT) -> Dict[str, str]:
    """Refresh the generated blocks to a fixed point, then dump the JSON ledger.

    The generated blocks are themselves ledger inputs (a registered case is one
    that appears in a block), so a single pass can flip hard violations from
    "unregistered" to "clean". Iterating to a fixed point makes ``--write``
    idempotent: the second call reports ``unchanged`` for both files.
    """
    state: Dict[str, str] = {}
    current = ledger
    for _ in range(4):
        state = _write_once(current, root)
        current = build_ledger(root)
        if all(v == "unchanged" for v in state.values()):
            break
    out = root / LEDGER_JSON
    payload = dict(current)
    payload.pop("status_md_verified_listed", None)
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    state[str(LEDGER_JSON)] = "written"
    return state


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def summarise(ledger: Dict[str, Any]) -> str:
    cnt, viol = ledger["counts"], ledger["violations"]
    parts = [
        "benchmarks status ledger",
        f"  pending dirs        : {cnt['pending_dirs']}",
        f"  verified dirs       : {cnt['verified_dirs']}",
        f"  pending index rows  : {cnt['pending_index_rows']}",
        f"  STATUS.md verified  : {cnt['status_md_verified_rows']} (advertised {cnt['advertised_verified']}/{cnt['advertised_pending']})",
        f"  machine pass/fail/unknown : {cnt['pending_machine_pass']}/{cnt['pending_machine_fail']}/{cnt['pending_machine_unknown']}",
        f"  hard violations     : {len(viol['hard'])}",
        f"  soft violations     : {len(viol['soft'])}",
    ]
    for v in viol["hard"]:
        parts.append(f"    [hard] {v['id']}: {v['case']} — {v['detail']}")
    ids: Dict[str, int] = {}
    for v in viol["soft"]:
        ids[v["id"]] = ids.get(v["id"], 0) + 1
    for key, n in sorted(ids.items()):
        parts.append(f"    [soft] {key}: {n}")
    return "\n".join(parts)


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Machine-checked benchmark status ledger")
    ap.add_argument("--root", default=str(REPO_ROOT), help="repository root")
    ap.add_argument("--json", default=None, help="write the ledger JSON to this path")
    ap.add_argument("--write", action="store_true", help="refresh STATUS_LEDGER.json and the generated md blocks")
    ap.add_argument("--check", action="store_true", help="exit 1 if any hard violation is present")
    args = ap.parse_args(list(argv) if argv is not None else None)
    root = Path(args.root).resolve()

    ledger = build_ledger(root)
    print(summarise(ledger))
    if args.write:
        for path, state in write_artifacts(ledger, root).items():
            print(f"  {state:9s} {path}")
        # Re-build: the refreshed md blocks are themselves ledger inputs, so the
        # JSON artifact must describe the post-refresh state (converged ledger).
        ledger = build_ledger(root)
        print("--- after refresh ---")
        print(summarise(ledger))
    if args.json:
        Path(args.json).write_text(json.dumps(ledger, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        print(f"  written   {args.json}")
    if args.check and ledger["violations"]["hard"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())