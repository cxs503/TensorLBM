#!/usr/bin/env python3
"""Machine audit of the graded reference rows in ``reference.json``.

Why this file exists
====================
``reference.json`` grades every referee row (``VERIFIED_TABLE`` / ``WITHHELD`` /
``TREND``) and names its source.  Until now a ``VERIFIED_TABLE`` row was still
an act of *trust*: the number had been read out of a full text in some earlier
mission, and nothing in the repository could tell a correct quote from a
mis-typed one.  For an anchor whose whole dispute (audit ``m16183``/``m16205``)
was "two different reference constants for the same measurement", an
un-auditable quote is exactly the failure mode worth closing.

This script re-derives the quoted rows from the pinned full text:

* the source is pinned by URL **and** sha256 of the fetched document, so a
  different (or edited) copy is rejected before any number is compared;
* the ``alpha = 0`` block of Table 1 is re-parsed from the served HTML
  (``ar5iv`` LaTeX-in-HTML tables) and the *Present* / *Sohankar et al.* /
  *Yoon et al.* columns are compared, cell by cell, with the ``ref``/``band``
  values recorded in ``reference.json``;
* every ``WITHHELD`` row must still carry a written ``withheld_reason`` and a
  null ``ref`` -- i.e. "no number" is a checked state, not an omission.

Exit codes: ``0`` = all checked rows agree, ``1`` = FAIL, and a run without
``--source`` reports ``SKIP`` (never ``PASS``) because nothing was audited.

Usage
-----
    python audit_reference.py --source /path/to/2411.03124.html [--json-out o.json]
    python audit_reference.py --selftest        # synthetic document, no network
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "reference.json"

SOURCE_URL = "https://ar5iv.labs.arxiv.org/html/2411.03124"
SOURCE_LABEL = "arXiv 2411.03124v4, Table 1, alpha = 0 deg block"
# sha256 of the document fetched in mission m16678 (2026-11-12).
SOURCE_SHA256 = "f95490265a0630da64308116c990794791eb8f706ddfe917503ffadc8c397d7a"

# Column order of the alpha=0 block of Table 1 (as served by ar5iv):
#   Re | Cd: Present | Cd: Sohankar et al. | Cd: Yoon et al. |
#        St: Present | St: Sohankar et al.
NUM_COLS = 5


def _cells(row_html: str) -> list[str]:
    out = []
    for c in re.findall(r"<td[^>]*>(.*?)</td>", row_html, re.S):
        txt = " ".join(re.sub(r"<[^>]+>", " ", c).split())
        if txt:
            out.append(txt)
    return out


def extract_table1(html: str) -> dict[int, dict[str, list[float]]]:
    """Return {Re: {'cd': [present, sohankar, yoon], 'st': [present, sohankar]}}.

    Only the *first* contiguous numeric block is taken: it is the
    ``alpha = 0`` block (the later blocks are the ``alpha = 45`` degrees
    validation and the Nusselt comparison, which share the same Re labels).
    """
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S)
    table: dict[int, dict[str, list[float]]] = {}
    seen_100 = False
    for r in rows:
        c = _cells(r)
        if not c or c[0] not in {"50", "100", "150"}:
            continue
        if len(c) < 1 + NUM_COLS:
            continue
        try:
            vals = [float(x) for x in c[1 : 1 + NUM_COLS]]
        except ValueError:
            continue
        re_val = int(c[0])
        if re_val == 100:
            if seen_100:
                break  # second table (alpha=45 block) starts -> stop
            seen_100 = True
        if 150 in table and re_val == 50:
            break
        table[re_val] = {"cd": vals[0:3], "st": vals[3:5]}
    return table


def audit(reference: dict, table: dict[int, dict[str, list[float]]]) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []

    # (1) the graded verifiable rows must reproduce from the fetched document
    for row in reference.get("rows", []):
        grade = row.get("grade")
        re_val, metric = row.get("re"), row.get("metric")
        if grade == "VERIFIED_TABLE":
            if re_val not in table or metric not in table[re_val]:
                checks.append((f"Re={re_val} {metric}: extractable from source", False, "row absent in extracted table"))
                continue
            got = table[re_val][metric]
            ref = row.get("ref")
            band = row.get("band") or []
            in_ref = any(abs(v - ref) < 1e-12 for v in got)
            in_band = all(any(abs(v - b) < 1e-12 for v in got) for b in band) if band else in_ref
            checks.append(
                (
                    f"Re={re_val} {metric}: ref {ref} appears in source column set {got}",
                    bool(in_ref),
                    f"columns={got} band={band}",
                )
            )
            checks.append(
                (
                    f"Re={re_val} {metric}: band {band} is covered by source values",
                    bool(in_band),
                    f"columns={got}",
                )
            )
        elif grade == "WITHHELD":
            ok = row.get("ref", "missing") is None and bool(str(row.get("withheld_reason", "")).strip())
            checks.append((f"Re={re_val} {metric}: WITHHELD keeps ref=null + written reason", ok, f"ref={row.get('ref')!r}"))
        elif grade == "TREND":
            checks.append((f"Re={re_val} {metric}: TREND row declares non-referee status", True, "informational"))
        else:
            checks.append((f"Re={re_val} {metric}: unknown grade {grade!r}", False, "grading discipline violated"))

    trend = reference.get("trend_expectation_re200")
    if trend:
        ok = "TREND" in str(trend.get("grade", "")) and "do not use as a pass/fail gate" in str(trend.get("grade", ""))
        checks.append(("trend_expectation_re200 is labelled non-referee", ok, str(trend.get("grade"))))
    return checks


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_SYNTH_OK = """
<table><tr><td>50</td><td>1.57</td><td>1.62</td><td>1.57</td><td>0.114</td><td>-</td></tr>
<tr><td>100</td><td>1.48</td><td>1.46</td><td>1.44</td><td>0.147</td><td>0.147</td></tr>
<tr><td>150</td><td>1.41</td><td>1.41</td><td>1.40</td><td>0.159</td><td>0.162</td></tr></table>
"""


def selftest() -> int:
    """Prove the auditor is discriminating: matching quotes PASS, tampered ones FAIL."""
    ref = json.loads(REFERENCE.read_text())
    ok_table = extract_table1(_SYNTH_OK)
    checks = audit(ref, ok_table)
    failures = [c for c in checks if not c[1]]
    print(f"[selftest] faithful document : {len(checks) - len(failures)} OK / {len(failures)} FAIL")
    rc = 0 if not failures else 1

    tampered = _SYNTH_OK.replace("1.48</td><td>1.46", "1.58</td><td>1.46")
    bad = [c for c in audit(ref, extract_table1(tampered)) if not c[1]]
    print(f"[selftest] tampered Cd cell  : {len(bad)} FAIL (expected >=1)")
    rc |= 0 if bad else 1

    withheld_broken = json.loads(REFERENCE.read_text())
    for row in withheld_broken.get("rows", []):
        if row.get("grade") == "WITHHELD":
            row["ref"] = 1.4  # pretend someone filled in a remembered number
            break
    bad2 = [c for c in audit(withheld_broken, ok_table) if not c[1]]
    print(f"[selftest] fabricated WITHHELD: {len(bad2)} FAIL (expected >=1)")
    rc |= 0 if bad2 else 1
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=None, help="path to the pinned full text (ar5iv HTML)")
    ap.add_argument("--reference", default=str(REFERENCE))
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    ref = json.loads(Path(args.reference).read_text())
    if not args.source:
        print(
            "SKIP: no --source supplied -> no number was audited against the full text "
            f"(pinned source: {SOURCE_URL}, sha256 {SOURCE_SHA256[:16]}...)"
        )
        return 0

    src = Path(args.source)
    if not src.exists():
        print(f"SKIP: source {src} not present on this machine")
        return 0
    html = src.read_text(encoding="utf-8", errors="replace")
    got_sha = sha256_text(html)
    pinned_ok = got_sha == SOURCE_SHA256
    print(f"source sha256 {got_sha}")
    print(f"pinned sha256 {SOURCE_SHA256} -> {'MATCH' if pinned_ok else 'MISMATCH'}")
    print(f"source label  {SOURCE_LABEL} ({SOURCE_URL})")
    if not pinned_ok:
        print("FAIL: document is not the pinned copy; refusing to audit numbers against it")
        return 1

    table = extract_table1(html)
    print(f"extracted Re keys: {sorted(table)}")
    checks = audit(ref, table)
    n_bad = 0
    for name, ok, detail in checks:
        print(f"[{'OK' if ok else 'FAIL'}] {name}  ({detail})")
        n_bad += 0 if ok else 1
    print(f"\ncounts: OK {len(checks) - n_bad} / FAIL {n_bad}")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "source_url": SOURCE_URL,
                    "source_sha256": got_sha,
                    "sha256_matches_pin": pinned_ok,
                    "checks": [{"name": n, "ok": o, "detail": d} for n, o, d in checks],
                    "failures": n_bad,
                },
                indent=2,
            )
        )
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())