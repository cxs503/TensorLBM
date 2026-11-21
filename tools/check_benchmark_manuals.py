#!/usr/bin/env python3
"""Validate the published TensorLBM benchmark tutorial catalogue.

Read-only, stdlib-only, and deliberately NOT a scientific acceptance gate.
Run: python tools/check_benchmark_manuals.py [--json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs/benchmarks/tutorials/README.md"
LINK = re.compile(r"\\[[^]]+\\]\\(([^)]+)\\)")
CASE = re.compile(r"[（(]([a-z][a-z0-9_]+)[）)]\\s*$")


def check(root: Path = ROOT) -> dict:
    index = root / "docs/benchmarks/tutorials/README.md"
    errors: list[str] = []
    records: list[dict] = []
    if not index.is_file():
        return {"cases": [], "errors": [f"missing index: {index}"]}
    seen: set[str] = set()
    for line_no, line in enumerate(index.read_text(encoding="utf-8").splitlines(), 1):
        if not line.startswith("| ["):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 6:
            errors.append(f"line {line_no}: expected 6 columns, got {len(cells)}")
            continue
        title, category, md, word, pdf, target = cells
        match = CASE.search(title.split("](")[0])
        if not match:
            errors.append(f"line {line_no}: missing stable case ID in title")
            continue
        case_id = match.group(1)
        if case_id in seen:
            errors.append(f"line {line_no}: duplicate case ID {case_id}")
        seen.add(case_id)
        record = {"id": case_id, "category": category, "manual": None,
                  "word": None, "pdf": None, "benchmark_path": None}
        for field, value in (("manual", md), ("word", word), ("pdf", pdf)):
            if value == "—":
                continue
            link = LINK.search(value)
            if not link:
                errors.append(f"line {line_no}: {case_id} invalid {field} link")
                continue
            rel = link.group(1)
            path = PurePosixPath(rel)
            if path.is_absolute() or ".." in path.parts or "://" in rel or "#" in rel:
                errors.append(f"line {line_no}: {case_id} unsafe {field} path {rel}")
                continue
            file = root / "docs/benchmarks" / rel
            if not file.is_file():
                errors.append(f"line {line_no}: {case_id} missing {field}: {rel}")
            record[field] = file.relative_to(root).as_posix()
        paths = re.findall(r"`([^`]+)`", target)
        if not paths:
            errors.append(f"line {line_no}: {case_id} missing benchmark target")
        else:
            for target_path in paths:
                p = PurePosixPath(target_path)
                if p.is_absolute() or ".." in p.parts or not target_path.startswith("benchmarks/"):
                    errors.append(f"line {line_no}: {case_id} invalid target {target_path}")
                elif not (root / target_path).is_dir():
                    errors.append(f"line {line_no}: {case_id} missing benchmark target {target_path}")
            record["benchmark_path"] = paths[0]
        records.append(record)
    if len(records) != 33:
        errors.append(f"expected 33 indexed cases, found {len(records)}")
    return {"cases": records, "errors": errors, "count": len(records),
            "pdf_count": sum(bool(r["pdf"]) for r in records),
            "word_count": sum(bool(r["word"]) for r in records)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable report")
    args = parser.parse_args()
    result = check()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Manual catalogue: {result['count']} cases; {result['pdf_count']} PDFs; {result['word_count']} Word files")
        for error in result["errors"]:
            print("ERROR:", error, file=sys.stderr)
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
