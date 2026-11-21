#!/usr/bin/env python3
"""Query the benchmark registry from tracked assets and the tutorial catalogue."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path, PurePosixPath

from check_benchmark_manuals import ROOT, check

BUCKETS = {"verified", "pending", "archived", "case_pool"}


def tracked_files(root: Path) -> set[PurePosixPath]:
    raw = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "benchmarks", "docs/benchmarks/tutorials"],
        cwd=root,
    )
    return {PurePosixPath(path) for path in raw.decode().split("\0") if path}


def build_registry(root: Path = ROOT) -> tuple[list[dict], list[str]]:
    catalogue = check(root)
    errors = list(catalogue["errors"])
    if errors:
        return [], errors
    files = tracked_files(root)
    by_dir = defaultdict(list)
    for path in sorted(files):
        by_dir[path.parent].append(path)

    def assets(directory: str) -> dict:
        path = PurePosixPath(directory)
        members = by_dir[path]
        result_paths = [
            p for p in members
            if p.suffix in {".json", ".csv", ".npz"}
            and p.stem.startswith(("result", "summary", "validation_status", "benchmark"))
        ]
        reference_paths = [
            p for p in members
            if p.stem.lower().startswith(("refdata", "reference", "bibliography"))
            and p.suffix != ".py"
        ]
        references = []
        for p in result_paths:
            if p.suffix != ".json":
                continue
            try:
                data = json.loads((root / p).read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValueError):
                continue
            if isinstance(data, dict):
                for key in ("reference", "references", "citation", "doi"):
                    if key in data:
                        references.append({"source": str(p), "field": key, "value": data[key]})
        return {
            "path": directory,
            "archive_bucket": path.parts[1],
            "source": [str(p) for p in members if p.suffix == ".py"],
            "config": [str(p) for p in members if p.suffix in {".yaml", ".yml", ".toml"}
                       or p.suffix == ".json" and p.stem.startswith(("config", "case"))],
            "documentation": [str(p) for p in members if p.name.lower() == "readme.md"],
            "reference_files": list(map(str, reference_paths)),
            "references": references,
            "results": list(map(str, result_paths)),
        }

    records = []
    covered = set()
    for row in catalogue["cases"]:
        paths = row["benchmark_paths"]
        covered.update(PurePosixPath(p) for p in paths)
        manuals = {}
        for field in ("manual", "word", "pdf"):
            if value := row[field]:
                if PurePosixPath(value) not in files:
                    errors.append(f"{row['id']}: untracked {field}: {value}")
                manuals[field] = {
                    "path": value, "sha256": hashlib.sha256((root / value).read_bytes()).hexdigest(),
                }
        for path in paths:
            if not any(PurePosixPath(path) in file.parents for file in files):
                errors.append(f"{row['id']}: no tracked assets under {path}")
        records.append({
            "id": row["id"], "category": row["category"], "tutorial": manuals,
            "variants": [assets(p) for p in paths], "current_acceptance": "not_evaluated",
        })

    roots = {
        PurePosixPath(*path.parts[:3]) for path in files
        if len(path.parts) >= 4 and path.parts[0] == "benchmarks" and path.parts[1] in BUCKETS
    }
    for path in sorted(roots - covered):
        records.append({
            "id": str(path), "category": None, "tutorial": {},
            "variants": [assets(str(path))], "current_acceptance": "not_evaluated",
        })
    return records, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--json", action="store_true", help="export the complete registry")
    group.add_argument("--case", help="look up a case ID or registered case path")
    args = parser.parse_args()
    records, errors = build_registry()
    if errors:
        parser.exit(1, "\n".join(errors) + "\n")
    if args.case:
        records = [
            row for row in records
            if row["id"] == args.case or any(v["path"] == args.case for v in row["variants"])
        ]
        if not records:
            parser.exit(1, f"Unknown case: {args.case}\n")
    if args.json or args.case:
        print(json.dumps({"schema_version": 1, "cases": records}, ensure_ascii=False, indent=2))
    else:
        print(f"Benchmark registry OK: {len(records)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
