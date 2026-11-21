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


ASSET_ROOTS = ("benchmarks", "docs", "development", "experiments", "xval", "audit", "audits")


def tracked_files(root: Path, *areas: str) -> set[PurePosixPath]:
    raw = subprocess.check_output(
        ["git", "ls-files", "-z", "--", *areas],
        cwd=root,
    )
    return {PurePosixPath(path) for path in raw.decode().split("\0") if path}


def attach_validation_matrix(records: list[dict], root: Path) -> list[str]:
    """Associate validation definitions by exact archived artifact path."""
    directory = root / "benchmarks/validation_matrix"
    definitions = json.loads((directory / "matrix.json").read_text(encoding="utf-8"))
    references = json.loads((directory / "references.json").read_text(encoding="utf-8"))["references"]
    report = json.loads((directory / "BENCHMARK_REF_VS_GOT.json").read_text(encoding="utf-8"))
    historical_rows = {row["case"]: row for row in report["cases"]}
    by_result = defaultdict(list)
    for record in records:
        record["validation_matrix"] = []
        for variant in record["variants"]:
            for result in variant["results"]:
                by_result[result].append(record)
    errors = []
    seen = set()
    for case in definitions["cases"]:
        case_id = case["id"]
        if case_id in seen:
            errors.append(f"Duplicate validation matrix ID: {case_id}")
            continue
        seen.add(case_id)
        artifact = case.get("measure", {}).get("artifact")
        matches = by_result.get(artifact, [])
        if len(matches) > 1:
            errors.append(f"{case_id}: ambiguous registry owners of {artifact}")
            continue
        if matches:
            record = matches[0]
        else:
            record = {"id": f"matrix:{case_id}", "category": None, "tutorial": {},
                      "variants": [], "current_acceptance": "not_evaluated", "validation_matrix": []}
            records.append(record)
        missing = set(case["refs"]) - references.keys()
        if missing:
            errors.append(f"{case_id}: undefined reference IDs {sorted(missing)}")
            continue
        archived = historical_rows.get(case_id)
        record["validation_matrix"].append({
            "id": case_id, "definition": case,
            "references": {key: references[key] for key in case["refs"]},
            "archived_comparison": None if archived is None else {
                "path": "benchmarks/validation_matrix/BENCHMARK_REF_VS_GOT.json",
                "status": archived["status"], "artifact": archived["artifact"],
                "artifact_sha256": archived["artifact_sha256"],
            },
        })
    return errors


def attach_declared_cases(records: list[dict], root: Path) -> list[str]:
    """Register standalone benchmark and cross-validation manifests from the Git index."""
    manifests = sorted(tracked_files(root, ":(glob)xval/*/benchmark_case.json",
                                     ":(glob)benchmarks/*/benchmark_case.json"))
    if not manifests:
        return []
    tracked = tracked_files(root)
    registered_directories = {variant["path"] for row in records for variant in row["variants"]}
    errors = []
    for manifest in manifests:
        if str(manifest.parent) in registered_directories:
            errors.append(f"{manifest}: directory already has a registered case")
            continue
        definition = json.loads((root / manifest).read_text(encoding="utf-8"))
        is_xval = manifest.parts[0] == "xval"
        namespace = "xval" if is_xval else "campaign"
        case_id = f"{namespace}:{manifest.parent.name}"
        category = "cross_validation" if is_xval else "standalone_campaign"
        fields = ("source", "config", "documentation", "reference_files", "results")
        if not isinstance(definition, dict):
            errors.append(f"{manifest}: expected a JSON object")
            continue
        previous_errors = len(errors)
        if not isinstance(definition.get("scope"), str) or not definition["scope"].strip():
            errors.append(f"{manifest}: missing case scope")
        if not isinstance(definition.get("criteria"), dict) or not definition["criteria"]:
            errors.append(f"{manifest}: missing case criteria")
        for field in fields:
            paths = definition.get(field)
            if not isinstance(paths, list):
                errors.append(f"{manifest}: expected a list of paths for {field}")
                continue
            for path in paths:
                if not isinstance(path, str) or PurePosixPath(path) not in tracked or not (root / path).is_file():
                    errors.append(f"{manifest}: missing tracked {field} asset {path!r}")
        if not definition.get("source") or not definition.get("results"):
            errors.append(f"{manifest}: a case requires a source and a result")
        if len(errors) == previous_errors:
            records.append({
                "id": case_id, "category": category, "tutorial": {},
                "current_acceptance": "not_evaluated", "validation_matrix": [],
                "variants": [{"path": str(manifest.parent), "archive_bucket": "xval" if is_xval else None,
                              **{field: definition[field] for field in fields},
                              "references": definition.get("references", [])}],
                category: {"manifest": str(manifest), "scope": definition["scope"],
                                     "criteria": definition["criteria"]},
            })
    return errors


def build_registry(root: Path = ROOT) -> tuple[list[dict], list[str]]:
    catalogue = check(root)
    errors = list(catalogue["errors"])
    if errors:
        return [], errors
    files = tracked_files(root, "benchmarks", "docs/benchmarks/tutorials")
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
    errors.extend(attach_validation_matrix(records, root))
    errors.extend(attach_declared_cases(records, root))
    return records, errors


def benchmark_asset_index(records: list[dict], root: Path) -> dict:
    """Index every tracked benchmark path, retaining the basis of case attribution."""
    variants = defaultdict(set)
    declared = defaultdict(set)
    for row in records:
        case_id = row["id"]
        for variant in row["variants"]:
            variants[PurePosixPath(variant["path"])].add(case_id)
            for field in ("source", "config", "reference_files", "results", "documentation"):
                for path in variant[field]:
                    declared[path].add(case_id)
        for entry in row["validation_matrix"]:
            artifact = entry["definition"].get("measure", {}).get("artifact")
            if artifact:
                declared[artifact].add(case_id)

    indexed = subprocess.check_output(["git", "ls-files", "--stage", "-z", "--", "benchmarks"], cwd=root)
    assets = []
    unassigned = defaultdict(int)
    for entry in indexed.split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode, oid, stage = metadata.decode("ascii").split()
        path = PurePosixPath(raw_path.decode("utf-8"))
        if stage != "0":
            raise ValueError(f"Unresolved merge for {path}")
        nearest = next((parent for parent in path.parents if parent in variants), None)
        directory_ids = sorted(variants[nearest]) if nearest else []
        declared_ids = sorted(declared.get(str(path), ()))
        if not directory_ids and not declared_ids:
            group = str(PurePosixPath(*path.parts[:2])) if len(path.parts) > 2 else "benchmarks"
            unassigned[group] += 1
        assets.append({"path": str(path), "git_oid": oid, "git_mode": mode,
                       "declared_case_ids": declared_ids, "directory_case_ids": directory_ids})
    return {"assets": assets, "tracked_count": len(assets),
            "unassigned_count": sum(unassigned.values()), "unassigned_by_directory": dict(sorted(unassigned.items()))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--json", action="store_true", help="export the complete registry")
    group.add_argument("--asset-index", action="store_true",
                       help="index every tracked benchmark file, with explicit and directory-based case links")
    group.add_argument("--case", help="look up a case ID or registered case path")
    group.add_argument("--find-asset", metavar="TEXT",
                       help="search tracked paths across benchmarks, docs, development, experiments, xval and audits")
    args = parser.parse_args()
    if args.find_asset is not None:
        query = args.find_asset.strip().casefold()
        if not query:
            parser.error("asset search requires a nonempty path fragment")
        matches = sorted(str(p) for p in tracked_files(ROOT, *ASSET_ROOTS)
                         if query in str(p).casefold())
        print(json.dumps({"query": args.find_asset, "paths": matches},
                         ensure_ascii=False, indent=2))
        return 0 if matches else 1
    try:
        records, errors = build_registry()
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Cannot read benchmark registry inputs: {exc}\n")
    if errors:
        parser.exit(1, "\n".join(errors) + "\n")
    if args.asset_index:
        try:
            inventory = benchmark_asset_index(records, ROOT)
        except ValueError as exc:
            parser.exit(1, f"Cannot index benchmark files: {exc}\n")
        print(json.dumps(inventory, ensure_ascii=False, indent=2))
        return 0
    if args.case:
        records = [
            row for row in records
            if row["id"] == args.case or any(v["path"] == args.case for v in row["variants"])
            or any(f"matrix:{entry['id']}" == args.case for entry in row["validation_matrix"])
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
