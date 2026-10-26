#!/usr/bin/env python3
"""Check repository hygiene without importing the solver or changing files.

Checks tracked files and untracked, non-ignored files. Historical exceptions
are exact paths, not directory-wide exemptions. Ignored, untracked run outputs
are deliberately outside the check; force-added outputs are checked in CI.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path, PurePosixPath

POLICY_PATH = "docs/architecture/repository_layout_policy.json"
MAX_FILE_BYTES = 5 * 1024 * 1024
GENERATED_ROOTS = {"runs", "runs_prop", "outputs"}
GENERATED_PARTS = {"__pycache__", "node_modules", ".venv", ".pytest_cache", ".ruff_cache"}
MODEL_SUFFIXES = {".pt", ".pth", ".ckpt", ".onnx", ".safetensors"}
PATH_LISTS = ("root_files", "legacy_root_files", "legacy_generated_files", "legacy_model_files")


def validate_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid policy path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value or value == ".":
        raise ValueError(f"Policy paths must be normalized repository-relative paths: {value!r}")
    return value


def load_policy(path: Path) -> dict:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(policy, dict) or policy.get("schema_version") != 1:
        raise ValueError("Expected repository layout policy schema_version=1")
    expected = {"schema_version", "baseline_revision", "large_files", *PATH_LISTS}
    if set(policy) != expected:
        raise ValueError(f"Expected policy keys: {sorted(expected)}")
    for key in PATH_LISTS:
        values = policy[key]
        if not isinstance(values, list):
            raise ValueError(f"{key} must be a list")
        paths = [validate_path(value) for value in values]
        if len(paths) != len(set(paths)):
            raise ValueError(f"Duplicate paths in {key}")
        if key in {"root_files", "legacy_root_files"} and any("/" in p for p in paths):
            raise ValueError(f"{key} may only contain root filenames")
    if set(policy["root_files"]) & set(policy["legacy_root_files"]):
        raise ValueError("Canonical and legacy root files must not overlap")
    if not isinstance(policy["large_files"], dict):
        raise ValueError("large_files must map exact paths to reviewed byte ceilings")
    for name, limit in policy["large_files"].items():
        validate_path(name)
        if type(limit) is not int or limit <= MAX_FILE_BYTES:
            raise ValueError(f"Invalid large-file byte ceiling for {name}")
    if not isinstance(policy["baseline_revision"], str) or not policy["baseline_revision"]:
        raise ValueError("baseline_revision must identify the original inventory")
    return policy


def candidate_files(repo: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return sorted(set(result.stdout.decode("utf-8", errors="surrogateescape").split("\0")) - {""})


def is_generated(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return (
        parts[0] in GENERATED_ROOTS
        or bool(set(parts[:-1]) & GENERATED_PARTS)
        or parts[:2] == ("artifacts", "runs")
        or parts[:2] == ("frontend-vue", "dist")
        or parts[0] in {"build", "dist"}
    )


def check_files(repo: Path, names: list[str], policy: dict) -> list[str]:
    failures = []
    roots = set(policy["root_files"]) | set(policy["legacy_root_files"])
    generated = set(policy["legacy_generated_files"])
    models = set(policy["legacy_model_files"])
    for name in names:
        path = repo / name
        try:
            # lstat avoids following symlinks outside the checkout.
            size = path.lstat().st_size
        except FileNotFoundError:
            # A tracked deletion in the working tree is not a new violation.
            continue
        if "/" not in name and name not in roots:
            failures.append(
                f"root-file: {name} — choose an owned directory or review the root policy"
            )
        if is_generated(name) and name not in generated:
            failures.append(f"generated-file: {name} — keep raw outputs out of Git")
        if PurePosixPath(name).suffix.lower() in MODEL_SUFFIXES and name not in models:
            failures.append(
                f"model-file: {name} — use a versioned external artifact with a checksum"
            )
        ceiling = policy["large_files"].get(name, MAX_FILE_BYTES)
        if size > ceiling:
            failures.append(f"large-file: {name} — {size} bytes exceeds reviewed ceiling {ceiling}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo = args.repo.resolve()
    try:
        policy = load_policy(repo / POLICY_PATH)
        names = candidate_files(repo)
        failures = check_files(repo, names, policy)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Repository layout check could not complete: {exc}")
        return 2
    if failures:
        print("Repository layout violations:")
        for failure in failures:
            print(f"  {failure}")
        print("Do not regenerate exceptions to hide new findings; request a policy review.")
        return 1
    print(f"Repository layout OK ({len(names)} candidate paths; historical exceptions retained).")
    print(
        "This check is not numerical validation, a full artifact inventory, or migration approval."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
