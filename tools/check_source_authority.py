#!/usr/bin/env python3
"""Enforce the canonical source tree without deleting historical code.

The old checkout is retained as a complete, frozen research archive. It
must not silently regain production consumers.  Existing consumers are an
explicit migration backlog, recorded by matching-line digest rather than being
mistaken for acceptable architecture.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "docs/architecture/source_authority_policy.json"
TOKEN = "TensorLBM/src"
ARCHIVE = "development/legacy/TensorLBM"
LEGACY_IMPORT_ROOT = Path("src/lbm")
CHECKOUT, SOURCE = TOKEN.split("/")
SPLIT_PATH = re.compile(rf"""['"]{CHECKOUT}['"]\s*[/,]\s*['"]{SOURCE}['"]""")
TEXT_SUFFIXES = {".py", ".sh", ".toml", ".yaml", ".yml"}
CONTROL_PATHS = {
    Path(__file__).resolve().relative_to(ROOT).as_posix(),
}


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def load_policy(path: Path = POLICY) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def tracked_paths() -> list[Path]:
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return [Path(item.decode()) for item in raw.split(b"\0") if item]


def _is_literal(node: ast.AST, value: str) -> bool:
    return isinstance(node, ast.Constant) and node.value == value


def matching_reference_lines(text: str, path: Path) -> list[str]:
    lines = text.splitlines()
    matched = {number for number, line in enumerate(lines, 1) if TOKEN in line or ARCHIVE in line}
    if path.suffix == ".py" and SPLIT_PATH.search(text):
        for node in ast.walk(ast.parse(text, filename=str(path))):
            pairs = ()
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                left = node.left
                if isinstance(left, ast.BinOp) and isinstance(left.op, ast.Div):
                    left = left.right
                pairs = ((left, node.right),)
            elif isinstance(node, ast.Call) and (
                isinstance(node.func, ast.Name) and node.func.id == "Path"
                or isinstance(node.func, ast.Attribute)
                and node.func.attr in {"Path", "join", "joinpath"}
            ):
                pairs = zip(node.args, node.args[1:])
            for left, right in pairs:
                if _is_literal(left, CHECKOUT) and _is_literal(right, SOURCE):
                    matched.update(range(left.lineno, right.end_lineno + 1))
    return [lines[number - 1].rstrip() for number in sorted(matched)]


def collect_active_references(policy: dict) -> dict[str, dict[str, object]]:
    roots = tuple(f"{root.rstrip('/')}/" for root in policy["active_roots"])
    result: dict[str, dict[str, object]] = {}
    for relative in tracked_paths():
        name = relative.as_posix()
        if (
            name in CONTROL_PATHS
            or relative.suffix not in TEXT_SUFFIXES
            or not name.startswith(roots)
        ):
            continue
        text = (ROOT / relative).read_text(encoding="utf-8", errors="strict")
        lines = matching_reference_lines(text, relative)
        if not lines:
            continue
        payload = "\n".join(lines).encode()
        result[name] = {
            "matches": len(lines),
            "matching_lines_sha256": hashlib.sha256(payload).hexdigest(),
        }
    return result


def validate(policy: dict) -> list[str]:
    errors: list[str] = []
    package = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    setuptools = package.get("tool", {}).get("setuptools", {})
    if setuptools.get("package-dir") != {"": policy["canonical_root"]}:
        errors.append("pyproject package-dir no longer selects the canonical src tree")
    find = setuptools.get("packages", {}).get("find", {})
    if find.get("where") != [policy["canonical_root"]]:
        errors.append("pyproject package discovery no longer selects only src")
    if set(find.get("include", ())) != {"tensorlbm", "tensorlbm.*", "lbm", "lbm.*"}:
        errors.append("pyproject must restrict discovery to tensorlbm and the existing lbm package")

    allowed_shims = policy["legacy_import_shims"]
    tracked_shims = {
        path.as_posix() for path in tracked_paths() if path.is_relative_to(LEGACY_IMPORT_ROOT)
    }
    for path in sorted(tracked_shims - allowed_shims.keys()):
        errors.append(f"unexpected file in the lbm compatibility package: {path}")
    for path, digest in sorted(allowed_shims.items()):
        shim = ROOT / path
        if path not in tracked_shims or not shim.is_file() or shim.is_symlink():
            errors.append(f"missing tracked lbm compatibility shim: {path}")
        elif hashlib.sha256(shim.read_bytes()).hexdigest() != digest:
            errors.append(f"lbm compatibility shim changed: {path}")

    actual_tree = _git("rev-parse", f"HEAD:{policy['historical_root']}")
    if actual_tree != policy["historical_tree_oid"]:
        errors.append(
            "historical checkout tree changed: "
            f"expected {policy['historical_tree_oid']}, got {actual_tree}"
        )

    actual_refs = collect_active_references(policy)
    expected_refs = policy["allowed_active_references"]
    for path in sorted(actual_refs.keys() - expected_refs.keys()):
        errors.append(f"new active nested-source reference: {path}")
    for path in sorted(expected_refs.keys() - actual_refs.keys()):
        errors.append(f"recorded nested-source reference changed or was removed: {path}")
    for path in sorted(actual_refs.keys() & expected_refs.keys()):
        if actual_refs[path] != expected_refs[path]:
            errors.append(f"nested-source reference lines changed: {path}")
    return errors


def refresh_reference_baseline(policy: dict) -> None:
    policy["allowed_active_references"] = collect_active_references(policy)
    POLICY.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-reference-baseline",
        action="store_true",
        help="mechanically refresh reviewed active-consumer evidence",
    )
    args = parser.parse_args()
    policy = load_policy()
    if args.refresh_reference_baseline:
        refresh_reference_baseline(policy)
        return 0
    errors = validate(policy)
    if errors:
        print("Source-authority violations:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(
        "Source authority OK: canonical wheel root is src; historical tree is frozen; "
        f"{len(policy['allowed_active_references'])} existing active references remain explicit."
    )
    print("This gate does not declare those historical consumers migrated or scientifically valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
