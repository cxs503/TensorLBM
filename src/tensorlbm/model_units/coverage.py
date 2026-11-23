"""Manifest-kind coverage: *every* model directory in the repo is a CI unit.

Why this module exists
----------------------
CFD-GO/TCLB's rule is that a model directory is the build and CI unit, and its
enumeration is generated (``src/models.R`` walks ``models/**/conf.mk``).  The
failure mode the rule is meant to prevent is *drift*: in TCLB itself
``models.R`` enumerates 61 model directories while ``cpu_test.yml`` lists 16 --
the other 45 are buildable but never exercised by CI.

TensorLBM grew two manifest dialects for the same convention, from two
independent deliveries of the same anchor:

``benchmarks/model_units/lbm/<name>/model.toml``   (+ ``case.py``)   -> ``tensorlbm.model_units``
``benchmarks/model_units/units/<id>/model.json``   (kernel + gates)  -> ``tensorlbm.models.ci_unit``

The generated matrix in ``.github/workflows/model-units.yml`` was produced from
the first dialect only, so the ``model.json`` directories satisfied the
"add a directory and it is a CI unit" promise on paper and not in fact: adding
one changed no job, and a broken one could not fail a job.  That is exactly the
TCLB drift the anchor is about, reproduced locally.

This module is the gate for it.  It enumerates every directory carrying a
*supported* manifest from the filesystem **and** the git index (the union
TCLB's ``models.R`` uses, so a not-yet-committed unit still builds locally),
maps each directory onto a generated matrix cell, and reports

``uncovered``   directory with a supported manifest, no matrix cell   -> fail
``orphans``     matrix cell whose directory/manifest is gone          -> fail
``duplicates``  two directories resolving to the same unit id        -> fail

Nothing here is a hand-maintained list: the dialects and their cell commands
are declared once, below, and the inventory comes from the repository.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .paths import MODEL_UNITS_ROOT, resolve_model_unit_root

TOML_KIND = "unit_toml"
JSON_KIND = "unit_json"

#: dialect -> manifest file name.  Adding a dialect is one entry here plus its
#: cell command in :func:`cell_command`; no unit is ever named by hand.
MANIFEST_KINDS: dict[str, str] = {TOML_KIND: "model.toml", JSON_KIND: "model.json"}

#: schema string required by the ``model.json`` dialect (strict, same value as
#: :mod:`tensorlbm.models.ci_unit`; duplicated as a literal so that the plan job
#: can validate the manifest *without* importing torch).
JSON_SCHEMA = "tensorlbm.model_unit/v1"

DEFAULT_MODELS_ROOT = MODEL_UNITS_ROOT


class CoverageError(RuntimeError):
    """Raised when a manifest directory cannot be classified (fail closed)."""


@dataclass(frozen=True)
class ModelDirectory:
    """One model directory and the manifest dialect it speaks.

    ``path`` is absolute (filesystem work); ``rel_dir`` is the repo-relative
    path used as the join key between directories and matrix cells.
    """

    path: Path
    manifest: Path
    kind: str
    unit_id: str
    rel_dir: str

    def to_dict(self) -> dict[str, str]:
        return {
            "dir": self.rel_dir,
            "manifest": self.manifest.as_posix(),
            "kind": self.kind,
            "unit_id": self.unit_id,
        }


def _git_manifests(repo_root: Path) -> list[Path]:
    """Union half: manifests tracked in the git index (empty when no git)."""
    patterns = [f"*{name}" for name in MANIFEST_KINDS.values()]
    try:
        out = subprocess.run(
            ["git", "ls-files", *patterns],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git binary
        return []
    if out.returncode != 0:
        return []
    return [repo_root / line.strip() for line in out.stdout.splitlines() if line.strip()]


def _filesystem_manifests(models_root: Path) -> list[Path]:
    if not models_root.is_dir():
        return []
    found: list[Path] = []
    for name in MANIFEST_KINDS.values():
        found.extend(sorted(models_root.rglob(name)))
    return found


def _json_unit_id(manifest: Path) -> str:
    """Read ``unit_id`` from a ``model.json`` without importing the kernel stack."""
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CoverageError(f"{manifest}: unreadable model.json ({exc})") from exc
    if not isinstance(payload, Mapping):
        raise CoverageError(f"{manifest}: model.json must be a JSON object")
    schema = payload.get("schema")
    if schema != JSON_SCHEMA:
        raise CoverageError(f"{manifest}: schema must be {JSON_SCHEMA!r}, got {schema!r}")
    unit_id = payload.get("unit_id")
    if not isinstance(unit_id, str) or not unit_id.strip():
        raise CoverageError(f"{manifest}: unit_id must be a non-empty string")
    return unit_id


def _toml_unit_id(manifest: Path) -> str:
    """Read ``[unit].name`` from a ``model.toml`` (the manifest module owns rules)."""
    from .manifest import ManifestError, parse_manifest

    try:
        return str(parse_manifest(manifest)["unit"]["name"])
    except (OSError, ManifestError, KeyError, TypeError) as exc:
        raise CoverageError(f"{manifest}: unreadable model.toml ({exc})") from exc


def _unit_id(manifest: Path, kind: str) -> str:
    return _toml_unit_id(manifest) if kind == TOML_KIND else _json_unit_id(manifest)


def discover_directories(
    models_root: Path | str = DEFAULT_MODELS_ROOT,
    *,
    repo_root: Path | str | None = None,
) -> list[ModelDirectory]:
    """Enumerate every model directory under ``models_root`` (git ∪ filesystem).

    Sorted by ``(kind, unit_id)`` so the emitted matrix is reproducible.
    """
    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    root = resolve_model_unit_root(models_root, repo)
    manifest_to_kind = {name: kind for kind, name in MANIFEST_KINDS.items()}

    candidates: dict[Path, None] = {}
    for path in _filesystem_manifests(root):
        candidates[path.resolve()] = None
    try:
        root_resolved = root.resolve()
    except OSError:  # pragma: no cover - defensive
        root_resolved = root
    for path in _git_manifests(repo):
        try:
            path.resolve().relative_to(root_resolved)
        except (ValueError, OSError):
            continue
        candidates[path.resolve()] = None

    directories: list[ModelDirectory] = []
    for manifest in sorted(candidates):
        kind = manifest_to_kind.get(manifest.name)
        if kind is None:  # pragma: no cover - guarded by the glob patterns
            continue
        directories.append(
            ModelDirectory(
                path=manifest.parent,
                manifest=manifest,
                kind=kind,
                unit_id=_unit_id(manifest, kind),
                rel_dir=_display(manifest.parent, root, repo),
            )
        )
    return sorted(directories, key=lambda d: (d.kind, d.unit_id))


def _display(path: Path, models_root: Path, repo_root: Path) -> str:
    """Prefer a repo-relative path, fall back to the absolute one."""
    for base in (repo_root, Path.cwd(), models_root.parent):
        try:
            return path.resolve().relative_to(base.resolve()).as_posix()
        except (ValueError, OSError):
            continue
    return path.as_posix()


def cell_command(kind: str, unit_id: str, *, python: str = "python") -> str:
    """The exact command a matrix cell runs for one dialect."""
    if kind == TOML_KIND:
        return (
            f"PYTHONPATH=src {python} -m tensorlbm.model_units build-and-test "
            f"--unit {unit_id}"
        )
    if kind == JSON_KIND:
        return (
            f"PYTHONPATH=src {python} -m tensorlbm.models.ci_unit ci "
            f"--only {unit_id} --out artifacts/model_units"
        )
    raise CoverageError(f"unsupported manifest kind {kind!r}")


def _json_metadata(manifest: Path) -> dict[str, Any]:
    """Lattice/dimension for the matrix label (never used for gating)."""
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        composition = payload.get("composition") or {}
        lattice = str(composition.get("lattice") or "unknown")
        collision = str(composition.get("collision") or "unknown")
    except (OSError, json.JSONDecodeError):  # pragma: no cover - validated earlier
        return {"lattice": "unknown", "dimension": "?", "tier": "unknown"}
    dimension = "2" if lattice.startswith("D2") else "3" if lattice.startswith("D3") else "?"
    return {"lattice": lattice, "dimension": dimension, "tier": collision}


def matrix_cells(
    directories: Iterable[ModelDirectory],
    *,
    models_root: Path | str = DEFAULT_MODELS_ROOT,
    repo_root: Path | str | None = None,
    python: str = "python",
) -> list[dict[str, str]]:
    """One generated matrix cell per model directory, whatever its dialect."""
    root = Path(models_root)
    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    cells: list[dict[str, str]] = []
    for directory in directories:
        cell = {
            "unit": directory.unit_id,
            "dir": directory.rel_dir,
            "kind": directory.kind,
            "group": directory.kind,
            "cmd": cell_command(directory.kind, directory.unit_id, python=python),
        }
        if directory.kind == TOML_KIND:
            from .manifest import load_unit

            unit = load_unit(directory.manifest)
            cell.update(
                {
                    "group": unit.group,
                    "lattice": unit.lattice,
                    "dimension": str(unit.dimension),
                    "tier": unit.tier,
                }
            )
        else:
            cell.update(_json_metadata(directory.manifest))
        cells.append(cell)
    return cells


def coverage(
    directories: Iterable[ModelDirectory],
    cells: Iterable[Mapping[str, str]],
    *,
    models_root: Path | str = DEFAULT_MODELS_ROOT,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Map directories onto matrix cells and report every drift class."""
    root = Path(models_root)
    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    dirs = list(directories)
    cell_list = [dict(cell) for cell in cells]

    wanted = {d.rel_dir: d for d in dirs}
    covered: list[str] = []
    orphans: list[str] = []
    for cell in cell_list:
        rel = cell.get("dir")
        if rel in wanted:
            covered.append(rel)
        else:  # a cell points at a directory that no longer carries a manifest
            orphans.append(rel or cell.get("unit", "<unnamed>"))

    uncovered = sorted(set(wanted) - set(covered))
    duplicates: list[str] = []
    seen: dict[str, str] = {}
    for d in dirs:
        if d.unit_id in seen and seen[d.unit_id] != d.rel_dir:
            duplicates.append(f"{d.unit_id}: {seen[d.unit_id]} vs {d.rel_dir}")
        else:
            seen[d.unit_id] = d.rel_dir

    by_kind: dict[str, int] = {}
    for d in dirs:
        by_kind[d.kind] = by_kind.get(d.kind, 0) + 1
    cells_by_kind: dict[str, int] = {}
    for cell in cell_list:
        kind = str(cell.get("kind", "unknown"))
        cells_by_kind[kind] = cells_by_kind.get(kind, 0) + 1

    return {
        "kind": "model_unit_coverage",
        "models_root": root.as_posix(),
        "directories": len(dirs),
        "cells": len(cell_list),
        "by_kind": by_kind,
        "cells_by_kind": cells_by_kind,
        "covered": sorted(covered),
        "uncovered": uncovered,
        "orphans": sorted(orphans),
        "duplicates": sorted(duplicates),
        "ok": not uncovered and not orphans and not duplicates,
    }


def coverage_report(
    *,
    models_root: Path | str = DEFAULT_MODELS_ROOT,
    repo_root: Path | str | None = None,
    python: str = "python",
) -> dict[str, Any]:
    """Convenience: discover, build the cell list, and gate it."""
    directories = discover_directories(models_root, repo_root=repo_root)
    cells = matrix_cells(
        directories, models_root=models_root, repo_root=repo_root, python=python
    )
    report = coverage(
        directories, cells, models_root=models_root, repo_root=repo_root
    )
    report["cells_detail"] = cells
    return report


__all__ = [
    "DEFAULT_MODELS_ROOT",
    "JSON_KIND",
    "JSON_SCHEMA",
    "MANIFEST_KINDS",
    "TOML_KIND",
    "CoverageError",
    "ModelDirectory",
    "cell_command",
    "coverage",
    "coverage_report",
    "discover_directories",
    "matrix_cells",
]