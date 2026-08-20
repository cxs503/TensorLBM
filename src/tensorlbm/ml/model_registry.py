"""Model asset layer: a checkpoint registry with store conventions and rich metadata.

Problem this module solves (AI4S platform gap "training artifacts have no
home"): trained weights previously lived only as loose ``*.ckpt`` files (e.g.
``checkpoints/suboff/*.ckpt``) with no registry convention — no task label, no
metrics, no link back to the dataset product that produced them, and no
programmatic way to load them back.  The serving registry
(:class:`tensorlbm.ml.serving.ModelRegistry`) records *serving descriptors* in
the ``models`` table, but it does not version checkpoint *assets*.

This module adds the missing asset layer on the ``ml/`` side (``data/`` is
untouched):

* a checkpoint **store layout convention** — ``<root>/<task>/<model_id>/``
  holds the weight file plus a human-readable ``meta.json`` sidecar, so a
  store remains navigable even without the index;
* an independent SQLite index (default ``<root>/model_registry.db``) with
  fail-closed metadata validation;
* a stable API — ``register(path, meta) -> model_id`` / ``list_models(filter)``
  / ``load_model(model_id)`` — plus lifecycle (stage transitions, archive) and
  serving cross-links (``link_serving_model``).

Metadata captured for every asset: ``task``, ``metrics``, the dataset
``product_id`` it was trained on, the ``git_sha`` (auto-captured when not
supplied) and the creation time — exactly the fields the platform needs to
trace a deployed model back to its data.

Storage is plain SQLite + JSON sidecars, consistent with the platform's other
registries (``tensorlbm.ai.database``, ``tensorlbm.data.catalog``); only the
standard library plus ``torch`` is required.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping
from uuid import uuid4

import torch
from torch import nn

__all__ = [
    "ModelAsset",
    "ModelAssetRegistry",
    "VALID_STAGES",
    "register_family_loader",
]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_assets (
    model_id           TEXT PRIMARY KEY,
    name               TEXT NOT NULL,
    task               TEXT NOT NULL,
    family             TEXT NOT NULL DEFAULT 'unknown',
    framework          TEXT NOT NULL DEFAULT 'torch',
    stage              TEXT NOT NULL DEFAULT 'development',
    checkpoint_path    TEXT NOT NULL,
    artifact_relpath   TEXT NOT NULL DEFAULT '',
    original_path      TEXT NOT NULL DEFAULT '',
    metrics_json       TEXT,
    arch_json          TEXT,
    dataset_product_id TEXT,
    training_job_id    TEXT,
    serving_model_id   INTEGER,
    git_sha            TEXT NOT NULL DEFAULT 'unknown',
    git_dirty          INTEGER NOT NULL DEFAULT 0,
    tags               TEXT NOT NULL DEFAULT '[]',
    description        TEXT NOT NULL DEFAULT '',
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_model_assets_task     ON model_assets(task);
CREATE INDEX IF NOT EXISTS idx_model_assets_family   ON model_assets(family);
CREATE INDEX IF NOT EXISTS idx_model_assets_dataset  ON model_assets(dataset_product_id);
"""

VALID_STAGES: tuple[str, ...] = ("development", "staging", "production", "archived")

# Lifecycle transitions.  ``archived`` is terminal; ``production`` cannot be
# silently demoted (re-register a new asset instead).
_STAGE_TRANSITIONS: dict[str, frozenset[str]] = {
    "development": frozenset({"development", "staging", "archived"}),
    "staging": frozenset({"staging", "development", "production", "archived"}),
    "production": frozenset({"production", "archived"}),
    "archived": frozenset(),
}

# Keys accepted by :meth:`ModelAssetRegistry.register`.  Fail closed on
# anything else so a typo cannot silently drop lineage metadata.
_ALLOWED_META_KEYS = frozenset({
    "model_id",
    "task",
    "name",
    "family",
    "framework",
    "metrics",
    "arch",
    "dataset_product_id",
    "training_job_id",
    "tags",
    "description",
    "git_sha",
    "stage",
    "copy",
})

_CHECKPOINT_FILENAME = "checkpoint"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ModelAsset:
    """One registered model checkpoint; an immutable view of an index row."""

    model_id: str
    name: str
    task: str
    family: str = "unknown"
    framework: str = "torch"
    stage: str = "development"
    checkpoint_path: str = ""
    artifact_relpath: str = ""
    original_path: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    arch: dict[str, Any] = field(default_factory=dict)
    dataset_product_id: str | None = None
    training_job_id: str | None = None
    serving_model_id: int | None = None
    git_sha: str = "unknown"
    git_dirty: bool = False
    tags: tuple[str, ...] = ()
    description: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the sidecar / report representation."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "task": self.task,
            "family": self.family,
            "framework": self.framework,
            "stage": self.stage,
            "checkpoint_path": self.checkpoint_path,
            "artifact_relpath": self.artifact_relpath,
            "original_path": self.original_path,
            "metrics": dict(self.metrics or {}),
            "arch": dict(self.arch or {}),
            "dataset_product_id": self.dataset_product_id,
            "training_job_id": self.training_job_id,
            "serving_model_id": self.serving_model_id,
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
            "tags": list(self.tags),
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(UTC).isoformat()


def _require_text(value: object, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validate_stage(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError("stage must be a string")
    if value not in VALID_STAGES:
        raise ValueError(f"stage must be one of {VALID_STAGES}")
    return value


def _encode_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _decode_json(value: str | None, default: Any) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _capture_git_sha() -> tuple[str, bool]:
    """Best-effort capture of the repository SHA the registration happens in.

    Returns ``(short_sha, dirty)``; ``("unknown", False)`` when the store is
    used outside a git checkout (e.g. an installed wheel).
    """
    repo_root = Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.run(  # noqa: S603 - fixed argv, repo-local
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        if not sha:
            return "unknown", False
        status = subprocess.run(  # noqa: S603
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
        return sha[:12], bool(status.strip())
    except Exception:
        return "unknown", False


def _row_to_asset(row: sqlite3.Row) -> ModelAsset:
    return ModelAsset(
        model_id=row["model_id"],
        name=row["name"],
        task=row["task"],
        family=row["family"],
        framework=row["framework"],
        stage=row["stage"],
        checkpoint_path=row["checkpoint_path"],
        artifact_relpath=row["artifact_relpath"],
        original_path=row["original_path"],
        metrics=_decode_json(row["metrics_json"], {}),
        arch=_decode_json(row["arch_json"], {}),
        dataset_product_id=row["dataset_product_id"],
        training_job_id=row["training_job_id"],
        serving_model_id=row["serving_model_id"],
        git_sha=row["git_sha"],
        git_dirty=bool(row["git_dirty"]),
        tags=tuple(_decode_json(row["tags"], [])),
        description=row["description"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


# ---------------------------------------------------------------------------
# Family loaders (checkpoint path -> nn.Module)
# ---------------------------------------------------------------------------

_FAMILY_LOADERS: dict[str, Callable[[str], nn.Module]] = {}


def register_family_loader(family: str, loader: Callable[[str], nn.Module]) -> None:
    """Teach :meth:`ModelAssetRegistry.load_model` a new checkpoint family."""
    if not isinstance(family, str) or not family.strip():
        raise ValueError("family must be a non-empty string")
    if not callable(loader):
        raise TypeError("loader must be callable")
    _FAMILY_LOADERS[family] = loader


def _load_by_family(family: str, path: str) -> nn.Module:
    loader = _FAMILY_LOADERS.get(family)
    if loader is not None:
        return loader(path)
    # Fall back to the serving layer's dispatch (fno2d, flow_transformer_ssl,
    # eddy_viscosity_mlp, pinn, gnn, inverse, diffusion, uq) so the asset
    # layer and the serving layer agree on checkpoint formats.
    from tensorlbm.ml.serving import InferenceService

    return InferenceService._load_by_family(family, path)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ModelAssetRegistry:
    """Registry for model checkpoint assets with a directory-store convention.

    The store root holds one SQLite index (``model_registry.db``) plus the
    checkpoint layout ``<root>/<task>/<model_id>/checkpoint.*`` with a
    ``meta.json`` sidecar next to every weight file.
    """

    def __init__(self, root: str | Path, conn: sqlite3.Connection) -> None:
        self.root = Path(root)
        self._conn = conn
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @classmethod
    def open(
        cls,
        root: str | Path,
        *,
        db_path: str | Path | None = None,
    ) -> "ModelAssetRegistry":
        """Open (and if necessary create) a model store rooted at ``root``."""
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        index_path = Path(db_path) if db_path is not None else root_path / "model_registry.db"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(index_path))
        return cls(root_path, conn)

    def close(self) -> None:
        self._conn.close()

    # -- registration --------------------------------------------------------

    def register(
        self,
        path: str | Path,
        meta: Mapping[str, Any],
    ) -> str:
        """Register a checkpoint file and return its ``model_id``.

        Args:
            path: Existing checkpoint file.  By default it is *copied* into the
                store (``<root>/<task>/<model_id>/checkpoint<suffix>``); pass
                ``meta["copy"] = False`` to register the original location
                by-reference instead (for huge files).
            meta: Asset metadata.  Allowed keys: ``model_id`` (optional
                explicit id), ``task``, ``name``, ``family``, ``framework``,
                ``metrics``, ``arch``, ``dataset_product_id``,
                ``training_job_id``, ``tags``, ``description``, ``git_sha``,
                ``stage``, ``copy``.  ``task`` and ``name`` are required;
                unknown keys raise so typos cannot silently lose lineage.
        """
        if not isinstance(meta, Mapping):
            raise TypeError("meta must be a mapping")
        unknown = sorted(set(meta) - _ALLOWED_META_KEYS)
        if unknown:
            raise ValueError(f"unsupported meta keys: {unknown}; allowed: {sorted(_ALLOWED_META_KEYS)}")

        task = _require_text(meta.get("task"), "meta['task']")
        name = _require_text(meta.get("name"), "meta['name']")
        family = str(meta.get("family") or "unknown")
        framework = str(meta.get("framework") or "torch")
        stage = _validate_stage(meta.get("stage", "development"))
        metrics = dict(meta.get("metrics") or {})
        arch = dict(meta.get("arch") or {})
        if not isinstance(metrics, dict) or not isinstance(arch, dict):
            raise TypeError("metrics and arch must be mappings")
        dataset_product_id = meta.get("dataset_product_id")
        if dataset_product_id is not None:
            dataset_product_id = _require_text(dataset_product_id, "meta['dataset_product_id']")
        training_job_id = meta.get("training_job_id")
        if training_job_id is not None:
            training_job_id = _require_text(training_job_id, "meta['training_job_id']")
        tags = tuple(str(t) for t in (meta.get("tags") or ()))
        description = str(meta.get("description") or "")
        copy = meta.get("copy", True)
        if not isinstance(copy, bool):
            raise TypeError("meta['copy'] must be a bool")

        git_sha = meta.get("git_sha")
        if git_sha is not None:
            git_sha, git_dirty = _require_text(git_sha, "meta['git_sha']"), False
        else:
            git_sha, git_dirty = _capture_git_sha()

        src = Path(path)
        if not src.is_file():
            raise FileNotFoundError(f"checkpoint file not found: {src}")

        model_id = str(meta.get("model_id") or f"mdl_{uuid4().hex[:12]}")
        model_id = _require_text(model_id, "meta['model_id']")
        if self.get_model(model_id) is not None:
            raise ValueError(f"model id {model_id!r} already exists")

        now = _now()
        if copy:
            dest_dir = self.root / task / model_id
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{_CHECKPOINT_FILENAME}{src.suffix or '.pt'}"
            shutil.copy2(src, dest)
            # Companion metadata written next to the weights by savers such as
            # ``tensorlbm.ai.fno.save_fno2d`` (``<ckpt>.pt.json`` holding the
            # arch) must travel with the weights or loaders fall back to
            # default architectures and fail on ``load_state_dict``.
            companion = src.with_suffix(src.suffix + ".json")
            if companion.is_file():
                shutil.copy2(companion, dest.with_suffix(dest.suffix + ".json"))
            checkpoint_path = str(dest.resolve())
            artifact_relpath = str(dest.relative_to(self.root))
        else:
            checkpoint_path = str(src.resolve())
            artifact_relpath = ""

        asset = ModelAsset(
            model_id=model_id, name=name, task=task, family=family,
            framework=framework, stage=stage,
            checkpoint_path=checkpoint_path, artifact_relpath=artifact_relpath,
            original_path=str(src.resolve()), metrics=metrics, arch=arch,
            dataset_product_id=dataset_product_id,
            training_job_id=training_job_id, serving_model_id=None,
            git_sha=git_sha, git_dirty=git_dirty, tags=tags,
            description=description, created_at=now, updated_at=now,
        )
        self._insert(asset)
        self._write_sidecar(asset)
        return model_id

    def _insert(self, asset: ModelAsset) -> None:
        self._conn.execute(
            "INSERT INTO model_assets "
            "(model_id, name, task, family, framework, stage, checkpoint_path, "
            " artifact_relpath, original_path, metrics_json, arch_json, "
            " dataset_product_id, training_job_id, serving_model_id, git_sha, "
            " git_dirty, tags, description, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                asset.model_id, asset.name, asset.task, asset.family,
                asset.framework, asset.stage, asset.checkpoint_path,
                asset.artifact_relpath, asset.original_path,
                _encode_json(asset.metrics), _encode_json(asset.arch),
                asset.dataset_product_id, asset.training_job_id,
                asset.serving_model_id, asset.git_sha,
                int(asset.git_dirty), _encode_json(list(asset.tags)),
                asset.description, asset.created_at, asset.updated_at,
            ),
        )
        self._conn.commit()

    def _write_sidecar(self, asset: ModelAsset) -> None:
        """Write the ``meta.json`` sidecar next to a stored checkpoint."""
        if not asset.artifact_relpath:
            return  # by-reference asset: the store owns no directory
        sidecar = self.root / asset.artifact_relpath
        sidecar = sidecar.parent / "meta.json"
        sidecar.write_text(json.dumps(asset.to_dict(), indent=2, ensure_ascii=False, default=str))

    # -- queries -------------------------------------------------------------

    def get_model(self, model_id: str) -> ModelAsset | None:
        row = self._conn.execute(
            "SELECT * FROM model_assets WHERE model_id = ?",
            (_require_text(model_id, "model_id"),),
        ).fetchone()
        return _row_to_asset(row) if row else None

    def _require_model(self, model_id: str) -> ModelAsset:
        asset = self.get_model(model_id)
        if asset is None:
            raise KeyError(f"no model asset {model_id!r}")
        return asset

    def list_models(
        self,
        *,
        task: str | None = None,
        family: str | None = None,
        name_contains: str | None = None,
        dataset_product_id: str | None = None,
        training_job_id: str | None = None,
        tag: str | None = None,
        stage: str | None = None,
        include_archived: bool = False,
        limit: int = 50,
    ) -> list[ModelAsset]:
        """List registered model assets, newest first.

        Every argument is an optional equality (or substring for
        ``name_contains`` / containment for ``tag``) filter; archived assets
        are excluded unless ``include_archived`` is set.
        """
        query = "SELECT * FROM model_assets WHERE 1=1"
        args: list[Any] = []
        if not include_archived:
            query += " AND stage != 'archived'"
        if task is not None:
            query += " AND task = ?"
            args.append(str(task))
        if family is not None:
            query += " AND family = ?"
            args.append(str(family))
        if name_contains is not None:
            query += " AND name LIKE ?"
            args.append(f"%{name_contains}%")
        if dataset_product_id is not None:
            query += " AND dataset_product_id = ?"
            args.append(str(dataset_product_id))
        if training_job_id is not None:
            query += " AND training_job_id = ?"
            args.append(str(training_job_id))
        if tag is not None:
            query += " AND tags LIKE ?"
            args.append(f'%"{tag}"%')
        if stage is not None:
            query += " AND stage = ?"
            args.append(_validate_stage(stage))
        query += " ORDER BY created_at DESC, model_id DESC LIMIT ?"
        args.append(int(limit))
        rows = self._conn.execute(query, args).fetchall()
        return [_row_to_asset(r) for r in rows]

    # -- loading -------------------------------------------------------------

    def load_model(self, model_id: str) -> nn.Module:
        """Load a registered checkpoint back into an eval-mode ``nn.Module``.

        Dispatch is by the asset's ``family``: custom loaders registered with
        :func:`register_family_loader` take precedence; otherwise the serving
        layer's family dispatch is reused.
        """
        asset = self._require_model(model_id)
        ckpt = Path(asset.checkpoint_path)
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"checkpoint for {model_id!r} is missing on disk: {ckpt}"
            )
        model = _load_by_family(asset.family, str(ckpt))
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"family {asset.family!r} loader returned "
                f"{type(model).__name__}, expected an nn.Module"
            )
        model.eval()
        return model

    # -- updates -------------------------------------------------------------

    def record_metrics(self, model_id: str, metrics: Mapping[str, Any]) -> ModelAsset:
        """Merge ``metrics`` into the asset's existing metrics mapping."""
        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping")
        asset = self._require_model(model_id)
        merged = dict(asset.metrics or {})
        merged.update(dict(metrics))
        return self._update_field(model_id, "metrics_json", _encode_json(merged))

    def set_stage(self, model_id: str, stage: str) -> ModelAsset:
        """Transition the asset through its lifecycle (validated)."""
        stage = _validate_stage(stage)
        asset = self._require_model(model_id)
        if stage not in _STAGE_TRANSITIONS[asset.stage]:
            raise ValueError(f"invalid stage transition {asset.stage!r} -> {stage!r}")
        return self._update_field(model_id, "stage", stage)

    def archive(self, model_id: str) -> ModelAsset:
        """Archive an asset (terminal; hidden from default ``list_models``)."""
        return self.set_stage(model_id, "archived")

    def link_serving_model(self, model_id: str, serving_model_id: int) -> ModelAsset:
        """Record which serving-registry id serves this asset (loop closure)."""
        if isinstance(serving_model_id, bool) or not isinstance(serving_model_id, int):
            raise TypeError("serving_model_id must be an int")
        self._require_model(model_id)
        return self._update_field(model_id, "serving_model_id", int(serving_model_id))

    def _update_field(self, model_id: str, field: str, value: Any) -> ModelAsset:
        if field not in {"metrics_json", "stage", "serving_model_id"}:
            raise ValueError(f"field {field!r} is not updatable")
        self._conn.execute(
            f"UPDATE model_assets SET {field} = ?, updated_at = ? WHERE model_id = ?",
            (value, _now(), model_id),
        )
        self._conn.commit()
        asset = self._require_model(model_id)
        self._write_sidecar(asset)
        return asset
