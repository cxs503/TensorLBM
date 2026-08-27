"""Model zoo: a manifest-driven registry of trained AI artifacts.

The AI side of TensorLBM grew four independent persistence conventions —
:func:`tensorlbm.ai.drag_surrogate.save_drag_regressor`,
:func:`tensorlbm.ai.fno.save_fno2d`, :func:`tensorlbm.ai.model.save_model`
and :func:`tensorlbm.ai.transformer.save_flow_transformer_model` — each
writing a weight file plus a JSON sidecar, but with no shared place for the
*provenance* of a trained artifact: which task it serves, on which dataset
(split included) it was trained, which metrics it achieved, and which code
revision produced it.  Trained weights therefore lived as loose files with no
answer to "which of these is the good one?".

The zoo fills that gap without inventing yet another serialization format.
Each entry is a plain directory inside a zoo root (default
``~/.tensorlbm/zoo``, overridable via the ``TENSORLBM_ZOO_ROOT`` environment
variable or an explicit path)::

    <zoo_root>/
      suboff-drag-fno-v1/          # entry dir == model_id
        model.pt                   # the artifact, copied/moved in on register
        model.pt.json              # saver sidecar (arch/norm), travels with it
        model.json                 # the zoo manifest (schema below)

The manifest is the single source of truth; there is no index database, so a
zoo directory stays self-describing and shareable (rsync/zip the folder).
Loading *reuses* the existing per-family loaders: the manifest stores a
``"module:attr"`` import string (e.g.
``"tensorlbm.ai.drag_surrogate:load_drag_regressor"``) that :meth:`ModelZoo.load`
resolves dynamically.

.. warning::
   ``load()`` executes the loader named in the manifest — effectively
   ``importlib.import_module(...)`` plus a call with arbitrary effects.
   Only register artifacts from sources you trust, and prefer the curated
   loader strings in :data:`SUGGESTED_LOADERS`.  Artifact integrity is
   protected by a SHA-256 recorded at registration time and re-checked by
   :meth:`ModelZoo.validate`.

Relationship to :class:`tensorlbm.ml.model_registry.ModelAssetRegistry`: the
``ml/`` asset registry is the platform-side index (SQLite, stage lifecycle,
serving cross-links) over training-job outputs; the zoo is the lighter,
file-manifest layer for *publishable* weight sets with standard evaluation
numbers.  Both dispatch to the same underlying ``save_*``/``load_*``
conventions, so an artifact can move between them without re-saving.

This module deliberately depends only on the standard library, so the zoo
schema and registry remain importable in minimal environments; importing the
loaders (and therefore ``torch``) happens lazily inside ``load``/``validate``.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

__all__ = [
    "KNOWN_TASKS",
    "ModelInfo",
    "SUGGESTED_LOADERS",
    "ZOO_SCHEMA_VERSION",
    "ZooError",
    "ZooManifestError",
    "ZooValidation",
    "ModelZoo",
    "info",
    "list_models",
    "load",
    "register",
    "resolve_zoo_root",
    "validate",
]

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

#: Manifest schema version understood by this module.  Manifests declaring a
#: different version are rejected (fail-closed) instead of guessed at.
ZOO_SCHEMA_VERSION = 1

#: Environment variable overriding the default zoo root.
ZOO_ROOT_ENV = "TENSORLBM_ZOO_ROOT"

#: Canonical manifest filename inside an entry directory.
MANIFEST_FILENAME = "model.json"

#: Tasks the zoo knows suggested loaders for.  Registering other task labels
#: is allowed (the field is free-form), but sticking to these keeps the zoo
#: greppable across projects.
KNOWN_TASKS: tuple[str, ...] = (
    "drag-surrogate",
    "eddy-viscosity",
    "flow-transformer",
    "fno2d",
)

#: Curated ``module:attr`` strings pointing at the existing (reused, not
#: re-implemented) ``load_*`` functions of :mod:`tensorlbm.ai`.
SUGGESTED_LOADERS: dict[str, str] = {
    "drag-surrogate": "tensorlbm.ai.drag_surrogate:load_drag_regressor",
    "eddy-viscosity": "tensorlbm.ai.model:load_model",
    "flow-transformer": "tensorlbm.ai.transformer:load_flow_transformer_model",
    "fno2d": "tensorlbm.ai.fno:load_fno2d",
}

# model_id / task: lowercase kebab- or snake-case identifier.
_MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_TASK_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
# loader: dotted python module path, colon, attribute name.
_LOADER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")
# SHA-256 hex digest.
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
# ISO-8601 timestamp with timezone information.
_ISO_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[Tt ]\d{2}:\d{2}:\d{2}(\.\d+)?([Zz]|[+-]\d{2}:?\d{2})$")

_ALLOWED_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "model_id",
        "task",
        "loader",
        "artifact",
        "artifact_sha256",
        "artifact_bytes",
        "artifact_companion",
        "metrics",
        "dataset",
        "code_sha",
        "created_at",
        "notes",
    }
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ZooError(ValueError):
    """Base class for zoo failures (registration, lookup, validation)."""


class ZooManifestError(ZooError):
    """A zoo manifest violates the schema (unreadable, missing, or extra fields)."""


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelInfo:
    """One zoo entry; an immutable view of its manifest.

    ``entry_dir`` (and the derived :attr:`artifact_path`) are filesystem
    locations, not manifest fields — they depend on which zoo root the entry
    was read from.
    """

    model_id: str
    task: str
    loader: str
    artifact: str
    artifact_sha256: str
    artifact_bytes: int
    entry_dir: str
    artifact_companion: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] | None = None
    code_sha: str = "unknown"
    created_at: str = ""
    notes: str = ""

    @property
    def artifact_path(self) -> Path:
        """Absolute path of the weight file."""
        return Path(self.entry_dir) / self.artifact

    def to_dict(self) -> dict[str, Any]:
        """Serialise back to the manifest representation."""
        return {
            "schema_version": ZOO_SCHEMA_VERSION,
            "model_id": self.model_id,
            "task": self.task,
            "loader": self.loader,
            "artifact": self.artifact,
            "artifact_sha256": self.artifact_sha256,
            "artifact_bytes": self.artifact_bytes,
            "artifact_companion": self.artifact_companion,
            "metrics": dict(self.metrics),
            "dataset": dict(self.dataset) if self.dataset is not None else None,
            "code_sha": self.code_sha,
            "created_at": self.created_at,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ZooValidation:
    """Result of :meth:`ModelZoo.validate`: per-check outcomes plus errors."""

    model_id: str
    checks: dict[str, bool] = field(default_factory=dict)
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True when every check passed and no error was collected."""
        return bool(self.checks) and all(self.checks.values()) and not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Report representation (for logs / JSON dumps)."""
        return {
            "model_id": self.model_id,
            "ok": self.ok,
            "checks": dict(self.checks),
            "errors": list(self.errors),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_zoo_root(root: str | Path | None = None) -> Path:
    """Resolve the zoo root: explicit ``root`` > ``TENSORLBM_ZOO_ROOT`` > home.

    The path is *not* created here; :meth:`ModelZoo.register` creates it (and
    the entry directory) on demand, so read-only operations on a missing zoo
    simply report no models.
    """
    if root is not None:
        return Path(root)
    env = os.environ.get(ZOO_ROOT_ENV)
    if env:
        return Path(env)
    return Path.home() / ".tensorlbm" / "zoo"


def _require_plain_filename(value: object, name: str) -> str:
    """Require a bare filename so a crafted manifest cannot traverse paths."""
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise ZooManifestError(f"{name} must be a non-empty string")
    if value in {".", ".."} or "/" in value or "\\" in value or os.sep in value:
        raise ZooManifestError(f"{name} must be a bare filename, got {value!r}")
    return value


def _validate_scalar_metric(key: str, value: object) -> None:
    if isinstance(value, bool) or isinstance(value, (int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ZooManifestError(f"metrics[{key!r}] must be finite, got {value!r}")
        return
    raise ZooManifestError(
        f"metrics[{key!r}] must be a JSON scalar (numbers/strings/bools), "
        f"got {type(value).__name__}"
    )


def _validate_manifest(data: object, *, source: str) -> dict[str, Any]:
    """Strict schema validation of a manifest mapping.

    Raises :class:`ZooManifestError` describing the first violation; returns
    the mapping unchanged for further use.
    """
    if not isinstance(data, dict):
        raise ZooManifestError(f"{source}: manifest must be a JSON object")

    unknown = sorted(set(data) - _ALLOWED_MANIFEST_KEYS)
    if unknown:
        raise ZooManifestError(
            f"{source}: unsupported manifest keys {unknown}; "
            f"allowed: {sorted(_ALLOWED_MANIFEST_KEYS)}"
        )

    version = data.get("schema_version")
    if isinstance(version, bool) or version != ZOO_SCHEMA_VERSION:
        raise ZooManifestError(
            f"{source}: schema_version must be {ZOO_SCHEMA_VERSION}, got {version!r}"
        )

    model_id = data.get("model_id")
    if not isinstance(model_id, str) or not _MODEL_ID_RE.match(model_id):
        raise ZooManifestError(
            f"{source}: model_id must match {_MODEL_ID_RE.pattern!r}, got {model_id!r}"
        )

    task = data.get("task")
    if not isinstance(task, str) or not _TASK_RE.match(task):
        raise ZooManifestError(f"{source}: task must match {_TASK_RE.pattern!r}, got {task!r}")

    loader = data.get("loader")
    if not isinstance(loader, str) or not _LOADER_RE.match(loader):
        raise ZooManifestError(f"{source}: loader must be 'module:attr', got {loader!r}")

    artifact = _require_plain_filename(data.get("artifact"), f"{source}: artifact")

    sha = data.get("artifact_sha256")
    if not isinstance(sha, str) or not _SHA256_RE.match(sha):
        raise ZooManifestError(
            f"{source}: artifact_sha256 must be a 64-char hex digest, got {sha!r}"
        )

    size = data.get("artifact_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ZooManifestError(f"{source}: artifact_bytes must be a non-negative int")

    companion = data.get("artifact_companion")
    if companion is not None:
        companion = _require_plain_filename(companion, f"{source}: artifact_companion")

    metrics = data.get("metrics")
    if metrics is None:
        metrics = {}
    if not isinstance(metrics, dict) or not all(isinstance(k, str) for k in metrics):
        raise ZooManifestError(f"{source}: metrics must be a mapping with str keys")
    for key, value in metrics.items():
        _validate_scalar_metric(key, value)

    dataset = data.get("dataset")
    if dataset is not None:
        if not isinstance(dataset, dict) or not all(isinstance(k, str) for k in dataset):
            raise ZooManifestError(
                f"{source}: dataset must be a mapping with str keys (recommended: path, split)"
            )

    code_sha = data.get("code_sha", "unknown")
    if not isinstance(code_sha, str) or not code_sha.strip():
        raise ZooManifestError(f"{source}: code_sha must be a non-empty string")

    created_at = data.get("created_at")
    if not isinstance(created_at, str) or not _ISO_TS_RE.match(created_at):
        raise ZooManifestError(
            f"{source}: created_at must be an ISO-8601 timestamp with timezone, got {created_at!r}"
        )

    notes = data.get("notes", "")
    if not isinstance(notes, str):
        raise ZooManifestError(f"{source}: notes must be a string")

    return {
        "schema_version": version,
        "model_id": model_id,
        "task": task,
        "loader": loader,
        "artifact": artifact,
        "artifact_sha256": sha,
        "artifact_bytes": size,
        "artifact_companion": companion,
        "metrics": metrics,
        "dataset": dataset,
        "code_sha": code_sha,
        "created_at": created_at,
        "notes": notes,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture_code_sha() -> str:
    """Best-effort git SHA of the repository the zoo is running from.

    Returns ``"unknown"`` outside a git checkout (e.g. an installed wheel) and
    appends ``"+dirty"`` when the working tree has local modifications.
    """
    repo_root = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(  # noqa: S603 - fixed argv, repo-local
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        if not sha:
            return "unknown"
        status = subprocess.run(  # noqa: S603
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        return sha[:12] + ("+dirty" if status.strip() else "")
    except Exception:
        return "unknown"


def _resolve_loader(loader: str) -> Callable[..., Any]:
    """Import and return the loader function named by a ``module:attr`` string.

    This executes the module's import-time code — only resolve loader strings
    from trusted manifests.
    """
    module_name, _, attr = loader.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(f"cannot import loader module {module_name!r}: {exc}") from exc
    try:
        func = getattr(module, attr)
    except AttributeError as exc:
        raise ImportError(f"module {module_name!r} has no attribute {attr!r}") from exc
    if not callable(func):
        raise TypeError(f"loader {loader!r} resolved to a non-callable")
    return func


def _companion_of(path: Path) -> Path:
    """Companion sidecar path following the repo-wide saver convention.

    Every ``save_*`` helper in :mod:`tensorlbm.ai` writes its JSON metadata as
    ``<weights><suffix>.json`` next to the weight file (``model.pt`` ->
    ``model.pt.json``; the transformer's ``.npz`` -> ``.npz.json``).
    """
    return path.with_suffix(path.suffix + ".json")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ModelZoo:
    """A manifest-driven zoo rooted at a filesystem directory.

    Unlike :class:`tensorlbm.ml.model_registry.ModelAssetRegistry` there is no
    index database: the set of entries *is* the set of ``<model_id>/model.json``
    manifests under the root, which keeps a zoo self-describing and shareable.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = resolve_zoo_root(root)

    @classmethod
    def open(cls, root: str | Path | None = None) -> "ModelZoo":
        """Open (without creating) the zoo at ``root`` (default-root if None)."""
        return cls(root)

    # -- layout ---------------------------------------------------------------

    def entry_dir(self, model_id: str) -> Path:
        """Directory that would hold the entry for ``model_id``."""
        if not isinstance(model_id, str) or not _MODEL_ID_RE.match(model_id):
            raise ZooError(f"invalid model_id {model_id!r}")
        return self.root / model_id

    def _manifest_path(self, model_id: str) -> Path:
        return self.entry_dir(model_id) / MANIFEST_FILENAME

    def _read_manifest(self, model_id: str) -> dict[str, Any]:
        path = self._manifest_path(model_id)
        if not path.is_file():
            raise KeyError(f"no zoo entry {model_id!r} (missing {path})")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ZooManifestError(f"{path}: unreadable manifest: {exc}") from exc
        return _validate_manifest(data, source=str(path))

    # -- registration ---------------------------------------------------------

    def register(
        self,
        path: str | Path,
        model_id: str,
        task: str,
        loader: str,
        *,
        metrics: dict[str, Any] | None = None,
        dataset: dict[str, Any] | None = None,
        notes: str = "",
        code_sha: str | None = None,
        move: bool = False,
        overwrite: bool = False,
    ) -> ModelInfo:
        """Register a trained artifact into the zoo and write its manifest.

        Args:
            path: Existing weight file saved by one of the ``save_*`` helpers
                (its ``.json`` sidecar, when present, travels with it).
            model_id: Unique entry id, lowercase kebab/snake (it becomes the
                entry directory name).
            task: Task label, e.g. one of :data:`KNOWN_TASKS`.
            loader: ``"module:attr"`` import string of the ``load_*`` function
                that restores the artifact, e.g. an entry of
                :data:`SUGGESTED_LOADERS`.  It is import-checked here so a
                typo fails at registration, not at load time.
            metrics: Standard evaluation numbers (scalar values only), e.g.
                ``{"test_mape": 3.1}``.
            dataset: Dataset lineage — recommended keys ``path`` and ``split``
                (plus anything else worth recording).
            notes: Free-form context.
            code_sha: Code revision that produced the artifact; auto-captured
                from git when omitted.
            move: Move the artifact into the zoo instead of copying it.
            overwrite: Replace an existing entry with the same ``model_id``;
                registration is fail-closed (raises) otherwise.

        Returns:
            The :class:`ModelInfo` of the new entry.

        Raises:
            FileNotFoundError: ``path`` does not exist.
            ZooError: Duplicate ``model_id`` (unless ``overwrite``), invalid
                ids/labels/loader string, or re-registering from inside the
                entry's own directory.
            ImportError: The loader string cannot be resolved.
        """
        if not isinstance(model_id, str) or not _MODEL_ID_RE.match(model_id):
            raise ZooError(
                f"model_id must match {_MODEL_ID_RE.pattern!r} (lowercase "
                f"kebab/snake), got {model_id!r}"
            )
        if not isinstance(task, str) or not _TASK_RE.match(task):
            raise ZooError(f"task must match {_TASK_RE.pattern!r}, got {task!r}")
        if not isinstance(loader, str) or not _LOADER_RE.match(loader):
            raise ZooError(f"loader must be 'module:attr', got {loader!r}")
        if not isinstance(move, bool) or not isinstance(overwrite, bool):
            raise TypeError("move and overwrite must be bools")
        if notes is not None and not isinstance(notes, str):
            raise ZooError("notes must be a string or None")

        # Validate the caller-supplied metadata *before* touching the zoo, so
        # an invalid manifest never leaves a half-registered entry behind.
        # The JSON round-trip additionally rejects values that cannot be
        # serialised (NaN/inf, non-JSON objects such as ``Path``).
        draft = {
            "schema_version": ZOO_SCHEMA_VERSION,
            "model_id": model_id,
            "task": task,
            "loader": loader,
            "artifact": "placeholder",
            "artifact_sha256": "0" * 64,
            "artifact_bytes": 0,
            "created_at": _now_iso(),
            "metrics": dict(metrics or {}),
            "dataset": dict(dataset) if dataset is not None else None,
            "notes": notes or "",
        }
        _validate_manifest(draft, source=f"register({model_id!r})")
        try:
            json.dumps(draft, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ZooManifestError(
                f"register({model_id!r}): metadata is not JSON-serialisable: {exc}"
            ) from exc

        # Fail early on an unresolvable loader (typo, missing dependency).
        _resolve_loader(loader)

        src = Path(path)
        if not src.is_file():
            raise FileNotFoundError(f"artifact file not found: {src}")
        src = src.resolve()

        entry = self.entry_dir(model_id)
        if src.parent == entry:
            raise ZooError(
                f"cannot register {src}: it already lives inside the entry "
                f"directory {entry} (re-register from a location outside the zoo)"
            )

        if entry.exists() and overwrite:
            shutil.rmtree(entry)
        elif self._manifest_path(model_id).is_file():
            raise ZooError(f"model id {model_id!r} already exists in {self.root}")

        entry.mkdir(parents=True, exist_ok=True)
        dest = entry / src.name
        if move:
            shutil.move(str(src), str(dest))
        else:
            shutil.copy2(src, dest)
        companion_name: str | None = None
        companion_src = _companion_of(src)
        if companion_src.is_file():
            companion_dest = _companion_of(dest)
            if move:
                shutil.move(str(companion_src), str(companion_dest))
            else:
                shutil.copy2(companion_src, companion_dest)
            companion_name = companion_dest.name

        manifest = _validate_manifest(
            {
                "schema_version": ZOO_SCHEMA_VERSION,
                "model_id": model_id,
                "task": task,
                "loader": loader,
                "artifact": dest.name,
                "artifact_sha256": _sha256_file(dest),
                "artifact_bytes": dest.stat().st_size,
                "artifact_companion": companion_name,
                "metrics": dict(metrics or {}),
                "dataset": dict(dataset) if dataset is not None else None,
                "code_sha": code_sha if code_sha is not None else _capture_code_sha(),
                "created_at": _now_iso(),
                "notes": str(notes or ""),
            },
            source=f"register({model_id!r})",
        )

        self._write_manifest(entry, manifest)
        return self._to_info(manifest, entry)

    def _write_manifest(self, entry: Path, manifest: dict[str, Any]) -> None:
        # Atomic-ish write: readers see either the old or the new manifest.
        tmp = entry / f"{MANIFEST_FILENAME}.tmp"
        payload = json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)
        tmp.write_text(payload + "\n", encoding="utf-8")
        os.replace(tmp, entry / MANIFEST_FILENAME)

    @staticmethod
    def _to_info(manifest: dict[str, Any], entry: Path) -> ModelInfo:
        return ModelInfo(
            model_id=manifest["model_id"],
            task=manifest["task"],
            loader=manifest["loader"],
            artifact=manifest["artifact"],
            artifact_sha256=manifest["artifact_sha256"],
            artifact_bytes=manifest["artifact_bytes"],
            entry_dir=str(entry.resolve()),
            artifact_companion=manifest["artifact_companion"],
            metrics=dict(manifest["metrics"]),
            dataset=manifest["dataset"],
            code_sha=manifest["code_sha"],
            created_at=manifest["created_at"],
            notes=manifest["notes"],
        )

    # -- queries ----------------------------------------------------------------

    def info(self, model_id: str) -> ModelInfo:
        """Return the manifest of a registered entry.

        Raises:
            KeyError: No entry with this id.
            ZooManifestError: The entry's manifest violates the schema.
        """
        return self._to_info(self._read_manifest(model_id), self.entry_dir(model_id))

    def list_models(self, task: str | None = None) -> list[ModelInfo]:
        """List zoo entries (oldest first), optionally filtered by task.

        Directories without a ``model.json`` are ignored (stray files, partial
        registrations).  A directory whose manifest is present but invalid
        raises :class:`ZooManifestError` — the zoo is curated, and a corrupt
        manifest should be fixed (see :meth:`validate`) rather than hidden.
        """
        if not self.root.is_dir():
            return []
        entries: list[ModelInfo] = []
        for child in sorted(self.root.iterdir()):
            if not child.is_dir() or not (child / MANIFEST_FILENAME).is_file():
                continue
            manifest = self._read_manifest(child.name)
            if manifest["model_id"] != child.name:
                raise ZooManifestError(
                    f"{child / MANIFEST_FILENAME}: entry directory is "
                    f"{child.name!r} but the manifest declares model_id "
                    f"{manifest['model_id']!r}"
                )
            if task is not None and manifest["task"] != task:
                continue
            entries.append(self._to_info(manifest, child))
        entries.sort(key=lambda m: (m.created_at, m.model_id))
        return entries

    # -- loading ----------------------------------------------------------------

    def load(self, model_id: str) -> Any:
        """Load a registered artifact through its manifest-declared loader.

        .. warning::
           This imports and calls the function named by the manifest's
           ``loader`` field — i.e. it executes code referenced by the manifest.
           Only load zoos whose entries come from trusted sources; check
           provenance with :meth:`info` and integrity with :meth:`validate`
           first when in doubt.

        Returns whatever the loader returns (e.g. a bare ``nn.Module`` for
        eddy-viscosity/FNO entries, a ``(model, norm[, pnorm])`` tuple for
        drag surrogates — ``pnorm`` carries the B1-7 conditioning stats).

        Raises:
            KeyError: No entry with this id.
            ZooManifestError: The manifest violates the schema.
            FileNotFoundError: The artifact file is missing on disk.
            ImportError: The loader cannot be imported/resolved.
        """
        manifest = self._read_manifest(model_id)
        artifact = self.entry_dir(model_id) / manifest["artifact"]
        if not artifact.is_file():
            raise FileNotFoundError(f"artifact for {model_id!r} is missing on disk: {artifact}")
        return _resolve_loader(manifest["loader"])(str(artifact))

    # -- validation ---------------------------------------------------------------

    def validate(self, model_id: str) -> ZooValidation:
        """Check an entry end-to-end without raising on entry problems.

        Verifies, in order: manifest schema, artifact (+ declared companion)
        presence, artifact integrity (SHA-256 + size recorded at
        registration), loader importability, and that the loader actually
        restores the model.  The outcome is reported per check; use
        :attr:`ZooValidation.ok` for a single verdict.

        Raises:
            KeyError: No entry directory with this id at all.
        """
        checks: dict[str, bool] = {}
        errors: list[str] = []

        manifest_path = self._manifest_path(model_id)
        if not manifest_path.is_file():
            raise KeyError(f"no zoo entry {model_id!r} (missing {manifest_path})")

        try:
            manifest = self._read_manifest(model_id)
            checks["manifest_schema"] = True
        except ZooManifestError as exc:
            checks["manifest_schema"] = False
            errors.append(str(exc))
            manifest = None

        entry = self.entry_dir(model_id)
        artifact = companion = None
        if manifest is not None:
            artifact = entry / manifest["artifact"]
            companion = (
                entry / manifest["artifact_companion"] if manifest["artifact_companion"] else None
            )
        artifact_ok = (
            artifact is not None
            and artifact.is_file()
            and (companion is None or companion.is_file())
        )
        checks["artifact_present"] = artifact_ok
        if not artifact_ok:
            errors.append(f"artifact or its declared companion missing under {entry}")

        if artifact_ok and manifest is not None:
            actual_sha = _sha256_file(artifact)
            actual_size = artifact.stat().st_size
            integrity = (
                actual_sha == manifest["artifact_sha256"]
                and actual_size == manifest["artifact_bytes"]
            )
            checks["integrity"] = integrity
            if not integrity:
                errors.append(
                    f"artifact changed since registration: recorded "
                    f"sha256={manifest['artifact_sha256'][:12]}…/{manifest['artifact_bytes']}B, "
                    f"actual {actual_sha[:12]}…/{actual_size}B"
                )

        loader = manifest["loader"] if manifest is not None else None
        loader_func = None
        if loader is not None:
            try:
                loader_func = _resolve_loader(loader)
                checks["loader_importable"] = True
            except (ImportError, TypeError) as exc:
                checks["loader_importable"] = False
                errors.append(str(exc))

        checks["model_loads"] = False
        if loader_func is not None and artifact_ok:
            try:
                loader_func(str(artifact))
                checks["model_loads"] = True
            except Exception as exc:  # noqa: BLE001 - reporting, not handling
                errors.append(f"loader {loader!r} failed on {artifact}: {exc}")

        return ZooValidation(model_id=model_id, checks=checks, errors=tuple(errors))


# ---------------------------------------------------------------------------
# Module-level convenience API (default zoo root)
# ---------------------------------------------------------------------------


def register(
    path: str | Path,
    model_id: str,
    task: str,
    loader: str,
    *,
    metrics: dict[str, Any] | None = None,
    dataset: dict[str, Any] | None = None,
    notes: str = "",
    code_sha: str | None = None,
    move: bool = False,
    overwrite: bool = False,
    root: str | Path | None = None,
) -> ModelInfo:
    """Register an artifact into the zoo at ``root`` (default-root if None).

    See :meth:`ModelZoo.register`.
    """
    return ModelZoo(root).register(
        path,
        model_id,
        task,
        loader,
        metrics=metrics,
        dataset=dataset,
        notes=notes,
        code_sha=code_sha,
        move=move,
        overwrite=overwrite,
    )


def load(model_id: str, *, root: str | Path | None = None) -> Any:
    """Load an artifact from the zoo at ``root`` (default-root if None).

    See :meth:`ModelZoo.load` — including its security warning about
    executing manifest-registered loaders.
    """
    return ModelZoo(root).load(model_id)


def list_models(
    task: str | None = None,
    *,
    root: str | Path | None = None,
) -> list[ModelInfo]:
    """List zoo entries at ``root`` (default-root if None)."""
    return ModelZoo(root).list_models(task)


def info(model_id: str, *, root: str | Path | None = None) -> ModelInfo:
    """Return the manifest of an entry at ``root`` (default-root if None)."""
    return ModelZoo(root).info(model_id)


def validate(model_id: str, *, root: str | Path | None = None) -> ZooValidation:
    """Validate an entry at ``root`` (default-root if None)."""
    return ModelZoo(root).validate(model_id)
