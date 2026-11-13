"""Per-model manifest: the self-describing unit of build and CI.

Cross-validation target (XVAL-LBM-MODEL-DIR-AS-BUILD-AND-CI-UNIT): CFD-GO/TCLB
treats *a model directory* as the build and CI unit -- ``models/<group>/<model>/
conf.mk`` is the only file that has to exist for a model to be discovered,
compiled (``CLB/<model>``) and tested by CI (``src/models.R`` enumerates
``conf.mk`` files; ``makefile.main`` lists ``models/*`` as prerequisites).

This module is the Python analogue of ``conf.mk``: one TOML file per model
directory, with no central registry to edit.  The schema is intentionally
small and every field is validated eagerly so a malformed unit fails at
discovery time (as ``models.R`` does with ``stop("Wrong value of TEST ...")``)
instead of during a CI job.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_NAME = "model.toml"
SUPPORTED_LATTICES = ("D2Q9", "D3Q19", "D3Q27")
SUPPORTED_DIMS = {2: ("D2Q9",), 3: ("D3Q19", "D3Q27")}

REQUIRED_UNIT_KEYS = ("name", "group", "dimension", "lattice", "kernel", "description")
REQUIRED_CASE_KEYS = ("script", "entry")
REQUIRED_ACCEPTANCE_KEYS = ("kpi", "threshold")


class ManifestError(ValueError):
    """Raised when a ``model.toml`` cannot be turned into a valid unit."""


@dataclass(frozen=True)
class ModelUnit:
    """One model directory: manifest + case script + kernel module binding."""

    name: str
    group: str
    path: Path
    manifest: dict[str, Any] = field(repr=False)
    manifest_sha256: str
    case_sha256: str
    kernel_sha256: str
    kernel_source: str

    # -- derived accessors -------------------------------------------------
    @property
    def unit_block(self) -> dict[str, Any]:
        return dict(self.manifest["unit"])

    @property
    def unit_id(self) -> str:
        """Stable identifier: ``group/name`` (TCLB prints ``<group>/<name>``)."""
        return f"{self.group}/{self.name}"

    @property
    def lattice(self) -> str:
        return str(self.unit_block["lattice"])

    @property
    def dimension(self) -> int:
        return int(self.unit_block["dimension"])

    @property
    def ci(self) -> bool:
        return bool(self.unit_block.get("ci", True))

    @property
    def tier(self) -> str:
        return str(self.unit_block.get("tier", "fast"))

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.manifest.get("params", {}))

    @property
    def case_path(self) -> Path:
        return self.path / str(self.manifest["case"]["script"])

    @property
    def acceptance(self) -> dict[str, Any]:
        return dict(self.manifest["acceptance"])

    @property
    def reference(self) -> str:
        return str(self.unit_block.get("reference", ""))

    def fingerprint(self) -> str:
        """Hash of the three files that define the unit's behaviour."""
        h = hashlib.sha256()
        h.update(self.manifest_sha256.encode())
        h.update(b"\0")
        h.update(self.case_sha256.encode())
        h.update(b"\0")
        h.update(self.kernel_sha256.encode())
        return h.hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit": self.unit_id,
            "name": self.name,
            "group": self.group,
            "path": str(self.path),
            "dimension": self.dimension,
            "lattice": self.lattice,
            "kernel": self.kernel_source,
            "ci": self.ci,
            "tier": self.tier,
            "reference": self.reference,
            "acceptance": self.acceptance,
            "manifest_sha256": self.manifest_sha256,
            "case_sha256": self.case_sha256,
            "kernel_sha256": self.kernel_sha256,
            "fingerprint": self.fingerprint(),
        }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kernel_module_file(module: str) -> Path | None:
    """Locate the source file of a ``tensorlbm.*`` module without importing it."""
    parts = module.split(".")
    if parts[0] != "tensorlbm" or len(parts) < 2:
        return None
    import tensorlbm  # local import: only used to resolve the package root

    root = Path(tensorlbm.__file__).resolve().parent
    rel = Path(*parts[1:])
    for candidate in (root / rel / "__init__.py", root / rel.with_suffix(".py")):
        if candidate.is_file():
            return candidate
    return None


def parse_manifest(path: Path) -> dict[str, Any]:
    """Read and validate one ``model.toml``; raise :class:`ManifestError`."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:  # pragma: no cover - IO path
        raise ManifestError(f"{path}: cannot parse manifest ({exc})") from exc
    if not isinstance(data, dict):
        raise ManifestError(f"{path}: manifest must be a TOML table")

    for section, keys in (
        ("unit", REQUIRED_UNIT_KEYS),
        ("case", REQUIRED_CASE_KEYS),
        ("acceptance", REQUIRED_ACCEPTANCE_KEYS),
    ):
        block = data.get(section)
        if not isinstance(block, dict):
            raise ManifestError(f"{path}: missing [{section}] table")
        missing = [k for k in keys if k not in block]
        if missing:
            raise ManifestError(f"{path}: [{section}] missing keys {missing}")

    unit = data["unit"]
    lattice = str(unit["lattice"])
    if lattice not in SUPPORTED_LATTICES:
        raise ManifestError(f"{path}: unsupported lattice {lattice!r}")
    dim = int(unit["dimension"])
    if lattice not in SUPPORTED_DIMS.get(dim, ()):
        raise ManifestError(f"{path}: lattice {lattice!r} is not {dim}D")

    name = str(unit["name"])
    if path.parent.name != name:
        # Directory name *is* the unit name in TCLB; keep that invariant so the
        # build target and the CI label are the directory, not a second source.
        raise ManifestError(
            f"{path}: unit.name {name!r} must equal the directory name {path.parent.name!r}"
        )

    case = data["case"]
    case_path = path.parent / str(case["script"])
    if not case_path.is_file():
        raise ManifestError(f"{path}: case script {case_path.name!r} not found")

    kernel_file = kernel_module_file(str(unit["kernel"]))
    if kernel_file is None:
        raise ManifestError(f"{path}: kernel module {unit['kernel']!r} not found in tensorlbm")

    acc = data["acceptance"]
    threshold = float(acc["threshold"])
    if not 0.0 < threshold <= 1.0:
        raise ManifestError(f"{path}: acceptance.threshold must be a relative tolerance")
    return data


def load_unit(path: Path) -> ModelUnit:
    """Load one ``model.toml`` into a :class:`ModelUnit`."""
    data = parse_manifest(path)
    unit = data["unit"]
    case_path = path.parent / str(data["case"]["script"])
    kernel_file = kernel_module_file(str(unit["kernel"]))
    assert kernel_file is not None  # parse_manifest guarantees this
    return ModelUnit(
        name=str(unit["name"]),
        group=str(unit["group"]),
        path=path.parent,
        manifest=data,
        manifest_sha256=_sha256_file(path),
        case_sha256=_sha256_file(case_path),
        kernel_sha256=_sha256_file(kernel_file),
        kernel_source=str(unit["kernel"]),
    )
