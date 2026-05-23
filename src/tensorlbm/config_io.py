"""JSON-based configuration serialisation / deserialisation helpers.

Provides a generic :func:`save_config_json` / :func:`load_config_json` pair
that round-trips any frozen ``@dataclass`` configuration to and from a JSON
file without external dependencies (no PyYAML or TOML required).

All runner config classes (``CylinderFlowConfig``, ``SphereFlowConfig``,
``ShipHullFlowConfig``, ``SphereFlowD3Q27Config``) gain ``.save()`` and
``.load()`` class-methods that delegate to these helpers.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import Protocol

    class DataclassInstance(Protocol):
        __dataclass_fields__: dict[str, Any]


def save_config_json(config: Any, path: Path | str) -> None:
    """Serialise a frozen dataclass *config* to a JSON file at *path*.

    Path objects and any ``None`` values are handled correctly.

    Args:
        config: A frozen ``@dataclass`` instance.
        path: Destination path (created or overwritten).
    """
    path = Path(path)
    raw = asdict(config)
    # Convert Path objects to strings
    raw = _paths_to_str(raw)
    path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_config_json(cls: type, path: Path | str) -> Any:
    """Deserialise a JSON file produced by :func:`save_config_json` into *cls*.

    Args:
        cls: The dataclass class to instantiate.
        path: Source JSON file path.

    Returns:
        A new ``cls`` instance with fields from the file.
    """
    path = Path(path)
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return cls(**raw)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _paths_to_str(obj: Any) -> Any:
    """Recursively convert :class:`~pathlib.Path` objects to strings."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _paths_to_str(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_paths_to_str(v) for v in obj]
    return obj


__all__ = ["save_config_json", "load_config_json"]
