"""Configuration file loading and environment-variable override utilities.

Supports loading :mod:`dataclasses`-based Config objects from YAML or TOML
files, with optional environment-variable overrides.

YAML requires ``pyyaml`` (``pip install pyyaml``).
TOML is supported natively in Python 3.11+ via :mod:`tomllib`; for older
Python use ``tomli`` (``pip install tomli``).

Also provides :func:`save_config_json` and :func:`load_config_json` for a
built-in, dependency-free JSON round-trip that the runner Config classes
expose via their ``save`` / ``load`` class methods.
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any, TypeVar, cast

_T = TypeVar("_T")


def _load_raw(path: Path) -> dict[str, Any]:
    """Load a YAML or TOML file and return its contents as a plain dict."""
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "pyyaml is required to load YAML configs: pip install pyyaml"
            ) from exc
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return dict(data) if data else {}

    if suffix == ".toml":
        import tomllib

        with path.open("rb") as fb:
            return tomllib.load(fb)

    raise ValueError(
        f"Unsupported config file format: {suffix!r}. Use .yaml, .yml, or .toml"
    )


def _apply_env_overrides(data: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Override *data* keys from environment variables prefixed with *prefix*.

    For example, if *prefix* = ``"TENSORLBM"`` then the environment variable
    ``TENSORLBM_NX=128`` will set ``data["nx"] = "128"`` (as a string; type
    coercion is handled by the dataclass constructor).
    """
    result = dict(data)
    prefix_upper = prefix.upper() + "_"
    for key, val in os.environ.items():
        if key.upper().startswith(prefix_upper):
            field_name = key[len(prefix_upper) :].lower()
            result[field_name] = val
    return result


def load_config(
    config_class: type[_T],
    path: str | Path,
    env_prefix: str = "TENSORLBM",
) -> _T:
    """Load a dataclass-based Config from a YAML or TOML file.

    Field values present in the file override dataclass defaults.
    Environment variables of the form ``{ENV_PREFIX}_{FIELD_NAME}`` (case-
    insensitive) are applied last, taking highest precedence.

    Args:
        config_class: A :func:`dataclasses.dataclass` type to instantiate.
        path: Path to a ``.yaml``, ``.yml``, or ``.toml`` configuration file.
        env_prefix: Prefix for environment-variable overrides
            (default ``"TENSORLBM"``).

    Returns:
        An instance of *config_class* populated from the file and env vars.

    Raises:
        ValueError: If the file format is not supported.
        ImportError: If the required YAML/TOML parser is not installed.
    """
    path = Path(path)
    raw = _load_raw(path)
    raw = _apply_env_overrides(raw, env_prefix)

    if dataclasses.is_dataclass(config_class):
        fields = {f.name: f for f in dataclasses.fields(config_class)}
        coerced: dict[str, Any] = {}
        for k, v in raw.items():
            if k in fields and isinstance(v, str):
                ftype = fields[k].type
                try:
                    if ftype in (int, "int"):
                        v = int(v)
                    elif ftype in (float, "float"):
                        v = float(v)
                    elif ftype in (bool, "bool"):
                        v = v.lower() not in {"0", "false", "no", "off"}
                except (ValueError, AttributeError):
                    pass
            coerced[k] = v
        return config_class(**coerced)

    return config_class(**raw)


def save_config_json(config: object, path: str | Path) -> Path:
    """Serialise a dataclass-based Config to a JSON file.

    Path-type fields are serialised as strings; *None* values are written as
    JSON ``null``.  The resulting file can be reloaded with
    :func:`load_config_json`.

    Args:
        config: A dataclass instance to serialise.
        path: Output path (should end with ``.json``).

    Returns:
        Resolved output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = dataclasses.asdict(cast("Any", config))
    # Convert Path objects to strings for JSON serialisation
    serialisable: dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, Path):
            serialisable[k] = str(v)
        else:
            serialisable[k] = v
    path.write_text(json.dumps(serialisable, indent=2) + "\n", encoding="utf-8")
    return path


def load_config_json(config_class: type[_T], path: str | Path) -> _T:
    """Load a dataclass-based Config from a JSON file written by
    :func:`save_config_json`.

    Path-type fields are reconstructed from their string representations.

    Args:
        config_class: A :func:`dataclasses.dataclass` type to instantiate.
        path: Path to a JSON config file.

    Returns:
        An instance of *config_class* populated from the file.
    """
    path = Path(path)
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

    if dataclasses.is_dataclass(config_class):
        fields = {f.name: f for f in dataclasses.fields(config_class)}
        coerced: dict[str, Any] = {}
        for k, v in raw.items():
            if k not in fields:
                continue
            field = fields[k]
            # Re-hydrate Path fields
            ftype = field.type
            if ftype in (Path, "Path") or (isinstance(ftype, str) and "Path" in ftype):
                coerced[k] = Path(v) if v is not None else None
            else:
                coerced[k] = v
        return config_class(**coerced)

    return config_class(**raw)


__all__ = ["load_config", "save_config_json", "load_config_json"]
