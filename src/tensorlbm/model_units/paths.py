"""Checkout paths shared by the two model-unit manifest formats."""

from pathlib import Path

MODEL_UNITS_ROOT = Path("benchmarks/model_units")
TOML_UNITS_ROOT = MODEL_UNITS_ROOT / "lbm"
JSON_UNITS_ROOT = MODEL_UNITS_ROOT / "units"


def resolve_model_unit_root(root: str | Path, repo: Path) -> Path:
    """Resolve the three former checkout roots for existing callers."""
    requested = Path(root)
    previous = (repo / "models").resolve()
    resolved = requested.resolve()
    if resolved == previous:
        if (previous / "lbm").is_dir() or (previous / "units").is_dir():
            return requested
        replacement = repo / MODEL_UNITS_ROOT
    elif resolved == previous / "lbm" and not requested.is_dir():
        replacement = repo / TOML_UNITS_ROOT
    elif resolved == previous / "units" and not requested.is_dir():
        replacement = repo / JSON_UNITS_ROOT
    else:
        return requested
    return replacement if replacement.is_dir() else requested
