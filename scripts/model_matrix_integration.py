#!/usr/bin/env python3
"""Compatibility entry point for the model matrix integration check."""

from pathlib import Path
from runpy import run_path

if __name__ == "__main__":
    run_path(str(Path(__file__).resolve().parents[1] / "tools" / "model_matrix_integration.py"), run_name="__main__")
