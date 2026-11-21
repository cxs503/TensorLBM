"""Compatibility entry point for the published Ludwig stress experiment."""
from pathlib import Path
import runpy


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    runner = directory.parents[2] / "xval/ludwig_fe_stress/run_xval.py"
    runpy.run_path(str(runner))["main"](default_output_dir=directory)
