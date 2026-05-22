from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_cylinder_flow_cli_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "outputs"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")

    cmd = [
        sys.executable,
        str(repo_root / "examples" / "cylinder_flow.py"),
        "--nx",
        "48",
        "--ny",
        "20",
        "--radius",
        "3",
        "--n-steps",
        "8",
        "--output-interval",
        "4",
        "--output-root",
        str(output_root),
        "--run-name",
        "smoke",
    ]
    subprocess.run(cmd, check=True, env=env, cwd=str(repo_root))

    run_dir = output_root / "cylinder_flow" / "smoke"
    metadata_path = run_dir / "run_metadata.json"
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["config"]["n_steps"] == 8
    assert metadata["diagnostics"]
    assert (run_dir / "flow_step_000008.png").exists()
