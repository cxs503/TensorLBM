"""Core imports must not require optional AI/SUBOFF modules."""

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _run(code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


_BLOCK_OPTIONAL_SUBOFF_UTILS = """
import importlib.abc
import sys

class BlockSuboffUtils(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'tensorlbm.ai.suboff_utils':
            raise ModuleNotFoundError("No module named 'tensorlbm.ai.suboff_utils'", name=fullname)
        return None

sys.meta_path.insert(0, BlockSuboffUtils())
"""


def test_core_d3q27_import_does_not_require_optional_suboff_utils():
    """Core LBM imports even when the optional SUBOFF helper is unavailable."""
    result = _run(_BLOCK_OPTIONAL_SUBOFF_UTILS + "\nimport tensorlbm.d3q27")

    assert result.returncode == 0, result.stderr


def test_explicit_ai_import_reports_missing_optional_dependency():
    """AI consumers get a direct error identifying the unavailable optional helper."""
    result = _run(_BLOCK_OPTIONAL_SUBOFF_UTILS + "\nimport tensorlbm.ai")

    assert result.returncode != 0
    assert "optional SUBOFF dependency" in result.stderr
    assert "suboff_utils" in result.stderr
