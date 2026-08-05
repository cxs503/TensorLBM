"""Import-boundary regressions for the Torch-only AI package."""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    source_root = str(_REPO_ROOT / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_ai_package_and_transformer_import_without_suboff_modules() -> None:
    result = _run_python(
        "import tensorlbm.ai; "
        "from tensorlbm.ai.transformer import train_flow_transformer_self_supervised; "
        "assert callable(train_flow_transformer_self_supervised)",
    )
    assert result.returncode == 0, result.stderr


def test_public_turbulence_api_remains_importable() -> None:
    from tensorlbm.ai import collide_ai_les_bgk, predict_nu_t_2d

    assert callable(collide_ai_les_bgk)
    assert callable(predict_nu_t_2d)


def test_suboff_availability_reports_missing_optional_modules() -> None:
    ai = importlib.import_module("tensorlbm.ai")

    report = ai.get_suboff_availability()

    assert report["available"] is False
    assert report["status"] == "NOT_AVAILABLE"
    assert "tensorlbm.ai.suboff_utils" in report["reason"]
    assert "build_suboff_model" not in ai.__all__


def test_suboff_loader_reraises_nonoptional_module_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ai = importlib.import_module("tensorlbm.ai")

    def raise_unrelated_module_not_found(_name: str) -> object:
        raise ModuleNotFoundError("No module named 'numpy'", name="numpy")

    monkeypatch.setattr(ai.importlib, "import_module", raise_unrelated_module_not_found)

    with pytest.raises(ModuleNotFoundError, match="numpy"):
        ai._load_optional_suboff_api()


def test_ai_package_does_not_declare_paddle_or_mindspore_support() -> None:
    source = (_REPO_ROOT / "src" / "tensorlbm" / "ai" / "__init__.py").read_text().lower()

    assert "paddle" not in source
    assert "mindspore" not in source
