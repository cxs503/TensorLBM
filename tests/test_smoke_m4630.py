"""Pipeline E2E smoke test for mission m4630.

The smoke marker lives beside this test (``tests/smoke_marker_m4630.py``)
because the repository layout policy rejects new root-level modules: only the
filenames listed in ``docs/architecture/repository_layout_policy.json`` may sit
at the repository root.  Importing the sibling module still exercises the same
end-to-end property (a marker module is produced and is importable by the test
suite), and pytest's default ``prepend`` import mode puts this directory on
``sys.path`` so the import resolves without a package.

``test_marker_module_is_not_repo_root_clutter`` pins the placement itself: the
layout gate that rejected the original PR #668 commit is a separate CI job, so
without this assertion a later re-move of the marker back to the repository
root would only be caught by ``repository-layout``, not by the test suite.
"""

from pathlib import Path

import smoke_marker_m4630
from smoke_marker_m4630 import PIPELINE_SMOKE_M4630

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_marker_value():
    """Assert the smoke marker constant equals the mission number string."""
    assert PIPELINE_SMOKE_M4630 == "4630"


def test_basic_arithmetic():
    """Sanity check: 1 + 1 == 2."""
    assert 1 + 1 == 2


def test_marker_module_is_not_repo_root_clutter():
    """The marker module must live in ``tests/``, never at the repository root."""
    marker_path = Path(smoke_marker_m4630.__file__).resolve()
    assert marker_path.parent == _REPO_ROOT / "tests"
    assert not (_REPO_ROOT / "smoke_marker_m4630.py").exists(), "marker moved back to the repo root"
