"""Pipeline E2E smoke test for mission m4630.

The smoke marker lives beside this test (``tests/smoke_marker_m4630.py``)
because the repository layout policy rejects new root-level modules: only the
filenames listed in ``docs/architecture/repository_layout_policy.json`` may sit
at the repository root.  Importing the sibling module still exercises the same
end-to-end property (a marker module is produced and is importable by the test
suite), and pytest's default ``prepend`` import mode puts this directory on
``sys.path`` so the import resolves without a package.
"""

from smoke_marker_m4630 import PIPELINE_SMOKE_M4630


def test_marker_value():
    """Assert the smoke marker constant equals the mission number string."""
    assert PIPELINE_SMOKE_M4630 == "4630"


def test_basic_arithmetic():
    """Sanity check: 1 + 1 == 2."""
    assert 1 + 1 == 2
