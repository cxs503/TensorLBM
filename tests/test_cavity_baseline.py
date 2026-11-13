"""Cavity regression test in the ``pylbm`` baseline-reference style.

The run is executed through the *production* entry point
(``run_lid_driven_cavity``), which writes ``reference_block.npz`` next to its
usual outputs.  The block is then compared against a checked-in baseline under
the upstream tolerance (``atol=1e-7, rtol=1e-14``), exactly like
``pylbm/tests/test_demo_2d.py`` compares its demo runs against
``tests/reference/*.h5``.

Generate mode (upstream ``--h5diff-generate-path``): set
``TLBM_BASELINE_GENERATE=<dir>`` and the reference is written instead of
compared, then the test is skipped.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tensorlbm import LidDrivenCavityConfig, run_lid_driven_cavity
from tensorlbm.baseline_io import read_block
from tensorlbm.verification.baseline_reference import BaselineStore

BASELINE_NAME = "cavity_re100_nx33_u0.1_s300.npz"
CASE = {"nx": 33, "re": 100.0, "u_lid": 0.1, "n_steps": 300}


def _run(tmp_path: Path) -> Path:
    config = LidDrivenCavityConfig(
        **CASE, output_root=tmp_path / "runs", run_name="baseline_case", overwrite=True
    )
    run_dir = run_lid_driven_cavity(config)
    return run_dir / "reference_block.npz"


def test_cavity_reference_block_contract(tmp_path: Path) -> None:
    """The production run must emit a complete, self-describing block."""
    block = _run(tmp_path)
    fields, meta = read_block(block)
    assert set(fields) >= {
        "ux_centerline_normalised",
        "uy_centerline_normalised",
        "rho_field",
        "mass",
        "mass_drift_relative",
    }
    assert fields["ux_centerline_normalised"].shape == (CASE["nx"],)
    assert fields["rho_field"].shape == (CASE["nx"], CASE["nx"])
    assert meta["config"]["re"] == CASE["re"] and meta["config"]["n_steps"] == CASE["n_steps"]
    assert meta["sha256"]
    # Mass drift over 300 steps stays inside 1 % (the defect class m18235 hit:
    # a broken lid BC diverges or leaks mass by several percent in 50 steps).
    assert abs(float(fields["mass_drift_relative"])) < 1e-2


def test_cavity_matches_checked_in_baseline(tmp_path: Path) -> None:
    block = _run(tmp_path)
    store = BaselineStore()
    if store.generating:
        store.write(BASELINE_NAME, read_block(block)[0], read_block(block)[1])
        pytest.skip("Skipping test, since generating data")
    result = store.compare(BASELINE_NAME, block)
    assert result.ok, result.message
