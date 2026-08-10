"""Measurement-only grid/time mapping checks for the Gallium PF Stefan case."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "scripts"))
from gallium_pf_grid_diagnostic import grid_time_mapping  # noqa: E402


def test_grid_time_mapping_preserves_fo_and_ra_for_twofold_refinement():
    coarse = grid_time_mapping(nx=40, ny=56, target_fo=(0.205, 0.410))
    fine = grid_time_mapping(nx=80, ny=112, target_fo=(0.205, 0.410))

    assert coarse["steps"] == (3280, 6560)
    assert fine["steps"] == (13120, 26240)
    assert fine["steps"][-1] == 4 * coarse["steps"][-1]
    assert abs(coarse["Ra"] - fine["Ra"]) < 1e-9
    assert abs(fine["gy"] - coarse["gy"] / 8.0) < 1e-15
