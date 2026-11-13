"""Entry-point tests for the CASE-BACKWARD-STEP-LBM pool case (m17819).

The library kernels are pinned by ``tests/test_case_pool_backward_step_armaly.py``;
this module pins the *entry point* ``benchmarks/case_pool/backward_step_armaly/run_case.py``
which was previously untested: the preset table (``GRID``) and the end-to-end
``--smoke`` path (config build -> ``run_armaly_case`` -> artefact schema).

The smoke run is a 16x174 lattice for 400 steps (~10 s on CPU): it is a pipeline
sanity check, not a physics verdict -- no reference tolerance is applied here.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "case_pool" / "backward_step_armaly"))

import run_case as RC  # noqa: E402

from tensorlbm.bfs_armaly import ARMALY_EXPANSION_RATIO, armaly_geometry  # noqa: E402


def test_grid_table_is_consistent():
    tags = [t for t, _, _ in RC.GRID]
    assert len(tags) == len(set(tags)) == 9
    # Armaly-caliber grids carry the exact experimental expansion ratio 101/52
    for scale in (1, 2):
        hi, s = armaly_geometry(scale)
        assert abs((hi + s) / hi - ARMALY_EXPANSION_RATIO) < 1e-12
    # reduced ladder: 26:25 -> ER = 51/26 = 1.9615 (+0.99 % vs 1.9423)
    assert ((26 + 25) / 26 - ARMALY_EXPANSION_RATIO) / ARMALY_EXPANSION_RATIO < 0.011


def test_build_config_keeps_tau_in_lattice_window():
    """build_config's stated contract: tau >= 0.55 on every grid/Re pair.

    Known documented exception: ``h1_re400`` (tau = 0.538) sits just below the
    BGK/TRT validity floor; the m16958 converged-ladder doc records it as a
    failing ``lattice`` condition -- the verdict gate, not the preset table,
    is where that is judged.
    """
    for tag, scale, re in RC.GRID:
        cfg = RC.build_config(tag, scale, re)
        if tag == "h1_re400":
            assert cfg.tau < 0.55  # documented exception, judged by verdict.py
            continue
        assert cfg.tau >= 0.55, f"{tag}: tau={cfg.tau:.4f} below lattice window"


def test_smoke_preset_runs_end_to_end_and_writes_schema(tmp_path):
    out = tmp_path / "smoke"
    rc = RC.main(["--smoke", "--out-dir", str(out)])
    assert rc == 0
    res = json.loads((out / "result.json").read_text())
    for key in ("case", "config", "derived", "result", "entry", "tag"):
        assert key in res, f"missing top-level key {key!r}"
    assert res["entry"] == "benchmarks/case_pool/backward_step_armaly/run_case.py"
    assert res["tag"] == "smoke"
    r = res["result"]
    assert math.isfinite(r["x1_over_s"])
    assert r["steps_run"] == 400
    assert r["n_averaged_samples"] >= 1
    assert 0.9 < r["rho_min"] and r["rho_max"] < 1.1
    assert (out / "wall_shear.csv").is_file()
    assert (out / "field.npz").is_file()


def test_unknown_preset_is_rejected():
    with pytest.raises(SystemExit):
        RC.main(["--preset", "no_such_preset"])
