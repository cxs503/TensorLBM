"""Gates for the acceptance sweep's bookkeeping (not for the physics itself).

Two things are pinned here, both of which caused the ``< 15 %`` acceptance of
``XVAL-LBM-AA-PATTERN-FUSED-STREAM-COLLIDE`` to stay unproduced for days:

1. **Thread pin** -- the acceptance driver must pin the torch intra-op pool.
   With the host default (one thread per core, 128 here) the launch overhead of
   the ~60 tiny kernels per step dominates by 4.4x, and a cavity arm that needs
   minutes stayed "running" for > 6 h in the scheduler queue.
2. **Honest verdict bookkeeping** -- a grid that did not run must never be
   counted as a pass, a non-monotone error sequence must not be fitted for an
   order, and the pass statement must be restricted to the grids that ran.

The physics gates (exact Couette profile, fused == materialised, mass
conservation, metric floor) live in ``tests/test_aa_fused_cavity_acceptance.py``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "benchmarks", "xval"))

import aa_fused_cavity_acceptance as ACC  # noqa: E402
import aa_fused_cavity_verdict as V  # noqa: E402


def test_acceptance_driver_pins_intraop_threads():
    """Launch-overhead-bound kernel: the sweep must not use the host's full pool."""
    n = torch.get_num_threads()
    assert n <= 4, f"intra-op pool not pinned (n={n}); cavity arms will crawl"
    assert ACC._THREADS == n


def _cavity_task(L, u_full, v_full, u_int=None, v_int=None, conv=None):
    return {"task": "cavity", "L": L, "tau": 0.5 + 3 * 0.1 * L / 100, "steps_run": 10,
            "converged_at": conv, "L2_u_full": u_full, "L2_v_full": v_full,
            "L2_u_interior": u_full if u_int is None else u_int,
            "L2_v_interior": v_full if v_int is None else v_int,
            "metric_floor": {"floor_u": 0.01}, "fluid_mass_rel_drift": 1e-12,
            "elapsed_s": 1.0, "artifact": "synthetic"}


def test_missing_grids_are_not_evidence():
    tasks = {2: _cavity_task(32, 0.026, 0.048)}
    v = V.build(tasks)
    assert v["grids_run"] == [32]
    assert v["grids_not_run"] == [64, 128, 192]
    # the pass statement is restricted to what ran; the absent grids are listed
    assert v["gate_pass_all_run_grids"] is True
    assert v["observed_order"]["u_interior"]["p"] is None
    assert "need >= 3" in v["observed_order"]["u_interior"]["reason"]


def test_no_cavity_artifact_means_no_verdict():
    v = V.build({0: {"task": "couette", "rel_err_max": 1e-6,
                     "fluid_mass_drift_rel": 1e-13, "gate_1pct": True}})
    assert v["cavity"] == []
    assert v["gate_pass_all_run_grids"] is None
    assert v["couette_gate"]["status"] == "ran"


def test_gate_fails_when_a_run_grid_misses_it():
    tasks = {2: _cavity_task(32, 0.026, 0.048),
             3: _cavity_task(64, 0.02, 0.21)}
    assert tasks[3]["L"] == 64
    v = V.build(tasks)
    assert v["gate_pass_all_run_grids"] is False
    assert v["worst_cavity_error"] == pytest.approx(0.21)


def test_observed_order_recovers_second_order_and_refuses_nonmonotone():
    second = {2 + i: _cavity_task(L, 0.4 / L ** 2, 0.3 / L ** 2)
              for i, L in enumerate((32, 64, 128, 192))}
    v = V.build(second)
    assert v["observed_order"]["u_interior"]["p"] == pytest.approx(2.0, abs=1e-9)

    bad = {2 + i: _cavity_task(L, e, e)
           for i, (L, e) in enumerate(((32, 0.05), (64, 0.02), (128, 0.03)))}
    v = V.build(bad)
    assert v["observed_order"]["u_interior"]["p"] is None
    assert "monoton" in v["observed_order"]["u_interior"]["reason"]


def test_markdown_table_pulls_numbers_from_artifacts():
    v = V.build({2: _cavity_task(32, 0.026, 0.048, conv=9000)})
    md = V.to_markdown(v)
    assert "| 32 |" in md and "0.0260" in md and "9000" in md


def test_real_artifacts_stay_consistent_with_the_verdict():
    """If acceptance artifacts exist in the tree, the verdict must agree with them."""
    tasks = V.load_tasks()
    if not tasks:
        pytest.skip("no acceptance artifacts in this checkout")
    v = V.build(tasks)
    for row in v["cavity"]:
        t = [a for a in tasks.values() if a.get("task") == "cavity" and a["L"] == row["L"]]
        assert t, row
        assert row["L2_u_full"] == pytest.approx(t[0]["L2_u_full"])
        assert np.isfinite(row["L2_u_full"])