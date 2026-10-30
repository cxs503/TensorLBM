"""bstep_3d reattachment-measurement guards (mission m469).

The pending case ``benchmarks/pending/bstep_3d`` published ``xr_h = 0.0`` for two
of its three records (Re=200: aborted after 500 steps; Re=400: ``xr_h = 0.0``
while the same file reports ``max_backflow_over_umax = -0.175`` at
``backflow_x_rel_h = 4.35``, i.e. a well-developed recirculation).  The root
cause was the zero-crossing estimator in ``run.py``/``run_highre.py``, which

* took the *first* negative->positive crossing of the near-wall velocity row,
* returned ``None`` (stored as ``0.0``) whenever the first downstream node was
  already positive, and
* folded "no crossing found" onto "X_r = 0" for three physically different
  outcomes.

These tests pin the replacement estimator (``reattach.py``) and its wiring into
the benchmark scripts.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_CASE_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "pending" / "bstep_3d"


def _load(module_name: str):
    """Load a benchmark script by path (they are not importable packages)."""
    name = f"bstep3d_{module_name}"
    path = _CASE_DIR / f"{module_name}.py"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses in reattach.py resolve annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rt():
    if str(_CASE_DIR) not in sys.path:
        sys.path.insert(0, str(_CASE_DIR))
    return _load("reattach")


pytestmark = [pytest.mark.unit, pytest.mark.regression]


def test_legacy_estimator_drops_separated_flow(rt):
    """Pre-fix defect: a positive first downstream node zeroes a real bubble.

    Shape of the row: the corner node is slightly positive (unresolved corner /
    lattice noise), then the main recirculation, then reattachment.
    """
    row = np.array([0.002, -0.010, -0.050, -0.020, 0.001, 0.010, 0.020])
    assert rt.legacy_first_crossing(row) is None  # -> recorded as xr_h = 0.0
    res = rt.locate_reattachment(row, rel_thresh=1e-2, u_scale=0.05)
    assert res.status == rt.REATTACHED
    assert res.x is not None and 3.0 < res.x < 5.0


def test_legacy_estimator_truncates_at_corner_cell(rt):
    """Corner recirculation cell: first crossing is *not* the main reattachment."""
    row = np.array([-0.010, -0.030, -0.005, 0.001, -0.040, -0.030, -0.010, 0.002, 0.010])
    assert rt.legacy_first_crossing(row) is not None
    assert abs(rt.legacy_first_crossing(row) - 2.833) < 0.05  # inside the corner cell
    res = rt.locate_reattachment(row, smooth=1, rel_thresh=1e-2, u_scale=0.05)
    assert res.status == rt.REATTACHED
    assert res.n_negative_runs == 2
    assert res.run_lengths == [3, 3]
    assert res.run_areas[1] > res.run_areas[0]  # the main bubble is the deeper one
    assert res.x is not None and res.x > 5.0  # main bubble closure, not the corner


def test_status_separates_no_backflow_from_not_reattached(rt):
    """``None`` used to mean three different things; X_r = 0 is only correct for one."""
    attached = rt.locate_reattachment(np.array([0.01, 0.02, 0.03]), rel_thresh=1e-2)
    assert attached.status == rt.NO_BACKFLOW and attached.x is None

    open_bubble = rt.locate_reattachment(np.array([-0.01, -0.02, -0.03]), rel_thresh=1e-2)
    assert open_bubble.status == rt.NOT_REATTACHED and open_bubble.x is None

    real = rt.locate_reattachment(np.array([-0.03, -0.02, -0.01, 0.01, 0.02]), rel_thresh=1e-2)
    assert real.status == rt.REATTACHED and real.x is not None


def test_single_cell_noise_blip_is_rejected(rt):
    """One-cell negative blip far downstream must not be mistaken for the main bubble."""
    clean = np.array([-0.05, -0.06, -0.04, -0.01, 0.01, 0.03, 0.04, 0.04, 0.03])
    noisy = clean.copy()
    noisy[8] = -0.004  # 1e-3 * u_scale threshold is 6e-5 -> smoothed away by width 3
    a = rt.locate_reattachment(clean, rel_thresh=1e-2, u_scale=0.06)
    b = rt.locate_reattachment(noisy, rel_thresh=1e-2, u_scale=0.06)
    assert a.status == b.status == rt.REATTACHED
    assert abs(a.x - b.x) < 0.3


def test_smoothing_uses_edge_replication(rt):
    """Zero padding would shrink the corner-node backflow and delay separation onset."""
    row = np.array([-0.3, -0.2, 0.1, 0.2])
    sm = rt._smooth(row, 3)
    assert sm[0] == pytest.approx((-0.3 - 0.3 - 0.2) / 3.0)


def test_measure_planes_excludes_invalid_planes(rt):
    """Per-plane statistics must not fold invalid planes in as xr_h = 0.0."""
    nz, ny, nx, x_step, h = 4, 4, 16, 6, 3
    ux = np.zeros((nz, ny, nx))
    sep = np.array([-0.05, -0.06, -0.03, 0.01, 0.03, 0.03, 0.02, 0.01, 0.0, 0.0])
    attached = np.full_like(sep, 0.05)
    ux[0, 1, x_step : x_step + sep.size] = sep
    ux[1, 1, x_step : x_step + sep.size] = sep
    ux[2, 1, x_step : x_step + sep.size] = attached
    ux[3, 1, x_step : x_step + sep.size] = attached
    out = rt.measure_planes(ux, x_step, h, u_scale=0.06)
    assert out["xr_planes_valid"] == 2
    assert out["xr_planes_invalid"] == 2
    assert out["xr_planes_mean"] > 0.0  # computed from the two separated planes only
    assert out["xr_plane_statuses"] == {rt.NO_BACKFLOW: 2, rt.REATTACHED: 2}


def test_wired_into_benchmark_script(rt):
    """``run.py`` must report a located X_r plus a status for a separated field."""
    torch = pytest.importorskip("torch")
    run = _load("run")
    nx, ny, nz, x_step = 48, 24, 4, 12
    h = 8
    ux = torch.zeros(nz, ny, nx)
    prof = np.array([-0.05, -0.06, -0.05, -0.03, -0.01, 0.005, 0.02, 0.03, 0.02, 0.0, 0.0, 0.0])
    ux[:, 1, x_step : x_step + prof.size] = torch.tensor(prof, dtype=torch.float32)
    out = run.measure_reattach_3d(ux, x_step, h, u_max=0.06)
    assert out["xr_status"] == rt.REATTACHED
    assert out["xr_h"] is not None and out["xr_h"] > 0.0
    assert out["xr_planes_valid"] == nz
    assert len(out["xr_plane_statuses"]) == 1


def test_recorded_records_are_internally_inconsistent(rt):
    """Regression on the shipped evidence: re400 claims X_r = 0 while separated.

    The published JSON is kept as-is (history); this test fails loudly if a
    future re-record ever reintroduces the "separated but X_r = 0" pattern.
    """
    path = _CASE_DIR / "result_re400_H49.json"
    rec = json.loads(path.read_text(encoding="utf-8"))
    res = rec["result"]
    separated = res["separation_bubble"]["max_backflow_over_umax"] < -1e-2
    assert separated, "record no longer shows a separated flow; revisit this guard"
    assert res["xr_h"] == 0.0, "pre-fix record expected; re-record and update this test"
    assert res["converged"] is True, "the pre-fix convergence logic accepted the null measurement"