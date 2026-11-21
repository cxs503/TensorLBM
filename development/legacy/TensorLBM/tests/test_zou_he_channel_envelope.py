"""Regression tests for the validity envelope + independent reproduction.

Companion of ``docs``-level record
``xval/XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI/ENVELOPE.md``.
The slow, publishable runs live in ``run_envelope.sh``; what is locked here is
the *structure* that those runs rely on, at a size that fits a normal test run:

1. the second code path (``independent.py``) streams and closes exactly like the
   library -- index table, algebra and streaming window all verified, so a slip
   in either path fails here instead of quietly biasing a report number;
2. the second code path reproduces the library channel solution (profile,
   amplitude, mass defect) on a small case;
3. the streaming window covers every column, i.e. the exact bug found while
   writing the independent solver (end columns never streamed) cannot come back;
4. the published envelope records stay inside the acceptance bounds, and the
   empirical wall-position law ``delta(tau) = 1/64 - (tau-1/2)^2/12`` still
   predicts the measured error (``rel_u_max ~= 4|delta|/H``).
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
XVAL = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "xval",
        "XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI",
    )
)
ART = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "xval_zouhe"))

torch = pytest.importorskip("torch")


def _load_independent():
    path = os.path.join(XVAL, "independent.py")
    if not os.path.exists(path):  # pragma: no cover - module is part of the repo
        pytest.skip("independent.py missing")
    spec = importlib.util.spec_from_file_location("xval_independent", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


IND = _load_independent()

from tensorlbm.boundary.d2q9 import C2Q9  # noqa: E402
from tensorlbm.verification.zou_he_channel import solver as SOLVER  # noqa: E402

ZOU = importlib.import_module("tensorlbm.boundary.zou_he_channel")

# library index -> independent index (the independent module uses the textbook
# table, the library the OSS permutation)
LIB2MINE = [0, 1, 3, 2, 4, 5, 6, 8, 7]
MINE2LIB = [0] * 9
for _k, _j in enumerate(LIB2MINE):
    MINE2LIB[_j] = _k


def test_index_map_is_a_real_direction_map():
    """The permutation used by every test below must map direction vectors."""
    for k in range(9):
        assert tuple(C2Q9[k].tolist()) == tuple(IND.C[LIB2MINE[k]].tolist())


def test_streaming_window_covers_every_column():
    """Only the closure unknowns may be left empty at the two end columns.

    Regression: an early version clipped the pull window with
    ``max(dest_lo, src_lo)`` -- mixing source and destination coordinates -- and
    silently left both end columns unstreamed.  The channel still produced a
    perfectly shaped profile (shape L2 unchanged) with a 19 % amplitude error.
    """
    ny, nx = 8, 12
    rng = np.random.default_rng(7)
    f = rng.random((9, ny, nx)) * 0.01 + 0.05
    out = IND.stream(f, ny, nx)
    zero_west = {i for i in range(9) if np.all(out[i, 1:ny - 1, 0] == 0.0)}
    zero_east = {i for i in range(9) if np.all(out[i, 1:ny - 1, -1] == 0.0)}
    assert zero_west == {1, 5, 8}, f"west unknowns wrong: {zero_west}"
    assert zero_east == {3, 6, 7}, f"east unknowns wrong: {zero_east}"
    # interior columns must be fully populated
    assert not np.any(out[:, 1:ny - 1, 1:-1] == 0.0)


def test_streaming_matches_library_on_fluid_rows():
    """Pull streaming + half-way bounce-back must agree cell by cell."""
    ny, nx = 8, 12
    rng = np.random.default_rng(11)
    f = rng.random((9, ny, nx)) * 0.01 + 0.05
    mine = IND.stream(f, ny, nx)
    fluid = torch.zeros((ny, nx), dtype=torch.bool)
    fluid[1:ny - 1, :] = True
    lib = SOLVER._stream(torch.tensor(f[LIB2MINE].copy()), fluid).numpy()[MINE2LIB]
    np.testing.assert_allclose(lib[:, 1:ny - 1, :], mine[:, 1:ny - 1, :],
                              rtol=0, atol=1e-15)


@pytest.mark.parametrize("rho_p", [1.002, 0.998])
@pytest.mark.parametrize("side,fn", [("west", IND._zou_he_west), ("east", IND._zou_he_east)])
def test_closure_algebra_matches_library(side, fn, rho_p):
    """Independent derivation must reproduce the library closure to round-off."""
    ny = 9
    rng = np.random.default_rng(3)
    line = rng.random((9, ny)) * 0.01 + 0.05
    mine = line.copy()
    fn(mine, np.full(ny, rho_p), np.zeros(ny))
    f = torch.zeros((9, ny, 3), dtype=torch.float64)
    f[:, :, 0 if side == "west" else 2] = torch.tensor(line[LIB2MINE].copy())
    ZOU.zou_he_wall(f, side, mode="pressure", rho=rho_p)
    lib = f[:, :, 0 if side == "west" else 2].numpy()[MINE2LIB]
    np.testing.assert_allclose(lib, mine, rtol=0, atol=1e-14)
    # the prescribed density is imposed exactly, the transverse momentum is 0
    rho = mine.sum(0)
    uy = (IND.C[:, 1, None] * mine).sum(0) / rho
    np.testing.assert_allclose(rho, rho_p, rtol=0, atol=1e-13)
    np.testing.assert_allclose(uy, 0.0, atol=1e-13)


def _run_both(height=8, nx=32, steps=1500, burn_in=700, avg_window=400):
    case = SOLVER.ChannelCase(
        height=height,
        length=nx,
        tau=0.65,
        drho=4.0e-3,
        arm="pressure-pressure",
        init="analytic",
        tol=0.0,
        max_steps=steps,
        check_every=10 ** 9,
        burn_in=burn_in,
        avg_window=avg_window,
        ramp_steps=0,
    )
    res = SOLVER.run_channel(case, verbose=False)
    lib = {
        "profile": res["u_profile"].numpy(),
        "u_max": float(res["u_profile"].abs().max()),
        "mass_defect": res["mass_defect"],
        "g": res["g_measured"],
    }
    ind = IND.run(
        height=height,
        length=nx,
        tau=0.65,
        drho=4.0e-3,
        steps=steps,
        burn_in=burn_in,
        avg_window=avg_window,
        verbose=False,
    )
    return lib, ind


def test_independent_reproduces_library_channel():
    """The second code path must land on the same solution, not merely close."""
    lib, ind = _run_both()
    mine = np.array(ind["profile_mid"])
    scale = np.abs(lib["profile"]).max()
    dev = np.abs(mine - lib["profile"]).max() / scale
    assert dev < 1e-8, f"profiles diverge: {dev:.2e}"
    assert abs(ind["metrics_vs_measured_gradient"]["u_max_sim"] - lib["u_max"]) / scale < 1e-8
    assert ind["mass_defect"] < 1e-3
    # both are inside the project target against the exact parabola
    assert ind["metrics_vs_measured_gradient"]["rel_umax"] < 0.03
    assert ind["metrics_vs_nominal_gradient"]["rel_umax"] < 0.03


# --------------------------------------------------------------------------- #
# published envelope (records written by run_envelope.sh)
# --------------------------------------------------------------------------- #
def _envelope_records():
    rows = []
    for path in sorted(glob.glob(os.path.join(ART, "envelope", "*.json"))):
        with open(path) as fh:
            rows.append(json.load(fh))
    if not rows:
        pytest.skip("no envelope records; run run_envelope.sh")
    return rows


def test_envelope_inside_acceptance():
    for d in _envelope_records():
        m = d["metrics_vs_measured_gradient"]
        assert m["rel_umax"] < 0.03, (d["tau"], d["drho"], m["rel_umax"])
        assert d["metrics_vs_nominal_gradient"]["rel_umax"] < 0.03
        assert d["mass_defect"] < 1e-4, (d["tau"], d["drho"], d["mass_defect"])


def test_wall_offset_law_predicts_error():
    """delta(tau) = 1/64 - (tau-1/2)^2/12 and rel_u_max ~= 4|delta|/H."""
    rows = [d for d in _envelope_records() if abs(d["drho"] - 4.0e-3) < 1e-12]
    if len(rows) < 4:
        pytest.skip("need the tau axis of the envelope")
    tau, delta, rel = [], [], []
    for d in rows:
        h = d["height"]
        y = np.arange(1, h + 1, dtype=float)
        A = np.vstack([y * y, y, np.ones_like(y)]).T
        co, *_ = np.linalg.lstsq(A, np.array(d["profile_mid"], float), rcond=None)
        roots = np.sort(np.roots(co))
        delta.append(0.5 * ((roots[0] - 0.5) - (roots[1] - (h + 0.5))))
        tau.append(d["tau"])
        rel.append(d["metrics_vs_measured_gradient"]["rel_umax"])
        h_last = h
    tau = np.array(tau)
    delta = np.array(delta)
    q = (tau - 0.5) ** 2
    (a, b), *_ = np.linalg.lstsq(np.vstack([np.ones_like(q), q]).T, delta, rcond=None)
    assert abs(a - 1.0 / 64.0) < 2e-3, f"constant term drifted: {a:.5f}"
    assert abs(b + 1.0 / 12.0) < 5e-3, f"quadratic coefficient drifted: {b:.5f}"
    model = 4.0 * np.abs(delta) / h_last
    # the closed-form model is a one-parameter fit of a lumped diagnostic: it
    # tracks the error to <=17 % over tau in [0.55, 1.5] (worst cell tau=0.55),
    # which is what makes it useful, but it is not an equality.
    np.testing.assert_allclose(model, np.array(rel), rtol=0.20,
                              err_msg="4|delta|/H no longer predicts the error")