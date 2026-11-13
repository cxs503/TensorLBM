"""Tests for XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS.

Fast (< 60 s) guards for the three things the report leans on:

1. the OSS oracle port is *upstream-equivalent* (compared against the
   unmodified ``lbm`` package driven through the numba stand-in);
2. the port's liquid ledger is conservative on a static column;
3. the TensorLBM D2Q9 Körner arm runs the shared case, keeps its ledger and
   produces a monotone front, and the pre-declared 15 % gate is wired into
   ``evaluate``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
XV = HERE.parent / "xval" / "oracle_minimal_d2q9_fs"
sys.path.insert(0, str(XV))
sys.path.insert(0, str(HERE.parent / "src"))


def test_design_declares_owner_gate():
    import run_xval as X

    assert X.DESIGN["thresholds_pct"]["z_max_rel_err"] == 15.0
    assert X.DESIGN["thresholds_pct"]["arm_arm_max_rel_diff"] == 15.0
    assert X.DESIGN["window_T"] == [0.5, 4.0]


def test_port_matches_upstream_snapshot():
    """Port == unmodified upstream (float32 round-off) on a static column."""
    import oss_oracle_port as P
    import oss_original as O

    # keep repo ``src/lbm`` from shadowing the OSS ``lbm`` package
    src_path = str(HERE.parent / "src")
    while src_path in sys.path:
        sys.path.remove(src_path)
    oss = O.import_oss()
    CT = oss["util"].CT
    fin, fout, equi, fdist, inlet, u, rho, mass, ct = O.make_state(40, 24, 8, 16, oss)
    st = P.make_oss_block(nx=40, ny=24, a=8, h0=16)
    for _ in range(5):
        vel, _ = oss["evolve"](1, fin, fout, equi, fdist, inlet, u, rho, mass, ct,
                               np.array([1.0]), oss["util"].v, oss["util"].v_inv,
                               oss["util"].t, np.array([0.0, 0.0]))
        u = vel[-1]
        P.step(st, omega=1.0, gravity=(0.0, 0.0), rho_prev_alias=True)
    assert np.isfinite(rho).all()
    assert (ct == st.flags()).all(), "cell types diverged from upstream"
    assert np.abs(fin - P.I(st.fin)).max() < 1e-5
    assert np.abs(mass - P.I(st.mass)).max() < 1e-5
    assert np.abs(u - P.I(st.u)).max() < 1e-5
    inv_up = float(rho[(ct & CT.FLUID) != 0].sum() + mass[(ct & CT.INTERFACE) != 0].sum())
    assert abs(inv_up - st.liquid_inventory()) < 1e-4


def test_port_ledger_conservative_static():
    import oss_oracle_port as P

    st = P.make_dambreak(nx=160, ny=96, a=32, h0=64)
    inv0 = st.liquid_inventory()
    for _ in range(25):
        P.step(st, omega=1.0, gravity=(0.0, 0.0))
    assert abs(st.liquid_inventory() - inv0) / inv0 < 1e-3


def test_tensorlbm_arm_smoke():
    import run_xval as X

    r = X.run_tensorlbm(a=8, tau=1.0, g=5e-5, t_max=1.0, sample_dt=0.25)
    assert r["status"] == "ok"
    Z = [s["Z"] for s in r["series"]]
    assert Z[0] == pytest.approx(1.0, abs=0.1)
    assert Z[-1] >= Z[0], "front retreated"
    assert max(Z) > 1.0, f"front never left the column: {Z}"
    assert max(abs(p["drift_pct"]) for p in r["inventory"]) < 0.5
    assert r["u_max"] < 0.5


def test_evaluate_gate_logic():
    import run_xval as X

    good = {"arm": "a", "a": 8, "tau": 1.0, "g": 5e-5,
            "series": [{"step": 0, "T": 0.5, "Z": 1.45}],
            "inventory": [{"step": 0, "drift_pct": 0.01}]}
    bad = json.loads(json.dumps(good))
    bad["series"][0]["Z"] = 2.5
    assert X.evaluate(good)["vs_mm"]["pass_15pct"] is True
    assert X.evaluate(bad)["vs_mm"]["pass_15pct"] is False