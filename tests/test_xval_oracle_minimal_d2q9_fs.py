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
XV = HERE.parent / "benchmarks" / "xval_oracle_minimal_d2q9_fs" / "reference_study"
sys.path.insert(0, str(XV))
sys.path.insert(0, str(HERE.parent / "src"))


def test_design_declares_owner_gate():
    import run_xval as X

    assert X.DESIGN["thresholds_pct"]["z_max_rel_err"] == 15.0
    assert X.DESIGN["thresholds_pct"]["arm_arm_max_rel_diff"] == 15.0
    assert X.DESIGN["window_T"] == [0.5, 4.0]


def test_port_matches_upstream_snapshot():
    """Port == unmodified upstream (float32 round-off) on a static column."""
    import oss_original as O

    if not (Path(O.OSS_ROOT) / "lbm" / "util.py").exists():
        pytest.skip(f"OSS snapshot absent ({O.OSS_ROOT}) — port/upstream "
                    "equivalence not checkable on this host")
    import oss_oracle_port as P

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

    a = 8
    r = X.run_tensorlbm(a=a, tau=1.0, g=5e-5, t_max=1.0, sample_dt=0.25)
    assert r["status"] == "ok"
    Z = [s["Z"] for s in r["series"]]
    Zf = [s["Z_fill"] for s in r["series"]]
    # legacy R1 convention counts the empty INTERFACE envelope -> (a+1)/a at t=0
    assert Z[0] == pytest.approx((a + 1) / a, abs=1e-12)
    # physical (fill-based) front is exactly the column width at t=0
    assert Zf[0] == pytest.approx(1.0, abs=1e-12)
    assert Z[0] - Zf[0] == pytest.approx(1.0 / a, abs=1e-12)
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


def test_front_metric_definitions_differ_by_one_interface_cell():
    """``Z_cell`` is physical (fill-based); ``Z_cell_flag`` counts the empty
    INTERFACE envelope and therefore reads exactly +1/a higher."""
    import numpy as np

    import resolution_study as R

    a, h0 = 4.0, 8.0
    fill = np.zeros((8, 20), dtype=np.float32)
    fill[:, 1:5] = 1.0                      # 4 liquid columns, indices 1..4
    mask = np.zeros((8, 20), dtype=bool)
    mask[:, 1:6] = True                     # + empty interface envelope at index 5
    m = R.snapshot_metrics(fill, a=a, h0=h0, fluid_mask=mask)
    assert m["Z_cell"] == pytest.approx(4 / a)
    assert m["Z_cell_flag"] == pytest.approx(5 / a)
    assert m["Z_cell_flag"] - m["Z_cell"] == pytest.approx(1.0 / a)
    # sub-cell metrics interpolate the toe and are not offset by a whole cell
    assert m["Z_sub0.1"] <= m["Z_cell"] + 1.0 / a


def test_series_roundtrip_through_log_line():
    """series_row -> series_from_rows must preserve every judged metric."""
    import resolution_study as R

    m = {"T": 1.25, "step": 400, "Z_cell": 2.5, "Z_cell_flag": 2.5625,
         "Z_sub0.02": 2.4375, "Z_sub0.05": 2.375, "Z_sub0.1": 2.3125}
    back = R.series_from_rows([R.series_row(m)])[0]
    for k in R.SER_KEYS:
        assert back[k] == pytest.approx(m[k], abs=1e-9)
    assert back["T"] == pytest.approx(m["T"])


def test_collector_rebuilds_runs_from_job_log(tmp_path, monkeypatch):
    """The stdout recovery path must rebuild the artifact analyzer consumes."""
    import json

    import collect_job_stdout as C
    import resolution_study as R

    rec = {"l": "tensorlbm_koerner_a16_tau1", "k": "arm", "a": 16, "tau": 1.0,
           "st": "ok", "steps": 3200, "wall": 54.6, "u": 0.2377,
           "art": "artifacts/runs_r3/tensorlbm_koerner_a16_tau1.json",
           "S": [R.series_row({"T": 0.5, "Z_cell": 1.6875, "Z_cell_flag": 1.75,
                               "Z_sub0.02": 1.65, "Z_sub0.05": 1.6, "Z_sub0.1": 1.55}),
                 R.series_row({"T": 1.0, "Z_cell": 2.125, "Z_cell_flag": 2.1875,
                               "Z_sub0.02": 2.1, "Z_sub0.05": 2.05, "Z_sub0.1": 2.0})],
           "inv": 0.0322}
    log = tmp_path / "cj9999.log"
    log.write_text(json.dumps({"state": "done", "rc": 0,
                               "output_tail": "R2SER:" + json.dumps(rec) + "\nR2END:ladder\n"}))
    monkeypatch.setattr(C, "LOG_DIR", tmp_path)
    got = C.parse_job("cj9999")
    assert got["complete"] is True
    run = got["runs"]["tensorlbm_koerner_a16_tau1"]
    assert run["steps"] == 3200 and run["a"] == 16
    assert len(run["series"]) == 2
    ev = R.eval_run(run, "mm")
    assert "Z_cell_flag_err_pct" in ev["vs_mm"]["points"][0]
    assert ev["vs_mm"]["Z_cell_flag"]["max_abs_rel_err_pct"] is not None
    # the two conventions must differ by exactly the declared +1 cell
    assert (ev["vs_mm"]["points"][0]["Z_cell_flag"]
            - ev["vs_mm"]["points"][0]["Z_cell"]) == pytest.approx(1.0 / 16.0, abs=1e-9)
