"""CASE-BACKWARD-STEP-LBM -- the compatibility CLI (``run.py``) must stay judgeable.

Round 1 shipped ``benchmarks/case_pool/backward_step_armaly/run.py`` with its own scalar
LBM loop and its own artefact schema.  Measured on a 1200-step probe, the round-1 payload
came back from ``verdict.py`` as ``note_missing = [error, steady, caliper, mass, lattice]``
-- i.e. no point it produced could be judged at all, because ``derived.mach_inlet``,
``result.mass_flux_rel_error``, ``result.x1_over_s_reverse_extent`` and
``convergence_history`` were absent.  Two implementations of one benchmark could also drift
apart silently (same probe: legacy caliper 1.43 vs library caliper 2.63 at Re=100, both
unconverged, which is exactly why a single caliper must be graded).

These tests pin the round-3 contract: ``run.py`` is a thin shim over
``tensorlbm.bfs_armaly`` and whatever it writes is machine-judgeable as-is.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CASE = REPO / "benchmarks" / "case_pool" / "backward_step_armaly"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(CASE))

import verdict as V  # noqa: E402


def _load_run_module():
    spec = importlib.util.spec_from_file_location("bfs_case_run_cli", CASE / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RUN = _load_run_module()
REFS = json.loads((CASE / "refs.json").read_text())


def _ns(**kw):
    base = dict(re=100.0, m=16, u=0.1, steps=1200, avg_frac=0.25, avg_stride=200,
                l_over_h=12.0, collision="trt", er2=False, tag=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_default_geometry_keeps_the_measured_armaly_ratio():
    """--m is the step height; the inlet height follows 52:49, so ER stays 1.94, not 2."""
    cfg = RUN.config_from_args(_ns(m=49))
    assert (cfg.hi, cfg.s) == (52, 49)
    assert cfg.expansion_ratio == pytest.approx(10.1 / 5.2, rel=1e-12)
    cfg16 = RUN.config_from_args(_ns(m=16))
    assert (cfg16.hi, cfg16.s) == (17, 16)
    assert abs(cfg16.expansion_ratio - 10.1 / 5.2) < 0.01


def test_er2_flag_restores_the_round1_lattice():
    cfg = RUN.config_from_args(_ns(m=26, er2=True))
    assert (cfg.hi, cfg.s) == (26, 26)
    assert cfg.expansion_ratio == pytest.approx(2.0)
    assert cfg.tag == "cli_re100_m26_er2"


def test_rlbm_collision_name_is_still_accepted_as_trt():
    assert RUN.config_from_args(_ns(collision="rlbm")).collision == "trt"
    assert RUN.config_from_args(_ns(collision="bgk")).collision == "bgk"


def test_cli_artefact_is_machine_judgeable(tmp_path):
    """Whatever the historical CLI writes must be gradable by verdict.py with no adapter."""
    out = tmp_path / "legacy_flat.json"
    rc = RUN.main(["--re", "100", "--m", "9", "--steps", "1200", "--u", "0.1",
                   "--threads", "2", "--artifacts-dir", str(tmp_path / "art"),
                   "--out", str(out)])
    assert rc == 0

    tag_dir = tmp_path / "art" / "cli_re100_m9"
    payload = json.loads((tag_dir / "result.json").read_text())
    # same artefact triple as run_case.py / the library
    for name in ("result.json", "wall_shear.csv", "field.npz"):
        assert (tag_dir / name).is_file(), name
    # --out mirrors the identical payload (round-1 habit kept)
    assert json.loads(out.read_text())["result"]["x1_over_s"] == payload["result"]["x1_over_s"]

    # every field the gate reads must exist: the round-1 payload lacked all four of these,
    # which is why verdict.py returned note_missing = [error, steady, caliper, mass, lattice]
    res, der = payload["result"], payload["derived"]
    assert der["mach_inlet"] == pytest.approx(1.5 * 0.1 / (1.0 / 3.0) ** 0.5, rel=1e-12)
    assert der["tau"] >= 0.55
    assert isinstance(res["mass_flux_rel_error"], float)
    assert isinstance(res["x1_over_s_reverse_extent"], float)
    assert payload["convergence_history"]
    assert "x1_over_s_instant" in payload["convergence_history"][-1]

    row = V.judge_run("cli_re100_m9", payload, REFS)
    assert row["checks"]["lattice"] is True          # tau >= 0.55 and mach <= 0.3
    assert row["checks"]["error"] is True            # |X1/s - anchor| within 15 %
    assert row["checks"]["caliper"] is True          # reverse extent <= X1/s + 0.5
    assert row["steadiness"]["n_checks"] >= 1
    # a 1200-step spin-up is *not* converged: the gate must say so, not pass it
    assert row["checks"]["steady"] is False
    assert "steady" in row["note_missing"]


def test_run_py_holds_no_solver_loop_of_its_own():
    """Regression guard: the duplicated collision/stream/caliper code must not come back."""
    src = (CASE / "run.py").read_text()
    for banned in ("first_upcrossing", "collide_bgk", "eq_rho_u", "for step in range"):
        assert banned not in src, f"{banned!r} re-introduces the round-1 duplicate solver"
    assert "run_armaly_case" in src

EVIDENCE = REPO / "evidence" / "case_backward_step_armaly_cli" / "cli_re100_m16"


@pytest.mark.skipif(not (EVIDENCE / "result.json").is_file(), reason="CLI evidence artefact absent")
def test_committed_cli_evidence_is_a_judgeable_transient_point():
    """The committed artefact of the documented reproduction command keeps its meaning.

    12000 steps at Re=100 is a *transient* (drift 4.5 % > 1 %), so the honest verdict is
    "error judged, steadiness/mass refused" -- if this ever turns into a PASS the run
    length changed and the doc statement must be revisited.
    """
    payload = json.loads((EVIDENCE / "result.json").read_text())
    row = V.judge_run("cli_re100_m16", payload, REFS)
    assert row["checks"]["error"] is True and abs(row["error_pct"]) < 15.0
    assert row["checks"]["lattice"] is True
    assert row["checks"]["steady"] is False
    assert "steady" in row["note_missing"]
    assert payload["config"]["hi"] == 17 and payload["config"]["s"] == 16
