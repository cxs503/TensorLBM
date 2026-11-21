"""Tests for the Re = 3200 lid-driven cavity reproduction (anchor ``REPRO-LBM-HOU1995``).

The assertions are *physical or contractual*, not "the code ran":

* the admitted Ghia (1982) Re = 3200 column must still hash to the value the
  builder recorded, and a tampered table must be refused (not silently scored);
* scoring must return exactly zero error when the profile *is* the reference
  (an interpolation/sign/units bug otherwise hides in the noise of a real run),
  and must report a known offset as that offset in percent of the lid speed;
* the quarantined OCR cell must stay out of the scored rows;
* the runner must drive the library's own solver (checked against an independent
  direct ``CavitySolver`` run of the same length) rather than a stub;
* a run without a reference is ``WITHHELD`` and a run against a deliberately
  wrong reference must report failure -- no silent pass in either direction.

Everything here is small (``n`` <= 33, <= 200 steps) so the file stays a fast
unit test; the production grids live in ``benchmarks/verified/cavity/re3200/out``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks" / "verified" / "cavity" / "re3200"

from tensorlbm.verification.cavity_re3200 import reference as refmod  # noqa: E402
from tensorlbm.verification.cavity_re3200 import runner, scoring  # noqa: E402


@pytest.fixture(scope="module")
def ref() -> dict:
    return refmod.load_reference()


# --------------------------------------------------------------------------- #
# reference table: admission gates
# --------------------------------------------------------------------------- #
def test_reference_is_admitted_and_self_consistent(ref: dict) -> None:
    assert float(ref["re"]) == 3200.0
    assert ref["_hash_verified"], "recorded content hash must reproduce"
    assert len(ref["u_re3200"]) == 16 and len(ref["v_re3200"]) == 17
    u = {float(a): float(b) for a, b in ref["u_re3200"]}
    v = {float(a): float(b) for a, b in ref["v_re3200"]}
    assert u[1.0] == pytest.approx(1.0)      # lid
    assert u[0.0] == pytest.approx(0.0)      # bottom wall
    assert v[0.0] == pytest.approx(0.0)      # left wall
    assert v[1.0] == pytest.approx(0.0)      # right wall


def test_tampered_reference_is_refused(tmp_path: Path, ref: dict) -> None:
    """A mutated table must raise, not score against a silently different value."""
    body = {k: v for k, v in ref.items() if not k.startswith("_")}
    body["u_re3200"][1][1] = 0.9            # 0.53236 -> 0.9
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(refmod.ReferenceError, match="hash mismatch"):
        refmod.load_reference(path)


def test_quarantined_cell_is_not_a_scored_row(ref: dict) -> None:
    cells = refmod.quarantined(ref)
    assert len(cells) == 1
    coord = float(cells[0]["coord"])
    scored = [float(c) for c, _ in ref["u_re3200"]]
    assert coord not in scored


# --------------------------------------------------------------------------- #
# scoring: units, sign, conventions
# --------------------------------------------------------------------------- #
def _linear_reference() -> dict:
    """A reference table that is *exactly* representable by any sampling grid.

    ``u = 1 - 2y`` and ``v = 1/2 - x`` are linear, so interpolating them at the
    run's lattice ordinates and back onto the table ordinates must be exact: any
    residual error is a bug in units, sign, or the coordinate convention, not
    modelling noise.
    """
    coords = scoring.coords(65, "node")
    return {
        "re": 3200.0,
        "u_re3200": [[float(c), 1.0 - 2.0 * float(c)] for c in coords[::-1]],
        "v_re3200": [[float(c), 0.5 - 1.0 * float(c)] for c in coords[::-1]],
        "provenance": {"quarantined": []},
    }


def _synthetic_linear_run(n: int = 65, convention: str = "wall",
                          offset: float = 0.0) -> dict:
    x = scoring.coords(n, convention)
    return {"n": n,
            "u_centerline": (1.0 - 2.0 * x + offset).tolist(),
            "v_centerline": (0.5 - 1.0 * x + offset).tolist()}


def _synthetic_run(ref: dict, n: int = 65, convention: str = "wall",
                   offset: float = 0.0) -> dict:
    """A run whose centreline follows the reference profile (optionally shifted)."""
    x = scoring.coords(n, convention)
    # the table rows are stored top-down (y = 1 first), np.interp wants ascending xp
    u_ref = sorted((float(a), float(b)) for a, b in ref["u_re3200"])
    v_ref = sorted((float(a), float(b)) for a, b in ref["v_re3200"])
    u = np.interp(x, [c for c, _ in u_ref], [b for _, b in u_ref]) + offset
    v = np.interp(x, [c for c, _ in v_ref], [b for _, b in v_ref]) + offset
    return {"n": n, "u_centerline": u.tolist(), "v_centerline": v.tolist()}


def test_scoring_is_exact_on_a_linear_profile() -> None:
    """Exactness invariant: linear truth in => zero error out, both conventions."""
    ref = _linear_reference()
    for convention in scoring.CONVENTIONS:
        s = scoring.score(_synthetic_linear_run(convention=convention), ref, convention)
        assert s["max_abs_err_pct"] == pytest.approx(0.0, abs=1e-10), convention
        assert s["pass"] is True
        assert len(s["u_rows"]) == 65 and len(s["v_rows"]) == 65
        for coord, r, got, err in s["u_rows"]:
            assert got == pytest.approx(r, abs=1e-12)
            assert err == pytest.approx(0.0, abs=1e-10)


def test_scoring_reports_offset_in_percent_of_lid_speed() -> None:
    """A +0.02 lid-speed offset must come back as +2.00 %, not 0.02 or 200."""
    ref = _linear_reference()
    s = scoring.score(_synthetic_linear_run(offset=0.02), ref, "wall", tol_pct=5.0)
    errs = [r[3] for r in s["u_rows"]]
    assert max(errs) == pytest.approx(2.0, abs=1e-9)
    assert min(errs) == pytest.approx(2.0, abs=1e-9)
    assert s["pass"] is True
    s_big = scoring.score(_synthetic_linear_run(offset=0.08), ref, "wall", tol_pct=5.0)
    assert s_big["pass"] is False
    assert s_big["max_abs_err_pct"] == pytest.approx(8.0, abs=1e-9)


def test_scoring_recovers_the_admitted_profile_within_sampling_noise(ref: dict) -> None:
    """Feeding the admitted profile back through the scorer is ~lossless.

    The residual is interpolation noise between the table's kinks and the run's
    ordinates; it must stay well below the 5 % tolerance at a resolvable grid, and
    the near-wall rows (a half cell from the kink) are reported, not hidden.
    """
    s = scoring.score(_synthetic_run(ref, n=257), ref, "wall")
    interior = [r for r in s["u_rows"] + s["v_rows"] if 0.02 <= r[0] <= 0.98]
    assert max(abs(r[3]) for r in interior) < runner.TOL_PCT
    assert s["max_abs_err_pct"] < runner.TOL_PCT
    # the two wall ordinates are the only rows a half-cell convention cannot hit
    ends = [r for r in s["u_rows"] if r[0] in (0.0, 1.0)]
    assert len(ends) == 2


def test_coordinate_conventions_differ_by_half_a_cell(ref: dict) -> None:
    n = 65
    node = scoring.coords(n, "node")
    wall = scoring.coords(n, "wall")
    assert node[0] == 0.0 and node[-1] == 1.0
    assert wall[0] == pytest.approx(-0.5 / (n - 2))
    assert wall[-1] == pytest.approx(1.0 + 0.5 / (n - 2))
    # interior rows differ by ~half a cell: every node ordinate sits between the
    # two neighbouring wall-convention ordinates
    # interior rows differ by ~half a cell: every node ordinate sits strictly
    # between the two neighbouring wall-convention ordinates
    # the two conventions are different grids (/ (n-1) vs / (n-2) with the walls
    # half a cell outside the fluid), so they drift apart by less than one cell
    assert np.max(np.abs(node - wall)) < 1.0 / (n - 2) + 1.0 / (n - 1)
    assert wall[1] < node[1] < 0.5 and node[1] - wall[1] < 1.0 / (n - 2)


def test_quarantine_sensitivity_covers_both_bounds(ref: dict) -> None:
    run = _synthetic_run(ref)
    sens = scoring.quarantine_sensitivity(run, ref, "wall")
    lo, hi = sens["admissible_interval"]
    assert lo < hi
    assert sens["err_pct_at_lower_bound"] == pytest.approx((sens["got"] - lo) * 100.0)
    assert sens["err_pct_at_upper_bound"] == pytest.approx((sens["got"] - hi) * 100.0)
    # the interval is narrow relative to the bracket it sits in
    assert sens["interval_width_pct"] < 25.0


def test_grid_convergence_needs_two_grids(ref: dict) -> None:
    single = {"128": scoring.score(_synthetic_run(ref, n=129), ref, "wall")}
    assert scoring.grid_convergence(single)["status"] == "WITHHELD"


# --------------------------------------------------------------------------- #
# runner contract
# --------------------------------------------------------------------------- #
def test_runner_drives_the_library_solver_not_a_stub() -> None:
    """The record must equal an independent direct solver run of the same length."""
    import torch
    from tensorlbm.verification.lid_driven_cavity.solver import CavityConfig, CavitySolver

    torch.set_num_threads(1)
    rec = runner.run_grid(25, max_steps=60, monitor_every=30, threads=1)
    cfg = CavityConfig(n=25, re=runner.RE, u_lid=runner.U_LID, use_rlbm=True,
                       ramp_steps=1000)   # runner.run_grid's default ramp
    solver = CavitySolver(cfg)
    for _ in range(60):
        solver.step()
    _, ux, uy = solver._macroscopic(solver.f)
    u, v = runner.centerline(ux, uy, 25, True)
    assert rec["tau"] == pytest.approx(cfg.tau)
    assert np.allclose(np.asarray(rec["u_centerline"]) * runner.U_LID, u, atol=1e-12)
    assert np.allclose(np.asarray(rec["v_centerline"]) * runner.U_LID, v, atol=1e-12)


def test_runner_record_schema_without_reference() -> None:
    rec = runner.run_grid(33, max_steps=120, monitor_every=40, threads=1)
    assert rec["run_status"] == "ok"
    assert rec["steps"] == 120
    assert rec["converged"] is False            # no resid_tol => not a steady claim
    assert rec["comparison"]["status"] == "WITHHELD"
    assert len(rec["u_centerline"]) == 33 and len(rec["v_centerline"]) == 33
    assert len(rec["y_centerline_wall"]) == 33
    assert np.all(np.isfinite(rec["u_centerline"]))
    assert rec["resid_history"], "the monitor must record residuals"
    assert all(row[1] is not None for row in rec["resid_history"])


def test_runner_resid_tol_stops_and_flags_convergence() -> None:
    rec = runner.run_grid(25, max_steps=600, monitor_every=100, resid_tol=1e3,
                          threads=1)
    assert rec["converged"] is True
    assert rec["steps"] == 100                  # stopped at the first monitor
    assert rec["run_status"] == "ok"


def test_runner_config_error_is_a_record_not_a_crash() -> None:
    rec = runner.run_grid(12, max_steps=10, monitor_every=5)
    assert rec["run_status"] == "config_error"
    assert "n=12" in rec["error"] or "too small" in rec["error"]
    assert rec["u_centerline"] is None
    assert rec["comparison"]["status"] == "VOID"


def test_runner_reports_failure_against_wrong_reference(ref: dict) -> None:
    """A wrong reference must produce a failing verdict, never a fabricated pass."""
    body = {k: v for k, v in ref.items() if not k.startswith("_")}
    body["u_re3200"] = [[c, -u] for c, u in body["u_re3200"]]
    body["v_re3200"] = [[c, -v] for c, v in body["v_re3200"]]
    wrong = dict(body)
    rec = runner.run_grid(25, max_steps=80, monitor_every=40, ref=wrong, threads=1)
    assert rec["comparison"]["wall"]["pass"] is False
    assert rec["comparison"]["wall"]["max_abs_err_pct"] > 50.0


def test_write_record_names_and_hashes(tmp_path: Path) -> None:
    rec = runner.run_grid(25, max_steps=40, monitor_every=20, threads=1)
    path = runner.write_record(rec, tmp_path)
    assert path.name == "re3200_n25.json"
    payload = json.loads(path.read_text())
    assert payload["n"] == 25 and payload["anchor"] == runner.ANCHOR
    side = (tmp_path / "re3200_n25.sha256").read_text().split()
    import hashlib
    assert side[0] == hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# the shipped entry point (benchmarks/verified/cavity/re3200/run.py)
# --------------------------------------------------------------------------- #
def test_benchmark_entry_writes_records_and_manifest(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(BENCH / "run.py"), "--n", "25", "--steps", "60",
         "--monitor-every", "30", "--out-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    rec = json.loads((tmp_path / "re3200_n25.json").read_text())
    assert rec["n"] == 25 and rec["case"] == "cavity_re3200_ghia"
    assert rec["comparison"]["wall"]["tolerance_pct"] == runner.TOL_PCT
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["anchor"] == runner.ANCHOR
    assert manifest["reference"]["hash_verified"] is True
    grid = manifest["grids"][0]
    assert "--n 25" in grid["repro_cmd"]
    import hashlib
    on_disk = hashlib.sha256((tmp_path / "re3200_n25.json").read_bytes()).hexdigest()
    assert grid["sha256"] == on_disk
    assert (tmp_path / "re3200_n25.sha256").read_text().split()[0] == on_disk


def test_benchmark_entry_without_reference_is_withheld(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(BENCH / "run.py"), "--n", "25", "--steps", "40",
         "--monitor-every", "20", "--no-ref", "--out-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    rec = json.loads((tmp_path / "re3200_n25.json").read_text())
    assert rec["comparison"]["status"] == "WITHHELD"
    assert "pass" not in rec["comparison"]


# --------------------------------------------------------------------------- #
# the committed production artefacts (skipped when the runs have not landed)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [128, 192, 256])
def test_committed_run_artifacts_are_wellformed(n: int) -> None:
    path = BENCH / "out" / f"re3200_n{n}.json"
    if not path.is_file():
        pytest.skip(f"{path.name} not present (run.py has not been executed here)")
    rec = json.loads(path.read_text())
    assert rec["n"] == n and rec["re"] == 3200.0
    assert rec["run_status"] in {"ok", "not_converged"}
    assert rec["steps"] > 0 and rec["elapsed_s"] > 0
    assert len(rec["u_centerline"]) == n
    cmp_ = rec["comparison"]
    assert cmp_["tol_pct"] == runner.TOL_PCT
    for conv in scoring.CONVENTIONS:
        block = cmp_[conv]
        assert len(block["u_rows"]) == 16 and len(block["v_rows"]) == 17
        recomputed = max(abs(r[3]) for r in block["u_rows"] + block["v_rows"])
        assert block["max_abs_err_pct"] == pytest.approx(recomputed)
        assert block["pass"] == (recomputed <= block["tolerance_pct"])
    assert "repro_cmd" in rec and "--n" in rec["repro_cmd"]
def test_cavity_modules_and_bundled_reference_use_the_installed_package():
    import tensorlbm
    from tensorlbm.verification.lid_driven_cavity import solver
    from tensorlbm.verification.square_duct_poiseuille import analytical

    root = Path(tensorlbm.__file__).resolve().parent
    for module in (runner, scoring, refmod, solver, analytical):
        assert Path(module.__file__).resolve().is_relative_to(root)
    bundled = Path(refmod.__file__).with_name(refmod.REF_NAME)
    assert bundled.read_bytes() == (BENCH / "reference" / refmod.REF_NAME).read_bytes()
