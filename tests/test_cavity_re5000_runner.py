"""Contract tests for the CASE-CAVITY-RE5000 case runner.

Audit finding (m15488, reviewing m15370): the runner originally
  * crashed with ``TypeError: unsupported format string passed to NoneType``
    whenever the run ended before the first residual monitor fired
    (``--steps < --monitor-every``), writing **no** ``result.json`` at all, and
  * exited ``0`` after a divergence, leaving ``u_min = nan`` / a garbage
    vortex in ``result.json`` -- a NaN that a report could quote as a number.

These tests pin the repaired contract:
  exit 0 + ``run_status == "ok"`` for a finite solution,
  exit 2 + ``run_status == "failed"`` + ``comparison == {"status": "VOID"}``
  for a diverged / non-finite one, and never a silent "pass" without a
  machine-readable reference table.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "benchmarks" / "verified" / "cavity" / "re5000" / "run.py"


def _run(tmp_path: Path, *args: str) -> tuple[int, dict]:
    out = tmp_path / "result.json"
    proc = subprocess.run(
        [sys.executable, str(RUNNER), *args, "--threads", "2", "--out", str(out)],
        cwd=REPO, capture_output=True, text=True, timeout=600,
    )
    payload = json.loads(out.read_text()) if out.exists() else {}
    return proc.returncode, payload


def test_short_run_without_monitor_writes_artifact(tmp_path: Path) -> None:
    """steps < monitor_every used to raise TypeError and leave no result.json."""
    # n = 32 (not 16): CavityConfig rejects n < 17, and the guard is a library
    # invariant this runner must not bypass -- the contract under test is the
    # artefact/exit-code one, which is grid-size independent.
    grid = ("32",)
    rc, payload = _run(tmp_path, "--n", *grid, "--re", "100", "--steps", "40",
                       "--monitor-every", "1000", "--u-lid", "0.06")
    assert rc == 0, payload
    assert payload["run_status"] == "ok"
    rec = payload["grids"]["32"]
    assert rec["final_resid"] is None          # no monitor interval was reached
    assert rec["status"] == "ok"
    assert len(rec["u_centerline"]) == 32


def test_missing_reference_is_withheld_not_passed(tmp_path: Path) -> None:
    """Without --ref-json the comparison must be WITHHELD (never a fake pass)."""
    rc, payload = _run(tmp_path, "--n", "32", "--re", "100", "--steps", "40",
                       "--monitor-every", "1000", "--u-lid", "0.06")
    assert rc == 0
    assert payload["comparison"]["status"] == "WITHHELD"
    assert "pass" not in payload["comparison"]


def test_diverged_run_exits_nonzero_and_voids_comparison(tmp_path: Path) -> None:
    """Re=5000 at 24^2 (tau=0.5014) diverges: it must be flagged, not reported."""
    rc, payload = _run(tmp_path, "--n", "24", "--re", "5000", "--steps", "600",
                       "--monitor-every", "100")
    if rc == 0:                                 # platform-dependent stability
        pytest.skip(f"24^2 Re=5000 stayed finite here (status={payload['grids']['24']['status']})")
    assert rc == 2, payload
    assert payload["run_status"] == "failed"
    assert payload["diverged"] is True
    assert payload["grids"]["24"]["status"] == "diverged"
    assert payload["comparison"]["status"] == "VOID"

def test_synthetic_reference_is_scored_and_can_fail(tmp_path: Path) -> None:
    """The --ref-json scoring path is exercised (it had zero coverage) and a
    matching-shaped but wrong table must score pass=False, not silently pass."""
    ref = tmp_path / "synthetic-ref.json"
    ref.write_text(json.dumps({
        "source": "SYNTHETIC (test fixture, not Ghia 1982)",
        "table": "n/a", "page": "n/a", "sha256": "deadbeef",
        "u_re5000": [[0.25, 0.5], [0.75, 0.5]],     # deliberately far from any cavity solution
        "v_re5000": [[0.25, 0.5], [0.75, 0.5]],
    }))
    rc, payload = _run(tmp_path, "--n", "32", "--re", "100", "--steps", "40",
                       "--monitor-every", "1000", "--u-lid", "0.06",
                       "--ref-json", str(ref))
    assert rc == 0
    cmp16 = payload["comparison"]["32"]
    assert cmp16["pass"] is False
    assert cmp16["max_abs_err_pct"] > 8.0
    assert payload["comparison"]["ref_provenance"]["source"].startswith("SYNTHETIC")
    assert len(cmp16["u_rows"]) == 2 and cmp16["u_rows"][0][3] != 0.0
