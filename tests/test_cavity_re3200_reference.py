"""Contract tests for the admitted Ghia (1982) Re = 3200 centreline reference.

Why Re = 3200
-------------
The anchor ``REPRO-LBM-HOU1995`` (Hou, Zou, Chen & Doolen, *J. Comput. Phys.*
**118** (1995) 329-347) reports the lid-driven cavity up to Re = 3200, and the
library's own ``GHIA_RE100/400/1000`` tables stop at 1000.  The Re = 3200 column
of Ghia's Table I/II was rejected earlier (``build_reference.py`` ERRATA) because
one OCR cell lost its decimal point, so the column was never admitted and no
Re = 3200 scoring could happen at all.

``benchmarks/verified/cavity/re3200/reference/ghia_re3200_reference.json``
admits it under an explicit quarantine policy:
one cell (Table I, y = 0.4531) is bounded by its neighbours, dropped from the
scored rows and exported with its admissible interval.  These tests pin that:

* the paper's own machine OCR layer still witnesses every scored value;
* the quarantined cell is not silently scored, and its reconstruction is never
  used as if it were a measurement;
* the column is physically consistent with its Re = 1000 / 5000 neighbours;
* the runner can load a reference that declares any Re (it used to hard-code
  ``u_re5000`` keys, which made a Re = 3200 file unloadable).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BUILDER = REPO / "benchmarks" / "verified" / "cavity" / "re5000" / "build_reference.py"
REF3200 = (
    REPO
    / "benchmarks"
    / "verified"
    / "cavity"
    / "re3200"
    / "reference"
    / "ghia_re3200_reference.json"
)
RUNNER = REPO / "benchmarks" / "verified" / "cavity" / "re5000" / "run.py"


@pytest.fixture(scope="module")
def ref() -> dict:
    assert REF3200.is_file(), f"missing admitted reference: {REF3200}"
    return json.loads(REF3200.read_text())


# --------------------------------------------------------------------------- #
# 1. the builder gates are still green (i.e. the JSON is reproducible)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("re_col", ["3200", "5000"])
def test_builder_gates_pass(re_col: str) -> None:
    """``--check`` exits 0 only when every witness/bracket/library gate passes."""
    proc = subprocess.run(
        [sys.executable, str(BUILDER), "--check", "--re", re_col],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "gates" in proc.stdout


def test_committed_json_is_reproducible_in_content(ref: dict) -> None:
    """Re-running the builder must reproduce the committed payload hash."""
    out = REF3200.parent.parent / "out" / "_regen_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, str(BUILDER), "--re", "3200", "--out", str(out)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    regen = json.loads(out.read_text())
    out.unlink()
    # the builder hash covers the payload only (sort_keys=True), so provenance
    # strings that legitimately change (library-vs-copy errata lists) are excluded
    assert regen["sha256"] == ref["sha256"]
    assert regen["u_re3200"] == ref["u_re3200"]
    assert regen["v_re3200"] == ref["v_re3200"]


# --------------------------------------------------------------------------- #
# 2. shape and provenance of the admitted column
# --------------------------------------------------------------------------- #
def test_rows_and_reynolds(ref: dict) -> None:
    assert float(ref["re"]) == 3200.0
    assert "u_re3200" in ref and "v_re3200" in ref
    # 17 Ghia grid points, one of which is quarantined -> 16 scored u rows
    assert len(ref["u_re3200"]) == 16
    assert len(ref["v_re3200"]) == 17
    assert ref["provenance"]["gates"], "gate log must be present"
    assert all(ref["provenance"]["gates"].values()), ref["provenance"]["gates"]


def test_endpoints_and_bounds(ref: dict) -> None:
    u = dict(map(tuple, ref["u_re3200"]))
    v = dict(map(tuple, ref["v_re3200"]))
    assert u[1.0] == pytest.approx(1.0)  # lid
    assert u[0.0] == pytest.approx(0.0)  # bottom wall
    assert v[0.0] == pytest.approx(0.0)  # left wall
    assert v[1.0] == pytest.approx(0.0)  # right wall
    assert max(abs(x) for x in u.values()) <= 1.0


# --------------------------------------------------------------------------- #
# 3. the quarantine is honest: not scored, and bounded
# --------------------------------------------------------------------------- #
def test_quarantined_cell_is_not_scored(ref: dict) -> None:
    quar = ref["provenance"]["quarantined"]
    assert len(quar) == 1
    cell = quar[0]
    assert cell["table"] == "I" and float(cell["re"]) == 3200.0
    assert cell["coord"] == pytest.approx(0.4531)
    assert cell["scored"] is False
    assert 0.4531 not in [c for c, _ in ref["u_re3200"]]


def test_quarantined_cell_lies_outside_its_bracket(ref: dict) -> None:
    """The dropped value must actually be the corrupt one -- not a convenience cut."""
    cell = ref["provenance"]["quarantined"][0]
    lo, hi = cell["admissible_interval"]
    assert lo < hi
    as_transcribed = ref["provenance"]["quarantine_check"]["I@0.4531"]["as_transcribed"]
    assert as_transcribed == pytest.approx(-0.86636)
    assert not (lo <= as_transcribed <= hi), "a bracketed value must not be quarantined"
    # the paper's neighbouring rows really do bound it
    u = dict(map(tuple, ref["u_re3200"]))
    assert lo == pytest.approx(u[0.2813])  # lower end: the y = 0.2813 row
    assert hi == pytest.approx(u[0.5])  # upper end: the y = 0.5 row
    assert lo <= cell["reconstructed_as"] <= hi


# --------------------------------------------------------------------------- #
# 4. cross-Re consistency with the neighbouring admitted columns
# --------------------------------------------------------------------------- #
def test_column_sits_between_re1000_and_re5000(ref: dict) -> None:
    """Profiles may overshoot between sparse tabulated extrema, but not by much."""
    slack = 0.05  # 5 % of the lid speed
    dropped = {0.4531}
    for tab in ("u_columns", "v_columns"):
        cols = ref[tab]
        coords = ref["coords"]["u_y"] if tab == "u_columns" else ref["coords"]["v_x"]
        for coord, low, mid, high in zip(coords, cols["1000"], cols["3200"], cols["5000"]):
            if coord in dropped:
                continue
            lo, hi = sorted((low, high))
            assert lo - slack <= mid <= hi + slack, (tab, coord, low, mid, high)


# --------------------------------------------------------------------------- #
# 5. the runner loads any declared Re (regression: hard-coded u_re5000 keys)
# --------------------------------------------------------------------------- #
def _load_runner():
    import importlib.util

    spec = importlib.util.spec_from_file_location("cavity_run_mod", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_runner_loads_re3200_reference(ref: dict) -> None:
    pytest.importorskip("torch")
    mod = _load_runner()
    y, u, x, v, prov = mod.load_reference(str(REF3200))
    assert (len(y), len(u), len(x), len(v)) == (16, 16, 17, 17)
    assert prov["re"] == 3200.0
    assert prov["scored_rows_note"] == ref["provenance"]["quarantined"]


def test_runner_rejects_mislabelled_reference(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    mod = _load_runner()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"re": 3200.0, "u_re5000": [[0.0, 0.0]], "v_re5000": [[0.0, 0.0]]}))
    with pytest.raises(ValueError, match="u_re3200"):
        mod.load_reference(str(bad))
