"""Report <-> artifact integrity guard for the SC-Laplace cross-validation.

``xval/XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION/REPORT.md`` cites four
figures and a long list of numbers.  Nothing in the earlier test set
(``tests/test_xval_sc_laplace.py``) checked that those claims are backed by
files that actually exist and agree with ``artifacts/result.json`` — the report
shipped while the four cited PNGs were missing from the tree.

The tests here are cheap (no lattice solver is run) and lock the contract:

1. every figure cited in the report exists, is a real PNG and is listed in
   ``artifacts/figures/PROVENANCE.json`` with a matching sha256, so a figure
   cannot go stale with respect to ``artifacts/result.json``;
2. no figure file is orphaned (present on disk but not cited in the report);
3. the report renders deterministically from ``artifacts/result.json`` through
   ``make_report.py`` — key numbers (``sigma``, ``R2``, per-radius ``dp``, the
   Maxwell coexistence values) are injected, never hand-copied;
4. the field snapshot shipped next to the figures is the R = 20 row of the
   result file;
5. the verdict block answers the <3 % / <15 % question and its deviation
   arithmetic reproduces from ``sigma`` and the locked library value.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
XVAL = REPO / "xval" / "XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION"
ART = XVAL / "artifacts"
FIG = ART / "figures"
REPORT_MD = XVAL / "REPORT.md"

EXPECTED_FIGURES = {
    "fig1_droplet_field.png",
    "fig2_laplace_fit.png",
    "fig3_sigma_vs_R.png",
    "fig4_sensitivity.png",
}


def _result() -> dict:
    return json.loads((ART / "result.json").read_text())


def _report() -> str:
    return REPORT_MD.read_text(encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _png_size(path: Path) -> tuple[int, int]:
    """Width/height straight out of the IHDR chunk (no Pillow dependency)."""
    head = path.read_bytes()[:33]
    assert head[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} lacks the PNG signature"
    assert head[12:16] == b"IHDR", f"{path.name} has no IHDR chunk"
    return struct.unpack(">II", head[16:24])


def _cited_figures(report: str) -> list[str]:
    return re.findall(r"`(artifacts/figures/[A-Za-z0-9_.\-]+\.png)`", report)


def test_report_cites_the_four_figures():
    cited = _cited_figures(_report())
    assert len(cited) == len(set(cited)), f"duplicate figure citations: {cited}"
    assert set(Path(c).name for c in cited) == EXPECTED_FIGURES
    assert (XVAL / "REPORT.md").is_file()


def test_cited_figures_exist_and_are_readable_pngs():
    for rel in _cited_figures(_report()):
        path = XVAL / rel
        assert path.is_file(), f"report cites a missing figure: {rel}"
        w, h = _png_size(path)
        assert w >= 600 and h >= 300, f"{rel} is suspiciously small: {w}x{h}"
        assert path.stat().st_size > 20_000, f"{rel} looks truncated ({path.stat().st_size} B)"


def test_no_orphan_figures_on_disk():
    on_disk = {p.name for p in FIG.glob("*.png")}
    assert on_disk == EXPECTED_FIGURES, f"figures on disk vs report: {sorted(on_disk)}"


def test_figure_provenance_matches_the_committed_bytes():
    prov = json.loads((FIG / "PROVENANCE.json").read_text())
    # the figures must have been rendered from *this* result file
    assert prov["result_json"]["sha256"] == _sha256(ART / "result.json")
    assert prov["field_snapshot"]["sha256"] == _sha256(ART / "field_snapshot_R20.json")
    assert set(prov["figures"]) == EXPECTED_FIGURES
    for name, meta in prov["figures"].items():
        path = FIG / name
        assert meta["sha256"] == _sha256(path), f"{name} changed since it was stamped"
        assert meta["bytes"] == path.stat().st_size


def test_field_snapshot_is_the_r20_row():
    snap = json.loads((ART / "field_snapshot_R20.json").read_text())
    rows = _result()["laplace"]["tensorlbm"]["rows"]
    row = next(q for q in rows if float(q["R_init"]) == 20.0)
    for key in ("R_eq", "dp", "p_in", "p_out", "rho_in", "rho_out", "u_max"):
        assert snap[key] == pytest.approx(row[key], rel=1e-9, abs=0.0), f"{key} drifted"


def test_report_numbers_are_injected_from_result_json():
    rep = _report()
    res = _result()
    for solver, fit in res["laplace"].items():
        assert f"{fit['sigma_through_origin']:.5f}" in rep, (solver, "sigma_through_origin")
        assert f"{fit['sigma_free_intercept']:.5f}" in rep, (solver, "sigma_free_intercept")
        assert f"{fit['R2']:.8f}" in rep, (solver, "R2")
        for point in fit["per_radius"]:
            assert f"{point['dp']:.6f}" in rep, (solver, point["R_eq"], "dp")
            assert f"{point['sigma_i']:.5f}" in rep, (solver, point["R_eq"], "sigma_i")
    mx = res["maxwell"]
    for key in ("rho_v", "rho_l"):
        assert f"{mx[key]:.5f}" in rep, key
    assert f"{mx['p_c']:.6f}" in rep
    # the implementation-vs-implementation spread quoted in §3.3
    assert f"{res['impl_vs_impl']['max_dp_rel_pct']:.3f}%" in rep


def test_report_regenerates_identically_from_result_json(tmp_path):
    """``make_report.py`` must be the only author of REPORT.md's numbers."""
    staged = tmp_path / "xval"
    (staged / "artifacts").mkdir(parents=True)
    (staged / "make_report.py").write_bytes((XVAL / "make_report.py").read_bytes())
    (staged / "artifacts" / "result.json").write_bytes((ART / "result.json").read_bytes())

    proc = subprocess.run(
        [sys.executable, "make_report.py"], cwd=staged, capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    rendered = (staged / "REPORT.md").read_text(encoding="utf-8")
    assert rendered == _report(), (
        "REPORT.md is not the render of artifacts/result.json — "
        "run `python make_report.py` after touching the result file"
    )


def test_verdict_answers_the_three_and_fifteen_percent_question():
    res = _result()
    locked = res["config"]["sigma_locked_doc"]
    assert res["pass_all_15pct"] is True
    for solver, v in res["verdict"].items():
        assert v["pass_3pct"] is True, solver
        assert v["pass_15pct"] is True, solver
        # the quoted deviation must reproduce from sigma and the locked value
        recomputed = abs(v["sigma"] - locked) / locked * 100.0
        assert v["dev_vs_locked_doc_pct"] == pytest.approx(recomputed, rel=1e-9), solver
        assert v["dev_vs_locked_doc_pct"] < 3.0, solver
        assert v["max_dev_pct"] < 3.0, solver
        assert v["laplace_linear_R2"] >= 0.999, solver
    rep = _report()
    assert "<3%" in rep and "<15%" in rep
    assert "## ⑥ 误差判定" in rep