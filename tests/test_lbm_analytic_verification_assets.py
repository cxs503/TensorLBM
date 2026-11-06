"""Drift gate for the published analytic-verification ledger (r1).

``docs/lbm_analytic_verification_r1.md`` publishes (a) a numeric error ledger,
(b) a figure and (c) two negative controls.  All three are produced by
``scripts/verify_lbm_analytic_convergence.py``.  Until now nothing checked that
the published copies still agree with each other or that they are even present
in the tree, so three silent failure modes were possible:

* the doc linked ``artifacts/lbm_analytic_verification/convergence.png`` while
  that file was **not committed** — the figure renders broken on GitHub;
* the doc was not listed in ``docs/README.md`` — unreachable from the index;
* the numbers typed into the doc table could drift from the regenerated
  ledger without any test turning red.

This module makes those three failures loud.  It is cheap (a few file reads,
no torch run); re-measuring is opt-in via ``TENSORLBM_HEAVY_TESTS=1`` because a
full provenance run costs ~3 min on one core.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "lbm_analytic_verification_r1.md"
INDEX = ROOT / "docs" / "README.md"
ARTIFACTS = ROOT / "artifacts" / "lbm_analytic_verification"
LEDGER = ARTIFACTS / "convergence.json"
FIGURE = ARTIFACTS / "convergence.png"
TABLES = ARTIFACTS / "tables.md"

pytestmark = [pytest.mark.physics]

# Markdown link / image target: `![alt](target)` or `[text](target)`.
_LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
# Leading number of a table cell, e.g. `3.378e-01 %` -> 0.3378.
_NUM_RE = re.compile(r"^(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def _ledger() -> dict:
    assert LEDGER.is_file(), f"missing published ledger: {LEDGER}"
    return json.loads(LEDGER.read_text(encoding="utf-8"))


def _doc_text() -> str:
    assert DOC.is_file(), f"missing doc: {DOC}"
    return DOC.read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    """Body of the ``heading`` section, up to the next level-2 heading."""
    out: list[str] = []
    inside = False
    for line in text.splitlines():
        if line.startswith(heading):
            inside = True
            continue
        if inside and line.startswith("## "):
            break
        if inside:
            out.append(line)
    assert out, f"section {heading!r} not found"
    return "\n".join(out)


def _tables(section: str) -> list[list[list[str]]]:
    """Contiguous markdown tables, each as a list of rows of cells."""
    tables: list[list[list[str]]] = []
    block: list[list[str]] = []
    for line in [*section.splitlines(), ""]:
        if line.startswith("|"):
            block.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif block:
            tables.append(block)
            block = []
    assert tables, "no markdown tables in section"
    return tables


def _table_with(tables: list[list[list[str]]], needle: str) -> list[list[str]]:
    """The table whose header row contains ``needle``."""
    for table in tables:
        if needle in " ".join(table[0]):
            return table
    raise AssertionError(f"no table whose header mentions {needle!r}")


def _cell_number(cell: str) -> float:
    """Pull the leading number out of a table cell (``3.378e-01 %`` -> 0.3378)."""
    m = _NUM_RE.match(cell.strip().replace("−", "-").strip("*"))
    assert m, f"cell is not numeric: {cell!r}"
    return float(m.group(1))


def _row_key(row: list[str]) -> str:
    """First token of the first cell: ``A `nx=64...``` -> ``A``, ``16`` -> ``16``."""
    return row[0].split()[0].strip("`*")


def _tracked(rel: str) -> bool:
    """True if ``rel`` is tracked by git (skip when git is unavailable)."""
    try:
        proc = subprocess.run(
            ["git", "ls-files", "--error-unmatch", rel],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git
        pytest.skip("git not available to check tracking status")
    return proc.returncode == 0 and proc.stdout.strip() != ""


def _rows_by_key(table: list[list[str]]) -> dict[str, list[str]]:
    """Data rows (header and separator dropped) keyed by :func:`_row_key`."""
    rows = {}
    for row in table:
        key = _row_key(row)
        if key in {"case", "16"} and len(row) < 4:  # header / separator artefacts
            continue
        if set(row[0]) <= set("-: "):  # separator row
            continue
        rows.setdefault(key, row)
    return rows


def test_doc_links_point_at_committed_files() -> None:
    """Every file the doc links to exists *and* is committed.

    A doc that links an uncommitted artifact path renders a broken image on
    GitHub, which is exactly how the r1 figure was lost before this gate.
    """
    targets = [
        t for t in _LINK_RE.findall(_doc_text()) if not t.startswith(("http://", "https://", "#"))
    ]
    assert targets, "doc links nothing; the figure/provenance table vanished?"
    resolved = set()
    for target in targets:
        path = (DOC.parent / target).resolve()
        assert path.is_file(), f"doc links missing file: {target}"
        rel = os.path.relpath(path, ROOT)
        assert _tracked(rel), f"doc links uncommitted file: {rel}"
        resolved.add(rel.replace(os.sep, "/"))
    assert "artifacts/lbm_analytic_verification/convergence.png" in resolved


def test_figure_is_a_real_png() -> None:
    """The published figure must be a committed, non-trivial PNG."""
    assert FIGURE.is_file(), f"missing figure: {FIGURE}"
    blob = FIGURE.read_bytes()
    assert blob.startswith(b"\x89PNG\r\n\x1a\n"), "figure is not a PNG"
    assert len(blob) > 20_000, f"figure suspiciously small ({len(blob)} B) — placeholder?"
    assert _tracked(os.path.relpath(FIGURE, ROOT)), "figure is not committed"


def test_doc_is_indexed_in_docs_readme() -> None:
    """The doc must be reachable from the docs index."""
    index = INDEX.read_text(encoding="utf-8")
    assert "lbm_analytic_verification_r1.md" in index, "doc missing from docs/README.md index"


def test_ledger_satisfies_tolerance_and_controls_fire() -> None:
    """The published ledger must pass the same criteria the provenance script gates on."""
    data = _ledger()
    tol = float(data["rel_tol_pct"])

    canonical = data["canonical"]
    assert set(canonical) == {"A_shear_wave", "B_couette", "C_poiseuille", "D_friction"}
    for name, rec in canonical.items():
        assert rec["err_pct"] < tol, f"canonical {name} {rec['err_pct']:.3e}% >= {tol}%"

    # observed convergence orders: shear wave is 4th order, Poiseuille 2nd order
    assert all(3.5 < order < 4.5 for order in data["shear_order"]), data["shear_order"]
    assert all(1.5 < order < 2.5 for order in data["poiseuille_order"]), data["poiseuille_order"]

    # negative controls must be *detected* by the same tolerance
    controls = data["controls"]
    assert controls["tau_shift"]["deviation_pct"] > tol
    assert controls["no_bounce_back"]["couette_err_pct"] > tol


def test_doc_canonical_table_matches_ledger() -> None:
    """§3 canonical table numbers == ledger numbers (no hand-typed drift)."""
    data = _ledger()["canonical"]
    rows = _rows_by_key(_table_with(_tables(_section(_doc_text(), "## 3.")), "analytic quantity"))

    expectations = {
        "A": ("A_shear_wave", "nu_measured", "nu_exact", "err_pct"),
        "C": ("C_poiseuille", "u_max_num", "u_max_exact", "err_pct"),
        "D": ("D_friction", "cf_num", "cf_exact", "err_pct"),
    }
    for key, (name, num_key, ana_key, err_key) in expectations.items():
        cells = rows[key]
        rec = data[name]
        assert _cell_number(cells[2]) == pytest.approx(rec[num_key], rel=1e-6), (key, num_key)
        assert _cell_number(cells[3]) == pytest.approx(rec[ana_key], rel=1e-6), (key, ana_key)
        assert _cell_number(cells[4]) == pytest.approx(rec[err_key], rel=5e-3, abs=1e-12), (
            key,
            err_key,
        )

    # case B publishes only the profile error
    assert _cell_number(rows["B"][4]) == pytest.approx(
        data["B_couette"]["err_pct"], rel=5e-3, abs=1e-12
    )


def test_doc_shear_table_matches_ledger() -> None:
    """§3 grid-refinement table == ledger shear entries."""
    rows = _rows_by_key(_table_with(_tables(_section(_doc_text(), "## 3.")), "Delta x"))
    shear = {str(rec["nx"]): rec for rec in _ledger()["shear"]}
    assert set(shear) <= set(rows), (sorted(shear), sorted(rows))
    for nx, rec in shear.items():
        cells = rows[nx]
        assert _cell_number(cells[1]) == pytest.approx(rec["kdx"], rel=1e-3), nx
        assert _cell_number(cells[2]) == pytest.approx(rec["nu_measured"], rel=1e-8), nx
        assert _cell_number(cells[3]) == pytest.approx(rec["nu_exact"], rel=1e-8), nx
        assert _cell_number(cells[4]) == pytest.approx(rec["err_pct"], rel=1e-2, abs=1e-12), nx


def test_provenance_script_exists_and_writes_these_artifacts() -> None:
    """The ledger must be reproducible by a committed script, not hand-edited."""
    script = ROOT / "scripts" / "verify_lbm_analytic_convergence.py"
    assert script.is_file()
    src = script.read_text(encoding="utf-8")
    for name in ("convergence.json", "convergence.png", "tables.md"):
        assert name in src, f"provenance script no longer writes {name}"
    assert TABLES.is_file(), f"missing table dump: {TABLES}"
    assert _tracked(os.path.relpath(TABLES, ROOT)), "tables.md is not committed"


@pytest.mark.slow
def test_ledger_reproducible_by_published_script() -> None:
    """Opt-in: re-run the provenance script and require exit code 0.

    ``TENSORLBM_HEAVY_TESTS=1 pytest tests/test_lbm_analytic_verification_assets.py -k reproducible``
    """
    if os.environ.get("TENSORLBM_HEAVY_TESTS") != "1":
        pytest.skip("set TENSORLBM_HEAVY_TESTS=1 to re-measure (~3 min, one core)")
    proc = subprocess.run(
        [sys.executable, "scripts/verify_lbm_analytic_convergence.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=3600,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
