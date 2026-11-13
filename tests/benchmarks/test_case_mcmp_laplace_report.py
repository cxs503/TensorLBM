"""Integrity gate for the CASE-MCMP-LAPLACE report bundle.

The report (``benchmarks/case_pool/mcmp_laplace/README.md`` + the canonical copy
``docs/CASE-MCMP-LAPLACE.md``) is *generated* from ``artifacts/summary.json`` by
``benchmarks/case_pool/mcmp_laplace/make_report.py``, so it is supposed to be
figure-complete at all times.  On main it was not: the four pngs under the case
``figures/`` directory had gone missing while both markdown files still linked
them (the same "dead link after merge" defect that was fixed once before), and
the ``docs/`` copy used ``figures/...`` links that resolve relative to ``docs/``,
i.e. dead even when the pngs exist.

Predeclared gates (fixed before running; no threshold tuning afterwards):

* **G1 link resolution** - every image link in ``README.md`` resolves relative to
  the case directory, every image link in ``docs/CASE-MCMP-LAPLACE.md``
  resolves relative to ``docs/``.
* **G2 no hand-copy** - ``README.md`` is byte-identical to
  ``make_report.render()`` re-run on the committed ``summary.json`` (so no number
  in the report can be edited without breaking the suite).
* **G3 doc == README** - the canonical doc is byte-identical to
  ``make_report.render_doc(README)`` (same content, doc-scoped link prefix).
* **G4 figures shipped** - each expected png exists, is a real PNG (magic bytes,
  > 5 kB of pixels) and is **not** git-ignored, which is the failure mode that
  silently drops generated artefacts from the next commit.

The suite is text/artefact-only: it never runs the solver, and
``make_report.render`` does not import matplotlib (plotting is confined to
``make_figures``), so a headless CI box can run it.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CASE = REPO / "benchmarks" / "case_pool" / "mcmp_laplace"
DOC = REPO / "docs" / "CASE-MCMP-LAPLACE.md"
MAKE_REPORT = CASE / "make_report.py"

#: Figure order produced by ``make_report.make_figures``; the report renders the
#: legend/captions in this order, so the list is part of the contract.
FIG_ORDER = ("laplace_law.png", "sigma_vs_radius.png", "droplet_field.png", "routeB_flatness.png")

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
MIN_PNG_BYTES = 5_000

IMAGE_LINK = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def _load_make_report():
    spec = importlib.util.spec_from_file_location("_mcmp_laplace_make_report", MAKE_REPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _summary() -> dict:
    path = CASE / "artifacts" / "summary.json"
    if not path.exists():
        pytest.skip(f"missing artefact {path}")
    return json.loads(path.read_text())


def _image_links(md: str) -> list[str]:
    return IMAGE_LINK.findall(md)


# ---------------------------------------------------------------------------
# G1: link resolution
# ---------------------------------------------------------------------------


def test_readme_image_links_resolve_in_case_dir():
    md = (CASE / "README.md").read_text()
    links = _image_links(md)
    assert links, "report has a '## 图' section; expected at least one image link"
    missing = [l for l in links if not (CASE / l).is_file()]
    assert not missing, f"dead image link(s) in {CASE / 'README.md'}: {missing}"


def test_doc_image_links_resolve_relative_to_docs():
    md = DOC.read_text()
    links = _image_links(md)
    assert links, "canonical doc must embed the same figures as the README"
    missing = [l for l in links if not (DOC.parent / l).is_file()]
    assert not missing, f"dead image link(s) in {DOC}: {missing}"


def test_doc_and_readme_link_the_same_figure_set():
    readme = {Path(l).name for l in _image_links((CASE / "README.md").read_text())}
    doc = {Path(l).name for l in _image_links(DOC.read_text())}
    assert readme == doc == set(FIG_ORDER)


# ---------------------------------------------------------------------------
# G2/G3: no hand-copy drift
# ---------------------------------------------------------------------------


def test_readme_is_exactly_the_rendered_summary():
    mod = _load_make_report()
    figs = [CASE / "figures" / name for name in FIG_ORDER]
    rendered = mod.render(_summary(), figs)
    assert (CASE / "README.md").read_text() == rendered


def test_doc_is_the_rendered_readme_with_doc_scoped_links():
    mod = _load_make_report()
    figs = [CASE / "figures" / name for name in FIG_ORDER]
    rendered_doc = mod.render_doc(mod.render(_summary(), figs))
    assert DOC.read_text() == rendered_doc


# ---------------------------------------------------------------------------
# G4: the figures are actually shipped (present, real, not ignored)
# ---------------------------------------------------------------------------


def test_case_figures_directory_holds_exactly_the_rendered_figures():
    on_disk = sorted(p.name for p in (CASE / "figures").glob("*.png"))
    assert on_disk == sorted(FIG_ORDER)


@pytest.mark.parametrize("name", FIG_ORDER)
def test_figure_is_a_real_png(name):
    path = CASE / "figures" / name
    assert path.is_file(), f"missing figure {path}"
    raw = path.read_bytes()
    assert raw.startswith(PNG_MAGIC), f"{name} is not a PNG"
    assert len(raw) >= MIN_PNG_BYTES, f"{name} is suspiciously small ({len(raw)} B)"


@pytest.mark.parametrize("name", FIG_ORDER)
def test_figure_is_not_git_ignored(name):
    path = CASE / "figures" / name
    if not path.is_file():
        pytest.skip(f"{path} missing (covered by the real-png test)")
    proc = subprocess.run(
        ["git", "-C", str(REPO), "check-ignore", "-q", str(path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode not in (0, 1):
        pytest.skip(f"git check-ignore unavailable: {proc.stderr.strip()}")
    assert proc.returncode == 1, (
        f"{path} is git-ignored, so the figures will be dropped from the next commit "
        f"and the report links go dead again (git check-ignore said: {proc.stdout.strip()})"
    )


def test_make_report_writes_figures_next_to_the_report():
    mod = _load_make_report()
    assert mod.FIG == CASE / "figures"
    assert mod.REPO_ROOT == REPO