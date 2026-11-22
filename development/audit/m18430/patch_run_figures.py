"""Minimal remediation for the m18339 xval entry (audit m18430).

Defect (reproduced 2026-11-13): ``run.py::main`` calls ``make_figures`` with the
raw *result* dicts (``r["result"]``) while ``make_figures`` also indexes the
judge columns (``l2_rel_pct``), which live in ``r["metrics"]``.  The full run
therefore raises ``KeyError: 'l2_rel_pct'`` in the figure stage, i.e. *after*
the sweep and *before* ``artifacts/results.json`` and the SUMMARY line are
written.  That is why the delivered artifact folder holds ``run.log`` /
``table.csv`` / ``oss_closure_check.json`` but no ``results.json`` and only
``fig1`` of the three figures promised by REPORT.md \u00a72(c).

Fix: merge the judge columns into the dict handed to ``make_figures`` (one line).
Usage: python scripts/audit_m18430/patch_run_figures.py <xval-dir-copy>
"""
from __future__ import annotations

import pathlib
import sys

OLD = 'figs = make_figures([r["result"] | {"closure": r["result"]["closure"]} for r in results],'
NEW = 'figs = make_figures([r["result"] | r["metrics"] | {"closure": r["result"]["closure"]} for r in results],'


def main(argv: list[str] | None = None) -> int:
    args = (argv or sys.argv[1:]) or []
    if len(args) != 1:
        print(__doc__)
        return 2
    path = pathlib.Path(args[0]) / "run.py"
    src = path.read_text()
    if NEW in src:
        print("already patched:", path)
        return 0
    if OLD not in src:
        print("call site not found in", path)
        return 1
    path.write_text(src.replace(OLD, NEW))
    print("patched:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())