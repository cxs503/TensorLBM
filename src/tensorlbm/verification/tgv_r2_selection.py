"""Verification suite: Taylor-Green anchor for the R2 BPTT selection pipeline.

This is the *verification-side caller* of the selection capability.  Solvers do
not import it; acceptance runs do.  It wraps
:func:`tensorlbm.les_graph_unroll.tgv_anchor.run_tgv_anchor` in the
``passed``/``rows`` contract used by the rest of ``tensorlbm.verification`` and
adds nothing physical of its own — every number comes from the anchor module
(``src/tensorlbm/les_graph_unroll/tgv_anchor.py``) and the selection layer
(``src/tensorlbm/les_graph_unroll/selection.py``).

Run the heavyweight XVAL reproduction with
``python scripts/xval_r2_selection_bptt_tgv.py all``; run the importable,
platform-sized acceptance check with::

    python -m tensorlbm.verification.tgv_r2_selection --quick --out acceptance.json

The acceptance contract (owner 2026 precision standard):

* every ``ref``/``got`` row in ``report["benchmark"]["rows"]`` is < 15 %;
* the fine-grid reference is itself anchored to the exact TGV solution, so a
  broken reference fails the suite instead of silently becoming the truth.
"""

from __future__ import annotations

import argparse
import json

from ..les_graph_unroll.tgv_anchor import AnchorConfig, run_tgv_anchor

__all__ = ["SUITE", "run_tgv_anchor_acceptance", "main"]

SUITE = "tgv-r2-selection-bptt"


def run_tgv_anchor_acceptance(*, quick: bool = False, convergence: bool = False) -> dict:
    """Run the TGV anchor and return the acceptance record."""
    cfg = AnchorConfig.quick() if quick else AnchorConfig()
    report = run_tgv_anchor(cfg, convergence=convergence)
    rows = report["benchmark"]["rows"]
    return {
        "suite": SUITE,
        "anchor": report["anchor"],
        "threshold": report["benchmark"]["threshold"],
        "passed": bool(report["benchmark"]["all_pass"]),
        "rows": rows,
        "failed": [r["id"] for r in rows if not r["pass"]],
        "repro_cmd": report["repro_cmd"],
        "wall_seconds": report["wall_seconds"],
        "report": report,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="TGV anchor acceptance for the R2 selection pipeline")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--convergence", action="store_true")
    ap.add_argument("--out", default="artifacts/xval_r2_selection_bptt_tgv/acceptance.json")
    a = ap.parse_args(argv)
    rec = run_tgv_anchor_acceptance(quick=a.quick, convergence=a.convergence)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True)
    print(json.dumps({"suite": rec["suite"], "passed": rec["passed"],
                      "failed": rec["failed"], "rows": len(rec["rows"])}, indent=2))
    return 0 if rec["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())