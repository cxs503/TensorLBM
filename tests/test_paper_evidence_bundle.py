#!/usr/bin/env python3
"""Gate for the paper-evidence bundle (GAP-DIFFLBM-PAPER-EVIDENCE).

The gate does not trust ``artifacts/paper_evidence/summary.json``: it re-reads the
raw item JSONs from ``runs/paper_evidence/`` and re-derives every verdict, then
cross-checks the summary against that derivation.  A hand-edited summary, a
missing resolution ladder, a row whose ``pass`` disagrees with its own
``err_pct``/``threshold_pct``, or fewer than 6 benchmarks all fail here.

Skips (not passes) when no run artifacts are present in the checkout.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs" / "paper_evidence"
ART = REPO / "artifacts" / "paper_evidence"

REQUIRED = {
    "analytic": 2,
    "literature": 3,
    "cross_solver": 1,
}


def _items() -> dict[str, dict]:
    out = {}
    for f in sorted(RUNS.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and "item" in d and "rows" in d:
            out[d["item"]] = d
    return out


@pytest.fixture(scope="module")
def items() -> dict[str, dict]:
    it = _items()
    if not it:
        pytest.skip(f"no paper-evidence run artifacts under {RUNS}")
    return it


def test_at_least_six_independent_benchmarks(items):
    cats: dict[str, int] = {}
    for it in items.values():
        cats[it["category"]] = cats.get(it["category"], 0) + 1
    assert len(items) >= 6, f"only {len(items)} benchmarks: {sorted(items)}"
    for cat, n in REQUIRED.items():
        assert cats.get(cat, 0) >= n, f"{cat}: {cats.get(cat, 0)} < required {n} (have {cats})"


def test_every_benchmark_has_a_reference_and_two_resolutions(items):
    for key, it in items.items():
        anchor = it.get("anchor") or {}
        assert anchor.get("citation"), f"{key}: no external anchor citation"
        st = it.get("settings") or {}
        ladders = []
        for name in ("resolutions", "cavity_sizes"):
            v = st.get(name)
            if isinstance(v, (list, tuple)):
                ladders.append(len(v))
            elif isinstance(v, dict):
                ladders.extend(len(x) for x in v.values())
        assert ladders, f"{key}: no resolution ladder recorded in settings"
        assert max(ladders) >= 2, f"{key}: needs >=2 resolutions/parameter points, got {ladders}"


def test_row_verdicts_are_self_consistent(items):
    for key, it in items.items():
        assert it["rows"], f"{key}: empty error table"
        for r in it["rows"]:
            assert set(("ref", "got", "err_pct", "threshold_pct", "pass")) <= set(r), \
                f"{key}/{r.get('quantity')}: row missing ref/got/err/threshold/pass"
            assert isinstance(r["err_pct"], (int, float)) and math.isfinite(r["err_pct"])
            assert r["err_pct"] >= 0.0, f"{key}/{r['quantity']}: negative error"
            expected = r["err_pct"] < r["threshold_pct"]
            assert bool(r["pass"]) == expected, (
                f"{key}/{r['quantity']}: pass={r['pass']} contradicts "
                f"err={r['err_pct']:.4g}% vs threshold={r['threshold_pct']}%"
            )


def test_error_is_consistent_with_ref_and_got(items):
    """err_pct must be re-derivable from ref/got (no free-floating numbers)."""
    for key, it in items.items():
        for r in it["rows"]:
            ref, got, err = r["ref"], r["got"], r["err_pct"]
            if not isinstance(ref, (int, float)) or not isinstance(got, (int, float)):
                continue
            if abs(ref) < 1e-12:
                # ref == 0 means "err_pct is already the deviation", only boundedness
                assert err >= 0.0
                continue
            derived = abs(got - ref) / abs(ref) * 100.0
            assert abs(derived - err) < max(0.05 * max(err, 1e-6), 1e-3), (
                f"{key}/{r['quantity']}: err_pct={err:.6g} but |got-ref|/ref={derived:.6g}"
            )


def test_summary_matches_rederivation(items):
    sp = ART / "summary.json"
    if not sp.exists():
        pytest.skip("summary.json not generated yet")
    summary = json.loads(sp.read_text())
    assert summary["n_items"] == len(items)
    by_key = {r["item"]: r for r in summary["items"]}
    assert set(by_key) == set(items)
    for key, it in items.items():
        rows3 = [r for r in it["rows"] if abs(r["threshold_pct"] - 3.0) < 1e-9]
        recomputed_ok = all(r["err_pct"] < 3.0 for r in rows3) if rows3 else None
        assert by_key[key]["all_under_3pct"] == recomputed_ok, (
            f"{key}: summary claims all_under_3pct={by_key[key]['all_under_3pct']}, "
            f"raw rows give {recomputed_ok}"
        )
        assert by_key[key]["n_rows"] == len(it["rows"])


def test_docs_and_figures_exist(items):
    doc = REPO / "docs" / "PAPER_EVIDENCE_BUNDLE.md"
    if not (ART / "summary.json").exists():
        pytest.skip("aggregate step not run yet")
    assert doc.exists(), "docs/PAPER_EVIDENCE_BUNDLE.md missing (aggregate.py writes it)"
    text = doc.read_text()
    assert "<!--TABLES-->" not in text, "table placeholder was never substituted"
    for key in items:
        assert key in text, f"{key} not mentioned in the generated document"