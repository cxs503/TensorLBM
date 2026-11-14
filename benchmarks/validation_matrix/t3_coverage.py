"""T3 nine-case coverage gate (GAP-LBM-T3-BENCH).

The validation matrix judges *cases*.  The T3 acceptance is a statement about
*nine physics lineages*, and the two are not the same thing: a passing laminar
channel case does not verify the turbulent-channel lineage, and a resistance
proxy does not verify the "resistance + wave" lineage.  This module makes that
distinction machine-checkable:

* ``t3_slots.json`` declares, per slot, the physics class and the components a
  passing case must exercise, plus per-case attributes (physics class,
  components, level basis, credit caveats).
* ``coverage()`` joins the fresh matrix verdicts with that contract and returns
  a per-slot state (``verified`` / ``partial`` / ``gap``) with the evidence that
  produced it and the explicit reason a substitute does not count.
* ``T3_COVERAGE.json`` freezes the result; ``--check`` fails on drift, which is
  what keeps the headline honest when a case flips.

Nothing here re-judges physics: case statuses come from ``runner.evaluate_case``
(the same code CI runs).  A failing slot stays a gap; there is no path from this
module to a pass.

Usage::

    python benchmarks/validation_matrix/t3_coverage.py           # print + write
    python benchmarks/validation_matrix/t3_coverage.py --check   # drift gate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from runner import evaluate_case, load_matrix  # noqa: E402

SLOTS_FILE = HERE / "t3_slots.json"
COVERAGE_FILE = HERE / "T3_COVERAGE.json"


def load_contract() -> dict:
    return json.loads(SLOTS_FILE.read_text(encoding="utf-8"))


def _ladder() -> dict:
    """Measured level ladder for the turbulent-channel slot, read from evidence."""
    case_dir = REPO_ROOT / "benchmarks" / "pending" / "channel_turb_3d"
    report = json.loads((case_dir / "validation_report.json").read_text(encoding="utf-8"))
    gate = json.loads((case_dir / "gate_reproduction.json").read_text(encoding="utf-8"))
    return {
        "source": "benchmarks/pending/channel_turb_3d/validation_report.json",
        "levels": [
            {
                "level": g["level"],
                "grid": g["grid"],
                "u_tau_err_pct": g["u_tau_err_pct"],
                "rms_y30_100_pct": g["rms_y30_100_pct"],
                "eddy_turnovers": g["eddy_turnovers"],
            }
            for g in report["grid_levels"]
        ],
        "gate_reproduction": {
            "path": "benchmarks/pending/channel_turb_3d/gate_reproduction.json",
            "recorded_rms_pct": gate["reproduction"]["recorded_rms_pct"],
            "recomputed_rms_pct": gate["reproduction"]["recomputed_rms_pct"],
            "delta_pp": gate["reproduction"]["delta_pp"],
            "measured_u_tau_rms_pct": gate["uplus_conventions"]["rel_offset0.5_measured_u_tau"]["rms_pct"],
        },
    }


def coverage() -> dict:
    matrix = load_matrix()
    contract = load_contract()
    attrs = contract["case_attributes"]
    verdicts = {c["id"]: evaluate_case(c) for c in matrix["cases"]}

    unknown = sorted(set(verdicts) - set(attrs))
    if unknown:
        raise SystemExit(f"t3_slots.json has no attributes for matrix cases: {unknown}")
    extra = sorted(set(attrs) - set(verdicts))
    if extra:
        raise SystemExit(f"t3_slots.json names cases absent from the matrix: {extra}")

    slot_rows = []
    for slot in contract["slots"]:
        name = slot["slot"]
        requires = slot["requires"]
        components = {}
        for comp in requires["components"]:
            qualifying = []
            proxies = []
            for cid, v in verdicts.items():
                a = attrs[cid]
                if v.get("slot") != name or comp not in a.get("components", []):
                    continue
                if v.get("status") != "passed":
                    continue
                if a.get("physics_class") not in requires["physics_class"]:
                    continue
                row = {
                    "case": cid,
                    "physics_class": a.get("physics_class"),
                    "level_basis": a.get("level_basis"),
                }
                if a.get("credit_caveats"):
                    row["credit_caveats"] = a["credit_caveats"]
                    proxies.append(row)
                else:
                    qualifying.append(row)
            if qualifying:
                components[comp] = {"state": "verified", "evidence": qualifying}
            elif proxies:
                components[comp] = {"state": "proxy", "evidence": proxies}
            else:
                components[comp] = {"state": "missing", "evidence": []}

        states = [c["state"] for c in components.values()]
        if all(s == "verified" for s in states):
            state = "verified"
        elif any(s in ("verified", "proxy") for s in states):
            state = "partial"
        else:
            state = "gap"

        substitutes = []
        blockers = []
        for cid, v in verdicts.items():
            a = attrs[cid]
            if v.get("slot") != name:
                continue
            if v.get("status") == "passed" and a.get("physics_class") not in requires["physics_class"]:
                substitutes.append({
                    "case": cid,
                    "physics_class": a.get("physics_class"),
                    "reason": "passing case does not exercise the slot's declared physics class; "
                              "recorded as substitute evidence only",
                })
            elif v.get("status") in ("failed", "pending"):
                blockers.append({"case": cid, "status": v.get("status"),
                                 "reason": (v.get("reason") or "")[:400]})

        row = {
            "slot": name,
            "lineage": slot["lineage"],
            "state": state,
            "requires": requires,
            "components": components,
            "substitute_evidence": substitutes,
            "blockers": blockers,
        }
        if name == "channel":
            row["measured_ladder"] = _ladder()
        slot_rows.append(row)

    headline = {s: sum(1 for r in slot_rows if r["state"] == s) for s in ("verified", "partial", "gap")}
    return {
        "schema": "tensorlbm-t3-coverage-v1",
        "contract": {"path": str(SLOTS_FILE.relative_to(REPO_ROOT)),
                     "version": contract["version"]},
        "matrix_version": matrix["version"],
        "headline": {
            **headline,
            "slots": len(slot_rows),
            "statement": (
                f"{headline['verified']}/{len(slot_rows)} T3 lineages are verified by a passing case "
                "whose physics matches the lineage and whose evidence carries the required components; "
                f"{headline['partial']} is partial and {headline['gap']} is a gap. The matrix case count "
                "('passed N') is not this number and must not be reported as slot coverage."
            ),
        },
        "slots": slot_rows,
    }


def render(report: dict) -> str:
    lines = [
        "| # | slot | lineage | state | component credit | blocker / substitute |",
        "|---|------|---------|-------|------------------|----------------------|",
    ]
    for i, r in enumerate(report["slots"], 1):
        comp = "; ".join(f"{k}={v['state']}" for k, v in r["components"].items())
        notes = [f"{b['case']} {b['status']}" for b in r["blockers"]]
        notes += [f"{s['case']} passed-but-substitute ({s['physics_class']})" for s in r["substitute_evidence"]]
        lines.append(f"| {i} | {r['slot']} | {r['lineage']} | **{r['state']}** | {comp} | "
                     f"{'; '.join(notes) or '—'} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="drift gate against the frozen output")
    args = parser.parse_args(argv)
    report = coverage()
    if args.check:
        if not COVERAGE_FILE.exists():
            print(f"MISSING {COVERAGE_FILE.relative_to(REPO_ROOT)}; run without --check")
            return 1
        frozen = json.loads(COVERAGE_FILE.read_text(encoding="utf-8"))
        drift = [k for k in ("headline", "slots", "matrix_version") if frozen.get(k) != report.get(k)]
        print(f"{'DRIFT' if drift else 'CLEAN'} {COVERAGE_FILE.relative_to(REPO_ROOT)} drift={drift}")
        return 1 if drift else 0
    COVERAGE_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(report["headline"]["statement"])
    print(render(report))
    print(f"wrote {COVERAGE_FILE.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())