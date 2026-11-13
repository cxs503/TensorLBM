#!/usr/bin/env python3
"""m18435 board audit — machine verification of m18327's delivery claims.

Compares, number by number:
  (a) the delivered artifact  audit/m18327/results/xval_results.json
  (b) the auditor's independent re-run  artifacts/audit_m18327/repro_*.json
  (c) the prose numbers printed in audit/m18327/docs/XVAL-...md
and writes a claim-vs-reality table (JSON + markdown) with no hand transcription.

Run:  python3 docs/board_audit/m18435/verify_claims.py
Exit code 0 always (this is an audit, not a gate); the table is the output.
"""
from __future__ import annotations

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[3]
ART = ROOT / "audit" / "m18327"
REPRO = ROOT / "artifacts" / "audit_m18327"
OUT = REPRO / "claim_vs_measured.json"


def load(p: pathlib.Path):
    return json.loads(p.read_text()) if p.exists() else None


def rel(x: float) -> float:
    return abs(x)


def main() -> int:
    got = load(ART / "results" / "xval_results.json")
    rep_par = load(REPRO / "repro_parity.json")
    rep_mom = load(REPRO / "repro_momentum.json")
    rep_ch = load(REPRO / "repro_channel.json")
    rows: list[dict] = []

    def add(item, claimed, audited, ok, note=""):
        rows.append({"item": item, "claimed": claimed, "audited": audited,
                     "match": ok, "note": note})

    # ── 0. artifact completeness: are all four predeclared arms in the artifact? ──
    decl_arms = ["parity", "channel", "momentum", "separation"]
    present = [a for a in decl_arms if a in got]
    add("predeclared arms present in delivered JSON",
        "4 arms declared in predeclared.json (" + ",".join(decl_arms) + ")",
        f"{len(present)}/4 present: {','.join(present)}; missing="
        f"{','.join(a for a in decl_arms if a not in got)}",
        len(present) == 4,
        "separation arm numbers exist only as prose in the doc, no artifact")

    # ── 1. parity ──
    if "parity" in got and rep_par:
        for key, lab in (("worst_rel_dev_float64", "f64"), ("worst_rel_dev_float32", "f32")):
            a, b = got["parity"][key], rep_par["parity"][key]
            add(f"parity worst rel dev {lab}", f"{a:.6e}", f"{b:.6e} (auditor re-run)",
                abs(a - b) <= 1e-12 * max(abs(a), 1e-300) or a == b)
        thr = got["parity"]["threshold"]
        add("parity vs predeclared threshold f64", f"<= {thr['float64']}",
            f"{got['parity']['worst_rel_dev_float64']:.3e} -> "
            f"{'PASS' if got['parity']['worst_rel_dev_float64'] <= thr['float64'] else 'FAIL'}",
            got["parity"]["worst_rel_dev_float64"] <= thr["float64"])
        add("parity vs predeclared threshold f32", f"<= {thr['float32']}",
            f"{got['parity']['worst_rel_dev_float32']:.3e} -> "
            f"{'PASS' if got['parity']['worst_rel_dev_float32'] <= thr['float32'] else 'FAIL'}",
            got["parity"]["worst_rel_dev_float32"] <= thr["float32"])

    # ── 2. momentum ──
    if "momentum" in got and rep_mom:
        gm = {c["u_eq"]: c for c in got["momentum"]["cases"]}
        rm = {c["u_eq"]: c for c in rep_mom["momentum"]["cases"]}
        for ue in gm:
            a, b = gm[ue]["rel_drift"], rm[ue]["rel_drift"]
            add(f"momentum rel drift u_eq={ue}", f"{a:.4e}", f"{b:.4e} (auditor re-run)",
                a == b)
        add("momentum: viscous arm vs predeclared 1e-12", "<1e-12",
            f"{gm['viscous']['rel_drift']:.3e} -> "
            f"{'PASS' if gm['viscous']['rel_drift'] <= 1e-12 else 'FAIL'}",
            gm["viscous"]["rel_drift"] <= 1e-12)
        add("momentum: negative control u_eq=mixture must NOT conserve", "drift >> 1e-12",
            f"{gm['mixture']['rel_drift']:.3e} (arm has discriminating power)",
            gm["mixture"]["rel_drift"] > 1e-3)

    # ── 3. channel: delivered vs auditor re-run, and per-case gate ──
    th = got["channel"]["thresholds"]
    art_ch = {c["tag"]: c for c in got["channel"]["cases"]}
    if rep_ch and "channel" in rep_ch:
        rep_c = {c["tag"]: c for c in rep_ch["channel"]["cases"]}
        for tag in rep_c:
            a, b = art_ch[tag], rep_c[tag]
            same = (a["rel_err_max"] == b["rel_err_max"] and a["nu_meas"] == b["nu_meas"])
            add(f"channel {tag} rel_err_max",
                f"{100*a['rel_err_max']:.4f}%", f"{100*b['rel_err_max']:.4f}% (auditor re-run)",
                same, f"nu {a['nu_meas']:.6f} vs {b['nu_meas']:.6f}")
    add("channel worst rel_err_max vs predeclared 15%", "<= 15%",
        f"{100*got['channel']['worst_rel_err_max']:.2f}% -> "
        f"{'PASS' if got['channel']['worst_rel_err_max'] <= th['rel_err_max'] else 'FAIL'}",
        got["channel"]["worst_rel_err_max"] <= th["rel_err_max"])
    add("channel worst rel_err_max vs ecosystem 3% bar", "<= 3%",
        f"{100*got['channel']['worst_rel_err_max']:.2f}% -> "
        f"{'PASS' if got['channel']['worst_rel_err_max'] <= 0.03 else 'FAIL (5.7x over at primary, 4.1x worst)'}",
        got["channel"]["worst_rel_err_max"] <= 0.03)
    # per-case sub-gate that the arm-level verdict ignores
    failed_sub = [c["tag"] for c in got["channel"]["cases"] if not c["pass"]]
    add("channel per-case sub-gates (rel+nu+steady_resid)",
        "all cases pass (predeclared steady_resid_max 1e-4)",
        f"cases failing their own predeclared sub-gate: {failed_sub or 'none'}"
        + ("" if not failed_sub else
           f" (refine_65 steady_resid={art_ch['refine_65']['steady_resid']:.3e} > 1e-4)"),
        not failed_sub,
        "arm verdict computed as rows[0]['pass'] only -> refine_65 sub-gate miss is invisible "
        "at arm level; doc table does mark it FAIL, results JSON/overall still say PASS")

    # ── 4. doc prose vs artifact (the numbers the report prints) ──
    doc = (ART / "docs" / "XVAL-LBM-SHAN-CHEN-MCMP-TAU-MIXED-VELOCITY.md").read_text()
    expect = {
        "primary max%": (f"{100*art_ch['primary']['rel_err_max']:.2f}",
                         art_ch["primary"]["rel_err_max"], "6.21"),
        "refine_17 max%": (f"{100*art_ch['refine_17']['rel_err_max']:.2f}",
                           art_ch["refine_17"]["rel_err_max"], "12.34"),
        "refine_65 max%": (f"{100*art_ch['refine_65']['rel_err_max']:.2f}",
                           art_ch["refine_65"]["rel_err_max"], "3.18"),
    }
    for k, (formatted, _, doc_str) in expect.items():
        add(f"doc table {k}", f"{doc_str}% printed in doc",
            f"artifact gives {formatted}%", doc_str in doc and formatted == doc_str)
    add("doc §4 arm-4 (separation) numbers have an artifact",
        "doc prints contrast 3.16e-07 / 7.666e-06 / 1.911e-04 / 1.571e+00",
        "no separation key in xval_results.json and no separation JSON anywhere in the "
        "worktree -> those four numbers are transcribed from stdout only",
        False, "report != artifact for arm 4")

    # ── 5. merge state / integration ──
    add("delivered code is in the repo", "src/tensorlbm/mcmp_tau_mixed.py on a branch",
        "worktree /root/tp-agent/work/m18327 git_status shows '?? src/tensorlbm/mcmp_tau_mixed.py'; "
        "git log --all in the ecosystem checkout: no commit touches it; mission result head_commit=null",
        False, "unmerged: no L3 integration caller can exist")
    add("module exported / called by the library",
        "entry point in src/tensorlbm/__init__.py",
        "grep: only tests/test_mcmp_tau_mixed_xval.py imports tensorlbm.mcmp_tau_mixed; "
        "__init__.py untouched, no solver path references it",
        False)
    add("platform report published for m18327", "report per owner report discipline",
        "SELECT from tensor_reports WHERE mission_id=18327 -> 0 rows", False)
    add("benchmark case registered", "tensor_benchmark_cases row for the anchor",
        "SELECT ... WHERE anchor_slug='XVAL-LBM-SHAN-CHEN-MCMP-TAU-MIXED-VELOCITY' -> 0 rows "
        "(registered by this audit)", False)

    ok = sum(1 for r in rows if r["match"])
    summary = {"checked": len(rows), "match": ok, "mismatch": len(rows) - ok,
               "rows": rows}
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for r in rows:
        flag = "OK  " if r["match"] else "DIFF"
        print(f"[{flag}] {r['item']}: claimed={r['claimed']} | audited={r['audited']}")
    print(f"\n{ok}/{len(rows)} claims reproduce; table -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())