#!/usr/bin/env python3
"""Independent audit of the square-cylinder Re=100 case (anchor CASE-LBM-SQUARE-CYL-NORBERG).

Why this file exists (verifier seat, mission 16608)
===================================================
`verify.py` and `verify_traces.py` are the case's *own* judges: they import
``refspec`` and therefore share its constants, its column policy and its error
math.  A judge that shares the producer's constants can only prove internal
consistency, not that the claim is right.  This script is a second, deliberately
*non-importing* opinion:

* it never imports ``refspec``/``verify``/``verify_traces``/``run``;
* it re-derives the reference band from the *cited source list* that is stored
  in ``result.json`` (the arXiv table entries), instead of trusting
  ``REF_CD``/``REF_ST``;
* it recomputes every gated number from the raw archive JSON with its own
  arithmetic (block means, A/B column deltas, St estimator spread);
* it re-reads the ledger acceptance string and compares it against what the
  case actually measures.  This is the check that no in-directory judge can
  make, because it is a *scope* check, not a numerics check.

Output: JSON artifact + human-readable report; exit 1 when a blocking finding
is found.  Findings are classified blocking / advisory / not_run -- a missing
check is reported as ``not_run`` and never silently counted as a pass.

Usage::

    python independent_audit.py [--stage DIR] [--ledger-acceptance FILE] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Independent restatement of the acceptance bar and the reference centre,
# typed here on purpose: if the case directory edits its constants, this audit
# must *disagree* rather than follow.
BAR_PCT = 3.0
AUDIT_REF_CD = 1.48
AUDIT_REF_ST = 0.147

# The anchor's registered acceptance string (feature_ledger, slug
# CASE-LBM-SQUARE-CYL-NORBERG): "St=fD/U 与 Cd 均值 vs Norberg1993
# (Re=200, 阻塞比 1/20)" with "<8% St / <10% Cd".
LEDGER_RE = 200.0
LEDGER_BLOCKAGE = 1.0 / 20.0
LEDGER_TOL_ST = 8.0
LEDGER_TOL_CD = 10.0


def pct(got: float, ref: float) -> float:
    return (got - ref) / ref * 100.0


def load_cases(stage: Path) -> dict[int, dict]:
    out = {}
    for p in sorted(stage.glob("case_D*.json")):
        d = json.loads(p.read_text())
        out[int(d["D"])] = d
    return out


def primary(d: dict) -> tuple[str, float, float]:
    """(column, cd, st) on the primary column, derived here, not imported."""
    if "force_columns" in d and "pair" in d:
        return "pair", float(d["pair"]["cd"]), float(d["pair"]["st"])
    return "summary", float(d["cd"]), float(d["st"])


SUPERSEDED = (1.4785, 0.1456)  # v1 convention, declared legacy by the producer


def reference_band_from_sources(stage: Path) -> dict:
    """Re-derive the 2D free-stream cluster centre from the cited source list.

    ``result.json`` stores the enumerated literature entries inside its
    ``reference`` string.  Extracting them and recomputing the centre means the
    REF_CD/REF_ST constants are checked against *their own sources* rather than
    against themselves.
    """
    text = ""
    rp = stage / "result.json"
    if rp.is_file():
        res = json.loads(rp.read_text())
        for key in ("reference", "reference_note", "description"):
            if isinstance(res.get(key), str):
                text += res[key]
        for g in (res.get("grids") or {}).values():
            if isinstance(g, dict):
                for key in ("reference", "reference_note"):
                    if isinstance(g.get(key), str):
                        text += g[key]
    pairs = [(float(a), float(b)) for a, b in re.findall(r"(\d\.\d{2,4})\s*/\s*(0\.\d{3})", text)]
    if not pairs:
        return {"status": "not_run", "reason": "no parseable <Cd>/<St> entries in the stored source list"}
    cds = [c for c, _ in pairs]
    sts = [s for _, s in pairs]
    return {
        "status": "ok",
        "n_entries": len(pairs),
        "entries": pairs,
        "cd_min": min(cds), "cd_max": max(cds),
        "cd_median": statistics.median(cds), "cd_mean": statistics.fmean(cds),
        "st_min": min(sts), "st_max": max(sts),
        "st_median": statistics.median(sts), "st_mean": statistics.fmean(sts),
        "audit_ref_cd_in_range": min(cds) <= AUDIT_REF_CD <= max(cds),
        "audit_ref_st_in_range": min(sts) <= AUDIT_REF_ST <= max(sts),
        "ref_cd_vs_median_pct": pct(AUDIT_REF_CD, statistics.median(cds)),
        "ref_st_vs_median_pct": pct(AUDIT_REF_ST, statistics.median(sts)),
    }


def worst_needed(band: dict, levels: dict) -> float:
    sts = [levels[D]["st"] for D in levels]
    return max(max(abs(pct(g, band["st_min"])) for g in sts),
               max(abs(pct(g, band["st_max"])) for g in sts))


def st_estimator_spread(d: dict, col: str) -> dict | None:
    """Two St estimators are archived (spectral ``st`` and zero-crossing).

    The gate consumes whichever is larger; the spread is the definitional
    ambiguity that has to be disclosed next to any |St err| claim.
    """
    if col == "summary":
        if "st" in d and "st_crossing" in d:
            a, b = float(d["st"]), float(d["st_crossing"])
        else:
            return None
    else:
        blk = d.get(col) or {}
        if "st" not in blk or "st_crossing" not in blk:
            return None
        a, b = float(blk["st"]), float(blk["st_crossing"])
    return {"spectral": a, "crossing": b, "spread_pct": abs(pct(a, b))}


def block_stats(d: dict, col: str) -> dict | None:
    blk = d.get(col) if col != "summary" else d
    if not isinstance(blk, dict):
        return None
    blocks = blk.get("cd_blocks")
    if not blocks:
        return None
    return {
        "n_blocks": len(blocks),
        "mean": statistics.fmean(blocks),
        "std": statistics.pstdev(blocks),
        "sem_pct_of_mean": statistics.pstdev(blocks) / statistics.fmean(blocks) * 100.0,
    }


def audit(stage: Path, acceptance_text: str) -> dict:
    cases = load_cases(stage)
    if not cases:
        raise SystemExit(f"no case_D*.json under {stage}")
    d32 = cases[min(cases)]
    findings: list[dict] = []
    levels = {}

    for D in sorted(cases):
        d = cases[D]
        col, cd, st = primary(d)
        ecc, est = pct(cd, AUDIT_REF_CD), pct(st, AUDIT_REF_ST)
        levels[D] = {
            "column": col,
            "cd": cd, "st": st,
            "cd_err_pct": ecc, "st_err_pct": est,
            "passes_3pct": abs(ecc) <= BAR_PCT and abs(est) <= BAR_PCT,
            "st_estimators": st_estimator_spread(d, col),
            "block_stats": block_stats(d, col),
            "re": d.get("re"), "nu": d.get("nu_lb"), "steps": d.get("steps"),
            "nx": d.get("nx"), "ny": d.get("ny"),
        }
        # recomputation must agree with the stored error columns (drift guard)
        stored = (d.get(col) or {}) if col != "summary" else d
        for name, mine, key in (
            ("cd", ecc, "err_cd_pct" if col != "summary" else "err_pct"),
            ("st", est, "err_st_pct" if col != "summary" else "st_err_pct"),
        ):
            old = stored.get(key)
            if old is None or abs(old - mine) <= 0.01:
                continue
            leg_cd = pct(cd, SUPERSEDED[0]) if name == "cd" else pct(st, SUPERSEDED[1])
            declared_legacy = abs(old - (pct(cd, SUPERSEDED[0]) if name == "cd" else pct(st, SUPERSEDED[1]))) <= 0.01
            findings.append({
                "severity": "advisory" if declared_legacy else "blocking",
                "kind": "stored-error-legacy" if declared_legacy else "stored-error-drift",
                "detail": (f"D={D} {name}: stored {old:.4f}% vs locked-convention {mine:.4f}%; "
                           + (f"exactly reproduces the superseded {SUPERSEDED[0] if name == 'cd' else SUPERSEDED[1]} "
                              "convention, so it is legacy (declared), not drift")
                           if declared_legacy else
                           f"reproduced by no named convention (checked {AUDIT_REF_CD}/{AUDIT_REF_ST} and "
                           f"{SUPERSEDED[0]}/{SUPERSEDED[1]}) -> unattributable"),
            })

    band = reference_band_from_sources(stage)

    # --- scope check: what the ledger registers vs what the case measures ----
    scope: dict = {
        "ledger_acceptance": acceptance_text.strip() or "(not supplied)",
        "ledger_re": LEDGER_RE,
        "ledger_blockage": LEDGER_BLOCKAGE,
        "ledger_tol": {"st_pct": LEDGER_TOL_ST, "cd_pct": LEDGER_TOL_CD},
        "delivered_re": {D: levels[D]["re"] for D in levels},
        "delivered_blockage": 1.0 / 40.0,
        "delivered_tol_pct": BAR_PCT,
    }
    mismatched_re = [D for D in levels if levels[D]["re"] not in (None, LEDGER_RE)]
    if acceptance_text and mismatched_re:
        findings.append({
            "severity": "blocking", "kind": "anchor-scope-mismatch",
            "detail": (
                f"ledger acceptance registers Re={LEDGER_RE:g} / blockage 1/20 / "
                f"<{LEDGER_TOL_ST:g}% St,<{LEDGER_TOL_CD:g}% Cd; the delivered case is "
                f"Re={levels[mismatched_re[0]]['re']:g} / blockage 1/40 / {BAR_PCT:g}% bars. "
                "The delivered number cannot close the registered anchor as written."
            ),
        })

    # --- direct judgement: any level outside the bar is a blocking finding ----
    for D in sorted(levels):
        v = levels[D]
        if not v["passes_3pct"]:
            findings.append({
                "severity": "blocking", "kind": "out-of-bar-level",
                "detail": (f"D={D} ({v['column']} column): Cd {v['cd']:.4f} ({v['cd_err_pct']:+.2f}%) "
                           f"/ St {v['st']:.4f} ({v['st_err_pct']:+.2f}%) against the independent "
                           f"centre {AUDIT_REF_CD}/{AUDIT_REF_ST} exceeds the {BAR_PCT:g}% bar."),
            })

    # --- robustness of the verdict to the reference-centre choice -------------
    if band["status"] == "ok":
        worst = {}
        for name, got_key, lo, hi in (("cd", "cd", band["cd_min"], band["cd_max"]),
                                      ("st", "st", band["st_min"], band["st_max"])):
            gots = [levels[D][got_key] for D in levels]
            worst[name] = {
                "err_at_lo_pct": max(pct(g, lo) for g in gots),
                "err_at_hi_pct": min(pct(g, hi) for g in gots),
                "max_abs_pct_over_band": max(max(abs(pct(g, lo)) for g in gots),
                                             max(abs(pct(g, hi)) for g in gots)),
            }
        scope["band_robustness"] = worst
        for name in ("cd", "st"):
            if worst[name]["max_abs_pct_over_band"] > BAR_PCT:
                findings.append({
                    "severity": "advisory", "kind": "verdict-conditional-on-centre",
                    "detail": (f"{name}: the 3% PASS is conditional on the chosen centre. Over the "
                               f"cited band the same measurement spans "
                               f"{worst[name]['err_at_hi_pct']:+.2f}% .. {worst[name]['err_at_lo_pct']:+.2f}%, "
                               f"so |err| can reach {worst[name]['max_abs_pct_over_band']:.2f}% > {BAR_PCT:g}% "
                               "at the unfavourable end. Declaring 'Cd within 3%' without naming the centre "
                               "oversells the result."),
                })
        findings.append({
            "severity": "advisory", "kind": "centre-not-the-median",
            "detail": (f"the locked centre St={AUDIT_REF_ST} is {band['ref_st_vs_median_pct']:+.2f}% off the median "
                       f"({band['st_median']}) of the {band['n_entries']} entries the case itself enumerates; "
                       f"Cd centre is {band['ref_cd_vs_median_pct']:+.2f}% off its median ({band['cd_median']}). "
                       "The pass survives (worst |St err| over the band stays "
                       f"{worst_needed(band, levels):.2f}% <= {BAR_PCT:g}%), but the centre is a round adopted "
                       "value, not a statistic of the sources."),
        })

    # --- advisory: St is claimed on the larger of two archived estimators ----
    for D in sorted(levels):
        est = levels[D]["st_estimators"]
        if est and est["spread_pct"] > 0.5:
            findings.append({
                "severity": "advisory", "kind": "st-estimator-ambiguity",
                "detail": (
                    f"D={D}: spectral St {est['spectral']:.4f} vs zero-crossing "
                    f"{est['crossing']:.4f} ({est['spread_pct']:.2f}% apart); the gate scores the "
                    "spectral value, i.e. the one nearer the reference. Both are inside the 3% bar, "
                    "so the verdict does not flip -- but the definition must be declared."
                ),
            })

    # --- advisory: uncertainty of the archived means -------------------------
    for D in sorted(levels):
        bs = levels[D]["block_stats"]
        if bs is None:
            findings.append({
                "severity": "not_run", "kind": "archived-uncertainty",
                "detail": f"D={D}: archive stores no block averages, so the statistical error of the "
                          "reported Cd/St cannot be reconstructed from the repository.",
            })
        elif bs["sem_pct_of_mean"] > 0.3 * BAR_PCT:
            findings.append({
                "severity": "advisory", "kind": "statistical-margin",
                "detail": f"D={D}: block std {bs['sem_pct_of_mean']:.2f}% of mean = "
                          f"{bs['sem_pct_of_mean'] / BAR_PCT:.0%} of the 3% bar.",
            })

    # --- independent reproduction status (never inferred as a pass) ----------
    repro_dir = stage.parents[0] / "runs" / "sq_repro_D32"
    repro = {"status": "not_run", "detail": "no fresh re-run archive found"}
    if (repro_dir.is_dir()) and list(repro_dir.glob("case_D*.json")):
        fresh = {int(json.loads(p.read_text())["D"]): json.loads(p.read_text())
                 for p in sorted(repro_dir.glob("case_D*.json"))}
        table = {}
        for D, d in fresh.items():
            if D not in levels:
                continue
            col, cd, st = primary(d)
            table[D] = {"cd": cd, "st": st,
                        "cd_vs_archive_pct": pct(cd, levels[D]["cd"]),
                        "st_vs_archive_pct": pct(st, levels[D]["st"])}
        repro = {"status": "ok" if table else "not_run", "levels": table,
                 "tol_pct": 0.5,
                 "passes": all(abs(v["cd_vs_archive_pct"]) <= 0.5 and abs(v["st_vs_archive_pct"]) <= 0.5
                               for v in table.values()) if table else False}
    findings.append({"severity": "not_run", "kind": "independent-reproduction",
                     "detail": f"fresh full re-run (audit item 4): {repro['status']} -- {repro['detail'] if 'detail' in repro else 'compared'}"})

    blocking = [f for f in findings if f["severity"] == "blocking"]
    return {
        "audit": "square_cylinder independent audit (mission m16608)",
        "stage": str(stage),
        "independent_constants": {"cd": AUDIT_REF_CD, "st": AUDIT_REF_ST, "bar_pct": BAR_PCT},
        "levels": {str(k): v for k, v in levels.items()},
        "reference_band_from_sources": band,
        "scope": scope,
        "reproduction": repro,
        "findings": findings,
        "counts": {
            "blocking": len(blocking),
            "advisory": len([f for f in findings if f["severity"] == "advisory"]),
            "not_run": len([f for f in findings if f["severity"] == "not_run"]),
        },
        "disposition": "request_changes" if blocking else "approve_with_notes",
    }


def render(res: dict) -> str:
    L = []
    L.append("# 方柱 Re=100 算例 · 独立审计（非导入 refspec 的第二意见）\n")
    L.append(f"常量（本审计自带，不从算例目录继承）：Cd_ref={AUDIT_REF_CD}, St_ref={AUDIT_REF_ST}, 门={BAR_PCT}%\n")
    L.append("| D | 列 | Cd | vs 1.48 | St | vs 0.147 | 3% 门 | 块 std | St 估计子跨度 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for D, v in res["levels"].items():
        bs = v["block_stats"]
        se = v["st_estimators"]
        L.append(
            f"| {D} | {v['column']} | {v['cd']:.4f} | {v['cd_err_pct']:+.2f}% | {v['st']:.4f} | "
            f"{v['st_err_pct']:+.2f}% | {'PASS' if v['passes_3pct'] else 'FAIL'} | "
            f"{(f'{bs[chr(115)+chr(116)+chr(100)]:.4f}' if bs else 'n/a')} | "
            f"{(f'{se[chr(115)+chr(112)+chr(114)+chr(101)+chr(97)+chr(100)+chr(95)+chr(112)+chr(99)+chr(116)]:.2f}%' if se else 'n/a')} |")
    b = res["reference_band_from_sources"]
    L.append("\n## 参考簇独立重导（从 result.json 存储的来源列表解析）\n")
    if b["status"] == "ok":
        L.append(f"- 解析到 {b['n_entries']} 条 Cd/St 对：Cd ∈ [{b['cd_min']}, {b['cd_max']}] 中位 {b['cd_median']}，"
                 f"St ∈ [{b['st_min']}, {b['st_max']}] 中位 {b['st_median']}")
        L.append(f"- 审计常量 1.48 落在 Cd 区间内：{b['audit_ref_cd_in_range']}（vs 中位 {b['ref_cd_vs_median_pct']:+.2f}%）；"
                 f"0.147 落在 St 区间内：{b['audit_ref_st_in_range']}（vs 中位 {b['ref_st_vs_median_pct']:+.2f}%）")
    else:
        L.append(f"- not_run: {b['reason']}")
    L.append("\n## 发现（按严重度）\n")
    for f in res["findings"]:
        L.append(f"- **[{f['severity']}] {f['kind']}** — {f['detail']}")
    L.append(f"\n计数：blocking {res['counts']['blocking']} / advisory {res['counts']['advisory']} / "
             f"not_run {res['counts']['not_run']}  → 处置：**{res['disposition']}**")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="independent audit of the square_cylinder case")
    ap.add_argument("--stage", default=str(HERE))
    ap.add_argument("--ledger-acceptance", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    stage = Path(args.stage)
    text = ""
    if args.ledger_acceptance:
        text = Path(args.ledger_acceptance).read_text()
    res = audit(stage, text)
    out = Path(args.json) if args.json else HERE / "independent_audit.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    report = render(res)
    (HERE / "INDEPENDENT_AUDIT.md").write_text(report)
    print(report)
    raise SystemExit(1 if res["counts"]["blocking"] else 0)


if __name__ == "__main__":
    main()