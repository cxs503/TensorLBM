#!/usr/bin/env python3
"""DTRC SUBOFF (1989/1998) external-anchor register — per-point census builder.

Why this file exists
-------------------
The W30 topic ``lbm-suboff-tau`` carried three task keys
(``paper-exp-lbm-suboff-tau-3/4/6``) whose missions reported ``succeeded``
while shipping **no evidence artifact at all**.  The drag anchor of the whole
topic (DTRC/SHD-1298-01 geometry + CRDKNSWC/HD-1298-11 Table 14 tow-tank
resistance) was, until now, only present as one derived register
(``benchmarks/anchors/lbm-suboff-tau.json``) built from a *single*
transcription lineage.

This builder does three things and nothing else:

1. **Per-point registration.**  It parses *every* in-repo transcription of the
   DTRC tow-tank resistance table by regex at build time (zero hand-copy) and
   registers each ``(V, R_T)`` pair as its own anchor point with a verbatim
   quote, a file locator and the sha256 of the quoted archive.
2. **Independent recomputation.**  ``C_F`` is recomputed from ITTC-1957
   ``C_F = 0.075/(log10 Re - 2)^2`` on two independent arithmetic paths
   (50-digit ``Decimal`` vs ``math.log10`` float) and compared, per point,
   against the ``C_F`` values that *other* in-repo registers recorded; ``C_T``
   is recomputed from ``R_T/(0.5 rho V^2 S)``.  Residuals are reported, not
   asserted away.
3. **Evidence grading.**  A/B/C/D are assigned by the same scale as the sibling
   register: a value whose primary document is not acquired in this environment
   cannot be A or B, whatever its internal cross-checks look like.

Two transcription families are in-repo and they **disagree by ~17%**, which is
itself registered here (``cross_family_conflict``).  Resolving that conflict
requires the primary report; the sandbox is offline, so it stays WITHHELD.

Nothing in this file may invent a number: every value comes from a parsed
archive, from a closed-form evaluation of a documented formula, or is marked
``WITHHELD`` / ``derived``.

Usage
-----
    python benchmarks/anchors/dtrc_suboff_1989_anchor.py            # build + verify
    python benchmarks/anchors/dtrc_suboff_1989_anchor.py --check    # verify only

Artifacts
---------
    benchmarks/anchors/dtrc-suboff-1989.json          (this register)
    papers/artifacts/w30/lbm-suboff-tau/suboff_anchor_census.json
                                                      (built by
                                                       benchmarks/anchors/
                                                       suboff_anchor_census.py)
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from decimal import Decimal, getcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "benchmarks/anchors/dtrc-suboff-1989.json"

# ---------------------------------------------------------------------------
# registries (in-repo transcription archives).  Paths are repo-relative.
# ---------------------------------------------------------------------------
REG_T14 = "docs/suboff_experimental_resistance_validation_2026-08-01.md"
REG_T14_X = "src/tensorlbm/suboff_reference_data.py"
REG_S1 = "benchmarks/pending/suboff_highre_resistance/refs_lock.md"
REG_S1_JSON = "benchmarks/pending/suboff_highre_resistance/refs_lock.json"
REG_SIBLING_ANCHOR = "benchmarks/anchors/lbm-suboff-tau.json"

# primary documents the transcriptions point back to (none acquired here)
PRIMARY_DOCS = [
    {
        "doc_id": "DTRC-SHD-1298-01",
        "citation": "P. E. Groves, T. T. Huang, S. S. Chang (1989). Geometric "
        "Characteristics of DARPA SUBOFF Models. DTRC Report SHD-1298-01.",
        "role": "hull geometry / offset tables (L_max, D_max, wetted area)",
        "acquisition_status": "WITHHELD",
        "what_is_missing": "the primary geometry report itself (offset tables + "
        "Table 1 principal particulars) and its sha256",
    },
    {
        "doc_id": "CRDKNSWC-HD-1298-11",
        "citation": "H.-L. Liu, T. T. Huang (1998). Summary of DARPA Suboff "
        "Experimental Program Data. CRDKNSWC/HD-1298-11 (DTIC ADA359226).",
        "role": "AFF-1 bare-hull tow-tank resistance table (Table 14)",
        "acquisition_status": "WITHHELD",
        "what_is_missing": "the primary report PDF (Table 14, report p.23) and "
        "its sha256 — the two in-repo transcriptions disagree by ~17%, so the "
        "primary is the only thing that can adjudicate",
    },
]

# ---------------------------------------------------------------------------
# conventions — declared assumptions, not measurements
# ---------------------------------------------------------------------------
RHO = 1000.0  # kg/m^3  (fresh water, declared assumption)
NU = 1.139e-06  # m^2/s   (fresh water ~15 degC, declared assumption)
KN = 0.514444  # m/s per knot (1852 m / 3600 s)
L_MAX = 4.356  # m
L_PP = 4.261  # m
D_MAX = 0.508  # m
DISPL = 0.708  # m^3
WSA = 5.998  # m^2  (Table 1 transcription; reference area for C_T)
CT_TOL_ABS = 1e-9  # closed-form agreement tolerance (criterion C2)
ITTC57_TOL_ABS = 1e-09

GRADE_SCALE = {
    "A": "primary document verified in this environment (page/PDF read, primary "
    "doc sha256 registered)",
    "B": "transcription + verbatim quote + source-document sha256 (source hashed, "
    "not re-read)",
    "C": "second-hand transcription: neither the primary document nor a "
    "hashed source document is available in this environment",
    "D": "second-hand reproduction / model-derived placeholder",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sha256_of(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - only when git is absent
        return "unknown"


def cf_ittc57_decimal(re: Decimal | float | str) -> Decimal:
    """C_F = 0.075/(log10 Re - 2)^2, 50-digit decimal path (independent #1)."""
    getcontext().prec = 50
    r = re if isinstance(re, Decimal) else Decimal(str(re))
    log10 = r.ln() / Decimal(10).ln()
    return Decimal("0.075") / (log10 - Decimal(2)) ** 2


def cf_ittc57_float(re: float) -> float:
    """Same closed form on the float path (independent #2)."""
    return 0.075 / (math.log10(re) - 2.0) ** 2


def ct_from_rt(rt: float, v_ms: float, rho: float = RHO, s: float = WSA) -> float:
    return rt / (0.5 * rho * v_ms * v_ms * s)


def ct_from_rt_decimal(rt: str, v_ms: Decimal) -> Decimal:
    getcontext().prec = 50
    rho = Decimal(str(RHO))
    s = Decimal(str(WSA))
    return Decimal(rt) / (Decimal("0.5") * rho * v_ms * v_ms * s)


def relative_dev_pct(value: float, reference: float) -> float:
    return (value / reference - 1.0) * 100.0


# ---------------------------------------------------------------------------
# registry parsers (zero hand-copy: every number comes out of a regex)
# ---------------------------------------------------------------------------
def parse_family_t14() -> dict:
    """Family T14: the Table 14 single-row transcription of HD-1298-11 p.23."""
    text = (ROOT / REG_T14).read_text(encoding="utf-8")
    lines = text.splitlines()
    hits = [
        i
        for i, ln in enumerate(lines)
        if "AFF-1" in ln and "87.40" in ln and ln.strip().startswith("|")
    ]
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly 1 AFF-1 Table-14 row, found {len(hits)}")
    idx = hits[0]
    line = lines[idx]
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells) != 3:
        raise RuntimeError(f"unexpected Table-14 row shape: {line!r}")
    speeds = [float(x) for x in re.findall(r"\d+\.?\d*", cells[1])]
    forces = [float(x) for x in re.findall(r"\d+\.?\d*", cells[2])]
    if len(speeds) != len(forces) or len(speeds) < 3:
        raise RuntimeError(f"Table-14 parse imbalance: {speeds} vs {forces}")
    char_offset = text.index(line)
    return {
        "family_id": "T14",
        "registry_path": REG_T14,
        "sha256": sha256_of(REG_T14),
        "locator": {
            "path": REG_T14,
            "line": idx + 1,
            "char_offset": char_offset,
            "table": "Table 14",
            "report_page": 23,
            "citation_line": "CRDKNSWC/HD-1298-11, ADA359226, Table 14, report p.23",
        },
        "verbatim_quote": line.strip(),
        "speed_unit": "knot",
        "force_unit": "N",
        "speeds": speeds,
        "forces": forces,
    }


def parse_family_s1() -> dict:
    """Family S1: the 5-point EFD column transcribed from a 2024 third-party paper."""
    text = (ROOT / REG_S1).read_text(encoding="utf-8")
    row_re = re.compile(r"^\|\s*(\d+\.\d+)\s*\|\s*(\d+\.\d+)\s*\|\s*$")
    rows = []
    for i, ln in enumerate(text.splitlines()):
        m = row_re.match(ln.strip())
        if m:
            rows.append((i + 1, ln.strip(), float(m.group(1)), float(m.group(2))))
    if len(rows) != 5:
        raise RuntimeError(f"expected 5 S1 EFD rows, found {len(rows)}: {rows}")
    ref = json.loads((ROOT / REG_S1_JSON).read_text(encoding="utf-8"))
    anchors = ref["anchors"]
    if len(anchors) != 5:
        raise RuntimeError("refs_lock.json anchors[] does not have 5 points")
    for row, a in zip(rows, anchors):
        if abs(row[2] - a["V_ms"]) > 5e-4 or abs(row[3] - a["RT_N"]) > 5e-3:
            raise RuntimeError(
                f"S1 markdown/json mismatch at line {row[0]}: {row[2:]} vs "
                f"({a['V_ms']}, {a['RT_N']})"
            )
    return {
        "family_id": "S1",
        "registry_path": REG_S1,
        "sha256": sha256_of(REG_S1),
        "machine_cross_source": {
            "path": REG_S1_JSON,
            "sha256": sha256_of(REG_S1_JSON),
            "agrees": True,
            "note": "markdown transcription and machine lock parse to the same "
            "(V, R_T) pairs to 5e-4 m/s / 5e-3 N",
        },
        "locator": {
            "path": REG_S1,
            "lines": [r[0] for r in rows],
            "table": "Table 3 (EFD column)",
            "cited_primary": "Liu & Huang 1998, CRDKNSWC/HD-1298-11 (ref [24] of "
            "CFD Letters 16(10) 2024, 126-139)",
        },
        "verbatim_quote": " / ".join(r[1] for r in rows),
        "speed_unit": "m/s",
        "force_unit": "N",
        "speeds": [r[2] for r in rows],
        "forces": [r[3] for r in rows],
        "s2_corroboration_rel_pct": [a["S2_rel_pct"] for a in anchors],
    }


def geometry_gate() -> dict:
    """Geometry before drag: Cb / L-over-D / wetted area, recomputed here."""
    a_mid = math.pi * (D_MAX / 2.0) ** 2
    cb = DISPL / (a_mid * L_MAX)
    ld = L_MAX / D_MAX
    wsa_analytic = 0.31576 * L_MAX**2  # in-repo analytic surface-of-revolution coef
    checks = {
        "A_mid_m2": a_mid,
        "Cb_recomputed": cb,
        "Cb_nominal": 0.80,
        "Cb_dev_pct": (cb / 0.80 - 1.0) * 100.0,
        "Cb_band_pct": 3.0,
        "L_over_D": ld,
        "L_over_D_recorded": 8.57,
        "L_over_D_dev_pct": (ld / 8.57 - 1.0) * 100.0,
        "L_over_D_band_pct": 1.0,
        "WSA_recorded_m2": WSA,
        "WSA_analytic_coef_L2": 0.31576,
        "WSA_analytic_m2": wsa_analytic,
        "WSA_dev_pct": (WSA / wsa_analytic - 1.0) * 100.0,
        "WSA_band_pct": 1.0,
    }
    checks["pass"] = (
        abs(checks["Cb_dev_pct"]) <= checks["Cb_band_pct"]
        and abs(checks["L_over_D_dev_pct"]) <= checks["L_over_D_band_pct"]
        and abs(checks["WSA_dev_pct"]) <= checks["WSA_band_pct"]
    )
    return checks


# ---------------------------------------------------------------------------
# point construction
# ---------------------------------------------------------------------------
def registered_cf_lookup() -> dict:
    """C_F / C_T values that *other* in-repo registers recorded, keyed by Re."""
    sib = json.loads((ROOT / REG_SIBLING_ANCHOR).read_text(encoding="utf-8"))
    out = {"T14": [], "S1": []}
    for p in sib["anchor_points"]:
        out["T14"].append(
            {
                "id": p["id"],
                "Re": p["Re"],
                "C_F": p["C_F"]["value"],
                "C_T": p["C_T"],
                "register": REG_SIBLING_ANCHOR,
            }
        )
    ref = json.loads((ROOT / REG_S1_JSON).read_text(encoding="utf-8"))
    for i, a in enumerate(ref["anchors"], start=1):
        out["S1"].append(
            {
                "id": f"S1-P{i}",
                "Re": a["Re_Lmax_nu1139"],
                "C_F": a["CT_ITTC1957"],
                "C_T": a["CT"],
                "register": REG_S1_JSON,
            }
        )
    return out


def build_points() -> list[dict]:
    fam_t14 = parse_family_t14()
    fam_s1 = parse_family_s1()
    reg = registered_cf_lookup()
    for fam_reg, key in ((reg["T14"], "T14"), (reg["S1"], "S1")):
        if len(fam_reg) != len(
            fam_t14["speeds"] if key == "T14" else fam_s1["speeds"]
        ):
            raise RuntimeError(f"register/registry point-count mismatch for {key}")

    points: list[dict] = []
    for fam, key, registered in ((fam_t14, "T14", reg["T14"]), (fam_s1, "S1", reg["S1"])):
        for i, (v_raw, rt) in enumerate(zip(fam["speeds"], fam["forces"]), start=1):
            v_ms = v_raw if fam["speed_unit"] == "m/s" else v_raw * KN
            v_kn = v_raw if fam["speed_unit"] == "knot" else v_raw / KN
            re_float = v_ms * L_MAX / NU
            cf_dec = cf_ittc57_decimal(Decimal(str(v_ms)) * Decimal(str(L_MAX)) / Decimal(str(NU)))
            cf_flt = cf_ittc57_float(re_float)
            ct = ct_from_rt(rt, v_ms)
            ct_dec = ct_from_rt_decimal(str(rt), Decimal(str(v_ms)))
            cf_reg = registered[i - 1]
            if abs(cf_reg["Re"] / re_float - 1.0) > 1e-9:
                raise RuntimeError(
                    f"{key}-P{i}: register Re {cf_reg['Re']} != recomputed {re_float}"
                )
            cf_dec_f = float(cf_dec)
            quote = (
                fam["verbatim_quote"]
                if key == "T14"
                else _row_quote(fam, i)
            )
            locator = dict(fam["locator"])
            if key == "S1":
                locator = dict(fam["locator"])
                locator["line"] = fam["locator"]["lines"][i - 1]
                del locator["lines"]
            grade = "C"
            points.append(
                {
                    "id": f"{key}-P{i}",
                    "family": key,
                    "hull_configuration": "AFF-1 bare hull (DARPA SUBOFF, model scale)",
                    "speed_kn": v_kn,
                    "speed_ms": v_ms,
                    "resistance_n": rt,
                    "Re": re_float,
                    "Re_length": "L_max = 4.356 m",
                    "C_T": ct,
                    "C_F": {
                        "value": cf_dec_f,
                        "kind": "ITTC57_CORRELATION_DERIVED",
                        "definition": "C_F = 0.075/(log10 Re - 2)^2 at this point's Re",
                        "arithmetic_path": "decimal(50 digits)",
                        "experimental_value": "WITHHELD",
                        "experimental_note": "the tow-tank programme reports total "
                        "resistance R_T only; no measured C_F for SUBOFF is "
                        "transcribed in-repo, so none is registered",
                    },
                    "recompute": {
                        "C_F_decimal": cf_dec_f,
                        "C_F_float": cf_flt,
                        "path_spread_abs": abs(cf_dec_f - cf_flt),
                        "C_T_from_RT": ct,
                        "C_T_decimal": float(ct_dec),
                        "C_T_path_spread_abs": abs(ct - float(ct_dec)),
                        "registered_C_F": cf_reg["C_F"],
                        "registered_C_F_register": cf_reg["register"],
                        "residual_vs_registered_C_F_abs": abs(cf_dec_f - cf_reg["C_F"]),
                        "registered_C_T": cf_reg["C_T"],
                        "residual_vs_registered_C_T_abs": abs(ct - cf_reg["C_T"]),
                        "residual_vs_registered_C_T_rel": abs(ct / cf_reg["C_T"] - 1.0),
                        "C_F_tolerance_abs": ITTC57_TOL_ABS,
                        "pass": abs(cf_dec_f - cf_reg["C_F"]) <= ITTC57_TOL_ABS,
                    },
                    "form_factor_CT_over_CF": ct / cf_dec_f,
                    "evidence_grade": grade,
                    "acquisition_status": "WITHHELD",
                    "grade_reason": _grade_reason(key),
                    "provenance": {
                        "verbatim_quote": quote,
                        "locator": locator,
                        "sha256": fam["sha256"],
                    },
                    "what_is_missing": _missing(key),
                }
            )
    return points


def _row_quote(fam: dict, i: int) -> str:
    text = (ROOT / fam["registry_path"]).read_text(encoding="utf-8").splitlines()
    return text[fam["locator"]["lines"][i - 1] - 1].strip()


def _grade_reason(key: str) -> str:
    if key == "T14":
        return (
            "grade C: the value is a verbatim transcription of Table 14 archived "
            "in-repo; the primary report CRDKNSWC/HD-1298-11 is not acquired "
            "(offline sandbox) and no sha256 of it can be registered. The in-repo "
            "python mirror (src/tensorlbm/suboff_reference_data.py) shares the "
            "same transcription lineage, so it is not an independent witness and "
            "cannot lift the grade to B."
        )
    return (
        "grade C: the value is a verbatim transcription of the EFD column of "
        "Table 3 of a third-party paper (CFD Letters 16(10) 2024), which itself "
        "cites HD-1298-11 as [24]. The paper PDF (refs/cfdletters2024_suboff.pdf) "
        "is NOT present in this checkout, so not even a secondary source document "
        "can be hashed -> this is C and not B."
    )


def _missing(key: str) -> list[str]:
    base = ["CRDKNSWC/HD-1298-11 (DTIC ADA359226), Table 14, report p.23, as a "
            "hashed primary PDF"]
    if key == "S1":
        base.append("the third-party source PDF refs/cfdletters2024_suboff.pdf "
                    "(absent from this checkout), which would raise the grade to B")
    else:
        base.append("a second independent transcription of the same Table 14 rows "
                    "(the existing one shares lineage with the python mirror)")
    return base


# ---------------------------------------------------------------------------
# cross-family conflict
# ---------------------------------------------------------------------------
def cross_family_conflict(points: list[dict]) -> dict:
    t14 = [p for p in points if p["family"] == "T14"]
    s1 = [p for p in points if p["family"] == "S1"]
    pairs = []
    for a in s1:
        best = min(t14, key=lambda b: abs(b["speed_kn"] - a["speed_kn"]))
        dkn = abs(best["speed_kn"] - a["speed_kn"])
        pairs.append(
            {
                "S1_id": a["id"],
                "T14_id": best["id"],
                "speed_kn": a["speed_kn"],
                "speed_gap_kn": dkn,
                "speed_gap_pct": dkn / a["speed_kn"] * 100.0,
                "RT_S1_N": a["resistance_n"],
                "RT_T14_N": best["resistance_n"],
                "ratio_S1_over_T14": a["resistance_n"] / best["resistance_n"],
                # R_T ~ V^1.9065 (in-repo power-law fit: C_T ~ Re^-0.0935), so a
                # speed mismatch of dp% perturbs the ratio by ~1.91*dp%.
                "speed_gap_perturbs_ratio_pct": 1.9065 * dkn / a["speed_kn"] * 100.0,
                "same_speed_within_0p2pct": dkn / a["speed_kn"] * 100.0 < 0.2,
            }
        )
    ratios = [p["ratio_S1_over_T14"] for p in pairs]
    mean = sum(ratios) / len(ratios)
    sd = (sum((r - mean) ** 2 for r in ratios) / (len(ratios) - 1)) ** 0.5
    # 6th point of each family: T14 has 17.99 kn; S1's 6th (17.79 kn) is derived
    # from the Flow360 error table and is NOT an S1 table row.
    t14_6 = t14[-1]
    ref = json.loads((ROOT / REG_S1_JSON).read_text(encoding="utf-8"))
    derived6 = ref["derived_6th"][0]
    ratio6 = derived6["RT_implied_N"] / t14_6["resistance_n"]
    return {
        "statement": "the two in-repo transcriptions of the same DTRC tow-tank "
        "table differ by a nearly constant factor ~1.17 across all overlapping "
        "speeds; they cannot both be Table 14 AFF-1 bare-hull values as recorded",
        "pairs": pairs,
        "ratio_mean": mean,
        "ratio_sd": sd,
        "ratio_min": min(ratios),
        "ratio_max": max(ratios),
        "constant_offset_share_pct": (1.0 - 1.0 / mean) * 100.0,
        "sixth_point_check": {
            "T14_id": t14_6["id"],
            "T14_speed_kn": t14_6["speed_kn"],
            "T14_RT_N": t14_6["resistance_n"],
            "S1_derived_speed_kn": derived6["V_kn"],
            "S1_derived_RT_N": derived6["RT_implied_N"],
            "S1_derived_source": f"{REG_S1_JSON}:/derived_6th[0] (back-derived "
            "from the Flow360 DDES error table — NOT an S1 table row)",
            "ratio": ratio6,
        },
        "speed_gap_tolerance_pct": 0.2,
        "speed_gap_max_pct": max(p["speed_gap_pct"] for p in pairs),
        "max_ratio_perturbation_from_speed_gap_pct": max(
            p["speed_gap_perturbs_ratio_pct"] for p in pairs
        ),
        "max_dev_pct_between_families": (mean + 3 * sd / math.sqrt(len(ratios)) - 1) * 100.0,
        "gate_15pct_verdict": "CONFLICT_EXCEEDS_GATE"
        if abs(mean - 1.0) * 100.0 > 15.0
        else "within_gate",
        "resolution_requires": "primary CRDKNSWC/HD-1298-11 Table 14 (p.23); "
        "offline this stays WITHHELD",
        "hypotheses_ranked": [
            {
                "id": "H1",
                "statement": "one of the two transcriptions is not the AFF-1 "
                "bare-hull row (wrong configuration column, e.g. a with-appendages "
                "or a different model-scale row)",
                "supporting": "the offset is a pure multiplicative constant "
                f"{mean:.3f} independent of speed (sd {sd:.4f}); appendages alone "
                "explain only ~2.6% (in-repo appendage share), so a column mix-up "
                "rather than a physics difference is the better fit",
                "test": "read Table 14 page 23 and match both the speed list and "
                "the R_T list; also compare against Table 15/16 configuration rows",
            },
            {
                "id": "H2",
                "statement": "one family was normalised at a different density or "
                "reference scale before being tabulated",
                "supporting": "rho band 997-1000 gives only 0.3% and a model-scale "
                "change scales as L^2 (factor 4 or 0.25), so neither closes 1.17 "
                "cleanly",
                "test": "check the primary for a reference-temperature / density "
                "footnote; both families would then need a re-derivation",
            },
            {
                "id": "H3",
                "statement": "the 6-significant-digit transcription pair is right "
                "and the third-party paper rescaled EFD to a different reference "
                "area for its C_T comparison",
                "supporting": "S1's Table 3 is titled 'resistance in comparison "
                "with EFD' and may tabulate a rescaled force; but the in-repo "
                "S1 transcription is dimensional N, not a coefficient",
                "test": "compare S1 Table 3 numbers with the paper's own CFD "
                "column and with its stated reference area",
            },
        ],
    }


# ---------------------------------------------------------------------------
# build / verify
# ---------------------------------------------------------------------------
def build() -> dict:
    points = build_points()
    fam_t14 = parse_family_t14()
    fam_s1 = parse_family_s1()
    geo = geometry_gate()
    conflict = cross_family_conflict(points)

    residuals = [p["recompute"]["residual_vs_registered_C_F_abs"] for p in points]
    spreads = [p["recompute"]["path_spread_abs"] for p in points]
    n_pass = sum(1 for p in points if p["recompute"]["pass"])

    criteria = [
        {
            "id": "C1_points_and_sha256",
            "statement": ">=3 numerical anchor points, each with verbatim quote + "
            "locator + recomputable sha256 of the quoted archive",
            "n_points": len(points),
            "n_families": 2,
            "n_sha256_checked": 3,
            "sha256_checked": {
                REG_T14: sha256_of(REG_T14),
                REG_S1: sha256_of(REG_S1),
                "benchmarks/pending/suboff_highre_resistance/formal/formal_verdict.json": sha256_of(
                    "benchmarks/pending/suboff_highre_resistance/formal/formal_verdict.json"
                ),
            },
            "pass": len(points) >= 3,
        },
        {
            "id": "C2_ittc57_recompute_residual",
            "statement": "independent ITTC-1957 recomputation agrees with every "
            "registered C_F to <= 1e-9 absolute",
            "tolerance_abs": ITTC57_TOL_ABS,
            "n_points": len(points),
            "n_pass": n_pass,
            "max_abs_residual": max(residuals),
            "max_two_path_spread": max(spreads),
            "pass": n_pass == len(points) and max(residuals) <= ITTC57_TOL_ABS,
        },
        {
            "id": "C3_cross_family_conflict",
            "statement": "the two in-repo transcription families agree within the "
            "15% drag gate",
            "ratio_mean": conflict["ratio_mean"],
            "ratio_sd": conflict["ratio_sd"],
            "dev_pct": (conflict["ratio_mean"] - 1.0) * 100.0,
            "gate_pct": 15.0,
            "pass": False,
            "honest_fail": True,
            "consequence": "an anchor-to-simulation deviation below 15% cannot be "
            "adjudicated from these two families until the primary is acquired; "
            "the census reports the deviation against BOTH families",
        },
        {
            "id": "C4_geometry_gate",
            "statement": "Cb / L-over-D / wetted area validated before the drag "
            "anchor is used",
            "detail": geo,
            "pass": geo["pass"],
        },
        {
            "id": "C5_evidence_grade",
            "statement": "evidence grade assigned as-is: A needs a read primary, B "
            "needs a hashed source document",
            "grades": {p["id"]: p["evidence_grade"] for p in points},
            "max_grade": "C",
            "pass": True,
            "note": "all points grade C by construction in this environment; a "
            "higher grade is REFUSED, not unavailable",
        },
    ]

    art = {
        "schema": "dtrc-suboff-1989-anchor/v2",
        "anchor_id": "dtrc-suboff-1989",
        "title": "DTRC/DARPA SUBOFF (Groves-Huang-Chang 1989; Liu-Huang 1998) "
        "tow-tank resistance anchor, registered per point",
        "repo": "TensorLBM",
        "discipline": "ship-total-performance / resistance (bare submarine hull)",
        "generator": "benchmarks/anchors/dtrc_suboff_1989_anchor.py",
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "git_commit": git_commit(),
        "status": "TRANSCRIBED",
        "status_meaning": {
            "TRANSCRIBED": "every value is a verbatim transcription of a table "
            "archived in this repo; no primary document was re-read here",
            "VERIFIED": "values re-read from the primary document and matched",
            "OBTAINED": "primary document stored in-repo with sha256",
        },
        "acquisition_status": "WITHHELD",
        "grade_scale": GRADE_SCALE,
        "environment": {
            "python": sys.version.split()[0],
            "offline": True,
            "note": "the sandbox has no network; primary-document acquisition is "
            "impossible here, which is why the ceiling grade is C",
        },
        "primary_documents": PRIMARY_DOCS,
        "registries": {
            "T14": {k: v for k, v in fam_t14.items() if k not in ("speeds", "forces")},
            "S1": {k: v for k, v in fam_s1.items() if k not in ("speeds", "forces")},
        },
        "conventions": {
            "rho_kg_m3": RHO,
            "rho_provenance": "declared fresh-water assumption (convention, not measured)",
            "nu_m2_s": NU,
            "nu_provenance": "declared fresh-water ~15 degC assumption "
            "(benchmarks/pending/suboff_highre_resistance/refs_lock.json:/assumptions)",
            "kn_to_ms": KN,
            "kn_to_ms_provenance": "international knot = 1852 m / 3600 s",
            "re_length": "L_max = 4.356 m",
            "ct_definition": "C_T = R_T / (0.5 * rho * V^2 * S_wetted)",
            "cf_definition": "C_F = 0.075/(log10(Re)-2)^2 (ITTC-1957 correlation line)",
            "reference_area": f"WSA = {WSA} m^2 (Table 1 transcription)",
        },
        "geometry": {
            "L_max_m": L_MAX,
            "L_PP_m": L_PP,
            "D_max_m": D_MAX,
            "displacement_m3": DISPL,
            "WSA_m2": WSA,
            "gate": geo,
        },
        "anchor_points": points,
        "cross_family_conflict": conflict,
        "criteria": criteria,
        "summary": {
            "n_anchor_points": len(points),
            "n_families": 2,
            "Re_range": [min(p["Re"] for p in points), max(p["Re"] for p in points)],
            "C_T_range": [min(p["C_T"] for p in points), max(p["C_T"] for p in points)],
            "max_ittc57_residual_abs": max(residuals),
            "all_ittc57_residuals_within_tol": n_pass == len(points),
            "all_criteria_pass_or_honest": all(
                c["pass"] or c.get("honest_fail") for c in criteria
            ),
            "blocking_issue": "cross-family conflict 17% > 15% gate (criterion C3); "
            "primary report needed",
        },
        "claim_boundary": {
            "may_be_cited_as_published_experimental_benchmark": False,
            "may_be_cited_as": "per-point transcription register of CRDKNSWC/"
            "HD-1298-11 Table 14 recovered from two in-repo transcription families "
            "that disagree by ~17%",
            "refused_labels": ["A", "B", "PRIMARY-VERIFIED", "PRIMARY-OBTAINED"],
            "consequence": "no simulation-to-anchor agreement claim above 'agrees "
            "with a second-hand transcription, and only within the family spread' "
            "is admissible",
        },
    }
    return art


def verify(path: Path = OUT_PATH) -> dict:
    art = json.loads(Path(path).read_text(encoding="utf-8"))
    problems: list[str] = []
    for key in (
        "schema",
        "status",
        "acquisition_status",
        "conventions",
        "geometry",
        "anchor_points",
        "registries",
        "criteria",
        "cross_family_conflict",
    ):
        if key not in art:
            problems.append(f"missing required key {key}")
    if art.get("status") not in ("TRANSCRIBED", "VERIFIED", "OBTAINED"):
        problems.append(f"bad status {art.get('status')!r}")
    if len(art.get("anchor_points", [])) < 3:
        problems.append("fewer than 3 anchor points")
    for p in art.get("anchor_points", []):
        loc = p["provenance"]["locator"]
        rel = loc["path"]
        text = (ROOT / rel).read_text(encoding="utf-8")
        if p["provenance"]["verbatim_quote"] not in text:
            problems.append(f"{p['id']}: quote not found in {rel}")
        if sha256_of(rel) != p["provenance"]["sha256"]:
            problems.append(f"{p['id']}: sha256 drift on {rel}")
        cf = float(cf_ittc57_decimal(Decimal(p["Re"])))
        if abs(cf - p["C_F"]["value"]) > ITTC57_TOL_ABS:
            problems.append(f"{p['id']}: ITTC-57 self-consistency > {ITTC57_TOL_ABS}")
        rc = p["recompute"]
        if rc["residual_vs_registered_C_F_abs"] > ITTC57_TOL_ABS:
            problems.append(
                f"{p['id']}: registered C_F residual {rc['residual_vs_registered_C_F_abs']:.3e}"
            )
        if p["evidence_grade"] not in GRADE_SCALE:
            problems.append(f"{p['id']}: bad grade {p['evidence_grade']!r}")
        if p["acquisition_status"] not in ("OBTAINED", "TRANSCRIBED", "WITHHELD"):
            problems.append(f"{p['id']}: bad acquisition_status")
    return {"path": str(Path(path)), "ok": not problems, "problems": problems}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--check" in argv:
        res = verify()
        print(json.dumps(res, indent=1, ensure_ascii=False))
        return 0 if res["ok"] else 1
    art = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(art, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    res = verify()
    print(
        json.dumps(
            {
                "written": str(OUT_PATH),
                "n_points": art["summary"]["n_anchor_points"],
                "max_ittc57_residual_abs": art["summary"]["max_ittc57_residual_abs"],
                "cross_family_ratio": art["cross_family_conflict"]["ratio_mean"],
                "verify_ok": res["ok"],
                "problems": res["problems"],
            },
            indent=1,
            ensure_ascii=False,
        )
    )
    return 0 if res["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())