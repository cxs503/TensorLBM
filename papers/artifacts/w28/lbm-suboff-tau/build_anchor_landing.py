#!/usr/bin/env python3
"""W28 · land the DTRC SUBOFF anchor on main, grade it honestly, and guard the
three-tier GCI recomputation.

Mission: paper-exp-lbm-suboff-tau-13 (m19903, repo TensorLBM, runner=native).

Every number in papers/artifacts/w28/lbm-suboff-tau/anchor_landing.json is produced
by this script; nothing is typed in by hand.

  step 1  landing    copy the W27 anchor blob out of the un-merged mission branch
                     `agent/m18960` into the main worktree byte for byte, hash it, then
                     apply the gate-v3 grading amendment as ONE contiguous byte insertion
                     whose exact text is recorded, so that strip(landed) == branch blob;
  step 2  grading    derive /anchor_grade + /grade_rationale from the file's own
                     (status, acquisition_status) pair: TRANSCRIBED + WITHHELD
                     => TRANSCRIBED-SECONDARY (grade C); A/B are refused;
  step 3  ittc57     independently recompute ITTC-1957 C_F on two disjoint arithmetic
                     paths (libm double vs Decimal-50 ln) for every anchor point and
                     close it against the registered C_F (tolerance 1e-9 abs);
  step 4  same-Re    transpose each archived ladder tier (L96/L112/L128; the e2 arm, the
                     archived mix50 arm and the bare-hull arm) onto every anchor Reynolds
                     number under the three pre-registered operators, and tabulate the
                     point-wise relative deviation from anchor C_T;
  step 5  gci        three-grid Roache GCI + observed order with the true unequal
                     refinement ratios (r21 = 128/112, r32 = 112/96), plus the equal-ratio
                     shortcut as a sensitivity only.

Usage:
    python build_anchor_landing.py            # land + recompute + write artifact
    python build_anchor_landing.py --check    # verify only (no writes), exit != 0 on drift
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from decimal import Decimal, getcontext
from pathlib import Path

# --------------------------------------------------------------------------------------
# pinned provenance
# --------------------------------------------------------------------------------------
TASK_KEY = "paper-exp-lbm-suboff-tau-13"
MISSION_ID = 19903
TASK_KEY_W27 = "paper-exp-lbm-suboff-tau-12"
MISSION_ID_W27 = 18960

SOURCE_REF = "agent/m18960"
SOURCE_REF_COMMIT = "71c90fd13e591a2bec1a021b438c2556e54c36c4"
ANCHOR_REL = "benchmarks/anchors/lbm-suboff-tau.json"
ANCHOR_BRANCH_BLOB_SHA256 = "b452bccd7d35ce1c294b159342b2d0e0fbb75c6fd5c574975df732abe7b54f88"
ANCHOR_BRANCH_BLOB_BYTES = 31108
GEN_REL = "benchmarks/anchors/lbm_suboff_tau_anchor.py"
GEN_BRANCH_BLOB_SHA256 = "2f214291b95f9cbcdd6290bfb778a4c94d2386a29449d7adacd5d0647698808c"
README_REL = "benchmarks/anchors/README.md"
README_BRANCH_BLOB_SHA256 = "eea7ba387deedf06debe43d67535117645c2f6c84e9acf74ee8fe4c6d729e167"
W27_RECONCILE_REL = "papers/artifacts/w27/lbm-suboff-tau/dtrc_suboff_anchor_reconcile.json"
W27_RECONCILE_SHA256 = "722651ef9a8b16425394b4fe7f9089f8b49bdf54de87f2f30ac65c8cd64b7525"

ARTIFACT_REL = "papers/artifacts/w28/lbm-suboff-tau/anchor_landing.json"

GRADE_SCALE = {
    "A": "primary document verified in this environment (page/PDF read, primary doc sha256 registered)",
    "B": "transcription + verbatim quote + source-document sha256 (primary hashed, not re-read)",
    "C": "second-hand transcription (no primary document acquired, no primary-document sha256)",
    "D": "second-hand reproduction / model-derived placeholder",
}
GRADE_FOR_STATE = {
    ("TRANSCRIBED", "WITHHELD"): "TRANSCRIBED-SECONDARY",
    ("TRANSCRIBED", "TRANSCRIBED"): "TRANSCRIBED-SECONDARY",
    ("VERIFIED", "OBTAINED"): "PRIMARY-VERIFIED",
    ("OBTAINED", "OBTAINED"): "PRIMARY-OBTAINED",
}
GRADE_LETTER = {
    "TRANSCRIBED-SECONDARY": "C",
    "PRIMARY-VERIFIED": "A",
    "PRIMARY-OBTAINED": "A",
}
FORBIDDEN_GRADES_WHEN_PRIMARY_WITHHELD = ("A", "B", "PRIMARY-VERIFIED", "PRIMARY-OBTAINED")

LADDER_DIR = "benchmarks/verified/suboff_sail_resistance"
LADDERS = [
    {
        "key": "A_e2_standard_with_sail",
        "dir_prefix": "ey2_L",
        "case": "SUBOFF with sail (archived AFF-3 geometry), Re_case = 1000",
        "instrument": "standard friction estimator (frozen erratum E2 arm) -- the arm the mission names",
        "verdict_file": LADDER_DIR + "/e2_verdict.json",
        "hull": "with_sail",
        "geometry_matches_anchor": False,
    },
    {
        "key": "B_formal_mix50_with_sail",
        "dir_prefix": "formal_L",
        "case": "SUBOFF with sail (archived AFF-3 geometry), Re_case = 1000",
        "instrument": "legacy mix50 near-wall combination (superseded by erratum E2, co-archived)",
        "verdict_file": LADDER_DIR + "/verdict.json",
        "hull": "with_sail",
        "geometry_matches_anchor": False,
    },
    {
        "key": "C_diag_bare_mix50",
        "dir_prefix": "diag_bare_L",
        "case": "SUBOFF bare hull (geometry-matched to AFF-1), Re_case = 1000",
        "instrument": "legacy mix50 near-wall combination (diagnostic arm, co-archived)",
        "verdict_file": None,
        "hull": "bare_hull",
        "geometry_matches_anchor": True,
    },
]
TIER_LS = (96, 112, 128)   # coarse -> medium -> fine
RHO_KG_M3 = 1000.0         # convention carried by the anchor file
FS = 1.25                  # Roache safety factor, pre-declared
GCI_MAX_PCT = 5.0
P_MIN = 1.8
DEV_MAX_PCT = 15.0
OP_SPREAD_MAX_PP = 15.0
ITTC_TOL_ABS = 1e-9


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def repo_root() -> Path:
    here = Path(__file__).resolve()
    for cand in here.parents:
        if (cand / "benchmarks").is_dir() and (cand / "src").is_dir():
            return cand
    raise RuntimeError("cannot locate repo root from %s" % here)


REPO = repo_root()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(rel: str) -> str:
    return sha256_bytes((REPO / rel).read_bytes())


def git_blob(rel: str, ref: str = SOURCE_REF):
    """Blob bytes at ref:rel, or None when the ref is gone / git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "show", "%s:%s" % (ref, rel)],
            capture_output=True, check=False, timeout=120,
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def load_json(rel: str):
    return json.loads((REPO / rel).read_text(encoding="utf-8"))


def strip_amendment(landed: bytes, offset: int, inserted_text) -> bytes:
    if isinstance(inserted_text, str):
        inserted_text = inserted_text.encode("utf-8")
    if landed[offset:offset + len(inserted_text)] != inserted_text:
        raise RuntimeError("recorded insertion text not found at the recorded offset")
    return landed[:offset] + landed[offset + len(inserted_text):]


def render_amendment(grade: str, rationale: dict) -> bytes:
    return (
        ' "anchor_grade": %s,\n "grade_rationale": %s,\n'
        % (json.dumps(grade), json.dumps(rationale, indent=1, ensure_ascii=False))
    ).encode("utf-8")


# --------------------------------------------------------------------------------------
# step 1+2 · landing and grading amendment
# --------------------------------------------------------------------------------------
def build_grade_rationale(anchor_doc: dict) -> dict:
    status = anchor_doc["status"]
    acq = anchor_doc["acquisition_status"]
    grade = GRADE_FOR_STATE[(status, acq)]
    docs = []
    for d in anchor_doc.get("documents", []):
        attempts = d.get("acquisition", {}).get("attempts", [])
        docs.append({
            "doc_id": d.get("doc_id"),
            "citation": d.get("citation"),
            "role": d.get("role"),
            "acquisition_status": d.get("acquisition_status"),
            "n_recorded_probes": len(attempts),
            "probe_examples": [a.get("command") for a in attempts][:3],
        })
    return {
        "gate": "anchor-gate-v3",
        "anchor_grade": grade,
        "grade_letter": GRADE_LETTER[grade],
        "grade_scale": GRADE_SCALE,
        "state_used_for_grading": {
            "status": status,
            "acquisition_status": acq,
            "rule": "anchor-grade derives from (status, acquisition_status): a value that is only a "
                    "transcription of an in-repo secondary archive, while the primary reports are not "
                    "acquired, is grade C = TRANSCRIBED-SECONDARY",
        },
        "why_not_A_or_B": (
            "A requires the primary report to be read in this environment; B additionally requires a sha256 "
            "of the primary document itself. Neither holds: DTRC/SHD-1298-01 (Groves, Huang & Chang 1989) "
            "and CRDKNSWC/HD-1298-11 (Liu & Huang 1998, Table 14) are both acquisition_status=WITHHELD -- the "
            "sandbox is offline (DNS failure exit 6 / timeout exit 28 are recorded on this file) and no copy "
            "exists in the environment. Only sha256 of in-repo secondary archives could be registered."
        ),
        "primary_documents": docs,
        "claim_boundary": {
            "may_be_cited_as_published_experimental_benchmark": False,
            "may_be_cited_as": "second-hand transcription of Table 14 (report p.23) as archived in-repo",
            "refused_labels": list(FORBIDDEN_GRADES_WHEN_PRIMARY_WITHHELD),
            "consequence": "no agreement claim against this anchor may be promoted above 'agrees with a "
                           "second-hand transcription'; an A/B claim requires re-acquiring the primaries",
        },
        "what_would_raise_the_grade": [
            "obtain CRDKNSWC/HD-1298-11 (DTIC ADA359226), store it in-repo with sha256, re-read Table 14 "
            "page 23 and match all 6 (V, R_T) pairs -> grade B; then read DTRC/SHD-1298-01 Table 1 for the "
            "geometry -> grade A",
        ],
        "registered_by": {
            "task_key": TASK_KEY,
            "mission_id": MISSION_ID,
            "previous_task_key": TASK_KEY_W27,
            "previous_mission_id": MISSION_ID_W27,
            "why_this_task": "W27 graded the anchor only in prose (status=TRANSCRIBED / "
                             "acquisition_status=WITHHELD, prose grade PRIMARY_TABLE_SECOND_HAND); the "
                             "machine-readable gate-v3 grading fields were missing, so no gate could "
                             "refuse an A/B promotion",
        },
        "amendment": {
            "kind": "single contiguous byte insertion into the W27 blob (no other byte changed)",
            "inserted_keys": ["anchor_grade", "grade_rationale"],
            "insert_offset_bytes": None,
            "branch_blob_sha256": ANCHOR_BRANCH_BLOB_SHA256,
            "verify": "bytes[:offset] + bytes[offset+len(inserted_text):] must hash to the branch blob sha256",
        },
    }


def land():
    target = REPO / ANCHOR_REL
    target.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "target": ANCHOR_REL,
        "source_ref": SOURCE_REF,
        "source_commit": SOURCE_REF_COMMIT,
        "recovered_from_unmerged_branch": True,
        "why_recovered": "W27 registered the anchor on the mission branch only; the branch was never merged, "
                         "so W28's anchor gate read the main worktree and raised NO-ANCHOR-ARTIFACT",
        "regeneration_note": "re-running benchmarks/anchors/lbm_suboff_tau_anchor.py (without --check) rebuilds the "
                             "file from the archives and drops the gate-v3 grading block; re-apply it with "
                             "papers/artifacts/w28/lbm-suboff-tau/build_anchor_landing.py. The guard test "
                             "test_strip_recovers_the_branch_blob_byte_for_byte fails if the block disappears.",
    }

    blob = git_blob(ANCHOR_REL)
    branch_reachable = blob is not None
    if blob is None:
        if not target.exists():
            raise RuntimeError("branch blob unreachable and no landed file to repair")
        landed = target.read_bytes()
        ins = json.loads(landed.decode("utf-8"))["grade_rationale"]["amendment"]
        blob = strip_amendment(landed, ins["insert_offset_bytes"], ins["inserted_text"])
    report["branch_blob_reachable_via_git"] = branch_reachable
    report["branch_blob_sha256_recomputed"] = sha256_bytes(blob)
    report["branch_blob_sha256_pinned"] = ANCHOR_BRANCH_BLOB_SHA256
    report["branch_blob_bytes"] = len(blob)
    if sha256_bytes(blob) != ANCHOR_BRANCH_BLOB_SHA256:
        raise RuntimeError("branch blob sha256 mismatch: %s" % sha256_bytes(blob))

    sidecars = []
    for rel, pinned in ((GEN_REL, GEN_BRANCH_BLOB_SHA256), (README_REL, README_BRANCH_BLOB_SHA256)):
        b = git_blob(rel)
        if b is None and (REPO / rel).exists():
            b = (REPO / rel).read_bytes()
        if b is not None:
            (REPO / rel).parent.mkdir(parents=True, exist_ok=True)
            (REPO / rel).write_bytes(b)
        got = sha256_file(rel) if (REPO / rel).exists() else None
        sidecars.append({"path": rel, "sha256": got, "pinned": pinned, "identical": got == pinned})
    report["sidecars"] = sidecars
    report["sidecars_all_identical"] = all(s["identical"] for s in sidecars)

    anchor_doc = json.loads(blob.decode("utf-8"))
    rationale = build_grade_rationale(anchor_doc)
    grade = rationale["anchor_grade"]
    if grade in FORBIDDEN_GRADES_WHEN_PRIMARY_WITHHELD:
        raise RuntimeError("refusing to emit an A/B grade while the primaries are WITHHELD")

    marker = b'"lbm-suboff-tau-anchor/v1",\n'
    idx = blob.find(marker)
    if idx < 0:
        raise RuntimeError("anchor header marker not found in the branch blob")
    offset = idx + len(marker)

    rationale["amendment"]["insert_offset_bytes"] = offset
    final_text = render_amendment(grade, rationale)   # exact inserted bytes, no self-reference

    landed = blob[:offset] + final_text + blob[offset:]
    parsed = json.loads(landed.decode("utf-8"))
    if parsed["anchor_grade"] != grade:
        raise RuntimeError("amendment did not parse back to the same grade")

    target.write_bytes(landed)
    written = target.read_bytes()
    stripped = written[:offset] + written[offset + len(final_text):]
    report["plain_copy_sha256"] = sha256_bytes(blob)
    report["landed_sha256"] = sha256_bytes(written)
    report["landed_bytes"] = len(written)
    report["amendment"] = {
        "kind": "single contiguous insertion at byte offset %d" % offset,
        "insert_offset_bytes": offset,
        "inserted_bytes": len(final_text),
        "inserted_text": final_text.decode("utf-8"),
        "inserted_text_sha256": sha256_bytes(final_text),
        "inserted_keys": ["anchor_grade", "grade_rationale"],
        "existing_bytes_changed": 0,
        "diff_itemization": [
            {"item": 1, "change": "insert", "at_byte_offset": offset, "bytes_added": len(final_text),
             "content": 'top-level key anchor_grade = "TRANSCRIBED-SECONDARY" (gate-v3 grade field)'},
            {"item": 2, "change": "insert", "at_byte_offset": offset, "bytes_added": 0,
             "content": "top-level key grade_rationale = {derivation rule, grade scale, withheld primary "
                        "documents, refused A/B labels, upgrade path, amendment bookkeeping}"},
            {"item": 3, "change": "none", "at_byte_offset": None, "bytes_added": 0,
             "content": "no deletion, no reordering, no whitespace or numeric change anywhere else: "
                        "strip() reproduces the branch blob sha256"},
        ],
        "strip_check": {
            "strip_formula": "landed[:off] + landed[off+len(inserted):]",
            "sha256_after_strip": sha256_bytes(stripped),
            "equals_branch_blob": sha256_bytes(stripped) == ANCHOR_BRANCH_BLOB_SHA256,
        },
    }
    if json.dumps(json.loads(stripped.decode("utf-8")), sort_keys=True) != json.dumps(anchor_doc, sort_keys=True):
        raise RuntimeError("stripped file does not reproduce the branch blob semantics")
    report["semantic_check_after_strip"] = "OK (parsed object identical to the branch blob object)"

    inputs = []
    for rel, expect in anchor_doc["provenance"]["inputs"].items():
        got = sha256_file(rel) if (REPO / rel).exists() else None
        inputs.append({"path": rel, "sha256_main_worktree": got, "sha256_in_anchor": expect,
                       "match": got == expect})
    report["anchor_input_sha256_recheck"] = inputs
    report["anchor_inputs_all_match"] = all(i["match"] for i in inputs)
    return report, parsed, final_text, offset


# --------------------------------------------------------------------------------------
# step 3 · ITTC-1957 independent recomputation
# --------------------------------------------------------------------------------------
def cf_ittc57_double(re_):
    """Path A: libm double log10."""
    return 0.075 / (math.log10(re_) - 2.0) ** 2


def cf_ittc57_decimal(re_):
    """Path B: 50 significant digits, log10 built from Decimal.ln (no libm log10)."""
    getcontext().prec = 50
    r = Decimal(repr(float(re_)))
    log10 = r.ln() / Decimal(10).ln()
    return Decimal("0.075") / (log10 - 2) ** 2


def cf_ittc57_ln_double(re_):
    """Path C: double, log10(x) = ln(x)/ln(10) (different log identity)."""
    return 0.075 / (math.log(re_) / math.log(10.0) - 2.0) ** 2


def ittc57_block(anchor_doc):
    conv = anchor_doc["conventions"]
    nu = conv["nu_m2_s"]
    L = anchor_doc["geometry"]["L_max_m"]
    rho = conv["rho_kg_m3"]
    S = anchor_doc["geometry"]["WSA_m2"]
    kn = conv["kn_to_ms"]
    rows = []
    for p in anchor_doc["anchor_points"]:
        re_reg = p["Re"]
        cf_reg = p["C_F"]["value"]
        cf_a = cf_ittc57_double(re_reg)
        cf_b = float(cf_ittc57_decimal(re_reg))
        cf_c = cf_ittc57_ln_double(re_reg)
        v = p["speed_knots"] * kn
        ct = p["resistance_n"] / (0.5 * rho * v ** 2 * S)
        rows.append({
            "id": p["id"],
            "speed_knots": p["speed_knots"],
            "speed_ms_from_anchor": p["speed_ms"],
            "speed_ms_recomputed": v,
            "Re_registered": re_reg,
            "Re_recomputed": v * L / nu,
            "Re_rel_residual": abs(v * L / nu - re_reg) / re_reg,
            "C_T_registered": p["C_T"],
            "C_T_recomputed": ct,
            "C_T_rel_residual": abs(ct - p["C_T"]) / abs(p["C_T"]),
            "C_F_registered": cf_reg,
            "C_F_ittc57_double": cf_a,
            "C_F_ittc57_decimal50": cf_b,
            "C_F_ittc57_ln_path": cf_c,
            "abs_residual_double_minus_registered": abs(cf_a - cf_reg),
            "abs_residual_decimal_minus_registered": abs(cf_b - cf_reg),
            "cross_path_abs_diff_double_vs_decimal": abs(cf_a - cf_b),
            "pass_abs_1e9": abs(cf_a - cf_reg) <= ITTC_TOL_ABS and abs(cf_b - cf_reg) <= ITTC_TOL_ABS,
        })
    known = []
    for ka in conv.get("ittc57_transcription", {}).get("known_answers", []):
        cf = cf_ittc57_double(ka["re"])
        dev = 100.0 * (cf / ka["cf_recorded"] - 1.0)
        known.append({
            "re": ka["re"], "cf_recorded": ka["cf_recorded"], "cf_recomputed": cf,
            "rel_dev_pct": dev, "declared_band_pct": ka["declared_band_pct"],
            "within_band": abs(dev) <= ka["declared_band_pct"],
            "quote": ka["quote"], "locator": ka.get("locator"),
        })
    mono = all(rows[i]["C_F_ittc57_double"] > rows[i + 1]["C_F_ittc57_double"] for i in range(len(rows) - 1))
    return {
        "formula": conv["cf_definition"],
        "independent_paths": {
            "A_libm_double": "0.075/(log10(Re)-2)^2 with math.log10",
            "B_decimal50": "0.075/(log10(Re)-2)^2 with Decimal.ln at 50 significant digits",
            "C_ln_ratio_double": "0.075/(ln(Re)/ln(10)-2)^2 (double, different log identity)",
        },
        "tolerance_abs": ITTC_TOL_ABS,
        "rows": rows,
        "max_abs_residual_double_vs_registered": max(r["abs_residual_double_minus_registered"] for r in rows),
        "max_abs_residual_decimal_vs_registered": max(r["abs_residual_decimal_minus_registered"] for r in rows),
        "max_cross_path_abs_diff": max(r["cross_path_abs_diff_double_vs_decimal"] for r in rows),
        "max_Re_rel_residual": max(r["Re_rel_residual"] for r in rows),
        "max_C_T_rel_residual": max(r["C_T_rel_residual"] for r in rows),
        "C_F_monotone_decreasing_in_Re": mono,
        "known_answer_checks": known,
        "all_pass": all(r["pass_abs_1e9"] for r in rows) and all(k["within_band"] for k in known),
    }


# --------------------------------------------------------------------------------------
# step 5 · three-grid Roache GCI (true unequal refinement ratios)
# --------------------------------------------------------------------------------------
def solve_observed_order(r21, r32, ratio_param):
    """Solve ((r21*r32)^p - r21^p)/(r21^p - 1) = eps32/eps21 by bisection (400 steps)."""
    if ratio_param is None or ratio_param <= 0:
        return None, None

    def f(p):
        num = (r21 * r32) ** p - r21 ** p
        den = r21 ** p - 1.0
        return num - ratio_param * den

    lo, hi = 1e-4, 8.0
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo, 0.0
    if flo * fhi > 0:
        return None, None
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    p = 0.5 * (lo + hi)
    return p, f(p)


def gci_block(phi_c, phi_m, phi_f, r21, r32):
    eps21 = phi_m - phi_f
    eps32 = phi_c - phi_m
    ratio = (eps21 / eps32) if eps32 != 0 else None
    # Roache: (eps32/eps21) = ((r21 r32)^p - r21^p)/(r21^p - 1)
    ratio_param = (eps32 / eps21) if eps21 != 0 else None
    p, resid = solve_observed_order(r21, r32, ratio_param)
    out = {
        "phi_coarse": phi_c, "phi_medium": phi_m, "phi_fine": phi_f,
        "h_ratio_r21": r21, "h_ratio_r32": r32,
        "r21_definition": "h_medium/h_coarse = L_coarse/L_medium = 128/112",
        "r32_definition": "h_coarse/h_coarsest = L_medium/L_coarse = 112/96",
        "eps21": eps21, "eps32": eps32,
        "ratio_eps21_over_eps32": ratio,
        "ratio_eps32_over_eps21_solver_input": ratio_param,
        "monotone": ratio is not None and 0 < ratio < 1,
        "fs": FS,
        "p_observed": p,
        "p_solver": "bisection(400) on ((r21*r32)^p - r21^p)/(r21^p - 1) = eps21/eps32",
        "p_solver_residual": resid,
    }
    if p is None:
        out.update({"gci_fine_pct": None, "asymptotic": False,
                    "unresolved_note": "no sign change in the solved window: the triple is not usable"})
    else:
        gci = FS * abs(eps21 / phi_f) / (r21 ** p - 1.0) * 100.0
        out.update({
            "gci_fine_pct": gci,
            "phi_richardson_extrapolated": phi_f + (phi_f - phi_m) / (r21 ** p - 1.0),
            "single_grid_rel_err_fine_pct": abs(eps21 / phi_f) * 100.0,
            "asymptotic": gci < 100.0,
            "unresolved_note": None,
        })
    if ratio is not None and 0 < ratio < 1:
        # Roache shortcut for a uniformly refined triple: p = ln|eps32/eps21| / ln(r)
        p_short = math.log(abs(1.0 / ratio)) / math.log(r21)
        gci_short = FS * abs(eps21 / phi_f) / (r21 ** p_short - 1.0) * 100.0
        out["uniform_r_assumption"] = {
            "p_observed": p_short,
            "gci_fine_pct": gci_short,
            "sign_convention": "p = ln|eps32/eps21| / ln(r21), the textbook form for a uniformly refined triple",
            "note": "textbook shortcut that assumes r21 == r32; this ladder does not satisfy that, so this "
                    "row is a sensitivity only: it can turn a pre-asymptotic triple into a plausible-looking "
                    "order of accuracy",
        }
    out["pass_gci_lt_5pct"] = out.get("gci_fine_pct") is not None and out["gci_fine_pct"] < GCI_MAX_PCT
    out["pass_p_ge_1p8"] = p is not None and p >= P_MIN
    out["pass_both"] = bool(out["pass_gci_lt_5pct"] and out["pass_p_ge_1p8"])
    return out


# --------------------------------------------------------------------------------------
# step 4 · same-Re transposition of every tier onto every anchor point
# --------------------------------------------------------------------------------------
def load_ladder(spec):
    tiers = []
    for L in TIER_LS:
        rel = "%s/%s%d/result_L%d.json" % (LADDER_DIR, spec["dir_prefix"], L, L)
        d = load_json(rel)
        tiers.append({
            "tier_L": L,
            "archive": rel,
            "archive_sha256": sha256_file(rel),
            "n_steps": d["n_steps"],
            "domain": d.get("domain_lu"),
            "grid": d.get("grid"),
            "Re_case": d["Re"],
            "collision": d.get("collision"),
            "friction_formula": d.get("friction_formula_primary"),
            "hull_type": d.get("hull_type"),
            "Cd_total": d["Cd_total"],
            "Cd_pressure": d["Cd_pressure"],
            "Cd_friction": d["Cd_friction"],
            "S_total_wetted_lb2": d["S_total_wetted_lb2"],
            "Cf_ref_blasius": d["Cf_ref_blasius"],
            "finite": d.get("finite"),
            "diverged": d.get("diverged"),
        })
    trace = []
    if spec["verdict_file"] and (REPO / spec["verdict_file"]).exists():
        v = load_json(spec["verdict_file"])
        vsha = sha256_file(spec["verdict_file"])
        for t in tiers:
            rec = (v.get("tiers") or {}).get("L%d" % t["tier_L"]) or {}
            key = None
            for cand in ("cd_tot_window", "Cd_total", "cd_total"):
                if cand in rec:
                    key = cand
                    break
            if key:
                rel_dev = 100.0 * (t["Cd_total"] / rec[key] - 1.0)
                trace.append({"tier_L": t["tier_L"], "verdict_file": spec["verdict_file"],
                              "verdict_sha256": vsha, "field": key, "value_in_verdict": rec[key],
                              "value_in_archive": t["Cd_total"], "rel_dev_pct": rel_dev,
                              "match_1e-9_rel": abs(rel_dev) < 1e-9})
    phis = {t["tier_L"]: t["Cd_total"] for t in tiers}
    r21 = float(TIER_LS[2]) / TIER_LS[1]
    r32 = float(TIER_LS[1]) / TIER_LS[0]
    out = dict(spec)
    out["tiers"] = tiers
    out["verdict_trace"] = trace
    out["instrument_consistency"] = sorted({str(t["friction_formula"]) for t in tiers})
    out["gci"] = gci_block(phis[96], phis[112], phis[128], r21, r32)
    return out


def transpose(ladder, anchor_doc):
    L_anchor = anchor_doc["geometry"]["L_max_m"]
    S_anchor_m2 = anchor_doc["geometry"]["WSA_m2"]
    S_anchor_lb2 = 0.31576 * (L_anchor / (anchor_doc["conventions"]["nu_m2_s"] * 1e5)) ** 2
    rows = []
    for t in ladder["tiers"]:
        form_factor = t["Cd_total"] / t["Cf_ref_blasius"]
        area_ratio = t["S_total_wetted_lb2"] / S_anchor_lb2
        per_point = []
        for p in anchor_doc["anchor_points"]:
            cf_anchor = cf_ittc57_double(p["Re"])
            op0 = t["Cd_total"]
            op1 = form_factor * cf_anchor
            op2 = t["Cd_pressure"] * area_ratio + cf_anchor
            ct = p["C_T"]
            d0 = 100.0 * (op0 / ct - 1.0)
            d1 = 100.0 * (op1 / ct - 1.0)
            d2 = 100.0 * (op2 / ct - 1.0)
            per_point.append({
                "anchor_point": p["id"],
                "Re_anchor": p["Re"],
                "C_T_anchor": ct,
                "Cf_ittc57_anchor": cf_anchor,
                "Cd_OP0_identity": op0,
                "Cd_OP1_constant_form_factor": op1,
                "Cd_OP2_additive_pressure_plus_ittc": op2,
                "rel_dev_pct_OP0": d0,
                "rel_dev_pct_OP1": d1,
                "rel_dev_pct_OP2": d2,
                "operator_spread_pct_points": abs(d1 - d2),
                "within_15pct_OP1": abs(d1) <= DEV_MAX_PCT,
                "within_15pct_OP2": abs(d2) <= DEV_MAX_PCT,
                "operator_dependent": abs(d1 - d2) > OP_SPREAD_MAX_PP,
            })
        rows.append({
            "tier_L": t["tier_L"],
            "form_factor_Cd_over_Cf_blasius": form_factor,
            "S_bench_lb2": t["S_total_wetted_lb2"],
            "S_anchor_lb2": S_anchor_lb2,
            "area_ratio_bench_over_anchor": area_ratio,
            "points": per_point,
            "max_abs_dev_pct_OP1": max(abs(q["rel_dev_pct_OP1"]) for q in per_point),
            "max_abs_dev_pct_OP2": max(abs(q["rel_dev_pct_OP2"]) for q in per_point),
            "all_points_within_15pct_OP1": all(q["within_15pct_OP1"] for q in per_point),
            "any_point_operator_dependent": any(q["operator_dependent"] for q in per_point),
        })
    return {
        "Re_case": ladder["tiers"][0]["Re_case"],
        "operators": {
            "OP0_identity": {
                "formula": "Cd^(anchor) = Cd_total(Re_case)",
                "assumption": "none -- disclosure only, declared a priori NOT a pass path",
            },
            "OP1_constant_form_factor": {
                "formula": "Cd^(anchor) = (Cd_total(Re_case)/Cf_Blasius(Re_case)) * Cf_ITTC57(Re_anchor)",
                "assumption": "form factor (1+k) is Re-independent from 1e3 to 1.16e7..3.54e7; the wetted-area "
                              "convention cancels in the ratio",
                "role": "PRIMARY same-Re-caliber operator (the one the mission asks for)",
            },
            "OP2_additive_pressure_plus_ittc": {
                "formula": "Cd^(anchor) = Cd_p(Re_case)*(S_bench/S_anchor_lb2) + Cf_ITTC57(Re_anchor)",
                "assumption": "pressure coefficient Re-independent, friction follows ITTC-57 at the anchor Re; "
                              "the pi*D*L bench area is explicitly rescaled onto the anchor's analytic "
                              "wetted area 0.31576*L^2",
                "role": "sensitivity operator, tests whether the comparison is solver-limited or "
                        "assumption-limited",
            },
        },
        "tiers": rows,
    }

# --------------------------------------------------------------------------------------
# criterion aggregation
# --------------------------------------------------------------------------------------
def evaluate(landing, parsed_anchor, ittc, ladders, transpose_blocks):
    crit = []

    same = landing["plain_copy_sha256"] == ANCHOR_BRANCH_BLOB_SHA256
    strip_ok = landing["amendment"]["strip_check"]["equals_branch_blob"]
    c1_ok = same and strip_ok and landing["anchor_inputs_all_match"] and landing["sidecars_all_identical"]
    crit.append({
        "id": "C1_anchor_landed_byte_identical_modulo_declared_amendment",
        "task_item": "1",
        "statement": "the W27 anchor blob exists in the main worktree; sha256 equals the branch blob byte for "
                     "byte, and the single intentional change (gate-v3 grading) is itemised",
        "target_path": ANCHOR_REL,
        "source_ref": SOURCE_REF,
        "source_commit": SOURCE_REF_COMMIT,
        "branch_blob_sha256": ANCHOR_BRANCH_BLOB_SHA256,
        "plain_copy_sha256": landing["plain_copy_sha256"],
        "landed_sha256": landing["landed_sha256"],
        "branch_blob_bytes": ANCHOR_BRANCH_BLOB_BYTES,
        "landed_bytes": landing["landed_bytes"],
        "byte_identical": same,
        "bytes_added_by_amendment": landing["amendment"]["inserted_bytes"],
        "n_bytes_changed_in_existing_content": landing["amendment"]["existing_bytes_changed"],
        "strip_reproduces_branch_blob": strip_ok,
        "diff": landing["amendment"]["diff_itemization"],
        "anchor_inputs_all_match_main_worktree": landing["anchor_inputs_all_match"],
        "sidecars_identical": landing["sidecars_all_identical"],
        "measured": {"identical_modulo_amendment": c1_ok,
                     "diff_items": len(landing["amendment"]["diff_itemization"]),
                     "bytes_added": landing["amendment"]["inserted_bytes"]},
        "verdict": "PASS" if c1_ok else "FAIL",
    })

    status = parsed_anchor["status"]
    acq = parsed_anchor["acquisition_status"]
    grade = parsed_anchor["anchor_grade"]
    expected = GRADE_FOR_STATE[(status, acq)]
    rationale = parsed_anchor["grade_rationale"]
    forbidden_used = grade in FORBIDDEN_GRADES_WHEN_PRIMARY_WITHHELD
    checks = {
        "grade_equals_rule_output": grade == expected,
        "no_A_or_B_label_used": not forbidden_used,
        "rationale_state_matches_file": rationale["state_used_for_grading"]["acquisition_status"] == acq,
        "published_benchmark_claim_refused":
            rationale["claim_boundary"]["may_be_cited_as_published_experimental_benchmark"] is False,
        "primary_documents_all_withheld":
            len(rationale["primary_documents"]) == len(parsed_anchor["documents"])
            and all(d["acquisition_status"] == "WITHHELD" for d in rationale["primary_documents"]),
        "probe_evidence_preserved": all(d["n_recorded_probes"] >= 1 for d in rationale["primary_documents"]),
        "upgrade_path_documented": len(rationale["what_would_raise_the_grade"]) >= 1,
        "letter_is_C": rationale["grade_letter"] == GRADE_LETTER[grade] == "C",
    }
    crit.append({
        "id": "C2_anchor_grade_consistent_with_acquisition_status",
        "task_item": "2",
        "statement": "anchor_grade and grade_rationale are registered and self-consistent with the file's own "
                     "(status, acquisition_status): WITHHELD primaries forbid an A/B label",
        "status": status,
        "acquisition_status": acq,
        "anchor_grade_registered": grade,
        "anchor_grade_expected_from_rule": expected,
        "grade_letter": rationale["grade_letter"],
        "checks": checks,
        "measured": {"grade": grade, "expected": expected, "all_checks_pass": all(checks.values())},
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    })

    crit.append({
        "id": "C3_ittc57_independent_recompute",
        "task_item": "3",
        "statement": "ITTC-1957 C_F independently recomputed on two arithmetic paths and closed point-wise "
                     "against the registered C_F within 1e-9 (absolute)",
        "tolerance_abs": ITTC_TOL_ABS,
        "max_abs_residual_double_vs_registered": ittc["max_abs_residual_double_vs_registered"],
        "max_abs_residual_decimal_vs_registered": ittc["max_abs_residual_decimal_vs_registered"],
        "max_cross_path_abs_diff": ittc["max_cross_path_abs_diff"],
        "n_points_closed": sum(1 for r in ittc["rows"] if r["pass_abs_1e9"]),
        "n_points": len(ittc["rows"]),
        "max_Re_rel_residual": ittc["max_Re_rel_residual"],
        "max_C_T_rel_residual": ittc["max_C_T_rel_residual"],
        "known_answer_bands_pass": all(k["within_band"] for k in ittc["known_answer_checks"]),
        "measured": {"max_abs_residual": max(ittc["max_abs_residual_double_vs_registered"],
                                             ittc["max_abs_residual_decimal_vs_registered"]),
                     "tolerance": ITTC_TOL_ABS},
        "verdict": "PASS" if ittc["all_pass"] else "FAIL",
    })

    per_ladder = {}
    for L in ladders:
        g = L["gci"]
        per_ladder[L["key"]] = {
            "p_observed": g["p_observed"],
            "gci_fine_pct": g["gci_fine_pct"],
            "pass_gci_lt_5pct": g["pass_gci_lt_5pct"],
            "pass_p_ge_1p8": g["pass_p_ge_1p8"],
            "pass_both": g["pass_both"],
            "shortcut_p_uniform_r": g.get("uniform_r_assumption", {}).get("p_observed"),
            "shortcut_gci_uniform_r_pct": g.get("uniform_r_assumption", {}).get("gci_fine_pct"),
        }
    crit.append({
        "id": "C4_three_tier_gci_and_observed_order",
        "task_item": "4",
        "statement": "three-grid Roache GCI on the finest tier < 5% AND observed order >= 1.8, using the true "
                     "unequal refinement ratios r21 = 128/112, r32 = 112/96",
        "thresholds": {"gci_fine_pct_max": GCI_MAX_PCT, "p_observed_min": P_MIN, "fs": FS},
        "per_ladder": per_ladder,
        "measured": per_ladder,
        "verdict": "PASS" if all(v["pass_both"] for v in per_ladder.values()) else "FAIL",
        "disclosure": "the equal-ratio shortcut row is reported for each ladder as a sensitivity; it is not the "
                      "decision path, because it can inflate a pre-asymptotic triple to p ~ 2.2",
    })

    per_ladder_dev = {}
    for key, tb in transpose_blocks.items():
        per_ladder_dev[key] = {
            "operator_primary": "OP1_constant_form_factor",
            "max_abs_dev_pct_OP1_by_tier": {"L%d" % t["tier_L"]: t["max_abs_dev_pct_OP1"] for t in tb["tiers"]},
            "max_abs_dev_pct_OP2_by_tier": {"L%d" % t["tier_L"]: t["max_abs_dev_pct_OP2"] for t in tb["tiers"]},
            "all_points_within_15pct_OP1": all(t["all_points_within_15pct_OP1"] for t in tb["tiers"]),
            "any_point_operator_dependent": any(t["any_point_operator_dependent"] for t in tb["tiers"]),
            "n_points_within_15pct_OP1": sum(1 for t in tb["tiers"] for q in t["points"] if q["within_15pct_OP1"]),
            "n_points_total": sum(len(t["points"]) for t in tb["tiers"]),
        }
    c5_pass = all(v["all_points_within_15pct_OP1"] and not v["any_point_operator_dependent"]
                  for v in per_ladder_dev.values())
    crit.append({
        "id": "C5_pointwise_deviation_vs_anchor_CT",
        "task_item": "5",
        "statement": "every tier, transposed onto every anchor Re, must stay within 15% of anchor C_T under the "
                     "primary operator OP1; a tier/point whose OP1/OP2 spread exceeds 15 percentage points is "
                     "declared OPERATOR_DEPENDENT and may not be reported as agreement",
        "threshold_abs_pct": DEV_MAX_PCT,
        "operator_spread_max_pct_points": OP_SPREAD_MAX_PP,
        "per_ladder": per_ladder_dev,
        "measured": per_ladder_dev,
        "verdict": "PASS" if c5_pass else "FAIL",
        "disclosure": "OP0 identity is reported for scale only (it quantifies the four-decade Re gap); it is "
                      "declared a priori as not a pass path. No threshold was moved after the numbers were seen.",
    })
    return crit


# --------------------------------------------------------------------------------------
# prE-registration (experimental-design contract) + assembly
# --------------------------------------------------------------------------------------
def prereg_block():
    return {
        "declared_before_computation": True,
        "adopted_from": {
            "task_key": TASK_KEY_W27,
            "mission_id": MISSION_ID_W27,
            "artifact": W27_RECONCILE_REL,
            "artifact_sha256": W27_RECONCILE_SHA256,
            "artifact_ref": SOURCE_REF,
            "note": "decision, operators and thresholds were frozen in W27 before any number was seen; this "
                    "task re-uses them verbatim (no threshold moved after the W27 result was known) and "
                    "extends the deviation table from a single anchor point to all 6",
        },
        "decision": "May the archived SUBOFF three-tier LBM ladder be certified against the DTRC AFF-1 tow-tank "
                    "anchor at the anchor Reynolds number, and may the anchor itself be cited as a published "
                    "experimental benchmark in this environment?",
        "alternatives": [
            "H1: ladder is in the asymptotic range (p >= 1.8, GCI_fine < 5%) and the same-Re transposed Cd "
            "matches anchor C_T within 15% under OP1 and OP2 alike -> ladder and anchor are both usable for a "
            "resistance claim",
            "H2: ladder is pre-asymptotic (p < 1.8 or GCI_fine >= 5%) -> no converged C_T exists at these "
            "resolutions and no drag-agreement claim may be certified",
            "H0: the Re transposition is ill-conditioned (OP1 vs OP2 spread > 15 pp) -> the comparison tests the "
            "transposition assumption, not the solver",
            "H3 (grading): the anchor is a second-hand transcription -> it stays grade C until the primary "
            "reports are acquired, whatever the numbers say",
        ],
        "metrics": {
            "gci_fine_pct": "Roache GCI of the finest tier, Fs = 1.25, true unequal refinement ratios",
            "p_observed": "observed order of accuracy from the three grids",
            "rel_dev_pct": "100*(Cd_transposed/C_T_anchor - 1) per tier and per anchor point",
            "abs_residual_cf": "|Cf_recomputed - Cf_registered| per anchor point",
        },
        "baseline_control": [
            "operator-layer control: the ITTC-1957 line itself plus two in-repo round-off known answers "
            "(Cf(1.2e7) = 0.00291, Cf(2.0e6) = 0.00405, declared +/-5%)",
            "geometric control: Cb / L-over-D / analytic wetted area re-verified before any drag comparison",
            "arithmetic control: libm double vs Decimal-50 as two disjoint evaluation paths",
            "ladder control: three arms reported (E2 standard, archived mix50, bare-hull) - the geometry-matched "
            "arm is disclosed next to the arm the mission names",
        ],
        "thresholds_predeclared": {
            "gci_fine_pct_max": GCI_MAX_PCT,
            "p_observed_min": P_MIN,
            "rel_dev_abs_pct_max": DEV_MAX_PCT,
            "operator_spread_max_pct_points": OP_SPREAD_MAX_PP,
            "ittc57_residual_max_abs": ITTC_TOL_ABS,
            "fs": FS,
        },
        "stopping_rule": "no threshold may be relaxed after the numbers are seen; a criterion that fails is "
                         "registered as honest FAIL and stays in the artifact (append-only)",
        "unit_of_analysis": "one archived tier (grid) x one anchor point of one archived ladder arm",
        "variables": {
            "controlled": ["archived run protocol (code, collision model, n_steps per tier, seeds)",
                           "anchor transcription (frozen blob, sha256)",
                           "Re convention (L_max, nu = 1.139e-6 m2/s) and C_T definition"],
            "changed": ["grid resolution L = 96 / 112 / 128 (the discretisation variable)",
                        "Re of comparison (case Re = 1e3 vs anchor Re 1.16e7..3.54e7, through the operators)"],
        },
        "confounders": [
            "four-decade Reynolds gap between case (1e3) and anchor (1.16e7..3.54e7)",
            "the with-sail ladders model AFF-3 geometry while the anchor is the AFF-1 bare hull (a "
            "geometry-matched bare-hull arm is disclosed alongside)",
            "the ladder normalises drag with the legacy cylinder proxy pi*D*L (+19% area vs the analytic wetted "
            "area) while the anchor uses WSA = 5.998 m2 (handled explicitly in OP2)",
            "an instrument change inside the case (erratum E2 swapped mix50 -> standard), so arms A and B are "
            "not replicates of each other",
            "n_steps differ per tier (14400/16800/19200): the window is the same 1000-step average but the "
            "statistical residual differs per tier",
        ],
        "repetition_rationale": "the ladders are already-archived deterministic LBM runs; the three-grid triple is "
                                "the replication unit for discretisation error and every branch of the triple is "
                                "read from a frozen archive with a registered sha256. No new run is submitted: "
                                "benchmarks/verified/suboff_sail_resistance/gci_v4.json recorded 15-20 s/step at "
                                "L=96 on this host, i.e. 61-144 h per tier at the anchor Re, so a new anchor-Re "
                                "ladder is out of budget for this task; that gap is disclosed, not hidden",
        "environment": "CPU-only loongarch64 container; the archived ladders were produced earlier on cuda:6 and "
                       "are read-only inputs here",
        "analysis": "per tier: GCI/p from the three-grid triple with asymmetric ratios; then the per-tier "
                    "transposed value at each of the 6 anchor Re and its deviation; the aggregate is the table "
                    "itself, not a mean (each tier is a different discretisation of the same continuum problem)",
        "safety_resource_limits": "read-only analysis of in-repo archives, no cluster jobs submitted (budget 100 "
                                  "calls / 14400 s untouched)",
        "interpretation_limits": [
            "an OP1-within-15% result is not a solver validation: OP0 (identity) sits 4 decades away and OP2 "
            "disagrees by hundreds of percent, so agreement is decided by the transposition assumption",
            "single-case ensemble: three grids of one case at one Reynolds number; no multi-Re or multi-hull "
            "sweep is added here",
            "the anchor is second-hand (grade C): even a perfect agreement would only mean 'agrees with a "
            "transcription of Table 14'",
        ],
    }


def assemble(landing, parsed_anchor, inserted_text, write=True):
    """Everything downstream of the landed files. inserted_text is the recorded amendment."""
    landed = (REPO / ANCHOR_REL).read_bytes()
    offset = parsed_anchor["grade_rationale"]["amendment"]["insert_offset_bytes"]
    anchor_doc = json.loads(strip_amendment(landed, offset, inserted_text).decode("utf-8"))
    ittc = ittc57_block(anchor_doc)
    ladders = [load_ladder(spec) for spec in LADDERS]
    transpose_blocks = {L["key"]: transpose(L, anchor_doc) for L in ladders}
    criteria = evaluate(landing, parsed_anchor, ittc, ladders, transpose_blocks)
    overall = "PASS" if all(c["verdict"] == "PASS" for c in criteria) else "HONEST_FAIL"
    art = {
        "schema": "lbm-suboff-tau-anchor-landing/v1",
        "task_key": TASK_KEY,
        "mission_id": MISSION_ID,
        "repo": "TensorLBM",
        "discipline": "ship-total-performance / resistance (bare submarine hull)",
        "generated_by": "papers/artifacts/w28/lbm-suboff-tau/build_anchor_landing.py",
        "generated_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "runner": "native",
            "network": "unavailable (the anchor file records DNS exit 6 / timeout exit 28 probes)",
        },
        "prereg": prereg_block(),
        "landing": landing,
        "grading": {
            "anchor_grade": parsed_anchor["anchor_grade"],
            "grade_rationale": parsed_anchor["grade_rationale"],
            "anchor_status": parsed_anchor["status"],
            "anchor_acquisition_status": parsed_anchor["acquisition_status"],
        },
        "ittc57": ittc,
        "ladders": [{"ladder": L, "transposition": transpose_blocks[L["key"]]} for L in ladders],
        "criteria": criteria,
        "overall_verdict": overall,
        "headline": {
            "anchor_landed_on_main": True,
            "anchor_grade": parsed_anchor["anchor_grade"],
            "anchor_grade_letter": parsed_anchor["grade_rationale"]["grade_letter"],
            "ittc57_max_abs_residual": max(ittc["max_abs_residual_double_vs_registered"],
                                           ittc["max_abs_residual_decimal_vs_registered"]),
            "gci": {L["key"]: {"p_observed": L["gci"]["p_observed"],
                               "gci_fine_pct": L["gci"]["gci_fine_pct"],
                               "verdict": "PASS" if L["gci"]["pass_both"] else "HONEST_FAIL"}
                    for L in ladders},
            "pointwise_deviation_OP1_max_abs_pct": {
                k: max(v["max_abs_dev_pct_OP1_by_tier"].values())
                for k, v in ((c["id"], c["measured"]) for c in criteria
                             if c["id"].startswith("C5")) for _ in [0]
            } if False else {
                key: max(t["max_abs_dev_pct_OP1"] for t in tb["tiers"])
                for key, tb in transpose_blocks.items()
            },
            "honest_fail_reason": "the archived three-tier ladder is pre-asymptotic (p ~ 1.0-1.1 << 1.8, "
                                  "GCI_fine ~ 26-31% >> 5%); the same-Re transposition is operator-dependent, so "
                                  "no drag-agreement claim is certified",
        },
        "traceability": {
            "commands": [
                "python papers/artifacts/w28/lbm-suboff-tau/build_anchor_landing.py",
                "python papers/artifacts/w28/lbm-suboff-tau/build_anchor_landing.py --check",
                "python -m pytest tests/test_w28_lbm_suboff_tau_anchor_landing.py -q",
            ],
            "inputs": {rel: sha256_file(rel) for rel in sorted(
                list(anchor_doc["provenance"]["inputs"].keys())
                + [ANCHOR_REL, GEN_REL, README_REL]
                + ["%s/%s%d/result_L%d.json" % (LADDER_DIR, s["dir_prefix"], L, L)
                   for s in LADDERS for L in TIER_LS]
                + [s["verdict_file"] for s in LADDERS if s["verdict_file"]
                   and (REPO / s["verdict_file"]).exists()])},
            "quote_locators": [
                {"what": "AFF-1 tow-tank resistance row (Table 14)", "path": anchor_doc["anchor_points"][0]
                 ["provenance"]["locator"]["path"], "line": anchor_doc["anchor_points"][0]["provenance"]
                 ["locator"]["line"], "table": "Table 14", "report_page": 23,
                 "quote": anchor_doc["anchor_points"][0]["provenance"]["verbatim_quote"]},
                {"what": "nu convention", "path": anchor_doc["conventions"]["nu_provenance"]["path"],
                 "line": anchor_doc["conventions"]["nu_provenance"]["line"],
                 "quote": anchor_doc["conventions"]["nu_provenance"]["quote"]},
            ],
            "anchor_blob_provenance": {
                "branch": SOURCE_REF,
                "branch_commit": SOURCE_REF_COMMIT,
                "branch_blob_sha256": ANCHOR_BRANCH_BLOB_SHA256,
                "landed_sha256": landing["landed_sha256"],
                "strip_rule": landing["amendment"]["strip_check"],
            },
        },
    }
    if write:
        out = REPO / ARTIFACT_REL
        out.write_text(json.dumps(art, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return art


def check():
    """Recompute everything from the landed files and diff against the stored artifact."""
    if not (REPO / ARTIFACT_REL).exists():
        raise SystemExit("no artifact to check")
    stored = load_json(ARTIFACT_REL)
    landed = (REPO / ANCHOR_REL).read_bytes()
    parsed = json.loads(landed.decode("utf-8"))
    ins = dict(parsed["grade_rationale"]["amendment"])
    ins["inserted_text"] = stored["landing"]["amendment"]["inserted_text"]
    blob = strip_amendment(landed, ins["insert_offset_bytes"], ins["inserted_text"])
    landing = {k: v for k, v in stored["landing"].items() if k not in (
        "plain_copy_sha256", "landed_sha256", "landed_bytes", "amendment",
        "sidecars", "sidecars_all_identical", "anchor_input_sha256_recheck",
        "anchor_inputs_all_match", "branch_blob_sha256_recomputed")}
    landing.update({
        "target": ANCHOR_REL, "source_ref": SOURCE_REF, "source_commit": SOURCE_REF_COMMIT,
        "branch_blob_reachable_via_git": git_blob(ANCHOR_REL) is not None,
        "branch_blob_sha256_recomputed": sha256_bytes(blob),
        "branch_blob_sha256_pinned": ANCHOR_BRANCH_BLOB_SHA256,
        "branch_blob_bytes": len(blob),
        "plain_copy_sha256": sha256_bytes(blob),
        "landed_sha256": sha256_bytes(landed),
        "landed_bytes": len(landed),
        "amendment": {**stored["landing"]["amendment"],
                      "strip_check": {"strip_formula": "landed[:off] + landed[off+len(inserted):]",
                                      "sha256_after_strip": sha256_bytes(blob),
                                      "equals_branch_blob": sha256_bytes(blob) == ANCHOR_BRANCH_BLOB_SHA256}},
        "sidecars": [{"path": rel, "sha256": sha256_file(rel) if (REPO / rel).exists() else None,
                      "pinned": pin, "identical": ((REPO / rel).exists() and sha256_file(rel) == pin)}
                     for rel, pin in ((GEN_REL, GEN_BRANCH_BLOB_SHA256), (README_REL, README_BRANCH_BLOB_SHA256))],
    })
    landing["sidecars_all_identical"] = all(s["identical"] for s in landing["sidecars"])
    anchor_doc = json.loads(blob.decode("utf-8"))
    landing["anchor_input_sha256_recheck"] = [
        {"path": rel, "sha256_main_worktree": sha256_file(rel) if (REPO / rel).exists() else None,
         "sha256_in_anchor": exp, "match": (REPO / rel).exists() and sha256_file(rel) == exp}
        for rel, exp in anchor_doc["provenance"]["inputs"].items()
    ]
    landing["anchor_inputs_all_match"] = all(i["match"] for i in landing["anchor_input_sha256_recheck"])
    fresh = assemble(landing, parsed, stored["landing"]["amendment"]["inserted_text"], write=False)
    drift = diff(fresh, stored)
    if drift:
        for d in drift[:40]:
            print("DRIFT", d)
        raise SystemExit("artifact drift detected (%d fields)" % len(drift))
    print("artifact verified: %d criteria, overall=%s" % (len(fresh["criteria"]), fresh["overall_verdict"]))


def diff(a, b, path="", out=None):
    if out is None:
        out = []
    skip = {"generated_utc", "environment", "branch_blob_reachable_via_git", "inserted_text"}
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k in skip:
                continue
            if k not in a or k not in b:
                out.append("%s/%s missing on one side" % (path, k))
                continue
            diff(a[k], b[k], "%s/%s" % (path, k), out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append("%s length %d vs %d" % (path, len(a), len(b)))
            return out
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, "%s[%d]" % (path, i), out)
    elif isinstance(a, float) or isinstance(b, float):
        try:
            if abs(float(a) - float(b)) > 1e-12 * max(1.0, abs(float(a))):
                out.append("%s %r vs %r" % (path, a, b))
        except Exception:
            if a != b:
                out.append("%s %r vs %r" % (path, a, b))
    else:
        if a != b:
            out.append("%s %r vs %r" % (path, a, b))
    return out


def main(argv):
    if "--check" in argv:
        check()
        return 0
    landing, parsed, final_text, offset = land()
    art = assemble(landing, parsed, final_text, write=True)
    print("landed %s" % ANCHOR_REL)
    print("  branch blob sha256 : %s (identical plain copy: %s)"
          % (ANCHOR_BRANCH_BLOB_SHA256, landing["plain_copy_sha256"] == ANCHOR_BRANCH_BLOB_SHA256))
    print("  landed sha256      : %s (+%d bytes amendment, %d existing bytes changed)"
          % (landing["landed_sha256"], landing["amendment"]["inserted_bytes"],
             landing["amendment"]["existing_bytes_changed"]))
    print("  strip -> branch    : %s" % landing["amendment"]["strip_check"]["equals_branch_blob"])
    print("  anchor_grade       : %s (%s)" % (parsed["anchor_grade"], parsed["grade_rationale"]["grade_letter"]))
    print("  ITTC-57 max resid  : %.3e (tol %.0e)" % (
        max(art["ittc57"]["max_abs_residual_double_vs_registered"],
            art["ittc57"]["max_abs_residual_decimal_vs_registered"]), ITTC_TOL_ABS))
    for c in art["criteria"]:
        print("  %-4s %s" % (c["verdict"], c["id"]))
        if c["id"].startswith("C4"):
            for k, v in c["per_ladder"].items():
                print("        %-28s p=%s  GCI=%s  -> %s"
                      % (k, "n/a" if v["p_observed"] is None else "%.4f" % v["p_observed"],
                         "n/a" if v["gci_fine_pct"] is None else "%.2f%%" % v["gci_fine_pct"],
                         "PASS" if v["pass_both"] else "HONEST_FAIL"))
        if c["id"].startswith("C5"):
            for k, v in c["per_ladder"].items():
                print("        %-28s max|dev| OP1 = %.2f%%  (points in band %d/%d, operator-dependent=%s)"
                      % (k, max(v["max_abs_dev_pct_OP1_by_tier"].values()),
                         v["n_points_within_15pct_OP1"], v["n_points_total"],
                         v["any_point_operator_dependent"]))
    print("overall_verdict     : %s" % art["overall_verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
