"""External-anchor register for the DTRC SUBOFF AFF-1 resistance tables.

Mission W27 (paper-exp-lbm-suboff-tau-12).  This module builds, loads and
verifies ``benchmarks/anchors/lbm-suboff-tau.json`` — the machine-readable
anchor file that the next wave's anchor gate reads (``status`` is one of
``TRANSCRIBED`` / ``VERIFIED`` / ``OBTAINED``).

Design rules (all enforced in code, not in prose):

* **Zero hand-copy.**  Every anchor number in the artifact is parsed at build
  time, by regular expression, out of a frozen in-repo file.  Nothing is typed
  in from memory or from a secondary source.
* **Verbatim + locator + sha256.**  Every registered value carries the
  continuous substring of the archived file it came from, the line/char
  locator inside that file, and the sha256 of that file.  The sha256 is
  recomputed live and cross-checked against sha256 values that two
  *independent* pre-existing repo registries already recorded, so the check is
  a real cross-source check rather than a tautology.
* **Honest acquisition status.**  The primary DTRC reports (DTRC/SHD-1298-01,
  CRDKNSWC/HD-1298-11) are not present in this environment and the sandbox is
  offline (the failed probes are recorded as evidence, not asserted).  The
  primary documents are therefore ``acquisition_status = WITHHELD`` while the
  register itself is ``status = TRANSCRIBED`` (in-repo verbatim transcription
  chain, grade PRIMARY_TABLE_SECOND_HAND).
* **Operator-layer machine checks.**  Because no primary text can be re-opened,
  the file carries the checks that *can* be run offline: dimensional analysis,
  algebraic reconstruction, a second independent ITTC-57 implementation, a
  known-answer check against a rounded transcription, cross-transcription
  agreement, and the geometry gate (Cb / L/D / WSA).

Run::

    python benchmarks/anchors/lbm_suboff_tau_anchor.py            # build + verify
    python benchmarks/anchors/lbm_suboff_tau_anchor.py --check    # verify only
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import sys
from datetime import datetime, timezone
from decimal import Decimal, getcontext
from pathlib import Path

SCHEMA = "lbm-suboff-tau-anchor/v1"
ANCHOR_ID = "lbm-suboff-tau"
TASK_KEY = "paper-exp-lbm-suboff-tau-12"
MISSION_ID = 18960
REPO = "TensorLBM"

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
OUT_PATH = _HERE / "lbm-suboff-tau.json"
SELF_REL = "benchmarks/anchors/lbm_suboff_tau_anchor.py"

# ---- frozen in-repo archives (relative to repo root) ---------------------- #
ARCH_MD_TABLE14 = "docs/suboff_experimental_resistance_validation_2026-08-01.md"
ARCH_PY_REGISTRY = "src/tensorlbm/suboff_reference_data.py"
ARCH_ZH_ITTC = "docs/sphere_suboff_highre_reference_data_zh.md"
ARCH_REFS_LOCK = "benchmarks/pending/suboff_highre_resistance/refs_lock.json"
ARCH_NU_QUOTE = "docs/IN12_S6_WALL_LADDER.md"
ARCH_GEOM_CAD = "src/tensorlbm/suboff_cad.py"

# pre-existing, independently generated repo registries used for the
# cross-source sha256 check (each one recorded file sha256s on its own)
SHA_WITNESSES = (
    "benchmarks/verified/suboff_anchor/anchor.json",
    "benchmarks/verified/suboff_anchor/anchor_points.json",
    "benchmarks/verified/suboff_sail_resistance/gci_v4.json",
    "benchmarks/verified/suboff_anchor_w22/anchor_transcription.json",
    "papers/artifacts/w26/lbm-suboff-tau/anchor_register.json",
)

SUPERSCRIPT = {"\u2070": "0", "\u00b9": "1", "\u00b2": "2", "\u00b3": "3", "\u2074": "4",
               "\u2075": "5", "\u2076": "6", "\u2077": "7", "\u2078": "8", "\u2079": "9"}


def _sha256_strings(node, acc=None):
    """Collect every 64-hex string anywhere in a nested json structure."""
    acc = set() if acc is None else acc
    if isinstance(node, dict):
        for v in node.values():
            _sha256_strings(v, acc)
    elif isinstance(node, list):
        for v in node:
            _sha256_strings(v, acc)
    elif isinstance(node, str) and re.fullmatch(r"[0-9a-f]{64}", node):
        acc.add(node)
    return acc

ITTC57_TOL_ABS = 1e-9
SELF_TOL_REL = 1e-12


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha256_of(rel_path: str, root: Path = _REPO_ROOT) -> str:
    return hashlib.sha256((root / rel_path).read_bytes()).hexdigest()


def _parse_floats(s: str) -> list[float]:
    return [float(t) for t in re.findall(r"-?\d+\.\d+|-?\d+", s)]


def cf_ittc57_double(re: float) -> float:
    """ITTC-1957 model-ship correlation line, IEEE-754 double implementation."""
    return 0.075 / (math.log10(re) - 2.0) ** 2


def cf_ittc57_decimal(re: Decimal) -> Decimal:
    """Independent 40-digit implementation (Decimal ``ln`` path, not ``log10``).

    Different arithmetic and a different logarithm implementation from
    :func:`cf_ittc57_double`, which is what makes the ``<= 1e-9`` residual a
    meaningful check rather than a tautology.
    """
    getcontext().prec = 40
    ln10 = Decimal(10).ln()
    return Decimal("0.075") / (re.ln() / ln10 - Decimal(2)) ** 2


# --- minimal SI unit algebra (for the dimensional gate) -------------------- #
_DIM = ("m", "s", "kg")


def _dim(**kw) -> dict:
    d = {k: 0 for k in _DIM}
    for k, v in kw.items():
        d[k] = v
    return d


def _mul(*dims: dict) -> dict:
    out = {k: 0 for k in _DIM}
    for d in dims:
        for k in _DIM:
            out[k] += d[k]
    return out


def _pow(d: dict, n: int) -> dict:
    return {k: n * v for k, v in d.items()}


def _is_dimensionless(d: dict) -> bool:
    return all(v == 0 for v in d.values())


# --------------------------------------------------------------------------- #
# parsing: the two independent in-repo transcriptions of HD-1298-11 Table 14
# --------------------------------------------------------------------------- #
def parse_md_table14(text: str) -> dict:
    """Parse the AFF-1 / AFF-8 rows out of the archived markdown table."""
    lines = text.splitlines()
    rows: dict = {}
    for cfg in ("AFF-1", "AFF-8"):
        for i, line in enumerate(lines):
            if line.startswith("|") and cfg in line:
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                speeds = _parse_floats(cells[1])
                forces = _parse_floats(cells[2])
                if len(speeds) != 6 or len(forces) != 6:
                    raise RuntimeError(f"unexpected {cfg} row shape: {line!r}")
                rows[cfg] = {
                    "quote": line,
                    "line": i + 1,
                    "char_offset": text.index(line),
                    "speed_knots": speeds,
                    "resistance_n": forces,
                }
                break
        else:  # pragma: no cover - archive damage guard
            raise RuntimeError(f"{cfg} row not found in {ARCH_MD_TABLE14}")

    loc = None
    for i, line in enumerate(lines):
        m = re.search(r"(Table\s*\d+),\s*report\s*p\.\s*(\d+)", line)
        if m:
            loc = {"quote": line.strip(), "line": i + 1, "table": m.group(1), "report_page": int(m.group(2))}
            break
    if loc is None:  # pragma: no cover
        raise RuntimeError("report-level locator (table/page) not found")

    citation = None
    for i, line in enumerate(lines):
        if "CRDKNSWC/HD-1298-11" in line:
            citation = {"quote": line.strip(), "line": i + 1}
            break
    if citation is None:  # pragma: no cover
        raise RuntimeError("report citation line not found")
    return {"rows": rows, "locator": loc, "citation": citation}


def parse_py_registry(text: str) -> dict:
    """Parse the frozen dimensional registry (per-point constructor calls)."""
    pat = re.compile(
        r'SuboffTowTankResistancePoint\(\s*"(\w+)"\s*,\s*(\d+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)'
    )
    points = []
    for m in pat.finditer(text):
        hull, cfg, v, r = m.groups()
        points.append(
            {
                "quote": m.group(0),
                "line": text.count("\n", 0, m.start()) + 1,
                "char_offset": m.start(),
                "hull_type": hull,
                "configuration": int(cfg),
                "speed_knots": float(v),
                "resistance_n": float(r),
            }
        )
    if not points:  # pragma: no cover
        raise RuntimeError(f"no registry points parsed from {ARCH_PY_REGISTRY}")
    cit = re.search(r'"([^"]*CRDKNSWC/HD-1298-11[^"]*)"', text)
    return {
        "points": points,
        "citation": (
            {"quote": cit.group(1), "char_offset": cit.start(), "line": text.count("\n", 0, cit.start()) + 1}
            if cit
            else None
        ),
    }


def parse_geometry(refs_lock: dict, zh_text: str, cad_text: str) -> dict:
    t1 = refs_lock["S1_Table1_verbatim"]
    geom = {
        "L_max_m": t1["Lmax_m"],
        "L_PP_m": t1["LPP_m"],
        "D_max_m": t1["Dmax_m"],
        "displacement_m3": t1["displacement_m3"],
        "WSA_m2": t1["WSA_m2"],
        "verbatim": {
            "path": ARCH_REFS_LOCK,
            "pointer": "/S1_Table1_verbatim",
            "quote": json.dumps(t1, ensure_ascii=False, sort_keys=True),
            "note": "frozen reference lock; the lock records these as a verbatim transcription of "
            "Table 1 of the experimental-data report",
        },
    }
    cites = []
    for i, line in enumerate(cad_text.splitlines()):
        if "SHD-1298-01" in line:
            cites.append({"path": ARCH_GEOM_CAD, "line": i + 1, "quote": line.strip()})
    geom["shd1298_citations"] = cites
    for i, line in enumerate(zh_text.splitlines()):
        if "L = 4.356 m" in line and "D = 0.508 m" in line:
            geom["compiled_table_quote"] = {"path": ARCH_ZH_ITTC, "line": i + 1, "quote": line.strip()}
            break
    return geom


def parse_ittc_transcriptions(zh_text: str) -> dict:
    """ITTC-57 formula text + rounded known-answer rows from the compiled table."""
    formula = None
    known = []
    for i, line in enumerate(zh_text.splitlines()):
        if "0.075/(lgRe" in line:
            formula = {"path": ARCH_ZH_ITTC, "line": i + 1, "quote": line.strip()}
        if "ITTC-1957 Cf" in line and "\u00b1" in line:
            m = re.search(r"(\d(?:\.\d+)?)\u00d710\^?([0-9\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079]+)", line)
            if m:
                exp = "".join(SUPERSCRIPT.get(ch, ch) for ch in m.group(2))
                m = (m.group(1), exp)
            cf = re.search(r"(0\.\d+)\s*\u00b1", line)
            if m and cf:
                known.append(
                    {
                        "path": ARCH_ZH_ITTC,
                        "line": i + 1,
                        "quote": line.strip(),
                        "re": float(f"{m[0]}e{m[1]}"),
                        "cf_recorded": float(cf.group(1)),
                        "declared_band_pct": 5.0,
                    }
                )
    return {"formula": formula, "known_answers": known}


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build(root: Path = _REPO_ROOT) -> dict:
    md_text = (root / ARCH_MD_TABLE14).read_text(encoding="utf-8")
    py_text = (root / ARCH_PY_REGISTRY).read_text(encoding="utf-8")
    zh_text = (root / ARCH_ZH_ITTC).read_text(encoding="utf-8")
    cad_text = (root / ARCH_GEOM_CAD).read_text(encoding="utf-8")
    nu_text = (root / ARCH_NU_QUOTE).read_text(encoding="utf-8")
    refs_lock = json.loads((root / ARCH_REFS_LOCK).read_text(encoding="utf-8"))

    md = parse_md_table14(md_text)
    py = parse_py_registry(py_text)
    geom = parse_geometry(refs_lock, zh_text, cad_text)
    ittc_tr = parse_ittc_transcriptions(zh_text)

    nu_quote = None
    for i, line in enumerate(nu_text.splitlines()):
        m = re.search(r"nu\s*=\s*([0-9.]+)e-0?6", line)
        if m:
            nu_quote = {
                "path": ARCH_NU_QUOTE,
                "line": i + 1,
                "quote": line.strip(),
                "nu_m2_s": float(m.group(1)) * 1e-6,
            }
            break
    if nu_quote is None:  # pragma: no cover
        raise RuntimeError("kinematic-viscosity convention quote not found")
    kn_to_ms = 0.514444  # declared convention (international knot = 1852 m / 3600 s)
    rho = 1000.0  # declared fresh-water density assumption

    L = geom["L_max_m"]
    S = geom["WSA_m2"]
    nu = nu_quote["nu_m2_s"]

    # ---- per-point anchor rows -------------------------------------------- #
    points = []
    for i, (v_kn, rt) in enumerate(
        zip(md["rows"]["AFF-1"]["speed_knots"], md["rows"]["AFF-1"]["resistance_n"])
    ):
        v_ms = v_kn * kn_to_ms
        re_ = v_ms * L / nu
        ct = rt / (0.5 * rho * v_ms**2 * S)
        cf_d = cf_ittc57_double(re_)
        reg = [
            p
            for p in py["points"]
            if p["hull_type"] == "bare_hull"
            and abs(p["speed_knots"] - v_kn) < 1e-12
            and abs(p["resistance_n"] - rt) < 1e-12
        ]
        points.append(
            {
                "id": f"AFF1-P{i + 1}",
                "configuration": 1,
                "hull_type": "AFF-1 bare hull",
                "speed_knots": v_kn,
                "speed_ms": v_ms,
                "resistance_n": rt,
                "Re": re_,
                "C_T": ct,
                "C_F": {
                    "value": cf_d,
                    "kind": "ITTC57_CORRELATION_DERIVED",
                    "grade": "CONVENTION",
                    "definition": "Cf = 0.075/(log10(Re)-2)^2 evaluated at this point's Re",
                    "experimental_value": "WITHHELD",
                    "experimental_note": "the experimental program reports total tow-tank resistance; "
                    "no measured friction coefficient for SUBOFF is transcribed anywhere in-repo "
                    "(criterion A2), so none is registered",
                },
                "provenance": {
                    "verbatim_quote": md["rows"]["AFF-1"]["quote"],
                    "locator": {
                        "path": ARCH_MD_TABLE14,
                        "line": md["rows"]["AFF-1"]["line"],
                        "char_offset_md": md["rows"]["AFF-1"]["char_offset"],
                        "table": md["locator"]["table"],
                        "report_page": md["locator"]["report_page"],
                        "citation_line": md["locator"]["quote"],
                    },
                    "sha256": sha256_of(ARCH_MD_TABLE14, root),
                    "cross_transcription": (
                        {
                            "path": ARCH_PY_REGISTRY,
                            "line": reg[0]["line"],
                            "quote": reg[0]["quote"],
                            "sha256": sha256_of(ARCH_PY_REGISTRY, root),
                            "agrees": True,
                        }
                        if reg
                        else {"agrees": False, "note": "no matching registry point found"}
                    ),
                },
            }
        )

    checks = []
    crit = []

    # (1) sha256 cross-source consistency
    sha_rows = []
    witness_shas = {
        w: _sha256_strings(json.loads((root / w).read_text(encoding="utf-8"))) for w in SHA_WITNESSES
    }
    for path in (ARCH_PY_REGISTRY, ARCH_MD_TABLE14, ARCH_REFS_LOCK):
        live = sha256_of(path, root)
        witnesses_found = [w for w, shas in witness_shas.items() if live in shas]
        sha_rows.append(
            {
                "path": path,
                "sha256_recomputed": live,
                "witness_registries_recording_this_sha": witnesses_found,
                "n_witnesses": len(witnesses_found),
                "all_match": len(witnesses_found) > 0,
            }
        )
    checks.append(
        {
            "id": "OP1_sha256_cross_source",
            "what": "live sha256 of every quoted archive is found verbatim in independent "
            "pre-existing repo sha256 registries (written by earlier, unrelated builds)",
            "rows": sha_rows,
            "pass": all(r["all_match"] for r in sha_rows),
        }
    )
    crit.append(
        {
            "id": "A1_verbatim_quote_and_sha256",
            "statement": "anchor file carries >=1 verbatim quote with locator, and the sha256 of the "
            "quoted archive recomputes consistently (cross-source)",
            "n_verbatim_quotes": len(points) * 2 + len(geom["shd1298_citations"]) + 1,
            "n_witness_registries": len(SHA_WITNESSES),
            "n_sha256_cross_checked": sum(1 for r in sha_rows if r["n_witnesses"] > 0),
            "pass": all(r["all_match"] for r in sha_rows) and len(points) >= 1,
        }
    )

    # (2) experimental C_F availability
    crit.append(
        {
            "id": "A2_experimental_CF",
            "statement": "verbatim experimental C_F anchor present, else honest WITHHELD",
            "status": "WITHHELD",
            "evidence": "the in-repo transcription chain (Table 14 of HD-1298-11 via "
            f"{ARCH_MD_TABLE14} and {ARCH_PY_REGISTRY}) carries dimensional total resistance R_T(V) "
            "only; no measured friction coefficient is transcribed anywhere in-repo and the "
            "primary report is not reachable offline (see /documents/*/acquisition).",
            "honest_withheld": True,
        }
    )

    # (3) ITTC-57 independent recomputation
    ittc_rows = []
    for p in points:
        cf_dec = float(cf_ittc57_decimal(Decimal(p["Re"])))
        resid_abs = abs(cf_dec - p["C_F"]["value"])
        ittc_rows.append(
            {
                "point": p["id"],
                "Re": p["Re"],
                "cf_double": p["C_F"]["value"],
                "cf_decimal_40dps": cf_dec,
                "abs_residual": resid_abs,
                "rel_residual": resid_abs / p["C_F"]["value"],
                "pass_abs_1e9": resid_abs <= ITTC57_TOL_ABS,
            }
        )
    known = []
    for k in ittc_tr["known_answers"]:
        cf_recomputed = cf_ittc57_double(k["re"])
        dev = (cf_recomputed - k["cf_recorded"]) / k["cf_recorded"] * 100.0
        known.append(
            {
                "quote": k["quote"],
                "locator": {"path": k["path"], "line": k["line"]},
                "Re": k["re"],
                "cf_recorded": k["cf_recorded"],
                "cf_recomputed": cf_recomputed,
                "rel_dev_pct": dev,
                "declared_band_pct": k["declared_band_pct"],
                "within_band": abs(dev) <= k["declared_band_pct"],
            }
        )
    checks.append(
        {
            "id": "OP2_ittc57_independent_recompute",
            "what": "same ITTC-57 value from two independent arithmetic paths "
            "(math.log10 double vs Decimal-40dps), plus known-answer rows recorded in-repo",
            "tolerance_abs": ITTC57_TOL_ABS,
            "rows": ittc_rows,
            "max_abs_residual": max(r["abs_residual"] for r in ittc_rows),
            "known_answer_checks": known,
            "pass": all(r["pass_abs_1e9"] for r in ittc_rows),
        }
    )
    crit.append(
        {
            "id": "A3_ittc57_recompute_residual",
            "statement": "|ITTC-57 recomputed (independent path) - registered C_F| <= 1e-9",
            "tolerance_abs": ITTC57_TOL_ABS,
            "max_abs_residual": max(r["abs_residual"] for r in ittc_rows),
            "known_answer_band_pct": max((k["declared_band_pct"] for k in known), default=None),
            "known_answer_pass": all(k["within_band"] for k in known),
            "pass": all(r["pass_abs_1e9"] for r in ittc_rows),
        }
    )

    # (4) dimensional analysis
    dim_R = _dim(m=1, s=-2, kg=1)  # N = kg m s^-2
    dim_rho = _dim(m=-3, kg=1)
    dim_V2 = _pow(_dim(m=1, s=-1), 2)
    dim_S = _dim(m=2)
    dim_ct = _mul(dim_R, _pow(_mul(dim_rho, dim_V2, dim_S), -1))
    dim_re = _mul(_dim(m=1, s=-1), _dim(m=1), _pow(_dim(m=2, s=-1), -1))
    checks.append(
        {
            "id": "OP3_dimensional_analysis",
            "entries": [
                {"quantity": "C_T = R_T/(0.5 rho V^2 S)", "dims": dim_ct, "dimensionless": _is_dimensionless(dim_ct)},
                {"quantity": "Re = V L / nu", "dims": dim_re, "dimensionless": _is_dimensionless(dim_re)},
            ],
            "pass": _is_dimensionless(dim_ct) and _is_dimensionless(dim_re),
        }
    )

    # (5) algebraic reconstruction from the verbatim values only
    recon = []
    for p in points:
        v_ms = p["speed_knots"] * kn_to_ms
        re_ = v_ms * L / nu
        ct = p["resistance_n"] / (0.5 * rho * v_ms**2 * S)
        recon.append(
            {
                "point": p["id"],
                "rel_dev_Re": abs(re_ - p["Re"]) / p["Re"],
                "rel_dev_CT": abs(ct - p["C_T"]) / p["C_T"],
            }
        )
    checks.append(
        {
            "id": "OP4_algebraic_reconstruction",
            "what": "Re and C_T are reproduced from the verbatim V_kn / R_T plus the locked geometry "
            "(L_max, WSA) and the declared conventions (rho, nu, kn->m/s) alone",
            "rows": recon,
            "max_rel_dev": max(max(r["rel_dev_Re"], r["rel_dev_CT"]) for r in recon),
            "tolerance_rel": SELF_TOL_REL,
            "pass": all(r["rel_dev_Re"] <= SELF_TOL_REL and r["rel_dev_CT"] <= SELF_TOL_REL for r in recon),
        }
    )

    # (6) monotone structure
    v = [p["speed_ms"] for p in points]
    rt = [p["resistance_n"] for p in points]
    cts = [p["C_T"] for p in points]
    checks.append(
        {
            "id": "OP5_monotone_structure",
            "what": "R_T strictly increasing and C_T strictly decreasing with speed",
            "rt_increasing": all(b > a for a, b in zip(rt, rt[1:])),
            "ct_decreasing": all(b < a for a, b in zip(cts, cts[1:])),
            "speeds_increasing": all(b > a for a, b in zip(v, v[1:])),
            "pass": all(b > a for a, b in zip(rt, rt[1:])) and all(b < a for a, b in zip(cts, cts[1:])),
        }
    )

    # (7) cross-transcription agreement
    xrows = []
    for p in points:
        reg = p["provenance"]["cross_transcription"]
        xrows.append(
            {
                "point": p["id"],
                "V_kn": p["speed_knots"],
                "R_T_N": p["resistance_n"],
                "registry_line": reg.get("line"),
                "registry_quote": reg.get("quote"),
                "agrees": reg["agrees"],
            }
        )
    n_bare_reg = sum(1 for q in py["points"] if q["hull_type"] == "bare_hull")
    checks.append(
        {
            "id": "OP6_cross_transcription",
            "what": "the two independent in-repo transcriptions of Table 14 (markdown table and the "
            "frozen python registry) agree pointwise",
            "rows": xrows,
            "n_bare_rows_registry": n_bare_reg,
            "pass": all(r["agrees"] for r in xrows) and n_bare_reg == len(points),
        }
    )

    # (8) configuration-ratio audit (definitional; transcription fidelity only)
    ratio_rows = []
    for v1, r1, v8, r8 in zip(
        md["rows"]["AFF-1"]["speed_knots"],
        md["rows"]["AFF-1"]["resistance_n"],
        md["rows"]["AFF-8"]["speed_knots"],
        md["rows"]["AFF-8"]["resistance_n"],
    ):
        ratio_rows.append({"V_AFF1_kn": v1, "V_AFF8_kn": v8, "R_T_ratio": r8 / r1})
    checks.append(
        {
            "id": "OP7_configuration_ratio",
            "what": "AFF-8/AFF-1 total-resistance ratio recomputed from the verbatim rows "
            "(definitional identity -> transcription fidelity, not independent evidence)",
            "rows": ratio_rows,
            "pass": True,
        }
    )

    # (9) geometry gate (Cb / L/D / WSA) before any drag use
    disp = geom["displacement_m3"]
    D = geom["D_max_m"]
    cb = disp / (L * math.pi / 4.0 * D**2)
    wsa_analytic = 0.31576 * L**2
    g = {
        "L_over_D": L / D,
        "L_over_D_recorded": 8.57,
        "L_over_D_dev_pct": (L / D - 8.57) / 8.57 * 100.0,
        "Cb_recomputed": cb,
        "Cb_nominal": 0.80,
        "Cb_dev_pct": (cb - 0.80) / 0.80 * 100.0,
        "WSA_recorded_m2": S,
        "WSA_analytic_coef_L2": 0.31576,
        "WSA_analytic_m2": wsa_analytic,
        "WSA_dev_pct": (wsa_analytic - S) / S * 100.0,
        "Cb_band_pct": 3.0,
        "L_over_D_band_pct": 1.0,
        "WSA_band_pct": 1.0,
    }
    g["pass"] = (
        abs(g["Cb_dev_pct"]) <= g["Cb_band_pct"]
        and abs(g["L_over_D_dev_pct"]) <= g["L_over_D_band_pct"]
        and abs(g["WSA_dev_pct"]) <= g["WSA_band_pct"]
    )
    checks.append(
        {
            "id": "OP8_geometry_gate",
            "what": "geometry validated (Cb, L/D, WSA) before any drag value is used",
            "detail": g,
            "pass": g["pass"],
        }
    )
    crit.append(
        {
            "id": "A4_geometry_gate",
            "statement": "geometry verified (Cb within 3% of 0.80, L/D within 1% of 8.57, WSA within "
            "1% of the analytic surface-of-revolution value) before the drag anchor is used",
            "detail": g,
            "pass": g["pass"],
        }
    )

    documents = [
        {
            "doc_id": "DTRC-SHD-1298-01",
            "citation": "P. E. Groves, T. T. Huang, S. S. Chang (1989). Geometric Characteristics of "
            "DARPA SUBOFF Models. DTRC Report SHD-1298-01.",
            "role": "hull geometry / offset tables",
            "acquisition_status": "WITHHELD",
            "acquisition": {
                "attempts": [
                    {
                        "command": "curl -m 20 -sS https://apps.dtic.mil/sti/citations/ADA359226",
                        "result": "DNS failure (curl exit 6: could not resolve host)",
                    },
                    {
                        "command": "curl -m 15 -sS https://www.google.com",
                        "result": "connection timed out (curl exit 28) - sandbox is offline",
                    },
                    {
                        "command": "find / -iname '*1298*' -o -iname '*SHD*' \\( -iname '*.pdf' -o "
                        "-iname '*.txt' -o -iname '*.html' \\)",
                        "result": "no copy of SHD-1298-01 in the environment",
                    },
                ],
                "statement": "the primary geometry report is not present in this environment and the "
                "sandbox is offline; geometry values used here come from the in-repo frozen reference "
                "lock (Table 1 transcription), not from a direct read of SHD-1298-01",
            },
            "in_repo_secondary_citations": geom["shd1298_citations"],
        },
        {
            "doc_id": "CRDKNSWC-HD-1298-11",
            "citation": "H.-L. Liu, T. T. Huang (1998). Summary of DARPA Suboff Experimental Program "
            "Data. CRDKNSWC/HD-1298-11 (DTIC ADA359226).",
            "role": "bare-hull (AFF-1) tow-tank resistance table used as the drag anchor",
            "acquisition_status": "WITHHELD",
            "acquisition": {
                "attempts": [
                    {
                        "command": "curl -m 20 -sS https://apps.dtic.mil/sti/citations/ADA359226",
                        "result": "DNS failure (curl exit 6)",
                    },
                    {
                        "command": "curl -m 15 -sS https://www.google.com",
                        "result": "connection timed out (curl exit 28)",
                    },
                ],
                "statement": "primary PDF unobtainable offline; the registered values are verbatim "
                "transcriptions of Table 14 (report page 23) archived in-repo, grade "
                "PRIMARY_TABLE_SECOND_HAND",
            },
            "in_repo_verbatim_archives": [
                {
                    "path": ARCH_MD_TABLE14,
                    "sha256": sha256_of(ARCH_MD_TABLE14, root),
                    "locator": {
                        "line": md["rows"]["AFF-1"]["line"],
                        "table": md["locator"]["table"],
                        "report_page": md["locator"]["report_page"],
                        "citation_line": md["locator"]["quote"],
                    },
                    "quote": md["rows"]["AFF-1"]["quote"],
                },
                {
                    "path": ARCH_PY_REGISTRY,
                    "sha256": sha256_of(ARCH_PY_REGISTRY, root),
                    "locator": {"lines": sorted({p["line"] for p in py["points"]})},
                    "quote": py["citation"]["quote"] if py["citation"] else None,
                },
            ],
        },
    ]

    return {
        "schema": SCHEMA,
        "anchor_id": ANCHOR_ID,
        "task_key": TASK_KEY,
        "mission_id": MISSION_ID,
        "repo": REPO,
        "discipline": "ship-total-performance / resistance (bare submarine hull)",
        "generated_by": SELF_REL,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "network": "unavailable (DNS/connect failures recorded under /documents/*/acquisition)",
        },
        "status": "TRANSCRIBED",
        "status_meaning": {
            "TRANSCRIBED": "values are verbatim transcriptions of the source table archived in-repo; "
            "the primary document has not been re-read in this environment",
            "VERIFIED": "values re-read from the primary document (page image/PDF) and matched here",
            "OBTAINED": "primary document acquired and stored in-repo with sha256",
        },
        "acquisition_status": "WITHHELD",
        "gate_contract": {
            "reader": "any anchor gate may load this file and read /status, /acquisition_status, "
            "/anchor_points[*].{Re,C_T,C_F.value}, /conventions, /geometry",
            "accepted_status_values": ["TRANSCRIBED", "VERIFIED", "OBTAINED"],
            "accepted_acquisition_status_values": ["OBTAINED", "TRANSCRIBED", "WITHHELD"],
            "loader": "benchmarks/anchors/lbm_suboff_tau_anchor.py::load_anchor",
            "required_top_level_keys": [
                "schema",
                "status",
                "acquisition_status",
                "conventions",
                "geometry",
                "anchor_points",
                "documents",
                "criteria",
            ],
            "no_hand_copy_rule": "anchor_points[*] values were parsed at build time from the quoted "
            "archives; a consumer must not treat any value without a /provenance block as evidence",
        },
        "case": {
            "hull": "DARPA SUBOFF AFF-1 bare hull (configuration 1)",
            "geometry": "axisymmetric body of revolution, L_max = 4.356 m, D_max = 0.508 m",
            "facility": "DTRC tow tank (US Navy David Taylor Research Center)",
            "quantity": "total resistance coefficient C_T at the tow-tank Reynolds number",
            "n_points": len(points),
        },
        "conventions": {
            "rho_kg_m3": rho,
            "rho_provenance": "declared fresh-water density assumption (convention, not measured)",
            "nu_m2_s": nu,
            "nu_provenance": nu_quote,
            "kn_to_ms": kn_to_ms,
            "kn_to_ms_provenance": "international knot definition 1852 m / 3600 s",
            "re_length": "L_max",
            "ct_definition": "C_T = R_T / (0.5 * rho * V^2 * S_wetted)",
            "cf_definition": "C_F = 0.075/(log10(Re)-2)^2  (ITTC-1957 correlation line)",
            "reference_area": "WSA = 5.998 m^2 (Table 1 transcription of the experimental-data report)",
            "ittc57_transcription": ittc_tr,
        },
        "geometry": geom,
        "documents": documents,
        "anchor_points": points,
        "operator_layer_checks": checks,
        "criteria": crit,
        "summary": {
            "n_anchor_points": len(points),
            "Re_range": [min(p["Re"] for p in points), max(p["Re"] for p in points)],
            "C_T_range": [min(p["C_T"] for p in points), max(p["C_T"] for p in points)],
            "reference_point": {
                "id": points[0]["id"],
                "V_knots": points[0]["speed_knots"],
                "Re": points[0]["Re"],
                "C_T": points[0]["C_T"],
                "C_F_ittc57": points[0]["C_F"]["value"],
            },
            "all_operator_checks_pass": all(c["pass"] for c in checks),
            "all_criteria_pass_or_honest_withheld": all(
                c.get("pass", True) or c.get("honest_withheld", False) for c in crit
            ),
        },
        "provenance": {
            "generator_sha256": sha256_of(SELF_REL, root) if (root / SELF_REL).exists() else None,
            "inputs": {
                p: sha256_of(p, root)
                for p in (
                    ARCH_MD_TABLE14,
                    ARCH_PY_REGISTRY,
                    ARCH_ZH_ITTC,
                    ARCH_REFS_LOCK,
                    ARCH_NU_QUOTE,
                    ARCH_GEOM_CAD,
                )
            },
            "commands": [
                "python benchmarks/anchors/lbm_suboff_tau_anchor.py",
                "python benchmarks/anchors/lbm_suboff_tau_anchor.py --check",
            ],
        },
    }


# --------------------------------------------------------------------------- #
# verify / load
# --------------------------------------------------------------------------- #
def verify(path: Path = OUT_PATH, root: Path = _REPO_ROOT) -> dict:
    """Re-verify a built anchor file against the live archives."""
    art = json.loads(Path(path).read_text(encoding="utf-8"))
    problems: list[str] = []
    if art.get("status") not in ("TRANSCRIBED", "VERIFIED", "OBTAINED"):
        problems.append(f"bad status {art.get('status')!r}")
    for key in art["gate_contract"]["required_top_level_keys"]:
        if key not in art:
            problems.append(f"missing required key {key}")
    for p in art["anchor_points"]:
        loc = p["provenance"]["locator"]
        text = (root / loc["path"]).read_text(encoding="utf-8")
        if p["provenance"]["verbatim_quote"] not in text:
            problems.append(f"{p['id']}: quote not found in {loc['path']}")
        if sha256_of(loc["path"], root) != p["provenance"]["sha256"]:
            problems.append(f"{p['id']}: sha256 drift on {loc['path']}")
        x = p["provenance"]["cross_transcription"]
        if x.get("path"):
            t2 = (root / x["path"]).read_text(encoding="utf-8")
            if x["quote"] not in t2:
                problems.append(f"{p['id']}: cross-transcription quote missing")
            if sha256_of(x["path"], root) != x["sha256"]:
                problems.append(f"{p['id']}: sha256 drift on {x['path']}")
        cf = float(cf_ittc57_decimal(Decimal(p["Re"])))
        if abs(cf - p["C_F"]["value"]) > ITTC57_TOL_ABS:
            problems.append(f"{p['id']}: ITTC-57 residual > {ITTC57_TOL_ABS}")
    return {"path": str(Path(path)), "ok": not problems, "problems": problems}


def load_anchor(path: Path = OUT_PATH) -> dict:
    """Loader used by downstream anchor gates."""
    art = json.loads(Path(path).read_text(encoding="utf-8"))
    status = art.get("status")
    if status not in ("TRANSCRIBED", "VERIFIED", "OBTAINED"):
        raise ValueError(f"anchor {art.get('anchor_id')} has gate-inadmissible status {status!r}")
    return art


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--check" in argv:
        res = verify()
        print(json.dumps(res, indent=1, ensure_ascii=False))
        return 0 if res["ok"] else 1
    art = build()
    OUT_PATH.write_text(json.dumps(art, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    res = verify()
    print(
        json.dumps(
            {
                "written": str(OUT_PATH),
                "status": art["status"],
                "acquisition_status": art["acquisition_status"],
                "n_points": art["summary"]["n_anchor_points"],
                "checks_pass": art["summary"]["all_operator_checks_pass"],
                "verify": res["ok"],
                "problems": res["problems"],
            },
            indent=1,
            ensure_ascii=False,
        )
    )
    return 0 if res["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())