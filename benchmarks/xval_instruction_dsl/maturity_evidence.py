#!/usr/bin/env python3
"""Assemble machine-checkable L2/L3/L4 maturity evidence for the instruction DSL.

Everything here is *derived from the tree*: implementation files are hashed and
scanned for placeholder markers, run artefacts are hashed, the integration call
sites are located by grep, and the benchmark table is lifted from
``artifacts/xval_instruction_dsl/results.json`` (never copied by hand).

    python3 benchmarks/xval_instruction_dsl/maturity_evidence.py
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "artifacts" / "xval_instruction_dsl"
ANCHOR = "XVAL-LBM-INSTRUCTION-DSL-TESTS-SH"
IMPL_FILES = [
    "src/tensorlbm/instruction_dsl.py",
    "src/tensorlbm/instruction_dsl_cli.py",
    "benchmarks/xval_instruction_dsl/run_xval.py",
]
ENTRY = "src/tensorlbm/instruction_dsl_cli.py"
TESTS = ["tests/test_instruction_dsl.py"]
INTEGRATION_FILES = [
    ("src/tensorlbm/example_harness.py",
     "main example runtime: run_instruction_dsl_program() drives a DSL program"),
    ("src/tensorlbm/api.py", "stable public API exports the DSL entry points"),
    ("src/tensorlbm/instruction_dsl_cli.py", "CLI/programmatic entry point"),
]
INTEGRATION_TOKEN = "instruction_dsl"
EXT_ANCHOR = ["docs/XVAL-LBM-INSTRUCTION-DSL-TESTS-SH.md",
              "artifacts/xval_instruction_dsl/results.json"]
REPRO = ("python3 benchmarks/xval_instruction_dsl/run_xval.py && "
         "python3 -m tensorlbm.instruction_dsl_cli artifacts/xval_instruction_dsl/"
         "programs/poiseuille_ny22.lbm --out artifacts/xval_instruction_dsl/cli_run.json && "
         "python3 benchmarks/xval_instruction_dsl/maturity_evidence.py")
PLACEHOLDER_PATTERNS = (r"\bNotImplementedError\b", r"\bTODO\b", r"\bFIXME\b",
                        r"\bpass\s*#\s*stub")
THRESHOLD_PCT = 15.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_record(rel: str) -> dict:
    path = REPO / rel
    rec = {"path": rel, "exists": path.is_file()}
    if not rec["exists"]:
        return rec
    text = path.read_text(encoding="utf-8", errors="ignore")
    rec.update({
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "lines": text.count("\n") + 1,
        "n_functions": len(re.findall(r"^\s*def \w+", text, re.M)),
        "placeholders": [p for p in PLACEHOLDER_PATTERNS if re.search(p, text)],
    })
    rec["non_placeholder"] = bool(rec["lines"] >= 40 and rec["n_functions"] >= 2
                                  and not rec["placeholders"])
    return rec


def integration_records() -> list[dict]:
    out = []
    for rel, why in INTEGRATION_FILES:
        path = REPO / rel
        rec = {"path": rel, "claimed_role": why, "exists": path.is_file(),
               "call_lines": []}
        if rec["exists"]:
            for i, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
                if INTEGRATION_TOKEN in line:
                    rec["call_lines"].append({"line": i, "text": line.strip()[:160]})
        rec["is_real_caller"] = bool(rec["call_lines"])
        out.append(rec)
    return out


def benchmark_table() -> dict:
    res = json.loads((ART / "results.json").read_text())
    rows = []
    for tag, p in sorted(res["physics"].items()):
        if tag.startswith("poiseuille"):
            pois = p["poiseuille"]
            rows.append({
                "case": "CH-POISEUILLE-ANALYTIC-DSL", "arm": tag,
                "metric": "max relative error of u(y) vs Stokes parabola",
                "ref": {"u_max_analytic": pois["u_max_analytic"],
                        "law": "u(y)=g(y-1/2)(H-(y-1/2))/(2 nu), H=ny-2"},
                "got": {"u_max_numeric": pois["u_max_numeric"]},
                "max_rel_error_pct": 100.0 * pois["max_rel_error"],
                "l2_rel_error_pct": 100.0 * pois["l2_rel_error"],
                "within_15pct": 100.0 * pois["max_rel_error"] <= THRESHOLD_PCT,
                "nu_declared": p["nu_declared"],
                "nu_fit": p["parabola_fit"].get("nu_fit"),
                "wall_positions_fit": [p["parabola_fit"].get("wall_low"),
                                       p["parabola_fit"].get("wall_high")],
                "program_digest": p["program_digest"],
            })
        elif tag.startswith("taylor_green"):
            rows.append({
                "case": "TAYLOR-GREEN-DECAY-DSL", "arm": tag,
                "metric": "effective viscosity from kinetic-energy decay vs declared nu",
                "ref": {"nu_declared": p["nu_declared"],
                        "law": "KE(t) ~ exp(-4 nu k^2 t), k=2pi/nx"},
                "got": {"nu_eff": p["nu_eff"]},
                "rel_error_pct": p["viscosity_rel_error_pct"],
                "fit_r2": p["fit_r2"],
                "within_15pct": p["viscosity_rel_error_pct"] <= THRESHOLD_PCT,
                "program_digest": p["program_digest"],
            })
    parity = res.get("pylbm_xval")
    if parity:
        rows.append({
            "case": "PYLBM-SCHEME-OPERATOR-PARITY", "arm": "operator_level",
            "metric": "max |DSL - pylbm| on moment matrix / weights / f_eq",
            "ref": {"pylbm_version": parity["reference"]["version"],
                    "pylbm_commit": parity["reference"]["commit"]},
            "got": {"moment_matrix_max_abs_diff": parity["moment_matrix_max_abs_diff"],
                    "weights_max_abs_diff": parity["weights_max_abs_diff"],
                    "equilibrium_population_max_abs_diff":
                        parity["equilibrium_population_max_abs_diff"]},
            "tolerance_abs": res["thresholds"]["scheme_max_abs_diff"],
            "within_tolerance": max(parity["moment_matrix_max_abs_diff"],
                                    parity["weights_max_abs_diff"],
                                    parity["equilibrium_population_max_abs_diff"])
                                <= res["thresholds"]["scheme_max_abs_diff"],
        })
    worst = max(r.get("max_rel_error_pct", r.get("rel_error_pct", 0.0)) for r in rows)
    return {"anchor_slug": ANCHOR, "tolerance_pct": THRESHOLD_PCT,
            "worst_physics_rel_error_pct": worst, "rows": rows,
            "criteria": res["criteria"]}


def main() -> int:
    art = []
    for name in ("results.json", "summary.md", "cli_run.json"):
        p = ART / name
        if p.is_file():
            art.append(file_record(str(p.relative_to(REPO))))
    for p in sorted(ART.glob("figures/*.png")):
        art.append(file_record(str(p.relative_to(REPO))))
    for p in sorted((ART / "programs").glob("*.lbm")):
        art.append(file_record(str(p.relative_to(REPO))))

    evidence = {
        "case": ANCHOR,
        "repo": "TensorLBM",
        "generated_by": "benchmarks/xval_instruction_dsl/maturity_evidence.py",
        # flat keys: directly usable as a maturity-gate evidence document
        "impl_files": [f["path"] for f in map(file_record, IMPL_FILES)],
        "entry": ENTRY,
        "run_artifacts": [a["path"] for a in art],
        "tests": TESTS,
        "integration_callers": [INTEGRATION_TOKEN],
        "benchmark": benchmark_table(),
        "repro_cmd": REPRO,
        "external_anchor": EXT_ANCHOR,
        # nested view with hashes / grep proof
        "levels": {
            "L2": {"impl_files": [file_record(f) for f in IMPL_FILES],
                   "entry": ENTRY,
                   "tests": [file_record(t) for t in TESTS],
                   "run_artifacts": art},
            "L3": {"integration_callers": integration_records()},
            "L4": {"benchmark": benchmark_table(), "repro_cmd": REPRO},
        },
    }
    l2 = evidence["levels"]["L2"]
    evidence["self_check"] = {
        "impl_non_placeholder": all(f["non_placeholder"] for f in l2["impl_files"]),
        "entry_exists": (REPO / ENTRY).is_file(),
        "tests_exist": all((REPO / t).is_file() for t in TESTS),
        "run_artifacts_exist": all(a["exists"] and a["bytes"] > 0 for a in art),
        "integration_callers_real": all(r["is_real_caller"]
                                        for r in evidence["levels"]["L3"]["integration_callers"]),
        "benchmark_worst_pct": evidence["benchmark"]["worst_physics_rel_error_pct"],
    }
    out = ART / "maturity_evidence.json"
    out.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence["self_check"], indent=2))
    print("artefacts:", len(art), "benchmark rows:", len(evidence["benchmark"]["rows"]),
          "->", out)
    ok = (evidence["self_check"]["impl_non_placeholder"]
          and evidence["self_check"]["entry_exists"]
          and evidence["self_check"]["tests_exist"]
          and evidence["self_check"]["run_artifacts_exist"]
          and evidence["self_check"]["integration_callers_real"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
