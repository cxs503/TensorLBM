#!/usr/bin/env python3
"""Independent board audit of mission m17013 (anchor KPI-LBM-VERIFIED-CAPS).

What this script does — and does *not* do
-----------------------------------------
It does NOT import ``benchmarks/verified/nwt_stokes2/run.py`` helpers and it does
not trust the shipped ``result.json``.  It re-derives the verdict from raw
behaviour:

  A1 fresh run       re-execute the shipped driver in a throw-away checkout and
                     diff its output against the shipped ``result.json``
                     (bitwise / key-by-key);
  A2 gate inventory  count gates and *distinct* gate names (repetition inflation
                     check) and read the achieved deviations out of the run;
  A3 referee row     look up ``tensor_benchmark_cases`` for the mission's anchor
                     (machine referee); report 0 rows instead of inventing one;
  A4 coverage probe  mutation testing: sabotage physics that the docstring
                     claims is verified (free-surface elevation) and check
                     whether the gate suite still says PASS;
  A5 control probe   sabotage a *covered* quantity (second-harmonic coefficient)
                     and confirm the suite does fail there (detection power);
  A6 coupling probe  static check whether any solver path is exercised by the
                     driver (no LBM/tank import), and whether a solver-side
                     case for the same capability is shipped as verified or is
                     still pending without a machine verdict.

Run:  python docs/audit_m17013/independent_audit.py [--out <dir>]
Exit: 0 always (this is evidence collection, not a CI gate); the verdict lives in
      ``audit.json`` / ``AUDIT_m17013.md``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CASE = Path("benchmarks/verified/nwt_stokes2")
DRIVER = CASE / "run.py"
SHIPPED = CASE / "result.json"
ANCHOR = "KPI-LBM-VERIFIED-CAPS"
ORIG_MISSION = 17013

# physics surfaces the driver's own docstring claims to referee
CLAIMED_SURFACES = (
    "dispersion relation",
    "second-harmonic surface elevation coefficient",
    "surface boundary conditions (second order in steepness)",
    "divergence-free / potential-matching velocity field",
)
# call sites that would have to appear in the driver to exercise the last two
SURFACE_CALL_SITES = (
    ".elevation(",
    ".potential(",
    ".kinematics(",
    ".surface_bc_residual(",
    ".horizontal_velocity_amplitude_at_swl(",
)


def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def load(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def probe_fresh_run(tmp: Path) -> dict:
    """A1 + A2: re-execute the driver in an isolated copy; compare to shipped."""
    shutil.copytree(REPO / "src", tmp / "src")
    (tmp / CASE).mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / DRIVER, tmp / DRIVER)
    code, out = run([sys.executable, str(tmp / DRIVER)], cwd=tmp)
    fresh_path = tmp / SHIPPED
    fresh = load(fresh_path) if fresh_path.exists() else {}
    shipped = load(REPO / SHIPPED)
    names = [c["gate"] for c in fresh.get("checks", [])]
    return {
        "exit_code": code,
        "stdout_tail": out.strip().splitlines()[-1] if out.strip() else "",
        "gates_passed": fresh.get("gates_passed"),
        "gates_total": fresh.get("gates_total"),
        "verdict": fresh.get("verdict"),
        "gate_calls": len(names),
        "distinct_gate_names": sorted(set(names)),
        "n_distinct_gate_names": len(set(names)),
        "repeated_gate_calls": len(names) - len(set(names)),
        "result_json_identical_to_shipped": fresh == shipped,
        "checks": fresh.get("checks", []),
    }


def probe_referee_row() -> dict:
    """A3: machine referee for the anchor, straight from the platform DB."""
    sql = ("SELECT case_name, metric, tolerance FROM tensor_benchmark_cases "
           f"WHERE anchor_slug='{ANCHOR}';")
    code, out = run(["psql", "-U", "postgres", "-d", "tensorplatform", "-tA", "-F", "|", "-c", sql])
    rows = [ln for ln in out.strip().splitlines() if ln and " " not in ln.split("|")[0][:0]]
    if code != 0:
        return {"query_ok": False, "error": out.strip()[:400], "rows": None}
    rows = [ln for ln in out.strip().splitlines() if "|" in ln]
    return {"query_ok": True, "n_rows": len(rows), "rows": rows}


def _mutate(tmp: Path, patch) -> None:
    p = tmp / "src/tensorlbm/nwt/stokes2.py"
    txt = p.read_text()
    new = patch(txt)
    assert new != txt, "mutation did not apply"
    p.write_text(new)


def probe_mutations() -> dict:
    """A4 + A5: does the suite notice when the physics it claims to referee rots?"""
    base = Path(tempfile.mkdtemp(prefix="m17013mut_"))

    # A4: free-surface elevation (claimed: "surface elevation coefficient" +
    #     docstring "surface boundary conditions") -> destroyed.
    a = base / "A"
    a.mkdir()
    shutil.copytree(REPO / "src", a / "src")
    (a / CASE).mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / DRIVER, a / DRIVER)
    _mutate(a, lambda t: t.replace(
        "    def elevation(self, x, t: float):",
        "    def elevation(self, x, t: float):\n        return 12345.0  # MUTATION A", 1))
    code_a, out_a = run([sys.executable, str(a / DRIVER)], cwd=a)
    verdict_a = load(a / SHIPPED) if (a / SHIPPED).exists() else {}

    # A5 (control): second-harmonic coefficient (covered) -> doubled.
    b = base / "B"
    b.mkdir()
    shutil.copytree(REPO / "src", b / "src")
    (b / CASE).mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / DRIVER, b / DRIVER)
    _mutate(b, lambda t: t.replace(
        "    def second_harmonic_amplitude(self) -> float:",
        "    def second_harmonic_amplitude(self) -> float:\n"
        "        return 2.0 * (0.25 * self.wavenumber * self.amplitude ** 2)  # MUTATION B", 1))
    code_b, out_b = run([sys.executable, str(b / DRIVER)], cwd=b)
    verdict_b = load(b / SHIPPED) if (b / SHIPPED).exists() else {}

    return {
        "A_elevation_destroyed_still_passes": {
            "mutation": "Stokes2Wave.elevation() -> 12345.0",
            "driver_exit": code_a,
            "driver_stdout": out_a.strip().splitlines()[-1] if out_a.strip() else "",
            "verdict": verdict_a.get("verdict"),
            "gates_passed": verdict_a.get("gates_passed"),
            "gates_total": verdict_a.get("gates_total"),
            "detected": verdict_a.get("verdict") != "PASS",
        },
        "B_harmonic_doubled_is_caught": {
            "mutation": "Stokes2Wave.second_harmonic_amplitude() -> x2 (deep-water form)",
            "driver_exit": code_b,
            "driver_stdout": out_b.strip().splitlines()[-1] if out_b.strip() else "",
            "verdict": verdict_b.get("verdict"),
            "gates_passed": verdict_b.get("gates_passed"),
            "gates_total": verdict_b.get("gates_total"),
            "detected": verdict_b.get("verdict") != "PASS",
            "failing_gates": [c["gate"] for c in verdict_b.get("checks", []) if not c["pass"]],
        },
        "tmpdir": str(base),
    }


def probe_coupling() -> dict:
    """A6: is any solver path exercised, and where does solver-side evidence live?"""
    src = (REPO / DRIVER).read_text()
    imports = re.findall(r"^from\s+(\S+)\s+import", src, re.M)
    call_sites = [s for s in SURFACE_CALL_SITES if s in src]
    solved = [s for s in ("tensorlbm.nwt.tank", "tensorlbm.solver", "tensorlbm.d2q9",
                          "tensorlbm.nwt.wavemaker", "tensorlbm.vof_free_surface") if s in src]
    pend = REPO / "benchmarks/pending/nwt_stokes2_2d"
    verdict_files = sorted(p.relative_to(REPO).as_posix() for p in pend.glob("*")
                           if p.name in ("result.json", "validation_status.json", "results.json"))
    ledger = load(REPO / "benchmarks/STATUS_LEDGER.json")
    pend_row = next((r for r in ledger.get("cases", [])
                     if r.get("case", "").startswith("nwt_stokes2")), None)
    return {
        "driver_imports": imports,
        "solver_modules_imported": solved,
        "runs_solver": bool(solved),
        "claimed_surface_call_sites_present": call_sites,
        "missing_surface_call_sites": [s for s in SURFACE_CALL_SITES if s not in src],
        "solver_side_case_dir": pend.relative_to(REPO).as_posix() if pend.exists() else None,
        "solver_side_verdict_files": verdict_files,
        "solver_side_ledger_row": pend_row,
        "note": "a solver-side case for the same capability exists but sits in pending/ "
                "with no machine verdict file",
    }


def probe_provenance() -> dict:
    """Which of the audited files predate the mission (git blame at file level)?"""
    out = {}
    for f in ("src/tensorlbm/nwt/stokes2.py", "tests/test_nwt_stokes2.py", "benchmarks/verified/nwt_stokes2/run.py"):
        code, log = run(["git", "log", "--diff-filter=A", "--format=%h %ad %s", "--date=short",
                         "--", f], cwd=REPO)
        code2, touched = run(["git", "log", "--format=%h %s", "--", f], cwd=REPO)
        out[f] = {
            "added_by": log.strip().splitlines()[-1] if log.strip() else None,
            "commits_touching_file": touched.strip().splitlines(),
        }
    code, stats = run(["git", "show", "--stat", "--format=%h %s", "bb45e2dd"], cwd=REPO)
    out["m17013_commit_stat"] = stats.strip().splitlines()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "docs/audit_m17013"))
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    tmp = Path(tempfile.mkdtemp(prefix="m17013fresh_"))
    audit = {
        "mission": ORIG_MISSION,
        "audit_mission": 17085,
        "anchor": ANCHOR,
        "repo": "TensorLBM",
        "driver": DRIVER.as_posix(),
        "shipped_result": SHIPPED.as_posix(),
        "fresh_run": probe_fresh_run(tmp),
        "machine_referee": probe_referee_row(),
        "mutation_probes": probe_mutations(),
        "coupling": probe_coupling(),
        "provenance": probe_provenance(),
    }
    fr = audit["fresh_run"]
    audit["gate_inventory"] = {
        "gate_calls": fr["gate_calls"],
        "distinct": fr["n_distinct_gate_names"],
        "repeated": fr["repeated_gate_calls"],
        "distinct_names": fr["distinct_gate_names"],
        "claimed_surfaces": list(CLAIMED_SURFACES),
        "surfaces_actually_exercised": [
            "dispersion relation",
            "second-harmonic surface elevation *coefficient*",
            "deep/shallow asymptotes (phase & group speed)",
            "wavenumber inversion round-trip",
        ],
        "surfaces_not_exercised": [
            "surface elevation eta(x,t) itself",
            "surface boundary conditions residual (2nd order in steepness)",
            "divergence-free / potential-matching velocity field",
        ],
    }
    (outdir / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({k: audit[k] for k in ("fresh_run", "machine_referee")}, indent=2)[:1800])
    print("\nmutation A (elevation destroyed):", audit["mutation_probes"]["A_elevation_destroyed_still_passes"])
    print("mutation B (harmonic doubled):  ", {k: v for k, v in audit["mutation_probes"]["B_harmonic_doubled_is_caught"].items() if k != "failing_gates"})
    print("coupling:", {k: audit["coupling"][k] for k in ("runs_solver", "missing_surface_call_sites",
                                                          "solver_side_case_dir", "solver_side_verdict_files")})
    print("wrote", outdir / "audit.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())