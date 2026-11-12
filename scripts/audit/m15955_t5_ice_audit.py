#!/usr/bin/env python3
"""m15955 board audit — independent re-derivation for GAP-LBM-T5-ICE-COUPLING (m15919).

Adversarial-review discipline: every number the audited mission reports is
recomputed here from an *independent* implementation (numpy, hand-rolled
UV-sphere facet geometry, beam closed form) instead of importing the mission's
own helpers.  Nothing is taken on trust from ``runs/t5_ice_coupling/*.json``.

Outputs ``artifacts/m15955_audit/audit_matrix.json``.

Run:  python3 scripts/audit/m15955_t5_ice_audit.py
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "m15955_audit"
MISSION_RUNS = Path("/root/tp-agent/work/m15919/runs/t5_ice_coupling")


# --------------------------------------------------------------------------
# independent geometry: UV-sphere triangles, exact facet areas, own normals
# --------------------------------------------------------------------------
def uv_sphere_hemisphere(n_theta: int, n_phi: int, radius: float = 1.0):
    """Curved part of a hemisphere (z >= 0) as a triangle soup.

    Independent of tensorlbm.three_way.interface3d: same *geometric object*
    (UV grid on the sphere), separate code path.
    """
    th = np.linspace(0.0, math.pi / 2.0, n_theta + 1)
    ph = np.linspace(0.0, 2.0 * math.pi, n_phi + 1)

    def vert(i, j):
        p = ph[j % n_phi]
        return np.array([math.sin(th[i]) * math.cos(p),
                         math.sin(th[i]) * math.sin(p),
                         math.cos(th[i])]) * radius

    area = np.zeros((n_theta, n_phi))
    nz_sum = 0.0
    ny_sum = 0.0
    nx_sum = 0.0
    for i in range(n_theta):
        for j in range(n_phi):
            a, b = vert(i, j), vert(i, j + 1)
            c, d = vert(i + 1, j), vert(i + 1, j + 1)
            # two triangles (a,c,d) and (a,d,b); polar band degenerates to one
            tris = [(a, c, d), (a, d, b)]
            if i == 0:                      # north pole: one non-degenerate tri
                tris = [(a, c, d)]
            for t in tris:
                cr = np.cross(t[1] - t[0], t[2] - t[0])
                A = 0.5 * np.linalg.norm(cr)
                if A < 1e-15:
                    continue
                n = cr / (2.0 * A)
                # outward orientation: dot with centroid direction
                if np.dot(n, (t[0] + t[1] + t[2]) / 3.0) < 0:
                    n = -n
                area[i, j] += A
                nz_sum += n[2] * A
                ny_sum += n[1] * A
                nx_sum += n[0] * A
    return area, np.array([nx_sum, ny_sum, nz_sum])


def full_sphere_closedness(n_theta=10, n_phi=20, radius=1.0):
    """sum(n dA) over a closed UV sphere (both hemispheres), own code."""
    th = np.linspace(0.0, math.pi, n_theta + 1)
    ph = np.linspace(0.0, 2.0 * math.pi, n_phi + 1)

    def vert(i, j):
        p = ph[j % n_phi]
        return np.array([math.sin(th[i]) * math.cos(p),
                         math.sin(th[i]) * math.sin(p),
                         math.cos(th[i])]) * radius

    tot = np.zeros(3)
    for i in range(n_theta):
        north = (i == 0)
        south = (i == n_theta - 1)
        for j in range(n_phi):
            a, b = vert(i, j), vert(i, j + 1)
            c, d = vert(i + 1, j), vert(i + 1, j + 1)
            if north:
                tris = [(a, c, d)]
            elif south:
                tris = [(a, d, b)]
            else:
                tris = [(a, c, d), (a, d, b)]
            for t in tris:
                cr = np.cross(t[1] - t[0], t[2] - t[0])
                A = 0.5 * np.linalg.norm(cr)
                if A < 1e-15:
                    continue
                n = cr / (2.0 * A)
                if np.dot(n, (t[0] + t[1] + t[2]) / 3.0) < 0:
                    n = -n
                tot += n * A
    return tot


def beam_tip_deflection(F, L, E, width, height):
    I = width * height ** 3 / 12.0
    return F * L ** 3 / (3.0 * E * I)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    audit: dict = {"mission": 15919, "anchor": "GAP-LBM-T5-ICE-COUPLING",
                   "auditor": "m15955"}

    # ---------- A1: hemisphere pressure force, independent recompute ----------
    claimed = json.loads((MISSION_RUNS / "benchmark.json").read_text())
    claimed_i1 = {c["description"]: c for c in claimed["cases"] if c["gate"] == "I1"}
    rows = []
    for (n_th, n_ph) in [(8, 16), (16, 32), (24, 48)]:
        area, nvec = uv_sphere_hemisphere(n_th, n_ph)
        # Surface3D.pressure_force(p) = -p * sum(n dA); the entry script reports
        # fz = -pressure_force[2], i.e. the magnitude of the pressure resultant.
        fz = abs(nvec[2])
        ref = math.pi
        key = f"hemisphere {n_th}x{n_ph}"
        rows.append({
            "n_theta": n_th, "n_phi": n_ph,
            "area_sum": float(area.sum()), "area_ref": 2.0 * math.pi,
            "fz_audit": float(fz), "fz_mission": claimed_i1[key]["got"],
            "mission_vs_audit_abs": abs(float(fz) - claimed_i1[key]["got"]),
            "ref": ref,
            "err_audit": abs(abs(float(fz)) - ref) / ref,
            "err_mission_quoted": claimed_i1[key]["rel_err"],
        })
    # convergence order honestly computed in n_theta (markers ~ h^-2)
    if len(rows) >= 2:
        e0, e1 = rows[0]["err_audit"], rows[1]["err_audit"]
        rows[1]["order_true_in_ntheta"] = math.log(e0 / e1) / math.log(2.0)
        m0, m1 = rows[0]["n_theta"] * rows[0]["n_phi"], rows[1]["n_theta"] * rows[1]["n_phi"]
        rows[1]["order_as_mission_reports_in_markers"] = math.log(e0 / e1) / math.log(m1 / m0)
    audit["A1_hemisphere"] = {
        "rows": rows,
        "pass_mission_numbers_reproducible": all(r["mission_vs_audit_abs"] < 1e-12 for r in rows),
    }

    # ---------- A2: closed-surface cancellation ----------
    tot = full_sphere_closedness()
    audit["A2_closed_surface"] = {
        "sum_ndA_audit": float(np.abs(tot).max()),
        "mission_quoted": 4.718447854656915e-16,
        "both_below_1e-14": bool(np.abs(tot).max() < 1e-14),
    }

    # ---------- A3: beam tip deflection closed form ----------
    E, w, h, L, F = 2.1e11, 0.1, 0.01, 1.0, 100.0
    audit["A3_beam"] = {
        "w_tip_audit": beam_tip_deflection(F, L, E, w, h),
        "mission_quoted": 0.01904761904762051,
        "rel_diff": abs(beam_tip_deflection(F, L, E, w, h) - 0.01904761904762051)
        / beam_tip_deflection(F, L, E, w, h),
        "note": "closed form and a 2-node-exact FEM agree; this verifies the FEM "
                "assembly, NOT a coupled three-way response",
    }

    # ---------- A4: provenance of the LBM leg ----------
    res = json.loads((ROOT / "benchmarks/verified/sphere_re100/result.json").read_text())

    def find_cd(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "cd" and isinstance(v, (int, float)):
                    return float(v)
                r = find_cd(v)
                if r is not None:
                    return r
        elif isinstance(obj, list):
            for v in obj:
                r = find_cd(v)
                if r is not None:
                    return r
        return None

    cd_json = find_cd(res)
    cd_ref = 24.0 / 100.0 * (1.0 + 0.15 * 100.0 ** 0.687)
    audit["A4_lbm_leg"] = {
        "mission_C_D_got": 1.12198,
        "verified_record_cd": cd_json,
        "quote_matches_record": bool(cd_json is not None and abs(cd_json - 1.12198) < 5e-6),
        "schiller_naumann_ref": cd_ref,
        "err_pct_recomputed": abs(1.12198 - cd_ref) / cd_ref * 100.0,
        "lbm_solver_invoked_in_mission_scripts": False,
        "evidence": "scripts/t5_ice_coupling_integration.py:57-58 hard-codes "
                    "C_D_ref=1.09173 / C_D_got=1.12198; the module imports are only "
                    "three_way.interface3d + beam_fem (grep '^from' -> no engine/DEM import)",
    }

    # ---------- A5: referee coverage ----------
    gates = json.loads((MISSION_RUNS / "gates.json").read_text())["gates"]
    bench_gates = sorted({c["gate"] for c in claimed["cases"]})
    audit["A5_referee"] = {
        "gates_in_gates_json": sorted(gates.keys()),
        "gates_with_benchmark_row": bench_gates,
        "gates_without_benchmark_row": sorted(set(gates.keys()) - set(bench_gates)),
        "n_benchmark_cases": len(claimed["cases"]),
        "thresholds_are_author_chosen": sorted({c["threshold"] for c in claimed["cases"]}),
        "tensor_benchmark_cases_rows_for_anchor": None,  # filled by caller via psql
    }

    # ---------- A6: artifact custody ----------
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    audit["A6_artifacts"] = {
        "runs_dir_in_commit": bool(git("ls-tree", "-r", "--name-only", "HEAD", "--", "runs/t5_ice_coupling")),
        "scripts_in_commit": git("ls-tree", "-r", "--name-only", "HEAD", "--", "scripts/t5_ice_coupling_entry.py"),
        "mission_run_files_present_on_disk": sorted(
            str(p.relative_to(MISSION_RUNS)) for p in MISSION_RUNS.rglob("*") if p.is_file()),
        "note": "run artifacts exist only in the mission worktree; the merge (PR #1866) "
                "carried scripts + docs + artifacts/ice_hydro_breakup only",
    }

    # ---------- A7: test surface ----------
    proc = subprocess.run([sys.executable, "-m", "pytest", "tests/test_three_way_interface3d.py", "-q"],
                          cwd=ROOT, capture_output=True, text=True,
                          env={**__import__("os").environ, "PYTHONPATH": "src"})
    tail = [ln for ln in proc.stdout.strip().splitlines() if "passed" in ln or "failed" in ln]
    audit["A7_tests"] = {"rc": proc.returncode, "summary": tail[-1] if tail else proc.stdout[-300:]}

    (OUT / "audit_matrix.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2)[:4000])
    print(f"\nwrote {OUT / 'audit_matrix.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())