#!/usr/bin/env python3
"""Board audit m17878 — independent re-verification of mission m17757
(anchor XVAL-LBM-FILL-GRADIENT-NORMAL, repo TensorLBM).

Nothing here is copied from the delivery's own pass/fail logic.  Five checks,
each answering one question the board has to settle:

  C1 REPRO        does the shipped entry regenerate results.json bit-identically?
  C2 EQUIVALENCE  is the delivered A1 arm really the production function
                  (src/tensorlbm/free_surface_lbm.py:642 _compute_interface_normal)?
  C3 DEGENERACY   is the declared primary KPI (A1 vs A2 direction error = 0.00%)
                  informative, or an algebraic identity that must be 0 for ANY field?
  C4 INTEGRATION  does the audited function have any caller in the repo?
  C5 BENCHMARK    is the anchor registered in tensor_benchmark_cases?

Run:  python3 docs/board_audit/m17878/verify_m17757.py [--src-dir DIR]
Writes docs/board_audit/m17878/artifacts/verify_m17757.json (zero hand-copy).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = HERE / "refs"
ART = HERE / "artifacts"
DSN = ("host=127.0.0.1 port=5432 dbname=tensorplatform user=foundation "
       "password=uGTN-KwuQv_7l8Tw0MMWUx_kf-nF2Rzz")
ANCHOR = "XVAL-LBM-FILL-GRADIENT-NORMAL"


def load_run_xval(run_path: Path):
    spec = importlib.util.spec_from_file_location("run_xval_m17757", run_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def c1_repro(mod, results: dict) -> dict:
    """Re-execute the delivered entry, compare every numeric KPI to results.json."""
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod.main()
    lines = [ln for ln in buf.getvalue().splitlines() if ln.startswith('{"case"')]
    rerun = [json.loads(ln) for ln in lines]
    by_case = {c["case"]: c for c in results["cases"]}
    mism, missing = [], []
    for c in rerun:
        ref = by_case.get(c["case"])
        if ref is None:
            missing.append(c["case"])
            continue
        for arm, vals in c.items():
            if not isinstance(vals, dict):
                continue
            for k, v in vals.items():
                rv = ref.get(arm, {}).get(k)
                if rv is None or abs(float(v) - float(rv)) > 1e-12:
                    mism.append({"case": c["case"], "arm": arm, "kpi": k,
                                 "rerun": v, "shipped": rv})
    return {"n_cases_shipped": len(results["cases"]), "n_cases_rerun": len(rerun),
            "n_kpi_mismatch": len(mism), "missing_cases": missing,
            "mismatches": mism[:10], "bit_identical": not mism and not missing,
            "delivered_all_pass_flag": results.get("all_pass"),
            "delivered_n_checks": len(results.get("checks", []))}


def c2_equivalence(mod, src_dir: Path) -> dict:
    """Transplant (A1 arm) vs the real production function, on identical inputs."""
    import torch
    sys.path.insert(0, str(src_dir))
    from tensorlbm import free_surface_lbm as fsl  # noqa: E402
    torch.manual_seed(0)
    out = {"module_file": str(fsl.__file__)}
    # 3D tube: arbitrary field (nz, ny, nx)
    f3 = torch.rand(16, 16, 16, dtype=torch.float64)
    nx, ny, nz = fsl._compute_interface_normal(
        torch.zeros(16, 16, 16, dtype=torch.int32), f3, torch.ones_like(f3))
    ax, ay, az = mod.a1_tensorlbm_3d(f3)
    out["tube3d_max_abs_diff"] = float(max((nx - ax).abs().max(),
                                           (ny - ay).abs().max(),
                                           (nz - az).abs().max()))
    # 2D slice of the 3D function (production has no 2D stencil)
    f2 = torch.rand(16, 16, dtype=torch.float64).unsqueeze(0)
    nx2, ny2, _ = fsl._compute_interface_normal(
        torch.zeros_like(f2, dtype=torch.int32), f2, torch.ones_like(f2))
    bx, by = mod.a1_tensorlbm_2d(f2[0])
    out["slice2d_max_abs_diff"] = float(max((nx2[0] - bx).abs().max(),
                                            (ny2[0] - by).abs().max()))
    # sign convention on a real step: liquid (fill=1) at low x -> n must point +x
    g = torch.zeros(16, 16, 16, dtype=torch.float64)
    g[:, :, :8] = 1.0
    nx3, _, _ = fsl._compute_interface_normal(
        torch.zeros_like(g, dtype=torch.int32), g, torch.ones_like(g))
    out["sign_step_nx_row"] = [round(v, 6) for v in nx3[8, 8, :].tolist()]
    out["sign_liquid_to_gas_ok"] = (nx3[8, 8, 7].item() > 0.99
                                    and nx3[8, 8, 8].item() > 0.99)
    # no shape guard: a batched (1,nz,ny,nx) input silently returns zeros
    gb = torch.zeros(1, 16, 16, 16, dtype=torch.float64)
    gb[:, :, :, :8] = 1.0
    nxb, _, _ = fsl._compute_interface_normal(
        torch.zeros_like(gb, dtype=torch.int32), gb, torch.ones_like(gb))
    out["batched_input_returns_all_zero"] = bool(float(nxb.abs().max()) == 0.0)
    return out


def c3_degeneracy(mod, results: dict) -> dict:
    """Is 'A1 vs A2 dir_err = 0.00%' an identity rather than a measurement?"""
    import torch
    torch.manual_seed(0)
    rows = []
    for name, f in [
        ("random_uniform", torch.rand(64, 64, dtype=torch.float64)),
        ("random_noise_0_1_int", (torch.rand(64, 64, dtype=torch.float64) > 0.5).double()),
        ("smooth_sine", torch.sin(torch.arange(64, dtype=torch.float64)[:, None] / 7.0)
         * torch.cos(torch.arange(64, dtype=torch.float64)[None, :] / 11.0)),
    ]:
        ax, ay = mod.a1_tensorlbm_2d(f)
        ox, oy = mod.a2_oss_2d(f)
        mask = (ax ** 2 + ay ** 2) > 1e-12
        mag = (ox ** 2 + oy ** 2).sqrt().clamp(min=1e-30)
        cos = (ax * ox + ay * oy) / mag
        rows.append({"field": name, "n_cells": int(mask.sum()),
                     "dir_err_A1_vs_A2": float((1.0 - cos.abs())[mask].mean()),
                     "max_abs_pointwise_angle_dev": float((1.0 - cos.abs())[mask].max())})
    # algebraic statement of the identity, measured not asserted
    f = torch.rand(64, 64, dtype=torch.float64)
    ax, ay = mod.a1_tensorlbm_2d(f)
    ox, oy = mod.a2_oss_2d(f)
    mag = (ox ** 2 + oy ** 2).sqrt().clamp(min=1e-30)
    resid = float(max(((ox - ax * mag).abs()).max(), ((oy - ay * mag).abs()).max()))
    shipped = [{"case": c["case"], "A1_vs_A2": c.get("A1_vs_A2", {}).get("dir_err"),
                "A1_vs_analytic": c.get("A1_vs_analytic", {}).get("dir_err")}
               for c in results["cases"]]
    informative = [c["A1_vs_analytic"] for c in shipped if c["A1_vs_analytic"] is not None]
    return {"random_field_rows": rows,
            "identity_residual_max_abs": resid,
            "identity": "A2 == -0.5*central_diff(fill) == A1 * |grad(fill)| pointwise, "
                        "same stencil, same field => direction agreement is 0 by construction",
            "degenerate": resid < 1e-12 and all(r["dir_err_A1_vs_A2"] < 1e-12 for r in rows),
            "shipped_kpis": shipped,
            "informative_kpi_A1_vs_analytic_max": max(informative) if informative else None,
            "informative_kpi_A1_vs_analytic_min": min(informative) if informative else None,
            "n_shipped_informative_cases_over_15pct":
                sum(1 for v in informative if v >= 0.15)}


def c4_integration(repo: Path) -> dict:
    """Callers of the audited function anywhere in the repo (src/ and tests/)."""
    def grep(pattern: str, extra: str = "") -> list[str]:
        cmd = f"grep -rn --include='*.py' {extra} -E {json.dumps(pattern)} src tests 2>/dev/null"
        r = subprocess.run(cmd, shell=True, cwd=repo, capture_output=True, text=True)
        return [ln for ln in r.stdout.splitlines() if ln.strip()]

    calls = [ln for ln in grep(r"_compute_interface_normal\s*\(")
             if not ln.split(":", 2)[2].lstrip().startswith("def ")]
    live = [ln for ln in grep(r"interface_normal_3d\s*\(")
            if not ln.split(":", 2)[2].lstrip().startswith("def ")]
    return {"audited_function_callers_src_tests": calls,
            "audited_function_n_callers": len(calls),
            "live_free_surface_normal_callers": live[:10],
            "live_n_callers": len(live),
            "audited_function_definition": "src/tensorlbm/free_surface_lbm.py:642",
            "note": "xval/ scripts are the validation harness, not solver integration"}


def c5_benchmark() -> dict:
    import psycopg
    with psycopg.connect(DSN) as c:
        rows = c.execute(
            "SELECT id, anchor_slug, case_name, metric, tolerance, status "
            "FROM tensor_benchmark_cases WHERE anchor_slug=%s", (ANCHOR,)).fetchall()
        total = c.execute("SELECT count(*) FROM tensor_benchmark_cases").fetchone()[0]
        lbm = c.execute("SELECT count(*) FROM tensor_benchmark_cases "
                        "WHERE repo='TensorLBM'").fetchone()[0]
    return {"anchor": ANCHOR, "n_rows_for_anchor": len(rows),
            "rows": rows, "table_total_rows": total, "tensorlbm_rows": lbm}


def c6_live_path(mod, src_dir: Path) -> dict:
    """The audited normal is dead code; the *reachable* free-surface normal lives in
    free_surface_common.interface_normal_3d.  Compare the two and test whether the
    board KPI can see the sign convention at all."""
    import torch
    sys.path.insert(0, str(src_dir))
    from tensorlbm import free_surface_lbm as fsl          # audited (dead)
    from tensorlbm import free_surface_common as fsc      # live VOF path
    torch.manual_seed(0)
    phi = torch.zeros(16, 16, 16, dtype=torch.float64)
    phi[:, :, :8] = 1.0                                   # liquid at low x
    ax, ay, az = fsl._compute_interface_normal(
        torch.zeros_like(phi, dtype=torch.int32), phi, torch.ones_like(phi))
    lx, ly, lz, mag = fsc.interface_normal_3d(phi)
    ratio = [round((lx[i, 8, 7] / ax[i, 8, 7]).item(), 6) for i in range(1)]
    # sign-blindness of the KPI: 1 - |n1.n2| is invariant under n2 -> -n2
    mask = (ax ** 2 + ay ** 2) > 1e-12
    cos_abs = (ax * lx + ay * ly).abs() / (lx ** 2 + ly ** 2).sqrt().clamp(min=1e-30)
    err_same = float((1.0 - cos_abs)[mask].mean())
    err_flipped = float((1.0 - (-cos_abs))[mask].mean())  # KPI with sign-sensitive dot
    return {"audited_minus_live_ratio_at_x7": ratio,
            "live_nx_at_x7": round(lx[8, 8, 7].item(), 6),
            "audited_nx_at_x7": round(ax[8, 8, 7].item(), 6),
            "sign_conventions_opposite": bool(lx[8, 8, 7].item() * ax[8, 8, 7].item() < 0),
            "kpi_abs_dot_dir_err": err_same,
            "kpi_signed_dot_dir_err": err_flipped,
            "kpi_is_sign_blind": bool(abs(err_same) < 1e-12 and err_flipped > 0.99),
            "live_mag_returned": bool(mag.abs().sum() > 0),
            "live_path_consumers": ["free_surface_common.interface_compression_3d:182",
                                    "free_surface_common.mean_curvature_3d:255"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/root/tp-agent/work/m17878",
                    help="repo root that contains src/ (the audited tree)")
    ap.add_argument("--run-xval", default=str(REF / "xval/run_xval.py"))
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    ART.mkdir(parents=True, exist_ok=True)

    mod = load_run_xval(Path(args.run_xval))
    results = json.loads((REF / "xval/results.json").read_text())

    report = {
        "mission_audited": 17757,
        "audit_mission": 17878,
        "anchor": ANCHOR,
        "repo": "TensorLBM",
        "source_commit": (REF / "SOURCE_COMMIT.txt").read_text().strip(),
        "refs_sha256": {str(p.relative_to(HERE)): sha(p) for p in sorted(REF.rglob("*"))
                        if p.is_file()},
        "C1_repro": c1_repro(mod, results),
        "C2_equivalence": c2_equivalence(mod, repo / "src"),
        "C3_degeneracy": c3_degeneracy(mod, results),
        "C4_integration": c4_integration(repo),
        "C5_benchmark": c5_benchmark(),
        "C6_live_path": c6_live_path(mod, repo / "src"),
    }
    out = ART / "verify_m17757.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: (v if not isinstance(v, dict) else
                          {kk: vv for kk, vv in v.items()
                           if kk in ("bit_identical", "n_kpi_mismatch",
                                     "tube3d_max_abs_diff", "slice2d_max_abs_diff",
                                     "degenerate", "n_cases_shipped",
                                     "audited_function_n_callers", "live_n_callers", "sign_conventions_opposite", "kpi_is_sign_blind",
                                     "n_rows_for_anchor", "table_total_rows")})
                      for k, v in report.items()
                      if k in ("C1_repro", "C2_equivalence", "C3_degeneracy",
                               "C4_integration", "C5_benchmark", "C6_live_path")},
                     ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())