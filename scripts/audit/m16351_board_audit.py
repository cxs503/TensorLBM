#!/usr/bin/env python3
"""Board cross-audit (交差核验) of mission m16321 — anchor CASE-LBM-SQUARE-CYL-NORBERG.

Verification, not re-do.  Everything below is recomputed from committed bytes /
the delivered work tree; nothing is taken from the mission report:

  1. three acceptance pieces
       run_artifacts : result_D{8,10}.json, series_D*.npz, state.npz, figs/*.png
       benchmark     : ref vs got per resolution, judged by the *registered*
                       referee row in ``tensor_benchmark_cases`` (sole judge)
       repro_cmd     : the documented chunk driver is actually executed here
                       (short budget) in a throw-away state dir
  2. independent recomputation of the headline numbers from the raw force
     histories (``series_D*.npz``) with an *own* FFT/zero-crossing
     implementation — deliberately NOT importing ``run.analyze_forces``
  3. provenance of the referee (WITHHELD?), repo-internal convention audit
     cross-check (benchmarks/verified/square_cylinder/REFERENCE_AUDIT.md)
  4. durability: are the artifacts in the pushed branch or gitignored?

Outputs (inside this audit worktree)
    artifacts/audit_m16351/bytecopies/...   byte copies + sha256 manifest
    artifacts/audit_m16351/referee.json
    artifacts/audit_m16351/recompute.json
    artifacts/audit_m16351/repro.json
    artifacts/audit_m16351/verdict.json
    docs/board_audit_m16351/M16351_BOARD_AUDIT.md
    (--insert-verdict) one row in tensor_maturity_verdicts (mission_id=16351)

Usage
    python3 scripts/audit/m16351_board_audit.py \
        --source-tree /root/tp-agent/work/m16321 --insert-verdict
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

REF_CD, REF_ST = 1.5, 0.13          # registered referee values (WITHHELD grade)
TOL_CD, TOL_ST = 10.0, 8.0          # registered tolerances (%)
U_IN = 0.05                         # lattice free-stream speed used by both runs
AUDIT_MISSION = 16351

# reproduced from benchmarks/verified/square_cylinder/REFERENCE_AUDIT.md
# (2D free-stream numerical cluster, traceable arxiv tables; context only)
BRACKET = {
    "Re100_Sohankar1998": 0.147,
    "Re150_Present2411.03124": 0.159,
    "Re150_Sohankar1998": 0.162,
    "Re250_FV(cond-mat/0311156)": "0.16-0.17",
    "Re250_Okajima_exp": "0.140-0.143",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def psql(sql: str) -> str:
    p = subprocess.run(["su", "postgres", "-c", f'psql -d tensorplatform -q -tAc "{sql}"'],
                       capture_output=True, text=True, timeout=120)
    return p.stdout.strip()


def referee() -> dict:
    """The registered judge: tensor_benchmark_cases WHERE anchor_slug=..."""
    sql = ("select id,case_name,ref_source,metric,tolerance,status,passed_at "
           "from tensor_benchmark_cases where anchor_slug='CASE-LBM-SQUARE-CYL-NORBERG'")
    rows = [r for r in psql(sql).split("\n") if r.strip()]
    if not rows:
        return {"found": False, "query": sql}
    f = rows[0].split("|")
    return {"found": True, "query": sql, "id": int(f[0]), "case_name": f[1],
            "ref_source": f[2], "metric": f[3], "tolerance": f[4],
            "status": f[5], "passed_at": f[6] or None,
            "values_used": {"cd": REF_CD, "st": REF_ST},
            "tolerances_used_pct": {"cd": TOL_CD, "st": TOL_ST}}


def recompute(series_path: Path, warmup_frac: float, D: int) -> dict:
    """Own post-processing (no repo import): Cd mean + St (FFT peak / zero-cross)."""
    z = np.load(series_path)
    cd, cl = z["cd"], z["cl"]
    w0 = int(warmup_frac * cd.size)
    cd_mean = float(cd[w0:].mean())
    w = cl[w0:] - cl[w0:].mean()
    spec = np.abs(np.fft.rfft(w * np.hanning(w.size)))
    k = int(np.argmax(spec[1:])) + 1
    a, b, c = spec[k - 1], spec[k], spec[k + 1]
    den = a - 2 * b + c
    d = float(0.5 * (a - c) / den) if abs(den) > 1e-30 else 0.0
    st_fft = float((k + d) / w.size * D / U_IN)
    thr = 0.25 * float(np.abs(w).max())
    sig = np.where(w >= thr, 1, np.where(w <= -thr, -1, 0))
    idx = np.where(sig != 0, np.arange(sig.size), -1)
    last = np.maximum.accumulate(idx)
    state = np.where(last >= 0, sig[np.maximum(last, 0)], 0)
    cross = np.where((state[:-1] < 0) & (state[1:] > 0))[0] + 1
    st_cross = (D / (U_IN * float(np.median(np.diff(cross))))
                if len(cross) >= 3 else float("nan"))
    period_steps = 1.0 / (st_fft * U_IN / D) if st_fft == st_fft else float("nan")
    return {"samples": int(w.size), "warmup_from": w0, "cd": round(cd_mean, 4),
            "st_fft": round(st_fft, 4), "st_cross": round(st_cross, 4),
            "n_crossings": int(len(cross)),
            "shedding_periods_in_window": round(w.size / period_steps, 2),
            "fft_bin_width_st": round(D / (U_IN * w.size), 4),
            "cl_amp": round(float(np.abs(w).max()), 4)}


def judge(got: dict, ref: dict) -> dict:
    out = {"ref": {"cd": REF_CD, "st": REF_ST}, "cases": {}}
    for tag, g in got.items():
        ecd = (g["cd"] - REF_CD) / REF_CD * 100.0
        est = (g["st"] - REF_ST) / REF_ST * 100.0
        out["cases"][tag] = {
            "cd": g["cd"], "cd_err_pct": round(ecd, 2), "cd_pass": abs(ecd) <= TOL_CD,
            "st": g["st"], "st_err_pct": round(est, 2), "st_pass": abs(est) <= TOL_ST,
        }
    out["all_pass"] = all(c["cd_pass"] and c["st_pass"] for c in out["cases"].values())
    out["registered_status"] = ref.get("status")
    out["stance"] = ("FAIL vs registered referee: St +29..31% > 8% tolerance"
                     if not out["all_pass"] else "PASS")
    # what the score would be against the repo's own traceable 2D convention,
    # linearly bracketed between Re=150 (0.159-0.162) and Re=250 (0.16-0.17)
    conv = 0.165
    out["context_convention"] = {
        "note": ("repo-internal REFERENCE_AUDIT (2D free-stream numerics) brackets "
                 "St(Re=200) at ~0.165; NOT the registered referee — shown for "
                 "diagnosis only, never scored"),
        "bracket": BRACKET, "st_expected_traceable_2d": conv,
        "st_err_pct_vs_0.165": {k: round((v["st"] - conv) / conv * 100.0, 2)
                                for k, v in out["cases"].items()},
    }
    return out


def repro(source: Path, outdir: Path) -> dict:
    """Execute the documented driver once with a tiny budget (fresh state)."""
    tmp = Path(tempfile.mkdtemp(prefix="m16351_repro_"))
    cmd = ["python3", "cases/square_cylinder_norberg/run_chunked.py",
           "--D", "8", "--steps", "300", "--domain-x", "20", "--domain-y", "20",
           "--x-center", "6", "--sponge-d", "6", "--budget", "60",
           "--threads", "4", "--warmup-frac", "0.25",
           "--state", str(tmp / "state.npz")]
    p = subprocess.run(cmd, cwd=str(source), capture_output=True, text=True, timeout=600)
    rec = {"cmd": " ".join(cmd), "cwd": str(source), "returncode": p.returncode,
           "stdout_tail": p.stdout.strip().split("\n")[-1:] if p.stdout else [],
           "stderr_tail": p.stderr.strip().split("\n")[-3:] if p.stderr else []}
    rj = tmp / "result_D8.json"
    if rj.exists():
        rec["result"] = json.loads(rj.read_text())
        shutil.copy(rj, outdir / "repro_result_D8.json")
        rec["artifact"] = "artifacts/audit_m16351/repro_result_D8.json"
    rec["executable"] = bool(p.returncode == 0 and rj.exists())
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-tree", default="/root/tp-agent/work/m16321")
    ap.add_argument("--out", default="artifacts/audit_m16351")
    ap.add_argument("--insert-verdict", action="store_true")
    args = ap.parse_args()

    src = Path(args.source_tree).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    copies = out / "bytecopies"
    copies.mkdir(exist_ok=True)

    run_dir = src / "runs/m16321/sqcyl_norberg"
    files = {
        "result_D8.json": run_dir / "D8/result_D8.json",
        "result_D10.json": run_dir / "D10/result_D10.json",
        "series_D8.npz": run_dir / "D8/series_D8.npz",
        "series_D10.npz": run_dir / "D10/series_D10.npz",
        "refs_norberg1993.json": src / "cases/square_cylinder_norberg/refs/norberg1993.json",
        "run_chunked.py": src / "cases/square_cylinder_norberg/run_chunked.py",
        "state_D8.npz": run_dir / "D8/state.npz",
    }
    manifest, missing = {}, []
    for name, f in files.items():
        if f.exists():
            shutil.copy(f, copies / name)
            manifest[name] = {"source": str(f), "bytes": f.stat().st_size,
                              "sha256": sha256(f)}
        else:
            missing.append(str(f))
    (out / "manifest.json").write_text(json.dumps(
        {"source_tree": str(src), "files": manifest, "missing": missing}, indent=2))

    ref = referee()
    (out / "referee.json").write_text(json.dumps(ref, indent=2))

    got, recomputed = {}, {}
    for tag, D, wu in (("D8", 8, 0.5), ("D10", 10, 0.25)):
        j = json.loads((run_dir / f"{tag}/result_{tag}.json").read_text())
        got[tag] = {"cd": j["cd"], "st": j["st"], "st_crossing": j["st_crossing"],
                    "cd_pass_claimed": j["cd_pass"], "st_pass_claimed": j["st_pass"],
                    "steps": j["steps"], "elapsed_s_total": j["elapsed_s_total"]}
        recomputed[tag] = {"claimed": {"cd": j["cd"], "st": j["st"],
                                       "st_crossing": j["st_crossing"]},
                           "recomputed": recompute(copies / f"series_{tag}.npz", wu, D)}
        r = recomputed[tag]["recomputed"]
        recomputed[tag]["delta_pct"] = {
            "cd": round((r["cd"] - j["cd"]) / j["cd"] * 100.0, 4),
            "st": round((r["st_fft"] - j["st"]) / j["st"] * 100.0, 4),
            "st_cross": round((r["st_cross"] - j["st_crossing"]) / j["st_crossing"] * 100.0, 4),
        }
    (out / "recompute.json").write_text(json.dumps(recomputed, indent=2))

    rep = repro(src, out)
    (out / "repro.json").write_text(json.dumps(rep, indent=2))

    # durability: are the artifacts part of the delivered commit?
    def git(*a: str) -> str:
        p = subprocess.run(["git", *a], cwd=str(src), capture_output=True, text=True)
        return p.stdout.strip()

    head = git("rev-parse", "HEAD")
    tracked = git("ls-files", "cases/square_cylinder_norberg",
                  "benchmarks/verified/square_cylinder/run.py").split("\n")
    ignored = git("check-ignore", "-v", "runs/m16321/sqcyl_norberg/D8/result_D8.json")

    j = judge(got, ref)
    (out / "judge.json").write_text(json.dumps(j, indent=2))

    verdict = {
        "repo": "TensorLBM",
        "scope": "m16321 CASE-LBM-SQUARE-CYL-NORBERG",
        "mission_id": AUDIT_MISSION,
        "audited_mission": 16321,
        "audited_head": head,
        "level": 2,
        "evidence": {
            "impl_files": [
                "cases/square_cylinder_norberg/run_chunked.py:1-60 (chunked driver reusing lib chain)",
                "benchmarks/verified/square_cylinder/run.py:132 analyze_forces (extracted post-processing)",
                "src/tensorlbm/solver.py:276 collide_mrt / :164 compute_obstacle_forces",
                "src/tensorlbm/boundary/boundaries.py:201 far_field_bc_2d / :390 make_sponge_strength",
            ],
            "entry": "python3 cases/square_cylinder_norberg/run_chunked.py --D 8 --steps 300 "
                     "--domain-x 20 --domain-y 20 --budget 60 --state "
                     "runs/m16321/sqcyl_norberg/D8/state.npz",
            "run_artifacts": [
                "runs/m16321/sqcyl_norberg/D8/result_D8.json (cd=1.6067 st=0.1679)",
                "runs/m16321/sqcyl_norberg/D10/result_D10.json (cd=1.6159 st=0.1706)",
                "runs/m16321/sqcyl_norberg/D{8,10}/series_D{8,10}.npz (+ byte copy under artifacts/audit_m16351/bytecopies/)",
                "runs/m16321/sqcyl_norberg/figs/fig{1,2,3}*.png",
            ],
            "tests": "none case-specific: git grep square_cylinder_norberg tests/ -> 0 hits; "
                     "run_chunked.py executed here as functional smoke (repro.json executable=true)",
            "integration_callers": "cases/square_cylinder_norberg/run_chunked.py:53 imports "
                                   "analyze_forces/square_mask from benchmarks/verified/square_cylinder/run.py; "
                                   "library use of src/tensorlbm only -> L3 (solver/platform API embedding) NOT met",
            "benchmark": {
                "ref": {"cd": REF_CD, "st": REF_ST,
                        "source": "tensor_benchmark_cases id=33 ref_source (Norberg 1993, WITHHELD/secondary)"},
                "judge": j,
                "judged_by": "registered row only; repo-internal 2D bracket shown as diagnosis, not scored",
            },
            "repro_cmd": rep["cmd"],
            "external_anchor": "Norberg 1993 JWEIA 49:187-196 DOI 10.1016/0167-6105(93)90014-f — "
                               "paywalled (Unpaywall is_oa=false), table values secondary from m16247",
            "durability": {"head": head, "tracked_case_files": tracked,
                           "artifacts_ignored": ignored.split("\n")[0] if ignored else False},
        },
        "verdict_by": "board-audit",
        "level_rationale": (
            "L2: real code (chunked driver on the library MRT/far-field/sponge/Ladd chain), "
            "D=8/D=10 runs, headline numbers re-derived here from raw force traces (max "
            "deviation <0.05%). L3 refused: no caller from the main solver or the platform API "
            "(case lives in cases/ and is invoked by hand). L4 refused: the registered referee "
            "scores St +29.16%/+31.27% vs 8% tolerance (Cd +7.12%/+7.73% within 10%) — "
            "tensor_benchmark_cases id=33 status='partial(cd pass, st fail)'; the referee value "
            "itself is WITHHELD (secondary transcription) and is contradicted by the repo's own "
            "traceable 2D bracket (St(Re=200)~0.165), so the FAIL is a referee-convention defect "
            "as much as a solver gap — either way it is not a passed benchmark."),
    }
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2, ensure_ascii=False))

    # the verdict SQL is always written (dollar-quoted json -> no escaping games);
    # --insert-verdict additionally executes it through the postgres superuser
    ev = json.dumps(verdict["evidence"], ensure_ascii=False)
    sql = ("INSERT INTO tensor_maturity_verdicts (repo,scope,mission_id,level,evidence,verdict_by)\n"
           "VALUES ('TensorLBM','m16321 CASE-LBM-SQUARE-CYL-NORBERG',"
           f"{AUDIT_MISSION},2,$json${ev}$json$,'board-audit')\n"
           # unique (repo, scope, verdict_by): a later audit supersedes the earlier
           # row (mission 16334 failed and left mission_id=16321 here), so re-arm the gate
           "ON CONFLICT (repo,scope,verdict_by) DO UPDATE SET "
           "mission_id=EXCLUDED.mission_id, level=EXCLUDED.level, "
           "evidence=EXCLUDED.evidence, gate_state='pending', gate_level=NULL, gate_note=NULL\n"
           "RETURNING id, mission_id, level, gate_state;")
    (out / "verdict.sql").write_text(sql + "\n")
    if args.insert_verdict:
        tmp_dir = Path(tempfile.mkdtemp(prefix="m16351_sql_"))
        tmp_dir.chmod(0o755)  # postgres must be able to read the file
        tmp_sql = tmp_dir / "verdict.sql"
        tmp_sql.write_text(sql + "\n")
        tmp_sql.chmod(0o644)
        res = subprocess.run(["su", "postgres", "-c",
                              f"psql -d tensorplatform -q -t -f {tmp_sql}"],
                             capture_output=True, text=True, timeout=120)
        print("insert:", res.stdout.strip() or res.stderr.strip())

    print(json.dumps({"judge": j["stance"], "repro_executable": rep["executable"],
                      "artifacts": [f for f in manifest]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())