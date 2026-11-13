#!/usr/bin/env python3
"""Board audit m17856 - delta-verification of m17758 ``XVAL-LBM-FLUCTUATING-LBM-GHOST-MODES``.

Cross-check only: the audited deliverable is never modified (the fresh re-run
happens in a throw-away copy under /tmp).  Every number below is re-derived
from a primary source: the audited work tree, a fresh execution of the
delivered entry point, the OSS artefact claimed as the external anchor, or the
platform DB.

Checks
------
S1 static : deliverable inventory + git identity (tracked vs untracked)
S2 static : claim "regeneration script" -> does extract_ludwig_reference.py exist?
S3 static : integration callers (is the module wired into the solver / platform?)
S4 static : unit contract - fresh pytest run vs the mission's reported outcome
S5 static : external anchor - does /tmp/ludwig_tar archive/object exist, sha256 match,
            are the certified symbols really exported, are the OSS *sources* present?
R1 runtime: fresh re-run of the delivered entry in a copy -> recorded-vs-fresh numbers
R2 runtime: linear-regime probe (kT -> 1e-9) of the ghost arm: the module's own
            analytic claim is Var(m)/3*rho*kT = 1 for injected modes; measured?
R3 runtime: A1 (stress-only, "known-good") arm vs the same FDT target
D1 db     : tensor_benchmark_cases rows for the anchor (the only referee)
D2 db     : mission row status / fail_stage / server quality gate
D3 db     : tensor_reports row for the mission (report discipline)

Usage::

    python scripts/audit/m17856_fluctuating_ghost_audit.py            # full
    python scripts/audit/m17856_fluctuating_ghost_audit.py --fast     # skip R1 re-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "artifacts" / "m17856_audit"
AUDITED = Path("/root/tp-agent/work/m17758")
DELIV = AUDITED / "xval" / "fluctuating_lbm_ghost_modes"
MODULE = AUDITED / "src" / "tensorlbm" / "fluctuating_ghost_modes.py"
TESTS = AUDITED / "tests" / "test_fluctuating_ghost_modes.py"
ARTIFACTS = DELIV / "artifacts"
ANCHOR = "XVAL-LBM-FLUCTUATING-LBM-GHOST-MODES"
LUDWIG_TAR = Path("/tmp/ludwig_tar")
LUDWIG_OBJ = LUDWIG_TAR / "ludwig-master" / "src" / "collision.o"
CLAIMED_OBJ_SHA = "ffc0b1a2426833b528f01603cca17730c9f3167b1235f300592867796fed252d"

findings: list[dict] = []


def finding(fid: str, severity: str, title: str, evidence: str) -> None:
    findings.append({"id": fid, "severity": severity, "title": title, "evidence": evidence})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def psql(sql: str) -> str:
    cmd = ["su", "postgres", "-c", f'psql -d tensorplatform -t -A -F"|" -c "{sql}"']
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str, cwd: Path = AUDITED) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True).stdout.strip()


def load_recorded() -> dict:
    return json.loads((ARTIFACTS / "results.json").read_text())


# ---------------------------------------------------------------------------
# static checks
# ---------------------------------------------------------------------------


def s1_inventory() -> dict:
    tracked = git("ls-files", "src/tensorlbm/fluctuating_ghost_modes.py",
                  "tests/test_fluctuating_ghost_modes.py", "xval/fluctuating_lbm_ghost_modes")
    status = git("status", "--short")
    # is the deliverable an ancestor of main / any branch?
    in_head = subprocess.run(
        ["git", "cat-file", "-e", "HEAD:src/tensorlbm/fluctuating_ghost_modes.py"],
        cwd=AUDITED, capture_output=True).returncode == 0
    rec = {
        "head": git("rev-parse", "--short", "HEAD"),
        "head_subject": git("log", "-1", "--format=%s"),
        "tracked_deliverable_paths": tracked.splitlines(),
        "git_status_short": status.splitlines(),
        "module_in_HEAD": in_head,
        "module_lines": len(MODULE.read_text().splitlines()),
        "entry_lines": len((DELIV / "run_xval.py").read_text().splitlines()),
        "tests_lines": len(TESTS.read_text().splitlines()),
        "n_commits_ahead_of_origin": git("rev-list", "--count", "origin/main..HEAD") or "n/a",
    }
    if not tracked:
        finding("F0", "high",
                "交付零 tracked 改动：实现只存在于工作树未跟踪文件，未入任何 commit/分支",
                f"m17758: git ls-files 对 src/tensorlbm/fluctuating_ghost_modes.py、tests/test_fluctuating_ghost_modes.py、xval/fluctuating_lbm_ghost_modes 全部零命中；git status --short={status.splitlines()}")
    return rec


def s2_regeneration_script() -> dict:
    refs = []
    for p in (MODULE, DELIV / "run_xval.py"):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if "extract_ludwig_reference" in line:
                refs.append(f"{p.relative_to(AUDITED)}:{i}")
    exists = (DELIV / "extract_ludwig_reference.py").exists()
    rec = {"claimed_script": "xval/fluctuating_lbm_ghost_modes/extract_ludwig_reference.py",
           "exists": exists,
           "module_docstring_refs": refs,
           "dir_listing": sorted(x.name for x in DELIV.iterdir())}
    if not exists:
        finding("F5", "medium",
                "模块文档声称的取证再生脚本 extract_ludwig_reference.py 不存在（取证逻辑实际内联在 run_xval.py）",
                f"m17758 模块 docstring 引用 {refs}；DELIV 目录仅 {rec['dir_listing']}")
    return rec


def s3_integration() -> dict:
    hits = subprocess.run(
        ["grep", "-rn", "fluctuating_ghost_modes", "--include=*.py", "--include=*.md",
         "-l", "."], cwd=AUDITED, capture_output=True, text=True).stdout.splitlines()
    hits = [h for h in hits if "__pycache__" not in h]
    callers = [h for h in hits if h not in (
        "./src/tensorlbm/fluctuating_ghost_modes.py",
        "./tests/test_fluctuating_ghost_modes.py",
        "./xval/fluctuating_lbm_ghost_modes/run_xval.py")]
    rec = {"grep_hits": hits, "integration_callers": callers}
    if not callers:
        finding("F6", "medium",
                "零集成：模块无任何主求解器/平台调用方（L3 未达成）",
                f"git grep 'fluctuating_ghost_modes' 全仓（排除自身+测试+xval）零命中；hits={hits}")
    return rec


def s4_tests() -> dict:
    env = dict(os.environ, PYTHONPATH=str(AUDITED / "src"))
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pytest", str(TESTS), "-q", "--no-header"],
                       cwd=AUDITED, capture_output=True, text=True, env=env)
    tail = (r.stdout or "").strip().splitlines()[-6:]
    failed = [ln.split("::")[-1].split(" ")[0] for ln in (r.stdout or "").splitlines()
              if ln.startswith("FAILED")]
    rec = {"cmd": "python -m pytest tests/test_fluctuating_ghost_modes.py -q",
           "returncode": r.returncode, "tail": tail, "failed_tests": failed,
           "wall_s": round(time.time() - t0, 2),
           "mission_claim": "6 passed, 1 failed（幽灵臂方差对比断言，因 A2 不稳定）"}
    if failed and not any("excites_ghost" in f for f in failed):
        finding("F2", "high",
                "测试契约实况与交付申报不符：失败的是 FDT 归一化断言 target_mode_variance(1.0)==3*rho*kT，非申报所称幽灵臂对比",
                f"本次重跑 {rec['tail'][-1] if rec['tail'] else ''}；failed={failed}；"
                f"mission.result.checks 申报 '6 passed, 1 failed（幽灵臂方差对比断言）'")
    return rec


def s5_external_anchor() -> dict:
    rec = {"archive_present": LUDWIG_TAR.exists(), "object_present": LUDWIG_OBJ.exists()}
    if LUDWIG_OBJ.exists():
        rec["object_sha256"] = sha256(LUDWIG_OBJ)
        rec["object_sha256_matches_claim"] = rec["object_sha256"] == CLAIMED_OBJ_SHA
        nm = subprocess.run(["nm", str(LUDWIG_OBJ)], capture_output=True, text=True).stdout
        want = ["lb_fluctuations_var_ghost", "lb_collision_ghost_modes_on",
                "lb_relaxation_time_ghosts"]
        rec["symbols_present"] = {w: (f" {w}" in nm or f"\n{w}" in nm or w in nm) for w in want}
        c_sources = sorted(p.name for p in (LUDWIG_TAR / "ludwig-master" / "src").glob("*.c"))
        py_sources = sorted(p.name for p in (LUDWIG_TAR / "ludwig-master" / "src").glob("*.py"))
        rec["c_sources_in_archive"] = c_sources[:10]
        rec["n_c_sources"] = len(c_sources)
        rec["n_py_sources"] = len(py_sources)
        rec["object_arch"] = subprocess.run(
            ["file", str(LUDWIG_OBJ)], capture_output=True, text=True).stdout.strip()
        if not c_sources:
            finding("F4", "high",
                    "外部锚只有编译产物、无源码：从未执行/比对 ludwig 的数值输出，XVAL「交叉对标」数值得不到任何 OSS 侧数字",
                    f"{LUDWIG_OBJ} 存在（sha256 匹配={rec['object_sha256_matches_claim']}），"
                    f"但同目录 *.c 源码数={len(c_sources)}；对标仅取到公式反推，无模拟输出级对标")
    else:
        finding("F4", "high", "外部锚工件缺失（/tmp/ludwig_tar 不存在或已清理），取证不可复现",
                str(LUDWIG_OBJ))
    return rec


# ---------------------------------------------------------------------------
# runtime checks
# ---------------------------------------------------------------------------


def r1_fresh_rerun() -> dict:
    """Copy the audited tree, re-run the delivered entry, compare to recorded numbers."""
    tmp = Path(tempfile.mkdtemp(prefix="m17856_rerun_"))
    try:
        for rel in ("src", "xval", "tests"):
            shutil.copytree(AUDITED / rel, tmp / rel,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        t0 = time.time()
        r = subprocess.run([sys.executable, "xval/fluctuating_lbm_ghost_modes/run_xval.py",
                            "--skip-extract", "--ludwig-obj", str(LUDWIG_OBJ)],
                           cwd=tmp, capture_output=True, text=True)
        wall = time.time() - t0
        fresh_path = tmp / "xval" / "fluctuating_lbm_ghost_modes" / "artifacts" / "results.json"
        rec = {"cmd": "python xval/fluctuating_lbm_ghost_modes/run_xval.py --skip-extract",
               "returncode": r.returncode, "wall_s": round(wall, 1),
               "stdout_tail": (r.stdout or "").strip().splitlines()[-10:],
               "stderr_tail": (r.stderr or "").strip().splitlines()[-5:],
               "fresh_artifact_written": fresh_path.exists()}
        if not fresh_path.exists():
            finding("F7", "blocking", "repro_cmd 不产出工件", str(rec["stdout_tail"]))
            return rec
        old, new = load_recorded(), json.loads(fresh_path.read_text())
        worst = 0.0
        per_arm = {}
        by_new = {a["arm"]: a for a in new["arms"]}
        for a in old["arms"]:
            b = by_new.get(a["arm"])
            if b is None:
                continue
            for mode, val in a["report"]["ratios"].items():
                nb = b["report"]["ratios"][mode]
                rel = abs(nb - val) / max(abs(val), 1e-30)
                worst = max(worst, rel)
                per_arm.setdefault(a["arm"], 0.0)
                per_arm[a["arm"]] = max(per_arm[a["arm"]], rel)
        shutil.copy2(fresh_path, OUT / "fresh_results.json")
        rec["fresh_results_copy"] = str((OUT / "fresh_results.json").relative_to(REPO_ROOT))
        rec["max_rel_diff_recorded_vs_fresh"] = worst
        rec["per_arm_max_rel_diff"] = {k: round(v, 12) for k, v in per_arm.items()}
        rec["reproducible_bitwise_ish"] = worst < 1e-9
        rec["n_arms"] = len(old["arms"])
        if not rec["reproducible_bitwise_ish"]:
            finding("F1", "high", "交付工件数值不能由入口原样复现",
                    f"max relative diff = {worst:.3e} over {len(old['arms'])} arms")
        return rec
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _probe(kT: float, thermostat: str, tau: float = 0.8, shape=(32, 32),
           n_samples: int = 1200, n_skip: int = 400, seed: int = 12345) -> dict:
    """Fresh in-memory arm run against the delivered module (no artifact writes)."""
    code = f"""
import json, sys
sys.path.insert(0, {str(AUDITED / 'src')!r})
from tensorlbm.fluctuating_ghost_modes import run_arm
r = run_arm("probe", thermostat={thermostat!r}, tau={tau}, kT={kT},
            shape={shape!r}, n_samples={n_samples}, n_skip={n_skip}, seed={seed})
d = r.as_dict()
print("PROBE_JSON" + json.dumps({{"ratios": d["report"]["ratios"], "rho": d["report"]["rho"],
      "variances": d["report"]["variances"]}}))
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("PROBE_JSON"):
            return json.loads(line[len("PROBE_JSON"):]) | {"kT": kT, "thermostat": thermostat,
                                                           "tau": tau}
    return {"error": (out.stdout + out.stderr)[-800:], "kT": kT, "thermostat": thermostat}


def r2_linear_regime_probe() -> dict:
    """Module's own analytic claim: injected-mode ratio -> 1.

    In the linear (small kT) regime the ghost-mode injection is decoupled from the
    hydrodynamics, so a ratio far from 1 cannot be blamed on a nonlinear blow-up.
    """
    hi = _probe(2.5e-4, "ghosts")
    lo = _probe(1e-9, "ghosts")
    rec = {"ghosts_kT2.5e-4": hi, "ghosts_kT1e-9": lo}
    for tag, r in (("kT=2.5e-4", hi), ("kT=1e-9", lo)):
        q = r.get("ratios", {}).get("qx")
        if q is None:
            finding("F3", "high", f"ghost 探针无法执行（{tag}）", str(r.get("error", ""))[:400])
        elif abs(q - 1.0) > 0.15:
            finding("F3", "high",
                    f"幽灵模热化未达 FDT 目标（{tag}）：Var(qx)/3rho kT={q:.3f}，偏 {100*(q-1):+.1f}%",
                    f"探针 run_arm(thermostat='ghosts', tau=0.8, {tag}) → ratios={ {k: round(v,4) for k,v in r['ratios'].items()} }"
                    + ("；该 kT 下流体自由度近乎静止，非线性爆炸不可作为解释" if tag.endswith("1e-9") else ""))
    return rec


def r3_stress_arm_vs_fdt() -> dict:
    """A1 is the mission's 'known-good' arm; check it against the same target."""
    r = _probe(2.5e-4, "stress")
    rec = {"stress_kT2.5e-4": r}
    q = r.get("ratios", {})
    if q:
        ratios = [v for k, v in q.items() if k not in ("rho", "jx", "jy")]
        mean = sum(ratios) / len(ratios)
        rec["nonconserved_mean_ratio"] = mean
        if abs(mean - 1.0) > 0.15:
            finding("F8", "high",
                    f"所谓「已知正确」的 stress-only 臂同样未达 FDT 目标：非守恒模平均比值 {mean:.4f}（偏 {100*(mean-1):+.1f}%）",
                    "说明该量规则不是可用裁判：A1 与 A2 都远离 1，无法据此判「transplant 有效/无效」，"
                    "更无法给出 <15% 的对标结论")
    return rec


def r4_formula_claim(rec_ref: dict) -> dict:
    """Executable check of the docstring's stationary-variance formula."""
    code = f"""
import sys
sys.path.insert(0, {str(AUDITED / 'src')!r})
import numpy as np
from tensorlbm.fluctuating_ghost_modes import (MODE_NAMES, FluctuatingLBMGhost,
                                               mode_weights, orthonormal_modes)
lbm = FluctuatingLBMGhost((4, 4), tau=0.8, kT=2.5e-4, seed=0, thermostat='ghosts')
W = mode_weights('D2Q9')
basis = orthonormal_modes('D2Q9')
print('W', [[n, round(float(W[i]),6)] for i, n in enumerate(MODE_NAMES)])
# analytic self-consistency: per-mode injected sd**2 vs target*(2tau-1)/tau**2
rho = np.ones(lbm.shape)
f = lbm.equilibrium(rho, np.zeros((2,)+lbm.shape))
g = lbm._add_mode_noise(f, rho)
dm = np.tensordot(basis, g - f, axes=([1],[0]))
for i, n in enumerate(MODE_NAMES):
    if n in lbm.noise_mode_names:
        print('MODE', n, 'measured_var', round(float(np.mean(dm[i]**2)),12),
              'analytic', round(3*2.5e-4*float(W[i])*(2*0.8-1)/0.8**2, 12))
print('SCALAR_API target_mode_variance(1.0)=', lbm.target_mode_variance(1.0),
      ' 3*rho*kT=', 3*2.5e-4)
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
    rec = {"stdout": lines, "stderr": (out.stderr or "")[-400:]}
    return rec


def r5_theory_referee() -> dict:
    """Independent referee that does NOT depend on the module's own mode weights.

    Kinetic equipartition for a D2Q9 isothermal lattice gas at rest:
    Var(u_x) = Var(u_y) = kT/rho and Var(rho) = rho**2 * kT.  These two numbers
    are the textbook statements the delivered module itself relies on (base
    ``FluctuatingLBM`` docstring: ``<u_a u_b> = (kT/rho) delta_ab``), so they can
    judge every arm without reusing its (possibly mis-normalised) target W_a.
    """
    code = f"""
import json, sys
import numpy as np
sys.path.insert(0, {str(AUDITED / 'src')!r})
from tensorlbm.fluctuating_ghost_modes import FluctuatingLBMGhost
from tensorlbm.fluctuating_lbm import FluctuatingLBM

def measure(cls, thermostat=None, kT=2.5e-4, tau=0.8, seed=12345, ns=1200, nskip=400):
    kw = dict(shape=(32, 32), tau=tau, kT=kT, seed=seed)
    lbm = cls(thermostat=thermostat, **kw) if thermostat else cls(**kw)
    ax = []; ay = []; ar = []
    for s in range(ns + nskip):
        lbm.step()
        if s >= nskip:
            rho, u = lbm._moments()
            ax.append(u[0]); ay.append(u[1]); ar.append(rho)
    ax = np.array(ax); ay = np.array(ay); ar = np.array(ar)
    rho_m = float(ar.mean())
    t_ux = kT / rho_m; t_rho = rho_m ** 2 * kT
    return dict(rho=rho_m, var_ux=float(ax.var()), var_uy=float(ay.var()),
                var_rho=float(ar.var()), target_ux=t_ux, target_rho=t_rho,
                err_ux=float(ax.var()) / t_ux - 1.0, err_rho=float(ar.var()) / t_rho - 1.0)

out = {{
    "A1_stress": measure(FluctuatingLBMGhost, "stress"),
    "A2_ghosts": measure(FluctuatingLBMGhost, "ghosts"),
    "A3_all": measure(FluctuatingLBMGhost, "all"),
    "base_FluctuatingLBM_kt_only": measure(FluctuatingLBM),
    "A2_ghosts_repeat": measure(FluctuatingLBMGhost, "ghosts"),
}}
out["deterministic_repeat_A2"] = (
    out["A2_ghosts"]["var_ux"] == out["A2_ghosts_repeat"]["var_ux"]
    and out["A2_ghosts"]["var_ux"] == out["A2_ghosts_repeat"]["var_ux"])
print("REFEREE" + json.dumps(out))
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    rec = {"targets": "Var(u_x)=Var(u_y)=kT/rho ; Var(rho)=rho**2*kT (rho=1, kT=2.5e-4 -> 2.5e-4 both)"}
    for ln in out.stdout.splitlines():
        if ln.startswith("REFEREE"):
            rec.update(json.loads(ln[len("REFEREE"):]))
            break
    else:
        rec["error"] = (out.stdout + out.stderr)[-600:]
        return rec
    bad = {a: (v.get("err_ux"), v.get("err_rho")) for a, v in rec.items()
           if isinstance(v, dict) and "err_ux" in v}
    for arm, (eux, erho) in bad.items():
        if arm.startswith("A2_ghosts_repeat"):
            continue
        worst = max(abs(eux), abs(erho))
        if worst > 0.15:
            finding("F12", "high" if arm.startswith("A2") else "medium",
                    f"独立理论裁判（U/rho 协方差，不依赖模块自己的 W_a）：{arm} 最差偏差 {100*worst:.1f}% > 15%",
                    f"{arm}: err(Var u_x)={100*eux:+.1f}%, err(Var rho)={100*erho:+.1f}% "
                    f"(目标 kT/rho=2.5e-4, rho^2 kT=2.5e-4)")
    return rec


# ---------------------------------------------------------------------------
# DB checks
# ---------------------------------------------------------------------------


def d1_benchmark() -> dict:
    rows = psql(f"SELECT case_name,metric,tolerance FROM tensor_benchmark_cases "
                f"WHERE anchor_slug='{ANCHOR}';")
    total = psql("SELECT count(*) FROM tensor_benchmark_cases;")
    rec = {"anchor": ANCHOR, "registered_rows": [r for r in rows.splitlines() if r],
           "n_registered_for_anchor": len([r for r in rows.splitlines() if r]),
           "tensor_benchmark_cases_total": total}
    if rec["n_registered_for_anchor"] == 0:
        finding("F9", "high",
                "基准裁判为空：tensor_benchmark_cases 无该锚点任何案例（无 case_name/metric/tolerance）",
                f"SELECT ... WHERE anchor_slug='{ANCHOR}' → 0 行；全表 {total} 行")
    return rec


def d2_mission() -> dict:
    raw = psql("SELECT status, fail_stage, budget_used, "
               "coalesce(result::json->>'accepted',''), "
               "coalesce(result::json->'server_quality_audit'->>'gate_valid','') "
               "FROM tensor_dev_missions WHERE id=17758;")
    parts = raw.split("|")
    rec = dict(zip(["status", "fail_stage", "budget_used", "accepted", "server_gate_valid"], parts))
    rec["raw"] = raw
    if rec.get("server_gate_valid") == "false":
        finding("F10", "medium",
                "平台自身质量门 gate_valid=false（key_present=false），succeeded 系 budget_exhausted 强制收尾",
                f"tensor_dev_missions(17758): fail_stage={rec.get('fail_stage')}, "
                f"server_quality_audit.gate_valid={rec.get('server_gate_valid')}")
    return rec


def d3_report() -> dict:
    n = psql("SELECT count(*) FROM tensor_reports WHERE mission_id='17758';")
    rec = {"n_reports": n}
    if n.strip() in ("0", ""):
        finding("F11", "medium", "m17758 未发布 tensor_reports 论文级报告（报告纪律未履）",
                "SELECT count(*) FROM tensor_reports WHERE mission_id='17758' → 0")
    return rec


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    res: dict = {"mission": "m17758", "anchor": ANCHOR, "audited_path": str(AUDITED)}
    res["S1_inventory"] = s1_inventory()
    res["S2_regeneration_script"] = s2_regeneration_script()
    res["S3_integration"] = s3_integration()
    res["S4_tests"] = s4_tests()
    res["S5_external_anchor"] = s5_external_anchor()
    res["D1_benchmark"] = d1_benchmark()
    res["D2_mission"] = d2_mission()
    res["D3_report"] = d3_report()
    if not args.fast:
        res["R1_fresh_rerun"] = r1_fresh_rerun()
    res["R2_linear_regime"] = r2_linear_regime_probe()
    res["R3_stress_arm"] = r3_stress_arm_vs_fdt()
    res["R4_formula"] = r4_formula_claim(res["S5_external_anchor"])
    res["R5_theory_referee"] = r5_theory_referee()
    res["findings"] = findings
    (OUT / "audit_m17856.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")

    print("=== m17856 audit summary ===")
    print("tracked deliverable:", res["S1_inventory"]["tracked_deliverable_paths"] or "NONE")
    print("tests:", res["S4_tests"]["tail"][-1] if res["S4_tests"]["tail"] else "n/a")
    print("registered benchmark cases:", res["D1_benchmark"]["n_registered_for_anchor"])
    if "R1_fresh_rerun" in res:
        print("fresh-vs-recorded max rel diff:", res["R1_fresh_rerun"].get("max_rel_diff_recorded_vs_fresh"))
    for r in ("ghosts_kT2.5e-4", "ghosts_kT1e-9"):
        q = res["R2_linear_regime"].get(r, {}).get("ratios", {})
        if q:
            print(r, {k: round(v, 4) for k, v in q.items()})
    print("stress arm nonconserved mean:",
          round(res["R3_stress_arm"].get("nonconserved_mean_ratio", float("nan")), 4))
    print("findings:")
    for f in findings:
        print(f" - [{f['severity']}] {f['id']}: {f['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())