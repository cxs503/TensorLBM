#!/usr/bin/env python3
"""m17742 交差核验：m17673（XVAL-LBM-COMPILETIME-PARAMS-PLUS-RUNTIME-GEOMETRY）三件套机器复核。

零手抄：所有数字由本脚本现场执行产出，写入 runs/m17742/verify_triple.json。
核对对象（不重做）：
  1. run_artifacts / impl_files / entry / tests 是否真存在（sha256 钉死）；
  2. 交付 parity 数值是否可独立复现（重跑两次，逐字节比对交付 JSON）；
  3. repro_cmd 是否可执行（rc）；
  4. 基准裁判（tensor_benchmark_cases，唯一裁判）是否有该 anchor 的行；
  5. 物理层集群作业 cj1641 是否真的跑过（state/rc/finished_at）；
  6. 报告六段是否齐全；
  7. 交付物是否在 git 里（commit 状态 = 可交付性）。
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

M = Path("/root/tp-agent/work/m17673")          # 被审计单工作树
WT = Path("/root/tp-agent/work/m17742")         # 本审计工作树
ECO = Path("/root/tensor-ecosystem/TensorLBM")  # 生态主仓 clone
RUN = WT / "runs/m17742"
RUN.mkdir(parents=True, exist_ok=True)

ANCHOR = "XVAL-LBM-COMPILETIME-PARAMS-PLUS-RUNTIME-GEOMETRY"
CD_ANCHOR = 5.57953523384  # Nabh 1998 谱解，benchmarks/verified/cylinder/re20_st/result.json


def sh(cmd: str, cwd: Path | None = None, timeout: int = 900) -> dict:
    r = subprocess.run(cmd, shell=True, cwd=str(cwd) if cwd else None,
                       capture_output=True, text=True, timeout=timeout)
    return {"cmd": cmd, "rc": r.returncode,
            "stdout_tail": r.stdout[-2500:], "stderr_tail": r.stderr[-1500:]}


def sql(q: str) -> str:
    return sh(f'psql -d tensorplatform -U postgres -tA -F"|" -c "{q}"')["stdout_tail"].strip()


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def files(paths):
    out = {}
    for p in paths:
        f = Path(p)
        out[str(f)] = {"exists": f.is_file(), "bytes": f.stat().st_size if f.is_file() else 0,
                       "sha256": sha(f) if f.is_file() else None}
    return out


DELIVERED = {
    "impl_files": [str(M / "src/tensorlbm/runtime_geometry.py"),
                   str(M / "src/tensorlbm/cylinder_st2d1.py")],
    "entry": [str(M / "xval/life_geometry_config/xval_life_geometry_config.py")],
    "tests": [str(M / "tests/test_runtime_geometry.py")],
    "run_artifacts": [str(M / "runs/m17673/xval_parity.json"),
                      str(M / "runs/m17673/baseline_D20.log"),
                      str(M / "artifacts/xval_life_geometry_config/fig1_geometry_parity.png"),
                      str(M / "artifacts/xval_life_geometry_config/fig2_dx_sensitivity.png")],
    "docs": [str(M / "docs/reports/XVAL_LIFE_GEOMETRY_CONFIG.md"),
             str(M / "docs/reports/XVAL_LIFE_GEOMETRY_CONFIG_diffs.md")],
}

ev: dict = {"anchor": ANCHOR, "audited_mission": 17673, "audit_mission": 17742,
            "repo": "TensorLBM"}

# ── 1. 存在性 + sha256 ───────────────────────────────────────────────
ev["files"] = {k: files(v) for k, v in DELIVERED.items()}

# ── 2. 交付物 git 状态（可交付性）────────────────────────────────────
ev["git"] = {
    "m17673_head": sh("git rev-parse --short HEAD", M)["stdout_tail"].strip(),
    "m17673_status": sh("git status --porcelain", M)["stdout_tail"].strip().splitlines(),
    "m17673_commits_touching_deliverable": sh(
        "git log --oneline -3 -- src/tensorlbm/runtime_geometry.py xval/life_geometry_config", M
    )["stdout_tail"].strip(),
    "main_clone_head": sh("git rev-parse --short HEAD", ECO)["stdout_tail"].strip(),
    "solid_mask_in_main": sh("grep -rn 'solid_mask' src/tensorlbm/cylinder_st2d1.py | wc -l", ECO
                             )["stdout_tail"].strip(),
    "runtime_geometry_in_main": sh("ls src/tensorlbm/runtime_geometry.py 2>&1", ECO)["stdout_tail"].strip(),
    "worktree_of_m17673_is_dirty": True,
}

# ── 3. repro_cmd 实跑（两次，判确定性）+ 与交付 JSON 比对 ────────────
repro_cmd = ("cd /root/tp-agent/work/m17673 && PYTHONPATH=src python3 "
             "xval/life_geometry_config/xval_life_geometry_config.py --mode parity")
ev["repro"] = {"cmd": repro_cmd}
for tag in ("run1", "run2"):
    out = RUN / f"parity_{tag}.json"
    ev["repro"][tag] = sh(f"{repro_cmd} --out {out}")
    ev["repro"][tag]["out"] = str(out)
    ev["repro"][tag]["sha256"] = sha(out) if out.is_file() else None

delivered = json.loads((M / "runs/m17673/xval_parity.json").read_text())
r1 = json.loads((RUN / "parity_run1.json").read_text())
r2 = json.loads((RUN / "parity_run2.json").read_text())
ev["repro"]["byte_identical_run1_vs_run2"] = ev["repro"]["run1"]["sha256"] == ev["repro"]["run2"]["sha256"]
ev["repro"]["audit_equals_delivered"] = r1 == delivered

# 逐项对表（ref vs got，交付 vs 本单复跑）
rows = []
for d, a in zip(delivered["rows"], r1["rows"]):
    rows.append({
        "grid": d["grid"], "dx_m": d["dx_m"], "nx": d["nx"], "ny": d["ny"],
        "mask_cells_ref": d["mask_cells_ref"], "mask_cells_got": d["mask_cells_got"],
        "mask_delta_cells": d["mask_delta_cells"], "audit_mask_delta_cells": a["mask_delta_cells"],
        "markers_ref_formula": d["markers_ref_formula"], "markers_got": d["markers_got"],
        "audit_markers_got": a["markers_got"],
        "perimeter_err_pct": round(d["perimeter_err_pct"], 6),
        "audit_perimeter_err_pct": round(a["perimeter_err_pct"], 6),
        "match": (d == a),
    })
ev["parity_table"] = rows
ev["parity_thresholds"] = delivered["thresholds"]
ev["mechanism"] = delivered["mechanism"]
ev["sensitivities"] = delivered["sensitivities"]

# 二阶收敛比（报告 ⑤ 的自声明是否成立）
e = [abs(r["perimeter_err_pct"]) for r in rows]
ev["second_order_ratio"] = [round(e[i] / e[i + 1], 3) for i in range(len(e) - 1)]

# 标记公式独立核算 floor(2*pi*r/dx)，r=0.05 m
import math
ev["marker_formula_recompute"] = {
    r["grid"]: {"floor_2pi_r_over_dx": int(math.floor(2 * math.pi * 0.05 / r["dx_m"])),
                "delivered": r["markers_got"]} for r in rows}

# ── 4. 基准裁判：tensor_benchmark_cases ──────────────────────────────
ev["benchmark_judge"] = {
    "rows_for_anchor": sql(f"SELECT case_name,metric,tolerance,repo,status FROM tensor_benchmark_cases "
                           f"WHERE anchor_slug='{ANCHOR}'"),
    "tensorlbm_rows_total": sql("SELECT count(*) FROM tensor_benchmark_cases WHERE repo='TensorLBM'"),
    "tensorlbm_rows": sql("SELECT id,anchor_slug,status FROM tensor_benchmark_cases "
                          "WHERE repo='TensorLBM' ORDER BY id"),
}

# ── 5. 物理层：集群作业 cj1641 ───────────────────────────────────────
ev["cluster_job"] = {
    "id": "cj1641",
    "row": sql("SELECT id,state,coalesce(rc::text,''),coalesce(started_at::text,''),"
               "coalesce(finished_at::text,''),left(cmd,240) FROM cluster_jobs WHERE id=1641"),
}
log = (M / "runs/m17673/baseline_D20.log")
last = [l for l in log.read_text().splitlines() if l.strip()][-1] if log.is_file() else ""
ev["cluster_job"]["delivered_log_last_line"] = last
m = re.search(r"(\d+)/(\d+)", last)
cd = re.search(r"Cd~([0-9.]+)", last)
ev["physics_midrun"] = {
    "steps_done": int(m.group(1)) if m else None,
    "steps_target": int(m.group(2)) if m else None,
    "cd_midrun": float(cd.group(1)) if cd else None,
    "cd_anchor": CD_ANCHOR,
    "dev_pct_vs_anchor": round((float(cd.group(1)) / CD_ANCHOR - 1) * 100, 2) if cd else None,
    "verdict": "NOT_RUN / NOT_CONVERGED — 半程未收敛值，既不可判 pass 也不可判 fail，不计入任何级别",
}

# ── 6. 报告六段齐全性 ────────────────────────────────────────────────
md = (M / "docs/reports/XVAL_LIFE_GEOMETRY_CONFIG.md").read_text()
secs = {"①案例介绍": "①", "②方法": "②", "③计算结果": "③", "④结果图/对照表": "④",
        "⑤注意点/敏感性": "⑤", "⑥误差判定": "⑥"}
ev["report_sections"] = {k: (f"## {v}" in md) for k, v in secs.items()}
ev["report_withheld_declared"] = "WITHHELD" in md
ev["report_figures_claimed"] = re.findall(r"fig\d_[a-z_]+\.png", md)

# ── 7. 测试复跑 ──────────────────────────────────────────────────────
ev["pytest"] = sh("cd /root/tp-agent/work/m17673 && python3 -m pytest tests/test_runtime_geometry.py -q",
                  timeout=900)
ev["pytest"]["n_tests_defined"] = len(re.findall(r"^def test", sha and
    (M / "tests/test_runtime_geometry.py").read_text(), re.M))

# ── 8. 外部锚（LIFE 源码）可复核性 ──────────────────────────────────
life = Path("/tmp/LIFE-master")
ev["external_anchor"] = {
    "path": str(life), "present": life.is_dir(),
    "durable": False,
    "note": "/tmp 非持久、不在任何仓内、无 commit/sha256 钉扎 → 一旦清理则 file:line 指针不可复核",
    "spot_checks": {
        "IBMBody.cpp floor(2pi r/dx)": sh("grep -n 'floor(2.0 \\* M_PI \\* radius' src/IBMBody.cpp | head -2", life)["stdout_tail"].strip(),
        "IBMBody.cpp deg->rad": sh("grep -n 'M_PI / 180.0' src/IBMBody.cpp | head -2", life)["stdout_tail"].strip(),
        "Utils.h rotation matrix": sh("grep -n 'getRotationMatrix' -A3 inc/Utils.h | head -8", life)["stdout_tail"].strip(),
        "Objects.cpp two-pass reserve": sh("sed -n '375,386p' src/Objects.cpp", life)["stdout_tail"].strip(),
    } if life.is_dir() else {},
}

# ── 9. mask 对照是否「构造性恒等」────────────────────────────────────
ev["parity_is_constructive"] = {
    "runtime_rasterise": "(xx-xc)**2+(yy-yc)**2 <= r*r  (physical->lattice via x_lat=(x_phys-origin)/dx)",
    "in_repo_cylinder_mask": "src/tensorlbm/boundary/boundaries.py:9-25 — (xx-cx)**2+(yy-cy)**2 <= radius**2",
    "conclusion": ("ref 用的就是本仓固有 cylinder_mask，got 是同一闭式解从物理坐标重推 → Δ=0 属精确恒等，"
                   "证明的是「插管不改变几何」而非独立数值验证；不得据此判 L4"),
}

out = RUN / "verify_triple.json"
out.write_text(json.dumps(ev, ensure_ascii=False, indent=1))
print(json.dumps({"wrote": str(out),
                  "git": ev["git"]["m17673_status"],
                  "solid_mask_in_main": ev["git"]["solid_mask_in_main"],
                  "parity_match": ev["repro"]["audit_equals_delivered"],
                  "bench_rows": ev["benchmark_judge"]["rows_for_anchor"],
                  "cj1641": ev["cluster_job"]["row"],
                  "pytest_rc": ev["pytest"]["rc"],
                  "report_sections": ev["report_sections"]}, ensure_ascii=False, indent=1))