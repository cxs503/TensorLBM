"""Register the round-2 method selection into ``oss_solutions``.

feature_slug = CASE-LBM-STOKES-FIRST-PROBLEM (case pool anchor).  Round-2 duty:
"调研成熟方法 + 结合推进" — the distilled transferable knowledge (what to reuse,
what was measured to be worse) goes here so later rounds/missions do not re-derive it.

Idempotent: deletes the previous row with the same feature_slug + tag before insert.
Run:  python scripts/register_oss_solution.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os

import psycopg

FEATURE = "CASE-LBM-STOKES-FIRST-PROBLEM"
TAG = "moving-wall-halfway-bb-float64"

METHOD = (
    "Stokes 第一问题(脉冲起动平板) u/U=erfc(eta) 逐点<2%(eta=0..2, t*=nu*t/delta_ref^2, delta_ref=25): "
    "D2Q9 BGK tau=0.65(nu=0.05) + pre-streaming 半步反弹动壁(Ladd 动量修正 f_new[opp]=f_pre[i]-2w rho (c.u_w)/cs^2, 壁面在 y=0.5) "
    "+ 顶壁自由滑移(specular)远场 + 阶跃初值真时变模拟 + float64。判分=数值剖面插值到固定 eta 网格逐点比 erfc, "
    "并额外报全离散节点误差、eta=2 尾部单点误差、含 eta=0 钳位端点的全窗口误差。"
)
FILES = ("benchmarks/verified/stokes_first_problem/run_tstar1.py; "
         "benchmarks/verified/stokes_first_problem/make_assets.py; "
         "tests/test_stokes_first_problem_tstar1.py; "
         "docs/stokes_first_problem_tstar1_20261112.md; "
         "benchmarks/verified/stokes_first_problem/scan_tstar1/{summary.json,table.csv,*.png}")
CONVENTIONS = (
    "1) 半步反弹壁面在格子坐标 y=0.5, 解析解必须用 y-0.5, 否则 eta=2 处 ~4% 系统偏差; "
    "2) 必须 float64: float32 在 U<=1e-2 时尾部误差 24%~98%(舍入吞掉 1e-6 量级尾部速度, 质量漂移 1.2e-2%); "
    "3) eta<1/(4 delta) 无离散节点(最近节点 eta1=1/(4delta)), eta=0 端点偏差有闭式 1-erfc(eta1) (delta=25->1.128%), 属坐标伪误差; "
    "4) 域须 4*delta<<H(本例 100<<200) 才有半无限近似, 否则 10% 级截断误差(B24 实测 H=100,delta=30 -> 10.5%); "
    "5) 本题 Ma^2 压缩性误差结构性为零(rho 恒 1, 偏差 1e-12), 降 U 无收益, 精度只由分辨率/壁面/浮点决定; "
    "6) TRT(Lambda=3/16) 壁面节点更准(1.5e-7)但尾部差 47 倍(0.19% vs 0.004%) -> 取 BGK; "
    "7) t* 判据未给定义时须显式固定 delta_ref 并写进代码常量与测试。"
)
TRANSFER = (
    "fits: 任何 LBM 扩散/边界层型解析解验证(Stokes 2nd problem、Womersley、Rayleigh-Benard 线性阶段、1D 导热 erfc) 都可复用同一套骨架: "
    "半步动壁 BB + 壁面 0.5 偏移进解析解 + float64 + eta 网格逐点判分 + 含尾部单点的判据表; "
    "torch 张量极小网格时务必 set_num_threads(1)(实测多线程把 12.5k 步从 ~18s 拖到 54s/3k 步)。"
    "不适用: 三维/湍流外推; 需要壁面亚格精确定位时应上 Bouzidi 插值/二次 BB(lbmpy/pylbm 谱系)。"
)
EVIDENCE = {
    "source_case_pool": "tensor_case_pool#LBM-STOKES-FIRST-PROBLEM",
    "oss_census": {
        "lbm-principles-practice/code": "moving wall half-way BB with Ladd momentum correction (MIT, Kruger textbook code)",
        "lssfau/lbmpy": "NoSlip/NoSlipLinearBouzidi/QuadraticBounceBack/FreeSlip boundary family (wall sub-grid placement)",
        "pylbm/pylbm": "LinearBounceBack 4/(1+2d), 2/(1+d) variants",
        "lettucecfd/lettuce": "same torch stack: BB + TRT/MRT + analytic-solution regression gates",
    },
    "literature": {
        "Stokes1851": "Trans. Camb. Phil. Soc. 9, 8-106 (erfc similarity solution)",
        "Schlichting1979": "Boundary-Layer Theory §5 (Rayleigh / impulsively started plate, delta=sqrt(nu t))",
        "Ladd1994": "J. Fluid Mech. 271, 285 (moving-wall bounce-back momentum correction)",
        "Ginzburg2008": "Commun. Comput. Phys. 3(2), 427-478 (TRT magic parameter)",
        "Bouzidi2001": "Phys. Fluids 13, 3452 (interpolated bounce-back wall placement)",
        "Skordos1993": "Phys. Rev. E 48, 4823 (LBM initial condition for impulsive start)",
        "case_pool_ref": "Stokes 1851 / Schlichting 1979 §5 erfc, DOI(secondary) 10.1002/zamm.19800600419",
    },
    "measured": {
        "verdict": "VERIFIED (7/7 configs < 2%)",
        "main_pointwise_pct_eta0.1_2": 0.0781,
        "tail_eta2_pct": 0.0040,
        "full_window_pct_incl_eta0_clamp": 1.1282,
        "refined_delta50_pct": 0.0154,
        "convergence_order": 2.34,
        "float32_tail_pct": {"U=5e-2": 0.977, "U=1e-2": 24.39, "U=1e-3": 98.11},
        "rho_max_dev": 1.05e-12,
        "artifacts": "benchmarks/verified/stokes_first_problem/scan_tstar1/summary.json",
        "report_id": 15335,
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    print(json.dumps({"feature": FEATURE, "tag": TAG, "method": METHOD}, ensure_ascii=False, indent=1)[:1200])
    if a.dry_run:
        return
    db = os.environ.get("TP_PG_DB", "tensorplatform")
    with psycopg.connect(f"dbname={db} user=postgres", connect_timeout=8) as con:
        cur = con.cursor()
        cur.execute("delete from oss_solutions where feature_slug=%s and tag=%s", (FEATURE, TAG))
        cur.execute(
            """insert into oss_solutions (repo, feature_slug, tag, method, files, conventions, transfer, evidence)
               values (%s,%s,%s,%s,%s,%s,%s,%s) returning id""",
            ("TensorLBM", FEATURE, TAG, METHOD, FILES, CONVENTIONS, TRANSFER, json.dumps(EVIDENCE)),
        )
        rid = cur.fetchone()[0]
        con.commit()
        print("inserted oss_solutions id", rid)


if __name__ == "__main__":
    main()