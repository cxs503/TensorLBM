"""Register the round-2 method selection into ``oss_solutions``.

feature_slug = CASE-COUETTE-2D (case pool anchor).  Round-2 duty:
"调研成熟方法 + 结合推进" — the distilled transferable knowledge (what to
reuse, what was measured to be worse, which open-source structure to learn
from) goes here so later rounds/missions do not re-derive it.

Idempotent: deletes the previous row with the same feature_slug + tag before
insert.  Run:  python scripts/register_oss_solution_couette_2d.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os

import psycopg

FEATURE = "CASE-COUETTE-2D"
TAG = "moving-wall-halfway-bb-d2q9-bgk"

METHOD = (
    "二维平板 Couette 流(上壁 U0 平动、下壁静止、x 周期) u(y)=U0*y/H 逐点<2% "
    "(H=32, tau=0.8, U0=0.05, Re=16, Ma=0.0866): "
    "D2Q9 BGK + pre-streaming 半步反弹动壁(Ladd/Zou-He 动量修正 "
    "f_new[opp]=f_pre[i]+2w rho (c.u_wall)/cs^2, 壁面在 y=0.5 与 y=ny-1.5, 有效缝宽 H_eff=ny-2) "
    "+ 下壁静止退化为普通反弹 + 流向周期 + 2000 步窗上壁速度漂移<1e-5 判稳 + 末 200 步时间平均去噪。"
    "判分=数值剖面插值到同一格子坐标逐点比闭式线性解, 并额外报近壁行、壁面剪应力(离散斜率×nu)、"
    "动量积分(闭区间[0,H]含壁节点梯形)、质量漂移、两独立力路(Route B D3Q19 插值反弹 momentum-exchange "
    "与 control-surface 应力积分)一致性。"
)
FILES = (
    "benchmarks/verified/couette_2d/run.py; "
    "benchmarks/case_pool/couette_2d/run_case.py; "
    "benchmarks/case_pool/couette_2d/make_figures.py; "
    "benchmarks/case_pool/couette_2d/make_report.py; "
    "tests/test_case_pool_couette_2d.py; "
    "benchmarks/case_pool/couette_2d/README.md; "
    "benchmarks/case_pool/couette_2d/artifacts/{summary.json,routeA_*.json,routeB_*.json,fig1_couette_profile.png,fig2_couette_errors.png}"
)
CONVENTIONS = (
    "1) 半步反弹壁面在格子坐标 y=0.5 与 y=ny-1.5, 解析解必须用 y-0.5 与 H_eff=ny-2, "
    "否则近壁行有 ~0.5/H 系统偏差; "
    "2) 动量积分参考量必须闭区间[0,H]含壁节点(y=0->u=0, y=H->u=U0)梯形积分, "
    "只积流体行会漏两个壁面半格三角形(H=16 时低估 6.25%); "
    "3) 两路(不同缝高)壁面剪应力直接比是尺度错误, 须比无量纲系数 C=tau_w*H/(nu*U0)(解析值恒 1); "
    "4) 逐点相对误差在 u->0 处发散, 须按 u_ref>0.2*U0 掩膜(预注册)避免假 FAIL; "
    "5) 半程 BB 对线性剖面是精确贴合壁面处理(无几何台阶误差), 误差主体为 float32 时间平均噪声 "
    "+ Ma^2 压缩性残差, 故不呈幂律下降而落 1e-4%~1e-3% 平台——测收敛阶需 H>=64 与 float64; "
    "6) 低黏性(tau=0.6)与 MRT 在壁行累积使质量漂移超 1e-3% 护栏(仍<1e-2% 远小于容差), "
    "属守恒精度非解精度问题, 需壁行 rho 重标定; "
    "7) 本问题剖面形状与 Re 无关(任意 Re 成立), 但转捩 Re~360 以上进入湍流不在本算例范围。"
)
TRANSFER = (
    "fits: 任何 LBM 动壁/剪切型解析解验证(Stokes 1st/2nd problem、Womersley、Rayleigh-Benard 线性阶段、"
    "1D 导热 erfc) 都可复用同一套骨架: 半步动壁 BB + 壁面 0.5 偏移进解析解 + 闭区间含壁节点积分口径 "
    "+ 无量纲系数跨缝高比较 + u_ref 掩膜逐点判分 + 稳态门(漂移<1e-5)+ 末段时间平均。"
    "开源横向: pylbm 的 Bounce_back/LinearBounceBack 边界族与 Poiseuille 演示(pylbm/demo/2D/Poiseuille.py) "
    "是同一壁面处理谱系(可搬其边界类结构); Maarten-vd-Sande_lbm 是自由表面 LBM(不适用本无自由面问题)。"
    "不适用: 三维/湍流外推; 需要壁面亚格精确定位时应上 Bouzidi 插值/二次 BB(lbmpy/pylbm 谱系)。"
)
EVIDENCE = {
    "source_case_pool": "tensor_case_pool#COUETTE-2D",
    "oss_census": {
        "pylbm/pylbm": "Bounce_back / LinearBounceBack 4/(1+2d), 2/(1+d) 边界族 + Poiseuille 演示(同壁面处理谱系)",
        "Maarten-vd-Sande_lbm": "自由表面 LBM(update_types.py 界面质量再分配)——不适用本无自由面问题",
        "lbm-principles-practice/code": "moving wall half-way BB with Ladd momentum correction (MIT, Kruger textbook code)",
        "lssfau/lbmpy": "NoSlip/NoSlipLinearBouzidi/QuadraticBounceBack/FreeSlip boundary family (wall sub-grid placement)",
    },
    "literature": {
        "ZouHe1997": "Phys. Fluids 9, 1591-1598 (pressure/velocity BC for LBM BGK)",
        "Ladd1994": "J. Fluid Mech. 271, 285-339 (moving-wall bounce-back momentum correction)",
        "Bouzidi2001": "Phys. Fluids 13, 3452 (interpolated bounce-back wall placement)",
        "Kruger2017": "The Lattice Boltzmann Method (Springer) — Couette/poiseuille canonical validation",
        "case_pool_ref": "线性解析解 u(y)=U0*y/H (N-S 在 Couette 边界下直接求得)",
    },
    "measured": {
        "verdict": "pool_tolerance PASS (all <2% at reference point H=32,tau=0.8,U0=0.05); "
                   "guard/diagnostic 3 items honest-FAIL (mass drift at tau=0.6/MRT, convergence order -0.77)",
        "profile_l2_pct": 0.01471,
        "profile_max_pct": 0.02434,
        "wall_shear_pct": 0.001489,
        "top_row_u_pct": 0.001570,
        "momentum_integral_pct": 0.01476,
        "mass_drift_pct": 0.00046,
        "routeB_force_me_pct": 0.0000324,
        "routeB_force_cv_pct": 0.001418,
        "cross_route_C_consistency_pct": 0.01231,
        "convergence_order_16_32": -0.77,
        "artifacts": "benchmarks/case_pool/couette_2d/artifacts/summary.json",
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
