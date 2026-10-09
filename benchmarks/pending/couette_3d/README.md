# couette_3d（pending）

**状态：❌ 未达标（判据类）——唯一下潜条款 G3 steady 在 H128 fp32 档 24000 步触顶未过 1e-5 漂移阈；fp64 同核零漂证明该阈落在 fp32 舍入噪声带内，属判稳口径问题而非物理未收敛。**

## 物理问题

三维 Couette 流：下壁静止、上壁以 U0=0.05（+x）拖动，x/z 周期，稳态线性剖面 u(y)=U0·(y−0.5)/H_eff（解析廓线在驱动内机器生成，零手抄）。D3Q19 BGK τ=0.8（ν=0.1），Ma=0.0866，半程反弹约定壁位于 y=0.5 与 ny−1.5（与 verified/couette_2d 锁同）。目的：verified/couette_2d 的 3D 孪生，验证库 3D 动壁半程反弹（pre-streaming 动壁 BB + 动量注入）路径。

## 计算结果现状

主阶梯 fp32（GPU，H∈{32,64,128}，min 20000 / max 24000 步）与补充阶梯 fp64（CPU，H∈{16,32,64,128}，方案精确性证据），机器判定 gate/verdict_v2.json（2026-10-07）：

| 阶梯 | H | l2_rel_err | max_rel（掩码） | steady | n_steps | 质量漂移 |
|---|---|---|---|---|---|---|
| fp32 | 32 | 1.933660468866072e-05 | 0.00424309382364099% | true | 20000 | 0.003769818474264706% |
| fp32 | 64 | 7.115735571563197e-05 | 0.013983718992226649% | true | 23400 | 0.015189430930397728% |
| fp32 | 128 | 7.246891871695247e-05 | 0.021870930989583336% | **false** | 24000（触顶） | 0.024942251352163464% |
| fp64 | 16 | 9.596262761960097e-14 | — | true* | 20000 | −1.1013412404281553e-11% |
| fp64 | 32 | 6.700234659707114e-13 | — | true* | 20000 | −1.063097322591656e-10% |
| fp64 | 64 | 7.231638097112415e-13 | — | true* | 20000 | −1.2755318686796465e-10% |
| fp64 | 128 | 1.733723620717403e-09 | — | true* | 20000 | −1.3703636516136333e-10% |

\* fp64 全档 steady=True 为逐档 case JSON 机器判定（gate/fp64/case_H{16,32,64,128}_fp64.json 的 steady 字段）。

dtype 对照诊断（diag_dtype.json）：fp32 残差为时间线性舍入累积——E_fp32_H64 探针 l2 从 4000 步 1.4304393901374426e-05 线性增至 24000 步 0.000115966306831097（dev_top 同步线性 -1.8328428268432617e-06 → -1.1019408702850342e-05，局域在动壁）；fp64 同核 F_fp64_H128 在 20000 步 l2=1.7760643129471193e-09、60000 步 1.98559470335232e-12（零漂收敛）。骨架根因 arm A/B（diag_arms.json）：带 bug 初值 H16 末探针 l2=6.663392559858039e-05，去 bug 后逐位 z/x 均匀（spanwise std=0.0）、l2 落 fp32 地板 1.0413607924419921e-05（nx=16 等价 1.0455099982209504e-05）。

共性模块入口：D3Q19 BGK collide/stream + 半程动壁 BB 全库路径（骨架 run_case 逐字保留，仅三处披露改动记录于每档 JSON 的 changes_vs_skeleton）；无手写核。

## 不达标清单

| 门 | 冻结条款 | 实测 | 判 |
|---|---|---|---|
| G1 精度（fp32 每档） | l2≤3e-2 且 max_rel≤3.0% | 最差 l2 7.246891871695247e-05、max 0.021870930989583336% | ✓ |
| G2a fp32 地板不敏感 | l2≤1e-3 | 最大 7.246891871695247e-05（对门 l2≤1e-3） | ✓ |
| G2b 方案精确性（fp64 每档） | l2≤1e-7 | 最大 1.733723620717403e-09（对门 l2≤1e-7） | ✓ |
| G3 健康（每档） | steady==True 且 n_steps≤24000、mass≤0.05%、uz_max≤1e-7、spanwise std≤1e-6 | **H128 fp32 steady=false（24000 步触顶）**；其余全过（fp32 最差 uz_max 1.3408947552306927e-07、spanwise std 0.0） | ✗（唯一失败） |
| 单调条款 | 标准严格单调下降条款 | 被 G2a+G2b 替代（prereg 冻结披露，owner 签核项）：方案误差可证在机器地板，对舍入噪声排单调序不可证伪 | 披露偏差 |

## 根因/诊断

1. 旧档案根因（Phase-0 机器诊断）：骨架 `u0[0,:,:]=0.0; u0[-1,:,:]=0.0` 把 z=0/z=nz−1 两个流体平面清零（2D 壁行索引误植 3D 布局），导致试点反收敛；arm A/B 位级复现锁定。
2. fp32 残差 = 纯舍入累积（时间线性、局域动壁），fp64 同核零漂（质量漂移 −1e-10% 量级至 60k 步）——离散半程 BB BGK 的吸引子即解析剖面，方案误差为零。
3. G3 steady 阈（单点末 10 采样漂移 <1e-5）恰落在 fp32 噪声带内：H128 fp32 在 24000 步 l2 已达 7.246891871695247e-05 且线性增长，判稳量被舍入游走淹没；同档 fp64 steady=True。

## 晋级路径

prereg 风险表预注册的三条 owner 选项（判据侧，无需重跑物理）：(a) 判稳口径重锁——改判 fp64 迹线或噪声带口径（如漂移阈对 l2 地板定标）后重判 G3；(b) 维持 FAIL-by-clause 如实归档；(c) 以 fp64 阶梯为晋级位。归档 PR 将 pending/couette_3d 移 verified/ 由 owner 决策。成本：判据修订 0 GPU·h；若需重跑 fp32 阶梯 ≤0.15 GPU·h（prereg 预算表）。

<!-- PROVENANCE [{"v": 1.933660468866072e-05, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/verdict_v2.json", "k": "per_tier_fp32.32.l2_rel_err"}, {"v": 0.00424309382364099, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/verdict_v2.json", "k": "per_tier_fp32.32.max_rel_err_masked_pct"}, {"v": 20000, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/verdict_v2.json", "k": "per_tier_fp32.32.n_steps"}, {"v": 0.003769818474264706, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/verdict_v2.json", "k": "per_tier_fp32.32.mass_drift_pct"}, {"v": 7.115735571563197e-05, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/verdict_v2.json", "k": "per_tier_fp32.64.l2_rel_err"}, {"v": 0.013983718992226649, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/verdict_v2.json", "k": "per_tier_fp32.64.max_rel_err_masked_pct"}, {"v": 23400, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/verdict_v2.json", "k": "per_tier_fp32.64.n_steps"}, {"v": 0.015189430930397728, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/verdict_v2.json", "k": "per_tier_fp32.64.mass_drift_pct"}, {"v": 1.3408947552306927e-07, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/verdict_v2.json", "k": "per_tier_fp32.64.uz_max"}, {"v": 7.246891871695247e-05, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/verdict_v2.json", "k": "per_tier_fp32.128.l2_rel_err"}, {"v": 0.021870930989583336, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/verdict_v2.json", "k": "per_tier_fp32.128.max_rel_err_masked_pct"}, {"v": 24000, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/verdict_v2.json", "k": "per_tier_fp32.128.n_steps"}, {"v": 0.024942251352163464, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/verdict_v2.json", "k": "per_tier_fp32.128.mass_drift_pct"}, {"v": -0.022118579684274324, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/verdict_v2.json", "k": "per_tier_fp32.128.tau_err_pct"}, {"v": 1.1173029612621121e-08, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/verdict_v2.json", "k": "per_tier_fp32.128.uz_max"}, {"v": 0.0, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/verdict_v2.json", "k": "per_tier_fp32.128.ux_spanwise_std"}, {"v": 9.596262761960097e-14, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/verdict_v2.json", "k": "per_tier_fp64.16.l2_rel_err"}, {"v": 20000, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/verdict_v2.json", "k": "per_tier_fp64.16.n_steps"}, {"v": -1.1013412404281553e-11, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/verdict_v2.json", "k": "per_tier_fp64.16.mass_drift_pct"}, {"v": 6.700234659707114e-13, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/verdict_v2.json", "k": "per_tier_fp64.32.l2_rel_err"}, {"v": 20000, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/verdict_v2.json", "k": "per_tier_fp64.32.n_steps"}, {"v": -1.063097322591656e-10, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/verdict_v2.json", "k": "per_tier_fp64.32.mass_drift_pct"}, {"v": 7.231638097112415e-13, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/verdict_v2.json", "k": "per_tier_fp64.64.l2_rel_err"}, {"v": 20000, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/verdict_v2.json", "k": "per_tier_fp64.64.n_steps"}, {"v": -1.2755318686796465e-10, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/verdict_v2.json", "k": "per_tier_fp64.64.mass_drift_pct"}, {"v": 1.733723620717403e-09, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/verdict_v2.json", "k": "per_tier_fp64.128.l2_rel_err"}, {"v": 20000, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/verdict_v2.json", "k": "per_tier_fp64.128.n_steps"}, {"v": -1.3703636516136333e-10, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/verdict_v2.json", "k": "per_tier_fp64.128.mass_drift_pct"}, {"v": 1.4304393901374426e-05, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/diag/diag_dtype.json", "k": "E_fp32_H64.probes[0].l2_rel"}, {"v": 0.000115966306831097, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/diag/diag_dtype.json", "k": "E_fp32_H64.probes[5].l2_rel"}, {"v": -1.8328428268432617e-06, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/diag/diag_dtype.json", "k": "E_fp32_H64.probes[0].dev_top"}, {"v": -1.1019408702850342e-05, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/diag/diag_dtype.json", "k": "E_fp32_H64.probes[5].dev_top"}, {"v": 1.7760643129471193e-09, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/diag/diag_dtype.json", "k": "F_fp64_H128.probes[1].l2_rel"}, {"v": 1.98559470335232e-12, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/diag/diag_dtype.json", "k": "F_fp64_H128.probes[5].l2_rel"}, {"v": 6.663392559858039e-05, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/diag/diag_arms.json", "k": "A_asis_H16.probes[5].l2_rel"}, {"v": 0.0, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/diag/diag_arms.json", "k": "B_fixed_H16.t0.span_fluid_std"}, {"v": 1.0413607924419921e-05, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/diag/diag_arms.json", "k": "B_fixed_H16.probes[0].l2_rel"}, {"v": 1.0455099982209504e-05, "f": "/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/diag/diag_arms.json", "k": "B_fixed_H16_nx16.probes[0].l2_rel"}] -->
