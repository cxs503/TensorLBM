# XH 混合引擎归档 README — benchmarks/archived/xh_hybrid_20261006（非 CLOSURE：无总判，只有已验证域声明）

## 定位（诚实域声明）
- XH-1 架构 = CG 动量骨架 × CAC 化学位势面力（双色两 τ 步进器，
  `engine/hybrid_cg_cac.py`，860 行）。
- **已验证域 = 低密度比（λ=ρ_r/ρ_b=2.0 族 PASS）**。ρ10 结构性不可行
  （静驻种群正性窗口为空、SL-3 跳闸），T6 判 N/A+FAIL，XH-1 降级
  "低密度比专用"后停批（NOTES.md XH-A5/A6）。
- 本目录 = 证据 + 复活种子，**不在包 import 路径上**：engine/ 为自含模块，
  复活时经显式路径导入，不得从 src/ 引用、不进 tensorlbm 包命名空间。

## Phase-1 机器门（json.load 机械提取，`path::key` 为数字来源）
- T1 Laplace — PASS（总判按 prereg 仅系于 (a)，XH-A4）：
  g_t1a_acceptance_le_3pct=true（route B，W*=4.0）、
  g_t1b_r_spread_le_1pct=false（W6 R 散布 2.10% 为真实物理，字面 FAIL 披露档）、
  g_t1c_linearity_2x=true、g_t4_ledger_le_1e-10=true
  `phase1/out/t1_laplace.json::gates`（v1 选臂 bug 原样件 = t1_laplace_v1_selectionbug.json）
- T2 驱动 Poiseuille — honest FAIL（SL-2 弃权，XH-A6 裁定 2）：
  g_t2_relL2_le_3pct_all=false、
  g_t2_monotone_both=true
  `phase1/out/t2_poiseuille.json::gates`
- R5 压力形式 — ρ2 绿、ρ10 结构性不可行：
  g_r5_rho2_contrast_retention_ge_0.9=true
  （smoke ρ2 C_retention=1.0024311581454124、3D 臂 1.0020186629185863）；
  g_r5_rho10_contrast_retention_ge_0.9=false；
  fix_attempt_count=3>2 → verdict_T6_infeasible_sl3_trip=true；
  verdict_T3_precondition_met_rho2=false（字面含越域 fp32 项；XH-A6 裁定 1 读法登记=物理前置成立）
  `phase1/out/r5_pressure_probe.json::gates, fix_attempt_count, verdict_*`
  ρ10 由平衡态本身播种：e2diag step-1 f_min_r=-4.555554030079033（解析 f_0=−4.5556 逐位）
  `phase1/out/e2diag.json::series.f_full[0].f_min_r`
- T3 静滴上升 — 总判 FAIL（λ=0.5 分判 FAIL / λ=2.0 PASS，诚实未调门）：
  g_t3_lam0.5_err_le_3pct=false
  （lam0.5_D64 err_vs_Ut=-0.023235421969777792、lam0.5_D96 err_vs_Ut=-0.043139409268072915）；
  g_t3_lam2.0_err_le_3pct=true
  （lam2.0_D64 err_vs_Ut=-0.008125045762985672、lam2.0_D96 err_vs_Ut=0.0062730995625461805）；
  g_t3_steady_le_2pct=false（唯一破者 lam0.5_D96 steady_change_in_window=0.021448089477768044）；
  g_t3_D_ladder_monotone=false、
  g_t4_ledger_le_1e-10=false（字面；T4 门定义域=fp64 臂，T3 fp32 膜=权重舍入地板，per-arm t4_fp32_attribution 随档）、
  g_all_stable=true
  `phase1/out/t3_hadroplet.json::gates, arms[*].err_vs_Ut, arms[*].steady_change_in_window`
- T3 归因臂（非门，attr_lam0.5_D64_aD16，a=D/16 减半）：err_vs_Ut=0.4960374436508499
  （a 减半误差剧增 → 离散/界面分辨率主导，非壁限域）`phase1/out/t3_hadroplet.json::arms`
- T6（ρ10 验收）— N/A+FAIL：机器依据 = r5_pressure_probe.json::verdict_T6_infeasible_sl3_trip
  （SL-3 3 次修复尝试超上限 2），prereg SL-3 后果条款执行；ρ10 验收臂未产出
  result JSON（判决不经臂档、径走 SL-3 条款；t6_density10.py 脚本随档备查）。

## Phase-0 探针（probe/out/*.json）
- P1 CG 力兼容 — 9/10 门 PASS，唯一 FAIL = β 耦合：
  g_p1c_spread_le_8pct=false
  （spread_R_beta=0.30400176181105465，30.4% honest FAIL）；
  σ 线性比=2.001391370833012、漂移=9.359487474232507e-17/步/色
  `probe/out/p1_cg_force_compat.json::gates, derived`
- P2 力移植 — 迁移门 FAIL：g_p2_ladder_w4_dev_le_0.04=false
  （abs_dev W4=0.4549630783725145 / W6=0.4781202836844932
  vs 3D 档案阶梯；2D 阶梯 s=0.01 W2/4/6/8 = 0.4251222388722442 / 0.46893280121703684 / 0.48662570365399227 / 0.4944881270692869）
  `probe/out/p2_force_portability.json::gates, derived`
- P3 密度对比度 — 塌缩复现、修正案路线杀死：
  g_m2_rescue_C_end_ge_1.5=false
  （m0 C_init=1.9977692495590362 → C_end=1.0247447658819375 塌缩复现；
  三臂 C_end 全档在 JSON）`probe/out/p3_density_contrast.json::gates, arms`
- P4 CAC Poiseuille 根因 — 壁机制病与黏度比无关：
  r0 Eu=0.07697692592908506（复刻归档 0.07702678080491326，|Δ|≈5e-5）；
  r1 Eu=0.07165150233604216 → discriminator_class="H_fw (force/wall machinery implicated)"；
  r0/r1 壁带 Eu=0.15636507793896656/0.1549892131895548（几乎相同）
  `probe/out/p4_cac_poiseuille_rootcause_r0.json, probe/out/p4_cac_poiseuille_rootcause_r1.json::arms[0]`

## Engine 冻结与 lint 披露
- `engine/hybrid_cg_cac.py` md5=02dbe4b1ec5e337f94e91f132c16d5e3（字节恒等复制，评估工件，永不重排），
  评估环境锚见 NOTES.md §2（worktree xh_hyb @ 4ba836c9…）。
- 该文件带 1 条 ruff 发现（repo config，刻意保留、按 ruff 原文逐字披露）：

```
I001 [*] Import block is un-sorted or un-formatted
  --> benchmarks/archived/xh_hybrid_20261006/engine/hybrid_cg_cac.py:50:1
   |
48 |   """
49 |
50 | / from __future__ import annotations
51 | |
52 | | import torch
53 | |
54 | | from .cac_lbm import (
55 | |     free_energy_params,
56 | |     iso_gradient_3d,
57 | |     iso_laplacian_3d,
58 | |     shifted_neighbors,
59 | | )
60 | | from .d2q9 import C as C_2D
61 | | from .d3q19 import C as C_3D
62 | | from .d3q19 import W_EXACT64
   | |____________________________^
63 |
64 |   CS2 = 1.0 / 3.0
   |
help: Organize imports
   |
60 | from .d2q9 import C as C_2D
61 + from .d3q19 import W_EXACT64
62 | from .d3q19 import C as C_3D
   - from .d3q19 import W_EXACT64
63 |
   |

Found 1 error.
[*] 1 fixable with the `--fix` option.
```

- 同理 `ruff format --check` 对该文件报 1 file would be reformatted——字节恒等
  理由同上，不修。其余 .py 经 lint 门入库（见 MANIFEST.md）。
