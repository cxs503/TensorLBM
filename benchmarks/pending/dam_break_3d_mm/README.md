# 3D 溃坝 dam_break_3d_mm（Martin & Moyce 溃坝柱前沿 X(T)）（pending）

**状态：❌ 未达标（引擎级缺陷五连，Wave-10 M3 根因链 2026-10-01 定源）——修正时间轴/参考表/卡尺三重测量错误后误差仍 +10.8~+27.9%（T=1），M3 正式阶梯 a40/a80 12/12 检查点非单调、反收敛（T*=2 误差 -2.25%→+7.53%），质量门双档皆败（-49.5%/-62.2%）；结构性核心 = 前锋推进与账本闭合是同一个开关，当前引擎无"既修账又保前锋"的臂。**

> 相邻散置文件 `../dam_break_recompute_mm.json`（含本案例与 dam_break_sc 的
> 口径重算行）**按指令保留原位、不移动**；其内容已并入下文表格与 PROVENANCE。

## 物理问题

Martin & Moyce (1952) 溃坝标准工况：矩形液柱（宽 a × 高 2a）在重力下坍塌，
无量纲前沿位置 X=x_front/a 对无量纲时间 T_MM=t·sqrt(2g/a)（M&M 约定，a=柱宽），
对比 Lethe 官方数字化表（T=1/2/2.96 → X=1.326/2.360/3.670）。模型 = 库
Körner 型自由表面模块 `tensorlbm.free_surface_lbm`（D3Q19 `free_surface_step`，
原版未修改），rho_gas=1.0（1:1 稳定化，非物理，已披露）。

## 计算结果现状

**表 1 — 卡尺修正复测（本目录 result.json `caliper_fix_remeasure`，2026-09-29，
g=1e-4，tau=0.8）**：修正了 H 高度取轴错误 + 参考表错误 + 检查点外推排除后：

| 档 | steps | T_max | X(T=1/2/3) | X err%（T=1/2/3） | H err%（T=1） | 质量漂移 | 界面胞膨胀 |
|----|-------|-------|-----------|-------------------|---------------|----------|------------|
| a=16 | 1200 | 3.0 | 2.0 / 3.0625 / 4.0625 | **81.82 / 70.14 / 50.46** | 52.24 | -28.54% | 17.2×（1311→22545） |
| a=32 | 566 | 1.0006 | 1.6875 | **53.41** | 56.25 | -1.36% | 6.7×（5183→34818） |

**表 2 — 双重错误修正后重算（`../dam_break_recompute_mm.json`，无重模拟，
从存档历史重算）**：旧 +50~82% 是双重错误（时间轴缺 sqrt(2) + 参考表错）的
乘积，**已撤回**；修正口径（T_MM 轴 + Lethe 表 + 前沿 toe 测量）后：

| 档 | T=1 | T=2 | T=2.96 | 判定 |
|----|------|------|--------|------|
| a=16 | **+27.93%** | +3.66% | -14.97% | FAIL |
| a=32 | **+10.77%** | —（T_max_mm=1.415 超出） | — | FAIL |

**表 3 — Wave-10 M3 正式阶梯（staging `root_cause_chain.json`，2026-10-01，
legacy 臂 = 库原样，n²=2 角柱）**：

| 档 | 质量漂移 | f 场库存漂移 | Z err%（T*=2） | max Z err% | H err%（τ=1/2/3） |
|----|----------|--------------|----------------|------------|-------------------|
| a=40 | **-49.46%** | +279.73% | -2.248 | 6.942（T*=1） | 23.962 / 100.0 / 246.861 |
| a=80 | **-62.19%** | +339.19% | **+7.529** | 20.678（T*=4.5） | 23.962 / 100.0 / 253.382 |

判决：12/12 检查点非单调、细化全面恶化（T*=2：-2.248→+7.529；T*=4.5：
+4.728→+20.678）——**缺陷随分辨率放大**。

## 不达标清单（严格标准：共性模块入口 + 直接观测量 ≤3% + ≥2 档网格单调收敛）

- 共性模块入口：✓（库 `free_surface_lbm.free_surface_step` 原版；本目录
  run.py 仅编排）。
- 直接观测量 ≤3%：✗（最佳重算口径 T=1 仍 +10.8%（a32）/+27.9%（a16）；
  H 检查点结构性不可达）。
- ≥2 档网格单调收敛：✗（反收敛：细化恶化所有 9 个 Z 检查点）。
- 质量守恒门：✗（正式阶梯 -49.5%/-62.2%；复测档 -28.5%）。

## 根因/诊断（M3 五连根因链，全部量化）

1. **重力双重施力**（库缺陷 #2 已登记）：速度平移 u_eq=u+tau·g（SC 式）叠加
   全 Guo 项 → g_eff=(2-0.5/tau)·g=1.5g（tau=1）；2D 静水柱内区拟合
   g_eff/g=1.522（legacy）→1.043（opt-in guo 方案）。3D 主线同样模式
   （free_surface_lbm.py ~634-655/~713-733，1eae1c3b 与 c3ac6216 皆在）。
   → 时间轴整体错标。
2. **无滑移边界层掐死前锋**：壁邻胞前锋推进被 no-slip 阻尼；slip 臂把 Z 误差带
   从 [-22%, +21.7%] 移到 [+2.4%, +16.5%]（maxZ +16.51 vs +21.73），中段误差变号。
3. **双向账本劈裂**：halo 晋升给新界面胞全额初始质量（造质量），账本又在
   气/界面边界漏记 → 两本账反向发散（a20 legacy：mass -39.12% vs 库存
   +212.74%；liq2iface 臂：-3.61% vs +6.42%）。
4. **LIQ→IFACE 回转缺失**：柱坍塌后库从不把表面邻接 LIQUID 胞转回 INTERFACE，
   高度示踪永不下降 → H 钉死在 1.0×（实测 1.025，界面胞取整）；H 检查点
   （+24/+100/+247%）结构性不可达 = Phase-1 止损触发。
5. **结构性核心：前锋推进与账本闭合是同一个开关**——前锋唯一引擎是
   halo 晋升"gas & 邻接 LIQUID → INTERFACE"+ _init_new 新质量（+296% 伪库存
   来源）；去掉 LIQUID 触发（关账）即杀前锋（liq2iface 臂账本近闭合
   -3.61%/+6.42% 但 maxZ +84.35%，前锋死、拓扑化石化 L/I/G=741/59/8004 恒定）。
   **当前引擎无任何臂能同时闭合账本并保住活前锋。**

附：A' 3D 静水柱探针（TL_FS_APRIME）改变缺陷符号但不闭合账本——
mass -11.36%（gate FAIL）、库存 -0.38%（s=800 交叉时刻假象，s=2400 为
-32.15%/-23.12% 平台不归零）；g=0 对照证明库存膨胀与重力无关（纯记账缺陷：
off 臂 +93.70%/+108.25%，on 臂 -0.19%/+7.03%）。M6 因此维持 PARKED。
rho_gas 物理化远未就绪（rg=0.5 时 mass -92~-98%）。

## 晋级路径

- **晋级引擎重设计**（M3 已登记未执行的提案）：前锋推进触发器从
  "LIQUID-邻居"改为"主动邻居/fill 阈值"，质量交换按 Körner 2006 权重
  （lettuce MIT / XLB Apache-2.0 可参照；FluidX3D 禁用未碰）——owner 候选轨。
- M3 已暂存的 opt-in 补丁（TL_FS_GRAVITY_SCHEME=guo、WALL_SLIP、LIQ_TO_IFACE、
  TL_FS_APRIME，默认位一致回归验证过）在 mp_w10 worktree 的 src_patched/，
  可作未来库 PR 原料；但根因 #5 表明参数级组合不可达标。
- 达标即归档 PR 移 `pending/dam_break_3d_mm` → `verified/`（需重设计引擎落地后
  重跑两档阶梯：X/H 双判据 ≤3% + 质量门 + 单调收敛）。

## 复现

```
本目录:  run.py + outputs_remeasure/{a16,a32}_g1e-04_rg1.0_result.json
重算:    ../dam_break_recompute_mm.json（scripts/recompute_mm_errors.py 产物）
M3 阶梯: /nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/
         root_cause_chain.json + result_a{40,80}_legacy.json + judgment_ladder_legacy.txt
```

<!-- PROVENANCE
[{"v": 1200, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_3d_mm/result.json", "k": "caliper_fix_remeasure.results.a16.steps"},
 {"v": 3.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_3d_mm/result.json", "k": "caliper_fix_remeasure.results.a16.T_max"},
 {"v": 2.0, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_3d_mm/result.json", "k": "caliper_fix_remeasure.results.a16.X.T1"},
 {"v": 3.0625, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_3d_mm/result.json", "k": "caliper_fix_remeasure.results.a16.X.T2"},
 {"v": 4.0625, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_3d_mm/result.json", "k": "caliper_fix_remeasure.results.a16.X.T3"},
 {"v": 81.82, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_3d_mm/result.json", "k": "caliper_fix_remeasure.results.a16.X_err_pct.T1"},
 {"v": 70.14, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_3d_mm/result.json", "k": "caliper_fix_remeasure.results.a16.X_err_pct.T2"},
 {"v": 50.46, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_3d_mm/result.json", "k": "caliper_fix_remeasure.results.a16.X_err_pct.T3"},
 {"v": 52.24, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_3d_mm/result.json", "k": "caliper_fix_remeasure.results.a16.H_err_pct_T1"},
 {"v": -0.2854, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_3d_mm/result.json", "k": "caliper_fix_remeasure.results.a16.mass_drift_rel"},
 {"v": 17.2, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_3d_mm/result.json", "k": "caliper_fix_remeasure.results.a16.iface.growth"},
 {"v": 566, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_3d_mm/result.json", "k": "caliper_fix_remeasure.results.a32.steps"},
 {"v": 1.0006, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_3d_mm/result.json", "k": "caliper_fix_remeasure.results.a32.T_max"},
 {"v": 1.6875, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_3d_mm/result.json", "k": "caliper_fix_remeasure.results.a32.X.T1"},
 {"v": 53.41, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_3d_mm/result.json", "k": "caliper_fix_remeasure.results.a32.X_err_pct_T1"},
 {"v": 56.25, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_3d_mm/result.json", "k": "caliper_fix_remeasure.results.a32.H_err_pct_T1"},
 {"v": -0.0136, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_3d_mm/result.json", "k": "caliper_fix_remeasure.results.a32.mass_drift_rel"},
 {"v": 6.7, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_3d_mm/result.json", "k": "caliper_fix_remeasure.results.a32.iface.growth"},
 {"v": 81.82, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_3d_mm/result.json", "k": "caliper_fix_remeasure.max_err_pct.a16"},
 {"v": 56.25, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_3d_mm/result.json", "k": "caliper_fix_remeasure.max_err_pct.a32"},
 {"v": 0.0001, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_3d_mm/result.json", "k": "caliper_fix_remeasure.config.g"},
 {"v": 4.242640687119286, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_recompute_mm.json", "k": "dam_break_3d_mm a=16 (g=1e-4).T_max_mm"},
 {"v": 27.932388875051593, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_recompute_mm.json", "k": "dam_break_3d_mm a=16 (g=1e-4).rows[0].newB_new.err_pct"},
 {"v": 3.660316800134939, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_recompute_mm.json", "k": "dam_break_3d_mm a=16 (g=1e-4).rows[1].newB_new.err_pct"},
 {"v": -14.96873173855279, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_recompute_mm.json", "k": "dam_break_3d_mm a=16 (g=1e-4).rows[2].newB_new.err_pct"},
 {"v": 10.765460030165906, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_recompute_mm.json", "k": "dam_break_3d_mm a=32 (g=1e-4).rows[0].newB_new.err_pct"},
 {"v": 1.415, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_recompute_mm.json", "k": "dam_break_3d_mm a=32 (g=1e-4).T_max_mm"},
 {"v": 81.8181818181818, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_recompute_mm.json", "k": "dam_break_3d_mm a=16 (g=1e-4).rows[0].old_old.err_pct"},
 {"v": -49.46, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "formal_fail_ladder.mass_drift_pct.a40"},
 {"v": -62.19, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "formal_fail_ladder.mass_drift_pct.a80"},
 {"v": 6.942, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "formal_fail_ladder.Z_err_pct.a40.T*=1"},
 {"v": -2.248, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "formal_fail_ladder.Z_err_pct.a40.T*=2"},
 {"v": 7.529, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "formal_fail_ladder.Z_err_pct.a80.T*=2"},
 {"v": 20.678, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "formal_fail_ladder.Z_err_pct.a80.T*=4.5"},
 {"v": 23.962, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "formal_fail_ladder.H_err_pct.a40.tau=1"},
 {"v": 100.0, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "formal_fail_ladder.H_err_pct.a40.tau=2"},
 {"v": 246.861, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "formal_fail_ladder.H_err_pct.a40.tau=3"},
 {"v": 253.382, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "formal_fail_ladder.H_err_pct.a80.tau=3"},
 {"v": -39.12, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "root_causes[2].arm_labeled_evidence_a20.legacy.mass_pct"},
 {"v": -3.61, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "root_causes[2].arm_labeled_evidence_a20.liq2iface_default_rg1.0.mass_pct"},
 {"v": -11.36, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "aprime_3d_probe.arms.aprime_on_g2e-4.mass_pct_at_800"},
 {"v": -0.38, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "aprime_3d_probe.arms.aprime_on_g2e-4.inv_pct_at_800"},
 {"v": 335, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "aprime_3d_probe.arms.aprime_on_g2e-4.iface[0]"},
 {"v": 2145, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "aprime_3d_probe.arms.aprime_off_g2e-4.iface[1]"},
 {"v": -0.19, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "aprime_3d_probe.arms.aprime_on_g0_control.mass_pct_at_800"},
 {"v": false, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/dam_break_3d_mm/result.json", "k": "verified"},
 {"v": 4.5, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "formal_fail_ladder.anti_convergence", "in_str": true},
 {"v": 4.728, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "formal_fail_ladder.anti_convergence", "in_str": true},
 {"v": 49.46, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "formal_fail_ladder.anti_convergence", "in_str": true},
 {"v": 62.19, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "formal_fail_ladder.anti_convergence", "in_str": true},
 {"v": 279.73, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "formal_fail_ladder.inventory_drift_pct.a40", "in_str": true},
 {"v": 339.19, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "formal_fail_ladder.inventory_drift_pct.a80", "in_str": true},
 {"v": 1.522, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "root_causes[0].measurement", "in_str": true},
 {"v": 1.043, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "root_causes[0].measurement", "in_str": true},
 {"v": 0.5, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "root_causes[0].measurement", "in_str": true},
 {"v": 2.4, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "root_causes[1].evidence", "in_str": true},
 {"v": 16.51, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "root_causes[1].evidence", "in_str": true},
 {"v": 21.73, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "root_causes[1].evidence", "in_str": true},
 {"v": 1.025, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "root_causes[3].evidence", "in_str": true},
 {"v": 212.74, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "root_causes[2].arm_labeled_evidence_a20.legacy.inventory_pct", "in_str": true},
 {"v": 6.42, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "root_causes[2].arm_labeled_evidence_a20.liq2iface_default_rg1.0.inventory_pct", "in_str": true},
 {"v": 84.35, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "root_causes[4].evidence_liq2iface", "in_str": true},
 {"v": 7.03, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "aprime_3d_probe.arms.aprime_on_g0_control.inv_pct_at_800", "in_str": true},
 {"v": 93.7, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "aprime_3d_probe.arms.aprime_off_g0_control.mass_pct_at_800", "in_str": true},
 {"v": 108.25, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "aprime_3d_probe.arms.aprime_off_g0_control.inv_pct_at_800", "in_str": true},
 {"v": -32.15, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "aprime_3d_probe.arms.aprime_on_g2e-4_s2400.mass_pct_at_2400"},
 {"v": -23.12, "f": "/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/root_cause_chain.json", "k": "aprime_3d_probe.arms.aprime_on_g2e-4_s2400.inv_pct_at_2400"}]
-->
