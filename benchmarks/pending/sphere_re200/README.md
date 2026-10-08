# 球 Re=200 阻力系数（sphere_re200）（pending）

**状态：⚠️ 未按战役标准验证(遗留档)——记录为库力法修复（PR #309 wet-node 估计器）之前的旧口径（Ladd standard 全固体 MEM），Cd +268.5% 是已知历史失效模式；案例从未按战役预注册协议用修复后估计器重跑**

## 物理问题

均匀自由流绕球 Re=200，观测量 = 阻力系数 Cd。参考（`result.json`）：Schiller-Naumann Cd_ref = 0.805615（精确值；任务书旧引 0.769 为算术错误，见 notes 字段；Clift-Gauvin 交叉值亦在 notes 字段记载）。

设置（`result.json`）：GeneralSimEngine PARAMETRIC_SPHERE，D3Q19、MRT、ForceMethod.MOMENTUM_EXCHANGE，extrap='none'，D=60（280×240×240 域），16000 步（monitor 缓慢下降至收敛）。

## 计算结果现状

| 力法 | Cd | 参考 | 误差 |
|------|----|------|------|
| MEM standard（全固体，Ladd 和形式） | 2.968987 | 0.805615 | **+268.5%** |

- 压力积分路径（extrap='none'）在同一配置读数系统性偏低（notes 字段记载）——两个旧口径一高一低，均不达标。
- verified=false；球绕流系列当时因力法缺口整体暂缓（notes 字段）。

## 不达标清单

（战役严格标准：共性模块入口 + 直接观测量对参考 ≤3% + ≥2 档网格单调收敛；修正/还原变换不算直接模拟）

1. **精度门 ✗**：err_pct_vs_sn = 268.5 ≫ 3%。
2. **网格阶梯缺失 ✗**：单档（D=60）记录，无 τ-matched/网格收敛阶梯。
3. **协议缺失 ✗**：无预注册（NOTES）、无参考锁、无稳态窗判据——2026-08 遗留档格式；且使用已被库修复取代的旧 MEM 估计器。

## 根因/诊断

已定源（同族案例 `sphere_re100` + `sphere_re100_mem` 的机器记录链）：**Ladd standard 全固体 MEM 在曲面球壁的失效模式**——和式在近壁链子集混合不同时间层分布，携带不抵消的自由流背景项（同族 Re=100 记录：收敛后 +264.6%）。该失效已由库 PR #309 修复（`momentum_exchange_wet_node` 遍历全部流体→固体穿越链；修复后估计器对独立控制体动量预算闭合 0.044%/0.154%，见 `sphere_re100_mem/result.json` cv_vs_ledger_rel_diff 字段）——本目录从未用修复后估计器重跑，**数值本身不反映当前库能力**。

## 晋级路径（转 verified 的条件）

1. **按 sphere_re100_mem（W5-B→W7-B→W8-A）的冻结协议在 Re=200 重制**：BFL 判定路线 + wet-node/链账本估计器 + 域形条款（up≥3D 入口距离——W7-B 已定源入口 Dirichlet 钳制为 Re=100 的 +11% 地板主因，且该伪影随 Re 增大而减）+ D∈{40,60} τ-matched 阶梯 + 参考锁（S-N 主锁 + Clift-Gauvin 交叉）。
2. 预算参考：同规模 Re=100 w8a T2 档（D60 大域、24000 步）显存峰值 26.672595977783203 GiB 可跑、600 步机器探针 wall_s=192.86548471450806 s；Re=200 τ 更低、步数量级相近，两档全阶梯约 10–20 GPU·h（估算，由每步成本外推）。
3. 达标后由归档 PR 将本目录 `pending/sphere_re200` → `verified/sphere_re200`（旧口径记录以披露条款随迁）。

<!-- PROVENANCE
[{"v": 2.968987, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re200/result.json", "k": "cd_total"},
 {"v": 0.805615, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re200/result.json", "k": "cd_ref_sn"},
 {"v": 268.5, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re200/result.json", "k": "err_pct_vs_sn"},
 {"v": 16000, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re200/result.json", "k": "steps"},
 {"v": 3.9804, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result.json", "k": "time_avg_last100.cd_mem_standard"},
 {"v": 264.6, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100/result.json", "k": "err_pct_vs_SN.cd_mem_standard"},
 {"v": 26.672595977783203, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w8a/mem_sparse_D60big.json", "k": "meta.mem_peak_loop_gib"},
 {"v": 24000, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/w8a/result.json", "k": "runs[1].steps"},
 {"v": 0.0004370084435827443, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/result.json", "k": "runs[2].cv_vs_ledger_rel_diff"},
 {"v": 0.0015388526780244283, "f": "/nfs/wangxi/worktrees/bm_pd/benchmarks/pending/sphere_re100_mem/result.json", "k": "runs[3].cv_vs_ledger_rel_diff"}]
-->
