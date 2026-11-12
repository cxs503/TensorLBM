# 交差审计 m16183 — CASE-LBM-SQUARE-CYL-NORBERG（方柱绕流 Re=100 涡脱 St / Cd）

审计单 m16205 · 判级 **L3** · 裁定 **reopened（request changes）**

## ① 案例介绍

二维自由流方柱绕流，Re = U·D/ν = 100，正置（α=0°）。物理目标量两项：平均阻力系数
Cd = Fx/(½ρU²D)、涡脱 Strouhal 数 St = f·D/U。求解设置（取自交付工件）：D2Q9 + MRT
碰撞，域 40D×40D（blockage 2.5%），方块中心距入口 10D，下游 10D sponge
（τ_eff = τ(1+ασ), α=10, power 2），入口 2 个 St_seed=0.14 周期、10% 幅值正弦播种，
u_in = 0.05，ν = u_in·D/Re，τ = 0.5+3ν，统计窗为后 50%。

参考口径（交付件自身的裁定，见 `evidence_audited/REFERENCE_AUDIT.md`）：**2D 自由流低阻塞数值簇**
Cd ∈ [1.44, 1.52]（中心 1.48），St ∈ [0.145, 0.149]（中心 0.147），
来源为 arxiv 2411.03124v4 / 2309.09197 / 2308.08085v3 / 2404.12123 / 2405.12834
的原文表格。交付件明确**排除**实验口径 Okajima 1982（St≈0.143–0.145，且原文无 Cd）
——理由是 2D 求解器不得对标 3D 实验。

**锚点名与交付口径的冲突（本单核心发现之一）**：锚点叫 `...-NORBERG`，而 Norberg 的
方柱/圆柱 St 系列是**实验**口径；交付件恰恰以「不得用实验对标 2D 求解器」为由放弃实验参考。
锚点名与交付参考体系不一致，需在锚点层裁决命名/口径（建议
`CASE-LBM-SQUARE-CYL-RE100-2D-LOWBLOCKAGE`）。

## ② 怎么做的（审计方法，非重做）

1. **取证**：读 m16183 的 mission 记录（`tensor_dev_missions.id=16183`）、其工作树
   `/root/tp-agent/work/m16183`、以及仓库内该算例目录
   `benchmarks/verified/square_cylinder/`；把被审工件复制进
   `runs/m16205_audit/evidence_audited/` 并留 sha256。
2. **裁判查询**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE
   anchor_slug='CASE-LBM-SQUARE-CYL-NORBERG'` → **0 行**（`tensor_benchmark_cases` 现存
   25 个 anchor，无一为该锚点）。锚点没有机器裁判行，容差判据无从机械裁定。
3. **门重放**：自写 `runs/m16205_audit/audit_recompute.py`（不 import torch/tensorlbm，
   只读 JSON）重算 6 类门：几何/存储误差自洽（双参考口径）/3% 验收/跨网格 span/
   测力列敏感性/裁判行存在性。结果见 ③。
4. **可执行性**：本机跑 `run.py --D 8 --steps 4000 --device cpu --compile-mode eager`
   冒烟，验证步进—测力—边界链路可执行（不用于精度判定）。
5. **工具链**：TensorLBM 主求解器（`src/tensorlbm/solver.py:276 collide_mrt`、
   `src/tensorlbm/boundary/boundaries.py:164 compute_obstacle_forces`、`:201
   far_field_bc_2d`、`:390 make_sponge_strength`）+ `benchmarks/compile_route.py`
   的 eager/编译路由；审计侧 python3 + psql，无 SDAA。

## ③ 计算结果（程序化注入，源自被审工件）

| 网格（测力列） | Cd got | Cd 误差 vs 1.48 | St got | St 误差 vs 0.147 | 3% 门 |
|---|---|---|---|---|---|
| D=32 (mem(compute_obstacle_forces)) | 1.4816 | +0.11% | 0.1482 | +0.82% | PASS |
| D=48 (mem(compute_obstacle_forces)) | 1.4844 | +0.30% | 0.1480 | +0.68% | PASS |
| D=64 (wet) | 1.6139 | +9.16% | 0.1479 | +1.58% | FAIL |
| D=64 (pair) | 1.4870 | +0.57% | 0.1479 | +1.58% | PASS |
| D=64 (mem) | 1.4870 | +0.57% | 0.1479 | +1.58% | PASS |

跨网格一致性（mem 列，D32/D48/D64）：
Cd = [1.4816, 1.4844, 1.487] → span 0.364%；St = [0.1482, 0.148, 0.1479] → span 0.203%。

门重放统计：**OK 23 项 / not-checkable 2 项 / FAIL 3 项**。

失败项：

- **D64/wet |Cd err| <= 3%** — +9.16%
- **wet - mem Cd < 3% (claim must not hinge on the column)** — +8.53% (wet column would give +9.05% error -> would FAIL the 3% bar)
- **tensor_benchmark_cases row for anchor** — rows=0 -> NO machine referee: got values cannot be machine-adjudicated

测力列 A/B（D=64）：wet Cd=1.6139，mem Cd=1.4870，
pair Cd=1.4870；wet−mem = **+8.53%**，
wet 列若作为主判据其 Cd 误差将达 +9.05%（超 3% 门）。

参考口径敏感性：同一 got 换用 `verify.py` 自身的参考常量（1.4785/0.1456）后，
D32 由 +0.11%/+0.82% 变为 +0.21%/+1.79%，D48 由 +0.30%/+0.68% 变为 +0.40%/+1.65%
——仍在 3% 内，但两个文件里并存两套参考常量（1.48/0.147 与 1.4785/0.1456），
同一份数据会给出两套误差数字，属可复现性缺陷。

**m16183 自身产出**：mission 记录 `status=succeeded`，但 `result.evidence.changed_files`
只有 `.scratch/m16183_checks.py`，`fail_stage=no_progress+forced_finish`，
headline 自述 honest-FAIL（预算耗尽于侦察，未进入实现与计算），`report_id` 为空
（tensor_reports 中无 mission 16183 的报告）。**该 mission 的交付增量 = 0**：
无实现、无算例运行、无基准对比、无报告、无测试。仓库里已有的
`benchmarks/verified/square_cylinder/` 是 2026-09-29（commit d9a45c85）前序工作的成果，
不能计入 m16183 名下。

## ④ 结果图

- `fig_ref_vs_got.png`：Cd、St 逐网格 got vs 参考簇带与 ±3% 门（数据源：被审工件 JSON）。
- `fig_force_columns.png`：D=64 三个测力列（wet/pair/mem）的 Cd 对比，暴露 8.53% 的
  测力列敏感性。

（本单为审计任务，图由被审工件派生，不宣称新仿真结果。）

## ⑤ 注意点 / 局限

1. **没有机器裁判**：`tensor_benchmark_cases` 无该锚点行 → 谁都无资格宣布「过基准」；
   本单只能重放工件内部自洽性，不能签发 L4。
2. **mission 交付为空而状态为 succeeded**：验收自动批准 + `no_progress+forced_finish`
   让「零产出」显示成「成功」，这是平台侧口径缺陷（交差单要能识别它）。
3. **仓内校验器跑不动**：`verify.py` 要求 `case_D32/48/64` 全部带 wet/pair/mem 三列，
   而实际 `case_D32.json`/`case_D48.json` 是 summary schema → `check("case files found", …)`
   报 `D=[]`，随后 `cases[ds[-1]]["pair"]` 抛 IndexError。所谓「verified」在现树上
   **无法被机械复核**。
4. **测力列敏感性 8.53%**：Cd 结论依赖 Ladd MEM 变体（wet vs pair/mem）。虽然 code
   文档把 wet 列为 diagnostic、pair/mem 闭合到 0.0%，但 8.5% 的量级说明该算例的
   Cd 不具备跨实现可迁移性，需在基准判据里锁定测力定义。
5. **全量重跑代价**：D32+D48 在 SDAA 上约 19 h（作者记录 146 s/553 s 为编译后步进的
   实际 elapsed，注明 2026-08-20 主机）；当前主机有 torch_sdaa 但 result.json 记录
   teco-inductor 在第 5 个后端缺陷处回退 eager。本单只做 D=8 冒烟，未做全量重跑。
6. **播种不锁频**：播种 St_seed=0.14、最终 St≈0.148，说明播种只触发不锁定；但这也意味着
   对称方柱要靠人工非对称扰动起振，属该算例的可复现性脆弱点。
7. 冒烟复现状态：未完成：本单审计机 D=8/4000 步 CPU 冒烟（`runs/m16205_audit/smoke_D8.log`）在审计窗口内未产出 result.json（99 控制面 load average ≈160，18 分钟后仍停在编译路由行）；补救路径=集群递交 cj1273（TensorLBM stage_dir=m16205, D=16/9000 步）排队未派发，已撤回。故本单**不提供自跑数值**，只做工件重放——这是范围声明，不是数值结论。。

## ⑥ 误差判定（显式回答是否 <3%）

- 被审工件主判据（mem/pair 测力列，D=32/48/64）：Cd 误差 +0.11%/+0.30%/+0.57%，
  St 误差 +0.82%/+0.68%/+1.58% → **是，全部 <3%**（最大 1.58%）。
- 但（a）wet 测力列 Cd 误差 **+9.16% → 不小于 3%**；（b）三个网格中 D=64 的判据常量为
  verify.py 的 1.4785/0.1456 而非 run.py 的 1.48/0.147，口径不统一；
  （c）m16183 本身**没有产生任何数值**，其「是否 <3%」无从判定（不是「通过」，是「无数据」）。
- 因此结论：**工件层面数值达标（<3%），锚点层面未验证（无裁判行、无 mission 交付）**。

## 判级与裁定

- **级别：L3**（系统集成级）。理由：真代码、输入→计算→输出完整（L2 满足），且经库
  solver/BC 集成（`run.py:88-90` 导入 `tensorlbm.boundaries`，`:163/:166` 步进函数调用
  `collide_mrt/stream/far_field_bc_2d/compute_obstacle_forces`，L3 满足）；交付件具备
  完整算例 + 误差分析 + repro_cmd（L4 的形态），但 **L4 不授予**，因为：无
  `tensor_benchmark_cases` 裁判行、仓内 verify.py 在现树不可运行、全量重跑未在审计中
  复现、测力列敏感性 8.53% 未被判据锁定。
- **裁定：reopened（request changes）**。返修清单（按优先级）：
  1. 注册 `tensor_benchmark_cases` 行：anchor=`CASE-LBM-SQUARE-CYL-NORBERG`（或改名），
     metric=`Cd/St vs 2D 自由流低阻塞数值簇`，tolerance=`<=3%（mem/pair 列，锁死测力定义）`；
  2. 修 `verify.py` 的 schema 兼容（summary schema 与 triple 列并存），使现树可机械复核；
  3. 统一参考常量（run.py 1.48/0.147 vs verify.py 1.4785/0.1456 二选一，写进基准行）；
  4. m16183 补报（tensor_reports）或由接棒 mission 补一次全量重跑并归档 Cl 时序；
  5. 在 `tests/` 增加该算例的回归测试（当前 `grep -rl square_cylinder tests/` 为空）。
- **归因**：L3 证据全部来自 commit d9a45c85（2026-09-29 前序工作）；m16183 交付增量 0。
