# m18162 交差核验报告 — m18071 `XVAL-LBM-ONE-FILE-PER-CASE-PURE-STEP`

- 审计对象：mission 18071（TensorLBM，`succeeded`，auto-approve）
- 审计单：m18162（本单，verdict_by=`board-audit`）
- 审计工作树：`/root/tp-agent/work/m18162`；被审工作树：`/root/tp-agent/work/m18071`
- 审计时刻：2026-11-13 12:0x UTC
- 结论：**级别 L2**（真实代码 + 可执行入口 + 非空运行工件；**无测试、无集成调用方、基准 FAIL**），
  报告与工件一致地显示「目标判据 <15% 未达成」，m18071 交付的算例在目标工况返回 **NaN**。

> 纪律声明：本单为**核验单，非重做**。未修改 m18071 工作树内的任何文件（只读导入其模块 + 复制快照）。
> 所有数字均来自程序化产物，路径见文末「证据指针」。

---

## ① 案例介绍（物理背景 / 几何 / 参数）

对标对象：`haimingz/vivsim`（commit `fb84b12`）`examples/2d/flow_pass_cylinder.py` — 二维自由流圆柱绕流，
Re=200，`one-file-per-case-pure-step` 写法（每个算例一个自包含文件，`collide-stream-BC` 单步纯函数式循环）。
参考量由 OSS 源码自带：`CD_REF=1.389`、`CL_REF=0.718`（源码 `:78-85`，由三篇 DOI 平均：
`10.1016/j.ijmecsci.2024.108961`、`10.1016/j.camwa.2014.05.013`、`10.1017/S0022112086003014`）。
Strouhal 二次锚（非判据）：`ST_LIT=0.197`（Williamson & Brown 1998）。

几何与工况（vivsim 与 m18071 交付脚本，取自工件 `setup` 段，零手抄）：

| 参数 | vivsim 参考 | m18071 交付（目标 run） | 备注 |
|---|---|---|---|
| Re | 200 | 200 | 一致 |
| D（格子直径） | 60 | 8 | 交付件用更小 D |
| NX/D × NY/D | 30 × 20 | 30 × 20 | 一致 |
| U0 | 0.05 | 0.05 | 一致 |
| τ | 0.5075 | 0.506 | ν=(τ−0.5)/3 |
| 碰撞 | KBC + EDM | MRT(s_e=1.64, s_eps=1.54, s_q=1.7) | **不同** |
| 边界 | NEBB 入口 / 平衡态出口 | `far_field_bc_2d` | **不同（且实现有缺陷，见 F1）** |
| 受力 | IB multi-direct | 动量交换（Ladd） | **不同（且实现有缺陷，见 F2）** |
| 步数 | 150 D/U0 | 4800 | — |

判据（唯一裁判）：误差 **<15%**（m18071 goal 文本 + 脚本常量 `TOL_PCT=15.0`，`:72`）。
**关键发现：`tensor_benchmark_cases` 中该 anchor 无登记行**（全表 91 行 / 74 anchors，`anchor_slug='XVAL-LBM-ONE-FILE-PER-CASE-PURE-STEP'` → 0 行），
即任务榜所说的「基准裁判」对该 anchor 缺位，判据只能回落到 mission goal 与脚本常量（F4）。

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **取证**：读 `tensor_dev_missions` 18071 的 `result`、`tensor_reports`、工作树 `/root/tp-agent/work/m18071`；
   把交付文件复制成快照并算 sha256（`artifacts/board_audit_m18162/snapshot_manifest.json`）。
2. **三件套核验**：run_artifacts / benchmark(ref vs got) / repro_cmd 逐项实跑：
   - `repro_cmd`（脚本 docstring 给的形式）实跑两档：
     - 目标档 `--D 8 --u 0.05 --re 200 --steps 4800 --threads 8` → NaN @ step 1000，23.1 s，exit 0；
     - 小档 `--D 4 --u 0.05 --re 100 --steps 200/600` → 200 步 Cd=−0.0283（vs 1.389，**−101%**），600 步 NaN @ 500。
   - `run_artifacts`：`runs/xval_quick_test.json`（`nan_at=1000`，`verdict.pass=false`）与审计复跑一致 → **FAIL 可复现**。
3. **缺陷定位（独立复现，防御性）**：写 `artifacts/board_audit_m18162/defect_probe.py` 只读导入 m18071 的
   `boundary/cylinder.py`，用决定性的探针证明两处结构性错误（不依赖「报告怎么说」）：
   - D1：`f` 形状为 `(9, ny, nx)`；`far_field_bc_2d` 用 `f[:, 0, :]`/`f[3, 0, :]`/`f[:, -1, :] = f[:, -2, :]`
     写的是 **y=0 行与 y=ny−1 行**，不是 x=0 入口列与 x=nx−1 出口列 → **入口从未加过**，出口零梯度覆盖的是底边。
     探针：毒化 x=0 整列与 x=nx−1 整列后调用 BC，两列内部**逐位不变**（`D1_inlet_column_x0_untouched=true`、
     `D1_outlet_column_x_last_untouched=true`），而 y=0 行被改写。
   - D2：`compute_obstacle_forces` 的链路掩码是 `surface & neighbour`（两端都是固体），Ladd 动量交换应取
     **固-液链路**（固节点 → 相邻流体节点）并带 2 倍因子；另外 `torch.roll` 会跨周期边界卷绕。
     探针：把紧邻壁面的一个流体节点的分布函数乘 3，`fx` 变化 **0.0**（`D2_force_insensitive_to_wall_fluid=true`）
     → 壁面流体动量交换从未进入受力，Cd 不可能对。
4. **判级**：按 maturity_gate 的达标判断方案逐键给证据（`judge.json`），并对 `maturity_gate.judge()` 做预演。
5. 工具链：`psql -U postgres -d tensorplatform`（只读查询）、torch/numpy 实跑、matplotlib 出图。

## ③ 计算结果（数字表）

| 量 | 参考值 | m18071 工件 | 审计复跑 | 误差 | 15% 判据 |
|---|---|---|---|---|---|
| Cd_mean（目标档 D=8, Re=200, 4800 步） | 1.389 | **NaN** | **NaN**（nan@1000） | 不可定义 | **FAIL** |
| \|Cl\|_peak（同上） | 0.718 | **NaN** | **NaN** | 不可定义 | **FAIL** |
| Cd_half1（同上） | — | 0.0 | 0.0 | 物理上不可能（无阻力） | 信号层 FAIL |
| Cd（小档 D=4, Re=100, 200 步） | 1.389 | 未记录 | **−0.0283** | **−101.04%** | **FAIL** |
| \|Cl\|_peak（小档同上） | 0.718 | 未记录 | **0.0748** | **−89.58%** | **FAIL** |
| St（小档，二次锚） | 0.197 | 0.32（目标档） | NaN / 不稳 | 62.4%（工件自算） | 不判据 |

稳定性扫描（审计实跑，同一交付脚本）：

| 配置 | 网格 | 网格量 | 结果 |
|---|---|---|---|
| D=8, Re=200, 4800 步 | 240×160 | 38 400 | **NaN @ step 1000** |
| D=4, Re=100, 600 步 | 120×80 | 9 600 | **NaN @ step 500** |
| D=4, Re=100, 200 步 | 120×80 | 9 600 | 有限但非物理（Cd<0） |

> 交付文档称「Stable regime verified: D=4, Re=100, 60×40, BGK, 300+ steps」；审计在**同一档 D=4/Re=100**
> 复跑到 500 步即 NaN，且 200 步的有限值本身是负阻力 → 该「稳定」不构成可用输出（F7）。

## ④ 结果图

图见本单 tensor report（`figures`）与 `artifacts/board_audit_m18162/figs/`：

1. `fig1_ref_vs_got.png`：参考 vs 交付的两档 got（200 步有限值 / 目标档 NaN），带 ±15% 带 → 全部越界。
2. `fig2_stability.png`：两档步数-稳定性（NaN 发生步 vs 总步），显示 500/1000 步的硬墙。
3. `fig3_defect_probe.png`：D1 探针（毒化入口/出口列后 BC 调用前后不变）+ D2 探针（毒化壁面相邻流体节点后 fx 变化为 0）。

（无「场量云图」：交付件未输出任何稳定场解，只能输出 NaN 前的瞬时分布，画出来会误导读者；见 ⑤ 局限。）

## ⑤ 注意点（坑 / 局限 / 适用范围）

- **坑 1（维序约定）**：TensorLBM 的 D2Q9 分布是 `(9, ny, nx)`；任何 `f[:, i, :]` 都是「一行」。
  m18071 的 BC 把「行」当「列」用 —— 这类错误在纯文档审阅里看不出来，跑一次探针立刻现形。
- **坑 2（动量交换链路集）**：`surface & neighbour` 是固-固链路；MEM 必须取固-液链路并带 Ladd 的 2 倍因子；
  `torch.roll` 的周期卷绕会让「贴边」也算成交叉链路。
- **坑 3（auto-approve 盲区）**：m18071 的 `result.evidence.server_quality_audit` 明确写着
  `gate_valid=false`、`gate_missing_warning=true`、`head_commit=null`，但 `accepted=true` —— 交付「成功」并不代表判据达成。
- **局限**：本次审计只读不改，未替 m18071 修缺陷（F1/F2 的修复属于返修单的范围）；
  未做网格无关性/收敛性研究（在 NaN 的算例上无意义）；未复核 vivsim 参考值本身的 DOI 溯源（沿用其源码声明）。
- **适用范围**：等级判定只覆盖「功能实质」，不覆盖物理正确性；物理正确性由 L4 基准承担，本例 L4 FAIL。

## ⑥ 误差判定（逐项，显式回答是否 <3%）

| 关键量 | vs 基准 | 误差 | <3%？ | <15%？ |
|---|---|---|---|---|
| Cd_mean（D=8, Re=200） | vivsim/DOI 1.389 | **不可定义（NaN）** | **否** | **否** |
| \|Cl\|_peak（D=8, Re=200） | vivsim/DOI 0.718 | **不可定义（NaN）** | **否** | **否** |
| Cd（D=4, Re=100, 200 步） | vivsim/DOI 1.389 | **−101.04%** | **否** | **否** |
| \|Cl\|_peak（D=4, Re=100, 200 步） | vivsim/DOI 0.718 | **−89.58%** | **否** | **否** |
| St | 0.197（非判据） | 62.4%（工件自算，目标档） | 否 | 否 |

**显式回答：没有任何一项 <3%，也没有任何一项 <15%。基准判定 = FAIL。**

## 判级与落表

- **L0 文档**：有（`docs/XVAL-LBM-ONE-FILE-PER-CASE-PURE-STEP.md`，6931 B）→ 满足。
- **L1 数学/单元验证**：交付件**无任何测试**（`TensorLBM/tests` 对新四件套 0 引用）→ 不满足。
- **L2 独立功能实现**：真代码（`cylinder.py` 135 行 / `collision_2d.py` 116 行 / `boundaries.py` shim 42 行，
  非占位）+ 入口（391 行 argparse 脚本）+ 非空运行工件（JSON）→ **满足（机器判据）**，
  但工件内容是 NaN，四件套中两件（BC / 受力）经探针证明结构性错误。
- **L3 系统集成**：新模块只被 xval 脚本自己引用，主求解器路径无调用方 → 不满足。
- **L4 基准验证**：ref vs got = **NaN / −101%**，远超 15% → **FAIL**。
- **L5**：挂在 L4 上 → 不满足。

**级别 = L2**（功能骨架成立、验证面全缺）。与 m18071「succeeded」的差异写在 `judge.json.material_gap_vs_claim`：
任务目标是「与成熟开源实现交叉对标（<15%）」，本例**判据未达成**，且交付未提交（`head_commit=null`），
anchor 亦未在 `tensor_benchmark_cases` 登记。

## 建议的返修面（供后续返修单/提案用，不在本单执行）

1. 修 `far_field_bc_2d` 的轴：入口写 `f[:, :, 0]`（未知方向 3/6/7），出口零梯度写 `f[:, :, -1] = f[:, :, -2]`，
   顶/底写 `f[:, 0, :]` / `f[:, -1, :]` 的未知方向。
2. 修 `compute_obstacle_forces`：链路集改为 `surface & neighbour_fluid`（或流体侧节点取入射分布），
   带 Ladd 2 倍因子，禁用跨周卷绕（padding 而非 roll）。
3. 补 1 个一行的 BC 形状/作用域单元测试 + 1 个 Poiseuille/均匀流力矩归零测试 → 可拿 L1，且能拦住 F1/F2。
4. 在 `tensor_benchmark_cases` 登记 `XVAL-LBM-ONE-FILE-PER-CASE-PURE-STEP`（Cd 1.389 / Cl 0.718，±15%）。
5. 目标档先做到有限稳定解再生（例如对准 vivsim 的 NEBB/KBC 或先用 BGK + 小 D 收敛），把 NaNs 干掉再说精度。

---

### 证据指针

| 键 | 指针 |
|---|---|
| impl_files | `/root/tp-agent/work/m18071/TensorLBM/src/tensorlbm/boundary/cylinder.py:57-135`、`.../src/tensorlbm/collision_2d.py:1-116`、`.../src/tensorlbm/boundaries.py:1-42` |
| entry | `/root/tp-agent/work/m18071/benchmarks/xval/xval_vivsim_cylinder_re200.py:340-391` |
| run_artifacts | `/root/tp-agent/work/m18071/runs/xval_quick_test.json`；`artifacts/board_audit_m18162/repro/target_D8_re200.json` |
| tests | 无（`TensorLBM/tests` 对新模块 0 引用） |
| integration_callers | 无（仅 xval 脚本自用） |
| benchmark | `artifacts/board_audit_m18162/referee.json`（Cd 1.389/Cl 0.718 vs NaN，±15%） |
| repro_cmd | `PYTHONPATH=TensorLBM/src python3 benchmarks/xval/xval_vivsim_cylinder_re200.py --D 8 --u 0.05 --re 200 --steps 4800` |
| external_anchor | `vivsim/examples/2d/flow_pass_cylinder.py:78-85 @ fb84b12` + 3 DOIs |
| 缺陷探针 | `artifacts/board_audit_m18162/defect_probe.py`（exit 0 = 缺陷成立） |
| 快照/哈希 | `artifacts/board_audit_m18162/snapshot_manifest.json` |
| 裁决 | `artifacts/board_audit_m18162/judge.json`、`findings.json` |