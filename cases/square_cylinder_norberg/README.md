# CASE-LBM-SQUARE-CYL-NORBERG — 方柱绕流涡脱 St / Cd @ Re=200（阻塞比 1/20）

第 2 轮（调研结合）交付 · mission m16321 · 仓 TensorLBM · 分支 `agent/m16321`

**诚实结论先行（honest partial）**：`Cd` 达标（+7.7% < 10%），`St` **未达标**（+31% > 8%）。
且**参考值本身证据等级只有 WITHHELD**（Norberg 1993 原文闭源，环境内无法机读原表，见 §1.2）。
不粉饰：本单给出的是真实运行数据 + 真实误差 + 参考口径的证据链缺陷。

---

## ① 问题设定与基准出处

**算例**：二维自由流方柱（正置 α=0°）绕流涡脱，目标量两项
- St = f·D/U（涡脱 Strouhal 数，f 为升力谱峰主频）
- Cd = ⟨Fx⟩/(½ρU²D)（平均阻力系数）

**工况（按锚点规定）**：Re = U·D/ν = 200，阻塞比 = D/H = 1/20 = 5%，α = 0°。

**基准出处（原文献）**：
Norberg, C. (1993) *Flow around rectangular cylinders: pressure forces and wake
frequencies*, **J. Wind Eng. Ind. Aerodyn. 49(1-3): 187–196**,
DOI `10.1016/0167-6105(93)90014-f`。

### 1.2 参考值的证据等级 = WITHHELD（本轮核心发现）

| 检索手段 | 结果 |
|---|---|
| Crossref `api.crossref.org/works/10.1016/0167-6105(93)90014-f` | ✅ 200：标题/期刊/卷期/页码/作者（C. Norberg）**逐项核对一致**；abstract 字段为空 |
| Unpaywall `api.unpaywall.org/v2/<doi>` | ❌ `is_oa=false, oa_status=closed` — 闭源，无合法开放全文 |
| fatcat / scholar.archive.org 副本查找 | ❌ 超时无响应（工作区策略亦拦截 ScienceDirect / 搜索引擎抓取） |
| arXiv 全文横向（`tools/norberg1993_ar5iv.py`，18 篇回读，候选并入） | ❌ 仅命中参考文献列表条目，无正文数值表引用 Norberg-1993 的 St/Cd |

→ **原表数值在本环境内不可机读**。本轮采用**二级转述值** `St=0.13, Cd=1.5`
（来源：第 1 轮 m16247 归因于 "Table 2 p.195 / p.194-195"，**该归因本轮无法复核**），
存于 `refs/norberg1993.json`，字段 `provenance` 明文标注 WITHHELD。
把它当作"锚点自带评分常数"使用，**不升级为 REAL 参考**。零手抄要求因此对基准侧打叉：
本单的"got"数字全部是程序化注入（`result_D*.json`），"ref"数字是二级转述。

---

## ② 方法与实现（方程/格式出处 + 代码指针）

工程链=复用仓内已验证求解路径，不另起炉灶：

| 环节 | 实现 | 代码指针 |
|---|---|---|
| 碰撞 | D2Q9 **MRT**（可带 `tau_field` 逐格松弛，用于 sponge） | `src/tensorlbm/solver.py:276 collide_mrt` |
| 迁移 | pull-scheme streaming | `src/tensorlbm/solver.py: stream` |
| 边界 | 入口自由流 Dirichlet + 上下自由流 + 出口零梯度 + 障碍半程反弹（一次调用） | `src/tensorlbm/boundary/boundaries.py:201 far_field_bc_2d` |
| 吸收层 | 下游 sponge：σ(x)=clamp((x−x0)/w,0,1)²，τ_eff=τ(1+ασ)，α=10 | `:390 make_sponge_strength` |
| 测力 | Ladd 动量交换法，post-stream / pre-bounce-back 采样 | `:164 compute_obstacle_forces` |
| 播种 | 入口 2 个 St_seed 周期、10% 幅值横向正弦，之后回归自由流 | `run_chunked.py: _step_seeded` |
| 后处理 | Cd 均值（后 75%/50% 窗）+ Cl 谱峰（Hann 窗 + 抛物线插值）+ 滞回过零交叉验证 | `benchmarks/verified/square_cylinder/run.py: analyze_forces`（本轮从 `run_case` 抽出为可复用函数） |
| 可续算驱动 | 每调用 ~100 s 墙钟预算，`f`+力历史 checkpoint 到 `.npz`，下次续算 | `cases/square_cylinder_norberg/run_chunked.py` |

数值参数：ν = U·D/Re = 0.05·D/200 = 2.5e-4·D（D=10 → ν=2.5e-3，τ=0.5075），
格子速度 U=0.05（Ma = U/c_s ≈ 0.087 < 0.1，不可压成立）。

**本轮修复的真实缺陷**（记录在案，不掩盖）：
`run_chunked.py` 初版在 **pre-stream** 状态上采样动量交换力，导致 D=10 全程
Cd = **−1.615**（符号翻转）。改为 post-stream / pre-bounce-back 后 Cd = **+1.6159**。
此类符号错误在"只跑通不看符号"的流程里会静默通过 —— 已用 `cd>0` 的物理断言在结果
JSON 暴露（`cd_pass`/`cd` 字段可见），并在代码注释中留因。

---

## ③ 参数表（无量纲数完整）

| 量 | D=8 | D=10 | 说明 |
|---|---|---|---|
| Re = UD/ν | 200 | 200 | 目标 |
| 域（x×y） | 20D × 20D | 20D × 20D | y 向 = 20D → **阻塞比 1/20 = 5%**（锚点规定） |
| 阻塞比 D/H | 0.05 | 0.05 | ✅ 与锚点一致 |
| 格子速度 U | 0.05 | 0.05 | Ma≈0.087 |
| 运动黏度 ν (lb) | 2.0e-3 | 2.5e-3 | = U·D/Re |
| 松弛 τ | 0.506 | 0.5075 | τ = 0.5+3ν |
| 方块中心 x_c | 6D | 6D | 上游 5.5D |
| 下游长度 | 14D | 14D | 含 6D sponge（α=10, power 2） |
| 播种 St_seed / 幅值 | 0.13 / 10% | 0.13 / 10% | 2 周期 |
| 总步数 | 14000 | 24000 | 涡脱周期 T=D/(St·U)≈1067/1333 步 |
| 统计学窗 | 后 50%（7000 步 ≈ 6.6 周期） | 后 25%（6000 步 ≈ 4.5 周期） | D=10 窗偏短（见 §⑤ 局限） |

---

## ④ 对照表（ref vs got + 误差，判据逐条打勾/打叉）

参考：`St_ref = 0.13`（容差 <8%）、`Cd_ref = 1.5`（容差 <10%）— 证据等级 **WITHHELD（二级转述）**。

| 网格 | Cd got | Cd 误差 | <10% 判据 | St got | St 误差 | <8% 判据 | 交叉验证 St(过零) |
|---|---|---|---|---|---|---|---|
| D=8 | 1.6067 | **+7.12%** | ✅ PASS | 0.1679 | **+29.16%** | ❌ FAIL | 0.1686 |
| D=10 | 1.6159 | **+7.73%** | ✅ PASS | 0.1706 | **+31.27%** | ❌ FAIL | 0.1687 |

判据合计：**Cd 2/2 打勾 · St 0/2 打叉** → 锚点验收 **部分达标（honest partial）**。

解读（不辩解，只陈述机制）：
1. Cd 系统性偏高 ~7-8%，与"阻塞比 5% 抬高阻力"的物理预期同向（本域 y=20D 完全按锚点的
   blockage 1/20 设置，未做阻塞修正）。
2. St 高 29-31%：二维求解器在低 Re 高阻塞下高估涡脱频率（三维展向失稳缺失 +
   阻塞加速）。**这不是数值未收敛**——两档网格 St 只差 1.6%，D=8→D=10 单调略升
   （0.1679→0.1706），趋向 ~0.17 而非 0.13。
3. 若改用"2D 低阻塞数值簇"口径（St≈0.15-0.16），误差可降到 ~+7%；但那**不是锚点规定的
   参考**，只能作为讨论，不计入判据（`refs/norberg1993.json: sensitivity_bracket`）。

---

## ⑤ 收敛性 / 参数敏感性

| 对比 | Cd | Δ | St | Δ |
|---|---|---|---|---|
| D=8 | 1.6067 | — | 0.1679 | — |
| D=10 | 1.6159 | **+0.57%** | 0.1706 | **+1.61%** |

- 两分辨率 span 均 <2% → 网格收敛性满足（判据线 3% 内）；但**单调性**显示细网格 St 仍在上升，
  真值更可能在 0.17 一侧，进一步坐实 St 判据的 FAIL 不是网格噪声。
- 独立参数点（St 提取法）：FFT 谱峰 vs 滞回过零交叉，D=8 差 0.4%、D=10 差 1.1%
  （0.1706 vs 0.1687）→ St 提取本身稳健，不是后处理伪影。
- 已知局限：D=10 的统计窗仅 ~4.5 个涡脱周期（预算所限，逐 ~100 s 续算 6 段），
  Cd 均值的统计不确定度估计 ~±2%；D=8 窗 ~6.6 周期。二者均不足以推翻 30% 的 St 偏差。

---

## ⑥ 失败模式与工件

**失败模式 / 摩擦（如实记录）**
1. **参考闭源**：Norberg-1993 原表无法机读（§1.2 四种检索手段全数记录）→ 基准侧证据 WITHHELD，
   锚点被要求"从 ref 原文查得（零手抄）"这一条**本单无法满足**，且不是本单可修复的（需机构订阅）。
2. **工具域硬限制**：单条 shell 调用 120 s 硬超时（`sleep 200` 被整组收割已复现）→ 长时程
   LBM 必须"分块续算 + checkpoint"（`run_chunked.py` 即为此而生）。
3. **集群排队**：probe 作业 cj1302 递交后 3 min 仍 `queued`（队列 ahead=467）→ 本单改为
   本地分块计算；集群路线未走通（honest 记录：cj1302 未取回工件）。
4. **本机 128 核被其他任务打满（load≈116）**：torch 默认线程数导致 184 ms/step；
   `torch.set_num_threads(4..5)` 后降到 ~10-15 ms/step（13× 提速），此坑值得进 TK。
5. **符号缺陷**：见 §②，pre-stream 采样致 Cd 负号，已修复并以正号结果重采样。
6. **未完成项**：未做三重网格（D=12/16）+ 未做阻塞比扫描（1/10、1/20、自由滑移）+
   未在集群复算（队列不可用）。这些是下一轮的明确增量。

**工件与复现**

```
# 一键复现（分块；每调用 ≤100 s，D=10 约需 4 次、D=8 约需 2 次）
cd <repo>
for i in $(seq 1 6); do python3 cases/square_cylinder_norberg/run_chunked.py \
  --D 10 --steps 24000 --domain-x 20 --domain-y 20 --x-center 6 --sponge-d 6 \
  --budget 90 --threads 5 --drop-history --warmup-frac 0.25 \
  --state runs/m16321/sqcyl_norberg/D10/state.npz; done
python3 cases/square_cylinder_norberg/make_figures.py
```

| 工件 | 路径 |
|---|---|
| D=10 结果（主） | `runs/m16321/sqcyl_norberg/D10/result_D10.json` |
| D=8 结果 | `runs/m16321/sqcyl_norberg/D8/result_D8.json` |
| 力时间序列（Cd/Cl 逐采样） | `runs/…/D{8,10}/series_D{8,10}.npz` |
| 续算状态（可断点复核） | `runs/…/D{8,10}/state.npz` |
| 图：力历史 / Cl 谱 / ref-vs-got | `runs/…/figs/fig1_force_traces.png`, `fig2_cl_spectrum.png`, `fig3_ref_vs_got.png` |
| 参考口径与证据链 | `cases/square_cylinder_norberg/refs/norberg1993.json` |
| 求解链复用与抖动 | `benchmarks/verified/square_cylinder/run.py`（新增 `analyze_forces`、`--domain-x/--domain-y/--ref-cd/--ref-st` 等，默认值保持 Re=100 原行为） |
| 驱动 | `cases/square_cylinder_norberg/run_chunked.py` |

**证据分级**：实现=REAL（库函数真调用，无 stub）；got 数字=REAL（真实运行 JSON）；
ref 数字=WITHHELD（二级转述，原表不可达）；St 判据=FAIL（真实、可复现）。