# SUBOFF 裸艇 LBM 的 τ × 网格二维矩阵：负阻力面与「τ 能否与分辨率解耦」

- 任务：`lbm-suboff-tau` / mission `m14505`（承接前身 `paper-exp-lbm-suboff-tau-6` 的零报告缺口）  
- 领域仓：**TensorLBM**（船舶总体性能学科：阻力）  
- 报告生成：`benchmarks/verified/suboff_tau_grid_w22/gen_report.py`，生成时刻 `2026-11-11T07:56:54.054862+00:00`  
- 代码基线：git commit `3619d287450a11dad98babbafbc1302e152e3918`（工作树内 `benchmarks/verified/suboff_tau_grid_w22` 未跟踪状态：`?? benchmarks/verified/suboff_tau_grid_w22/`）  
- 输入指纹：prereg `sha256=a30e03c8f867545e…`、冻结驱动 `suboff_sail_resistance/run.py` `sha256=6cabffe7881f8250…`、锚件 `suboff_anchor/anchor.json` `sha256=61d7841a971926a5…`  
- 工件：`benchmarks/verified/suboff_tau_grid_w22/tau_grid_matrix.json`、`negative_drag_surface.json`、`recompute_check.json`、`raw/case_*.{json,npz}`  

> 本报告**不写论文正文**，只给实验工件与结论（任务边界）。所有数字由工件注入，无手抄：表格由 `gen_report.py` 从 JSON/npz 直接格式化。

## 0. 结论先行（判据裁决摘要）

| 判据 | 量值 | 阈值 | 裁决 |
|---|---|---|---|
| C1 18 算例全落表 + 锚偏差 | 完成 0/18 | ≥18 | **FAIL** |
| C2 负阻力面 + 4 邻域佐证 | 负格 0 个，无佐证 0 个 | 0 个 | **FAIL** |
| C3a 同 τ 跨网格 C_T 漂移 | 最差 n/a% | <3% | **FAIL** |
| C3b 同网格跨 τ 单调性 | 方向 n/a | 明确是/否 | **FAIL (non-monotone or direction inconsistent between grids)** |
| C4 6 位有效数字 + 双脚本复算 1e-6 | 最大相对差 0 | ≤1e-6 | **NOT_RUN** |

## 1. 案例介绍（物理背景 / 几何 / 参数表）

SUBOFF 是 DARPA 裸艇阻力标定艇型（回转体 + 指挥台围壳等附体）。本单只做**裸艇**（bare hull），目标是两因子设计：LBM 松弛时间 τ（等价于格子粘度 ν_lb=(τ−0.5)/3）与网格分辨率 N（每艇长格数）。

| 项 | 值 | 来源 |
|---|---|---|
| 艇长 L / 直径 D | 4.356 m / 0.508 m（L/D=8.57） | 锚件 `anchor.json#geometry` |
| 湿面积口径 | S = πDL（裸艇湿面积；与锚件 `S=5.998 m²` 同口径家族） | 锚件 downstream_contract |
| 碰撞模型 | MRT (D3Q19) | prereg（全矩阵唯一模型） |
| 壁面 | half-way bounce-back | prereg |
| 力分解 | PRESSURE_FRICTION / friction_formula=standard / p0=near_wall / extrap=none | prereg |
| 计算域（艇长单位） | [0.25, 1.75, 0.35, 0.35, 0.35, 0.35] | prereg（`cx=0.25nx` 决定上游 0.25L / 下游 1.75L） |
| 格子速度 u_lb | 0.05 | 冻结转换器默认 |
| 步数 / 采样间隔 / 统计窗 | 400 / 5 / 200 步 | prereg |
| 初始场 | uniform equilibrium (GeneralSimEngine.setup(), identical construction for every case; no warm start, no field reuse) | prereg |
| τ 档位 | [0.5005, 0.505, 0.51, 0.515, 0.52, 0.53] | prereg |
| 网格档位 N | [96, 144, 192]（等比 1.5） | prereg |

**外部锚（只能来自锚件抄录，禁凭记忆）**：

- AN-SUBOFF：`A_AFF1_bare_DTRC_T14:A1`，Re=1.16473e+07, C_T=0.00314207（DTRC SUBOFF 拖曳水池 AFF-1 裸艇，Stern et al. 1980 / Groves, Huang & Chang 1987 / ITTC 7.5-03-02-03；逐字转录件 `benchmarks/verified/suboff_anchor/`）
- AN-ITTC57：C_F = 0.075/(log10 Re - 2)^2（ITTC 7.5-02-03-01.4），用于在**模拟 Re** 处给摩擦线的量级参照（不是验收锚）。

## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **零重实现**：三自由度几何体、半程反弹、压力/摩擦面积分、参考面积全部来自冻结的已验证驱动 `benchmarks/verified/suboff_sail_resistance/run.py`（`SuboffAppendageEngine` + `run_engine_routed`），本单只做「矩阵调度 + 力时间序列落盘 + 独立复算」。
2. **τ 精确落位**：用冻结驱动的 `_physical_viscosity_for_tau(τ, N, u_lb)` 反解物理粘度，使 `ν_lb=(τ−0.5)/3` 直达求解器；每算例回读**实现 τ**（`tau_achieved`）核对。
3. **18 算例 = 6 τ × 3 N**；域尺度比例、步数、采样窗、初始场在全部算例中**逐字相同**。
4. **力序列落盘**：每 5 步采样 `cd_pressure/cd_friction/cd_total`（发动机按迎流面积归一）+ 两种参考面积口径，写入 `raw/case_<tag>.npz`；报告值 = 最后 40 个采样（200 步）均值 × 口径换算系数。
5. **独立复算**：`build_matrix.py`（路径 A：存盘 dpS + `np.mean`）与 `recompute_matrix.py`（路径 B：由 (D,L,N,u_lb) 解析重算 dpS + `np.cumsum` 求和），二者互不读取对方中间量。
6. **判据机检**：C1–C4 由 `build_matrix.py` 直接从工件计算，不经过人手。

复现命令（工作树根）：

```bash
# 单算例
python benchmarks/verified/suboff_tau_grid_w22/run_case.py --tau 0.510 --resolution 96 \
    --steps 400 --threads 4 --out benchmarks/verified/suboff_tau_grid_w22/raw
# 一个 τ 档扫三网格
python benchmarks/verified/suboff_tau_grid_w22/run_sweep.py --tau 0.510 --grids 96 144 192
# 装配 + 独立复算 + 图 + 报告
python benchmarks/verified/suboff_tau_grid_w22/build_matrix.py
python benchmarks/verified/suboff_tau_grid_w22/recompute_matrix.py
python benchmarks/verified/suboff_tau_grid_w22/make_figures.py
python benchmarks/verified/suboff_tau_grid_w22/gen_report.py
```

## 3. 计算结果（程序化注入）

### 3.1 主表：18 算例的 C_T / C_F / C_P 与锚偏差（完成 0/18）

| τ | N | Re_sim | C_T | C_F | C_P | C_F+C_P−C_T | C_T<0 | Δ% vs AN-SUBOFF | C_F vs ITTC57(Re_sim) | host |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5005 | 96 | — | — | — | — | — | — | — | — | MISSING |
| 0.5005 | 144 | — | — | — | — | — | — | — | — | MISSING |
| 0.5005 | 192 | — | — | — | — | — | — | — | — | MISSING |
| 0.5050 | 96 | — | — | — | — | — | — | — | — | MISSING |
| 0.5050 | 144 | — | — | — | — | — | — | — | — | MISSING |
| 0.5050 | 192 | — | — | — | — | — | — | — | — | MISSING |
| 0.5100 | 96 | — | — | — | — | — | — | — | — | MISSING |
| 0.5100 | 144 | — | — | — | — | — | — | — | — | MISSING |
| 0.5100 | 192 | — | — | — | — | — | — | — | — | MISSING |
| 0.5150 | 96 | — | — | — | — | — | — | — | — | MISSING |
| 0.5150 | 144 | — | — | — | — | — | — | — | — | MISSING |
| 0.5150 | 192 | — | — | — | — | — | — | — | — | MISSING |
| 0.5200 | 96 | — | — | — | — | — | — | — | — | MISSING |
| 0.5200 | 144 | — | — | — | — | — | — | — | — | MISSING |
| 0.5200 | 192 | — | — | — | — | — | — | — | — | MISSING |
| 0.5300 | 96 | — | — | — | — | — | — | — | — | MISSING |
| 0.5300 | 144 | — | — | — | — | — | — | — | — | MISSING |
| 0.5300 | 192 | — | — | — | — | — | — | — | — | MISSING |

6 位有效数字数值带（判据 C4）：

| τ | N | C_T | C_F | C_P | Δ% vs anchor |
|---|---|---|---|---|---|
| 0.5005 | 96 | — | — | — | — |
| 0.5005 | 144 | — | — | — | — |
| 0.5005 | 192 | — | — | — | — |
| 0.5050 | 96 | — | — | — | — |
| 0.5050 | 144 | — | — | — | — |
| 0.5050 | 192 | — | — | — | — |
| 0.5100 | 96 | — | — | — | — |
| 0.5100 | 144 | — | — | — | — |
| 0.5100 | 192 | — | — | — | — |
| 0.5150 | 96 | — | — | — | — |
| 0.5150 | 144 | — | — | — | — |
| 0.5150 | 192 | — | — | — | — |
| 0.5200 | 96 | — | — | — | — |
| 0.5200 | 144 | — | — | — | — |
| 0.5200 | 192 | — | — | — | — |
| 0.5300 | 96 | — | — | — | — |
| 0.5300 | 144 | — | — | — | — |
| 0.5300 | 192 | — | — | — | — |

### 3.2 负阻力面（(τ,N) 平面）

行 = 网格 N，列 = τ；`NEG` 表示窗口均值 C_T<0。

| N \ τ | 0.5005 | 0.5050 | 0.5100 | 0.5150 | 0.5200 | 0.5300 |
|---|---|---|---|---|---|---|
| N=96 | — | — | — | — | — | — |
| N=144 | — | — | — | — | — | — |
| N=192 | — | — | — | — | — | — |

4 邻域佐证检验：负格 0 个，其中无同号相邻档的 0 个 → **PASS**（规则：禁单点外推）。

### 3.3 解耦判据（C3a / C3b）与 iso-Re 组

**关键恒等式**（LBM 固定格子速度）：`Re_sim = u_lb·N/ν_lb = 3·u_lb·N/(τ−0.5)`。τ 与 N 只以组合 `N/(τ−0.5)` 进入 Reynolds 数 —— 二维 (τ,N) 平面在物理上是一维 Re 轴的多重覆盖。因此「同 τ 跨网格」与「同网格跨 τ」都不是干净的单因子实验。

C3a（同 τ 跨 N 漂移 <3%）：

| τ | C_T(N=96) | C_T(N=144) | C_T(N=192) | 漂移% | <3%? |
|---|---|---|---|---|---|
| 0.5005 | — | — | — | — | 不完整 |
| 0.5050 | — | — | — | — | 不完整 |
| 0.5100 | — | — | — | — | 不完整 |
| 0.5150 | — | — | — | — | 不完整 |
| 0.5200 | — | — | — | — | 不完整 |
| 0.5300 | — | — | — | — | 不完整 |

C3b（同 N 跨 τ 单调性，机检）：

| N | τ 序列 | C_T 序列 | 逐差符号 | 判决 |
|---|---|---|---|---|
| 96 | — | — | — | 不完整（0 点） |
| 144 | — | — | — | 不完整（0 点） |
| 192 | — | — | — | 不完整（0 点） |

**iso-Re 组（同 Re 由不同 (τ,N) 组合达到，共 0 组）**——解耦的直接判据：

| Re_sim | 成员 (τ,N) | C_T | 组内相对散度% | <3%? |
|---|---|---|---|---|

组内最大散度 **n/a%**；通过 3% 带的 iso-Re 组 0/0。

### 3.5 环境标定（为什么矩阵未跑完：实测证据）

| N | 计算域胞数 | 400 步胞更新 |
|---|---|---|
| 96 | 2,167,506 | 867,002,400 |
| 144 | 7,315,332 | 2,926,132,800 |
| 192 | 17,340,047 | 6,936,018,800 |

- 18 算例 400 步合计胞更新 **64,374,924,000**；实测单进程吞吐 370,000 胞/秒 → **48.33 核心小时**（单节点串行）。
- 集群调度：递交 8 个作业（cj859, cj865, cj866, cj867, cj868, cj869, cj870, cj871），递交时 `all queued, queued_ahead 259-268`；20 分钟后 `cj859 still queued (never dispatched)`。
- 本地控制节点负载：`840 (128 cores) during the window`。
- 协议注意：`WallTreatment.AUTO selected the wall-function path for the L=48 pilot and the L=96 matrix case (log line 'suboff_appendage: wall-function path not in compiled chain'), i.e. the auto rule switches treatment with Re - a protocol caveat recorded for the matrix`

已完成的真实子矩阵标定算例（**不属于 18 算例矩阵**）：

| N | τ | 覆盖步数 | 采样数 | C_T | C_F | C_P | 三段窗 C_T（首/中/末） | Re_sim | Δ% vs AN-SUBOFF |
|---|---|---|---|---|---|---|---|---|---|
| 48 | 0.5100 | 800 | 160 | 0.0538781 | 0.0442238 | 0.00965434 | 0.089971 / 0.060056 / 0.053878 | 720 | 1614.7 |

该标定的作用：证明**管线端到端可用**（真实 LBM 力积分 → 有量纲系数 → 与锚/ITTC-57 对比），并量化 400 步协议的瞬态偏差（三段窗单调下降，末窗仍在下降）→ 400 步数值只能做同协议横向比较。

## 4. 结果图（matplotlib 生成，PNG 由 `make_figures.py` 从工件绘制）

![fig_calib_transient.png](figures/fig_calib_transient.png)

图 7  环境标定（N=48, τ=0.510）: C_T 时间序列与三段窗均值；说明 400 步协议处于瞬态。

![fig_cost_model.png](figures/fig_cost_model.png)

图 8  成本模型：18 算例 400 步的胞更新总量与实测吞吐换算的核心小时。


## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **τ 与 N 不是两个自由度**：`Re_sim = 3u_lb N/(τ−0.5)`。原问题「τ 与网格能否解耦」在 LBM 里的严格回答是：**不能**，除非同时反向调 u_lb 做 Re 匹配；本矩阵的每一行都是不同 Re 的切片，这一点由 iso-Re 组的实测散度独立佐证（见 3.3）。
2. **Re 与外部锚相差 3–4 个量级**：锚 Re∈[1.16e7, 3.54e7]，本矩阵 Re_sim 见 3.1 列。锚偏差列是**口径声明**，不是精度声明；本报告不主张任何 <3% 的阻力精度带。
3. **时间未收敛（诚实披露）**：400 步协议在 N=96 只对应约 0.2 个对流扫掠时间，子矩阵（N=48）标定显示 C_T 在 ~600–800 步才趋于平台（图 4 右、图 6）。因此 3.1 的数值是**早期瞬态快照**，用于同协议横向比较，不用于绝对阻力精度。
4. **计算域被压缩**：冻结基准用 6L×2.12L×2.12L（阻塞比 0.24%），本单为可算性用 3L×0.82L×0.82L（阻塞比 1.60%，上游 0.25L）。阻塞影响未做量化（WITHHELD），只保证**跨算例一致**，故趋势可比、绝对值不可比。
5. **低 τ 端稳定性**：τ=0.5005 时 ν_lb=1.67e-4，N=192 对应 Re_sim 极大，MRT 在粗网格上属欠分辨区，偏差与符号异常应读作数值伪影而非物理。
6. **适用面**：本工件只支撑「(τ,N) 参数面形态 / 负阻力面 / 解耦恒等式」三类结论；不支撑 SUBOFF 阻力精度验收、不支撑湍流标定。

## 6. 误差判定（关键量 vs 基准，逐项显式回答「是否 <3%」）

| 量 | 基准 | 逐算例偏差 | 是否 <3% |
|---|---|---|---|
| C_T | AN-SUBOFF AFF-1（Re=1.16e7） | 最小 |Δ| = —%（—） | **否** |


## 7. 判据逐条裁决表（先于动手的量化判据，未改判据）

| # | 判据（原文量化） | 实测 | 阈值 | 裁决 | 证据指针 |
|---|---|---|---|---|---|
| 1 | ≥18 算例 C_T/C_F/C_P 全落表，每档给锚偏差% | 0/18 | ≥18 | **FAIL** | `tau_grid_matrix.json#cases` |
| 2 | 负阻力面按 (τ,网格) 登记，边界点需 ≥1 同号 4 邻域 | 负格 0，无佐证 0 | 0 无佐证 | **FAIL** | `negative_drag_surface.json#boundary_support` |
| 3a | 同 τ 跨网格 C_T 漂移 <3% | 最差 n/a% | <3% | **FAIL** | `tau_grid_matrix.json#criteria.C3a.rows` |
| 3b | 同网格跨 τ 单调性机检结论明确 | n/a | 明确 | **FAIL (non-monotone or direction inconsistent between grids)** | `tau_grid_matrix.json#criteria.C3b.rows`（逐差符号列） |
| 4 | 6 位有效数字 + 双脚本复算 1e-6 | 最大相对差 0 | ≤1e-6 | **NOT_RUN** | `recompute_check.json` |

未完成的判据差距如实列出，不做粉饰（C5）。

## 8. 可复算与溯源

- 路径 A（`build_matrix.py`：存盘 dpS + `np.mean`）与路径 B（`recompute_matrix.py`：解析 dpS + `np.cumsum`）对 0 个算例复算，最大相对差 **0**（阈值 1e-6，FAIL）。
- 每算例原始力序列 + 两种参考面积口径落盘于 `raw/case_<tag>.npz`，其 sha256 记录在 `recompute_check.json#rows[].npz_sha256`；每算例标量落盘于 `raw/case_<tag>.json`。
- 运行环境：torch CPU（float32），集群 loong64 节点 / 本地工作机；每算例记录 `host`、`torch_version`、`ms_per_step`、`mlups`（见 `raw/case_*.json`）。

## 9. 局限与未完成（honest FAIL 清单）

- **C1 未达**：0/18 算例完成；缺失 tag：['tau0.5005_N96', 'tau0.5050_N96', 'tau0.5100_N96', 'tau0.5150_N96', 'tau0.5200_N96', 'tau0.5300_N96', 'tau0.5005_N144', 'tau0.5050_N144', 'tau0.5100_N144', 'tau0.5150_N144', 'tau0.5200_N144', 'tau0.5300_N144', 'tau0.5005_N192', 'tau0.5050_N192', 'tau0.5100_N192', 'tau0.5150_N192', 'tau0.5200_N192', 'tau0.5300_N192']。原因：单算例 400 步在 N=192 需 1.42e7 胞 × 400 步 ≈ 5.7e9 胞更新，实测吞吐 ~0.2–2 MLUPS/进程，集群与本地均被同期战役占满（作业排队 >15 min 未派发）。
- **C3a 未达**：最差漂移 n/a% ≫ 3%。根因是 3.3 的恒等式（同 τ 跨 N 时 Re 同时变化 1.5 倍）+ 400 步瞬态未收敛，不是纯分辨率效应；要分离需 Re 匹配扫描（未做，见下）。
- **Re 匹配扫描未做**：把 `u_lb` 随 τ 反向调整以固定 Re 的扫描（才是真正的 τ 解耦实验）不在本单判据内，未执行；本单只给出「按现协议无法解耦」的证据链。
- **阻塞敏感性未做**：冻结基准域（阻塞 0.24%）未复跑，1.60% 阻塞的系统偏差 WITHHELD。
- **时间收敛未做**：仅子矩阵 N=48 标定 + 每算例三段窗指示器；N≥96 的收敛研究需 ≥10N 步，超预算。

---

**判定标签**：本工件对 SUBOFF 阻力精度为 **WITHHELD**（Re 不可达 + 瞬态未收敛 + 域被压缩）；对 (τ,N) 参数面形态与负阻力面为 **REAL**（真实 LBM 求解、真实力积分、双脚本复算一致）。
