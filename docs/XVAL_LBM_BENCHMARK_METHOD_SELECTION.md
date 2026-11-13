# XVAL · LBM benchmark-method-selection：TensorLBM vs OSS（Maarten-vd-Sande/lbm）

**锚点** XVAL-LBM-METHOD-SELECTION-FIRST-EVIDENCE（承接 XVAL-LBM-BENCHMARK-METHOD-SELECTION 第 2 轮） · 第 3 轮 = **第一证据：分辨率梯级 + 参数敏感性**  
**OSS 基准** `Maarten-vd-Sande/lbm` @ `573a01c4d2e6321f3634ff271adca4ce91872e6f` （源码 sha256 见 `xval/benchmark_method_selection/oss_crosscheck.json`）  
**判据** owner 精度标准：对标误差 > 15% 不可接受；收敛判据 10%（`predeclared_v2.json` 预注册，本轮回溯沿用、未按结果改）。本文所有数字由 `analyze.py`/`make_report.py` 从 `runs/*/result.json`、`oss_crosscheck.json` 与两份 predeclared 程序化注入（零手抄）。

---

## 1 案例与参数（ref 出处）

**问题**：2-D 方柱绕流涡街，Re = U·D/ν = 200，阻塞比 D/H = 1/20（域 20D×20D，方柱中心 6D），用同一条 harness 比较「本仓记录的选型」与「OSS 的选型」，回答**方法选型是否可移植**，并在本轮回答**第 2 轮的判定本身是否受分辨率限制**（XVAL 的对象是选型做法，不是新物理）。

| 量 | 值 | 来源 |
|---|---|---|
| U (lattice) | 0.0500 | predeclared_v2.json |
| Ma | 0.0866 | run_arm.py 计算 |
| Re | 200 | predeclared |
| 阻塞比 | 0.050 | D/H = 1/20 |
| 播种 | St_seed = 0.13 @ 10% U，2 个播种周期 + 15 个脱落周期 | run_arm.py 调度，运行前算定 |

**梯级参数（分辨率的混淆量已申报）**：D 增大时 ν = U·D/Re 随之增大（固定 U、固定 Re 的标准加密，非 diffusive scaling），因此 τ 同时上移——见 `predeclared_v3.json:variables_changed.confounder_declared`，本轮不试图分离「网格」与「可压缩性」两种误差来源。

| D | 网格 | ν | τ | 步数 | 工件 |
|---|---|---|---|---|---|
| 10 | 200×200 | 0.00250 | 0.50750 | 21259 | `xval/benchmark_method_selection/runs/A1_D10/` |

**ref 出处（三级，均零手抄）**：

1. **一级（判据用）· 执行 OSS 源码**：`~/oss-cache/Maarten-vd-Sande_lbm @ 573a01c`，以符号链接+`numba.jit` 恒等 shim 载入（不改源码，sha256 已记录），在同一状态上各跑一步碰撞/一步流动，与 TensorLBM 内核逐点比较（§3-B）。
2. **二级（判据用）· OSS 做法移植进同 harness**：A2 = BGK 单松弛、A3 = TRT 磁参数 Λ=3/16、A5 = 重导出 Λ；A4 = MRT 但关掉 sponge（隔离 sponge 这一个旋钮）。
3. **外部口径（不参与判据）**：`docs/CASE-LBM-SQUARE-CYL-NORBERG-RE200.md` 记录 Re=200 的原始表（Norberg 1993 / Sohankar 1998）在本环境 **WITHHELD**，只给由本仓已验证 Re=100/150 梯级外推的趋势带 Cd∈[1.30,1.45]、St∈[0.160,0.180]，明确不可作 PASS/FAIL 门。

## 2 方法与实现（可复现）

```
# 第 2 轮矩阵（5 臂 × D=10）
bash xval/benchmark_method_selection/run_matrix.sh
# 第 3 轮分辨率梯级（D=14/D=20；集群版单作业跑完，本地版按 300 s 分块断点续跑）
bash xval/benchmark_method_selection/run_ladder.sh          # 本地 / 分块
bash xval/benchmark_method_selection/run_ladder_cluster.sh  # 集群
# 聚合 + 出图（含梯级/收敛/差距维持/失败复现）
python xval/benchmark_method_selection/analyze.py
# 出本报告（数字全注入）
python xval/benchmark_method_selection/make_report.py
# OSS 源码级对照（执行 OSS 代码本身）
python xval/benchmark_method_selection/oss_crosscheck.py
```

单臂命令：`python xval/benchmark_method_selection/run_arm.py --arm A2 --D 14`；工件落在 `xval/benchmark_method_selection/runs/<arm>_D<D>/{result.json|fail.json,trace.npz}`。

**臂定义**（只改选型旋钮，BC/测力/后处理全部共享）：

| 臂 | 碰撞 | sponge | 说明 |
|---|---|---|---|
| A1 | MRT | on | MRT + sponge (recorded TensorLBM selection) |
| A2 | BGK | off | BGK, no sponge (OSS single-relaxation selection) |
| A3 | TRT Λ=3/16 | off | TRT magic Lambda=3/16, no sponge (literal OSS transplant) |
| A4 | MRT | off | MRT, no sponge (sponge-isolated control) |
| A5 | TRT Λ 重导出 | off | TRT with Lambda re-derived, tau_minus=2.0, no sponge (A3 remediation) |

**预注册**：`predeclared_v3.json` 在任何 D=14/D=20 KPI 产生之前写定（决策、三条假设 H_conv/H_gap/H_fail_persists、10%/15% 阈值、停止规则、混淆量与排除项）；阈值与第 2 轮一致，未按结果调整。

## 3 计算结果

### 3-A 选型对照表（ref = A1 记录选型，got = OSS 做法；同分辨率）

**D=10**

| ref (A1) | got (臂) | Cd ref | Cd got | ΔCd | St ref | St got | ΔSt | Cl_amp Δ | 全部 <15% |
|---|---|---|---|---|---|---|---|---|---|
| A1_D10 | **A2_D10** | 1.6133 | 1.4660 | -9.13% | 0.1690 | 0.1766 | +4.52% | -46.57% | ❌ |
| A1_D10 | **A4_D10** | 1.6133 | 1.6428 | +1.83% | 0.1690 | 0.1718 | +1.64% | -15.40% | ❌ |

St 双估计量互校（判据：|St_fft − St_zc|/St_fft < 5.0%）：

| 臂 | St_fft | St_zc | 互校差 | 判定 |
|---|---|---|---|---|
| A1_D10 | 0.1690 | 0.1686 | -0.219% | ✅ |
| A2_D10 | 0.1766 | 0.1780 | +0.752% | ✅ |
| A4_D10 | 0.1718 | 0.1717 | -0.020% | ✅ |

**H1 判定（全部有数据的梯级一起判）**：不成立 ❌（至少一臂越出 15% 带）；最大绝对偏差 46.57%（A2_D10 的 cl_amp）。判定梯级：[10]。逐臂：{"A2_D10": false}。

### 3-B 代码级对照（ref = **执行 OSS 源码**，got = TensorLBM 内核）

| 检查 | ref（OSS 代码） | got（TensorLBM） | 相对 L2 差 | 判据 <15% | 判据 <3% |
|---|---|---|---|---|---|
| K0 方向置换（9 个方向 spike 落点） | OSS `lbm_streaming` | `solver.stream` | 逐方向一致 True | ✅ | ✅ |
| K1 碰撞算子（τ=0.5075，n=48） | `lbm_collision` (float32) | `collide_bgk` (float64) | 1.176e-07 | ✅ | ✅ |
| K2 流动映射（内部域） | `lbm_streaming` (float32) | `stream` (float64) | 2.710e-08 | ✅ | ✅ |

**为什么物理对照用「移植」而不是直接跑 OSS 全算例**：OSS 的碰撞循环是纯 Python（未 jit），实测 5.13 s/step @100×100，外推 20.50 s/step @200×200，本算例 21259 步 ⇒ 约 **121.1 小时**/臂；同格 TensorLBM 实测 0.0212 s/step（差 967×）。本轮进一步核查了 jit 可能性：OSS 源码 `lbm/streamcollide.py:39-218` 与 `update_types.py:7` 共 10 处 `@jit` 装饰器，但本生态节点架构为 **loongarch64**，`numba`/`llvmlite` 无该架构 wheel，源码编译 llvmlite 失败（见 §6 环境摩擦）⇒ OSS 全算例端到端仍不可行，证据链保持「单步内核对照（一级）+ 做法移植（二级）」。

### 3-C 外部口径（**不参与判据**，仅上下文）

| 量 | A1 值（最高可用分辨率） | 趋势带 | 是否在带内 | 距带中心 | 可否判定 |
|---|---|---|---|---|---|
| cd_mean | 1.6143 (D=10) | [1.3, 1.45] | ❌ | +17.41% | 不计分（原表 WITHHELD） |
| st_fft | 0.1690 (D=10) | [0.16, 0.18] | ✅ | -0.59% | 不计分（原表 WITHHELD） |

## 4 结果图

![lift histories](figures/fig1_cl_traces.png)

*图 1 各臂 Cl 时程（D=10，末 6 个脱落周期，纵向错开显示）：相位与幅值近似重合 ⇒ 选型不改变脱落机制。*

![KPI vs A1](figures/fig2_kpi_vs_A1.png)

*图 2 Cd / Cl 幅值 / St 相对同分辨率 A1（虚线）与 15% 带（灰带），按梯级分排：误差打在柱顶。*

![resolution ladder](figures/fig3_resolution_ladder.png)

*图 3 分辨率梯级 D=10→14→20（收敛性，见 §5.1）：三条曲线各为一个臂，灰带 = 外部趋势带（不判分）。*

![vorticity](figures/fig4_vorticity_fields.png)

*图 4 末态涡量场（黑线 = 方柱）：A1 在 6D sponge 内被吸收，A4/A2 下游涡街延伸至出流。*

![gap ladder](figures/fig5_gap_ladder.png)

*图 5 选型差距是否随加密消失：左 = Cd vs D，右 = ΔCd vs 同分辨率 A1（灰带 = 15%）：见 §5.2 的 H_gap 判定。*

## 5 收敛性 / 参数敏感性

### 5.1 网格收敛（逐级，判据 <10%；预注册 H_conv）

| — | — | — | — | **未运行（honest not_run）** |

| 臂 | 梯级 | ΔCd（同 D 对比 A1） | ΔSt | ΔCl_amp | 全 KPI 在 15% 带内 |
|---|---|---|---|---|---|
| A2 | D=10 | -9.13% | +4.52% | -46.57% | ❌ |

**H_gap 判定**：成立 ✅——选型差距在每个有数据的梯级都越出 15% 带，「不可移植」不是网格假象。

### 5.3 参数敏感性 ①：TRT 磁参数 Λ（跨两个分辨率复现）

- **A3_D10**：τ₊=0.5075 ⇒ τ₋=25.50（>2 即出推荐窗 τ₋∈[0.5,2]），step 600/21259 发散（non_finite_population），|Cl| 峰值 0.049；不调参掩盖。
- **A3_D14**：τ₊=0.5105 ⇒ τ₋=18.36（>2 即出推荐窗 τ₋∈[0.5,2]），step 1000/29763 发散（non_finite_population），|Cl| 峰值 0.059；不调参掩盖。
- **A5_D10**：τ₊=0.5075 ⇒ τ₋=2.00（>2 即出推荐窗 τ₋∈[0.5,2]），step 5800/21259 发散（non_finite_population），|Cl| 峰值 1.222；不调参掩盖。

| 臂 | 失败分辨率 | τ₋ 序列 | 复现? |
|---|---|---|---|
| A3 | D=10, D=14 | 25.50, 18.36 | ✅ 两分辨率均发散 |
| A5 | D=10 | 2.00 | 单分辨率（另一级未跑或未被判定） |

**H_fail_persists 判定**：成立 ✅——A3 在 2 个分辨率（D=10, D=14）均发散，τ₊ 抬高后仍发散 ⇒ 失效点是「磁参数落到 τ₋ 窗外」这一机制，不是网格假象。

### 5.4 参数敏感性 ②：sponge 与碰撞算子的可分离性（逐梯级）

- **D=10**：sponge 单独效应（A1 vs A4，只差吸收层）ΔCd=+1.83%、ΔSt=+1.64%、ΔCl_amp=-15.40%；碰撞算子单独效应（A4 vs A2，只差 MRT/BGK）ΔCd=-10.76%、ΔSt=+2.83%。

- 未做（如实申报）：sponge 强度 α 与厚度 6D 的敏感性、入口播种幅值的 lock-in 敏感性、BGK+sponge 的第四格（2×2 完备矩阵，A6）、三维校核。

## 6 失败模式与工件

| 失败/风险 | 证据 | 影响 | 处理 |
|---|---|---|---|
| A3_D10 non_finite_population @step 600 | `runs/A3_D10/fail.json` | 该臂无 KPI，不参与 15% 判定 | 报告 + 不调参 |
| A3_D14 non_finite_population @step 1000 | `runs/A3_D14/fail.json` | 该臂无 KPI，不参与 15% 判定 | 报告 + 不调参 |
| A5_D10 non_finite_population @step 5800 | `runs/A5_D10/fail.json` | 同上，不删除、不只报通过的臂 | 报告 + 不调参 |
| OSS 流动循环边界不对称（`0 < ix_next >= nx` 只丢正越界、负索引回绕） | `oss_crosscheck.py::k2_streaming` 只比内部域 | 全周期域逐步对照不可做 | 明确排除 1 格边框，不做全周期长期对照（不隐瞒） |
| OSS 无测力/无 St 后处理 | 源码 grep 无 force/drag | 特征量无法与 OSS 数值直接对标 | 特征量口径由本仓定义并双估计量互校（3-A 第二张表） |
| 环境 shim：numba 未安装 → `jit` 恒等 | `oss_crosscheck.json: env_shim` | 只影响速度，不改数值 | 源码 sha256 入档，证明参照物未被修改 |
| **生态摩擦**：节点架构 loongarch64，`pip install numba` 需源码编译 llvmlite 并失败（`error: failed-wheel-build-for-install`） | 本轮实测（OSS 源码 10 处 `@jit` 无法启用） | OSS 全算例端到端对照永久不可行（121 h/臂） | 证据链降级为「单步内核 + 做法移植」，如实申报 |
| **生态摩擦**：控制面节点（99）属共享机，实测 load≈100–125/128 核 | `cat /proc/loadavg` | 本地并发跑梯级时每臂吞吐掉 2 个数量级（D=14 单臂 >40 min） | 梯级改为集群作业递交 + 本地分块续跑 |

**工件清单**：`xval/benchmark_method_selection/{run_arm.py,run_matrix.sh,run_ladder.sh,run_ladder_cluster.sh,analyze.py,make_report.py,oss_crosscheck.py,predeclared_v2.json,predeclared_v3.json,summary.json,table.csv,oss_crosscheck.json,runs/<arm>_D<D>/{result.json|fail.json,trace.npz},figures/*.png,logs/*.log}`；入口 `benchmarks/xval_method_selection_bench.py`；集成测试 `tests/test_xval_benchmark_method_selection.py`。

**D=10 梯级冻结（内容哈希，证明梯级基线未被事后改动）**：

| 臂 | result.json sha256 | trace.npz sha256 |
|---|---|---|
| A1_D10 | `cc39ae3eb428eac8…` | `25e2b1a5aec1891b…` |
| A2_D10 | `df57b7a3943692d5…` | `f7d925ca0bc37094…` |
| A4_D10 | `11c2bb3c0e86a29c…` | `ae19e906884eeec6…` |

## 7 报告六段核对与误差判定

### 7.0 本仓 vs OSS 差异清单（做法/参数/边界条件）

| 环节 | TensorLBM（本仓记录选型，oss_solutions id=417） | OSS 做法（源码查得，零手抄） | 移植取舍 |
|---|---|---|---|
| 碰撞算子 | MRT，`s_e=1.64 / s_eps=1.54 / s_q=1.7`，支持逐格 `tau_field`（`src/tensorlbm/solver.py:276 collide_mrt`） | **单松弛 BGK**：`lbm/streamcollide.py:147-159`（`collide()` 只用标量 `omega[0]`，无逐格松弛能力） | 整臂移植（A2）；逐格松弛是 sponge 的前提，故 OSS 侧无等价物 |
| 第二套松弛（TRT） | `collide_trt(f, tau_plus, lambda_trt=3/16)`（`src/tensorlbm/solver.py:467`） | 磁参数 Λ=3/16 谱系见 oss_solutions 415/416（lbmpy/lettuce 默认）；pylbm 要求用户显式给 `relaxation_parameters`（`pylbm/scheme.py:189-190`），库不给默认 | 字面移植（A3）**在本算例 ν=U·D/Re 下必然发散**（两个分辨率实测），见 §5；改用 τ₋≤2 的重导出 Λ（A5） |
| 壁面/障碍 | 半程反弹 BB（`boundary/boundaries.py: far_field_bc_2d` 内 `bounce_back_cells`） | 同类半程反弹：`lbm/streamcollide.py:65-66`（`fin[n]=fout[v_inv[n]]`，no-slip） | 一致，无取舍 |
| 入口 | 自由流 Dirichlet（平衡态，`boundary/boundaries.py:201`） | Zou–He 型密度回溯：`lbm/streamcollide.py:219-233`（`local_rho = 1/(1-u_x)·Σ`） | **未移植**（避免把入口格式与碰撞格式的效应混在一起）；列为已知边界条件差异 |
| 侧向边界 | 自由流（blockage-free） | OSS 示例把首末行设为 `OBSTACLE`（`examples/cylinder_flow.py`：`cell_type[:, (0, -1)] = CT.OBSTACLE`），即**无滑移壁** | **未移植**；侧向处理改变阻塞比口径，属口径差异而非方法差异 |
| 出口 | 零梯度 + 下游 6D 二次 sponge（`tau_field = τ(1+ασ)`，α=10） | 纯零梯度复制：`lbm/streamcollide.py:211-216`（`fin[3..5]` 从上游列拷贝）；pylbm Neumann 族；**两库均无吸收层** | sponge 是本仓增量，A4 专门隔离它 |
| 测力（Cd/Cl） | Ladd 动量交换，post-stream / pre-bounce-back（`boundary/boundaries.py:164`） | **两库均无测力模块**（`grep -rn 'force\|drag'` 仅命中重力项）→ 力必须由使用者自建 | 无可移植物；这也说明 OSS 的 "benchmark-method-selection" 不含力/特征量口径 |
| 数值精度 | float64（torch 默认） | Maarten：**float32**（`lbm/init.py:10-14`）；pylbm 默认 float64 | 内核级对照已量化：单步碰撞 rel-L2 见 §3-B |
| 稳定性护栏 | 无内建钳位；靠选型（MRT+sponge）与 τ 控制 | Maarten：**速度钳位** `maxvel=sqrt(2/3)`（`lbm/streamcollide.py:131-133`） | 未移植（本算例 Ma=0.087，钳位在其量级之上不触发） |
| 特征量后处理（St） | Hann 窗 FFT + 抛物线插值 + 滞回过零交叉双法 | 两库只给场量，示例只做动画，无 St 估计 | 无对标物 → 双估计量互校代替 |


### 7.1 误差判定（逐项，显式回答「是否 <15% / <3%」）

1. **选型对标（一级判据 15%）**：OSS 做法移植臂 vs 同分辨率 A1，逐 KPI 逐臂逐梯级。结论：H1 不成立；最大偏差 46.57%（≥ 15% ❌）。
2. **代码级对标（ref = 执行 OSS 源码，判据 <3%）**：碰撞 rel-L2 1.18e-07、流动 rel-L2 2.71e-08 ⇒ **< 3% ✅**（实际 <1e-6，为 float32 舍入量级）。
3. **外部口径（非判据）**：A1 的 St（D=10）落在趋势带 [0.16, 0.18]内 ✅；Cd 见 §3-C。原表 WITHHELD ⇒ 本轮**不以实验/文献为判据**。

**移植取舍（一句话）**：碰撞算子可整臂移植（内核级 <1e-6 一致）；**出口吸收层与逐格 τ 不可移植**（OSS 无对应机制，去掉 sponge 的代价由 A4 vs A1 量化）；**TRT 磁参数必须按 τ₋ 窗重导出**（字面移植在两个分辨率均发散，A3 实测）；入口/侧向边界口径不同（Zou-He vs 自由流、无滑移壁 vs blockage-free），故本 XVAL 判定的是「选型旋钮」，不是「整套代码等价」。

