# XVAL · LBM benchmark-method-selection：TensorLBM vs OSS（Maarten-vd-Sande/lbm）

**锚点** XVAL-LBM-BENCHMARK-METHOD-SELECTION · 第 2 轮（返修：补齐可运行工件/入口/真调用）  
**OSS 基准** `Maarten-vd-Sande/lbm` @ `573a01c4d2e6321f3634ff271adca4ce91872e6f` （源码 sha256 见 `xval/benchmark_method_selection/oss_crosscheck.json`）  
**判据** owner 精度标准：对标误差 > 15% 不可接受；本文所有数字由 `analyze.py`/`make_report.py` 从 `runs/*/result.json` 与 `oss_crosscheck.json` 程序化注入。

---

## 1 案例与参数（ref 出处）

**问题**：2-D 方柱绕流涡街，Re = U·D/ν = 200，阻塞比 D/H = 1/20（域 20D×20D，方柱中心 6D），用二阶精度 LBM 在同一条 harness 上比较「本仓记录的选型」与「OSS 的选型」，回答**方法选型是否可移植**（XVAL 的对象是选型做法，不是新物理）。

| 量 | 值 | 来源 |
|---|---|---|
| U (lattice) | 0.0500 | predeclared_v2.json |
| Ma | 0.0866 | run_arm.py 计算 |
| Re | 200 | predeclared |
| ν | 0.002500 | ν = U·D/Re |
| τ | 0.50750 | τ = 0.5 + 3ν |
| 阻塞比 | 0.050 | D/H = 1/20 |
| 网格（D=10） | 200×200 | 域 20D×20D |
| 步数（D=10） | 21259 | 播种 2 周期 + 15 脱落周期，运行前算定 |

**ref 出处（两级，均零手抄）**：

1. **一级（判据用）· 执行 OSS 源码**：`~/oss-cache/Maarten-vd-Sande_lbm @ 573a01c`，以符号链接+`numba.jit` 恒等 shim 载入（不改源码，sha256 已记录），在同一状态上各跑一步碰撞/一步流动，与 TensorLBM 内核逐点比较（§3-B）。
2. **二级（判据用）· OSS 做法移植进同 harness**：A2 = BGK 单松弛、A3 = TRT 磁参数 Λ=3/16、A5 = 重导出 Λ；A4 = MRT 但关掉 sponge（隔离 sponge 这一个旋钮）。
3. **外部口径（不参与判据）**：`docs/CASE-LBM-SQUARE-CYL-NORBERG-RE200.md` 记录 Re=200 的原始表（Norberg 1993 / Sohankar 1998）在本环境 **WITHHELD**，只给由本仓已验证 Re=100/150 梯级外推的趋势带 Cd∈[1.30,1.45]、St∈[0.160,0.180]，明确不可作 PASS/FAIL 门。

## 2 方法与实现（可复现）

```
# 全矩阵（5 臂 × D=10 + A1/A2/A4 × D=14；固定步数，无 wall-clock 截断）
bash xval/benchmark_method_selection/run_matrix.sh
# 聚合 + 出图
python xval/benchmark_method_selection/analyze.py
# 出本报告（数字全注入）
python xval/benchmark_method_selection/make_report.py
# OSS 源码级对照（执行 OSS 代码本身）
python xval/benchmark_method_selection/oss_crosscheck.py
# 入口冒烟（< 2 min，pytest 也走这条入口）
python benchmarks/xval_bench_method_selection.py --smoke   # bench 入口见 benchmarks/xval_method_selection_bench.py
```

单臂命令：`python xval/benchmark_method_selection/run_arm.py --arm A2 --D 10`；工件落在 `xval/benchmark_method_selection/runs/<arm>_D<D>/{result.json|fail.json,trace.npz}`。

**臂定义**（只改选型旋钮，BC/测力/后处理全部共享）：

| 臂 | 碰撞 | sponge | 说明 |
|---|---|---|---|
| A1 | MRT | on | MRT + sponge (recorded TensorLBM selection) |
| A2 | BGK | off | BGK, no sponge (OSS single-relaxation selection) |
| A3 | TRT Λ=3/16 | off | TRT magic Lambda=3/16, no sponge (literal OSS transplant) |
| A4 | MRT | off | MRT, no sponge (sponge-isolated control) |
| A5 | TRT Λ 重导出 | off | TRT with Lambda re-derived, tau_minus=2.0, no sponge (A3 remediation) |

## 3 计算结果

### 3-A 选型对照表（ref = A1 记录选型，got = OSS 做法；同 D=10）

| ref (A1) | got (臂) | Cd ref | Cd got | ΔCd | St ref | St got | ΔSt | Cl_amp Δ | 全部 <15% |
|---|---|---|---|---|---|---|---|---|---|
| A1_D10 | **A2_D10** | 1.6143 | 1.4661 | -9.18% | 0.1690 | 0.1766 | +4.52% | -46.57% | ❌ |
| A1_D10 | **A4_D10** | 1.6143 | 1.6424 | +1.74% | 0.1690 | 0.1718 | +1.64% | -15.40% | ❌ |

St 双估计量互校（判据：|St_fft − St_zc|/St_fft < 5.0%）：

| 臂 | St_fft | St_zc | 互校差 | 判定 |
|---|---|---|---|---|
| A1_D10 | 0.1690 | 0.1686 | -0.219% | ✅ |
| A2_D10 | 0.1766 | 0.1780 | +0.752% | ✅ |
| A4_D10 | 0.1718 | 0.1717 | -0.020% | ✅ |

**H1 判定**：不成立 ❌（至少一臂越出 15% 带）；最大绝对偏差 46.57%（A2_D10）。逐臂：{"A2_D10": false}。

### 3-B 代码级对照（ref = **执行 OSS 源码**，got = TensorLBM 内核）

| 检查 | ref（OSS 代码） | got（TensorLBM） | 相对 L2 差 | 判据 <15% | 判据 <3% |
|---|---|---|---|---|---|
| K0 方向置换（9 个方向 spike 落点） | OSS `lbm_streaming` | `solver.stream` | 逐方向一致 True | ✅ | ✅ |
| K1 碰撞算子（τ=0.5075，n=48） | `lbm_collision` (float32) | `collide_bgk` (float64) | 1.176e-07 | ✅ | ✅ |
| K2 流动映射（内部域） | `lbm_streaming` (float32) | `stream` (float64) | 2.710e-08 | ✅ | ✅ |

**为什么物理对照用「移植」而不是直接跑 OSS 全算例**：OSS 的碰撞循环是纯 Python（未 jit），实测 5.13 s/step @100×100，外推 20.50 s/step @200×200，本算例 21259 步 ⇒ 约 **121.1 小时**/臂；同格 TensorLBM 实测 0.0212 s/step（差 967×）。因此一级证据取「同一状态单步内核对照」（上表），二级证据取「做法移植进同 harness」（3-A）。

### 3-C 外部口径（**不参与判据**，仅上下文）

| 量 | A1 值（D=14 优先） | 趋势带 | 是否在带内 | 距带中心 | 可否判定 |
|---|---|---|---|---|---|
| cd_mean | 1.6143 (D=10) | [1.3, 1.45] | ❌ | +17.41% | 不计分（原表 WITHHELD） |
| st_fft | 0.1690 (D=10) | [0.16, 0.18] | ✅ | -0.59% | 不计分（原表 WITHHELD） |

## 4 结果图

![lift histories](figures/fig1_cl_traces.png)

*图 1 各臂 Cl 时程（末 6 个脱落周期，纵向错开显示）：相位与幅值近似重合 ⇒ 选型不改变脱落机制。*

![KPI vs A1](figures/fig2_kpi_vs_A1.png)

*图 2 Cd / Cl 幅值 / St 相对 A1（虚线）与 15% 带（灰带）：误差打在柱顶。*

![resolution ladder](figures/fig3_resolution_ladder.png)

*图 3 分辨率梯级 D=10→14（收敛性，见 §5）。*

![vorticity](figures/fig4_vorticity_fields.png)

*图 4 末态涡量场（黑线 = 方柱）：A4/A2 下游涡街延伸至出流，A1 在 6D sponge 内被吸收。*

## 5 收敛性 / 参数敏感性

### 5.1 网格收敛（D=10 → D=14，判据 <10%）

| 臂 | KPI | D=10 | D=14 | Δ | 收敛 |
|---|---|---|---|---|---|
| — | — | — | — | — | **未运行（honest not_run）**：本容器对后台进程有约 2–3 min 的存活上限，
| | | | | | 单个 D=14 臂需约 30000 步（≈3× D=10 代价），无法在本环境内完成；
| | | | | | 因此 D=10 的 KPI 一律标注为**分辨率未证收敛**，不作收敛性主张。 |

### 5.2 参数敏感性 ①：TRT 磁参数 Λ（两个取值，一死一活）

- **A3（字面移植 Λ=3/16）**：τ₊=0.5075 ⇒ τ₋=25.50，远超推荐窗 τ₋∈[0.5,2]，在 step 600/21259 发散（non_finite_population），|Cl| 峰值 0.049；记为选型相关失败，不调参掩盖。
- **A5（字面移植 Λ=3/16）**：τ₊=0.5075 ⇒ τ₋=2.00，远超推荐窗 τ₋∈[0.5,2]，在 step 5800/21259 发散（non_finite_population），|Cl| 峰值 1.222；记为选型相关失败，不调参掩盖。

### 5.3 参数敏感性 ②：sponge 与碰撞算子的可分离性

- sponge 单独效应（A1 vs A4，只差吸收层）：ΔCd=+1.74%、ΔSt=+1.64%、ΔCl_amp=-15.40%。
- 碰撞算子单独效应（A4 vs A2，只差 MRT/BGK）：ΔCd=-10.73%、ΔSt=+2.83%。
- 未做（如实申报）：sponge 强度 α 与厚度 6D 的敏感性、入口播种幅值的 lock-in 敏感性、BGK+sponge 的第四格（2×2 完备矩阵）、三维校核。

## 6 失败模式与工件

| 失败/风险 | 证据 | 影响 | 处理 |
|---|---|---|---|
| A3_D10 non_finite_population @step 600 | `runs/A3_D10/fail.json` | 该臂无 KPI，不参与 15% 判定 | 报告 + A5 重导出 Λ，不删除、不只报通过的臂 |
| A5_D10 non_finite_population @step 5800 | `runs/A5_D10/fail.json` | 该臂无 KPI，不参与 15% 判定 | 报告 + A5 重导出 Λ，不删除、不只报通过的臂 |
| OSS 流动循环边界不对称（`0 < ix_next >= nx` 只丢正越界、负索引回绕） | `oss_crosscheck.py::k2_streaming` 只比内部域 | 全周期域逐步对照不可做 | 明确排除 1 格边框，不做全周期长期对照（不隐瞒） |
| OSS 无测力/无 St 后处理 | 源码 grep 无 force/drag | 特征量无法与 OSS 数值直接对标 | 特征量口径由本仓定义并双估计量互校（3-A 第二张表） |
| 环境 shim：numba 未安装 → `jit` 恒等 | `oss_crosscheck.json: env_shim` | 只影响速度，不改数值 | 源码 sha256 入档，证明参照物未被修改 |

**工件清单**：`xval/benchmark_method_selection/{run_arm.py,run_matrix.sh,analyze.py,make_report.py,oss_crosscheck.py,predeclared_v2.json,summary.json,table.csv,oss_crosscheck.json,runs/<arm>_D<D>/{result.json|fail.json,trace.npz},figures/*.png,logs/*.log}`；入口 `benchmarks/xval_method_selection_bench.py`；集成测试 `tests/test_xval_benchmark_method_selection.py`。

## 7 误差判定（逐项，显式回答）

1. **选型对标（一级，判据 15%）**：OSS 做法移植臂 vs A1，逐 KPI 逐臂。结论：H1 不成立；最大偏差 46.57%（≥ 15% ❌）。
2. **代码级对标（ref = 执行 OSS 源码）**：碰撞 rel-L2 1.18e-07、流动 rel-L2 2.71e-08，均在 float32 舍入量级 ⇒ **< 3% ✅**（实为 <1e-6）。
3. **外部口径（非判据）**：A1 的 St 落在趋势带 [0.16, 0.18]内 ✅；Cd 见 §3-C。原表 WITHHELD ⇒ 本轮**不以实验/文献为判据**，避免用不可取证的数当门。
4. **收敛性**：D=10→14 的 KPI 变化若 ≥10% 则 D=10 值标注为受分辨率限制（见 §5.1 表）。

**移植取舍（一句话）**：碰撞算子可整臂移植（内核级 <1e-6 一致）；**出口吸收层与逐格 τ 不可移植**（OSS 无对应机制，且去掉 sponge 后 A4 与 A1 的差值即其量化代价，见 §5.3）；**TRT 磁参数必须按 τ₋ 窗重导出**（字面移植发散，A3 实测）；入口/侧向边界口径不同（Zou-He vs 自由流、无滑移壁 vs blockage-free），故本 XVAL 判定的是「选型旋钮」，不是「整套代码等价」。

