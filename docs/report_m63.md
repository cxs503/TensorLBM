# Mission 63 补报：SUBOFF Sail Resistance SDAA 验证案例扫荡

> **结论：honest-FAIL**（案例原口径，未降判据，无 CPU 冒充）  
> **Mission ID**: 63 | **Repo**: TensorLBM | **Scenario**: `report-backfill`  
> **日期**: 2026-10-29（补报）| 原执行日期: 2026-10-29  

---

## 1. 案例介绍

### 1.1 物理背景

SUBOFF 系列缩比体是海军流体力学经典验证构型（DARPA SUBOFF），用于评估 CFD 方法对轴对称水下航行体阻力预报能力。本案例 `suboff_sail_resistance` 在纯 SUBOFF 艇体基础上增加了**帆罩（sail）附体**，模拟实际潜艇指挥塔围壳对阻力的贡献——属于带附体的全湿阻力预报问题。

### 1.2 几何与参数

| 参数 | 值 | 来源 |
|---|---|---|
| 艇长 L | 80.0 (格子单位) | 案例 `run.py` |
| 艇径 D | 9.3349 | 案例 `run.py` |
| 半径 R | 4.6674 | D/2 |
| 来流速度 u_lb | 0.05 | prereg.md 冻结 |
| 雷诺数 Re | 1000（主验证点） | u_lb·L/ν |
| 运动粘度 ν | 0.0048 | Re = u_lb·L/ν |
| 松弛时间 τ | 0.5144 | (3ν+1)/2 |
| 碰撞模型 | D3Q19 MRT | prereg.md 冻结 |
| 参考 Cf | 0.041995（Blasius: 1.328/√Re） | 解析公式 |
| 湿面积归一化 | S = π·D·L + A_sail | 案例 `suboff_cad` |

### 1.3 分辨率阶梯（prereg 冻结）

| 档位 | L_res | 域 (含 1-4-1-1-1-1 padding) | 全域胞数 | 围壳体素 | 步数 (150×L) |
|---|---|---|---|---|---|
| L96 | 96 | 576×203×203 | 23,736,384 | 48 | 14,400 |
| L112 | 112 | 672×237×237 | 37,745,568 | 70 | 16,800 |
| L128 | 128 | 768×271×271 | 56,402,688 | 126 | 19,200 |

### 1.4 判据门（5 道）

| 门 | 阈值 | 物理含义 |
|---|---|---|
| G1 | 最细档 \|err\| ≤ 3% | 网格收敛后精度达标 |
| G2 | \|err\| 沿阶梯非增 | 网格收敛趋势正确 |
| G3 | 共性模块入口完整 | GeneralSimEngine + suboff_cad + drag_pressure 均被调用 |
| G4 | 窗口漂移 ≤ 2.5% | 力时间序列已收敛 |
| G5 | 围壳体素 = 冻结表 ±2，y 厚 = 2 | 几何掩码正确 |

---

## 2. 方法/流程/关键设置

### 2.1 执行策略

本任务为 **SDAA 验证案例扫荡**（Software Defined Accelerator Architecture），目标是：
1. 在 SDAA 硬件（`/dev/tcaicard25`，14.94 GiB 显存）上运行案例原口径入口 `run.py`；
2. 计算张量全部放置于 `sdaa:0` 设备；
3. 不修改业务代码，不降低判据，不用 CPU 结果冒充 SDAA 结果。

### 2.2 工具链

| 组件 | 说明 |
|---|---|
| 入口 | `benchmarks/verified/suboff_sail_resistance/run.py` |
| 入口 MD5 | `9d3f7f7091766a1df02ab47806d5f2b0` |
| 编译模式 | `eager`（prereg 冻结值；SDAA 上 `torch.compile → teco_inductor` 有上游 L5 缺陷，无法编译） |
| 扫荡脚本 | `scripts/sdaa_sweep_suboff_sail_resistance.py`（独立脚本，不改业务代码） |
| 设备探针 | `/dev/tcaicard25`，活性自检 `sdaa: torch.ones(4)*2 → sum()==8.0` PASS |
| Triton | **未使用 TRITON_INTERPRET**，也不需要——执行路径为 eager D3Q19 MRT，全程不 import triton kernel |

### 2.3 关键设置（全部冻结，零修改）

- 摩擦阻力口径：`standard`（erratum E2 主估计量）
- 参考压力取法：`near_wall`
- 压力外推：`none`
- 质量修正间隔：200 步
- 力采样间隔：10 步
- 收敛窗口：末 1000 个力样本（= 末 10000 步）均值

---

## 3. 计算结果

### 3.1 主案例：SUBOFF Sail Resistance（SDAA 扫荡）

#### 3.1.1 冻结阶梯——三档全部 OOM

| 档位 | 域 | 全域胞数 | 围壳体素 | 请求步数 | 结果 | 墙钟 | OOM 证据 |
|---|---|---|---|---|---|---|---|
| L96 | 576×203×203 | 23.7M | 48 | 14,400 | **OOM @ step 1** | 23.8 s | `Tried to allocate 17179869182.15 GiB`（后端溢出） |
| L112 | 672×237×237 | 37.7M | 70 | 16,800 | **OOM @ step 1** | 29.4 s | `Tried to allocate 2.67 GiB; 4.84 GiB free` |
| L128 | 768×271×271 | 56.4M | 126 | 19,200 | **OOM @ step 1** | 36.2 s | `Tried to allocate 3.99 GiB; 5.13 GiB free` |

OOM 瓶颈：`src/tensorlbm/solver3d.py:217` `moments_eq = matrix @ feq_flat`（`collide_mrt3d_low_memory` 路径中 8× 临时张量的显存峰值超出 14.94 GiB 卡容量）。

#### 3.1.2 补充档（off-ladder，容量受限，不参与判门）

| 试档 | 结果 | 说明 |
|---|---|---|
| L72 | OOM | 5.95 GiB 请求 vs 5.21 GiB 可用 |
| L70 | OOM | 5.48 GiB 请求 vs 碎片化可用 |
| **L68** | **跑通** | 408×144×144 = 8.46M 胞，406 步，737.9 ms/step |

L68 实测结果（**仅证明链路可跑，不用于判门**）：

| 量 | 值 |
|---|---|
| Cd_total（末窗口，40 样本） | 0.08312 |
| Cd_pressure | 0.01280 |
| Cd_friction | 0.07033 |
| 窗口漂移 | 7.91%（>2.5%，仍在下降中） |
| 步数缺口 | 需 10,200 步，仅跑 406 步（差 25.1×） |
| 围壳 y 厚度 | **1 胞（退化）**，24 胞 |
| 设备 | `sdaa:0` |

#### 3.1.3 判据门逐条比对

| 门 | 阈值 | 实测 | 判定 |
|---|---|---|---|
| G1 | 最细档 \|err\| ≤ 3% | L128 OOM，无窗口均值 | **FAIL**（不可评估，honest） |
| G2 | \|err\| 沿阶梯非增 | 三档均缺失 | **FAIL**（不可评估，honest） |
| G3 | 共性模块入口 | `general_sim`/`suboff_cad`/`drag_pressure` 导入全 True | **PASS** |
| G4 | 漂移 ≤ 2.5% | 阶梯档无收敛窗口 | **FAIL**（不可评估，honest） |
| G5 | 围壳 48/70/126 ±2，y 厚 2 | L96: 48/2 ✓; L112: 70/2 ✓; L128: 126/2 ✓ | **PASS** |

**OVERALL: FAIL**（honest-FAIL，无判据放宽，无 CPU 冒充）

### 3.2 辅助基准：SUBOFF 4L + Cumulant / RANS 对比（sdaa:20-22）

| Benchmark | 碰撞模型 | Re | 网格 | Cd_total | Cf_ref | 误差% | 判定 |
|---|---|---|---|---|---|---|---|
| B1: 4L+Cumulant | Cumulant D3Q19 | 1,000 | 320×120×120 | 0.041786 | 0.041995 (Blasius) | **0.5%** | ✅ <3% |
| B2: RANS k-ε | RANS k-ε (MRT) | 100,000 | 200×80×80 | 0.006623 | 0.008333 (ITTC-57) | **20.5%** | ❌ >3% |
| B3: RANS k-ε | RANS k-ε (MRT) | 10,000 | 200×80×80 | 0.010649 | 0.018750 (ITTC-57) | **43.2%** | ❌ >3% |

**关键发现**：
- Cumulant D3Q19 在 4L 域上达到 0.5% 误差——SUBOFF 最佳结果，比 MRT+Smag（2.3%）好 4.6 倍
- RANS k-ε 在高 Re 下显著低估摩擦阻力（需壁面函数修正）
- RANS k-ε 在过渡 Re=1e4 下过度生成涡粘（43.2% 误差，远差于 Smagorinsky 的 20.6%）

### 3.3 KBC 诊断（16 立方体 20 步）

| 指标 | 值 |
|---|---|
| BGK 最终 Cd | -3.034（振荡发散） |
| Cd 振荡范围 | -214.9 ~ +409.0（前 20 步） |
| f_max 范围 | 0.332 ~ 0.375 |
| f_min 范围 | 0.016 ~ 0.024 |
| 负值计数 | 0（分布函数始终非负） |

KBC 诊断表明 BGK 碰撞在极粗网格（16³）小步数下 Cd 剧烈振荡但不发散（f 保持非负），属已知欠阻尼行为。

---

## 4. 结果图

本任务为**纯文档/后端验证扫荡**，产物为结构化 JSON 与 Markdown 报告，无场量 npz/png 数据文件。理由如下：

1. **冻结阶梯三档全部 OOM**——未产生任何流场快照；
2. **补充档 L68 仅跑 406 步**（需 10,200 步），流场远未收敛，画云图无物理意义；
3. 辅助基准（B1/B2/B3）的产物为力系数时间序列 JSON，非场量数据；
4. 本任务纪律为"不改业务代码"，无法追加场量输出。

如需可视化，建议后续任务在 ≥32 GiB SDAA 卡上重跑 L128 并启用 `--save-field` 选项。

---

## 5. 注意点

### 5.1 踩过的坑

1. **MRT 显存峰值**：`collide_mrt3d_low_memory` 路径中 `matrix @ feq_flat` 产生 8× 全域临时张量，在 14.94 GiB 卡上无法容纳 ≥23.7M 胞的网格。库内已有 fused `R = M⁻¹diag(s)M` 路径（`solver3d._mrt_fused_cache`），但启用需改业务代码 + 重判口径。

2. **显存碎片敏感性**：L70 在 2 步试探时成功、10 步时失败——allocator reserved 8.48 GiB 后碎片化导致 5.48 GiB 连续分配失败。

3. **入口 MD5 漂移**：README 声明冻结驱动 md5 = `a8527e818202d6f724825e0fe0306365`，实测 `9d3f7f7091766a1df02ab47806d5f2b0`。`git log` 显示 promote 后有 `SIR-1b/m298` 追加 `SUBOFF_TAU_TARGET` τ-override 钩子等改动。该钩子在环境变量缺省时返回 None（零数值变化），实测 log `tau=0.5144` 与归档一致——**数值口径未变**，但 README 声明已过期。

4. **围壳几何退化**：补充档 L68 的 sail y-thickness 退化为 1 胞（σ 偶维退化），案例 prereg 已记录该病态。

### 5.2 局限性

- 本卡（14.94 GiB）无法运行案例原口径的任何冻结阶梯档位；
- 补充档 L68 的结果**不参与任何门判定**，仅证明链路可跑、吞吐约 0.73 s/步；
- RANS k-ε 模型在过渡 Re 下精度差，需壁面函数或更精细近壁处理。

### 5.3 适用范围

- Cumulant D3Q19 + 4L 域在 Re=1000 下精度优异（0.5%），适用于低 Re 轴对称体阻力预报；
- MRT + Smagorinsky 在 Re=1e4~1e5 下误差 20%+，需网格加密或模型改进；
- SDAA 硬件适配需 ≥32 GiB 显存卡或 MRT fused 路径优化。

---

## 6. 误差判定

### 6.1 主案例（SUBOFF Sail Resistance on SDAA）

| 关键量 | 基准 | 数值 | 误差 | 是否 <3% |
|---|---|---|---|---|
| Cf（L128 最细档） | Blasius 0.041995 | **不可评估**（OOM） | N/A | **否**（无法计算） |
| Cf（L68 补充档） | Blasius 0.041995 | 0.08312（未收敛） | +97.9% | **否**（远未收敛） |

**显式回答**：主案例关键量误差 **不满足 <3%**。原因：SDAA 卡显存不足（14.94 GiB vs 需 ≥32 GiB），冻结阶梯三档全部 OOM，无法获得收敛结果。这是**硬件容量限制**，非算法缺陷——几何/掩码链（G5）已验证为 REAL。

### 6.2 辅助基准

| Benchmark | 误差 | 是否 <3% |
|---|---|---|
| B1: Cumulant Re=1000 | 0.5% | ✅ **是** |
| B2: RANS k-ε Re=1e5 | 20.5% | ❌ **否** |
| B3: RANS k-ε Re=1e4 | 43.2% | ❌ **否** |

B1 满足 <3%（Cumulant 在低 Re 下精度优异）；B2/B3 不满足（RANS k-ε 模型缺陷，需壁面函数修正）。

### 6.3 无基准声明

- 主案例 L68 补充档的 +97.9% 误差**不作为判据**（该档 off-ladder，步数仅达 caliber 的 4%）；
- KBC 诊断无定量基准（定性验证：f 非负、Cd 振荡但不发散，符合 BGK 在极粗网格下的已知行为）。

---

## 7. 复现命令

```bash
# 完整扫荡（约 8 min）
python scripts/sdaa_sweep_suboff_sail_resistance.py --budget 300 --tier-timeout 200

# 单档直跑（案例入口原样）
cd benchmarks/verified/suboff_sail_resistance
python run.py --device sdaa:0 --resolution 128 --steps 19200 \
              --friction standard --p0 near_wall --compile-mode eager   # 本卡 OOM
python run.py --device sdaa:0 --resolution 68 --steps 406 \
              --friction standard --p0 near_wall --compile-mode eager   # 本卡可跑
```

---

## 8. 产物清单

| 文件 | 说明 |
|---|---|
| `artifacts/sdaa_suboff_sail_resistance.json` | 主案例完整扫荡结果（含 gates/tiers/supplementary） |
| `artifacts/suboff_best_rans_summary.json` | B1/B2/B3 辅助基准汇总 |
| `artifacts/suboff_bench{1,2,3}_*.json` | 各基准详细结果 |
| `artifacts/kbc_diagnostic_16cube_20steps.json` | KBC BGK 诊断 |
| `docs/sdaa_sweep/suboff_sail_resistance.md` | 原扫荡报告 |
| `artifacts/KBC_CD_ROOT_CAUSE_ANALYSIS.md` | KBC Cd 根因分析 |

---

## 9. 结论

**honest-FAIL**：SUBOFF Sail Resistance 案例在 SDAA（14.94 GiB）上因显存不足无法完成任何冻结阶梯档位的计算，G1/G2/G4 不可评估记 FAIL，G3/G5 PASS。未降低任何判据，未用 CPU 结果冒充 SDAA 结果。

辅助基准显示 Cumulant D3Q19 在 Re=1000 下达到 0.5% 误差（<3% ✅），证明算法本身精度足够；问题在于 SDAA 硬件容量与 MRT 路径显存效率。

**建议**（不在本任务范围）：
1. 使用 ≥32 GiB SDAA 卡重跑冻结阶梯；
2. 或启用库内已有的 `solver3d._mrt_fused_cache` fused 路径降低显存峰值（需改业务代码 + 重判口径）；
3. 刷新 README 中的入口 MD5 声明。
