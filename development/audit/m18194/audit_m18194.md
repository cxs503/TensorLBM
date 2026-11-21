# 交差审计报告：m18194 XVAL-LBM-PRESSURE-EVOLUTION-ALLENCAHN-PHASEFIELD

**审计人**: board-audit (m18312)  
**审计日期**: 2026-11-13  
**对象**: m18194（TensorLBM × CFD-GO/TCLB 压力演化+Allen-Cahn 相场交叉对标）  
**原状态**: succeeded（auto-approve dev_acceptance）

---

## ① 取证核验

### 1.1 三件套存在性

| 项目 | 状态 | 路径 |
|---|---|---|
| run_artifacts | ✅ 存在 | `artifacts/xval_m18194/` 下 8 个文件（summary.json, oss_extract.json, fields.npz, 4×fig*.png, figures.json） |
| benchmark ref vs got | ✅ 存在 | summary.json 含 12 个 drop 算例 + 1 个 Taylor-Green 算例，全部带 err_pct |
| repro_cmd | ✅ 可执行 | `scripts/xval_m18194_pf_pressure_evolution.py`（770 行），`python -m pytest tests/test_phasefield_pf_pressure_evolution.py` → 7 passed |

### 1.2 基准判据核验（predeclared.json 阈值 vs summary.json 实测）

| 判据 | 阈值 | 实测 | 判定 |
|---|---|---|---|
| G1 Laplace 最大误差 | ≤15% | 3.13%（ratio=100, R=23） | ✅ PASS |
| G2 黏度恢复误差 | ≤15% | 0.013% | ✅ PASS |
| G3 相质量守恒 | ≤1e-10 | 5.95e-13 | ✅ PASS |
| G4 密度比一致性 | ≤15% | 1.06% | ✅ PASS |

**12 个独立 drop 算例全部 <15%**，最大 3.13%，远优于门槛。

### 1.3 报告六段完整性

| 段 | 状态 |
|---|---|
| ① 案例介绍（物理背景/几何/参数表） | ✅ 完整 |
| ② 方法/流程/关键设置 | ✅ 完整（P1-P14 结构核验 + 物理判据） |
| ③ 计算结果（数字表） | ✅ 完整（12 行对照表 + G2/G3/G4） |
| ④ 结果图 | ✅ 4 张（fig1-fig4，base64 内嵌于 report id=17090） |
| ⑤ 注意点（踩坑/局限） | ✅ 完整（OSS 不可执行、混合差分源项、未覆盖范围） |
| ⑥ 误差判定 | ✅ 完整（逐项列出，显式回答 <3%） |

### 1.4 回归测试

```
tests/test_phasefield_pf_pressure_evolution.py: 7 passed (46.6s)
- test_moment_matrix_and_viscous_modes
- test_gbar_equilibrium_gives_pressure
- test_interface_term_has_no_pressure_source_when_static
- test_mass_conservation_is_exact
- test_single_phase_control_is_quiescent
- test_taylor_green_recovers_viscosity
- test_laplace_law_coarse
```

### 1.5 结构核验（P1-P14）

14 项全部 PASS，最大残差 8.3e-17（P4 velocity），其余多项为 0。OSS 原文 sha256 + 行号已落盘 `oss_extract.json`。

---

## ② 判级（L0-L5）

| 级别 | 标准 | 判定 | 理由 |
|---|---|---|---|
| L0 | 文档成果 | ✅ | 完整报告 + 差异清单 + 预注册 |
| L1 | 数学/单元验证 | ✅ | 7 个单元测试通过 |
| L2 | 独立功能实现 | ✅ | `pf_pressure_evolution.py` 610 行，输入→计算→输出 |
| L3 | 系统集成 | ⚠️ 部分 | 新模块独立可调用，但未接入主求解器管线（无壁面 BC、无动态算例） |
| L4 | 基准验证 | ✅ | 12 个独立算例 + 误差分析 + 可复现驱动 + 预注册阈值 |
| L5 | 科研工程成果 | ❌ | OSS 二进制不可执行（Python 2 + nvcc 缺失），无字节级 TCLB 对比；范围限于周期边界静态液滴 + Taylor-Green |

**最终级别: L4**

---

## ③ 对抗性审查发现

### 3.1 严重性分级

| 严重度 | 发现 | 影响 |
|---|---|---|
| **中** | OSS 二进制不可执行 → "与 TCLB 逐场一致"未被数值证明，仅证明"与解析解一致 + 与 OSS 公式结构等价" | 降低外部锚定强度，但不影响物理有效性 |
| **低** | 混合（迎风）差分权重和仅二阶为零 → f 群有 O(截断) 压力源，ratio=1 算例 6000 步仍 −1.9% 漂移 | OSS 继承缺陷，非移植缺陷；已量化并设回归带 |
| **低** | 壁面/入口出口节点类型未移植 → 无壁面算例（layered Poiseuille、RT、气泡上升）本轮不做 | 范围限制，已在报告中诚实声明 |
| **信息** | tensor_benchmark_cases 表中无此 anchor 的条目 → 判据完全来自 predeclared.json | 基准数据库未覆盖此 XVAL 锚点 |

### 3.2 未运行检查

- [ ] OSS 二进制实际运行对比（环境不可用：Python 2 + nvcc 缺失）
- [ ] 壁面边界条件算例（未移植）
- [ ] 3D（d3q27_pf_velocity）压力处理（范围外）
- [ ] 动态湿润/接触角算例（范围外）

### 3.3 过度声明检查

报告中明确声明了"OSS 侧未能数值执行，故'与 TCLB 逐场一致'未被证明"——**无过度声明**。误差数字全部来自 summary.json 程序化注入，零手抄。

---

## ④ 裁定

**Disposition: APPROVE with notes**

- m18194 交付物完整、数字可信、测试通过、报告六段齐全
- 四门判据全部通过（G1 3.13%, G2 0.013%, G3 5.95e-13, G4 1.06%），远优于 15% 门槛
- 成熟度 L4（基准验证级），未达 L5 的原因已如实记录
- 建议后续：移植壁面 BC、补充动态算例、在 TCLB 可运行环境中做字节级对比

**Verdict 已落表**: `tensor_maturity_verdicts` mission_id=18312, level=4
