# Mission 21 补报：SDAA Sweep — droplet_oscillation

**仓库**: TensorLBM  
**案例**: `benchmarks/verified/droplet_oscillation`  
**任务类型**: SDAA 设备验证案例扫荡（backfill report）  
**结论**: **honest-FAIL** — SDAA 扫描已启动但未在预算内完成，所有判据均为 PENDING  

---

## 1. 目的

TensorLBM 的 `droplet_oscillation` 是一个经过严格预注册的验证基准（W2-B 标准），用于检验 Shan-Chen 单组分赝势 LBM（D2Q9 SCMP BGK）中 m=2 毛细模态液滴振荡频率的精度。本任务的目标是：

1. 将该基准从 NVIDIA CUDA（RTX 5090）移植到 SDAA 设备（/dev/tcaicard1）运行
2. 编写独立运行脚本 `scripts/sdaa_sweep_droplet_oscillation.py`，不改业务代码
3. 在 SDAA 预算（~20 分钟）内完成多半径扫描并验证预注册判据
4. 确认 SDAA 设备上的物理精度与 CUDA 参考一致

---

## 2. 方法

### 2.1 基准物理模型

| 参数 | 值 |
|------|-----|
| 模型 | SC94 单组分赝势 LBM (D2Q9 BGK) |
| 状态方程 | ψ = 1 − exp(−ρ) |
| 相互作用强度 | G_eff = −5.0（物理），G_lib = +5.0（库符号约定） |
| 松弛时间 | τ = 1.0 |
| 精度 | float32 |
| 域 | L = 4R，周期边界 |
| 初始条件 | 椭圆 tanh 液滴 R(θ) = R₀(1 + 0.05·cos2θ)，W = 4 |

### 2.2 预注册判据（5 项）

1. **|err| ≤ 3%** 所有半径级别（basis C 主判据）
2. **|err| 随 R 单调递减**（网格收敛性）
3. **每级别模拟 ≥ 4 个理论周期**
4. **拟合 σ_ω/ω_d < 1%**（统计不确定度）
5. **质量漂移 < 2×10⁻³**

### 2.3 SDAA 扫描配置

| 配置项 | 原始 (CUDA) | SDAA 扫描 |
|--------|-------------|-----------|
| 半径 | [128, 160, 224] | [64, 80, 96] |
| 运行时间 | ~1 小时 | ~10-20 分钟（预估） |
| 设备 | RTX 5090 (cuda:0) | /dev/tcaicard1 (sdaa:0) |
| 判据 | 相同 | 相同 |
| 模型 | 相同 | 相同 |

**关键决策**：
- 半径从 [128,160,224] 缩减至 [64,80,96]，以适应 20 分钟 SDAA 预算
- R=32 因液滴过小导致 NaN，从 R=64 开始
- 无 Triton 依赖，纯 PyTorch 操作，无需 TRITON_INTERPRET 变通

### 2.4 脚本实现

- **主脚本**: `scripts/sdaa_sweep_droplet_oscillation.py`
- **最小验证脚本**: `scripts/sdaa_sweep_droplet_oscillation_minimal.py`
- 脚本通过 `sys.path.insert` 导入原始 `run.py`，确保物理模型完全一致
- 设备证据通过 `torch.sdaa.get_device_name(0)` 确认

---

## 3. 关键结果

### 3.1 SDAA 扫描状态

| 指标 | 值 |
|------|-----|
| 状态 | **IN_PROGRESS** |
| 设备 | sdaa:0 (/dev/tcaicard1) |
| 设备确认 | ✅ torch.sdaa.get_device_name(0) = '/dev/tcaicard1' |
| 脚本启动 | ✅ 成功启动并在 SDAA 上执行 |
| 半径 | [64, 80, 96] |
| 结果填充 | ❌ 空（results = {}） |
| 判据评估 | ❌ 全部 PENDING |
| 总运行时间 | null（未完成） |

### 3.2 判据验证状态

| 判据 | 状态 |
|------|------|
| err_le_3pct_all_levels | PENDING |
| err_monotone_decreasing | PENDING |
| simulated_ge_4_periods_all_levels | PENDING |
| sigma_omega_rel_lt_1pct_all_levels | PENDING |
| mass_drift_lt_2e3_all_levels | PENDING |
| **all_criteria_pass** | **null** |

### 3.3 原始 CUDA 参考结果（result.json）

原始基准在 CUDA (RTX 5090) 上**完全通过**所有判据：

| 半径 R | err_pct (basis C) | err_pct (basis A) | 单调性 | 通过 |
|--------|-------------------|-------------------|--------|------|
| 128 | −2.649% | −1.321% | ✅ | ✅ |
| 160 | −2.404% | −1.001% | ✅ | ✅ |
| 224 | −2.005% | −0.511% | ✅ | ✅ |

- **all_within_3pct**: true
- **monotone_convergence**: true
- **fit_quality_ok**: true
- **pass**: true

### 3.4 其他 Mission 21 工件摘要

| 工件 | 关键指标 | 结论 |
|------|----------|------|
| suboff_bench1_cumulant_sdaa20 | Cf error = 0.50% | ✅ PASS |
| suboff_bench2_rans_re1e5_sdaa21 | Cf error = 20.53% | ❌ FAIL (>15%) |
| suboff_bench3_rans_re1e4_sdaa22 | Cf error = 43.21% | ❌ FAIL (>15%) |
| kbc_diagnostic_16cube_20steps | H-theorem violation, Cd ratio = −7.27 | ⚠️ 诊断发现 |

---

## 4. 审计与发现

### 4.1 设备兼容性

- ✅ SDAA 设备驱动正常，`torch.sdaa` API 可用
- ✅ 脚本成功在 SDAA 上启动并执行
- ✅ 无 Triton 依赖，纯 PyTorch 路径可行

### 4.2 预算与性能

- ⚠️ 原始基准在 RTX 5090 上需 ~1 小时
- ⚠️ SDAA 预算 ~20 分钟，半径缩减至 [64,80,96]
- ❌ 即使缩减半径，模拟仍在超时前未完成
- **根因**: 运行时间 ∝ R² × steps ≈ O(R³)，R=64 的 static + oscillation 仍需较长时间

### 4.3 物理一致性

- 原始 CUDA 结果验证了物理模型的正确性（所有判据通过）
- SDAA 应产生相同物理结果（相同模型、相同代码路径）
- 但由于未完成运行，无法程序化验证 SDAA 数值一致性

### 4.4 工程实现

- ✅ 脚本设计合理：导入原始 `run.py`，零业务代码修改
- ✅ 设备证据采集完整
- ✅ 判据框架已搭建，等待结果填充
- ⚠️ 缺少运行日志的持久化（仅 `/tmp/sdaa_droplet_minimal.log`）

---

## 5. 局限与后续

### 5.1 本任务局限

1. **未完成**: SDAA 扫描未能在预算内完成，所有判据均为 PENDING
2. **无法验证**: 无法确认 SDAA 设备上的数值精度是否与 CUDA 一致
3. **预算不足**: 20 分钟不足以完成即使缩减半径的完整扫描
4. **日志缺失**: 运行日志在 /tmp，未持久化到工件目录

### 5.2 后续建议

1. **增加预算**: 为 droplet_oscillation SDAA 扫描分配 ≥1 小时
2. **分步执行**: 先完成 R=64 的 static Laplace + oscillation，验证单点
3. **日志持久化**: 将运行日志保存到 `artifacts/sdaa_droplet_oscillation.log`
4. **结果回填**: 完成后更新 `artifacts/sdaa_droplet_oscillation.json` 并重新评估判据
5. **RANS 改进**: suboff bench2/bench3 的 RANS 误差 >15%，需调查湍流模型在 SDAA 上的表现

### 5.3 诚实声明

本任务为 **honest-FAIL**：
- SDAA 扫描已正确启动并在设备上执行
- 但由于运行时间超出预算，未能获得完整结果
- 所有判据均为 PENDING，无法给出 PASS/FAIL 结论
- 原始 CUDA 基准已验证物理模型正确，SDAA 预期应一致，但需程序化证据

---

## 附录：工件清单

### 脚本
- `scripts/sdaa_sweep_droplet_oscillation.py` — 主扫描脚本
- `scripts/sdaa_sweep_droplet_oscillation_minimal.py` — 最小验证脚本

### 工件
- `artifacts/sdaa_droplet_oscillation.json` — SDAA 扫描结果（IN_PROGRESS）
- `artifacts/suboff_bench1_cumulant_sdaa20.json` — SUBOFF Cumulant Re=1000
- `artifacts/suboff_bench2_rans_re1e5_sdaa21.json` — SUBOFF RANS Re=1e5
- `artifacts/suboff_bench3_rans_re1e4_sdaa22.json` — SUBOFF RANS Re=1e4
- `artifacts/suboff_best_rans_summary.json` — SUBOFF 综合摘要
- `artifacts/kbc_diagnostic_16cube_20steps.json` — KBC 诊断

### 文档
- `docs/sdaa_sweep/droplet_oscillation.md` — SDAA 扫描文档
- `benchmarks/verified/droplet_oscillation/README.md` — 原始基准说明
- `benchmarks/verified/droplet_oscillation/result.json` — CUDA 参考结果

---

**报告生成时间**: 2026-01-XX  
**报告类型**: 补报（mission 21 backfill）  
**结论**: honest-FAIL（SDAA 扫描未完成，判据 PENDING）
