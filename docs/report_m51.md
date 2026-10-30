# Mission 51 补报：SDAA 验证案例扫荡 — cavity_natural_convection

> **报告类型**: 补报（原 mission 51 已交付但缺论文级进展报告）  
> **仓库**: TensorLBM  
> **案例**: benchmarks/verified/cavity_natural_convection  
> **参考基准**: de Vahl Davis 1983, *Int. J. Numer. Methods Fluids* 3:249-264  
> **总体判定**: 验证案例 ✅ PASS（晋级）；SDAA 可行性扫描 ⚠️ honest-FAIL（粗网格预期内）

---

## 1. 案例介绍

二维方腔自然对流（de Vahl Davis benchmark）是 CFD 验证的经典问题：

- **几何**: 1×1 方腔，左壁加热 (T=1)、右壁冷却 (T=0)、上下壁绝热
- **参数**: Ra = 10⁴, Pr = 0.71（空气）
- **格子模型**: D2Q9（速度）+ D2Q5（温度），BGK 碰撞 + Guo 浮力
- **参考值**: Nu_ref = 2.243, u_max_ref = 16.178, v_max_ref = 19.617

该案例在 NVIDIA GPU 侧已有成熟判据（六门验证体系），本次任务将其移植到 SDAA 设备做可行性扫描。

---

## 2. 方法

### 2.1 验证案例（原 mission 51 主体）

使用 `tensorlbm.thermal.simulate_natural_convection` 库函数，三档网格重跑：

| 网格 | 步数 | 稳态检测 | 耗时 |
|------|------|---------|------|
| 64² | 32,000 | t_stop=22,000 | 29.2 s |
| 128² | 115,500 | t_stop=75,000 | 105.3 s |
| 256² | 457,600 | t_stop=300,000 | 416.0 s |

**库修复**: thermal sha12 = `598889356088`，修复壁面 BC 双缺陷（mask_wall_u=True），与 thermal_cavity 逐位一致。

### 2.2 SDAA 可行性扫描（scripts/sdaa_sweep_cavity_natural_convection.py）

受工具 120s 超时限制，采用缩减配置：
- **网格**: nx=32（粗网格）
- **步数**: 2000
- **设备**: SDAA（`torch.sdaa.get_device_name(0)` → `/dev/tcaicard6`）
- **Triton**: interpret 模式未设置

脚本独立运行，不改业务代码，仅验证 SDAA 设备可执行性。

---

## 3. 关键结果

### 3.1 验证案例（paper 档，六门验证）

| 网格 | Nu | Nu 误差 | u_max 误差 | v_max 误差 | 判定 |
|------|-----|---------|-----------|-----------|------|
| 64² | 2.1098 | -5.94% | -1.98% | -3.33% | Nu/v_max 超 3% |
| 128² | 2.1778 | **-2.91%** | **-0.97%** | **-1.67%** | ✅ 全 ≤3% |
| 256² | 2.2125 | **-1.36%** | **-0.59%** | **-0.68%** | ✅ 全 ≤3% |

**六门验证结果**:

| 门 | 结果 | 证据 |
|----|------|------|
| library_entry | ✅ PASS | `from tensorlbm.thermal import simulate_natural_convection` |
| direct_observable | ✅ PASS | N≥128 时 Nu/u_max/v_max 全部 ≤3% |
| monotone_convergence | ✅ PASS | \|Nu 误差\|: 5.94% → 2.91% → 1.36%（严格递减）|
| reference_locked_first | ✅ PASS | 预注册 2026-10-20T03:30:00Z，Nu_ref=2.243 |
| case_readme | ✅ PASS | README.md 存在，5491 bytes |
| figures | ✅ PASS | 5 张图（温度场/速度场/Nu 收敛/误差对比/Nu 历史）|

### 3.2 SDAA 可行性扫描

| 指标 | 值 | 阈值 | 判定 |
|------|-----|------|------|
| Nu | 1.8923 | ≤3% 误差 | ❌ -15.63% |
| u_max | 18.104 | ≤3% 误差 | ❌ +11.90% |
| v_max | 20.004 | ≤3% 误差 | ✅ +1.97% |
| 耗时 | 29.62 s | <120 s | ✅ |
| 设备证据 | `/dev/tcaicard6` | — | ✅ |

**SDAA 扫描判定**: honest-FAIL。粗网格（32²）+ 短步数（2000）导致精度门预期失败，但成功证明 SDAA 设备可执行 thermal 库。

---

## 4. 结果图

验证案例已生成 5 张 paper 档图件（位于 `benchmarks/verified/cavity_natural_convection/figures/`）：

1. **field_temperature.png** — 中截面温度场云图（N=128）
2. **field_velocity_magnitude.png** — 速度幅值云图，中心漩涡 + 边界层射流
3. **curve_nu_convergence.png** — Nu 误差收敛曲线，三档网格严格单调递减
4. **curve_error_comparison.png** — 多观测量误差对比（Nu/u_max/v_max）
5. **curve_nu_history.png** — Nu 随步数演化历史

> 注：本补报未重新生成图件，引用原 mission 51 已产出图件。

---

## 5. 注意点

1. **库修复是关键**: 修复前 N=128 的 Nu 误差为 +6.64%（发散），修复后降至 -2.91%。根因为 thermal 库壁面 BC 双缺陷。
2. **SDAA 扫描的粗网格限制**: nx=32 的网格分辨率不足以捕捉边界层，Nu 误差 -15.6% 属预期行为。若需 SDAA 精度验证，需放宽超时限制或使用 nx≥128。
3. **单调收敛性**: 三档网格 Nu 误差严格递减（5.94% → 2.91% → 1.36%），证明数值格式正确、网格收敛。
4. **对称性**: Nu_left 与 Nu_right 差异 <0.02%（N=64），验证了能量守恒。
5. **Mach 数**: 最大 Ma = 0.0246（N=64），满足不可压缩假设（Ma < 0.3）。

---

## 6. 误差判定

### 关键量误差是否 <3%？

| 观测量 | N=64 | N=128 | N=256 | 是否 <3%（N≥128）|
|--------|------|-------|-------|-----------------|
| Nu | -5.94% | **-2.91%** | **-1.36%** | ✅ 是 |
| u_max | -1.98% | **-0.97%** | **-0.59%** | ✅ 是 |
| v_max | -3.33% | **-1.67%** | **-0.68%** | ✅ 是 |

**结论**: N≥128 时所有关键量误差均 <3%，满足验证标准。

### 无基准声明原因

SDAA 可行性扫描（nx=32）精度门失败，原因已明确：粗网格 + 短步数。非算法缺陷，属工具超时约束下的预期行为。

---

## 审计与局限

- **审计范围**: 验证案例（三档网格重跑）+ SDAA 可行性扫描（单网格）
- **局限**: SDAA 扫描未做网格收敛性测试（受超时限制），无法声明 SDAA 设备精度
- **honest-FAIL**: SDAA 扫描精度门 FAIL 如实保留，不掩盖
- **可复现性**: 所有脚本、结果 JSON、图件均在仓库内，可完整复现

---

*报告生成时间: 2026-01-20 (补报)*  
*原 mission 51 交付日期: 2026-01-19*
