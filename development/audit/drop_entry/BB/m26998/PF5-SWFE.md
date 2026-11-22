# DE-X-PF5-SWFE · Swift-Osborn-Bartholomew 自由能（∇ψ 张量力）密度比可达性

**波次**: TRACK_COMB_W1（组合比较矩阵波，owner 令 2026-11-21）
**轴**: 多相 · Swift 自由能
**单**: m26998（R1 重派，前单 m26956 no_progress 死）
**日期**: 2026-11-22

## 1. 任务

验证 Swift-Osborn-Bartholomew (2009) 自由能模型（∇ψ 张量力，自由能压力
`p = ρc_s² + ψμc_s²`）在**密度比 10 起**的静态液滴可达性：
1. 静态液滴（重相悬浮于轻浴，无体积力）
2. 判据：稳定（无 NaN）、伪流 `u_max`、体积守恒、Laplace 压
3. 若 10:1 能跑 → 加粗带入水（n≤80）判深度方向

DE 锚（17 位全精度取自原文件）：实验 1.857434D@10.394ms(pc)；
ctrl 1.435220D@11.0906ms(abs)/10.5433(pc)；窗 24.005ms；mach 0.2598；
contact 45；drift 4.8e-6；1 cell=0.125D≈6.7%。

## 2. 方法

**模型**: Swift 自由能两相 LBM（`tensorlbm.swift_free_energy`，D2Q9 + D3Q19）：
- 自由能 `F = ∫[(a/2)ψ² − (a/4)ψ⁴ + (κ/2)|∇ψ|²]dV`，`a = −2κ/ξ²`
- 化学势 `μ = aψ(1−ψ²) − κ∇²ψ`
- 密度比 `ρ = ρ_g + (ρ_l−ρ_g)(ψ+1)/2`
- 自由能压力 `p = ρc_s² + ψμc_s²`（∇ψ 张量力 = `ψμc_s²` 毛细压贡献）
- 2-D 核复用已验证 sailfish 各向异性系数（`free_energy_pressure.anisotropic_coefficients`），
  二阶矩闭合到自由能压力张量；3-D 核用 D3Q19 标准 BGK 权重（各向异性 Korteweg 应力
  忽略，Swift 2009 原式用各向同性自由能压力）

**参数**（静态探针）: n=64，R=16，κ=0.02，ξ=3.0，τ_f=τ_g=1.0，ρ_g=1.0

## 3. 结果

### 3.1 静态液滴（64×64，R=16，300 步）

| ratio | stable | nan_step | u_max | 体积漂移 | Δp | Δp 解析 | 相对误差 |
|---|---|---|---|---|---|---|---|
| 10 | **False** | 38 | 5.5e+116 | — | — | 5.6e-4 | — |
| 100 | **False** | 14 | 9.7e+110 | — | — | 5.6e-4 | — |
| 1000 | **False** | 14 | 6.8e+120 | — | — | 5.6e-4 | — |

**关键发现**:
- **密度比 ≥2 即失稳**（ratio=1.0/1.1 稳定，ratio=2 在 step 19 NaN，ratio=10 在 step 7-38 NaN）。
- 加大 τ_f/τ_g（至 50）、加宽界面（ξ 至 8）、减弱表面张力（κ 至 0.005）均**无法**稳定 ratio=10。
- 失稳源：自由能压力 `p = ρ/3 + ψμ/3` 在界面处有大幅跳变（密度比所致），驱动非物理加速。
- 与仓内已验证 sailfish 移植（等密度，`free_energy_pressure`）对照：等密度下本实现
  质量守恒、伪流 ~4e-3，行为一致。

### 3.2 等密度对照（ratio=1，验证实现正确性）

| 项 | 值 |
|---|---|
| 稳定 | True（300 步无 NaN） |
| 伪流 u_max | 3.9e-3 |
| 质量漂移 | <1e-10（2-D） |
| 序参量守恒 | <1e-10（2-D） |

![stability](../../artifacts/pf5_swfe_stability.png)

## 4. 结论

**honest FAIL**：Swift 自由能模型（∇ψ 张量力，自由能压力 `p = ρc_s² + ψμc_s²`）在
**密度比 ≥2 即失稳**，密度比 10/100/1000 全部在 7-38 步内 NaN。加大黏度（τ_f 至 50）、
加宽界面（ξ 至 8）、减弱表面张力（κ 至 0.005）均无法挽救。

**矩阵补全结论**：Swift 自由能模型在液滴入水锚（水/气密度比 ~1000）下**不可达**，
与谱系已知弱点一致。该模型仅适用于**近等密度**（ratio ≤ 1.1）场景，不适用于
水/气量级密度比。W2 选型时 Swift 自由能臂应标记为**高密度比不可用**。

## 5. 产物

- `src/tensorlbm/swift_free_energy.py` — Swift 自由能模型（D2Q9 + D3Q19，密度比支持）
- `benchmarks/water_entry/single_drop/swift_static_droplet.py` — 静态液滴验证脚本
- `benchmarks/water_entry/single_drop/swift_static_results.json` — 验证结果
- `benchmarks/water_entry/single_drop/swift_static_figure.py` — 结果图生成
- `tests/test_swift_free_energy.py` — 单元测试（9 通过 + 1 xfail）
- `artifacts/pf5_swfe_stability.png` — 稳定性图

## 6. 误差判定

- 等密度（ratio=1）：质量守恒 <1e-10（2-D），伪流 3.9e-3（与 sailfish 移植一致）
- 高密度比（10/100/1000）：**失稳**，无 Laplace 压可测（honest FAIL）
- 是否 <3%：**不适用**（高密度比失稳，无稳态可测）；等密度下实现正确性已验证
