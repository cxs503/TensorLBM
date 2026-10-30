# Mission 24 补报：SDAA 验证扫荡 — Laplace CAC 相场两相 LBM 静态液滴基准

> **仓库**: TensorLBM · **分支**: agent/m24 · **案例**: `benchmarks/verified/laplace_cac_phasefield`
> **状态**: honest-FAIL（物理判据全 PASS，但臂覆盖不足导致整体 FAIL）

---

## 1. 案例介绍

**物理问题**：周期立方域中心放置静态液滴，CAC（Conservative Allen–Cahn）/ HCH（Half-Central Hermite）相场两相 LBM 演化至稳态。液滴内外压差满足 Young–Laplace 方程（3D 球）：

$$\Delta p = p_{in} - p_{out} = \frac{2\sigma}{R}, \quad \sigma_{rec} = \frac{\Delta p \cdot R}{2}$$

**数值模型**：
- CAC/HCH 双分布 D3Q19（Liang et al., PRE 97, 033309）
- h-LBE 输运相场 φ（CAC 对流重生项），g-LBE 输运压力/动量（HCH）
- fp64 计算；μ(φ) 连续混合律（harmonic）
- 界面宽度 W = 4；迁移率 M = 0.05；p₀ = 0.01；周期边界

**Route-B σ 标定**：CAC 方案的实测 σ_eff(W) ≠ σ_input，是 W 依赖的标定常数。W=4 时 σ_eff/σ_input = 0.9239（−7.61%），故模型输入一次性补偿 σ_model = σ_target / 0.9239 = 0.010824。

**基准设计**（NVIDIA 侧已验证）：
| 臂 | 描述 | 网格 | 步数 |
|---|---|---|---|
| B1-R20 | R=20, ρ比=100 | 128³ | 12000 |
| B1-R32 | R=32, ρ比=100 | 128³ | 12000 |
| B1-R48 | R=48, ρ比=100 | 128³ | 12000 |
| B3-ρ10 | R=20, ρ比=10 | 96³ | 12000 |
| B3-ρ100 | R=20, ρ比=100 | 96³ | 12000 |

---

## 2. 方法

**独立运行脚本**：`scripts/sdaa_sdaa_sweep_laplace_cac_phasefield.py`（不改业务代码）

- 导入案例自身 `run.py` 获取判据/标定常数（source of truth）
- 计算张量全部放置于 `sdaa:0`（`/dev/tcaicard27`，torch 2.7.1，16 GB）
- 无 CPU 回退；triton 路径扫描确认 cac_lbm 调用链无 triton kernel
- **缩减配置**（受 SDAA 卡吞吐限制）：64³ 网格、10000 步、仅跑 b3_rho100 单臂
- 全量案例预估耗时：B1@128³×3 ≈ 8.68 h + B3@96³×2 ≈ 2.44 h = **11.12 h**（超出 20 min 扫荡预算）

---

## 3. 关键结果

### 3.1 SDAA 单臂结果（b3_rho100, 64³, 10000 步）

| 指标 | 目标/阈值 | SDAA 实测值 | 判定 |
|---|---|---|---|
| σ_rec 窗口均值 | 0.01 | **0.00999433** | — |
| \|σ_rec/σ_target − 1\| | ≤ 1.5% | **0.057%** | ✅ PASS |
| 伪速度 max\|u\|（窗内） | ≤ 2×10⁻⁴ | **6.79×10⁻⁵** | ✅ PASS |
| φ 积分漂移 @10k 步 | ≤ 10⁻⁶ | **8.61×10⁻¹³** | ✅ PASS |
| NaN-free | 必须 | **true** | ✅ PASS |

### 3.2 与 NVIDIA 参考值对比

| 指标 | NVIDIA (96³, 12000步) | SDAA (64³, 10000步) | 差异来源 |
|---|---|---|---|
| σ_rec (b3_rho100) | 0.00999382 | 0.00999433 | 网格/步数缩减 |
| 误差 | 0.062% | 0.057% | 一致量级 |
| u_max | 6.74×10⁻⁵ | 6.79×10⁻⁵ | 一致量级 |
| φ drift | 8.62×10⁻¹³ | 8.61×10⁻¹³ | 一致量级 |

### 3.3 吞吐

| 指标 | 值 |
|---|---|
| 单步耗时（64³） | 108.47 ms |
| ms/step/cell | 4.14×10⁻⁴ |
| 全量案例预估 | 11.12 h |

### 3.4 门控汇总

| 门控 | 判定 |
|---|---|
| σ_rec 误差 ≤ 1.5%（b3_rho100） | ✅ PASS |
| B1 R-ladder 散布 ≤ 1% | ⚠️ NOT_EVALUATED（需 128³×3 臂，超时） |
| 伪速度 ≤ 2e-4 | ✅ PASS |
| NaN-free | ✅ PASS |
| φ 漂移 ≤ 1e-6 | ✅ PASS |
| 臂覆盖（5/5） | ❌ FAIL（仅 1/5） |

**整体判定：FAIL**（4 PASS / 1 FAIL / 1 NOT_EVALUATED）

---

## 4. 结果图

本补报无 npz/png 工件。理由：SDAA 扫荡脚本仅输出 JSON 时间序列（21 个采样点），未保存场数据（φ/u/p 的 npz）。如需可视化，可在后续全量运行中增加 `--save-fields` 选项导出 VTK/npz，再用 matplotlib 绘制 σ_rec 收敛曲线与液滴截面云图。

---

## 5. 注意点

1. **网格缩减效应**：SDAA 扫荡使用 64³（案例要求 96³/128³），σ_rec 误差 0.057% 仍在门限内，但网格收敛性未在此配置下验证。
2. **步数缩减**：10000 步 vs 案例 12000 步；末 2000 步窗口仅 4 个采样点（每 500 步），统计量偏少但已足够稳定（tail σ_rec 标准差 < 10⁻⁵）。
3. **臂覆盖不足**：仅跑了 b3_rho100 一臂；B1 三半径阶梯与 b3_rho10 均未执行，原因是 SDAA 卡吞吐（108 ms/step@64³）外推至全量配置需 11+ 小时，超出扫荡预算。
4. **Route-B 标定常数跨设备迁移**：σ_model = 0.010824 在 SDAA 上同样有效（σ_rec 误差 0.057%），表明标定常数是数值格式属性而非硬件属性。
5. **triton 无关**：cac_lbm 调用链（d3q19/boundaries3d/phasefield.stream_boundary_contract）经源码扫描确认无 triton kernel，无需 TRITON_INTERPRET 变通。

---

## 6. 误差判定

**关键量 σ_rec 误差是否 < 3%？**

✅ **是**。SDAA 上 b3_rho100 臂 σ_rec 相对误差 = **0.057%**，远小于 3% 阈值，也小于案例自身的 1.5% 门限。

**无基准声明的原因**：
- 本案例的"基准"是 Young–Laplace 解析关系 σ_rec = Δp·R/2，目标值 σ_target = 0.01 由 Route-B 标定给出
- NVIDIA 侧 5 臂全量运行已验证该标定在 R∈{20,32,48} 与 ρ比∈{10,100} 上均闭合 ≤ 0.45%
- SDAA 单臂结果与 NVIDIA 参考值一致（σ_rec 差异 < 10⁻⁶），物理自洽

---

## 附录：工件清单

| 文件 | 说明 |
|---|---|
| `artifacts/sdaa_laplace_cac_phasefield.json` | SDAA 扫荡完整输出（设备证据/配置/时间序列/门控） |
| `scripts/sdaa_sweep_laplace_cac_phasefield.py` | 独立运行脚本（不改业务代码） |
| `benchmarks/verified/laplace_cac_phasefield/README.md` | 案例说明与判据 |
| `benchmarks/verified/laplace_cac_phasefield/result.json` | NVIDIA 参考结果 |
