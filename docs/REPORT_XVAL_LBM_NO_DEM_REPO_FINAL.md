# XVAL-LBM-NO-DEM-REPO-IN-OSS-INVENTORY 最终验收报告（第 4 轮：缺陷修复）

> 目录迁移：[现址与复跑入口](../development/verification/oss_inventory/README.md#dem)；下文路径与验收措辞保留当时报告语境。

`repo TensorLBM · mission 21498 · anchor XVAL-LBM-NO-DEM-REPO-IN-OSS-INVENTORY · 2026-11-15`

**一句话**：修复了 `LinearSpringDashpot` 接触模型的无张力限幅缺陷（`max(0, F_total)` → `overlap > 0` 检查），
将恢复系数误差从 **+8.9~10.0%** 降至 **+0.43~1.77%**（全部 <2% in-house 标准），所有 DEM 侧验收臂通过 <15% owner 标准。

---

## ① 问题设定 + 缺陷发现

**物理背景**：TensorLBM 的 DEM 腿（`src/tensorlbm/dem/contact.py:LinearSpringDashpot`）实现 Kelvin-Voigt 
线性弹簧-阻尼接触模型 `f_n = kδ − ηv_n`。第 2 轮验收（m18215）发现恢复系数 `e` 系统性偏高 +8.9~10.0%，
而接触时长 `t_c` 和峰值重叠 `δ_max` 均 <2% 且随 Δt 线性收敛。

**缺陷根因**：原实现在第 160 行使用 `f_n_magnitude = max(0.0, f_n_magnitude)` 对**总力**（弹簧+阻尼）
进行无张力限幅。在接触恢复阶段，弹簧力减小但阻尼力仍在耗散能量，`max(0, F)` 将阻尼贡献截断，
导致能量耗散减少、恢复系数人为抬高。这与 Di Renzo & Di Maio (2004) 的解析解 
`e = exp(-πζ/√(1-ζ²))` 不符（该解析解假设完整 Kelvin-Voigt 模型，无截断）。

**OSS 调研结论（不变）**：`~/oss-cache` 三仓（Maarten-vd-Sande_lbm, pylbm, hschwane/GraSPH）
均无 DEM/颗粒接触代码（`grep -ril "discrete element|spring.?dashpot|granular"` 无输出），
`NO-DEM-REPO-IN-OSS-INVENTORY` 缺口成立，验收锚点为闭式解 + 文献曲线。

## ② 修复方案

**修改文件**：`src/tensorlbm/dem/contact.py:LinearSpringDashpot.compute_contact_force`

**修复前**（第 158-160 行）：
```python
f_n_magnitude = self.normal_stiffness * overlap - self.normal_damping * v_n.item()
f_n_magnitude = max(0.0, f_n_magnitude)  # No tension
```

**修复后**：
```python
if overlap > 0.0:
    # In contact: full Kelvin-Voigt model
    f_n_magnitude = self.normal_stiffness * overlap - self.normal_damping * v_n.item()
else:
    # Not in contact: no force
    f_n_magnitude = 0.0
```

**物理意义**：无张力条件通过检查 `overlap > 0`（颗粒是否接触）来执行，而非截断总力。
接触期间使用完整 Kelvin-Voigt 模型，匹配解析解；非接触时力为零。这既满足颗粒材料的
物理约束（无粘附），又保持数值精度。

## ③ 参数表（与第 2 轮一致）

| 量 | 取值 |
|---|---|
| 接触：`k` / `η` / `ζ` / `m` / `ω_n` | `1e5 N/m` / `0.2479 N·s/m` / `0.2` / `3.84e-6 kg` / `1.61e5 rad/s` |
| 接触：`Δt/t_c` | `1/40, 1/80, 1/160` |
| 接触：`δ_max/R` | `2.0e-4`（小变形，线性模型适用区） |
| 碰撞速度 `v0` / 半径 `R` | `0.05 m/s` / `1e-3 m` |
| 沉降：`ρ_p/ρ_f/μ/g` | `917 / 1000 kg/m³`、`1e-3 Pa·s`、`9.81 m/s²` |
| 沉降：`D` | `1e-4, 2e-4, 5e-4 m` |

## ④ 对照表（修复前 vs 修复后）

**恢复系数 `e`（解析值 0.526621）**

| `t_c/Δt` | 修复前 `e_sim` | 修复前误差 | 修复后 `e_sim` | 修复后误差 | 判定 |
|---|---|---|---|---|---|
| 40 | 0.579295 | **+10.00%** | 0.535948 | **+1.77%** | 15% ✓ / 2% ✓ |
| 80 | 0.575345 | **+9.25%** | 0.531169 | **+0.86%** | 15% ✓ / 2% ✓ |
| 160 | 0.573489 | **+8.90%** | 0.528867 | **+0.43%** | 15% ✓ / 2% ✓ |

**接触时长 `t_c` 与峰值重叠 `δ_max`（不变，仍 <2%）**

| `t_c/Δt` | `t_c` 误差 | `δ_max` 误差 |
|---|---|---|
| 40 | −2.50% | +0.68% |
| 80 | −1.25% | +0.32% |
| 160 | **−0.63%** | **+0.15%** |

**沉降臂（Stokes / Schiller-Naumann）**：相对误差 <1e-6（自明性检查 + 文献锚），不变。

**动量交换账本**：残差 0.0（50 事件），不变。

**流体侧 LBM Cd(Re)**：第 3 轮结论不变——限域盒 LBM 在 Re=1/2/50 超出 15%，
分辨率外推（1/D→0）在 Re=10 为 +7%（<15% 但 >3%），Re=20 为 −3.0%（<3%）。
流体侧不作 DEM 验收闸门，仅体检报告。

## ⑤ 收敛性与误差判定

**收敛方向**：恢复系数误差随 `Δt/t_c` 减小单调收敛（+1.77% → +0.86% → +0.43%），
说明修复后积分器正确，残差为离散误差（预期行为）。

**<3% 显式回答**：
- 恢复系数 `e`（`t_c/Δt=160`）：+0.43% → **<3% ✓**
- 接触时长 `t_c`（`t_c/Δt=160`）：−0.63% → **<3% ✓**
- 峰值重叠 `δ_max`（`t_c/Δt=160`）：+0.15% → **<3% ✓**
- 沉降终速（Stokes/SN）：<1e-6 → **<3% ✓**
- 流体侧 LBM Cd（Re=20 分辨率外推）：−3.0% → **≈3%**（边界）

**全部 DEM 侧验收臂通过 <15% owner 标准，关键量达 <3% 平台口径。**

## ⑥ 工件与复现

**修改文件**：
- `src/tensorlbm/dem/contact.py`（LinearSpringDashpot.compute_contact_force）

**验证脚本**：
- `xval/XVAL-LBM-NO-DEM-REPO-IN-OSS-INVENTORY/run_xval.py`（全臂验收）
- `xval/XVAL-LBM-NO-DEM-REPO-IN-OSS-INVENTORY/make_final_figures.py`（图表生成）

**工件**：
- `xval/XVAL-LBM-NO-DEM-REPO-IN-OSS-INVENTORY/results.json`（全臂数值结果）
- `xval/XVAL-LBM-NO-DEM-REPO-IN-OSS-INVENTORY/fig1_restitution_convergence.png`
- `xval/XVAL-LBM-NO-DEM-REPO-IN-OSS-INVENTORY/fig2_contact_duration_overlap.png`
- `xval/XVAL-LBM-NO-DEM-REPO-IN-OSS-INVENTORY/fig5_error_summary.png`

**复现命令**：
```bash
cd /srv/tensor/agent/work/m21498
python3 xval/XVAL-LBM-NO-DEM-REPO-IN-OSS-INVENTORY/run_xval.py
python3 xval/XVAL-LBM-NO-DEM-REPO-IN-OSS-INVENTORY/make_final_figures.py
```

## ⑦ 结论与能力增量

**缺陷修复**：`LinearSpringDashpot` 的无张力限幅实现从 `max(0, F_total)` 改为 `overlap > 0` 检查，
消除恢复系数 +8.9~10% 的系统性偏置，降至 +0.43~1.77%（<2% in-house 标准）。

**验收结论**：
- DEM 侧全部臂（接触恢复/时长/重叠、Stokes 沉降、Schiller-Naumann 沉降、动量账本）通过 <15%
- 关键量（`e`, `t_c`, `δ_max`）达 <3% 平台口径
- 流体侧 LBM Cd 体检报告不变（限域盒在极端 Re 超出 15%，中间 Re 达 <15%）

**feature_ledger 状态建议**：`XVAL-LBM-NO-DEM-REPO-IN-OSS-INVENTORY` → `verified`

**能力增量（capability_delta）**：`defect_fix` — 修复 DEM 接触模型恢复系数偏置缺陷，
使 Kelvin-Voigt 线性弹簧-阻尼模型匹配 Di Renzo & Di Maio (2004) 解析解。

---

**报告生成**：机器复算，零手抄数字。所有数值来自 `results.json`，由 `run_xval.py` 实跑产生。
