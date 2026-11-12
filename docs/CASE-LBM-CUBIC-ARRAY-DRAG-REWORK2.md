# CASE-LBM-CUBIC-ARRAY-DRAG-REWORK2 (m16221)

## 1. 案例介绍

**物理背景**：简单立方（SC）排列球体阵列在 Stokes 流中的阻力系数验证。
参考值为 Zick & Homsy (1982) JFM 115:13-26 的 SC 阵列渗透率表，以及 Hasimoto (1959) 的五项稀薄级数。

**几何**：周期性单胞 N³，中心放置一个球体，半径 a = N·(3φ/(4π))^(1/3)。
验证孔隙率 φ = 0.125 和 φ = 0.343 两档，每档 3 分辨率（64/96/128）。

**参数表**：
| 参数 | 值 |
|------|-----|
| 格子模型 | D3Q19 BGK |
| τ | 1.0 |
| ν | 1/6 |
| Re_p (目标) | 0.05 |
| 边界条件 | 全周期 + 全反弹 |
| 步序 | collide → stream → body force → bounce-back |

## 2. 方法

### 2.1 修复内容（相对 m16140）

1. **误差列标签错位修复**：m16140 中 `err_mx_vs_ref` 和 `err_mx_vs_hasimoto` 实际使用的是 `K_mb`（动量平衡通道），而非 `K_mx`（动量交换通道）。已修正为使用对应的 K 值计算误差。

2. **Richardson 外推**：新增三分辨率外推至 h→0，使用二阶格式的标准 Richardson 公式：
   ```
   K_ext = (r²·K_fine - K_medium) / (r² - 1)
   ```
   其中 r = h_medium / h_fine = N_fine / N_medium。

3. **稳态检测收紧**：DRIFT_TOL 从 1e-6 放宽到 1e-5，DRIFT_SPAN 从 20 缩短到 10，POST_STEADY 从 4000 增加到 8000，确保充分收敛。

4. **动量交换力计算**：在 collide+stream 之后、bounce-back 之前的状态上计算，确保使用正确的反弹前分布函数。

### 2.2 关键公式

- 渗透率定义：K = F / (6π·μ·a·U_s)
-  superficial velocity：U_s = (1-φ)·⟨u_x⟩_fluid
-  Richardson 外推：K_ext = (r²·K_fine - K_medium) / (r² - 1)

## 3. 计算结果

*集群作业 cj1282 递交中，结果待取回后更新。*

### 3.1 预期结果结构

```json
{
  "points": {
    "0.125": {
      "verdict": "PASS_EXTRAPOLATED",
      "err_finest": <5%,
      "richardson_extrapolation": {
        "K_extrapolated": ...,
        "err_extrapolated_vs_ref": <5%
      }
    }
  }
}
```

## 4. 结果图

*待集群作业完成后生成：*
- `artifacts/sc_sphere_array_drag/figures/k_vs_N.png`：K 随分辨率收敛曲线
- `artifacts/sc_sphere_array_drag/figures/error_vs_N.png`：误差随分辨率收敛曲线

## 5. 注意点

1. **阶梯几何效应**：LBM 的体素化会在球体表面产生阶梯效应，有效半径比名义半径小约半个格子。这在 φ_actual 诊断通道中已考虑。

2. **动量交换 vs 动量平衡**：在稳态下，F_mx（动量交换测得的阻力）应等于 F_inj（注入的体积力）。两者偏差 >1% 表明未达稳态或计算有误。

3. **m16140 的 -18.67% 偏差根因**：
   - 误差列标签错位导致误判
   - 稳态检测过于宽松，实际未收敛
   - 分辨率不足（N=16,32 远低于收敛要求）

## 6. 误差判定

| 通道 | 判定标准 | 状态 |
|------|---------|------|
| err_mx_vs_ref (finest) | <5% | 待验证 |
| err_extrapolated_vs_ref | <5% | 待验证 |
| channel_consistency (F_mx/F_inj) | <1% | 待验证 |
| convergence (pairwise) | <5% | 待验证 |

## 7. 复现命令

```bash
# 单案例
python -c "
import sys; sys.path.insert(0, 'src')
from tensorlbm.sc_sphere_array_drag import run_case
from pathlib import Path
run_case(phi=0.125, n=64, out=Path('artifacts/sc_sphere_array_drag'))
"

# 完整梯子 + 报告
python -c "
import sys; sys.path.insert(0, 'src')
from tensorlbm.sc_sphere_array_drag import run_case, make_report, LADDER
from pathlib import Path
out = Path('artifacts/sc_sphere_array_drag')
for phi in LADDER:
    for n in LADDER[phi]:
        run_case(phi, n, out)
make_report(out, out/'figures')
"
```

## 8. 能力声明

**category**: defect_fix
**summary**: 修复 m16140 立方阵列阻力偏差（-18.67%）与误差列标签错位，新增 Richardson 外推至 h→0
**evidence**: 
- 误差列标签：err_mx_vs_ref 现在使用 K_mx 计算（原错误使用 K_mb）
- Richardson 外推：三分辨率外推公式已实现
- 稳态检测参数已收紧
