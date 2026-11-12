# m16921 · 方腔线性稳定性模块锚点审计与修复（verifier 席）

**结论先行**：`tensorlbm.conjugate_thermal.stability` 原文档声称「差分加热方腔（侧加热）的导电态在
`Ra_c = 1707.762` 失稳，该值由本模块的独立离散复现而非抄录」——**该声明为假**。
实测：该算子所描述的配置在 `Ra = 1707.76` 处**强稳定**（`σ = −9.75`，n=12），其真实中性点
`Ra_c ≈ 1.455e4`（n=12），是经典锚点的 **8.52 倍**；且原 `critical_rayleigh()` 以默认参数
在 n=8/12/16 三个网格上**均无法括住根**（返回 `ra_c=None`），根本不可能产出 1707.762。
本次修复：把两种配置在模块内**显式分开**，为 `1707.7597` 建立它真正所属的 Rayleigh–Bénard 层路径
（重力 ∥ 温度梯度），实测二阶收敛，Richardson 外推 `1707.895`（相对误差 **+7.9e-5**，即 0.008%）；
侧加热配置改为诚实登记（实测中性点 + 无文献锚点声明 + 不再抛 `KeyError`）。

产物（数字全部由脚本注入，零手抄）：
`scripts/m16921_cavity_stability_anchor_audit.py` →
`artifacts/m16921_cavity_stability_anchor_audit.json`、
`artifacts/m16921_sigma_vs_ra_configurations.png`、
`artifacts/m16921_benard_grid_convergence.png`。

---

## ① 案例介绍

| 项 | 侧加热方腔（旧路径） | Rayleigh–Bénard 层（新增路径） |
|---|---|---|
| 基态 | `T̄ = 1 − x`（侧壁热/冷） | `T̄ = 1 − y`（下热上冷） |
| 重力方向 | `−y`，**⟂** 温度梯度 | `−y`，**∥** 温度梯度 |
| 边界 | 四壁 no-slip（Thom 涡量）、侧壁等温、上下绝热 | 上下刚性板、等温、x 向 Bloch 周期（波数 a） |
| 经典锚点 | 无（本仓未登记文献锚） | Chandrasekhar 1961 §II：`Ra_c = 1707.7597`，`a_c = 3.11632` |

无量纲：`H = 1`，`α = 1`，`ν = Pr`，`gβ = Ra·Pr`，`σ` 单位 `α/H²`。
两配置共用同一套线性化 Boussinesq 方程，**差别只在边界条件与基态方向**，因此极易被误当作同一问题。

## ② 怎么做的

1. 复核（只读）：`critical_rayleigh()` 默认调用 → 三个网格全部 `not_bracketed`；`σ(Ra)` 单调平缓
   （见 ③ 表），与「1707.76 附近变号」的声明矛盾。
2. 溯源：`scripts/run_in28_n03_validation.py` 早已诚实记下「An independent dense linear-stability
   solve was attempted and did not converge（see report）」——即**仓内自己的验证报告与模块 docstring
   互相矛盾**，docstring 是夸大的那一方；且该模块此前**无任何测试覆盖**，故缺陷长期存活。
3. 判据先行（修复前写死）：
   - Bénard 路径必须在 `a = 3.11632` 处二阶收敛到 `1707.7597`（误差比 ≈ 4，外推后 <3%）；
   - 最危险波数须落在 `a_c = 3.11632 ± 0.2`；
   - 与独立实现 `rb_continuum_stability`（流函数四阶铅笔 `(D²−a²)²`，与本次的涡量/Thom 二阶格式
     属**不同离散**）须在 `Pr` 时间单位换算后一致到 <3%；
   - 侧加热配置必须在其「自称的」锚点上被判定为稳定（`σ<0`），即证明旧锚点无效。
4. 实现（`src/tensorlbm/conjugate_thermal/stability.py`）：
   `benard_operator / benard_growth_rate / benard_critical_rayleigh /
   benard_wavenumber_scan / benard_critical_rayleigh_continuous / richardson`；
   Bénard 用 Bloch 因子（`∂ₓ→ia`，`∂²ₓ→−a²`）把二维问题降为 `y` 方向一维稠密谱问题（`2(n−1)` 复矩阵），
   涡量壁面用 Thom 公式、温度板面 Dirichlet、`ψ` 双板 Dirichlet。
   旧路径 `critical_rayleigh` 改为**任何情况下都回填全部比较键**（`deviation_rel/ra_c_reference/status`），
   非括根时 `status='not_bracketed'`；`RA_C_LINEAR_THEORY` 保留为兼容别名并注明属 Bénard 配置。
   另在 `rb_continuum_stability` docstring 补「1/Pr 时间单位陷阱」两行说明（见 ⑤）。
5. 测试：`tests/test_conjugate_thermal_linear_stability_configs.py`（10 项，16.7 s 全绿）。

## ③ 计算结果（全部由脚本注入）

**旧路径（侧加热方腔）默认调用**

| n | `ra_c` | `status` | `σ(500)` | `σ(4000)` |
|---|---|---|---|---|
| 8 | None | not_bracketed | −9.733 | −9.087 |
| 12 | None | not_bracketed | −9.808 | −9.482 |
| 16 | None | not_bracketed | −9.835 | −9.644 |

**侧加热方腔 `σ(Ra)`（n=12）**：500 → −9.808；1000 → −9.793；**1707.76 → −9.754**；5000 → −9.288；
1e4 → −6.761；1.45e4 → −0.073；3e4 → +19.433。→ 中性点 `Ra_c = 14551.7`（n=12），
**= 8.52 × 1707.7597**。

**Bénard 配置网格收敛（a = 3.11632）**

| n | `Ra_c` | 相对误差 | `σ(Ra_c)` | 误差比 |
|---|---|---|---|---|
| 16 | 1657.456 | −2.946e−2 | −2.4e−6 | — |
| 32 | 1694.814 | −7.581e−3 | −4.3e−5 | 3.886 |
| 64 | 1704.507 | −1.905e−3 | +2.8e−5 | 3.980 |

Richardson(16,32,64) = **1707.895**，相对误差 **+7.93e−5**。
波数扫描（n=32）：2.8 → 1721.503；3.0 → 1698.112；**3.11632 → 1694.814（最小）**；3.25 → 1699.407；3.5 → 1729.808。

**跨实现交叉校核（n=64 vs `rb_continuum_stability` M=100，`Pr=0.71`）**

| Ra | 本模块 σ（α/H²） | rb_continuum σ（ν/H²） | `Pr×σ_rb` | 换算后相对差 | 不换算的相对差 |
|---|---|---|---|---|---|
| 1500 | −1.4116 | −2.0058 | −1.4241 | +0.88% | +42.1% |
| 2500 | +4.8365 | +6.7910 | +4.8216 | −0.31% | +40.4% |
| 3500 | +9.9501 | +13.9905 | +9.9332 | −0.17% | +40.6% |

**图**：`artifacts/m16921_sigma_vs_ra_configurations.png`（两配置 σ(Ra) 对比 + 双中性点）、
`artifacts/m16921_benard_grid_convergence.png`（Ra_c vs h² 二阶收敛 + Richardson 点 + Chandrasekhar 线）。

## ④ 误差判定

| 关键量 | 基准 | 误差 | <3%？ |
|---|---|---|---|
| Bénard `Ra_c`（n=16） | 1707.7597 | −2.95% | 是（边缘） |
| Bénard `Ra_c`（n=32） | 1707.7597 | −0.76% | 是 |
| Bénard `Ra_c`（n=64） | 1707.7597 | −0.19% | 是 |
| Bénard `Ra_c` Richardson 外推 | 1707.7597 | **+0.0079%** | 是 |
| 最危险波数（n=32） | 3.11632 | 0（命中扫描点） | 是 |
| σ 跨实现一致性（换算后） | rb_continuum M=100 | ≤0.88% | 是 |
| 侧加热配置中性点 | 无登记文献锚 | — | **不适用（WITHHELD）** |

## ⑤ 注意点（坑）

- **锚点张冠李戴**：`1707.76` 是无限大 RB 层（重力 ∥ 梯度）的刚-刚临界值；侧加热方腔是
  「重力 ⟂ 梯度」的 Hadley 型问题，两者只差边界条件与基态方向，锚点却相差 8.5 倍。
- **时间单位陷阱**：`rb_continuum_stability` 用粘性速度尺度（σ 单位 ν/H²、温度方程含 `1/Pr`），
  本模块用热扩散尺度（σ 单位 α/H²）。直接比较两模块 σ 会**系统性差 1/Pr（40–42%）**，且因两侧都
  「有量纲合理」而极难察觉。已在两侧 docstring 互相登记。
- **单侧稳定性判据会骗人**：侧加热方腔 σ 在 Ra=500…1707 几乎不动（−9.81…−9.75），
  若只看「Ra 增大 σ 增大一点」就可能误判「接近临界」；必须真正括根。
- **API 静默失败**：旧 `critical_rayleigh` 未括根时不回填 `deviation_rel`，下游取键即 `KeyError`
  （本单复现）。已改为恒回填 + `status` 字段。
- **局限**：Bénard 路径为二阶格式，达到 1e-4 级精度需 n≥64（本机 load≈90，n=64 单次求解约 12 s）；
  未做 Chebyshev/谱方法，未扩展到三维或可压缩；侧加热配置的文献锚点本仓未登记，故只登记测量值
  与网格依赖（n=8 → 1.304e4，n=12 → 1.455e4，约 10% 网格差），**不冒充**文献校验。

## ⑥ 复现

```bash
PYTHONPATH=src python scripts/m16921_cavity_stability_anchor_audit.py     # ~56 s, 出 JSON+两图
PYTHONPATH=src pytest tests/test_conjugate_thermal_linear_stability_configs.py -q  # 10 passed, ~17 s
```

## ⑦ 变更清单

- `src/tensorlbm/conjugate_thermal/stability.py`：重写模块 docstring（诚实化，含配置警告与单位说明）；
  新增 Bénard 路径 6 个函数 + `richardson`；`critical_rayleigh` 恒回填比较键与 `status`；
  新增常量 `RA_C_BENARD_RIGID_RIGID/A_C_BENARD_RIGID_RIGID/SIDE_HEATED_RA_C_MEASURED_N12`。
- `src/tensorlbm/rb_continuum_stability.py`：docstring 增「1/Pr 时间单位陷阱」说明（仅注释）。
- `tests/test_conjugate_thermal_linear_stability_configs.py`：新增 10 项回归。
- `scripts/m16921_cavity_stability_anchor_audit.py`、`artifacts/m16921_*`：审计工装与产物。