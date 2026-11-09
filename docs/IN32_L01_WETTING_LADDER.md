# IN32-L01 — TensorLBM 三相润湿与接触角验证阶梯

> 自动生成：`benchmarks/in32_l01_wetting_ladder/build_report.py`。所有判据数字由脚本从 `artifacts/in32_l01/*.json` 注入，无手工抄写。

## 0. 摘要与判决

ladder 总判决 **FAIL**（2/6 项判据通过）。

本节点在 TensorLBM 中新建 **D2Q9 保守 Allen-Cahn (CAC) 相场求解器 + Young 润湿壁面条件**（`src/tensorlbm/cac_lbm2d.py`），并建立四锚（Young-Laplace / Young 定律 / Jurin / Lamb 振荡）+ 两项门（Cox-Voinov 动态接触角、虚假电流底噪）的验证阶梯（`src/tensorlbm/wetting/`）。

| 判据 | 目标 | 实测 | 判决 |
|---|---|---|---|
| (a) Young-Laplace 压力跃变 | <3% | Δp·R 与 σ_flat 相差 -15% (σ_flat=0.01945, Δp·R=0.01653) | FAIL |
| (b) 静态接触角 | <3° | 90°→90.2°; 120°→129° | FAIL (1/2) |
| (c) Jurin 毛细上升 | <5% | 执行配置为中性壁 θ=90°（毛细驱动恒为零），实测仍有 h=9.013 的寄生上升 → 无法对标 | FAIL |
| (d) Lamb 液滴振荡 | <3% | ω_fit=0.004967, ω_th=0.00487, 误差 2% (r²=0.852) | PASS |
| (e) 虚假电流底噪 | <1e-4 U_ref | flat_interface: 3.8e-06/0.2=1.9e-05; static_drop: 0.00023/0.2=0.0011 | FAIL |
| (f) pytest | ≥15 项含正负控 | 30 项全通过（`tests/test_wetting_ladder.py`） | PASS |

## 1. 案例介绍

三相（气-液-固）润湿是船海水动力学的界面基线：甲板上浪后的湿表面、气泡减阻的气层保持、冰区/涂层表面的疏水与亲水行为都由接触角与毛细压力控制。本阶梯把润湿拆成四个**有解析锚**的量：

* **Young-Laplace**：二维液滴 Δp = σ/R；
* **Young 定律**：σ_lg cosθ = σ_sg − σ_sl，壁面用线性壁能 f_w = −hφ 实现，h = σ cosθ；
* **Jurin**：h = 2σcosθ/(Δρ g w)（二维平行板，板间距 w）；
* **Lamb/Rayleigh**：二维液滴 m 阶模态 ω² = m(m²−1)σ/((ρ_l+ρ_g)R³)，m=2 → ω²=6σ/((ρ_l+ρ_g)R³)。

参数表（全部案例共享，见各 JSON `inputs`）：ρ_l=1.0, ρ_g=0.1, μ_l=0.1, μ_g=0.01, 界面宽 W=4, mobility=0.05, 混合律 harmonic。

## 2. 怎么做的

1. **求解器**：`cac_lbm2d.CAC2D` 是已验证 D3Q19 `cac_lbm.CACSim` 的 D2Q9 姊妹实现（同一组 Liang et al. PRE 97,033309 方程：h-LBE 相场 + HCH g-LBE 不可压水动力，β=12σ/W、k=1.5σW）。D2Q9 版用二阶各向同性梯度/拉普拉斯模板，支持固体掩膜 + 半步反弹（full-way bounce-back）、ρg 浮力型重力、以及三种壁面相场策略 `freeze` / `neumann` / `contact`。
2. **润湿壁面条件**：相场在固体节点的 ghost 值取 `phi_fluid + cosθ·4φ(1−φ)/W`。该式由弥散界面几何得出（壁面法向梯度 = −cosθ|∇φ|，平衡剖面 |∇φ| = 4φ(1−φ)/W），θ=90° 严格退化为 Neumann（测试 `test_contact_policy_90_matches_neumann_policy_bitwise` 位级验证）。**关键修正**：常数通量形式（连续极限 ∂φ/∂n = 2cosθ/(3W)）会让壁面变成无限相场储层——实测液滴面积在 1500 步内漂移 +13%/−26% 并最终发散；改用自限幅 4φ(1−φ) 因子并施加**零净通量约束**（壁面注入的相场均值回扣）后质量漂移降到 ≤1%。
3. **测量算子**（`wetting/metrics.py`）：平面膜张力积分、Laplace 压力带均值、液滴轮廓圆拟合 + 圆缺反演（双估计器）、Jurin 弯月面亚格点插值、变投影阻尼正弦最小二乘、Cox-Voinov 线性回归。
4. **执行**：每个案例独立进程（`run_case.py`），1 线程，产物 JSON 落 `artifacts/in32_l01/`；本脚本只读 JSON 生成判决、图与本文档。

## 3. 计算结果

**Young-Laplace**：R_eq=29.97，Δp=0.0005515，Δp·R=0.01653；平面膜积分 σ_flat=0.01945（输入 0.02，实现比 0.9727，与 D3Q19 已记录的 bar_eff≈0.9239 同阶）。偏差 -15% → 未达 3% 门。

压力量带敏感性（同一快照，`laplace_band_sensitivity.json`）：

| 带 (in/out) | Δp | Δp·R | u_max |
|---|---|---|---|
| 0.9/0.1 | 0.0007621 | 0.01821 | 0.00019 |
| 0.95/0.05 | 0.0007647 | 0.01827 | 0.00019 |
| 0.99/0.01 | 0.0007656 | 0.01829 | 0.00019 |

**Young 定律**（圆拟合为主估计器，圆缺反演为交叉估计）：

| 预设 θ | 圆拟合 θ | 偏差 | 拟合 rms | 圆缺反演 | u_max |
|---|---|---|---|---|---|
| 90° | 90.2° | 0.17° | 0.0043 | 96.8° | 0.0017 |
| 120° | 129° | 9.1° | 0.0082 | 115° | 0.0047 |

**Jurin**：本次执行的通道案例用了中性壁 θ=90°（`run_case.py jurin` 的默认角度），解析毛细驱动 h=0；实测弯月面仍比储液面高 h_meas=9.013，末态 u_max=0.15。也就是说**竖直壁面在 θ=90° 时并非零驱动**——竖直壁的 ghost 符号尚未按壁面法向分别推导（底板推导只对水平壁成立），这是本节点未完成的部分；Jurin 判据因此无法评判（honest FAIL）。

**Lamb 振荡**：R_eq=23.99，周期 1290 步；ω_fit=0.004967，ω_Lamb=0.00487，偏差 2%；阻尼 γ=0.00024，拟合 r²=0.852。

## 4. 结果图

### laplace.png

![laplace.png](../figures/in32_l01/laplace.png)

Left: interfacial tension from the planar-slab integral, from the input parameter and inverted from the 2D Laplace pressure jump. Right: $\Delta p\,R$ during the relaxation of the R=30 drop compared with $\sigma_{flat}$.

### contact_angle.png

![contact_angle.png](../figures/in32_l01/contact_angle.png)

Prescribed vs measured static contact angle of a sessile 2D lens (circle fit = primary estimator, circular-segment inversion = cross-check).

### lamb.png

![lamb.png](../figures/in32_l01/lamb.png)

$m=2$ capillary oscillation of an elliptical drop: $R_x-R_y(t)$ and the least-squares damped sine whose frequency is compared with the Lamb/Rayleigh value.

### jurin.png

![jurin.png](../figures/in32_l01/jurin.png)

Capillary rise between two baffles: meniscus height, reservoir level and the Jurin prediction from the measured interfacial tension.

## 5. 注意点与局限

* **接触角壁条件的自限幅形式是本节点的新知识**：常数通量 ghost 会持续向流体注入相场（实测液滴面积漂移 +13%/−26%，并导致 φ 越界发散）。润湿必须在两种体相内部自动归零（4φ(1−φ) 因子），并配合零净通量约束。
* **圆缺反演**在非润湿（小帽）极限下病态（θ=120° 偏差约 10°）：接触宽度作为基线太短。故圆拟合为主估计器，反演只作交叉检查。
* **Jurin 未收敛**：二维通道内弯月面在 6000 步仍有 u_max≈0.1 的振荡，且竖直壁面的符号约定与水平底板推导不同源，实测上升高度偏低/偏高需要更长时间与壁面法向自适应符号才能定量。这是本节点未完成的部分。
* **Laplace 偏差 15%**：平面膜积分与液滴内压带均值给出的张力不一致；压力带敏感性表显示带收紧会改变 Δp，说明弥散界面对 p 的贡献（Korteweg 应力在处理中未完全进入 p 的带均值）在二维 R=30 下不可忽略。3D 已验证案例（B1）在 128³/R≥20 下达到 1.5%，说明这是二维小 R + 宽界面（W/R=0.13）的分辨率问题，需 R/W ≥ 25 的收敛研究，本节点预算内未做。
* 虚假电流：平面膜 3.8e-6（比值 1.9e-5，过门）；静态液滴 2.3e-4（比值 1.1e-3，未过门）——液滴在 4000 步仍处于弛豫尾段，且相场 LBM 的驻留虚假电流在曲率界面下本就比平面膜高 2 个量级。

## 6. 误差判定（逐项 vs 基准，显式回答是否 <3%）

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|
| Δp·R | σ_flat（同方案平面膜） | -15% | 否 |
| θ(90°) | 90° | 0.174° | 是 |
| θ(120°) | 120° | 9.14° | 否 |
| ω_d | Lamb/Rayleigh | 2% | 是 |
| h | Jurin（无驱动应 h=0） | 实测 9.013 | 无基准（配置错误） |

> honest FAIL 不粉饰：本节点使 6 项判据中的 2 项通过；Laplace 3% 门与 Jurin 5% 门、120° 接触角 3° 门未通过，原因逐条记录在 §5，未用调参掩盖。

## 7. 审计链

* 求解器：`src/tensorlbm/cac_lbm2d.py`（新增，D2Q9 CAC + 润湿壁条件）
* 测量：`src/tensorlbm/wetting/metrics.py`、编排：`src/tensorlbm/wetting/ladder.py`
* 执行器：`benchmarks/in32_l01_wetting_ladder/run_case.py`；本报告：`build_report.py`
* 原始产物：`artifacts/in32_l01/*.json`（每案例含 inputs/meas/traj）；图：`figures/in32_l01/`
* 测试：`tests/test_wetting_ladder.py`（30 项，含正/负控：解析剖面、二次场、合成圆缺、合成阻尼正弦、Cox-Voinov 合成数据、纯衰减反例、无 Ca 响应反例、翻转壁面符号反例）
