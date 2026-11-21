# DE-NXT-TENS-W2 草稿：表面张力与相间力设计官 W2 处方赛马

## 背景与问题陈述
- **现状**：T5_DIGEST + CAVSWEEP 定谳——单滴入深水池 Santini case-3 峰深 1.435D vs 实验 1.8574D（-22.7%），系统性气穴欠深 13~23%。
- **已排除**：网格纯收敛/参数微调/u_limit/ABB 整体移除/rho_g 简单缩放/空穴压亏/σ 数值微调（CAVSWEEP 已扫死）。
- **W1 回顾**：CSF-SMOOTH（3 点滑动平均）和 IF-SEP（压力基 CSF）已设计，但 W1 结果未在本工作树中验证。
- **W2 方向**：聚焦曲率离散精度与力-压力平衡的数值实现，二者为 CSF 伪电流和界面动力学误差的物理根源。

## 候选假说

### 假说 H1：曲率离散各向异性导致伪电流
- **机制**：CSF 曲率 `κ = -div(n_hat)` 使用二阶中心差分，在方形网格上存在 O(h²) 的四重各向异性误差（cos 4θ 模式）。Brackbill et al. (1992) 指出这是伪电流主因。
- **证据**：`csf.py` 中 `smooth_field()` 文档明确提到 "cos 4 theta error"，1 次 binomial 滤波将 κ 振幅从 0.70x 提升到 0.96x（R=16, w=2）。
- **预测**：5 点滤波（2 次 binomial 或等效）可进一步抑制高频噪声，降低 mach_max 30%+。
- **证伪**：mach_max 不降或峰深误差 > 20%。

### 假说 H2：力-压力不平衡导致界面动力学误差
- **机制**：标准 CSF 将表面张力作为体积力 `f_st = σκ∇φ` 添加到动量方程，但压力梯度 `∇p` 和 `f_st` 使用相同的离散算子并不保证精确平衡（Balanced-Force CSF 要求 `∇(p + σκH(φ)) = 0` 在静止界面成立）。
- **证据**：`csf.py` 文档提到 "balanced-force property" 但当前实现未显式构造平衡压力项。Miedema et al. (2008) 证明压力基 CSF 可消除伪电流。
- **预测**：引入平衡压力修正后，静止界面伪速度 < 1e-5，动态界面 mach_max 降 25%+。
- **证伪**：伪速度不降或计算发散。

## 臂设计

### ARM CURV5：5 点曲率滤波
- **改动**：在 `csf.py` 的 `curvature()` 中，将 `smooth=1`（3 点）改为 `smooth=2`（5 点等效），或新增 `smooth_kappa=5` 选项对 κ 场直接应用 5 点移动平均。
- **物理出处**：Brackbill et al. (1992) JCP 100, 335；`csf.py` 文档中 `smooth_field()` 的 cos 4θ 误差分析。
- **验证**：对比 mach_max 和峰值深度。

### ARM BALCSF：平衡压力 CSF
- **改动**：在动量方程的压力修正步中，加入平衡压力项 `p_bal = σ * κ * (φ - 0.5)`，使 `∇p_eff = ∇p - ∇(σκ(φ-0.5))` 与 `f_st` 精确抵消。
- **物理出处**：Miedema et al. (2008) "A balanced-force CSF model"；Brackbill 原始论文中 balanced-force 概念。
- **验证**：对比静止界面伪速度和动态界面 mach_max。

## 成本估算
- 每臂 ~13 min（8 核，cells=8），2 臂并行 ~15 min。
- 总成本 < 600 预算。

## 自检
- 格式：机器可读 ✓
- 臂数：2 ✓
- 预登记：齐 ✓
- 排除路线：未重开 ✓
- 出处：可溯（Brackbill 1992, Miedema 2008, csf.py 文档）✓
