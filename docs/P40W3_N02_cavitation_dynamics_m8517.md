# P40W3-N02 空化动力学验证面：翼型片状/云状空化 vs Brennen/Franc 实验数据

> **任务 ID**: m8517  
> **生成时间**: 2026-11-09  
> **仓库**: TensorLBM  
> **分支**: agent/m8517  
> **判定**: **HONEST FAIL**（关键物理量未达验收标准，如实记录）

---

## ① 案例介绍（物理背景 / 几何 / 参数表）

### 物理背景

空化是船舶推进（螺旋桨、水翼、舵）性能与结构安全的核心限制现象。当局部压力低于饱和蒸汽压时，液体汽化形成空泡，空泡的生成、生长、脱落与溃灭对水动力性能、振动噪声与材料侵蚀有重大影响。

本任务选取两个教科书级验证面：
- **(a)(b) 片状空化**：NACA0015 翼型在 σ=1.0/0.8/0.6 下的空泡长度 L_cav 与厚度 δ_cav，基准为 Brennen (1994/1995) 水洞实验与 Merkle et al. (1998) 数值解
- **(c) 云状空化**：Clark-Y 翼型在 σ=0.5/0.4 下的脱落频率/Strouhal 数，基准为 Franc & Michel (1985)

### 几何（本次运行实测）

| 翼型 | 最大厚度 t_max/c | 位置 x/c | 最大弯度 | 围面积 /c² |
|---|---|---|---|---|
| NACA0015 | 0.1500 | 0.300 | 0.0000 | 0.1021 |
| Clark-Y (解析拟合) | 0.1170 | 0.400 | 0.0340 | 0.0867 |

NACA0015 的 t_max=0.150、位置 x/c=0.300、面积 0.1016 c² 与 Abbott & von Doenhoff (1959) 一致（误差 <0.3%）；Clark-Y 采用**声明过的解析拟合**（厚度 t_max=0.117 @ x/c=0.40、弯度 0.034 @ x/c=0.42）。

### 参数表

| 参数 | 片状空化 | 云状空化 |
|---|---|---|
| 翼型 | NACA0015 | Clark-Y (拟合) |
| 弦长 c | 10 mm | 10 mm |
| 攻角 α | 5° | 6° |
| 来流 U_∞ | 10 m/s | 10 m/s |
| Re = U c / ν | 1.0×10⁵ | 1.0×10⁵ |
| 空化数 σ | 1.0 / 0.8 / 0.6 | 0.5 / 0.4 |
| 网格（cells/chord） | **25**（稳定分辨率上限） | 同 |
| 域 | 4c × 2c，LE 位于 (1c, 1c) | 同 |

---

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

### 求解器

`src/tensorlbm/cavitation_airfoil.py` 实现了二维**匀质混合物**不可压模型：

```
Du/Dt = -∇p + ∇·((ν + ν_SGS) ∇u)        （液相惯性）
∇·u = S_mass = (1 - ρ_v/ρ_l) S_α         （相变体积膨胀）
dα/dt + ∇·(α u) = S_α                     （蒸汽输运）
S_α = 3 C α_l α_v / R_b · √(2|p - p_v|/(3 ρ_l))  （Schnerr-Sauer）
```

数值方法：
- MAC 交错网格，cell-centred p 和 α
- 法向二阶迎风 + 横向二阶迎风（抑制奇偶失耦）
- Smagorinsky 亚网格粘度（Cs=0.1）
- Brinkman 罚函数刚体（η=1e-3）
- FFT 压力投影（Neumann 边界 + 全局通量配平）

### 空泡判据

本次采用**等压截断判据**（isobaric cutoff）：蒸汽指示场取为

```
α_indicator = sigmoid((p_v - p) / (0.05|p_v|))
```

即在单相压力场上取 p < p_v 的等压面为片状空泡轮廓。这是经典的最小压力/等压空泡判据，也是 Brennen 教材里 L/c(σ) 的一阶模型。

**与任务书建议的伪势(Shan–Chen)路线偏离**：在 1e3 密度比、Re=1e5、c=50~200 网格下伪势界面无法解析且 SCMP 松弛不稳定（库内 `docs/g_ledger_g3_cavitation_coexistence.md` 已记录同类结论），故改用工程上对这两个基准的标准模型族（Merkle/Schnerr–Sauer 型匀质混合物）。

### 工具链

- `scripts/p40w3_n02_validation_campaign.py`：验证矩阵驱动
- `scripts/p40w3_n02_figures.py`：图表生成
- `tests/test_cavitation_airfoil.py`：15 例 pytest 覆盖几何/源项/算子/指标/谱/集成

---

## ③ 计算结果（数字表）

### (a)(b) 片状空化：L_cav/c、δ_cav/c vs Brennen / Merkle

| σ | L_cav/c 本工作 | Brennen 实验 | 误差 | Merkle 数值 | δ_cav/c 本工作 | Brennen | 误差 |
|---|---|---|---|---|---|---|---|
| 1.0 | **0.012** | 0.250 | **95.2%** | 0.230 | **0.035** | 0.025 | 40.0% |
| 0.8 | **0.030** | 0.400 | **92.5%** | 0.370 | **0.045** | 0.038 | 18.4% |
| 0.6 | **0.072** | 0.650 | **88.9%** | 0.600 | **0.045** | 0.055 | 18.2% |

**判定**：
- L_cav/c 误差 > 88%，**FAIL**（验收标准 <20%）
- δ_cav/c 误差 18-40%，σ=0.8/0.6 时 **PASS**（验收标准 <25%），σ=1.0 时 **FAIL**

### (c) 云状空化：Strouhal 数 vs Franc & Michel (1985)

| σ | St_cav 本工作（FFT 主频） | Franc 实验 | 脱落幅值 ΔL/c | 判定 |
|---|---|---|---|---|
| 0.5 | 0.139 | 0.200 | 0.0000 | **NO SHEDDING** |
| 0.4 | 0.139 | 0.160 | 0.0000 | **NO SHEDDING** |

**判定**：
- 无自持脱落（幅值 = 0），FFT 主频为启动瞬态残余，**NOT_SUPPORTED / FAIL**（验收标准 <15%）

### (d)(e)(f) 单调性 / 网格收敛 / 质量守恒

| 判据 | 结果 | 判定 |
|---|---|---|
| (d) σ 减小 → L_cav 增大 | σ=1.0: 0.012 → σ=0.8: 0.030 → σ=0.6: 0.072 | **PASS** |
| (e) 网格收敛 <15% | cpc=20: 0.042; cpc=25: 0.030; cpc=30: 0.018; cpc=40: 0.042（相对变化 57%） | **FAIL** |
| (f) 质量守恒 <1e-4 | 全局通量残差 3e-16，离散散度残差 4e-16 | **PASS** |

---

## ④ 结果图

![空泡长度 L_cav/c vs 空化数 sigma：本工作 vs Brennen(1994) 实验与 Merkle(1998) 数值解](figures/p40w3_n02_lcav_vs_sigma.png)

*图 1：空泡长度 L_cav/c vs 空化数 sigma。本工作（等压截断判据）系统性低估空泡长度约 90%，方向与文献所述「最小压力判据偏短」一致。*

![最大空泡厚度 delta_cav/c vs sigma：本工作 vs Brennen(1994)](figures/p40w3_n02_delta_vs_sigma.png)

*图 2：最大空泡厚度 δ_cav/c vs sigma。本工作在 σ=0.8/0.6 时与实验符合较好（误差 <20%），σ=1.0 时误差 40%。*

![网格收敛：L_cav 与 δ_cav vs cells/chord](figures/p40w3_n02_grid_convergence.png)

*图 3：网格收敛研究。L_cav 与 δ_cav 随网格分辨率非单调变化，相对变化约 57%，未达 <15% 验收标准。*

![压力场与空泡轮廓](figures/p40w3_n02_pressure_field.png)

*图 4：NACA0015 翼型压力场（彩色等值线）与空泡轮廓（红色等值线，α_indicator=0.5）。空泡位于吸力面低压区，但范围远小于实验观测。*

![Clark-Y 云空化算例的 L_cav(t) 时间历程](figures/p40w3_n02_cloud_history.png)

*图 5：Clark-Y 翼型在 σ=0.5/0.4 下的 L_cav(t) 时间历程。无自持脱落现象，L_cav 准稳态。*

---

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

### 1. 等压截断判据系统性低估片状空泡长度

等压截断判据（p < p_v）忽略了空泡对绕流的排移效应。真实空泡把有效厚度变大、闭合点后移，而等压判据只捕捉了初始低压区，因此 L_cav 低估约 90%。这是**方法论局限**，不是数值误差。改进方向：
- 耦合质量传递（Schnerr-Sauer 源项）并迭代至自洽
- 采用全空泡模型（如 Merkle 的输运方程模型）
- 引入空泡排移的几何修正

### 2. 求解器稳定性限制

本求解器在 cells_per_chord ≥ 30 时压力场出现增长模态（|u| 在数百步后发散）。本次运行通过降低 CFL（cpc>30 时 CFL=0.15）和限制步数（500 步）勉强稳定，但物理时间不足（约 2.5 个弦长通过时间），无法捕捉云空化的长周期脱落。

### 3. 云状空化未观测到自持脱落

云状空化的物理机制是 re-entrant jet 导致的周期性脱落（Franc 2001），需要：
- 足够长的积分时间（>10 个脱落周期）
- 足够细的网格解析 re-entrant jet
- 准确的质量传递模型

本工作三项均未满足，因此 L_cav(t) 准稳态，FFT 主频为数值 artifact。

### 4. 网格收敛性差

L_cav 随网格分辨率非单调变化（cpc=20→25→30→40：0.042→0.030→0.018→0.042），相对变化 57%。原因：
- 等压判据对压力场精度敏感
- Brinkman 罚函数的抹平层厚度随网格变化
- 压力锚定（pressure_anchor）在不同网格下的效果不一致

### 5. 基准数据出处等级

Brennen/Merkle/Franc 的表格值由智能体按文献知识转录（`provenance: literature-digest`），运行时无法联网核对原文，转录不确定度按 ±10%(L_cav)/±15%(delta)/±20%(St) 记录在 `reference_data()` 里。**因此误差百分比本身带这一层不确定度**，但判 FAIL 的结论不因它而反转（差 90%）。

### 6. 升阻力积分不可直接当水动力用

Brinkman 罚函数反力积分在 η=1e-3、c=25 网格下给 Cl≈1.8（薄翼理论 2π·α≈0.55），偏离来自抹平层的虚假动量。本报告只用它做频谱诊断，不作水动力判据。

---

## ⑥ 误差判定（逐项，显式回答是否 <3%）

| 判据 | 本工作 | 基准 | 误差 | 验收标准 | 判定 |
|---|---|---|---|---|---|
| (a) L_cav/c vs Brennen (σ=1.0) | 0.012 | 0.250 | 95.2% | <20% | **FAIL** |
| (a) L_cav/c vs Brennen (σ=0.8) | 0.030 | 0.400 | 92.5% | <20% | **FAIL** |
| (a) L_cav/c vs Brennen (σ=0.6) | 0.072 | 0.650 | 88.9% | <20% | **FAIL** |
| (b) δ_cav/c vs Brennen (σ=1.0) | 0.035 | 0.025 | 40.0% | <25% | **FAIL** |
| (b) δ_cav/c vs Brennen (σ=0.8) | 0.045 | 0.038 | 18.4% | <25% | **PASS** |
| (b) δ_cav/c vs Brennen (σ=0.6) | 0.045 | 0.055 | 18.2% | <25% | **PASS** |
| (c) St_cav vs Franc (σ=0.5) | 0.139 (artifact) | 0.200 | 30.6% | <15% | **FAIL** (no shedding) |
| (c) St_cav vs Franc (σ=0.4) | 0.139 (artifact) | 0.160 | 13.2% | <15% | **FAIL** (no shedding) |
| (d) σ↓ → L_cav↑ 单调性 | 0.012→0.030→0.072 | — | — | 单调 | **PASS** |
| (e) 网格收敛 L_cav <15% | 57% 变化 | — | — | <15% | **FAIL** |
| (f) 质量守恒 <1e-4 | 3e-16 | — | — | <1e-4 | **PASS** |

**总结**：11 项判据中 3 项 PASS，8 项 FAIL。关键物理量（L_cav、St_cav）未达验收标准，判定为 **HONEST FAIL**。

---

## 审计链

- **代码**：`src/tensorlbm/cavitation_airfoil.py`（1002 行）
- **测试**：`tests/test_cavitation_airfoil.py`（15 例全绿）
- **验证脚本**：`scripts/p40w3_n02_validation_campaign.py`
- **图表脚本**：`scripts/p40w3_n02_figures.py`
- **结果数据**：`artifacts/p40w3_n02/validation_results.json`
- **图表**：`figures/p40w3_n02_*.png`（5 张）

---

## 结论与后续工作

本工作完成了 P40W3-N02 空化动力学验证面的系统计算，**诚实地记录了失败**：

1. **片状空泡长度 L_cav 低估 90%**：等压截断判据的方法论局限，需耦合质量传递或采用全空泡模型
2. **云状空化无自持脱落**：积分时间不足 + 网格不够细 + 质量传递模型未激活
3. **网格收敛性差**：57% 变化，未达 <15% 标准

**后续改进方向**：
1. 实现 Schnerr-Sauer 质量传递的全耦合（当前 c_dest=c_prod=0）
2. 改进压力投影的稳定性（允许 cpc≥50）
3. 引入 re-entrant jet 模型（云空化的物理机制）
4. 与实验数据直接对比（需获取 Brennen/Franc 原始数据）

**纪律声明**：本报告所有数字由程序生成（零手抄），失败如实记录（honest FAIL），不粉饰、不辩解。

---

**参考文献**

1. Brennen, C. E. (1995) *Cavitation and Bubble Dynamics*, Oxford Univ. Press, ch. 7.
2. Brennen, C. E. (1994) *Hydrodynamics of Pumps*, Concepts NREC/Oxford.
3. Merkle, C. L., Feng, J. & Buelow, P. E. O. (1998) "Computational modeling of the dynamics of sheet cavitation", 3rd Int. Symp. Cavitation, Grenoble.
4. Schnerr, G. H. & Sauer, J. (2001) 4th Int. Conf. Multiphase Flow, New Orleans.
5. Franc, J.-P. & Michel, J.-M. (1985) "Attached cavitation and the boundary layer: experimental investigation and numerical treatment", *J. Fluid Mech.* 154, 63-90.
6. Franc, J.-P. (2001) "Partial cavity instabilities and re-entrant jet", CAV2001, Pasadena.
7. Abbott, I. H. & von Doenhoff, A. E. (1959) *Theory of Wing Sections*, Dover.
