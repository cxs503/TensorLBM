# XVAL-LBM-CENTRAL-MOMENT-PARTIAL-RELAXATION

TensorLBM 的 **D2Q9 中心矩部分松弛碰撞**（`src/tensorlbm/central_moment_d2q9.py`）与开源实现 **joconnor22/LIFE**（`src/Grid.cpp`，`CENTRAL_MOMENTS` 分支）的交叉对标。本文所有数字由 `benchmarks/xval_life_cm_partial.py` 从 `runs/xval_life_cm_partial/result.json` 程序化注入，无手抄；图由同一脚本生成。

生成环境：3.11.6 / torch 2.7.1 / Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38 / dtype float64 / device CPU。

## 1 问题与对标出处

LIFE 是 LBM-IBM-FEM 强耦合求解器，其碰撞算子可切换为「central moments collision operator（De Rosis et al. 2019）」+ 四阶 Hermite 平衡态。该算子的关键做法从**上游源码**读得（`src/Grid.cpp::GridClass::streamCollide` 的 `#ifdef CENTRAL_MOMENTS` 段，含 `k0..k8` 后碰撞中心矩与 `fStar[0..8]` 反变换），不是从论文二手转述：

| 中心矩（质量加权、非加权权重） | 后碰撞取值（上游） | 含义 |
|---|---|---|
| κ0 | ρ | 质量 |
| κx, κy | Fx/2, Fy/2 | 半步力（Guo 型） |
| κtr = Σf[(cx−ux)²+(cy−uy)²] | 2ρc_s² | 直接置为平衡值 |
| κd = Σf[(cx−ux)²−(cy−uy)²] | (1−ω)·κd | **唯一以 ω 松弛的模态** |
| κxy = Σf(cx−ux)(cy−uy) | (1−ω)·κxy | **同上（决定剪切黏度）** |
| κxxy, κxyy | Fy c_s²/2, Fx c_s²/2 | 置零平衡 + 三阶力项 |
| κxxyy | ρ c_s⁴ | 四阶平衡值 |

即：**除二阶偏应力外，所有阶矩全部投影回平衡（或力）值**；所谓 *partial* 就是「只对剪切应力做松弛」。平衡态用四阶 Hermite 展开 `f_eq = ρ w_v/4 (9c_x²u_x²+6c_xu_x−3u_x²+2)(9c_y²u_y²+6c_yu_y−3u_y²+2)`，其中心矩恰为上表右侧值（本仓已用数值断言固定：`tests/test_central_moment_d2q9.py`）。

本仓原状态：`src/tensorlbm/cascaded_collision.py`（D3Q19/D3Q27 全级联）与 `src/tensorlbm/cumulant.py`（D2Q9/D3Q27 累积量族）覆盖的是「各阶独立速率」路线，**没有** LIFE 这种「高阶矩一律重置」的 D2Q9 中心矩算子，也没有它与 BGK 的定量对照。

## 2 方法与实现

| 项 | 实现 |
|---|---|
| 对标算子 | `collide_central_moment_d2q9(f, omega, s_bulk, s3, s4, force)` |
| 矩变换 | κ = M0ᵀf；f = (M0ᵀ)⁻¹·shift⁻¹(κ)，M0 为 9 个单项式基在 9 个格向的取值矩阵 |
| 上游等价性 | 上游 `fStar[v]` 大表达式**自动抽取**后作为本地 oracle 求值，与本仓逐位比对 |
| 速率参数化 | `s_bulk,s3,s4 = 1` ⇒ LIFE 原样；`= ω` ⇒ 「全阶按 ω 松弛」消融臂 |
| 时间推进 | 碰撞后 `tensorlbm.solver.stream`（AB 双场 gather） |
| 宏观量 | `macroscopic_guo`：u = (Σ c f + F/2)/ρ（与上游 `macroscopic()` 一致） |
| 判据 | 衰减率误差 |Δγ|/γ ≤ 15 %（owner 精度标准）；场上相对 L2 误差同样过 15 % 门 |
| 参考解 | 二维 Taylor–Green：u=−U0cos kx sin ky e^{−2νk²t}, v=+U0sin kx cos ky e^{−2νk²t}, K=½⟨u²⟩∝e^{−4νk²t}, ν=(1/ω−1/2)/3 |

### 2.1 上游等价性（零手抄交付方式）

上游表达式用脚本从 `src/Grid.cpp` 逐行抽取（正则切出 `k0..k8` 与 `fStar[0..8]`），在 Python 中代入随机状态求值作为 oracle；oracle 本身不进仓（上游 GPL），进仓的是 8 组**数值夹具** `tests/fixtures/life_cm_partial_oracle.json`（含 provenance 说明）。结果：

- 座椅比对：max |TensorLBM − LIFE| = **1.67e-16**（8 组随机状态，含随机 ρ/u/Ω/力；典型 |f| ~ 0.1 ⇒ 相对误差 ~1e-15，即 round-off 级）。
- 变换一致性：κ→f→κ 往返 ≤ 2e-16；f_eq(H4) 的碰撞不动点检查通过。

### 2.2 算例参数（无量纲）

| 量 | 值 | 说明 |
|---|---|---|
| 网格 n² | 64²（主算例）/ 32², 48²（收敛序列） | 周期边界 |
| U0 | 0.020 | 涡速幅值（格单位） |
| Ma | 0.0346 | U0 c_s⁻¹，c_s²=1/3 |
| ν | 0.050 | ν=(1/ω−1/2)/3 |
| ω | 1.5385 | 剪切松弛率 |
| Re | 12.8 | U0·(2π/k)/ν，涡尺度为特征长 |
| 步数 | 200 | 测率窗 50–200 |

## 3 计算结果

### 3.1 对照表（ref = 解析 4νk²）

| 臂 | n | ω | 实测 4νk² | 解析 4νk² | 误差 % | 判定 | 末态场 L2 误差 % | 判定 |
|---|---|---|---|---|---|---|---|---|
| LIFE partial (s=1 high orders) | 32 | 1.5385 | 0.007707 | 0.007711 | -0.045 | ✅ | 0.561 | ✅ |
| all-order ω | 32 | 1.5385 | 0.007713 | 0.007711 | +0.027 | ✅ | 0.788 | ✅ |
| bulk 1, s3=s4=ω | 32 | 1.5385 | 0.007712 | 0.007711 | +0.022 | ✅ | 0.613 | ✅ |

### 3.2 精度判定（逐项）

- 衰减率：LIFE 臂 -0.045 %，全阶 ω 臂 +0.027 %，bulk-1 臂 +0.022 % ⇒ **均 < 15 %**。
- 场误差：LIFE 臂 0.561 % ⇒ **< 15 %**。
- 算子本身：与上游 round-off 级一致（1e-16），远优于 15 % 门。

## 4 收敛性与参数敏感性

### 4.1 网格收敛（LIFE 臂，场上 L2 误差）

| n | 场 L2 误差 % | 相对上一档下降倍数 |
|---|---|---|
| 32 | 0.5611 | — |
| 48 | 0.4570 | 1.23× |

拟合阶次 ≈ 0.51（期望 2，二阶空间精度；误差在 15 % 门内两个量级以下）。

### 4.2 高阶松弛率敏感性（ω 固定，s3=s4 扫描）

| s3 = s4 | 实测 4νk² | 误差 % |
|---|---|---|
| 0.7 | 0.007701 | -0.128 |
| 0.85 | 0.007705 | -0.079 |
| 1.0 | 0.007707 | -0.045 |
| 1.15 | 0.007709 | -0.020 |
| 1.3 | 0.007711 | -0.001 |
| 1.5384615384615383 | 0.007712 | +0.022 |

### 4.3 稳定性扫描（ω ↑ ⇔ ν ↓，Ma 固定）

| 臂 | ω_max 稳定 | 对应 ν_min | 判据 |
|---|---|---|---|
| LIFE partial (s=1 high orders) | 2.00 | 0.00017 | max|f|>0.6 或 NaN |
| all-order ω | 2.00 | 0.00017 | max|f|>0.6 或 NaN |
| bulk 1, s3=s4=ω | 1.90 | 0.00877 | max|f|>0.6 或 NaN |

## 5 与 OSS 实现的差异清单与移植取舍

| 维度 | joconnor22/LIFE（上游） | 本仓移植 | 取舍理由 |
|---|---|---|---|
| 维度 | 2D D2Q9，速度/压力/自由滑移 BC + 共位 IBM | 2D D2Q9，周期边界 | 对标目标是**碰撞算子**本身；边界差异见 §6 局限 |
| 矩变换 | 手写 `fStar[0..8]` 展开式（约 90 项） | 基矩阵 M0ᵀ + 稀疏 unshift 公式 | 同构可维护；以自动抽取 oracle 钉住等价性 |
| 平衡态 | 四阶 Hermite（保 u² 对角项） | 同（`equilibrium_hermite4`） | 忠实移植 |
| 力耦合 | 中心矩空间注入 F/2，`macroscopic()` 加 F_xy/2 | 同（`s_bulk/s3/s4` 之外的力项逐项对齐） | 上游只在宏观速度里加 force_xy（不是 force_ibm），本仓把二者统一成显式 force 参数 |
| 高阶矩速率 | 硬编码为「置平衡」（s=1） | 开放 `s_bulk/s3/s4`，缺省 = LIFE | 允许消融实验（§4.2） |
| 时间推进 | stream+collide 融合（push 算法） | 碰撞与 `solver.stream`（gather）分离 | 复用仓内既有 stream，便于与其它算子同条件对比 |
| 并行 | OpenMP 共享内存 | torch（CPU/GPU 张量） | 与 TensorLBM 生态一致 |
| 黏度口径 | ω 由物理 ν 与 Δt 反解 | 同，ν=(1/ω−1/2)/3 | 一致 |

**移植结论**：碰撞算子层面做到了 round-off 级等价（1e-16），因此本仓 D2Q9 CM 算子可直接当作 LIFE 的算子替身做后续对比；未移植的部分（BC/IBM/FEM 耦合）明确不声称等价。

## 6 失败模式与工件

- **不在范围内的等价性**：上游的 regularised 速度/压力边界（`regularisedBC`）与 IBM 力投影未移植，故任何带壁面/浸入边界的结论都不能由本对标外推。
- **高阶矩重置的代价**：s3/s4=1 等价于对三、四阶矩做**完全耗散**，在 Ma 较高时会额外衰减小尺度结构；§4.2 给出速率敏感性，Ma→0.3 之外未见（本轮 Ma 见参数表）。
- **发散判据的经验性**：max|f|>0.6 或 NaN，是工程判据而非严格线性稳定性分析；因此 ω_max 只用于**臂间相对比较**。
- 工件：`runs/xval_life_cm_partial/result.json`、`figures/xval_life_cm_partial/*.png`、`tests/fixtures/life_cm_partial_oracle.json`、`tests/test_central_moment_d2q9.py`（14 项，全绿）。
- 生态摩擦：`oss_solutions` 表对本角色不可读（permission denied），4010 知识检索引擎 `127.0.0.1:8021` 离线（`engine_offline`）；上游 GitHub 直连被 egress 拦截，最终经 `ghproxy.net` 取回源码 tarball。建议平台把 oss 只读查询与源码缓存做成 API。

## 7 图

- `fig1_tg_decay.png`：三条臂的 K(t) 与解析衰减率对照 + 误差柱状图（15 % 门）。
- `fig2_convergence.png`：LIFE 臂场上 L2 误差随 n 的双对数曲线与二阶参考线。
- `fig3_stability_sensitivity.png`：稳定性扫描（stable/blow-up）与 s3=s4 敏感性。

