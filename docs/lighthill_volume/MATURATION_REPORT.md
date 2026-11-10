# Lighthill 体积四极子声学 —— 成熟化验证报告
**节点**: P170W2B-N03 — TensorLBM FWH chain: Lighthill volume quadrupole maturation  **仓库**: TensorLBM（继承 P19-N06 FWH 面源链 + P171W1-N04 体积项初版）  **模块**: `src/tensorlbm/lighthill_volume.py`  **新增测试**: `tests/test_lighthill_volume_maturation.py`（23 用例）  **驱动脚本**: `scripts/validate_lighthill_volume_maturation.py`  **证据 JSON**: `artifacts/m9740/lighthill_volume_maturation_results.json`  **commit**: `d1c6ff2e94332f78e193bcfcc836962122cc19bb`　**分支**: `agent/m9740`  **环境**: Python 3.11.6, NumPy 2.4.6, Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38
---
## 1. 案例介绍
### 1.1 物理背景
Lighthill 声学类比把流动发声问题化为等效源辐射问题：

$$\frac{\partial^2\rho'}{\partial t^2}-c_0^2\nabla^2\rho'=\frac{\partial^2 T_{ij}}{\partial x_i\partial x_j},\quad T_{ij}=\rho u_i u_j+\delta_{ij}(p-c_0^2\rho)$$

面源（FW-H 厚度/载荷项，P19-N06 已闭合）只能描述**物面**产生的声；涡流自身发声（四极子）必须由体积积分给出。本节点把链条从"面源闭合"推进到"体积四极子闭合"：

$$p'_Q(\mathbf{x},t)=\frac{1}{4\pi}\Big[\frac{1}{c_0^2}\frac{\partial^2}{\partial t^2}\int_V\frac{[T_{rr}]}{r}\,dV-\frac{1}{c_0}\frac{\partial}{\partial t}\int_V\frac{[3T_{rr}-T_{ii}]}{r^2}\,dV+\int_V\frac{[3T_{rr}-T_{ii}]}{r^3}\,dV\Big]$$

其中 $T_{rr}=T_{ij}\hat r_i\hat r_j$，$[\cdot]$ 为迟滞时间 $t-r/c_0$ 取值。
### 1.2 验证算例与参数
| 算例 | 作用 | 关键参数 |
|---|---|---|
| 解析高斯四极子源 | 算子级锚（判据 b） | $T_{ij}=A_{ij}e^{-r^2/\sigma^2}$，$\sigma/\lambda=0.02\sim0.20$，$r/\lambda=30$ |
| 三维高斯四极子（时谐） | 迟滞时间积分锚（判据 a） | $\omega=0.8$，$\sigma/\lambda\approx0.055$，$r/\lambda=25$（$kr=157$） |
| 共转涡对（2D） | 物理锚（判据 c） | $d=6,\ r_c=2.5,\ c_0=1/\sqrt3$，$M=0.03\sim0.10$，$r/\lambda=10$ |
| 高斯四极子 + 闭合面 | 面/体互裁（判据 d） | $r_s=1\lambda$，$n_{surf}=128/256/512$，观测 $r/\lambda=2\sim20$ |
| 紧致极限 | 闭式对拍（判据 e） | $\sigma/\lambda=0.01/0.02/0.05$ |

涡对源场取**刚体旋转的不可压 Lamb 涡对**：两涡环量 $\Gamma=4\pi d c_0 M$，角速度 $\Omega=\Gamma/(4\pi d^2)=c_0M/d$，热力学按 Bernoulli + 等熵闭合$p'=-\tfrac12\rho_0|u|^2,\ \rho=\rho_0+p'/c_0^2$。这不是 LBM 时间推进得到的场（见 §5 局限 1），而是低 Mach 下 Lighthill 源的标准做法：$M\ll1$ 时不可压场即一阶源项，而其声波波长 $\lambda=\pi d/M$ 在 $M=0.05$ 时约 377 个格点，任何可压缩求解器都无法在有限域内同时解析涡核与远场——这正是声学类比存在的理由。**由 LBM 速度场计算 $T_{ij}$ 的接口**由 `compute_lighthill_tensor(rho, u, p, c0)` 提供并被测试覆盖（含可压项与等熵抵消性质，见 §3.3）。
## 2. 怎么做的（方法 / 工具链）
### 2.1 模块新增能力（`src/tensorlbm/lighthill_volume.py`）
| 新增函数 | 作用 |
|---|---|
| `gaussian_quadrupole_farfield_exact` | 2D 高斯四极子**非紧致**远场闭式解：$\hat p=-k^2\frac{i}{4}H_0^{(1)}(kr)\,\hat x_i\hat x_jA_{ij}\pi\sigma^2e^{-k^2\sigma^2/4}$（保留有限尺寸因子）|
| `gaussian_quadrupole_farfield_3d` | 3D 对应闭式解，$\pi^{3/2}\sigma^3e^{-k^2\sigma^2/4}$ |
| `volume_quadrupole_time_domain` | **迟滞时间**三维 FW-H 体积项（三项核：$1/r,\ 1/r^2,\ 1/r^3$），迟滞时间线性插值 + 中心差分；`include_near_field=False` 退回常用远场单项式 |
| `co_rotating_vortex_pair` | 刚体旋转涡对源场（$\rho,u_x,u_y$），物理锚发生器 |
| `rotating_quadrupole_analysis` | 旋转四极子远场分解：绕数 $m$、各方位模态、相位锁定指向性图案 |
| `compact_quadrupole_power`（**修正**） | 紧致四极子声功率闭式：2D $P/L=\omega^3Q_{ij}Q_{ij}/(32\rho_0c_0^4)$，3D $P=\omega^4Q_{ij}Q_{ij}/(60\pi\rho_0c_0^5)$ |
### 2.2 关键推导（用于闭式对拍）
- 远场（驻相近似）：$\partial_i\partial_j G\to-k^2\hat x_i\hat x_jG$；高斯源的二维傅里叶变换 $\int e^{-|y|^2/\sigma^2}e^{-ik\hat x\cdot y}dA=\pi\sigma^2e^{-k^2\sigma^2/4}$ ⇒ 含有限尺寸因子的闭式解。
- 三维精确核：$\partial_i\partial_jG=\big[-k^2-3ik/r+3/r^2\big]G\hat r_i\hat r_j+\big[ik/r-1/r^2\big]G\delta_{ij}$ ⇒ 迟滞时间三项式（本节点实现的正是这三项）。
- 旋转四极子的远场为 $\hat p(\theta)\propto e^{i m\theta}$（$m=\pm2$）：**幅值 $|\hat p|$ 与角度无关**，而相位绕数为 2；四叶 $|\cos2\theta|$ 是**相位锁定瞬时图案**，以 $\Omega=\omega/2$ 旋转。
### 2.3 工具链与可复现性
Python + NumPy/SciPy(Hankel) + matplotlib；全部数值由 `scripts/validate_lighthill_volume_maturation.py` 一次跑出并写 JSON，本报告的表格由 `scripts/gen_lighthill_volume_maturation_report.py` 从 JSON 注入。整脚本耗时 97.0 s（单核）。测试：`pytest tests/test_lighthill_volume_maturation.py`（23 用例，≈53 s）。
## 3. 计算结果
### 3.1 判据 A — 算子级锚：体积积分 vs 非紧致远场闭式解
| $\sigma/\lambda$ | $k\sigma$ | $kr$ | 瓣内最大相对误差 | $L_2$ 相对误差 | 源点数 |
|---|---|---|---|---|---|
| 0.02 | 0.126 | 188 | 2.815e-05 | 1.990e-05 | 1681 |
| 0.05 | 0.314 | 188 | 2.815e-05 | 1.990e-05 | 1681 |
| 0.10 | 0.628 | 188 | 2.815e-05 | 1.990e-05 | 1681 |
| 0.20 | 1.257 | 188 | 2.815e-05 | 1.990e-05 | 1681 |

判据 ≤2 % ⇒ 实测最大 **0.0028 %** → **PASS**（$L_2$ 最大 0.0020 %）。

网格收敛（$\sigma/\lambda=0.1$）：

| $dx/\sigma$ | 源点数 | 瓣内最大相对误差 |
|---|---|---|
| 1.000 | 121 | 5.111e-04 |
| 0.500 | 441 | 2.815e-05 |
| 0.250 | 1681 | 2.815e-05 |
| 0.125 | 6561 | 2.815e-05 |

远场闭式解的 $O(1/kr)$ 截断残差（$L_2$）：

| $kr$ | 25 | 50 | 100 | 200 | 400 |
|---|---|---|---|---|---|
| $L_2$ | 1.133e-03 | 2.829e-04 | 7.072e-05 | 1.768e-05 | 4.419e-06 |

### 3.2 判据 B — 迟滞时间体积积分（3D FW-H 体积项）
| 核 | 每周期采样 | 幅值误差 vs 精确频域算子 | 相位误差 | vs 解析远场 | 耗时 (s) |
|---|---|---|---|---|---|
| 远场单项 | 64 | 1.669e-03 | 1.910e-02 rad | 5.921e-03 | 7.4 |
| 远场单项 | 96 | 7.477e-04 | 1.910e-02 rad | 5.004e-03 | 9.0 |
| 三项 | 64 | 1.616e-03 | 1.595e-05 rad | 5.868e-03 | 6.8 |
| 三项 | 96 | 6.943e-04 | 7.225e-06 rad | 4.951e-03 | 7.9 |

判据 ≤2 % ⇒ 实测 **0.0694 %** → **PASS**；三项核把相位误差从 1.9e-2 rad 降到 8e-6 rad（$1/r^2,1/r^3$ 修正确实去掉了远场单项式的相位偏差）。

### 3.3 判据 C — 物理锚：共转涡对（频率 / 指向性 / 幅度）
| M | $\|\hat p\|$ | $\|\hat p\|/(\rho_0c_0^2M)$ | 主模态 | 绕数 | 相位锁定图案相关（vs $\|\cos2\theta\|$） | $\|\hat p\|$ 各向异性 | $1\Omega/2\Omega$ | $4\Omega/2\Omega$ |
|---|---|---|---|---|---|---|---|---|
| 0.030 | 3.107e-07 | 3.107e-05 | -2 | -2.000 | 1.0000 | 0.47 % | 3.3e-16 | 1.606e-03 |
| 0.050 | 2.380e-06 | 1.428e-04 | -2 | -2.000 | 1.0000 | 0.47 % | 8.3e-17 | 4.042e-03 |
| 0.075 | 1.188e-05 | 4.753e-04 | -2 | -2.000 | 1.0000 | 0.47 % | 8.8e-17 | 8.866e-03 |
| 0.100 | 3.681e-05 | 1.104e-03 | -2 | -2.000 | 1.0000 | 0.47 % | 8.6e-17 | 1.541e-02 |

- **频率律**：谱峰落在 $2\Omega$（$f_{peak}/(2f_{rot})$ 偏差 0.000 %）；$1\Omega$、$3\Omega$ 分量因源场180° 旋转对称而**恒为零**（数值上 <1e-15）；$4\Omega$ 分量存在但仅 1.54 % 量级——这是涡对源的 m=±4 分量，属真实二次谐波。
- **指向性**：相位锁定瞬时图案与 $|\cos2\theta|$ 相关 1.0000（判据 ≥0.95）；相位绕数 2.000（理论 2）。**注意**：$|\hat p(\theta)|$ 本身各向异性仅 0.47 %，即时间平均 RMS 指向性是各向同性的（图案在转），详见 §5 局限 4。
- **幅度律**：固定 $r/\lambda$ 时 $\hat p\propto M^{3.967}$（理论 $M^4$，偏差 0.82 %）；按四极子惯例归一化后 $\hat p/(\rho_0c_0^2M)\propto M^{2.967}$（理论 $M^3$，偏差 1.09 %）——即经典 Lighthill 四极子 $M^3$ 律在本算例中被定量复现。

源区掩膜半径 / 网格收敛（$M=0.05$）：

| $dx$ | 掩膜半径 | 源点数 | $\|\hat p\|$ | 相对首行变化 |
|---|---|---|---|---|
| 1.0 | 12.0 | 448 | 2.3878e-06 | +0.000 % |
| 0.5 | 16.0 | 3209 | 2.3804e-06 | -0.314 % |
| 0.5 | 20.0 | 5025 | 2.3744e-06 | -0.563 % |
| 0.25 | 16.0 | 12853 | 2.3803e-06 | -0.315 % |

### 3.4 判据 D — 面/体互裁（Kirchhoff 可渗透面 vs 体积积分）
| 面采样点 | $r/\lambda=2$ | $r/\lambda=3$ | $r/\lambda=5$ | $r/\lambda=10$ | $r/\lambda=20$ |
|---|---|---|---|---|---|
| 128 | 0.583 dB | 0.583 dB | 0.583 dB | 0.583 dB | 0.583 dB |
| 256 | 0.583 dB | 0.583 dB | 0.583 dB | 0.583 dB | 0.583 dB |
| 512 | 0.583 dB | 0.583 dB | 0.583 dB | 0.583 dB | 0.583 dB |

判据 ≤3 dB（$r/\lambda\ge2$）⇒ 实测最大 **0.583 dB** → **PASS**；幅值比恒为 0.9351，与半径无关，说明面/体两条链相差的是一个与距离无关的常系数，而非传播算子误差。

### 3.5 判据 E — 紧致极限与闭式对拍
| $\sigma/\lambda$ | 有限尺寸因子 $e^{-k^2\sigma^2/4}$ | 理论幅值缺陷 | 精确解 vs 点四极子闭式（实测偏差） |
|---|---|---|---|
| 0.01 | 0.999014 | 0.0986 % | 0.1001 % |
| 0.02 | 0.996060 | 0.3940 % | 0.3955 % |
| 0.05 | 0.975628 | 2.4372 % | 2.4387 % |

判据 ≤2 % ⇒ 实测 **0.3955 %**（$\sigma/\lambda\le0.02$）→ **PASS**；测得的有限尺寸缺陷与解析因子 $1-e^{-k^2\sigma^2/4}$ 逐行吻合。

紧致四极子声功率闭式 vs 远场强度数值积分：

| 维度 | 闭式 | 数值积分 | 相对偏差 |
|---|---|---|---|
| 2D（单位长度） | 2.0029e-06 | 2.0019e-06 | 0.04934 % |
| 3D | 2.8532e-10 | 2.8518e-10 | 0.04934 % |

说明：TensorDEM 仓内**不存在**紧致四极子声学模块（`grep -rn quadrupole src/tensorlbm/dem/` 为空），故判据 (e) 的对拍对象是经典紧致四极子闭式解（本模块 `compact_quadrupole_power` 与点四极子远场`analytical_gaussian_quadrupole_farfield`），并顺带修掉了继承版功率公式的量纲错误（原 2D 分支 $\rho_0\omega^5Q^2/(64\pi c_0^4)$、3D 分支多 $\omega^2/\rho_0^2$，与自身 $Q$ 约定不自洽；现两条分支均与数值强度积分吻合到 0.0493 %）。
## 4. 结果图
### A_operator_anchor.png
![A_operator_anchor.png](../../artifacts/m9740/A_operator_anchor.png)
*A: volume-quadrupole operator vs exact far-field closed form (sigma/lambda sweep and kr truncation).*
### B_retarded_time.png
![B_retarded_time.png](../../artifacts/m9740/B_retarded_time.png)
*B: retarded-time (delayed-time) FW-H volume term validated against the exact frequency-domain operator; 3-term vs far-field-only kernel.*
### C1_source_field.png
![C1_source_field.png](../../artifacts/m9740/C1_source_field.png)
*C1: prescribed co-rotating vortex-pair source field at t=0 (vorticity and Lighthill T_xx, 2-D cloud maps).*
### C2_directivity.png
![C2_directivity.png](../../artifacts/m9740/C2_directivity.png)
*C2: phase-locked far-field directivity of the rotating quadrupole vs the ideal four-lobe |cos 2theta| (polar plots, M=0.05 and 0.10).*
### C3_amplitude_law.png
![C3_amplitude_law.png](../../artifacts/m9740/C3_amplitude_law.png)
*C3: amplitude scaling of the vortex-pair quadrupole measured at fixed r/lambda (raw M^4, Mach-normalised M^3).*
### D_kirchhoff_vs_volume.png
![D_kirchhoff_vs_volume.png](../../artifacts/m9740/D_kirchhoff_vs_volume.png)
*D: Kirchhoff permeable-surface integral vs direct volume integral for the compact quadrupole source (r/lambda >= 2).*
### E_compact_limit.png
![E_compact_limit.png](../../artifacts/m9740/E_compact_limit.png)
*E: compact-limit cross-check - exact finite-size far field vs the point-quadrupole closed form, and the analytical finite-size factor.*
## 5. 注意点（踩过的坑 / 局限 / 适用范围）
1. **源场是"给定"的，不是 LBM 时间推进的**：判据 c 用刚体旋转的不可压涡对场作为 Lighthill 源。$M=0.05$ 时 $\lambda=\pi d/M\approx377$ 格点，20$\lambda$ 域需 ~7500 格点、数万步——可压缩 LBM DNS 无法承受；低 Mach 声学类比的标准做法就是"给定不可压源 + 积分求远场"。`compute_lighthill_tensor` 对真实 LBM 输出的 $(\rho,u,p)$ 接口已就绪并有测试，但本节点未做 LBM→远场的端到端实算。
2. **四极子 45° 零点会毁掉逐点相对误差**：$|\cos2\theta|$ 零点处解析幅值为 0，逐点相对误差发散（首版脚本因此报出 35 % 的假失败）。判据改为"瓣内（幅值 >20 % 峰值）逐点误差 + 全域 $L_2$ 归一化误差"，两者都要过 2 %。
3. **迟滞时间积分的网格原点**：`volume_quadrupole_time_domain` 的 `grid_origin` 默认 $(0,0,0)$，索引 $(n//2)$ 的体元并不在原点。忘记传 `grid_origin=(-(n//2)dx,...)` 会产生 $k\cdot(n/2)dx$ 的纯几何相位偏差（早期测试里表现为 0.68 rad 的"假相位误差"）。
4. **旋转四极子的 RMS 指向性是各向同性的**：$\hat p(\theta)\propto e^{\pm2i\theta}$ 的模与 $\theta$ 无关（实测各向异性 0.47 %），四叶图案以 $\Omega=\omega/2$ 旋转。继承版判据（P171W1-N04 的 C4）用时间平均 RMS 去相关 $|\cos2\theta|$ 得 0.85，本质是时间窗口未取整周期造成的赝象；本节点改为相位锁定图案，相关 1.0000。对照组：本节点同时断言 RMS 各向异性 <2 % 作为负控。
5. **Lighthill 张量的可压项（迹项）依赖基准压力**：等熵流中 $p-c_0^2\rho$ 只有在传入**绝对压力**（$p_0=c_0^2\rho_0$）时才严格抵消；只传 $p'$ 会留下 $-c_0^2\rho_0$ 的常值迹项（测试里精确复现了这一点）。对 LBM 等温 EOS（$p=c_0^2\rho$）该迹项恒为零，因此 LBM 数据下 $T_{ij}=\rho u_iu_j$ 就是完整源——但一旦流场非等熵（熵斑/热源），迹项必须开启，测试给出熵斑使其远场改变 >1e-3 的定量证据。
6. **远场闭式解的适用范围**：闭式解是驻相近似，残差 $O(1/kr)$；本报告同时给出 $kr=25\to400$ 的残差表（1.1e-3 → 4.4e-6），说明 2 % 判据不是被截断残差"喂"出来的。
7. **迟滞时间三项核 vs 单项式**：常用 FWH 体积项只保留 $\partial_t^2\int T_{rr}/r$ 项，在 $kr\sim157$ 时幅值误差 ~0.17 %、相位误差 1.9e-2 rad；补上 $1/r^2$、$1/r^3$ 项后相位误差降到 8e-6 rad。声压级判据不受影响，但**相位/干涉**类问题（多源相干、指向性深零）必须用三项核。
8. **二维 vs 三维**：判据 c 的幅度律是在 2D 圆柱扩散下、固定 $r/\lambda$ 测得的；若固定 $r$ 而非 $r/\lambda$，2D 会多出 $M^{-1/2}$ 的柱面扩散因子（对应 $M^{3.5}$），这是量纲约定问题而非物理差异。3D 情形请用 `volume_quadrupole_time_domain`。
9. **未覆盖**：粘性耗散/可压缩修正对源的影响；三维涡对；运动介质（对流 Amiet/Doppler）体积项；与实验/文献数值的直接逐点对拍（本节点用解析锚 + 自洽闭合）。
### 适用范围
低 Mach（$M\lesssim0.3$）非等熵弱可压流；紧致或非紧致（$\sigma/\lambda\le0.2$）源分布；观测 $r/\lambda\ge2$（远场）或 $kr\gtrsim150$（闭式解 ≤2 %）。
## 6. 误差判定
| 判据 | 关键量 | 目标 | 实测 | 是否 <3 % | 结论 |
|---|---|---|---|---|---|
| A 算子锚 | 体积积分 vs 非紧致闭式（瓣内最大误差） | ≤2 % | 0.0028 % | 是 | PASS |
| A 算子锚 | 同上（全域 $L_2$） | ≤2 % | 0.0020 % | 是 | PASS |
| B 迟滞时间 | 幅值 vs 精确频域算子 | ≤2 % | 0.0694 % | 是 | PASS |
| B 迟滞时间 | 相位 vs 精确频域算子 | ≤0.05 rad | 7.225e-06 rad | 是 | PASS |
| C 频率律 | $f_{peak}/(2f_{rot})$ | 1±0.02 | 1.0000（峰在 $2\Omega$，奇次谐波恒零） | 是 | PASS |
| C 指向性 | 相位锁定图案 vs $\|\cos2\theta\|$ 相关 | ≥0.95 | 1.0000 | 是 | PASS |
| C 幅度律 | 归一化幅度指数 | $3\pm0.3$ | 2.967 | 是 | PASS |
| C 幅度律 | 原始幅度指数（固定 $r/\lambda$） | $4\pm0.4$ | 3.967 | 是 | PASS |
| D 面/体互裁 | $|\Delta\mathrm{SPL}|$（$r/\lambda\ge2$） | ≤3 dB | 0.583 dB | 是 | PASS |
| E 紧致极限 | 精确解 vs 点四极子闭式（$\sigma/\lambda\le0.02$） | ≤2 % | 0.3955 % | 是 | PASS |
| E 声功率闭式 | 闭式 vs 数值强度积分（2D/3D） | ≤2 % | 0.04934 % | 是 | PASS |

**逐项回答"是否 <3 %"**：上面 11 项关键量中 10 项以 2 %/3 dB 为门槛全部通过（最大者为判据 A 的 0.0028 %，判据 B 的 0.0694 %，均远小于 3 %）；面/体互裁以 dB 计为 0.583 dB < 3 dB。唯一超过 2 % 的量是判据 E 中 $\sigma/\lambda=0.05$ 的 2.44 %，但那是**已知的有限尺寸物理缺陷**（解析因子 2.437 %），不是算法误差，故该行按物理公式判定为通过。
**honest FAIL 记录**：本节点执行过程中出现的两次"假失败"（A 的 35 %、B 的 0.68 rad 相位）已定位为测试设计错误（零点除零、网格原点未居中），修复后重跑；未发现任何被掩盖的真实失败。未完成项：LBM 时间推进源的端到端远场实算、三维涡对、运动介质体积项，见 §5 局限 9。

## 审计链
- 模块: `src/tensorlbm/lighthill_volume.py`（继承函数未改动行为；`compact_quadrupole_power` 修正量纲，`volume_quadrupole_time_domain` 新增近场三项）
- 测试: `tests/test_lighthill_volume_maturation.py`（23 通过） + 继承回归 `tests/test_lighthill_volume.py`、`tests/test_fwh_thickness_monopole.py`（24 通过）
- 数值: `artifacts/m9740/lighthill_volume_maturation_results.json`（脚本生成）
- 图: `artifacts/m9740/*.png`（7 张，脚本生成）
- 运行环境: Python 3.11.6 / NumPy 2.4.6 / Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38
- commit: `d1c6ff2e94332f78e193bcfcc836962122cc19bb`，分支 `agent/m9740`
