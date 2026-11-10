# 被动标量输运验证面 P74-N10（passive scalar transport surface）

**状态：FAIL** — 8 门中 7/8 通过（机器判定，artifact
`artifacts/passive_scalar_p74n10.json`，每个数字可回溯源文件）。

物理域：标量输运/混合（船舶总体性能·绿色航行与排放混合、压载水/尾气稀释、冰区示踪剂）。
本面回答工程问题：**被动标量核在什么 $(Pe_r, N_y)$ 上能把剪切弥散系数复现到 5% 以内，
失效边界在哪里。**

## 1 案例介绍

三项独立物理锚，全部用**矩/衰减率**度量（峰值法会被网格抹平骗过，禁用）：

| 锚 | 解析参考 | 参数 |
|---|---|---|
| 半无限 erf 扩散 | $C=\tfrac12\mathrm{erfc}((x-x_0)/\sqrt{4Dt})$ | $D=0.1$, $t=200$, 阶梯初值, 扩散标定 $dt\sim dx^2$ |
| Taylor–Aris 平面缝剪切弥散 | $D_{eff}/D=1+Pe_r^2/210$, $Pe_r=UH/D$ | $Pe_r\in\{5,10,20,40\}$, $N_y\in\{9,17,33\}$, 固定格子 Mach $|u|=0.09$ |
| $Pe_r\to0$ / 负控 | $D_{eff}/D\to1$ | $Pe_r\in\{0.5,0.25,0.1\}$；另 $u\equiv0$ |

标量核 = `tensorlbm.scalar` D2Q5（零通量壁半程反弹）；弥散计量 = `tensorlbm.taylor_aris`
单 Fourier 模态衰减率（`method="mode"`）；速度场 = 解析 Poiseuille 抛物线（单向耦合，
动量求解另有 verified 案例）。**零手写物理核**。

### 域长规则（本次新增的测量学约束）

模态衰减等于 $D_{eff}$ 要求纵向衰减远慢于横向弛豫：

$$\frac{D_{eff}}{D}\Big(\frac{2\pi h}{n_x}\Big)^2\Big/\pi^2 \le 0.05,\qquad h=N_y/2$$

$n_x$ 按此式逐格点选定（2 的幂，钳在 [64, 512]）。实测证据：$Pe_r=40, N_y=17$ 在
$n_x=128$ 给 **+8.70%**，同一格点 $n_x=256$ 给 **−0.02%**；$Pe_r=40, N_y=65, n_x=128$
（比值 2.2）给出 **−65.6%** —— 违反该条件时残差属于**计量窗口**而非标量核。

## 2 怎么做的

1. `passive_scalar_verification.py`（P73W2B-N05）修复：其导入的 5 个 D2Q5 核已迁到
   `tensorlbm.scalar`，旧导入使模块与 pytest 全部 ImportError（本次缺陷修复，见 §6）。
2. 新增 `tensorlbm.passive_scalar_transport_verification`：可断点续跑的用例集
   （每例结果 JSON 缓存，参数变则失效重算）、8 门判据、$Pe_r\times N_y$ 精度域表。
3. 运行器 `benchmarks/pending/passive_scalar_verification/run_p74n10.py`
   （`--only` 分块 / `--assemble` 出判据+图+artifact），单线程避免控制面争抢。

## 3 计算结果

### 3.1 门判决

| 门 | 判据 | 结果 | 实测 |
|---|---|---|---|
| A erf 二阶收敛 | 最细网格 rel $L_2\le1\%$ 且 $p\ge1.8$ | ✅ PASS | rel $L_2$=0.0019%, $p$=['2.001', '1.994'] |
| B Taylor–Aris ≤5% | $Pe_r\in\{5,10,20,40\}$ 各有实现 ≤5% | ✅ PASS | 各 $Pe_r$ 最好 4.70% 内（见 3.2） |
| C $Pe_r\to0$ | $\lvert D_{eff}/D-1\rvert\le2\%$ | ✅ PASS | 最大偏差 0.567% |
| D Schmidt 无关 | 散度 ≤3% | ✅ PASS | 0.000% |
| E 质量守恒 | ≤$10^{-10}$/1e4 步 | ✅ PASS | 1.084e-12/1e4 步 |
| F 负控 $u=0$ | $D_{eff}/D\in[0.95,1.05]$ | ✅ PASS | 0.994288 |
| G 收敛质量 | $R^2\ge0.98$ 且误差单调降 | ❌ FAIL | erf $R^2$=1.00000 单调；弥散 $R^2_{\min}$=1.00000，单调=**False** |
| H 精度域表 | $Pe_r\times N_y$ 全格点 | ✅ PASS | 12 格点（见 3.3） |

### 3.2 Taylor–Aris 阶梯（固定 Mach 0.09）

| $Pe_r$ | $N_y$ ladder | err (%) per rung | best | gate B | monotone |
|---|---|---|---|---|---|
| 5 | 9, 17, 33 | -5.47, -4.88, -4.70 | 4.70 % @ $N_y$=33 | yes | yes |
| 10 | 9, 17, 33 | -4.62, -2.77, -2.23 | 2.23 % @ $N_y$=33 | yes | yes |
| 20 | 9, 17, 33 | -5.83, -1.95, +0.46 | 0.46 % @ $N_y$=33 | yes | yes |
| 40 | 9, 17, 33 | -5.29, -0.02, +1.47 | 0.02 % @ $N_y$=17 | yes | no |

### 3.3 精度域表 $Pe_r\times N_y$（时推进 LBM vs 闭式解）

| $Pe_r$ | $N_y$ | $n_x$ | $D$ | $\tau_d$ | steps | $D_{eff}/D$ (LBM) | $1+Pe_r^2/210$ | err | meter ratio | status |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 9 | 64 | 0.1620 | 0.986 | 2000 | 1.05783 | 1.11905 | -5.47 % | 0.022 | exceeds_gate |
| 5 | 17 | 128 | 0.3060 | 1.418 | 2000 | 1.06444 | 1.11905 | -4.88 % | 0.020 | pass |
| 5 | 33 | 256 | 0.5940 | 2.282 | 2000 | 1.06642 | 1.11905 | -4.70 % | 0.019 | pass |
| 10 | 9 | 64 | 0.0810 | 0.743 | 2000 | 1.40794 | 1.47619 | -4.62 % | 0.029 | pass |
| 10 | 17 | 128 | 0.1530 | 0.959 | 2000 | 1.43527 | 1.47619 | -2.77 % | 0.026 | pass |
| 10 | 33 | 256 | 0.2970 | 1.391 | 3667 | 1.44326 | 1.47619 | -2.23 % | 0.025 | pass |
| 20 | 9 | 128 | 0.0405 | 0.621 | 2000 | 2.73546 | 2.90476 | -5.83 % | 0.014 | exceeds_gate |
| 20 | 17 | 256 | 0.0765 | 0.730 | 3778 | 2.84798 | 2.90476 | -1.95 % | 0.013 | pass |
| 20 | 33 | 256 | 0.1485 | 0.946 | 7333 | 2.91824 | 2.90476 | +0.46 % | 0.048 | pass |
| 40 | 9 | 128 | 0.0202 | 0.561 | 4000 | 8.16298 | 8.61905 | -5.29 % | 0.043 | exceeds_gate |
| 40 | 17 | 256 | 0.0382 | 0.615 | 7556 | 8.61763 | 8.61905 | -0.02 % | 0.038 | pass |
| 40 | 33 | 512 | 0.0742 | 0.723 | 14667 | 8.74543 | 8.61905 | +1.47 % | 0.036 | pass |

**可实现精度域**（≤5%）：$N_y\ge N_y^{\min}$，实测
`{"5": 17, "10": 9, "20": 17, "40": 17}`。
粗档 $N_y=9$ 在 $Pe_r\ge20$ 落到 −5.3…−5.8%：横向格点 Péclet $u/D=Pe_r/N_y$ 已 >2，
Aris 横向剖面的解析尺度 $h/Pe_h$ 不再被网格分辨 —— 这就是失效边界。

### 3.4 erf 收敛序列

| $n_x$ | $\Delta x$ | steps | $L_2$ | $L_2$ (rel) |
|---|---|---|---|---|
| 128 | 1 | 200 | 1.967e-04 | 0.0299 % |
| 256 | 0.5 | 800 | 4.912e-05 | 0.0075 % |
| 512 | 0.25 | 3200 | 1.233e-05 | 0.0019 % |

### 3.5 $Pe_r\to0$ 与负控

| $Pe_r$ | $D$ | $u$ | $D_{eff}/D$ | closed form | dev. from 1 |
|---|---|---|---|---|---|
| 0.1 | 0.0500 | 0.00029 | 0.99433 | 1.000048 | -0.567 % |
| 0.25 | 0.0500 | 0.00074 | 0.99458 | 1.000298 | -0.542 % |
| 0.5 | 0.0500 | 0.00147 | 0.99545 | 1.001190 | -0.455 % |

负控（$u\equiv0$，$D=0.05$）：$D_{eff}/D=0.994288$（带 $[0.95,1.05]$）——
$u=0$ 时标量退化为纯扩散，无虚假对流。

### 3.6 Schmidt 无关性（固定 $D$，Sc 走 $\nu$）

| $Sc=\nu/D$ | $\nu$ | $\tau_f$ | $D$ | $\tau_d$ | $D_{eff}/D$ | err vs closed |
|---|---|---|---|---|---|---|
| 0.1 | 0.005 | 0.515 | 0.0500 | 0.650 | 1.110291 | -0.782 % |
| 1 | 0.05 | 0.650 | 0.0500 | 0.650 | 1.110291 | -0.782 % |
| 10 | 0.5 | 2.000 | 0.0500 | 0.650 | 1.110291 | -0.782 % |

散度 **0.000%**（判据 3%）。机制说明：被动标量单向耦合于给定速度场，
标量解只依赖 $(Pe_r,\tau_d)$；扫 $\nu$（即 $\tau_f$）不改变 $D_{eff}/D$。若 $\nu$ 反被钉死、
用 $D=\nu/Sc$ 扫 $Sc$（节点的另一种读法），$D$ 会跨两个数量级：

| $Sc$ | $D=\nu/Sc$ | $\tau_d$ | $D_{eff}/D$ | err vs closed |
|---|---|---|---|---|
| 0.1 | 1.000 | 3.500 | 0.87668 | -12.44 % |
| 1 | 0.100 | 0.800 | 0.98950 | -1.17 % |
| 10 | 0.010 | 0.530 | 0.99988 | -0.13 % |

该读法散度 **12.90%**，原因不是 Sc 依赖，
而是核的扩散数偏置（见 3.7）：$Sc=0.1$ 时 $D=1.0$、$\tau_d=3.5$ 已出精度带。

### 3.7 核的扩散数精度域（u=0，纯扩散）

| $D$ | $\tau_d$ | $D_{eff}/D$ | err |
|---|---|---|---|
| 0.020 | 0.560 | 0.99777 | -0.22 % |
| 0.050 | 0.650 | 0.99429 | -0.57 % |
| 0.100 | 0.800 | 0.98835 | -1.17 % |
| 0.250 | 1.250 | 0.97031 | -2.97 % |
| 0.500 | 2.000 | 0.93963 | -6.04 % |
| 1.000 | 3.500 | 0.87687 | -12.31 % |

拟合（本面程序化）：$D_{meas}/D = 1 -0.0046\,D -0.1187\,D^2$，
6 点最大残差 0.0285%；即 $D_{meas}=D-0.0005(\tau_d-\tfrac12)^2$。
**$D\le0.1$（$\tau_d\le0.8$）内偏置 ≤1.2%**，$D\ge0.5$ 时 >6% —— 这是标量核的
$\tau_d$ 精度域，也是判 C/F 残差（≈−0.57%）的来源。

### 3.8 质量预算（fp64）

| case | drift per $10^4$ steps |
|---|---|
| naivesc_sc10 | 1.084e-12 |
| probe_diff_D0.02 | 1.007e-12 |
| disp_pe40_ny9 | 9.687e-13 |
| pezero_pe0.1 | 8.847e-13 |
| disp_pe20_ny9 | 8.749e-13 |
| schmidt_sc0.1 | 8.667e-13 |
| schmidt_sc1 | 8.667e-13 |
| schmidt_sc10 | 8.667e-13 |
| negative_control_u0 | 8.647e-13 |
| probe_diff_D0.05 | 8.647e-13 |
| disp_pe40_ny17 | 8.449e-13 |
| pezero_pe0.5 | 8.340e-13 |
| pezero_pe0.25 | 8.154e-13 |
| naivesc_sc1 | 7.868e-13 |
| disp_pe40_ny33 | 7.476e-13 |
| disp_pe10_ny9 | 7.355e-13 |
| disp_pe20_ny17 | 7.123e-13 |
| probe_diff_D0.1 | 7.007e-13 |
| erf_L2 | 6.554e-13 |
| probe_pe5_ny17_u0.03 | 6.188e-13 |
| erf_L1 | 6.176e-13 |
| probe_pe5_ny17_u0.045 | 5.995e-13 |
| erf_L0 | 5.940e-13 |
| disp_pe20_ny33 | 5.714e-13 |
| probe_pe5_ny17_u0.06 | 5.640e-13 |
| disp_pe5_ny9 | 5.534e-13 |
| disp_pe10_ny17 | 5.418e-13 |
| probe_diff_D0.25 | 4.463e-13 |
| disp_pe10_ny33 | 3.887e-13 |
| disp_pe5_ny17 | 3.664e-13 |
| probe_pe5_ny17_u0.09 | 3.664e-13 |
| probe_diff_D0.5 | 2.776e-13 |
| disp_pe5_ny33 | 2.376e-13 |
| naivesc_sc0.1 | 1.604e-13 |
| probe_diff_D1 | 1.538e-13 |

上限 1.084e-12 / 1e4 步（判据 $10^{-10}$）。fp64 下 D2Q5 权重和
$1/3+4\times1/6$ 恰好舍入为 1；fp32 下为 $1+2.98\times10^{-8}$，每步重标漂移
$\sim10^{-4}$/1e4 步（pytest `test_float32_breaks_the_mass_budget` 定量锁定）——
**判据 E 实际钉住的是工作精度**。

## 4 图

* `artifacts/passive_scalar_p74n10/figures/fig1_erf_diffusion.png` — erf 前沿 + 收敛序列
* `artifacts/passive_scalar_p74n10/figures/fig2_taylor_aris_accuracy_domain.png` — $Pe_r$ 阶梯 + 精度域热图
* `artifacts/passive_scalar_p74n10/figures/fig3_schmidt_mass.png` — Schmidt 无关性 + 质量预算

## 5 注意点与局限

1. **域长是被测项的一部分**：模态法在 $n_x$ 过短时把横向瞬态读成 $D_{eff}$（+8.7%），在
   $n_x$ 过长时也出现 −1.8% 残差（$Pe_r=40,N_y=17,n_x=512$）；本面用 3.3 的比值列把每格点的
   计量条件公开标注，违反者不进判据记账。
2. **门 G 在 $Pe_r=40$ 未过**：误差符号在 $N_y=9\to17\to33$ 上为 −5.29% → −0.02% → **+1.47%**，
   $|err|$ 非单调。与仓内既有 `pending/taylor_aris_dispersion` 的定源一致：负号 $O(1/N_y^2)$ 项
   与正号缓变地板在 $N_y\approx20$ 附近变号，fp64/窗口/renorm 关闭均不改变（探针见
   `artifacts/passive_scalar_p74n10/probes/probe_renorm.log`）。honest FAIL，未做口径稀释。
3. 速度场为解析抛物线：本面验证**标量核**，不验证动量求解（另有 verified/poiseuille_3d_pipe 等）。
4. 圆管 Taylor–Aris（$1+Pe_r^2/48$）由离散 Aris 辅助问题覆盖（P73W2B-N05 面），本面不再重复；
   轴对称时推进需要额外源项，库内标量核是笛卡尔。
5. $Pe_r=40$ 的格点需 $n_x=512$，单格点 181 s（单线程）；全表 35 例 ≈ 8 min，已分块/缓存。

## 6 缺陷修复（本单附带）

`tensorlbm/passive_scalar_verification.py` 从 `tensorlbm.scalar_transport` 导入
`scalar_collision / scalar_equilibrium / scalar_macroscopic / scalar_stream / tau_from_diffusivity`，
这 5 个 D2Q5 核已迁到 `tensorlbm.scalar`（`scalar_collide_bgk`）——原状下模块与
`tests/test_passive_scalar_verification.py` 全部 ImportError（P73W2B-N05 面整面失效）。
改指向 `tensorlbm.scalar` 后：该面 17 个测试全绿（46 s），本面 14 个测试全绿。

## 7 证据文件

| 内容 | 路径 |
|---|---|
| 判据/精度域表（机器可读） | `artifacts/passive_scalar_p74n10.json` |
| 每例原始测量（35 例，参数+结果） | `artifacts/passive_scalar_p74n10/cases/*.json` |
| 图 | `artifacts/passive_scalar_p74n10/figures/*.png` |
| 探针（域长/renorm/时间步/扩散数） | `artifacts/passive_scalar_p74n10/probes/` |

## 8 复现

```bash
python benchmarks/pending/passive_scalar_verification/run_p74n10.py --list
python benchmarks/pending/passive_scalar_verification/run_p74n10.py --all --assemble   # ~8 min 单线程
python benchmarks/pending/passive_scalar_verification/run_p74n10.py --assemble         # 仅判据+图
pytest tests/test_passive_scalar_transport_verification.py -q   # 14 passed (~75 s)
pytest tests/test_passive_scalar_verification.py -q             # 17 passed (~46 s, 修复后)
```

参考：Taylor (1953) *Proc. R. Soc. A* 219:186；Aris (1956) *Proc. R. Soc. A* 235:67；
Wooding (1960) *JFM* 9:183。
