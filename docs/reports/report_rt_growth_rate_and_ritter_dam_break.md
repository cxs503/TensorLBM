# TensorLBM 自由面多相流重力驱动基准：Rayleigh–Taylor 增长率 + 溃坝 Ritter 自相似律

**战役节点** P55-N01 ｜ **仓库** TensorLBM ｜ **分支** `agent/m5423` ｜ 证据分级：REAL（新增两套求解器 + 两套解析锚基准）

## ① 案例介绍

本单把 TensorLBM 的多相/自由面能力从“静态 Laplace 压差”推进到**重力驱动、带解析锚**的
动力量级检验，选两个经典自由面问题：

| 案例 | 解析锚 | 参数 |
|---|---|---|
| Rayleigh–Taylor 单模不稳定 | $\sigma=\sqrt{Agk}$（Rayleigh 1883 / Taylor 1950） | $\rho_1/\rho_2=2\Rightarrow A=1/3$，$\eta_0=\lambda/100$，$\lambda\in\{32,64,128\}$ 格子，$g=10^{-2}$，$\tau=0.51$ |
| Ritter (1892) 溃坝 | $x_f=2\sqrt{gh_0}\,t$，$h=\frac{(2c_0-\xi)^2}{9g}$ | 干床、无摩阻；无量纲窗口 $t\in[0,5]\sqrt{h_0/g}$；$nx\in\{400,800,1600\}$ |

物理定义域与几何自检：RT 域 $nx=\lambda$（周期），$ny=3\lambda$，界面位于 $ny/2$，
两侧壁距界面 $\ge1.5\lambda$（远大于 $1/k$），界面初始 $\tanh$ 型、初始扰动幅度占
$\lambda$ 的 1 %（与任务书一致）。溃坝域 $x\in[-4,12]h_0$，坝在 $x=0$，左端镜面壁
（$t^*=5$ 时反射波尚未到达前沿），右端零梯度出流。

## ② 怎么做的

**RT（`src/tensorlbm/rayleigh_taylor.py`）**：D2Q9 双分布——流动分布只施加 Boussinesq
浮力偏差 $F_y=-Ag\phi$（平均静水压扣除 ⇒ 参考密度恒 1，无 $\rho_0gH$ 压缩伪影）；相场分布
按 $g_i^{eq}=w_i\phi(1+3c_i\!\cdot\!u)$ 输运（$D=c_s^2(\tau_\phi-\tfrac12)=10^{-3}$）。
半程反弹固壁、$x$ 周期。初值为**线性本征模态**：$u_y=\sigma\eta\cos kx\,e^{-k|y-y_{\rm int}|}$，
$u_x=\pm\sigma\eta\sin kx\,e^{-k|y-y_{\rm int}|}$，只激发增长模。

**增长率测量**：界面由 $\phi=0$ 等值线逐列线性插值，取第一 Fourier 模 $a(t)$；
窗口 $a/\lambda\in[0.01,0.05]$，双模投影 $a=c_+e^{\sigma t}+c_-e^{-\sigma t}$（$\sigma$ 一维扫描 +
$c_\pm$ 线性最小二乘）拟合，可吸收启动衰减模。独立估计量：重流体柱高守恒矩。

**溃坝（`src/tensorlbm/shallow_water.py`）**：1-D 有限体积 Rusanov 通量 + 干床处理 +
镜面虚网格壁面（lake-at-rest 精确保持、严格通量形式）。前沿用两个估计量：
(a) 几何尖端跟踪（复用 `benchmarks/bench_dam_break.py` 的工装 —— 本单把
`tensorlbm.dam_break._find_front_x` 提升为公共 `find_front_x`，两者共享同一插值逻辑）；
(b) **Ritter 律积分形式反解** $x_f=27V_{\rm down}/(4h_0)$，$V_{\rm down}=\int_0^{2c_0t}h\,dx$
是数值守恒量，不受干床尖端涂抹影响。

工具链：torch(CPU, float64/float32)、pytest、matplotlib；全部可复现脚本见两处 README。

## ③ 计算结果

### RT 增长率（`benchmarks/rayleigh_taylor/result.json`）

| $\lambda$ | $\sigma_{\rm sim}$ | $\sqrt{Agk}$ | 比值 | 相对误差 | $R^2$ | 相场质量漂移 |
|---|---|---|---|---|---|---|
| 32 | 0.023435 | 0.025583 | 0.9160 | **−8.40 %** | 0.99917 | 1.4e−4 % |
| 64 | 0.016254 | 0.018090 | 0.8985 | **−10.15 %** | 0.99941 | 7.9e−4 % |
| 128 | 0.010372 | 0.012792 | 0.8109 | **−18.91 %** | 0.99915 | 1.5e−3 % |

### 溃坝（`benchmarks/dam_break_ritter/result.json`，判据窗口 $t^*\in[0.5,5]$）

| nx | dx | 前沿误差（质量积分，max） | 前沿误差（几何尖端，max） | 质量漂移 | 剖面 $L_2$ |
|---|---|---|---|---|---|
| 400 | 0.040 | 6.41 % | 25.94 % | 2.2e−14 % | 8.02 % |
| 800 | 0.020 | 6.41 % | 25.94 % | 2.0e−13 % | 6.70 % |
| 1600 | 0.010 | **3.68 %** | 25.47 % | **1.1e−13 %** | 5.91 % |

## ④ 结果图

* `figures/rt_amplitude_growth.png`：三档 $\lambda$ 的 $a(t)$ 半对数曲线与 $\sigma_{\rm sim}$ 拟合线；
  可在同一张图上看清线性区间的指数律（$R^2>0.999$）与各档 $\sigma$ 的差异。
* `figures/rt_convergence.png`：左——$\sigma_{\rm sim}/\sqrt{Agk}$ 与 $\sigma_{\rm sim}/\sigma_{\rm visc}$
  随 $\lambda$ 的变化及 ±3 % 门带；右——归一化误差随 $\lambda$ 的细化曲线（观测阶 无定义：误差非单调下降）。
* `figures/ritter_front_law.png`：左——两种估计量的 $x_f(t)$ 与 $2\sqrt{gh_0}t$；右——各分辨率前沿误差
  随 $t^*$ 的演化与 ±5 % 门带。
* `figures/ritter_profiles.png`：左——三时刻水深剖面与解析解（虚线）；右——自相似塌缩
  $h\,t^{*2}/h_0$ vs $x/(t\sqrt{gh_0})$，展示自相似律成立。

## ⑤ 注意点（踩坑与局限）

1. **RT 的系统性偏低不是数值不稳定**：$\tau$ 从 0.51 调到 0.53（$\Delta\sigma_{\rm visc}=-1.0\%$）
   实测只降 1.2 %，说明有效粘性与理论一致；界面宽度 $\delta=1.0/1.5/2.0$ 与 $\tau_\phi=0.5005/0.503/0.53$
   的敏感性扫描也都在 ±2 % 内（`runs/tmp_probe/rt_sens.log`）。主因是**启动瞬态**：
   $\eta_0=\lambda/100$ 与线性窗口上限 $0.05\lambda$ 之间只有 $\ln 5\approx1.6$ 个 e-fold，
   衰减模衰减不完，双模拟合仍被向下拉；$\lambda$ 越大 $\sigma$ 越小、可用 e-fold 越少，偏差越大
   （−8.4 % → −10.2 % → −18.9 %），与观测一致。
2. 拟合窗口敏感性大：同一算例 $\lambda=32$ 用 $a/\lambda\le0.035$ 与 $\le0.08$ 分别得
   $0.85\sigma$ 与 $1.08\sigma$（`runs/tmp_probe/rt_final_sweep.log`）。$a/\lambda\gtrsim0.06$ 后
   非线性加速明显（RT 气泡加速），故窗口上限取 0.05。
3. **溃坝的浅水 LBM 路线被排除**：D1Q3 “非理想压力力”模型需要在低 Froude 区抵消
   $\sim2/(3gh)$ 倍大的格点压力项，干湿前沿数值失稳（爆炸/负分布函数）；改为有限体积
   Rusanov 作为参考浅水求解器，并把该排除记录在模块 docstring 与 README。
4. **几何尖端跟踪 ~25 % 偏低是方法固有**：一阶格式在 $h\to0$ 尖端的数值扩散；
   细化只缓慢改善，故不作为判据，仅透明报告。
5. 适用范围：RT 结论仅在线性区（$a/\lambda\le0.05$，$Ma\lesssim0.1$）；溃坝在干床、无摩阻、
   $Fr$ 与网格足够（$dx\le0.01h_0$）时成立。

## ⑥ 误差判定（逐项、显式回答“是否 <3%”）

| 关键量 | 基准 | 误差 | 是否 < 3 % |
|---|---|---|---|
| $\sigma$（$\lambda=32$） | $\sqrt{Agk}=0.025583$ | −8.40 % | **否** |
| $\sigma$（$\lambda=64$） | 0.018090 | −10.15 % | **否** |
| $\sigma$（$\lambda=128$） | 0.012792 | −18.91 % | **否** |
| 网格收敛观测阶（GCI） | 要求 ≥1.5 | 误差非单调降（0.086→0.108→0.199），观测阶无定义 | **否**（无法给出 ≥1.5 的阶） |
| 前沿位置（质量积分，nx=1600，max） | $2\sqrt{gh_0}t$ | **+3.68 %** | 否（但满足本单 ≤5 % 判据） |
| 前沿位置（质量积分，nx=400/800） | 同上 | +6.41 % | **否**（粗网格，收敛研究） |
| 总质量守恒（溃坝，三档） | 初始质量 | ≤1.1e−13 % | **是**（≪1 %） |
| 相场质量守恒（RT） | 初始相场质量 | ≤1.5e−3 % | **是** |
| 自相似剖面 $L_2$（nx=1600） | Ritter 剖面 | 5.91 % | **否** |

### 结论（honest，不粉饰）

* **溃坝 Ritter 前沿律：PASS**（声明参考分辨率 nx=1600，最大误差 3.68 % ≤ 5 %），
  质量守恒 PASS（机器精度）。
* **RT 线性增长率 3 % 门：FAIL**（−8.4 %/−10.2 %/−18.9 %），网格收敛阶亦无法给出 ≥1.5。
  失败原因已定位到“可用 e-fold 太少 + 扩散界面阻尼”的测量/模型层面，并给出可执行的改进路径
  （数值本征模初值 / 模纯化 / 界面锐化），未在本单实现。
* $\lambda=128$ 曾出现长时程相场失稳（$\tau_\phi\to0.5$ 近无扩散极限），已通过采样设置复现出
  稳定算例并以 `phase_mass_drift_pct` 显式监控；不稳定那次结果不参与统计。

## 附：新增/修改文件

* 新增 `src/tensorlbm/rayleigh_taylor.py`、`src/tensorlbm/shallow_water.py`
* 新增 `benchmarks/rayleigh_taylor/{growth_rate.py, make_figures.py, result.json, README.md, runs/, figures/}`
* 新增 `benchmarks/dam_break_ritter/{self_similar.py, make_figures.py, result.json, README.md, runs/, figures/}`
* 修改 `src/tensorlbm/dam_break.py`：`_find_front_x` → 公共 `find_front_x`（共享溃坝工装）
* 新增测试 `tests/test_rayleigh_taylor.py`、`tests/test_shallow_water_ritter.py`、
  `tests/test_bench_rt_ritter.py`