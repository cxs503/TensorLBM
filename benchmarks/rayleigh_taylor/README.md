# Rayleigh–Taylor 线性增长率基准（重力驱动自由面解析锚）

## 1. 物理背景

重流体（$\rho_1$）位于轻流体（$\rho_2$）之上，重力 $g$ 向下。界面带单模扰动
$\lambda$、振幅 $\eta_0=\lambda/100$。无粘、不互溶、不可压两层流的线性增长率（Rayleigh 1883；
Taylor 1950）为

$$
\sigma_{\rm analytic}=\sqrt{A g k},\qquad
A=\frac{\rho_1-\rho_2}{\rho_1+\rho_2},\qquad k=\frac{2\pi}{\lambda}.
$$

计入等运动粘度 $\nu$ 的精确线性结果为
$\sigma_{\rm visc}=\sqrt{Agk+\nu^2k^4}-\nu k^2$（Chandrasekhar 1961, §92）。

本基准取 $\rho_1/\rho_2=2$ → $A=1/3$，$g=10^{-2}$（格子），$\tau=0.51$（$\nu=3.33\times10^{-3}$），
$\lambda\in\{32,64,128\}$。

## 2. 数值模型（`src/tensorlbm/rayleigh_taylor.py`）

* **流动**：D2Q9 + BGK，只施加**Boussinesq 浮力偏差** $F_y=-Ag\phi$（平均静水压部分扣除，
  参考密度恒为 1，故无 $\rho_0 g H$ 压缩伪影）；上下为半程反弹壁，$x$ 方向周期（一波长一盒）。
* **相场**：第二套 D2Q9 分布 $g_i$，平衡态 $g_i^{eq}=w_i\phi(1+3c_i\!\cdot\!u)$，
  扩散率 $D=c_s^2(\tau_\phi-1/2)$；界面 $\phi=\tanh((y-y_{\rm int})/\delta)$，$\delta=1.5$。
* **初值**：线性本征模态速度场 $u_y=\sigma\eta\cos kx\,e^{-k|y-y_{\rm int}|}$、
  $u_x=\pm\sigma\eta\sin kx\,e^{-k|y-y_{\rm int}|}$，使只激发增长模（无衰减模瞬态）。

## 3. 增长率测量协议

界面形状由 $\phi=0$ 等值线逐列线性插值得到 $\eta(x,t)$，取第一 Fourier 模 $a(t)$。
在窗口 $a/\lambda\in[\eta_0/\lambda,\,0.05]$ 内用**双模投影**
$a(t)=c_+e^{\sigma t}+c_-e^{-\sigma t}$ 拟合 $\sigma$（对 $\sigma$ 一维扫描，$c_\pm$ 线性最小二乘）。
双模基可吸收脉冲启动产生的衰减模，避免 $\ln a$–$t$ 直线拟合对 $\sigma$ 的系统低估。
另一独立估计量（重流体柱高守恒矩）用于交叉检查。

## 4. 结果（`result.json`，驱动 `growth_rate.py`）

| $\lambda$ | $\sigma_{\rm sim}$ | $\sqrt{Agk}$ | 比值 | 相对误差 | $R^2$ |
|---|---|---|---|---|---|
| 32 | 0.02344 | 0.02558 | 0.916 | −8.4 % | 0.999 |
| 64 | 0.01625 | 0.01809 | 0.899 | −10.2 % | 0.999 |
| 128 | 见 `result.json` | 0.01279 | — | — | — |

**判定（honest）**：增长率 3 % 门 **未通过**（三档 $\lambda$ 均在门外的 8–20 % 区间），
因此网格收敛阶 GCI 也无法给出有意义的 ≥1.5。物理量级、指数律（$R^2>0.999$）、
$1/\lambda$ 趋势与粘性阻力方向均正确，但存在系统性偏低。已定位的偏差来源：

1. **启动瞬态**：$\eta_0=\lambda/100$ 与线性窗口上限 $0.05\lambda$ 之间只有 $\ln 5\approx1.6$
   个 e-fold，衰减模（幅值与初值失配同量级）无法衰减完，双模拟合仍被拉低；
   该效应随 $\lambda$ 增大（$\sigma$ 变小、e-fold 更少）而加重，与观测的
   −8.4 % → −10.2 % 趋势一致。
2. **扩散界面阻尼**：相场扩散使 $\delta$ 随时间/分辨率变化，$k\delta$ 有限带来 $\sigma$ 的下修。
3. $\lambda=128$ 的长时程算例在 $\tau_\phi\to0.5$ 的近无扩散极限下出现相场不稳定
   （`phase_mass_drift_pct` 非有限值），结果已如实标注为无效，不参与收敛阶统计。

改进方向（已在 README 中记录，未在本单实现）：把初值改为**扩散界面模型的数值本征模**
（用短预热 + 幅度重整的“模纯化”），或改用 $k\delta\ll1$ 的界面锐化格式。

## 5. 复现

```bash
PYTHONPATH=src python benchmarks/rayleigh_taylor/growth_rate.py          # 全量三档
PYTHONPATH=src python benchmarks/rayleigh_taylor/growth_rate.py --fast   # 冒烟
PYTHONPATH=src python benchmarks/rayleigh_taylor/make_figures.py         # 出图
PYTHONPATH=src python -m pytest tests/test_rayleigh_taylor.py -q
```

产物：`result.json`（机读判据）、`runs/lam*/growth_history.csv`、`runs/lam*/case.json`、
`figures/rt_amplitude_growth.png`、`figures/rt_convergence.png`。