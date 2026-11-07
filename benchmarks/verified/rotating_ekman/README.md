# 旋转参考系 Ekman 层（D3Q19）— 解析螺旋 benchmark

**判定: 达标 (VERIFIED)** — $\delta_E$ 相对误差 3.1 %（判据 < 5 %）、五点剖面最大误差 2.6 %（判据 < 3 %）。

## 解析解

旋转参考系中，无滑移平板（在旋转系内静止）之上的定常线性边界层。
Coriolis 参数 $f=2\Omega_z$，常体力 $\mathbf F=(0,fU_g,0)$ 扮演压强梯度，
其地转平衡态为均匀流 $(U_g,0,0)$：$-2\boldsymbol\Omega\times\mathbf u_g+\mathbf F=0$。
令 $\zeta=z/\delta_E$、$\delta_E=\sqrt{2\nu/f}$（等价写法 $\sqrt{\nu/\Omega_z}$）：

    u_∥(ζ) = U_g [1 − e^{−ζ} cos ζ]      （沿地转/切向分量）
    u_⊥(ζ) = U_g  e^{−ζ} sin ζ           （跨等压线/径向分量）

速矢端迹为指数螺旋：每深入一个 $\delta_E$ 转动 1 rad，$z=\pi\delta_E$ 处速度矢量首次与
地转方向平行（`spiral_phase_depth`）。线性平衡 $\nu w''=ifw$（$w=u_∥+iu_⊥$ 的扰动量）
在 `tests/test_rotating_frame.py` 里用二阶有限差分固化。

## 仿真构成（真实仿真，无外推修正/无人工因子）

- D3Q19 BGK（$\nu=0.1$，$\tau=0.8$）+ Guo(2002) 力项，半程速度不动点迭代 2 次；
  网格 $(n_z,4,4)$，$x/y$ 周期（流动水平均匀，4×4 列是精确的）；
  $z=0$ 半程反弹无滑移板，$z=H$ **自由滑移**盖板（不产生自己的 Ekman 层，$H=9\delta_E$
  处的残余为 $e^{-9}$）。
- $\delta_E$ 由格子参数反解：$f=2\nu/\delta_E^2$，因此解析厚度恰为 `--delta-nodes` 格，
  判据检验的是求解器而不是人工调参。
- 起动方式两种并存：`--init rest`（从静止起动的独立复算）与 `--init analytic`
  （把解析剖面放进初值，检验它是否为离散定常解 —— 不动点检验）。
- 收敛判据：**惯性周期平均残差** $|u(t)-u(t-T_f)|<\text{tol}$，$T_f=2\pi/\Omega_z$。
  旋转流中瞬时残差含不衰减的惯性振荡，用瞬时残差会逼出无意义的超长积分。

## 结果（`records/ekman_delta16_fixedpoint.json`）

| 量 | 数值 |
|---|---|
| $\delta_E$ 拟合 / 解析 | 见 JSON `fit.delta_fit` / `fit.delta_ana` |
| $\delta_E$ 相对误差 | 3.13 %（判据 5 %） |
| 五点（$\zeta$=0.5,1,1.5,2,2.5）剖面对比最大误差 | 2.63 %（判据 3 %） |
| 整层（$1\le z\le2.5\delta_E$）最大误差 | 2.66 % |
| 拟合 $U_g$ / 解析 | 与 0.02 同量级，见 JSON |

**尺度映射**（`lattice_scaling`，程序化输出）：物理 $\nu=0.01$ m²/s、Coriolis 参数
$f=0.01$ s⁻¹（即框架角速度 $\Omega=0.005$ rad/s）→ $\Delta x=0.0707$ m、
$\Delta t=0.05$ s、$\delta_E=1.414$ m。战役给出的 $\delta_E=\sqrt{2\nu/\Omega}$ 与
地球物理标准 $\delta_E=\sqrt{2\nu/f}$ 在「战役 $\Omega$ 即 Coriolis 参数 $f$」的约定下逐字一致；
`lattice_scaling` 同时输出 $f\Delta t$ 闭合校验（$f_\text{phys}\Delta t=f_\text{lat}$）。

## 踩坑记录（本案例暴露的两个真实缺陷）

1. **Guo 半程速度被累积**：初版把 $u^*=u+F/(2\rho)$ 写在迭代循环里累加、且只在
   $\Omega\ne0$ 分支执行 —— 有效力偏大 3/8，Ekman 厚度系统性偏薄 18.5 %（$13.03$ vs $16$ 格），
   纯体力（$\Omega=0$）时有效力则只有 $3/8$。修正方式：每轮迭代都从 $u_\text{raw}$ 出发，
   并在 $\Omega=0$ 时同样执行。回归哨兵 = 体压力驱动的 Poiseuille 剖面测有效黏度
   （修正后 $\nu_\text{eff}/\nu$ 与 1 差 0.4 %，修正前 2.68 倍）。
2. **全实体行做横向平均会 0/0**：平板行/盖板行没有流体格点，`sum/count` 得 NaN 并污染整条剖面。
   现在对计数 `clamp(min=1)`，并用 `[1:-1]` 参与拟合。

## 复现

    python benchmarks/verified/rotating_ekman/run.py single out.json \
        --delta-nodes 16 --init analytic --threads 2 --figure out.png
    python benchmarks/verified/rotating_ekman/run.py scan scan_dir \
        --delta-nodes 12 16 24

## 局限

- $\delta_E$ 只有 12–24 格分辨，二阶精度下残差量级 1–3 %（与实测一致）。
- 只验证线性（$Ro\to0$，$Re_\delta=U_g\delta/\nu=2$）区；非线性 Ekman 层/湍流 Ekman 层不在本案例范围。