# 旋转槽 source-sink 流（D3Q19）— Taylor–Proudman 柱状径向流 benchmark

**判定: 部分达标 (PARTIAL)** — 径向幂律指数误差 0.043（判据 ≤ 0.05，通过）；
轴向均匀性 12.1 %（判据 ≤ 5 %，**未通过**，物理原因见下）。

## 预期物理

封闭圆柱槽以 $\Omega_z$ 旋转，中心区注入、外缘环带抽出，强度 $Q$ 相等。
环带间的质量守恒要求

    v_r(r) = Q / (2 π r H ρ)  ∝ 1/r,

而慢速定常（$Ro=v_r/(2\Omega L)\ll1$）下的 Taylor–Proudman 定理 $\partial u/\partial z=0$
要求内区径向速度沿轴向不变 —— 即旋转槽里"源—汇"流呈贯穿全深的柱状
（Hide 1968；Greenspan 1968 ch.3）。端盖埃克曼层承担把流体"带回共转"的任务，
其泵吸量级为 $\sqrt{E}$，$E=\nu/(\Omega H^2)$。

## 仿真构成

- 体素圆柱槽：$r>R$ 为固壁、$z=0$ 与 $z=n_z-1$ 为随槽旋转的端盖（旋转系内静止 → 无滑移），
  $x/y$ 周期（整圈固壁在盒内，等价无穷远侧壁）。
- **源/汇是体积质量源** $S(\mathbf x)$（每步每格质量），以 $S w_i$ 加入碰撞后分布：
  因 $\sum_i c_i w_i=0$，只注入质量不注入动量，连续性变为
  $\partial_t\rho+\nabla\!\cdot(\rho\mathbf u)=S$，动量方程不变。
  $v_r$ 完全由解给出，不作任何规定。
- $\nu=0.1$（$\tau=0.8$）、$\Omega_z=0.01$、$Q=1.5$、$n_z=24$、槽半径 13.5、源 $r<3.5$、汇宽 3。
- 收敛判据：惯性周期（$T_f=2\pi/\Omega=628$ 步）平均残差 $<5\times10^{-7}$，最少 6 个周期。

## 结果（`records/source_sink_34x34x24.json`）

| 量 | 实测 | 判据 | 结论 |
|---|---|---|---|
| 径向幂律指数 $n$（$v_r=ar^n$） | **−1.0434** | $\|n+1\|\le0.05$ | ✅（0.043） |
| 内区轴向均匀性 $\max\lvert v_r-\bar v_r\rvert/\bar v_r$ | **12.1 %** | ≤ 5 % | ❌ |
| 拟合 $R^2$ | 见 JSON `radial_profile.r2` | — | — |
| 相对涡旋 $\max\lvert v_\theta\rvert/(\Omega r)$ | 见 JSON `checks.max_relative_swirl` | — | — |

## 为什么轴向均匀性不达标（物理诊断，不做粉饰）

端盖埃克曼泵吸把内区径向速度"吸"向端盖，量级即为 $\sqrt E$。本算例
$E=\nu/(\Omega H^2)=0.1/(0.01\cdot24^2)=1.7\times10^{-2}$，$\sqrt E=13.1\%$ ——
实测 12.1 % 与之一致。要把轴向非均匀性压到 5 % 以下需 $\sqrt E\lesssim0.05$，即

- 保持端盖埃克曼层可分辨（$\delta_E=\sqrt{\nu/\Omega}\gtrsim3$ 格 → $\nu\gtrsim0.09$@$\Omega$=0.01）、
- 同时 $H\gtrsim\sqrt{\nu/(0.0025\,\Omega)}\approx 60$ 格（$n_z\ge60$）。

即"高而窄"的槽 ($n_z\approx60$–64，$\nu\approx0.1$)，算例已在
`run.py single --nz 60 …` 给出（本战役预算内该配置的计算尚未跑完，属未完成部分）。
这是埃克曼数—分辨率的固有张力，不是求解器缺陷：同一求解器在 Ekman 案例里
（$\delta_E$=16 格、$\sqrt E$ 无关）达到 3 % 精度。

## 踩坑记录

1. **体积分注入的收敛判据必须相对化**：残差是绝对速度差，$v_r\sim10^{-3}$ 时
   绝对门限 $10^{-4}$ 会让"几乎静止"的初值状态被判为收敛（曾出现 2028 步假收敛，
   轴向非均匀性 157 %）。现在最少积分 6 个惯性周期并配合 $10^{-7}$ 量级门限。
2. **壳层半径范围**：拟合幂律需要足够的杠杆臂（$r_\text{out}/r_\text{in}\gtrsim3$），
   小域 + 大源半径会让壳层退化（曾得到 $-0.65$ 的伪指数）；现在若
   $r_\text{out}\le1.5r_\text{in}$ 直接报错退出。
3. **质量记账的 float32 精度**：$S\sim10^{-4}$ 相对总质量 $10^{2}$ 时，
   float32 下 $\Delta M$ 的测量误差约 2 %（见单测，用 float64 才精确）。

## 复现

    python benchmarks/verified/rotating_source_sink/run.py single out.json \
        --ny 34 --nx 34 --nz 24 --nu 0.1 --omega 0.01 --Q 1.5 --tol 5e-7 --threads 4
    python benchmarks/verified/rotating_source_sink/run.py single tall.json \
        --ny 34 --nx 34 --nz 60 --nu 0.1 --omega 0.01 --Q 1.5 --min-periods 14