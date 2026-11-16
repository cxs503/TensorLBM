# B14-ellipse / body-force caliber — 3D 椭圆管 Poiseuille（D3Q19）

**状态**：`run.py` + 判据（冻结于跑档之前）+ 网格阶梯 `ladder/`（20/40/80，a:b=2:1）。
晋级判据与结果见 `ladder/result.json` 的 `gates` 节；判定由 `build_summary` 从原始
per-grid 数字程序化重算，`tests/test_poiseuille_3d_ellipse_bodyforce.py` 会独立复核该
判定与门限一致（防"存储旗标漂移"）。

## 为什么在速度入口口径之外再加一个口径

`../poiseuille_3d_ellipse/` 用「均匀速度入口 + 压力出口 + L=6a 长管」驱动，比较时用
**单尺度面积等效** `s^Q`（圆管 `R_eff^Q` 的类比），记录为 `not_verified`：a=20 给 5.11 %，
a=40 给 2.43 %，误差随 1/a 单调但只有一档过 3 % 门。该口径里有两项**不属于被测物理**
的东西：

1. 流向发展长度（入口段/体松弛）会污染中截面剖面；
2. 单尺度 s^Q 只能表达各向同性壁面偏移，而数字楼梯椭圆壁是**各向异性**的。

本口径把两项都去掉：

* x 方向**周期** + 均匀**体力** `g`（体力在 LBE 里等价于 `-∂p/∂x`，周期性要求平均压力梯度
  为零，动量平衡完全由体力与粘性扩散闭合）；中截面即处处充分发展；
* 比较基准是**名义半轴 (a, b) 的解析解**，不使用任何拟合/标定尺度；
  另用**两个积分观测量**（总流量 Q、中心速度 Uc）水力反演有效半轴 (a_e, b_e) 作为主指标
  的几何标定——这正是 `../poiseuille_3d_ellipse/README.md` 列出的晋级路径 2，剖面形状
  从不参与反演。

## 物理与解析解

稳态、充分发展、单向流动（对流项恒为零 ⇒ 该 Poiseuille 解对任意 Re 都是 Navier-Stokes
的精确解；Batchelor 1967 §4.2；White, *Viscous Fluid Flow* §3.3）：

```
u(y,z) = Uc (1 - y'^2/a^2 - z'^2/b^2)
Uc     = g a^2 b^2 / (2 nu (a^2 + b^2)),    nu = (tau - 1/2)/3
Q      = pi a b Uc / 2          （椭圆抛物面均值恰为 Uc/2）
dp/dx  = -rho g
```

代入检验：`nu (u_yy + u_zz) + g = nu Uc (-2/a^2 - 2/b^2) + g = 0`。
`tests/...::test_analytic_satisfies_stokes_balance` 用有限差分重做该代入，
`test_flow_rate_identity` 用椭圆极坐标积分复核 `Q = pi a b Uc/2`。

## 数值设置（三档只变分辨率）

| 项 | 值 |
|---|---|
| 晶格 / 碰撞 | D3Q19 / BGK（`solver3d.collide_bgk3d`） |
| 迁移 | `solver3d.stream3d`（x 周期，`nx = 4`：流动单向且无 x 依赖） |
| 壁面 | `boundaries3d.bounce_back_cells_3d`，掩膜=**纯固相格**（迁移后整组交换 ⇒ 无滑移面落在流-固连线中点，标准 half-way 反弹；把流相边界格也放进掩膜会双反射） |
| 驱动力 | Guo 2002 零和源项（脚本内显式，库只带 D2Q9 版），`F = g e_x`，`u = Σc f + F/(2ρ)` |
| 初始条件 | **静止**（u=0）——稳态幅值由动力学自己长出来，不种解析解 |
| τ / ν / g | 0.8 / 0.1 / 1.25e-5（Ma = Uc/√(1/3) ≈ 8.7e-3，压缩性误差可忽略） |
| 稳态判据 | 每 500 步的中心速度相对变化 ≤ 1e-7（连续三窗） |
| 网格阶梯 | a:b = 2:1；a = 20, 40, 80（b = 10, 20, 40） |
| 外推 | **无**（零网格外推，verdict 直接来自实测） |

## 冻结判据（跑档前写定）

* **G1** 主指标 `bin_cal`（λ 分箱椭圆平均剖面、中央区 `u_ana > 0.2 Uc` 的最大相对误差，
  基准 =(a_e,b_e) 解析解）**每一档** ≤ 3 %
* **G2** `bin_cal` 随 a 严格单调下降且档数 ≥ 3
* **G3** `cell_cal`（逐格最大，最严）严格单调下降
* **G4** 壁面偏移物理性：`0 < a_e - a ≤ 1` 且 `0 < b_e - b ≤ 1` 格（half-way 反弹把无滑移面
  放在一格之内侧；这是可被证伪的定量预测）
* **G5**（披露、不参与判定）裸名义几何误差 `bin_raw` 为 O(1/a)，最细档 ≤ 3 %

`verified = G1 ∧ G2 ∧ G3 ∧ G4`。

## 运行

```bash
cd benchmarks/pending/poiseuille_3d_ellipse_bodyforce
python run.py scan ladder --a 20 40 80 --threads 8      # 三档阶梯 → ladder/result.json
python run.py single 40 /tmp/c40.json --threads 8       # 单档
python -m pytest tests/test_poiseuille_3d_ellipse_bodyforce.py -q
```

工件：`ladder/result.json`（判定 + 门限 + 逐档指标 + 观测阶）、`ladder/case_a*_b*.json`
（逐档原始记录：分箱剖面、观测历史、反演结果）。
