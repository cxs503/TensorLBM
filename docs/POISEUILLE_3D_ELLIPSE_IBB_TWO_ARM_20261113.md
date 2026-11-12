# 椭圆管 Poiseuille：半程反弹 vs 亚格 SDF 插值反弹（两臂对照）

**节点**：tp-ds-3 自找活 · mission 16685 · 分支 `agent/m16685` · 仓 TensorLBM
**产物**：`benchmarks/pending/poiseuille_3d_ellipse/run_ibb.py`（两臂驱动）、`ibb_result_{a8,a20,circle_a20}.json`（实测）、
`ibb_tables.md` + `figures/ibb_two_arm.png`（程序化注入）、`tests/test_poiseuille_3d_ellipse_ibb.py`（4 项）
**上游**：`benchmarks/pending/poiseuille_3d_ellipse/README.md`（m6422）、`docs/POISEUILLE_3D_ELLIPSE_REFERENCE_GEOMETRY_AUDIT.md`

**一句话结论（honest-FAIL）**：在**同一驱动、同一测量算子**下，亚格 SDF 二点 Bouzidi 插值反弹
在椭圆柱上**没有**消除楼梯壁偏置——θ_sQ 由 10.70 % 升到 15.35 %（a=20），有效壁面外移由
+0.68 格增到 +1.53 格；圆管对照臂方向相反（θ_sQ 10.62 % → 4.50 %，但残余 +0.28 格外移）。
故「插值反弹可修复椭圆管楼梯偏置」这一假设在本装置上被**否定**，并留下可复现的诊断入口。

## ① 案例介绍（物理背景/几何/参数）

三维直管、椭圆截面（半轴 a:b = 2:1，y 向 a、z 向 b）内充分发展层流，沿 x 施加均匀体积力 g，
x 向周期（充分发展 ⇒ 解与 x 无关，nx 只需覆盖 D3Q19 模板，不是分辨率旋钮）。

解析解（White, *Viscous Fluid Flow*, 3rd ed., §3-3；μ = ρν，G = ρg）：

    u(y,z) = U_max (1 − (y−y_c)²/a² − (z−z_c)²/b²)
    U_max  = g a²b² / (2ν(a²+b²)),      Q = π a b U_max / 2

| 参数 | 值 |
|---|---|
| 截面半轴 | a=8/20（b=a/2），圆管对照 a=b=20 |
| 格子 / 碰撞 | D3Q19 / BGK，τ = 0.8（ν = 0.1） |
| 体积力 g | 由 `ellipse_g(a,b,ν,U_max)` 反解，U_max = 0.02（Ma = 0.035） |
| x 向 | 周期，nx = 6 |
| 域 | ny = 2a+3, nz = 2b+3（与 m6422 基线同一格点约定，1 格外实壁余量） |
| 步数 | 3000–6000 步 + 100–200 步时间平均 |

**两臂**（其余全部相同）：
- `halfway` = 产线算子 `boundaries3d.bounce_back_cells_3d`（无滑移面在楼梯面）；
- `ibb` = `boundary.interpolated_bounce_back.interpolated_bounce_back_3d`
  （二点 Bouzidi 闭合，无滑移面在**连续椭圆**上，q 由亚格符号距离 sdf = −h/|∇h| 给出）。

## ② 怎么做的（方法/流程/关键设置）

1. 同一 `ellipse_setup()` 生成 sdf / fluid / solid / λ；`halfway` 与 `ibb` 两臂共用同一张量、
   同一力源 `s_i = 3 w_i (c_i·F)`、同一初始平衡态；
2. 每步 `collide_bgk3d + src → stream3d → 壁面闭合`（`ibb` 传入 post-collision `f_pre` 与
   streamed `f_new`，与 `benchmarks/pending/staircase_wall_offset/run_inclined.py` 的 pipe3d 用法一致）；
3. x 向平均后得到截面 u(y,z)，再用**与 m6422 相同**的单一积分观测量判据：
   `s^Q = sqrt(2Q/(πabU_max))`，λ = (y/a)²+(z/b)² 分环比较 `U_max(1−λ/s^Q²)`，给出 θ_sQ；
   另给名义几何误差 θ_nom（s=1，对连续椭圆）、各向异性双参数拟合诊断 (a_fit, b_fit)；
4. 表与图由 `make_ibb_figures.py` 从 JSON 程序化生成（零手抄）。

## ③ 计算结果

完整表见 [`benchmarks/pending/poiseuille_3d_ellipse/ibb_tables.md`](../benchmarks/pending/poiseuille_3d_ellipse/ibb_tables.md)（程序化注入），图见
[`figures/ibb_two_arm.png`](../benchmarks/pending/poiseuille_3d_ellipse/figures/ibb_two_arm.png)。

| 案例 | 臂 | θ_nom [%] | θ_sQ [%] | δ_a^eff (格) | δ_b^eff (格) | a_fit−a | b_fit−b |
|---|---|---|---|---|---|---|---|
| 椭圆 a=8 b=4 | halfway | 20.906 | 2.356 | +0.252 | +0.126 | +0.604 | −0.076 |
| 椭圆 a=8 b=4 | ibb | 67.437 | 15.358 | +1.175 | +0.587 | +0.751 | +0.271 |
| 椭圆 a=20 b=10 | halfway | 25.640 | 10.700 | +0.682 | +0.341 | +1.071 | −0.115 |
| 椭圆 a=20 b=10 | ibb | 41.950 | 15.354 | +1.531 | +0.765 | +0.540 | +0.408 |
| 圆 R=20（对照） | halfway | 9.174 | 10.615 | −0.121 | −0.121 | +0.273 | +0.273 |
| 圆 R=20（对照） | ibb | 16.339 | 4.496 | +0.280 | +0.280 | +0.501 | +0.501 |

（以上数值与 `ibb_tables.md` 同源；本表为可读摘要，正式引用以 JSON 为准。）

## ④ 结果图

`figures/ibb_two_arm.png`：(a) a=20 椭圆截面 u/U_max vs λ，解析抛物线与两臂叠图；
(b) 各臂 θ_sQ 与 δ_a^eff 条形图（含 3 % 判据线）；(c) 圆管对照臂剖面。

## ⑤ 注意点（坑/局限/适用范围）

1. **不可与 m6422 基线直接比数**：m6422 是**速度入口**驱动（Zou-He 进/出口，U_max = 2u_in），
   本单是**体积力 + x 周期**驱动；同一 a=20 几何上本装置 halfway 臂 θ_sQ = 10.700 % 而非记录的
   5.107 %，δ_a^eff = +0.682 格而非 +0.225 格 ⇒ 本装置**未复现**记录值。单臂之间的
   *相对* 比较是受控的（同驱动/同算子），但本单不对 m6422 的判决做任何覆盖或重判。
2. 体积力以朴素源项 `s_i = 3 w_i (c_i·F)` 注入（未做 Guo 半步力修正），两臂同等承受，故只影响
   绝对 δ 标定，不影响臂间差；但这解释了 δ 量级与 verified/poiseuille_3d_pipe（R_eff−R ≈ +0.11）不可比。
3. `converged=False`（触 max_steps），a=20 的 3000 步与 6000 步两次运行同名指标在小数点后三位一致，
   视为统计稳态（第二次运行未归档为工件，按 not_archived 如实标注）。
4. **a=40 档 not_run**（本机 load15 ≈ 90–128 已触容量红线，单次调用 120 s 收割；按容量纪律未起重算力）。
5. 亚格 sdf 是一阶 Taylor 距离（`docs` 测试实测近壁最大偏差 0.095 格 @ a=12），对 Bouzidi 权重够用，
   但不构成"精确距离"。

## ⑥ 误差判定（逐项回答"是否 <3 %"）

| 关键量 | 数值 | 基准 | 是否 <3 % |
|---|---|---|---|
| 椭圆 a=20 θ_sQ（halfway） | 10.700 % | 0（解析） | **否** |
| 椭圆 a=20 θ_sQ（ibb） | 15.354 % | 0（解析） | **否**（比 halfway 更差 4.65 pp） |
| 椭圆 a=8 θ_sQ（halfway → ibb） | 2.356 % → 15.358 % | 0 | halfway 是；ibb **否**（劣化 13.0 pp） |
| 圆 R=20 θ_sQ（halfway → ibb） | 10.615 % → 4.496 % | 0 | 两臂均否，但 ibb **改善 6.12 pp** |
| 圆 R=20 δ_eff（解析应为 ~0） | ibb +0.280 格 | 0 | **否**（残余外移） |
| 椭圆各向异性 δ_a/δ_b（halfway） | +0.682 / +0.341 = 2.00 | 2（几何比） | 各向同性缩放（未消除各向异性） |
| 椭圆各向异性（ibb） | +1.531 / +0.765 = 2.00 | 2 | 同上，且量级翻 2.2× |
| 解析参考自检（Q 两条公式） | 相对差 < 1e-12 | White §3-3 | **是** |
| 近壁 sdf vs 精确椭圆距离 | 最大 0.095 格 | 0 | 是（<0.12 格，一阶余项） |

**判定**：假设「亚格 SDF 二点 Bouzidi 插值反弹可消除椭圆管楼梯壁偏置、使 θ ≤ 3 %」→ **FALSIFIED**。
圆管臂显示闭合本身在**形状**上有效（θ_sQ 减半），但在椭圆上被某种系统性外移（+1.5 格）压倒。

## ⑦ 后续线索（未完成部分）

1. **分支隔离实验（首选）**：在 45° 斜壁通道上分别统计 q<1/2 与 q≥1/2 两支的贡献
   （`interpolated_bounce_back` 的 `return_report` 已给 `fallback_links`），判断外移是否来自
   「q<1/2 且身后节点为固体」的回退支路在薄椭圆上占比过高；
2. **外移符号复现**：把同一两臂装置跑 `verified/poiseuille_3d_pipe` 同款入口驱动，检查 pipe3d
   的 interp 臂是否也出 +0.3 格外移（若复现，则问题在算子应用层而非几何）；
3. 若确认算子缺陷 → 建议新锚点 `GAP-LBM-IBB-CURVED-WALL-OFFSET`（量化验收：圆管 ibb 臂
   |δ_eff| ≤ 0.1 格、椭圆 a=20/40 θ_sQ ≤ 3 %）。

## 复现

```bash
cd benchmarks/pending/poiseuille_3d_ellipse
PYTHONPATH=<repo>/src python run_ibb.py --a 20 --max-steps 3000 --acc-steps 100 --out ibb_result_a20.json
PYTHONPATH=<repo>/src python run_ibb.py --a 20 --ratio 1.0 --max-steps 2000 --acc-steps 100 --out ibb_result_circle_a20.json
PYTHONPATH=<repo>/src python make_ibb_figures.py
PYTHONPATH=<repo>/src python -m pytest tests/test_poiseuille_3d_ellipse_ibb.py -q
```