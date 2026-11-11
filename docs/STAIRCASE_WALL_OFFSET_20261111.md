# Staircase / inclined wall effective wall offset: interpolated vs half-way bounce-back

**状态：❌ 主判据未达成（honest FAIL）——2 点插值反弹（Bouzidi）在本 45° 通道上没有把 "
"effective wall offset 清零，theta 反而略升。** 算子正确性（q=1/2 退化为半程反弹、
静水不动点、算力预算守恒）已由单测钉住；失败的是「插值=一阶壁面位置修正」这一假设本身。

## 1 案例介绍
物理：45° 斜壁平面 Poiseuille 通道（双周期盒内，壁法向坐标 s = wrap(x−y)/√2，
流体 |s| ≤ H），体积力沿壁面轴向 (1,1)/√2 驱动，无滑移壁。
解析：u(s) = g(H²−s²)/(2ν)，Q = 2gH³/(3ν)（White, *Viscous Fluid Flow*，充分发展层流）。
参数表：tau = 0.8（ν = 0.1），u_max^target = 0.02（g = 2νu_max/H²），台阶数 1500 + 100 步时均，
两档分辨率 (L=40,H=8)、(L=56,H=11)；两次独立观测量 = 体积流量 Q（→δ_eff）与分环剖面（→theta）。

## 2 怎么做的
新增产品级算子 `src/tensorlbm/boundary/interpolated_bounce_back.py`：2 点 Bouzidi 线性闭式，
`q` 由亚格子符号距离场（SDF）沿 link 线性插值得到（q = s_f/(s_f−s_s)），可作半程反弹的
drop-in 替换（post-streaming 两时间层形式；q=1/2 时与 `bounce_back_cells*` 逐位相同）。
驱动 `benchmarks/pending/staircase_wall_offset/run_inclined.py`：同一 solid 掩码上跑
halfway / interp 两种壁面格式，单一观测量 δ_eff = H_eff − H（H_eff 由 Q 反解，不拟合剖面）。
工具链：torch（CPU）+ 仓内 collide/stream/BC 原语，Guo 型动量源 Σ s_i c_i = F 精确。

## 3 计算结果（程序化注入，零手抄）
| L | H | wall | delta_eff (cell) | theta (%) | Q/Q_ana | u_max / u_max^ana |
|---|---|---|---|---|---|---|
| 40 | 8 | halfway | +0.3104 | 8.843 | 1.1210 | 0.02155 / 0.02000 |
| 40 | 8 | interp | +0.3355 | 9.506 | 1.1312 | 0.02168 / 0.02000 |
| 56 | 11 | halfway | -0.0189 | 3.182 | 0.9949 | 0.01958 / 0.02000 |
| 56 | 11 | interp | +0.1708 | 6.816 | 1.0473 | 0.02023 / 0.02000 |

## 4 结果图
![inclined](attachment:staircase_offset_inclined.png)
左：L=40 档剖面（halfway / interp）vs 解析抛物线；右：δ_eff 随名义 H 的走势（虚线=0 基准）。

## 5 注意点 / 局限
1. **主判据未达成**：interp 的 |δ_eff| ≤ 0.05 只在单档附近，theta ≤ 3 % 未达成（6.8 %/9.5 %）。
   与假设相反，插值在本案上使 theta 从 3.18 %升到 6.82 %（L=56）。
2. **可能的机制（未证）**：45° 楼梯壁的切向滑移不是「壁面沿法向平移」可表达的量——
   插值把每条 link 的壁面位置搬到真实壁上，但楼梯壁的 *corner* 处 link 集合本身不连续，
   一阶插值在 corner 附近引入新的偏差；另外 `q<1/2` 分支需要壁后节点为流体，薄流体层处回退半程，
   本次未统计回退占比（报告字段 `fallback_links` 已留）。
3. **收敛性未做完**：仅 1500 步（τ_diff = H²/ν = 640~1210），未做步数翻倍与网格再加密两档；
   两档分辨率下 halfway 的 δ_eff 从 +0.31 变到 −0.02，说明统计噪声/未完全收敛不能排除。
4. 3D 管道与椭圆（原 B14-ellipse 6.51 %/4.85 %）本次**未跑**（单机 120 s 工具时限内跑不完，
   已写好 `pipe3d` 用例与本仓 25600 步的椭圆驱动，下一轮上集群）。
5. 力源用 Guo 型源项（一阶），未做 F/2 速度修正；对 u_max 的偏差量级 ~O(F·tau) 未单独量化。

## 6 误差判定（是否 <3 %）
| 关键量 | 数值 | 基准 | 相对差 | <3 %? |
|---|---|---|---|---|
| delta_eff（halfway, L=40） | +0.3104 cell | 0（真实壁） | 0.3104 cell | **否** |
| delta_eff（interp, L=40） | +0.3355 cell | 0（真实壁） | 0.3355 cell | **否** |
| delta_eff（halfway, L=56） | -0.0189 cell | 0（真实壁） | 0.0189 cell | 是 |
| delta_eff（interp, L=56） | +0.1708 cell | 0（真实壁） | 0.1708 cell | **否** |
| theta（halfway, L=40） | 8.843 % | 解析剖面 | — | **否** |
| theta（interp, L=40） | 9.506 % | 解析剖面 | — | **否** |
| theta（halfway, L=56） | 3.182 % | 解析剖面 | — | **否** |
| theta（interp, L=56） | 6.816 % | 解析剖面 | — | **否** |

**结论**：`delta_eff <= 0.05 cell` 与 `theta <= 3 %` 在插值格式下**未同时成立**
（acceptance: theta_le_3pct_interp = False，
delta_eff_le_0p05_interp = False，
halfway_shows_offset = True）。
据此**否决**"两点插值反弹可作斜壁一阶壁面位置修正"这一假设；B14-ellipse 继续 not_verified。
建议锚点（总账外新缺口）：`GAP-LBM-STAIRCASE-WALL-OFFSET`——验收改为
①≥3 案例（45° 斜壁 / 3D 圆管 / 3D 椭圆）逐案例给出 δ_eff 基线数字表；
②**任一格式**在 ≥2 档网格上 theta ≤3 % 且 δ_eff 随 1/h 单调收敛（而非固定 0.05 格）；
③若插值仍不达标，须给出 corner 处 link 连续性的定量诊断。

## 7 证据指针
* 算子：`src/tensorlbm/boundary/interpolated_bounce_back.py`
* 驱动：`benchmarks/pending/staircase_wall_offset/run_inclined.py`（`--case inclined|pipe3d|both --all`）
* 结果：`benchmarks/pending/staircase_wall_offset/artifacts/result.json`
* 图：`benchmarks/pending/staircase_wall_offset/artifacts/staircase_offset_inclined.png`
* 测试：`tests/test_staircase_wall_offset.py`（4 passed / 1 deselected：q=1/2 逐位等价、
  静水不动点、输入契约、q>1/2 权重有界；artifact 门因主判据未达成而跳过）
