# 屈服应力/宾汉 LBM 通道驱动的壁面反弹顺序缺陷：定位、修复与残余欠账（2026-11-09, TensorLBM）

`mission 7523 · repo TensorLBM · branch agent/m7523 · 自找活 selfstart`

## 0 结论（一句话）

`src/tensorlbm/yield_stress_channel.py` 三个通道驱动（`run_bingham_couette` /
`run_bingham_poiseuille` / `run_hb_poiseuille`）的壁面反弹**顺序反了**——先把分布
`stream` 出去、再在壁面格点上做反弹，使壁面格点里混入流入分布与周期回卷分布，
反弹后仍残留一个非零滑移速度。全库自己的一致性判据 `tests/test_yield_stress_lbm.py::TestNewtonianLimit`
在 main 上**长期红**（Newtonian 极限 L2 误差 **26.27%**，判据 5%）。改成与解析解阶梯
（`tests/test_lbm_verification.py`）同一约定「回填碰撞前分布（NoDynamics）→ 对碰撞后
分布反弹 → 再 stream」后：Newtonian 极限 **26.27% → 2.27%**（判据 5%，通过），
Couette 动壁 **4.52% → 0.00%**（机器精度）。**但**屈服应力（τ_y>0）客体的定量精度
仍是**未清偿欠账**：同一驱动在 H=8、m=100 下对解析 Bingham 剖面的 L2 误差
**24–35%**，与壁面方案无关（见 §5），并有 3 条 main 上同样红的旧用例作为独立佐证。

## 1 案例介绍（对象与背景）

对象是 TensorLBM 的屈服应力（宾汉 / Herschel-Bulkley）通道验证驱动，它把
Papanastasiou 正则化的 Bingham/HB 碰撞核包成「几何 + 体力/动壁」的稳态求解器，
供 `tests/test_yield_stress_lbm.py` 与 `docs/herschel_bulkley_yield_stress.md` 的
验证面使用。物理背景：压力驱动（体力 ax）或剪切驱动（动壁 u_wall）的平行板通道，
流体为带屈服应力 τ_y 的非牛顿材料，中心存在 plug 区。

| 参数 | 取值（本单全部实测口径） |
|---|---|
| 通道流体层数 H | 8（壁在第 0、H+1 行，半格壁位 y=−0.5 / 7.5） |
| 塑料粘度 mu_0 | 0.02（τ=0.56）/ 0.0833（τ=0.75）/ 1/6（τ=1.0） |
| Papanastasiou 正则化参数 m | 100 |
| 体力 ax | mu_0/2000（保持 u_max≈4e-3，Ma 可忽略） |
| 屈服应力扫描 | τ_y/τ_w ∈ {0, 0.25, 0.5, 0.75}，τ_w = ax·H/2 |
| 精度 | float64，CPU，单线程，max_iter=15000，tol=1e-9 |
| 收敛判据（驱动内置） | 每 100 步 \|Σu_x(t)−Σu_x(t−100)\| < tol |

## 2 怎么做的（方法/流程/工具链）

1. **从红光出发**：`pytest tests/test_yield_stress_lbm.py` 在 main 上 4 红，第一条就是
   `TestNewtonianLimit::test_tau_y_zero_vs_newtonian_poiseuille`（断言 L2<5%，实测 26.27%）。
2. **先排除「是判据错还是代码错」**：把仿真剖面与解析抛物线逐点比对，发现
   二阶差分**逐点恒定**（−0.5e-3 = −G/ν），即**曲率（粘度）正确**，但整条曲线被
   一个常数抬高 ≈0.75e-3（相当于壁面半格平面上残留滑移 0.71e-3 ≈ 18% u_max）——
   这只能来自边界条件，不来自碰撞核。
3. **τ 扫描排除「壁位错」假设**：τ=0.56/0.75/1.0 三档旧方案误差为
   26.42% / 27.08% / 29.20%，**不随 τ 收敛**；经典半路反弹的壁位误差随 (τ−1/2) 变，
   所以不是壁位效应，而是**顺序**效应。
4. **对照全库已验证约定**：`tests/test_lbm_verification.py` 的 A–E 解析解阶梯用的是
   「NoDynamics（固体格点回填碰撞前快照）→ `bounce_back_cells`（对碰撞后值反射）→ `stream`」，
   且该文件已记录同一陷阱（用碰撞后快照回填会让 Poiseuille 误差涨到 17%）。
5. **最小改动修复**：新增私有助手 `_solid_restore_and_bounce(f, f_pre, wall, u_wall, moving_mask)`
   = `where(wall, f_pre, f)` + 既有 `_moving_wall_bounce_back`（含 Ladd 动壁修正，保留）；
   三个驱动的循环改成 `f_pre = f → collide → 回填+反弹 → stream`。
   旧顺序不删除，收进 `bounce_scheme="post_stream"` 分支（默认 `"halfway"`），
   专供负控回归测试「证明该缺陷确有鉴别力」，未知取值抛 `ValueError`。
6. **量化**：`artifacts/yield_stress_bounce_back_20261109/run_scan.py` 跑 13 个算例
   生成 `metrics.json`（本文全部数字由此注入）+ 两张图。

## 3 计算结果

### 3.1 Newtonian 极限（τ_y=0，判定量 L2 相对误差）

| τ | 旧顺序 post_stream | 修复后 halfway | u_max 仿真 | u_max 解析 |
|---|---|---|---|---|
| 0.56 | 26.42% | 2.27% | 0.00387 | 0.00394 |
| 0.75 | 27.08% | 2.14% | 0.00387 | 0.00394 |
| 1.0 | 29.20% | 0.72% | 0.00392 | 0.00394 |

- 判据 5%（沿用原用例）：修复后 3/3 通过；旧顺序 3/3 违反，且**误差与 τ 无关**，
  即该缺陷不是壁位分辨率问题；
- 修复后残差随 τ→1 下降（2.27% → 2.14% → 0.72%），符合「半路反弹在 τ=1 附近最准」的已知行为。

### 3.2 Couette 动壁（u_wall=0.01，线性解析解）

| 方案 | L2 误差 | 迭代 |
|---|---|---|
| post_stream | 4.52% | 7200 |
| halfway | 0.00% | 5900 |

动壁路径（Ladd 修正 `6 w_i ρ c_i·u_wall`）在修复后**精确**复现线性剖面，
说明修复没有破坏动壁修正。

### 3.3 Bingham 屈服应力客体（L2 vs 解析 Bingham 剖面）

| τ_y/τ_w | 旧顺序 | 修复后 | 修复后 u_max 误差 | plug 半宽（解析 / 探测） |
|---|---|---|---|---|
| 0.0 | 26.42% | 2.27% | 1.68% | 0.00 / 0.50 |
| 0.25 | 37.91% | 29.12% | 29.57% | 1.00 / 0.50 |
| 0.5 | 31.25% | 34.78% | 29.24% | 2.00 / 0.50 |
| 0.75 | 30.06% | 23.88% | 4.35% | 3.00 / 0.50 |

## 4 结果图

- `artifacts/yield_stress_bounce_back_20261109/fig1_scheme_error.png`：左=Newtonian 极限
  L2 误差随 τ（两方案 + 5% 判据线）；右=Bingham 误差随 τ_y/τ_w。
- `artifacts/yield_stress_bounce_back_20261109/fig2_profiles.png`：左=τ_y/τ_w=0.75 的
  Bingham 剖面（解析 vs 两方案）；右=Couette 线性剖面。

## 5 注意点 / 局限（诚实申报）

1. **本修复只清偿了「壁面顺序」这一项欠账，没有清偿屈服应力定量精度**：
   τ_y>0 时两方案的 L2 误差都在 24–35%，修复后并不比旧方案一致地更好
   （τ_y/τ_w=0.25/0.5 时 halfway 反而略差）。该量级与 H=8 + m=100 的
   Papanastasiou 正则化 + 粗网格 plug 区离散相容，但**本单未做网格收敛证据**，
   因此只能判定为「未定位的残余欠账」，不能称「正则化误差」。
2. **plug 半宽探测器不可用**：`y_p_sim` 用 `u > 0.99·u_max` 判 plug，在 H=8 的
   解析 plug 半宽 1.0–3.0 格下恒返回 0.5——判据本身已失效，故 §3.3 的 plug 列只作
   现状留证，不作精度证据。
3. **main 上另有 3 条同类红用例**（本单前后实测均红，非本单引入）：
   `test_m_scan_monotonic_error`（m=100→500 误差 0.3127→0.3914 非单调）、
   `test_viscous_dissipation_equals_work`（能量残差 0.306）、
   `test_lbm_vs_algebraic_bn_01_to_1`（Bn=0.1：f_lbm=1.11e-3 vs f_alg=9.96e-5，差 1012%，
   疑为代数基准自身量纲/定义错）。三者指向同一片残余欠账（屈服应力客体的
   定量/守恒面），建议后续单独立单。
4. 判据口径：L2 = ‖u_sim−u_an‖₂/‖u_an‖₂（全通道逐格点）；τ_y=0 且 u_an 极小时无归一化问题。
5. 全部测量在 CPU 单线程 float64 下完成；本机 load15 ≈ 40（nproc=128），无 GPU 参与。

## 6 误差判定（逐项回答「是否 <3%」）

| 关键量 | 基准 | 误差（修复后） | <3% ? |
|---|---|---|---|
| Newtonian Poiseuille 剖面 L2（τ=0.56） | 解析抛物线 | 2.27% | 是 |
| Newtonian Poiseuille 剖面 L2（τ=0.75） | 解析抛物线 | 2.14% | 是 |
| Newtonian Poiseuille 剖面 L2（τ=1.0） | 解析抛物线 | 0.72% | 是 |
| Couette 线性剖面 L2 | 解析线性 | 0.00% | 是 |
| Newtonian u_max（τ=0.56，修复后） | 解析 | 1.68% | 是 |
| Bingham 剖面 L2（τ_y/τ_w=0.25） | 解析 Bingham | 29.12% | **否** |
| Bingham 剖面 L2（τ_y/τ_w=0.5） | 解析 Bingham | 34.78% | **否** |
| Bingham 剖面 L2（τ_y/τ_w=0.75） | 解析 Bingham | 23.88% | **否** |
| Bingham u_max（τ_y/τ_w=0.75） | 解析 | 4.35% | **否** |

结论：牛顿极限与动壁判据 <3%；**屈服应力客体的定量判据不达标，如实列为未清偿欠账**。

## 7 证据指针

- 代码：`src/tensorlbm/yield_stress_channel.py`（`_solid_restore_and_bounce`、
  `_check_bounce_scheme`、三驱动 `bounce_scheme` 参数）
- 测试：`tests/test_yield_stress_lbm.py::TestBounceBackOrder`（5 条，含 1 条负控
  `test_post_stream_scheme_violates_no_slip`）+ 既有 `TestNewtonianLimit`（本单后转绿）
- 工件：`artifacts/yield_stress_bounce_back_20261109/{run_scan.py, metrics.json, fig1_scheme_error.png, fig2_profiles.png}`
- 复核命令：
  `PYTHONPATH=src python3 -m pytest tests/test_yield_stress_lbm.py::TestBounceBackOrder tests/test_yield_stress_lbm.py::TestNewtonianLimit -q`（7 passed）
  `PYTHONPATH=src python3 artifacts/yield_stress_bounce_back_20261109/run_scan.py`（重算全部数字与图）
