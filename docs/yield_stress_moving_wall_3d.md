# D3Q19 屈服应力 LBM + 移动壁（P82W5-N04）

本文件记录 `src/tensorlbm/yield_stress_moving_wall_3d.py` 的物理约定、实现选择与
适用范围；解析参考解与高精度参考解的推导见
`src/tensorlbm/yield_stress_analytical_moving_wall.py`。

## 1. 内核

* 分布函数形状 `(19, nz, ny, nx)`，`stream3d` 为周期性（x/y/z 均周期），壁面以
  **固体层** 表示（`y=0`、`y=H+1`；方管另加 `z=0`、`z=N+1`）。
* 碰撞：`yield_stress_lbm_3d.collide_bingham_bgk_forced_3d` /
  `collide_hb_bgk_forced_3d`（Papanastasiou 与 Herschel–Bulkley 正则化，复用
  `yield_stress_rheology_3d` 的应力/剪切率恢复），体积力用 Guo(2002) 格式。
  `tau_y = 0` 退化为常量黏度 BGK + Guo 力（`_collide_newtonian_guo`）。
* 剪切率恢复两种模式（`use_neq`）：
  * `use_neq=True`：由非平衡应力 `Π^neq` 反解（`γ̇ = 3|Π|/(√2 ρ τ)`）；
  * `use_neq=False`：由速度场中心差分得到应变率张量。
  两者在**塞区**行为差别很大：应力反解在塞区被 `|Π| ~ τ_y` 钉住，等效黏度饱和在
  `μ_0 + τ_y/γ̇_min`，塞区不可能真正刚性；速度梯度反解可让塞区黏度升到
  `μ_0 + m τ_y`，塞区接近刚性。实验表明 `m ≥ 1e5` 之后 Papanastasiou 式子
  `μ_eff = μ_0 + τ_y(1-e^{-mγ̇})/γ̇` 在 `mγ̇ ≫ 1` 区已与 `m` 无关
  （退化为 `μ_0 + τ_y/γ̇`），因此**继续增大 m 不再改善塞区**，这是正则化模型的
  结构性限制，不是实现缺陷。

## 2. 移动壁（半路反弹 + Ladd）

选择 **half-way bounce-back + Ladd 速度修正**，而不是 Bouzidi 插值：本节点的壁面
全部是落在格点面上的平面壁，此时半路反弹本身是二阶精度，Bouzidi 的插值因子退化
为 `q=1/2`，二者等价，而半路反弹不需要亚格点插值状态。

实现写作「stream 之后覆写」的等价形式（与 `docs/yield_stress_bounce_back_convention_20261109.md`
记录的已验证顺序一致）：

```
f_i(x_b, t+1) = f_ī^post(x_b, t) + 6 w_i ρ(x_b) (c_i · u_w)
              对每条 x_b - c_i 是固体的连接
```

`x_b` 为壁面相邻流体格点，`f_ī^post` 为碰撞后反向分布，`ρ` 取该格点局部密度（缺
`ρ` 在 `ρ≈1` 时看不见，`ρ=1.2` 时是 33% 的修正量，见 2D 侧的审计记录）。
固体层每个时间步重置为静止平衡态（`ρ=1, u=0`），防止陈值扩散。

## 3. 动量账本（MEM）

对每条被反弹的连接，流体得到的动量精确为

```
M_link = c_i (f_i^new - f_i^streamed)
```

把它拆成「无滑移反射」与「Ladd 注入」两部分，并对所有连接求和：

```
M_drag   = 2 Σ_links c_i f_ī^post           (无滑移部分, 即壁面阻力)
M_inject = Σ_links c_i · 6 w_i ρ (c_i·u_w)  (移动壁注入)
M_wall   = M_drag + M_inject
```

（系数 2 来自「流体向固体的流出」与「BB 回填」两项，推导见模块 docstring。）
由此可得两条可机器判定的账：

1. **离散恒等式** `P(t+1) - P(t) = M_body + M_wall`
   （`P` 为流体格点动量总和，`M_body = ρ a V`）——闭合率应等于 1 到舍入误差；
2. **稳态力平衡** `M_wall = -M_body`（有体积力）或 `|M_wall,静止壁| = |M_wall,移动壁|`
   （纯 Couette，无体积力）。

稳态判据必须在**最后一个检查窗口**上取，而不是整段平均：起始段 `M_drag` 从 0 爬
升到稳态值，整段平均会把闭合率压到 ~0.98（本节点复现了该陷阱）。

## 4. 纯 Couette 的退化：为什么没有「部分塞区」

零压力梯度下 `dτ/dy = 0` ⇒ 剪应力在间隙内**均匀**。于是

* `|τ_w| = μ_0|U_w|/H ≥ τ_y`：全间隙屈服，速度剖面线性，**塞区厚度为 0**；
* `|τ_w| < τ_y`：材料不能变形，但两壁无滑移要求 `u(0)=0, u(H)=U_w≠0`，刚性体
  无法同时满足 ⇒ **不存在稳态解**。

因此 `delta_p = H - μ_0 U_w/τ_y`（历史实现 `yield_stress_analytical_3d.py`）不是
Couette 问题的解；`bingham_plane_couette_regime()` 显式给出该分类，带部分塞区的
闭式锚改用 **Couette–Poiseuille**（一壁移动 + 体积力）：塞厚 `δ_p = 2τ_y/G`，
塞区位置由底壁剪应力 `τ_0` 唯一确定（单调标量方程，二分到机器精度）。

## 5. 高精度参考解

* 牛顿方管：Fourier 级数（`newtonian_square_duct_series`，N=32 时形状因子
  0.4222 vs 解析 0.4217）。
* Bingham 方管/矩形管：**无正则化**的变分问题用增广拉格朗日（Glowinski）在细网格
  上求解（`bingham_square_duct_reference` / `bingham_rectangular_duct_reference`），
  `u` 更新为常系数 Poisson（正弦变换精确求解），塑性乘子用标准投影步。
  `refine` 为偶数时细网格节点与 LBM 格心**完全重合**（`j = refine*(i-0.5)`），
  无需插值。参考解自身的离散误差需用 refine 阶梯单独审计。

## 6. 适用范围与已知限制

* 壁面必须是落在格点面上的平面壁；曲面/斜面壁需改为 Bouzidi 或 IBM。
* 正则化 Bingham 的塞区不是严格刚性，塞内残余剪切 `γ̇ ~ (τ_y/μ_0)e^{-M}`，
  `M = mτ_y/μ_0`；要求塞区速度变异系数 ≤1e-3 需要 `M ≳ 500`（黏度比 ~500），
  此时塞区黏度 `ν_plug = μ_0(1+M)` 很大，需要相应放宽 `nu_max`/`tau_max`。
* `mγ̇ ≫ 1` 区等效黏度与 `m` 无关，故「加大概率化参数」不能治愈塞区涂抹。
* 后端为 PyTorch CPU/GPU 通用；`z` 方向在通道算例中周期、在方管中为壁面。

## 7. 用法

```python
from tensorlbm.yield_stress_moving_wall_3d import run_channel_3d, run_duct_3d

# Bingham Couette-Poiseuille：上壁以 U_w 移动 + 体积力 ax
res = run_channel_3d(H=32, ax=2e-6, u_wall=(1.28e-3, 0.0, 0.0), moving_top=True,
                     tau_y=4e-6, mu_0=0.1, m=1e6, use_neq=False,
                     nu_max=0.1 * (1 + 1e6 * 4e-6 / 0.1) * 10)
res["ux_profile"], res["ledger"]["force_balance"]

# Bingham 方管（周期 x）
res = run_duct_3d(N=32, ax=1e-5, tau_y=2e-5, mu_0=0.1, m=1e5, use_neq=False)
res["ux_cross_iyz"], res["Q"]
```

`res["ledger"]` 给出 `closure_run` / `closure_window` / `force_balance` /
`couette_balance`；`res["ledger"]["run"]` 与 `["window"]` 保留原始累计量。