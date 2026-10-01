# 运动壁修正项 -6w 系数核查（Bouzidi 标准公式对比）

日期 2026-09-22 · 工作区 `/root/TensorLBM_feat2` @ `18eedca`（工作树 clean，本次未改任何仓库文件）
背景：`docs/shell_bfl_force_analytic_validation_20260922.md` §3 结论 3 提出
`octree_boundary/bfl.py` 的运动壁修正项与"仓库标准 Bouzidi 公式"不一致（线性 `-6wρ(c·u_w)` 无 q、
二次 `-(3/q)wρ(c·u_w)`），并观测到共转旋转壁解析零力矩被破坏（Tz=−390.9 ∝ Ω）。

---

## 1. 结论（一句话）

**`-6`（线性）与 `-(3/q)`（二次）这一对系数是正确的标准 Bouzidi 运动壁系数，不是 bug。**
真正与标准不一致的是 `bfl.py` 的**插值结构**：两个分支都用
`f_opp_post = f[opp]`（从固体侧流入的后流值）作为插值第一项，
而标准 Bouzidi（以及 `docs/octree-boundary-design.md` §3.4 与 `bfl_common`）用的是
**流体侧后碰撞量** `fp_d = f_prev[d](x_f)`、`fp_up = f_prev[d](x_f−c_d)`、`fp_opp = f_prev[opp](x_f)`。
这个"结构 × 系数"错配正是 Tz=−390.9（严格 ∝ Ω、不随细化消失）的来源。

## 2. 文献依据（Bouzidi 运动壁项的确切形式）

Bouzidi–Firdaouss–Lallemand (2001) 的两点式 IBB，按 arXiv:1906.05445（*A comparative study of
immersed boundary method and interpolated bounce-back scheme…* Part I，式 (10)(11a)(11b)）的整理，
动壁项统一写作 `+2ρ₀w_i(e_i·u_w)/cs²`（`e_i` = 未知/被写入 population 的方向；cs²=1/3 ⇒ `6w_i(e_i·u_w)`）：

* 半个格距（Plain BB，式 (10)）：`f_i = f*_ī(x_f) + 2ρ₀w_i(e_i·u_w)/cs²` —— **系数 1**。
* `q<0.5`（式 (11a)）：插值结果**外加**该动壁项 —— **系数 1**（**无 q**）。
* `q≥0.5`（式 (11b)）：动壁项在括号里与 `f*_ī(x_f)` 一起被 `1/(q(2q+1))` 缩放：

  `f_i = [f*_ī(x_f) + 2ρ₀w_i(e_i·u_w)/cs²] / (q(2q+1)) + ((2q−1)/q) f_i(x_ff) − …`

  两点式退化（`1/(2q)` 因子）即：

  `f_i = [f*_ī(x_f) + 2ρ₀w_i(e_i·u_w)/cs²]/(2q) + ((2q−1)/(2q)) f_i(x_ff)`

   ⇒ 二次分支的动壁项系数 = `1/(2q)`。

换算到本仓库的索引约定（`d` = 指向壁面的边界链接方向，未知量写入 `f_out[opp_d]`；
`e_i = c_opp_d = −c_d`，`w_i = w_d`）：

| 分支 | 标准（2 点 Bouzidi）动壁修正 | 本仓库写法 |
|---|---|---|
| `q<0.5` | `+2ρ w(c_opp·u_w)/cs² × 1 = −6 w_d ρ (c_d·u_w)` | 线性支 `-6·moving_base` ✓ |
| `q≥0.5` | `+2ρ w(c_opp·u_w)/cs² × 1/(2q) = −(3/q) w_d ρ (c_d·u_w)` | 二次支 `-(3/safe_q)·moving_base` ✓ |

**注意式 (11b) 括号里与动壁项同括号的第一项是 `f*_ī(x_f)` = 指向壁面的（出射）后碰撞量
= 本仓库的 `fp_d`**（不是 `f_opp_post`）。这就是第 4 节的结构判据。

## 3. 解析零检验：系数对/结构必须配套

共转（slip）严格判据：均匀平衡流 `u` + 以 `u_w = u` 共转的滑移壁 ⇒ 精确解就是该均匀平衡场
⇒ 重构必须给出 `f_bc = feq_opp(u_w)`（否则边界方案连均匀流都保持不住）。
逐链接代数/数值核算（CPU，float64）：

| 结构 | 动壁系数 | q=0.25/0.4/0.5/0.6/0.75/0.9 是否精确 |
|---|---|---|
| 标准（流体侧后碰撞：`fp_d`,`fp_up`,`fp_opp`） | `(-6, −3/q)` | **全 q 精确**（误差 ~1e-17） |
| 标准（同上） | `(-6, −6)`（即 `bfl_common`） | q≤0.5 精确；**q>0.5 有 O(Ma) 误差** |
| `bfl.py` 现状（`f_opp_post`） | `(-6(1−2q), 0)` | 全 q 精确（但该系数对文献/库内均无对应） |

对 `bfl_common`/`bfl_d3q19`（扁平网格、真实库函数）直接实测该判据（uniform `u=0.06 x̂`、
`u_w=u`、方向 3/7/10，动壁切向力应 ≡0）：

```
q=0.25: Fx=+0.000000e+00     (线性支，精确)
q=0.40: Fx=+0.000000e+00     (线性支，精确)
q=0.50: Fx=+0.000000e+00     (退化)
q=0.60: Fx=-3.333333e-03     (二次支，伪切向力)
q=0.75: Fx=-6.666667e-03
q=0.90: Fx=-8.888889e-03
```

即：**仓库被当作"标准"的扁平实现 `bfl_common`/`bfl_d3q19`/`bfl_moving_wall_correction`
（两支统一 `-6`）本身在 q>0.5 的动壁支不满足该判据**，而 `-(3/q)` 才是与 Bouzidi 式
(11b) 一致、且让判据精确成立的形式。（`sphere_amr_common.py` 用的正是 `(-6, -3/q)`，
与文献一致。）

## 4. 代码现状对比

```python
# src/tensorlbm/octree_boundary/bfl.py  (现状)
f_opp_post = f[opp].to(torch.float64)              # 从固体侧流入的后流值
f_bc_lin  = 2.0*qq*f_opp_post + (1.0-2.0*qq)*fp_d          # ← 第一项应为 fp_d
f_bc_quad = f_opp_post/(2*safe_q) + (2*safe_q-1)/(2*safe_q)*fp_opp  # ← 第一项应为 fp_d
f_bc = torch.where(lin, f_bc-6.0*moving_base, f_bc-(3.0/safe_q)*moving_base)

# 标准（bfl_common / docs/octree-boundary-design.md §3.4）
f_bc_lin  = 2.0*q*fp_d + (1.0-2.0*q)*fp_up        # fp_up = f_prev[d](x_f - c_d)（donor gather）
f_bc_quad = fp_d/(2*safe_q) + (2*safe_q-1)/(2*safe_q)*fp_opp
```

`bfl.py` 自身的模块 docstring 与 `HEAD` commit message（"pre-existing Bouzidi source bug
(f_opp_post, …)"）都已指向这一点；`bfl_apply_gather` 里 `fp_up` 已经算好可用。

另：`src/tensorlbm/wall_surface_bfl.py` **完全没有运动壁项**（仅静止壁），其 header 只给静止
Bouzidi 公式（线性支的 `2q / (1−2q)` 是**插值权重**，不是 u_w 修正系数）——因此它不能作为
动壁系数基准；"它与标准不一致"的判断应属对插值权重的误读。

## 5. CPU 验证（`scripts/validate_shell_bfl_force_analytic_cpu.py`，R=12，D3Q27，float64）

对 `bfl.py` 打补丁逐变体重跑 case B（刚性旋转 + 共转壁，解析 F=0、Tz=0），以及 case A
（均匀来流 + 静止壁，冻结场参考）：

| 变体（结构 / 动壁系数） | d1 Tz Ω=.002 / .004 | d2 Tz Ω=.002 / .004 | case A Fx d1/d2 |
|---|---|---|---|
| V1 现状 `f_opp_post` / `(-6,−3/q)` | **−390.9 / −781.8** | **−402.3 / −804.6** | 70.37 / 65.33 |
| V2 `f_opp_post` / `(-6,−6)` | −486.9 / −973.7 | −504.0 / −1008.1 | 70.37 / 65.33 |
| V3 标准结构 / `(-6,−6)`（设计文档字面） | −96.1 / −192.2 | −101.3 / −202.6 | 233.54 / 232.02 |
| **V4 标准结构 / `(-6,−3/q)`（推荐）** | **−0.155 / −0.311** | **+0.432 / +0.865** | 233.54 / 232.02 |
| V5 标准结构 / 无动壁项 | +169.2 / +338.4 | +154.6 / +309.2 | 70.37 / 65.33 |
| V6 `f_opp_post` / `(−6(1−2q), 0)` | −0.008 / −0.016 | +0.379 / +0.757 | 70.37 / 65.33 |

* V1（现状）破坏解析零检验 ~4e2（∝Ω，线性外推）：实锤。
* **V4 把残差压低 3–4 个量级（≈0，余量是探针自身 O(dx) 几何噪声：leaf_center 与
  host-cell 半格偏移、fanout 均值、ghost 采样），同时 F 保持 ~1e-14** ⇒ 结构修复有效。
* V6 说明 `f_opp_post` 结构也能凑出零解，但其配套系数 `(−6(1−2q), 0)` 与文献/库内其他实现
  都不一致 ⇒ 不应选它。
* V3 说明"只按设计文档字面把结构改回、系数也用 `bfl_common` 的 `(-6,−6)`"仍不过关。

## 6. 修复建议

1. **主修复（推荐 = V4）**：把 `bfl.py` 两分支的第一项改回流体侧后碰撞量，保留
   `(-6, -3/q)` 系数：

   ```python
   f_bc_lin = 2.0*qq*fp_d + (1.0-2.0*qq)*fp_up
   f_bc_quad = fp_d/(2*safe_q) + (2*safe_q-1)/(2*safe_q)*fp_opp
   ```
   （`f_opp_post` 仅保留给 `return_force` 之外不再使用可删；`fp_up` 已由 donor gather 提供。）
2. 备选（不改结构）：系数改 `(−6(1−2q), 0)`（= V6，Tz 过关）——**不推荐**：非文献形式，
   与 `bfl_common`/`sphere_amr_common` 均不一致。
3. **采纳前必须复核**：本修复会把 case A（均匀来流 + 静止壁，冻结场）的 Fx 从 70.4 改成 233.5
   （×3.3；理想半程反射参考 274）。该 case 无物理标定（且探针把 `f=f_prev` 传入，
   `f_opp_post` 取到的是解析场而非真实固体格值），所以务必在**真实球体阻力基准**
   （`sphere_bfl_control_volume` / 单卡收敛 Cd）或 Couette/TC 上复核 Cd 后再采纳。
4. **附带（独立发现，建议维护者复核）**：`bfl_common.bfl_bounce_back_common` /
   `bfl_moving_wall_correction` / `bfl_d3q19(.vec)` 的二次支运动壁系数为 `-6`（无 q），
   与 Bouzidi 式 (11b) 的 `1/(2q)` 不一致；第 3 节的共转滑移判据在 q>0.5 给出 O(Ma) 伪力。
   建议改为 `-(3/q)`（或至少在 docstring 标注为 O(Ma) 近似），并补 `q>0.5` 的动壁单元测试
   （现有 `tests/test_bfl_suboff.py::test_wall_model_slip_bfl_preserves_uniform_tangential_flow`
   只在 `q=0.5` 覆盖）。
5. 把 case B（共转旋转壁解析零力矩，Ω 扫描 × d_max 扫描）上升为 octree 壳层 BFL 的常设
   gate：现状 −390.9 明显不合格，修复后 ≈0。

## 7. 复现

```bash
cd /root/TensorLBM_feat2
# 现状（应复现 Tz=-390.9 / -781.8 / -402.3 / -804.6）
python scripts/validate_shell_bfl_force_analytic_cpu.py --radius 12 --dmax 1 2
# 变体扫描（临时改写 bfl.py 后自动还原；脚本在 /tmp，不落仓库）
python /tmp/variant_run.py      # V1..V5
python /tmp/v6.py               # V6
git diff --stat                 # 应为空
```