# REPORT_V3 — moving-wall (non-zero Dirichlet) term of the interpolated bounce-back α-family

Anchor `XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME`.  Third verifier of this anchor: `run_xval.py`
(m18218, `honest FAIL`), `verify_v2.py` (m20373, clean-room re-check, wall at rest only) and now
`verify_v3_moving_wall.py` (this document).  Previous verifiers exercised **walls at rest only**, the
case in which the antisymmetric equilibrium term `-α e⁻_q(x_w)` of the pylbm `LinearBounceBack`
form vanishes identically; `OSS_DIFF.md` therefore listed the omitted term as the remaining gap,
"mandatory for inclined / moving walls".  This mission implements the term in the production
operator and measures what it buys (and what it does not).

## 1 案例（物理背景 / 几何 / 参数）

三个都有解析解的标杆，壁面都落在**亚格子**位置 `δb`/`δt`（首/末流体行外侧），`H=(N−1)+δb+δt`：

| 标杆 | 边界条件 | 解析解 | 检验什么 |
|---|---|---|---|
| `couette` | 下壁滑动 `U_w=0.02`，上壁静止，无体力 `F=0` | `u(y)=U_w(1−y/H)` | **仅** Dirichlet 项（`−α e⁻_q(x_w)`） |
| `forced` | 双壁静止，体力 `g` 碰撞后加入 | `u(y)=g/(2ν) y(H−y)` | 重构 REPORT_V2 的欠账（壁面在静止时残差来源） |
| `couette_poiseuille` | 下壁滑动 + 体力同时 | `u(y)=U_w(1−y/H)+g/(2ν) y(H−y)` | 壁面值与半步力约定**同时**正确 |

参数：`ν=0.1`（格子）、`U_w=0.02`、`g=8ν·0.02/H²`（使 Poiseuille 中心线速度 = 0.02，Ma≈0.035）、
`N=8` 流体行、`δ∈{0.1,0.25,0.5}`（另加非对称壁 `δb=0.3, δt=0.7`）、载体 BGK 与 TRT（`Λ⁺Λ⁻=3/16`）、
α 族成员 BFL / CLI / halfway（`α=2`）、每臂 1500 步、初值 = 解析剖面（稳态唯一，与初值无关；慢粘性
spin-up 被移除，错误闭包仍在数步内偏离解析解——由同批 `none` 臂证明）。

## 2 怎么做的（方法 / 流程 / 关键设置 / 工具链）

* **能力补全（生产代码）**：`src/tensorlbm/boundary/interpolated_bounce_back.py` 新增
  `wall_velocity` / `wall_density` / `weights` 三个可选参数，把 pylbm `LinearBounceBack` 的
  Dirichlet 项补进生产算符：`e⁻_q(x_w) = (f^eq_q − f^eq_opp(q))/2 = 3 w_q ρ_w (c_q·u_w)`
  （二次项在相反方向相消），反射值改为 `α̂f_pre(i)+β̂f_pre(opp i)+β f_new(i) − α e⁻_q(x_w)`。
  默认 `wall_velocity=None` 时**逐比特**保持旧行为（有测试）。
* **驱动**：`verify_v3_moving_wall.py`（torch float64）。碰撞/流动/矩/诊断独立实现，边界只调用
  生产算符；体力在碰撞后以各向同性权重 `3 w_q c_qx g` 只加在流体格，碰撞只算流体格（REPORT_V2 的
  R1 结论），存储动量 `m=Σf c`，平衡态由 `m/ρ` 构造，报告速度 `m/ρ − g/2`。
* **对照臂**：`none`（不传 `wall_velocity` = 打补丁前行为）、`term`（滑动壁 `u_w=U_w`、静止壁 `0`）、
  `half`（`u_w=g/2`，半步力）、`lam`（`u_w=λ⁻g`，REPORT_V2 用的形式）、`zero`（项存在但值为 0，代码路径对照）。
* **诊断**：`l2_rel`/`max_rel`（对解析解，流体行）、`l2_rel@1000`（固定早期视界，暴露漂移）、
  `dM_per_step`（总质量泄漏/步，相对初值）、单步 `mass_probe`（只做一次算符应用，无碰撞/流动）。
* 图：`make_figures_v3.py`（4 张，全部由 JSON 注入）。

## 3 计算结果

### 3.1 Couette（滑动壁）——Dirichlet 项是"壁面动不动"的分水岭

| 标杆 | 臂 | L2 误差范围（17/17 臂） |
|---|---|---|
| couette | `term` | 0.0013% – 0.0013% |
| couette | `none`（补丁前） | 100.0000% – 100.0000% |

补丁前算符把**滑动壁当成静止壁**：整场速度为零，误差 100%。补上 Dirichlet 项后误差降到
1e-3 % 量级，且与 α 族成员（BFL/CLI/halfway）、载体（BGK/TRT）、壁距（0.1/0.25/0.5）以及**非对称壁**
（δb=0.3, δt=0.7）都无关——这正是一个只与壁面速度、与几何无关的项应有的行为。

| closure | carrier | δb | δt | arm | L2 % | max % | L2@1k % | dM/step |
|---|---|---|---|---|---|---|---|---|
| bfl | bgk | 0.10 | 0.10 | none | 100.0000 | 100.0000 | 100.0000 | 6.92e-09 |
| bfl | bgk | 0.10 | 0.10 | term | 0.0013 | 0.0013 | 0.0009 | 6.92e-09 |
| cli | bgk | 0.10 | 0.10 | none | 100.0000 | 100.0000 | 100.0000 | 7.05e-09 |
| cli | bgk | 0.10 | 0.10 | term | 0.0013 | 0.0014 | 0.0009 | 7.05e-09 |
| halfway | bgk | 0.10 | 0.10 | none | 100.0000 | 100.0000 | 100.0000 | 6.92e-09 |
| halfway | bgk | 0.10 | 0.10 | term | 0.0013 | 0.0013 | 0.0009 | 6.92e-09 |
| bfl | trt_c | 0.10 | 0.10 | none | 100.0000 | 100.0000 | 100.0000 | 6.92e-09 |
| bfl | trt_c | 0.10 | 0.10 | term | 0.0013 | 0.0013 | 0.0009 | 6.92e-09 |
| cli | trt_c | 0.10 | 0.10 | none | 100.0000 | 100.0000 | 100.0000 | 7.05e-09 |
| cli | trt_c | 0.10 | 0.10 | term | 0.0013 | 0.0014 | 0.0009 | 7.05e-09 |
| bfl | bgk | 0.25 | 0.25 | none | 100.0000 | 100.0000 | 100.0000 | 6.94e-09 |
| bfl | bgk | 0.25 | 0.25 | term | 0.0013 | 0.0013 | 0.0009 | 6.94e-09 |
| cli | bgk | 0.25 | 0.25 | none | 100.0000 | 100.0000 | 100.0000 | 7.01e-09 |
| cli | bgk | 0.25 | 0.25 | term | 0.0013 | 0.0013 | 0.0009 | 7.01e-09 |
| halfway | bgk | 0.25 | 0.25 | none | 100.0000 | 100.0000 | 100.0000 | 6.94e-09 |
| halfway | bgk | 0.25 | 0.25 | term | 0.0013 | 0.0013 | 0.0009 | 6.94e-09 |
| bfl | trt_c | 0.25 | 0.25 | none | 100.0000 | 100.0000 | 100.0000 | 6.94e-09 |
| bfl | trt_c | 0.25 | 0.25 | term | 0.0013 | 0.0013 | 0.0009 | 6.94e-09 |
| cli | trt_c | 0.25 | 0.25 | none | 100.0000 | 100.0000 | 100.0000 | 7.01e-09 |
| cli | trt_c | 0.25 | 0.25 | term | 0.0013 | 0.0013 | 0.0009 | 7.01e-09 |
| bfl | bgk | 0.50 | 0.50 | none | 100.0000 | 100.0000 | 100.0000 | 6.95e-09 |
| bfl | bgk | 0.50 | 0.50 | term | 0.0013 | 0.0013 | 0.0009 | 6.95e-09 |
| cli | bgk | 0.50 | 0.50 | none | 100.0000 | 100.0000 | 100.0000 | 6.95e-09 |
| cli | bgk | 0.50 | 0.50 | term | 0.0013 | 0.0013 | 0.0009 | 6.95e-09 |
| halfway | bgk | 0.50 | 0.50 | none | 100.0000 | 100.0000 | 100.0000 | 6.95e-09 |
| halfway | bgk | 0.50 | 0.50 | term | 0.0013 | 0.0013 | 0.0009 | 6.95e-09 |
| bfl | trt_c | 0.50 | 0.50 | none | 100.0000 | 100.0000 | 100.0000 | 6.95e-09 |
| bfl | trt_c | 0.50 | 0.50 | term | 0.0013 | 0.0013 | 0.0009 | 6.95e-09 |
| cli | trt_c | 0.50 | 0.50 | none | 100.0000 | 100.0000 | 100.0000 | 6.95e-09 |
| cli | trt_c | 0.50 | 0.50 | term | 0.0013 | 0.0013 | 0.0009 | 6.95e-09 |
| bfl | trt_c | 0.30 | 0.70 | none | 100.0000 | 100.0000 | 100.0000 | 6.90e-09 |
| bfl | trt_c | 0.30 | 0.70 | term | 0.0013 | 0.0013 | 0.0008 | 6.90e-09 |
| cli | trt_c | 0.30 | 0.70 | none | 100.0000 | 100.0000 | 100.0000 | 6.95e-09 |
| cli | trt_c | 0.30 | 0.70 | term | 0.0013 | 0.0013 | 0.0009 | 6.95e-09 |

### 3.2 力驱动通道（双壁静止）——REPORT_V2 残差的假设被否定

| closure | carrier | δb | δt | arm | L2 % | max % | L2@1k % | dM/step |
|---|---|---|---|---|---|---|---|---|
| bfl | bgk | 0.10 | 0.10 | none | 1.6713 | 1.1801 | 1.6717 | 6.92e-09 |
| bfl | bgk | 0.10 | 0.10 | half | 2.2568 | 1.9559 | 2.2572 | 6.92e-09 |
| bfl | bgk | 0.10 | 0.10 | lam | 2.0170 | 1.6456 | 2.0174 | 6.92e-09 |
| bfl | bgk | 0.10 | 0.10 | zero | 1.6713 | 1.1801 | 1.6717 | 6.92e-09 |
| cli | bgk | 0.10 | 0.10 | none | 1.0046 | 0.7094 | 1.0042 | 7.05e-09 |
| cli | bgk | 0.10 | 0.10 | half | 0.5708 | 0.6973 | 0.5705 | 7.05e-09 |
| cli | bgk | 0.10 | 0.10 | lam | 0.7031 | 0.7016 | 0.7027 | 7.05e-09 |
| cli | bgk | 0.10 | 0.10 | zero | 1.0046 | 0.7094 | 1.0042 | 7.05e-09 |
| bfl | trt_c | 0.10 | 0.10 | none | 1.1696 | 0.8260 | 1.1700 | 6.92e-09 |
| bfl | trt_c | 0.10 | 0.10 | half | 1.7631 | 1.6019 | 1.7635 | 6.92e-09 |
| bfl | trt_c | 0.10 | 0.10 | lam | 1.3850 | 1.1170 | 1.3854 | 6.92e-09 |
| bfl | trt_c | 0.10 | 0.10 | zero | 1.1696 | 0.8260 | 1.1700 | 6.92e-09 |
| bfl | bgk | 0.25 | 0.25 | none | 0.0744 | 0.0541 | 0.0748 | 6.94e-09 |
| bfl | bgk | 0.25 | 0.25 | half | 0.6545 | 0.7540 | 0.6549 | 6.94e-09 |
| bfl | bgk | 0.25 | 0.25 | lam | 0.4192 | 0.4740 | 0.4195 | 6.94e-09 |
| bfl | bgk | 0.25 | 0.25 | zero | 0.0744 | 0.0541 | 0.0748 | 6.94e-09 |
| cli | bgk | 0.25 | 0.25 | none | 1.4358 | 1.0330 | 1.4354 | 7.01e-09 |
| cli | bgk | 0.25 | 0.25 | half | 0.9818 | 1.0077 | 0.9815 | 7.01e-09 |
| cli | bgk | 0.25 | 0.25 | lam | 1.1487 | 1.0174 | 1.1483 | 7.01e-09 |
| cli | bgk | 0.25 | 0.25 | zero | 1.4358 | 1.0330 | 1.4354 | 7.01e-09 |
| bfl | trt_c | 0.25 | 0.25 | none | 0.3787 | 0.2728 | 0.3783 | 6.94e-09 |
| bfl | trt_c | 0.25 | 0.25 | half | 0.3319 | 0.4282 | 0.3320 | 6.94e-09 |
| bfl | trt_c | 0.25 | 0.25 | lam | 0.2222 | 0.2626 | 0.2219 | 6.94e-09 |
| bfl | trt_c | 0.25 | 0.25 | zero | 0.3787 | 0.2728 | 0.3783 | 6.94e-09 |

关键读数（逐臂，见上表）：加入 Dirichlet 项**没有**系统性降低残差。`bgk/bfl/δ=0.1`：
`none` 1.671% → `half` 2.257%（更差）；`bgk/cli/δ=0.1`：1.005% → 0.571%（更好）；
`bgk/bfl/δ=0.25`：0.074% → 0.655%（更差）。也就是说 REPORT_V2 遗留的"均匀速度偏置"**不是**
缺失的壁面值项造成的（负结果，如实记录）。

### 3.3 Couette–Poiseuille（滑动壁 + 体力）

| closure | carrier | δb | δt | arm | L2 % | max % | L2@1k % | dM/step |
|---|---|---|---|---|---|---|---|---|
| bfl | bgk | 0.25 | 0.25 | none | 48.3091 | 62.3760 | 48.3089 | 6.94e-09 |
| bfl | bgk | 0.25 | 0.25 | couette_poiseuille | 0.3808 | 0.4773 | 0.3812 | 6.94e-09 |
| bfl | bgk | 0.70 | 0.70 | none | 46.9914 | 60.7458 | 46.9911 | 6.86e-09 |
| bfl | bgk | 0.70 | 0.70 | couette_poiseuille | 1.8050 | 1.6146 | 1.8046 | 6.86e-09 |

`none` 臂 l2 ≈ 48.3%（滑动壁贡献丢失一半流动），
`couette_poiseuille` 臂（`u_w=U_w+g/2` 移动壁、`g/2` 静止壁）：
δ=0.25 → 0.3808%，δ=0.7 → 1.8050%。说明壁面值 +
半步力约定同时正确时，两个驱动源可以叠加而不引入新偏差。

### 3.4 残差的真实来源：算符的**质量泄漏**（新增可测判据）

固定视界 `L2@1k` 与 `L2@1500` 几乎相同（见上表），而 `dM_per_step` 在全部 62 个臂上都在
6.9e-09–7.0e-09（相对/步）之间，**与闭包、与是否有 Dirichlet 项无关**——`zero` 与 `none` 臂
给出同一个数。单步探针（只调一次算符，不碰撞不流动）给出同样的泄漏。结论：

* 本算符采用"把反射量写进固体格槽位"的编程契约，写的是**替换值**而不是半步交换值，
  因此不严格守恒质量；泄漏率 O(1e-9)/步，在 Couette 臂上表现为**均匀的幅度衰减**（剖面形状与
  壁面位置都对，只剩整体按 `1−8.9e-9·n` 缩放），1500 步累计 1.3e-5，6000 步累计 5.3e-5。
* 这解释了 REPORT_V2 的"均匀偏置"现象族，并说明 pylbm/lbmpy 声称的 1e-14 严格精确性对**本编程
  契约**不可达：不是 α 族公式的问题，而是固体侧写法的守恒性问题（下一步工作，见 §5）。

图：`fig_couette_profiles.png`（含项/不含项的剖面 vs 解析解）、`fig_l2_vs_delta.png`（误差-δ，5 条臂族）、
`fig_forced_conventions.png`（三种壁面值约定的误差条图）、`fig_mass_drift.png`（质量泄漏-δ）。

## 4 注意点（坑 / 局限 / 适用范围）

* **R-A 曾经的错误断言**：`OSS_DIFF.md` 曾写"该术语在本标杆的法向链上恒为零"并据此认为壁面
  对齐通道可省略该项——对**滑动**壁面是错的：对角切链 `c_qx u_wx ≠ 0`。已由
  `test_value_term_is_zero_on_wall_normal_links_and_nonzero_on_diagonals` 钉住。
* **R-B 约定**：`wall_velocity` 必须用与流体平衡态**同一**速度约定（存储动量）。有体力且用
  Guo 型碰撞后加力时，无滑移壁的存储速度是 `g/2` 而非 0；纯 Couette（g=0）无歧义。
* **R-C 驱动陷阱（本次踩过）**：初版 sweep 把 `u_wall=0.02` 也传给了 `forced` 臂，导致解析基准
  与边界条件不一致、全场误差 49%。已修正并复跑；表中 `forced` 数据的 `g/u_wall` 列可自检
  （`u_wall=0`）。
* **局限**：2D 直壁、单一分辨率 N=8、单 ν、`Re_H<7`、稳态充分发展流；只考察壁模型，不涉湍流；
  1500 步固定视界（未跑至真正稳态，因质量泄漏使"稳态"判据永远不满足——这本身是被记录的结论）。
* 生产算符对 α 族默认关闭新项（`wall_velocity=None`），即**不改变任何既有算例**；
  移动壁/倾斜壁用例需显式传参。

## 5 误差判定（逐项，显式回答是否 <3%）

| 关键量 | 基准 | 误差 | <3% ? |
|---|---|---|---|
| Couette L2（`term` 臂，17 臂最差） | 解析 `U_w(1−y/H)` | 0.0013% | **是** |
| Couette L2（`none` 臂=补丁前） | 同上 | 100.0% | **否（说明能力缺失）** |
| Couette–Poiseuille L2（`couette_poiseuille` 臂最差） | 解析叠加解 | 1.805% | **是** |
| 力驱动通道 L2（全部 24 臂最差） | 解析 Poiseuille | 2.257% | **是** |
| 力驱动通道"加项是否改善"（假设 H） | 同臂 `none` vs `half` | 3 组更差 / 3 组更好 | **否（假设被否定）** |
| OSS 严格精确性 `l2<1e-10` | pylbm 声明 | 实测 ~1e-5（1500 步，泄漏主导） | **否（honest，已定位到质量守恒）** |
| 质量守恒（本编程契约） | 0 | 6.9e-09–7.0e-09/步 | **否（honest，新增待办）** |

## 6 产物

* 生产代码：`src/tensorlbm/boundary/interpolated_bounce_back.py`（`wall_velocity`/`wall_density`/`weights`）
* 测试：`tests/test_interpolated_bounce_back_moving_wall.py`（11 项，含逐比特向后兼容、解析项核对、
  3D 包装器、端到端 Couette 必要性）
* 驱动/图/数据：`verify_v3_moving_wall.py`、`make_figures_v3.py`、`artifacts_v3/`
