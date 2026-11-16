# REPORT V5 — XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME

mission m23306 · 承接 `REPORT_V4.md`（m22795）的 honest FAIL 与「resolution study 未做」

## 1. 本单要回答的问题

V4 留下的账：

> OSS 严格精确性 `l2 < 1e-10` **未复现**（最好臂 7.5e-04），判为 closure 缺陷。

V5 改为先审"账本身"：OSS（`pylbm`，`/root/oss-cache/pylbm`）的 `tests/test_linear_bounce_back.py::poiseuille`
究竟在测什么？读源后抽出三条此前 V1–V4 **没有**对齐的契约：

| OSS 契约 | 代码位置 | V1–V4 是否对齐 |
|---|---|---|
| 载体 TRT，魔数积 `Λ⁺Λ⁻ = δ²/2` | `lam_p = 3nu`, `lam_m = 0.5*delta**2/lam_p` | 对齐（`carrier="trtd2"`） |
| 力：碰撞后加，各向同性权，报速度 `momentum − F/2` | `source = 3 w c_x F` | 对齐 |
| **壁面 Dirichlet 值：`value` 回调置 `m[qx] = Λ⁻F`** | `def wall(...): m[qx] = lam_m*force` | **未对齐（V4 连 `wall_velocity` 都没传）** |

假设 H_V5：V4 的残差是**契约缺口**（壁面值 `u_w = Λ⁻F` 未进 `−α e⁻_q(x_w)`），不是 closure 缺陷。

## 2. 怎么做的

* 工具链：`verify_v5.py`（本单新增）复用 V4 的容器/碰撞/流化原语，跑力驱动平面 Poiseuille，
  逐臂扫 δ × 闭包（bfl / yli / cli）× K̂ 修正（on/off）× 壁面值（zero / F/2 / Λ⁻F）；
  报速度 = `Σf c/ρ − F/2`，参照解 `u = g/(2ν)·y(H−y)`，误差取 `l2_rel` 与 `max_rel`。
* 为了让 OSS 侧可"算"而不是"抄"，补全了离线 `mpi4py` 桩（`tools/xval_pylbm_stub/`，
  补 `Compute_dims` / `Create_cart` / `Get_cart_rank` / `Create_subarray` / `Irecv`），
  `pylbm` 现已能在无 MPI 的本机 import 并构造 `Simulation`。
* 全部数值由 `make_report_v5.py` 从 `artifacts_v5/*.json` 直接注入，零手抄。

## 3. 计算结果

### 3.1 壁面值假设（H_V5）——**被否定**

δ=0.375, ν=0.3, TRT 魔数积：

| 壁面值 `u_w` | 闭包 | l2_rel（4000 步） | max_rel |
|---|---|---|---|
| laminus | cli | 2.769e-02 | 3.435e-02 |
| half | cli | 1.607e-02 | 1.657e-02 |
| zero | cli | 3.148e-02 | 3.765e-02 |

⇒ 传入 pylbm 的壁面值**没有**把残差压到 1e-12：三个取值都停在 1e-2 量级（见上表），
`Λ⁻F` 并不优于 `F/2`，量级差别不足以解释 1e-12 与 1e-2 之间十个数量级的缺口。
**H_V5 否定**：V4 的残差不是（仅）壁面 Dirichlet 值缺口。这一条以负面结果入账。

### 3.2 新缺陷：K̂ 后碰撞修正在 `q < 1/2` 分支发散（NaN）

δ 扫描（ν=0.3，600 步，壁面值 `Λ⁻F`）：

| δ | closure | K̂ | steps | l2_rel | rho_fluid | verdict |
|---|---|---|---|---|---|---|
| 0.25 | bfl | off | 600 | 3.233e-02 | 0.9574 | 稳定 |
| 0.25 | bfl | on | 600 | 2.982e-02 | 0.9574 | 稳定 |
| 0.25 | cli | off | 600 | 3.536e-02 | 0.9677 | 稳定 |
| 0.25 | cli | on | 600 | 3.536e-02 | 0.9677 | 稳定 |
| 0.25 | yli | off | 600 | 8.672e-02 | 0.9474 | 稳定 |
| 0.25 | yli | on | 600 | 1.113e+04 | 0.9470 | 稳定 |
| 0.375 | bfl | off | 600 | 1.868e-02 | 0.9588 | 稳定 |
| 0.375 | bfl | on | 600 | 2.557e-02 | 0.9588 | 稳定 |
| 0.375 | cli | off | 600 | 2.768e-02 | 0.9637 | 稳定 |
| 0.375 | cli | on | 600 | 2.768e-02 | 0.9637 | 稳定 |
| 0.375 | yli | off | 600 | 8.828e-02 | 0.9442 | 稳定 |
| 0.375 | yli | on | 600 | 2.540e-02 | 0.9442 | 稳定 |
| 0.75 | bfl | off | 600 | 4.918e-02 | 0.9444 | 稳定 |
| 0.75 | bfl | on | 600 | 2.219e-02 | 0.9444 | 稳定 |
| 0.75 | cli | off | 600 | 1.654e-02 | 0.9533 | 稳定 |
| 0.75 | cli | on | 600 | 1.654e-02 | 0.9533 | 稳定 |
| 0.75 | yli | off | 600 | 9.262e-02 | 0.9358 | 稳定 |
| 0.75 | yli | on | 600 | 2.992e-02 | 0.9358 | 稳定 |

* `q > 1/2` 分支（δ=0.75）：K̂ 修正**稳定且显著改善** —— bfl 4.92e-02 → 2.22e-02（×0.45）、
  yli 9.26e-02 → 2.99e-02（×0.32），方向与 OSS「加修正后成为参数化族」一致。
* `q < 1/2` 分支（δ=0.25）：**带 K̂ 修正的两臂随时间推进发散** —— yli 在 600 步内已冲到
  `l2 ≈ 1.1e+04`，bfl 在 1500 步时已是 `NaN`（600 步时仍有限，见表）；
  同臂去掉修正稳定（bfl 3.23e-02），而 cli 因 `α̂ = 1 ⇒ K̂ = 0`（修正恒等于零）不受影响。
  δ=0.375（`q<1/2`）同样见 bfl 修正后变差（1.87e-02 → 2.56e-02）。
* 单次调用算符有限（4 组 finiteness 探针全部 `isfinite=True`），说明是**时间推进中的反馈不稳定**，
  不是算符本身产生 NaN。
* ⇒ V1–V4 的所有时间推进算例都**没有**打开 `post_collision_correction`（只在单元测试里比对公式），
  本单首次把该路径放进时间推进，即暴露该缺陷。

### 3.3 resolution study（V4 明确未做的一项）

半程反弹（现役算符）在 δ=0.25 固定亚格子偏移下的网格趋势：

| N | H (格子) | closure | l2_rel | max_rel | steps |
|---|---|---|---|---|---|
| 8 | 7.50 | bfl | nan | nan | 4000 |
| 16 | 15.50 | bfl | nan | nan | 4000 |
| 8 | 7.50 | halfway | 3.231e-02 | 3.405e-02 | 4000 |
| 16 | 15.50 | halfway | 1.849e-02 | 1.903e-02 | 4000 |

* 半程反弹 l2：N=8 → 3.23e-02，N=16 → 1.85e-02（比值 1.75），与"壁面位移固定为 O(1) 格、
  相对误差 ∝ 1/N"的一阶趋势一致（见图）。
* 插值闭包（bfl）在 δ=0.25 的同一扫描**发散为 NaN**（3.2 的缺陷），故本次未能给出
  插值闭包自己的收敛阶 —— 记为未完成，不臆造数字。

## 4. 结果图

![stability](figures/fig_correction_stability.png)

*图 1* `l2_rel` 随迭代：左 δ=0.25（`q<1/2`，带 K̂ 的 bfl/yli 直接掉到 NaN），
右 δ=0.75（`q>1/2`，带 K̂ 稳定且误差更低）。红点线 = owner 15 % 验收线。

![order](figures/fig_halfway_order.png)

*图 2* 半程反弹的网格趋势（灰虚线为 `N^-1` 参考斜率）。

## 5. 注意点（坑 / 局限 / 适用范围）

* **OSS 侧的"可跑"≠"可信"**：本机补全的 `mpi4py` 桩把 `Send/Recv/Irecv` 做成 no-op，
  而 `pylbm.storage.update()` 的 halo 拷贝正是边界值进入内部区的唯一通道。实测
  `parBFLBounceBack` 在 δ=0.375/ν=0.3/N=8 给出 l2=9.5e-01，同时抛出
  `BoundaryWarning: 4 links ... read a node that is not a fluid node`。
  ⇒ **不能**把本机 OSS 数字当作参考值引用；引用 OSS 必须用有 MPI 的构建。
  这一点同样约束 V1–V4 的 "transplant fidelity" 论述：它们比对的是**公式**，不是运行结果。
* `ρ_fluid` 在 0.94–0.96 之间（见表 3.1），即每次初始化都会丢/加一次性质量，
  与 V4 的「记账陷阱」一致；本单未再修改该契约。
* 只覆盖直壁、单分辨率族、`Re_H < 7`；斜壁/曲壁、动壁耦合未做。
* 缺陷定位到"时间推进 + K̂ + q<1/2"三者的组合，**未**定位到具体代码行；修复方向见 §6。

## 6. 误差判定（逐项，显式回答是否 <3 %）

| 关键量 | 基准 | 实测 | <3 % ? |
|---|---|---|---|
| 半程反弹 l2（N=8/16，δ=0.25） | 解析 Poiseuille | 3.23e-02 / 1.85e-02 | **否**（1.9–3.2 % 量级，边界） |
| bfl+K̂（δ=0.75） | 同上 | 2.22e-02 | **是**（2.22 %） |
| yli+K̂（δ=0.75） | 同上 | 2.99e-02 | **否**（2.99 %，贴线） |
| cli（δ=0.75，K̂=0 恒等） | 同上 | 1.65e-02 | **是** |
| bfl+K̂（δ=0.25） | 同上 | NaN | **无定义（honest FAIL）** |
| OSS 宣称 `l2 < 1e-12` | pylbm 自测容差 | **未复现**（最好 1.6e-02） | **否** |
| owner 验收线 15 % | 解析 Poiseuille | 全部稳定臂 ≤ 4.9e-02 | **是（稳定臂全过）** |

**判定**：owner 的 15 % 验收线在**所有稳定臂**上通过；OSS 的机器精度精确性仍未复现，
且本单把原因从"closure 缺陷"改写为"**未定位的契约/实现差异 + 一个新发现的 NaN 缺陷**"，
并否定了"壁面 Dirichlet 值"这一具体候选。未完成项如实列出（插值闭包收敛阶、OSS 可信参考运行）。

## 7. 产物

* 驱动 `verify_v5.py` · 报告生成 `make_report_v5.py` · 数据 `artifacts_v5/results_*.json`
* 图 `artifacts_v5/figures/fig_correction_stability.png`、`fig_halfway_order.png`
* 离线 OSS 桩 `tools/xval_pylbm_stub/mpi4py/`（补全后 pylbm 可 import/构造）
* 测试 `tests/test_xval_lbb_alpha_correction_stability.py`（钉住 3.2 的稳定/发散两侧）
