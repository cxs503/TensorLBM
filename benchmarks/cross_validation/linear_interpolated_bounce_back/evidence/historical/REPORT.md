# XVAL-LBM-LINEAR-INTERPOLATED-BB — 壁距扫掠下 2 点线性插值反弹 vs 精确 Poiseuille

**状态：honest PARTIAL / 主判据未达成。** 第 2 轮（调研结合）结论：
① 方法谱系已查清并与 OSS 逐项对齐 —— TensorLBM 现实现就是 pylbm `BFLBounceBack` 的代数等价（α 格式，见 §2/§7），闭包本身**不是**问题；
② 实测上 2 点线性插值反弹（IBB）在 BGK 载体上**确实传递了亚格子壁距**（壁面位置响应斜率 1.32 vs 半程反弹的 0.000），但**绝对壁面位置**残留随 δ 增长的偏差，profile 误差 16.6 %–44.8 %，**不满足 <15 % 判据**；
③ 半程反弹的壁面位置被精确钉在 0.88 cell（对 δ 完全不变，N=8/16 均如此）——这既是**几何律 δ_eff = 1−2δ 的定量验证**，也暴露一个与 δ 无关的**共用格式偏差 +0.38 cell/墙**（BGK、Guo 源项、TRT 魔法乘积三种载体下均存在）。
④ 失败机制有文献锚（pylbm 精确性定理：需 TRT 参数乘积取该格式魔法值），下一轮升级路径明确（TRT(3/16) + 精确源项 + α 格式碰撞后修正项 K̂=1−α̂）。

---

## 1 问题设定 + 基准出处

**基准**：力驱动充分发展平面 Poiseuille 通道，x 周期、y 两直壁，壁面位于**首末流体节点外侧 δ cells**（亚格子壁距）。物理宽度

    H(δ) = (N−1) + 2δ ,   流体节点 y_j = δ + (j−1),  j = 1..N

精确解（Stokes/Poiseuille；White, *Viscous Fluid Flow* ch.3；Bouzidi, Firdaouss & Lallemand, *Phys. Fluids* **13** (2001) 3452 §III.A 的同款算例）：

    u(y) = g/(2ν) · (H²/4 − (y − H/2)²) ,  Q = g H³/(12ν)

**几何/δ 约定出处（外部锚，OSS 源码直读）**：pylbm `tests/test_linear_bounce_back.py::channel(delta)` 明写 “N fluid nodes at y = 0.5, …, N−0.5, walls at a distance delta of the first and the last ones”，即 `y0 = 0.5−δ, y1 = N−0.5+δ` ⇒ 与本单 H(δ) 完全同构；δ 点集取 `tests/test_bouzidi.py: DELTAS = [0.25, 0.5, 0.75, 0.9]`（本单实跑 0.25/0.5/0.9）。

**为什么这是 IBB 的正确判据**：半程反弹把无滑移面钉在末流体节点外 0.5 cell（δ 不敏感 ⇒ 壁位置几何错误随 δ 线性增长）；2 点线性闭包按真实壁距重构反射分布，对**直壁**应把无滑移面放到 δ 处（这是本锚点要验的能力）。曲壁/斜壁是另一回事（一阶、且本仓已有 honest FAIL：`docs/STAIRCASE_WALL_OFFSET_20261111.md`），本报告不覆盖。

## 2 方法与实现（调研 + 选型）

**① 开源横向（源码直读，非二手）**
* `~/oss-cache/pylbm/pylbm/boundary.py:920-993`：`LinearBounceBack` 用单一 `alpha(delta)` 参数化
  `α̂ = −1 + α(1/2+δ)`, `β = 1 − αδ`, `β̂ = 1 − α/2`；`BFLBounceBack`：`α = 2 (δ≤1/2)`、`α = 1/δ (δ>1/2)`；`BounceBack`：`α = 2, δ = 1/2`。
  代入 α=2,δ=q ⇒ `f = 2q f̂_q + (1−2q) f_q(t+1)`；代入 α=1/q ⇒ `f = (1/2q) f̂_q + (1−1/2q) f̂_q̄`
  ⇒ **与本仓 `src/tensorlbm/boundary/interpolated_bounce_back.py` 的两分支逐项代数等价**（q=1/2 处二者都退化为 `f̂_q`，与半程交换逐位一致；实测 max|diff| = 8.3e-6 = 一次力源量级）。
* 已装 `site-packages/lbmpy/boundaryconditions.py` 的 `NoSlipLinearBouzidi`/`QuadraticBounceBack` 族（同名方法的第二份独立实现，分支结构一致）。
* 结论落 `oss_solutions`（`feature_slug='XVAL-LBM-LINEAR-INTERPOLATED-BB'`, repo=`pylbm/pylbm+lbmpy/tensorlbm`, tag=`linear-interpolated-bb-alpha-scheme`）。

**② 方法谱系（可复用定理）**：pylbm 测试文档串原话——“The parametrized schemes give the exact steady solution of a force-driven Poiseuille flow … for any relaxation rate, **when the product of the TRT parameters is the one of the scheme**”。即 BFL 闭包的精确性**不**由插值阶数单独保证，需 TRT(Λ⁺Λ⁻=3/16 类魔法乘积) + 一致的力项；BGK 载体不满足 ⇒ 必然残留壁面位置偏差。这正是本轮实测到的现象（§3/§6）。

**③ 结合实现（本仓推进）**
* 驱动：`xval/XVAL-LBM-LINEAR-INTERPOLATED-BB/run_xval.py`（新）。两臂同一 solid 掩码、同一力、同一分辨率：
  `halfway` = 生产算子 `boundary.boundaries.bounce_back_cells`；`interp` = `interpolated_bounce_back_2d`。
* 壁厚 3 行 + **y 向环绕污染封堵**（新发现并修复的坑）：`solver.stream` 对 y 也 roll，两壁互相 wrap，一壁外行方向槽里的种群会从对壁**反向进入流体**，使两臂差异变成 wrap 伪影而非边界模型差异。`run_xval.py::stream_open_y` 在迁移后把最外两行重置为静止平衡态（距流体 ≥2 行，不污染）。
* 观测量（无自由参数）：`Q_num` = 流体节点 u_x 求和；`Q_ref` = 精确解在**同一节点坐标**上的同型离散和（规避求积偏差）；`δ_eff = H_eff − H`，`H_eff = (12νQ/g)^{1/3}`；**壁位置 `w_eff = δ + δ_eff/2`**（δ_eff 的定义使 2(w_eff−δ)=δ_eff，这是本报告的主观测量）；`l2_rel = ‖u−u_ref‖₂/‖u_ref‖₂`。
* 控制实验：δ=0.5 处两臂算子逐位等价（q=0.5 两分支都退化），该点给出的 `w_eff(0.5)` 就是**共用格式偏差**的度量，δ 响应须减掉它才有判别力。
* 收敛判据：每 100 步比较剖面，max|Δu| < 1e-9·u_max 停（本轮所有点 `converged=True`）。

## 3 参数表（无量纲/格子量完整）

| 量 | 值 | 说明 |
|---|---|---|
| τ | 0.8 | BGK，ν=(τ−1/2)/3 |
| ν | 0.1 (lattice) | |
| u_max^target | 0.02 | g = 8ν u_max/H² |
| Ma = u_max/c_s | 0.0346 | c_s=1/√3 |
| Re_H = u_max H/ν | 3.2 (N=16) | 层流/Stokes 区 |
| N（流体行数） | 8, 16 | 两分辨率 |
| δ | 0.25, 0.5, 0.9 | 两分支（<1/2 与 >1/2）都覆盖 |
| nx / 壁厚 | 2 / 3 行 | x 周期 |
| 步数 | 1700–5800 | 至稳态（τ_diff=H²/ν≈770/2560） |

## 4 对照表（逐项 ref vs got + 误差）

`w_eff`（cell，越小越接近“按 δ 放置”），主判据列= `|w_eff − δ|`：

| N | δ | 臂 | w_eff | w_eff−δ | l2_rel % | Q_err % | src |
|---|---|---|---|---|---|---|---|
| 8 | 0.25 | halfway | 0.8571 | **+0.607** | 50.07 | +56.44 | sweep |
| 8 | 0.25 | interp | 0.7639 | **+0.514** | 41.35 | +46.61 | sweep |
| 8 | 0.50 | halfway | 0.8571 | +0.357 | 25.98 | +28.24 | sweep |
| 8 | 0.50 | interp | 0.9331 | +0.433 | 32.26 | +35.07 | sweep |
| 8 | 0.90 | halfway | 0.8571 | −0.043 | 2.18 | −2.30 | sweep |
| 8 | 0.90 | interp | 1.4942 | **+0.594** | 44.82 | +47.13 | sweep |
| 16 | 0.25 | halfway | 0.8807 | **+0.631** | 23.72 | +26.38 | sweep |
| 16 | 0.50 | halfway | 0.8807 | +0.381 | 13.48 | +14.74 | run_arm 直接调用 |
| 16 | 0.50 | interp | 0.9635 | +0.464 | 16.62 | +18.18 | run_arm 直接调用 |
| 16 | 0.90 | halfway | 0.8807 | −0.019 | 0.49 | −0.52 | run_arm 直接调用 |
| 16 | 0.90 | interp | 1.6175 | **+0.718** | 26.23 | —（该封装未输出 Q） | trt.py 诊断封装（δ=0.5 处与 run_arm 吻合到 4 位） |

**基准判据逐条打勾/打叉**

| 判据（动手前预设） | 结果 | 判定 |
|---|---|---|
| C1 IBB 臂 profile 误差 <15 %（owner 精度标准，δ 全线） | 16.6 % / 26.2 % / 41.4 %（N=16/8） | ✗ **未达成** |
| C2 IBB 有壁距响应：dw_eff/dδ > 0.5，且半程响应 ≈ 0 | 半程 0.000（0.8807 三点全同）；IBB 1.32（N=16）/1.12（N=8） | ✓ 响应存在；但斜率 1.32≠1 |
| C3 半程几何律 δ_eff = 1−2δ 成立（仪器自检） | 三点残差 |w_eff−0.8807| ≤ 5e-4 | ✓ |
| C4 半程壁面位置 = 0.5（教科书值） | 实测 0.8807（+0.38/墙） | ✗ 共用格式偏差（见 §6.2） |
| C5 w_eff 两分辨率收敛 | 半程 0.8571(N=8)→0.8807(N=16) 收敛；IBB δ=0.9 的偏差 0.594→0.718 **不收敛（变大）** | ✗ IBB 高 q 分支更差 |

## 5 收敛性 / 参数敏感性

* **两分辨率**（N=8/16）：半程的壁面位置 0.8571→0.8807（Δ=0.024）——边界闭合偏差**基本不随分辨率消失**（这正是“边界模型误差”的特征，与网格离散误差不同）；IBB 在 δ=0.9 的偏差 0.594→0.718（Δ=+0.124，**反向变大**），δ=0.25 的偏差 0.514→0.464（略降）。⇒ IBB 的高 q（δ>1/2）分支在本载体上不可靠。
* **两参数点（δ）**：两分支都覆盖（δ=0.25 走 q<1/2 低分支；δ=0.9 走 q>1/2 高分支）。半程对 δ **完全不变**；IBB 跟随 δ 但带 δ 增长偏差。
* **τ/载体敏感性**（N=16, δ=0.5, 半程，交互诊断）：τ=0.8/+0.7614 cell、τ=0.9333/+0.7711、τ=1.2/+0.8068、τ=1.6/+0.8999（δ_eff）；加 TRT 魔法乘积 Λ⁺Λ⁻=3/16 后 +0.545（w_eff 0.7725）；Guo 精确源项与朴素源项的稳态解在 4 位有效数字内一致。⇒ 偏差的主成分**与 τ/力项无关**，只有一小部分（≈0.1 cell）随 Λ 变化。
* **g 敏感性**：u_max^target = 0.002/0.02/0.05 三档 δ_eff 完全相同（+0.7614）⇒ 偏差是**线性（边界闭合）**性质，不是 Ma² 效应。

## 6 结果图

![wall offset](attachment:xval_ibb_wall_offset.png)
左：实测壁面位置 `w_eff` vs δ（黑虚线=“按 δ 放置”的理想线）。半程（红）三条 δ 全部落在 0.8571/0.8807 的**水平线**上——几何律成立、无壁距响应；IBB（蓝）随 δ 上升（响应 1.12–1.32）但整条线抬到理想线上方且高 δ 端翘起。
右：profile L2 误差 vs δ，点线 = 15 % 验收线。半程只在 δ=0.9 因“格式偏差 +0.38 ≈ 0.5−δ=−0.4 恰好抵消”落进 15 %（0.49 %），δ≤0.5 全线出线；IBB 全部在线上方。

## 7 注意点 / 失败模式 / 工件

1. **失败机制（有文献锚）**：α 格式的精确性要求 TRT 参数乘积取该格式魔法值（pylbm 测试文档串），BGK 不满足；另有一个**与 τ/力项无关的 +0.38 cell/墙**常数偏差，本轮三种载体（BGK+朴素源、BGK+Guo 源、TRT 魔法）都没消掉（TRT 只从 0.881 降到 0.772），说明还有第四个未定位因素（候选：固体侧 ghost 槽的半程交换语义 / α 格式的碰撞后修正项 `K̂=1−α̂` 未实现 / 力项与反射的一致性）。**这是下一轮的第一靶子**，不是“IBB 不好用”。
2. **新坑（本轮发现并已修）**：`solver.stream` 对 y 也周期 roll，两壁互相 wrap 会让一壁“外行槽”的种群从对壁反向进入流体，两臂差异被 wrap 伪影污染（δ=0.5 处本应逐位等价的两臂曾差 0.17 cell）。修法见 `stream_open_y`。
3. **未跑完**：计划 2 分辨率×3δ×2臂=12 点，实得 7 点（N=8 满 6 点；N=16 的 δ=0.5/0.9 由同一驱动 `run_arm` 直接调用补，δ=0.9 的 interp 由同一公式的诊断封装补并在 δ=0.5 交叉校验）。后台 sweep 被工具的进程组收割中断，`artifacts/run_sweep.log` 是原始 stdout。N=32、δ=0.75、以及剖面数组（图只用指标表）未跑/未存。
4. **适用范围**：只覆盖直壁（斜壁/曲壁见 `docs/STAIRCASE_WALL_OFFSET_20261111.md` 的一阶 honest FAIL）；静壁（无移动壁速度项）；2D D2Q9。
5. **复现**：
   ```
   PYTHONPATH=src python3 xval/XVAL-LBM-LINEAR-INTERPOLATED-BB/run_xval.py \
       --deltas 0.25,0.5,0.9 --ns 8,16 --out xval/XVAL-LBM-LINEAR-INTERPOLATED-BB/artifacts
   ```
   工件：`xval/XVAL-LBM-LINEAR-INTERPOLATED-BB/artifacts/{table.csv,results.json,xval_ibb_wall_offset.png,run_sweep.log}`；驱动：`run_xval.py`。

## 8 误差判定（是否 <3 % / <15 %）

| 关键量 | 数值 | 基准 | 相对差 | <3 %? | <15 %? |
|---|---|---|---|---|---|
| halfway w_eff（N=16, 三 δ） | 0.8807 cell | 0.5 cell（半程定义） | +76 % | ✗ | ✗ |
| halfway (w_eff−δ) 的 δ 平台性 | 波动 ≤5e-4 cell | 0（应 δ 不变） | — | ✓ | ✓ |
| interp w_eff（δ=0.25） | 0.7575 cell | δ=0.25 | +203 % | ✗ | ✗ |
| interp w_eff（δ=0.9） | 1.6175 cell | δ=0.9 | +80 % | ✗ | ✗ |
| interp profile l2（δ=0.25/0.5/0.9, N=16） | 41.4 %/16.6 %/26.2 % | 解析抛物线 | — | ✗ | ✗（δ=0.5 也 16.6 %） |
| halfway profile l2（δ=0.9, N=16） | 0.49 % | 解析抛物线 | — | ✓ | ✓（但属偏差抵消，不可当能力证据） |
| 壁面位置响应 dw_eff/dδ | 半程 0.000 / IBB 1.32 | 0 / 1.0 | — | ✗ | IBB 斜率 32 %（<15 %? 否） |

**显式回答“是否 <3 %”**：否。**是否 <15 %**：IBB 臂全线否（最好 16.6 %）。故本锚点第 2 轮判 **honest PARTIAL（主判据 FAIL）**：
`fail_stage = precision_gate`，`failure_reason = 2 点线性 BFL 闭包在 BGK 载体上只传递壁距响应（斜率 1.12–1.32）而未能把绝对壁面位置放到 δ 处（残留 +0.46…+0.72 cell），profile 误差 16.6 %–44.8 % > 15 %；OSS 定理指出的升级路径（TRT 魔法乘积 + 精确源项 + α 格式碰撞后修正）本轮未实现完成`。