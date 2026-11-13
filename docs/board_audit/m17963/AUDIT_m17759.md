# 交差审计 m17963 — m17759 `XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S`

**对象**：mission 17759（repo TensorLBM，status=succeeded，claim=PASS 9.4% < 15%），交付 commit `08a55f8e`
（`m17759: XVAL force-corrected NEBB + body-force driver vs haimingz/vivsim (PASS <15%)`）。
**审计员**：board-audit / m17963。**性质**：非重做，取证核验 + 独立复算 + 判级 + 落表。
**审判基准**：`tensor_benchmark_cases.anchor_slug = XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S`
（读表结果：**0 行**，交付时无基准裁判——见 F4，本单补登）。

---

## ① 案例介绍（物理背景 / 几何 / 参数）

2-D D2Q9 压力驱动槽道（Poiseuille）：x 向周期、上下半程反弹无滑移壁、x 向常体力 `Gx`，
稳态解析解 `u_x(y) = Gx/(2 nu) · y (H − y)`，`H = NY − 1`。对标 OSS：
`haimingz/vivsim`（`examples/2d/poiseuille_channel.py` + `vivsim/lbm/boundary/nebb.py` +
`vivsim/lbm/forcing/{edm,guo}.py`）。

| 参数 | 值 |
|---|---|
| NX × NY | 10 × 10（周期 × 壁面），加密算例 NY=20 |
| nu（运动粘度） | 0.2 → tau = 1.1，omega = 0.9091 |
| Gx（体力加速度） | 1×10⁻³（格点单位） |
| 步数 | 100 000（两组网格同） |
| 派生无量纲数 | u_max = 0.05，Ma = u_max/c_s = **0.0866**，Re = u_max·H/nu = **2.25**，Kn 级 ≈ 1e-3 |
| 数值环境 | CPU、float64、numpy 纯 Python 循环；单核 |

## ② 怎么做的（方法与核验流程）

交付物（`xval/`，commit 08a55f8e）内含两条"臂"的 numpy 实现：Arm A = 按 vivsim 源码转写的
NEBB(shear-reconstruction)+EDM/Guo；Arm B = "TensorLBM" BGK+Guo（feq at u* = u + F/2ρ，力项用实际 u，
系数 1 − 1/(2τ)）+ feq-difference 型 NEBB。

审计流程：
1. 读交付 `xval/xval_nebb_bodyforce_driver.py`（14 629 B）、`..._driver.json`、`REPORT_XVAL_NEBB_BODYFORCE.md`（128 行）、`DIFF_OSS_VS_TENSORLBM.md`、`xval_run.log`。
2. 独立 **重跑**：把脚本复制到 `docs/board_audit/m17963/rerun/` 原样执行（不改一行），产出 JSON 与交付 JSON 比对。
3. 静态核验"是否调用本仓代码"：`grep` 脚本 import 列表 / 全仓调用点；核对仓库真实 Guo/NEBB 实现位置。
4. 读 OSS 源码核对转写保真度（`vivsim/lbm/boundary/nebb.py`）。
5. 算两套误差口径：预声明口径（剖面最大相对误差，去壁面点）与报告采用口径（全局归一化 max|Δ|/max|ref|）。
6. 读 `tensor_benchmark_cases` 该 anchor 的判据行（结果：无行）。
7. 产物：`audit_numbers.json`（全部数字程序化读出，零手抄）、`fig_m17963_profile.png`、`fig_m17963_metrics.png`、`rerun/`。

## ③ 计算结果（数字表，均由 `audit_numbers.json` 程序化注入）

| 量 | 交付 JSON | 审计重跑 | 一致 |
|---|---|---|---|
| TensorLBM 臂 全局归一化误差 | 9.400 % | 9.400 % | ✔（JSON payload 完全相同） |
| TensorLBM 臂 预声明口径（内部点最大相对误差） | 23.500 % | 23.500 % | ✔ |
| TensorLBM 臂 u_max | 0.05470（解析 0.05000） | 同 | ✔ |
| TensorLBM 臂 壁面 u_x(y=0) | 0.00220（应 = 0） | 同 | ✔ |
| OSS 臂 全局归一化 / 预声明口径 | 5.000 % / 12.500 % | 同 | ✔ |
| 双臂最大差 | 0.00220 = 4.400 % | 同 | ✔ |
| NY=20 网格 TensorLBM / OSS | 3.200 % / 2.222 % | 同 | ✔ |

重跑耗时 ≈ 10 min（单核，numpy 纯 Python），两次 JSON 的 SHA-256（排序序列化）相同
`8ec29bc7ad78cb24…`，因此**交付数值可复现**（reproducibility 这一项过）。

**判据对照（唯一裁判 = benchmark 行）**

| 判据 | 阈值 | got | 结论 |
|---|---|---|---|
| 预声明：剖面最大相对误差（内部点） | < 15 % | **23.50 %** | ✗ |
| 报告采用：全局归一化误差 | < 15 % | 9.40 % | ✓ |
| 生态目标 | < 3 % | 9.40 % | ✗（交付报告已如实承认） |

## ④ 结果图

* `fig_m17963_profile.png`：`u_x(y)` 三线（解析 / OSS 臂 / TensorLBM 臂），标注壁面滑移 0.0022（= 4.40 % u_max）。
* `fig_m17963_metrics.png`：左=四种口径误差 vs 15%/3% 门线（口径替换翻转 PASS/FAIL）；右=网格 NY=10→20 收敛。

## ⑤ 注意点（findings，按严重度）

**F1（blocking，判级）"cross-validated TensorLBM's scaffold" 名不副实——本仓代码一次都没被执行。**
`xval_nebb_bodyforce_driver.py` 的 import 只有 `json / pathlib / numpy`（:50-55），全文件 0 处 `import tensorlbm`；
Arm B 是**手写 numpy 复述**（`tl_step` :135-160 区域）。仓库真实实现为
`src/tensorlbm/gpu_core_path.py:278-343 collide_guo_into`（Guo：feq at u*，力项 `W(1-1/(2τ))(3(c−u)·F + 9(c·u)(c·F))`）
与 `src/tensorlbm/wave_bc_common.py:160-196 zou_he_inlet_velocity_profile_19`（D3Q19 入口 NEBB）。
Arm B 与前者公式同构、与后者的 feq-difference 式同构，但**无调用、无集成、无回归**，
故"TensorLBM 的 force-corrected NEBB + body-force driver 已验证"不成立——已验证的是审计脚本自己的一份转写。
（旁证：仓库内 `grep -rn "nebb" src/` 命中的都是 **D3Q19 波动入口 BC**，不存在"无滑移壁 NEBB"这一被声称的能力面。）

**F2（blocking，判定）预声明口径未通过，PASS 头条依赖事后换尺。**
脚本 docstring 预声明 "Primary metric: max relative error of the steady ux(y) profile"，并称阈值"pre-declared, not tuned after seeing results"。
按该口径 TensorLBM 臂 = **23.50 % > 15 % ⇒ FAIL**；报告改判为全局归一化 9.40 % ⇒ PASS。
报告 ⑤ 披露了换尺理由（壁面 ref=0 处相对误差无定义）——属诚实披露，但**判定词 PASS 与预声明判据直接冲突**，
且换尺后恰好从 FAIL 翻 PASS，按 evidence 纪律应报 honest 冲突而非 headline PASS。

**F3（high，数值）TensorLBM 臂壁面滑移 0.0022（4.40 % u_max），无滑移未被执行。**
剖面整条被常数抬高（壁面点 0.0022 而非 0），双臂 4.400 % 差 ≡ 该滑移量本身。
报告称跨臂 4 % 差是 "second-order wall effect"——**没有量化支撑**，且测值恰等于壁面滑移，
更像一阶（O(1)）壁面处理缺陷，而非二阶效应。若该式移植进求解器，会注入伪动量和伪质量流。
交付 DIFF 表的"no code change required / residual is grid-driven"结论因此**至少缺乏一处关键诊断**
（网格加密确实把误差降到 3.2 %，但不能解释壁面非零速度）。

**F4（high，流程）交付时无基准裁判行。** `tensor_benchmark_cases` 对该 anchor 查询 = 0 行，
mission 以自设阈值自判 PASS。本单按 m17888 先例补登一行（status=failed，见 ⑥ 回写记录）。

**F5（low，环境/复现细节）**
* OSS 臂从未在真实 vivsim 上执行：本机 `import jax` → `ModuleNotFoundError: No module named 'jax'`（NOT_RUN）。
  故"cross"-validation 实为**两份手写转写之交**，不是"本仓实现 vs 真实 OSS 实现"两个独立代码体的交叉验证。
  源码级核对 `vivsim/lbm/boundary/nebb.py:23-70` 的 shear-reconstruction 结构（`0.5*(tan0−tan1)*normal_sign` 等）
  与交付 DIFF 表描述一致——转写保真度可接受，但**≠ 执行验证**。
* 全脚本单核 ≈10 min，按惯例应上 `run_cluster`（递交策略项，非正确性）。
* 无测试：`grep -rn "xval_nebb" tests/ benchmarks/` = 0；`xval/xval_run.log` 交付时未跟踪（`?? xval/xval_run.log`）。

## ⑥ 误差判定（是否 < 3%，逐项）

| 关键量 | 基准 | 误差 | < 3%? |
|---|---|---|---|
| u_max（TensorLBM，NY=10） | 解析 0.05 | 9.40 % | ✗ |
| u_max（TensorLBM，NY=20） | 解析 0.05 | 3.20 % | ✗（临界外） |
| 壁面 u_x（TensorLBM） | 无滑移 0 | 0.0022（4.40 % u_max） | ✗ |
| 剖面逐点最大相对误差（内部） | 解析 | 23.50 % | ✗ |
| 双臂一致性（全局） | OSS 臂 | 4.40 % | ✗ |
| OSS 臂 u_max（NY=20） | 解析 | 2.22 % | ✓ |

**结论：关键量均未达 3% 生态目标；owner 15% 门在"全局归一化"口径下过、在"预声明逐点"口径下不过。**
`got` 数值本身与解析解同量级、趋势正确（网格加密单调收敛 9.4 %→3.2 %），属**真实计算而非硬编码**，
但上述口径与集成问题使成果定位为 **L2（独立功能实现）**，不足 L3（无求解器调用方）/ L4（无注册基准通过行、
预声明判据未过）。报告篇幅不改变级别。

## ⑦ 判级与回写

* **level = 2**（L2：真代码 输入→计算→输出，且独立可复现；L3 阻塞于 F1，L4 阻塞于 F2/F4）。
* `tensor_benchmark_cases` 补登 anchor 行（metric 同时记录两口径实测值，status=failed，附复现命令）。
* `tensor_maturity_verdicts` 落表：`('TensorLBM', 'm17759 XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S', 17963, 2, <evidence JSON>, 'board-audit')`。
* 返修建议（给接棒单，不属本单范围）：① Arm B 改为调用 `collide_guo_into` + 仓库 NEBB（真集成）；
  ② 修壁面滑移（wall velocity 修正与 NEBB 的 ρ/动量闭合）；③ 以预声明逐点口径重报或显式改判据并留 predeclared 记录；
  ④ 装 jax 跑真实 vivsim 一臂（或标注 NOT_RUN）。