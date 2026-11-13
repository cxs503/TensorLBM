# XVAL — force-corrected NEBB + body-force driver vs `haimingz/vivsim`

**Anchor:** `XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S` · 仓 TensorLBM · 轮次 2（返修轮）
**Comparand:** OSS `haimingz/vivsim`（JAX，Apache-2.0）——`lbm/boundary/nebb.py`、
`lbm/boundary/_helpers.py`、`lbm/forcing/{edm,guo}.py`、`lbm/basic.py`、
`examples/2d/poiseuille_channel.py`
**判据:** owner 精度标准 <15%；生态 stretch <3%；交付口径另设 **arm-to-arm 一致性 <1e-9**
**状态:** PASS（arm-to-arm 6.4e-14；两条 arm 对闭式解均在舍入误差级）

> 本轮取代第 1 轮的 `xval/xval_nebb_bodyforce_driver.py`。旧脚手架把解析解取成
> `H = ny-1` 并与"墙上密度任意"的 NEBB 比较，5 % 的偏差其实是壁面位置约定（见 §5），
> 不是实现误差；本轮把该项做成了可测量的诊断量 `effective_wall_offset`。

## ① 问题 + ref 出处

等截面二维槽道、x 周期、上下两排无滑移壁、x 向常体积力 `gx`；定常解为抛物线
`u_x(y) = gx/(2ν)·(y-y0)(H-(y-y0))`。要交叉验证的是三件耦合的事：

1. **力修正壁速** `u_wall - g/(2ρ_w)`（壁面上半个力步的速度修正）；
2. **NEBB 壁核**（非平衡反弹 / Zou–He 的剪切重构写法）——本仓此前只有
   `wave_bc_common` 的粗糙形式 `f_k = feq_k - feq_opp + f_opp`；
3. **体积力方案** EDM 与 Guo-BGK，及其各自要求的"碰撞用修正前/修正后速度"约定。

上游出处（零手抄，逐文件 sha256 记录在 `runs/XVAL-NEBB-BODYFORCE/results.json`
的 `oss_provenance`，转写树 `/tmp/vivsim_src`）：

| 本仓实现 | 上游文件/函数 |
|---|---|
| `nebb_row` | `lbm/boundary/nebb.py::boundary_nebb` |
| `corrected_wall_velocity` | `lbm/boundary/_helpers.py::get_corrected_wall_velocity` |
| `recovered_wall_density` | `lbm/boundary/_helpers.py::get_rho_wall_from_velocity` |
| `velocity_correction` | `lbm/basic.py::get_velocity_correction` |
| `guo_forcing_term` / `forcing_edm` / `forcing_guo_bgk` | `lbm/forcing/{guo,edm}.py` |
| `BodyForceChannel.step` 顺序 | `examples/2d/poiseuille_channel.py` |

**已披露局限**：上游是 JAX 实现，本环境 `pip install jax` 超时（记入 `friction`），
因此 arm A 是**逐行 NumPy 转写**（保留上游 `f_i = f_i^full - w_i` 的移位存储约定，
`rho_wall = 1` 默认），不是运行上游二进制。`oss_binary_executed: false`。它证明"同一
公式的独立第二实现与我们的实现一致"，不证明上游 JIT 路径逐位一致。

## ② 方法与实现

* 新能力模块 `src/tensorlbm/nebb_bodyforce.py`（NumPy 后端，`nebb_row` 同时支持 torch 张量）：
  `equilibrium/macroscopic`、`guo_forcing_term`、`forcing_edm`、`forcing_guo_bgk`、
  `nebb_row`（剪切重构 NEBB）、`nonequilibrium_nebb_row`（经典 Zou/He，作参照）、
  `corrected_wall_velocity`、`velocity_correction`、`recovered_wall_density`、
  `apply_force_corrected_nebb`、`analytic_poiseuille`、`effective_wall_offset`、
  `profile_errors`、驱动 `BodyForceChannel`（4 种壁模型 × 2 种力方案）；
  CLI：`python -m tensorlbm.nebb_bodyforce`。
* 壁面方向表由格点常数**推导**（非手抄），单测与上游表逐字段对齐（4 个壁）。
* 交叉对标脚本 `benchmarks/xval/xval_vivsim_nebb_bodyforce.py`：arm A（上游转写）
  vs arm B（本仓模块），并为"同一 arm 内换壁模型/力方案/网格/力幅值/黏度"提供敏感性。
* 证据构建 `benchmarks/xval/nebb_bodyforce_evidence.py`：impl 非占位审计、入口、
  运行工件 sha256、调用点 grep 验证、benchmark 行**从原始 profile 重算**。

## ③ 参数表

| 量 | 主算例 | 无量纲解释 |
|---|---|---|
| 网格 | `nx=8`（周期，解与 x 无关）、`ny=10/20/40` | — |
| 运动黏度 ν | 0.2（敏感 0.05 / 0.5） | `tau = 3ν+0.5 = 1.1`（0.65 / 2.0） |
| 体积力 gx | 1e-3（敏感 1e-2 / 1e-1） | `u_max/c_s ≈ 0.087`（0.87 / 8.66），`c_s=1/√3` |
| 时间步 | 至残差 <1e-13 | ny=10: 1401 步；ny=40: 22401 步 |
| 初场 | `f = feq(ρ=1, u = -g/2)`（上游 example 初值） | — |
| 精度 | NumPy float64，CPU，单进程 | — |

## ④ 对照表（ref = 闭式抛物线，壁面在节点平面）

主算例 ny=10、ν=0.2、gx=1e-3；η = max|u−u_ref| / max|u_ref|：

| 样例 | u_max(ref) | u_max(got) | η | <15 %? | <3 %? | 判定 |
|---|---|---|---|---|---|---|
| arm A EDM（上游转写） | 0.05 | 0.05 | 7.1e-15 | ✓ | ✓ | PASS |
| arm A Guo-BGK | 0.05 | 0.05 | 1.2e-14 | ✓ | ✓ | PASS |
| **arm B EDM（本仓模块）** | 0.05 | 0.05 | **5.7e-14** | ✓ | ✓ | PASS |
| **arm B Guo-BGK** | 0.05 | 0.05 | **1.2e-14** | ✓ | ✓ | PASS |
| arm B，壁密度用上游 recover 口径 | 0.05 | 0.05 | 3.2e-14 | ✓ | ✓ | PASS |
| arm B，经典 Zou/He NEBB（无剪切重构） | 0.05 | 0.0522 | 4.4e-2 | ✓ | ✗ | 敏感性 |
| arm B，半程反弹壁（无修正） | 0.05 | 0.0376 | 0.249 | ✗ | ✗ | 负对照 |

* **arm-to-arm**：`max|B−A|/max|A|` = 6.4e-14（阈值 1e-9）→ PASS。
* 残余质量漂移：ny=10 至定常 −5.6e-14；壁面物理速度 `u_wall,corrected` = −4.3e-19（即精确无滑移）。
* 半程反弹壁的 0.249 全部是**壁面位置**：把参考解平移 0.5 格后 η = 9.3e-3（见 §5）。

## ⑤ 收敛性 / 参数敏感性（≥2 点，实做 5 点）

**网格细化**（η，壁面在节点平面）：

| ny | arm A Guo | arm B Guo | arm B recovered | Zou/He NEBB | 半程反弹 |
|---|---|---|---|---|---|
| 10 | 1.2e-14 | 3.2e-14 | 3.2e-14 | 4.40e-2（偏移 0.097） | 0.249（偏移 0.523） |
| 20 | 1.6e-13 | 7.6e-14 | 7.6e-14 | 9.78e-3（偏移 0.046） | 0.111（偏移 0.512） |
| 40 | 3.2e-13 | 8.6e-13 | 8.6e-13 | 2.32e-3（偏移 0.023） | 0.0526（偏移 0.506） |

* 力修正 NEBB + EDM/Guo 的定常解与闭式解**逐点相等到舍入误差**（该问题存在精确离散
  定常解），细化不改变结论。
* 经典 Zou/He NEBB 的效率是"壁面位置偏移 ≈ 0.9/ny 格"，误差 ~1/ny²，收敛但同网格下比
  剪切重构差 12 个数量级 → 本仓 `wave_bc_common` 旧形式属这一类，量化为 4.4e-2（ny=10）。
* 半程反弹的偏移恒为 0.5 格（0.523→0.506），误差 ~1/ny（一阶），与文献一致。

**力方案 / 力幅值 / 黏度**：

| 变化 | η | 备注 |
|---|---|---|
| EDM vs Guo-BGK（ny=10） | 5.7e-14 / 1.2e-14 | 定常解一致；**约定必须配对**（EDM 用修正前速度碰撞，Guo 用修正后），混用实测 45.5 % 误差 |
| gx = 1e-2（Ma≈0.87） | 6.9e-14 | 定常抛物解与 Ma 无关 |
| gx = 1e-1（Ma≈8.7，超压缩性极限） | 9.3e-11 | 仍收敛，但已出 LBM 有效域 |
| ν = 0.05（tau=0.65） | 2.3e-13（4801 步） | — |
| ν = 0.5（tau=2.0） | 1.0e-14（801 步） | 定常解与 ν 成反比，η 不变 |

## ⑥ 本仓 vs OSS 差异清单 + 移植取舍

| 项 | vivsim | 本仓本轮 | 取舍理由 / 实测影响 |
|---|---|---|---|
| 存储约定 | 移位 `f−w` | 全量 `f`；`nebb_row` 移位不变（单测 1e-15） | 与其他 TensorLBM 2D 核一致；等价性由单测证明 |
| 壁密度 | `wrap_force_corrected` 默认 `rho_wall=1` | 默认取场上壁密度；另有 `"recover"` 口径复刻上游 `get_rho_wall_from_velocity` | 两者差 6.4e-14（ny=10），故默认取更自洽者，同时保留上游口径供对标 |
| 剪切重构 | `boundary_nebb` 用实测剪切 | 同式（1:1 转写）+ 另存经典 Zou/He 形式 | 量化出 12 个数量级差距，是本轮最有价值的一条 |
| 后端 | JAX（jit/scan） | NumPy（float64），壁核兼容 torch | 集群 loong64 无 JAX/torch CUDA 栈；torch 兼容用于主求解器调用 |
| 方向表 | 手写 `BOUNDARY_SPEC` | 由 `C` 推导 + 单测对齐上游表 | 防格点改动后残留陈旧索引 |
| 力方案 | EDM / Guo-BGK / Guo-MRT / KBC+EDM | EDM / Guo-BGK（BGK） | MRT/KBC 属另一能力面，本轮不扩 |
| 壁面速度约定 | `u−g/(2ρ)` | 同；驱动内另给 `recovered` 密度变体 | 保持"物理速度无滑移"语义，`u_wall,corrected ≈ 1e-19` |

**移植取舍总结**：只移植"力修正壁速 + 剪切重构 NEBB + 两种力方案 + 驱动步序"这条最小
闭环，MRT/KBC 与 IBM 不在本轮；上游 JAX 的向量化/JIT 不移植（本节已量化：数值结论与
向量化方式无关，arm A/B 差 6.4e-14）。

## 失败模式与工件

* 混合力方案约定（EDM 用修正后速度）→ 注入动量差 (1−ω/2) 倍，实测 η=45.5 %：已在
  `BodyForceChannel.step` 里写死配对并在 docstring 记为 API 契约。
* 用 `u* = j/ρ + g/(2ρ)` 的"修正速度"内部约定 + NEBB 壁不修正 → 壁面无滑移落到错位置。
* 首轮脚手架的 5 % 偏差 = 参考解壁面位置取错（`H=ny−1` vs 偏移 0），本轮用
  `effective_wall_offset` 把它变成可测量量。
* 工件：`runs/XVAL-NEBB-BODYFORCE/{results.json, run.log, evidence.json,
  cli_guo_ny10.json, fig_profile.png, fig_cross_arm.png, fig_convergence.png,
  fig_field.png}`；证据门 `evidence.json` 全项 PASS（impl/entry/artifacts/callers/
  tests/benchmark 重算/benchmark 通过）。
* 复现：
  `python benchmarks/xval/xval_vivsim_nebb_bodyforce.py --out runs/XVAL-NEBB-BODYFORCE` →
  `python benchmarks/xval/nebb_bodyforce_evidence.py --run-dir runs/XVAL-NEBB-BODYFORCE --strict` →
  `python -m pytest tests/test_nebb_bodyforce.py -q`。

## 生态摩擦（如实记）

1. `pip install jax` 在本环境超时（无可用 wheel/网络）→ 上游二进制无法执行，arm A 退化为
   逐行转写（已在 §①披露并按文件 sha256 留痕）。
2. `knowledge_base` / `oss_solutions` / `negative_knowledge_ledger` 对 agent 角色
   `permission denied`；`/knowledge/search?q=nebb` 返回 0 条（库内仅 1 条 ice 域记录）→
   仓库内无同域可复用能力条目，`cited_kb` 为空。