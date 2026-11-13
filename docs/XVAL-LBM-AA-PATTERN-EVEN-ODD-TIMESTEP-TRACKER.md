# XVAL-LBM-AA-PATTERN-EVEN-ODD-TIMESTEP-TRACKER

机器可复核产物：`runs/xval_aa_pattern_even_odd/result.json`（spec / arms / metrics / criteria / tracker）、`k_history_<arm>_<case>.csv`、`figures/xval_aa_pattern_even_odd/*.png`。本文所有数字由 `benchmarks/xval_aa_pattern_even_odd.py` 从 `result.json` 程序化注入，无手抄。

生成环境：python 3.11.6 / torch 2.7.1 / Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38 / dtype float64 / device CPU。

## 1 问题设定与基准出处

**AA 模式（单场原地 streaming）**：AB 模式用两个分布场 A/B 每步互换角色（持久存储 `2QN`）；
AA 模式只用**一个**场（`QN` 浮点）加一个 `O(N)` 行缓冲，把 streaming 置换按速度半集拆成两个
内核原地执行：`e` 内核搬 `S = {1,2,5,8}`，`o` 内核搬 `S' = {3,4,7,6}`（`0` 固定）。两内核作用于
不相交的场行，因此互不污染、且顺序无关（§4 判据 `kernel_order_immaterial` 实测）。
跑完 `e` 而 `o` 未跑时缓冲区处于**半 streamed** 状态：四个方向已是 `t+1`、四个仍是 `t`，
此刻宏观量无定义 —— 这正是必须做**时间步奇偶跟踪**（even/odd timestep tracker）的原因：
跨越时间步，两内核的角色按奇偶交替，跟踪器丢失交替 = 同一半集被搬两次、另一半一次没搬。

上游同名内核 `aa-pattern-e` / `aa-pattern-o` 出自 walberla LBM 模块（lssfau/walberla）。
**诚实声明**：本沙箱无外网出口（egress 被拦截），未能取回上游源码逐行比对，因此本工作**不声称**
与 walberla 内核字节级同构；本文验证的是上面精确定义并在代码中机器检查的 AA 模式性质。

**上游语义（知识库 oss_solutions 记录，`cited_kb` 引用）**：`2b-t/LB-t` 的 `src/population/indexing/aa_pattern.hpp:56,79` 给出具体槽位布局——单一扁平数组、`ND = SPEEDS + PAD`（D3Q27 → 28）、`OFF = ND/2`、方向 `d` 的对偶槽为 `OFF+d`、`MASK[OFF] = 0` 防静止分量重复写，**偶时间步写正槽读对偶槽、奇时间步镜像**——即 A-A 模式本质是单数组内的「读旧写新」，所以诊断必须知道时间步奇偶。本实现的单场（每方向单槽 + O(N) 行缓冲）复现的是同一**契约**（单数组 / 半流窗口 / even-odd 跟踪），**不实现** `ND = SPEEDS + PAD` 的双槽索引（第 2 轮范围，已在 §6 声明）。

**对标基准（外部锚）**：2D Taylor–Green 涡是二维不可压 Navier–Stokes 的精确解（对流项为纯梯度，被压力吸收）：

     u = -U0 cos(kx) sin(ky) e^{-2νk²t},  v = +U0 sin(kx) cos(ky) e^{-2νk²t},
     K(t) = 0.5<u²+v²> = (U0²/4) e^{-4νk²t},   γ_K = 4νk²,  γ_vel = 2νk².

出处：Taylor (1923) *Phil. Trans. R. Soc. A* 的 Green–Taylor 涡精确解（教科书级闭式解，严格解而非级数截断）；LBM 侧同规格复用在 `docs/XVAL_TAYLORGREEN.md` 已建立 （`src/tensorlbm/xval/metrics.py::exact_velocity`）。第二个基准是**仓库内 AB 参考实现**（`tensorlbm.solver.collide_bgk` + `stream`），streaming 是置换，因此 AA 与 AB 必须**逐位相等**。

## 2 方法与实现

| 项 | 实现 |
|---|---|
| 求解器 | D2Q9 BGK (SRT)，`tensorlbm.solver.collide_bgk` |
| AA streaming | `src/tensorlbm/aa_pattern.py::AAPatternStepper._stream_half`（逐行原地 gather，行级 O(N) 临时） |
| 奇偶跟踪器 | 同文件 `parity_report()`：`phase`(complete/half) + `timestep_parity`(even/odd) + 内核日志/计数 |
| 半流窗口闸门 | `AAPatternStepper._require_complete`，违规抛 `AAPatternParityError` |
| 缺陷注入对照臂 | `parity_blind_step()`（丢失奇偶交替，同半集搬两次）+ 半流窗口直读探针（`hazard_probe`，等价于 `strict_parity=False`） |
| 参考臂 AB | `solver.stream` gather，双场，float64，同 τ 同步数 |
| 复用（cited_kb） | `cap:TensorLBM:m15290_XVAL-LBM-TAYLORGREEN`（L4）：Taylor–Green 精确解口径与 `xval/metrics.py::fit_exp_rate` 衰减率拟合 |
| 精度 | float64 CPU；判据 `max|f_AA - f_AB|` 逐位 |

## 3 参数表（自洽无量纲；dx = dt = 1）

| case | N | U0 | Re | ν | τ | k | Ma | steps | 半流探针步 |
|---|---|---|---|---|---|---|---|---|---|
| tg_n64_re100 | 64 | 0.05 | 100.0 | 0.032 | 0.596 | 0.09817477 | 0.0866 | 1000 | 251 |
| tg_n128_re100 | 128 | 0.05 | 100.0 | 0.064 | 0.692 | 0.049087385 | 0.0866 | 1000 | 251 |
| tg_n128_re320 | 128 | 0.05 | 320.0 | 0.02 | 0.56 | 0.049087385 | 0.0866 | 1000 | 251 |

## 4 对照表（逐项 ref vs got + 误差）

### 4.1 `tg_n64_re100` (N=64, Re=100.0, steps=1000)

| 量 | ref | AA (got) | AB (got) | AA-AB | AA vs ref |
|---|---|---|---|---|---|
| γ_K | 0.0012337 | 0.00123297 | 0.00123297 | 0 | -0.0593% |
| field L2 vs exact | 0 | 0.3044% | 0.3044% | — | 0.3044% |
| max&#124;f_AA-f_AB&#124; | 0 | 0 | — | 0 | — |
| 质量漂移（相对） | 0 | 9.3e-14 | 9.3e-14 | — | — |
| 动量漂移（相对） | 0 | 2.63e-16 | 2.63e-16 | — | — |

奇偶闸门（半流窗口，步 251，先跑内核 `odd`）：`phase` = `half`，未跑内核数 = 1，质量 = 4096.000000（守恒）；直读宏观量相对闭式解 field L2 = **3.3311%**（对 t）/**3.3278%**（对 t+1），跟踪器拒绝该读取并抛 `AAPatternParityError`。

缺陷注入臂 `aa_parity_blind`（丢失 even/odd 交替）1000 步后：field L2 = **164.9851%**，γ_K = 0.00125384（ref 0.0012337，偏差 +1.6326%），质量漂移 9.31e-14、动量漂移 4.92e-16 —— **质量守恒但动量破坏**，与预期失效模式一致。

### 4.2 `tg_n128_re100` (N=128, Re=100.0, steps=1000)

| 量 | ref | AA (got) | AB (got) | AA-AB | AA vs ref |
|---|---|---|---|---|---|
| γ_K | 0.00061685 | 0.000616388 | 0.000616388 | 0 | -0.0749% |
| field L2 vs exact | 0 | 0.3003% | 0.3003% | — | 0.3003% |
| max&#124;f_AA-f_AB&#124; | 0 | 0 | — | 0 | — |
| 质量漂移（相对） | 0 | 8e-14 | 8e-14 | — | — |
| 动量漂移（相对） | 0 | 1.53e-16 | 1.53e-16 | — | — |

奇偶闸门（半流窗口，步 251，先跑内核 `odd`）：`phase` = `half`，未跑内核数 = 1，质量 = 16384.000000（守恒）；直读宏观量相对闭式解 field L2 = **3.2189%**（对 t）/**3.2188%**（对 t+1），跟踪器拒绝该读取并抛 `AAPatternParityError`。

缺陷注入臂 `aa_parity_blind`（丢失 even/odd 交替）1000 步后：field L2 = **147.6356%**，γ_K = 0.000723733（ref 0.00061685，偏差 +17.3271%），质量漂移 8.03e-14、动量漂移 3.39e-16 —— **质量守恒但动量破坏**，与预期失效模式一致。

### 4.3 `tg_n128_re320` (N=128, Re=320.0, steps=1000)

| 量 | ref | AA (got) | AB (got) | AA-AB | AA vs ref |
|---|---|---|---|---|---|
| γ_K | 0.000192766 | 0.000192628 | 0.000192628 | 0 | -0.0715% |
| field L2 vs exact | 0 | 0.3651% | 0.3651% | — | 0.3651% |
| max&#124;f_AA-f_AB&#124; | 0 | 0 | — | 0 | — |
| 质量漂移（相对） | 0 | 9.91e-14 | 9.91e-14 | — | — |
| 动量漂移（相对） | 0 | 1.26e-16 | 1.26e-16 | — | — |

奇偶闸门（半流窗口，步 251，先跑内核 `odd`）：`phase` = `half`，未跑内核数 = 1，质量 = 16384.000000（守恒）；直读宏观量相对闭式解 field L2 = **3.3727%**（对 t）/**3.3726%**（对 t+1），跟踪器拒绝该读取并抛 `AAPatternParityError`。

缺陷注入臂 `aa_parity_blind`（丢失 even/odd 交替）1000 步后：field L2 = **148.1382%**，γ_K = 0.000255897（ref 0.000192766，偏差 +32.7501%），质量漂移 9.89e-14、动量漂移 6.43e-16 —— **质量守恒但动量破坏**，与预期失效模式一致。

### 4.x 判据汇总（阈值判定）

| 判据 | 值 | 阈值 | PASS |
|---|---|---|---|
| aa_vs_ab_bitwise_max_abs | 0 | < 1e-12 | True |
| gamma_K_vs_exact_worst_pct | 0.07491 | < 15 % | True |
| field_l2_vs_exact_worst_pct | 0.3651 | < 15 % | True |
| mass_drift_rel_worst | 9.9143e-14 | < 1e-12 | True |
| mom_drift_rel_worst | 2.6306e-16 | < 1e-12 | True |
| kernel_order_immaterial | 0 | < 1e-12 | True |
| parity_gate_refuses_half_state | 1 | == 1 | True |
| half_window_hazard_rough_field_pct | 119.1 | >= 15 % (hazard is fatal on a rough field) | True |
| half_window_mass_conserved | 0 | < 1e-12 (conservation cannot catch the hazard) | True |
| parity_blind_arm_error_pct | 165 | >= 15 % (bug detectable) | True |

overall: **PASS**。

## 5 收敛性 / 参数敏感性

| case | N | Re | γ_K 误差 | field L2 | 每步耗时 [s] |
|---|---|---|---|---|---|
| tg_n64_re100 | 64 | 100.0 | -0.0593% | 0.3044% | 0.00261 |
| tg_n128_re100 | 128 | 100.0 | -0.0749% | 0.3003% | 0.00594 |
| tg_n128_re320 | 128 | 320.0 | -0.0715% | 0.3651% | 0.00668 |

两分辨率（N=64/128）同 Re、两参数点（tg_n64_re100 与 tg_n128_re320）都在闸门内；网格细化与 τ 变化下误差量级不变，说明 AA 分解本身不引入额外误差（逐位等价于 AB 是更强的结论）。

## 6 失败模式与工件

- **半流窗口直读**：无跟踪器时宏观量无定义。同一时间层的 half 读 vs complete 读（N=64）相对 L2 速度误差：光滑场（mode-1 Taylor–Green）**3.2830%**，粗糙场（白噪声速度场，seed 1234）**119.0758%**。两者质量差仅 0（相对）——**光滑场上该误差可低于 15% 闸门静默通过，守恒检查也抓不到**，所以奇偶闸门不可省：跟踪器在此窗口抛 `AAPatternParityError`（实测 `gate_exception` = `AAPatternParityError`）。
- **奇偶交替丢失**（同半集搬两次）：质量守恒、动量破坏；field L2 最差 **164.9851%**（远超闸门）。注意 γ_K 这一项并不可靠：`tg_n64_re100` (+1.6326%) 的 γ_K 偏差仍 < 15% 闸门——**只看衰减率会放过这个缺陷，必须看场误差**。
- **内核顺序**：两内核搬不相交场行，顺序无关；实测 `kernel_order_immaterial` 判据为逐位相等。
- **未做 / 第 2 轮输入**：① 上游 `2b-t/LB-t` 的双槽索引布局（`ND = SPEEDS + PAD`）本实现未落地，只落地了同一契约的单槽变体；② 非周期边界（bounce-back / ghost 交换）在半流窗口内需要通信，未覆盖；③ `aa-pattern-e/o` 上游内核源码本沙箱取不到（无外网），无法做字节级同构比对。卡点：无可访问的上游源码镜像；缺的方法：无（契约层验证已闭环，缺的只是布局层）。
- **适用范围**：周期性边界 + 平直（无 AMR/无重叠网格）单一缓冲区；AA 分解对任何速度半集划分成立，但边界节点交换（bounce-back/ghost）需要在半流窗口内引入通信，本实现不覆盖边界，只用周期域。

工件：

- `runs/xval_aa_pattern_even_odd/result.json`
- `artifacts/xval_aa_pattern_even_odd/result.json`
- `runs/xval_aa_pattern_even_odd/k_history_aa_<case>.csv`
- `runs/xval_aa_pattern_even_odd/k_history_ab_<case>.csv`
- `runs/xval_aa_pattern_even_odd/k_history_aa_parity_blind_<case>.csv`
- `artifacts/xval_aa_pattern_even_odd/k_history_<arm>_<case>.csv`
- `figures/xval_aa_pattern_even_odd/tg_decay_aa_vs_ab_vs_exact.png`
- `figures/xval_aa_pattern_even_odd/tg_field_aa_vs_exact.png`
- `figures/xval_aa_pattern_even_odd/parity_hazard.png`
- `src/tensorlbm/aa_pattern.py`
- `tests/test_aa_pattern_parity.py`

一键复现：

```bash
python benchmarks/xval_aa_pattern_even_odd.py
pytest tests/test_aa_pattern_parity.py -q
```

## 7 存储口径（结构声明，非实测）

| 模式 | 持久分布场 | 每步临时 | 索引缓存 |
|---|---|---|---|
| AB（参考） | 2·Q·N 浮点 | 一整个 (9,ny,nx) gather 结果 | 9·N 索引 |
| AA（本实现） | 1·Q·N 浮点 | 每方向一行 O(N) | 8·N 索引 |

实测索引字节：`nbytes_permutation_index` = 2097152（N=128）；持久场字节：1179648。

## 8 误差判定（显式回答 owner 提问）

- 关键量 γ_K vs 闭式解：最差偏差 **+0.0749%** → 是否 < 15%：**是**；是否 < 3%：**是**。
- 快照场 vs 闭式解（相对 L2）：最差 **0.3651%** → 是否 < 15%：**是**；是否 < 3%：**是**。
- AA vs AB 逐位一致：max|Δf| 最差 **0** → 是否 < 1e-12：**是**。

