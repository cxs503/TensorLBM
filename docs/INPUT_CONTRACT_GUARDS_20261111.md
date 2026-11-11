# TensorLBM 核心输入契约防线（审计 D5/D6/D7/D9 销项）

**日期**：2026-11-11 · **仓**：TensorLBM · **接续**：`docs/AUDIT_RUNTEST_6_TENSORLBM.md`（m14496，2026-11-11 审计，修复仅 D1/D2/D3）

## 1. 案例介绍（问题背景）

m14496 的抽样审计把「同一非法输入，看路径决定是否报错」记为缺陷（D5/D6/D7/D9），但该单只修了 Poiseuille 壁面臂（D1/D2/D3）。本单销项其**剩余四条**：非法标量进入时间循环后不会立刻失败，而是在几十步之后以场量 `nan` 的形式爆出来——诊断成本远高于入口处一个 `ValueError`。

| 缺陷 | 入口 | 旧行为（实测） |
|---|---|---|
| D7 | `solver.collide_bgk` / `collide_bgk_fused` | `tau=0.5` → 静默（零黏性）；`tau=0.0` → `inf`/`nan` 场；`tau=nan` → 全 `nan` 场。同仓 `collide_advanced_3d` 对 `tau=0.5` 已抛 `ValueError`（`tests/test_collision_auto_discovery.py::test_tau_must_exceed_half`） |
| D6 | `zou_he_outlet_pressure` / `_3d` / `_z` / `_27` | `rho_out=0` → 出口整列 `nan`（除以 `rho_out`），无诊断 |
| D5 | `GeneralSimConfig` / `GeneralSimEngine` | `viscosity=0`、`resolution=0`、`max_steps=-5` 全部静默接受；`general_sim.py` 全文 0 个 `raise ValueError` |
| D9 | `benchmarks/verified/poiseuille_2d/run.py` | `H=0` → `delta_rho = 24*nu*u_max/(cs2*H)` 抛裸 `ZeroDivisionError` |

## 2. 怎么做的（方法/关键设置）

1. 新增共享守卫模块 `src/tensorlbm/input_contract.py`（3 个函数，纯 Python 标量比较、不触碰张量，故可安全用于 `torch.compile` 追踪与逐步循环）：
   - `check_relaxation_time(tau)`：非有限值或 `tau<=0.5` → `ValueError`；**张量 `tau`（逐格 Smagorinsky 等）原样放行**，由各自路径校验（不在热路径上做隐式 device sync 或断梯度）。
   - `check_positive_scalar(value, name, allow_zero=False)`：`nan/inf` 必拒，`<=0` 拒（`allow_zero` 用于来流速度这类合法零值）。
   - `check_positive_int(value, name, min_value)`：布尔/字符串/非整浮点拒收（`True` 不算 1）。
2. 接线：`solver.collide_bgk`、`collide_bgk_fused`；四个 Zou-He 压力出口；`GeneralSimConfig.validate()`（新增，返回 `self`，并在 `GeneralSimEngine.__init__` 首行强制调用，`resolution>=4`、`max_steps>=0`、`viscosity>0`、`reference_length>0`、`inlet_velocity>=0`）。
3. 基准 CLI：`poiseuille_2d.run_case` 入口显式校验 `H` 为正整数；`poiseuille_3d_pipe --device` 默认从 `cuda:2` 改为 `auto`（有 GPU→`cuda:0`，否则 `cpu`），消除 CPU-only 宿主上的 `AssertionError`（审计 D4）。
4. 证据脚本：`scripts/contract_guards/negative_probe.py`（22 个负向探针，同一脚本在改动前后各跑一次，输出 `artifacts/contract_guards/probe_before.json` / `probe_after.json`）。改动前的基线是在**未修改的工作树上**先跑的，因此前后对照是同一探针、同一环境。

## 3. 计算结果（程序化注入，零手抄）

探针判定分两类：`raised`（入口拦截，合格）与 `silent`（静默接受，缺陷）。`baseline_ok` 一项是"默认配置应通过"的正向探针，`silent` 即合格。

| 探针组 | 改动前 silent | 改动后 silent |
|---|---|---|
| `collide_bgk` × {0.5, 0.0, nan, −0.7} | 4/4 | 0/4 |
| `collide_bgk_fused` × 同上 | 3/4（`tau=0` 为裸 `ZeroDivisionError`，非域校验） | 0/4 |
| Zou-He 出口 × 3 个格构 / `rho_out∈{0,−1,nan}` | 7/7 | 0/7 |
| `GeneralSimConfig` 非法配置 | 5/5 | 0/5 |
| `poiseuille_2d H=0` | 抛裸 `ZeroDivisionError` | 抛 `ValueError: poiseuille_2d: H=0 must be a positive integer...` |
| **合计** | **20 / 22 silent** | **1 / 22 silent（仅正向探针 `baseline_ok`）** |

回归测试：`tests/test_input_contract_guards.py` **29 passed**（13.2 s）。每条非法输入断言*具体* `ValueError` 且消息含参数名（不是"抛了就算"），并配正向断言（质量守恒 `sum` 守恒 rel 1e-6、场有限）以防「一律拒绝」式的假绿。
旁证抽样：`tests/test_general_sim_d3q27.py`、`test_heat_convection_benchmarks.py`、`test_collision_viscosity_audit.py`、`test_cylinder_cd.py` 与 `test_powerlaw.py`/`test_pancake_ice.py`（`tau=3ν+0.5` 形式）→ 前四文件 36 passed / **3 failed**，后两文件 6 passed。

## 4. 结果图

无图：本单是后端输入契约改造，产物是判定矩阵（上表）+ JSON 探针记录，不存在场量云图或与解析解的误差曲线；强行作图只会是表格的重绘。全部数字由 `scripts/contract_guards/negative_probe.py` 与 pytest 直接产生。

## 5. 注意点（坑与局限）

- **3 条失败是既有的，不是本单引入**：`tests/test_general_sim_d3q27.py` 三条以 `KeyError: 'q'` 失败（第 161 行读返回 dict 的 `q` 键），位置在我未触碰的返回值组装处，且我的守卫只会抛 `ValueError`、不会改变 dict 键集合。**未在干净 HEAD 上复跑确认**（本机 load15≈70–98，按容量纪律未起全量基线），故按 `not_run` 记账，不写成 passed。
- **`tau` 张量刻意放行**：逐格弛豫时间（Smagorinsky / 壁面函数）走各自校验；若在此处 `float()` 强转，会断梯度并逐步 device sync。
- **`tau=0.5` 由「允许」改为「拒绝」**：与仓内 `collide_advanced_3d`、`CollisionViscosityAuditConfig.validate`、Couette tutorial 的既有契约一致。物理上 `ν=(τ−0.5)/3=0`，属退化参数；确有近无黏需求时用 `0.5+1e-9`（测试已覆盖该边界仍可用）。
- `resolution>=4` 的下界是保守值（`nx` 至少要能容纳一道入流+一道出流+流体格）；比"必须能跑通"宽松，故不排除极小域仍需另行校验。
- 未跑全量测试套件（m14496 记录全量两次均未跑完），本单只做定向抽样。

## 6. 误差判定

本单无物理基准量（不是新算例），故不适用「关键量 vs 基准 <3%」。可判定的判据是**契约判定率**：非法输入拦截率 **20/22 → 21/22**（唯一 `silent` 为正向探针，期望值即不抛），且 29 条契约测试全绿；正向不变量（质量守恒 rel<1e-6、场量有限）无一并退化。**显式回答：无基准误差可比，因为交付物是入口判据而非数值预报；若需物理精度结论，见 `docs/AUDIT_RUNTEST_6_TENSORLBM.md` §5 与后续收敛/基准单。**
