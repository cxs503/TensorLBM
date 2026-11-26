# XVAL-LBM-INSTRUCTION-DSL-TESTS-SH（第 2 轮）— 指令 DSL 的 shell 驱动一致性测试套件（TCLB `instruction-dsl-tests-sh` 做法）

> [Current archive and reproduction instructions](../development/verification/instruction_dsl/README.md). The historical report retains its original delivery paths.

仓库 TensorLBM｜分支 agent/m20996｜日期 2026-11-14｜锚点 `XVAL-LBM-INSTRUCTION-DSL-TESTS-SH`（第 1 轮 m17968 交付了解析器+pytest，
本单补齐「测试由 shell 驱动」这一半）。

## ① 案例介绍（物理背景 / 几何 / 参数）

指令 DSL 把一次 LBM 运行写成可审阅的文本（stencil / polynomial / equilibrium / relaxation / domain /
boundary / init / force / run），先做格点与物理不变量自检，再降阶为可执行计划。第 1 轮只有 pytest 用例；
TCLB 的做法不同：**测试本身是 shell 脚本，驱动可执行入口、只看进程状态与两条流**，因此不依赖任何 Python
测试框架，任何装了包的 CI 镜像都能跑，失败以退出码呈现。本单把该做法落地为 17 个用例
（5 个「必须跑对」+ 12 个「必须按特定诊断拒绝」）。

物理锚点（格子单位）：

| 量 | 值 |
|---|---|
| 松驰率 s（矩 3..8；守恒矩 0） | 1.8 |
| τ = 1/s | 0.555555556 |
| ν = (τ−1/2)/3 | 0.018518518519 |
| c_s² | 0.333333333333 |
| 通道 | nx=8, ny=22, 壁面 y=1/2 与 ny−3/2 → H=20；g=1e-05；12000 步 |
| Taylor-Green | 32², A=0.05, mode=1, k=2π/L, 200 步 |

## ② 怎么做的（方法 / 流程 / 工具链）

1. `make_cases.py` 生成用例：期望值全部由**声明量**推导（ν=(1/s−1/2)/3、k=2π/nx、H=ny−2、力每一步加一个
   动量单位），没有任何参考值手抄；同时把每个 `program.lbm` 的 SHA-256 钉成回归基准（文本一改就红）。
2. `run_cases.sh`（POSIX sh + awk，无 Python 测试框架）逐用例执行
   `python3 -m tensorlbm.instruction_dsl_cli <program> --kv`，判定：
   退出码、`ERROR <Class>: <msg>` 的错误类与诊断子串、字段 `eq.`（字符串相等）/`approx.`（相对容差）/
   `abs.`（绝对容差），逐用例留 `artifacts/logs/<case>.out|.err` 原始流，汇总 `results.tsv`/`results.json`。
3. CLI 增加**退出码契约**（0 成功 / 2 语法错 / 3 语义错 / 4 用法错），拒绝时只打一行诊断、不打 traceback
   —— 这是 shell 测试能断言「错误类」而不是「反正失败了」的前提。
4. 顺带修掉一个真实缺陷：标量指令（domain/run/init/relaxation/force/boundary/stencil）重复声明时原本**静默后者覆盖前者**，
   复制粘贴或改错一处就能悄悄换掉算例。现在报 `duplicate instruction '<name>' (first declared at line N)`（用例 `i11`）。
5. `tests/test_instruction_dsl_shell_suite.py` 把 shell 套件接进 pytest，并对套件本身做**变异测试**：
   改错一个解析参考、写错诊断串、写错退出码、改动被钉的指令文本、删掉 `exit=` 行、空用例目录 —— 六种破坏都必须让
   runner 变红；否则「测试通过」没有意义。

## ③ 计算结果

套件判定（程序化写出，零手抄）：**17/17 通过，失败 0**。

| case | verdict | expected / measured |
|---|---|---|
| `i01_unknown_instruction` | PASS | i01_unknown_instruction: a word that is not an instruction must name the offender and the vocabulary |
| `i02_unknown_argument` | PASS | i02_unknown_argument: an argument outside the instruction's whitelist must list what is allowed |
| `i03_expression_not_executable` | PASS | i03_expression_not_executable: instruction expressions are sandboxed: attribute access is not lowered |
| `i04_relaxation_out_of_range` | PASS | i04_relaxation_out_of_range: s=2.5 would give a negative viscosity; the invariant guard must refuse it |
| `i05_singular_moment_matrix` | PASS | i05_singular_moment_matrix: duplicated moment polynomial makes M singular -> no invertible lowering |
| `i06_equilibrium_loses_mass` | PASS | i06_equilibrium_loses_mass: an equilibrium that does not recover rho is not an equilibrium |
| `i07_unsupported_boundary_mix` | PASS | i07_unsupported_boundary_mix: a boundary layout the 2-D lowering cannot honour is refused, not approximated |
| `i08_three_dimensional_stencil` | PASS | i08_three_dimensional_stencil: dim=3 stencil must be refused by the 2-D lowering instead of silently run |
| `i09_mach_guard` | PASS | i09_mach_guard: u=0.5 lattice units (Ma~0.87) is beyond the incompressible guard |
| `i10_missing_run_instruction` | PASS | i10_missing_run_instruction: a program without run control cannot be executed |
| `i11_duplicate_instruction` | PASS | i11_duplicate_instruction: a second 'domain' line silently overriding the first is a defect, not a shorthand |
| `i12_steps_below_one` | PASS | i12_steps_below_one: run steps=0 would produce a record that claims a run happened |
| `v01_static_uniform` | PASS | v01_static_uniform: uniform rho=1 at rest, 10 steps: no spurious velocity or kinetic energy |
| `v02_force_unit_momentum` | PASS | v02_force_unit_momentum: body force gx=1e-6 for 100 steps: u_max = n*g exactly (momentum invariant) |
| `v03_tg_decay_32` | PASS | v03_tg_decay_32: Taylor-Green 32^2 A=0.05, 200 steps: u_max and KE against exp(-2nu k^2 t) decay |
| `v04_channel_poiseuille` | PASS | v04_channel_poiseuille: channel 8x22 walls y=1/2,ny-3/2 (H=ny-2), g=1e-5, 12000 steps: u_max = gH^2/(8nu) |
| `v05_lattice_invariants` | PASS | v05_lattice_invariants: scheme self-checks surfaced through the CLI: cs2=1/3, tau=1/s, nu=(tau-1/2)/3 |

物理锚点数值：

| 算例 | 量 | 数值解 | 解析/声明值 | 相对误差 | 判据 |
|---|---|---|---|---|---|
| 通道 Poiseuille | u_max | 0.026747854 | 0.027000000 = gH²/(8ν) | 0.934 % | <3 % → YES |
| 通道 Poiseuille | 剖面最大偏差 | — | 解析抛物线 | 0.686 % | <3 % → YES |
| Taylor-Green | ν 拟合 | 0.018549208 | 0.018518519（声明） | 0.166 % | <3 % → YES |
| Taylor-Green | KE(t) | 见曲线 | 0.25A²e^{−4νk²t} | 1.612 % | <3 % → YES |

逐条断言与容差余量（红线=允许容差）：

| claim | measured error | acceptance band | margin (decades) | <3 %? |
|---|---|---|---|---|
| `v01_static_uniform:u_max` | 0 (bit-exact) | 1.0e-15 (abs) | inf | n/a (exact) |
| `v01_static_uniform:kinetic_energy` | 0 (bit-exact) | 1.0e-30 (abs) | inf | n/a (exact) |
| `v02_force_unit_momentum:u_max` | 4.136e-12 | 1.0e-08 (approx) | 3.4 | YES |
| `v03_tg_decay_32:u_max` | 7.005e-03 | 3.0e-02 (approx) | 0.6 | YES |
| `v03_tg_decay_32:kinetic_energy` | 1.276e-02 | 3.0e-02 (approx) | 0.4 | YES |
| `v03_tg_decay_32:nu` | 0 (bit-exact) | 1.0e-10 (approx) | inf | YES |
| `v04_channel_poiseuille:u_max` | 9.339e-03 | 3.0e-02 (approx) | 0.5 | YES |
| `v04_channel_poiseuille:nu` | 0 (bit-exact) | 1.0e-10 (approx) | inf | YES |
| `v05_lattice_invariants:cs2` | 1.665e-16 | 1.0e-10 (approx) | 5.8 | YES |
| `v05_lattice_invariants:tau` | 0 (bit-exact) | 1.0e-10 (approx) | inf | YES |
| `v05_lattice_invariants:nu` | 0 (bit-exact) | 1.0e-10 (approx) | inf | YES |

图：`fig_poiseuille.png`（剖面 vs 解析抛物线）、`fig_tg_decay.png`（KE 衰减 + ln KE 线性拟合）、
`fig_case_margins.png`（每条数值断言的误差 vs 容差）。

## ④ 注意点（坑 / 局限 / 适用范围）

* 通道算例 `converged=False`：run 的 `tol` 收敛判据未触发（未声明 tol），剖面误差靠解析解判定，
  不靠自报收敛 —— 别把「跑完」当「收敛」。
* 用例只覆盖 DSL 目前降阶支持的两种布局（全周期、y 向半程反弹通道）与两种初场（uniform / taylor_green）；
  其他布局按设计是**拒绝**（用例 `i07`/`i08`），不是不支持。
* 钉 SHA-256 的用例会因**空白改动**变红：这是刻意的溯源设计，改文本就用 `make_cases.py` 重钉。
* 套件默认跑一遍约 12 s（其中 12000 步通道 ~8 s）；重算全量不要压控制面。

## ⑤ 误差判定（逐项回答「是否 <3%」）

* 通道 u_max：0.934 % → **<3 % YES**（也远小于 15 % 的 owner 门槛）。
* 通道剖面最大偏差：0.686 % → **<3 % YES**。
* Taylor-Green ν 拟合：0.166 %（R²=0.999858）→ **<3 % YES**。
* Taylor-Green KE 全程最大偏差：1.612 % → **<3 % YES**。
* 精确类断言（静止不动、力每步一个动量单位、cs2/τ/ν 复算）误差为 0 / ≤1.3e-02，无基准缺失问题。
* 未做的检查：未做 MPI/多进程、3D、GPU 路径（DSL 不降阶这些）；未对 pylbm 重跑对表（第 1 轮已做，见
  `docs/XVAL-LBM-INSTRUCTION-DSL-TESTS-SH.md`）。

## ⑥ 复现

```sh
cd xval/XVAL-LBM-INSTRUCTION-DSL-TESTS-SH/dsl_shell_suite
PYTHONPATH=../../../src python3 make_cases.py      # 重钉期望（改行为时才需要）
./run_cases.sh                                     # shell 套件本体，退出码即结论
PYTHONPATH=../../../src python3 make_figures.py    # 图 + figures_data.json
python3 make_report.py --post                      # 本文档 + 平台报告
cd ../../.. && python3 -m pytest tests/test_instruction_dsl_shell_suite.py tests/test_instruction_dsl.py -q
```
