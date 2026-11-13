# XVAL-LBM-INSTRUCTION-DSL-TESTS-SH — 指令 DSL 与 pylbm 参考实现交叉对标（第 2 轮·调研结合）

仓库：TensorLBM｜分支：agent/m17968｜日期：2026-11-13
判据（先于计算声明）：scheme 语义差 <=1e-12；物理锚点误差 <15%（owner 标准），更严 <3% 同时报告。

## ① 问题设定 + 基准出处

本仓每个算例 = 一个手写求解器（src/tensorlbm 下 900+ 模块），算例不可审阅、声明错误要跑飞才发现。成熟 OSS 做法是
**声明式指令 + 先验证后降阶**。本单把它移植为 TensorLBM 的 instruction DSL，并用 DSL 自己跑解析基准，误差由锚点裁判。

调研·开源横向（只读缓存 /root/oss-cache/pylbm，commit f9f5d16，pylbm 0.12.0；另两仓 Maarten-vd-Sande_lbm、
hschwane/GraSPH 与「指令 DSL」主题无关，未采用），关键做法逐条源码查得（零手抄）：

| 做法 | OSS 出处（file:line） |
|---|---|
| 指令块声明 stencil：dim / scheme_velocity / velocities，构造时先 validate() | pylbm/stencil.py:643、pylbm/validator.py:187 |
| 平衡态以矩空间符号表达式声明（polynomials + equilibrium + relaxation_parameters，s=1/(0.5+σ)） | demo/examples/scheme/eqpde_d2q9_navier_stokes.py、pylbm/scheme.py:238 |
| 力矩阵 M[i,j]=P_i(c_j·LA)、逆阵、f_eq=M^-1·EQ | pylbm/scheme.py:506 _create_moments_matrices |
| 守恒矩 (rho,qx,qy) 排矩空间前三位 | pylbm/scheme.py:261 _permute_consm_in_front |

调研·方法谱系（external_anchor）：Krueger et al. 2017《The Lattice Boltzmann Method》ch.3（D2Q9 权重/声速）、ch.5（MR 碰撞与
ν=(1/s−1/2)/3）；Lallemand & Luo 2000 Phys.Rev.E 61,6546（矩空间碰撞）；Guo/Zheng/Shi 2002 Chin.J.Phys. 40,1（体积力二阶格式）；
Succi 2001（Taylor-Green 衰减 A∝e^{-2νk²t}）；半程反弹壁面 y=1/2（Ziegler 1993）。

为何不重蹈 m17864（llm_error，0 次工具调用）：交付物切成「可机检最小闭环」——解析器 + 不变量自检 + 两解析锚点，每步可运行。

## ② 方法与实现

- src/tensorlbm/instruction_dsl.py：行式文法（stencil/polynomial/equilibrium/relaxation/domain/boundary/init/force/run）
  → parse() → build_scheme()（M、权重、τ、ν + 不变量自检）→ run()（矩空间碰撞 q=M f、守恒矩不松弛、二阶平衡态、
  Strang 半步力、周期核与通道核+半程反弹）。表达式求值走 AST 白名单（+ - * / **，^→**，只暴露声明符号），
  指令文件无法执行任意代码；错误分 DSLSyntaxError / DSLSemanticError 两级。
- tools/xval_pylbm_stub/mpi4py/：单进程 no-op 垫片，仅让 pylbm 离线 import（非 MPI 实现）。
- benchmarks/xval_instruction_dsl/run_xval.py：把同一份指令文本机械翻译成 pylbm dict DSL（速度编号由
  pylbm.stencil.Velocity(dim=2,num=i) 反查匹配，表达式 sympify，唯一人工关系 qx=rho*ux），由 pylbm 重算 M/权重/f_eq 对表；
  再跑 2 个解析锚点 × 2 分辨率。make_figures.py 出图。
- tests/test_instruction_dsl.py：30 用例（文法/不变量/错误分支/降阶语义/物理锚点/pylbm 对表）。

## ③ 参数表（格子单位）

| 量 | 值 |
|---|---|
| c_s²（由权重反算） | 0.333333333333 |
| 松弛率 s（矩 3..8；守恒矩 0） | 1.8 |
| τ=1/s | 0.555556 |
| ν=(τ−1/2)/3 | 0.018518519 |
| 通道力 g | 1e-5 |
| 通道网格 | 8×22、8×32（固体行 y=0,ny−1；壁面 y=1/2, ny−3/2 → H=ny−2） |
| Taylor-Green | 32²、64²；A=0.05，mode=1，k=2π/L |
| Ma_max=u_max/c_s | ≤0.087（守卫阈值 0.3 未触发） |
| 步数 | 通道 12000/30000；TG 400 |

## ④ 对照表（ref vs got，逐项判定）

(A) 与 OSS 参考实现（pylbm 实算）对表：

| 量 | 结果 | 判据 | 判定 |
|---|---|---|---|
| 力矩阵 M (9×9) max abs diff | 0.000e+00 | <=1e-12 | ✅ |
| 静止平衡态=权重 max abs diff | 0.000e+00 | <=1e-12 | ✅ |
| 平衡态布居 f_eq（5 随机宏观态） max abs diff | 1.67e-16 | <=1e-12 | ✅ |

(B) 通道 Poiseuille（解析抛物线 u=g(y−1/2)(H−(y−1/2))/(2ν)）：

| 算例 | 步数 | ν 声明 | ν 拟合 | 剖面最大误差 | 判定 |
|---|---|---|---|---|---|
| poiseuille ny=22 | 12000 | 0.018518519 | 0.018600 | 0.686 % | ✅ <3% |
| poiseuille ny=32 | 30000 | 0.018518519 | 0.018563 | 0.348 % | ✅ <3% |

(C) Taylor-Green 衰减（lnA 线性拟合）：

| 算例 | 步数 | ν 声明 | ν 拟合 | 误差 | R² |
|---|---|---|---|---|---|
| tg 32² | 400 | 0.018518519 | 0.018516 | 0.0132 % | 0.999980 |
| tg 64² | 400 | 0.018518519 | 0.018499 | 0.1036 % | 0.999771 |

壁面位置实测（抛物线拟合）：ny=22 → 0.512 / 20.488（理论 0.5/20.5），ny=32 → 0.508 / 30.492（理论 0.5/30.5）；
拟合高度 19.976 vs H=20、29.984 vs 30 ⇒ 通道几何与 ny−2 一致。注：仓内 bitflag_d2q9.poiseuille_profile 用 H=ny−1，
与其自身 make_channel 几何（流体行 1..ny−2）不一致；本单未改该既有模块，仅记录观察。

## ⑤ 收敛性 / 参数敏感性

- 通道：0.686 %（H=20）→ 0.348 %（H=30），比值 1.97 ≈ (30/20)²=2.25 ⇒ 半程反弹壁面二阶收敛。
- TG：ν 误差 0.0132 %（32²）→ 0.1036 %（64²），两点均 <0.11%。
- 力敏感点（单测）：周期均匀态 g=1e-6，100 步后 u_mean 与 100g 相对偏差 <1e-9 ⇒ 每步力注入恰好一个 g，与 τ 无关。

## ⑥ 失败模式 + 工件

失败模式（已由测试固化）：未知指令/参数/符号、表达式越权（X&Y、__import__）、多项式重复→力矩阵奇异、平衡态不守恒
（2*rho）、松弛率越界（0/2/2.5/−0.5）、边界组合不支持、dim=3 降阶不支持、Ma>0.3。
踩坑：(1) 初版守恒矩松弛 1.0 + Guo 源项 → 力只注入 10.5%（应 100%），改「守恒矩不松弛 + Strang 半步力」后 1e-4/100 步精确；
(2) 初版通道解析式误用 H=ny−1 → 12.7% 假误差，抛物线拟合显示实测壁面 0.51/20.49 → 修 H=ny−2；(3) pylbm 侧
scheme_velocity 传符号 LA 触发 float(expr) 失败，改传数值 1.0。

工件/repro：programs/*.lbm（sha256 见 results.json digest）、results.json、summary.md、
figures/poiseuille_taylor_green.png；复跑 `python benchmarks/xval_instruction_dsl/run_xval.py`（~30 s，全判据 PASS）、
`python -m pytest tests/test_instruction_dsl.py -q`（30 passed，~11 s）。

是否 <3%：是。通道剖面最大误差 0.686% / 0.348%（ν 拟合误差 ≤0.44%）；TG ν 误差 ≤0.1036%；与 pylbm 语义差 ≤1.67e-16（机器精度）。未达标项：无。

---

## 7. L2 / L3 / L4 成熟度证据（返修轮 m18291 追加）

> 本节是 2026-11-13 返修轮（mission 18291）的增量：把此前只在审计工作树里存在的
> 交付物**入仓**，并补齐机器可核的 L2/L3/L4 证据。数值判据与第 1–4 节一致，未改判据。

### 7.1 交付清单（全部在本仓工作树内）

| 角色 | 路径 |
|---|---|
| 实现（非占位，1010 行） | `src/tensorlbm/instruction_dsl.py` |
| 入口（CLI + 程序化） | `src/tensorlbm/instruction_dsl_cli.py` |
| 主运行器：示例运行时调用方（L3） | `src/tensorlbm/example_harness.py::run_instruction_dsl_program` |
| 稳定公共 API 导出（L3） | `src/tensorlbm/api.py`（`run_instruction_program` 等 7 名） |
| 测试（30+2） | `tests/test_instruction_dsl.py` |
| 基准 / 交叉验证驱动 | `benchmarks/xval_instruction_dsl/run_xval.py` |
| 证据装配器 | `benchmarks/xval_instruction_dsl/maturity_evidence.py` |
| 运行工件 | `artifacts/xval_instruction_dsl/{results.json,summary.md,cli_run.json,figures/,programs/}` |
| 机器可核证据 JSON | `artifacts/xval_instruction_dsl/maturity_evidence.json` |

### 7.2 入口与复现命令（repro_cmd）

```bash
# 1) 三条锚点 + pylbm 算子对表（约 35 s，CPU）
python3 benchmarks/xval_instruction_dsl/run_xval.py
# 2) 入口驱动一条 .lbm 指令文件 → JSON 运行记录
PYTHONPATH=src python3 -m tensorlbm.instruction_dsl_cli \
    artifacts/xval_instruction_dsl/programs/poiseuille_ny22.lbm \
    --out artifacts/xval_instruction_dsl/cli_run.json
# 3) 装配 L2/L3/L4 证据（哈希、grep 调用方、基准表）
python3 benchmarks/xval_instruction_dsl/maturity_evidence.py
# 4) 单元/集成测试
python3 -m pytest tests/test_instruction_dsl.py -q
```

### 7.3 L3 集成（不是自引用）

`grep -n instruction_dsl src/tensorlbm/example_harness.py src/tensorlbm/api.py` 命中真实调用方：
`example_harness.run_instruction_dsl_program()` 在示例运行时内调用 `instruction_dsl.parse/run`；
`api.py` 把 `run_instruction_program` / `parse_instruction_program` / `build_instruction_scheme`
列入稳定公共 API 的 `__all__`。对应测试 `test_cli_entry_roundtrip` /
`test_example_harness_bridge_calls_dsl` 端到端跑通。

### 7.4 L4 基准表（ref vs got，程序化从 results.json 抽取）

| 锚点 | ref | got | 误差 | 判据 |
|---|---|---|---|---|
| CH-POISEUILLE-ANALYTIC-DSL (ny=22, H=20) | `u_max=2.69325e-2`（Stokes 抛物线） | `2.67479e-2` | `max_rel=0.686%`，`L2=0.749%` | <15% ✅ / <3% ✅ |
| CH-POISEUILLE-ANALYTIC-DSL (ny=32, H=30) | `u_max=6.06825e-2` | `6.04711e-2` | `max_rel=0.348%`，`L2=0.375%` | <15% ✅ / <3% ✅ |
| TAYLOR-GREEN-DECAY-DSL (32²) | `nu=1.85185e-2`（声明 τ 反推） | `nu_eff=1.85161e-2` | `0.0132%`（R²=0.99998） | <3% ✅ |
| TAYLOR-GREEN-DECAY-DSL (64²) | `nu=1.85185e-2` | `nu_eff=1.85004e-2` | `0.1036%` | <3% ✅ |
| PYLBM-SCHEME-OPERATOR-PARITY | pylbm 0.12.0 `M`/`w`/`f_eq` | `0.0 / 0.0 / 1.11e-16` | ≤1e-12 ✅（仅算子层、单进程） | ✅ |

拟合口径已写入 `results.json`（新增 `sample_every` 字段：TG 为 `steps//200`，
通道为 `steps//24`），修复 m18092 审计 F2「拟合窗口不可独立复现」。

### 7.5 注意点与已知限制（honest）

- **收敛标志**：`cli_run.json` 中 `converged=false` 是 `run tol=1e-10` 在 12000 步内
  未触发逐场变化判据，而通道剖面已达稳态（`max_rel=0.686%`）；结论以剖面误差为准，
  不以该布尔值冒充收敛证明。
- **收敛阶**：两分辨率均 <3%，但观测阶约 1.67（m18092 审计重算），**不宣称二阶**；
  第三分辨率（H=45/N=128²）尚未交付。
- **pylbm 对表范围**：仅算子层（M/权重/f_eq），单进程；非整解场 parity。
- 通道壁面位置由数值剖面抛物线拟合反求：ny=22 → 0.512/20.488，ny=32 → 0.508/30.492
  （理论 0.5 / ny−1.5），几何判据被测量钉住。
