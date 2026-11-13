# m17139 · TensorLBM 文档—代码一致性审计（README / 示例 / API 抽查 18 项）

**任务**：生态巡更 · TensorLBM · 文档一致性 —— README/示例与实际代码 API 抽查对照 ≥10 项，
列出不一致处并修复最关键 3 处；示例可运行性验证。
**分支**：`agent/m17139`　**判据**：机器可裁（pytest 门禁 + 可溯源数字 + 实跑产物）

---

## ① 案例介绍（审计对象与参数表）

本单不涉及新的物理算例，而是审计「**文档声明的可执行契约**」与「**代码/工件真实状态**」是否一致。
被审对象与规模（全部由 `artifacts/m17139/audit_probe.py` 程序化统计，非手抄）：

| 维度 | 取值 |
|---|---|
| 带可执行命令的文档 | `README.md`、`README.zh-CN.md`、`docs/software_manual.md`、`docs/suboff_platform_manual.md` |
| 文档内 `PYTHONPATH=src python …` 命令块 | 40 条（覆盖 17 个不同脚本/基准） |
| README 本地相对链接 | README 22 条 + 中文 11 条 |
| CLI 旗标 / 带 `choices` 的取值 | 123 个旗标 / 12 个枚举取值（`argparse` AST 静态解析） |
| 能力符号抽样 | 10 个（RANS k-ε/k-ω SST、湍流统计、MultiGPU 2D/3D、DFSEM、Digital Filter、Q 准则、λ₂ 准则、FWH-SPL） |
| 实跑复现命令 | 4 条（2D 圆柱 smoke / quick-start、3D 球体 smoke、批量扫描）＋1 条演示 GIF（缩参） |
| 审计项总数 | **18** 项（12 PASS、5 不一致已修、1 注意点） |

审计判据（先于动手声明）：
1. 文档中出现的相对路径/脚本/链接必须在树内**存在**；
2. 文档命令里的每个 `--flag` 必须被对应脚本的 `argparse` **接受**，带 `choices` 的取值必须**合法**；
3. 文档引用的能力必须以**真实符号**（可 `import`/`hasattr`）兑现；
4. 定量表中的数字必须能在仓内受审证据文档中**逐一溯源**（零手抄）；
5. 文档给出的命令必须**实际跑通**并产出文档声明的文件布局。

## ② 怎么做的（方法 / 流程 / 工具链）

1. **任务池查重**：先查同域历史审计（`audit_m*.md`、`docs/AUDIT_*.md`）与仓内既有门禁
   `tests/test_readme_quickstart.py`（前者已 pin README 的 Python 片段），确认本单做的是
   *CLI/链接/数字溯源* 维度，不重复既有工作。
2. **静态取证（无需 import torch）**：用 `ast` 解析每个被引脚本的 `parser.add_argument(...)`，
   得到 `{flag: choices}`；与文档命令逐 token 对照（旗标存在性 + 枚举取值合法性）。
3. **动态取证**：按 README 原样复跑 4 条命令（缩小到可接受墙钟，参数与 README 一致或按 README
   自己的 smoke 档），回读 `run_metadata.json` / `forces.csv` / `fields_step_*.npz` / GIF 落点。
4. **零手抄溯源**：定量表 2D 圆柱行 → `docs/CASE-KARMAN-RE100.md`（该文档自身记录 refs PDF 的
   sha256 与解析脚本），把行内每个数值回查该文档文本，命中率必须 100%。
5. **门禁化**：把上述判据写成 `tests/test_docs_consistency.py`（8 个测试，3.9–5.2 s，纯 CPU、
   不跑求解器），先取**红基线**（回退三份文档 → 4 failed / 4 passed），修完再取**绿基线**（8 passed）。
6. **修复最小增量**：只改文档（`README.md`、`README.zh-CN.md`、`docs/software_manual.md`）
   ＋新增 1 个测试文件；**不改任何求解器/API 代码**（假的是文档，不是代码）。

## ③ 计算结果（数字表，程序化注入）

### 3.1 审计矩阵（18 项，机器判定）

| # | 文档声明 | 取证结果 | 判定 |
|---|---|---|---|
| L1 | README.md 的相对链接全部落盘 | 22 条本地链接，缺失 0 | PASS |
| L2 | README.zh-CN.md 的相对链接全部落盘 | 11 条本地链接，缺失 0 | PASS |
| S1 | 文档命令引用的脚本/基准脚本均存在 | 扫描 4 份文档共 40 条命令，脚本缺失 0 | PASS |
| S2 | 文档命令中的 CLI 旗标与取值被脚本 argparse 接受 | 校验 123 个旗标 / 12 个带 choices 的取值，违例 0 | **不一致 → 已修** |
| S3 | README --device 罗列值与 cylinder_flow.py 的 choices 一致 | 文档 ['cpu', 'cuda', 'sdaa'] vs CLI ['cpu', 'cuda', 'sdaa'] | **不一致 → 已修** |
| A1 | README「API layering」列出的 tensorlbm.api/experimental/lattice_models/physics/cad/ai/backends 均存在 | 7 个命名空间，缺失 0 | PASS |
| A2 | D3Q27 原语以实名 equilibrium27/macroscopic27 记录，且不谎称属于 tensorlbm.api | tensorlbm.d3q27 实名存在=True；api 稳定导出 D2Q9/D3Q19 原语=True；D3Q27 名在 bullet 中=True | **不一致 → 已修** |
| Q1 | 定量汇总表 2D 圆柱行的数字可溯源到 docs/CASE-KARMAN-RE100.md | 行内 7 个数值，不可溯源 0 | **不一致 → 已修** |
| P1 | README「Restart / resume」所引平台路由与字段存在 | cylinder-flow=True, sphere-flow=True, resume_checkpoint+resume_from_job_id=True, job_manager 续跑=True | PASS |
| C1 | README 能力条目抽样 10 个符号（RANS/统计/MultiGPU/DFSEM/DFM/Q准则/λ₂/FWH-SPL）真实存在 | 抽样 10 个，缺失 0 | PASS |
| R1 | README 小 smoke 命令 `cylinder_flow.py --nx 64 --ny 24 --radius 4 --n-steps 20` 可运行 | steps=20，runtime nan s, steady_diagnostics=2 | PASS |
| R2 | README quick-start 命令 `cylinder_flow.py --nx 128 --ny 48 --re 200 --n-steps 500` 可运行 | steps=500，runtime nan s, steady_diagnostics=3 | PASS |
| R3 | README 3D smoke 命令 `sphere_flow_3d.py --nx 60 --ny 30 --nz 30 --n-steps 50` 可运行 | steps=50，runtime nan s, steady_diagnostics=2 | PASS |
| R4 | README 批量扫描命令 `param_scan.py` 可运行并产出汇总 CSV/PNG | steps=2，scan_summary.csv 存在 | PASS |
| B1 | README 记录 `--backend`: torch(默认)/paddle/mindspore，与 CLI choices 一致 | CLI choices=['mindspore', 'paddle', 'torch'] | PASS |
| D1 | README「Tests and checks」对 CI 的描述与 .github/workflows/ci.yml 实际命令一致 | README 已去掉「same test command」表述=True；CI 实跑 pytest 且附加 --cov/--timeout/--junitxml=True | **不一致 → 已修** |
| R5 | scripts/make_demo_gif.py 输出路径与 README 声明一致（outputs/lid_driven_cavity/demo_gif/…） | 实跑落点 artifacts/m17139/gif/lid_driven_cavity/demo_gif/lid_driven_cavity_demo.gif 存在=True | PASS |
| R6 | 缩放鲁棒性注意点：lid-driven cavity 在 nx=32/Re=400/100 步时发散（mass=nan） | README 默认 nx=96；降分辨率不做保证 | 注意点 |

> 状态口径：`PASS` = 文档与代码一致；`不一致 → 已修` = 修复前实红（回退文档重跑测试确认），
> 当前树已绿并由此测试长期守住；`注意点` = 非文档矛盾，属适用范围限制。

### 3.2 不一致明细（4 处，含最关键 3 处）

| # | 文档原文（修复前） | 代码/工件真相 | 修复 |
|---|---|---|---|
| 1（关键） | `--device`: `cpu` (default), `cuda`, or `mps` (Apple Silicon) | `examples/cylinder_flow.py` 的 `--device` choices = `cpu/sdaa/cuda`；实跑 `--device mps` → `error: invalid choice: 'mps'`。**同款错误还出现在 `docs/software_manual.md` §6.4** | README 改为 `cpu/sdaa/cuda` 并说明 MPS 需经库层 `resolve_device`；software_manual §6.4 换成库层调用说明 |
| 2（关键） | D2Q9/D3Q19/D3Q27 lattice primitives (`equilibrium`, `macroscopic`, lattice constants) | D3Q27 实名 `equilibrium27`/`macroscopic27`，位于 `tensorlbm.d3q27`，**不在** `tensorlbm.api.__all__` | README + 中文 README 改为逐格子实名（含 `C/W/OPPOSITE`、`C3D/W3D/OPPOSITE3D`、`W_EXACT64`）与归属模块 |
| 3（关键） | 2D cylinder flow \| St (Re=100) \| ≈ 0.183 \| 0.166 (Williamson 1989) \| ~10% | 仓内受审证据 `docs/CASE-KARMAN-RE100.md`：St=0.1705（vs 锚 0.16300，+4.60%）、Cd=1.5445（vs 1.3300，+16.13%，**判 FAIL**）；0.183/0.166 在仓内任何工件中都不存在 | 表格行改为受审真值 + 指向证据文档，并显式写出 Cd 的 honest FAIL |
| 4（次要） | GitHub Actions runs the same test command | CI 实跑 `pytest -q <scope> -o timeout_func_only=false --cov=… --junitxml=…` | 改为「同一套测试 + 覆盖率/超时/JUnit 附加」 |

### 3.3 示例可运行性（复跑回读 `run_metadata.json`）

| 复跑命令 | nx | ny | nz | steps | Re | 墙钟 (s) |
|---|---|---|---|---|---|---|
| cylinder smoke | 64 | 24 | - | 20 | 100.0 | nan |
| cylinder quick-start | 128 | 48 | - | 500 | 200.0 | nan |
| sphere smoke | 60 | 30 | 30 | 50 | 50.0 | nan |

- 产物布局与 README §Output organization 一致：`run_metadata.json`、`forces.csv`、
  `flow_step_*.png`、`fields_step_*.npz`、`checkpoint_f.pt`/`checkpoint_meta.json`
  （3D 分支同构，无 `forces.csv` 之外的差异即 `flow_step_*` 文件名随 `--output-interval` 变化）。
- 演示 GIF：`scripts/make_demo_gif.py` 缩参实跑（`--nx 32 --n-steps 100 --frame-every 50`）落点
  `artifacts/m17139/gif/lid_driven_cavity/demo_gif/lid_driven_cavity_demo.gif`，与 README 声明的
  `outputs/lid_driven_cavity/demo_gif/lid_driven_cavity_demo.gif` **路径结构一致**。
- 批量扫描：`param_scan.py --re 20 40 --nx 64 --ny 24 --n-steps 100` 产出 `scan_summary.csv` +
  `scan_summary.png`（注：该缩短步数下 Cd 仍为 `nan`，属统计窗不足，非命令失效；README 原命令
  用 2000 步统计窗）。

### 3.4 物理基准数字（从受审证据文档程序化解析，非手抄）

| 量 | TensorLBM | 参考锚 | 误差 | 仓内判定门 | 结论 |
|---|---|---|---|---|---|
| Strouhal St (Re=100) | 0.1705 | 0.16300 | +4.60% | ±8.0% | **通过** |
| 阻力系数 Cd (Re=100) | 1.5445 | 1.3300 | +16.13% | ±8.0% | **FAIL（超出 15% 不可接受线）** |

## ④ 结果图

**图 1｜文档 quick-start 命令（`cylinder_flow.py --nx 128 --ny 48 --re 200 --n-steps 500`）
第 500 步的涡量场云图**（黑线=圆柱固壁面，数据源 `artifacts/m17139/qs/…/fields_step_000500.npz`，
直接由被审文档命令产出）：

![图1 涡量场](../artifacts/m17139/figs/fig1_field.png)

**图 2｜关键量 vs 参考锚与 ±8% 判据带**（数值解析自 `docs/CASE-KARMAN-RE100.md`）：
左 St 落带内（+4.60%），右 Cd 出界（+16.13%）——这正是 README 定量表原先掩盖的事实。

![图2 基准对照](../artifacts/m17139/figs/fig2_karman.png)

**图 3｜审计矩阵（18 项状态）**：绿色=一致，橙色=修复前不一致、本单已修，灰色=注意点。

![图3 审计矩阵](../artifacts/m17139/figs/fig3_matrix.png)

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **文档不是代码的影子，两边都可能是错的**：`--device mps` 的根因是**示例 CLI 收窄了 choices**
   （库层 `resolve_device` 支持 mps 与 npu/mlu/musa），故正确修法是改文档而不是放宽 CLI——
   示例脚本并未验证 MPS 路径。不要用「改代码迁就文档」抹平差异。
2. **数字漂移最危险**：README 定量表引 0.183/0.166 且给「~10%」的乐观印象，与仓内受审证据
   （0.1705/0.16300，Cd 超标被判 FAIL）冲突。此类漂移必须靠「行内数字必须能在证据文档中命中」
   的机器判据守住，人工 review 抓不到。
3. **既有门禁的盲区**：`tests/test_readme_quickstart.py` 只 pin Python 片段，不覆盖 CLI/链接/数字。
   本单新增的门禁补的是这块；两者互补，不重复。
4. **覆盖范围限制（诚实交代）**：仅扫描 4 份「带可执行命令」的文档；`docs/` 下其余 600+ 篇文档
   未做命令级扫描（其中 `docs/council/…` 等为历史议事记录，本身不是契约文档）。
5. **`--device` 只校验「文档列出的值 ⊆ CLI choices」**，不保证该设备在目标机上真的可用
   （CUDA/SDAA 不可用时会显式报错，不静默降级——这属库层行为，已被既有测试覆盖）。
6. **方腔降分辨率发散**：`scripts/make_demo_gif.py --nx 32 --n-steps 100` 出现 `mass=nan`
   （README 默认 `nx=96`），说明该算例在粗网格/短窗下不稳定；本单不改求解器，仅登记为适用范围限制。
7. **`param_scan` 短窗 Cd=nan** 同样是统计窗不足所致，README 原命令的 2000 步档才有意义。

## ⑥ 误差判定（逐项，显式回答「是否 < 3%」）

| 审计量 | 判定方式 | 误差定义 | 结果 | < 3%？ |
|---|---|---|---|---|
| 文档链接落地 | 22+11 条相对链接存在性 | 缺失数 | 0 缺失 | **是（0 误差）** |
| 脚本存在性 | 40 条命令引用的 17 个脚本 | 缺失数 | 0 缺失 | **是（0 误差）** |
| CLI 契约 | 123 旗标 + 12 枚举取值 | 违例数 | 修复前 2 例（`--device mps` in README 与 software_manual），修复后 0 | **是（0 误差）** |
| 能力符号 | 10 个符号 `hasattr` | 缺失数 | 0 缺失 | **是（0 误差）** |
| 数字溯源 | 定量表圆柱行全部数值 vs 证据文档 | 不可溯源数 | 修复前 2/2 不可溯源（0.183、0.166），修复后 0 | **是（0 误差）** |
| 门禁测试 | `tests/test_docs_consistency.py` | 通过数 | 红基线 4 failed/4 passed → 修复后 **8 passed** | **是** |
| 物理量 St（顺带复核，非本单产物） | vs 文献锚 0.16300 | 相对误差 | **+4.60%** | **否（>3%，但在 ±8% 门内）** |
| 物理量 Cd（顺带复核，非本单产物） | vs 文献锚 1.3300 | 相对误差 | **+16.13%** | **否（≫3%，超 15% 不可接受线，仓内已判 FAIL）** |

**结论**：文档—代码一致性维度 6 项判据全部达到 0 误差（结构可用机器裁决，无「近似」空间）；
本单顺带纠正的物理数字标注了真实误差 **St +4.60% / Cd +16.13%**，**均不满足 <3%**，其中 Cd
超出 15% 不可接受线——该 FAIL 属求解器精度问题（`docs/CASE-KARMAN-RE100.md` 已诚实记录），
本单不改物理、也不粉饰，只保证 README 不再给出与此相反的数字。

---

### 复现

```bash
# 1) 门禁（纯 CPU，~5 s）
PYTHONPATH=src python -m pytest tests/test_docs_consistency.py -q
# 2) 审计矩阵 + 三张图（读回实跑产物 + 解析证据文档）
PYTHONPATH=src python artifacts/m17139/audit_probe.py
# 3) 文档命令复跑（任意一条，例）
PYTHONPATH=src python examples/cylinder_flow.py --nx 64 --ny 24 --radius 4 --n-steps 20 \
  --output-interval 10 --output-root artifacts/m17139/cyl --run-name smoke --overwrite
```
