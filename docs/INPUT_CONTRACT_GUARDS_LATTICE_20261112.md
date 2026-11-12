# 输入契约防线（第二层）：格构/特性兼容性 fail-fast —— `GeneralSimEngine` D3Q27

**日期**：2026-11-12 · **仓**：TensorLBM · **接续**：`docs/INPUT_CONTRACT_GUARDS_20261111.md`（m14763，审计 D5/D6/D7/D9 标量输入防线）、`docs/AUDIT_RUNTEST_6_TENSORLBM.md`（m14496）

## 1. 案例介绍（问题背景）

`docs/INPUT_CONTRACT_GUARDS_20261111.md` 把「同一非法输入，看路径决定是否报错」这一缺陷类在**标量**层面关掉了（`tau<=0.5`、`rho_out=0`、`viscosity=0`、`H=0`）。

同一缺陷类在**配置**层面仍然活着：`lattice=D3Q27` 与只存在 D3Q19 内核的特性（wall function、Smagorinsky-BGK/MRT、TRT）组合时，`GeneralSimConfig.validate()` 接受、`GeneralSimEngine.setup()` 也接受，直到 `run()` 里才抛 `NotImplementedError` —— 而那时几何已经体素化、27 方向分布函数已经分配。真实网格上这是数分钟与数 GB 的浪费，只为知道这个配置根本不可能跑。

第二条缺陷：`setup()` 返回的契约字典用大写 `"Q"`，而 `tests/test_general_sim_d3q27.py` 与 API 消费方按下写 `info["q"]` 读取；同时缺少 `engine.is_d3q27` 属性。三条测试因此在 `main` 上长期红：`tests/test_general_sim_d3q27.py` = **3 failed / 10 passed**。

复现（本单实测，pristine HEAD）：

```
python -m pytest tests/test_general_sim_d3q27.py -q
# FAILED test_engine_setup_routes_d3q27_kernels      KeyError: 'q'
# FAILED test_engine_d3q19_path_unchanged            KeyError: 'q'
# FAILED test_engine_d3q27_rejects_d3q19_only_features AttributeError: '_check_d3q27_supported'
# 3 failed, 10 passed
```

## 2. 怎么做的（方法/关键设置）

判定先于动手（判据可由机器裁）：**同一条不受支持的配置，被拒绝时分配的 `f` 字节数必须为 0**，且正向对照（受支持的组合）必须照旧跑通。

1. `src/tensorlbm/general_sim.py`
   - 新增配置分类器 `D3Q27_COLLISIONS = {bgk, mrt}` / `D3Q27_WALL_TREATMENTS = {auto, bounce_back}`（仓内 D3Q27 路径真实存在的内核，来源 `d3q27.py` 与 `_get_collide_fn`）。
   - `_d3q27_unsupported_reasons()`：纯配置分类，返回不受支持项列表（空=可用）。`AUTO` 先按 Re 解析成具体值（与 `setup()` 的自动选择同一函数 `_auto_select_collision/_auto_select_wall_treatment`），避免「AUTO 落到可用内核却被误拒」与「AUTO 落到 D3Q19-only 内核却被误收」。唯一豁免：`AUTO→wall_function` 且 `GeometrySource.NONE`（空河道没有固体，wall function 无事可做，保持既有可跑行为）。
   - `_check_d3q27_supported()`：`NotImplementedError`，消息点名违规项并给出可用替代（`D3Q19` / `bgk|mrt` + `bounce_back`）。该方法名由既有测试 `tests/test_general_sim_d3q27.py::test_engine_d3q27_rejects_d3q19_only_features` 早已声明。
   - `setup()` 首行调用它（在单位换算、几何体素化、`f` 分配之前）。
   - `is_d3q27` 属性；`setup()` 返回字典补 `"q"`（保留历史 `"Q"`，纯增量）。
   - `run()` 内既有的 wall-function 拒绝保留为纵深防御（它依赖 `solid.any()`，语义不同，未改）。
2. 证据脚本 `scripts/contract_guards/lattice_feature_probe.py`（新，自包含、不依赖 pytest）：对每条组合记录 `stage`（config/setup/run/ran）、`cells`、`field_bytes`、`elapsed_s`、异常类型与消息；`expect` 字段区分「应拒」与「正向对照」。脚本记录 `tensorlbm.__file__` 作为出处（本机 editable install 会解析到兄弟 worktree，必须显式 `PYTHONPATH=<本树>/src`，否则测的不是本树代码）。
3. 前后对照：`git archive HEAD` 导出 pristine 树到 `/tmp/lattice_before`（**同一份探针脚本**、同一环境、同一 8 格分辨率配置），before/after 只差 `general_sim.py` 的守卫。
4. 作图 `scripts/contract_guards/plot_lattice_probe.py` 读两份 JSON 直出图（零手抄）。

## 3. 计算结果（程序化注入，零手抄）

探针（8 格分辨率、16 128 格、Re=100，除非另注）：

| 组合 | before stage | before `field_bytes` | after stage | after `field_bytes` |
|---|---|---|---|---|
| `d3q27 + wall_function` | run | 1 741 824 | **setup** | **0** |
| `d3q27 + smagorinsky_mrt` | run | 1 741 824 | **setup** | **0** |
| `d3q27 + smagorinsky_bgk` | run | 1 741 824 | **setup** | **0** |
| `d3q27 + AUTO(wall) @ Re=1e4` | run | 1 741 824 | **setup** | **0** |
| `d3q27 + trt` | run | 1 741 824 | **setup** | **0** |
| 正向对照 ×4（d3q27 bgk/mrt、d3q19 wall_function/smagorinsky_mrt） | ran | — | ran | — |

汇总（`artifacts/contract_guards/lattice_probe_{before,after}.json` 的 `summary`）：

| 指标 | before | after |
|---|---|---|
| unsupported_combos | 5 | 5 |
| rejected_before_allocation | 0 | **5** |
| rejected_during_run | 5 | **0** |
| silently_accepted | 0 | 0 |
| bytes_allocated_before_rejection | 8 709 120（8.3 MB） | **0** |
| positive_controls_ok | 4/4 | 4/4 |

`f` 的分配量按 `27 × N³ × 4 B` 随网格三次方增长（8 格 → 1.66 MiB；128³ → 0.43 GB；256³ → 3.4 GB），故「拒在 setup」在生产网格上省下的是 GB 级分配，不只是微秒级。

测试（本机实跑）：

| 套件 | before | after |
|---|---|---|
| `tests/test_general_sim_d3q27.py` | 3 failed / 10 passed | **13 passed** |
| `tests/test_general_sim_lattice_contract.py`（新，18 条） | 不存在 | **18 passed** |
| 六文件定向集（`test_input_contract_guards` / `test_general_sim_d3q27{,_activation}` / `test_general_sim_lattice_contract` / `test_gpu_core_path` / `test_momentum_exchange_wet_node`） | — | **108 passed, 1 skipped**（70.9 s） |

## 4. 结果图

`figures/lattice_contract_probe.png`（由 `scripts/contract_guards/plot_lattice_probe.py` 从两份 JSON 直出）：

- 面板 A：每条不受支持组合「被拒绝前已分配的 `f` 字节数」before（红）vs after（绿，全 0）。
- 面板 B：拒绝发生阶段 before（全部 `run`）→ after（全部 `setup`）。

本单是后端契约改造，没有场量云图或解析解误差曲线；唯一的定量结论就是上面这条 fail-fast 指标，故作图只用探针数据，不重绘表格。

## 5. 注意点（坑与局限）

- **有意收紧既有契约，并改了一条旧测试**：`tests/test_general_sim_d3q27_activation.py::test_negative_unsupported_collision_raises` 原先要求「`setup()` 成功、只在 `_get_collide_fn()` 抛」。新契约是「`setup()` 即拒」，故该测试改为断言 `setup()` 抛 `smagorinsky_mrt` 且 `eng.f is None`，并保留 `_get_collide_fn()` 的纵深防御断言（绕开 `setup()` 直设 `_lattice_q=27`）。这是本单唯一的行为变更点，已在测试 docstring 与本文中声明。
- **AUTO 按 Re 解析后再判**：`Re ≥ 1e4` 的 D3Q27 + `AUTO` 现在在 `setup()` 就报错（此前是在 `run()` 报同一个错）。空河道（`GeometrySource.NONE`）豁免，避免「没有固体却因 wall function 被拒」的假失败。
- **未修（如实申报）**：D3Q19 路径 `_get_collide_fn()` 对 `CollisionModel.TRT` / 其它未列值走 `else: return collide_mrt3d_low_memory`，即**静默替换成 MRT**——同一「静默替换」缺陷类。改动它会影响现有调用方的物理预言，超出本单（格构兼容性）范围，留作后续单；本单只保证 D3Q27 侧不再静默。
- **探针分辨率小**（8 格，16 128 格）：`field_bytes` 绝对值小，但结论（阶段前移、分配为 0）与分辨率无关；生产网格的 GB 级放大是由 `27N³×4` 直接推出的，不是实测。
- **出处陷阱**：本机 editable install 使 `import tensorlbm` 解析到 `/root/tp-agent/work/m13963/src`（兄弟 worktree）。第一次 before 跑就踩了，已在脚本里记录 `package_path` 并要求显式 `PYTHONPATH=<本树>/src`；两份 JSON 的 `package_path` 分别是 `/tmp/lattice_before/src` 与 `/root/tp-agent/work/m16162/src`。
- **全量测试**：本机只跑定向集（`make test-all` 规模过大，按容量纪律上集群，作业 `cj1251` 递交 `pytest -m 'not slow'`）。若队列在预算内未派发完成，按未完成记账，不写成 passed。

## 6. 误差判定

本单无物理基准量（不是新算例），判据是**契约指标**，逐项显式回答：

| 判据 | before | after | 是否达标 |
|---|---|---|---|
| 不受支持组合被拒绝前分配的 `f` 字节数 | 8 709 120 | 0 | ✅ 达标（判据=0） |
| 不受支持组合在 `run()` 才被拒的数量 | 5/5 | 0/5 | ✅ 达标（判据=0） |
| 正向对照仍可跑（无「一律拒绝」假绿） | 4/4 | 4/4 | ✅ 达标（判据=4/4） |
| `tests/test_general_sim_d3q27.py` | 3 failed | 13 passed | ✅ 达标（判据=0 failed） |
| 新增契约测试 | — | 18 passed | ✅ 达标 |

**是否 <3%**：不适用——本单交付物是入口判据而非数值预报，无「关键量 vs 基准」可比。若需要物理精度结论，见 `docs/AUDIT_RUNTEST_6_TENSORLBM.md` §5 与后续收敛/基准单；本单不触碰任何求解器数值路径（只加配置分类与一个字典键），求解结果不变。