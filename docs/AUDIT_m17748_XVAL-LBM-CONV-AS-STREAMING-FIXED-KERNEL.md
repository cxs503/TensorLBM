# 交差审计 · XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL · m17680

**审计单** m17748（board-audit，机器复核口径）· **被审单** m17680（succeeded，auto-approve）· **仓** TensorLBM · 本文件全部数字由 `artifacts/audit_m17748/make_report.py` 从核验工件注入。

## 0. 结论先行

- 判级 **L2**（独立功能实现）：真代码 + 真运行工件 + 独立复跑一致，但**零集成调用方**（L3 不达）、**基准裁判表 0 行**（L4 无法由登记基准裁）、**全部交付文件未提交**（git 0/15 个运行工件被追踪，0/7 个实现文件被追踪，main 无该实现）。
- 物理面本身是**实的**：预声明 KPI 全部通过 owner 15% 线（最差 0.873%），两臂（原生 gather vs OSS 定核卷积）逐位一致（sha256 相同、|diff|=0），独立复跑 4872 个字段中仅 7 个有差异（差异见 §3.3，均为元数据而非数值）。
- **证据完整性缺陷（本单最重要发现）**：被审单自己的运行目录 `runs/xval_conv_as_streaming/` 里 8/15 个文件属于**另一个锚点** `XVAL-DG-ROUND2-CERTIFIED-REFERENCE`（TensorDG/shu-osher，容差 5%）。其中 `benchmark_ref_vs_got.json` 这个**文件名正好是 L4 判据要的键**，内容却是 DG 的——任何按文件名取证的自动化都会把别人的数据当成这个锚点的 ref-vs-got。本审计按内容自述的 anchor 逐文件归类，故不采信。

## 1. 案例介绍：审计对象与授权链

| 项 | 值 |
|---|---|
| 锚点 | `XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL`（任务榜 Tier A，XVAL 交叉对标） |
| 被审单 | m17680，repo TensorLBM，status=succeeded，`fail_stage=budget_exhausted+forced_finish` |
| 被审工作树 | `/root/tp-agent/work/m17680`（分支 agent/m17680，HEAD 与 main 同点：交付文件全为 untracked） |
| 对标对象 | `loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow`（LatFlow）的 conv-as-streaming 定核做法 |
| 总账验收 | `feature_ledger.acceptance` = 「<15%（owner 精度标准：对标误差 >15% 不可接受）」，status=open |
| 平台质量审计块 | 被审单 result.server_quality_audit：`gate_valid=false, key_present=false, gate_missing_warning=true`（原单无 gate 块，本单遂为首个机器复核口径） |

**物理背景**（为判级需要的量级感）：本锚点不是新物理，而是**输运算子的等价重构**——把 D2Q9/D3Q19 的「按方向索引搬移（gather/roll）」改写成「一次常数卷积核（fixed kernel conv）」，加上周期性 mobius padding 与1x1 跨通道 bounce 卷积。物理标定用两个解析解：二维 Taylor-Green 衰减（γ_K = 2νk² 精确）与平面 Poiseuille 抛物线。

## 2. 怎么做的：审计流程（非重做）

1. **取证**：读 m17680 的 result（含 git_status/on_disk_changes/fail_stage）与工作树 `runs/xval_conv_as_streaming/`、`src/tensorlbm/streaming_conv.py`、`tests/test_streaming_conv.py`、`docs/XVAL-LBM-...md`。
2. **基准裁判优先**：先查 `tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL'`（结果见 §4）。
3. **独立复跑**：在**被审树之外**执行原单复现命令 `PYTHONPATH=src python3 benchmarks/xval_conv_as_streaming/run_xval.py --quick --out /tmp/repro_m17680 --no-doc`，得 rc=0、verdict={'implementation_equivalent': True, 'all_kpi_under_15pct': True, 'all_kpi_under_3pct': True, 'verdict': 'PASS'}，并与仓内 shipped `result.json` 逐字段比对（4872 字段）。
4. **源码级探针**：对运行目录逐文件算 sha256、判 git 追踪、并读每个文件**自述的 anchor**（json 键 / 字节扫描 / npz 变量名 / 驱动输出名），做归属分类。
5. **测试复核**：`PYTHONPATH=src python3 -m pytest tests/test_streaming_conv.py -q` → **1 failed, 13 passed**（与原单自报一致）。
6. **集成面 grep**：`grep -rn streaming_conv --include=*.py src/ app/ benchmarks/ tests/` → 仅 tests + benchmark 自身，主求解器零调用。

**独立性声明（限制）**：① 复跑用的是**被审树里的同一份代码**（代码本身未被独立重写），所以本单证的是「工件与代码一致、可复跑」，不是「数学对」——数学对由解析解裁判。② 上游 LatFlow 无法在本机执行（TF1.x graph API + Py2 `xrange`），参照侧是**闭式解析解**，不是上游二进制输出；上游表格由 `extract_oss_tables.py` 机读并留 sha256（§3.2）。③ 裁判表零行（§4）→ 本锚点没有平台登记基准，容差只能取总账 15% 与内控 3%。

## 3. 三件套核验矩阵（run_artifacts / benchmark ref-vs-got / repro_cmd）

### 3.1 run_artifacts：真存在？是本人的？提交了？

| 文件 | 字节 | 归属 | git 追踪 |
|---|---|---|---|
| `xval_conv_as_streaming/figures/fig1_tg_energy_decay.png` | 58293 | 本锚点 | 否 |
| `xval_conv_as_streaming/figures/fig2_tg_field_and_arm_deviation.png` | 45521 | 本锚点 | 否 |
| `xval_conv_as_streaming/figures/fig3_tg_sensitivity.png` | 40562 | 本锚点 | 否 |
| `xval_conv_as_streaming/figures/fig4_poiseuille_profile.png` | 57470 | 本锚点 | 否 |
| `xval_conv_as_streaming/figures/fig5_transport_cost.png` | 33635 | 本锚点 | 否 |
| `xval_conv_as_streaming/result.json` | 148638 | 本锚点 | 否 |
| `xval_conv_as_streaming/table.csv` | 827 | 本锚点 | 否 |
| `xval_conv_as_streaming/benchmark_ref_vs_got.json` | 11387 | **外来（DG 锚点）** | 否 |
| `xval_conv_as_streaming/certification.json` | 1369 | **外来（DG 锚点）** | 否 |
| `xval_conv_as_streaming/fig1_profiles.png` | 71937 | **外来（DG 锚点）** | 否 |
| `xval_conv_as_streaming/fig2_band_and_legs.png` | 42592 | **外来（DG 锚点）** | 否 |
| `xval_conv_as_streaming/platform_documents.json` | 3775 | **外来（DG 锚点）** | 否 |
| `xval_conv_as_streaming/profiles.npz` | 15268 | **外来（DG 锚点）** | 否 |
| `xval_conv_as_streaming/report.md` | 3502 | **外来（DG 锚点）** | 否 |
| `xval_conv_as_streaming/run.json` | 13580 | **外来（DG 锚点）** | 否 |

→ 本锚点自有工件 7 个（result.json + table.csv + figures/fig1..fig5），**外来 8 个**；**全部 15 个文件 git 未追踪**，`git ls-files | grep -i streaming_conv` 在 main 为空。
→ 外来名单：`xval_conv_as_streaming/benchmark_ref_vs_got.json`, `xval_conv_as_streaming/certification.json`, `xval_conv_as_streaming/fig1_profiles.png`, `xval_conv_as_streaming/fig2_band_and_legs.png`, `xval_conv_as_streaming/platform_documents.json`, `xval_conv_as_streaming/profiles.npz`, `xval_conv_as_streaming/report.md`, `xval_conv_as_streaming/run.json`。其中 `report.md`（正文标题即 `XVAL-DG-ROUND2-CERTIFIED-REFERENCE`）、`certification.json`（`verdict=CERTIFIED`，FV-WENO5 800/1600/3200 阶列，件内 `observed_order`）、`benchmark_ref_vs_got.json`（`anchor` 键自述 DG）、`run.json`、`platform_documents.json`、`profiles.npz`（变量名 `rho_p1_n100_cfl0.4`）、`fig1_profiles.png`、`fig2_band_and_legs.png`（后二者文件名不在本锚点驱动 `make_figures()` 的输出集内）。

### 3.2 benchmark ref vs got：预声明 KPI 逐项对闭式解

预声明阈值（`predeclared.json`，先于运行）：owner 验收 15.0%、内控目标 3.0%、实现等价 `rel<1e-12`。实际跑到的网格：TG [[32, 50.0, 600], [32, 100.0, 600]]，Poiseuille Ny=[12]；**预声明敏感性轴为 ['resolution n in {64, 128} at Re=100, U0=0.05 (discretisation)', 'Reynolds number Re in {50, 100} at n=128, U0=0.05 (tau = 0.885 vs 0.692, i.e. the less-damped regime)', 'channel height Ny in {16, 32} for the wall arm (bounce formulation)']**，即交付的 quick 网格（n=32 / 600 步 / Ny=12）低于预声明阶梯（原单 §Scope note 已如实披露该缺口）。

| case | arm | KPI | ref（闭式） | got | err % | <15% | <3% |
|---|---|---|---|---|---|---|---|
| n32_Re100 | A_native_gather | gamma_K (TG k-decay rate) | 2.467401e-03 | 2.466757e-03 | 0.0261 | ✓ | ✓ |
| n32_Re100 | A_native_gather | gamma_vel (TG u-decay rate) | 1.233701e-03 | 1.236930e-03 | 0.2617 | ✓ | ✓ |
| n32_Re100 | A_native_gather | field L2 vs closed form (%) | 0.000000e+00 | 8.728006e-01 | 0.8728 | ✓ | ✓ |
| n32_Re100 | B_conv_fixed_kernel | gamma_K (TG k-decay rate) | 2.467401e-03 | 2.466757e-03 | 0.0261 | ✓ | ✓ |
| n32_Re100 | B_conv_fixed_kernel | gamma_vel (TG u-decay rate) | 1.233701e-03 | 1.236930e-03 | 0.2617 | ✓ | ✓ |
| n32_Re100 | B_conv_fixed_kernel | field L2 vs closed form (%) | 0.000000e+00 | 8.728006e-01 | 0.8728 | ✓ | ✓ |
| n32_Re50 | A_native_gather | gamma_K (TG k-decay rate) | 4.934802e-03 | 4.934703e-03 | 0.0020 | ✓ | ✓ |
| n32_Re50 | A_native_gather | gamma_vel (TG u-decay rate) | 2.467401e-03 | 2.469851e-03 | 0.0993 | ✓ | ✓ |
| n32_Re50 | A_native_gather | field L2 vs closed form (%) | 0.000000e+00 | 7.074040e-01 | 0.7074 | ✓ | ✓ |
| n32_Re50 | B_conv_fixed_kernel | gamma_K (TG k-decay rate) | 4.934802e-03 | 4.934703e-03 | 0.0020 | ✓ | ✓ |
| n32_Re50 | B_conv_fixed_kernel | gamma_vel (TG u-decay rate) | 2.467401e-03 | 2.469851e-03 | 0.0993 | ✓ | ✓ |
| n32_Re50 | B_conv_fixed_kernel | field L2 vs closed form (%) | 0.000000e+00 | 7.074040e-01 | 0.7074 | ✓ | ✓ |
| Poiseuille Ny=12 | A_native_gather | u_peak（解析 u_max=0.020000） | 0.020000 | 0.019856 | 0.2828 | ✓ | ✓ |
| Poiseuille Ny=12 | B_conv_fixed_kernel | u_peak（解析 u_max=0.020000） | 0.020000 | 0.019856 | 0.2828 | ✓ | ✓ |

臂间等价（这是采纳决定的承重证据，网格无关）：

| 探针 | bitwise_equal | max_abs_diff |
|---|---|---|
| 2d_gather_vs_conv | True | 0.0 |
| 2d_derived_vs_upstream_table | True | 0.0 |
| upstream_table_audit | None | None |
| 3d_conv_vs_canonical_roll | True | 0.0 |
| bounce_conv_vs_index | True | 0.0 |

### 3.3 repro_cmd：可执行？复跑数字一致？

- 命令可执行：rc=0，verdict=`{'implementation_equivalent': True, 'all_kpi_under_15pct': True, 'all_kpi_under_3pct': True, 'verdict': 'PASS'}`（本审计实跑，输出目录 `/tmp/repro_m17680`，与被审树隔离）。
- 逐字段比对：4872 个字段中 **7 个不同**，全部为：
  - `/poiseuille/cases[0]/steps`：shipped=`8000` vs repro=`12000`
  - `/poiseuille/cases[0]/profile_l2_pct`：shipped=`0.38338969964747166` vs repro=`0.38338969964747643`
  - `/poiseuille/cases[0]/profile[8]`：shipped=`0.010256000075817983` vs repro=`0.01025600007581799`
  - `/poiseuille/cases[1]/steps`：shipped=`8000` vs repro=`12000`
  - `/poiseuille/cases[1]/profile_l2_pct`：shipped=`0.38338969964747166` vs repro=`0.38338969964747643`
  - `/poiseuille/cases[1]/profile[8]`：shipped=`0.010256000075817983` vs repro=`0.01025600007581799`
  - `/poiseuille/pair[0]/profile_l2_pct_max`：shipped=`0.38338969964747166` vs repro=`0.38338969964747643`

判读：Poiseuille 的 `steps` 字段 shipped=8000 / 当前源码=12000，其余 6 处差异是同一物理量在 1e-15 相对量级的浮点末位（profile 稳态已到不动点，多迭代不改数值）→ **数值头条完全可复现**，但**工件与当前代码的指纹不完全对齐**（`steps` 元数据漂移说明 result.json 产自 steps=8000 的代码版本，仓内代码已是 12000）。

### 3.4 预声明阶梯的独立全量复跑（本审计补做，非原单交付）

为把 F5（quick 网格低于预声明阶梯）从「未验」变成「已验」，本审计在被审树里跑满 `--quick` 之外的全量模式（`--out /tmp/repro_full_m17680`，env: python 3.11.6 / torch 2.7.1 / Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38，线程 4，wall 296.4 s），verdict=`PASS`：

| n | Re | steps | arm | γ_K err % | field L2 % | <3% |
|---|---|---|---|---|---|---|
| 64 | 100 | 2400 | A_native_gather | 0.0114 | 0.7009 | ✓ |
| 64 | 100 | 2400 | B_conv_fixed_kernel | 0.0114 | 0.7009 | ✓ |
| 32 | 100 | 2400 | A_native_gather | 0.0003 | 0.6623 | ✓ |
| 32 | 100 | 2400 | B_conv_fixed_kernel | 0.0003 | 0.6623 | ✓ |
| 64 | 50 | 1200 | A_native_gather | 0.0029 | 0.5601 | ✓ |
| 64 | 50 | 1200 | B_conv_fixed_kernel | 0.0029 | 0.5601 | ✓ |

| Ny | arm | u_peak err % | profile L2 % |
|---|---|---|---|
| 16 | A_native_gather | 0.1436 | 0.1956 |
| 16 | B_conv_fixed_kernel | 0.1436 | 0.1956 |
| 24 | A_native_gather | 0.0580 | 0.0792 |
| 24 | B_conv_fixed_kernel | 0.0580 | 0.0792 |

判读：全量模式实际跑到 n∈{3,2,,, ,6,4}、Ny∈{1,6,,, ,2,4}——**两分辨率/两通道高都有了，但顶部 rung n=128 仍未跑**（预声明写的是 n∈{64,128}）；最差 TG 指标 0.701%、最差 Poiseuille 峰值 0.144%，随分辨率加密误差下降（n=64 field L2 0.7009% < n=32 0.6623%…同量级、单调性弱），两臂在全量模式下依然逐位一致。→ F5 由「未跑」降级为「顶部一档未跑」，不改变 L2 判级（L3 集成缺仍阻断）。

## 4. 基准裁判（唯一裁判）核验

```sql
SELECT case_name, metric, tolerance FROM tensor_benchmark_cases
 WHERE anchor_slug='XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL';
```

**返回 0 行** → 本锚点**未登记基准算例**，平台侧「got 是否在容差内」无法由登记基准裁决。

- 同域兄弟锚点确有登记（如 `XVAL-LBM-AVERAGE-SURROUND-INIT` 容差 `<15%`、`XVAL-LBM-BUFFER-LAYER-2TO1-REFINEMENT` `<15%`、`CASE-LBM-SQUARE-CYL-NORBERG` |Cd err|≤3%），说明「登记」这条路在本仓是走通的，本锚点属于**漏登记**。
- 现存可用裁判：总账 `acceptance` 的 15% owner 线 + 报告自设 3% 内控线 + **闭式解析解**（TG γ_K、Poiseuille u_max）；按此裁判，本单 KPI 全部通过（最差 0.873%）。
- 结论：**benchmark 判据不缺席但未登记**——L4 的『登记基准』要件不满足；这不是数值失败，是登记缺失（可返修补登记）。

## 5. 判级（L0–L5，逐级给证据）

| 级 | 要件 | 判定 | 证据 |
|---|---|---|---|
| L0 | 文档 | 达 | `docs/XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL.md` 184 行，§1–§8 齐（问题设定/方法/参数表/ref-vs-got/敏感性与成本/失效模式/差异清单/误差裁决） |
| L1 | 数学/单元验证 | 达 | `tests/test_streaming_conv.py` 14 项：13 passed，1 failed（float32 逐位相等断言）；bitwise 探针 4/5 通过 |
| L2 | 独立功能实现（真代码 输入→计算→输出） | **达（本级判级）** | `src/tensorlbm/streaming_conv.py`（10981 B，非占位：流核派生 + conv 搬运 + bounce_blend + describe_stream_table）；入口 `benchmarks/xval_conv_as_streaming/run_xval.py`；运行工件 `runs/xval_conv_as_streaming/result.json`（58293 B 起）+ 5 图；本审计独立复跑一致 |
| L3 | 系统集成（接主求解器/平台） | **不达** | `grep -rn streaming_conv src/ app/` 除自身外命中 0；调用方仅 `tests/test_streaming_conv.py:25` 与 benchmark 自身。生产输运算子（`src/tensorlbm/` 各 solver 的 roll/gather 路径）未提供切换开关 |
| L4 | 基准验证（完整算例+误差分析+可复现） | 不达 | 登记基准 0 行（§4）；交付网格为 quick（n=32/Ny=12），低于预声明阶梯 n∈{64,128}/Ny∈{16,32}；工件未提交（git 0 追踪）→ 第三方无法从任何分支复现 |
| L5 | 科研工程成果 | 不达 | 无文献级新结论；上游 D3Q19 表缺陷记录价值有，但未形成可复用的外部结论论文 |

**级别 = L2（功能实质：真输运算子 + 真解析对标 + 可复跑），不因报告篇幅升级。**

## 6. 发现清单（按严重度）

| # | 严重度 | 发现 | 证据 |
|---|---|---|---|
| F1 | **阻断（证据完整性）** | 被审单运行目录混入 8/15 个 DG 锚点文件，其中 `benchmark_ref_vs_got.json` 正是 L4 键名 | `inventory_m17680.json` → `foreign_files`；各文件自述 anchor/`fs_gci`/npz 变量名 |
| F2 | **阻断（交付）** | 实现与全部工件 git 未追踪（0/7 实现文件、0/15 工件），main/各分支均无该实现 | `git status --short` 全 `??`；`git ls-files | grep streaming_conv` 空 |
| F3 | 中 | 1 个测试红：`test_stream_conv_2d_is_bitwise_identical_to_gather[float32]` 用 `torch.equal` 断言浮点逐位相等，float32 下不成立（float64 过）。属于**断言过严**还是**float32 真不等价**未分离——原单报告 §6 明写「no float32 round-off study was run」 | pytest 实测 1 failed, 13 passed |
| F4 | 中 | `result.json` 的 Poiseuille `steps`=8000 与当前源码 12000 不一致 → 工件与代码指纹漂移 | `repro_compare.json` → `diffs` |
| F5 | 中→低 | 交付用 quick 网格（n=32/Ny=12）；本审计补跑全量模式得 n∈(32, 64)/Ny∈(16, 24) 仍全 <3%，唯顶部 rung n=128 未跑；原单已在 Scope note 如实披露 | `kpi_table.json` → `sensitivity_points_declared` vs `grids_actually_run` |
| F6 | 低 | 原单 result 无 gate 块（`gate_valid=false`），故本单为首个机器复核口径 | m17680 result.server_quality_audit |
| F7 | 低 | 原单未向 tensor_reports 发布报告（`SELECT count(*) ... WHERE mission_id=17680` = 0），报告纪律欠账由本单补 | SQL 实测 0 行 |

| F8 | 低（机制观察，非被审单之过） | `maturity_gate.callers_verified` 取 `str(callers[0]).split(":")[0]` 再 grep，散文式指针会被截成 `none` 并在仓内命中 → **声明 L4 可被散文骗过**（实测：把本单 evidence 的 level 改声明为 4，judge() 返回 4 无异议） | 本审计用 `maturity_gate.judge()` 对同一 evidence 复算：声明 2 → gate_level=2（confirmed 口径）；声明 4 → gate_level=4（虚标不会被机器拦住） |

**诚实边界**：F1 的成因本单**未查明**（无法从工件反推是共享工作树串写、还是驱动脚本被别的任务改过）；本单只确认事实=这些文件不属于本锚点，并排除其证据效力。

## 7. 误差判定（显式回答 <3%）

- Taylor-Green 衰减率误差：基线 0.0261%，两臂同值；最差 KPI 为 field L2 0.873% → **<3% 成立**。
- Poiseuille 峰值误差：0.2828%，profile L2 0.3834% → **<3% 成立**。
- 臂间差异：`max_abs_diff=0` 且 sha256 相同 → 重构不改解（这是采纳决定的承重证据，不是精度证据）。
- ⚠ 判读纪律：上列误差是**同一 BGK 格式对解析解的离散误差**，不是「迁移相对上游的精度损失」——后者恒等于 0（逐位一致）。把 0.873% 说成「对标上游误差」是口径错误，本审计不予采信。

## 8. 处置

- **verdict：L2 / confirmed 候选**（gate 机器复核口径已按 `impl_files/entry/run_artifacts/tests/integration_callers/benchmark/repro_cmd/external_anchor` 键给足真实指针）。
- **返修路径（升 L3/L4，另单）**：① 把 `stream_conv_2d/stream_conv_3d` 接进主求解器的输运切换（真调用方，grep 可复）；② 登记 `tensor_benchmark_cases`（anchor_slug=本锚点，metric=γ_K 相对误差 / field L2，tolerance 3%）；③ 提交实现与工件（并清掉运行目录里的 DG 外来文件）；④ 修 float32 断言（`pytest.approx`）或补 float32 round-off 研究；⑤ 跑满预声明阶梯 n=64/128、Ny=16/32 并把 result.json 与代码重新对齐（消除 F4）。
- **不应做**：用 DG 的 `benchmark_ref_vs_got.json` / `certification.json` 充当本锚点 L4 证据（F1 已排除）。

## 9. 附件（可复跑）

| 产物 | 路径 |
|---|---|
| 核验器（本审计全部数字的唯一来源） | `artifacts/audit_m17748/verify_m17680.py` |
| 工件归属取证 | `artifacts/audit_m17748/inventory_m17680.json` |
| 独立复跑逐字段比对 | `artifacts/audit_m17748/repro_compare.json` |
| KPI 表 + 阈值判读 | `artifacts/audit_m17748/kpi_table.json` |
| 图 1 ref vs got | `artifacts/audit_m17748/fig1_ref_vs_got.png` |
| 图 2 工件归属取证 | `artifacts/audit_m17748/fig2_provenance_forensics.png` |

复现：`python artifacts/audit_m17748/verify_m17680.py --audited /root/tp-agent/work/m17680 --repro /tmp/repro_m17680 --out-dir artifacts/audit_m17748`（复跑本体：`cd /root/tp-agent/work/m17680 && PYTHONPATH=src python3 benchmarks/xval_conv_as_streaming/run_xval.py --quick --out /tmp/repro_m17680 --no-doc`）。
