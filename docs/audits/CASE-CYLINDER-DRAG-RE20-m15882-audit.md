# 交差审计 · CASE-CYLINDER-DRAG-RE20（圆柱阻力 Re=20 / TensorLBM）

- **被审对象**：mission **15777**（`succeeded`，owner 2026-11-12 算例池令首轮返修单），commit **f2de132a**，PR **#1851**，工作树 `/root/tp-agent/work/m15777`
- **审计单**：mission **15882**（board-audit，交差核验 = 非重做），工作树 `/root/tp-agent/work/m15882`
- **审计席位独立性**：本方未参与 m15777 实现；所有数字由本方从工件二次计算或独立实跑得出（见 §4）。
- **基准裁判**：`tensor_benchmark_cases`（`anchor_slug='CASE-CYLINDER-DRAG-RE20'`）→ **0 行**（全表 25 行，无任何 cylinder 锚）。仓内 `refs.json` 因此在事实上是唯一裁判：Cd ref = **5.57953523384**，容差 **< 5 %**。

---

## 1. 结论摘要

| 项 | 判定 | 证据 |
|---|---|---|
| 实现真实性（真代码 输入→计算→输出） | ✅ | `src/tensorlbm/cylinder_st2d1.py` 451 行 torch D2Q9-MRT；工件 `cd` 由 `series.csv` 独立重算，差 **2.26e-6**（相对 3.9e-7） |
| run_artifacts 存在 | ✅ 存在 / ⚠️ 命名与 provenance 不符 | `runs/cylinder_drag_re20/D16/{results.json,series.csv,fields.npz}` 302 KB+117 KB+2 KB；**缺 `summary.json`，且文件名不是 `run_case.py` 的输出契约**（详见 F4） |
| ref vs got 数值 | ❌ **不达容差** | Cd = 5.87226 vs ref 5.57954 → **+5.246 %** > 5 %；工件自报 `pass:false`（如实） |
| repro_cmd 可执行 | ❌ | `refs.json:quick_cmd` 实跑 **rc=1**（`ValueError: lattice Mach 0.433 > ma_limit 0.300 at D_cells=8`）；`repro_cmd`（默认 τ=0.8 / D=20 40）**无法复现交付工件**（工件是 τ=0.62 / `--u-lb 0.05` / D=16，该命令行未记录） |
| 报告六段 | ❌ **缺席** | `docs/CASE_CYLINDER_DRAG_RE20.md` 不存在、`docs/assets/case_cylinder_drag_re20/` 空、`make_report.py` 从未执行、`tensor_reports` 无 mission 15777 行（`report_id` 为 NULL） |
| 判级 | **L3** | L2/L3 成立（真实现 + 应用层路由注册并实跑可达）；**L4 不成立**（主量 FAIL + 裁判缺案号 + 单网格无收敛分析 + 工件不可由声明命令复现） |

> 能力实质：**"能把 Schäfer–Turek 2D-1 在 CPU 上端到端跑出来、有真流场、有 API 端点、有测试"是真的；"该算例通过 <5 % 基准验收"不成立。**
> 与同锚点先例一致：m15411（同锚点）判 **L3 / reopened**（verdict id=368），本单在 CPU 可复现性上有实质增益，但基准结论仍是 FAIL。

---

## 2. 被审交付清单（commit f2de132a）

```
M app/backend/routers/cylinder_bench.py      (+44 行：POST /st2d1/validate → 委派库入口)
M src/tensorlbm/__init__.py                  (+8 行：lazy export 4 项 + TYPE_CHECKING + __all__)
A src/tensorlbm/cylinder_st2d1.py            (451 行：GEOMETRY/REFERENCE + run_cylinder_st2d1 + compare_to_reference)
A benchmarks/case_pool/cylinder_drag_re20/run_case.py   (193 行 CLI)
A benchmarks/case_pool/cylinder_drag_re20/refs.json     (2567 B，ref 与 provenance 钉扎)
A benchmarks/case_pool/cylinder_drag_re20/make_report.py(270 行，**未执行**)
A tests/test_cylinder_st2d1.py               (161 行)
工件（未入库，runs/ 被 gitignore）：runs/cylinder_drag_re20/D16/{results.json,series.csv,fields.npz} + D16.log + D20.log（无 JSON）
```

---

## 3. 基准对照（唯一裁判 = refs.json / 仓内锚点 sha256 已核）

ref 出处链：`refs.json.provenance.anchor_file = benchmarks/verified/cylinder/re20_st/result.json`，
本方实测 `sha256sum = df69f767ec9dec6db64169872e7a4a437de57077972bc76aab26ba9f3ee59eda`，与
`refs.json.provenance.anchor_file_sha256` **逐字符相同** ✅；该文件 `ref_cd = 5.57953523384`、`ref_source = "Nabh 1998 谱方法，FeatFlow 官方（ST 2D-1 Re=20）"` ✅。**数字来源可追溯（零手抄成立）。**

| 量 | ref | got（交付工件） | 误差 | 容差 | 判定 |
|---|---|---|---|---|---|
| Cd（gating） | 5.57953523384 | 5.872258219122886 | **+5.2464 %** | < 5 % | ❌ FAIL |
| CL（非 gating，自报） | 0.010618948146 | −0.031230416893959 | −394.1 % | — | ❌（符号反，如实披露） |

本方**从 `series.csv` 原始逐源重算**（不调用被审代码）：末 8000 步（序列 10 步抽取 → 末 800 采样）均值
**5.8722605**，误差 **+5.24641 %**，与自报 `cd` 差 **2.26e-6**、与自报误差差 **4.1e-5 pp** → 工件内部自洽，**自报数字无修饰**。
平台散度（本方从 `fields.npz` 中心差分重算）：rms **1.07e-3**、p99 6.3e-3、max 1.2e-2（格子单位；LBM 离散散度 O(Ma²)=7.5e-3，属正常量级）→ 流场是**真解**，非占位数据。

几何/无量纲自检（全部通过，本方独立重算）：
`blockage = 0.24390243902439`（= 0.1/0.41 ✓）· `nx=352, ny=66`（= 2.2/0.00625, 0.41/0.00625 ✓）· `Re = u_mean·D/ν = 0.05·16/0.04 = 20.0` ✓ · `Ma = 0.0866` ✓ · 固壁格点数 197（πr² = 201.1，半格阶梯 ✓）。

---

## 4. 三件套逐项核（① 取证）

### 4.1 run_artifacts —— 存在但 provenance 断裂（F4）

`run_case.py:117-135` 的输出契约是 `results_D{D}.json` / `series_D{D}.csv` / `fields_D{D}.npz` + **`summary.json`**；
磁盘上交付的是 `results.json` / `series.csv` / `fields.npz`，**无 `summary.json`**，`D16.log` 尾部还含一段被 tee 进去的 manifest JSON 与一行 `artifacts → runs/cylinder_drag_re20/D16`（`run_case.py` 不打印这一行）→
**该工件不是由仓内入口 `run_case.py` 产出，而是会话内未入库的临时驱动产出**。`D20` 只剩 64 行日志（被 kill 于 20000 步，drift 3.55e-3 未到平台）。

### 4.2 ref vs got —— 数值真、但 FAIL（F1）

见 §3。工件 `verdict` 段自报 `passed:false / gating:true / err_pct:5.2464`，**诚实**，未粉饰。

### 4.3 repro_cmd —— 不可执行且不可复现交付（F3）

实测（`/root/tp-agent/work/m15882`，同 commit 代码）：
```
PYTHONPATH=src python benchmarks/case_pool/cylinder_drag_re20/run_case.py --grids 8 --steps 300 --out /tmp/cyl_audit_quick
→ ValueError: lattice Mach 0.433 > ma_limit 0.300 at D_cells=8 (u_mean=0.2500): fix u_lb explicitly
   rc=1，输出目录空
```
即 `refs.json:quick_cmd`（自带的"快速复现"）**开箱即崩**：τ=0.8 默认路径在 D=8 上 u_mean=0.25。
`refs.json:repro_cmd = --grids 20 40`（默认 τ=0.8 → D=20 时 u_mean=0.1、τ=0.8；D=16 时 u_mean=0.125）**与交付工件参数完全不同**（工件 `lattice_units.u_mean=0.05, tau=0.62`，只能来自 `--u-lb 0.05`）。
另：m15411 审计已实测同机 CPU 档 300 步 >6 min（128 核 load≈628），`--grids 40` 300k 步在本机不可行 → 该 repro_cmd 在本宿主上**既不匹配又不现实**。

### 4.4 报告六段 —— 缺席（F5）

`make_report.py` 存在但从未运行：`docs/CASE_CYLINDER_DRAG_RE20.md` **不存在**，`docs/assets/case_cylinder_drag_re20/` **空目录**，
`SELECT ... FROM tensor_reports WHERE mission_id=15777` → **0 行**（该表存在 14508（m15411 报告）与 14830（m15411 审计），**没有 m15777**）。
m15777 自己的 headline 也承认"平台报告未发"。

---

## 5. 判级（② L0–L5）

| 级 | 判据 | 本方结论 |
|---|---|---|
| L0 文档 | — | 超 |
| L1 数学/单元验证 | — | 超（`tests/test_cylinder_st2d1.py` 7 条，含锚点 pin sha256 一致性、`compare_to_reference` 语义、Ma 护栏） |
| L2 独立功能实现 | 真代码 输入→计算→输出 | ✅ **成立**：451 行 torch MRT + Zou-He + 半程反弹 + Ladd 动量交换；工件可由原始序列重算（差 2.3e-6）；真流场（散度 rms 1.1e-3） |
| L3 系统集成 | 接入主求解器/平台 | ✅ **成立**：`app/backend/routers/cylinder_bench.py:128-155` 新增 `POST /api/cylinder-bench/st2d1/validate`，**委派** `tensorlbm.cylinder_st2d1.run_cylinder_st2d1`；`app/backend/main.py:37,200` 注册 `/api/cylinder-bench`；本方用 `fastapi.testclient` 起 app 实测该路由**存在**（路由表 9 条 cylinder 相关，含 `/api/cylinder-bench/st2d1/validate`）；包级 lazy export 4 项已入 `__all__`/`TYPE_CHECKING` |
| L4 基准验证 | 完整算例 + 误差分析 + 可复现 | ❌ **不成立**：① 主量 Cd **+5.246 % 超 5 % 门**；② `tensor_benchmark_cases` **0 行**（机器裁判缺席）；③ **单网格**（D=16）无网格收敛/敏感性分析，D=20 被 kill 无工件；④ 交付工件无法由声明命令复现（F3/F4）；⑤ 六段报告缺席（F5）；⑥ 二阶量 Cl −394 % 未验证 |
| L5 科研工程成果 | — | 不适用 |

**判级 = L3**（与同锚点先例 m15411 口径一致；本单在"CPU 可复现 + API 端点 + 测试守护"三点上有实质增益，但基准结论仍 FAIL，**不进 L4**）。

---

## 6. 发现清单（按严重度）

| # | 严重度 | 发现 | 证据 / 复现 |
|---|---|---|---|
| F1 | **高·阻断 L4** | 唯一交付网格 Cd 超门 +5.246 %；诚实自报 FAIL | 工件 `results.json:verdict.rows[0]`；本方从 `series.csv` 重算 +5.24641 % |
| F2 | **高** | 唯一机器裁判 `tensor_benchmark_cases` 无该案号（0 行）→ L4 结构上不可裁定 | `SELECT count(*) ... anchor_slug='CASE-CYLINDER-DRAG-RE20'` → 0 |
| F3 | **高** | `quick_cmd` 开箱 rc=1；`repro_cmd` 参数≠交付参数，不能复现工件 | 本方实跑日志 `runs/audit_m15882/*`；工件 `lattice_units.u_mean=0.05, tau=0.62` |
| F4 | **高** | 工件命名/provenance 与入口契约不符，无 `summary.json`；D20 仅有日志 | 磁盘 vs `run_case.py:117-135` |
| F5 | **中高** | 六段报告缺席（仓内 md 缺 + 平台 `tensor_reports` 0 行）；`make_report.py` 从未执行 | §4.4 |
| F6 | **中** | 台账口径：mission `succeeded/accepted`、顶层 `fail_stage` 空，而 `result.fail_stage='budget_exhausted+forced_finish'`、headline 自述"半成品"、`report_owed=false` | `tensor_dev_missions` id=15777 行 |
| F7 | **中** | 单网格交付 → 任务书⑤"至少两点收敛性/敏感性"未满足；D20 未到平台（drift 3.55e-3） | `D20.log` 末行 |
| F8 | **中** | Cl 符号反、−394 %（与 m15411 同一缺陷，返修**未修**） | 工件 `cl`/`err_cl_pct` |
| F9 | **中低** | API 默认请求把判据跑在容差之外的数值区间：`ST2D1Request(D=12, steps=4000)` + τ=0.8 → u_mean=0.1667 → **Ma=0.289**（仅低于 0.3 护栏 4 %，Ma² 可压缩误差 ≈8 % ≫ 5 % 门） | `cylinder_bench.py:113-116` + `cylinder_st2d1.py:203-218` |
| F10 | 低 | `series.csv` 每 10 步抽取（36000 步 → 3600 行）却无 header 说明，`avg_window` 是"步"而文件是"采样"→ 直读重算会错（本方首轮即踩） | `series.csv` 步长 diff 恒为 10 |
| F11 | 低 | 固壁内速度非零（max\|u\|=4.6e-3，≈9 % u_mean）：`fields.npz` 的速度快照未把固壁格点清零，噪声进图/进散度统计 | 本方从 `fields.npz` 重算 |

**通过项（正面核验，防误伤）**：ref sha256 pin 一致；几何/Re/Ma/固壁格点数自洽；Cd 独立重算一致到 2.3e-6；散度量级正常（真解）；`compare_to_reference` 语义正确（gating/non-gating 分离、stationary 才 PASS）；Ma 护栏真实生效（会拒绝 D=8）。

---

## 7. 返修建议（给下一轮，按性价比排序）

1. **先注册裁判**：`INSERT INTO tensor_benchmark_cases(anchor_slug,case_name,ref_source,metric,tolerance,repo)` VALUES ('CASE-CYLINDER-DRAG-RE20','Schäfer–Turek 2D-1 Re=20','Nabh 1998 谱解 (FeffFlow 官方) / 仓内 benchmarks/verified/cylinder/re20_st/result.json','Cd','0.05','TensorLBM')，并把 Cl 作为非 gating 信息列另记。
2. **把交付网格搬到验证过的参数档**：同锚点已验证档 = τ=0.8 / D=40（Cd=5.54843, −0.558 %）、D=80（−2.875 %）。返修应以 **D≥40、τ=0.8** 为主档（该档在本机 CPU 不可行 → 走 `run_cluster`，禁指定节点），D=16/τ=0.62 只能作趋势低档，不能作交付主档。
3. **工件由入口产出**：让 `run_case.py` 写 `results_D{D}.json` 并在 `latest/`（或 `--label`）下留 `results.json` 别名；`series.csv` 写 header（含抽帧步长）。
4. **repro_cmd 自检**：`refs.json.quick_cmd` 必须带 `--u-lb 0.05`（并加一条 CI 冒烟测试：quick_cmd 在 D=8/300 步 rc=0，覆盖 F3）。
5. **补六段报告**：执行 `make_report.py`（补齐其两处未完成修补）+ 平台 POST，反式注入数字，`figures` 必须带 png_base64（先例 m15411 报告空 base64 已被判缺陷）。
6. **修 Cl**：网格 D=40→80 的 Cl 从 −0.0141 走到 −0.0015，属未收敛态；要么给出 Cl 收敛研究，要么在 `refs.json` 明确"本算例只判 Cd，Cl 不是可验证量"并附证据（现在是两头不靠）。
7. **台账口径**：mission 15777 顶层 `fail_stage` 与 `result.fail_stage` 不一致，建议审计修正为 honest-partial（避免"successed 掩盖 FAIL"）。

---

## 8. 本方实跑/未跑（诚实边界）

**已跑**：
- `psql` 直查 `tensor_benchmark_cases`（0 行）、`tensor_maturity_verdicts`（同锚点先例 id=368）、`tensor_reports`（无 15777 行）、`tensor_dev_missions` id=15777 全行。
- `scripts/audit/m15882_cylinder_audit.py`：从 `series.csv`/`fields.npz` 独立重算 Cd 统计、散度、几何自检 → `runs/audit_m15882/independent_recompute.json`（同内容随提交入库：`docs/audits/CASE-CYLINDER-DRAG-RE20-m15882-recompute.json`）。
- 图：`runs/audit_m15882/audit_cd_convergence.png`（Cd(t)+ref+±5 % 带）、`runs/audit_m15882/audit_field_cloudmap.png`（\|u\| + 涡量云图）。
- `refs.json:quick_cmd` 实跑 → rc=1（F3 的直接复现）。
- `fastapi.testclient` 起 `app.backend.main`：`/api/cylinder-bench/st2d1/validate` 路由存在（集成面核验）。
- `pytest tests/test_cylinder_st2d1.py`（结果见 §9 与判词 `tests` 键）。
- 二次启动 `run_case.py --grids 20 --u-lb 0.05 --stop-on-plateau`（网格敏感性独立探针），结果见 §9。

**未跑 / 不可断言**：
- 未跑 D=40 / D=80 主档（τ=0.8，300k 步）：本宿主 128 核 load≈400，CPU 档单档预估 ≫ 1 h；本方按纪律**不压控制面**，且本方定位为核验单（非重做），故**不宣称**"τ=0.8 档在此模块上仍复现 −0.56 %"——该数字只属于仓内既有锚点（2026-08-19 GPU 跑），本模块在 τ=0.8/D=40 上**未被验证过**（这是 L4 的第 7 条缺口，非本方结论）。
- 未修改任何被审代码（审计只读）。
- 未核 GPU/CUDA 路径（本机 `cuda_avail=False`）。
- 未验证 `run_vorticity`/`fields` 后处理公式（仅用其输出做量级检查）。

---

## 9. 附：本方独立探针结果


### 9.1 `pytest tests/test_cylinder_st2d1.py`（全 7 条，本方实跑）

```
7 passed, 1 warning in 204.85s (0:03:24)      # runs/audit_m15882/pytest.log
```
**优于被审单自述**（m15777 只跑了前 5 条，声称后 2 条未跑）→ 测试面不是缺口，**已全绿**。

### 9.2 API 端点实跑（集成面，本方实跑）

```
routes: [... '/api/cylinder-bench/st2d1/validate' ...]          # 9 条 cylinder 路由
POST /api/cylinder-bench/st2d1/validate {"D":12,"steps":1200} → status 200
  lattice_units: u_mean=0.1667, nu=0.1, tau=0.8, Ma=0.2887, Re=20.0
  cd=8.5862, stationary=false, stop_reason=max_steps, verdict.passed=false
```
→ **L3 集成由"执行"证实**（不是只读代码）：端点真跑真解、真返回 ref-vs-got、非稳态时如实 `passed:false`。
→ 同时**坐实 F9**：端点默认档（D=12、τ=0.8）把判据跑在 **Ma=0.2887**（距 0.3 护栏 3.8 %，Ma²≈8 % 可压缩误差 ≫ 5 % 门），
   默认请求 cd=8.59（瞬态）——端点"能跑"但默认参数**不在可验收区间**，需要一个显式 `--u-lb`/D≥20 的默认值。

### 9.3 网格敏感性独立探针（诚实：**未完成**）

本方二次启动 `run_case.py --grids 20 --u-lb 0.05 --steps 80000 --stop-on-plateau`（网格敏感性独立探针），
在本宿主 128 核 load≈400 环境下进程被沙箱回收（`runs/audit_m15882/D20/` 空、日志仅 1 行），**未产出结果**。
因此本方**不宣称**任何 D=20 平台值；网格趋势只能引被审单自己的证据：D=20 在第 20000 步 Cd≈5.8239（+4.38 %）但 **drift=3.55e-3 未到平台**
（`runs/cylinder_drag_re20/D20.log`），按 `compare_to_reference` 语义 **stationary=false ⇒ 不可 PASS**。F7 成立：单网格交付，任务书⑤未满足。
