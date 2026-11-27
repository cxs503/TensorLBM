# 交差核验：m16321 — CASE-LBM-SQUARE-CYL-NORBERG（TensorLBM）

- 审计任务：m16351 · 仓库 TensorLBM · 分支 `agent/m16351` · 审计席位 board-audit
- 被审任务：m16321（claim: tp-glm52-2, finished_status=succeeded, 2026-11-12 12:46）
- 被审提交：`97e24985` *m16321: CASE-LBM-SQUARE-CYL-NORBERG Re=200 blockage-1/20 square cylinder …*
- 性质：**核验，非重做**。以下每个数字都由本审计独立重算或从注册裁判行直读，不引用被审报告的自述值。

---

## 0. 结论先行（honest）

| 项 | 结论 |
|---|---|
| 三件套是否真存在 | **run_artifacts 存在**（本地 worktree；但被 `.gitignore:108 runs/` 排除，不在交付分支里）· **benchmark ref vs got 存在**（裁判行 id=33）· **repro_cmd 可执行**（本审计实跑，returncode=0，产出 result json） |
| 注册裁判裁定 | **partial(cd pass, st fail)** —— Cd +7.1%/+7.7%（<10% ✅）、St +29.2%/+31.2%（<8% ❌） |
| 头部数字可复现性 | **完全复现**：D=8/D=10 的 Cd、St（FFT 峰）、St（过零）与工件 json **逐位一致（Δ=0.00%）** |
| 成熟度判级 | **L2**（拒绝 L3：无主求解器/平台 API 调用者；拒绝 L4：注册裁判 St 未达标） |
| 关键增量发现 | 裁判值 `St=0.13` 本身是 **WITHHELD 二级转述**，且被本仓自有的**可溯源 2D 数值簇**（`benchmarks/verified/square_cylinder/REFERENCE_AUDIT.md`：Re=150 → St 0.159–0.162，Re=250 → 0.16–0.17）**反证为 Re≈200 处偏保守/口径错**；got 0.1679/0.1706 落在 0.165 附近 **+1.8%/+3.4%**。即：**"St FAIL" 一半是求解器偏差、一半是裁判口径缺陷**，两者都必须记，不能只用其中一半消账 |

---

## 1. 三件套取证（逐项，可指针）

| 验收件 | 位置 / 证据 | 判定 |
|---|---|---|
| run_artifacts | `runs/m16321/sqcyl_norberg/D8/{result_D8.json, series_D8.npz, state.npz}`、`…/D10/{result_D10.json, series_D10.npz, state.npz}`、`…/figs/fig{1,2,3}*.png`；sha256 清单 `artifacts/audit_m16351/manifest.json`（字节副本 `artifacts/audit_m16351/bytecopies/`） | **存在**；⚠ **未入交付分支**（`.gitignore:108 runs/`，`git check-ignore` 确认）→ 证据只活在被审 worktree 里，臂支推送后即失传 |
| benchmark ref vs got | 裁判行 `tensor_benchmark_cases id=33`：`case_name=square_cyl_re200_blockage20`、`metric=St=fD/U ; Cd=<Fx>/(0.5 rho U^2 D)`、`tolerance=St<8% ; Cd<10%`、`ref_source=Norberg 1993 … [WITHHELD: paywalled, values secondary: St=0.13, Cd=1.5]`、`status=partial(cd pass, st fail)`（`artifacts/audit_m16351/referee.json` 原样落盘） | **存在**（该行由 m16321 本轮新建，此前 0 行——记其功，也记其自定裁判的口径风险，见 §4 F2） |
| repro_cmd 可执行 | 实跑：`python3 cases/square_cylinder_norberg/run_chunked.py --D 8 --steps 300 --domain-x 20 --domain-y 20 --x-center 6 --sponge-d 6 --budget 60 --threads 4 --warmup-frac 0.25 --state <tmp>/state.npz` → `returncode=0`，8.35 ms/step，产出 `result_D8.json`（`artifacts/audit_m16351/repro.json` + `repro_result_D8.json`） | **可执行**（注：本次为 300 步短程冒烟；全量 D=10 复现按报告文档需 ~6×100 s 分块） |

---

## 2. 注册裁判逐项裁定（唯一裁判）

`Cd_ref=1.5 (tol 10%)`、`St_ref=0.13 (tol 8%)`，容差与裁判行**逐字一致**（本审计未改判据）。

| 网格 | 步数 / 分析窗 | Cd got | Cd 误差 | <10% | St got (FFT) | St 误差 | <8% | St (过零) |
|---|---|---|---|---|---|---|---|---|
| D=8 (160²) | 14000 / 后 50% | 1.6067 | **+7.11%** | ✅ PASS | 0.1679 | **+29.15%** | ❌ FAIL | 0.1690 |
| D=10 (200²) | 24000 / 后 25% | 1.6159 | **+7.73%** | ✅ PASS | 0.1706 | **+31.23%** | ❌ FAIL | 0.1687 |

（本表由 `artifacts/audit_m16351/judge.json` 程序化生成；与裁判行 `status=partial(cd pass, st fail)` 一致 → **锚点整体不通过**。）

---

## 3. 独立重算（不 import 仓内后处理，自己写 FFT/过零）

用 `series_D{8,10}.npz` 原始力历史，从零实现 Hann 窗谱峰 + 抛物线插值 + 滞回过零，与被审 json 对比：

| 网格 | 样本 | 涡脱周期数 | Cd | St(FFT) | St(过零) | 与工件偏差 |
|---|---|---|---|---|---|---|
| D=8 | 7000 | 7.35 | 1.6067 | 0.1679 | 0.1690 | **Δ = 0.00% / 0.00% / 0.00%** |
| D=10 | 4500 | 3.84 | 1.6159 | 0.1706 | 0.1687 | **Δ = 0.00% / 0.00% / 0.00%** |

→ 头部数字不是手抄、不是拼装，**可由工件独立重算**。这是本单最硬的一条证据。

> 分辨率附注（本审计新增）：D=10 的 FFT 单 bin 宽度 = 0.0444（= St 的 26%），插值前裸峰在 0.1778、过零法 0.1687——两法相差 5.4%，说明 **D=10 的 St 精度主要靠抛物线插值兜住，统计窗仅 3.84 周期偏短**；D=8（7.35 周期）才是可信档。两档都远离 0.13，故"FAIL 非噪声"的结论方向不受此影响。

---

## 4. 发现（按严重度排序，均附可复现指针）

**F1｜Medium（证据耐久性）— 工件被 gitignore，交付分支不含证据。**
`git check-ignore -v runs/m16321/sqcyl_norberg/D8/result_D8.json` → `.gitignore:108:runs/`；`git ls-files cases/square_cylinder_norberg` 只有 README/make_figures/run_chunked/refs。交付 commit 只带代码不带数字 → 后来者必须重跑 ~6×100 s 才能复核，且状态 `.npz` 不可找回。**这是平台级通用缺陷（不是本单特有）：evidence 生成路径落在 `runs/` 就等于没交。**
*修复方向*：`cases/<case>/results/` 或 `artifacts/<anchor>/` 落小体积 result json + 图（npz 可留 runs/ 并给 sha256）。

**F2｜Medium（裁判口径）— 注册基准值 St=0.13 与仓内可溯源口径矛盾，且自身 WITHHELD。**
裁判行 `ref_source` 自述 `WITHHELD: paywalled, values secondary`（源自 m16247 的二手归因）。而本仓 `benchmarks/verified/square_cylinder/REFERENCE_AUDIT.md:14,26-33` 记录：Okajima 1982 实验 Re=110 → St 0.143–0.145、Re=250 → 0.140–0.143；2D 自由流数值 Re=150 → St 0.159–0.162、Re=250 → 0.16–0.17。→ Re=200 处 **St 0.13 更像 Re≈80 的实验值，不是 Re=200 的 2D 自由流数值口径**；got 0.1679/0.1706 对可溯源的 0.165 只差 **+1.8%/+3.4%**（`judge.json:context_convention`）。
*处置建议*：由 owner 侧对 `tensor_benchmark_cases` 该行做**口径换锚**（ref 改为可溯源 2D 数值簇并写明 Re 依赖），换锚后本算例 St 判据应在容差内；**换锚前不得给 L4**——本条不是"报告好看就改判"，必须走裁判行修订 + 重跑登记。

**F3｜Low（报告一致性）— 报告 §④ 交叉验证列 D=8 写 0.1686，工件 json 为 0.169，独立重算 0.1690。**
差 0.24%，不影响结论；但"零手抄"要求下，报告表格应与 result json 程序化注入（其余数字均已一致）。

**F4｜Low（可复用性/回归）— 无本案例专属 pytest，回归靠人工重跑。**
`git grep -n square_cylinder_norberg tests/` → 0 命中。`analyze_forces` 是从 `run_case` 抽出的新公共函数，但没有单测锁住其行为（FFT 峰、过零、`min_samples` 边界）。抽取本身正确（默认参数保持 Re=100 原行为：`domain_x_d/domain_y_d/sponge_d/x_center_d` 默认值 = 原常量），但缺回归网。

**F5｜Info（可证伪的正面事实）— 被审自述的"符号缺陷修复"可验证**：`run_chunked.py` 现取 post-stream / pre-bounce-back 采样；本审计执行复现时 Cd=+1.578（正号），未复发负号。

---

## 5. 判级论证（L0–L5，级别=功能实质）

- **L1 数学/单元验证**：达标（MRT/D2Q9 单元面在仓内既有）。
- **L2 独立功能实现**：**达标**。真实代码 `cases/square_cylinder_norberg/run_chunked.py` + 复用 `src/tensorlbm` 主链（`collide_mrt` / `stream` / `far_field_bc_2d` / `make_sponge_strength` / `compute_obstacle_forces`），输入→计算→输出完整，两档网格真跑，头部数字独立可复算。
- **L3 系统集成**：**未达标**。唯一"集成"是 `run_chunked.py` 从 `benchmarks/verified/square_cylinder/run.py` import `analyze_forces/square_mask`（代码复用，方向为 case→harness），**没有**主求解器或平台 API 调用本 case 的入口；裁判行入库是数据登记，不是能力接入。
- **L4 基准验证**：**未达标**。完整算例+误差分析+可复现三项齐，但注册裁判 St 两项均超 8% 容差（+29.15%/+31.23%），裁判行 status=partial。**基准判据是唯一裁判，报告与解释不改变裁定**；F2 说明裁判本身也可疑，但那是"换锚"而非"免判"。
- **L5 科研工程成果**：未达标（无新方法/新结论级别的贡献，且基准未过）。

→ **level = 2**。附带条件（返修单应包含）：①裁判口径换锚评审（F2）；②工件入交付分支（F1）；③补 `analyze_forces` 单测（F4）；④报告 §④ 数字改为程序化注入（F3）。换锚+重登记通过后，本算例具备直接升 L4 的全部条件（算例/误差分析/可复现已就位）。

---

## 6. 未做 / 局限（not_run，不得当作 passed）

- 未重跑全量 D=8/D=10（约 390 s + 226 s 墙钟）；复现只做 300 步冒烟以证 repro_cmd 可执行 —— 复算用的是被审留存的原始力历史（已 sha256 固化），属"证据可重算"而非"算例可重跑"。
- 未在集群（loong64）复算；未做 D=12/D=16 三重网格或阻塞比扫描（被审已诚实声明 not_run）。
- 未获取 Norberg 1993 原文（闭源）；F2 的对标用的是**本仓 REFERENCE_AUDIT 引用的 arxiv 表格**，未独立下载核验（属仓内既有取证，跨 Re 外推 0.165 为线性夹逼估计，标为 context 不计分）。
- 未审计 `run.py` 重构对 Re=100 已验证算例的**数值**等价性（只做了默认参数静态比对，未跑 Re=100 复算）。

---

## 7. 处置

**request changes（返修，不消账）**：level=2 已落表；L4 待 F2 换锚 + 重登记后复评。被判任务"Cd 达标、St 未达标"的诚实申报应予肯定（未粉饰），但按裁判行裁定**锚点整体不通过**，交差不得记全功。

工件索引：`artifacts/audit_m16351/{manifest,referee,recompute,repro,judge,verdict}.json`、`bytecopies/`、`docs/board_audit_m16351/figs/`。