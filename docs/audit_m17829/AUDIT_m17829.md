# m17829 交差核验：m17728「XVAL-LBM-EDT-TWO-SEGMENT-GEOMETRY-TAG-COMPRESSION」

- 被核对象：m17728（TensorLBM，commit `4af97ef7`，branch `agent/m17728`，任务态 succeeded）
- 核验单：m17829（本单，非重做；取证 → 判级 → 落表）
- 审计性质：独立重跑 + 独立追加对抗臂 + 基准裁判表对照
- 审计工作目录：`/root/tp-agent/work/m17829`；被核工作树：`/root/tp-agent/work/m17728`
- 重跑隔离副本：`/tmp/audit_m17829/repro`（`src/` 与 `xval/…` 由 `4af97ef7` 树拷贝，未改动被核工作树）

---

## 0. 一句话结论

**判级 L2**。交付的 EDT 两段打标 + int16 调色板压缩是**真代码**，24 个测试全绿，与上游 MPLBM-UT
`create_geom_edist` 的 19 组 parity 臂**逐体素 0 不一致**且被本次独立重跑**逐位复现**（K 差 0.0）；
但 **① 无任何求解器/平台调用方（L3 缺）**、**② 基准裁判表 `tensor_benchmark_cases` 本 anchor 0 行**、
**③ 唯一端到端水力算例 N=16/24/32 三档超 15% 容差（18.55/18.73/28.13%）只有最细 N=48 的 12.48% 过线**、
**④「tag 压缩省算力」这一核心卖点被自身计时否定（tagged 每步比 plain 慢 12.8%）**、
**⑤ 平台报告 `tensor_reports` mission 17728 = 0 行**。L4/L5 不成立。

---

## 1. 取证：三件套是否真存在

| 项 | 结论 | 证据 |
|---|---|---|
| run_artifacts 文件 | **存在**（10 个） | `xval/edt_two_segment_geometry_tag_compression/runs/{parity,flow,compression,sensitivity}.json` + 5 个 `*_tags.npz`；另有 `summary.json`(38.8 KB)、`table.csv`、`figures/*.png` ×4 |
| benchmark ref vs got | **存在且可复算** | `summary.json:L1_parity`/`L3_flow`；本次重跑见 §3，`max|ΔK|=0.0` |
| repro_cmd 可执行 | **可执行（本次实跑）** | `cd <xval>/ && PYTHONPATH=<repo>/src python3 run_xval.py` → 381.2 s，写出 `summary.json`（`created_utc=2026-11-13T09:31:27Z`，与交付方 `08:47:16Z` 不同 → 确属新跑，非复制件） |
| 上游锚真伪 | **真** | `xval/…/oss_reference.py` 为 `je-santos/MPLBM-UT:src/python/mplbm_utils/pore_utils.py::create_geom_edist` 原文（唯一改动=argparse Namespace→关键字参数）；与本地上游快照 `/tmp/mplbm_src/src__python__mplbm_utils__pore_utils.py` 的函数体逐行一致（该快照含 `create_nw_fluid_mask` 等相邻函数，非为交付裁剪）。**离线限制**：本单网络出口被封（curl 被安全策略拦截），无法在线核对 GitHub 原文 sha256 → 该条按"本地快照一致"计，标注 UNKNOWN 而非通过 |

**基准裁判（唯一裁判）**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-EDT-TWO-SEGMENT-GEOMETRY-TAG-COMPRESSION'` → **0 行**。
即该 anchor **没有登记过机器判据**，报告里"通过/未通过"只能由交付方自己声明的阈值（parity=0%、K 差<1%、K vs Hasimoto<15%）来裁——本单按该声明阈值核，并显式记一条平台缺账。

**报告六段**：`docs/XVAL-LBM-EDT-TWO-SEGMENT-GEOMETRY-TAG-COMPRESSION.md`（143 行）六段齐全
（①案例②方法③结果④图⑤坑/局限⑥误差判定<K15%?>），且 §⑥ 自报 FAIL 处（N=48 12.48% 判 ✗ 的表述与表格一致）。
但 **`tensor_reports` 里 mission 17728 报告 0 行**（只查到 8284 的两条 MPLBM-UT 盘点报告）→ 报告纪律欠账，见 F6。

---

## 2. 判据逐项核（基准判据是唯一裁判）

| # | 预注册判据（交付方声明） | got（交付记录） | 本次重跑 | 容差 | 判 |
|---|---|---|---|---|---|
| 1 | L1 tag 与 MPLBM-UT 逐体素不一致率 = 0（19 臂） | 0.000000% | 0.000000%（19 臂全 0） | 0% | ✅ |
| 2 | L1' 本审计追加的 8 臂（交付未覆盖输入） | — | 0.000000% ×8 | 0% | ✅ |
| 3 | L2 tag 分派 vs 全固体反弹 K 差 <1% | 0.106263% | 0.106263% | <1% | ✅ |
| 4 | L2 存储压缩 float64→int16 = 4×；→RLE ≥ 20× | 4.0×；24.6–77.8× | 同（6 例，RLE 无损 roundtrip=True） | — | ✅ |
| 5 | L2「压缩在求解端省钱」（脚本未立判据，文档/docstring 宣称） | 每步 9.26 ms(tagged) vs 8.21 ms(plain) | 同向（比值 0.801，重跑更差） | >1 | ❌ **反向** |
| 6 | L3 最细分辨率 K vs Hasimoto(1959) <15% | N=48: 12.4843% | 12.4843% | <15% | ✅ |
| 7 | L3 **所有**分辨率 <15%（脚本 `flow_ok=max(...)<15`） | False（N=16 28.13% / N=24 18.55% / N=32 18.73%） | False（同值） | <15% | ❌ |
| 8 | L3' 收敛单调性（误差随 N 单调降） | 18.55(N24) → 18.73(N32) → 12.48(N48)，24→32 反向 +0.18pp | 同 | 单调 | ❌ 未单调 |

**逐位复现证据**：`artifacts/audit_m17829/audit_m17829.json → flow.convergence[*].K_rec/K_rep`
——N=16 `3.0830`、N=24 `3.4935469521536437`、N=32 `3.4859`、N=48 `3.753926226978821`，
`K_rec == K_rep`（`max|ΔK| = 0.0`）；mass 漂移 `6.466763904366957e-05 %` 亦逐位相同。
唯一不逐位相同的是墙钟时间（273.3 s vs 381.2 s，CPU 争用），据此得到的 `speedup_plain_over_tagged`
从 0.8867 变 0.8011——**两个数都 <1，结论方向一致**（tagged 更慢），故计时不可靠但结论稳健。

---

## 3. 判级（L0–L5）

| 级 | 判定 | 依据 |
|---|---|---|
| L0 文档 | 有（超出） | 143 行六段报告 + 4 图 |
| L1 数学/单元验证 | **有** | 24 tests 全绿（本次独立实跑 24 passed / 0.94 s）；`test_two_segment_split_matches_bruteforce_edt` 用暴力距离 oracle 校验分段语义（不依赖 scipy EDT 判据）；19 臂 parity + 本单 8 臂 parity 全 0 |
| L2 独立功能实现 | **有（本判定落点）** | `src/tensorlbm/geom_edt_tag.py:105` `edt_two_segment_tag()` 输入 rock→EDT→两段→int16 调色板→可 pad/swap/回填；`compress_tags/rle_encode/rle_decode/tag_stats/compression_report` 真实现；工件 `runs/*.npz` 落盘；RLE 无损 roundtrip 6/6 |
| L3 系统集成 | **缺** | 全仓 grep `edt_two_segment_tag|geom_edt_tag` 在 `src/`、`app/`、根入口**零调用方**（仅 `xval/`、`tests/`、`docs/`）；`src/tensorlbm/__init__.py` 未导出；commit `4af97ef7` `git branch --contains` 仅 `agent/m17728`，未进主线（本单分支 `be78a87b` 不含该文件）。求解端仅由 xval 脚本自己拼 `stream3d/collide_bgk3d/bounce_back_cells_3d` + 密集网格 `torch.where(interior, nodyn, f)`（`run_xval.py:281-286`）——没有稀疏活跃元列表，grain 体素每步照样被触碰，这正是计时不省反慢的原因 |
| L4 基准验证 | **缺** | `tensor_benchmark_cases` 本 anchor **0 行**（无机器裁判）；端到端唯一水力算例 3/4 档超 15%（§2 行 7）；无场量云图级验证、误差非单调 |
| L5 科研工程 | 缺 | 未进主线、无生产调用方、无 POC |

> 为什么不给 L3 给 L2：L3 的字面要求是"接入主求解器/平台"。本单的唯一"接入"是 xval 脚本自己调用内核，
> 属**验证脚手架**而非产品路径；库侧无入口（无 `load_tagged_geometry()`、无 solver 配置项、无 CLI/API 端点），
> 且未合并主线。判 L2 而非 L1：确是真功能实现 + 真外部锚 parity + 真工件。

> 为什么不再上一层：`docs/…md:19-21` docstring 宣称 grain 段是"pure zero-compute occupancy"，
> 而实测 tagged 每步 9.26 ms > plain 8.21 ms。**存储层压缩（4×，RLE 24.6–77.8×）成立，计算层省钱不成立**，
> 该宣称属超出实测的表述（记 F3，不否定 L2 功能实质，但禁止它在 L4 位置上出现）。

---

## 4. Findings（按严重度）

| ID | 严重度 | 发现 | 证据 |
|---|---|---|---|
| F1 | high(平台) | 本 anchor 在 `tensor_benchmark_cases` 无任何登记判据（0 行），机器门无裁判可比 ⇒ 报告通过与否只能靠交付方自定阈值；同族缺口已在多条同况审计出现 | SQL 直查；`audit_m17829.json` 未列该字段，SQL 见本文件 §1 |
| F2 | high | 「tag 压缩省计算」核心卖点未实现也未测：密集网格 + 每步 `torch.where` 覆盖 grain ⇒ tagged 比 plain 慢（记录 0.8867、重跑 0.8011） | `run_xval.py:281-286`；`summary.json:L2_compression.speedup_plain_over_tagged`；`runs/flow.json` ms_per_step 9.261/8.212 |
| F3 | med | docstring 与报告措辞称 grain 段 "zero-compute occupancy"，与 F2 实测相反（存储层才是省的地方） | `src/tensorlbm/geom_edt_tag.py:14` |
| F4 | med | L2 通过位 `passed=False` **不是**中性判据失败，而是判据写错：`l2_pass = delta<1.0 and thr1["mass_drift_rel_pct"] > 1.0`（把"阈值 1.0 应当炸掉 >1% 质量"当失败模式探测器），实测漂移 6.47e-05% 故恒 False；中性判据 delta=0.106% 其实**通过** | `run_xval.py:433`；`summary.json:L2_compression.passed=false` |
| F5 | med | 端到端水力算例 3/4 分辨率超 15%（18.55/18.73/28.13%），且误差 24→32 非单调；交付方 §⑤ 归因为球面阶梯化（可信：parity=0 证明打标本身无偏差），但**能力不可宣称"求解端答案正确"** | `summary.json:L3_flow`；本单重跑同值 |
| F6 | med | `tensor_reports` mission 17728 = 0 行 ⇒ 平台报告欠账（`xval/…/make_report.py` + `figures.json` 具备，未发 API） | SQL 直查 + 存在 `make_report.py`(19.9 KB)、`figures.json` |
| F7 | low | `speedup_plain_over_tagged = b.ms/a.ms` 命名与语义相反（值<1 表示 tagged 更慢），易被下游误读为"加速" | `run_xval.py:458` |
| F8 | low | 上游锚为本地快照核对，网络出口被封无法在线验 sha256 ⇒ 该链条按 UNKNOWN 记（不是失败，但不计"通过"） | curl 被安全策略拦截 |

---

## 5. 本次审计做了什么（可复现）

```bash
# 1) 交付测试（独立实跑）
cd /tmp/audit_m17829/repro && PYTHONPATH=src python3 -m pytest tests/test_geom_edt_tag.py -q
#   -> 24 passed, 1 warning in 0.94s

# 2) 交付 XVAL 全量重跑（381.2 s，写出新 summary.json）
cd /tmp/audit_m17829/repro/xval/edt_two_segment_geometry_tag_compression \
  && PYTHONPATH=/tmp/audit_m17829/repro/src python3 run_xval.py

# 3) 本审计追加的 8 条对抗 parity 臂（交付方 19 臂未覆盖的输入）
python3 artifacts/audit_m17829/probe_extra_arms.py     # -> ALL_PASS = True（8/8, 0 不一致）

# 4) 记录值 vs 重跑值机器比对 + 出图
python3 artifacts/audit_m17829/build_audit.py          # -> max|ΔK| = 0.0
```

隔离声明：被核工作树 `m17728` 的工件**只读**使用；所有重跑写盘发生在 `/tmp/audit_m17829/repro`。
反向自欺防护：`build_audit.py` 断言重跑 `summary.json.created_utc != 记录 created_utc`，
否则拒绝对比（防"把交付方 summary 拷过来跟自己比"）。

---

## 6. 产物

- `artifacts/audit_m17829/audit_m17829.json`（机器读数：记录 vs 重跑逐项）
- `artifacts/audit_m17829/probe_extra_arms.py` / `probe_extra_arms.json`（8 条独立臂，全 0）
- `artifacts/audit_m17829/build_audit.py`（比对 + 图）
- `docs/audit_m17829/fig_m17829_evidence.png`（parity / 收敛阶梯 vs 15% 线 / 压缩卖点对照）
- `docs/audit_m17829/verdict_m17829.sql`、`verdict_m17829.json`（落表行）
- 落表：`tensor_maturity_verdicts(repo='TensorLBM', scope='m17728 XVAL-LBM-EDT-TWO-SEGMENT-GEOMETRY-TAG-COMPRESSION', mission_id=17829, level=2, verdict_by='board-audit')`

## 7. 未做 / 局限

- 未在线核对 GitHub 原文（出口被封）→ F8 UNKNOWN。
- 未在 loong64 集群复跑（本单为核验单，单机 4 线程已逐位复现；跨架构差异不适用——纯 numpy/scipy/torch CPU 路径，无架构相关浮点断言）。
- 未重跑交付方 4 张图（`make_report.py` 依赖已存在 summary，未重生成；图仅作记录/重跑读数对照，见 fig_m17829_evidence.png）。
- 未评判"该能力是否应合并主线"（属工程决定，不在本单判级范围）。