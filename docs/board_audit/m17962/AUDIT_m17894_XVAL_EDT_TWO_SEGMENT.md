# 交差审计 m17894 — XVAL-LBM-EDT-TWO-SEGMENT-GEOMETRY-TAG-COMPRESSION（TensorLBM）

审计单：m17962（board-audit，非重做）｜被审单：m17894（succeeded，report id 16822）｜
同锚点前序：m17728（→ 审计 m17829，verdict id=643，L2，gate=**reopened**，原因＝evidence 指针占位/缺失）
裁判：`tensor_benchmark_cases`（唯一）+ `tensor_case_pool`（辅助）→ **两表均 0 行** ⇒ 无登记裁判，L4 不予。

## 1 案例介绍（被审交付是什么）

MPLBM-UT（je-santos，GPL-3.0）的岩心几何预处理链：`rock/3` → `erase_regions` → `create_geom_edist`
（scipy 精确 EDT）→ 二段分割（`0<d<2` 为 boundary，`d>=2` 为 interior）→ Palabos `int16` 码
2608/2609/2610/2611 → `.flatten().tofile()`。锚点要求 TensorLBM 复刻该链，并提出「标签压缩 + 两段几何」的
可验证扩展。被审交付（commit `dd38f652`，分支 `agent/m17894`）含：

| 组件 | 位置 | 规模 |
| --- | --- | --- |
| 库实现 | `src/tensorlbm/edt_geometry_tags.py` | 711 行：k-d tree 精确 EDT、`scipy_edt` 对照、二段分割、Palabos 码、`num_slices`/`swap_xz`/nw 种子、per-grain id 侧信道、2 bit/体素无损打包、`.dat`+JSON sidecar 写盘、解析胶囊/两段几何与解析距离场 |
| 求解器适配层 | `src/tensorlbm/preprocess_geo.py:908` `palabos_geometry_tags` | mask → `torch.int16` 标签 + `float32` 深度 + meta |
| 物理扩展 | `src/tensorlbm/multiphase3d.py:1324` `wall_adhesion_force_3d` | Shan-Chen 壁面力，逐颗粒 `G_wall`（混合润湿） |
| XVAL 入口 | `tools/xval_edt_two_segment.py` | 888 行，6 算例（L/V 两段、胶囊、平板缝、三球堆、nw+padding），27 条判据 |
| 测试 | `tests/test_edt_geometry_tags.py`、`tests/test_xval_edt_two_segment_runner.py` | 35 用例 |
| 运行工件 | `runs/xval_edt_two_segment/{summary.json,table.csv,report.md,geometry/*.dat+*.json,figures/fig1-4.png}` | 25 文件 / 1.8 MB |
| 六段报告 | `docs/xval/EDT_TWO_SEGMENT_GEOMETRY_TAG_COMPRESSION.md` | 219 行 |

审计保全（被审 worktree 可被回收）：`docs/board_audit/m17962/preserved_m17894/`（MANIFEST sha256）。

## 2 方法（怎么核的，判据先于动手）

1. **三件套存在性**：run_artifacts 文件落盘与大小；`summary.json` 的 ref/got 数值与 `table.csv` 逐行；
   `repro_cmd` 实跑。
2. **复现**：在 m17894 worktree 内 `PYTHONPATH=src python tools/xval_edt_two_segment.py --out /tmp/m17962_repro
   --no-figures`，与交付 `summary.json` 逐字段比对（`tag_agreement`、`tag_disagreement_count`、
   `dist_max_absdiff_edt_algorithms`、`packed_bytes`、6 算例 × 27 判据、`all_checks_pass`）。
3. **测试**：`python -m pytest tests/test_edt_geometry_tags.py tests/test_xval_edt_two_segment_runner.py -q`。
4. **裁判检查**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-EDT-TWO-SEGMENT-GEOMETRY-TAG-COMPRESSION'`
   → 0 行；`SELECT ... FROM tensor_case_pool WHERE linked_anchor=...` → 0 行；全表 anchor 清单旁证该 anchor 未登记。
5. **独立性对抗**：对「agreement=1.0」类判据追问 ref 与 got 是否用同一原语；对 32× 压缩追问是否落盘。
6. **交付耐久性**：`git status`、`git log --all`、生态克隆 `/root/tensor-ecosystem/TensorLBM` HEAD `be78a87b`
   是否含该模块。
7. **判据门槛**：平台 15%（不可接受线）、生态 <3%（论文线）；级别按功能实质（L1 验证 / L2 真实现 / L3 集成 /
   L4 基准 / L5 科研工程）。

## 3 结果（数字）

**(a) 三件套核验**

| 项 | 核验结果 |
| --- | --- |
| run_artifacts | 25 文件实存并已入 commit；`summary.json` 21663 B、`table.csv` 3101 B（27 数据行）、`report.md` 138 行、4 张 PNG 32–75 KB、6 组 `.dat`+`.json`（int16 文本 + sidecar） |
| ref vs got 数值 | `table.csv` 27 行逐行有 ref/got/误差/门槛/通过位；`all_checks_pass=True`；顶层 `tag_disagreement_count=0`（6 算例） |
| repro_cmd 可执行 | 实跑成功，**6/6 算例逐字段与交付相等**（无 DIFF：`tag_agreement`、0 不符体素、EDT 互检 0.0、`packed_bytes`、`all_checks_pass=True`、conv slope 1.81 复现） |
| 测试 | **35 passed / 5.69 s**（本单复跑，rc=0；仅 `test_edt_geometry_tags.py` 单独跑 32 passed / 7.43 s） |

**(b) 27 条判据的误差分布（本单从 `table.csv` 程序化统计，零手抄）**

- 相对误差行 **24** 条：**≤3% 22 条**，**3–15% 2 条**（V 形近界面深部相对误差 9.522%、胶囊深部 3.597%），
  **>15% 0 条**。
- 绝对偏差行 **3** 条（EDT 深度最大绝对偏差，cells）：1.1231 / 1.2620 / 0.4340，门槛 1.5 cells → 全过。
- 唯一超 <3% 的物理项：V 形凹角 `d≈2` 近界面相对深度 9.52%，成因是点阵 EDT 度量「到最近流体体素中心」而
  解析场度量「到连续表面」，绝对偏差 ≤1.26 cells；同算例深部（d>4）2.64%。报告 6e 已显式声明「否（<15% PASS）」，
  未粉饰 ⇒ 与平台「>15% 不可接受」一致，且未把该项自报成 <3%。

**(c) 裁判表**

| 表 | 查询 | 结果 |
| --- | --- | --- |
| `tensor_benchmark_cases` | `anchor_slug='XVAL-LBM-EDT-TWO-SEGMENT-GEOMETRY-TAG-COMPRESSION'` | **0 行** |
| `tensor_case_pool` | `linked_anchor=同上` | **0 行** |

⇒ 交付里的 15% 门槛来自 runner 自设 + 平台底线，**不是**裁判表登记的容差；按「基准判据唯一裁判」纪律，
L4（基准验证）不成立，尽管误差表自身是干净的。

**(d) 交付耐久性**

| 项 | 结果 |
| --- | --- |
| m17894 worktree | `git status` **干净**，交付在 commit `dd38f652`（25 文件） |
| 生态克隆 | 分支 `agent/m17894` 含 `dd38f652`；**main HEAD `be78a87b` 不含** `src/tensorlbm/edt_geometry_tags.py`（`ls` 确认不存在）⇒ 生态内零可达消费者 |
| 集成调用方 | `grep -rn palabos_geometry_tags|wall_adhesion_force_3d --include=*.py` → 命中仅 `tests/test_edt_geometry_tags.py`（+ docstring 引用），无求解器/算例调用 |

**(e) 级别判定：L2**

- L1 成立：35 用例（k-d tree EDT vs scipy、解析距离场闭式、标签码/分割、压缩往返、IO、壁面力解析闭式）。
- L2 成立：真代码 input→compute→output，入口 CLI 实跑，工件落盘，复现逐字段相等。
- L3 不成立：适配层 `palabos_geometry_tags` 与 `wall_adhesion_force_3d` 无任何非测试调用方；主仓 `be78a87b`
  无该模块 ⇒ 生态内无集成路径（同 m17673/m17728 判例）。
- L4 不成立：裁判表 0 行；且「vs OSS 位级一致」的独立性有限（见 5-F3）。
- L5 不成立：外部锚为 pin 的 commit，仓内无 vendor 快照，ref 是自撰转写、不可执行复核。

## 4 结果图

`docs/board_audit/m17962/fig_audit_m17894_overview.png`（本单生成，matplotlib，数据源 `table.csv`）：
左＝24 条相对判据误差（对数轴，3%／15% 参考线，红/橙/绿分级）＋ 3 条绝对偏差注记；右＝L1–L5 判据状态阶梯。

被审交付的场图（已核，未重绘）：`preserved_m17894/runs_xval_edt_two_segment/figures/fig1_class_map.png`
（标签与二段分割）、`fig2_distance_field.png`（解析 vs 点阵深度 + 差场）、`fig3_convergence.png`（体积误差
log-log，斜率 1.81）、`fig4_sensitivity.png`（depth_split / 网格敏感性）。

## 5 注意点（发现，分级）

| ID | 严重度 | 发现 | 证据 |
| --- | --- | --- | --- |
| F1 | 阻塞（耐久，非级别） | 交付未合入 main：`be78a87b` 无 `edt_geometry_tags.py`，仅分支 `agent/m17894` 有 `dd38f652`；worktree 回收后若不推送，能力在生态内不可达 | `git branch -a --contains dd38f652`；`ls /root/tensor-ecosystem/TensorLBM/src/tensorlbm/` |
| F2 | 阻塞（L4） | 裁判缺位：`tensor_benchmark_cases`／`tensor_case_pool` 对该 anchor **均 0 行**；15% 门槛系 runner 自设 + 平台底线 | 上述两条 SQL；`summary.json:threshold_pct=15.0` |
| F3 | 中 | 「vs OSS 位级一致」独立性有限：ref 是 runner 内**自撰转写**（非执行上游代码），分割规则/码值同样取自 OSS ⇒ 该判据度量的是转写保真度；真正的独立证据是 k-d tree vs scipy 互检（max diff 0.0）、解析几何分类一致率 0.9953/0.9929/0.9724、解析体积 2.60%/1.94%、conv slope 1.81 | `tools/xval_edt_two_segment.py:96-200`（`oss_create_geom_edist`）；`edt_geometry_tags.py:125-195`（cKDTree） |
| F4 | 低（措辞） | 「32× 存储下降」是编解码器级属性：落盘 `.dat` 仍是 `int16`（4×），2 bit/体素类图以 hex 存进 JSON sidecar。报告 4b 同时列 int16 B 与 2-bit B（诚实），但 headline 易被读成端到端 32× | `edt_geometry_tags.py:671-711`（`write_palabos_geometry` → `<i2`）；报告 4b 表 |
| F5 | 低 | 锚点重复派单：m17728（08:22）→ m17829 审计（L2, gate reopened 因指针占位）→ m17894（10:00）重做同一锚点；三单产出同一能力，平台侧应按 reopen 返修而非新单重做 | `tensor_dev_missions.id IN (17728,17829,17894)` 三行 succeeded；verdict id=643 |
| F6 | 低（记录） | 生态摩擦（被审单自述）：`skimage` 与 numpy 2.4 ABI 不兼容 → OSS `erase_regions` 用 scipy 等价实现替代（差异已在 6c 列表登记，测试固化二值图上 no-op）；`raw.githubusercontent.com` 被 reset，源码经 API 取回且无 vendor 快照 | 报告 6c/6f；`oss_erase_regions`（`tools/xval_edt_two_segment.py:104`） |

**报告六段核验**：①案例/几何/参数 ②方法 ③结果 ④对照表+图 ⑤敏感性/失败模式 ⑥误差判定（显式「是否<3%」）
—— 六段齐全（文档行 7/34/56/69/114/143/195），另有 6c OSS 差异清单、6d 移植取舍、6f 生态摩擦，超出模板要求。

## 6 误差判定（逐项回答是否 <3%）

| 关键量 | 交付误差 | <3%？ | 审计意见 |
| --- | --- | --- | --- |
| Palabos 标签 vs OSS 转写（6 算例） | 0 体素不符 | 是 | 成立，但为同规则转写的保真度（F3） |
| k-d tree EDT vs scipy EDT | max diff 0.0 cells | 是 | 成立（两独立算法，本单代码复核确认非 shortcut） |
| 二段分类 vs 解析几何 | 0.474% / 0.710% / 2.756% | 是 | 成立 |
| 深部（d>4）相对深度误差 | 0.367% / 2.640% | 是 | 成立 |
| 胶囊/三球堆体积 vs 解析 | 2.597% / 1.943% | 是 | 成立，且收敛率 1.81（≈2 阶）自洽 |
| 平板固相分数 | 0.000% | 是 | 成立（精确） |
| V 形近界面（d≈2）相对深度 | 9.522% | **否** | 已诚实登记，属点阵 EDT 体素中心度量本质（绝对偏差 1.262 cells），<15% 判据内 PASS；不构成实现缺陷 |
| 物理层 Cd/St（若有人想引「LBM 结果」） | 未测 | — | 本锚点是**几何预处理**，无流动解；不得引作水动力精度证据 |

**Disposition**：approve with notes → **L2**（append 到台账，gate 机器复核 evidence 指针）。
不予 L3/L4/L5 的理由见 3(e)；F1/F2 为阻塞项但属交付/生态流程，不改变功能实质级别。

## 7 本单做了什么（可审计）

1. 只读取证 m17894（worktree / DB / 生态克隆），未改被审代码。
2. 复跑 `repro_cmd`（→ `/tmp/m17962_repro`）与 35 用例，逐字段比对。
3. 保全被审交付到 `docs/board_audit/m17962/preserved_m17894/`（含 MANIFEST sha256，防 worktree 回收灭证）。
4. 程序化统计 27 条判据误差分布、生成审计图、写本报告、落 `tensor_maturity_verdicts`（L2, verdict_by=board-audit）。