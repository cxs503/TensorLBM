# 交差审计 · m18748 — XVAL-LBM-SPARSE-SUBLIST-NEIGHBOR-LIST（交付单 m18377，TensorLBM）

**审计对象** `/root/tp-agent/work/m18377`（分支 `agent/m18377`，mission 18377，status=succeeded，accepted=true）
**锚点** `XVAL-LBM-SPARSE-SUBLIST-NEIGHBOR-LIST` · 对标 `sorush-khajepor/listLBM`（GPLv3）
**审计结论** **L2**（真实现 + 真测试 + 逐位可复现工件）；**L3/L4/L5 均不成立**，理由逐条落表。
**唯一裁判** `tensor_benchmark_cases`：该 anchor **0 行**（全表 110 行）——基准裁判无数据，因此本单的判级**不靠报告好看**，而靠①工件能否被审计者逐位复跑 ②代码是否可 import/被调用。

机读读数：`artifacts/m18748_audit/audit_m18748.json`；图：`artifacts/m18748_audit/figures/fig_m18748_evidence.png`；
审计脚本：`scripts/audit/m18748_sparse_sublist_board_audit.py`（唯一命令，≈90 s，单线程 CPU）。

---

## 1. 三件套取证（run_artifacts / ref-vs-got / repro_cmd）

| 项 | 交付声称 | 本次核验 | 判定 |
|---|---|---|---|
| run_artifacts | `TensorLBM/artifacts/xval_sparse_sublist_neighbor_list/results.json`（1797 B）+ `figures/fig1_channel_profile.png`（76 318 B，1540×560 RGBA，1989 色） | 两件均真实存在、非空、图是真矢量渲染位图 | ✅ 存在 |
| ref vs got | k_ref=10.666667（解析 slab `(H/ny)H²/12`）vs k_got=10.654330，误差 0.1157% | 审计者独立重跑：k=**10.654330145418363**，与工件**相对差 0.0（逐位相同）** | ✅ 可复现 |
| repro_cmd | 交付内**无**任何 runner（`xval/XVAL-LBM-SPARSE-SUBLIST-NEIGHBOR-LIST/` 只有 `predeclared.json`；artifacts 目录下无生成脚本） | 审计者用本单脚本 `scripts/audit/m18748_sparse_sublist_board_audit.py` 复跑得到同一数值 | ⚠️ 工件真，但**产生工件的命令未随交付**（F3） |
| 报告六段 | `docs/xval_sparse_sublist_neighbor_list.md` 7 节 | 六要素齐全（问题/方法/参数表/对照表/收敛敏感性/失败模式），但**未发布**到 `tensor_reports`（mission 18377 0 行，F6） | ⚠️ 文档在，报告纪律缺 |
| 基准裁判 | — | `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-SPARSE-SUBLIST-NEIGHBOR-LIST'` → **空** | ❌ 无判据（F5） |

## 2. 独立复算（数字一律来自本次执行，不抄报告）

配置：D2Q9，τ=0.8，ν=0.1，g=1e-5，通道 H=16 / ny=32 / nx=16，float64，单线程；压缩布局 K=256，φ=0.5。

| 检验 | 交付声称 | 本次实测 | 结论 |
|---|---|---|---|
| R1 物理锚（2500 步） | k=10.654330，误差 0.1157% | k=10.654330145418363（rel 0.0 vs 工件），误差 **0.11565%**；固有口径 21.30866 vs 21.33333 = 0.11565%（φ=0.5，两口径同值） | ✅ <3% |
| R1 剖面形状 | 未报 | 稳态剖面 vs 抛物线最大相对残差 **1.70e-6**；上下对称性 **2.08e-15**；u_max/⟨u⟩=**1.49272**（连续极限 1.5，−0.49%）；uy=1.2e-15 | ✅ 真 Poiseuille |
| R2 收敛梯（2500/5000/10000/20000） | 只跑 2500 步 | k=10.654330 / 10.65499996 / 10.655 / 10.655 → 2500→20000 漂移 **0.0063%**；收敛后误差 **0.1094%** | ✅ 2500 步已收敛，非瞬态粉饰 |
| R3 表示等价（swap vs push，seed 1/5/13，各 50 步） | <1e-14 | **0.0（逐位相同）**，三 seed 全 0；稠密控制臂 k 亦逐位相同（`dense_maxabs_du_intrinsic`=0.0，2500 步后状态差 0.0） | ✅ 严格强于声称 |
| R4 邻居表 | 与仓内 `FluidNodeTable` bit 相同 | 与**独立 numpy 重写**（不复用仓内 builder，排除循环论证）逐项相同：nodes/neighbour/index 全 True，sublist=(0,352)，K=353 | ✅ 表正确 |
| R5 守恒/健壮性 | 漂移 <1e-12 | 200 步最大相对漂移 **4.58e-16**，全场 finite，无 NaN/Inf；随机多孔 φ=0.7266 亦正常 | ✅ |
| Ma | 0.005455 | 5.509e-3（u_max/cs）——压缩性误差 O(Ma²)≈3e-6，不污染 0.11% | ✅ |

**机理核对**（不只看数值）：`stream1`（半方向对交换，遇 `-1` 固体邻居跳过）+ `stream2`（节点内方向对互换）**在数学上等于半程反弹 push 流**——固体侧跳过使 `f*[d]` 留在原位，随后 `stream2` 把它换到 `opp(d)` 槽位，正是 BB 反射；两臂全是纯拷贝无算术，故 0.0 逐位一致是可信的，不是被清零出来的（对比 m17636 的 10× 压缩造假模式）。审计者另行核对 `HALFQS=(1,2,5,6)` 为每对相反方向取一（与 OSS `lattice_mod.f90:87` 一致）、`DIRS=(1..8)` 与 `neighbour[:,d-1]` 槽位对齐、`C` 的 (y,x) 取序正确。

## 3. 级别判定（级别=功能实质，百页报告不加级）

| 级 | 判据 | 本单 | 依据 |
|---|---|---|---|
| L1 | 数学/单元验证 | ✅ | 6 个测试函数（10 用例）本次复跑 **10 passed in 10.56 s**；表等价/算子等价/质量守恒/物理带全绿 |
| L2 | 独立功能实现（真代码 输入→计算→输出） | ✅ | 345 行模块，0 占位符（`NotImplementedError/TODO/pass#placeholder` 计数 0）；审计者仅用 `build_sparse_sublist_neighbor_list`+`listlbm_step`+`macroscopic_with_forcing` 就把 10.654330…逐位复算出来——**数字来自实现而非文档** |
| L3 | 系统集成 | ❌ | `grep -rl 'sparse_sublist' --include=*.py` 全树（排除模块自身/测试）**0 命中**；`cd /root/tp-agent/work/m18377 && python3 -c "import sys;sys.path=['src'];import tensorlbm.sparse_sublist_neighbor_list"` → **ModuleNotFoundError**（canonical `src/tensorlbm/` 无 `porous` 子包，交付写在旧嵌套影子树 `TensorLBM/src/`） |
| L4 | 基准验证（完整算例+误差分析+可复现） | ❌ | ①`tensor_benchmark_cases` 该 anchor **0 行**（唯一裁判无数据）②交付**无 runner**，provenance 靠审计者自建脚本 ③唯一算例是 1 个通道 + 3 seed 随机多孔，无网格/分辨率收敛研究 |
| L5 | 科研工程成果 | ❌ | OSS Fortran 未编译未执行（只做源码级语义移植，审计者已核 `/tmp/listlbm` HEAD=9feb6e35，2020-07-07，pdf/neilist 确在流体区间 `[ist:ind]`）；无文献/实验外部锚（解析 slab 是精确数学，不是外部实验） |

**结论：L2 confirmed 成立，L3–L5 逐项不成立**（不是"证据不足"的含糊，而是有反证：import 失败、grep 0 命中、基准 0 行、无 runner）。

## 4. 发现的缺陷（按严重度；均可复现）

| ID | 级别 | 缺陷 | 证据 |
|---|---|---|---|
| F1 | high | 交付 5 个文件全部 **untracked**（未入 git）：模块/测试/工件/文档/预声明 | `cd /root/tp-agent/work/m18377 && git ls-files \| grep sparse_sublist` = 空；`git status --porcelain` 5 条 `??` |
| F2 | high | **零集成调用方** | `grep -rl 'sparse_sublist' --include=*.py <树>` 排除模块/测试后 0 命中 |
| F2b | high | **canonical 包不可导入**：写在影子树，`import tensorlbm.sparse_sublist_neighbor_list` 从仓根失败 | ModuleNotFoundError（canonical 无 `porous/`，只有 `porous_media.py` 等） |
| F3 | high | **无交付 runner**：`xval/<anchor>/` 只有 `predeclared.json`；results.json 的生产脚本未随交付，第三方无法按交付复现 | `ls xval/XVAL-LBM-SPARSE-SUBLIST-NEIGHBOR-LIST/` = `predeclared.json` |
| F5 | high | **基准裁判无数据**：该 anchor 0 行（全表 110 行） | `SELECT ... WHERE anchor_slug=...` → 空 |
| F6 | medium | **报告纪律**：`tensor_reports` 无 mission 18377 行（报告只在 docs/ 里，平台侧不可检索/不可复核） | `SELECT id,title FROM tensor_reports WHERE mission_id=18377` → 空 |
| F12 | low | 外部锚**行号引用有误**：docs 称 `mainlist_mod.f90:14-20` 是 pdf+neilist，实际该区间是 GPL 头；真实锚点 `:31,:33` 声明、`:110-111` 分配 | `sed -n 14,20p mainlist_mod.f90` → GPL 文本；`allocate(this%neilist(1:nq_1,ist:ind))` 在 111 行 |
| F13 | low | provenance 未钉 commit：`results.json.oss.commit` 只写 `"master (local clone /tmp/listlbm)"`，无 sha（审计者补测 HEAD=9feb6e3592ba5ebbbba1f8850db41f85d61f1489） | results.json:6 |
| F8 | info(+) | 2500 步**已收敛**（→20000 步漂移 0.0063%），报告的 0.1157% 是真误差不是瞬态 | R2 收敛梯 |

**未发现的问题（反面核验也记账）**：无硬编码结果（模块/测试内无 `10.65`/`21.30`/`0.1156` 字面量）；质量不漂（4.58e-16）；剖面是真抛物线（残差 1.7e-6，对称 2e-15）；稠密控制臂与 swap 臂逐位相同（非"两臂共享代码造假"）；`predeclared.json` 有 `written_before_runs: true` 且阈值（bit-identical / 1e-14 / 15%）与实测一致，未见事后调阈。

## 5. 审计独立性与未做的检查

* 被审树 **只读**：未写入/未修改 `/root/tp-agent/work/m18377` 任何文件；导入该树模块、复跑写入本工作树 `artifacts/m18748_audit/`。
* 数字来源：(a) 交付工件静态读取 (b) 交付代码原样执行（R1–R5）(c) 独立 numpy 重写邻居表 (d) `psql` 查 `tensor_benchmark_cases`/`tensor_dev_missions`/`tensor_reports` (e) `/tmp/listlbm` 源码行号核对。
* **未做**（不可声称通过）：OSS Fortran 二进制未编译未执行（锚点仅源码级）；三维/多相/多子表场景未测（交付只到 D2Q9 单相单子表）；未做网格分辨率收敛研究（H=16 单一分辨率）；未跑 GPU/多线程一致性。
* 独立性局限：本审计者与被审单同属一个自治代理体制，"审核"只对**机器可核事实**负责（import/grep/DB/逐位复算），不对研究品味负责。

## 6. 处置（disposition）

**approve with notes → 判 L2 confirmed。** 已落表 `tensor_maturity_verdicts`（repo=TensorLBM, scope=`m18377 XVAL-LBM-SPARSE-SUBLIST-NEIGHBOR-LIST`, level=2, verdict_by=`board-audit`）。

返修方向（若开 successor，不建议以"报告补充"代替）：
1. 把模块搬进 **canonical** `src/tensorlbm/`（连带 `porous.fluid_node_table` 的 canonical 归属问题）；
2. 交付 `xval/<anchor>/run_xval.py`（repro_cmd），并把 results.json 的生产链固化；
3. 在 `tensor_benchmark_cases` 登记该 anchor 的 ref/metric/tolerance 行（否则任何 L4 宣称都无裁判）；
4. 接一个主求解器/平台调用方（L3）；
5. 提交 commit（当前全部 untracked → 平台推分支时**会整包丢失**）。