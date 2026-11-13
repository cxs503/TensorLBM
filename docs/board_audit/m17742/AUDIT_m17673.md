# 交差核验审计：m17673（XVAL-LBM-COMPILETIME-PARAMS-PLUS-RUNTIME-GEOMETRY）

审计单 m17742 · 仓 TensorLBM · 被审单 m17673（succeeded, auto-approve dev_acceptance）
判级 **L2** · verdict id **626** · `verdict_by='board-audit'` · gate_state=pending
证据原始件：`runs/m17742/verify_triple.json`（本单现场执行产出，零手抄）· `docs/board_audit/m17742/verdict_evidence.json`

---

## ① 案例介绍（物理背景 / 几何 / 参数）

被审对象不是物理算例，而是**输入约定互操作**：对标 `joconnor22/LIFE`（GPL-3.0）的
「编译期 `inc/params.h` + 运行期 `input/geometry.config`」两文件约定，在 TensorLBM 侧实现
`CaseParams`（frozen、改则须显式 rebuild）+ `Scene`（运行期体清单，增删不改指纹）。

对表口径：把 LIFE 自己 in-tree 的 Turek–Hron 行 `CIRCLE 1 0 0.2 0.2 0.0 0.0 0.05`
喂进本仓，映射到已验证通道算例 `CASE-CYLINDER-DRAG-RE20`（Schäfer–Turek 2D-1，锚
`Cd=5.57953523384`，Nabh 1998 谱解）。

| 量 | 值 | 出处 |
|---|---|---|
| Re | 20 | 已验证算例配置 |
| D / H | 0.1 m / 0.41 m | 阻塞比 0.2439 |
| dx | 0.01 / 0.005 / 0.0025 m（D10/20/40） | `verify_triple.json:parity_table` |
| 标记数口径 | `floor(2πr/dx)` / `floor(L/dx)+1` | LIFE `src/IBMBody.cpp:55`（实测行号） |
| 角度 | 度→弧度一次转换，`[[c,-s],[s,c]]` | `src/IBMBody.cpp:98`, `inc/Utils.h:258` |

## ② 怎么做的（审计方法）

1. **三件套存在性 + sha256 钉扎**：impl_files / entry / tests / run_artifacts / 报告，逐件 `Path.is_file()` + sha256（`verify_triple.py:files`）。
2. **repro_cmd 独立实跑 2 次**：`PYTHONPATH=src python3 xval/life_geometry_config/xval_life_geometry_config.py --mode parity`，两次输出 sha256 相同，且与交付 `runs/m17673/xval_parity.json` **逐字段相等**（`audit_equals_delivered=true`）。
3. **测试复跑**：`pytest tests/test_runtime_geometry.py -q` → 29 passed / 15.82 s（rc=0）。
4. **基准裁判直查**：`tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-COMPILETIME-PARAMS-PLUS-RUNTIME-GEOMETRY'`。
5. **集群真伪核**：`cluster_jobs WHERE id=1641`（state/rc/started_at/finished_at）。
6. **外部锚抽查**：在 `/tmp/LIFE-master` 内逐条 grep 报告引用的 file:line。
7. **可交付性核**：`git status --porcelain` + `git log -- <deliverable>` + main clone grep。
8. **gate 判据预演**：直接 import `/root/tp-agent/maturity_gate.py::judge` 用同一函数复议。

## ③ 计算结果（数字表，全部程序化注入）

### 3.1 几何/机制层（ref vs got，本单复跑）

| 网格 | dx [m] | mask ref | mask got | Δ格 | 标记 ref 公式 `floor(2πr/dx)` | 标记 got | 独立复算 | 周长误差 | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| D10 | 0.0100 | 81 | 81 | 0 | 31 | 31 | 31 | −0.1711% | ✅ 完全相等 |
| D20 | 0.0050 | 317 | 317 | 0 | 62 | 62 | 62 | −0.0428% | ✅ 完全相等 |
| D40 | 0.0025 | 1257 | 1257 | 0 | 125 | 125 | 125 | −0.0105% | ✅ 完全相等 |

二阶收敛核算：误差比 3.998 / 4.065（理论 4.0）→ 报告 ⑤ 的自声明成立。
机制层：几何增删后参数指纹 `d53748af7d09246b` 不变；改 τ 无 rebuild 被拒；rebuild 后 `65ed7299f96fd925`。

### 3.2 物理层（唯一可能触达 15% 判据的量）—— **没有数**

| 项 | 交付声明 | 本单核验 |
|---|---|---|
| 集群作业 cj1641 | 「排队未派发」 | `state=queued`, `rc=NULL`, `started_at=NULL`, `finished_at=NULL` → **确未执行** |
| Cd vs 锚 5.5795352 | WITHHELD | 交付日志唯一一行：5000/60000 步 Cd≈6.1102（+9.51%），**半程未收敛，既不可判 pass 也不可判 fail** |
| tensor_benchmark_cases | — | 该 anchor **0 行**（TensorLBM 全表 6 行，无一属此 anchor） |
| 误差是否 <3% | — | **UNDETERMINED**（无收敛数值，无登记基准） |

### 3.3 gate 机器复核预演

`maturity_gate.judge(2, evidence, 'TensorLBM', 17742)` → **gate_level=2, why=[]**（即 confirmed）。
反证：若声称 L3 → `L3 缺: integration_callers(grep 无引用)`；声称 L4 → 再缺 `benchmark/repro`。故本单取 2，不虚标。

## ④ 结果图

`runs/m17742/fig_m17742_audit_m17673.png`：(a) 三网格 ref/got 实心格数（对数轴，Δ=0 标注）；(b) 周长误差 vs dx 双对数，实测斜率贴合 −2 参考线；(c) 锚点 Cd 与「半程未收敛」点，整块底色标注 **不可据此作任何结论**。

## ⑤ 注意点 / 局限

- **F1（阻断可交付性）**：m17673 `succeeded`，但 `git status` 有 7 条未提交路径，`agent/m17673` HEAD 9bcc6db6 **等于** main clone HEAD c05d7af3，`git log -- src/tensorlbm/runtime_geometry.py` 为空 → 交付物从未进入任何 commit，「成熟后自动推送」在物理上无法发生。**worktree 一旦回收即丢失**；本单已按字节保全至 `docs/board_audit/m17742/preserved_m17673/`（MANIFEST.json 内含 sha256）。保全 ≠ 补 commit，返修仍应把文件落 commit。
- **F2 基准裁判缺位**：anchor 无 `tensor_benchmark_cases` 行 + 物理作业从未跑 → 15%/3% 精度判据在此单**无从适用**。
- **F3 mask Δ=0 是构造性恒等**：ref 用的就是本仓 `cylinder_mask`（`boundary/boundaries.py:9-25`，闭式 `(x−cx)²+(y−cy)²≤r²`），got 是同一闭式解从物理坐标重推 → 证明的是「插管不改变几何」，**不是**独立数值验证。标记数对照才有外部语义（对齐 LIFE `floor(2πr/dx)` 实测源码）。
- **F4 外部锚不可持久**：`/tmp/LIFE-master` 不在任何仓内、无 commit/sha256 钉扎，清理后 file:line 指针不可复核；引用行号实测偏 2–5 行（`IBMBody.cpp:47-105→实际 55`、`:100→实际 98`），语义正确、低危。
- **局限（审计侧）**：本单未重跑 60k 步物理算例（重计算应上集群，且 cj1641 在队列中），故对「运行期几何能否复现 Cd 锚」不作任何结论；也未做 LIFE 源码与 `runtime_geometry.py` 的抄袭级 diff（GPL 风险仅按「不同语言/不同范式重实现」判断，未逐句比对）。
- **值得肯定的**：报告显式写 WITHHELD 未粉饰；六段齐全；`test_solver_injection_*` 两类测试把「默认路径逐位不变」与「形状误用即报错」钉成断言，注入点 21 行 diff 形状校验 + provenance 齐备。

## ⑥ 误差判定（逐项，显式回答）

| 关键量 | ref | got | 误差 | 是否 <3% |
|---|---|---|---|---|
| mask 实心格数 D10/20/40 | 81/317/1257 | 81/317/1257 | 0 格（0.000%） | ✅ <3%（**构造性恒等**，不具独立性） |
| 标记数 D10/20/40 | 31/62/125 | 31/62/125 | 0.000% | ✅ <3% |
| 标记弦周长 vs 2πr | 0.3141593 m | 0.3136218/0.3140248/0.3141262 m | −0.1711/−0.0428/−0.0105% | ✅ <3%（二阶） |
| 角度跨度 0/30/45/90° | 0.35 m | 0.35 m | ≤1.59e−14% | ✅ <3% |
| **Cd（Re=20 锚 5.5795352）** | 5.5795352 | 无（cj1641 queued） | — | ❌ **UNDETERMINED，非通过** |

**判定：L2**（真代码 + 入口 + 工件 + 29 测试 + 求解器级注入被执行验证）。L3 不予（集成点仅存于未提交 diff，main 无 `solid_mask`/无 `runtime_geometry.py`，生态无消费者可达）；L4 不予（裁判缺位 + 物理层未跑 + 上述构造性恒等）；L5 不予。
**给下游的动作建议**：返修应以「commit + 合并 + 跑完 cj1641 得收敛 Cd」为唯一路径，而非补文档。

## 复现

```bash
cd /root/tp-agent/work/m17742
python3 docs/board_audit/m17742/verify_triple.py        # → runs/m17742/verify_triple.json
python3 docs/board_audit/m17742/make_figure.py          # → runs/m17742/fig_m17742_audit_m17673.png
python3 docs/board_audit/m17742/insert_verdict.py       # → verdict_m17673.sql 落表 + gate 预演
```