# AUDIT m17851 — 交差核验 m17720（XVAL-LBM-CARTESIAN-MPI，TensorLBM）

审计席位：board-audit ｜ 审计单 m17851 ｜ 被审单 m17720（TensorLBM，status=succeeded）
被审工作树：`/root/tp-agent/work/m17720` ｜ 审计工作树：`/root/tp-agent/work/m17851`
判级：**L2（独立功能实现，含已定位缺陷）** ｜ 基准裁判结论：**FAIL（25.35% ≫ 15% 容差）**

## ① 三件套取证（真存在？真数值？真复现？）

| 项 | 核验结果 | 证据指针 |
|---|---|---|
| run_artifacts | 存在且**真值**：本次重跑逐数 bit-identical（monolithic 0.04988328383918091 / l2 3.7627977838496767；decomposed 0.03732374758250377 / l2 6.282042859429132） | `m17720/runs/xval_cartesian_mpi/result.json`；比对表 `artifacts/m17851_audit/repro_matrix.json:shipped_vs_reproduced`（全部 `identical: true`） |
| benchmark ref vs got | ref=解析 Poiseuille 剖面 `4Umax/H²·y(H−y)`（case_spec.py:105-119）；got 见下表；**交付件自判 status="fail"** | `m17720/runs/xval_cartesian_mpi/result.json:criteria`；`artifacts/m17851_audit/runs/m17720_shipped_result.json` |
| repro_cmd | 可执行：`PYTHONPATH=src python3 -m tensorlbm.xval_cartesian_mpi --out-dir <out> --nx 128 --ny 32 --steps 1000` | 本审计实跑 6 组配置，全部产出 result.json（`artifacts/m17851_audit/runs/*`） |

⚠ repro 缺口：交付工件用 `nx=128, ny=32, steps=1000`（非包默认 `DEFAULT_SPEC=256/64/8000`），无参入口**不复现**交付件；工件内 `warmup_steps=2000 > steps=1000`，且 `warmup_steps` 在 runners 中**从未被读取**（死参数）。

## ② 基准裁判（tensor_benchmark_cases / 平台登记）

- `SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-CARTESIAN-MPI'` → **0 行**（全表 59 例，无一行含 CARTESIAN）。
- `tensor_xval_cases` 同锚点 → **0 行**：交付的 xval_table 未落入平台工件表。
- 故机器裁判缺位；本审计以 **mission goal 的 <15% 判据** + 解析解剖面为裁判（result.json 亦内嵌 `accept_threshold_pct=15.0`）。
- 报告登记：`tensor_reports WHERE mission_id=17720` → **0 行**（m17720 result 记 "API submission failed 401"）→ 报告纪律未达。

## ③ 数值裁判（程序化重跑，无手抄）

| 配置（128×32/1000 步，除注明） | mono 中心线误差 | decomposed 误差 | 判据 15% | 场一致 |
|---|---|---|---|---|
| npx=1 npy=1 | 0.2334% | 0.2334% | PASS | 一致 |
| npx=1 npy=2（仅 y 分块） | 0.2334% | 1.7038% | PASS | — |
| npx=2 npy=1（仅 x 分块） | 0.2334% | 0.2334% | PASS | — |
| **npx=2 npy=2（交付默认）** | 0.2334% | **25.3525%** | **FAIL** | fields_identical=false |
| npx=2 npy=2, 4000 步 | 2.2378% | **36.5066%** | **FAIL** | — |
| npx=2 npy=2, 256×64/1000 步 | 0.0405% | **25.1073%** | **FAIL** | — |

结论：误差随网格加密/长积分**不收敛消失**（25.1%→36.5%），是结构性缺陷而非分辨率/瞬态问题。交付件 `status="fail"`，故「XVAL 对标通过」**不成立**。

## ④ 缺陷定位（根因 + 修复验证）

- 空间定位：误差集中于 x 分块界面列与周期缝合列（x=0,1,63,64,126,127）；`interior_col_mean_abs_err=2.44e-3`（约 4.87% Umax），界面列 `5.32e-3`；`max|Δu|=1.76e-2`。`umax_centreline` 取样点 `ux[ny//2, nx//2]` 恰落在 x=64 界面列（0.03732 vs mono 0.04988），故 25.35% 指标由界面列主导——但整场不一致（fields_identical=false），非纯指标选取问题。
- 根因：`runners._halo_exchange` 的 y 面交换写**整行**（`new_f[bottom][:, -h:, :] = f[:, h:2h, :]`，runners.py:441-446），用**交换前**快照覆盖了本轮 x 面已写好的**角点**；D2Q9 对角分布 f5..f8 依赖角点 halo ⇒ 角点数据陈旧 → 界面列伪值。`halo_slices()`（domain_decomp.py:198-226）的 send/recv 切片同址（`(slice(h,h)…, slice(0,h))` 两侧相同），是未被调用的死代码缺陷。
- 修复验证（不改被审树，运行时 monkeypatch，其余完全不变）：角点感知交换（面只写内部跨距 + 对角角块显式交换）后，**25.3525% → 0.2334%**（l2 6.282%→3.7628%，与 monolithic 完全一致）。`artifacts/m17851_audit/fix_probe.json`；脚本 `docs/audit_m17851/fix_probe.py`。
- 附注：`mpi4py` 未安装（ImportError 已证实），所谓 cartesian-**MPI** 全程为单进程模拟，MPI 通信路径**从未执行**（result.json:frictions 自认）。pylbm 源在 `/root/oss-cache/pylbm`（mpi_topology.py 存在），锚点出处真实。

## ⑤ 六段报告齐全性 & 生态接入

- 交付目录无 REPORT.md（仅 result.json + CSV + PNG 图）；mission result 里 checks 自述 "needs debugging" —— 六段名义不齐。
- 测试：`tests/`、`benchmarks/`、`docs/`、`examples/` 对 `xval_cartesian_mpi` **零命中** → 无任何测试。
- 集成调用方：`src/tensorlbm/` 其余模块零引用；`git status` 仅 `?? src/tensorlbm/xval_cartesian_mpi/`（**未 commit、零 tracked 改动**）⇒ 不构成 L3。
- 重复建设：`src/tensorlbm/domain_decomposition.py`（m3379/m368 已有 slab 分解 + HaloExchange）未复用、未扩展。

## ⑥ 判级（L0-L5）

| 级 | 判据 | 是否达 |
|---|---|---|
| L1 数学/单元验证 | 解析解对照存在（mono 0.2334%），但无测试文件 | 达（弱） |
| **L2 独立功能实现** | 1316 行真实现、CLI 入口可跑、6 组配置工件可复现（非占位） | **达（本判级）** |
| L3 系统集成 | 无调用方、未入库 | 未达 |
| L4 基准验证 | 完整算例有，但误差 25.35%/36.5% ≫ 15%，自判 fail | **未达** |
| L5 科研工程 | 需 L4 | 未达 |

**verdict：L2**。理由：代码为真实独立实现且可复现，但核心能力（2D 分块 halo 交换）在交付默认配置下错误，XVAL 判据未过 ⇒ 只认功能实质，不认"已通过"叙述。建议板面按 L2 confirmed 记账的同时，**以缺陷单跟进角点 halo 修复**（修复方案已由本审计验证，1 处函数改动即可把 2×2 打到 0.23%）。

## 附：本次实跑命令与工件

- 重跑：`PYTHONPATH=src python3 -m tensorlbm.xval_cartesian_mpi --out-dir /tmp/m17851_repro_1000 --nx 128 --ny 32 --steps 1000`（另跑 1×1/1×2/2×1、4000 步、256×64 各一次）
- 证据包：`artifacts/m17851_audit/{repro_matrix.json, fix_probe.json, fig_m17851.png, runs/*, field_*_final.csv}`
- 生成脚本：`docs/audit_m17851/build_audit_artifacts.py`（数字全部从工件读出，零手抄）、`docs/audit_m17851/fix_probe.py`