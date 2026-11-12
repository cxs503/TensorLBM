# 交差审计 · m15399 / GAP-PP-4-opt-loop（TensorLBM）— board-audit m15449

**被审对象**：mission 15399（`feat(opt-loop): differentiable tau/Re recovery demo on Taylor-Green decay`，分支 HEAD `e4a7ee910d01`，worktree `/root/tp-agent/work/m15399` 只读），PR #1808（merged），报告 tensor_reports id=14481。
**审计者**：m15449（board-audit）。**性质**：交差核验（非重做）；判据以证据为准，报告文字不参与判级。

## 0. 结论

| 项 | 判决 |
|---|---|
| 等级 | **L2**（独立功能实现：真代码 输入→计算→输出，数值面自洽且可复算） |
| 未再升原因 | **L3 未达**：模块无仓内调用者（除 tests/scripts），无 `tensorlbm.inverse` 包/API，无平台接线；**L4 无裁判**：本锚点在 `tensor_benchmark_cases` 与 `tensor_case_pool` 均 **0 行**，唯一裁判缺位，got 无法被机器判定 |
| 数值诚实性 | **PASS**：报告④表 6 行×3 种子全部可从 JSON 复现，最大偏差 4.24e-05 pp（约 0） |
| 工件真实性 | **PASS（附耐久性缺陷）**：8 件 run_artifacts + 图 + 模块 + 测试均在 mission 分支；但 canonical JSON、6 个 shard JSON、PNG、logs **未随 merge #1808 进 main** |
| 测试 | **PASS（独立复跑）**：`pytest tests/test_opt_loop_tau_recovery.py -q` → **9 passed in 41.60s**（本轮在 `/tmp` 干净副本内重跑，loongarch64） |
| repro_cmd | **PASS**：`python -m tensorlbm.opt_loop_tau_recovery --help` exit 0，CLI 面 --shard/--merge/--seeds/--iters/--polish/--threads 齐全 |

## ① 案例介绍

Taylor–Green 衰减（周期方盒，一个波长横跨 n 个格距）：`u_x = A u0 sin(kx)cos(ky) exp(-2 nu k^2 t)`，`u_y = -A u0 cos(kx)sin(ky) exp(-2 nu k^2 t)`。该式是不可压 NS 精确解（Taylor & Green 1937, Proc. R. Soc. A 158:499；衰减率 2νk² 见 Pope, *Turbulent Flows* §9），同时是 BGK-LBM 衰减模态，`nu=(tau-0.5)/3`（格子单位）。

| 参数 | 值 | 来源 |
|---|---|---|
| u0 | 0.05 | 工件 case.u0 |
| tau_true / nu_true | 0.55 / 0.0166666667 | 工件 |
| n（analytic16/analytic32） | 16 / 32 | 工件 case.n |
| Re_true | 48.0000（n=16）/ 96.0000（n=32） | 工件 |
| 观测时刻（n=16） | [21, 78, 137, 195] | 工件 |
| 噪声档 | 0 / 0.2% / 1.0%·u0 | 6 轨 |
| 未知量 | tau（→nu→Re）；幅值 A 为 nuisance | 报告 §① |

## ② 怎么做的

1. 梯度＝全程可微 LBM：`tensorlbm.autograd_path.rollout`（D3Q19 BGK+周期 gather streaming），tau 以图连通 0 维张量进碰撞步（`src/tensorlbm/opt_loop_tau_recovery.py:81` 导入；`recover()` :370、`run_job()` :635、`assemble()` :708）。
2. 幅值 A 作为 nuisance 剖掉（选型 A，配控制臂反证）；优化变量取 log nu + box-clamp tau ∈ [0.5005,1.9]；Adam(lr=0.3,2 步热身)+L-BFGS(strong Wolfe,10 步)。
3. 损失：各观测时刻相对场残差等权平均；观测窗跳过前 10% 启动瞬态。
4. 工具链：torch 2.7.1（工件记录）；`--shard i 6` 分片 → `--merge` 合并 → `scripts/make_opt_loop_figures.py` 出图。
5. 本轮复核工具：`docs/board_audit/m15449/audit_m15399.py`（C1–C7 七项独立核，输出 `audit_evidence.json`）+ `make_figure.py`。

## ③ 计算结果（程序化注入，零手抄）

自 JSON 重算（不读报告）：18 次反演，逐轨最差误差

| track | 数据源 | A | n | max abs rel err [%] | 判定 |
|---|---|---|---|---|---|
| analytic16 | analytic | free | 16 | 0.1796 | PASS |
| analytic32 | analytic | free | 32 | 0.0999 | PASS |
| twin16 | twin | free | 16 | 0.0000 | PASS |
| analytic16_fixed | analytic | pinned | 16 | 3.4163 | FAIL |
| twin16_noise0.2 | twin | free | 16 | 0.1001 | PASS |
| twin16_noise1.0 | twin | free | 16 | 0.4988 | PASS |

* 内部自洽（C2）：`nu_hat`/`Re_hat` 全部可从 `tau_hat` 复算到 ≤1e-9（0 处不符）。**指标基混淆（low，本轮新发现）**：`rel_err_pct` 实为 **ν 相对误差** (nu_hat-nu_true)/nu_true（18/18 条逐位吻合），而验收门的文字是「**Re 恢复**<3%」。因 Re∝1/ν，两基符号相反、量级一阶相同：受门最差 twin16_noise1.0/seed0 存 -0.4988% → Re 基 0.5013%（仍 <3%，判定不变）；控制臂存 -0.0999% → Re 基 0.1000%（差 0.2000 pp，两基皆超门）。结论：门边界 3% 附近该基差会给偏乐观读数（本例最多 0.12 pp），报告未声明基 → 判 low，不改变等级。
* 受门各轨最大误差 0.4988%（twin16_noise1.0），距 3% 门约 6.0× 余量。
* 报告④表 vs JSON（C3）：3 行全部解析，最差偏差 **4.24e-05 pp** → 零手抄成立。
* 控制臂诚实性：analytic16_fixed（幅值钉死）3.4163% 超门，被明确标注为「故意的选型反证」；`all_tracks_pass` 排除控制臂后为 True —— 未粉饰。

## ④ 结果图

被审方图 `figures/opt_loop_tau_recovery.png`（2475×690，三面板：衰减曲线 vs 精确解 / 18 次反演逐迭代误差 log 纵轴+3% 门线 / 各臂最差误差柱）；本轮审计图 `docs/board_audit/m15449/fig_audit_m15449.png`：面板 A=从 JSON 重算的逐轨最差误差 vs 预声明 3% 门（控制臂红）；面板 B=报告表 vs JSON 的逐轨偏差（零手抄检验）。

## ⑤ 注意点 / 坑 / 局限

1. **基准裁判缺位（medium，平台侧）**：`tensor_benchmark_cases` 本锚点 0 行、`tensor_case_pool` 0 行（SQL 见 audit_evidence.json: C6_arbiter）。3% 门唯一书面出处是 mission goal 原文（预声明，非事后回填）。与 m15298(Kirsch) 的 F1 同类，但那次池里至少有一条 tol_policy；本单两表皆空。
2. **工件耐久性（medium）**：merge #1808 仅带入 4 个文件（doc/src/tests/scripts）；`artifacts/opt_loop_*.json`（7 个）、`figures/opt_loop_tau_recovery.png`、`artifacts/logs/*.log` 未进 main。main 上 `docs/OPT_LOOP_TAU_RECOVERY.md` 引用的工件在干净检出中不存在（与 m15298 F3「png 被过滤」同族，此处连 JSON 也丢）。
3. **shard 标签陈旧（low）**：shard 文件中控制臂名为 `twin16_fixed`，canonical 为 `analytic16_fixed`；21 条 shard 记录中 18 条键值全等（≤1e-12），3 条仅标签不同、tau 全同 → 工件集非同一次运行的同构分解；`--merge` 会产出带旧标签 payload（`TRACK_ORDER.get` 兜底不报错）。
4. **mission 自报未完成项（诚实，但计入交差）**：goal ①②③ 的调研/沉淀（`oss_inventory` 读、`oss_solutions` 写）未做（DB permission denied）；复核：`oss_solutions` 236 行/151 slug 中无本锚点记录。选型仅落仓内文档。
5. **未跑的检查（unknown，非 passed）**：18 次反演的本轮独立重算已递交集群 cj1126（提交时队列前 327，审计窗口内未派发）；「数值可复现」证据 = 9 项测试（梯度 vs 中心差分 2.06e-06/8.86e-06、twin 反演、种子复现）+ 工件内部自洽，**不是**逐位重跑 18 次反演。GitHub CI(x86) 未审；全部运行在 loongarch64。

## ⑥ 误差判定（是否 <3%）

| 关键量 | ref | got | 误差 | <3% |
|---|---|---|---|---|
| Re, analytic16（解析精确解，n=16） | 48.0000 | 47.9140–47.9140 | 0.1796% | **是** |
| Re, analytic32（网格加密，n=32） | 96.0000 | 96.0960–96.0960 | 0.0999% | **是** |
| Re, twin16（同求解器自洽） | 48.0000 | 48.0000–48.0000 | 1.22e-06% | **是** |
| Re, 1% 观测噪声 | 48.0000 | 47.9183–48.2406 | 0.4988% | **是** |
| 控制臂 analytic16_fixed（幅值钉死） | 48.0000 | 49.6978 | 3.4163% | **否（故意保留的反证，不计入受门）** |

显式回答：受门各轨是（<3%），最差 0.4988%；唯一超门臂是方法选型控制臂，报告与工件均标注为故意保留。但按本单「基准裁判是唯一裁判」的规则，该判据未进基准表 → 判据数值 PASS / 裁判登记 **NOT_MEASURED**。

## ⑦ 交付与落表

* 工件：本文件、`audit_m15399.py`、`audit_evidence.json`、`fig_audit_m15449.png`、`verdict.sql`。
* verdict 落 `tensor_maturity_verdicts`（repo=TensorLBM, scope='m15399 GAP-PP-4-opt-loop', level=2, verdict_by='board-audit'）。
* 返修建议（给续作单，不砍本次）：(i) 注册 `tensor_benchmark_cases` 行（metric='TG 衰减 Re 相对误差', tolerance='<3%'）；(ii) artifacts/figures 镜像进 `docs/` 以免 merge 过滤丢件；(iii) 抽出 `tensorlbm.inverse` 公共 API 并接入平台反演端点（L3）；(iv) 重跑分片使标签与新代码一致；(v) 文档声明 rel_err 符号约定。
