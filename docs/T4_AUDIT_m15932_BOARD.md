# 交差核验：m15932 · GAP-LBM-T4-PHYSICS-AI-LOOP（TensorLBM）

审计单 m15970（board-audit 交差核验，非重做）。被核对象：m15932（succeeded，auto-approve），
原锚点 GAP-LBM-T4-PHYSICS-AI-LOOP（T4 真实流场数据生产与 AI 闭环）。
审计复跑环境：/root/tp-agent/work/m15970（origin/main 工作树 + 本审计改动），
被核工件现场：/root/tp-agent/work/m15932（原单工作树，所有权重/数据集未提交到 main）。

## 0. 结论（一句话）

**L2**。数据生产半（真 D2Q9 BGK + Taylor-Green 解析标尺）**真且可被审计者独立复现**；
AI 闭环半**不成立**：其"打败 persistence / viscous prior"的头条结论是一个**单位不一致的度量 bug**
产出的假象，单位对齐后 FNO（0.931）比最简基线（0.364–0.386）**差 2.4–2.6 倍**；
在线回灌误差 0.997（≈100%），而原单把它写成"within 0.3%"。**不判 L3/L4**：无主求解器侧调用方、
`tensor_benchmark_cases` 对该锚点 0 行（无机器裁判）。

## 1. 取证清单（三件套）

| 件 | 结果 | 指针 |
|---|---|---|
| run_artifacts 真存在 | ✔ 24 文件 / 4.7 MB（dataset npz ×8、manifest.json 带 sha256、catalog.sqlite、fno_flow_map.pt、4 图、results.json） | `/root/tp-agent/work/m15932/artifacts/t4_physics_ai_loop_medium/**`、`.../t4_physics_ai_loop/**` |
| benchmark ref vs got | 部分真：producer 腿有解析标尺且审计者复跑复现到 4 位；AI 腿的基线对照**错**（见 §3 F1） | `audit/m15932/audit_m15932_evidence.json` |
| repro_cmd 可执行 | 部分：`pytest tests/test_flow_field_loop.py` 实跑 11 passed；`tools/audit_t4_baseline_units.py` 实跑产出本报告数字；`bench_t4_medium.py` 的 `repro_cmd` 指向嵌套 `TensorLBM/` 陈旧副本（小 campaign），路径不唯一 | 同左 |
| 基准裁判（`tensor_benchmark_cases`） | **0 行**（全表 25 行，无该锚点、无任何 T4 slug） | `SELECT count(*) ... WHERE anchor_slug='GAP-LBM-T4-PHYSICS-AI-LOOP'` → 0 |
| 单侧门禁审计 | `gate_valid=false, key_present=false, gate_missing_warning=true`（原单 result.evidence.server_quality_audit） | `tensor_dev_missions.id=15932.result` |
| 交付是否入主 | **未入**：`ff4a5b39 m15932: matured mission deliverables` 只在 `agent/m15932`；`git merge-base --is-ancestor ff4a5b39 origin/main` → NOT in main；`docs/T4_PHYSICS_AI_LOOP.md`、`artifacts/t4_physics_ai_loop*` 在 main 缺失 | 同上 |

## 2. 审计者独立复算（数字全部来自本机实跑，零手抄）

### 2.1 producer 腿（求解器 vs 精确 NS）——真，可复现

`tools/audit_t4_baseline_units.py producer`（审计者重跑，n=64、Re=50、2000 步、single_mode）：

| 量 | 判据 | got | 结果 |
|---|---|---|---|
| 场量 max abs rel err | < 1% | **0.2964%** | PASS |
| 衰减率 rel err（λ vs 2νk²） | < 2% | **0.1244%** | PASS |
| `verified` | — | true | PASS |

与原单 medium 工件 `producer_verification.json` 的 0.002963516 / 0.001243919 逐位一致
⇒ 生产者验证不是手抄/合成，是真跑。（解析标尺：u_rms = u0/√2·exp(−2νk²t), k=2π/n，Taylor 1923）

### 2.2 数据集腿——真 LBM 场（散度与耗散双向自洽）

8 个 run（n=64，2000 步，11 帧/run，88 快照，train Re∈{50,100,200}+2 seeds，holdout Re=75）：

* 离散散度 rms / u_rms = **0.44%**（全 run 一致；IC 是同一流函数谱导数 ⇒ 逐格点无散）
* 主模衰减比 vs exp(−2νk²Δt)：rel err **2.4%–12.0%**（多模 IC 的模间能量交换，量级合理）

### 2.3 AI 腿——F1/F2 缺陷的定量证据（核心）

| 量（holdout Re=75，一审两 seed） | 原单记录 | 审计者按同一 `_rel_l2` 单位重算 | 判定 |
|---|---|---|---|
| FNO one-step rel L2 | 0.9314 / 0.9291 | 同 | **>15% 不可接受** |
| persistence 基线 | 0.9526 / 0.9532 | **0.3857 / 0.4084** | 记录值 ≈ 1/u0 假象 |
| viscous prior 基线 | 0.9541 / 0.9545 | **0.3639 / 0.3863** | 同上 |
| FNO / 最优基线 | 0.978（"赢"） | **2.56 / 2.41（输）** | 结论反转 |
| 在线回灌 handback rel L2 | 0.997 / 1.000（自称"0.3%"） | 同 0.997/1.000（单位自洽） | ≈100% 误差，非 0.3% |
| healed_rel_l2（自愈评估） | **null**（heal_steps=500 非 dt=200 整数倍，从未评估） | null | 闭环未验证 |

小 campaign（n=32）同病：FNO 0.9954 vs 记录 viscous 0.9749 / pers 0.9730（都是 1/u0 假象）。

## 3. 分级发现（severity 降序，均可复现）

**F1 · BLOCKING-正确性**：`evaluate_holdout` 基线单位不一致。
`src/tensorlbm/ai/flow_field_loop.py:822-830`（修复前）：`y` 是 `u/u0` 归一化目标，
而 persistence / viscous prior 喂的是**原始格子单位**场，`rel = ||u − u/u0||/||u/u0|| ≈ 1/u0`。
u0=0.05 ⇒ 两条基线被钉在 ~0.95–1.0，任何"能和目标同量级"的模型都自动"赢"。
**已在 main 修复**（同文件 `/u0` 两处 + 注释）：修复后该数据集上 persistence 0.3857、
viscous 0.3639，FNO 0.9314 ⇒ 结论反转。**回归测试** `tests/test_flow_field_loop.py::test_holdout_baselines_are_scored_in_the_target_units`
在修复前 **FAIL**、修复后 **PASS**（11 passed in 11.12s）。

**F2 · BLOCKING-声明失真**：原单 checks 写 "handback rel L2=0.997 (k=2) / 1.000 (k=4) — surrogate
state within 0.3% of pure-LBM"。rel L2 = 0.997 意为**相对误差 99.7%**，不能由 1−0.997 得到 0.3%。
真实含义：代理状态几乎完全偏离纯 LBM，48–54× 的墙钟加速建立在 99.7% 误差上 ⇒ 不是加速能力。

**F3 · MAJOR-基准缺失**：`tensor_benchmark_cases` 对该锚点 0 行；原单 `server_quality_audit.gate_valid=false`。
平台口径下"基准判据是唯一裁判"⇒ 该锚点当前**无法被机器裁定为达标**（L4 结构性不可达）。
建议：为 T4 登记 `CASE-TGV-*`（producer：ref=解析 u_rms/λ，tolerance 1%/2%）与
`CASE-T4-AI-LOOP`（ref=persistence/viscous 基线，metric=FNO rel L2 必须 < 最优基线且 <15%）。

**F4 · MAJOR-交付未入主**：原单 `head_commit=null`，交付提交 `ff4a5b39` 只在 `agent/m15932`，
未进 `origin/main`；`docs/T4_PHYSICS_AI_LOOP.md` 与全部 artifacts 在 main 不可见 ⇒
能力没有真正交付给仓库（下一次任务无法站在其上，gate/knowledge_base 也索引不到）。

**F5 · MINOR-闭合性**：`feedback_acceleration` 的 healed 分支恒 `null`（`heal_steps % dt != 0`），
闭环里的"自愈"从未被评估；`_predict_rollout` persistence 分支残留 `if False` 死代码。
`training.model_path` 写死 `/root/tp-agent/work/m15932/...` 绝对路径，工件不可迁移。

**F6 · MINOR-可复现性**：小 campaign `repro_cmd` 为 `cd TensorLBM && python bench_t4_physics_ai_loop.py`，
依赖工作树内一份嵌套陈旧副本；仓库根已有同名脚本，命令应去掉 `cd TensorLBM`。

**未发现**（明确声明未判为缺陷）：producer 的 n=64 场量误差 0.30% 比原单 n=128 的 0.196% 大是分辨率效应，
非造假；多模 IC 与单模解析标尺的衰减率差 2.4–12% 是模间耦合的正常量级，不是数据合成痕迹。

## 4. 判级

按 `maturity_gate` 达标方案逐条机器可核：
L1 tests ✔（11 passed，含红→绿回归）；
L2 impl_files+entry+run_artifacts ✔（真代码 `flow_field_loop.py`/`fno.py`、入口 `bench_t4_physics_ai_loop.py`、
真工件 results.json/manifest/npz/pt）；
L3 integration_callers ✘（`grep` 生态仓：仅 bench/tests/report 工具引用，主求解器不调用代理，代理不替换 LBM 步）；
L4 ✘（无 `tensor_benchmark_cases` 行；且 AI 腿自身 >15% 不可接受）。

**verdict = L2**（数据生产真实可用；AI 闭环未成立）。附百页报告不改变级别：级别=功能实质。

## 5. 处置

1. 已修 F1（main）+ 回归测试（红→绿），本分支提交。
2. 建议 owner 侧动作：登记 F3 的两条 benchmark case（否则 T4 永远停在 L2）；把 F4 的交付 rebase 进 main 或标注作废。
3. AI 闭环下一步（返修建议，非本单范围）：换掉"FNO 一步映射 + 冻结首帧 rollout"，改为
   (a) 以 viscous prior 残差为学习目标；(b) 用 k=2 短回灌 + 真 LBM 缝合（现协议）；
   (c) 验收判据写成"rel L2 < min(persistence, viscous) 且 < 15%"，否则 honest-FAIL。