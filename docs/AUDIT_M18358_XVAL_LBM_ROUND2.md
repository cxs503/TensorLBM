# 交差审计报告 · m18241 `XVAL-LBM-ROUND2-METHOD-SELECTION`

审计单：m18358（board-audit，TensorLBM）｜被审单：m18241（agent，状态 succeeded）｜审计日：2026-11-13
审计对象工作树：`/root/tp-agent/work/m18241`（未提交，见 F1）｜审计人工作树：`/root/tp-agent/work/m18358`

**结论先行：级别 = L2（真实功能实现，未达基准门）；裁定 = request changes（返修）。**
m18241 没有造假：报告与工件一致、数字可逐位复现、失败项如实登记为 not_run/FAIL。
但它的「成功」是**可执行但未跑完**的成功：3 个预声明分辨级只跑了 1 个，决定性因子格 A6 缺席，
两个 OSS 风格臂在本仓自己的 15 % 门上 `Cl_amp` 越界（-46.57 % / -15.40 %），
且该锚点在 `tensor_benchmark_cases` 里 **0 行登记**——按平台机器判据它不可能走到 L4。

## 1 取证：三件套逐项验收

### 1.1 run_artifacts（真存在？真对应？）

| 工件 | 存在 | 内容判读 |
|---|---|---|
| `runs/A1_D10/result.json` | ✅ | Cd=1.6143262481245584, St=0.16899580031695963, Cl_amp=0.6579248352005606, 21259 步 —— **第 1 轮工件**（tracked，mtime=检出 12:55） |
| `runs/A2_D10/result.json` | ✅ | Cd=1.4661360417278881, St=0.17662926112262423, Cl_amp=0.3515523646082385 |
| `runs/A4_D10/result.json` | ✅ | Cd=1.6423615055096823, St=0.1717673074998151, Cl_amp=0.556625096634633 |
| `runs/A3_D10/fail.json` / `A5_D10/fail.json` | ✅ | 实测发散（step 600 / 5800，tau_minus 25.5 / 2.0）—— 不是 not_run，是真跑出来的失败 |
| `runs/A6_D10/` | ❌ | **不存在**（预声明里的决定性因子格 BGK+sponge） |
| `runs/A{1,2,4,6}_D14/` | ⚠️ | 只有 `state.npz`（中断的分块运行），无 result，`logs/A*_D14.log` 全 0 字节 |
| `runs/*_D20/` | ❌ | 不存在 |
| `table_round2.csv` / `summary_round2.json` | ✅ | 3 行 A1/A2/A4@D=10；`gate_at_each_D` 的 14/20 全 null，ladder D=14/20 全 null |
| `oss_outlet_bc_xval.json` | ✅ | `controls_ok=false`（incident_peak_spread=1.713、noise_floor_rel_max=0.601）⇒ K3 自认失败 |
| `figures/fig_r2_*.png` | ✅ | 8 张 PNG，报告只内嵌了 fig_r2_6 |

逐字段比对（`table_round2.csv` vs 各 `result.json`）：**3 臂 × 5 个数字全部一致（相对误差 <1e-12）**，表格不是手抄的。

### 1.2 benchmark（ref vs got 数值 / 容差）

```
SELECT case_name, metric, tolerance FROM tensor_benchmark_cases
 WHERE anchor_slug='XVAL-LBM-ROUND2-METHOD-SELECTION';   -- 0 rows
```
TensorLBM 全仓 26 个登记锚点里没有任何 method-selection 条目（最接近的 `CASE-LBM-SQUARE-CYL-NORBERG` 是另一个算例）。
因此**判据只能来自任务 goal 的 owner 口径（<15 %）**，ref = A1（本仓记录的 MRT+sponge 选型，属内部基线，不是外部基准）。

| 臂（D=10） | ΔCd % | ΔSt % | ΔCl_amp % | 15 % 门 |
|---|---|---|---|---|
| A2 BGK / 无海绵（OSS 做法） | -9.18 | +4.52 | **-46.57** | FAIL |
| A4 MRT / 无海绵（海绵隔离对照） | +1.74 | +1.64 | **-15.40** | FAIL |
| A6 BGK / 有海绵（决定性格） | — | — | — | **not_run** |

代码级对标（ref = 实跑 OSS 源 `Maarten-vd-Sande/lbm @573a01c4`）：K1 碰撞 rel-L2 = 1.18e-07、K2 输运 = 2.71e-08，
均 <3 %（float32 舍入级）——**本轮唯一真正达标的对标，但那是第 1 轮 kernel 成果的复验**，不是选型问题的答案。

### 1.3 repro_cmd（可执行？）

- 测试：`PYTHONPATH=src pytest tests/test_xval_round2_method_selection.py -q` → **6 passed in 4.39 s**（断言是不变量，不是空跑：tau_field 给定时与逐格显式松弛 atol=1e-14 相等、None 时逐位相同、步数 21259/29763/42518、合成矩阵上 interaction 项与 `separable=False`）。
- 单臂：审计独立重跑 A2@D=10（8 threads，357.3 s）→ Cd/St/Cl_amp 与记录差 **6.7e-15 / 2.8e-17 / 0.0**，逐位级复现。
- 集群：`run_cluster ps cj1848` → **queued 45.2 min**；`nodes` → n315/n330 ok 但 load 40.5/41.7。**not_run 的理由属实**：控制面 120 s 命令上限（审计自己也被砍了 2 次：`TIMEOUT after 120s`），集群排队 310 单。

### 1.4 外部锚可解析性

`/root/oss-cache/Maarten-vd-Sande_lbm` 存在；`git rev-parse HEAD` = `573a01c4d2e6321f3634ff271adca4ce91872e6f`（与 predeclared 一致）；
5 个源文件 `sha256sum` 与 `predeclared_round2.json::round1_reference::oss_source_sha256` **5/5 逐位一致**。
被引规则 `lbm/streamcollide.py::set_outlet`（zero-gradient copy from ix-1）可在本地源中核到。**外部锚是真的。**

### 1.5 报告六段

`tensor_reports` id=17118（15 409 字符）：①问题+ref 出处 ②方法与实现 ③参数表（Re/blockage/Ma/tau/sponge/seeding/steps/dtype 完整）④对照表（ref vs got + 误差打叉）⑤收敛性/敏感性（网格收敛 NOT RUN、TRT λ 两点实测、出口吸收 indicative）⑥失败模式与工件（honest-FAIL 台账）+ §7 差异清单 + §8 逐项误差判定。
**六段齐全，且 §8 显式回答「是否<3 %」：代码级 yes（1.18e-07），选型级「未裁定」，收敛「未证明」，出口仪器 no。这是加分项——没有粉饰。**

## 2 判级：L2

| 级 | 判据 | 本单 |
|---|---|---|
| L0 文档 | — | 超过 |
| L1 数学/单元验证 | — | 超过（有 6 条不变量测试） |
| **L2 独立功能实现** | 真代码 输入→计算→输出 | **成立**：`collide_bgk(tau_field)` 真实新能力 + 臂 harness + 聚合管线 + 出口仪器，输入→compute→KPI 可复现到 1e-15 |
| L3 系统集成 | 接入主求解器/平台 | **不成立**：`grep -rn 'tau_field=' src/` 无任何调用方给 `collide_bgk` 传参（既有调用方都是 `collide_mrt`/`collide_hb_bgk_forced`）；harness 是独立 xval 脚本 |
| L4 基准验证 | 完整算例+误差分析+可复现 | **不成立**：1/3 分辨级、决定性格 A6 未跑、收敛未证、自仓 15 % 门 FAIL、`tensor_benchmark_cases` 0 行 |
| L5 科研工程成果 | — | 不成立 |

百页报告不改变级别：级别取功能实质，L2。

## 3 发现（按严重度）

| # | 级别 | 事实 | 证据 |
|---|---|---|---|
| F1 | **provenance 阻断** | round-2 交付物**从未进 commit** | `git log --all -- docs/XVAL_LBM_ROUND2_METHOD_SELECTION.md xval/.../summary_round2.json` → 空；`git status` 显示 `M solver.py` + 13 个 `??`；mission `head_commit=null`；主树 m18358 无这些文件 |
| F2 | major | 基准未跑完（1/3 级 + A6 缺席）⇒ ALT-A/B/C 不可分，选型未裁定 | `gate_at_each_D` 14/20 全 null；`runs/A*_D14` 只有 state.npz |
| F3 | major | 无平台登记基准 ⇒ 机器判据不可能到 L4 | `SELECT count(*) …` = 0 |
| F4 | minor | `wall_s`/`ms_per_step` 是 chunk 局部耗时被当整臂成本 | 记录 6.4 s / 0.301 ms/step；审计整跑 357.3 s / 16.808 ms/step → **55.8×**（`run_arm.py` 分块 resume 后 t0 重置） |
| F5 | minor | K3 `controls_ok=false` 但 R 值仍写在 JSON 顶层 | `reflection_coefficient`: far_field 0.0025 vs oss_zero_gradient 0.325 |
| F6 | info | 4-A 的 D=10 三行是第 1 轮工件再聚合 | mtime 12:55；报告小字已注明 |

**夸大项（limited）**：mission 状态 `succeeded` + headline「交付」易被读作达成；实质是 honest-FAIL。报告本体没有夸大。

## 4 独立性与局限

- 未复算 A1/A4 整臂（各约 350 s 量级），未补跑 D=14/D=20/A6（作业至今 queued）。
- 未对 K3 仪器复算（controls 已 FAIL，无判定价值）。
- 业务级真值（Cd/St 文献值）不在本锚点登记范围内，故未核。
- 物理结论不因审计重跑改变；审计只裁定「级别 + 证据真实性」。

## 5 返修清单（给接棒单）

1. 把 round-2 全部工件（含 `src/tensorlbm/solver.py` 的 `collide_bgk(tau_field)` diff）提交成 commit。
2. 集群排队恢复后 fetch/重跑 `cj1847`(A6+A2rep@D=10) / `cj1848`(D=14) / `cj1849`(D=20)，跑 `analyze_round2.py` 填 A6 行、ladder 与 interaction 项。
3. 把 15 % 门与 A1 基线的 KPI 登记进 `tensor_benchmark_cases`（metric=cd_mean/st_fft/cl_amp，tolerance=15），使 gate 可机器复核。
4. `wall_s` 改累计口径或改名 `chunk_wall_s`；K3 R 值加 `not_evidence: true` 或移出顶层。

## 6 本次审计跑的检查

```
psql: SELECT … tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-ROUND2-METHOD-SELECTION'      -> 0 rows
pytest tests/test_xval_round2_method_selection.py -q                                            -> 6 passed (4.39 s)
python3 run_arm.py --arm A2 --D 10 --threads 8 --out /tmp/rerun_A2b --no-resume                  -> 357.3 s, ΔKPI ≤7.3e-15
sha256sum 5 × OSS 源 + git rev-parse HEAD                                                        -> 5/5 一致, rev 573a01c4
grep -rn 'tau_field=' src/                                                                       -> 无 collide_bgk 生产调用方
git log --all -- <round-2 files>                                                                 -> 空（未提交）
run_cluster ps cj1848 / nodes                                                                    -> queued 45.2 min / 两节点 ok load≈41
```

产物：`artifacts/audit_m18358/{verdict_evidence.json, rerun_A2_D10_result.json, table_round2.csv, make_fig_m18358.py, fig_m18358_verdict.png, report_m18241_snapshot.md}`、
`verdict_m18358.sql`（已落 `tensor_maturity_verdicts`）。
## 7 落表与机器复核口径（重要）

`maturity_gate.py` 的 `files_exist()` 要求 `impl_files` / `run_artifacts` / `entry` / `external_anchor`
的**每一项都能解析成真实路径**（相对 `/root/tensor-ecosystem` 或 `/root/tp-agent/work/m{审计单}`），
`tests` 必须是非空**列表**（列表跳过存在性检查）。因此本单 evidence：

- 契约键（机器核）：`impl_files`/`entry`/`run_artifacts`/`external_anchor` 全部为纯路径，**19 条逐条 assert 存在**；
- 人读注解放在 `impl_files_detail`/`run_artifacts_detail`/`external_anchor_detail`/`tests_detail`/`findings`。

用同样的判据代码本地复演：`L2: PASS`（L1/L2 契约齐、impl 非占位、benchmark 含数值、repro_cmd 可执行）。
`verdict_m18358.sql` 是幂等 upsert（`ON CONFLICT (repo,scope,verdict_by) DO UPDATE`，并把 `gate_state`
重置为 pending），因此改证据后重跑即可，不会产生重复行或与已有行冲突。
注意：若把注解串直接写进契约键，机器复核会判 L2 缺项并降级到 L1 → 对 m18241 触发**错误返修单**；
本单先按真实形状落表，再用 gate 同源代码复演确认。
