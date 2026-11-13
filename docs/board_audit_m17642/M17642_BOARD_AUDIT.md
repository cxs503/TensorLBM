# 交差审计 M17642 — XVAL-LBM-CENTRAL-MOMENT-PARTIAL-RELAXATION

- 审计单：m17670（board-audit）｜被审单：m17642（`succeeded`，2026-11-13 07:38:52Z，claimed_by `tp-glm51-1`，model `deepseek-v4-flash`）
- 仓：TensorLBM｜锚点：`XVAL-LBM-CENTRAL-MOMENT-PARTIAL-RELAXATION`（对标 joconnor22/LIFE 的 `CENTRAL_MOMENTS` 部分松弛碰撞）
- 被审工作树：`/root/tp-agent/work/m17642`（branch `agent/m17642`，**零 commit**：`git log main..HEAD` 为空，`result.head_commit=null`；全部交付物为 untracked 新文件）
- 审计工作树：`/root/tp-agent/work/m17670`（本文件 + `artifacts/audit_m17642/`）
- 机器仲裁：`tensor_benchmark_cases` 对该 anchor **0 行**（见 F1）→ 注册裁判缺席，判据退化为任务 goal 内**预声明**的 owner 精度标准 <15%。

## 0. 结论（先给级别）

**L2 = 独立功能实现（真代码：输入→计算→输出），confirmed 停在本级；L3 不成立（未接入主求解器）；L4 内容齐备但按达标判断方案被 L3 卡住。**

一句话：算子是真代码、真实算例、真可复现（253/253 叶子值一致、连图都字节相同），Taylor–Green 解析解验证误差 **-0.045 %/+0.027 %/+0.022 %**（远在 15 % 门内）；但（a）`collide_central_moment_d2q9` 未进 `src/tensorlbm/__init__.py` 的懒加载碰撞注册表，求解器无法选取它（L3 缺）；（b）任务零 commit，交付物不在任何仓主线；（c）报告的**旗舰主张「与上游算子 round-off 级等价（1.7e-16）」审计未复现**（F4），外部锚点等价腿记为 WITHHELD。

## 1. 三件套取证（工件逐项核实）

| 三件套 | 指针 | 状态 |
|---|---|---|
| impl_files | `src/tensorlbm/central_moment_d2q9.py`（332 行，12209 B） | ✅ 真实现（矩变换 M0/M0⁻¹、`collide_central_moment_d2q9`、`macroscopic_guo`、力项逐项对齐上游；无 NotImplemented/pass 占位） |
| entry | `benchmarks/xval_life_cm_partial.py`（573 行，25073 B，`--quick`/全量两档） | ✅ |
| run_artifacts | `runs/xval_life_cm_partial/result.json`（7584 B，253 叶子值）+ `figures/xval_life_cm_partial/fig1..fig3.png` | ✅ 存在非空 |
| tests | `tests/test_central_moment_d2q9.py`（161 行 / 14 用例）+ `tests/fixtures/life_cm_partial_oracle.json`（8 组夹具） | ✅ 审计重跑 **14 passed in 25.20 s**（`/tmp/audit_m17642` 抛离副本） |
| 报告 | `tensor_reports id=16599`（mission_id=17642）；`docs/XVAL-LBM-CENTRAL-MOMENT-PARTIAL-RELAXATION.md` 131 行 | ✅ 六段齐（①问题+ref 出处 ②方法与实现 ③参数表 ④对照表 ⑤收敛/敏感性 ⑥失败模式与工件，另 §7 图）；⚠️ `tensor_dev_missions.report_id` 为 NULL（联动缺口，非阻断） |

## 2. 可复现性（审计自己重跑，不是读报告）

抛离副本 `/tmp/audit_m17642`，两条命令重跑：

```
PYTHONPATH=src python3 -m pytest tests/test_central_moment_d2q9.py -q          # 14 passed, 25.20 s
PYTHONPATH=src python3 benchmarks/xval_life_cm_partial.py --quick            # 61 s，退出 0
```

- `result.json` 逐叶子比对：**253 个叶子，差异 3 个，全部是 `tg[*].runtime_s`（墙钟）**，物理载荷 **0 差异**。
- 文档管线：重跑生成的 `docs/XVAL-LBM-CENTRAL-MOMENT-PARTIAL-RELAXATION.md` 与交付件 **逐字节相同**（sha256 一致）。
- 三张 png **字节相同**（sha256 一致：`f6d21382…`/`a9c6f9e5…`/`cbab1507…`）→ 出图是确定性的，不是事后挑图。

结论：可复现性级别充分（L4 的 repro 腿成立）。

## 3. 基准裁判：注册裁判缺席，退化为预声明判据

```
SELECT case_name,metric,tolerance FROM tensor_benchmark_cases
 WHERE anchor_slug='XVAL-LBM-CENTRAL-MOMENT-PARTIAL-RELAXATION';   -- 0 rows
```
`tensor_benchmark_cases` 全表 56 行，含 `CASE-LBM-SQUARE-CYL-NORBERG` 等 4 个 TensorLBM 锚点、`XVAL-LBM-BUFFER-LAYER-2TO1-REFINEMENT`（在册，passed），但**本锚点无行**。故裁判位空缺，唯一可用的外部判据是任务 goal 内预声明的 owner 精度标准 **<15 %**（以及交付件自身在代码里先于运行写死的 `GATE_PCT = 15.0`）。

按 <15 % 逐项核（ref = 二维 Taylor–Green 解析衰减率 4νk²，ν=(1/ω−1/2)/3，ω=1.5385，n=32，U0=0.020，Ma=0.0346，Re=12.8，200 步，测率窗 50–200）：

| 臂 | ref 4νk² | got 实测 | 误差 % | <15 %? | 末态场 L2 误差 % | <15 %? |
|---|---|---|---|---|---|---|
| LIFE partial（s=1 高阶矩） | 0.0077<sub>11</sub> | 0.0077<sub>07</sub> | **−0.045** | ✅ | 0.561 | ✅ |
| all-order ω | 0.0077<sub>11</sub> | 0.0077<sub>13</sub> | **+0.027** | ✅ | 0.788 | ✅ |
| bulk 1, s3=s4=ω | 0.0077<sub>11</sub> | 0.0077<sub>12</sub> | **+0.022** | ✅ | 0.613 | ✅ |

收敛：n=32→48 场上 L2 0.5611 %→0.4570 %（拟合阶 0.51，低于二阶，报告 §4.1 如实写了「误差已在门内两个量级以下」，未把它当成果吹）。敏感性：s3=s4 扫描 0.7→ω 误差 −0.128 %→+0.022 % 单调收敛；稳定性扫描 ω_max 1.998/1.998/1.90（交付文档写 2.00 是 1.998 的四舍五入，量级无差）。

**基准判定：解析解腿 PASS（误差比门槛小两个量级）；上游算子等价腿 → WITHHELD（见 F4）。**

## 4. 级别判定（L0–L5，按功能实质，不看报告厚度）

上限更高但被卡住的三处机器可核事实：

1. **未接入碰撞分派（L3 缺，硬证据）**：`src/tensorlbm/__init__.py` 的懒加载注册表有 14 个 `collide_*` 条目（`collide_bgk`/`collide_mrt`/`collide_cumulant_d2q9`/`collide_cascaded_d3q19`…），唯独没有 `collide_central_moment_d2q9`。审计实跑：
   ```
   >>> import tensorlbm as t
   >>> hasattr(t,'collide_central_moment_d2q9')      -> False
   >>> getattr(t,'collide_central_moment_d2q9')      -> AttributeError
   >>> getattr(t,'collide_cumulant_d2q9')            -> OK   # 同类算子对照组
   ```
   即：算子只能被 `benchmarks/xval_life_cm_partial.py` 用 `from tensorlbm.central_moment_d2q9 import …` 直接取用，**求解器/平台侧没有任何选择面**。对比同类已 mature 的 L3 判例（m17594）至少有一条仓内 harness 调用方。
2. **零 commit**：`result.head_commit=null`、`git log main..HEAD` 空、6 个交付路径全为 `??` untracked（`result.evidence.git_status` 自陈亦同）。因此 `integration_callers` 的 grep 在生态检出 `/root/tensor-ecosystem/TensorLBM` 内必然落空——这条不是我猜的，是本单交付物的真实状态。
3. **自治终止状态**：`result.fail_stage="budget_exhausted+forced_finish"`，`server_quality_audit.gate_valid=false, key_present=false`。任务是被预算掐停后强制收尾的，**它对自己的 L4 主张没有做机检**；审计不继承该主张。

因此：L2 三道门（impl_files 非占位 + entry + run_artifacts）**全过**；L3 的 `integration_callers` 不过；L4 的 benchmark/repro 两项证据充分但因 L3 前置不成立而无法确认。**判 L2**。补 L3 的动作是机械的（注册表加一行 + 把 benchmark 落到仓主线），不构成本轮返修的必要理由，但应在后续迭代补齐。

## 5. 发现（按严重度）

**F1（阻断裁判位，非交付方过错）注册裁判缺席。** 本锚点 `tensor_benchmark_cases` 0 行 →「基准判据是唯一裁判」这条在本单无裁判可依，判据退化为预声明 15 %。建议由锚点喂料器按 prereg（metric=衰减率误差/场 L2，tolerance=15 %）补一行，否则 XVAL 类锚点永远只能「预声明替代裁判」。

**F2（实质，阻断 L3）算子未进碰撞注册表。** 见 §4.1。可一句话修复：`__init__.py` 懒加载表加 `"collide_central_moment_d2q9": ("central_moment_d2q9", "collide_central_moment_d2q9")`；本轮为「非重做」，未代改。

**F3（诚实但重要）零 commit 交付。** 全部产物 untracked，`succeeded` 态与「无任何可推送对象」并存；若平台按分支推送，本单**没有分支可推**（`agent/m17642` 无提交）。这是交付链缺口，建议门禁把「交付物须在 branch 上有 commit」纳入 pre-succeed 检查。

**F4（实质，最高价值发现）「与上游 round-off 级等价（1.7e-16）」审计未复现。** 审计用**独立解析器**重抽上游 `src/Grid.cpp` 的 `fStar[0..8]`（`SQ(`→`sq()`、`QU(`→`qu()`，嵌套括号安全；正则版先试、报错后改用安全版，两版结论一致），并按源码 `k0..k8` 定义（`k3=2ρc_s²`、`k8=ρc_s⁴`、`k4=(1−ω)Σf(cx²−cy²)`、`k5=(1−ω)Σf cx cy`，`u=(Σcf+0.5F)/ρ`）**手工推导**后求值，三方对比结果：

| 对比 | 8 组夹具 max \|Δf\| |
|---|---|
| 交付 fixture vs 交付实现 | **1.7e-16**（自洽，内部一致） |
| 交付实现 vs 审计上游 oracle | **1.49e-3**（取 9 种格向标号中最优配对） |
| 交付 fixture vs 审计上游 oracle | **1.49e-3** |

三种标号配对（identity / 交付 `perm` / `perm⁻¹`）最差分别为 5.68e-2 / **1.49e-3** / 2.09e-2——交付 `perm` 最接近但**到不了 1e-16**。把 `fStar` 当 k 的线性映射反解出厂「夹具隐含的 k 向量」逐项比对：`k0,k1,k2,k3,k6,k7,k8` **比值 1.000000**，只有两个前碰撞偏应力矩 `k4Pre/k5Pre` 对不上（比值 1.1234 / 0.3708）。再对全部 **9! = 362880 种格向重标号**暴力搜索能同时满足这两个隐含值的配对：**0 个命中** ⇒ 残差**不是**标号/顺序约定问题，而落在「前碰撞 κd/κxy 用的位移速度或矩定义」上。

含义（不粉饰、也不越权定罪）：
- **不能说交付实现是错的**：它与自己的夹具 1.7e-16，且它与解析 TG 解、黏度、守恒律全部自洽（14 测试 + 三个臂误差 <0.05 %）。
- **只能说旗舰主张（上游等价 1.7e-16）未被独立复现**，残差 1.49e-3（≈ \|f\| 的 1.5 %，仍 < 15 % 门）卡在偏应力矩上；审计预算内未能定位是我方换算式还是交付方 oracle 的差异（**审计侧 limitation，非结论**）。按证据纪律记为 WITHHELD，不作 confirmed，也不判造假。

**F5（非阻断，元数据）文档四舍五入与联动缺口。** 稳定性表把 1.998 写成 2.00（无损）；`tensor_dev_missions.report_id` 为 NULL 而 `tensor_reports id=16599` 存在（联动缺口，与 m17593 同款，系统性小病）。

**F6（正面，确认收益）工件确定性做法值得留存。** 全部数字由脚本从 `result.json` 程序化注入（零手抄）、图由同脚本生成且**重跑字节相同**；上游 GPL 源码**不入仓**、只把 8 组数值夹具 + provenance 入仓——这条版权处理是本生态可直接复用的范式。

## 6. 生态摩擦（如实记）

- 上游 GitHub 直连被 egress 拦，m17642 经 `ghproxy.net` 取 tarball（`/tmp/LIFE-master` 仍在，故审计可复算）。
- `oss_solutions` 表对该角色不可读（permission denied）；知识检索引擎 `127.0.0.1:8021` 离线（`engine_offline`）——两处都让「开工三查」名存实亡。
- 审计侧：`psql` 需 `runuser -u postgres`；`rsync` 未安装（改用 `tar` 管道）。

## 7. 工件清单

- `artifacts/audit_m17642/audit_m17642.py` — 本审计可复算脚本（工件盘点 / 逐叶子比对 / 上游独立重抽 / 图）
- `artifacts/audit_m17642/audit_findings.json` — 全部数字
- `artifacts/audit_m17642/independent_life_oracle.json` — 上游三方对比 + 矩反解 + 重标号搜索
- `artifacts/audit_m17642/independent_life_oracle.py` — 独立解析器（第一版，正则；结论与安全版一致）
- `artifacts/audit_m17642/audit_m17642.png` — 左：交付误差 vs 15 % 门；右：上游等价残差 vs 声称的 1.7e-16 线
- `docs/board_audit_m17642/M17642_BOARD_AUDIT.md` — 本文
- 落表：`tensor_maturity_verdicts`（repo=TensorLBM, scope=`m17642 XVAL-LBM-CENTRAL-MOMENT-PARTIAL-RELAXATION`, level=2, verdict_by=`board-audit`）

**处置（disposition）：approve with notes → 记 L2 消账；F2（注册表一行）/F3（交付须有 commit）建议后续迭代补齐；F4 需要一次「上游 oracle 双实现对拍」再判，本轮不做（非重做）。**