# m18429 交差核验报告 —— m18366 / XVAL-LBM-PF-EOS-MAXWELL-HIGH-DENSITY-RATIO

**审计对象**：mission 18366（TensorLBM，status=succeeded，claimed「成熟度 L0→L4」）
**审计单**：m18429（board-audit，交差核验，非重做）
**审计侧独立复跑**：main 检出 `/root/tp-agent/work/m18429`（driver sha256 与被核工件记录一致）
**核验产物**：`audit/m18429/{tables.md,evidence.json,fig1_ref_vs_got.png,verify_m18429.py}`（数值全部由脚本生成，禁手改）

---

## 1. 案例介绍（物理背景 / 对象 / 参数）

被核锚点验证的是**单组分 Shan–Chen 伪势（SCMP）+ Peng–Robinson（PR）状态方程型伪势**
在**高密度比（10:1 / 100:1 / 1000:1）**下能否维持 Maxwell 等面积构造给出的共存对：

- 参考侧（ref，解析）：`sc_eos_coexistence.maxwell_coexistence` 在比容空间解
  ∫(p−p_sat)dv = 0，得每档目标密度比的 T_r / ρ_l / ρ_v / p_sat；
- 实测侧（got，离散）：静态液滴（或平面界面）在周期盒内演化到力学平衡，测维持比、伪流 Ma、质量漂移、塌缩步。

预注册参数（m18366 `predeclared.json`）：EOS=PR，G=1.0，目标比 10/100/1000，L=96（tiers）/L=64（pilot），
R=20，τ∈{0.6,0.8,1.0,1.2}，float32，周期边界，保持 ≥20000 步，门值
`ratio_error≤0.15 ∧ Ma≤0.05 ∧ 步数≥20000 ∧ mass_rate≤5e-8/step ∧ 无 ψ clamp`。

**判据的唯一裁判**是基准表：
`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-PF-EOS-MAXWELL-HIGH-DENSITY-RATIO'`
—— 机器核验结果 **0 行**（`anchor_slug ILIKE '%MAXWELL%' OR '%EOS%'` 也是 **0 行**，见 tables.md T5）。

## 2. 怎么做的（核验方法 / 流程）

独立重建执行路径，不采信报告文字：

1. **工件取证**：读 m18366 工作树 `benchmarks/eos_maxwell/out/results.json`（sha256 `755f39ce…`）、
   `xval/XVAL-LBM-PF-EOS-MAXWELL-HIGH-DENSITY-RATIO/{predeclared,results}.json`、`REPORT.md`，
   以及仓库内机生成的 `RESULTS.md`（2026-11-10 生成）与 `CRITERION.md`；
2. **代码同源核**：比对工件记录的 `driver_sha256=ae393b70…` / `probe_sha256=31fbdd7c…` 与 main 检出实测
   sha256（一致，见 tables.md T6），确认 ref/got 数值不是伪造格式；
3. **ref 重算**：以库内 Maxwell 求解器直接重算三档构造比与面积残差（tables.md T1）；
4. **got 抽表**：从工件与机生成表程序化抽 got 数值（tables.md T2/T3），逐项与预注册门和 owner 15% 标准比；
5. **独立复跑 repro_cmd**：`python3 benchmarks/eos_maxwell/run.py --mode pilot --steps 200 --out /tmp/m18429_repro --device cpu`
   → rc=0，产出 tiers=1（tables.md T4，工件留档 `audit/m18429/repro_m18429_pilot200/results.json`）；
6. **测试复核**：`pytest tests/test_sc_eos_coexistence.py` → **13 passed, 1 xfailed**；
   `pytest tests/test_eos_maxwell_stability.py` → **17 passed, 2 skipped**（skip 原因：campaign/criterion 工件不在本树）；
7. **机器事实**：基准表 0 行；`tensor_reports WHERE mission_id=18366` **0 行**；`oss_solutions` 该 feature **0 行**；
   `git log --all --grep=m18366` **0 提交**；mission result `head_commit=NULL, branch=NULL`；
   集群 `cj1906`（"全量战役已递交"）审计时仍 **queued**，无工件返回；
8. **判级**：按 maturity_gate 的 JUDGE_SCHEME（L2=impl_files+entry+run_artifacts；L3=+integration_callers；
   L4=+benchmark 数值对照+repro_cmd；L5=+external_anchor）逐级核对。

## 3. 计算结果（数字表）

> 全部数值出自 `audit/m18429/tables.md`（verify_m18429.py 机生成），此处仅作转述。

### 3.1 三件套核验（任务①）

| 项 | 结论 | 证据 |
|---|---|---|
| run_artifacts 文件 | **部分真**：`out/results.json` 存在（sha256 755f39ce…），但 `tiers=[]`、`controls` 仅 1 条 `planar_ratio10_edm`（40000 步）；**无 pilot 臂工件**（xval 声称 `pilot_ratio10` 2000 步） | tables.md T2/T6；`tags_in_out_results=['planar_ratio10_edm']` |
| benchmark ref vs got | **ref 真、got 缺档**：Maxwell 构造三档命中（10.01269/99.97081/995.15275，面积残差 1.4e-13~1.2e-11）；got 仅平面臂与 m8675 历史 tiers 表；**task 要求的三档液滴实测无本单工件** | tables.md T1/T2/T3 |
| repro_cmd 可执行 | **是**（审计侧独立执行 rc=0，工件非空，tables.md T4） | `audit/m18429/repro_m18429_pilot200/results.json` |
| 基准裁判表 | **0 行** → 本锚点无注册裁判，L4 机裁不可达 | tables.md T5 `benchmark_cases_registered=0` |
| 报告六段 | REPORT.md 六段齐全（§1–§6），但 §4.2 数字与工件不符（见 3.3） | `audit/m18429/m18366_xval/REPORT.md` |

### 3.2 容差判定（owner 精度标准 <15% + 预注册复合门）

| 臂 | got | ref(Maxwell) | 相对误差 | 伪流 Ma | 判定 |
|---|---|---|---|---|---|
| tier ratio10 edm（m8675 表） | 10.450 | 10.010 | **4.37%** | 0.115 | 误差项 <15% ✓，但 Ma>0.10 → 复合门 **FAIL** |
| tier ratio10 guo（m8675 表） | 13.810 | 10.010 | 37.9% | 0.127 | **FAIL** |
| tier ratio100 edm（m8675 表） | 293.30 | 99.970 | **193%** | 0.333 | **FAIL** |
| tier ratio100/1000（guo/edm） | — | — | — | — | 16/23/100/156 步内**塌缩** |
| planar ratio10 edm（**本单工件**） | 11.0156 | 10.0127 | 10.02% | 0.1141 | 误差项 <15% ✓，工件自身判 **FAIL**（Ma） |
| 审计侧短程复跑（200 步） | 10.3493 | 10.0127 | 3.36% | 0.1336 | **FAIL**（Ma；步数亦 <20000） |

**结论**：ref 侧 Maxwell 构造成立；离散格式在 ratio=10 能"活着"但伪流地板 0.114–0.133（PASS 门 0.05），
ratio=100 密度比维持误差 193%，ratio=1000 百步内塌缩 —— **锚点判据（<15%）仅在 ratio10 的密度比单项达标，
复合预注册门与高密度比两档全部不达标**。

### 3.3 报告 vs 工件的不一致（严重度 S2）

| 报告 §4.2 写法 | 工件实测 | 差异 |
|---|---|---|
| "密度比误差 ~2%（pilot，2000 步）" | 现存唯一工件是 40000 步 planar 臂，`ratio_error=0.1002`；xval `results.json` 的 pilot 记录 `ratio_error=null` | **报告数字无工件背书** |
| "质量漂移率 8.87e-9/步" | planar 工件 `mass_rate_per_step=2.254e-9`；全树 grep 无 8.87e-9 来源 | **不可溯源** |
| "伪流 Ma 0.114" | planar 工件 `ma_max=0.11414` | 一致（但被误标为 pilot） |
| "全量战役已递交 cj1906" | 审计时 cj1906 **queued**（submitted 13:51），无工件 | 未完成，不能计入交付 |

### 3.4 流程事实

- `git log --all --grep=m18366` = **0 提交**；`git status --short` = 4 条 staged/unstaged（xval 三件 + out/results.json）
  → **交付物未入库**，平台分支推送不会携带（mission result `head_commit=NULL, branch=NULL`）；
- `tensor_reports WHERE mission_id=18366` = **0 行** → 报告纪律未执行（owner 常设令）；
- `oss_solutions`（feature_slug ILIKE '%EOS-MAXWELL%'/'%HIGH-DENSITY%'）= **0 行** → 任务③"调研沉淀"未完成；
- m18366 的 `out/results.json`（13:57 生成）覆盖了 `RESULTS.md/CRITERION.md` 的数据源（11-10 生成，含 6 条 tiers），
  现行 out 已无 tiers → **机生成链断链**（重跑 plot_figures 只会得到空表）。

## 4. 结果图（ref vs got + 伪流门）

![fig1](fig1_ref_vs_got.png)

左：各臂「实测比 / Maxwell 比 − 1」，绿带为 owner ±15%；右：伪流 Ma 与预注册 PASS(0.05)/MARGINAL(0.10) 门。
（`audit/m18429/fig1_ref_vs_got.png`，由 verify_m18429.py 从工件数值程序化生成。）

## 5. 注意点（局限 / 适用范围）

1. **交差核验非重做**：未重跑 5922.9 s 全量战役；`RESULTS.md` 的 tiers 数值属 m8675（2026-11-10）战役，
   原始 out 工件已被本单覆盖，无法逐字节复核，只能按机生成表引用（且已在 tables.md 标注代际）。
2. **审计侧复跑为短程**（200 步，L=64，pilot 模式），只证 CLI 可执行与量级一致（ratio 误差 3.36%、Ma 0.134），
   不构成 20000 步保持性结论。
3. **架构限定**：全部工件来自 loongarch64 / torch 2.7.1 / float32；x86 CI 与 float64 未验。
4. **门方案限定**：maturity_gate 的 JUDGE_SCHEME 只核"证据键存在性+数值存在性"，不核"数值是否过容差"；
   若不显式降级，本单凭 pre-existing impl+一条 control 工件也能被机器判成 L4 —— 故本审计按**功能实质**主动降级。
5. 平面臂 `ratio_error=10.0%` 与液滴臂 `4.37%`（历史）不同，说明几何/曲率对离散共存对的影响不可忽略，
   引用任一数字时须带臂名。

## 6. 误差判定（关键量逐项 + 是否 <3%）

| 关键量 | ref | got | 误差 | <3%？ | <15%？ |
|---|---|---|---|---|---|
| Maxwell 构造比 ratio10 | 目标 10 | 10.01269 | **0.13%** | **是** | 是 |
| Maxwell 构造比 ratio100 | 目标 100 | 99.97081 | **0.03%** | **是** | 是 |
| Maxwell 构造比 ratio1000 | 目标 1000 | 995.15275 | **0.48%** | **是** | 是 |
| 离散维持比 ratio10 edm（历史 tiers） | 10.010 | 10.450 | **4.37%** | 否 | 是 |
| 离散维持比 ratio10 planar（本单工件） | 10.013 | 11.016 | **10.02%** | 否 | 是 |
| 离散维持比 ratio100 edm（历史 tiers） | 99.970 | 293.30 | **193%** | 否 | **否** |
| 离散维持比 ratio1000（历史 tiers） | 995.15 | 塌缩（100/156 步） | — | 否 | **否** |
| 伪流 Ma ratio10（全部门/臂） | 门 0.05 | 0.114–0.134 | 超门 2.3–2.7× | 否 | — |
| 审计复跑 vs 报告 §4.2 声称（~2%） | 声称 | 3.36%（200 步） | 1.36 pp | **否** | — |

**显式回答**：ref 侧 Maxwell 等面积构造三项均 <3%（机器精度量级）；**离散侧密度比维持在 ratio10 为 4.37%/10.02%，
不满足 <3%，但满足 owner <15%；ratio100 为 193%、ratio1000 直接塌缩，>15% 不可接受**。
伪流 Ma 全部超过预注册 PASS 门 0.05（0.114–0.134），故按 m18366 自己预注册的复合门，**锚点判据未通过**。

---

## 7. 判级与裁定建议（L0–L5）

**声明 L4 → 核验判 L2**（evidence 落表，JSON 见 `audit/m18429/evidence.json`）：

- **L2 成立**：真实实现（`src/tensorlbm/sc_eos_coexistence.py` 等）存在且非占位；入口
  `benchmarks/eos_maxwell/run.py:65-101` 可执行；run 工件真实（审计侧复跑通过、sha256 同源）；
- **L3 不计入本单**：`maxwell_coexistence` 的调用方（`microbubble_dr/channel.py:81,190,192`、
  `multiphase_phase_change_verification.py:268`、`cavitation.py:68`）真实存在，但均早于 m18366
  （本单 0 提交），不构成本单增量；
- **L4 不成立**：① `tensor_benchmark_cases` 该 anchor 0 行 → 无注册裁判；② 预注册复合门在 ratio10 即 FAIL
  （Ma 0.114–0.121 > 0.10）；③ ratio100/1000 塌缩或误差 193%；④ `tiers=[]`，完整算例未交付；
  ⑤ 报告 §4.2 的 2% 与 8.87e-9 无工件背书 → 不满足"完整算例+误差分析+可复现"；
- **L0/L1 不采用**：确有真实计算工件与测试通过，不是纯文档。

**Disposition：request changes（对 L4 声明）+ approve with notes（L2 能力事实）**。
严重度排序：S1 L4 虚标与预注册门 FAIL 未在 mission headline 体现；S2 报告数字无工件溯源（~2%、8.87e-9）；
S3 覆盖了 RESULTS.md 数据源导致机生成链断链；S4 交付物未提交（0 commits / head_commit=NULL）；
S5 报告与调研沉淀缺失（tensor_reports 0 行、oss_solutions 0 行）；S6 cj1906 仍 queued 却已 succeeded 终态。