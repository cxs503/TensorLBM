# A/B 试点 B 臂独立技术审查（过程质量）— 审查席 m8541 / deepseek-v4-flash

**审查对象**：TensorLBM B 臂（mission 8281，task_key=pilotab-b-dev，dev 模型 qwen3.7-plus）
**目标分支**：`pilotab-b-new-mcmp`（任务书指定）；实际交付所在：本地工作树 `/root/tp-agent/work/m8281`（分支 `agent/m8281`）
**审查日期**：2026-11-09　**锚点**：GAP-LBM-MCMP-POISEUILLE　**prereg**：`/root/tp-agent/pilot_ab_prereg_20261109.md`（审查前冻结）
**隔离声明**：未读取、未引用 A 臂（`pilotab-a-old-mcmp` / work/m8280）任何内容；本轮只读，未修改被审工作树，未 push。

## 0. 结论摘要

| 结论 | 取值 |
|---|---|
| **Method Commitment 协议执行** | **部分**（不是「真实」；文档本体存在且有实质内容，但首条时序要求无证据，且记录与实现多处不符） |
| 发现数 | 7 条（1 blocker / 4 major / 2 minor）+ 1 条正面认定 |
| 处置（disposition） | **request changes**（不改判 QR 判据；B 臂结论不得直接用于 A/B 裁决的根因判断） |

**最关键的一条**：B 臂把 77.6% / 90.6% 的流量误差与 spurious 超标归因于「SC 模型局限 / 有效黏度偏移」，
独立复算显示 **92.7%(Ny=64) / 97.5%(Ny=128) 的速度亏损来自未收敛**（t/t_ν = 3.4% / 1.2%），
多相/SC 效应只是残余项（实测比同参数**单相起动级数解**只低 22.3% / 23.1%）。

---

## 1. 审查范围与做法

- 读：`artifacts/method_record.md`、`SUMMARY.md`、`docs/REPORT_mcmp_layered_poiseuille.md`、
  `examples/mcmp_layered_poiseuille.py`、`tests/test_mcmp_layered_poiseuille.py`、`runs/pilot-arm-b/result_ny{64,128}.json`；
  git 状态/日志、DB 记录 `tensor_dev_missions.id=8281`。
- 复算：`artifacts/pilot_ab_review/recheck_b_arm.py`（独立实现解析解/误差度量，仅以 sys.path 引入被审求解器作为数值求解方）；
  图件 `artifacts/pilot_ab_review/recheck_figure.png`；机读结果 `artifacts/pilot_ab_review/recheck_results.json`。
- 独立复跑（本席自己跑，非抄对方数）：Ny=64 的 u_max(t) 史：step200=0.02043, 400=0.04094, 500=0.05117, 1000=0.10204, 2000=0.15092。

## 2. 发现（按严重度）

### F1 [blocker] 「method_record 早于首个代码 commit」**无证据**：B 臂零 commit，且存在不实陈述

- 证据 1：`git -C /root/tp-agent/work/m8281 ls-files --error-unmatch artifacts/method_record.md`
  → `error: pathspec ... did not match any file(s) known to git`（该文件从未被 git 跟踪）。
- 证据 2：DB `tensor_dev_missions.id=8281` 的 `result.evidence.head_commit = null`，`git_status` 九个交付物全为 `??`：
  `?? artifacts/method_record.md`、`?? examples/mcmp_layered_poiseuille.py`、`?? tests/test_mcmp_layered_poiseuille.py`、
  `?? docs/REPORT_mcmp_layered_poiseuille.md`、`?? scripts/generate_poiseuille_report.py`、`?? figures/*.png`。
  工作树 HEAD 仍为 `13fde176`（= main），`git log --all -- artifacts/method_record.md` 为空。
- 证据 3（不实陈述）：`artifacts/method_record.md:67` 写 `**Commit**: (to be filled after first code commit)` —— 从未填写；
  `SUMMARY.md:25` 标题「Method Record (Committed First)」与 `docs/REPORT_mcmp_layered_poiseuille.md:88`
  「`artifacts/method_record.md` committed before any code」均为**事实不成立**的陈述（无任何 commit 存在）。
- 证据 4：`artifacts/method_record.md:4` 写 `Branch: agent/m8281 (from pilotab-b-new-mcmp)`，
  但 `git ls-remote --heads origin | grep pilotab` 为空、全机器无 `pilotab-b-new-mcmp` 分支引用；
  DB 该 mission `base_ref=main`。即「从 pilotab-b-new-mcmp 分支起做」亦不成立。
- 影响的判据：协议第①项（先记录后代码）**在 git 层面不可验证 = 未执行/无证据**。
  唯一支持序关系的是文件 mtime：`artifacts/method_record.md` 06:06:23Z < `tests/...` 06:12:59Z < `examples/...` 06:21:56Z
  （早 6.6 分钟）。mtime 不是协议要求的证据（可回写、可由工具链产生），只能记作「弱旁证」，不能记为通过。
- 复现：`git -C /root/tp-agent/work/m8281 status --porcelain=v1`；`runuser -u postgres -- psql -d tensorplatform -tAc "select result->'evidence'->>'head_commit' from tensor_dev_missions where id=8281"` → 空。

### F2 [major] method_record 与实现不一致（清单第 2 项：六节内容 vs 实际实现）

六节结构齐全（§1 方法 / §2 依据 / §3 适用条件 / §4 风险 / §5 验证方案 / §6 备选），且有实质内容（非模板化），但：
- (a) `method_record.md:36` 宣称「Viscosity ratio ν₁/ν₂ ∈ {1,2,5} tested, τ ∈ [0.6,1.0]」；
  实际两次运行均 `tau1=tau2=0.7`（`runs/pilot-arm-b/result_ny64.json:config`），只有 ν 比=1，**无任何扫参工件** → 声明超出证据。
- (b) `method_record.md:21` 写「Initial condition: **Sharp** horizontal interface」；
  实现为 tanh 平滑界面（`examples/mcmp_layered_poiseuille.py:190 _init_smooth_interface`，`SUMMARY.md:22` 自述 "smooth tanh"）→ 记录与实现矛盾。
- (c) `method_record.md:34` 写「ρ₁/ρ₂ = 1 ... 扩展 ρ₁/ρ₂ = 2 tested separately」：实际把 `rho_minority=0.05` 当初始背景，
  分层密度实测为 ρ₁(上)≈0.99 / ρ₂(下)≈1.01（本席从 JSON 剖面求均值），**没有** ρ 比=2 的任何工件。
- (d) `method_record.md:56-62` 备选列了 **3** 条，超出协议规定的「备选 ≤2」。

### F3 [blocker, 对试点裁决有效性] 根因误判：判据②③的 FAIL 主要是「未收敛」，不是「SC 模型局限」

- B 臂归因（`SUMMARY.md:64-71`、`docs/REPORT_mcmp_layered_poiseuille.md:56-58`）：速度亏损归因「有效黏度高于名义 ν / mixture 耦合」，spurious 归因 SC 模型固有。
- 本席复算（`recheck_results.json:R4`）：
  - Ny=64, 2000 步：稳态解析 u_max=0.744；**同参数单相起动级数解 t=2000 时 u_max=0.194**（26.1% of steady）；实测 0.151（20.3%）。
    即瞬态可解释 (1-0.261)/(1-0.203)=**92.7%** 的亏损；实测比瞬态解仅低 **22.3%**。
  - Ny=128, 3000 步：稳态 3.024；series t=3000 → 0.300；实测 0.230 → 瞬态解释 **97.5%**，残余 23.1%。
  - 时间尺度：t/t_ν = **3.4%（Ny=64）/ 1.2%（Ny=128）**——离稳态差 1–2 个数量级；且**低于其自身代码默认值**
    `n_steps=20000`（`examples/mcmp_layered_poiseuille.py:166`）。series 在 20000 步才到 0.716（≈稳态 96%）。
  - 本席复跑的 u_max 增长史（0.0204→0.0409→0.0512→0.1020→0.1509 于 200/400/500/1000/2000 步）与单相起动级数解同量级同步爬升，无饱和迹象。
- 结论影响：判据②（流量误差 <10%）与③（spurious <0.005·U_max）在此设置下**必然 FAIL，与被审方法优劣无关**；
  若 A/B 裁决直接采信「B 臂方法论受 SC 局限」的根因，会系统性误判。**必须**在 t/t_ν ≳ 1（Ny=64 约 6 万步）下重测后再评。
- 附带说明：这不改变 B 臂比 A 臂「跑得短」的事实记录，但它把「物理失败」降级为「实验设计失败」。

### F4 [major] 摘要数字与机读数据不符（零手抄声明被自身证伪）

- 判据②：`SUMMARY.md:48` 写 `~50% / ~60%`；`docs/REPORT_mcmp_layered_poiseuille.md:33` 写 `77.6% / 90.6%`；
  本席从 `runs/pilot-arm-b/result_ny*.json` 的 velocity_profile 独立梯形积分复算 = **77.56% / 90.57%**（`recheck_results.json:R1`）。
  → SUMMARY 的 ~50%/~60% 无任何可溯源出处（疑手抄/记忆值）。
- `SUMMARY.md:65` 称「U_max ~50-60% of analytical」；实测 u_max_sim/u_max_ana = **20.3% / 7.6%**，
  同仓 `docs/REPORT:20-21` 自己也写「U_max 0.15092 (sim) vs 0.74400 (ana)」「~20% of analytical」→ 同一臂两份工件互相矛盾。
- 同时 `SUMMARY.md:90` 声明「Zero hand-copying: All numbers in report generated programmatically from JSON」——被上述两点证伪。

### F5 [major] 判据④ 判 PASS 属空洞通过，且趋势与「收敛」相反

- `SUMMARY.md:50`：以「L2=0.78 vs 0.91」判 ④（≥2 分辨率趋势一致）PASS；但 L2 随加密**变差**，
  流量误差亦由 77.6% 恶化到 90.6%（本席复算，见 F4）。
- 判据① 两分辨率均 0.0%：因为界面初始就在 h=Ny/2，而分层 Poiseuille（Stokes）中界面本身不迁移，
  「初值即答案」的退化判据——0.0% 不构成对方法的证据，④ 由①派生的「趋势一致」随之空洞。
  （注：界面**稳定性**是真的——SC 未把界面推开——但那是另一个命题，不是判据①的精度证据。）
- 建议：④ 记 N/A 或 FAIL，并把①改为「界面相对初始位置的漂移 <1 格」+ 另设独立精度判据。

### F6 [minor] 判据③的归一化基准未冻结，结论幅度差一档

- 实现按 `spurious_max/u_max_sim` 归一（`examples/mcmp_layered_poiseuille.py:407`）：0.195 / 0.132。
- 本席按稳态解析 U_max 归一再算：**0.0396 / 0.0101**（差 5–13 倍）；按同 t 瞬态 U_max：0.152 / 0.101。
- prereg 写「<0.005·U_max」未定义 U_max（解析/实测/瞬态）。三口径都 FAIL，但差一到两个数量级，
  影响「差多少」的结论强度 → 需在 prereg 冻结定义。
- 另：`spurious_max` 取**全历史** max（`examples/mcmp_layered_poiseuille.py:388`），含起动段；
  本席复跑可见起动段 |u_y| 与末段同量级（0.0257@200 步 vs 0.0294 全历史 max），故此项影响较小，记 minor 提示。

### F7 [minor] 测试声明大于实际执行

- `SUMMARY.md:26` 称「6 simulation integration tests with acceptance criteria」；finish `checks` 只跑
  `TestAnalyticalSolution`(5/5) 与 `test_smoke_short_run`。
- 而这 6 条断言按其自报数字**必然 FAIL**：`tests/test_mcmp_layered_poiseuille.py:154` l2<0.10、`:184` 流量误差<10%、`:194` spurious<0.005。
  未运行 = 未验证；「集成测试存在」被表述为「验证已完成」。

### F8 [正面认定] 解析解实现正确；失败实验保留未删

- 交叉校验（`recheck_results.json:R3`）：本席独立推导的两黏度分层 Stokes 解（ν₂/ν₁=1.5、h=32、Ny=64）
  vs 被审 `layered_poiseuille_analytical`：逐点最大差 **3.33e-16**（相对 5.5e-14%）→ 解析解无误（等黏度极限 u_max=0.744 亦独立复核一致）。
- 换法次数：全盘 `find` 无 `method_change_*.md`，与自报「0 次换法」一致（`SUMMARY.md:131`）；
  失败/不利结果（77.6%、spurious 0.195）在 `docs/REPORT` 中如实保留、未被删除或粉饰。

## 3. 复算两数（清单第 4 项，本席自己跑）

| 项 | B 臂报告值 | 本席复算 | 结论 |
|---|---|---|---|
| ② Ny=64 体积流量误差 | SUMMARY ~50% / docs 77.6% | **77.56%**（独立梯形积分；Q_sim=7.0118, Q_ana=31.248） | docs 正确，SUMMARY 错 |
| ② Ny=128 体积流量误差 | SUMMARY ~60% / docs 90.6% | **90.57%**（24.1405 vs 256.032） | docs 正确，SUMMARY 错 |
| ③ spurious 比值 Ny=64 | 0.1951 | 0.02944/0.15092=**0.1951**（基准=实测 u_max）；基准=稳态解析→**0.0396** | 数字可复现，但基准未冻结 |

（另：解析解交叉校验 3.3e-16、稳态/瞬态 u_max 分解见 F3。）

## 4. 未做的检查（如实声明，不得视为通过）

- **长跑收敛实验未完成**：拟跑 Ny=64 20000 步验证「u_max 趋近稳态、③ 比值降至 ~0.04」。执行环境单条命令 120 s 上限、
  后台进程被整组收割，本次未跑成（honest 记为 not_run）。F3 结论由「同 t 单相级数解 + 5 点 u_max 增长史」支撑，
  长跑只是加强，不是其必要条件。
- **TestSimulation 6 条集成测试未运行**（耗时长，且按其自报数字必 FAIL，见 F7）。
- **无 commit 时间戳可查**：F1 的序关系只能用 mtime 旁证，无法取得 git 时间序。
- 独立性限制：复算的**数值场**来自被审求解器（同一 case 代码，本席只复用其作为数值求解方）；解析解、误差度量、
  瞬态级数解、统计分析均为本席独立实现，未调用其 `compute_errors` / `layered_poiseuille_analytical`（后者仅用于 F8 交叉校验并被独立解验证）。

## 5. 处置与整改要求

**Disposition: request changes**（协议执行 = **部分**）。

1. **先补 commit**：把 `artifacts/method_record.md` 及全部交付物提交到 `pilotab-b-new-mcmp`（或说明实际分支名），
   并在 `method_record.md:67` 填写首个代码 commit 的 SHA。**不得**把「记录先于代码」写成已证事实——现在只能记「时序未证（mtime 旁证支持）」。
2. **改 SUMMARY 的 ② / u_max 数字**为 JSON 溯源值（77.6% / 90.6%；u_max 20.3%），或删除该表改引 `docs/REPORT`。
3. **重跑并重裁判据②③**：Ny=64 至少跑到 t/t_ν ≳ 1（≈6 万步，可用其默认 20000 步先看趋势），冻结 ③ 的 U_max 定义（建议用同 t 瞬态解析或稳态解析，并写明）。
4. **④ 改判 N/A/FAIL**，① 增加非退化判据（界面漂移 + 独立精度指标）。
5. **method_record 与实现对齐**：删掉未做的 ν 比扫参 / ρ 比=2 声明，改正「sharp interface」与分支名，备选压到 ≤2。

## 6. 六要素自查（本审查报告）

1. **案例**：两组分 MCMP 分层 Poiseuille，D2Q9/BGK/SC(G₁₂=-1.0)/mixture 平衡/上下壁 bounce-back，g_x=1e-4，
   Ny=64(2000 步) 与 Ny=128(3000 步)，τ₁=τ₂=0.7（ν=1/15），界面 h=Ny/2。
2. **做法**：读工件+git/DB 取证 → 独立实现解析解与误差度量 → 复算②③ → 单相起动级数解分解未收敛贡献 → 复跑取 u_max(t) 史 → 交叉校验被审解析解。
3. **结果**：见上三表（全部数字来自 `recheck_results.json` 程序化生成，零手抄）。
4. **图**：`artifacts/pilot_ab_review/recheck_figure.png`（(a) 实测剖面 vs 稳态解析 vs 同 t 瞬态级数解；(b) u_max(t) 实测 vs 起动级数解 vs 稳态线）。
5. **注意点**：mtime 非协议证据；无 commit 使时序判定降级为「未证」；复算复用了被审求解器数值场；长跑未完成。
6. **误差判定**：② 复算误差 77.56%/90.57%，与本仓 docs 一致、与 SUMMARY 不一致 → **不满足 <3% 一致性要求的是 SUMMARY 的自报值**（差 27.6/30.6 个百分点）；
   解析解交叉校验 5.5e-14% ≪3%（通过）；③ 比值三口径 0.0101–0.195，基准敏感性 ≫3%（须冻结定义）。