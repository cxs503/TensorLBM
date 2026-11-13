# AUDIT m18089 — `XVAL-LBM-HORIZONTAL-SURVEY-EMPTY`（m17981 交差核验：三件套是否真存在、级别判多少）

> 席位 board-audit · 仓 TensorLBM · 分支 `agent/m18089` · 被核对象 = 任务 **m17981**
> （`XVAL-LBM-HORIZONTAL-SURVEY-EMPTY`，repo TensorLBM，status=succeeded，2026-11-13 11:15Z）
> 基准裁判 = `tensor_benchmark_cases` 中该 anchor 的行（唯一裁判），报告好看不算数。
> 本单是**交差核验，非重做**：不改 m17981 的任何物理设置、不改容差、不覆盖任何 honest-FAIL 工件。
> 我的全部数字由 `tools/audit_m17981_verify.py` 从冻结工件重算生成（JSON:
> `artifacts/board_audit/m18089/audit_m17981.json`，图: `artifacts/board_audit/m18089/fig_audit_m17981.png`）。

## ① 案例介绍（对象与验收判据）

**物理问题**：圆柱在振荡流中的惯性（附加质量）系数——耐波性与 PMM 辨识的主力量。
**被核配置**：双周期 D2Q9 盒、固定圆柱、仅流体节点上加简谐体力 `a(t)=a0 cos(ωt)`（二阶 Guo 源），
全程反弹，同一算例上跑两个独立力估计器（E1 Ladd 动量交换 / E2 表面牵引），Morison 谐波最小二乘。

**anchor 的原始命题**（`oss_solutions` id=386，`oss-inventory/tensorplatform` / `GAP-DEM-CLUMP` / tag
`horizontal-survey-empty`）：水平扫描成熟实现返回空集 ⇒ 该能力只能从文献谱系供给，不能从别家代码移植。

**验收三件套（本单逐项核）**：① 真工件（impl / entry / run_artifacts / tests）；② benchmark ref vs got
数值与容差；③ `repro_cmd` 可执行。**基准判据**（registry 行，`anchor_slug='XVAL-LBM-HORIZONTAL-SURVEY-EMPTY'`）：

- metric：`C_M(E1, D_halfway)`，Morison 拟合最后 3 周期；got base = 1.5742086608150845 → err **−21.29%**；
  res16 1.6153 (−19.24%)、kc05 1.4342 (−28.29%)、beta50 1.6715 (−16.43%)、beta200 1.5134 (−24.33%)、
  box16 1.1687 (−41.56%)；2 点 1/D 外推 C_M(∞)=1.4813 (−25.93%)；predeclared 的 res32/kc20 **从未跑**。
- tolerance：`<15%`（mission m17844 预告，owner 精度标准）；生态线 `<3%`；**"gate NOT passed (status left
  pending on purpose)"**，行由审计 m17888 代注册（交付时缺行）。

## ② 怎么做的（复核方法 / 流程 / 跑过什么）

原则：**只复核，不重算物理**（本机控制面负载高、容量纪律）。所有结论从 m17981 冻结工件重算：

1. **重走执行路径**：`xval/horizontal_survey_empty/predeclared.json`（预告的 15% 判据、8 个 case、停止规则）
   → `src/tensorlbm/oscillatory_cylinder_inertia.py:265-375`（`run_arm`：几何 `:104-139`、谐波分解 `:194`、
   归一化 `:234`、同掩模势流参考 `:244`）→ `cases/oscillatory_cylinder_inertia/run.py:23`（入口包装）
   → `runs/oscillatory_cylinder_inertia/{quick,res16,base_m17844}/{result.json,trace.npz}`。
2. **独立重拟合（不信 result.json）**：从 `trace.npz` 的 `F_mem/F_trac`（旧臂 `F1/F2`）用**独立写的**
   最小二乘（1 + 2 阶谐波、按 predeclared 的整数周期窗 `[transient·T, +measure·T)`）重算 `C_I`，
   归一化体积 `V=πD_area²/4` 由迹里的 `A_solid` 反推。
3. **重导参考量**：直接调用 `potential_reference(nx,nx,D)`（`fsi_added_mass.potential_added_mass` 同掩模）
   重算 `c_a_area / c_a_halfway`，并与解析无界锚 `C_a=1` 对照；两种口径各判一次 15% 门。
4. **端到端复现**：`python -m tensorlbm.oscillatory_cylinder_inertia --case quick --out /tmp/repro_quick`；
   `python -m pytest tests/test_oscillatory_cylinder_inertia.py -q`。
5. **交互检查**：`grep -rn oscillatory_cylinder_inertia`（非自身调用者）；survey 仪器在 dump 上重跑，
   8 个 family 的 verdict 与提交件比对。
6. **本审计自身工具**：`tools/audit_m17981_verify.py`（本次新增，可复跑：
   `python tools/audit_m17981_verify.py --work /root/tp-agent/work/m17981 --out artifacts/board_audit/m18089
   --bench-row artifacts/board_audit/m18089/benchmark_case_row.json`）。

## ③ 计算结果（全部程序化注入，零手抄）

**C1 报告的 `C_I` 能否从冻结迹重算出来（交付数字的可溯源度）**

| 臂 | 报出 E1_area | 我重拟合 E1_area | 偏差 | 报出 E2_area | 我重拟合 E2_area | 判定 |
|---|---|---|---|---|---|---|
| quick (D=12, 6D, 2T 窗) | 1.6758723 | **1.6758723** | 0.0 % | 1.6164303 | **1.6164303** | 逐位复现 |
| res16 (D=16, 12D, 3T 窗) | 1.8316966 | **1.8316966** | 0.0 % | 1.8806907 | **1.8806907** | 逐位复现 |
| base_m17844（旧臂，联合 C_M/C_D 拟合） | 1.6937151 | 1.7257322 | −1.85 % | −3.0933050 | −3.1364139 | 差在旧臂把 `|U|U` 阻力基纳入联合拟合（我按纯谐波拟合），非造数 |

⇒ 新臂（m17981 交付的模块臂）**数字可从 trace 逐位重算**；旧臂差异可由拟合基解释（1.8%），不影响 −21.29% 那条登记值：
`C_M(E1,D_halfway)=1.5742420` vs Morison 2.0 = **−21.29%**，与我按 `1.5742414`（registry 值）复核一致。

**C2 两种参考口径 × 全部臂的误差（判据 = 15% 门，生态线 3%）**

| 臂 | D | box | got `C_I`(E1, area) | 同掩模势流参考 `c_a` | err vs 同掩模 | err vs 解析 `C_a=1` | 过 15% 门 |
|---|---|---|---|---|---|---|---|
| quick | 12 | 6D | 1.675872 | 1.061515 | **+57.88 %** | +67.59 % | **否** |
| res16 | 16 | 12D | 1.831697 | 1.059697 | **+72.85 %** | +83.17 % | **否** |
| base（旧臂报值） | 24 | 12D | 1.693715 | 1.044081 | **+62.22 %** | +69.37 % | **否** |
| base（旧臂 Morison 口径，registry 行） | 24 | 12D | 1.574242 (E1 halfway) | 2.000000 | −21.29 % | — | **否** |

三档分辨率 D=12/16/24 的同掩模误差 +57.9 % / +72.9 % / +62.2 %：**最小 57.9 %，仍是门的 3.9 倍**，
且不随分辨率收敛到门内 ⇒ 不是离散化伪影（与交付 §5 的定性判断一致，此处给出三档数值）。

**C3 测量不是短板**（交付头条 "离散度 222%→0.06%" 的复核）：E1 vs E2 面归离散度 quick **0.0594 %**、
res16 **0.0490 %**（旧臂 4.86 %，其 result 自报 222.5 % 属另一符号约定缺陷）⇒ 双估计器一致，误差在模型/参考，不在力测量。

**C4 参考量本身的不确定带**：同掩模势流 `c_a` 原始 1.09626（D=16、208 固体格）→ 面积归一 1.05970，
**比解析无界锚 `C_a=1` 高 6.0 %**。两种口径都 FAIL，故不影响判决；但"同掩模参考"是本仓代码算的，
不是完全外部锚（见 ⑤-F4）。

**C5 三件套与 registry 的机械核验**

| 项 | 结果 |
|---|---|
| impl_files | 模块 `src/tensorlbm/oscillatory_cylinder_inertia.py`（411 行）、`cases/oscillatory_cylinder_inertia/run.py`、survey 仪器 3 件（survey/run_arm/analyze）— **全部存在** |
| entry | `python -m tensorlbm.oscillatory_cylinder_inertia --case quick` 我实跑：`C_I=1.6758722921650897`，与工件**逐位相同**，err 57.8756 % 相同 |
| run_artifacts | `runs/oscillatory_cylinder_inertia/{quick,res16,base_m17844}/{result.json,trace.npz}` + `benchmark_table.json` — **全部存在**；trace 含 `t/F_mem/F_trac/U/omega/a0/A_solid`，可重拟合 |
| tests | `tests/test_oscillatory_cylinder_inertia.py`（8 条）— 我实跑 **8 passed in 18.69 s**（首跑 15.56 s） |
| benchmark | registry 行存在，tolerance `<15%`，got −21.29 %（旧臂口径）/ +57.9…+72.9 %（同掩模口径）⇒ **两口径均未过门** |
| repro_cmd | `pytest ...` ✓；`run.py/module --case` ✓（bit-identical）；survey 仪器 ✓（见 ⑤-F8 的 `--out` 需先 mkdir） |
| 报告六段 | `docs/XVAL_LBM_HORIZONTAL_SURVEY_EMPTY.md` §1-§8 + 3 张 PNG 全在；元素①-⑥ 齐备，但**无一句显式回答 "<3%"** |

## ④ 结果图

`artifacts/board_audit/m18089/fig_audit_m17981.png`（左：各臂 got `C_I` vs 同掩模参考柱状；右：误差 %
与 ±15% 绿带、以及 vs 解析 `C_a=1` 的虚线）——三臂全部落在绿带之外，最低 57.9 %。
（m17981 自带 `docs/assets/xval_lbm_horizontal_survey_empty/fig{1,2,3}.png` 三张，本审计**不重绘**其像素内容，
只核对存在性与报告引用，见 ⑤-"未做的检查"。）

## ⑤ 注意点（发现，按严重度分级；含对交付声明的抗辩）

**F1（阻断 L4，事实澄清：非造假）** 物理门未过：全部臂 |err| 19–83 %，判 15% 门与生态 3% 线**双双失败**。
交付把这一点写成 honest FAIL 并在 §6 定位原因（β≈115 时 δ=1.35 格、力峰滞后加速度 45°、谐波残差 0.5–0.8σ），
**诚实申报成立**，本审计确认"FAIL 不是粉饰"。

**F2（中）registry 与交付口径脱节**：登记行的 got 指向**旧臂 Morison 口径**（1.5742, −21.29 %），
不含 rework 的同掩模口径（+57.9/+72.9 %）与三档分辨率；两个口径结论同为 FAIL，但登记行**不能**独立支撑交付头条。
属"登记未随 rework 更新"，建议（下一手）在 registry 行补 same-mask 口径与 D=12/16/24 三档，**不改容差**。

**F3（中）流程：registry 行不是交付者注册的**：行由审计 m17888 代注册（reason 已写在 tolerance 文本里）。
登记责任从交付侧漂移到审计侧，是系统性缺口（红队：会掩盖"交付不带基准"）。

**F4（中）参考量的独立性打折**：同掩模 `c_a` 由本仓 `fsi_added_mass.potential_added_mass` 算（内部代码），
与解析锚 `C_a=1` 差 **+6.0 %**；口径切换（Morison `C_M=2` → 同掩模 `C_a≈1.06`）发生在**观测到 −21 % 之后**。
交付把它作为"约定纠正"并同时报出两口径（诚实），但"reference"不是完全外部量，参考不确定带 ≈6 % 必须记账。

**F5（中）预告预算未执行**：predeclared 的 `res32`、`kc20` **从未跑**（registry 自述，我核对 runs/ 只有
quick/res16/base_m17844）；docs §5 的窗口敏感性数字 `C_I=1.541`（`[1.5T,4.5T]` 窗）**没有落盘工件**，不可复核。
⇒ "收敛/敏感性已覆盖"只能算 2 档分辨率 + 1 条无工件重分析，属**有限成立**的声明。

**F6（中）集成度为 0**：`grep -rn oscillatory_cylinder_inertia` 在 m17981 全树只命中自身模块、自测、
case 入口与 docs ⇒ **非自身调用者 0**；未进 `src/tensorlbm/__init__.py`、无 capacity/registry 条目。L3 未达。

**F7（低-中）测试绿 ≠ 物理对**：8 条测试全在**辅助层**（几何读数 `:27`、谐波分解恢复合成信号 `:38`、
窗正交性 `:50`、非线性残差旗标 `:61`、归一化 `:73`、势流参考界与约定 `:79`、quick 自洽 `:86`、CLI `:104`），
**没有一条断言"物理量对参考的误差"**（也不可能，因为过不了门）。按 semantic gate：数字未对、未与基线对照。

**F8（低）m17981 自己的 repro_cmd 有两处小缺陷**：(a) `survey.py --out <dir>` **不做 mkdir**，按文档原命令跑
直接 `FileNotFoundError: .../survey_table.csv`（我 mkdir 后重跑通过，8/8 family verdict 与提交件**完全一致**，
⇒ "empty" 判决可复现）；(b) survey 的"空集"只对**本地 dump**（`oss_inventory` 33 行 / `oss_solutions` 261 行 / 45 仓）
成立，不是全网检索——OSS LBM（lettuce/lbmpy 等）自带圆柱绕流算例，族关键词（morison/keulegan/added mass/pmm）
偏窄，故"无成熟实现可移植"这一命题的**范围**是"本仓台账空"，不是"世界空"。

**F9（低）口径小混用**：`run_arm` 用**名义** `a0` 归一 `C_I`，而 KC/β 用**实测** `U` 幅值；
`a0_measured/a0_nominal=1.014` ⇒ 若改用实测 a0，res16 `C_I` 由 1.8317 降到 1.806（err 72.9 %→70.4 %）。
量级 1.4 %，不改变判决，但 KPI 定义应在文档里写明。

**F10（低）报告纪律**：平台报告 `tensor_reports.id=16915` 有 markdown（9273 字符）、**figures 数组 = 0**，
而工件里已有 3 张 PNG ⇒ 缺图且未写"无图理由"；docs/报告均**未显式回答 "<3%"**（只写了 15% 门）。

**F11（低）文档与工件不一致**：docs §3 参数表写"transient = 1.5 T, measured window = 3 T"，
而 `quick` 实际是 `transient=1.0, measure=2.0`（`ArmCase` 定义 + result.json `case` 字段）。

### 未做的检查（不能算通过）

- 未重跑物理算例（`res16` 单线程 ≈ 61 s 以上、`base` 更久；本单按容量纪律只做逐位重拟合 + quick 端到端复现）。
- 未复核 `res32`/`kc20`（**不存在**，不能核）。
- 未重绘 m17981 自带的三张 PNG 像素内容（只核存在性/字节数）；我自己的图由本审计脚本生成。
- 未用外部实验数据（无 OSC 圆柱 C_a 实测带）；外部锚只有文献理想流体 `C_a=1`。
- 未评估 `os_inventory/oss_solutions` dump 与生产库的一致性（只核 PROVENANCE 的 sha256 与行数口径）。

## ⑥ 误差判定（逐项，显式回答"是否 <3%"）

| 关键量 | got | 同掩模基准 | 误差 | <3%？ | <15%？ |
|---|---|---|---|---|---|
| `C_I`(E1,area) quick | 1.675872 | 1.061515 | +57.88 % | **否** | 否 |
| `C_I`(E1,area) res16 | 1.831697 | 1.059697 | +72.85 % | **否** | 否 |
| `C_I`(E1,area) base | 1.693715 | 1.044081 | +62.22 % | **否** | 否 |
| `C_M`(E1,halfway) base（registry 口径） | 1.574242 | 2.000000 | −21.29 % | **否** | 否 |
| 双估计器离散度 quick | E1 1.675872 / E2 1.616430 | 自洽量 | 0.0594 % | 是 | 是 |
| 双估计器离散度 res16 | E1 1.831697 / E2 1.880691 | 自洽量 | 0.0490 % | 是 | 是 |

**结论（不含糊）**：物理量对基准**没有一项 <3 %，也没有一项 <15 %**；唯一 <3 % 的是"两个估计器彼此一致"这一
**内部自洽量**，它证明的是测量管线可靠，**不是**物理正确。故本能力 = **未验证（WITHHELD）**，与交付 honest FAIL 一致。

## 判级与落表

**L2**（独立功能实现：真代码 输入→计算→输出 + 真工件 + 可复现；8 测试绿）。
- **不给 L3**：非自身调用者 0（F6），未接入主求解器/平台。
- **不给 L4**：基准门未过（F1），且 predeclared 2 个 case 未跑（F5）；"完整算例+误差分析+可复现"三件虽具备，
  但 L4 的实质是**通过基准验证**，此处结论是 FAIL ⇒ 不成立。
- **不给 L1**：已超出数学/单元验证，是完整物理算例的可执行实现（quick 端到端逐位复现）。

落表：`verdict_m18089.sql` → `tensor_maturity_verdicts('TensorLBM','m17981 XVAL-LBM-HORIZONTAL-SURVEY-EMPTY',
mission_id=18089, level=2, evidence=<JSON>, verdict_by='board-audit')`；由 maturity_gate 自动裁定。

**disposition**：`approve with notes`——交差核验通过（三件套真实、数字可溯源、honest FAIL 无粉饰），
级别 L2；F2/F5/F6/F10 作为返修建议（不阻断本单交差），F1/F4 已由熵值本身说明"能力仍 WITHHELD"。