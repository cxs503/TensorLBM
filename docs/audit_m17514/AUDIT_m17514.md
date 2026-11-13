# 交差审计 m17514 — XVAL-LBM-AA-PATTERN-EVEN-ODD-TIMESTEP-TRACKER

审计单：m17572（board-audit） · 被审计单：m17514（TensorLBM，succeeded，返修轮1 承接 m17399）
被审计工作树：`/root/tp-agent/work/m17514`（分支 `agent/m17514`）
审计工作树：`/root/tp-agent/work/m17572`（分支 `agent/m17572`）——本文所有引用文件在
`docs/audit_m17514/evidence_snapshot/`（sha256 见同目录 `SHA256SUMS.txt`）有冻结副本，
因为被审计工作树带 `worktree_recyclable` 语义且**其交付从未进 git**（见 F1）。

---

## 1 案例介绍（被审计对象是什么）

被审计交付声称实现并交叉验证 **AA 模式（单分布场原地 streaming，even/odd 内核）** 的
D2Q9 BGK 时间步进器 + **时间步奇偶跟踪器**，并对标两个参考：

* 内部参考：仓库既有 AB 模式（双场 + 指针交换，`tensorlbm.solver.stream`），要求**逐位相等**；
* 外部锚：2D Taylor–Green 涡精确解（Taylor 1923 / Green–Taylor；`K(t)=U0²/4·e^{-4νk²t}`，
  `γ_K=4νk²`），要求误差 < 15 %（owner 精度标准）。

上游锚点语义来自 `lssfau/walberla` 的 `aa-pattern-e` / `aa-pattern-o`；交付文档**自认**
沙箱无外网、取不到上游源码，因此只验证契约层（单数组/半流窗口/奇偶跟踪），不声称字节级同构。

参数口径（交付 `--quick` 配置，即入库工件实际使用的配置）：

| case | N | U0 | Re | ν | τ | Ma | steps |
|---|---|---|---|---|---|---|---|
| tg_n32_re100 | 32 | 0.05 | 100 | 0.016 | 0.548 | 0.0866 | 400 |
| tg_n48_re100 | 48 | 0.05 | 100 | 0.024 | 0.572 | 0.0866 | 400 |

## 2 怎么做的（审计方法与工具链）

1. **取证**：读 `tensor_dev_missions` 的 m17514 记录/result、`tensor_reports`（查报告）、
   `/root/tp-agent/work/m17514` 工作树、`tensor_benchmark_cases`（查机器裁判行）。
2. **三件套复核**：run_artifacts 是否真存在且是跑出来的；benchmark 的 ref vs got 是否真在容差内；
   repro_cmd 是否一键可执行（我**亲自重跑**，且在不含作者任何缓存的隔离副本中重跑）。
3. **独立复算**：`docs/audit_m17514/independent_audit.py`
   * 探针 A：隔离副本重跑 bench → 与入库 `result.json` 逐字段比对（只允许 wall time 不同）；
   * 探针 B：**自写** D2Q9（自写 c/w/f_eq/BGK + `np.roll` 平移）作为独立参考，与 `AAPatternStepper`
     对比 50 步，检查 AA 原地置换的数值正确性（不复用作者任何求解器内核）；
   * 探针 C：变异探针（去掉奇偶翻转 / 置换偏一行 / 两半集重叠）→ 判据是否抓得住；
   * 探针 D：存储口径复核（§7 表格 vs 代码实际分配）；
   * 探针 E：文档陈述 vs 交付自身数字的一致性。
4. **图**：`docs/audit_m17514/gen_figs_audit.py`（全部数字从工件 CSV/`result.json` 读入，
   零手抄）：`fig1_gammaK_…png`、`fig2_field_…png`（独立 400 步重跑场量云图 vs 闭式解 vs 差场）、
   `fig3_parity_hazard.png`。
5. **判级**：按任务的 L0–L5 定义 + `maturity_gate.py` 的 SCHEME（L2=impl+entry+run_artifacts；
   L3 追加 integration_callers；L4 追加 benchmark+repro_cmd；L5 追加 external_anchor）逐条机器可核地填 evidence。

## 3 计算结果（ref vs got，逐项；基准裁判=唯一裁判）

**机器裁判行缺失**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE
anchor_slug='XVAL-LBM-AA-PATTERN-EVEN-ODD-TIMESTEP-TRACKER'` → **0 行**。
即本锚点在平台基准表里没有裁判行，实际裁判只能是交付自带的 `criteria` 列表（下表的 ref 列）。
→ 平台级缺口（记为 F6），不影响本次数值核验，但意味着"L4 走 TensorBench 跑分实证"这条路径在本锚点上未接通。

交付 `artifacts/xval_aa_pattern_even_odd/result.json` 判据（我隔离重跑逐条复现，见探针 A）：

| 判据 | ref（独立口径） | got | 阈值 | 判 |
|---|---|---|---|---|
| aa_vs_ab_bitwise_max_abs | AB 参考逐位 | 0 | < 1e-12 | ✅ |
| gamma_K_vs_exact_worst_pct | γ_K=4νk² 闭式 | 0.2809 % | < 15 % | ✅ |
| field_l2_vs_exact_worst_pct | TG 闭式场 | 1.4485 % | < 15 % | ✅ |
| mass_drift_rel_worst | 0 | 4.05e-14 | < 1e-12 | ✅ |
| mom_drift_rel_worst | 0 | 4.46e-16 | < 1e-12 | ✅ |
| kernel_order_immaterial | 0 | 0 | < 1e-12 | ✅ |
| parity_gate_refuses_half_state | 1 | 1 | == 1 | ✅ |
| half_window_hazard_rough_field_pct | — | 115.55 % | ≥ 15 % | ✅ |
| half_window_mass_conserved | 0 | 1.11e-16 | < 1e-12 | ✅ |
| parity_blind_arm_error_pct | — | 168.27 % | ≥ 15 % | ✅ |

分分辨率：γ_K 误差 −0.1056 %（N=32）/ +0.2809 %（N=48）；场相对 L2 1.4485 %（N=32）/ 1.4226 %（N=48）。

**我的独立复算（不采信交付自评）**：

| 探针 | 结果 | 结论 |
|---|---|---|
| A 隔离副本重跑 | 10/10 判据 PASS；除 `wall_s`/`s_per_step` 外**全字段逐位相同** | 可复现：真 |
| B 自写参考 50 步（N=16、N=32） | `max|f_AA − f_own| = 0.0`（步 11 与步 50 均为 0） | AA 原地置换数值正确：真 |
| B' 独立 400 步重跑（N=48） | 场相对 L2 = **1.4226 %**（交付 1.4226 %），误差 vs 闭式 < 3 % | 15 % 闸门远满足 |
| 旁证 | `k_history_aa_*.csv` 与 `k_history_ab_*.csv` sha256 **相同** | AA≡AB 逐位一致（独立佐证） |
| C 变异 M2（置换偏一行） | bench 崩在 `aa_pattern.py:206` `IndexError` | 错误被暴露（非静默） |
| C 变异 M3（两半集重叠） | bench 非零退出（相位/分割断言拒绝） | 错误被暴露 |
| C 变异 M1（**去掉奇偶翻转**） | 10/10 判据仍 PASS，且各判据值与对照**逐位相同** | 奇偶翻转对数值**不承重**（见 F2） |

## 4 结果图

* `fig1_gammaK_aa_vs_ab_vs_exact.png` — 两分辨率下 K(t) 衰减：AA、AB、闭式解三线（CSV 实测）。
* `fig2_field_aa_vs_exact_independent.png` — 独立 400 步重跑的 ux 场 / 闭式场 / 差场（云图）。
* `fig3_parity_hazard.png` — 半流窗口读误差与 parity-blind 误差对 15 % 闸门的对比。

## 5 注意点（发现清单，severity 排序；含复现）

### F1（blocking-integrity）交付从未进入 git，m17514 = 只在本地工作树里的成果
`git -C /root/tp-agent/work/m17514 status --short` → `M src/tensorlbm/__init__.py` +
5 个 `??`（`src/tensorlbm/aa_pattern.py`、`benchmarks/…`、`tests/…`、`docs/…`、`artifacts/…`、`figures/…`）；
`git ls-files src/tensorlbm/aa_pattern.py` → 空；分支 head 仍是 `c05d7af3 Merge #2000`。
mission result 同步为 `changed_files: []`、`head_commit: null`、`branch: null`、
`server_quality_audit.gate_valid=false, key_present=false`。
→ 结论：m17514 的"成功"是**未入库的本地工件**；工作树一旦回收（该单 `worktree_recyclable` 语义 + 已有
大量同族工作树被回收的先例），能力即消失，且 gate 的 evidence 指针全断。
本审计已把全部工件冻结到 `docs/audit_m17514/evidence_snapshot/`（含 sha256）以保住证据链。
**返修动作**：把这 6 个路径提交到 `agent/m17514` 再推（内容我已验证可复现，无需重做数值）。

### F2（substantive-semantics）"跟踪器承重"只对**相位断言**成立，对**奇偶翻转**不成立，文档措辞过度
文档 §1/§6 称"跨越时间步，两内核角色按奇偶交替，跟踪器丢失交替 = 同一半集被搬两次、另一半一次没搬"。
实测（探针 C·M1）：把 `step()` 里的 `timestep_parity` 翻转删掉 → 全部判据 PASS 且数值与对照逐位相同。
原因：本实现每个时间步都分别搬完两个半集，翻转只决定"谁先"，而两半集作用在不交集上行，先手无关
（作者自己的 `kernel_order_immaterial` 判据正是这条）。真正承重的是 `_require_complete` 相位闸门
（半流窗口直读 hazard 实测 6.62 % 光滑场 / 115.55 % 噪声场），不是 parity 翻转。
作者的 `parity_blind_step()` 是**自造**的断裂臂（连搬同一半集两次），不代表"丢掉翻转"这一自然变异。
→ 建议：文档把"tracker 承重"限定为"相位闸门承重"，并补一条 M1 式判据（去掉翻转须逐位不变）作为
正反两面的证据；否则该结论属 overstatement（不改变 L 级别，但结论强度需下调）。

### F3（documentation-defect）文档 §6 的一句话被自己 §4 的数字否证，且残留模板占位符"（无）"
文档原文："注意 γ_K 这一项并不可靠：（无） 的 γ_K 偏差仍 < 15% 闸门——只看衰减率会放过这个缺陷"。
交付自身数据：parity-blind 臂 γ_K 偏差 **+36.83 %（N=32）/ +19.41 %（N=48）**，两者都 **> 15 %**。
→ 该句为假（应为"γ_K 会以 19–37 % 暴露该缺陷，但仍远小于场误差 166–168 %，故须看场误差"），
且"（无）"是渲染模板未填的占位残留（`doc_mentions_placeholder_artifact=true`）。
复现：`docs/audit_m17514/independent_audit.py` 探针 E。

### F4（overstated-claim）§7 存储口径表与代码实际分配不符，本实现的"省内存"结论反了
| 项 | 文档 §7 | 实测（探针 D，N=48） |
|---|---|---|
| 每步临时 | "每方向一行 O(N) 行缓冲" | 每方向一个**完整 (ny,nx) 平面**（2304 元素；一行仅 48）——`self.f[i][sy,sx]` 高级索引必须物化整平面 |
| 索引缓存 | "8·N 索引" | 每方向 2·N 个 int64 → 8 方向 = 16·N 值 = **128·N 字节**（294912 B，文档少算 16 倍） |
| 合计对比 | 未做合计 | AA = 165888 + 294912 = **460800 B** > AB 双场 331776 B（**多 129024 B**） |

→ AA 的"单持久场"结构声明成立（`n_persistent_fields=1`），但"AA 省内存"这一**价值主张在本实现下不成立**：
索引缓存 1.78× 场本身。文档把该表标注为"结构声明，非实测"并只列两个原始字节数、不做合计，读者会
得到相反结论。建议：补合计行 + 注明索引缓存可复用/可现场生成以恢复内存优势，或明确"本实现的 AA 只
证明数值等价，不证明内存收益"。

### F5（evidence-scope）入库工件 = `--quick`（CI）配置，而完整配置从未跑过
`benchmarks/xval_aa_pattern_even_odd.py:626-643`：`--quick` → N=32/48、400 步；**默认（完整）** →
N=64/128（Re=100）+ N=128（Re=320）、1000 步。交付 `result.json` 的 `specs` 是 32/48/400，即 CI 配置。
文档 §5 以这两点声称"两分辨率 + 两参数点"，但两者是**同一 Re、不同 N**，"两参数点"之说偏弱，
且 Re=320 点缺失；两个分辨率下场误差 1.4485 %→1.4226 %（跨度 −1.8 %）并未下降，更像 Ma=0.0866 的
可压缩性误差底噪（O(Ma²)≈0.75 % 量级），不足以支撑"网格收敛"的表述。
（审计方尝试在隔离副本跑完整配置做旁证：本地 shell 有 120 s 上限、后台进程被工具进程组收割，`docs/audit_m17514/full_config_probe.py` 未能跑完 → 完整配置**至今无人跑过**，F5 成立；该探针保留在审计包里供返修单直接执行。）

### F6（platform-gap）本锚点在 `tensor_benchmark_cases` 无裁判行
0 行 → "基准判据是唯一裁判"在本锚点上退化为"交付自带判据列表"。平台侧应把该锚点的
（case_name, metric, tolerance）行补进基准表，否则 gate 无法独立复核 got。

### F7（report-debt）m17514 无任何论文级报告
`tensor_reports` 中 mission_id=17514 的行数 = **0**，`tensor_dev_missions.report_id` 为 NULL，
但 `report_owed=false`（欠账被静默抹平，未见报告）→ 违反"报告未发=本单未完成"纪律。

### 非问题（已核验为真，记录以免下轮重复质疑）
* `aa_pattern.py` 非占位：353 行真实现（半集划分断言、行置换缓存、原地双内核、相位状态机、
  parity 计数/日志），`impl_real` 判据通过；`__init__.py` 公共 API 导出可 import（实测）。
* 测试 15 条为**不变量断言**（单持久场字节数、单内核对 ≡ AB streaming、多步逐位等价、
  探针不扰动、质量动量守恒、半流拒读、输入契约、不别名调用方 buffer），非"执行即通过"。
* `repro_cmd` 可执行：`python benchmarks/xval_aa_pattern_even_odd.py && pytest tests/test_aa_pattern_parity.py -q`
  → 我在隔离副本跑通 10/10 PASS，且 pytest 15 passed（7.3 s）。
* 数值不虚：AA vs AB 逐位为 0 且我用**自写**参考独立复现 0.0；TG 场误差 1.42 % 我用独立 400 步重跑复现。
* 外部锚真实存在（Taylor–Green 闭式解，教科书级严格解），但**只是解析解**，无实验/文献离散数据集
  → 不构成 L5 所需的外部实验锚。

## 6 误差判定（显式回答"是否 < 3%"）

| 关键量 | 基准 | 交付 got | 我的复算 | 误差 | <3 %？ | <15 %？ |
|---|---|---|---|---|---|---|
| γ_K（N=32） | 4νk²=2.4674e-3 | −0.1056 % | 判据值复现 | 0.11 % | **是** | 是 |
| γ_K（N=48） | 4νk²=1.6449e-3 | +0.2809 % | 判据值复现 | 0.28 % | **是** | 是 |
| 速度场 L2（N=32，t=400） | TG 闭式 | 1.4485 % | 判据值复现 | 1.45 % | **是** | 是 |
| 速度场 L2（N=48，t=400） | TG 闭式 | 1.4226 % | **1.4226 %**（独立重跑） | 1.42 % | **是** | 是 |
| AA vs AB | 逐位 | 0 | **0.0**（自写参考，50 步） | 0 | 是（<1e-12） | 是 |

即：数值面**全部 < 3 %**，远优于 15 % 闸门；误差不满足的是**交付/证据形态**（F1/F3/F4/F5/F7），
不是物理量。

## 7 分级判定：**L2**

逐级核（按 `maturity_gate.SCHEME`，evidence 键即契约）：

| 级别 | 要求 | 证据 | 判 |
|---|---|---|---|
| L1 | tests | `tests/test_aa_pattern_parity.py`（15 tests，15 passed，我亲跑） | ✅ |
| L2 | impl_files + entry + run_artifacts | `src/tensorlbm/aa_pattern.py`（353 行真实现，非占位）；entry=`benchmarks/xval_aa_pattern_even_odd.py` + 公共 API `tensorlbm.AAPatternStepper`；run_artifacts=`artifacts/xval_aa_pattern_even_odd/result.json`（10 判据 + 6 CSV + 3 PNG，我隔离重跑逐位复现） | ✅ |
| L3 | + integration_callers | **无**：全仓 `grep -rn "aa_pattern\|AAPattern"` 除本模块自身/测试/benchmark 外只有 `src/tensorlbm/__init__.py` 的**导出登记**（第 77-81 行、2961-2962 行），无任何主求解器/平台侧调用点（`src/tensorlbm/solver.py`、`app/backend/` 均 0 引用） | ❌ |
| L4 | + benchmark + repro_cmd | benchmark（ref vs got 齐备、误差 0.28 %/1.45 %、逐位 0）、repro_cmd（我亲跑 10/10 PASS）**在实质上已满足** | ⚠️ 实质达标但受 L3 阻断 |
| L5 | + external_anchor | 只有 Taylor–Green 解析解，无实验/文献离散数据，且上游 walberla 源码不可得 | ❌ |

结论：**综合级别 = L2**。理由：功能实质 = 一个真实现、可复现、经独立复算的 AA 步进器 + 强基准证据，
但**未接入主求解器/平台**（0 调用点），按累积口径不能算 L3/L4；L5 不成立（无外部离散锚）。
同时如实记录：**L4 的实质要件（完整算例+误差分析+可复现）已具备**，缺的只是集成环节与入库形态。

## 8 未跑 / 未覆盖（unknown ≠ passed）

* 未与 walberla `aa-pattern-e/o` 上游源码做逐行/字节级对比（沙箱无外网；交付亦然）→
  "与上游同构"仍是 **未验证**，不是"通过"。
* 未覆盖非周期边界（bounce-back/ghost 在半流窗口内的通信）、未覆盖 `ND = SPEEDS + PAD` 双槽布局、
  未覆盖 AMR/多 GPU —— 交付已自认，我也未补验。
* 未做 Ma 细化研究（Ma 单一 0.0866）→ 无法把 1.4 % 场误差分解为离散/可压缩性两部分。
* 未审计该锚点的 TensorBench 跑分（平台无裁判行，无从跑分）。

## 9 处置（disposition）

**request changes（有限）**：数值交付**接受**（可复现、独立复算一致、<3 %），级别 **L2 确认**；
以下四项须随返修闭环（不涉及重算数值）：
1. **F1**：把 6 个路径提交进 `agent/m17514` 并推送（当前成果只存在于可回收工作树）；
2. **F3**：修 §6 那句自相矛盾的陈述 + 清掉"（无）"占位；
3. **F4**：§7 表补实测合计行，并把"省内存"降级为"结构声明（本实现索引缓存 > 场本身，无内存收益）"；
4. **F2/F5**：把"tracker 承重"限定为相位闸门；§5 明确写"入库工件为 CI 尺寸配置（N=32/48，Re=100）"，
   删去隐含的收敛/两参数点强表述。
F6/F7 记平台侧欠账（基准裁判行、报告）。