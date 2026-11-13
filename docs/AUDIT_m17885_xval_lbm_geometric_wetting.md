# m17885 交差核验：m17840 `XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE`（TensorLBM）

审计对象：mission 17840（status=succeeded，2026-11-13 09:52:33），交付 commit `adc9eea9`，
工作树 `/root/tp-agent/work/m17840`，报告 `tensor_reports.id=16792`。
本次为**交差核验**（非重做）。机器读数：`artifacts/m17885_audit/audit_m17885.json`（由
`build_audit.py` 程序化生成，零手抄）；报告 `tensor_reports.id=16805`；
结论 verdict `tensor_maturity_verdicts.id=656`（level=2, verdict_by=board-audit, gate_state=pending）。

---

## ① 案例介绍（被核验的对象是什么）

几何润湿边界条件（color-gradient LBM）：在壁面邻层把色梯度法向绕壁法向旋转到与壁面成
θ，使相邻固体格色值与该旋转法向一致（Akai et al., PRE 98, 043302 (2018)）。m17840 对标
`PorousMediaSimulation/openLBMPM` 的 *geometric-wetting-contact-angle* 做法，判据=接触角
相对误差 ≤15%（owner 精度标准）。

交付三件套（shape 核对）：

| 件 | 交付物 | 存在性 |
|---|---|---|
| 实现 | `src/tensorlbm/wetting/geometric_two_candidate.py`（384 行，`__all__` 4 项） | ✅ 在 `adc9eea9` 内 |
| 工件 | `benchmarks/xval_openlbmpm_wetting/{operator_study.json, results_cot60.json, results_ctl60.json, results.json, equivalence_check.json, phi_*.npy, fig_*.png}` | ✅ 全部在 commit 内 |
| 复现 | `operator_study.py` / `run.py` / `make_figures.py` + 单测 | ✅ 可执行（见 ③） |

## ② 怎么核的（方法/流程）

1. **基准裁判优先**：先查登记基准 `tensor_benchmark_cases`（`anchor_slug='XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE'`）→ **0 行**（全表 65 行；WETT 相关只有
   `XVAL-LBM-DIFFUSE-INTERFACE-WETTING-VIA-WALL-ORDER`）。即本锚点**没有任何登记基准案例**，
   判据只能退回 mission goal 的 15% 与报告自述指标。
2. **独立重跑**（不看报告结论，只跑工件）：
   - `PYTHONPATH=src pytest tests/test_wetting_geometric_two_candidate.py tests/test_cg_geometrical_wetting.py -q` → **36 passed**（4.3 s）。
   - `python3 benchmarks/xval_openlbmpm_wetting/operator_study.py` → 4 变体 × 27 例，**与交付 JSON 逐项逐位相同**（`max_repro_delta_m1=0.0`，`max_repro_delta_m2=0.0`）。
   - `python3 benchmarks/xval_openlbmpm_wetting/run.py --arms cot60 --steps 4000`（在 `/tmp` 副本中跑，避免污染交付树，73 s）→ `theta_capgeom=165.64`、`rel_err=176.06%`、`|u|max=6.0e-3`，**与交付 JSON 逐位相同**（`repro_delta_theta=0.0`）。
3. **调用方扫描**：`grep -rn geometric_two_candidate src/ tests/ benchmarks/` → 仅 `tests/` 与
   `benchmarks/xval_openlbmpm_wetting/`，**`src/` 内零生产调用方**；`src/tensorlbm/wetting/__init__.py`
   未导出该模块（该文件最后一次改动在旧 commit `422c59c4`）。
4. **外部锚可解析性**：报告引用的 `docs/oss-inventory-openLBMPM.md` **既不在交付 commit
   `adc9eea9`，也不在 `origin/main`**（`git cat-file -e` 均 rc=128）；可解析的只有数据库
   `oss_solutions.id=5`（`PorousMediaSimulation/openLBMPM / NEW:CAP-LBM-VALIDATION-SURFACE /
   geometric-wetting-contact-angle`），同文件实体只存在于别的任务工作树
   （如 `/root/tp-agent/work/m15182/docs/oss-inventory-openLBMPM.md`）。

## ③ 计算结果（重跑数字，程序化读出）

### 3.1 算子层（解析帽 128×128，θ∈{30…150}° × ξ∈{1.5,2,3}，27 例/变体）

| 变体 | 记录 M1 max | 重跑 M1 max | 记录 M2 ⟨\|Δφ\|⟩ | 重跑 M2 | sign flips | 15% 门 |
|---|---|---|---|---|---|---|
| V1_closed2pt（既有闭式） | 1.579e-14 % | 1.579e-14 % | 0.0063923 | 0.0063923 | 0 | 过 |
| V2_cand2pt（参考机制，交付默认） | 1.579e-14 % | 1.579e-14 % | 0.0063923 | 0.0063923 | 0 | 过 |
| V3_candIso9 | 2.368e-14 % | 2.368e-14 % | 0.0071287 | 0.0071287 | 0 | 过 |
| V4_Iso9+SolidColorDiff=0.5 | 2.368e-14 % | 2.368e-14 % | 0.0269401 | 0.0269401 | 3070 | 过 |

关键读法（审计意见，不采信报告措辞）：
- **M1 是代数往返自洽，不是精度**。解析帽 → 边界条件写 ghost → *同一模板*反读，误差 1e-14%
  量级只说明"写进去什么能读回来什么"；它无法区分正误实现（任何一致的错误也会 0 误差）。
- **V2 与 V1 在测试配置下等价**：`mean(M2_V2 − M2_V1) = −3.35e-10`（差 1e-10 量级，浮点结合
  顺序），`sign flips = 0`。即参考实现的特色机制（两候选法向选择）在镜像填充的解析帽上
  **严格退化**，XVAL 并没有触到该实现的区分性自由度。
- **唯一有量纲意义的保真度指标 M2 无阈值、无参考值**：V4 比 V1 差 4.21×（0.02694/0.00639），
  这是"不移植 SolidColorDiff"的依据，属定性排序而非过门。

### 3.2 求解器层（96×64，A=0.005，4000 步，sessile drop）

| 量 | 规定值 | 记录值 | 重跑值 | 误差 | 判据 |
|---|---|---|---|---|---|
| cap 几何接触角（参考配方，主判据） | 60.00° | 165.639° | **165.639°** | **+176.06%** | ❌ 超 15% 11.7 倍 |
| 圆拟合 θ | 60° | 81.376° | 81.376° | +35.6% | ❌ |
| BC 反读 θ（n=8） | 60° | 60.000° | — | 0.00% | ✅（但只证明写对了 BC，不证明物理角） |
| 对照臂 ctl60（关 BC） | — | 128.967° | — | 43.30%（对 90° 分母）/ 114.9%（对初始 60°） | 敏感性对照按预期 |

三条独立测量互差 **84.26°**（capgeom 165.6 vs circle 81.4），且 `h=62.19` 已达域高 `ny=64`
的 **97.2%**（`h` 在 2000→2500 步从 7.26 跳到 61.93：液帽被射流打穿到顶壁）。即末态的"帽"
根本不是帽，**cap 几何配方在该场上已失效但仍返回 `status=ok`**（估值器无 h/域高 合理性守卫）。

## ④ 结果图

- `fig_m17885_operator_solver.png`（左：算子层 M1 往返误差 vs 15% 门；右：M2 保真度四变体对照）
- `fig_m17885_solver_angles.png`（求解器层四条测量：规定 60° / capgeom 165.64° / 圆拟合 81.38° / BC 反读 60°）

## ⑤ 注意点（发现的缺陷与过度主张）

| # | 级别 | finding | 证据 |
|---|---|---|---|
| F1 | 非阻塞（诚实申报正确） | 求解器层 honest-FAIL 真实：176.06% 可逐位复现，未粉饰 | `audit_m17885.json: solver_layer` |
| F2 | 中 | "27 例解析反读 ≤2.4e-14%" 被当作精度证据；实为代数往返自洽 | `operator_study.py:118-172`（readback 与 BC 同模板） |
| F3 | 中 | XVAL 未触到参考的区分性机制（V2≡V1，flips=0）⇒"对标 openLBMPM 做法"的表述强于事实 | `operator_study.json: paired_vs_V1` |
| F4 | 中 | 无登记基准案例（`tensor_benchmark_cases` 0 行）⇒ 基准裁判缺席，判据不可机器裁定 | `audit_m17885.json: benchmark_registry` |
| F5 | 中 | 无生产集成：`src/` 零调用方，`wetting/__init__.py` 未导出；求解器层用的是 benchmark 自带 `make_step` 循环 | `audit_m17885.json: integration` |
| F6 | 低 | 引用出处 `docs/oss-inventory-openLBMPM.md` 悬空（不在本分支也不在 main） | `git cat-file -e adc9eea9:...` rc=128 |
| F7 | 低 | 对照臂误差分母不一致：报告写 115%（对 60°），工件 `rel_err_pct_capgeom=43.30%`（对 90°） | `results_ctl60.json` |
| F8 | 低 | `cap_geometry_contact_angle` 在 h≈域高时不降级（返 `status=ok`），易在迁移算例中给出假数 | `geometric_two_candidate.py:301-384` |
| F9 | 低 | 宿主 A-扰动表面张力缺陷（A≥0.01 射流打顶）已如实记录，属真实生态摩擦 | `docs/XVAL-LBM-OPENLBMPM-GEOMETRIC-WETTING.md` §6 |

未跑的检查（记 unknown 不记 pass）：openLBMPM 源码逐行复核（无外网，行号引用
`RKCG2D/AcceleratedRKGPU2D.py:1560/:1584-1632/:2430-2492` 无法验证）；`oss60/oss90/ctl90` 其余
臂求解器重算；网格/时间步收敛扫描（交付未做，审计亦未做）。

## ⑥ 误差判定与判级

- 关键量 vs 基准**逐项**回答"是否 <3%"：
  - 算子层 M1 往返误差 2.37e-14% → **是（<3%）**，但该量不表征物理精度（见 F2）。
  - 参考测量配方在解析圆帽上复现 θ：|Δ|<0.6°（θ=30…150°）→ 相对 ≤2 % → **是（<3%）**。
  - 求解器级端到端静态接触角 165.64° vs 60° = 176.06% → **否（远超 15%，更远超 3%）**；且
    两个独立测量互差 84.26° ⇒ 端到端角度在本宿主上是**不可测**，不是"偏大"。
  - 登记基准案例：**无**（0 行），故"对基准 <3%"无对象可判。

**判级 = L2（独立功能实现）**，理由（按功能实质，不按报告厚度）：
- 达到 L2：`geometric_two_candidate.py` 是真代码（输入 φ/solid/θ → 输出 ghost 场 + 诊断字典），
  含候选法向、各向同性 9 点色梯度、SolidColorDiff 三要素与 `cap_geometry_contact_angle` 估值器；
  36 项单测断言的是**不变量**（镜像填充退化、θ=90° 严格镜像、界面门限回退、clamp、非法参数抛错、
  解析圆帽角复现 <0.6°），非空跑；工件可逐位复现。
- 不到 L3：`src/` 无生产调用方，包 `__init__` 未导出，未接入 `color_gradient2d`/主求解器路径，
  求解器层评估用的是 benchmark 私有 `make_step` 循环。
- 不到 L4：完整算例的端到端判据 FAIL（176.06%），且无登记基准案例、无收敛性扫描；
  算子层"过 15% 门"过的是自洽门而非精度门。
- 附加注记（压向 L2 下沿）：交付默认配置 `stencil=central2, solid_color_offset=0` 与既有
  `cg_wetting2d.geometrical_wetting_ghost` **逐值等价**，故本单对求解器**能力的净增≈0**；
  净增落在"测量配方 + 诊断字典 + 负知识（参考机制在镜像填充下退化、SolidColorDiff 保真度差 4.21×）"。