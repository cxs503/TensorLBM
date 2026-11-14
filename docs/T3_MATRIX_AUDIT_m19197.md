# T3 validation-matrix audit (GAP-LBM-T3-BENCH) — coverage, substitutions, evidence

- 审计者：verifier 席（独立复核，作者不在环内）
- 对象：`benchmarks/validation_matrix/` matrix v1.1（12 案例 / 9 谱系）
- 日期：2026-11-14
- 结论摘要：**矩阵的"案例判决"可复现（12/12 与冻结证据一致），但"9 谱系覆盖"此前从未被机器判定；实测 = 6 验证 / 1 部分 / 2 缺口**。
  另外发现 3 项可复现的证据/口径缺陷（其中 1 项影响标题数字的解读）。

## 1. 审计范围与执行路径

| 检查 | 命令 | 结果 |
|---|---|---|
| 参考锚点出处（零手抄） | `python -m benchmarks.validation_matrix audit` | 全部锚点重算一致 |
| 冻结证据重判 | `python -m benchmarks.validation_matrix check` | 12 案例：passed 8 / failed 3 / pending 1，与 `STATUS_MATRIX.md` 一致 |
| 文档状态漂移 | `tests/test_validation_matrix.py` | 25 passed（含新契约测试） |
| 谱系覆盖（新增） | `python benchmarks/validation_matrix/t3_coverage.py` | **6/9 verified, 1 partial, 2 gap** |
| 槽道门数复算（新增） | `python benchmarks/validation_matrix/channel_budget.py` | 记录值 5.331% 复算 5.331%，Δ=0.0000 pp |
| 证据 sha256 链 | `tests/...::test_status_files_match_fresh_evaluation` | 一致 |

未执行（如实记录）：未重跑任何求解器案例（矩阵 `check` 语义即重判冻结证据）；未做 3-D 自由面/Michell 波浪算例、未做 Re_tau=180 的加密 DNS 级重算（预算外，见 §5）。

## 2. 关键发现（按严重度）

### F1（高，已修）"passed 8" 被当成了 T3 谱系覆盖
矩阵只判**案例**。`channel_core_path_ci`（层流槽道，解析 Poiseuille）与
`channel_turb_re180`（湍流槽道，MKM-1999 DNS）共用 `channel` 槽，于是层流案例的 passed
在状态表里与湍流谱系并列，读起来像"槽道谱系已验证"。实测：该槽 `state=gap`
（`T3_COVERAGE.json` → `slots[3].substitute_evidence` 记录 `channel_core_path_ci`
为 substitute，physics=wall_bounded_laminar）。
同类：`ship` 槽只有阻力（且为代理锚，见 F3），波浪分量 WITHHELD ⇒ `partial`。
**处置**：新增 `t3_slots.json`（逐槽 required physics/components + 逐案例 physics_class/
level_basis/credit_caveats）与 `t3_coverage.py`（机器判定 + `--check` 漂移门），
headline 由"案例 passed 数"改为"谱系 verified 数"。CI：`tests/benchmarks/test_t3_coverage_gate.py`。

### F2（高，已修）槽道槽的证据被登记为 "unresolved"，实测 FAIL 阶梯不可见
`matrix.json` 中 `channel_turb_re180.measure.artifact` 指向
`benchmarks/pending/channel_turb_3d/result.json`，该文件的 `grids.*` 路径不存在，
因此状态表该行是 `— / unresolved`，而真实测量（20.43% → 11.67% → 5.33%，严格单调）
冻结在 `validation_report.json` 里。审计把阶梯接入覆盖账（`slots[3].measured_ladder`，
由 `validation_report.json` 程序化读取），并新增门数复算模块。
**处置**：`channel_budget.py` + `gate_reproduction.json`；测试固化 `Δ≤0.01 pp`。

### F3（中，已修/已披露）船体阻力槽用的是"平板摩擦代理锚"
`ship_resistance_suboff` 的参考值是 Blasius 层流平板 `Cf=0.041995`（Re=1000，湿面积 πDL），
而该案例计算的是 SUBOFF 裸艇体**总阻力**（摩擦+压差，D3Q19/MRT）。这是代理锚，不是该
Re 下的船体实验；工件自身已记录 `TODO Ct=0.004` 属于全尺度 Re=2e6、不适用。
−1.15%（L=80）建议表述为"总阻力对平板摩擦系数的一致性"，不能表述为"船体阻力实验对标"。
**处置**：`t3_slots.json` 记 `credit_caveats: [friction_proxy_anchor]`，覆盖账中该分量为
`proxy`，`ship` 槽因此不能算 verified。

### F4（中，已披露/需口径声明）槽道门数是归一化敏感量：5.33% ↔ 15.8%（×2.96）
记录值 5.331% 只在 `u_tau = sqrt(target_wall_shear)`（目标摩擦速度）+ wall origin 0.5 +
相对 RMS 口径下成立（本审计逐位复现，Δ=0.0000 pp）。但 `u+` 的常规定义用**实测**摩擦速度
（本例低 7.64%），同一冻结场给出 **15.798%**。因此该门数不是"无口径的 5.33%"；矩阵条目
未声明归一化口径。
**处置**：`uplus_conventions` 五种口径全部入账；"FAIL" 结论在两种口径下都成立（更严）。

### F5（低，已记录）Taylor-Green 的"收敛"是同一算例的时间分段
`taylor_green_2d` 的两个 level 是一次运行的前/后半段（拟合稳定性），不是网格/分辨率序列。
解析衰减率对照本身成立，故槽位仍 verified，但 `level_basis=temporal_split` 已在契约里标注，
不得表述为"两网格收敛"。

### F6（低）矩阵 README 表格停留在 v1（称为 3 失败），与 v1.1 状态表不一致
状态表为准；本审计不以 README 叙述作为证据来源（CI 只锁 `STATUS_MATRIX.md` 与
`T3_COVERAGE.json` 两个机器产物）。

## 3. 被夸大的表述（逐条）

| 表述 | 问题 | 正确说法 |
|---|---|---|
| "9 算例谱系全部覆盖 / passed 8" | 案例数≠谱系数，层流替代品计入了湍流槽 | 6/9 verified，1 partial（ship），2 gap（channel, rayleigh_taylor） |
| "槽道案例证据 unresolved" | 真实测量阶梯存在 | 3 档单调 FAIL：20.43% → 11.67% → 5.33%（>3% 门） |
| "SUBOFF 阻力误差 −1.15%，对标实验" | 锚是平板摩擦代理，Re=1000 | 与 Blasius 摩擦代理 −1.15%，非船体实验 |
| "槽道 u+ 误差 5.33%" | 归一化敏感 | 目标 u_τ 口径 5.33%，实测 u_τ 口径 15.80% |

## 4. 未判定 / 未执行（unknown, not "passed"）

- 3-D 自由面波浪对 Michell：WITHHELD（成本，λ/Δx≥20 需 ~35 h），本次未触碰。
- Rayleigh-Taylor：两次模型类均无增长窗口（SCMP 参数面 + VOF 结构），本次未重跑，保持 failed。
- Re_tau=180 加密重算（Lx/h≥12.6、dy+≤3）：预算外，未递交集群；若后续跑，判据应沿用
  `rms_y30_100_pct` 且**必须声明** u_τ 归一化口径（见 F4）。
- 未做 `matrix.json` 的 schema 变更（F2 的 artifact 重指向）：会改动 matrix.json sha256 并使
  12 份冻结 `validation_status.json` 的证据链失效；本次以覆盖账 addendum 承载修正，重指向
  留待一次专门的矩阵版本升级（v1.2）统一刷新证据链。

## 5. 处置（disposition）

**approve with notes**：矩阵的案例判决与证据链可复现、无隐藏通过；但"谱系覆盖"标题必须改用
机器判定（6/9）并披露三类替代/代理证据。已落地的修复：`t3_slots.json`、`t3_coverage.py`、
`channel_budget.py`、`make_t3_figures.py`、`tests/benchmarks/test_t3_coverage_gate.py`
（9 项契约测试）。剩余缺口（channel / rayleigh_taylor / ship-wave）保持 honest FAIL /
WITHHELD，不得粉饰。

## 6. 复现

```bash
PYTHONPATH=src python -m benchmarks.validation_matrix check          # case verdicts
PYTHONPATH=src python benchmarks/validation_matrix/t3_coverage.py    # 6/9 lineage coverage
PYTHONPATH=src python benchmarks/validation_matrix/channel_budget.py # gate reproduction
PYTHONPATH=src python benchmarks/validation_matrix/make_t3_figures.py
PYTHONPATH=src python -m pytest tests/benchmarks/test_t3_coverage_gate.py -q
```