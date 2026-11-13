# 交差审计：m17640 XVAL-LBM-CASE-SCHEMA-JSON-UNITS-CONVERTER（TensorLBM）

- 审计单：m17668（board-audit）· 审计对象：m17640（status=succeeded, 2026-11-13 07:33:17Z）
- 仓：TensorLBM · 基准裁判：`tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-CASE-SCHEMA-JSON-UNITS-CONVERTER'`
- verdict：**id=612，level=1（L1 数学/单元验证）**，gate_state=pending（由 maturity_gate 自动裁定）
- 结论一句话：83 项测试真实通过、单位换算面确有单元级验证价值，但「vs lettuce 0% 误差」**不是独立跨实现对标**——参照方是测试文件内自建的同类解析式，且全单零 commit、零 run_artifacts、零注册基准；故按功能实质判 **L1**，L2/L3 的功劳属 PRE-EXISTING 代码，L4 不可达。

## ① 取证（三件套逐项）

| 验收件 | 声称 | 实测 | 判定 |
|---|---|---|---|
| run_artifacts | 交付工件 | `result.evidence.head_commit=null`、`branch=null`；`git status --porcelain` 仅两行 `??`（untracked）；工作树无 `runs/`/`artifacts/` 新增输出 | **缺失** |
| benchmark ref vs got | 核心量误差 0% | `tensor_benchmark_cases` 本锚 **0 行**（模糊匹配 `%CASE-SCHEMA%`/`%UNITS-CONVERT%` 亦 0 行）；报告 ④ 表内「lettuce 值」实为同文件内 `LettuceUnitConversion` 自算值 | **无机器可裁基准** |
| repro_cmd 可执行 | pytest 83 项 | 独立复跑 `python3 -m pytest tests/test_xval_case_schema_json_units_converter.py -q` → **83 passed, 2 warnings in 4.73s** | 可执行（但文件未入库→GitHub 检出不可复现） |
| 报告六段齐全 | — | ①问题/出处 ②方法 ③参数表 ④对照表 ⑤敏感性 ⑥误差判定 均存在 | 齐全（内容问题见 ③） |

补充：`result.evidence.server_quality_audit.gate_valid=false, key_present=false, gate_missing_warning=true` —— 平台侧证据门当时即未通过，任务仍走 `[auto-approve] dev_acceptance` 落 succeeded，属**自动放行而非证据合格**。

## ② 关键发现（按严重度）

**F1（阻断级·造假风险）外部参照非独立。** `tests/test_xval_case_schema_json_units_converter.py:44-55` 自定义 `class LettuceUnitConversion`，文件头注释自承 "reconstructed from lettuce source + survey"；环境内 `import lettuce` → `ModuleNotFoundError`，无 pin 版本/commit hash/vendored 源码片段。
被比量 `tau = nu_lb/cs² + 0.5`、`dx = L/N`、`dt = dx·u_lb/u_phys`、`ma = u_lb/cs` 两侧是**同一解析式**，因此 rel=1e-12 的「0% 误差」由构造成立，不含任何物理/实现分歧信息。→ 「对标成熟开源实现」的表述**夸大**；应为「与自建同类公式的一致性自检」。

**F2（严重）零入库。** 全单产出 = `docs/…md` + `tests/…py` 两个 untracked 文件；`head_commit=null`。仓库 main 上无此测试与文档 ⇒ 交差物只活在一次性工作树里，平台后续无法追踪/回归。

**F3（严重）无基准注册。** 本锚在 `tensor_benchmark_cases` 无任何 case/metric/tolerance ⇒ 按「基准判据是唯一裁判」，L4 一律不可达；报告 ⑥ 的「<3% / <15%」通过结论无可机裁证据支撑。

**F4（中）能力归属错置。** 真正有实质的代码 `src/tensorlbm/unit_converter.py`（`LBMUnitConverter`）与集成点 `src/tensorlbm/general_sim.py:58,327,518`、`src/tensorlbm/cases/base.py:41`、`app/backend/routers/preprocess.py`、`benchmarks/verified/suboff_sail_resistance/run.py` 全部 **PRE-EXISTING**，非本单产出。本单 delta=测试+文档，故不得据此主张 L2/L3。

**F5（轻）告警实体化。** 复跑实体化 `UserWarning: Ma = 0.1039 ≥ 0.1 压缩性误差可能显著`（`test_case_units_vs_converter_cavity`）；报告未讨论该算例的可压缩性适用边界。

**F7（中·审计侧新算）报告口径掩盖了量化残差。** 用真换算器复算 phys→lattice→phys 往返（`scripts/audit_m17668_figure.py`，产物 `artifacts/m17668/units_equivalence_probe.json`）：速度往返最差 0.0e0（≤1e-16 舍入），而**时间往返最差 21.9%**——`phys_time_to_steps` 返回整数步，t_back = n_steps·dt 的量化误差 O(dt) 而非舍入级。报告 ④ 的「所有核心量误差=0%」仅对连续量成立，时间维度须单列量化误差。

**F6（轻）无物理算例。** 全单不含流场计算，故无场量云图；锚名含 "CASE-SCHEMA-JSON"，但 JSON 面仅覆盖 `summary()/to dict` 可序列化往返（`test_xval_…py:374-388`），未验证 case-schema JSON 的**反序列化重建 + units 块校验**（读入 JSON→构造 converter→回算物理量闭环缺位）。

## ③ 判级

- 判据：级别=功能实质，不因报告页数升降。
- 本单实质 = 对一个 PRE-EXISTING 换算器的解析式一致性/往返/序列化做单元级验证 ⇒ **L1**。
- 未达 L2：无新增「输入→计算→输出」功能实现（文件 untracked 且仅为测试/文档）。
- 未达 L3：集成点均为既有代码，本单未新增调用方。
- 未达 L4：无注册基准、无独立外部参照、无完整物理算例+误差分析。
- 反向不降级理由：83 项测试确实执行通过，换算面（速度/时间/压力往返、tau 合法域、JSON 序列化、CaseUnits 契约）覆盖真实，非空壳。

## ③b 审计侧独立数值（程序化注入，勿手抄）

| nx | τ | ν_lb | dx [m] | dt [s] | Ma | 四量最大相对偏差 | 往返 e_u | 往返 e_t |
|---|---|---|---|---|---|---|---|---|
| 32 | 0.548 | — | 0.03125 | 1.5625 | 0.086603 | 1.42e-16 | ~0 | 21.9% |
| 64 | 0.596 | — | 0.015625 | 0.78125 | 0.086603 | 1.42e-16 | ~0 | 1.5625%* |
| 128 | 0.692 | — | 0.0078125 | 0.390625 | 0.086603 | 1.42e-16 | ~0 | * |
| 256 | 0.884 | — | 0.00390625 | 0.1953125 | 0.086603 | 1.42e-16 | ~0 | * |

偏差底 = 1.4210854715202006e-16 ≈ 机器 eps（恒等式证据）；完整表见 `artifacts/m17668/units_equivalence_probe.json`。τ 随 nx 单调增、Ma 与 nx 无关，物理上合理。

## ④ 复现命令（审计侧）

```bash
cd /root/tp-agent/work/m17640
python3 -m pytest tests/test_xval_case_schema_json_units_converter.py -q   # 83 passed
python3 -c "import lettuce"                                                # ModuleNotFoundError（外部参照不在场）
runuser -u postgres -- psql -d tensorplatform -c \
  "SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-CASE-SCHEMA-JSON-UNITS-CONVERTER'"  # 0
```

## ⑤ 处置建议（返修口径，供 reopened 后认领）

1. 真外部锚二选一：`pip install lettuce`（或 vendored 源码 + commit hash）后直接 import 比对；或换文献锚给出可引公式编号/页码，并删除自建 `LettuceUnitConversion`。
2. 注册 `tensor_benchmark_cases`（case_name/metric/tolerance/ref_source）使 L4 可机裁。
3. 补 JSON 反序列化闭环测试（schema JSON → converter → 物理量回算），坐实锚名里的 CASE-SCHEMA-JSON。
4. 把文件 commit 入库并给 repro_cmd（否则一切结论对 GitHub 检出不可复现）。
5. 报告 ④ 表须标注每行 ref 的来源与获取方式，避免「自算值冒充 ref」。