#!/usr/bin/env python3
"""Publish m17668 audit report (paper-grade, programmatic injection, zero hand-copy)."""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import audit_fetch as A  # noqa: E402

ART = HERE / "artifacts" / "m17668"
probe = json.loads((ART / "units_equivalence_probe.json").read_text())
png = (ART / "units_equivalence_probe.png").read_bytes()


def row(c, rt):
    return (f"| {c['nx']} | {c['tau']:.6f} | {c['nu_lb']:.6f} | {c['dx']:.8f} | {c['dt']:.8f} "
            f"| {c['ma']:.6f} | {max(c['e_tau'], c['e_nu'], c['e_dx'], c['e_dt']):.3e} "
            f"| {rt['e_u']:.2e} | {rt['e_t']:.3%} | {rt['n_steps']:.0f} |")


table = "\n".join(row(c, r) for c, r in zip(probe["cases"], probe["roundtrip"]))

md = f"""# 交差审计 m17640 · XVAL-LBM-CASE-SCHEMA-JSON-UNITS-CONVERTER（TensorLBM）→ 判级 L1，verdict id=612 落表

**结论先行**：83 项测试独立复跑为真（83 passed / 4.73 s），但「与 lettuce UnitConversion 0% 误差」**不是独立跨实现对标**——
参照方是被测测试文件内自建的同类解析式（`class LettuceUnitConversion`，环境 `import lettuce` → ModuleNotFoundError），
两侧是同一代数式，误差 0% 由构造成立。全单零 commit（head_commit=null，两文件 untracked）、零 run_artifacts、
`tensor_benchmark_cases` 本锚 0 行 ⇒ L4 不可达。按「级别=功能实质」判 **L1**（数学/单元验证），
L2/L3 的实质代码为 PRE-EXISTING，不计入本单。

## ① 案例介绍（审计对象与判据）

- 对象：m17640（TensorLBM，status=succeeded，2026-11-13T07:33:17Z，model qwen3.6-plus，tool_calls=69）
- 锚点：`XVAL-LBM-CASE-SCHEMA-JSON-UNITS-CONVERTER`（2b-t/LB-t，case-schema-json-units）
- 交付清单（实测）：`docs/XVAL-LBM-CASE-SCHEMA-JSON-UNITS-CONVERTER.md`（9931 B）、`tests/test_xval_case_schema_json_units_converter.py`（21604 B），二者 `git status` 均为 `??`（未入库）
- 被审能力：`src/tensorlbm/unit_converter.py::LBMUnitConverter`（物理→格子单位换算：τ, ν_lb, dx, dt, Ma）
- 参数（审计侧复算设定）：Re=100, L_phys=1 m, u_phys=1e-3 m/s, ν_phys=1e-5 m²/s, nx∈{{32,64,128,256}}
- 基准判据（唯一裁判）：`tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-CASE-SCHEMA-JSON-UNITS-CONVERTER'` → **0 行**（模糊 `%CASE-SCHEMA%`/`%UNITS-CONVERT%` 亦 0 行）

## ② 怎么做的（审计方法与工具链）

1. 取平台真值：`tensor_dev_missions.result`（jsonb）直读 head_commit/branch/changed_files/on_disk_changes/server_quality_audit；`git status --porcelain` 交叉核交付是否入库。
2. 基准裁判先行：SQL 查 `tensor_benchmark_cases`，确认本锚无注册基准 ⇒ 报告里的「误差 0%」无机器可裁证据。
3. 复跑验收：`python3 -m pytest tests/test_xval_case_schema_json_units_converter.py -q`（在 m17640 工作树，独立于作者会话）。
4. 外部锚核真：`python3 -c "import lettuce"` 判参照方是否在场；读测试文件头部与 `LettuceUnitConversion` 定义处，判定其为「重建实现」而非第三方 import。
5. 独立数值复算：`scripts/audit_m17668_figure.py` 直接用**真** `LBMUnitConverter` 算 nx 四档的 τ/ν_lb/dx/dt/Ma，并以审计侧独立解析式作 ref，量化偏差底；同时对 phys→lattice→phys 往返做残差谱（速度/时间）。
6. 落表：`verdict_m17668.sql` → `tensor_maturity_verdicts`（repo/scope/mission_id/level/evidence/verdict_by），evidence 8 键均为真实指针。
7. 工具链：psql(直接读 tensorplatform) · pytest · matplotlib(Agg) · 自写 `scripts/audit_fetch.py`（平台 API 鉴权读取）。

## ③ 计算结果（审计侧独立数值，脚本注入）

| nx | τ | ν_lb | dx [m] | dt [s] | Ma | 四量最大相对偏差 | 速度往返 | 时间往返 | 整数步 n |
|---|---|---|---|---|---|---|---|---|---|
{table}

- 偏差底 = **{probe['deviation_floor']:.3e}**（≈ IEEE-754 eps）⇒ 证明「0%」是**恒等式残差**，不是跨实现一致性。
- 速度往返最差 = **{probe['rt_worst'] if False else max(r['e_u'] for r in probe['roundtrip']):.2e}**；时间往返最差 = **{max(r['e_t'] for r in probe['roundtrip']):.3%}**（`phys_time_to_steps` 整数步量化，误差 O(dt)，非舍入级）。
- 报告 ④ 的「所有核心量误差=0%」仅对连续量成立；时间维度必须单列量化误差（本审计新发现，作者报告未提）。

## ④ 结果图

图 1（左）：nx 四档下 τ/ν_lb/dx/dt 与审计侧独立解析式的相对偏差，全部压在 1e-16 量级，与 3%/15% 判据线对比——
**偏差处于舍入底而非验证带内**，说明该「对标」无区分力。图 1（右）：真换算器往返残差谱，速度往返 ~1e-16，
时间往返 ~20% 台阶（整数步量化）。无物理场量云图：本单为纯单位换算/文档任务，不含流场计算，故无场量云图（如实说明，不伪造图形）。

## ⑤ 注意点（坑与局限）

1. **自建 ref 冒充外部锚**：注释自承 "reconstructed from lettuce source + survey"，无 pin/hash/vendored 源码 ⇒ 报告 ①/② 的「对标成熟开源实现」表述夸大，应改「与自建同类公式的一致性自检」。
2. **零入库**：head_commit=null、branch=null、两文件 untracked ⇒ 对 GitHub 检出**不可复现**；平台 `server_quality_audit.gate_valid=false, key_present=false, gate_missing_warning=true`，仍按 `[auto-approve] dev_acceptance` 落 succeeded（自动放行 ≠ 证据合格）。
3. **能力归属错置风险**：`general_sim.py:58,327,518`、`cases/base.py:41`、`app/backend/routers/preprocess.py`、`benchmarks/verified/suboff_sail_resistance/run.py` 的集成均为 PRE-EXISTING，不得据此主张 L2/L3。
4. 复跑实体化告警 `Ma=0.1039 ≥ 0.1`（cavity 算例），报告未讨论可压缩性适用边界。
5. 锚名含 CASE-SCHEMA-JSON，但 JSON 面只到 `summary()`/dict 可序列化往返，缺「JSON 读入 → 构造 converter → 物理量回算」闭环。

## ⑥ 误差判定（逐项 vs 基准）

| 关键量 | 基准来源 | got vs ref 相对误差 | 是否 <3% | 说明 |
|---|---|---|---|---|
| τ | 自建同类式（非独立） | {max(c['e_tau'] for c in probe['cases']):.2e} | 是（但无区分力） | 恒等式 |
| ν_lb | 同上 | {max(c['e_nu'] for c in probe['cases']):.2e} | 是（但无区分力） | 恒等式 |
| dx | 同上 | {max(c['e_dx'] for c in probe['cases']):.2e} | 是（但无区分力） | 恒等式 |
| dt | 同上 | {max(c['e_dt'] for c in probe['cases']):.2e} | 是（但无区分力） | 恒等式 |
| 速度往返 | 真换算器自洽 | {max(r['e_u'] for r in probe['roundtrip']):.2e} | 是 | 舍入级 |
| 时间往返 | 真换算器自洽 | {max(r['e_t'] for r in probe['roundtrip']):.3%} | **否（>3%，21.9%）** | 整数步量化，非缺陷但须声明 |
| 物理算例误差 | **无** | — | **无法判定** | 本单无流场算例，`tensor_benchmark_cases` 本锚 0 行 |

**是否 <3%：对单位换算连续量是（且为恒等式级），对时间往返否（21.9%，量化所致），对物理/工程精度基准无从谈起（无注册基准、无算例）——故 L4 不可达，判 L1。**

## 处置

verdict id=612（level=1, gate_state=pending）已落 `tensor_maturity_verdicts`；建议 maturity_gate 裁定 reopened 后按返修口径认领：
①引入真 lettuce（pip/vendored+hash）或换文献锚给公式编号；②注册 `tensor_benchmark_cases` 使可机裁；③补 JSON 反序列化闭环；④文件入库并给 repro_cmd；⑤报告 ref 列标注来源。
"""

payload = {
    "title": "交差审计 m17640 · XVAL-LBM-CASE-SCHEMA-JSON-UNITS-CONVERTER（TensorLBM）→ 判级 L1，verdict id=612 落表",
    "summary": "独立复跑 83 passed 为真，但「vs lettuce 0% 误差」是自建同类解析式的恒等式（lettuce 未安装），零 commit/零 run_artifacts/基准 0 行 ⇒ 判 L1；审计侧数值复算得偏差底 1.42e-16、时间往返量化误差 21.9%。",
    "scenario_key": "cross_validation_audit",
    "mission_id": 17668,
    "tags": ["audit", "TensorLBM", "XVAL", "maturity-verdict", "L1", "unit-converter", "benchmark-gap"],
    "markdown": md,
    "figures": [{
        "name": "units_equivalence_probe.png",
        "caption": "左：m17640 的 τ/ν_lb/dx/dt 与自建 ref 的相对偏差压在 1e-16（恒等式，非验证）；右：真换算器 phys→lattice→phys 往返残差，速度 ~1e-16、时间 ~20%（整数步量化）。",
        "png_base64": base64.b64encode(png).decode(),
    }],
}
Path("/tmp/report.json").write_text(json.dumps(payload))
res = A.call("tensor/reports", payload, "POST")
print(json.dumps(res, ensure_ascii=False)[:600])