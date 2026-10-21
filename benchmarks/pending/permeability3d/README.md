# 3D 周期球阵列 Stokes 渗透率（pending）

**状态：⏸️ superseded — φ=0.343 档已晋级 `verified/permeability3d/`（2026-10-20）**

φ=0.343 档在冻结三档与修正四档规则下双 PASS（4.63%→3.10%→2.05%→1.45% 严格单调），
已归档至 `benchmarks/verified/permeability3d/`。

稀释档（φ=0.125/0.216）因 fp32 注入地板与负 O(Re) 惯性修正拖累未达标，
根因已定源（非 solver 共性路径缺陷）。后续晋级路径见原 README 备份或 verified 目录。

## 归档内容

`benchmarks/verified/permeability3d/` 包含：
- `README.md` — 验证摘要与判定表
- `result.json` — 仅含 φ=0.343 四档数据与判决
- `validation_report.json` — 四门判据工件
- `gates.json` — 机器门禁三元组
- `NOTES.md` — 预注册参考锁定与诊断注记
- `run.py` — 共性模块入口驱动
- `verify.py` — 独立验算脚本
- `case_phi0.343_N{64,96,128,160}*.json` — 四档原始结果

## 原晋级路径（已执行）

1. ✅ **φ=0.343 档**：owner 裁定晋升——冻结三档（最细 2.05%）与修正四档（最细 1.45%）
   双过，已归档
2. φ=0.125/0.216 补跑（未执行）：fp64 或 fs≥100 + Re_p→0.05 消注入地板与惯性偏移
3. 归档 PR：`pending/permeability3d` → `verified/permeability3d`（已完成）
