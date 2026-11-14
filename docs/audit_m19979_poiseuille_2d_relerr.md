# Board Audit: m19979 CASE-POISEUILLE-2D-RELERR

## Audit Summary

**Mission**: m19979 — 二维 Poiseuille 速度剖面相对误差复测 (TensorLBM)
**Auditor**: m20062 (board-audit)
**Date**: 2026-11-14
**Verdict Level**: L4 (基准验证)

## ① 取证据

### 三件套核验

| 项目 | 状态 | 证据 |
|---|---|---|
| run_artifacts | ✅ 存在 | `runs/m19979_poiseuille2d/` 含 summary.json, comparison.csv, profile_rows.csv, profiles.json, raw/*.json, archived/*.json, figures/*.png |
| benchmark ref vs got | ✅ 完整 | 6 个网格配置，每个有 ref/got 数值对比； finest grid (H=60) eps_L2=0.193% |
| repro_cmd | ✅ 可执行 | `PYTHONPATH=src python benchmarks/verified/poiseuille_2d/run.py single 60 ...` |

### 报告六段核验

| 段落 | 状态 |
|---|---|
| ① 案例介绍（物理/几何/参数） | ✅ |
| ② 方法（流程/工具链） | ✅ |
| ③ 计算结果（数字表） | ✅ |
| ④ 结果图（3 张 PNG） | ✅ |
| ⑤ 注意点（踩坑/局限） | ✅ |
| ⑥ 误差判定 | ✅ |

### 基准裁判

PREREG.json 预注册阈值: eps_L2_rel < 1%

| run | eps_L2_rel | PASS |
|---|---|---|
| H20 | 1.198% | ❌ (粗网格，预期内) |
| H40 (4 variants) | 0.458%~0.716% | ✅ |
| H60 | 0.193% | ✅ |

可复现性: archived 0.1914% vs retest 0.1932%, rel_diff=0.94% < 5% → PASS

## ② 判级

**L4 = 基准验证（完整算例+误差分析+可复现）**

理由：
- 完整算例：6 个网格/参数配置，覆盖网格收敛 + 参数敏感性
- 误差分析：eps_L2, eps_inf, u_max_err, u_max/u_mean, 轴向梯度比, 质量漂移
- 可复现：archived vs retest 偏差 0.94%
- 独立验证路径：arm B (Guo 体力驱动) 旁证
- 预注册：PREREG.json 在计算前写入，阈值不事后调整
- 诚实报告：H20 FAIL、arm B steady=false、τ=0.6 超时均如实记录

## ③ 落表

```sql
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm19979 CASE-POISEUILLE-2D-RELERR', 20062, 4, '<JSON>', 'board-audit');
```

已执行，id=58。

## 发现

### 正面
1. 预注册制度执行严格（PREREG.json written_before_cluster_results=true）
2. 参考解独立生成（SymPy + FD + Richardson 三重验证）
3. 失败模式诚实记录（H20 FAIL, arm B steady=false, τ=0.6 超时）
4. 可复现性量化验证（0.94% < 5%）

### 注意
1. `tensor_benchmark_cases` 表不存在于数据库中；基准判据来自 PREREG.json 预注册
2. arm B 的 steady=false 导致 arm_B_PASS=false，但 eps_L2 本身 <1%
3. 队列工件回取失败（cj2402 tar size=0），改本机短跑——生态摩擦已记录

## 裁定

**confirmed** — m19979 的 CASE-POISEUILLE-2D-RELERR 成果有效，L4 级别确认。
