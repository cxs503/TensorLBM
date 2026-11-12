# 交差审计 m15917 — CASE-MCMP-CAPILLARY-RISE（毛细上升 Jurin 定律，TensorLBM）

审计单 m15936（board-audit）· 2026-11-12 · 对象 mission m15917（succeeded，PR #1867 已并入 main，merge commit e35ca08c）

## 裁定

**level = 3（系统集成）**，与报告自述一致（报告 L4 行自标「有对照，主判据 FAIL」，未宣称 L4 通过）。
L4 不判：基准裁判（唯一裁判）= 主判据 θ=60° got 4.180 vs ref 16.667 = **−74.9 %**，容差 <10 %（tensor_case_pool
case_code=MCMP-CAPILLARY-RISE tol_policy；mission goal 同值）→ **不在容差内**，且 step≈440 发散 NaN（F3 未修）。
「完整算例+误差分析+可复现」三件俱在，但基准未过 = 能力未验证，级别封顶 L3。

## ① 取证（逐项核验收三件套）

| 项 | 指针 | 核验结果 |
|---|---|---|
| run_artifacts | `artifacts/m15917/{jurin_neutral_theta90,jurin_theta60_wetting,ab_wall_fix_theta90}.json` + `docs/reports/m15917/figs/fig{1,2,3}_*.png` | ✓ 存在非空；**注意**：PR #1867 合并时 3 个 JSON 被门禁滤除未进 main，本审计在 `audit/m15917/` 恢复字节级副本（sha256 与原工件一致：017d26d2…/5663e8a8…/cd2559d0…） |
| benchmark ref vs got | 工件 JSON + 报告 §4 | ✓ 数值齐全（下表）；**gap**：`tensor_benchmark_cases` 对 anchor_slug='CASE-MCMP-CAPILLARY-RISE' **0 行**（25 行中无此锚点），容差只能溯源到 tensor_case_pool（status=dispatched, mission 15469）——基准裁判表缺登记，建议算例池侧补行 |
| repro_cmd 可执行 | 报告 §7 | ✓ 审计亲跑两条：θ=90 → h_got=0.865、spread 3.19 %（与工件 0.8647/3.19 % 一致）；θ=60 → ref=16.667、pass=False、~step 440 NaN（与报告一致，wall 17.6 s）。小缺陷：θ=60 失稳时 `jurin_cac.py:247` figure() 对 got=None 抛 TypeError（JSON 已先落盘，不影响工件） |

### 基准裁判（got vs 容差，逐项）

| 量 | ref | got | 误差 | 容差 | 判定 |
|---|---|---|---|---|---|
| h（θ=60° 主判据，step 375 末有限样本） | 16.667 | 4.180 | −74.9 % | <10 % | ✗ FAIL（未收敛即失稳） |
| h（θ=90° 负对照，3000 步） | 0（cos90°=0） | 0.865 | 5.19 % of 16.667 尺度 | <10 % | ✓ PASS |
| A/B 壁面修复（θ=90°，1250 步） | 质量守恒 | old_both h=−19.399/漂移 −50.26 % → fixed h=0.549/漂移 +1.06 % | 修复有效 | — | ✓（A/B 对照数字真实） |
| σ 标定 | 0.03（输入） | 0.02918 | −2.73 % | — | ✓ |

### 报告六段（tensor_reports id=15064）

①问题+ref 出处（§1，Jurin 1718 + de Gennes ch.2，σ 自测零手抄）✓ ②方法（§2）✓ ③参数表含 Bo=1.33（§3）✓
④对照表打勾叉（§4）✓ ⑤收敛/敏感性三点（§5）✓ ⑥失败模式 F1–F4+工件（§6）✓；另 §7 repro_cmd、§8 显式回答「是否<3%」。
图 3 张（A/B 曲线/中性对照/θ60 失稳）✓。

## ② 判级依据（L0–L5）

- L1 ✓：`tests/test_case_pool_capillary_rise_cac.py` 6 tests，审计亲跑 **6 passed in 12.81 s**。
- L2 ✓：impl 非占位（`cac_lbm2d.py` _shift 逐轴策略 + refresh_solid_ghosts 法向 ghost 修复，diff 实核）；entry CLI 可跑；run_artifacts 真实。
- L3 ✓：主求解器侧真调用 `benchmarks/in32_l01_wetting_ladder/run_case.py:28`（`from tensorlbm.wetting import ladder as L` → `L.jurin`（ladder.py:188）→ `CAC2D`）；修复落在主求解器 `src/tensorlbm/cac_lbm2d.py`。
- L4 ✗：主判据 −74.9 % vs 10 % 门，基准未过（F3：cosθ≠0 壁面通量不守恒，∫φ −14.09 %、φ_min −0.0986、~440 步 NaN；接棒点已写明）。
- L5 —：文献锚在（Jurin 1718），但 L4 未达，不评。

## ③ 落表

`INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES
('TensorLBM','m15917 CASE-MCMP-CAPILLARY-RISE',15936,3,'<evidence.json>','board-audit')`
（evidence 键=impl_files/entry/run_artifacts/tests/integration_callers/benchmark/repro_cmd/external_anchor，
值=真实指针；gate 机器复核模拟：declared L3 → gate_level=3, why=[] → confirmed 消账。）

## 审计留痕（复现）

```
cd /root/tp-agent/work/m15917   # 或 main（PR #1867 已并）
PYTHONPATH=src python3 benchmarks/case_pool/capillary_rise/jurin_cac.py --theta 90 --steps 3000 --sample-every 250 --no-calibration --threads 4 --out /tmp/audit_jurin90.json   # h_got=0.865 spread=3.19%
PYTHONPATH=src python3 benchmarks/case_pool/capillary_rise/jurin_cac.py --theta 60 --steps 1000 --sample-every 125 --no-calibration --out /tmp/audit_jurin60.json   # ref=16.667 pass=False NaN~440
python3 -m pytest tests/test_case_pool_capillary_rise_cac.py -q   # 6 passed
```
