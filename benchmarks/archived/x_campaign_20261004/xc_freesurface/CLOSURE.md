# XC 轨 CLOSURE — benchmarks/archived/x_campaign_20261004/xc_freesurface（终态）

终裁（控制器 2026-10-05）：三路径裁定 (iii) **维持现状归档 = 终态**。
判决数字由汇编机 `/nfs/wangxi/tmp/xarch_closure_gen.py` json.load 本目录归档
JSON 机械提取，行内 `path::key` 为唯一数字来源，无手抄。

## W1 壁门 — PASS（读法 A；控制器独立复核 18/18）
- success_tau055_reading_A=true；
  success_tau_lt1_reading_B=false（双旗照发，归档按读法 A）
  `out/w1_battery/verdict_w1_battery.json::success_tau055_reading_A, success_tau_lt1_reading_B`
- 主臂 W1-AB-055 三判据：T-D1 膜坡度=-0.0031210478771454384（门 0.001，过）；
  T-D2 max|uy|=0.024401161819696426（审计 runmax=0.02603684552013874，门 0.1，过）；
  T-D3 pass=true；位一致护栏 250 步 pass=true
  `out/w1_battery/verdict_w1_battery.json::runs.W1-AB-055.td1..td3, bit_identity_guard`
- 消融归因：A-only 杀天花板、B-only 杀膜/僵尸、AB 缺一不可（B-055 / off-055 T-D2 FAIL，
  全臂档在 out/w1_battery/）

## 溃坝（M3 冻结口径）— Z FAIL 结构性定案
- 正式臂 capture_mirror（τ=1.0）：a20 max|Z|=94.629%、
  a40 max|Z|=114.77%（v1.1 档 a20=55.386/a40=39.340，双双恶化）
  `out/m3_w1/result_a20_w1_ab.json::judgment.max_abs_Z_err_pct, out/m3_w1/result_a40_w1_ab.json::judgment.max_abs_Z_err_pct`
- mirror-only 探针（τ=1.0，前锋滞后与壁闭合无关）：a20 max|Z|=56.169%、
  a40 max|Z|=40.906%
  `out/m3_w1/result_a20_probe_mirror.json::judgment.max_abs_Z_err_pct, out/m3_w1/result_a40_probe_mirror.json::judgment.max_abs_Z_err_pct`
- τ 阶梯（控制器诊断，mirror-only 承载，max|Z| 全精度）：
  - a40（τ=1.0 锚 40.906 →）：τ=0.8: 33.47；τ=0.7: 27.991；τ=0.6: 18.207；τ=0.55: 63.996（nonfinite_at=642）；τ=0.5: 78.867（nonfinite_at=277）
  - a20（τ=1.0 锚 56.169 →）：τ=0.8: 49.124；τ=0.7: 43.645；τ=0.6: 34.253；τ=0.55: 27.208；τ=0.5: 76.519（nonfinite_at=274）
  - 改善单调到 τ=0.6 后反转：τ=0.55（W1 壁锚）a40 失稳 nonfinite_at=642，
    τ=0.5 双格失稳（nonfinite_at=274/277）；
    a40 稳定行最优 18.207% = 门 3.0% 的
    6.07 倍 → τ 通道到不了 3%
  `ctrl/diag_tau*.json::judgment.max_abs_Z_err_pct, nonfinite_at`
- 阶梯判决器：`out/m3_w1/judge_ladder.txt`（LADDER VERDICT: FAIL；narrowed=False 双格）
- 归因链（机器全验）：Z 滞后=黏性+前锋截断（M8 归因独立复证）；冻结口径被出生带毒化；
  出生式救援与前锋位置可观测结构性不相容；H 改善全部来自 mirror、与出生机制正交
  （out/m3_w1/probe_front_attribution_a20 及 _rescue 两件 JSON）

## 终态处置
- M6 复活腿：PARKED（解锁条件=溃坝门，未达）。
- W1 壁包（halfway+mirror）= 已验证库资产（H 改善正交、静水柱三门绿）；
  引擎 md5 链在 out/w1_battery/*.json::provenance.engine_w1_md5=8782d48dc7967caee07c8521f5aa22d8。
