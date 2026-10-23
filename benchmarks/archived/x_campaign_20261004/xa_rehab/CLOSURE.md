# XA 轨 CLOSURE — benchmarks/archived/x_campaign_20261004/xa_rehab（终态）

判决数字由汇编机 `/nfs/wangxi/tmp/xarch_closure_gen.py`（2026-10-06）json.load
本目录归档 JSON 机械提取，行内 `path::key` 为唯一数字来源，无手抄。

## XA-1 C1-G2 晋级包 — crosscheck 全过，归档件 = archive_c1g2/
- crosscheck 72/72 全对（all_ok=true）
  `archive_c1g2/crosscheck_report.json::all_ok,n_checks`
- 4/4 臂过门（stable ∧ relL2≤0.03 ∧ 逐色质量漂移≤1e-10/步）；最差臂
  g10_ny64 relL2=0.010918219215161354
  `archive_c1g2/result.json::arms[*].relL2, gate.relL2_max_per_arm`
- vendored 引擎 color_gradient2d.py md5=81203d6d6f00104f4853a6db81815fe0、
  求值器 cg_reference.py md5=e78c6d1eb22d628182636c0df74fb26a，与冻结值逐位同
  `archive_c1g2/result.json::module_md5`

## XA-2 M2-G2 噪声复跑 — R-C 维持 honest FAIL
- R-A 确定性 9/9 PASS（前缀尾统计 vs formal rel_diff=0.0）
  `xa2_m2g2_noise/out/xa2_judgment.json::verdict.R_A_determinism, r_a_receipt`
- R-B 单调条款 FAIL：err_ext(N)= 0.004593045455743975 /
  0.0055056268328453695 / 0.004060566990017528
  （N=96/144/192），monotone_non_increasing=false；
  bump 原 +0.0683120433805271pp → ext +0.09125813771013944pp（窗噪声解释不动，
  结构性非单调）`xa2_m2g2_noise/out/xa2_judgment.json::verdict.R_B, verdict.original_bump_pp, verdict.ext_bump_pp`
- 总判（R-C）：R-C: 144 bump not explained by window noise -> structural non-monotonicity; M2 honest FAIL maintained `xa2_m2g2_noise/out/xa2_judgment.json::verdict.rule`

## XA-3 σ-失配假设检验 — KILLED
- 判决串：KILLED: sigma-mismatch hypothesis rejected; T2 G1a FAIL stands; residual common-mode attribution -> remaining candidates (mobility term / lambda_eff != 1) `xa3_t2_sigma/out/xa3_judgment.json::rules.verdict`
- 读法 in_band：A=6/11、
  B=6/11、
  C=6/11；
  S1=false、
  S2=false、
  S3=true
  `xa3_t2_sigma/out/xa3_judgment.json::readings.*.in_band_count, rules.S1..S3.pass`
- P 臂 σ_eff/σ_rec−1：stp_R16=-0.059265161043244974 / stp_R16_b8=-0.05925470323104154 / stp_R24=-0.06008842646763091 / stp_R32=-0.060513064363562075 / stp_R24_mu045=-0.06009160630051613
  （族平坦 ≈−6%）；W 复刻回执 4/4 bit_level_pass（rel_diff=0.0，
  含 GPU6 档案 ↔ GPU5 重跑跨设备逐位同）
  `xa3_t2_sigma/out/xa3_judgment.json::diagnostics.*.sigma_eff_over_rec_minus1, receipts.W_stp_*.bit_level_pass`
- 消费链交付：A-CAC 三点 F_sigma_pct=0.9207944922423941
  `x1_crossval/out/x1_crossval.json::case_a.CAC.F_sigma_pct`

## XA-4 m7 伺服 — 不立项（决策记录）
- owner 决策 2026-10-06：不立项。判据档 xa4_m7_servo/out/xa4_judgment.json 随档保存，
  本页按决策记录口径不引数字。

## XA-5 β 扫描 — 判据规则输出 report（无 F 行）
- 规则输出：no complete beta row (missing arms) -> report；F_primary/F_secondary 为空
  `xa5_cg_beta/out/xa5_judgment.json::rules.D, F_primary, F_secondary`
- 确定性三连 (128,25) β=0.5/0.7/0.9 对 C1 档案 rel_diff=0.0 逐位
  （0.010212267240284484 / 0.010619020900319402 / 0.010833453085584525）
  `xa5_cg_beta/out/xa5_judgment.json::receipts.determinism_128_25`

## X1 跨模型归因电池 — judge 三过 RC=0 定稿（归因仪器，非判决门）
- 定稿 md5=b1c434be2c8d394bbed52cdc19f43c0b（归档件与 staging 逐位同）
- battery_status="non-verdict attribution instrument" `x1_crossval/out/x1_crossval.json::battery_status`
- A-CAC（XA-3 P 消费链）：F_sigma_pct=0.9207944922423941、
  mode="consumed XA-3 variant-P outputs (dedup per prereg §2)"
  `x1_crossval/out/x1_crossval.json::case_a.CAC.*`
- X1-J1（judge L99 f 前缀笔误致 A-CAC 行缺失）已修复，Laplace|CAC 行恢复；
  X1-J2 纯披露维持：receipts.mcmp_analytic_recompute=H64:false,
  H128:false
  （冻结 1e-9 绝对容差字面读法；表值=fp64 重算，与档案差 fp32 ε 量级）
  `x1_crossval/out/x1_crossval.json::receipts.mcmp_analytic_recompute`
- B-CAC 确定性回执：rel_diff_Eu=0.0
  `x1_crossval/out/x1_crossval.json::receipts.b_cac_ny100_determinism`
