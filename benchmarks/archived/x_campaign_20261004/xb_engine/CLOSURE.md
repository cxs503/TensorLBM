# XB 轨 CLOSURE — benchmarks/archived/x_campaign_20261004/xb_engine（终态）

判决数字由汇编机 `/nfs/wangxi/tmp/xarch_closure_gen.py` json.load 本目录归档
JSON 机械提取，行内 `path::key` 为唯一数字来源，无手抄。

## XB-1 iso4 正式电池 — A1/A3/A4 honest FAIL（控制器独立复核 16/16+5/5+9/9，NOTES.md 记录）
- A1（G1 门 FAIL）：clause_a_W4_or_Wstar=false、
  clause_b_monotone_nonincreasing=true →
  G1_pass=false；1−σ̄_eff(W) 阶梯 = 0.047093610211133785 / 0.02749881238841423 / 0.01693347655890476 / 0.009541851713363747 / 0.0034875411738527484（W=3/4/5/6/8，
  W*=8）`out/xb1_formal_summary.json::A1_gate.*`
- A3（NaN 门 FAIL）：W2 三臂 nan_max=39845888（全 NaN，M1 P1 基线复现）→
  A3_pass=false `out/xb1_formal_summary.json::A3_gate`
- A4（G2 门 FAIL）：err_signed(ch2, a40)=-0.7821703214720683（门 |err|≤3%）；
  bonus 臂 a20=-0.27932639787249325；ch1 机器复核 a40 err_signed=-0.7222637705568704
  `out/xb1_formal_summary.json::A4_gate.*`
- A5 位一致（非门披露）：window_mean_bit_equal=true、
  max_rel_diff_sigma_rec=0.0、五序列 25/25 逐位
  `out/xb1_formal_summary.json::A5_bitrepro`
- A2 归因（非门）：sigma_eff_ratio g4only=0.9463444251913189 /
  l4only=0.9480928245456904（context iso4_full=0.9718014336608061,
  iso2_default=0.9235675731544108）
  `out/xb1_formal_summary.json::A2_attribution`

## XB-2 多量程电池 v3 — 总判 FAIL（显式判决块 = B5_verdict.pass_）
- B1 位对档案复刻：base_source="archive"，cs@0.9 与 pr@0.55
  all_equal 均 true（逐字段+hist 前缀逐位）`out/xb2_v3_formal.json::B1_bitcheck`
- B2 扫描 42 臂：drift gate_pass 35/42；不稳定 4 臂
  （b2_G2m0p08_pr_Tr0p55=nonfinite/negative rho at step 1000；b2_G2m0p1_pr_Tr0p55=nonfinite/negative rho at step 1000；b2_G2m0p15_pr_Tr0p55=nonfinite/negative rho at step 1000；b2_G2m0p2_pr_Tr0p55=nonfinite/negative rho at step 1000）
  `out/xb2_v3_formal.json::B2_runs[*].drift.gate_pass, B2_runs[*].stable/error`
- B3 选择：best_g2=-0.02（7/7 tier 过 drift 门）
  `out/xb2_v3_formal.json::B3_selection`
- B4 通道门 FAIL：resid_all_L=false、
  mono_not_degraded=false →
  pass_gate=false；基线 σ(L)=L128=0.32022612409321427 / L192=0.32649180156335816 / L256=0.32942224369200646
  `out/xb2_v3_formal.json::B4_ii_laplace.gate, B4_ii_laplace.baseline_G2_0.per_L.*.sigma`
- B5 总判（显式块）：pass_=false；三 tier 全 pass_le_5x=false：
  cs@0.6 u_max_base=0.25662991404533386 → u_max_g2=0.26434972882270813（reduction=0.9707969635083238）；pr@0.6 u_max_base=0.17427799105644226 → u_max_g2=0.18031419813632965（reduction=0.9665239501809857）；pr@0.55 u_max_base=0.2041354477405548 → u_max_g2=0.21032962203025818（reduction=0.9705501572726059）
  `out/xb2_v3_formal.json::B5_verdict`
- 判决数字所用字段声明：总判=out/xb2_v3_formal.json::B5_verdict.pass_（JSON 内显式
  "verdict" 块）；per-tier=B5_verdict.tiers.*.pass_le_5x；per-arm=B2_runs[*].drift.gate_pass；
  B4 通道=B4_ii_laplace.gate.pass_gate。
- 崩溃时间线证据：out/xb2_partial.json（v1 B2 臂 1/42 崩溃前 B1 双臂位同落盘）+
  out/xb2_v3_partial.json（v3 运行中档）+ out/xb2_formal.log / out/xb2_v3_formal.log 原样保留。
