# W9-C append-only diagnostics (post-formal, non-judgment)

Formal verdicts (out/result.json, out/README.md) computed ONLY from out/ =
frozen prereg v1.1 ladders. Files here are gap-96/gap-72 extension runs in
out_diag/, executed 2026-09-29 after the formal scan for SOURCE ATTRIBUTION
of the two failure modes. They change nothing in the frozen verdicts.

## Why fs=20: fp32 injection floor

a_body ∝ W^-3 at fixed tier; at gap 96 the W4-A guard (margin = 3·a_body/2^-24
>= 14) rejected fs=10 (margins 8.67/10.20/11.35). All diagnostics therefore
run at force_scale=20 → Re_cell ≈ 1.0 (formal convention 0.5). Inertia shift
Re 0.5→1.0 measured in situ by S1: −0.256% (sq0.3 W64), −0.073% (hex0.75
W265); corrections of this size apply to the numbers below and are ~10×
smaller than the signals being discriminated.

## D1 — hex near-packing: Gebart reference bias vs staircase artifact

Formal ladders (vs Gebart, signed %):
  hex 0.70: +1.66 (W100) / +0.47 (W198) / +0.13 (W396) / −2.10 (W594, gap 72)
  hex 0.75: +2.38 (W132) / +0.34 (W265) / −0.85 (W530) / −2.03 (W795, gap 72)

Extension (gap 96, fs20):
  hex 0.75 W1058: signed −3.6963 % (steady 94k steps, S2 0.0004%)
  hex 0.70  W790: signed −1.6284 % (steady 95k steps, S2 0.0001%)

Discrimination: a staircase artifact decays toward 0 as gap grows; a
reference bias is revealed as the staircase part shrinks. Observed: err
remains O(2-4%) NEGATIVE and (for 0.75) keeps descending at gap 96 →
resolved K is genuinely BELOW Gebart. Staircase decomposition
(signed = (1+st)/(1+b) − 1) gives monotone-decreasing staircase st(W):
  hex 0.75: st ≈ 4.9 / 2.9 / 1.6 / 0.5 % at gap 12/24/48/72, b ≈ +2.4 %
  hex 0.70: st ≈ 6.3 / 4.4 / 4.0 / 1.8 %, b ≈ +3.9 % (W790: st ≈ 2.3 %)

Conclusion: hex 0.70/0.75 main-gate failure = REFERENCE-SIDE. Gebart hex
bias at s = sqrt(vf_max/vf)−1 = 0.138/0.099 is O(2-4%); the square bias
curve at the same s (vf_equiv 0.61/0.65) gives +6.2/+4.4% — same order.
Premature-steady ruled out analytically (steady tol 1e-5/2000 steps bounds
hidden drift ≪ 0.1%; S2 20/20 pass); inertia ruled out by magnitude (S1:
0.073% per ΔRe 0.5 at hex0.75). These are the first resolved LBM
measurements of hex 0.70/0.75 transverse K; solver validated by 4
independent channels (square exact tiers, hex dilute tiers, gebart
consistency channel bit-closure, S1/S2).

## D2 — sq 0.40 monotonicity break: quantization-floor oscillation

Formal: +17.25 / +6.92 / +2.42 / +2.63 % (W42/84/168/252) — mono broken by
+0.21 pp between rungs 3 and 4, both far below gate. Extension W335
(gap 96, fs20): signed −0.6475 %. The error returns deep under the gate →
the rung-3→4 uptick is geometry-quantization oscillation at the resolution
floor (δR/R ≈ 0.5/R amplified by |dlnK/dlnR| ≈ 6.7 at φ=0.4), not a
divergent trend. Frozen criterion applied as written: tier FAIL
(monotonicity), finest 2.63% ≤ 3%.

## D3 — sq 0.70 disclosure completion (gap 72, fs20)

Formal (3 rungs): +17.12 / +6.47 / +3.85 % (gap 12/24/48). Extension
W1293 (gap 72): signed −1.4686 % (magnitude 1.47% ≤ 3%; gebart channel
−4.318% = (1−0.0147)·(1/1.02978)−1 bit-consistent). Square 0.70 crosses
under the gate at gap 72; kept non-gating disclosure per prereg (tier was
frozen at 3 rungs).

## Summary attribution

- sq array: solver + exact reference agree at 0.22/2.63/0.24/0.88 %
  finest (0.30/0.40/0.50/0.60) and 1.47% at 0.70 gap-72. Only failure is
  the strict-monotonicity clause at 0.40 (quantization oscillation).
- hex array: solver converges to Gebart within 0.62-2.16% at dilute
  (0.30/0.45/0.60, monotone); near-packing (0.70/0.75) resolves BELOW
  Gebart by 2-4% — Gebart hex reference bias at s=0.10-0.14, anticipated
  qualitatively in prereg section 2 ("近密堆渐近式"), quantitatively larger
  than the square-0.75 anchor (+0.27%) suggested.
- S1 (fs linearity) fails at sq0.3 W64 (0.256% > 0.1% tol): real O(Re)
  inertia at the inherited Re_cell=0.5 (a_body = Re_target·ν²/(k·R)·fs with
  fs=10 → effective Re = 0.5). Physical, correctly measured; hex0.75 pair
  passes (0.073%). Recorded per frozen protocol (FAIL, no re-run).
