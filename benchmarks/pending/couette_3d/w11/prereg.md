# prereg.md — couette_3d (W11-D case 1) — FROZEN 2026-10-06 (Phase 0 停批稿)

Append-only. md5 receipt at freeze recorded in `prereg.md.md5` (this file,
at the end of Phase-0). Gate runs may not start before owner approval.
Any post-freeze change = appended section + re-issued md5, never edits.

## 0. Case

3D Couette flow: bottom wall stationary, top wall moving at U0 (+x),
x/z periodic, steady linear profile. Purpose: verified/ promotion of the
3D moving-wall half-way bounce-back path (D3Q19 BGK), 3D twin of
verified/couette_2d.

## 1. Old-archive machine diagnosis (deliverable ①)

- worktree `benchmarks/pending/couette_3d/`: only README.md (md5 recorded
  in diag_arms.json provenance) + run.py skeleton. NO result JSONs.
- README claims "✅ 已入库 (2026-08-20)" with a H∈{40,80} results table in
  which every number is a literal `...` placeholder, claims "误差随 H 单调
  下降 ✅" without any data, cites a run path under
  `/home/wxsc/cxs/TensorLBM/benchmarks/verified/couette_3d` which does not
  exist, and states U0=0.1 while its own run.py default is U0=0.05.
  Machine verdict: status claim unsubstantiated; treat as skeleton-only.
- Skeleton run.py PILOT (verbatim, CPU, diag/pilot/case_H{16,32,64}.json):
  l2 = 6.776e-05 → 3.295e-04 → 3.364e-03 (ANTI-convergent, ~x10 per H
  doubling), spanwise std ~1e-3 (field not z-uniform), steps-to-steady
  3600→7400→17200, mass drift 0.0035→0.0127→0.0301%.
- ROOT CAUSE (diag_arms.json arm A vs B, bitwise reproduction): run.py
  lines `u0[0,:,:]=0.0; u0[-1,:,:]=0.0` zero the z=0 / z=nz-1 FLUID planes
  (2D wall-row indexing lifted from couette_2d; in the (nz,ny,nx) layout
  index 0 is the z axis). Arm A (bug present) reproduces the pilot error
  at H16 (6.66e-05 vs pilot 6.78e-05); arm B (bug removed) gives
  exactly z/x-uniform fields (spanwise std 0.0) and l2 at the fp32 floor.
- Second finding (diag_dtype.json): with the init fixed, fp32 errors are a
  TIME-LINEAR rounding accumulation (H64: l2 1.43e-05@4k → 1.16e-04@24k,
  mass +0.0049→+0.0245%, both linear in t), localized at the moving wall
  (dev_top linear; mid/bottom constant). In fp64 the SAME kernels show
  ZERO drift (mass −0.00000% at 60k steps) and l2 at 1e-13..2e-12 floor:
  the discrete half-way-BB BGK attractor IS the analytic profile — the
  scheme error is zero and the fp32 residual is pure accumulation.
  Consequence: the standard "strictly monotone decreasing error" clause is
  not resolvable/meaningful here (it would measure run length, not
  resolution); replaced by frozen clauses G2a/G2b below, flagged for
  owner sign-off as a design deviation.

## 2. Reference lock (deliverable ②)

Analytic, derived here (no external number to hand-copy):
- Half-way bounce-back convention (locked to verified/couette_2d and the
  skeleton docstring): effective walls at y_wall = 0.5 and y_wall =
  ny − 1.5 lattice units; gap H_eff = ny − 2 = H.
- Steady solution of the shear problem for ANY Re, Ma in this setup:
  u_ana(y) = U0 · (y − 0.5) / H_eff for fluid rows y = 1..ny−2;
  wall shear τ_ana = ν·U0/H_eff; u_ana vector is machine-generated inside
  the driver (zero hand-copy), stored in every case JSON (u_analytic).
- Machine-verified wall-direction convention (source-read + skeleton
  docstring, to be re-asserted in the gate logs): top-wall into-fluid
  directions are cy<0 = {4,8,9,16,18} of the D3Q19 C table; the injection
  `f_new = f_pre[opp] + 2 w ρ (c·u_wall)/cs²`, cs²=1/3, is applied on all
  19 directions at wall cells (only into-fluid directions stream).

## 3. Frozen design (deliverable ③)

Driver: `run_v2.py` in this directory (staging; worktree untouched).
Lineage: skeleton run_case VERBATIM except exactly three disclosed
changes (recorded in the "changes_vs_skeleton" field of every case JSON):
  1. init = clamped analytic ramp on ALL cells (skeleton z-plane zeroing
     bug removed; evidence §1);
  2. domain cube nx = nz = H (skeleton 8; equivalence measured at H16:
     l2 1.041e-05 (nx=8) vs 1.046e-05 (nx=16), diag_arms.json arms B/B16);
  3. dtype parameter + equal run length for all tiers.
Everything else verbatim: estimator, steady rule, moving-wall function
(md5 of skeleton run.py recorded in diag_arms.json / diag_dtype.json).

- Lattice D3Q19, collision BGK τ=0.8 (ν=0.1), U0=0.05, Ma=0.0866,
  Re = U0·H/ν per tier (8/16/32/64 across the fp64 ladder; 16/32/64 fp32).
- Primary ladder (fp32, GPU5): H ∈ {32, 64, 128}.
- Supplementary ladder (fp64, CPU, OMP≤16, background OK): H ∈ {16, 32,
  64, 128}; purpose = scheme-exactness evidence (G2b), NOT a physics tier.
- Steps: min_steps = 20000, max_steps = 24000 for EVERY tier (equal run
  length; steady rule exits at min_steps — the ramp init starts at the
  attractor, transient amplitude ~1e-8, decayed long before 20000 in both
  dtypes; diag_dtype.json).
- Estimator (VERBATIM skeleton): measurement line (z0,x0) = (nz//2, nx//2);
  steady when the drift of u(z0, ny−2, x0) over the last 10 samples
  (1 sample / 200 steps) is < 1e-5 after min_steps; profile = time-mean of
  ux(z0,:,x0) over the final 200 steps; l2_rel over the H fluid rows vs
  u_ana; max_rel over rows with u_ana > 0.2·U0; secondary diagnostics
  u_top_row_err, tau_err (discrete slope), uz_max, ux_spanwise_std,
  total_x_momentum, mass_drift, finite.

### Gates (frozen; verbatim strings for the verifier)

- G1 (accuracy, per fp32 tier): `l2_rel_err ≤ 3e-2` AND
  `max_rel_err_masked_pct ≤ 3.0`.
- G2a (fp32 floor-insensitivity, per fp32 tier): `l2_rel_err ≤ 1e-3`.
  Rationale + measured margin: errors are time-linear fp32 accumulation
  (§1); measured ≤ 1.2e-4 at 24k steps (diag_dtype.json arm E), 8× margin.
- G2b (scheme exactness, per fp64 tier): `l2_rel_err ≤ 1e-7`. Measured
  margins: 7.5e-14 (H16) / 3.4e-13 (H32) / 8.6e-13 (H64) / 1.8e-9 (H128 at
  exactly 20000 steps; full decay 2e-12 at 60k), ≥ 55× margin.
- G2-disclosure (mandatory rows in the result): per-tier fp32
  (l2_rel_err, mass_drift_pct, n_steps); statement that fp32 residuals
  are fp32 rounding accumulation, not resolution error, evidenced by the
  fp64 contrast; **design deviation for owner sign-off**: the standard
  "strictly monotone decreasing over ≥2 grid levels" clause is replaced
  by G2a+G2b because the scheme error is provably at machine floor (a
  monotone-decreasing requirement on rounding noise would be unfalsifiable
  noise-ordering).
- G3 (health, per tier both ladders): `steady == True` (with
  `n_steps ≤ 24000`), `mass_drift_pct ≤ 0.05`, `uz_max ≤ 1e-7`,
  `ux_spanwise_std ≤ 1e-6`, `finite == True`.
- Stop-loss: any gate clause FAIL → complete the whole ladder anyway
  (runs are minutes), write all case JSONs, report FAIL with the frozen
  clause text; NO parameter re-runs (nothing may be tuned post-hoc).
  fp64 CPU ladder wall-clock > 90 min for one tier → keep it running in
  background and report from the completed tiers; do not shorten steps.

### Frozen run commands (gate phase, after owner approval)

```
# GPU snapshot first (GPU5 only, shared with C-track; 0-4/7 勿碰)
nvidia-smi --query-gpu=index,memory.used --format=csv
cd /nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/nfs/wangxi/worktrees/bm_w11/src \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 \
  /nfs/wangxi/venvs/tensorlbm/bin/python run_v2.py --dtype fp32 \
  --device cuda:0 --H 32 64 128 --min-steps 20000 --max-steps 24000 \
  --out-dir gate/fp32   |& tee gate/fp32.run.log
OMP_NUM_THREADS=16 PYTHONPATH=/nfs/wangxi/worktrees/bm_w11/src \
  PYTHONDONTWRITEBYTECODE=1 \
  /nfs/wangxi/venvs/tensorlbm/bin/python run_v2.py --dtype fp64 \
  --device cpu --H 16 32 64 128 --min-steps 20000 --max-steps 24000 \
  --out-dir gate/fp64   |& tee gate/fp64.run.log
```

## 4. Budget (deliverable ④)

| item | where | est |
|---|---|---|
| fp32 ladder 32/64/128 cube, 20.2k steps ea | GPU5 | ≤ 0.15 h |
| fp64 ladder 16/32/64/128 cube, 20.2k steps ea | CPU only | 0 GPU·h |
| Phase-0 diagnostics already spent | CPU only | 0 GPU·h |
Couette total GPU: **≤ 0.15 h**. Combined with sphere27 (≤ 4.3 h): total
≤ 4.5 GPU·h vs 6 GPU·h budget.

## 5. Risk register (deliverable ⑥)

| risk | preregistered handling |
|---|---|
| fp32 tier hits > 1e-3 (G2a) via faster-than-measured accumulation | G1 still the accuracy gate; if G2a FAILs while G1+G2b pass, verdict = "G2a FAIL (disclosed accumulation law), promotion withheld, owner decides" — no re-tuning |
| fp64 H128 not fully decayed at 20000 steps | frozen G2b threshold 1e-7 chosen with 1.78e-9 measured at exactly 20000 steps (56× margin); no threshold change post-hoc |
| GPU5 contention with C-track (OOM/thermal) | snapshot before launch; if free memory < 2 GB, wait and retry; cube-H128 needs < 1 GB |
| owner rejects the G2 clause replacement | fallback (owner's prerogative, NOT pre-authorized): strict fp32 monotone clause would order rounding noise; case would be reported FAIL-by-clause — recorded here so the consequence is visible before approval |
| driver-side undetected divergence from skeleton semantics | only 3 disclosed diffs, each machine-evidenced (§1, §3); skeleton md5 recorded; estimator code paths verbatim |

## 6. Verdict semantics

verified/ promotion candidate iff G1 ∧ G2a ∧ G2b ∧ G3 hold on all tiers
AND the G2-disclosure rows are present in the result. Numbers flow
machine-written case JSONs → verdict JSON; no hand transcription.

## Appendix A (appended 2026-10-07 after gate runs; append-only, md5 re-issued)

Gate runs executed per frozen commands (GPU5 off-peak vs C-track; orchestrator
gate_run.sh). Results machine-written to couette3d/gate/ (case_H{32,64,128}_fp32.json,
case_H{16,32,64,128}_fp64.json, fp32.run.log, fp64.run.log) and judged by
verify_v2.py into gate/verdict_v2.json. Verdict: FAIL, sole failing clause
G3 steady==True at H128 fp32 (steady=False, n_steps=24000 cap); G1/G2a/G2b all
pass; fp64 ladder all tiers steady=True. No re-runs, no clause changes.

## Appendix B (receipt completion, appended 2026-10-07; append-only, no other edits)

冻结版全文 md5 = 6f19466d1d1d3ceb8f204424bd615fd1（2026-10-06 Phase-0 冻结，控制器 22/22 亲验）。
gate 附录 A 追加于其后（append-only，冻结前缀一字未动）；附录 A 后全文 md5 =
b5a6f230ec75d8cdbc72978149758402（2026-10-07 首次重发边车时的值）。本附录 B 再追加后
的现行全文 md5 与 sha256 见同目录 prereg.md.md5 边车（frozen/current/sha256 三行）。
