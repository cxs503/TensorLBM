# prereg.md — sphere_re100_d3q27 (W11-D case 2) — FROZEN 2026-10-06 (Phase 0 停批稿)

Append-only. md5 receipt at freeze in `prereg.md.md5`. Gate runs may not
start before owner approval. Any post-freeze change = appended section +
re-issued md5, never edits.

## 0. Case

D3Q27 stencil-robustness twin of verified/sphere_re100 (remote main,
commit 20cd23472, promoted 2026-10-05): same geometry, domain, τ, u_lb,
steps, wall treatment, force observable and reference — lattice swapped
D3Q19→D3Q27 and collision MRT-low-memory→BGK27 (controller-designated;
disclosed deviation, see §3).

## 1. Old/remote archive machine diagnosis (deliverable ①)

- Remote main pulled via GitHub contents API (token, not echoed) into
  `diag/remote_pull/` (manifest.json with blob shas + sha256; branches/main
  sha bea7ec2f, promotion commit 20cd23472). The local worktree
  (4ba836c9) predates the promotion by 21 benchmarks/ commits — all
  protocol truth is taken from the pull, not the worktree.
- verified/sphere_re100/result.json (machine-read):
  cd_ref = 1.0917310910948732 (Schiller-Naumann 24/Re(1+0.15Re^0.687)),
  cross-refs schiller_naumann_0681 = 1.0685190542670633,
  clift_gauvin = 1.1092345787735252, band [1.069, 1.109], gate 3%.
  Tiers (the "big-domain" files t1_d40big_12k.json / t2_d60big_24k.json,
  NOT the older formal_bfl_D{40,60}.json which used lat2/up1.25/down2.25):
  D40 280×280×320, τ=0.56, u_lb=0.05, re_eff=100.0, 12000 steps,
  bfl_links 28722, cd 1.1219804015117896, err 2.7707656824703957%,
  drift −0.14475665095754808%, cv_closure 1.496104414234331e-04,
  mass −0.0797 ppm, wall 1185.8 s;
  D60 420×420×480, τ=0.59, 24000 steps, 64794 links,
  cd 1.119462001144486, err 2.5400861325477218%,
  drift −0.0420075943755402%, cv 7.813769713472235e-05, mass 0.0 ppm,
  wall 7676.5 s; span 0.231%, monotone true.
- Old pending/sphere_re100_d3q27 (remote): README + run.py only, an
  INCOMPLETE DIFFERENT approach — cumulant_geier/mrt27 collision, Ladd MEM
  force (compute_obstacle_forces_27), plain library far_field_bc_27,
  ≥10D ad-hoc domain, "正式网格待跑" (no results). Its G7 finding
  (GeneralSimEngine D3Q27 is a dead enum; solver3d.stream3d is
  D3Q19-hardcoded) is confirmed by my source reads and shapes this
  design: engine for geometry only, D3Q27 library kernels directly.
  Superseded by this prereg; nothing reused except the G7 lesson.
- Library lattice/force audit (diag/lattice_audit.json, overall_pass
  true): Σw=1 fp64 both lattices; 2nd-order isotropy max dev 1.7e-16;
  D3Q27 4th-order isotropy dev 3.3e-16; OPPOSITE involutive+negation
  consistent; bfl_common _C19/_W19/_OPP19/_C27/_W27/_OPP27 are the
  d3q19/d3q27 module tables themselves (same objects; W is the
  W_EXACT64 f32 cast). The BFL force ledger branches on C/OPPOSITE only —
  NO weight table enters the force; weights enter equilibrium27 (W_EXACT64
  rational cast) and moving-wall corrections (unused: sphere walls are
  stationary). The known ~2× d3q27 docstring debt
  (moving_wall_linkwise_me_force_torque, Taylor-Couette torque) is a
  diagnostic-only primitive NOT in this force path.
- D3Q27 chain dress rehearsal (diag/mini27_D12.json, D=12 Re≈30 CPU):
  q19 standalone == common bitwise; the 19 shared direction vectors in
  the 27-field == the 19-field bitwise (2610 links); CV port == archived
  instrument exactly at 19; sparse-27 bounce == dense common-27 bounce
  torch.equal; both lattices finite, n_wrapped 0. (CV closure 1.50 at 60
  steps is the startup transient, recorded not judged.)

## 2. Reference lock (deliverable ②)

`reflock.json` (this directory) pins, by path + sha256:
- diag/remote_pull/verified__sphere_re100/result.json — cd_ref and
  cross-refs (the ONLY reference source; verifier reads this file, zero
  hand-copy);
- t1_d40big_12k.json / t2_d60big_24k.json — tier twins (domain/τ/steps/
  u_lb/re_eff), machine-checked at runtime by run27.py
  (`archive_twin_check` refuses to run on mismatch);
- remote_pull/manifest.json — provenance chain (main sha bea7ec2f,
  commit 20cd23472).
Judgment observable: window-mean Cd over the last 20% of samples of the
BFL per-link momentum ledger (laboratory frame), identical to archived
NOTES.md §7. Cd caliber dpS = 0.5·u_lb²·π·R_lb². No corrections applied
to any observable (correct_mass every 200 steps is protocol-inherited
from the archived case, not a post-hoc correction).

## 3. Frozen design (deliverable ③)

Driver `run27.py` (staging; worktree untouched). Loop mirrors the archived
w8a/run.py sparse route step-for-step (NoDynamics solid gather → collide →
restore → full-way BB → f_pre_stream clone → stream → far-field →
ledger+bounce sparse → sample/50 → correct_mass/200, divergence check/500).
Deviations, each machine-locked:

 D1 lattice D3Q27: stream27_roll, compute_q_sphere_common(lattice="D3Q27"),
    bfl_boundary_link_indices(lattice="D3Q27") with n_wrapped==0 assert.
 D2 collision collide_bgk27 (controller-designated; library has no
    low-memory MRT27). Cross-stencil/collision disclosure rows (G4) use the
    archived zero-cost decomposition t1_bgk vs t1_mrt_base (w7/ tree).
 D3 memory-mirrored implementations P1–P5 (feq27 per-direction into
    preallocated feq; in-place collide `tmp=f−feq; tmp/=τ; f−=tmp`;
    solid gather/scatter bounce-back; in-place slab far-field with the
    library write order inlet→outlet→y−→y+→z−→z+; in-place mass
    correction). All five torch.equal vs library originals on two field
    families (diag/ab27.json, `all_pass: true`); arithmetic association
    replicated operator-for-operator. Purpose: cap the per-step live-set
    at 3 full (27,N) copies (library forms reach 4–6 copies → OOM at
    D60big on 32 GB). Peak plan: stream = f + f_pre_stream + out
    ≈ 27.5 GiB; collide ≈ 2.2 copies; PYTORCH_CUDA_ALLOC_CONF=
    expandable_segments:True.
 D4 init rebuilt with feq27 at (ρ=1, ux=u_lb, ux=0 on solid) — the
    archived engine init state; runtime guard asserts the engine's own
    D3Q19 init ux (fluid mean) equals u_lb before discarding it
    (smoke: 0.05000000074505806).

Tiers (machine-checked twins): T1 D40 (280×280×320, τ=0.56, 12000 steps,
sample 50, cv_tail 2400), T2 D60 (420×420×480, τ=0.59, 24000 steps,
sample 50, cv_tail 4800); u_lb 0.05, re_eff 100 both. Seed 0. fp32.

### Gates (frozen; verbatim strings for the verifier)

- G1 (accuracy): err_pct(T1) ≤ 3.0 AND err_pct(T2) ≤ 3.0 where
  err_pct = |cd_window − cd_ref|/cd_ref·100, cd_window = mean of cd over
  samples with step > 0.8·steps, cd_ref from reflock result.json.
- G2 (monotone): err_pct(T2) < err_pct(T1) strictly (same two-tier
  ladder shape as the archived case).
- G3 (health, both tiers): steady window drift
  |mean(last20%) − mean(prev20%)|/mean(last20%) < 0.3%; CV momentum
  closure mean over tail |cv_mom_cd − cd|/|mean cd| ≤ 1e-2;
  mass_drift_ppm ≤ 100; finite = true; n_wrapped = 0; bfl_links recorded
  (expected larger than the 19-lattice 28722/64794; geometry-determined,
  no threshold).
- G4 (cross-stencil disclosure rows, non-gate, mandatory): Cd27 vs
  archived Cd19-MRT at D40/D60 (absolute + pct); Δcollision row
  t1_bgk − t1_mrt_base from w7/diag/ (both at the archive small domain
  200×200×180, D40, τ=0.56 — domain disclosed in the row; isolates how
  much of Cd27−Cd19 is collision choice vs stencil; machine values
  bgk 1.215140640211194 vs mrt 1.214927032369151, Δ = +0.0176%);
  span27 = |cd(T2)−cd(T1)|/cd(T1)·100 vs archived 0.231%.
- Stop-loss / OOM ladder (frozen order): (a) OOM → one retry after
  torch.cuda.empty_cache(); (b) still OOM → torch.compile default mode on
  BOTH tiers (consistency rule: identical compile state); (c) still OOM →
  T2 abandoned, report T1-only + cross-stencil row, verdict "T2
  memory-blocked", owner decides; divergence/NaN → record diverged_at,
  verdict FAIL, no parameter change.

### Frozen pre-gate checks (run at gate time, before T1)

P1–P5 = rerun `diag/ab27.py` (must exit 0). P6 = archive twin check runs
inside run27.py per tier. GPU5 snapshot mandatory before launch (shared
with C-track; 0-4/7 勿碰; need ≥ 29 GB free).

### Frozen run commands (gate phase, after owner approval)

```
nvidia-smi --query-gpu=index,memory.used --format=csv
cd /nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/sphere27
OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/nfs/wangxi/worktrees/bm_w11/src \
  /nfs/wangxi/venvs/tensorlbm/bin/python diag/ab27.py
CUDA_VISIBLE_DEVICES=5 W27_DEV=cuda:0 PYTHONDONTWRITEBYTECODE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 \
PYTHONPATH=/nfs/wangxi/worktrees/bm_w11/src \
W27_ARCHIVE_DIR=$PWD/diag/remote_pull/verified__sphere_re100 \
  /nfs/wangxi/venvs/tensorlbm/bin/python run27.py gate/t1_D40_27.json \
  --D 40 --cv-tail 2400 --mem-stats  |& tee gate/t1.run.log
# (T2 same with --D 60 --cv-tail 4800, gate/t2_D60_27.json)
/nfs/wangxi/venvs/tensorlbm/bin/python verify27.py   # reads only JSONs
```

## 4. Budget (deliverable ④)

| item | est (scaling: archived D19 wall × 27/19 bytes × 1.1 mirrored-op overhead) |
|---|---|
| T1 D40-27 (1185.8 s × 1.55) | ≈ 0.51 h |
| T2 D60-27 (7676.5 s × 1.55) | ≈ 3.30 h |
| pre-gate checks + smoke (already done, CPU/GPU trivial) | ≈ 0.05 h |
Sphere27 total GPU: **≤ 3.9 h**. Couette ≤ 0.15 h. Combined ≤ 4.1 h
(≤ 6 GPU·h budget, margin ≈ 1.9 h). CPU diagnostics do not draw GPU.

## 5. Risk register (deliverable ⑥)

| risk | preregistered handling |
|---|---|
| BGK27 monotonicity breaks G2 (precedent: W5-C bgk27 fixed-Re τ-ladder non-monotone) | G2 FAIL reported as-is with G4 decomposition rows (Δcollision vs Δstencil separated); no collision swap post-hoc. Mitigation already in design: identical τ per tier as archived (τ enters via ν at fixed Re — the archived τ pair 0.56/0.59 was monotone under MRT19) |
| D3Q27 force/weight convention error | closed by §1 audit + mini bitwise chain + P1–P5 + zero-weight ledger (source-read); any P-check failure aborts before T1 |
| D60-27 OOM (27.5 GiB peak vs 32 GB card, C-track sharing GPU5) | snapshot gate ≥ 29 GB free; stop-loss ladder (a)-(c) frozen in §3 |
| far-field slab feq not bitwise (broadcast vs full-field) | P4 torch.equal green (ab27.json); elementwise ops on identical scalars |
| CV-27 port subtle range/margin bug | port proven exactly equal to archived instrument at 19 (mini27_D12.json); only the direction range 1..26 differs (rest dir c=0 contributes nothing) |
| engine lattice-u conversion silently ≠ 0.05 | runtime guard (engine init ux check) + re_eff assert + archive twin check |
| slower than 1.55× (per-q feq launch overhead) | budget margin 1.9 h; if projection at 25% of T2 exceeds 4.6 h total, finish T1, report T2 as budget-blocked (owner decides) — no step-count reduction |

## 6. Verdict semantics

verified/ promotion candidate iff G1 ∧ G2 ∧ G3 hold on both tiers AND all
G4 disclosure rows present. All numbers machine-written by verify27.py
from tier JSONs + reflock files; zero hand transcription. Cross-stencil
comparison vs the archived D3Q19 numbers is a disclosure row, not a gate.

## Appendix A (appended 2026-10-07 after gate runs; append-only, md5 re-issued)

Gate runs executed per frozen commands (P1-P5 rerun exit 0; GPU5). Results
machine-written to sphere27/gate/ (t1_D40_27.json, t2_D60_27.json, t1/t2.run.log)
and judged by verify27.py into gate/verdict27.json: PASS (G1/G2/G3 all true);
G4 disclosure rows included. No re-runs, no clause changes.

## Appendix B (receipt completion, appended 2026-10-07; append-only, no other edits)

冻结版全文 md5 = 8a74eeceb16320d16251adca1fcbd775（2026-10-06 Phase-0 冻结，控制器 22/22 亲验）。
gate 附录 A 追加于其后（append-only，冻结前缀一字未动）；附录 A 后全文 md5 =
2f4e232c724c1481f5fd3ee53c366d8e（2026-10-07 首次重发边车时的值）。本附录 B 再追加后
的现行全文 md5 与 sha256 见同目录 prereg.md.md5 边车（frozen/current/sha256 三行）。
