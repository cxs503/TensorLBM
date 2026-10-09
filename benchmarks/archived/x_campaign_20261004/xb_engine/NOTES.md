# XB engine track — execution log

Staging: /nfs/wangxi/runs/x_campaign_20261004/xb_engine/
Baseline worktree: /nfs/wangxi/worktrees/mp_w11 @ a84745e56f (READ-ONLY)
Iron rules: all compute on 5090 (ssh LBM@192.168.215.56), /nfs only,
python=/nfs/wangxi/venvs/tensorlbm/bin/python, GPU6 = campaign time-share
(snapshot before use, log window below), GPU5=T1 track (do not touch),
0-4/7 do not use. OMP_NUM_THREADS<=16 on CPU.

## Step 0 diff conclusions (2026-10-04, pre-prereg)
- XB-1: library cac_lbm.py (mp_w11, md5 ccc671ee4738d9266967e74f1fd8924b)
  vs M1 staging src_patched/cac_lbm.py (abc815766211aa2bb504cbb4b714dbe7):
  AST-semantically identical (97 formatting lines + one unused OPPOSITE
  import in M1 copy). Library version copied to src_patched/tensorlbm/.
  => M1 P1 law (1-sigma_eff ~ 1.1/W^2) was measured on exactly these
  operator semantics; fix targets iso_gradient_3d / iso_laplacian_3d.
- XB-2: library multiphase.py (44df8b51...) is NEWER than M4 staging copy
  (af7ea82e...): #322 landed correct-sign psi factories
  (make_psi_carnahan_starling a=5/b=4, psi_eos_carnahan_starling,
  make_psi_eos) with documented convention p = rho*cs2 - (G_lib/2)*cs2*psi^2,
  G_lib>0 attractive (library F = -G*psi*backward-gather). M4 harness
  (scmp_lib.py) ran on mp_w10 tensorlbm + local make_psi_correct with
  G=+1 pairing — physics-identical to landed convention.
- Engine stack: mp_w11 (a84745e5=#322+#323) vs mp_w10 (1eae1c3b) differ in
  multiple modules incl. multiphase.py -> bit-reproduction arms A6(XB-1)/
  B1(XB-2) against archives are mandatory; fallback comparator = own rerun
  (pre-declared in preregs).
- guo forcing dead at T_r<=0.85 (p3/p4_guo_corrected: NaN @step 1000, tau
  ladder does not save) -> XB-2 multirange arms use sc forcing, edm spot-check.

## GPU6 window log
- 2026-10-04 snapshot: GPU0-4 ~32GB used (not ours), GPU5 5MiB (T1 idle
  right now but reserved), GPU6 5MiB 0% -> free for XB.

## Locked references (refs/, md5 frozen in refs/step0_copies.md5)
cs_maxwell.json / pr_maxwell.json (Maxwell coexistence tables, machine
precision), a_maxwell.json, b_laplace.json (sigma_by_L pr@0.55:
0.32022612409321427 / 0.32649180 / 0.32942224), c_spurious_currents.json
(c-table baseline u_max, R=20 L=256 tau=1 G=+1 psi_correct, sc==edm @tau=1),
b5_a40_k0.5.json / b5_a20_k0.5.json (M1 Route-B frozen arms),
calib_W4_R28.json (bit-reproduction anchor), calib_sigma.py /
b5_hr_droplet.py / scmp_lib.py / run_formal_c.py (protocol sources).

## Formal battery freeze + GPU6 window (2026-10-05)
- Discriminating tests ALL PASS before formal batteries (controller
  condition): tests/test_multirange.json (11/11) 2026-10-04 14:16;
  tests/test_iso4.json (9/9, T3 fixed: per-lattice vs physical units
  erratum documented in-file) 2026-10-04 14:25.
- Runner md5 frozen BEFORE launch (smoke-tested: XB-2 harness G2=0/-0.1
  finite + exact mass conservation; XB-1 iso4 32^3 20 steps finite; one
  pre-launch bug caught by smoke = _force positional scheme/G2 mixup,
  fixed to keyword args):
  scmp_lib_xb.py  97d2259bbc7373d2dd769943cf0743c5
  xb1_formal.py   3e885d7c244a53083af10409a7adeb1d
  xb2_formal.py   5d7935fb92e567d96d9b77a575c699f0
- GPU6 window OPEN 2026-10-05 10:4x CST: snapshot before open = 5 MiB /
  0% (GPU0-4 occupied by others ~32GB, GPU5 5MiB reserved T1, not touched).
  Chained: xb2_formal.py (B1->B6) then xb1_formal.py (A5->A1->A2->A3->A4),
  CUDA_VISIBLE_DEVICES=6, OMP_NUM_THREADS=8, logs out/xb2_formal.log,
  out/xb1_formal.log.  Budget plan ~6.4 GPU*h (approved 6.7).
- xb2_formal.py v2 relaunch fix (append-only erratum, 2026-10-05): B2 first
  arm crashed in drift_block generator (NameError 'o' -> 'out'); B1 bit-check
  arms already in out/xb2_partial.json before crash, rerun is cheap and
  fresh-process deterministic.  New md5: see below.  Relaunch chained after
  xb1 completes.
  xb2_formal.py  6dcff3090e8daa83cebf0087399e69e1
  (correction 2026-10-05: the xb2 v2 md5 line above carried a wrong value
   written before the file hash was taken; frozen v2 md5 on disk =
   d9fb7a4d8194da10191bd8683c734ce6, verified post-deploy.)
- xb2 v3 disclosure (2026-10-05, controller instructions 1+2): crash
  timeline 10:33 v1 run aborted in B2 arm 1/42 (drift_block genexpr
  NameError 'o'); evidence out/xb2_partial.json (B1 2 arms bit-equal ->
  base_source=archive) + out/xb2_formal.log (traceback) PRESERVED, never
  deleted.  v3 = deployed xb2_formal.py md5 ca25c95af0ce253617a5e631a313bc11
  (controller label 'v3'; supersedes my earlier NOTES line '6dcff...' which
  was a pre-hash miswrite, and the d9fb... v2 which this extends).  Glue-code
  diffs v1->v3, prereg criteria untouched: (1) drift_block 'o'->'out';
  (2) partial output renamed xb2_v3_partial.json + end-of-run os.remove
  dropped (evidence-preservation); (3) final output renamed
  out/xb2_v3_formal.json.  Relay waiter xb2_v3_waiter.sh (md5
  50ffe5a2beb95dbd529e72465098cef5, pid 2839758, replaces my ad-hoc
  watcher 2832836): waits for xb1 exit, snapshots GPU6 into NOTES
  snapshot-before-use, runs v3, writes out/xb2_v3_done.marker, log
  out/xb2_v3_formal.log (crash log kept verbatim).  Race vs XA-3 queue
  watcher (pid 2835022): XA-3 needs 3 min of no-xb[12]+GPU6<500MiB, v3
  allocates within ~85s of xb1 exit -> XA-3 defers, no conflict.
- CONTROLLER (2026-10-05 13:5x): xb1 battery COMPLETE 13:37 (A1-A5 + formal
  summary; A1/A3/A4 gates all honest FAIL, controller-verified 16/16+5/5+9/9).
  XA-3 window opened 13:39:30 (xa3_run pid 2857406).  xb2 v1 waiter found
  deadlocked (pgrep self-launcher match on 2839752) -> retired together with
  its launcher; relay re-armed as waiter2 (xb2_v3_waiter2.sh, keys on
  xa3_run exit + 3-min GPU6 idle) with identical v3 run/marker semantics.
- xb2 v3 relay window (waiter v2, 2026-10-05 ~15:08): 
  xa3_run exited + GPU6 idle 3 min; snapshot before v3 start: 5 MiB, 0 %;
  launching xb2_formal.py v3 (md5 below); v1 waiter 2839758 retired
  (deadlocked on launcher 2839752 cmdline match, see NOTES above):
  ca25c95af0ce253617a5e631a313bc11  xb2_formal.py
