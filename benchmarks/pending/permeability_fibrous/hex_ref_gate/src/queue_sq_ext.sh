#!/bin/bash
# Extension ladder per controller 2026-09-29 directive (gate 0.5% unchanged):
# vf0.7 += 2048/3072/4096; vf0.6 += 1792/2304; vf0.4 += 1536/1792; vf0.5 += 1536/1792.
# All 9 runs IN PARALLEL (192-core box, each run ~1 core; memory dominated by
# the 4096 splu factors, budgeted << 1TB). Idempotent per run_case.py SKIP.
cd /nfs/wangxi/runs/bm_widen_w9_20260929/hex_ref/src
PY=/nfs/wangxi/venvs/hexref/bin/python
export TMPDIR=/nfs/wangxi/tmp
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
L=../out/logs
$PY run_case.py sq 0.7 2048 2048 >> $L/queue_sq_ext_vf0.7_nx2048.log 2>&1 &
$PY run_case.py sq 0.7 3072 3072 >> $L/queue_sq_ext_vf0.7_nx3072.log 2>&1 &
$PY run_case.py sq 0.7 4096 4096 >> $L/queue_sq_ext_vf0.7_nx4096.log 2>&1 &
$PY run_case.py sq 0.6 1792 1792 >> $L/queue_sq_ext_vf0.6_nx1792.log 2>&1 &
$PY run_case.py sq 0.6 2304 2304 >> $L/queue_sq_ext_vf0.6_nx2304.log 2>&1 &
$PY run_case.py sq 0.4 1536 1536 >> $L/queue_sq_ext_vf0.4_nx1536.log 2>&1 &
$PY run_case.py sq 0.4 1792 1792 >> $L/queue_sq_ext_vf0.4_nx1792.log 2>&1 &
$PY run_case.py sq 0.5 1536 1536 >> $L/queue_sq_ext_vf0.5_nx1536.log 2>&1 &
$PY run_case.py sq 0.5 1792 1792 >> $L/queue_sq_ext_vf0.5_nx1792.log 2>&1 &
wait
echo QUEUE_SQ_EXT_ALL_DONE
