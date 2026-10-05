#!/bin/bash
# queue1: smoke + sq vf=0.30 ladder (prereg rung set is larger; this is the
# first-wave rolling queue, coarse rungs first so results start landing).
cd /nfs/wangxi/runs/bm_widen_w9_20260929/hex_ref/src
PY=/nfs/wangxi/venvs/hexref/bin/python
set -x
$PY run_case.py sq 0.30 96 96
$PY run_case.py sq 0.30 192 192
$PY run_case.py sq 0.30 288 288
$PY run_case.py sq 0.30 384 384
$PY run_case.py sq 0.30 512 512
$PY run_case.py sq 0.30 768 768
echo QUEUE1_ALL_DONE
