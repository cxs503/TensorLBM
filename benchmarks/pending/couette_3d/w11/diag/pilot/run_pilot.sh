#!/usr/bin/env bash
set -x
D=/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings
WT=/nfs/wangxi/worktrees/bm_w11
PY=/nfs/wangxi/venvs/tensorlbm/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 TMPDIR=/nfs/wangxi/tmp
cd $D/couette3d/diag/pilot
PYTHONPATH=$WT/src $PY -c "import tensorlbm,sys;print(\"tensorlbm:\",tensorlbm.__file__);print(sys.version)" > provenance.txt 2>&1
for spec in "16 2000 12000" "32 6000 24000" "64 16000 48000"; do
  set -- $spec
  PYTHONPATH=$WT/src $PY $WT/benchmarks/pending/couette_3d/run.py single $1 $D/couette3d/diag/pilot/case_H$1.json --tau 0.8 --u0 0.05 --min-steps $2 --max-steps $3 --device cpu
done
echo PILOT_DONE
