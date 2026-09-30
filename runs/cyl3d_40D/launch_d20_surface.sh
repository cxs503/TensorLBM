#!/usr/bin/env bash
# Live NEW-CODE confirmation run: surface/all/interior MEM calibers reported
# in-process by run.py.  D=20 @ 40D domain, nz=1 (nz-independent), to plateau.
set -u
cd /root/TensorLBM_feat2
export SDAA_HOME=/opt/tecoai
export LD_LIBRARY_PATH=/opt/tecoai/lib64:/usr/lib64/openmpi/lib:/usr/local/lib:
export OMP_NUM_THREADS=2
export PYTHONPATH=/root/TensorLBM_feat2/src
OUT=runs/cyl3d_40D
mkdir -p "$OUT"
setsid nohup python benchmarks/pending/cylinder_3d/run.py single 20 \
  --out "$OUT/D20_L40_surface.json" --save-field --steps 40000 --nz 1 --lateral 40 \
  --device sdaa:3 --sample 100 --compile-mode eager \
  > "$OUT/D20_L40_surface.log" 2>&1 < /dev/null &
echo "launched D20_L40_surface pid=$!"