#!/bin/bash
# Launch cylinder_3d investigation runs (2026-09-29).
set -u
cd /root/TensorLBM_feat2
export PYTHONPATH=/root/TensorLBM_feat2/src
export FT_TIMEOUT=86400
export OMP_NUM_THREADS=8
unset TL_FS_TOGAS_EPS TL_FS_DIAG_FIELD TL_FS_WALL_MODE TL_FS_APRIME TL_FS_LIQ_TO_IFACE 2>/dev/null || true

LOGDIR=/root/TensorLBM_feat2/runs/cyl3d_20260929
mkdir -p "$LOGDIR"
B=/root/TensorLBM_feat2/benchmarks/pending/cylinder_3d

launch () {
  local name="$1"; shift
  setsid nohup "$@" > "$LOGDIR/${name}.log" 2>&1 &
  echo "launched $name pid=$!"
}

# 3D extruded probes (multi-formula) at several blockage / resolution settings
launch p3d_D20_L16 python -u "$B/_probe_formulas.py" \
  --D 20 --lateral 16 --nz 4 --steps 80000 --device sdaa:0 \
  --out /tmp/cyl3d_probe_D20_L16.json

launch p3d_D20_L12 python -u "$B/_probe_formulas.py" \
  --D 20 --lateral 12 --nz 4 --steps 80000 --device sdaa:1 \
  --out /tmp/cyl3d_probe_D20_L12.json

launch p3d_D40_L16 python -u "$B/_probe_formulas.py" \
  --D 40 --lateral 16 --nz 4 --steps 80000 --device sdaa:2 \
  --out /tmp/cyl3d_probe_D40_L16.json

# independent 2D free-stream cross-checks (library D2Q9 + Ladd MEM)
launch p2d_D40 python -u "$B/_ref2d_re40.py" \
  --D 40 --domain-D 40 --steps 60000 --device sdaa:3 --out /tmp/cyl2d_re40_D40.json

launch p2d_D64 python -u "$B/_ref2d_re40.py" \
  --D 64 --domain-D 40 --steps 60000 --device sdaa:4 --out /tmp/cyl2d_re40_D64.json

sleep 5
echo "all launched"