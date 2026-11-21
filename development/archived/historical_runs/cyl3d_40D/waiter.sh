#!/usr/bin/env bash
# Wait until all four cylinder_3d 40D runs have exited.
cd /root/TensorLBM_feat2
while true; do
  alive=$(pgrep -f "cylinder_3d/run.py single .* --lateral 40" | wc -l)
  alive=$((alive + $(pgrep -f "cylinder_3d/run.py single .* --lateral 32" | wc -l)))
  if [ "$alive" -eq 0 ]; then break; fi
  sleep 60
done
echo "ALL FOUR 40D RUNS FINISHED"
for f in runs/cyl3d_40D/D20_L40.log runs/cyl3d_40D/D40_L40.log runs/cyl3d_40D/D20_L32.log runs/cyl3d_40D/D40_L32.log; do
  echo "--- $f ---"; grep -E "FINAL|saved" "$f" | tail -3
done