#!/bin/bash
# Continue m459 backward_step checkpoint/resume run from seg5 (step 25000)
# Target: 150k steps total (need 125k more = 25 segments of 5k each)

set -e

cd /root/tp-agent/work/m459

echo "=========================================="
echo "Mission m459 — Continue Checkpoint/Resume"
echo "Starting from: seg5 (step 25000)"
echo "Target: 150000 steps (25 more segments)"
echo "=========================================="

START_SEG=6
END_SEG=30
STEPS_PER_SEG=5000

for i in $(seq $START_SEG $END_SEG); do
  prev=$((i-1))
  ckpt="artifacts/m459_ckpt/seg${prev}.pt"
  
  if [ ! -f "$ckpt" ]; then
    echo "ERROR: Checkpoint $ckpt not found!"
    exit 1
  fi
  
  echo ""
  echo "------------------------------------------"
  echo "Segment $i/$END_SEG (resuming from step $((prev * STEPS_PER_SEG)))"
  echo "------------------------------------------"
  
  timeout 90 python3 benchmarks/pending/backward_step/run_segment.py \
    --m 32 --steps $STEPS_PER_SEG --device sdaa:0 \
    --resume "$ckpt" \
    --out "artifacts/m459_run/seg${i}.json" \
    --save-ckpt "artifacts/m459_ckpt/seg${i}.pt" \
    --seg-id $i --compile-mode eager
  
  if [ $? -ne 0 ]; then
    echo "Segment $i failed!"
    exit 1
  fi
  
  echo "✓ Segment $i complete"
done

echo ""
echo "=========================================="
echo "All segments complete!"
echo "Total steps: $((END_SEG * STEPS_PER_SEG))"
echo "=========================================="

# Re-run analysis
echo ""
echo "Running analysis..."
python3 benchmarks/pending/backward_step/analyze_m459.py

# Re-generate plots
echo ""
echo "Generating plots..."
python3 benchmarks/pending/backward_step/plot_m459.py

# Re-generate report
echo ""
echo "Generating report..."
python3 benchmarks/pending/backward_step/generate_report_m459.py

echo ""
echo "=========================================="
echo "DONE! Report at /tmp/report_m459.json"
echo "POST to API with:"
echo "  curl -X POST \$TENSOR_API_BASE/tensor/reports \\"
echo "    -H 'Authorization: Bearer \$TOKEN' \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d @/tmp/report_m459.json"
echo "=========================================="
