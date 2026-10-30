#!/usr/bin/env bash
# sphere_re200 checkpoint-resume runner
# Runs in stages (2500 → 6000 → 9000 → 12000), saving intermediate results.
# Re-entrant: if killed, re-running picks up from last completed stage.
#
# Usage:
#   bash run_checkpoint_resume.sh [D] [target_steps]
#   e.g. bash run_checkpoint_resume.sh 20 12000
set -euo pipefail
cd "$(dirname "$0")"

D=${1:-20}
TARGET=${2:-12000}
KERNEL=${KERNEL:-sparse}
LAT=${LAT:-3.0}
UP=${UP:-3.0}
DOWN=${DOWN:-4.0}
ULB=${ULB:-0.05}
SAMPLE=${SAMPLE:-50}

export PYTHONPATH=${PYTHONPATH:-/root/TensorLBM_feat2/src}
export W8A_DEV=${W8A_DEV:-sdaa:0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTBASE="re200_D${D}_checkpoint"
LOCKFILE="${OUTBASE}.lock"
STAGES=(2500 6000 9000 ${TARGET})

# Calculate tau
TAU=$(python3 -c "print(0.5 + 3.0 * ${ULB} * ${D} / 200.0)")

echo "=== sphere_re200 checkpoint runner ==="
echo "D=${D}, target=${TARGET}, tau=${TAU}, kernel=${KERNEL}"
echo "Stages: ${STAGES[*]}"

# Check lock
if [ -f "$LOCKFILE" ]; then
    PID=$(cat "$LOCKFILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Another instance running (PID $PID). Exiting."
        exit 1
    else
        echo "Stale lock found. Removing."
        rm -f "$LOCKFILE"
    fi
fi
echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

COMPLETED_STAGES="${OUTBASE}.progress"
touch "$COMPLETED_STAGES"

for STAGE in "${STAGES[@]}"; do
    if grep -q "^${STAGE}$" "$COMPLETED_STAGES" 2>/dev/null; then
        echo "Stage ${STAGE} already completed. Skipping."
        continue
    fi
    
    STAGE_OUT="${OUTBASE}_stage${STAGE}.json"
    echo ""
    echo "--- Running stage: ${STAGE} steps ---"
    
    python3 run.py "$STAGE_OUT" \
        --kernel "$KERNEL" --D "$D" --steps "$STAGE" \
        --lat "$LAT" --up "$UP" --down "$DOWN" \
        --tau "$TAU" --ulb "$ULB" --sample "$SAMPLE"
    
    echo "$STAGE" >> "$COMPLETED_STAGES"
    
    # Extract Cd at this stage
    python3 -c "
import json
d = json.load(open('${STAGE_OUT}'))
h = d['history']
vals = [x['cd'] for x in h]
n = len(vals)
w = max(1, int(n*0.2))
tail = vals[-w:]
cd = sum(tail)/len(tail)
ref = 24.0/200*(1+0.15*200**0.687)
err = (cd-ref)/ref*100
print(f'  Stage ${STAGE}: Cd={cd:.6f}, err={err:+.3f}%, samples={n}, window={w}')
"
done

echo ""
echo "=== All stages complete ==="
echo "Progress file: ${COMPLETED_STAGES}"

# Run convergence analysis
python3 analyze_convergence.py
