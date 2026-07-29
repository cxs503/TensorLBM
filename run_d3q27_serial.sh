#!/bin/bash
# Serial D3Q27 benchmark runner — runs all 8 cases one at a time
# Logs to /tmp/d3q27_serial/ directory
set -e

cd /root/TensorLBM_dev
export PYTHONPATH=src

LOGDIR=/tmp/d3q27_serial
mkdir -p "$LOGDIR"
rm -f "$LOGDIR"/*.log /tmp/d3q27_serial_results.json

echo "=== D3Q27 CASCADED/CUMULANT SERIAL BENCHMARKS ==="
echo "Start: $(date)"
echo ""

# Define all 8 benchmarks: name did nx ny nz hl steps op
run_one() {
    local name=$1 did=$2 nx=$3 ny=$4 nz=$5 hl=$6 steps=$7 op=$8
    local log="$LOGDIR/${name}.log"
    echo ">>> [$name] Starting: ${nx}x${ny}x${nz} hl=${hl} steps=${steps} op=${op} SDAA:${did} ($(date))"
    python -u _d3q27_worker.py "$did" "$name" "$nx" "$ny" "$nz" "$hl" "$steps" "$op" > "$log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "<<< [$name] COMPLETED ($(date))"
    else
        echo "<<< [$name] FAILED rc=$rc ($(date))"
    fi
    # Extract last few lines for quick preview
    tail -3 "$log" | sed "s/^/  [$name] /"
    echo ""
}

# ── A. SHIP HULLS ──
run_one "kvlcc2"   0 200 60 60 80.0  2000 cascaded
run_one "wigley"   1 200 60 60 80.0  2000 cascaded
run_one "kcs"      2 200 60 60 80.0  2000 cascaded

# ── B. CONFIRMATION ──
run_one "suboff"            3 200 80 80 80.0  3000 cascaded
run_one "suboff_cumulant"   0 200 80 80 80.0  2000 cumulant

# ── C. BLUFF BODIES ──
run_one "cylinder"  0 200 80 4  24.0  2000 cascaded
run_one "sphere"    1 120 60 60 24.0  2000 cascaded

# ── D. LARGER GRID ──
run_one "suboff"    0 256 103 103 102.4 2000 cascaded

echo ""
echo "=== ALL DONE ($(date)) ==="
echo ""

# ── Summary ──
echo "=== SUMMARY ==="
for log in "$LOGDIR"/*.log; do
    name=$(basename "$log" .log)
    last=$(tail -5 "$log" | grep -E "DONE|DIV|ERROR|FAIL|Ct=" | tail -1 || echo "???")
    echo "[$name] $last"
done

echo ""
if [ -f /tmp/d3q27_serial_results.json ]; then
    python3 -c "
import json, math
data = json.load(open('/tmp/d3q27_serial_results.json'))
print(f'{\"Case\":<20s} {\"Grid\":<12s} {\"Op\":<10s} {\"Ct/Cd\":>8s} {\"Err%\":>7s} {\"Time\":>7s} {\"Fin\":>5s}')
print('-'*70)
for r in sorted(data, key=lambda x: x.get('did',0)):
    val = r.get('Ct_total', float('nan'))
    err = r.get('error_pct', float('nan'))
    t = r.get('elapsed_s', 0)
    fin = 'YES' if r.get('finite') else 'NO'
    val_s = f'{val:>8.5f}' if isinstance(val,(int,float)) and math.isfinite(val) else 'N/A'
    err_s = f'{err:>7.1f}' if isinstance(err,(int,float)) and math.isfinite(err) else 'N/A'
    print(f'{r[\"case\"]:<20s} {r[\"grid\"]:<12s} {r[\"collision\"]:<10s} {val_s} {err_s} {t:>7.0f}s {fin:>5s}')
"
else
    echo "No results file found."
fi
