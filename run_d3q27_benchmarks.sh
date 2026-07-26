#!/bin/bash
# D3Q27 Benchmark runner — all 8 SDAA workers in parallel with log files
set -e

cd /root/TensorLBM_dev
export PYTHONPATH=src
LOGDIR=/tmp/d3q27_logs
mkdir -p "$LOGDIR"
rm -f "$LOGDIR"/*.log /tmp/d3q27_bench_results.json

echo "=== D3Q27 CASCADED/CUMULANT BENCHMARKS ==="
echo "Launching 8 workers on 8 SDAA cards..."
echo ""

# Launch all 8 workers
python _d3q27_worker.py 0 suboff    200 80  80  80.0  5000 cascaded  > "$LOGDIR/0_suboff_cascaded.log" 2>&1 &
python _d3q27_worker.py 1 suboff    256 103 103 102.4 3000 cascaded  > "$LOGDIR/1_suboff256_cascaded.log" 2>&1 &
python _d3q27_worker.py 2 kvlcc2    200 80  80  100.0 3000 cascaded  > "$LOGDIR/2_kvlcc2_cascaded.log" 2>&1 &
python _d3q27_worker.py 3 wigley    200 80  80  100.0 3000 cascaded  > "$LOGDIR/3_wigley_cascaded.log" 2>&1 &
python _d3q27_worker.py 4 kcs       200 80  80  100.0 3000 cascaded  > "$LOGDIR/4_kcs_cascaded.log" 2>&1 &
python _d3q27_worker.py 5 cylinder  200 80  4   24.0  3000 cascaded  > "$LOGDIR/5_cylinder_cascaded.log" 2>&1 &
python _d3q27_worker.py 6 sphere    120 60  60  24.0  3000 cascaded  > "$LOGDIR/6_sphere_cascaded.log" 2>&1 &
python _d3q27_worker.py 7 suboff    200 80  80  80.0  5000 cumulant  > "$LOGDIR/7_suboff_cumulant.log" 2>&1 &

echo "All workers launched. Wait for completion..."
echo "Logs in $LOGDIR/"
echo ""

# Wait for all background jobs
wait

echo ""
echo "=== ALL WORKERS COMPLETE ==="
echo ""

# Show summary from each log
for log in "$LOGDIR"/*.log; do
    name=$(basename "$log" .log)
    last=$(tail -5 "$log" | grep -E "DONE|DIV|ERROR|FAIL" | tail -1 || echo "???")
    echo "[$name] $last"
done

echo ""
echo "=== AGGREGATED RESULTS ==="
if [ -f /tmp/d3q27_bench_results.json ]; then
    python3 -c "
import json
data = json.load(open('/tmp/d3q27_bench_results.json'))
print(f'{\"Case\":<20s} {\"Grid\":<12s} {\"Op\":<10s} {\"Ct/Cd\":>8s} {\"Err%\":>7s} {\"Time\":>7s} {\"Fin\":>5s}')
print('-'*70)
for r in sorted(data, key=lambda x: x['did']):
    val = r['Ct_total']
    err = r.get('error_pct', float('nan'))
    t = r['elapsed_s']
    fin = 'YES' if r['finite'] else 'NO'
    err_s = f'{err:.1f}' if isinstance(err, (int,float)) and err==err else 'N/A'
    print(f'{r[\"case\"]:<20s} {r[\"grid\"]:<12s} {r[\"collision\"]:<10s} {val:>8.5f} {err_s:>7s} {t:>7.0f}s {fin:>5s}')
"
else
    echo "No results file found."
fi
