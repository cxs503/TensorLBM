#!/bin/bash
# Run 5 ship hull drag benchmarks in parallel on SDAA 8-12
# Output: /tmp/ship_bench_results.json

set -e

export PYTHONPATH=/root/TensorLBM_dev/src
cd /root/TensorLBM_dev
mkdir -p /tmp/ship_bench_logs

HUllS=("wigley" "series60" "kcs" "kvlcc2" "npl")
DEVICES=("sdaa:8" "sdaa:9" "sdaa:10" "sdaa:11" "sdaa:12")
PIDS=()

echo "Starting 5 ship hull benchmarks..."
for i in "${!HUllS[@]}"; do
    hull="${HUllS[$i]}"
    dev="${DEVICES[$i]}"
    log="/tmp/ship_bench_logs/${hull}.log"
    echo "  [$hull] on $dev → $log"
    python ship_bench_worker.py \
        --hull "$hull" --device "$dev" --cs 0.05 \
        --n-steps 3000 --warmup 1000 \
        --nx 200 --ny 60 --nz 60 \
        --re 2000000 --hull-length 80 --u-in 0.06 \
        > "$log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All launched. PIDs: ${PIDS[*]}"
echo "Monitor with: tail -f /tmp/ship_bench_logs/*.log"
echo ""

# Wait for all
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    hull="${HUllS[$i]}"
    echo "Waiting for $hull (pid $pid)..."
    wait "$pid"
    echo "  [$hull] DONE (exit=$?)"
done

echo ""
echo "All benchmarks complete. Collecting results..."

# Collect JSON results
python3 -c "
import json, glob
results = {}
for logfile in sorted(glob.glob('/tmp/ship_bench_logs/*.log')):
    hull = logfile.split('/')[-1].replace('.log','')
    try:
        with open(logfile) as f:
            lines = f.read().strip().splitlines()
            # Find the last valid JSON
            for line in reversed(lines):
                try:
                    data = json.loads(line)
                    if 'Ct_total' in data:
                        results[hull] = data
                        break
                except:
                    continue
    except Exception as e:
        results[hull] = {'status': 'ERROR', 'error': str(e)}
    if hull not in results:
        results[hull] = {'status': 'NO_RESULT'}

config = {
    'lattice': 'D3Q19',
    'collision': 'MRT+Smagorinsky',
    'C_s': 0.05,
    'grid': '200×60×60',
    'Re': 2_000_000,
    'hull_length': 80,
    'u_in': 0.06,
    'n_steps': 3000,
    'warmup': 1000,
}
summary = {'config': config, 'results': results}

with open('/tmp/ship_bench_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Print summary
print('RESULTS:')
print(f\"{'Hull':<12} {'Ct_fric':>10} {'Ct_pres':>10} {'Ct_total':>10} {'Ct_ref':>10} {'Err%':>8}\")
print('-' * 62)
for hull in ['wigley', 'series60', 'kcs', 'kvlcc2', 'npl']:
    r = results.get(hull, {})
    ct_ref = r.get('Ct_reference', 0)
    if 'Ct_total' in r:
        print(f\"{hull:<12} {r['Ct_fric']:>10.5f} {r['Ct_pres']:>10.5f} {r['Ct_total']:>10.5f} {ct_ref:>10.5f} {r['error_pct']:>7.1f}%\")
    else:
        print(f\"{hull:<12} {'—':>10} {'—':>10} {'—':>10} {ct_ref:>10.5f} {'—':>8}\")
print()
"
