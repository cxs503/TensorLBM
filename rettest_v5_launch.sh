#!/bin/bash
# Launch all 14 RetestV5 cases in parallel on SDAA 0-13
cd /root/TensorLBM_dev
rm -rf /tmp/rettest_v5
mkdir -p /tmp/rettest_v5

for i in $(seq 1 14); do
    did=$((i - 1))
    outfile=$(printf "/tmp/rettest_v5/case%02d.json" $i)
    echo "Launching case $i on SDAA:$did -> $outfile"
    PYTHONPATH=src nohup python rettest_v5_worker.py $i $did $outfile \
        > /tmp/rettest_v5/case${i}.log 2>&1 &
done

echo "All 14 cases launched. PIDs:"
jobs -p
wait
echo "All cases complete."
