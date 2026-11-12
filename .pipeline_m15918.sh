#!/usr/bin/env bash
# m15918 pipeline: wait for the running grid levels, then aggregate + field + report.
set -x
cd /root/tp-agent/work/m15918
export PYTHONPATH=/root/tp-agent/work/m15918/src
ART=benchmarks/verified/cylinder/re100_dfg/artifacts

# 1. wait for the two background level runs (D28, D40) to exit
while pgrep -f "re100_dfg/run.py --levels 40" > /dev/null || \
      pgrep -f "re100_dfg/run.py --levels 28" > /dev/null; do
  sleep 10
done
echo "levels finished $(date)" >> logs_local/pipeline.log

# 2. field snapshot (D=20, 8000 steps, separate tag so it cannot clobber the ladder)
python benchmarks/verified/cylinder/re100_dfg/run.py --levels 20 --steps 8000 \
  --threads 6 --tag snapshot --field >> logs_local/pipeline_field.log 2>&1
echo "field done rc=$? $(date)" >> logs_local/pipeline.log

# 3. aggregate the three-level ladder (errors re-derived from bands + units)
python benchmarks/verified/cylinder/re100_dfg/run.py --aggregate-only \
  --levels 20 28 40 >> logs_local/pipeline_agg.log 2>&1
echo "aggregate done rc=$? $(date)" >> logs_local/pipeline.log

# 4. report + figures + /tmp/report.json + artifacts/benchmark.json
python benchmarks/verified/cylinder/re100_dfg/make_report.py --tag u0833 \
  --levels 20 28 40 >> logs_local/pipeline_report.log 2>&1
echo "report done rc=$? $(date)" >> logs_local/pipeline.log
touch logs_local/PIPELINE_DONE