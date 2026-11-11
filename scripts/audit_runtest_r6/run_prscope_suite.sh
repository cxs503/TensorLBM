cd /root/tp-agent/work/m14496 || exit 1
R=runs/audit_runtest_r6
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src
nice -n 8 python -m pytest tests -q -p no:cacheprovider --timeout=900 -m "not gpu and not slow" -n 16 \
  --junitxml=$R/junit_prscope.xml > $R/pytest_prscope.log 2>&1
echo "prscope rc=$? $(date +%H:%M)" >> $R/bench_batch_progress.txt