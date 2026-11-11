cd /root/tp-agent/work/m14496 || exit 1
# Deterministic systematic sample of the test suite (every 20th test file in
# sorted order, plus the poiseuille/wall-model regression) as the fallback
# evidence for item 1: two full-sweep attempts were killed by host contention
# (load average ~1300 on 128 cores), see docs/AUDIT_RUNTEST_6_TENSORLBM.md.
R=runs/audit_runtest_r6
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src
FILES=$(ls tests/test_*.py | sort | awk 'NR%20==1')
echo "$FILES" > $R/sample_files.txt
nice -n 8 python -m pytest $FILES tests/test_poiseuille_wallmodel_arms.py \
  -q -p no:cacheprovider --timeout=600 -n 8 --tb=line \
  --junitxml=$R/junit_sample.xml > $R/pytest_sample.log 2>&1
echo "sample rc=$? $(date +%H:%M)" >> $R/bench_batch_progress.txt