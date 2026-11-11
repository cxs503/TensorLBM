cd /root/tp-agent/work/m14496 || exit 1
R=runs/audit_runtest_r6
export PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# H=10/20 keeps the archived step budget (5000) -> valid ladder, now with the
# steady-state diagnostic recorded per case.
python benchmarks/verified/poiseuille_convergence_order/run.py --H 10 20 --steps 5000 \
  --out $R/conv_fix_H10_20.json > $R/conv_fix_H10_20.log 2>&1
echo "H10_20 rc=$? $(date +%H:%M)" >> $R/convfix_progress.txt
# H=20/40 with the step budget scaled like H^2 (transient time ~ H^2 / nu):
# 8000 steps was shown to be under-converged (apparent order went negative).
python benchmarks/verified/poiseuille_convergence_order/run.py --H 20 40 --steps 40000 \
  --out $R/conv_fix_H20_40.json > $R/conv_fix_H20_40.log 2>&1
echo "H20_40_40k rc=$? $(date +%H:%M)" >> $R/convfix_progress.txt