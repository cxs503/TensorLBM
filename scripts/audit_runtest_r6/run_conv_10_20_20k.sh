cd /root/tp-agent/work/m14496 || exit 1
R=runs/audit_runtest_r6
export PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# steady fine grid: 5000 steps left H=20 with a 2e-3 residual transient, which is
# the same size as the discretisation error being measured (finding D10).
python benchmarks/verified/poiseuille_convergence_order/run.py --H 10 20 --steps 20000 \
  --out $R/conv_fix_H10_20_steps20k.json > $R/conv_fix_H10_20_steps20k.log 2>&1
echo "H10_20_20k rc=$? $(date +%H:%M)" >> $R/convfix_progress.txt