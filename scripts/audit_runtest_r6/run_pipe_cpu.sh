cd /root/tp-agent/work/m14496 || exit 1
R=runs/audit_runtest_r6
export PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=1
python benchmarks/verified/poiseuille_3d_pipe/run.py single 20 $R/pipe_R20.json --device cpu --compile-mode eager > $R/bench_poiseuille_3d_pipe.log 2>&1
echo "pipe_cpu_eager rc=$? $(date +%H:%M)" >> $R/bench_batch_progress.txt