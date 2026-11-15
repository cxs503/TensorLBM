cd /root/tp-agent/work/m9102
export OMP_NUM_THREADS=16
LABEL=elbm_re180_small RETAU=180 UTAU=0.012 NX=32 NY=34 NZ=16 STEPS=14000 WARMUP=4000 SAMPLE=10 NITER=6 nohup python .m9102/run_case.py > artifacts/elbm/logs/re180_small.log 2>&1 &
