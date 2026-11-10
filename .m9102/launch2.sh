cd /root/tp-agent/work/m9102
export OMP_NUM_THREADS=48
LABEL=elbm_re180 RETAU=180 UTAU=0.012 NX=64 NY=66 NZ=32 STEPS=26000 WARMUP=8000 SAMPLE=15 NITER=6 nohup python .m9102/run_case.py > artifacts/elbm/logs/re180.log 2>&1 &
