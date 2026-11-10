cd /root/tp-agent/work/m9102
export OMP_NUM_THREADS=48
LABEL=elbm_re180 RETAU=180 UTAU=0.012 NX=64 NY=66 NZ=32 STEPS=50000 WARMUP=15000 SAMPLE=20 NITER=6 nohup python .m9102/run_case.py > artifacts/elbm/logs/re180.log 2>&1 &
LABEL=elbm_re395 RETAU=395 UTAU=0.012 NX=64 NY=66 NZ=32 STEPS=35000 WARMUP=12000 SAMPLE=20 NITER=6 nohup python .m9102/run_case.py > artifacts/elbm/logs/re395.log 2>&1 &
