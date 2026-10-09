#!/bin/bash
# Launch all 4 mem_vs_pf benchmarks in parallel on SDAA 12-15
cd /root/TensorLBM_dev
export PYTHONPATH=src

echo "=== Launching 4 benchmarks in parallel ==="
echo "Start: $(date)"

# Cylinder on SDAA:12
python mem_vs_pf_worker.py cylinder 12 mem_vs_pf_cylinder_sdaa12.json \
  > log_mem_vs_pf_cylinder_sdaa12.txt 2>&1 &
PID_CYL=$!

# Couette on SDAA:13
python mem_vs_pf_worker.py couette 13 mem_vs_pf_couette_sdaa13.json \
  > log_mem_vs_pf_couette_sdaa13.txt 2>&1 &
PID_COU=$!

# Poiseuille on SDAA:14
python mem_vs_pf_worker.py poiseuille 14 mem_vs_pf_poiseuille_sdaa14.json \
  > log_mem_vs_pf_poiseuille_sdaa14.txt 2>&1 &
PID_POI=$!

# SUBOFF on SDAA:15
python mem_vs_pf_worker.py suboff 15 mem_vs_pf_suboff_sdaa15.json \
  > log_mem_vs_pf_suboff_sdaa15.txt 2>&1 &
PID_SUB=$!

echo "PIDs: cyl=$PID_CYL cou=$PID_COU poi=$PID_POI sub=$PID_SUB"

# Wait for all
wait $PID_CYL; echo "Cylinder done (exit=$?)"
wait $PID_COU; echo "Couette done (exit=$?)"
wait $PID_POI; echo "Poiseuille done (exit=$?)"
wait $PID_SUB; echo "SUBOFF done (exit=$?)"

echo "All done: $(date)"
