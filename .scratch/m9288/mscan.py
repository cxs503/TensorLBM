import sys, torch
sys.path.insert(0, sys.argv[1])
from tensorlbm.yield_stress_channel import run_bingham_poiseuille
errs = []
for m in [10.0, 50.0, 100.0, 500.0, 1000.0]:
    r = run_bingham_poiseuille(8, 1e-5, 2e-5, 0.02, m, max_iter=5000, tol=1e-4, dtype=torch.float64)
    errs.append(r["l2_error"])
print(sys.argv[1], [round(e,4) for e in errs])
