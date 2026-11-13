"""Run the full (active) arm matrix of XVAL-LBM-LAYERED-POISEUILLE-ANALYTIC-INLET.

Sequential on one node; every arm writes its own JSON so an interrupted run keeps
the completed arms.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "TensorLBM", "src"))

from tensorlbm.verification.layered_poiseuille import cases
from tensorlbm.verification.layered_poiseuille.run_case import run_arm

OUT = os.environ.get("LP_OUT", os.path.join(ROOT, "artifacts",
                                            "xval_layered_poiseuille_analytic_inlet"))

if __name__ == "__main__":
    only = sys.argv[1:] or None
    t_all = time.time()
    for spec in cases.ACTIVE_ARMS:
        if only and spec["label"] not in only:
            continue
        t0 = time.time()
        print(f"=== {spec['label']} (H={spec['height']}, {spec['x_boundary']}) ===", flush=True)
        run_arm(spec, OUT)
        print(f"    -> {time.time() - t0:.0f}s", flush=True)
    print(f"ALL DONE in {time.time() - t_all:.0f}s -> {OUT}", flush=True)
