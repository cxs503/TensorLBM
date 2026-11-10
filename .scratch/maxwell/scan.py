import sys, time
sys.path.insert(0, "/root/tp-agent/work/m8675/src")
import torch
from tensorlbm.multiphase import make_psi_eos
from tensorlbm.sc_eos_coexistence import maxwell_coexistence

TC = {"pr": 0.07291903717301021, "cs": 0.09432870314611914}
RIMAX = {"pr": 0.98*21/2, "cs": 0.98*4/4.0}

for Tr in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.42, 0.40]:
    T = Tr*TC["pr"]
    t0=time.time()
    r = maxwell_coexistence(+1.0, make_psi_eos("pr", T), rho_lo=1e-6, rho_hi=RIMAX["pr"], n=12000)
    print(f"Tr={Tr:.2f} T={T:.6f} found={r.found} ratio={r.density_ratio} "
          f"rl={r.rho_liquid} rv={r.rho_vapour} psat={r.p_sat} area={r.area_residual} "
          f"spin=({r.rho_spinodal_vapour},{r.rho_spinodal_liquid}) {time.time()-t0:.1f}s", flush=True)
