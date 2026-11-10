import sys, json, math, time
sys.path.insert(0,'/root/tp-agent/work/m9874/src')
import torch
torch.set_num_threads(2)
from tensorlbm.contact_line import three_phase as tp
t0=time.time()
G_LG = 0.05639
for f in (0.4, 0.6, 0.8):
    r = tp.wetting_tension_ti(f, ny=48, n_ladder=9, n_steps=4000)
    th = tp.young_angle_deg(r.gamma_sg, r.gamma_sl, G_LG)
    print("f=%.2f dgamma=%.5f cos=%.3f theta_Y=%.2f (gsl=%.5f gsg=%.5f) drift=%.2e/%.2e t=%.0fs" % (
        f, r.delta_gamma, r.delta_gamma/G_LG, th, r.gamma_sl, r.gamma_sg, r.mass_drift_l, r.mass_drift_g, time.time()-t0), flush=True)
    print("   ladder moments l:", ["%.3f"%m['moment_l'] for m in r.ladder], flush=True)
    print("   ladder moments g:", ["%.3f"%m['moment_g'] for m in r.ladder], flush=True)
