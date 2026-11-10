import sys, json, math, time
sys.path.insert(0,'/root/tp-agent/work/m9874/src')
import torch
torch.set_num_threads(2)
from tensorlbm.contact_line.three_phase import (planar_liquid_gas_tension, planar_wall_tension, young_angle_deg)
t0=time.time()
lg = planar_liquid_gas_tension(n_steps=6000, n_resolutions=1) if False else planar_liquid_gas_tension(n_steps=6000)
print("gamma_lg", lg['gamma'], "laplace", lg['gamma_laplace_reference'], "reldiff", lg['relative_difference'], "rho_l/v", lg['rho_l'], lg['rho_v'])
print("t", time.time()-t0, flush=True)
g_lg = lg['gamma']
out={}
for f in (0.3,0.4,0.5,0.6,0.7,0.8,0.9):
    sl = planar_wall_tension(bulk_density=1.957, wall_fraction=f, ny=64, n_steps=4000, n_resolutions=1)
    sg = planar_wall_tension(bulk_density=0.1596, wall_fraction=f, ny=64, n_steps=4000, n_resolutions=1)
    th = young_angle_deg(sg.gamma, sl.gamma, g_lg)
    out[f]=dict(sl=sl.gamma, sg=sg.gamma, dg=sg.gamma-sl.gamma, theta=th, rho_wall_l=sl.rho_wall, rho_wall_g=sg.rho_wall, drift=sl.mass_drift)
    print(f, "gsl=%.5f gsg=%.5f dg=%.5f cos=%.3f theta=%.1f rho_wall_l=%.2f rho_wall_g=%.3f t=%.0fs" % (sl.gamma, sg.gamma, sg.gamma-sl.gamma, (sg.gamma-sl.gamma)/g_lg, th, sl.rho_wall, sg.rho_wall, time.time()-t0), flush=True)
json.dump(out, open('/root/tp-agent/work/m9874/.scratch/pilot_tension.json','w'), indent=1)
