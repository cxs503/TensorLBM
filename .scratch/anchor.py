import sys, time; sys.path.insert(0,'/root/tp-agent/work/m9874/src')
import torch; torch.set_num_threads(2)
from tensorlbm.contact_line import three_phase as tp
t0=time.time()
for f in (1.0, 0.6, 0.3):
    r = tp.wetting_tension_ti(f, ny=48, n_ladder=7, n_steps=2500)
    print('f=%.2f dgamma=%.4f gsl=%.4f gsg=%.4f t=%.0f'%(f, r.delta_gamma, r.gamma_sl, r.gamma_sg, time.time()-t0), flush=True)
    print('  I_l', ['%.4f'%m['moment_l'] for m in r.ladder], flush=True)
    print('  I_g', ['%.4f'%m['moment_g'] for m in r.ladder], flush=True)
