import sys, json, math, time, os
sys.path.insert(0,'/root/tp-agent/work/m9874/src')
import torch
torch.set_num_threads(2)
from tensorlbm.contact_line.engine import Channel, WallStepper, sessile_drop_state
from tensorlbm.contact_line import three_phase as tp
from tensorlbm.d2q9 import macroscopic

def run(f, nx=128, ny=64, steps=20000):
    ch = Channel(nx=nx, ny=ny, wall_rows=2)
    fdist, _ = sessile_drop_state(ch, radius=ny*0.32)
    st = WallStepper(ch, G=5.0, tau=1.0, wall_fraction=f, forcing="edm")
    solid = ch.solid_mask()
    traj=[]
    for n in range(1, steps+1):
        fdist = st.step(fdist, 0.0)
        if n % 1000 == 0:
            rho = macroscopic(fdist)[0]
            try:
                a = tp.sessile_contact_angle(rho, y_wall=ch.y_wall(), solid=solid, level=0.5*(1.949+0.159), side="cap")
            except Exception as e:
                a = {"theta_deg": float('nan'), "unc_deg": float('nan')}
            traj.append((n, a["theta_deg"]))
    rho = macroscopic(fdist)[0]
    ux, uy = st.velocity(fdist)
    return dict(f=f, theta=a["theta_deg"], unc=a["unc_deg"], traj=traj,
                umax=float(torch.sqrt(ux**2+uy**2)[~solid].max()),
                mass_drift=float(abs(rho[~solid].sum()-torch.where(solid, torch.zeros_like(rho), torch.full_like(rho, 0)).sum())/1.0) if False else None)

import multiprocessing as mp
if __name__ == "__main__":
    fracs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    t0=time.time()
    with mp.get_context("fork").Pool(6) as pool:
        res = pool.map(run, fracs)
    for r in res:
        print("f=%.2f theta=%.2f+-%.2f umax=%.2e traj=%s" % (r['f'], r['theta'], r['unc'], r['umax'], ["%.1f"%t for _,t in r['traj']][::3]), flush=True)
    print("t", time.time()-t0)
    json.dump([{k:v for k,v in r.items()} for r in res], open('/root/tp-agent/work/m9874/.scratch/pilot_drop.json','w'))
