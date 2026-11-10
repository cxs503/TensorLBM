import sys, json, math, time
sys.path.insert(0,'/root/tp-agent/work/m9874/src')
import torch
torch.set_num_threads(4)
from tensorlbm.contact_line.engine import Channel, WallStepper, meniscus_state
from tensorlbm.contact_line import three_phase as tp
from tensorlbm.d2q9 import macroscopic

t0=time.time()
droplet = tp.free_droplet(n=160, radius=30.0, n_steps=8000)
print("Laplace gamma_lg=%.5f R=%.2f dp=%.5f rho_l=%.3f rho_v=%.3f level_spread=%.3f t=%.0fs" % (
    droplet['gamma'], droplet['radius'], droplet['dp'], droplet['rho_l'], droplet['rho_v'], droplet['gamma_level_spread'], time.time()-t0), flush=True)
g_lg = droplet['gamma']

f = 0.6
ch = Channel(nx=64, ny=64, wall_rows=2)
fdist, rho0 = meniscus_state(ch, fill_x=ch.nx/2)
st = WallStepper(ch, G=5.0, tau=1.0, wall_fraction=f, forcing="edm", dtype=torch.float64)
solid = ch.solid_mask()
psi_l = float(tp.psi_exp(torch.tensor(1.957, dtype=torch.float64)))
wall_psi = f*psi_l
fdist = fdist.to(torch.float64)
for n in range(1, 24001):
    fdist = st.step(fdist, 0.0)
    if n % 4000 == 0:
        rho = macroscopic(fdist)[0]
        wf = tp.wall_force(rho, solid_mask=solid, wall_psi=wall_psi)
        ux, uy = st.velocity(fdist)
        sp = float(torch.sqrt(ux**2+uy**2)[~solid].max())
        try:
            ang = tp.wedge_contact_angle(rho, y_wall=ch.y_wall(), solid=solid, level=0.5*(droplet['rho_l']+droplet['rho_v']), branch='max')
            th = ang['theta_deg']; unc = ang['unc_deg']
        except Exception as e:
            th = float('nan'); unc=float('nan')
        dg = wf['fx_per_length']/2.0
        print("n=%d umax=%.2e fx/len=%.5f dgamma=%.5f theta=%.2f+-%.2f cos=%.3f young_cos=%.3f t=%.0f" % (
            n, sp, wf['fx_per_length'], dg, th, unc, math.cos(math.radians(th)) if th==th else float('nan'), dg/g_lg, time.time()-t0), flush=True)
