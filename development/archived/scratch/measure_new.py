import sys, time
import numpy as np, torch
sys.path.insert(0,'/srv/tensor/agent/work/m21247/src')
from tensorlbm.spectral_saffman_taylor import SpectralSaffmanTaylor2D
torch.set_num_threads(8)
NY,NX,ASPECT,PE,MODE,W,DT=128,256,2.0,500.0,4,4.0,0.1
def measure(R, elliptic, ny=NY, nx=NX, dt=DT, t_fit=(2.0,8.0), warm=True):
    sim=SpectralSaffmanTaylor2D(ny,nx,aspect=ASPECT,pe=PE,viscosity_ratio=R,dt=dt,mode_index=MODE,elliptic=elliptic)
    sim.set_concentration(sim.single_mode_ic(x0=0.5*ASPECT*PE,width=W,amplitude=1e-3,n_mode=MODE))
    ts,amps=[],[]
    t0=time.time()
    for _ in range(int(round(t_fit[1]/dt))+1):
        d=sim.diagnostics(); ts.append(d.t); amps.append(d.mode_amplitude); sim.step()
    ts,amps=np.array(ts),np.array(amps)
    m=(ts>=t_fit[0])&(ts<=t_fit[1])&(amps>0)
    return float(np.polyfit(ts[m],np.log(amps[m]),1)[0]), time.time()-t0, sim.solver_stats
if __name__=="__main__":
    for R in (0.0,-1.0):
        s,el,st=measure(R,"exact_varcoef"); print(f"R={R:+.1f} sigma={s:+.6e} t={el:.1f}s stats={st}")
