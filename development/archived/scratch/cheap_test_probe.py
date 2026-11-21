import sys, time
import numpy as np, torch
sys.path.insert(0,'/srv/tensor/agent/work/m21247/src')
sys.path.insert(0,'/srv/tensor/agent/work/m21247/xval/XVAL-LBM-SPECTRAL-SAFFMAN-TAYLOR-INTEGRATING-FACT')
from tensorlbm.spectral_saffman_taylor import SpectralSaffmanTaylor2D
from xval_linear_theory import exact_linear_sigma
torch.set_num_threads(4)
def meas(R, ny, nx, mode, tfit=(2.0,5.0), dt=0.1, el="exact_varcoef"):
    k=2*np.pi*mode/500.0
    s=SpectralSaffmanTaylor2D(ny,nx,aspect=2.0,pe=500.0,viscosity_ratio=R,dt=dt,mode_index=mode,elliptic=el)
    s.set_concentration(s.single_mode_ic(x0=500.0,width=4.0,amplitude=1e-3,n_mode=mode))
    ts,amps=[],[]
    t0=time.time()
    for _ in range(int(tfit[1]/dt)+1):
        d=s.diagnostics(); ts.append(d.t); amps.append(d.mode_amplitude); s.step()
    ts,amps=np.array(ts),np.array(amps); m=(ts>=tfit[0])
    sig=float(np.polyfit(ts[m],np.log(amps[m]),1)[0])
    th=exact_linear_sigma(k,R,4.0)["sigma"].real
    return sig, th, time.time()-t0
for R in (2.0,-2.0,1.0,-1.0):
    sig,th,el=meas(R,64,128,2)
    print(f"R={R:+.0f} ny64 nx128 mode2: sigma={sig:+.5e} theory={th:+.5e} rel={abs(sig-th)/abs(th)*100:.1f}% t={el:.1f}s")
