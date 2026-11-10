import sys, time
sys.path.insert(0,'/root/tp-agent/work/m9469/src')
import torch; torch.set_num_threads(1)
from tensorlbm.taylor_aris import DispersionConfig, simulate_channel_dispersion
U=0.09
for pe, ny, nx, steps in [(40.0,33,256,14653),(40.0,9,256,4000),(40.0,9,128,4000)]:
    D=U*ny/pe; t0=time.time()
    out=simulate_channel_dispersion(DispersionConfig(nx=nx,ny=ny,diffusivity=D,u_mean=U,steps=steps,method="mode",dtype="float64",report_every=max(1,steps//200)))
    ratio=out["d_eff"]/out["d_molecular"]; closed=1.0+pe**2/210.0
    rr = closed*(2*3.141592653589793/nx*0.5*ny)**2/3.141592653589793**2
    print(f"pe={pe:g} ny={ny} nx={nx} D={D:.4f} ratio={ratio:.5f} closed={closed:.5f} err={100*(ratio/closed-1):+.3f}% rate_ratio={rr:.3f} r2={out['r2']:.4f} t={time.time()-t0:.1f}s", flush=True)
