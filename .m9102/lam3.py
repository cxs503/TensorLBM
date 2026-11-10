import sys; sys.path.insert(0,'src')
import numpy as np, torch
torch.set_num_threads(8)
from tensorlbm.elbm_channel import ElbmChannelConfig, run_elbm_channel, _initial_field
from tensorlbm.elbm import gauss_entropy, entropic_equilibrium_gauss_d3q27, solve_entropy_condition
from tensorlbm.d3q27 import W_EXACT64 as W27, macroscopic27
cfg = ElbmChannelConfig(nx=4, ny=21, nz=4, re_tau=5.0, u_tau=0.01, steps=300, warmup=250, sample_every=10, log_every=100, noise_amplitude=1e-8, dtype="float64")
res = run_elbm_channel(cfg, progress=True)
w = W27.view(27,1,1,1)
# reconstruct state at the end? instead: reproduce a sheared equilibrium state and study H(alpha)
f = _initial_field(cfg, torch.float64, torch.device("cpu"))
r,u1,u2,u3 = macroscopic27(f)
feq = entropic_equilibrium_gauss_d3q27(r,u1,u2,u3)
tau = cfg.tau
H0 = gauss_entropy(f,w)
for y in (1,2,5,10,15,19):
    fc = f[:,:,y,:].reshape(27,-1)
    fe = feq[:,:,y,:].reshape(27,-1)
    H0c = gauss_entropy(fc, w)
    line=[]
    for a in (0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.5,2.0):
        fs = fc + (a/tau)*(fe-fc)
        line.append(round(float((gauss_entropy(fs,w)-H0c).mean()),8))
    info = solve_entropy_condition(fc, fe, tau, w=w, n_iter=12, alpha_cap=1.0)
    print(f"y={y:2d} dH(alpha)={line} alpha_py={float(info.alpha.mean()):.4f} u={float(u1[0,y,0]):.5f}")
