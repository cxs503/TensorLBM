import sys; sys.path.insert(0,'src')
import torch
from tensorlbm.elbm import *
from tensorlbm.d3q27 import macroscopic27, W_EXACT64 as W27

torch.manual_seed(0)
dt = torch.float64
nz,ny,nx = 2,3,4
rho = 1.0 + 0.05*torch.randn(nz,ny,nx,dtype=dt)
ux = 0.1*torch.randn(nz,ny,nx,dtype=dt)
uy = 0.1*torch.randn(nz,ny,nx,dtype=dt)
uz = 0.1*torch.randn(nz,ny,nx,dtype=dt)
feq = entropic_equilibrium_gauss_d3q27(rho,ux,uy,uz)
r,u1,u2,u3 = macroscopic27(feq)
print("closed-form density err", float((r-rho).abs().max()))
print("momentum err", float((u1-ux).abs().max()), float((u2-uy).abs().max()), float((u3-uz).abs().max()))
print("min pop", float(feq.min()))
feq2 = entropic_equilibrium_dual(rho,ux,uy,uz,lattice="d3q27",entropy="gauss",n_newton=40,tol=1e-15)
print("closed-form vs dual rel err", float((feq-feq2).abs().max()/feq.abs().max()))
print("dual moment err", float((feq2.sum(0)-rho).abs().max()))
H = gauss_entropy(feq, W27.view(27,1,1,1)); H2 = gauss_entropy(feq2, W27.view(27,1,1,1))
print("H closed - H dual", float((H-H2).max()), float((H2-H).max()))
fbe = entropic_equilibrium_dual(rho,ux,uy,uz,lattice="d3q27",entropy="bose-einstein",n_newton=80,tol=1e-15)
rb,a,b,cc = macroscopic27(fbe)
print("BE moment err", float((rb-rho).abs().max()), float((a-ux).abs().max()))
Hbe = bose_einstein_entropy(fbe, W27.view(27,1,1,1)); Hbe_g = bose_einstein_entropy(feq, W27.view(27,1,1,1))
print("H_BE(be)<=H_BE(gauss)?", float((Hbe-Hbe_g).max()))
# H-theorem on distorted states
f0 = entropic_equilibrium_gauss_d3q27(torch.ones_like(rho), 0.2*torch.randn_like(rho), 0.2*torch.randn_like(rho), 0.2*torch.randn_like(rho))
r0,_,_,_ = macroscopic27(f0)
f = torch.clamp(f0*(1+0.25*torch.randn_like(f0)), min=1e-8)
tau = 0.51
fs, info = collide_elbm_d3q27(f, tau, return_info=True)
print("alpha summary", info.summary())
Hf = gauss_entropy(f, W27.view(27,1,1,1)); Hfs = gauss_entropy(fs, W27.view(27,1,1,1))
print("H increase max", float((Hfs-Hf).max()), "min", float((Hfs-Hf).min()))
print("min post-collision pop", float(fs.min()))
