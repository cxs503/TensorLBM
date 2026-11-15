import sys, torch
sys.path.insert(0,'src')
torch.set_num_threads(6)
from tensorlbm.phasefield.operators import laplacian_3d
from tensorlbm.allen_cahn_lbm import free_energy_parameters, step_allen_cahn_field
n,R,W=32,8,4
z,y,x=torch.meshgrid(torch.arange(n),torch.arange(n),torch.arange(n),indexing='ij')
r=torch.sqrt((x-n/2+0.5)**2+(y-n/2+0.5)**2+(z-n/2+0.5)**2)
phi=torch.tanh((R-r)/W).double()
p=free_energy_parameters(1e-2,W)
print("beta",p.beta,"kappa",p.kappa)
mu=p.beta*phi*(phi*phi-1)-p.kappa*laplacian_3d(phi,boundary='periodic')
print("step0 center phi",float(phi[0,16,16]),"max|mu|",float(mu.abs().max()),"mu@center",float(mu[0,16,16]),"mu@R",float(mu[0,16,24]))
for s in range(10):
    phi=step_allen_cahn_field(phi,beta=p.beta,kappa=p.kappa,mobility=5e-3,u=None,boundary='periodic')
    print(f"AC-only s={s+1} center={float(phi[0,16,16]):.5f} max|phi|={float(phi.abs().max()):.5f}")
