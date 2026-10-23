"""Fixed finite-size disk: half-way bounce-back and excluded interior liquid."""
from pathlib import Path
import json,hashlib
import torch
from tensorlbm.d2q9 import C,OPPOSITE,macroscopic
from tensorlbm.solver import collide_bgk_matmul
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/assets/perimeter-boundary'

from tensorlbm.fixed_disk_boundary_2d import bounce_step


def run(n):
    dx=1/n;dt=.0005*(32/n)**2;steps=round(.025/dt);mass=1000*dx**2*.2
    y,x=torch.meshgrid(torch.arange(n)*dx,torch.arange(n)*dx,indexing='ij')
    solid=(x-.5)**2+(y-.5)**2<.125**2
    w=torch.tensor([4/9,*([1/9]*4),*([1/36]*4)],dtype=torch.float64)
    velocity=torch.tensor([.02,0.],dtype=torch.float64)*dt/dx;cu=C.double()@velocity
    f=(w*(1+3*cu+4.5*cu**2-1.5*(velocity**2).sum()))[:,None,None].expand(9,n,n).clone();f[:,solid]=0
    initial=f.clone();p0=mass*dx/dt*torch.einsum('ihw,ij->j',f,C.double())
    impulse=torch.zeros(2,dtype=torch.float64);history=[]
    for k in range(steps):
        before=f.clone();f,reaction=bounce_step(f,solid,.5+3*.02*dt/dx**2)
        impulse+=reaction*mass*dx/dt
        P=mass*dx/dt*torch.einsum('ihw,ij->j',f,C.double())
        history.append({'time_s':(k+1)*dt,'mass_relative_error':abs(float(f.sum()/initial.sum())-1),
          'momentum_error_kg_m_s':float((P-p0+impulse).abs().max()),'solid_population_max':float(f[:,solid].abs().max())})
    return {'schema':'tensorlbm.fixed-link/1','n':n,'dx_m':dx,'dt_s':dt,'steps':steps,
      'mask':solid.tolist(),'initial_population':initial.tolist(),'before_last_population':before.tolist(),
      'final_population':f.tolist(),'solid_impulse_Ns':impulse.tolist(),'history':history,
      'gates':{'mass':max(h['mass_relative_error'] for h in history)<1e-11,
       'momentum':max(h['momentum_error_kg_m_s'] for h in history)<1e-10,
       'excluded_interior':all(h['solid_population_max']==0 for h in history)},
      'physical_accuracy_qualified':False,'moving_boundary_qualified':False}

if __name__=='__main__':
    reports=[]
    for n in (32,64):
        data=run(n);name=f'fixed-link-n{n}.json';(OUT/name).write_text(json.dumps(data))
        reports.append({'file':name,'sha256':hashlib.sha256((OUT/name).read_bytes()).hexdigest(),'gates':data['gates'],
          'max_mass_error':max(h['mass_relative_error'] for h in data['history']),
          'max_momentum_error':max(h['momentum_error_kg_m_s'] for h in data['history'])})
    manifest={'cases':reports,'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['examples/perimeter_boundary/fixed_link.py','src/tensorlbm/fixed_disk_boundary_2d.py','src/tensorlbm/solver.py','src/tensorlbm/d2q9.py']}}
    (OUT/'fixed-link-study.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(reports,indent=2))
