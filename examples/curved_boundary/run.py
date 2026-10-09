"""Actual transient fixed-circle study, SI fixed dt across spatial grids."""
from pathlib import Path
import hashlib,json
import torch
from tensorlbm.curved_boundary_2d import FixedCircle
from tensorlbm.d2q9 import C
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/assets/curved-boundary'
SOURCES=['src/tensorlbm/curved_boundary_2d.py','src/tensorlbm/solver.py','src/tensorlbm/d2q9.py','examples/curved_boundary/run.py']
def run(n,dt,interpolated):
    dx=1/n;steps=round(.15/dt);boundary=FixedCircle(n,dx,(.503,.497),.125)
    mass=1000*dx*dx*.2;scale=mass*dx/dt
    w=torch.tensor([4/9,*([1/9]*4),*([1/36]*4)],dtype=torch.float64)
    u=torch.tensor([.02,0.],dtype=torch.float64)*dt/dx;cu=C.double()@u
    f=(w*(1+3*cu+4.5*cu**2-1.5*u.square().sum()))[:,None,None].expand(9,n,n).clone();f[:,boundary.solid]=0
    initial=f.clone();p0=scale*torch.einsum('ihw,ij->j',f,C.double());impulse=torch.zeros(2,dtype=torch.float64);history=[]
    for k in range(steps):
        before=f.clone();f,reaction,info=boundary.step(f,.5+3*.02*dt/dx**2,interpolated)
        impulse+=reaction*scale;p=scale*torch.einsum('ihw,ij->j',f,C.double())
        history.append(dict(time_s=(k+1)*dt,force_N=(reaction*scale/dt).tolist(),mass_relative_error=abs(float(f.sum()/initial.sum())-1),momentum_error_kg_m_s=float((p-p0+impulse).abs().max()),**info))
    return dict(schema='tensorlbm.curved-fixed-circle/1',n=n,dx_m=dx,dt_s=dt,duration_s=.15,rho_kg_m3=1000,thickness_m=.2,viscosity_m2_s=.02,initial_velocity_m_s=[.02,0],center_m=[.503,.497],radius_m=.125,interpolated=interpolated,initial_population=initial.tolist(),before_last_population=before.tolist(),final_population=f.tolist(),solid_impulse_Ns=impulse.tolist(),history=history,physical_accuracy_qualified=False,moving_boundary_qualified=False)
if __name__=='__main__':
    cases=[]
    for n,dt,kind in [(32,.0005,True),(48,.0005,True),(64,.0005,True),(64,.00025,True),(32,.0005,False),(48,.0005,False),(64,.0005,False)]:
        data=run(n,dt,kind);name=f'{"bfl" if kind else "staircase"}-n{n}-dt{dt}.json';path=OUT/name;path.write_text(json.dumps(data))
        h=data['history'];cases.append(dict(file=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),n=n,dt_s=dt,interpolated=kind,impulse_x_Ns=data['solid_impulse_Ns'][0],max_mass_error=max(x['mass_relative_error'] for x in h),max_momentum_error=max(x['momentum_error_kg_m_s'] for x in h),minimum_population=min(x['minimum_fluid_population'] for x in h),repair_l1_accumulated=sum(x['rest_mass_repair_l1'] for x in h)))
    b=cases[:4];manifest=dict(cases=cases,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCES},spatial_last_pair_relative_change=abs(b[2]['impulse_x_Ns']/b[1]['impulse_x_Ns']-1),temporal_half_relative_change=abs(b[3]['impulse_x_Ns']/b[2]['impulse_x_Ns']-1),physical_accuracy_qualified=False)
    (OUT/'study.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
