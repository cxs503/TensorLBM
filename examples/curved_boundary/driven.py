"""Smoothly driven zero-velocity startup; fixed SI force density and late windows."""
from pathlib import Path
import hashlib,json,math
import torch
from tensorlbm.curved_boundary_2d import FixedCircle
from tensorlbm.d2q9 import C
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/assets/curved-boundary'
def run(n,dt):
    dx=1/n;duration=2.;steps=round(duration/dt);b=FixedCircle(n,dx,(.503,.497),.125)
    mass=1000*dx*dx*.2;scale=mass*dx/dt
    w=torch.tensor([4/9,*([1/9]*4),*([1/36]*4)],dtype=torch.float64)
    f=w[:,None,None].expand(9,n,n).clone();f[:,b.solid]=0;initial=f.clone()
    drive=torch.zeros(2,dtype=torch.float64);impulse=drive.clone();history=[]
    for k in range(steps):
        t=(k+.5)*dt;a=.02*math.sin(min(t/.25,1)*math.pi/2)**2
        kick=3*w[:,None,None]*C[:,0,None,None]*a*dt**2/dx
        kick=kick.expand_as(f).clone();kick[:,b.solid]=0
        before=f.clone();f+=kick
        drive+=scale*torch.einsum('ihw,ij->j',kick,C.double())
        f,reaction,info=b.step(f,.5+3*.02*dt/dx**2)
        impulse+=reaction*scale;p=scale*torch.einsum('ihw,ij->j',f,C.double())
        history.append(dict(time_s=(k+1)*dt,force_x_N=float(reaction[0]*scale/dt),mass_relative_error=abs(float(f.sum()/initial.sum())-1),momentum_error_kg_m_s=float((p+impulse-drive).abs().max()),**info))
    early=[h['force_x_N'] for h in history if 1.5<h['time_s']<=1.75];late=[h['force_x_N'] for h in history if h['time_s']>1.75]
    average=sum(late)/len(late);drift=abs(average/(sum(early)/len(early))-1)
    return dict(schema='tensorlbm.curved-driven-circle/1',n=n,dx_m=dx,dt_s=dt,duration_s=duration,ramp_s=.25,acceleration_m_s2=.02,nu_m2_s=.02,rho_kg_m3=1000,thickness_m=.2,center_m=[.503,.497],radius_m=.125,initial_population=initial.tolist(),before_last_population=before.tolist(),final_population=f.tolist(),solid_impulse_Ns=impulse.tolist(),drive_impulse_Ns=drive.tolist(),history=history,late_mean_force_N=average,late_window_drift=drift,steady_qualified=drift<.01,physical_accuracy_qualified=False)
if __name__=='__main__':
    cases=[]
    for n,dt in [(32,.0005),(48,.0005),(64,.0005),(64,.00025)]:
        data=run(n,dt);name=f'driven-n{n}-dt{dt}.json';path=OUT/name;path.write_text(json.dumps(data));h=data['history']
        cases.append(dict(file=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),n=n,dt_s=dt,late_mean_force_N=data['late_mean_force_N'],late_window_drift=data['late_window_drift'],steady_qualified=data['steady_qualified'],max_mass_error=max(x['mass_relative_error'] for x in h),max_momentum_error=max(x['momentum_error_kg_m_s'] for x in h),minimum_population=min(x['minimum_fluid_population'] for x in h)))
    manifest=dict(cases=cases,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in ['examples/curved_boundary/driven.py','src/tensorlbm/curved_boundary_2d.py','src/tensorlbm/solver.py','src/tensorlbm/d2q9.py']},space_change=abs(cases[2]['late_mean_force_N']/cases[1]['late_mean_force_N']-1),dt_eos_change=abs(cases[3]['late_mean_force_N']/cases[2]['late_mean_force_N']-1),physical_accuracy_qualified=False)
    (OUT/'driven-study.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
