"""Long developed fixture: unchanged smooth drive, explicit fixed steady gate."""
from pathlib import Path
import hashlib,json,math,argparse
import torch
from tensorlbm.curved_boundary_2d import FixedCircle
from tensorlbm.d2q9 import C
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/assets/curved-boundary'
SOURCES=['examples/curved_boundary/developed.py','src/tensorlbm/curved_boundary_2d.py','src/tensorlbm/solver.py','src/tensorlbm/d2q9.py']
def run(n,dt,duration):
    dx=1/n;steps=round(duration/dt);b=FixedCircle(n,dx,(.503,.497),.125)
    mass=1000*dx*dx*.2;scale=mass*dx/dt
    w=torch.tensor([4/9,*([1/9]*4),*([1/36]*4)],dtype=torch.float64)
    f=w[:,None,None].expand(9,n,n).clone();f[:,b.solid]=0;initial=f.clone();initial_sum=float(f.sum())
    drive=torch.zeros(2,dtype=torch.float64);impulse=drive.clone();history=[];windows=[];first_pass=None
    maxmass=maxmoment=0.;minpop=float('inf');repair_total=0.;window_sum=0.;window_steps=round(.5/dt)
    fixedkick=(3*w[:,None,None]*C[:,0,None,None]*.02*dt**2/dx).expand_as(f).clone();fixedkick[:,b.solid]=0
    fixeddrive=scale*torch.einsum('ihw,ij->j',fixedkick,C.double())
    for k in range(steps):
        t=(k+.5)*dt;ramp=math.sin(min(t/.25,1)*math.pi/2)**2
        kick=fixedkick*ramp
        before=f.clone() if k==steps-1 else None
        f+=kick;drive+=fixeddrive*ramp
        f,reaction,info=b.step(f,.5+3*.02*dt/dx**2)
        impulse+=reaction*scale;p=scale*torch.einsum('ihw,ij->j',f,C.double())
        masserr=abs(float(f.sum())/initial_sum-1);momenterr=float((p+impulse-drive).abs().max());force=float(reaction[0]*scale/dt)
        maxmass=max(maxmass,masserr);maxmoment=max(maxmoment,momenterr);minpop=min(minpop,info['minimum_fluid_population']);repair_total+=info['rest_mass_repair_l1'];window_sum+=force
        if (k+1)%100==0 or k==steps-1:
            history.append(dict(time_s=(k+1)*dt,force_x_N=force,mass_relative_error=masserr,momentum_error_kg_m_s=momenterr,**info))
        if (k+1)%window_steps==0:
            average=window_sum/window_steps;window_sum=0.
            drift=None if not windows else abs(average/windows[-1]['mean_force_N']-1)
            passed=drift is not None and drift<.01
            if passed and first_pass is None:first_pass=(k+1)*dt
            windows.append(dict(end_time_s=(k+1)*dt,mean_force_N=average,drift=drift,passed=passed))
    return dict(schema='tensorlbm.curved-developed-circle/1',n=n,dx_m=dx,dt_s=dt,duration_s=duration,ramp_s=.25,acceleration_m_s2=.02,nu_m2_s=.02,rho_kg_m3=1000,thickness_m=.2,center_m=[.503,.497],radius_m=.125,history_sample_every_steps=100,steady_window_s=.5,steady_drift_threshold=.01,initial_population=initial.tolist(),before_last_population=before.tolist(),final_population=f.tolist(),solid_impulse_Ns=impulse.tolist(),drive_impulse_Ns=drive.tolist(),history=history,windows=windows,first_steady_gate_pass_time_s=first_pass,late_mean_force_N=windows[-1]['mean_force_N'],late_window_drift=windows[-1]['drift'],steady_qualified=windows[-1]['passed'],max_mass_error=maxmass,max_momentum_error=maxmoment,minimum_population=minpop,repair_l1_accumulated=repair_total,physical_accuracy_qualified=False)
def save(n,dt,duration):
    data=run(n,dt,duration);name=f'developed-n{n}-dt{dt}-t{duration}.json';path=OUT/name;path.write_text(json.dumps(data))
    summary={k:data[k] for k in ['n','dt_s','duration_s','late_mean_force_N','late_window_drift','first_steady_gate_pass_time_s','steady_qualified','max_mass_error','max_momentum_error','minimum_population']};summary.update(file=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest());print(json.dumps(summary),flush=True);return summary
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--probe',action='store_true');parser.add_argument('--duration',type=float,default=10.);args=parser.parse_args()
    sequence=[(48,.0005)] if args.probe else [(32,.0005),(48,.0005),(64,.0005),(64,.00025)]
    cases=[save(n,dt,args.duration) for n,dt in sequence]
    manifest=dict(cases=cases,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCES},physical_accuracy_qualified=False)
    if len(cases)==4:manifest.update(space_change=abs(cases[2]['late_mean_force_N']/cases[1]['late_mean_force_N']-1),dt_eos_change=abs(cases[3]['late_mean_force_N']/cases[2]['late_mean_force_N']-1))
    (OUT/('developed-probe.json' if args.probe else 'developed-study.json')).write_text(json.dumps(manifest,indent=2))
