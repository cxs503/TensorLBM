"""Raw reconstruction of smooth-drive experiment, including external impulse."""
import json,hashlib,math
from pathlib import Path
import torch
from tensorlbm.curved_boundary_2d import FixedCircle
from tensorlbm.d2q9 import C
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/assets/curved-boundary'
m=json.loads((OUT/'driven-study.json').read_text())
for s,h in m['source_sha256'].items():assert hashlib.sha256((ROOT/s).read_bytes()).hexdigest()==h
for case in m['cases']:
    path=OUT/case['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==case['sha256']
    d=json.loads(path.read_text());b=FixedCircle(d['n'],d['dx_m'],d['center_m'],d['radius_m']);dx=d['dx_m'];dt=d['dt_s']
    initial=torch.tensor(d['initial_population'],dtype=torch.float64);before=torch.tensor(d['before_last_population'],dtype=torch.float64);final=torch.tensor(d['final_population'],dtype=torch.float64)
    w=torch.tensor([4/9,*([1/9]*4),*([1/36]*4)],dtype=torch.float64)
    t=d['duration_s']-.5*dt;a=d['acceleration_m_s2']*math.sin(min(t/d['ramp_s'],1)*math.pi/2)**2
    kick=(3*w[:,None,None]*C[:,0,None,None]*a*dt**2/dx).expand_as(before).clone();kick[:,b.solid]=0
    actual,reaction,info=b.step(before+kick,.5+3*d['nu_m2_s']*dt/dx**2);assert torch.equal(actual,final)
    scale=d['rho_kg_m3']*dx**3*d['thickness_m']/dt
    p=scale*torch.einsum('ihw,ij->j',final-initial,C.double());j=torch.tensor(d['solid_impulse_Ns'],dtype=torch.float64);drive=torch.tensor(d['drive_impulse_Ns'],dtype=torch.float64)
    expected_drive=d['rho_kg_m3']*dx**2*d['thickness_m']*int((~b.solid).sum())*sum(d['acceleration_m_s2']*math.sin(min((k+.5)*dt/d['ramp_s'],1)*math.pi/2)**2*dt for k in range(round(d['duration_s']/dt)))
    assert abs(float(drive[0])-expected_drive)<1e-9 and float(drive[1])==0
    assert float((p+j-drive).abs().max())<1e-8
    assert abs(float(final.sum()/initial.sum())-1)<1e-11
    assert not final[:,b.solid].any();assert final[:,~b.solid].min()>0
    history=d['history'];early=[x['force_x_N'] for x in history if 1.5<x['time_s']<=1.75];late=[x['force_x_N'] for x in history if x['time_s']>1.75]
    average=sum(late)/len(late);drift=abs(average/(sum(early)/len(early))-1)
    assert average==d['late_mean_force_N']==case['late_mean_force_N'];assert drift==d['late_window_drift']==case['late_window_drift']
    assert (drift<.01)==d['steady_qualified']==case['steady_qualified']
    assert max(x['mass_relative_error'] for x in history)==case['max_mass_error']
    assert max(x['momentum_error_kg_m_s'] for x in history)==case['max_momentum_error']
    print(case['file'],'raw replay/external-impulse/mass/late-window PASS')
assert m['space_change']==abs(m['cases'][2]['late_mean_force_N']/m['cases'][1]['late_mean_force_N']-1)
assert m['dt_eos_change']==abs(m['cases'][3]['late_mean_force_N']/m['cases'][2]['late_mean_force_N']-1)
print('Physical accuracy remains UNQUALIFIED')
