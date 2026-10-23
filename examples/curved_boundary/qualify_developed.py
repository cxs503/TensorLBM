"""Qualify developed records against force drift AND net fluid acceleration.

Separate postprocessing preserves the executed integrator source; raw step-force
records reconstruct window impulses. It does not alter physics or extend runs.
"""
import math,json,hashlib
from pathlib import Path
import torch
from tensorlbm.d2q9 import C
from tensorlbm.curved_boundary_2d import FixedCircle
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/assets/curved-boundary'
def accounting(d):
    dt=d['dt_s'];dx=d['dx_m'];b=FixedCircle(d['n'],dx,d['center_m'],d['radius_m'])
    fluidmass=d['rho_kg_m3']*dx**2*d['thickness_m']*int((~b.solid).sum())
    steps=round(d['duration_s']/dt);count=round(d['steady_window_s']/dt)
    force=d['force_history_N'];assert len(force)==steps
    drives=[fluidmass*d['acceleration_m_s2']*math.sin(min((k+.5)*dt/d['ramp_s'],1)*math.pi/2)**2*dt for k in range(steps)]
    start=sum(drives[:-count])-sum(force[:-count])*dt
    end=sum(drives)-sum(force)*dt
    driverate=sum(drives[-count:])/d['steady_window_s'];reaction=sum(force[-count:])*dt/d['steady_window_s']
    imbalance=abs(reaction-driverate)/abs(driverate)
    f=torch.tensor(d['final_population'],dtype=torch.float64);actual=d['rho_kg_m3']*dx**3*d['thickness_m']/dt*torch.einsum('ihw,ij->j',f,C.double())
    assert abs(float(actual[0])-end)<1e-8
    return dict(mean_drive_force_N=driverate,mean_reaction_force_N=reaction,fluid_momentum_window_start_x_kg_m_s=start,fluid_momentum_window_end_x_kg_m_s=end,fluid_momentum_rate_x_N=(end-start)/d['steady_window_s'],reaction_drive_relative_imbalance=imbalance,net_acceleration_passed=imbalance<.01,window_drift_passed=d['late_window_drift']<.01,steady_qualified=(d['late_window_drift']<.01 and imbalance<.01))
if __name__=='__main__':
    m=json.loads((OUT/'developed-study.json').read_text())
    for case in m['cases']:
        path=OUT/case['file'];d=json.loads(path.read_text());result=accounting(d);d.update(result);d['first_window_drift_pass_time_s']=d.pop('first_steady_gate_pass_time_s',d.get('first_window_drift_pass_time_s'));case['first_window_drift_pass_time_s']=case.pop('first_steady_gate_pass_time_s',case.get('first_window_drift_pass_time_s'));path.write_text(json.dumps(d));case.update(result);case['sha256']=hashlib.sha256(path.read_bytes()).hexdigest();print(json.dumps(dict(n=d['n'],dt_s=d['dt_s'],**result)))
    source='examples/curved_boundary/qualify_developed.py';m['source_sha256'][source]=hashlib.sha256((ROOT/source).read_bytes()).hexdigest();(OUT/'developed-study.json').write_text(json.dumps(m,indent=2))
