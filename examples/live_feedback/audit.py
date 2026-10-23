import hashlib
import json
from pathlib import Path
import torch
from tensorlbm.feedback_moving_disk import MovingDiskFeedback
from run import fluid_ke

ROOT=Path(__file__).resolve().parents[2]
data=json.loads((Path(__file__).parent/'evidence.json').read_text())
for p,h in data['sources'].items():
    assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
f=MovingDiskFeedback.restore(data['final_fluid_snapshot']); mass=data['body_mass_kg']
initial=torch.tensor(data['initial_body_velocity_m_s'],dtype=torch.float64)*mass
reservoir=torch.zeros(2,dtype=torch.float64); body=initial.clone()
for r in data['history']:
    e=r['exchange'];body+=torch.tensor(e['impulse_on_body_Ns'],dtype=torch.float64)
    reservoir+=torch.tensor(e['reservoir_impulse_on_fluid_Ns'],dtype=torch.float64)
    assert abs(.5*mass*sum(v*v for v in r['body_velocity_m_s'])-r['body_kinetic_J'])<1e-12
assert torch.allclose(body/mass,torch.tensor(data['final_velocity_m_s'],dtype=torch.float64),atol=1e-14,rtol=0)
residual=torch.tensor(f.fluid_momentum_Ns(),dtype=torch.float64)+body-initial-reservoir
assert residual.abs().max()<1e-10
assert abs(fluid_ke(f.solver.f,f.config)-data['history'][-1]['fluid_kinetic_J'])<1e-12
assert data['physical_accuracy_qualified'] is False and data['energy_accuracy_qualified'] is False
print(json.dumps(dict(raw_field_audit_passed=True,final_momentum_residual_Ns=float(residual.abs().max()),covered_events=data['covered_events'])))
