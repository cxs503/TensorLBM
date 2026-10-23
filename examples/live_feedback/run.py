"""Actual variable-motion fluid/rigid translation fixture; no contact or ice."""
import hashlib
import json
from pathlib import Path
import torch
from tensorlbm.feedback_moving_disk import MovingDiskFeedback
from tensorlbm.moving_boundary_2d import C

ROOT=Path(__file__).resolve().parents[2]

def fluid_ke(f, config):
    rho=f.sum(0); p=torch.einsum('iyx,ia->ayx',f,C.double())
    return float((.5*p.square().sum(0)/rho.clamp_min(1e-30)).sum())*config['density_kg_m3']*config['thickness_m']*config['dx_m']**4/config['dt_s']**2

def run(steps=160):
    fluid=MovingDiskFeedback(thickness_m=.2,center_m=(.16,.16)); mass=5.; velocity=[0.,.03]
    initial_p=torch.tensor(velocity,dtype=torch.float64)*mass; initial_ke=.5*mass*sum(v*v for v in velocity)
    reservoir=torch.zeros(2,dtype=torch.float64); records=[]; restarted=None
    for i in range(steps):
        if i==steps//2: restarted=MovingDiskFeedback.restore(json.loads(json.dumps(fluid.snapshot())))
        entry=fluid.advance(time_s=fluid.time_s,center_m=fluid.center_m,velocity_m_s=velocity)
        if restarted is not None:
            assert entry==restarted.advance(time_s=restarted.time_s,center_m=restarted.center_m,velocity_m_s=velocity)
            assert torch.equal(fluid.solver.f,restarted.solver.f)
        old_ke=.5*mass*sum(v*v for v in velocity)
        velocity=[v+j/mass for v,j in zip(velocity,entry['impulse_on_body_Ns'])]
        reservoir+=torch.tensor(entry['reservoir_impulse_on_fluid_Ns'],dtype=torch.float64)
        body_ke=.5*mass*sum(v*v for v in velocity)
        residual=torch.tensor(fluid.fluid_momentum_Ns(),dtype=torch.float64)+torch.tensor(velocity,dtype=torch.float64)*mass-initial_p-reservoir
        records.append(dict(exchange=entry,body_velocity_m_s=velocity,body_kinetic_J=body_ke,
            body_kick_work_J=body_ke-old_ke,explicit_kick_defect_J=body_ke-old_ke-entry['body_interface_work_J'],
            fluid_kinetic_J=fluid_ke(fluid.solver.f,fluid.config),momentum_residual_Ns=float(residual.abs().max())))
    sources=['src/tensorlbm/feedback_moving_disk.py','src/tensorlbm/moving_boundary_2d.py','src/tensorlbm/solver.py','src/tensorlbm/d2q9.py']
    return dict(schema='tensorlbm.live-feedback-fixture/1',sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        body_mass_kg=mass,initial_body_velocity_m_s=[0.,.03],initial_body_kinetic_J=initial_ke,
        final_velocity_m_s=velocity,reservoir_impulse_on_fluid_Ns=reservoir.tolist(),history=records,
        final_fluid_snapshot=fluid.snapshot(),restart_bitwise=True,
        maximum_total_momentum_residual_Ns=max(r['momentum_residual_Ns'] for r in records),
        covered_events=sum(r['exchange']['covered'] for r in records),
        total_kinetic_change_J=records[-1]['body_kinetic_J']+records[-1]['fluid_kinetic_J']-initial_ke,
        energy_accuracy_qualified=False,physical_accuracy_qualified=False,
        limitations=['global reservoir is external','halfway voxel boundary','explicit staggered exchange','no gravity/free surface/ice/contact','energy change is exposed, not certified viscosity'])

if __name__=='__main__':
    out=ROOT/'examples/live_feedback/evidence.json'; result=run();out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['maximum_total_momentum_residual_Ns','covered_events','total_kinetic_change_J','restart_bitwise']}))
