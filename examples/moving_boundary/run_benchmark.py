"""Execute moving-mask BGK cases; publish lattice-unit raw states and ledger."""
import hashlib
import json
from pathlib import Path
from tensorlbm.moving_boundary_2d import MovingDisk

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'examples/moving_boundary/evidence'

def main():
    OUT.mkdir(exist_ok=True)
    reports=[]
    # Same displacement but different lattice velocity and duration. At fixed
    # tau this changes Re and diffusion time: motion-rate sensitivity, NOT a
    # same-physical-problem time refinement claim.
    for name,speed,steps,comoving,tau,dt in [('comoving',.03,100,True,.8,.01),('moving',.03,100,False,.8,.01),('motion-half',.015,200,False,.8,.01),('motion-quarter',.0075,400,False,.8,.01),('time-half',.015,200,False,.65,.005),('time-quarter',.0075,400,False,.575,.0025)]:
        sim=MovingDisk(velocity=(speed,0),fluid_velocity=(speed if comoving else 0,0),tau=tau)
        for _ in range(steps):sim.step()
        import torch
        resumed=MovingDisk.restore(json.loads(json.dumps(sim.snapshot())))
        reference=MovingDisk.restore(json.loads(json.dumps(sim.snapshot())))
        for _ in range(5):reference.step();resumed.step()
        h=sim.history
        report=dict(name=name,steps=steps,physical_accuracy_qualified=False,
            si=dict(dx_m=.01,dt_s=dt,rho_kg_m3=1000.,thickness_m=.01,nu_m2_s=(tau-.5)/3*.01**2/dt,velocity_m_s=speed*.01/dt,duration_s=steps*dt),
            restart_bitwise=torch.equal(reference.f,resumed.f),
            max_momentum_residual=max(x['momentum_residual'] for x in h),
            max_mass_residual=max(abs(x['mass_residual']) for x in h),
            covered=sum(x['covered'] for x in h),exposed=sum(x['exposed'] for x in h),
            cumulative_reservoir_mass=sum(x['reservoir_mass'] for x in h),
            total_absolute_reservoir_mass=sum(abs(x['reservoir_mass']) for x in h),
            solid_impulse=[sum(x['wall_impulse_on_solid'][j]+x['conversion_impulse_on_solid'][j] for x in h) for j in range(2)])
        report['solid_impulse_si_Ns']=[x*1000*.01**4/dt for x in report['solid_impulse']]
        path=OUT/(name+'.json');path.write_text(json.dumps(dict(report=report,state=sim.snapshot()),indent=2)+'\n')
        report['sha256']=hashlib.sha256(path.read_bytes()).hexdigest();reports.append(report)
    summary=dict(schema='tensorlbm.moving-disk-study/1',units='lattice units: dx=dt=rho0=1',
        algorithm='moving halfway bounce-back; adjacent density wall-speed refill; global mass reservoir',
        physical_accuracy_qualified=False,temporal_convergence_qualified=False,
        rate_sensitivity_changes_reynolds_number=True,
        source_sha256=hashlib.sha256((ROOT/'src/tensorlbm/moving_boundary_2d.py').read_bytes()).hexdigest(),cases=reports)
    (OUT/'study.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
