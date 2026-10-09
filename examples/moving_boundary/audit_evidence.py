"""Reconstruct published populations, momentum, mass and step ledger."""
import hashlib
import json
from pathlib import Path
import torch
from tensorlbm.moving_boundary_2d import MovingDisk, momentum

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'examples/moving_boundary/evidence'

def main():
    study=json.loads((OUT/'study.json').read_text())
    assert study['source_sha256']==hashlib.sha256((ROOT/'src/tensorlbm/moving_boundary_2d.py').read_bytes()).hexdigest()
    results=[]
    for case in study['cases']:
        path=OUT/(case['name']+'.json')
        assert case['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
        raw=json.loads(path.read_text());sim=MovingDisk.restore(raw['state'])
        assert sim.time==case['steps']
        assert torch.equal(sim.solid,sim.mask())
        assert abs(float(sim.f.sum())-sim.target_mass)<1e-9
        last=sim.history[-1]
        assert torch.allclose(momentum(sim.f),torch.tensor(last['momentum'],dtype=torch.float64),atol=1e-12,rtol=0)
        old=MovingDisk(**sim.config);old_mass=old.target_mass;old_p=momentum(old.f)
        for h in sim.history:
            new_p=torch.tensor(h['momentum'],dtype=torch.float64)
            wall=torch.tensor(h['wall_impulse_on_solid'],dtype=torch.float64)
            conv=torch.tensor(h['conversion_impulse_on_solid'],dtype=torch.float64)
            reservoir=torch.tensor(h['reservoir_momentum'],dtype=torch.float64)
            assert float((new_p-old_p+wall+conv-reservoir).abs().max())<1e-10
            assert abs(h['mass']-old_mass-h['boundary_mass_change']-h['added_mass']+h['removed_mass']-h['reservoir_mass'])<1e-9
            old_p=new_p;old_mass=h['mass']
        # Actual replay checks the published raw field, not only its checksums.
        replay=MovingDisk(**sim.config)
        for _ in range(sim.time):replay.step()
        assert torch.equal(replay.f,sim.f)
        results.append(dict(name=case['name'],raw_field_and_ledger_passed=True))
    base,half,quarter=[next(x for x in study['cases'] if x['name']==n) for n in ('moving','time-half','time-quarter')]
    difference=abs(quarter['solid_impulse_si_Ns'][0]-half['solid_impulse_si_Ns'][0])/abs(quarter['solid_impulse_si_Ns'][0])*100
    report=dict(cases=results,time_half_to_quarter_impulse_change_percent=difference,declared_threshold_percent=3.,temporal_convergence_qualified=difference<=3.,physical_accuracy_qualified=False)
    (OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
