"""Actual local-transfer experiments; no physical qualification claim."""
import hashlib,json
from pathlib import Path
import torch
from tensorlbm.balanced_local_moving_boundary_2d import BalancedLocalMovingDisk
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/assets/balanced-local-moving-boundary'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cases=[('cross-grid',dict(velocity=(.02,0.)),120),('co-moving',dict(velocity=(.02,0.),fluid_velocity=(.02,0.)),120),('periodic-wrap',dict(center=(39.2,16.1),velocity=(.02,0.)),120),('paired-small-disk',dict(radius=1.5,center=(16.,16.),velocity=(.05,0.),fluid_velocity=(.05,0.)),120),('co-moving-half',dict(nx=80,ny=64,radius=8.,center=(32.4,32.2),velocity=(.01,0.),fluid_velocity=(.01,0.),tau=.8),480),('step-half',dict(nx=80,ny=64,radius=8.,center=(32.4,32.2),velocity=(.01,0.),tau=.8),480)]
    report=[]
    for name,config,steps in cases:
        sim=BalancedLocalMovingDisk(**config)
        initial=sim.snapshot();restart=None
        for k in range(steps):
            sim.step()
            if k==steps//2:restart=BalancedLocalMovingDisk.restore(json.loads(json.dumps(sim.snapshot())))
            elif restart is not None:restart.step()
        exact=torch.equal(sim.f,restart.f) and sim.snapshot()==restart.snapshot()
        record=dict(case=name,initial=initial,final=sim.snapshot(),restart_bitwise=exact,source_sha256=digest(ROOT/'src/tensorlbm/balanced_local_moving_boundary_2d.py'),base_source_sha256=digest(ROOT/'src/tensorlbm/moving_boundary_2d.py'))
        path=OUT/(name+'.json');path.write_text(json.dumps(record,indent=2)+'\n')
        hist=sim.history
        report.append(dict(case=name,artifact=path.name,sha256=digest(path),restart_bitwise=exact,covered=sum(h['covered'] for h in hist),exposed=sum(h['exposed'] for h in hist),mass_drift=sim.f.sum().item()-sim.target_mass,max_momentum_residual=max(h['momentum_residual'] for h in hist),minimum_population=min(h['minimum_population'] for h in hist),wall_impulse=[sum(h['wall_impulse_on_solid'][j]+h['conversion_impulse_on_solid'][j] for h in hist) for j in range(2)]))
    # Diffusive refinement: dx halves, dt quarters, lattice velocity halves;
    # mass in 2D SI scales dx^2 and velocity scales dx/dt, so impulse factor halves.
    coarse=report[0]['wall_impulse'][0];fine=.5*report[5]['wall_impulse'][0]
    result=dict(schema='balanced-local-moving-study/1',cases=report,refinement=dict(diffusive_scaling=True,relative_impulse_change=abs(fine-coarse)/max(abs(coarse),1e-30),threshold=.03,passed=abs(fine-coarse)/max(abs(coarse),1e-30)<.03),physical_accuracy_qualified=False,global_rescale=False,exact_swept_volume=False)
    result['co_moving_equilibrium']=dict(expected_impulse=[0.,0.],absolute_tolerance=1e-8,actual_impulse=report[1]['wall_impulse'],passed=max(map(abs,report[1]['wall_impulse']))<1e-8)
    (OUT/'study.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
