"""Actual shared-face and diffuse-wall flux updates of uniform co-moving states."""
import gzip,hashlib,json
from pathlib import Path
import torch
from tensorlbm.swept_uniform_transport_2d import UniformSweptTransport
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/assets/swept-uniform-transport'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,torch.Tensor):return x.tolist()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    return x

def main():
    OUT.mkdir(parents=True,exist_ok=True);cases=[]
    configs=[('static',{},20,.1),('diagonal',{'velocity_m_s':(.08,.03)},20,.1),('reverse',{'velocity_m_s':(-.08,-.03)},20,.1),('time-half',{'velocity_m_s':(.08,.03)},40,.05)]
    for name,geom,steps,dt in configs:
        if name=='static':geom={'velocity_m_s':(0.,0.)}
        s=UniformSweptTransport(geometry=geom);initial=s.snapshot();records=[];restored=None
        for k in range(steps):
            if k==steps//2:restored=UniformSweptTransport.restore(json.loads(json.dumps(s.snapshot())))
            before=s.Q.tolist();interval=s.geometry.interval(s.time,dt);entry=s.step(dt)
            records.append({'geometry_interval':plain(interval),'Q_before_kg':before,'Q_after_kg':s.Q.tolist(),'entry':entry})
            if restored is not None:
                if entry!=restored.step(dt) or not torch.equal(s.Q,restored.Q):raise ValueError('uniform flux restart failed')
        raw={'schema':'tensorlbm.uniform-swept-case/1','initial':initial,'final':s.snapshot(),'steps':records,'restart_bitwise':s.snapshot()==restored.snapshot(),'physical_accuracy_qualified':False}
        path=OUT/(name+'.json.gz');path.write_bytes(gzip.compress((json.dumps(raw,separators=(',',':'),allow_nan=False)+'\n').encode(),mtime=0))
        cases.append({'case':name,'artifact':path.name,'sha256':digest(path),'restart_bitwise':raw['restart_bitwise'],
                      'birth_cells':sum(e['entry']['birth_cells'] for e in records),'death_cells':sum(e['entry']['death_cells'] for e in records),
                      'max_density_relative_error':max(e['entry']['maximum_density_error_kg_m3']/1000 for e in records),
                      'max_velocity_error_m_s':max(e['entry']['maximum_velocity_error_m_s'] for e in records),
                      'max_momentum_balance_residual_Ns':max(e['entry']['momentum_balance_residual_Ns'] for e in records),
                      'total_body_impulse_Ns':[sum(e['entry']['body_impulse_Ns'][j] for e in records) for j in range(2)]})
    paths=['src/tensorlbm/swept_uniform_transport_2d.py','src/tensorlbm/swept_rectangle_geometry_2d.py','src/tensorlbm/moving_boundary_2d.py','src/tensorlbm/d2q9.py','examples/swept_uniform_transport/run.py']
    study={'schema':'tensorlbm.uniform-swept-study/1','source_sha256':{p:digest(ROOT/p) for p in paths},'cases':cases,'physical_accuracy_qualified':False,'arbitrary_state_transport':False,'scope':'uniform co-moving equilibrium, shared-face FV flux plus diffuse moving-wall flux, no BGK or nonuniform solution'}
    (OUT/'study.json').write_text(json.dumps(study,indent=2)+'\n');print(json.dumps(study,indent=2))
if __name__=='__main__':main()
