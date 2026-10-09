"""Run geometry only; this does not initialize or advance fluid populations."""
import gzip,hashlib,json
from pathlib import Path
import torch
from tensorlbm.swept_rectangle_geometry_2d import SweptRectangleGeometry
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/assets/swept-rectangle-geometry'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def plain(value):
    if isinstance(value,torch.Tensor):return value.tolist()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    return value

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cases=[('diagonal',{},.08,20),('reverse',dict(velocity_m_s=(-.12,-.09)),.08,20),
           ('axial',dict(velocity_m_s=(.12,0.)),.08,20),
           ('static-aligned',dict(width_m=.4,height_m=.2,center_m=(.5,.5),velocity_m_s=(0.,0.)),.08,20),
           ('space-time-half',dict(nx=24,ny=20,dx_m=.05),.02,80)]
    summaries=[]
    for name,cfg,dt,count in cases:
        geo=SweptRectangleGeometry(**cfg);history=[];births=deaths=0
        for k in range(count):
            record=geo.interval(k*dt,dt);history.append(plain(record))
            va=record['start']['fluid_volume_m2'];vb=record['end']['fluid_volume_m2']
            births+=int(((va==0)&(vb>0)).sum());deaths+=int(((va>0)&(vb==0)).sum())
        raw=dict(schema='swept-rectangle-raw/1',case=name,geometry_only=True,
                 physical_accuracy_qualified=False,config=geo.config(),history=history,
                 source_sha256=sha(ROOT/'src/tensorlbm/swept_rectangle_geometry_2d.py'),
                 runner_sha256=sha(Path(__file__)))
        path=OUT/(name+'.json.gz');path.write_bytes(gzip.compress((json.dumps(raw,indent=2)+'\n').encode(),mtime=0))
        gcl=max(abs(v) for h in history for row in h['gcl_residual_m2'] for v in row)
        closure=max(abs(v) for h in history for row in h['closure_integral_residual_m_s'] for cell in row for v in cell)
        area=geo.nx*geo.ny*geo.dx_m**2-geo.width_m*geo.height_m
        drift=max(abs(sum(sum(row) for row in h['end']['fluid_volume_m2'])-area) for h in history)
        gate=1e-12*geo.dx_m**2
        summaries.append(dict(case=name,artifact=path.name,sha256=sha(path),steps=count,duration_s=count*dt,birth_events=births,death_events=deaths,
            max_gcl_residual_m2=gcl,max_closure_integral_residual_m_s=closure,max_total_area_error_m2=drift,
            gcl_threshold_m2=gate,closure_threshold_m_s=1e-12*geo.dx_m*dt,area_threshold_m2=1e-12*area,
            passed=gcl<gate and closure<1e-12*geo.dx_m*dt and drift<1e-12*area))
    study=dict(schema='swept-rectangle-study/1',cases=summaries,geometry_only=True,
               fluid_transport_implemented=False,physical_accuracy_qualified=False,periodic_supported=False)
    (OUT/'study.json').write_text(json.dumps(study,indent=2)+'\n');print(json.dumps(study,indent=2))
if __name__=='__main__':main()
