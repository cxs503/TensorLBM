"""Short transient perimeter verification, not drag/ice validation."""
from pathlib import Path
import hashlib,json,math
import torch
from tensorlbm.perimeter_boundary_2d import circle_markers,project_velocity
from tensorlbm.ice_coupling_2d import bilinear_map,cross2
from tensorlbm.d2q9 import C,macroscopic
from tensorlbm.solver import collide_bgk_matmul,stream
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/assets/perimeter-boundary'

def run(n,moving,compliance):
    dx=1/n;dt=.0005*(32/n)**2;steps=round(.025/dt)
    weights=torch.tensor([4/9,*([1/9]*4),*([1/36]*4)],dtype=torch.float64)
    c=C.double(); vscale=dx/dt;cell_mass=1000*dx*dx*.2
    initial=torch.tensor([.02,0.],dtype=torch.float64)/vscale
    cu=c@initial
    f=(weights*(1+3*cu+4.5*cu**2-1.5*(initial**2).sum()))[:,None,None].expand(9,n,n).clone()
    normal_target=torch.tensor([.01,0.] if moving else [0.,0.],dtype=torch.float64)
    grid=torch.stack(torch.meshgrid(torch.arange(n)*dx,torch.arange(n)*dx,indexing='ij'),-1).flip(-1).reshape(-1,2).double()
    initialP=cell_mass*(torch.stack(macroscopic(f)[1:],-1).reshape(-1,2)*vscale).sum(0)
    totalI=torch.zeros(2,dtype=torch.float64); history=[]
    for k in range(steps):
        center=[.5+float(normal_target[0])*k*dt,.5]
        positions,normals=circle_markers(center,.125,dx*1.5)
        rho,ux,uy=macroscopic(f);u=torch.stack((ux,uy),-1).reshape(-1,2)*vscale
        target=normal_target.expand(len(positions),2)
        result=project_velocity(positions,(n,n),dx,u,rho.flatten()*cell_mass,target,compliance)
        I=result['grid_impulse'];du_lattice=I/(cell_mass*vscale)
        f=f+(3*weights[:,None]*(c@du_lattice.T)).reshape(9,n,n)
        corrected=result['corrected_velocity']
        sample,nprobe=circle_markers(center,.125,dx*.375)
        probe=bilinear_map(sample,(n,n),dx)@corrected-normal_target
        power_error=abs(float((u*I).sum()-(result['J']@u*result['marker_impulse']).sum()))
        moment_error=abs(float(cross2(grid,I).sum()-cross2(positions,result['marker_impulse']).sum()))
        f=stream(collide_bgk_matmul(f,.5+3*.02*dt/dx**2))
        post=torch.stack(macroscopic(f)[1:],-1).reshape(-1,2)*vscale
        post_slip=result['J']@post-target
        totalI+=I.sum(0)
        P=cell_mass*(f.sum(0).flatten()[:,None]*post).sum(0)
        history.append({'time_s':(k+1)*dt,'marker_slip_m_s':float(result['slip'].norm(dim=1).max()),
          'off_marker_normal_slip_m_s':float((probe*nprobe).sum(1).abs().max()),
          'post_stream_marker_slip_m_s':float(post_slip.norm(dim=1).max()),
          'force_map_error_Ns':float((I.sum(0)-result['marker_impulse'].sum(0)).abs().max()),
          'moment_map_error_Nms':moment_error,'work_map_error_J':power_error,
          'momentum_error_kg_m_s':float((P-initialP-totalI).abs().max()),
          'mass_relative_error':abs(float(f.sum())/(n*n)-1)})
    case={'grid':n,'dx_m':dx,'dt_s':dt,'duration_s':steps*dt,'compliance_inverse_kg':compliance,
          'moving':moving,'marker_spacing_dx':1.5,'probe_spacing_dx':.375,'history':history,
          'final_population':f.tolist(),'final_positions':positions.tolist(),'final_normals':normals.tolist(),
          'target_velocity':target.tolist(),'before_projection_velocity':u.tolist(),
          'projection_cell_mass':(rho.flatten()*cell_mass).tolist(),
          'marker_impulse':result['marker_impulse'].tolist(),'corrected_velocity':corrected.tolist(),
          'total_impulse':totalI.tolist(),'initial_momentum':initialP.tolist(),'cell_mass_kg':cell_mass,
          'physical_accuracy_qualified':False}
    metrics={key:max(h[key] for h in history) for key in history[0] if key!='time_s'}
    case['metrics']=metrics
    case['gates']={'marker_slip':metrics['marker_slip_m_s']<2e-4,
       'off_marker_normal_slip':metrics['off_marker_normal_slip_m_s']<2e-4,
       'post_stream_slip':metrics['post_stream_marker_slip_m_s']<2e-4,
       'conservation':metrics['momentum_error_kg_m_s']<1e-10 and metrics['mass_relative_error']<1e-11}
    return case

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);study=[]
    for n in (32,64):
      for moving in (False,True):
       for compliance in (0.,.1):
        case=run(n,moving,compliance);name=f"n{n}-{'moving' if moving else 'fixed'}-c{compliance:g}.json"
        (OUT/name).write_text(json.dumps(case));study.append({'file':name,'metrics':case['metrics'],'gates':case['gates']})
    source_paths=['src/tensorlbm/perimeter_boundary_2d.py','src/tensorlbm/ice_coupling_2d.py','src/tensorlbm/d2q9.py','src/tensorlbm/solver.py','examples/perimeter_boundary/run.py']
    report={'schema':'tensorlbm.perimeter-boundary/1','physical_accuracy_qualified':False,'cases':study,
      'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths},
      'artifact_sha256':{c['file']:hashlib.sha256((OUT/c['file']).read_bytes()).hexdigest() for c in study}}
    (OUT/'study.json').write_text(json.dumps(report,indent=2));print(json.dumps(study,indent=2))
