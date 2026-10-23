"""Reconstruct final projection, momentum and off-marker leakage from raw fields."""
from pathlib import Path
import hashlib,json
import torch
from tensorlbm.perimeter_boundary_2d import project_velocity,circle_markers
from tensorlbm.ice_coupling_2d import bilinear_map,cross2
from tensorlbm.d2q9 import macroscopic
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/assets/perimeter-boundary'
report=json.loads((OUT/'study.json').read_text())
for name,value in report['source_sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value,name
for record in report['cases']:
    name=record['file'];path=OUT/name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==report['artifact_sha256'][name]
    data=json.loads(path.read_text());n=data['grid'];dx=data['dx_m'];dt=data['dt_s']
    tensor=lambda key:torch.tensor(data[key],dtype=torch.float64)
    pos=tensor('final_positions');target=tensor('target_velocity')
    result=project_velocity(pos,(n,n),dx,tensor('before_projection_velocity'),
                            tensor('projection_cell_mass'),target,data['compliance_inverse_kg'])
    torch.testing.assert_close(result['marker_impulse'],tensor('marker_impulse'),atol=1e-14,rtol=0)
    torch.testing.assert_close(result['corrected_velocity'],tensor('corrected_velocity'),atol=1e-14,rtol=0)
    f=tensor('final_population');rho,ux,uy=macroscopic(f)
    u=torch.stack((ux,uy),-1).reshape(-1,2)*dx/dt
    P=data['cell_mass_kg']*(rho.flatten()[:,None]*u).sum(0)
    last=data['history'][-1]
    momentum=float((P-tensor('initial_momentum')-tensor('total_impulse')).abs().max())
    assert abs(momentum-last['momentum_error_kg_m_s'])<1e-13
    slip=float((result['J']@u-target).norm(dim=1).max())
    assert abs(slip-last['post_stream_marker_slip_m_s'])<1e-13
    center=pos.mean(0);probe,normals=circle_markers(center,.125,dx*.375)
    probe_slip=bilinear_map(probe,(n,n),dx)@result['corrected_velocity']-target[0]
    leakage=float((probe_slip*normals).sum(1).abs().max())
    assert abs(leakage-last['off_marker_normal_slip_m_s'])<1e-12
    for key in data['metrics']:
        assert data['metrics'][key]==max(h[key] for h in data['history'])
    print(name+': raw reconstruction PASS; impermeability qualified FALSE')
print('Eight raw fields and source/artifact hashes verified; physical qualification remains false.')
