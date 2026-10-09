"""Reconstruct conserved quantities and last actual step from stored populations."""
import json,hashlib
from pathlib import Path
import torch
from tensorlbm.curved_boundary_2d import FixedCircle
from tensorlbm.d2q9 import C
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/assets/curved-boundary'
manifest=json.loads((OUT/'study.json').read_text())
for source,expected in manifest['source_sha256'].items():
    assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==expected,source
for case in manifest['cases']:
    path=OUT/case['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==case['sha256']
    data=json.loads(path.read_text());boundary=FixedCircle(data['n'],data['dx_m'],data['center_m'],data['radius_m'])
    initial=torch.tensor(data['initial_population'],dtype=torch.float64);before=torch.tensor(data['before_last_population'],dtype=torch.float64);final=torch.tensor(data['final_population'],dtype=torch.float64)
    actual,reaction,info=boundary.step(before,.5+3*data['viscosity_m2_s']*data['dt_s']/data['dx_m']**2,data['interpolated'])
    assert torch.equal(actual,final),'last-step replay'
    scale=data['rho_kg_m3']*data['dx_m']**3*data['thickness_m']/data['dt_s']
    deltaP=scale*torch.einsum('ihw,ij->j',final-initial,C.double())
    impulse=torch.tensor(data['solid_impulse_Ns'],dtype=torch.float64)
    assert float((deltaP+impulse).abs().max())<1e-9
    assert abs(float(final.sum()/initial.sum())-1)<1e-11
    assert torch.count_nonzero(final[:,boundary.solid])==0
    assert final[:,~boundary.solid].min()>0
    last=data['history'][-1]
    assert info['rest_mass_repair_l1']==last['rest_mass_repair_l1']
    assert torch.equal(reaction*scale/data['dt_s'],torch.tensor(last['force_N'],dtype=torch.float64))
    assert max(h['mass_relative_error'] for h in data['history'])==case['max_mass_error']
    assert max(h['momentum_error_kg_m_s'] for h in data['history'])==case['max_momentum_error']
    print(case['file'],'raw replay/mass/momentum PASS')
b=manifest['cases'][:4]
assert manifest['spatial_last_pair_relative_change']==abs(b[2]['impulse_x_Ns']/b[1]['impulse_x_Ns']-1)
assert manifest['temporal_half_relative_change']==abs(b[3]['impulse_x_Ns']/b[2]['impulse_x_Ns']-1)
assert manifest['physical_accuracy_qualified'] is False
print('Spatial and temporal convergence NOT QUALIFIED')
