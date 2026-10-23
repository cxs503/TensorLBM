"""Audit hashes and reconstruct the final real fixed-boundary step."""
from pathlib import Path
import hashlib,json
import torch
from tensorlbm.fixed_disk_boundary_2d import bounce_step
from tensorlbm.d2q9 import C
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/assets/perimeter-boundary'
study=json.loads((OUT/'fixed-link-study.json').read_text())
for name,value in study['source_sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value
for record in study['cases']:
    path=OUT/record['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
    data=json.loads(path.read_text());tensor=lambda k:torch.tensor(data[k],dtype=torch.float64)
    mask=torch.tensor(data['mask'],dtype=torch.bool);initial=tensor('initial_population');final=tensor('final_population')
    rebuilt,reaction=bounce_step(tensor('before_last_population'),mask,.5+3*.02*data['dt_s']/data['dx_m']**2)
    torch.testing.assert_close(rebuilt,final,atol=0,rtol=0)
    assert final[:,mask].count_nonzero()==0
    scale=1000*data['dx_m']**2*.2*data['dx_m']/data['dt_s']
    P=scale*torch.einsum('ihw,ij->j',final-initial,C.double())
    residual=float((P+tensor('solid_impulse_Ns')).abs().max())
    assert residual<1e-10
    assert abs(float(final.sum()/initial.sum())-1)<1e-11
    assert all(data['gates'].values())
    print(record['file']+': final-step/raw-mass/excluded-interior/momentum/hash PASS')
