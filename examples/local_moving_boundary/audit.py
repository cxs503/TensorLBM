"""Reconstruct conserved quantities from raw populations, not solver badges."""
import hashlib,json
from pathlib import Path
import torch
from tensorlbm.d2q9 import C
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/assets/local-moving-boundary'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    study=json.loads((OUT/'study.json').read_text())
    actual_impulses={}
    for item in study['cases']:
        p=OUT/item['artifact'];assert sha(p)==item['sha256'];case=json.loads(p.read_text())
        assert case['source_sha256']==sha(ROOT/'src/tensorlbm/local_moving_boundary_2d.py')
        assert case['base_source_sha256']==sha(ROOT/'src/tensorlbm/moving_boundary_2d.py')
        start=torch.tensor(case['initial']['f'],dtype=torch.float64);end=torch.tensor(case['final']['f'],dtype=torch.float64)
        assert (end>=0).all() and torch.isfinite(end).all()
        mass=float(end.sum()-start.sum());assert abs(mass)<1e-8
        dp=torch.einsum('iyx,ia->a',end-start,C.double())
        hist=case['final']['history'];imp=torch.tensor([sum(h['wall_impulse_on_solid'][j]+h['conversion_impulse_on_solid'][j] for h in hist) for j in range(2)],dtype=torch.float64)
        assert float((dp+imp).abs().max())<1e-8
        assert all(h['global_mass_rescale'] is False and h['maximum_transfer_radius']==4 and abs(h['mass_residual'])<1e-9 for h in hist)
        actual_impulses[item['case']]=imp.tolist()
        assert case['restart_bitwise']
        print(item['case'], 'raw mass/momentum/restart PASS',mass)
    coarse=actual_impulses['cross-grid'][0];fine=.5*actual_impulses['step-half'][0]
    change=abs(fine-coarse)/abs(coarse)
    assert abs(change-study['refinement']['relative_impulse_change'])<1e-14
    assert study['refinement']['passed']==(change<.03)
    assert study['physical_accuracy_qualified'] is False
    print('Refinement passed:',study['refinement']['passed'],'physical qualification: false')
if __name__=='__main__':main()
