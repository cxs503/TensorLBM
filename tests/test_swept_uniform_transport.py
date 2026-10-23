import copy
import json
import torch
import pytest
from tensorlbm.swept_uniform_transport_2d import UniformSweptTransport

@pytest.mark.parametrize('velocity',[(0.,0.),(.08,.03),(-.08,-.03),(.08,0.)])
def test_actual_uniform_flux_crosses_cells_and_restarts(velocity):
    s=UniformSweptTransport(geometry={'velocity_m_s':velocity});mass=float(s.Q.sum());body=torch.zeros(2,dtype=torch.float64)
    for _ in range(10):s.step(.1)
    raw=json.loads(json.dumps(s.snapshot()));saved=copy.deepcopy(raw);r=UniformSweptTransport.restore(raw)
    for _ in range(10):
        entry=s.step(.1);assert entry==r.step(.1)
        body+=torch.tensor(entry['body_impulse_Ns'])
        assert entry['maximum_density_error_kg_m3']/1000<1e-11
        assert entry['maximum_velocity_error_m_s']<1e-11
        assert entry['momentum_balance_residual_Ns']<1e-10
    assert torch.equal(s.Q,r.Q) and s.snapshot()==r.snapshot() and raw==saved
    assert abs(float(s.Q.sum())-mass)<1e-10 and float(body.abs().max())<1e-10
    if velocity!=(0.,0.):
        assert sum(e['birth_cells'] for e in s.history)>0 and sum(e['death_cells'] for e in s.history)>0


def test_wrong_aperture_rejected_without_mutating_state(monkeypatch):
    s=UniformSweptTransport();before=s.snapshot();original=s.geometry.interval
    def broken(self,*args):
        d=original(*args);d['vertical_open_integral_m_s'][0,1]+=.01;return d
    monkeypatch.setattr(type(s.geometry),'interval',broken)
    with pytest.raises(ValueError):s.step(.1)
    assert s.snapshot()==before


def test_nonuniform_input_is_rejected_instead_of_reset():
    s=UniformSweptTransport();s.Q[0,0,0]+=.1;before=s.snapshot()
    with pytest.raises(ValueError,match='only uniform'):s.step(.1)
    assert s.snapshot()==before


def test_bad_restart_and_outside_domain_rejected():
    s=UniformSweptTransport();s.step(.1);raw=s.snapshot()
    for edit in [lambda d:d['Q_kg'][0][0].__setitem__(0,-1.),lambda d:d['history'][0].update(physical_accuracy_qualified=True),lambda d:d.update(time_s=1.),lambda d:d['Q_kg'][0][0].__setitem__(0,100.)]:
        bad=copy.deepcopy(raw);edit(bad)
        with pytest.raises(ValueError):UniformSweptTransport.restore(bad)
    before=s.snapshot()
    with pytest.raises(ValueError,match='inside domain'):s.step(100.)
    assert s.snapshot()==before

@pytest.mark.parametrize('config',[{'density_kg_m3':0},{'thickness_m':True},{'kinetic_speed_m_s':float('nan')}])
def test_bad_scales(config):
    with pytest.raises(ValueError):UniformSweptTransport(**config)
