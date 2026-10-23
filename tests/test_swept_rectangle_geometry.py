import math
import pytest
import torch
from tensorlbm.swept_rectangle_geometry_2d import SweptRectangleGeometry

@pytest.mark.parametrize('velocity',[(.08,.03),(-.12,.09),(.07,-.11),(-.08,-.06),(0.,.08),(.1,0.),(0.,0.)])
def test_exact_local_closure_gcl_global_volume(velocity):
    g=SweptRectangleGeometry(velocity_m_s=velocity)
    r=g.interval(.1,.9)
    total=g.nx*g.ny*g.dx_m**2-g.width_m*g.height_m
    for snap in (r['start'],r['end']):
        assert snap['fluid_volume_m2'].min()>=0
        assert snap['vertical_open_m'].min()>=0
        assert snap['horizontal_open_m'].min()>=0
        assert abs(snap['fluid_volume_m2'].sum().item()-total)<1e-14
        assert snap['closure_residual_m'].abs().max()<1e-14
    assert r['gcl_residual_m2'].abs().max()<1e-14
    assert r['closure_integral_residual_m_s'].abs().max()<1e-14
    assert abs(r['swept_volume_m2'].sum().item())<1e-14
    # Closed body pressure resultant and first moment of normals.
    assert r['wall_normal_integral_m_s'].sum((0,1)).abs().max()<1e-14


def test_known_partial_cell_area_and_normals():
    g=SweptRectangleGeometry(nx=8,ny=8,dx_m=1.,width_m=1.5,height_m=1.5,center_m=(3.,3.),velocity_m_s=(0.,0.))
    s=g.at()
    assert s['fluid_volume_m2'][2,2].item()==1.-.75**2
    assert s['vertical_open_m'][2,3].item()==.25
    assert s['horizontal_open_m'][3,2].item()==.25
    assert {tuple(f['normal_xy']) for f in s['wall_facets']}=={(1.,0.),(-1.,0.),(0.,1.),(0.,-1.)}


def test_exact_grid_aligned_static():
    g=SweptRectangleGeometry(width_m=.4,height_m=.2,center_m=(.5,.5),velocity_m_s=(0.,0.))
    r=g.interval(dt_s=2.)
    assert torch.equal(r['start']['fluid_volume_m2'],r['end']['fluid_volume_m2'])
    assert r['swept_volume_m2'].count_nonzero()==0
    assert r['closure_integral_residual_m_s'].abs().max()<1e-14
    assert r['start']['closure_residual_m'].abs().max()<1e-14


def test_birth_death_are_continuous_areas():
    g=SweptRectangleGeometry(width_m=.4,height_m=.3,center_m=(.5,.45),velocity_m_s=(.08,.03))
    first=g.at();last=g.at(1.8)
    born=(first['fluid_volume_m2']==0)&(last['fluid_volume_m2']>0)
    dead=(first['fluid_volume_m2']>0)&(last['fluid_volume_m2']==0)
    assert born.any() and dead.any()
    h=1e-7;t=.625
    jump=(g.at(t+h)['fluid_volume_m2']-g.at(t-h)['fluid_volume_m2']).abs().max().item()
    assert 0<jump<1e-8


def test_interval_additivity_with_crossings():
    g=SweptRectangleGeometry(velocity_m_s=(.17,.13))
    full=g.interval(0.,1.);a=g.interval(0.,.37);b=g.interval(.37,.63)
    for name in ('vertical_open_integral_m_s','horizontal_open_integral_m_s','wall_normal_integral_m_s','swept_volume_m2'):
        assert (full[name]-a[name]-b[name]).abs().max()<1e-14
    assert len(full['event_times_s'])>4


def test_perimeter_and_moment_integrals():
    g=SweptRectangleGeometry();r=g.interval(.1,.8)
    total=sum(f['integrated_length_m_s'] for f in r['wall_facets'])
    assert abs(total-2*(g.width_m+g.height_m)*.8)<1e-14
    moment=sum(f['normal_xy'][j]*f['integrated_position_length_m2_s'][j] for f in r['wall_facets'] for j in range(2))
    # Divergence theorem over inner solid, fluid normal has opposite sign.
    assert abs(moment+2*g.width_m*g.height_m*.8)<1e-14


def test_time_refinement_has_same_integrated_geometry():
    g=SweptRectangleGeometry()
    full=g.interval(0.,1.)
    summed=torch.zeros_like(full['swept_volume_m2'])
    for k in range(20):summed+=g.interval(k*.05,.05)['swept_volume_m2']
    assert (summed-full['swept_volume_m2']).abs().max()<1e-14


def test_outside_domain_and_invalid_time_rejected():
    g=SweptRectangleGeometry()
    with pytest.raises(ValueError,match='inside'):g.interval(dt_s=20.)
    for dt in [0.,-1.,float('nan'),True]:
        with pytest.raises(ValueError):g.interval(dt_s=dt)
    with pytest.raises(ValueError):g.at(-1.)

@pytest.mark.parametrize('kwargs',[dict(nx=True),dict(dx_m=0.),dict(width_m=-1.),dict(height_m=float('inf')),dict(center_m=(True,.5)),dict(velocity_m_s=(.1,)),dict(center_m=(.02,.5))])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):SweptRectangleGeometry(**kwargs)
