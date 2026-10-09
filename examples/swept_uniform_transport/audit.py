"""Independent NumPy reconstruction of the restricted uniform-state flux proof.

Imports no TensorLBM solver, geometry or transport module. A success here
qualifies only the recorded uniform co-moving finite-volume flux experiment.
"""
import gzip,hashlib,json,math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/assets/swept-uniform-transport'
C=np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]],dtype=np.float64)
W=np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36],dtype=np.float64)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def array(value):return np.asarray(value,dtype=np.float64)
def close(actual,expected,tolerance,label):
    if not np.isfinite(actual).all() or not np.isfinite(expected).all() or np.max(np.abs(actual-expected),initial=0.)>tolerance:
        raise AssertionError(label)

def bounds(cfg,t):
    center=array(cfg['center_m'])+array(cfg['velocity_m_s'])*t
    return center[0]-cfg['width_m']/2,center[0]+cfg['width_m']/2,center[1]-cfg['height_m']/2,center[1]+cfg['height_m']/2

def overlap(edges,lo,hi):return np.maximum(0.,np.minimum(edges[1:],hi)-np.maximum(edges[:-1],lo))
def faces(cfg,t):
    dx=cfg['dx_m'];x=np.arange(cfg['nx']+1,dtype=np.float64)*dx;y=np.arange(cfg['ny']+1,dtype=np.float64)*dx
    a,b,c,d=bounds(cfg,t);eps=16*np.finfo(float).eps*max(1.,abs(a),abs(b),abs(c),abs(d))
    ox=overlap(x,a,b);oy=overlap(y,c,d)
    vertical=np.diff(y)[:,None]-oy[:,None]*((x>=a-eps)&(x<=b+eps))[None,:]
    horizontal=np.diff(x)[None,:]-((y>=c-eps)&(y<=d+eps))[:,None]*ox[None,:]
    return vertical,horizontal

def geometry_check(h,cfg):
    dx=cfg['dx_m'];nx=cfg['nx'];ny=cfg['ny'];x=np.arange(nx+1)*dx;y=np.arange(ny+1)*dx
    area=nx*ny*dx*dx-cfg['width_m']*cfg['height_m'];t0=h['time_start_s'];dt=h['dt_s']
    for key,t in [('start',t0),('end',t0+dt)]:
        snap=h[key];assert abs(snap['time_s']-t)<1e-13
        a,b,c,d=bounds(cfg,t)
        exact=np.diff(y)[:,None]*np.diff(x)[None,:]-overlap(y,c,d)[:,None]*overlap(x,a,b)[None,:]
        vol=array(snap['fluid_volume_m2']);close(vol,exact,1e-12*dx*dx,'raw volume vs independent rectangle intersection')
        assert vol.min()>=0 and abs(vol.sum()-area)<1e-12*area
        vf,hf=faces(cfg,t);close(array(snap['vertical_open_m']),vf,1e-12*dx,'snapshot vertical faces')
        close(array(snap['horizontal_open_m']),hf,1e-12*dx,'snapshot horizontal faces')
        normals=np.zeros((ny,nx,2))
        for f in snap['wall_facets']:
            j,k=f['cell_yx'];normals[j,k]+=array(f['normal_xy'])*f['length_m']
        closure=np.stack((vf[:,1:]-vf[:,:-1],hf[1:,:]-hf[:-1,:]),axis=-1)+normals
        close(closure,np.zeros_like(closure),1e-12*dx,'snapshot normal closure')
        close(normals,array(snap['wall_normal_length_m']),1e-12*dx,'snapshot facet aggregation')
    events=h['event_times_s'];assert events[0]==t0 and events[-1]==t0+dt and all(a<b for a,b in zip(events[:-1],events[1:]))
    iv=np.zeros((ny,nx+1));ih=np.zeros((ny+1,nx))
    # Gauss2 exact integration of the linear apertures on each event interval.
    for ta,tb in zip(events[:-1],events[1:]):
        half=(tb-ta)/2;mid=(ta+tb)/2
        for t in (mid-half/math.sqrt(3),mid+half/math.sqrt(3)):
            vf,hf=faces(cfg,t);iv+=half*vf;ih+=half*hf
    close(array(h['vertical_open_integral_m_s']),iv,1e-12*dx*dt,'space-time vertical face integral')
    close(array(h['horizontal_open_integral_m_s']),ih,1e-12*dx*dt,'space-time horizontal face integral')
    normals=np.zeros((ny,nx,2));perimeter=moment=0.
    for f in h['wall_facets']:
        j,k=f['cell_yx'];n=array(f['normal_xy']);length=f['integrated_length_m_s']
        assert length>0 and tuple(n) in {(1.,0.),(-1.,0.),(0.,1.),(0.,-1.)}
        assert f['time_start_s']>=t0-1e-13 and f['time_end_s']<=t0+dt+1e-13
        close(array(f['wall_velocity_m_s']),array(cfg['velocity_m_s']),1e-15,'facet velocity')
        normals[j,k]+=length*n;perimeter+=length;moment+=n@array(f['integrated_position_length_m2_s'])
    assert abs(perimeter-2*(cfg['width_m']+cfg['height_m'])*dt)<1e-12*dx*dt
    assert abs(moment+2*cfg['width_m']*cfg['height_m']*dt)<1e-12*dx*dx*dt
    close(normals,array(h['wall_normal_integral_m_s']),1e-12*dx*dt,'integrated wall aggregation')
    closure=np.stack((iv[:,1:]-iv[:,:-1],ih[1:,:]-ih[:-1,:]),axis=-1)+normals
    close(closure,np.zeros_like(closure),1e-12*dx*dt,'integrated normal closure')
    delta=array(h['end']['fluid_volume_m2'])-array(h['start']['fluid_volume_m2'])
    swept=normals@array(cfg['velocity_m_s']);gcl=delta-swept
    close(gcl,np.zeros_like(gcl),1e-12*dx*dx,'independent geometric conservation law')
    close(delta,array(h['fluid_volume_change_m2']),1e-14*dx*dx,'recorded volume difference')
    close(swept,array(h['swept_volume_m2']),1e-14*dx*dx,'recorded swept volume')
    return float(np.max(np.abs(gcl))),float(np.max(np.abs(closure)))

def case_check(path,item):
    assert sha(path)==item['sha256'];raw=json.loads(gzip.decompress(path.read_bytes()))
    assert raw['schema']=='tensorlbm.uniform-swept-case/1' and raw['physical_accuracy_qualified'] is False
    assert raw['restart_bitwise'] is True and item['restart_bitwise'] is True
    assert raw['initial']['schema']==raw['final']['schema']=='tensorlbm.uniform-swept-transport/1'
    cfg=raw['initial']['config'];geom=cfg['geometry'];thickness=cfg['thickness_m'];rho0=cfg['density_kg_m3'];speed=cfg['kinetic_speed_m_s']
    c=C*speed;uw=array(geom['velocity_m_s']);u=uw/speed;cu=C@u
    unit=W*(1+3*cu+4.5*cu**2-1.5*(u@u));initial=array(raw['initial']['Q_kg']);previous=initial.copy()
    initial_mass=float(initial.sum());initial_p=np.einsum('iyx,ia->a',initial,c)
    initial_ke=.5*initial_mass*(uw@uw);totals_wall=np.zeros(2);totals_boundary=np.zeros(2);totals_cleanup=np.zeros(2)
    totals_body_mass=totals_boundary_mass=totals_cleanup_mass=0.
    maxima={k:0. for k in ('population_update_error_kg','population_balance_residual_kg','mass_balance_residual_kg','momentum_balance_residual_Ns','density_relative_error','velocity_error_m_s','gcl_residual_m2','closure_residual_m_s','density_free_energy_abs_J','kinetic_energy_drift_J','wall_mass_flux_kg','cleanup_mass_abs_kg')}
    births=deaths=0;clock=0.
    assert raw['initial']['time_s']==0 and raw['initial']['history']==[]
    for index,step in enumerate(raw['steps']):
        entry=step['entry'];g=step['geometry_interval'];assert entry['physical_accuracy_qualified'] is False and entry['arbitrary_state_transport'] is False
        assert g['config']==geom and abs(entry['time_start_s']-clock)<1e-13 and g['time_start_s']==entry['time_start_s']
        assert abs(entry['time_end_s']-entry['time_start_s']-g['dt_s'])<1e-13
        qa=array(step['Q_before_kg']);qb=array(step['Q_after_kg']);assert np.array_equal(qa,previous)
        assert (qa>=0).all() and (qb>=0).all() and np.isfinite(qa).all() and np.isfinite(qb).all()
        gg,cc=geometry_check(g,geom);maxima['gcl_residual_m2']=max(maxima['gcl_residual_m2'],gg);maxima['closure_residual_m_s']=max(maxima['closure_residual_m_s'],cc)
        v0=array(g['start']['fluid_volume_m2'])*thickness;v1=array(g['end']['fluid_volume_m2'])*thickness
        trace=qa.sum(axis=(1,2))/v0.sum();rho=trace.sum()
        close(qa,trace[:,None,None]*v0[None],1e-11*max(float(qa.max()),1.),'supported uniform old state')
        close(trace,rho*unit,1e-11*rho,'co-moving equilibrium trace')
        fx=trace[:,None,None]*c[:,0,None,None]*array(g['vertical_open_integral_m_s'])[None]*thickness
        fy=trace[:,None,None]*c[:,1,None,None]*array(g['horizontal_open_integral_m_s'])[None]*thickness
        cart=fx[:,:,1:]-fx[:,:,:-1]+fy[:,1:,:]-fy[:,:-1,:];wall=np.zeros_like(qa)
        for facet in g['wall_facets']:
            j,k=facet['cell_yx'];relative=(c-uw)@array(facet['normal_xy']);outgoing=relative>0;incoming=relative<0
            rate_out=np.dot(relative[outgoing],trace[outgoing]);capacity=np.dot(-relative[incoming],unit[incoming]);assert capacity>0
            wall_density=rate_out/capacity;fwall=np.where(outgoing,trace,wall_density*unit)
            flux=relative*fwall*facet['integrated_length_m_s']*thickness
            assert abs(flux.sum())<1e-12*initial_mass
            wall[:,j,k]+=flux
        candidate=qa-cart-wall;dead=v1==0;removed=candidate[:,dead].copy();cleanup=removed.sum(axis=1)
        assert np.max(np.abs(removed),initial=0.)<1e-12*max(float(qa.max()),1.)
        candidate[:,dead]=0
        update_error=float(np.max(np.abs(candidate-qb)));close(candidate,qb,1e-12*max(float(qa.max()),1.),'actual Qold minus Cartesian/wall flux')
        assert (qb[:,dead]==0).all()
        before=qa.sum(axis=(1,2));after=qb.sum(axis=(1,2));boundary=cart.sum(axis=(1,2));body=wall.sum(axis=(1,2))
        residual=after-before+boundary+body+cleanup
        moment_residual=residual@c;body_j=body@c;boundary_j=boundary@c;cleanup_j=cleanup@c
        for name,value in [('body_impulse_Ns',body_j),('boundary_outgoing_impulse_Ns',boundary_j),('roundoff_cleanup_impulse_Ns',cleanup_j)]:close(array(entry[name]),value,1e-12*initial_mass*speed,name)
        assert abs(entry['roundoff_cleanup_mass_kg']-cleanup.sum())<1e-12*initial_mass
        assert abs(entry['mass_kg']-qb.sum())<1e-12*initial_mass
        population_residual=float(np.max(np.abs(residual)));momentum_residual=float(np.max(np.abs(moment_residual)))
        assert population_residual<1e-11*initial_mass and momentum_residual<1e-11*initial_mass*speed
        mask=v1>0;mass=qb.sum(axis=0)[mask];density=mass/v1[mask];moment=np.einsum('iyx,ia->ayx',qb,c)[:,mask];velocity=moment/mass
        rho_error=float(np.max(np.abs(density-rho0))/rho0);velocity_error=float(np.max(np.abs(velocity-uw[:,None])))
        assert rho_error<1e-11 and velocity_error<1e-11*speed
        assert abs(entry['maximum_density_error_kg_m3']-rho_error*rho0)<1e-12*rho0
        assert abs(entry['maximum_velocity_error_m_s']-velocity_error)<1e-12*speed
        assert abs(entry['population_balance_residual_kg']-population_residual)<1e-12*initial_mass
        assert abs(entry['momentum_balance_residual_Ns']-momentum_residual)<1e-12*initial_mass*speed
        compression=float((v1[mask]*(density*np.log(density/rho)-density+rho)).sum())*speed**2/3
        assert abs(compression-entry['density_free_energy_J'])<1e-12*initial_mass*speed**2
        ke=float(.5*(mass*np.sum(velocity**2,axis=0)).sum())
        b=int(np.count_nonzero((v0==0)&(v1>0)));d=int(np.count_nonzero((v0>0)&(v1==0)))
        assert b==entry['birth_cells'] and d==entry['death_cells'];births+=b;deaths+=d
        values={'population_update_error_kg':update_error,'population_balance_residual_kg':population_residual,'mass_balance_residual_kg':abs(float(residual.sum())),
                'momentum_balance_residual_Ns':momentum_residual,'density_relative_error':rho_error,'velocity_error_m_s':velocity_error,
                'density_free_energy_abs_J':abs(compression),'kinetic_energy_drift_J':abs(ke-initial_ke),'wall_mass_flux_kg':abs(float(body.sum())),'cleanup_mass_abs_kg':abs(float(cleanup.sum()))}
        for key,value in values.items():maxima[key]=max(maxima[key],value)
        totals_wall+=body_j;totals_boundary+=boundary_j;totals_cleanup+=cleanup_j
        totals_body_mass+=float(body.sum());totals_boundary_mass+=float(boundary.sum());totals_cleanup_mass+=float(cleanup.sum())
        previous=qb;clock=entry['time_end_s']
        assert raw['final']['history'][index]==entry
    final=raw['final'];assert final['config']==cfg and final['time_s']==clock and np.array_equal(array(final['Q_kg']),previous)
    assert births==item['birth_cells'] and deaths==item['death_cells']
    final_p=np.einsum('iyx,ia->a',previous,c);total_p_residual=final_p-initial_p+totals_wall+totals_boundary+totals_cleanup
    assert np.max(np.abs(total_p_residual))<1e-11*initial_mass*speed
    total_mass_residual=float(previous.sum())-initial_mass+totals_body_mass+totals_boundary_mass+totals_cleanup_mass
    assert abs(total_mass_residual)<1e-11*initial_mass
    close(array(item['total_body_impulse_Ns']),totals_wall,1e-12*initial_mass*speed,'study body impulse')
    assert abs(item['max_density_relative_error']-maxima['density_relative_error'])<1e-12
    assert abs(item['max_velocity_error_m_s']-maxima['velocity_error_m_s'])<1e-12*speed
    assert abs(item['max_momentum_balance_residual_Ns']-maxima['momentum_balance_residual_Ns'])<1e-12*initial_mass*speed
    assert np.max(np.abs(totals_wall))<1e-10*initial_mass*speed
    assert maxima['density_free_energy_abs_J']<1e-11*initial_mass*speed**2 and maxima['kinetic_energy_drift_J']<1e-11*initial_mass*speed**2
    return dict(case=item['case'],artifact=path.name,artifact_sha256=sha(path),steps=len(raw['steps']),birth_cells=births,death_cells=deaths,maxima=maxima,
        total_body_impulse_Ns=totals_wall.tolist(),total_mass_balance_residual_kg=total_mass_residual,total_momentum_balance_residual_Ns=total_p_residual.tolist(),
        actual_flux_updates_reconstructed=True,raw_geometry_reconstructed=True,restart_bitwise_recorded=True,physical_accuracy_qualified=False,arbitrary_state_transport=False,passed=True)

def main():
    study=json.loads((OUT/'study.json').read_text())
    assert study['schema']=='tensorlbm.uniform-swept-study/1' and study['physical_accuracy_qualified'] is False and study['arbitrary_state_transport'] is False
    for path,expected in study['source_sha256'].items():assert sha(ROOT/path)==expected,path
    cases=[case_check(OUT/item['artifact'],item) for item in study['cases']]
    report=dict(schema='tensorlbm.uniform-swept-independent-audit/1',study_sha256=sha(OUT/'study.json'),auditor_sha256=sha(Path(__file__)),
        reconstruction='independent NumPy; no TensorLBM geometry or transport imports',cases=cases,
        scope='uniform co-moving finite-volume D2Q9 flux verification only; no arbitrary fluid BGK solve',physical_accuracy_qualified=False,arbitrary_state_transport=False,passed=True)
    (OUT/'audit-report.json').write_text(json.dumps(report,indent=2)+'\n')
    for case in cases:print(case['case'],'independent geometry/actual population flux/mass/momentum/uniform-state PASS')
    print('physical_accuracy_qualified=false; arbitrary_state_transport=false')
if __name__=='__main__':main()
