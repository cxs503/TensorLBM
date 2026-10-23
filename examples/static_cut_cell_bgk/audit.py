"""Independent NumPy reconstruction of static nonuniform FV-BGK raw steps.

No production solver imports. Reconstructs collision, local upwind/diffuse
transport, merged-volume ownership, energies, CFL and kinetic Fourier oracles.
Restart exactness remains a producer execution claim, not an independent rerun.
"""
import gzip,hashlib,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/assets/static-cut-cell-bgk'
C=np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]],dtype=float)
W=np.array([4/9,*([1/9]*4),*([1/36]*4)])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,tol=1e-10):
    aa=np.asarray(a);bb=np.asarray(b)
    if not np.isfinite(aa).all() or not np.isfinite(bb).all():raise ValueError('nonfinite raw field')
    if aa.size == 0 and bb.size == 0:return 0.0
    if aa.shape != bb.shape and aa.ndim and bb.ndim:raise ValueError('raw field shape mismatch')
    err=float(np.max(abs(aa-bb),initial=0))
    if err>tol:raise ValueError('independent raw reconstruction mismatch: '+str(err))
    return err

def collision(Q,V,c,speed,h,tau):
    mass=Q.sum(1);rho=mass/V;u=Q@c/mass[:,None];cu=u@c.T/speed**2
    eq=V[:,None]*rho[:,None]*W[None]*(1+3*cu+4.5*cu**2-1.5*np.sum(u*u,axis=1)[:,None]/speed**2)
    weight=-math.expm1(-h/tau)
    return (1-weight)*Q+weight*eq

def energy(Q,V,c,config):
    mass=Q.sum(1);rho=mass/V;p=Q@c;ref=config['density_reference_kg_m3'];cs2=config['kinetic_speed_m_s']**2/3
    kinetic=float(np.sum(.5*np.sum(p*p,axis=1)/mass));compression=float(np.sum(V*(rho*np.log(rho/ref)-rho+ref)))*cs2
    mask=Q>0;ratio=Q/(V[:,None]*W[None]*ref);entropy=float(np.sum(Q[mask]*np.log(ratio[mask])))*cs2
    return dict(kinetic_J=kinetic,compression_J=compression,mechanical_J=kinetic+compression,population_entropy_J=entropy)

def expm(A):
    # Independent scaling/squaring Taylor; nine-dimensional small stable mode.
    n=len(A);norm=float(np.linalg.norm(A,ord=np.inf));s=max(0,math.ceil(math.log2(norm/.5))) if norm else 0
    X=A/(2**s);F=np.eye(n,dtype=complex);term=F.copy()
    for k in range(1,100):
        term=term@X/k;F+=term
        if np.linalg.norm(term,ord=np.inf)<1e-17*max(1.,np.linalg.norm(F,ord=np.inf)):break
    else:raise ValueError('independent matrix exponential failed')
    for _ in range(s):F=F@F
    return F

def geometry(g,config):
    gm=np.asarray(g['group_map'],dtype=int);area=np.asarray(g['fluid_volume_m2']);dx=config['geometry']['dx_m'];thick=config['thickness_m'];ny,nx=gm.shape
    x0,y0=config['geometry']['center_m'];w=config['geometry']['width_m'];h=config['geometry']['height_m']
    overlapx=np.maximum(0,np.minimum((np.arange(nx)+1)*dx,x0+w/2)-np.maximum(np.arange(nx)*dx,x0-w/2))
    overlapy=np.maximum(0,np.minimum((np.arange(ny)+1)*dx,y0+h/2)-np.maximum(np.arange(ny)*dx,y0-h/2))
    expected=np.ones((ny,nx))*dx*dx-(overlapy[:,None]*overlapx[None] if config['solid_rectangle'] else 0)
    close(area,expected,1e-14)
    V=np.asarray(g['group_volume_m3']);vol=np.zeros(len(V));np.add.at(vol,gm[gm>=0],area[gm>=0]*thick);close(V,vol,1e-14)
    if np.any(V<=0) or np.any((gm<0)&(area>1e-14)):raise ValueError('invalid merged volume ownership')
    normals=np.zeros((len(V),2));faces=g['faces'];seen=set()
    for f in faces:
        l,r=f['group_left'],f['group_right'];axis=f['axis'];y,x=f['face_yx'];key=(axis,y,x)
        if key in seen:raise ValueError('duplicate shared face')
        seen.add(key)
        expected_left=gm[y,(x-1)%nx] if axis==0 else gm[(y-1)%ny,x]
        if l!=expected_left or r!=gm[y,x] or l==r:raise ValueError('invalid shared face ownership')
        actual=float((np.asarray(g['vertical_open_m'])[y,x] if axis==0 else np.asarray(g['horizontal_open_m'])[y,x])*thick)
        close(actual,f['area_m2'],1e-14);normals[l,axis]+=actual;normals[r,axis]-=actual
    # All nonzero interfaces between distinct groups must occur exactly once.
    expected_keys=set()
    for axis in (0,1):
        aperture=np.asarray(g['vertical_open_m'] if axis==0 else g['horizontal_open_m'])
        for y in range(ny):
            for x in range(nx):
                l=gm[y,(x-1)%nx] if axis==0 else gm[(y-1)%ny,x];r=gm[y,x]
                if aperture[y,x]>0 and l!=r:expected_keys.add((axis,y,x))
    if seen!=expected_keys:raise ValueError('missing shared interface')
    for f in g['wall_facets']:
        y,x=f['cell_yx']
        if f['group']!=gm[y,x]:raise ValueError('invalid wall group owner')
        close(f['area_m2'],f['length_m']*thick,1e-14);normals[f['group']]+=np.asarray(f['normal_xy'])*f['area_m2']
    close(normals,0,1e-13)
    return gm,V,area*thick

def main():
    study=json.loads((OUT/'study.json').read_text())
    if study['physical_accuracy_qualified'] is not False or study['moving_geometry_implemented'] is not False:raise ValueError('false physical/moving qualification')
    cases=[];summaries={r['case']:r for r in study['cases']}
    for summary in study['cases']:
        path=OUT/summary['artifact']
        if sha(path)!=summary['sha256']:raise ValueError('raw hash mismatch')
        raw=json.loads(gzip.decompress(path.read_bytes()));config=raw['initial']['config'];g=raw['geometry_static']
        if raw['physical_accuracy_qualified'] is not False or raw['moving_geometry_implemented'] is not False:raise ValueError('false raw qualification')
        for name,digest in raw['source_sha256'].items():
            if sha(ROOT/name)!=digest:raise ValueError('source SHA mismatch: '+name)
        gh=hashlib.sha256(json.dumps(g,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if gh!=raw['geometry_sha256'] or gh!=raw['initial']['geometry_sha256'] or gh!=raw['final']['geometry_sha256']:raise ValueError('geometry SHA mismatch')
        gm,V,vol=geometry(g,config);speed=config['kinetic_speed_m_s'];c=C*speed;Q0=np.asarray(raw['initial_requested_cell_Q_kg'])
        grouped=np.zeros((len(V),9));mask=gm>=0;np.add.at(grouped,gm[mask],Q0[:,mask].T);close(grouped,raw['initial']['group_Q_kg'])
        close(Q0.sum(),grouped.sum());close(np.einsum('iyx,ia->a',Q0,c),grouped.sum(0)@c)
        projected=np.zeros_like(Q0);projected[:,mask]=(grouped[gm[mask]]/V[gm[mask],None]*vol[mask,None]).T
        ip=raw['initial']['initial_projection'];close(ip['maximum_cell_population_change_kg'],np.max(abs(projected-Q0)))
        close(ip['energy_after']['mechanical_J'],energy(grouped,V,c,config)['mechanical_J'])
        faces=g['faces'];walls=g['wall_facets'];l=np.array([f['group_left'] for f in faces],int);r=np.array([f['group_right'] for f in faces],int)
        rates=np.array([c[:,f['axis']]*f['area_m2'] for f in faces]).reshape(-1,9)
        wg=np.array([w['group'] for w in walls],int);wr=np.array([np.asarray(w['normal_xy'])@c.T for w in walls]).reshape(-1,9);wa=np.array([w['area_m2'] for w in walls])
        loss=np.zeros_like(grouped);np.add.at(loss,l,np.maximum(rates,0));np.add.at(loss,r,np.maximum(-rates,0));np.add.at(loss,wg,np.maximum(wr,0)*wa[:,None])
        limit=config['max_cfl']/np.max(loss/V[:,None]);close(limit,summary['cfl_limit_s'])
        previous=grouped;clock=0.;maxq=0.;maxp=0.;bodytotal=np.zeros(2);peak=0.;max_energy_increase=-float('inf')
        for step in raw['steps']:
            e=step['entry'];before=np.asarray(step['group_Q_before_kg']);after=np.asarray(step['group_Q_after_kg']);dt=e['dt_s']
            if e['physical_accuracy_qualified'] is not False or e['arbitrary_state_transport'] is not True or e['moving_geometry_implemented'] is not False:raise ValueError('false step qualification')
            close(before,previous);close(clock,e['time_start_s']);clock+=dt;close(clock,e['time_end_s']);close(dt,e['time_end_s']-e['time_start_s'])
            if dt>limit*(1+1e-13):raise ValueError('CFL violation accepted')
            half=collision(before,V,c,speed,dt/2,config['relaxation_time_s']);close(half,step['half_collision_group_Q_kg']);density=half/V[:,None]
            flux=dt*rates*np.where(rates>=0,density[l],density[r]);close(flux,step['face_flux_kg'])
            delta=np.zeros_like(before);np.add.at(delta,l,flux);np.add.at(delta,r,-flux)
            if len(walls):
                rate_out=np.sum(np.maximum(wr,0)*density[wg],axis=1);cap=np.sum(np.maximum(-wr,0)*W[None],axis=1)
                wallf=np.where(wr>0,density[wg],rate_out[:,None]/cap[:,None]*W[None]);wf=dt*wr*wallf*wa[:,None]
                np.add.at(delta,wg,wf)
            else:wf=np.zeros((0,9))
            close(wf,step['wall_flux_kg']);transport=half-delta;close(transport,step['transport_group_Q_kg'])
            computed=collision(transport,V,c,speed,dt/2,config['relaxation_time_s']);maxq=max(maxq,close(computed,after))
            if np.any(after<0):raise ValueError('negative transported population')
            wall=wf.sum(0);coll=(half-before).sum(0)+(after-transport).sum(0);res=after.sum(0)-before.sum(0)+wall-coll
            close(res.sum(),e['mass_balance_residual_kg']);close(res@c,e['momentum_balance_residual_Ns']);close(wall.sum(),e['wall_mass_flux_kg']);close(wall@c,e['body_impulse_Ns'])
            close((half-before).sum(1),0);close((after-transport).sum(1),0);close((half-before)@c,0);close((after-transport)@c,0)
            maxp=max(maxp,float(np.max(abs(res@c))));bodytotal+=wall@c;peak=max(peak,float(np.linalg.norm(wall@c))/dt)
            for Q,key in [(before,'energy_before'),(half,'energy_half_collision'),(transport,'energy_after_transport'),(after,'energy_after')]:
                for name,val in energy(Q,V,c,config).items():close(val,e[key][name],1e-9)
            max_energy_increase=max(max_energy_increase,e['energy_after']['mechanical_J']-e['energy_before']['mechanical_J'])
            previous=after
        close(previous,raw['final']['group_Q_kg']);close(clock,raw['final']['time_s']);close(bodytotal,summary['total_body_impulse_Ns']);close(peak,summary['peak_body_force_N'])
        row={'case':raw['case'],'steps':len(raw['steps']),'max_Q_reconstruction_error_kg':maxq,'max_momentum_residual_Ns':maxp,'maximum_step_macroscopic_energy_increase_J':max_energy_increase}
        ref=raw['reference']
        if ref:
            k=2*math.pi/(config['geometry']['ny']*config['geometry']['dx_m']);P=W[:,None]*(1+3*(c@c.T)/speed**2);A=-(np.eye(9)-P)/config['relaxation_time_s']-np.diag(1j*k*c[:,1]);dx=config['geometry']['dx_m']
            up=np.where(c[:,1]>=0,-c[:,1]*(1-np.exp(-1j*k*dx))/dx,-c[:,1]*(np.exp(1j*k*dx)-1)/dx)
            B=-(np.eye(9)-P)/config['relaxation_time_s']+np.diag(up);initial=1000*3*W*c[:,0]*1e-4/speed**2
            amp=float((c[:,0]@(expm(A*clock)@initial)).real/1000);semi=float((c[:,0]@(expm(B*clock)@initial)).real/1000)
            close(A.real,ref['continuous_generator_real']);close(A.imag,ref['continuous_generator_imag']);close(B.real,ref['semidiscrete_generator_real']);close(B.imag,ref['semidiscrete_generator_imag']);close(initial,ref['initial_mode_real']);close(amp,ref['continuous_amplitude_m_s'],1e-15);close(semi,ref['semidiscrete_amplitude_m_s'],1e-15)
            cell=previous[gm]/V[gm,None]*vol[:,:,None];ux=(cell@c)[...,0]/cell.sum(2);y=(np.arange(gm.shape[0])+.5)*dx;mode=float(2*np.mean(ux*np.sin(k*y)[:,None]));close(mode,raw['final_shear_amplitude_m_s'],1e-15)
            err=abs(mode-amp)/abs(amp);terr=abs(mode-semi)/abs(semi);close(err,summary['continuous_relative_error'],1e-12);close(terr,summary['time_relative_error'],1e-12)
            row.update(continuous_kinetic_relative_error=err,semidiscrete_time_relative_error=terr)
        cases.append(row)
    base=np.asarray(summaries['nonuniform-cut']['total_body_impulse_Ns']);fine=np.asarray(summaries['nonuniform-grid-half']['total_body_impulse_Ns']);half=np.asarray(summaries['nonuniform-time-half']['total_body_impulse_Ns'])
    grid=float(np.linalg.norm(fine-base)/np.linalg.norm(base));time=float(np.linalg.norm(half-base)/np.linalg.norm(base));ref=study['refinement']
    close(grid,ref['total_impulse_grid_half_relative_change']);close(time,ref['total_impulse_time_half_relative_change'])
    if ref['total_impulse_grid_passed']!=(grid<.03) or ref['total_impulse_time_passed']!=(time<.03) or ref['dynamic_grid_passed']!=(grid<.03 and ref['grid_passed']):raise ValueError('false refinement gate')
    report={'schema':'tensorlbm.independent-static-bgk-audit/1','cases':cases,'study_sha256':sha(OUT/'study.json'),'auditor_sha256':sha(Path(__file__)),'total_impulse_grid_relative_change':grid,'total_impulse_time_relative_change':time,'physical_accuracy_qualified':False,'moving_geometry_implemented':False,'scope':'NumPy raw collision/transport/geometry/CFL/Fourier reconstruction; no production solver imports or independent restart execution'}
    (OUT/'audit-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
