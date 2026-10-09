"""Actual nonuniform FV-BGK trajectories; fixed parameters and retained failures."""
import cmath,gzip,hashlib,json,math
from pathlib import Path
import torch
from tensorlbm.cut_cell_bgk_2d import CutCellBGK2D,C,W
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/assets/static-cut-cell-bgk'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fields(geometry,kind):
    nx,ny,dx=geometry['nx'],geometry['ny'],geometry['dx_m'];y,x=torch.meshgrid((torch.arange(ny,dtype=torch.float64)+.5)*dx,(torch.arange(nx,dtype=torch.float64)+.5)*dx,indexing='ij')
    rho=torch.full((ny,nx),1000.,dtype=torch.float64);velocity=torch.zeros(ny,nx,2,dtype=torch.float64)
    if kind=='waves':
        rho*=1+.001*torch.sin(2*math.pi*x/(nx*dx))*torch.cos(2*math.pi*y/(ny*dx))
        velocity[...,0]=.01*torch.sin(2*math.pi*y/(ny*dx));velocity[...,1]=.004*torch.cos(2*math.pi*x/(nx*dx))
    elif kind=='shear':velocity[...,0]=1e-4*torch.sin(2*math.pi*y/(ny*dx))
    elif kind=='comoving':velocity[...,0]=.02;velocity[...,1]=-.015
    return rho,velocity

def shear_reference(sim,duration_s,dt_s):
    k=2*math.pi/(sim.geometry.ny*sim.geometry.dx_m);c=sim.c;speed=sim.config['kinetic_speed_m_s'];tau=sim.config['relaxation_time_s'];amp=1e-4;rho=1000.
    P=W[:,None]*(1+3*(c@c.T)/speed**2);I=torch.eye(9,dtype=torch.float64)
    collision=-(I-P)/tau
    generator=collision.to(torch.complex128)-torch.diag(1j*k*c[:,1])
    dx=sim.geometry.dx_m
    transport=torch.where(c[:,1]>=0,-c[:,1]*(1-cmath.exp(-1j*k*dx))/dx,-c[:,1]*(cmath.exp(1j*k*dx)-1)/dx)
    discrete_generator=collision.to(torch.complex128)+torch.diag(transport)
    initial=(rho*3*W*c[:,0]*amp/speed**2).to(torch.complex128)
    continuous=c[:,0].to(torch.complex128)@(torch.linalg.matrix_exp(generator*duration_s)@initial)/rho
    semidiscrete=c[:,0].to(torch.complex128)@(torch.linalg.matrix_exp(discrete_generator*duration_s)@initial)/rho
    return dict(kind='analytical linearized continuous D2Q9 BGK Fourier mode, not nonlinear Navier-Stokes',wavenumber_m_inv=k,
        continuous_generator_real=generator.real.tolist(),continuous_generator_imag=generator.imag.tolist(),semidiscrete_generator_real=discrete_generator.real.tolist(),semidiscrete_generator_imag=discrete_generator.imag.tolist(),
        initial_mode_real=initial.real.tolist(),continuous_amplitude_m_s=float(continuous.real),semidiscrete_amplitude_m_s=float(semidiscrete.real),
        neglected_nonlinear_velocity_order=amp**2/speed**2,nominal_kinetic_viscosity_m2_s=speed**2*tau/3,
        roll_lbm_tau_mapping_used=False)

def main():
    OUT.mkdir(parents=True,exist_ok=True);cases=[]
    cut=dict(nx=12,ny=10,dx_m=.1,center_m=(.56,.47),velocity_m_s=(0.,0.))
    fine=dict(cut,nx=24,ny=20,dx_m=.05)
    configs=[('static-cut',cut,True,'static',.01,12),('nonuniform-cut',cut,True,'waves',.01,12),
        ('nonuniform-time-half',cut,True,'waves',.005,24),('nonuniform-grid-half',fine,True,'waves',.005,24),
        ('periodic-comoving',cut,False,'comoving',.01,12)]
    for n,dt in [(16,.01),(32,.005),(64,.0025)]:
        configs.append((f'shear-{n}',dict(nx=n,ny=n,dx_m=.8/n,center_m=(.4,.4),width_m=.21,height_m=.19,velocity_m_s=(0.,0.)),False,'shear',dt,round(.1/dt)))
    configs.append(('shear-32-time-half',dict(nx=32,ny=32,dx_m=.025,center_m=(.4,.4),width_m=.21,height_m=.19,velocity_m_s=(0.,0.)),False,'shear',.0025,40))
    for name,geom,solid,kind,dt,count in configs:
        rho,velocity=fields(geom,kind);s=CutCellBGK2D(geometry=geom,solid_rectangle=solid,density_kg_m3=rho,velocity_m_s=velocity)
        initial=s.snapshot();history=[];restart=None
        for k in range(count):
            if k==count//2:restart=CutCellBGK2D.restore(json.loads(json.dumps(s.snapshot())))
            before=s.group_Q.tolist();entry=s.step(dt)
            history.append(dict(group_Q_before_kg=before,group_Q_after_kg=s.group_Q.tolist(),entry=entry,**s.last_step_raw))
            if restart is not None:
                if entry!=restart.step(dt) or not torch.equal(s.group_Q,restart.group_Q):raise ValueError('restart not bitwise')
        reference=None;mode=None
        if kind=='shear':
            reference=shear_reference(s,count*dt,dt)
            Q=s.Q;ux=torch.einsum('iyx,ia->ayx',Q,s.c)[0]/Q.sum(0);y=(torch.arange(s.geometry.ny,dtype=torch.float64)+.5)*s.geometry.dx_m
            mode=float(2*(ux*torch.sin(reference['wavenumber_m_inv']*y)[:,None]).mean())
        paths=['src/tensorlbm/cut_cell_bgk_2d.py','src/tensorlbm/swept_rectangle_geometry_2d.py','src/tensorlbm/d2q9.py','examples/static_cut_cell_bgk/run.py']
        raw=dict(schema='tensorlbm.static-cut-cell-case/1',case=name,initial_requested_cell_Q_kg=s.initial_cell_Q_kg.tolist(),initial=initial,final=s.snapshot(),geometry_static=s.geometry_data,
            geometry_sha256=s.geometry_sha256,steps=history,reference=reference,final_shear_amplitude_m_s=mode,restart_bitwise=s.snapshot()==restart.snapshot(),
            physical_accuracy_qualified=False,moving_geometry_implemented=False,source_sha256={p:sha(ROOT/p) for p in paths})
        path=OUT/(name+'.json.gz');path.write_bytes(gzip.compress((json.dumps(raw,separators=(',',':'),allow_nan=False)+'\n').encode(),mtime=0))
        entries=[h['entry'] for h in history];peak=max(math.hypot(*e['body_impulse_Ns'])/dt for e in entries)
        summary=dict(case=name,artifact=path.name,sha256=sha(path),steps=count,duration_s=count*dt,initial_kind=kind,body=solid,group_count=s.group_count,
            merged_cell_count=int((s.volume_m3>0).sum())-s.group_count,cfl_limit_s=s.cfl_limit_s,peak_body_force_N=peak,
            total_body_impulse_Ns=[sum(e['body_impulse_Ns'][j] for e in entries) for j in range(2)],
            max_mass_residual_kg=max(abs(e['mass_balance_residual_kg']) for e in entries),
            max_momentum_residual_Ns=max(max(map(abs,e['momentum_balance_residual_Ns'])) for e in entries),
            minimum_population_density_kg_m3=min(e['minimum_population_density_kg_m3'] for e in entries),
            restart_bitwise=raw['restart_bitwise'],initial_projection=initial['initial_projection'])
        if reference:
            summary.update(final_shear_amplitude_m_s=mode,continuous_kinetic_reference_amplitude_m_s=reference['continuous_amplitude_m_s'],
                semidiscrete_reference_amplitude_m_s=reference['semidiscrete_amplitude_m_s'],
                continuous_relative_error=abs(mode-reference['continuous_amplitude_m_s'])/abs(reference['continuous_amplitude_m_s']),
                time_relative_error=abs(mode-reference['semidiscrete_amplitude_m_s'])/abs(reference['semidiscrete_amplitude_m_s']))
        cases.append(summary);print(name,'completed',flush=True)
    by={c['case']:c for c in cases};base=by['nonuniform-cut']['peak_body_force_N'];half=by['nonuniform-time-half']['peak_body_force_N'];fine=by['nonuniform-grid-half']['peak_body_force_N']
    study=dict(schema='tensorlbm.static-cut-cell-study/1',cases=cases,static_nonuniform_transport_implemented=True,moving_geometry_implemented=False,physical_accuracy_qualified=False,
        continuous_kinetic_tau_s=.2,nominal_kinetic_viscosity_m2_s=1/3*.2,roll_lbm_tau_mapping_used=False,
        refinement=dict(peak_force_time_half_relative_change=abs(half-base)/abs(base),peak_force_grid_half_relative_change=abs(fine-base)/abs(base),threshold=.03,
            time_passed=abs(half-base)/abs(base)<.03,grid_passed=abs(fine-base)/abs(base)<.03,
            shear_finest_continuous_error=by['shear-64']['continuous_relative_error'],shear_finest_passed=by['shear-64']['continuous_relative_error']<.03,
            shear_errors_monotone=by['shear-64']['continuous_relative_error']<by['shear-32']['continuous_relative_error']<by['shear-16']['continuous_relative_error']))
    j0=by['nonuniform-cut']['total_body_impulse_Ns'];jh=by['nonuniform-time-half']['total_body_impulse_Ns'];jf=by['nonuniform-grid-half']['total_body_impulse_Ns']
    jnorm=math.hypot(*j0)
    jt=math.hypot(*(a-b for a,b in zip(jh,j0)))/jnorm
    jg=math.hypot(*(a-b for a,b in zip(jf,j0)))/jnorm
    study['refinement'].update(total_impulse_time_half_relative_change=jt,total_impulse_grid_half_relative_change=jg,
        total_impulse_time_passed=jt<.03,total_impulse_grid_passed=jg<.03,
        dynamic_time_passed=jt<.03 and study['refinement']['time_passed'],dynamic_grid_passed=jg<.03 and study['refinement']['grid_passed'])
    (OUT/'study.json').write_text(json.dumps(study,indent=2)+'\n');print(json.dumps(study['refinement'],indent=2))
if __name__=='__main__':main()
