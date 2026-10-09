"""Run actual solvers, save complete JSON fields/restarts and evidence hashes."""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from tensorlbm.ice_coupling_2d import CoupledIce2D,CoupledIceConfig


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='docs/assets/ice-coupling')
    parser.add_argument('--duration',type=float,default=.15)
    args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    c=CoupledIceConfig(duration_s=args.duration,exchange_steps=2)
    cases={'dry':replace(c,wet=False),'wet':c,
        'time_half':replace(c,fluid_dt_s=c.fluid_dt_s/2,exchange_steps=4),
        'exchange_half':replace(c,exchange_steps=1),
        'particle_fine':replace(c,ice_nx=18,ice_ny=4,ice_radius_m=.0125),
        'intact_control':replace(c,breaking_strain=10.,shear_breaking_strain=10.)}
    import tensordem.dem as demmodule
    import tensorlbm.ice_coupling_2d as coupler
    import tensorlbm.solver as solver
    files=[Path(demmodule.__file__),Path(coupler.__file__),Path(solver.__file__),Path(__file__),
        Path(demmodule.__file__).with_name('hull_contact.py'),
        Path(coupler.__file__).with_name('d2q9.py'),Path(coupler.__file__).with_name('icebreaking.py')]
    versions={p.parent.parent.name+'/'+p.name:sha(p) for p in files}
    record={'schema':'tensorlbm.dem-coupling-study/1','source_sha256':versions,
        'scope':'single-phase periodic liquid; slipping point IBM; no gravity/free surface/ice calibration',
        'physical_accuracy_pass':False,'cases':{},'artifacts_sha256':{}}
    for name,config in cases.items():
        sim=CoupledIce2D(config)
        count=round(config.duration_s/config.fluid_dt_s)
        for _ in range(count):sim.step()
        h=sim.history
        peak=max(abs(v['total_fy_n']) for v in h)
        result={'config':asdict(config),'steps':count,'dem_substeps':sim.substeps,
            'peak_total_tool_force_N':peak,
            'tool_impulse_Ns':sum(v['total_fy_n']*config.fluid_dt_s for v in h),
            'broken_bonds':sim.dem.broken_bonds,'initial_bonds':int(sim.dem.bonded.sum()),
            'fragment_count':len(sim.dem.fragments()),'fragments':sim.dem.fragments(),
            'physical_ice_mass_kg':len(sim.dem.positions)*sim.dem.config.mass,
            'max_momentum_residual_kg_m_s':max(max(abs(x) for x in v['momentum_residual_kg_m_s']) for v in h),
            'max_mass_relative_error':max(v['mass_relative_error'] for v in h),
            'max_force_mapping_error_N':max(v['force_mapping_error_N'] for v in h),
            'max_moment_mapping_error_Nm':max(v['moment_mapping_error_Nm'] for v in h),
            'max_adjoint_power_error_W':max(v['adjoint_power_error_W'] for v in h),
            'final_dem_energy_residual_J':h[-1]['dem_discrete_energy_residual_J'],
            'final_staggered_interface_work_residual_J':h[-1]['staggered_interface_work_residual_J'],
            'max_marker_slip_rms_m_s':max(v['marker_slip_rms_m_s'] for v in h),
            'final_fracture_release_J':h[-1]['fracture_release_J']}
        result['interface_verification_pass']=all((result['max_momentum_residual_kg_m_s']<1e-9,
            result['max_mass_relative_error']<1e-11,result['max_force_mapping_error_N']<1e-11,
            result['max_moment_mapping_error_Nm']<1e-11,result['max_adjoint_power_error_W']<1e-11))
        record['cases'][name]=result
        path=out/(name+'.json')
        path.write_text(json.dumps({'summary':result,'state':sim.snapshot()},indent=2,allow_nan=False)+'\n')
        record['artifacts_sha256'][path.name]=sha(path)
        print(name,json.dumps(result),flush=True)
    base=record['cases']['wet']
    record['sensitivity']={name:{key:abs(record['cases'][name][key]/base[key]-1)
        for key in ('peak_total_tool_force_N','tool_impulse_Ns')}
        for name in ('time_half','exchange_half','particle_fine')}
    record['sensitivity_pass_3percent']=all(v<.03 for s in record['sensitivity'].values() for v in s.values())
    (out/'study.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print('study',record['sensitivity'])

if __name__=='__main__':main()
