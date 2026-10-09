"""Reconstruct final fields and force/energy ledgers, never trust stored flags."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import torch
from tensorlbm.ice_coupling_2d import CoupledIce2D,cross2


def close(a,b,atol=1e-10):
    if not math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=atol):
        raise AssertionError((a,b))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',default='docs/assets/ice-coupling')
    args=parser.parse_args();root=Path(args.directory)
    study=json.loads((root/'study.json').read_text())
    import tensordem.dem as demmodule
    import tensorlbm.ice_coupling_2d as coupledmodule
    sources={
        'src/dem.py':Path(demmodule.__file__),
        'src/hull_contact.py':Path(demmodule.__file__).with_name('hull_contact.py'),
        'examples/run_benchmark.py':Path(__file__).with_name('run_benchmark.py')}
    for filename in ('ice_coupling_2d.py','solver.py','d2q9.py','icebreaking.py'):
        sources['src/'+filename]=Path(coupledmodule.__file__).with_name(filename)
    for name,digest in study['source_sha256'].items():
        assert hashlib.sha256(sources[name].read_bytes()).hexdigest()==digest, 'source changed: '+name
    for name,sha in study['artifacts_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
        data=json.loads((root/name).read_text());summary=data['summary'];s=CoupledIce2D.from_snapshot(data['state'])
        h=s.history[-1];n=len(s.dem.positions)
        rho=s.f.sum(0).reshape(-1);gridu=s.fluid_velocity()
        p=s.cell_mass*(rho[:,None]*gridu).sum(0)+s.dem.config.mass*s.dem.velocities.sum(0)
        torch.testing.assert_close(p-s.total_initial_momentum-s.external_impulse,
            torch.tensor(h['momentum_residual_kg_m_s'],dtype=torch.float64),rtol=0,atol=1e-12)
        close(.5*s.cell_mass*(rho[:,None]*gridu.square()).sum(),h['fluid_kinetic_J'])
        close(s.dem.mechanical_energy()['mechanical_J'],h['dem_mechanical_J'])
        close(s.dem.config.mass*n,summary['physical_ice_mass_kg'])
        close(int((s.dem.bonded & ~s.dem.alive).sum()),summary['broken_bonds'])
        assert s.dem.fragments()==summary['fragments']
        close(sum(f['mass_kg'] for f in summary['fragments']),summary['physical_ice_mass_kg'])
        a=s.dem.accounting
        demres=h['dem_mechanical_J']+a['damping_dissipation_J']+a['fracture_release_J']-s.initial_dem_energy-a['external_work_J']-a['tool_work_J']
        close(demres,h['dem_discrete_energy_residual_J'])
        interface=s.fluid_force_work+a['external_work_J']-s.tool_fluid_work+s.regularization_dissipation
        close(interface,h['staggered_interface_work_residual_J'])
        total=h['dem_mechanical_J']+h['fluid_kinetic_J']+a['damping_dissipation_J']+a['fracture_release_J']+s.fluid_collision_loss+s.regularization_dissipation-s.initial_dem_energy-s.initial_fluid_energy-a['tool_work_J']-s.tool_fluid_work
        close(total,demres+interface)
        close((s.fluid_grid_force.sum(0)-s.marker_force.sum(0)).norm(),h['force_mapping_error_N'])
        close(abs(cross2(s.grid_xy,s.fluid_grid_force).sum()-cross2(s.marker_positions,s.marker_force).sum()),h['moment_mapping_error_Nm'])
        close(h['total_fy_n'],h['fluid_fy_n']+h['ice_contact_fy_n']+h['other_fy_n'])
        for row in s.history:
            assert row['force_hold_age_s']>=-1e-10
            assert row['force_hold_age_s']<s.config.exchange_steps*s.config.fluid_dt_s+1e-10
            close(row['total_fy_n'],row['fluid_fy_n']+row['ice_contact_fy_n'])
        close(max(abs(row['total_fy_n']) for row in s.history),summary['peak_total_tool_force_N'])
        close(sum(row['total_fy_n']*s.config.fluid_dt_s for row in s.history),summary['tool_impulse_Ns'])
        assert summary['max_momentum_residual_kg_m_s']<1e-9
        print(name,'hash, final fields, momentum, energy decomposition, topology and time alignment: PASS')
    print('Physical accuracy remains unqualified; particle sensitivity fails 3%.')

if __name__=='__main__':main()
