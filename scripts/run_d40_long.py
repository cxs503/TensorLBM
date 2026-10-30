#!/usr/bin/env python3
"""Long-running D40 SDAA simulation (2000 steps)."""
import sys, time, json
sys.path.insert(0, '/root/tp-agent/work/m363/src')
from tensorlbm.general_sim import *

config = GeneralSimConfig(
    name='sphere_re100_d40_sdaa_long',
    geometry=GeometryConfig(
        source=GeometrySource.PARAMETRIC_SPHERE,
        sphere_radius=0.5, sphere_center=(0.0, 0.0, 0.0),
    ),
    physics=PhysicsConfig(
        density=1000.0, viscosity=1.0e-6,
        inlet_velocity=1.0e-4, reference_length=1.0,
    ),
    solver=SolverConfig(
        lattice=LatticeModel.D3Q19,
        collision=CollisionModel.BGK,
        resolution=40,
        domain_padding=(2.0, 2.0, 2.0, 2.0, 2.0, 3.0),
        max_steps=2000,
        warmup_steps=None,
        snapshot_interval=50000,
        force_sample_interval=10,
        device='sdaa:0',
        wall_treatment=WallTreatment.AUTO,
        force_method=ForceMethod.PRESSURE_FRICTION,
        pressure_extrap='none',
        p0_method='near_wall',
        friction_formula='standard',
        mass_correction=True,
        mass_correction_interval=200,
    ),
    output=OutputConfig(
        directory='/tmp/sphere_re100_d40_long',
        formats=[], save_macroscopic=False, save_forces=True,
    ),
)
engine = GeneralSimEngine(config)
setup_info = engine.setup()
t0 = time.time()
run_info = engine.run(steps=2000)
t_run = time.time()-t0
res = engine.results()
cd = res['Cd_Cl']
forces = engine.forces_log
result = {
    'setup': {k: setup_info[k] for k in ['Re','tau','u_lb','domain_lu','total_cells','device']},
    'run': {'steps': run_info['steps'], 'diverged': run_info['diverged'], 't_run_s': round(t_run,1), 'ms_per_step': round(t_run/max(run_info['steps'],1)*1000,1)},
    'Cd_total': cd.get('Cd_total'),
    'Cd_pressure': cd.get('Cd_pressure'),
    'Cd_friction': cd.get('Cd_friction'),
    'forces_log': forces,
}
with open('/root/tp-agent/work/m363/artifacts/sdaa_d40_2000steps.json', 'w') as f:
    json.dump(result, f)
print('DONE')
