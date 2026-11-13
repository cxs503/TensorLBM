#!/usr/bin/env python3
"""Diagnostic velocity-field snapshot from the delivered coupled solver (m18749).

1500 steps at the delivered Re=20 configuration; used only for the qualitative
cloud map in the audit report (the run is NOT converged -- see verify_output.json
C_repro_rerun).
"""
import json, numpy as np, torch
torch.set_num_threads(4)
from tensorlbm.verification.cylinder_flow import CylinderFlowConfig, CylinderFlowSolver

cfg = CylinderFlowConfig(nx=320, ny=160, max_steps=1500, sample_interval=50)
s = CylinderFlowSolver(cfg)
s.run()
ux, uy = s.get_velocity_field()
np.savez('/root/tp-agent/work/m18779/artifacts/m18779/field_1500steps.npz', ux=ux, uy=uy,
         r=cfg.cylinder_radius, cx=cfg.cylinder_x, cy=cfg.cylinder_y, u_inlet=cfg.u_inlet)
json.dump(s.force_history, open('/root/tp-agent/work/m18779/artifacts/m18779/field_run_force_history.json', 'w'))
print('saved', ux.shape, 'last Cd', s.force_history[-1]['Cd'])
