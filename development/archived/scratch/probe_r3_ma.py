"""Probe: how does the steady-state error of the periodic-x + dP/dx channel
depend on the Mach number (u_max) at *fixed* grid?  Needed before declaring a
grid-convergence order."""
import sys, time
import numpy as np

sys.path.insert(0, "src")
from tensorlbm.periodic_channel_dpdx import (
    DpdxChannel, WALL_HALFWAY_BB, WALL_REGULARISED,
    FORCE_SPLIT_GUO, FORCE_GUO_HALF_FROZEN,
)

TAU = 0.5375
NU = (TAU - 0.5) / 3.0

ARMS = {
    "PORT-REG": (WALL_REGULARISED, FORCE_GUO_HALF_FROZEN),
    "PORT-BB": (WALL_HALFWAY_BB, FORCE_SPLIT_GUO),
}

for ny in (17,):
    for ut in (0.0025, 0.005, 0.01, 0.02, 0.04):
        for name, (wall, fs) in ARMS.items():
            H = float(ny - 1) if wall == WALL_REGULARISED else float(ny - 2)
            g = 8.0 * NU * ut / H ** 2
            ch = DpdxChannel(nx=2, ny=ny, tau=TAU, force=g * 1.0, wall=wall, force_scheme=fs)
            t0 = time.time()
            info = ch.run_until_steady(tol=1e-9, max_steps=400000)
            u = ch.profile()
            s, ua = ch.analytic()
            l2 = float(np.linalg.norm(u - ua) / np.linalg.norm(ua))
            umax = float(np.max(np.abs(u)))
            print(f"{name} ny={ny} ut={ut:.4f} steps={info['steps']:7d} conv={info['converged']} "
                  f"umax={umax:.8f} err_umax={abs(umax/ut-1):.5f} L2={l2:.6f} "
                  f"umax/umean={umax/ (u.mean()):.5f} t={time.time()-t0:.1f}s")