"""Prototype 2: growth rate of the module with the EXACT variable-coefficient solve
vs the round-2 first-principles anchor."""
import sys, time
import numpy as np
import torch

sys.path.insert(0, "/srv/tensor/agent/work/m21247/src")
sys.path.insert(0, "/srv/tensor/agent/work/m21247/xval/XVAL-LBM-SPECTRAL-SAFFMAN-TAYLOR-INTEGRATING-FACT")
from tensorlbm.spectral_saffman_taylor import SpectralSaffmanTaylor2D
from proto_varcoef import gmres, apply_A, apply_Pinv

torch.set_num_threads(8)
sys.path.insert(0, "/srv/tensor/agent/work/m21247/_scratch")

# reuse the R2 anchor
import importlib.util
spec = importlib.util.spec_from_file_location(
    "xval", "/srv/tensor/agent/work/m21247/xval/XVAL-LBM-SPECTRAL-SAFFMAN-TAYLOR-INTEGRATING-FACT/xval_linear_theory.py")
xval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xval)

NY, NX, ASPECT, PE, MODE, W, DT = 128, 256, 2.0, 500.0, 4, 4.0, 0.1


def patch(sim, tol=1e-10, restart=30):
    psi_holder = {"psi": None}

    def _update_flow():
        b = -sim.viscosity_ratio * sim.c_y.real
        psi, it, res = gmres(sim, b, tol=tol, restart=restart)
        ph = torch.fft.fft2(psi)
        sim.phi_x = torch.fft.ifft2(ph * sim._km * 1j).real
        sim.phi_y = torch.fft.ifft2(ph * sim._kn * 1j).real
        psi_holder["last"] = (it, res)
    sim._update_flow = _update_flow
    return sim


def measure(R, ny=NY, nx=NX, dt=DT, t_fit=(2.0, 8.0)):
    sim = patch(SpectralSaffmanTaylor2D(ny, nx, aspect=ASPECT, pe=PE, viscosity_ratio=R,
                                        dt=dt, mode_index=MODE))
    sim.set_concentration(sim.single_mode_ic(x0=0.5 * ASPECT * PE, width=W,
                                             amplitude=1e-3, n_mode=MODE))
    ts, amps = [], []
    t0 = time.time()
    for _ in range(int(round(t_fit[1] / dt)) + 1):
        d = sim.diagnostics()
        ts.append(d.t); amps.append(d.mode_amplitude)
        sim.step()
    ts, amps = np.array(ts), np.array(amps)
    m = (ts >= t_fit[0]) & (ts <= t_fit[1]) & (amps > 0)
    slope = float(np.polyfit(ts[m], np.log(amps[m]), 1)[0])
    return slope, time.time() - t0, ts, amps


if __name__ == "__main__":
    k = 2 * np.pi * MODE / PE
    for R in (0.0, -1.0, 1.0, -2.0, 2.0):
        sig, el, ts, amps = measure(R)
        th = xval.exact_linear_sigma(k, R, W)["sigma"].real
        rel = abs(sig - th) / max(abs(th), 1e-30) * 100
        print(f"R={R:+.1f}  sigma_module(exact)={sig:+.6e}  sigma_theory={th:+.6e}  "
              f"rel={rel:6.2f}%  ({el:.1f}s)")