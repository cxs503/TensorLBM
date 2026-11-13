"""Pin the m18372 single-component Poiseuille force anchor (XVAL vs FluidX3D).

Guards (a) the reference-profile symmetry fix in
``tensorlbm.channel_kernel_numpy.poiseuille_exact``, (b) that the anchor's
repo-path driver reproduces the repository's own operator sequence bit-exactly,
(c) that the FluidX3D-semantics transplant recovers the analytic Poiseuille
profile, and (d) the anchor finding that the OSS half-force forcing convention
and the repo full-impulse convention coincide for a steady uniform body force.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ANCHOR = REPO / "xval" / "XVAL-LBM-SINGLE-COMPONENT-POISEUILLE-FORCE-ANCHOR"
sys.path.insert(0, str(ANCHOR))
sys.path.insert(0, str(REPO / "src"))

import poiseuille_kernels as K  # noqa: E402
import tensorlbm.channel_kernel_numpy as ck  # noqa: E402


def test_reference_profile_is_symmetric_about_midplane():
    ny, nu, force = 32, 0.1, 1.7777777777777783e-05
    y, u = K.analytic_profile(ny, nu, force)
    assert np.allclose(u, u[::-1], rtol=1e-13, atol=0)  # mirror symmetry
    _y2, u2 = ck.poiseuille_exact(ny, nu, force, 0.5)  # repo helper (fixed form)
    assert np.allclose(u, u2, rtol=1e-13, atol=0)
    # wall values must vanish at both bounce-back planes y=0.5 and y=ny-1.5
    lower, upper = K.analytic_walls(ny, 0.5)
    assert u[0] / (force / (2 * nu)) < (lower + 1 - lower) * (upper - 1) + 1e-9
    # the pre-fix form put the far zero at the solid node y = ny-1 -> +3.3 % u_max
    s = y - 0.5
    buggy = force / (2 * nu) * s * ((ny - 2 + 1.0) - 0.5 - s)
    assert buggy.max() / u.max() > 1.03


def test_repo_path_driver_is_bit_identical_to_repo_step():
    ny, nx, nz, tau = 16, 2, 2, 0.8
    force = 8 * 0.1 * 0.02 / 196
    solid = np.zeros((nx, ny, nz), dtype=bool)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    zeros = np.zeros((nx, ny, nz))
    f0 = ck.equilibrium(np.ones((nx, ny, nz)), zeros, zeros, zeros)
    cfg = ck.ChannelKernelConfig(nx=nx, ny=ny, nz=nz, collision="bgk", initialization="laminar")
    u_tau = np.sqrt(force * (ny - 2) / 2)
    object.__setattr__(cfg, "u_tau", u_tau)
    object.__setattr__(cfg, "re_tau", u_tau * (0.5 * (ny - 2)) / 0.1)
    state = ck.ChannelKernelState(f=f0.copy(), config=cfg, solid=solid)
    f = f0.copy()
    for _ in range(20):
        ck.step(state)
        f = K.repo_step(f, solid, tau, force, forcing="repo")
    assert np.max(np.abs(state.f - f)) == 0.0


def test_oss_transplant_recovers_analytic_poiseuille():
    ny, tau = 16, 0.8
    nu = (tau - 0.5) / 3.0
    force = 8 * nu * 0.02 / (ny - 2) ** 2
    rec = K.simulate_oss(ny, tau, force, nx=4, steps=40000, tol=1e-9, check_every=500)
    met = K.evaluate(rec)
    assert met["rel_err_u_max_pct"] < 2.0
    assert abs(met["umax_over_umean"] - 1.5) < 0.02
    assert met["rho_deviation_max"] < 1e-12
    assert met["u_cross_max"] < 1e-12


def test_forcing_convention_is_immaterial_for_steady_uniform_force():
    ny, tau = 16, 0.8
    nu = (tau - 0.5) / 3.0
    force = 8 * nu * 0.02 / (ny - 2) ** 2
    a = K.simulate_repo(ny, tau, force, forcing="repo", steps=40000, tol=1e-9, check_every=500)
    b = K.simulate_repo(ny, tau, force, forcing="oss", steps=40000, tol=1e-9, check_every=500)
    ua, ub = np.asarray(a["u_phys"]), np.asarray(b["u_phys"])
    assert np.max(np.abs(ua - ub)) / np.max(np.abs(ua)) < 1e-4
    for rec in (a, b):
        assert K.evaluate(rec)["rel_err_u_max_pct"] < 2.0
