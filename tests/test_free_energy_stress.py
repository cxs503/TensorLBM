"""XVAL regression: ludwig pressure-tensor body force reproduces the analytic
surface tension and a vanishing equilibrium body force (<15% criterion)."""
import math
import torch

from tensorlbm.free_energy_stress import (
    FreeEnergyParams, grad, interface_width, planar_profile, droplet_profile,
    stress_divergence_force, grad_mu_force, surface_tension_analytic, pressure_tensor,
)

torch.set_default_dtype(torch.float64)


def test_surface_tension_from_pressure_tensor():
    for A, kappa in ((1.0, 2.0), (1.0, 8.0), (1.0, 32.0)):
        p = FreeEnergyParams(A, 1.0, kappa)
        L, xi = 256, interface_width(p)
        phi = planar_profile(L, L, p)
        gx, _ = grad(phi, "iso9")
        sl = slice(L // 2 - int(8 * xi), L // 2 + int(8 * xi))
        sig = float(p.kappa * (gx[0, sl] ** 2).sum())
        ref = surface_tension_analytic(p)
        assert abs(sig - ref) / ref < 0.15, (kappa, sig, ref)


def test_force_vanishes_on_equilibrium_profile():
    p = FreeEnergyParams(1.0, 1.0, 8.0)
    L, xi = 128, interface_width(p)
    phi = planar_profile(L, L, p)
    sl = slice(L // 2 - int(8 * xi), L // 2 + int(8 * xi))
    fscale = surface_tension_analytic(p) / xi ** 2
    for F in (stress_divergence_force(phi, p, "iso9"), grad_mu_force(phi, p, "iso9")):
        assert max(float(f[0, sl].abs().max()) for f in F) / fscale < 0.15


def test_curved_interface_surface_tension():
    p = FreeEnergyParams(1.0, 1.0, 8.0)
    L, R = 128, 32
    phi = droplet_profile(L, L, p, radius=R)
    gx, _ = grad(phi, "iso9")
    c = L // 2
    sl = slice(c + 1, c + R + int(6 * interface_width(p)))
    sig = float(p.kappa * (gx[c, sl] ** 2).sum())
    assert abs(sig - surface_tension_analytic(p)) / surface_tension_analytic(p) < 0.15
