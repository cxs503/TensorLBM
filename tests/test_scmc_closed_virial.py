"""CASE-MCMP-LAPLACE round 3 — the closing virial (pairing="closed").

The Shan(2006) two-point tensor (pairing="sym") does not satisfy the static
flat-interface normal-stress conservation for this library's force format
(reverse gather + velocity shift): measured flatness ~45% at G12=-2.5.
The "closed" pairing (empirically calibrated, see
``tensorlbm.scmc_pressure_tensor``) restores flatness to <1% on the same
relaxed slab.  These tests lock that behaviour in with a small, fast slab.
"""
from __future__ import annotations

import torch

from tensorlbm.d2q9 import equilibrium
from tensorlbm.multiphase import collide_sc_two_component
from tensorlbm.scmc_pressure_tensor import (
    flat_interface_pressure_profile,
    pressure_consistency_report,
    sigma_eos_flat,
)
from tensorlbm.solver import stream

G12 = -2.5
TAU = 1.0
NY, NX = 48, 8
HALF, W0 = 12.0, 2.0
STEPS = 1500
# coexistence densities of the G12=-2.5 planar slab (n=96, 3000 steps):
R1L, R1G = 0.8818, 0.1371
R2L, R2G = 0.1367, 0.8824


def _relaxed_slab() -> tuple[torch.Tensor, torch.Tensor]:
    ys = torch.arange(NY, dtype=torch.float32).view(NY, 1)
    c = NY / 2.0
    s_in = (
        0.5 * (1.0 + torch.tanh((ys - (c - HALF)) / W0))
        * 0.5 * (1.0 - torch.tanh((ys - (c + HALF)) / W0))
    ).expand(NY, NX).clone()
    rho1 = R1G + s_in * (R1L - R1G)
    rho2 = R2G + s_in * (R2L - R2G)
    z = torch.zeros_like(rho1)
    f1 = equilibrium(rho1, z, z)
    f2 = equilibrium(rho2, z, z)
    for _ in range(STEPS):
        f1, f2 = collide_sc_two_component(f1, f2, G_12=G12, tau1=TAU, tau2=TAU)
        f1 = stream(f1)
        f2 = stream(f2)
    return f1, f2


def test_closed_pairing_flattens_normal_stress():
    f1, f2 = _relaxed_slab()
    prof_sym = flat_interface_pressure_profile(f1, f2, G12, pairing="sym")
    prof_closed = flat_interface_pressure_profile(f1, f2, G12, pairing="closed")
    # the closed tensor must pass the pre-declared 5% validity gate
    assert prof_closed["flatness"] < 0.05, prof_closed["flatness"]
    # and must be dramatically flatter than the Shan(2006) two-point form
    assert prof_closed["flatness"] < 0.25 * prof_sym["flatness"], (
        prof_closed["flatness"],
        prof_sym["flatness"],
    )
    # the closed tensor must give a positive mechanical surface tension
    assert prof_closed["sigma_flat"] > 0.0


def test_sigma_eos_flat_positive():
    f1, f2 = _relaxed_slab()
    out = sigma_eos_flat(f1, f2, G12, n_interfaces=2)
    assert out["sigma_eos"] > 0.0
    assert out["p_bulk"] > 0.0
    assert out["p_eos_dip"] > 0.0


def test_pressure_consistency_report_keys():
    f1, f2 = _relaxed_slab()
    rep = pressure_consistency_report(f1, f2, G12, n_interfaces=2, sigma_laplace=0.18777)
    for key in (
        "flatness_sym",
        "flatness_closed",
        "sigma_mech_sym",
        "sigma_mech_closed",
        "sigma_eos",
        "sigma_laplace",
        "ratio_mech_to_laplace",
        "ratio_eos_to_laplace",
        "closed_gate_pass",
    ):
        assert key in rep, key
    assert rep["closed_gate_pass"] is True
    # the three estimators are distinct quantities for this force format:
    # the mechanical (closed) sigma is far below the droplet Laplace sigma.
    assert rep["ratio_mech_to_laplace"] < 0.5
    # and the three estimators do not collapse onto one value (the absolute
    # sigma is not a single well-defined number for this force discretisation).
    sigmas = [rep["sigma_mech_closed"], rep["sigma_eos"], rep["sigma_laplace"]]
    assert max(sigmas) / min(sigmas) > 1.5
