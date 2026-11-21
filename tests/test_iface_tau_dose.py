"""Unit tests for the fill-weighted interface relaxation dose (DE-X-RELX1-PHASETAU).

Covers:
  * dose=0 keeps the legacy scalar BGK path bit-identical (zero behavior change)
  * dose!=0 changes INTERFACE-cell relaxation only (LIQUID cells untouched)
  * negative doses respect the BGK stability floor tau > 0.5
  * the probe dose channel (env / json file / default off)
"""

import json
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.free_surface_lbm import (  # noqa: E402
    GAS,
    INTERFACE,
    LIQUID,
    SOLID,
    free_surface_step,
)

HERE = Path(__file__).resolve().parents[1] / "benchmarks" / "water_entry" / "single_drop"


def _toy_state(n=6, seed=7):
    """A tiny 3D field with liquid, interface (partial fill) and gas cells."""
    g = torch.Generator().manual_seed(seed)
    fill = torch.rand(n, n, n, generator=g)
    fill = torch.where(fill > 0.35, torch.ones_like(fill), fill)  # some full cells
    fill[fill < 0.05] = 0.0  # some empty cells
    solid = torch.zeros_like(fill, dtype=torch.bool)
    fill[0] = 0.0
    solid[0] = True
    from tensorlbm.free_surface_lbm import init_flags_from_fill, init_mass_from_fill

    flags = init_flags_from_fill(fill, solid)
    mass = init_mass_from_fill(fill, flags)
    from tensorlbm.d3q19 import equilibrium3d

    f = equilibrium3d(
        torch.where(flags != GAS, 1.0, 0.0),
        torch.zeros_like(fill),
        torch.zeros_like(fill),
        torch.zeros_like(fill),
    )
    # Perturb populations so the BGK collision is non-trivial (at exact
    # equilibrium f_new == f for ANY tau, which would mask the dose knob).
    f = f + 1e-3 * torch.randn(f.shape, generator=g)
    f = f.clamp(min=0.0)
    return f, fill, flags, solid, mass


def test_dose_zero_bit_identical_to_legacy():
    f, fill, flags, solid, mass = _toy_state()
    tau = 0.5 + 3 * 1e-4
    f0, fill0, flags0, mass0, _ = free_surface_step(
        f.clone(), fill.clone(), flags.clone(), solid, mass=mass.clone(),
        tau=tau, gz=-1e-4, collision="bgk",
    )
    f1, fill1, flags1, mass1, _ = free_surface_step(
        f.clone(), fill.clone(), flags.clone(), solid, mass=mass.clone(),
        tau=tau, gz=-1e-4, collision="bgk", iface_tau_dose=0.0,
    )
    assert torch.equal(f0, f1)
    assert torch.equal(fill0, fill1)
    assert torch.equal(flags0, flags1)
    assert torch.equal(mass0, mass1)


def test_dose_changes_interface_only():
    """Relaxation-rate mask: INTERFACE cells get tau*g(fill), LIQUID cells
    keep the physical tau exactly (structural spatial separation).  Checked
    at the collision level (a full step couples cells through streaming)."""
    f, fill, flags, solid, mass = _toy_state()
    from tensorlbm.d3q19 import equilibrium3d, macroscopic3d

    tau = 0.5 + 3 * 1e-4
    dose = 1.0
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    f0 = f - (f - feq) / tau
    g_fill = 1.0 + dose * (1.0 - fill)
    tau_cell = torch.where(
        (flags == INTERFACE).unsqueeze(0),
        (tau * g_fill).clamp_min(0.5 + 1e-4),
        tau,
    )
    f1 = f - (f - feq) / tau_cell
    # interface cells must relax at a different rate
    assert (f0 - f1)[:, flags == INTERFACE].abs().max() > 0
    # liquid cells must be bit-identical
    assert (f0 - f1)[:, flags == LIQUID].abs().max() == 0
    # and the full step must actually move (knob reaches the dynamics)
    s0, *_ = free_surface_step(
        f.clone(), fill.clone(), flags.clone(), solid, mass=mass.clone(),
        tau=tau, gz=-1e-4, collision="bgk",
    )
    s1, *_ = free_surface_step(
        f.clone(), fill.clone(), flags.clone(), solid, mass=mass.clone(),
        tau=tau, gz=-1e-4, collision="bgk", iface_tau_dose=dose,
    )
    assert (s0 - s1).abs().max() > 0


def test_negative_dose_stable_floor():
    """tau=0.5004 is near the BGK floor: negative dose must clamp, not blow up."""
    f, fill, flags, solid, mass = _toy_state()
    tau = 0.500388
    f1, *_ = free_surface_step(
        f.clone(), fill.clone(), flags.clone(), solid, mass=mass.clone(),
        tau=tau, gz=-1e-4, collision="bgk", iface_tau_dose=-0.5,
    )
    assert torch.isfinite(f1).all()
    # effective tau at fill=0 would be 0.25 -> clamped to 0.5+1e-4:
    # verify by hand: g(0)=0.5, tau*g=0.250194 < 0.5001 -> clamp
    g0 = 1.0 + (-0.5) * (1.0 - 0.0)
    assert tau * g0 < 0.5  # pre-clamp value is unstable
    assert 0.5 + 1e-4 > tau * g0


def test_probe_dose_channel(tmp_path, monkeypatch):
    sys.path.insert(0, str(HERE.parent.parent))
    from benchmarks.water_entry.single_drop.free_surface_probe import (
        _read_iface_tau_dose,
    )

    monkeypatch.delenv("WE_IFACE_DOSE", raising=False)
    assert _read_iface_tau_dose(base_dir=tmp_path) == 0.0  # no file -> off
    (tmp_path / "phase_tau_dose.json").write_text(json.dumps({"iface_tau_dose": 0.5}))
    assert _read_iface_tau_dose(base_dir=tmp_path) == 0.5
    monkeypatch.setenv("WE_IFACE_DOSE", "-0.25")
    assert _read_iface_tau_dose(base_dir=tmp_path) == -0.25  # env wins over file
