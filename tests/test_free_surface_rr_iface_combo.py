"""DE-X-CMB8-IFACECOMBO: rr_iface x SMAG x HYDRO stack contracts.

The interface-local RR branch (DE-X-RELX3-RRIFACE, port ba9d05c74) is
extended with the CMB3 RR x SGS idiom: with ``C_s > 0`` the per-cell
Smagorinsky eddy-viscosity relaxation time ``tau_eff`` is handed to *both*
sub-branches (interface RR kernel and bulk BGK relaxation), so the SGS
dissipation acts on the mode RR retains at the interface and on the bulk
BGK mode alike.  ``C_s == 0`` keeps the ported branch bit-identical.

These tests pin the operator-level contract of the stack; the
problem-specific full-window behaviour is established by the cluster
runs (runs/cmb8_*).
"""

from __future__ import annotations

import torch

from tensorlbm.free_surface_lbm import (
    free_surface_step,
    init_flags_from_fill,
    init_mass_from_fill,
    total_liquid_inventory,
)


def _flat_pool_field(nz=12, ny=12, nx=12, level=8.5):
    z = torch.arange(nz).view(nz, 1, 1)
    fill = (level - z + 0.5).clamp(0, 1)  # flat pool, gas above
    fill = fill.expand(nz, ny, nx).contiguous()
    solid = torch.zeros_like(fill, dtype=torch.bool)
    solid[[0, -1]] = True
    fill[solid] = 0
    flags = init_flags_from_fill(fill, solid)
    rho = (1.0 + 3.0 * (0.02 * (level - z).clamp_min(0))).expand(nz, ny, nx)
    uz = (-0.05 * (z > level + 0.5)).expand(nz, ny, nx)
    fluid = (flags != 0) & (flags != 3)  # liquid + interface carry density
    from tensorlbm.d3q19 import equilibrium3d

    f = equilibrium3d(rho * fluid, torch.zeros_like(fill), torch.zeros_like(fill), uz)
    mass = init_mass_from_fill(fill, flags)
    return f, fill, flags, solid, mass


def _step(f, fill, flags, solid, mass, **kw):
    base = dict(
        mass=mass,
        tau=0.55,
        gz=-0.02,
        surface_tension=0.01,
        rho_gas=1.0,
    )
    base.update(kw)
    return free_surface_step(f, fill, flags, solid, **base)


def test_probe_cs_channel_cli_wins_over_env() -> None:
    """The probe's C_s resolution: explicit CLI value wins; None reads the
    TL_FS_CS environment channel (empty/absent = 0.0 = legacy path)."""
    import os

    import benchmarks.water_entry.single_drop.free_surface_probe as P

    os.environ.pop("TL_FS_CS", None)
    try:
        assert P._cs_from_cli(None) == 0.0
        os.environ["TL_FS_CS"] = "0.075"
        assert P._cs_from_cli(None) == 0.075
        assert P._cs_from_cli(0.1) == 0.1  # explicit CLI wins over env
        os.environ["TL_FS_CS"] = ""
        assert P._cs_from_cli(None) == 0.0  # empty env = off
    finally:
        os.environ.pop("TL_FS_CS", None)


def test_rr_iface_cs0_zero_behavior() -> None:
    """C_s == 0 on rr_iface: the arm runs, stays finite, and its bulk cells
    stay bit-identical to the plain BGK arm far from the interface (the
    ported branch is the C_s == 0 code path)."""
    f0, fill0, flags0, solid, mass0 = _flat_pool_field()
    outs = {}
    for mode in ("bgk", "rr_iface"):
        f, fill, flags, mass = f0.clone(), fill0.clone(), flags0.clone(), mass0.clone()
        for _ in range(3):
            f, fill, flags, mass, _ = _step(
                f, fill, flags, solid, mass, collision=mode, C_s=0.0
            )
        outs[mode] = f
    f_bgk, f_mix = outs["bgk"], outs["rr_iface"]
    nz = f0.shape[1]
    deep = torch.zeros(nz, dtype=torch.bool)
    deep[: int(8.5) - 5] = True
    assert torch.equal(f_bgk[:, deep], f_mix[:, deep])


def test_rr_iface_smag_stack_changes_trajectory() -> None:
    """C_s > 0 on rr_iface must move the trajectory off the C_s == 0 arm."""
    f0, fill0, flags0, solid, mass0 = _flat_pool_field()
    finals = {}
    for cs in (0.0, 0.075):
        f, fill, flags, mass = f0.clone(), fill0.clone(), flags0.clone(), mass0.clone()
        for _ in range(20):
            f, fill, flags, mass, _ = _step(
                f, fill, flags, solid, mass, collision="rr_iface", C_s=cs
            )
        assert torch.isfinite(f).all()
        finals[cs] = f
    assert not torch.equal(finals[0.0], finals[0.075])


def test_rr_iface_smag_bulk_matches_bgk_sgs() -> None:
    """Locality of the stack: bulk cells far from the interface in
    rr_iface+SMAG stay bit-identical to the plain BGK+SMAG arm, because both
    sub-branches share the same tau_eff field and the BGK sub-branch is the
    plain BGK+SGS expression; the interface perturbation cannot reach the
    deep liquid within 3 steps (lattice speed 1/3)."""
    f0, fill0, flags0, solid, mass0 = _flat_pool_field()
    outs = {}
    for mode in ("bgk", "rr_iface"):
        f, fill, flags, mass = f0.clone(), fill0.clone(), flags0.clone(), mass0.clone()
        for _ in range(3):
            f, fill, flags, mass, _ = _step(
                f, fill, flags, solid, mass, collision=mode, C_s=0.075
            )
        outs[mode] = f
    f_bgk, f_mix = outs["bgk"], outs["rr_iface"]
    nz = f0.shape[1]  # f is (19, nz, ny, nx)
    level = 8.5
    deep = torch.zeros(nz, dtype=torch.bool)
    deep[: int(level) - 5] = True  # deep liquid, >=5 cells below the interface
    assert deep.any()
    assert torch.equal(f_bgk[:, deep], f_mix[:, deep])


def test_rr_iface_smag_stack_finite_and_conserved() -> None:
    """The stacked arm steps a small field: finite populations, and the
    liquid inventory stays close to its initial value (no mass blow-up)."""
    f0, fill0, flags0, solid, mass0 = _flat_pool_field()
    inv0 = total_liquid_inventory(f0, fill0, flags0).item()
    f, fill, flags, mass = f0.clone(), fill0.clone(), flags0.clone(), mass0.clone()
    for _ in range(30):
        f, fill, flags, mass, _ = _step(
            f, fill, flags, solid, mass, collision="rr_iface", C_s=0.1
        )
    assert torch.isfinite(f).all()
    drift = abs(total_liquid_inventory(f, fill, flags).item() / inv0 - 1)
    assert drift < 1e-3
