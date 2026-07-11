"""Regression coverage for propeller_improved's BFL startup path."""

import importlib.util
from pathlib import Path

import torch


_SPEC = importlib.util.spec_from_file_location(
    "propeller_improved",
    Path(__file__).parents[1] / "examples" / "propeller_improved.py",
)
_PROPELLER = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_PROPELLER)


def test_bfl_startup_has_links_and_keeps_macroscopic_fields_finite():
    """One CPU step exercises a crossing BFL link without rho_local NameError."""
    state = _PROPELLER.run_improved_propeller(
        nx=64,
        ny=64,
        nz=64,
        n_steps=1,
        device="cpu",
        mask_interval_deg=90.0,
        use_bfl=True,
    )

    assert state["bfl_link_count"] > 0
    for name in ("f", "rho", "ux", "uy", "uz", "me_force", "me_torque"):
        assert torch.isfinite(state[name]).all(), name


def test_momentum_exchange_diagnostic_does_not_change_bfl_evolution():
    kwargs = dict(
        nx=64, ny=64, nz=64, n_steps=1, device="cpu",
        mask_interval_deg=90.0, use_bfl=True,
    )
    without_diagnostic = _PROPELLER.run_improved_propeller(
        **kwargs, collect_me_diagnostic=False,
    )
    with_diagnostic = _PROPELLER.run_improved_propeller(
        **kwargs, collect_me_diagnostic=True,
    )

    assert torch.equal(without_diagnostic["f"], with_diagnostic["f"])
    assert torch.equal(without_diagnostic["rho"], with_diagnostic["rho"])
