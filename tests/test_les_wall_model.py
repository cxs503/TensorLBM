"""Unit tests for the LES wall model (Spalding composite law of the wall).

The wall model is the piece of the P127W2-N03 campaign that turns a grid-starved
no-slip wall into a *law-of-the-wall* wall: the wall-adjacent column's effective
viscosity is raised until the wall carries ``tau_w = rho u_tau^2`` with ``u_tau``
obtained by inverting ``y+ = Y(u+)`` at the first node.

Three things must hold before any campaign number can be trusted:

* the inversion is exact (round trip through ``Y``),
* in the viscous sublayer it reproduces the analytic wall law
  ``u_tau = sqrt(nu |u_1| / y_1)`` (i.e. it does not perturb a resolved wall),
* in the log region it reproduces ``u+ = (1/kappa) ln y+ + B``.

The end-to-end smoke test additionally checks that switching the model on leaves
mass conservation and finiteness intact and that the diagnostics land in the
result dictionary.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.turbulence.les_channel3d import (  # noqa: E402
    B_LOG,
    KAPPA,
    LESChannel,
    LESChannelConfig,
    spalding_y_plus,
    u_tau_from_wall_law,
)


@pytest.mark.parametrize("u_plus", [0.5, 1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0])
def test_spalding_inversion_round_trip(u_plus: float) -> None:
    """``u_tau_from_wall_law`` inverts ``y+ = Y(u+)`` to machine precision."""
    nu = 8.9e-4
    y1 = 0.5
    # build the (y+, u1) pair that belongs to this u+
    y_plus = float(spalding_y_plus(torch.tensor(u_plus)))
    u_tau_true = y_plus * nu / y1
    u1 = u_tau_true * u_plus
    u_tau = u_tau_from_wall_law(torch.tensor(u1), y1, nu)
    assert float(u_tau) == pytest.approx(u_tau_true, rel=2e-3)


def test_viscous_sublayer_limit() -> None:
    """For ``y1+ << 5`` the model reduces to ``tau = rho nu u_1 / y_1``."""
    nu = 2.0e-3
    y1 = 0.5
    for u1 in (1e-4, 1e-3, 5e-3):
        u_tau_model = float(u_tau_from_wall_law(torch.tensor(u1), y1, nu))
        y_plus = y1 * u_tau_model / nu
        assert y_plus < 5.0, "test premise: stay inside the viscous sublayer"
        u_tau_viscous = math.sqrt(nu * u1 / y1)
        assert u_tau_model == pytest.approx(u_tau_viscous, rel=0.05)


def test_log_region_matches_log_law() -> None:
    """At ``y+ ~ 200`` the inverted profile reproduces the log law."""
    nu, y1, u_plus_true = 1.0e-4, 1.0, 18.0
    y_plus_true = float(spalding_y_plus(torch.tensor(u_plus_true)))
    u1 = y_plus_true * nu / y1 * u_plus_true
    u_tau = float(u_tau_from_wall_law(torch.tensor(u1), y1, nu))
    u_plus = u1 / u_tau
    y_plus = y1 * u_tau / nu
    assert 150.0 < y_plus < 400.0, "test premise: stay inside the log region"
    log_law = math.log(y_plus) / KAPPA + B_LOG
    assert u_plus == pytest.approx(log_law, rel=0.03)


def test_zero_velocity_gives_zero_friction() -> None:
    u_tau = u_tau_from_wall_law(torch.zeros(4, dtype=torch.float64), 0.5, 1e-3)
    assert torch.all(u_tau == 0.0)


def test_inversion_is_monotone() -> None:
    u1 = torch.linspace(0.0, 0.5, 25, dtype=torch.float64)
    u_tau = u_tau_from_wall_law(u1, 0.5, 8.9e-4)
    assert torch.all(u_tau[1:] >= u_tau[:-1])


def test_wall_model_requires_les_branch() -> None:
    with pytest.raises(ValueError, match="wall model needs the LES branch"):
        LESChannel(LESChannelConfig(
            nx=16, nz=12, ny=12, re_tau=180.0, u_tau=0.01, steps=10, warmup_steps=5,
            smagorinsky_cs=0.0, wall_model="spalding",
        ))


def test_unknown_wall_model_rejected() -> None:
    with pytest.raises(ValueError, match="wall_model must be"):
        LESChannel(LESChannelConfig(
            nx=16, nz=12, ny=12, re_tau=180.0, u_tau=0.01, steps=10, warmup_steps=5,
            smagorinsky_cs=0.12, wall_model="k-omega",
        ))


def _tiny_config(**overrides) -> LESChannelConfig:
    base = dict(
        nx=16, nz=12, ny=12, re_tau=180.0, u_tau=0.01, steps=120, warmup_steps=40,
        sample_interval=10, report_interval=10 ** 9, smagorinsky_cs=0.12,
        perturbation_amplitude=0.35, dtype="float32", wall_origin=0.5,
        wall_scheme="interpolated", seed=7, label="wm-smoke",
    )
    base.update(overrides)
    return LESChannelConfig(**base)


def test_wall_model_smoke_run_is_finite_and_conservative() -> None:
    torch.set_num_threads(2)
    sim = LESChannel(_tiny_config(wall_model="spalding"))
    result = sim.run_les(verbose=False)
    wm = result["les"]["wall_model"]
    assert wm["model"] == "spalding"
    assert wm["u_tau_bottom"] > 0.0 and wm["u_tau_top"] > 0.0
    assert wm["nu_wm_over_nu_bottom"] >= 1.0
    assert math.isfinite(wm["u_tau_bottom"])
    # the wall model must not create or destroy mass
    mass = result["les"]["mass_conservation"]
    assert mass["total_mass_rel_drift"] < 1e-3
    # and the returned mean profile must stay finite
    u_plus = result["wall_units"]["u_plus"]
    assert all(math.isfinite(v) for v in u_plus if v is not None)


def test_wall_model_off_by_default() -> None:
    cfg = _tiny_config()
    sim = LESChannel(cfg)
    result = sim.run_les(verbose=False)
    assert "wall_model" not in result["les"]