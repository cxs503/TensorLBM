"""Integration test: the library entry point really drives the two-layer channel.

Ensures ``tensorlbm.cases.two_layer_poiseuille.run_two_layer_poiseuille`` is a
callable, non-placeholder solver path (L3 integration caller of the
GAP-LBM-MCMP-POISEUILLE benchmark), and that a short run reproduces the
two-layer Poiseuille flow rate to within a loose transient tolerance.
"""
from __future__ import annotations

import pytest

from tensorlbm.cases import run_two_layer_poiseuille
from cases.mcmp_stratified_poiseuille.analytic import layer_viscosities, layered_poiseuille


def test_entry_point_is_importable():
    assert callable(run_two_layer_poiseuille)


@pytest.mark.parametrize("sc_form", ["edm", "standard"])
def test_short_run_reproduces_flow_rate(sc_form):
    ny, steps = 25, 4000
    rec = run_two_layer_poiseuille(ny=ny, steps=steps, sc_form=sc_form)
    assert rec["finite"] is True
    nu_lo, nu_up = layer_viscosities(1.0, 0.75, 0.7, 0.3, "mixture")
    ana = layered_poiseuille(ny, (ny - 2) / 2.0, 5e-5, nu_lo, nu_up, H=ny - 2)
    err = (rec["Q_mix"] - ana["Q_total"]) / ana["Q_total"]
    # short transient run: the steady-state gate (10 %) belongs to the
    # benchmark; here we only require the solver to be on the right branch.
    assert abs(err) < 0.25, f"flow-rate error {err:.3%} at ny={ny} ({sc_form})"
    y_target = 0.5 + (ny - 2) / 2.0
    assert abs(rec["interface_crossing"] - y_target) / y_target < 0.10
