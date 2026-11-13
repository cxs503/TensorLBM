"""Tests for the graph-unroll BPTT LES calibration
(``XVAL-LBM-GRAPH-UNROLL-BPTT-LES-PARAMETER``).

The three properties that the cross-validation depends on are asserted here in
miniature so a regression cannot silently invalidate the report:

* the differentiable chain reproduces the numpy reference kernel (bit-level
  agreement at ``Cs = 0``), i.e. ``graph_unroll`` is the same solver;
* the autograd (BPTT) gradient of the unrolled field loss matches a central
  finite difference;
* the closure prefactor is right: the realised eddy viscosity matches the
  analytic ``Cs^2 |S|`` for a linear shear probe.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.les_graph_unroll import reference2d as ref
from tensorlbm.les_graph_unroll.calibrate import (
    CalibProblem,
    autograd_gradient,
    bptt_calibrate,
    fd_gradient,
    field_loss,
    measure_eddy_viscosity,
)
from tensorlbm.les_graph_unroll.solver2d import macroscopic, unroll


@pytest.fixture(autouse=True)
def _single_thread():
    torch.set_num_threads(1)


@pytest.mark.unit
def test_unroll_matches_reference_kernel_at_zero_cs():
    """``Cs = 0`` unroll == numpy BGK reference run (same velocity/weights)."""
    f0, _, _ = ref.tgv_ic(24, u0=0.05)
    tau = ref.tau_for_re(24, 0.05, 300.0)
    f_ref, _ = ref.run_reference(f0, 40, tau)
    f_t, _ = unroll(torch.as_tensor(f0, dtype=torch.float64), 40, tau, 0.0)
    assert np.allclose(np.asarray(f_t.detach()), f_ref, rtol=1e-10, atol=1e-13)


@pytest.mark.unit
def test_cs_zero_is_plain_bgk_and_cs_raises_viscosity():
    """The LES term only ever *adds* dissipation, and vanishes at ``Cs = 0``."""
    f0, _, _ = ref.tgv_ic(16, u0=0.05)
    tau = 0.55

    def energy(cs):
        f, _ = unroll(torch.as_tensor(f0, dtype=torch.float64), 60, tau, cs,
                      snapshots=(60,))
        _, ux, uy = macroscopic(f)
        return 0.5 * float((ux * ux + uy * uy).mean())

    e0, e1, e2 = energy(0.0), energy(0.15), energy(0.4)
    assert e1 < e0 and e2 < e1


@pytest.mark.unit
def test_closure_eddy_viscosity_matches_analytic():
    """Realised eddy viscosity vs ``Cs^2 |S|`` for a linear shear probe (<15%)."""
    for cs in (0.1, 0.3):
        row = measure_eddy_viscosity(cs)
        err = abs(row["nu_t_measured"] - row["nu_t_predicted"]) / row["nu_t_predicted"] * 100
        assert err < 15.0, row


@pytest.mark.unit
def test_bptt_gradient_matches_finite_difference():
    """Autograd-through-unroll dJ/dCs vs central FD of the same loss (<5%)."""
    f0, _, _ = ref.tgv_ic(16, u0=0.05)
    tau = ref.tau_for_re(16, 0.05, 500.0)
    snaps = {}
    f = torch.as_tensor(f0, dtype=torch.float64)
    with torch.no_grad():
        _, states = unroll(f, 40, tau, 0.3, snapshots=(20, 40))
        for k, fs in states.items():
            _, ux, uy = macroscopic(fs)
            snaps[k] = (ux.clone(), uy.clone())
    prob = CalibProblem(f0=f, ref_snapshots=snaps, tau0=tau, n_steps=40)
    for cs in (0.05, 0.2):
        g_ad = autograd_gradient(prob, cs)
        g_fd = fd_gradient(prob, cs, h=1e-4)
        assert abs(g_ad - g_fd) <= 0.05 * abs(g_fd), (cs, g_ad, g_fd)


@pytest.mark.integration
def test_bptt_recovers_known_constant():
    """Identical-twin recovery: BPTT finds a planted ``Cs`` from far away."""
    f0, _, _ = ref.tgv_ic(16, u0=0.05)
    tau = ref.tau_for_re(16, 0.05, 500.0)
    truth = 0.3
    with torch.no_grad():
        _, states = unroll(torch.as_tensor(f0, dtype=torch.float64), 60, tau, truth,
                           snapshots=(30, 60))
    snaps = {}
    for k, fs in states.items():
        _, ux, uy = macroscopic(fs)
        snaps[k] = (ux.clone(), uy.clone())
    prob = CalibProblem(f0=torch.as_tensor(f0, dtype=torch.float64),
                        ref_snapshots=snaps, tau0=tau, n_steps=60)
    out = bptt_calibrate(prob, 0.05, iters=15, lr=0.15)
    assert abs(out["cs"] - truth) / truth < 0.15, out
    assert out["loss"] < 0.05 * out["loss_init"]


@pytest.mark.unit
def test_oss_proxy_and_literature_closures_differ_by_prefactor():
    """OSS proxy ``6 Q Sc`` vs literature ``18 sqrt(2) Q Cs^2``: ratio 4.243."""
    lit = measure_eddy_viscosity(0.17)
    oss = measure_eddy_viscosity(0.17, closure="oss_proxy")
    ratio = lit["nu_t_measured"] / oss["nu_t_measured"]
    assert abs(ratio - 18.0 * math.sqrt(2.0) / 6.0) / (18.0 * math.sqrt(2.0) / 6.0) < 0.10