"""Tests for the objective-selection layer (XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR).

The case is deliberately tiny (16x16, 20 steps) so the suite stays fast; the
physics of the anchor case lives in ``scripts/xval_r2_selection_bptt_tgv.py``
and its ``results.json``.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from tensorlbm.les_graph_unroll import reference2d as ref2d
from tensorlbm.les_graph_unroll.selection import (
    OBJECTIVES,
    SelectionCase,
    autograd_gradient,
    bptt_select,
    enstrophy,
    fd_gradient,
    golden_select,
    holdout_metrics,
    kinetic_energy,
    objective_loss,
    snapshot_fields,
    spectrum_np,
    sweep_select,
)
from tensorlbm.les_graph_unroll.solver2d import unroll

NX = 16
TRAIN = (10, 20)
HOLD = 30
TAU0 = 0.515563


def _case(closure: str = "smagorinsky", theta_ref: float = 0.0) -> SelectionCase:
    f0, _, _ = ref2d.tgv_ic(NX, 0.1)
    f0t = torch.as_tensor(f0, dtype=torch.float64)
    _, snaps = unroll(f0t, HOLD, TAU0, theta_ref, closure,
                      snapshots=tuple(sorted(set(TRAIN) | {HOLD})))
    ref = {}
    for k, f in snaps.items():
        rho = f.sum(dim=0)
        ref[k] = ((f * torch.tensor([c[0] for c in ref2d.C]).view(9, 1, 1)).sum(0) / rho).numpy(), \
                 ((f * torch.tensor([c[1] for c in ref2d.C]).view(9, 1, 1)).sum(0) / rho).numpy()
    return SelectionCase(f0=f0t, ref=ref, tau0=TAU0, n_steps=HOLD, closure=closure,
                         train_steps=TRAIN, holdout_step=HOLD)


def test_closure_list():
    assert set(OBJECTIVES) == {"field", "energy", "enstrophy"}


def test_objectives_vanish_on_their_own_reference():
    """A case whose reference *is* the theta=0 trajectory must give J(0) ~ 0."""
    case = _case(theta_ref=0.0)
    for obj in OBJECTIVES:
        j = float(objective_loss(case, obj, 0.0))
        assert abs(j) < 1e-20, (obj, j)


def test_autograd_matches_finite_difference():
    """The BPTT gradient equals a central difference for every objective."""
    # evaluated *away* from the reference constant: at theta = theta_ref the
    # loss is exactly zero, so both gradients vanish and the check is vacuous.
    case = _case(theta_ref=0.02)
    for obj in OBJECTIVES:
        ga = autograd_gradient(case, obj, 0.08)
        gf = fd_gradient(case, obj, 0.08, h=1e-3)
        scale = max(abs(ga), abs(gf), 1e-12)
        assert abs(ga - gf) / scale < 0.05, (obj, ga, gf)


def test_energy_and_enstrophy_match_numpy():
    """Differentiable KE/enstrophy agree with an independent numpy evaluation."""
    case = _case()
    with torch.no_grad():
        f = unroll(case.f0, 10, TAU0, 0.03, "smagorinsky", snapshots=(10,))[1][10]
    rho = f.sum(0)
    ex = torch.tensor([c[0] for c in ref2d.C], dtype=torch.float64).view(9, 1, 1)
    ey = torch.tensor([c[1] for c in ref2d.C], dtype=torch.float64).view(9, 1, 1)
    ux = ((f * ex).sum(0) / rho).numpy()
    uy = ((f * ey).sum(0) / rho).numpy()
    ke_np = 0.5 * float(np.mean(ux * ux + uy * uy))
    w = np.gradient(uy, axis=0) - np.gradient(ux, axis=1)
    z_np = 0.5 * float(np.mean(w * w))
    ke_t = float(kinetic_energy(*(torch.as_tensor(ux), torch.as_tensor(uy))))
    z_t = float(enstrophy(*(torch.as_tensor(ux), torch.as_tensor(uy))))
    assert abs(ke_t - ke_np) / ke_np < 0.02
    assert abs(z_t - z_np) / z_np < 0.05


def test_bptt_reduces_loss_and_stays_in_bounds():
    case = _case(theta_ref=0.05)
    out = bptt_select(case, "field", 0.2, iters=8, lr=0.1)
    assert out["loss"] <= out["loss_init"] + 1e-15
    assert 1e-3 - 1e-12 <= out["theta"] <= 1.0 + 1e-12
    assert len(out["history"]) == 8


def test_controls_are_autograd_free_and_consistent():
    case = _case(theta_ref=0.05)
    grid = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6]
    sw = sweep_select(case, "field", grid)
    gd = golden_select(case, "field", 0.0, 0.6, iters=20)
    assert sw["theta"] in grid
    assert sw["loss"] <= max(p[1] for p in sw["grid"])
    assert 0.0 <= gd["theta"] <= 0.6


def test_holdout_metrics_structure():
    case = _case(theta_ref=0.05)
    h = holdout_metrics(case, 0.05)
    for k in ("field_err", "ke_err", "z_err", "spec_err"):
        assert 0.0 <= h[k] < 10.0
    assert h["step"] == HOLD
    # the reference at theta_ref is itself, so the error is not zero (theta differs)
    h_same = holdout_metrics(case, 0.05)  # the case's own reference constant
    assert h_same["ke_err"] < 1e-9
    assert h_same["field_err"] < 1e-9


def test_spectrum_normalised():
    ux = np.sin(np.arange(NX).reshape(NX, 1) * 2 * math.pi / NX) * np.ones((1, NX))
    s = spectrum_np(ux, np.zeros_like(ux))
    assert abs(s.sum() - 1.0) < 1e-10
    assert s[1] > 0.99  # all energy in the k=1 shell