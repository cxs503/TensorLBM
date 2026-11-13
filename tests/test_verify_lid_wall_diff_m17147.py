"""Tests for the independent verifier of GAP-PP-2-lid-wall-diff (m17147).

These pin the invariants the *verifier* relies on, so the audit numbers in
``docs/AUDIT_m17147_GAP_PP_2_LID_WALL_DIFF.md`` are machine-checkable rather
than asserted in prose:

1. the Ghia (1982) table -> probe-layout conversion reproduces the tabulated
   values exactly at tabulated abscissae and keeps the bottom->top /
   left->right ordering (a flipped abscissa would silently invert the target);
2. the production lid chain (clone + in-place ``zou_he_moving_lid``) and the
   graph-preserving chain agree to the last bit -- the equivalence claim of the
   feature, checked here on an independent code path
   (``verify_lid_wall_diff_m17147._prod_rollout``);
3. the autograd gradient matches a central finite difference *and* its
   Richardson extrapolation on a functional that is **not** the shipped
   centreline MSE, so criterion ② is not an artefact of the implementing
   driver's own loss.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from tensorlbm.d2q9 import macroscopic
from tensorlbm.diff_lid_cavity import (
    GHIA_STATIONS,
    CavityGeometry,
    initial_field,
    rollout,
    tau_from_reynolds,
)

DT = torch.float64
_ROOT = Path(__file__).resolve().parents[1]


def _load_verifier():
    path = _ROOT / "scripts" / "verify_lid_wall_diff_m17147.py"
    spec = importlib.util.spec_from_file_location("verify_m17147", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V = _load_verifier()


# ---------------------------------------------------------------------------
# 1. Ghia target construction
# ---------------------------------------------------------------------------


def test_linear_interp_is_exact_on_tabulated_nodes() -> None:
    from tensorlbm.lid_driven_cavity import GHIA_RE100

    y = list(GHIA_RE100["y"])[::-1]
    u = list(GHIA_RE100["u"])[::-1]
    for yi, ui in zip(y, u):
        assert V._lin_interp(y, u, yi) == pytest.approx(ui, abs=1e-12)


def test_ghia_target_shape_and_ordering() -> None:
    target = V.ghia_target()
    assert target.shape == (2 * len(GHIA_STATIONS),)
    u_part = target[: len(GHIA_STATIONS)]
    v_part = target[len(GHIA_STATIONS):]
    # Ghia Re=100: u along the vertical centreline grows from ~0 at the bottom
    # to 1 at the lid -> the vector must be bottom->top ordered (positive slope).
    assert u_part[0] < 0.0 < 0.5 < u_part[-1]
    # top station 0.9375 sits between the tabulated 0.9531 (0.68717) and
    # 0.8516 (0.23151) rows -> u ~ 0.60, not 1.0 (the lid value is not a station)
    assert 0.5 < u_part[-1] < 0.75
    # v along the horizontal centreline is positive on the left, negative on the
    # right (single vortex rotating clockwise).
    assert v_part[0] > 0.0 > v_part[-1]
    # exact tabulated node: station y = 0.5 is Ghia table row y = 0.5000
    from tensorlbm.lid_driven_cavity import GHIA_RE100

    idx = GHIA_STATIONS.index(0.5)
    tab = GHIA_RE100["u"][GHIA_RE100["y"].index(0.5)]
    assert u_part[idx] == pytest.approx(tab, abs=1e-12)


def test_ghia_target_requires_ascending_abscissa() -> None:
    """``_lin_interp`` is ascending-only; ``ghia_target`` must do the reversal.

    Guard against a future "simplification" that drops the ``[::-1]``: on the
    raw descending table the same call returns the wrong end of the profile
    (1.0 instead of -0.04192) instead of raising, which would silently invert
    the whole inverse problem.
    """
    from tensorlbm.lid_driven_cavity import GHIA_RE100

    y_raw = list(GHIA_RE100["y"])
    u_raw = list(GHIA_RE100["u"])
    assert y_raw[0] > y_raw[-1]
    y = y_raw[::-1]
    u = u_raw[::-1]
    assert V._lin_interp(y, u, 0.0625) == pytest.approx(-0.04192, abs=1e-12)
    assert V._lin_interp(y_raw, u_raw, 0.0625) != pytest.approx(-0.04192, abs=1e-12)

    target = V.ghia_target()
    idx = list(GHIA_STATIONS).index(0.0625)
    assert target[idx] == pytest.approx(-0.04192, abs=1e-12)


# ---------------------------------------------------------------------------
# 2. production chain vs graph-preserving chain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,steps", [(21, 50), (33, 40)])
def test_production_and_diff_lid_chains_are_bit_identical(n: int, steps: int) -> None:
    geom = CavityGeometry(ny=n, nx=n)
    f0 = initial_field(geom)
    tau = tau_from_reynolds(100.0, 0.1, length=n - 1)
    f_diff = rollout(f0, torch.tensor(tau, dtype=DT), 0.1, steps, wall_mask=geom.wall_mask)
    f_prod = V._prod_rollout(f0, tau, 0.1, steps, geom.wall_mask)
    assert float((f_diff - f_prod).abs().max()) == 0.0


# ---------------------------------------------------------------------------
# 3. gradient cross-check on an independent functional
# ---------------------------------------------------------------------------


def test_autograd_matches_fd_and_richardson_on_energy_mismatch() -> None:
    torch.set_num_threads(2)
    n, steps = 17, 40
    geom = CavityGeometry(ny=n, nx=n)
    mask = geom.wall_mask
    f0 = initial_field(geom)
    tau_true = tau_from_reynolds(100.0, 0.1, length=n - 1)
    u_lid = 0.1

    with torch.no_grad():
        f_ref = rollout(f0, torch.tensor(tau_true + 0.02, dtype=DT), u_lid, steps, wall_mask=mask)
        _, ux_r, uy_r = macroscopic(f_ref)

    def loss(tau: float, u: float) -> float:
        with torch.no_grad():
            f = rollout(f0, torch.tensor(tau, dtype=DT), u, steps, wall_mask=mask)
            return float(V._mismatch_loss(f, mask, ux_r, uy_r))

    tau = torch.tensor(tau_true, dtype=DT, requires_grad=True)
    u = torch.tensor(u_lid, dtype=DT, requires_grad=True)
    f = rollout(f0, tau, u, steps, wall_mask=mask)
    g_tau, g_u = torch.autograd.grad(V._mismatch_loss(f, mask, ux_r, uy_r), (tau, u))

    for grad, fn, x0 in ((g_tau, lambda x: loss(x, u_lid), tau_true),
                         (g_u, lambda x: loss(tau_true, x), u_lid)):
        h = 1e-4
        fd = (fn(x0 + h) - fn(x0 - h)) / (2 * h)
        assert abs(fd - float(grad)) / abs(float(grad)) < 1e-4
        # Richardson extrapolation of the O(h^2) truncation error
        h2 = 2e-4
        fd2 = (fn(x0 + h2) - fn(x0 - h2)) / (2 * h2)
        rich = fd + (fd - fd2) / (4.0 - 1.0)
        assert abs(rich - float(grad)) / abs(float(grad)) < 1e-5