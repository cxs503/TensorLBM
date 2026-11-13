"""Tests for the fused push-pull D2Q9 stepper (XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN).

The contract under test: :class:`tensorlbm.fused_push_pull.FusedPushPullStepper`
reproduces, to floating-point round-off, (a) the upstream
``lbm-principles-practice/code`` chapter-13 fused kernel (transliterated in
``xval/fused_push_pull_aa_pattern/oss_port.py``) and (b) this repository's own
AB-pattern reference (``solver.stream`` + ``collide_bgk``), and its
Taylor-Green decay passes the owner's 15 % gate while a deliberately broken
convention (West/East swap) does not.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "xval" / "fused_push_pull_aa_pattern"))

from oss_port import OSSFusedStepper, taylor_green_eval  # noqa: E402

from tensorlbm.d2q9 import C, OPPOSITE, macroscopic  # noqa: E402
from tensorlbm.fused_push_pull import (  # noqa: E402
    FusedPushPullStepper,
    equilibrium_upstream,
    nu_from_tau,
    tau_from_nu,
    tg_l2_errors,
)
from tensorlbm.solver import collide_bgk, stream  # noqa: E402

GATE = 0.15


def _ic(nx: int = 16, nu: float = 1.0 / 6.0, u_max: float = 0.02):
    rho, ux, uy = taylor_green_eval(0, nx, nx, nu, u_max)
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)  # noqa: E731
    return t(rho), t(ux), t(uy)


def _eq(nx: int = 16, nu: float = 1.0 / 6.0, u_max: float = 0.02):
    rho, ux, uy = _ic(nx, nu, u_max)
    return equilibrium_upstream(rho, ux, uy)


# ------------------------------------------------------------------ conventions
def test_tau_nu_relation_matches_upstream_header():
    # chapter13/gpu/LBM.h: tau = 3.0*nu + 0.5 ; LBM.cu: tauinv = 2/(6 nu + 1)
    for nu in (1.0 / 6.0, 0.05, 0.2):
        tau = tau_from_nu(nu)
        assert tau == pytest.approx(3.0 * nu + 0.5, rel=0, abs=1e-15)
        assert 1.0 / tau == pytest.approx(2.0 / (6.0 * nu + 1.0), rel=1e-15)
        assert nu_from_tau(tau) == pytest.approx(nu, rel=1e-15)


def test_velocity_table_matches_upstream_numbering():
    """Upstream ``6 2 5 / 3 0 1 / 7 4 8`` with 1=E, 2=N, 3=W, 4=S, 5=NE, 6=NW, 7=SW, 8=SE."""
    expected = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
    assert [tuple(int(v) for v in row) for row in C.tolist()] == expected
    assert [int(OPPOSITE[i]) for i in range(9)] == [0, 3, 4, 1, 2, 7, 8, 5, 6]


def test_equilibrium_matches_repo_d2q9_equilibrium():
    from tensorlbm.d2q9 import equilibrium

    rho, ux, uy = _ic(8)
    a = equilibrium_upstream(rho, ux, uy)
    b = equilibrium(rho, ux, uy)
    assert torch.allclose(a, b, rtol=0, atol=1e-15)


# ---------------------------------------------------------------- equivalence
def test_gather_index_is_periodic_shift():
    """One step on a field where only direction 1 is non-zero must shift it by (-1, 0)."""
    nx = 8
    torch.manual_seed(0)
    f = torch.rand(9, nx, nx, dtype=torch.float64)
    st = FusedPushPullStepper(f, 1.0, keep_momenta=False)
    p = st._pull(st.f_pre)
    for i in range(8):
        d = i + 1
        cx, cy = int(C[d, 0]), int(C[d, 1])
        # dst[y, x] = src[y - cy, x - cx]  <=>  torch.roll(shift=(cy, cx))
        assert torch.equal(p[i], torch.roll(st.f_pre[i], shifts=(cy, cx), dims=(0, 1))), d


def test_reproduces_upstream_numpy_port_bitwise():
    """The torch stepper and the upstream transliteration are bit-identical."""
    nx, nsteps, tau = 16, 40, 1.0
    ref = OSSFusedStepper(nx, nx, nu_from_tau(tau), 0.02)
    st = FusedPushPullStepper(_eq(nx), tau)
    for _ in range(nsteps):
        ref.step()
        st.step()
    assert np.array_equal(ref.fields(), st.fields().numpy())
    # the stored fields are bit-identical; the recomputed moments differ only by
    # summation order (kernel accumulates ft1..ft8 sequentially, macroscopic()
    # sums the stacked field), so compare them at round-off tolerance.
    rho, ux, uy = st.macroscopic()
    assert np.allclose(rho.numpy(), ref.rho, rtol=0, atol=1e-15)
    assert np.allclose(ux.numpy(), ref.ux, rtol=0, atol=1e-15)


def test_single_buffer_equals_double_buffer():
    nx, nsteps, tau = 16, 40, 1.0
    a = FusedPushPullStepper(_eq(nx), tau)
    b = FusedPushPullStepper(_eq(nx), tau, single_buffer=True)
    for _ in range(nsteps):
        a.step()
        b.step()
    assert torch.equal(a.fields(), b.fields())
    assert b.persistent_doubles_per_node() == 9
    assert a.persistent_doubles_per_node() == 17


def test_agrees_with_repo_ab_reference():
    """solver.stream + collide_bgk (AB, two fields) vs the fused step: round-off only."""
    nx, nsteps, tau = 16, 40, 1.0
    f = _eq(nx)
    st = FusedPushPullStepper(f, tau)
    for _ in range(nsteps):
        f = collide_bgk(stream(f), tau)
        st.step()
    ref = f.numpy()
    got = st.fields().numpy()
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert rel < 1e-14, rel
    rho_a, ux_a, uy_a = macroscopic(f)
    rho_b, ux_b, uy_b = st.macroscopic()
    assert torch.allclose(rho_a, rho_b, rtol=0, atol=1e-15)
    assert float((ux_a - ux_b).abs().max()) < 1e-15


def test_mass_and_momentum_conservation():
    nx, tau = 16, 0.8
    st = FusedPushPullStepper(_eq(nx), tau)
    rho0, ux0, uy0 = st.macroscopic()
    m0 = float(rho0.sum())
    px0 = float((rho0 * ux0).sum())
    for _ in range(50):
        stats = st.step(keep_momenta=True) if False else st.step()
    assert abs(stats.mass - m0) < 1e-12
    assert abs(stats.momentum_x - px0) < 1e-12
    assert abs(stats.momentum_y - 0.0) < 1e-12


# ---------------------------------------------------------------- accuracy gate
@pytest.mark.parametrize("nx,nsteps", [(32, 200)])
def test_tg_decay_passes_gate(nx, nsteps):
    cfg = dict(nx=nx, ny=nx, u_max=0.02, nu=1.0 / 6.0, rho0=1.0)
    st = FusedPushPullStepper(_eq(nx), tau_from_nu(cfg["nu"]))
    for _ in range(nsteps):
        st.step()
    rho, ux, uy = st.macroscopic()
    err = tg_l2_errors(rho, ux, uy, t=nsteps, **cfg)
    assert err["ux_lu2"] < GATE
    assert err["uy_lu2"] < GATE
    assert err["rho_lu2"] < GATE
    # the harder "<3 %" read-out is met for the velocity field at this resolution
    assert err["ux_lu2"] < 0.03


def test_nu_inversion_from_energy_decay():
    nx, tau = 32, 1.0
    nu = nu_from_tau(tau)
    st = FusedPushPullStepper(_eq(nx), tau)
    ts, es = [], []
    for n in range(200):
        st.step()
        if (n + 1) % 50 == 0:
            _, ux, uy = st.macroscopic()
            ts.append(n + 1)
            es.append(float((ux**2 + uy**2).sum()))
    slope = np.polyfit(ts, np.log(es), 1)[0]
    k = 2.0 * math.pi / nx
    nu_fit = -slope / (2.0 * 2.0 * k * k)
    assert abs(nu_fit - nu) / nu < GATE


def test_mirror_x_control_fails_the_gate():
    """West/East swap (a classic porting bug) must be rejected by the same gate."""
    nx, nsteps = 32, 200
    ref = OSSFusedStepper(nx, nx, 1.0 / 6.0, 0.02)
    bad = OSSFusedStepper(nx, nx, 1.0 / 6.0, 0.02)
    from run import _variant_step  # xval bench helper

    for n in range(nsteps):
        ref.step()
        _variant_step(bad, 1.0 / 6.0, "mirror_x")
        bad.f1, bad.f2 = bad.f2, bad.f1
        bad.time += 1
    ra, uxa, uya = taylor_green_eval(nsteps, nx, nx, 1.0 / 6.0, 0.02)
    err_bad = math.sqrt(((bad.uy - uya) ** 2).sum() / (uya**2).sum())
    err_ref = math.sqrt(((ref.uy - uya) ** 2).sum() / (uya**2).sum())
    assert err_ref < GATE
    assert err_bad > GATE
    # population-level comparison is *insensitive* here (rel_l2 ~ 1e-3, the dense
    # rest population masks it); the macroscopic velocity is what separates them.
    ux_rel = np.linalg.norm(ref.ux - bad.ux) / np.linalg.norm(ref.ux)
    assert ux_rel > GATE


def test_inplace_per_node_update_is_not_safe():
    """The single-buffer arm is only safe because the pull is materialised first.

    A per-node sequential in-place update (what a naive GPU/loop port does) reads
    values the same step already overwrote; the AA-pattern *hazard*.  We show the
    difference exists, which is why ``single_buffer`` documents its precondition.
    """
    nx, tau = 8, 1.0
    f = _eq(nx).numpy()
    st = FusedPushPullStepper(torch.as_tensor(f), tau)
    for _ in range(10):
        st.step()
    safe = st.fields().numpy()

    # naive: write each direction back into the source array in scan order
    g = f.copy()
    for _ in range(10):
        g = _naive_inplace_step(g, tau)
    differing_cells = int((np.abs(safe - g) > 1e-12).sum())
    assert differing_cells > 0


def _naive_inplace_step(f: np.ndarray, tau: float) -> np.ndarray:
    """Per-node sequential single-buffer step: gather-then-overwrite, no ping-pong.

    This is what a naive GPU/loop port of the fused step does.  Node ``(x, y)``
    overwrites its own nine slots, and a later node reading slot ``d`` from
    ``(x, y)`` then sees a *post-collision* value -- the hazard the upstream
    second buffer exists to remove.
    """
    c = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
    w = [4 / 9] + [1 / 9] * 4 + [1 / 36] * 4
    ny, nx = f.shape[1], f.shape[2]
    for y in range(ny):
        for x in range(nx):
            p = [f[0, y, x]]
            for d in range(1, 9):
                cx, cy = c[d]
                p.append(f[d, (y - cy) % ny, (x - cx) % nx])
            rho = sum(p)
            ux = (p[1] + p[5] + p[8] - (p[3] + p[6] + p[7])) / rho
            uy = (p[2] + p[5] + p[6] - (p[4] + p[7] + p[8])) / rho
            for d in range(9):
                cx, cy = c[d]
                cu = 3.0 * (cx * ux + cy * uy)
                feq = w[d] * rho * (1.0 - 1.5 * (ux * ux + uy * uy) + cu * (1.0 + 0.5 * cu))
                f[d, y, x] = p[d] + (feq - p[d]) / tau
    return f