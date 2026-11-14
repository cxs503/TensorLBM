"""Acceptance tests for the 3-D memory-bounded differentiable LBM (GAP-DIFFLBM-3D-COST).

What is asserted (invariants, not smoke):

1. ``step3d`` conserves mass and momentum exactly (periodic, no forcing).
2. The checkpointed reverse mode reproduces the store-everything BPTT
   gradient (same arithmetic, so round-off only).
3. The checkpointed gradient agrees with an **independent** central /
   Richardson finite difference to ``<= 1e-6`` relative — the gate.
4. The gradient agrees with the continuum ABC-decay anchor
   ``dJ/dtau = -(2/3) k^2 T J`` to within 3 % (discretisation limit), and the
   error decreases with resolution.
5. The peak-memory model does not depend on the number of steps, and the
   measured high-water mark is flat in ``T`` (boundedness, run in subprocesses).
6. ``recover_tau`` (solver entry ``solver3d.inversion_gradient_3d``) recovers a
   hidden relaxation time from a terminal snapshot.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch

from tensorlbm import diff3d
from tensorlbm.solver3d import inversion_gradient_3d

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE = os.path.join(REPO, "benchmarks", "gap_difflbm_3d_cost", "memprobe.py")

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_step_conserves_mass_and_momentum():
    """BGK collision + exact shift conserve mass and momentum identically.

    A deterministic random field is used because the ABC flow has zero net
    momentum, which would make a relative comparison meaningless.
    """
    c, w = diff3d.constants()
    g = torch.Generator().manual_seed(20261113)
    f0 = torch.full((19, 10, 10, 10), 1.0 / 19.0, dtype=torch.float64)
    f0 = f0 + 1e-3 * torch.rand((19, 10, 10, 10), generator=g, dtype=torch.float64)
    f1 = diff3d.step3d(f0, 0.8, c, w)
    rho0, ux0, uy0, uz0 = diff3d.macro3d(f0, c)
    rho1, ux1, uy1, uz1 = diff3d.macro3d(f1, c)
    assert float(rho1.sum()) == pytest.approx(float(rho0.sum()), rel=1e-14)
    for a, b in (((rho0 * ux0).sum(), (rho1 * ux1).sum()),
                 ((rho0 * uy0).sum(), (rho1 * uy1).sum()),
                 ((rho0 * uz0).sum(), (rho1 * uz1).sum())):
        assert float(b) == pytest.approx(float(a), rel=1e-12)
        assert abs(float(a)) > 1e-3  # net momentum actually non-zero


def test_checkpointed_matches_full_bptt():
    f0 = diff3d.abc_initial(10, amp=0.05)
    ad = diff3d.checkpointed_gradient(f0, 0.8, 16, segment=4)
    bp = diff3d.full_bptt_gradient(f0, 0.8, 16)
    assert ad.value == pytest.approx(bp.value, rel=1e-14, abs=0.0)
    rel = abs(ad.grad["tau"] - bp.grad["tau"]) / abs(bp.grad["tau"])
    assert rel < 1e-12, rel


def test_checkpointed_matches_finite_difference_gate():
    f0 = diff3d.abc_initial(12, amp=0.05)
    ad = diff3d.checkpointed_gradient(f0, 0.8, 24, segment=3)
    fd = diff3d.finite_difference_gradient(f0, 0.8, 24, h=1e-4, richardson=True)
    rel = abs(ad.grad["tau"] - fd.grad["tau"]) / abs(fd.grad["tau"])
    assert rel <= 1e-6, rel
    # a wrong-sign or off-by-a-factor gradient could not pass this
    assert ad.grad["tau"] < 0.0


def test_gradient_matches_continuum_anchor_and_converges():
    ana_err = []
    for n in (12, 24):
        f0 = diff3d.abc_initial(n, amp=0.05)
        ad = diff3d.checkpointed_gradient(f0, 0.8, 48, segment=4)
        ana = diff3d.analytic_decay_gradient(n, 0.8, 48, amp=0.05,
                                            j_value=ad.value)
        ana_err.append(abs(ad.grad["tau"] - ana["dJ_dtau"]) / abs(ana["dJ_dtau"]))
    assert ana_err[0] < 0.03, ana_err
    assert ana_err[1] < ana_err[0], ana_err  # O(k^2) discretisation error shrinks


def test_segment_does_not_change_gradient():
    f0 = diff3d.abc_initial(8, amp=0.05)
    g = {}
    for seg in (1, 2, 5, 16):
        r = diff3d.checkpointed_gradient(f0, 0.8, 16, segment=seg)
        g[seg] = r.grad["tau"]
        assert r.checkpoints == len(range(0, 16, seg))
    assert max(g.values()) - min(g.values()) <= 1e-12 * abs(g[1])


def test_memory_model_independent_of_steps():
    a = diff3d.memory_model_bytes(256, segment=1, dtype=torch.float32)
    b = diff3d.memory_model_bytes(256, segment=1, dtype=torch.float32,
                                  f_obs=True)
    assert b > a
    assert diff3d.state_bytes(256, torch.float64) == 19 * 256 ** 3 * 8
    # the model has no `steps` argument at all: bounded by construction
    with pytest.raises(TypeError):
        diff3d.memory_model_bytes(256, segment=1, steps=1000)


def test_measured_peak_memory_flat_in_steps():
    """High-water mark must not grow with T (else checkpointing is a no-op)."""
    if not os.path.exists(PROBE):
        pytest.skip("memprobe.py missing")
    peaks = []
    for steps in (8, 32):
        out = subprocess.run(
            [sys.executable, PROBE, "--grid", "16", "--steps", str(steps),
             "--mode", "grad", "--segment", "4", "--threads", "2"],
            capture_output=True, text=True, timeout=600)
        assert out.returncode == 0, out.stderr[-1000:]
        peaks.append(json.loads(out.stdout.strip().splitlines()[-1])["peak_bytes"])
        out = subprocess.run(
            [sys.executable, PROBE, "--grid", "16", "--steps", str(steps),
             "--mode", "grad", "--segment", "4", "--threads", "2"],
            capture_output=True, text=True, timeout=600)
        rec = json.loads(out.stdout.strip().splitlines()[-1])
        peaks[-1] = rec["peak_bytes"]
    # 4x more steps must not increase the peak by more than 50 %
    assert peaks[1] <= 1.5 * peaks[0], peaks


def test_inversion_recovers_tau():
    n, steps = 12, 32
    c, w = diff3d.constants()
    f0 = diff3d.abc_initial(n, amp=0.05)
    with torch.no_grad():
        f_obs = f0.clone()
        for _ in range(steps):
            f_obs = diff3d.step3d(f_obs, 0.8, c, w)
    tau_hat, history = inversion_gradient_3d(f0, f_obs, steps, tau0=0.74,
                                             lr=0.02, iters=20, segment=4)
    assert len(history) == 20
    assert history[-1]["loss"] <= history[0]["loss"]
    assert abs(tau_hat - 0.8) / 0.8 <= 0.02, (tau_hat, history)