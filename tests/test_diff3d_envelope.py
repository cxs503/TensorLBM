"""Tests for the memory-bounded 3D D3Q19 expanded-form AD (``diff3d_envelope``).

The gradient contract is checked three ways:

* segmented (disk replay) AD == full-graph AD -- the replay is *exact*
  discrete differentiation, not an approximation;
* segmented AD vs a 5-point central finite difference of the forward solver
  in float64 (the acceptance criterion of GAP-DIFFLBM-3D-COST);
* the stored boundary states are exact (bitwise) and are cleaned up.

Run: ``python -m pytest tests/test_diff3d_envelope.py -q`` (a few seconds).
"""

from __future__ import annotations

import glob
import math
import tempfile

import pytest
import torch

from tensorlbm.autograd_path import rollout
from tensorlbm.diff3d_envelope import (
    SegmentStore,
    divergence_free_ic,
    segmented_rollout,
    shear_wave_amplitude,
    shear_wave_ic,
    taylor_green_2d_ic,
    taylor_green_energy,
)
from tensorlbm.solver3d import equilibrium3d, macroscopic3d

TORCH_DT = torch.float64
STEPS = 3
N = 8


def _step(f: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    from tensorlbm.autograd_path import differentiable_step

    return differentiable_step(f, tau, None)


def _ic(u: torch.Tensor) -> torch.Tensor:
    return equilibrium3d(torch.ones_like(u[0]), u[0], u[1], u[2]).to(u.dtype)


def _loss(f: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _, ux, uy, uz = macroscopic3d(f)
    du = torch.stack([ux, uy, uz]).to(torch.float64) - target.to(torch.float64)
    return 0.5 * du.square().sum() / du[0].numel()


def _problem():
    torch.manual_seed(0)
    u0 = divergence_free_ic(N, seed=0, amplitude=0.05, dtype=TORCH_DT)
    f0 = _ic(u0).clone().requires_grad_(True)
    tau_true = 0.60
    with torch.no_grad():
        f = f0.detach()
        t_true = torch.tensor(tau_true, dtype=TORCH_DT)
        for _ in range(STEPS):
            f = _step(f, t_true)
    _, ux, uy, uz = macroscopic3d(f)
    return u0, f0, torch.stack([ux, uy, uz])


def test_segment_store_roundtrip_is_bitwise() -> None:
    f = torch.randn(19, 4, 4, 4, dtype=torch.float64)
    root = tempfile.mkdtemp(prefix="segstore_test_")
    store = SegmentStore(root, 2, f.shape, f.dtype)
    store.write(0, f)
    store.write(1, 2 * f)
    back = store.read(0)
    assert torch.equal(back, f)  # exact, no quantisation
    assert torch.equal(store.read(1), 2 * f)
    assert store.size_gib() == pytest.approx(2 * f.numel() * 8 / 2**30)
    store.close(remove=True)
    assert not glob.glob(root + "/*")


def test_segmented_gradient_equals_full_graph() -> None:
    """The disk replay must return the *same* gradient as the unrolled graph."""
    u0, f0, target = _problem()
    tau_a = torch.tensor(0.55, dtype=TORCH_DT, requires_grad=True)
    tau_b = torch.tensor(0.55, dtype=TORCH_DT, requires_grad=True)

    f_seg = segmented_rollout(f0, tau_a, STEPS, _step, seg_len=1)
    j_seg = _loss(f_seg, target)
    g_seg = torch.autograd.grad(j_seg, [tau_a, f0])
    assert torch.isfinite(g_seg[1]).all()  # f0 gradient flows through the replay

    f_full = rollout(f0.detach(), STEPS, tau_b, None, checkpoint=False)
    j_full = _loss(f_full, target)
    g_full = torch.autograd.grad(j_full, [tau_b])[0]

    assert float(j_seg) == pytest.approx(float(j_full), rel=1e-12)
    assert float(g_seg[0]) == pytest.approx(float(g_full), rel=1e-10)


def test_segmented_gradient_matches_finite_difference() -> None:
    """Acceptance criterion: relative error AD-vs-FD <= 1e-6 (float64)."""
    u0, f0, target = _problem()
    tau0, h = 0.55, 1e-3
    tau = torch.tensor(tau0, dtype=TORCH_DT, requires_grad=True)
    f_T = segmented_rollout(f0.detach(), tau, STEPS, _step, seg_len=2)
    g_ad = float(torch.autograd.grad(_loss(f_T, target), [tau])[0])

    def loss_at(tau_x: float) -> float:
        with torch.no_grad():
            f = f0.detach()
            t = torch.tensor(tau_x, dtype=TORCH_DT)
            for _ in range(STEPS):
                f = _step(f, t)
        return float(_loss(f, target))

    lp1, lm1 = loss_at(tau0 + h), loss_at(tau0 - h)
    lp2, lm2 = loss_at(tau0 + 2 * h), loss_at(tau0 - 2 * h)
    g_fd = (-lp2 + 8 * lp1 - 8 * lm1 + lm2) / (12 * h)  # 5-point, O(h^4)
    assert abs(g_ad - g_fd) / abs(g_fd) <= 1e-6, (g_ad, g_fd)


def test_divergence_free_ic_has_zero_divergence() -> None:
    u = divergence_free_ic(16, seed=3, amplitude=0.05, dtype=torch.float64)
    k = torch.fft.fftfreq(16, d=1.0 / (2 * math.pi)).double()
    k[16 // 2] = 0.0
    kmat = torch.stack(torch.meshgrid(k, k, k, indexing="ij"))
    div = (kmat * torch.fft.fftn(u, dim=(1, 2, 3))).sum(0)
    scale = float(kmat.abs().max()) * float(u.abs().max())
    assert float(div.abs().max()) < 1e-9 * scale


def test_zero_flow_taylor_green_decays_like_the_analytic_solution() -> None:
    """Physics anchor: the 3D operator must reproduce E(t) = E0 exp(-4 nu k^2 t)."""
    n, amp, tau = 16, 0.02, 0.55
    nu = (tau - 0.5) / 3
    k = 2 * math.pi / n
    f = _ic(taylor_green_2d_ic(n, amplitude=amp, dtype=torch.float32))
    t = torch.tensor(tau)
    steps = 60
    with torch.no_grad():
        for _ in range(steps):
            f = _step(f, t)
    _, ux, uy, uz = macroscopic3d(f)
    e = float(0.5 * (ux.double() ** 2 + uy.double() ** 2 + uz.double() ** 2).mean())
    e_ref = taylor_green_energy(steps, amplitude=amp, nu=nu, k=k)
    # n=16 is coarse: the resolution study (n=32 -> 1.3 %, n=64 -> 0.36 %) is
    # in docs/DIFFLBM_3D_COST_ENVELOPE.md; here the bound only guards the
    # operator against gross errors.
    assert abs(e - e_ref) / e_ref < 0.08


def test_shear_wave_is_transverse_and_decays_like_the_analytic_solution() -> None:
    n, amp, tau = 16, 0.02, 0.55
    nu = (tau - 0.5) / 3
    k = 2 * math.pi / n
    u0 = shear_wave_ic(n, amplitude=amp, dtype=torch.float32)
    # transverse: u_x varies along z, so the divergence and u_y/u_z are ~0
    assert float(u0[1].abs().max()) == 0.0 and float(u0[2].abs().max()) == 0.0
    f = _ic(u0)
    t = torch.tensor(tau)
    steps = 200
    with torch.no_grad():
        for _ in range(steps):
            f = _step(f, t)
    _, ux, uy, uz = macroscopic3d(f)
    prof = ux.double().mean(dim=(1, 2))
    amp_meas = 2 * float(torch.fft.rfft(prof)[1].abs()) / n
    amp_ref = shear_wave_amplitude(steps, amplitude=amp, nu=nu, k=k)
    assert abs(amp_meas - amp_ref) / amp_ref < 0.05


def test_temporary_store_is_cleaned_up() -> None:
    """An auto-created store must not leak when the graph is released."""
    u0, f0, target = _problem()
    before = set(glob.glob(tempfile.gettempdir() + "/lbm3d_segstore_*"))
    tau = torch.tensor(0.55, dtype=TORCH_DT, requires_grad=True)
    f = segmented_rollout(f0.detach(), tau, STEPS, _step, seg_len=1)
    j = _loss(f, target)
    torch.autograd.grad(j, [tau])
    del f, j, tau
    import gc

    gc.collect()
    after = set(glob.glob(tempfile.gettempdir() + "/lbm3d_segstore_*"))
    assert after == before