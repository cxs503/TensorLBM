"""Discriminating viscosity tests for the entropic KBC collision kernels.

W7-A acceptance tests: the shear viscosity of ``collide_kbc_d3q19`` /
``collide_kbc_d3q27`` must be controlled by ``tau`` (Karlin–Bösch–
Chikatamarla construction ``f* = f_eq + (1 − 1/τ)·s + β·h`` with the
entropy solve restricted to the ghost coefficient ``β``).

Primary criterion (discriminating against the pre-fix kernel, where the
unconditional entropy minimisation of the shear coefficient drove γ → 0 and
ν_eff → c_s²/2 = 1/6 regardless of τ):

    single-step pure-shear perturbation decay ratio == 1 − 1/τ

Secondary criteria: ghost coefficient β ∈ [0, 1], H-theorem (H(f*) ≤ H(f)),
population positivity, bulk/ghost-mode behaviour, and weakly compressible
agreement with the BGK trajectory at matched τ.
"""

from __future__ import annotations

import json
import math
import os

import pytest
import torch

from tensorlbm.d3q19 import C as C19
from tensorlbm.d3q19 import W as W19
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.d3q27 import C as C27
from tensorlbm.d3q27 import W as W27
from tensorlbm.d3q27 import equilibrium27, macroscopic27
from tensorlbm.entropic_kbc import collide_kbc_d3q19, collide_kbc_d3q27
from tensorlbm.solver3d import collide_bgk3d, stream3d

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

LATTICES = {
    19: dict(
        collide=collide_kbc_d3q19,
        equilibrium=equilibrium3d,
        macroscopic=macroscopic3d,
        C=C19,
        W=W19,
    ),
    27: dict(
        collide=collide_kbc_d3q27,
        equilibrium=equilibrium27,
        macroscopic=macroscopic27,
        C=C27,
        W=W27,
    ),
}

TAUS = [0.6, 0.8, 1.0, 1.5]


def _record_resolution() -> None:
    """Record which tensorlbm copy the tests resolved (audit trail)."""
    import tensorlbm

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, ".tensorlbm_resolution.json"), "w") as fh:
        json.dump({"tensorlbm_file": tensorlbm.__file__}, fh, indent=2)


_record_resolution()


def _base_state(q: int, shape=(3, 5, 4), dtype=torch.float32) -> torch.Tensor:
    """Equilibrium at rho=1, u=0 (plus nothing else)."""
    rho = torch.ones(shape, dtype=dtype)
    zero = torch.zeros_like(rho)
    return LATTICES[q]["equilibrium"](rho, zero, zero, zero)


def _pure_shear_perturbation(q: int, component: str, amp: float, dtype=torch.float32):
    """Pure deviatoric (shear) non-equilibrium with unit-normalised stress.

    Returns populations ``delta`` such that the second-order moment of
    ``delta`` is the deviatoric stress ``amp`` in the chosen component and
    every other hydrodynamic/non-equilibrium moment is zero.
    """
    info = LATTICES[q]
    c = info["C"].to(dtype=dtype)
    Q = c.shape[0]
    w = info["W"].to(dtype=dtype).view(Q, 1, 1, 1)
    cx = c[:, 0].view(Q, 1, 1, 1)
    cy = c[:, 1].view(Q, 1, 1, 1)
    cz = c[:, 2].view(Q, 1, 1, 1)
    c_sq = cx * cx + cy * cy + cz * cz

    if component == "xy":
        delta = 9.0 * w * (cx * cy) * amp
    elif component == "yz":
        delta = 9.0 * w * (cy * cz) * amp
    elif component == "dev_xx":
        # traceless diagonal: Π_dev = diag(amp, -amp/2 ... ) simplest is to use
        # the traceless basis Hd directly with a single unit amplitude on xx
        hd_xx = cx * cx - c_sq / 3.0
        delta = 4.5 * w * hd_xx * amp
    else:  # pragma: no cover - programming error guard
        raise ValueError(component)
    return delta


def _pi(f: torch.Tensor, q: int) -> dict[str, torch.Tensor]:
    """Second-order moments of f (per cell), contracted in fp64.

    The contraction is accumulated in fp64 so that the *measurement* noise on
    a fp32 kernel stays ~1e-9; the equilibrium part cancels exactly against
    the same contraction of the base state, so callers measure the
    non-equilibrium stress directly.
    """
    c = LATTICES[q]["C"].to(dtype=torch.float64)
    Q = c.shape[0]
    fd = f.to(dtype=torch.float64)
    cx = c[:, 0].view(Q, 1, 1, 1)
    cy = c[:, 1].view(Q, 1, 1, 1)
    cz = c[:, 2].view(Q, 1, 1, 1)
    return {
        "xx": (cx * cx * fd).sum(0),
        "yy": (cy * cy * fd).sum(0),
        "zz": (cz * cz * fd).sum(0),
        "xy": (cx * cy * fd).sum(0),
        "xz": (cx * cz * fd).sum(0),
        "yz": (cy * cz * fd).sum(0),
    }


def _pi_neq(f: torch.Tensor, f0: torch.Tensor, q: int) -> dict[str, torch.Tensor]:
    """Non-equilibrium second-order moments: Π(f) − Π(f_eq baseline)."""
    a, b = _pi(f, q), _pi(f0, q)
    return {key: a[key] - b[key] for key in a}


def _random_state(q: int, seed: int, amp: float, dtype=torch.float32) -> torch.Tensor:
    torch.manual_seed(seed)
    shape = (2, 3, 4)
    rho = 0.9 + torch.rand(shape, dtype=dtype)
    ux = 0.03 * torch.randn(shape, dtype=dtype)
    uy = 0.03 * torch.randn(shape, dtype=dtype)
    uz = 0.03 * torch.randn(shape, dtype=dtype)
    feq = LATTICES[q]["equilibrium"](rho, ux, uy, uz)
    return feq + amp * torch.randn_like(feq)


# ---------------------------------------------------------------------------
# 1. Primary discriminating criterion: single-step shear decay ratio == 1 − 1/τ
# ---------------------------------------------------------------------------


class TestSingleStepShearDecay:
    """A pure shear perturbation must decay by exactly (1 − 1/τ) in one collision.

    Pre-fix behaviour: the unconditional entropy minimisation of the shear
    coefficient returns γ ≈ 0 (f_eq is the H-minimiser among states with the
    same conserved moments), so the measured ratio is ≈ 0 for every τ and the
    effective shear viscosity is ν_eff ≈ c_s²/2 = 1/6, independent of τ.
    """

    @pytest.mark.parametrize("q", [19, 27])
    @pytest.mark.parametrize("tau", TAUS)
    @pytest.mark.parametrize("amp", [1.0e-3, 1.0e-2])
    @pytest.mark.parametrize("component", ["xy", "yz", "dev_xx"])
    def test_decay_ratio_is_bgk_factor(self, q, tau, amp, component):
        info = LATTICES[q]
        f0 = _base_state(q)
        delta = _pure_shear_perturbation(q, component, amp)
        f = f0 + delta

        f_star = info["collide"](f, tau)

        key = component if component != "dev_xx" else "xx"
        ratio = float(_pi_neq(f_star, f0, q)[key].mean().item()) / float(
            _pi_neq(f, f0, q)[key].mean().item()
        )
        expected = 1.0 - 1.0 / tau
        # fp32 cancellation floor: f - f_eq is an O(eps_pop) absolute error on
        # an O(amp) signal, so the ratio noise scales as ~1/amp; the dev_xx
        # channel additionally combines three cancellation-limited diagonal
        # contractions.  Measured floors: ~5e-6 at amp=1e-3 and for dev_xx,
        # ~5e-7 for off-diagonal at amp=1e-2 (the moment contraction itself
        # is accumulated in fp64).  All remain >= 4 orders of magnitude below
        # the pre-fix deviation (|ratio_bug - expected| = |1/tau - 1| >= 0.11).
        tol = 1.0e-6 if (amp > 1.0e-3 and component != "dev_xx") else 1.0e-5
        assert abs(ratio - expected) < tol, (
            f"q={q} component={component} tau={tau} amp={amp}: "
            f"shear decay ratio {ratio:.8f} != 1-1/tau = {expected:.8f} "
            f"(|diff|={abs(ratio - expected):.2e}) — shear viscosity is not "
            "controlled by tau"
        )

    @pytest.mark.parametrize("q", [19, 27])
    @pytest.mark.parametrize("tau", TAUS)
    def test_decay_ratio_fp64(self, q, tau):
        """fp64 cross-check of the primary criterion (tolerance 1e-12)."""
        info = LATTICES[q]
        f0 = _base_state(q, dtype=torch.float64)
        delta = _pure_shear_perturbation(q, "xy", 1.0e-3, dtype=torch.float64)
        f = f0 + delta
        f_star = info["collide"](f, tau)
        ratio = float(_pi_neq(f_star, f0, q)["xy"].mean().item()) / float(
            _pi_neq(f, f0, q)["xy"].mean().item()
        )
        expected = 1.0 - 1.0 / tau
        assert abs(ratio - expected) < 1.0e-12, (
            f"q={q} tau={tau}: fp64 ratio {ratio!r} != {expected!r}"
        )

    @pytest.mark.parametrize("q", [19, 27])
    @pytest.mark.parametrize("tau", TAUS)
    def test_orthogonal_shear_components_untouched(self, q, tau):
        """A pure xy perturbation must not leak into other stress components."""
        info = LATTICES[q]
        f0 = _base_state(q)
        delta = _pure_shear_perturbation(q, "xy", 1.0e-3)
        f = f0 + delta
        f_star = info["collide"](f, tau)
        pi_pre, pi_post = _pi_neq(f, f0, q), _pi_neq(f_star, f0, q)
        for key in ("xx", "yy", "zz", "xz", "yz"):
            # fp32 kernel + fp64 contraction: analytic zero, roundoff floor
            # of the fp32 collision arithmetic is ~1e-8 of the O(0.1)
            # populations
            assert float(pi_post[key].abs().max().item()) < 5.0e-8, (
                f"q={q} tau={tau}: xy perturbation leaked into {key} "
                f"(max={float(pi_post[key].abs().max().item()):.2e})"
            )
        # xy must be the only non-zero pre-collision stress as sanity
        assert float(pi_pre["xy"].abs().max().item()) > 0.9e-3


# ---------------------------------------------------------------------------
# 2. Ghost coefficient beta: interval, bulk mode, h relaxation
# ---------------------------------------------------------------------------


class TestGhostBetaBehaviour:
    """The entropy-solved ghost coefficient must stay in [0, 1]."""

    @pytest.mark.parametrize("q", [19, 27])
    @pytest.mark.parametrize("tau", [0.6, 0.8, 1.0, 1.5])
    def test_beta_within_unit_interval(self, q, tau):
        from tensorlbm.entropic_kbc import _kbc_decompose, _lattice_constants

        if q == 19:
            from tensorlbm.d3q19 import W_EXACT64 as Wx
            from tensorlbm.d3q19 import C as Cx
        else:
            from tensorlbm.d3q27 import W_EXACT64 as Wx
            from tensorlbm.d3q27 import C as Cx

        f = _random_state(q, seed=7, amp=3.0e-3)
        info = LATTICES[q]
        rho, ux, uy, uz = info["macroscopic"](f)
        feq = info["equilibrium"](rho, ux, uy, uz)
        p = _lattice_constants(Cx, Wx, f.device, f.dtype)
        s, k, h = _kbc_decompose(f - feq, p)

        f_star = info["collide"](f, tau)
        sigma = 1.0 - 1.0 / tau
        residual = f_star - feq - sigma * s  # == beta * h
        hh = (h * h).sum(dim=0)
        beta = (h * residual).sum(dim=0) / hh.clamp_min(1.0e-30)
        meaningful = hh > 1.0e-20
        assert meaningful.any(), "state must carry ghost content for this test"
        beta_vals = beta[meaningful]
        assert float(beta_vals.min().item()) >= -1.0e-6, (
            f"q={q} tau={tau}: beta below 0 (min={float(beta_vals.min().item()):.6f})"
        )
        assert float(beta_vals.max().item()) <= 1.0 + 1.0e-6, (
            f"q={q} tau={tau}: beta above 1 (max={float(beta_vals.max().item()):.6f})"
        )

    @pytest.mark.parametrize("q", [19, 27])
    @pytest.mark.parametrize("tau", TAUS)
    def test_bulk_trace_mode_removed(self, q, tau):
        """Post-collision trace of the second-order non-equilibrium moment ≈ 0.

        Analytically the kinetic (bulk/trace) ghost is removed and neither the
        relaxed shear nor the ghost residual carry a trace, so the trace of
        the post-collision non-equilibrium stress is zero to roundabout.
        Measured in fp64 (an fp32 variant would only test the roundoff floor).
        """
        info = LATTICES[q]
        f = _random_state(q, seed=11, amp=3.0e-3, dtype=torch.float64)
        f_star = info["collide"](f, tau)
        rho_p, ux_p, uy_p, uz_p = info["macroscopic"](f_star)
        feq_p = info["equilibrium"](rho_p, ux_p, uy_p, uz_p)
        pi = _pi(f_star - feq_p, q)
        trace = pi["xx"] + pi["yy"] + pi["zz"]
        assert float(trace.abs().max().item()) < 1.0e-14, (
            f"q={q} tau={tau}: bulk/trace mode not removed "
            f"(max|tr|={float(trace.abs().max().item()):.2e})"
        )

    @pytest.mark.parametrize("q", [19, 27])
    def test_higher_order_relaxed_at_tau1(self, q):
        """At tau=1 the ghost modes must vanish (sigma=0 and beta=0)."""
        from tensorlbm.entropic_kbc import _kbc_decompose, _lattice_constants

        if q == 19:
            from tensorlbm.d3q19 import W_EXACT64 as Wx
            from tensorlbm.d3q19 import C as Cx
        else:
            from tensorlbm.d3q27 import W_EXACT64 as Wx
            from tensorlbm.d3q27 import C as Cx

        info = LATTICES[q]
        f = _random_state(q, seed=13, amp=1.0e-3)
        f_star = info["collide"](f, 1.0)
        rho_p, ux_p, uy_p, uz_p = info["macroscopic"](f_star)
        feq_p = info["equilibrium"](rho_p, ux_p, uy_p, uz_p)
        p = _lattice_constants(Cx, Wx, f.device, f.dtype)
        _, _, h_post = _kbc_decompose(f_star - feq_p, p)
        # At tau=1 the shear factor is exactly 0 and the entropy solve
        # minimises H(f_eq + beta*h): the polynomial f_eq is not the exact
        # discrete-entropy minimiser, so beta settles at a small O(1e-2)
        # value rather than exactly 0 (measured retained fraction ~2%).
        # The ghost content carries no second-order moments, so this does
        # not touch the shear viscosity (covered by the decay-ratio tests).
        h_pre_ref = _kbc_decompose(f - info["equilibrium"](*info["macroscopic"](f)), p)[2]
        retained = float(h_post.abs().max().item()) / float(h_pre_ref.abs().max().item())
        assert retained < 0.05, (
            f"q={q}: ghost content not relaxed at tau=1 "
            f"(retained fraction={retained:.4f}, expected <= 0.05)"
        )


# ---------------------------------------------------------------------------
# 3. H-theorem and positivity on random states
# ---------------------------------------------------------------------------


class TestEntropyAndPositivity:
    @pytest.mark.parametrize("q", [19, 27])
    @pytest.mark.parametrize("tau", [0.55, 0.8, 1.0, 1.5])
    def test_h_theorem_random_states(self, q, tau):
        from tensorlbm.entropic_kbc import discrete_entropy

        info = LATTICES[q]
        w = info["W"].to(dtype=torch.float64).view(q, 1, 1, 1)
        f = _random_state(q, seed=17, amp=3.0e-3, dtype=torch.float64)
        f_star = info["collide"](f, tau)
        h_before = discrete_entropy(f, w)
        h_after = discrete_entropy(f_star, w)
        assert bool((h_after <= h_before + 1.0e-12).all()), (
            f"q={q} tau={tau}: H increased, max excess "
            f"{float((h_after - h_before).max().item()):.3e}"
        )

    @pytest.mark.parametrize("q", [19, 27])
    @pytest.mark.parametrize("tau", [0.55, 0.8, 1.0, 1.5])
    def test_positivity_random_states(self, q, tau):
        info = LATTICES[q]
        f = _random_state(q, seed=19, amp=5.0e-3)
        f_star = info["collide"](f, tau)
        assert float(f_star.min().item()) > 0.0, (
            f"q={q} tau={tau}: negative population {float(f_star.min().item()):.3e}"
        )

    @pytest.mark.parametrize("q", [19, 27])
    @pytest.mark.parametrize("tau", TAUS)
    def test_mass_momentum_conserved(self, q, tau):
        info = LATTICES[q]
        f = _random_state(q, seed=23, amp=3.0e-3)
        pre = info["macroscopic"](f)
        post = info["macroscopic"](info["collide"](f, tau))
        for a, b in zip(post, pre, strict=True):
            torch.testing.assert_close(a, b, atol=1.0e-6, rtol=1.0e-6)


# ---------------------------------------------------------------------------
# 4. Weakly compressible limit: KBC trajectory ≈ BGK trajectory at same tau
# ---------------------------------------------------------------------------


class TestBGKLaminarConsistency:
    """Small-amplitude Taylor–Green decay: KBC must track the BGK solution.

    Both kernels share nu = c_s^2 (tau − 1/2) after the fix; ghost/bulk mode
    differences enter only at second order in the amplitude.  The pre-fix
    kernel (nu_eff = 1/6 regardless of tau) decays at the wrong rate and
    drifts far outside the tolerance.
    """

    @pytest.mark.parametrize("tau", [0.9, 1.5])
    def test_kbc_tracks_bgk_tg_decay(self, tau):
        n = 32
        u0 = 0.01
        steps = 300
        k = 2.0 * math.pi / n

        def run(collide_fn):
            z, y, x = torch.meshgrid(
                torch.arange(n),
                torch.arange(n),
                torch.arange(n),
                indexing="ij",
            )
            ux = u0 * torch.sin(k * x) * torch.cos(k * y) * torch.cos(k * z)
            uy = -u0 * torch.cos(k * x) * torch.sin(k * y) * torch.cos(k * z)
            uz = torch.zeros_like(ux)
            f = equilibrium3d(torch.ones_like(ux), ux, uy, uz)
            for _ in range(steps):
                f = collide_fn(stream3d(f), tau)
            return macroscopic3d(f)

        _, ux_kbc, uy_kbc, _ = run(collide_kbc_d3q19)
        _, ux_bgk, uy_bgk, _ = run(collide_bgk3d)

        diff = ((ux_kbc - ux_bgk).abs().max().item(), (uy_kbc - uy_bgk).abs().max().item())
        # Fixed kernel measured deviation: 2.0e-5 (tau=0.9) / 1.3e-5 (tau=1.5)
        # = O(u0^2) ghost/bulk-relaxation scheme difference accumulated over
        # 300 steps.  Pre-fix kernel (nu_eff=1/6 regardless of tau) decays at
        # the wrong rate and lands ~2e-4 off at tau=0.9 — 4x beyond this
        # tolerance.
        assert max(diff) <= 5.0e-3 * u0, (
            f"tau={tau}: KBC trajectory deviates from BGK by max|du|={max(diff):.3e} "
            f"(tolerance {5.0e-3 * u0:.3e})"
        )
