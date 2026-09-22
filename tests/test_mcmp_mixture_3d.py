"""W7-C discriminative tests: 3D MCMP mixture-momentum equilibrium (u_eq="mixture").

Mirror of ``tests/test_mcmp_mixture.py`` (2D, W5-A / PR #308) for the D3Q19
port in ``tensorlbm.multiphase3d.collide_sc_two_component_3d`` and the D3Q27
counterpart ``collide_sc_two_component_27``:

T0  default-path bitwise identity: a call WITHOUT ``u_eq`` must be
    bit-identical to an explicit ``u_eq="self"`` call on a deterministic
    random-state battery (G x tau x forcing x guo, D3Q19 + D3Q27) — int32
    bit patterns, NaN-safe and stricter than ``torch.equal``.  (The
    migration bitwise identity against the *pre-patch* module — 112/112
    cases including solid masks, SGS sub-branches and chained trajectories
    — was evidenced on the development server and recorded in the W7-C
    staging NOTES.)
T1  closed-domain F=0 momentum behaviour (uniform periodic field): total
    momentum is conserved by both conventions, but only "mixture" damps the
    *relative* component velocity — "self" leaves it unchanged forever
    (f = feq(u_sigma) is a fixed point).  With unequal relaxation times the
    mixture exchange transiently biases total momentum by
    O(d(1/tau) * |u1-u2|) (2D T1 mirror, same thresholds).
T2  rho2 -> 0 single-component limit: both conventions converge, deviation
    shrinks monotonically with rho2 (2D T2 mirror).
T3  Galilean boost invariance of a 3D interface (periodic, no walls) at the
    mixture-stable working point G_12 = -1.0 (2D T3 mirror).  Measured on
    D3Q19 48x12x12 / 800 steps: dev_u = 3.45e-4, dev_rho = 2.27e-6 — inside
    the 2D-locked thresholds (1e-3 / 2e-5).
T4  single-collision closed-form momentum exchange: per cell,
    dM_sigma = F_sigma + (rho_sigma/tau_sigma)(u_mix - u_sigma)  (fp32
    roundoff; measured max |residual| <= 4.3e-8, threshold 1e-6).  Equal-tau
    total momentum conservation on random periodic states over 20
    collide+stream steps (measured <= 5.3e-6 vs |P0| ~ 2.6).  Includes the
    D3Q27 closed-form check.
T5  mixture vs self discrimination: outputs differ (bit level) on states
    with nonzero relative velocity and agree to fp32 barycentre-recompute
    roundoff when u1 == u2 pointwise; invalid ``u_eq`` and the
    ``use_guo=True`` x ``u_eq="mixture"`` combination raise ValueError.

The unequal-tau mixture pathology (structural net momentum source
(1/tau2 - 1/tau1) rho1 rho2 (u1-u2)/rho_tot -> NaN at strong segregation)
is a docstring note on the patched functions, established by W5-A in 2D;
it is algebraically dimension-independent and deliberately NOT re-derived
here.
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.d3q27 import equilibrium27, macroscopic27
from tensorlbm.multiphase3d import collide_sc_two_component_3d
from tensorlbm.multiphase3d_d3q27 import collide_sc_two_component_27
from tensorlbm.solver3d import stream3d

# GPU discipline: GPU5 only (cuda:0 when launched under CUDA_VISIBLE_DEVICES=5).
_NGPU = torch.cuda.device_count() if torch.cuda.is_available() else 0
DEV = torch.device(f"cuda:{min(5, _NGPU - 1)}") if _NGPU else torch.device("cpu")

G12 = -2.5  # library sign convention: segregation working point (W4-B record)
G12_STABLE = -1.0  # mixture-stable working point for unequal taus (2D explore E3)


# ------------------------------------------------------------------ helpers


def _bit_equal(x: torch.Tensor, y: torch.Tensor) -> bool:
    """Strict bitwise equality (int32 bit patterns; NaN == identical NaN)."""
    return torch.equal(x.contiguous().view(torch.int32), y.contiguous().view(torch.int32))


def _uniform_pair(nz, ny, nx, rho1, rho2, u1x, u2x):
    r1 = torch.full((nz, ny, nx), rho1, device=DEV)
    r2 = torch.full((nz, ny, nx), rho2, device=DEV)
    u1 = torch.full((nz, ny, nx), u1x, device=DEV)
    u2 = torch.full((nz, ny, nx), u2x, device=DEV)
    z = torch.zeros_like(u1)
    return equilibrium3d(r1, u1, z, z), equilibrium3d(r2, u2, z, z)


def _step(f1, f2, tau1, tau2, u_eq, gx=0.0, gy=0.0, gz=0.0, g12=G12):
    f1, f2 = collide_sc_two_component_3d(
        f1, f2, G_12=g12, tau1=tau1, tau2=tau2, gx=gx, gy=gy, gz=gz, u_eq=u_eq
    )
    return stream3d(f1), stream3d(f2)


def _moments(f1, f2):
    r1, ux1, _, _ = macroscopic3d(f1)
    r2, ux2, _, _ = macroscopic3d(f2)
    p_tot = float((r1 * ux1 + r2 * ux2).sum())
    u_rel = float((ux1 - ux2).abs().max())
    p_scale = float((r1 * ux1).abs().sum() + (r2 * ux2).abs().sum())
    return p_tot, u_rel, p_scale


def _bounded_state(Q, nz, ny, nx, seed, same_velocity=False):
    """Deterministic state with strictly positive densities and O(0.1)
    velocities (multiplicative noise keeps every cell density bounded away
    from zero, so the 1e-12 clamp never engages and all fields stay finite).

    ``same_velocity=True`` builds pure equilibria of both components from ONE
    velocity field (no noise), so the macroscopic u1 and u2 agree to fp32
    roundoff and the barycentre u_mix reproduces u up to ~1 ulp."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    rho1 = 0.3 + 0.7 * torch.rand((nz, ny, nx), generator=gen)
    rho2 = 0.3 + 0.7 * torch.rand((nz, ny, nx), generator=gen)
    u1 = 0.1 * (torch.rand((3, nz, ny, nx), generator=gen) - 0.5)
    if same_velocity:
        u2 = u1
        eq = equilibrium3d if Q == 19 else equilibrium27
        f1 = eq(rho1, u1[0], u1[1], u1[2]).to(DEV)
        f2 = eq(rho2, u2[0], u2[1], u2[2]).to(DEV)
        return f1, f2
    u2 = 0.1 * (torch.rand((3, nz, ny, nx), generator=gen) - 0.5)
    eq = equilibrium3d if Q == 19 else equilibrium27
    f1 = eq(rho1, u1[0], u1[1], u1[2]) * (1.0 + 0.05 * torch.randn((Q, nz, ny, nx), generator=gen))
    f2 = eq(rho2, u2[0], u2[1], u2[2]) * (1.0 + 0.05 * torch.randn((Q, nz, ny, nx), generator=gen))
    return f1.to(DEV), f2.to(DEV)


# ------------------------------------------------------------------ T0


def test_t0_default_path_bitwise_identical_to_explicit_self():
    """Default path (u_eq unset) must be bit-identical to u_eq="self" on a
    deterministic battery covering both lattices, unequal taus, interaction
    strengths, gravity and the Guo forcing branch.  (Bitwise identity of the
    default path against the *pre-patch* module — 112/112 cases including
    solid masks, SGS sub-branches and chained trajectories — was evidenced
    on the development server; see the W7-C staging NOTES.)"""
    # both entry points must expose the new parameter
    assert "u_eq" in collide_sc_two_component_3d.__code__.co_varnames
    assert "u_eq" in collide_sc_two_component_27.__code__.co_varnames

    cases19 = [
        dict(G_12=0.0, tau1=1.0, tau2=1.0),
        dict(G_12=-2.5, tau1=1.0, tau2=0.75, gz=-1e-3),
        dict(G_12=0.9, tau1=0.8, tau2=0.8, use_guo=True),
    ]
    for seed, kw in enumerate(cases19):
        f1, f2 = _bounded_state(19, 8, 8, 12, seed=101 + seed)
        d1, d2 = collide_sc_two_component_3d(f1, f2, **kw)
        s1, s2 = collide_sc_two_component_3d(f1, f2, u_eq="self", **kw)
        assert _bit_equal(d1, s1) and _bit_equal(d2, s2), kw

    f1, f2 = _bounded_state(27, 8, 8, 12, seed=107)
    kw = dict(G_12=-2.5, tau1=1.0, tau2=0.75)
    d1, d2 = collide_sc_two_component_27(f1, f2, **kw)
    s1, s2 = collide_sc_two_component_27(f1, f2, u_eq="self", **kw)
    assert _bit_equal(d1, s1) and _bit_equal(d2, s2)


# ------------------------------------------------------------------ T1


def test_t1_closed_domain_momentum_and_relative_velocity_damping():
    """F=0 closed (periodic) domain, uniform densities -> SC force vanishes.

    Equal taus: mixture conserves total momentum *and* damps the relative
    velocity; self conserves total momentum but the relative velocity is a
    fixed point (missing inter-component momentum coupling)."""
    nz, ny, nx = 8, 8, 16
    rho1, rho2 = 0.7, 0.3
    u1x, u2x = 0.05, -0.05 * rho1 / rho2  # zero total momentum
    u_rel0 = abs(u1x - u2x)
    m1_scale = rho1 * abs(u1x) * nz * ny * nx
    p_scale0 = (rho1 * abs(u1x) + rho2 * abs(u2x)) * nz * ny * nx

    for u_eq, expect_damped in (("mixture", True), ("self", False)):
        f1, f2 = _uniform_pair(nz, ny, nx, rho1, rho2, u1x, u2x)
        for _ in range(200):
            f1, f2 = _step(f1, f2, 1.0, 1.0, u_eq)
        p_tot, u_rel, p_scale = _moments(f1, f2)
        assert abs(p_tot) < 1e-5 * p_scale0, (u_eq, p_tot, p_scale0)
        if expect_damped:
            assert u_rel < 1e-3 * u_rel0, (u_eq, u_rel, u_rel0)
        else:
            assert u_rel > 0.999 * u_rel0, (u_eq, u_rel, u_rel0)

    # Unequal taus: self still never damps and still conserves; mixture damps
    # with a quantified transient total-momentum bias O(d(1/tau)*|u1-u2|).
    for u_eq, expect_damped in (("mixture", True), ("self", False)):
        f1, f2 = _uniform_pair(nz, ny, nx, rho1, rho2, u1x, u2x)
        for _ in range(200):
            f1, f2 = _step(f1, f2, 1.0, 0.75, u_eq)
        p_tot, u_rel, p_scale = _moments(f1, f2)
        if expect_damped:
            assert u_rel < 0.05 * u_rel0, (u_eq, u_rel, u_rel0)
            assert abs(p_tot) < 0.5 * m1_scale, (u_eq, p_tot, m1_scale)
        else:
            assert u_rel > 0.999 * u_rel0, (u_eq, u_rel, u_rel0)
            assert abs(p_tot) < 1e-5 * p_scale0, (u_eq, p_tot, p_scale0)


# ------------------------------------------------------------------ T2


def test_t2_single_component_limit_agreement():
    """rho2 -> 0: mixture and self converge (deviation shrinks with rho2)."""
    nz, ny, nx = 8, 8, 16
    devs = {}
    for rho2 in (1e-2, 1e-4):
        finals = {}
        for u_eq in ("mixture", "self"):
            f1, f2 = _uniform_pair(nz, ny, nx, 1.0 - rho2, rho2, 0.03, 0.02)
            for _ in range(300):
                f1, f2 = _step(f1, f2, 1.0, 0.75, u_eq, gx=1e-4)
            r1, ux1, _, _ = macroscopic3d(f1)
            finals[u_eq] = float(ux1.max())
        devs[rho2] = abs(finals["mixture"] - finals["self"])
    uscale = 0.03
    assert devs[1e-2] < 1e-2 * uscale, devs
    assert devs[1e-4] < devs[1e-2] / 3.0, devs


# ------------------------------------------------------------------ T3


def _galilean_dev(u_eq, u0=0.05, nz=48, ny=12, nx=12, steps=800):
    """Interface under boost: returns (max|u_B - u0 - u_A|, max|rho_w,B - rho_w,A|)."""

    def run(boost):
        zs = torch.arange(nz, dtype=torch.float32, device=DEV).view(nz, 1, 1)
        alpha = 0.5 * (1.0 - torch.tanh((zs - nz / 2.0) / 3.0))
        alpha = alpha.expand(nz, ny, nx)
        rw = 0.7 * alpha + 0.3 * (1 - alpha)
        rg = 0.3 * alpha + 0.7 * (1 - alpha)
        ux = torch.full((nz, ny, nx), boost, device=DEV)
        z = torch.zeros_like(ux)
        f1, f2 = equilibrium3d(rw, ux, z, z), equilibrium3d(rg, ux, z, z)
        for _ in range(steps):
            f1, f2 = _step(f1, f2, 1.0, 0.75, u_eq, g12=G12_STABLE)
        r1, ux1, _, _ = macroscopic3d(f1)
        r2, ux2, _, _ = macroscopic3d(f2)
        rt = (r1 + r2).clamp(min=1e-12)
        return (r1 * ux1 + r2 * ux2) / rt, r1

    uxA, rwA = run(0.0)
    uxB, rwB = run(u0)
    return float((uxB - u0 - uxA).abs().max()), float((rwB - rwA).abs().max())


def test_t3_galilean_boost_invariance_interface():
    """Boost by u0 at the mixture-stable working point G_12=-1.0: mixture
    physics is invariant up to the lattice-velocity truncation error
    (2D-locked thresholds; 3D measured 3.45e-4 / 2.27e-6)."""
    dev_u, dev_rho = _galilean_dev("mixture")
    dev_u_self, _ = _galilean_dev("self")
    assert dev_u < 1.0e-3, (dev_u, dev_u_self)
    assert dev_rho < 2.0e-5, dev_rho


# ------------------------------------------------------------------ T4


def _closed_form_check(collide, macroscopic, force, Q, nz, ny, nx, seed):
    """dM_sigma = F_sigma + (rho_sigma/tau_sigma)(u_mix - u_sigma) per cell."""
    f1, f2 = _bounded_state(Q, nz, ny, nx, seed=seed)
    worst = 0.0
    for G, (gx, gy, gz), (tau1, tau2) in (
        (0.0, (0.0, 0.0, 0.0), (1.0, 1.0)),
        (0.0, (0.0, 0.0, 0.0), (1.0, 0.75)),
        (0.9, (1e-4, -1e-4, 2e-4), (1.0, 0.75)),
    ):
        o1, o2 = collide(f1, f2, G_12=G, tau1=tau1, tau2=tau2, gx=gx, gy=gy, gz=gz, u_eq="mixture")
        r1, ux1, uy1, uz1 = macroscopic(f1)
        r2, ux2, uy2, uz2 = macroscopic(f2)
        n1, vx1, vy1, vz1 = macroscopic(o1)
        n2, vx2, vy2, vz2 = macroscopic(o2)
        Fx1, Fy1, Fz1, Fx2, Fy2, Fz2 = force(r1, r2, G, gx, gy, gz)
        r1s, r2s = r1.clamp(min=1e-12), r2.clamp(min=1e-12)
        rt = r1s + r2s
        for sigma in (1, 2):
            if sigma == 1:
                rs, us, vs, ws = r1s, ux1, uy1, uz1
                ns, uvx, uvy, uvz = n1, vx1, vy1, vz1
                Fstk, tau = torch.stack([Fx1, Fy1, Fz1]), tau1
            else:
                rs, us, vs, ws = r2s, ux2, uy2, uz2
                ns, uvx, uvy, uvz = n2, vx2, vy2, vz2
                Fstk, tau = torch.stack([Fx2, Fy2, Fz2]), tau2
            umx = (r1s * ux1 + r2s * ux2) / rt
            umy = (r1s * uy1 + r2s * uy2) / rt
            umz = (r1s * uz1 + r2s * uz2) / rt
            closed = torch.stack(
                [rs * (umx - us) / tau, rs * (umy - vs) / tau, rs * (umz - ws) / tau]
            )
            dM = torch.stack([ns * uvx - rs * us, ns * uvy - rs * vs, ns * uvz - rs * ws])
            worst = max(worst, float((dM - Fstk - closed).abs().max()))
    return worst


def test_t4a_closed_form_exchange_d3q19():
    """Single-collision momentum exchange matches the Shan-Doolen closed form
    (measured <= 4.3e-8 in fp32)."""
    from tensorlbm.multiphase3d import sc_two_component_force_3d

    worst = _closed_form_check(
        collide_sc_two_component_3d,
        macroscopic3d,
        sc_two_component_force_3d,
        19,
        12,
        12,
        24,
        seed=777,
    )
    assert worst < 1e-6, worst


def test_t4b_closed_form_exchange_d3q27():
    """D3Q27 port obeys the same closed form (measured same magnitude)."""
    from tensorlbm.multiphase3d_d3q27 import sc_two_component_force_27

    worst = _closed_form_check(
        collide_sc_two_component_27,
        macroscopic27,
        sc_two_component_force_27,
        27,
        8,
        8,
        12,
        seed=778,
    )
    assert worst < 1e-6, worst


def test_t4c_equal_tau_total_momentum_conservation():
    """Random periodic state, 20 collide+stream steps, equal taus, no external
    force: total momentum conserved for BOTH conventions (SC pair forces and
    the mixture exchange cancel in the periodic total; measured <= 5.3e-6)."""
    f1, f2 = _bounded_state(19, 12, 12, 24, seed=779)
    r1, ux1, uy1, uz1 = macroscopic3d(f1)
    r2, ux2, uy2, uz2 = macroscopic3d(f2)
    p0_axes = [float((r1 * u + r2 * v).sum()) for u, v in ((ux1, ux2), (uy1, uy2), (uz1, uz2))]
    p0 = max(abs(p) for p in p0_axes)
    for u_eq in ("mixture", "self"):
        a, b = f1, f2
        for _ in range(20):
            a, b = _step(a, b, 1.0, 1.0, u_eq, g12=0.9)
        na1, ax1, ay1, az1 = macroscopic3d(a)
        na2, ax2, ay2, az2 = macroscopic3d(b)
        for axis, (u, v) in enumerate(((ax1, ax2), (ay1, ay2), (az1, az2))):
            dp = abs(float((na1 * u + na2 * v).sum()) - p0_axes[axis])
            assert dp < 1e-4 * max(p0, 1.0), (u_eq, dp, p0)


# ------------------------------------------------------------------ T5


def test_t5_mixture_self_discrimination():
    """mixture != self on nonzero-relative-velocity states (bit level, with a
    meaningful magnitude); u1 == u2 pointwise -> agreement to fp32 barycentre
    roundoff; error paths raise."""
    f1, f2 = _bounded_state(19, 12, 12, 24, seed=123)
    m1, m2 = collide_sc_two_component_3d(f1, f2, G_12=0.9, tau1=1.0, tau2=0.75, u_eq="mixture")
    s1, s2 = collide_sc_two_component_3d(f1, f2, G_12=0.9, tau1=1.0, tau2=0.75, u_eq="self")
    assert not _bit_equal(m1, s1)
    assert float((m1 - s1).abs().max()) > 1e-6, float((m1 - s1).abs().max())

    # u1 == u2 pointwise: u_mix reproduces u up to 1-ulp barycentre recompute
    g1, g2 = _bounded_state(19, 12, 12, 24, seed=124, same_velocity=True)
    q1, q2 = collide_sc_two_component_3d(g1, g2, G_12=0.9, tau1=1.0, tau2=0.75, u_eq="mixture")
    w1, w2 = collide_sc_two_component_3d(g1, g2, G_12=0.9, tau1=1.0, tau2=0.75, u_eq="self")
    assert float((q1 - w1).abs().max()) < 1e-6, float((q1 - w1).abs().max())

    # error paths, both lattices
    for collide in (collide_sc_two_component_3d, collide_sc_two_component_27):
        with pytest.raises(ValueError, match="u_eq must be"):
            collide(f1, f2, u_eq="bogus")
        with pytest.raises(ValueError, match="use_guo=False"):
            collide(f1, f2, u_eq="mixture", use_guo=True)
