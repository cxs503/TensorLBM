"""Issue #83: kernel-internal Smagorinsky closure + split-wall mode tests.

Guards three properties of
``tensorlbm.triton_fused_obstacle.triton_fused_obstacle_xfar_les``:

1. **Internal Smagorinsky is NOT a silent no-op** for the CM/CUMULANT
   families.  Before the fix, the ``USE_EXTERNAL_TAU=False`` branches of
   those families ignored ``Cs`` entirely (constant molecular omega), so
   ``Cs=0.1`` and ``Cs=0.0`` produced bit-identical output while the
   caller believed LES was active — under-resolved CUMULANT at Re=1e5
   then NaN'd (n=320 ~step 4800, n=1024 w8 step 7974).  With the fix,
   the kernel applies the production Hou-closure
   (:func:`tensorlbm.turbulence._smagorinsky_tau` semantics) in all
   three families, so ``Cs > 0`` must change the result.
2. **Internal tau_eff == external-chain tau_eff** (BGK probe): the
   per-cell omega the kernel applies must equal 1/tau_eff of the
   faithful external chain
   ``_smagorinsky_tau(tau0, _neq_stress_norm_3d(h - feq), rho)`` on the
   same post-stream, pre-bounce-back state ``h = BB(stream(f))``.
3. **wall="split" reproduces wall="fused"** — same wrapper, one step
   and multi-step (far-field BC + mass correction), all three families
   x Cs in {0, 0.1}, forces included.  BGK is bit-for-bit identical in
   every configuration, and so are CM/CUMULANT whenever the LES closure
   is active (Cs > 0 — the production configurations).  For CM/CUMULANT
   at Cs = 0 the split main pass compiles the reflected gather out, and
   Triton's binary specialization can re-schedule the CM/CUMULANT
   cascade reductions by 1 ULP on ~0.1% of lanes (measured: 1.5e-07 on
   2071/2.1M lanes CM, 3e-08 on 11.5k/2.1M CUMULANT); those two cases
   are therefore asserted to the 1-ULP bound instead of exact bits.
   The default wall="fused" mode never specializes anything away and
   stays bitwise the historical kernel for all families.

Requires CUDA.  Run:
  CUDA_VISIBLE_DEVICES=0 pytest tests/test_triton_kernel_smag_split_wall.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.boundaries3d import bounce_back_cells_3d, sphere_mask
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import stream3d
from tensorlbm.triton_fused_obstacle import (
    apply_far_field_bc_6face,
    apply_mass_correction,
    triton_fused_obstacle_xfar_les,
)
from tensorlbm.turbulence import _neq_stress_norm_3d, _smagorinsky_tau

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

DEV = "cuda:0"
NX, NY, NZ = 32, 16, 16
U_IN = 0.05
# Production-like point: n=320, Re=1e5, u=0.05 (tau0 - 0.5 ~ 9e-4).
NU_RE1E5 = 0.05 * 0.6 * 320 / 1e5
CS = 0.1


def _make_case():
    solid = sphere_mask(NX, NY, NZ, 12.0, 8.0, 8.0, 4.5, DEV)
    rho0 = torch.ones((NZ, NY, NX), device=DEV)
    ux0 = torch.full_like(rho0, U_IN)
    ux0[solid] = 0.0
    f0 = equilibrium3d(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0))
    # Non-equilibrium perturbation so the SGS term is active somewhere.
    f0 = f0 + 0.02 * torch.randn_like(f0)
    return solid.to(torch.int8), f0


# ---------------------------------------------------------------------------
# 1. internal Smagorinsky must not be a no-op (regression: issue #83a)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("collision", ["BGK", "CM", "CUMULANT"])
def test_internal_smag_changes_solution(collision):
    solid_i8, f0 = _make_case()
    out_cs0 = triton_fused_obstacle_xfar_les(
        f0.clone(), NU_RE1E5, solid_i8, 0.0, 1.0, collision=collision
    )
    out_cs1 = triton_fused_obstacle_xfar_les(
        f0.clone(), NU_RE1E5, solid_i8, CS, 1.0, collision=collision
    )
    torch.cuda.synchronize()
    assert not torch.equal(out_cs0, out_cs1), (
        f"{collision}: Cs={CS} changed nothing — internal Smagorinsky "
        "is a silent no-op (issue #83 regression)"
    )
    diff = (out_cs0 - out_cs1).abs().max()
    assert float(diff) > 1e-6, (
        f"{collision}: max |Cs=0 - Cs=0.1| = {float(diff):.3e} too small for an active LES closure"
    )


# ---------------------------------------------------------------------------
# 2. internal tau_eff parity vs the faithful external chain (BGK probe)
# ---------------------------------------------------------------------------


def test_internal_smag_tau_matches_external_chain():
    solid_i8, f0 = _make_case()
    solid = solid_i8.bool()
    out = triton_fused_obstacle_xfar_les(f0.clone(), NU_RE1E5, solid_i8, CS, 1.0, collision="BGK")
    torch.cuda.synchronize()

    h = bounce_back_cells_3d(stream3d(f0), solid)
    rho, ux, uy, uz = macroscopic3d(h)
    feq = equilibrium3d(rho, ux, uy, uz)
    tau0 = 3.0 * NU_RE1E5 + 0.5
    tau_ext = _smagorinsky_tau(tau0, _neq_stress_norm_3d(h - feq), rho, CS)

    # BGK writes out = h - omega*(h - feq)  ->  recover omega per cell/vel.
    fneq = h - feq
    omega_k = (h - out) / fneq
    tau_k = 1.0 / omega_k
    tau_k = torch.where(torch.isfinite(tau_k), tau_k, torch.full_like(tau_k, 1e30))

    # The probe (h - out)/(h - feq) amplifies the fp32 rounding of the
    # stored ``out`` by 1/|fneq|, so parity is asserted on
    # well-conditioned lanes with a bar safely above that noise floor
    # (measured 2.3e-5 at |fneq| > 1e-3) and far below any real closure
    # discrepancy (a missing factor or wrong clamp is O(1)).
    rel = (tau_k - tau_ext[None]).abs() / tau_ext.clamp(min=1e-6)[None]
    probe_ok = fneq.abs() > 1e-3
    worst = rel[probe_ok]
    assert float(worst.max()) < 1e-4, (
        f"kernel-internal tau_eff deviates from the external chain: "
        f"max rel diff {float(worst.max()):.3e} over "
        f"{int(probe_ok.sum())} lanes"
    )
    # And the LES must actually act: tau_eff rises above molecular
    # somewhere with strong non-equilibrium.
    strong = fneq.abs().amax(0) > 1e-2
    assert strong.any() and float(tau_ext[strong].min()) > tau0, (
        "external tau_eff never exceeds the molecular tau — probe setup "
        "does not exercise the closure"
    )


# ---------------------------------------------------------------------------
# 3. wall="split" must be bitwise-identical to wall="fused"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("collision", ["BGK", "CM", "CUMULANT"])
@pytest.mark.parametrize("cs", [0.0, 0.1])
def test_split_wall_bitwise_single_step(collision, cs):
    solid_i8, f0 = _make_case()
    fx = torch.zeros((), device=DEV)
    fy = torch.zeros_like(fx)
    fz = torch.zeros_like(fx)

    fx.zero_()
    fy.zero_()
    fz.zero_()
    a, fxa, fya, fza = (
        triton_fused_obstacle_xfar_les(
            f0.clone(),
            NU_RE1E5,
            solid_i8,
            cs,
            1.0,
            collision=collision,
            wall="fused",
            fx_buf=fx,
            fy_buf=fy,
            fz_buf=fz,
        ),
        fx.item(),
        fy.item(),
        fz.item(),
    )

    fx.zero_()
    fy.zero_()
    fz.zero_()
    b, fxb, fyb, fzb = (
        triton_fused_obstacle_xfar_les(
            f0.clone(),
            NU_RE1E5,
            solid_i8,
            cs,
            1.0,
            collision=collision,
            wall="split",
            fx_buf=fx,
            fy_buf=fy,
            fz_buf=fz,
        ),
        fx.item(),
        fy.item(),
        fz.item(),
    )
    torch.cuda.synchronize()

    exact = torch.equal(a, b)
    maxd = float((a - b).abs().max()) if not exact else 0.0
    if collision == "BGK" or cs > 0.0:
        assert exact, (
            f"{collision} Cs={cs}: wall='split' is not bitwise-identical "
            f"to wall='fused' (max diff {maxd:.3e})"
        )
    else:
        # CM/CUMULANT at Cs=0: Triton binary specialization may re-schedule
        # the cascade by 1 ULP on a small lane subset (see module docstring).
        n_bad = int((a != b).sum())
        assert maxd <= 3e-07, (
            f"{collision} Cs={cs}: split vs fused max diff {maxd:.3e} "
            "exceeds the 1-ULP codegen-noise bound"
        )
        assert n_bad <= 0.02 * a.numel(), (
            f"{collision} Cs={cs}: {n_bad}/{a.numel()} lanes differ — "
            "far more than the ~0.1% codegen-noise population"
        )
    # Force scalars are accumulated with atomic_add (non-deterministic
    # ORDER, same terms); compare with tight tolerance, not exact bits.
    assert abs(fxa - fxb) <= 1e-5 * (1.0 + abs(fxa))
    assert abs(fya - fyb) <= 1e-5 * (1.0 + abs(fya))
    assert abs(fza - fzb) <= 1e-5 * (1.0 + abs(fza))


@pytest.mark.parametrize("collision", ["BGK", "CM", "CUMULANT"])
def test_split_wall_trajectory_bitwise(collision):
    """40-step trajectory with the LES closure active (Cs=0.1; BGK, CM
    and the production CUMULANT), far-field BC every step + mass
    correction every 10 — must stay bit-for-bit on every step."""
    solid_i8, f0 = _make_case()
    m0 = float(f0.sum().item())
    fa = f0.clone()
    fb = f0.clone()
    fx = torch.zeros((), device=DEV)
    fy = torch.zeros_like(fx)
    fz = torch.zeros_like(fx)
    for s in range(1, 41):
        fx.zero_()
        fy.zero_()
        fz.zero_()
        fa = triton_fused_obstacle_xfar_les(
            fa,
            NU_RE1E5,
            solid_i8,
            CS,
            1.0,
            collision=collision,
            wall="fused",
            fx_buf=fx,
            fy_buf=fy,
            fz_buf=fz,
        )
        fx.zero_()
        fy.zero_()
        fz.zero_()
        fb = triton_fused_obstacle_xfar_les(
            fb,
            NU_RE1E5,
            solid_i8,
            CS,
            1.0,
            collision=collision,
            wall="split",
            fx_buf=fx,
            fy_buf=fy,
            fz_buf=fz,
        )
        assert torch.equal(fa, fb), (
            f"trajectory diverged at step {s} (max diff {float((fa - fb).abs().max()):.3e})"
        )
        apply_far_field_bc_6face(fa, U_IN)
        apply_far_field_bc_6face(fb, U_IN)
        if s % 10 == 0:
            fa = apply_mass_correction(fa, m0)
            fb = apply_mass_correction(fb, m0)
    torch.cuda.synchronize()
    assert torch.equal(fa, fb)


# ---------------------------------------------------------------------------
# 4. the tau0 >= 1.0 regime: closure must be active and match the CPU chain
# ---------------------------------------------------------------------------
# Before 2026-08-22 the upper clamp was an absolute tau_eff <= 1.0, on both
# the CPU helpers (_TAU_EFF_MAX) and the kernel (_TAU_EFF_MAX_K).  Because
# tau_eff >= tau_mol always, every run at molecular tau >= 1.0 (omega <= 1)
# collapsed to exactly the no-SGS collision.  These tests pin the fixed
# regime: tau_mol = 1.0 exactly.


def _tau_one_case():
    """Same probe setup as _make_case but at nu_lb = 1/6 (tau_mol = 1.0)."""
    solid = sphere_mask(NX, NY, NZ, 12.0, 8.0, 8.0, 4.5, DEV)
    rho0 = torch.ones((NZ, NY, NX), device=DEV)
    ux0 = torch.full_like(rho0, U_IN)
    ux0[solid] = 0.0
    f0 = equilibrium3d(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0))
    torch.manual_seed(20260822)
    f0 = f0 + 0.05 * torch.randn_like(f0)
    return solid.to(torch.int8), f0


NU_TAU_ONE = 1.0 / 6.0  # tau_mol = 3*nu + 0.5 = 1.0


@pytest.mark.parametrize("collision", ["BGK", "CM", "CUMULANT"])
def test_smag_active_at_molecular_tau_one(collision):
    """tau_mol = 1.0: Cs > 0 must change the solution (was a silent no-op)."""
    solid_i8, f0 = _tau_one_case()
    out_cs0 = triton_fused_obstacle_xfar_les(
        f0.clone(), NU_TAU_ONE, solid_i8, 0.0, 1.0, collision=collision
    )
    out_cs1 = triton_fused_obstacle_xfar_les(
        f0.clone(), NU_TAU_ONE, solid_i8, CS, 1.0, collision=collision
    )
    torch.cuda.synchronize()
    assert not torch.equal(out_cs0, out_cs1), (
        f"{collision}: Cs={CS} changed nothing at tau_mol=1.0 — the absolute "
        "tau_eff<=1.0 clamp disabled the SGS closure (regression)"
    )
    # Magnitude bar (as in test #1): Cs=0 binary specialisation alone can
    # reshuffle reductions by ~1 ULP, so require a physically meaningful
    # change, not just any bit flip.
    diff = (out_cs0 - out_cs1).abs().max()
    assert float(diff) > 1e-6, (
        f"{collision}: max |Cs=0 - Cs=0.1| = {float(diff):.3e} at tau_mol=1.0 "
        "too small for an active LES closure"
    )


def test_internal_smag_tau_matches_external_chain_at_tau_one():
    """BGK probe at tau_mol = 1.0: kernel tau_eff must equal the CPU
    ``_smagorinsky_tau`` chain, which must itself exceed tau_mol."""
    solid_i8, f0 = _tau_one_case()
    solid = solid_i8.bool()
    out = triton_fused_obstacle_xfar_les(f0.clone(), NU_TAU_ONE, solid_i8, CS, 1.0, collision="BGK")
    torch.cuda.synchronize()

    h = bounce_back_cells_3d(stream3d(f0), solid)
    rho, ux, uy, uz = macroscopic3d(h)
    feq = equilibrium3d(rho, ux, uy, uz)
    tau0 = 3.0 * NU_TAU_ONE + 0.5
    tau_ext = _smagorinsky_tau(tau0, _neq_stress_norm_3d(h - feq), rho, CS)

    # The CPU chain must be active in this regime (it returned exactly
    # tau0 = 1.0 before the clamp fix).
    strong = (h - feq).abs().amax(0) > 1e-2
    assert strong.any() and float(tau_ext[strong].min()) > tau0, (
        "CPU tau_eff chain still clamped to the molecular tau at tau0 = 1.0"
    )

    fneq = h - feq
    omega_k = (h - out) / fneq
    tau_k = 1.0 / omega_k
    tau_k = torch.where(torch.isfinite(tau_k), tau_k, torch.full_like(tau_k, 1e30))

    rel = (tau_k - tau_ext[None]).abs() / tau_ext.clamp(min=1e-6)[None]
    probe_ok = fneq.abs() > 1e-3
    worst = rel[probe_ok]
    assert float(worst.max()) < 1e-4, (
        f"kernel-internal tau_eff deviates from the CPU chain at tau0=1.0: "
        f"max rel diff {float(worst.max()):.3e} over {int(probe_ok.sum())} lanes"
    )
