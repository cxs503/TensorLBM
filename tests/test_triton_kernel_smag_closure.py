"""Issue #83: kernel-internal Smagorinsky closure tests (CM/CUMULANT no-op fix).

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
from tensorlbm.turbulence import _neq_stress_norm_3d, _smagorinsky_tau
from tensorlbm.triton_fused_obstacle import (
    apply_far_field_bc_6face,
    apply_mass_correction,
    triton_fused_obstacle_xfar_les,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA")

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
    f0 = equilibrium3d(rho0, ux0, torch.zeros_like(rho0),
                       torch.zeros_like(rho0))
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
        f0.clone(), NU_RE1E5, solid_i8, 0.0, 1.0, collision=collision)
    out_cs1 = triton_fused_obstacle_xfar_les(
        f0.clone(), NU_RE1E5, solid_i8, CS, 1.0, collision=collision)
    torch.cuda.synchronize()
    assert not torch.equal(out_cs0, out_cs1), (
        f"{collision}: Cs={CS} changed nothing — internal Smagorinsky "
        "is a silent no-op (issue #83 regression)"
    )
    diff = (out_cs0 - out_cs1).abs().max()
    assert float(diff) > 1e-6, (
        f"{collision}: max |Cs=0 - Cs=0.1| = {float(diff):.3e} too small "
        "for an active LES closure"
    )


# ---------------------------------------------------------------------------
# 2. internal tau_eff parity vs the faithful external chain (BGK probe)
# ---------------------------------------------------------------------------

def test_internal_smag_tau_matches_external_chain():
    solid_i8, f0 = _make_case()
    solid = solid_i8.bool()
    out = triton_fused_obstacle_xfar_les(
        f0.clone(), NU_RE1E5, solid_i8, CS, 1.0, collision="BGK")
    torch.cuda.synchronize()

    h = bounce_back_cells_3d(stream3d(f0), solid)
    rho, ux, uy, uz = macroscopic3d(h)
    feq = equilibrium3d(rho, ux, uy, uz)
    tau0 = 3.0 * NU_RE1E5 + 0.5
    tau_ext = _smagorinsky_tau(
        tau0, _neq_stress_norm_3d(h - feq), rho, CS)

    # BGK writes out = h - omega*(h - feq)  ->  recover omega per cell/vel.
    fneq = h - feq
    omega_k = (h - out) / fneq
    tau_k = 1.0 / omega_k
    tau_k = torch.where(torch.isfinite(tau_k), tau_k,
                        torch.full_like(tau_k, 1e30))

    # The probe (h - out)/(h - feq) amplifies the fp32 rounding of the
    # stored ``out`` by 1/|fneq|, so parity is asserted on
    # well-conditioned lanes with a bar safely above that noise floor
    # (measured 2.3e-5 at |fneq| > 1e-3) and far below any real closure
    # discrepancy (a missing factor or wrong clamp is O(1)).
    rel = ((tau_k - tau_ext[None]).abs() / tau_ext.clamp(min=1e-6)[None])
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
