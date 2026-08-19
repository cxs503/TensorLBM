"""Parity tests: Triton fused obstacle kernel wall treatment vs production chain.

Guards two correctness properties of
``tensorlbm.triton_fused_obstacle.triton_fused_obstacle_xfar_les``:

1. **Full-way bounce-back gather** — the kernel's streaming stage must
   implement exactly the production wall semantics
   ``bounce_back_cells_3d(stream3d(f), solid)`` (pull form):
   ``f_eff[q, x] = f[q, x - c_q]`` for fluid destination cells (plain pull,
   including pulls whose SOURCE is solid) and ``f_eff[q, x] =
   f[opp(q), x + c_q]`` for solid destination cells.
2. **Canonical D3Q19 lattice tables** — the direction tables used by the
   kernel must agree bit-for-bit with ``tensorlbm.d3q19.C`` / ``W``
   (the upstream ``triton_fused.make_lattice_tensors`` tables had six
   wrong-sign lanes, q in {8, 10, 12, 14, 16, 18}).

Plus a short multi-step force-trajectory parity check against the
production PyTorch chain (miniature version of the n=96/128 validation).

Requires CUDA.  Run:
  CUDA_VISIBLE_DEVICES=0 pytest tests/test_triton_obstacle_wall_parity.py -q
"""
from __future__ import annotations

import sys

import pytest
import torch

sys.path.insert(0, "/nfs/wangxi/TensorLBM/src")

from tensorlbm.advanced_collision_contract import collide_advanced_3d
from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d, \
    sphere_mask
from tensorlbm.d3q19 import C, OPPOSITE, W, equilibrium3d, macroscopic3d
from tensorlbm.obstacles import compute_obstacle_forces_3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.suboff_cmk_kbc_runner import (
    SuboffCmkKbcConfig,
    _collide_with_sgs,
    _compute_sgs_tau_eff,
)
from tensorlbm.triton_fused_obstacle import (
    _lattice_tensors_canonical,
    apply_far_field_bc_6face,
    apply_mass_correction,
    triton_fused_obstacle_xfar_les,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA")

DEV = "cuda:0"
NX, NY, NZ = 32, 16, 16
U_IN = 0.05
NU = 0.005
TAU = 3.0 * NU + 0.5


def _make_case():
    solid = sphere_mask(NX, NY, NZ, 12.0, 8.0, 8.0, 4.5, DEV)
    rho0 = torch.ones((NZ, NY, NX), device=DEV)
    ux0 = torch.full_like(rho0, U_IN)
    ux0[solid] = 0.0
    f0 = equilibrium3d(rho0, ux0, torch.zeros_like(rho0),
                       torch.zeros_like(rho0))
    return solid, f0


# ---------------------------------------------------------------------------
# 1. lattice tables
# ---------------------------------------------------------------------------

def test_lattice_tables_match_canonical_d3q19():
    lat = _lattice_tensors_canonical(DEV)
    c_ref = C.to(DEV)
    for key in ("cxi", "cyi", "czi"):
        got = lat[key][:19]  # tables are zero-padded to 32 kernel lanes
        assert got.shape == (19,), got.shape
        col = {"cxi": 0, "cyi": 1, "czi": 2}[key]
        assert torch.equal(got.to(torch.int64), c_ref[:, col].to(torch.int64)), \
            f"{key} disagrees with d3q19.C (wrong-sign lanes?)"
    for key in ("cxf", "cyf", "czf"):
        col = {"cxf": 0, "cyf": 1, "czf": 2}[key]
        assert torch.allclose(lat[key][:19], C.to(DEV).float()[:, col]), key
    assert torch.allclose(lat["w"][:19], W.to(DEV).float())


# ---------------------------------------------------------------------------
# 2. single-step exact parity of the wall gather (whole domain, incl. walls)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("collision", ["BGK", "CM"])
def test_single_step_parity_full_way_bb(collision):
    """kernel(f0) must equal collide(BB(stream(f0))) everywhere.

    Single call, no external tau (scalar tau from nu_lb): isolates the
    gather + collision branch.  Periodic wrap in both paths, so the whole
    domain (solid, wall-adjacent and interior fluid) is compared.
    """
    solid, f0 = _make_case()
    solid_i8 = solid.to(torch.int8)
    f_eff_ref = bounce_back_cells_3d(stream3d(f0), solid)
    ref = collide_advanced_3d("d3q19", collision, f_eff_ref, tau=TAU)

    out = triton_fused_obstacle_xfar_les(
        f0.clone(), NU, solid_i8, 0.0, 1.0, collision=collision)
    torch.cuda.synchronize()

    d = (out - ref).abs()
    assert float(d.max()) < 1e-6, float(d.max())

    # the pre-fix kernel violated parity exactly at the wall: solid cells and
    # fluid cells pulling from solid sources
    wall = sphere_mask(NX, NY, NZ, 12.0, 8.0, 8.0, 5.5, DEV)
    nb = wall & ~solid
    assert float(d[:, solid].max()) < 1e-6
    assert float(d[:, nb].max()) < 1e-6


def test_gather_addressing_bitwise_canary():
    """Bitwise check of the gather addressing with an integer-encoded field.

    f[q, z, y, x] = q*2^18 + z*2^12 + y*2^6 + x  (exactly representable in
    fp32).  With omega ~ 1e-9 the kernel output decodes to the gathered
    source index; compare against the reference pull-form full-way BB.
    """
    solid, _ = _make_case()
    solid_i8 = solid.to(torch.int8)
    q_i = torch.arange(19, device=DEV)
    z_i = torch.arange(NZ, device=DEV).view(NZ, 1, 1)
    y_i = torch.arange(NY, device=DEV).view(1, NY, 1)
    x_i = torch.arange(NX, device=DEV).view(1, 1, NX)
    f_canary = (q_i.view(19, 1, 1, 1).float() * 2**18
                + z_i.float() * 2**12 + y_i.float() * 2**6 + x_i.float())

    nu_huge = 1e9  # omega = 1/(3*nu+0.5) ~ 3e-10 -> output ~ f_eff
    out = triton_fused_obstacle_xfar_les(
        f_canary.clone(), nu_huge, solid_i8, 0.0, 1.0, collision="BGK")
    torch.cuda.synchronize()
    got = out.round().to(torch.int64)

    # reference: f_eff = where(solid, f[opp, x + c_q], f[q, x - c_q])
    opp = OPPOSITE.to(DEV)
    ref = torch.empty_like(f_canary)
    for q in range(19):
        cx, cy, cz = int(C[q, 0]), int(C[q, 1]), int(C[q, 2])
        src_pull = f_canary[q].roll(shifts=(cz, cy, cx), dims=(0, 1, 2))
        src_refl = f_canary[int(opp[q])].roll(shifts=(-cz, -cy, -cx),
                                              dims=(0, 1, 2))
        # roll(shifts=c) moves element at x-c_q to x  (pull);
        # reflected lane reads the cell at x+c_q.
        ref[q] = torch.where(solid, src_refl, src_pull)

    mism = (got != ref.round().to(torch.int64))
    n_bad = int(mism.sum().item())
    assert n_bad == 0, (
        f"{n_bad}/{mism.numel()} lanes with wrong gather address; "
        f"per-q bad = {[int(mism[q].sum().item()) for q in range(19)]}")


# ---------------------------------------------------------------------------
# 3. force sampling phase (Ladd, post-stream pre-BB) parity
# ---------------------------------------------------------------------------

def test_force_buffer_matches_production_ladd():
    solid, f0 = _make_case()
    solid_i8 = solid.to(torch.int8)
    fx = torch.zeros((), device=DEV)
    fy = torch.zeros((), device=DEV)
    fz = torch.zeros((), device=DEV)
    triton_fused_obstacle_xfar_les(
        f0.clone(), NU, solid_i8, 0.0, 1.0, collision="BGK",
        fx_buf=fx, fy_buf=fy, fz_buf=fz)
    torch.cuda.synchronize()

    # production: force sampled on the streamed state (collide then stream),
    # before bounce-back
    f_c = collide_advanced_3d("d3q19", "BGK", f0, tau=TAU)
    f_s = stream3d(f_c)
    fx_r, fy_r, fz_r = compute_obstacle_forces_3d(f_s, solid)
    # fx strictly relative; transverse components are ~0 by symmetry, so
    # compare them against the fx scale
    scale = max(abs(float(fx_r)), 1e-30)
    for got, want, name in ((fx, fx_r, "fx"), (fy, fy_r, "fy"),
                            (fz, fz_r, "fz")):
        diff = abs(float(got) - float(want))
        lim = 1e-5 * scale if name == "fx" else 1e-4 * scale
        assert diff < lim, (name, float(got), float(want), diff, lim)


# ---------------------------------------------------------------------------
# 4. multi-step trajectory vs production chain (mini validation b)
# ---------------------------------------------------------------------------

def test_multistep_force_trajectory_parity():
    n_steps = 30
    cfg = SuboffCmkKbcConfig(
        re=90.0, collision="CM", turbulence_model="smagorinsky",
        nx=NX, ny=NY, nz=NZ, n_steps=n_steps, u_in=U_IN, hull_length=9.0,
        device=DEV, use_triton_step=False)
    solid, f0 = _make_case()
    solid_i8 = solid.to(torch.int8)
    mass0 = float(f0.sum().item())

    h = f0.clone()                      # production chain
    b = f0.clone()                      # fixed triton path
    scale = 0.0
    for step in range(1, n_steps + 1):
        h = _collide_with_sgs(h, cfg, cfg.tau)
        h = stream3d(h)
        fx_a, _, _ = compute_obstacle_forces_3d(h, solid)
        h = far_field_bc_3d(h, U_IN, obstacle_mask=solid)
        if step % 10 == 0:
            h = correct_mass3d(h, mass0)

        tau_eff = _compute_sgs_tau_eff(
            bounce_back_cells_3d(stream3d(b), solid), cfg, cfg.tau)
        fx = torch.zeros((), device=DEV)
        fy = torch.zeros((), device=DEV)
        fz = torch.zeros((), device=DEV)
        b = triton_fused_obstacle_xfar_les(
            b, cfg.nu, solid_i8, cfg.C_s, 1.0, collision="CM",
            tau_eff=tau_eff, fx_buf=fx, fy_buf=fy, fz_buf=fz)
        apply_far_field_bc_6face(b, U_IN)
        if step % 10 == 0:
            b = apply_mass_correction(b, mass0)

        scale = max(scale, abs(float(fx_a)))  # series crosses zero: scale-rel
        rel = abs(float(fx) - float(fx_a)) / max(scale, 1e-30)
        assert rel < 1e-4, (step, float(fx), float(fx_a), rel)

    # final states also agree in the wall neighbourhood
    h_c = _collide_with_sgs(h, cfg, cfg.tau)
    nb = sphere_mask(NX, NY, NZ, 12.0, 8.0, 8.0, 6.5, DEV)
    d = float((b[:, nb] - h_c[:, nb]).abs().max())
    assert d < 1e-6, d


# ---------------------------------------------------------------------------
# 5. misc invariants
# ---------------------------------------------------------------------------

def test_obstacle_mask_and_shape_unchanged():
    solid, f0 = _make_case()
    solid_i8 = solid.to(torch.int8)
    before = solid_i8.clone()
    out = triton_fused_obstacle_xfar_les(
        f0.clone(), NU, solid_i8, 0.0, 1.0, collision="BGK")
    assert out.shape == (19, NZ, NY, NX)
    assert out.dtype == torch.float32
    assert torch.equal(solid_i8, before)
