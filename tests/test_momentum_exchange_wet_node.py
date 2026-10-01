"""W5-B discriminating tests — wet-node MEM (momentum_exchange.py patch).

Run against the library package:

    python -m pytest tests/test_momentum_exchange_wet_node.py -q

Covers the four properties that separate the repaired estimator from the
historically broken ones:

1. empty domain  -> every MEM variant returns exactly 0
2. engine-init background (free-stream feq outside, body at rest) ->
   wet-node returns exactly 0 while the standard Ladd pairing returns a
   large spurious force (the historical +264.6% @ Re=100 failure mode)
3. brute-force reference: wet-node and pair match a pure-python double loop
   over the same link set
4. corner links: the full crossing set strictly contains the near-wall
   (face-adjacent) set for a voxelised sphere
5. symmetry/physics: short sphere run keeps Cl << Cd and positive drag
6. engine wiring: mem_variant='wet_node'/'pair'/'all' record the new
   estimators; default 'standard' behaviour is unchanged
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d, sphere_mask
from tensorlbm.d3q19 import OPPOSITE, C, equilibrium3d
from tensorlbm.drag_pressure import get_near_wall_3d
from tensorlbm.momentum_exchange import (
    momentum_exchange_background_subtracted,
    momentum_exchange_pair,
    momentum_exchange_standard,
    momentum_exchange_wet_node,
)
from tensorlbm.solver3d import collide_mrt3d_low_memory, stream3d

DEV = "cuda:6" if torch.cuda.is_available() else "cpu"


def _uniform_freestream(shape, u0=0.05, device=DEV):
    nz, ny, nx = shape
    rho = torch.ones(shape, device=device)
    ux = torch.full(shape, u0, device=device)
    zeros = torch.zeros(shape, device=device)
    return equilibrium3d(rho, ux, zeros, zeros)


def _dpS(R_lb, u0=0.05):
    return 0.5 * u0**2 * math.pi * R_lb**2


# ---------------------------------------------------------------------------
# 1. empty domain
# ---------------------------------------------------------------------------
def test_empty_domain_all_variants_zero():
    shape = (14, 14, 14)
    f = _uniform_freestream(shape)
    solid = torch.zeros(shape, dtype=torch.bool, device=f.device)
    near = torch.zeros(shape, dtype=torch.bool, device=f.device)
    assert momentum_exchange_standard(f, solid, near) == (0.0, 0.0, 0.0)
    assert momentum_exchange_wet_node(f, solid, near) == (0.0, 0.0, 0.0)
    assert momentum_exchange_pair(f, solid, near) == (0.0, 0.0, 0.0)
    bg = momentum_exchange_background_subtracted(f, solid, near, rho0=1.0, u0=(0.05, 0.0, 0.0))
    assert bg == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# 2. uniform free-stream background closure (the historical failure mode)
# ---------------------------------------------------------------------------
def test_uniform_background_closure_wet_node_zero():
    """Engine-initialization convention: free-stream feq in the fluid, body
    cells at rest equilibrium (u=0 inside the solid)."""
    nz = ny = nx = 40
    R = 8.0
    device = DEV
    rho = torch.ones((nz, ny, nx), device=device)
    ux = torch.full((nz, ny, nx), 0.05, device=device)
    zeros = torch.zeros_like(rho)
    solid = sphere_mask(nx, ny, nz, nx / 2, ny / 2, nz / 2, R, device)
    ux[solid] = 0.0  # body at rest — engine init convention
    f = equilibrium3d(rho, ux, zeros, zeros)
    near = get_near_wall_3d(solid)
    dpS = _dpS(R)

    fw = momentum_exchange_wet_node(f, solid, near)
    # pairwise cancellation over the centrally symmetric link set: fp32
    # torch reductions leave ~1e-5 Cd-units of rounding noise on ~2.4e4
    # links of O(0.03) each — five orders below the standard estimator's
    # spurious force below.
    assert abs(fw[0]) / dpS < 1e-4
    assert abs(fw[1]) / dpS < 1e-4
    assert abs(fw[2]) / dpS < 1e-4

    # exact closure in float64 arithmetic (same call, double input)
    fw64 = momentum_exchange_wet_node(f.double(), solid, near)
    assert abs(fw64[0]) / dpS < 1e-10
    assert abs(fw64[1]) / dpS < 1e-10
    assert abs(fw64[2]) / dpS < 1e-10

    # the standard pairing does NOT close: spurious drag of order the
    # free-stream dynamic force on the projected area (>> any physical Cd)
    fs = momentum_exchange_standard(f, solid, near)
    assert abs(fs[0]) / dpS > 1.0

    # bg_sub is a no-op on this direction-symmetric crossing set (up to
    # fp32 reduction order between the two implementations)
    fb = momentum_exchange_background_subtracted(f, solid, near, rho0=1.0, u0=(0.05, 0.0, 0.0))
    assert abs(fb[0] - fs[0]) / dpS < 1e-4
    assert abs(fb[1] - fs[1]) / dpS < 1e-4

    # pair = wet + link-storage transient (non-zero here, but same order
    # as the standard estimator's background, i.e. a transient not a bias)
    fp = momentum_exchange_pair(f, solid, near)
    assert abs(fp[0]) / dpS > 1.0


# ---------------------------------------------------------------------------
# 3. brute-force reference (pure-python double loop)
# ---------------------------------------------------------------------------
def _brute_wet_and_pair(f, solid, near):
    nz, ny, nx = solid.shape
    c = C.to(f.device).tolist()
    opp = OPPOSITE.to(f.device).tolist()
    wet = [0.0, 0.0, 0.0]
    pair = [0.0, 0.0, 0.0]
    n_links = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if solid[k, j, i]:
                    continue
                for q in range(1, 19):
                    cq = c[q]
                    i2, j2, k2 = i + cq[0], j + cq[1], k + cq[2]
                    if not (0 <= i2 < nx and 0 <= j2 < ny and 0 <= k2 < nz):
                        continue
                    if not solid[k2, j2, i2]:
                        continue
                    n_links += 1
                    f_s = float(f[q, k2, j2, i2].item())
                    f_opp_f = float(f[opp[q], k, j, i].item())
                    for a in range(3):
                        wet[a] += 2.0 * cq[a] * f_s
                        pair[a] += cq[a] * (f_s + f_opp_f)
    return wet, pair, n_links


def test_wet_node_matches_brute_force_reference():
    nz = ny = nx = 20
    R = 5.0
    torch.manual_seed(0)
    f = torch.rand((19, nz, ny, nx), device=DEV) * 0.2 + 0.02
    solid = sphere_mask(nx, ny, nz, nx / 2, ny / 2, nz / 2, R, f.device)
    near = get_near_wall_3d(solid)

    fw = momentum_exchange_wet_node(f, solid, near)
    fp = momentum_exchange_pair(f, solid, near)
    bw, bp, n_links = _brute_wet_and_pair(f.cpu(), solid.cpu(), near.cpu())
    assert n_links > 200  # non-trivial link set

    scale = max(abs(bw[0]), 1e-30)
    for a in range(3):
        assert abs(fw[a] - bw[a]) / scale < 1e-5  # fp32 sum-order epsilon
        assert abs(fp[a] - bp[a]) / scale < 1e-5


# ---------------------------------------------------------------------------
# 4. corner links exist (full set strictly larger than near set)
# ---------------------------------------------------------------------------
def test_full_crossing_set_contains_corner_links():
    nz = ny = nx = 40
    R = 8.0
    solid = sphere_mask(nx, ny, nz, nx / 2, ny / 2, nz / 2, R, DEV)
    near = get_near_wall_3d(solid)
    fluid = ~solid
    c = C.to(DEV)
    n_near = n_full = 0
    for q in range(1, 19):
        di, dj, dk = int(c[q][0]), int(c[q][1]), int(c[q][2])
        shifted = torch.roll(solid, (-dk, -dj, -di), dims=(0, 1, 2))
        full = fluid & shifted
        n_full += int(full.sum().item())
        n_near += int((full & near).sum().item())
    assert n_full > n_near
    assert (n_full - n_near) / n_full > 0.10  # ~14% corner links at this D


# ---------------------------------------------------------------------------
# 5. short sphere run: sign, magnitude, symmetry
# ---------------------------------------------------------------------------
def test_short_sphere_run_sign_and_symmetry():
    nz = ny = 30
    nx = 36
    D = 10
    R = D / 2
    u0 = 0.05
    tau = 0.5 + 3.0 * (u0 * D / 100.0)
    shape = (nz, ny, nx)
    f = _uniform_freestream(shape, u0)
    solid = sphere_mask(nx, ny, nz, nx / 2, ny / 2, nz / 2, R, f.device)
    near = get_near_wall_3d(solid)
    solid = solid.to(DEV)
    near = near.to(DEV)
    f = f.to(DEV)
    dpS = _dpS(R, u0)

    bc_config = {"far_field_faces": ["y-", "y+", "z-", "z+"], "periodic_faces": []}
    for _ in range(300):
        f_pre_solid = f[:, solid].clone()
        f = collide_mrt3d_low_memory(f, tau=tau)
        f[:, solid] = f_pre_solid
        f = bounce_back_cells_3d(f, solid)
        f = stream3d(f)
        f = far_field_bc_3d(f, u0, bc_config=bc_config)

    fx, fy, fz = momentum_exchange_wet_node(f, solid, near)
    cd, cl = fx / dpS, fy / dpS
    assert 0.5 < cd < 6.0  # developing drag, right order of magnitude
    assert abs(cl) < 0.05 * abs(cd)  # symmetric body: no lift
    # steady-state identity: pair == wet to a few percent
    fxp, _, _ = momentum_exchange_pair(f, solid, near)
    assert abs(fxp - fx) / abs(fx) < 0.05


# ---------------------------------------------------------------------------
# 6. engine wiring
# ---------------------------------------------------------------------------
def _tiny_engine(mem_variant, steps=4):
    from tensorlbm.general_sim import (
        CollisionModel,
        ForceMethod,
        GeneralSimConfig,
        GeneralSimEngine,
        GeometryConfig,
        GeometrySource,
        LatticeModel,
        OutputConfig,
        OutputFormat,
        PhysicsConfig,
        SolverConfig,
        WallTreatment,
    )

    cfg = GeneralSimConfig(
        name=f"w5b_test_{mem_variant}",
        geometry=GeometryConfig(
            source=GeometrySource.PARAMETRIC_SPHERE,
            sphere_radius=0.5,
            sphere_center=(0.0, 0.0, 0.0),
        ),
        physics=PhysicsConfig(
            density=1000.0,
            viscosity=1.0e-6,
            inlet_velocity=1.0e-4,
            reference_length=1.0,
        ),
        solver=SolverConfig(
            lattice=LatticeModel.D3Q19,
            collision=CollisionModel.MRT,
            resolution=8,
            domain_padding=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            max_steps=steps,
            snapshot_interval=10**9,
            force_sample_interval=2,
            device=DEV,
            wall_treatment=WallTreatment.BOUNCE_BACK,
            mass_correction=False,
            force_method=ForceMethod.MOMENTUM_EXCHANGE,
            mem_variant=mem_variant,
        ),
        output=OutputConfig(
            directory="/nfs/wangxi/tmp/w5b_test_out"
            if DEV.startswith("cuda")
            else "/tmp/w5b_test_out",
            formats=[OutputFormat.NPY],
            save_macroscopic=False,
            save_forces=True,
        ),
    )
    eng = GeneralSimEngine(cfg)
    eng.setup()
    eng.run(steps=steps)
    return eng


@pytest.mark.parametrize("variant", ["wet_node", "pair"])
def test_engine_wiring_new_variants(variant):
    eng = _tiny_engine(variant)
    assert eng.forces_log
    e = eng.forces_log[-1]
    assert "cd_mem" in e
    assert e["cd_total"] == e["cd_mem"]
    assert e["fx"] == e["fx_mem"]
    assert e["cd_mem"] != 0.0


def test_engine_wiring_all_records_every_variant():
    eng = _tiny_engine("all")
    e = eng.forces_log[-1]
    for key in (
        "cd_mem_standard",
        "cd_mem_galilean",
        "cd_mem_bgsub",
        "cd_mem_wet_node",
        "cd_mem_pair",
    ):
        assert key in e
    assert e["cd_mem"] == e["cd_mem_standard"]


def test_engine_default_standard_unchanged():
    eng = _tiny_engine("standard")
    e = eng.forces_log[-1]
    direct = momentum_exchange_standard(eng.f, eng.solid, eng.near)
    assert e["fx_mem"] == direct[0]
    assert e["cd_mem"] == direct[0] / (0.5 * eng.uc.u_lb**2 * math.pi * (4.0) ** 2)
