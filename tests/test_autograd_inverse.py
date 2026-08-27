"""Tests for the soft-solid inverse-design path (``tensorlbm.soft_geometry``).

``tests/test_autograd_path.py`` pinned the composition contract of the
differentiable step chain; this file pins its **differentiable-geometry**
extension — the opt-in soft solid that turns SDF parameters into loss
variables:

1. SDF value contracts: the sphere SDF reproduces
   ``boundaries3d.sphere_mask`` as its hard limit, the box SDF matches
   hand-computed distances, the ellipsoid zero set hits the semi-axis
   endpoints, weights live in [0, 1] and saturate to *exact* 0/1 beyond the
   documented band;
2. the soft step equals the manual soft composition (soft NoDynamics ->
   stream -> boundary conditions -> soft bounce-back), and the default
   ``soft=None`` chain is bit-for-bit unchanged — both in-test and against
   the frozen baseline artefacts of the pre-feature commit
   (``scripts/gen_autograd_inverse_baseline.py``, env-gated);
3. the hard limit: at a saturating temperature the whole soft chain (states,
   probes, drag) degenerates ``torch.equal``-exactly to the hard-mask chain,
   and the soft force satisfies the per-step action-reaction balance
   (F = -d(momentum of the field)/dt, periodic box) to machine precision —
   the numerical closure of the momentum-exchange derivation;
4. geometry derivatives: d(drag)/d(radius), d(drag)/d(ellipsoid semi-axis),
   d(drag)/d(box half-extent), d(drag)/d(centre) and d(drag)/d(epsilon)
   against central finite differences of the same discrete objective, fp64;
5. soft -> hard C_D convergence: the drag coefficient of a soft sphere
   converges monotonically to the same-parameter hard-mask sphere as
   epsilon shrinks (the convergence table);
6. solver-in-the-loop inverse design: gradient descent on the radius from a
   wrong guess hits a reference C_D to <1%, monotonically.
"""

from __future__ import annotations

import importlib.util
import math
import os
import pathlib

import pytest
import torch

from tensorlbm.autograd_path import (
    InletSpec,
    OutletSpec,
    WallSpec,
    differentiable_step,
    obstacle_force,
    rollout,
)
from tensorlbm.boundaries3d import sphere_mask
from tensorlbm.d3q19 import OPPOSITE, C, equilibrium3d, macroscopic3d
from tensorlbm.soft_geometry import SoftGeometry
from tensorlbm.solver3d import collide_bgk3d, stream3d

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Small bounded case (the section-7 grid of tests/test_autograd_path.py).
BNZ, BNY, BNX = 8, 10, 20
BCX, BCY, BCZ = 6.0, 5.0, 4.0
BR = 2.3  # non-integer: no node sits on the surface (no hard-mask ties)
U_IN = 0.08
EPS_SAT = 1e-3  # saturating temperature: |phi|/eps > 30 everywhere
EPS_OPT = 0.25  # the optimisation temperature (gradient-favoured)

# Hand-derived specular-reflection tables (independent replay of the walls):
# FLIP[axis][q] is the index of the lattice velocity with that one transverse
# component negated; the unknown sets are the directions that wrap around the
# domain through the respective face during streaming.
FLIP_Y = (0, 1, 2, 4, 3, 5, 6, 9, 10, 7, 8, 11, 12, 13, 14, 18, 17, 16, 15)
FLIP_Z = (0, 1, 2, 3, 4, 6, 5, 7, 8, 9, 10, 13, 14, 11, 12, 17, 18, 15, 16)
UNK_Y0 = (3, 7, 10, 15, 17)  # c_y = +1: wrapped at y = 0
UNK_Y1 = (4, 8, 9, 16, 18)  # c_y = -1: wrapped at y = ny - 1
UNK_Z0 = (5, 11, 14, 15, 18)  # c_z = +1: wrapped at z = 0
UNK_Z1 = (6, 12, 13, 16, 17)  # c_z = -1: wrapped at z = nz - 1

BASELINE_ENV = "TENSORLBM_INVERSE_BASELINE_DIR"


def sphere_geometry(radius: float | torch.Tensor, epsilon: float | torch.Tensor) -> SoftGeometry:
    """The test sphere on the bounded grid (centred, non-integer radius)."""
    return SoftGeometry(kind="sphere", center=(BCX, BCY, BCZ), size=(radius,), epsilon=epsilon)


def moved_center_sphere(cx: float | torch.Tensor) -> SoftGeometry:
    """Test sphere with a movable centre x (kept off the wake symmetry)."""
    return SoftGeometry(kind="sphere", center=(cx, BCY, BCZ), size=(BR,), epsilon=EPS_OPT)


def bounded_f0(dtype: torch.dtype, device: torch.device, seed: int = 23) -> torch.Tensor:
    """Uniform equilibrium inflow plus deterministic off-equilibrium noise."""
    ones = torch.ones(BNZ, BNY, BNX, dtype=dtype, device=device)
    zeros = torch.zeros_like(ones)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.rand((19, BNZ, BNY, BNX), generator=gen, dtype=torch.float64).to(dtype) - 0.5
    return equilibrium3d(ones, U_IN * ones, zeros, zeros).to(device) + 0.02 * noise.to(device)


def bounded_specs() -> tuple[InletSpec, OutletSpec]:
    return InletSpec(ux=U_IN, method="zouhe"), OutletSpec(method="convective")


def soft_drag(
    f0: torch.Tensor,
    steps: int,
    tau: float,
    geometry: SoftGeometry,
    inlet: InletSpec | None,
    outlet: OutletSpec | None,
) -> torch.Tensor:
    """Accumulated soft momentum-exchange drag over a rollout (graph kept)."""
    nz, ny, nx = f0.shape[1], f0.shape[2], f0.shape[3]
    _f, probes = rollout(
        f0,
        steps,
        tau,
        None,
        soft=geometry,
        inlet=inlet,
        outlet=outlet,
        return_probes=True,
    )
    solid = geometry.solid_weight(nz, ny, nx, dtype=f0.dtype, device=f0.device)
    return sum(obstacle_force(p, solid)[0] for p in probes)


# ---------------------------------------------------------------------------
# 1. SDF / weight value contracts
# ---------------------------------------------------------------------------


def test_sdf_sphere_hard_mask_matches_sphere_mask() -> None:
    """The epsilon -> 0 limit of the sphere SDF is exactly sphere_mask."""
    geom = sphere_geometry(BR, EPS_OPT)
    soft = geom.hard_mask(BNZ, BNY, BNX, device=DEVICE)
    hard = sphere_mask(BNX, BNY, BNZ, BCX, BCY, BCZ, BR, device=DEVICE)
    assert torch.equal(soft, hard)
    assert soft.shape == (BNZ, BNY, BNX)


def test_sdf_values_sphere_box_ellipsoid() -> None:
    """Exact SDF values on sample nodes: sphere distance, box faces/edges,
    ellipsoid semi-axis endpoints on the zero set."""
    dtype = torch.float64
    device = torch.device("cpu")  # value contract: CPU-canonical, device-independent
    nz, ny, nx = 9, 11, 13
    cx, cy, cz = 6.0, 5.0, 4.0

    sph = SoftGeometry(kind="sphere", center=(cx, cy, cz), size=(2.0,), epsilon=0.5)
    phi = sph.sdf(nz, ny, nx, dtype=dtype, device=device)
    # node (z, y, x) = (4, 5, 8): distance 2 along +x -> phi = 0
    assert phi[4, 5, 8] == pytest.approx(0.0, abs=1e-14)
    assert phi[4, 5, 7] == pytest.approx(-1.0, abs=1e-14)  # inside, 1 from surface
    assert phi[4, 5, 9] == pytest.approx(1.0, abs=1e-14)  # outside
    # centre node: distance 0 -> phi = -radius (sqrt clamped, finite)
    assert phi[4, 5, 6] == pytest.approx(-2.0, abs=1e-12)

    box = SoftGeometry(kind="box", center=(cx, cy, cz), size=(2.0, 1.5, 1.0), epsilon=0.5)
    phi = box.sdf(nz, ny, nx, dtype=dtype, device=device)
    assert phi[4, 5, 8] == pytest.approx(0.0, abs=1e-14)  # on the +x face
    assert phi[4, 5, 7] == pytest.approx(-1.0, abs=1e-14)  # 1 inside from the face
    assert phi[4, 5, 9] == pytest.approx(1.0, abs=1e-14)  # 1 outside the face
    # face-corner region: outside along x by 1 and y by 0.5 -> sqrt(1 + 0.25)
    assert phi[5, 7, 9] == pytest.approx(math.sqrt(1.0**2 + 0.5**2), abs=1e-14)
    # deep inside: distance to the nearest face (z half-extent 1)
    assert phi[4, 5, 6] == pytest.approx(-1.0, abs=1e-14)

    ell = SoftGeometry(kind="ellipsoid", center=(cx, cy, cz), size=(3.0, 2.0, 1.0), epsilon=0.5)
    phi = ell.sdf(nz, ny, nx, dtype=dtype, device=device)
    # semi-axis endpoints sit exactly on the surface (phi = 0)
    assert phi[4, 5, 9] == pytest.approx(0.0, abs=1e-14)  # +x endpoint (cx + 3)
    assert phi[4, 7, 6] == pytest.approx(0.0, abs=1e-14)  # +y endpoint (cy + 2)
    assert phi[5, 5, 6] == pytest.approx(0.0, abs=1e-14)  # +z endpoint (cz + 1)
    # inside along +x at normalised radius t = 1/3: phi = (t - 1) * min(a, b, c)
    assert phi[4, 5, 7] == pytest.approx((1.0 / 3.0 - 1.0) * 1.0, abs=1e-14)
    assert bool((phi[4, 5, 6] < 0).item())  # centre strictly inside


def test_fluid_weight_bounds_and_saturation() -> None:
    """w in [0, 1]; exactly 0/1 beyond |phi/eps| = 30; band strictly inside."""
    dtype = torch.float64
    # saturating temperature: the closest node to the surface sits ~0.064
    # lattice units from it, so |phi|/eps > 30 everywhere and the weight is
    # exactly black/white over the whole grid
    geom = sphere_geometry(BR, EPS_SAT)
    w = geom.fluid_weight(BNZ, BNY, BNX, dtype=dtype, device=DEVICE)
    assert bool(((w >= 0.0) & (w <= 1.0)).all())
    assert bool(((w == 0.0) | (w == 1.0)).all())
    assert torch.equal(w.bool(), ~geom.hard_mask(BNZ, BNY, BNX, device=DEVICE))

    # finite temperature: transition band values strictly between 0 and 1
    soft = sphere_geometry(BR, 0.5).fluid_weight(BNZ, BNY, BNX, dtype=dtype, device=DEVICE)
    mask = sphere_mask(BNX, BNY, BNZ, BCX, BCY, BCZ, BR, device=DEVICE)
    assert bool((soft[~mask] > 0.5).all())  # fluid side above half
    assert bool((soft[mask] < 0.5).all())  # solid side below half
    assert bool(((soft > 0.0) & (soft < 1.0)).any())  # the transition band exists


def test_soft_geometry_spec_validation() -> None:
    """Malformed specs fail loudly."""
    with pytest.raises(ValueError, match="kind must be one of"):
        SoftGeometry(kind="capsule", size=(1.0,))
    with pytest.raises(ValueError, match="center must be"):
        SoftGeometry(kind="sphere", center=(1.0, 2.0))
    with pytest.raises(ValueError, match="sphere needs 1 size"):
        SoftGeometry(kind="sphere", size=(1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="ellipsoid needs 3 size"):
        SoftGeometry(kind="ellipsoid", size=(1.0,))
    with pytest.raises(ValueError, match="box needs 3 size"):
        SoftGeometry(kind="box", size=(1.0, 2.0))
    with pytest.raises(ValueError, match="epsilon must be > 0"):
        SoftGeometry(kind="sphere", size=(1.0,), epsilon=0.0)
    with pytest.raises(ValueError, match="epsilon must be > 0"):
        SoftGeometry(kind="sphere", size=(1.0,), epsilon=torch.tensor(-0.5))
    # tensor parameters construct and evaluate (0-dim, graph-connected)
    geom = SoftGeometry(
        kind="box",
        center=(torch.tensor(6.0), BCY, BCZ),
        size=(2.0, 1.5, 1.0),
        epsilon=torch.tensor(0.5),
    )
    assert geom.sdf(BNZ, BNY, BNX, dtype=torch.float64).shape == (BNZ, BNY, BNX)


def test_soft_geometry_parameter_gradients_finite() -> None:
    """d(sum w)/d(param) exists, is finite and non-zero for every parameter
    of every shape (covers the sqrt-floor NaN fix at the exact-centre node)."""
    dtype = torch.float64
    for kind, size in (
        ("sphere", (BR,)),
        ("ellipsoid", (2.3, 1.8, 1.5)),
        ("box", (2.2, 1.7, 1.4)),
    ):
        center = [torch.tensor(v, dtype=dtype, requires_grad=True) for v in (6.3, 5.2, 3.8)]
        first_size = torch.tensor(size[0], dtype=dtype, requires_grad=True)
        eps = torch.tensor(0.5, dtype=dtype, requires_grad=True)
        geom = SoftGeometry(
            kind=kind,
            center=tuple(center),
            size=(first_size, *size[1:]),
            epsilon=eps,
        )
        w = geom.fluid_weight(BNZ, BNY, BNX, dtype=dtype, device=DEVICE)
        cx_grad, _cy, _cz, size_grad, eps_grad = torch.autograd.grad(
            w.sum(), [center[0], center[1], center[2], first_size, eps]
        )
        for value in (cx_grad, size_grad, eps_grad):
            assert torch.isfinite(value).all()
            assert value.abs() > 0.0


# ---------------------------------------------------------------------------
# 2. Soft step value contract + default-path bitwise identity
# ---------------------------------------------------------------------------


def test_soft_step_equals_manual_composition() -> None:
    """Soft step == soft collision mix -> stream -> inlet/outlet -> soft BB
    (independent replay of every phase, Zou/He inlet + convective outlet)."""
    dtype = torch.float64
    tau = 0.7
    f = bounded_f0(dtype, DEVICE)
    geom = sphere_geometry(BR, 0.5)
    inlet, outlet = bounded_specs()
    out, probe = differentiable_step(
        f, tau, soft=geom, return_probe=True, inlet=inlet, outlet=outlet
    )

    w = geom.fluid_weight(BNZ, BNY, BNX, dtype=dtype, device=DEVICE).unsqueeze(0)
    mixed = w * collide_bgk3d(f, tau) + (1.0 - w) * f  # soft NoDynamics
    f_str = stream3d(mixed)

    # Zou/He inlet on x = 0 (replayed): plane density from the known moments
    plane = f_str[..., :1]
    ux = torch.tensor(U_IN, dtype=dtype, device=DEVICE)
    rest = plane[[0, 3, 4, 5, 6, 15, 16, 17, 18]].sum(dim=0)  # c_x = 0
    outgoing = plane[[2, 8, 10, 12, 14]].sum(dim=0)  # c_x = -1
    rho_in = (rest + 2.0 * outgoing) / (1.0 - ux)
    ones = torch.ones_like(rho_in)
    feq = equilibrium3d(rho_in, ux * ones, 0.0 * ones, 0.0 * ones, DEVICE)
    opp = OPPOSITE.to(DEVICE)
    cand = feq + (plane[opp] - feq[opp])
    sel_in = torch.zeros((19, 1, 1, 1), dtype=torch.bool, device=DEVICE)
    sel_in[[1, 7, 9, 11, 13], 0, 0, 0] = True  # c_x = +1: the unknowns
    plane_new = torch.where(sel_in, cand, plane)
    f_bc = torch.cat([plane_new, f_str[..., 1:]], dim=-1)

    # convective outlet on x = nx-1 (replayed): upwind recursion seeded from
    # the initial condition's own outlet face
    neighbour = f_bc[..., -2:-1]
    f_prev = f[..., -1:]
    cand_o = f_prev + ux * (neighbour - f_prev)
    sel_out = torch.zeros((19, 1, 1, 1), dtype=torch.bool, device=DEVICE)
    sel_out[[2, 8, 10, 12, 14], 0, 0, 0] = True  # c_x = -1: the unknowns
    plane_o = torch.where(sel_out, cand_o, f_bc[..., -1:])
    f_bc = torch.cat([f_bc[..., :-1], plane_o], dim=-1)

    assert torch.equal(probe, f_bc)
    expected = w * f_bc + (1.0 - w) * f_bc[opp]  # soft full-way bounce-back
    assert torch.equal(out, expected)


def test_soft_step_walls_equals_manual_composition() -> None:
    """Soft step with lateral walls == manual replay (free-slip faces close
    y = 0 and y = ny-1, z = 0 stays periodic, z = nz-1 is a free-stream face
    overriding the whole plane)."""
    dtype = torch.float64
    tau = 0.7
    f = bounded_f0(dtype, DEVICE, seed=31)
    geom = sphere_geometry(BR, 0.5)
    walls = WallSpec(
        method="free-slip",
        overrides={
            "-z": WallSpec(method="periodic"),
            "+z": WallSpec(method="freestream", rho0=1.03, ux=0.06, uy=-0.02),
        },
    )
    out = differentiable_step(f, tau, soft=geom, walls=walls)

    w = geom.fluid_weight(BNZ, BNY, BNX, dtype=dtype, device=DEVICE).unsqueeze(0)
    f_str = stream3d(w * collide_bgk3d(f, tau) + (1.0 - w) * f)

    def slip(plane: torch.Tensor, flip, unknown) -> torch.Tensor:
        new = plane.clone()
        for q in unknown:
            new[q] = plane[flip[q]]
        return new

    # faces close in the order y = 0, y = ny-1, z = 0, z = nz-1
    f_m = torch.cat([slip(f_str[:, :, :1], FLIP_Y, UNK_Y0), f_str[:, :, 1:]], dim=2)
    f_m = torch.cat([f_m[:, :, :-1], slip(f_m[:, :, -1:], FLIP_Y, UNK_Y1)], dim=2)
    # z = 0 periodic: no-op
    feq = equilibrium3d(
        torch.tensor(1.03, dtype=dtype, device=DEVICE),
        torch.tensor(0.06, dtype=dtype, device=DEVICE),
        torch.tensor(-0.02, dtype=dtype, device=DEVICE),
        torch.tensor(0.0, dtype=dtype, device=DEVICE),
        DEVICE,
    )
    f_m = torch.cat([f_m[:, :-1, :, :], feq.expand(19, 1, BNY, BNX)], dim=1)

    opp = OPPOSITE.to(DEVICE)
    expected = w * f_m + (1.0 - w) * f_m[opp]
    assert torch.equal(out, expected)


def test_soft_none_default_path_bitwise_unchanged() -> None:
    """Explicit soft=None: bit-for-bit the original hard-mask chain."""
    dtype = torch.float64
    steps, tau = 5, 0.8
    f0 = bounded_f0(dtype, DEVICE)
    mask = sphere_mask(BNX, BNY, BNZ, BCX, BCY, BCZ, BR, device=DEVICE)

    f = f0.clone()
    opp = OPPOSITE.to(DEVICE)
    for _ in range(steps):
        f_col = torch.where(mask.unsqueeze(0), f, collide_bgk3d(f, tau))
        f_str = stream3d(f_col)
        f = torch.where(mask.unsqueeze(0), f_str[opp], f_str)
    base = rollout(f0, steps, tau, mask)

    assert torch.equal(base, f)
    assert torch.equal(rollout(f0, steps, tau, mask, soft=None), f)
    with pytest.raises(ValueError, match="not both"):
        rollout(f0, steps, tau, mask, soft=sphere_geometry(BR, EPS_OPT))


@pytest.mark.skipif(
    os.environ.get(BASELINE_ENV) is None,
    reason=f"{BASELINE_ENV} not set (baseline.pt from scripts/gen_autograd_inverse_baseline.py)",
)
def test_default_path_bitwise_vs_baseline_artifacts() -> None:
    """Default-path rollouts vs the frozen baseline.pt of the pre-feature
    commit 02275f55 (configs regenerated through the generator script, so
    the two sides cannot drift apart)."""
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "gen_autograd_inverse_baseline.py"
    spec = importlib.util.spec_from_file_location("gen_inv_baseline", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    reference = torch.load(
        pathlib.Path(os.environ[BASELINE_ENV]) / "baseline.pt", weights_only=True
    )
    fresh = module.run_rollouts()
    assert set(fresh) == set(reference["configs"])
    for name, entry in fresh.items():
        ref = reference["configs"][name]
        assert torch.equal(entry["f"].cpu(), ref["f"].cpu()), name
        assert torch.equal(entry["drag"].cpu(), ref["drag"].cpu()), name


# ---------------------------------------------------------------------------
# 3. Hard limit: saturated weights degenerate to the hard-mask chain
# ---------------------------------------------------------------------------


def test_soft_chain_hard_limit_bitwise() -> None:
    """At a saturating temperature (every node > 30*eps from the surface)
    the whole soft chain — final state, probes and drag — is torch.equal to
    the hard-mask chain of the same parameters."""
    dtype = torch.float64
    steps, tau = 12, 0.55
    f0 = bounded_f0(dtype, DEVICE)
    mask = sphere_mask(BNX, BNY, BNZ, BCX, BCY, BCZ, BR, device=DEVICE)
    geom = sphere_geometry(BR, EPS_SAT)
    inlet, outlet = bounded_specs()

    f_hard, probes_hard = rollout(
        f0, steps, tau, mask, inlet=inlet, outlet=outlet, return_probes=True
    )
    f_soft, probes_soft = rollout(
        f0, steps, tau, None, soft=geom, inlet=inlet, outlet=outlet, return_probes=True
    )
    assert torch.equal(f_soft, f_hard)
    assert len(probes_soft) == len(probes_hard)
    assert all(torch.equal(a, b) for a, b in zip(probes_soft, probes_hard))

    solid = geom.solid_weight(BNZ, BNY, BNX, dtype=dtype, device=DEVICE)
    drag_hard = sum(obstacle_force(p, mask)[0] for p in probes_hard)
    drag_soft = sum(obstacle_force(p, solid)[0] for p in probes_soft)
    assert torch.equal(drag_soft, drag_hard)
    # the boolean mask itself is the saturated solid weight
    assert torch.equal(solid.bool(), mask)


def test_soft_force_action_reaction_balance() -> None:
    """Per-step momentum bookkeeping of the soft chain (periodic box):
    F = -d(momentum of the field)/dt exactly — collision and streaming
    conserve the global momentum, so the soft bounce-back must hand the
    soft force to the body.  Machine-precision closure of the derivation."""
    dtype = torch.float64
    f0 = bounded_f0(dtype, DEVICE, seed=41)
    geom = sphere_geometry(BR, 0.4)

    f1, probe = differentiable_step(f0, 0.7, soft=geom, return_probe=True)
    solid = geom.solid_weight(BNZ, BNY, BNX, dtype=dtype, device=DEVICE)
    force = obstacle_force(probe, solid)

    c = C.to(device=DEVICE, dtype=dtype)
    mom0 = torch.matmul(f0.sum(dim=(1, 2, 3)), c)
    mom1 = torch.matmul(f1.sum(dim=(1, 2, 3)), c)
    delta = mom1 - mom0
    for axis in range(3):
        assert float(force[axis]) == pytest.approx(-float(delta[axis]), rel=1e-12, abs=1e-12), axis


# ---------------------------------------------------------------------------
# 4. Geometry derivatives vs central finite differences (fp64)
# ---------------------------------------------------------------------------


def _fd_drag(value: float, eps_fd: float, geometry_of) -> float:
    """Central difference of the accumulated soft drag w.r.t. one parameter."""
    dtype = torch.float64
    f0 = bounded_f0(dtype, DEVICE)
    inlet, outlet = bounded_specs()
    plus = float(soft_drag(f0, 5, 0.55, geometry_of(value + eps_fd), inlet, outlet))
    minus = float(soft_drag(f0, 5, 0.55, geometry_of(value - eps_fd), inlet, outlet))
    return (plus - minus) / (2.0 * eps_fd)


@pytest.mark.parametrize(
    ("kind", "size", "rtol"),
    [
        ("sphere", (BR,), 1e-6),
        ("ellipsoid", (2.3, 1.8, 1.5), 1e-6),
        ("box", (2.2, 1.7, 1.4), 1e-6),
    ],
    ids=["sphere-radius", "ellipsoid-semi-axis", "box-half-extent"],
)
def test_drag_size_gradient_matches_finite_difference(kind, size, rtol) -> None:
    """d(accumulated drag)/d(size[0]) through the solver == central FD, fp64."""
    dtype = torch.float64

    def geometry_of(value):
        return SoftGeometry(
            kind=kind, center=(BCX, BCY, BCZ), size=(value, *size[1:]), epsilon=EPS_OPT
        )

    leaf = torch.tensor(size[0], dtype=dtype, requires_grad=True)
    f0 = bounded_f0(dtype, DEVICE)
    inlet, outlet = bounded_specs()

    drag = soft_drag(f0, 5, 0.55, geometry_of(leaf), inlet, outlet)
    (grad,) = torch.autograd.grad(drag, leaf)
    fd = _fd_drag(size[0], 1e-5, geometry_of)

    assert torch.isfinite(grad)
    assert float(grad) != 0.0
    denom = max(abs(float(grad)), abs(fd), 1e-30)
    assert abs(float(grad) - fd) / denom < rtol, (float(grad), fd)


def test_drag_center_gradient_matches_finite_difference() -> None:
    """d(drag)/d(cx): a smaller signal, so the FD step is widened to keep the
    quotient above the fp64 round-off of the drag difference."""
    dtype = torch.float64
    cx0, eps_fd = 6.3, 1e-4
    leaf = torch.tensor(cx0, dtype=dtype, requires_grad=True)
    f0 = bounded_f0(dtype, DEVICE)
    inlet, outlet = bounded_specs()

    drag = soft_drag(f0, 5, 0.55, moved_center_sphere(leaf), inlet, outlet)
    (grad,) = torch.autograd.grad(drag, leaf)
    fd = _fd_drag(cx0, eps_fd, moved_center_sphere)

    assert torch.isfinite(grad)
    denom = max(abs(float(grad)), abs(fd), 1e-30)
    assert abs(float(grad) - fd) / denom < 1e-5, (float(grad), fd)


def test_epsilon_gradient_matches_finite_difference() -> None:
    """The temperature itself is a learnable parameter of the chain."""
    dtype = torch.float64
    eps0 = 0.5
    leaf = torch.tensor(eps0, dtype=dtype, requires_grad=True)
    f0 = bounded_f0(dtype, DEVICE)
    inlet, outlet = bounded_specs()

    drag = soft_drag(f0, 5, 0.55, sphere_geometry(BR, leaf), inlet, outlet)
    (grad,) = torch.autograd.grad(drag, leaf)

    def geometry_of(value):
        return sphere_geometry(BR, value)

    fd = _fd_drag(eps0, 1e-5, geometry_of)

    assert torch.isfinite(grad)
    denom = max(abs(float(grad)), abs(fd), 1e-30)
    assert abs(float(grad) - fd) / denom < 1e-6, (float(grad), fd)


def test_soft_rollout_checkpoint_gradients_equal() -> None:
    """checkpoint=True reproduces the plain soft-chain gradients exactly."""
    dtype = torch.float64
    steps = 8
    f0 = bounded_f0(dtype, DEVICE)
    inlet, outlet = bounded_specs()
    # fluid-only velocity sample: nodes away from the obstacle and boundaries
    ux_sample = (slice(1, -1), slice(1, -1), slice(1, BNX - 1))

    def loss_and_grads(use_checkpoint: bool):
        radius = torch.tensor(BR, dtype=dtype, device=DEVICE, requires_grad=True)
        tau = torch.tensor(0.7, dtype=dtype, device=DEVICE, requires_grad=True)
        geom = sphere_geometry(radius, EPS_OPT)
        f, probes = rollout(
            f0,
            steps,
            tau,
            None,
            soft=geom,
            checkpoint=use_checkpoint,
            inlet=inlet,
            outlet=outlet,
            return_probes=True,
        )
        _rho, ux, _uy, _uz = macroscopic3d(f)
        solid = geom.solid_weight(BNZ, BNY, BNX, dtype=dtype, device=DEVICE)
        loss = (ux[ux_sample] ** 2).mean() + sum(obstacle_force(p, solid)[0] for p in probes)
        grad_r, grad_tau = torch.autograd.grad(loss, [radius, tau])
        return loss.detach(), grad_r, grad_tau

    loss_plain, grad_r_p, grad_tau_p = loss_and_grads(False)
    loss_ckpt, grad_r_c, grad_tau_c = loss_and_grads(True)

    assert torch.allclose(loss_plain, loss_ckpt, rtol=1e-12)
    assert torch.allclose(grad_r_p, grad_r_c, rtol=1e-10, atol=1e-14)
    assert torch.allclose(grad_tau_p, grad_tau_c, rtol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_parity_soft_chain() -> None:
    """Same fp32 soft rollout and radius gradient on CPU and CUDA."""
    dtype, steps = torch.float32, 6
    inlet, outlet = bounded_specs()

    def run(f0: torch.Tensor, radius: torch.Tensor):
        geom = sphere_geometry(radius, EPS_OPT)
        _f, probes = rollout(
            f0,
            steps,
            0.55,
            None,
            soft=geom,
            inlet=inlet,
            outlet=outlet,
            return_probes=True,
        )
        solid = geom.solid_weight(BNZ, BNY, BNX, dtype=dtype, device=f0.device)
        loss = sum(obstacle_force(p, solid)[0] for p in probes)
        (grad,) = torch.autograd.grad(loss, radius)
        return float(loss.detach()), float(grad)

    f0_cpu = bounded_f0(dtype, torch.device("cpu"))
    radius_cpu = torch.tensor(BR, dtype=dtype, requires_grad=True)
    loss_c, grad_c = run(f0_cpu, radius_cpu)

    f0_cuda = f0_cpu.to("cuda")
    radius_cuda = torch.tensor(BR, dtype=dtype, device="cuda", requires_grad=True)
    loss_g, grad_g = run(f0_cuda, radius_cuda)

    assert loss_c == pytest.approx(loss_g, rel=1e-4)
    assert grad_c == pytest.approx(grad_g, rel=1e-3, abs=1e-8)


# ---------------------------------------------------------------------------
# 5. Soft -> hard C_D convergence (the table)
# ---------------------------------------------------------------------------

# Bigger campaign grid: the sphere must not choke the bounded box.
CNZ, CNY, CNX = 12, 16, 26
CCX, CCY, CCZ, CR = 7.0, 8.0, 6.0, 2.3
CD_STEPS, CD_WINDOW = 400, (300, 400)
CD_EPS_SWEEP = (0.5, 0.25, 0.125, 0.0625, 0.02)


def _cd_campaign(obstacle) -> float:
    """Window-mean drag coefficient of one bounded campaign (fp64, CPU)."""
    dtype = torch.float64
    device = torch.device("cpu")
    ones = torch.ones(CNZ, CNY, CNX, dtype=dtype, device=device)
    zeros = torch.zeros_like(ones)
    f = equilibrium3d(ones, U_IN * ones, zeros, zeros)
    inlet, outlet = InletSpec(ux=U_IN), OutletSpec()
    soft = isinstance(obstacle, SoftGeometry)
    weight = obstacle.solid_weight(CNZ, CNY, CNX, dtype=dtype, device=device) if soft else obstacle
    drags = []
    for _ in range(CD_STEPS):
        if soft:
            f, probe = differentiable_step(
                f, 0.55, soft=obstacle, return_probe=True, inlet=inlet, outlet=outlet
            )
        else:
            f, probe = differentiable_step(
                f, 0.55, obstacle, return_probe=True, inlet=inlet, outlet=outlet
            )
        drags.append(obstacle_force(probe, weight)[0].detach())
    window = torch.stack(drags[slice(*CD_WINDOW)])
    area = math.pi * CR * CR
    return float(window.mean()) / (0.5 * U_IN * U_IN * area)


def test_soft_to_hard_cd_convergence() -> None:
    """C_D of the soft sphere converges monotonically to the hard-mask
    sphere of the same parameters as epsilon shrinks; the finest temperature
    lands within 1% (the docs table of the inverse-design section)."""
    mask = sphere_mask(CNX, CNY, CNZ, CCX, CCY, CCZ, CR, device=torch.device("cpu"))
    cd_hard = _cd_campaign(mask)

    errors = []
    print(f"\n  eps      C_D(soft)   C_D(hard)={cd_hard:.6f}   rel err")
    for eps in CD_EPS_SWEEP:
        geom = SoftGeometry(kind="sphere", center=(CCX, CCY, CCZ), size=(CR,), epsilon=eps)
        cd = _cd_campaign(geom)
        err = (cd - cd_hard) / cd_hard
        errors.append(abs(err))
        print(f"  {eps:6.3f}   {cd:9.6f}   {cd_hard:9.6f}   {err:+.3e}")

    assert errors[-1] < 1e-2  # 0.02 lattice units: within 1% of the hard value
    # monotone convergence: each halving of epsilon shrinks the error
    for coarse, fine in zip(errors, errors[1:]):
        assert fine < coarse


# ---------------------------------------------------------------------------
# 6. Solver-in-the-loop inverse design (the demo, small grid)
# ---------------------------------------------------------------------------

ONZ, ONY, ONX = 10, 13, 22
OCX, OCY, OCZ, OR_STAR = 6.0, 6.5, 5.0, 2.0
O_STEPS, O_PROBE_START = 50, 25
O_ITERS, O_LR = 150, 0.03


def test_inverse_design_radius_recovery() -> None:
    """Gradient descent on the radius from a wrong guess hits the reference
    C_D (same-epsilon soft rollout) to <1%, with the loss falling
    monotonically after the transient."""
    dtype = torch.float64
    device = torch.device("cpu")
    ones = torch.ones(ONZ, ONY, ONX, dtype=dtype, device=device)
    zeros = torch.zeros_like(ones)
    f0 = equilibrium3d(ones, U_IN * ones, zeros, zeros)
    inlet, outlet = InletSpec(ux=U_IN), OutletSpec()
    area = math.pi * OR_STAR * OR_STAR
    q_dyn = 0.5 * U_IN * U_IN

    def window_drag(radius: float | torch.Tensor) -> torch.Tensor:
        geom = SoftGeometry(kind="sphere", center=(OCX, OCY, OCZ), size=(radius,), epsilon=EPS_OPT)
        _f, probes = rollout(
            f0,
            O_STEPS,
            0.55,
            None,
            soft=geom,
            inlet=inlet,
            outlet=outlet,
            return_probes=True,
            probe_start=O_PROBE_START,
        )
        solid = geom.solid_weight(ONZ, ONY, ONX, dtype=dtype, device=device)
        return sum(obstacle_force(p, solid)[0] for p in probes) / len(probes)

    with torch.no_grad():
        drag_star = window_drag(torch.tensor(OR_STAR, dtype=dtype))
    cd_star = float(drag_star) / (q_dyn * area)

    radius = torch.tensor(2.6, dtype=dtype, requires_grad=True)
    optim = torch.optim.Adam([radius], lr=O_LR)
    losses = []
    for it in range(O_ITERS):
        # cosine learning-rate decay with a 5% floor (keeps a landing velocity)
        for group in optim.param_groups:
            group["lr"] = O_LR * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * it / O_ITERS)))
        loss = ((window_drag(radius) - drag_star) / drag_star) ** 2
        optim.zero_grad()
        loss.backward()
        optim.step()
        with torch.no_grad():
            radius.clamp_(0.5, min(ONY, ONZ) / 2 - 0.5)
        losses.append(float(loss.detach()))

    with torch.no_grad():
        cd_end = float(window_drag(radius.detach())) / (q_dyn * area)
    cd_err = abs(cd_end - cd_star) / cd_star
    print(
        f"\n  inverse design: C_D* = {cd_star:.6f}, endpoint C_D = {cd_end:.6f} "
        f"(err {cd_err:.2e}), radius {float(radius.detach()):.6f} vs truth {OR_STAR}"
    )

    assert cd_err < 1e-2  # endpoint within 1% of the target C_D
    assert losses[-1] < 1e-2 * losses[0]  # loss fell by > 100x
    assert abs(float(radius.detach()) - OR_STAR) < 5e-3  # the truth radius
    # monotone convergence: each quarter's mean loss strictly below the
    # previous one (Adam's per-iterate noise averages out over the quarter)
    quarters = [losses[k * O_ITERS // 4 : (k + 1) * O_ITERS // 4] for k in range(4)]
    means = [sum(q) / len(q) for q in quarters]
    assert means[3] < means[2] < means[1] < means[0], means
    assert max(quarters[3]) < means[0]  # the settled regime stays settled
