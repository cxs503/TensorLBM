"""SoA octree BFL donor reconstruction — defect probes and contract tests.

libfix_20261011 (branch ``fix/soa-solid-donor``).  Locks, in both
directions, the interpolation-source axis of
:func:`tensorlbm.octree_boundary.bfl.bfl_apply_gather`
(``bfl_source`` / env ``OCTREE_BFL_INTERP``):

* ``f_opp_post`` (historical default) reads the streamed unknown slot
  ``f[opp[d]]``; because ``stream_gather``'s SOLID branch cell-locally
  bounces, that slot is bitwise the local post-collision outgoing ``fp_d``
  and the linear Bouzidi branch degenerates to ``fp_d`` (q-blind) — probed
  here on a varying-q analytic-sphere shell.
* ``legacy`` is the documented contract (design doc §3.4 =
  ``bfl_common.bfl_bounce_back_common`` = ``bfl_d3q19``): purely
  post-collision sources ``fp_d`` / donor ``fp_up`` / ``fp_opp``, honoured
  per-link q — verified against an independent scalar recomputation of the
  Bouzidi formulas on hand-checked links.
* on a q≡0.5 geometry (the campaign shell convention,
  ``inside_fn`` without analytic q) both modes are bit-identical: the
  half-way bounce-back fixed point.
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import OPPOSITE, C, W, equilibrium3d
from tensorlbm.octree_boundary.bfl import bfl_apply_gather
from tensorlbm.octree_boundary.geometry import build_octree_shell
from tensorlbm.octree_boundary.geometry_adapters import solid_mask_inside_fn
from tensorlbm.octree_boundary.stepping import build_ghost_plan, stream_gather

DEV = torch.device("cpu")
HOST = (24, 24, 24)
CENTER = (12.0, 12.0, 12.0)
RADIUS = 5.0


def _analytic_q_shell():
    """Varying-q shell (analytic sphere q in (0, 1])."""
    octree = build_octree_shell(
        HOST,
        center=CENTER,
        radius=RADIUS,
        bl_thickness_cells=2.0,
        d_max=1,
        transition=1,
        lattice="D3Q19",
        device=DEV,
    )
    plan = build_ghost_plan(octree, HOST)
    return octree, plan


def _flat_q_shell():
    """Campaign-convention shell: mask from the neighbour table, q ≡ 0.5."""
    zz, yy, xx = torch.meshgrid(
        torch.arange(HOST[0]),
        torch.arange(HOST[1]),
        torch.arange(HOST[2]),
        indexing="ij",
    )
    solid = (xx - CENTER[2]) ** 2 + (yy - CENTER[1]) ** 2 + (zz - CENTER[0]) ** 2 <= 4.0**2
    octree = build_octree_shell(
        HOST,
        center=CENTER,
        radius=1.0,
        bl_thickness_cells=4.0,
        d_max=1,
        transition=1,
        lattice="D3Q19",
        device=DEV,
        inside_fn=solid_mask_inside_fn(solid),
    )
    plan = build_ghost_plan(octree, HOST)
    return octree, plan


def _fields(octree, plan, *, seed=7):
    """Deterministic nonuniform post-collision state + streamed state."""
    n = octree.n_leaf
    q = octree.Q
    g = torch.Generator(device="cpu").manual_seed(seed)
    rho = torch.ones((1, 1, n))
    ux = 0.05 * (1.0 + 0.4 * torch.rand((1, 1, n), generator=g))
    uy = 0.02 * (torch.rand((1, 1, n), generator=g) - 0.5)
    uz = 0.02 * (torch.rand((1, 1, n), generator=g) - 0.5)
    f = equilibrium3d(rho, ux, uy, uz, device=DEV).reshape(q, n).contiguous()
    ghost_vals = torch.zeros((q, int(plan.n_ghost)), device=DEV)
    out = stream_gather(octree, plan, f.clone(), f.clone(), ghost_vals)
    return f, out, ghost_vals


def _masked_links(octree):
    dirs, leaf = octree.bfl_mask.nonzero(as_tuple=True)
    return dirs, leaf


# ---------------------------------------------------------------------------
# (a) the degeneracy probe, both directions
# ---------------------------------------------------------------------------


def test_solid_branch_local_bounce_identity() -> None:
    """stream_gather writes the cell-local bounce fp[d] into f[opp[d]]."""
    octree, plan = _analytic_q_shell()
    f, out, _ghost = _fields(octree, plan)
    opp = OPPOSITE.to(DEV)
    dirs, leaf = _masked_links(octree)
    assert dirs.numel() > 0
    lhs = out[opp[dirs], leaf]  # f_opp_post as read by bfl_apply_gather
    rhs = f[dirs, leaf]  # fp_d (local post-collision outgoing)
    assert bool(torch.equal(lhs, rhs)), (
        "stream_gather SOLID branch must leave the cell-local bounce in the "
        "unknown slot (the degeneracy premise); if this fails the premise "
        "changed and the axis documentation must be revisited"
    )


def test_default_mode_write_is_local_bounce_on_linear_links() -> None:
    """f_opp_post mode: linear-branch write == fp_d at every masked link."""
    octree, plan = _analytic_q_shell()
    f, out, ghost_vals = _fields(octree, plan)
    opp = OPPOSITE.to(DEV)
    dirs, leaf = _masked_links(octree)
    q64 = octree.q_field[dirs, leaf].to(torch.float64)
    lin = q64 < 0.5
    assert bool(lin.any()) and bool((~lin).any())  # both branches exercised
    f_out = bfl_apply_gather(
        octree,
        out,
        f,
        ghost_plan=plan,
        ghost_vals=ghost_vals,
        bfl_source="f_opp_post",
    )
    assert bool(torch.equal(f_out[opp[dirs][lin], leaf[lin]], f[dirs[lin], leaf[lin]]))


def test_legacy_mode_write_carries_donor_information() -> None:
    """legacy mode: linear-branch write != fp_d (q/donor honoured)."""
    octree, plan = _analytic_q_shell()
    f, out, ghost_vals = _fields(octree, plan)
    opp = OPPOSITE.to(DEV)
    dirs, leaf = _masked_links(octree)
    q64 = octree.q_field[dirs, leaf].to(torch.float64)
    lin = q64 < 0.5
    nt = octree.neighbor_table
    donor = nt[opp[dirs], leaf]
    plain = lin & (donor >= 0)  # links whose fp_up is a plain-leaf donor
    assert int(plain.sum()) > 0
    f_out = bfl_apply_gather(
        octree,
        out,
        f,
        ghost_plan=plan,
        ghost_vals=ghost_vals,
        bfl_source="legacy",
    )
    write = f_out[opp[dirs][plain], leaf[plain]]
    local_bounce = f[dirs[plain], leaf[plain]]
    frac_equal = float((write == local_bounce).to(torch.float64).mean())
    assert frac_equal < 1.0, (
        "with the donor form the linear-branch reconstruction must differ "
        f"from the local bounce on q!=0.5 links (frac_equal={frac_equal})"
    )
    # the donor population itself must differ from the local outgoing on
    # some links (wall information actually flows through fp_up)
    fp_up = f[dirs[plain], donor[plain]]
    frac_donor_equal = float((fp_up == local_bounce).to(torch.float64).mean())
    assert frac_donor_equal < 1.0


def test_q_half_geometry_fix_is_bit_neutral() -> None:
    """q ≡ 0.5 (campaign shell): legacy == f_opp_post, bit for bit."""
    octree, plan = _flat_q_shell()
    f, out, ghost_vals = _fields(octree, plan)
    assert bool((octree.q_field[octree.bfl_mask] == 0.5).all())
    f_def = bfl_apply_gather(
        octree,
        out,
        f,
        ghost_plan=plan,
        ghost_vals=ghost_vals,
        bfl_source="f_opp_post",
    )
    f_fix = bfl_apply_gather(
        octree,
        out,
        f,
        ghost_plan=plan,
        ghost_vals=ghost_vals,
        bfl_source="legacy",
    )
    assert bool(torch.equal(f_def, f_fix))


# ---------------------------------------------------------------------------
# (b) hand-computed Bouzidi values on two links
# ---------------------------------------------------------------------------


def _ray_sphere_q(center_leaf, c_d, dx, sphere_c, radius):
    """First ray-sphere intersection parameter s in (0, 1] (float64).

    Independent scalar recomputation of the wall fraction along the link
    (same clamping contract as qfield.compute_q_sphere_at_points).
    """
    v = [float(c_d[0]) * dx, float(c_d[1]) * dx, float(c_d[2]) * dx]
    a = sum(vi * vi for vi in v)
    b = 2.0 * sum(vi * (float(cl) - sc) for vi, cl, sc in zip(v, center_leaf, sphere_c))
    cst = sum((float(cl) - sc) ** 2 for cl, sc in zip(center_leaf, sphere_c)) - radius**2
    disc = b * b - 4.0 * a * cst
    s = (-b - disc**0.5) / (2.0 * a)
    return min(max(s, 1e-6), 1.0)


@pytest.mark.parametrize("want_linear", [True, False])
def test_hand_computed_bouzidi_matches_gather(want_linear: bool) -> None:
    octree, plan = _analytic_q_shell()
    f, out, ghost_vals = _fields(octree, plan)
    opp = OPPOSITE.to(DEV)
    nt = octree.neighbor_table
    c64 = C.to(DEV)
    centers = octree.leaf_center.to(torch.float64)
    levels = octree.leaf_level

    chosen = None
    dirs, leaf = _masked_links(octree)
    order = torch.argsort(dirs.double() * octree.n_leaf + leaf.double())
    for k in order.tolist():
        d, i = int(dirs[k]), int(leaf[k])
        if nt[opp[d], i] < 0:  # need a plain-leaf donor for the hand check
            continue
        q = float(octree.q_field[d, i])
        if (q < 0.5) == want_linear:
            chosen = (d, i, q)
            break
    assert chosen is not None, "no link with the requested branch and donor"
    d, i, q = chosen
    donor = int(nt[opp[d], i])

    # independent q from the leaf geometry (float32 rounding like qfield)
    dx = 2.0 ** (-float(levels[i]))
    q_indep = _ray_sphere_q(centers[i], c64[d], dx, CENTER, RADIUS)
    q_indep32 = float(torch.tensor(q_indep, dtype=torch.float64).to(torch.float32))
    assert q_indep32 == q

    fp_d = float(f[d, i])  # local outgoing post-collision
    fp_up = float(f[d, donor])  # donor outgoing post-collision
    fp_opp = float(f[opp[d], i])
    if want_linear:
        expected = 2.0 * q * fp_d + (1.0 - 2.0 * q) * fp_up
    else:
        expected = fp_d / (2.0 * q) + (2.0 * q - 1.0) / (2.0 * q) * fp_opp
    expected32 = torch.tensor(expected, dtype=torch.float64).to(torch.float32)

    f_out = bfl_apply_gather(
        octree,
        out,
        f,
        ghost_plan=plan,
        ghost_vals=ghost_vals,
        bfl_source="legacy",
    )
    written = f_out[int(opp[d]), i]
    assert float(written) == float(expected32), (
        f"link (d={d}, i={i}, q={q}): written {float(written)} != hand-computed {float(expected32)}"
    )


# ---------------------------------------------------------------------------
# (c) axis plumbing: env gate, parameter precedence, fail-closed
# ---------------------------------------------------------------------------


def test_env_gate_matches_parameter(monkeypatch) -> None:
    octree, plan = _analytic_q_shell()
    f, out, ghost_vals = _fields(octree, plan)
    kw = {"ghost_plan": plan, "ghost_vals": ghost_vals}
    f_param = bfl_apply_gather(octree, out, f, bfl_source="legacy", **kw)
    monkeypatch.setenv("OCTREE_BFL_INTERP", "legacy")
    f_env = bfl_apply_gather(octree, out, f, bfl_source=None, **kw)
    assert bool(torch.equal(f_param, f_env))
    # unset env + no parameter = the historical default arithmetic
    monkeypatch.delenv("OCTREE_BFL_INTERP", raising=False)
    f_default = bfl_apply_gather(octree, out, f, **kw)
    f_off = bfl_apply_gather(octree, out, f, bfl_source="f_opp_post", **kw)
    assert bool(torch.equal(f_default, f_off))


def test_unknown_axis_fails_closed(monkeypatch) -> None:
    octree, plan = _analytic_q_shell()
    f, out, ghost_vals = _fields(octree, plan)
    with pytest.raises(ValueError, match="f_opp_post.*legacy"):
        bfl_apply_gather(
            octree,
            out,
            f,
            ghost_plan=plan,
            ghost_vals=ghost_vals,
            bfl_source="bogus",
        )
    monkeypatch.setenv("OCTREE_BFL_INTERP", "also-bogus")
    with pytest.raises(ValueError, match="f_opp_post.*legacy"):
        bfl_apply_gather(
            octree,
            out,
            f,
            ghost_plan=plan,
            ghost_vals=ghost_vals,
        )


# ---------------------------------------------------------------------------
# (d) unified leaf-wall ledger == independently computed impulse
# ---------------------------------------------------------------------------


def test_unified_leaf_ledger_equals_independent_impulse(monkeypatch) -> None:
    from tensorlbm.octree_boundary.leaf_wall import (
        build_leaf_wall_chain,
        leaf_wall_model_slip_velocity,
        leaf_wall_shear_traction,
        make_leaf_wall_chain_bfl_fn,
    )

    # the ledger is validated on the FIXED path (donor form): the independent
    # loop below recomputes the donor-based Bouzidi value, so both sides of
    # the comparison must run the same interpolation source
    monkeypatch.setenv("OCTREE_BFL_INTERP", "legacy")

    octree, plan = _analytic_q_shell()
    f, out, ghost_vals = _fields(octree, plan)
    chain = build_leaf_wall_chain(
        octree,
        u_in=0.06,
        length_host_cells=10.0,
        reynolds_physical=1.0e5,
    )
    bfl_fn, probe = make_leaf_wall_chain_bfl_fn(
        octree,
        chain,
        get_step=lambda: 4,
        ramp_steps=3,
    )
    f_out, substep_force = bfl_fn(octree, out, f, plan, ghost_vals, substep=0)
    assert probe.convention == "body"
    assert bool(torch.equal(substep_force, probe.total_force))

    # --- reproduce the chain's stages and account the impulse independently
    rho_w, uwx, uwy, uwz = leaf_wall_model_slip_velocity(
        f,
        chain.normals,
        normal_activation=1.0,
    )
    f_bfl, link_force = bfl_apply_gather(
        octree,
        out,
        f,
        ghost_plan=plan,
        ghost_vals=ghost_vals,
        wall_velocity=(uwx, uwy, uwz),
        wall_density=rho_w,
        return_force=True,
    )
    stress_normals = chain.stress_normals
    if stress_normals is None:
        stress_normals = chain.normals
    f_sheared, diag = leaf_wall_shear_traction(
        f_bfl,
        stress_normals,
        chain.near,
        chain.wall_nu,
        wall_law=chain.wall_law,
        y_val=chain.y_val,
        shear_activation=1.0,
        leaf_level=chain.leaf_level,
    )
    assert bool(torch.equal(f_sheared, f_out))  # chain determinism

    # shear on the body = -(momentum injected into the fluid by the Guo
    # forcing), measured as the discrete momentum change of the field
    c64 = C.to(torch.float64)
    dp_fluid = f_sheared.to(torch.float64) - f_bfl.to(torch.float64)
    dp_momentum = c64.t() @ dp_fluid.sum(dim=1)

    # link exchange, independently looped per link (scalar arithmetic)
    opp = OPPOSITE.to(DEV)
    nt = octree.neighbor_table
    w64 = [float(W[d]) for d in range(octree.Q)]
    link_expected = torch.zeros(3, dtype=torch.float64)
    dirs, leaf = _masked_links(octree)
    for k in range(dirs.numel()):
        d, i = int(dirs[k]), int(leaf[k])
        q = float(octree.q_field[d, i])
        fp_d = float(f[d, i])
        fp_opp = float(f[opp[d], i])
        donor = int(nt[opp[d], i])
        fp_up = float(f[d, donor]) if donor >= 0 else fp_d
        if q < 0.5:
            fbc = 2.0 * q * fp_d + (1.0 - 2.0 * q) * fp_up
        else:
            fbc = fp_d / (2.0 * q) + (2.0 * q - 1.0) / (2.0 * q) * fp_opp
        cd = c64[d]
        cu = (
            float(cd[0]) * float(uwx[i])
            + float(cd[1]) * float(uwy[i])
            + float(cd[2]) * float(uwz[i])
        )
        base = (w64[d] * float(rho_w[i])) * cu
        fbc = fbc - (6.0 if q < 0.5 else 3.0 / q) * base
        ex = fp_d + fbc
        link_expected[0] += float(cd[0]) * ex
        link_expected[1] += float(cd[1]) * ex
        link_expected[2] += float(cd[2]) * ex

    expected_total = link_expected - dp_momentum
    total = probe.total_force.to(torch.float64)
    scale = float(expected_total.abs().max()) + 1e-30
    # atol covers float64 summation-order noise only (sequential loop vs
    # torch pairwise reductions, ~1e-8 relative here); a convention error
    # would show up at ~1e-3 relative (the shear reaction is O(link)/100)
    assert torch.allclose(total, expected_total, rtol=1e-9, atol=1e-6 * scale), (
        f"unified ledger {total.tolist()} != independent impulse {expected_total.tolist()}"
    )
    # convention bookkeeping: shear entry is the reaction of the fluid-side
    # traction; the legacy mixed reading is recoverable and distinct
    shear_fluid = diag["shear_force"].to(torch.float64)
    assert torch.allclose(
        probe.shear_force.to(torch.float64),
        -shear_fluid,
        rtol=0.0,
        atol=1e-15,
    )
    assert torch.allclose(
        probe.legacy_mixed_force.to(torch.float64),
        probe.link_force.to(torch.float64) + shear_fluid,
        rtol=0.0,
        atol=1e-15,
    )
    assert not torch.allclose(
        probe.legacy_mixed_force.to(torch.float64),
        probe.total_force.to(torch.float64),
        rtol=0.0,
        atol=1e-12 * scale,
    )
    assert torch.allclose(
        probe.link_force.to(torch.float64),
        link_force.to(torch.float64),
        rtol=0.0,
        atol=1e-12 * scale,
    )


def test_legacy_mixed_convention_accumulation() -> None:
    """convention='legacy_mixed' keeps the historical near-cancelling sum."""
    from tensorlbm.octree_boundary.leaf_wall import (
        build_leaf_wall_chain,
        make_leaf_wall_chain_bfl_fn,
    )

    octree, plan = _analytic_q_shell()
    f, out, ghost_vals = _fields(octree, plan)
    chain = build_leaf_wall_chain(
        octree,
        u_in=0.06,
        length_host_cells=10.0,
        reynolds_physical=1.0e5,
    )
    bfl_fn, probe = make_leaf_wall_chain_bfl_fn(
        octree,
        chain,
        get_step=lambda: 4,
        ramp_steps=3,
        convention="legacy_mixed",
    )
    _f_out, substep_force = bfl_fn(octree, out, f, plan, ghost_vals, substep=0)
    # in legacy_mixed the shear entry is the fluid-side traction itself and
    # total == legacy_mixed_force == link + shear_fluid
    assert bool(torch.equal(substep_force, probe.total_force))
    assert bool(torch.equal(probe.total_force, probe.legacy_mixed_force))
    with pytest.raises(ValueError):
        # fail-closed on unknown conventions is on the dataclass directly
        from tensorlbm.octree_boundary.leaf_wall import LeafWallForceProbe

        LeafWallForceProbe(convention="bogus")
