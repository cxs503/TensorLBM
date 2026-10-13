"""Leaf-force ledger cadence frame regression (E3 d2 fix, 2026-10-11).

Pins the Phase 0.5 E3 lockstep adjudication
(``runs/suboff_highre_20261010/phase05/e3_lockstep_ab2.json``):

* ``leaf_force_weights``' default weight is the legacy
  "one shell call per host-grid lattice step" frame and must stay
  bit-identical (``torch.equal``) to the historical formula
  ``2^-(d_max - d)`` (× the convective spatial factor with
  ``include_spatial=True``);
* under the hybrid ``pre_restrict_hook`` cadence (shell called once per
  outer-root step, host advancing 2 lattice steps in between) the default
  ledger reads exactly HALF the true impulse at every leaf depth — the E3
  arms A/A_tau ``force_ratio_corr_over_lib = 2.0`` — reproduced here on a
  synthetic CPU case through the real ``step_octree_shell`` stepper with an
  independent analytic true force;
* ``host_steps_per_shell_call=2.0`` (the frame factor
  ``host_steps_per_root_step / shell_calls_per_root_step``) restores
  ledger/true = 1.0 at both depth tiers, reproducing bit-for-bit the formal
  runner's manual ``leaf_force_weights(...) * 2.0`` cadence-P convention.
"""

import math

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.octree_boundary.bfl import (
    bfl_apply_gather,
    leaf_force_spatial_weights,
    leaf_force_weights,
)
from tensorlbm.octree_boundary.force import substep_force_weights
from tensorlbm.octree_boundary.geometry import SOLID, build_octree_shell
from tensorlbm.octree_boundary.stepping import build_ghost_plan, step_octree_shell
from tensorlbm.static_block_amr import AMRAdvanceResult

U_IN = 0.05
TAU_HOST = 0.7
HOST_STEPS_PER_ROOT = 2.0  # StaticBlockAMR3D ratio: host lattice steps / root step
FP_TOL = 1.0e-9


def _shell(d_max: int):
    return build_octree_shell(
        (32, 32, 32),
        center=(16.0, 16.0, 16.0),
        radius=9.0,
        bl_thickness_cells=4.0,
        d_max=d_max,
        transition=1,
        lattice="D3Q19",
        device="cpu",
    )


def _legacy_weights(octree, *, include_spatial: bool) -> torch.Tensor:
    levels = octree.leaf_level.to(torch.float64)
    w = 2.0 ** (-(float(octree.d_max) - levels))
    if include_spatial:
        w = w * leaf_force_spatial_weights(octree, reference_level=1)
    return w


def _uniform_feq(q_n: int) -> torch.Tensor:
    one = torch.ones((1, 1, 1))
    return equilibrium3d(
        one,
        torch.full_like(one, U_IN),
        torch.zeros_like(one),
        torch.zeros_like(one),
    ).reshape(q_n)


# ---------------------------------------------------------------- weights --


@pytest.mark.parametrize("d_max", (1, 2))
@pytest.mark.parametrize("include_spatial", (False, True))
def test_default_weights_bit_identical_to_legacy_formula(d_max, include_spatial):
    """The fix changes no default behaviour: torch.equal on every leaf."""
    octree = _shell(d_max)
    w = leaf_force_weights(octree, include_spatial=include_spatial)
    assert torch.equal(w, _legacy_weights(octree, include_spatial=include_spatial))


@pytest.mark.parametrize("d_max", (1, 2))
@pytest.mark.parametrize("kappa", (2.0, 4.0))
def test_cadence_factor_is_exact_power_of_two_scaling(d_max, kappa):
    """kappa a power of two: the scaled weight is bit-exact (formal-runner
    ``w_lib * 2.0`` cadence-P convention == ``host_steps_per_shell_call=2``)."""
    octree = _shell(d_max)
    for include_spatial in (False, True):
        w = leaf_force_weights(
            octree,
            include_spatial=include_spatial,
            host_steps_per_shell_call=kappa,
        )
        legacy = _legacy_weights(octree, include_spatial=include_spatial)
        assert torch.equal(w, legacy * kappa)


@pytest.mark.parametrize("bad", (0.0, -1.0, float("nan"), float("inf")))
def test_cadence_frame_validation(bad):
    octree = _shell(1)
    with pytest.raises(ValueError, match="host_steps_per_shell_call"):
        leaf_force_weights(octree, host_steps_per_shell_call=bad)


def test_force_module_passthrough():
    octree = _shell(2)
    assert torch.equal(
        substep_force_weights(octree, host_steps_per_shell_call=2.0),
        leaf_force_weights(octree, include_spatial=True, host_steps_per_shell_call=2.0),
    )
    assert torch.equal(substep_force_weights(octree), _legacy_weights(octree, include_spatial=True))


# ------------------------------------------------------------ end-to-end ---


class _Capture:
    def __init__(self):
        self.samples = []

    def sink(self, d, idx, link):  # noqa: ANN001
        self.samples.append((idx, link.detach().to(torch.float64)))

    def drain(self):
        samples, self.samples = self.samples, []
        return samples


def _analytic_wall_links(octree, feq_vec):
    """Independent Bouzidi link impulses for the uniform-equilibrium field.

    ``f_opp_post`` (the post-stream unknown-direction value) mirrors the
    streamer's donor semantics: the donor of ``opp[d]`` is
    ``neighbor_table[d, i]``; a SOLID donor takes the bounce-back fallback
    ``populations[d, i]`` (== feq[d] under the static uniform field), any
    fluid-side donor (leaf / ghost / fan-out mean) carries feq[opp[d]].
    """
    opp = octree._opp
    c_vec = octree._c_vec.to(torch.float64)
    mask = octree.bfl_mask.clone()
    mask[0] = False
    cells = torch.nonzero(mask, as_tuple=False)
    d_c, i_c = cells[:, 0], cells[:, 1]
    q = octree.q_field.to(torch.float64)[d_c, i_c]
    fp_d = feq_vec[d_c].to(torch.float64)
    fp_opp = feq_vec[opp[d_c]].to(torch.float64)
    donor = octree.neighbor_table[d_c, i_c]
    f_opp_post = torch.where(donor == SOLID, fp_d, fp_opp)
    lin = q < 0.5
    safe_q = torch.where(lin, torch.ones_like(q), q)
    f_bc_lin = 2.0 * q * f_opp_post + (1.0 - 2.0 * q) * fp_d
    f_bc_quad = f_opp_post / (2.0 * safe_q) + (2.0 * safe_q - 1.0) / (2.0 * safe_q) * fp_opp
    f_bc = torch.where(lin, f_bc_lin, f_bc_quad)
    return i_c, (fp_d + f_bc).unsqueeze(1) * c_vec[d_c]


def _run_static_ledger(octree, *, calls_per_root: int, root_steps: int, weights):
    """Drive the real stepper on a uniform-equilibrium (static) field.

    Returns (ledger, truth) per leaf depth and combined: the ledger is the
    consumer convention (per-substep weighted links), the truth the E3
    cadence-correct convention (per-link sampled rate over the interval T).
    """
    shape = tuple(octree.meta["shape"])
    ghost_plan = build_ghost_plan(octree, shape)
    q_n = octree.Q
    n_leaf = int(octree.n_leaf)
    levels = octree.leaf_level.to(torch.int64)
    rho = torch.ones(shape)
    ux = torch.full_like(rho, U_IN)
    zero = torch.zeros_like(rho)
    host_eq = equilibrium3d(rho, ux, zero, zero)
    feq_vec = _uniform_feq(q_n)
    host = octree.leaf_host_cell
    octree.f_leaf = host_eq[:, host[:, 0], host[:, 1], host[:, 2]].clone()
    feq_leaf = feq_vec.unsqueeze(1).expand(q_n, n_leaf).contiguous()

    def advance(f, tau, level, substep):  # noqa: ANN001
        out = feq_leaf.clone()
        return AMRAdvanceResult(out, out)

    capture = _Capture()

    def bfl_fn(oct, out, post, gplan, ghost_vals, *, substep):  # noqa: ANN001
        f_out, _force = bfl_apply_gather(
            oct,
            out,
            post,
            ghost_plan=gplan,
            ghost_vals=ghost_vals,
            force_weights=None,  # raw impulses; weighting is ours
            return_force=True,
            link_sink=capture.sink,
        )
        # static channel: return the UNWRITTEN streamed field so the solid-
        # donor fallback of the next substep's stream stays feq (the BFL
        # boundary write would make the "unknown" directions carry f_bc and
        # drift the per-substep impulse away from the analytic value).
        return out, _force

    i_c, j_vec = _analytic_wall_links(octree, feq_vec)
    spatial = leaf_force_spatial_weights(octree, reference_level=1)
    # independent truth (constant in the static channel): per-link impulse
    # J per substep of dt_level = 2^-level host steps; over T = R host steps
    # the true impulse is spatial * J / dt_level * T.
    scale_all = spatial[i_c] * HOST_STEPS_PER_ROOT * (2.0 ** levels[i_c].to(torch.float64))
    truth = (j_vec * scale_all.unsqueeze(1)).sum(dim=0) * root_steps
    truth_depth: dict[int, torch.Tensor] = {}
    for depth in torch.unique(levels[i_c]).tolist():
        m = levels[i_c] == depth
        truth_depth[depth] = (j_vec[m] * scale_all[m].unsqueeze(1)).sum(dim=0) * root_steps

    ledger = torch.zeros(3, dtype=torch.float64)
    ledger_depth: dict[int, torch.Tensor] = {}
    for _step in range(root_steps):
        samples = []
        for _call in range(calls_per_root):
            step_octree_shell(
                octree,
                advance,
                host_eq,
                host_eq.clone(),
                tau_coarse=TAU_HOST,
                l1_post=None,
                shell_level=1,
                reflux=False,
                ghost_plan=ghost_plan,
                bfl_fn=bfl_fn,
            )
            samples.extend(capture.drain())
        # consumer ledger: sum over the root step's substeps of w * J
        vec_acc = torch.zeros((n_leaf, 3), dtype=torch.float64)
        for idx, link in samples:
            vec_acc.index_add_(0, idx, link)
        w64 = weights.to(torch.float64)
        ledger += (vec_acc * w64.unsqueeze(1)).sum(dim=0)
        for depth in sorted(truth_depth):
            # per-LEAF rows (i_c carries duplicate leaves across directions);
            # only link-carrying tiers are comparable (in the small d2 shell
            # the whole refined ring hugs the wall, so its links sit on
            # tier 2 while the d1 shell covers tier 1)
            sel = torch.nonzero(levels == depth).squeeze(1)
            part = (vec_acc[sel] * w64[sel].unsqueeze(1)).sum(dim=0)
            ledger_depth[depth] = ledger_depth.get(depth, torch.zeros(3)) + part
    return ledger, truth, ledger_depth, truth_depth


@pytest.mark.parametrize("d_max", (1, 2))
def test_hook_cadence_default_ledger_reads_half_the_true_force(d_max):
    """The E3 bug, nailed: cadence P (hook), stock weights -> exactly 0.5
    at BOTH leaf depths (true/library = 2.0, arms A/A_tau)."""
    octree = _shell(d_max)
    w_default = leaf_force_weights(octree, include_spatial=True)
    ledger, truth, ledger_depth, truth_depth = _run_static_ledger(
        octree, calls_per_root=1, root_steps=4, weights=w_default
    )
    assert abs(float(ledger[0] / truth[0]) - 0.5) <= FP_TOL
    for depth, part in ledger_depth.items():
        assert abs(float(part[0] / truth_depth[depth][0]) - 0.5) <= FP_TOL


@pytest.mark.parametrize("d_max", (1, 2))
def test_hook_cadence_frame_corrected_ledger_matches_true_force(d_max):
    """The fix: host_steps_per_shell_call = 2 (hook frame) -> exactly 1.0
    at every leaf depth (and bit-equal to the formal runner's w_lib * 2.0)."""
    octree = _shell(d_max)
    w_fixed = leaf_force_weights(octree, include_spatial=True, host_steps_per_shell_call=2.0)
    assert torch.equal(w_fixed, leaf_force_weights(octree, include_spatial=True) * 2.0)
    ledger, truth, ledger_depth, truth_depth = _run_static_ledger(
        octree, calls_per_root=1, root_steps=4, weights=w_fixed
    )
    assert abs(float(ledger[0] / truth[0]) - 1.0) <= FP_TOL
    for depth, part in ledger_depth.items():
        assert abs(float(part[0] / truth_depth[depth][0]) - 1.0) <= FP_TOL


@pytest.mark.parametrize("d_max", (1, 2))
def test_wrapper_cadence_default_ledger_matches_true_force(d_max):
    """Legacy frame (one shell call per host lattice step, arms A_S/B):
    stock weights are exact — unchanged behaviour to protect."""
    octree = _shell(d_max)
    w_default = leaf_force_weights(octree, include_spatial=True)
    ledger, truth, ledger_depth, truth_depth = _run_static_ledger(
        octree, calls_per_root=2, root_steps=4, weights=w_default
    )
    assert abs(float(ledger[0] / truth[0]) - 1.0) <= FP_TOL
    for depth, part in ledger_depth.items():
        assert abs(float(part[0] / truth_depth[depth][0]) - 1.0) <= FP_TOL


def test_static_truth_is_finite_and_nonzero():
    """Sanity: the analytic truth channel actually carries signal."""
    octree = _shell(2)
    feq_vec = _uniform_feq(octree.Q)
    _i_c, j_vec = _analytic_wall_links(octree, feq_vec)
    total = j_vec.sum(dim=0)
    assert bool(torch.isfinite(total).all())
    assert float(total.abs().max()) > 0.0
    assert math.isfinite(float(total[0]))
