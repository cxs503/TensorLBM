#!/usr/bin/env python3
"""Minimal CPU reproducer: leaf-force ledger cadence frame (E3 d2 fix, 2026-10-11).

Nails the Phase 0.5 E3 lockstep finding on a synthetic small case, without
any GPU and without the SUBOFF stack:

* a small octree shell with genuine mixed-depth leaves (``d_max=2``: level-1
  band + level-2 wall-adjacent ring, plus a ``d_max=1`` sibling) is driven
  through the real ``step_octree_shell`` stepper under the two campaign
  cadences:

  - cadence P ("hook", arms A/A_tau): ONE shell call per outer-root step
    while the host nominally advances 2 lattice steps in between
    (``StaticBlockAMR3D.step`` fires ``pre_restrict_hook`` once, after
    ``config.ratio`` = 2 fine substeps);
  - cadence S ("wrapper", arms A_S/A_S_tau): one shell call per HOST
    lattice step (2 calls per root step; the legacy two-level frame).

* the library ledger is accumulated exactly the way a consumer does
  (per-substep links weighted by ``leaf_force_weights``); the TRUE force is
  computed independently of the library weight path:

  - channel "static": every substep's populations are pinned to the uniform
    equilibrium, so each link's per-substep impulse is constant and its
    analytic value is re-derived here from the Bouzidi formula by hand
    (independent ``f_bc`` reconstruction; the aggregate first-substep raw
    link sum is cross-checked against it);
  - channel "dynamic": real BGK marching for tens of root steps; the truth
    re-derives the time normalisation from first principles per depth
    (per-link sampled rate times the interval ``T``), independent of the
    weight table.

Truth convention (the E3 ``cd_true_cadence_correct`` one): a wall link at
leaf level ``d`` carries impulse ``J`` per substep of its own convective
``dt_d = 2^-d`` host steps; over the caller's force interval ``T = R`` host
steps the true impulse is ``spatial * (J / dt_d) * T``.

Expected (measured, not assumed):

* cadence P, stock (default) weights: ledger/true = 0.5 at BOTH leaf depths
  — i.e. true/library = 2.0 exactly, the E3 ``force_ratio_corr_over_lib``
  of arms A/A_tau; the discriminator is the cadence frame (shell calls per
  root step vs host steps per root step), NOT ``d_max`` — the ``d_max=1``
  sibling shows the same 0.5 under cadence P.
* cadence S, stock weights: ledger/true = 1.0 (arms A_S/A_S_tau/B/B2).
* cadence P with the frame-corrected weight (``host_steps_per_shell_call=2``
  on the fixed tree; the explicit ``legacy * kappa`` tensor on any tree —
  the formal runner's manual cadence-P convention ``w_lib * 2.0``): 1.0.

The script runs unchanged against the frozen campaign tree (pre-fix) and
the fixed tree; ``--compare`` cross-checks the two artifacts (default
weights must be bit-identical: the fix changes no default behaviour).

Usage:
    PYTHONPATH=<tree>/src python scripts/repro_leaf_force_cadence_oversample.py \
        --out artifact.json
    python scripts/repro_leaf_force_cadence_oversample.py \
        --compare frozen.json fixed.json --compare-out verdict.json
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
import time

import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.octree_boundary.bfl import (
    bfl_apply_gather,
    leaf_force_spatial_weights,
    leaf_force_weights,
)
from tensorlbm.octree_boundary.geometry import SOLID, build_octree_shell
from tensorlbm.octree_boundary.stepping import build_ghost_plan, step_octree_shell
from tensorlbm.static_block_amr import AMRAdvanceResult

# ---------------------------------------------------------------- settings
HOST_SHAPE = (40, 40, 40)  # (nz, ny, nx) of the L1 host block (CPU)
CENTER = (20.0, 20.0, 20.0)
RADIUS = 10.0
BL = 5.0
TRANSITION = 1
U_IN = 0.05
TAU_HOST = 0.7  # drives the dynamic channel's BGK collide (stock chain)
STATIC_ROOT_STEPS = 6
DYNAMIC_ROOT_STEPS = 40
HOST_STEPS_PER_ROOT = 2.0  # StaticBlockAMR3D ratio of the campaign L1 host
CALLS_PER_ROOT = {"P": 1, "S": 2}  # cadence name -> shell calls per root step
FP_TOL = 1.0e-9  # relative tolerance on the exact rational ratios (0.5 / 1.0)


def _tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().cpu().to(torch.float64).contiguous().numpy().tobytes()
    ).hexdigest()


def _git_rev(path: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _feq_uniform(q: int, device: torch.device) -> torch.Tensor:
    rho = torch.ones((1, 1, 1), device=device)
    ux = torch.full_like(rho, U_IN)
    zero = torch.zeros_like(rho)
    return equilibrium3d(rho, ux, zero, zero, device=device).reshape(q)


def _analytic_wall_links(octree, feq_vec: torch.Tensor):
    """Independent Bouzidi link impulses for the uniform-equilibrium field.

    Re-derives ``f_bc`` from the documented formula (linear ``q < 0.5`` /
    quadratic branch, stationary wall — no library force path involved) and
    returns ``(leaf_idx, direction, impulse)`` with the impulse
    ``c_d * (f_prev[d] + f_bc)`` (float64, ``(n_m, 3)``).
    """
    device = feq_vec.device
    opp = octree._opp.to(device)
    c_vec = octree._c_vec.to(device).to(torch.float64)
    mask = octree.bfl_mask.to(device).clone()
    mask[0] = False  # the legacy contract never touches the rest direction
    cells = torch.nonzero(mask, as_tuple=False)
    d_c, i_c = cells[:, 0], cells[:, 1]
    q = octree.q_field.to(device).to(torch.float64)[d_c, i_c]
    fp_d = feq_vec[d_c].to(torch.float64)
    fp_opp = feq_vec[opp[d_c]].to(torch.float64)
    # Post-stream value of the unknown direction opp[d]: the streamer's donor
    # for opp[d] is ``neighbor_table[d, i]`` (the wall-pointing side).  A SOLID
    # donor takes the bounce-back fallback ``populations[d, i]`` — the leaf's
    # own post-collision outgoing population, == feq[d] under the uniform
    # static field — while any fluid-side donor (leaf / ghost / fan-out mean)
    # carries feq[opp[d]].  Mirroring the streamer's donor semantics here is
    # what makes the analytic independent of (but exact against) the capture.
    donor = octree.neighbor_table.to(device)[d_c, i_c]
    f_opp_post = torch.where(donor == SOLID, fp_d, fp_opp)
    lin = q < 0.5
    safe_q = torch.where(lin, torch.ones_like(q), q)
    f_bc_lin = 2.0 * q * f_opp_post + (1.0 - 2.0 * q) * fp_d
    f_bc_quad = f_opp_post / (2.0 * safe_q) + (2.0 * safe_q - 1.0) / (2.0 * safe_q) * fp_opp
    f_bc = torch.where(lin, f_bc_lin, f_bc_quad)
    exchange = fp_d + f_bc
    return i_c, d_c, exchange.unsqueeze(1) * c_vec[d_c]


class Capture:
    """Raw (unweighted) per-substep link impulse recorder (library sink)."""

    def __init__(self) -> None:
        self.samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def sink(self, d, idx, link):  # noqa: ANN001 (library callback protocol)
        self.samples.append(
            (torch.as_tensor(idx), torch.as_tensor(d), link.detach().to(torch.float64).cpu()),
        )

    def drain(self):
        samples, self.samples = self.samples, []
        return samples


def _bfl_fn_with_capture(capture: Capture, *, write_bfl: bool):
    def bfl_fn(oct, out, post, gplan, ghost_vals, *, substep):  # noqa: ANN001
        f_out, _force = bfl_apply_gather(
            oct,
            out,
            post,
            ghost_plan=gplan,
            ghost_vals=ghost_vals,
            force_weights=None,  # capture RAW impulses; weighting is ours
            return_force=True,
            link_sink=capture.sink,
        )
        if not write_bfl:
            # static channel: return the UNWRITTEN streamed field so the
            # next substep's solid-donor stream fallback stays feq (the BFL
            # boundary write would make the "unknown" directions carry f_bc
            # and drift the per-substep impulse away from the analytic value
            # that the truth channel assumes).
            return out, _force
        return f_out, _force

    return bfl_fn


def _scale_samples(samples, weights, spatial, levels, n_leaf, samples_per_link):
    """Consumer-side ledger + independent truth from the SAME raw samples.

    ``samples`` pools ALL shell calls of ONE root step (the caller's force
    interval ``T = HOST_STEPS_PER_ROOT``); ``samples_per_link`` is how often
    ONE link was sampled inside that pool — ``calls * 2^d_max`` under the
    lockstep march (every substep processes every masked link; the raw
    ``len(samples)`` instead counts direction batches, 18+ per substep, and
    is NOT the per-link count):

    ledger  = sum_substeps sum_links w_leaf * J                (the library
              consumer convention, e.g. the E3 "library" channel / the
              formal runner's weighted force),
    truth   = sum_links spatial * (mean_s J) / dt_level * T     (rate over
              the interval T, E3 cd_true_cadence_correct convention).

    Both are returned per leaf depth and combined, so the ratio is exact
    per tier.
    """
    vec_acc = torch.zeros((n_leaf, 3), dtype=torch.float64)
    for idx, _d, link in samples:
        vec_acc.index_add_(0, idx, link)
    n_samples_per_link = int(samples_per_link)
    t_host = HOST_STEPS_PER_ROOT
    w64 = weights.to(torch.float64)
    lv64 = levels.to(torch.float64)
    truth_scale = spatial * t_host * (2.0**lv64) / n_samples_per_link
    seen = torch.zeros(n_leaf, dtype=torch.bool)
    for idx, _d, _link in samples:
        seen[idx] = True
    link_list = torch.nonzero(seen, as_tuple=False).squeeze(1)
    ledger = (vec_acc * w64.unsqueeze(1)).sum(dim=0)
    truth = (vec_acc[link_list] * truth_scale[link_list].unsqueeze(1)).sum(dim=0)
    per_depth = {}
    for depth in torch.unique(levels[link_list]).tolist():
        # ledger tiers aggregate per-LEAF rows (link_list carries duplicate
        # leaves across directions); truth tiers stay per-link
        sel_leaf = torch.nonzero(levels == depth).squeeze(1)
        sel_link = link_list[levels[link_list] == depth]
        per_depth[int(depth)] = {
            "ledger": (vec_acc[sel_leaf] * w64[sel_leaf].unsqueeze(1)).sum(dim=0),
            "truth": (vec_acc[sel_link] * truth_scale[sel_link].unsqueeze(1)).sum(dim=0),
        }
    return ledger, truth, per_depth, link_list


def run_case(*, d_max: int, cadence: str, channel: str, device: torch.device) -> dict:
    torch.manual_seed(0)
    octree = build_octree_shell(
        HOST_SHAPE,
        center=CENTER,
        radius=RADIUS,
        bl_thickness_cells=BL,
        d_max=d_max,
        transition=TRANSITION,
        lattice="D3Q19",
        device=device,
    )
    ghost_plan = build_ghost_plan(octree, HOST_SHAPE)
    q_n = octree.Q
    n_leaf = int(octree.n_leaf)
    levels = octree.leaf_level.to(torch.int64)

    rho = torch.ones(HOST_SHAPE, device=device)
    ux = torch.full_like(rho, U_IN)
    zero = torch.zeros_like(rho)
    host_eq = equilibrium3d(rho, ux, zero, zero, device=device)
    feq_vec = _feq_uniform(q_n, device)
    host = octree.leaf_host_cell
    octree.f_leaf = host_eq[:, host[:, 0], host[:, 1], host[:, 2]].clone()

    if channel == "static":
        feq_leaf = feq_vec.unsqueeze(1).expand(q_n, n_leaf).contiguous()

        def advance(f, tau, level, substep):  # noqa: ANN001
            out = feq_leaf.clone()
            return AMRAdvanceResult(out, out)

        root_steps = STATIC_ROOT_STEPS
    else:

        def advance(f, tau, level, substep):  # noqa: ANN001
            from tensorlbm.solver3d import collide_bgk3d

            post = collide_bgk3d(f.view(q_n, 1, 1, -1), tau).view(q_n, -1).to(f.dtype)
            return AMRAdvanceResult(post.view_as(f).clone(), post.view_as(f))

        root_steps = DYNAMIC_ROOT_STEPS

    capture = Capture()
    bfl_fn = _bfl_fn_with_capture(capture, write_bfl=channel != "static")

    # weights --------------------------------------------------------------
    w_default = leaf_force_weights(octree, include_spatial=True)
    spatial = leaf_force_spatial_weights(octree, reference_level=1)
    legacy_inline = (2.0 ** (-(float(d_max) - levels.to(torch.float64)))) * spatial
    default_bit_identical = bool(torch.equal(w_default.to(torch.float64), legacy_inline))
    calls = CALLS_PER_ROOT[cadence]
    kappa = HOST_STEPS_PER_ROOT / calls
    w_frame = legacy_inline * kappa
    sig = inspect.signature(leaf_force_weights)
    w_kwargs = None
    if "host_steps_per_shell_call" in sig.parameters:
        w_kwargs = leaf_force_weights(
            octree,
            include_spatial=True,
            host_steps_per_shell_call=kappa,
        )

    # independent analytic truth (static channel) ---------------------------
    analytic = _analytic_wall_links(octree, feq_vec)
    analytic_total = analytic[2].sum(dim=0)
    first_raw_total = None

    # run -------------------------------------------------------------------
    ledger_default = torch.zeros(3, dtype=torch.float64)
    ledger_frame = torch.zeros(3, dtype=torch.float64)
    truth_total = torch.zeros(3, dtype=torch.float64)
    ledger_default_depth: dict[int, torch.Tensor] = {}
    ledger_frame_depth: dict[int, torch.Tensor] = {}
    truth_depth: dict[int, torch.Tensor] = {}
    ratios_default_series: list[float] = []
    t0 = time.time()
    for _step in range(root_steps):
        step_samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for _call in range(calls):
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
            step_samples.extend(capture.drain())
            if first_raw_total is None:
                first_raw_total = torch.cat([link for _i, _d, link in step_samples], dim=0).sum(
                    dim=0
                )
        led_d, tru, per_depth, _links = _scale_samples(
            step_samples,
            w_default.to(device),
            spatial,
            levels,
            n_leaf,
            calls * (1 << d_max),
        )
        led_f, _tru2, _pd2, _l2 = _scale_samples(
            step_samples,
            w_frame.to(device),
            spatial,
            levels,
            n_leaf,
            calls * (1 << d_max),
        )
        ledger_default += led_d
        ledger_frame += led_f
        truth_total += tru
        for depth, parts in per_depth.items():
            ledger_default_depth[depth] = (
                ledger_default_depth.get(depth, torch.zeros(3)) + parts["ledger"]
            )
            truth_depth[depth] = truth_depth.get(depth, torch.zeros(3)) + parts["truth"]
        for depth, parts in _pd2.items():
            ledger_frame_depth[depth] = (
                ledger_frame_depth.get(depth, torch.zeros(3)) + parts["ledger"]
            )
        ratios_default_series.append(float(led_d[0] / tru[0]) if tru[0] != 0.0 else float("nan"))
    wall = time.time() - t0

    def _ratio(a, b):
        return float(a[0] / b[0]) if b[0] != 0.0 else float("nan")

    n_by_depth = {int(d): int((levels == d).sum().item()) for d in range(1, d_max + 1)}
    case = {
        "d_max": d_max,
        "cadence": cadence,
        "channel": channel,
        "host_steps_per_root_step": HOST_STEPS_PER_ROOT,
        "shell_calls_per_root_step": float(calls),
        "kappa_host_steps_per_shell_call": kappa,
        "n_leaf": n_leaf,
        "n_leaf_by_depth": n_by_depth,
        "bfl_wall_links": int(octree.bfl_mask.sum().item()),
        "samples_per_link_per_root_step": float(calls * (1 << d_max)),
        "root_steps": root_steps,
        "weights_default_bit_identical_to_legacy_formula": default_bit_identical,
        "weights_default_sha256": _tensor_sha256(w_default),
        "weights_default_unique_by_depth": {
            str(d): sorted(set(w_default[levels == d].to(torch.float64).tolist()))
            for d in n_by_depth
        },
        "weights_kwarg_available": w_kwargs is not None,
        "weights_kwarg_bit_identical_to_legacy_times_kappa": (
            None if w_kwargs is None else bool(torch.equal(w_kwargs.to(torch.float64), w_frame))
        ),
        "ratio_ledger_default_over_truth": _ratio(ledger_default, truth_total),
        "ratio_ledger_frame_over_truth": _ratio(ledger_frame, truth_total),
        "truth_over_ledger_default": 1.0 / _ratio(ledger_default, truth_total),
        "ratio_ledger_default_over_truth_by_depth": {
            str(d): _ratio(ledger_default_depth[d], truth_depth[d]) for d in sorted(truth_depth)
        },
        "ratio_ledger_frame_over_truth_by_depth": {
            str(d): _ratio(ledger_frame_depth[d], truth_depth[d]) for d in sorted(truth_depth)
        },
        "ratio_default_series_min": min(ratios_default_series),
        "ratio_default_series_max": max(ratios_default_series),
        "finite": bool(
            torch.isfinite(ledger_default).all()
            and torch.isfinite(ledger_frame).all()
            and torch.isfinite(truth_total).all()
        ),
        "wall_seconds": wall,
        "ledger_default_total": [float(x) for x in ledger_default.tolist()],
        "ledger_frame_total": [float(x) for x in ledger_frame.tolist()],
        "truth_total": [float(x) for x in truth_total.tolist()],
    }
    if channel == "static":
        # the first call pools all its 2^d_max substeps, each of which carries
        # exactly the (constant) analytic per-substep impulse
        n_substeps = 1 << d_max
        rel = (first_raw_total - analytic_total * n_substeps).abs().max().item() / max(
            (analytic_total * n_substeps).abs().max().item(), 1.0e-300
        )
        case["static_first_substep_raw_vs_analytic_rel"] = rel
        i_c, _d_c, j_vec = analytic
        rate_scale = spatial[i_c] * HOST_STEPS_PER_ROOT * (2.0 ** levels[i_c].to(torch.float64))
        analytic_rate = (j_vec * rate_scale.unsqueeze(1)).sum(dim=0) * root_steps
        # compare on the signal axis (x): the transverse components cancel to
        # roundoff noise (~1e-13 absolute) where an allclose would trip on
        # atol alone
        case["static_truth_equals_analytic_rate_x"] = bool(
            abs(float(truth_total[0]) - float(analytic_rate[0]))
            <= 1.0e-12 * max(abs(float(analytic_rate[0])), 1.0e-300)
        )
        case["static_truth_vs_analytic_rel_x"] = abs(
            float(truth_total[0]) - float(analytic_rate[0])
        ) / max(abs(float(analytic_rate[0])), 1.0e-300)
        case["static_analytic_rate_truth"] = [float(x) for x in analytic_rate.tolist()]
    return case


def run_all(device: torch.device) -> dict:
    cases = []
    for d_max in (2, 1):
        for cadence in ("P", "S"):
            cases.append(run_case(d_max=d_max, cadence=cadence, channel="static", device=device))
    for cadence in ("P", "S"):
        cases.append(run_case(d_max=2, cadence=cadence, channel="dynamic", device=device))
    tensorlbm_file = sys.modules["tensorlbm"].__file__ or ""
    root = "/".join(tensorlbm_file.split("/")[:-2])
    return {
        "schema": "suboff-highre-d2fix-leaf-force-cadence-repro/v1",
        "tensorlbm_file": tensorlbm_file,
        "tree_root": root,
        "tree_git_rev": _git_rev(root),
        "device": str(device),
        "torch_version": torch.__version__,
        "fp_tol": FP_TOL,
        "settings": {
            "host_shape": list(HOST_SHAPE),
            "center": list(CENTER),
            "radius": RADIUS,
            "bl": BL,
            "transition": TRANSITION,
            "u_in": U_IN,
            "tau_host": TAU_HOST,
            "static_root_steps": STATIC_ROOT_STEPS,
            "dynamic_root_steps": DYNAMIC_ROOT_STEPS,
            "host_steps_per_root_step": HOST_STEPS_PER_ROOT,
            "calls_per_root_step": CALLS_PER_ROOT,
        },
        "cases": cases,
    }


def compare(old_path: str, new_path: str, out_path: str | None) -> dict:
    with open(old_path) as fh:
        old = json.load(fh)
    with open(new_path) as fh:
        new = json.load(fh)
    by_key_old = {(c["d_max"], c["cadence"], c["channel"]): c for c in old["cases"]}
    by_key_new = {(c["d_max"], c["cadence"], c["channel"]): c for c in new["cases"]}
    verdict: dict = {
        "schema": "suboff-highre-d2fix-leaf-force-cadence-verdict/v1",
        "old": {"path": old_path, "tree_git_rev": old.get("tree_git_rev")},
        "new": {"path": new_path, "tree_git_rev": new.get("tree_git_rev")},
        "fp_tol": FP_TOL,
        "default_weights_bit_identical_across_trees": {},
        "true_over_library_default": {},
        "frame_corrected_over_truth": {},
        "true_over_library_default_by_depth": {},
        "frame_corrected_over_truth_by_depth": {},
    }
    tol = max(old.get("fp_tol", FP_TOL), new.get("fp_tol", FP_TOL))
    for key in sorted(by_key_old):
        label = f"d{key[0]}_{key[1]}_{key[2]}"
        c_old, c_new = by_key_old[key], by_key_new[key]
        verdict["default_weights_bit_identical_across_trees"][label] = (
            c_old["weights_default_sha256"] == c_new["weights_default_sha256"]
        )
        verdict["true_over_library_default"][label] = c_new["truth_over_ledger_default"]
        verdict["frame_corrected_over_truth"][label] = c_new["ratio_ledger_frame_over_truth"]
        verdict["true_over_library_default_by_depth"][label] = {
            d: 1.0 / r for d, r in c_new["ratio_ledger_default_over_truth_by_depth"].items()
        }
        verdict["frame_corrected_over_truth_by_depth"][label] = c_new[
            "ratio_ledger_frame_over_truth_by_depth"
        ]
    gates = {
        "default_weights_bit_identical_across_trees": all(
            verdict["default_weights_bit_identical_across_trees"].values()
        ),
        "hook_true_over_library_equals_2_both_depths": all(
            abs(v - 2.0) <= tol
            for k, v in verdict["true_over_library_default"].items()
            if "_P_" in k
        ),
        "hook_true_over_library_equals_2_both_depths_by_tier": all(
            abs(v - 2.0) <= tol
            for k, tiers in verdict["true_over_library_default_by_depth"].items()
            if "_P_" in k
            for v in tiers.values()
        ),
        "wrapper_true_over_library_equals_1": all(
            abs(v - 1.0) <= tol
            for k, v in verdict["true_over_library_default"].items()
            if "_S_" in k
        ),
        "hook_frame_corrected_equals_1_both_depths": all(
            abs(v - 1.0) <= tol
            for k, v in verdict["frame_corrected_over_truth"].items()
            if "_P_" in k
        ),
        "wrapper_frame_corrected_equals_1": all(
            abs(v - 1.0) <= tol
            for k, v in verdict["frame_corrected_over_truth"].items()
            if "_S_" in k
        ),
        "kwarg_available_on_new_tree": by_key_new[(2, "P", "static")]["weights_kwarg_available"],
        "kwarg_bit_identical_to_legacy_times_kappa": all(
            c["weights_kwarg_bit_identical_to_legacy_times_kappa"]
            for c in by_key_new.values()
            if c["weights_kwarg_bit_identical_to_legacy_times_kappa"] is not None
        ),
        "static_raw_links_match_analytic": all(
            c.get("static_first_substep_raw_vs_analytic_rel", 0.0) <= 1.0e-12
            for c in by_key_new.values()
        ),
    }
    verdict["gates"] = gates
    verdict["all_gates_pass"] = all(gates.values())
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(verdict, fh, indent=1, sort_keys=True)
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default=None, help="write the artifact JSON here")
    parser.add_argument("--compare", nargs=2, metavar=("OLD", "NEW"))
    parser.add_argument("--compare-out", type=str, default=None)
    args = parser.parse_args()
    if args.compare:
        verdict = compare(args.compare[0], args.compare[1], args.compare_out)
        print(json.dumps(verdict, indent=1, sort_keys=True))
        return
    device = torch.device("cpu")
    report = run_all(device)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1, sort_keys=True)
    for c in report["cases"]:
        print(
            f"[repro] d_max={c['d_max']} cadence={c['cadence']} {c['channel']}: "
            f"lib/true={c['ratio_ledger_default_over_truth']:.12f} "
            f"frame/true={c['ratio_ledger_frame_over_truth']:.12f} "
            f"by_depth={c['ratio_ledger_default_over_truth_by_depth']} "
            f"n_leaf={c['n_leaf']} links={c['bfl_wall_links']} "
            f"wall={c['wall_seconds']:.1f}s",
            flush=True,
        )
    print(f"tree={report['tree_root']} rev={report['tree_git_rev']}")
    if args.out:
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
