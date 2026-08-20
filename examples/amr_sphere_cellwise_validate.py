#!/usr/bin/env python3
"""Cell-wise adaptive AMR sphere drag validation: body-fitted patches + patch-local BFL.

Route-3 adaptive refinement: only cells flagged by the error indicator are
refined (no whole-box concept).  This runner fixes the root causes of the
poor Cd (~64% error) of examples/amr_sphere_adaptive_validate.py and of the
first cellwise revision (Cd ~ 14, ~1200% error at Re=100):

a. **Body-fitted guarantee** — the indicator forces indicator = 1.0 on every
   coarse fluid cell adjacent to the sphere surface (the cells whose plain
   bounce-back links the fine BFL must replace), on top of
   ``boundary_layer_indicator_3d(mask, re, char_length=2R)`` which flags the
   fluid cells within δ = 5·(2R)/√Re of the surface.  With refine_threshold
   at 1e-6 the grouped patches are guaranteed to cover 100% of the sphere
   surface regardless of δ (a too-small δ on the coarse grid leaves surface
   cells on the coarse plain bounce-back grid, which is the dominant error
   source).

b. **Convective fine-grid relaxation time** — ``AdaptiveSolver3D._tau_for_level``
   computes τ_f = 0.5 + (τ_c−0.5)/2, which *halves* the lattice viscosity on
   the fine level.  With dt_f = dt_c/2 that runs the fine patch at Re_f =
   4·Re_c (here 400 instead of 100), depressing the fine BFL drag to ~0.47
   instead of ~1.1.  The repo's canonical convective scaling (used by the
   L3 reference, ``static_block_amr.convective_refined_tau``) requires
   τ_f = 0.5 + ratio·(τ_c−0.5) so the physical viscosity is preserved across
   the 2:1 interface.  The runner overrides every patch's τ after each
   adaptation (collision, FH interface rescale and restriction all read
   ``patch.tau``).

c. **Body-fitted drag measurement** — the coarse-grid control-volume force
   (``cd_cv_only``) already contains the coarse plain bounce-back sphere's
   surface momentum sink, which at R=7 coarse resolution is resolution
   garbage (Cd ≈ 14 instead of 1.09).  Adding the fine BFL force on top
   double-counts the surface.  The reported drag is therefore the
   patch-local BFL momentum-exchange force (fine lattice units, converted to
   coarse units by /ratio² — equivalently normalised by the fine dynamic
   area 0.5·u²·π·(R·ratio)²); the coarse CV force is kept as a diagnostic
   (``cd_cv_only``) and as the plain-BB baseline when ``--bfl-on-patches
   False``.

This is a validation runner: status=measured_candidate, physical_validation
is only set when the run passes all gates.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch
from torch.nn.functional import conv3d

from tensorlbm.adaptive_refinement import (
    AdaptationSchedule,
    AdaptiveSolver3D,
    AMRPatch3D,
    boundary_layer_indicator_3d,
    nonequilibrium_indicator_3d,
    vorticity_indicator_3d,
)
from tensorlbm.bfl_d3q19 import bouzidi_bounce_back_d3q19
from tensorlbm.boundaries3d import (
    bounce_back_cells_3d,
    far_field_bc_3d,
    sphere_mask,
)
from tensorlbm.control_volume_force import observe_control_volume_force
from tensorlbm.cumulant import collide_cumulant_d3q19
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.evidence_io import common_schema_fields, write_evidence
from tensorlbm.external_open_boundary import non_equilibrium_far_field_bc_3d
from tensorlbm.interpolated_bc import compute_q_sphere
from tensorlbm.solver3d import stream3d
from tensorlbm.sphere_amr_common import (
    build_control_volume,
    build_sphere_geometry,
    ramp_activation,
    summarize_force_history,
)
from tensorlbm.sponge_layer import (
    apply_equilibrium_difference_sponge,
    build_sponge_sigma_3d,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--nx", type=int, default=160)
    p.add_argument("--ny", type=int, default=112)
    p.add_argument("--nz", type=int, default=112)
    p.add_argument("--radius", type=float, default=8.0)
    p.add_argument("--reynolds", type=float, default=100.0)
    p.add_argument("--lattice-speed", type=float, default=0.06)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--warmup-steps", type=int, default=1500)
    p.add_argument("--ramp-steps", type=int, default=500)
    p.add_argument("--adapt-interval", type=int, default=50)
    p.add_argument("--adapt-warmup", type=int, default=200)
    p.add_argument("--max-patches", type=int, default=8)
    p.add_argument("--max-levels", type=int, default=2)
    p.add_argument("--refine-threshold", type=float, default=1e-6)
    p.add_argument("--coarsen-threshold", type=float, default=1e-5)
    p.add_argument("--sponge-width", type=int, default=16)
    p.add_argument("--sponge-strength", type=float, default=0.2)
    p.add_argument("--cv-margin", type=int, default=5)
    p.add_argument("--report-interval", type=int, default=500)
    p.add_argument("--statistics-window-steps", type=int, default=0)
    p.add_argument(
        "--indicator",
        choices=("nonequilibrium", "vorticity", "boundary_layer", "bl+neq"),
        default="boundary_layer",
        help=(
            "refinement indicator: boundary_layer is body-fitted (guarantees "
            "patches cover the sphere); bl+neq takes the element-wise max of "
            "boundary_layer and nonequilibrium (sphere + wake refinement)."
        ),
    )

    def _parse_bool(value: str) -> bool:
        return value.lower() in ("1", "true", "yes", "on")

    p.add_argument(
        "--bfl-on-patches",
        nargs="?",
        const=True,
        type=_parse_bool,
        default=True,
        help=(
            "apply patch-local Bouzidi BFL on the fine sphere surface inside "
            "every patch (replaces plain bounce-back there).  Accepts "
            "'--bfl-on-patches' (true) or '--bfl-on-patches False'.  Disable "
            "to get the plain-BB baseline for comparison."
        ),
    )
    p.add_argument(
        "--far-field-mode",
        choices=("non_equilibrium_extrapolation", "legacy_hard_equilibrium"),
        default="non_equilibrium_extrapolation",
    )
    p.add_argument("--output", required=True)
    return p


def main() -> None:
    args = parser().parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    shape = (args.nz, args.ny, args.nx)
    cx, cy, cz = args.nx * 0.5, args.ny / 2.0, args.nz / 2.0
    solid, solid_q = build_sphere_geometry(
        args.nx,
        args.ny,
        args.nz,
        cx,
        cy,
        cz,
        args.radius,
        device,
    )
    shape_q = (19, *shape)

    nu = args.lattice_speed * (2.0 * args.radius) / args.reynolds
    tau = 0.5 + 3.0 * nu

    rho = torch.ones(shape, device=device)
    ux = torch.full_like(rho, args.lattice_speed)
    zero = torch.zeros_like(rho)
    f = equilibrium3d(rho, ux, zero, zero, device=device)

    schedule = AdaptationSchedule(
        interval=args.adapt_interval,
        warmup=args.adapt_warmup,
        max_patches=args.max_patches,
        max_levels=args.max_levels,
        refine_threshold=args.refine_threshold,
        # mark_cells_for_refinement requires coarsen < refine; the boundary
        # layer indicator is binary (1.0 on the body-fitted surface, 0
        # elsewhere) so the exact coarsen level is immaterial — patches that
        # carry the sphere surface always keep local max = 1.0.
        coarsen_threshold=min(
            args.coarsen_threshold,
            args.refine_threshold * 0.5,
        ),
        tau=tau,
    )
    solver = AdaptiveSolver3D(f, schedule=schedule, mask=solid)

    cv = build_control_volume(
        shape,
        (cx, cy, cz),
        args.radius,
        args.cv_margin,
        device,
    )
    sponge_faces = ("x+", "y-", "y+", "z-", "z+")
    sigma = build_sponge_sigma_3d(
        shape,
        width=args.sponge_width,
        max_strength=args.sponge_strength,
        device=device,
        faces=sponge_faces,
    )
    dynamic_area = 0.5 * args.lattice_speed**2 * math.pi * args.radius**2

    # ------------------------------------------------------------------
    # Per-patch geometry (rebuilt at every adaptation step).
    #   patch_data[id(p)] = {
    #       "solid":    fine analytical sphere mask in patch-local coords,
    #       "bfl_mask": compute_q_sphere fluid-boundary mask (or None),
    #       "bfl_q":    per-direction fractional distances (or None),
    #   }
    # The patch fine tensor has no ghost cells: fine index k covers global
    # coarse interval [origin + k/ratio, origin + (k+1)/ratio), so the
    # patch-local fine sphere centre is (global_center - origin) * ratio and
    # the fine radius is radius * ratio (single-layer fine_center convention
    # with ghost width g = 0).
    # ------------------------------------------------------------------
    patch_data: dict[int, dict[str, object]] = {}
    patch_by_id: dict[int, AMRPatch3D] = {}
    patch_post: dict[int, torch.Tensor] = {}
    bfl_accum: dict[int, tuple[float, float, float]] = {}
    bfl_count: dict[int, int] = {}

    # Coarse sphere-surface cells: fluid cells with a solid neighbour
    # (Chebyshev distance 1) — exactly the cells whose plain bounce-back
    # links the fine BFL must replace.  Used for the body-fitted indicator
    # and the surface-coverage diagnostic.
    _spread = conv3d(
        solid.unsqueeze(0).unsqueeze(0).float(),
        torch.ones((1, 1, 3, 3, 3), dtype=torch.float32, device=device),
        padding=1,
    ).clamp(0, 1)
    surface_cells = (_spread.squeeze() > 0) & ~solid

    def _patch_global_origin(p) -> tuple[float, float, float]:
        """Global coarse-coordinate origin of patch p's fine grid."""
        if p.parent_level == 0:
            return (float(p.box.x0), float(p.box.y0), float(p.box.z0))
        parent = None
        for q in solver.patches:
            if q.level == p.parent_level:
                b, pb = q.box, p.box
                if (
                    b.x0 <= pb.x0
                    and b.x1 >= pb.x1
                    and b.y0 <= pb.y0
                    and b.y1 >= pb.y1
                    and b.z0 <= pb.z0
                    and b.z1 >= pb.z1
                ):
                    parent = q
                    break
        if parent is None:
            return (float(p.box.x0), float(p.box.y0), float(p.box.z0))
        pox, poy, poz = _patch_global_origin(parent)
        rp = float(parent.ratio)
        return (pox + p.box.x0 / rp, poy + p.box.y0 / rp, poz + p.box.z0 / rp)

    def _build_patch_data() -> None:
        patch_data.clear()
        for p in solver.patches:
            gx, gy, gz = _patch_global_origin(p)
            r = p.ratio
            nz_f, ny_f, nx_f = p.f.shape[1:]
            cxf = (cx - gx) * r
            cyf = (cy - gy) * r
            czf = (cz - gz) * r
            local_solid = sphere_mask(
                nx_f,
                ny_f,
                nz_f,
                cxf,
                cyf,
                czf,
                args.radius * r,
                device=device,
            )
            entry: dict[str, object] = {
                "solid": local_solid,
                "bfl_mask": None,
                "bfl_q": None,
            }
            if bool(local_solid.any()):
                bfl_mask, bfl_q = compute_q_sphere(
                    nx_f,
                    ny_f,
                    nz_f,
                    cxf,
                    cyf,
                    czf,
                    args.radius * r,
                    device=device,
                )
                entry["bfl_mask"] = bfl_mask
                entry["bfl_q"] = bfl_q
            patch_data[id(p)] = entry

    def _solid_coverage() -> float:
        """Fraction of coarse sphere solid cells inside any patch box."""
        cover = torch.zeros(shape, dtype=torch.bool, device=device)
        for p in solver.patches:
            b = p.box
            cover[b.z0 : b.z1, b.y0 : b.y1, b.x0 : b.x1] = True
        total = int(solid.sum().item())
        if total == 0:
            return 1.0
        return float((cover & solid).sum().item()) / total

    def _surface_coverage() -> float:
        """Fraction of coarse sphere-surface cells inside a BFL-capable patch.

        Only patches whose fine sphere actually touches the surface
        (``bfl_mask`` present) count — a strip box that merely overlaps the
        δ-shell but contains no sphere still does plain bounce-back there.
        """
        cover = torch.zeros(shape, dtype=torch.bool, device=device)
        for p in solver.patches:
            entry = patch_data.get(id(p), {})
            if entry.get("bfl_mask") is None:
                continue
            b = p.box
            cover[b.z0 : b.z1, b.y0 : b.y1, b.x0 : b.x1] = True
        total = int(surface_cells.sum().item())
        if total == 0:
            return 1.0
        return float((cover & surface_cells).sum().item()) / total

    def _body_fitted_indicator(
        bl: torch.Tensor,
    ) -> torch.Tensor:
        """Boundary-layer indicator with the sphere surface forced to 1.0.

        Guarantees the grouped patches cover 100% of the sphere surface even
        when δ = 5·(2R)/√Re is smaller than one coarse cell: every coarse
        fluid cell adjacent to the sphere (the cell whose plain bounce-back
        link the fine BFL must replace) is set to the maximum indicator
        value, far above refine_threshold.
        """
        return torch.where(
            surface_cells,
            torch.ones_like(bl),
            bl,
        )

    def collide_fn(state: torch.Tensor) -> torch.Tensor:
        # Coarse background grid: freeze solid cells at their pre-collision
        # state (plain no-slip bounce-back via the boundary stage).
        if state.shape == shape_q:
            collided = collide_cumulant_d3q19(state, tau, C_s=0.0)
            return torch.where(solid_q, state, collided)
        # Fine patches: freeze the fine analytical sphere and stash the
        # post-collision, pre-stream state — the BFL f_prev for this substep.
        for p in solver.patches:
            if p.f is state:
                entry = patch_data.get(id(p))
                collided = collide_cumulant_d3q19(state, p.tau, C_s=0.0)
                if entry is None:
                    return collided
                local_q = entry["solid"].unsqueeze(0).expand_as(state)
                out = torch.where(local_q, state, collided)
                patch_post[id(p)] = out
                return out
        return collide_cumulant_d3q19(state, tau, C_s=0.0)

    def boundary_fn(state: torch.Tensor) -> torch.Tensor:
        if state.shape == shape_q:
            state = bounce_back_cells_3d(state, solid)
            if args.far_field_mode == "non_equilibrium_extrapolation":
                state = non_equilibrium_far_field_bc_3d(
                    state,
                    u_in=args.lattice_speed,
                )
            else:
                state = far_field_bc_3d(state, u_in=args.lattice_speed)
            state = apply_equilibrium_difference_sponge(
                state,
                sigma,
                velocity_target=(args.lattice_speed, 0.0, 0.0),
            )
            if args.far_field_mode == "non_equilibrium_extrapolation":
                state = non_equilibrium_far_field_bc_3d(
                    state,
                    u_in=args.lattice_speed,
                )
            else:
                state = far_field_bc_3d(state, u_in=args.lattice_speed)
            return state
        # Patch tensors are interior; the border is re-injected from the
        # parent every patch substep.  With BFL enabled the fine sphere
        # surface inside the patch is handled by the curved-wall Bouzidi
        # reconstruction (replacing plain bounce-back); the frozen solid
        # cells set during collision never feed the fluid directly because
        # every solid→fluid link is reconstructed by the BFL.  Without BFL
        # we fall back to plain post-stream bounce-back (baseline).
        for p in solver.patches:
            if p.f is state:
                entry = patch_data.get(id(p))
                if entry is None:
                    return state
                if args.bfl_on_patches and entry["bfl_mask"] is not None:
                    post = patch_post.get(id(p))
                    if post is not None:
                        rho_post, ux_post, uy_post, uz_post = macroscopic3d(post)
                        activation = ramp_activation(current_step, args.ramp_steps)
                        wall_velocity = (
                            (1.0 - activation) * ux_post,
                            (1.0 - activation) * uy_post,
                            (1.0 - activation) * uz_post,
                        )
                        state, bfl_force = bouzidi_bounce_back_d3q19(
                            state,
                            post,
                            entry["bfl_mask"],
                            entry["bfl_q"],
                            wall_velocity=wall_velocity,
                            wall_density=rho_post,
                            return_force=True,
                        )
                        # Accumulate the per-fine-step momentum exchange over
                        # all patch substeps of this coarse step; the mean is
                        # converted to coarse lattice units below (/ratio²).
                        sx, sy, sz = bfl_accum.get(id(p), (0.0, 0.0, 0.0))
                        bfl_accum[id(p)] = (
                            sx + float(bfl_force[0]),
                            sy + float(bfl_force[1]),
                            sz + float(bfl_force[2]),
                        )
                        bfl_count[id(p)] = bfl_count.get(id(p), 0) + 1
                else:
                    state = bounce_back_cells_3d(state, entry["solid"])
                return state
        return state

    forces: list[float] = []
    cv_forces: list[float] = []
    bfl_forces: list[float] = []
    patch_counts: list[int] = []
    coverages: list[float] = []
    surface_coverages: list[float] = []
    started = time.time()

    current_step = 0
    for current_step in range(1, args.steps + 1):
        before = solver.coarse_f.clone()
        bfl_accum.clear()
        bfl_count.clear()
        solver.step(collide_fn, stream3d, boundary_fn)
        # Per-patch BFL momentum-exchange force of this coarse step: mean of
        # the per-fine-step exchanges over the patch substeps, converted to
        # coarse lattice units.  With dx_f = dx_c/ratio and dt_f = dt_c/ratio
        # a fine-lattice force scales as (dx_f^4/dt_f^2) =
        # (dx_c^4/dt_c^2)/ratio², hence /ratio² (equivalently: normalise the
        # fine-unit force by the fine dynamic area 0.5·u²·π·(R·ratio)² —
        # both give the same Cd).
        total_bfl_fine = 0.0
        fine_ratio = 1
        for pid, force_sum in bfl_accum.items():
            p = patch_by_id[pid]
            n = max(1, bfl_count.get(pid, 1))
            total_bfl_fine += force_sum[0] / n
            fine_ratio = max(fine_ratio, int(p.ratio))
        total_bfl_coarse = total_bfl_fine / float(fine_ratio**2)
        if current_step > args.warmup_steps:
            # observe_control_volume_force's third argument must be the
            # post-collision, pre-stream state (streaming_momentum_import
            # samples populations from their pre-stream source cells).
            # collide is deterministic and side-effect free on the coarse
            # grid, so recompute it from the pre-step state.
            post_collide = collide_fn(before)
            cv_force = float(
                observe_control_volume_force(
                    before,
                    solver.coarse_f,
                    post_collide,
                    cv,
                    solid=solid,
                )
                .force_on_body[0]
                .item()
            )
            # Body-fitted drag: the fine BFL surface force.  The coarse CV
            # force already contains the coarse plain-BB sphere's surface
            # momentum sink (resolution garbage at R=7, Cd≈14); adding it
            # would double-count the surface.  It is kept as cd_cv_only and
            # used as the total when BFL is disabled (plain-BB baseline).
            if args.bfl_on_patches:
                total_force = total_bfl_coarse
            else:
                total_force = cv_force
            forces.append(total_force)
            cv_forces.append(cv_force)
            bfl_forces.append(total_bfl_coarse)
        patch_counts.append(len(solver.patches))
        if solver.should_adapt(current_step):
            rho, ux, uy, uz = macroscopic3d(solver.coarse_f)
            if args.indicator == "nonequilibrium":
                indicator = nonequilibrium_indicator_3d(
                    solver.coarse_f,
                    rho,
                    ux,
                    uy,
                    uz,
                )
            elif args.indicator == "vorticity":
                indicator = vorticity_indicator_3d(ux, uy, uz)
            elif args.indicator == "boundary_layer":
                indicator = _body_fitted_indicator(
                    boundary_layer_indicator_3d(
                        solid,
                        args.reynolds,
                        char_length=2.0 * args.radius,
                    )
                )
            else:  # bl+neq: element-wise max — body-fitted sphere + wake
                bl = _body_fitted_indicator(
                    boundary_layer_indicator_3d(
                        solid,
                        args.reynolds,
                        char_length=2.0 * args.radius,
                    )
                )
                neq = nonequilibrium_indicator_3d(
                    solver.coarse_f,
                    rho,
                    ux,
                    uy,
                    uz,
                )
                indicator = torch.maximum(bl, neq)
            solver.adapt(indicator)
            # Convective fine-grid relaxation time: preserve the physical
            # viscosity across the 2:1 interface (τ_f = 0.5 + ratio·(τ_c−0.5),
            # see static_block_amr.convective_refined_tau).  The solver's
            # internal _tau_for_level halves (τ_c−0.5) instead, which with
            # dt_f = dt_c/2 would run the fine grid at Re_f = 4·Re_c.
            # Collision, FH interface rescale and restriction all read
            # patch.tau, so overriding it here fixes all three.
            for p in solver.patches:
                p.tau = 0.5 + (2.0**p.level) * (tau - 0.5)
            _build_patch_data()
            patch_by_id = {id(p): p for p in solver.patches}
            coverage = _solid_coverage()
            coverages.append(coverage)
            surface_coverage = _surface_coverage()
            surface_coverages.append(surface_coverage)
            bfl_active = sum(
                1 for p in solver.patches if patch_data.get(id(p), {}).get("bfl_mask") is not None
            )
            print(
                f"  adapt@{current_step}: indicator={args.indicator} "
                f"max={float(indicator.max()):.3e} "
                f"refine_cells={int((indicator > args.refine_threshold).sum())} "
                f"patches={len(solver.patches)} "
                f"sphere_covered={coverage * 100.0:.1f}% "
                f"surface_covered={surface_coverage * 100.0:.1f}% "
                f"bfl_patches={bfl_active}",
                flush=True,
            )
        if args.report_interval and current_step % args.report_interval == 0:
            recent = forces[-min(len(forces), args.report_interval) :]
            recent_cv = cv_forces[-min(len(cv_forces), args.report_interval) :]
            recent_bfl = bfl_forces[-min(len(bfl_forces), args.report_interval) :]

            def _cd(seq):
                return sum(seq) / len(seq) / dynamic_area if seq else math.nan

            print(
                f"step={current_step}/{args.steps} recent_Cd(total)="
                f"{_cd(recent):.6f} (cv={_cd(recent_cv):.6f} "
                f"bfl={_cd(recent_bfl):.6f}) "
                f"patches={len(solver.patches)} "
                f"steps/s={current_step / (time.time() - started):.2f}",
                flush=True,
            )
        if not bool(torch.isfinite(solver.coarse_f).all()):
            raise FloatingPointError(f"adaptive sphere diverged at step {current_step}")
        for p in solver.patches:
            if not bool(torch.isfinite(p.f).all()):
                raise FloatingPointError(f"patch diverged at step {current_step}")

    summary = summarize_force_history(
        forces,
        dynamic_area,
        args.reynolds,
        args.statistics_window_steps,
    )
    cd = summary["cd"]
    reference = summary["reference_cd"]
    reference_error = summary["reference_error_pct"]
    mean_force = summary["mean_force_lu"]
    stationarity_dict = summary["stationarity"]
    statistics_window = args.statistics_window_steps or len(forces)
    selected = forces[-statistics_window:]
    selected_cv = cv_forces[-statistics_window:]
    selected_bfl = bfl_forces[-statistics_window:]
    mean_cv_force = sum(selected_cv) / len(selected_cv)
    mean_bfl_force = sum(selected_bfl) / len(selected_bfl)
    cd_cv_only = mean_cv_force / dynamic_area
    cd_bfl_only = mean_bfl_force / dynamic_area
    # Fine-lattice diagnostics: per-fine-step BFL momentum exchange in fine
    # lattice units and the fine dynamic area 0.5·u²·π·(R·ratio)².  The two
    # normalisations are exactly equivalent (force/ratio² / dynamic_area ==
    # force_fine / dynamic_area_fine).
    mean_bfl_force_fine = 0.0
    dynamic_area_fine = dynamic_area
    for pid, force_sum in bfl_accum.items():
        p = patch_by_id[pid]
        mean_bfl_force_fine += force_sum[0] / max(1, bfl_count.get(pid, 1))
        dynamic_area_fine = max(
            dynamic_area_fine,
            0.5 * args.lattice_speed**2 * math.pi * (args.radius * p.ratio) ** 2,
        )
    result = {
        **common_schema_fields("sphere-cellwise-amr-cv-v1"),
        "case": (
            f"cell-wise adaptive AMR sphere Re={args.reynolds}: coarse "
            f"{list(shape)} indicator={args.indicator} "
            f"max_levels={args.max_levels} max_patches={args.max_patches} "
            f"bfl_on_patches={args.bfl_on_patches}"
        ),
        "configuration": {
            "shape_zyx": list(shape),
            "radius": args.radius,
            "reynolds": args.reynolds,
            "lattice_speed": args.lattice_speed,
            "tau": tau,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "ramp_steps": args.ramp_steps,
            "adapt_interval": args.adapt_interval,
            "adapt_warmup": args.adapt_warmup,
            "max_patches": args.max_patches,
            "max_levels": args.max_levels,
            "refine_threshold": args.refine_threshold,
            "coarsen_threshold": args.coarsen_threshold,
            "sponge_width": args.sponge_width,
            "sponge_strength": args.sponge_strength,
            "cv_margin": args.cv_margin,
            "indicator": args.indicator,
            "bfl_on_patches": args.bfl_on_patches,
            "far_field_mode": args.far_field_mode,
        },
        "result": {
            "cd_control_volume": cd,
            "cd_cv_only": cd_cv_only,
            "cd_bfl_only": cd_bfl_only,
            "reference_cd": reference,
            "reference_error_pct": reference_error,
            "mean_force_lu": mean_force,
            "mean_cv_force_lu": mean_cv_force,
            "mean_bfl_force_lu_coarse": mean_bfl_force,
            "mean_bfl_force_lu_fine": mean_bfl_force_fine,
            "dynamic_area_lu2": dynamic_area,
            "dynamic_area_fine_lu2": dynamic_area_fine,
            "stationarity": stationarity_dict,
            "mean_patch_count": (sum(patch_counts) / len(patch_counts) if patch_counts else 0.0),
            "max_patch_count": max(patch_counts, default=0),
            "sphere_solid_covered_fraction": (
                sum(coverages) / len(coverages) if coverages else 0.0
            ),
            "max_sphere_solid_covered_fraction": max(coverages, default=0.0),
            "sphere_surface_covered_fraction": (
                sum(surface_coverages) / len(surface_coverages) if surface_coverages else 0.0
            ),
            "max_sphere_surface_covered_fraction": max(
                surface_coverages,
                default=0.0,
            ),
            "wall_time_s": time.time() - started,
        },
        "artifacts": {"output": args.output},
    }
    write_evidence(result, args.output)
    print(json.dumps(result["result"], indent=2), flush=True)


if __name__ == "__main__":
    main()
