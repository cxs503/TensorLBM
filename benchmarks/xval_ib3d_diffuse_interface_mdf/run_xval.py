"""Cross-validation (XVAL): TensorLBM 3D diffuse-interface MDF  vs  haimingz/vivsim.

Anchor: ``XVAL-LBM-IB3D-DIFFUSE-INTERFACE-MDF``.

The OSS reference is **vendored and executed**, not re-described:
``vendor/vivsim/{ib,ib3d}`` are byte-identical copies of ``haimingz/vivsim``
(MIT, commit ``fb84b12f02e18f3b9dfdf738d6a1a318d81539f9``, sha256 in
``vendor/PROVENANCE.json``) and are imported through the NumPy compatibility
shim in ``jaxshim/`` (JAX has no wheel for this ``loongarch64`` host).

Stages
------
``--stage operator``
    Operator-level cross-validation on identical markers/stencils/fields
    (float64): stencil parity, force/spread parity, MDF slip-residual series,
    partition of unity, area weights, kernel and marker-density sensitivity.
``--stage physics``
    One coupled IB-LBM run: sphere at Re=100 in a periodic Galilean box driven
    by a uniform body force.  ``--impl mdf`` (new operator) or ``--impl legacy``
    (TensorLBM's historical single-pass ``ibm_direct_forcing_3d``).
``--stage summarize``
    Aggregate per-run JSON artifacts into ``artifacts/xval_summary.json`` and the
    figures used by the report.

Every number in the report is read back from these JSON files (no hand-typing).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
ART.mkdir(exist_ok=True)


def _add_paths() -> None:
    sys.path.insert(0, str(HERE / "jaxshim"))
    sys.path.insert(0, str(HERE / "vendor"))
    root = HERE.parents[1]
    sys.path.insert(0, str(root / "src"))


_add_paths()

OSS_COMMIT = "fb84b12f02e18f3b9dfdf738d6a1a318d81539f9"

OSS_KERNEL_NAMES = {
    "peskin4pt": "kernel_peskin_4pt",
    "cosine4pt": "kernel_cosine_4pt",
    "peskin3pt": "kernel_peskin_3pt",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment() -> dict:
    """Recorded so the run is reproducible (design: environment controls)."""
    try:
        cpu = subprocess.run(
            ["bash", "-lc", "lscpu | grep -m1 'Model name' | cut -d: -f2- | xargs"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except Exception:  # pragma: no cover
        cpu = "unknown"
    return {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "cpu": cpu or "unknown",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_threads": torch.get_num_threads(),
        "jax": "absent (loongarch64 wheel unavailable) -> jaxshim numpy backend",
        "oss_repo": "https://github.com/haimingz/vivsim",
        "oss_commit": OSS_COMMIT,
        "oss_license": "MIT (c) 2024 Haiming Zhu",
    }


def icosphere(subdivisions: int, radius: float, center) -> tuple[torch.Tensor, torch.Tensor]:
    """Icosphere identical to the one in ``vivsim/examples/3d/flow_past_sphere.py``."""
    phi = (1 + math.sqrt(5)) / 2
    verts = np.array(
        [
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
        ],
        dtype=np.float64,
    )
    verts /= np.linalg.norm(verts[0])
    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )
    for _ in range(int(subdivisions)):
        edge_midpoint: dict = {}
        new_faces = []
        for tri in faces:
            mids = []
            for i in range(3):
                edge = tuple(sorted((tri[i], tri[(i + 1) % 3])))
                if edge not in edge_midpoint:
                    mid = (verts[edge[0]] + verts[edge[1]]) / 2
                    mid /= np.linalg.norm(mid)
                    edge_midpoint[edge] = len(verts)
                    verts = np.vstack([verts, mid])
                mids.append(edge_midpoint[edge])
            a, b, c = tri
            m0, m1, m2 = mids
            new_faces.extend([[a, m0, m2], [b, m1, m0], [c, m2, m1], [m0, m1, m2]])
        faces = np.array(new_faces, dtype=np.int64)
    verts = verts * radius + np.asarray(center, dtype=np.float64)
    return (
        torch.tensor(verts, dtype=torch.float64),
        torch.tensor(faces, dtype=torch.long),
    )


def analytic_field(coords: torch.Tensor) -> torch.Tensor:
    """Smooth divergence-free-ish test field, ``(N, 3)`` for markers ``(N, 3)``."""
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    ux = 0.40 + 0.10 * torch.sin(0.7 * y) + 0.05 * torch.cos(0.3 * z)
    uy = -0.20 + 0.08 * torch.cos(0.5 * x) - 0.03 * torch.sin(0.2 * z)
    uz = 0.07 * torch.sin(0.6 * x) + 0.04 * torch.cos(0.4 * y)
    return torch.stack([ux, uy, uz], dim=1)


def make_grid_field(shape_zyx: tuple[int, int, int], dtype=torch.float64) -> torch.Tensor:
    """Evaluate :func:`analytic_field` on the ``(nz, ny, nx)`` grid."""
    nz, ny, nx = shape_zyx
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=dtype),
        torch.arange(ny, dtype=dtype),
        torch.arange(nx, dtype=dtype),
        indexing="ij",
    )
    coords = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=1)
    field = analytic_field(coords).reshape(nz, ny, nx, 3)
    return field.permute(3, 0, 1, 2).contiguous()


# --------------------------------------------------------------------------- #
# Stage: operator-level cross-validation
# --------------------------------------------------------------------------- #


def stage_operator(out: Path) -> dict:
    """Compare the torch port against the executed OSS operator, cell by cell."""
    import jax.numpy as jnp

    import vivsim.ib as oss_ib
    import vivsim.ib3d as oss_ib3d

    from tensorlbm import ibm3d_mdf as tl

    torch.set_grad_enabled(False)

    shape_zyx = (24, 28, 32)          # nz, ny, nx
    center = (16.0, 14.0, 12.0)       # x, y, z
    radius = 4.0

    verts, faces = icosphere(3, radius, center)
    ds_tl = tl.surface_weights(verts, faces)
    ds_oss = oss_ib3d.get_ds(jnp.asarray(verts.numpy()), jnp.asarray(faces.numpy()))
    area_analytic = 4.0 * math.pi * radius * radius
    area_poly = float(tl.surface_area(verts, faces))

    grid_tl = make_grid_field(shape_zyx)
    grid_oss = jnp.asarray(grid_tl.permute(0, 3, 2, 1).numpy())  # (3, nx, ny, nz)

    marker_coords = verts
    u_target = torch.zeros_like(marker_coords)

    cases = []
    for kernel in ("peskin4pt", "cosine4pt", "peskin3pt"):
        w_tl, idx_tl = tl.build_stencil(marker_coords, shape_zyx, kernel=kernel)
        w_oss, idx_oss = oss_ib3d.get_ib_stencil(
            jnp.asarray(marker_coords.numpy()),
            tuple(shape_zyx[::-1]),
            kernel=getattr(oss_ib3d, OSS_KERNEL_NAMES[kernel]),
        )
        w_oss_np = np.asarray(w_oss)
        idx_oss_np = np.asarray(idx_oss)
        # decode OSS flat index (x-major) and re-encode in TensorLBM (z-major) order
        nx, ny, nz = shape_zyx[2], shape_zyx[1], shape_zyx[0]
        ix = idx_oss_np // (ny * nz)
        rem = idx_oss_np % (ny * nz)
        iy = rem // nz
        iz = rem % nz
        idx_oss_re = iz * (ny * nx) + iy * nx + ix

        weights_rel = float(
            np.abs(w_oss_np - w_tl.numpy()).max() / max(np.abs(w_oss_np).max(), 1e-300)
        )
        index_match = float((idx_oss_re == idx_tl.numpy()).mean())

        # ---- force chain, parallel iterations for n_iter = 1 .. 8
        residual_curve: dict[str, list[float]] = {}
        force_parity: dict[str, dict] = {}
        for n_iter in (1, 2, 3, 5, 8):
            res_tl = tl.multi_direct_forcing(
                grid_tl, w_tl, idx_tl, u_target, ds_tl, n_iter=n_iter, return_result=True
            )
            res_oss = oss_ib.multi_direct_forcing(
                grid_oss,
                w_oss,
                idx_oss,
                jnp.asarray(u_target.numpy()),
                jnp.asarray(ds_tl.numpy()),
                n_iter=n_iter,
            )
            g_oss = np.asarray(res_oss[0])            # (3, nx, ny, nz)
            r_oss = np.asarray(res_oss[1])            # (N, 3)
            g_oss_zyx = torch.tensor(g_oss).permute(0, 3, 2, 1)  # -> (3, nz, ny, nx)
            scale = max(float(res_tl.grid_force.abs().max()), 1e-300)
            force_parity[f"n_iter={n_iter}"] = {
                "grid_force_rel_max": float((g_oss_zyx - res_tl.grid_force).abs().max() / scale),
                "marker_reaction_rel_max": float(
                    torch.tensor(r_oss).sub(res_tl.marker_reaction).abs().max()
                    / max(float(res_tl.marker_reaction.abs().max()), 1e-300)
                ),
                "grid_force_sum_x_tl": float(res_tl.grid_force[0].sum()),
                "grid_force_sum_x_oss": float(g_oss_zyx[0].sum()),
                "reaction_sum_x_tl": float(res_tl.marker_reaction[:, 0].sum()),
                "reaction_sum_x_oss": float(np.asarray(r_oss)[:, 0].sum()),
                "momentum_closure_rel": float(
                    abs(float(res_tl.grid_force[0].sum()) + float(res_tl.marker_reaction[:, 0].sum()))
                    / max(abs(float(res_tl.marker_reaction[:, 0].sum())), 1e-300)
                ),
            }
            residual_curve[f"n_iter={n_iter}"] = res_tl.residuals

        # ---- full residual series 1..10 for the two recipes
        res_series_tl = tl.mdf_residuals(grid_tl, w_tl, idx_tl, u_target, ds_tl, max_iter=10)
        res_series_oss = tl.mdf_residuals(grid_tl, w_tl, idx_tl, u_target, ds_tl, max_iter=10)
        res_series_accum = tl.multi_direct_forcing(
            grid_tl, w_tl, idx_tl, u_target, ds_tl, n_iter=10, accumulate=True, return_result=True
        ).residuals

        cases.append(
            {
                "kernel": kernel,
                "n_markers": int(verts.shape[0]),
                "stencil_points": int(w_tl.shape[1]),
                "stencil_weight_rel_diff": weights_rel,
                "stencil_index_match_fraction": index_match,
                "partition_of_unity_min": float(
                    torch.as_tensor(w_oss_np).sum(dim=1).min()
                ),
                "partition_of_unity_max": float(torch.as_tensor(w_oss_np).sum(dim=1).max()),
                "force_parity": force_parity,
                "residual_series_tl": res_series_tl,
                "residual_series_oss_recipe_via_tl": res_series_oss,
                "residual_series_accumulated_recipe": res_series_accum,
            }
        )

    # ---- ds / area weighting
    area_check = {
        "sum_ds_tl": float(ds_tl.sum()),
        "sum_ds_oss": float(np.asarray(ds_oss).sum()),
        "ds_rel_diff": float(
            np.abs(np.asarray(ds_oss) - ds_tl.numpy()).max() / max(np.abs(np.asarray(ds_oss)).max(), 1e-300)
        ),
        "analytic_sphere_area": area_analytic,
        "polyhedral_area": area_poly,
        "polyhedral_vs_analytic_rel": (area_poly - area_analytic) / area_analytic,
    }

    # ---- marker-density sensitivity (convergence depends on ds / cell area)
    density = []
    for subdiv in (2, 3, 4):
        v, f = icosphere(subdiv, radius, center)
        dsv = tl.surface_weights(v, f)
        w, idx = tl.build_stencil(v, shape_zyx, kernel="peskin4pt")
        res = tl.multi_direct_forcing(
            grid_tl, w, idx, torch.zeros((v.shape[0], 3), dtype=torch.float64), dsv,
            n_iter=5, return_result=True,
        )
        density.append(
            {
                "subdivisions": subdiv,
                "n_markers": int(v.shape[0]),
                "mean_ds": float(dsv.mean()),
                "mean_markers_per_cell_face": float(v.shape[0] / area_poly),
                "slip_residual_n_iter_5": res.residuals[-1],
                "slip_residual_n_iter_1": res.residuals[0],
            }
        )

    # ---- single-pass legacy operator (TensorLBM historical) parity gap
    from tensorlbm.ibm_vec import ibm_direct_forcing_3d_vec

    legacy = {}
    for kernel_name in ("hat", "4pt"):
        zero_t = torch.zeros((), dtype=torch.float64)
        fx, fy, fz = ibm_direct_forcing_3d_vec(
            grid_tl[0], grid_tl[1], grid_tl[2],
            marker_coords[:, 0], marker_coords[:, 1], marker_coords[:, 2],
            zero_t, zero_t, zero_t,
            kernel=kernel_name,
        )
        legacy_force = torch.stack([fx, fy, fz])
        ref = tl.multi_direct_forcing(
            grid_tl, tl.build_stencil(marker_coords, shape_zyx, kernel="peskin4pt")[0],
            tl.build_stencil(marker_coords, shape_zyx, kernel="peskin4pt")[1],
            u_target, ds_tl, n_iter=5, return_result=True,
        )
        # legacy marker-force definition vs area-weighted: F = 2·ds·Δu
        legacy[kernel_name] = {
            "grid_force_sum_x": float(legacy_force[0].sum()),
            "physical_force_estimate_x": 2.0 * float(
                (legacy_force[0] * ds_tl.reshape(-1).mean()).sum()
            ),
            "mdf_grid_force_sum_x": float(ref.grid_force[0].sum()),
            "legacy_vs_mdf_magnitude_ratio": float(
                legacy_force.abs().max() / max(float(ref.grid_force.abs().max()), 1e-300)
            ),
            "note": "legacy = single pass, no ds weighting, no factor 2 (F=2 rho A du)",
        }

    payload = {
        "stage": "operator",
        "environment": environment(),
        "setup": {
            "grid_shape_zyx": list(shape_zyx),
            "sphere_center_xyz": list(center),
            "sphere_radius": radius,
            "icosphere_subdivisions": 3,
            "dtype": "float64",
        },
        "area_check": area_check,
        "cases": cases,
        "marker_density": density,
        "legacy_operator": legacy,
        "artifacts": {
            "vendored_oss_files": {
                str(p.relative_to(HERE / "vendor")): sha256(p)
                for p in sorted((HERE / "vendor" / "vivsim").rglob("*.py"))
            }
        },
    }
    out.write_text(json.dumps(payload, indent=2, default=float))
    return payload


# --------------------------------------------------------------------------- #
# Stage: physics — coupled IB-LBM, sphere at Re=100
# --------------------------------------------------------------------------- #


def cd_schiller_naumann(re: float) -> float:
    return 24.0 / re * (1.0 + 0.15 * re ** 0.687)


CD_LITERATURE = {
    # flow past a sphere, free stream; values as compiled in Clift, Grace & Weber
    # (1978) / Johnson & Patel (1999) / Dennis & Walker (1971)
    "schiller_naumann": cd_schiller_naumann,
}


def stage_physics(args) -> dict:
    import vivsim.ib as oss_ib      # noqa: F401  (import proves the OSS side is runnable)
    import jax.numpy as jnp

    from tensorlbm import ibm3d_mdf as tl
    from tensorlbm.ibm_sphere_surface_force import _lbm_step
    from tensorlbm.solver3d import equilibrium3d, macroscopic3d, stream3d

    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    D = args.diameter
    nx = int(round(args.streamwise * D))
    ny = int(round(args.lateral * D))
    nz = ny
    shape = (nz, ny, nx)
    center = (nx / 2.0, ny / 2.0, nz / 2.0)
    U = args.speed
    nu = U * D / args.re
    tau = 0.5 + 3.0 * nu
    area = 0.25 * math.pi * D * D

    # geometry: same icosphere family as the OSS example, area-weighted markers
    verts, faces = icosphere(args.subdivisions, D / 2.0, center)
    verts = verts.to(dtype)          # keep the whole run in one precision
    ds = tl.surface_weights(verts, faces)
    if args.impl == "mdf":
        weights, indices = tl.build_stencil(verts, shape, kernel=args.kernel)

    # geometric fluid mask (drive force is applied on fluid cells only)
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=dtype),
        torch.arange(ny, dtype=dtype),
        torch.arange(nx, dtype=dtype),
        indexing="ij",
    )
    rr = torch.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2)
    fluid = (rr > D / 2.0).to(dtype)
    n_fluid = float(fluid.sum())

    rho0 = torch.ones(shape, dtype=dtype)
    ux0 = U * fluid
    zeros = torch.zeros(shape, dtype=dtype)
    f = equilibrium3d(rho0, ux0, zeros, zeros)
    u_target = torch.zeros((verts.shape[0], 3), dtype=dtype)

    g = 0.5 * U * U * area * cd_schiller_naumann(args.re) / n_fluid  # initial drive guess
    history: list[dict] = []
    t0 = time.time()

    for step in range(1, args.max_steps + 1):
        rho, ux, uy, uz = macroscopic3d(f)
        if args.impl == "mdf":
            grid_u = torch.stack([ux, uy, uz])
            res = tl.multi_direct_forcing(
                grid_u,
                weights,
                indices,
                u_target,
                ds,
                n_iter=args.n_iter,
                accumulate=args.accumulate,
                return_result=True,
            )
            ib_force = res.grid_force
            force_body_x = float(res.marker_reaction[:, 0].sum())
            slip = res.residuals[-1]
        else:
            from tensorlbm.ibm_vec import ibm_direct_forcing_3d_vec

            zero_t = torch.zeros((), dtype=dtype)
            fx, fy, fz = ibm_direct_forcing_3d_vec(
                ux, uy, uz, verts[:, 0], verts[:, 1], verts[:, 2],
                zero_t, zero_t, zero_t, kernel=args.kernel_legacy
            )
            ib_force = torch.stack([fx, fy, fz])
            # legacy grid force == Δu spread; the physical force is 2·Σ ds·Δu,
            # approximated here by the momentum in the grid force field
            force_body_x = 2.0 * float(ib_force[0].sum())
            slip = float("nan")

        drive = torch.zeros_like(ib_force)
        drive[0] = (g * fluid).to(dtype)
        force = ib_force + drive

        f = _lbm_step(
            f,
            force,
            tau,
            stream=stream3d,
            collision=args.collision,
            guo=True,
            moments=(rho, ux, uy, uz),
        )

        if step % args.report_interval == 0:
            rho, ux, uy, uz = macroscopic3d(f)
            u_fluid = float((ux * fluid).sum() / fluid.sum())
            drive_force = g * float(rho.sum() * 0.0) + g * float((rho * fluid).sum())
            cd_marker = tl.drag_coefficient(
                torch.tensor([force_body_x, 0.0, 0.0]), 1.0, abs(u_fluid), D
            )
            cd_drive = tl.drag_coefficient(
                torch.tensor([drive_force, 0.0, 0.0]), 1.0, abs(u_fluid), D
            )
            history.append(
                {
                    "step": step,
                    "cd_marker": cd_marker,
                    "cd_drive": cd_drive,
                    "u_mean": u_fluid,
                    "re_measured": abs(u_fluid) * D / nu,
                    "g": g,
                    "drive_force": drive_force,
                    "slip_residual": slip,
                    "mass": float(f.sum()),
                    "elapsed_s": time.time() - t0,
                }
            )
            if args.verbose:
                print(
                    f"[{args.tag}] step {step:5d} Re={history[-1]['re_measured']:7.2f} "
                    f"Cd_marker={cd_marker:7.4f} Cd_drive={cd_drive:7.4f} "
                    f"u={u_fluid:.4f} t={history[-1]['elapsed_s']:.0f}s",
                    flush=True,
                )
            if u_fluid > 0:
                ratio = U / u_fluid
                g *= min(max(1.0 + args.gain * (ratio * ratio - 1.0), 0.5), 1.5)

    # ---- statistics over the second half (predeclared: last 50% of history)
    half = history[len(history) // 2 :] or history
    cd_drive_mean = sum(h["cd_drive"] for h in half) / len(half)
    cd_marker_mean = sum(h["cd_marker"] for h in half) / len(half)
    re_mean = sum(h["re_measured"] for h in half) / len(half)
    ref_sn = cd_schiller_naumann(re_mean)

    payload = {
        "stage": "physics",
        "tag": args.tag,
        "environment": environment(),
        "parameters": {
            "impl": args.impl,
            "n_iter": args.n_iter if args.impl == "mdf" else 1,
            "accumulate": bool(args.accumulate),
            "kernel": args.kernel if args.impl == "mdf" else args.kernel_legacy,
            "collision": args.collision,
            "diameter": D,
            "grid_zyx": list(shape),
            "domain_streamwise_D": args.streamwise,
            "domain_lateral_D": args.lateral,
            "blockage": D / args.lateral,
            "reynolds_target": args.re,
            "speed": U,
            "nu": nu,
            "tau": tau,
            "mach": U / math.sqrt(1.0 / 3.0),
            "n_markers": int(verts.shape[0]),
            "icosphere_subdivisions": args.subdivisions,
            "dtype": args.dtype,
            "max_steps": args.max_steps,
            "report_interval": args.report_interval,
        },
        "results": {
            "cd_drive_mean": cd_drive_mean,
            "cd_marker_mean": cd_marker_mean,
            "re_measured": re_mean,
            "cd_reference_schiller_naumann": ref_sn,
            "error_drive_vs_ref": (cd_drive_mean - ref_sn) / ref_sn,
            "error_marker_vs_ref": (cd_marker_mean - ref_sn) / ref_sn,
            "marker_vs_drive_rel": (cd_marker_mean - cd_drive_mean) / cd_drive_mean,
            "mass_drift_rel": abs(history[-1]["mass"] - history[0]["mass"]) / history[0]["mass"],
            "wall_time_s": history[-1]["elapsed_s"] if history else float("nan"),
            "s_per_step": (history[-1]["elapsed_s"] / history[-1]["step"]) if history else float("nan"),
        },
        "history": history,
    }
    out = ART / f"physics_{args.tag}.json"
    out.write_text(json.dumps(payload, indent=2, default=float))
    print(f"[{args.tag}] Cd_drive={cd_drive_mean:.4f} Cd_marker={cd_marker_mean:.4f} "
          f"ref={ref_sn:.4f} err_drive={payload['results']['error_drive_vs_ref']*100:+.2f}%", flush=True)
    return payload


# --------------------------------------------------------------------------- #
# Stage: summarize
# --------------------------------------------------------------------------- #


def stage_summarize(out: Path) -> dict:
    physics = {}
    for path in sorted(ART.glob("physics_*.json")):
        data = json.loads(path.read_text())
        physics[data["tag"]] = {k: data[k] for k in ("parameters", "results")}

    operator_path = ART / "operator_xval.json"
    operator = json.loads(operator_path.read_text()) if operator_path.exists() else None

    summary = {"stage": "summarize", "environment": environment(), "physics": physics}
    if operator is not None:
        summary["operator"] = {
            "area_check": operator["area_check"],
            "stencil_weight_rel_diff_max": max(
                c["stencil_weight_rel_diff"] for c in operator["cases"]
            ),
            "stencil_index_match_min": min(
                c["stencil_index_match_fraction"] for c in operator["cases"]
            ),
            "force_parity_rel_max": max(
                max(v["grid_force_rel_max"] for v in c["force_parity"].values())
                for c in operator["cases"]
            ),
            "reaction_parity_rel_max": max(
                max(v["marker_reaction_rel_max"] for v in c["force_parity"].values())
                for c in operator["cases"]
            ),
        }
    out.write_text(json.dumps(summary, indent=2, default=float))
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("operator", "physics", "summarize"), required=True)
    parser.add_argument("--impl", choices=("mdf", "legacy"), default="mdf")
    parser.add_argument("--n-iter", type=int, default=5)
    parser.add_argument("--accumulate", action="store_true")
    parser.add_argument("--kernel", default="peskin4pt")
    parser.add_argument("--kernel-legacy", default="4pt")
    parser.add_argument("--collision", default="bgk")
    parser.add_argument("--diameter", type=float, default=8.0)
    parser.add_argument("--re", type=float, default=100.0)
    parser.add_argument("--speed", type=float, default=0.1)
    parser.add_argument("--lateral", type=float, default=6.0)
    parser.add_argument("--streamwise", type=float, default=8.0)
    parser.add_argument("--subdivisions", type=int, default=3)
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--report-interval", type=int, default=50)
    parser.add_argument("--gain", type=float, default=0.7)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--tag", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.stage == "operator":
        payload = stage_operator(ART / "operator_xval.json")
        print(json.dumps({k: payload[k] for k in ("area_check",)}, indent=2, default=float))
        for case in payload["cases"]:
            print(
                f"{case['kernel']:>10}: weight_rel_diff={case['stencil_weight_rel_diff']:.3e} "
                f"index_match={case['stencil_index_match_fraction']:.3f} "
                f"force_rel_max={max(v['grid_force_rel_max'] for v in case['force_parity'].values()):.3e}"
            )
        return 0

    if args.stage == "physics":
        if not args.tag:
            args.tag = (
                f"{args.impl}_iter{args.n_iter}_{args.kernel}_D{args.diameter:.0f}"
                f"_Re{args.re:.0f}_L{args.lateral:.0f}x{args.streamwise:.0f}"
            )
        stage_physics(args)
        return 0

    stage_summarize(ART / "xval_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())