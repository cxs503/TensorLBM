"""XVAL R2 — Peskin tensor-product delta stencil + multi-direct forcing (MDF).

Round 2 of the cross-validation of ``haimingz/vivsim``'s
``peskin-tensor-product-stencil-mdf`` recipe as implemented in
``tensorlbm.ibm3d_mdf`` (feature ``XVAL-LBM-PESKIN-TENSOR-PRODUCT-STENCIL-MDF``).

Round 1 (``benchmarks/xval_ib3d_diffuse_interface_mdf``) established bit-level
parity of the operator with the OSS source, but its physics verdict was a
30.6 % drag error — outside the 15 % acceptance band.  Round 2 asks a
different question: **is that error a defect of the operator, or of the
validation run?**

Stages
------
``operator``  analytic identities of the tensor-product stencil (partition of
              unity, first/second lattice moments, linear-field reproduction,
              interpolation order, spread/interp adjointness, exact momentum
              conservation, MDF contraction).  Every expected value is a
              closed-form number.

``physics``   flow past a fixed sphere in a **fully periodic** box, driven by a
              constant body force on the fluid cells only.  With a periodic
              domain there is no wall flux, so at a *steady* state the momentum
              balance is exact:
                  F_drag = sum_cells g rho_fluid        (drive measure)
                  F_drag = -sum_markers F_m             (IBM / marker measure)
              Agreement of the two independent drag measures is the
              steady-state / momentum-closure gate.

``summarize``  box-size extrapolation Cd(phi) -> Cd_inf against the dilute
              periodic-array (Hasimoto) picture, plus the acceptance verdict
              against Schiller-Naumann / Clift at the measured Re.

Usage
-----
    python run_xval.py --stage operator
    python run_xval.py --stage physics --box 6 --diameter 8 --re 100 \
        --max-steps 12000 --threads 32 --tag box6_D8
    python run_xval.py --stage summarize
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

torch.set_grad_enabled(False)


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #

def git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - provenance only
        return "unknown"


def environment() -> dict:
    import tensorlbm

    return {
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "tensorlbm": str(getattr(tensorlbm, "__version__", "unknown")),
        "git_head": git_head(),
        "oss_repo": "https://github.com/haimingz/vivsim",
        "oss_commit": "fb84b12f02e18f3b9dfdf738d6a1a318d81539f9",
        "run_id": os.environ.get("XV_RUN_ID", ""),
    }


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #

def icosphere(subdivisions: int, radius: float, center) -> tuple[torch.Tensor, torch.Tensor]:
    """Subdivided icosahedron: quasi-uniform triangulation of a sphere.

    Corners are duplicated per triangle; ``surface_weights`` lumps each
    triangle's area into its three corners, so the duplicated corners receive
    identical weights and the marker set stays exactly area-consistent.
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    base = torch.tensor(
        [
            [-1.0, phi, 0.0], [1.0, phi, 0.0], [-1.0, -phi, 0.0], [1.0, -phi, 0.0],
            [0.0, -1.0, phi], [0.0, 1.0, phi], [0.0, -1.0, -phi], [0.0, 1.0, -phi],
            [phi, 0.0, -1.0], [phi, 0.0, 1.0], [-phi, 0.0, -1.0], [-phi, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    base = base / base.norm(dim=1, keepdim=True)
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]
    verts = [base[i].clone() for i in range(len(base))]
    tri = [list(f) for f in faces]
    for _ in range(subdivisions):
        cache: dict[tuple[int, int], int] = {}
        new_tri = []

        def midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in cache:
                mid = (verts[a] + verts[b]) / 2.0
                verts.append(mid / mid.norm())
                cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in tri:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_tri += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        tri = new_tri
    # midpoint cache keeps every vertex unique: subdiv 0,1,2,3 -> 12, 42, 162, 642
    v = torch.stack(verts) * radius + torch.tensor(list(center), dtype=torch.float64)
    return v, torch.tensor(tri, dtype=torch.long)


# --------------------------------------------------------------------------- #
# analytic reference values
# --------------------------------------------------------------------------- #

def cd_schiller_naumann(re: float) -> float:
    """Schiller-Naumann: Cd = 24/Re (1 + 0.15 Re^0.687), Re <= 800."""
    return 24.0 / re * (1.0 + 0.15 * re ** 0.687)


def cd_clift(re: float) -> float:
    """Clift-Grace-Weber (1978) rigid-sphere correlation."""
    re = min(max(re, 1e-6), 800.0)
    if re <= 20.0:
        return 24.0 / re * (1.0 + 0.1935 * re ** 0.6305)
    return 10 ** (1.6435 - 1.1242 * math.log10(re) + 0.1558 * math.log10(re) ** 2)


def hasimoto_volume_fraction(d_over_h: float) -> float:
    """Sphere volume fraction of a simple cubic array, phi = (pi/6)(D/H)^3."""
    return math.pi / 6.0 * d_over_h ** 3


def hasimoto_correction(phi: float) -> float:
    """Hasimoto (1959) dilute simple-cubic-array drag / isolated-sphere drag.

    F/F0 = 1 / (1 - 1.7601 phi^(1/3) + phi - 1.5593 phi^2).
    Stokes-limit result; used here only as the *scaling* of the box-size
    extrapolation (leading term ~ phi^(1/3) ~ D/H), never as a quantitative
    prediction at Re = 100.
    """
    c = phi ** (1.0 / 3.0)
    return 1.0 / (1.0 - 1.7601 * c + phi - 1.5593 * phi * phi)


def continued_moment(kernel_fn, order: int, lo: float = -2.5, hi: float = 2.5) -> float:
    """Integral of r^order * phi(r) dr by composite Simpson quadrature."""
    n = 200001
    h = (hi - lo) / (n - 1)
    xs = torch.linspace(lo, hi, n, dtype=torch.float64)
    ys = kernel_fn(xs) * xs.abs() ** order
    w = torch.full((n,), 2.0, dtype=torch.float64)
    w[0] = w[-1] = 1.0
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return float((ys * w).sum() * h / 3.0)


# --------------------------------------------------------------------------- #
# stage: operator
# --------------------------------------------------------------------------- #

def _random_markers(n: int, shape, dtype, gen) -> torch.Tensor:
    nz, ny, nx = shape
    lo = torch.tensor([3.0, 3.0, 3.0], dtype=dtype)
    hi = torch.tensor([nx - 3.0, ny - 3.0, nz - 3.0], dtype=dtype)
    u = torch.rand((n, 3), generator=gen, dtype=dtype)
    return lo + u * (hi - lo)


def stage_operator() -> dict:
    from tensorlbm import ibm3d_mdf as mdf

    dtype = torch.float64
    gen = torch.Generator().manual_seed(20261115)
    shape = (18, 20, 22)
    nz, ny, nx = shape
    n = 512
    markers = _random_markers(n, shape, dtype, gen)

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=dtype),
        torch.arange(ny, dtype=dtype),
        torch.arange(nx, dtype=dtype),
        indexing="ij",
    )
    kx = 2.0 * math.pi / nx
    ky = 2.0 * math.pi / ny
    u_analytic = torch.stack(
        [
            torch.sin(kx * xx + 0.3) * torch.cos(ky * yy),
            torch.cos(kx * xx) * torch.sin(ky * yy + 0.7),
            0.01 * zz - 0.2,
        ]
    )

    cases = []
    for name in ("peskin4pt", "peskin3pt", "cosine4pt"):
        kfn, radius = mdf._resolve_kernel(name)
        w, idx = mdf.build_stencil(markers, shape, kernel=name)
        iz = idx // (ny * nx)
        iy = (idx % (ny * nx)) // nx
        ix = idx % nx
        dx = ix.to(dtype) - markers[:, 0:1]
        dy = iy.to(dtype) - markers[:, 1:2]
        dz = iz.to(dtype) - markers[:, 2:3]

        sum_w = w.sum(dim=1)
        m1 = max(
            float((w * dx).sum(dim=1).abs().max()),
            float((w * dy).sum(dim=1).abs().max()),
            float((w * dz).sum(dim=1).abs().max()),
        )
        m2x = (w * dx * dx).sum(dim=1)
        m2y = (w * dy * dy).sum(dim=1)
        m2z = (w * dz * dz).sum(dim=1)
        mxy = (w * dx * dy).sum(dim=1)
        m2_ref = continued_moment(kfn, 2)

        lin = torch.stack(
            [
                1.0 + 0.5 * xx + 2.0 * yy - 0.25 * zz,
                -3.0 + 0.7 * xx - 1.1 * yy + 0.9 * zz,
                0.25 * xx + 0.3 * yy + 4.0,
            ]
        )
        lin_m = mdf.interpolate(lin, w, idx)
        lin_ref = torch.stack(
            [
                1.0 + 0.5 * markers[:, 0] + 2.0 * markers[:, 1] - 0.25 * markers[:, 2],
                -3.0 + 0.7 * markers[:, 0] - 1.1 * markers[:, 1] + 0.9 * markers[:, 2],
                0.25 * markers[:, 0] + 0.3 * markers[:, 1] + 4.0,
            ],
            dim=1,
        )
        lin_err = float((lin_m - lin_ref).abs().max() / lin_ref.abs().max())

        u_m = mdf.interpolate(u_analytic, w, idx)
        u_ref = torch.stack(
            [
                torch.sin(kx * markers[:, 0] + 0.3) * torch.cos(ky * markers[:, 1]),
                torch.cos(kx * markers[:, 0]) * torch.sin(ky * markers[:, 1] + 0.7),
                0.01 * markers[:, 2] - 0.2,
            ],
            dim=1,
        )
        smooth_err = float((u_m - u_ref).abs().max() / u_ref.abs().max())

        q = torch.randn((n, 3), generator=gen, dtype=dtype)
        spread_q = mdf.spread(q, w, idx, shape)
        lhs = float((spread_q * u_analytic).sum())
        rhs = float((q * u_m).sum())
        adj_rel = abs(lhs - rhs) / max(abs(lhs), 1e-30)
        mom_rel = abs(float(spread_q.sum()) - float(q.sum())) / max(float(q.sum().abs()), 1e-30)

        cases.append(
            {
                "kernel": name,
                "stencil_points": int((2 * radius) ** 3),
                "partition_of_unity_max_abs_err": float((sum_w - 1.0).abs().max()),
                "first_moment_max_abs": m1,
                "second_moment_continuous_ref": m2_ref,
                "second_moment_x_mean": float(m2x.mean()),
                "second_moment_translation_spread": float(m2x.max() - m2x.min()),
                "second_moment_rel_err_vs_continuous": float((m2x.mean() - m2_ref) / m2_ref),
                "second_moment_y_abs_err_vs_x": float((m2y - m2x).abs().max()),
                "second_moment_z_abs_err_vs_x": float((m2z - m2x).abs().max()),
                "offdiag_xy_max_abs": float(mxy.abs().max()),
                "linear_field_reproduction_rel_err": lin_err,
                "smooth_field_interp_rel_err": smooth_err,
                "spread_interp_adjointness_rel_err": adj_rel,
                "momentum_conservation_rel_err": mom_rel,
            }
        )

    # interpolation order at fixed physical wavenumber
    order_rows = []
    for nref in (8, 16, 32, 48, 64):
        shp = (nref, nref, nref)
        mk = _random_markers(256, shp, dtype, gen)
        wr, ir = mdf.build_stencil(mk, shp, kernel="peskin4pt")
        kk = 2.0 * math.pi  # one wave over the unit domain, fixed physics
        h = 1.0 / nref
        # lattice point i sits at physical x = i*h (IBM stencil convention)
        fld = torch.sin(kk * torch.arange(nref, dtype=dtype) * h)
        fld = fld.reshape(1, 1, nref).expand(shp).contiguous()
        got = mdf.interpolate(fld, wr, ir)
        ref = torch.sin(kk * mk[:, 0] * h)
        order_rows.append({"n": nref, "max_abs_err": float((got - ref).abs().max())})
    errs = [r["max_abs_err"] for r in order_rows]
    order_p = math.log(errs[0] / errs[-1]) / math.log(order_rows[-1]["n"] / order_rows[0]["n"])
    pair_orders = [
        math.log(errs[i] / errs[i + 1]) / math.log(order_rows[i + 1]["n"] / order_rows[i]["n"])
        for i in range(len(errs) - 1)
    ]

    # MDF contraction and closure
    from tensorlbm.ibm3d_mdf import multi_direct_forcing

    verts, faces = icosphere(2, 3.0, (nx / 2.0, ny / 2.0, nz / 2.0))
    ds = mdf.surface_weights(verts, faces)
    ws, isx = mdf.build_stencil(verts.to(dtype), shape, kernel="peskin4pt")
    grid_u = torch.stack(
        [0.01 * xx + 0.0 * yy, 0.02 * torch.sin(kx * xx).expand(nz, ny, nx).clone(), 0.0 * zz]
    )
    u_target = torch.zeros((verts.shape[0], 3), dtype=dtype)
    mdf_series = {}
    for n_iter in (1, 2, 3, 5, 8, 12):
        res = multi_direct_forcing(grid_u, ws, isx, u_target, ds, n_iter=n_iter)
        gf = float(res.grid_force[0].sum())
        mr = float(res.marker_reaction[:, 0].sum())
        mdf_series[n_iter] = {
            "residual_series": res.residuals,
            "final_slip_rel": res.residuals[-1],
            "grid_force_sum_x": gf,
            "marker_reaction_sum_x": mr,
            "closure_rel": abs(gf + mr) / max(abs(mr), 1e-30),
        }
    contraction = []
    for seq in (v["residual_series"] for v in mdf_series.values()):
        contraction += [seq[i + 1] / seq[i] for i in range(len(seq) - 1) if seq[i] > 0]

    payload = {
        "stage": "operator",
        "environment": environment(),
        "setup": {
            "grid_zyx": list(shape),
            "n_markers": n,
            "sphere_radius": 3.0,
            "sphere_markers": int(verts.shape[0]),
            "dtype": "float64",
        },
        "cases": cases,
        "interpolation_order": {
            "rows": order_rows,
            "observed_order": order_p,
            "pair_orders": pair_orders,
            "asymptotic_order": pair_orders[-1],
        },
        "mdf": {"series": mdf_series, "max_contraction_ratio": max(contraction)},
    }
    (ART / "operator_r2.json").write_text(json.dumps(payload, indent=2, default=float))
    return payload


# --------------------------------------------------------------------------- #
# stage: physics
# --------------------------------------------------------------------------- #

def stage_physics(args) -> dict:
    import numpy as np

    from tensorlbm import ibm3d_mdf as tl
    from tensorlbm.ibm_sphere_surface_force import _lbm_step
    from tensorlbm.solver3d import equilibrium3d, macroscopic3d, stream3d

    if args.threads:
        torch.set_num_threads(args.threads)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    torch.manual_seed(7)

    D = args.diameter
    H = int(round(args.box * D))
    nz = H
    ny = H
    nx = H if args.streamwise is None else int(round(args.streamwise * D))
    shape = (nz, ny, nx)
    center = (nx / 2.0, ny / 2.0, nz / 2.0)
    U = args.speed
    nu = U * D / args.re
    tau = 0.5 + 3.0 * nu
    area = 0.25 * math.pi * D * D
    d_over_h = D / H
    phi = hasimoto_volume_fraction(d_over_h)

    verts, faces = icosphere(args.subdivisions, D / 2.0, center)
    verts = verts.to(dtype)
    ds = tl.surface_weights(verts, faces)
    w, idx = tl.build_stencil(verts, shape, kernel=args.kernel)
    u_target = torch.zeros((verts.shape[0], 3), dtype=dtype)

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=dtype),
        torch.arange(ny, dtype=dtype),
        torch.arange(nx, dtype=dtype),
        indexing="ij",
    )
    rr = torch.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2)
    fluid = (rr > D / 2.0).to(dtype)
    n_fluid = float(fluid.sum())

    zeros = torch.zeros(shape, dtype=dtype)
    f = equilibrium3d(torch.ones(shape, dtype=dtype), U * fluid, zeros, zeros)

    # start already at the steady body force implied by the box-corrected
    # Schiller-Naumann estimate, so the momentum hold-up transient is short
    g = 0.5 * U * U * area * cd_schiller_naumann(args.re) * hasimoto_correction(phi) / n_fluid
    history: list[dict] = []
    windows: list[dict] = []
    win_acc: list[dict] = []
    t0 = time.time()
    g_frozen = None
    converged_at = None

    for step in range(1, args.max_steps + 1):
        rho, ux, uy, uz = macroscopic3d(f)
        res = tl.multi_direct_forcing(
            torch.stack([ux, uy, uz]), w, idx, u_target, ds, n_iter=args.n_iter
        )
        drive = torch.zeros_like(res.grid_force)
        drive[0] = g * rho * fluid
        f = _lbm_step(
            f, res.grid_force + drive, tau,
            stream=stream3d, collision="bgk", guo=True, moments=(rho, ux, uy, uz),
        )

        if step % args.report_interval == 0:
            rho, ux, uy, uz = macroscopic3d(f)
            rho_flu = float((rho * fluid).sum())
            u_flux = float((rho * ux * fluid).sum()) / rho_flu
            drive_force = g * rho_flu
            ib_drag = float(res.marker_reaction[:, 0].sum())
            cd_drive = tl.drag_coefficient(
                torch.tensor([drive_force, 0.0, 0.0], dtype=dtype), 1.0, abs(u_flux), D
            )
            cd_marker = tl.drag_coefficient(
                torch.tensor([ib_drag, 0.0, 0.0], dtype=dtype), 1.0, abs(u_flux), D
            )
            u_grid = torch.stack([ux, uy, uz])
            slip_vec = u_target - tl.interpolate(u_grid, w, idx)
            slip_rms = float(torch.sqrt((slip_vec ** 2).sum(dim=1).mean())) / max(abs(u_flux), 1e-30)
            leakage = float(
                (torch.sqrt(ux ** 2 + uy ** 2 + uz ** 2) * (1.0 - fluid)).gt(0.1 * abs(u_flux)).sum()
            ) / max(float((1.0 - fluid).sum()), 1.0)
            rec = {
                "step": step,
                "cd_drive": cd_drive,
                "cd_marker": cd_marker,
                "slip_rms_rel": slip_rms,
                "leak_cells_rel": leakage,
                "ib_force_sum_x": ib_drag,
                "drive_force_sum_x": drive_force,
                "momentum_x": float((rho * ux).sum()),
                "u_flux": u_flux,
                "re_measured": abs(u_flux) * D / nu,
                "g": g,
                "slip_rel": float(res.residuals[-1]),
                "closure_rel": (cd_marker - cd_drive) / cd_drive,
                "mass_drift_rel": abs(float(f.sum()) - n_fluid) / n_fluid,
                "elapsed_s": time.time() - t0,
            }
            history.append(rec)
            win_acc.append(rec)

            if g_frozen is None and step < args.warmup:
                if u_flux > 0:
                    ratio = U / u_flux
                    g *= min(max(1.0 + args.gain * (ratio * ratio - 1.0), 0.5), 1.5)
            elif g_frozen is None:
                g_frozen = g
                if args.verbose:
                    print(f"[{args.tag}] frozen g={g:.6e} at step {step}", flush=True)

            if g_frozen is not None and step % (args.report_interval * args.window) == 0:
                recent = win_acc[-args.window:]
                prior = win_acc[-2 * args.window: -args.window]
                if len(prior) == args.window and len(recent) == args.window:
                    m_recent = sum(r["cd_drive"] for r in recent) / len(recent)
                    m_prior = sum(r["cd_drive"] for r in prior) / len(prior)
                    drift = abs(m_recent - m_prior) / max(abs(m_recent), 1e-12)
                    windows.append({"step": step, "cd_drive_mean": m_recent, "drift": drift})
                    if args.verbose:
                        print(
                            f"[{args.tag}] step {step} Cd_drive={m_recent:.4f} "
                            f"drift={drift*100:.2f}% Cd_marker={cd_marker:.4f} "
                            f"slip={rec['slip_rel']:.3f} t={rec['elapsed_s']:.0f}s",
                            flush=True,
                        )
                    if drift < args.steady_tol and step >= args.min_steps:
                        converged_at = step
                        break
            if step % 500 == 0 and args.verbose:
                print(
                    f"[{args.tag}] step {step} Re={rec['re_measured']:.2f} "
                    f"Cd_drive={cd_drive:.4f} Cd_marker={cd_marker:.4f} "
                    f"t={rec['elapsed_s']:.0f}s", flush=True,
                )

    cut = (converged_at or history[-1]["step"]) - args.avg_steps
    tail = [r for r in history if r["step"] > cut] or history[-1:]
    mean = lambda k: sum(r[k] for r in tail) / len(tail)
    std = lambda k: (sum((r[k] - mean(k)) ** 2 for r in tail) / max(len(tail) - 1, 1)) ** 0.5

    re_mean = mean("re_measured")
    cd_drive_mean = mean("cd_drive")
    cd_marker_mean = mean("cd_marker")
    ref_sn = cd_schiller_naumann(re_mean)
    ref_cw = cd_clift(re_mean)
    cd_pred = ref_sn * hasimoto_correction(phi)

    rho, ux, uy, uz = macroscopic3d(f)
    j = ny // 2
    np.savez(
        ART / f"field_{args.tag}.npz",
        ux=ux[:, j, :].cpu().numpy(),
        uz=uz[:, j, :].cpu().numpy(),
        rho=rho[:, j, :].cpu().numpy(),
        center=np.array(center), diameter=np.array(D),
    )

    payload = {
        "stage": "physics",
        "tag": args.tag,
        "environment": environment(),
        "parameters": {
            "impl": "ibm3d_mdf (peskin tensor-product stencil + multi-direct forcing)",
            "kernel": args.kernel,
            "n_iter": args.n_iter,
            "subdivisions": args.subdivisions,
            "diameter": D,
            "grid_zyx": list(shape),
            "box_D": args.box,
            "streamwise_D": nx / D,
            "d_over_h": d_over_h,
            "volume_fraction": phi,
            "reynolds_target": args.re,
            "speed_ref": U,
            "nu": nu,
            "tau": tau,
            "mach": U / math.sqrt(1.0 / 3.0),
            "n_markers": int(verts.shape[0]),
            "marker_area_total": float(ds.sum()),
            "analytic_sphere_area": math.pi * D * D,
            "marker_area_rel_err": float(ds.sum()) / (math.pi * D * D) - 1.0,
            "dtype": args.dtype,
            "max_steps": args.max_steps,
            "warmup": args.warmup,
            "steady_tol": args.steady_tol,
            "frozen_g": g_frozen,
            "steps_run": history[-1]["step"] if history else 0,
            "converged_at": converged_at,
        },
        "results": {
            "cd_drive_mean": cd_drive_mean,
            "cd_drive_std": std("cd_drive"),
            "cd_marker_mean": cd_marker_mean,
            "cd_marker_std": std("cd_marker"),
            "marker_vs_drive_rel": (cd_marker_mean - cd_drive_mean) / cd_drive_mean,
            "reynolds_measured": re_mean,
            "u_flux_mean": mean("u_flux"),
            "slip_rel_mean": mean("slip_rel"),
            "mass_drift_rel": max(r["mass_drift_rel"] for r in history),
            "cd_reference_schiller_naumann": ref_sn,
            "cd_reference_clift": ref_cw,
            "error_drive_vs_sn": (cd_drive_mean - ref_sn) / ref_sn,
            "error_marker_vs_sn": (cd_marker_mean - ref_sn) / ref_sn,
            "hasimoto_factor": hasimoto_correction(phi),
            "cd_predicted_hasimoto_x_sn": cd_pred,
            "error_drive_vs_hasimoto": (cd_drive_mean - cd_pred) / cd_pred,
            "wall_time_s": history[-1]["elapsed_s"] if history else float("nan"),
            "steps": history[-1]["step"] if history else 0,
        },
        "windows": windows,
        "history": history,
    }
    (ART / f"physics_{args.tag}.json").write_text(json.dumps(payload, indent=2, default=float))
    print(
        f"[{args.tag}] D={D:g} box={args.box:g} phi={phi:.5f} Re={re_mean:.2f} "
        f"Cd_drive={cd_drive_mean:.4f}+-{std('cd_drive'):.4f} Cd_marker={cd_marker_mean:.4f} "
        f"SN={ref_sn:.4f} err_vs_SN={(cd_drive_mean - ref_sn) / ref_sn * 100:+.2f}% "
        f"hasi_pred={cd_pred:.4f} err_vs_hasi={(cd_drive_mean - cd_pred) / cd_pred * 100:+.2f}% "
        f"steps={payload['results']['steps']} t={payload['results']['wall_time_s']:.0f}s",
        flush=True,
    )
    return payload


# --------------------------------------------------------------------------- #
# stage: summarize
# --------------------------------------------------------------------------- #

def stage_summarize() -> dict:
    import numpy as np

    rows = []
    for path in sorted(ART.glob("physics_*.json")):
        d = json.loads(path.read_text())
        p, r = d["parameters"], d["results"]
        rows.append(
            {
                "tag": d["tag"],
                "diameter": p["diameter"],
                "box_D": p["box_D"],
                "d_over_h": p["d_over_h"],
                "volume_fraction": p["volume_fraction"],
                "n_iter": p["n_iter"],
                "dtype": p["dtype"],
                "reynolds_measured": r["reynolds_measured"],
                "cd_drive": r["cd_drive_mean"],
                "cd_drive_std": r["cd_drive_std"],
                "cd_marker": r["cd_marker_mean"],
                "marker_vs_drive_rel": r["marker_vs_drive_rel"],
                "cd_schiller_naumann": r["cd_reference_schiller_naumann"],
                "error_vs_sn": r["error_drive_vs_sn"],
                "hasimoto_factor": r["hasimoto_factor"],
                "cd_pred_hasimoto": r["cd_predicted_hasimoto_x_sn"],
                "error_vs_hasimoto": r["error_drive_vs_hasimoto"],
                "steps": r["steps"],
                "converged_at": p["converged_at"],
                "wall_time_s": r["wall_time_s"],
            }
        )

    extrap = {}
    for group, sel in (("all", [r for r in rows if r["n_iter"] == 5]),):
        if len(sel) < 2:
            continue
        x = np.array([(math.pi / 6.0) ** (1.0 / 3.0) * r["d_over_h"] for r in sel])
        y = np.array([r["cd_drive"] for r in sel])
        A = np.vstack([np.ones_like(x), x]).T
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        resid = y - A @ coef
        if len(x) > 2:
            sigma2 = float((resid ** 2).sum() / (len(x) - 2))
            cov = sigma2 * np.linalg.inv(A.T @ A)
            cd_inf_sigma = float(math.sqrt(cov[0, 0]))
        else:
            cd_inf_sigma = float("nan")
        re_mean = float(np.mean([r["reynolds_measured"] for r in sel]))
        sn = cd_schiller_naumann(re_mean)
        cw = cd_clift(re_mean)
        extrap[group] = {
            "boxes": [r["tag"] for r in sel],
            "cd_inf": float(coef[0]),
            "cd_inf_sigma": cd_inf_sigma,
            "slope_b": float(coef[1]),
            "residuals": resid.tolist(),
            "reynolds_mean": re_mean,
            "schiller_naumann": sn,
            "clift": cw,
            "error_vs_sn": (float(coef[0]) - sn) / sn,
            "error_vs_clift": (float(coef[0]) - cw) / cw,
        }

    payload = {
        "stage": "summarize",
        "environment": environment(),
        "rows": rows,
        "extrapolation": extrap,
        "acceptance": {
            "band": 0.15,
            "per_case": {
                r["tag"]: {
                    "vs_sn": r["error_vs_sn"],
                    "vs_hasimoto": r["error_vs_hasimoto"],
                    "marker_vs_drive": r["marker_vs_drive_rel"],
                }
                for r in rows
            },
            "extrapolated_vs_sn": extrap.get("all", {}).get("error_vs_sn"),
        },
    }
    (ART / "xval_summary_r2.json").write_text(json.dumps(payload, indent=2, default=float))
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("operator", "physics", "summarize"), required=True)
    ap.add_argument("--box", type=float, default=6.0, help="lateral box size in diameters")
    ap.add_argument("--streamwise", type=float, default=None)
    ap.add_argument("--diameter", type=float, default=8.0)
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--speed", type=float, default=0.1)
    ap.add_argument("--n-iter", type=int, default=5)
    ap.add_argument("--kernel", default="peskin4pt")
    ap.add_argument("--subdivisions", type=int, default=3)
    ap.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    ap.add_argument("--max-steps", type=int, default=12000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--min-steps", type=int, default=5000)
    ap.add_argument("--avg-steps", type=int, default=2000)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--steady-tol", type=float, default=0.005)
    ap.add_argument("--report-interval", type=int, default=50)
    ap.add_argument("--gain", type=float, default=0.7)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tag", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    ART.mkdir(parents=True, exist_ok=True)
    if args.stage == "operator":
        p = stage_operator()
        for c in p["cases"]:
            print(
                f"{c['kernel']:>10}: PoU={c['partition_of_unity_max_abs_err']:.2e} "
                f"m1={c['first_moment_max_abs']:.2e} "
                f"m2spread={c['second_moment_translation_spread']:.2e} "
                f"m2relerr={c['second_moment_rel_err_vs_continuous']:+.3e} "
                f"lin={c['linear_field_reproduction_rel_err']:.2e} "
                f"smooth={c['smooth_field_interp_rel_err']:.3e} "
                f"adj={c['spread_interp_adjointness_rel_err']:.2e} "
                f"mom={c['momentum_conservation_rel_err']:.2e}"
            )
        print(f"interpolation order p = {p['interpolation_order']['observed_order']:.3f}")
        print(f"MDF max contraction ratio = {p['mdf']['max_contraction_ratio']:.3f}")
        for k, v in p["mdf"]["series"].items():
            print(f"  n_iter={k}: slip={v['final_slip_rel']:.3e} closure={v['closure_rel']:.2e}")
        return 0

    if args.stage == "physics":
        if not args.tag:
            args.tag = (
                f"{args.kernel}_it{args.n_iter}_D{args.diameter:g}_box{args.box:g}"
                f"_Re{args.re:g}_{args.dtype}"
            )
        stage_physics(args)
        return 0

    p = stage_summarize()
    print(json.dumps(p["extrapolation"], indent=2, default=float))
    for r in p["rows"]:
        print(
            f"{r['tag']:>40} Re={r['reynolds_measured']:7.2f} Cd={r['cd_drive']:.4f} "
            f"+-{r['cd_drive_std']:.4f} vs_SN={r['error_vs_sn'] * 100:+6.2f}% "
            f"vs_hasi={r['error_vs_hasimoto'] * 100:+6.2f}% "
            f"marker-drive={r['marker_vs_drive_rel'] * 100:+6.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())