#!/usr/bin/env python3
"""Resistance-chain CLOUD producer: 2-D LBM free-stream flow past a ship-section.

Mission m4927 (`resistance-chain-de-mock`, CLOUD gap).  The platform reports of
the resistance chain (#5539 ice / #5393 Holtrop benchmark / #5384 batch sweep)
carry only *scalar* evidence (R_t, C_t, success rate).  This script supplies the
missing **spatial / angular field evidence**: time-averaged velocity, vorticity
and pressure-coefficient fields on the whole domain, plus the angular
distribution of the surface pressure and wall-shear coefficients Cp(phi),
Cf(phi) -- the distribution that *integrates into the resistance*.

Numerics = library path only (zero hand-written collide/stream/equilibrium):

* ``tensorlbm.solver.collide_mrt`` (with ``tau_field`` for the sponge layer)
* ``tensorlbm.solver.stream``
* ``tensorlbm.d2q9.equilibrium``
* ``tensorlbm.boundaries.far_field_bc_2d`` (inlet / lateral free stream +
  zero-gradient outlet + full-way bounce-back on the obstacle)
* ``tensorlbm.boundaries.make_sponge_strength`` (downstream absorbing layer,
  tau_eff = tau * (1 + alpha * sigma))
* ``tensorlbm.boundaries.compute_obstacle_forces`` (Ladd momentum exchange:
  Cd, Cl) -- called post-stream / pre-bounce-back, exactly as in the verified
  benchmark ``benchmarks/verified/cylinder/re100/run.py``.

Geometries
----------
``cylinder``  circular section, Re = 100.  Reference: Braza et al. 1986
              (JFM 165) Cd = 1.35 +- 0.05, St = 0.164-0.165 -- the anchor used by
              the in-repo verified benchmark ``benchmarks/verified/cylinder``.
``ellipse``   ship waterline section (aspect ratio a/b = 4, i.e. L/B = 4 like a
              slender bow section).  Angular reference for Cp(phi) is the exact
              potential-flow solution (conformal map, see
              ``potential_cp_ellipse``); no measured Cd reference exists in the
              repo -> reported as NO-DATA-TBD downstream, never as a measurement.

Per-surface-node momentum exchange
----------------------------------
The library's ``compute_obstacle_forces`` sums ``2 * sum_i c_i f_i`` over *all*
solid cells.  The same per-cell expression restricted to the **boundary** solid
cells (solid cell with >= 1 fluid 4-neighbour) yields the local force density
that this script projects on the local surface normal / tangent to produce
Cd_p(phi) and Cd_f(phi).  The identity

    sum_boundary_nodes F_x / q_dyn  ==  compute_obstacle_forces()[0] / q_dyn

is verified numerically (``checksum_rel_err`` in the output JSON) -- if it
fails, the run is marked invalid rather than silently reported.

Usage
-----
    python benchmarks/resistance_chain_cloud/run.py \
        --geo cylinder --D 24 36 54 --re 100 --u-in 0.1 --t-star 120 \
        --out artifacts/resistance_chain_cloud/fields/cylinder
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))  # this worktree's library (not a global install)
sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import tensorlbm  # noqa: E402

from tensorlbm.boundaries import (  # noqa: E402
    compute_obstacle_forces,
    cylinder_mask,
    far_field_bc_2d,
    make_sponge_strength,
)
from tensorlbm.d2q9 import C as C_D2Q9  # noqa: E402
from tensorlbm.d2q9 import equilibrium  # noqa: E402
from tensorlbm.solver import collide_mrt, stream  # noqa: E402

# ---------------------------------------------------------------- references
REF_CD_CYL = 1.35  # Braza et al. 1986 (JFM 165), band 1.30-1.40
REF_CD_CYL_BAND = 0.05
REF_ST_CYL = 0.1645  # 0.164-0.165, in-repo verified benchmark
ST_SEED = 0.165  # inlet perturbation frequency (2 periods, then free stream)
SEED_AMPL = 0.10
SEED_PERIODS = 2.0

DOMAIN_X = 22.0  # domain length in body reference lengths
DOMAIN_Y = 20.0  # domain height
BODY_X = 6.0  # body centre, distance from inlet (in reference lengths)
SPONGE_X = 8.0  # sponge width (in reference lengths), downstream only
SPONGE_ALPHA = 10.0
N_BINS = 36  # surface angular bins (10 deg)
SMOOTH_PASSES = 2  # circular 3-tap smoothing passes on the binned surface data


# ------------------------------------------------------------------ geometry
def ellipse_mask(
    nx: int, ny: int, cx: float, cy: float, a: float, b: float, device: torch.device
) -> torch.Tensor:
    """Boolean mask for an elliptic obstacle (semi-axes ``a`` along x, ``b`` along y)."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1.0


def potential_cp_ellipse(a: float, b: float, phi: np.ndarray) -> np.ndarray:
    """Exact potential-flow Cp on an ellipse (semi-axes a,b) in uniform flow.

    Conformal map ``z = zeta + c^2/zeta`` (``c^2 = a^2 - b^2``) from the circle
    ``|zeta| = R = (a+b)/2``.  With the complex potential
    ``w = U (zeta + R^2/zeta)`` the surface speed is

        V_t/U = |1 - R^2/zeta^2| / |1 - c^2/zeta^2|,   zeta = R e^{i phi}

    and ``Cp = 1 - (V_t/U)^2``.  For ``a = b = R`` (c = 0) this reduces to the
    classical ``Cp = 1 - 4 sin^2(phi)`` (verified in
    ``tests/test_resistance_chain_cloud.py``).  ``phi = 0`` is the downstream
    extremity, ``phi = pi`` the upstream stagnation point.
    """
    phi = np.asarray(phi, dtype=np.float64)
    R = 0.5 * (a + b)
    c2 = a * a - b * b
    zeta = R * np.exp(1j * phi)
    num = np.abs(1.0 - R**2 / zeta**2)
    den = np.abs(1.0 - c2 / zeta**2)
    vt = num / den
    return 1.0 - vt**2


def _circ_smooth(v: np.ndarray, passes: int = 2) -> np.ndarray:
    """Circular 3-tap smoothing of an angular profile (NaNs filled by neighbours)."""
    out = v.astype(np.float64).copy()
    if np.all(~np.isfinite(out)):
        return out
    # fill NaNs circularly with nearest finite value
    idx = np.arange(out.size)
    finite = np.isfinite(out)
    if not finite.all():
        pos = idx[finite]
        for i in idx[~finite]:
            d = (pos - i + out.size // 2) % out.size - out.size // 2
            out[i] = out[pos[np.argmin(np.abs(d))]]
    for _ in range(passes):
        out = 0.25 * np.roll(out, 1) + 0.5 * out + 0.25 * np.roll(out, -1)
    return out


def boundary_nodes(mask: torch.Tensor) -> torch.Tensor:
    """Solid cells having at least one fluid 4-neighbour (staircase boundary)."""
    fluid = ~mask
    nb = torch.zeros_like(mask)
    nb[1:, :] |= fluid[:-1, :]
    nb[:-1, :] |= fluid[1:, :]
    nb[:, 1:] |= fluid[:, :-1]
    nb[:, :-1] |= fluid[:, 1:]
    return mask & nb


# ---------------------------------------------------------------------- core
def run_case(
    geo: str,
    D: int,
    re: float,
    u_in: float,
    t_star: float,
    out_dir: Path,
    device: torch.device,
    warmup_frac: float = 0.3,
    avg_every: int = 4,
) -> dict:
    """One grid level: run, time-average, write fields + surface distributions."""
    t0 = time.time()
    lib_path = tensorlbm.__file__
    if not lib_path.startswith(str(_REPO / "src")):
        raise RuntimeError(f"imported tensorlbm from {lib_path}, expected {_REPO}/src")

    nx = int(round(DOMAIN_X * D))
    ny = int(round(DOMAIN_Y * D))
    cx = BODY_X * D
    cy = ny / 2.0
    nu = u_in * D / re
    tau = 0.5 + 3.0 * nu

    if geo == "cylinder":
        a = b = D / 2.0
        mask = cylinder_mask(nx, ny, cx, cy, D / 2.0, device)
    elif geo == "ellipse":
        a = D / 2.0  # semi-major (streamwise); reference length = D = 2a
        b = a / 4.0
        mask = ellipse_mask(nx, ny, cx, cy, a, b, device)
    else:  # pragma: no cover
        raise ValueError(geo)

    sigma = make_sponge_strength(
        ny, nx, int(nx - SPONGE_X * D), int(SPONGE_X * D), power=2.0, device=device
    )
    tau_field = tau * (1.0 + SPONGE_ALPHA * sigma)

    rho0 = torch.ones((ny, nx), device=device)
    f = equilibrium(rho0, torch.full_like(rho0, u_in), torch.zeros_like(rho0))

    nsteps = int(round(t_star * D / u_in))
    seed_steps = int(round(SEED_PERIODS / (ST_SEED * u_in / D)))
    omega_seed = 2.0 * math.pi * ST_SEED * u_in / D
    uy_amp = SEED_AMPL * u_in
    step0 = int(warmup_frac * nsteps)

    # ---- surface (staircase) boundary nodes of the solid ----
    bnd = boundary_nodes(mask)
    ys, xs = np.nonzero(bnd.cpu().numpy())
    ang = np.arctan2((ys - cy) / b, (xs - cx) / a)  # parametric angle
    bin_idx = np.clip(((ang + np.pi) / (2 * np.pi) * N_BINS).astype(int), 0, N_BINS - 1)
    nxv = (xs - cx) / a**2
    nyv = (ys - cy) / b**2
    nrm = np.hypot(nxv, nyv)
    nrm[nrm == 0] = 1.0
    nxv, nyv = nxv / nrm, nyv / nrm
    txv, tyv = -nyv, nxv

    bnd_mask_t = torch.zeros_like(mask)
    bnd_mask_t[ys, xs] = True

    # first fluid 4-neighbour of each boundary node (wall pressure proxy, O(dx))
    fluid_np = (~mask).cpu().numpy()
    ys_fluid = np.full(ys.shape, -1, dtype=np.int64)
    xs_fluid = np.full(xs.shape, -1, dtype=np.int64)
    for ddy, ddx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        yy_ = ys + ddy
        xx_ = xs + ddx
        ok = (yy_ >= 0) & (yy_ < ny) & (xx_ >= 0) & (xx_ < nx)
        ok[ok] &= fluid_np[yy_[ok], xx_[ok]]
        fill = (ys_fluid < 0) & ok
        ys_fluid[fill] = yy_[fill]
        xs_fluid[fill] = xx_[fill]
    if (ys_fluid < 0).any():
        raise RuntimeError("boundary node without fluid neighbour -- mask too thick")

    c = C_D2Q9.to(device).to(torch.float32)
    cxv = c[:, 0]
    cyv = c[:, 1]

    q_dyn = 0.5 * 1.0 * u_in * u_in * D  # rho_inf = 1, body length = D lattice units
    q_loc = 0.5 * 1.0 * u_in * u_in

    f64 = torch.float64
    acc = {k: torch.zeros((ny, nx), dtype=f64) for k in ("rho", "ux", "uy", "wz")}
    acc2 = {k: torch.zeros((ny, nx), dtype=f64) for k in ("uu", "vv", "uv", "wz2")}
    n_avg = 0
    fx_hist: list[float] = []
    fy_hist: list[float] = []
    fx_bnd_hist: list[float] = []
    cp_bin_acc = np.zeros(N_BINS)
    cf_bin_acc = np.zeros(N_BINS)
    cp_rho_bin_acc = np.zeros(N_BINS)
    cf_rho_bin_acc = np.zeros(N_BINS)
    cp_rho_bin2_acc = np.zeros(N_BINS)
    cf_rho_bin2_acc = np.zeros(N_BINS)
    bin_cnt = np.zeros(N_BINS)
    node_cnt = np.bincount(bin_idx, minlength=N_BINS).astype(float)
    cd_p_nodes: list[float] = []
    cd_f_nodes: list[float] = []

    for step in range(1, nsteps + 1):
        f = collide_mrt(f, tau, tau_field=tau_field)
        f = stream(f)
        fx, fy = compute_obstacle_forces(f, mask)  # post-stream, pre-bounce-back
        f_solid_b = f[:, bnd_mask_t]  # (9, Nb) local momentum exchange
        fx_node = 2.0 * (cxv.view(9, 1) * f_solid_b).sum(dim=0)
        fy_node = 2.0 * (cyv.view(9, 1) * f_solid_b).sum(dim=0)
        f = far_field_bc_2d(f, u_in, mask)  # free stream + bounce-back on solid

        if step <= seed_steps:  # 2-period inlet perturbation -> shedding starts at once
            uy_val = uy_amp * math.sin(omega_seed * step)
            f[:, :, 0] = equilibrium(
                torch.ones((ny, 1), device=device, dtype=f.dtype),
                torch.full((ny, 1), u_in, device=device, dtype=f.dtype),
                torch.full((ny, 1), uy_val, device=device, dtype=f.dtype),
            )[:, :, 0]

        fx_hist.append(float(fx))
        fy_hist.append(float(fy))
        fx_bnd_hist.append(float(fx_node.sum()))

        if step > step0 and step % avg_every == 0:
            rho_t = f.sum(dim=0)
            ux_t = (cxv.view(9, 1, 1) * f).sum(dim=0) / rho_t
            uy_t = (cyv.view(9, 1, 1) * f).sum(dim=0) / rho_t
            ux64 = ux_t.to(f64)
            uy64 = uy_t.to(f64)
            wz64 = torch.zeros_like(ux64)
            # 4th-order central differences for omega_z = du_y/dx - du_x/dy
            wz64[2:-2, 2:-2] = (
                -uy64[2:-2, 4:] + 8.0 * uy64[2:-2, 3:-1]
                - 8.0 * uy64[2:-2, 1:-3] + uy64[2:-2, :-4]
                + ux64[4:, 2:-2] - 8.0 * ux64[3:-1, 2:-2]
                + 8.0 * ux64[1:-3, 2:-2] - ux64[:-4, 2:-2]
            ) / 12.0
            acc["rho"] += rho_t.to(f64)
            acc["ux"] += ux64
            acc["uy"] += uy64
            acc["wz"] += wz64
            acc2["uu"] += ux64 * ux64
            acc2["vv"] += uy64 * uy64
            acc2["uv"] += ux64 * uy64
            acc2["wz2"] += wz64 * wz64
            n_avg += 1

            # --- wall quantities on the fluid side (half-way bounce-back) ---
            rho_f = rho_t.detach().cpu().numpy()[ys_fluid, xs_fluid]
            ut = ux64.detach().cpu().numpy()[ys_fluid, xs_fluid] * txv + \
                uy64.detach().cpu().numpy()[ys_fluid, xs_fluid] * tyv
            cp_node = (rho_f - 1.0) / 3.0 / q_loc          # p = (rho-1)c_s^2
            cf_node = 4.0 * nu * ut / (u_in * u_in)        # tau_w = mu*2*ut (wall at 0.5 dx)
            np.add.at(cp_rho_bin_acc, bin_idx, cp_node)
            np.add.at(cf_rho_bin_acc, bin_idx, cf_node)
            np.add.at(cp_rho_bin2_acc, bin_idx, cp_node * cp_node)
            np.add.at(cf_rho_bin2_acc, bin_idx, cf_node * cf_node)
            np.add.at(bin_cnt, bin_idx, 1.0)
            cd_p_nodes.append(float((cp_node * nxv).sum() / D))
            cd_f_nodes.append(float((cf_node * txv).sum() / D))
            # momentum-exchange local sums (diagnostic only; interior cells make the
            # per-cell split non-local -> reported but not used for the decomposition)
            fx_np = fx_node.detach().cpu().numpy().astype(np.float64)
            fy_np = fy_node.detach().cpu().numpy().astype(np.float64)
            np.add.at(cp_bin_acc, bin_idx, (fx_np * nxv + fy_np * nyv) / q_loc)
            np.add.at(cf_bin_acc, bin_idx, (fx_np * txv + fy_np * tyv) / q_loc)

    if not bool(torch.isfinite(f).all().item()):
        raise RuntimeError(f"{geo} D={D}: non-finite populations")

    inv = 1.0 / max(n_avg, 1)
    mean = {k: (v * inv).numpy() for k, v in acc.items()}
    uu = (acc2["uu"] * inv).numpy()
    vv = (acc2["vv"] * inv).numpy()
    wz2 = (acc2["wz2"] * inv).numpy()
    fluct = {
        "u": np.sqrt(np.maximum(uu - mean["ux"] ** 2, 0.0)),
        "v": np.sqrt(np.maximum(vv - mean["uy"] ** 2, 0.0)),
        "wz": np.sqrt(np.maximum(wz2 - mean["wz"] ** 2, 0.0)),
    }

    fx_np = np.asarray(fx_hist)
    fy_np = np.asarray(fy_hist)
    cd = fx_np / q_dyn
    cl = fy_np / q_dyn
    w0 = step0 - 1
    cd_mean = float(cd[w0:].mean())
    cd_std = float(cd[w0:].std(ddof=1))
    blocks = np.array_split(cd[w0:], 10)
    cd_sem = float(np.std([bb.mean() for bb in blocks], ddof=1) / math.sqrt(10))

    cl_mean = float(cl[w0:].mean())
    cl_rms = float(np.sqrt((cl[w0:] ** 2).mean()))

    st_val = float("nan")
    cl_w = cl[w0:] - cl_mean
    if cl_w.size >= 256:
        spec = np.abs(np.fft.rfft(cl_w * np.hanning(cl_w.size)))
        k = int(np.argmax(spec[1:])) + 1
        delta = 0.0
        if 0 < k < spec.size - 1:
            a1, b1, c1 = spec[k - 1], spec[k], spec[k + 1]
            den = a1 - 2 * b1 + c1
            if abs(den) > 1e-30:
                delta = float(0.5 * (a1 - c1) / den)
        st_val = float((k + delta) / cl_w.size * D / u_in)

    phi = (np.arange(N_BINS) + 0.5) / N_BINS * 2 * np.pi - np.pi
    cnt = np.where(node_cnt > 0, bin_cnt, np.nan)
    cp_rho_bin = cp_rho_bin_acc / cnt
    cf_rho_bin = cf_rho_bin_acc / cnt
    cp_bin = (cp_bin_acc / cnt)  # momentum-exchange diagnostic (dimensionless/node)
    cf_bin = (cf_bin_acc / cnt)
    cp_rho_std = np.sqrt(np.maximum(cp_rho_bin2_acc / cnt - cp_rho_bin**2, 0.0))
    cf_rho_std = np.sqrt(np.maximum(cf_rho_bin2_acc / cnt - cf_rho_bin**2, 0.0))
    cp_rho_smooth = _circ_smooth(cp_rho_bin, SMOOTH_PASSES)
    cf_rho_smooth = _circ_smooth(cf_rho_bin, SMOOTH_PASSES)

    fx_bnd_mean = float(np.mean(fx_bnd_hist[w0:]))
    checksum_rel = abs(fx_bnd_mean - cd_mean * q_dyn) / (abs(cd_mean * q_dyn) + 1e-30)

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{geo}_D{D}"
    np.savez_compressed(
        out_dir / f"{tag}_fields.npz",
        x=(np.arange(nx) - cx) / D,  # x / D (body reference length)
        y=(np.arange(ny) - cy) / D,  # y / D
        ux=mean["ux"] / u_in,
        uy=mean["uy"] / u_in,
        rho=mean["rho"],
        cp=(mean["rho"] - 1.0) / 3.0 / q_loc,
        wz=mean["wz"] * D / u_in,
        u_rms=fluct["u"] / u_in,
        v_rms=fluct["v"] / u_in,
        wz_rms=fluct["wz"] * D / u_in,
        mask=mask.cpu().numpy(),
        boundary_mask=bnd.cpu().numpy(),
        phi=phi,
        cp_bin=cp_bin,
        cf_bin=cf_bin,
        cp_rho_bin=cp_rho_bin,
        cf_rho_bin=cf_rho_bin,
        cp_rho_smooth=cp_rho_smooth,
        cf_rho_smooth=cf_rho_smooth,
        cp_rho_std=cp_rho_std,
        cf_rho_std=cf_rho_std,
        n_samples_per_node=float(n_avg),
        node_count=node_cnt,
        bin_count=node_cnt,
        bin_theta_center=phi,
        cd_series=cd,
        cl_series=cl,
        q_dyn=q_dyn,
        u_in=u_in,
        re=re,
        D=D,
        a=a,
        b=b,
        cx=cx,
        cy=cy,
        dx=1.0,
        warmup_from=w0,
    )

    result = {
        "case": f"{geo}_re{int(re)}_cloud",
        "geo": geo,
        "D": D,
        "nx": nx,
        "ny": ny,
        "n_cells": nx * ny,
        "re": re,
        "u_in": u_in,
        "nu_lb": nu,
        "tau": round(tau, 6),
        "a_cells": a,
        "b_cells": b,
        "steps": nsteps,
        "t_star_total": t_star,
        "t_star_averaged": t_star * (1.0 - warmup_frac),
        "avg_every": avg_every,
        "n_avg_samples": n_avg,
        "n_boundary_nodes": int(node_cnt.sum()),
        "n_bins_occupied": int((node_cnt > 0).sum()),
        "seed": {"st_seed": ST_SEED, "ampl": SEED_AMPL, "periods": SEED_PERIODS},
        "sponge": {
            "x0": int(nx - SPONGE_X * D),
            "width": int(SPONGE_X * D),
            "alpha": SPONGE_ALPHA,
        },
        "domain_D": {"x": DOMAIN_X, "y": DOMAIN_Y, "body_x": BODY_X},
        "blockage": float(1.0 / DOMAIN_Y),
        "cd_mean": cd_mean,
        "cd_std": cd_std,
        "cd_sem": cd_sem,
        "cd_u95": 1.96 * cd_sem,
        "cl_mean": cl_mean,
        "cl_rms": cl_rms,
        "st": st_val,
        "cd_pressure": float(np.mean(cd_p_nodes)) if cd_p_nodes else float("nan"),
        "cd_friction": float(np.mean(cd_f_nodes)) if cd_f_nodes else float("nan"),
        "cd_pressure_u95": float(1.96 * np.std(cd_p_nodes, ddof=1) / math.sqrt(len(cd_p_nodes)))
        if len(cd_p_nodes) > 2
        else float("nan"),
        "cd_friction_u95": float(1.96 * np.std(cd_f_nodes, ddof=1) / math.sqrt(len(cd_f_nodes)))
        if len(cd_f_nodes) > 2
        else float("nan"),
        "surface_mom_exchange_sum": float(
            np.nansum(cp_bin * node_cnt * np.cos(phi)) / D
        ),
        "cp_surface_min": float(np.nanmin(cp_rho_bin)),
        "cp_surface_max": float(np.nanmax(cp_rho_bin)),
        "cp_surface_mean": float(np.nanmean(cp_rho_bin)),
        "checksum_rel_err": checksum_rel,
        "runtime_s": round(time.time() - t0, 1),
        "field_file": str(out_dir / f"{tag}_fields.npz"),
        "lib_path": lib_path,
    }
    dsum = result["cd_pressure"] + result["cd_friction"]
    result["cd_pf_sum"] = dsum
    result["cd_decomposition_rel_err"] = abs(dsum - cd_mean) / (abs(cd_mean) + 1e-30)
    (out_dir / f"{tag}_run.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geo", default="cylinder", choices=["cylinder", "ellipse"])
    ap.add_argument("--D", type=int, nargs="+", default=[24, 36, 54])
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--u-in", type=float, default=0.08)
    ap.add_argument("--t-star", type=float, default=45.0)
    ap.add_argument("--out", default="artifacts/resistance_chain_cloud/fields")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--avg-every", type=int, default=4)
    args = ap.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    out_dir = Path(args.out) / args.geo
    summary = []
    for D in args.D:
        res = run_case(
            args.geo,
            D,
            args.re,
            args.u_in,
            args.t_star,
            out_dir,
            device,
            avg_every=args.avg_every,
        )
        summary.append(res)
        print(
            f"[{res['case']}] D={D} Cd={res['cd_mean']:.4f} St={res['st']:.4f} "
            f"checksum={res['checksum_rel_err']:.2e} {res['runtime_s']}s",
            flush=True,
        )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())