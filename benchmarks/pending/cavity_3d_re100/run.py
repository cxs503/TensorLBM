"""3D lid-driven cubic cavity, Re=100, 33^3 — benchmark vs Ku, Hirsh & Taylor (1987).

Anchor: CASE-LBM-CAVITY-3D-RE100 (owner 2026-11-12 算例池令).

Reference
---------
H. C. Ku, R. S. Hirsh, T. D. Taylor, "A pseudospectral method for solution of the
three-dimensional incompressible Navier-Stokes equations", J. Comput. Phys. 70
(1987) 439-462, DOI:10.1016/0021-9991(87)90190-2.  3D cubic cavity [0,1]^3,
5 stationary walls + moving lid, Chebyshev pseudospectral, 33^3 grid,
Re = U*L/nu = 100/400/1000.

Method (reused TensorLBM capability, no new solver)
---------------------------------------------------
* D3Q19, TRT collision (Lambda = 3/16, Ginzburg 2008)  -> tensorlbm.solver3d.collide_trt3d
* 5 stationary walls: pre-streaming half-way bounce-back (validated V3-3D ordering,
  identical to benchmarks/verified/cavity_3d_full/run.py)
* moving lid (y = ny-1): Zou/He moving-wall D3Q19 -> tensorlbm.boundary.boundaries3d
* streaming: tensorlbm.solver3d.stream3d

Normalisation: U_lid = 1, L = 1 (cavity side). In lattice units the lid speed is
u_lid << 1 (Mach number control); all reported velocities are u/u_lid.

Usage
-----
    python run.py --n 33 --re 100 --u-lid 0.1 --out result_33.json
    python run.py --n 33 --smoke            # 300 steps plumbing check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from tensorlbm.boundary.boundaries3d import zou_he_moving_lid_3d  # noqa: E402
from tensorlbm.d3q19 import OPPOSITE, equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.solver3d import collide_trt3d, stream3d  # noqa: E402


def stationary_pre_bounce3d(f_pre: torch.Tensor, f: torch.Tensor, wall: torch.Tensor) -> torch.Tensor:
    """Pre-streaming half-way bounce-back on stationary walls (V3-3D ordering)."""
    opp = OPPOSITE.to(f.device)
    return torch.where(wall.unsqueeze(0), f_pre[opp], f)


def run_case(
    n: int = 33,
    re: float = 100.0,
    u_lid: float = 0.1,
    max_steps: int = 400_000,
    resid_tol: float = 1e-9,
    resid_interval: int = 500,
    length_convention: str = "n-1",
    dump_path: str | None = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
) -> dict:
    """Run one cavity case; return extrema + full mid-plane centreline profiles."""
    nz = ny = nx = n
    # physical cavity side in lattice units: walls live on nodes 0 and n-1,
    # half-way bounce-back places the wall half a cell outside -> L = n-1.
    L = (n - 1) if length_convention == "n-1" else n
    nu = u_lid * L / re
    tau = 3.0 * nu + 0.5

    dev = torch.device(device)
    rho0 = torch.ones((nz, ny, nx), dtype=dtype, device=dev)
    zero = torch.zeros((nz, ny, nx), dtype=dtype, device=dev)
    f = equilibrium3d(rho0, zero, zero, zero)
    mass0 = float(rho0.sum().item())

    wall = torch.zeros((nz, ny, nx), dtype=torch.bool, device=dev)
    wall[:, :, 0] = True  # x = 0
    wall[:, :, -1] = True  # x = n-1
    wall[:, 0, :] = True  # y = 0  (bottom)
    wall[0, :, :] = True  # z = 0
    wall[-1, :, :] = True  # z = n-1
    interior = ~wall
    interior[:, -1, :] = False  # lid row excluded from residual

    _, ux_p, uy_p, uz_p = macroscopic3d(f)
    ux_p, uy_p, uz_p = ux_p.clone(), uy_p.clone(), uz_p.clone()
    resid_hist = []
    t0 = time.time()
    last_resid = float("nan")
    converged = False
    step = 0
    for step in range(1, max_steps + 1):
        f_pre = f
        f = collide_trt3d(f, tau)
        f = stationary_pre_bounce3d(f_pre, f, wall)
        f = stream3d(f)
        f = zou_he_moving_lid_3d(f, u_lid)
        if step % resid_interval == 0:
            _, ux, uy, uz = macroscopic3d(f)
            du = float(
                torch.maximum(
                    torch.maximum(
                        torch.abs(ux[interior] - ux_p[interior]),
                        torch.abs(uy[interior] - uy_p[interior]),
                    ),
                    torch.abs(uz[interior] - uz_p[interior]),
                )
                .max()
                .item()
            )
            last_resid = du
            resid_hist.append((step, du))
            ux_p, uy_p, uz_p = ux.clone(), uy.clone(), uz.clone()
            if verbose and (len(resid_hist) % 10 == 0):
                print(f"  step {step:7d}  dU_max {du:.3e}  t {time.time()-t0:6.1f}s", flush=True)
            if du < resid_tol:
                converged = True
                break
    elapsed = time.time() - t0

    rho, ux, uy, uz = macroscopic3d(f)
    mass = float(rho.sum().item())

    kz = (nz - 1) // 2  # mid span plane z = 0.5
    ix_c = (nx - 1) // 2  # x = 0.5
    iy_c = (ny - 1) // 2  # y = 0.5

    y = np.arange(ny) / (ny - 1.0)
    x = np.arange(nx) / (nx - 1.0)
    # vertical centreline (x=0.5, z=0.5): u_x(y); drop the lid node (imposed BC)
    u_vc = (ux[kz, :, ix_c] / u_lid).cpu().numpy()
    w_vc = (uz[kz, :, ix_c] / u_lid).cpu().numpy()
    u_vc_fluid = u_vc[1:-1]
    y_fluid = y[1:-1]
    # horizontal centreline (y=0.5, z=0.5): u_y(x); drop both side walls
    v_hc = (uy[kz, iy_c, :] / u_lid).cpu().numpy()
    v_hc_fluid = v_hc[1:-1]
    x_fluid = x[1:-1]

    def extremum(profile: np.ndarray, coords: np.ndarray, kind: str) -> tuple[float, float]:
        i = int(np.argmin(profile)) if kind == "min" else int(np.argmax(profile))
        val = float(profile[i])
        # parabolic refinement of the extremum location (3-point stencil)
        pos = float(coords[i])
        if 0 < i < len(profile) - 1:
            ym, y0, yp = profile[i - 1], profile[i], profile[i + 1]
            denom = ym - 2.0 * y0 + yp
            if abs(denom) > 1e-30:
                delta = 0.5 * (ym - yp) / denom
                delta = float(np.clip(delta, -0.5, 0.5))
                pos = float(coords[i] + delta * (coords[1] - coords[0]))
        return val, pos

    u_min, y_umin = extremum(u_vc_fluid, y_fluid, "min")
    u_max, y_umax = extremum(u_vc_fluid, y_fluid, "max")
    v_min, x_vmin = extremum(v_hc_fluid, x_fluid, "min")
    v_max, x_vmax = extremum(v_hc_fluid, x_fluid, "max")

    # spanwise symmetry error on the mid plane (physical check, uz(w) antisymmetry)
    uz_np = (uz / u_lid).cpu().numpy()
    ux_np = (ux / u_lid).cpu().numpy()
    span = {
        "uz_midplane_max_abs": float(np.abs(uz_np[kz]).max()),
        "uz_global_max_abs": float(np.abs(uz_np).max()),
        "u_midplane_asym_adjacent": float(np.abs(ux_np[kz + 1] - ux_np[kz - 1]).max()),
    }

    out = {
        "case": "cavity_3d_re100",
        "n": n,
        "re": re,
        "u_lid": u_lid,
        "length_convention": length_convention,
        "L_lattice": L,
        "nu_lattice": nu,
        "tau": tau,
        "collision": "TRT (Lambda=3/16)",
        "steps": step,
        "converged": converged,
        "resid_tol": resid_tol,
        "last_resid": last_resid,
        "resid_hist": resid_hist,
        "elapsed_s": round(elapsed, 2),
        "mass_drift": mass - mass0,
        "finite": bool(torch.isfinite(f).all().item()),
        "u_min": u_min,
        "u_min_y": y_umin,
        "u_max": u_max,
        "u_max_y": y_umax,
        "v_min": v_min,
        "v_min_x": x_vmin,
        "v_max": v_max,
        "v_max_x": x_vmax,
        "u_centerline_mid": [float(a) for a in u_vc],
        "v_centerline_mid": [float(a) for a in v_hc],
        "y_coords": [float(a) for a in y],
        "x_coords": [float(a) for a in x],
        "span_symmetry": span,
    }
    if dump_path:
        Path(dump_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dump_path,
            ux=ux_np.astype(np.float32),
            uy_np=(uy / u_lid).cpu().numpy().astype(np.float32),
            uz=uz_np.astype(np.float32),
            rho=rho.cpu().numpy().astype(np.float32),
            meta=np.array([n, re, u_lid], dtype=np.float64),
        )
        out["dump_path"] = dump_path
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=33)
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--u-lid", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=400_000)
    ap.add_argument("--resid-tol", type=float, default=1e-9)
    ap.add_argument("--resid-interval", type=int, default=500)
    ap.add_argument("--length-convention", default="n-1", choices=["n-1", "n"])
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dump", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    if args.smoke:
        args.steps = 300
        args.resid_interval = 100

    res = run_case(
        n=args.n,
        re=args.re,
        u_lid=args.u_lid,
        max_steps=args.steps,
        resid_tol=args.resid_tol,
        resid_interval=args.resid_interval,
        length_convention=args.length_convention,
        dump_path=args.dump,
        device=args.device,
        dtype=torch.float64 if args.dtype == "float64" else torch.float32,
    )
    keys = ["n", "re", "u_lid", "tau", "steps", "converged", "last_resid", "elapsed_s",
            "u_min", "u_min_y", "v_min", "v_min_x", "u_max", "u_max_y", "v_max", "v_max_x",
            "span_symmetry"]
    print(json.dumps({k: res[k] for k in keys}, indent=1, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())