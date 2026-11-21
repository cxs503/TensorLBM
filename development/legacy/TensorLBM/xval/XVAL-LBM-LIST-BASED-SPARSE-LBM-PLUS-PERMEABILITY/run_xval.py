#!/usr/bin/env python3
"""XVAL-LBM-LIST-BASED-SPARSE-LBM-PLUS-PERMEABILITY runner.

Anchor: cross-validation of the waLBerla / lbmpy **list-based sparse LBM +
permeability** recipe (``lssfau/walberla``, OSS tag
``list-based-sparse-lbm-plus-permeability``) against TensorLBM's own
implementation.

Predeclared design (written before the runs, see ``predeclared.json``):

* **Decision** -- adopt the list-based sparse index layout in TensorLBM only if
  (i) it is representation neutral w.r.t. the in-repo dense lattice
  (elementwise |Δf| after 100 steps < 1e-12), (ii) the *address table* is
  equivalent to the OSS one (100 % of the (cell, direction) entries map to the
  same neighbour cell and the same velocity direction), (iii) the time-stepping
  agrees with the OSS generated kernel to < 1e-4 absolute, and (iv) Darcy
  permeability from the sparse solve is within 15 % of the analytic
  parallel-plate value (owner criterion) and within 1 % of the dense control.
* **Stopping rule** -- if (i)-(iv) all pass the layout is accepted as the
  porous-media production path; the first failing criterion stops promotion and
  is reported as an honest FAIL for that criterion only.
* **Sensitivity** -- two knobs, predeclared: wall distance in lattice units
  ``H ∈ {8,16,32}`` (discretisation of the half-way bounce-back wall) and
  relaxation time ``τ ∈ {0.6,0.8,1.0,1.5}`` (compressibility / slip of the
  bounce-back wall).  Analysis = Darcy fit ``k = ν <u> / g`` at steady state,
  ``<u>`` = superficial (all lattice sites) mean streamwise velocity.

Arms
----
``S`` : TensorLBM list-based sparse index array (``tensorlbm.porous.sparse_index_lbm``)
``D`` : TensorLBM dense full-lattice path (``tensorlbm.porous.dense_step``), control
``C`` : OSS reference, lbmpy/pystencils-generated ``stream_pull_collide`` kernel
        driven by ``SparseLbMapper.create_index_array`` (``oss_ref.py``)

Case 1 ``kernel``      wall-bounded box + obstacles, 100 steps, body force
Case 2 ``channel``     periodic plane channel, analytic ``k = H^2/12``
Case 3 ``percolation`` random percolating matrix (waLBerla Percolation analogue)
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tensorlbm.porous import (  # noqa: E402
    build_index_table,
    dense_step,
    dense_to_sparse,
    equilibrium_sparse,
    intrinsic_permeability,
    macroscopic_sparse,
    sparse_step,
    to_dense,
)
from tensorlbm.porous.fluid_node_table import macroscopic_with_forcing  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

RES = {"meta": {}, "case1_kernel": {}, "case2_channel": {}, "case3_percolation": {},
       "address_table": {}, "footprint": {}, "sensitivity": {}, "verdict": {}}


# --------------------------------------------------------------------------- utils
def dense_equilibrium(ny: int, nx: int, rho: float = 1.0) -> torch.Tensor:
    from tensorlbm.d2q9 import W_EXACT

    f = torch.zeros((9, ny, nx))
    f[:] = torch.tensor(W_EXACT).view(9, 1, 1) * rho
    return f


def run_dense(solid: torch.Tensor, tau: float, g: float, steps: int,
              tol: float = 0.0, f0: torch.Tensor | None = None):
    """Dense control: collide+stream in lock-step; returns (f, u_mean_streamwise, hist)."""
    ny, nx = solid.shape
    f = dense_equilibrium(ny, nx) if f0 is None else f0.clone()
    hist = []
    u_prev = None
    for _ in range(steps):
        f = dense_step(f, solid, tau=tau, force_x=g)
        _rho, ux, _uy = macroscopic_with_forcing(f, g, 0.0)
        u = float(ux[~solid].mean())
        hist.append(u)
        if tol and u_prev is not None and abs(u - u_prev) <= tol * max(abs(u), 1e-30):
            break
        u_prev = u
    return f, hist


def run_sparse(table, tau: float, g: float, steps: int, tol: float = 0.0,
               f0: torch.Tensor | None = None):
    """List-based sparse arm (TensorLBM index array)."""
    f = equilibrium_sparse(table) if f0 is None else f0.clone()
    hist = []
    u_prev = None
    for _ in range(steps):
        f = sparse_step(f, table, tau=tau, force_x=g)
        _rho, ux, _uy = macroscopic_sparse(f, table, g, 0.0)
        u = float(ux.mean())
        hist.append(u)
        if tol and u_prev is not None and abs(u - u_prev) <= tol * max(abs(u), 1e-30):
            break
        u_prev = u
    return f, hist


def darcy_k(u_super: float, nu: float, g: float) -> float:
    return nu * u_super / g


# ---------------------------------------------------------------------- case 1
def case1_kernel() -> dict:
    """Kernel equivalence on a wall-bounded box with obstacles (OSS arm runnable)."""
    import oss_ref as O

    ny, nx = 24, 32
    solid = np.zeros((ny, nx), dtype=bool)
    solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
    solid[8:12, 10:12] = True
    solid[14:16, 20:24] = True
    solid_t = torch.as_tensor(solid)
    table = build_index_table(solid_t, periodic=(False, False))
    g, tau, steps = 1e-4, 0.8, 100

    # --- address table comparison (coordinate exact, no ordering assumption)
    idx_oss, n_oss, mapper = O.oss_index_array(solid)
    coord2oss = {tuple(int(a) for a in mapper.coordinate(i)): i for i in range(n_oss)}
    ent = [tuple(int(a) for a in e) for e in O.oss_stencil()]
    from tensorlbm.d2q9 import C

    vec2oss = {ent[j]: j for j in range(9)}
    N = table.n_fluid
    assert N == n_oss, (N, n_oss)
    same_cell = same_dir = total = 0
    for k in range(N):
        y, x = divmod(int(table.cell[k]), nx)
        k_oss = coord2oss[(x, y)]
        for d in range(1, 9):
            v = int(table.index[k, d - 1])
            kt, dt = v % N, v // N
            yt, xt = divmod(int(table.cell[kt]), nx)
            w = int(idx_oss[k_oss, vec2oss[tuple(int(a) for a in C[d])] - 1])
            ko, do = w % N, w // N
            xo, yo = [int(a) for a in mapper.coordinate(ko)]
            total += 1
            same_cell += int((xo, yo) == (xt, yt))
            same_dir += int(ent[do] == ent[vec2oss[tuple(int(a) for a in C[dt])]])
    RES["address_table"] = {
        "n_fluid": N, "entries": total,
        "neighbour_cell_agreement": same_cell / total,
        "direction_agreement": same_dir / total,
        "oss_index_uint32_per_cell": int(idx_oss.shape[1]),
        "tensorlbm_index_int64_per_cell": int(table.index.shape[1]),
    }

    # --- three arms, identical initial state
    f_s = equilibrium_sparse(table)
    fd = torch.zeros((9, ny, nx))
    fd[:] = torch.tensor(__import__("tensorlbm.d2q9", fromlist=["W_EXACT"]).W_EXACT).view(9, 1, 1)

    stepper = O.OssSparseStepper(solid, tau=tau, force=(g, 0.0))
    stepper.set_pdfs_from_tensorlbm(f_s.numpy())

    d_sc, d_sd, d_cd = [], [], []
    t0 = time.time()
    for _ in range(steps):
        f_s = sparse_step(f_s, table, tau=tau, force_x=g)
        fd = dense_step(fd, solid_t, tau=tau, force_x=g)
        stepper.step(1)
        c = torch.as_tensor(stepper.tensorlbm_state())
        d_sc.append(float((c - f_s).abs().max()))
        d_sd.append(float((to_dense(f_s, table)[:, ~solid_t] - fd[:, ~solid_t]).abs().max()))
        d_cd.append(float((to_dense(c, table)[:, ~solid_t] - fd[:, ~solid_t]).abs().max()))
    wall = time.time() - t0

    # Darcy permeability of the three arms on the same case
    def k_of(f_state):
        _r, ux, _u = macroscopic_sparse(f_state, table, g, 0.0)
        phi = table.porosity
        return darcy_k(float(ux.mean()), (tau - 0.5) / 3.0, g), phi

    k_s, phi = k_of(f_s)
    k_c, _ = k_of(torch.as_tensor(stepper.tensorlbm_state()))
    _r, ux_d, _u = macroscopic_with_forcing(fd, g, 0.0)
    k_d = darcy_k(float(ux_d[~solid_t].mean()) * phi, (tau - 0.5) / 3.0, g)
    RES["case1_kernel"] = {
        "grid": [ny, nx], "porosity": phi, "tau": tau, "force": g, "steps": steps,
        "sparse_vs_dense_max_abs": max(d_sd), "sparse_vs_oss_max_abs": max(d_sc),
        "oss_vs_dense_max_abs": max(d_cd),
        "scale_max_pdf": float(f_s.abs().max()),
        "k_super_sparse": k_s, "k_super_oss": k_c, "k_super_dense": k_d,
        "k_rel_diff_sparse_vs_oss": abs(k_s - k_c) / abs(k_s),
        "k_rel_diff_sparse_vs_dense": abs(k_s - k_d) / abs(k_s),
        "wall_seconds": wall,
        "history_sparse_vs_oss": d_sc[::10],
    }
    return RES["case1_kernel"]


# ---------------------------------------------------------------------- case 2
def case2_channel(H: int, tau: float = 0.8, g: float = 1e-5, nx: int = 8,
                  max_steps: int = 9000) -> dict:
    """Periodic plane channel: analytic intrinsic k = H^2/12."""
    ny = H + 1          # fluid rows 1..H, one wall row at y=0 (half-way wall at y=0.5)
    solid = np.zeros((ny, nx), dtype=bool)
    solid[0, :] = True
    solid_t = torch.as_tensor(solid)
    table = build_index_table(solid_t, periodic=(True, True))
    nu = (tau - 0.5) / 3.0
    f_s, hist_s = run_sparse(table, tau, g, max_steps, tol=1e-10)
    f_d, hist_d = run_dense(solid_t, tau, g, min(len(hist_s), max_steps), tol=0.0)
    phi = table.porosity
    _r, ux_s, _u = macroscopic_sparse(f_s, table, g, 0.0)
    k_super = darcy_k(float(ux_s.mean()), nu, g)
    k_int = intrinsic_permeability(k_super, phi)
    # analytic: channel of height H between two half-way walls => H lattice cells
    k_exact = H * H / 12.0
    _r, ux_d, _u = macroscopic_with_forcing(f_d, g, 0.0)
    k_int_dense = float(ux_d[~solid_t].mean()) * nu / g
    # parabolic profile reconstruction: u(y) from the analytic solution
    y = np.arange(1, H + 1)
    prof = np.asarray([float(ux_s[(table.cell // nx) == yy].mean()) for yy in y])
    u_an = 4.0 * float(ux_s.mean()) * (y - 0.5) * (H - (y - 0.5)) / (H * H)
    prof_err = float(np.abs(prof - u_an).max() / u_an.max())
    return {
        "H": H, "tau": tau, "nu": nu, "porosity": phi, "steps": len(hist_s),
        "u_super_mean": float(ux_s.mean()), "Ma": float(abs(ux_s).max()) * math.sqrt(3.0),
        "k_super": k_super, "k_intrinsic": k_int, "k_analytic": k_exact,
        "err_vs_analytic": (k_int - k_exact) / k_exact,
        "k_intrinsic_dense": k_int_dense,
        "err_dense_vs_analytic": (k_int_dense - k_exact) / k_exact,
        "profile_max_rel_err_vs_parabola": prof_err,
        "profile_y": y.tolist(), "profile_u": prof.tolist(), "profile_u_analytic": u_an.tolist(),
    }


# ---------------------------------------------------------------------- case 3
def random_matrix(ny: int, nx: int, phi: float, seed: int, radius: int = 1) -> np.ndarray:
    """Random percolating matrix: dilated random sites, all four borders solid."""
    rng = np.random.default_rng(seed)
    p = 1.0 - phi
    solid = rng.random((ny, nx)) < p
    for _ in range(radius - 1):
        solid |= np.roll(solid, 1, 0) | np.roll(solid, 1, 1)
    solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
    return solid


def case3_percolation() -> dict:
    out = {"cases": []}
    g, tau, steps = 2e-5, 0.8, 6000
    for seed, ny, nx in [(11, 24, 24), (12, 32, 32)]:
        solid = random_matrix(ny, nx, phi=0.70, seed=seed)
        solid_t = torch.as_tensor(solid)
        table = build_index_table(solid_t, periodic=(True, True))
        f_s, hist_s = run_sparse(table, tau, g, steps, tol=1e-8)
        f_d, _ = run_dense(solid_t, tau, g, len(hist_s))
        _r, ux_s, _u = macroscopic_sparse(f_s, table, g, 0.0)
        _r, ux_d, _u = macroscopic_with_forcing(f_d, g, 0.0)
        nu = (tau - 0.5) / 3.0
        phi = table.porosity
        k_s = darcy_k(float(ux_s.mean()), nu, g)
        k_d = darcy_k(float(ux_d[~solid_t].mean()) * phi, nu, g)
        fp = table.state_bytes()
        out["cases"].append({
            "seed": seed, "grid": [ny, nx], "porosity": phi, "steps": len(hist_s),
            "k_super_sparse": k_s, "k_intrinsic_sparse": intrinsic_permeability(k_s, phi),
            "k_super_dense": k_d, "rel_diff_sparse_vs_dense": abs(k_s - k_d) / k_s,
            "max_abs_pdf_diff": float((to_dense(f_s, table)[:, ~solid_t]
                                       - f_d[:, ~solid_t]).abs().max()),
            "u_profile": ux_s.numpy().reshape(-1).tolist(),
            "footprint": fp,
        })
    RES["case3_percolation"] = out
    return out


# ------------------------------------------------------------------- main flows
def main() -> int:
    t0 = time.time()
    RES["meta"] = {
        "xval": "XVAL-LBM-LIST-BASED-SPARSE-LBM-PLUS-PERMEABILITY",
        "oss": "lssfau/walberla (list-based-sparse-lbm-plus-permeability), "
               "reference arm = lbmpy 2.0 + pystencils generator (waLBerla's sparse code path)",
        "python": sys.version.split()[0], "torch": torch.__version__,
        "numpy": np.__version__,
        "dtype": "float64", "threads": 1,
    }
    import oss_ref as O

    RES["oss_probe"] = O.probe_periodic_support(8)
    case1_kernel()

    # case 2: resolution ladder + tau sensitivity at H = 16
    ladder = [case2_channel(H) for H in (8, 16, 32)]
    RES["case2_channel"] = {"ladder": ladder}
    RES["sensitivity"] = {"tau": [case2_channel(16, tau=t) for t in (0.6, 0.8, 1.0, 1.5)]}

    case3_percolation()

    # footprint of the two layouts on the percolation case
    solid = random_matrix(32, 32, 0.70, 11)
    table = build_index_table(torch.as_tensor(solid), periodic=(True, True))
    RES["footprint"] = {"percolation_32x32": table.state_bytes(index_int=4)}

    c1 = RES["case1_kernel"]
    c2 = RES["case2_channel"]["ladder"]
    c3 = RES["case3_percolation"]["cases"]
    crit = {
        "c1_representation_neutrality_lt_1e-12": c1["sparse_vs_dense_max_abs"] < 1e-12,
        "c2_address_table_equivalence_100pct": (
            RES["address_table"]["neighbour_cell_agreement"] == 1.0
            and RES["address_table"]["direction_agreement"] == 1.0),
        "c3_oss_kernel_agreement_lt_1e-4": c1["sparse_vs_oss_max_abs"] < 1e-4,
        "c4_channel_k_lt_15pct": all(abs(r["err_vs_analytic"]) < 0.15 for r in c2),
        "c5_sparse_vs_dense_k_lt_1pct": (c1["k_rel_diff_sparse_vs_dense"] < 0.01
                                        and all(r["rel_diff_sparse_vs_dense"] < 0.01 for r in c3)),
    }
    RES["verdict"] = {"criteria": crit, "all_pass": all(crit.values()),
                      "runtime_seconds": time.time() - t0}
    with open(os.path.join(HERE, "results.json"), "w") as fh:
        json.dump(RES, fh, indent=1)
    print(json.dumps(RES["verdict"], indent=1))
    print(json.dumps({k: v for k, v in c1.items() if not isinstance(v, list)}, indent=1))
    for r in c2:
        print("H=%d k_int=%.6g analytic=%.6g err=%+.3f%% dense_err=%+.3f%% steps=%d"
              % (r["H"], r["k_intrinsic"], r["k_analytic"], 100 * r["err_vs_analytic"],
                 100 * r["err_dense_vs_analytic"], r["steps"]))
    for r in c3:
        print("perc %s k_super=%.6g dense=%.6g rel=%.2e" % (r["grid"], r["k_super_sparse"],
                                                           r["k_super_dense"],
                                                           r["rel_diff_sparse_vs_dense"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())