"""Runner: XVAL-LBM-INDIRECT-ADDRESSING-WALL-TABLES (mission 17863).

Arms (see predeclared.json, written before this script was run):

* ``channel_A1``  table-driven D3Q19 channel Poiseuille (indirect addressing)
* ``channel_A2``  dense-mask control of the same case (addressing is the only
                  changed variable)
* ``normal``      ISO8 wall normal, projected vs OSS-literal, on a plane wall
                  and on a sphere
* ``continuation`` la_weight continuation semantics against the exact
                  parabolic channel profile
* ``footprint``   table bytes / build time / per-step boundary-op time

Usage::

    python xval/XVAL-LBM-INDIRECT-ADDRESSING-WALL-TABLES/run_arm.py \
        --out artifacts/xval_indirect_addressing_wall_tables

Nothing here is tuned after looking at a KPI: step budgets, resolutions and
thresholds are the predeclared ones.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.d3q19 import C, OPPOSITE, W_EXACT64  # noqa: E402
from tensorlbm.wall_tables import (  # noqa: E402
    build_wall_tables,
    iso8_wall_normal,
    la_weight_extrapolate,
    table_footprint_bytes,
    dense_footprint_bytes,
)

torch.set_num_threads(1)
DTYPE = torch.float64
C64 = C.to(DTYPE)
W64 = W_EXACT64.to(DTYPE)
Q = int(C.shape[0])
STEPS_MAX = {8: 4000, 16: 5000}
STEADY_TOL = 1e-6
STEADY_WINDOW = 100


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def channel_geometry(height: int, nx: int = 2, nz: int = 2) -> torch.Tensor:
    """Wall-bounded channel: solid rows j=0 and j=H+1, periodic x and z."""
    walls = torch.zeros(nz, height + 2, nx, dtype=torch.bool)
    walls[:, 0, :] = True
    walls[:, -1, :] = True
    return walls


def sphere_geometry(n: int, radius: float | None = None) -> torch.Tensor:
    centre = (n - 1) / 2.0
    radius = n / 4.0 if radius is None else radius
    z, y, x = torch.meshgrid(
        torch.arange(n, dtype=DTYPE), torch.arange(n, dtype=DTYPE), torch.arange(n, dtype=DTYPE), indexing="ij"
    )
    return (z - centre) ** 2 + (y - centre) ** 2 + (x - centre) ** 2 < radius * radius


def _shift(field: torch.Tensor, offset, fill: float) -> torch.Tensor:
    """``out[x] = field[x + offset]`` with constant fill outside (dense control)."""
    o = tuple(int(v) for v in offset)
    pad = max(abs(v) for v in o)
    pads = []
    for _ in range(3):
        pads.extend((pad, pad))
    padded = torch.nn.functional.pad(field.to(DTYPE), pads, mode="constant", value=float(fill))
    n0, n1, n2 = (int(s) for s in field.shape)
    return padded[pad + o[0]: pad + o[0] + n0, pad + o[1]: pad + o[1] + n1, pad + o[2]: pad + o[2] + n2]


def _nb_periodic(walls: torch.Tensor, d: int) -> torch.Tensor:
    """Solidity of the neighbour ``x + c_d`` with periodic wrap on every axis.

    The channel is periodic in x and z and bounded by solid rows in y, so the
    wrap cannot move a *fluid* cell across a wall; this is the dense twin of
    ``build_wall_tables(..., periodic_axes=(True, False, True))``.
    """
    shift = (int(C[d, 0]), int(C[d, 1]), int(C[d, 2]))
    return torch.roll(walls, shifts=(-shift[0], -shift[1], -shift[2]), dims=(0, 1, 2))


# --------------------------------------------------------------------------- #
# D3Q19 BGK + Guo forcing, half-way bounce-back, two addressing routes
# --------------------------------------------------------------------------- #
def equilibrium(rho: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    cu = (C64[:, 0, None, None, None] * u[0] + C64[:, 1, None, None, None] * u[1]
          + C64[:, 2, None, None, None] * u[2])
    u2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2]
    return rho.unsqueeze(0) * W64.view(Q, 1, 1, 1) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2.unsqueeze(0))


def stream_pull(f: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(f)
    for i in range(Q):
        out[i] = torch.roll(f[i], shifts=(int(C[i, 0]), int(C[i, 1]), int(C[i, 2])), dims=(0, 1, 2))
    return out


def run_channel(height: int, tau: float, *, addressing: str, g: float = 1e-5,
                nx: int = 2, nz: int = 2, steps_max: int | None = None) -> dict:
    """Plane-channel Poiseuille, body force ``g`` along x.

    ``addressing`` selects the boundary route: ``"table"`` uses the
    indirect-addressing link table, ``"dense"`` rebuilds the same operations
    from a full-domain mask scan.
    """
    if addressing not in ("table", "dense"):
        raise ValueError("addressing must be 'table' or 'dense'")
    if steps_max is None:
        steps_max = STEPS_MAX.get(height, 8000)
    walls = channel_geometry(height, nx=nx, nz=nz)
    shape = walls.shape
    fluid = ~walls
    nu = (tau - 0.5) / 3.0
    t_build = time.perf_counter()
    if addressing == "table":
        tables = build_wall_tables(walls, periodic_axes=(True, False, True))
        link_dir = tables.links.direction
        link_fluid = tables.links.fluid_flat
        n_links = len(tables.links)
        build_seconds = time.perf_counter() - t_build
    else:
        masks = [fluid & _nb_periodic(walls, d) for d in range(1, Q)]
        build_seconds = time.perf_counter() - t_build
        n_links = int(sum(int(m.sum()) for m in masks))

    rho = torch.ones(shape, dtype=DTYPE)
    u = torch.zeros(3, *shape, dtype=DTYPE)
    f = equilibrium(rho, u)
    # half-way bounce-back uses f_post, so the wall force is accumulated there
    wall_force_y = torch.zeros(2, dtype=DTYPE)
    steps_done = 0
    steady_at = None
    u_prev = u[0].clone()
    mass0 = float(f.sum())
    for step in range(1, steps_max + 1):
        rho = f.sum(0)
        mom = (f.unsqueeze(1) * C64.view(Q, 3, 1, 1, 1)).sum(0)
        u = mom / rho.unsqueeze(0)
        u[0] = u[0] + 0.5 * g / rho  # Guo half-force velocity shift
        feq = equilibrium(rho, u)
        cu = (C64[:, 0, None, None, None] * u[0] + C64[:, 1, None, None, None] * u[1]
              + C64[:, 2, None, None, None] * u[2])
        src = 3.0 * (C64[:, 0, None, None, None] - u[0].unsqueeze(0)) * g + 9.0 * cu * C64[:, 0, None, None, None] * g
        f_post = f - (f - feq) / tau + (1.0 - 0.5 / tau) * W64.view(Q, 1, 1, 1) * src
        f_new = stream_pull(f_post)
        if addressing == "table":
            vals = f_post.view(Q, -1)[link_dir, link_fluid]
            f_new.view(Q, -1)[OPPOSITE[link_dir], link_fluid] = vals
            # momentum-exchange wall force: 2 * f_post * c_d per link
            per_link = 2.0 * (vals - W64[link_dir]) * C64[link_dir, 1]
            j_of_solid = (tables.links.solid_flat // nx) % shape[1]
            wall_force_y[0] = per_link[(j_of_solid == 0)].sum()
            wall_force_y[1] = per_link[(j_of_solid == shape[1] - 1)].sum()
        else:
            for d in range(1, Q):
                m = masks[d - 1]
                f_new[OPPOSITE[d]][m] = f_post[d][m]
            side = torch.zeros(2, dtype=DTYPE)
            for d in range(1, Q):
                m = masks[d - 1]
                if not bool(m.any()):
                    continue
                jj = m.nonzero(as_tuple=False)[:, 1]
                contribution = 2.0 * C64[d, 1] * (f_post[d][m] - W64[d])
                side[0] += contribution[jj == 0].sum()
                side[1] += contribution[jj == shape[1] - 1].sum()
            wall_force_y = side
        f = f_new
        steps_done = step
        if step % STEADY_WINDOW == 0:
            delta = float(((u[0] - u_prev).abs().max()) / max(float(u[0].abs().max()), 1e-30))
            u_prev = u[0].clone()
            if delta < STEADY_TOL and step > 4 * STEADY_WINDOW:
                steady_at = step
                break

    # ---- addressing identity on the final state -------------------------
    dense_masks = [fluid & _nb_periodic(walls, d) for d in range(1, Q)]
    f_table = stream_pull(f_post)
    f_table.view(Q, -1)[OPPOSITE[link_dir], link_fluid] = f_post.view(Q, -1)[link_dir, link_fluid]
    f_dense = stream_pull(f_post)
    for d in range(1, Q):
        m = dense_masks[d - 1]
        f_dense[OPPOSITE[d]][m] = f_post[d][m]
    identity_max_abs = float((f_table - f_dense).abs().max())
    force_dense = torch.zeros(2, dtype=DTYPE)
    for d in range(1, Q):
        m = dense_masks[d - 1]
        if not bool(m.any()):
            continue
        jj = m.nonzero(as_tuple=False)[:, 1]
        contribution = 2.0 * C64[d, 1] * (f_post[d][m] - W64[d])
        force_dense[0] += contribution[jj == 0].sum()
        force_dense[1] += contribution[jj == shape[1] - 1].sum()

    # ---- KPIs -----------------------------------------------------------
    ux = u[0]
    h_total = shape[1]
    u_max = float(ux[fluid].max())
    u_all = float(ux.mean())  # superficial (all cells of the cross-section)
    u_analytic_max = g * height * height / (8.0 * nu)
    k_super = nu * u_all / g
    k_ref = (height / h_total) * height * height / 12.0
    tau_w = float(wall_force_y.abs().mean()) / (nx * nz)
    # independent wall-shear route: first-order wall gradient from the first
    # off-wall fluid row (half-way bounce-back places the wall 0.5 dx outside it)
    u_first = float(ux[:, 1, :].mean())
    tau_w_profile = nu * u_first / 0.5
    tau_ref = g * height / 2.0
    mass_drift = float((rho[fluid] - 1.0).abs().max())
    ma_max = float(u.norm(dim=0).max() / math.sqrt(1.0 / 3.0))
    y = torch.arange(shape[1], dtype=DTYPE) - 0.5
    u_exact = g * y * (height - y) / (2.0 * nu)
    profile_err = float((ux[:, 1:-1, :].mean(dim=(0, 2)) - u_exact[1:-1]).abs().max() / u_analytic_max)
    return {
        "arm": f"channel_{'A1' if addressing == 'table' else 'A2'}_H{height}_tau{tau}",
        "addressing": addressing,
        "height": height, "tau": tau, "nu": nu, "g": g, "nx": nx, "nz": nz,
        "steps": steps_done, "steady_at": steady_at,
        "n_links": n_links, "build_seconds": build_seconds,
        "u_max": u_max, "u_max_analytic": u_analytic_max,
        "u_max_rel_err": abs(u_max - u_analytic_max) / u_analytic_max,
        "k_super": k_super, "k_super_ref": k_ref,
        "k_super_rel_err": abs(k_super - k_ref) / k_ref,
        "tau_w": tau_w, "tau_w_ref": tau_ref,
        "tau_w_rel_err": abs(tau_w - tau_ref) / tau_ref,
        "tau_w_profile": tau_w_profile,
        "tau_w_profile_rel_err": abs(tau_w_profile - tau_ref) / tau_ref,
        "momentum_balance_rel": abs(g * height - 2.0 * tau_w) / (g * height),
        "mass_drift": mass_drift, "mass_total_rel": abs(float(f.sum()) - mass0) / mass0,
        "ma_max": ma_max,
        "profile_max_rel_err": profile_err,
        "profile_y": y[1:-1].tolist(),
        "profile_u": ux[:, 1:-1, :].mean(dim=(0, 2)).tolist(),
        "profile_exact": u_exact[1:-1].tolist(),
        "identity_table_vs_dense_max_abs": identity_max_abs,
        "identity_table_vs_dense_rel": identity_max_abs / max(float(f_table.abs().max()), 1e-300),
        "wall_force_table": wall_force_y.tolist(),
        "wall_force_dense": force_dense.tolist(),
        "wall_force_rel_diff": float((wall_force_y - force_dense).abs().max() / max(float(force_dense.abs().max()), 1e-300)),
    }


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #
def normal_arm() -> dict:
    plane = torch.zeros(20, 12, 12, dtype=torch.bool)
    plane[0] = True
    out = {"plane_wall": {}, "sphere": {}}
    for variant in ("projected", "oss_literal"):
        n = iso8_wall_normal(plane, periodic_axes=(False, True, True), variant=variant)
        got = n[1, 6, 6].tolist()
        out["plane_wall"][variant] = {
            "normal_at_first_fluid_node": got,
            "abs_dev_from_exact": float(torch.tensor(got).sub(torch.tensor([-1.0, 0.0, 0.0])).abs().max()),
            "angle_deg": math.degrees(math.acos(min(1.0, abs(got[0]) / math.sqrt(sum(v * v for v in got))))),
        }
    n_cells = 48
    walls = sphere_geometry(n_cells)
    tables = {}
    for variant in ("projected", "oss_literal"):
        t0 = time.perf_counter()
        tables[variant] = build_wall_tables(walls, periodic_axes=(True, True, True), normal_variant=variant)
        build_s = time.perf_counter() - t0
        normals = tables[variant].fluid_boundary.normal
        centre = (n_cells - 1) / 2.0
        rad = tables[variant].fluid_boundary.ijk.to(DTYPE) - centre
        rad = rad / rad.norm(dim=1, keepdim=True)
        cos = -(normals * rad).sum(dim=1)  # normals point inward for a solid blob
        out["sphere"][variant] = {
            "n_fluid_boundary": int(len(normals)),
            "build_seconds": build_s,
            "cos_mean": float(cos.mean()),
            "cos_median": float(cos.median()),
            "cos_p05": float(cos.quantile(0.05)),
            "cos_min": float(cos.min()),
            "cos_hist": torch.histc(cos, bins=20, min=-1.0, max=1.0).tolist(),
            "angle_mean_deg": float(torch.rad2deg(torch.arccos(cos.clamp(-1, 1))).mean()),
            "angle_max_deg": float(torch.rad2deg(torch.arccos(cos.clamp(-1, 1))).max()),
        }
    out["sphere"]["n_cells"] = n_cells
    out["sphere"]["radius"] = n_cells / 4.0
    return out


def continuation_arm(channel: dict) -> dict:
    """la_weight continuation vs the exact parabolic profile of the H=32 run."""
    height = channel["height"]
    g, nu = channel["g"], channel["nu"]
    walls = channel_geometry(height)
    tables = build_wall_tables(walls, periodic_axes=(True, False, True))
    ux = torch.zeros(walls.shape, dtype=DTYPE)
    # rebuild the converged field from the stored profile (mean over x,z planes)
    prof = torch.tensor(channel["profile_u"], dtype=DTYPE)
    ux[:, 1:-1, :] = prof.view(1, -1, 1)
    out = la_weight_extrapolate(ux, tables, periodic_axes=(True, False, True))
    sb = tables.solid_boundary
    rows = sb.neighbor_rows()
    y_coord = torch.arange(walls.shape[1], dtype=DTYPE) - 0.5  # half-way wall at y=0

    def exact(yv: float) -> float:
        return g * yv * (height - yv) / (2.0 * nu)

    node_err, centroid_err, offsets, wall_values = [], [], [], []
    for row in range(len(sb)):
        i, j, k = (int(v) for v in sb.ijk[row].tolist())
        if j not in (0, walls.shape[1] - 1):
            continue
        weight = 0.0
        moment = 0.0
        for d in rows[row].tolist():
            if d < 0:
                continue
            w = float(W_EXACT64[d])
            weight += w
            moment += w * (j + int(C[d, 1]))
        y_c = moment / weight - 0.5
        node_err.append(abs(float(out[i, j, k]) - exact(float(y_coord[j]))) / abs(exact(float(y_coord[j]))))
        centroid_err.append(abs(float(out[i, j, k]) - exact(y_c)) / max(abs(exact(y_c)), 1e-300))
        offsets.append(y_c - float(y_coord[j]))
        wall_values.append(float(out[i, j, k]))
    return {
        "height": height,
        "n_wall_nodes": len(node_err),
        "output_vs_exact_at_node_rel": max(node_err),
        "output_vs_exact_at_centroid_rel": max(centroid_err),
        "centroid_offset_lattice_units_max": max(offsets),
        "centroid_offset_lattice_units_mean": sum(offsets) / len(offsets),
        "la_weight_output_mean": sum(wall_values) / len(wall_values),
        "exact_at_node_mean": exact(float(y_coord[0])),
        "exact_at_wall_mean": 0.0,
        "note": "u_exact at the solid node (y=-0.5 dx) is a linear extrapolation of the parabolic profile; the operator never claims to reproduce it",
    }


def footprint_arm() -> dict:
    out = {}
    for height in (8, 16, 32):
        walls = channel_geometry(height)
        t0 = time.perf_counter()
        tables = build_wall_tables(walls, periodic_axes=(True, False, True))
        build_s = time.perf_counter() - t0
        fp = table_footprint_bytes(tables)
        dense = dense_footprint_bytes(walls.shape)
        out[f"channel_H{height}"] = {
            "build_seconds": build_s,
            "n_solid_boundary": fp["n_solid_boundary"],
            "table_bytes": fp["bytes_total_implemented"],
            "oss_row_bytes": fp["bytes_total_mf_lbm_equivalent"],
            "dense_mask_plus_q_bytes": dense["bytes_mask_plus_q"],
            "ratio_table_over_dense": fp["bytes_total_implemented"] / dense["bytes_mask_plus_q"],
        }
    walls = sphere_geometry(48)
    t0 = time.perf_counter()
    tables = build_wall_tables(walls, periodic_axes=(True, True, True))
    build_s = time.perf_counter() - t0
    fp = table_footprint_bytes(tables)
    dense = dense_footprint_bytes(walls.shape)
    out["sphere_48"] = {
        "build_seconds": build_s,
        "n_solid_boundary": fp["n_solid_boundary"],
        "n_fluid_boundary": fp["n_fluid_boundary"],
        "n_links": len(tables.links),
        "table_bytes": fp["bytes_total_implemented"],
        "oss_row_bytes": fp["bytes_total_mf_lbm_equivalent"],
        "dense_mask_plus_q_bytes": dense["bytes_mask_plus_q"],
        "ratio_table_over_dense": fp["bytes_total_implemented"] / dense["bytes_mask_plus_q"],
        "ratio_table_over_oss_rows": fp["bytes_total_implemented"] / fp["bytes_total_mf_lbm_equivalent"],
    }
    return out


def boundary_op_timing_arm(repeats: int = 200) -> dict:
    out = {}
    walls = channel_geometry(16, nx=16, nz=16)
    fluid = ~walls
    tables = build_wall_tables(walls, periodic_axes=(True, False, True))
    f = torch.rand(Q, *walls.shape, dtype=DTYPE)
    idx = tables.links.fluid_flat
    d = tables.links.direction
    # table traversal
    t0 = time.perf_counter()
    for _ in range(repeats):
        f.view(Q, -1)[OPPOSITE[d], idx] = f.view(Q, -1)[d, idx]
    table_s = (time.perf_counter() - t0) / repeats
    # dense scan
    masks = [fluid & (_shift(walls, C[i].tolist(), fill=1.0) > 0.5) for i in range(1, Q)]
    g = torch.rand(Q, *walls.shape, dtype=DTYPE)
    t0 = time.perf_counter()
    for _ in range(repeats):
        for i in range(1, Q):
            m = masks[i - 1]
            g[OPPOSITE[i]][m] = f[i][m]
    dense_s = (time.perf_counter() - t0) / repeats
    out["channel_16x18x16"] = {
        "n_cells": int(walls.numel()),
        "n_links": len(tables.links),
        "table_seconds": table_s,
        "dense_seconds": dense_s,
        "speedup": dense_s / table_s,
        "repeats": repeats,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/xval_indirect_addressing_wall_tables")
    args = ap.parse_args()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {"predeclared": str(Path("xval/XVAL-LBM-INDIRECT-ADDRESSING-WALL-TABLES/predeclared.json")),
                     "arms": {}}

    # --- channel physics + representation identity -----------------------
    for height, routes in ((8, ("table",)), (16, ("table",))):
        for addressing in routes:
            res = run_channel(height, 0.8, addressing=addressing)
            results["arms"][res["arm"]] = res
            print(f"[{res['arm']}] u_max_err={res['u_max_rel_err']:.4e} k_err={res['k_super_rel_err']:.4e} "
                  f"tau_err={res['tau_w_rel_err']:.4e} steps={res['steps']}")
    for tau in (0.6, 1.0):
        res = run_channel(16, tau, addressing="table")
        results["arms"][res["arm"]] = res
        print(f"[{res['arm']}] u_max_err={res['u_max_rel_err']:.4e} k_err={res['k_super_rel_err']:.4e}")

    (out_dir / "results_partial_physics.json").write_text(json.dumps(results, indent=2))
    sys.stdout.flush()

    # --- identity of the two addressing routes ---------------------------
    identity = {}
    for height in (8, 16):
        a = results["arms"][f"channel_A1_H{height}_tau0.8"]
        identity[f"H{height}"] = {
            "streamed_field_max_abs_diff": a["identity_table_vs_dense_max_abs"],
            "streamed_field_rel_diff": a["identity_table_vs_dense_rel"],
            "wall_force_rel_diff": a["wall_force_rel_diff"],
            "bitwise_identical": a["identity_table_vs_dense_max_abs"] == 0.0 and a["wall_force_rel_diff"] == 0.0,
            "within_predeclared_1e_12": a["identity_table_vs_dense_rel"] <= 1e-12 and a["wall_force_rel_diff"] <= 1e-12,
        }
    results["identity_A1_vs_A2"] = identity
    print("[identity]", json.dumps({k: v["bitwise_identical"] for k, v in identity.items()}))

    # --- normal arms -----------------------------------------------------
    results["normal"] = normal_arm()
    print("[normal]", json.dumps({v: round(results["normal"]["sphere"][v]["cos_mean"], 4)
                                  for v in ("projected", "oss_literal")}))

    # --- continuation semantics -----------------------------------------
    results["continuation"] = continuation_arm(results["arms"]["channel_A1_H16_tau0.8"])
    print("[continuation]", json.dumps({k: results["continuation"][k] for k in
                                        ("output_vs_exact_at_node_rel", "output_vs_exact_at_centroid_rel",
                                         "centroid_offset_lattice_units_mean")}))

    # --- footprint / timing ---------------------------------------------
    results["footprint"] = footprint_arm()
    results["boundary_op_timing"] = boundary_op_timing_arm()
    print("[footprint]", json.dumps(results["footprint"]["sphere_48"]))

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    # --- flat KPI table --------------------------------------------------
    rows = ["arm,height,tau,steps,u_max_rel_err,k_super_rel_err,tau_w_rel_err,momentum_balance_rel,"
            "mass_drift,ma_max,n_links,build_seconds"]
    for name, res in results["arms"].items():
        rows.append(
            f"{name},{res['height']},{res['tau']},{res['steps']},{res['u_max_rel_err']:.6e},"
            f"{res['k_super_rel_err']:.6e},{res['tau_w_rel_err']:.6e},{res['momentum_balance_rel']:.6e},"
            f"{res['mass_drift']:.6e},{res['ma_max']:.6e},{res['n_links']},{res['build_seconds']:.3f}"
        )
    (out_dir / "table.csv").write_text("\n".join(rows) + "\n")
    print(f"wrote {out_dir/'results.json'} and {out_dir/'table.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())