"""Round-2 cross-validation for XVAL-LBM-SPARSE-LIST-BASED-AOS-LBM (D3Q19 + interior solids).

Round 1 validated the D2Q9 list-based AoS port against ``lbmpy.sparse`` and two
analytic references.  Round 2 asks the two questions round 1 left open:

* does the port survive the jump to the production stencil (D3Q19, 3-D), and
* does the *sparse* part -- solid cells that are not stored at all -- still give
  the right no-slip physics when the solid sits **inside** the fluid list instead
  of on the domain border?

Cases (see ``predeclared.json``, written before any run):

  P1  3-D plane Poiseuille (walls in y, periodic x/z)      exact parabola
  P2  3-D rectangular duct (walls in y and z)              White duct series
  P3  3-D shear-wave decay  u = A sin(k z) exp(-nu k^2 t)  exact NS solution
  P4  periodic array of *interior* solid slabs (no border walls at all)
                                                          exact channel parabola
  P5  arm equivalence dense / list-SoA / list-AoS / stencil order / OSS-driven
  P6  index array vs ``lbmpy.sparse.SparseLbMapper.create_index_array`` (D3Q19)
  P7  storage cost + throughput vs solid fraction

Run:  PYTHONPATH=src python development/verification/sparse_list_aos/round2_d3q19/run_r2.py
Artifacts: results.json, tables.md (both written by this script, no hand copying).
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from tensorlbm.sparse_list_aos import (
    C_D2Q9,
    C_D3Q19,
    C_D3Q19_LBMPY,
    FLUID,
    GHOST,
    WALL,
    SparseListLBM,
    build_index_array,
    dense_collide_stream,
    duct_poiseuille_analytic,
    equilibrium_vec,
    poiseuille_analytic,
    velocity_from_vec,
)

HERE = Path(__file__).resolve().parent
TAU = 0.8
NU = (TAU - 0.5) / 3.0
U_MAX = 0.03
MISS = float("nan")


# --------------------------------------------------------------------------- #
# geometry builders (ghost layer of 1 everywhere, as in lbmpy's sparse mapper)
# --------------------------------------------------------------------------- #
def flags_plane_channel(nx: int, height: int, nz: int) -> np.ndarray:
    """Fluid box, no-slip wall planes in y, periodic in x and z."""
    flags = np.full((nx + 2, height + 2, nz + 2), GHOST, dtype=np.uint32)
    flags[1:-1, 1:-1, 1:-1] = FLUID
    flags[:, 0, :] = WALL
    flags[:, -1, :] = WALL
    return flags


def flags_duct(nx: int, hy: int, hz: int) -> np.ndarray:
    """Fluid box, no-slip wall planes in y *and* z, periodic in x only."""
    flags = np.full((nx + 2, hy + 2, hz + 2), GHOST, dtype=np.uint32)
    flags[1:-1, 1:-1, 1:-1] = FLUID
    flags[:, 0, :] = WALL
    flags[:, -1, :] = WALL
    flags[:, :, 0] = WALL
    flags[:, :, -1] = WALL
    return flags


def flags_box(n: tuple[int, int, int]) -> np.ndarray:
    flags = np.full(tuple(n_i + 2 for n_i in n), GHOST, dtype=np.uint32)
    flags[1:-1, 1:-1, 1:-1] = FLUID
    return flags


def flags_interior_slabs(nx: int, channels: int, height_per_channel: int, nz: int):
    """Fully periodic box with ``channels`` solid slabs *inside* the fluid list.

    ``channels`` equal-height channels are carved out of one periodic domain by
    solid slab planes; because the slabs are solid cells (not ghost/border cells)
    every channel wall is handled by the no-slip index aliasing, and the domain
    itself has no wall plane at all.  Channel ``k`` spans fluid rows
    ``1 + k*(Hc+1) .. k*(Hc+1) + Hc`` of the interior box.
    """
    hc = height_per_channel
    ny = channels * (hc + 1)                     # interior fluid rows per period
    flags = np.full((nx + 2, ny + 2, nz + 2), GHOST, dtype=np.uint32)
    flags[1:-1, 1:-1, 1:-1] = FLUID
    # slab rows: first slab at row 2 so that the periodic wrap of row 1 lands on
    # fluid (the ghost row wraps onto the *last* interior row, which must not be solid)
    slab_rows = [2 + k * (hc + 1) for k in range(channels)]
    for r in slab_rows:
        flags[:, r, :] = WALL
    # each channel is returned as its list of (row, channel-local index j) pairs
    chan_rows = []
    for k, r in enumerate(slab_rows):
        nxt = slab_rows[k + 1] if k + 1 < len(slab_rows) else ny + 1
        rows = [(y - 1, y - r - 1) for y in range(r + 1, min(nxt, ny + 1))]
        if k + 1 == len(slab_rows):              # the wrapping channel continues
            rows += [(y - 1, y + ny - r - 1) for y in range(1, slab_rows[0])]
        chan_rows.append(rows)
    return flags, chan_rows, ny


# --------------------------------------------------------------------------- #
# marching helpers
# --------------------------------------------------------------------------- #
def march_sparse(sp: SparseListLBM, u_max_ref: float, tol=1e-10, cap=200_000, check=200):
    u_ref = u_max_ref
    """Advance until max|du_x|/u_ref < tol, return (steps, achieved drift, drift history)."""
    prev = None
    steps = 0
    drift = float("inf")
    history = []
    while steps < cap:
        for _ in range(check):
            sp.step()
        steps += check
        ux = sp.macroscopic()[1]
        if prev is not None:
            drift = float(np.max(np.abs(ux - prev))) / u_ref
            history.append([steps, drift])
            if drift < tol:
                break
        prev = ux.copy()
    return steps, drift, history


def dense_run(shape, tau, force, periodic, steps, stencil=None, tol=None, check=200, u_ref=U_MAX):
    """Dense control arm from rest; returns (f, steps, achieved drift)."""
    q = (C_D2Q9 if stencil is None else np.asarray(stencil)).shape[0]
    n = int(np.prod(shape))
    f0 = equilibrium_vec(np.ones(n), np.zeros((n, 3 if len(shape) == 3 else 2)),
                         stencil).reshape(tuple(shape) + (q,))
    f = np.moveaxis(f0, -1, 0).copy()
    done = 0
    prev = None
    drift = float("inf")
    while done < steps:
        block = check if tol is not None else steps - done
        if tol is not None:
            block = min(check, steps - done)
        for _ in range(block):
            f = dense_collide_stream(f, tau=tau, force=force, periodic=periodic, stencil=stencil)
        done += block
        if tol is not None:
            u = velocity_from_vec(np.moveaxis(f, 0, -1).reshape(-1, q), force, stencil)[1][:, 0]
            if prev is not None:
                drift = float(np.max(np.abs(u - prev))) / u_ref
                if drift < tol:
                    break
            prev = u.copy()
    return f, done, drift


def dense_velocity(f, force, shape, stencil=None):
    q = f.shape[0]
    rho, u = velocity_from_vec(np.moveaxis(f, 0, -1).reshape(-1, q), force, stencil)
    return rho.reshape(shape), u.reshape(tuple(shape) + (u.shape[-1],))


def sparse_velocity(sp: SparseListLBM, shape):
    rho, *comp = sp.macroscopic()
    return rho, np.stack(comp, axis=-1), sp.coords.astype(int) - 1


def rel_errors(got: np.ndarray, ref: np.ndarray) -> dict:
    diff = np.abs(got - ref)
    scale = float(np.max(np.abs(ref)))
    return {
        "linf_abs": float(np.max(diff)),
        "linf_rel": float(np.max(diff) / scale) if scale else MISS,
        "l2_rel": float(np.sqrt(np.mean(diff**2)) / scale) if scale else MISS,
    }


def force_for_umax_plane(height: int, umax: float = U_MAX) -> float:
    return 8.0 * NU * umax / height**2


# --------------------------------------------------------------------------- #
# P1: 3-D plane Poiseuille
# --------------------------------------------------------------------------- #
def case_plane_poiseuille(heights=(8, 16, 32), nx=2, nz=8):
    heights = tuple(int(v) for v in os.environ.get("R2_HEIGHTS", ",".join(map(str, heights))).split(","))
    out = []
    for h in heights:
        g = force_for_umax_plane(h)
        flags = flags_plane_channel(nx, h, nz)
        ref_line = poiseuille_analytic(np.arange(h, dtype=float), h, g, NU)
        rec = {
            "case": "P1_plane_poiseuille_3d", "height": h, "nx": nx, "nz": nz,
            "tau": TAU, "nu": NU, "force_x": g,
            "u_max_ref": float(np.max(ref_line)),
            "Re": float(np.max(ref_line) * h / NU),
            "Ma": float(np.max(ref_line) / np.sqrt(1.0 / 3.0)),
            "cells_fluid": int(np.prod(flags.shape)),
            "arms": {},
        }
        flags_cells = int(np.count_nonzero(flags & FLUID))
        rec["cells_fluid"] = flags_cells
        for layout in ("soa", "aos"):
            sp = SparseListLBM(flags, tau=TAU, force=(g, 0.0, 0.0), layout=layout,
                               stencil=C_D3Q19)
            steps, drift, hist = march_sparse(sp, u_max_ref=ref_line.max())
            rho, u, coords = sparse_velocity(sp, flags.shape)
            prof = np.array([u[coords[:, 1] == y, 0].mean() for y in range(h)])
            err = rel_errors(prof, ref_line)
            rec["arms"][f"sparse_{layout}"] = {
                "steps": steps, "drift": drift, "profile": prof.tolist(),
                "err": err, "mass_rel": float(abs(rho.sum() - len(rho)) / len(rho)),
                "converged": bool(drift < 1e-10),
            }
        # dense control arm at the same discrete scheme
        f, steps_d, drift_d = dense_run((nx, h, nz), TAU, (g, 0.0, 0.0), (True, False, True),
                                        cap := 200_000, stencil=C_D3Q19, tol=1e-10, u_ref=ref_line.max())
        _, u_d = dense_velocity(f, (g, 0.0, 0.0), (nx, h, nz), C_D3Q19)
        prof_d = u_d[:, :, :, 0].mean(axis=(0, 2))
        rec["arms"]["dense"] = {
            "steps": steps_d, "drift": drift_d, "profile": prof_d.tolist(),
            "err": rel_errors(prof_d, ref_line), "converged": bool(drift_d < 1e-10),
        }
        # arm-to-arm equivalence on the full velocity field
        sp_soa = SparseListLBM(flags, tau=TAU, force=(g, 0.0, 0.0), layout="soa", stencil=C_D3Q19)
        sp_aos = SparseListLBM(flags, tau=TAU, force=(g, 0.0, 0.0), layout="aos", stencil=C_D3Q19)
        for _ in range(200):
            sp_soa.step(); sp_aos.step()
        _, u_soa, c_soa = sparse_velocity(sp_soa, flags.shape)
        _, u_aos, c_aos = sparse_velocity(sp_aos, flags.shape)
        same = np.array_equal(c_soa, c_aos)
        eq = {"soa_vs_aos": float(np.max(np.abs(u_soa - u_aos))) / ref_line.max(),
              "coords_equal": bool(same)}
        f2, _, _ = dense_run((nx, h, nz), TAU, (g, 0.0, 0.0), (True, False, True), 200,
                             stencil=C_D3Q19)
        _, u_d2 = dense_velocity(f2, (g, 0.0, 0.0), (nx, h, nz), C_D3Q19)
        eq["dense_vs_sparse_soa"] = float(np.max(np.abs(u_soa - u_d2[c_soa[:, 0], c_soa[:, 1], c_soa[:, 2]]))) / ref_line.max()
        rec["arm_equivalence_200_steps"] = eq
        out.append(rec)
    return out


# --------------------------------------------------------------------------- #
# P2: 3-D rectangular duct (two wall pairs, periodic only in x)
# --------------------------------------------------------------------------- #
def case_duct(pairs=((8, 12), (16, 24)), nx=2, tau=TAU):
    if os.environ.get("R2_DUCTS"):
        pairs = tuple(tuple(int(v) for v in pr.split("x")) for pr in os.environ["R2_DUCTS"].split(","))
    out = []
    for hy, hz in pairs:
        unit = duct_poiseuille_analytic(np.arange(hy)[:, None], np.arange(hz)[None, :], hy, hz, 1.0, 1.0)
        g = U_MAX / float(np.max(unit))
        flags = flags_duct(nx, hy, hz)
        ref = duct_poiseuille_analytic(np.arange(hy)[:, None], np.arange(hz)[None, :], hy, hz, g, NU)
        rec = {"case": "P2_duct_3d", "hy": hy, "hz": hz, "nx": nx, "tau": tau, "nu": NU,
               "force_x": g, "u_max_ref": float(np.max(ref)),
               "Re": float(np.max(ref) * hy / NU), "Ma": float(np.max(ref) / np.sqrt(1 / 3)),
               "cells_fluid": int(np.count_nonzero(flags & FLUID)), "arms": {}}
        for layout in ("soa", "aos"):
            sp = SparseListLBM(flags, tau=tau, force=(g, 0.0, 0.0), layout=layout, stencil=C_D3Q19)
            steps, drift, _ = march_sparse(sp, u_max_ref=rec["u_max_ref"])
            rho, u, coords = sparse_velocity(sp, flags.shape)
            got = np.empty((hy, hz))
            for iy in range(hy):
                for iz in range(hz):
                    sel = (coords[:, 1] == iy) & (coords[:, 2] == iz)
                    got[iy, iz] = u[sel, 0].mean()
            rec["arms"][f"sparse_{layout}"] = {
                "steps": steps, "drift": drift, "field": got.tolist(),
                "err": rel_errors(got, ref), "converged": bool(drift < 1e-10),
                "mass_rel": float(abs(rho.sum() - len(rho)) / len(rho)),
            }
        rec["ref_field"] = ref.tolist()
        out.append(rec)
    return out


# --------------------------------------------------------------------------- #
# P3: 3-D shear wave, exact Navier-Stokes solution u = A sin(k z) exp(-nu k^2 t)
# --------------------------------------------------------------------------- #
def case_shear_wave(sizes=(16, 32), amplitude=0.02, orient=("z", 0)):
    sizes = tuple(int(v) for v in os.environ.get("R2_SIZES", ",".join(map(str, sizes))).split(","))
    out = []
    axis, comp = orient
    for n in sizes:
        shape = (n, n, n)
        flags = flags_box(shape)
        k = 2.0 * np.pi / n
        omega_ref = NU * k**2
        t_e = 1.0 / omega_ref
        times = [round(i * (2.0 * t_e) / 4.0) for i in range(5)]
        coords3 = np.stack(np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij"), axis=-1)
        ax = {"x": 0, "y": 1, "z": 2}[axis]
        phase = k * coords3[..., ax]
        u0 = np.zeros((n**3, 3))
        u0[:, comp] = amplitude * np.sin(phase).ravel()
        rec = {"case": "P3_shear_wave_3d", "n": n, "k": k, "nu": NU, "tau": TAU,
               "omega_ref": omega_ref, "t_e": t_e, "times": times, "amplitude0": amplitude,
               "velocity_component": comp, "varies_along": axis, "arms": {}}

        def amplitude_of(field_ux):
            # projection on the sin mode, normalised so that A is returned exactly
            return float(2.0 * np.mean(field_ux.ravel() * np.sin(phase).ravel()))

        for layout in ("soa", "aos"):
            sp = SparseListLBM(flags, tau=TAU, force=(0.0, 0.0, 0.0), layout=layout, stencil=C_D3Q19)
            sp.reset(rho=1.0, u=u0)
            amps, done = [], 0
            for t in times:
                while done < t:
                    sp.step(); done += 1
                _, u, _ = sparse_velocity(sp, flags.shape)
                amps.append(amplitude_of(u[:, comp]))
            slope = np.polyfit(times, np.log(np.abs(amps)), 1)[0]
            rec["arms"][f"sparse_{layout}"] = {
                "amplitudes": amps, "omega_got": float(-slope),
                "rel_err": float(abs(-slope - omega_ref) / omega_ref),
                "mass_rel": float(abs(sp.populations().sum() / sp.n_cells - 1.0)),
            }
        f, done = None, 0
        f = np.moveaxis(
            equilibrium_vec(np.ones(n**3), u0, C_D3Q19).reshape(shape + (19,)), -1, 0
        ).copy()
        amps_d = []
        for t in times:
            while done < t:
                f = dense_collide_stream(f, tau=TAU, force=(0.0, 0.0, 0.0), periodic=(True, True, True), stencil=C_D3Q19)
                done += 1
            _, u_d = dense_velocity(f, (0.0, 0.0, 0.0), shape, C_D3Q19)
            amps_d.append(amplitude_of(u_d[..., comp]))
        slope_d = np.polyfit(times, np.log(np.abs(amps_d)), 1)[0]
        rec["arms"]["dense"] = {"amplitudes": amps_d, "omega_got": float(-slope_d),
                                "rel_err": float(abs(-slope_d - omega_ref) / omega_ref)}
        out.append(rec)
    return out


# --------------------------------------------------------------------------- #
# P4: interior solid slabs -- sparse-only capability, exact channel reference
# --------------------------------------------------------------------------- #
def case_interior_slabs(channels=2, hc=8, nx=2, nz=4):
    hc = int(os.environ.get("R2_HC", hc))
    flags, chan_rows, ny = flags_interior_slabs(nx, channels, hc, nz)
    g = force_for_umax_plane(hc)
    ref_line = poiseuille_analytic(np.arange(hc, dtype=float), hc, g, NU)
    rec = {"case": "P4_interior_slab_channels", "channels": channels, "channel_height": hc,
           "nx": nx, "nz": nz, "tau": TAU, "nu": NU, "force_x": g,
           "u_max_ref": float(ref_line.max()), "channel_rows": [[y for (y, j) in rows] for rows in chan_rows],
           "cells_fluid": int(np.count_nonzero(flags & FLUID)),
           "solid_fraction": float(1.0 - np.count_nonzero(flags & FLUID) /
                                   np.prod([c for c in (nx, ny, nz)])),
           "arms": {}}
    for layout in ("soa", "aos"):
        sp = SparseListLBM(flags, tau=TAU, force=(g, 0.0, 0.0), layout=layout, stencil=C_D3Q19)
        steps, drift, _ = march_sparse(sp, u_max_ref=ref_line.max())
        rho, u, coords = sparse_velocity(sp, flags.shape)
        errs, profs = [], []
        for rows in chan_rows:
            prof = np.array([u[coords[:, 1] == y, 0].mean() for (y, j) in rows])
            prof = prof[np.argsort([j for (y, j) in rows])]
            profs.append(prof.tolist())
            errs.append(rel_errors(prof, ref_line))
        rec["arms"][f"sparse_{layout}"] = {
            "steps": steps, "drift": drift, "profiles": profs, "err_per_channel": errs,
            "err": {k: float(max(e[k] for e in errs)) for k in errs[0]},
            "converged": bool(drift < 1e-10),
            "mass_rel": float(abs(rho.sum() - len(rho)) / len(rho)),
            # cells next to a slab must be non-zero (no-slip is a wall, not a lid)
            "u_at_wall_adjacent": [float(profs[i][0]) for i in range(len(profs))],
        }
    rec["ref_profile"] = ref_line.tolist()
    return rec


# --------------------------------------------------------------------------- #
# P5/P6: OSS index array in 3D, stencil-order invariance, layout-mismatch control
# --------------------------------------------------------------------------- #
def case_index_array_and_layouts():
    from lbmpy.sparse import SparseLbMapper
    from lbmpy.stencils import LBStencil

    flags = flags_duct(2, 8, 12)
    coords_std, arr_std = build_index_array(flags, layout="soa", stencil=C_D3Q19_LBMPY)
    coords_repo, arr_repo = build_index_array(flags, layout="soa", stencil=C_D3Q19)
    coords_aos, arr_aos = build_index_array(flags, layout="aos", stencil=C_D3Q19_LBMPY)

    mapper = SparseLbMapper(LBStencil("D3Q19"), np.asarray(flags, dtype=np.uint32), FLUID, WALL, 0)
    arr_oss = mapper.create_index_array()

    rec = {
        "case": "P6_index_array_3d",
        "oss": "lbmpy 2.0 lbmpy.sparse.SparseLbMapper.create_index_array (D3Q19)",
        "shape": list(arr_std.shape), "dtype": str(arr_std.dtype),
        "coords_equal_oss_order": bool(np.array_equal(coords_std, coords_repo)),
        "mismatches_vs_oss": int(np.count_nonzero(arr_std != arr_oss)),
        "max_abs_diff_vs_oss": int(np.max(np.abs(arr_std.astype(np.int64) - arr_oss.astype(np.int64)))),
        "oss_shape": list(arr_oss.shape),
        "soa_vs_aos_columns_differ": int(np.count_nonzero(arr_std != arr_aos)),
    }
    # stencil-order invariance of the physics (P5)
    g = force_for_umax_plane(8)
    flags_p = flags_plane_channel(2, 8, 4)
    u_lines = {}
    for name, st in (("repo_order", C_D3Q19), ("lbmpy_order", C_D3Q19_LBMPY)):
        sp = SparseListLBM(flags_p, tau=TAU, force=(g, 0.0, 0.0), layout="soa", stencil=st)
        march_sparse(sp, u_max_ref=U_MAX, tol=1e-9, cap=60_000)
        _, u, coords = sparse_velocity(sp, flags_p.shape)
        u_lines[name] = np.array([u[coords[:, 1] == y, 0].mean() for y in range(8)])
    rec["stencil_order_max_delta_over_umax"] = float(
        np.max(np.abs(u_lines["repo_order"] - u_lines["lbmpy_order"])) / U_MAX
    )

    # OSS array driving the TensorLBM list solver (correct layout: SoA)
    sp_oss = SparseListLBM(flags, tau=TAU, force=(g, 0.0, 0.0), layout="soa",
                           index_array=arr_oss, stencil=C_D3Q19_LBMPY)
    sp_mine = SparseListLBM(flags, tau=TAU, force=(g, 0.0, 0.0), layout="soa",
                            stencil=C_D3Q19_LBMPY)
    for _ in range(100):
        sp_oss.step(); sp_mine.step()
    _, u_o, c_o = sparse_velocity(sp_oss, flags.shape)
    _, u_m, c_m = sparse_velocity(sp_mine, flags.shape)
    rec["oss_array_driven_vs_ported"] = float(np.max(np.abs(u_o - u_m))) / max(U_MAX, 1e-30)

    # negative control (H4): OSS/SoA offsets injected into the AoS layout
    try:
        sp_bad = SparseListLBM(flags, tau=TAU, force=(g, 0.0, 0.0), layout="aos",
                               index_array=arr_oss, stencil=C_D3Q19_LBMPY)
        for _ in range(100):
            sp_bad.step()
        _, u_b, _ = sparse_velocity(sp_bad, flags.shape)
        dev = float(np.max(np.abs(u_b - u_m))) / max(U_MAX, 1e-30)
        if not np.isfinite(dev):
            rec["layout_mismatch_control"] = {"deviation": "non-finite", "detected": True}
        else:
            rec["layout_mismatch_control"] = {"deviation": dev, "detected": bool(dev > 1e-6)}
    except Exception as exc:  # an addressing bug may also blow up outright
        rec["layout_mismatch_control"] = {"deviation": f"exception: {type(exc).__name__}: {exc}",
                                          "detected": True}
    return rec


# --------------------------------------------------------------------------- #
# P7: storage cost and throughput vs solid fraction
# --------------------------------------------------------------------------- #
def case_storage(n=24, solid_sides=(0, 8, 12, 15, 18), steps=10):
    n = int(os.environ.get("R2_N", n))
    solid_sides = tuple(s for s in solid_sides if s <= n - 4) or (0,)
    out = []
    for s in solid_sides:
        flags = flags_box((n, n, n))
        if s > 0:
            lo = (n - s) // 2 + 1
            flags[lo:lo + s, lo:lo + s, lo:lo + s] = WALL
        n_fluid = int(np.count_nonzero(flags & FLUID))
        rec = {"case": "P7_storage", "n": n, "solid_side": s,
               "solid_fraction": float(1.0 - n_fluid / n**3),
               "cells_fluid": n_fluid, "cells_dense": int(n**3)}
        for layout in ("soa", "aos"):
            t0 = time.perf_counter()
            sp = SparseListLBM(flags, tau=TAU, force=(1e-6, 0.0, 0.0), layout=layout, stencil=C_D3Q19)
            t_build = time.perf_counter() - t0
            t0 = time.perf_counter()
            for _ in range(steps):
                sp.step()
            t_step = (time.perf_counter() - t0) / steps
            pdf_bytes = sp.pdf.nbytes
            idx_bytes = sp.index_array.nbytes + sp.coords.nbytes + sp._write_offsets.nbytes
            rec[layout] = {"build_s": t_build, "step_s": t_step,
                           "pdf_bytes": int(pdf_bytes), "aux_bytes": int(idx_bytes),
                           "total_bytes": int(pdf_bytes + idx_bytes)}
        dense = {"pdf_bytes": int(n**3 * 19 * 8)}
        rec["dense"] = dense
        rec["bytes_ratio_sparse_over_dense"] = rec["soa"]["pdf_bytes"] / dense["pdf_bytes"]
        rec["total_bytes_ratio_sparse_over_dense"] = rec["soa"]["total_bytes"] / dense["pdf_bytes"]
        # dense throughput on the same box for reference
        f = np.moveaxis(
            equilibrium_vec(np.ones(n**3), np.zeros((n**3, 3)), C_D3Q19).reshape((n, n, n, 19)),
            -1, 0).copy()
        t0 = time.perf_counter()
        for _ in range(steps):
            f = dense_collide_stream(f, tau=TAU, force=(1e-6, 0.0, 0.0), periodic=(True, True, True), stencil=C_D3Q19)
        rec["dense"]["step_s"] = (time.perf_counter() - t0) / steps
        out.append(rec)
    return out


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def fmt(x, nd=4):
    if isinstance(x, str):
        return x
    if x is None:
        return "-"
    if x != x:
        return "n/a"
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e4:
        return f"{x:.3e}"
    return f"{x:.{nd}f}"


def build_tables(res: dict) -> str:
    L = []
    L.append("# XVAL-LBM-SPARSE-LIST-BASED-AOS-LBM — round 2 (D3Q19 + interior solids)\n")
    L.append("Generated by `round2_d3q19/run_r2.py`; every number below is written by the script.\n")
    L.append(f"owner precision standard: <15 % · tau = {TAU} · nu = {NU:.6f} · float64 · single core\n")

    L.append("## P1 — 3-D plane Poiseuille (walls in y, periodic x/z), analytic parabola\n")
    L.append("| H | arm | steps | conv | u_max got | u_max ref | L∞/ref | L2 rel | mass | <15% |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in res["P1"]:
        ref = r["u_max_ref"]
        for arm in ("dense", "sparse_soa", "sparse_aos"):
            a = r["arms"][arm]
            umax_got = float(np.max(a["profile"]))
            ok = a["err"]["linf_rel"] < 0.15
            L.append(f"| {r['height']} | {arm} | {a['steps']} | {a['converged']} | "
                     f"{fmt(umax_got)} | {fmt(ref)} | {fmt(a['err']['linf_rel'])} | "
                     f"{fmt(a['err']['l2_rel'])} | {fmt(a.get('mass_rel'))} | {'OK' if ok else 'FAIL'} |")
    L.append("\n### P1 arm equivalence (200 steps, full field, max|du|/u_max)\n")
    L.append("| H | dense vs sparse-SoA | SoA vs AoS | coords equal |")
    L.append("|---|---|---|---|")
    for r in res["P1"]:
        e = r["arm_equivalence_200_steps"]
        L.append(f"| {r['height']} | {fmt(e['dense_vs_sparse_soa'])} | {fmt(e['soa_vs_aos'])} | {e['coords_equal']} |")

    L.append("\n## P2 — 3-D rectangular duct (walls in y and z), White duct Fourier series\n")
    L.append("| Hy x Hz | arm | steps | conv | u_max got | u_max ref | L∞/ref | L2 rel | mass | <15% |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in res["P2"]:
        for arm in ("sparse_soa", "sparse_aos"):
            a = r["arms"][arm]
            fld = np.array(a["field"])
            ok = a["err"]["linf_rel"] < 0.15
            L.append(f"| {r['hy']}x{r['hz']} | {arm} | {a['steps']} | {a['converged']} | "
                     f"{fmt(fld.max())} | {fmt(r['u_max_ref'])} | {fmt(a['err']['linf_rel'])} | "
                     f"{fmt(a['err']['l2_rel'])} | {fmt(a['mass_rel'])} | {'OK' if ok else 'FAIL'} |")

    L.append("\n## P3 — 3-D shear wave, exact decay rate omega = nu k^2\n")
    L.append("| n | arm | omega got | omega ref | rel | amplitudes | <15% |")
    L.append("|---|---|---|---|---|---|---|")
    for r in res["P3"]:
        for arm in ("dense", "sparse_soa", "sparse_aos"):
            a = r["arms"][arm]
            L.append(f"| {r['n']} | {arm} | {fmt(a['omega_got'], 6)} | {fmt(r['omega_ref'], 6)} | "
                     f"{fmt(a['rel_err'])} | {', '.join(fmt(v, 5) for v in a['amplitudes'])} | "
                     f"{'OK' if a['rel_err'] < 0.15 else 'FAIL'} |")

    L.append("\n## P4 — periodic array of *interior* solid slabs (no border walls)\n")
    L.append("| arm | steps | conv | channel profiles (got) | ref profile | L∞/ref | mass | u at slab-adjacent cell | <15% |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    r = res["P4"]
    for arm in ("sparse_soa", "sparse_aos"):
        a = r["arms"][arm]
        prof = "; ".join(", ".join(fmt(v) for v in p) for p in a["profiles"])
        L.append(f"| {arm} | {a['steps']} | {a['converged']} | {prof} | "
                 f"{', '.join(fmt(v) for v in r['ref_profile'])} | {fmt(a['err']['linf_rel'])} | "
                 f"{fmt(a['mass_rel'])} | {', '.join(fmt(v) for v in a['u_at_wall_adjacent'])} | "
                 f"{'OK' if a['err']['linf_rel'] < 0.15 else 'FAIL'} |")

    L.append("\n## P5/P6 — index array, stencil order, layout mismatch\n")
    q = res["P5"]
    L.append(f"* OSS (`lbmpy.sparse.SparseLbMapper`, D3Q19, in-process): shape {q['shape']} {q['dtype']}, "
             f"mismatches vs port = **{q['mismatches_vs_oss']}**, max|diff| = {q['max_abs_diff_vs_oss']}")
    L.append(f"* SoA-vs-AoS columns that differ (same geometry) = {q['soa_vs_aos_columns_differ']}")
    L.append(f"* stencil order invariance (repo order vs lbmpy order), max|du|/u_max = "
             f"{fmt(q['stencil_order_max_delta_over_umax'])}")
    L.append(f"* OSS-array-driven list solver vs ported array, max|du|/u_max = "
             f"{fmt(q['oss_array_driven_vs_ported'])}")
    L.append(f"* negative control (SoA offsets in AoS layout): {q['layout_mismatch_control']}")

    L.append("\n## P7 — storage cost and throughput vs solid fraction (24^3 box, D3Q19)\n")
    L.append("| solid frac | fluid cells | pdf bytes dense | pdf bytes sparse | total sparse/dense | step dense [s] | step sparse [s] |")
    L.append("|---|---|---|---|---|---|---|")
    for r in res["P7"]:
        L.append(f"| {fmt(r['solid_fraction'], 3)} | {r['cells_fluid']} | {r['dense']['pdf_bytes']} | "
                 f"{r['soa']['pdf_bytes']} | {fmt(r['total_bytes_ratio_sparse_over_dense'], 3)} | "
                 f"{fmt(r['dense']['step_s'], 5)} | {fmt(r['soa']['step_s'], 5)} |")
    return "\n".join(L) + "\n"


def main() -> None:
    t_start = time.perf_counter()
    res = {"anchor": "XVAL-LBM-SPARSE-LIST-BASED-AOS-LBM", "round": 2, "mission": 21977,
           "tau": TAU, "nu": NU, "u_max_target": U_MAX,
           "acceptance_band": 0.15,
           "resolution_note": os.environ.get("R2_NOTE", ""),
           "python": platform.python_version(), "numpy": np.__version__,
           "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}

    print("P6/P5 index array + layouts ...", flush=True)
    res["P5"] = case_index_array_and_layouts()
    assert res["P5"]["mismatches_vs_oss"] == 0, "port does not match the OSS index array in 3D"
    print("  mismatches vs OSS:", res["P5"]["mismatches_vs_oss"], flush=True)

    print("P1 3-D plane Poiseuille ...", flush=True)
    res["P1"] = case_plane_poiseuille()
    print("P2 3-D duct ...", flush=True)
    res["P2"] = case_duct()
    print("P3 3-D shear wave ...", flush=True)
    res["P3"] = case_shear_wave()
    print("P4 interior slabs ...", flush=True)
    res["P4"] = case_interior_slabs()
    print("P7 storage/throughput ...", flush=True)
    res["P7"] = case_storage()
    res["elapsed_s"] = time.perf_counter() - t_start

    # verdicts -------------------------------------------------------------
    def arm_errs(rows, arms, key="linf_rel"):
        vals = []
        for r in rows:
            for a in arms:
                vals.append(r["arms"][a]["err"][key])
        return vals

    checks = {
        "P1_plane_poiseuille_all_arms_below_15pct": all(
            v < 0.15 for v in arm_errs(res["P1"], ("dense", "sparse_soa", "sparse_aos"))),
        "P1_errors_decrease_with_H": all(
            res["P1"][i]["arms"]["sparse_soa"]["err"]["linf_rel"] >
            res["P1"][i + 1]["arms"]["sparse_soa"]["err"]["linf_rel"]
            for i in range(len(res["P1"]) - 1)),
        "P2_duct_below_15pct": all(v < 0.15 for v in arm_errs(res["P2"], ("sparse_soa", "sparse_aos"))),
        "P2_duct_finer_below_coarser": (
            res["P2"][0]["arms"]["sparse_soa"]["err"]["linf_rel"] >
            res["P2"][-1]["arms"]["sparse_soa"]["err"]["linf_rel"]),
        "P3_shear_wave_below_15pct": all(
            r["arms"][a]["rel_err"] < 0.15 for r in res["P3"] for a in ("dense", "sparse_soa", "sparse_aos")),
        "P4_interior_slabs_below_15pct": all(
            res["P4"]["arms"][a]["err"]["linf_rel"] < 0.15 for a in ("sparse_soa", "sparse_aos")),
        "P4_mass_conserved": all(res["P4"]["arms"][a]["mass_rel"] < 1e-12 for a in ("sparse_soa", "sparse_aos")),
        "P5_arms_equivalent_1e12": all(
            r["arm_equivalence_200_steps"]["soa_vs_aos"] < 1e-12 and
            r["arm_equivalence_200_steps"]["dense_vs_sparse_soa"] < 1e-12 for r in res["P1"]),
        "P5_stencil_order_invariant": res["P5"]["stencil_order_max_delta_over_umax"] < 1e-12,
        "P5_oss_array_driven_equals_port": res["P5"]["oss_array_driven_vs_ported"] < 1e-12,
        "P6_index_array_bit_identical_to_oss": res["P5"]["mismatches_vs_oss"] == 0,
        "P6_layout_mismatch_detected": bool(res["P5"]["layout_mismatch_control"]["detected"]),
    }
    res["checks"] = checks
    res["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    res["elapsed_s"] = time.perf_counter() - t_start

    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=lambda o: float(o)))
    (HERE / "tables.md").write_text(build_tables(res))
    print(build_tables(res))
    print("verdict:", res["verdict"], "failed:", [k for k, v in checks.items() if not v])
    print(f"elapsed {res['elapsed_s']:.1f} s -> {HERE/'results.json'}")


if __name__ == "__main__":
    main()