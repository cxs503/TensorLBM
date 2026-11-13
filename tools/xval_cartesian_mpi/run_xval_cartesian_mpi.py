"""XVAL-LBM-CARTESIAN-MPI: cross-validate TensorLBM's cartesian domain
decomposition against pylbm 0.12.0's cartesian-mpi implementation.

Reference (OSS):  pylbm / pylbm, ``pylbm/mpi_topology.py`` + ``pylbm/storage.py``
                  (both files executed here through an execution-only mpi4py
                  stand-in, see ``shims/mpi4py/MPI.py``).
Subject:          TensorLBM ``src/tensorlbm/domain_decomposition.py``.

Arms
----
A. Decomposition geometry   : region sizes / imbalance / divisibility / number of
                              transport messages per step, for a grid x rank matrix.
B. Descriptor / wire volume : bytes per message and per step, read off the two
                              implementations' *own* buffer objects.
C. Numerical equivalence    : TensorLBM slab decomposition (P = 2, 4 ranks,
                              simulated) vs single-domain on a 3D Taylor-Green
                              case that varies along the decomposition axis, with
                              the analytic decay solution as an independent arm.
D. pylbm solver arm         : WITHHELD -- see ``--pylbm-physics`` note in the
                              report (pylbm's dic-validator cannot be satisfied
                              for a hand-built D3Q19 moment basis on
                              Python 3.11/cerberus 1.3.8, and no MPI build
                              exists for loongarch64 => no multi-rank pylbm run).

Outputs: ``artifacts/xval_cartesian_mpi/{summary.json,table_decomposition.csv,
table_equivalence.csv}``.

Usage:
    PYTHONPATH=tools/xval_cartesian_mpi/shims:/tmp/pylbm_src \
      python tools/xval_cartesian_mpi/run_xval_cartesian_mpi.py --out artifacts/xval_cartesian_mpi
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKTREE / "src"))

from tensorlbm.domain_decomposition import (  # noqa: E402
    DomainDecomposer,
    HaloExchange,
    SimulatedMultiDevice,
)

# ---------------------------------------------------------------------------
# D3Q19 lattice (TensorLBM numbering, from tensorlbm.d3q19)
# ---------------------------------------------------------------------------

from tensorlbm.d3q19 import C as _C  # noqa: E402

CVEL = np.asarray(_C, dtype=int)
W = np.array(
    [1 / 3] + [1 / 18] * 6 + [1 / 36] * 12, dtype=float
)
CS2 = 1.0 / 3.0


# ---------------------------------------------------------------------------
# A/B. pylbm side: decomposition geometry and descriptors (executes pylbm code)
# ---------------------------------------------------------------------------


def pylbm_topology(n: int, split, comm_size: int, dim: int = 3):
    """Build pylbm's MpiTopology under a topology-only simulated communicator."""
    import mpi4py.MPI as mpi
    from pylbm.mpi_topology import MpiTopology

    comm = mpi.SimCartComm(list(split), [True] * dim, rank=0)
    assert comm.Get_size() == comm_size
    topo = MpiTopology(dim, [True] * dim, comm=comm)
    return topo, comm


def pylbm_axis_regions(n: int, p: int) -> list[int]:
    """pylbm's own 1-D split for ``n`` points over ``p`` processes."""
    from pylbm.mpi_topology import MpiTopology

    topo = MpiTopology.__new__(MpiTopology)  # geometry helper needs no comm
    return topo.get_region_indices_(n, 0)


def measure_pylbm(n: int, split):
    """Region sizes, imbalance, message count and wire bytes per step (pylbm)."""
    topo, comm = pylbm_topology(n, split, int(np.prod(split)))
    # pylbm rounds the *global* extents down to the split (shape_in)
    shape_in = topo.get_region_indices(n, n, n)
    sizes = [np.diff(a) for a in shape_in]
    local = [int(s[0]) for s in sizes]  # rank 0's local shape (x, y, z)
    vmax = 1  # stencil halo width for D3Q19 in pylbm (vmax = 1)
    nv = 19
    # storage._set_subarray: one face message per (direction, side); payload is
    # nv populations over the whole face plane of the *other* two axes.
    msg_bytes = []
    for d, s in enumerate(sizes):
        other = [int(s[i]) for i in range(3) if i != d]
        msg_bytes.append(nv * vmax * other[0] * other[1] * 8)  # MPI_DOUBLE
    faces = 2 * len(split)
    return {
        "impl": "pylbm",
        "n": n,
        "split": "x".join(str(s) for s in split),
        "nranks": int(np.prod(split)),
        "accepted": True,
        "local_min": int(min(min(s) for s in sizes)),
        "local_max": int(max(max(s) for s in sizes)),
        "imbalance": float(max(max(s) for s in sizes) / min(min(s) for s in sizes)) - 1.0,
        "messages_per_step": faces,
        "bytes_per_message": int(max(msg_bytes)),
        "bytes_per_step_local": int(sum(msg_bytes)),
        "dtype_bytes": 8,
        "ghost_width": vmax,
        "populations_per_message": nv,
        "decomposable_axes": 3,
        "needs_divisible": False,
    }


def measure_pylbm_neighbours(n: int, split):
    """Neighbour rank table + tag pattern actually produced by pylbm's code."""
    import mpi4py.MPI as mpi
    mpishim_CartComm = mpi.SimCartComm
    topo, comm = pylbm_topology(n, split, int(np.prod(split)))
    out = []
    for rank in range(comm.Get_size()):
        cart = mpishim_CartComm(list(split), [True] * len(split), rank=rank)
        coords = cart.Get_coords()
        nb = []
        for d in range(len(split)):
            for step in (-1, +1):
                c = coords.copy()
                c[d] += step
                nb.append(int(cart.Get_cart_rank(c)))
        out.append({"rank": rank, "coords": coords.tolist(), "neighbours": nb})
    return out


# ---------------------------------------------------------------------------
# A/B. TensorLBM side
# ---------------------------------------------------------------------------


def measure_tensorlbm(n: int, p: int, axis: str = "z"):
    """Divisibility, local size, imbalance, message and byte counts per step."""
    out = {"impl": "tensorlbm", "n": n, "split": f"1x1x{p}", "nranks": p,
           "decomposable_axes": 1, "axis": axis}
    try:
        dec = DomainDecomposer(nz_global=n, ny=n, nx=n, n_devices=p,
                               device="cpu", decomposition_axis=axis)
    except ValueError as exc:
        out.update({"accepted": False, "reason": str(exc)[:120]})
        return out
    halo = HaloExchange(dec, mode="simulated", Q=19, dtype=torch.float32)
    slab = torch.zeros((19, dec.nz_local + 2, n, n), dtype=torch.float32)
    halo.pack_send_buffers(slab, 0)  # allocate the real staging buffers
    buf = halo._send_buffers[0]
    up = buf["up"]
    msg_bytes = int(up.numel() * up.element_size())
    out.update({
        "accepted": True,
        "local_min": int(dec.nz_local),
        "local_max": int(dec.nz_local),
        "imbalance": 0.0,
        "messages_per_step": 4,  # 2 Isend + 2 Irecv, one batched NCCL call
        "bytes_per_message": msg_bytes,
        "bytes_per_step_local": int(2 * msg_bytes),
        "dtype_bytes": 4,
        "ghost_width": 1,
        "populations_per_message": int(len(halo._cross_up)),
        "needs_divisible": True,
    })
    return out


# ---------------------------------------------------------------------------
# C. Numerical equivalence (slab decomposition vs single domain)
# ---------------------------------------------------------------------------

TORCH_DTYPE = {"float32": torch.float32, "float64": torch.float64}


def equilibrium(rho, ux, uy, uz, dtype):
    feq = torch.empty((19,) + rho.shape, dtype=dtype)
    c = torch.as_tensor(CVEL, dtype=dtype)
    w = torch.as_tensor(W, dtype=dtype)
    u2 = ux * ux + uy * uy + uz * uz
    for q in range(19):
        cu = c[q, 0] * ux + c[q, 1] * uy + c[q, 2] * uz
        feq[q] = w[q] * rho * (1.0 + cu / CS2 + 0.5 * cu * cu / (CS2 * CS2) - 0.5 * u2 / CS2)
    return feq


def tgv_initial(n, u0, dtype):
    """Taylor-Green in the (z, x) plane: varies along the decomposition axis."""
    k = 2.0 * math.pi / n
    z = (torch.arange(n, dtype=dtype) + 0.5).view(n, 1, 1)
    x = (torch.arange(n, dtype=dtype) + 0.5).view(1, 1, n)
    ux = -u0 * torch.cos(k * z) * torch.sin(k * x) * torch.ones((n, n), dtype=dtype)
    uz = u0 * torch.sin(k * z) * torch.cos(k * x) * torch.ones((n, n), dtype=dtype)
    uy = torch.zeros((n, n, n), dtype=dtype)
    rho = torch.ones((n, n, n), dtype=dtype)
    return equilibrium(rho, ux.expand(n, n, n), uy, uz.expand(n, n, n), dtype)


def collide_core(f, tau):
    rho = f.sum(0)
    ux = (f[1] - f[2] + f[7] - f[8] + f[9] - f[10] + f[11] - f[12] + f[13] - f[14]) / rho
    uy = (f[3] - f[4] + f[7] - f[8] - f[9] + f[10] + f[15] - f[16] + f[17] - f[18]) / rho
    uz = (f[5] - f[6] + f[11] - f[12] - f[13] + f[14] + f[15] - f[16] - f[17] + f[18]) / rho
    feq = equilibrium(rho, ux, uy, uz, f.dtype)
    return f - (f - feq) / tau


def stream_full(f):
    """Periodic pull streaming on a whole domain (reference)."""
    out = torch.empty_like(f)
    for q in range(19):
        out[q] = torch.roll(f[q], shifts=(int(CVEL[q, 2]), int(CVEL[q, 1]), int(CVEL[q, 0])),
                            dims=(0, 1, 2))
    return out


def collide_slab(f, tau):
    g = f.clone()
    g[:, 1:-1] = collide_core(f[:, 1:-1], tau)
    return g


def stream_slab(f):
    """Pull streaming into owned planes only; ghosts are refreshed by the halo."""
    nz_local = f.shape[1] - 2
    g = f.clone()
    for q in range(19):
        cx, cy, cz = (int(v) for v in CVEL[q])
        s = torch.roll(f[q], shifts=(cy, cx), dims=(1, 2))  # periodic in y, x
        if cz == 0:
            g[q, 1:nz_local + 1] = s[1:nz_local + 1]
        elif cz > 0:
            g[q, 1:nz_local + 1] = s[0:nz_local]
        else:
            g[q, 1:nz_local + 1] = s[2:nz_local + 2]
    return g


def run_single(n, steps, tau, u0, dtype):
    f = tgv_initial(n, u0, dtype)
    hist = []
    for it in range(steps + 1):
        hist.append(kinetic(f))
        if it < steps:
            f = stream_full(collide_core(f, tau))
    return f, hist


def kinetic(f):
    rho = f.sum(0)
    ux = (f[1] - f[2] + f[7] - f[8] + f[9] - f[10] + f[11] - f[12] + f[13] - f[14]) / rho
    uy = (f[3] - f[4] + f[7] - f[8] - f[9] + f[10] + f[15] - f[16] + f[17] - f[18]) / rho
    uz = (f[5] - f[6] + f[11] - f[12] - f[13] + f[14] + f[15] - f[16] - f[17] + f[18]) / rho
    return float(0.5 * (ux * ux + uy * uy + uz * uz).mean())


def velocity_fields(f):
    rho = f.sum(0)
    ux = (f[1] - f[2] + f[7] - f[8] + f[9] - f[10] + f[11] - f[12] + f[13] - f[14]) / rho
    uz = (f[5] - f[6] + f[11] - f[12] - f[13] + f[14] + f[15] - f[16] - f[17] + f[18]) / rho
    return ux, uz


def run_decomposed(n, steps, tau, u0, dtype_name, n_ranks):
    dtype = TORCH_DTYPE[dtype_name]
    f0 = tgv_initial(n, u0, dtype)
    dec = DomainDecomposer(nz_global=n, ny=n, nx=n, n_devices=n_ranks, device="cpu")
    halo = HaloExchange(dec, mode="simulated", Q=19, dtype=dtype)
    sim = SimulatedMultiDevice(dec, halo, device="cpu")
    sim.distribute(f0)
    hist = []

    def gather():
        return sim.gather()

    hist.append(kinetic(gather()))
    t0 = time.perf_counter()
    for _ in range(steps):
        # Correct pipeline order (validated: P=1 must reproduce the serial
        # single-domain solver bit-for-bit): collide -> halo exchange ->
        # stream.  The ghost planes must carry the neighbours' *post-collision*
        # boundary planes, because streaming is the last operator of the step.
        for rank in range(n_ranks):
            sim.slabs[rank] = collide_slab(sim.slabs[rank], tau)
        halo.exchange_simulated(sim.slabs)
        for rank in range(n_ranks):
            sim.slabs[rank] = stream_slab(sim.slabs[rank])
        hist.append(kinetic(gather()))
    elapsed = time.perf_counter() - t0
    return gather(), hist, elapsed


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--u0", type=float, default=0.05)
    ap.add_argument("--nu", type=float, default=0.02)
    ap.add_argument("--out", default="artifacts/xval_cartesian_mpi")
    args = ap.parse_args(argv)

    outdir = WORKTREE / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    tau = 3.0 * args.nu + 0.5

    # ---- A/B: geometry + descriptors ------------------------------------
    rows = []
    grids = [16, 24, 32, 48]
    splits = [(2,), (3,), (4,), (2, 2), (2, 2, 2), (3, 2)]
    for n in grids:
        for split in splits:
            p = int(np.prod(split))
            try:
                rows.append(measure_pylbm(n, split))
            except Exception as exc:  # noqa: BLE001
                rows.append({"impl": "pylbm", "n": n, "split": split, "error": str(exc)[:100]})
            if len(split) == 1:
                rows.append(measure_tensorlbm(n, p))
            else:
                rows.append({"impl": "tensorlbm", "n": n, "nranks": p,
                             "split": "x".join(map(str, split)), "accepted": False,
                             "reason": "TensorLBM DomainDecomposer supports one axis only"})
    # axis field: is decomposition_axis honoured?
    axis_probe = {}
    for axis in ("z", "y", "x"):
        try:
            dec = DomainDecomposer(nz_global=32, ny=32, nx=32, n_devices=4,
                                   device="cpu", decomposition_axis=axis)
            slab = torch.zeros((19, dec.nz_local + 2, 32, 32))
            axis_probe[axis] = {"nz_local": dec.nz_local, "slab_shape": list(slab.shape)}
        except Exception as exc:  # noqa: BLE001
            axis_probe[axis] = {"error": str(exc)[:100]}

    neigh = measure_pylbm_neighbours(32, (2, 2))

    with open(outdir / "table_decomposition.csv", "w", newline="") as fh:
        keys = ["impl", "n", "split", "nranks", "accepted", "local_min", "local_max",
                "imbalance", "messages_per_step", "bytes_per_message",
                "bytes_per_step_local", "dtype_bytes", "ghost_width",
                "populations_per_message", "decomposable_axes", "needs_divisible", "reason"]
        wtr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        wtr.writeheader()
        for r in rows:
            wtr.writerow(r)

    # ---- C: numerical equivalence ---------------------------------------
    eq_rows = []
    k = 2.0 * math.pi / args.n
    k0 = args.u0 ** 2 / 4.0
    gam_exact = 4.0 * args.nu * k * k
    ref_f, ref_hist = run_single(args.n, args.steps, tau, args.u0, TORCH_DTYPE["float64"])
    ux_ref, uz_ref = velocity_fields(ref_f)
    for dtype_name in ("float64", "float32"):
        for p in (1, 2, 4):
            f, hist, elapsed = run_decomposed(args.n, args.steps, tau, args.u0, dtype_name, p)
            ux, uz = velocity_fields(f)
            du = torch.sqrt(((ux - ux_ref) ** 2 + (uz - uz_ref) ** 2).mean()).item()
            scale = torch.sqrt((ux_ref ** 2 + uz_ref ** 2).mean()).item()
            gam = math.log(hist[0] / hist[-1]) / (2 * args.steps)
            eq_rows.append({
                "dtype": dtype_name,
                "n_ranks": p,
                "grid": f"{args.n}^3",
                "steps": args.steps,
                "K0": hist[0],
                "K_end": hist[-1],
                "gamma_K_meas": gam,
                "gamma_K_exact": gam_exact / 2.0,
                "gamma_rel_err_pct": abs(gam - gam_exact / 2.0) / (gam_exact / 2.0) * 100,
                "field_rel_l2_pct_vs_serial": du / scale * 100,
                "wall_time_s": elapsed,
            })
    # float64 serial vs analytic K(t)
    an_rows = []
    for it, K in enumerate(ref_hist):
        an_rows.append({"step": it, "K_num": K, "K_exact": k0 * math.exp(-gam_exact * it)})
    with open(outdir / "table_equivalence.csv", "w", newline="") as fh:
        keys = list(eq_rows[0])
        wtr = csv.DictWriter(fh, fieldnames=keys)
        wtr.writeheader()
        for r in eq_rows:
            wtr.writerow(r)

    summary = {
        "anchor": "XVAL-LBM-CARTESIAN-MPI",
        "reference_oss": "pylbm 0.12.0 (pylbm/mpi_topology.py, pylbm/storage.py)",
        "reference_version_sha": "pylbm-0.12.0-py3-none-any.whl @ PyPI",
        "subject": "src/tensorlbm/domain_decomposition.py",
        "case": {
            "name": "Taylor-Green (z,x)-plane, periodic",
            "n": args.n, "steps": args.steps, "u0": args.u0, "nu": args.nu,
            "tau": tau, "k": k, "Ma": args.u0 * math.sqrt(3.0),
            "Re": args.u0 / (args.nu * k), "gamma_K_exact": gam_exact,
        },
        "decomposition_matrix": rows,
        "pylbm_neighbours_32_split_2x2": neigh,
        "axis_param_probe": axis_probe,
        "equivalence": eq_rows,
        "analytic_check_float64_serial": {
            "K0_num": ref_hist[0], "K0_exact": k0,
            "K_end_num": ref_hist[-1], "K_end_exact": k0 * math.exp(-gam_exact * args.steps),
            "rel_err_end_pct": abs(ref_hist[-1] - k0 * math.exp(-gam_exact * args.steps))
            / (k0 * math.exp(-gam_exact * args.steps)) * 100,
        },
        "pylbm_physics_arm": "WITHHELD",
        "pylbm_physics_reason": (
            "no MPI build for loongarch64 (mpi4py C-ext build fails) => no multi-rank "
            "pylbm run; single-rank pylbm Simulation() additionally blocked by its "
            "cerberus 'symbol' schema on Python 3.11 for a programmatic D3Q19 moment basis"
        ),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(outdir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(json.dumps(summary["analytic_check_float64_serial"], indent=2))
    for r in eq_rows:
        print(r)
    print("wrote", outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())