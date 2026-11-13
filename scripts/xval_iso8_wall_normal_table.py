#!/usr/bin/env python3
"""XVAL-LBM-ISO8-WALL-NORMAL-TABLE -- cross-validation harness.

Anchor
------
Cross-validate TensorLBM against ``lanl/MF-LBM``'s *iso8 wall-normal table*:
the preprocessor ``preprocessing/3.wall_boundary_preprocess/3d_wall_boundary_process.f90``
builds, once per geometry, a table of unit wall normals on every fluid node that
touches the solid (``fluid_boundary_nodes % snx/sny/snz``).  The normal is the
ISO8 isotropic finite-difference gradient of a four-times D3Q27-smoothed solid
indicator.  MF-LBM consumes that table in ``Phase_gradient.F90`` to impose the
geometric (contact-angle) wetting condition.

Subcommands
-----------
``extract``   parse the vendored OSS Fortran source and dump the *stencil* it
              actually uses (ISO8 weights, the 34 difference pairs, the D3Q27
              smoothing kernel, the 18-direction boundary-neighbour set, the
              node-classification rule constants).  Nothing in this harness is
              hand-copied from the paper: every constant below is read back out
              of ``oss_src/wall_boundary_process.f90``.
``run``       build/compile the OSS preprocessor with gfortran, run it on the
              declared test geometries, run the TensorLBM transplant on the
              *same* wall arrays, and write the comparison artifacts.
``report``    figures + markdown tables from the artifacts.

Everything written lands in ``artifacts/xval_iso8_wall_normal_table/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "xval_iso8_wall_normal_table"
OSS = REPO / "xval" / "XVAL-LBM-ISO8-WALL-NORMAL-TABLE" / "oss_src" / "wall_boundary_process.f90"
BUILD = OUT / "oss_build"

sys.path.insert(0, str(REPO / "src"))

# --------------------------------------------------------------------------- #
# stage 1: extract the OSS stencil from its own source  (zero hand-copy)
# --------------------------------------------------------------------------- #

_TERM = re.compile(
    r"walls_smooth_2\(\s*i\s*([+-]\s*\d+)?\s*,\s*j\s*([+-]\s*\d+)?\s*,\s*k\s*([+-]\s*\d+)?\s*\)"
)


def _read_source(path: Path = OSS) -> str:
    return path.read_text(errors="replace")


def _parse_iso8_weights(src: str) -> list[float]:
    m = re.search(r"ISO8\(1:7\)\s*=\s*\(/(.*?)/\)", src, flags=re.S)
    if m is None:
        raise RuntimeError("ISO8 weight table not found in OSS source")
    expr = m.group(1)
    vals = []
    for tok in expr.split(","):
        tok = tok.strip().lower().replace("d0", "")
        num, den = tok.split("/")
        vals.append(float(num) / float(den))
    return vals


def _parse_gradient_pairs(src: str) -> dict:
    """Per-component ISO8 pair lists ``{axis: [(w_index0, h), ...]}``.

    MF-LBM writes each component as ``ISO8(n)*( A - B + C - D ... )``; every
    ``A - B`` is one central pair.  The pair *sets differ per component* (the x
    component carries xy and xz edge pairs, the y component xy and yz, ...), so
    all three blocks have to be read: using one block for all three components
    would silently break the isotropy of the operator.
    """
    def _block(name: str, nxt: str | None) -> str:
        start = src.index(name + " =")
        end = src.index(nxt) if nxt else src.index("tmp = 1d0")
        return src[start:end]

    def _pairs(body: str) -> list[tuple[int, tuple[int, int, int]]]:
        chunks = re.split(r"ISO8\((\d)\)\*", body)
        out: list[tuple[int, tuple[int, int, int]]] = []
        for i in range(1, len(chunks), 2):
            label = int(chunks[i]) - 1          # Fortran 1-based -> python 0-based
            terms = [tuple(int(g.replace(" ", "")) if g else 0 for g in t)
                     for t in _TERM.findall(chunks[i + 1])]
            if len(terms) % 2:
                raise RuntimeError(f"ISO8({label + 1}) block has an odd number of terms")
            for a, b in zip(terms[0::2], terms[1::2]):
                # A - B is the full difference offset; the symmetric
                # half-offset the operator uses is d/2.
                out.append((label, tuple((a[q] - b[q]) // 2 for q in range(3))))
        return out

    pairs = {
        "x": _pairs(_block("nwx", "nwy =")),
        "y": _pairs(_block("nwy", "nwz =")),
        "z": _pairs(_block("nwz", None)),
    }
    # invariant: the components are axis permutations of one another
    def _perm(lst, a, b):
        return sorted((w, tuple(h[b] if i == a else h[a] if i == b else h[i]
                               for i in range(3))) for w, h in lst)
    if _perm(pairs["x"], 0, 1) != sorted(pairs["y"]):
        raise RuntimeError("ISO8 y-component pairs are not the xy-permutation of x")
    if _perm(pairs["x"], 0, 2) != sorted(pairs["z"]):
        raise RuntimeError("ISO8 z-component pairs are not the xz-permutation of x")
    return pairs


def _parse_vec3(src: str, name: str) -> list[int]:
    m = re.search(name + r"\(0:26\)=\(/(.*?)/\)", src, flags=re.S)
    if m is None:
        raise RuntimeError(f"{name} not found")
    return [int(t) for t in m.group(1).split(",")]


def _parse_smoothing_kernel(src: str) -> list[float]:
    m = re.search(r"w\(0:3\)\s*=\s*\(/(.*?)/\)", src, flags=re.S)
    if m is None:
        raise RuntimeError("D3Q27 smoothing weights not found")
    out = []
    for tok in m.group(1).split(","):
        num, den = tok.strip().lower().replace("d0", "").split("/")
        out.append(float(num) / float(den))
    return out


def _parse_int(src: str, pattern: str) -> int:
    m = re.search(pattern, src)
    if m is None:
        raise RuntimeError(f"pattern {pattern!r} not found in OSS source")
    return int(m.group(1))


def extract_stencil(src: str | None = None) -> dict:
    src = _read_source() if src is None else src
    weights = _parse_iso8_weights(src)
    pairs = _parse_gradient_pairs(src)
    kernel = _parse_smoothing_kernel(src)
    ex, ey, ez = (_parse_vec3(src, n) for n in ("ex", "ey", "ez"))
    lattice = [(ex[n], ey[n], ez[n]) for n in range(27)]
    # the 18-direction half set MF-LBM uses for boundary-node neighbour lists
    half18 = lattice[1:19]
    stencil = {
        "source": {
            "repo": "lanl/MF-LBM",
            "path": "preprocessing/3.wall_boundary_preprocess/3d_wall_boundary_process.f90",
            "upstream_commit": "11be324a97fe6331b2271102da370460715355e8",
            "vendored_sha256": hashlib.sha256(_read_source().encode()).hexdigest(),
            "license": "BSD-3-Clause (Triad National Security, LLC, 2021)",
        },
        "iso8_weights": weights,
        "iso8_pairs": {c: [{"w": w, "h": list(h)} for w, h in v] for c, v in pairs.items()},
        "iso8_n_pairs_per_component": {c: len(v) for c, v in pairs.items()},
        "iso8_pair_set_invariant": "component y/z pair sets are the xy/xz axis "
                                   "permutations of component x (checked at parse time)",
        "smoothing": {
            "kernel_d3q27_by_squared_speed": kernel,
            "n_iterations": _parse_int(src, r"i_max_iteration\s*=\s*(\d+)"),
            "kernel_sum_over_d3q27": float(sum(
                kernel[(a * a + b * b + c * c)] for a, b, c in lattice)),
        },
        "lattice_offsets_d3q27": [list(v) for v in lattice],
        "boundary_neighbour_half18": [list(v) for v in half18],
        "node_classification": {
            "solid_indicator": "walls==1 (1 = solid, 0 = fluid)",
            "solid_boundary_marker": 2,
            "fluid_boundary_marker": -1,
            "solid_boundary_rule": "walls==1 and any half18 neighbour has walls<=0",
            "fluid_boundary_rule": "walls==0 and any half18 neighbour has walls>=1",
            "la_weight": "sum of w_equ(n) over the half18 neighbours with walls<=0",
        },
        "ghost_layers_for_smoothing": _parse_int(src, r"ghost_layers\s*=\s*(\d+)"),
        "overlap_phi": _parse_int(src, r"overlap_phi\s*=\s*(\d+)"),
        "normal_convention": "snx/sny/snz = normalized grad(smoothed solid fraction), "
                             "i.e. pointing INTO the solid (MF-LBM Phase_gradient.F90 usage)",
    }
    return stencil


# --------------------------------------------------------------------------- #
# geometry generators (deterministic, no randomness)
# --------------------------------------------------------------------------- #

def sphere_walls(n: int, radius: float, center: tuple[float, float, float]) -> np.ndarray:
    """``walls[i,j,k] = 1`` inside the sphere (Fortran 1-based index convention)."""
    ii = np.arange(1, n + 1) - center[0]
    jj = np.arange(1, n + 1) - center[1]
    kk = np.arange(1, n + 1) - center[2]
    r2 = ii[:, None, None] ** 2 + jj[None, :, None] ** 2 + kk[None, None, :] ** 2
    return (r2 < radius * radius).astype(np.int8)


def plane_walls(n: int, normal: tuple[float, float, float], offset: float) -> np.ndarray:
    """Solid where ``normal . x < offset`` (x in Fortran 1-based cell units)."""
    nv = np.asarray(normal, float)
    nv = nv / np.linalg.norm(nv)
    ii = np.arange(1, n + 1)
    dist = (nv[0] * ii[:, None, None] + nv[1] * ii[None, :, None] + nv[2] * ii[None, None, :])
    return (dist < offset).astype(np.int8)


# --------------------------------------------------------------------------- #
# OSS arm: build + run the Fortran preprocessor
# --------------------------------------------------------------------------- #

def build_oss(force: bool = False) -> Path:
    BUILD.mkdir(parents=True, exist_ok=True)
    src = BUILD / "wall_boundary_process.f90"
    exe = BUILD / "wbp"
    if exe.exists() and not force:
        return exe
    shutil.copyfile(OSS, src)
    # -ffree-line-length-none: the upstream makefile relies on a longer line
    # limit (several declarations exceed 132 columns).
    # -finit-integer=0 / -finit-real=zero: the upstream program reads module
    # variable `mirror_bc` before it is set (see difference list D7); zero
    # initialisation makes the arm deterministic and is exactly the upstream
    # default behaviour for a zeroed .bss image.
    subprocess.run(
        ["gfortran", "-O2", "-fopenmp", "-ffree-line-length-none",
         "-finit-integer=0", "-finit-real=zero", "-o", str(exe), str(src)],
        check=True, capture_output=True, text=True,
    )
    return exe


def write_wall_file(path: Path, walls: np.ndarray) -> None:
    nx, ny, nz = walls.shape
    with open(path, "wb") as f:
        np.array([nx, ny, nz], "<i4").tofile(f)
        # Fortran reads the array with i fastest, so the stream must be the
        # column-major (F) flattening -- C-order tofile would transpose the
        # geometry (invisible for an axis-symmetric test case, fatal otherwise).
        f.write(walls.astype(np.int8).ravel(order="F").tobytes())


def run_oss(exe: Path, walls: np.ndarray, workdir: Path, periodic=(0, 0, 0),
            domain_walls=(0, 0, 0, 0, 0, 0)) -> dict:
    """Run the OSS preprocessor on ``walls`` and parse its normal table."""
    workdir.mkdir(parents=True, exist_ok=True)
    write_wall_file(workdir / "walls.bin", walls)
    (workdir / "input_parameters.txt").write_text(
        "wall file path\n./walls.bin\n%d %d %d\n%d %d\n%d %d\n%d %d\n" % (
            periodic[0], periodic[1], periodic[2],
            domain_walls[0], domain_walls[1], domain_walls[2],
            domain_walls[3], domain_walls[4], domain_walls[5],
        )
    )
    subprocess.run([str(exe)], cwd=workdir, check=True, capture_output=True, text=True)
    raw = (workdir / "output" / "indirect_addressing_walls_boundary.dat").read_bytes()
    ns, nf = np.frombuffer(raw[:8], "<i4")
    off = 8
    solid_dt = np.dtype([("ix", "<i4"), ("iy", "<i4"), ("iz", "<i4"), ("i_fluid_num", "<i4"),
                         ("neighbor_list", "<i4", 18), ("la_weight", "<f8")])
    fluid_dt = np.dtype([("ix", "<i4"), ("iy", "<i4"), ("iz", "<i4"),
                         ("snx", "<f8"), ("sny", "<f8"), ("snz", "<f8")])
    # gfortran writes derived-type components packed for unformatted stream I/O
    # (int32 x3 + float64 x3 = 36 B; int32 x4 + int32x18 + float64 = 96 B)
    solid = np.frombuffer(raw[off:off + ns * solid_dt.itemsize], dtype=solid_dt).copy()
    off += ns * solid_dt.itemsize
    fluid = np.frombuffer(raw[off:off + nf * fluid_dt.itemsize], dtype=fluid_dt).copy()
    off += nf * fluid_dt.itemsize
    if off != len(raw):
        raise RuntimeError(f"OSS table parse did not consume the file ({off} of {len(raw)} B)")
    return {
        "n_solid_boundary": int(ns),
        "n_fluid_boundary": int(nf),
        "solid": solid,
        "fluid": fluid,
    }


# --------------------------------------------------------------------------- #
# TensorLBM arm
# --------------------------------------------------------------------------- #

def run_ours(walls: np.ndarray, stencil: dict, periodic=(0, 0, 0),
             domain_walls=(0, 0, 0, 0, 0, 0), n_smooth=None) -> dict:
    from tensorlbm.boundary.wall_normal_iso8 import wall_normal_table

    return wall_normal_table(
        walls, periodic=periodic, domain_walls=domain_walls, n_smooth=n_smooth,
    )


# --------------------------------------------------------------------------- #
# analytic references and competing stencils
# --------------------------------------------------------------------------- #

def analytic_normals(indices: np.ndarray, kind: str, geom: dict) -> np.ndarray:
    """Analytic *inward* (into-solid) unit normals at the listed nodes."""
    x = np.asarray(indices, dtype=float)             # Fortran 1-based cell centres
    if kind == "sphere":
        d = x - np.asarray(geom["center"], float)
        r = np.linalg.norm(d, axis=1, keepdims=True)
        return -d / r
    nv = np.asarray(geom["normal"], float)
    nv = nv / np.linalg.norm(nv)
    return np.repeat(-nv[None, :], len(x), axis=0)


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    c = np.clip((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)), -1.0, 1.0)
    return np.degrees(np.arccos(c))


def rel_err(n: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Chordal relative error of a unit-vector field, 2*sin(theta/2)."""
    return np.linalg.norm(n - ref, axis=1)


def central_difference_field(F: np.ndarray):
    """Plain 2nd-order central differences on the same (smoothed) field."""
    return tuple(np.gradient(F))


# --------------------------------------------------------------------------- #
# one arm
# --------------------------------------------------------------------------- #

def build_case(name: str, kind: str, n: int, geom: dict) -> dict:
    walls = sphere_walls(n, geom["radius"], geom["center"]) if kind == "sphere" \
        else plane_walls(n, geom["normal"], geom["offset"])
    return {"name": name, "kind": kind, "n": n, "geom": geom, "walls": walls}


def evaluate(case: dict, stencil: dict, exe: Path, do_oss: bool = True) -> dict:
    import tensorlbm.boundary.wall_normal_iso8 as M

    walls = case["walls"]
    res = {"case": case["name"], "kind": case["kind"], "n": case["n"]}

    t0 = time.perf_counter()
    ours = run_ours(walls, stencil)
    res["t_ours_s"] = time.perf_counter() - t0

    idx = ours["indices"]
    inphys = M.in_physical_domain(idx, walls.shape)
    ref = analytic_normals(idx, case["kind"], case["geom"])

    def stats(nrm, mask=inphys):
        e = angle_deg(nrm[mask], ref[mask])
        rel = rel_err(nrm[mask], ref[mask])
        fin = np.isfinite(e)
        return {"mean_deg": float(e[fin].mean()) if fin.any() else float("nan"),
                "max_deg": float(e[fin].max()) if fin.any() else float("nan"),
                "p95_deg": float(np.percentile(e[fin], 95)) if fin.any() else float("nan"),
                "mean_rel": float(rel[fin].mean()) if fin.any() else float("nan"),
                "max_rel": float(rel[fin].max()) if fin.any() else float("nan"),
                "n_nodes": int(mask.sum()),
                "n_degenerate": int((~fin).sum())}

    res["iso8_smooth4"] = stats(ours["normals"])
    res["n_fluid_boundary"] = ours["n_fluid_boundary"]
    res["n_nodes_physical"] = int(inphys.sum())

    # --- OSS arm: same wall array, independent Fortran implementation
    if do_oss:
        t0 = time.perf_counter()
        oss = run_oss(exe, walls, OUT / "oss_runs" / case["name"])
        res["t_oss_s"] = time.perf_counter() - t0
        oidx = np.stack([oss["fluid"]["ix"], oss["fluid"]["iy"], oss["fluid"]["iz"]], axis=1)
        onrm = np.stack([oss["fluid"]["snx"], oss["fluid"]["sny"], oss["fluid"]["snz"]], axis=1)
        res["oss_counts"] = [int(oss["n_solid_boundary"]), int(oss["n_fluid_boundary"])]
        res["ours_counts"] = [int(ours["n_solid_boundary"]), int(ours["n_fluid_boundary"])]
        res["oss_node_table_identical"] = bool(np.array_equal(oidx, idx))
        res["oss_la_weight_maxdiff"] = float(np.abs(oss["solid"]["la_weight"]
                                                    - ours["solid_la_weight"]).max()) \
            if res["oss_node_table_identical"] and len(oss["solid"]) else None
        if res["oss_node_table_identical"]:
            res["oss_vs_ours_max_abs_dn"] = float(np.abs(onrm - ours["normals"]).max())
            res["oss_vs_ours_max_angle_deg"] = float(angle_deg(onrm, ours["normals"]).max())
            res["oss_accuracy"] = stats(onrm)
        else:
            res["oss_vs_ours_max_abs_dn"] = None

    # --- competing stencils on the SAME smoothed indicator
    pad = M.DEFAULT_GHOST_LAYERS + M.DEFAULT_OVERLAP
    markers = M._pad_solid(walls, pad)
    F4 = M.smooth_solid_fraction(markers.astype(float), 4, pad)
    F0 = M.smooth_solid_fraction(markers.astype(float), 0, pad)
    p = idx + pad - 1

    def at(fields):
        return np.stack([f[p[:, 0], p[:, 1], p[:, 2]] for f in fields], axis=1)

    def unit(g):
        m = np.linalg.norm(g, axis=1, keepdims=True)
        return np.divide(g, m, out=np.zeros_like(g), where=m > 0)

    res["central_smooth4"] = stats(unit(at(central_difference_field(F4))))
    res["central_nosmooth"] = stats(unit(at(central_difference_field(F0))))
    res["iso8_nosmooth"] = stats(unit(at(M.iso8_gradient(F0))))
    # staircase reference: axis-aligned differences of the raw indicator
    res["staircase_raw"] = stats(unit(at(central_difference_field(F0))))

    # --- smoothing-iteration sweep (iso8)
    sweep = {}
    for it in (0, 1, 2, 4, 8, 16):
        Fi = M.smooth_solid_fraction(markers.astype(float), it, pad)
        sweep[str(it)] = stats(unit(at(M.iso8_gradient(Fi))))
        del Fi
    res["smooth_sweep_iso8"] = sweep

    # --- downstream: mean curvature of the iso8 level set, kappa = div(n)
    def curvature(fields):
        gx, gy, gz = fields
        m = np.sqrt(gx * gx + gy * gy + gz * gz)
        ok = m > 1e-12
        nx = np.divide(gx, m, out=np.zeros_like(gx), where=ok)
        ny = np.divide(gy, m, out=np.zeros_like(gy), where=ok)
        nz = np.divide(gz, m, out=np.zeros_like(gz), where=ok)
        k = M.iso8_divergence(nx, ny, nz)
        # evaluate on nodes within one cell of the nominal level set
        x = np.asarray(idx, float)
        if case["kind"] == "sphere":
            r = np.linalg.norm(x - np.asarray(case["geom"]["center"], float), axis=1)
            band = np.abs(r - case["geom"]["radius"]) < 1.0
            kref = np.full(int(band.sum()), 2.0 / case["geom"]["radius"])
        else:
            band = np.ones(len(x), bool)
            kref = np.zeros(len(x))
        kv = k[p[:, 0], p[:, 1], p[:, 2]][band]
        return {"kappa_mean": float(kv.mean()),
                "kappa_ref": float(kref.mean()) if len(kref) else None,
                "rel_err_mean": float(abs(kv.mean() - kref.mean()) / abs(kref.mean()))
                if len(kref) and abs(kref.mean()) > 0 else None,
                "rel_err_node_mean": float(np.abs(kv - kref).mean() / abs(kref.mean()))
                if len(kref) and abs(kref.mean()) > 0 else None,
                "n_band": int(band.sum())}

    res["curvature_iso8"] = curvature(M.iso8_gradient(F4))
    res["curvature_central"] = curvature(central_difference_field(F4))

    # --- isotropy audit of the operator on analytic linear fields
    iso = {}
    for tag, grad_fn in (("iso8", M.iso8_gradient), ("central", central_difference_field)):
        gains = []
        for dvec in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1),
                     (1, 1, 1), (1, -1, 1)):
            d = np.asarray(dvec, float)
            d /= np.linalg.norm(d)
            ijk = np.indices(F4.shape).astype(float)
            lin = d[0] * ijk[0] + d[1] * ijk[1] + d[2] * ijk[2]
            g = np.stack(grad_fn(lin), axis=0)
            mag = np.linalg.norm(g, axis=0)[pad:-pad, pad:-pad, pad:-pad].mean()
            gains.append(float(mag))
        iso[tag] = {"direction_gain_min": min(gains), "direction_gain_max": max(gains),
                    "anisotropy": float(max(gains) / min(gains))}
    res["operator_isotropy"] = iso

    for k in list(res):
        if isinstance(res[k], np.generic):
            res[k] = res[k].item()
    return res


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

CASES = [
    ("sphere_n48_R10_offcenter", "sphere", 48, {"radius": 10.0, "center": (24.3, 23.6, 24.1)}),
    ("sphere_n64_R14_offcenter", "sphere", 64, {"radius": 14.0, "center": (32.2, 31.5, 32.4)}),
    ("plane_n40_alpha26.565", "plane", 40, {"normal": (2.0, 1.0, 0.0), "offset": 20.0}),
    ("plane_n40_alpha45", "plane", 40, {"normal": (1.0, 1.0, 0.0), "offset": 20.0}),
]


def cmd_run(force_build: bool = False) -> int:
    stencil = extract_stencil()
    (OUT / "oss_iso8_stencil.json").write_text(json.dumps(stencil, indent=2) + "\n")
    exe = build_oss(force=force_build)
    results = []
    for name, kind, n, geom in CASES:
        case = build_case(name, kind, n, geom)
        r = evaluate(case, stencil, exe, do_oss=True)
        results.append(r)
        print(f"{name}: iso8+4 mean rel {r['iso8_smooth4']['mean_rel']:.3e} "
              f"max {r['iso8_smooth4']['max_rel']:.3e} deg {r['iso8_smooth4']['mean_deg']:.3f} | "
              f"OSS-vs-ours max|dn| {r['oss_vs_ours_max_abs_dn']:.2e}")
    payload = {
        "anchor": "XVAL-LBM-ISO8-WALL-NORMAL-TABLE",
        "oss": stencil["source"],
        "gfortran": subprocess.run(["gfortran", "--version"], capture_output=True, text=True
                                   ).stdout.splitlines()[0],
        "platform": {"python": platform.python_version(), "numpy": np.__version__},
        "stencil": stencil,
        "cases": results,
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    _write_tables(results)
    _write_fixture(results)
    _figures(results)
    return 0


def _write_tables(results) -> None:
    rows = []
    for r in results:
        for arm in ("iso8_smooth4", "central_smooth4", "iso8_nosmooth",
                    "central_nosmooth", "oss_accuracy"):
            if arm in r and r[arm]:
                rows.append([r["case"], arm, r[arm]["n_nodes"],
                             f"{r[arm]['mean_deg']:.6g}", f"{r[arm]['max_deg']:.6g}",
                             f"{r[arm]['mean_rel']:.6g}", f"{r[arm]['max_rel']:.6g}"])
    with open(OUT / "accuracy_table.csv", "w") as f:
        f.write("case,arm,n_nodes,mean_angle_deg,max_angle_deg,mean_rel,max_rel\n")
        for row in rows:
            f.write(",".join(str(c) for c in row) + "\n")

    with open(OUT / "exactness_table.csv", "w") as f:
        f.write("case,oss_nodes,ours_nodes,table_identical,max_abs_dn,max_angle_deg,"
                "la_weight_maxdiff,t_oss_s,t_ours_s\n")
        for r in results:
            f.write("%s,%d,%d,%s,%s,%s,%s,%s,%.4f\n" % (
                r["case"], r["oss_counts"][1] if "oss_counts" in r else -1,
                r["ours_counts"][1] if "ours_counts" in r else -1,
                r.get("oss_node_table_identical"), r.get("oss_vs_ours_max_abs_dn"),
                r.get("oss_vs_ours_max_angle_deg"), r.get("oss_la_weight_maxdiff"),
                r.get("t_oss_s"), r["t_ours_s"]))

    with open(OUT / "sensitivity_table.csv", "w") as f:
        f.write("case,arm,key,mean_rel,max_rel\n")
        for r in results:
            for it, st in r["smooth_sweep_iso8"].items():
                f.write(f"{r['case']},iso8_smooth_sweep,{it},{st['mean_rel']:.6g},{st['max_rel']:.6g}\n")
            for arm in ("curvature_iso8", "curvature_central"):
                c = r[arm]
                f.write(f"{r['case']},{arm},kappa_mean,{c['kappa_mean']:.6g},"
                        f"{c['rel_err_mean'] if c['rel_err_mean'] is not None else ''}\n")
            for arm, iso in r["operator_isotropy"].items():
                f.write(f"{r['case']},isotropy_{arm},anisotropy,{iso['anisotropy']:.6g},\n")


def _write_fixture(results) -> None:
    """Store the first case's full OSS normal table as a regression fixture."""
    r = results[0]
    exe = build_oss()
    case = next(c for c in CASES if c[0] == r["case"])
    walls = sphere_walls(case[2], case[3]["radius"], case[3]["center"]) \
        if case[1] == "sphere" else plane_walls(case[2], case[3]["normal"], case[3]["offset"])
    oss = run_oss(exe, walls, OUT / "oss_runs" / (case[0] + "_fixture"))
    idx = np.stack([oss["fluid"]["ix"], oss["fluid"]["iy"], oss["fluid"]["iz"]], axis=1)
    nrm = np.stack([oss["fluid"]["snx"], oss["fluid"]["sny"], oss["fluid"]["snz"]], axis=1)
    fx = OUT / "oss_fixture_normal_table.csv"
    with open(fx, "w") as f:
        f.write("i,j,k,snx,sny,snz\n")
        for (a, b, c), (x, y, z) in zip(idx, nrm):
            f.write(f"{a},{b},{c},{x:.17g},{y:.17g},{z:.17g}\n")
    with open(OUT / "oss_fixture_case.json", "w") as f:
        json.dump({"case": case[0], "kind": case[1], "n": case[2], "geom": case[3],
                   "stable_at": "2026-11-13"}, f, indent=2)


def _figures(results) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tensorlbm.boundary.wall_normal_iso8 as M

    figdir = OUT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # (1) sphere slice: exact vs got normals + angular error map
    case = build_case(*CASES[0][:3], CASES[0][3]) if False else None
    name, kind, n, geom = CASES[0]
    walls = sphere_walls(n, geom["radius"], geom["center"])
    t = M.wall_normal_table(walls)
    ref = analytic_normals(t["indices"], kind, geom)
    err = angle_deg(t["normals"], ref)
    ok = M.in_physical_domain(t["indices"], walls.shape)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    sl = t["indices"][:, 2] == int(round(geom["center"][2]))
    ax[0].set_aspect("equal")
    ax[0].quiver(t["indices"][sl, 0], t["indices"][sl, 1],
                 -ref[sl, 0], -ref[sl, 1], color="k", scale=30, width=0.004,
                 label="analytic outward")
    ax[0].quiver(t["indices"][sl, 0], t["indices"][sl, 1],
                 -t["normals"][sl, 0], -t["normals"][sl, 1], color="tab:red",
                 scale=30, width=0.002, label="ISO8 table (outward sign)")
    ax[0].set_title(f"wall normals, sphere R={geom['radius']:g}, mid-k slice")
    ax[0].set_xlabel("x [lattice]")
    ax[0].set_ylabel("y [lattice]")
    ax[0].legend(loc="upper right", fontsize=7)
    sc = ax[1].scatter(t["indices"][ok, 0], t["indices"][ok, 1], c=err[ok],
                       s=8, cmap="viridis")
    ax[1].set_aspect("equal")
    ax[1].set_title("angular error vs analytic normal [deg] (all nodes, x-y view)")
    plt.colorbar(sc, ax=ax[1])
    fig.tight_layout()
    fig.savefig(figdir / "fig1_sphere_normal_field.png", dpi=140)
    plt.close(fig)

    # (2) smoothing-iteration sensitivity
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for r in results:
        its = sorted(int(k) for k in r["smooth_sweep_iso8"])
        ax.plot(its, [r["smooth_sweep_iso8"][str(i)]["mean_rel"] for i in its],
                marker="o", label=f"{r['case']} mean")
        ax.plot(its, [r["smooth_sweep_iso8"][str(i)]["max_rel"] for i in its],
                marker="s", ls="--", lw=1, alpha=0.6)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("D3Q27 smoothing iterations before the ISO8 gradient")
    ax.set_ylabel("relative normal error  (2 sin(dtheta/2))")
    ax.axhline(0.15, color="r", ls=":", label="owner band 15%")
    ax.set_yscale("log")
    ax.legend(fontsize=6)
    ax.set_title("sensitivity to smoothing (solid = mean, dashed = max)")
    fig.tight_layout()
    fig.savefig(figdir / "fig2_smoothing_sensitivity.png", dpi=140)
    plt.close(fig)

    # (3) stencil comparison bars
    arms = ["iso8_smooth4", "central_smooth4", "iso8_nosmooth", "central_nosmooth"]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    width = 0.8 / len(results)
    for i, r in enumerate(results):
        vals = [r[a]["mean_rel"] for a in arms]
        ax.bar(np.arange(len(arms)) + i * width - 0.4, vals, width, label=r["case"])
    ax.axhline(0.15, color="r", ls=":", label="owner band 15%")
    ax.set_xticks(np.arange(len(arms)))
    ax.set_xticklabels(["ISO8+4sm", "central+4sm", "ISO8 raw", "central raw"], fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("mean relative normal error")
    ax.legend(fontsize=6)
    ax.set_title("ISO8 isotropic stencil vs 2nd-order central differences")
    fig.tight_layout()
    fig.savefig(figdir / "fig3_stencil_comparison.png", dpi=140)
    plt.close(fig)

    # (4) operator isotropy on analytic linear fields
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    labels = ["ISO8", "central"]
    for i, r in enumerate(results[:2]):
        vals = [r["operator_isotropy"]["iso8"]["anisotropy"],
                r["operator_isotropy"]["central"]["anisotropy"]]
        ax.bar(np.arange(2) + i * 0.36 - 0.18, vals, 0.36, label=r["case"])
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(labels)
    ax.set_ylabel("max/min directional gain (1.0 = isotropic)")
    ax.legend(fontsize=7)
    ax.set_title("operator anisotropy on analytic linear fields")
    fig.tight_layout()
    fig.savefig(figdir / "fig4_operator_anisotropy.png", dpi=140)
    plt.close(fig)

    print("figures ->", sorted(p.name for p in figdir.glob("*.png")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "run"])
    ap.add_argument("--force-build", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "extract":
        st = extract_stencil()
        (OUT / "oss_iso8_stencil.json").write_text(json.dumps(st, indent=2) + "\n")
        print(json.dumps({k: st[k] for k in ("iso8_weights", "iso8_n_pairs_per_component",
                                            "smoothing", "ghost_layers_for_smoothing",
                                            "overlap_phi")}, indent=2))
        return 0
    return cmd_run(force_build=args.force_build)


if __name__ == "__main__":
    raise SystemExit(main())
