#!/usr/bin/env python3
"""XVAL-LBM-EDT-TWO-SEGMENT-GEOMETRY-TAG-COMPRESSION — TensorLBM vs MPLBM-UT.

Cross-validation of the *EDT two-segment geometry tagging + tag compression*
step (``je-santos/MPLBM-UT`` ``src/python/mplbm_utils/pore_utils.py::
create_geom_edist``) against this repo's independent implementation
:mod:`tensorlbm.geom_edt_tag`.

Three question layers, all machine-adjudicated:

L1 **tag parity** — same rock image in, same int16 tag geometry out?  Metric:
   fraction of voxels whose tag differs; pass = 0 mismatched voxels.
L2 **compression** — does the tag compression buy storage/compute without
   touching the answer?  Metrics: bytes (float64 EDT vs int16 palette vs RLE),
   entropy bits/voxel; and the hydraulic answer K from the two dynamics
   dispatches (boundary-segment bounce-back + NoDynamics grain vs. plain
   all-solid bounce-back).
L3 **physical consistency** — does the tagged geometry still give the
   reference drag of the geometry it came from?  Metric: K vs Hasimoto (1959)
   SC series computed in code (:func:`tensorlbm.array_drag_reference.
   array_drag_factor_from_phi`) — owner criterion < 15%.

Reproduce: ``python xval/edt_two_segment_geometry_tag_compression/run_xval.py``
Writes ``runs/<arm>.json`` + ``summary.json`` + ``table.csv`` next to this file.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import numpy as np
import torch

# small grids: 128 default threads oversubscribe badly (41 ms/step at N=24 vs 7.8 at 4)
torch.set_num_threads(4)

import tensorlbm
from tensorlbm.array_drag_reference import array_drag_factor_from_phi
from tensorlbm.boundary.boundaries3d import bounce_back_cells_3d
from tensorlbm.d3q19 import C as D3Q19_C
from tensorlbm.d3q19 import W_EXACT64, equilibrium3d, macroscopic3d
from tensorlbm.geom_edt_tag import (
    GRAIN_TAG,
    NW_TAG,
    PORE_TAG,
    WALL_TAG,
    compression_report,
    decompress_tags,
    edt_two_segment_tag,
    rle_decode,
    rle_encode,
    tag_stats,
)
from tensorlbm.solver3d import collide_bgk3d, stream3d

from oss_reference import reference_tag_geometry

assert str(Path(tensorlbm.__file__).resolve()).startswith(str(REPO / "src")), tensorlbm.__file__

RUNS = HERE / "runs"
TAU = 1.0
NU = (TAU - 0.5) / 3.0

# OSS lock table (benchmarks/verified/permeability3d/NOTES.md §1.2-1.4); used
# only as a *second* witness next to the in-code Hasimoto series.
SC_TABLE = {0.125: 4.292, 0.45: 28.09}


# --------------------------------------------------------------------------
# geometry generators (rock image: 0 = pore, 1 = solid)
# --------------------------------------------------------------------------
def sc_sphere_mask(n: int, phi: float) -> np.ndarray:
    """Periodic simple-cubic sphere array, one sphere at the cell centre."""
    a = n * (3.0 * phi / (4.0 * math.pi)) ** (1.0 / 3.0)
    idx = np.arange(n, dtype=np.float64)
    d = np.abs(idx - n / 2.0)
    d = np.minimum(d, n - d)
    dist2 = d[:, None, None] ** 2 + d[None, :, None] ** 2 + d[None, None, :] ** 2
    return (dist2 <= a * a).astype(np.int8)


def random_sphere_medium(shape: tuple[int, int, int], phi: float, seed: int = 7) -> np.ndarray:
    """Non-overlapping random sphere packing (periodic), target solid fraction."""
    rng = np.random.default_rng(seed)
    nx, ny, nz = shape
    target_volume = phi * nx * ny * nz
    r = 3.0
    volume = 0.0
    solid = np.zeros(shape, dtype=np.int8)
    grid = np.indices(shape).astype(np.float64)
    attempts = 0
    while volume < target_volume and attempts < 4000:
        attempts += 1
        c = rng.integers(0, [nx, ny, nz]).astype(np.float64)
        # rejection against existing solid voxels (periodic min-image)
        d = np.minimum(np.abs(grid - c[:, None, None, None]), np.array(shape)[:, None, None, None]
                       - np.abs(grid - c[:, None, None, None]))
        sph = (d ** 2).sum(axis=0) <= (r + 0.5) ** 2
        if np.any(solid[sph]):
            continue
        shell = (d ** 2).sum(axis=0) <= r * r
        solid[shell] = 1
        volume = float(solid.sum())
    return solid


def cylinder_in_box(nx: int, ny: int, nz: int, radius: float, touch_wall: bool = False) -> np.ndarray:
    """Streamwise cylinder (flow axis x); either centred or touching one wall."""
    yy = np.arange(ny)[:, None] - (ny / 2.0)
    zz = np.arange(nz)[None, :] - (nz / 2.0)
    if touch_wall:
        yy = np.arange(ny)[:, None] - 2.0
    r2 = yy ** 2 + zz ** 2
    disk = (r2 <= radius * radius).astype(np.int8)
    return np.repeat(disk[None, :, :], nx, axis=0)


def slab_touching_face(nx: int, ny: int, nz: int) -> np.ndarray:
    """Half-domain solid slab: the face-forcing rule is load-bearing here."""
    rock = np.zeros((nx, ny, nz), dtype=np.int8)
    rock[nx // 2:, :, :] = 1
    return rock


def geometries() -> list[dict]:
    nx, ny, nz = 64, 32, 32
    g = [
        {"id": "sc_spheres_phi0.125_N32", "rock": sc_sphere_mask(32, 0.125), "kind": "sphere_array"},
        {"id": "sc_spheres_phi0.45_N32", "rock": sc_sphere_mask(32, 0.45), "kind": "sphere_array"},
        {"id": "random_spheres_phi0.10_N32", "rock": random_sphere_medium((32, 32, 32), 0.10), "kind": "random"},
        {"id": "cylinder_D16_in_64x32x32", "rock": cylinder_in_box(nx, ny, nz, 8.0), "kind": "cylinder"},
        {"id": "cylinder_wall_touching", "rock": cylinder_in_box(nx, ny, nz, 8.0, touch_wall=True), "kind": "cylinder"},
        {"id": "slab_touching_face", "rock": slab_touching_face(32, 32, 32), "kind": "slab"},
    ]
    for item in g:
        rock = item["rock"]
        item["shape"] = list(rock.shape)
        item["phi_input"] = float(rock.mean())
    return g


# --------------------------------------------------------------------------
# L1: tag parity
# --------------------------------------------------------------------------
def _recompress(geom: np.ndarray) -> np.ndarray:
    """palette -> compact labels -> palette, bit-for-bit identity check."""
    compact = decompress_tags(geom)
    out = np.zeros(compact.shape, dtype=np.int16)
    for sym, tag in enumerate((PORE_TAG, WALL_TAG, GRAIN_TAG, NW_TAG)):
        out[compact == sym] = tag
    return out


def tag_parity(rock: np.ndarray, **kwargs) -> dict:
    ours = edt_two_segment_tag(rock, **kwargs)
    ref_geom, ref_name = reference_tag_geometry(
        rock,
        swap_xz=kwargs.get("swap_xz", False),
        num_slices=kwargs.get("num_slices", 0),
        name=kwargs.get("name", "geom"),
        print_size=kwargs.get("print_size", False),
        nw_fluid_mask=kwargs.get("nw_fluid_mask"),
    )
    if ours.geom.shape != ref_geom.shape:
        return {"shape_ours": list(ours.geom.shape), "shape_ref": list(ref_geom.shape),
                "shape_match": False, "mismatch_pct": 100.0}
    n = ref_geom.size
    mismatch = int((ours.geom != ref_geom).sum())
    c_ours = tag_stats(ours.geom)
    c_ref = tag_stats(ref_geom)
    per_tag = {}
    for tag in (PORE_TAG, WALL_TAG, GRAIN_TAG, NW_TAG):
        o, r = c_ours[str(tag)], c_ref[str(tag)]
        per_tag[str(tag)] = {
            "ours": o, "ref": r,
            "rel_err_pct": (0.0 if r == 0 and o == 0 else (abs(o - r) / r * 100.0 if r else float("inf"))),
        }
    return {
        "shape": list(ours.geom.shape),
        "shape_match": True,
        "voxels": n,
        "mismatch_cells": mismatch,
        "mismatch_pct": 100.0 * mismatch / n,
        "per_tag": per_tag,
        "name_ours": ours.name,
        "name_ref": ref_name,
        "name_match": ours.name == ref_name,
        "shell_threshold": ours.shell_threshold,
        "rle_roundtrip_exact": bool(np.array_equal(
            rle_decode(rle_encode(ours.geom), ours.geom.shape), ours.geom)),
        "decompress_roundtrip_exact": bool(np.array_equal(
            _recompress(ours.geom), ours.geom)),
    }


# --------------------------------------------------------------------------
# L2: compression accounting
# --------------------------------------------------------------------------
def compression_arm(geom_id: str, tagged) -> dict:
    rep = compression_report(tagged)
    counts = tag_stats(tagged.geom)
    np.savez_compressed(RUNS / f"{geom_id}_tags.npz", geom=tagged.geom)
    on_disk = (RUNS / f"{geom_id}_tags.npz").stat().st_size
    return {
        "geometry": geom_id,
        **rep,
        "counts": counts,
        "npz_on_disk_bytes": int(on_disk),
        "bytes_per_voxel_npz": on_disk / rep["voxels"],
    }


# --------------------------------------------------------------------------
# L3: LBM drag with the two dynamics dispatches
# --------------------------------------------------------------------------
def _apply_body_force_3d(f, a_x, c, w):
    rho = f.sum(dim=0)
    return f + w * 3.0 * rho.unsqueeze(0) * c[:, 0].view(19, 1, 1, 1) * a_x


def lbm_sc_drag(n: int, phi: float, dispatch: str, shell_threshold: float = 2.0,
                steps: int | None = None, measure_last: int = 800,
                seed_dtype=torch.float32) -> dict:
    """Periodic SC sphere array driven by a body force; K = F/(6 pi nu a U_s)."""
    dt = seed_dtype
    c = D3Q19_C.to(dtype=dt)
    w = W_EXACT64.to(dtype=dt).view(19, 1, 1, 1)

    # rock image in (nx, ny, nz) = tagging orientation; torch field is (nz, ny, nx)
    rock = sc_sphere_mask(n, phi)
    tagged = edt_two_segment_tag(rock, name="sc", shell_threshold=shell_threshold)

    geom = tagged.geom
    wall_mask = torch.as_tensor((geom == WALL_TAG), dtype=torch.bool).permute(2, 1, 0).contiguous()
    grain_mask = torch.as_tensor((geom == GRAIN_TAG), dtype=torch.bool).permute(2, 1, 0).contiguous()
    fluid_mask = ~(wall_mask | grain_mask)
    solid_all = wall_mask | grain_mask

    n_fluid = int(fluid_mask.sum())
    phi_actual = 1.0 - n_fluid / float(n ** 3)
    a_nom = n * (3.0 * phi / (4.0 * math.pi)) ** (1.0 / 3.0)
    k_design = SC_TABLE.get(phi, array_drag_factor_from_phi(phi))
    a_body = 0.05 * NU * NU * 3.0 * math.pi * k_design / n_fluid  # Re_p ~ 0.05

    if dispatch == "oss_tag":
        bb_mask, interior = wall_mask, grain_mask
    elif dispatch == "plain_all_solid":
        bb_mask, interior = solid_all, torch.zeros_like(solid_all)
    else:
        raise ValueError(f"unknown dispatch {dispatch}")
    if steps is None:
        steps = 3000 if n >= 32 else 2000
    fluid_b = fluid_mask.unsqueeze(0)
    # Palabos NoDynamics semantics: the grain holds a FROZEN equilibrium
    # feq(rho=1, u=0).  Zero-filling instead destroys the populations that the
    # boundary segment reflects towards the interior and diverges (measured:
    # rho range +-1e20 after 20 steps at N=16) -- see docs, failure modes.
    nodyn = (W_EXACT64.to(dtype=dt).view(19, 1, 1, 1)
             * torch.ones((n, n, n), dtype=dt).unsqueeze(0))

    ones = torch.ones((n, n, n), dtype=dt)
    zeros = torch.zeros((n, n, n), dtype=dt)
    f = equilibrium3d(ones, zeros, zeros, zeros).to(dt)
    mass0 = float(f[:, fluid_mask].sum())
    bb_b = bb_mask.unsqueeze(0)

    um_hist: list[float] = []
    mass_hist: list[float] = []
    t0 = time.perf_counter()
    for step_i in range(steps):
        f = collide_bgk3d(f, TAU)
        f = stream3d(f)
        if dispatch == "oss_tag":
            f = torch.where(interior.unsqueeze(0), nodyn, f)  # NoDynamics restore
        f = torch.where(fluid_b, _apply_body_force_3d(f, a_body, c, w), f)
        f = bounce_back_cells_3d(f, bb_mask)
        if step_i >= steps - measure_last and step_i % 20 == 0:
            with torch.no_grad():
                _, ux, _, _ = macroscopic3d(f)
                um_hist.append(float(ux[fluid_mask].mean()))
                mass_hist.append(float(f[:, fluid_mask].sum()))
    wall_s = time.perf_counter() - t0

    um = float(np.mean(um_hist))
    U_s = (1.0 - phi_actual) * um
    F_inj = a_body * n_fluid
    K = F_inj / (6.0 * math.pi * NU * a_nom * U_s)
    k_ref = array_drag_factor_from_phi(phi)
    k_tab = SC_TABLE.get(phi)

    return {
        "N": n,
        "phi_nominal": phi,
        "phi_actual": phi_actual,
        "dispatch": dispatch,
        "shell_threshold": shell_threshold,
        "wall_cells": int(wall_mask.sum()),
        "grain_cells": int(grain_mask.sum()),
        "fluid_cells": n_fluid,
        "wall_plus_grain_fraction": 1.0 - n_fluid / n ** 3,
        "steps": steps,
        "wall_seconds": wall_s,
        "ms_per_step": 1e3 * wall_s / steps,
        "um_fluid": um,
        "U_s": U_s,
        "Re_p": U_s * 2.0 * a_nom / NU,
        "K": K,
        "K_hasimoto": k_ref,
        "err_vs_hasimoto_pct": abs(K - k_ref) / k_ref * 100.0,
        "K_table": k_tab,
        "err_vs_table_pct": (None if k_tab is None else abs(K - k_tab) / k_tab * 100.0),
        "mass_drift_rel_pct": 100.0 * max(abs(m / mass0 - 1.0) for m in mass_hist),
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    RUNS.mkdir(exist_ok=True)
    t_start = time.perf_counter()

    try:
        git_head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        git_head = None

    env = {
        "created_utc": utc_now(),
        "git_head": git_head,
        "tensorlbm": str(tensorlbm.__file__),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "threads": torch.get_num_threads(),
    }

    # ---------------- L1 parity (+ padding / swap / NW back-fill variants)
    geoms = geometries()
    parity_rows = []
    variants = [
        ("plain", {}),
        ("pad2", {"num_slices": 2, "print_size": True}),
        ("swap_xz", {"swap_xz": True}),
    ]
    for item in geoms:
        for vname, kwargs in variants:
            row = {"geometry": item["id"], "kind": item["kind"], "variant": vname,
                   "phi_input": item["phi_input"], **tag_parity(item["rock"], **kwargs)}
            parity_rows.append(row)
            print(f"[parity] {item['id']:32s} {vname:8s} mismatch={row['mismatch_pct']:.6f}% "
                  f"name_match={row['name_match']}", flush=True)

    # NW back-fill variant (non-wetting label 3 must survive as tag 2611)
    nw_rock = geoms[0]["rock"]
    nw_mask = np.zeros(nw_rock.shape, dtype=np.int8)
    nw_mask[:, : nw_rock.shape[1] // 4, :] = 3
    row = {"geometry": "sc_spheres_phi0.125_N32+nw_band", "kind": "two_phase", "variant": "nw_backfill",
           "phi_input": float(nw_rock.mean()), **tag_parity(nw_rock, nw_fluid_mask=nw_mask)}
    parities_nw = row["per_tag"]["2611"]
    parity_rows.append(row)
    print(f"[parity] nw_backfill mismatch={row['mismatch_pct']:.6f}% tag2611 ours={parities_nw['ours']} "
          f"ref={parities_nw['ref']}", flush=True)

    json.dump({"env": env, "rows": parity_rows}, open(RUNS / "parity.json", "w"), indent=2)

    # ---------------- L1b sensitivity: shell threshold and face forcing
    sens_rows = []
    probe = geoms[2]["rock"]  # random medium: the richest distance histogram
    for thr in (1.0, 1.5, math.sqrt(2.0), 2.0, 3.0):
        r = tag_parity(probe, shell_threshold=thr)
        sens_rows.append({"knob": "shell_threshold", "value": thr, "mismatch_pct": r["mismatch_pct"],
                          "wall_cells": r["per_tag"]["2609"]["ours"],
                          "grain_cells": r["per_tag"]["2610"]["ours"]})
    for bc in (True, False):
        r = tag_parity(probe, bc_faces=bc)
        sens_rows.append({"knob": "bc_faces", "value": bc, "mismatch_pct": r["mismatch_pct"],
                          "wall_cells": r["per_tag"]["2609"]["ours"],
                          "grain_cells": r["per_tag"]["2610"]["ours"]})
    json.dump({"rows": sens_rows}, open(RUNS / "sensitivity.json", "w"), indent=2)

    # ---------------- L2 compression accounting
    comp_rows = []
    for item in geoms:
        tagged = edt_two_segment_tag(item["rock"], name=item["id"])
        comp_rows.append(compression_arm(item["id"], tagged))
    json.dump({"rows": comp_rows}, open(RUNS / "compression.json", "w"), indent=2)

    # ---------------- L3 flow / dispatch equivalence + convergence + failure mode
    flow_rows = []
    # (a) dispatch equivalence at the reference point, N=24
    a = lbm_sc_drag(24, 0.125, "oss_tag")
    b = lbm_sc_drag(24, 0.125, "plain_all_solid")
    flow_rows += [a, b]
    delta_dispatch = abs(a["K"] - b["K"]) / a["K"] * 100.0
    print(f"[flow] N=24 tag={a['K']:.4f} plain={b['K']:.4f} delta={delta_dispatch:.4f}% "
          f"err_vs_hasimoto={a['err_vs_hasimoto_pct']:.2f}%", flush=True)

    # (b) resolution convergence of the tagging-based answer
    conv = [a]
    for n in (16, 32, 48):
        conv.append(lbm_sc_drag(n, 0.125, "oss_tag"))
        print(f"[flow] N={n} K={conv[-1]['K']:.4f} err={conv[-1]['err_vs_hasimoto_pct']:.2f}%", flush=True)

    # (c) failure mode: shell threshold below sqrt2 leaves diagonal links leaking
    thr1 = lbm_sc_drag(24, 0.125, "oss_tag", shell_threshold=1.0)
    flow_rows.append(thr1)
    print(f"[flow] threshold=1.0 mass_drift={thr1['mass_drift_rel_pct']:.3f}% "
          f"K={thr1['K']:.4f}", flush=True)

    json.dump({"rows": flow_rows, "convergence": conv, "delta_dispatch_pct": delta_dispatch},
              open(RUNS / "flow.json", "w"), indent=2)

    # ---------------- verdict
    max_parity = max(r["mismatch_pct"] for r in parity_rows)
    l1_pass = max_parity == 0.0
    l2_pass = delta_dispatch < 1.0 and thr1["mass_drift_rel_pct"] > 1.0
    flow_ok = max(c["err_vs_hasimoto_pct"] for c in conv) < 15.0
    finest = max(conv, key=lambda r: r["N"])
    summary = {
        "anchor": "XVAL-LBM-EDT-TWO-SEGMENT-GEOMETRY-TAG-COMPRESSION",
        "mission": 17728,
        "env": env,
        "oss_reference": {
            "repo": "je-santos/MPLBM-UT",
            "file": "src/python/mplbm_utils/pore_utils.py::create_geom_edist (lines 11-68)",
            "call_sites": "create_geom_for_palabos.py:47, create_geom_for_rel_perm.py:48,112,140",
            "dispatch": "src/2-phase_LBM/ShanChen.cpp:292-313",
            "palette": {"2608": "pore", "2609": "boundary", "2610": "grain", "2611": "nw phase"},
        },
        "L1_parity": {
            "arms": len(parity_rows),
            "max_mismatch_pct": max_parity,
            "all_names_match": all(r["name_match"] for r in parity_rows),
            "passed": l1_pass,
        },
        "L1b_sensitivity": sens_rows,
        "L2_compression": {
            "examples": comp_rows,
            "delta_K_dispatch_pct": delta_dispatch,
            "threshold1_mass_drift_pct": thr1["mass_drift_rel_pct"],
            "speedup_plain_over_tagged": b["ms_per_step"] / a["ms_per_step"],
            "passed": l2_pass,
        },
        "L3_flow": {
            "convergence": conv,
            "finest_N": finest["N"],
            "finest_K": finest["K"],
            "finest_err_vs_hasimoto_pct": finest["err_vs_hasimoto_pct"],
            "passed_15pct": flow_ok,
        },
        "wall_seconds": time.perf_counter() - t_start,
        "rows": {"parity": parity_rows, "flow": flow_rows, "compression": comp_rows,
                 "sensitivity": sens_rows},
    }
    json.dump(summary, open(HERE / "summary.json", "w"), indent=2)

    with open(HERE / "table.csv", "w") as fh:
        fh.write("layer,arm,metric,value,unit\n")
        for r in parity_rows:
            fh.write(f"L1,{r['geometry']}/{r['variant']},tag_mismatch,{r['mismatch_pct']:.8f},%\n")
        for r in sens_rows:
            fh.write(f"L1b,{r['knob']}={r['value']},tag_mismatch,{r['mismatch_pct']:.4f},%\n")
        for r in comp_rows:
            fh.write(f"L2,{r['geometry']},ratio_float64_to_int16,{r['ratio_float64_to_int16']:.2f},x\n")
            fh.write(f"L2,{r['geometry']},bytes_per_voxel_npz,{r['bytes_per_voxel_npz']:.3f},B\n")
            fh.write(f"L2,{r['geometry']},entropy,{r['entropy_bits_per_voxel']:.4f},bit/voxel\n")
        fh.write(f"L2,dispatch_equivalence,delta_K,{delta_dispatch:.6f},%\n")
        for r in conv + [thr1]:
            fh.write(f"L3,N{r['N']}/{r['dispatch']}/thr{r['shell_threshold']},K,{r['K']:.5f},-\n")
            fh.write(f"L3,N{r['N']}/{r['dispatch']}/thr{r['shell_threshold']},err_vs_hasimoto,"
                     f"{r['err_vs_hasimoto_pct']:.4f},%\n")

    print(json.dumps({k: summary[k] for k in ("L1_parity", "L2_compression", "L3_flow")},
                     indent=2, default=str))
    print(f"[done] wall={summary['wall_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())