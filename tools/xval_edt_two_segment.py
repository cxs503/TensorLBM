#!/usr/bin/env python3
"""XVAL runner: EDT two-segment geometry tagging vs je-santos/MPLBM-UT.

Anchor: XVAL-LBM-EDT-TWO-SEGMENT-GEOMETRY-TAG-COMPRESSION

Reference implementation (``ref``)
---------------------------------
A line-by-line transcription of the OSS geometry pipeline, pinned to
``je-santos/MPLBM-UT`` @ ``a9fc496688486c8bbc882ec7d780619f6712fa54``:

* ``src/python/mplbm_utils/create_geom_for_palabos.py`` (blob ``0cca7926``)
  — reads/crops the rock image, ``rock = rock/3``, ``erase_regions``,
  ``create_geom_edist``, ``rock4sim.flatten().tofile(...)``.
* ``src/python/mplbm_utils/pore_utils.py`` (blob ``8d713e78``)
  — ``create_geom_edist`` / ``create_nw_fluid_mask`` / ``erase_regions``.

The transcription is kept *here*, in the runner, and deliberately not in the
library: the library is our independent implementation, the runner's copy is
the yardstick.

Candidate implementation (``got``)
---------------------------------
:mod:`tensorlbm.edt_geometry_tags` — independent EDT (k-d tree), vectorised
class assignment, bit-packed tag codec, per-grain id side channel.

Outputs (all under ``--out``, default ``runs/xval_edt_two_segment``)
--------------------------------------------------------------------
``summary.json``   machine-readable numbers + provenance + repro command
``table.csv``      ref vs got table with per-row error and pass/fail (<15%)
``report.md``      six-section campaign report
``geometry/``      raw ``int16`` ``.dat`` + JSON sidecar per case (got)
``figures/``       class map, distance field vs analytic, convergence, sensitivity

Usage::

    python tools/xval_edt_two_segment.py --out runs/xval_edt_two_segment
    python tools/xval_edt_two_segment.py --quick        # small grids, no figures
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.edt_geometry_tags import (  # noqa: E402
    DEPTH_SPLIT,
    PALABOS_BOUNDARY,
    PALABOS_GRAIN,
    PALABOS_PORE,
    build_geometry_tags,
    decode_palabos_tags,
    distance_to_segments,
    exact_edt,
    scipy_edt,
    two_segment_mask,
    write_palabos_geometry,
)

OSS = {
    "repo": "je-santos/MPLBM-UT",
    "commit": "a9fc496688486c8bbc882ec7d780619f6712fa54",
    "commit_date": "2023-12-01T16:44:09Z",
    "files": {
        "src/python/mplbm_utils/pore_utils.py": "8d713e7849e9084f945eeb145d45844a55214e17",
        "src/python/mplbm_utils/create_geom_for_palabos.py": "0cca792603261f2d6d619fc82894087d4ba0a489",
    },
    "functions": ["create_geom_edist", "create_nw_fluid_mask", "erase_regions"],
}


# ---------------------------------------------------------------------------
# OSS reference transcription (ref)
# ---------------------------------------------------------------------------
def oss_erase_regions(rock: np.ndarray) -> np.ndarray:
    """``pore_utils.erase_regions`` (skimage ``label(rock, background=1)``)."""
    from scipy import ndimage as ndi

    rock = np.array(rock, copy=True)
    # skimage.measure.label(rock, background=1, connectivity=1): label regions
    # of ``rock != 1`` with 6-connectivity, then zero every label > 1.
    labels, _ = ndi.label(rock != 1, structure=ndi.generate_binary_structure(3, 1))
    rock[labels > 1] = 0
    return rock


def oss_create_geom_edist(
    rock: np.ndarray,
    *,
    name: str = "geom",
    num_slices: int = 0,
    swap_xz: bool = False,
    print_size: bool = True,
    nw_fluid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Verbatim transcription of ``pore_utils.create_geom_edist``.

    ``rock``: 0 = pore / wetting fluid, >0 = grain, 3 = non-wetting seeds.
    Returns the ``int16`` Palabos geometry (codes 2608..2611) and its name.
    """
    from scipy.ndimage import distance_transform_edt as edist

    if swap_xz:
        rock = rock.transpose([2, 1, 0])

    erock = edist(rock)                      # OSS: erock = edist(rock)

    # make sure all the BCs have bounce back nodes
    erock[0, :, :] = 1
    erock[:, 0, :] = 1
    erock[:, :, 0] = 1
    erock[-1, :, :] = 1
    erock[:, -1, :] = 1
    erock[:, :, -1] = 1

    erock[rock == 0] = 0                     # re open the pores

    erock[(erock > 0) * (erock < 2)] = 1     # boundary class
    erock[erock > 1] = 2                     # grain interior

    if num_slices:
        erock = np.pad(erock, [(num_slices, num_slices), (0, 0), (0, 0)])

    if print_size:
        size = erock.shape
        geom_name = f"{name}_{size[0]}_{size[1]}_{size[2]}"
    else:
        geom_name = name

    erock = erock.astype(np.int16)
    erock[erock == 0] = 2608
    erock[erock == 1] = 2609
    erock[erock == 2] = 2610
    if nw_fluid_mask is not None:
        erock[nw_fluid_mask == 3] = 2611
    return erock.astype(np.int16), geom_name


def oss_create_nw_fluid_mask(
    rock: np.ndarray,
    *,
    num_slices: int = 0,
    swap_xz: bool = False,
    inlet_fluid: float = 0.0,
    outlet_fluid: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Transcription of ``pore_utils.create_nw_fluid_mask`` (unconditional call
    in ``create_geom_for_palabos``)."""
    rock_tmp = np.copy(rock)
    if swap_xz:
        rock_tmp = rock_tmp.transpose([2, 1, 0])
    if num_slices:
        rock_tmp = np.pad(
            rock_tmp,
            [(num_slices, num_slices), (0, 0), (0, 0)],
            "constant",
            constant_values=(inlet_fluid, outlet_fluid),
        )
    fluid_mask = np.where(rock_tmp == 3, rock_tmp, 0)
    rock = np.where(rock == 3, 0, rock)
    return rock, fluid_mask


def oss_pipeline(
    solid: np.ndarray,
    *,
    name: str = "geom",
    num_slices: int = 0,
    swap_xz: bool = False,
    nw_seeds: np.ndarray | None = None,
    print_size: bool = True,
) -> tuple[np.ndarray, str]:
    """``create_geom_for_palabos`` head: rock/3 -> erase_regions ->
    create_nw_fluid_mask -> create_geom_edist."""
    rock = solid.astype(np.float64) / 3.0         # OSS: rock = rock/3
    if nw_seeds is not None:
        rock = np.where(np.asarray(nw_seeds, dtype=bool), 3.0, rock)
    rock = oss_erase_regions(rock)
    rock, nw_mask = oss_create_nw_fluid_mask(rock, num_slices=num_slices, swap_xz=swap_xz)
    return oss_create_geom_edist(
        rock, name=name, num_slices=num_slices, swap_xz=swap_xz,
        print_size=print_size, nw_fluid_mask=nw_mask,
    )


# ---------------------------------------------------------------------------
# Case geometries
# ---------------------------------------------------------------------------
def case_two_segment_L(scale: float = 1.0) -> dict:
    """L-bend micro-model: one horizontal + one vertical capsule segment."""
    nz, ny, nx = int(20 * scale), int(48 * scale), int(80 * scale)
    r = 5.0 * scale
    segs = [
        ((nz / 2, ny / 2, 0.20 * nx), (nz / 2, ny / 2, 0.55 * nx)),
        ((nz / 2, ny / 2, 0.55 * nx), (nz / 2, 0.75 * ny, 0.55 * nx)),
    ]
    mask = two_segment_mask((nz, ny, nx), segs, radius=r)
    return {"name": "two_segment_L", "solid": mask, "segments": segs, "radius": r,
            "analytic": "capsule_union_on_grid"}


def case_two_segment_V(scale: float = 1.0) -> dict:
    """V-bend micro-model: two capsules meeting at 60 degrees."""
    nz, ny, nx = int(20 * scale), int(48 * scale), int(80 * scale)
    r = 4.5 * scale
    segs = [
        ((nz / 2, 0.20 * ny, 0.22 * nx), (nz / 2, 0.78 * ny, 0.50 * nx)),
        ((nz / 2, 0.78 * ny, 0.50 * nx), (nz / 2, 0.20 * ny, 0.78 * nx)),
    ]
    mask = two_segment_mask((nz, ny, nx), segs, radius=r)
    return {"name": "two_segment_V", "solid": mask, "segments": segs, "radius": r,
            "analytic": "capsule_union_on_grid"}


def case_capsule(radius: float = 12.0, length: float = 20.0) -> dict:
    """Single capsule aligned with x: exact volume pi r^2 L + 4/3 pi r^3."""
    r = float(radius)
    L = float(length)
    pad = 3.0
    nz = int(2 * r + 2 * pad)
    ny = int(2 * r + 2 * pad)
    nx = int(L + 2 * r + 2 * pad)
    p0 = (nz / 2.0, ny / 2.0, pad + r)
    p1 = (nz / 2.0, ny / 2.0, pad + r + L)
    mask = two_segment_mask((nz, ny, nx), [(p0, p1)], radius=r)
    vol_exact = np.pi * r * r * L + 4.0 / 3.0 * np.pi * r**3
    return {
        "name": f"capsule_r{int(round(r))}_L{int(round(L))}",
        "solid": mask,
        "segments": [(p0, p1)],
        "radius": r,
        "analytic": "exact_capsule_volume",
        "volume_exact": float(vol_exact),
    }


def case_slab(t_plate: int = 4, gap: int = 24) -> dict:
    """Two parallel plates, exact solid fraction = 2 t / (gap + 2 t)."""
    ny = gap + 2 * t_plate
    nz, nx = 20, 40
    mask = np.zeros((nz, ny, nx), dtype=bool)
    mask[:, :t_plate, :] = True
    mask[:, -t_plate:, :] = True
    frac_exact = 2.0 * t_plate / ny
    return {"name": f"slab_t{t_plate}_gap{gap}", "solid": mask,
            "analytic": "exact_solid_fraction", "fraction_exact": float(frac_exact)}


def case_sphere_pack(r: float = 8.0) -> dict:
    """Three-sphere bead pack (MPLBM-UT bead-pack analogue), exact volume."""
    pad = 4.0
    nz = int(2 * r + 2 * pad)
    ny = int(6 * r + 2 * pad)
    nx = int(2 * r + 2 * pad)
    centres = [(nz / 2, pad + r, nx / 2),
               (nz / 2, pad + 3 * r, nx / 2),
               (nz / 2, pad + 5 * r, nx / 2)]
    grids = np.meshgrid(*[np.arange(s) for s in (nz, ny, nx)], indexing="ij")
    pts = np.stack([g.ravel() for g in grids], axis=1).astype(np.float64)
    d = np.full(pts.shape[0], np.inf)
    for c in centres:
        d = np.minimum(d, np.linalg.norm(pts - np.asarray(c), axis=1))
    mask = (d < r).reshape(nz, ny, nx)
    vol_exact = 3 * 4.0 / 3.0 * np.pi * r**3
    return {"name": f"sphere_pack_r{int(round(r))}", "solid": mask,
            "analytic": "exact_sphere_union_volume", "volume_exact": float(vol_exact)}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compare_case(case: dict, *, num_slices: int = 0, depth_split: float = DEPTH_SPLIT,
                 nw_seeds: np.ndarray | None = None, swap_xz: bool = False) -> dict:
    """Run ref and got on one case and compute the comparison row-set."""
    solid = case["solid"]
    n_vox = int(solid.size)

    t0 = time.perf_counter()
    got = build_geometry_tags(solid, num_slices=num_slices, depth_split=depth_split,
                              nw_mask=None if nw_seeds is None else np.where(nw_seeds, 3, 0),
                              swap_xz=swap_xz, geom_name=case["name"])
    t_got = time.perf_counter() - t0

    t0 = time.perf_counter()
    ref_tags, ref_name = oss_pipeline(solid, name=case["name"], num_slices=num_slices,
                                      nw_seeds=nw_seeds, swap_xz=swap_xz)
    t_ref = time.perf_counter() - t0

    # ---- tag agreement (our classes vs OSS int16 codes) ----
    got_classes = got.classes
    ref_classes = decode_palabos_tags(ref_tags)
    agree = float(np.mean(got_classes == ref_classes))
    n_disagree = int(np.count_nonzero(got_classes != ref_classes))

    # ---- EDT algorithm cross-check (k-d tree vs scipy) ----
    d_exact = exact_edt(solid)
    d_scipy = scipy_edt(solid)
    dd = np.abs(d_exact - d_scipy)
    finite = np.isfinite(dd)

    # ---- analytic distance for the capsule cases ----
    analytic = {}
    analytic_ok = (case["analytic"] in ("capsule_union_on_grid", "exact_capsule_volume")
                   and num_slices == 0 and nw_seeds is None and not swap_xz)
    if analytic_ok:
        shape = solid.shape
        grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
        pts = np.stack([g.ravel() for g in grids], axis=1).astype(np.float64)
        segs = [(np.asarray(p0, float), np.asarray(p1, float)) for p0, p1 in case["segments"]]
        phi = distance_to_segments(pts, segs, radius=case["radius"]).reshape(shape)
        inside_analytic = phi < 0.0
        mask_agree = float(np.mean(inside_analytic == solid))
        cls_analytic = np.zeros(shape, dtype=np.uint8)
        cls_analytic[inside_analytic & (phi > -depth_split)] = 1     # boundary band
        cls_analytic[inside_analytic & (phi <= -depth_split)] = 2    # interior
        analytic["class_agreement_vs_analytic"] = float(np.mean(cls_analytic == got_classes))
        # distance relative error on solid cells deeper than the split
        deep = solid & (phi <= -depth_split)
        denom = np.abs(phi[deep])
        rel = np.abs(d_exact[deep] - denom) / denom if deep.any() else np.array([0.0])
        abs_err = np.abs(d_exact[deep] - denom) if deep.any() else np.array([0.0])
        analytic["n_deep_cells"] = int(deep.sum())
        analytic["dist_rel_err_mean_deep"] = float(rel.mean())
        analytic["dist_rel_err_max_deep"] = float(rel.max())
        analytic["dist_abs_err_mean_deep"] = float(abs_err.mean())
        analytic["dist_abs_err_max_deep"] = float(abs_err.max())
        # deep band: lattice EDT measures centre-to-centre, so the discretisation
        # bias (~0.5..1 cell) dominates near the split; the far field is the
        # physically meaningful distance comparison.
        deep4 = solid & (phi <= -4.0)
        if deep4.any():
            d4 = np.abs(phi[deep4])
            r4 = np.abs(d_exact[deep4] - d4) / d4
            analytic["n_deep4_cells"] = int(deep4.sum())
            analytic["dist_rel_err_mean_deep4"] = float(r4.mean())
            analytic["dist_rel_err_max_deep4"] = float(r4.max())
        analytic["mask_agreement_vs_analytic"] = mask_agree

    # ---- volume / fraction vs analytic ----
    frac_got = float(np.count_nonzero(got.classes > 0) / got.classes.size)
    solid_frac_raw = float(np.count_nonzero(solid) / n_vox)
    vol = {}
    if "volume_exact" in case:
        vol_got = float(np.count_nonzero(solid)) * 1.0
        vol["volume_ref"] = case["volume_exact"]
        vol["volume_got"] = vol_got
        vol["volume_rel_err"] = abs(vol_got - case["volume_exact"]) / case["volume_exact"]
    if "fraction_exact" in case:
        vol["fraction_ref"] = case["fraction_exact"]
        vol["fraction_got"] = frac_got
        vol["fraction_rel_err"] = abs(frac_got - case["fraction_exact"]) / case["fraction_exact"]

    packed = got.pack_classes()
    row = {
        "case": case["name"],
        "shape": list(got.shape),
        "voxels": int(got.classes.size),
        "ref_impl": "MPLBM-UT@a9fc4966 create_geom_edist (transcription)",
        "ref_geom_name": ref_name,
        "got_geom_name": got.geom_name,
        "tag_agreement": agree,
        "tag_disagreement_count": n_disagree,
        "tag_error_pct": 100.0 * (1.0 - agree),
        "dist_max_absdiff_edt_algorithms": float(np.nanmax(dd[finite])) if finite.any() else 0.0,
        "dist_rms_edt_algorithms": float(np.sqrt(np.mean(dd[finite] ** 2))) if finite.any() else 0.0,
        "counts": got.counts(),
        "fractions": got.fractions(),
        "solid_fraction_voxel": solid_frac_raw,
        "class_fraction_padded": frac_got,
        "compression": got.meta["compression"],
        "packed_bytes": packed.nbytes,
        "analytic": analytic,
        "volume": vol,
        "timing_s": {"got": t_got, "ref": t_ref},
    }
    row["checks"] = _checks(row)
    return {"row": row, "got": got, "ref_tags": ref_tags}


def _checks(row: dict) -> list[dict]:
    """ref vs got rows: relative rows use the owner <15% criterion, absolute rows
    (lattice-unit discretisation bias) carry their own cell-unit threshold."""
    checks = []

    def add_rel(name: str, ref, got, err_pct: float, threshold_pct: float = 15.0) -> None:
        checks.append({
            "quantity": name,
            "ref": float(ref),
            "got": float(got),
            "error_pct": float(err_pct),
            "threshold_pct": float(threshold_pct),
            "kind": "rel",
            "pass": bool(err_pct < threshold_pct),
        })

    def add_abs(name: str, ref, got, err_abs: float, threshold_abs: float) -> None:
        checks.append({
            "quantity": name,
            "ref": float(ref),
            "got": float(got),
            "error_abs": float(err_abs),
            "threshold_abs": float(threshold_abs),
            "kind": "abs",
            "pass": bool(err_abs <= threshold_abs),
        })

    add_rel("tag agreement vs OSS pipeline (fraction, higher better)",
            1.0, row["tag_agreement"], 100.0 * (1.0 - row["tag_agreement"]))
    add_rel("EDT algorithm: max |k-d tree - scipy| (cells, vs 1-cell band)",
            0.0, row["dist_max_absdiff_edt_algorithms"],
            100.0 * row["dist_max_absdiff_edt_algorithms"] / 1.0)
    a = row["analytic"]
    if "class_agreement_vs_analytic" in a:
        add_rel("two-class agreement vs analytic geometry (fraction)",
                1.0, a["class_agreement_vs_analytic"],
                100.0 * (1.0 - a["class_agreement_vs_analytic"]))
        add_rel("EDT depth rel. error, cells deeper than split (mean)",
                0.0, a["dist_rel_err_mean_deep"], 100.0 * a["dist_rel_err_mean_deep"])
        if "dist_rel_err_mean_deep4" in a:
            add_rel("EDT depth rel. error, cells deeper than 4 cells (mean)",
                    0.0, a["dist_rel_err_mean_deep4"],
                    100.0 * a["dist_rel_err_mean_deep4"])
        add_abs("EDT depth absolute bias, cells deeper than split (max, cells)",
                a["dist_rel_err_max_deep"] * 0.0, a["dist_abs_err_max_deep"],
                a["dist_abs_err_max_deep"], 1.5)
    v = row["volume"]
    if "volume_rel_err" in v:
        add_rel("displaced volume (voxel count) vs exact",
                v["volume_ref"], v["volume_got"], 100.0 * v["volume_rel_err"])
    if "fraction_rel_err" in v:
        add_rel("solid fraction vs exact",
                v["fraction_ref"], v["fraction_got"], 100.0 * v["fraction_rel_err"])
    return checks


def _capsule_phase_average(radius: float, length: float, n_phase_axis: int = 2) -> dict:
    """Volume-error statistics of a voxelised capsule over sub-voxel lattice phases.

    A single lattice phase biases the voxel count by O(h) in an essentially
    random direction; averaging the exact capsule over a ``n^3`` grid of
    sub-voxel centre offsets removes that noise so the convergence order of the
    discretisation itself can be measured.
    """
    r = float(radius)
    L = float(length)
    pad = 3.0
    shape = (int(2 * r + 2 * pad), int(2 * r + 2 * pad), int(L + 2 * r + 2 * pad))
    base_p0 = np.array([shape[0] / 2.0, shape[1] / 2.0, pad + r])
    base_p1 = base_p0 + np.array([0.0, 0.0, L])
    exact = np.pi * r * r * L + 4.0 / 3.0 * np.pi * r**3
    offs = np.linspace(0.0, 1.0, n_phase_axis, endpoint=False)
    counts = []
    for oz in offs:
        for oy in offs:
            for ox in offs:
                off = np.array([oz, oy, ox])
                m = two_segment_mask(shape, [(base_p0 + off, base_p1 + off)], radius=r)
                counts.append(float(np.count_nonzero(m)))
    counts = np.asarray(counts)
    mean = float(counts.mean())
    return {
        "radius_cells": r,
        "length_cells": L,
        "h": 1.0 / r,
        "voxels": int(np.prod(shape)),
        "volume_ref": float(exact),
        "volume_got_mean": mean,
        "volume_got_std": float(counts.std(ddof=1)) if counts.size > 1 else 0.0,
        "rel_err": float(abs(mean - exact) / exact),
        "n_phases": int(counts.size),
    }


def convergence_study(base_radius: float = 6.0) -> dict:
    """Grid refinement of a capsule: phase-averaged volume error vs h = 1/r."""
    rows = [_capsule_phase_average(base_radius * f, 12.0 * f) for f in (1.0, 1.5, 2.0)]
    h = np.array([r["h"] for r in rows])
    e = np.array([max(r["rel_err"], 1e-12) for r in rows])
    slope = float(np.polyfit(np.log(h), np.log(e), 1)[0])
    return {"rows": rows, "loglog_slope": slope}


def sensitivity_study() -> dict:
    """Sensitivity to the boundary/interior threshold and to grid resolution."""
    case = case_two_segment_L(1.0)
    splits = []
    for s in (1.5, 2.0, 2.5, 3.0):
        r = compare_case(case, depth_split=s)["row"]
        splits.append({
            "depth_split": s,
            "boundary_frac": r["fractions"]["boundary"],
            "grain_frac": r["fractions"]["grain"],
            "tag_agreement": r["tag_agreement"],
        })
    resolutions = []
    for scale in (0.75, 1.0, 1.5):
        c = case_two_segment_L(scale)
        got = build_geometry_tags(c["solid"], geom_name=f"two_segment_L_x{scale}")
        resolutions.append({
            "scale": scale,
            "voxels": int(c["solid"].size),
            "boundary_frac": got.fractions()["boundary"],
            "grain_frac": got.fractions()["grain"],
            "boundary_over_solid": float(
                got.counts()["boundary"] / max(1, got.counts()["boundary"] + got.counts()["grain"])
            ),
        })
    return {"depth_split": splits, "resolution": resolutions}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(out_dir: Path, primary: dict, conv: dict, sens: dict) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figs = []
    got = primary["got"]
    case = primary["case"]

    # --- fig 1: class map of the two-segment geometry (incl. OSS tags) ---
    mid = got.tags.shape[0] // 2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].imshow(got.tags[mid], origin="lower", cmap="viridis", interpolation="nearest")
    axes[0].set_title(f"{case['name']}: Palabos tags {PALABOS_PORE}/{PALABOS_BOUNDARY}/{PALABOS_GRAIN}")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    cls = got.classes[mid]
    im = axes[1].imshow(cls, origin="lower", cmap="tab10", vmin=0, vmax=2, interpolation="nearest")
    axes[1].set_title("two-segment segmentation (0 pore / 1 boundary / 2 interior)")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    fig.colorbar(im, ax=axes[1], shrink=0.85)
    fig.tight_layout()
    p = fig_dir / "fig1_class_map.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    figs.append({"name": "fig1_class_map", "path": str(p),
                 "caption": "Palabos geometry tags and two-class segmentation of the two-segment (L-bend) micro-model."})

    # --- fig 2: lattice EDT vs analytic signed distance ---
    if case.get("segments"):
        shape = case["solid"].shape
        grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
        pts = np.stack([g.ravel() for g in grids], axis=1).astype(np.float64)
        segs = [(np.asarray(a, float), np.asarray(b, float)) for a, b in case["segments"]]
        phi = distance_to_segments(pts, segs, radius=case["radius"]).reshape(shape)
        d_lat = exact_edt(case["solid"])
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
        im0 = axes[0].imshow(-phi[mid], origin="lower", cmap="magma")
        axes[0].set_title("analytic depth -phi (capsule union)")
        fig.colorbar(im0, ax=axes[0], shrink=0.8)
        im1 = axes[1].imshow(d_lat[mid], origin="lower", cmap="magma")
        axes[1].set_title("lattice EDT depth (k-d tree)")
        fig.colorbar(im1, ax=axes[1], shrink=0.8)
        diff = d_lat[mid] - (-phi[mid])
        im2 = axes[2].imshow(diff, origin="lower", cmap="coolwarm", vmin=-1, vmax=1)
        axes[2].set_title("difference (lattice - analytic), cells")
        fig.colorbar(im2, ax=axes[2], shrink=0.8)
        for ax in axes:
            ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.tight_layout()
        p = fig_dir / "fig2_distance_field.png"
        fig.savefig(p, dpi=140); plt.close(fig)
        figs.append({"name": "fig2_distance_field", "path": str(p),
                     "caption": "Mid-plane depth: analytic capsule distance vs lattice EDT vs difference."})

    # --- fig 3: convergence ---
    rows = conv["rows"]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    r = np.array([x["radius_cells"] for x in rows])
    e = np.array([x["rel_err"] for x in rows])
    ax.loglog(1.0 / r, e, "o-", label=f"volume rel. error (slope {conv['loglog_slope']:.2f})")
    ax.set_xlabel("1 / radius_cells (h)"); ax.set_ylabel("relative volume error")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout()
    p = fig_dir / "fig3_convergence.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    figs.append({"name": "fig3_convergence", "path": str(p),
                 "caption": "Displaced-volume error of the voxelised capsule vs 1/h (log-log)."})

    # --- fig 4: sensitivity ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    d = sens["depth_split"]
    axes[0].plot([x["depth_split"] for x in d], [x["boundary_frac"] for x in d], "o-", label="boundary")
    axes[0].plot([x["depth_split"] for x in d], [x["grain_frac"] for x in d], "s-", label="interior")
    axes[0].set_xlabel("depth_split (cells)"); axes[0].set_ylabel("class fraction")
    axes[0].grid(alpha=0.3); axes[0].legend(); axes[0].set_title("threshold sensitivity")
    res = sens["resolution"]
    axes[1].plot([x["scale"] for x in res], [x["boundary_over_solid"] for x in res], "o-")
    axes[1].set_xlabel("grid scale"); axes[1].set_ylabel("boundary / solid")
    axes[1].grid(alpha=0.3); axes[1].set_title("resolution sensitivity")
    fig.tight_layout()
    p = fig_dir / "fig4_sensitivity.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    figs.append({"name": "fig4_sensitivity", "path": str(p),
                 "caption": "Boundary/interior split vs threshold and vs grid resolution."})
    return figs


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report_md(out_dir: Path, summary: dict) -> Path:
    rows = summary["cases"]
    lines = [
        "# XVAL — EDT two-segment geometry tagging vs je-santos/MPLBM-UT",
        "",
        f"* anchor: `XVAL-LBM-EDT-TWO-SEGMENT-GEOMETRY-TAG-COMPRESSION`",
        f"* ref: `{OSS['repo']}` @ `{OSS['commit']}` ({OSS['commit_date']}) — "
        "`mplbm_utils/pore_utils.py::create_geom_edist`",
        f"* got: `src/tensorlbm/edt_geometry_tags.py::build_geometry_tags`",
        f"* generated: {summary['generated_utc']}  |  python {summary['env']['python']}",
        f"* repro: `{summary['repro_cmd']}`",
        "",
        "## 1  Problem and reference source",
        "",
        "MPLBM-UT prepares a Palabos geometry by (i) computing the Euclidean distance",
        "transform inside the grain phase, (ii) forcing the six outer faces to depth 1 so",
        "every boundary plane has bounce-back nodes, (iii) re-opening the pore cells,",
        "(iv) segmenting the field into **two depth classes** (`0 < d < 2` boundary,",
        "`d >= 2` interior) and (v) quantising to `int16` Palabos codes",
        "2608 pore / 2609 boundary / 2610 grain (+2611 non-wetting seeds).  Step (iv)+(v)",
        "is the *tag compression*: a float64 distance field (8 B/voxel) collapses to a",
        "4-symbol int16 map (2 B/voxel).",
        "",
        "## 2  Method",
        "",
        "`got` re-implements the recipe independently: exact EDT through a k-d tree over",
        "fluid-voxel centres (`exact_edt`), vectorised class assignment, lossless 2-bit",
        "packing of the class map (`compress_label_map`), optional per-grain id side",
        "channel, and the OSS on-disk format (`<name>_<nz>_<ny>_<nx>.dat`, int16 raw).",
        "`ref` is a line-by-line transcription of the OSS functions executed in-process",
        "(`oss_create_geom_edist` in `tools/xval_edt_two_segment.py`).",
        "",
        "## 3  Parameters",
        "",
        "| item | value |",
        "| --- | --- |",
        f"| lattice spacing | 1 (lattice units), D3Q19 geometry convention |",
        f"| depth split | {DEPTH_SPLIT} cells (OSS) |",
        "| BC faces | forced to depth 1 on all six faces (OSS) |",
        "| Palabos codes | 2608 / 2609 / 2610 / 2611 |",
        "| two-segment radius | L-bend 5 cells, V-bend 4.5, capsule 12, spheres 8 |",
        "| threshold | 15% relative error (owner standard) |",
        "",
        "## 4  ref vs got",
        "",
        "| quantity | ref | got | error | pass |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        for c in r["checks"]:
            ok = "PASS" if c["pass"] else "**FAIL**"
            if c["kind"] == "abs":
                err = f"{c['error_abs']:.3f} cells (limit {c['threshold_abs']:.2f})"
            else:
                err = f"{c['error_pct']:.3e} % (limit {c['threshold_pct']:.0f}%)"
            lines.append(
                f"| {r['case']}: {c['quantity']} | {c['ref']:.6g} | {c['got']:.6g} | "
                f"{err} | {ok} |"
            )
    lines += [
        "",
        "### 4b  EDT algorithm cross-check and compression",
        "",
        "| case | max abs diff (k-d tree vs scipy) | RMS | float64 B | int16 B | 2-bit B | ratio |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        c = r["compression"]
        ratio = c.get("ratio_vs_float64")
        lines.append(
            f"| {r['case']} | {r['dist_max_absdiff_edt_algorithms']:.3e} | "
            f"{r['dist_rms_edt_algorithms']:.3e} | {c['float64_edt_bytes']} | "
            f"{c['int16_codes_bytes']} | {c['packed_bytes']} | "
            f"{ratio:.2f}x |" if ratio else
            f"| {r['case']} | {r['dist_max_absdiff_edt_algorithms']:.3e} | "
            f"{r['dist_rms_edt_algorithms']:.3e} | {c['float64_edt_bytes']} | "
            f"{c['int16_codes_bytes']} | {c['packed_bytes']} | n/a |"
        )
    conv = summary["convergence"]
    sens = summary["sensitivity"]
    lines += [
        "",
        "## 5  Convergence and sensitivity",
        "",
        "Capsule displaced-volume error vs exact `pi r^2 L + 4/3 pi r^3`:",
        "",
        "| radius (cells) | voxels | ref volume | got voxels (phase-averaged) | rel err |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in conv["rows"]:
        lines.append(
            f"| {r['radius_cells']:.1f} | {r['voxels']} | {r['volume_ref']:.3f} | "
            f"{r['volume_got_mean']:.1f} +/- {r['volume_got_std']:.1f} (n={r['n_phases']}) | "
            f"{r['rel_err']:.3e} |"
        )
    lines += [
        "",
        f"log-log slope of |volume error| vs h = **{conv['loglog_slope']:.2f}** "
        "(second order expected for a voxelised curved surface).",
        "",
        "Depth-split sensitivity (L-bend case):",
        "",
        "| depth_split | boundary frac | interior frac | tag agreement vs OSS |",
        "| --- | --- | --- | --- |",
    ]
    for s in sens["depth_split"]:
        lines.append(
            f"| {s['depth_split']:.1f} | {s['boundary_frac']:.4f} | "
            f"{s['grain_frac']:.4f} | {s['tag_agreement']:.6f} |"
        )
    lines += [
        "",
        "Resolution sensitivity (L-bend case):",
        "",
        "| grid scale | voxels | boundary frac | boundary/solid |",
        "| --- | --- | --- | --- |",
    ]
    for s in sens["resolution"]:
        lines.append(
            f"| {s['scale']:.2f} | {s['voxels']} | {s['boundary_frac']:.4f} | "
            f"{s['boundary_over_solid']:.4f} |"
        )
    lines += [
        "",
        "## 6  Failure modes, differences and artefacts",
        "",
        "Differences vs OSS (see `docs/xval/EDT_TWO_SEGMENT_GEOMETRY_TAG_COMPRESSION.md`",
        "for the full table): per-grain identity is not preserved by the 4-class map, so",
        "mixed wettability needs the id side channel added here; `erase_regions` is a",
        "no-op for binary images; the OSS code raises `NotImplementedError` for",
        "`scale_2`/`add_mesh` (we keep the same semantics by not implementing them).",
        "",
        "Failure modes: all-solid domain yields `inf` depths (same as scipy) and is",
        "rejected upstream; anisotropy is honoured through `spacing`; `swap_xz` is",
        "applied before padding exactly like OSS.",
        "",
        "Artifacts:",
        "",
    ]
    for a in summary["artifacts"]:
        lines.append(f"* `{a}`")
    lines += ["", "Figures:", ""]
    for f in summary.get("figures", []):
        lines.append(f"* `{f['path']}` — {f['caption']}")
    p = out_dir / "report.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def write_table_csv(out_dir: Path, rows: list[dict]) -> Path:
    import csv

    p = out_dir / "table.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "quantity", "kind", "ref", "got", "error_pct",
                    "threshold_pct", "error_abs", "threshold_abs", "pass"])
        for r in rows:
            for c in r["checks"]:
                w.writerow([
                    r["case"], c["quantity"], c["kind"], f"{c['ref']:.10g}",
                    f"{c['got']:.10g}",
                    "" if c["kind"] == "abs" else f"{c['error_pct']:.6e}",
                    "" if c["kind"] == "abs" else c["threshold_pct"],
                    f"{c['error_abs']:.6e}" if c["kind"] == "abs" else "",
                    c["threshold_abs"] if c["kind"] == "abs" else "",
                    c["pass"],
                ])
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "xval_edt_two_segment"))
    ap.add_argument("--quick", action="store_true", help="small grids, no figures")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "geometry").mkdir(exist_ok=True)

    scale = 0.6 if args.quick else 1.0
    cases = [case_two_segment_L(scale), case_two_segment_V(scale), case_capsule(), case_slab(), case_sphere_pack()]

    results = []
    primary = None
    for case in cases:
        res = compare_case(case)
        res["case"] = case
        results.append(res)
        if primary is None:
            primary = res
        name = case["name"]
        write_palabos_geometry(out / "geometry" / f"{res['row']['got_geom_name']}.dat", res["got"])
        # OSS reference dump for the same case (bit-comparable yardstick)
        np.ascontiguousarray(res["ref_tags"], dtype="<i2").tofile(
            out / "geometry" / f"{res['row']['ref_geom_name']}_oss.dat"
        )

    # feature combination: padded pressure-BC slices + non-wetting seeds, i.e.
    # the full OSS entry-point path (create_nw_fluid_mask + create_geom_edist).
    feat = case_two_segment_L(scale)
    nw_seeds = np.zeros_like(feat["solid"], dtype=bool)
    nw_seeds[:, :, 2:6] = ~feat["solid"][:, :, 2:6]      # fluid cells near inlet
    feat["name"] = "two_segment_L_nw_padded"
    res = compare_case(feat, num_slices=2, nw_seeds=nw_seeds)
    res["case"] = feat
    results.append(res)
    write_palabos_geometry(out / "geometry" / f"{res['row']['got_geom_name']}.dat", res["got"])
    np.ascontiguousarray(res["ref_tags"], dtype="<i2").tofile(
        out / "geometry" / f"{res['row']['ref_geom_name']}_oss.dat"
    )

    conv = convergence_study()
    sens = sensitivity_study()

    figs = []
    if not (args.quick or args.no_figures):
        figs = make_figures(out, primary, conv, sens)

    rows = [r["row"] for r in results]
    all_pass = all(c["pass"] for r in rows for c in r["checks"])
    summary = {
        "anchor": "XVAL-LBM-EDT-TWO-SEGMENT-GEOMETRY-TAG-COMPRESSION",
        "mission": 17894,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "oss": OSS,
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "repro_cmd": "python tools/xval_edt_two_segment.py --out runs/xval_edt_two_segment",
        "threshold_pct": 15.0,
        "cases": [{k: v for k, v in r.items() if k not in ("packed_classes_data_hex",)} for r in rows],
        "convergence": conv,
        "sensitivity": sens,
        "all_checks_pass": all_pass,
        "figures": figs,
    }
    summary["artifacts"] = [
        str((out / "summary.json")),
        str((out / "table.csv")),
        str((out / "report.md")),
        str(out / "geometry"),
    ]
    table = write_table_csv(out, rows)
    summary["artifacts"].append(str(table))
    # report written after figures so the list is complete
    rp = write_report_md(out, summary)
    summary["artifacts"].append(str(rp))
    fig_public = [{"name": f["name"], "path": str(Path(f["path"]).relative_to(REPO_ROOT))
                   if Path(f["path"]).is_relative_to(REPO_ROOT) else f["path"],
                   "caption": f["caption"]} for f in figs]
    summary["figures"] = fig_public
    # rewrite report with repo-relative figure paths
    write_report_md(out, summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"[xval] out={out}")
    print(f"[xval] all_checks_pass={all_pass}")
    for r in rows:
        for c in r["checks"]:
            if c["kind"] == "abs":
                err = f"err={c['error_abs']:.3f} cells"
            else:
                err = f"err={c['error_pct']:.3e}%"
            print(f"  {'PASS' if c['pass'] else 'FAIL'} {r['case']:>24} | "
                  f"{c['quantity'][:56]:<56} ref={c['ref']:.6g} got={c['got']:.6g} {err}")
    print(f"[xval] conv slope={conv['loglog_slope']:.2f}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())