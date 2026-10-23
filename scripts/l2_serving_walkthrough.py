#!/usr/bin/env python3
"""L2 new-geometry serving walkthrough: an STL hull to a drag curve.

One command exercises every piece of the merged L2 stack through the real
service API, telling the external-user story end to end:

    CAD parameters -> occupancy mask -> **STL export** -> **STL re-ingest**
    (``tensorlbm.geometry_stl.stl_to_sdf``) -> leave-one-design-out
    ``FieldProvider`` pool -> ``load_bundle_pool`` frozen ensemble ->
    ``DragSurrogateService.predict(field_policy="field_borrow")`` ->
    drag curve + honest provenance (donor, retrieval distance, guard
    verdict, ensemble sigma).

The demonstration design is corpus design 106 — the slender
``l_over_d=1.30`` hull that carries all 25 held-out rows of the 406-row
corpus — served as if it were new: every row of the design leaves the
retrieval pool (the leave-one-design-out rule of the 2026-09-04 e2e LODO
campaign, ``/nfs/wangxi/runs/l2_e2e_validation_20260904``), so the donor,
the borrowed reference field and the SDF all come from OTHER designs.
Truth (the corpus C_D labels of the design's rows) is then an oracle-level
accuracy check, not an input.

The STL detour is lossless: the boxel exporter
(:func:`tensorlbm.geometry_stl.mask_to_stl`) inverts bit-exactly through
re-voxelisation, and the re-ingested SDF is bit-identical to the stored
corpus SDF of this design — the demo validates the external geometry
interface, not an approximation of it.

Usage (paths default to the 5090 production layout; every one is a flag):

    python scripts/l2_serving_walkthrough.py
    python scripts/l2_serving_walkthrough.py --device cpu --out /tmp/walk

Writes ``leg1.json`` (all numbers, including the verbatim service
``info`` block) and prints the per-Re table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tensorlbm.ai.ckpt_bundle import (  # noqa: E402
    PerMemberEnsembleBackend,
    load_bundle_pool,
)
from tensorlbm.ai.drag_cond import (  # noqa: E402
    PRODUCTION_GRID,
    condition_v3,
    geometry_channels,
    suboff_geometry_features,
)
from tensorlbm.ai.field_provider import FieldProvider  # noqa: E402
from tensorlbm.ai.inference_service import (  # noqa: E402
    DragSurrogateService,
    EnvelopeMahalanobisGuardrail,
)
from tensorlbm.geometry_stl import stl_to_mask, stl_to_sdf, write_mask_stl  # noqa: E402
from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask  # noqa: E402

#: Design-key multiplier columns (the e2e ``design_key`` schema).
MULT_KEYS = ("l_over_d_mult", "nose_len_mult", "stern_len_mult", "sail_x_mult")

#: The demonstration design: index into the sorted unique design keys of
#: the 406-row corpus.  106 = the slender l_over_d=1.30 hull, the one
#: design whose 25 held-out rows make it fully held (asserted at runtime).
DESIGN = 106


def default_paths() -> dict[str, str]:
    """Production corpus layout of the 5090 server (all overridable)."""
    runs = "/nfs/wangxi/runs"
    return {
        "fam_dir": f"{runs}/b4_fam_20260824",
        "ext_dir": f"{runs}/sdf_slender_20260828",
        "sdf2_dir": f"{runs}/b4_sdf2_20260825",
        "v3_dir": f"{runs}/b4_v3_20260824",
        "promo_dir": f"{runs}/anchor_promo_20260829",
        "bundles_dir": f"{runs}/ckpt_bundle_pm20260831",
    }


def add_path_args(parser: argparse.ArgumentParser) -> None:
    for name, value in default_paths().items():
        parser.add_argument(f"--{name}", default=value, help=f"(default: {value})")
    parser.add_argument("--out", default="/nfs/wangxi/runs/l2_walkthrough_20260906/leg1.json")
    parser.add_argument("--design", type=int, default=DESIGN)
    parser.add_argument("--arm", default="ts2", choices=("ts2", "ts4"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--uq-temperature", type=float, default=1.5)
    parser.add_argument("--max-target-rows", type=int, default=5)


# --------------------------------------------------------------------------- #
# 1. Corpus (the e2e ``load_fam`` assembly, verbatim)
# --------------------------------------------------------------------------- #
def load_corpus(p: argparse.Namespace) -> dict[str, Any]:
    """The 406-row corpus: arrays, per-row metadata, design ids."""
    fam = Path(p.fam_dir)
    ext = Path(p.ext_dir)
    z = np.load(fam / "cache_fam.npz")
    d = {k: z[k] for k in ("x", "dsi", "re", "uin", "sail", "fin", "hull", "cd")}
    ze = np.load(ext / "cache_ext56.npz")
    for k in d:
        d[k] = np.concatenate([d[k], ze[k]])
    sdf_ref = np.load(Path(p.sdf2_dir) / "sdf_fam350.npz")["sdf"]
    zs = np.load(ext / "sdf_ext2.npz")
    ext_sdf = np.stack([zs[f"d{int(i)}"] for i in d["dsi"][len(sdf_ref) :]])
    d["sdf"] = np.concatenate([sdf_ref, ext_sdf])
    d["meta"] = (
        json.load(open(Path(p.v3_dir) / "cache_meta.json"))
        + json.load(open(fam / "cache_fam_meta.json"))
        + json.load(open(ext / "meta_ext.json"))
    )
    keys = [design_key(r) for r in d["meta"]]
    uniq = sorted(set(keys))
    kid = {k: i for i, k in enumerate(uniq)}
    d["karr"] = np.array([kid[k] for k in keys])
    d["uniq"] = uniq
    return d


def design_key(row: dict[str, Any]) -> tuple:
    return (
        str(row["hull"]),
        round(float(row["sail"]), 9),
        round(float(row["fin"]), 9),
    ) + tuple(round(float(row.get(k, 1.0)), 9) for k in MULT_KEYS)


def pick_targets(
    d: dict[str, Any], design: int, promo_dir: str, max_rows: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Truth rows of the design, e2e rule: non-fit rows by Re, evenly spaced."""
    c = np.load(Path(promo_dir) / "corpus353.npz")
    held25 = np.asarray(c["held25_idx406"])
    fit_idx = set(sorted(c["idx406"][c["role"] != 1].tolist()))
    karr = d["karr"]
    assert len({int(karr[i]) for i in held25}) == 1 and int(karr[held25[0]]) == design, (
        f"design {design} does not carry all 25 held rows — pick the slender design"
    )
    rows = np.where(karr == design)[0]
    nonfit = np.array([r for r in rows if int(r) not in fit_idx])
    cand = nonfit if len(nonfit) >= max_rows else rows
    cand = cand[np.argsort(d["re"][cand], kind="stable")]
    pos = np.unique(np.linspace(0, len(cand) - 1, max_rows).round().astype(int))
    tgt = cand[pos] if len(cand) > max_rows else cand
    info = dict(
        n_rows_design=int(len(rows)),
        n_held=int(sum(1 for r in tgt if r in set(held25.tolist()))),
        rows=[int(r) for r in tgt],
        rule="non-fit rows sorted by Re, evenly spaced (e2e target rule)",
    )
    return tgt, info


# --------------------------------------------------------------------------- #
# 2. Geometry: CAD parameters -> mask -> STL -> SDF (the #280 interface)
# --------------------------------------------------------------------------- #
def build_design_mask(key: tuple) -> np.ndarray:
    """Occupancy mask of one design key at the production grid (CPU)."""
    hull, sail, fin, lod, nose, stern, sailx = key
    cfg = SuboffConfig(
        sail_scale=sail,
        fin_scale=fin,
        l_over_d_mult=lod,
        nose_len_mult=nose,
        stern_len_mult=stern,
        sail_x_mult=sailx,
    )
    g = PRODUCTION_GRID
    m, _ = build_suboff_mask(
        hull_type=hull,
        nx=g.nx,
        ny=g.ny,
        nz=g.nz,
        cx=g.cx,
        cy=g.cy,
        cz=g.cz,
        length=g.length,
        config=cfg,
        device="cpu",
    )
    return np.asarray(m)


def export_ingest_sdf(mask: np.ndarray, stl_path: Path, device: str) -> tuple[np.ndarray, Any]:
    """Mask -> STL file -> pooled SDF volume; returns (sdf, roundtrip mask)."""
    write_mask_stl(stl_path, mask)
    shape = (PRODUCTION_GRID.nz, PRODUCTION_GRID.ny, PRODUCTION_GRID.nx)
    rt_mask = stl_to_mask(str(stl_path), shape)
    sdf = stl_to_sdf(str(stl_path), shape, device=device)
    return sdf, rt_mask


# --------------------------------------------------------------------------- #
# 3. Service: LODO pool -> bundle ensemble -> provider + guard -> service
# --------------------------------------------------------------------------- #
def build_lodo_provider(
    full: FieldProvider, karr: np.ndarray, design: int
) -> tuple[FieldProvider, np.ndarray]:
    """Leave-one-design-out retrieval pool: 406 rows minus the design's rows."""
    keep = np.where(karr != design)[0]
    assert full.pool_sdfs is not None and full.pool_cond is not None
    provider = FieldProvider(
        full.pool_fields[keep],
        pool_sdfs=full.pool_sdfs[keep],
        pool_cond=full.pool_cond[keep],
    )
    return provider, keep


def fit_guard(d: dict[str, Any], keep: np.ndarray) -> EnvelopeMahalanobisGuardrail:
    """Envelope guard fit on the condition_v3 rows of the LODO pool."""
    geo_cache: dict[tuple[str, float, float], np.ndarray] = {}
    geo = np.empty((len(keep), 4))
    for j, i in enumerate(keep):
        m = d["meta"][int(i)]
        key3 = (str(m["hull"]), float(m["sail"]), float(m["fin"]))
        if key3 not in geo_cache:
            geo_cache[key3] = geometry_channels(
                suboff_geometry_features(*key3, grid=PRODUCTION_GRID)
            )
        geo[j] = geo_cache[key3]
    cond = condition_v3(d["re"][keep], d["uin"][keep], d["sail"][keep], d["fin"][keep], geo)
    return EnvelopeMahalanobisGuardrail(cond)


def build_service(
    bundles_dir: str,
    arm: str,
    device: str,
    provider: FieldProvider,
    guard: EnvelopeMahalanobisGuardrail,
    uq_temperature: float,
) -> DragSurrogateService:
    """The frozen serving stack: bundle pool -> per-member backend -> service."""
    members = load_bundle_pool(bundles_dir, arm=arm, device=device)
    backend = PerMemberEnsembleBackend.from_bundles(members, device=device)
    return DragSurrogateService(
        backend,
        guard,
        grid=PRODUCTION_GRID,
        field_provider=provider,
        uq_temperature=uq_temperature,
    )


def serve_curve(
    svc: DragSurrogateService,
    hull: str,
    sail: float,
    fin: float,
    re_grid: np.ndarray,
    u_in: float,
    sdf: np.ndarray,
):
    """One new-geometry sweep through the real service API (borrow on miss)."""
    return svc.predict(
        hull,
        sail,
        fin,
        re_grid,
        u_in=u_in,
        sdf=sdf,
        field_policy="field_borrow",
    )


# --------------------------------------------------------------------------- #
# 4. Scoring
# --------------------------------------------------------------------------- #
def score(res, truth: np.ndarray) -> dict[str, Any]:
    """MAPE-style agreement of the served curve vs the known truth."""
    ape = np.abs(res.cd / truth - 1.0) * 100.0
    return dict(
        mape=float(np.mean(ape)),
        mape_median=float(np.median(ape)),
        mape_max=float(np.max(ape)),
        rmse_log10=float(np.sqrt(np.mean((np.log10(res.cd) - np.log10(truth)) ** 2))),
        coverage_2std=float(np.mean(np.abs(res.cd - truth) <= 2.0 * res.std)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_path_args(parser)
    p = parser.parse_args(argv)
    Path(p.out).parent.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    t0 = time.perf_counter()

    d = load_corpus(p)
    tgt, tgt_info = pick_targets(d, p.design, p.promo_dir, p.max_target_rows)
    key = d["uniq"][p.design]
    uin_vals = np.unique(d["uin"][np.where(d["karr"] == p.design)[0]])
    assert uin_vals.size == 1, f"design rows must share one u_in, got {uin_vals}"
    u_in = float(uin_vals[0])
    print(f"[design {p.design}] key={key} targets={tgt.tolist()} u_in={u_in}")

    mask = build_design_mask(key)
    stl_path = Path(p.out).with_name(f"design{p.design}.stl")
    sdf, rt_mask = export_ingest_sdf(mask, stl_path, p.device)
    assert (rt_mask == mask).all(), "STL round-trip lost voxels"
    rows = np.where(d["karr"] == p.design)[0]
    bitexact = bool(np.array(sdf[None] == d["sdf"][rows]).all())
    print(f"[geometry] stl={stl_path.name} roundtrip bit-exact; sdf bitexact={bitexact}")

    # corpus view directory for FieldProvider.from_corpus (symlinks, no copies)
    view = Path(p.out).with_name("corpus_view")
    view.mkdir(parents=True, exist_ok=True)
    for src, name in (
        (Path(p.fam_dir) / "cache_fam.npz", "cache_fam.npz"),
        (Path(p.ext_dir) / "cache_ext56.npz", "cache_ext56.npz"),
        (Path(p.sdf2_dir) / "sdf_fam350.npz", "sdf_fam350.npz"),
        (Path(p.ext_dir) / "sdf_ext2.npz", "sdf_ext2.npz"),
    ):
        link = view / name
        if not link.exists():
            link.symlink_to(src)
    full = FieldProvider.from_corpus(str(view))
    provider, keep = build_lodo_provider(full, d["karr"], p.design)
    guard = fit_guard(d, keep)
    svc = build_service(p.bundles_dir, p.arm, p.device, provider, guard, p.uq_temperature)
    print(f"[service] pool={provider.pool_fields.shape[0]} rows, {svc.backend.kind}")

    res = serve_curve(svc, key[0], key[1], key[2], d["re"][tgt], u_in, sdf)
    prov = res.info["field_borrow"]
    donor_row = int(keep[prov["donor_index"]])
    print(
        f"[borrow] donor corpus row {donor_row} distance={prov['distance']:.4f} "
        f"guard_ok={prov['guard_ok']} rel_l2={prov['guard_rel_l2']:.4f}"
    )
    for j, r in enumerate(tgt):
        print(
            f"  re={res.re[j]:9.2f}  cd={res.cd[j]:8.4f}+-{res.std[j]:.4f}  "
            f"truth={d['cd'][r]:8.4f}  ape={abs(res.cd[j] / d['cd'][r] - 1) * 100:.4f}%"
        )

    result = dict(
        design=p.design,
        key=list(key),
        u_in=u_in,
        stl=str(stl_path),
        stl_bytes=stl_path.stat().st_size,
        mask_roundtrip_bitexact=True,
        sdf_bitexact_vs_stored=bitexact,
        targets=tgt_info,
        re=[float(v) for v in res.re],
        cd=[float(v) for v in res.cd],
        std=[float(v) for v in res.std],
        truth=[float(v) for v in d["cd"][tgt]],
        guard=res.guard.as_dict(),
        info=res.info,
        donor_corpus_row=donor_row,
        donor_design_key=list(d["uniq"][int(d["karr"][donor_row])]),
        score=score(res, d["cd"][tgt]),
        wall_s=time.perf_counter() - t0,
    )
    out = Path(p.out)
    out.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(f"[done] mape={result['score']['mape']:.4f}% -> {out} ({result['wall_s']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
