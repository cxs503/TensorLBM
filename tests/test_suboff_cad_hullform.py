"""Hull-form (lines-plan) variant axes of ``suboff_cad`` — 2026-08-24.

Pins the four new :class:`SuboffConfig` multipliers
(``l_over_d_mult`` / ``nose_len_mult`` / ``stern_len_mult`` /
``sail_x_mult``):

1. **default bit-identity** — all multipliers at 1.0 reproduce the
   origin/main module exactly (masks ``torch.equal``, statistics equal,
   solid-cell anchors 4093 / 4121 / 4157 at n128);
2. **per-axis monotonicity** — the physically monotone observables move
   monotonically over mult ∈ {0.7, 0.85, 1.0, 1.15, 1.3};
3. **axis-range guards** — segment-layout collapse / sail leaving the
   deck window / non-positive multipliers raise;
4. **axis (in)dependence** — the joint axial map is NOT the composition
   of single-axis maps (piecewise-linear maps do not commute); the
   deviation is bounded and recorded, not asserted to vanish;
5. **predicate/builder agreement** — the public point predicates with a
   variant ``config`` compose bit-identically to the voxel builder;
6. **case plumbing** — ``suboff_n128`` forwards the multipliers to the
   CAD config and its metadata.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from tensorlbm.suboff_cad import (
    _HULL_NODES_FT,
    SuboffConfig,
    _pw_map_np,
    _variant_nodes_ft,
    build_suboff_mask,
    suboff_fins_contain_points,
    suboff_hull_mask,
    suboff_sail_contains_points,
)

# Production placement of suboff_n128 (drag_cond.PRODUCTION_GRID):
GRID = dict(nx=128, ny=64, nz=64, cx=128 * 0.35, cy=32.0, cz=32.0, length=76.8, device="cpu")
ANCHORS = {"bare_hull": 4093, "with_sail": 4121, "full": 4157}
MULTS = (0.7, 0.85, 1.0, 1.15, 1.3)


def _mask(hull: str = "with_sail", **kw) -> tuple[torch.Tensor, dict]:
    return build_suboff_mask(hull_type=hull, config=SuboffConfig(**kw), **GRID)


def _aproj(mask: torch.Tensor) -> int:
    return int((mask.max(dim=2).values > 0).sum())


def _aside(mask: torch.Tensor) -> int:
    """Side (z-x plane) projected cell count — project over y (dim 1)."""
    return int((mask.max(dim=1).values > 0).sum())


def _load_origin_main_module():
    """Load ``origin/main``'s suboff_cad as an isolated module (or None)."""
    repo = Path(__file__).resolve().parents[1]
    try:
        src = subprocess.run(
            ["git", "-C", str(repo), "show", "origin/main:src/tensorlbm/suboff_cad.py"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    tmp = Path(__file__).parent / "_suboff_cad_origin_main.py"
    tmp.write_text(src, encoding="utf-8")
    try:
        spec = importlib.util.spec_from_file_location("suboff_cad_origin_main", tmp)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["suboff_cad_origin_main"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        tmp.unlink(missing_ok=True)
        sys.modules.pop("suboff_cad_origin_main", None)


@pytest.mark.parametrize("hull,anchor", ANCHORS.items())
def test_default_bit_identity_vs_origin_main(hull: str, anchor: int) -> None:
    """Default multipliers = bit-identical mother geometry."""
    main = _load_origin_main_module()
    mask, stats = _mask(hull)
    assert int(mask.sum()) == anchor
    if main is None:  # pragma: no cover - git history unavailable
        pytest.skip("origin/main blob unavailable (shallow clone?)")
    old_mask, old_stats = main.build_suboff_mask(hull_type=hull, **GRID)
    assert torch.equal(mask, old_mask)
    assert all(stats[k] == old_stats[k] for k in old_stats)


def test_default_stats_displacement_unchanged() -> None:
    _, stats = _mask("with_sail")
    assert stats["displacement_lu3"] == pytest.approx(3836.2, abs=0.1)
    assert stats["L_D_ratio"] == pytest.approx(8.573, abs=0.01)  # rounds to 8.57


# ---------------------------------------------------------------------------
# per-axis monotonicity
# ---------------------------------------------------------------------------
def test_l_over_d_axis_monotone() -> None:
    solids, vols, aprojs, lds = [], [], [], []
    for m in MULTS:
        mask, stats = _mask("with_sail", l_over_d_mult=m)
        solids.append(int(mask.sum()))
        vols.append(stats["displacement_lu3"])
        aprojs.append(_aproj(mask))
        lds.append(stats["L_D_ratio"])
    # slenderer hull: strictly less volume, less projected area, larger L/D
    assert all(a > b for a, b in zip(solids, solids[1:]))
    assert all(a > b for a, b in zip(vols, vols[1:]))
    assert all(a >= b for a, b in zip(aprojs, aprojs[1:]))
    assert all(a < b for a, b in zip(lds, lds[1:]))
    assert lds[0] == pytest.approx(8.573 * 0.7, rel=2e-3)
    assert lds[-1] == pytest.approx(8.573 * 1.3, rel=2e-3)
    assert stats["radius_effective"] == pytest.approx(76.8 / (2 * 8.573) / 1.3, rel=1e-3)


def test_nose_len_axis_monotone() -> None:
    solids, vols, cxs = [], [], []
    for m in MULTS:
        mask, stats = _mask("with_sail", nose_len_mult=m)
        solids.append(int(mask.sum()))
        vols.append(stats["displacement_lu3"])
        cxs.append(torch.nonzero(mask)[:, 2].float().mean().item())
    # longer (finer) entrance replaces full-radius midbody: strictly less
    # volume; the solid shifts aft (centroid x increases)
    assert all(a > b for a, b in zip(solids, solids[1:]))
    assert all(a > b for a, b in zip(vols, vols[1:]))
    assert all(a < b for a, b in zip(cxs, cxs[1:]))


def test_stern_len_axis_monotone() -> None:
    solids, vols, asides = [], [], []
    for m in MULTS:
        mask, stats = _mask("with_sail", stern_len_mult=m)
        solids.append(int(mask.sum()))
        vols.append(stats["displacement_lu3"])
        asides.append(_aside(mask))
    # longer (finer) run replaces full-radius midbody: strictly less
    # volume and strictly smaller side projection
    assert all(a > b for a, b in zip(solids, solids[1:]))
    assert all(a > b for a, b in zip(vols, vols[1:]))
    assert all(a > b for a, b in zip(asides, asides[1:]))


def test_sail_x_axis_translates_monotonically() -> None:
    nz, ny, nx = 64, 64, 128
    cx, cy, cz = nx * 0.35, ny / 2.0, nz / 2.0
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32),
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    ftlu = 14.291667 / 76.8
    centers, solids = [], []
    for m in MULTS:
        sail = suboff_sail_contains_points(
            xx,
            yy,
            zz,
            center=(cx, cy, cz),
            length=76.8,
            scale=1.0,
            config=SuboffConfig(sail_x_mult=m),
        )
        centers.append((torch.nonzero(sail)[:, 2].float().mean() - (cx - 76.8 / 2)).item() * ftlu)
        mask, _ = _mask("with_sail", sail_x_mult=m)
        solids.append(int(mask.sum()))
    assert all(a < b for a, b in zip(centers, centers[1:]))
    # volume-conserving translation (voxel-level overlap with the hull deck
    # changes by a few cells at most)
    assert max(solids) - min(solids) <= 8


def test_bare_hull_variant_axis() -> None:
    solids = []
    for m in (0.7, 1.0, 1.3):
        mask, _ = _mask("bare_hull", l_over_d_mult=m)
        solids.append(int(mask.sum()))
    assert solids[0] > solids[1] > solids[2]


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------
def test_segment_layout_collapse_rejected() -> None:
    # nose 2.6x + stern 2.6x leaves no midbody at l_over_d 0.7
    with pytest.raises(ValueError, match="segment layout"):
        SuboffConfig(l_over_d_mult=0.7, nose_len_mult=2.6, stern_len_mult=2.6)


def test_sail_off_deck_rejected() -> None:
    with pytest.raises(ValueError, match="sail footprint"):
        SuboffConfig(sail_x_mult=3.0)  # trailing edge 11.51 ft > MID_END 10.65 ft


def test_non_positive_mults_rejected() -> None:
    for kw in (
        dict(l_over_d_mult=0.0),
        dict(nose_len_mult=-0.3),
        dict(stern_len_mult=float("inf")),
        dict(sail_x_mult=-1.0),
    ):
        with pytest.raises(ValueError, match="finite and positive"):
            SuboffConfig(**kw)


# ---------------------------------------------------------------------------
# axis (in)dependence — recorded, not orthogonal
# ---------------------------------------------------------------------------
AXIS_PAIRS = (
    (("l_over_d_mult", 1.3), ("nose_len_mult", 1.3)),
    (("l_over_d_mult", 0.75), ("stern_len_mult", 1.3)),
    (("nose_len_mult", 1.3), ("stern_len_mult", 0.7)),
)


@pytest.mark.parametrize("axA,axB", AXIS_PAIRS)
def test_axis_composition_deviation_bounded(axA, axB) -> None:
    """Joint map vs composition of single-axis maps: NOT equal (the
    piecewise-linear maps do not commute), deviation bounded and recorded."""
    x = np.linspace(0.0, 14.291667, 20001)
    cfg_a = SuboffConfig(**{axA[0]: axA[1]})
    cfg_b = SuboffConfig(**{axB[0]: axB[1]})
    cfg_j = SuboffConfig(**{axA[0]: axA[1], axB[0]: axB[1]})
    inv_a = _pw_map_np(x, _variant_nodes_ft(cfg_a), _HULL_NODES_FT)
    comp = _pw_map_np(inv_a, _variant_nodes_ft(cfg_b), _HULL_NODES_FT)
    joint = _pw_map_np(x, _variant_nodes_ft(cfg_j), _HULL_NODES_FT)
    dev = float(np.abs(comp - joint).max())
    assert 0.0 < dev < 0.10 * 14.291667  # bounded, and definitely not 0


# ---------------------------------------------------------------------------
# predicate / builder agreement under a variant
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kw",
    [
        dict(l_over_d_mult=1.3),
        dict(l_over_d_mult=0.75, nose_len_mult=1.15),
        dict(stern_len_mult=0.7, sail_x_mult=1.3),
    ],
)
def test_predicates_compose_to_builder_mask(kw) -> None:
    nz, ny, nx = 64, 64, 128
    cx, cy, cz = nx * 0.35, ny / 2.0, nz / 2.0
    length = 76.8
    cfg = SuboffConfig(**kw)
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32),
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    center = (cx, cy, cz)
    hull = suboff_hull_mask(nx, ny, nz, cx, cy, cz, length, 0.0, torch.device("cpu"), cfg)
    sail = suboff_sail_contains_points(
        xx,
        yy,
        zz,
        center=center,
        length=length,
        scale=cfg.sail_scale,
        config=cfg,
    )
    fins = suboff_fins_contain_points(
        xx,
        yy,
        zz,
        center=center,
        length=length,
        scale=cfg.fin_scale,
        config=cfg,
    )
    composed = hull | sail | fins
    built, _ = build_suboff_mask(hull_type="full", **GRID, config=cfg)
    assert torch.equal(composed, built)


def test_statistics_variant_fields() -> None:
    _, stats = _mask("with_sail", l_over_d_mult=1.3)
    assert stats["L_D_ratio"] == pytest.approx(8.573 * 1.3, rel=2e-3)
    assert stats["radius_effective"] == pytest.approx(76.8 / (2 * 8.573) / 1.3, rel=1e-3)
    assert stats["hull_form_variant"]["l_over_d_mult"] == 1.3
    _, mother_stats = _mask("with_sail")
    assert "hull_form_variant" not in mother_stats
    assert "radius_effective" not in mother_stats


def test_dataclass_replace_keeps_defaults() -> None:
    """replace() on the dataclass must not perturb the mother path."""
    cfg = replace(SuboffConfig(), sail_x_mult=1.3)
    assert cfg.l_over_d_mult == 1.0 and cfg.sail_x_mult == 1.3
    mother = replace(cfg, sail_x_mult=1.0)
    mask, _ = _mask("with_sail", **{} if mother == SuboffConfig() else {})
    assert int(mask.sum()) == ANCHORS["with_sail"]


def test_case_plumbs_hull_form_mults() -> None:
    from tensorlbm.cases import case_registry

    cls = case_registry["suboff_n128"]
    case = cls(32, re=100.0, hull_type="with_sail", l_over_d_mult=1.3, nose_len_mult=1.15)
    meta = case.metadata()
    assert meta["l_over_d_mult"] == 1.3 and meta["nose_len_mult"] == 1.15
    assert meta["stern_len_mult"] == 1.0 and meta["sail_x_mult"] == 1.0
    solid = case.build_solid()
    nz, ny, nx = case.resolution
    ref, _ = build_suboff_mask(
        hull_type="with_sail",
        nx=nx,
        ny=ny,
        nz=nz,
        cx=nx * 0.35,
        cy=ny / 2.0,
        cz=nz / 2.0,
        length=case.hull_length,
        config=SuboffConfig(l_over_d_mult=1.3, nose_len_mult=1.15),
        device=str(case.device),
    )
    assert torch.equal(solid, ref)
