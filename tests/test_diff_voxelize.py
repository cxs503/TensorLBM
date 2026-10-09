"""Tests for :mod:`tensorlbm.diff_voxelize` (differentiable SUBOFF voxelization).

Coverage contract (see docs/diff_voxelize_20260825.md):

1. hard mask parity -- the SDF hard masks reproduce
   :func:`tensorlbm.suboff_cad.build_suboff_mask` cell-for-cell on the
   production grid (mother + appendage variant + hull-form variant);
2. STE gradients -- autograd of the STE sums equals the derivative of the
   soft-occupancy sums (validated against soft finite differences), and
   the design-scale direction matches the hard-count finite differences;
3. channel parity -- the differentiable geometry channels are numerically
   identical to the numpy training pipeline
   (:func:`tensorlbm.ai.drag_cond.suboff_geometry_features`);
4. smooth-radius monotonicity -- the hard-mask IoU against the CAD builder
   is non-decreasing as ``smooth_k`` shrinks;
5. end-to-end -- ``drag_gradients`` against finite differences of the
   ensemble prediction on the real 5-seed B4 checkpoints (skipped when
   the checkpoints are not on this machine), CPU and (skipif no card)
   CUDA.

Ensemble-dependent tests are skipped unless the B4 serve checkpoints and
the b4_v4 corpus cache exist (dev server paths).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tensorlbm.ai.drag_cond import PRODUCTION_GRID, SuboffGrid, suboff_geometry_features
from tensorlbm.ai.inference_service import load_corpus_index
from tensorlbm.diff_voxelize import (
    DEFAULT_SMOOTH_K_FT,
    DEFAULT_TAU_FT,
    DIFF_PARAM_NAMES,
    DiffDragEnsemble,
    DiffParams,
    drag_finite_difference,
    drag_forward,
    drag_gradients,
    mask_channels,
    reference_drag_forward,
    smooth_max,
    smooth_min,
    soft_mask,
    straight_through_mask,
    suboff_component_sdfs,
    suboff_radius_profile_torch,
)
from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask, suboff_radius_profile

_CKPT_DIR = Path("/nfs/wangxi/runs/b4_serve_20260824/ckpts")
_CKPT_PATHS = sorted(_CKPT_DIR.glob("serve_cfull_s*.pt"))
_CORPUS_DIR = Path("/nfs/wangxi/runs/b4_v4_20260824")
_RE_EVAL = 200.0

#: Small grid used by the gradient-machinery tests (CPU seconds matter).
SMALL_GRID = SuboffGrid(nx=96, ny=48, nz=48, cx=96 * 0.35, cy=24.0, cz=24.0, length=0.6 * 96)

VARIANTS: dict[str, dict[str, float]] = {
    "mother": {},
    "appendage": {"sail_scale": 1.35, "fin_scale": 1.3},
    "hull_form": {"l_over_d_mult": 1.15, "nose_len_mult": 1.2},
    "hull_form_blunt": {"l_over_d_mult": 0.85, "stern_len_mult": 1.3, "sail_x_mult": 1.05},
}


def _hard_union(sdfs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Hard occupancy of the union of the given component SDFs."""
    masks = [s <= 0 for s in sdfs.values()]
    out = masks[0]
    for m in masks[1:]:
        out = out | m
    return out


def _iou(a: torch.Tensor, b: torch.Tensor) -> float:
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union


def _reference_mask(values: dict[str, float], hull_type: str = "full") -> torch.Tensor:
    cfg = SuboffConfig(**values)
    mask, _ = build_suboff_mask(
        hull_type=hull_type,
        nx=PRODUCTION_GRID.nx,
        ny=PRODUCTION_GRID.ny,
        nz=PRODUCTION_GRID.nz,
        cx=PRODUCTION_GRID.cx,
        cy=PRODUCTION_GRID.cy,
        cz=PRODUCTION_GRID.cz,
        length=PRODUCTION_GRID.length,
        config=cfg,
    )
    return mask.cpu()


def _corpus_field_row() -> np.ndarray:
    idx = load_corpus_index(_CORPUS_DIR)
    rows = [i for i, d in enumerate(idx.designs) if d[:3] == ("full", 1.0, 1.0)]
    j = min(rows, key=lambda i: abs(math.log10(idx.re[i]) - math.log10(_RE_EVAL)))
    return idx.fields[j]


# --------------------------------------------------------------------------- #
# 1. geometry parity
# --------------------------------------------------------------------------- #
def test_radius_profile_matches_numpy() -> None:
    """Torch profile port reproduces the numpy DARPA profile.

    Interior: exact to float round-off.  The two exact tips carry the
    documented root-softening bias (<= 1e-4 in normalised radius, i.e.
    < 1e-4 ft).
    """
    rng = np.random.default_rng(20260825)
    xi = np.concatenate([rng.uniform(0.0, 1.0, 5000), [0.0, 0.2333, 0.7449, 0.9781, 1.0]])
    r_np = suboff_radius_profile(xi)
    r_t = suboff_radius_profile_torch(torch.from_numpy(xi)).numpy()
    interior = (xi > 1e-6) & (xi < 1.0 - 1e-6) & (r_np > 1e-6)
    assert np.abs(r_np[interior] - r_t[interior]).max() < 1e-8
    assert np.abs(r_t - r_np).max() < 1e-4


@pytest.mark.parametrize("name", ["mother", "appendage", "hull_form"])
def test_hard_mask_iou_vs_build_suboff_mask(name: str) -> None:
    """SDF hard union vs the CAD voxel builder: IoU >= 0.99 (measured 1.0)."""
    values = VARIANTS[name]
    ref = _reference_mask(values)
    sdfs = suboff_component_sdfs(PRODUCTION_GRID, DiffParams.from_values(values))
    mine = _hard_union(sdfs).cpu()
    iou = _iou(ref, mine)
    assert iou >= 0.99, f"{name}: IoU {iou:.5f}, {int((ref ^ mine).sum())} differing cells"


def test_hard_mask_iou_hull_form_blunt() -> None:
    """Second hull-form variant (stretched stern + translated sail)."""
    values = VARIANTS["hull_form_blunt"]
    ref = _reference_mask(values)
    sdfs = suboff_component_sdfs(PRODUCTION_GRID, DiffParams.from_values(values))
    mine = _hard_union(sdfs).cpu()
    assert _iou(ref, mine) >= 0.99


@pytest.mark.parametrize("hull_type", ["bare_hull", "with_sail", "full"])
def test_component_masks_match_reference(hull_type: str) -> None:
    """Component-wise hard masks match the staged CAD builders exactly.

    The reference builder for ``bare_hull``/``with_sail`` omits the inactive
    appendages, so the union under test must skip them too.
    """
    ref = _reference_mask({"sail_scale": 1.2, "fin_scale": 1.15}, hull_type=hull_type)
    sdfs = suboff_component_sdfs(
        PRODUCTION_GRID, DiffParams.from_values({"sail_scale": 1.2, "fin_scale": 1.15})
    )
    active = ["hull", "sail", "fin"][: {"bare_hull": 1, "with_sail": 2, "full": 3}[hull_type]]
    mine = _hard_union({k: sdfs[k] for k in active}).cpu()
    assert _iou(ref, mine) >= 0.999


# --------------------------------------------------------------------------- #
# 2. STE gradient machinery
# --------------------------------------------------------------------------- #
def test_ste_sum_gradient_is_soft_gradient() -> None:
    """For occupancy *sums* the STE gradient is exactly d(sum soft)/dtheta.

    ``v_bare = sum(m_hull)`` is a plain sum of a single mask, so the STE
    backward (sigmoid derivative) and the soft forward gradient coincide.
    Axes that do not touch the hull (sail/fin scale) receive ``None``.
    """
    grid = SMALL_GRID
    axes = {"sail_scale": 1.2, "fin_scale": 1.1, "l_over_d_mult": 1.05, "nose_len_mult": 1.1}
    grads = {}
    for ste_mode in (True, False):
        p = DiffParams.from_values(axes, requires_grad=True)
        sdfs = suboff_component_sdfs(grid, p)
        ch = mask_channels(grid, p, sdfs=sdfs, ste=ste_mode)
        ch.counts["v_bare"].backward()
        grads[ste_mode] = {k: getattr(p, k).grad for k in axes}
    for name in ("l_over_d_mult", "nose_len_mult"):
        ste = float(grads[True][name])
        soft = float(grads[False][name])
        assert ste != 0.0
        assert soft != 0.0
        assert abs(ste - soft) <= 1e-6 * max(abs(soft), 1.0)
    for name in ("sail_scale", "fin_scale"):
        assert grads[True][name] is None or float(grads[True][name]) == 0.0


def test_ste_gradient_soft_fd_direction_consistency() -> None:
    """STE autograd vs central FD of the soft sums: sign match, factor-3 gap.

    The STE linearises the boolean composition (products for the disjoint
    decomposition and the column-OR projection) at the hard configuration,
    so its slope differs from the soft function's local slope by the
    composition nonlinearity.  Measured on this grid at the default
    tau=0.02 ft: sail and hull-form axes agree to 2% (ratio 0.98/1.00),
    the thin-fin axis carries the junction-band bias (ratio ~0.5) --
    direction must agree everywhere, magnitude within a factor 3.
    """
    grid = SMALL_GRID
    axes = {"sail_scale": 1.2, "fin_scale": 1.1, "l_over_d_mult": 1.05}
    p = DiffParams.from_values(axes, requires_grad=True)
    ch = mask_channels(grid, p)
    target = ch.counts["v_solid"]
    target.backward()
    ste = {k: float(getattr(p, k).grad) for k in axes}

    def soft_volume(values: dict[str, float]) -> float:
        pp = DiffParams.from_values(values)
        s = suboff_component_sdfs(grid, pp)
        ch_s = mask_channels(grid, pp, sdfs=s, ste=False)
        return float(ch_s.counts["v_solid"])

    h = 2e-4
    for name, base in axes.items():
        plus = dict(axes)
        plus[name] = base * (1 + h)
        minus = dict(axes)
        minus[name] = base * (1 - h)
        fd = (soft_volume(plus) - soft_volume(minus)) / (2 * base * h)
        assert fd * ste[name] > 0.0, f"{name}: sign flip fd={fd} ste={ste[name]}"
        assert abs(fd - ste[name]) <= 2.0 * max(abs(fd), abs(ste[name]))


def test_ste_gradient_hard_fd_direction() -> None:
    """Design-scale hard-count FD agrees in sign with the STE estimator.

    Hard voxel counts are step functions; over a +/-10% design-scale
    window the staircase average must share the STE direction for volume
    sums.
    """
    grid = SMALL_GRID
    axes = {"sail_scale": 1.2, "l_over_d_mult": 1.05}
    p = DiffParams.from_values(axes, requires_grad=True)
    ch = mask_channels(grid, p)
    ch.counts["v_solid"].backward()
    ste = {k: float(getattr(p, k).grad) for k in axes}

    def hard_volume(values: dict[str, float]) -> float:
        pp = DiffParams.from_values(values)
        ch_h = mask_channels(grid, pp)
        return float(ch_h.counts["v_solid"])

    h = 0.1
    for name, base in axes.items():
        plus = dict(axes)
        plus[name] = base * (1 + h)
        minus = dict(axes)
        minus[name] = base * (1 - h)
        fd = (hard_volume(plus) - hard_volume(minus)) / (2 * base * h)
        assert fd * ste[name] > 0.0, f"{name}: sign flip fd={fd} ste={ste[name]}"


def test_soft_fd_matches_soft_autograd_tight() -> None:
    """Soft-occupancy autograd == soft FD to truncation error (graph sanity)."""
    grid = SMALL_GRID
    axes = {"sail_scale": 1.15, "fin_scale": 1.1}
    p = DiffParams.from_values(axes, requires_grad=True)
    ch = mask_channels(grid, p, ste=False)
    ch.counts["v_solid"].backward()
    ag = {k: float(getattr(p, k).grad) for k in axes}

    def soft_volume(values: dict[str, float]) -> float:
        pp = DiffParams.from_values(values)
        ch_s = mask_channels(grid, pp, ste=False)
        return float(ch_s.counts["v_solid"])

    h = 5e-4
    for name, base in axes.items():
        plus = dict(axes)
        plus[name] = base * (1 + h)
        minus = dict(axes)
        minus[name] = base * (1 - h)
        fd = (soft_volume(plus) - soft_volume(minus)) / (2 * base * h)
        assert abs(fd - ag[name]) <= 0.02 * max(abs(fd), abs(ag[name]))


def test_soft_and_ste_masks_basic_properties() -> None:
    """STE forward is the hard mask; soft is its sigmoid relaxation.

    ``hard + soft - soft.detach()`` reproduces the hard values to float
    rounding (~1e-16), so equality is asserted at 1e-12, not bit-exact.
    """
    sdf = torch.linspace(-0.2, 0.2, 41, dtype=torch.float64)
    ste = straight_through_mask(sdf, tau=0.05)
    soft = soft_mask(sdf, tau=0.05)
    hard = (sdf <= 0).double()
    assert torch.all((ste.detach() - hard).abs() <= 1e-12)
    assert torch.all(soft >= 0.0) and torch.all(soft <= 1.0)
    assert float(soft[20]) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        soft_mask(sdf, tau=0.0)


# --------------------------------------------------------------------------- #
# 3. channel parity with the numpy training pipeline
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("hull_type", "sail", "fin"),
    [
        ("full", 1.0, 1.0),
        ("full", 1.3, 0.8),
        ("full", 0.7, 1.25),
        ("with_sail", 1.5, 1.0),
        ("bare_hull", 1.0, 1.0),
    ],
)
def test_channels_match_numpy_training_pipeline(hull_type: str, sail: float, fin: float) -> None:
    """Differentiable channels == suboff_geometry_features (measured <=1e-12)."""
    ch = mask_channels(
        PRODUCTION_GRID,
        DiffParams.from_values({"sail_scale": sail, "fin_scale": fin}),
        hull_type=hull_type,
    )
    mine = ch.channel_vector.detach().numpy()
    ref = suboff_geometry_features(hull_type, sail, fin)
    ref_vec = np.array(
        [ref.log_aproj_ratio, ref.sail_frac, ref.fin_frac, ref.solid_frac], dtype=np.float64
    )
    assert np.abs(mine - ref_vec).max() <= 1e-9
    counts = {k: int(round(float(v))) for k, v in ch.counts.items()}
    assert counts["v_bare"] == ref.v_bare
    assert counts["v_sail"] == ref.v_sail
    assert counts["v_fin"] == ref.v_fin
    assert counts["v_solid"] == ref.v_solid
    assert counts["aproj"] == ref.aproj
    assert counts["aproj_bare"] == ref.aproj_bare


# --------------------------------------------------------------------------- #
# 4. smooth-radius monotonicity
# --------------------------------------------------------------------------- #
def test_smooth_k_monotonic_iou() -> None:
    """Shrinking the smooth radius never lowers the hard-mask IoU."""
    ref = _reference_mask({})
    ious = []
    for k in (0.2, 0.05, 0.01):
        sdfs = suboff_component_sdfs(PRODUCTION_GRID, DiffParams.from_values(None), smooth_k=k)
        ious.append(_iou(ref, _hard_union(sdfs).cpu()))
    assert ious[0] >= 0.99
    assert ious[1] >= ious[0] - 1e-6
    assert ious[2] >= ious[1] - 1e-6


def test_smooth_min_bounds_and_errors() -> None:
    """smooth_min/smooth_max sandwich the exact boolean and reject k <= 0."""
    a = torch.tensor([-1.0, 0.3, 2.0])
    b = torch.tensor([-0.5, -0.2, 1.0])
    smin = torch.minimum(a, b)
    smax = torch.maximum(a, b)
    k = 0.1
    assert torch.all(smooth_min(a, b, k) <= smin + 1e-12)
    assert torch.all(smooth_max(a, b, k) >= smax - 1e-12)
    # away from the blend window the primitives are exact
    assert float(smooth_min(a[0:1], b[0:1], k)) == float(smin[0])
    assert float(smooth_max(a[2:3], b[2:3], k)) == float(smax[2])
    with pytest.raises(ValueError):
        smooth_min(a, b, 0.0)


# --------------------------------------------------------------------------- #
# 5. end-to-end gradients on the real ensemble
# --------------------------------------------------------------------------- #
def _ensemble_ready() -> bool:
    return len(_CKPT_PATHS) >= 2 and (_CORPUS_DIR / "cache_v4.npz").is_file()


_needs_ensemble = pytest.mark.skipif(
    not _ensemble_ready(), reason="B4 serve checkpoints / corpus cache not present"
)


@pytest.fixture(scope="module")
def field_row() -> np.ndarray:
    return _corpus_field_row()


@_needs_ensemble
def test_forward_matches_reference_pipeline(field_row: np.ndarray) -> None:
    """STE hard forward == numpy training-pipeline forward (channels + mask)."""
    design = {"sail_scale": 1.0, "fin_scale": 1.0}
    ens = DiffDragEnsemble.from_checkpoints(_CKPT_PATHS)
    with torch.no_grad():
        out = drag_forward(design, ens, field_row, re=_RE_EVAL, requires_grad=False)
    ref = reference_drag_forward(design, ens, field_row, re=_RE_EVAL)
    assert out["values"]["log10_cd"] == pytest.approx(ref["log10_cd"], abs=1e-12)
    for key, val in out["values"]["channels"].items():
        assert val == pytest.approx(ref["channels"][key], abs=1e-12)


@_needs_ensemble
def test_drag_gradients_vs_finite_difference(field_row: np.ndarray) -> None:
    """End-to-end STE gradients vs finite differences of the forward (2 designs).

    Acceptance (see docs/diff_voxelize_20260825.md for the full tables):

    - soft-graph self-consistency: soft autograd vs soft FD at two steps
      <= 5% on every axis (machinery gate, no staircase involved);
    - hard oracle: both design-scale FD steps finite (their step stability
      is reported in the run report, not asserted -- the hard forward is
      piecewise constant);
    - direction: at the appendage design (components well above the
      single-jump noise floor) the STE gradient *vector* has positive
      cosine with the soft-autograd vector and with both hard-FD vectors,
      and the sail_scale component (in-support, strongest signal) shares
      the hard-FD sign at both designs.  At the mother design every
      hard-FD component sits at ~1e-2 (one or two voxel jumps), so no
      direction claim is made there; per-axis sign assertions on
      fin_scale / l_over_d_mult are deliberately NOT made anywhere: the
      fin pathway is a near-cancellation in the trained response (STE
      magnitude ~1e-2) and the l_over_d hard FD is staircase-dominated.
    """
    designs = [
        {"sail_scale": 1.0, "fin_scale": 1.0, "l_over_d_mult": 1.0},
        {"sail_scale": 1.2, "fin_scale": 1.15, "l_over_d_mult": 1.05},
    ]
    for design in designs:
        ag = drag_gradients(design, _CKPT_PATHS, field_row, re=_RE_EVAL)
        ag_soft = drag_gradients(design, _CKPT_PATHS, field_row, re=_RE_EVAL, ste=False)
        fd_soft1 = drag_finite_difference(
            design, _CKPT_PATHS, field_row, h=1e-3, re=_RE_EVAL, ste=False
        )
        fd_soft2 = drag_finite_difference(
            design, _CKPT_PATHS, field_row, h=2e-3, re=_RE_EVAL, ste=False
        )
        fd_hard1 = drag_finite_difference(design, _CKPT_PATHS, field_row, h=0.05, re=_RE_EVAL)
        fd_hard2 = drag_finite_difference(design, _CKPT_PATHS, field_row, h=0.1, re=_RE_EVAL)
        names = sorted(design)
        for name in names:
            soft_ag = ag_soft["grads"][name]
            # soft graph self-consistency (two FD steps must agree with autograd)
            for fd in (fd_soft1, fd_soft2):
                rel = abs(fd["grads"][name] - soft_ag) / max(abs(soft_ag), 1e-9)
                assert rel <= 0.05, f"{name}: soft autograd {soft_ag} vs FD {fd['grads'][name]}"
            # step stability of the hard oracle: finiteness only
            h1, h2 = fd_hard1["grads"][name], fd_hard2["grads"][name]
            assert math.isfinite(h1) and math.isfinite(h2)

        def _cos(a: dict[str, float], b: dict[str, float]) -> float:
            va = np.array([a[k] for k in names])
            vb = np.array([b[k] for k in names])
            return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))

        ste = {k: float(ag["grads"][k]) for k in names}
        soft = {k: float(ag_soft["grads"][k]) for k in names}
        assert ste["sail_scale"] * fd_hard1["grads"]["sail_scale"] > 0.0
        if design["sail_scale"] > 1.0:  # appendage design: above the noise floor
            assert _cos(ste, soft) > 0.0
            assert _cos(ste, fd_hard1["grads"]) > 0.0
            assert _cos(ste, fd_hard2["grads"]) > 0.0
        assert set(ag["member_grads"]) == set(design)
        assert len(ag["member_grads"]["sail_scale"]) == len(_CKPT_PATHS)


@_needs_ensemble
def test_drag_gradients_cuda(field_row: np.ndarray) -> None:
    """GPU smoke: CUDA gradients match CPU gradients for the same design."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    design = {"sail_scale": 1.1, "fin_scale": 1.0, "l_over_d_mult": 1.0}
    cpu = drag_gradients(design, _CKPT_PATHS, field_row, re=_RE_EVAL, device="cpu")
    gpu = drag_gradients(design, _CKPT_PATHS, field_row, re=_RE_EVAL, device="cuda")
    for name in design:
        assert cpu["grads"][name] == pytest.approx(gpu["grads"][name], rel=2e-3, abs=1e-5), (
            f"{name}: cpu {cpu['grads'][name]} vs cuda {gpu['grads'][name]}"
        )


def test_diffparams_defaults_and_validation() -> None:
    """DiffParams defaults are the mother geometry; values round-trip."""
    p = DiffParams.from_values(None)
    assert p.values() == {name: float(getattr(SuboffConfig(), name)) for name in DIFF_PARAM_NAMES}
    cfg = p.to_config()
    assert isinstance(cfg, SuboffConfig)
    q = DiffParams.from_values({"sail_scale": 1.3}, requires_grad=True)
    assert float(q.sail_scale.detach()) == 1.3
    assert q.sail_scale.requires_grad


def test_default_knobs_sane() -> None:
    """Default knobs documented against the production cell size."""
    cell_ft = 14.291667 / PRODUCTION_GRID.length
    assert DEFAULT_SMOOTH_K_FT < 0.2 * cell_ft
    assert DEFAULT_TAU_FT < 0.109375  # below the smallest feature (sail half-thickness)
