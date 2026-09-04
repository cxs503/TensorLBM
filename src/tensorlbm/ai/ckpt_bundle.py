"""Sidecar-bundled checkpoints for the SDF two-stage drag-pool members.

The W16-B production pool (``/nfs/wangxi/runs/anchor_promo_20260829/runs/``)
stores each ensemble member as TWO bare ``state_dict`` files with no
sidecar, which forced every consumer to re-derive out-of-band knowledge.
This module closes the three serving gaps measured in the 2026-08-30
bit-exact sanity run (``/nfs/wangxi/runs/sdf_serve_sanity_20260830``):

**GAP 1 — no arch/norm sidecar.**  ``stage1_s{seed}.pt`` is a bare
``SupervisedSDFEncoder.state_dict()`` (trunk + linear probe) and
``stage2_{arm}_s{seed}.pt`` is a bare ``CondFNODrag.state_dict()``.
:func:`infer_member_arch` recovers the full architecture from the bare
tensor *shapes* (verified against the pool layout), and
:func:`save_member_bundle` writes a single self-describing file.

**GAP 2 — the ``.fno`` remap contract** (documented here as the format
contract, previously only encoded in the sanity driver): a bare stage-2
``CondFNODrag`` state_dict does NOT load into ``TwoStageCondFNODrag``
directly.  The trainer saved the body alone, so serving must land it on
the ``.fno`` submodule of a two-stage model built around the SAME-seed
stage-1 trunk::

    sup  = SupervisedSDFEncoder(SDFEncoderV2(latent_dim=32, base=12), ...)
    sup.load_state_dict(sd1)                  # strict (sd1 carries probe.*)
    full = TwoStageCondFNODrag(sup.encoder, param_dim=..., latent_dim=...,
                               aux_dim=..., **body_kwargs).freeze_encoder()
    full.fno.load_state_dict(sd2)             # strict — THE remap

The stage-1 sd loads strict into the ``SupervisedSDFEncoder`` *wrapper*
(it owns the ``probe.*`` head, which the two-stage model does not), and
``full.encoder`` must be the same module object as ``sup.encoder`` so the
served latent is the trained one.  :func:`load_two_stage_member`
implements exactly this pattern.

**GAP 3 — per-member (encoder, body) pairs.**  The base
:class:`~tensorlbm.ai.inference_service.ModelEnsembleBackend` applies ONE
cond matrix to every member, but each pool member needs its own seed's
latent.  :class:`PerMemberEnsembleBackend` serves an explicit list of
(encoder, body) pairs through the same normalise / forward / aggregate
arithmetic as the base backend.

Bundle format (``format`` = ``"tensorlbm.cond-drag-member"``, version 1)
-----------------------------------------------------------------------
``torch.save`` of a plain dict::

    {"format": "tensorlbm.cond-drag-member",
     "version": 1,
     "arch": {"encoder": {"latent_dim", "base", "in_ch", "target_dim",
                          "probe_hidden"},
              "body": {"param_dim", "latent_dim", "aux_dim", "in_ch",
                       "width", "n_layers", "modes": [my, mx],
                       "mlp_hidden", "film_hidden"}},
     "norm": {"ch_mean", "ch_std", "p_mean", "p_std", "y_mean", "y_std"},
     "state_dicts": {"encoder": SupervisedSDFEncoder.state_dict(),
                     "body": bare CondFNODrag.state_dict()},
     "meta": {...}}

``state_dicts.body`` stays in the BARE body layout (GAP 2 remap applies on
load), so the bundle is a strict superset of the pool files.  ``norm``
follows the training convention: ``ch_*`` over the field channels,
``p_mean``/``p_std`` over the **param columns only**, ``y_*`` scalars.
The latent columns are served RAW (tanh-bounded, never z-scored) — which
is exactly equivalent to padding the cond-vector norms with zeros/ones on
the latent columns when serving the bare body through the base backend
(the verified adapter of the sanity run, rel 0.0).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from warnings import warn

import numpy as np
import torch
from torch import nn

from .geom_encoder import SDFEncoderV2
from .inference_service import ModelEnsembleBackend
from .sdf_two_stage import SupervisedSDFEncoder, TwoStageCondFNODrag

__all__ = [
    "BUNDLE_FORMAT",
    "BUNDLE_VERSION",
    "BundleDiscoveryWarning",
    "LoadedMember",
    "PerMemberEnsembleBackend",
    "discover_bundles",
    "infer_member_arch",
    "load_bundle_pool",
    "load_member_bundle",
    "load_two_stage_member",
    "save_member_bundle",
]

#: ``format`` marker of the member-bundle file format.
BUNDLE_FORMAT = "tensorlbm.cond-drag-member"

#: Current member-bundle file-format version.
BUNDLE_VERSION = 1

#: Norm keys a member sidecar must carry (same six as the single-model
#: ``CondDragCheckpoint`` format of :mod:`tensorlbm.ai.inference_service`).
_MEMBER_NORM_KEYS = ("ch_mean", "ch_std", "p_mean", "p_std", "y_mean", "y_std")

#: Serving arm of a bundle, keyed by the arch key the detection reads:
#: ``arch["body"]["param_dim"]`` (verified against the production pool
#: 2026-09-04: ts2 bodies carry 2, ts4 carry 4).  Equivalently cond_dim
#: 34 vs 36 (param columns + the 32-column raw latent) or the body param
#: counts 4,240,073 vs 4,240,713 — param_dim is the first-class bundle
#: field of the three, so that is the key used.  Detection deliberately
#: does NOT trust the optional ``meta["arm"]`` label.
_ARM_BY_PARAM_DIM = {2: "ts2", 4: "ts4"}


class BundleDiscoveryWarning(UserWarning):
    """A candidate ``.pt`` file was skipped while scanning for bundles.

    Emitted by :func:`discover_bundles` for every non-bundle candidate
    (wrong payload, or a file ``torch.load`` cannot read at all) — a stray
    ``junk.pt`` sitting next to a production pool must degrade to a
    warning, never take serving down with an exception.
    """


def _pos_int(cfg: Mapping[str, Any], key: str, block: str) -> int:
    raw = cfg.get(key)
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < 1:
        raise ValueError(f"arch.{block}.{key} must be a positive int, got {raw!r}")
    return int(raw)


def _canonical_arch(arch: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate/normalise an arch block into the canonical bundle layout."""
    if not isinstance(arch, Mapping) or "encoder" not in arch or "body" not in arch:
        raise ValueError("arch must carry 'encoder' and 'body' blocks (see module docstring)")
    enc, body = arch["encoder"], arch["body"]
    if not isinstance(enc, Mapping) or not isinstance(body, Mapping):
        raise ValueError("arch.encoder / arch.body must be mappings")
    modes = body.get("modes")
    if not isinstance(modes, (list, tuple)) or len(modes) != 2:
        raise ValueError(f"arch.body.modes must be a (my, mx) pair, got {modes!r}")
    probe_hidden = enc.get("probe_hidden", 0)
    if not isinstance(probe_hidden, int) or isinstance(probe_hidden, bool) or probe_hidden < 0:
        raise ValueError(f"arch.encoder.probe_hidden must be an int >= 0, got {probe_hidden!r}")
    aux_dim = body.get("aux_dim", 0)
    if not isinstance(aux_dim, int) or isinstance(aux_dim, bool) or aux_dim < 0:
        raise ValueError(f"arch.body.aux_dim must be an int >= 0, got {aux_dim!r}")
    param_dim = body.get("param_dim", 0)
    if not isinstance(param_dim, int) or isinstance(param_dim, bool) or param_dim < 0:
        raise ValueError(f"arch.body.param_dim must be an int >= 0, got {param_dim!r}")
    enc_arch = {
        "latent_dim": _pos_int(enc, "latent_dim", "encoder"),
        "base": _pos_int(enc, "base", "encoder"),
        "in_ch": _pos_int(enc, "in_ch", "encoder"),
        "target_dim": _pos_int(enc, "target_dim", "encoder"),
        "probe_hidden": int(probe_hidden),
    }
    body_arch = {
        "param_dim": int(param_dim),
        "latent_dim": _pos_int(body, "latent_dim", "body"),
        "aux_dim": int(aux_dim),
        "in_ch": _pos_int(body, "in_ch", "body"),
        "width": _pos_int(body, "width", "body"),
        "n_layers": _pos_int(body, "n_layers", "body"),
        "modes": [int(modes[0]), int(modes[1])],
        "mlp_hidden": _pos_int(body, "mlp_hidden", "body"),
        "film_hidden": _pos_int(body, "film_hidden", "body"),
    }
    if enc_arch["latent_dim"] != body_arch["latent_dim"]:
        raise ValueError(
            f"arch encoder latent_dim {enc_arch['latent_dim']} != body latent_dim "
            f"{body_arch['latent_dim']}"
        )
    return {"encoder": enc_arch, "body": body_arch}


def infer_member_arch(
    encoder_state_dict: Mapping[str, torch.Tensor],
    body_state_dict: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Recover the canonical arch block from BARE pool state-dict shapes.

    Closes GAP 1 for the existing pool without retraining anything: every
    constructor argument of ``SDFEncoderV2`` / ``SupervisedSDFEncoder`` /
    ``TwoStageCondFNODrag`` is readable off the saved tensors (verified
    against the production pool: ``stage1_s0`` / ``stage2_ts2_s0`` shapes
    reproduce ``latent_dim=32, base=12, target_dim=12`` and
    ``width=32, n_layers=4, modes=(16, 32), mlp_hidden=128,
    film_hidden=64, aux_dim=8, param_dim=2``).

    Body parameter counts are arm-dependent, so do not read the ts2 figure
    as universal: the ts2 body (``cond_dim`` 34 = 2 params + 32 latent)
    carries 4,240,073 parameters while the ts4 body (``cond_dim`` 36 = 4
    params + 32 latent) carries 4,240,713.
    """
    enc_sd = dict(encoder_state_dict)
    body_sd = dict(body_state_dict)

    def shape(key: str, sd: dict[str, torch.Tensor]) -> tuple[int, ...]:
        t = sd.get(key)
        if not isinstance(t, torch.Tensor) or t.ndim < 1:
            raise ValueError(f"state_dict key {key!r} is missing or not a shaped tensor")
        return tuple(int(s) for s in t.shape)

    stem = shape("encoder.stem.0.weight", enc_sd)  # (base, in_ch, 3, 3, 3)
    enc_head = shape("encoder.head.weight", enc_sd)  # (latent_dim, 20 * base)
    base, enc_in, latent = stem[0], stem[1], enc_head[0]
    if enc_head[1] != 20 * base:
        raise ValueError(
            f"encoder.head.weight in-features {enc_head[1]} != 20*base ({base}); "
            "the stage-1 file is not an SDFEncoderV2 trunk state_dict"
        )
    probe0 = shape("probe.0.weight", enc_sd)  # (target, latent) or (hidden, latent)
    if "probe.2.weight" in enc_sd:
        probe2 = shape("probe.2.weight", enc_sd)  # (target, hidden)
        target_dim, probe_hidden = probe2[0], probe0[0]
        if probe2[1] != probe_hidden or probe0[1] != latent:
            raise ValueError("probe head shapes are inconsistent with the trunk latent")
    else:
        target_dim, probe_hidden = probe0[0], 0
        if probe0[1] != latent:
            raise ValueError(f"probe.0.weight in-features {probe0[1]} != trunk latent_dim {latent}")

    lift = shape("lift.weight", body_sd)  # (width, in_ch, 1, 1)
    width, body_in = lift[0], lift[1]
    layer_idx = sorted(
        int(k.split(".")[1]) for k in body_sd if k.startswith("spectral.") and k.endswith(".weight")
    )
    if not layer_idx or layer_idx != list(range(len(layer_idx))):
        raise ValueError("stage-2 file has no contiguous spectral.{i}.weight stack")
    spec0 = shape("spectral.0.weight", body_sd)  # (width, width, my, mx, 2)
    if spec0[0] != width or spec0[1] != width:
        raise ValueError(f"spectral width {spec0[:2]} != lift width {width}")
    cond_embed = shape("cond_embed.0.weight", body_sd)  # (film_hidden, cond_dim)
    film_hidden, cond_dim = cond_embed[0], cond_embed[1]
    head0 = shape("head.0.weight", body_sd)  # (mlp_hidden, width + cond_dim)
    if head0[1] != width + cond_dim:
        raise ValueError(
            f"head.0.weight in-features {head0[1]} != width+cond_dim ({width}+{cond_dim})"
        )
    aux_dim = shape("aux_head.2.weight", body_sd)[0] if "aux_head.2.weight" in body_sd else 0
    param_dim = cond_dim - latent
    if param_dim < 0:
        raise ValueError(
            f"body cond_dim {cond_dim} < trunk latent_dim {latent}: not a two-stage pair"
        )
    return _canonical_arch(
        {
            "encoder": {
                "latent_dim": latent,
                "base": base,
                "in_ch": enc_in,
                "target_dim": target_dim,
                "probe_hidden": probe_hidden,
            },
            "body": {
                "param_dim": param_dim,
                "latent_dim": latent,
                "aux_dim": aux_dim,
                "in_ch": body_in,
                "width": width,
                "n_layers": len(layer_idx),
                "modes": [spec0[2], spec0[3]],
                "mlp_hidden": head0[0],
                "film_hidden": film_hidden,
            },
        }
    )


def _build_member_pair(arch: Mapping[str, Any]) -> tuple[SupervisedSDFEncoder, TwoStageCondFNODrag]:
    """Construct the (stage-1 wrapper, two-stage model) pair from an arch block.

    The two-stage model is built around ``sup.encoder`` — the same module
    object — exactly as the verified pool loader does.
    """
    enc_arch, body_arch = arch["encoder"], arch["body"]
    trunk = SDFEncoderV2(
        latent_dim=int(enc_arch["latent_dim"]),
        base=int(enc_arch["base"]),
        in_ch=int(enc_arch["in_ch"]),
    )
    sup = SupervisedSDFEncoder(
        trunk, target_dim=int(enc_arch["target_dim"]), hidden=int(enc_arch["probe_hidden"])
    )
    full = TwoStageCondFNODrag(
        sup.encoder,
        param_dim=int(body_arch["param_dim"]),
        latent_dim=int(body_arch["latent_dim"]),
        aux_dim=int(body_arch["aux_dim"]),
        in_ch=int(body_arch["in_ch"]),
        width=int(body_arch["width"]),
        n_layers=int(body_arch["n_layers"]),
        modes=(int(body_arch["modes"][0]), int(body_arch["modes"][1])),
        mlp_hidden=int(body_arch["mlp_hidden"]),
        film_hidden=int(body_arch["film_hidden"]),
    )
    return sup, full


def _load_member_weights(
    sup: SupervisedSDFEncoder,
    full: TwoStageCondFNODrag,
    encoder_state_dict: Mapping[str, torch.Tensor],
    body_state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Strict loads of the verified pattern — GAP 2 contract in one place.

    ``encoder_state_dict`` lands on the ``SupervisedSDFEncoder`` wrapper
    (it owns the ``probe.*`` head); ``body_state_dict`` is a BARE
    ``CondFNODrag`` state_dict and lands on ``full.fno``.  Both loads are
    strict: any key mismatch is an error, never a silent partial load.
    Both modules leave inference-ready: eval mode with every parameter
    ``requires_grad=False``.
    """
    sup.load_state_dict(encoder_state_dict, strict=True)
    full.fno.load_state_dict(body_state_dict, strict=True)
    # inference-ready: eval + EVERY parameter frozen, not just the trunk
    # (``freeze_encoder`` covers the shared trunk; this also freezes the body
    # and the stage-1 probe head).  Value-neutral for serving; training use
    # must re-enable grads explicitly.
    sup.requires_grad_(False)
    full.requires_grad_(False)
    full.freeze_encoder()
    sup.eval()
    full.eval()


def load_two_stage_member(
    stage1_path: str | Path,
    stage2_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    arch_config: Mapping[str, Any] | None = None,
) -> TwoStageCondFNODrag:
    """Load one legacy bare two-file pool member (the VERIFIED pattern).

    Contract (GAP 2): ``stage1_path`` holds a bare
    ``SupervisedSDFEncoder.state_dict()`` and ``stage2_path`` a BARE
    ``CondFNODrag.state_dict()``.  The rebuilt model is a
    ``TwoStageCondFNODrag`` whose ``.encoder`` is the loaded stage-1 trunk
    (same module object as the wrapper's) and whose ``.fno`` carries the
    stage-2 body weights — i.e. the bare body keys are remapped into the
    ``.fno`` submodule, NOT loaded at the top level.  Both loads are
    strict.  Loaders hand out inference-ready modules: the model is
    returned in eval mode with EVERY parameter ``requires_grad=False``
    (trunk, body and any probe); training use must re-enable grads
    explicitly.

    ``arch_config`` overrides the architecture inference of
    :func:`infer_member_arch` (defaults to inferring it from the bare
    tensor shapes).  Returns the rebuilt model in eval mode on ``device``.
    """
    sd1: Any = torch.load(Path(stage1_path), map_location="cpu", weights_only=True)
    sd2: Any = torch.load(Path(stage2_path), map_location="cpu", weights_only=True)
    arch = _canonical_arch(arch_config) if arch_config is not None else infer_member_arch(sd1, sd2)
    sup, full = _build_member_pair(arch)
    _load_member_weights(sup, full, sd1, sd2)
    dev = torch.device(device)
    sup.to(dev)
    full.to(dev)
    return full


def save_member_bundle(
    path: str | Path,
    encoder_state_dict: Mapping[str, torch.Tensor],
    body_state_dict: Mapping[str, torch.Tensor],
    *,
    arch_config: Mapping[str, Any] | None = None,
    norm_stats: Mapping[str, Any],
    meta: Mapping[str, Any] | None = None,
) -> str:
    """Write one self-describing member file (bundle format v1, GAP 1).

    ``arch_config`` defaults to ``None``, in which case the arch block is
    INFERRED from the bare state-dict shapes via
    :func:`infer_member_arch` (the pool layout carries exactly the shapes
    it reads); pass an explicit block to pin or override it.

    ``encoder_state_dict`` / ``body_state_dict`` are the raw pool-layout
    state dicts (``SupervisedSDFEncoder`` / bare ``CondFNODrag``); the
    body stays in the BARE layout, so the GAP 2 remap applies on load.
    ``norm_stats`` must carry the six fit-stat keys (``ch_mean``/``ch_std``
    over field channels, ``p_mean``/``p_std`` over the **param columns
    only**, ``y_mean``/``y_std`` scalars — latent columns are served raw).
    """
    arch = (
        _canonical_arch(arch_config)
        if arch_config is not None
        else infer_member_arch(encoder_state_dict, body_state_dict)
    )
    missing = [k for k in _MEMBER_NORM_KEYS if k not in norm_stats]
    if missing:
        raise ValueError(f"norm_stats missing keys: {missing}")
    payload: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "arch": arch,
        "norm": {k: np.asarray(v) for k, v in norm_stats.items()},
        "state_dicts": {
            "encoder": {k: v.detach().cpu() for k, v in encoder_state_dict.items()},
            "body": {k: v.detach().cpu() for k, v in body_state_dict.items()},
        },
        "meta": dict(meta) if meta is not None else {},
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, p)
    return str(p.resolve())


@dataclass(frozen=True)
class LoadedMember:
    """A rebuilt two-stage member plus the sidecar it was rebuilt with.

    ``model`` is the serving object and ``stage1`` its stage-1 wrapper;
    loaders hand out inference-ready modules — eval mode with EVERY
    parameter (trunk, body, probe) ``requires_grad=False``, so training
    use must re-enable grads explicitly (``model.encoder is
    stage1.encoder``, both on the requested device); ``norm`` the six
    fit-stat arrays (empty for a bare pool pair — the pool carries no
    sidecar, the caller rebuilds or supplies the fit stats); ``source``
    is ``"bundle"`` or ``"bare-pair"``.
    """

    model: TwoStageCondFNODrag
    stage1: SupervisedSDFEncoder
    arch: dict[str, Any]
    norm: dict[str, np.ndarray]
    meta: dict[str, Any]
    source: str

    @property
    def encoder(self) -> nn.Module:
        """The trunk module both ``model`` and ``stage1`` share."""
        return self.model.encoder

    @property
    def param_dim(self) -> int:
        return int(self.model.param_dim)

    @property
    def latent_dim(self) -> int:
        return int(self.model.latent_dim)


def load_member_bundle(
    path: str | Path,
    *,
    stage1_path: str | Path | None = None,
    norm_stats: Mapping[str, Any] | None = None,
    device: str | torch.device = "cpu",
    weights_only: bool = False,
) -> LoadedMember:
    """Load a member from a bundle file OR a legacy bare pool file (sniffed).

    Loaders hand out inference-ready modules: ``model`` and ``stage1`` are
    in eval mode with EVERY parameter ``requires_grad=False``; training
    use must re-enable grads explicitly.

    ``weights_only`` is passed through to the internal ``torch.load``.  It
    defaults to ``False`` because bundle v1 EMBEDS the norm sidecar as
    numpy arrays, which ``torch.load(weights_only=True)`` refuses to
    unpickle — ``True`` therefore only works for payloads whose norms were
    written as pure tensors.  Callers who control their whole bundle
    pipeline (write and read) may enforce ``True`` for the stronger load
    guarantee.

    The file at ``path`` is sniffed by its dict keys:

    * bundle (``format`` marker / ``state_dicts`` block) — the member is
      rebuilt from the embedded arch and norm (``norm_stats`` optionally
      overrides the embedded stats, e.g. a serving-time refresh);
    * legacy bare stage-2 body sd (``lift.*`` / ``cond_embed.*`` keys) —
      ``stage1_path`` must point at the matching bare stage-1 sd; the arch
      is inferred from the shapes and ``norm`` comes from ``norm_stats``
      (the pool has no sidecar); ``norm_stats=None`` yields an empty norm;
    * anything else — a ``ValueError`` names the mismatch (a single-model
      ``CondDragCheckpoint`` file points at
      :func:`tensorlbm.ai.inference_service.load_checkpoint`, a bare
      stage-1 sd asks to be passed as ``stage1_path``).
    """
    payload: Any = torch.load(Path(path), map_location="cpu", weights_only=weights_only)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(
            f"{path} is not a member bundle or legacy bare state_dict (got {type(payload).__name__})"
        )
    fmt = payload.get("format")
    if (isinstance(fmt, str) and fmt == BUNDLE_FORMAT) or "state_dicts" in payload:
        sds = payload.get("state_dicts")
        if not isinstance(sds, dict) or "encoder" not in sds or "body" not in sds:
            raise ValueError(f"{path}: bundle state_dicts block must carry 'encoder' and 'body'")
        arch = _canonical_arch(payload.get("arch"))
        sup, full = _build_member_pair(arch)
        _load_member_weights(sup, full, sds["encoder"], sds["body"])
        src_norm = payload.get("norm", {}) if norm_stats is None else norm_stats
        norm = {k: np.asarray(v) for k, v in src_norm.items()}
        meta = dict(payload.get("meta", {}))
        source = "bundle"
    elif "arch" in payload and "state_dict" in payload:
        raise ValueError(
            f"{path} is a single-model CondDragCheckpoint file; load it with "
            "tensorlbm.ai.inference_service.load_checkpoint (it has no encoder half)"
        )
    elif "lift.weight" in payload and "cond_embed.0.weight" in payload:
        if stage1_path is None:
            raise ValueError(
                f"{path} is a legacy bare stage-2 body state_dict; pass the matching "
                "stage-1 file as stage1_path= (or use load_two_stage_member)"
            )
        sd1: Any = torch.load(Path(stage1_path), map_location="cpu", weights_only=True)
        arch = infer_member_arch(sd1, payload)
        sup, full = _build_member_pair(arch)
        _load_member_weights(sup, full, sd1, payload)
        norm = {} if norm_stats is None else {k: np.asarray(v) for k, v in norm_stats.items()}
        meta = {}
        source = "bare-pair"
    elif "encoder.head.weight" in payload and "probe.0.weight" in payload:
        raise ValueError(
            f"{path} is a legacy bare stage-1 encoder state_dict; pass the stage-2 body "
            "file as the main path and this file as stage1_path="
        )
    else:
        raise ValueError(
            f"{path} does not look like a member bundle or a legacy bare stage file "
            f"(first keys: {list(payload)[:6]})"
        )
    dev = torch.device(device)
    sup.to(dev).eval()
    full.to(dev).eval()
    return LoadedMember(model=full, stage1=sup, arch=arch, norm=norm, meta=meta, source=source)


def _is_bundle_payload(payload: Any) -> bool:
    """The same sniff :func:`load_member_bundle` uses to spot a bundle."""
    fmt = payload.get("format") if isinstance(payload, dict) else None
    return isinstance(payload, dict) and (
        (isinstance(fmt, str) and fmt == BUNDLE_FORMAT) or "state_dicts" in payload
    )


def _candidate_files(path: str | Path | Sequence[str | Path]) -> list[Path]:
    """Resolve the scan input to an explicit, sorted list of files.

    A directory is scanned for ``*.pt`` (non-.pt entries are ignored);
    an explicit file — or an explicit list of files — is taken as given,
    so a caller naming a stray ``.txt`` gets it inspected (and warned
    about) rather than silently dropped.  Output is sorted by file name
    so downstream order never depends on filesystem iteration order.
    """
    if isinstance(path, (str, Path)):
        p = Path(path)
        if p.is_dir():
            return sorted((f for f in p.glob("*.pt") if f.is_file()), key=lambda f: f.name)
        if p.is_file():
            return [p]
        raise FileNotFoundError(f"bundle scan path {p} is neither a directory nor a file")
    files = [Path(x) for x in path]
    for f in files:
        if not f.is_file():
            raise FileNotFoundError(f"bundle scan path {f} is not a file")
    return sorted(files, key=lambda f: f.name)


def _bundle_descriptor(file: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """The light per-bundle descriptor :func:`discover_bundles` returns."""
    arch = payload.get("arch")
    body = arch.get("body") if isinstance(arch, Mapping) else None
    raw_pd = body.get("param_dim") if isinstance(body, Mapping) else None
    param_dim = raw_pd if isinstance(raw_pd, int) and not isinstance(raw_pd, bool) else None
    meta = payload.get("meta")
    member = meta.get("member", file.stem) if isinstance(meta, Mapping) else file.stem
    return {
        "file": file.name,
        "path": str(file.resolve()),
        "member": str(member),
        "arm": _ARM_BY_PARAM_DIM.get(param_dim) if param_dim is not None else None,
        "param_dim": param_dim,
    }


def discover_bundles(path: str | Path | Sequence[str | Path]) -> list[dict[str, Any]]:
    """Scan a directory (or an explicit list of files) for member bundles.

    Serving-side companion of the bundle format (POSTMERGE_RUNBOOK step 3,
    ``/nfs/wangxi/runs/ckpt_bundle_rehearsal_20260831/POSTMERGE_RUNBOOK.md``):
    the operational question when pointing a loader at a pool directory is
    "which members live here, and what arm is each" — answered without
    building any module.  Returns one light descriptor per ``*.pt`` that
    loads as a bundle dict (the same format-marker sniff as
    :func:`load_member_bundle`), sorted by file name:

    ``{"file", "path", "member", "arm", "param_dim"}``

    * ``member`` — ``meta["member"]`` when the bundle carries it, else the
      file stem (so ``ts2_s3.pt`` still reads ``ts2_s3``);
    * ``arm`` — read from the arch block, exactly the key
      ``arch["body"]["param_dim"]``: ``2`` -> ``"ts2"`` (cond_dim 34 = 2
      param columns + the 32-column raw latent), ``4`` -> ``"ts4"``
      (cond_dim 36); any other value yields ``None``, which never matches
      an arm filter in :func:`load_bundle_pool`.

    Every other ``.pt`` candidate — wrong payload shape, or a file
    ``torch.load`` cannot read at all — is skipped with a
    :class:`BundleDiscoveryWarning` carrying the file name and the reason,
    never an exception.
    """
    descriptors: list[dict[str, Any]] = []
    for file in _candidate_files(path):
        try:
            payload: Any = torch.load(file, map_location="cpu", weights_only=False)
        except Exception as exc:  # a junk file must skip, not kill the pool build
            warn(
                f"{file.name}: skipped (torch.load failed: {exc})",
                BundleDiscoveryWarning,
                stacklevel=2,
            )
            continue
        if _is_bundle_payload(payload):
            descriptors.append(_bundle_descriptor(file, payload))
        else:
            got = (
                type(payload).__name__
                if not isinstance(payload, dict)
                else f"dict, first keys: {list(payload)[:6]}"
            )
            warn(
                f"{file.name}: skipped (not a member bundle: {got})",
                BundleDiscoveryWarning,
                stacklevel=2,
            )
    return descriptors


def _describe_scan_root(path: str | Path | Sequence[str | Path]) -> str:
    """Human-readable name of a scan input for error messages."""
    if isinstance(path, (str, Path)):
        return str(Path(path))
    files = [Path(x) for x in path]
    return f"[{len(files)} explicit paths, first: {files[0]}]" if files else "[]"


def load_bundle_pool(
    path: str | Path | Sequence[str | Path],
    *,
    arm: str = "ts2",
    device: str | torch.device = "cpu",
) -> list[LoadedMember]:
    """Load a serving-ready member pool from bundle-v1 files (one arm).

    The recommended serving load path (POSTMERGE_RUNBOOK step 3): scan
    with :func:`discover_bundles`, keep ONE arm — the serving pin is
    ts2, hence the default — and rebuild every matching member through
    :func:`load_member_bundle`, which enforces the inference-ready
    contract (eval mode, EVERY parameter ``requires_grad=False``) and
    reads the embedded norm sidecar.  A bundle built for weight
    comparison is NOT a serving artifact until its six fit-stat keys are
    real (the pm20260831 identity-norm incident, fixed 2026-09-01; see
    the README of ``/nfs/wangxi/runs/ckpt_bundle_pm20260831``).

    Ops case for the switch (2026-08-31 production rehearsal,
    ``/nfs/wangxi/runs/ckpt_bundle_rehearsal_20260831/rehearsal.json``):
    cold 20-member build 0.623 s from bundles vs 0.688 s from the bare
    two-file ckpts, storage ratio 1.029, and bit-exactness of every
    rebuilt tensor vs the bare pool established by the 1160-tensor
    ``torch.equal`` sweep.  The bare ckpts stay the archival source of
    truth (runbook step 4).

    Members are returned in stable sorted-by-filename order (s0..s9 for
    the production naming), ready for
    ``PerMemberEnsembleBackend.from_bundles(load_bundle_pool(dir,
    arm="ts2"))``.  Zero matches raises a :class:`ValueError` naming the
    scan root, the arm filter and what was discovered instead.
    """
    descriptors = discover_bundles(path)
    matching = [d for d in descriptors if d["arm"] == arm]
    if not matching:
        arms_found: dict[str, int] = {}
        for d in descriptors:
            key = d["arm"] if d["arm"] is not None else f"param_dim={d['param_dim']!r}"
            arms_found[key] = arms_found.get(key, 0) + 1
        summary = ", ".join(f"{k} x {v}" for k, v in sorted(arms_found.items()))
        raise ValueError(
            f"no member bundles with arm={arm!r} under {_describe_scan_root(path)} "
            f"(discovered instead: {summary or 'no bundles at all'}); pass a "
            f"matching arm= (bundles carry ts2/ts4 via arch.body.param_dim) or "
            f"point at the directory holding the {arm} bundles"
        )
    return [load_member_bundle(d["path"], device=device) for d in matching]


class PerMemberEnsembleBackend(ModelEnsembleBackend):
    """Ensemble over explicit ``(encoder, body)`` pairs — one per member.

    Closes GAP 3: the base backend builds one shared cond matrix for all
    members, but a two-stage pool member needs its OWN seed's latent.  Each
    member here is ``(encoder, body)`` where ``body`` is a served
    ``TwoStageCondFNODrag`` (or a :class:`LoadedMember`, from which the
    pair and the norm sidecar are taken); ``predict`` normalises the field
    and the param columns with each member's fit stats, recomputes the
    latent from the query SDF through that member's own trunk and returns
    the ``(M, N)`` linear-C_D member matrix with the base class's exact
    aggregation arithmetic (``10 ** (z * y_std + y_mean)``, stacked).

    Norm contract: ``p_mean``/``p_std`` cover the **param columns only**
    (length ``param_dim``); the latent columns ride raw.  This is
    bit-identical to serving the bare body through the base backend with
    the cond-vector norms padded (zeros/ones on the latent columns).

    The trunk is served in eval mode; the backend itself leaves
    ``requires_grad`` untouched — loader-built members arrive fully frozen
    (inference-ready, see :func:`load_member_bundle`) and hand-built pairs
    keep whatever grad state they had (``predict`` runs under ``no_grad``,
    so the frozen-detach branch of ``TwoStageCondFNODrag.forward`` is
    value-neutral either way).  Note the SDF input
    has no slot in the v3/v4 service query API, so this backend is a
    model-level ensemble, not a drop-in for ``DragSurrogateService``.
    """

    def __init__(
        self,
        pairs: Sequence[Any],
        norms: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
        *,
        device: str | torch.device = "cpu",
        labels: Sequence[str] | None = None,
    ) -> None:
        if not pairs:
            raise ValueError("ensemble needs at least one (encoder, body) pair")
        resolved: list[tuple[nn.Module, nn.Module]] = []
        bundled_norms: list[Mapping[str, Any]] = []
        member_meta: list[dict[str, Any]] = []
        all_bundled = True
        for item in pairs:
            if isinstance(item, LoadedMember):
                resolved.append((item.stage1, item.model))
                bundled_norms.append(item.norm)
                member_meta.append(dict(item.meta))
            elif isinstance(item, tuple) and len(item) == 2:
                enc, body = item
                resolved.append((enc, body))
                member_meta.append({})
                all_bundled = False
            else:
                raise ValueError(
                    f"each pair must be (encoder, body) or a LoadedMember, got {type(item).__name__}"
                )
        if norms is None:
            if not all_bundled:
                raise ValueError(
                    "norms is required unless every member is a LoadedMember with a full sidecar"
                )
            norm_seq: list[Mapping[str, Any]] = bundled_norms
        elif isinstance(norms, Mapping):
            norm_seq = [norms] * len(resolved)
        else:
            norm_seq = list(norms)
            if len(norm_seq) != len(resolved):
                raise ValueError(f"got {len(resolved)} members but {len(norm_seq)} norm blocks")
        for i, n in enumerate(norm_seq):
            missing = [k for k in _MEMBER_NORM_KEYS if k not in n]
            if missing:
                raise ValueError(f"member {i} norm missing keys: {missing}")

        param_dims: set[int] = set()
        for enc, body in resolved:
            pd = getattr(body, "param_dim", None)
            if not isinstance(pd, int) or isinstance(pd, bool) or pd < 0:
                raise ValueError(
                    "each body must expose an int param_dim (TwoStageCondFNODrag does)"
                )
            ld_enc = getattr(enc, "latent_dim", None)
            ld_body = getattr(body, "latent_dim", None)
            if ld_enc != ld_body or not isinstance(ld_body, int):
                raise ValueError(
                    f"pair latent_dim mismatch: encoder {ld_enc!r} vs body {ld_body!r}"
                )
            trunk = enc.encoder if isinstance(enc, SupervisedSDFEncoder) else enc
            body_encoder = getattr(body, "encoder", None)
            if body_encoder is not None and body_encoder is not trunk and body_encoder is not enc:
                raise ValueError(
                    "pair encoder is not the trunk inside the body module — the served "
                    "latent would silently come from a different encoder"
                )
            param_dims.add(int(pd))
        if len(param_dims) != 1:
            raise ValueError(f"ensemble members disagree on param_dim: {sorted(param_dims)}")
        self._param_dim = param_dims.pop()
        for i, n in enumerate(norm_seq):
            if (
                len(np.asarray(n["p_mean"]).reshape(-1)) != self._param_dim
                or len(np.asarray(n["p_std"]).reshape(-1)) != self._param_dim
            ):
                raise ValueError(
                    f"member {i}: p_mean/p_std must have exactly param_dim={self._param_dim} "
                    "columns (latent columns are served raw, never z-scored)"
                )

        self.device = torch.device(device)
        self._members = resolved
        # base-compatible member storage (the base list is CondFNODrag-typed;
        # these are TwoStageCondFNODrag bodies, so the slot is widened to Any)
        self._models: list[Any] = [body.to(self.device).eval() for _, body in resolved]
        for enc, _ in resolved:
            enc.eval()
        self._norms = [{k: np.asarray(v) for k, v in n.items()} for n in norm_seq]
        self._member_meta = member_meta
        self._labels = [str(x) for x in labels] if labels is not None else None

    @classmethod
    def from_bundles(
        cls,
        bundles: Sequence[LoadedMember],
        *,
        device: str | torch.device = "cpu",
    ) -> PerMemberEnsembleBackend:
        """Build the ensemble from :class:`LoadedMember` sidecars directly."""
        return cls(list(bundles), norms=None, device=device)

    @property
    def cond_dim(self) -> int:
        """Condition width :meth:`predict` accepts — the param columns."""
        return self._param_dim

    @property
    def kind(self) -> str:
        return "per-member-model"

    def member_labels(self) -> list[str]:
        if self._labels is not None:
            return list(self._labels)
        return [str(m.get("member", f"m{i}")) for i, m in enumerate(self._member_meta)]

    def _sdf_tensor(self, sdf: np.ndarray, *, batched: bool = False) -> torch.Tensor:
        """Coerce an SDF input to ``(1, C, D, H, W)`` (or ``(G, C, D, H, W)``)."""
        a = np.asarray(sdf, dtype=np.float32)
        if batched:
            if a.ndim == 4:  # (G, D, H, W) — one single-channel volume per geometry
                a = a[:, None]
            elif a.ndim == 6 and a.shape[1] == 1:  # (G, 1, C, D, H, W)
                a = a[:, 0]
            elif a.ndim != 5:
                raise ValueError(
                    f"batched sdf must be (G, D, H, W), (G, C, D, H, W) or "
                    f"(G, 1, C, D, H, W), got {a.shape}"
                )
        elif a.ndim == 3:  # (D, H, W) — single-channel volume
            a = a[None, None]
        elif a.ndim == 4:  # (C, D, H, W)
            a = a[None]
        elif a.ndim != 5 or a.shape[0] != 1:
            raise ValueError(
                f"sdf must be (D, H, W), (C, D, H, W) or (1, C, D, H, W), got {a.shape}"
            )
        return torch.from_numpy(np.ascontiguousarray(a)).to(self.device)

    # The SDF input has no slot in the base (encoder-less) signature, so the
    # override deliberately widens it; the aggregation arithmetic is shared.
    def predict(self, fields: np.ndarray, sdf: np.ndarray, cond: np.ndarray) -> np.ndarray:  # type: ignore[override]
        """Serve one design: ``(M, N)`` member C_D rows in linear space.

        Batch contract: the ``N`` cond rows are served against ONE shared
        field/sdf — the latent is recomputed once per member on a
        batch-of-``N`` SDF pass and the field is expanded ``N``-fold.  For
        per-row fields/sdfs use :meth:`predict_batch` (``counts`` grouping)
        instead.

        Per member this is exactly a direct ``TwoStageCondFNODrag.forward``
        on the normalised inputs (field z-scored by ``ch_*``, param columns
        z-scored by ``p_*``, latent recomputed raw from ``sdf``); the
        aggregation is the base class's (``10 ** (z * y_std + y_mean)``
        stacked over members).  ``fields`` is the ``(5, ny, nx)`` reference
        field, ``sdf`` the query geometry volume and ``cond`` the
        ``(N, param_dim)`` param rows.

        Batch composition matters at the last-ulp level: a per-row
        single-design query through this method (``N == 1`` per call) pays
        a batch-1 vs batch-``N`` latent-recompute fp difference of ~1e-5
        and ~20 ms per call overhead versus folding the row into one
        shared-design batch (2026-08-31 production rehearsal,
        ``/nfs/wangxi/runs/ckpt_bundle_rehearsal_20260831``).  Compose
        batches of a shared design rather than looping per row when the
        rows share one field/sdf.
        """
        fields = np.asarray(fields, dtype=np.float32)
        cond = np.asarray(cond, dtype=np.float64)
        if fields.ndim != 3 or fields.shape[0] != 5:
            raise ValueError(f"fields must be (5, ny, nx), got {fields.shape}")
        if cond.ndim != 2 or cond.shape[1] != self._param_dim:
            raise ValueError(f"cond must be (N, {self._param_dim}), got {cond.shape}")
        n = cond.shape[0]
        x = torch.from_numpy(fields).to(self.device)
        p = torch.from_numpy(cond.astype(np.float32)).to(self.device)
        xn = x.unsqueeze(0).expand(n, -1, -1, -1)  # (N, 5, ny, nx)
        sdft = self._sdf_tensor(sdf).expand(n, -1, -1, -1, -1)
        outs = []
        with torch.no_grad():
            for model, norm in zip(self._models, self._norms):
                ch_m = torch.as_tensor(norm["ch_mean"], dtype=torch.float32, device=self.device)
                ch_s = torch.as_tensor(norm["ch_std"], dtype=torch.float32, device=self.device)
                p_m = torch.as_tensor(norm["p_mean"], dtype=torch.float32, device=self.device)
                p_s = torch.as_tensor(norm["p_std"], dtype=torch.float32, device=self.device)
                y_m = float(norm["y_mean"])
                y_s = float(norm["y_std"])
                x_norm = (xn - ch_m.view(1, -1, 1, 1)) / ch_s.view(1, -1, 1, 1)
                p_norm = (p - p_m) / p_s
                z = model(x_norm, sdft, p_norm)
                outs.append(10.0 ** (z.double().cpu().numpy() * y_s + y_m))
        return np.stack(outs, axis=0)  # (M, N)

    def predict_batch(  # type: ignore[override]
        self,
        fields: np.ndarray,
        sdfs: np.ndarray,
        cond: np.ndarray,
        counts: np.ndarray,
    ) -> np.ndarray:
        """Batched multi-geometry variant of :meth:`predict`.

        Batch contract: this is the per-row-fields/sdfs path — geometry
        ``g``'s field and sdf carry exactly its ``counts[g]`` cond rows
        (``repeat_interleave`` grouping), unlike :meth:`predict`, which
        serves all ``N`` cond rows against ONE shared field/sdf.

        ``fields`` is ``(G, 5, ny, nx)``, ``sdfs`` ``(G, C, D, H, W)`` or
        ``(G, 1, C, D, H, W)``, ``cond`` the concatenated param rows and
        ``counts`` the per-geometry row counts — same expansion
        (``repeat_interleave``), normalisation and de-scaling as the base
        :meth:`~tensorlbm.ai.inference_service.ModelEnsembleBackend.predict_batch`.
        As with :meth:`predict`, batch composition matters at the last-ulp
        level (batch-kernel fp differences); group rows by shared design
        rather than re-issuing per-row calls.
        """
        fields = np.asarray(fields, dtype=np.float32)
        cond = np.asarray(cond, dtype=np.float64)
        counts = np.asarray(counts, dtype=np.int64)
        if fields.ndim != 4 or fields.shape[1] != 5:
            raise ValueError(f"fields must be (G, 5, ny, nx), got {fields.shape}")
        if cond.ndim != 2 or cond.shape[1] != self._param_dim:
            raise ValueError(f"cond must be (N, {self._param_dim}), got {cond.shape}")
        if counts.ndim != 1 or counts.size != fields.shape[0] or not (counts > 0).all():
            raise ValueError(
                f"counts must be positive with one entry per field, got {counts!r} "
                f"for {fields.shape[0]} fields"
            )
        if int(counts.sum()) != cond.shape[0]:
            raise ValueError(f"counts sum {int(counts.sum())} != condition rows {cond.shape[0]}")
        x = torch.from_numpy(fields).to(self.device)
        p = torch.from_numpy(cond.astype(np.float32)).to(self.device)
        sdft = self._sdf_tensor(sdfs, batched=True)
        if sdft.shape[0] != fields.shape[0]:
            raise ValueError(f"sdfs batch {sdft.shape[0]} != fields batch {fields.shape[0]}")
        reps = torch.as_tensor(counts, device=self.device)
        xn = torch.repeat_interleave(x, reps, dim=0)  # (N, 5, ny, nx)
        zn = torch.repeat_interleave(sdft, reps, dim=0)  # (N, 1, C, D, H, W)
        outs = []
        with torch.no_grad():
            for model, norm in zip(self._models, self._norms):
                ch_m = torch.as_tensor(norm["ch_mean"], dtype=torch.float32, device=self.device)
                ch_s = torch.as_tensor(norm["ch_std"], dtype=torch.float32, device=self.device)
                p_m = torch.as_tensor(norm["p_mean"], dtype=torch.float32, device=self.device)
                p_s = torch.as_tensor(norm["p_std"], dtype=torch.float32, device=self.device)
                y_m = float(norm["y_mean"])
                y_s = float(norm["y_std"])
                x_norm = (xn - ch_m.view(1, -1, 1, 1)) / ch_s.view(1, -1, 1, 1)
                p_norm = (p - p_m) / p_s
                z = model(x_norm, zn, p_norm)
                outs.append(10.0 ** (z.double().cpu().numpy() * y_s + y_m))
        return np.stack(outs, axis=0)  # (M, N)
