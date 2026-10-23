"""Two-stage SDF geometry encoder: supervised probe, then a frozen-latent drag surrogate.

Repair route for the joint-training negative result of B4 SDF-2 (PR #247 /
``docs/geom_encoder_v2_20260825.md``).  Under the locked joint protocol the
``tanh`` latent collapses -- participation ratio 0.00 for ``v2``/``v2_reg``
(corpus-constant), 1.01 for the margin-guarded ``v2_reg2`` -- even though the
geometry information IS recoverable from the input TODO.  The joint
objective, not encoder capacity, is the bottleneck.  This module implements
the two-stage alternative proposed in the v2 post-mortem (§6 "train the
encoder with a reconstruction/contrastive objective BEFORE the joint fit"):

**Stage 1 — supervised probe.**  The encoder trunk is trained to regress
Re-independent geometry descriptors (log sail/fin scales, hull indicators,
mask-derived geometry channels, CAD family multipliers) from the pooled SDF
through a *linear* probe head on the latent.  A linear (not MLP) probe keeps
all nonlinearity inside the trunk, so the targets stay linearly readable
from the latent — which is what stage 2 and any linear readout consume.

**Stage 2 — frozen-latent surrogate.**  The encoder is frozen, latents are
precomputed once, and the unchanged v3 FiLM-FNO body
(:class:`tensorlbm.ai.drag_cond.CondFNODrag`) is fitted on
``[z-scored scalar conditions | frozen latent]``.

Purely additive: composes the EXISTING encoder trunks
(:class:`~tensorlbm.ai.geom_encoder.SDFEncoder` /
:class:`~tensorlbm.ai.geom_encoder.SDFEncoderV2`) and regressor body; no
behaviour in ``geom_encoder.py`` / ``drag_cond.py`` changes.  The experiment
driver (per-corpus splits, quota sampling, target assembly) lives in the run
directory, not here.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from .drag_cond import CondFNODrag
from .geom_encoder import logit_margin_penalty

__all__ = [
    "SupervisedSDFEncoder",
    "TwoStageCondFNODrag",
    "build_probe_head",
    "r2_per_target",
]


def build_probe_head(latent_dim: int, target_dim: int, *, hidden: int = 0) -> nn.Sequential:
    """Linear (default) or small-MLP probe head from the latent to the targets.

    ``hidden=0`` — a single ``nn.Linear`` (the classic probing choice; keeps
    the target->latent map linear, see the module docstring).  ``hidden>0`` —
    ``Linear -> GELU -> Linear`` for capacity attribution.
    """
    if latent_dim < 1 or target_dim < 1 or hidden < 0:
        raise ValueError("latent_dim, target_dim must be >= 1 and hidden >= 0")
    if hidden == 0:
        return nn.Sequential(nn.Linear(latent_dim, target_dim))
    return nn.Sequential(nn.Linear(latent_dim, hidden), nn.GELU(), nn.Linear(hidden, target_dim))


class SupervisedSDFEncoder(nn.Module):
    """Stage-1 model: encoder trunk + probe head on its latent.

    ``encoder`` is any ``(B, in_ch, D, H, W) -> (B, latent_dim)`` trunk (the
    v1/v2 SDF encoders).  The probe head regresses the (z-scored) geometry
    target block from the latent; training minimises MSE on the targets,
    optionally plus the v2-lesson ``logit_margin`` guard when the trunk
    exposes pre-``tanh`` logits (``SDFEncoderV2.forward_with_logits``) — the
    guard keeps the ``tanh`` head responsive so the supervision cannot decay
    into the rank-0 dead equilibrium of the joint protocol.

    Use :meth:`margin_penalty` inside the training loop; the module itself
    stays loss-agnostic (same convention as ``geom_encoder``).
    """

    def __init__(self, encoder: nn.Module, target_dim: int, *, hidden: int = 0) -> None:
        super().__init__()
        latent_dim = getattr(encoder, "latent_dim", None)
        if not isinstance(latent_dim, int) or latent_dim < 1:
            raise ValueError(
                "encoder must expose an int latent_dim attribute "
                f"(SDFEncoder/SDFEncoderV2 do), got {latent_dim!r}"
            )
        if target_dim < 1:
            raise ValueError(f"target_dim must be >= 1, got {target_dim}")
        self.encoder = encoder
        self.latent_dim = int(latent_dim)
        self.target_dim = int(target_dim)
        self.probe = build_probe_head(self.latent_dim, target_dim, hidden=hidden)

    def forward(self, sdf: Tensor) -> tuple[Tensor, Tensor]:
        """``(latent, target_hat)`` — both differentiable for stage-1 training."""
        z = self.encoder(sdf)
        return z, self.probe(z)

    @torch.no_grad()
    def predict(self, sdf: Tensor) -> tuple[Tensor, Tensor]:
        """Eval convenience: ``(latent, target_hat)`` — :meth:`forward` without
        an autograd graph (same return order)."""
        return self.forward(sdf)

    def margin_penalty(self, sdf: Tensor, *, margin: float = 2.0) -> Tensor:
        """``logit_margin_penalty`` of the trunk, when it exposes logits.

        Returns a zero scalar for trunks without
        ``forward_with_logits`` (e.g. the v1 ``SDFEncoder``), so a stage-1
        loop can add it unconditionally.
        """
        fwd = getattr(self.encoder, "forward_with_logits", None)
        if fwd is None:
            return sdf.new_zeros(())
        _, logits = fwd(sdf)
        return logit_margin_penalty(logits, margin=margin)


class TwoStageCondFNODrag(nn.Module):
    """Stage-2 model: a FROZEN encoder latent + the unchanged v3 FiLM-FNO body.

    Mirrors :class:`tensorlbm.ai.geom_encoder.SDFCondFNODragV2` (same
    condition assembly ``[z-scored p | raw bounded latent]``, same body
    kwargs) but takes the encoder trunk as an argument and adds the frozen
    path:

    - :meth:`freeze_encoder` — ``requires_grad_(False)`` on the whole trunk;
      afterwards :meth:`forward` detaches the latent, so the body trains
      alone (the precomputed-latent fast path is :meth:`forward_from_latent`
      — bitwise the same model, see the tests).
    - :meth:`latents` — no-grad corpus latent extraction for stage 2.

    The trunk has no norm layers, so train/eval modes coincide for it; the
    frozen latent is therefore identical whether the parent module is in
    train or eval mode.
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        param_dim: int = 2,
        latent_dim: int | None = None,
        aux_dim: int = 0,
        in_ch: int = 5,
        width: int = 32,
        n_layers: int = 4,
        modes: tuple[int, int] = (16, 32),
        mlp_hidden: int = 128,
        film_hidden: int = 64,
    ) -> None:
        super().__init__()
        if param_dim < 0:
            raise ValueError(f"param_dim must be >= 0, got {param_dim}")
        ld = getattr(encoder, "latent_dim", None) if latent_dim is None else latent_dim
        if not isinstance(ld, int) or ld < 1:
            raise ValueError(f"latent_dim must be a positive int, got {ld!r}")
        if param_dim == 0 and ld < 1:
            raise ValueError("need at least one condition column (param or latent)")
        self.param_dim = int(param_dim)
        self.latent_dim = int(ld)
        self.encoder = encoder
        self.fno = CondFNODrag(
            in_ch=in_ch,
            width=width,
            n_layers=n_layers,
            modes=modes,
            cond_dim=self.param_dim + self.latent_dim,
            mlp_hidden=mlp_hidden,
            film_hidden=film_hidden,
            aux_dim=aux_dim,
        )

    @property
    def encoder_frozen(self) -> bool:
        """True once every encoder parameter has ``requires_grad=False``."""
        return all(not p.requires_grad for p in self.encoder.parameters())

    def freeze_encoder(self) -> "TwoStageCondFNODrag":
        """Freeze the trunk in place (idempotent); returns ``self``."""
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        return self

    def unfreeze_encoder(self) -> "TwoStageCondFNODrag":
        """Restore a trainable trunk (reduces :class:`SupervisedSDFEncoder`)."""
        self.encoder.requires_grad_(True)
        return self

    @torch.no_grad()
    def latents(self, sdf: Tensor) -> Tensor:
        """Corpus latent extraction for stage 2 — ``(B, latent_dim)``."""
        return cast(Tensor, self.encoder(sdf))

    def forward_from_latent(
        self, x: Tensor, z: Tensor, p: Tensor, return_aux: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Body forward on a PRECOMPUTED latent (stage-2 fast path).

        Bitwise equal to :meth:`forward` with the same latent — the encoder
        is skipped, not bypassed semantically.
        """
        if z.shape != (x.shape[0], self.latent_dim):
            raise ValueError(
                f"z must have shape ({x.shape[0]}, {self.latent_dim}), got {tuple(z.shape)}"
            )
        if p.shape[1] != self.param_dim:
            raise ValueError(f"p must have {self.param_dim} columns, got {p.shape[1]}")
        out = self.fno(x, torch.cat([p, z], dim=1), return_aux=return_aux)
        return cast(Tensor | tuple[Tensor, Tensor], out)

    def forward(
        self, x: Tensor, sdf: Tensor, p: Tensor, return_aux: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Full path: encoder (detached when frozen) -> FiLM-FNO body."""
        if p.shape[1] != self.param_dim:
            raise ValueError(f"p must have {self.param_dim} columns, got {p.shape[1]}")
        z = self.encoder(sdf)
        if self.encoder_frozen:
            z = z.detach()
        out = self.fno(x, torch.cat([p, z], dim=1), return_aux=return_aux)
        if return_aux:
            y, aux = cast(tuple[Tensor, Tensor], out)
            return y, aux
        return cast(Tensor, out)


def r2_per_target(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Coefficient of determination per target column (numpy, float64).

    ``1 - SSE/SST`` per column of the ``(N, T)`` blocks; a degenerate column
    (zero variance in ``true``) scores ``nan`` rather than a fake 1.0 — a
    constant target cannot evidence encoding.
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    if p.ndim != 2 or t.ndim != 2 or p.shape != t.shape:
        raise ValueError(f"pred/true must be matching (N, T) blocks, got {p.shape} vs {t.shape}")
    if p.shape[0] < 2:
        raise ValueError(f"need >= 2 rows, got {p.shape[0]}")
    sst = ((t - t.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    sse = ((p - t) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sst > 0.0, 1.0 - sse / sst, np.nan)
