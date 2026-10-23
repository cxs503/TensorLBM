"""Tests for the two-stage SDF encoder (``tensorlbm.ai.sdf_two_stage``).

Pins the two-stage contract that repairs the #247 joint-training latent
collapse:

- stage-1 :class:`SupervisedSDFEncoder` — shapes, latent/probe wiring, a
  linear-by-default probe head, real training on a tiny geometry task
  (distinct SDFs -> distinct targets, loss must fall), and the
  ``logit_margin`` guard plumbing for v2 trunks (zero for v1 trunks);
- stage-2 :class:`TwoStageCondFNODrag` — condition assembly
  ``[p | latent]``, freezing semantics (no encoder gradients reach the
  trunk, trunk weights bit-identical after body-only optimiser steps),
  precomputed-latent fast path bitwise equal to the full forward, and
  deterministic no-grad corpus extraction;
- :func:`r2_per_target` — perfect/constant/degenerate column behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tensorlbm.ai.drag_cond import CondFNODrag
from tensorlbm.ai.geom_encoder import SDFEncoder, SDFEncoderV2
from tensorlbm.ai.sdf_two_stage import (
    SupervisedSDFEncoder,
    TwoStageCondFNODrag,
    build_probe_head,
    r2_per_target,
)
from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask

TORCH_MANUAL_SEED = 0


def suboff_sdf(hull: str, sail: float, fin: float, *, res: int = 64) -> torch.Tensor:
    """Pooled SDF of a small SUBOFF placement (res 64 -> 16x16x32 canvas)."""
    from tensorlbm.ai.geom_encoder import sdf_volume

    nx = res
    mask, _ = build_suboff_mask(
        hull_type=hull,
        nx=nx,
        ny=res // 2,
        nz=res // 2,
        cx=nx * 0.35,
        cy=res / 4.0,
        cz=res / 4.0,
        length=0.6 * nx,
        config=SuboffConfig(sail_scale=sail, fin_scale=fin),
        device="cpu",
    )
    return sdf_volume(torch.as_tensor(mask, dtype=torch.bool))


def batch_of(n: int, *, res: int = 32) -> torch.Tensor:
    """Deterministic synthetic SDF-ish batch ``(n, 1, D, H, W)`` in [-1, 1]."""
    g = torch.Generator().manual_seed(1234)
    return torch.randn(n, 1, res // 2, res // 2, res, generator=g).clamp(-1, 1)


# ---------------------------------------------------------------------------
# probe head + supervised encoder
# ---------------------------------------------------------------------------
def test_probe_head_linear_by_default() -> None:
    head = build_probe_head(8, 5)
    assert isinstance(head[0], torch.nn.Linear)
    assert head[0].in_features == 8 and head[0].out_features == 5
    assert len(head) == 1


def test_probe_head_mlp_variant_and_validation() -> None:
    head = build_probe_head(8, 5, hidden=16)
    assert [m.out_features for m in head if isinstance(m, torch.nn.Linear)] == [16, 5]
    with pytest.raises(ValueError):
        build_probe_head(0, 5)
    with pytest.raises(ValueError):
        build_probe_head(8, 0)
    with pytest.raises(ValueError):
        build_probe_head(8, 5, hidden=-1)


def test_supervised_encoder_shapes_and_wiring() -> None:
    enc = SupervisedSDFEncoder(SDFEncoderV2(latent_dim=6, base=2), target_dim=4)
    sdf = batch_of(3, res=32)
    z, t = enc(sdf)
    assert z.shape == (3, 6) and t.shape == (3, 4)
    assert z.min() >= -1.0 and z.max() <= 1.0  # tanh-bounded trunk
    # the probe reads the latent: identical sdf -> identical targets
    z2, t2 = enc.predict(sdf)
    assert torch.equal(t2, enc.probe(enc.encoder(sdf)))
    assert torch.equal(z, z2)  # no_grad path reproduces the graph forward


def test_supervised_encoder_requires_latent_dim_attr() -> None:
    class NoDim(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    with pytest.raises(ValueError):
        SupervisedSDFEncoder(NoDim(), target_dim=4)


def test_margin_penalty_zero_for_v1_positive_for_saturated_v2() -> None:
    v1 = SupervisedSDFEncoder(SDFEncoder(latent_dim=4, base=2), target_dim=3)
    sdf = batch_of(4, res=32)
    assert float(v1.margin_penalty(sdf)) == 0.0  # v1 trunk has no logits API
    v2 = SupervisedSDFEncoder(SDFEncoderV2(latent_dim=4, base=2), target_dim=3)
    with torch.no_grad():  # saturate the head so |logits| > 2
        v2.encoder.head.weight.mul_(50.0)
        v2.encoder.head.bias.fill_(5.0)
    p2 = float(v2.margin_penalty(sdf).detach())
    p10 = float(v2.margin_penalty(sdf, margin=10.0).detach())
    assert p2 > 0.0 and p10 <= p2  # relu(|x|-m)^2 is non-increasing in m


def test_stage1_learns_distinct_targets_on_real_geometry() -> None:
    """Sail-scale regression must train through the probe (CPU, fast)."""
    torch.set_num_threads(4)  # tiny tensors: avoid oversubscription stalls
    torch.manual_seed(TORCH_MANUAL_SEED)
    sails = (0.25, 1.0, 4.0)
    sdf = torch.cat([suboff_sdf("with_sail", s, 1.0) for s in sails])
    target = torch.tensor([[np.log10(s)] for s in sails], dtype=torch.float32)
    enc = SupervisedSDFEncoder(SDFEncoderV2(latent_dim=8, base=4), target_dim=1)
    opt = torch.optim.AdamW(enc.parameters(), lr=3e-3, weight_decay=1e-4)
    first = last = None
    for step in range(300):
        opt.zero_grad()
        _, t = enc(sdf)
        loss = torch.nn.functional.mse_loss(t, target) + 0.1 * enc.margin_penalty(sdf)
        loss.backward()
        opt.step()
        if step == 0:
            first = float(loss.detach())
        last = float(loss.detach())
    assert last < 0.05 * first
    _, t = enc.predict(sdf)
    assert t[0, 0] < t[1, 0] < t[2, 0]  # sail-scale ordering recovered
    assert float((t - target).abs().max()) < 0.15


# ---------------------------------------------------------------------------
# stage-2 frozen model
# ---------------------------------------------------------------------------
def test_two_stage_condition_assembly_matches_v2() -> None:
    enc = SDFEncoderV2(latent_dim=7, base=2)
    m = TwoStageCondFNODrag(enc, param_dim=3, latent_dim=7, aux_dim=4, width=8, mlp_hidden=16)
    body = m.fno
    ref = CondFNODrag(in_ch=5, width=8, cond_dim=3 + 7, mlp_hidden=16, aux_dim=4)
    assert body.head[0].in_features == ref.head[0].in_features  # width + cond_dim


def test_two_stage_validation() -> None:
    with pytest.raises(ValueError):
        TwoStageCondFNODrag(SDFEncoderV2(latent_dim=4, base=2), param_dim=-1)
    with pytest.raises(ValueError):
        TwoStageCondFNODrag(SDFEncoderV2(latent_dim=4, base=2), latent_dim=0)


def test_freeze_blocks_encoder_gradients() -> None:
    torch.manual_seed(TORCH_MANUAL_SEED)
    m = TwoStageCondFNODrag(
        SDFEncoderV2(latent_dim=5, base=2),
        param_dim=2,
        width=8,
        mlp_hidden=16,
        modes=(4, 8),
        aux_dim=0,
    )
    assert not m.encoder_frozen
    m.freeze_encoder().freeze_encoder()  # idempotent
    assert m.encoder_frozen
    m.unfreeze_encoder()
    assert not m.encoder_frozen
    m.freeze_encoder()

    x = torch.randn(4, 5, 8, 16)
    sdf = batch_of(4, res=16)
    p = torch.randn(4, 2)
    y = m(x, sdf, p)
    y.sum().backward()
    enc_grads = [pp.grad for pp in m.encoder.parameters()]
    assert all(g is None for g in enc_grads)  # frozen: no gradient reaches the trunk
    assert any(g is not None and float(g.abs().sum()) > 0 for g in m.fno.parameters())


def test_frozen_trunk_weights_untouched_by_body_steps() -> None:
    torch.manual_seed(TORCH_MANUAL_SEED)
    m = TwoStageCondFNODrag(
        SDFEncoderV2(latent_dim=5, base=2), param_dim=2, width=8, mlp_hidden=16, modes=(4, 8)
    ).freeze_encoder()
    before = {k: v.clone() for k, v in m.encoder.state_dict().items()}
    opt = torch.optim.AdamW(filter(lambda q: q.requires_grad, m.parameters()), lr=1e-2)
    x = torch.randn(4, 5, 8, 16)
    sdf = batch_of(4, res=16)
    p = torch.randn(4, 2)
    for _ in range(3):
        opt.zero_grad()
        m(x, sdf, p).sum().backward()
        opt.step()
    for k, v in m.encoder.state_dict().items():
        assert torch.equal(before[k], v), f"encoder weight {k} moved under freeze"


def test_forward_from_latent_bitwise_equal() -> None:
    torch.manual_seed(TORCH_MANUAL_SEED)
    m = TwoStageCondFNODrag(
        SDFEncoderV2(latent_dim=5, base=2),
        param_dim=2,
        width=8,
        mlp_hidden=16,
        modes=(4, 8),
        aux_dim=3,
    ).freeze_encoder()
    m.eval()
    x = torch.randn(4, 5, 8, 16)
    sdf = batch_of(4, res=16)
    p = torch.randn(4, 2)
    y1, a1 = m(x, sdf, p, return_aux=True)
    y2, a2 = m.forward_from_latent(x, m.latents(sdf), p, return_aux=True)
    assert torch.equal(y1, y2) and torch.equal(a1, a2)
    with pytest.raises(ValueError):
        m.forward_from_latent(x, torch.zeros(4, 6), p)  # wrong latent width
    with pytest.raises(ValueError):
        m.forward_from_latent(x, m.latents(sdf), torch.zeros(4, 3))  # wrong p width
    with pytest.raises(ValueError):
        m(x, sdf, torch.zeros(4, 3))  # forward validates p too


def test_latents_deterministic_and_no_grad() -> None:
    m = TwoStageCondFNODrag(SDFEncoderV2(latent_dim=5, base=2), param_dim=2, width=8)
    m.train()  # must not matter: no norm layers in the trunk
    sdf = batch_of(3, res=16).requires_grad_(True)
    z = m.latents(sdf)
    assert not z.requires_grad
    assert torch.equal(z, m.latents(sdf))
    assert z.shape == (3, 5)


# ---------------------------------------------------------------------------
# r2_per_target
# ---------------------------------------------------------------------------
def test_r2_per_target_behaviour() -> None:
    t = np.array([[0.0, 4.0, 1.0], [1.0, 5.0, 2.0], [2.0, 6.0, 3.0], [3.0, 7.0, 4.0]])
    assert r2_per_target(t.copy(), t).tolist() == [1.0, 1.0, 1.0]
    r = r2_per_target(np.zeros_like(t), t)
    assert r[0] < 0.0  # constant-zero prediction of a varying target
    with pytest.raises(ValueError):
        r2_per_target(np.zeros((2, 2)), np.zeros((3, 2)))
    with pytest.raises(ValueError):
        r2_per_target(np.zeros((1, 2)), np.zeros((1, 2)))


def test_r2_per_target_degenerate_column_is_nan() -> None:
    t = np.array([[0.0, 5.0], [1.0, 5.0], [2.0, 5.0]])
    assert np.isnan(r2_per_target(t.copy(), t)[1])  # constant target, not a fake 1.0
