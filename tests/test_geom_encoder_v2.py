"""Tests for the B4 SDF-2 encoder (``tensorlbm.ai.geom_encoder`` v2 additions).

Pins the v2 representation guarantees alongside the untouched v1 contract:

- v1 backward compatibility: the v1 classes keep their identity, signature
  defaults and behaviour (the v1 test file stays green unmodified);
- v2 encoder shapes, dtype, ``tanh`` bounds and multi-scale feature shapes;
- parameter budget: 253,208 (base 12) / 446,400 (base 16), both < 500k;
- determinism: repeated forwards bitwise-identical on one device; CPU vs
  CUDA agree to float round-off (float32 conv3d reductions are not bitwise
  portable across devices — max observed diff ~3e-5);
- effective-dimensionality helpers (:func:`latent_spectrum`,
  :func:`participation_ratio`) on synthetic corpora of known rank;
- :func:`vicreg_latent_penalty` unit behaviour: large on a rank-1 (collapsed)
  batch, near the sampling-noise floor on an uncorrelated unit-std batch,
  zero for an exactly-isotropic uncorrelated batch;
- joint :class:`SDFCondFNODragV2` forward shapes (param+latent, aux head,
  zero-param ``sdf_only`` wiring) and a backward smoke test.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tensorlbm.ai.geom_encoder import (
    SDFCondFNODrag,
    SDFCondFNODragV2,
    SDFEncoder,
    SDFEncoderV2,
    latent_spectrum,
    logit_margin_penalty,
    participation_ratio,
    vicreg_latent_penalty,
)
from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask


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


def _helmert(n: int, k: int) -> np.ndarray:
    """First ``k`` Helmert basis columns: orthonormal, each exactly mean-zero."""
    h = np.zeros((n, k))
    for j in range(k):
        h[: j + 1, j] = 1.0
        h[j + 1, j] = -(j + 1)
        h[:, j] /= np.sqrt((j + 1) * (j + 2))
    return h


class TestV1BackwardCompat:
    def test_v1_classes_are_the_v1_objects(self):
        """v2 is purely additive: v1 classes/defaults untouched."""
        assert SDFEncoder(latent_dim=8).latent_dim == 8
        enc = SDFEncoder(latent_dim=32, base=8)
        assert sum(p.numel() for p in enc.parameters()) == 46_288
        m = SDFCondFNODrag(param_dim=4, hand_dim=0, latent_dim=8)
        assert m.encoder.latent_dim == 8
        assert type(m.encoder) is SDFEncoder  # v1 joint model still wires v1

    def test_v1_latent_contract_unchanged(self):
        enc = SDFEncoder(latent_dim=8)
        enc.eval()
        sdf = torch.rand(2, 1, 32, 32, 64) * 2 - 1
        with torch.no_grad():
            z = enc(sdf)
        assert z.shape == (2, 8) and z.dtype == torch.float32
        assert float(z.abs().max()) <= 1.0

    def test_v2_joint_does_not_alias_v1(self):
        assert SDFCondFNODragV2 is not SDFCondFNODrag
        m = SDFCondFNODragV2()
        assert type(m.encoder) is SDFEncoderV2


class TestEncoderV2:
    def test_shapes_dtype_bounds(self):
        enc = SDFEncoderV2(latent_dim=32)
        sdf = torch.rand(3, 1, 32, 32, 64) * 2 - 1
        with torch.no_grad():
            z = enc(sdf)
        assert z.shape == (3, 32)
        assert z.dtype == torch.float32
        assert float(z.abs().max()) <= 1.0

    def test_multiscale_shapes(self):
        """Per-scale pooled features: (B, 4b), (B, 8b), (B, 8b)."""
        for base in (4, 12, 16):
            enc = SDFEncoderV2(latent_dim=8, base=base)
            f1, f2, f3 = enc.scale_features(torch.rand(2, 1, 32, 32, 64))
            assert f1.shape == (2, base * 4)
            assert f2.shape == (2, base * 8)
            assert f3.shape == (2, base * 8)

    def test_multiscale_spatial_depths(self):
        """Scale maps live at 16x16x32 / 8x8x16 / 4x4x8 (features from 3 depths)."""
        enc = SDFEncoderV2(latent_dim=8, base=4)
        x = torch.rand(1, 1, 32, 32, 64)
        x1 = enc.scale1(enc.stem(x))
        x2 = enc.scale2(x1)
        x3 = enc.scale3(x2)
        assert tuple(x1.shape) == (1, 8, 16, 16, 32)
        assert tuple(x2.shape) == (1, 16, 8, 8, 16)
        assert tuple(x3.shape) == (1, 16, 4, 4, 8)

    def test_parameter_budget(self):
        n12 = sum(p.numel() for p in SDFEncoderV2(latent_dim=32, base=12).parameters())
        n16 = sum(p.numel() for p in SDFEncoderV2(latent_dim=32, base=16).parameters())
        assert n12 == 253_208
        assert n16 == 446_400
        assert n12 < 500_000 and n16 < 500_000

    def test_latent_bitwise_reproducible(self):
        enc = SDFEncoderV2(latent_dim=16)
        enc.eval()
        sdf = torch.rand(2, 1, 32, 32, 64) * 2 - 1
        with torch.no_grad():
            a, b = enc(sdf), enc(sdf)
        assert torch.equal(a, b)

    def test_same_geometry_same_latent(self):
        enc = SDFEncoderV2(latent_dim=8)
        enc.eval()
        sdf = suboff_sdf("full", 1.2, 0.8)
        with torch.no_grad():
            z = torch.cat([enc(sdf), enc(sdf), enc(sdf)], dim=0)
        assert torch.equal(z[0], z[1]) and torch.equal(z[1], z[2])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
    def test_cpu_cuda_agree(self):
        """CPU and CUDA latents agree to float round-off (NOT bitwise).

        Float32 conv3d reductions use different summation orders on CPU
        and CUDA; the v1 bitwise guarantee covers the integer EDT only.
        Observed max |diff| ~3e-5 on the production input size.
        """
        torch.manual_seed(0)
        enc = SDFEncoderV2(latent_dim=32, base=4)
        enc.eval()
        sdf = torch.rand(2, 1, 32, 32, 64) * 2 - 1
        with torch.no_grad():
            cpu = enc(sdf)
            gpu = enc.cuda()(sdf.cuda()).cpu()
        assert torch.allclose(cpu, gpu, atol=1e-4, rtol=0.0)
        assert float((cpu - gpu).abs().max()) < 1e-4

    def test_thin_appendage_changes_latent(self):
        """A 1-voxel-thick slab (fin-scale support) moves the v2 latent."""
        enc = SDFEncoderV2(latent_dim=8, base=4)
        enc.eval()
        body = torch.zeros(1, 1, 16, 16, 32)
        body[:, :, 4:12, 6:10, 6:26] = -1.0  # thick body
        thin = body.clone()
        thin[:, 0, 8, 8, 2:30] = -0.05  # 1-voxel-thick appendage, weak contrast
        with torch.no_grad():
            z_body, z_thin = enc(body), enc(thin)
        assert not torch.equal(z_body, z_thin)

    def test_invalid_dims(self):
        with pytest.raises(ValueError, match=">= 1"):
            SDFEncoderV2(latent_dim=0)
        with pytest.raises(ValueError, match=">= 1"):
            SDFEncoderV2(latent_dim=8, base=0)


class TestResidualBlock:
    def test_identity_skip(self):
        from tensorlbm.ai.geom_encoder import ResidualBlock3d

        blk = ResidualBlock3d(8, 8, stride=1)
        assert isinstance(blk.skip, torch.nn.Identity)
        assert blk(torch.rand(2, 8, 8, 8, 16)).shape == (2, 8, 8, 8, 16)

    def test_projection_skip(self):
        from tensorlbm.ai.geom_encoder import ResidualBlock3d

        blk = ResidualBlock3d(4, 8, stride=2)
        assert isinstance(blk.skip, torch.nn.Conv3d)
        assert blk(torch.rand(2, 4, 16, 16, 32)).shape == (2, 8, 8, 8, 16)

    def test_invalid(self):
        from tensorlbm.ai.geom_encoder import ResidualBlock3d

        with pytest.raises(ValueError, match=">= 1"):
            ResidualBlock3d(0, 8)


class TestVicregPenalty:
    def test_rank1_much_larger_than_full_rank(self):
        torch.manual_seed(0)
        full = torch.randn(64, 32)
        rank1 = torch.randn(64, 1).repeat(1, 32)
        assert float(vicreg_latent_penalty(rank1)) > 20.0
        assert float(vicreg_latent_penalty(rank1)) > 10 * float(vicreg_latent_penalty(full))

    def test_full_rank_below_sampling_floor(self):
        """Uncorrelated unit-std batch: cov term ~ (d-1)/B, variance hinge 0."""
        torch.manual_seed(0)
        full = torch.randn(512, 32)
        assert float(vicreg_latent_penalty(full)) < 0.5

    def test_zero_for_exactly_isotropic_uncorrelated_batch(self):
        """Helmert basis scaled to unbiased std 1: hinge 0, corr off-diag 0."""
        b, d = 64, 32
        z = torch.as_tensor(_helmert(b, d) * np.sqrt(b - 1), dtype=torch.float32)
        assert float(vicreg_latent_penalty(z)) == pytest.approx(0.0, abs=1e-4)

    def test_saturated_column_gradient_is_finite(self):
        """Exact +-1 columns (saturated tanh, zero variance): finite grads.

        Regression pin: the un-eps'd ``torch.std`` produced inf gradients
        here (d sqrt(var)/d var -> inf at var = 0) and NaN weights late in
        v2_reg training.
        """
        pat = torch.where(torch.arange(64) % 2 == 0, -1.0, 1.0)
        scale = torch.full((1, 32), 0.5)
        scale[0, 0] = 1.0  # column 0 saturated at +-1, std ~ 1 (hinge off)
        z = (pat.unsqueeze(1) * scale).requires_grad_(True)
        vicreg_latent_penalty(z).backward()
        assert torch.isfinite(z.grad).all()
        assert bool((z.grad[:, 0].abs() > 0).all())  # cov term still flows
        assert bool((z.grad[:, 1].abs() > 0).all())  # hinge + cov term

    def test_collapsed_column_penalised(self):
        """A dead dimension on an otherwise exact batch adds hinge/32 exactly."""
        b, d = 64, 32
        z = torch.as_tensor(_helmert(b, d) * np.sqrt(b - 1), dtype=torch.float32)
        z1 = z.clone()
        z1[:, 5] = 0.0
        # dead column: std = sqrt(VAR_EPS) = 0.01 -> hinge 0.99, mean over d
        assert float(vicreg_latent_penalty(z1)) == pytest.approx(0.99 / d, abs=1e-4)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError, match=r"\(B, D\)"):
            vicreg_latent_penalty(torch.randn(5))
        with pytest.raises(ValueError, match=">= 2 rows"):
            vicreg_latent_penalty(torch.randn(1, 8))
        with pytest.raises(ValueError, match="positive"):
            vicreg_latent_penalty(torch.randn(8, 4), var_target=0.0)
        with pytest.raises(ValueError, match=">= 0"):
            vicreg_latent_penalty(torch.randn(8, 4), gamma=-1.0)


class TestEffectiveDim:
    def test_known_rank(self):
        """Isotropic rank-r corpus: PR == r (independent of the scale)."""
        rng = np.random.default_rng(0)
        for rank in (1, 4, 8):
            q = np.linalg.qr(rng.standard_normal((2000, rank)))[0]
            assert participation_ratio(q * 0.5) == pytest.approx(rank, rel=0.05)

    def test_anisotropic_pr_between_one_and_rank(self):
        rng = np.random.default_rng(3)
        q = np.linalg.qr(rng.standard_normal((2000, 8)))[0]
        z = q * np.linspace(8.0, 1.0, 8)  # decaying spectrum: 1 < PR < 8
        pr = participation_ratio(z)
        assert 1.0 < pr < 8.0

    def test_isotropic_full_rank(self):
        rng = np.random.default_rng(1)
        z = rng.standard_normal((5000, 16))
        assert participation_ratio(z) == pytest.approx(16.0, abs=1.0)

    def test_spectrum_properties(self):
        rng = np.random.default_rng(2)
        z = np.linalg.qr(rng.standard_normal((300, 6)))[0] * np.array([9, 3, 1, 1, 1, 1])
        ev = latent_spectrum(z)
        assert ev.shape == (min(300, 6),)
        assert float(ev.sum()) == pytest.approx(1.0)
        assert bool(np.all(np.diff(ev) <= 1e-12))  # descending
        assert float(ev[0]) > 0.5
        assert latent_spectrum(z, n_components=3).shape == (3,)

    def test_invalid(self):
        with pytest.raises(ValueError, match=r"\(N, D\)"):
            latent_spectrum(np.zeros((1, 4)))
        with pytest.raises(ValueError, match=">= 1"):
            latent_spectrum(np.zeros((4, 2)), n_components=0)
        with pytest.raises(ValueError, match="zero total variance"):
            latent_spectrum(np.ones((4, 2)))


class TestLogitMarginPenalty:
    def test_zero_inside_margin(self):
        """All |logits| <= margin: no penalty (keeps tanh' >= 0.07 at margin 2)."""
        z = torch.zeros(8, 4)
        z[0, 0], z[0, 1] = 2.0, -2.0  # exactly at the boundary
        assert float(logit_margin_penalty(z, margin=2.0)) == 0.0

    def test_quadratic_above_margin(self):
        z = torch.zeros(4, 2)
        z[0, 0] = 3.0  # excess 1 on one of 8 entries -> mean = 1/8
        assert float(logit_margin_penalty(z, margin=2.0)) == pytest.approx(1.0 / 8.0)
        z2 = torch.zeros(4, 2)
        z2[0, 0] = 4.0  # excess 2 -> squared 4, one of 8 entries
        assert float(logit_margin_penalty(z2, margin=2.0)) == pytest.approx(4.0 / 8.0)

    def test_symmetric_in_sign(self):
        torch.manual_seed(0)
        z = torch.randn(16, 8) * 3
        a = logit_margin_penalty(z, margin=1.0)
        b = logit_margin_penalty(-z, margin=1.0)
        assert float(a) == float(b)

    def test_saturated_logits_get_gradient(self):
        """Regression pin for the v2_reg2 guard: |logits| >> margin still flows."""
        z = (torch.randn(16, 4) * 40).requires_grad_(True)
        logit_margin_penalty(z, margin=2.0).backward()
        assert torch.isfinite(z.grad).all()
        assert float(z.grad.abs().max()) > 0.0

    def test_invalid(self):
        with pytest.raises(ValueError, match=r"\(B, D\)"):
            logit_margin_penalty(torch.randn(6))
        with pytest.raises(ValueError, match="positive"):
            logit_margin_penalty(torch.randn(6, 2), margin=-1.0)


class TestLatentPair:
    def test_forward_with_logits_matches_forward(self):
        enc = SDFEncoderV2(latent_dim=8, base=4)
        enc.eval()
        sdf = torch.rand(2, 1, 32, 32, 64) * 2 - 1
        with torch.no_grad():
            z, logits = enc.forward_with_logits(sdf)
        assert z.shape == (2, 8) and logits.shape == (2, 8)
        assert torch.equal(z, enc(sdf))
        assert torch.equal(z, torch.tanh(logits))

    def test_joint_latent_pair(self):
        m = SDFCondFNODragV2(param_dim=4, latent_dim=8, aux_dim=4, encoder_base=4)
        m.eval()
        sdf = torch.rand(2, 1, 32, 32, 64)
        with torch.no_grad():
            z, logits = m.latent_pair(sdf)
        assert torch.equal(z, m.encoder(sdf))
        assert torch.equal(z, torch.tanh(logits))

    def test_v1_encoder_has_no_pair(self):
        """v1 stays untouched: no forward_with_logits / latent_pair additions."""
        assert not hasattr(SDFEncoder(latent_dim=8), "forward_with_logits")
        assert not hasattr(SDFCondFNODrag(param_dim=4, hand_dim=0, latent_dim=8), "latent_pair")


class TestJointModelV2:
    def _model(self, **kw):
        return SDFCondFNODragV2(
            in_ch=5, width=8, n_layers=2, modes=(4, 8), mlp_hidden=16, film_hidden=8, **kw
        )

    def test_forward_shapes_param_plus_latent(self):
        m = self._model(param_dim=4, latent_dim=8, encoder_base=4)
        x = torch.rand(3, 5, 64, 128)
        sdf = torch.rand(3, 1, 32, 32, 64)
        p = torch.randn(3, 4)
        y = m(x, sdf, p)
        assert y.shape == (3,)

    def test_forward_with_aux(self):
        m = self._model(param_dim=4, latent_dim=8, aux_dim=4, encoder_base=4)
        y, aux = m(
            torch.rand(2, 5, 64, 128),
            torch.rand(2, 1, 32, 32, 64),
            torch.randn(2, 4),
            return_aux=True,
        )
        assert y.shape == (2,) and aux.shape == (2, 4)

    def test_sdf_only_zero_param_columns(self):
        m = self._model(param_dim=0, latent_dim=8, encoder_base=4)
        y = m(torch.rand(2, 5, 64, 128), torch.rand(2, 1, 32, 32, 64), torch.zeros(2, 0))
        assert y.shape == (2,)

    def test_wrong_p_width_raises(self):
        m = self._model(param_dim=4, latent_dim=8, encoder_base=4)
        with pytest.raises(ValueError, match="columns"):
            m(torch.rand(2, 5, 64, 128), torch.rand(2, 1, 32, 32, 64), torch.randn(2, 3))

    def test_invalid_dims(self):
        with pytest.raises(ValueError, match=">= 0"):
            self._model(param_dim=-1)

    def test_backward_smoke(self):
        m = self._model(param_dim=4, latent_dim=8, aux_dim=4, encoder_base=4)
        y, aux = m(
            torch.rand(2, 5, 64, 128),
            torch.rand(2, 1, 32, 32, 64),
            torch.randn(2, 4),
            return_aux=True,
        )
        (y.sum() + aux.sum()).backward()
        grads = [p.grad for p in m.encoder.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
