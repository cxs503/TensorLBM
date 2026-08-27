"""Tests for the B4 SDF geometry encoder (``tensorlbm.ai.geom_encoder``).

Pins the representation guarantees the SDF conditioning protocol relies on:

- exactness of the scipy-free EDT against the brute-force definition
  (minimum over the full opposite-phase voxel set) and an analytic
  single-voxel case;
- determinism (repeated calls, arbitrary chunking, CPU vs CUDA bitwise);
- sign convention (inside negative, outside positive, voxel units);
- bare-hull bit-invariance under (sail, fin) scaling at the SDF layer and
  fin no-op on ``with_sail`` (the v3 mask guarantees, carried through the
  distance transform and the pooling);
- clip/pool range and even-translation consistency of the downsampled SDF;
- encoder shapes, parameter budget and bitwise-reproducible latents;
- joint ``SDFCondFNODrag`` forward shapes (incl. the zero-param
  ``sdf_only`` wiring) and condition-block assembly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tensorlbm.ai.geom_encoder import (
    SDF_CLIP_VOXELS,
    SDFCondFNODrag,
    SDFEncoder,
    condition_sdf_params,
    sdf_volume,
    signed_distance_field,
)
from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask

HULLS = ("bare_hull", "with_sail", "full")


def small_grid():
    """A resolution-64 SUBOFF placement (32x32x64) — fast on CPU."""
    nx = 64
    return dict(nx=nx, ny=32, nz=32, cx=nx * 0.35, cy=16.0, cz=16.0, length=0.6 * nx)


def suboff_mask(hull: str, sail: float, fin: float) -> torch.Tensor:
    m, _ = build_suboff_mask(
        hull_type=hull,
        device="cpu",
        config=SuboffConfig(sail_scale=sail, fin_scale=fin),
        **small_grid(),
    )
    return m.bool()


def blob_mask(d: int = 16, h: int = 16, w: int = 26) -> torch.Tensor:
    """An off-boundary ellipsoid + sphere pair (shift-safe synthetic solid)."""
    zz, yy, xx = torch.meshgrid(
        torch.arange(d) * 1.0, torch.arange(h) * 1.0, torch.arange(w) * 1.0, indexing="ij"
    )
    ell = ((zz - 8) / 3.5) ** 2 + ((yy - 8) / 2.5) ** 2 + ((xx - 12) / 5.0) ** 2 <= 1.0
    sph = ((zz - 10) / 2.5) ** 2 + ((yy - 6) / 2.5) ** 2 + ((xx - 19) / 2.5) ** 2 <= 1.0
    return ell | sph


def brute_force_sdf(mask: torch.Tensor) -> torch.Tensor:
    """Ground-truth signed EDT: minimum over the FULL opposite-phase set."""
    m = mask.numpy()
    solid = np.argwhere(m)
    comp = np.argwhere(~m)
    grids = np.meshgrid(
        np.arange(m.shape[0]), np.arange(m.shape[1]), np.arange(m.shape[2]), indexing="ij"
    )
    coords = np.stack(grids, axis=-1).reshape(-1, 3).astype(np.int64)

    def nearest_sq(rows: np.ndarray) -> np.ndarray:
        out = np.full(len(coords), 10**9, dtype=np.int64)
        for r in rows:
            out = np.minimum(out, ((coords - r) ** 2).sum(axis=1))
        return out

    d_solid = nearest_sq(solid) if len(solid) else np.full(len(coords), 10**9)
    d_comp = nearest_sq(comp) if len(comp) else np.full(len(coords), 10**9)
    flat = m.reshape(-1)
    d2 = np.where(flat, d_comp, d_solid)
    phi = np.sqrt(d2.astype(np.float64)).astype(np.float32).reshape(m.shape)
    return torch.from_numpy(np.where(m, -phi, phi))


class TestSignedDistance:
    def test_exact_vs_brute_force_definition(self):
        g = torch.Generator().manual_seed(7)
        for trial in range(4):
            mask = torch.rand(10, 9, 8, generator=g) < (0.25 + 0.15 * trial)
            got = signed_distance_field(mask)
            want = brute_force_sdf(mask)
            assert torch.equal(got, want), f"trial {trial} SDF mismatch"

    def test_analytic_single_voxel(self):
        m = torch.zeros(9, 9, 9, dtype=torch.bool)
        m[4, 4, 4] = True
        phi = signed_distance_field(m)
        assert phi[4, 4, 4].item() == -1.0  # solid voxel, nearest complement
        assert phi[4, 4, 5].item() == 1.0  # outside, nearest solid voxel
        assert phi[5, 5, 5].item() == pytest.approx(3.0**0.5)  # diagonal
        assert phi[6, 4, 4].item() == 2.0
        assert phi[0, 0, 0].item() == pytest.approx((3 * 16.0) ** 0.5)

    def test_sign_convention_voxel_units(self):
        mask = blob_mask()
        phi = signed_distance_field(mask)
        assert (phi[mask] < 0).all() and (phi[~mask] > 0).all()
        # surface-adjacent voxels are exactly one voxel from the other phase
        assert phi[mask].max() == pytest.approx(-1.0)
        assert phi[~mask].min() == pytest.approx(1.0)

    def test_deterministic_and_chunk_invariant(self):
        mask = blob_mask()
        a = signed_distance_field(mask)
        b = signed_distance_field(mask)
        c = signed_distance_field(mask, v_chunk=7, t_chunk=13)
        assert torch.equal(a, b)
        assert torch.equal(a, c)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
    def test_cpu_cuda_bitwise_identical(self):
        mask = blob_mask()
        cpu = signed_distance_field(mask)
        gpu = signed_distance_field(mask.cuda()).cpu()
        assert torch.equal(cpu, gpu)

    def test_rejects_bad_mask(self):
        with pytest.raises(ValueError, match="must be"):
            signed_distance_field(torch.zeros(4, 4))
        with pytest.raises(TypeError, match="boolean"):
            signed_distance_field(torch.zeros(4, 4, 4))

    def test_degenerate_single_phase(self):
        empty = torch.zeros(6, 6, 6, dtype=torch.bool)
        phi = signed_distance_field(empty)
        # no solid anywhere -> outside distance saturates at the sentinel
        assert (phi > SDF_CLIP_VOXELS).all()


class TestSDFVolume:
    def test_shape_range_and_clip(self):
        v = sdf_volume(blob_mask())
        assert v.shape == (1, 1, 8, 8, 13)
        assert v.dtype == torch.float32
        assert -1.0 <= float(v.min()) and float(v.max()) <= 1.0
        assert float(v.abs().max()) == 1.0  # far field saturates the clip

    def test_deterministic(self):
        m = blob_mask()
        assert torch.equal(sdf_volume(m), sdf_volume(m))

    def test_even_translation_consistency_bitwise(self):
        """Stride-2 pooling commutes bitwise with even voxel shifts."""
        m = blob_mask()
        shifted = torch.zeros_like(m)
        shifted[2:, 2:, 2:] = m[:-2, :-2, :-2]
        a = sdf_volume(m)
        b = sdf_volume(shifted)
        assert torch.equal(a[..., :-1, :-1, :-1], b[..., 1:, 1:, 1:])

    def test_odd_shift_changes_sdf(self):
        """A one-voxel shift moves the surface — pooled SDF must change."""
        m = blob_mask()
        shifted = torch.zeros_like(m)
        shifted[1:, 1:, 1:] = m[:-1, :-1, :-1]
        assert not torch.equal(sdf_volume(m), sdf_volume(shifted))

    def test_invalid_args(self):
        m = blob_mask()
        with pytest.raises(ValueError, match="clip"):
            sdf_volume(m, clip=0.0)
        with pytest.raises(ValueError, match="pool"):
            sdf_volume(m, pool=0)


class TestSuboffMasks:
    def test_bare_hull_scale_invariance_bitwise(self):
        """SDF of the bare hull is bit-identical under any (sail, fin)."""
        ref = sdf_volume(suboff_mask("bare_hull", 1.0, 1.0))
        for s, f in ((0.4, 3.0), (2.5, 0.7), (1.9, 1.9)):
            other = sdf_volume(suboff_mask("bare_hull", s, f))
            assert torch.equal(ref, other), f"bare SDF changed at ({s}, {f})"

    def test_fin_noop_on_with_sail(self):
        base = sdf_volume(suboff_mask("with_sail", 1.3, 1.0))
        for f in (0.4, 2.2):
            other = sdf_volume(suboff_mask("with_sail", 1.3, f))
            assert torch.equal(base, other), f"fin leaked into with_sail SDF at f={f}"

    def test_appendages_change_sdf(self):
        bare = sdf_volume(suboff_mask("bare_hull", 1.0, 1.0))
        sail = sdf_volume(suboff_mask("with_sail", 1.0, 1.0))
        full = sdf_volume(suboff_mask("full", 1.0, 1.0))
        scaled = sdf_volume(suboff_mask("full", 2.0, 2.0))
        assert not torch.equal(bare, sail)
        assert not torch.equal(sail, full)
        assert not torch.equal(full, scaled)


class TestEncoder:
    def test_shapes_and_latent_bounds(self):
        enc = SDFEncoder(latent_dim=32)
        sdf = torch.rand(3, 1, 32, 32, 64) * 2 - 1
        z = enc(sdf)
        assert z.shape == (3, 32)
        assert float(z.abs().max()) <= 1.0

    def test_parameter_budget(self):
        enc = SDFEncoder(latent_dim=32, base=8)
        n = sum(p.numel() for p in enc.parameters())
        assert n == 224 + 3472 + 13856 + 27680 + 1056  # 46_288 at base=8
        assert n < 1_000_000

    def test_latent_bitwise_reproducible(self):
        enc = SDFEncoder(latent_dim=16)
        enc.eval()
        sdf = torch.rand(2, 1, 32, 32, 64) * 2 - 1
        with torch.no_grad():
            a = enc(sdf)
            b = enc(sdf)
        assert torch.equal(a, b)

    def test_same_geometry_same_latent(self):
        """Repeated forwards of one geometry give identical latents."""
        enc = SDFEncoder(latent_dim=8)
        enc.eval()
        sdf = sdf_volume(suboff_mask("full", 1.2, 0.8))
        with torch.no_grad():
            z = torch.cat([enc(sdf), enc(sdf), enc(sdf)], dim=0)
        assert torch.equal(z[0], z[1]) and torch.equal(z[1], z[2])

    def test_invalid_dims(self):
        with pytest.raises(ValueError, match=">= 1"):
            SDFEncoder(latent_dim=0)


class TestJointModel:
    def _model(self, **kw):
        return SDFCondFNODrag(
            in_ch=5, width=8, n_layers=2, modes=(4, 8), mlp_hidden=16, film_hidden=8, **kw
        )

    def test_forward_shapes_param_plus_latent(self):
        m = self._model(param_dim=4, latent_dim=8, aux_dim=0)
        x = torch.randn(2, 5, 64, 128)
        sdf = torch.rand(2, 1, 32, 32, 64)
        p = torch.randn(2, 4)
        y = m(x, sdf, p)
        assert y.shape == (2,)
        assert m.fno.cond_embed[0].in_features == 12

    def test_forward_with_aux(self):
        m = self._model(param_dim=4, latent_dim=8, aux_dim=8)
        x = torch.randn(2, 5, 64, 128)
        sdf = torch.rand(2, 1, 32, 32, 64)
        p = torch.randn(2, 4)
        y, aux = m(x, sdf, p, return_aux=True)
        assert y.shape == (2,) and aux.shape == (2, 8)

    def test_sdf_only_zero_param_columns(self):
        m = self._model(param_dim=0, hand_dim=0, latent_dim=8)
        x = torch.randn(2, 5, 64, 128)
        sdf = torch.rand(2, 1, 32, 32, 64)
        p = torch.zeros(2, 0)
        assert m(x, sdf, p).shape == (2,)

    def test_sdf_plus_hand(self):
        m = self._model(param_dim=4, hand_dim=4, latent_dim=8)
        x = torch.randn(2, 5, 64, 128)
        sdf = torch.rand(2, 1, 32, 32, 64)
        p = torch.randn(2, 8)
        assert m(x, sdf, p).shape == (2,)
        assert m.fno.cond_embed[0].in_features == 16

    def test_wrong_p_width_raises(self):
        m = self._model(param_dim=4, latent_dim=8)
        with pytest.raises(ValueError, match="columns"):
            m(torch.randn(2, 5, 64, 128), torch.rand(2, 1, 32, 32, 64), torch.randn(2, 3))

    def test_aux_requires_aux_dim(self):
        m = self._model(param_dim=4, latent_dim=8, aux_dim=0)
        with pytest.raises(RuntimeError, match="aux"):
            m(
                torch.randn(2, 5, 64, 128),
                torch.rand(2, 1, 32, 32, 64),
                torch.randn(2, 4),
                return_aux=True,
            )

    def test_backward_smoke(self):
        m = self._model(param_dim=4, latent_dim=8)
        x = torch.randn(2, 5, 64, 128)
        sdf = torch.rand(2, 1, 32, 32, 64)
        p = torch.randn(2, 4)
        m(x, sdf, p).square().mean().backward()
        grads = [q.grad for q in m.encoder.parameters()]
        assert all(g is not None for g in grads), "encoder must receive gradient"


class TestConditionBlock:
    def test_param_block_and_plus_hand(self):
        re = np.asarray([100.0, 200.0])
        u = np.asarray([0.1, 0.1])
        s = np.asarray([1.0, 2.0])
        f = np.asarray([1.0, 0.5])
        b = condition_sdf_params(re, u, s, f)
        assert b.shape == (2, 4)
        assert b[0, 0] == 2.0 and b[1, 2] == pytest.approx(0.3010299)
        geo = np.zeros((2, 4))
        bp = condition_sdf_params(re, u, s, f, geo)
        assert bp.shape == (2, 8)
        assert (bp[:, :4] == b).all() and (bp[:, 4:] == 0).all()

    def test_geometry_shape_validation(self):
        with pytest.raises(ValueError, match="must be"):
            condition_sdf_params(
                np.asarray([100.0]),
                np.asarray([0.1]),
                np.asarray([1.0]),
                np.asarray([1.0]),
                np.zeros((2, 4)),
            )
