"""Tests for the member-bundle / per-member ensemble path (``ckpt_bundle``).

Pins the three closed serving gaps (measured 2026-08-30 sanity run) against
tiny CPU models — latent width kept at the production 32, all other hidden
dims shrunk:

- GAP 1 — :func:`infer_member_arch` recovers the arch from bare shapes;
  the bundle roundtrip preserves the exact state-dict tensors and norm;
- GAP 2 — :func:`load_two_stage_member` implements the verified bare
  two-file pattern (strict load into the ``SupervisedSDFEncoder`` wrapper,
  bare body sd remapped onto ``TwoStageCondFNODrag.fno``) and the bundle
  and bare paths are bit-exact on the same weights;
- GAP 3 — :class:`PerMemberEnsembleBackend` equals per-member direct
  forwards, equals the BASE backend on the shared-encoder degenerate case
  (cond norms padded zeros/ones over the latent columns), and rejects
  norm blocks that try to z-score the latent.

The ``test_fix1_..``–``test_fix4_..`` block below pins the four friction
fixes from the 2026-08-31 production rehearsal
(``/nfs/wangxi/runs/ckpt_bundle_rehearsal_20260831``): inference-ready
frozen modules, the ``weights_only`` opt-out, ``save_member_bundle``
arch inference, and the batch-contract / param-count docs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from tensorlbm.ai.ckpt_bundle import (
    LoadedMember,
    PerMemberEnsembleBackend,
    infer_member_arch,
    load_member_bundle,
    load_two_stage_member,
    save_member_bundle,
)
from tensorlbm.ai.geom_encoder import SDFEncoderV2
from tensorlbm.ai.inference_service import (
    CondDragCheckpoint,
    ModelEnsembleBackend,
    save_checkpoint,
)
from tensorlbm.ai.sdf_two_stage import SupervisedSDFEncoder, TwoStageCondFNODrag

#: Production latent width (kept real); every other hidden dim is shrunk.
LATENT = 32
NY, NX = 16, 32
SDF_SHAPE = (8, 8, 16)

TINY_ARCH = {
    "encoder": dict(latent_dim=LATENT, base=4, in_ch=1, target_dim=3, probe_hidden=0),
    "body": dict(
        param_dim=2,
        latent_dim=LATENT,
        aux_dim=0,
        in_ch=5,
        width=8,
        n_layers=2,
        modes=[4, 8],
        mlp_hidden=16,
        film_hidden=12,
    ),
}


def build_member(seed: int, **body_over: Any) -> tuple[SupervisedSDFEncoder, TwoStageCondFNODrag]:
    """One tiny (stage-1 wrapper, two-stage model) member; deterministic."""
    body_kwargs = {
        k: v for k, v in TINY_ARCH["body"].items() if k not in ("param_dim", "latent_dim")
    }
    torch.manual_seed(seed)
    trunk = SDFEncoderV2(latent_dim=LATENT, base=TINY_ARCH["encoder"]["base"])
    sup = SupervisedSDFEncoder(trunk, target_dim=TINY_ARCH["encoder"]["target_dim"])
    full = TwoStageCondFNODrag(
        sup.encoder,
        param_dim=TINY_ARCH["body"]["param_dim"],
        latent_dim=LATENT,
        **{**body_kwargs, **body_over},
    )
    full.freeze_encoder()
    sup.eval()
    full.eval()
    return sup, full


def tiny_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    fields = rng.standard_normal((5, NY, NX)).astype(np.float32)
    sdf = rng.standard_normal((1, 1) + SDF_SHAPE).astype(np.float32)
    cond = rng.standard_normal((4, 2))
    return fields, sdf, cond


def tiny_norm() -> dict[str, Any]:
    rng = np.random.default_rng(11)
    return dict(
        ch_mean=rng.standard_normal(5),
        ch_std=np.abs(rng.standard_normal(5)) + 0.5,
        p_mean=rng.standard_normal(2),
        p_std=np.abs(rng.standard_normal(2)) + 0.5,
        y_mean=0.3,
        y_std=0.2,
    )


def _forward_batch(
    full: TwoStageCondFNODrag, fields: np.ndarray, sdf: np.ndarray, cond: np.ndarray
) -> np.ndarray:
    """Direct ``TwoStageCondFNODrag.forward`` on the raw (unnormalised) inputs."""
    x = torch.from_numpy(fields).unsqueeze(0).expand(cond.shape[0], -1, -1, -1)
    sdft = torch.from_numpy(sdf).expand(cond.shape[0], -1, -1, -1, -1)
    p = torch.from_numpy(cond.astype(np.float32))
    with torch.no_grad():
        return full(x, sdft, p).double().numpy()


def _descale(z: np.ndarray, norm: dict[str, Any]) -> np.ndarray:
    """The backends' de-scaling: ``10 ** (z * y_std + y_mean)`` (float64)."""
    return 10.0 ** (z * norm["y_std"] + norm["y_mean"])


# ---------------------------------------------------------------------------
# GAP 1 — arch inference + bundle roundtrip
# ---------------------------------------------------------------------------


def test_infer_member_arch_recovers_config() -> None:
    sup, full = build_member(0)
    arch = infer_member_arch(sup.state_dict(), full.fno.state_dict())
    assert arch == TINY_ARCH
    # the auxiliary head is recoverable too
    sup_aux, full_aux = build_member(1, aux_dim=2)
    arch_aux = infer_member_arch(sup_aux.state_dict(), full_aux.fno.state_dict())
    assert arch_aux["body"]["aux_dim"] == 2
    assert arch_aux["body"]["param_dim"] == 2


def test_bundle_roundtrip_exact_tensors(tmp_path: Any) -> None:
    sup, full = build_member(3)
    sd1, sd2 = sup.state_dict(), full.fno.state_dict()
    norm = tiny_norm()
    path = save_member_bundle(
        tmp_path / "member_s3.pt",
        sd1,
        sd2,
        arch_config=TINY_ARCH,
        norm_stats=norm,
        meta=dict(member="s3", arm="ts2"),
    )
    loaded = load_member_bundle(path)
    assert isinstance(loaded, LoadedMember)
    assert loaded.source == "bundle"
    assert loaded.meta["member"] == "s3"
    assert loaded.param_dim == 2 and loaded.latent_dim == LATENT
    assert loaded.model.encoder is loaded.stage1.encoder
    assert not loaded.model.training and not loaded.stage1.training
    assert loaded.model.encoder_frozen
    assert set(loaded.stage1.state_dict()) == set(sd1)
    for k, v in loaded.stage1.state_dict().items():
        assert torch.equal(v, sd1[k])
    assert set(loaded.model.fno.state_dict()) == set(sd2)
    for k, v in loaded.model.fno.state_dict().items():
        assert torch.equal(v, sd2[k])
    for k, v in norm.items():
        assert np.array_equal(loaded.norm[k], v)
    assert loaded.arch == TINY_ARCH


# ---------------------------------------------------------------------------
# GAP 2 — the verified bare two-file loader (.fno remap)
# ---------------------------------------------------------------------------


def test_load_two_stage_member_bare_legacy(tmp_path: Any) -> None:
    sup, full = build_member(5)
    sd1, sd2 = sup.state_dict(), full.fno.state_dict()
    p1, p2 = tmp_path / "stage1_s5.pt", tmp_path / "stage2_ts2_s5.pt"
    torch.save(sd1, p1)
    torch.save(sd2, p2)

    model = load_two_stage_member(p1, p2)
    assert isinstance(model, TwoStageCondFNODrag)
    assert not model.training and model.encoder_frozen
    # the GAP-2 remap: bare body keys landed on .fno (strict), trunk from sd1
    for k, v in model.fno.state_dict().items():
        assert torch.equal(v, sd2[k])
    for k in ("stem.0.weight", "head.weight"):
        assert torch.equal(model.encoder.state_dict()[k], sd1[f"encoder.{k}"])
    fields, sdf, cond = tiny_inputs()
    y = _forward_batch(model, fields, sdf, cond)
    assert y.shape == (4,)
    assert np.isfinite(y).all()

    # the explicit arch override rebuilds the same weights
    model_ovr = load_two_stage_member(p1, p2, arch_config=TINY_ARCH)
    assert torch.equal(model_ovr.fno.state_dict()["lift.weight"], sd2["lift.weight"])

    # strictness: a body sd with a dropped key must fail loudly, never partially
    torch.save({k: v for k, v in sd2.items() if k != "lift.bias"}, p2)
    with pytest.raises(RuntimeError):
        load_two_stage_member(p1, p2)


def test_bundle_and_bare_paths_bitexact(tmp_path: Any) -> None:
    sup, full = build_member(9)
    sd1, sd2 = sup.state_dict(), full.fno.state_dict()
    p1, p2 = tmp_path / "stage1_s9.pt", tmp_path / "stage2_ts2_s9.pt"
    torch.save(sd1, p1)
    torch.save(sd2, p2)
    via_bare = load_two_stage_member(p1, p2)
    via_bundle = load_member_bundle(
        save_member_bundle(
            tmp_path / "member_s9.pt", sd1, sd2, arch_config=TINY_ARCH, norm_stats=tiny_norm()
        )
    ).model
    fields, sdf, cond = tiny_inputs()
    y_direct = _forward_batch(full, fields, sdf, cond)
    assert np.array_equal(y_direct, _forward_batch(via_bare, fields, sdf, cond))
    assert np.array_equal(y_direct, _forward_batch(via_bundle, fields, sdf, cond))


# ---------------------------------------------------------------------------
# GAP 3 — PerMemberEnsembleBackend
# ---------------------------------------------------------------------------


def test_per_member_backend_matches_direct_forwards() -> None:
    members = [build_member(s) for s in (0, 1, 2)]
    norm = tiny_norm()
    be = PerMemberEnsembleBackend(members, norms=norm)
    assert be.n_members == 3
    assert be.cond_dim == 2
    assert be.kind == "per-member-model"
    assert be.member_labels() == ["m0", "m1", "m2"]

    fields, sdf, cond = tiny_inputs()
    got = be.predict(fields, sdf, cond)
    assert got.shape == (3, 4)

    # replicate the base-class aggregation exactly (same expressions)
    ch_m = torch.as_tensor(norm["ch_mean"], dtype=torch.float32).view(1, -1, 1, 1)
    ch_s = torch.as_tensor(norm["ch_std"], dtype=torch.float32).view(1, -1, 1, 1)
    p_m = torch.as_tensor(norm["p_mean"], dtype=torch.float32)
    p_s = torch.as_tensor(norm["p_std"], dtype=torch.float32)
    x = torch.from_numpy(fields).unsqueeze(0).expand(4, -1, -1, -1)
    sdft = torch.from_numpy(sdf).expand(4, -1, -1, -1, -1)
    p = torch.from_numpy(cond.astype(np.float32))
    with torch.no_grad():
        expected = np.stack(
            [
                _descale(full((x - ch_m) / ch_s, sdft, (p - p_m) / p_s).double().numpy(), norm)
                for _, full in members
            ],
            axis=0,
        )
    assert np.array_equal(got, expected)


def test_shared_encoder_degenerate_equals_base_backend() -> None:
    """One trunk shared by all bodies == base backend on padded cond rows.

    This is the verified adapter of the sanity run, now pinned as a test:
    serving the bare body through the base backend with cond rows
    ``[p | z]`` and the cond-vector norms padded with zeros/ones over the
    latent columns is BIT-IDENTICAL to the per-member pair path — at
    matched batch composition (float32 conv kernels may differ in the last
    ulp between a batch-1 and a batch-N latent pass, so the reference
    latent here is computed on the same batch the pair path uses).
    """
    torch.manual_seed(21)
    trunk = SDFEncoderV2(latent_dim=LATENT, base=TINY_ARCH["encoder"]["base"])
    body_kwargs = {
        k: v for k, v in TINY_ARCH["body"].items() if k not in ("param_dim", "latent_dim")
    }
    members = []
    for s in (31, 32, 33):
        torch.manual_seed(s)
        sup = SupervisedSDFEncoder(trunk, target_dim=TINY_ARCH["encoder"]["target_dim"])
        full = TwoStageCondFNODrag(
            trunk, param_dim=TINY_ARCH["body"]["param_dim"], latent_dim=LATENT, **body_kwargs
        )
        full.freeze_encoder()
        sup.eval()
        full.eval()
        members.append((sup, full))

    norm = tiny_norm()
    fields, sdf, cond = tiny_inputs()
    be_pm = PerMemberEnsembleBackend(members, norms=norm)
    got = be_pm.predict(fields, sdf, cond)

    ckpts = []
    for i, (_, full) in enumerate(members):
        padded = dict(norm)
        padded["p_mean"] = np.concatenate([norm["p_mean"], np.zeros(LATENT)])
        padded["p_std"] = np.concatenate([norm["p_std"], np.ones(LATENT)])
        ckpts.append(
            CondDragCheckpoint(
                arch=dict(
                    in_ch=TINY_ARCH["body"]["in_ch"],
                    width=TINY_ARCH["body"]["width"],
                    n_layers=TINY_ARCH["body"]["n_layers"],
                    modes=tuple(TINY_ARCH["body"]["modes"]),
                    cond_dim=TINY_ARCH["body"]["param_dim"] + LATENT,
                    mlp_hidden=TINY_ARCH["body"]["mlp_hidden"],
                    film_hidden=TINY_ARCH["body"]["film_hidden"],
                    aux_dim=TINY_ARCH["body"]["aux_dim"],
                ),
                state_dict=full.fno.state_dict(),
                norm=padded,
                meta=dict(member=f"m{i}"),
            )
        )
    be_base = ModelEnsembleBackend(ckpts)
    # reference latent on the SAME batch composition the pair path uses
    sdft = torch.from_numpy(sdf).expand(cond.shape[0], -1, -1, -1, -1)
    with torch.no_grad():
        z = trunk(sdft).numpy()  # (N, LATENT), raw tanh latent
    cond_padded = np.concatenate([cond, z], axis=1)
    assert np.array_equal(got, be_base.predict(fields, cond_padded))

    # the sanity-run adapter shape — a single-row query — is bit-exact too
    with torch.no_grad():
        z1 = trunk(torch.from_numpy(sdf)).numpy()
    cond_one = np.concatenate([cond[:1], z1], axis=1)
    assert np.array_equal(be_pm.predict(fields, sdf, cond[:1]), be_base.predict(fields, cond_one))


def test_per_member_backend_norm_contract() -> None:
    members = [build_member(0)]
    norm = tiny_norm()
    # padded cond-vector norms (the BASE adapter contract) are rejected here:
    # the latent columns must be served raw
    padded = dict(norm, p_mean=np.concatenate([norm["p_mean"], np.zeros(LATENT)]))
    with pytest.raises(ValueError, match="param_dim"):
        PerMemberEnsembleBackend(members, norms=padded)
    with pytest.raises(ValueError, match="missing keys"):
        PerMemberEnsembleBackend(members, norms={"ch_mean": np.zeros(5)})
    with pytest.raises(ValueError, match="norms is required"):
        PerMemberEnsembleBackend(members, norms=None)
    with pytest.raises(ValueError, match="norm blocks"):
        PerMemberEnsembleBackend([build_member(1)], norms=[norm, norm])
    # a pair whose encoder is NOT the body's trunk is rejected loudly
    sup_a, full_a = build_member(0)
    sup_b, _ = build_member(1)
    with pytest.raises(ValueError, match="trunk"):
        PerMemberEnsembleBackend([(sup_b, full_a)], norms=norm)
    with pytest.raises(ValueError, match="(encoder, body)"):
        PerMemberEnsembleBackend([full_a], norms=norm)


def test_from_bundles_labels_and_predict(tmp_path: Any) -> None:
    sup, full = build_member(2)
    path = save_member_bundle(
        tmp_path / "m2.pt",
        sup.state_dict(),
        full.fno.state_dict(),
        arch_config=TINY_ARCH,
        norm_stats=tiny_norm(),
        meta=dict(member="s2"),
    )
    be = PerMemberEnsembleBackend.from_bundles([load_member_bundle(path)])
    assert be.member_labels() == ["s2"]
    fields, sdf, cond = tiny_inputs()
    got = be.predict(fields, sdf, cond)
    assert got.shape == (1, 4)
    # per-member equivalence to the direct forward on the normalised inputs
    n = tiny_norm()
    x = torch.from_numpy(fields).unsqueeze(0).expand(4, -1, -1, -1)
    x = (
        x - torch.as_tensor(n["ch_mean"], dtype=torch.float32).view(1, -1, 1, 1)
    ) / torch.as_tensor(n["ch_std"], dtype=torch.float32).view(1, -1, 1, 1)
    p = torch.from_numpy(cond.astype(np.float32))
    p = (p - torch.as_tensor(n["p_mean"], dtype=torch.float32)) / torch.as_tensor(
        n["p_std"], dtype=torch.float32
    )
    sdft = torch.from_numpy(sdf).expand(4, -1, -1, -1, -1)
    with torch.no_grad():
        expected = _descale(full(x, sdft, p).double().numpy(), n)
    assert np.array_equal(got[0], expected)


def test_predict_batch_matches_predict() -> None:
    members = [build_member(s) for s in (4, 5)]
    be = PerMemberEnsembleBackend(members, norms=tiny_norm())
    fields, sdf, cond = tiny_inputs()
    single = be.predict(fields, sdf, cond)
    # G == 1 reproduces predict bit-identically
    assert np.array_equal(single, be.predict_batch(fields[None], sdf[None], cond, counts=[4]))
    # G == 2: rows match the per-geometry predict calls to float32
    # batch-kernel noise (same caveat as the base predict_batch, ~1e-8 rel)
    fields2 = np.stack([fields, (fields * 0.5).astype(np.float32)])
    sdf2 = np.concatenate([sdf, (sdf * 0.5).astype(np.float32)])
    cond2 = np.concatenate([cond[:2], cond[2:]])
    out2 = be.predict_batch(fields2, sdf2, cond2, counts=[2, 2])
    np.testing.assert_allclose(out2[:, :2], be.predict(fields2[0], sdf2[0], cond2[:2]), rtol=1e-6)
    np.testing.assert_allclose(out2[:, 2:], be.predict(fields2[1], sdf2[1], cond2[2:]), rtol=1e-6)


# ---------------------------------------------------------------------------
# sniffing branches of load_member_bundle
# ---------------------------------------------------------------------------


def test_load_member_bundle_sniffing(tmp_path: Any) -> None:
    sup, full = build_member(0)
    sd1, sd2 = sup.state_dict(), full.fno.state_dict()
    # single-model CondDragCheckpoint file -> pointed at the existing loader
    save_checkpoint(
        CondDragCheckpoint(
            arch=dict(
                in_ch=5,
                width=8,
                n_layers=2,
                modes=(4, 8),
                cond_dim=2 + LATENT,
                mlp_hidden=16,
                film_hidden=12,
            ),
            state_dict=sd2,
            norm=tiny_norm(),
        ),
        str(tmp_path / "legacy_ckpt.pt"),
    )
    with pytest.raises(ValueError, match="load_checkpoint"):
        load_member_bundle(tmp_path / "legacy_ckpt.pt")
    # bare stage-1 alone -> tells the caller to pass it as stage1_path
    torch.save(sd1, tmp_path / "stage1_s0.pt")
    with pytest.raises(ValueError, match="stage1_path"):
        load_member_bundle(tmp_path / "stage1_s0.pt")
    # bare stage-2 without its stage-1 half
    torch.save(sd2, tmp_path / "stage2_ts2_s0.pt")
    with pytest.raises(ValueError, match="stage1_path"):
        load_member_bundle(tmp_path / "stage2_ts2_s0.pt")
    # bare pair through the sniffing loader
    loaded = load_member_bundle(
        tmp_path / "stage2_ts2_s0.pt",
        stage1_path=tmp_path / "stage1_s0.pt",
        norm_stats=tiny_norm(),
    )
    assert loaded.source == "bare-pair"
    assert loaded.param_dim == 2 and len(loaded.norm) == 6
    for k, v in loaded.model.fno.state_dict().items():
        assert torch.equal(v, sd2[k])
    # and a non-state-dict payload
    torch.save([1, 2, 3], tmp_path / "junk.pt")
    with pytest.raises(ValueError, match="not a member bundle"):
        load_member_bundle(tmp_path / "junk.pt")


# ---------------------------------------------------------------------------
# 2026-08-31 rehearsal friction fixes (register items 1-4)
# ---------------------------------------------------------------------------


def test_fix1_loaders_return_inference_ready_modules(tmp_path: Any) -> None:
    """Every loader path hands out eval modules with ALL params frozen."""
    sup, full = build_member(7)
    sd1, sd2 = sup.state_dict(), full.fno.state_dict()
    p1, p2 = tmp_path / "stage1_s7.pt", tmp_path / "stage2_ts2_s7.pt"
    torch.save(sd1, p1)
    torch.save(sd2, p2)

    # bare two-file loader: the returned model is fully frozen
    model = load_two_stage_member(p1, p2)
    assert all(not p.requires_grad for p in model.parameters())
    assert not model.training

    # bundle loader: BOTH the full model and the encoder wrapper
    path = save_member_bundle(
        tmp_path / "m7.pt", sd1, sd2, arch_config=TINY_ARCH, norm_stats=tiny_norm()
    )
    loaded = load_member_bundle(path)
    assert all(not p.requires_grad for p in loaded.model.parameters())
    assert all(not p.requires_grad for p in loaded.stage1.parameters())
    assert not loaded.model.training and not loaded.stage1.training

    # bare-pair sniffing branch of the bundle loader too
    loaded_bare = load_member_bundle(p2, stage1_path=p1, norm_stats=tiny_norm())
    assert loaded_bare.source == "bare-pair"
    assert all(not p.requires_grad for p in loaded_bare.model.parameters())
    assert all(not p.requires_grad for p in loaded_bare.stage1.parameters())

    # frozen params change no values: forward still runs and is bit-exact
    fields, sdf, cond = tiny_inputs()
    y = _forward_batch(loaded.model, fields, sdf, cond)
    assert y.shape == (4,)
    assert np.isfinite(y).all()
    assert np.array_equal(y, _forward_batch(full, fields, sdf, cond))
    # and a frozen member still serves through the per-member backend
    got = PerMemberEnsembleBackend.from_bundles([loaded]).predict(fields, sdf, cond)
    assert got.shape == (1, 4)
    assert np.isfinite(got).all()


def test_fix2_weights_only_opt_out(tmp_path: Any) -> None:
    """Default loads the numpy-norm bundle; True is an explicit opt-in."""
    sup, full = build_member(8)
    sd1, sd2 = sup.state_dict(), full.fno.state_dict()
    norm = tiny_norm()
    path = save_member_bundle(tmp_path / "np_norm.pt", sd1, sd2, norm_stats=norm)

    # default (weights_only=False) loads the standard numpy-norm bundle fine
    loaded = load_member_bundle(path)
    assert loaded.source == "bundle"
    for k, v in norm.items():
        assert np.array_equal(loaded.norm[k], v)

    # weights_only=True refuses the embedded numpy norm arrays
    with pytest.raises(Exception) as excinfo:
        load_member_bundle(path, weights_only=True)
    assert "weights_only" in str(excinfo.value).lower()

    # the documented contract states WHY the default is False
    doc = load_member_bundle.__doc__ or ""
    assert "weights_only" in doc
    assert "numpy" in doc

    # a pure-tensor-norm bundle of the same member DOES load under True
    payload: Any = torch.load(path, map_location="cpu", weights_only=False)
    payload["norm"] = {k: torch.as_tensor(np.asarray(v)) for k, v in payload["norm"].items()}
    pure = str(tmp_path / "tensor_norm.pt")
    torch.save(payload, pure)
    strict = load_member_bundle(pure, weights_only=True)
    assert strict.source == "bundle"
    for k, v in norm.items():
        assert np.array_equal(strict.norm[k], v)


def test_fix3_save_member_bundle_infers_arch(tmp_path: Any) -> None:
    """arch_config=None infers the arch; identical to an explicit save."""
    sup, full = build_member(11)
    sd1, sd2 = sup.state_dict(), full.fno.state_dict()
    norm = tiny_norm()
    p_infer = save_member_bundle(
        tmp_path / "inferred.pt", sd1, sd2, arch_config=None, norm_stats=norm
    )
    p_explicit = save_member_bundle(
        tmp_path / "explicit.pt", sd1, sd2, arch_config=TINY_ARCH, norm_stats=norm
    )
    inferred = load_member_bundle(p_infer)
    explicit = load_member_bundle(p_explicit)
    assert inferred.arch == explicit.arch == TINY_ARCH
    # same member rebuilt either way: identical tensors on both halves
    for k, v in inferred.model.fno.state_dict().items():
        assert torch.equal(v, explicit.model.fno.state_dict()[k])
    for k, v in inferred.stage1.state_dict().items():
        assert torch.equal(v, explicit.stage1.state_dict()[k])


def test_fix4_batch_contract_and_param_counts_documented() -> None:
    """Docs pin the batch contract and the arm-dependent body param counts."""

    def flat(doc: str) -> str:
        return " ".join(doc.split())

    predict_doc = flat(PerMemberEnsembleBackend.predict.__doc__ or "")
    batch_doc = flat(PerMemberEnsembleBackend.predict_batch.__doc__ or "")
    # predict: N cond rows against ONE shared field/sdf; predict_batch: per-row
    assert "ONE shared field/sdf" in predict_doc
    assert "ONE shared field/sdf" in batch_doc
    assert "counts" in batch_doc
    assert "predict_batch" in predict_doc
    # the rehearsal cost note lives on the single-design path
    assert "1e-5" in predict_doc
    # ts2 is not the universal body size
    infer_doc = flat(infer_member_arch.__doc__ or "")
    assert "4,240,073" in infer_doc
    assert "4,240,713" in infer_doc
    assert "cond_dim" in infer_doc
