"""Tests for the bundle-pool serving loader (``discover_bundles`` / ``load_bundle_pool``).

POSTMERGE_RUNBOOK step 3 (``/nfs/wangxi/runs/ckpt_bundle_rehearsal_20260831/
POSTMERGE_RUNBOOK.md``) makes bundles the recommended serving load path;
``load_bundle_pool(dir, arm="ts2")`` is that path.  These tests pin the
discovery, arm-filter and failure contracts on tiny CPU bundles built as
REAL save+load round-trips — ``save_member_bundle`` with synthetic state
dicts, the arch inferred from the bare shapes (``arch_config=None``) and
the six-key norm stats — mirroring the fixture conventions of
``tests/test_ckpt_bundle.py``.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest
import torch

from tensorlbm.ai.ckpt_bundle import (
    BundleDiscoveryWarning,
    discover_bundles,
    load_bundle_pool,
    save_member_bundle,
)
from tensorlbm.ai.geom_encoder import SDFEncoderV2
from tensorlbm.ai.sdf_two_stage import SupervisedSDFEncoder, TwoStageCondFNODrag

#: Production latent width (kept real); every other hidden dim is shrunk.
LATENT = 32
#: Serving arms keyed by their arch.body.param_dim (the detection key).
ARMS = {"ts2": 2, "ts4": 4}
TINY_BODY = dict(in_ch=5, width=8, n_layers=2, modes=[4, 8], mlp_hidden=16, film_hidden=12)


def build_member(seed: int, param_dim: int) -> tuple[SupervisedSDFEncoder, TwoStageCondFNODrag]:
    """One tiny (stage-1 wrapper, two-stage model) member; deterministic."""
    torch.manual_seed(seed)
    trunk = SDFEncoderV2(latent_dim=LATENT, base=4)
    sup = SupervisedSDFEncoder(trunk, target_dim=3)
    full = TwoStageCondFNODrag(sup.encoder, param_dim=param_dim, latent_dim=LATENT, **TINY_BODY)
    full.freeze_encoder()
    sup.eval()
    full.eval()
    return sup, full


def tiny_norm(param_dim: int) -> dict[str, Any]:
    rng = np.random.default_rng(11 + param_dim)
    return dict(
        ch_mean=rng.standard_normal(5),
        ch_std=np.abs(rng.standard_normal(5)) + 0.5,
        p_mean=rng.standard_normal(param_dim),
        p_std=np.abs(rng.standard_normal(param_dim)) + 0.5,
        y_mean=0.3,
        y_std=0.2,
    )


def write_pool(
    dirpath: Any,
    arms: tuple[str, ...] = ("ts2", "ts4"),
    seeds: tuple[int, ...] = (0, 1),
    *,
    junk: bool = False,
    meta: bool = True,
) -> dict[str, tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]]:
    """Save real tiny bundles; returns ``{name: (sd1, sd2, norm)}`` per member.

    ``junk`` additionally drops a non-bundle ``.pt`` and a ``.txt`` into the
    directory; ``meta=False`` writes bundles with an EMPTY meta block (the
    member-label fallback path).
    """
    saved: dict[str, tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]] = {}
    for arm in arms:
        for seed in seeds:
            name = f"{arm}_s{seed}"
            sup, full = build_member(seed + 100 * ARMS[arm], ARMS[arm])
            sd1, sd2 = sup.state_dict(), full.fno.state_dict()
            norm = tiny_norm(ARMS[arm])
            save_member_bundle(
                dirpath / f"{name}.pt",
                sd1,
                sd2,
                arch_config=None,
                norm_stats=norm,
                meta=dict(member=name, arm=arm, seed=seed) if meta else None,
            )
            saved[name] = (sd1, sd2, norm)
    if junk:
        torch.save([1, 2, 3], dirpath / "junk.pt")
    (dirpath / "README.txt").write_text("not a checkpoint\n")
    return saved


# ---------------------------------------------------------------------------
# discover_bundles
# ---------------------------------------------------------------------------


def test_discover_mixed_dir_skips_junk_with_warning(tmp_path: Any) -> None:
    d = tmp_path / "pool"
    d.mkdir()
    write_pool(d, junk=True)

    with pytest.warns(BundleDiscoveryWarning) as rec:
        desc = discover_bundles(d)
    assert [x["file"] for x in desc] == [
        "ts2_s0.pt",
        "ts2_s1.pt",
        "ts4_s0.pt",
        "ts4_s1.pt",
    ]
    # exactly one collected warning, for the junk file, naming the reason
    junk_w = [w for w in rec if "junk.pt" in str(w.message)]
    assert len(junk_w) == 1
    assert "not a member bundle" in str(junk_w[0].message)
    # non-.pt entries are never inspected -> never warned about
    assert not any("README" in str(w.message) for w in rec)
    by_file = {x["file"]: x for x in desc}
    # arm detection key: arch.body.param_dim (2 -> ts2, 4 -> ts4)
    assert by_file["ts2_s0.pt"]["arm"] == "ts2"
    assert by_file["ts2_s0.pt"]["param_dim"] == 2
    assert by_file["ts4_s1.pt"]["arm"] == "ts4"
    assert by_file["ts4_s1.pt"]["param_dim"] == 4
    # member label from meta; path resolved for the loader
    assert by_file["ts2_s0.pt"]["member"] == "ts2_s0"
    assert by_file["ts2_s0.pt"]["path"] == str((d / "ts2_s0.pt").resolve())


def test_discover_member_label_falls_back_to_stem(tmp_path: Any) -> None:
    d = tmp_path / "nometa"
    d.mkdir()
    write_pool(d, arms=("ts2",), seeds=(0,), meta=False)
    (desc,) = discover_bundles(d)
    assert desc["member"] == "ts2_s0"  # file stem, meta block empty
    assert desc["arm"] == "ts2"


def test_discover_accepts_explicit_file_list(tmp_path: Any) -> None:
    d = tmp_path / "pool"
    d.mkdir()
    write_pool(d, junk=True)
    with pytest.warns(BundleDiscoveryWarning, match="junk.pt"):
        desc = discover_bundles([d / "ts4_s0.pt", d / "junk.pt", d / "ts2_s1.pt"])
    # explicit list comes back sorted by filename, junk warned and skipped
    assert [x["file"] for x in desc] == ["ts2_s1.pt", "ts4_s0.pt"]
    # a named-but-missing file is a hard error, not a skip
    with pytest.raises(FileNotFoundError):
        discover_bundles([d / "ts2_s1.pt", d / "missing.pt"])
    # so is a scan root that does not exist
    with pytest.raises(FileNotFoundError):
        discover_bundles(tmp_path / "no_such_dir")


# ---------------------------------------------------------------------------
# load_bundle_pool
# ---------------------------------------------------------------------------


def test_load_bundle_pool_ts2_roundtrip(tmp_path: Any) -> None:
    d = tmp_path / "pool"
    d.mkdir()
    saved = write_pool(d, junk=True)

    with pytest.warns(BundleDiscoveryWarning, match="junk.pt"):
        members = load_bundle_pool(d, arm="ts2")
    assert len(members) == 2
    # stable sorted-by-filename order
    assert [m.meta["member"] for m in members] == ["ts2_s0", "ts2_s1"]
    for m, name in zip(members, ("ts2_s0", "ts2_s1")):
        sd1, sd2, norm = saved[name]
        assert m.source == "bundle"
        assert m.param_dim == 2 and m.latent_dim == LATENT
        # the loader contract: eval + EVERY parameter frozen, trunk shared
        assert not m.model.training and not m.stage1.training
        assert all(not p.requires_grad for p in m.model.parameters())
        assert all(not p.requires_grad for p in m.stage1.parameters())
        assert m.model.encoder is m.stage1.encoder
        # real round-trip: weights and the six-key norm sidecar bit-exact
        for k, v in m.stage1.state_dict().items():
            assert torch.equal(v, sd1[k])
        for k, v in m.model.fno.state_dict().items():
            assert torch.equal(v, sd2[k])
        for k, v in norm.items():
            assert np.array_equal(m.norm[k], v)

    # the other arm filters to its own half of the same directory
    with pytest.warns(BundleDiscoveryWarning, match="junk.pt"):
        members_ts4 = load_bundle_pool(d, arm="ts4")
    assert [m.meta["member"] for m in members_ts4] == ["ts4_s0", "ts4_s1"]
    assert all(m.param_dim == 4 for m in members_ts4)


def test_load_bundle_pool_wrong_arm_names_dir_and_arm(tmp_path: Any) -> None:
    d = tmp_path / "ts2only"
    d.mkdir()
    write_pool(d, arms=("ts2",))
    with pytest.raises(ValueError, match=re.escape(str(d))) as ei:
        load_bundle_pool(d, arm="ts4")
    msg = str(ei.value)
    assert "ts4" in msg
    assert "ts2 x 2" in msg  # what was found instead, with counts


def test_load_bundle_pool_empty_dir_clear_error(tmp_path: Any) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match=re.escape(str(empty))) as ei:
        load_bundle_pool(empty, arm="ts2")
    msg = str(ei.value)
    assert "ts2" in msg and "no bundles at all" in msg

    # a directory whose only .pt is junk fails the same way (after warning)
    junkonly = tmp_path / "junkonly"
    junkonly.mkdir()
    torch.save({"some": "dict"}, junkonly / "stray.pt")
    with pytest.warns(BundleDiscoveryWarning, match="stray.pt"):
        with pytest.raises(ValueError, match="no bundles at all"):
            load_bundle_pool(junkonly, arm="ts2")
