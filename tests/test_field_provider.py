"""Tests for ``tensorlbm.ai.field_provider`` (retrieval-based reference fields).

Pins the v1 contract against small pure-numpy fixtures (no torch / GPU):

- retrieval correctness: ``sdf_near`` / ``cond_near`` pick the argmin
  distance donor and populate the distance honestly;
- the strategy fallback chain, plus precise ``ValueError`` messages when
  an explicit strategy lacks its inputs;
- the mean fallback (``donor_index=None``, guard trivially satisfied);
- the in-manifold guard trips on an out-of-pool donor;
- the API contract: ``BorrowedField`` has NO SDF attribute at all — the
  x57.9 donor-SDF mistake is unrepresentable;
- determinism (identical inputs -> identical donor);
- ``FieldProvider.from_corpus`` against a synthetic production-layout
  directory, a synthetic ``.npz`` snapshot, and (smoke, skipif absent)
  the real 406-row production corpus.

Evidence base: ``/nfs/wangxi/runs/l2_field_sensitivity_20260904``.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np
import pytest

from tensorlbm.ai.field_provider import (
    FIELD_CHANNELS,
    STRATEGIES,
    BorrowedField,
    FieldProvider,
)

#: Real production corpus files (smoke-skipped where absent, e.g. CI).
_PROD = {
    "fam": "/nfs/wangxi/runs/b4_fam_20260824/cache_fam.npz",
    "ext": "/nfs/wangxi/runs/sdf_slender_20260828/cache_ext56.npz",
    "sdf_fam": "/nfs/wangxi/runs/b4_sdf2_20260825/sdf_fam350.npz",
    "sdf_ext": "/nfs/wangxi/runs/sdf_slender_20260828/sdf_ext2.npz",
}
_PROD_MISSING = [p for p in _PROD.values() if not os.path.isfile(p)]

_SDF_SHAPE = (3, 4, 4)


def make_pool(n: int = 3, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Small deterministic pool: (n, 5, 4, 4) fields, (n, 3, 4, 4) SDFs, (n, 4) cond."""
    rng = np.random.default_rng(seed)
    fields = rng.standard_normal((n, FIELD_CHANNELS, 4, 4)).astype(np.float32)
    sdfs = rng.standard_normal((n, *_SDF_SHAPE)).astype(np.float32)
    cond = rng.standard_normal((n, 4))
    return fields, sdfs, cond


class TestRetrievalCorrectness:
    def test_sdf_near_picks_argmin_and_reports_distance(self) -> None:
        fields, sdfs, cond = make_pool(n=4)
        provider = FieldProvider(fields, pool_sdfs=sdfs, pool_cond=cond)
        target = sdfs[2] + np.float32(1e-3)
        got = provider.borrow(target_sdf=target)
        assert got.strategy == "sdf_near"
        assert got.donor_index == 2
        expected = float(np.linalg.norm((sdfs[2].astype(np.float64) - target).ravel()))
        assert got.distance == pytest.approx(expected, rel=1e-12)

    def test_sdf_near_exact_copy_gives_zero_distance(self) -> None:
        fields, sdfs, cond = make_pool(n=3)
        got = FieldProvider(fields, pool_sdfs=sdfs, pool_cond=cond).borrow(
            target_sdf=sdfs[1].copy()
        )
        assert got.donor_index == 1
        assert got.distance == pytest.approx(0.0, abs=1e-12)

    def test_cond_near_picks_argmin(self) -> None:
        fields, sdfs, cond = make_pool(n=5)
        provider = FieldProvider(fields, pool_sdfs=sdfs, pool_cond=cond)
        target = cond[3] + 1e-4
        got = provider.borrow(target_cond=target)
        assert got.strategy == "cond_near"
        assert got.donor_index == 3
        expected = float(np.linalg.norm(cond.astype(np.float64) - target, axis=1).min())
        assert got.distance == pytest.approx(expected, rel=1e-12)

    def test_borrowed_fields_are_a_copy_not_a_view(self) -> None:
        fields, sdfs, _ = make_pool(n=3)
        got = FieldProvider(fields, pool_sdfs=sdfs).borrow(target_sdf=sdfs[0])
        got.fields[...] = 0.0
        assert not np.array_equal(fields[0], np.zeros_like(fields[0]))


class TestStrategyResolution:
    def test_default_prefers_sdf_near_when_possible(self) -> None:
        fields, sdfs, cond = make_pool()
        got = FieldProvider(fields, pool_sdfs=sdfs, pool_cond=cond).borrow(
            target_sdf=sdfs[0], target_cond=cond[0]
        )
        assert got.strategy == "sdf_near"

    def test_default_falls_back_to_cond_near(self) -> None:
        fields, _, cond = make_pool()
        got = FieldProvider(fields, pool_cond=cond).borrow(target_cond=cond[1])
        assert got.strategy == "cond_near"

    def test_default_falls_back_to_cond_when_target_sdf_missing(self) -> None:
        fields, sdfs, cond = make_pool()
        got = FieldProvider(fields, pool_sdfs=sdfs, pool_cond=cond).borrow(target_cond=cond[1])
        assert got.strategy == "cond_near"

    def test_default_falls_back_to_mean(self) -> None:
        fields, _, _ = make_pool()
        got = FieldProvider(fields).borrow()
        assert got.strategy == "mean"

    def test_explicit_sdf_near_without_pool_sdfs(self) -> None:
        fields, _, cond = make_pool()
        with pytest.raises(ValueError, match="requires pool_sdfs"):
            FieldProvider(fields, pool_cond=cond).borrow(
                target_sdf=np.zeros(_SDF_SHAPE, np.float32), strategy="sdf_near"
            )

    def test_explicit_sdf_near_without_target_sdf(self) -> None:
        fields, sdfs, _ = make_pool()
        with pytest.raises(ValueError, match="requires target_sdf"):
            FieldProvider(fields, pool_sdfs=sdfs).borrow(strategy="sdf_near")

    def test_explicit_cond_near_without_pool_cond(self) -> None:
        fields, sdfs, _ = make_pool()
        with pytest.raises(ValueError, match="requires pool_cond"):
            FieldProvider(fields, pool_sdfs=sdfs).borrow(
                target_cond=np.zeros(4), strategy="cond_near"
            )

    def test_explicit_cond_near_without_target_cond(self) -> None:
        fields, _, cond = make_pool()
        with pytest.raises(ValueError, match="requires target_cond"):
            FieldProvider(fields, pool_cond=cond).borrow(strategy="cond_near")

    def test_unknown_strategy(self) -> None:
        fields, _, _ = make_pool()
        with pytest.raises(ValueError, match="unknown strategy 'sdf'"):
            FieldProvider(fields).borrow(strategy="sdf")

    def test_explicit_mean_never_needs_inputs(self) -> None:
        fields, _, _ = make_pool()
        assert FieldProvider(fields).borrow(strategy="mean").strategy == "mean"


class TestMeanFallback:
    def test_mean_fields_are_the_pool_mean(self) -> None:
        fields, _, _ = make_pool(n=5)
        got = FieldProvider(fields).borrow()
        assert got.strategy == "mean"
        assert got.donor_index is None
        assert got.distance is None
        expected = np.mean(fields, axis=0, dtype=np.float64).astype(fields.dtype)
        np.testing.assert_array_equal(got.fields, expected)

    def test_mean_guard_trivially_ok(self) -> None:
        fields, _, _ = make_pool()
        got = FieldProvider(fields).borrow()
        assert got.guard_rel_l2 == 0.0
        assert got.guard_ok is True
        assert got.provenance["donor_key"] is None
        assert got.provenance["pool_size"] == fields.shape[0]


class TestGuard:
    @staticmethod
    def _outlier_pool() -> tuple[np.ndarray, np.ndarray]:
        """Rows 0/1 near each other, row 2 at rel-L2 1.0 from the pool mean."""
        base = np.ones((3, FIELD_CHANNELS, 4, 4), dtype=np.float32)
        base[1] *= 1.02
        base[2] *= 4.0  # mean = ~2.01 -> ||row2 - mean|| / ||mean|| ~= 0.99
        sdfs = np.arange(3 * 3 * 4 * 4, dtype=np.float64).reshape(3, *_SDF_SHAPE)
        return base, sdfs

    def test_guard_trips_on_out_of_pool_donor(self) -> None:
        fields, sdfs = self._outlier_pool()
        target = sdfs[2] + 0.5
        got = FieldProvider(fields, pool_sdfs=sdfs, guard_threshold=0.15).borrow(target_sdf=target)
        assert got.donor_index == 2
        assert got.guard_rel_l2 == pytest.approx(0.99, abs=0.02)
        assert got.guard_ok is False

    def test_guard_passes_when_threshold_loosened(self) -> None:
        fields, sdfs = self._outlier_pool()
        got = FieldProvider(fields, pool_sdfs=sdfs, guard_threshold=2.0).borrow(
            target_sdf=sdfs[2] + 0.5
        )
        assert got.guard_ok is True
        assert got.guard_threshold == 2.0

    def test_guard_threshold_is_a_constructor_arg(self) -> None:
        fields, _, _ = make_pool()
        with pytest.raises(ValueError, match="guard_threshold must be > 0"):
            FieldProvider(fields, guard_threshold=0.0)

    def test_guard_is_measured_against_pool_mean(self) -> None:
        fields, sdfs = self._outlier_pool()
        got = FieldProvider(fields, pool_sdfs=sdfs, guard_threshold=1e9).borrow(target_sdf=sdfs[0])
        mean = np.mean(fields, axis=0, dtype=np.float64)
        expected = float(np.linalg.norm((fields[0].astype(np.float64) - mean).ravel())) / float(
            np.linalg.norm(mean.ravel())
        )
        assert got.guard_rel_l2 == pytest.approx(expected, rel=1e-6)


class TestContract:
    def test_borrowed_field_has_no_sdf_attribute(self) -> None:
        fields, sdfs, cond = make_pool()
        for got in (
            FieldProvider(fields, pool_sdfs=sdfs, pool_cond=cond).borrow(target_sdf=sdfs[0]),
            FieldProvider(fields, pool_cond=cond).borrow(target_cond=cond[0]),
            FieldProvider(fields).borrow(),
        ):
            assert not hasattr(got, "sdf")
            assert not hasattr(got, "donor_sdf")
            assert not hasattr(got, "pool_sdfs")

    def test_borrowed_field_field_names_exact(self) -> None:
        names = {f.name for f in dataclasses.fields(BorrowedField)}
        assert names == {
            "fields",
            "donor_index",
            "strategy",
            "distance",
            "guard_ok",
            "guard_rel_l2",
            "guard_threshold",
            "provenance",
        }
        assert not (names & {"sdf", "donor_sdf"})

    def test_provenance_cites_study_and_contract(self) -> None:
        fields, _, _ = make_pool()
        prov = FieldProvider(fields).borrow().provenance
        assert prov["study"] == "/nfs/wangxi/runs/l2_field_sensitivity_20260904"
        assert "never returns a donor SDF" in prov["contract"]
        assert prov["pool_size"] == 3

    def test_strategies_constant(self) -> None:
        assert STRATEGIES == ("sdf_near", "cond_near", "mean")


class TestDeterminism:
    def test_same_inputs_same_donor(self) -> None:
        fields, sdfs, cond = make_pool(n=6)
        provider = FieldProvider(fields, pool_sdfs=sdfs, pool_cond=cond)
        a = provider.borrow(target_sdf=sdfs[3] + 0.25)
        b = provider.borrow(target_sdf=sdfs[3] + 0.25)
        assert a.donor_index == b.donor_index
        assert a.distance == b.distance
        assert a.guard_rel_l2 == b.guard_rel_l2
        np.testing.assert_array_equal(a.fields, b.fields)

    def test_tie_breaks_to_lowest_index(self) -> None:
        fields, sdfs, _ = make_pool(n=4)
        sdfs[3] = sdfs[1].copy()  # exact duplicate donor
        got = FieldProvider(fields, pool_sdfs=sdfs).borrow(target_sdf=sdfs[3].copy())
        assert got.donor_index == 1


class TestValidation:
    def test_wrong_channel_count(self) -> None:
        with pytest.raises(ValueError, match=r"\(N, 5, \.\.\.\)"):
            FieldProvider(np.zeros((3, 6, 4, 4), np.float32))

    def test_empty_pool(self) -> None:
        with pytest.raises(ValueError, match="at least one row"):
            FieldProvider(np.zeros((0, 5, 4, 4), np.float32))

    def test_pool_sdfs_row_mismatch(self) -> None:
        fields, sdfs, _ = make_pool(n=3)
        with pytest.raises(ValueError, match="matching"):
            FieldProvider(fields, pool_sdfs=sdfs[:-1])

    def test_pool_cond_shape(self) -> None:
        fields, _, cond = make_pool(n=3)
        with pytest.raises(ValueError, match=r"\(N, C\)"):
            FieldProvider(fields, pool_cond=cond.ravel())

    def test_target_sdf_shape_mismatch(self) -> None:
        fields, sdfs, _ = make_pool(n=3)
        with pytest.raises(ValueError, match="to match the pool SDF rows"):
            FieldProvider(fields, pool_sdfs=sdfs).borrow(target_sdf=np.zeros((5, 5, 5), np.float32))

    def test_target_cond_shape_mismatch(self) -> None:
        fields, _, cond = make_pool(n=3)
        with pytest.raises(ValueError, match="to match the"):
            FieldProvider(fields, pool_cond=cond).borrow(target_cond=np.zeros((1, cond.shape[1])))


def write_production_layout(directory: Path) -> dict[str, np.ndarray]:
    """Tiny synthetic mirror of the four-file production corpus layout.

    cache_fam.npz carries 2 rows, cache_ext56.npz 1 row (dsi=10, whose SDF
    is the single d10 member of sdf_ext2.npz).
    """
    rng = np.random.default_rng(7)
    fam = {
        "x": rng.standard_normal((2, FIELD_CHANNELS, 4, 4)).astype(np.float32),
        "dsi": np.array([0, 1], dtype=np.int64),
        "re": np.array([100.0, 200.0]),
        "uin": np.array([0.1, 0.2]),
        "sail": np.array([1.0, 0.9]),
        "fin": np.array([0.8, 1.1]),
    }
    ext = {
        "x": rng.standard_normal((1, FIELD_CHANNELS, 4, 4)).astype(np.float32),
        "dsi": np.array([10], dtype=np.int64),
        "re": np.array([300.0]),
        "uin": np.array([0.3]),
        "sail": np.array([1.2]),
        "fin": np.array([0.7]),
    }
    sdf_fam = rng.standard_normal((2, *_SDF_SHAPE)).astype(np.float32)
    d10 = rng.standard_normal(_SDF_SHAPE).astype(np.float32)
    np.savez(directory / "cache_fam.npz", **fam)
    np.savez(directory / "cache_ext56.npz", **ext)
    np.savez(directory / "sdf_fam350.npz", sdf=sdf_fam)
    np.savez(directory / "sdf_ext2.npz", d10=d10)
    return {"fam": fam, "ext": ext, "sdf_fam": sdf_fam, "d10": d10}


class TestFromCorpus:
    def test_production_dir_layout(self, tmp_path: Path) -> None:
        ref = write_production_layout(tmp_path)
        provider = FieldProvider.from_corpus(str(tmp_path))
        assert provider.pool_fields.shape == (3, FIELD_CHANNELS, 4, 4)
        np.testing.assert_array_equal(
            provider.pool_fields, np.concatenate([ref["fam"]["x"], ref["ext"]["x"]])
        )
        np.testing.assert_array_equal(
            provider.pool_sdfs, np.concatenate([ref["sdf_fam"], ref["d10"][None]])
        )
        expected_cond = np.stack(
            [
                np.log10(np.concatenate([ref["fam"][k], ref["ext"][k]]))
                for k in ("re", "uin", "sail", "fin")
            ],
            axis=1,
        )
        np.testing.assert_allclose(provider.pool_cond, expected_cond, rtol=0, atol=0)
        # retrieval against the assembled pool: an ext-row SDF retrieves the ext row
        got = provider.borrow(target_sdf=ref["d10"].copy())
        assert got.donor_index == 2
        assert got.provenance["donor_key"].startswith("dsi=10 re=300")
        assert got.provenance["source"] == f"production-dir:{tmp_path}"

    def test_production_dir_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="cache_fam.npz"):
            FieldProvider.from_corpus(str(tmp_path))

    def test_production_dir_missing_key(self, tmp_path: Path) -> None:
        (tmp_path / "cache_fam.npz").write_bytes(b"")
        np.savez(tmp_path / "cache_fam.npz", x=np.zeros((1, 5, 2, 2)))
        with pytest.raises(ValueError, match="missing required keys"):
            FieldProvider.from_corpus(str(tmp_path))

    def test_nonpositive_rejected(self, tmp_path: Path) -> None:
        write_production_layout(tmp_path)
        with np.load(tmp_path / "cache_fam.npz") as z:
            fam = {k: z[k] for k in z.files}
        fam["re"][0] = -1.0
        np.savez(tmp_path / "cache_fam.npz", **fam)
        with pytest.raises(ValueError, match="non-positive"):
            FieldProvider.from_corpus(str(tmp_path))

    def test_snapshot_npz_layout(self, tmp_path: Path) -> None:
        fields, sdfs, cond = make_pool(n=3)
        snap = tmp_path / "snapshot.npz"
        np.savez(
            snap,
            x=fields,
            sdf=sdfs,
            cond=cond,
            dsi=np.array([0, 1, 2]),
            re=np.array([10.0, 20.0, 30.0]),
        )
        provider = FieldProvider.from_corpus(snap)
        np.testing.assert_array_equal(provider.pool_fields, fields)
        np.testing.assert_array_equal(provider.pool_sdfs, sdfs)
        got = provider.borrow(target_sdf=sdfs[1])
        assert got.donor_index == 1
        assert got.provenance["donor_key"] == "dsi=1 re=20"
        assert got.provenance["source"] == f"snapshot-npz:{snap}"

    def test_snapshot_fields_alias(self, tmp_path: Path) -> None:
        fields, _, _ = make_pool(n=2)
        snap = tmp_path / "alias.npz"
        np.savez(snap, fields=fields)
        provider = FieldProvider.from_corpus(snap)
        np.testing.assert_array_equal(provider.pool_fields, fields)
        assert provider.pool_sdfs is None and provider.pool_cond is None

    def test_snapshot_missing_fields(self, tmp_path: Path) -> None:
        snap = tmp_path / "bad.npz"
        np.savez(snap, sdf=np.zeros((1, 3, 4, 4)))
        with pytest.raises(ValueError, match="neither 'x' nor 'fields'"):
            FieldProvider.from_corpus(snap)

    def test_missing_path(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="neither an .npz snapshot nor a directory"):
            FieldProvider.from_corpus(str(tmp_path / "nope"))

    @pytest.mark.skipif(
        bool(_PROD_MISSING),
        reason=f"production corpus not on this machine (missing: {_PROD_MISSING})",
    )
    def test_real_production_corpus_smoke(self, tmp_path: Path) -> None:
        """Real 406-row corpus via a read-only symlink VIEW directory."""
        for name in ("fam", "ext", "sdf_fam", "sdf_ext"):
            os.symlink(
                _PROD[name],
                tmp_path
                / {
                    "fam": "cache_fam.npz",
                    "ext": "cache_ext56.npz",
                    "sdf_fam": "sdf_fam350.npz",
                    "sdf_ext": "sdf_ext2.npz",
                }[name],
            )
        provider = FieldProvider.from_corpus(str(tmp_path))
        assert provider.pool_fields.shape == (406, FIELD_CHANNELS, 64, 128)
        assert provider.pool_sdfs is not None and provider.pool_sdfs.shape == (406, 32, 32, 64)
        assert provider.pool_cond is not None and provider.pool_cond.shape == (406, 4)
        with np.load(_PROD["fam"]) as z:
            np.testing.assert_array_equal(provider.pool_fields[:350], z["x"])
            np.testing.assert_array_equal(
                provider.pool_sdfs[:350], np.load(_PROD["sdf_fam"])["sdf"]
            )
        got = provider.borrow(target_sdf=provider.pool_sdfs[0])
        assert got.fields.shape == (FIELD_CHANNELS, 64, 128)
        assert got.provenance["pool_size"] == 406
