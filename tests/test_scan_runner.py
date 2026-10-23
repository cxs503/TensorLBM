"""Tests for tensorlbm.scan_runner (DoE plan -> executor -> catalog -> dataset).

CPU-only: the executor's serial path is exercised end to end on the
``cavity`` case at toy resolution; the GPU spawn path shares
``run_scan_point`` with the serial path and is covered by the on-server
validation runs (see /nfs/wangxi/as_scan_20260820/report.md).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from tensorlbm.cases import get_case, run_case
from tensorlbm.data.catalog import FieldDataCatalog
from tensorlbm.data.solver_export import read_snapshot
from tensorlbm.scan_runner import (
    ScanExecutor,
    ScanPlan,
    ScanPoint,
    ScanVariable,
    assign_points_to_gpus,
    coerce_case_params,
    run_scan_point,
    split_points,
)

CODE_SHA = "0" * 40


def _tiny_plan(**overrides) -> ScanPlan:
    kwargs = dict(  # noqa: C408
        scan_id="test-scan",
        case="cavity",
        variables=[ScanVariable(name="re", levels=[100.0, 400.0])],
        method="full_factorial",
        n_points=2,
        seed=0,
        steps=30,
        snapshot_every=10,
        code_sha=CODE_SHA,
        fixed_params={"resolution": 16},
    )
    kwargs.update(overrides)
    return ScanPlan.generate(**kwargs)


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


class TestScanPlan:
    def test_deterministic_same_seed(self):
        a = ScanPlan.generate(
            scan_id="lhs",
            case="cavity",
            variables=[
                ScanVariable(name="re", low=100.0, high=1000.0),
                ScanVariable(name="u_lid", low=0.03, high=0.09),
            ],
            method="latin_hypercube",
            n_points=8,
            seed=42,
            code_sha=CODE_SHA,
        )
        b = ScanPlan.generate(
            scan_id="lhs",
            case="cavity",
            variables=[
                ScanVariable(name="re", low=100.0, high=1000.0),
                ScanVariable(name="u_lid", low=0.03, high=0.09),
            ],
            method="latin_hypercube",
            n_points=8,
            seed=42,
            code_sha=CODE_SHA,
        )
        assert [p.params for p in a.points] == [p.params for p in b.points]
        assert [p.point_id for p in a.points] == [p.point_id for p in b.points]
        assert a.plan_digest() == b.plan_digest()

    def test_different_seed_differs(self):
        common = dict(  # noqa: C408
            scan_id="lhs",
            case="cavity",
            variables=[ScanVariable(name="re", low=100.0, high=1000.0)],
            method="latin_hypercube",
            n_points=8,
            code_sha=CODE_SHA,
        )
        a = ScanPlan.generate(seed=1, **common)
        b = ScanPlan.generate(seed=2, **common)
        assert [p.params for p in a.points] != [p.params for p in b.points]

    def test_json_roundtrip(self, tmp_path):
        plan = _tiny_plan()
        path = plan.save(tmp_path / "plan.json")
        loaded = ScanPlan.load(path)
        assert loaded.to_dict() == plan.to_dict()
        assert loaded.plan_digest() == plan.plan_digest()

    def test_factorial_point_count(self):
        plan = ScanPlan.generate(
            scan_id="fac",
            case="cavity",
            variables=[
                ScanVariable(name="re", levels=[100.0, 400.0]),
                ScanVariable(name="u_lid", levels=[0.05, 0.08]),
            ],
            method="full_factorial",
            n_points=4,  # ignored for factorial
            code_sha=CODE_SHA,
        )
        assert len(plan.points) == 4

    def test_factorial_n_points_reconciled_to_len_points(self):
        """``plan.n_points`` records the realised count, not the request."""
        plan = ScanPlan.generate(
            scan_id="fac22-default-request",
            case="cavity",
            variables=[
                ScanVariable(name="re", levels=[100.0, 400.0]),
                ScanVariable(name="u_lid", levels=[0.05, 0.08]),
            ],
            method="full_factorial",  # request left at the default 16
            code_sha=CODE_SHA,
        )
        assert len(plan.points) == 4
        assert plan.n_points == 4 == len(plan.points)
        assert plan.to_dict()["n_points"] == 4

    def test_factorial_three_variables_product_count(self):
        plan = ScanPlan.generate(
            scan_id="fac234",
            case="cavity",
            variables=[
                ScanVariable(name="a", levels=[1.0, 2.0]),
                ScanVariable(name="b", levels=[1.0, 2.0, 3.0]),
                ScanVariable(name="c", levels=[1.0, 2.0, 3.0, 4.0]),
            ],
            method="full_factorial",
            code_sha=CODE_SHA,
        )
        assert plan.n_points == len(plan.points) == 24

    def test_direct_construction_reconciles_n_points(self):
        points = tuple(
            ScanPoint(
                index=i,
                point_id=f"p{i:04d}",
                run_id=f"manual-p{i:04d}",
                params={"re": 100.0},
            )
            for i in range(3)
        )
        plan = ScanPlan(
            scan_id="manual",
            case="cavity",
            variables=(ScanVariable(name="re", levels=[100.0, 400.0]),),
            method="full_factorial",
            n_points=16,  # mismatching request -> reconciled to len(points)
            seed=0,
            steps=30,
            snapshot_every=10,
            code_sha=CODE_SHA,
            points=points,
        )
        assert plan.n_points == 3 == len(plan.points)

    def test_from_dict_heals_legacy_mismatched_n_points(self):
        plan = _tiny_plan()
        data = plan.to_dict()
        data["n_points"] = 16  # legacy plan.json written before reconciliation
        loaded = ScanPlan.from_dict(data)
        assert loaded.n_points == len(loaded.points) == 2
        assert loaded.to_dict() == plan.to_dict()

    def test_categorical_params_roundtrip(self):
        """String sweep params (e.g. hull_type) survive plan serialisation.

        DoE rows are floats, but directly-constructed points may carry
        categoricals; metadata/plan writes must not float()-cast them
        (the B1-4 hull campaign passes hull_type this way).
        """
        points = tuple(
            ScanPoint(
                index=i,
                point_id=f"p{i:04d}",
                run_id=f"hull-p{i:04d}",
                params={"hull_type": hull, "re": re},
            )
            for i, (hull, re) in enumerate(
                [("bare_hull", 800.0), ("with_sail", 200.0), ("full", 50.0)]
            )
        )
        plan = ScanPlan(
            scan_id="hull-scan",
            case="suboff_n128",
            variables=(ScanVariable(name="re", low=50.0, high=800.0),),
            method="lhs",
            n_points=len(points),
            seed=0,
            steps=30,
            snapshot_every=10,
            code_sha=CODE_SHA,
            points=points,
        )
        data = plan.to_dict()
        assert data["points"][0]["params"]["hull_type"] == "bare_hull"
        loaded = ScanPlan.from_dict(data)
        assert [pp.params["hull_type"] for pp in loaded.points] == [
            "bare_hull",
            "with_sail",
            "full",
        ]
        assert loaded.points[2].params["re"] == 50.0
        # the metadata helper passes numerics through as floats, categoricals verbatim
        from tensorlbm.scan_runner import _param_meta

        assert _param_meta(800) == 800.0 and isinstance(_param_meta(800), float)
        assert _param_meta("with_sail") == "with_sail"
        assert _param_meta(True) is True

    def test_lhs_n_points_authoritative(self):
        plan = ScanPlan.generate(
            scan_id="lhs7",
            case="cavity",
            variables=[ScanVariable(name="re", low=100.0, high=1000.0)],
            method="latin_hypercube",
            n_points=7,
            seed=3,
            code_sha=CODE_SHA,
        )
        assert plan.n_points == len(plan.points) == 7

    def test_rejects_bad_code_sha(self):
        with pytest.raises(ValueError, match="40 lowercase hex"):
            ScanPlan.generate(
                scan_id="x",
                case="cavity",
                variables=[ScanVariable(name="re", levels=[1.0, 2.0])],
                method="full_factorial",
                code_sha="deadbeef",
            )

    def test_rejects_snapshot_every_gt_steps(self):
        with pytest.raises(ValueError, match="snapshot_every"):
            _tiny_plan(steps=10, snapshot_every=20)

    def test_params_within_bounds(self):
        plan = ScanPlan.generate(
            scan_id="lhs",
            case="cavity",
            variables=[
                ScanVariable(name="re", low=100.0, high=1000.0),
                ScanVariable(name="u_lid", low=0.03, high=0.09),
            ],
            method="latin_hypercube",
            n_points=12,
            seed=7,
            code_sha=CODE_SHA,
        )
        for point in plan.points:
            assert 100.0 <= point.params["re"] <= 1000.0
            assert 0.03 <= point.params["u_lid"] <= 0.09


# ---------------------------------------------------------------------------
# Scheduling helpers
# ---------------------------------------------------------------------------


class TestScheduling:
    def test_round_robin_assignment(self):
        table = assign_points_to_gpus(7, [0, 1])
        assert sorted(table) == [0, 1]
        flat = sorted(i for ids in table.values() for i in ids)
        assert flat == list(range(7))
        assert abs(len(table[0]) - len(table[1])) <= 1
        assert table[0] == [0, 2, 4, 6]
        assert table[1] == [1, 3, 5]

    def test_assignment_deterministic(self):
        assert assign_points_to_gpus(9, [0, 1]) == assign_points_to_gpus(9, [0, 1])

    def test_assignment_rejects_duplicates(self):
        with pytest.raises(ValueError, match="unique"):
            assign_points_to_gpus(4, [0, 0])

    def test_split_points_disjoint_and_complete(self):
        for n in (1, 2, 7, 32):
            split = split_points(n, seed=3)
            assert sorted(split["train"] + split["val"] + split["test"]) == list(range(n))
            assert split["train"]
            assert not (set(split["train"]) & set(split["val"]))
            assert split == split_points(n, seed=3)

    def test_coerce_params_int_cast(self):
        out = coerce_case_params("cavity", {"resolution": 32.0, "re": 400.0})
        assert isinstance(out["resolution"], int)
        assert out["resolution"] == 32
        assert isinstance(out["re"], float)

    def test_coerce_params_rounds_fractional(self):
        # LHS samples non-integral resolutions; the constructor needs ints.
        out = coerce_case_params("cavity", {"resolution": 53.217})
        assert out["resolution"] == 53 and isinstance(out["resolution"], int)


# ---------------------------------------------------------------------------
# Per-point execution: bit alignment with run_case
# ---------------------------------------------------------------------------


class TestScanPointMatchesRunCase:
    def test_factorial_point_bitwise_aligned(self, tmp_path):
        plan = _tiny_plan()
        executor = ScanExecutor(plan, tmp_path, serial_device="cpu")
        catalog = executor.catalog()
        try:
            for point in plan.points:
                outcome = run_scan_point(plan, point, tmp_path, catalog, "cpu")
                assert outcome.status == "completed"
                assert outcome.exported_steps == [10, 20, 30]
                assert len(outcome.product_ids) == 3

                ref_case = get_case("cavity", resolution=16, re=point.params["re"])
                ref = run_case(ref_case, steps=30)

                arrays, attrs = read_snapshot(
                    tmp_path / "points" / point.point_id / "fields.h5", 30
                )
                assert attrs["run_id"] == point.run_id
                assert attrs["scan_id"] == plan.scan_id
                # Bit-level agreement with the direct registry run (same
                # step chain); fall back to the 1e-6 gate on any platform
                # where eager kernels are not run-to-run deterministic.
                for key, ref_field in (
                    ("rho", ref.rho),
                    ("ux", ref.ux),
                    ("uy", ref.uy),
                    ("uz", ref.uz),
                ):
                    got = arrays[key]
                    want = ref_field.detach().cpu().numpy()
                    if not np.array_equal(got, want):
                        assert np.max(np.abs(got - want)) <= 1e-6, key
        finally:
            executor.close()


# ---------------------------------------------------------------------------
# Executor: run, resume, finalise
# ---------------------------------------------------------------------------


class TestScanExecutor:
    def _run_small(self, tmp_path, steps=24):
        plan = _tiny_plan(steps=steps, snapshot_every=12)
        executor = ScanExecutor(plan, tmp_path, serial_device="cpu")
        summary = executor.run()
        return plan, summary, tmp_path

    def test_run_produces_products_and_dataset(self, tmp_path):
        plan, summary, root = self._run_small(tmp_path)
        assert summary["n_completed"] == len(plan.points)
        assert summary["n_failed"] == 0
        assert (root / "plan.json").exists()
        assert (root / "catalog.db").exists()
        assert (root / "scan_summary.json").exists()
        dataset = summary["dataset"]
        assert dataset["n_samples"] == 2 * 2  # 2 points x 2 snapshots
        assert sum(dataset["splits"].values()) == dataset["n_samples"]
        assert len(dataset["training_input_fingerprint"]) == 64

    def test_resume_skips_completed_points(self, tmp_path):
        plan, first, root = self._run_small(tmp_path)
        n_products_first = first["dataset"]["n_samples"]

        executor = ScanExecutor(plan, tmp_path, serial_device="cpu")
        second = executor.run(resume=True)
        assert second["n_skipped"] == len(plan.points)
        assert second["n_completed"] == len(plan.points)
        assert second["n_failed"] == 0
        assert second["dataset"]["n_samples"] == n_products_first

    def test_resume_reruns_incomplete_point(self, tmp_path):
        plan, first, root = self._run_small(tmp_path)
        # Simulate an interrupted rerun: drop one point's status file.
        status = root / "points" / "p0001" / "status.json"
        status.unlink()
        executor = ScanExecutor(plan, tmp_path, serial_device="cpu")
        third = executor.run(resume=True)
        assert third["n_skipped"] == 1
        outcomes = {o["point_id"]: o for o in third["outcomes"]}
        assert outcomes["p0001"]["status"] == "completed"
        # The dataset is rebuilt from product existence: no duplicates.
        catalog = FieldDataCatalog.open(root / "catalog.db")
        try:
            products = catalog.find_assets_by_metadata(
                "point_id", "p0001", kind="field_product", limit=100
            )
            assert len(products) == 2  # active products only
        finally:
            catalog.close()

    def test_finalize_lineage_chain(self, tmp_path):
        plan, summary, root = self._run_small(tmp_path)
        catalog = FieldDataCatalog.open(root / "catalog.db")
        try:
            dataset_id = summary["dataset"]["asset_id"]
            plan_id = summary["dataset"]["plan_asset_id"]
            upstream = set(catalog.upstream(dataset_id))
            assert plan_id in upstream
            for point in plan.points:
                assert f"run:{point.run_id}" in upstream
                for product_id in summary["dataset"]["products_by_point"][point.point_id]:
                    assert product_id in upstream
            # dataset.json mirrors the catalog view
            info = json.loads((root / "dataset.json").read_text())
            assert info["asset_id"] == dataset_id
            assert (
                info["training_input_fingerprint"]
                == summary["dataset"]["training_input_fingerprint"]
            )
        finally:
            catalog.close()

    def test_split_leakage_safety(self, tmp_path):
        plan, summary, root = self._run_small(tmp_path)
        info = json.loads((root / "dataset.json").read_text())
        by_point = {
            pid: product_id
            for pid, products in info["products_by_point"].items()
            for product_id in products
        }
        # Each product id appears in exactly one split's point set.
        split_products = {
            split: {by_point[pid] for pid in points}
            for split, points in info["split_points"].items()
        }
        train, val, test = (
            split_products["train"],
            split_products["val"],
            split_products["test"],
        )
        assert not (train & val) and not (train & test) and not (val & test)
        assert train | val | test == set(by_point.values())
        # No point id in two splits.
        all_points = [p for pts in info["split_points"].values() for p in pts]
        assert len(all_points) == len(set(all_points))

    def test_unknown_case_raises(self, tmp_path):
        plan = _tiny_plan()
        object.__setattr__(plan, "case", "no-such-case")
        with pytest.raises(KeyError, match="no-such-case"):
            ScanExecutor(plan, tmp_path, serial_device="cpu").run()


# ---------------------------------------------------------------------------
# GPU worker plumbing (no CUDA needed)
# ---------------------------------------------------------------------------


class TestWorkerPlumbing:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
    def test_gpu_worker_entry_skips_when_complete(self, tmp_path):
        """The in-worker resume guard skips completed points (CPU device)."""
        from tensorlbm.scan_runner import _gpu_worker_entry

        plan = _tiny_plan(steps=12, snapshot_every=6)
        executor = ScanExecutor(plan, tmp_path, serial_device="cpu")
        executor.run()
        executor.close()
        # Rewrite point p0000's device entry as if produced by "gpu 0".
        outcome_path = tmp_path / "logs" / "gpu0.outcomes.json"
        log_path = tmp_path / "logs" / "gpu0.log"
        _gpu_worker_entry(
            plan.to_dict(),
            str(tmp_path),
            0,
            [plan.points[0].index],
            str(outcome_path),
            str(log_path),
            60.0,
        )
        entries = json.loads(outcome_path.read_text())
        assert len(entries) == 1
        assert entries[0]["status"] == "skipped"
