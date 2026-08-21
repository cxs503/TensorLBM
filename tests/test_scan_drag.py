"""Tests for the optional drag survey (``tensorlbm.scan_drag`` + scan hook).

CPU suite: spec/plan round-trip and digest policy, synthetic observer
correctness against a hand-computed kinetic balance, scan integration on
``suboff_n128`` (small grid), bit-identity of field products with and
without the survey, checkpoint-resume continuity of the sidecar, and the
error paths.  One real-CUDA integration test is included and skips where
no device is present.
"""

from __future__ import annotations

import json
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorlbm.d3q19 import C
from tensorlbm.data.solver_export import read_snapshot
from tensorlbm.scan_drag import DRAG_HISTORY_SCHEMA, DragSurveyObserver, DragSurveySpec
from tensorlbm.scan_runner import ScanExecutor, ScanPlan, ScanVariable, run_scan_point

CODE_SHA = "0" * 40


def _cavity_plan(**overrides) -> ScanPlan:
    kwargs = dict(  # noqa: C408
        scan_id="drag-cavity",
        case="cavity",
        variables=[ScanVariable(name="re", levels=[100.0, 400.0])],
        method="full_factorial",
        n_points=2,
        seed=0,
        steps=10,
        snapshot_every=5,
        code_sha=CODE_SHA,
        fixed_params={"resolution": 16},
    )
    kwargs.update(overrides)
    return ScanPlan.generate(**kwargs)


def _suboff_plan(**overrides) -> ScanPlan:
    kwargs = dict(  # noqa: C408
        scan_id="drag-suboff",
        case="suboff_n128",
        variables=[ScanVariable(name="re", levels=[148.0, 156.0])],
        method="full_factorial",
        n_points=2,
        seed=0,
        steps=60,
        snapshot_every=30,
        code_sha=CODE_SHA,
        fixed_params={"resolution": 24},
    )
    kwargs.update(overrides)
    return ScanPlan.generate(**kwargs)


def _block(shape, z0, z1, y0, y1, x0, x1) -> torch.Tensor:
    mask = torch.zeros(shape, dtype=torch.bool)
    mask[z0:z1, y0:y1, x0:x1] = True
    return mask


def _fake_case(solid: torch.Tensor):
    return SimpleNamespace(
        solid_mask=lambda: solid,
        periodic_axes=lambda: {"x": False, "y": False, "z": False},
    )


def _observer(solid: torch.Tensor, spec: DragSurveySpec, tmp_path) -> DragSurveyObserver:
    return DragSurveyObserver(
        spec,
        case=_fake_case(solid),
        scan_id="drag-synthetic",
        point_id="p0000",
        run_id="drag-synthetic-p0000",
        point_dir=tmp_path,
    )


# ---------------------------------------------------------------------------
# Spec + plan serialisation
# ---------------------------------------------------------------------------


class TestDragSurveySpec:
    def test_defaults_and_roundtrip(self) -> None:
        spec = DragSurveySpec()
        assert (spec.margin, spec.interval) == (6, 50)
        assert spec.to_dict() == {"margin": 6, "interval": 50}
        assert DragSurveySpec.from_dict(spec.to_dict()) == spec
        assert DragSurveySpec.from_dict(None) is None

    def test_rejects_non_positive(self) -> None:
        for bad in (0, -1, True, 2.5):
            with pytest.raises(ValueError, match="margin"):
                DragSurveySpec(margin=bad)
            with pytest.raises(ValueError, match="interval"):
                DragSurveySpec(interval=bad)


class TestPlanSerialization:
    def test_roundtrip_with_and_without_survey(self) -> None:
        for payload in (None, {"margin": 3, "interval": 5}):
            plan = _suboff_plan(drag_survey=payload)
            data = plan.to_dict()
            assert data["drag_survey"] == payload
            loaded = ScanPlan.from_dict(json.loads(json.dumps(data)))
            assert loaded.drag_survey == plan.drag_survey
            assert loaded.plan_digest() == plan.plan_digest()

    def test_legacy_dict_without_key_loads_as_none(self) -> None:
        data = _suboff_plan().to_dict()
        del data["drag_survey"]
        assert ScanPlan.from_dict(data).drag_survey is None

    def test_survey_less_digest_is_the_pre_feature_payload(self) -> None:
        plan = _suboff_plan()
        legacy_payload = json.dumps(
            {
                "scan_id": plan.scan_id,
                "case": plan.case,
                "variables": [v.to_dict() for v in plan.variables],
                "method": plan.method,
                "seed": plan.seed,
                "steps": plan.steps,
                "snapshot_every": plan.snapshot_every,
                "points": [p.params for p in plan.points],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        # Old checkpoints (identity = plan_digest) must keep resuming.
        assert plan.plan_digest() == sha256(legacy_payload).hexdigest()

    def test_survey_participates_in_digest(self) -> None:
        plain = _suboff_plan()
        surveyed = _suboff_plan(drag_survey={"margin": 3, "interval": 5})
        other = _suboff_plan(drag_survey={"margin": 4, "interval": 5})
        assert surveyed.plan_digest() != plain.plan_digest()
        assert other.plan_digest() != surveyed.plan_digest()


# ---------------------------------------------------------------------------
# Synthetic observer correctness
# ---------------------------------------------------------------------------

SHAPE = (10, 11, 12)


class TestObserverSynthetic:
    def _solid(self) -> torch.Tensor:
        # bbox z[4,5] y[4,6] x[5,7]
        return _block(SHAPE, 4, 6, 4, 7, 5, 8)

    def test_control_volume_bounds_and_clamping(self, tmp_path) -> None:
        obs = _observer(self._solid(), DragSurveySpec(margin=2, interval=1), tmp_path)
        assert obs.bounds == {"z0": 2, "z1": 8, "y0": 2, "y1": 9, "x0": 3, "x1": 10}
        assert obs.n_solid_cells == 18
        # An oversized margin clamps to the largest strictly interior window.
        wide = _observer(self._solid(), DragSurveySpec(margin=50, interval=1), tmp_path)
        assert wide.bounds == {"z0": 2, "z1": 8, "y0": 2, "y1": 9, "x0": 2, "x1": 10}

    def test_solid_touching_domain_edge_is_rejected(self, tmp_path) -> None:
        solid = _block(SHAPE, 0, 2, 2, 4, 2, 4)  # z face touches the boundary
        with pytest.raises(ValueError, match="strictly interior"):
            _observer(solid, DragSurveySpec(), tmp_path)

    def test_missing_or_empty_solid_is_rejected(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="solid_mask"):
            _observer(None, DragSurveySpec(), tmp_path)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="empty"):
            _observer(torch.zeros(SHAPE, dtype=torch.bool), DragSurveySpec(), tmp_path)

    def test_sample_matches_hand_computed_kinetic_balance(self, tmp_path) -> None:
        torch.manual_seed(20260821)
        solid = self._solid()
        obs = _observer(solid, DragSurveySpec(margin=2, interval=1), tmp_path)
        f_old = 0.05 + 0.02 * torch.rand((19, *SHAPE), dtype=torch.float64)
        f_new = f_old + 1.0e-3 * torch.randn((19, *SHAPE), dtype=torch.float64)
        f_pc = 0.05 + 0.02 * torch.rand((19, *SHAPE), dtype=torch.float64)

        entry = obs.sample(7, f_old=f_old, f_post_collision=f_pc, f_new=f_new)

        # Independent reference: padded (non-rolling) destination lookup.
        cv = torch.zeros(SHAPE, dtype=torch.bool)
        cv[
            obs.bounds["z0"] : obs.bounds["z1"],
            obs.bounds["y0"] : obs.bounds["y1"],
            obs.bounds["x0"] : obs.bounds["x1"],
        ] = True
        padded = torch.zeros((SHAPE[0] + 2, SHAPE[1] + 2, SHAPE[2] + 2), dtype=torch.bool)
        padded[1:-1, 1:-1, 1:-1] = cv
        owned = cv & ~solid
        c = C.to(dtype=torch.float64)
        imported = torch.zeros(3, dtype=torch.float64)
        change = torch.zeros(3, dtype=torch.float64)
        for q in range(1, 19):
            cx, cy, cz = (int(v) for v in c[q].tolist())
            dest = padded[
                1 + cz : 1 + cz + SHAPE[0], 1 + cy : 1 + cy + SHAPE[1], 1 + cx : 1 + cx + SHAPE[2]
            ]
            incoming = (~cv) & dest
            outgoing = cv & (~dest)
            imported = imported + (f_pc[q][incoming].sum() - f_pc[q][outgoing].sum()) * c[q]
            change = change + (f_new[q][owned] - f_old[q][owned]).sum() * c[q]
        expected = imported - change

        assert entry["step"] == 7
        for axis, name in ((0, "force_x"), (1, "force_y"), (2, "force_z")):
            assert entry[name] == pytest.approx(float(expected[axis]), rel=1e-12, abs=1e-14), name
        assert entry["force_abs"] == pytest.approx(float(expected.square().sum().sqrt()), rel=1e-12)

    def test_single_population_flux_is_exact(self, tmp_path) -> None:
        obs = _observer(self._solid(), DragSurveySpec(margin=2, interval=1), tmp_path)
        f = torch.zeros((19, *SHAPE), dtype=torch.float64)
        f_pc = torch.zeros_like(f)
        # Direction +x, source cell just outside the CV x-face: the whole
        # population enters the control volume in one step.
        f_pc[1, 4, 4, obs.bounds["x0"] - 1] = 0.2
        entry = obs.sample(3, f_old=f, f_post_collision=f_pc, f_new=f)
        assert entry["force_x"] == pytest.approx(0.2 * float(C[1, 0]), rel=1e-15)
        assert entry["force_y"] == pytest.approx(0.0, abs=1e-15)
        assert entry["force_z"] == pytest.approx(0.0, abs=1e-15)

    def test_summary_tail_mean(self, tmp_path) -> None:
        obs = _observer(self._solid(), DragSurveySpec(margin=2, interval=1), tmp_path)
        assert obs.summary() == {"drag_final": None, "drag_mean_tail": None}
        values = [0.5, 0.4, 0.3, 0.25, 0.22, 0.21, 0.205, 0.2025]
        for step, fx in enumerate(values, start=10):
            obs._history.append(
                {"step": step, "force_x": fx, "force_y": 0.0, "force_z": 0.0, "force_abs": fx}
            )
        summary = obs.summary()
        assert summary["drag_final"] == pytest.approx(values[-1])
        assert summary["drag_mean_tail"] == pytest.approx(
            sum(values[-2:]) / 2  # max(1, 8 // 4) = 2 tail samples
        )

    def test_mass_correction_injection_is_compensated(self, tmp_path) -> None:
        torch.manual_seed(20260822)
        obs = _observer(self._solid(), DragSurveySpec(margin=2, interval=1), tmp_path)
        f_old = 0.05 + 0.02 * torch.rand((19, *SHAPE), dtype=torch.float64)
        f_pc = 0.05 + 0.02 * torch.rand((19, *SHAPE), dtype=torch.float64)
        f_new = f_old + 1.0e-3 * torch.randn((19, *SHAPE), dtype=torch.float64)
        plain = obs.sample(10, f_old=f_old, f_post_collision=f_pc, f_new=f_new)

        # A global population rescale (correct_mass3d) between samples
        # shifts the fluid momentum inside the CV; the next sample must
        # add exactly that impulse back on top of its own balance.
        rescaled = f_new * 1.001
        obs.note_mass_correction(f_new, rescaled)
        f_old2, f_pc2 = rescaled, f_pc + 0.01 * torch.rand((19, *SHAPE), dtype=torch.float64)
        f_new2 = f_old2 + 1.0e-3 * torch.randn((19, *SHAPE), dtype=torch.float64)
        compensated = obs.sample(20, f_old=f_old2, f_post_collision=f_pc2, f_new=f_new2)

        reference = DragSurveyObserver(
            DragSurveySpec(margin=2, interval=1),
            case=_fake_case(self._solid()),
            scan_id="drag-synthetic",
            point_id="p0000",
            run_id="drag-synthetic-p0000",
            point_dir=tmp_path / "ref",
        )
        raw = reference.sample(20, f_old=f_old2, f_post_collision=f_pc2, f_new=f_new2)
        from tensorlbm.control_volume_force import fluid_momentum

        impulse = fluid_momentum(rescaled, reference._cv, solid=reference._solid) - fluid_momentum(
            f_new, reference._cv, solid=reference._solid
        )
        assert compensated["force_x"] == pytest.approx(
            raw["force_x"] + float(impulse[0]), rel=1e-12, abs=1e-14
        )
        assert plain["force_x"] != pytest.approx(compensated["force_x"])

    def test_compensation_covers_only_the_sampled_step(self, tmp_path) -> None:
        """interval > mass_correction_period must not over-compensate.

        Each sample is a one-step balance, so rescales applied on *other*
        (unsampled) steps belong to no measurement: only the rescale of
        the sampled step itself may be added back.  Regression for the
        n128 survey where accumulating injections since the previous
        sample biased margin-2 vs margin-4 CVs apart by ~5%.
        """
        torch.manual_seed(20260823)
        obs = _observer(self._solid(), DragSurveySpec(margin=2, interval=25), tmp_path)
        f = 0.05 + 0.02 * torch.rand((19, *SHAPE), dtype=torch.float64)
        f_pc = 0.05 + 0.02 * torch.rand((19, *SHAPE), dtype=torch.float64)
        f_next = f + 1.0e-3 * torch.randn((19, *SHAPE), dtype=torch.float64)

        # Two corrections on unsampled steps, then a third on the sampled
        # step: the sample must compensate ONLY the third.
        obs.note_mass_correction(f, f * 1.001)
        obs.note_mass_correction(f, f * 1.002)
        rescaled = f_next * 1.003
        obs.note_mass_correction(f_next, rescaled)
        compensated = obs.sample(50, f_old=f, f_post_collision=f_pc, f_new=rescaled)

        reference = DragSurveyObserver(
            DragSurveySpec(margin=2, interval=25),
            case=_fake_case(self._solid()),
            scan_id="drag-synthetic",
            point_id="p0000",
            run_id="drag-synthetic-p0000",
            point_dir=tmp_path / "ref",
        )
        raw = reference.sample(50, f_old=f, f_post_collision=f_pc, f_new=rescaled)
        from tensorlbm.control_volume_force import fluid_momentum

        impulse = fluid_momentum(rescaled, reference._cv, solid=reference._solid) - fluid_momentum(
            f_next, reference._cv, solid=reference._solid
        )
        assert compensated["force_x"] == pytest.approx(
            raw["force_x"] + float(impulse[0]), rel=1e-12, abs=1e-14
        )
        # And the accumulator is consumed: a following sample on a step
        # without correction is uncompensated.
        plain = obs.sample(75, f_old=rescaled, f_post_collision=f_pc, f_new=rescaled)
        raw2 = reference.sample(75, f_old=rescaled, f_post_collision=f_pc, f_new=rescaled)
        assert plain["force_x"] == pytest.approx(raw2["force_x"], rel=1e-15)

    def test_stale_injection_from_an_earlier_step_is_ignored(self, tmp_path) -> None:
        """A step-tagged rescale must not leak into a later sample.

        With interval=25 and a mass correction every 10 steps, every other
        sample lands on an uncorrected step whose most recent
        ``note_mass_correction`` belongs to an earlier step; applying it
        biased the n128 tail means apart (m2 2.972 vs m4 2.934 while the
        per-step forces agreed to ~1e-5).
        """
        torch.manual_seed(20260824)
        obs = _observer(self._solid(), DragSurveySpec(margin=2, interval=25), tmp_path)
        f = 0.05 + 0.02 * torch.rand((19, *SHAPE), dtype=torch.float64)
        f_pc = 0.05 + 0.02 * torch.rand((19, *SHAPE), dtype=torch.float64)
        f_next = f + 1.0e-3 * torch.randn((19, *SHAPE), dtype=torch.float64)

        reference = DragSurveyObserver(
            DragSurveySpec(margin=2, interval=25),
            case=_fake_case(self._solid()),
            scan_id="drag-synthetic",
            point_id="p0000",
            run_id="drag-synthetic-p0000",
            point_dir=tmp_path / "ref",
        )

        # Correction on step 20 (unsampled), sample on step 25 (uncorrected):
        # the stale impulse must NOT be added.
        obs.note_mass_correction(f, f * 1.001, step=20)
        stale = obs.sample(25, f_old=f, f_post_collision=f_pc, f_new=f_next)
        raw = reference.sample(25, f_old=f, f_post_collision=f_pc, f_new=f_next)
        assert stale["force_x"] == pytest.approx(raw["force_x"], rel=1e-15)

        # Correction on the sampled step itself is still compensated.
        rescaled = f_next * 1.002
        obs.note_mass_correction(f_next, rescaled, step=50)
        aligned = obs.sample(50, f_old=f, f_post_collision=f_pc, f_new=rescaled)
        from tensorlbm.control_volume_force import fluid_momentum

        impulse = fluid_momentum(rescaled, reference._cv, solid=reference._solid) - fluid_momentum(
            f_next, reference._cv, solid=reference._solid
        )
        raw2 = reference.sample(50, f_old=f, f_post_collision=f_pc, f_new=rescaled)
        assert aligned["force_x"] == pytest.approx(
            raw2["force_x"] + float(impulse[0]), rel=1e-12, abs=1e-14
        )

    def test_sidecar_resume_filters_and_tolerates_damage(self, tmp_path) -> None:
        obs = _observer(self._solid(), DragSurveySpec(margin=2, interval=10), tmp_path)
        f = torch.full((19, *SHAPE), 0.02, dtype=torch.float64)
        for step in (10, 20, 30):
            obs.sample(step, f_old=f, f_post_collision=f.clone(), f_new=f)
        path = tmp_path / "drag_history.json"
        assert path.is_file()

        fresh = _observer(self._solid(), DragSurveySpec(margin=2, interval=10), tmp_path)
        assert fresh.resume_from_sidecar(20) == 2
        assert [s["step"] for s in fresh.samples] == [10, 20]
        assert fresh.resume_from_sidecar(5) == 0  # everything beyond is dropped

        path.write_text("{ not json", encoding="utf-8")
        assert fresh.resume_from_sidecar(20) == 0
        path.write_text(
            json.dumps(
                {"schema": DRAG_HISTORY_SCHEMA, "point_id": "other", "run_id": "x", "samples": []}
            ),
            encoding="utf-8",
        )
        assert fresh.resume_from_sidecar(20) == 0  # foreign identity
        path.unlink()
        assert fresh.resume_from_sidecar(20) == 0  # missing sidecar


# ---------------------------------------------------------------------------
# Scan integration
# ---------------------------------------------------------------------------


class TestScanIntegration:
    def test_suboff_point_writes_drag_history_and_status(self, tmp_path) -> None:
        plan = _suboff_plan(
            steps=60,
            snapshot_every=30,
            drag_survey=DragSurveySpec(margin=2, interval=10),
        )
        executor = ScanExecutor(plan, tmp_path, serial_device="cpu")
        try:
            summary = executor.run()
        finally:
            executor.close()
        assert summary["n_failed"] == 0

        point = plan.points[0]
        point_dir = tmp_path / "points" / point.point_id
        doc = json.loads((point_dir / "drag_history.json").read_text(encoding="utf-8"))
        assert doc["schema"] == DRAG_HISTORY_SCHEMA
        assert doc["lattice_units"] is True
        assert "LATTICE units" in doc["units"]
        assert doc["interval"] == 10
        assert doc["point_id"] == point.point_id
        assert doc["control_volume"]["margin"] == 2

        steps = [s["step"] for s in doc["samples"]]
        assert steps == list(range(10, 61, 10))
        assert all(s["force_x"] > 0.0 for s in doc["samples"])

        status = json.loads((point_dir / "status.json").read_text(encoding="utf-8"))
        assert status["drag_final"] == pytest.approx(doc["samples"][-1]["force_x"])
        tail = doc["samples"][-max(1, len(doc["samples"]) // 4) :]
        assert status["drag_mean_tail"] == pytest.approx(
            sum(s["force_x"] for s in tail) / len(tail)
        )

    def test_suboff_tail_converges_over_300_steps(self, tmp_path) -> None:
        plan = _suboff_plan(
            steps=300,
            snapshot_every=150,
            drag_survey=DragSurveySpec(margin=2, interval=5),
        )
        executor = ScanExecutor(plan, tmp_path, serial_device="cpu")
        catalog = executor.catalog()
        try:
            outcome = run_scan_point(plan, plan.points[0], tmp_path, catalog, "cpu")
        finally:
            executor.close()
        assert outcome.status == "completed"

        doc = json.loads(
            (tmp_path / "points" / plan.points[0].point_id / "drag_history.json").read_text(
                encoding="utf-8"
            )
        )
        fx = [s["force_x"] for s in doc["samples"]]
        assert len(fx) == 60
        quarter = max(2, len(fx) // 4)

        def mean_step_drift(values: list[float]) -> float:
            return float(np.mean([abs(b - a) for a, b in zip(values, values[1:])]))

        assert mean_step_drift(fx[-quarter:]) < mean_step_drift(fx[:quarter])
        assert fx[-1] > 0.0


# ---------------------------------------------------------------------------
# Bit-identity of the default (survey-less) path
# ---------------------------------------------------------------------------


_LEGACY_STATUS_KEYS = (
    "point_id",
    "status",
    "product_ids",
    "exported_steps",
    "completed_steps",
    "early_stopped",
    "early_stop_reason",
    "params",
    "device",
)


class TestBitIdentity:
    def test_cavity_default_plan_is_deterministic(self, tmp_path) -> None:
        snapshots = []
        statuses = []
        for run in ("a", "b"):
            root = tmp_path / run
            executor = ScanExecutor(_cavity_plan(), root, serial_device="cpu")
            try:
                executor.run()
            finally:
                executor.close()
            point_dir = root / "points" / "p0000"
            arrays, _ = read_snapshot(point_dir / "fields.h5", 10)
            snapshots.append(arrays)
            statuses.append(json.loads((point_dir / "status.json").read_text(encoding="utf-8")))
        for key in ("rho", "ux", "uy", "uz"):
            assert np.array_equal(snapshots[0][key], snapshots[1][key]), key
        for key in _LEGACY_STATUS_KEYS:
            assert statuses[0][key] == statuses[1][key], key
        assert statuses[0]["drag_final"] is None
        assert statuses[0]["drag_mean_tail"] is None

    def test_survey_leaves_field_products_bit_identical(self, tmp_path) -> None:
        plans = {
            "off": _suboff_plan(steps=60, snapshot_every=30),
            "on": _suboff_plan(
                steps=60,
                snapshot_every=30,
                drag_survey=DragSurveySpec(margin=2, interval=10),
            ),
        }
        for name, plan in plans.items():
            root = tmp_path / name
            executor = ScanExecutor(plan, root, serial_device="cpu")
            try:
                summary = executor.run()
            finally:
                executor.close()
            assert summary["n_failed"] == 0

        for point_id in ("p0000", "p0001"):
            arrays_off, _ = read_snapshot(tmp_path / "off" / "points" / point_id / "fields.h5", 60)
            arrays_on, _ = read_snapshot(tmp_path / "on" / "points" / point_id / "fields.h5", 60)
            for key in ("rho", "ux", "uy", "uz"):
                assert np.array_equal(arrays_off[key], arrays_on[key]), key
            status_off = json.loads(
                (tmp_path / "off" / "points" / point_id / "status.json").read_text(encoding="utf-8")
            )
            status_on = json.loads(
                (tmp_path / "on" / "points" / point_id / "status.json").read_text(encoding="utf-8")
            )
            for key in _LEGACY_STATUS_KEYS:
                assert status_off[key] == status_on[key], key
            assert status_off["drag_final"] is None
            assert status_on["drag_final"] is not None
        assert not (tmp_path / "off" / "points" / "p0000" / "drag_history.json").exists()

    def test_cavity_drag_survey_requires_interior_solid(self, tmp_path) -> None:
        plan = _cavity_plan(drag_survey=DragSurveySpec(margin=2, interval=5))
        executor = ScanExecutor(plan, tmp_path, serial_device="cpu")
        catalog = executor.catalog()
        try:
            with pytest.raises(ValueError, match="strictly interior"):
                run_scan_point(plan, plan.points[0], tmp_path, catalog, "cpu")
        finally:
            executor.close()


# ---------------------------------------------------------------------------
# Checkpoint resume continuity
# ---------------------------------------------------------------------------


class TestResume:
    def _crash_at(self, monkeypatch, step: int) -> None:
        import tensorlbm.reporters as reporters_mod

        real_dispatch = reporters_mod.dispatch

        def exploding(ctx, reps) -> None:
            real_dispatch(ctx, reps)
            if ctx.step >= step:
                raise RuntimeError("simulated crash")

        monkeypatch.setattr(reporters_mod, "dispatch", exploding)

    def test_kill_and_resume_continues_drag_history(self, tmp_path, monkeypatch) -> None:
        plan = _suboff_plan(
            steps=60,
            snapshot_every=20,
            drag_survey=DragSurveySpec(margin=2, interval=10),
        )
        point = plan.points[0]
        executor = ScanExecutor(plan, tmp_path, serial_device="cpu")
        catalog = executor.catalog()
        point_dir = tmp_path / "points" / point.point_id
        try:
            self._crash_at(monkeypatch, 35)
            with pytest.raises(RuntimeError, match="simulated crash"):
                run_scan_point(plan, point, tmp_path, catalog, "cpu", checkpoint_every=20)
            monkeypatch.undo()

            partial = json.loads((point_dir / "drag_history.json").read_text(encoding="utf-8"))
            assert [s["step"] for s in partial["samples"]] == [10, 20, 30]

            outcome = run_scan_point(plan, point, tmp_path, catalog, "cpu", checkpoint_every=20)
        finally:
            executor.close()
        assert outcome.status == "completed"
        assert outcome.completed_steps == 60

        doc = json.loads((point_dir / "drag_history.json").read_text(encoding="utf-8"))
        steps = [s["step"] for s in doc["samples"]]
        assert steps == sorted(set(steps)) == list(range(10, 61, 10))
        assert outcome.drag_final == pytest.approx(doc["samples"][-1]["force_x"])

    def test_missing_sidecar_restarts_history_without_failing(self, tmp_path, monkeypatch) -> None:
        plan = _suboff_plan(
            steps=60,
            snapshot_every=20,
            drag_survey=DragSurveySpec(margin=2, interval=10),
        )
        point = plan.points[0]
        executor = ScanExecutor(plan, tmp_path, serial_device="cpu")
        catalog = executor.catalog()
        point_dir = tmp_path / "points" / point.point_id
        try:
            self._crash_at(monkeypatch, 35)
            with pytest.raises(RuntimeError, match="simulated crash"):
                run_scan_point(plan, point, tmp_path, catalog, "cpu", checkpoint_every=20)
            monkeypatch.undo()
            (point_dir / "drag_history.json").unlink()

            outcome = run_scan_point(plan, point, tmp_path, catalog, "cpu", checkpoint_every=20)
        finally:
            executor.close()
        assert outcome.status == "completed"
        doc = json.loads((point_dir / "drag_history.json").read_text(encoding="utf-8"))
        # Fresh history from the resume step onward: no failure, no duplicates.
        assert [s["step"] for s in doc["samples"]] == [30, 40, 50, 60]
        assert outcome.drag_final == pytest.approx(doc["samples"][-1]["force_x"])


# ---------------------------------------------------------------------------
# Real CUDA integration
# ---------------------------------------------------------------------------


class TestCudaDragSurvey:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
    def test_suboff_cuda_positive_force_and_nested_cv_invariance(self, tmp_path) -> None:
        tail_forces = {}
        for margin in (2, 5):
            plan = _suboff_plan(
                steps=120,
                snapshot_every=60,
                drag_survey=DragSurveySpec(margin=margin, interval=10),
            )
            root = tmp_path / f"m{margin}"
            executor = ScanExecutor(plan, root, serial_device="cuda")
            try:
                summary = executor.run()
            finally:
                executor.close()
            assert summary["n_failed"] == 0
            doc = json.loads(
                (root / "points" / "p0000" / "drag_history.json").read_text(encoding="utf-8")
            )
            forces = [s["force_x"] for s in doc["samples"]]
            assert [s["step"] for s in doc["samples"]] == list(range(10, 121, 10))
            assert all(f > 0.0 and np.isfinite(f) for f in forces)
            tail_forces[margin] = float(np.mean(forces[-3:]))
        # With the per-step mass-correction impulse compensated (see
        # DragSurveyObserver.note_mass_correction), nested control volumes
        # measure the same body force to ~1e-4 relative.
        relative = abs(tail_forces[5] - tail_forces[2]) / tail_forces[2]
        assert relative < 1.0e-3, f"nested-CV mismatch: {tail_forces}"
