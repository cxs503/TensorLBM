"""Tests for tensorlbm.checkpoint.

Covers the v1 tensor+JSON pair (``save_checkpoint`` / ``load_checkpoint``)
and the v2 unified solver-state convention (``save_solver_checkpoint`` /
``load_solver_checkpoint``): eager / triton_fused adapters, fail-closed
integrity checks, sharded-state equivalence with the multi-GPU rank-file
convention, and the scan_runner campaign resume hook.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from tensorlbm import (
    CheckpointError,
    case_checkpoint_path,
    eager_load_state_dict,
    eager_state_dict,
    load_case_checkpoint,
    load_checkpoint,
    load_solver_checkpoint,
    save_case_checkpoint,
    save_checkpoint,
    save_solver_checkpoint,
    triton_fused_load_state_dict,
    triton_fused_state_dict,
)
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.data.solver_export import read_snapshot
from tensorlbm.lbm_step import LBMStepExecutor
from tensorlbm.scan_runner import ScanExecutor, ScanPlan, ScanVariable

if TYPE_CHECKING:
    from pathlib import Path


class TestSaveCheckpoint:
    def test_returns_run_dir(self, tmp_path: Path) -> None:
        f = torch.ones((9, 4, 6))
        result = save_checkpoint(f, step=10, run_dir=tmp_path)
        assert result == tmp_path

    def test_creates_tensor_file(self, tmp_path: Path) -> None:
        f = torch.ones((9, 4, 6))
        save_checkpoint(f, step=5, run_dir=tmp_path)
        assert (tmp_path / "checkpoint_f.pt").exists()

    def test_creates_meta_file(self, tmp_path: Path) -> None:
        f = torch.ones((9, 4, 6))
        save_checkpoint(f, step=5, run_dir=tmp_path)
        assert (tmp_path / "checkpoint_meta.json").exists()

    def test_creates_directory_if_missing(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "subdir" / "nested"
        f = torch.ones((9, 4, 6))
        save_checkpoint(f, step=1, run_dir=new_dir)
        assert new_dir.exists()

    def test_extra_metadata_stored(self, tmp_path: Path) -> None:
        import json

        f = torch.ones((9, 4, 6))
        save_checkpoint(f, step=7, run_dir=tmp_path, extra={"re": 100.0, "label": "test"})
        meta = json.loads((tmp_path / "checkpoint_meta.json").read_text(encoding="utf-8"))
        assert meta["re"] == 100.0
        assert meta["label"] == "test"
        assert meta["step"] == 7

    def test_step_written_to_meta(self, tmp_path: Path) -> None:
        import json

        f = torch.ones((9, 4, 6))
        save_checkpoint(f, step=42, run_dir=tmp_path)
        meta = json.loads((tmp_path / "checkpoint_meta.json").read_text(encoding="utf-8"))
        assert meta["step"] == 42
        assert meta["format_version"] == 1
        assert meta["tensor_shape"] == [9, 4, 6]


class TestLoadCheckpoint:
    def _save(self, tmp_path: Path, f: torch.Tensor, step: int, extra: dict | None = None) -> None:
        save_checkpoint(f, step=step, run_dir=tmp_path, extra=extra)

    def test_roundtrip_tensor(self, tmp_path: Path) -> None:
        f_orig = torch.rand((9, 4, 6))
        self._save(tmp_path, f_orig, step=3)
        f_loaded, step, meta = load_checkpoint(tmp_path)
        assert torch.allclose(f_loaded, f_orig, atol=1e-6)

    def test_roundtrip_step(self, tmp_path: Path) -> None:
        f = torch.ones((9, 4, 6))
        self._save(tmp_path, f, step=99)
        _, step, _ = load_checkpoint(tmp_path)
        assert step == 99

    def test_roundtrip_meta_contains_step(self, tmp_path: Path) -> None:
        f = torch.ones((9, 4, 6))
        self._save(tmp_path, f, step=12)
        _, _, meta = load_checkpoint(tmp_path)
        assert meta["step"] == 12

    def test_roundtrip_extra_metadata(self, tmp_path: Path) -> None:
        f = torch.ones((9, 4, 6))
        self._save(tmp_path, f, step=1, extra={"nu": 0.01})
        _, _, meta = load_checkpoint(tmp_path)
        assert meta["nu"] == 0.01

    def test_missing_tensor_raises(self, tmp_path: Path) -> None:
        f = torch.ones((9, 4, 6))
        self._save(tmp_path, f, step=1)
        (tmp_path / "checkpoint_f.pt").unlink()
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path)

    def test_missing_meta_raises(self, tmp_path: Path) -> None:
        f = torch.ones((9, 4, 6))
        self._save(tmp_path, f, step=1)
        (tmp_path / "checkpoint_meta.json").unlink()
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path)

    def test_missing_step_metadata_raises(self, tmp_path: Path) -> None:
        import json

        f = torch.ones((9, 4, 6))
        self._save(tmp_path, f, step=1)
        (tmp_path / "checkpoint_meta.json").write_text(
            json.dumps({"label": "corrupt"}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="missing 'step' key"):
            load_checkpoint(tmp_path)

    def test_non_integer_step_metadata_raises(self, tmp_path: Path) -> None:
        import json

        f = torch.ones((9, 4, 6))
        self._save(tmp_path, f, step=1)
        (tmp_path / "checkpoint_meta.json").write_text(
            json.dumps({"step": "1"}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="'step' must be an integer"):
            load_checkpoint(tmp_path)

    def test_3d_tensor_roundtrip(self, tmp_path: Path) -> None:
        f_orig = torch.rand((19, 4, 6, 8))
        self._save(tmp_path, f_orig, step=5)
        f_loaded, step, _ = load_checkpoint(tmp_path)
        assert f_loaded.shape == f_orig.shape
        assert torch.allclose(f_loaded, f_orig, atol=1e-6)

    def test_loaded_on_cpu(self, tmp_path: Path) -> None:
        f = torch.ones((9, 4, 6))
        self._save(tmp_path, f, step=1)
        f_loaded, _, _ = load_checkpoint(tmp_path, device=torch.device("cpu"))
        assert f_loaded.device.type == "cpu"

    def test_incompatible_expected_shape_raises(self, tmp_path: Path) -> None:
        f = torch.ones((9, 4, 6))
        self._save(tmp_path, f, step=1)
        with pytest.raises(ValueError, match="incompatible with current run shape"):
            load_checkpoint(tmp_path, expected_shape=(19, 4, 6))

    def test_incompatible_expected_lattice_raises(self, tmp_path: Path) -> None:
        f = torch.ones((9, 4, 6))
        self._save(tmp_path, f, step=1)
        with pytest.raises(ValueError, match="incompatible with current lattice model"):
            load_checkpoint(tmp_path, expected_lattice_directions=19)


# ---------------------------------------------------------------------------
# Unified solver-state checkpoints (format v2)
# ---------------------------------------------------------------------------

CODE_SHA = "0" * 40


def _make_eager_solver() -> LBMStepExecutor:
    return LBMStepExecutor(
        "D3Q19",
        collide_fn="bgk",
        device=torch.device("cpu"),
        nx=6,
        ny=5,
        nz=4,
        tau=0.6,
    )


def _initial_f() -> torch.Tensor:
    torch.manual_seed(20260821)
    nz, ny, nx = 4, 5, 6
    rho = 1.0 + 0.02 * torch.rand(nz, ny, nx)
    ux = 0.05 * torch.rand(nz, ny, nx)
    uy = 0.02 * torch.rand(nz, ny, nx)
    uz = 0.01 * torch.rand(nz, ny, nx)
    return equilibrium3d(rho, ux, uy, uz).clone()


def _tiny_campaign_plan(scan_id: str = "ckpt-scan") -> ScanPlan:
    return ScanPlan.generate(
        scan_id=scan_id,
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


class TestSolverStateRoundTrip:
    def test_resume_is_bitwise_identical(self, tmp_path: Path) -> None:
        f0 = _initial_f()
        reference = _make_eager_solver()
        f_ref = f0.clone()
        for _ in range(10):
            f_ref, _ = reference.step(f_ref)

        solver = _make_eager_solver()
        f_a = f0.clone()
        for _ in range(5):
            f_a, _ = solver.step(f_a)
        state = eager_state_dict(solver, f_a, step=5)
        path = save_solver_checkpoint(tmp_path / "solver.ckpt", state, metadata={"run": "a7"})
        assert path.is_file()

        restored_solver = _make_eager_solver()
        checkpoint = load_solver_checkpoint(path)
        assert checkpoint.step == 5
        assert checkpoint.lattice == "D3Q19"
        assert checkpoint.grid == (4, 5, 6)
        assert checkpoint.q == 19
        assert checkpoint.metadata == {"run": "a7"}
        assert checkpoint.state["tau"] == pytest.approx(0.6)
        f_b = eager_load_state_dict(restored_solver, checkpoint.state)
        assert restored_solver._report_step == 5
        for _ in range(5):
            f_b, _ = restored_solver.step(f_b)
        assert torch.equal(f_b, f_ref)

    def test_state_dict_extracts_solver_identity(self) -> None:
        solver = _make_eager_solver()
        f = _initial_f()
        state = eager_state_dict(solver, f, step=9)
        assert state["lattice"] == "D3Q19"
        assert state["grid"] == (4, 5, 6)
        assert state["q"] == 19
        assert state["tau"] == pytest.approx(0.6)
        assert state["dtype"] == "torch.float32"
        assert state["step"] == 9

    def test_no_temporary_files_left_behind(self, tmp_path: Path) -> None:
        path = save_solver_checkpoint(
            tmp_path / "solver.ckpt", {"populations": torch.ones(9, 4, 6), "step": 3}
        )
        assert path.is_file()
        assert list(tmp_path.glob(".*.tmp-*")) == []

    def test_rng_state_restored_on_load(self, tmp_path: Path) -> None:
        torch.manual_seed(1234)
        expected = torch.randn(4)
        torch.manual_seed(1234)  # rewind so the checkpoint saves the pre-randn state
        save_solver_checkpoint(
            tmp_path / "solver.ckpt", {"populations": torch.ones(9, 4, 6), "step": 0}
        )
        torch.randn(64)  # advance the generator past the saved state
        load_solver_checkpoint(tmp_path / "solver.ckpt")
        assert torch.equal(torch.randn(4), expected)

    def test_state_dict_requires_populations_and_step(self, tmp_path: Path) -> None:
        with pytest.raises(CheckpointError, match="populations"):
            save_solver_checkpoint(tmp_path / "bad.ckpt", {"step": 1})
        with pytest.raises(CheckpointError, match="step"):
            save_solver_checkpoint(tmp_path / "bad.ckpt", {"populations": torch.ones(9, 4, 6)})


class TestFailClosed:
    def _save(self, tmp_path: Path) -> Path:
        return save_solver_checkpoint(
            tmp_path / "solver.ckpt",
            {
                "populations": torch.randn(19, 4, 5, 6),
                "step": 7,
                "lattice": "D3Q19",
                "grid": (4, 5, 6),
                "q": 19,
                "tau": 0.6,
            },
        )

    def _rewrite(self, path: Path, mutate) -> None:
        envelope = torch.load(path, map_location="cpu", weights_only=True)
        mutate(envelope)
        torch.save(envelope, path)

    def test_truncated_write_raises(self, tmp_path: Path) -> None:
        path = self._save(tmp_path)
        data = path.read_bytes()
        path.write_bytes(data[: len(data) // 2])
        with pytest.raises(CheckpointError, match="could not be decoded"):
            load_solver_checkpoint(path)

    def test_garbage_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "solver.ckpt"
        path.write_bytes(b"definitely not a torch archive")
        with pytest.raises(CheckpointError, match="could not be decoded"):
            load_solver_checkpoint(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CheckpointError, match="not found"):
            load_solver_checkpoint(tmp_path / "absent.ckpt")

    def test_digest_mismatch_raises(self, tmp_path: Path) -> None:
        path = self._save(tmp_path)
        self._rewrite(path, lambda env: env["state"]["populations"].add_(1.0))
        with pytest.raises(CheckpointError, match="integrity digest"):
            load_solver_checkpoint(path)

    def test_unsupported_format_version_raises(self, tmp_path: Path) -> None:
        path = self._save(tmp_path)
        self._rewrite(path, lambda env: env.__setitem__("format_version", 99))
        with pytest.raises(CheckpointError, match="format_version"):
            load_solver_checkpoint(path)

    def test_missing_format_tag_raises(self, tmp_path: Path) -> None:
        path = self._save(tmp_path)
        self._rewrite(path, lambda env: env.__setitem__("format", "something-else"))
        with pytest.raises(CheckpointError, match="not a unified solver checkpoint"):
            load_solver_checkpoint(path)

    def test_expected_identity_mismatch_raises(self, tmp_path: Path) -> None:
        path = self._save(tmp_path)
        with pytest.raises(CheckpointError, match="lattice mismatch"):
            load_solver_checkpoint(path, expected_lattice="D3Q27")
        with pytest.raises(CheckpointError, match="grid mismatch"):
            load_solver_checkpoint(path, expected_grid=(8, 5, 6))
        with pytest.raises(CheckpointError, match="Q mismatch"):
            load_solver_checkpoint(path, expected_q=27)

    def test_adapter_solver_mismatch_raises(self, tmp_path: Path) -> None:
        path = self._save(tmp_path)
        checkpoint = load_solver_checkpoint(path)
        wrong_solver = LBMStepExecutor(
            "D3Q27",
            collide_fn="bgk",
            device=torch.device("cpu"),
            nx=6,
            ny=5,
            nz=4,
            tau=0.6,
        )
        with pytest.raises(CheckpointError, match="does not match solver"):
            eager_load_state_dict(wrong_solver, checkpoint.state)
        wrong_tau = LBMStepExecutor(
            "D3Q19",
            collide_fn="bgk",
            device=torch.device("cpu"),
            nx=6,
            ny=5,
            nz=4,
            tau=0.65,
        )
        with pytest.raises(CheckpointError, match="tau"):
            eager_load_state_dict(wrong_tau, checkpoint.state)


class TestTritonFusedAdapter:
    def _stub_solver(self, **overrides):
        base = {
            "lattice": None,
            "nz": 8,
            "ny": 8,
            "nx": 8,
            "tau": 0.6,
            "dtype": torch.float32,
            "device": torch.device("cpu"),
            "precision_policy": None,
            "_report_step": 0,
        }
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_state_dict_records_precision_and_dtype(self) -> None:
        solver = self._stub_solver(
            precision_policy=SimpleNamespace(name="FP32FP16"), dtype=torch.float16
        )
        f = torch.zeros(19, 8, 8, 8, dtype=torch.float16)
        state = triton_fused_state_dict(solver, f, step=4)
        assert state["lattice"] == "D3Q19"
        assert state["q"] == 19
        assert state["grid"] == (8, 8, 8)
        assert state["precision_policy"] == "FP32FP16"
        assert state["storage_dtype"] == "torch.float16"
        assert state["step"] == 4

    def test_policy_mismatch_fails_closed(self) -> None:
        solver = self._stub_solver(precision_policy=None)
        state = {
            "populations": torch.zeros(19, 8, 8, 8),
            "step": 1,
            "lattice": "D3Q19",
            "grid": (8, 8, 8),
            "q": 19,
            "tau": 0.6,
            "precision_policy": "FP32FP16",
            "storage_dtype": "torch.float16",
        }
        with pytest.raises(CheckpointError, match="precision policy"):
            triton_fused_load_state_dict(solver, state)

    def test_dtype_mismatch_fails_closed(self) -> None:
        solver = self._stub_solver(dtype=torch.float32)
        state = {
            "populations": torch.zeros(19, 8, 8, 8),
            "step": 1,
            "lattice": "D3Q19",
            "grid": (8, 8, 8),
            "q": 19,
            "tau": 0.6,
            "precision_policy": None,
            "storage_dtype": "torch.float16",
        }
        with pytest.raises(CheckpointError, match="storage dtype"):
            triton_fused_load_state_dict(solver, state)

    @pytest.mark.skipif(torch.cuda.device_count() < 1, reason="requires CUDA + triton")
    def test_round_trip_bitwise_on_gpu(self, tmp_path: Path) -> None:
        from tensorlbm.triton_fused import TritonFusedSolver3D, is_available

        if not is_available():
            pytest.skip("triton_fused not available on this host")
        device = "cuda:0"
        nz = ny = nx = 8
        rho = torch.full((nz, ny, nx), 1.0 + 1e-3, device=device)
        u = torch.full((nz, ny, nx), 0.05, device=device)
        zero = torch.zeros_like(rho)
        f0 = equilibrium3d(rho.cpu(), u.cpu(), zero.cpu(), zero.cpu()).to(device)

        reference = TritonFusedSolver3D(nz, ny, nx, tau=0.6, device=device)
        f_ref = f0.clone()
        for _ in range(6):
            f_ref = reference.step(f_ref)

        solver = TritonFusedSolver3D(nz, ny, nx, tau=0.6, device=device)
        f_a = f0.clone()
        for _ in range(3):
            f_a = solver.step(f_a)
        state = triton_fused_state_dict(solver, f_a, step=3)
        path = save_solver_checkpoint(tmp_path / "triton.ckpt", state)

        restored = TritonFusedSolver3D(nz, ny, nx, tau=0.6, device=device)
        checkpoint = load_solver_checkpoint(path)
        f_b = triton_fused_load_state_dict(restored, checkpoint.state)
        assert restored._report_step == 3
        for _ in range(3):
            f_b = restored.step(f_b)
        assert torch.equal(f_b, f_ref)


class TestShardedStateEquivalence:
    """Unified API vs the multi-GPU rank-file convention.

    The torchrun restart exactness of
    :class:`tensorlbm.multi_gpu.D3Q19GlooTransport` is covered by
    ``tests/test_multi_gpu_d3q19_checkpoint.py`` (needs a 3-rank gloo
    job); this is the CPU-runnable equivalence check: each rank's owned
    slab + step round-trips bit-exactly through the unified API, so the
    two conventions stay interchangeable per rank.
    """

    def test_rank_slabs_round_trip_and_reassemble(self, tmp_path: Path) -> None:
        cuts = (0, 3, 6, 10)
        world = len(cuts) - 1
        torch.manual_seed(20260714)
        full = torch.randn(19, 3, 4, 10, dtype=torch.float64)
        step = 10
        restored = []
        for rank in range(world):
            owned = full[..., cuts[rank] : cuts[rank + 1]].clone()
            path = save_solver_checkpoint(
                tmp_path / f"rank-{rank}.pt",
                {
                    "populations": owned,
                    "step": step,
                    "lattice": "D3Q19",
                    "grid": (3, 4, owned.shape[-1]),
                    "q": 19,
                },
                metadata={"rank": rank, "world_size": world},
            )
            checkpoint = load_solver_checkpoint(path, expected_q=19)
            assert checkpoint.metadata["rank"] == rank
            assert checkpoint.metadata["world_size"] == world
            assert checkpoint.step == step
            assert torch.equal(checkpoint.populations, owned)
            restored.append(checkpoint.populations)
        assert torch.equal(torch.cat(restored, dim=-1), full)


class TestCaseCheckpointHelpers:
    IDENTITY = {"scan_id": "s1", "point_id": "p0000", "plan_digest": "d1"}

    def test_identity_gates_resume(self, tmp_path: Path) -> None:
        save_case_checkpoint(
            tmp_path,
            f=torch.ones(19, 4, 5, 6),
            step=6,
            lattice="D3Q19",
            grid=(4, 5, 6),
            identity=self.IDENTITY,
        )
        loaded = load_case_checkpoint(tmp_path, identity=self.IDENTITY)
        assert loaded is not None
        assert loaded.step == 6
        assert loaded.populations.shape == (19, 4, 5, 6)
        mismatched = dict(self.IDENTITY, point_id="p9999")
        assert load_case_checkpoint(tmp_path, identity=mismatched) is None

    def test_corrupt_or_missing_returns_none(self, tmp_path: Path) -> None:
        assert load_case_checkpoint(tmp_path, identity=self.IDENTITY) is None
        save_case_checkpoint(
            tmp_path,
            f=torch.ones(19, 4, 5, 6),
            step=6,
            lattice="D3Q19",
            grid=(4, 5, 6),
            identity=self.IDENTITY,
        )
        path = case_checkpoint_path(tmp_path)
        path.write_bytes(path.read_bytes()[:64])
        assert load_case_checkpoint(tmp_path, identity=self.IDENTITY) is None


class TestCampaignResumeHook:
    def _patch_dispatch(self, monkeypatch, crash_step: int | None = 15) -> list[int]:
        """Patch reporters.dispatch: crash once at *crash_step*, record steps.

        The wrapper stays active for the whole test (one-shot crash), so
        *steps* accumulates every executed step across all executor runs.
        """
        import tensorlbm.reporters as reporters_module

        real_dispatch = reporters_module.dispatch
        crashed = {"once": False}
        steps_seen: list[int] = []

        def dispatch_wrapper(ctx, reporters):
            if crash_step is not None and ctx.step == crash_step and not crashed["once"]:
                crashed["once"] = True
                raise RuntimeError("simulated interruption")
            steps_seen.append(ctx.step)
            return real_dispatch(ctx, reporters)

        monkeypatch.setattr(reporters_module, "dispatch", dispatch_wrapper)
        return steps_seen

    def test_scan_without_checkpointing_writes_no_case_state(self, tmp_path: Path) -> None:
        plan = _tiny_campaign_plan()
        summary = ScanExecutor(plan, tmp_path, serial_device="cpu").run()
        assert summary["n_failed"] == 0
        assert summary["n_completed"] == 2
        assert list(tmp_path.glob("points/*/case-checkpoint.pt")) == []

    def test_interrupted_scan_resumes_mid_point(self, tmp_path: Path, monkeypatch) -> None:
        plan = _tiny_campaign_plan()
        steps_seen = self._patch_dispatch(monkeypatch, crash_step=15)
        first = ScanExecutor(plan, tmp_path, serial_device="cpu", checkpoint_every=10).run()
        assert first["n_failed"] == 1

        point_dir = tmp_path / "points" / "p0000"
        identity = {
            "scan_id": plan.scan_id,
            "point_id": "p0000",
            "run_id": plan.points[0].run_id,
            "case": plan.case,
            "plan_digest": plan.plan_digest(),
        }
        prior = load_case_checkpoint(point_dir, identity=identity)
        assert prior is not None
        assert prior.step == 10

        second = ScanExecutor(plan, tmp_path, serial_device="cpu", checkpoint_every=10).run(
            resume=True
        )
        assert second["n_failed"] == 0
        assert second["n_skipped"] == 1  # p0001 already complete
        assert second["n_completed"] == 2
        # First pass: p0000 ran 1..14 then crashed, p0001 ran 1..30.
        # Second pass: p0001 skipped, p0000 resumed from its step-10 checkpoint.
        assert steps_seen[:14] == list(range(1, 15))
        assert steps_seen[14:44] == list(range(1, 31))
        assert steps_seen[44:] == list(range(11, 31))
        # completed points clean up their in-run state
        assert list(tmp_path.glob("points/*/case-checkpoint.pt")) == []
        status = json.loads((point_dir / "status.json").read_text(encoding="utf-8"))
        assert status["status"] == "completed"
        assert status["exported_steps"] == [10, 20, 30]
        assert len(status["product_ids"]) == 3
        assert second["dataset"]["n_samples"] == 6

    def test_identity_mismatch_restarts_point_from_zero(self, tmp_path: Path, monkeypatch) -> None:
        plan = _tiny_campaign_plan()
        steps_seen = self._patch_dispatch(monkeypatch, crash_step=15)
        first = ScanExecutor(plan, tmp_path, serial_device="cpu", checkpoint_every=10).run()
        assert first["n_failed"] == 1

        # Tamper the checkpoint as if it belonged to another sweep.
        path = case_checkpoint_path(tmp_path / "points" / "p0000")
        assert path.is_file()
        envelope = torch.load(path, map_location="cpu", weights_only=True)
        envelope["metadata"]["identity"]["scan_id"] = "some-other-scan"
        torch.save(envelope, path)

        second = ScanExecutor(plan, tmp_path, serial_device="cpu", checkpoint_every=10).run(
            resume=True
        )
        assert second["n_failed"] == 0
        # p0000 restarted from step 0: step 5 ran once more on top of
        # the two first-pass executions (p0000 pre-crash + p0001).
        assert steps_seen.count(5) == 3
        assert steps_seen[-30:] == list(range(1, 31))
        assert second["dataset"]["n_samples"] == 6

    def test_resumed_fields_match_uninterrupted_run(self, tmp_path_factory, monkeypatch) -> None:
        reference_root = tmp_path_factory.mktemp("reference")
        broken_root = tmp_path_factory.mktemp("broken")
        reference_plan = _tiny_campaign_plan()
        broken_plan = _tiny_campaign_plan()

        ScanExecutor(reference_plan, reference_root, serial_device="cpu").run()

        self._patch_dispatch(monkeypatch, crash_step=15)
        first = ScanExecutor(
            broken_plan, broken_root, serial_device="cpu", checkpoint_every=10
        ).run()
        assert first["n_failed"] == 1
        second = ScanExecutor(
            broken_plan, broken_root, serial_device="cpu", checkpoint_every=10
        ).run(resume=True)
        assert second["n_failed"] == 0

        for point_id in ("p0000", "p0001"):
            ref_arrays, _ = read_snapshot(reference_root / "points" / point_id / "fields.h5", 20)
            new_arrays, _ = read_snapshot(broken_root / "points" / point_id / "fields.h5", 20)
            assert set(ref_arrays) == set(new_arrays)
            for name, ref in ref_arrays.items():
                assert np.array_equal(ref, new_arrays[name]), name
