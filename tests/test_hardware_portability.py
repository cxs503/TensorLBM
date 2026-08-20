"""Device-agnostic portability gates.

These tests pin TensorLBM's portability contract:

1. The **eager solver path** (``solver3d`` gather/roll streaming, BGK/TRT
   collision) runs on plain CPU torch — wherever torch runs, the baseline
   simulation path follows.
2. The **core data chain** (``solver_export`` → catalog → load) works
   without any accelerator, so CI machines with no GPU still gate the
   cold-path contracts.
3. **No bare ``.cuda()``** remains anywhere under ``src/tensorlbm`` —
   device placement must go through ``utils.resolve_device`` /
   ``ops.to_device`` / ``tensorlbm.hardware`` so non-CUDA backends
   (NPU/MLU/SDAA/MUSA) are never silently locked out.
4. The ``tensorlbm.hardware`` capability probe produces serialisable
   profiles and actionable ``require()`` failures with degradation advice.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tensorlbm import hardware
from tensorlbm.data.catalog import FieldDataCatalog
from tensorlbm.data.solver_export import (
    load_product,
    load_product_arrays,
    register_product,
    save_fields_hdf5,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _REPO_ROOT / "src" / "tensorlbm"

_CODE_SHA = "b" * 40


# ---------------------------------------------------------------------------
# Gate 1: eager solver path on CPU
# ---------------------------------------------------------------------------

def test_stream3d_gather_and_roll_agree_on_cpu() -> None:
    """The two eager streaming kernels must be numerically identical."""
    from tensorlbm.solver3d import stream3d, stream3d_roll

    generator = torch.Generator(device="cpu").manual_seed(7)
    f = torch.rand((19, 6, 8, 10), generator=generator, dtype=torch.float32) * 0.05 + 1e-4
    gathered = stream3d(f.clone())
    rolled = stream3d_roll(f.clone())
    assert gathered.shape == f.shape
    assert rolled.shape == f.shape
    torch.testing.assert_close(gathered, rolled, rtol=0.0, atol=0.0)
    # Periodic streaming conserves total populations.
    assert torch.isclose(gathered.sum(), f.sum(), rtol=1e-6)


def test_collide_stream_eager_step_on_cpu() -> None:
    """One full eager BGK step (collide → stream) stays finite and mass-conserving."""
    from tensorlbm.d3q19 import equilibrium3d
    from tensorlbm.solver3d import collide_bgk3d, stream3d

    nz, ny, nx = 5, 7, 9
    zeros = torch.zeros(nz, ny, nx)
    rho = torch.ones(nz, ny, nx)
    ux = torch.full((nz, ny, nx), 0.05)  # uniform streamwise flow
    f = equilibrium3d(rho, ux, zeros, zeros)
    initial_mass = float(f.sum())
    for _ in range(3):
        f = stream3d(collide_bgk3d(f, tau=0.8))
    assert torch.isfinite(f).all()
    assert float(f.sum()) == pytest.approx(initial_mass, rel=1e-4)


def test_collide_bgk3d_preserves_mass_on_cpu() -> None:
    from tensorlbm.d3q19 import equilibrium3d
    from tensorlbm.solver3d import collide_bgk3d

    zeros = torch.zeros(4, 4, 4)
    rho = torch.ones(4, 4, 4)
    uy = torch.full((4, 4, 4), 0.03)
    f = equilibrium3d(rho, zeros, uy, zeros)
    collided = collide_bgk3d(f, tau=0.6)
    assert torch.isfinite(collided).all()
    assert float(collided.sum()) == pytest.approx(float(f.sum()), rel=1e-5)


# ---------------------------------------------------------------------------
# Gate 2: core data chain on CPU
# ---------------------------------------------------------------------------

@pytest.fixture
def registered_product(tmp_path: Path) -> tuple[FieldDataCatalog, str, Path]:
    rng = np.random.default_rng(0)
    shape = (4, 6, 8)
    fields = {
        "rho": (1.0 + 0.001 * rng.standard_normal(shape)).astype(np.float32),
        "ux": (0.05 * rng.standard_normal(shape)).astype(np.float32),
        "uy": (0.01 * rng.standard_normal(shape)).astype(np.float32),
        "uz": (0.01 * rng.standard_normal(shape)).astype(np.float32),
    }
    h5_path = save_fields_hdf5(
        tmp_path / "run.h5",
        fields,
        attrs={"step": 10, "run_id": "portability-run", "case": "portability"},
    )
    catalog = FieldDataCatalog.open(tmp_path / "catalog.db")
    product_id = register_product(
        catalog,
        h5_path,
        {
            "run_id": "portability-run",
            "case": "portability",
            "step": 10,
            "code_sha": _CODE_SHA,
            "device": "cpu",
        },
        blob_root=tmp_path / "blobs",
    )
    return catalog, product_id, tmp_path


def test_data_chain_export_catalog_load_on_cpu(
    registered_product: tuple[FieldDataCatalog, str, Path],
) -> None:
    """save_fields_hdf5 → register_product → load_product with no accelerator."""
    catalog, product_id, _ = registered_product
    asset = catalog.get_asset(product_id)
    assert asset is not None and asset.kind == "field_product"

    product = load_product(catalog, product_id)
    arrays = load_product_arrays(product)
    assert set(arrays) >= {"velocity", "rho"}
    assert arrays["velocity"].shape[-1] == 3
    assert np.isfinite(arrays["velocity"]).all()


def test_data_chain_quality_and_lineage_recorded(
    registered_product: tuple[FieldDataCatalog, str, Path],
) -> None:
    catalog, product_id, _ = registered_product
    reports = catalog.get_quality_reports(product_id)
    assert reports, "solver_export registration must record quality checks"
    lineage = catalog.get_lineage(product_id)
    assert any(record.relation_type == "derived_from" for record in lineage)


# ---------------------------------------------------------------------------
# Gate 3: static scan — no bare .cuda() anywhere in src/tensorlbm
# ---------------------------------------------------------------------------

def test_no_bare_cuda_calls_in_src() -> None:
    """Device placement must be device-parameter driven, not CUDA literal.

    ``.cuda()`` without an argument pins placement to CUDA and silently
    breaks every other torch backend.  Allowed escape hatches are
    ``ops.to_device``-style helpers that resolve the method from the device
    string (see ``backends/paddle_backend._place`` and
    ``tensorlbm.utils.resolve_device``).
    """
    offenders: list[str] = []
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            code = line.split("#", 1)[0]
            if ".cuda()" in code:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{line_number}: {line.strip()}")
    assert not offenders, "bare .cuda() found in src/tensorlbm:\n" + "\n".join(offenders)


# ---------------------------------------------------------------------------
# Gate 4: hardware capability probe
# ---------------------------------------------------------------------------

def test_probe_reports_cpu_and_serialises() -> None:
    profile = hardware.probe(refresh=True)
    assert profile.has_backend("cpu")
    assert profile.backend("cpu") is not None and profile.backend("cpu").available

    snapshot = profile.to_dict()
    # JSON-safe: benchmark_observability embeds this into run_status.json.
    json.dumps(snapshot)
    assert snapshot["torch_version"] == torch.__version__
    names = {entry["name"] for entry in snapshot["backends"]}
    assert {"cpu", "cuda", "npu", "mlu", "sdaa", "musa", "mps"} <= names
    for entry in snapshot["backends"]:
        assert isinstance(entry["available"], bool)
        assert isinstance(entry["device_count"], int)


def test_probe_is_cached_and_refreshable() -> None:
    first = hardware.probe()
    second = hardware.probe()
    assert first is second
    refreshed = hardware.probe(refresh=True)
    assert refreshed is not first


def test_require_missing_backend_raises_with_advice() -> None:
    profile = hardware.probe(refresh=True)
    synthetic = hardware.HardwareProfile(
        backends=tuple(
            hardware.BackendInfo(
                name=info.name,
                available=False,
                device_count=0,
                plugin=info.plugin,
                note=info.note,
            )
            if info.name != "cpu"
            else info
            for info in profile.backends
        ),
        collectives=profile.collectives,
        fp16_storage=False,
        bf16_storage=False,
        triton_available=False,
        triton_version=None,
        default_device="cpu",
        torch_version=profile.torch_version,
        python_version=profile.python_version,
        hostname=profile.hostname,
    )
    with pytest.raises(hardware.HardwareCapabilityError) as excinfo:
        hardware.require("cuda", profile=synthetic)
    assert "eager" in excinfo.value.advice
    assert excinfo.value.capability == "cuda"


def test_require_fp16_degrades_to_fp32fp32() -> None:
    profile = hardware.probe(refresh=True)
    synthetic = hardware.HardwareProfile(
        backends=profile.backends,
        collectives=profile.collectives,
        fp16_storage=False,
        bf16_storage=profile.bf16_storage,
        triton_available=profile.triton_available,
        triton_version=profile.triton_version,
        default_device=profile.default_device,
        torch_version=profile.torch_version,
        python_version=profile.python_version,
        hostname=profile.hostname,
    )
    with pytest.raises(hardware.HardwareCapabilityError) as excinfo:
        hardware.require("fp16_storage", profile=synthetic)
    assert "FP32FP32" in str(excinfo.value)


def test_require_triton_degrades_to_eager_path() -> None:
    profile = hardware.probe(refresh=True)
    synthetic = hardware.HardwareProfile(
        backends=profile.backends,
        collectives=profile.collectives,
        fp16_storage=profile.fp16_storage,
        bf16_storage=profile.bf16_storage,
        triton_available=False,
        triton_version=None,
        default_device=profile.default_device,
        torch_version=profile.torch_version,
        python_version=profile.python_version,
        hostname=profile.hostname,
    )
    with pytest.raises(hardware.HardwareCapabilityError) as excinfo:
        hardware.require("triton", profile=synthetic)
    assert "eager" in str(excinfo.value)
    assert "stream3d" in str(excinfo.value)


def test_require_unknown_capability_is_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown capability"):
        hardware.require("warp_drive")


def test_require_cpu_always_passes() -> None:
    profile = hardware.require("cpu")
    assert profile.has_backend("cpu")


# ---------------------------------------------------------------------------
# Gate 5: observability records carry the hardware profile
# ---------------------------------------------------------------------------

def test_resolve_benchmark_device_embeds_hardware_profile() -> None:
    from tensorlbm.benchmark_observability import resolve_benchmark_device

    metadata = resolve_benchmark_device("cpu")
    snapshot = metadata["hardware_profile"]
    assert snapshot is not None
    names = {entry["name"] for entry in snapshot["backends"]}
    assert "cpu" in names
    json.dumps(snapshot)


def test_benchmark_reporter_status_contains_hardware(tmp_path: Path) -> None:
    from tensorlbm.benchmark_observability import BenchmarkReporter

    reporter = BenchmarkReporter(tmp_path, "portability_gate", 1, {"requested": "cpu"})
    reporter.start()
    reporter.finish(1, "COMPLETED", None, {"metric": 1.0})
    status = json.loads((tmp_path / "run_status.json").read_text())
    assert status["hardware"]["default_device"]
    assert any(
        entry["name"] == "cpu" for entry in status["hardware"]["backends"]
    )
