"""NumPy 2 compatibility regressions for post-processing endpoints."""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app.backend.routers import postprocess


def test_acoustics_spectrum_integrates_band_power_without_np_trapz(
    monkeypatch, tmp_path: Path,
) -> None:
    """The real endpoint must retain finite, positive band power under NumPy 2."""
    csv_path = tmp_path / "fwh_acoustic.csv"
    samples = np.sin(np.linspace(0.0, 32.0 * np.pi, 512, endpoint=False))
    csv_path.write_text(
        "p_prime\n" + "\n".join(str(value) for value in samples) + "\n",
        encoding="utf-8",
    )
    job = SimpleNamespace(
        status=SimpleNamespace(value="completed"),
        output_dir=tmp_path,
    )
    monkeypatch.setattr(postprocess.job_manager, "get_job", lambda job_id: job)
    monkeypatch.delattr(np, "trapz", raising=False)

    result = asyncio.run(
        postprocess.acoustics_spectrum(
            "job-1", fs=1_000.0, window="hann", nperseg=256, p_ref=2e-5
        )
    )

    assert result["n_samples"] == len(samples)
    assert np.isfinite(result["p_rms"])
    assert result["p_rms"] > 0.0
    assert result["third_octave_bands"]
    assert all(
        np.isfinite(band["spl_db"]) for band in result["third_octave_bands"]
    )
