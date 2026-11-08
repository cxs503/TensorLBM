"""The batch-mean recorder must be observation-only.

``WallResolvedChannel3DConfig.profile_batch_count`` / ``wall_exchange_series_stride``
exist so the statistical-error estimator has the raw material it needs
(consecutive batch means + a correlated wall-shear time series).  Two properties
are non-negotiable and are pinned here:

1. switching the recorder **on does not change the trajectory** (identical
   profiles for the same seed, i.e. the legacy default stays bit-for-bit);
2. the record is *complete* -- batch means reconstruct the global mean profile
   to round-off, the batch sample counts sum to ``profile_samples``, and the
   recorded wall-shear series are positive, finite and of the expected length.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.boundary.wall_resolved_channel3d import (
    WallResolvedChannel3DConfig,
    run_wall_resolved_channel3d,
)
from tensorlbm.validation.statistical_error import (
    batch_means_summary,
    half_window_max_diff,
    integral_time,
)


def _config(tmp_path, name, **overrides):
    base = dict(
        nx=16,
        ny=14,
        nz=12,
        re_tau=180.0,
        u_tau=0.005,
        steps=160,
        warmup_steps=80,
        sample_interval=5,
        report_interval=120,
        checkpoint_interval=10**9,
        collision_model="cumulant",
        compile_natural_kbc=False,
        initialization_mode="spectral_solenoidal",
        perturbation_fraction=0.05,
        random_noise_fraction=0.0,
        spectral_mode_count=8,
        spectral_max_wavenumber=2,
        minimum_statistics_eddy_turnovers=0.01,
        stationarity_window_eddy_turnovers=0.2,
        seed=424242,
        device="cpu",
        momentum_exchange_diagnostic=True,
        output=tmp_path / f"{name}.json",
        checkpoint=tmp_path / f"{name}.ckpt",
    )
    base.update(overrides)
    return WallResolvedChannel3DConfig(**base)


def test_batch_recorder_is_observation_only(tmp_path):
    torch.set_num_threads(1)
    legacy = run_wall_resolved_channel3d(_config(tmp_path, "legacy"))
    recorded = run_wall_resolved_channel3d(
        _config(
            tmp_path,
            "recorded",
            profile_batch_count=8,
            wall_exchange_series_stride=1,
        )
    )
    # (1) the physics is untouched
    assert (
        legacy["statistics"]["mean_velocity_profile"]
        == recorded["statistics"]["mean_velocity_profile"]
    )
    assert (
        legacy["statistics"]["reynolds_stress_profiles_uu_vv_ww_uv"]
        == recorded["statistics"]["reynolds_stress_profiles_uu_vv_ww_uv"]
    )
    # (2) legacy emits an empty record, the recorder emits a complete one
    assert legacy["statistics"]["profile_batch_moment_means"] == []
    assert legacy["wall_exchange"]["shear_series"] == []

    stats = recorded["statistics"]
    batches = stats["profile_batch_moment_means"]
    counts = stats["profile_batch_counts"]
    assert len(batches) == 8 and len(counts) == 8
    assert sum(counts) == stats["profile_samples"]
    assert all(count > 0 for count in counts)

    # batch means reproduce the global mean profile (u-velocity row 0)
    u_batches = [row[0] for row in batches]
    reconstructed = [
        sum(count * profile[row] for count, profile in zip(counts, u_batches))
        / sum(counts)
        for row in range(len(u_batches[0]))
    ]
    assert reconstructed == pytest.approx(
        recorded["statistics"]["mean_velocity_profile"], rel=0.0, abs=1.0e-12
    )

    # the estimators consume the record without further massaging
    ci = batch_means_summary(u_batches)
    assert ci["n_batches"] == 8
    assert ci["max_ci_half"] >= 0.0
    gate = half_window_max_diff(u_batches)
    assert math.isfinite(gate["half_window_max_diff"])

    series = recorded["wall_exchange"]["shear_series"]
    assert recorded["wall_exchange"]["shear_series_stride"] == 1
    assert len(series) == stats["statistics_steps"]
    assert all(math.isfinite(value) for value in series)
    # a one-eddy-turnover window is too short for a meaningful T_int at this
    # size, but the estimator must still return a finite, positive value
    t_int = integral_time(series)
    assert t_int["t_int_samples"] > 0.0
    assert t_int["acf_window"] >= 1


def test_recorder_validation_rejects_negative_counts(tmp_path):
    with pytest.raises(ValueError):
        _config(tmp_path, "bad", profile_batch_count=-1).validate()
    with pytest.raises(ValueError):
        _config(tmp_path, "bad2", wall_exchange_series_stride=-3).validate()

def test_resume_reproduces_the_uninterrupted_record(tmp_path):
    """A checkpoint restart must be bit-for-bit equivalent to a straight run.

    The campaign relies on this: the compute nodes abort long runs with glibc
    heap corruption, so every arm is checkpointed and restarted.  A restart is
    only legitimate if the *statistics* (mean profile, batch means, wall-shear
    series) are identical to the uninterrupted run -- the batch partition is a
    pure function of the step index, so it must be.
    """
    torch.set_num_threads(1)
    common = dict(
        steps=200,
        warmup_steps=100,
        profile_batch_count=4,
        wall_exchange_series_stride=1,
    )
    straight = run_wall_resolved_channel3d(_config(tmp_path, "straight", **common))

    # A first segment that stops before any sample is collected (step 101 is not
    # a multiple of sample_interval = 5), so the resumed run owns the whole
    # averaging window and the batch partition is the final one.
    # the segment collects no sample, so the kernel fails closed after writing
    # the checkpoint -- exactly the state a crashed long run leaves behind
    with pytest.raises(RuntimeError, match="no profile samples"):
        run_wall_resolved_channel3d(
            _config(
                tmp_path,
                "resumed",
                steps=101,
                warmup_steps=100,
                profile_batch_count=4,
                wall_exchange_series_stride=1,
                checkpoint_interval=100,
            )
        )
    resumed = run_wall_resolved_channel3d(
        _config(
            tmp_path,
            "resumed",
            resume=True,
            checkpoint_interval=100,
            **common,
        )
    )
    assert (
        resumed["statistics"]["profile_batch_moment_means"]
        == straight["statistics"]["profile_batch_moment_means"]
    )
    assert (
        resumed["statistics"]["profile_batch_counts"]
        == straight["statistics"]["profile_batch_counts"]
    )
    assert resumed["statistics"]["mean_velocity_profile"] == straight["statistics"][
        "mean_velocity_profile"
    ]
    assert resumed["wall_exchange"]["shear_series"] == straight["wall_exchange"][
        "shear_series"
    ]
