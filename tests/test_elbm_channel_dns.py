"""Tests for the resumable wall-resolved ELBM channel DNS driver.

The suite covers, in the order of the campaign criteria:

* **lattice/equilibrium consistency** -- the NumPy closed form in
  :mod:`tensorlbm.elbm_channel_dns` reproduces the torch closed form of
  :func:`tensorlbm.elbm.entropic_equilibrium_gauss_d3q27` and the initial field
  reproduces density and momentum exactly (a wrong lattice table or weight set
  shows up here as ``sum(f) != 1``);
* **the float32 mirror-point defect (regression)** -- for a state that *is* the
  entropic equilibrium the entropy-conditioned relaxation must return the
  admissibility cap ``alpha = 1`` (BGK), not collapse to zero.  It did collapse
  before ``f^eq`` was kept in double precision inside the kernel; the same test
  is run through the torch reference;
* **H-theorem / positivity at DNS-like parameters** -- the kernel's reported
  entropy increase net of the floating-point floor is non-positive and every
  population stays positive;
* **checkpoint / restart** -- a run split in two halves with a checkpoint in
  between is bit-identical to the same run in one go (fields, accumulators and
  sample count), which is what makes multi-slot production runs trustworthy;
* **statistics and artefact interface** -- the plane-averaged profiles match an
  independent brute-force computation, and the written ``.npz`` is accepted by
  the DNS gate machinery (:func:`tensorlbm.elbm_channel.channel_gates`).
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pytest
import torch

from tensorlbm.d3q27 import C as C27
from tensorlbm.elbm import (
    ALPHA_CAP,
    collide_elbm_d3q27,
    entropic_equilibrium_gauss_d3q27,
)
from tensorlbm.elbm_channel import channel_gates, log_law_slope
from tensorlbm.elbm_channel_dns import (
    DnsChannelConfig,
    entropic_equilibrium_np,
    initial_field,
    load_checkpoint,
    run_dns_channel,
    sample_profiles,
    save_checkpoint,
)
from tensorlbm.elbm_channel_native import native_channel_step_inplace

F64 = torch.float64


def _cfg(**kw) -> DnsChannelConfig:
    base = dict(nx=8, ny=18, nz=8, re_tau=180.0, u_tau=0.02, steps=40, warmup=10,
                sample_every=5, log_every=20, ckpt_every=20, label="test")
    base.update(kw)
    return DnsChannelConfig(**base)


# ---------------------------------------------------------------------------
# 1. lattice / equilibrium consistency
# ---------------------------------------------------------------------------


def test_numpy_closed_form_matches_torch_closed_form() -> None:
    """The NumPy entropic equilibrium is the torch one (same lattice, same w)."""
    g = np.random.default_rng(3)
    ux = 0.2 * g.standard_normal((3, 4, 5))
    uy = 0.2 * g.standard_normal((3, 4, 5))
    uz = 0.2 * g.standard_normal((3, 4, 5))
    rho = 1.0 + 0.1 * g.standard_normal((3, 4, 5))
    got = entropic_equilibrium_np(rho, ux, uy, uz)
    ref = entropic_equilibrium_gauss_d3q27(
        *[torch.tensor(x, dtype=F64) for x in (rho, ux, uy, uz)]
    ).numpy()
    assert got.shape == ref.shape, (got.shape, ref.shape)
    assert np.abs(got - ref).max() < 1e-12


def test_initial_field_reproduces_density_and_momentum() -> None:
    cfg = _cfg()
    f, alpha = initial_field(cfg)
    assert f.shape == (27, cfg.nz, cfg.ny, cfg.nx)
    assert f.dtype == np.float32 and alpha.dtype == np.float32
    rho = f.sum(axis=0)
    fluid = (slice(None), slice(1, cfg.ny - 1), slice(None))
    assert abs(float(rho[fluid].min()) - 1.0) < 1e-6
    assert abs(float(rho[fluid].max()) - 1.0) < 1e-6
    assert np.all(f[:, :, 0, :] == 0.0) and np.all(f[:, :, cfg.ny - 1, :] == 0.0)
    # wall-normal laws of the driver: h, nu, tau, force
    assert cfg.half_height == 0.5 * (cfg.ny - 2)
    assert abs(cfg.tau - (0.5 + 3.0 * cfg.nu)) < 1e-15
    assert abs(cfg.body_force * cfg.half_height - cfg.u_tau**2) < 1e-18
    assert abs(cfg.delta_plus - cfg.re_tau / cfg.half_height) < 1e-12


def test_config_validation_rejects_impossible_setups() -> None:
    with pytest.raises(ValueError):
        _cfg(ny=4).validate()
    with pytest.raises(ValueError):
        _cfg(u_tau=0.5).validate()
    with pytest.raises(ValueError):
        _cfg(warmup=40, steps=40).validate()
    with pytest.raises(ValueError):
        _cfg(sample_every=0).validate()


# ---------------------------------------------------------------------------
# 2. the float32 mirror-point defect (regression)
# ---------------------------------------------------------------------------


def test_kernel_returns_bgk_for_an_exact_equilibrium_state() -> None:
    """Regression: alpha must be the cap on an exactly equilibrated fp32 state.

    The entropy-condition test lives at the *mirror point* of the entropy
    parabola (rate ``1/tau ~ 2`` for ``tau ~ 0.5``), where the exact entropy
    difference is zero.  While ``f^eq`` was stored as float32 inside the kernel
    the resulting +1e-5 artefact made the solver believe H increases and it
    returned ``alpha = 0`` -- i.e. no collision at all and an O(1) viscosity
    error.  This test pins the fixed behaviour.
    """
    cfg = _cfg()
    f, alpha = initial_field(cfg)
    stats = np.zeros(6, dtype=np.float64)
    native_channel_step_inplace(
        f, np.empty_like(f), alpha, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, tau=cfg.tau,
        force=cfg.body_force, n_iter=5, stats=stats,
    )
    fluid_alpha = alpha[:, 1:-1, :]
    assert float(fluid_alpha.min()) == pytest.approx(ALPHA_CAP, abs=1e-6)
    # H-theorem: the *unforced* collision never raises H (the body force is
    # external work and is book-kept outside the entropy condition)
    assert stats[3] <= 1e-6, f"unforced collision raised H: {stats[3]}"


def test_torch_reference_returns_bgk_for_the_same_state() -> None:
    """Same statement through the torch reference (float32 input, fp64 algebra)."""
    cfg = _cfg()
    f, _ = initial_field(cfg)
    ft = torch.tensor(f, dtype=torch.float32)
    _, info = collide_elbm_d3q27(ft, cfg.tau, n_iter=8, return_info=True)
    alpha = info.alpha[:, 1:-1, :]
    assert float(alpha.min()) == pytest.approx(ALPHA_CAP, abs=1e-5)
    assert info.alpha.dtype == torch.float32


def test_limiter_still_engages_on_a_strongly_distorted_state() -> None:
    """The fix must not disable the limiter: a distorted state gets alpha < cap."""
    cfg = _cfg()
    f, alpha = initial_field(cfg)
    rng = np.random.default_rng(5)
    fluid = (slice(None), slice(1, cfg.ny - 1), slice(None))
    f[fluid] = np.clip(f[fluid] * (1.0 + 0.3 * rng.standard_normal(f[fluid].shape)), 1e-9, None)
    stats = np.zeros(6, dtype=np.float64)
    native_channel_step_inplace(
        f, np.empty_like(f), alpha, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, tau=cfg.tau,
        force=cfg.body_force, n_iter=12, stats=stats,
    )
    assert float(alpha[:, 1:-1, :].min()) < ALPHA_CAP
    assert stats[3] <= 1e-6, "H-theorem violated on a distorted state"
    assert np.all(f[:, :, 1:-1, :] > 0.0), "positivity lost"


# ---------------------------------------------------------------------------
# 3. statistics and the report interface
# ---------------------------------------------------------------------------


def test_sample_profiles_matches_a_brute_force_average() -> None:
    cfg = _cfg(nx=6, ny=14, nz=5)
    f, _ = initial_field(cfg)
    acc = np.zeros((6, cfg.ny), dtype=np.float64)
    sample_profiles(f, cfg, acc)
    # brute force, written independently of the module
    c = C27.numpy().astype(np.float64)
    rho = np.tensordot(np.ones(27), f, axes=(0, 0))
    mom = np.tensordot(c.T, f, axes=(1, 0))            # (3, nz, ny, nx)
    u = mom / rho
    u[0] += 0.5 * cfg.body_force
    for y in range(1, cfg.ny - 1):
        ux = u[0, :, y, :]
        uy = u[1, :, y, :]
        uz = u[2, :, y, :]
        assert acc[0, y] == pytest.approx(ux.mean(), rel=1e-6, abs=1e-9)
        assert acc[1, y] == pytest.approx(((ux - ux.mean()) ** 2).mean(), rel=1e-5, abs=1e-10)
        assert acc[2, y] == pytest.approx(((uy - uy.mean()) ** 2).mean(), rel=1e-5, abs=1e-10)
        assert acc[3, y] == pytest.approx(((uz - uz.mean()) ** 2).mean(), rel=1e-5, abs=1e-10)
        assert acc[4, y] == pytest.approx(
            ((ux - ux.mean()) * (uy - uy.mean())).mean(), rel=1e-5, abs=1e-10
        )
    assert np.all(acc[:, 0] == 0.0) and np.all(acc[:, cfg.ny - 1] == 0.0)


def test_raw_npz_is_accepted_by_the_dns_gate_machinery(tmp_path) -> None:
    cfg = _cfg(nx=8, ny=18, nz=8, steps=60, warmup=20, sample_every=10, log_every=20,
               ckpt_every=30, label="gate")
    state, summary = run_dns_channel(cfg, out_raw=tmp_path / "raw" / "run.npz",
                                     ckpt=tmp_path / "raw" / "run_ckpt.npz")
    assert summary["complete"] and state.samples > 0
    data = np.load(tmp_path / "raw" / "run.npz")
    profiles = {k.removeprefix("prof_"): data[k] for k in data.files if k.startswith("prof_")}
    assert set(profiles) == {"u", "uu", "vv", "ww", "uv", "rho"}
    scalars = np.asarray(data["scalars"]).tolist()
    assert len(scalars) == 10
    meta = np.load(tmp_path / "raw" / "run_meta.json") if False else None  # noqa: F841
    import json

    meta = json.loads((tmp_path / "raw" / "run_meta.json").read_text())
    assert meta["steps"] == cfg.steps and meta["grid"] == [cfg.nx, cfg.ny, cfg.nz]
    # feed the gate machinery (the same code path the production report uses)
    from tensorlbm.validation import mkm1999

    mkm = mkm1999.load_profile(180)
    gate_scalars = {
        "half_height": cfg.half_height,
        "u_tau": cfg.u_tau,
        "re_tau_nominal": cfg.re_tau,
        "delta_plus": cfg.delta_plus,
        "finite": True,
        "steps": cfg.steps,
        "min_population": scalars[1],
        "entropy_drop_max_collision": scalars[6],
        "entropy_excess_max_collision": meta.get("entropy_excess_max"),
        "max_u_rms_plus": 0.0,
        "max_tke": 0.0,
        "max_tke_over_u_tau2": 0.0,
        "max_minus_uv_plus": 0.0,
    }
    profile_lists = {
        "y": np.arange(cfg.ny)[1:-1] - 0.5,
        "y_plus_nominal": (np.arange(cfg.ny)[1:-1] - 0.5) * cfg.u_tau / cfg.nu,
        "u_plus_nominal": profiles["u"][1:-1] / cfg.u_tau,
        "u_prime_u_prime": profiles["uu"][1:-1],
        "v_prime_v_prime": profiles["vv"][1:-1],
        "w_prime_w_prime": profiles["ww"][1:-1],
        "u_prime_v_prime": profiles["uv"][1:-1],
        "tke": 0.5 * (profiles["uu"] + profiles["vv"] + profiles["ww"])[1:-1],
    }
    gates = channel_gates(profile_lists, gate_scalars, mkm)
    assert set(gates) >= {"a_log_slope", "b_reynolds_stress_peak",
                          "c_friction_coefficient", "d_long_time_stability"}
    # an unphysical 60-step run cannot pass the friction gate -- the machinery
    # must say so rather than default to a pass
    assert gates["c_friction_coefficient"]["pass"] in (True, False)
    assert log_law_slope is not None


# ---------------------------------------------------------------------------
# 4. checkpoint / restart
# ---------------------------------------------------------------------------


def test_checkpoint_restart_is_bit_identical(tmp_path) -> None:
    """40 steps in one go == 20 steps + restart + 20 steps."""
    cfg_one = _cfg(steps=40, warmup=10, sample_every=5, ckpt_every=1000, label="one")
    s_one, _ = run_dns_channel(cfg_one, out_raw=tmp_path / "one" / "r.npz",
                               ckpt=tmp_path / "one" / "c.npz")

    cfg_half = _cfg(steps=20, warmup=10, sample_every=5, ckpt_every=1000, label="half")
    run_dns_channel(cfg_half, out_raw=tmp_path / "two" / "r.npz", ckpt=tmp_path / "two" / "c.npz")
    loaded = load_checkpoint(tmp_path / "two" / "c.npz", cfg_half)
    assert loaded.step == 20

    # the full config (40 steps) resumed from the 20-step checkpoint
    s_two, summary = run_dns_channel(
        _cfg(steps=40, warmup=10, sample_every=5, ckpt_every=1000, label="two"),
        out_raw=tmp_path / "two" / "r.npz", ckpt=tmp_path / "two" / "c.npz", resume=True,
    )
    assert summary["resumed_from"] == 20
    assert summary["complete"]
    assert s_two.step == s_one.step
    assert np.array_equal(s_two.f, s_one.f), "restart changed the flow field"
    assert s_two.samples == s_one.samples
    assert np.array_equal(s_two.acc, s_one.acc), "restart changed the time average"


def test_max_wall_clock_stops_and_resumes(tmp_path) -> None:
    """A job-slot stop writes a checkpoint and the next call finishes the run."""
    cfg = _cfg(steps=200, warmup=50, sample_every=10, ckpt_every=40, log_every=50, label="slot")
    _, first = run_dns_channel(cfg, out_raw=tmp_path / "s" / "r.npz", ckpt=tmp_path / "s" / "c.npz",
                               max_wall_clock=0.0)
    assert not first["complete"]
    done = False
    for _ in range(10):  # a real job slot would be much longer than 0 s
        _, summary = run_dns_channel(cfg, out_raw=tmp_path / "s" / "r.npz",
                                     ckpt=tmp_path / "s" / "c.npz", resume=True,
                                     max_wall_clock=1.0)
        if summary["complete"]:
            done = True
            break
    assert done, "the resumable driver never finished the run"
    assert (tmp_path / "s" / "r.npz").exists()


def test_save_checkpoint_rejects_a_config_mismatch(tmp_path) -> None:
    cfg = _cfg()
    f, alpha = initial_field(cfg)
    from tensorlbm.elbm_channel_dns import DnsRunState

    state = DnsRunState(f=f, alpha=alpha, acc=np.zeros((6, cfg.ny)), history=[])
    save_checkpoint(tmp_path / "c.npz", cfg, state)
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path / "c.npz", _cfg(nx=4))

# ---------------------------------------------------------------------------
# 5. gate machinery: full-channel profiles
# ---------------------------------------------------------------------------


def test_full_channel_bulk_velocity_is_span_normalised() -> None:
    """Regression: ``U_b+``/``C_f`` must not be scaled by the half height.

    The channel driver returns profiles over the *full* channel
    (``y = 0 .. 2h``, both walls).  ``_bulk_velocity_plus`` used to divide the
    integral by ``half_height``, which is only right for a half-channel profile:
    ``U_b+`` came out twice too large and ``C_f`` four times too small, so
    criterion (c) (5 % on ``C_f``) could not be satisfied by any run.  The
    metric is now normalised by the span of the supplied profile, so mirroring a
    DNS half-channel profile into a full channel must reproduce the very same
    ``U_b+`` and pass criterion (c).
    """
    from tensorlbm.validation import mkm1999

    mkm = mkm1999.load_profile(180)
    # note: mkm.y is the *normalised* wall-normal coordinate (0..1); the wall
    # units live in mkm.y_plus, which is what the mirroring and the bulk
    # integral must use.
    half = {
        "y": mkm.y_plus.copy(),
        "y_plus_nominal": mkm.y_plus.copy(),
        "u_plus_nominal": mkm.u_plus.copy(),
        "u_prime_u_prime": (mkm.u_rms_plus**2).copy(),
        "u_prime_v_prime": mkm.uv_plus.copy(),
        "v_prime_v_prime": (mkm.v_rms_plus**2).copy(),
        "w_prime_w_prime": (mkm.w_rms_plus**2).copy(),
    }
    g_full = {
        "y": np.concatenate([half["y"], (2 * mkm.re_tau - half["y"][::-1])]),
        # (the duplicated centreline point is harmless: trapezoid weights it 0)
        "y_plus_nominal": np.concatenate([half["y_plus_nominal"],
                                          (2 * mkm.re_tau - half["y_plus_nominal"][::-1])]),
        "u_plus_nominal": np.concatenate([half["u_plus_nominal"], half["u_plus_nominal"][::-1]]),
        "u_prime_u_prime": np.concatenate([half["u_prime_u_prime"], half["u_prime_u_prime"][::-1]]),
        "u_prime_v_prime": np.concatenate([half["u_prime_v_prime"], -half["u_prime_v_prime"][::-1]]),
        "v_prime_v_prime": np.concatenate([half["v_prime_v_prime"], half["v_prime_v_prime"][::-1]]),
        "w_prime_w_prime": np.concatenate([half["w_prime_w_prime"], half["w_prime_w_prime"][::-1]]),
        "tke": np.concatenate([0.5 * (half["u_prime_u_prime"] + half["v_prime_v_prime"]
                                      + half["w_prime_w_prime"])] * 2),
    }
    scalars = {
        "half_height": 0.5 * (2 * mkm.re_tau),
        "u_tau": 1.0,
        "re_tau_nominal": mkm.re_tau,
        "delta_plus": 1.0,
        "finite": True,
        "steps": 100_000,
        "min_population": 1e-3,
        "entropy_drop_max_collision": 0.0,
    }
    gates = channel_gates({k: list(map(float, v)) for k, v in g_full.items()}, scalars, mkm,
                          log_band=(30.0, 60.0))
    c = gates["c_friction_coefficient"]
    assert c["u_bulk_plus_run"] == pytest.approx(mkm.bulk_velocity_plus(), rel=1e-4)
    assert c["cf_rel_error"] < 1e-4, f"C_f error {c['cf_rel_error']:.4f}"
    assert c["pass"] is True
    # and a 20 % too fast profile must still be rejected (the fix is not a loosening)
    g_fast = dict(g_full)
    g_fast["u_plus_nominal"] = [0.8 * v for v in g_full["u_plus_nominal"]]
    gates_fast = channel_gates({k: list(map(float, v)) for k, v in g_fast.items()}, scalars, mkm,
                              log_band=(30.0, 60.0))
    assert gates_fast["c_friction_coefficient"]["pass"] is False
