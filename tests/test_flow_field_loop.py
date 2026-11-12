"""Tests for the T4 physics-AI loop (:mod:`tensorlbm.ai.flow_field_loop`).

The invariants that matter for the T4 gap:

* the field producer is a real solver with the advertised viscosity
  (single-mode Taylor-Green decay vs the exact Navier-Stokes solution);
* the produced corpus carries sha256 provenance and a train/holdout split and
  is registered in the TD-side catalog;
* the dataset is *not* an analytic field in disguise (multi-mode runs differ
  from the closed-form single-mode solution);
* the FNO flow map trains, beats nothing by accident, and its held-out
  evaluation only ever touches runs whose ``Re`` is excluded from training;
* the surrogate can be handed back to the real solver, and the solver restart
  from an exact state is deterministic (the control that makes the handback
  error meaningful).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from tensorlbm.ai.flow_field_loop import (
    DATASET_SCHEMA,
    FieldRunConfig,
    FlowMapConfig,
    _rel_l2,
    analytic_single_mode,
    evaluate_holdout,
    feedback_acceleration,
    initial_velocity,
    load_dataset,
    produce_dataset,
    run_case,
    train_flow_map,
    verify_producer,
    viscous_prior_step,
)

torch.set_num_threads(2)


# ---------------------------------------------------------------------------
# Producer
# ---------------------------------------------------------------------------


def _spectral_divergence(ux: torch.Tensor, uy: torch.Tensor) -> float:
    n = ux.shape[-1]
    kx = torch.fft.fftfreq(n) * 2.0 * math.pi
    ky = torch.fft.fftfreq(n) * 2.0 * math.pi
    div = torch.fft.ifft2(
        torch.fft.fft2(ux) * (1j * kx).view(1, -1) + torch.fft.fft2(uy) * (1j * ky).view(-1, 1)
    )
    return float(div.abs().max())


def test_initial_field_is_divergence_free_with_requested_scale():
    cfg = FieldRunConfig(n=32, re=100.0, seed=3)
    ux, uy = initial_velocity(cfg)
    assert float(torch.sqrt((ux**2 + uy**2).mean())) == pytest.approx(cfg.u0, rel=1e-4)
    assert _spectral_divergence(ux, uy) < 1e-5


def test_producer_reproduces_exact_taylor_green_decay():
    """The producer must be a real solver: measured decay = 2 nu k^2."""
    out = verify_producer(n=64, re_values=(50.0, 200.0), n_steps=1500, sample_every=150)
    assert out["verified"] is True, json.dumps(out, indent=1)
    for row in out["rows"]:
        assert row["max_abs_rel_err"] < 0.01
        # LBM recovers the advertised viscosity to better than 2 %
        assert abs(row["decay_rate_rel_err"]) < 0.02
        assert row["tau"] == pytest.approx(3.0 * row["nu"] + 0.5)


def test_multi_mode_dataset_is_not_an_analytic_field():
    """Multi-mode runs must differ from the closed-form single-mode solution."""
    cfg = FieldRunConfig(n=32, re=100.0, seed=1, n_steps=0, sample_every=10)
    ux, uy = initial_velocity(cfg)
    k = 2.0 * math.pi / cfg.n
    x = torch.arange(cfg.n).float().view(1, -1).expand(cfg.n, cfg.n)
    y = torch.arange(cfg.n).float().view(-1, 1).expand(cfg.n, cfg.n)
    ana = torch.stack(
        [
            cfg.u0 * torch.sin(k * x) * torch.cos(k * y),
            -cfg.u0 * torch.cos(k * x) * torch.sin(k * y),
        ]
    )
    assert _rel_l2(torch.stack([ux, uy]), ana) > 0.1


def test_solver_restart_from_exact_state_is_deterministic():
    """Control for the feedback test: handing back the *true* state is exact.

    Restart goes through ``equilibrium(rho, u)``, so the non-equilibrium part
    of the populations is dropped: the residual restart error is ~4e-6
    relative (measured), four orders of magnitude below the surrogate handback
    errors reported by :func:`feedback_acceleration`.
    """
    cfg = FieldRunConfig(n=32, re=100.0, seed=0, n_steps=40, sample_every=20)
    full = run_case(cfg)
    restart = run_case(cfg, initial=(full.ux[0], full.uy[0]), n_steps=20)
    assert _rel_l2(restart.ux[-1], full.ux[1]) < 1e-4
    assert _rel_l2(restart.uy[-1], full.uy[1]) < 1e-4


def test_viscous_prior_reproduces_exponential_mode_decay():
    cfg = FieldRunConfig(n=32, re=100.0, seed=0, n_steps=0, sample_every=10, single_mode=True)
    ux, uy = initial_velocity(cfg)
    dt = 100
    pux, puy = viscous_prior_step(ux, uy, cfg.nu, dt)
    factor = math.exp(-2.0 * cfg.nu * cfg.k1**2 * dt)
    assert _rel_l2(pux, ux * factor) < 5e-3


# ---------------------------------------------------------------------------
# Dataset asset
# ---------------------------------------------------------------------------


def _tiny_configs():
    return [
        FieldRunConfig(n=32, re=50.0, seed=0, n_steps=60, sample_every=20, tag="t"),
        FieldRunConfig(n=32, re=100.0, seed=0, n_steps=60, sample_every=20, tag="t"),
        FieldRunConfig(n=32, re=75.0, seed=0, n_steps=60, sample_every=20, tag="t"),
    ]


def test_dataset_manifest_hashes_catalog_and_split(tmp_path: Path):
    ds = tmp_path / "ds"
    manifest = produce_dataset(
        _tiny_configs(),
        ds,
        dataset_name="t4-test",
        split={"train_re": [50.0, 100.0], "holdout_re": [75.0]},
    )
    assert manifest["schema"] == DATASET_SCHEMA
    assert manifest["n_files"] == 3
    assert manifest["generator"]["real_lbm"] is True
    assert manifest["split"]["holdout_re"] == [75.0]

    on_disk = json.loads((ds / "manifest.json").read_text())
    for entry in on_disk["files"]:
        payload = (ds / entry["file"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
        assert entry["provenance"]["nu"] > 0.0
        assert entry["provenance"]["tau"] > 0.5

    from tensorlbm.data.catalog import FieldDataCatalog

    cat = FieldDataCatalog.open(ds / "catalog.sqlite")
    try:
        asset = cat.get_asset("t4-test:dataset")
        assert asset is not None and asset.kind == "dataset"
        keys = {m.key: m.value for m in cat.get_metadata("t4-test:dataset")}
        assert keys["manifest_sha256"] == manifest["manifest_sha256"]
        field_ids = [a.asset_id for a in cat.list_assets(kind="field_product")]
        assert len(field_ids) == 3
        assert all(cat.upstream("t4-test:dataset") for _ in [0])
    finally:
        cat.close()


# ---------------------------------------------------------------------------
# AI step + feedback
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory) -> Path:
    ds = tmp_path_factory.mktemp("t4ds") / "ds"
    produce_dataset(
        _tiny_configs(),
        ds,
        dataset_name="t4-test",
        split={"train_re": [50.0, 100.0], "holdout_re": [75.0]},
    )
    return ds


def test_train_and_holdout_loop_end_to_end(tiny_dataset: Path):
    cfg = FlowMapConfig(width=8, n_layers=2, modes=4, mlp_hidden=32, epochs=3, batch_size=4,
                        patience=10)
    trained = train_flow_map(
        tiny_dataset, tiny_dataset / "m.pt", config=cfg,
        train_res=[50.0, 100.0], holdout_res=[75.0], grid_n=32, log=False,
    )
    assert trained["n_train_pairs"] == 4  # 2 train runs x 2 pairs
    assert math.isfinite(trained["best_val_mse"])

    ev = evaluate_holdout(
        trained["model"], tiny_dataset, train_res=[50.0, 100.0], holdout_res=[75.0],
        n_rollout=2, grid_n=32,
    )
    assert ev["n_holdout_runs"] == 1
    row = ev["per_run"][0]
    assert row["re"] == 75.0 and row["in_train"] is False
    for key in ("fno", "viscous_prior", "persistence"):
        assert math.isfinite(row["one_step_rel_l2"][key])
    for key in ("fno", "viscous", "persistence"):
        assert len(row["rollout"][key]["rel_l2_series"]) == 2


def test_holdout_baselines_are_scored_in_the_target_units(tiny_dataset: Path):
    """Regression for the m15932 board audit (baseline unit mismatch).

    ``evaluate_holdout`` scores the FNO against ``y = u/u0``.  The persistence and
    viscous-prior baselines used to be handed the *raw lattice-unit* fields, so
    ``||u(t) - u(t+dt)/u0|| / ||u(t+dt)/u0||`` pinned both baselines at ~1/u0
    (u0 = 0.05 -> ~1.0) no matter how good or bad they were, and the reported
    "FNO beats persistence / viscous prior" ordering was an artefact.  This test
    recomputes both baselines independently in u0 units and requires the
    reported numbers to match.
    """
    cfg = FlowMapConfig(width=8, n_layers=2, modes=4, mlp_hidden=32, epochs=2,
                        batch_size=4, patience=10)
    trained = train_flow_map(
        tiny_dataset, tiny_dataset / "m_units.pt", config=cfg,
        train_res=[50.0, 100.0], holdout_res=[75.0], grid_n=32, log=False,
    )
    ev = evaluate_holdout(
        trained["model"], tiny_dataset, train_res=[50.0, 100.0], holdout_res=[75.0],
        grid_n=None,
    )
    row = ev["per_run"][0]
    run = next(r for r in load_dataset(tiny_dataset)["runs"].values()
               if round(float(r["re"]), 6) == 75.0)
    u0 = run["u0"]
    dt = run["steps"][1] - run["steps"][0]
    n_pairs = run["ux"].shape[0] - 1

    def _manual(pred_fn):
        errs = []
        for i in range(n_pairs):
            ref = torch.stack([run["ux"][i + 1], run["uy"][i + 1]]) / u0
            errs.append(_rel_l2(pred_fn(i) / u0, ref))
        return sum(errs) / len(errs)

    pers_ref = _manual(lambda i: torch.stack([run["ux"][i], run["uy"][i]]))
    vis_ref = _manual(
        lambda i: torch.stack(viscous_prior_step(run["ux"][i], run["uy"][i], run["nu"], dt))
    )
    assert row["one_step_rel_l2"]["persistence"] == pytest.approx(pers_ref, rel=1e-5)
    assert row["one_step_rel_l2"]["viscous_prior"] == pytest.approx(vis_ref, rel=1e-5)
    # The pre-fix signature was a baseline pinned at ~1/u0 (20x for u0=0.05);
    # the u0-consistent number is a genuine field-change ratio.
    assert all(err["one_step_rel_l2"]["persistence"] < 0.9 / u0
               for err in ev["per_run"])


def test_feedback_hands_state_back_to_the_solver(tiny_dataset: Path):
    cfg = FlowMapConfig(width=8, n_layers=2, modes=4, mlp_hidden=32, epochs=2, batch_size=4,
                        patience=10)
    trained = train_flow_map(
        tiny_dataset, tiny_dataset / "m2.pt", config=cfg,
        train_res=[50.0, 100.0], holdout_res=[75.0], grid_n=32, log=False,
    )
    fb = feedback_acceleration(
        trained["model"], tiny_dataset, holdout_res=[75.0], k_values=(1,), heal_steps=20,
    )
    assert len(fb["rows"]) == 1
    row = fb["rows"][0]
    assert row["lbm_steps_replaced"] == 20
    assert math.isfinite(row["handback_rel_l2"])
    assert row["handback_rel_l2"] < 1.5  # a finite, sane error — not a NaN pass-through
    assert row["healed_rel_l2"] is None or math.isfinite(row["healed_rel_l2"])


def test_load_dataset_roundtrips_fields(tiny_dataset: Path):
    data = load_dataset(tiny_dataset)
    assert len(data["runs"]) == 3
    run = next(iter(data["runs"].values()))
    assert run["ux"].shape[1:] == (32, 32)
    assert run["steps"][0] == 0
    # sampled fields are the solver output: non-zero and finite
    assert float(run["ux"].abs().max()) > 0.0
    assert bool(torch.isfinite(run["ux"]).all())


def test_analytic_reference_is_the_closed_form():
    vals = analytic_single_mode(64, 0.05, 0.01, [0, 100])
    assert vals[0] == pytest.approx(0.05 / math.sqrt(2.0))
    assert vals[1] / vals[0] == pytest.approx(
        math.exp(-2 * 0.01 * (2 * math.pi / 64) ** 2 * 100)
    )