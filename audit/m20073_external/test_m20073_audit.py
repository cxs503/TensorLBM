"""Audit invariants for m20073 GAP-DIFFLBM-INVERSE-SHAPE (m20105 board audit).

These tests assert the *invariants* the verdict rests on, not merely that code runs:

I1  the delivered observable pair is blind/insensitive to cy (forward model, no optimiser):
    |drag(cy=9.0, r=r_true) - drag(cy_true, r_true)| / |drag_true| < 1e-6
I2  the loss is non-identifying along cy at r = truth: loss(cy=7.0) and loss(cy=9.0)
    are both below 1e-4 (i.e. a +/-12.5% error in cy looks as good as the truth)
I3  the delivered optimisation does NOT converge to the truth: the recorded 20-iteration
    re-run has cy error strictly growing (12.50% -> 16.41%) and the script's own
    `success` flag is False.
I4  therefore the delivered <=3% acceptance line is not met by either the 5- or the
    20-iteration run (both parameter errors above 3%).

Cheap (reads the audit's own JSON artifacts); I1 re-runs two forward evaluations.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent

run5 = json.loads((HERE / "run5/results.json").read_text())
run20 = json.loads((HERE / "run20/results.json").read_text())
probe = json.loads((HERE / "probe/identifiability.json").read_text())


def test_i1_drag_is_blind_to_lateral_centre():
    grid = {(round(g["cy"], 6), round(g["r"], 6)): g for g in probe["grid"]}
    ref = probe["reference"]
    row = grid[(9.0, round(ref["r"], 6))]
    assert abs(row["rel_drag"]) < 1e-6, f"drag not blind to cy: {row['rel_drag']:.3e}"
    # ... while the same shift changes the wake centre by a measurable amount
    assert 1e-3 < abs(row["rel_cy_wake"]) < 1e-2


def test_i2_loss_is_flat_in_cy_at_true_radius():
    ref = probe["reference"]
    rows = {round(g["cy"], 6): g for g in probe["grid"] if abs(g["r"] - ref["r"]) < 1e-12}
    assert rows[7.0]["loss"] < 1e-4, rows[7.0]["loss"]
    assert rows[9.0]["loss"] < 1e-4, rows[9.0]["loss"]
    # the 3%-wide window around the truth is NOT separable from a +/-12.5% error
    assert abs(rows[7.0]["loss"] - rows[8.0]["loss"]) < 1e-4


def test_i3_delivered_run_diverges_away_from_truth():
    cy_true = run20["true_params"]["cy"]
    errs = [abs(h["cy"] - cy_true) / cy_true for h in run20["loss_history"]]
    losses = [h["loss"] for h in run20["loss_history"]]
    assert losses[-1] < 0.05 * losses[0], "loss did not collapse as claimed"
    assert errs[-1] > errs[0], "cy error shrank; the audit's falsification does not hold"
    assert min(errs) >= errs[0] - 1e-12, "cy error improved somewhere; re-examine the verdict"
    # delivered script's own success flag
    assert run20["success"] is False and run5["success"] is False


def test_i4_acceptance_line_not_met():
    for run in (run5, run20):
        for name, err in run["relative_errors"].items():
            assert err > 0.03, f"{name} error {err:.4f} would meet the declared 3% line"


@pytest.mark.parametrize("cy_probe", [9.0])
def test_i5_forward_drag_invariance_recomputed(cy_probe):
    """Independent recomputation of I1 straight from the delivered forward model."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "inverse_shape_multiobs", HERE / "inverse_shape_multiobs.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    dtype, device = torch.float64, torch.device("cpu")

    def drag(cy: float) -> float:
        with torch.no_grad():
            d, _, _ = mod.run_forward(cy, mod.R_TRUE, mod.U_IN, mod.STEPS,
                                      mod.PROBE_START, mod.EPS_SOFT, dtype, device)
        return float(d)

    d_true, d_probe = drag(mod.CY_TRUE), drag(cy_probe)
    rel = abs(d_probe - d_true) / abs(d_true)
    assert rel < 1e-6, f"recomputed drag sensitivity to cy = {rel:.3e} (expected < 1e-6)"