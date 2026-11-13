#!/usr/bin/env python3
"""Root-cause probe for the float32-weight leak in ``collide_bgk_fused``.

Measured fact (pre-fix): one float64 fused BGK step moved the total mass by
+9.313e-09 relative, while every other collision operator stayed at ~1e-14.

Hypothesis: the fused path built its weights from ``d2q9.W`` (a **float32**
table) and cast it to float64, so ``sum_q w_q = 1 + 7.4506e-9`` instead of 1.
A fused equilibrium then injects ``bias/tau`` of spurious density per step.

This probe tests the hypothesis quantitatively, without reverting the fix:
it (a) measures the weight-table bias, (b) predicts the per-step drift,
(c) reproduces the buggy operator locally from the float32 table and checks the
prediction against it, and (d) shows the shipped operator is now clean.

Usage: PYTHONPATH=src python3 benchmarks/xval_bitwise_refdata/probe_weight_bias.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent.parent))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import harness  # noqa: E402
import torch  # noqa: E402

from tensorlbm import d2q9, solver  # noqa: E402

TAU = 0.8
STEPS = 200


def legacy_fused(f: torch.Tensor, tau: float) -> torch.Tensor:
    """The pre-fix ``collide_bgk_fused``: float32 weight table cast to f.dtype."""
    w = d2q9.W.to(f.dtype)  # <-- the defect: float32 rounding kept in float64
    rho = f.sum(dim=0)
    c = d2q9.C
    ux = (f * c[:, 0].view(9, 1, 1).to(f.dtype)).sum(dim=0) / rho
    uy = (f * c[:, 1].view(9, 1, 1).to(f.dtype)).sum(dim=0) / rho
    u_sq = ux * ux + uy * uy
    cu = c[:, 0].view(9, 1, 1).to(f.dtype) * ux + c[:, 1].view(9, 1, 1).to(f.dtype) * uy
    feq = (
        w.view(9, 1, 1)
        * rho.unsqueeze(0)
        * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq.unsqueeze(0))
    )
    return f * (1.0 - 1.0 / tau) + feq * (1.0 / tau)


def chain(f0: torch.Tensor, collide, steps: int) -> float:
    m0 = float(f0.sum())
    f = f0
    for _ in range(steps):
        f = solver.stream(collide(f))
    return (float(f.sum()) - m0) / m0


def main() -> int:
    f0 = harness.feq_field(16, 16)
    bias = float(d2q9.W.double().sum()) - 1.0
    predicted_step = bias / TAU
    predicted_200 = STEPS * predicted_step

    legacy_step = (float(legacy_fused(f0, TAU).sum()) - float(f0.sum())) / float(f0.sum())
    shipped_step = (float(solver.collide_bgk_fused(f0, TAU).sum()) - float(f0.sum())) / float(
        f0.sum()
    )
    legacy_200 = chain(f0, lambda x: legacy_fused(x, TAU), STEPS)
    shipped_200 = chain(f0, lambda x: solver.collide_bgk_fused(x, TAU), STEPS)
    finite = chain(f0, lambda x: solver.collide_bgk(x, TAU), STEPS)

    out = {
        "tau": TAU,
        "steps": STEPS,
        "weight_table_bias_f32_in_f64": bias,
        "predicted_step_drift": predicted_step,
        "predicted_steps_drift": predicted_200,
        "measured_legacy_step_drift": legacy_step,
        "measured_legacy_steps_drift": legacy_200,
        "measured_shipped_step_drift": shipped_step,
        "measured_shipped_steps_drift": shipped_200,
        "measured_bgk_steps_drift": finite,
        "step_prediction_rel_err": abs(legacy_step - predicted_step) / abs(predicted_step),
        "steps_prediction_rel_err": abs(legacy_200 - predicted_200) / abs(predicted_200),
        "hypothesis_confirmed": bool(
            abs(legacy_step - predicted_step) / abs(predicted_step) < 0.02
            and abs(legacy_200 - predicted_200) / abs(predicted_200) < 0.02
            and abs(shipped_200) < 1e-12
        ),
    }
    out_path = HERE / "probe_weight_bias.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"[probe] wrote {out_path}")
    return 0 if out["hypothesis_confirmed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
