"""Fail-fast probe: D3Q19-only features requested on a D3Q27 ``GeneralSimEngine``.

``docs/AUDIT_RUNTEST_6_TENSORLBM.md`` closed the *scalar* input-contract holes
(D5/D6/D7/D9).  The same class of defect survives one level up: an
**unsupported configuration** — ``lattice=D3Q27`` combined with a feature that
only has a D3Q19 kernel — is accepted by ``GeneralSimConfig.validate()`` and by
``setup()``, and only explodes inside ``run()``, i.e. *after* the geometry has
been voxelised and the distribution function allocated.  On a production grid
that is minutes of work and gigabytes of allocation spent to learn that the
configuration could never have run.

This probe measures, for each unsupported ``(lattice, collision, wall)``
combination, **how much work is done before the rejection**:

* ``stage``    — ``config`` (validate), ``setup`` (voxelise + allocate) or ``run``
* ``cells``    — lattice cells allocated before the error
* ``field_bytes`` — bytes of ``f`` allocated before the error
* ``elapsed_s`` — wall time to the error

A good guard moves every row to ``stage=config/setup`` with ``field_bytes=0``.
The script is self-contained (no pytest) so the *same* command can be run against
the pre-fix and post-fix trees:

    python scripts/contract_guards/lattice_feature_probe.py --out artifacts/contract_guards/lattice_probe_before.json --label before
    python scripts/contract_guards/lattice_feature_probe.py --out artifacts/contract_guards/lattice_probe_after.json  --label after
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]


def _config(lattice: int, *, collision=None, wall=None, viscosity: float = 1.0e-6):
    from tensorlbm.general_sim import (
        CollisionModel,
        GeneralSimConfig,
        GeometryConfig,
        GeometrySource,
        LatticeModel,
        OutputConfig,
        OutputFormat,
        PhysicsConfig,
        SolverConfig,
        WallTreatment,
    )

    return GeneralSimConfig(
        name=f"probe_d3q{lattice}",
        geometry=GeometryConfig(
            source=GeometrySource.PARAMETRIC_SPHERE,
            sphere_radius=0.5,
            sphere_center=(0.0, 0.0, 0.0),
        ),
        physics=PhysicsConfig(
            density=1000.0,
            viscosity=viscosity,
            inlet_velocity=1.0e-4,
            reference_length=1.0,
        ),
        solver=SolverConfig(
            lattice=LatticeModel.D3Q27 if lattice == 27 else LatticeModel.D3Q19,
            collision=collision or CollisionModel.BGK,
            wall_treatment=wall or WallTreatment.AUTO,
            resolution=8,
            max_steps=4,
            snapshot_interval=10**9,
            force_sample_interval=2,
            device="cpu",
            mass_correction=False,
        ),
        output=OutputConfig(
            directory="/tmp/tensorlbm_lattice_probe",
            formats=[OutputFormat.NPY],
            save_macroscopic=False,
            save_forces=False,
        ),
    )


def _probe(
    name: str,
    lattice: int,
    *,
    collision=None,
    wall=None,
    viscosity: float = 1.0e-6,
    expect: str = "reject",
):
    """Run one configuration and record *where* it was rejected.

    ``expect='reject'`` marks a combination that has no kernel on that lattice
    (rejection is the correct outcome; the metric is how early it happens).
    ``expect='accept'`` is a positive control: rejection there would be a
    regression (blanket rejection).
    """
    from tensorlbm.general_sim import GeneralSimEngine

    row: dict = {"probe": name, "lattice": f"d3q{lattice}", "expect": expect}
    t0 = time.perf_counter()
    stage = "config"
    try:
        cfg = _config(lattice, collision=collision, wall=wall, viscosity=viscosity)
        eng = GeneralSimEngine(cfg)
        stage = "setup"
        info = eng.setup()
        row["cells"] = int(info["total_cells"])
        stage = "run"
        eng.run(steps=2)
    except Exception as exc:  # noqa: BLE001 - the probe is about *when* it fails
        row.update(
            {
                "stage": stage,
                "outcome": "raised",
                "exc_type": type(exc).__name__,
                "exc_msg": str(exc)[:160],
                "elapsed_s": round(time.perf_counter() - t0, 3),
            }
        )
        f = getattr(locals().get("eng", None), "f", None)
        if isinstance(f, torch.Tensor):
            row["field_bytes"] = int(f.numel() * f.element_size())
            row["cells"] = int(f.shape[1] * f.shape[2] * f.shape[3])
        else:
            row["field_bytes"] = 0
        return row
    row.update(
        {
            "stage": "ran",
            "outcome": "accepted",
            "elapsed_s": round(time.perf_counter() - t0, 3),
            "field_bytes": None,
        }
    )
    return row


def main() -> int:
    from tensorlbm.general_sim import CollisionModel, WallTreatment

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    rows = [
        # D3Q27 x D3Q19-only features (the defect under test)
        _probe("d3q27:wall_function", 27, wall=WallTreatment.WALL_FUNCTION),
        _probe("d3q27:smagorinsky_mrt", 27, collision=CollisionModel.SMAGORINSKY_MRT),
        _probe("d3q27:smagorinsky_bgk", 27, collision=CollisionModel.SMAGORINSKY_BGK),
        # AUTO wall treatment resolves to wall_function at Re >= 1e4 -> same hole
        _probe("d3q27:auto_wall_high_re", 27, viscosity=1.0e-8),
        _probe("d3q27:trt", 27, collision=CollisionModel.TRT),
        # Positive controls: must keep working (no blanket rejection)
        _probe("d3q27:bgk_bounce_back", 27, expect="accept"),
        _probe("d3q27:mrt_bounce_back", 27, collision=CollisionModel.MRT, expect="accept"),
        _probe("d3q19:wall_function", 19, wall=WallTreatment.WALL_FUNCTION, expect="accept"),
        _probe(
            "d3q19:smagorinsky_mrt",
            19,
            collision=CollisionModel.SMAGORINSKY_MRT,
            expect="accept",
        ),
    ]

    import tensorlbm

    payload = {
        "label": args.label,
        "package_path": str(Path(tensorlbm.__file__).resolve().parents[1]),
        "torch": torch.__version__,
        "rows": rows,
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))

    should_reject = [r for r in rows if r["expect"] == "reject"]
    should_accept = [r for r in rows if r["expect"] == "accept"]
    early = [r for r in should_reject if r.get("stage") in {"config", "setup"}]
    late = [r for r in should_reject if r.get("stage") == "run"]
    leaked = [r for r in should_reject if r["outcome"] == "accepted"]
    over_rejected = [r for r in should_accept if r["outcome"] != "accepted"]
    summary = {
        "label": args.label,
        "unsupported_combos": len(should_reject),
        "rejected_before_allocation": len(early),
        "rejected_during_run": len(late),
        "silently_accepted": len(leaked),
        "bytes_allocated_before_rejection": sum(r.get("field_bytes") or 0 for r in should_reject),
        "positive_controls_ok": f"{len(should_accept) - len(over_rejected)}/{len(should_accept)}",
    }
    payload["summary"] = summary
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print("\n" + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
