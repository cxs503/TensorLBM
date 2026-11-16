"""Mutation / detection-power probe for the model matrix gate.

Why this module exists
---------------------
A green gate is only evidence if it can go red.  The matrix currently reports
"47/47 cells PASS"; that statement is worthless until the smallest defect each
case can detect is known.  This module injects *controlled* defects into a
registered model and re-runs the gate, so every PASS can be paired with the
answer to "how wrong would this operator have to be before we noticed?".

Mutation classes (each emulates a real defect mode, not a synthetic typo)
-------------------------------------------------------------------------
``nu_scale``
    The shear channel relaxes with a wrong viscosity: ``nu' = (1+d) nu``.
    Emulates the classic ``nu=(tau-1/2)/3`` vs ``nu=(tau-1/2)/2`` /
    ``omega``-vs-``tau`` family of bugs.  Swept in ``d`` to find the
    detection threshold.
``bulk_only``
    Only the trace/energy eigenvalue is corrupted (``s_b' = k s_b``), which is
    what a mis-wired bulk parameter (``s_e``/``s_bulk``/``omega_b``) or a
    wrong bulk-viscosity formula looks like in production.  Invisible to a
    shear-only matrix by construction -- this is the blind spot the
    ``acoustic_decay`` case exists to close.
``mass_leak``
    Every collision multiplies the distribution by ``1+eps``, so mass grows
    without any hydrodynamics changing.  Emulates an accumulator/reset bug.
``momentum_leak``
    A spurious momentum ``eps * c_x`` is added after collision: the
    equilibrium is no longer a fixed point.

Every mutation is a *wrapper* around the real production operator (no numerics
are reimplemented here), and the metrics are compared against the *unmutated*
analytic reference, so a mutation that the gate misses is reported as missed.

``python -m tensorlbm.model_matrix mutate`` writes the JSON + Markdown
evidence used by ``docs/MODEL_MATRIX_CI_PLUS_COVERAGE_R2.md``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch

from .cases import Grid, run_case
from .registry import ModelSpec, build_collider, get_model, registry

CASES_UNDER_TEST = ("feq_fixed_point", "shear_decay", "acoustic_decay")
"""The gate cases whose detection power is measured (tg_decay is advisory)."""


def _register(spec: ModelSpec) -> ModelSpec:
    reg = registry()
    if spec.name in reg:
        raise RuntimeError(f"mutation model name collides with a registry key: {spec.name}")
    reg[spec.name] = spec
    return spec


def make_mutant(
    base_name: str,
    kind: str,
    magnitude: float,
    *,
    tau: float,
) -> ModelSpec:
    """Return a registered :class:`ModelSpec` that misbehaves in a known way."""
    base = get_model(base_name)
    inner = build_collider(base_name, tau)
    ops_shape_ok = True

    if kind == "nu_scale":
        tau_eff = 0.5 + (tau - 0.5) * (1.0 + magnitude)
        shifted = build_collider(base_name, tau_eff)

        def collide(f: torch.Tensor, **_kw: Any) -> torch.Tensor:
            return shifted(f)

    elif kind == "bulk_only":
        if base.bulk_param is None:
            raise ValueError(
                f"{base_name} is single-rate (bulk_param=None): its trace and shear "
                "channels share one eigenvalue, so a bulk-only defect does not exist"
            )
        s_b = float(base.params[base.bulk_param])
        target_s = s_b * (1.0 + magnitude)

        def collide(f: torch.Tensor, **_kw: Any) -> torch.Tensor:
            return base.collide(f, **{base.tau_arg: tau}, **{**base.params, base.bulk_param: target_s})

    elif kind == "mass_leak":
        def collide(f: torch.Tensor, **_kw: Any) -> torch.Tensor:
            return inner(f) * (1.0 + magnitude)

    elif kind == "momentum_leak":
        def collide(f: torch.Tensor, **_kw: Any) -> torch.Tensor:
            post = inner(f)
            rho, *_ = _macro(base, post)
            bump = torch.zeros_like(post)
            bump[1] = magnitude * rho  # d3q19/d3q27 index 1 is the +x direction
            return post + bump

    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"unknown mutation kind {kind!r}")

    name = f"{base_name}::{kind}::{magnitude:g}"
    return _register(dataclasses.replace(base, name=name, collide=collide))


def _macro(spec: ModelSpec, f: torch.Tensor):
    from .registry import lattice_ops  # noqa: PLC0415

    return lattice_ops(spec.lattice)["macroscopic"](f)


DEFAULT_SWEEP: dict[str, Sequence[float]] = {
    "nu_scale": (0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10),
    "bulk_only": (0.02, 0.05, 0.10, 0.25, 0.50),
    "mass_leak": (1e-9, 1e-8, 1e-7, 1e-6),
    "momentum_leak": (1e-10, 1e-9, 1e-8, 1e-7),
}


def probe(
    model_names: Iterable[str],
    *,
    kinds: Sequence[str] | None = None,
    sweep: dict[str, Sequence[float]] | None = None,
    grid: Grid | None = None,
    tau: float = 0.6,
) -> dict[str, Any]:
    """Inject each mutation and record which cases go red."""
    grid = grid or Grid(16, 16, 1)
    sweep = dict(sweep or DEFAULT_SWEEP)
    kinds = list(kinds or sweep)
    records: list[dict[str, Any]] = []

    for name in model_names:
        base = get_model(name)
        # control: an unmutated run must be green, otherwise the mutation
        # study would be measuring noise.
        ctrl = {
            case: run_case(case, base, grid, tau)
            for case in CASES_UNDER_TEST
            if case in base.cases
        }
        records.append(
            {
                "model": name,
                "kind": "control",
                "magnitude": 0.0,
                "statuses": {c: r.status for c, r in ctrl.items()},
                "detected": False,
                "detected_by": [],
            }
        )
        for kind in kinds:
            for mag in sweep[kind]:
                try:
                    mutant = make_mutant(name, kind, mag, tau=tau)
                except ValueError as exc:
                    records.append(
                        {
                            "model": name,
                            "kind": kind,
                            "magnitude": mag,
                            "statuses": {},
                            "detected": None,
                            "detected_by": [],
                            "note": str(exc),
                        }
                    )
                    continue
                statuses: dict[str, str] = {}
                for case in CASES_UNDER_TEST:
                    if case not in mutant.cases and case not in mutant.inapplicable_reason:
                        continue
                    statuses[case] = run_case(case, mutant, grid, tau).status
                fired = [c for c, st in statuses.items() if st in ("FAIL", "ERROR")]
                records.append(
                    {
                        "model": name,
                        "kind": kind,
                        "magnitude": mag,
                        "statuses": statuses,
                        "detected": bool(fired),
                        "detected_by": fired,
                    }
                )
                registry().pop(mutant.name, None)

    thresholds = _thresholds(records)
    return {
        "tau": tau,
        "grid": {"nx": grid.nx, "ny": grid.ny, "nz": grid.nz},
        "cases": list(CASES_UNDER_TEST),
        "records": records,
        "thresholds": thresholds,
    }


def _thresholds(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Smallest detected magnitude per (model, kind), plus the cases that fired."""
    out: list[dict[str, Any]] = []
    keys = {(r["model"], r["kind"]) for r in records if r["kind"] != "control"}
    for model, kind in sorted(keys):
        rows = [
            r
            for r in records
            if r["model"] == model and r["kind"] == kind and r["detected"] is not None
        ]
        hit = [r for r in rows if r["detected"]]
        if not rows:
            out.append({"model": model, "kind": kind, "threshold": None, "note": "not applicable"})
            continue
        if not hit:
            out.append(
                {
                    "model": model,
                    "kind": kind,
                    "threshold": None,
                    "max_tested": max(r["magnitude"] for r in rows),
                    "note": "UNDETECTED at every tested magnitude",
                }
            )
            continue
        first = min(hit, key=lambda r: r["magnitude"])
        out.append(
            {
                "model": model,
                "kind": kind,
                "threshold": first["magnitude"],
                "detected_by": first["detected_by"],
            }
        )
    return out


def write_evidence(study: dict[str, Any], outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    jp = outdir / "mutation_detection.json"
    jp.write_text(json.dumps(study, indent=2) + "\n")

    lines = [
        "# Gate detection power (mutation study, machine-generated)",
        "",
        "Each row: the smallest injected defect magnitude the gate turned red on,",
        "and which case did the detecting. `UNDETECTED` means every tested",
        "magnitude stayed green -- a blind spot, not a pass.",
        "",
        f"grid {study['grid']['nx']}x{study['grid']['ny']}x{study['grid']['nz']}, "
        f"tau={study['tau']}, cases={', '.join(study['cases'])}",
        "",
        "| model | mutation | detection threshold | detected by | note |",
        "|---|---|---|---|---|",
    ]
    for t in study["thresholds"]:
        thr = "UNDETECTED" if t.get("threshold") is None and "max_tested" in t else (
            "n/a" if t.get("threshold") is None else f"{t['threshold']:g}"
        )
        by = ", ".join(t.get("detected_by", [])) or "-"
        note = t.get("note", "")
        lines.append(f"| `{t['model']}` | {t['kind']} | {thr} | {by} | {note} |")
    mp = outdir / "MUTATION_DETECTION.md"
    mp.write_text("\n".join(lines) + "\n")
    return [jp, mp]


def main(argv: Sequence[str] | None = None) -> int:
    import argparse  # noqa: PLC0415
    import sys  # noqa: PLC0415

    from .__main__ import _repo_root  # noqa: PLC0415

    ap = argparse.ArgumentParser(prog="model_matrix mutate")
    ap.add_argument(
        "--models",
        default="d3q19/bgk,d3q19/mrt,d3q19/cascaded,d3q27/kbc",
        help="comma separated registry keys to mutate",
    )
    ap.add_argument("--out", default="artifacts/model_matrix_plus/mutation")
    ap.add_argument("--tau", type=float, default=0.6)
    args = ap.parse_args(argv)

    names = [s.strip() for s in args.models.split(",") if s.strip()]
    study = probe(names, tau=args.tau)
    paths = write_evidence(study, Path(args.out))
    for p in paths:
        print(f"[mutate] wrote {p}", file=sys.stderr)
    for t in study["thresholds"]:
        thr = "UNDETECTED" if t.get("threshold") is None and "max_tested" in t else t.get("threshold")
        print(f"[mutate] {t['model']:22s} {t['kind']:14s} {thr}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())