"""Falsifiability audit of the fail-closed wall/refinement combination gate.

Anchor (m16911): ``test_unavailable_collision_contract_cannot_be_promoted_by_baseline_shape``
sat in ``tests/ci_quarantine.txt`` (rewritten to a non-strict xfail by ``tests/conftest.py``),
so the collision axis of the gate was never exercised by CI.  Its expectation had also
drifted: ``collision_capability_matrix()`` now registers all 7 families as AVAILABLE for
both D3Q19 and D3Q27, so *no* family expressible by the gate's ``CollisionFamily`` enum can
reach ``WITHHELD_UNSUPPORTED_COLLISION``.  A guard no input can trigger is not evidence.

This tool measures the gate rather than asserting that it "works":

1. exhaustively sweeps the expressible combination space and records the decision and the
   reason codes for every point;
2. checks reason-code reachability (a ``WITHHELD_*`` constant that never fires is dead code);
3. runs a *differential* arm: withdraw exactly one (lattice, family) cell from the capability
   matrix and confirm the gate flips exactly the decisions that use that cell -- i.e. the
   collision axis is causally wired, not decorative;
4. renders ``figures/wall_refinement_gate_admission_matrix.png`` and writes
   ``artifacts/wall_refinement_gate/falsifiability_audit.json`` (machine-rendered numbers;
   nothing hand-copied).

Usage::

    PYTHONPATH=src python3 tools/wall_refinement_gate_falsifiability.py
"""

from __future__ import annotations

import itertools
import json
import pathlib
from dataclasses import replace

import tensorlbm.boundary.wall_refinement_combination_gate as gate

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACT = REPO_ROOT / "artifacts" / "wall_refinement_gate" / "falsifiability_audit.json"
FIGURE = REPO_ROOT / "figures" / "wall_refinement_gate_admission_matrix.png"


def sweep() -> list[gate.CombinationGateDecision]:
    decisions = []
    for lattice, family, wall, ref, own, geom, phys in itertools.product(
        gate.Lattice,
        gate.CollisionFamily,
        gate.WallTreatment,
        gate.RefinementType,
        gate.GeometryOwnership,
        gate.GeometryKind,
        gate.PhysicsModel,
    ):
        combination = gate.WallRefinementCombination(
            lattice=lattice,
            collision=family,
            wall_treatment=wall,
            refinement=ref,
            geometry_ownership=own,
            geometry_kind=geom,
            physics=phys,
        )
        decisions.append(gate.assess_wall_refinement_combination(combination))
    return decisions


def decision_table() -> dict[tuple[str, str, str, str], bool]:
    """Admission over (lattice, family, wall_treatment, refinement) on the plain shape."""
    table = {}
    for lattice, family, wall, ref in itertools.product(
        gate.Lattice, gate.CollisionFamily, gate.WallTreatment, gate.RefinementType
    ):
        combination = gate.WallRefinementCombination(
            lattice=lattice,
            collision=family,
            wall_treatment=wall,
            refinement=ref,
            geometry_ownership=gate.GeometryOwnership.SINGLE_LEVEL,
        )
        table[(lattice.value, family.value, wall.value, ref.value)] = gate.assess_wall_refinement_combination(
            combination
        ).allowed
    return table


def differential_arm() -> list[dict]:
    """Withdraw one matrix cell at a time; the gate must flip exactly those decisions."""
    base_matrix = gate.collision_capability_matrix()
    base_table = decision_table()
    original = gate.collision_capability_matrix
    arms = []
    try:
        for lattice, family in itertools.product(gate.Lattice, gate.CollisionFamily):
            matrix = json.loads(json.dumps({"cells": {}}))  # placeholder, replaced below
            matrix = {
                lat: {fam: replace(cap) for fam, cap in fams.items()}
                for lat, fams in base_matrix.items()
            }
            matrix[lattice.value][family.value] = replace(
                matrix[lattice.value][family.value], available=False, status="WITHDRAWN_FOR_AUDIT"
            )
            gate.collision_capability_matrix = lambda m=matrix: m
            arm_table = decision_table()
            flipped = [key for key, allowed in base_table.items() if arm_table[key] != allowed]
            expected = [
                key
                for key, allowed in base_table.items()
                if allowed
                and key[0] == lattice.value
                and key[1] == family.value
            ]
            arms.append(
                {
                    "withdrawn_cell": [lattice.value, family.value],
                    "n_allowed_baseline": len(expected),
                    "n_flipped": len(flipped),
                    "flipped_all_and_only_that_family": set(flipped) == set(expected),
                    "unexpected_flips": sorted(set(flipped) - set(expected))[:5],
                }
            )
    finally:
        gate.collision_capability_matrix = original
    return arms


def reason_reachability() -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in sweep():
        for reason in decision.reasons:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def witness_coverage(arms: list[dict]) -> dict[str, object]:
    """Reasons reachable from public input only, versus needing an injected matrix."""
    defined = {name for name in dir(gate) if name.startswith("WITHHELD_") and name.isupper()}
    public = set(reason_reachability())
    injected = set()
    original = gate.collision_capability_matrix
    try:
        matrix = {
            lat: {fam: replace(cap) for fam, cap in fams.items()}
            for lat, fams in original().items()
        }
        for lat in matrix:
            for fam in matrix[lat]:
                matrix[lat][fam] = replace(matrix[lat][fam], available=False)
        gate.collision_capability_matrix = lambda m=matrix: m
        injected = set(reason_reachability())
    finally:
        gate.collision_capability_matrix = original
    return {
        "defined": sorted(defined),
        "reachable_from_public_input": sorted(public),
        "reachable_only_with_injected_matrix": sorted(injected - public),
        "never_reachable": sorted(defined - public - injected),
    }


def render_figure(table: dict[tuple[str, str, str, str], bool], path: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    walls = [w.value for w in gate.WallTreatment]
    refs = [r.value for r in gate.RefinementType]
    families = [f.value for f in gate.CollisionFamily]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    for ax, lattice in zip(axes, [gate.Lattice.D3Q19, gate.Lattice.D3Q27]):
        grid = np.zeros((len(families), len(walls) * len(refs)))
        cols = []
        for j, (wall, ref) in enumerate(itertools.product(walls, refs)):
            cols.append(f"{wall}\n{ref}")
            for i, family in enumerate(families):
                grid[i, j] = 1.0 if table[(lattice.value, family, wall, ref)] else 0.0
        ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=90, fontsize=6)
        ax.set_yticks(range(len(families)))
        ax.set_yticklabels(families, fontsize=8)
        ax.set_title(f"{lattice.value}: gate decision (green = ALLOWED) on the plain shape")
        for i in range(len(families)):
            for j in range(len(cols)):
                ax.text(
                    j,
                    i,
                    "A" if grid[i, j] else "W",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black",
                )
    fig.suptitle(
        "Fail-closed wall/refinement combination gate: admission surface "
        "(single-level planar single-phase; A=ALLOWED, W=WITHHELD)"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    decisions = sweep()
    n_allowed = sum(1 for d in decisions if d.allowed)
    n_withheld = len(decisions) - n_allowed
    table = decision_table()
    arms = differential_arm()
    coverage = witness_coverage(arms)
    payload = {
        "anchor": "m16911 / test_unavailable_collision_contract_cannot_be_promoted_by_baseline_shape",
        "space": {
            "n_points": len(decisions),
            "n_allowed": n_allowed,
            "n_withheld": n_withheld,
            "dimensions": "2 lattice x 3 collision x 4 wall x 4 refinement x 4 ownership x 3 geometry x 2 physics",
        },
        "reason_histogram": reason_reachability(),
        "reason_coverage": coverage,
        "differential_arms": arms,
        "differential_arms_all_exact": all(a["flipped_all_and_only_that_family"] for a in arms),
        "plain_shape_admission": {
            "|".join(key): value for key, value in sorted(table.items())
        },
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    render_figure(table, FIGURE)

    print(f"points={len(decisions)} allowed={n_allowed} withheld={n_withheld}")
    print(f"reason histogram: {payload['reason_histogram']}")
    print(f"never reachable: {coverage['never_reachable']}")
    print(f"injected-only: {coverage['reachable_only_with_injected_matrix']}")
    print(f"differential arms exact: {payload['differential_arms_all_exact']}")
    for arm in arms:
        print(
            f"  withdraw {arm['withdrawn_cell']}: flipped {arm['n_flipped']} "
            f"(expected {arm['n_allowed_baseline']}) exact={arm['flipped_all_and_only_that_family']}"
        )
    print(f"artifact -> {ARTIFACT.relative_to(REPO_ROOT)}")
    print(f"figure   -> {FIGURE.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()