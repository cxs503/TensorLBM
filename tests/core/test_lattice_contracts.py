"""Contract tests for domain-neutral lattice descriptors."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tensorlbm.core.lattice import D3Q19, D3Q27, LatticeDescriptor


@pytest.mark.parametrize("lattice, q", [(D3Q19, 19), (D3Q27, 27)])
def test_existing_three_dimensional_lattices_adapt_to_descriptor(
    lattice: LatticeDescriptor, q: int
) -> None:
    assert lattice.q == q
    assert len(lattice.directions) == q
    assert all(len(direction) == 3 for direction in lattice.directions)
    assert len(lattice.weights) == q
    assert len(lattice.opposite) == q
    assert lattice.cs2 == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize("lattice", [D3Q19, D3Q27])
def test_lattice_weights_sum_to_one(lattice: LatticeDescriptor) -> None:
    assert sum(lattice.weights) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("lattice", [D3Q19, D3Q27])
def test_opposite_directions_are_involutive_negations(lattice: LatticeDescriptor) -> None:
    assert tuple(lattice.opposite[index] for index in lattice.opposite) == tuple(range(lattice.q))
    assert tuple(lattice.directions[index] for index in lattice.opposite) == tuple(
        tuple(-component for component in direction) for direction in lattice.directions
    )


def test_lattice_descriptor_source_has_no_tensor_framework_import() -> None:
    source = Path("src/tensorlbm/core/lattice.py").read_text(encoding="utf-8")
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert imported_roots.isdisjoint({"torch", "paddle", "mindspore"})
