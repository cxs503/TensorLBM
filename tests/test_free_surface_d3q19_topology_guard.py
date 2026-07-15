"""Strict post-conversion D3Q19 topology contract.

This test owns an oracle independent from the production neighbour traversal.
It validates every moving D3Q19 link, including edge diagonals and periodic
seams, across a real conversion plus its following real timestep.
"""
from __future__ import annotations

import ast
import inspect

import pytest
import torch

from tensorlbm.d3q19 import C, equilibrium3d
import tensorlbm.free_surface_lbm as free_surface_lbm
from tensorlbm.free_surface_lbm import GAS, INTERFACE, LIQUID, free_surface_step


MOVING_LINKS = tuple((q, tuple(int(component) for component in C[q])) for q in range(1, 19))


def _case_name(case: tuple[int, tuple[int, int, int]]) -> str:
    q, (cx, cy, cz) = case
    return f"q{q}-c({cx:+d},{cy:+d},{cz:+d})"


def _field_delta(q: int) -> tuple[int, int, int]:
    """Map lattice C(x, y, z) into state-tensor coordinates (z, y, x)."""
    return int(C[q, 2]), int(C[q, 1]), int(C[q, 0])


def _source_of(site: tuple[int, int, int], q: int, shape: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple((index - delta) % extent for index, delta, extent in zip(site, _field_delta(q), shape))  # type: ignore[return-value]


def _assert_full18_separation(flags: torch.Tensor) -> None:
    violations: dict[str, int] = {}
    for q, velocity in MOVING_LINKS:
        is_direct = (flags == LIQUID) & (flags.roll(_field_delta(q), dims=(0, 1, 2)) == GAS)
        if bool(is_direct.any()):
            violations[f"q{q}:{velocity}"] = int(is_direct.sum())
    assert not violations, f"direct D3Q19 LIQUID→GAS links: {violations}"


def _fixture(selected_q: int, at_seam: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (5, 6, 7)
    if at_seam:
        dz, dy, dx = _field_delta(selected_q)
        centre: tuple[int, int, int] = (
            0 if dz > 0 else shape[0] - 1 if dz < 0 else shape[0] // 2,
            0 if dy > 0 else shape[1] - 1 if dy < 0 else shape[1] // 2,
            0 if dx > 0 else shape[2] - 1 if dx < 0 else shape[2] // 2,
        )
    else:
        centre = (2, 3, 3)
    flags = torch.full(shape, GAS, dtype=torch.int8)
    fill = torch.zeros(shape)
    flags[centre] = INTERFACE
    fill[centre] = 1.0
    for q, _ in MOVING_LINKS:
        if q != selected_q:
            flags[_source_of(centre, q, shape)] = INTERFACE
            fill[_source_of(centre, q, shape)] = 0.5
    zeros = torch.zeros(shape)
    populations = equilibrium3d(torch.ones(shape), zeros, zeros, zeros)
    return populations, fill, flags, torch.zeros(shape, dtype=torch.bool)


@pytest.mark.parametrize("case", MOVING_LINKS, ids=_case_name)
@pytest.mark.parametrize("at_seam", [False, True], ids=["interior", "periodic-seam"])
def test_conversion_retains_every_d3q19_interface_link(
    case: tuple[int, tuple[int, int, int]], at_seam: bool
) -> None:
    """A conversion must preserve the full 18-link L/INTERFACE/G envelope."""
    q, _ = case
    f, fill, flags, solid = _fixture(q, at_seam)
    f, fill, flags, mass, _ = free_surface_step(f, fill, flags, solid, mass=fill.clone())
    _assert_full18_separation(flags)
    _, _, next_flags, _, _ = free_surface_step(f, fill, flags, solid, mass=mass)
    _assert_full18_separation(next_flags)


def test_topology_paths_use_canonical_moving_stencil_utility() -> None:
    """Topology traversal must not recreate a (z, y, x) D3Q19 shift table."""
    source = inspect.getsource(free_surface_lbm)
    tree = ast.parse(source)
    assert "_D3Q19_TENSOR_SHIFTS" not in source

    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "core.d3q19_stencil"
        for alias in node.names
    }
    expected = {
        "all_moving_neighbor_masks",
        "assert_no_direct_phase_links",
        "moving_tensor_shifts",
        "roll_from_pull_source",
        "roll_to_neighbor",
    }
    assert expected <= imported_names

    used_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert expected <= used_names
