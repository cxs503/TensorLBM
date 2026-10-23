"""Case runner: step loop + optional solver→data export/registration.

Ties the case registry to the #182 solver-export path
(:mod:`tensorlbm.data.solver_export`): when an :class:`ExportSpec` is
given, snapshots are written with :func:`save_fields_hdf5` and registered
as PASS-gated ``FieldDataProductR2`` assets in a
:class:`~tensorlbm.data.catalog.FieldDataCatalog`, so every registry run
can land directly in the AI4S data catalog with full lineage.  Export is
strictly opt-in (``export=None`` keeps the runner free of h5py and I/O).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

from .base import CaseBase

__all__ = ["CaseRunResult", "ExportSpec", "run_case"]


@dataclass
class ExportSpec:
    """Opt-in snapshot export + catalog registration settings.

    Attributes:
        h5_path: HDF5 file for snapshots (``tensorlbm.solver-export/v1``).
        catalog: ``FieldDataCatalog`` to register products into.
        code_sha: 40-hex solver revision (see
            :func:`tensorlbm.data.solver_export.register_product`).
        snapshot_every: write a snapshot every N steps (0 = final only).
        run_id: registration run id (defaults to ``"{case}-{nx}x{nz}"``).
        extra_metadata: additional metadata rows for every snapshot.
    """

    h5_path: str | Path
    catalog: Any
    code_sha: str
    snapshot_every: int = 0
    run_id: str | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseRunResult:
    """Final state and provenance of one :func:`run_case` execution."""

    case: CaseBase
    steps: int
    f: torch.Tensor
    rho: torch.Tensor
    ux: torch.Tensor
    uy: torch.Tensor
    uz: torch.Tensor
    elapsed_s: float
    product_ids: list[str] = field(default_factory=list)


def _export_snapshot(
    case: CaseBase,
    f: torch.Tensor,
    step: int,
    export: ExportSpec,
    run_id: str,
) -> str:
    from ..data.solver_export import register_product, save_fields_hdf5

    if case.lattice == "D3Q27":
        from ..d3q27 import macroscopic27

        rho, ux, uy, uz = macroscopic27(f)
    else:
        from ..d3q19 import macroscopic3d

        rho, ux, uy, uz = macroscopic3d(f)
    arrays: dict[str, Any] = {"rho": rho, "ux": ux, "uy": uy, "uz": uz}
    solid = case.solid_mask()
    if solid is not None:
        arrays["solid_mask"] = solid

    attrs = dict(case.metadata())
    attrs.update(export.extra_metadata)
    attrs["run_id"] = run_id
    attrs["step"] = int(step)
    attrs["code_sha"] = export.code_sha
    attrs["n_steps"] = int(step)
    save_fields_hdf5(export.h5_path, arrays, attrs)
    return register_product(export.catalog, export.h5_path, attrs)


def run_case(
    case: CaseBase | str,
    steps: int,
    *,
    resolution: int | tuple[int, int, int] | None = None,
    re: float | None = None,
    export: ExportSpec | None = None,
    progress: Callable[[int, torch.Tensor], None] | None = None,
    **case_kwargs: Any,
) -> CaseRunResult:
    """Run *steps* LBM steps of a case (by name or instance).

    The loop reproduces the worker order exactly:
    ``f = step(f)`` then the case's periodic mass correction (if any)
    then snapshot export (if requested).

    Args:
        case: registered case name or a :class:`CaseBase` instance.
        steps: number of LBM steps to execute.
        resolution, re, **case_kwargs: constructor arguments when *case*
            is a name.
        export: optional :class:`ExportSpec` (HDF5 + catalog registration).
        progress: optional ``progress(step, f)`` callback (outside the
            hot path; called after every step).

    Returns:
        :class:`CaseRunResult` with the final populations, macroscopic
        fields and registered product ids.
    """
    if isinstance(case, str):
        from .registry import get_case

        case = get_case(case, resolution=resolution, re=re, **case_kwargs)
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}")

    if case.lattice == "D3Q27":
        from ..d3q27 import macroscopic27 as _macro
    else:
        from ..d3q19 import macroscopic3d as _macro

    f = case.initial_f()
    initial_mass = float(f.sum().item())
    step_fn = case.make_step()
    mass_every = int(getattr(case, "mass_correction_interval", 0))
    if mass_every > 0:
        from ..solver3d import correct_mass3d

    product_ids: list[str] = []
    run_id = (
        export.run_id
        if export is not None and export.run_id is not None
        else f"{case.name}-{case.resolution[2]}x{case.resolution[0]}"
    )

    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        f = step_fn(f)
        if mass_every > 0 and step % mass_every == 0:
            f = correct_mass3d(f, initial_mass)
        if export is not None and step != steps:
            if export.snapshot_every > 0 and step % export.snapshot_every == 0:
                product_ids.append(_export_snapshot(case, f, step, export, run_id))
        if progress is not None:
            progress(step, f)
    if export is not None:
        # Final snapshot is always registered (matching examples/ai4s_export.py).
        product_ids.append(_export_snapshot(case, f, steps, export, run_id))
    elapsed = time.perf_counter() - t0

    rho, ux, uy, uz = _macro(f)
    return CaseRunResult(
        case=case,
        steps=steps,
        f=f,
        rho=rho,
        ux=ux,
        uy=uy,
        uz=uz,
        elapsed_s=elapsed,
        product_ids=product_ids,
    )
