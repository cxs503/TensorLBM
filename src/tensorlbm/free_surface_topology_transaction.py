"""Detached, fail-closed topology commit for D3Q19 free-surface states.

The module owns only the cold conversion path.  It deliberately receives flag
values and precomputed redistribution increments from the hot solver so it has
no dependency on :mod:`free_surface_lbm` and cannot alter collision, streaming,
ABB, or mass-exchange arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import pickle
import weakref
from typing import Mapping

import torch

from .core.d3q19_stencil import (
    D3Q19_MOVING_Q,
    all_moving_neighbor_masks,
    assert_no_direct_phase_links,
    moving_tensor_shifts,
    roll_from_pull_source,
    roll_to_neighbor,
)
from .d3q19 import equilibrium3d
from .free_surface_inventory_reconciliation import inventory_measurement


class TopologyTransactionError(ValueError):
    """A staged topology candidate violated the solver's terminal contract."""


@dataclass(frozen=True)
class ReplayTensorRecord:
    """Description of a tensor serialized into immutable replay evidence."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    sha256: str


@dataclass(frozen=True)
class ReplayEvidence:
    """In-process capture-integrity evidence, not provenance or attestation.

    Hashes detect accidental/view corruption for an object captured in this
    process.  They do not make publicly constructed or cross-process evidence
    trustworthy; audits require a private in-process identity capability.
    """

    invocation_payload: bytes
    invocation_sha256: str
    phase_payload: bytes
    phase_sha256: str
    candidate_payload: bytes
    candidate_sha256: str
    tensor_records: tuple[ReplayTensorRecord, ...]


_TRUSTED_REPLAY_EVIDENCE: dict[int, weakref.ReferenceType[ReplayEvidence]] = {}


def _register_trusted_replay_evidence(evidence: ReplayEvidence) -> ReplayEvidence:
    """Give a production capture an identity-bound, process-local capability."""
    evidence_id = id(evidence)

    def _discard(_: weakref.ReferenceType[ReplayEvidence]) -> None:
        _TRUSTED_REPLAY_EVIDENCE.pop(evidence_id, None)

    _TRUSTED_REPLAY_EVIDENCE[evidence_id] = weakref.ref(evidence, _discard)
    return evidence


def is_trusted_replay_evidence(evidence: object) -> bool:
    """True only for the exact object emitted by this process' capture path."""
    reference = _TRUSTED_REPLAY_EVIDENCE.get(id(evidence))
    return reference is not None and reference() is evidence


def _freeze_replay_payload(value: object) -> tuple[bytes, str]:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    payload = buffer.getvalue()
    return payload, hashlib.sha256(payload).hexdigest()


def restore_replay_payload(payload: bytes, expected_sha256: str) -> object:
    """Safely restore a hash-checked primitive/tensor capture payload."""
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise TopologyTransactionError("WITHHELD: replay evidence digest mismatch")
    try:
        return torch.load(io.BytesIO(payload), weights_only=True)
    except (RuntimeError, ValueError, TypeError, pickle.UnpicklingError) as error:
        raise TopologyTransactionError("WITHHELD: replay evidence payload cannot be safely loaded") from error


def _replay_tensor_records(value: object, prefix: str = "") -> tuple[ReplayTensorRecord, ...]:
    if isinstance(value, torch.Tensor):
        _, digest = _freeze_replay_payload(value.detach().clone())
        return (ReplayTensorRecord(prefix, str(value.dtype), tuple(value.shape), digest),)
    if isinstance(value, Mapping):
        return tuple(
            record
            for key, item in value.items()
            for record in _replay_tensor_records(item, f"{prefix}.{key}" if prefix else str(key))
        )
    if isinstance(value, tuple):
        return tuple(
            record
            for index, item in enumerate(value)
            for record in _replay_tensor_records(item, f"{prefix}[{index}]")
        )
    return ()


@dataclass(frozen=True)
class IToGOwnershipTransaction:
    """Exact local independent-mass debit/credit for staged INTERFACE→GAS cells.

    ``donor_debit`` and ``receiver_credit`` are independently scatter/reduced
    same-dtype tensors.  They are signed records, never aliases of a common
    credit sum: a nonzero residual is WITHHELD rather than accepted under a
    numerical tolerance.
    """

    receiver_increment: torch.Tensor
    receiver_mask: torch.Tensor
    links: tuple[dict[str, object], ...]
    donor_debit: torch.Tensor
    receiver_credit: torch.Tensor
    residual: torch.Tensor
    donor_debit_records: torch.Tensor
    receiver_credit_records: torch.Tensor

    def validate(self) -> None:
        """Reject altered or non-exact I→G ownership evidence fail-closed."""
        if self.donor_debit.dtype != self.receiver_credit.dtype:
            raise TopologyTransactionError(
                "WITHHELD: entire free_surface_step topology candidate has mixed I→G debit/credit dtypes"
            )
        if not bool(self.donor_debit == self.donor_debit_records.sum()) or not bool(self.receiver_credit == self.receiver_credit_records.sum()):
            raise TopologyTransactionError(
                "WITHHELD: entire free_surface_step topology candidate has tampered I→G debit/credit records"
            )
        if not torch.equal(self.receiver_increment, self.receiver_credit_records):
            raise TopologyTransactionError(
                "WITHHELD: entire free_surface_step topology candidate has tampered I→G receiver credit aggregation"
            )
        actual_residual = self.donor_debit + self.receiver_credit
        if not bool(actual_residual == 0):
            raise TopologyTransactionError(
                "WITHHELD: entire free_surface_step topology candidate has non-exact I→G debit/credit closure"
            )
        if not bool(self.residual == actual_residual):
            raise TopologyTransactionError(
                "WITHHELD: entire free_surface_step topology candidate has tampered I→G debit/credit residual"
            )


def _serialize_i_to_g_ownership(value: IToGOwnershipTransaction | None) -> dict[str, object] | None:
    """Encode ownership as tensors/primitives accepted by safe torch loading."""
    if value is None:
        return None
    return {
        "receiver_increment": value.receiver_increment.clone(),
        "receiver_mask": value.receiver_mask.clone(),
        "donor_debit": value.donor_debit.clone(),
        "receiver_credit": value.receiver_credit.clone(),
        "residual": value.residual.clone(),
        "donor_debit_records": value.donor_debit_records.clone(),
        "receiver_credit_records": value.receiver_credit_records.clone(),
    }


def restore_i_to_g_ownership(value: object) -> IToGOwnershipTransaction | None:
    """Validate safe payload data before restoring the transaction API object."""
    if value is None:
        return None
    required = (
        "receiver_increment", "receiver_mask", "donor_debit", "receiver_credit",
        "residual", "donor_debit_records", "receiver_credit_records",
    )
    if not isinstance(value, dict) or set(value) != set(required) or any(
        not isinstance(value[name], torch.Tensor) for name in required
    ):
        raise TopologyTransactionError("WITHHELD: captured I→G ownership has invalid schema")
    transaction = IToGOwnershipTransaction(
        value["receiver_increment"], value["receiver_mask"], (), value["donor_debit"],
        value["receiver_credit"], value["residual"], value["donor_debit_records"],
        value["receiver_credit_records"],
    )
    transaction.validate()
    return transaction


def build_i_to_g_ownership_transaction(
    flags: torch.Tensor, mass: torch.Tensor, *, to_gas: torch.Tensor,
    to_liq: torch.Tensor, solid_mask: torch.Tensor, gas_flag: int,
    liquid_flag: int, interface_flag: int, rho_liquid: float,
) -> IToGOwnershipTransaction:
    """Construct a local, paired D3Q19 I→G mass transaction or fail closed.

    Only independent mass/fill ownership moves. Populations remain kinetic
    state: transferring them to an INTERFACE receiver would double-count a
    quantity not owned by the independent mass field.
    """
    if to_gas.shape != flags.shape or to_liq.shape != flags.shape or mass.shape != flags.shape:
        raise TopologyTransactionError("I→G ownership fields must match flags shape")
    donor = to_gas & ~solid_mask
    if bool((donor & (flags != interface_flag)).any()):
        raise TopologyTransactionError("WITHHELD: LIQUID→GAS has no declared local inventory owner")
    receiver_mask = (flags == interface_flag) & ~to_gas & ~to_liq & ~solid_mask
    receiver_by_q = torch.stack(all_moving_neighbor_masks(receiver_mask))
    receiver_count = receiver_by_q.sum(dim=0)
    if bool((donor & (receiver_count == 0)).any()):
        raise TopologyTransactionError(
            "WITHHELD: entire free_surface_step topology candidate I→G donor has no legal INTERFACE receiver"
        )
    debit_field = torch.where(donor, mass, torch.zeros_like(mass))
    credit_per_link = debit_field / receiver_count.clamp(min=1).to(mass.dtype)
    increment = torch.stack([
        roll_to_neighbor(credit_per_link, q) * receiver_mask for q in D3Q19_MOVING_Q
    ]).sum(dim=0)
    capacity = torch.where(receiver_mask, float(rho_liquid) - mass, torch.zeros_like(mass))
    if bool((increment > capacity).any()):
        raise TopologyTransactionError(
            "WITHHELD: entire free_surface_step topology candidate I→G receiver capacity would overflow"
        )
    shape = tuple(int(value) for value in mass.shape)
    links: list[dict[str, object]] = []
    for q, shift in zip(D3Q19_MOVING_Q, moving_tensor_shifts()):
        dz, dy, dx = shift
        receiver_for_donor = roll_from_pull_source(receiver_mask, q)
        for raw in torch.nonzero(donor & receiver_for_donor, as_tuple=False).tolist():
            z, y, x = (int(value) for value in raw)
            credit = credit_per_link[z, y, x]
            links.append({"donor": (z, y, x), "receiver": ((z - dz) % shape[0], (y - dy) % shape[1], (x - dx) % shape[2]), "q": q, "shift": (dz, dy, dx), "debit": float(-credit), "credit": float(credit), "event_id": "i_to_g_independent_mass_ownership", "operator": "i_to_g_independent_mass_ownership"})
    transaction = IToGOwnershipTransaction(
        increment, receiver_mask & (increment != 0.0), tuple(links),
        -debit_field.sum(), increment.sum(), torch.zeros((), dtype=mass.dtype, device=mass.device),
        -debit_field, increment,
    )
    transaction = IToGOwnershipTransaction(
        transaction.receiver_increment, transaction.receiver_mask, transaction.links,
        transaction.donor_debit, transaction.receiver_credit,
        transaction.donor_debit + transaction.receiver_credit,
        transaction.donor_debit_records, transaction.receiver_credit_records,
    )
    transaction.validate()
    return transaction


@dataclass(frozen=True)
class TopologyTransactionPlan:
    """Immutable handle for a fully detached staged conversion candidate."""

    f: torch.Tensor
    fill: torch.Tensor
    flags: torch.Tensor
    mass: torch.Tensor
    mass_after_redistribution: float
    mass_after_clamp: float
    mass_after_conversion: float
    mass_after_isolation: float
    inventory_stages: dict[str, dict[str, float]] | None
    conversion_evidence: dict[str, object] | None
    gas_flag: int
    liquid_flag: int
    interface_flag: int
    solid_flag: int
    solid_mask: torch.Tensor
    _replay_evidence: ReplayEvidence | None = None

    @property
    def replay_evidence(self) -> ReplayEvidence | None:
        """Immutable serialized evidence; never a writable tensor/mapping view."""
        return self._replay_evidence

    @property
    def replay_stages(self) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None:
        """Compatibility copy rebuilt from verified private evidence each access."""
        if self._replay_evidence is None:
            return None
        restored = restore_replay_payload(
            self._replay_evidence.phase_payload, self._replay_evidence.phase_sha256,
        )
        if not isinstance(restored, dict):
            raise TopologyTransactionError("WITHHELD: replay phase evidence has invalid schema")
        return restored


def _init_new(
    f: torch.Tensor, flags: torch.Tensor, mask: torch.Tensor, rho_init: float,
    ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor, liquid_flag: int, interface_flag: int,
) -> torch.Tensor:
    active = (flags == liquid_flag) | (flags == interface_flag)
    neighbours = torch.stack(all_moving_neighbor_masks(active)).to(f.dtype)
    count = neighbours.sum(dim=0).clamp(min=1)
    # pull-source rolls are generated through the canonical stencil helper.
    from .core.d3q19_stencil import roll_from_pull_source
    ux_mean = (torch.stack([roll_from_pull_source(ux, q) for q in D3Q19_MOVING_Q]) * neighbours).sum(dim=0) / count
    uy_mean = (torch.stack([roll_from_pull_source(uy, q) for q in D3Q19_MOVING_Q]) * neighbours).sum(dim=0) / count
    uz_mean = (torch.stack([roll_from_pull_source(uz, q) for q in D3Q19_MOVING_Q]) * neighbours).sum(dim=0) / count
    feq = equilibrium3d(torch.full_like(ux, float(rho_init)), ux_mean, uy_mean, uz_mean)
    return torch.where(mask.unsqueeze(0), feq, f)


def _validate_candidate(
    f: torch.Tensor, fill: torch.Tensor, flags: torch.Tensor, mass: torch.Tensor,
    solid_mask: torch.Tensor, gas_flag: int, liquid_flag: int, interface_flag: int, solid_flag: int,
) -> None:
    fields = {"f": f, "fill": fill, "mass": mass}
    for name, field in fields.items():
        if not bool(torch.isfinite(field).all()):
            raise TopologyTransactionError(f"topology candidate has non-finite {name}")
    legal = (flags == gas_flag) | (flags == liquid_flag) | (flags == interface_flag) | (flags == solid_flag)
    if not bool(legal.all()):
        raise TopologyTransactionError("topology candidate has an invalid flag value")
    if not bool((flags[solid_mask] == solid_flag).all()):
        raise TopologyTransactionError("topology candidate violates solid flag consistency")
    if bool((fill < 0).any()) or bool((fill > 1).any()):
        raise TopologyTransactionError("topology candidate fill is outside [0, 1]")
    if bool((mass < 0).any()):
        raise TopologyTransactionError("topology candidate has negative mass")
    try:
        assert_no_direct_phase_links(flags, liquid_flag, gas_flag, "direct LIQUID-GAS D3Q19")
    except ValueError as error:
        raise TopologyTransactionError(str(error)) from error


def build_topology_transaction(
    f: torch.Tensor, fill: torch.Tensor, flags: torch.Tensor, mass: torch.Tensor, *,
    to_iface: torch.Tensor, to_liq: torch.Tensor, to_gas: torch.Tensor, recv_new: torch.Tensor,
    redistribution_increment: torch.Tensor, rho_liquid: float, rho_gas: float, solid_mask: torch.Tensor,
    gas_flag: int, liquid_flag: int, interface_flag: int, solid_flag: int,
    ux: torch.Tensor | None = None, uy: torch.Tensor | None = None, uz: torch.Tensor | None = None,
    capture_evidence: bool = False,
    capture_inventory: bool = False,
    redistribution_link_evidence: tuple[dict[str, object], ...] = (),
    i_to_g_increment: torch.Tensor | None = None,
    i_to_g_ownership: IToGOwnershipTransaction | None = None,
    capture_replay_stages: bool = False,
) -> TopologyTransactionPlan:
    """Build a detached candidate in the legacy conversion/halo/cleanup order.

    With I→G ownership, this is an all-or-nothing ``free_surface_step``
    topology candidate: any capacity or accounting failure is WITHHELD, with
    no partial topology publication.
    """
    if ux is None or uy is None or uz is None:
        raise TopologyTransactionError("topology transaction requires pre-conversion velocity fields")
    cf, cfill, cflags, cmass = (value.clone() for value in (f, fill, flags, mass))
    if (i_to_g_increment is None) != (i_to_g_ownership is None):
        raise TopologyTransactionError("I→G increment and ownership evidence must be supplied together")
    if i_to_g_ownership is not None:
        i_to_g_ownership.validate()
        if i_to_g_increment is None or i_to_g_increment.shape != flags.shape:
            raise TopologyTransactionError("I→G increment must match topology fields")
        if not torch.equal(i_to_g_increment, i_to_g_ownership.receiver_increment):
            raise TopologyTransactionError(
                "WITHHELD: entire free_surface_step topology candidate has tampered I→G increment evidence"
            )
        if i_to_g_ownership.receiver_mask.shape != flags.shape:
            raise TopologyTransactionError("I→G receiver mask must match topology fields")
        if bool((i_to_g_ownership.receiver_mask & ((flags != interface_flag) | to_gas | to_liq)).any()):
            raise TopologyTransactionError("I→G receiver must be a surviving pre-topology INTERFACE cell")
    inventory_stages = {} if capture_inventory else None
    replay_stages = {} if capture_replay_stages else None
    replay_invocation = None
    if capture_replay_stages:
        replay_invocation = {
            "f": f.clone(), "fill": fill.clone(), "flags": flags.clone(), "mass": mass.clone(),
            "to_iface": to_iface.clone(), "to_liq": to_liq.clone(), "to_gas": to_gas.clone(),
            "recv_new": recv_new.clone(), "redistribution_increment": redistribution_increment.clone(),
            "rho_liquid": rho_liquid, "rho_gas": rho_gas, "solid_mask": solid_mask.clone(),
            "gas_flag": gas_flag, "liquid_flag": liquid_flag, "interface_flag": interface_flag,
            "solid_flag": solid_flag, "ux": ux.clone(), "uy": uy.clone(), "uz": uz.clone(),
            "i_to_g_increment": None if i_to_g_increment is None else i_to_g_increment.clone(),
            "i_to_g_ownership": _serialize_i_to_g_ownership(i_to_g_ownership),
        }
    gas_mask = cflags == gas_flag
    cf = _init_new(cf, cflags, to_iface, rho_gas, ux, uy, uz, liquid_flag, interface_flag)
    cflags = torch.where(to_iface, torch.full_like(cflags, interface_flag), cflags)
    if replay_stages is not None:
        replay_stages["to_iface_initialization"] = tuple(value.clone() for value in (cf, cfill, cflags, cmass))

    mass_before_redistribution = cmass.clone() if capture_evidence else None
    combined_increment = redistribution_increment
    if i_to_g_increment is not None:
        assert i_to_g_ownership is not None
        combined_increment = redistribution_increment + i_to_g_increment
        combined_receivers = i_to_g_ownership.receiver_mask
        combined_candidate = cmass + combined_increment
        if bool((((combined_candidate < 0.0) | (combined_candidate > float(rho_liquid))) & combined_receivers).any()):
            raise TopologyTransactionError(
                "WITHHELD: entire free_surface_step topology candidate combined receiver capacity is outside [0, rho_liquid] before clamp"
            )
    cmass = cmass + combined_increment
    if replay_stages is not None:
        replay_stages["legacy_redistribution_and_i_to_g_increment"] = tuple(value.clone() for value in (cf, cfill, cflags, cmass))
    if inventory_stages is not None:
        inventory_stages["after_topology_redistribution"] = inventory_measurement(
            cf, cfill, cflags, cmass, rho_liquid=rho_liquid,
        )
    mass_after_redistribution = float(cmass.sum())
    cmass = cmass.clamp(0.0, rho_liquid)
    if replay_stages is not None:
        replay_stages["clamp"] = tuple(value.clone() for value in (cf, cfill, cflags, cmass))
    if inventory_stages is not None:
        inventory_stages["after_topology_clamp"] = inventory_measurement(
            cf, cfill, cflags, cmass, rho_liquid=rho_liquid,
        )
    mass_after_clamp = float(cmass.sum())
    mass_before_conversion = cmass.clone() if capture_evidence else None
    fill_before_conversion = cfill.clone() if capture_evidence else None
    flags_before_conversion = cflags.clone() if capture_evidence else None
    f_before_conversion = cf.clone() if capture_evidence else None

    cflags = torch.where(to_liq, torch.full_like(cflags, liquid_flag), cflags)
    cfill = torch.where(to_liq, torch.ones_like(cfill), cfill)
    cmass = torch.where(to_liq, torch.full_like(cmass, rho_liquid), cmass)
    cflags = torch.where(to_gas, torch.full_like(cflags, gas_flag), cflags)
    cfill = torch.where(to_gas, torch.zeros_like(cfill), cfill)
    cmass = torch.where(to_gas, torch.zeros_like(cmass), cmass)
    cf = torch.where(to_gas.unsqueeze(0), torch.zeros_like(cf), cf)
    if i_to_g_ownership is not None:
        # No f transfer: independent mass/fill and population density are
        # separate representations at INTERFACE, so copying f would double count.
        cfill = torch.where(i_to_g_ownership.receiver_mask, cmass / float(rho_liquid), cfill)
    if replay_stages is not None:
        replay_stages["to_liq_to_gas_conversion"] = tuple(value.clone() for value in (cf, cfill, cflags, cmass))
    if inventory_stages is not None:
        inventory_stages["after_topology_conversion"] = inventory_measurement(
            cf, cfill, cflags, cmass, rho_liquid=rho_liquid,
        )
    mass_after_conversion = float(cmass.sum())
    # Evidence attributes conversion itself, not the subsequent envelope halo.
    # Preserve the legacy post-conversion/pre-halo observation boundary.
    flags_after_conversion = cflags.clone() if capture_evidence else None
    fill_after_conversion = cfill.clone() if capture_evidence else None
    mass_after_conversion_field = cmass.clone() if capture_evidence else None
    f_after_conversion = cf.clone() if capture_evidence else None

    shifted_flags = torch.stack(all_moving_neighbor_masks(cflags))
    is_neighbor = ((shifted_flags == liquid_flag) | (shifted_flags == interface_flag)).any(dim=0)
    to_i = (((gas_mask | to_gas) & is_neighbor & ~solid_mask) | recv_new)
    cf = _init_new(cf, cflags, to_i, rho_gas, ux, uy, uz, liquid_flag, interface_flag)
    cflags = torch.where(to_i, torch.full_like(cflags, interface_flag), cflags)
    cfill = torch.where(to_i & ~recv_new, torch.zeros_like(cfill), cfill)
    cmass = torch.where(to_i & ~recv_new, torch.zeros_like(cmass), cmass)
    if replay_stages is not None:
        replay_stages["halo_boundary"] = tuple(value.clone() for value in (cf, cfill, cflags, cmass))
    interface_mask = cflags == interface_flag
    has_neighbor = ((torch.stack(all_moving_neighbor_masks(cflags)) == liquid_flag) | (torch.stack(all_moving_neighbor_masks(cflags)) == interface_flag)).any(dim=0)
    isolated = interface_mask & ~has_neighbor & ~solid_mask
    cflags = torch.where(isolated, torch.full_like(cflags, gas_flag), cflags)
    cfill = torch.where(isolated, torch.zeros_like(cfill), cfill)
    cmass = torch.where(isolated, torch.zeros_like(cmass), cmass)
    cf = torch.where(isolated.unsqueeze(0), torch.zeros_like(cf), cf)
    if replay_stages is not None:
        replay_stages["isolated_interface"] = tuple(value.clone() for value in (cf, cfill, cflags, cmass))
    cflags = torch.where(solid_mask, torch.full_like(cflags, solid_flag), cflags)
    if replay_stages is not None:
        replay_stages["solid_enforcement"] = tuple(value.clone() for value in (cf, cfill, cflags, cmass))
    if inventory_stages is not None:
        inventory_stages["after_topology_halo_isolation_boundary"] = inventory_measurement(
            cf, cfill, cflags, cmass, rho_liquid=rho_liquid,
        )
    mass_after_isolation = float(cmass.sum())

    evidence = None
    if capture_evidence:
        assert mass_before_redistribution is not None and mass_before_conversion is not None
        assert fill_before_conversion is not None and flags_before_conversion is not None and f_before_conversion is not None
        assert flags_after_conversion is not None and fill_after_conversion is not None
        assert mass_after_conversion_field is not None and f_after_conversion is not None
        conversion_delta = mass_after_conversion_field - mass_before_conversion
        cells = []
        for cell in torch.nonzero(to_liq | to_gas, as_tuple=False).tolist():
            z, y, x = (int(value) for value in cell)
            cells.append({"cell": (z, y, x), "flag_before": int(flags_before_conversion[z, y, x]), "flag_after": int(flags_after_conversion[z, y, x]), "mass_before": float(mass_before_conversion[z, y, x]), "mass_after": float(mass_after_conversion_field[z, y, x]), "mass_delta": float(conversion_delta[z, y, x]), "fill_before": float(fill_before_conversion[z, y, x]), "fill_after": float(fill_after_conversion[z, y, x]), "f_before": tuple(float(value) for value in f_before_conversion[:, z, y, x]), "f_after": tuple(float(value) for value in f_after_conversion[:, z, y, x]), "population_before": float(f_before_conversion[:, z, y, x].sum()), "population_after": float(f_after_conversion[:, z, y, x].sum()), "event_id": "conversion", "operator": "conversion"})
        links = []
        for raw_link in redistribution_link_evidence:
            donor = raw_link["donor"]
            receiver = raw_link["receiver"]
            dz, dy, dx = donor  # type: ignore[misc]
            rz, ry, rx = receiver  # type: ignore[misc]
            links.append({
                **raw_link,
                "donor_mass_before_redistribution": float(mass_before_redistribution[dz, dy, dx]),
                "receiver_flag_before": int(flags_before_conversion[rz, ry, rx]),
                "receiver_flag_after": int(flags_after_conversion[rz, ry, rx]),
                "receiver_fill_before": float(fill_before_conversion[rz, ry, rx]),
                "receiver_fill_after": float(fill_after_conversion[rz, ry, rx]),
                "receiver_mass_before": float(mass_before_redistribution[rz, ry, rx]),
                "receiver_mass_after": float(mass_after_conversion_field[rz, ry, rx]),
                "receiver_f_before": tuple(float(value) for value in f_before_conversion[:, rz, ry, rx]),
                "receiver_f_after": tuple(float(value) for value in f_after_conversion[:, rz, ry, rx]),
            })
        evidence = {
            "snapshot_kind": "actual_sparse_pre_redistribution_to_post_conversion",
            "conversion_cells": tuple(cells),
            "redistribution_links": tuple(links),
            "conversion_cell_delta_sum": float(sum(cell["mass_delta"] for cell in cells)),
            "conversion_tensor_delta_sum": float(conversion_delta.sum()),
            "redistribution_link_delta_sum": float(sum(float(link["mass_delta"]) for link in links)),
            "i_to_g_ownership_links": () if i_to_g_ownership is None else i_to_g_ownership.links,
            "i_to_g_ownership_debit": None if i_to_g_ownership is None else float(i_to_g_ownership.donor_debit),
            "i_to_g_ownership_credit": None if i_to_g_ownership is None else float(i_to_g_ownership.receiver_credit),
            "i_to_g_ownership_residual": None if i_to_g_ownership is None else float(i_to_g_ownership.residual),
            "i_to_g_population_owner_status": "WITHHELD_NO_POPULATION_TRANSFER",
        }
    _validate_candidate(cf, cfill, cflags, cmass, solid_mask, gas_flag, liquid_flag, interface_flag, solid_flag)
    replay_evidence = None
    if replay_stages is not None:
        assert replay_invocation is not None
        invocation_payload, invocation_sha256 = _freeze_replay_payload(replay_invocation)
        phase_payload, phase_sha256 = _freeze_replay_payload(replay_stages)
        candidate_payload, candidate_sha256 = _freeze_replay_payload((cf.clone(), cfill.clone(), cflags.clone(), cmass.clone()))
        replay_evidence = _register_trusted_replay_evidence(ReplayEvidence(
            invocation_payload, invocation_sha256, phase_payload, phase_sha256,
            candidate_payload, candidate_sha256,
            _replay_tensor_records(replay_invocation, "invocation")
            + _replay_tensor_records(replay_stages, "phases")
            + _replay_tensor_records((cf, cfill, cflags, cmass), "candidate"),
        ))
    return TopologyTransactionPlan(cf, cfill, cflags, cmass, mass_after_redistribution, mass_after_clamp, mass_after_conversion, mass_after_isolation, inventory_stages, evidence, gas_flag, liquid_flag, interface_flag, solid_flag, solid_mask.clone(), replay_evidence)


def commit_topology_transaction(
    plan: TopologyTransactionPlan, *, candidate: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    solid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate then return detached committed state; inputs are never mutated."""
    f, fill, flags, mass = candidate if candidate is not None else (plan.f, plan.fill, plan.flags, plan.mass)
    _validate_candidate(f, fill, flags, mass, plan.solid_mask if solid_mask is None else solid_mask, plan.gas_flag, plan.liquid_flag, plan.interface_flag, plan.solid_flag)
    return f.clone(), fill.clone(), flags.clone(), mass.clone()
