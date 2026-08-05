"""Cold, fail-closed D3Q19 I→G float32 ledger feasibility diagnostics.

This module consumes detached pre-topology tensors and never calls the solver or
mutates caller-owned state.  It deliberately separates three claims:

* exact sums of already-rounded float32 link records;
* exact integer-quanta diagnostic aggregation; and
* the float32 increments/state mutation the solver would actually store.

It is not a physical closure, a correction mechanism, or a production topology
path.  In particular, a ledger whose debit is redefined as a sum of rounded
credits is reported separately from the donor's actual independent-mass state
delta.
"""
from __future__ import annotations

from dataclasses import dataclass
import struct

import torch

from .core.d3q19_stencil import D3Q19_MOVING_Q, all_moving_neighbor_masks, moving_tensor_shifts, roll_from_pull_source, roll_to_neighbor


WITHHELD_NOT_REPRESENTABLE = "WITHHELD_NOT_REPRESENTABLE"
DIAGNOSTIC_ONLY_NOT_STATE_CONSERVATION = "DIAGNOSTIC_ONLY_NOT_STATE_CONSERVATION"


@dataclass(frozen=True)
class LedgerMethodReport:
    """One representation's evidence and deliberately narrow feasibility status."""

    status: str
    reason: str | None
    debit_quanta: int
    credit_quanta: int
    residual_quanta: int
    float32_debit: float
    float32_credit: float
    float32_residual: float


@dataclass(frozen=True)
class IToGExactLedgerReport:
    """Immutable observer result for one detached pre-topology I→G candidate."""

    status: str
    physical_closure_claim: bool
    mutates_solver_state: bool
    dtype: str
    donor_count: int
    receiver_count: int
    link_count: int
    legal_receivers: bool
    capacity_ok: bool
    donor_state_debit: float
    receiver_increment_credit: float
    actual_float32_operation_residual: float
    donor_vs_rounded_link_residual_quanta: int
    receiver_aggregation_nonzero_count: int
    receiver_aggregation_max_abs_quanta: int
    method_a: LedgerMethodReport
    method_b: LedgerMethodReport
    method_c: LedgerMethodReport
    population_owner_status: str


def _float32_quanta(value: float) -> int:
    """Encode an IEEE-754 binary32 value in exact units of 2**-149."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    sign = -1 if bits >> 31 else 1
    exponent = (bits >> 23) & 0xFF
    fraction = bits & 0x7FFFFF
    if exponent == 0:
        return sign * fraction
    if exponent == 0xFF:
        raise ValueError("float32 ledger values must be finite")
    return sign * ((1 << 23) | fraction) * (1 << (exponent - 1))


def _tensor_quanta(value: torch.Tensor) -> int:
    if value.dtype != torch.float32:
        raise ValueError("I→G exact ledger accepts float32 mass only")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("I→G exact ledger requires finite float32 values")
    return sum(_float32_quanta(float(item)) for item in value.detach().cpu().reshape(-1).tolist())


def _scalar_quanta(value: torch.Tensor) -> int:
    return _tensor_quanta(value.reshape(1))


def _as_float32_from_quanta(quanta: int) -> torch.Tensor | None:
    # A correction is representable only when it can first be represented as
    # binary32.  It is still not usable if a subsequent receiver addition rounds
    # it away; that state-mutation check is performed by the caller.
    value = quanta * 2.0 ** -149
    candidate = torch.tensor(value, dtype=torch.float32)
    if not bool(torch.isfinite(candidate)) or _scalar_quanta(candidate) != quanta:
        return None
    return candidate


def diagnose_i_to_g_exact_ledger(
    flags: torch.Tensor,
    mass: torch.Tensor,
    *,
    to_gas: torch.Tensor,
    to_liq: torch.Tensor,
    solid_mask: torch.Tensor,
    gas_flag: int,
    liquid_flag: int,
    interface_flag: int,
    rho_liquid: float,
) -> IToGExactLedgerReport:
    """Diagnose A/B/C without mutating solver state or applying a correction.

    A uses debit = exact sum of actual rounded per-link credits.  B encodes each
    float32 record as an integer number of binary32 subnormal quanta.  C assigns
    a donor's rounded-link residual to its first declared D3Q19 receiver *only
    in a detached simulation* and rejects it unless the proposed correction is
    representable, survives receiver aggregation, preserves capacity, and makes
    the observed float32 receiver increment equal its corrected exact record.
    """
    del gas_flag, liquid_flag  # Kept in the API to make phase ownership explicit.
    fields = (flags, mass, to_gas, to_liq, solid_mask)
    if any(field.shape != flags.shape for field in fields[1:]):
        raise ValueError("I→G exact ledger fields must share one spatial shape")
    if mass.dtype != torch.float32:
        raise ValueError("I→G exact ledger accepts float32 mass only")
    if flags.dtype != torch.int8 or to_gas.dtype != torch.bool or to_liq.dtype != torch.bool or solid_mask.dtype != torch.bool:
        raise ValueError("I→G exact ledger requires int8 flags and bool masks")
    if not bool(torch.isfinite(mass).all()):
        raise ValueError("I→G exact ledger requires finite mass")

    donor = to_gas & ~solid_mask
    legal_donors = not bool((donor & (flags != interface_flag)).any())
    receiver_mask = (flags == interface_flag) & ~to_gas & ~to_liq & ~solid_mask
    receiver_by_q = torch.stack(all_moving_neighbor_masks(receiver_mask))
    receiver_count = receiver_by_q.sum(dim=0)
    legal_receivers = legal_donors and not bool((donor & (receiver_count == 0)).any())
    zero = torch.zeros_like(mass)
    debit_field = torch.where(donor, mass, zero)
    credit_per_link = debit_field / receiver_count.clamp(min=1).to(mass.dtype)
    link_fields = tuple(
        roll_to_neighbor(credit_per_link, q) * receiver_mask for q in D3Q19_MOVING_Q
    )
    receiver_increment = torch.stack(link_fields).sum(dim=0)
    capacity = torch.where(receiver_mask, float(rho_liquid) - mass, zero)
    capacity_ok = legal_receivers and not bool((receiver_increment > capacity).any())

    donor_quanta = _tensor_quanta(debit_field)
    link_quanta = tuple(_tensor_quanta(field) for field in link_fields)
    rounded_credit_quanta = sum(link_quanta)
    stored_receiver_quanta = _tensor_quanta(receiver_increment)
    # Python integers are intentional: binary32 values near one have roughly
    # 2**149 subnormal quanta and cannot fit int64.  This is a cold diagnostic,
    # not a solver allocation or a state representation.
    receiver_exact_incoming: dict[tuple[int, int, int], int] = {}
    for field in link_fields:
        for raw in torch.nonzero(field != 0, as_tuple=False).tolist():
            z, y, x = (int(item) for item in raw)
            index = (z, y, x)
            receiver_exact_incoming[index] = receiver_exact_incoming.get(index, 0) + _float32_quanta(float(field[index]))
    local_residuals = {
        index: exact - _float32_quanta(float(receiver_increment[index]))
        for index, exact in receiver_exact_incoming.items()
    }
    receiver_aggregation_nonzero_count = sum(residual != 0 for residual in local_residuals.values())
    receiver_aggregation_max_abs_quanta = max((abs(residual) for residual in local_residuals.values()), default=0)

    donor_state_debit = -debit_field.sum()
    receiver_increment_credit = receiver_increment.sum()
    operation_residual = donor_state_debit + receiver_increment_credit

    method_a = LedgerMethodReport(
        status=DIAGNOSTIC_ONLY_NOT_STATE_CONSERVATION if legal_receivers else WITHHELD_NOT_REPRESENTABLE,
        reason=(
            "debit is redefined as exact sum of rounded link credits; it does not equal the donor state mutation"
            if legal_receivers else "no legal local receiver or donor phase is invalid"
        ),
        debit_quanta=-rounded_credit_quanta,
        credit_quanta=rounded_credit_quanta,
        residual_quanta=0,
        float32_debit=float(-sum(float(field.sum()) for field in link_fields)),
        float32_credit=float(sum(float(field.sum()) for field in link_fields)),
        float32_residual=0.0,
    )
    method_b = LedgerMethodReport(
        status=DIAGNOSTIC_ONLY_NOT_STATE_CONSERVATION if legal_receivers else WITHHELD_NOT_REPRESENTABLE,
        reason=(
            "integer binary32-quanta aggregation is exact only for diagnostic records, not float32 state mutation"
            if legal_receivers else "no legal local receiver or donor phase is invalid"
        ),
        debit_quanta=-rounded_credit_quanta,
        credit_quanta=rounded_credit_quanta,
        residual_quanta=0,
        float32_debit=method_a.float32_debit,
        float32_credit=method_a.float32_credit,
        float32_residual=0.0,
    )

    # C: select a declared local link deterministically (D3Q19 order), then
    # evaluate the actual detached float32 *mass-state* addition.  Verifying an
    # increment alone is insufficient: a representable correction can still be
    # rounded away when the solver performs ``mass + increment``.
    corrected = receiver_increment.clone()
    intended_quanta = dict(receiver_exact_incoming)
    c_reason: str | None = None
    if not legal_receivers:
        c_reason = "no legal local receiver or donor phase is invalid"
    else:
        assigned = torch.zeros_like(donor)
        for q in D3Q19_MOVING_Q:
            receiver_for_donor = roll_from_pull_source(receiver_mask, q)
            selected = donor & receiver_for_donor & ~assigned
            if not bool(selected.any()):
                continue
            # Per donor must be calculated individually; the vector loop is cold.
            for raw in torch.nonzero(selected, as_tuple=False).tolist():
                z, y, x = (int(item) for item in raw)
                donor_value = mass[z, y, x].reshape(1)
                each_credit = credit_per_link[z, y, x].reshape(1)
                links_for_donor = int(receiver_count[z, y, x])
                residual = _scalar_quanta(donor_value) - links_for_donor * _scalar_quanta(each_credit)
                correction = _as_float32_from_quanta(residual)
                dz, dy, dx = moving_tensor_shifts()[D3Q19_MOVING_Q.index(q)]
                receiver = ((z - dz) % mass.shape[0], (y - dy) % mass.shape[1], (x - dx) % mass.shape[2])
                if correction is None:
                    c_reason = "per-donor residual is not representable as float32"
                    break
                before_increment = corrected[receiver].clone()
                after_increment = before_increment + correction.to(device=before_increment.device)
                before_state = mass[receiver].clone()
                after_state = before_state + after_increment
                uncorrected_state = before_state + before_increment
                if _scalar_quanta(after_increment - before_increment) != residual:
                    c_reason = "per-donor residual is rounded away or altered by receiver float32 increment aggregation"
                    break
                if _scalar_quanta(after_state - uncorrected_state) != residual:
                    c_reason = "per-donor residual is rounded away or altered by actual receiver float32 state addition"
                    break
                corrected[receiver] = after_increment
                intended_quanta[receiver] += residual
                assigned[z, y, x] = True
            if c_reason is not None:
                break
        if c_reason is None and not bool(torch.equal(assigned, donor)):
            c_reason = "a donor has no deterministically declared local receiver"
        if c_reason is None and bool((corrected > capacity).any()):
            c_reason = "declared local receiver correction exceeds pre-clamp capacity"
        if c_reason is None and any(
            _float32_quanta(float(corrected[index])) != expected
            for index, expected in intended_quanta.items()
        ):
            c_reason = "corrected receiver increments do not equal exact assigned records"
        if c_reason is None:
            candidate_mass = mass + corrected
            candidate_mass = torch.where(donor, torch.zeros_like(candidate_mass), candidate_mass)
            if _tensor_quanta(candidate_mass) != _tensor_quanta(mass):
                c_reason = "actual detached float32 state mutation has non-exact independent-mass residual"
    # The observer intentionally does not receive the same-step legacy
    # redistribution increment or replay the later clamp/conversion/halo stages.
    # A local success therefore cannot prove the complete solver state mutation.
    # Fail closed instead of promoting an increment-only or partial replay.
    if c_reason is None:
        c_reason = "complete same-order topology mutation is unavailable to this cold observer"
    c_status = WITHHELD_NOT_REPRESENTABLE
    method_c = LedgerMethodReport(
        status=c_status,
        reason=c_reason,
        debit_quanta=-donor_quanta,
        credit_quanta=stored_receiver_quanta if c_reason is not None else donor_quanta,
        residual_quanta=(-donor_quanta + stored_receiver_quanta if c_reason is not None else 0),
        float32_debit=float(donor_state_debit),
        float32_credit=float(receiver_increment_credit if c_reason is not None else -donor_state_debit),
        float32_residual=float(operation_residual if c_reason is not None else 0.0),
    )
    overall = WITHHELD_NOT_REPRESENTABLE
    return IToGExactLedgerReport(
        status=overall,
        physical_closure_claim=False,
        mutates_solver_state=False,
        dtype=str(mass.dtype),
        donor_count=int(donor.sum()),
        receiver_count=int((receiver_increment != 0).sum()),
        link_count=sum(int((field != 0).sum()) for field in link_fields),
        legal_receivers=legal_receivers,
        capacity_ok=capacity_ok,
        donor_state_debit=float(donor_state_debit),
        receiver_increment_credit=float(receiver_increment_credit),
        actual_float32_operation_residual=float(operation_residual),
        donor_vs_rounded_link_residual_quanta=donor_quanta - rounded_credit_quanta,
        receiver_aggregation_nonzero_count=receiver_aggregation_nonzero_count,
        receiver_aggregation_max_abs_quanta=receiver_aggregation_max_abs_quanta,
        method_a=method_a,
        method_b=method_b,
        method_c=method_c,
        population_owner_status="WITHHELD_NO_POPULATION_TRANSFER",
    )
