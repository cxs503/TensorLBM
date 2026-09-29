"""Detached, fail-closed topology commit for D3Q19 free-surface states.

The module owns only the cold conversion path.  It deliberately receives flag
values and precomputed redistribution increments from the hot solver so it has
no dependency on :mod:`free_surface_lbm` and cannot alter collision, streaming,
ABB, or mass-exchange arithmetic.
"""

from __future__ import annotations

import hashlib
import io
import pickle
import weakref
from dataclasses import dataclass
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


@dataclass(frozen=True)
class StrictFailureReplayEvidence:
    """Trusted pre-invocation capture of an exact strict builder rejection.

    No phase or final-candidate tensors exist: the builder failed before it
    could publish either.  This is deliberately not a ``ReplayEvidence``.
    """

    invocation_payload: bytes
    invocation_sha256: str
    tensor_records: tuple[ReplayTensorRecord, ...]
    builder: str
    error_type: str
    error_message: str


_TRUSTED_REPLAY_EVIDENCE: dict[int, weakref.ReferenceType[ReplayEvidence]] = {}
_TRUSTED_STRICT_FAILURE_EVIDENCE: dict[int, weakref.ReferenceType[StrictFailureReplayEvidence]] = {}


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


def _register_trusted_strict_failure_evidence(
    evidence: StrictFailureReplayEvidence,
) -> StrictFailureReplayEvidence:
    evidence_id = id(evidence)

    def _discard(_: weakref.ReferenceType[StrictFailureReplayEvidence]) -> None:
        _TRUSTED_STRICT_FAILURE_EVIDENCE.pop(evidence_id, None)

    _TRUSTED_STRICT_FAILURE_EVIDENCE[evidence_id] = weakref.ref(evidence, _discard)
    return evidence


def is_trusted_strict_failure_evidence(evidence: object) -> bool:
    """True only for the exact failure evidence emitted in this process."""
    reference = _TRUSTED_STRICT_FAILURE_EVIDENCE.get(id(evidence))
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
        raise TopologyTransactionError(
            "WITHHELD: replay evidence payload cannot be safely loaded"
        ) from error


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
        if not bool(self.donor_debit == self.donor_debit_records.sum()) or not bool(
            self.receiver_credit == self.receiver_credit_records.sum()
        ):
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
        "receiver_increment",
        "receiver_mask",
        "donor_debit",
        "receiver_credit",
        "residual",
        "donor_debit_records",
        "receiver_credit_records",
    )
    if (
        not isinstance(value, dict)
        or set(value) != set(required)
        or any(not isinstance(value[name], torch.Tensor) for name in required)
    ):
        raise TopologyTransactionError("WITHHELD: captured I→G ownership has invalid schema")
    transaction = IToGOwnershipTransaction(
        value["receiver_increment"],
        value["receiver_mask"],
        (),
        value["donor_debit"],
        value["receiver_credit"],
        value["residual"],
        value["donor_debit_records"],
        value["receiver_credit_records"],
    )
    transaction.validate()
    return transaction


_I_TO_G_FAILURE_INVOCATION_KEYS = frozenset(
    {
        "flags",
        "mass",
        "to_gas",
        "to_liq",
        "solid_mask",
        "gas_flag",
        "liquid_flag",
        "interface_flag",
        "rho_liquid",
    }
)


def capture_strict_failure_invocation(
    invocation: Mapping[str, object],
    *,
    builder: str,
) -> tuple[bytes, str, tuple[ReplayTensorRecord, ...], str]:
    """Freeze a complete declared builder invocation before an opt-in failure."""
    frozen = dict(invocation)
    if builder != "i_to_g_ownership" or set(frozen) != _I_TO_G_FAILURE_INVOCATION_KEYS:
        raise TopologyTransactionError(
            "strict failure capture requires the complete I→G invocation"
        )
    for name, value in tuple(frozen.items()):
        if isinstance(value, torch.Tensor):
            frozen[name] = value.detach().clone()
    payload, digest = _freeze_replay_payload(frozen)
    return payload, digest, _replay_tensor_records(frozen, "invocation"), builder


def publish_strict_failure_evidence(
    capture: tuple[bytes, str, tuple[ReplayTensorRecord, ...], str],
    error: TopologyTransactionError,
) -> StrictFailureReplayEvidence:
    """Bind a pre-call snapshot to the exact rejection while preserving the raise."""
    payload, digest, records, builder = capture
    return _register_trusted_strict_failure_evidence(
        StrictFailureReplayEvidence(
            payload,
            digest,
            records,
            builder,
            type(error).__name__,
            str(error),
        )
    )


def restore_strict_failure_invocation(evidence: StrictFailureReplayEvidence) -> dict[str, object]:
    """Restore a complete detached invocation; validation is replayed by caller."""
    invocation = restore_replay_payload(evidence.invocation_payload, evidence.invocation_sha256)
    if not isinstance(invocation, dict) or evidence.builder != "i_to_g_ownership":
        raise TopologyTransactionError("WITHHELD: strict failure invocation schema is invalid")
    if set(invocation) != _I_TO_G_FAILURE_INVOCATION_KEYS:
        raise TopologyTransactionError("WITHHELD: strict failure I→G invocation schema is invalid")
    if _replay_tensor_records(invocation, "invocation") != evidence.tensor_records:
        raise TopologyTransactionError(
            "WITHHELD: strict failure tensor records do not match payload tensors"
        )
    return invocation


def build_i_to_g_ownership_transaction(
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
    increment = torch.stack(
        [roll_to_neighbor(credit_per_link, q) * receiver_mask for q in D3Q19_MOVING_Q]
    ).sum(dim=0)
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
            links.append(
                {
                    "donor": (z, y, x),
                    "receiver": ((z - dz) % shape[0], (y - dy) % shape[1], (x - dx) % shape[2]),
                    "q": q,
                    "shift": (dz, dy, dx),
                    "debit": float(-credit),
                    "credit": float(credit),
                    "event_id": "i_to_g_independent_mass_ownership",
                    "operator": "i_to_g_independent_mass_ownership",
                }
            )
    transaction = IToGOwnershipTransaction(
        increment,
        receiver_mask & (increment != 0.0),
        tuple(links),
        -debit_field.sum(),
        increment.sum(),
        torch.zeros((), dtype=mass.dtype, device=mass.device),
        -debit_field,
        increment,
    )
    transaction = IToGOwnershipTransaction(
        transaction.receiver_increment,
        transaction.receiver_mask,
        transaction.links,
        transaction.donor_debit,
        transaction.receiver_credit,
        transaction.donor_debit + transaction.receiver_credit,
        transaction.donor_debit_records,
        transaction.receiver_credit_records,
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
    def replay_stages(
        self,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None:
        """Compatibility copy rebuilt from verified private evidence each access."""
        if self._replay_evidence is None:
            return None
        restored = restore_replay_payload(
            self._replay_evidence.phase_payload,
            self._replay_evidence.phase_sha256,
        )
        if not isinstance(restored, dict):
            raise TopologyTransactionError("WITHHELD: replay phase evidence has invalid schema")
        return restored


# D3Q19 weights cached for directional birth reconstruction.
try:  # local import kept lazy to avoid a hard d3q19 import cycle
    from .d3q19 import W as _D3Q19_W
except Exception:  # pragma: no cover - d3q19 is always importable in practice
    _D3Q19_W = None


def _read_aprime() -> bool:
    """A-prime closure switch (opt-in, default OFF = bit-exact legacy path).

    ``TL_FS_APRIME=1`` enables the §6.2 closure redesign.  This module reads it
    directly (like ``TL_FS_BIRTH_MODE``) so the halo/isolation boundary can be
    brought under the same per-link legality predicate as the exchange end.
    """
    import os as _os

    return _os.environ.get("TL_FS_APRIME", "0").strip().lower() not in (
        "0",
        "false",
        "no",
        "",
    )


def _read_birth_mode() -> tuple[str, float]:
    """(mode, rho_env) for empty-shell birth, env-gated for A/B diagnosis.

    ``TL_FS_BIRTH_MODE`` controls how a freshly (re)born, mass-carrying zero
    INTERFACE envelope cell is seeded:

      ``legacy``  equilibrium(rho_liquid)  -- current behaviour
      ``zero``    f = 0
      ``gas``     equilibrium(rho_gas)
      ``env``     equilibrium(TL_FS_BIRTH_RHO)  -- decoupled from rho_gas
      ``nbr``     equilibrium(neighbour-averaged active density)
      ``liqface`` directional: liquid-facing links carry rho_liquid,
                  gas-facing links carry TL_FS_BIRTH_RHO (decoupled).
    """
    import os as _os

    mode = _os.environ.get("TL_FS_BIRTH_MODE", "legacy").strip().lower()
    rho_env = float(_os.environ.get("TL_FS_BIRTH_RHO", "0.0"))
    return mode, rho_env


def _birth_density_field(
    f, flags, rho_liquid, rho_gas, mode, rho_env, liquid_flag, interface_flag
):
    """Per-cell birth density (scalar modes).  Returns ``None`` for others."""
    if mode == "zero":
        return torch.zeros_like(f.sum(0))
    if mode == "gas":
        return torch.full_like(f.sum(0), float(rho_gas))
    if mode == "env":
        return torch.full_like(f.sum(0), float(rho_env))
    if mode == "nbr":
        # Neighbour-averaged *active* density: a shell backed by liquid inherits
        # a liquid-scale density, a gas-only shell inherits near-zero.
        active = (flags == liquid_flag) | (flags == interface_flag)
        nb = torch.stack(all_moving_neighbor_masks(active)).to(f.dtype)
        cnt = nb.sum(dim=0).clamp(min=1.0)
        rho = f.sum(0)
        rho_nb = (
            torch.stack([roll_from_pull_source(rho, q) for q in D3Q19_MOVING_Q]) * nb
        ).sum(dim=0) / cnt
        return torch.where(nb.sum(dim=0) > 0, rho_nb, torch.zeros_like(rho_nb))
    return None


def _init_new(
    f: torch.Tensor,
    flags: torch.Tensor,
    mask: torch.Tensor,
    rho_init,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    liquid_flag: int,
    interface_flag: int,
) -> torch.Tensor:
    active = (flags == liquid_flag) | (flags == interface_flag)
    neighbours = torch.stack(all_moving_neighbor_masks(active)).to(f.dtype)
    count = neighbours.sum(dim=0).clamp(min=1)
    # pull-source rolls are generated through the canonical stencil helper.
    from .core.d3q19_stencil import roll_from_pull_source

    ux_mean = (
        torch.stack([roll_from_pull_source(ux, q) for q in D3Q19_MOVING_Q]) * neighbours
    ).sum(dim=0) / count
    uy_mean = (
        torch.stack([roll_from_pull_source(uy, q) for q in D3Q19_MOVING_Q]) * neighbours
    ).sum(dim=0) / count
    uz_mean = (
        torch.stack([roll_from_pull_source(uz, q) for q in D3Q19_MOVING_Q]) * neighbours
    ).sum(dim=0) / count
    if isinstance(rho_init, torch.Tensor):
        rho_field = rho_init.to(ux.dtype)
    else:
        rho_field = torch.full_like(ux, float(rho_init))
    feq = equilibrium3d(rho_field, ux_mean, uy_mean, uz_mean)
    return torch.where(mask.unsqueeze(0), feq, f)


def _init_new_directional(
    f, flags, mask, rho_liquid, rho_env, ux, uy, uz, liquid_flag, interface_flag
):
    """Directional consistent birth for an empty envelope cell.

    A zero-mass INTERFACE shell is part liquid-facing and part gas-facing.  Its
    population must carry the *liquid* equilibrium on the links that actually
    face bulk liquid (so the Körner L/I exchange is balanced at rest) while the
    gas-facing links carry a small, ``rho_gas``-decoupled envelope density
    ``rho_env``.  A uniform equilibrium at either density violates that split:
    ``rho_liquid`` over-pressurises the gas side, ``rho_gas`` starves the liquid
    side, and both leave a systematic net L/I flux (the pseudo-wetting film).
    """
    from .core.d3q19_stencil import roll_from_pull_source
    from .d3q19 import C as _C

    device = f.device
    w = _D3Q19_W.to(device=device, dtype=f.dtype).view(19, 1, 1, 1)
    c = _C.to(device).to(f.dtype)
    active = (flags == liquid_flag) | (flags == interface_flag)
    neighbours = torch.stack(all_moving_neighbor_masks(active)).to(f.dtype)
    count = neighbours.sum(dim=0).clamp(min=1)
    ux_mean = (
        torch.stack([roll_from_pull_source(ux, q) for q in D3Q19_MOVING_Q]) * neighbours
    ).sum(dim=0) / count
    uy_mean = (
        torch.stack([roll_from_pull_source(uy, q) for q in D3Q19_MOVING_Q]) * neighbours
    ).sum(dim=0) / count
    uz_mean = (
        torch.stack([roll_from_pull_source(uz, q) for q in D3Q19_MOVING_Q]) * neighbours
    ).sum(dim=0) / count
    nbf = torch.stack(
        [roll_from_pull_source(flags.to(f.dtype), q) for q in D3Q19_MOVING_Q]
    )
    rho_dir = torch.where(
        nbf == float(liquid_flag),
        torch.full_like(nbf, float(rho_liquid)),
        torch.full_like(nbf, float(rho_env)),
    )
    u_sq = ux_mean * ux_mean + uy_mean * uy_mean + uz_mean * uz_mean
    cu = (
        c[:, 0].view(19, 1, 1, 1) * ux_mean
        + c[:, 1].view(19, 1, 1, 1) * uy_mean
        + c[:, 2].view(19, 1, 1, 1) * uz_mean
    )
    feq = w * rho_dir * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq.unsqueeze(0))
    return torch.where(mask.unsqueeze(0), feq, f)


def _init_new_birth(
    f, flags, mask, mode, rho_liquid, rho_gas, rho_env, ux, uy, uz, liquid_flag, interface_flag
):
    """Dispatcher for the empty-shell birth state (env-gated A/B hook)."""
    if mode == "legacy":
        return _init_new(
            f, flags, mask, rho_liquid, ux, uy, uz, liquid_flag, interface_flag
        )
    if mode == "liqface":
        return _init_new_directional(
            f, flags, mask, rho_liquid, rho_env, ux, uy, uz, liquid_flag, interface_flag
        )
    rho_field = _birth_density_field(
        f, flags, rho_liquid, rho_gas, mode, rho_env, liquid_flag, interface_flag
    )
    if rho_field is None:
        return _init_new(
            f, flags, mask, rho_liquid, ux, uy, uz, liquid_flag, interface_flag
        )
    return _init_new(
        f, flags, mask, rho_field, ux, uy, uz, liquid_flag, interface_flag
    )


def conservative_clamp_conserve(
    field: torch.Tensor, upper: float, weight: torch.Tensor | None = None
) -> torch.Tensor:
    """Clip ``field`` into ``[0, upper]`` while preserving ``field.sum()``.

    F2 (Körner conservation fix).  A plain ``field.clamp(0, upper)`` is a silent
    mass source/sink: the paired liquid/interface debit can drive a LIQUID cell
    below zero (clip-up creates mass) and redistribution can push a receiver
    above ``rho_liquid`` (clip-down destroys mass).  Instead of dropping the
    clipped amount, this clips each cell to the legal band and *returns the
    clipped mass to cells that can still absorb it*:

    * clip-up (net created): the surplus is drawn back out of the cells that
      still carry tracked mass, proportionally to that mass, so no cell is
      driven below zero — the excess is explicitly pushed back to its
      liquid/interface neighbours rather than silently retained;
    * clip-down (net destroyed): the deficit is pushed forward to the cells
      that still have headroom below ``rho_liquid``.

    ``weight`` optionally restricts where the correction is applied (e.g. to the
    non-solid cells).  The returned tensor is bounded to ``[0, upper]`` and its
    sum is exactly the input sum up to float rounding.
    """
    hi = float(upper)
    clipped = field.clamp(0.0, hi)
    net = float((clipped - field).sum())
    if net == 0.0:
        return clipped
    if weight is None:
        weight = torch.ones_like(field)
    else:
        weight = weight.to(field.dtype)
    if net > 0.0:
        # Clipped negatives lifted mass; remove it from cells that hold mass.
        mass_weight = clipped * weight
        weight_sum = float(mass_weight.sum())
        if weight_sum <= 0.0:
            return clipped
        return (clipped - mass_weight * (net / weight_sum)).clamp(0.0, hi)
    # Clipped overflow destroyed mass; give it back to cells with headroom.
    room = (hi - clipped) * weight
    room_sum = float(room.sum())
    if room_sum <= 0.0:
        return clipped
    return (clipped + room * ((-net) / room_sum)).clamp(0.0, hi)


def pair_conversion_conservation(
    field: torch.Tensor,
    target_total: float,
    credit: torch.Tensor,
    movable: torch.Tensor,
    upper: float,
) -> torch.Tensor:
    """Pair the redistribution credit with the conversion debit on one field.

    F2b (conversion/conservation pairing).  ``excess`` is credited to the
    distribution receivers here, but the matching donor debit is (legacy) only
    applied later by the ``to_liq`` → ``rho_liquid`` / ``to_gas`` → ``0``
    relabel.  The conservation-preserving clamp runs *between* the two, so it
    absorbs the donor overflow into the field first and the conversion relabel
    then becomes a near no-op: the receiver credit survives as a net mass
    source every time an I→L conversion fires (measured: ``A ≈ +0.5`` with the
    full ``A`` appearing as per-step drift).

    Recompute the pairing on the *post-clamp* conversion field: the residual
    ``target_total - field.sum()`` is charged to (or drawn from) the credited
    receivers proportionally to the credit they received, so the redistribution
    and the conversion cancel exactly and conversion fabricates no mass.

    ``credit`` is the per-cell redistribution increment (non-negative at
    receivers); ``movable`` excludes the solid interior; the result is bounded
    to ``[0, upper]``.
    """
    delta = float(target_total) - float(field.sum())
    if delta == 0.0:
        return field
    movable = movable.to(field.dtype)
    # Phase 1: the credited receivers give back (or are given) their share,
    # bounded so a receiver never loses more than the credit it received.
    credit_weight = credit.clamp(min=0.0).to(field.dtype) * movable
    credit_sum = float(credit_weight.sum())
    out = field
    if credit_sum > 0.0:
        magnitude = min(abs(delta), credit_sum)
        sign = 1.0 if delta > 0.0 else -1.0
        out = (field + credit_weight * (sign * magnitude / credit_sum)).clamp(0.0, float(upper))
    # Phase 2: settle any residual over the whole movable band (held mass when
    # removing, available headroom when adding) so the total is exact.
    residual = float(target_total) - float(out.sum())
    if abs(residual) > 1e-12:
        if residual < 0.0:
            band = out * movable
            band_sum = float(band.sum())
            if band_sum > 0.0:
                take = min(-residual, band_sum)
                out = (out - band * (take / band_sum)).clamp(0.0, float(upper))
        else:
            band = (float(upper) - out) * movable
            band_sum = float(band.sum())
            if band_sum > 0.0:
                give = min(residual, band_sum)
                out = (out + band * (give / band_sum)).clamp(0.0, float(upper))
    return out


def _validate_candidate(
    f: torch.Tensor,
    fill: torch.Tensor,
    flags: torch.Tensor,
    mass: torch.Tensor,
    solid_mask: torch.Tensor,
    gas_flag: int,
    liquid_flag: int,
    interface_flag: int,
    solid_flag: int,
) -> None:
    fields = {"f": f, "fill": fill, "mass": mass}
    for name, field in fields.items():
        if not bool(torch.isfinite(field).all()):
            raise TopologyTransactionError(f"topology candidate has non-finite {name}")
    legal = (
        (flags == gas_flag)
        | (flags == liquid_flag)
        | (flags == interface_flag)
        | (flags == solid_flag)
    )
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
    f: torch.Tensor,
    fill: torch.Tensor,
    flags: torch.Tensor,
    mass: torch.Tensor,
    *,
    to_iface: torch.Tensor,
    to_liq: torch.Tensor,
    to_gas: torch.Tensor,
    recv_new: torch.Tensor,
    redistribution_increment: torch.Tensor,
    rho_liquid: float,
    rho_gas: float,
    solid_mask: torch.Tensor,
    gas_flag: int,
    liquid_flag: int,
    interface_flag: int,
    solid_flag: int,
    ux: torch.Tensor | None = None,
    uy: torch.Tensor | None = None,
    uz: torch.Tensor | None = None,
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
        raise TopologyTransactionError(
            "topology transaction requires pre-conversion velocity fields"
        )
    cf, cfill, cflags, cmass = (value.clone() for value in (f, fill, flags, mass))
    if (i_to_g_increment is None) != (i_to_g_ownership is None):
        raise TopologyTransactionError(
            "I→G increment and ownership evidence must be supplied together"
        )
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
        if bool(
            (i_to_g_ownership.receiver_mask & ((flags != interface_flag) | to_gas | to_liq)).any()
        ):
            raise TopologyTransactionError(
                "I→G receiver must be a surviving pre-topology INTERFACE cell"
            )
    inventory_stages = {} if capture_inventory else None
    replay_stages = {} if capture_replay_stages else None
    replay_invocation = None
    if capture_replay_stages:
        replay_invocation = {
            "f": f.clone(),
            "fill": fill.clone(),
            "flags": flags.clone(),
            "mass": mass.clone(),
            "to_iface": to_iface.clone(),
            "to_liq": to_liq.clone(),
            "to_gas": to_gas.clone(),
            "recv_new": recv_new.clone(),
            "redistribution_increment": redistribution_increment.clone(),
            "rho_liquid": rho_liquid,
            "rho_gas": rho_gas,
            "solid_mask": solid_mask.clone(),
            "gas_flag": gas_flag,
            "liquid_flag": liquid_flag,
            "interface_flag": interface_flag,
            "solid_flag": solid_flag,
            "ux": ux.clone(),
            "uy": uy.clone(),
            "uz": uz.clone(),
            "i_to_g_increment": None if i_to_g_increment is None else i_to_g_increment.clone(),
            "i_to_g_ownership": _serialize_i_to_g_ownership(i_to_g_ownership),
        }
    gas_mask = cflags == gas_flag
    _birth_mode, _birth_rho = _read_birth_mode()
    cf = _init_new_birth(
        cf, cflags, to_iface, _birth_mode, rho_liquid, rho_gas, _birth_rho,
        ux, uy, uz, liquid_flag, interface_flag,
    )
    cflags = torch.where(to_iface, torch.full_like(cflags, interface_flag), cflags)
    if replay_stages is not None:
        replay_stages["to_iface_initialization"] = tuple(
            value.clone() for value in (cf, cfill, cflags, cmass)
        )

    mass_before_redistribution = cmass.clone() if capture_evidence else None
    combined_increment = redistribution_increment
    if i_to_g_increment is not None:
        assert i_to_g_ownership is not None
        combined_increment = redistribution_increment + i_to_g_increment
        combined_receivers = i_to_g_ownership.receiver_mask
        combined_candidate = cmass + combined_increment
        if bool(
            (
                ((combined_candidate < 0.0) | (combined_candidate > float(rho_liquid)))
                & combined_receivers
            ).any()
        ):
            raise TopologyTransactionError(
                "WITHHELD: entire free_surface_step topology candidate combined receiver capacity is outside [0, rho_liquid] before clamp"
            )
    cmass = cmass + combined_increment
    if replay_stages is not None:
        replay_stages["legacy_redistribution_and_i_to_g_increment"] = tuple(
            value.clone() for value in (cf, cfill, cflags, cmass)
        )
    if inventory_stages is not None:
        inventory_stages["after_topology_redistribution"] = inventory_measurement(
            cf,
            cfill,
            cflags,
            cmass,
            rho_liquid=rho_liquid,
        )
    mass_after_redistribution = float(cmass.sum())
    import os as _os_dbg
    if _os_dbg.environ.get("TL_FS_DBG", "0").strip().lower() not in ("0", "false", "no", ""):
        _g = (cflags == gas_flag)
        _xl = cflags == liquid_flag
        _xi = cflags == interface_flag
        _neg = cmass < 0
        print(f"        [TXN gasmass] pre_clamp gas_sum={float(cmass[_g].sum()):.6f} "
              f"min={float(cmass.min()):.6f} neg_n={int(_neg.sum())} "
              f"neg_liq={int((_neg & _xl).sum())} neg_iface={int((_neg & _xi).sum())} "
              f"neg_liq_mass={float(cmass[_neg & _xl].sum()):.6f} "
              f"near_empty_liq={int((_xl & (cmass <= 0.01)).sum())}")
    # F2 conservation fix: replace the silent clamp with a conservation-
    # preserving one.  Restrict the correction to the movable (non-solid) cells
    # so the solid interior is never credited or debited.
    cmass = conservative_clamp_conserve(cmass, rho_liquid, weight=(~solid_mask).to(cmass.dtype))
    if replay_stages is not None:
        replay_stages["clamp"] = tuple(value.clone() for value in (cf, cfill, cflags, cmass))
    if inventory_stages is not None:
        inventory_stages["after_topology_clamp"] = inventory_measurement(
            cf,
            cfill,
            cflags,
            cmass,
            rho_liquid=rho_liquid,
        )
    mass_after_clamp = float(cmass.sum())
    mass_before_conversion = cmass.clone() if capture_evidence else None
    fill_before_conversion = cfill.clone() if capture_evidence else None
    flags_before_conversion = cflags.clone() if capture_evidence else None
    f_before_conversion = cf.clone() if capture_evidence else None

    # H2: to_liq conversion reinitializes f at the rho_liquid equilibrium
    # (neighbor-averaged velocity) instead of inheriting interface
    # populations that were inflated by ABB gas-pressure reconstruction.
    import os as _os2

    if _os2.environ.get("TL_FS_DBG", "0").strip().lower() not in ("0", "false", "no", ""):
        _dbg_liq_before = float(cmass[to_liq].sum())
        _dbg_gas_before = float(cmass[to_gas].sum())
        _dbg_liq_n = int(to_liq.sum())
        _dbg_gas_n = int(to_gas.sum())
    cf = _init_new(cf, cflags, to_liq, rho_liquid, ux, uy, uz, liquid_flag, interface_flag)
    cflags = torch.where(to_liq, torch.full_like(cflags, liquid_flag), cflags)
    cfill = torch.where(to_liq, torch.ones_like(cfill), cfill)
    cmass = torch.where(to_liq, torch.full_like(cmass, rho_liquid), cmass)
    cflags = torch.where(to_gas, torch.full_like(cflags, gas_flag), cflags)
    cfill = torch.where(to_gas, torch.zeros_like(cfill), cfill)
    cmass = torch.where(to_gas, torch.zeros_like(cmass), cmass)
    cf = torch.where(to_gas.unsqueeze(0), torch.zeros_like(cf), cf)
    # F2b: pair the redistribution credit with the conversion donor debit on the
    # post-clamp conversion field, so the to_liq relabel cannot fabricate mass.
    cmass = pair_conversion_conservation(
        cmass,
        float(mass.sum()),
        combined_increment,
        (~solid_mask),
        float(rho_liquid),
    )
    if _os2.environ.get("TL_FS_DBG", "0").strip().lower() not in ("0", "false", "no", ""):
        _dbg_liq_after = float(cmass[to_liq].sum()) if _dbg_liq_n else 0.0
        _dbg_gas_after = float(cmass[to_gas].sum()) if _dbg_gas_n else 0.0
        print(f"        [TXN conv] to_liq n={_dbg_liq_n} mass {_dbg_liq_before:.4f}->{_dbg_liq_after:.4f} "
              f"| to_gas n={_dbg_gas_n} mass {_dbg_gas_before:.4f}->{_dbg_gas_after:.4f}")
    if i_to_g_ownership is not None:
        # No f transfer: independent mass/fill and population density are
        # separate representations at INTERFACE, so copying f would double count.
        cfill = torch.where(i_to_g_ownership.receiver_mask, cmass / float(rho_liquid), cfill)
    if replay_stages is not None:
        replay_stages["to_liq_to_gas_conversion"] = tuple(
            value.clone() for value in (cf, cfill, cflags, cmass)
        )
    if inventory_stages is not None:
        inventory_stages["after_topology_conversion"] = inventory_measurement(
            cf,
            cfill,
            cflags,
            cmass,
            rho_liquid=rho_liquid,
        )
    mass_after_conversion = float(cmass.sum())
    # Evidence attributes conversion itself, not the subsequent envelope halo.
    # Preserve the legacy post-conversion/pre-halo observation boundary.
    flags_after_conversion = cflags.clone() if capture_evidence else None
    fill_after_conversion = cfill.clone() if capture_evidence else None
    mass_after_conversion_field = cmass.clone() if capture_evidence else None
    f_after_conversion = cf.clone() if capture_evidence else None

    shifted_flags = torch.stack(all_moving_neighbor_masks(cflags))
    cmass_before_halo = cmass.clone()
    if _os2.environ.get("TL_FS_DBG", "0").strip().lower() not in ("0", "false", "no", ""):
        _gas_stage = (cflags == gas_flag)
        print(f"        [TXN gasmass] pre_halo gas_sum={float(cmass[_gas_stage].sum()):.6f} "
              f"gas_n={int(_gas_stage.sum())}")
    # H1: halo promotion requires a directly adjacent LIQUID cell.  A gas
    # cell neighbouring only interface cells must not self-propagate the
    # interface layer (quiescent column 743->32291 interface explosion).
    is_neighbor = (shifted_flags == liquid_flag).any(dim=0)
    to_i = ((gas_mask | to_gas) & is_neighbor & ~solid_mask) | recv_new
    # ABLATION (diagnostic only): TL_FS_ABL_HALO suppresses the gas->interface
    # envelope halo promotion (only residual recv_new receivers survive).
    import os as _os

    if _os.environ.get("TL_FS_ABL_HALO", "0").strip().lower() not in ("0", "false", "no", ""):
        to_i = recv_new
    cf = _init_new_birth(
        cf, cflags, to_i, _birth_mode, rho_liquid, rho_gas, _birth_rho,
        ux, uy, uz, liquid_flag, interface_flag,
    )
    cflags = torch.where(to_i, torch.full_like(cflags, interface_flag), cflags)
    # A-prime: bring the halo/isolation boundary under the same per-link
    # legality predicate as the exchange end.  A gas cell that is promoted to
    # INTERFACE because it directly neighbours LIQUID is normally reset to
    # mass/fill 0 (the "unsupported birth" convention).  But a cell that the
    # redistribution / conversion pairing has already credited with tracked
    # mass is, by the A-prime receiver predicate, a *legal receiver*: zeroing
    # it destroys booked mass (the isolation-stage residual).  Keep that mass
    # and treat the cell as a receiver; only truly empty shells are zeroed.
    _aprime = _read_aprime()
    if _aprime:
        _carry = to_i & (cmass > 0.0)
        cfill = torch.where(
            _carry, (cmass / float(rho_liquid)).clamp(0.0, 1.0), cfill
        )
        _halo_zero = to_i & ~recv_new & ~_carry
    else:
        _halo_zero = to_i & ~recv_new
    cfill = torch.where(_halo_zero, torch.zeros_like(cfill), cfill)
    cmass = torch.where(_halo_zero, torch.zeros_like(cmass), cmass)
    if _os.environ.get("TL_FS_DBG", "0").strip().lower() not in ("0", "false", "no", ""):
        if bool(_halo_zero.any()):
            print(f"        [TXN halo] zeroed_n={int(_halo_zero.sum())} "
                  f"mass_before_zero={float(cmass_before_halo[_halo_zero].sum()):.6f} "
                  f"recv_new_kept={int(recv_new.sum())} "
                  f"mass_recv_new={float(cmass[recv_new].sum()):.6f} "
                  f"carry_kept={int((to_i & ~_halo_zero & ~recv_new).sum())}")
    if replay_stages is not None:
        replay_stages["halo_boundary"] = tuple(
            value.clone() for value in (cf, cfill, cflags, cmass)
        )
    interface_mask = cflags == interface_flag
    has_neighbor = (
        (torch.stack(all_moving_neighbor_masks(cflags)) == liquid_flag)
        | (torch.stack(all_moving_neighbor_masks(cflags)) == interface_flag)
    ).any(dim=0)
    isolated = interface_mask & ~has_neighbor & ~solid_mask
    if _os.environ.get("TL_FS_DBG", "0").strip().lower() not in ("0", "false", "no", ""):
        _iso_n = int(isolated.sum())
        if _iso_n:
            _neg_liq = (cflags == liquid_flag) & (mass < 0.0)
            _neg_liq_nb = torch.stack(all_moving_neighbor_masks(_neg_liq)).any(dim=0)
            print(f"        [TXN iso] n={_iso_n} mass={float(cmass[isolated].sum()):.6f} "
                  f"recv_new_iso={int((isolated & recv_new).sum())} "
                  f"to_gas_iso={int((isolated & to_gas).sum())} "
                  f"maxmass={float(cmass[isolated].max()):.6f} "
                  f"negliq_nb_iso={int((isolated & _neg_liq_nb).sum())} "
                  f"negliq_tot={int(_neg_liq.sum())} negliq_mass={float(mass[_neg_liq].sum()):.6f} "
                  f"iso_mass_after={float(cmass[isolated].sum()):.6f} "
                  f"iso_flag_pre={int((cflags[isolated] == interface_flag).sum())}")
    cflags = torch.where(isolated, torch.full_like(cflags, gas_flag), cflags)
    cfill = torch.where(isolated, torch.zeros_like(cfill), cfill)
    cmass = torch.where(isolated, torch.zeros_like(cmass), cmass)
    cf = torch.where(isolated.unsqueeze(0), torch.zeros_like(cf), cf)
    if replay_stages is not None:
        replay_stages["isolated_interface"] = tuple(
            value.clone() for value in (cf, cfill, cflags, cmass)
        )
    cflags = torch.where(solid_mask, torch.full_like(cflags, solid_flag), cflags)
    if replay_stages is not None:
        replay_stages["solid_enforcement"] = tuple(
            value.clone() for value in (cf, cfill, cflags, cmass)
        )
    if inventory_stages is not None:
        inventory_stages["after_topology_halo_isolation_boundary"] = inventory_measurement(
            cf,
            cfill,
            cflags,
            cmass,
            rho_liquid=rho_liquid,
        )
    mass_after_isolation = float(cmass.sum())
    if _os.environ.get("TL_FS_DBG", "0").strip().lower() not in ("0", "false", "no", ""):
        _toi_new = int((to_i & ~recv_new).sum())
        print(f"        [TXN stage] redist={mass_after_redistribution:.6f} "
              f"clamp={mass_after_clamp:.6f} conv={mass_after_conversion:.6f} "
              f"iso={mass_after_isolation:.6f} to_i_new={_toi_new} "
              f"cmass_on_toi_new={float(cmass[to_i & ~recv_new].sum()):.6f}")

    evidence = None
    if capture_evidence:
        assert mass_before_redistribution is not None and mass_before_conversion is not None
        assert (
            fill_before_conversion is not None
            and flags_before_conversion is not None
            and f_before_conversion is not None
        )
        assert flags_after_conversion is not None and fill_after_conversion is not None
        assert mass_after_conversion_field is not None and f_after_conversion is not None
        conversion_delta = mass_after_conversion_field - mass_before_conversion
        cells = []
        for cell in torch.nonzero(to_liq | to_gas, as_tuple=False).tolist():
            z, y, x = (int(value) for value in cell)
            cells.append(
                {
                    "cell": (z, y, x),
                    "flag_before": int(flags_before_conversion[z, y, x]),
                    "flag_after": int(flags_after_conversion[z, y, x]),
                    "mass_before": float(mass_before_conversion[z, y, x]),
                    "mass_after": float(mass_after_conversion_field[z, y, x]),
                    "mass_delta": float(conversion_delta[z, y, x]),
                    "fill_before": float(fill_before_conversion[z, y, x]),
                    "fill_after": float(fill_after_conversion[z, y, x]),
                    "f_before": tuple(float(value) for value in f_before_conversion[:, z, y, x]),
                    "f_after": tuple(float(value) for value in f_after_conversion[:, z, y, x]),
                    "population_before": float(f_before_conversion[:, z, y, x].sum()),
                    "population_after": float(f_after_conversion[:, z, y, x].sum()),
                    "event_id": "conversion",
                    "operator": "conversion",
                }
            )
        links = []
        for raw_link in redistribution_link_evidence:
            donor = raw_link["donor"]
            receiver = raw_link["receiver"]
            dz, dy, dx = donor  # type: ignore[misc]
            rz, ry, rx = receiver  # type: ignore[misc]
            links.append(
                {
                    **raw_link,
                    "donor_mass_before_redistribution": float(
                        mass_before_redistribution[dz, dy, dx]
                    ),
                    "receiver_flag_before": int(flags_before_conversion[rz, ry, rx]),
                    "receiver_flag_after": int(flags_after_conversion[rz, ry, rx]),
                    "receiver_fill_before": float(fill_before_conversion[rz, ry, rx]),
                    "receiver_fill_after": float(fill_after_conversion[rz, ry, rx]),
                    "receiver_mass_before": float(mass_before_redistribution[rz, ry, rx]),
                    "receiver_mass_after": float(mass_after_conversion_field[rz, ry, rx]),
                    "receiver_f_before": tuple(
                        float(value) for value in f_before_conversion[:, rz, ry, rx]
                    ),
                    "receiver_f_after": tuple(
                        float(value) for value in f_after_conversion[:, rz, ry, rx]
                    ),
                }
            )
        evidence = {
            "snapshot_kind": "actual_sparse_pre_redistribution_to_post_conversion",
            "conversion_cells": tuple(cells),
            "redistribution_links": tuple(links),
            "conversion_cell_delta_sum": float(sum(cell["mass_delta"] for cell in cells)),
            "conversion_tensor_delta_sum": float(conversion_delta.sum()),
            "redistribution_link_delta_sum": float(
                sum(float(link["mass_delta"]) for link in links)
            ),
            "i_to_g_ownership_links": () if i_to_g_ownership is None else i_to_g_ownership.links,
            "i_to_g_ownership_debit": None
            if i_to_g_ownership is None
            else float(i_to_g_ownership.donor_debit),
            "i_to_g_ownership_credit": None
            if i_to_g_ownership is None
            else float(i_to_g_ownership.receiver_credit),
            "i_to_g_ownership_residual": None
            if i_to_g_ownership is None
            else float(i_to_g_ownership.residual),
            "i_to_g_population_owner_status": "WITHHELD_NO_POPULATION_TRANSFER",
        }
    _validate_candidate(
        cf, cfill, cflags, cmass, solid_mask, gas_flag, liquid_flag, interface_flag, solid_flag
    )
    replay_evidence = None
    if replay_stages is not None:
        assert replay_invocation is not None
        invocation_payload, invocation_sha256 = _freeze_replay_payload(replay_invocation)
        phase_payload, phase_sha256 = _freeze_replay_payload(replay_stages)
        candidate_payload, candidate_sha256 = _freeze_replay_payload(
            (cf.clone(), cfill.clone(), cflags.clone(), cmass.clone())
        )
        replay_evidence = _register_trusted_replay_evidence(
            ReplayEvidence(
                invocation_payload,
                invocation_sha256,
                phase_payload,
                phase_sha256,
                candidate_payload,
                candidate_sha256,
                _replay_tensor_records(replay_invocation, "invocation")
                + _replay_tensor_records(replay_stages, "phases")
                + _replay_tensor_records((cf, cfill, cflags, cmass), "candidate"),
            )
        )
    return TopologyTransactionPlan(
        cf,
        cfill,
        cflags,
        cmass,
        mass_after_redistribution,
        mass_after_clamp,
        mass_after_conversion,
        mass_after_isolation,
        inventory_stages,
        evidence,
        gas_flag,
        liquid_flag,
        interface_flag,
        solid_flag,
        solid_mask.clone(),
        replay_evidence,
    )


def commit_topology_transaction(
    plan: TopologyTransactionPlan,
    *,
    candidate: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    solid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate then return detached committed state; inputs are never mutated."""
    f, fill, flags, mass = (
        candidate if candidate is not None else (plan.f, plan.fill, plan.flags, plan.mass)
    )
    _validate_candidate(
        f,
        fill,
        flags,
        mass,
        plan.solid_mask if solid_mask is None else solid_mask,
        plan.gas_flag,
        plan.liquid_flag,
        plan.interface_flag,
        plan.solid_flag,
    )
    return f.clone(), fill.clone(), flags.clone(), mass.clone()
