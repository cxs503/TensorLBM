"""Free-surface LBM (D3Q19) — full Körner model with mass tracking.

Implements the complete Körner et al. (2005) free-surface LBM:
  - Cell types: GAS / INTERFACE / LIQUID / SOLID
  - Mass tracking: independent mass variable (mass ≠ rho)
  - Mass redistribution: excess mass distributed to interface neighbors
  - Interface gas pressure: anti-bounce-back at interface cells
  - Neighbor flags: prevents isolated cells

References: Körner et al. (2005), waLBerla free_surface/, Maarten-vd-Sande/lbm
"""

from __future__ import annotations

import os
from copy import deepcopy

import torch

from .boundaries3d import bounce_back_cells_3d, free_slip_cells_3d
from .core.d3q19_stencil import (
    D3Q19_MOVING_Q,
    all_moving_neighbor_masks,
    assert_no_direct_phase_links,
    moving_tensor_shifts,
    roll_from_pull_source,
    roll_to_neighbor,
)
from .d3q19 import C, W, equilibrium3d, macroscopic3d
from .free_surface_inventory_reconciliation import (
    CANONICAL_STAGE_ORDER,
    inventory_measurement,
    inventory_stage_deltas,
)
from .free_surface_topology_transaction import (
    TopologyTransactionError,
    build_i_to_g_ownership_transaction,
    build_topology_transaction,
    capture_strict_failure_invocation,
    commit_topology_transaction,
    publish_strict_failure_evidence,
)
from .solver3d import _get_d3q19_mrt_matrices
from .turbulence import (
    _neq_stress_norm_3d,
    _nu_t_to_tau_eff,
    _smagorinsky_tau,
    _vreman_nu_t_3d,
    _wale_nu_t_3d,
)

GAS = 0
LIQUID = 1
INTERFACE = 2
SOLID = 3


def _env_bool(name: str, default: bool = False) -> bool:
    """Read an opt-out boolean env gate for A/B diagnosis of the FS fixes."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "")

# D3Q19 velocity vectors and weights
_C = C  # (19, 3)
_W = W
KAPPA = 0.41
B_CONST = 5.0
# Opposite direction indices for D3Q19
_OPP = torch.tensor([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17])
_C19_SHIFTS = [(int(C[q, 0]), int(C[q, 1]), int(C[q, 2])) for q in range(19)]


def _stream19_roll(f):
    """D3Q19 streaming via torch.roll (pull scheme, shifts from C for ordering consistency)."""
    out = torch.empty_like(f)
    for q in range(19):
        sx, sy, sz = _C19_SHIFTS[q]
        out[q] = torch.roll(f[q], shifts=(sz, sy, sx), dims=(0, 1, 2))
    return out


def _assert_no_direct_liquid_gas_links(flags):
    """Reject states where a D3Q19 liquid link reaches GAS without INTERFACE.

    Körner free-surface boundaries are represented by INTERFACE cells.  A
    direct LIQUID/GAS streaming link has neither ABB reconstruction nor an
    interface mass ledger, so it is an invalid solver state rather than a
    wall-like boundary condition.
    """
    try:
        assert_no_direct_phase_links(flags, LIQUID, GAS, "direct LIQUID-GAS D3Q19")
    except ValueError as error:
        if "direct phase link" not in str(error):
            raise
        raise ValueError(f"{error}; insert INTERFACE cells between LIQUID and GAS") from None


def _append_runtime_ledger(
    ledger,
    *,
    mass_start,
    mass_after_exchange,
    mass_after_redistribution,
    mass_after_clamp,
    mass_after_conversion,
    mass_after_isolation,
    mass_end,
    abb_population_delta,
    exchange_liquid_credit,
    exchange_interface_credit,
    exchange_bulk_debit,
    paired_liquid_interface_debit,
    conversion_evidence=None,
):
    """Record a physical tracked-mass budget without correcting the solver.

    Reference convention: ``mass.sum()`` is lattice liquid mass.  ABB is a
    population reconstruction, rather than a tracked-mass source.  L/I
    exchange is internal only if both endpoints are recorded; a one-sided
    interface credit is explicitly left unexplained, never offset artificially.
    """
    steps = ledger.setdefault("steps", [])
    redistribution = mass_after_redistribution - mass_after_exchange
    clamp = mass_after_clamp - mass_after_redistribution
    conversion = mass_after_conversion - mass_after_clamp
    isolation = mass_after_isolation - mass_after_conversion
    boundary = mass_end - mass_after_isolation
    mass_drift = mass_end - mass_start
    paired_net = exchange_liquid_credit + exchange_bulk_debit
    # A paired L/I transfer is internal only after its explicit liquid-source
    # debit has been applied; a legacy one-sided credit remains unexplained.
    paired_explained = paired_net if paired_liquid_interface_debit else 0.0
    explained = paired_explained + redistribution + clamp + conversion + isolation + boundary
    unexplained = mass_drift - explained
    step_number = len(steps) + 1
    # Keep raw operator activity distinct from a root-cause claim.  In
    # particular ABB is a population reconstruction observation, not a
    # tracked-mass source, and the paired L/I event retains both endpoints
    # rather than hiding them in a correction.
    events = (
        {
            "event_id": f"step:{step_number}:operator:conversion",
            "operator": "conversion",
            "tracked_mass": True,
            "net_delta": float(conversion),
            "gross_magnitude": abs(float(conversion)),
        },
        {
            "event_id": f"step:{step_number}:operator:redistribution",
            "operator": "redistribution",
            "tracked_mass": True,
            "net_delta": float(redistribution),
            "gross_magnitude": abs(float(redistribution)),
        },
        {
            "event_id": f"step:{step_number}:operator:clamp",
            "operator": "clamp",
            "tracked_mass": True,
            "net_delta": float(clamp),
            "gross_magnitude": abs(float(clamp)),
        },
        {
            "event_id": f"step:{step_number}:operator:isolation",
            "operator": "isolation",
            "tracked_mass": True,
            "net_delta": float(isolation),
            "gross_magnitude": abs(float(isolation)),
        },
        {
            "event_id": f"step:{step_number}:operator:boundary",
            "operator": "boundary",
            "tracked_mass": True,
            "net_delta": float(boundary),
            "gross_magnitude": abs(float(boundary)),
        },
        {
            "event_id": f"step:{step_number}:operator:abb",
            "operator": "abb",
            "tracked_mass": False,
            "population_delta": float(abb_population_delta),
            "net_delta": 0.0,
            "gross_magnitude": abs(float(abb_population_delta)),
        },
        {
            "event_id": f"step:{step_number}:operator:interface_paired_debit",
            "operator": "interface_paired_debit",
            "tracked_mass": True,
            "interface_credit": float(exchange_liquid_credit),
            "bulk_debit": float(exchange_bulk_debit),
            "net_delta": float(paired_net),
            "gross_magnitude": (
                abs(float(exchange_liquid_credit)) + abs(float(exchange_bulk_debit))
            ),
        },
    )
    tracked_events = tuple(event for event in events if event["tracked_mass"])
    gross = max(events, key=lambda event: float(event["gross_magnitude"]))
    # Reconcile *all* tracked deltas independently of the legacy unexplained
    # field (which deliberately leaves a one-sided L/I credit unexplained).
    # No tolerance is used for attribution: a cause is named only when exactly
    # one non-zero tracked delta supplies the expected drift.  Thus large,
    # opposite conversion/redistribution activity cannot be mislabeled as the
    # cause of their tiny cancellation residual.
    tracked_delta_sum = sum(float(event["net_delta"]) for event in tracked_events)
    reconciliation_residual = float(mass_drift) - tracked_delta_sum
    active_events = tuple(event for event in tracked_events if float(event["net_delta"]) != 0.0)
    if len(active_events) == 1 and reconciliation_residual == 0.0:
        root_event = active_events[0]
        root_operator = root_event["operator"]
        root_reason = "single_tracked_operator_reconciles_observed_drift"
    elif len(active_events) == 0:
        root_event = None
        root_operator = "withheld/unexplained"
        root_reason = "no_nonzero_tracked_operator"
    elif reconciliation_residual != 0.0:
        root_event = None
        root_operator = "withheld/unexplained"
        root_reason = "tracked_deltas_do_not_reconcile_observed_drift"
    else:
        root_event = None
        root_operator = "withheld/unexplained"
        root_reason = "multiple_tracked_operators_no_unique_residual_cause"
    attribution = {
        "gross_activity_event_id": gross["event_id"],
        "gross_activity_operator": gross["operator"],
        "gross_activity_magnitude": float(gross["gross_magnitude"]),
        "dominant_event_id": None if root_event is None else root_event["event_id"],
        "dominant_operator": root_operator,
        "dominant_magnitude": 0.0 if root_event is None else abs(float(root_event["net_delta"])),
        "reason": root_reason,
        "events": events,
    }
    reconciliation = {
        "sum_tracked_deltas": float(tracked_delta_sum),
        "expected_drift": float(tracked_delta_sum),
        "observed_drift": float(mass_drift),
        "residual": float(reconciliation_residual),
    }
    record = {
        "step": step_number,
        "mass_start": float(mass_start),
        "mass_end": float(mass_end),
        "mass_drift": float(mass_drift),
        # Short aliases are intentionally stable campaign-facing budget names.
        "drift": float(mass_drift),
        "mass_after_exchange": float(mass_after_exchange),
        "mass_after_redistribution": float(mass_after_redistribution),
        "mass_after_clamp": float(mass_after_clamp),
        "mass_after_conversion": float(mass_after_conversion),
        "mass_after_isolation": float(mass_after_isolation),
        "mass_unit": "lattice liquid mass (sum of independent mass field)",
        "abb_population_delta": float(abb_population_delta),
        "abb_tracked_mass_source": 0.0,
        "liquid_interface_exchange": 0.0,
        "liquid_interface_interface_credit": float(exchange_liquid_credit),
        "liquid_interface_neighbor_credit": float(exchange_interface_credit),
        "liquid_interface_bulk_debit": float(exchange_bulk_debit),
        "liquid_interface_paired_residual": float(paired_net),
        "liquid_interface_paired": bool(paired_liquid_interface_debit),
        "redistribution": float(redistribution),
        "clamp": float(clamp),
        "conversion": float(conversion),
        "isolation": float(isolation),
        "boundary": float(boundary),
        "unexplained_residual": float(unexplained),
        "unexplained": float(unexplained),
        "paired_residual": float(paired_net),
        "paired": bool(paired_liquid_interface_debit),
        # roundoff even though every link has an equal/opposite counterpart.
        "closed_domain_conserved": abs(float(unexplained)) <= 1.0e-6,
        "diagnostic": (
            "one-sided liquid/interface mass credit or unpaired interface/interface credit"
            if abs(float(unexplained)) > 1.0e-6
            else "tracked-mass ledger balances; not a physical/PV closure claim"
        ),
        "operator_attribution": attribution,
        "residual_reconciliation": reconciliation,
        # Observation only: exact conversion/redistribution cell-link evidence
        # prevents reduction-order residuals from becoming root-cause claims.
        "conversion_evidence": conversion_evidence,
        "direct_liquid_gas_links": 0,
        "directLG": 0,
    }
    steps.append(record)
    # A curve is append-only and retains the exact event identity used for each
    # point, so a long-run gate can cite an operator rather than a bare step.
    ledger.setdefault("operator_curve", []).append(
        {
            "step": step_number,
            "mass_drift": float(mass_drift),
            "unexplained_residual": float(unexplained),
            "sum_tracked_deltas": reconciliation["sum_tracked_deltas"],
            "expected_drift": reconciliation["expected_drift"],
            "reconciliation_residual": reconciliation["residual"],
            "dominant_operator": attribution["dominant_operator"],
            "dominant_event_id": attribution["dominant_event_id"],
            "gross_activity_operator": attribution["gross_activity_operator"],
            "attribution_reason": attribution["reason"],
        }
    )
    ledger.update(record)


def _append_ownership_ledger(
    ledger,
    *,
    flags,
    mass_delta_liquid,
    liquid_interface_mask,
    paired_liquid_interface_debit,
    conversion_evidence,
    abb_population_delta,
):
    """Append cold tracked-state ownership evidence without solver feedback."""
    from .free_surface_ownership_ledger import build_ownership_ledger

    state = build_ownership_ledger(
        flags=flags,
        mass_delta_liquid=mass_delta_liquid,
        liquid_interface_mask=liquid_interface_mask,
        paired_liquid_interface_debit=paired_liquid_interface_debit,
        conversion_evidence=conversion_evidence,
        abb_population_delta=abb_population_delta,
    )
    ledger.setdefault("steps", []).append(state)
    ledger["latest"] = state


def _append_inventory_reconciliation(ledger, stages):
    """Publish cold actual-state stage measurements after a successful step."""
    if tuple(stages) != CANONICAL_STAGE_ORDER:
        raise ValueError("inventory reconciliation stages must use canonical chronological order")
    deltas = inventory_stage_deltas(stages)
    total_delta = (
        stages["after_topology_halo_isolation_boundary"]["total_liquid_inventory"]
        - stages["before_collision"]["total_liquid_inventory"]
    )
    summed = sum(delta["total_liquid_inventory"] for delta in deltas.values())
    ledger["status"] = "DIAGNOSTIC_WITHHELD_NOT_PHYSICAL_CLOSURE"
    ledger["operator_attribution_status"] = "OBSERVED_COMBINED_NOT_ATOMIC"
    ledger["stages"] = stages
    ledger["stage_deltas"] = deltas
    ledger["pre_topology_combined_total_liquid_inventory_delta"] = float(
        stages["after_mass_exchange"]["total_liquid_inventory"]
        - stages["before_collision"]["total_liquid_inventory"]
    )
    ledger["observed_total_liquid_inventory_delta"] = float(total_delta)
    ledger["sum_stage_total_liquid_inventory_delta"] = float(summed)
    ledger["total_liquid_inventory_reconciliation_residual"] = float(total_delta - summed)
    ledger["abb_inventory_status"] = "POPULATION_ONLY_WITHHELD"


# ===========================================================================
# Initialization
# ===========================================================================


def init_fill_rectangular(nz, ny, nx, column_width, column_height, device):
    """Initialize fill field for rectangular liquid column (dam-break IC)."""
    fill = torch.zeros((nz, ny, nx), dtype=torch.float32, device=device)
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    solid[:, :, 0] = True
    solid[:, :, -1] = True
    cw, ch = int(column_width), int(column_height)
    fill[:, :ch, 1:cw] = 1.0
    fx, fy = column_width - cw, column_height - ch
    if fx > 0 and cw < nx - 1:
        fill[:, :ch, cw] = fx
    if fy > 0 and ch < ny - 1:
        fill[:, ch, 1:cw] = fy
    if fx > 0 and fy > 0 and cw < nx - 1 and ch < ny - 1:
        fill[:, ch, cw] = 0.5 * (fx + fy)
    return fill, solid


def init_flags_from_fill(fill, solid_mask):
    """Set flags from fill and create the required D3Q19 interface envelope.

    A zero-fill GAS cell directly linked to LIQUID has no Körner boundary
    reconstruction.  Such cells begin as empty INTERFACE cells so every
    D3Q19 liquid/gas link is represented by the interface model.
    """
    flags = torch.full_like(fill, GAS, dtype=torch.int8)
    flags[fill >= 1.0] = LIQUID
    flags[(fill > 0) & (fill < 1)] = INTERFACE
    liquid = flags == LIQUID
    liquid_neighbor = torch.stack(all_moving_neighbor_masks(liquid)).any(dim=0)
    flags[(flags == GAS) & liquid_neighbor & ~solid_mask] = INTERFACE
    flags[solid_mask] = SOLID
    return flags


def init_mass_from_fill(fill, flags, rho_liquid=1.0):
    """Initialize mass field: mass = fill * rho_liquid for liquid/interface, 0 for gas."""
    mass = torch.zeros_like(fill)
    mass[flags == LIQUID] = rho_liquid
    mass[flags == INTERFACE] = fill[flags == INTERFACE] * rho_liquid
    return mass


def total_liquid_inventory(f, fill, flags, rho_liquid=1.0):
    """Return liquid inventory with bulk density in LIQUID and fill mass at INTERFACE.

    This diagnostic deliberately does not use the independent ``mass`` field
    for LIQUID cells: their physically represented amount is the LBM density
    ``sum_q f_q``.  Interface cells instead contribute their bounded liquid
    fill ``rho_liquid * fill``.  GAS and SOLID cells contribute nothing.
    """
    rho = f.sum(dim=0)
    return (
        torch.where(flags == LIQUID, rho, torch.zeros_like(rho)).sum()
        + torch.where(flags == INTERFACE, fill * rho_liquid, torch.zeros_like(fill)).sum()
    )


# ===========================================================================
# Helper: init new cells with neighbor-averaged velocity
# ===========================================================================


def _init_new(f, flags, mask, rho_init, device, ux=None, uy=None, uz=None):
    """Init newly converted cells with neighbor-averaged velocity.

    Vectorized — no .any() sync (multicard-safe under TCCL).
    torch.where handles empty mask as no-op.
    """
    if ux is None or uy is None or uz is None:
        _, ux, uy, uz = macroscopic3d(f)
    active = (flags == LIQUID) | (flags == INTERFACE)
    sl_stack = torch.stack(all_moving_neighbor_masks(active))
    ux_stack = torch.stack([roll_from_pull_source(ux, q) for q in D3Q19_MOVING_Q])
    uy_stack = torch.stack([roll_from_pull_source(uy, q) for q in D3Q19_MOVING_Q])
    uz_stack = torch.stack([roll_from_pull_source(uz, q) for q in D3Q19_MOVING_Q])
    sl_f = sl_stack.float()
    uxa = (ux_stack * sl_f).sum(dim=0)
    uya = (uy_stack * sl_f).sum(dim=0)
    uza = (uz_stack * sl_f).sum(dim=0)
    cnt = sl_f.sum(dim=0).clamp(min=1)
    rho_f = torch.full_like(ux, float(rho_init))
    feq = equilibrium3d(rho_f, uxa / cnt, uya / cnt, uza / cnt)
    return torch.where(mask.unsqueeze(0), feq, f)


# ===========================================================================
# Interface normal computation (for mass redistribution)
# ===========================================================================


def _compute_interface_normal(flags, mass, rho):
    """Compute interface normal n = -∇fill / |∇fill| (points from liquid to gas)."""
    fill = mass / rho.clamp(min=1e-6)
    # Gradient via central difference
    grad_x = 0.5 * (fill.roll(-1, dims=2) - fill.roll(1, dims=2))
    grad_y = 0.5 * (fill.roll(-1, dims=1) - fill.roll(1, dims=1))
    grad_z = 0.5 * (fill.roll(-1, dims=0) - fill.roll(1, dims=0))
    mag = (grad_x**2 + grad_y**2 + grad_z**2).sqrt().clamp(min=1e-10)
    return -grad_x / mag, -grad_y / mag, -grad_z / mag


# ===========================================================================
# MRT collision with per-cell effective relaxation time (SGS support)
# ===========================================================================


def _collide_mrt3d_with_tau_eff(
    f: torch.Tensor,
    feq: torch.Tensor,
    tau_eff: torch.Tensor,
    device: torch.device,
    s_e: float = 1.19,
    s_eps: float = 1.4,
    s_q: float = 1.2,
    s_pi: float | None = None,
) -> torch.Tensor:
    """D3Q19 MRT collision with per-cell effective relaxation time.

    Identical to :func:`~tensorlbm.solver3d.collide_mrt3d` except that the
    stress-mode relaxation rates (rows 9–13) use the per-cell ``1/τ_eff(x)``
    instead of a scalar ``1/τ``.  This is the standard pattern for coupling
    MRT with any algebraic SGS model (Smagorinsky, WALE, Vreman).
    """
    if s_pi is None:
        s_pi = s_e
    M, M_inv = _get_d3q19_mrt_matrices(device, f.dtype)
    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    f_flat = f.reshape(19, -1)
    feq_flat = feq.reshape(19, -1)
    s_nu_flat = (1.0 / tau_eff).reshape(-1)  # (N,)

    m = M @ f_flat
    m_eq = M @ feq_flat
    dm = m - m_eq

    s_fixed = torch.tensor(
        [
            0.0,
            s_e,
            s_eps,
            0.0,
            s_q,
            0.0,
            s_q,
            0.0,
            s_q,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            s_pi,
            s_pi,
            1.0,
            1.0,
            1.0,
        ],
        dtype=f.dtype,
        device=device,
    )
    m_star = m - s_fixed.unsqueeze(1) * dm
    for k in (9, 10, 11, 12, 13):
        m_star[k] = m[k] - s_nu_flat * dm[k]
    return (M_inv @ m_star).reshape(19, nz, ny, nx)


# ===========================================================================
# Core timestep — full Körner model
# ===========================================================================


def free_surface_step(
    f,
    fill,
    flags,
    solid_mask,
    mass=None,
    tau=1.0,
    gx=0.0,
    gy=0.0,
    gz=0.0,
    rho_liquid=1.0,
    rho_gas=1.0,
    surface_tension=0.0,
    C_s=0.0,
    sgs_model="smagorinsky",
    free_slip_y=False,
    y_wall_mask=None,
    bubble_pressure=None,
    collision="bgk",
    wall_function=False,
    near_mask=None,
    y_val=0.5,
    wf_force_coef=None,
    mass_ledger=None,
    freeze_topology=False,
    runtime_ledger=None,
    paired_liquid_interface_debit=False,
    ownership_ledger=None,
    inventory_reconciliation_ledger=None,
    conversion_density_audit_ledger=None,
    enable_i_to_g_ownership_closure=False,
    capture_replay_stages=False,
    replay_capture=None,
):
    """One free-surface LBM timestep (full Körner model).

    Added vs simplified version:
      - Mass tracking (independent mass variable)
      - Mass redistribution during cell conversion
      - Interface gas pressure (anti-bounce-back)
      - Neighbor flags (prevents isolated cells)
    """
    if not isinstance(enable_i_to_g_ownership_closure, bool):
        raise ValueError("enable_i_to_g_ownership_closure must be bool")
    if not isinstance(capture_replay_stages, bool):
        raise ValueError("capture_replay_stages must be bool")
    if replay_capture is not None and not isinstance(replay_capture, dict):
        raise ValueError("replay_capture must be a dict or None")
    _VALID_SGS_MODELS = ("smagorinsky", "wale", "vreman")
    if sgs_model not in _VALID_SGS_MODELS:
        raise ValueError(f"sgs_model must be one of {_VALID_SGS_MODELS}, got {sgs_model!r}")

    # Ledger output is transactional too: topology validation may fail after
    # ABB/exchange observations were calculated.  Accumulate into a detached
    # copy and publish it only on a successful return, so a failed step never
    # leaks partial diagnostic state to its caller.
    published_mass_ledger = mass_ledger
    mass_ledger = None if published_mass_ledger is None else deepcopy(published_mass_ledger)
    published_runtime_ledger = runtime_ledger
    runtime_ledger = (
        None if published_runtime_ledger is None else deepcopy(published_runtime_ledger)
    )
    published_ownership_ledger = ownership_ledger
    ownership_ledger = (
        None if published_ownership_ledger is None else deepcopy(published_ownership_ledger)
    )
    published_inventory_reconciliation_ledger = inventory_reconciliation_ledger
    inventory_reconciliation_ledger = (
        None
        if published_inventory_reconciliation_ledger is None
        else deepcopy(published_inventory_reconciliation_ledger)
    )
    published_conversion_density_audit_ledger = conversion_density_audit_ledger
    conversion_density_audit_ledger = (
        None
        if published_conversion_density_audit_ledger is None
        else deepcopy(published_conversion_density_audit_ledger)
    )
    # Owners must be read from the pre-topology state.  This cold clone exists
    # only when callers request diagnostic ownership evidence.
    ownership_flags = None if ownership_ledger is None else flags.clone()

    device = f.device
    _assert_no_direct_liquid_gas_links(flags)
    c_dev = _C.to(device).float()
    non_gas = ~(flags == GAS)

    # Initialize mass if not provided
    if mass is None:
        mass = init_mass_from_fill(fill, flags, rho_liquid)
    mass_start_value = float(mass.sum())
    inventory_stages = None
    if inventory_reconciliation_ledger is not None:
        inventory_stages = {
            "before_collision": inventory_measurement(f, fill, flags, mass, rho_liquid=rho_liquid),
        }
    # Filled after conversion only for runtime-ledger callers; this avoids any
    # diagnostic allocation in production paths that do not request evidence.
    conversion_evidence = None
    if mass_ledger is not None:
        mass_ledger["start"] = float(mass.sum())
        mass_ledger["interface_start"] = float(mass[flags == INTERFACE].sum())
        mass_ledger["liquid_start"] = float(mass[flags == LIQUID].sum())
        mass_ledger["gas_start"] = float(mass[flags == GAS].sum())
        mass_ledger["fill_mass_start"] = float((fill * rho_liquid).sum())

    # ---- 1. Macroscopic + collision ----
    rho, ux, uy, uz = macroscopic3d(f)
    # (c) Velocity guard at the source.  ``macroscopic3d`` clamps rho only up to
    # a 1e-12 floor, so at a near-empty INTERFACE cell u = momentum / rho is
    # unbounded.  That u feeds the ABB gas reconstruction
    # (f_eq_gas + f_eq_gas[opp] - f_post[opp]) and every new-cell equilibrium,
    # where it drives f to inf/nan within a handful of steps.  Bound it to the
    # stable lattice range *before* it is used anywhere.
    if _env_bool("TL_FS_INPUT_GUARD", True):
        ux = ux.clamp(-0.5, 0.5)
        uy = uy.clamp(-0.5, 0.5)
        uz = uz.clamp(-0.5, 0.5)
    rho_s = rho.clamp(min=1e-6, max=rho_liquid * 3.0)
    ux_eq = (ux + tau * gx).clamp(-0.5, 0.5)
    uy_eq = (uy + tau * gy).clamp(-0.5, 0.5)
    uz_eq = (uz + tau * gz).clamp(-0.5, 0.5)
    feq = equilibrium3d(rho_s, ux_eq, uy_eq, uz_eq)

    # For advanced operators: set gas cells to small equilibrium (prevent NaN)
    if collision != "bgk":
        feq_gas = equilibrium3d(
            torch.full_like(rho_s, rho_gas),
            torch.zeros_like(rho_s),
            torch.zeros_like(rho_s),
            torch.zeros_like(rho_s),
        )
        f_collide = torch.where(non_gas.unsqueeze(0), f, feq_gas)
    else:
        f_collide = f

    if collision == "kbc":
        from .advanced_collision_d3q19 import collide_kbc_d3q19

        f = collide_kbc_d3q19(f_collide, tau, C_s=C_s if C_s > 0 else 0.1)
    elif collision == "cascaded":
        from .advanced_collision_d3q19 import collide_cascaded_d3q19

        f = collide_cascaded_d3q19(f_collide, tau, C_s=C_s if C_s > 0 else 0.1)
    elif collision == "cumulant":
        from .advanced_collision_d3q19 import collide_cumulant_d3q19

        f = collide_cumulant_d3q19(f_collide, tau, C_s=C_s if C_s > 0 else 0.1)
    elif collision == "mrt":
        if C_s > 0:
            if sgs_model == "smagorinsky":
                tau_eff = _smagorinsky_tau(
                    tau,
                    _neq_stress_norm_3d(f_collide - feq),
                    rho_s,
                    C_s,
                )
            elif sgs_model == "wale":
                nu_t = _wale_nu_t_3d(ux, uy, uz, C_s)
                tau_eff = _nu_t_to_tau_eff(tau, nu_t)
            else:  # vreman
                nu_t = _vreman_nu_t_3d(ux, uy, uz, C_s)
                tau_eff = _nu_t_to_tau_eff(tau, nu_t)
            f = _collide_mrt3d_with_tau_eff(f_collide, feq, tau_eff, device)
        else:
            from .solver3d import collide_mrt3d

            f = collide_mrt3d(f_collide, tau)
    else:  # bgk
        if C_s > 0:
            if sgs_model == "smagorinsky":
                tau_eff = _smagorinsky_tau(
                    tau,
                    _neq_stress_norm_3d(f_collide - feq),
                    rho_s,
                    C_s,
                )
            elif sgs_model == "wale":
                nu_t = _wale_nu_t_3d(ux, uy, uz, C_s)
                tau_eff = _nu_t_to_tau_eff(tau, nu_t)
            else:  # vreman
                nu_t = _vreman_nu_t_3d(ux, uy, uz, C_s)
                tau_eff = _nu_t_to_tau_eff(tau, nu_t)
            f = f_collide - (f_collide - feq) / tau_eff.unsqueeze(0)
        else:
            f = f_collide - (f_collide - feq) / tau

    # Guo gravity force (only when gravity is non-zero)
    if gx != 0.0 or gy != 0.0 or gz != 0.0:
        cs2 = 1.0 / 3.0
        cx = c_dev[:, 0].view(19, 1, 1, 1)
        cy = c_dev[:, 1].view(19, 1, 1, 1)
        cz = c_dev[:, 2].view(19, 1, 1, 1)
        w_dev = _W.to(device).float().view(19, 1, 1, 1)
        ng = non_gas.float()
        Fx = rho_liquid * gx * ng
        Fy = rho_liquid * gy * ng
        Fz = rho_liquid * gz * ng
        cu_force = cx * Fx.unsqueeze(0) + cy * Fy.unsqueeze(0) + cz * Fz.unsqueeze(0)
        f = f + (1.0 - 0.5 / tau) * w_dev * cu_force / cs2

    # Surface tension force (curvature correction, standard Körner)
    if surface_tension > 0:
        cx = c_dev[:, 0].view(19, 1, 1, 1)
        cy = c_dev[:, 1].view(19, 1, 1, 1)
        cz = c_dev[:, 2].view(19, 1, 1, 1)
        w_dev = _W.to(device).float().view(19, 1, 1, 1)
        cs2 = 1.0 / 3.0
        fill_field = mass / rho_s.clamp(min=1e-6)
        grad_x = 0.5 * (fill_field.roll(-1, dims=2) - fill_field.roll(1, dims=2))
        grad_y = 0.5 * (fill_field.roll(-1, dims=1) - fill_field.roll(1, dims=1))
        grad_z = 0.5 * (fill_field.roll(-1, dims=0) - fill_field.roll(1, dims=0))
        mag = (grad_x**2 + grad_y**2 + grad_z**2).sqrt().clamp(min=1e-10)
        nx, ny, nz = -grad_x / mag, -grad_y / mag, -grad_z / mag
        kappa = 0.5 * (
            (nx.roll(-1, dims=2) - nx.roll(1, dims=2))
            + (ny.roll(-1, dims=1) - ny.roll(1, dims=1))
            + (nz.roll(-1, dims=0) - nz.roll(1, dims=0))
        )
        Fx_st = surface_tension * kappa * grad_x
        Fy_st = surface_tension * kappa * grad_y
        Fz_st = surface_tension * kappa * grad_z
        cu_st = cx * Fx_st.unsqueeze(0) + cy * Fy_st.unsqueeze(0) + cz * Fz_st.unsqueeze(0)
        f = f + (1.0 - 0.5 / tau) * w_dev * cu_st / cs2

    # Clamp f to non-negative (numerical stability, prevents negative rho → NaN)
    f = f.clamp(min=0.0, max=rho_liquid * 3.0)

    # Remove NaN for non-BGK
    if collision != "bgk":
        f = torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    if inventory_stages is not None:
        inventory_stages["after_collision_and_forcing"] = inventory_measurement(
            f,
            fill,
            flags,
            mass,
            rho_liquid=rho_liquid,
        )

    # Preserve post-collision outgoing populations for anti-bounce-back (ABB).
    # For a missing pull population q at x, Körner ABB uses the *local*
    # outgoing f_bar(q)^*(x), not the population streamed into x in bar(q).
    f_post = torch.where(non_gas.unsqueeze(0), f, torch.zeros_like(f))
    f = f_post

    # ---- 2. Stream ----
    # ---- 2. Stream ----
    f = _stream19_roll(f)

    # ---- 2b. Zero gas cells AFTER streaming (prevent mass leak into gas) ----
    gas_mask_pre = flags == GAS
    f = torch.where(gas_mask_pre.unsqueeze(0), torch.zeros_like(f), f)
    if _env_bool("TL_FS_INPUT_GUARD", True):
        # (c) Exchange input guard: never let a single non-finite / runaway
        # population enter the tracked-mass stencil.  The ABB reconstruction
        # below reads the same field, so this is also the last line of defence
        # before a non-finite f is committed for the next step.
        f = torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0).clamp(
            min=0.0, max=rho_liquid * 3.0
        )
    # F3: freeze the mass-exchange population *before* the anti-bounce-back gas
    # reconstruction.  The exchange must read the pure streamed state; letting
    # the ABB-reconstructed populations (magnitude ~ rho_gas) into the exchange
    # couples the tracked liquid mass to the gas pressure BC and amplifies the
    # exchange delta ~100x between rho_gas = 0.1 and 1.0.
    f_exchange = f
    if inventory_stages is not None:
        inventory_stages["after_stream_and_gas_zero"] = inventory_measurement(
            f,
            fill,
            flags,
            mass,
            rho_liquid=rho_liquid,
        )

    # ---- 2c. Anti-bounce-back for interface cells (gas pressure) ----
    # Standard Körner: interface cells get f[q] = f[opp[q]] from gas directions
    # Vectorized: batch all 19 directions, reuse neighbor_flags in mass exchange
    # (no .any() sync — multicard-safe under TCCL; torch.where handles empty mask)
    iface_abb = flags == INTERFACE
    # _stream19_roll is a pull stream: population q at x originated at
    # x-c[q].  Interface link classification must use that source cell.
    neighbor_flags = torch.stack(
        [flags.roll(sz, dims=0).roll(sy, dims=1).roll(sx, dims=2) for sx, sy, sz in _C19_SHIFTS]
    )  # (19, nz, ny, nx)
    need_abb = iface_abb.unsqueeze(0) & (neighbor_flags == GAS)
    # Standard Körner ABB (pressure boundary) for a missing pull population:
    # f_q(x,t+dt) = f_q^eq(rho_g,u_g) + f_barq^eq(rho_g,u_g)
    #                 - f_barq^*(x,t).  With u_g=u_interface this fixes p_g.
    # The implementation has no separate gas velocity field, so use the local
    # interface velocity from the pre-collision macroscopics.
    _abb_mode = os.environ.get("TL_FS_ABB_MODE", "legacy").strip().lower()
    rho_g_field = torch.full_like(rho, float(rho_gas))
    # ------------------------------------------------------------------
    # Hydrostatic free-surface pressure closure (OPT-IN: TL_FS_ABB_MODE in
    # {"hydro","hydrofill"}; legacy is bit-for-bit preserved otherwise).
    #
    # Physics (Körner 2005 / Thürey 2007 free-surface pressure boundary).
    # On a free surface the ambient pressure is p_gas = rho_gas c_s^2 and the
    # standard ABB below already imposes it.  What the legacy form omits is the
    # *interfacial pressure closure* of a finite liquid column: a static column
    # of height H is in hydrostatic equilibrium (grad p = rho_l g_vec), so the
    # pressure the interface must "see" at depth below the local free surface is
    #
    #     p_int(x) = p_gas + (p_fill(x) - p_gas) + rho_l * g_vec . (x - x_ref)
    #
    # with
    #   * p_fill = c_s^2 [ rho_gas + (rho_l - rho_gas) fill ]  -- Körner pressure
    #     closure: a half-full interface cell carries the fill-weighted
    #     pressure, not the full rho_l (the H~W cube has a large fill gradient).
    #   * x_ref  = local free-surface height in the same vertical column, so the
    #     hydrostatic head rho_l g_vec.(x - x_ref) vanishes on the surface and
    #     grows with depth (the H~W-column hydrostatic imbalance).
    #
    # Both terms collapse to one equivalent ABB reference density
    #
    #     rho_ref(x) = rho_gas + (rho_l - rho_gas) fill
    #                  + coef * rho_l * (g_vec . (x - x_ref)) / c_s^2
    #
    # so the same ABB arithmetic can be reused.  TL_FS_HYDRO_COEF (default 1)
    # scales the head term (0 = fill-consistent pressure only).
    # ------------------------------------------------------------------
    if _abb_mode in ("hydro", "hydrofill"):
        _cs2 = 1.0 / 3.0
        _coef = float(os.environ.get("TL_FS_HYDRO_COEF", "1.0"))
        rho_ref = torch.full_like(rho, float(rho_gas))
        if _abb_mode == "hydrofill":
            rho_ref = rho_ref + (float(rho_liquid) - float(rho_gas)) * fill
        _gvec = (float(gx), float(gy), float(gz))
        _gax = max(range(3), key=lambda k: abs(_gvec[k]))
        if _coef != 0.0 and _gvec[_gax] != 0.0:
            idxb = torch.arange(rho.shape[_gax], device=device, dtype=rho.dtype)
            view = [1, 1, 1]
            view[_gax] = rho.shape[_gax]
            idxb = idxb.view(view)
            nongas_f = (flags != GAS).to(rho.dtype)
            _ref = torch.where(nongas_f > 0, idxb, torch.full_like(nongas_f, -1.0))
            ref = _ref.amax(dim=_gax, keepdim=True)
            # x - x_ref along the gravity axis (<= 0 below the free surface)
            dpos = idxb - ref
            rho_ref = rho_ref + _coef * float(rho_liquid) * _gvec[_gax] * dpos / _cs2
        rho_g_field = rho_ref.clamp(min=0.0)
    f_eq_gas = equilibrium3d(rho_g_field, ux, uy, uz)
    # Candidate physics: the legacy form subtracts the *local post-collision*
    # outgoing population f_post[opp], whose magnitude is O(rho_interface) ~
    # O(rho_liquid).  At high density ratio (rho_gas << rho_liquid) that makes
    # every reconstructed population strongly negative -- the interface
    # density then collapses and u = m/rho diverges.  The variants below keep
    # the equilibrium part at the gas pressure and only carry the outgoing
    # non-equilibrium stress, which is the physically meaningful content of a
    # free-surface pressure boundary.
    if _abb_mode == "eq":
        f_abb = f_eq_gas.expand_as(f)
    elif _abb_mode == "eqref":
        f_eq_local = equilibrium3d(rho_s, ux, uy, uz)
        f_abb = f_eq_gas + (f_post[_OPP.to(device)] - f_eq_local[_OPP.to(device)])
    elif _abb_mode == "eqrefflip":
        f_eq_local = equilibrium3d(rho_s, ux, uy, uz)
        f_abb = f_eq_gas - (f_post[_OPP.to(device)] - f_eq_local[_OPP.to(device)])
    elif _abb_mode == "sumeq":
        f_eq_local = equilibrium3d(rho_s, ux, uy, uz)
        f_abb = f_eq_gas + f_eq_gas[_OPP.to(device)] - f_eq_local[_OPP.to(device)]
    else:
        f_abb = f_eq_gas + f_eq_gas[_OPP.to(device)] - f_post[_OPP.to(device)]
    abb_delta = torch.where(need_abb, f_abb - f, torch.zeros_like(f))
    if mass_ledger is not None:
        # This is a population (not tracked-liquid-mass) change.  Keeping it
        # separate makes a gas-pressure boundary source distinguishable from
        # the subsequent liquid/interface mass stencil.
        mass_ledger["abb_population_delta"] = float(abb_delta.sum())
        mass_ledger["abb_population_abs_delta"] = float(abb_delta.abs().sum())
    # ABLATION (diagnostic only): TL_FS_ABL_ABB suppresses the gas-pressure
    # anti-bounce-back reconstruction to test whether it drives the interface.
    if _env_bool("TL_FS_ABL_ABB", False):
        f = torch.where(need_abb, f, f)
    else:
        f = torch.where(need_abb, f_abb, f)
    if inventory_stages is not None:
        inventory_stages["after_abb"] = inventory_measurement(
            f, fill, flags, mass, rho_liquid=rho_liquid
        )

    # ---- 3. Wall BCs ----
    # OPT-IN hydrostatic-consistent half-way bounce-back (TL_FS_WALL_MODE).
    #
    # Legacy: ``bounce_back_cells_3d`` reflects at the SOLID cell via an in-place
    # q <-> opp(q) swap.  In this pull-stream framework the *fluid* cell adjacent
    # to a wall reads its wall-entering populations from that solid cell BEFORE
    # the swap, i.e. it samples the post-collision solid state (and, on step 1,
    # the zero initial solid state).  The one-step budget shows the resulting
    # spurious wall-layer deficit: rho 1.0 -> 0.694 and u_y spike -0.20 EVEN AT
    # g = 0 (gravity-independent), which drives the boundary layer into free
    # vibration and feeds the interface chain (pseudo-wetting film creep).
    #
    # Corrected form reflects at the *fluid* side using its own outgoing
    # populations (true half-way BB).  ``halfway_hydro`` additionally adds the
    # hydrostatic pressure head of a static column p = rho_l g_vec.(x - x_ref):
    #     f_q(x_f) = f_post[opp(q)](x_f) - 2 w_q rho_l (g_vec.c_q) / cs^2
    # (the discrete drop p(x_f) - p(x_f - c_q) mirrored into the entering
    # population, so it vanishes at the free surface x_ref and supports the fluid
    # against gravity near the wall).  Both collapse to a single where;
    # TL_FS_WALL_HYDRO_COEF (default 1) scales the head term.
    _wall_mode = os.environ.get("TL_FS_WALL_MODE", "legacy").strip().lower()
    if _wall_mode in ("halfway", "halfway_hydro"):
        _src_solid = torch.stack(
            [
                solid_mask.roll(sz, dims=0).roll(sy, dims=1).roll(sx, dims=2)
                for sx, sy, sz in _C19_SHIFTS
            ]
        )  # (19,nz,ny,nx): pull source x - c_q lies in SOLID
        _fluid_bb = (~solid_mask).unsqueeze(0) & _src_solid
        _f_reflect = f_post[_OPP.to(device)]
        if _wall_mode == "halfway_hydro" and (gx != 0.0 or gy != 0.0 or gz != 0.0):
            _cs2_w = 1.0 / 3.0
            _coef_w = float(os.environ.get("TL_FS_WALL_HYDRO_COEF", "1.0"))
            _gvec_w = torch.tensor([float(gx), float(gy), float(gz)], device=device)
            _gc_w = (c_dev * _gvec_w.view(1, 3)).sum(dim=1)  # (19,) = g_vec . c_q
            _w19_w = _W.to(device).float()
            _head_w = (
                -2.0 / _cs2_w
            ) * _coef_w * float(rho_liquid) * _w19_w * _gc_w  # (19,)
            _f_reflect = _f_reflect + _head_w.view(19, 1, 1, 1)
        f = torch.where(_fluid_bb, _f_reflect, f)
    f = bounce_back_cells_3d(f, solid_mask)
    if free_slip_y and y_wall_mask is not None:
        f = free_slip_cells_3d(f, y_wall_mask, axis=1)

    # ---- 3b. Wall function (optional, for hull resistance) ----
    # Vectorized Newton iteration (no bool(turb.any()) sync — multicard-safe
    # under TCCL). Apply Newton to ALL cells, then select with torch.where.
    df = torch.tensor(0.0, device=device, dtype=f.dtype)
    if wall_function and near_mask is not None:
        rho_wf, ux_wf, uy_wf, uz_wf = macroscopic3d(f)
        u_mag = torch.sqrt(ux_wf**2 + uy_wf**2 + uz_wf**2).clamp(min=1e-12)
        nu_lat = (tau - 0.5) / 3.0
        u_tau = torch.sqrt(nu_lat * u_mag / y_val).clamp(min=1e-12)
        y_plus = y_val * u_tau / nu_lat
        turb = (y_plus > 11.6) & near_mask
        # Vectorized Newton: apply to ALL cells, then select with torch.where
        ut = u_tau.clone()
        um = u_mag
        for _ in range(8):
            lyp = torch.log(y_val * ut / nu_lat)
            fv = ut * (lyp / KAPPA + B_CONST) - um
            fp = (lyp / KAPPA + B_CONST) + 1.0 / KAPPA
            ut = (ut - fv / fp.clamp(min=1e-10)).clamp(min=1e-12)
        u_tau = torch.where(turb, ut, u_tau)
        tau_w = u_tau * u_tau
        inv_umag = 1.0 / u_mag
        coef = -(tau_w / y_val) * near_mask.to(f.dtype)
        fx = coef * (ux_wf * inv_umag)
        fy = coef * (uy_wf * inv_umag)
        fz = coef * (uz_wf * inv_umag)
        cx = c_dev[:, 0].view(19, 1, 1, 1)
        cy = c_dev[:, 1].view(19, 1, 1, 1)
        cz = c_dev[:, 2].view(19, 1, 1, 1)
        w_dev = _W.to(device).float().view(19, 1, 1, 1)
        cs2 = 1.0 / 3.0
        cu_force = cx * fx + cy * fy + cz * fz
        # Wall-function forcing: decoupled from tau when wf_force_coef is set
        # (at high Re, tau≈0.5 → standard Guo factor (1-0.5/tau)≈0, so the
        # wall force is never applied and the flow never decelerates).
        wf_coef = wf_force_coef if wf_force_coef is not None else (1.0 - 0.5 / tau)
        forcing = wf_coef * w_dev * cu_force / cs2
        f = f + forcing
        f = f.clamp(min=0.0, max=rho_liquid * 3.0)  # prevent inf from wall function forcing
        df = (tau_w * near_mask.to(f.dtype)).sum()
    if inventory_stages is not None:
        inventory_stages["after_wall_boundary"] = inventory_measurement(
            f,
            fill,
            flags,
            mass,
            rho_liquid=rho_liquid,
        )

    # ---- 4. Mass exchange (standard Körner, independent mass variable) ----
    # (no .any() sync — multicard-safe under TCCL; torch.where handles empty masks)
    iface_mask = flags == INTERFACE
    # F1: unified interface birth/receive semantics.
    # Every INTERFACE cell is a legal mass-exchange receiver — including a
    # freshly-born envelope cell at fill = 0.  The previous ``fill > 1e-3``
    # gate sealed the column: ``init_flags_from_fill`` / ``to_i`` birth the
    # envelope at fill = 0, so a gated receiver could never fill and no
    # L/I -> I/L conversion could ever fire (front frozen at X = 1.0).
    # Conservation is restored by pairing every I/I link (antisymmetric half
    # weight) and every L/I link with an explicit bulk debit below, so an
    # ungated receiver is a *transfer*, never a source.
    recv_ok = iface_mask
    # F1': GRADED receive gate — resolves the "freeze vs pseudo-advance" tension.
    #
    # The F1 open gate (``recv_ok = iface_mask``) treats *every* INTERFACE cell
    # as a legal mass-exchange receiver, including a freshly-born envelope cell
    # at fill = 0.  ``init_flags_from_fill`` / the halo promotion birth that
    # envelope with an equilibrium population at rho_liquid but tracked mass 0.
    # A single D3Q19 link into such an empty shell then siphons real liquid mass
    # into it (gravity-independent, ~0.04 cell/step): once mass >= 0.999 rho_L
    # the cell converts to LIQUID and the halo lifts the next GAS cell to
    # INTERFACE -> a pseudo-wetting film that pollutes X and H.
    #
    # The pre-F1 strict gate (fill > 1e-3) closed *all* empty shells, which also
    # sealed the genuine gravity-driven spread (front frozen at X = 1.0).
    #
    # The graded gate distinguishes the two cases by the *support* of the shell:
    #   tier 1 (established): the INTERFACE cell already carries tracked liquid
    #     mass (mass > m_eps) -> always a legal receiver (real interface).
    #   tier 2 (supported birth): a fresh / empty shell (mass <= m_eps) may
    #     receive only when it is backed by a *real liquid surface*, i.e. it has
    #     at least N_MIN of its D3Q19 neighbours in LIQUID.  A lone corner finger
    #     (1-2 liquid links) — the pseudo-film seed — is locked out; a proper
    #     face (>= 3-4 liquid links) is admitted so gravity-driven spreading can
    #     still fill the leading interface.
    if _env_bool("TL_FS_RECV_GRADED", False):
        m_eps = float(os.environ.get("TL_FS_RECV_MASS_EPS", "0.01")) * rho_liquid
        n_min = int(os.environ.get("TL_FS_RECV_NMIN", "3"))
        established = mass > m_eps
        n_liq_nbr_pre = (neighbor_flags == LIQUID).sum(dim=0)
        supported = n_liq_nbr_pre >= n_min
        recv_ok = iface_mask & (established | supported)
    # ABLATION (diagnostic only): TL_FS_ABL_RECVCLOSED restores the pre-F1
    # fill gate so a freshly-born envelope cell at fill=0 cannot receive.
    if _env_bool("TL_FS_ABL_RECVCLOSED", False):
        recv_ok = iface_mask & (fill > 1e-3)
    # EXPERIMENTAL knobs: apply the gate to the L/I and I/I channels separately.
    _liq_only = _env_bool("TL_FS_RECV_LIQ_ONLY", False)
    _iface_only = _env_bool("TL_FS_RECV_IFACE_ONLY", False)
    recv_ok_liq = recv_ok if not _iface_only else iface_mask
    recv_ok_iface = recv_ok if not _liq_only else iface_mask
    recv_19 = recv_ok_liq.unsqueeze(0)
    recv_19_iface = recv_ok_iface.unsqueeze(0)

    # --- A-prime exchange closure (OPT-IN: TL_FS_APRIME=1) ----------------
    # Design doc §6.2 closure redesign, exchange end.
    #
    # Legacy: the L/I channel is paired by the explicit bulk debit below, but
    # the I/I channel is a bare per-cell half-weight whose net vanishes only
    # when the receiver gate is *symmetric across every link*.  A graded gate
    # (graded_n3..n5) is asymmetric (one endpoint established, the other not):
    # measured 288 asymmetric I/I links, +1.7..2% static-column drift.  Even
    # the open gate still siphons real liquid into the fill=0 shells born at
    # mass 0 while their populations sit at rho_liquid -- the pseudo-wetting
    # film.
    #
    # A-prime fixes both ends of the exchange:
    #  (a) legal-receiver promotion (per-link): an I/I link is legal only when
    #      BOTH endpoints are legal receivers;  an L/I link only when its
    #      INTERFACE target is a legal receiver.
    #  (b) fill~=0 guard, paired accounting: a link is dead when EITHER
    #      endpoint is an *unsupported* near-empty shell (mass <= eps_shell and
    #      fewer than n_min LIQUID neighbours) -- the pseudo-film seed.  The
    #      predicate is a symmetric function of the undirected link, so the
    #      paired half-weight stays exactly antisymmetric and the I/I net is
    #      identically zero.  No global rescale, no topology mutation.
    #
    # Legacy behaviour is untouched unless the switch is set (bit-identical).
    if _env_bool("TL_FS_APRIME", False):
        _nbr_recv = torch.stack(list(all_moving_neighbor_masks(recv_ok_iface)))
        _nbr_recv = torch.cat([recv_ok_iface.unsqueeze(0), _nbr_recv], dim=0)
        _eps_shell = float(os.environ.get("TL_FS_APRIME_SHELL_EPS", "0.01")) * rho_liquid
        _n_min = int(os.environ.get("TL_FS_APRIME_NMIN", "3"))
        _n_liq_nbr = (neighbor_flags == LIQUID).sum(dim=0)
        _shell = (mass <= _eps_shell) & iface_mask
        _unsupported = _shell & (_n_liq_nbr < _n_min)
        _live_mv = ~_unsupported.unsqueeze(0) & ~torch.stack(
            list(all_moving_neighbor_masks(_unsupported))
        )
        # Index 0 (rest direction) is a no-op link: keep it live so the (19,)
        # mask aligns with ``neighbor_flags``.
        _live = torch.cat([torch.ones_like(_live_mv[:1]), _live_mv], dim=0)
        recv_19 = recv_ok_liq.unsqueeze(0) & _live
        recv_19_iface = recv_ok_iface.unsqueeze(0) & _nbr_recv & _live
    # neighbor_flags always computed in anti-bounce-back above (no None check)
    # For pull link q at x, the opposing outgoing population belongs to x
    # itself: f_bar(q)^*(x).  Sampling it at x-c_q mixes two different links.
    f_opp_nb = f_post[_OPP.to(device)]  # (19, nz, ny, nx)
    from_liq = recv_19 & (neighbor_flags == LIQUID)
    from_iface = recv_19_iface & (neighbor_flags == INTERFACE)
    mass_delta_liquid = torch.where(from_liq, f_exchange - f_opp_nb, torch.zeros_like(f))
    mass_delta_interface = torch.where(
        from_iface, (f_exchange - f_opp_nb) * 0.5, torch.zeros_like(f)
    )
    # A L/I credit at interface target x is paired link-by-link with a debit
    # at its pull source x-c_q.  This uses only existing D3Q19 links; it is
    # neither a global rescale nor a topology mutation.
    mass_delta_bulk_debit = -torch.stack(
        [
            mass_delta_liquid[q].roll((-sz, -sy, -sx), dims=(0, 1, 2))
            for q, (sx, sy, sz) in enumerate(_C19_SHIFTS)
        ]
    ).sum(0)
    mass_delta = (
        mass_delta_liquid
        +
        # Gas is a pressure boundary, not a liquid-mass reservoir.  Adding
        # its reconstructed population here spuriously creates tracked liquid
        # mass in a quiescent closed column.
        torch.zeros_like(f)
        + mass_delta_interface
    ).sum(0)
    if paired_liquid_interface_debit:
        valid_bulk_owner = flags == LIQUID
        invalid_debit = mass_delta_bulk_debit.masked_select(~valid_bulk_owner)
        if bool((invalid_debit.abs() > 1.0e-8).any()):
            raise RuntimeError("L/I paired debit has no LIQUID bulk owner")
        mass_delta = mass_delta + torch.where(
            valid_bulk_owner, mass_delta_bulk_debit, torch.zeros_like(mass_delta_bulk_debit)
        )
    if _env_bool("TL_FS_INPUT_GUARD", True):
        # (c) Bound the tracked-mass increment: a non-finite or runaway exchange
        # delta must not be committed to the independent mass field.
        mass_delta_preclamp = mass_delta
        mass_delta = torch.nan_to_num(mass_delta, nan=0.0, posinf=0.0, neginf=0.0).clamp(
            -float(rho_liquid), float(rho_liquid)
        )
    else:
        mass_delta_preclamp = mass_delta
    if _env_bool("TL_FS_DBG", False) and mass_ledger is not None:
        mass_ledger["dbg_mass_delta_preclamp_sum"] = float(mass_delta_preclamp.sum())
        mass_ledger["dbg_mass_delta_postclamp_sum"] = float(mass_delta.sum())
        mass_ledger["dbg_mass_delta_absmax"] = float(mass_delta_preclamp.abs().max())
        mass_ledger["dbg_clamp_touch"] = float(
            (mass_delta_preclamp.abs() > float(rho_liquid)).sum()
        )
        mass_ledger["dbg_debit_nonliq_abs"] = float(
            mass_delta_bulk_debit.masked_select(~(flags == LIQUID)).abs().sum()
        )
        # A-prime leak forensics: the I/I (interface-interface) channel is
        # applied as a per-cell half-weight, so its net requires BOTH link
        # endpoints to pass the receiver gate.  An asymmetric gate (one endpoint
        # established, the other not) breaks the antisymmetry and leaks.
        mass_ledger["dbg_mass_delta_interface_sum"] = float(mass_delta_interface.sum())
        mass_ledger["dbg_mass_delta_liquid_sum"] = float(mass_delta_liquid.sum())
        mass_ledger["dbg_mass_delta_bulk_debit_sum"] = float(mass_delta_bulk_debit.sum())
        _ii_gate = torch.stack(list(all_moving_neighbor_masks(recv_ok_iface)))
        # asymmetric I/I links: gate(x) xor gate(pull source y) on I/I links.
        _ii_nb = neighbor_flags[list(D3Q19_MOVING_Q)]
        _ii_asym = (
            (_ii_nb == INTERFACE)
            & recv_ok_iface.unsqueeze(0)
            & ~_ii_gate
        )
        mass_ledger["dbg_ii_asym_links"] = float(_ii_asym.sum())
    mass = torch.where(~solid_mask, mass + mass_delta, mass)
    if _env_bool("TL_FS_DIAG_FIELD", False):
        globals()["_FS_DIAG"] = {
            "mass_delta": mass_delta.clone(),
            "from_liq": from_liq.clone(),
            "from_iface": from_iface.clone(),
            "f_exchange": f_exchange.clone(),
            "f_post": f_post.clone(),
            "flags": flags.clone(),
            "mass_after_exchange": mass.clone(),
            "mass_delta_liquid": mass_delta_liquid.clone().sum(0),
        }
    mass_after_exchange_value = float(mass.sum())
    fill = torch.where(~solid_mask, (mass / rho_liquid).clamp(0.0, 1.0), fill)
    if inventory_stages is not None:
        inventory_stages["after_mass_exchange"] = inventory_measurement(
            f,
            fill,
            flags,
            mass,
            rho_liquid=rho_liquid,
        )
    if mass_ledger is not None:
        mass_ledger["exchange"] = float(mass.sum())
        mass_ledger["exchange_liquid_delta"] = float(mass_delta_liquid.sum())
        mass_ledger["exchange_interface_delta"] = float(mass_delta_interface.sum())
        mass_ledger["exchange_bulk_debit"] = float(
            mass_delta_bulk_debit.sum() if paired_liquid_interface_debit else 0.0
        )
        mass_ledger["exchange_gas_delta"] = 0.0
        mass_ledger["fill_mass_after_exchange"] = float((fill * rho_liquid).sum())

    # Diagnostic mode: retain the post-stream/post-ABB populations and the
    # exchange result, but deliberately forbid flag conversion, redistribution
    # and halo propagation.  This isolates one mixed L/I/G topology update.
    if freeze_topology:
        if mass_ledger is not None:
            frozen_total = float(mass.sum())
            mass_ledger["redistribution"] = frozen_total
            mass_ledger["clamp"] = frozen_total
            mass_ledger["conversion"] = frozen_total
            mass_ledger["isolation"] = frozen_total
            mass_ledger["boundary"] = frozen_total
            mass_ledger["fill_mass_final"] = float((fill * rho_liquid).sum())
        if runtime_ledger is not None:
            _append_runtime_ledger(
                runtime_ledger,
                mass_start=mass_start_value,
                mass_after_exchange=mass_after_exchange_value,
                mass_after_redistribution=mass_after_exchange_value,
                mass_after_clamp=mass_after_exchange_value,
                mass_after_conversion=mass_after_exchange_value,
                mass_after_isolation=mass_after_exchange_value,
                mass_end=float(mass.sum()),
                abb_population_delta=float(abb_delta.sum()),
                exchange_liquid_credit=float(mass_delta_liquid.sum()),
                exchange_interface_credit=float(mass_delta_interface.sum()),
                exchange_bulk_debit=float(
                    mass_delta_bulk_debit.sum() if paired_liquid_interface_debit else 0.0
                ),
                paired_liquid_interface_debit=paired_liquid_interface_debit,
                conversion_evidence=conversion_evidence,
            )
        if ownership_ledger is not None:
            assert ownership_flags is not None
            _append_ownership_ledger(
                ownership_ledger,
                flags=ownership_flags,
                mass_delta_liquid=mass_delta_liquid,
                liquid_interface_mask=from_liq,
                paired_liquid_interface_debit=paired_liquid_interface_debit,
                conversion_evidence=conversion_evidence,
                abb_population_delta=float(abb_delta.sum()),
            )
        if inventory_reconciliation_ledger is not None:
            assert inventory_stages is not None
            after_mass_exchange = inventory_stages["after_mass_exchange"]
            _append_inventory_reconciliation(
                inventory_reconciliation_ledger,
                {
                    **inventory_stages,
                    "after_topology_redistribution": after_mass_exchange,
                    "after_topology_clamp": after_mass_exchange,
                    "after_topology_conversion": after_mass_exchange,
                    "after_topology_halo_isolation_boundary": after_mass_exchange,
                },
            )
        if conversion_density_audit_ledger is not None:
            from .free_surface_conversion_density_audit import build_conversion_density_audit

            conversion_density_audit_ledger["audit"] = build_conversion_density_audit(
                None, rho_liquid=rho_liquid
            )
            conversion_density_audit_ledger["status"] = "DIAGNOSTIC_WITHHELD_NOT_PHYSICAL_CLOSURE"
        if published_mass_ledger is not None:
            published_mass_ledger.clear()
            published_mass_ledger.update(mass_ledger)
        if published_runtime_ledger is not None:
            published_runtime_ledger.clear()
            published_runtime_ledger.update(runtime_ledger)
        if published_ownership_ledger is not None:
            published_ownership_ledger.clear()
            published_ownership_ledger.update(ownership_ledger)
        if published_conversion_density_audit_ledger is not None:
            published_conversion_density_audit_ledger.clear()
            published_conversion_density_audit_ledger.update(conversion_density_audit_ledger)
        if published_inventory_reconciliation_ledger is not None:
            published_inventory_reconciliation_ledger.clear()
            published_inventory_reconciliation_ledger.update(inventory_reconciliation_ledger)
        return f, fill, flags, mass, df
    gas_mask = flags == GAS
    interface_mask = flags == INTERFACE
    liquid_mask = flags == LIQUID

    # (a) Mass-based conversion gates.  ``fill`` is already
    # ``clamp(mass/rho_liquid, 0, 1)``, so it is sign-blind: a numerically
    # negative-mass INTERFACE cell reads as fill = 0 (indistinguishable from an
    # empty cell) and an over-full cell reads as fill = 1.  Reading the tracked
    # mass directly means a negative cell can never be mistaken for a full I→L
    # donor, and I→G only fires on a genuinely (near-)empty non-negative cell.
    if _env_bool("TL_FS_MASS_GATE", True):
        mass_gate = mass.clamp(min=0.0)
        _tog_eps = float(os.environ.get("TL_FS_TOGAS_EPS", "0.01")) * rho_liquid
        to_iface = gas_mask & (mass_gate > 0.01 * rho_liquid) & (~solid_mask)
        # ABLATION (diagnostic only): TL_FS_ABL_TOIFACE suppresses the gas->interface birth.
        if _env_bool("TL_FS_ABL_TOIFACE", False):
            to_iface = torch.zeros_like(to_iface)
        to_liq = interface_mask & (mass >= 0.999 * rho_liquid) & (~solid_mask)
        if _env_bool("TL_FS_TOGAS_NONNEG", False):
            # A numerically negative-mass cell must not enter I→G either: its
            # sign aliases into the redistribution and the conversion then
            # deletes it, which is the observed tracked-mass leak.  Leave it to
            # the conservation-preserving clamp, which settles it without a net
            # source/sink.
            to_gas = (
                (interface_mask | liquid_mask)
                & (mass >= 0.0)
                & (mass <= 0.01 * rho_liquid)
                & (~solid_mask)
            )
        else:
            to_gas = (
                (interface_mask | liquid_mask) & (mass_gate <= _tog_eps) & (~solid_mask)
            )
    else:
        # Legacy fill-gated path (A/B reference).
        to_iface = gas_mask & (fill > 0.01) & (~solid_mask)
        to_liq = interface_mask & (fill >= 0.999) & (~solid_mask)
        to_gas = (interface_mask | liquid_mask) & (fill <= 0.01) & (~solid_mask)

    # ---- 5a. Körner mass redistribution (excess → interface neighbors) ----
    # Excess mass at converting cells (vectorized, no bool sync)
    # I→L overflow and I→G independent-mass ownership are different
    # transactions.  The latter is an opt-in experimental diagnostic/proposal:
    # default solver arithmetic and topology stay bit-for-bit on the legacy
    # path until a strict local closure representation is established.
    i_to_g = to_gas & interface_mask
    # Disabled is the exact legacy solver path: this diagnostic/proposal must
    # not alter its tensors, topology, or existing campaign ledgers.  Only the
    # opt-in proposal removes I→G mass from the legacy redistribution before it
    # stages its own all-or-nothing candidate.
    redistribution_to_g = (
        to_gas if not enable_i_to_g_ownership_closure else (to_gas & ~interface_mask)
    )
    # (b) Positive-overflow-only excess.  The legacy expression adds an
    # INTERFACE cell's raw (possibly negative) mass as "excess"; a negative
    # contribution aliases the sign and cancels a genuine positive overflow
    # elsewhere, so the redistributed sum quietly nets to ~0 and the clamped
    # conversion destroys the difference.  Keep only the non-negative part.
    exp_pos_on = _env_bool("TL_FS_EXP_POS", True)
    if exp_pos_on:
        excess_to_liq = (mass - rho_liquid).clamp(min=0.0)
        excess_to_g = mass.clamp(min=0.0)
    else:
        excess_to_liq = mass - rho_liquid
        excess_to_g = mass
    excess = (
        torch.where(to_liq, excess_to_liq, torch.zeros_like(mass))
        + torch.where(redistribution_to_g, excess_to_g, torch.zeros_like(mass))
    ).clamp(min=0.0)
    # (b) Explicit donor for the negative mass that I→G clears.  Clamping the
    # positive excess discards the negative part; the negative cell must then be
    # settled *explicitly*, otherwise the conversion deletes it (a spurious
    # source) and the conservation-preserving clamp re-owns it (a double count).
    # The donor therefore (i) zeroes the negative INTERFACE donor cell itself and
    # (ii) charges the same negative amount to surviving INTERFACE receivers over
    # the moving D3Q19 links.  Donor + receivers net to exactly zero.
    donor_on = _env_bool("TL_FS_EXP_DONOR", False)
    neg_donor = (
        torch.where(redistribution_to_g & (mass < 0.0), mass, torch.zeros_like(mass))
        if donor_on
        else torch.zeros_like(mass)
    )
    # Existing interface cells receive first.  If a converting interface has
    # none, promote its adjacent gas halo to receivers in this same step; a
    # conversion must never silently discard excess merely because topology
    # propagation runs later in the step.
    recv_iface = interface_mask & ~to_liq & ~to_gas
    # Only positive overflow needs a newly promoted gas receiver.  An emptying
    # interface retains the established interface-only redistribution path.
    adjacent_converting = torch.stack(all_moving_neighbor_masks(to_liq)).any(dim=0)
    recv_new = gas_mask & adjacent_converting & ~solid_mask
    # ABLATION (diagnostic only): TL_FS_ABL_RECVNEW suppresses promoting gas
    # cells adjacent to a converting cell into redistribution receivers.
    if _env_bool("TL_FS_ABL_RECVNEW", False):
        recv_new = torch.zeros_like(recv_new)
    recv_mask = recv_iface | recv_new
    i_to_g_ownership = None
    if enable_i_to_g_ownership_closure and bool(i_to_g.any()):
        strict_failure_capture = None
        builder_invocation = {
            "flags": flags,
            "mass": mass,
            "to_gas": i_to_g,
            "to_liq": to_liq,
            "solid_mask": solid_mask,
            "gas_flag": GAS,
            "liquid_flag": LIQUID,
            "interface_flag": INTERFACE,
            "rho_liquid": rho_liquid,
        }
        if capture_replay_stages and replay_capture is not None:
            # Freeze before the builder receives any writable value.  The
            # capture also defines the exact detached strict-error replay.
            strict_failure_capture = capture_strict_failure_invocation(
                builder_invocation,
                builder="i_to_g_ownership",
            )
        # Keep capture disabled on the legacy call path.  Opt-in replay capture
        # makes the builder a transactional boundary: it may not retain or
        # mutate solver-owned fields, even on its exceptional path.
        builder_inputs = (
            {
                name: value.detach().clone() if isinstance(value, torch.Tensor) else value
                for name, value in builder_invocation.items()
            }
            if strict_failure_capture is not None
            else builder_invocation
        )
        try:
            i_to_g_ownership = build_i_to_g_ownership_transaction(
                builder_inputs["flags"],
                builder_inputs["mass"],
                to_gas=builder_inputs["to_gas"],
                to_liq=builder_inputs["to_liq"],
                solid_mask=builder_inputs["solid_mask"],
                gas_flag=GAS,
                liquid_flag=LIQUID,
                interface_flag=INTERFACE,
                rho_liquid=rho_liquid,
            )
        except TopologyTransactionError as error:
            if strict_failure_capture is not None:
                replay_capture["strict_failure_evidence"] = publish_strict_failure_evidence(
                    strict_failure_capture,
                    error,
                )
            raise
    # Count receiving cells per donor over every moving D3Q19 link.
    shifted_recv = torch.stack(all_moving_neighbor_masks(recv_mask))
    n_recv = shifted_recv.sum(dim=0).float().clamp(min=1.0)
    # Excess per receiving neighbor
    excess_per_nb = excess / n_recv
    # (b) Explicit negative-mass donor: charge the discarded negative mass to
    # surviving INTERFACE cells only, over the same moving D3Q19 links.  The
    # distributing to_gas donor is itself a (converting) INTERFACE cell, so
    # this is exactly "to_gas INTERFACE as donor, credit landing on surviving
    # INTERFACE".
    if donor_on and bool(neg_donor.any()):
        shifted_recv_iface = torch.stack(all_moving_neighbor_masks(recv_iface))
        n_cnt = shifted_recv_iface.sum(dim=0)
        n_recv_iface = n_cnt.float().clamp(min=1.0)
        neg_per_nb = neg_donor / n_recv_iface
        neg_dist = torch.stack(
            [roll_to_neighbor(neg_per_nb, q) * recv_iface for q in D3Q19_MOVING_Q]
        ).sum(dim=0)
        # Only the part that actually has a surviving-INTERFACE receiver may be
        # self-credited; a receiver-less donor is left to the clamp.
        distributable = torch.where(n_cnt > 0, neg_donor, torch.zeros_like(neg_donor))
        neg_increment = -distributable + neg_dist
    else:
        neg_increment = torch.zeros_like(mass)

    # Aggregate every D3Q19 receiver contribution in the mass dtype, then
    # commit it once.  Sequential float32 rebinding rounds the same mass field
    # 18 times and leaves a transaction residual when conversion removes the
    # donor excess.  This preserves each link/mask contribution and topology;
    # only their deterministic same-dtype aggregation precedes one commit.
    legacy_redistribution_increment = (
        torch.stack(
            [roll_to_neighbor(excess_per_nb, q) * recv_mask for q in D3Q19_MOVING_Q]
        ).sum(dim=0)
        + neg_increment
    )
    # ABLATION (diagnostic only): TL_FS_ABL_REDIST suppresses the Körner
    # excess-mass redistribution entirely (conversion still clamps).
    if _env_bool("TL_FS_ABL_REDIST", False):
        legacy_redistribution_increment = torch.zeros_like(legacy_redistribution_increment)
    if _env_bool("TL_FS_DBG", False) and mass_ledger is not None:
        mass_ledger["dbg_redist_inc_sum"] = float(legacy_redistribution_increment.sum())
        mass_ledger["dbg_excess_sum"] = float(excess.sum())
        mass_ledger["dbg_excess_max"] = float(excess.max())
        mass_ledger["dbg_recv_sum"] = float(recv_mask.sum())
        mass_ledger["dbg_recv_new_sum"] = float(recv_new.sum())
        mass_ledger["dbg_to_liq_sum"] = float(to_liq.sum())
        mass_ledger["dbg_to_gas_sum"] = float(to_gas.sum())
        mass_ledger["dbg_to_iface_sum"] = float(to_iface.sum())
        mass_ledger["dbg_min_mass"] = float(mass.min())
        mass_ledger["dbg_max_mass"] = float(mass.max())
        mass_ledger["dbg_nrecv_zero"] = float((n_recv < 1.0).sum())
    redistribution_link_evidence = ()
    if runtime_ledger is not None or ownership_ledger is not None:
        links = []
        shape = mass.shape
        for q, shift in zip(D3Q19_MOVING_Q, moving_tensor_shifts()):
            dz, dy, dx = shift
            receiver_for_donor = roll_from_pull_source(recv_mask, q)
            for donor in torch.nonzero(
                (excess_per_nb != 0.0) & receiver_for_donor, as_tuple=False
            ).tolist():
                z, y, x = (int(value) for value in donor)
                links.append(
                    {
                        "donor": (z, y, x),
                        "receiver": ((z - dz) % shape[0], (y - dy) % shape[1], (x - dx) % shape[2]),
                        "shift": (dz, dy, dx),
                        "mass_delta": float(excess_per_nb[z, y, x]),
                        "event_id": "redistribution",
                        "operator": "redistribution",
                    }
                )
        redistribution_link_evidence = tuple(links)
    # Publish I→G debit/credit paths through the same topology evidence
    # channel; each receiver is a declared surviving INTERFACE owner.
    redistribution_link_evidence = redistribution_link_evidence + tuple(
        {**link, "mass_delta": float(link["credit"])}
        for link in (() if i_to_g_ownership is None else i_to_g_ownership.links)
    )
    plan = build_topology_transaction(
        f,
        fill,
        flags,
        mass,
        to_iface=to_iface,
        to_liq=to_liq,
        to_gas=to_gas,
        recv_new=recv_new,
        redistribution_increment=legacy_redistribution_increment,
        i_to_g_increment=None if i_to_g_ownership is None else i_to_g_ownership.receiver_increment,
        rho_liquid=rho_liquid,
        rho_gas=rho_gas,
        solid_mask=solid_mask,
        gas_flag=GAS,
        liquid_flag=LIQUID,
        interface_flag=INTERFACE,
        solid_flag=SOLID,
        ux=ux,
        uy=uy,
        uz=uz,
        capture_evidence=(
            runtime_ledger is not None
            or ownership_ledger is not None
            or conversion_density_audit_ledger is not None
        ),
        capture_inventory=(
            inventory_reconciliation_ledger is not None
            or conversion_density_audit_ledger is not None
        ),
        redistribution_link_evidence=redistribution_link_evidence,
        i_to_g_ownership=i_to_g_ownership,
        capture_replay_stages=capture_replay_stages,
    )
    f, fill, flags, mass = commit_topology_transaction(plan)
    if replay_capture is not None and plan.replay_evidence is not None:
        replay_capture["evidence"] = plan.replay_evidence
    if inventory_reconciliation_ledger is not None:
        assert inventory_stages is not None and plan.inventory_stages is not None
        _append_inventory_reconciliation(
            inventory_reconciliation_ledger,
            {
                **inventory_stages,
                **plan.inventory_stages,
            },
        )
    if conversion_density_audit_ledger is not None:
        from .free_surface_conversion_density_audit import build_conversion_density_audit

        conversion_density_audit_ledger["audit"] = build_conversion_density_audit(
            plan.conversion_evidence,
            rho_liquid=rho_liquid,
            observed_conversion_inventory_delta=(
                None
                if plan.inventory_stages is None
                else plan.inventory_stages["after_topology_conversion"]["total_liquid_inventory"]
                - plan.inventory_stages["after_topology_clamp"]["total_liquid_inventory"]
            ),
        )
        conversion_density_audit_ledger["status"] = "DIAGNOSTIC_WITHHELD_NOT_PHYSICAL_CLOSURE"
    if mass_ledger is not None:
        mass_ledger["redistribution"] = plan.mass_after_redistribution
        mass_ledger["clamp"] = plan.mass_after_clamp
        mass_ledger["conversion"] = plan.mass_after_conversion
        mass_ledger["isolation"] = plan.mass_after_isolation
        mass_ledger["boundary"] = float(mass.sum())
        mass_ledger["fill_mass_final"] = float((fill * rho_liquid).sum())
    if runtime_ledger is not None:
        _append_runtime_ledger(
            runtime_ledger,
            mass_start=mass_start_value,
            mass_after_exchange=mass_after_exchange_value,
            mass_after_redistribution=plan.mass_after_redistribution,
            mass_after_clamp=plan.mass_after_clamp,
            mass_after_conversion=plan.mass_after_conversion,
            mass_after_isolation=plan.mass_after_isolation,
            mass_end=float(mass.sum()),
            abb_population_delta=float(abb_delta.sum()),
            exchange_liquid_credit=float(mass_delta_liquid.sum()),
            exchange_interface_credit=float(mass_delta_interface.sum()),
            exchange_bulk_debit=float(
                mass_delta_bulk_debit.sum() if paired_liquid_interface_debit else 0.0
            ),
            paired_liquid_interface_debit=paired_liquid_interface_debit,
            conversion_evidence=plan.conversion_evidence,
        )
    if ownership_ledger is not None:
        assert ownership_flags is not None
        _append_ownership_ledger(
            ownership_ledger,
            flags=ownership_flags,
            mass_delta_liquid=mass_delta_liquid,
            liquid_interface_mask=from_liq,
            paired_liquid_interface_debit=paired_liquid_interface_debit,
            conversion_evidence=plan.conversion_evidence,
            abb_population_delta=float(abb_delta.sum()),
        )

    if published_mass_ledger is not None:
        published_mass_ledger.clear()
        published_mass_ledger.update(mass_ledger)
    if published_runtime_ledger is not None:
        published_runtime_ledger.clear()
        published_runtime_ledger.update(runtime_ledger)
    if published_ownership_ledger is not None:
        published_ownership_ledger.clear()
        published_ownership_ledger.update(ownership_ledger)
    if published_conversion_density_audit_ledger is not None:
        published_conversion_density_audit_ledger.clear()
        published_conversion_density_audit_ledger.update(conversion_density_audit_ledger)
    if published_inventory_reconciliation_ledger is not None:
        published_inventory_reconciliation_ledger.clear()
        published_inventory_reconciliation_ledger.update(inventory_reconciliation_ledger)
    return f, fill, flags, mass, df


def _redistribute_mass(mass, flags, mex, nx, ny, nz, c, device):
    """Redistribute excess mass to interface neighbors (vectorized)."""
    # Simple: distribute excess mass equally to interface neighbors
    interface_mask = flags == INTERFACE
    # Count interface neighbours over every moving D3Q19 link.
    shifted_iface = torch.stack(all_moving_neighbor_masks(interface_mask))
    n_iface_neighbors = shifted_iface.sum(dim=0).clamp(min=1)
    # Excess mass to distribute (per neighbor)
    excess_per_neighbor = mex / n_iface_neighbors
    # Aggregate all D3Q19 link increments before the one mass-field commit.
    redistribution_increment = torch.stack(
        [
            roll_to_neighbor(excess_per_neighbor, q) * roll_to_neighbor(interface_mask, q)
            for q in D3Q19_MOVING_Q
        ]
    ).sum(dim=0)
    mass = mass + redistribution_increment
    return mass
