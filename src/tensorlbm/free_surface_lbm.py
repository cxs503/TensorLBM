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

from copy import deepcopy

import torch

from .d3q19 import C, W, equilibrium3d, macroscopic3d
from .boundaries3d import bounce_back_cells_3d, free_slip_cells_3d
from .core.d3q19_stencil import (
    D3Q19_MOVING_Q,
    all_moving_neighbor_masks,
    assert_no_direct_phase_links,
    moving_tensor_shifts,
    roll_from_pull_source,
    roll_to_neighbor,
)
from .solver3d import stream3d as _stream3d
from .turbulence import _neq_stress_norm_3d, _smagorinsky_tau
from .free_surface_topology_transaction import (
    build_topology_transaction,
    build_i_to_g_ownership_transaction,
    commit_topology_transaction,
)
from .free_surface_inventory_reconciliation import (
    CANONICAL_STAGE_ORDER,
    inventory_measurement,
    inventory_stage_deltas,
)

GAS = 0; LIQUID = 1; INTERFACE = 2; SOLID = 3

# D3Q19 velocity vectors and weights
_C = C  # (19, 3)
_W = W
KAPPA = 0.41
B_CONST = 5.0
# Opposite direction indices for D3Q19
_OPP = torch.tensor([0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17])
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
        raise ValueError(
            f"{error}; insert INTERFACE cells between LIQUID and GAS"
        ) from None


def _append_runtime_ledger(ledger, *, mass_start, mass_after_exchange,
                           mass_after_redistribution, mass_after_clamp,
                           mass_after_conversion, mass_after_isolation, mass_end,
                           abb_population_delta, exchange_liquid_credit,
                           exchange_interface_credit, exchange_bulk_debit,
                           paired_liquid_interface_debit, conversion_evidence=None):
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
    ledger.setdefault("operator_curve", []).append({
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
    })
    ledger.update(record)


def _append_ownership_ledger(
    ledger, *, flags, mass_delta_liquid, liquid_interface_mask,
    paired_liquid_interface_debit, conversion_evidence, abb_population_delta,
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
    solid[:, 0, :] = True; solid[:, -1, :] = True
    solid[:, :, 0] = True; solid[:, :, -1] = True
    cw, ch = int(column_width), int(column_height)
    fill[:, :ch, 1:cw] = 1.0
    fx, fy = column_width - cw, column_height - ch
    if fx > 0 and cw < nx - 1: fill[:, :ch, cw] = fx
    if fy > 0 and ch < ny - 1: fill[:, ch, 1:cw] = fy
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
# Core timestep — full Körner model
# ===========================================================================

def free_surface_step(
    f, fill, flags, solid_mask, mass=None,
    tau=1.0, gx=0.0, gy=0.0, gz=0.0,
    rho_liquid=1.0, rho_gas=1.0,
    surface_tension=0.0, C_s=0.0,
    free_slip_y=False, y_wall_mask=None,
    bubble_pressure=None,
    collision='bgk',
    wall_function=False, near_mask=None, y_val=0.5,
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

    # Ledger output is transactional too: topology validation may fail after
    # ABB/exchange observations were calculated.  Accumulate into a detached
    # copy and publish it only on a successful return, so a failed step never
    # leaks partial diagnostic state to its caller.
    published_mass_ledger = mass_ledger
    mass_ledger = None if published_mass_ledger is None else deepcopy(published_mass_ledger)
    published_runtime_ledger = runtime_ledger
    runtime_ledger = None if published_runtime_ledger is None else deepcopy(published_runtime_ledger)
    published_ownership_ledger = ownership_ledger
    ownership_ledger = None if published_ownership_ledger is None else deepcopy(published_ownership_ledger)
    published_inventory_reconciliation_ledger = inventory_reconciliation_ledger
    inventory_reconciliation_ledger = (
        None if published_inventory_reconciliation_ledger is None
        else deepcopy(published_inventory_reconciliation_ledger)
    )
    published_conversion_density_audit_ledger = conversion_density_audit_ledger
    conversion_density_audit_ledger = (
        None if published_conversion_density_audit_ledger is None
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
        mass_ledger['start'] = float(mass.sum())
        mass_ledger['interface_start'] = float(mass[flags == INTERFACE].sum())
        mass_ledger['liquid_start'] = float(mass[flags == LIQUID].sum())
        mass_ledger['gas_start'] = float(mass[flags == GAS].sum())
        mass_ledger['fill_mass_start'] = float((fill * rho_liquid).sum())

    # ---- 1. Macroscopic + collision ----
    rho, ux, uy, uz = macroscopic3d(f)
    rho_s = rho.clamp(min=1e-6, max=rho_liquid * 3.0)
    ux_eq = (ux + tau * gx).clamp(-0.5, 0.5)
    uy_eq = (uy + tau * gy).clamp(-0.5, 0.5)
    uz_eq = (uz + tau * gz).clamp(-0.5, 0.5)
    feq = equilibrium3d(rho_s, ux_eq, uy_eq, uz_eq)

    # For advanced operators: set gas cells to small equilibrium (prevent NaN)
    if collision != 'bgk':
        feq_gas = equilibrium3d(torch.full_like(rho_s, rho_gas),
                                torch.zeros_like(rho_s), torch.zeros_like(rho_s), torch.zeros_like(rho_s))
        f_collide = torch.where(non_gas.unsqueeze(0), f, feq_gas)
    else:
        f_collide = f

    if collision == 'kbc':
        from .advanced_collision_d3q19 import collide_kbc_d3q19
        f = collide_kbc_d3q19(f_collide, tau, C_s=C_s if C_s > 0 else 0.1)
    elif collision == 'cascaded':
        from .advanced_collision_d3q19 import collide_cascaded_d3q19
        f = collide_cascaded_d3q19(f_collide, tau, C_s=C_s if C_s > 0 else 0.1)
    elif collision == 'cumulant':
        from .advanced_collision_d3q19 import collide_cumulant_d3q19
        f = collide_cumulant_d3q19(f_collide, tau, C_s=C_s if C_s > 0 else 0.1)
    elif collision == 'mrt':
        from .solver3d import collide_mrt3d
        f = collide_mrt3d(f_collide, tau)
    else:  # bgk
        if C_s > 0:
            tau_eff = _smagorinsky_tau(tau, _neq_stress_norm_3d(f_collide - feq), rho_s, C_s)
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
        f = f + (1.0 - 0.5/tau) * w_dev * cu_force / cs2

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
        nx, ny, nz = -grad_x/mag, -grad_y/mag, -grad_z/mag
        kappa = 0.5 * ((nx.roll(-1, dims=2) - nx.roll(1, dims=2)) +
                       (ny.roll(-1, dims=1) - ny.roll(1, dims=1)) +
                       (nz.roll(-1, dims=0) - nz.roll(1, dims=0)))
        Fx_st = surface_tension * kappa * grad_x
        Fy_st = surface_tension * kappa * grad_y
        Fz_st = surface_tension * kappa * grad_z
        cu_st = cx * Fx_st.unsqueeze(0) + cy * Fy_st.unsqueeze(0) + cz * Fz_st.unsqueeze(0)
        f = f + (1.0 - 0.5/tau) * w_dev * cu_st / cs2

    # Clamp f to non-negative (numerical stability, prevents negative rho → NaN)
    f = f.clamp(min=0.0, max=rho_liquid * 3.0)

    # Remove NaN for non-BGK
    if collision != 'bgk':
        f = torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    if inventory_stages is not None:
        inventory_stages["after_collision_and_forcing"] = inventory_measurement(
            f, fill, flags, mass, rho_liquid=rho_liquid,
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
    gas_mask_pre = (flags == GAS)
    f = torch.where(gas_mask_pre.unsqueeze(0), torch.zeros_like(f), f)
    if inventory_stages is not None:
        inventory_stages["after_stream_and_gas_zero"] = inventory_measurement(
            f, fill, flags, mass, rho_liquid=rho_liquid,
        )

    # ---- 2c. Anti-bounce-back for interface cells (gas pressure) ----
    # Standard Körner: interface cells get f[q] = f[opp[q]] from gas directions
    # Vectorized: batch all 19 directions, reuse neighbor_flags in mass exchange
    # (no .any() sync — multicard-safe under TCCL; torch.where handles empty mask)
    iface_abb = (flags == INTERFACE)
    # _stream19_roll is a pull stream: population q at x originated at
    # x-c[q].  Interface link classification must use that source cell.
    neighbor_flags = torch.stack([
        flags.roll(sz, dims=0).roll(sy, dims=1).roll(sx, dims=2)
        for sx, sy, sz in _C19_SHIFTS
    ])  # (19, nz, ny, nx)
    need_abb = iface_abb.unsqueeze(0) & (neighbor_flags == GAS)
    # Standard Körner ABB (pressure boundary) for a missing pull population:
    # f_q(x,t+dt) = f_q^eq(rho_g,u_g) + f_barq^eq(rho_g,u_g)
    #                 - f_barq^*(x,t).  With u_g=u_interface this fixes p_g.
    # The implementation has no separate gas velocity field, so use the local
    # interface velocity from the pre-collision macroscopics.
    rho_g_field = torch.full_like(rho, float(rho_gas))
    f_eq_gas = equilibrium3d(rho_g_field, ux, uy, uz)
    f_abb = f_eq_gas + f_eq_gas[_OPP.to(device)] - f_post[_OPP.to(device)]
    abb_delta = torch.where(need_abb, f_abb - f, torch.zeros_like(f))
    if mass_ledger is not None:
        # This is a population (not tracked-liquid-mass) change.  Keeping it
        # separate makes a gas-pressure boundary source distinguishable from
        # the subsequent liquid/interface mass stencil.
        mass_ledger['abb_population_delta'] = float(abb_delta.sum())
        mass_ledger['abb_population_abs_delta'] = float(abb_delta.abs().sum())
    f = torch.where(need_abb, f_abb, f)
    if inventory_stages is not None:
        inventory_stages["after_abb"] = inventory_measurement(f, fill, flags, mass, rho_liquid=rho_liquid)

    # ---- 3. Wall BCs ----
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
        ut = u_tau.clone(); um = u_mag
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
        wf_coef = wf_force_coef if wf_force_coef is not None else (1.0 - 0.5/tau)
        forcing = wf_coef * w_dev * cu_force / cs2
        f = f + forcing
        f = f.clamp(min=0.0, max=rho_liquid * 3.0)  # prevent inf from wall function forcing
        df = (tau_w * near_mask.to(f.dtype)).sum()
    if inventory_stages is not None:
        inventory_stages["after_wall_boundary"] = inventory_measurement(
            f, fill, flags, mass, rho_liquid=rho_liquid,
        )

    # ---- 4. Mass exchange (standard Körner, independent mass variable) ----
    # (no .any() sync — multicard-safe under TCCL; torch.where handles empty masks)
    rho_new = f.sum(dim=0)
    iface_mask = (flags == INTERFACE)
    # neighbor_flags always computed in anti-bounce-back above (no None check)
    # For pull link q at x, the opposing outgoing population belongs to x
    # itself: f_bar(q)^*(x).  Sampling it at x-c_q mixes two different links.
    f_opp_nb = f_post[_OPP.to(device)]  # (19, nz, ny, nx)
    iface_19 = iface_mask.unsqueeze(0)
    from_liq = iface_19 & (neighbor_flags == LIQUID)
    from_gas = iface_19 & (neighbor_flags == GAS)
    from_iface = iface_19 & (neighbor_flags == INTERFACE)
    mass_delta_liquid = torch.where(from_liq, f - f_opp_nb, torch.zeros_like(f))
    mass_delta_interface = torch.where(
        from_iface, (f - f_opp_nb) * 0.5, torch.zeros_like(f)
    )
    # A L/I credit at interface target x is paired link-by-link with a debit
    # at its pull source x-c_q.  This uses only existing D3Q19 links; it is
    # neither a global rescale nor a topology mutation.
    mass_delta_bulk_debit = -torch.stack([
        mass_delta_liquid[q].roll((-sz, -sy, -sx), dims=(0, 1, 2))
        for q, (sx, sy, sz) in enumerate(_C19_SHIFTS)
    ]).sum(0)
    mass_delta = (
        mass_delta_liquid +
        # Gas is a pressure boundary, not a liquid-mass reservoir.  Adding
        # its reconstructed population here spuriously creates tracked liquid
        # mass in a quiescent closed column.
        torch.zeros_like(f) + mass_delta_interface
    ).sum(0)
    if paired_liquid_interface_debit:
        valid_bulk_owner = flags == LIQUID
        invalid_debit = mass_delta_bulk_debit.masked_select(~valid_bulk_owner)
        if bool((invalid_debit.abs() > 1.0e-8).any()):
            raise RuntimeError("L/I paired debit has no LIQUID bulk owner")
        mass_delta = mass_delta + torch.where(
            valid_bulk_owner, mass_delta_bulk_debit, torch.zeros_like(mass_delta_bulk_debit)
        )
    mass = torch.where(~solid_mask, mass + mass_delta, mass)
    mass_after_exchange_value = float(mass.sum())
    fill = torch.where(~solid_mask, (mass / rho_liquid).clamp(0.0, 1.0), fill)
    if inventory_stages is not None:
        inventory_stages["after_mass_exchange"] = inventory_measurement(
            f, fill, flags, mass, rho_liquid=rho_liquid,
        )
    if mass_ledger is not None:
        mass_ledger['exchange'] = float(mass.sum())
        mass_ledger['exchange_liquid_delta'] = float(mass_delta_liquid.sum())
        mass_ledger['exchange_interface_delta'] = float(mass_delta_interface.sum())
        mass_ledger['exchange_bulk_debit'] = float(
            mass_delta_bulk_debit.sum() if paired_liquid_interface_debit else 0.0
        )
        mass_ledger['exchange_gas_delta'] = 0.0
        mass_ledger['fill_mass_after_exchange'] = float((fill * rho_liquid).sum())

    # Diagnostic mode: retain the post-stream/post-ABB populations and the
    # exchange result, but deliberately forbid flag conversion, redistribution
    # and halo propagation.  This isolates one mixed L/I/G topology update.
    if freeze_topology:
        if mass_ledger is not None:
            frozen_total = float(mass.sum())
            mass_ledger['redistribution'] = frozen_total
            mass_ledger['clamp'] = frozen_total
            mass_ledger['conversion'] = frozen_total
            mass_ledger['isolation'] = frozen_total
            mass_ledger['boundary'] = frozen_total
            mass_ledger['fill_mass_final'] = float((fill * rho_liquid).sum())
        if runtime_ledger is not None:
            _append_runtime_ledger(
                runtime_ledger, mass_start=mass_start_value,
                mass_after_exchange=mass_after_exchange_value,
                mass_after_redistribution=mass_after_exchange_value,
                mass_after_clamp=mass_after_exchange_value,
                mass_after_conversion=mass_after_exchange_value,
                mass_after_isolation=mass_after_exchange_value,
                mass_end=float(mass.sum()), abb_population_delta=float(abb_delta.sum()),
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
                ownership_ledger, flags=ownership_flags,
                mass_delta_liquid=mass_delta_liquid,
                liquid_interface_mask=from_liq,
                paired_liquid_interface_debit=paired_liquid_interface_debit,
                conversion_evidence=conversion_evidence,
                abb_population_delta=float(abb_delta.sum()),
            )
        if inventory_reconciliation_ledger is not None:
            assert inventory_stages is not None
            after_mass_exchange = inventory_stages["after_mass_exchange"]
            _append_inventory_reconciliation(inventory_reconciliation_ledger, {
                **inventory_stages,
                "after_topology_redistribution": after_mass_exchange,
                "after_topology_clamp": after_mass_exchange,
                "after_topology_conversion": after_mass_exchange,
                "after_topology_halo_isolation_boundary": after_mass_exchange,
            })
        if conversion_density_audit_ledger is not None:
            from .free_surface_conversion_density_audit import build_conversion_density_audit
            conversion_density_audit_ledger["audit"] = build_conversion_density_audit(None, rho_liquid=rho_liquid)
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
    gas_mask = (flags == GAS)
    interface_mask = (flags == INTERFACE)
    liquid_mask = (flags == LIQUID)

    # Gas → Interface (received mass from streaming)
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
    redistribution_to_g = to_gas if not enable_i_to_g_ownership_closure else (to_gas & ~interface_mask)
    excess = (
        torch.where(to_liq, mass - rho_liquid, torch.zeros_like(mass))
        + torch.where(redistribution_to_g, mass, torch.zeros_like(mass))
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
    recv_mask = recv_iface | recv_new
    i_to_g_ownership = None
    if enable_i_to_g_ownership_closure and bool(i_to_g.any()):
        i_to_g_ownership = build_i_to_g_ownership_transaction(
            flags, mass, to_gas=i_to_g, to_liq=to_liq, solid_mask=solid_mask,
            gas_flag=GAS, liquid_flag=LIQUID, interface_flag=INTERFACE,
            rho_liquid=rho_liquid,
        )
    # Count receiving cells per donor over every moving D3Q19 link.
    shifted_recv = torch.stack(all_moving_neighbor_masks(recv_mask))
    n_recv = shifted_recv.sum(dim=0).float().clamp(min=1.0)
    # Excess per receiving neighbor
    excess_per_nb = excess / n_recv

    # Aggregate every D3Q19 receiver contribution in the mass dtype, then
    # commit it once.  Sequential float32 rebinding rounds the same mass field
    # 18 times and leaves a transaction residual when conversion removes the
    # donor excess.  This preserves each link/mask contribution and topology;
    # only their deterministic same-dtype aggregation precedes one commit.
    legacy_redistribution_increment = torch.stack([
        roll_to_neighbor(excess_per_nb, q) * recv_mask for q in D3Q19_MOVING_Q
    ]).sum(dim=0)
    redistribution_link_evidence = ()
    if runtime_ledger is not None or ownership_ledger is not None:
        links = []
        shape = mass.shape
        for q, shift in zip(D3Q19_MOVING_Q, moving_tensor_shifts()):
            dz, dy, dx = shift
            receiver_for_donor = roll_from_pull_source(recv_mask, q)
            for donor in torch.nonzero((excess_per_nb != 0.0) & receiver_for_donor, as_tuple=False).tolist():
                z, y, x = (int(value) for value in donor)
                links.append({"donor": (z, y, x), "receiver": ((z - dz) % shape[0], (y - dy) % shape[1], (x - dx) % shape[2]), "shift": (dz, dy, dx), "mass_delta": float(excess_per_nb[z, y, x]), "event_id": "redistribution", "operator": "redistribution"})
        redistribution_link_evidence = tuple(links)
    # Publish I→G debit/credit paths through the same topology evidence
    # channel; each receiver is a declared surviving INTERFACE owner.
    redistribution_link_evidence = redistribution_link_evidence + tuple(
        {**link, "mass_delta": float(link["credit"])}
        for link in (() if i_to_g_ownership is None else i_to_g_ownership.links)
    )
    plan = build_topology_transaction(
        f, fill, flags, mass, to_iface=to_iface, to_liq=to_liq, to_gas=to_gas,
        recv_new=recv_new, redistribution_increment=legacy_redistribution_increment,
        i_to_g_increment=None if i_to_g_ownership is None else i_to_g_ownership.receiver_increment,
        rho_liquid=rho_liquid, rho_gas=rho_gas, solid_mask=solid_mask,
        gas_flag=GAS, liquid_flag=LIQUID, interface_flag=INTERFACE, solid_flag=SOLID,
        ux=ux, uy=uy, uz=uz,
        capture_evidence=(
            runtime_ledger is not None or ownership_ledger is not None
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
        _append_inventory_reconciliation(inventory_reconciliation_ledger, {
            **inventory_stages,
            **plan.inventory_stages,
        })
    if conversion_density_audit_ledger is not None:
        from .free_surface_conversion_density_audit import build_conversion_density_audit
        conversion_density_audit_ledger["audit"] = build_conversion_density_audit(
            plan.conversion_evidence, rho_liquid=rho_liquid,
            observed_conversion_inventory_delta=(
                None if plan.inventory_stages is None
                else plan.inventory_stages["after_topology_conversion"]["total_liquid_inventory"]
                - plan.inventory_stages["after_topology_clamp"]["total_liquid_inventory"]
            ),
        )
        conversion_density_audit_ledger["status"] = "DIAGNOSTIC_WITHHELD_NOT_PHYSICAL_CLOSURE"
    if mass_ledger is not None:
        mass_ledger['redistribution'] = plan.mass_after_redistribution
        mass_ledger['clamp'] = plan.mass_after_clamp
        mass_ledger['conversion'] = plan.mass_after_conversion
        mass_ledger['isolation'] = plan.mass_after_isolation
        mass_ledger['boundary'] = float(mass.sum())
        mass_ledger['fill_mass_final'] = float((fill * rho_liquid).sum())
    if runtime_ledger is not None:
        _append_runtime_ledger(
            runtime_ledger, mass_start=mass_start_value,
            mass_after_exchange=mass_after_exchange_value,
            mass_after_redistribution=plan.mass_after_redistribution,
            mass_after_clamp=plan.mass_after_clamp,
            mass_after_conversion=plan.mass_after_conversion,
            mass_after_isolation=plan.mass_after_isolation,
            mass_end=float(mass.sum()), abb_population_delta=float(abb_delta.sum()),
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
            ownership_ledger, flags=ownership_flags,
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
    interface_mask = (flags == INTERFACE)
    # Count interface neighbours over every moving D3Q19 link.
    shifted_iface = torch.stack(all_moving_neighbor_masks(interface_mask))
    n_iface_neighbors = shifted_iface.sum(dim=0).clamp(min=1)
    # Excess mass to distribute (per neighbor)
    excess_per_neighbor = mex / n_iface_neighbors
    # Aggregate all D3Q19 link increments before the one mass-field commit.
    redistribution_increment = torch.stack([
        roll_to_neighbor(excess_per_neighbor, q) * roll_to_neighbor(interface_mask, q)
        for q in D3Q19_MOVING_Q
    ]).sum(dim=0)
    mass = mass + redistribution_increment
    return mass
