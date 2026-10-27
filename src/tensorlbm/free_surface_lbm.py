"""Free-surface LBM (D3Q19) — full Körner model with mass tracking.

Implements the complete Körner et al. (2005) free-surface LBM with the
reference (waLBerla ``free_surface`` / Thürey 2007) mechanics:
  - Cell types: GAS / INTERFACE / LIQUID / SOLID
  - Independent tracked mass (mass ≠ rho); fill = mass / rho_local
  - Single Guo body force, fill-weighted at INTERFACE (Körner eq. 15)
  - ABB gas-pressure reconstruction with Laplace/bubble pressure
  - Fluid-side half-way walls with selective free-slip
  - Körner mass exchange (eq. 9) with the Thürey table-4.1 I/I selector
  - Interface-only conversion (1+κ/-κ + forced rules), even excess
    redistribution, conversion-driven envelope (no clamp, no halo sweep)

References: Körner et al. (2005), Thürey (2007), waLBerla free_surface/,
Maarten-vd-Sande/lbm
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
_OPP_INDEX = tuple(int(v) for v in _OPP.tolist())


def _specular_wall_mirror_index(solid_mask, device):
    """Per-link specular (free-slip) reflected source indices for wall cells.

    OPT-IN wall-stagnation fix (``TL_FS_WALL_SLIP=1``).

    ``halfway`` no-slip bounce-back imposes u = 0 at the wall, so at the low
    Reynolds numbers reachable on a small ``a`` (nu = 1/3*(tau-0.5) is O(0.1)
    while the column is only ``a`` cells wide) the no-slip boundary layer spans
    the entire column.  The wall-adjacent *interface* column then carries
    u_y ~ 3-5x smaller than the interior, drains 3-5x slower, never empties at
    the free surface and pins the measured residual height H (the "back-wall
    corner column" freeze).  Free-slip (specular) walls remove only the
    *wall-normal* momentum while preserving the tangential component, so the
    wall column falls with the interior.

    For every D3Q19 link ``q`` at a fluid cell whose pull source ``p - c_q``
    lies in SOLID, the reflected link reverses each coordinate component of
    ``c_q`` for which the corresponding signed axis neighbour is SOLID
    (a face wall reverses one component -> specular; an edge/corner reverses
    two/three -> full bounce).  Returns an ``(19, nz, ny, nx)`` int64 tensor of
    source link indices to gather from ``f_post``.
    """
    smx = solid_mask.roll(1, dims=2)  # SOLID at p-(1,0,0)
    spx = solid_mask.roll(-1, dims=2)  # SOLID at p+(1,0,0)
    smy = solid_mask.roll(1, dims=1)
    spy = solid_mask.roll(-1, dims=1)
    smz = solid_mask.roll(1, dims=0)
    spz = solid_mask.roll(-1, dims=0)
    off2idx = {(sx, sy, sz): q for q, (sx, sy, sz) in enumerate(_C19_SHIFTS)}
    table = torch.zeros((19, 2, 2, 2), dtype=torch.long)
    for q, (cx, cy, cz) in enumerate(_C19_SHIFTS):
        for fx in (0, 1):
            for fy in (0, 1):
                for fz in (0, 1):
                    table[q, fx, fy, fz] = off2idx[
                        (-cx if fx else cx, -cy if fy else cy, -cz if fz else cz)
                    ]
    table = table.to(device)
    shape = (19,) + tuple(solid_mask.shape)
    idx = torch.empty(shape, dtype=torch.long, device=device)
    for q, (cx, cy, cz) in enumerate(_C19_SHIFTS):
        sx, sy, sz = _C19_SHIFTS[q]
        src_solid = solid_mask.roll(sz, dims=0).roll(sy, dims=1).roll(sx, dims=2)
        fx = ((cx > 0) & smx) | ((cx < 0) & spx)
        fy = ((cy > 0) & smy) | ((cy < 0) & spy)
        fz = ((cz > 0) & smz) | ((cz < 0) & spz)
        refl = table[q, fx.long(), fy.long(), fz.long()]
        # Safety fallback: a solid source detected with no flagged axis (not
        # reachable for axis-aligned walls) keeps the plain opposite link.
        refl = torch.where(src_solid & ~(fx | fy | fz), torch.full_like(refl, _OPP_INDEX[q]), refl)
        idx[q] = refl
    return idx


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
    paired_net = exchange_liquid_credit + exchange_interface_credit + exchange_bulk_debit
    # ``paired_liquid_interface_debit`` is a deprecated no-op switch: the
    # legacy paired bulk debit no longer exists (LIQUID fill is pinned at 1).
    # The Körner exchange is attributed in full instead.
    explained = (
        exchange_liquid_credit
        + exchange_interface_credit
        + redistribution
        + clamp
        + conversion
        + isolation
        + boundary
    )
    unexplained = mass_drift - explained
    # Float32 summation rounding on an O(10) tracked total leaves an absolute
    # residual of a few e-6 even when every operator is attributed in full,
    # so the conserved/diagnostic boundary is relative to the step total.
    _conserved_tol = 1.0e-6 * max(1.0, abs(float(mass_start)))
    _ledger_balances = abs(float(unexplained)) <= _conserved_tol
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
        "closed_domain_conserved": _ledger_balances,
        "diagnostic": (
            "tracked-mass ledger balances; not a physical/PV closure claim"
            if _ledger_balances
            else "one-sided liquid/interface mass credit or unpaired interface/interface credit"
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


def _read_shell_mode(rho_gas: float) -> tuple[str, float]:
    """(mode, rho_env) for the initial empty-shell density.

    ``TL_FS_SHELL_MODE`` (falling back to ``TL_FS_BIRTH_MODE``) selects how the
    *initial* mass-free INTERFACE envelope (fill == 0) is seeded:

      ``legacy``  rho_liquid (historical, over-pressurised gas side)
      ``gas``     rho_gas
      ``fill``    rho_gas + (rho_liquid - rho_gas) * fill  (Körner closure)
      ``env``     TL_FS_BIRTH_RHO
      ``zero``    0
      ``liqface`` directional: liquid-facing links at rho_liquid,
                  gas-facing links at TL_FS_BIRTH_RHO

    ``fill``/``gas`` make an empty shell gas-consistent, matching the ABB gas
    pressure boundary and ``TL_FS_BIRTH_MODE`` birth semantics.
    """
    mode = os.environ.get("TL_FS_SHELL_MODE")
    if mode is None:
        mode = os.environ.get("TL_FS_BIRTH_MODE", "legacy")
    mode = mode.strip().lower()
    rho_env = float(os.environ.get("TL_FS_BIRTH_RHO", "0.0"))
    return mode, rho_env


def init_population_from_fill(
    fill,
    flags,
    rho_liquid=1.0,
    rho_gas=1.0,
    mode=None,
    rho_env=None,
):
    """Initial D3Q19 populations for a free-surface state, shell-density aware.

    LIQUID cells start at ``equilibrium(rho_liquid)``.  INTERFACE cells use a
    density that is *gas-consistent*:

    * a partially filled INTERFACE cell (0 < fill < 1) carries the Körner
      pressure-closure density ``rho_gas + (rho_liquid - rho_gas) * fill``;
    * a mass-free envelope cell (fill == 0) would therefore carry ``rho_gas``.

    The historical ``build_domain`` instead seeded *every* active cell
    (LIQUID *and* the mass=0 envelope) at ``rho_liquid``.  That over-pressures
    the gas-facing links of the envelope: the ABB reconstruction and the
    gas-vent exchange both read it, so a mass-free shell starts with a
    spurious liquid-scale population and siphons real liquid (the large
    ``TL_FS_GAS_CHANNEL=naive/redist`` drift and part of the pseudo-film).

    ``mode``/``rho_env`` default to ``_read_shell_mode(rho_gas)`` (env-gated;
    ``legacy`` reproduces the historical initial state bit-for-bit).
    """
    if mode is None or rho_env is None:
        m_env, r_env = _read_shell_mode(rho_gas)
        mode = mode if mode is not None else m_env
        rho_env = rho_env if rho_env is not None else r_env
    mode = str(mode).strip().lower()
    device = fill.device
    shape = tuple(fill.shape)
    zero = torch.zeros(shape, device=device, dtype=fill.dtype)
    liquid = flags == LIQUID
    iface = flags == INTERFACE
    if mode in ("legacy", "liq", "off", "0", ""):
        rho = torch.where(liquid | iface, torch.full_like(fill, float(rho_liquid)), zero)
        return equilibrium3d(rho, zero, zero, zero)
    if mode == "liqface":
        active = liquid | iface
        nb_liq = torch.stack(all_moving_neighbor_masks(liquid)).any(dim=0)
        feq_liq = equilibrium3d(
            torch.where(active, torch.full_like(fill, float(rho_liquid)), zero),
            zero,
            zero,
            zero,
        )
        feq_env = equilibrium3d(
            torch.where(active, torch.full_like(fill, float(rho_env)), zero),
            zero,
            zero,
            zero,
        )
        return torch.where((iface & nb_liq).unsqueeze(0), feq_liq, feq_env)
    if mode == "zero":
        rho_i = zero
    elif mode == "env":
        rho_i = torch.full_like(fill, float(rho_env))
    elif mode == "gas":
        rho_i = torch.full_like(fill, float(rho_gas))
    else:  # "fill" (and any unrecognised mode) -> Körner fill-weighted closure
        rho_i = float(rho_gas) + (float(rho_liquid) - float(rho_gas)) * fill
    rho = torch.where(liquid, torch.full_like(fill, float(rho_liquid)), zero)
    rho = torch.where(iface, rho_i, rho)
    return equilibrium3d(rho, zero, zero, zero)


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
    cell_conversion_threshold=1e-2,
    cell_conversion_force_threshold=0.1,
    free_slip_mask=None,
    free_slip_all=False,
):
    """One free-surface LBM timestep (full Körner model).

    Reference mechanics (Körner et al. 2005; Thürey 2007; waLBerla
    ``free_surface``; Maarten-vd-Sande/lbm):

      1. BGK/MRT/KBC collision (skipped in SOLID/GAS) + single Guo body force
         (fill-weighted at INTERFACE, Körner eq. 15)
      2. Pull streaming + gas zeroing
      2c. ABB gas-pressure reconstruction (Thürey eq. 4.5) with Laplace
          pressure (``surface_tension``) and ``bubble_pressure`` support
      3. Fluid-side half-way wall bounce-back, with selective specular
         (free-slip) reflection on ``free_slip_mask`` / y-walls / all walls
      4. Körner mass exchange (eq. 9): full-weight L/I links, fill-averaged
         I/I links with the Thürey table-4.1 selector, no gas flux
      5. Conversion marking: only INTERFACE converts (fill > 1+κ / < -κ plus
         forced neighbourhood rules)
      5a. Even excess redistribution to surviving + conversion-driven new
          receivers (local donor density)
      5b. Topology transaction: conversion (I→L keeps PDFs), conversion-driven
          envelope + LIQUID demotion, no clamp, no isolation stage

    ``paired_liquid_interface_debit`` and ``enable_i_to_g_ownership_closure``
    are deprecated no-ops retained for call compatibility.
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
    # Pre-collision cell-type masks.  The Körner body force (eq. 15) acts on
    # LIQUID at full strength and on INTERFACE weighted by the fill fraction;
    # GAS and SOLID cells carry no liquid and are excluded from both the force
    # and the collision below (waLBerla ``ForceDensitySweep`` /
    # ``free_surface::InterfaceCell`` dynamics).
    liquid_pre = flags == LIQUID
    iface_pre = flags == INTERFACE
    active_pre = liquid_pre | iface_pre
    force_weight = torch.where(
        iface_pre, fill.clamp(0.0, 1.0), torch.ones_like(fill)
    )
    force_weight = torch.where(active_pre, force_weight, torch.zeros_like(force_weight))
    # Second-order Guo forcing shifts the equilibrium velocity by F*dt/2rho
    # (Guo et al. 2002).  The legacy ``tau * g`` shift double-counted gravity
    # together with the explicit Guo source below (measured gain 1.5x/1.375x
    # instead of 1.0x); a single Guo force with the half-step shift gives an
    # exact momentum gain of F per step.
    ux_eq = (ux + 0.5 * force_weight * gx).clamp(-0.5, 0.5)
    uy_eq = (uy + 0.5 * force_weight * gy).clamp(-0.5, 0.5)
    uz_eq = (uz + 0.5 * force_weight * gz).clamp(-0.5, 0.5)
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

    # Guo body force (only when gravity is non-zero).  Single mechanism with the
    # full second-order Guo source term; the legacy truncated ``(c.F)`` form is
    # only first-order accurate.  Körner eq. 15: F = a * rho * fill at
    # INTERFACE, F = a * rho at LIQUID, zero elsewhere.
    if gx != 0.0 or gy != 0.0 or gz != 0.0:
        cs2 = 1.0 / 3.0
        cx = c_dev[:, 0].view(19, 1, 1, 1)
        cy = c_dev[:, 1].view(19, 1, 1, 1)
        cz = c_dev[:, 2].view(19, 1, 1, 1)
        w_dev = _W.to(device).float().view(19, 1, 1, 1)
        Fx = rho_s * gx * force_weight
        Fy = rho_s * gy * force_weight
        Fz = rho_s * gz * force_weight
        cu = (cx * ux + cy * uy + cz * uz) / cs2
        forcing = (
            ((cx - ux) * Fx + (cy - uy) * Fy + (cz - uz) * Fz) / cs2
            + cu * (cx * Fx + cy * Fy + cz * Fz) / cs2
        )
        f = f + (1.0 - 0.5 / tau) * w_dev * forcing

    # NOTE: surface tension is intentionally *not* a CSF body force here.
    # The reference free-surface closure (Thürey 2007 §4.2, waLBerla
    # ``GetLaplacePressure.h``) carries surface tension as a Laplace pressure
    # jump inside the ABB gas density; see section 2c below.  The legacy
    # ``surface_tension * kappa * grad`` body force is removed; the
    # ``surface_tension`` argument now feeds the Laplace term.

    # Clamp f to non-negative (numerical stability, prevents negative rho → NaN)
    f = f.clamp(min=0.0, max=rho_liquid * 3.0)

    # Remove NaN for non-BGK
    if collision != "bgk":
        f = torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)

    # Collision is skipped in SOLID and GAS cells (waLBerla applies the LBM
    # dynamics only to liquid/interface cells; gas has no populations).  The
    # legacy path collided everywhere, letting the wall/gas state feed back
    # into the fluid through the next streaming step.
    f = torch.where((solid_mask | ~active_pre).unsqueeze(0), f_collide, f)
    f = torch.where(
        (~active_pre & ~solid_mask).unsqueeze(0),
        torch.zeros_like(f),
        f,
    )
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
    # The gas velocity is the local *post-collision* interface velocity: at
    # stream time the PDF field holds the post-collision state (waLBerla
    # ``ReconstructInterfaceCellABB`` reads the post-collision PDFs).
    _opp_dev = _OPP.to(device)
    _rho_pc, _ux_pc, _uy_pc, _uz_pc = macroscopic3d(f_post)
    if _env_bool("TL_FS_INPUT_GUARD", True):
        _ux_pc = _ux_pc.clamp(-0.5, 0.5)
        _uy_pc = _uy_pc.clamp(-0.5, 0.5)
        _uz_pc = _uz_pc.clamp(-0.5, 0.5)
    # Gas reference density: the bubble pressure overrides rho_gas when given
    # (enclosed, possibly pressurised gas volume; waLBerla ``BubbleModel``).
    _rho_gas_base = float(bubble_pressure) if bubble_pressure is not None else float(rho_gas)
    rho_g_field = torch.full_like(rho, _rho_gas_base)
    if surface_tension > 0:
        # Laplace pressure jump inside the ABB gas density (Thürey 2007 §4.2,
        # waLBerla ``computeDeltaRhoLaplacePressure``): Δρ = Δp/cs² with
        # Δp = 2σH and the total curvature κ = 2H = div(n) computed below, so
        # Δρ = 3σκ, limited to ±0.1 as in the reference implementation.
        fill_field = fill.clamp(0.0, 1.0)
        grad_x = 0.5 * (fill_field.roll(-1, dims=2) - fill_field.roll(1, dims=2))
        grad_y = 0.5 * (fill_field.roll(-1, dims=1) - fill_field.roll(1, dims=1))
        grad_z = 0.5 * (fill_field.roll(-1, dims=0) - fill_field.roll(1, dims=0))
        mag = (grad_x**2 + grad_y**2 + grad_z**2).sqrt().clamp(min=1e-10)
        nx, ny_n, nz_n = -grad_x / mag, -grad_y / mag, -grad_z / mag
        kappa = 0.5 * (
            (nx.roll(-1, dims=2) - nx.roll(1, dims=2))
            + (ny_n.roll(-1, dims=1) - ny_n.roll(1, dims=1))
            + (nz_n.roll(-1, dims=0) - nz_n.roll(1, dims=0))
        )
        delta_rho = (3.0 * float(surface_tension) * kappa).clamp(-0.1, 0.1)
        rho_g_field = rho_g_field + torch.where(
            iface_abb, delta_rho, torch.zeros_like(delta_rho)
        )
    f_eq_gas = equilibrium3d(rho_g_field, _ux_pc, _uy_pc, _uz_pc)
    # The reference ABB subtracts the *local post-collision* outgoing
    # population f_post[opp] (magnitude O(rho_liquid)).  This is consistent
    # only when the gas reference density is O(rho_liquid): for an open free
    # surface the correct choice is rho_gas = rho_liquid (pressure-matched
    # ambient, waLBerla ``BubbleModelConstantPressure``/Pohl 2008).  A small
    # rho_gas (e.g. a real air density) makes every reconstructed population
    # negative and the interface collapses -- that is why the legacy
    # TL_FS_ABB_MODE variants existed; they are removed with the fix.
    f_abb = f_eq_gas + f_eq_gas[_opp_dev] - f_post[_opp_dev]
    abb_delta = torch.where(need_abb, f_abb - f, torch.zeros_like(f))
    if mass_ledger is not None:
        # This is a population (not tracked-liquid-mass) change.  Keeping it
        # separate makes a gas-pressure boundary source distinguishable from
        # the subsequent liquid/interface mass stencil.
        mass_ledger["abb_population_delta"] = float(abb_delta.sum())
        mass_ledger["abb_population_abs_delta"] = float(abb_delta.abs().sum())
    f = torch.where(need_abb, f_abb, f)
    if inventory_stages is not None:
        inventory_stages["after_abb"] = inventory_measurement(
            f, fill, flags, mass, rho_liquid=rho_liquid
        )

    # ---- 3. Wall BCs ----
    # Fluid-side half-way bounce-back (default; the reference-correct form).
    #
    # The legacy ``bounce_back_cells_3d`` reflects at the SOLID cell via an
    # in-place q <-> opp(q) swap.  In this pull-stream framework the *fluid*
    # cell adjacent to a wall reads its wall-entering populations from that
    # solid cell BEFORE the swap, i.e. it samples the post-collision solid
    # state (and, on step 1, the zero initial solid state).  The one-step
    # budget shows the resulting spurious wall-layer deficit: rho 1.0 -> 0.81
    # and a u spike of -0.20 EVEN AT g = 0 (gravity-independent).
    #
    # The corrected form reflects at the *fluid* side using its own outgoing
    # populations (true half-way BB): f_q(x_f) = f_post[opp(q)](x_f).
    # ``TL_FS_WALL_MODE=legacy`` restores the old post-stream solid-swap-only
    # path for A/B comparison.
    _wall_mode = os.environ.get("TL_FS_WALL_MODE", "halfway").strip().lower()
    if _wall_mode != "legacy":
        _src_solid = torch.stack(
            [
                solid_mask.roll(sz, dims=0).roll(sy, dims=1).roll(sx, dims=2)
                for sx, sy, sz in _C19_SHIFTS
            ]
        )  # (19,nz,ny,nx): pull source x - c_q lies in SOLID
        _fluid_bb = (~solid_mask).unsqueeze(0) & _src_solid
        _f_reflect = f_post[_OPP.to(device)]
        # Selective free-slip (specular) reflection: links whose pull source
        # lies in a free-slip wall reverse only the wall-normal component
        # (waLBerla ``FreeSlip``), all other wall links bounce back.  The
        # y-wall form matches the historical ``free_slip_y``/``y_wall_mask``
        # contract; ``free_slip_mask``/``free_slip_all`` generalise it to an
        # arbitrary slip-wall set (dam-break validation uses free-slip on all
        # walls, waLBerla ``DamBreakRectangular.prm``).
        _slip_mask = free_slip_mask
        if _slip_mask is None and free_slip_all:
            _slip_mask = solid_mask
        if _slip_mask is not None:
            _src_slip = torch.stack(
                [
                    _slip_mask.roll(sz, dims=0).roll(sy, dims=1).roll(sx, dims=2)
                    for sx, sy, sz in _C19_SHIFTS
                ]
            )
            _mir_idx = _specular_wall_mirror_index(_slip_mask, device)
            _f_reflect = torch.where(
                _fluid_bb & _src_slip, torch.gather(f_post, 0, _mir_idx), _f_reflect
            )
        elif free_slip_y and y_wall_mask is not None:
            from .boundary.boundaries3d import _D3Q19_MIRROR_Y

            _src_y = torch.stack(
                [
                    y_wall_mask.roll(sz, dims=0).roll(sy, dims=1).roll(sx, dims=2)
                    for sx, sy, sz in _C19_SHIFTS
                ]
            )
            _f_reflect = torch.where(
                _fluid_bb & _src_y,
                f_post[_D3Q19_MIRROR_Y.to(device)],
                _f_reflect,
            )
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

    # ---- 4. Mass exchange (Koerner eq. 9 / Thuerey table 4.1) ----
    # (no .any() sync — multicard-safe under TCCL; torch.where handles empty masks)
    #
    # Every INTERFACE cell receives over every D3Q19 link (no receiver gate):
    #   * L/I link:  dm = f_in - f_out            (full weight, Koerner eq. 9)
    #   * I/I link:  dm = 0.5 (eps + eps_nb) (f_in - f_out), skipped when the
    #     pull source has no LIQUID neighbour while the cell itself has no GAS
    #     neighbour (Thuerey table 4.1; waLBerla ``useSimpleMassExchange=false``).
    #   * gas link:  no mass flux (gas is a pressure boundary, not a liquid
    #     mass reservoir).
    # f_in is the pure streamed state (frozen pre-ABB in f_exchange); f_out is
    # the local post-collision outgoing population.  LIQUID cells are never
    # debited: their fill is pinned at 1 by definition (waLBerla never touches
    # the liquid fill; the legacy paired bulk debit is removed).
    iface_mask = flags == INTERFACE
    # neighbor_flags always computed in anti-bounce-back above (no None check)
    # For pull link q at x, the opposing outgoing population belongs to x
    # itself: f_bar(q)^*(x).  Sampling it at x-c_q mixes two different links.
    f_opp_nb = f_post[_OPP.to(device)]  # (19, nz, ny, nx)
    from_liq = iface_mask.unsqueeze(0) & (neighbor_flags == LIQUID)
    from_iface = iface_mask.unsqueeze(0) & (neighbor_flags == INTERFACE)
    diff = f_exchange - f_opp_nb
    mass_delta_liquid = torch.where(from_liq, diff, torch.zeros_like(f))
    # I/I fill-averaged weight 0.5 (eps(x) + eps(x - c_q)) from the
    # pre-exchange fill field.
    fill_nb = torch.stack(
        [fill] + [roll_from_pull_source(fill, q) for q in D3Q19_MOVING_Q]
    )  # (19, nz, ny, nx): fill at the pull source x - c_q
    ii_weight = 0.5 * (fill.unsqueeze(0) + fill_nb)
    # Thuerey table 4.1 selector: an I/I link is dead when its pull source has
    # no LIQUID neighbour while the receiving cell has no GAS neighbour.
    liq_nbr_count = (neighbor_flags == LIQUID).sum(dim=0)
    liq_nbr_at_source = torch.stack(
        [liq_nbr_count] + [roll_from_pull_source(liq_nbr_count, q) for q in D3Q19_MOVING_Q]
    )
    gas_nbr_here = (neighbor_flags == GAS).any(dim=0).unsqueeze(0)
    ii_dead = from_iface & (liq_nbr_at_source == 0) & ~gas_nbr_here
    live_iface = from_iface & ~ii_dead
    mass_delta_interface = torch.where(live_iface, ii_weight * diff, torch.zeros_like(f))
    mass_delta = (mass_delta_liquid + mass_delta_interface).sum(0)
    # The exchange increment is committed as-is (no clamp): over/undershoot is
    # the physical conversion trigger (fill > 1 + kappa / fill < -kappa) and
    # the redistribution/conversion pair below settles it conservatively.
    # Only non-finite values are sanitised (last line of defence before the
    # independent mass field is committed).
    mass_delta = torch.nan_to_num(mass_delta, nan=0.0, posinf=0.0, neginf=0.0)
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
    # Post-boundary local density: the fill level is mass / rho_local
    # (waLBerla ``TotalMassComputer`` convention: m = fill * rho_local), with
    # LIQUID pinned at 1 and GAS/SOLID at 0.  The interface fill is
    # deliberately unclamped so the conversion gates below read the true
    # over/undershoot (fill > 1 + kappa / fill < -kappa).
    rho_new = f.sum(dim=0).clamp(min=1e-6)
    fill = torch.where(iface_mask, mass / rho_new, fill)
    fill = torch.where(liquid_pre, torch.ones_like(fill), fill)
    fill = torch.where(gas_mask_pre | solid_mask, torch.zeros_like(fill), fill)
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
        mass_ledger["exchange_bulk_debit"] = 0.0
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
                exchange_bulk_debit=0.0,
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

    # ---- 5. Cell conversion marking (waLBerla ``FindInterfaceCellConversion``).
    # Only INTERFACE cells convert.  Threshold conversion at fill > 1 + kappa
    # (to LIQUID) and fill < -kappa (to GAS), with forced conversion for cells
    # that lost their required neighbourhood:
    #   * forced LIQUID: no GAS neighbour and (fill > 1 - kappa_f or no
    #     INTERFACE neighbour);
    #   * forced GAS: no LIQUID neighbour and (fill < kappa_f or no INTERFACE
    #     neighbour).
    # There is no direct LIQUID -> GAS conversion and no GAS -> INTERFACE
    # birth by mass: the envelope is rebuilt conversion-driven around cells
    # that actually converted (waLBerla ``CellConversionSweep``).
    _kappa = float(cell_conversion_threshold)
    _kappa_f = float(cell_conversion_force_threshold)
    to_iface = torch.zeros_like(gas_mask)
    to_liq = interface_mask & (fill > 1.0 + _kappa)
    to_gas = interface_mask & (fill < -_kappa)
    # neighbor_flags still describes the pre-conversion flags (flags are
    # untouched between the ABB section and here).
    _has_gas_nb = (neighbor_flags == GAS).any(dim=0)
    _has_liq_nb = (neighbor_flags == LIQUID).any(dim=0)
    _has_if_nb = (neighbor_flags == INTERFACE).any(dim=0)
    to_liq = to_liq | (interface_mask & ~_has_gas_nb & ((fill > 1.0 - _kappa_f) | ~_has_if_nb))
    to_gas = to_gas | (interface_mask & ~_has_liq_nb & ((fill < _kappa_f) | ~_has_if_nb))
    # Liquid priority on the (pathological) double trigger, as in the
    # reference if/else chain.
    to_gas = to_gas & ~to_liq
    # ---- 5a. Körner mass redistribution (excess → interface neighbors) ----
    # Excess mass at converting cells (vectorized, no bool sync).
    # I->L overflow:  excess = m - rho_donor (fill - 1 in mass units);
    # I->G residual:  excess = m (the leftover tracked mass, usually a small
    # negative value).  Both are distributed evenly over the surviving
    # INTERFACE neighbours and the conversion-driven new receivers below
    # (waLBerla ``ExcessMassDistributionSweepInterfaceEvenly`` with
    # ``EvenlyAllInterface``).  Negative excess is distributed as-is; the
    # donor reset in the topology transaction pairs it exactly, so conversion
    # neither creates nor destroys mass when receivers exist.
    excess_to_liq = mass - rho_new
    excess_to_g = mass
    excess = torch.where(to_liq, excess_to_liq, torch.zeros_like(mass)) + torch.where(
        to_gas, excess_to_g, torch.zeros_like(mass)
    )
    # Receivers: surviving INTERFACE cells plus gas cells adjacent to a
    # converting I->L cell (promoted to INTERFACE by the transaction, which
    # keeps their credited mass).
    recv_iface = interface_mask & ~to_liq & ~to_gas
    adjacent_converting = torch.stack(all_moving_neighbor_masks(to_liq)).any(dim=0)
    recv_new = gas_mask & adjacent_converting & ~solid_mask
    recv_mask = recv_iface | recv_new
    # Count receiving cells per donor over every moving D3Q19 link.
    shifted_recv = torch.stack(all_moving_neighbor_masks(recv_mask))
    n_recv = shifted_recv.sum(dim=0).float().clamp(min=1.0)
    # Excess per receiving neighbor
    excess_per_nb = excess / n_recv
    # Aggregate every D3Q19 receiver contribution in the mass dtype, then
    # commit it once.
    legacy_redistribution_increment = torch.stack(
        [roll_to_neighbor(excess_per_nb, q) * recv_mask for q in D3Q19_MOVING_Q]
    ).sum(dim=0)
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
        rho=rho_new,
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
            exchange_bulk_debit=0.0,
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
