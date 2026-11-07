"""Device-resident, per-step mass ledger for the D3Q19 Körner free surface.

Why this module exists
----------------------
The D3Q19 free-surface solver (:mod:`tensorlbm.free_surface_lbm`) advances a
*geometric* fill field ``fill`` and an *independent* interface mass field
``mass``, and it converts cells between GAS / INTERFACE / LIQUID.  Before this
ledger a run could report a scalar "mass drift" only by copying a Python float
out of the solver every step, so:

* the drift was **not an artifact** -- no per-step series was persisted and no
  threshold could be re-evaluated later;
* the reported number mixed three physically different populations (empty
  envelope cells, saturated cells, partially filled cells) into one scalar,
  so a loss inside the interface layer and a loss at a phase conversion were
  indistinguishable;
* any measurement required a host synchronisation per step, which is exactly
  what a multi-card / TCCL run must avoid.

``MassLedger`` fixes all three: it classifies every cell by ``fill`` into the
three disjoint classes ``{fill <= 0}``, ``{0 < fill < 1}``, ``{fill >= 1}``,
sums the class fills (plus the cell counts and the two-sided excursions) and
writes them into a pre-allocated device buffer -- **no host synchronisation
inside** :meth:`MassLedger.record`.  The single host transfer happens in
:meth:`MassLedger.summary`, at the end of a run.

What is conserved
-----------------
Two different quantities are recorded, because they are *not* the same in a
compressible lattice-Boltzmann fluid:

``fill_total``
    ``sum(fill)`` over the whole domain, i.e. the geometric liquid volume in
    lattice cells.  The Körner exchange feeds the interface from the *bulk
    population*, so this quantity drifts at the O(1e-6)/step level whenever the
    bulk density leaves 1 (hydrostatic compression).  It is recorded, not
    asserted against the ulp criterion.

``inventory``
    ``sum(rho over LIQUID) + sum(fill * rho_liquid over INTERFACE)`` -- the
    module-level conserved liquid inventory already used by
    :func:`tensorlbm.free_surface_lbm.total_liquid_inventory`.  In a closed
    domain (no inlet, no outlet, no far-field reset) this is the quantity that
    conserves to float32 ulp; it is the quantity the J1/J4 criteria are
    evaluated on, and ``ulp_relative`` gives the float32 reference scale.

The three-class split matters for the negative controls: a defect that drops
the credit of a newly born interface cell (the "interface refill" of the
conversion-driven envelope) shows up as ``fill_partial``/``inventory`` loss
with ``cell_partial`` growing, while a leak that only touches saturated cells
shows up in ``fill_full``.  A single scalar cannot tell them apart.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import torch

from .free_surface_lbm import GAS, INTERFACE, LIQUID, SOLID

__all__ = [
    "LEDGER_FIELDS",
    "MassLedger",
    "fill_class_masks",
    "fill_class_sums",
    "ulp_relative",
    "refill_ablation",
    "FRONT_WINDOW",
]

#: Per-step record fields, in buffer order.
LEDGER_FIELDS: tuple[str, ...] = (
    "fill_gas",  # sum(fill) over the {fill <= 0} class (0 unless numerical undershoot)
    "fill_partial",  # sum(fill) over the {0 < fill < 1} class
    "fill_full",  # sum(fill) over the {fill >= 1} class (LIQUID + oversaturated)
    "fill_total",  # sum(fill) over every cell; must equal the three sums above
    "fill_deficit",  # sum(min(fill, 0))  -- measure-zero undershoot
    "fill_excess",  # sum(max(fill - 1, 0)) -- measure-zero overshoot
    "cell_gas",  # count of the {fill <= 0} class
    "cell_partial",  # count of the {0 < fill < 1} class
    "cell_full",  # count of the {fill >= 1} class
    "mass_total",  # sum(mass) -- solver interface mass field
    "inventory",  # sum_L rho + sum_I fill*rho_liquid (conserved liquid inventory)
    "rho_total",  # sum(rho) over LIQUID+INTERFACE -- the LBM's own fluid mass
    "cell_liquid",  # count of flags == LIQUID
    "cell_interface",  # count of flags == INTERFACE
    "cell_solid",  # count of flags == SOLID
    "cell_changed",  # cells whose flag changed since the previous record
)

#: Default dimensionless-time window for the front-coefficient fit.  The
#: platform dam-break baseline (``c = 1.2016``) was fitted over ``tau`` in
#: [0.5, 4.0]; the startup transient below 0.5 is excluded for the same reason.
FRONT_WINDOW: tuple[float, float] = (0.5, 4.0)


def fill_class_masks(fill: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the disjoint (empty, partial, full) masks of the fill classes.

    ``empty``   : ``fill <= 0``   -- gas / empty envelope
    ``partial`` : ``0 < fill < 1`` -- the Körner interface layer
    ``full``    : ``fill >= 1``   -- liquid, plus any oversaturated interface
    """
    empty = fill <= 0
    full = fill >= 1
    partial = ~(empty | full)
    return empty, partial, full


def fill_class_sums(fill: torch.Tensor) -> torch.Tensor:
    """Stacked ``(3,)`` tensor of the class fills, plus the two excursions.

    Device-only: every entry is a 0-dim tensor summed on ``fill.device``.
    """
    empty, partial, full = fill_class_masks(fill)
    zero = torch.zeros((), dtype=fill.dtype, device=fill.device)
    deficit = torch.minimum(fill, zero)
    excess = torch.maximum(fill - 1.0, zero)
    return torch.stack(
        (
            torch.where(empty, fill, zero).sum(),
            torch.where(partial, fill, zero).sum(),
            torch.where(full, fill, zero).sum(),
            deficit.sum(),
            excess.sum(),
        )
    )


def ulp_relative(value: float, dtype: torch.dtype = torch.float32) -> float:
    """float32 relative spacing of ``value`` -- the rounding floor of a total.

    A sum of ``N`` float32 terms carries an absolute rounding floor of about
    ``sqrt(N) * eps * max|term| * N``; quoting a *relative* criterion without
    this floor is meaningless.  ``ulp`` of the total itself (``eps * |value|``)
    is the conservative lower bound used by the J1/J4 gates.
    """
    eps = float(torch.finfo(dtype).eps)
    return eps * max(abs(float(value)), 1.0)


class MassLedger:
    """Per-step, device-resident fill-classification ledger.

    Parameters
    ----------
    steps:
        Number of steps that will be recorded; the buffer is allocated once so
        the hot loop never allocates.
    device, dtype:
        Buffer placement.  ``dtype`` defaults to ``float64`` so that the
        *accumulated* difference of float32-audited stages does not itself add
        rounding; the recorded values are float32 sums upcast on the device.
    rho_liquid:
        Liquid density used by the inventory definition.

    Notes
    -----
    :meth:`record` performs no host synchronisation: it only issues tensor ops
    and a device-to-device buffer write.  Index bookkeeping uses a Python
    ``int`` counter, never a tensor value.
    """

    FIELDS = LEDGER_FIELDS

    def __init__(
        self,
        steps: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
        rho_liquid: float = 1.0,
    ) -> None:
        if int(steps) <= 0:
            raise ValueError("MassLedger needs steps >= 1")
        self.steps = int(steps)
        self.dtype = dtype
        self.rho_liquid = float(rho_liquid)
        self._buf = torch.zeros(
            (len(self.FIELDS), self.steps),
            dtype=dtype,
            device=device if device is not None else "cpu",
        )
        self._cursor = 0
        self._prev_flags: torch.Tensor | None = None

    # ------------------------------------------------------------------ record
    @property
    def device(self) -> torch.device:
        return self._buf.device

    @property
    def steps_recorded(self) -> int:
        return self._cursor

    def record(
        self,
        f: torch.Tensor,
        fill: torch.Tensor,
        flags: torch.Tensor,
        mass: torch.Tensor,
        rho_liquid: float | None = None,
    ) -> None:
        """Record one step.  No host sync, no allocation of a Python float."""
        if self._cursor >= self.steps:
            raise RuntimeError("MassLedger buffer exhausted: allocate more steps")
        rho_l = self.rho_liquid if rho_liquid is None else float(rho_liquid)
        rho = f.sum(dim=0)
        empty, partial, full = fill_class_masks(fill)
        zero = torch.zeros((), dtype=fill.dtype, device=fill.device)
        liquid = flags == LIQUID
        interface = flags == INTERFACE
        inventory = torch.where(liquid, rho, zero).sum() + (
            torch.where(interface, fill, zero).sum() * rho_l
        )
        fluid = liquid | interface
        rho_total = torch.where(fluid, rho, zero).sum()
        if self._prev_flags is None:
            changed = torch.zeros((), dtype=fill.dtype, device=fill.device)
        else:
            changed = (flags != self._prev_flags).sum().to(fill.dtype)
        self._prev_flags = flags.clone()
        row = torch.stack(
            (
                torch.where(empty, fill, zero).sum(),
                torch.where(partial, fill, zero).sum(),
                torch.where(full, fill, zero).sum(),
                fill.sum(),
                torch.minimum(fill, zero).sum(),
                torch.maximum(fill - 1.0, zero).sum(),
                empty.sum().to(fill.dtype),
                partial.sum().to(fill.dtype),
                full.sum().to(fill.dtype),
                mass.sum(),
                inventory,
                rho_total,
                liquid.sum().to(fill.dtype),
                interface.sum().to(fill.dtype),
                (flags == SOLID).sum().to(fill.dtype),
                changed,
            )
        )
        self._buf[:, self._cursor] = row.to(self._buf.dtype)
        self._cursor += 1

    # ------------------------------------------------------------------ readout
    def field(self, name: str) -> torch.Tensor:
        """Recorded series of ``name`` (device tensor; slice needs no sync)."""
        if name not in self.FIELDS:
            raise KeyError(f"unknown ledger field {name!r}")
        return self._buf[self.FIELDS.index(name), : self._cursor]

    def relative_change(self, name: str) -> torch.Tensor:
        """``(q_k - q_0) / q_0`` per recorded step (device tensor)."""
        series = self.field(name)
        return (series - series[0]) / series[0]

    def max_step_change(self, name: str) -> torch.Tensor:
        """max over steps of ``|q_k - q_{k-1}| / |q_0|`` (device scalar)."""
        series = self.field(name)
        if series.numel() < 2:
            return torch.zeros((), dtype=series.dtype, device=series.device)
        return (series[1:] - series[:-1]).abs().max() / series[0].abs()

    def tensor(self) -> torch.Tensor:
        """The raw ``(n_fields, steps_recorded)`` buffer."""
        return self._buf[:, : self._cursor]

    def summary(self) -> dict[str, object]:
        """Single host transfer: per-field totals, drift extremes and ulp floor.

        For every field the summary reports

        * ``initial`` / ``final`` / ``min`` / ``max``;
        * ``max_abs_relative_drift`` -- ``max_k |q_k - q_0| / |q_0|``;
        * ``max_abs_relative_step_change`` -- ``max_k |q_k - q_{k-1}| / |q_0|``;
        * ``max_abs_relative_step_change_quiet`` -- the same statistic restricted
          to steps on which **no cell changed its class** (``cell_changed == 0``).
          A Korner phase conversion switches a cell between the density
          representation (LIQUID contributes ``rho``) and the fill
          representation (INTERFACE contributes ``fill``); on a compressed
          bulk (``rho != 1``) that switch moves the reported inventory by
          ``sum(rho - 1)`` over the converted cells.  That jump is a
          representational discontinuity, not a conservation error, so the
          per-step conservation identity (J1) is evaluated on the quiet steps
          and every topology event is reported separately;
        * ``max_abs_change`` -- absolute version, used when ``q_0 == 0``
          (relative numbers are meaningless for the empty fill classes).

        ``float32_ulp_floor`` gives ``eps * |q_0|`` per field: the rounding floor
        of a float32 sum of that magnitude, i.e. the tolerance a
        "<= 1e-6 relative" statement has to be read against.
        """
        out: dict[str, object] = {
            "steps_recorded": self._cursor,
            "fields": list(self.FIELDS),
            "rho_liquid": self.rho_liquid,
            "float32_ulp_floor": {},
        }
        host = self.tensor().to("cpu", torch.float64)
        if host.numel() == 0:
            return out
        changed = host[self.FIELDS.index("cell_changed")]
        quiet = changed == 0
        for i, name in enumerate(self.FIELDS):
            series = host[i]
            q0 = series[0]
            entry: dict[str, object] = {
                "initial": float(q0),
                "final": float(series[-1]),
                "min": float(series.min()),
                "max": float(series.max()),
                "max_abs_change": float((series - q0).abs().max()),
            }
            if q0 != 0:
                entry["max_abs_relative_drift"] = float(((series - q0) / q0).abs().max())
            else:
                entry["max_abs_relative_drift"] = None
            if series.numel() > 1:
                steps = (series[1:] - series[:-1]).abs()
                denom = abs(q0) if q0 != 0 else 1.0
                entry["max_abs_relative_step_change"] = float((steps / denom).max())
                quiet_mask = quiet[1:]
                entry["max_abs_relative_step_change_quiet"] = (
                    float((steps[quiet_mask] / denom).max()) if bool(quiet_mask.any()) else 0.0
                )
                entry["mean_abs_relative_step_change_quiet"] = (
                    float((steps[quiet_mask] / denom).mean()) if bool(quiet_mask.any()) else 0.0
                )
                entry["quiet_steps"] = int(quiet_mask.sum())
            else:
                entry["max_abs_relative_step_change"] = 0.0
                entry["max_abs_relative_step_change_quiet"] = 0.0
                entry["quiet_steps"] = 0
            out[name] = entry
            out["float32_ulp_floor"][name] = ulp_relative(float(q0))
        out["float32_ulp_relative_floor"] = float(torch.finfo(torch.float32).eps)
        out["topology_steps"] = int((changed > 0).sum())
        return out


def refill_ablation(
    previous_flags: torch.Tensor,
    flags: torch.Tensor,
    mass: torch.Tensor,
    fill: torch.Tensor,
    rho_liquid: float = 1.0,
    include_demotion: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Negative-control defect: drop the mass credit of newly born interface cells.

    The conversion-driven envelope promotes GAS cells next to a converting
    I->L donor to INTERFACE and credits them with the donor's excess mass
    (``recv_new`` receivers in :func:`free_surface_topology_transaction.
    build_topology_transaction`).  With the refill switched off the donor is
    still debited but the receiver keeps nothing, so liquid mass is destroyed
    once per birth event.

    ``include_demotion`` extends "newly born" to every cell that becomes
    INTERFACE, i.e. the LIQUID -> INTERFACE demotion of the conversion-driven
    envelope is also left without its refilled mass.  That is the same defect
    one layer further out (no interface cell is seeded on birth) and is the
    strong arm of the negative control.

    This is a *defect injection for the ledger's negative control*, not a
    physical model and not reachable from the production solver path: it is
    applied by the caller between two :func:`free_surface_step` calls.
    """
    if include_demotion:
        born = ((previous_flags == GAS) | (previous_flags == LIQUID)) & (flags == INTERFACE)
    else:
        born = (previous_flags == GAS) & (flags == INTERFACE)
    zero = torch.zeros((), dtype=mass.dtype, device=mass.device)
    mass = torch.where(born, zero, mass)
    fill = torch.where(born, zero, fill)
    # keep the two representations consistent on the surviving cells
    consistent = torch.where(
        flags == INTERFACE,
        mass / rho_liquid,
        fill,
    )
    return mass, consistent


def linear_fit_slope(
    times: Iterable[float], values: Iterable[float], window: tuple[float, float] = FRONT_WINDOW
) -> dict[str, float]:
    """Least-squares slope of ``values`` vs ``times`` inside ``window``.

    Returns ``{"slope", "intercept", "n", "t_lo", "t_hi", "r2"}``; a fit with
    fewer than three samples or a degenerate time spread is reported with
    ``slope = nan`` rather than silently extrapolated.
    """
    xs: list[float] = []
    ys: list[float] = []
    for t, v in zip(times, values):
        if window[0] <= float(t) <= window[1]:
            xs.append(float(t))
            ys.append(float(v))
    n = len(xs)
    if n < 3:
        return {"slope": float("nan"), "intercept": float("nan"), "n": n, "r2": float("nan")}
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        return {"slope": float("nan"), "intercept": float("nan"), "n": n, "r2": float("nan")}
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "slope": slope,
        "intercept": intercept,
        "n": n,
        "t_lo": xs[0],
        "t_hi": xs[-1],
        "r2": r2,
    }


def observed_order(
    resolutions: Mapping[str, float], rates: Mapping[str, float]
) -> dict[str, float]:
    """Observed convergence order of ``rates`` (drift per unit time) vs ``dx``.

    ``rate ~ dx^p`` with :math:`p = d\ln(rate)/d\ln(dx)`, so ``p > 0`` means the
    drift rate vanishes under refinement and ``p = 0`` means it is
    resolution-independent.  Non-positive rates cannot enter a log fit and are
    reported as ``p = nan`` with the count of usable points.
    """
    xs: list[float] = []
    ys: list[float] = []
    for key, dx in resolutions.items():
        rate = rates.get(key)
        if rate is None:
            continue
        rate = float(rate)
        if not (rate > 0.0):
            continue
        import math

        xs.append(math.log(float(dx)))
        ys.append(math.log(rate))
    if len(xs) < 2:
        return {"order": float("nan"), "n": len(xs), "r2": float("nan")}
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    if sxx <= 0.0:
        return {"order": float("nan"), "n": n, "r2": float("nan")}
    slope = sxy / sxx
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + mean_y - slope * mean_x)) ** 2 for x, y in zip(xs, ys))
    return {
        "order": slope,
        "n": n,
        "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }