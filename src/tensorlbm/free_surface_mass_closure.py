"""Stage-resolved closure ledger for the free-surface LBM liquid inventory.

Why this exists (IN13-W2-N04)
-----------------------------
:mod:`tensorlbm.free_surface_mass_ledger` turned the free-surface "mass drift"
from *unmeasured* into a per-step series, but the series still answers only
"how much", never "through which stage".  The companion report
``docs/reports/report_fs_mass_ledger_in13_w2_n02.md`` therefore had to explain a
per-step non-conservation of ``O(1e-6)`` with prose ("ABB gas closure plus
conversion") -- one scalar cannot say which of the eight stages of
:func:`tensorlbm.free_surface_lbm.free_surface_step` actually moved the
inventory.

This module closes that gap.  The solver already exposes an *observer-only*
stage probe (:func:`tensorlbm.free_surface_inventory_reconciliation.
inventory_measurement`, switched on by ``inventory_reconciliation_ledger=`` which
does not feed any value back).  :class:`ClosureLedger` consumes the per-stage
deltas of one step and writes them into a pre-allocated device buffer, so a run
ends up with a ``(n_channels, n_steps)`` matrix of *attributed* inventory
changes.  Three things follow for free:

``closure residual``
    ``observed_total_delta - sum(stage deltas)``.  It is only zero when the
    channel set is **complete** -- an un-attributed pathway would show up here
    as a residual far above round-off.  This is the machine-checkable half of
    the "分段恒等式": the segments must add up to the measured change.

``interior vs boundary channels``
    ``mass_exchange`` and the ``topology_*`` stages are *interior bookkeeping*:
    in a closed domain they must not create or destroy liquid.  ``stream_and_
    gas_zero``, ``abb_gas_closure`` and ``wall_boundary`` are *boundary/physical*
    stages: the gas volume is a reservoir and the pressure boundary is a source,
    so a non-zero delta there is physics (compressibility, gas closure), not a
    solver defect.  Splitting the two makes a threshold meaningful -- the same
    number means opposite things in the two groups.

``channel signature of a defect``
    A negative control (e.g. switching off the conversion-driven interface
    refill) has a *fingerprint*: it must appear on the ``mass_exchange`` /
    ``topology_conversion`` channels and not on ``abb_gas_closure``.  A single
    scalar drift cannot distinguish the two, which is why the N02 report had to
    fall back on "the ledger resolves the defect at all".

Everything in :meth:`ClosureLedger.record_step` runs on tensors: the stage
deltas arrive as Python floats *because the solver-side probe already synced*,
but the accumulation itself is a device-to-device write and
:meth:`ClosureLedger.summary` performs the single host transfer of the run.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import torch

__all__ = [
    "CANONICAL_CHANNELS",
    "EXCHANGE_CHANNEL",
    "TOPOLOGY_CHANNELS",
    "INTERIOR_CHANNELS",
    "BOUNDARY_CHANNELS",
    "ClosureLedger",
    "class_mass_identity",
    "channel_signature",
    "front_coefficient",
]

#: Ordered stage channels, matching ``CANONICAL_STAGE_ORDER`` of
#: :mod:`tensorlbm.free_surface_inventory_reconciliation` (the solver emits the
#: ``after_*`` names below as the key of each stage delta).
CANONICAL_CHANNELS: tuple[str, ...] = (
    "after_collision_and_forcing",
    "after_stream_and_gas_zero",
    "after_abb",
    "after_wall_boundary",
    "after_mass_exchange",
    "after_topology_redistribution",
    "after_topology_clamp",
    "after_topology_conversion",
    "after_topology_halo_isolation_boundary",
)

#: Channels that must not create or destroy tracked liquid in a closed domain:
#: the Körner exchange stencil and the conversion bookkeeping.  A drift here is
#: a bookkeeping defect, and this is the group the J1 threshold is applied to.
INTERIOR_CHANNELS: tuple[str, ...] = (
    "after_mass_exchange",
    "after_topology_redistribution",
    "after_topology_clamp",
    "after_topology_conversion",
    "after_topology_halo_isolation_boundary",
)

#: Channels with a physical reservoir: the gas volume that is zeroed after the
#: stream, the anti-bounce-back gas pressure boundary, the wall bounce-back.
#: A non-zero delta here is compressibility / gas closure, not a leak.
BOUNDARY_CHANNELS: tuple[str, ...] = (
    "after_collision_and_forcing",
    "after_stream_and_gas_zero",
    "after_abb",
    "after_wall_boundary",
)

#: The Körner mass-exchange stencil: the one interior stage that must conserve
#: liquid *exactly* (it only redistributes, it cannot create), so it is the
#: channel the conservation threshold is applied to.
EXCHANGE_CHANNEL: str = "after_mass_exchange"

#: The conversion stages: they switch a cell between the density
#: representation (LIQUID contributes ``rho``) and the fill representation
#: (INTERFACE contributes ``fill*rho_l``), so on a compressed bulk they carry a
#: representation jump ``sum(rho - 1)`` that is *not* a conservation error.
TOPOLOGY_CHANNELS: tuple[str, ...] = tuple(
    name for name in INTERIOR_CHANNELS if name != "after_mass_exchange"
)

#: Measure the closure ledger audits on.
MEASURE = "total_liquid_inventory"


def class_mass_identity(fill: torch.Tensor, rho_liquid: float = 1.0) -> dict[str, torch.Tensor]:
    """The three-class identity of the fill segmentation (device tensors only).

    ``{fill <= 0} + {0 < fill < 1} + {fill >= 1} == fill`` on *every* cell, so
    the three class sums must reproduce ``sum(fill)`` to the round-off of the
    summation order.  Returned as device scalars; the residual is normalised by
    the float32 spacing of the total, i.e. it is quoted in ulp.
    """
    empty = fill <= 0
    full = fill >= 1
    partial = ~(empty | full)
    zero = torch.zeros((), dtype=fill.dtype, device=fill.device)
    s_empty = torch.where(empty, fill, zero).sum()
    s_partial = torch.where(partial, fill, zero).sum()
    s_full = torch.where(full, fill, zero).sum()
    total = fill.sum()
    segmented = s_empty + s_partial + s_full
    eps = torch.finfo(fill.dtype).eps
    scale = torch.maximum(total.abs() * rho_liquid, torch.ones_like(total))
    return {
        "total": total,
        "segmented": segmented,
        "residual": segmented - total,
        "residual_ulps": (segmented - total).abs() / (eps * scale),
        "empty": s_empty,
        "partial": s_partial,
        "full": s_full,
    }


class ClosureLedger:
    """Per-step, per-stage attribution of the liquid-inventory change.

    Parameters
    ----------
    n_steps:
        Number of steps that will be recorded (the buffer is pre-allocated, so
        recording itself never grows memory).
    channels:
        Channel names; must be keys of the stage-delta mapping the solver
        produces.  Defaults to :data:`CANONICAL_CHANNELS`.
    measure:
        Which measured quantity to attribute; default ``total_liquid_inventory``
        (population density in LIQUID + ``fill*rho_liquid`` in INTERFACE).
    device, dtype:
        Buffer placement; ``device=None`` keeps the buffer on the CPU.

    Notes
    -----
    ``record_step`` takes a mapping of Python floats because the solver-side
    observer has already performed the host transfer (a stage probe that did
    *not* sync would have to be a different, in-kernel design).  The ledger's
    own contribution is that it never syncs again: the buffer write is a device
    op and the whole matrix is read back once, in :meth:`summary`.
    """

    def __init__(
        self,
        n_steps: int,
        channels: Iterable[str] = CANONICAL_CHANNELS,
        *,
        measure: str = MEASURE,
        device=None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.channels = tuple(channels)
        self.measure = measure
        self.n_steps = int(n_steps)
        self._buf = torch.zeros(
            (len(self.channels), self.n_steps), dtype=dtype, device=device
        )
        self._observed = torch.zeros(self.n_steps, dtype=dtype, device=device)
        self._v0 = torch.zeros(1, dtype=dtype, device=device)
        self._v_start = torch.zeros(self.n_steps, dtype=dtype, device=device)
        self._cursor = 0

    # ------------------------------------------------------------------ record
    @property
    def steps_recorded(self) -> int:
        return self._cursor

    @property
    def device(self):
        return self._buf.device

    def set_initial(self, value: float) -> None:
        """Store the reference measure ``V0`` used to normalise every channel."""
        self._v0[0] = float(value)

    def record_step(
        self,
        stage_deltas: Mapping[str, Mapping[str, float]],
        observed_delta: float | None = None,
        v_before: float | None = None,
    ) -> None:
        """Append one step of attributed deltas.

        ``stage_deltas`` is the ``stage_deltas`` mapping of the solver's
        ``inventory_reconciliation_ledger`` (``{stage: {measure: delta}}``).
        Missing stages are recorded as zero -- which keeps the channel set fixed
        across steps, and makes a *silently absent* channel visible as a
        non-zero closure residual instead of a shifted index.
        """
        if self._cursor >= self.n_steps:
            raise IndexError("ClosureLedger buffer is full")
        row = torch.zeros(len(self.channels), dtype=self._buf.dtype, device=self._buf.device)
        for i, name in enumerate(self.channels):
            entry = stage_deltas.get(name)
            if entry is not None:
                row[i] = float(entry.get(self.measure, 0.0))
        self._buf[:, self._cursor] = row
        if observed_delta is None:
            # only here is a host transfer unavoidable: the caller did not
            # supply the independently observed total for this step.
            observed_delta = float(row.sum().item())
        self._observed[self._cursor] = float(observed_delta)
        if v_before is not None:
            self._v_start[self._cursor] = float(v_before)
        self._cursor += 1

    # ------------------------------------------------------------------ readout
    def channel_series(self, name: str) -> torch.Tensor:
        """Recorded per-step delta of one channel (device tensor)."""
        return self._buf[self.channels.index(name), : self._cursor]

    def tensor(self) -> torch.Tensor:
        """The raw ``(n_channels, steps_recorded)`` matrix."""
        return self._buf[:, : self._cursor]

    def residual(self) -> torch.Tensor:
        """Per-step ``observed - sum(channels)``, in the measure's units."""
        return self._observed[: self._cursor] - self._buf[:, : self._cursor].sum(dim=0)

    def summary(self, ulp_scale: float | None = None) -> dict[str, object]:
        """Single host transfer: channel table + closure residual in ulp.

        ``ulp_scale`` is the absolute round-off floor of the measure (typically
        ``eps * |V0|``); every residual is quoted both absolutely and in these
        units, because "a residual of 1e-6" is meaningless without it.
        """
        host = self.tensor().to("cpu", torch.float64)
        observed = self._observed[: self._cursor].to("cpu", torch.float64)
        v0 = float(self._v0.item())
        out: dict[str, object] = {
            "steps_recorded": self._cursor,
            "channels": list(self.channels),
            "interior_channels": list(INTERIOR_CHANNELS),
            "boundary_channels": list(BOUNDARY_CHANNELS),
            "measure": self.measure,
            "reference_value": v0,
        }
        if host.numel() == 0:
            return out
        if ulp_scale is None:
            ulp_scale = float(torch.finfo(torch.float32).eps) * max(abs(v0), 1.0)
        out["ulp_scale"] = float(ulp_scale)
        total_abs = obs_sum = 0.0
        per_channel: dict[str, dict[str, float]] = {}
        for i, name in enumerate(self.channels):
            series = host[i]
            cumulative = float(series.sum())
            abs_series = series.abs()
            per_channel[name] = {
                "cumulative_delta": cumulative,
                "cumulative_relative": cumulative / v0 if v0 else None,
                "max_abs_step_delta": float(abs_series.max()),
                "max_abs_step_delta_relative": float(abs_series.max()) / abs(v0) if v0 else None,
                "max_abs_step_delta_ulps": float(abs_series.max()) / ulp_scale,
                "mean_abs_step_delta": float(abs_series.mean()),
                "steps_nonzero": int((series != 0).sum()),
                "group": "interior" if name in INTERIOR_CHANNELS else "boundary",
            }
            total_abs += float(abs_series.sum())
            obs_sum += abs(cumulative)
        out["channels_detail"] = per_channel
        for name, entry in per_channel.items():
            entry["share_of_abs_attribution"] = (
                float(entry["mean_abs_step_delta"] * self._cursor) / total_abs
                if total_abs > 0
                else 0.0
            )
        # churn: how much the stages move against each other before they cancel
        churn = float(host.abs().sum())
        net = float(observed.abs().sum())
        out["churn_absolute_sum"] = churn
        out["net_absolute_sum"] = net
        out["churn_ratio"] = churn / net if net > 0 else float("inf")
        residual = observed - host.sum(dim=0)
        out["observed_total_delta"] = float(observed.sum())
        out["summed_channel_delta"] = float(host.sum())
        out["closure_residual_max_abs"] = float(residual.abs().max())
        out["closure_residual_max_ulps"] = float(residual.abs().max()) / ulp_scale
        out["closure_residual_max_relative"] = (
            float(residual.abs().max()) / abs(v0) if v0 else None
        )
        interior = [self.channels.index(n) for n in INTERIOR_CHANNELS if n in self.channels]
        if interior:
            interior_delta = host[interior].sum(dim=0)
            out["interior_cumulative_delta"] = float(interior_delta.sum())
            out["interior_max_abs_step_delta"] = float(interior_delta.abs().max())
            out["interior_max_abs_step_delta_ulps"] = (
                float(interior_delta.abs().max()) / ulp_scale
            )
            out["interior_max_abs_step_delta_relative"] = (
                float(interior_delta.abs().max()) / abs(v0) if v0 else None
            )
        boundary = [self.channels.index(n) for n in BOUNDARY_CHANNELS if n in self.channels]
        if boundary:
            boundary_delta = host[boundary].sum(dim=0)
            out["boundary_cumulative_delta"] = float(boundary_delta.sum())
            out["boundary_max_abs_step_delta_ulps"] = (
                float(boundary_delta.abs().max()) / ulp_scale
            )
        topology_idx = [self.channels.index(n) for n in TOPOLOGY_CHANNELS if n in self.channels]
        topology_series = host[topology_idx].sum(dim=0) if topology_idx else torch.zeros_like(observed)
        quiet = topology_series == 0
        out["quiet_steps"] = int(quiet.sum())
        if bool(quiet.any()):
            denom_q = abs(v0) if v0 else 1.0
            out["observed_quiet_max_abs_step_delta"] = float(observed[quiet].abs().max())
            out["observed_quiet_max_abs_step_delta_relative"] = (
                float(observed[quiet].abs().max()) / denom_q
            )
            out["observed_quiet_max_abs_step_delta_ulps"] = (
                float(observed[quiet].abs().max()) / ulp_scale
            )
            out["observed_quiet_mean_abs_step_delta_relative"] = (
                float(observed[quiet].abs().mean()) / denom_q
            )
        exchange_idx = [self.channels.index(EXCHANGE_CHANNEL)] if EXCHANGE_CHANNEL in self.channels else []
        if exchange_idx:
            exchange_series = host[exchange_idx].sum(dim=0)
            if bool(quiet.any()):
                denom = abs(v0) if v0 else 1.0
                out["exchange_quiet_max_abs_step_delta"] = float(exchange_series[quiet].abs().max())
                out["exchange_quiet_max_abs_step_delta_relative"] = (
                    float(exchange_series[quiet].abs().max()) / denom
                )
                out["exchange_quiet_max_abs_step_delta_ulps"] = (
                    float(exchange_series[quiet].abs().max()) / ulp_scale
                )
                out["exchange_quiet_mean_abs_step_delta_relative"] = (
                    float(exchange_series[quiet].abs().mean()) / denom
                )
        for label, names in (
            ("exchange", (EXCHANGE_CHANNEL,)),
            ("topology", TOPOLOGY_CHANNELS),
        ):
            idx = [self.channels.index(n) for n in names if n in self.channels]
            if not idx:
                continue
            series = host[idx].sum(dim=0)
            out[f"{label}_cumulative_delta"] = float(series.sum())
            out[f"{label}_max_abs_step_delta"] = float(series.abs().max())
            out[f"{label}_max_abs_step_delta_ulps"] = float(series.abs().max()) / ulp_scale
            out[f"{label}_max_abs_step_delta_relative"] = (
                float(series.abs().max()) / abs(v0) if v0 else None
            )
            out[f"{label}_steps_nonzero"] = int((series != 0).sum())
        out["observed_max_abs_step_delta_ulps"] = (
            float(observed.diff().abs().max()) / ulp_scale if observed.numel() > 1 else 0.0
        )
        # steps on which the interior bookkeeping moved at all: conversions
        out["topology_steps"] = int((topology_series != 0).sum())
        return out


def channel_signature(
    summary: Mapping[str, object], channels: Iterable[str] | None = None
) -> dict[str, float]:
    """Flatten a :meth:`ClosureLedger.summary` into ``{channel: max abs ulps}``.

    The signature is what a negative control is compared on: a defect must raise
    the ulp level of a *named* channel, not merely the scalar drift.
    """
    detail = summary.get("channels_detail", {})  # type: ignore[assignment]
    names = tuple(channels) if channels is not None else tuple(detail)
    return {
        name: float(detail[name]["max_abs_step_delta_ulps"])  # type: ignore[index]
        for name in names
        if name in detail  # type: ignore[operator]
    }


def front_coefficient(
    t_dimensionless: Iterable[float],
    z_over_a: Iterable[float],
    window: tuple[float, float] = (0.5, 3.0),
) -> dict[str, float]:
    """Least-squares slope ``c`` of ``Z/a`` against ``T = t sqrt(2 g / a)``.

    Same estimator as the platform dam-break baseline (``c = 1.2016``): a
    straight line through the origin is *not* fitted, the intercept is kept
    because a free-surface run needs a transient before the front is formed.
    Points outside ``window`` are dropped.
    """
    t = torch.as_tensor(list(t_dimensionless), dtype=torch.float64)
    z = torch.as_tensor(list(z_over_a), dtype=torch.float64)
    mask = (t >= window[0]) & (t <= window[1])
    t, z = t[mask], z[mask]
    n = int(t.numel())
    if n < 3:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"), "samples": n}
    t_mean, z_mean = t.mean(), z.mean()
    dt = t - t_mean
    denom = (dt * dt).sum()
    slope = float((dt * (z - z_mean)).sum() / denom) if float(denom) > 0 else float("nan")
    intercept = float(z_mean - slope * t_mean)
    pred = slope * t + intercept
    ss_res = float(((z - pred) ** 2).sum())
    ss_tot = float(((z - z_mean) ** 2).sum())
    return {
        "slope": slope,
        "intercept": intercept,
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "samples": n,
    }