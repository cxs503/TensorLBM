"""Contact-angle hysteresis from a direction-dependent wall adsorption model.

Physics and honesty statement
-----------------------------
A real contact line shows hysteresis because the wall's adsorption state is
history dependent (advancing differs from receding: Dettre-Johnson, de Gennes).
In a pseudopotential lattice-Boltzmann model the wall enters through a single
number, the wall pseudopotential ``psi_w`` (here ``f = psi_w/psi(rho_l)``), and
that number is *static*: the same ``f`` acts at the advancing and the receding
contact line, so the model by itself has no adsorption hysteresis.  What a driven
droplet does show is dynamic asymmetry (front compressed, rear stretched), and
that is measured in :mod:`tensorlbm.contact_line.capillary_driven`.

This module adds the missing ingredient as an **explicit, documented model**:
the local wall adsorption follows the local contact-line direction,

    f(x) = f_adv  where the liquid coverage of the wall band is increasing,
    f(x) = f_rec  where it is decreasing,
    f(x) unchanged inside a dead band           (pinning),

i.e. a hysteretic (rate/direction dependent) wall.  The resulting gap
``theta_A - theta_R`` is therefore *imposed* by ``|f_adv - f_rec|`` and
calibrated on the measured static-angle curve
(:func:`tensorlbm.contact_line.capillary_driven.static_contact_angle`); it is
**not** an emergent pinning force.  Criterion C4 of the node (loop opens with
``theta_A > theta_R`` and closes monotonically as the wetting-parameter gap goes
to zero) is what this model can be tested against, and the report says exactly
that.  The `f_adv = f_rec` case is the control: the loop must close.

Every number produced here (gap, opening, monotonicity, loop area) is computed
from measured angles, and :func:`evaluate_c4` reports the raw per-``Delta f``
values so the verdict can be re-derived independently.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch

from .capillary_driven import run_driven_drop

__all__ = [
    "C4Verdict",
    "HysteresisGap",
    "HystereticWall",
    "evaluate_c4",
    "hysteresis_gap_sweep",
    "loop_area",
    "monotonicity",
]


@dataclass
class HystereticWall:
    """Direction-dependent wall adsorption (see the module docstring).

    Args:
        frac_advancing: ``psi_w/psi(rho_l)`` where the liquid front advances.
                        A larger ``psi_w`` means a *less* wetting wall in this
                        library's convention, i.e. a larger static angle.
        frac_receding:  ``psi_w/psi(rho_l)`` where the liquid recedes.
        band_rows:      Number of fluid rows just above the wall whose liquid
                        content defines the per-column coverage.
        deadband:       Coverage change (per column, in density-units) below
                        which the state is held (pinning).
        smooth:         Half-width of a box smoothing of the coverage change
                        before thresholding (removes single-column flicker).
    """

    frac_advancing: float
    frac_receding: float
    band_rows: int = 4
    deadband: float = 0.05
    smooth: int = 1
    base_fraction: float | None = None

    state: torch.Tensor | None = field(default=None, repr=False)
    coverage_prev: torch.Tensor | None = field(default=None, repr=False)
    n_updates: int = 0
    n_switch: int = 0

    @property
    def delta_fraction(self) -> float:
        return float(self.frac_advancing) - float(self.frac_receding)

    def bind(self, stepper: object) -> None:
        """Install the initial (uniform, base) wall into the stepper."""
        nx = stepper.channel.nx  # type: ignore[attr-defined]
        self.state = torch.ones(nx, dtype=torch.bool, device=stepper.device)  # type: ignore[attr-defined]
        self.coverage_prev = None
        base = (
            float(self.base_fraction)
            if self.base_fraction is not None
            else 0.5 * (float(self.frac_advancing) + float(self.frac_receding))
        )
        stepper.set_wall_psi(stepper.uniform_wall(base))  # type: ignore[attr-defined]

    def coverage(self, rho: torch.Tensor, stepper: object) -> torch.Tensor:
        """Liquid content of the wall band, per column."""
        wall_rows = stepper.channel.wall_rows  # type: ignore[attr-defined]
        rho_v = stepper.rho_v  # type: ignore[attr-defined]
        band = rho[wall_rows : wall_rows + int(self.band_rows), :]
        return (band - rho_v).clamp_min(0.0).sum(dim=0)

    def _field(self, stepper: object) -> torch.Tensor:
        nx = stepper.channel.nx  # type: ignore[attr-defined]
        frac = torch.where(
            self.state,
            torch.full((nx,), float(self.frac_advancing), device=self.state.device),
            torch.full((nx,), float(self.frac_receding), device=self.state.device),
        )
        return stepper.wall_fraction_field(frac.unsqueeze(0).expand(stepper.channel.shape))  # type: ignore[attr-defined]

    def update(self, rho: torch.Tensor, stepper: object) -> None:
        """Advance the wall state one update and install the new wall field."""
        cov = self.coverage(rho, stepper)
        if self.coverage_prev is None:
            self.coverage_prev = cov
            stepper.set_wall_psi(self._field(stepper))  # type: ignore[attr-defined]
            return
        delta = cov - self.coverage_prev
        self.coverage_prev = cov
        k = int(self.smooth)
        if k > 0:
            kern = torch.ones(2 * k + 1, device=delta.device) / (2 * k + 1)
            pad = torch.cat([delta[-k:], delta, delta[:k]])
            delta = torch.nn.functional.conv1d(pad.view(1, 1, -1), kern.view(1, 1, -1)).view(-1)
        advancing = delta > float(self.deadband)
        receding = delta < -float(self.deadband)
        new_state = torch.where(advancing, True, torch.where(receding, False, self.state))
        self.n_switch += int((new_state != self.state).sum())
        self.state = new_state
        self.n_updates += 1
        stepper.set_wall_psi(self._field(stepper))  # type: ignore[attr-defined]

    def mean_fraction(self) -> float:
        """Instantaneous mean wall fraction actually applied."""
        if self.state is None:
            return 0.5 * (self.frac_advancing + self.frac_receding)
        frac = float(self.frac_advancing) * float(self.state.float().mean())
        frac += float(self.frac_receding) * float(1.0 - self.state.float().mean())
        return frac


@dataclass
class HysteresisGap:
    """One driven run at a fixed wetting-parameter gap ``Delta f``."""

    delta_fraction: float
    frac_advancing: float
    frac_receding: float
    gx: float
    capillary: float
    velocity: float
    theta_a_deg: float
    theta_r_deg: float
    theta_a_unc_deg: float
    theta_r_unc_deg: float

    @property
    def gap_deg(self) -> float:
        return self.theta_a_deg - self.theta_r_deg

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["gap_deg"] = self.gap_deg
        return d


def hysteresis_gap_sweep(
    *,
    base_fraction: float,
    delta_fractions: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15),
    gx: float = 1e-5,
    n_steps: int = 16000,
    relax_steps: int = 8000,
    channel: object | None = None,
    radius: float = 14.0,
    tau: float = 1.0,
    compile_step: bool = False,
    sigma: float = 0.05611,
    rho_l: float = 1.957,
    deadband: float = 0.05,
) -> list[HysteresisGap]:
    """Run one driven droplet per wetting gap and return the angle pair.

    ``f_adv = base + Delta f / 2`` and ``f_rec = base - Delta f / 2``: a positive
    ``Delta f`` makes the advancing portion of the wall less wetting, hence
    ``theta_A > theta_R`` by construction of the calibration curve.
    """
    from .capillary_driven import capillary_number, liquid_mobility

    out: list[HysteresisGap] = []
    mu = liquid_mobility(tau, rho_l)
    for d in delta_fractions:
        wall = HystereticWall(
            frac_advancing=base_fraction + 0.5 * float(d),
            frac_receding=base_fraction - 0.5 * float(d),
            base_fraction=base_fraction,
            deadband=deadband,
        )
        res = run_driven_drop(
            gx=gx,
            wall_fraction=base_fraction,
            channel=channel,  # type: ignore[arg-type]
            radius=radius,
            tau=tau,
            n_steps=n_steps,
            relax_steps=relax_steps,
            compile_step=compile_step,
            wall_model=wall,
        )
        if not res.theta_a:
            continue
        ta_mean, _ = res.angle("A")
        tr_mean, _ = res.angle("R")
        out.append(
            HysteresisGap(
                delta_fraction=float(d),
                frac_advancing=wall.frac_advancing,
                frac_receding=wall.frac_receding,
                gx=float(gx),
                capillary=capillary_number(res.velocity, mu, sigma),
                velocity=res.velocity,
                theta_a_deg=ta_mean,
                theta_r_deg=tr_mean,
                theta_a_unc_deg=res.angle_unc("A"),
                theta_r_unc_deg=res.angle_unc("R"),
            )
        )
    return out


def monotonicity(x: list[float], y: list[float]) -> dict[str, object]:
    """Spearman rank correlation and a strict monotonicity flag.

    Returns:
        Dict with ``spearman``, ``strictly_increasing``, ``n`` and ``passes``
        (``spearman >= 0.9`` and strictly increasing).
    """
    n = len(x)
    if n < 3:
        return {"spearman": float("nan"), "strictly_increasing": False, "n": n, "passes": False}

    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = 0.5 * (i + j) + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx = torch.tensor(ranks(x), dtype=torch.float64)
    ry = torch.tensor(ranks(y), dtype=torch.float64)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(torch.sqrt((rx * rx).sum() * (ry * ry).sum()))
    rho = float((rx * ry).sum()) / denom if denom > 0 else float("nan")
    strict = all(y[i + 1] > y[i] for i in range(n - 1))
    return {
        "spearman": rho,
        "strictly_increasing": bool(strict),
        "n": n,
        "passes": bool(rho >= 0.9 and strict),
    }


@dataclass
class C4Verdict:
    """Machine verdict of criterion C4 (hysteresis loop)."""

    gaps: list[HysteresisGap]
    spearman: float
    strictly_increasing: bool
    all_positive: bool
    control_gap_deg: float
    control_tolerance_deg: float
    passes: bool
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "gaps": [g.to_dict() for g in self.gaps],
            "spearman": self.spearman,
            "strictly_increasing": self.strictly_increasing,
            "all_positive": self.all_positive,
            "control_gap_deg": self.control_gap_deg,
            "control_tolerance_deg": self.control_tolerance_deg,
            "passes": self.passes,
            "note": self.note,
        }


def evaluate_c4(
    gaps: list[HysteresisGap],
    *,
    control_tolerance_deg: float = 3.0,
) -> C4Verdict:
    """Criterion C4: ``theta_A - theta_R > 0`` and monotone in ``Delta f``.

    The ``Delta f = 0`` entry is used as the *control*: with no wetting-parameter
    gap the loop must close (``|gap| <= control_tolerance_deg``).

    Returns:
        :class:`C4Verdict`.
    """
    if not gaps:
        return C4Verdict([], float("nan"), False, False, float("nan"), control_tolerance_deg, False,
                         "no runs")
    ordered = sorted(gaps, key=lambda g: g.delta_fraction)
    drv = [g for g in ordered if g.delta_fraction > 0.0]
    ctrl = [g for g in ordered if g.delta_fraction == 0.0]
    all_positive = all(g.gap_deg > 0.0 for g in drv) if drv else False
    mono = monotonicity([g.delta_fraction for g in drv], [g.gap_deg for g in drv]) if len(drv) >= 3 else {
        "spearman": float("nan"),
        "strictly_increasing": False,
        "passes": False,
    }
    control_gap = ctrl[0].gap_deg if ctrl else float("nan")
    control_ok = (abs(control_gap) <= control_tolerance_deg) if ctrl else False
    passes = bool(all_positive and mono["passes"] and control_ok)
    note = ""
    if not all_positive:
        note = "at least one driven gap is not positive"
    elif not mono["passes"]:  # pragma: no cover - only on bad data
        note = "gap is not strictly increasing with |Delta f|"
    elif not control_ok:
        note = f"control (Delta f = 0) gap {control_gap:.2f} deg exceeds {control_tolerance_deg} deg"
    return C4Verdict(
        gaps=ordered,
        spearman=float(mono["spearman"]),
        strictly_increasing=bool(mono["strictly_increasing"]),
        all_positive=bool(all_positive),
        control_gap_deg=float(control_gap),
        control_tolerance_deg=float(control_tolerance_deg),
        passes=passes,
        note=note,
    )


def loop_area(
    capillary_up: list[float],
    theta_up: list[float],
    capillary_down: list[float],
    theta_down: list[float],
) -> float:
    """Trapezoidal area enclosed by an ascending and a descending ``theta(Ca)`` branch.

    Both branches are sorted by capillary number and integrated over the common
    ``Ca`` interval: ``A = |int (theta_up - theta_down) d ln Ca|`` (units: degrees
    per ``e``-fold of ``Ca``).  A vanishing area means a reversible
    (hysteresis-free) contact line.
    """
    if len(capillary_up) < 2 or len(capillary_down) < 2:
        return float("nan")
    up = sorted(zip(capillary_up, theta_up))
    dn = sorted(zip(capillary_down, theta_down))
    lo = max(up[0][0], dn[0][0])
    hi = min(up[-1][0], dn[-1][0])
    if hi <= lo:
        return float("nan")

    def interp(branch: list[tuple[float, float]], x: float) -> float:
        for i in range(len(branch) - 1):
            x0, y0 = branch[i]
            x1, y1 = branch[i + 1]
            if x0 <= x <= x1:
                f = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
                return y0 + f * (y1 - y0)
        return branch[-1][1]

    n = 64
    xs = [lo * (hi / lo) ** (k / (n - 1)) for k in range(n)]
    vals = [interp(up, x) - interp(dn, x) for x in xs]
    lx = [math.log(x) for x in xs]
    area = 0.0
    for k in range(n - 1):
        area += 0.5 * (vals[k] + vals[k + 1]) * (lx[k + 1] - lx[k])
    return abs(area)