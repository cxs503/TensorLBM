"""L_cav error attribution toolkit (p447w2-N06).

The sheet-cavity length ``L_cav`` predicted by the homogeneous-mixture
cavitating-airfoil solver of :mod:`tensorlbm.cavitation_airfoil` differs from
the Brennen (1994) water-tunnel data by O(10^2 %).  This module provides the
minimal, dependency-light machinery needed to split that error into

* a **criterion-caliber** contribution ``Delta_crit`` (which iso-surface is
  called "the cavity": isobaric ``p < p_v``, a vapour-fraction threshold
  ``alpha_v >= tau``, or the gas-volume-equivalent length ``L_eq``);
* a **discretisation** contribution ``Delta_disc`` (grid refinement +
  Richardson extrapolation ``h -> 0``, GCI band);
* an **unmodelled residual** ``Delta_resid`` (everything left at ``h -> 0``).

Design rules
------------
* No literature constants are hidden here: the reference length is passed in
  by the caller (see :func:`tensorlbm.cavitation_airfoil.reference_data`).
* Every metric is a pure function of a field so that the whole attribution is
  reproducible from the saved field snapshots (``artifacts/``).
* The additive decomposition implemented by :func:`error_budget` is *exactly*
  telescoping: ``Delta_crit + Delta_disc + Delta_resid == L_ref - L_base`` for
  the declared base value ``L_base`` (no hidden renormalisation).

Conventions
-----------
* Non-dimensionalisation follows :mod:`tensorlbm.cavitation_airfoil`
  (chord ``c = 1``, ``U_inf = 1``, ``p_inf = 0``, ``p_v = -sigma/2``), so all
  lengths below are in chords.
* ``alpha`` is the vapour volume fraction field of the mixture.
* Grid spacing ``h = 1 / cells_per_chord``.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import torch

from .cavitation_airfoil import CavitationFlowConfig, airfoil_surface, cavity_metrics

__all__ = [
    "ALPHA_THRESHOLDS",
    "CRITERIA",
    "alpha_relative_thresholds",
    "criterion_lengths",
    "equivalent_length",
    "error_budget",
    "isobaric_indicator",
    "richardson_generalized",
    "two_way_anova",
]

#: absolute vapour-fraction thresholds named in the mission brief
ALPHA_THRESHOLDS: tuple[float, ...] = (0.1, 0.5, 0.9)

#: the three primary criteria of the 4-grid x 3-criterion matrix
CRITERIA: tuple[str, ...] = ("isobaric", "alpha_v>=0.5", "L_eq")


# ---------------------------------------------------------------------------
# Vapour indicators / criteria
# ---------------------------------------------------------------------------


def isobaric_indicator(p: torch.Tensor, p_v: float, rel_band: float = 0.05) -> torch.Tensor:
    """Smooth indicator of the isobaric cavity criterion ``p < p_v``.

    ``I = sigmoid((p_v - p) / (rel_band * |p_v|))`` so that ``I > 0.5`` where
    the mixture pressure is below the vapour pressure.  ``rel_band`` is the
    transition width of the one-cell ramp (same convention as the P41-N02
    runs, so the numbers are comparable with that baseline).
    """
    if not math.isfinite(p_v) or p_v == 0.0:
        msg = "p_v must be finite and non-zero"
        raise ValueError(msg)
    pf = p.double()
    return torch.sigmoid((p_v - pf) / (rel_band * abs(p_v)))


def alpha_relative_thresholds(
    alpha: torch.Tensor, levels: Sequence[float] = (0.1, 0.5, 0.9)
) -> tuple[float, ...]:
    """Thresholds scaled by the field maximum (``tau_i = level_i * max(alpha)``).

    The absolute thresholds ``alpha_v >= {0.1, 0.5, 0.9}`` are the mission
    brief's criterion family, but when the mixture solver's vapour field never
    reaches O(10^-2) they are all degenerate (``L_cav == 0``).  The relative
    family is the scale-free twin used to test that the criterion is at least
    *monotone* in its threshold on the field the solver actually produces.
    Returns zeros-safe thresholds: if ``max(alpha) == 0`` the result is all
    zeros, and the corresponding lengths are all zero as well.
    """
    amax = float(alpha.max()) if alpha.numel() else 0.0
    return tuple(float(lv) * amax for lv in levels)


def equivalent_length(
    alpha: torch.Tensor,
    h: float,
    cfg: CavitationFlowConfig | None = None,
    geometry=None,
    rel_level: float = 0.1,
) -> dict:
    """Gas-volume-equivalent cavity length ``L_eq = int(alpha dV) / (t c)``.

    ``int alpha dV`` is the vapour volume per unit span (``c^2`` units for the
    2-D case).  The reference thickness ``t`` is the maximum attached thickness
    of the vapour *footprint*, taken as the region where ``alpha`` exceeds
    ``rel_level * max(alpha)`` (scale-free, so the measure is defined even when
    the absolute vapour fraction stays well below 1).  With ``t`` floored at
    one cell (``h``) so that a vanishing footprint gives ``L_eq -> 0`` instead
    of a division by zero.
    """
    if h <= 0:
        msg = "h must be positive"
        raise ValueError(msg)
    a = alpha.double().clamp(0.0, 1.0)
    volume = float(a.sum()) * h**2  # per unit span
    tau_foot = rel_level * float(a.max())
    delta = 0.0
    extent = 0.0
    if tau_foot > 0.0 and cfg is not None:
        m = cavity_metrics(a, cfg, threshold=tau_foot, geometry=geometry)
        delta = float(m["delta_cav"])
        extent = float(m["l_cav"])
    t_eff = max(delta, h)
    return {
        "volume": volume,
        "delta_footprint": delta,
        "extent_footprint": extent,
        "t_effective": t_eff,
        "l_eq": volume / (t_eff * 1.0),
        "rel_level": rel_level,
        "tau_footprint": tau_foot,
    }


def criterion_lengths(
    p: torch.Tensor,
    alpha: torch.Tensor,
    cfg: CavitationFlowConfig,
    geometry=None,
    alpha_thresholds: Sequence[float] = ALPHA_THRESHOLDS,
    rel_band: float = 0.05,
    eq_rel_level: float = 0.1,
) -> dict:
    """Evaluate the whole criterion family on one (mean) field pair.

    Returns a flat dictionary with ``l_<criterion>`` entries for

    * ``isobaric``  -- :func:`isobaric_indicator` cut at 0.5,
    * ``alpha_v>=tau`` for every ``tau`` in ``alpha_thresholds``,
    * ``alpha_rel>=<level>`` for the scale-free thresholds,
    * ``L_eq`` -- :func:`equivalent_length`.
    """
    if geometry is None:
        geometry = airfoil_surface(cfg.airfoil)
    out: dict = {}

    ind = isobaric_indicator(p, cfg.p_v, rel_band=rel_band)
    m_iso = cavity_metrics(ind, cfg, threshold=0.5, geometry=geometry)
    out["l_isobaric"] = m_iso["l_cav"]
    out["delta_isobaric"] = m_iso["delta_cav"]
    out["detachment_isobaric"] = m_iso["detachment_x"]
    out["closure_isobaric"] = m_iso["closure_x"]
    out["alpha_max"] = float(alpha.max())
    out["p_min"] = float(p.min())
    out["p_rel_min"] = float(p.min()) - cfg.p_v

    abs_lengths: dict[str, float] = {}
    for tau in alpha_thresholds:
        key = f"alpha_v>={tau:g}"
        m = cavity_metrics(alpha, cfg, threshold=float(tau), geometry=geometry)
        abs_lengths[key] = float(m["l_cav"])
        out[f"l_{key}"] = float(m["l_cav"])
        out[f"delta_{key}"] = float(m["delta_cav"])

    for tau in alpha_relative_thresholds(alpha, (0.1, 0.5, 0.9)):
        key = f"alpha_rel>={tau:.6g}"
        m = cavity_metrics(alpha, cfg, threshold=float(tau), geometry=geometry)
        out[f"l_{key}"] = float(m["l_cav"])

    eq = equivalent_length(alpha, cfg.h, cfg=cfg, geometry=geometry, rel_level=eq_rel_level)
    out["l_L_eq"] = float(eq["l_eq"])
    out.update({f"eq_{k}": v for k, v in eq.items()})
    out["criterion_lengths"] = {
        "isobaric": out["l_isobaric"],
        "alpha_v>=0.5": abs_lengths["alpha_v>=0.5"],
        "L_eq": out["l_L_eq"],
    }
    out["alpha_absolute_lengths"] = abs_lengths
    return out


# ---------------------------------------------------------------------------
# Grid convergence: generalised Richardson + GCI
# ---------------------------------------------------------------------------


def richardson_generalized(
    h: Sequence[float], f: Sequence[float], safety: float = 1.25
) -> dict:
    """Generalised Richardson extrapolation with a three-grid sequence.

    Parameters
    ----------
    h:
        Grid spacings ordered **fine -> coarse** (``h[0] < h[1] < h[2]``).
    f:
        The corresponding integral quantities (same order).
    safety:
        Factor of safety ``F_s`` in the GCI definition (Roache 1994;
        Celik et al. 2008 use ``F_s = 1.25`` for three-grid studies).

    Returns
    -------
    dict with the observed order ``p``, the extrapolated value ``f_extrap``,
    the relative extrapolation error ``e_a`` and the fine-grid ``GCI_fine``;
    ``valid=False`` (with a ``reason``) when the sequence is not monotone or
    not in the asymptotic range, in which case ``p``/``f_extrap``/``GCI`` are
    returned as ``nan`` -- never silently invented.

    Notes
    -----
    The refinement ratios are allowed to be non-uniform; the fixed-point
    iteration is the standard generalised form

    ``p = |ln|e32/e21| + q(p)| / ln(r21)``,
    ``q(p) = ln((r21^p - s) / (r32^p - s))``, ``s = sign(e32/e21)``.
    """
    if len(h) != 3 or len(f) != 3:
        msg = "richardson_generalized needs exactly three grids"
        raise ValueError(msg)
    hs = [float(x) for x in h]
    fs = [float(x) for x in f]
    if not (hs[0] < hs[1] < hs[2]):
        msg = "h must be strictly increasing (fine -> coarse)"
        raise ValueError(msg)
    r21 = hs[1] / hs[0]
    r32 = hs[2] / hs[1]
    e21 = fs[1] - fs[0]
    e32 = fs[2] - fs[1]
    out = {
        "h": hs,
        "f": fs,
        "r21": r21,
        "r32": r32,
        "e21": e21,
        "e32": e32,
        "ratio_r": (e32 / e21) if e21 != 0.0 else float("nan"),
        "p": float("nan"),
        "f_extrap": float("nan"),
        "e_a": float("nan"),
        "gci_fine": float("nan"),
        "valid": False,
        "reason": "",
    }
    if e21 == 0.0 or e32 == 0.0:
        out["reason"] = "zero difference between successive grids (degenerate sequence)"
        return out
    if not (e21 * e32 > 0.0):
        out["reason"] = "non-monotone sequence (e21*e32 < 0): no asymptotic range"
        return out
    if abs(fs[0]) < 1e-300:
        out["reason"] = "fine-grid value is zero: GCI undefined"
        return out

    s = 1.0 if (e32 / e21) > 0 else -1.0
    p = 1.0
    converged = False
    for _ in range(200):
        q = math.log((r21**p - s) / (r32**p - s))
        p_new = abs(math.log(abs(e32 / e21)) + q) / math.log(r21)
        if not math.isfinite(p_new):
            break
        if abs(p_new - p) < 1e-12:
            p = p_new
            converged = True
            break
        p = p_new
    if not converged or not math.isfinite(p) or p <= 0.0:
        out["reason"] = "Richardson fixed point did not converge to a positive order"
        return out

    f_extrap = fs[0] + (fs[0] - fs[1]) / (r21**p - 1.0)
    e_a = abs((fs[0] - fs[1]) / fs[0]) if fs[0] != 0.0 else float("nan")
    gci = safety * e_a / abs(r21**p - 1.0)
    out.update(
        {
            "p": float(p),
            "f_extrap": float(f_extrap),
            "e_a": float(e_a),
            "gci_fine": float(gci),
            "valid": True,
            # monotone convergence is the necessary condition actually used
            # here; with non-uniform refinement ratios |R| > 1 is admissible,
            # so ``ratio_r`` is reported but not used as a filter.
            "asymptotic": bool(e21 * e32 > 0.0),
        }
    )
    return out


# ---------------------------------------------------------------------------
# Two-factor variance decomposition (criterion x grid)
# ---------------------------------------------------------------------------


def two_way_anova(table: Mapping[tuple, Sequence[float]]) -> dict:
    """Two-way ANOVA with replication on a ``(grid, criterion) -> samples`` table.

    The samples are block-averaged replicates (time blocks of the converged
    window), so the interaction term is *estimated* instead of being absorbed
    into the residual.  Returns sums of squares, degrees of freedom, mean
    squares, F statistics and the percentage of total sum of squares carried
    by each term -- the percentages the mission brief asks for.

    Raises ``ValueError`` if the table is ragged (unequal replicate counts).
    """
    grids = sorted({k[0] for k in table})
    crits = sorted({k[1] for k in table})
    n: int | None = None
    for key in table:
        m = len(table[key])
        if n is None:
            n = m
        elif m != n:
            msg = "ragged design: all cells need the same number of replicates"
            raise ValueError(msg)
    if not n:
        msg = "empty design"
        raise ValueError(msg)

    allv = [float(v) for key in table for v in table[key]]
    grand = sum(allv) / len(allv)
    ss_tot = sum((v - grand) ** 2 for v in allv)
    ss_a = 0.0
    for c in crits:
        vals = [float(v) for g in grids for v in table[(g, c)]]
        mc = sum(vals) / len(vals)
        ss_a += len(grids) * n * (mc - grand) ** 2
    ss_b = 0.0
    for g in grids:
        vals = [float(v) for c in crits for v in table[(g, c)]]
        mg = sum(vals) / len(vals)
        ss_b += len(crits) * n * (mg - grand) ** 2
    ss_ab = 0.0
    ss_e = 0.0
    for g in grids:
        for c in crits:
            vals = [float(v) for v in table[(g, c)]]
            mcell = sum(vals) / n
            ss_ab += n * (mcell - grand) ** 2
            ss_e += sum((v - mcell) ** 2 for v in vals)
    ss_ab -= ss_a + ss_b

    df_a = len(crits) - 1
    df_b = len(grids) - 1
    df_ab = df_a * df_b
    df_e = len(grids) * len(crits) * (n - 1)
    df_tot = df_a + df_b + df_ab + df_e
    ms_e = ss_e / df_e if df_e > 0 else float("nan")
    out = {
        "grids": grids,
        "criteria": crits,
        "n_replicates": n,
        "grand_mean": grand,
        "ss_total": ss_tot,
        "ss_criterion": ss_a,
        "ss_grid": ss_b,
        "ss_interaction": max(ss_ab, 0.0),
        "ss_residual": ss_e,
        "df_criterion": df_a,
        "df_grid": df_b,
        "df_interaction": df_ab,
        "df_residual": df_e,
        "df_total": df_tot,
    }
    denom = ss_tot if ss_tot > 0 else 1.0
    out["pct_criterion"] = 100.0 * out["ss_criterion"] / denom
    out["pct_grid"] = 100.0 * out["ss_grid"] / denom
    out["pct_interaction"] = 100.0 * out["ss_interaction"] / denom
    out["pct_residual"] = 100.0 * out["ss_residual"] / denom
    out["ms_criterion"] = out["ss_criterion"] / df_a
    out["ms_grid"] = out["ss_grid"] / df_b
    out["ms_interaction"] = out["ss_interaction"] / df_ab if df_ab else float("nan")
    for name in ("criterion", "grid", "interaction"):
        ms = out[f"ms_{name}"]
        out[f"F_{name}"] = (ms / ms_e) if (df_e > 0 and ms_e > 0) else float("nan")
    return out


# ---------------------------------------------------------------------------
# Additive error budget
# ---------------------------------------------------------------------------


def error_budget(
    l_base: float,
    l_criterion: float,
    l_extrap: float,
    l_reference: float,
    l_disc_fine: float | None = None,
) -> dict:
    """Telescoping split of ``L_ref - L_base`` into criterion/discretisation/residual.

    Chain used (each step answers one question):

    ``L_base``      -- a *named* base value, by default the lowest criterion
                       (the un-augmented vapour-fraction threshold column),
    ``+Delta_crit`` -- switching to the selected (optimal) criterion caliber:
                       ``Delta_crit = L_criterion - L_base``,
    ``+Delta_disc`` -- refining the grid to ``h -> 0`` at fixed criterion:
                       ``Delta_disc = L_extrap - L_criterion``,
    ``+Delta_resid``-- what is left against the reference:
                       ``Delta_resid = L_ref - L_extrap``.

    The three terms sum *exactly* to ``L_ref - L_base``; percentages are taken
    with respect to that total so that they are additive as well.
    """
    total = l_reference - l_base
    d_crit = l_criterion - l_base
    d_disc = l_extrap - l_criterion
    d_resid = l_reference - l_extrap
    scale = total if abs(total) > 1e-300 else 1.0
    out = {
        "l_base": l_base,
        "l_criterion": l_criterion,
        "l_extrap": l_extrap,
        "l_reference": l_reference,
        "delta_criterion": d_crit,
        "delta_discretisation": d_disc,
        "delta_residual": d_resid,
        "delta_total": total,
        "closure_check": (d_crit + d_disc + d_resid) - total,
        "pct_criterion": 100.0 * d_crit / scale,
        "pct_discretisation": 100.0 * d_disc / scale,
        "pct_residual": 100.0 * d_resid / scale,
    }
    if l_disc_fine is not None:
        out["l_finest_grid"] = l_disc_fine
        out["delta_finest_grid"] = l_disc_fine - l_criterion
        out["pct_finest_grid"] = 100.0 * (l_disc_fine - l_criterion) / scale
    return out


def relative_error(value: float, reference: float) -> float:
    """``|value - reference| / |reference|`` (guard for a zero reference)."""
    if reference == 0.0:
        msg = "reference must be non-zero"
        raise ValueError(msg)
    return abs(value - reference) / abs(reference)


def block_means(samples: Iterable[float], n_blocks: int) -> list[float]:
    """Split a sample sequence into ``n_blocks`` contiguous block means."""
    vals = [float(v) for v in samples]
    if n_blocks <= 0:
        msg = "n_blocks must be positive"
        raise ValueError(msg)
    if len(vals) < n_blocks:
        msg = "not enough samples for the requested number of blocks"
        raise ValueError(msg)
    edges = [round(i * len(vals) / n_blocks) for i in range(n_blocks + 1)]
    return [sum(vals[edges[i] : edges[i + 1]]) / (edges[i + 1] - edges[i]) for i in range(n_blocks)]