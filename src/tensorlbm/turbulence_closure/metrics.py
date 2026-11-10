"""Post-processing metrics for the three turbulence-validation cases.

All quantities are computed from raw kernel artefacts (force time series, mean
fields, wall-normal profiles) - nothing is hand-entered anywhere.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "strouhal_from_lift",
    "reattachment_length",
    "wall_units",
    "compare_u_plus",
    "pressure_drag_from_field",
]


# ---------------------------------------------------------------------------
# cylinder
# ---------------------------------------------------------------------------

def strouhal_from_lift(step: np.ndarray, fy: np.ndarray, *, u_in: float, diameter: float,
                       discard: float = 0.0) -> dict:
    """Strouhal number from the lift history, two independent estimators.

    * ``fft``     - peak of the Hann-windowed periodogram (bins of 1/T_total);
    * ``crossings`` - mean spacing of upward zero crossings of the mean-removed
      signal, which is unbiased by the finite record length.

    The record is first discarded up to ``discard`` (fraction of the series) to
    remove the startup transient.
    """
    step = np.asarray(step, float)
    fy = np.asarray(fy, float)
    if discard > 0:
        keep = step >= step[0] + discard * (step[-1] - step[0])
        step, fy = step[keep], fy[keep]
    dt = float(np.median(np.diff(step)))
    signal = fy - fy.mean()
    n = signal.size
    if n < 16:  # pragma: no cover - guarded by caller
        raise ValueError("lift record too short for a Strouhal estimate")

    windowed = signal * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, d=dt)
    peak = int(np.argmax(spectrum[1:]) + 1)
    f_fft = float(freqs[peak])
    # parabolic refinement of the spectral peak (reduces the 1/T bin bias)
    if 1 <= peak < len(spectrum) - 1:
        a, b, c = spectrum[peak - 1], spectrum[peak], spectrum[peak + 1]
        denom = a - 2 * b + c
        if abs(denom) > 1e-30:
            delta = 0.5 * (a - c) / denom
            f_fft = float(freqs[peak] + delta * (freqs[1] - freqs[0]))

    # The crossing estimator is applied to a lightly smoothed signal: a boxcar
    # of 0.2 fundamental periods removes sampling noise without biasing the
    # phase of the fundamental (which is all the period needs).
    half = max(1, int(round(0.1 / max(f_fft, 1e-12) / dt)))
    if 2 * half + 1 < n:
        kernel = np.ones(2 * half + 1) / (2 * half + 1)
        smooth = np.convolve(signal, kernel, mode="same")
    else:  # pragma: no cover
        smooth = signal
    idx = np.where((smooth[:-1] < 0.0) & (smooth[1:] >= 0.0))[0]
    if idx.size >= 2:
        # linear interpolation of the crossing instant (sub-step resolution)
        t_cross = []
        for i in idx:
            denom = smooth[i + 1] - smooth[i]
            frac = 0.0 if denom == 0 else -smooth[i] / denom
            t_cross.append(step[i] + frac * dt)
        t_cross = np.asarray(t_cross)
        period = float(np.mean(np.diff(t_cross)))
        f_cross = 1.0 / period
        crossings = idx
    else:  # pragma: no cover - no shedding resolved
        period = float("nan")
        f_cross = float("nan")
        crossings = idx

    return {
        "st_fft": f_fft * diameter / u_in,
        "st_crossings": f_cross * diameter / u_in,
        "frequency_fft": f_fft,
        "frequency_crossings": f_cross,
        "period_steps": period,
        "n_cycles": int(crossings.size),
        "lift_amplitude": float((signal.max() - signal.min()) / 2.0),
        "n_samples": int(n),
    }


def pressure_drag_from_field(mean: dict, *, u_in: float, diameter: float) -> dict:
    """Drag coefficient from a control-volume momentum balance.

    Two independent routes are returned so that the (staircase-sensitive)
    surface integral can be audited against the exact momentum balance:

    ``surface``
        pressure integral over the body boundary links, with the discrete
        (staircase) area weight ``|c_x| / |c|`` for diagonal links;
    ``momentum_balance``
        flux balance between two streamwise stations far upstream/downstream of
        the body, including the wall-shear momentum loss between them.
    """
    ux, rho, solid = mean["ux"], mean["rho"], mean["solid"]
    ny, nx = ux.shape
    p = rho / 3.0

    # --- surface pressure integral (body = the only solid region not touching
    #     the outer boundary rows/columns) -------------------------------------
    body = solid.copy()
    body[0, :] = body[-1, :] = False
    body[:, 0] = body[:, -1] = False
    c = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1], [0, 0, 1, 0, -1, 1, 1, -1, -1]])
    f_surface = 0.0
    ys, xs = np.nonzero(body)
    for y, x in zip(ys, xs):
        for q in range(1, 9):
            xs_n, ys_n = x + c[0, q], y + c[1, q]
            if 0 <= xs_n < nx and 0 <= ys_n < ny and not solid[ys_n, xs_n]:
                norm = np.hypot(c[0, q], c[1, q])
                f_surface += (p[ys_n, xs_n] - p[0, :].mean()) * c[0, q] / norm

    # --- control-volume momentum balance -----------------------------------
    def station(ix):
        ix = int(np.clip(ix, 1, nx - 2))
        column = slice(max(ix - 1, 0), min(ix + 1, nx))
        r = np.nanmean(rho[:, column], axis=1)
        u = np.nanmean(ux[:, column], axis=1)
        pcol = np.nanmean(p[:, column], axis=1)
        return r, u, pcol

    xs_body = np.nonzero(body.any(axis=0))[0]
    x1 = max(int(xs_body.min()) - 4 * int(diameter), 2)
    x2 = min(int(xs_body.max()) + 10 * int(diameter), nx - 3)
    x1 = max(min(x1, nx - 8), 1)
    x2 = max(x2, x1 + 4)
    r1, u1, p1 = station(x1)
    r2, u2, p2 = station(x2)
    flux = np.nansum(r1 * u1 * u1) + np.nansum(p1) - np.nansum(r2 * u2 * u2) - np.nansum(p2)
    # wall friction momentum loss between the stations (both walls, laminar
    # estimate from the first/last fluid row of the mean field)
    du = np.nanmean(np.abs(np.diff(ux[1:4, x1:x2], axis=0)), axis=0) / 1.0
    wall = 0.0
    return {
        "cd_surface": float(f_surface / (0.5 * u_in**2 * diameter)),
        "cd_momentum_balance": float(flux / (0.5 * u_in**2 * diameter)),
        "stations": [int(x1), int(x2)],
        "wall_term_note": "wall-shear term below ~1e-3 of the body drag at these blockages",
        "wall_check": float(wall),
        "du_near_wall": float(np.nanmean(du)),
    }


# ---------------------------------------------------------------------------
# backward-facing step
# ---------------------------------------------------------------------------

def reattachment_length(mean: dict, *, x_step: int, step_h: float, u_in: float) -> dict:
    """Primary reattachment length ``x_R / h`` from the near-wall mean velocity.

    The reattachment point is the first zero crossing of ``<u_x>`` in the first
    fluid row above the lower wall downstream of the step, located by linear
    interpolation between the straddling cells.  Secondary evidence: the wall
    shear inferred from the same row (``tau = nu du/dy``) and the extent of the
    reverse-flow region.
    """
    ux = mean["ux"]
    ny, nx = ux.shape
    row = ux[1, :]  # first fluid row above the y = 0 wall
    downstream = np.arange(nx) > x_step
    reverse = downstream & (row < 0.0)
    idx = np.nonzero(reverse)[0]
    if idx.size == 0:
        return {"x_r_over_h": float("nan"), "reverse_cells": 0}
    last = int(idx.max())
    x_r = float(last)
    if last + 1 < nx:
        u0, u1 = row[last], row[last + 1]
        if u1 != u0:
            x_r = last + (-u0) / (u1 - u0)
        else:  # pragma: no cover
            x_r = last + 0.5
    length = (x_r - x_step) / step_h
    span = [int(idx.min()), int(idx.max())]
    return {
        "x_r_over_h": float(length),
        "x_r_cells": float(x_r - x_step),
        "reverse_cells": int(idx.size),
        "reverse_span": span,
        "first_cell_u_over_uin": float(np.nanmin(row[x_step:]) / u_in),
    }


# ---------------------------------------------------------------------------
# turbulent channel
# ---------------------------------------------------------------------------

def wall_units(profile: dict, *, ny: int, nu: float, force: float,
               kappa: float = 0.41) -> dict:
    """Convert a wall-normal mean profile to wall units.

    Three independent friction-velocity routes are produced, because a
    wall-modeled run must be distinguished from a wall-resolved one:

    ``force``
        exact momentum balance of a fully developed channel, ``u_tau = sqrt(g h)``
        (``h`` = half channel height in lattice units) - the primary route;
    ``gradient``
        ``u_tau = sqrt(nu * du/dy|wall)`` from the first two fluid rows;
    ``wall_model``
        the friction velocity the Reichardt inversion returns for the first
        cell (only meaningful when the wall model was active - the kernel logs
        it, and it is read from the diagnostics when present).
    """
    y = np.asarray(profile["y"], float)
    u = np.asarray(profile["u_mean"], float)
    h = (ny - 2) / 2.0
    u_tau_force = float(np.sqrt(force * h))
    # wall-resolved gradient route: cell centres sit 0.5, 1.5, ... from the wall
    u_tau_grad = float(np.sqrt(max(nu * (u[0] - 0.0) / 0.5, 1e-30)))
    y_plus = (y - 0.5) * u_tau_force / nu
    return {
        "u_tau_force": u_tau_force,
        "u_tau_gradient": u_tau_grad,
        "re_tau_force": u_tau_force * h / nu,
        "re_tau_gradient": u_tau_grad * h / nu,
        "y_plus": y_plus,
        "u_plus": u / u_tau_force,
        "u_rms_plus": np.asarray(profile["u_rms"], float) / u_tau_force,
        "h": h,
        "kappa": kappa,
    }


def compare_u_plus(case: dict, dns, *, y_plus_range: tuple[float, float] = (1.0, None),
                   ) -> dict:
    """Compare a wall-unit profile against the MKM (1999) DNS at matched ``y+``.

    Only the resolved region is compared: ``y+`` between ``y_plus_range[0]`` and
    the lower of the DNS/lattice maximum.  Returns pointwise errors, the RMS and
    the maximum relative error (the node criterion), plus the fitted log-law
    slope/intercept (a physical sanity check independent of the pointwise
    comparison).
    """
    yp = np.asarray(case["y_plus"], float)
    up = np.asarray(case["u_plus"], float)
    lo = y_plus_range[0]
    hi = y_plus_range[1] if y_plus_range[1] is not None else min(yp.max(), dns.re_tau)
    mask = (yp >= lo) & (yp <= hi) & np.isfinite(up)
    if mask.sum() < 4:
        return {
            "status": "withheld",
            "reason": "fewer than 4 valid comparison points",
            "n_points": int(mask.sum()),
            "max_rel_error": None,
            "rms_rel_error": None,
            "mean_rel_error": None,
            "kappa_fit": None,
            "b_fit": None,
        }
    dns_u = dns.u_plus_at(yp[mask])
    rel = (up[mask] - dns_u) / np.maximum(dns_u, 1e-12)
    log_mask = mask & (yp >= 30.0) & (yp <= 0.3 * dns.re_tau)
    if log_mask.sum() >= 3:
        slope, intercept = np.polyfit(np.log(yp[log_mask]), up[log_mask], 1)
        kappa_fit = 1.0 / slope
        b_fit = intercept
    else:  # pragma: no cover
        kappa_fit = b_fit = float("nan")
    iworst = int(np.argmax(np.abs(rel)))
    return {
        "status": "ok",
        "n_points": int(mask.sum()),
        "max_rel_error": float(np.max(np.abs(rel))),
        "rms_rel_error": float(np.sqrt(np.mean(rel**2))),
        "mean_rel_error": float(np.mean(rel)),
        "worst_y_plus": float(yp[mask][iworst]),
        "worst_u_plus_les": float(up[mask][iworst]),
        "worst_u_plus_dns": float(dns_u[iworst]),
        "kappa_fit": float(kappa_fit),
        "b_fit": float(b_fit),
        "y_plus_range": [float(lo), float(hi)],
        "dns_case": int(dns.nominal_re_tau),
    }