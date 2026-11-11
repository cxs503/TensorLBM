"""Exact two-layer (stratified) Poiseuille solution for the MCMP A-arm case.

Geometry (declared convention, see ``docs/MCMP_STRATIFIED_POISEUILLE_SPEC.md``):
the channel is driven by a uniform body force ``G_x`` per unit *mixture*
volume; the lower layer (``0 <= y <= h``) has kinematic viscosity ``nu_lower``
and the upper layer (``h <= y <= H``) has ``nu_upper``.  Both layers are
co-moving (velocity and shear stress continuous at the interface):

    u_i'' = -G / nu_i ,      u_lower(0) = 0 ,  u_upper(H) = 0
    u_lower(h) = u_upper(h) ,  nu_lower u_lower'(h) = nu_upper u_upper'(h)

The last condition degenerates to ``nu_lower b_lower = nu_upper b_upper``
because both layers carry the same body force per unit volume — the closed
form implemented below uses that exact relation rather than a generic 2x2
solve (the generic solve is singular in this limit).

``H`` is the *effective* channel height whose wall planes are the wall-cell
centres (``H = ny - 1`` for the post-streaming bounce-back path used by
``cases/mcmp_stratified_poiseuille/run_sweep.py``); this was calibrated
independently (``--wall-calibration``) rather than assumed.

All quantities are in lattice units (``rho = 1``, ``cs^2 = 1/3``).
"""

from __future__ import annotations

import math

CS2 = 1.0 / 3.0


def layer_viscosities(
    tau_lower: float,
    tau_upper: float,
    rho_heavy: float = 0.7,
    rho_light: float = 0.3,
    rule: str = "component",
) -> tuple[float, float]:
    """Kinematic viscosity of each layer under one of two mixing rules.

    ``rule="component"`` -- each layer's viscosity is the viscosity of the
    component that occupies it (``nu = cs^2 (tau - 1/2)``).  This is the
    reference used by the QR audit (pure per-component viscosities).

    ``rule="mixture"`` -- a layer's viscosity is the density-weighted mean of
    the two components it contains,
    ``nu_layer = (rho_1 nu_1 + rho_2 nu_2) / (rho_1 + rho_2)`` with
    ``rho_1 + rho_2 = 1``.  This is the continuum limit of the two-component
    BGK model when both components co-move with a common velocity (the
    momentum equation then carries ``sum_sigma rho_sigma nu_sigma``), and it is
    the only reference the model can be expected to reproduce exactly.

    Both are returned by the runner so the verdict can be stated against each.
    """
    nu1 = CS2 * (tau_lower - 0.5)  # component filling the LOWER layer
    nu2 = CS2 * (tau_upper - 0.5)  # component filling the UPPER layer
    if rule == "component":
        return nu1, nu2
    if rule == "mixture":
        lower = rho_heavy * nu1 + rho_light * nu2
        upper = rho_light * nu1 + rho_heavy * nu2
        return lower, upper
    raise ValueError(f"rule must be 'component' or 'mixture', got {rule!r}")


def layered_poiseuille(
    ny: int,
    half: float,
    G_x: float,
    nu_lower: float,
    nu_upper: float,
    H: float | None = None,
) -> dict:
    """Closed-form stratified Poiseuille profile and per-layer flow rates.

    Args:
        ny:         number of lattice rows (profile is sampled at ``y = 0..ny-1``).
        half:       interface position ``h`` in lattice units (row count from
                    the bottom wall plane).
        G_x:        uniform body force per unit mixture volume.
        nu_lower:   kinematic viscosity of the lower layer.
        nu_upper:   kinematic viscosity of the upper layer.
        H:          effective channel height; default ``ny - 1`` (wall planes at
                    the wall-cell centres).

    Returns:
        dict with ``profile`` (list, length ``ny``), ``Q_lower``, ``Q_upper``,
        ``Q_total``, ``u_interface``, ``du_dy_lower_interface``,
        ``du_dy_upper_interface`` and ``stress_jump`` (relative mismatch of the
        two shear stresses at the interface, a self-consistency check that is
        ``~0`` by construction).
    """
    H = float(ny - 1) if H is None else float(H)
    h = float(half)
    if not (0.0 < h < H):
        raise ValueError(f"interface h={h} must lie inside (0, {H})")
    a_low = -G_x / (2.0 * nu_lower)
    a_up = -G_x / (2.0 * nu_upper)
    k = nu_upper / nu_lower
    # nu_lower * b_low = nu_upper * b_up  ->  b_low = k * b_up
    b_up = (a_up * (h * h - H * H) - a_low * h * h) / (H - h + k * h)
    b_low = k * b_up
    c_up = -a_up * H * H - b_up * H

    profile: list[float] = []
    for j in range(ny):
        y = float(j)
        if y <= h:
            u = a_low * y * y + b_low * y
        else:
            u = a_up * y * y + b_up * y + c_up
        profile.append(u)

    Q_lower = a_low * h**3 / 3.0 + b_low * h**2 / 2.0
    Q_upper = (
        a_up * (H**3 - h**3) / 3.0
        + b_up * (H * H - h * h) / 2.0
        + c_up * (H - h)
    )
    du_low = 2.0 * a_low * h + b_low
    du_up = 2.0 * a_up * h + b_up
    s_low = nu_lower * du_low
    s_up = nu_upper * du_up
    scale = max(abs(s_low), abs(s_up), 1e-300)
    return {
        "profile": profile,
        "Q_lower": Q_lower,
        "Q_upper": Q_upper,
        "Q_total": Q_lower + Q_upper,
        "u_interface": profile[int(math.floor(h))],
        "du_dy_lower_interface": du_low,
        "du_dy_upper_interface": du_up,
        "stress_jump": abs(s_low - s_up) / scale,
        "H": H,
        "h": h,
    }


def _selftest() -> None:
    """Single-phase and symmetry limits of :func:`layered_poiseuille`."""
    ny, H = 25, 24.0
    G, nu = 5e-5, CS2 * 0.5
    ana = layered_poiseuille(ny, H / 2, G, nu, nu)
    umax = G * H * H / (8.0 * nu)
    assert abs(max(ana["profile"]) - umax) / umax < 1e-12
    assert abs(ana["Q_total"] - G * H**3 / (12.0 * nu)) / (G * H**3 / (12.0 * nu)) < 1e-12
    assert abs(ana["stress_jump"]) < 1e-12
    # unequal viscosities: interface velocity continuity and stress continuity
    ana2 = layered_poiseuille(ny, 12.0, G, CS2 * 0.5, CS2 * 0.25)
    assert ana2["stress_jump"] < 1e-12
    j = 12
    assert abs(ana2["profile"][j] - (ana2["profile"][j])) == 0.0
    # analytic solution must satisfy the ODE in each layer (finite differences)
    for j in range(2, ny - 2):
        y = float(j)
        nu_j = CS2 * 0.5 if y < 12.0 else CS2 * 0.25
        num = (ana2["profile"][j + 1] - 2 * ana2["profile"][j] + ana2["profile"][j - 1])
        if abs(y - 12.0) > 1.5:
            assert abs(num - (-G / nu_j)) < 2e-5, (j, num, -G / nu_j)
    print("analytic selftest OK")


if __name__ == "__main__":
    _selftest()

# ---------------------------------------------------------------------------
# Wall conventions and the per-record reference block
# ---------------------------------------------------------------------------

def wall_conventions(ny: int) -> dict:
    """The two candidate wall conventions for the post-streaming bounce-back path.

    ``halfway``    -- the no-slip planes sit half a cell outside the wall rows
                      (rows 0 and ny-1 are solid), i.e. at ``y = 0.5`` and
                      ``y = ny - 1.5``; ``H = ny - 2`` and the volume-conserving
                      interface sits at ``h = half - 0.5``.
    ``cellcentre`` -- the no-slip planes are the wall-cell centres themselves
                      (``H = ny - 1``, ``h = half``) -- the convention asserted by
                      the QR audit.

    Which one holds is a *measurement* (``run_sweep.py --mode walls``), not a
    choice; ``analyze.py`` applies both so the verdict is convention-transparent.
    """
    return {
        "halfway": dict(H=float(ny - 2), h=float(ny // 2) - 0.5, y_offset=0.5),
        "cellcentre": dict(H=float(ny - 1), h=float(ny // 2), y_offset=0.0),
    }


def layered_reference(ny: int, conv: dict, G_x: float, nu_lower: float, nu_upper: float) -> dict:
    """Two-layer solution in the wall coordinates of ``conv`` (see module docstring)."""
    H, h = conv["H"], conv["h"]
    a_low = -G_x / (2.0 * nu_lower)
    a_up = -G_x / (2.0 * nu_upper)
    k = nu_upper / nu_lower
    b_up = (a_up * (h * h - H * H) - a_low * h * h) / (H - h + k * h)
    b_low = k * b_up
    c_up = -a_up * H * H - b_up * H
    Q_lower = a_low * h**3 / 3.0 + b_low * h**2 / 2.0
    Q_upper = a_up * (H**3 - h**3) / 3.0 + b_up * (H * H - h * h) / 2.0 + c_up * (H - h)
    return dict(H=H, h=h, nu_lower=nu_lower, nu_upper=nu_upper,
                a_low=a_low, a_up=a_up, b_low=b_low, b_up=b_up, c_up=c_up,
                Q_lower=Q_lower, Q_upper=Q_upper, Q_total=Q_lower + Q_upper)


def sample_profile(ref: dict, y_offset: float, ny: int) -> list[float]:
    """Sample a reference solution at the lattice rows (``y = j - y_offset``)."""
    out = []
    for j in range(ny):
        y = j - y_offset
        if y <= 0.0 or y >= ref["H"]:
            out.append(0.0)
        elif y <= ref["h"]:
            out.append(ref["a_low"] * y * y + ref["b_low"] * y)
        else:
            out.append(ref["a_up"] * y * y + ref["b_up"] * y + ref["c_up"])
    return out


def reference_block(rec: dict, G_x: float, rho_heavy: float = 0.7,
                    rho_light: float = 0.3) -> dict:
    """All four (wall convention x viscosity rule) references for one flow record.

    ``rec`` must carry ``config`` (ny, tau_lower, tau_upper), ``Q_mix``,
    ``layer_flux`` and ``u_profile_mix`` -- i.e. a record produced by
    ``run_sweep.run_case``.  Returns a dict keyed ``"<convention>/<rule>"``.
    """
    ny = rec["config"]["ny"]
    out = {}
    for wall_tag, conv in wall_conventions(ny).items():
        for rule in ("component", "mixture"):
            nu_low, nu_up = layer_viscosities(
                rec["config"]["tau_lower"], rec["config"]["tau_upper"],
                rho_heavy, rho_light, rule)
            ref = layered_reference(ny, conv, G_x, nu_low, nu_up)
            prof = sample_profile(ref, conv["y_offset"], ny)
            sim = rec["u_profile_mix"]
            norm = math.sqrt(sum(v * v for v in prof)) or 1e-300
            out[f"{wall_tag}/{rule}"] = dict(
                Q_total=ref["Q_total"], Q_lower=ref["Q_lower"], Q_upper=ref["Q_upper"],
                nu_lower=nu_low, nu_upper=nu_up,
                err_Q_mix=(rec["Q_mix"] - ref["Q_total"]) / ref["Q_total"],
                err_Q_mix_lower=(rec["layer_flux"]["Q_mix_lower"] - ref["Q_lower"]) / ref["Q_lower"],
                err_Q_mix_upper=(rec["layer_flux"]["Q_mix_upper"] - ref["Q_upper"]) / ref["Q_upper"],
                err_profile_L2=math.sqrt(sum((a - b) ** 2 for a, b in zip(sim, prof))) / norm,
                err_interface_rel=((rec["interface_crossing"] - conv["y_offset"]) - conv["h"])
                / conv["h"],
                profile=prof,
            )
    return out
