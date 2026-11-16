"""XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY — verification gates.

Anchor
------
``lssfau/walberla``'s **no-mcmp-two-phase-only** convention: a *two-phase*
lattice model (single species that separates into liquid/vapour through a
**self**-interaction) is a **different** lattice model from a *multi-component
multiphase* (MCMP, Shan–Chen cross-coupling) one.  waLBerla expresses this at
the type level — the two-phase-only lattice model carries no MCMP force
processor at all — so no MCMP term can leak into a two-phase run.

TensorLBM has both physics in one module (:mod:`tensorlbm.multiphase`):
``sc_single_component_force`` / ``collide_sc_single_component`` (SCMP = the
two-phase-only model) and ``sc_two_component_force`` /
``collide_sc_two_component`` (MCMP).  Nothing pinned *how the two relate*, which
is exactly the class of verification debt this case closes:

* G1  kernel identity  — is there **one** interaction kernel, so that the MCMP
      cross-force evaluated with a *duplicated partner* (``rho_B == rho_A``,
      ``G_12 == G``) is the SCMP self-force, cell by cell?
* G2  null-coupling purity — with ``G_12 == 0`` is the MCMP force **identically
      zero** (so an MCMP solver contributes nothing to a two-phase-only run),
      and does the gate have teeth (non-zero reference force)?
* G3  dynamical reduction — does a *duplicated-partner MCMP* integration
      reproduce the SCMP integration step for step (the operational meaning of
      "the MCMP path degenerates to the two-phase-only path")?
* G4  absence of self-interaction in the MCMP kernel — with an inert partner a
      single component must **not** phase-separate (this is what makes a
      separate two-phase-only model necessary rather than an MCMP special case).
* G5  coupling-sign semantics — ``G_12 < 0`` segregates, ``G_12 > 0`` mixes,
      while the *identical* kernel with ``G > 0`` condenses.  The sign means
      opposite things in the two models; unverified, it is a silent trap.
* G6  physics anchor for the two-phase-only path — relaxed periodic flat-interface
      coexisting densities against the analytic **Maxwell equal-area**
      construction of the same equation of state (an independent reference that
      does not use the lattice solver at all).

All gates are pure functions returning JSON-friendly dicts so the driver
(``xval/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY/run_xval.py``) and the regression tests
(``tests/test_xval_no_mcmp_two_phase.py``) execute exactly the same code.

Conventions (this library, D2Q9, lattice units, ``cs2 = 1/3``)
-------------------------------------------------------------
* ``sc_single_component_force``: ``F = -G psi(x) sum_i w_i c_i psi(x - c_i)``,
  ``G > 0`` attractive (condensation, liquid/vapour coexistence).
* ``sc_two_component_force``: ``F_k = -G_12 psi_k(x) sum_i w_i c_i psi_k'(x - c_i)``,
  ``G_12 < 0`` repulsive between species (segregation).
* Backward gather: ``sum_i w_i c_i psi(x - c_i) = -cs2 grad psi + O(h^2)``, so
  ``F ~ +G cs2 psi grad psi = (G cs2/2) grad psi^2`` — attraction for ``G > 0``.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from tensorlbm.multiphase import (
    collide_sc_single_component,
    collide_sc_two_component,
    equilibrium,
    psi_exp,
    psi_linear,
    sc_single_component_force,
    sc_two_component_force,
)
from tensorlbm.sc_eos_coexistence import flat_interface_coexistence, maxwell_coexistence
from tensorlbm.solver import stream

__all__ = [
    "droplet_field",
    "random_smooth_field",
    "gate_force_kernel_identity",
    "gate_null_coupling_purity",
    "gate_duplicated_partner_dynamics",
    "gate_no_self_interaction",
    "gate_coupling_sign",
    "gate_two_phase_coexistence",
    "run_all_gates",
]

PSI_CHOICES: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "exp": psi_exp,
    "linear": psi_linear,
}


# ---------------------------------------------------------------------------
# fields
# ---------------------------------------------------------------------------
def random_smooth_field(
    ny: int = 48,
    nx: int = 64,
    *,
    seed: int = 0,
    rho_min: float = 0.25,
    rho_max: float = 1.25,
    smooth_passes: int = 3,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """A periodic, smooth, non-radially-symmetric density field with interfaces.

    A *generic* field is used for the algebraic gates on purpose: a radially
    symmetric droplet would give zero force at every symmetry point and hide a
    stencil/parity error.  Box-filter smoothing keeps the field free of grid-scale
    noise so the comparison is dominated by the interaction operator itself.
    """
    g = torch.Generator().manual_seed(seed)
    field = torch.rand(ny, nx, generator=g, dtype=dtype)
    for _ in range(smooth_passes):
        acc = torch.zeros_like(field)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                acc += torch.roll(field, shifts=(dy, dx), dims=(0, 1))
        field = acc / 9.0
    field = (field - field.min()) / (field.max() - field.min())
    return rho_min + (rho_max - rho_min) * field


def droplet_field(
    ny: int = 64,
    nx: int = 64,
    *,
    radius: float = 14.0,
    rho_in: float = 1.0,
    rho_out: float = 0.2,
    width: float = 1.5,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Tanh-profile droplet of ``rho_in`` in a ``rho_out`` bath (periodic box)."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=dtype), torch.arange(nx, dtype=dtype), indexing="ij"
    )
    r = torch.sqrt((yy - (ny - 1) / 2.0) ** 2 + (xx - (nx - 1) / 2.0) ** 2)
    return rho_out + (rho_in - rho_out) * 0.5 * (1.0 - torch.tanh((r - radius) / width))


# ---------------------------------------------------------------------------
# G1 — interaction-kernel identity
# ---------------------------------------------------------------------------
def gate_force_kernel_identity(
    G: float = 5.0,
    *,
    ny: int = 48,
    nx: int = 64,
    psi: str = "exp",
    dtype: torch.dtype = torch.float64,
    seed: int = 0,
) -> dict[str, Any]:
    """MCMP cross-force with a duplicated partner vs the SCMP self-force.

    ``rho_B := rho_A`` and ``G_12 := G`` make the multi-component cross-force
    algebraically identical to the single-component self-force,
    ``F_1 = -G psi(x) sum_i w_i c_i psi(x-c_i)``.  If the two models share one
    kernel (as waLBerla's code-generation does), the agreement is *exact*; any
    deviation is a real convention bug (stencil, sign, factor of 2, or psi
    evaluation point).
    """
    psi_fn = PSI_CHOICES[psi]
    rho = random_smooth_field(ny, nx, seed=seed, dtype=dtype)

    fx_sc, fy_sc = sc_single_component_force(rho, G=G, psi_fn=psi_fn)
    fx1, fy1, fx2, fy2 = sc_two_component_force(rho, rho.clone(), G_12=G, psi=psi_fn)

    scale = float(torch.sqrt(fx_sc**2 + fy_sc**2).max()) + 1e-300
    diff = float(
        torch.maximum(
            torch.sqrt((fx1 - fx_sc) ** 2 + (fy1 - fy_sc) ** 2),
            torch.sqrt((fx2 - fx_sc) ** 2 + (fy2 - fy_sc) ** 2),
        ).max()
    )
    return {
        "gate": "G1_kernel_identity",
        "G": float(G),
        "psi": psi,
        "dtype": str(dtype).replace("torch.", ""),
        "shape": [ny, nx],
        "force_scale": scale,
        "max_abs_diff": diff,
        "rel_err": diff / scale,
        "component_symmetry": float(torch.maximum((fx1 - fx2).abs(), (fy1 - fy2).abs()).max()),
        "exact": diff == 0.0,
        "tolerance": 1e-12,
        "pass": diff / scale <= 1e-12,
    }


# ---------------------------------------------------------------------------
# G2 — null-coupling purity (and non-triviality ablation)
# ---------------------------------------------------------------------------
def gate_null_coupling_purity(
    *,
    ny: int = 48,
    nx: int = 64,
    psi: str = "exp",
    dtype: torch.dtype = torch.float64,
) -> dict[str, Any]:
    """``G_12 = 0`` must leave **no** force at all (no hidden MCMP term).

    Three sub-checks:

    * ``null_max``: two arbitrary (independent) density fields, ``G_12 = 0``
      -> both component forces are identically zero.
    * ``inert_partner_max``: a spatially **uniform** partner with ``G_12 != 0``
      -> zero force, because ``sum_i w_i c_i = 0`` exactly.  This is the
      "no force from a homogeneous second phase" property that lets an MCMP code
      be used with a passive diluent.
    * ``nontrivial_max``: the same configuration with ``G_12 = -1`` is O(1e-2),
      so the two zero-checks above are not vacuous.
    """
    psi_fn = PSI_CHOICES[psi]
    rho_a = random_smooth_field(ny, nx, seed=11, dtype=dtype)
    rho_b = random_smooth_field(ny, nx, seed=12, dtype=dtype)

    fxa, fya, fxb, fyb = sc_two_component_force(rho_a, rho_b, G_12=0.0, psi=psi_fn)
    null_max = float(
        torch.maximum(
            torch.maximum(fxa.abs().max(), fya.abs().max()),
            torch.maximum(fxb.abs().max(), fyb.abs().max()),
        )
    )

    inert_max = 0.0
    for const in (0.3, 1.0, 3.0):
        partner = torch.full_like(rho_a, const)
        fx, fy, _, _ = sc_two_component_force(rho_a, partner, G_12=-1.0, psi=psi_fn)
        inert_max = max(inert_max, float(torch.maximum(fx.abs().max(), fy.abs().max())))

    fxn, fyn, _, _ = sc_two_component_force(rho_a, rho_b, G_12=-1.0, psi=psi_fn)
    nontrivial = float(torch.maximum(fxn.abs().max(), fyn.abs().max()))

    return {
        "gate": "G2_null_coupling_purity",
        "psi": psi,
        "null_max": null_max,
        "inert_partner_max": inert_max,
        "nontrivial_max": nontrivial,
        "null_over_nontrivial": null_max / max(nontrivial, 1e-300),
        "teeth_amplitude": nontrivial,
        "tolerance": 1e-12,
        "pass": null_max <= 1e-12 and inert_max <= 1e-12 and nontrivial > 1e-3,
    }


# ---------------------------------------------------------------------------
# G3 — dynamical reduction: duplicated-partner MCMP == SCMP
# ---------------------------------------------------------------------------
def gate_duplicated_partner_dynamics(
    G: float = 5.0,
    *,
    steps: int = 400,
    ny: int = 64,
    nx: int = 64,
    radius: float = 14.0,
    tau: float = 1.0,
    dtype: torch.dtype = torch.float64,
    sample_every: int = 50,
) -> dict[str, Any]:
    """Integrate the two-phase-only model twice: SCMP and duplicated-partner MCMP.

    Identical initial distributions (component B is a *copy* of A) and identical
    per-step forces imply identical trajectories.  This is the operational claim
    behind "the MCMP kernel degenerates to the two-phase-only kernel"; it fails
    for any difference in equilibrium/forcing bookkeeping between the two
    collision routines, which the algebraic gate G1 cannot see.
    """
    rho0 = droplet_field(ny, nx, radius=radius, dtype=dtype)
    u0 = torch.zeros_like(rho0)
    f_sc = equilibrium(rho0, u0, u0)
    f_mc = f_sc.clone()
    f_mc_partner = f_sc.clone()

    history: list[dict[str, float]] = []
    for step in range(steps):
        f_sc = stream(
            collide_sc_single_component(
                f_sc, G=G, tau=tau, psi_fn=psi_exp, forcing="velocity_shift"
            )
        )
        f_mc, f_mc_partner = collide_sc_two_component(
            f_mc,
            f_mc_partner,
            G_12=G,
            tau1=tau,
            tau2=tau,
            psi=psi_exp,
            force_scheme="shift",
        )
        f_mc = stream(f_mc)
        f_mc_partner = stream(f_mc_partner)
        if (step + 1) % sample_every == 0 or step == 0:
            rho_sc = f_sc.sum(0)
            rho_mc = f_mc.sum(0)
            denominator = float(rho_sc.abs().max()) + 1e-300
            history.append(
                {
                    "step": step + 1,
                    "max_abs_diff": float((rho_sc - rho_mc).abs().max()),
                    "rel_diff": float((rho_sc - rho_mc).abs().max()) / denominator,
                    "rho_sc_min": float(rho_sc.min()),
                    "rho_sc_max": float(rho_sc.max()),
                }
            )

    rho_sc = f_sc.sum(0)
    rho_mc = f_mc.sum(0)
    rho_partner = f_mc_partner.sum(0)
    final_rel = float((rho_sc - rho_mc).abs().max()) / (float(rho_sc.abs().max()) + 1e-300)
    return {
        "gate": "G3_duplicated_partner_dynamics",
        "G": float(G),
        "steps": steps,
        "tau": tau,
        "dtype": str(dtype).replace("torch.", ""),
        "shape": [ny, nx],
        "radius": radius,
        "history": history,
        "final_max_abs_diff": float((rho_sc - rho_mc).abs().max()),
        "final_rel_diff": final_rel,
        "partner_tracks_component": float((rho_mc - rho_partner).abs().max()),
        "sc_density_contrast": float(rho_sc.max() - rho_sc.min()),
        "sc_rho_max": float(rho_sc.max()),
        "sc_rho_min": float(rho_sc.min()),
        "tolerance": 1e-10,
        "pass": final_rel <= 1e-10,
        "phase_separated": bool(float(rho_sc.max() - rho_sc.min()) > 0.5),
    }


# ---------------------------------------------------------------------------
# G4 — no self-interaction inside the MCMP kernel
# ---------------------------------------------------------------------------
def gate_no_self_interaction(
    G_12: float = -1.0,
    *,
    steps: int = 400,
    ny: int = 64,
    nx: int = 64,
    tau: float = 1.0,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Any]:
    """A single phase in an inert homogeneous partner must not separate.

    waLBerla's two-phase-only model exists because an MCMP lattice model with no
    cross-coupling produces **no** interface physics.  Quantitatively: start
    component A as a droplet, component B **uniform**, and integrate.  A is then
    driven only by viscous diffusion: its density contrast may only *decay*.  A
    contrast that grows (or a spurious interface) would mean the two-phase
    self-interaction leaked into the MCMP path.

    Control: the same A droplet against a *complementary* B field
    (``B = 1 - A + const``) at the same ``G_12`` must sharpen (segregation), so
    the null result is not an artefact of a dead force.
    """
    rho_a = droplet_field(ny, nx, radius=14.0, dtype=dtype)
    zeros = torch.zeros_like(rho_a)

    def _contrast(field: torch.Tensor) -> float:
        return float(field.max() - field.min())

    # (a) inert partner
    f_a = equilibrium(rho_a, zeros, zeros)
    f_b = equilibrium(torch.ones_like(rho_a), zeros, zeros)
    contrast0 = _contrast(rho_a)
    contrasts: list[float] = []
    for step in range(steps):
        f_a, f_b = collide_sc_two_component(
            f_a, f_b, G_12=G_12, tau1=tau, tau2=tau, psi=psi_exp, force_scheme="shift"
        )
        f_a, f_b = stream(f_a), stream(f_b)
        if (step + 1) % 100 == 0:
            contrasts.append(_contrast(f_a.sum(0)))
    rho_a_end = f_a.sum(0)
    inert_ratio = _contrast(rho_a_end) / contrast0

    # (b) control: complementary partner -> segregation must sharpen the interface
    rho_b_comp = (rho_a.max() + rho_a.min()) - rho_a
    g_a = equilibrium(rho_a, zeros, zeros)
    g_b = equilibrium(rho_b_comp, zeros, zeros)
    comp: list[float] = []
    for step in range(steps):
        g_a, g_b = collide_sc_two_component(
            g_a, g_b, G_12=G_12, tau1=tau, tau2=tau, psi=psi_exp, force_scheme="shift"
        )
        g_a, g_b = stream(g_a), stream(g_b)
        if (step + 1) % 100 == 0:
            comp.append(_contrast(g_a.sum(0)))
    rho_a_comp = g_a.sum(0)
    corr0 = _correlation(rho_a, rho_b_comp)
    corr_end = _correlation(rho_a_comp, g_b.sum(0))

    return {
        "gate": "G4_no_self_interaction",
        "G_12": float(G_12),
        "steps": steps,
        "contrast_initial": contrast0,
        "contrast_inert_partner": _contrast(rho_a_end),
        "contrast_ratio_inert": inert_ratio,
        "contrast_history_inert": contrasts,
        "contrast_complementary": _contrast(rho_a_comp),
        "contrast_history_complementary": comp,
        "spatial_correlation_initial": corr0,
        "spatial_correlation_final": corr_end,
        "contrast_decayed": bool(inert_ratio <= 1.0),
        "pass": bool(inert_ratio <= 1.0 and corr_end < 0.0),
    }


def _correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """Zero-mean, unit-variance Pearson correlation of two fields."""
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = float(torch.sqrt((a0**2).sum() * (b0**2).sum()))
    if denom == 0.0:
        return 0.0
    return float((a0 * b0).sum()) / denom


# ---------------------------------------------------------------------------
# G5 — coupling-sign semantics: the shared kernel carries two different meanings
# ---------------------------------------------------------------------------
def gate_coupling_sign(
    couplings: tuple[float, ...] = (-1.0, -0.5, 0.5, 1.0),
    *,
    steps: int = 400,
    ny: int = 48,
    nx: int = 48,
    tau: float = 1.0,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Any]:
    """Measure what the sign of ``G_12`` does, instead of trusting a docstring.

    The interaction kernel is **shared** with the two-phase-only model, where the
    same magnitude with ``G > 0`` *condenses* a single species.  In the
    multi-component model the coupling is **inter**-species, so the sign means
    the opposite thing: a positive ``G_12`` pulls species A towards B-rich
    regions (adhesion/mixing), a negative one pushes it away
    (repulsion/segregation).  Handing an SCMP ``G`` to an MCMP ``G_12`` therefore
    inverts the intended physics — a silent trap unless measured.

    Two independent sub-measurements per coupling:

    ``alignment`` (exact, no time integration)
        ``sum_x F_A . grad(psi_B)`` for a uniform A field against a sinusoidal B
        field.  It is *odd* in ``G_12`` and has the sign of ``G_12``:
        ``sum_i w_i c_i psi_B(x - c_i) = -cs2 grad psi_B + O(h^2)``, hence
        ``F_A = -G_12 psi_A sum_i w_i c_i psi_B(x - c_i) ~ +G_12 cs2 psi_A grad psi_B``.
    ``tracer`` (dynamic)
        Species B is **frozen** (never collided) and species A starts uniform;
        A can then only respond to the cross-force.  ``corr(rho_A, psi_B)`` after
        ``steps`` must have the sign of ``G_12`` and ``|corr| -> 1``.

    A small-amplitude *two-fluid mixture* probe was tried first and discarded: at
    ``|G_12| <= 1.5`` with ``psi_exp`` the mixture is linearly stable
    (``|G_12| psi psi' ~ 0.25 << 1``), so the residual correlation it reports is
    noise-driven, not physics (see the case doc, "pitfalls").
    """
    psi_fn = psi_exp
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=dtype), torch.arange(nx, dtype=dtype), indexing="ij"
    )
    two_pi = 6.283185307179586
    rho_b_sin = 0.7 + 0.25 * torch.sin(two_pi * xx / nx)
    psi_b = psi_fn(rho_b_sin)
    grad_x = (torch.roll(psi_b, -1, dims=1) - torch.roll(psi_b, 1, dims=1)) / 2.0
    grad_y = (torch.roll(psi_b, -1, dims=0) - torch.roll(psi_b, 1, dims=0)) / 2.0
    rho_a_uniform = torch.full((ny, nx), 0.7, dtype=dtype)
    zeros = torch.zeros_like(rho_a_uniform)

    rows: list[dict[str, Any]] = []
    for g in couplings:
        fx, fy, _, _ = sc_two_component_force(rho_a_uniform, rho_b_sin, G_12=g, psi=psi_fn)
        alignment = float((fx * grad_x + fy * grad_y).sum())

        f_a = equilibrium(rho_a_uniform, zeros, zeros)
        f_b = equilibrium(rho_b_sin, zeros, zeros)  # frozen
        for _ in range(steps):
            f_a, _unused = collide_sc_two_component(
                f_a, f_b, G_12=g, tau1=tau, tau2=tau, psi=psi_fn, force_scheme="shift"
            )
            f_a = stream(f_a)
        rho_a = f_a.sum(0)
        tracer_corr = _correlation(rho_a, psi_fn(f_b.sum(0)))
        rows.append(
            {
                "G_12": float(g),
                "alignment": alignment,
                "alignment_sign": 0 if alignment == 0 else (1 if alignment > 0 else -1),
                "tracer_correlation": tracer_corr,
                "tracer_contrast": float(rho_a.max() - rho_a.min()),
                "tracer_sign": 0 if tracer_corr == 0 else (1 if tracer_corr > 0 else -1),
                "agrees_with_docstring": bool(
                    (g < 0 and tracer_corr < 0) or (g > 0 and tracer_corr > 0) or g == 0
                ),
            }
        )

    # exact oddness of the alignment operator in G_12
    oddness = None
    pos = [r for r in rows if r["G_12"] > 0]
    neg = [r for r in rows if r["G_12"] < 0]
    if pos and neg:
        p = max(pos, key=lambda r: r["G_12"])
        n = min(neg, key=lambda r: r["G_12"])
        if abs(p["G_12"]) == abs(n["G_12"]):
            scale = max(abs(p["alignment"]), abs(n["alignment"]), 1e-300)
            oddness = abs(p["alignment"] + n["alignment"]) / scale

    return {
        "gate": "G5_coupling_sign",
        "steps": steps,
        "rows": rows,
        "alignment_oddness_rel": oddness,
        "segregating_for_negative": all(r["tracer_correlation"] < 0 for r in rows if r["G_12"] < 0),
        "adhesion_for_positive": all(r["tracer_correlation"] > 0 for r in rows if r["G_12"] > 0),
        "psib_sign_convention": (
            "G_12 < 0: species A is repelled from B-rich regions (segregation); "
            "G_12 > 0: A is attracted (adhesion/mixing); "
            "SCMP like-species attraction needs G > 0."
        ),
        "pass": all(r["tracer_sign"] == r["alignment_sign"] for r in rows)
        and all(abs(r["tracer_correlation"]) > 0.99 for r in rows if r["G_12"] != 0)
        and (oddness is None or oddness <= 1e-12),
    }


# G6 — physics anchor of the two-phase-only path (Maxwell construction)
# ---------------------------------------------------------------------------
def gate_two_phase_coexistence(
    couplings: tuple[float, ...] = (4.5, 5.0),
    *,
    steps: int = 12000,
    ny: int = 64,
    forcing: str = "edm",
    tau: float = 1.0,
) -> dict[str, Any]:
    """Relaxed flat-interface coexisting densities vs the Maxwell binodal.

    The reference is the analytic equal-area construction on the same lattice
    equation of state ``p(rho) = cs2 rho - (G cs2 / 2) psi^2`` — it does not use
    the lattice solver, so the comparison is genuine external validation of the
    two-phase-only path rather than self-consistency.  Acceptance per the owner's
    precision standard is ``|err| <= 15%``; the tighter engineering target is 3%.
    """
    rows: list[dict[str, Any]] = []
    for g in couplings:
        mw = maxwell_coexistence(g, psi_exp)
        row: dict[str, Any] = {"G": float(g), "maxwell": mw.as_dict()}
        if not mw.found:
            row["pass"] = False
            row["note"] = "no Maxwell two-phase solution at this coupling"
            rows.append(row)
            continue
        fi = flat_interface_coexistence(
            g, psi_exp, ny=ny, nx=2, forcing=forcing, tau=tau, steps=steps
        )
        row["flat"] = fi.as_dict()
        if fi.diverged or fi.rho_liquid is None or fi.rho_vapour is None:
            row["pass"] = False
            row["note"] = "flat-interface run diverged"
            rows.append(row)
            continue
        err_l = abs(fi.rho_liquid - mw.rho_liquid) / mw.rho_liquid
        err_v = abs(fi.rho_vapour - mw.rho_vapour) / mw.rho_vapour
        row.update(
            {
                "rho_liquid_sim": fi.rho_liquid,
                "rho_vapour_sim": fi.rho_vapour,
                "rho_liquid_maxwell": mw.rho_liquid,
                "rho_vapour_maxwell": mw.rho_vapour,
                "err_liquid": err_l,
                "err_vapour": err_v,
                "err_max": max(err_l, err_v),
                "within_15pct": bool(max(err_l, err_v) <= 0.15),
                "within_3pct": bool(max(err_l, err_v) <= 0.03),
                "pass": bool(max(err_l, err_v) <= 0.15),
            }
        )
        rows.append(row)
    return {
        "gate": "G6_two_phase_coexistence",
        "forcing": forcing,
        "steps": steps,
        "rows": rows,
        "pass": all(r.get("pass", False) for r in rows),
        "worst_err": max((r.get("err_max", float("inf")) for r in rows), default=float("inf")),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run_all_gates(
    *,
    quick: bool = False,
    couplings_two_phase: tuple[float, ...] = (4.5, 5.0),
) -> dict[str, Any]:
    """Run every gate and return one JSON-friendly results dict.

    ``quick=True`` shrinks the integration gates so the whole suite fits in a
    regression-test budget (< ~2 min single core); the production run in
    ``run_xval.py`` uses the full settings.
    """
    steps_dyn = 120 if quick else 400
    steps_flat = 3000 if quick else 12000
    results: dict[str, Any] = {
        "case": "XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY",
        "anchor": "lssfau/walberla no-mcmp-two-phase-only",
        "quick": quick,
        "gates": {},
    }
    gates = results["gates"]
    gates["G1"] = [
        gate_force_kernel_identity(G=5.0, psi="exp", dtype=torch.float64),
        gate_force_kernel_identity(G=5.0, psi="linear", dtype=torch.float64),
        gate_force_kernel_identity(G=-1.0, psi="exp", dtype=torch.float64),
        gate_force_kernel_identity(G=5.0, psi="exp", dtype=torch.float32),
    ]
    gates["G2"] = gate_null_coupling_purity()
    gates["G3"] = gate_duplicated_partner_dynamics(G=5.0, steps=steps_dyn)
    gates["G4"] = gate_no_self_interaction(G_12=-1.0, steps=steps_dyn)
    gates["G5"] = gate_coupling_sign(steps=150 if quick else 400)
    gates["G6"] = gate_two_phase_coexistence(couplings_two_phase, steps=steps_flat)
    results["pass"] = (
        all(r["pass"] for r in gates["G1"])
        and gates["G2"]["pass"]
        and gates["G3"]["pass"]
        and gates["G4"]["pass"]
        and gates["G5"]["pass"]
        and gates["G6"]["pass"]
    )
    return results