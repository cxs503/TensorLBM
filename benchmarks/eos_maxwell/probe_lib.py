"""EOS-Maxwell construction stability probe (library).

Purpose
-------
TensorLBM's single-component pseudopotential (SCMP) path can be combined with an
equation-of-state pseudopotential (``multiphase.make_psi_eos``, Carnahan-Starling
or Peng-Robinson, Yuan & Schaefer 2006 parameters) whose *coexistence* pair is
fixed by a Maxwell equal-area construction (``sc_eos_coexistence.maxwell_coexistence``).
That construction is a statement about the *EOS*; whether the *discrete lattice
scheme* actually holds the predicted liquid/vapour density contrast in a static
droplet, for how long, and at what spurious-current level, is a separate and
strictly empirical question.  This module answers exactly that question, and it
answers it with a *collapse-is-a-result* contract: a tier that dies returns a
structured failure record, never an exception.

Nothing here is hard-coded from a previous campaign: the coexistence pairs are
recomputed at run time by bisecting the temperature for a requested density
ratio, and every number written to the artefact is produced by this code.

Design (experimental design)
---------------------------
* **Decision**: at which target density ratios (10 / 100 / 1000) is a PR-EOS
  SCMP static droplet a *usable* construct, and what is the cheapest observable
  that predicts collapse?
* **Unit of analysis**: one (target ratio, forcing scheme) run.
* **Baseline / control**: (i) the Maxwell construction itself (analytic target,
  per tier); (ii) a *planar* interface control run at the same EOS point, which
  measures the coexistence pair the lattice scheme actually selects in the
  absence of curvature and in a fully periodic box — it separates
  "EOS-vs-scheme discretisation error" from "droplet geometry / finite domain
  / not-yet-relaxed" as causes of a ratio-maintenance miss.
* **Predeclared thresholds** are in ``PREREG`` below and are frozen before any
  production number is inspected (see benchmarks/eos_maxwell/README.md).

Consumers
---------
The probe does not implement physics kernels: it only orchestrates library
functions (``equilibrium``, ``collide_sc_single_component``, ``stream``,
``make_psi_eos``, ``maxwell_coexistence``) and measures.  ``grep`` self-check:
no ``def collide|stream|equilibrium|maxwell_coexistence`` re-implementation.
"""

from __future__ import annotations

import hashlib
import math
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import tensorlbm  # noqa: E402
from tensorlbm.d2q9 import C, equilibrium  # noqa: E402
from tensorlbm.multiphase import collide_sc_single_component, make_psi_eos  # noqa: E402
from tensorlbm.sc_eos_coexistence import maxwell_coexistence  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402

CS2 = 1.0 / 3.0
CS = math.sqrt(CS2)

# ---------------------------------------------------------------------------
# EOS bookkeeping (parameters of Yuan & Schaefer 2006 as implemented by the
# library; the pseudopotential is *imported*, not restated here).
# ---------------------------------------------------------------------------

EOS_PARAMS = {
    # name: (T_c, covolume b, admissible-density ceiling factor)
    # T_c values are the library's own critical temperatures (multiphase.py
    # convention note); rho_cross solves p_EOS(rho_cross) = rho_cross*cs^2 and
    # is recomputed at run time by _rho_cross(), never assumed.
    "pr": dict(T_c=0.07291903717301021, b=2.0 / 21.0),
    "cs": dict(T_c=0.09432870314611914, b=4.0 / 4.0),
}

# ---------------------------------------------------------------------------
# Predeclared decision criteria (frozen before production runs; see README).
# ---------------------------------------------------------------------------

PREREG = {
    "frozen": "2026-11-09 (before any production run)",
    "target_ratios": [10.0, 100.0, 1000.0],
    "ratio_target_tol": 0.01,  # bisected Maxwell ratio within 1 % of target
    "maxwell_area_tol": 1e-6,  # |area residual| / (p_sat*(v_v - v_l))
    "stability_steps_min": 20000,  # "stable hold" = no collapse for >= N steps
    "mass_drift_rate_max": 5e-8,  # |dm/m|/step, same gate as laplace_pr_eos
    "ratio_error_max": 0.15,  # |ratio_sim - ratio_maxwell| / ratio_maxwell
    "spurious_pass_ma": 0.05,  # max |u| / cs  -> PASS
    "spurious_warn_ma": 0.10,  # PASS <0.05 <= MARGINAL <=0.10 < FAIL
    "clamp_rule": "rho_max_field must stay below rho_cross (psi clamp unused)",
    "negativity_tol": (
        "density negativity counts as collapse only beyond roundoff, "
        "rho_min < -1e-6*rho_max (float32 roundoff in the far vapour field is "
        "O(1e-30) and is not a physical failure)"
    ),
    "negativity_tol_rel": 1e-6,
    # ------------------------------------------------------------------
    # Second predeclaration (2026-11-09, written *before* the tau arm was run
    # and after the core tiers/boundary arms had already been reported).  Kept
    # separate from the block above so no earlier gate is retuned.
    # ------------------------------------------------------------------
    "tau_arm": {
        "declared": "2026-11-09 (before the tau arm was run)",
        "grid": {"eos": "pr", "target_ratios": [10.0, 100.0, 1000.0], "forcing": "edm",
                 "tau": [0.6, 0.8, 1.0, 1.6, 2.0], "steps": 20000, "L": 96, "R": 20.0},
        "question": (
            "is the spurious-current gate a *flow-regime* statement (tunable by the "
            "relaxation time tau) or a *construct* statement, and does tau move the "
            "density-ratio maintenance error at all?"
        ),
        "caveat_rule": (
            "added with the arm, before its result was seen: the hold length of a "
            "tau run must cover the slowest linear timescale at *that* tau "
            "(viscous diffusion across the droplet radius, R**2/nu).  A run whose "
            "hold/steps_slowest < 2 is labelled 'not_relaxation_resolved' and may "
            "not be used as positive evidence (it can still falsify: a collapse is "
            "a collapse).  Consequence, also fixed in advance: if the Ma gate is only "
            "cleared by runs that are not relaxation-resolved the decision string is "
            "downgraded to 'ma_pass_only_in_unrelaxed_runs' instead of claiming a "
            "tunable gate.  This is the price of keeping the tier hold length fixed."
        ),
        "decision_rule": (
            "per target ratio, on stable runs only: MA_OK = exists tau with spurious "
            "Ma <= spurious_pass_ma; RATIO_OK = exists tau with ratio_error <= "
            "ratio_error_max.  (MA_OK, RATIO_OK)=(1,1) -> construct usable at that "
            "ratio; (1,0) -> spurious gate is viscosity-tunable but the ratio miss is "
            "structural; (0,1) -> ratio held at every tau but the current floor stays "
            "above the PASS gate; (0,0) -> neither is reachable by tau alone.  No tau "
            "is added to, or removed from, the tier grid after looking at the result: "
            "the tier gate stays at tau=1.0."
        ),
    },
    "honest_fail_rule": (
        "a tier that dies (non-finite rho, rho<0, |u| > Ma 0.5, or clamp "
        "exposure) is reported as FAIL with the onset step and the last finite "
        "state; it is never retried with tuned parameters to manufacture a pass"
    ),
}


def _rho_cross(eos: str, T: float, n: int = 200000, hi_factor: float = 0.999) -> float:
    """Density where p_EOS(rho) = rho*cs^2 (beyond it the psi clamp engages).

    Solved by bisection on the library pseudopotential itself:
    ``psi(rho) > 0  <=>  p_EOS(rho) < rho*cs^2`` for the corrected-sign factory,
    so the first zero of psi on the dense branch is the clamp threshold.
    """
    b = EOS_PARAMS[eos]["b"]
    lo = 0.5 / b
    hi = hi_factor / b  # 1 - b*rho -> 0
    psi_fn = make_psi_eos(eos, T)
    t = torch.tensor([hi], dtype=torch.float64)
    if float(psi_fn(t)[0]) > 0.0:
        return hi
    for _ in range(n.bit_length() * 4):
        mid = 0.5 * (lo + hi)
        if float(psi_fn(torch.tensor([mid], dtype=torch.float64))[0]) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def maxwell_at_ratio(
    target_ratio: float,
    eos: str = "pr",
    G: float = 1.0,
    tr_bracket: tuple[float, float] = (0.40, 0.96),
    tr_tol: float = 1e-6,
    max_iter: int = 80,
) -> dict:
    """Bisect the reduced temperature for a target Maxwell density ratio.

    The coexistence ratio is monotone decreasing in ``T_r`` for a subcritical
    van der Waals-like EOS (verified per call by the bracket check), so plain
    bisection is exact to ``tr_tol``.  Every evaluation calls the library
    Maxwell solver, so the returned pair is machine-computed, not transcribed.

    Returns:
        dict with ``target_ratio``, ``T_r``, ``T``, ``rho_liquid``,
        ``rho_vapour``, ``ratio``, ``p_sat``, ``rho_cross``,
        ``area_residual_rel``, ``bisect_log`` and ``found``.
    """
    T_c = EOS_PARAMS[eos]["T_c"]
    b = EOS_PARAMS[eos]["b"]
    rho_hi = 0.98 / b
    log: list[dict] = []

    def evaluate(tr: float) -> dict:
        T = tr * T_c
        res = maxwell_coexistence(G, make_psi_eos(eos, T), rho_lo=1e-6, rho_hi=rho_hi, n=12000)
        rec = dict(
            T_r=tr,
            T=T,
            found=res.found,
            ratio=res.density_ratio,
            rho_liquid=res.rho_liquid,
            rho_vapour=res.rho_vapour,
            p_sat=res.p_sat,
            area_residual=res.area_residual,
            rho_spinodal_vapour=res.rho_spinodal_vapour,
            rho_spinodal_liquid=res.rho_spinodal_liquid,
            reason=res.reason if not res.found else "equal-area construction converged",
        )
        if res.found and res.p_sat and res.p_sat > 0:
            v_l, v_v = 1.0 / res.rho_liquid, 1.0 / res.rho_vapour
            rec["area_residual_rel"] = abs(res.area_residual) / (res.p_sat * (v_v - v_l))
        log.append(rec)
        return rec

    lo, hi = tr_bracket
    r_lo, r_hi = evaluate(lo), evaluate(hi)
    if not (r_lo["found"] and r_hi["found"]):
        return dict(target_ratio=target_ratio, found=False, reason="bracket endpoint has no coexistence", bisect_log=log)
    if not (r_lo["ratio"] >= target_ratio >= r_hi["ratio"]):
        return dict(
            target_ratio=target_ratio,
            found=False,
            reason=(
                f"target outside achievable bracket: ratio(T_r={lo})={r_lo['ratio']:.3g}, "
                f"ratio(T_r={hi})={r_hi['ratio']:.3g}"
            ),
            bisect_log=log,
        )
    a, b_tr = lo, hi
    best = r_lo if abs(r_lo["ratio"] - target_ratio) < abs(r_hi["ratio"] - target_ratio) else r_hi
    for _ in range(max_iter):
        mid = 0.5 * (a + b_tr)
        rec = evaluate(mid)
        if not rec["found"]:
            return dict(target_ratio=target_ratio, found=False, reason=f"no coexistence at T_r={mid}", bisect_log=log)
        if abs(rec["ratio"] - target_ratio) < abs(best["ratio"] - target_ratio):
            best = rec
        if abs(rec["ratio"] - target_ratio) / target_ratio <= PREREG["ratio_target_tol"]:
            best = rec
            break
        if rec["ratio"] > target_ratio:
            a = mid
        else:
            b_tr = mid
        if b_tr - a < tr_tol:
            break
    out = {k: v for k, v in best.items() if k != "T_r"}
    out.update(target_ratio=target_ratio, T_r=best["T_r"], found=True, bisect_log=log)
    out["rho_cross"] = _rho_cross(eos, best["T"])
    out["within_target_tol"] = abs(out["ratio"] - target_ratio) / target_ratio <= PREREG["ratio_target_tol"]
    return out


def maxwell_at_tr(eos: str, T_r: float, G: float = 1.0) -> dict:
    """One Maxwell evaluation at a prescribed reduced temperature (boundary map)."""
    T = T_r * EOS_PARAMS[eos]["T_c"]
    b = EOS_PARAMS[eos]["b"]
    res = maxwell_coexistence(G, make_psi_eos(eos, T), rho_lo=1e-6, rho_hi=0.98 / b, n=12000)
    out = dict(
        target_ratio=None,
        T_r=T_r,
        T=T,
        found=res.found,
        ratio=res.density_ratio,
        rho_liquid=res.rho_liquid,
        rho_vapour=res.rho_vapour,
        p_sat=res.p_sat,
        area_residual=res.area_residual,
        reason=res.reason,
    )
    out["rho_cross"] = _rho_cross(eos, T) if res.found else float("inf")
    return out


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def droplet_field(L: int, R: float, rho_l: float, rho_v: float, device, width: float = 4.0):
    """Circular tanh droplet (liquid core radius R) on an L x L periodic box."""
    ys = torch.arange(L, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, ys, indexing="ij")
    r = torch.sqrt((xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2)
    rho = rho_v + 0.5 * (rho_l - rho_v) * (1.0 + torch.tanh((R - r) / width))
    return rho, r


def planar_field(ny: int, nx: int, rho_l: float, rho_v: float, device, width: float = 4.0):
    """Flat-interface control: a liquid slab centred in a periodic box.

    A single step (liquid below / vapour above) is periodic-*incompatible*: the
    wrap-around joins rho_l to rho_v across one cell, producing an artificial
    one-cell jump that dominates every force diagnostic (measured: max|F| ~6x
    the droplet value at the same interface width).  The slab has two smooth
    tanh interfaces at ``y = ny/4`` and ``y = 3ny/4``, so the field is periodic
    by construction and the plateaus are genuine bulk phases.
    """
    ys = torch.arange(ny, dtype=torch.float32, device=device).view(ny, 1)
    y1, y2 = ny / 4.0, 3.0 * ny / 4.0
    slab = 0.5 * (torch.tanh((ys - y1) / width) - torch.tanh((ys - y2) / width))
    rho = rho_v + (rho_l - rho_v) * slab
    return rho.expand(ny, nx).contiguous(), ys.expand(ny, nx)


def planar_regions(ny: int, width: float):
    """Plateau masks for the slab: liquid centre, vapour film next to the walls."""
    ys = torch.arange(ny, dtype=torch.float32).view(ny, 1).expand(ny, ny)
    y1, y2 = ny / 4.0, 3.0 * ny / 4.0
    liquid = (ys - ny / 2.0).abs() <= ny / 8.0
    pad = 3.0 * width
    vapour = (ys <= y1 - pad) | (ys >= y2 + pad)
    return liquid, vapour


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _macroscopic_momentum(f):
    rho = f.sum(dim=0)
    c = C.to(f.device).float()
    mx = (f * c[:, 0].view(9, 1, 1)).sum(dim=0)
    my = (f * c[:, 1].view(9, 1, 1)).sum(dim=0)
    return rho, mx, my


def radial_profile(rho, r, n_bins: int = 48) -> list[dict]:
    """Annulus-averaged density profile (bin width = L/2/n_bins)."""
    L = rho.shape[0]
    dr = (L / 2.0) / n_bins
    idx = torch.clamp((r / dr).long(), 0, n_bins - 1)
    flat_idx, flat_rho = idx.reshape(-1), rho.reshape(-1)
    out = []
    for k in range(n_bins):
        m = flat_idx == k
        n = int(m.sum().item())
        if n == 0:
            continue
        vals = flat_rho[m]
        out.append(
            dict(
                r_mid=(k + 0.5) * dr,
                r_lo=k * dr,
                r_hi=(k + 1) * dr,
                rho_mean=float(vals.mean().item()),
                rho_min=float(vals.min().item()),
                rho_max=float(vals.max().item()),
                n_cells=n,
            )
        )
    return out


@dataclass
class RunSpec:
    """One probe run (droplet tier, planar control, or boundary-map cell)."""

    tag: str
    eos: str = "pr"
    target_ratio: float | None = None
    T_r: float | None = None
    L: int = 96
    R: float = 20.0
    G: float = 1.0
    tau: float = 1.0
    forcing: str = "edm"
    scheme: str = "standard"
    steps: int = 20000
    sample: int = 100
    geometry: str = "droplet"  # "droplet" | "planar"
    n_bins: int = 48
    device: str = "cpu"
    width: float = 4.0
    threads: int | None = None
    noise: float = 0.0
    seed: int = 0
    history: list = field(default_factory=list)


def run_probe(spec: RunSpec, maxwell: dict) -> dict:
    """Run one probe; always returns a record (collapse included, never raises)."""
    device = torch.device(spec.device)
    # Thread count fixes the float32 reduction order; at a *marginal* construct it
    # can flip the outcome, so it is part of the recorded configuration.
    torch.set_num_threads(spec.threads if spec.threads else max(1, min(8, torch.get_num_threads())))
    psi_fn = make_psi_eos(spec.eos, maxwell["T"])
    rho_cross = maxwell.get("rho_cross", float("inf"))
    rho_l0, rho_v0 = maxwell["rho_liquid"], maxwell["rho_vapour"]

    if spec.geometry == "droplet":
        rho0, r = droplet_field(spec.L, spec.R, rho_l0, rho_v0, device, spec.width)
        ny = nx = spec.L
        far = r >= 0.80 * spec.L / 2.0
        core0 = r <= 0.25 * spec.R
    else:
        ny = nx = spec.L
        rho0, ys = planar_field(ny, nx, rho_l0, rho_v0, device, spec.width)
        r = None
        core0, far = planar_regions(ny, spec.width)  # liquid core / vapour film

    if spec.noise:
        # Chaotic-margin sensitivity control: a physically irrelevant density
        # perturbation (amplitude << 1 %) tests whether "stable for N steps" is
        # a property of the construct or a property of one trajectory.
        gen = torch.Generator().manual_seed(spec.seed)
        jitter = torch.rand(rho0.shape, generator=gen) - 0.5
        rho0 = rho0 * (1.0 + spec.noise * jitter.to(rho0.device))
    zero = torch.zeros_like(rho0)
    f = equilibrium(rho0, zero, zero)
    rho_ref = rho0.clone()
    mass0 = float(rho0.sum().item())

    hist: list[dict] = []
    stable, err, step_i, t0 = True, None, 0, time.time()
    transient = None
    for step_i in range(1, spec.steps + 1):
        f = stream(
            collide_sc_single_component(
                f, G=spec.G, tau=spec.tau, psi_fn=psi_fn, forcing=spec.forcing, scheme=spec.scheme
            )
        )
        rho = f.sum(dim=0)
        # collapse onset is resolved to the exact step (cheap reduction only)
        if step_i == 1:
            rho1, mx1, my1 = _macroscopic_momentum(f)
            transient = dict(
                step=1,
                rho_max=float(rho1.max().item()),
                rho_min=float(rho1.min().item()),
                u_max=float(
                    torch.sqrt((mx1 / rho1.clamp(min=1e-12)) ** 2 + (my1 / rho1.clamp(min=1e-12)) ** 2).max().item()
                ),
                u_max_over_cs=float(
                    torch.sqrt((mx1 / rho1.clamp(min=1e-12)) ** 2 + (my1 / rho1.clamp(min=1e-12)) ** 2).max().item() / CS
                ),
                grad_rho_max=max(
                    float(torch.diff(rho1, dim=1).abs().max().item()),
                    float(torch.diff(rho1, dim=0).abs().max().item()),
                ),
            )
        neg_tol = -PREREG["negativity_tol_rel"] * max(float(rho.max().item()), 1.0)
        if not torch.isfinite(rho).all() or float(rho.min().item()) < neg_tol:
            stable = False
            err = (
                f"non-finite or negative density at step {step_i} "
                f"(rho_min={float(rho.min().item()):.3e}, rho_max={float(rho.max().item()):.3e})"
            )
            break
        if step_i % spec.sample:
            continue
        rho, mx, my = _macroscopic_momentum(f)
        rho_min, rho_max = float(rho.min().item()), float(rho.max().item())
        rho_s = rho.clamp(min=1e-12)
        speed = torch.sqrt((mx / rho_s) ** 2 + (my / rho_s) ** 2)
        u_max = float(speed.max().item())
        if u_max > 0.5 * CS:
            stable = False
            err = f"spurious-current blow-up at step {step_i}: max|u|/cs={u_max / CS:.3f} > 0.5"
            break
        rho_l = float(rho[core0].mean().item())
        rho_v = float(rho[far].mean().item())
        if spec.geometry == "droplet":
            r_eff = float(
                math.sqrt(
                    max(float((rho - rho_v).sum().item()), 0.0)
                    / (math.pi * max(rho_l - rho_v, 1e-12))
                )
            )
            if r_eff > 0:
                core = r <= 0.5 * r_eff
                band = (r > 0.5 * r_eff) & (r < r_eff + 6.0)
                rho_l = float(rho[core].mean().item())
                rho_v = float(rho[far].mean().item())
                u_band = float(speed[band].max().item()) if bool(band.any()) else float("nan")
            else:
                u_band = float("nan")
        else:
            r_eff = float("nan")
            u_band = float("nan")
        mass = float(rho.sum().item())
        rec = dict(
            step=step_i,
            rho_l=rho_l,
            rho_v=rho_v,
            ratio=rho_l / max(rho_v, 1e-12),
            r_eff=r_eff,
            u_max=u_max,
            ma_max=u_max / CS,
            u_max_band=u_band,
            u_rms=float(torch.sqrt((speed**2).mean()).item()),
            rho_min_field=rho_min,
            rho_max_field=rho_max,
            mass=mass,
            mass_drift=abs(mass - mass0) / mass0,
            clamp_exposed=bool(rho_max >= rho_cross),
        )
        hist.append(rec)
        if rho_max >= rho_cross:
            stable = False
            err = f"psi clamp engaged at step {step_i}: rho_max={rho_max:.4f} >= rho_cross={rho_cross:.4f}"
            break
    wall = time.time() - t0

    tail = hist[-5:] if hist else []
    out = dict(
        tag=spec.tag,
        spec=dict(
            eos=spec.eos,
            geometry=spec.geometry,
            L=spec.L,
            R=spec.R,
            G=spec.G,
            tau=spec.tau,
            forcing=spec.forcing,
            scheme=spec.scheme,
            steps=spec.steps,
            sample=spec.sample,
            width=spec.width,
            threads=torch.get_num_threads(),
            noise=spec.noise,
            seed=spec.seed,
        ),
        target_ratio=spec.target_ratio,
        T_r=maxwell.get("T_r"),
        T=maxwell["T"],
        rho_l_maxwell=rho_l0,
        rho_v_maxwell=rho_v0,
        ratio_maxwell=maxwell.get("ratio"),
        rho_cross=rho_cross,
        maxwell_found=bool(maxwell.get("found", True)),
        steps_run=int(step_i),
        stable=bool(stable),
        collapse_step=None if stable else int(step_i),
        error=err,
        transient=transient,
        wall_s=round(wall, 1),
        ms_per_step=round(1000.0 * wall / max(step_i, 1), 3),
        mass_drift_final=float(hist[-1]["mass_drift"]) if hist else None,
        hist=hist,
    )
    if tail:
        def mean(key):
            return sum(t[key] for t in tail) / len(tail)

        out.update(
            rho_l=mean("rho_l"),
            rho_v=mean("rho_v"),
            ratio=mean("ratio"),
            r_eff=mean("r_eff"),
            u_max=max(t["u_max"] for t in tail),
            ma_max=max(t["ma_max"] for t in tail),
            u_max_band=max((t["u_max_band"] for t in tail if t["u_max_band"] == t["u_max_band"]), default=None),
            u_rms=mean("u_rms"),
            rho_max_field=max(t["rho_max_field"] for t in tail),
            rho_min_field=min(t["rho_min_field"] for t in tail),
            clamp_exposed=any(t["clamp_exposed"] for t in tail),
            mass_drift=max(t["mass_drift"] for t in tail),
            mass_rate_per_step=max(t["mass_drift"] for t in tail) / max(step_i, 1),
            u_max_transient=max(t["u_max"] for t in hist),
        )
        out["ratio_error"] = abs(out["ratio"] - out["ratio_maxwell"]) / max(out["ratio_maxwell"], 1e-12)
    # profile of the final state (only if the run is finite)
    if stable or hist:
        rho_final = f.sum(dim=0)
        if bool(torch.isfinite(rho_final).all()):
            if spec.geometry == "droplet":
                prof = radial_profile(rho_final, r if r is not None else _radial_grid(spec.L, device), spec.n_bins)
                out["radial_profile"] = prof
                out["radial_profile_note"] = "annulus mean of rho on the final state"
            else:
                rows = rho_final.mean(dim=1)
                out["y_profile"] = [
                    dict(y=int(i), rho_mean=float(rows[i].item())) for i in range(rows.numel())
                ]
                out["y_profile_note"] = "row mean of rho on the final state (planar control)"
    return out


def _radial_grid(L: int, device) -> torch.Tensor:
    ys = torch.arange(L, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, ys, indexing="ij")
    return torch.sqrt((xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2)


def classify(run: dict) -> dict:
    """Apply the predeclared gates to one run record -> verdict block."""
    if not run.get("stable", False):
        return dict(
            cls="FAIL",
            reason="collapse",
            detail=run.get("error"),
            collapse_step=run.get("collapse_step"),
        )
    msgs, cls = [], "PASS"
    if run.get("ratio_error") is not None and run["ratio_error"] > PREREG["ratio_error_max"]:
        msgs.append(
            f"ratio_error={run['ratio_error']:.3f} > {PREREG['ratio_error_max']}"
        )
        cls = "FAIL"
    ma = run.get("ma_max")
    if ma is not None:
        if ma > PREREG["spurious_warn_ma"]:
            msgs.append(f"spurious Ma={ma:.3f} > {PREREG['spurious_warn_ma']}")
            cls = "FAIL"
        elif ma > PREREG["spurious_pass_ma"] and cls == "PASS":
            msgs.append(f"spurious Ma={ma:.3f} in ({PREREG['spurious_pass_ma']}, {PREREG['spurious_warn_ma']}]")
            cls = "MARGINAL"
    if run.get("mass_rate_per_step") is not None and run["mass_rate_per_step"] > PREREG["mass_drift_rate_max"]:
        msgs.append(f"mass rate {run['mass_rate_per_step']:.2e} > {PREREG['mass_drift_rate_max']:.0e}")
        cls = "FAIL"
    if run.get("clamp_exposed"):
        msgs.append("psi clamp engaged in the sampled tail")
        cls = "FAIL"
    if run.get("steps_run", 0) < PREREG["stability_steps_min"]:
        msgs.append(f"steps_run={run['steps_run']} < {PREREG['stability_steps_min']} hold requirement")
        if cls == "PASS":
            cls = "MARGINAL"
    return dict(cls=cls, reason="; ".join(msgs) if msgs else "all predeclared gates met", detail=None)


def stability_budget(R: float = 20.0, tau: float = 1.0, L: int = 96, width: float = 4.0) -> dict:
    """Why the hold length N is what it is (timescale budget, not a round number).

    Three linear timescales of the construct, in lattice steps:

    * acoustic transit of the box, ``L / c_s``  (how long a pressure wave needs
      to cross the domain once);
    * viscous diffusion across the *initial interface width*, ``w**2 / nu``
      (how long the tanh profile takes to be replaced by the scheme's own
      profile);
    * viscous diffusion across the *droplet radius*, ``R**2 / nu`` (the slowest
      linear mode that can still change the measured density contrast),
      with ``nu = c_s**2 (tau - 1/2)``.

    A hold length is only meaningful if it is several times the slowest of these;
    ``PREREG['stability_steps_min']`` is chosen that way and the ratio is
    reported here so the choice can be re-checked, not trusted.
    """
    nu = (1.0 / 3.0) * (tau - 0.5)
    times = {
        "acoustic_transit": L / CS,
        "viscous_interface": width**2 / nu,
        "viscous_droplet": R**2 / nu,
    }
    slowest_key = max(times, key=times.get)
    n = float(PREREG["stability_steps_min"])
    return dict(
        tau=tau,
        nu=nu,
        timescales=times,
        slowest=slowest_key,
        slowest_steps=times[slowest_key],
        hold_steps=n,
        hold_over_slowest=n / times[slowest_key],
    )


def tau_lever_summary(rows: list[dict]) -> dict:
    """Apply ``PREREG['tau_arm']['decision_rule']`` to the tau-arm runs.

    Pure post-processing of ``run_probe`` records: no physics, no thresholds
    beyond the predeclared gates (``spurious_pass_ma``, ``ratio_error_max``).
    Runs that collapsed contribute no gate evidence (they cannot clear anything);
    a ratio with no surviving run at all is reported as ``no_stable_run``.
    """
    pre = PREREG
    out: dict[str, dict] = {}
    for ratio in pre["target_ratios"]:
        rows_r = [r for r in rows if r.get("target_ratio") == ratio]
        alive = [r for r in rows_r if r.get("stable")]
        entry: dict = {
            "target_ratio": ratio,
            "taus": sorted({r["spec"]["tau"] for r in rows_r}),
            "n_collapsed": len(rows_r) - len(alive),
            "ma_min": None,
            "ma_min_tau": None,
            "ratio_error_min": None,
            "ratio_error_min_tau": None,
            "ma_ok": False,
            "ratio_ok": False,
        }
        if alive:
            coverage = {
                r["spec"]["tau"]: float(
                    PREREG["stability_steps_min"]
                    / stability_budget(R=float(r["spec"].get("R", 20.0)),
                                       tau=float(r["spec"]["tau"]),
                                       L=int(r["spec"].get("L", 96)),
                                       width=float(r["spec"].get("width", 4.0)))["timescales"]["viscous_droplet"]
                )
                for r in alive
            }
            entry["coverage"] = coverage
            entry["not_relaxation_resolved"] = [t for t, c in coverage.items() if c < 2.0]
            best_ma = min(alive, key=lambda r: r["ma_max"])
            best_ratio = min(alive, key=lambda r: r["ratio_error"])
            entry.update(
                ma_min=best_ma["ma_max"],
                ma_min_tau=best_ma["spec"]["tau"],
                ratio_error_min=best_ratio["ratio_error"],
                ratio_error_min_tau=best_ratio["spec"]["tau"],
                mas={r["spec"]["tau"]: r["ma_max"] for r in alive},
                ratio_errors={r["spec"]["tau"]: r["ratio_error"] for r in alive},
                ma_ok_relaxed=bool(
                min(
                    (r["ma_max"] for r in alive
                     if coverage[r["spec"]["tau"]] >= 2.0),
                    default=None,
                )
                is not None
                and min(r["ma_max"] for r in alive if coverage[r["spec"]["tau"]] >= 2.0)
                <= pre["spurious_pass_ma"]
            ),
            ma_ok=bool(best_ma["ma_max"] <= pre["spurious_pass_ma"]),
                ratio_ok=bool(best_ratio["ratio_error"] <= pre["ratio_error_max"]),
            )
            if entry["ma_ok"] and not entry["ma_ok_relaxed"]:
                entry["decision"] = "ma_pass_only_in_unrelaxed_runs"
            elif entry["ma_ok"] and entry["ratio_ok"]:
                entry["decision"] = "usable_at_some_tau"
            elif entry["ma_ok"]:
                entry["decision"] = "spurious_tunable_ratio_structural"
            elif entry["ratio_ok"]:
                entry["decision"] = "ratio_held_currents_above_pass_gate"
            else:
                entry["decision"] = "dead_at_all_tau"
        elif rows_r:
            entry["decision"] = "no_stable_run"
        else:
            entry["decision"] = "arm_not_run"
        out[f"{ratio:g}"] = entry
    return out


def environment() -> dict:
    try:
        import importlib.metadata as md

        tl = md.version("tensorlbm")
    except Exception:  # noqa: BLE001
        tl = None
    return dict(
        python=sys.version.split()[0],
        platform=platform.platform(),
        machine=platform.machine(),
        torch=torch.__version__,
        tensorlbm_version=tl,
        tensorlbm_file=tensorlbm.__file__,
        threads=torch.get_num_threads(),
        probe_lib_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )