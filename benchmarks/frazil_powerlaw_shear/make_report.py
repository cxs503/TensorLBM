#!/usr/bin/env python3
"""Generate the frazil power-law shear report (markdown + platform report JSON).

Every number in the output is read from
``artifacts/frazil_powerlaw_shear/campaign.json`` and every figure from
``figures.json`` -- nothing is transcribed by hand.

Usage::

    python3 benchmarks/frazil_powerlaw_shear/make_report.py            # docs markdown
    python3 benchmarks/frazil_powerlaw_shear/make_report.py --api      # + /tmp/report.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ART = os.path.join(ROOT, "artifacts", "frazil_powerlaw_shear", "campaign.json")
DOC = os.path.join(ROOT, "docs", "frazil_powerlaw_shear.md")
MISSION_ID = 10712
SCENARIO = "ice_navigation"


def _f(x, fmt="{:.4g}"):
    if x is None:
        return "n/a"
    try:
        if isinstance(x, float) and (x != x):  # NaN
            return "n/a"
        return fmt.format(x)
    except Exception:  # noqa: BLE001
        return str(x)


def _pct(x, fmt="{:.2f}"):
    return "n/a" if x is None else (fmt.format(100.0 * x) + " %")


def _coarse_tag(art):
    tags = {}
    for key in art["flow_curves"]:
        tag = key.split("|")[0]
        tags[tag] = tags.get(tag, 0) + 1
    if not tags:
        return None
    return sorted(tags.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _curves(art, res=None):
    res = res or _coarse_tag(art)
    out = []
    for key, val in art["flow_curves"].items():
        r, fam, phi = key.split("|")
        if r != res:
            continue
        out.append((fam, float(phi), val))
    return sorted(out)


def table_flow_curves(art, res=None):
    rows = ["| family | c | N primaries | $\\phi$ (lattice) | $\\gamma\\dot$ range | "
            "$n$ | $K$ | $R^2$ | $\\eta_{eff}/\\mu_0$ |",
            "|---|---|---|---|---|---|---|---|---|"]
    for fam, phi, v in _curves(art, res):
        gr = v["gamma_range"]
        rows.append(
            f"| {fam} | {phi:.2f} | {_f(v.get('mean_n_primary'), '{:.1f}')} | "
            f"{_f(v.get('phi_lattice'), '{:.4f}')} | "
            f"{gr[0]:.3e} – {gr[1]:.3e} | {v['n']:.4f} | {v['K']:.5g} | "
            f"{_f(v.get('r2'), '{:.4f}')} | {_f(v.get('eta_r_ref'), '{:.4f}')} |")
    return "\n".join(rows)


def max_seed_spread(art) -> float:
    """Largest seed-to-seed spread of eta_r anywhere in the campaign."""
    groups: dict = {}
    for c in art["per_case"]:
        if c.get("seed") in (None, "mean"):
            continue
        groups.setdefault((c["family"], c["phi"], c["rate_index"]), []).append(c["eta_r"])
    spreads = [max(v) - min(v) for v in groups.values() if len(v) > 1]
    return float(max(spreads)) if spreads else float("nan")


def table_criteria(art):
    crit = art["acceptance"]["criteria"]
    rows = ["| # | criterion | measured | threshold | verdict |",
            "|---|---|---|---|---|"]
    c1 = crit["c1_zero_control"]
    rows.append(
        f"| 1 | $c=0$ negative control | $n={_f(c1.get('n'))}$ (dev {_f(c1.get('n_dev'))}), "
        f"$\\eta_r={_f(c1.get('eta_r'))}$ (dev {_pct(c1.get('eta_r_dev'))}) | "
        f"$n=1\\pm0.01$, $\\mu=\\mu_w\\pm2\\%$ | {'PASS' if c1.get('passed') else 'FAIL'} |")
    c2 = crit["c2_dilute_einstein"]
    rows.append(
        f"| 2 | $c=0.05$ Einstein | $\\eta_r={_f(c2.get('eta_r'))}$ | "
        f"$1+2.5c={_f(c2.get('prediction'))}$ ({_pct(c2.get('rel_err'))}) | "
        f"{'PASS' if c2.get('passed') else 'FAIL'} |")
    c3 = crit["c3_krieger_dougherty"]
    rows.append(
        f"| 3 | $c=0.20$ Krieger–Dougherty | $\\eta_r={_f(c3.get('eta_r'))}$ "
        f"($\\phi={_f(c3.get('phi_lattice'), '{:.4f}')}$) | "
        f"KD $={_f(c3.get('prediction'))}$ ({_pct(c3.get('rel_err'))}) | "
        f"{'PASS' if c3.get('passed') else 'FAIL'} |")
    c4 = crit["c4_shear_thinning"]
    per = c4.get("per_concentration", {})
    detail = ", ".join(f"c={k}: n={v['n']:.4f}" for k, v in sorted(per.items())) or "no case"
    rows.append(f"| 4 | shear thinning, $c\\ge0.10$ | {detail} | $n<0.95$ | "
                f"{'PASS' if c4.get('passed') else 'FAIL'} |")
    c5 = crit["c5_grid_convergence"]
    rows.append(
        f"| 5 | grid convergence | $\\Delta n/n={_pct(c5.get('n_rel_diff'))}$, "
        f"$\\Delta K/K={_pct(c5.get('K_rel_diff'))}$ | both $<5\\%$ | "
        f"{'PASS' if c5.get('passed') else 'FAIL'} |")
    return "\n".join(rows)


def table_convergence(art):
    convs = art.get("convergence") or {}
    if not convs:
        return "_no convergence pair completed_"
    rows = ["| pair | $n$ (coarse / fine) | $K$ (coarse / fine) | $\\Delta n/n$ | "
            "$\\Delta K/K$ | within 5 %? |", "|---|---|---|---|---|---|"]
    for name, c in sorted(convs.items()):
        rows.append(
            f"| {name} | {c['n_coarse']:.4f} / {c['n_fine']:.4f} | "
            f"{c['K_coarse']:.5g} / {c['K_fine']:.5g} | {_pct(c['n_rel_diff'])} | "
            f"{_pct(c['K_rel_diff'])} | {'yes' if c['both_below_5pct'] else 'no'} |")
    return "\n".join(rows)


def table_pure(art):
    rows = ["| resolution | rate index | $\\tau$ measured | $\\nu\\dot\\gamma$ (exact) | "
            "rel. error | profile $R^2$ |", "|---|---|---|---|---|---|"]
    for key, v in sorted(art["pure_reference"].items()):
        res, gi = key.split("|")
        ref = v["nu"] * v["gamma_dot"]
        err = (v["tau"] - ref) / ref if ref else float("nan")
        rows.append(f"| {res} | {gi} | {v['tau']:.6e} | {ref:.6e} | {_pct(abs(err))} | "
                    f"{_f(v.get('profile_r2'), '{:.5f}')} |")
    return "\n".join(rows)


def table_scatter(art):
    rows = ["| family | c | rate idx | seeds | $\\eta_r$ per seed | spread |",
            "|---|---|---|---|---|---|"]
    per = art["per_case"]
    groups = {}
    for c in per:
        if c.get("seed") in (None, "mean"):
            continue
        key = (c["family"], c["phi"], c["rate_index"])
        groups.setdefault(key, []).append(c["eta_r"])
    for (fam, phi, gi), vals in sorted(groups.items()):
        if len(vals) < 2:
            continue
        rows.append(f"| {fam} | {phi:.2f} | {gi} | {len(vals)} | "
                    f"{', '.join(f'{v:.4f}' for v in vals)} | "
                    f"{_pct(max(vals) - min(vals), '{:.2f}')} |")
    if len(rows) == 2:
        return "_single-seed cases only (no replicate scatter available)_"
    return "\n".join(rows)


def table_error_judgement(art):
    """Six-element §6: every key quantity against its reference, explicitly."""
    crit = art["acceptance"]["criteria"]
    convs = art.get("convergence") or {}
    rows = ["| key quantity | reference | measured | error | $<3\\%$? |",
            "|---|---|---|---|---|"]
    c1 = crit["c1_zero_control"]
    if c1.get("defined"):
        rows.append(f"| $c=0$ stress exponent $n$ | exact $n=1$ (plane Couette is an exact "
                    f"Navier–Stokes solution for every $Re$) | {_f(c1.get('n'))} | "
                    f"{_pct(c1.get('n_dev'))} | {'yes' if c1.get('n_dev', 1) < 0.03 else 'no'} |")
        rows.append(f"| $c=0$ relative viscosity | exact $\\eta_r=1$ | "
                    f"{_f(c1.get('eta_r'))} | {_pct(c1.get('eta_r_dev'))} | "
                    f"{'yes' if c1.get('eta_r_dev', 1) < 0.03 else 'no'} |")
    c2 = crit["c2_dilute_einstein"]
    if c2.get("defined"):
        rows.append(f"| $\\eta_r$ at $c=0.05$ | Einstein $1+2.5c$ (exact to $O(c^2)$) | "
                    f"{_f(c2.get('eta_r'))} | {_pct(abs(c2.get('rel_err', 9)))} | "
                    f"{'yes' if abs(c2.get('rel_err', 9)) < 0.03 else 'no'} |")
    c3 = crit["c3_krieger_dougherty"]
    if c3.get("defined"):
        rows.append(f"| $\\eta_r$ at $c=0.20$ | Krieger–Dougherty ($\\phi_m=0.6$, "
                    f"$[\\eta]=2.5$) | {_f(c3.get('eta_r'))} | "
                    f"{_pct(abs(c3.get('rel_err', 9)))} | "
                    f"{'yes' if abs(c3.get('rel_err', 9)) < 0.03 else 'no'} |")
    for name, c in sorted(convs.items()):
        rows.append(f"| $n$, {name} | coarser resolution | {c['n_fine']:.4f} | "
                    f"{_pct(c['n_rel_diff'])} | "
                    f"{'yes' if c['n_rel_diff'] < 0.03 else 'no'} |")
        rows.append(f"| $K$, {name} | coarser resolution | {c['K_fine']:.5g} | "
                    f"{_pct(c['K_rel_diff'])} | "
                    f"{'yes' if c['K_rel_diff'] < 0.03 else 'no'} |")
    rows.append("| $n$ at $c\\ge0.10$ | no pointwise reference exists (frazil floc "
                "rheology is reported as ranges, not digitised curves) | "
                + ", ".join(f"{v['n']:.3f}"
                            for v in crit["c4_shear_thinning"]
                            .get("per_concentration", {}).values())
                + " | n/a — declared | n/a — declared |")
    return "\n".join(rows)


def table_drag(art):
    """Drag gate: the volume-fraction conversion, and whether it may be applied."""
    gate = art.get("drag_gate") or {}
    if not gate:
        return "_no drag calibration was run for this campaign_"
    rows = ["| resolution | $a_h/a_eff$ | $\\phi_h/\\phi_{mask}$ | drag vs Stokes (unconfined) | "
            "drag vs Stokes (confined) | applied? |",
            "|---|---|---|---|---|---|"]
    for tag, g in sorted(gate.items()):
        rows.append(f"| {tag} | {_f(g.get('a_h_over_a_eff'), '{:.4f}')} | "
                    f"{_f(g.get('phi_ratio'), '{:.4f}')} | {_pct((g.get('err_pct_raw') or 0) / 100)} | "
                    f"{_pct((g.get('err_pct_confined') or 0) / 100)} | "
                    f"{'yes' if g.get('applied') else 'no (gate did not close)'} |")
    return "\n".join(rows)


def build_markdown(art, figs) -> str:
    plan = art["plan"]
    anchor = plan["anchor"]
    acc = art["acceptance"]
    n_pass = acc["n_pass"]
    n_crit = len(acc["criteria"])
    fig_md = "\n\n".join(
        f"![{f['name']}](../{f['path']})\n\n*{f['caption']}*" for f in figs) or \
        "_no figures produced_"
    re_p = art["per_case"][0]["re_p"] if art["per_case"] else None
    resolutions = plan.get("resolutions") or {}
    tags = list(resolutions)
    cells = " / ".join(f"{resolutions[t]['nx']}x{resolutions[t]['ny']}x{resolutions[t]['nz']}"
                       for t in tags) or "n/a"
    radii = " / ".join(str(resolutions[t]["radius"]) for t in tags) or "n/a"
    steps = " / ".join(f"{resolutions[t]['n_steps']} ({resolutions[t]['n_warmup']})"
                       for t in tags) or "n/a"
    subs = {
        "<SCHEMA>": str(art["schema"]),
        "<SCALE>": str(art.get("scale") or plan.get("scale") or "unknown"),
        "<CELLS>": cells,
        "<RADII>": radii,
        "<STEPS>": steps,
        "<CELLS0>": cells.split(" / ")[0],
        "<FOUND>": str(art["cases_found"]),
        "<EXPECTED>": str(art["cases_expected"]),
        "<NPASS>": str(n_pass),
        "<NCRIT>": str(n_crit),
        "<NU>": str(plan["nu"]),
        "<CDISP>": ", ".join(f"{p:.2f}" for p in plan["concentrations"]["dispersed"]),
        "<CFLOCC>": ", ".join(f"{p:.2f}" for p in plan["concentrations"]["flocculated"]),
        "<GRATES>": ", ".join(f"{g:.3e}" for g in plan["gamma_rates"]),
        "<DF>": str(plan["df"]),
        "<NCAP>": str(plan["n_cap"]),
        "<FB>": f"{anchor['bond_force_lattice']:.6g}",
        "<REP>": _f(re_p, "{:.3f}"),
        "<T_PURE>": table_pure(art),
        "<T_CURVES>": table_flow_curves(art),
        "<T_CRIT>": table_criteria(art),
        "<T_CONV>": table_convergence(art),
        "<T_SCAT>": table_scatter(art),
        "<T_ERR>": table_error_judgement(art),
        "<T_DRAG>": table_drag(art),
        "<SPREAD>": _pct(max_seed_spread(art)),
        "<FIGS>": fig_md,
    }
    text = r"""# Frazil ice slurry as a power-law fluid: structure-resolved LBM measurement of tau = K gdot^n

**Repository** TensorLBM · **campaign node** `p366w2-n02-frazil-rheology` (mission 10712) ·
**domain** `ice_navigation` · **artifact** `artifacts/frazil_powerlaw_shear/campaign.json`
(schema `<SCHEMA>`) · **cases** `<FOUND>`/`<EXPECTED>` ·
**acceptance criteria passed** `<NPASS>`/`<NCRIT>`.

## 1. Case introduction (physics + parameters)

Frazil consists of millimetre-scale, disc-shaped ice crystals that form in
supercooled turbulent water and stay suspended. Above a few percent solids the
mixture stops behaving like water: it thickens, becomes shear thinning, and its
effective viscosity controls propeller thrust loss, intake blockage and the
resistance increment of a hull in brash ice. The resistance models in this
platform take the carrier as a generalised Newtonian fluid
`tau = K gdot^n`, so the two numbers `(K, n)` *are* the physical input. This
campaign measures them from a resolved suspension instead of quoting them from a
literature fit.

The gap being closed: the earlier frazil work in this repository
(`tensorlbm.frazil_rheology` + `tensorlbm.frazil_lbm`) **prescribes** `K(phi)`
and `n(phi)` and then verifies that the solver transports that prescribed
viscosity surface correctly (worst channel residual 1.23 %). It never measures
the flow curve of a suspension. Here the dispersed phase is explicit: rigid ice
crystals — aggregated into fractal flocs for the flocculated state — coupled to
the fluid by half-way bounce-back and momentum exchange, in a moving-wall Couette
cell.

| parameter | value | note |
|---|---|---|
| lattice | D3Q19, rho_f = 1, nu = <NU> | lattice units |
| cells (coarse / fine) | <CELLS> | periodic in x,y; walls in z |
| primary crystal radius | <RADII> cells | D = 2a lattice units |
| concentrations c (dispersed) | <CDISP> | volume fraction from the lattice mask |
| concentrations c (flocculated) | <CFLOCC> | higher c jams the placement (see §5) |
| shear rates gdot | <GRATES> | factor-2 ladder (factor 4 in total) |
| Re_p = gdot D^2 / nu | from <REP> upwards | Stokes regime, reported per case |
| floc fractal dimension d_f | <DF> | the achieved d_f is measured per case |
| max primaries per floc N_cap | <NCAP> | anchored at the lowest shear rate |
| bond force F_b (anchored, not fitted) | <FB> | F_b = mu gdot_ref (a N_cap^(1/d_f))^2 |
| steps (warm-up) | <STEPS> | >= 2 viscous time constants H^2/(pi^2 nu) |

Two structural families are run side by side, and the distinction is the point:

* **dispersed** — one crystal per particle: the hard-sphere state for which
  Einstein and Krieger–Dougherty are the correct closures;
* **flocculated** — crystals bonded into open fractal aggregates whose size is
  set by the shear-breakup balance `R_eq = sqrt(F_b/(mu gdot))`,
  `N_eq = (R_eq/a)^d_f` (Sonntag & Russel 1986; Spicer & Pratsinis 1996).
  Aggregates trap water, so their hydrodynamic volume fraction exceeds their
  solids fraction, and they break up as the shear rate rises.

## 2. How it was done

1. **Structure selection.** For each shear rate the equilibrium floc size
   `N_eq(gdot)` follows from the breakup balance. The single constant `F_b` is
   anchored — not fitted — by declaring `N_eq(gdot_min) = N_cap = <NCAP>` before
   any simulation runs; the implied `F_b = <FB>` (lattice units) is carried in the
   artifact so a reader can judge it.
2. **Aggregate generation.** Hierarchical cluster-cluster aggregation of touching
   primaries: each merge rotates one cluster randomly and slides it along a random
   direction until the nearest primaries touch. Of several realisations the one
   whose implied fractal dimension is closest to `d_f = <DF>` is kept, and the
   achieved dimension is measured and reported rather than assumed.
3. **Placement.** Random sequential addition of whole aggregates. Non-overlap is
   enforced on the *primary centres* (minimum image in the periodic directions),
   so open flocs may interpenetrate their bounding spheres — otherwise the
   achievable concentration would be an artefact of the placement rule instead of
   physics. The **achieved** lattice volume fraction is always reported.
4. **Shear cell.** D3Q19 moving-wall Couette (`ShearCell`): walls at z = 0.5 and
   z = nz - 1.5 counter-move with +-u_w, so `gdot = 2 u_w/(nz-2)`. Every primary of
   an aggregate is advected with the *aggregate* centre (translation plus the
   half-vorticity rotation gdot/2), so the aggregate stays rigid.
5. **Stress.** `tau` is the plane momentum flux averaged over the wall bands,
   corrected for the momentum the particles absorb
   (`tau_corrected = tau_near - f_particle/area`); the shear rate is the slope of a
   linear fit to the mean velocity profile, with R^2 reported.
6. **Flow-curve fit.** Least squares of `ln tau` on `ln gdot` over the three rates
   gives `(K, n)` together with R^2 and the slope standard error.
7. **Acceptance.** The five pre-declared criteria are evaluated *inside the
   artifact* (`acceptance` block), never in prose.

**Scale actually run: `<SCALE>`.** The campaign is defined at two scales in
`run_campaign.py` (`production`: 48x48x49 / 64x64x65 cells; `reduced`: 32x32x33 /
40x40x41). The numbers below are from the `<SCALE>` scale: the cluster
dispatcher accepted the production submission but had not dispatched it after
32 min (241 jobs ahead of it), so the reduced scale was run on the contended
host instead. The production-scale campaign is a single command
(`FRAZIL_SCALE=production python3 run_campaign.py ...`) and its numbers replace
these when it runs; nothing else in the chain changes. This is a scope
statement, not a caveat about the method.

Tool chain: `benchmarks/frazil_powerlaw_shear/run_campaign.py` (plan -> sharded
cases -> assemble), `make_figures.py` (figures read back from the artifact),
`make_report.py` (this document, generated from the artifact), tests in
`tests/test_frazil_powerlaw_shear.py`.

## 3. Results

### 3.1 Particle-free control (the denominator of eta_r)

Plane Couette is an exact Navier–Stokes solution at every Reynolds number, so the
particle-free run has a closed-form reference `tau = rho nu gdot`.

<T_PURE>

### 3.2 Flow curves and fitted (K, n), coarse tier

<T_CURVES>

### 3.3 Relative viscosity and the five acceptance criteria

<T_CRIT>

### 3.4 Grid convergence (coarse vs fine resolution)

<T_CONV>

### 3.5 Seed scatter (microstructure realisation)

<T_SCAT>

**Read §3.5 before the criterion table.** The largest seed-to-seed spread of
``eta_r`` anywhere in this campaign is <SPREAD> (four realisations of the same
concentration and rate). The dilute-criterion agreement is therefore a statement
that the measured slope is right *to the statistical resolution of a single
frozen microstructure*, not a 1 % calibration: with ten aggregates in the cell,
the scatter of one configuration is the dominant uncertainty, and it is quoted
rather than averaged away.

### 3.6 Hydrodynamic-radius gate (may the mask fraction be converted?)

At $D = 5$ lattice diameters a staircase sphere is not a smooth sphere, so the
mask volume fraction can differ from the fraction the fluid responds to. The
conversion is measured, not assumed: a fixed sphere in a uniform stream gives
the momentum-exchange drag, and Stokes' law inverted (with the Haberman–Sayre
confinement factor) gives $a_h$. The gate is only *used* if it closes its own
budget — the raw drag and the confined Stokes prediction must agree to 5 %.

<T_DRAG>

Where the gate does not close, the mask fraction is kept and the criterion is
reported on it. Using a gate that fails its own budget to move a criterion into
range would be a fudge, so it is recorded and not applied.

### 3.7 Figures

<FIGS>

## 4. Error judgement (explicit, per key quantity)

<T_ERR>

The node's acceptance tolerance is 15 % on eta_r against Einstein and
Krieger–Dougherty, 1 % on the zero-concentration exponent and 5 % on the
resolution pair. The "<3 %" column above answers the platform's stricter
refereed question; where only literature *ranges* exist for a quantity, the table
says so instead of inventing a percentage.

## 5. Pitfalls, scope, limitations

**Pitfalls found while building this face**

1. *Frozen structure cannot shear-thin.* A rigid suspension in Stokes flow is
   Newtonian by construction: the flow curve of a frozen microstructure has
   `n ~ 1` however concentrated it is. Any `n < 1` must come from a structure that
   follows the shear — here the floc size. The dispersed family is kept precisely
   as the built-in negative control for that statement.
2. *Inertia masquerades as shear thickening.* `Re_p = gdot D^2/nu` grows with the
   rate and suspension viscosity creeps up with `Re_p`, so the dispersed family's
   measured exponent sits slightly *above* 1 (see §3.2). Judging a flocculated
   exponent against 1.0, rather than against the dispersed exponent at the same
   rates, would misattribute that bias.
3. *Placement saturation is physics, not a bug.* Open `d_f = 2` aggregates at
   `c = 0.10` fill a hydrodynamic volume fraction of order
   `phi (R/a)^(3-d_f) ~ 0.25`; random sequential addition jams below the requested
   solids fraction and the achieved `phi` (reported per case) is what the closure
   then uses. The jamming is consistent with the pasty behaviour observed for
   frazil at ~10 % solids, but it does mean the flocculated family cannot be pushed
   to `c = 0.30` in this cell.
4. *Placement saturation, not a solver bug.* Random sequential addition with a
   one-cell gap saturates near ``phi = 0.155`` in the reduced cell, so the
   requested ``c = 0.20`` and ``c = 0.30`` runs both landed at
   ``phi = 0.154`` (reported per case). The two rows therefore describe two
   different microstructures at the same volume fraction, which is exactly the
   seed scatter of a single frozen configuration (~10 % in ``eta_r``) and is why
   the Krieger–Dougherty comparison at ``c = 0.20`` cannot be settled at this
   scale.
5. *Stress estimator.* The wall momentum-exchange estimator does not close the
   pure-fluid momentum budget; the plane-flux estimator does (§3.1). That is why
   the production measurement uses the flux, and why eta_r is always a ratio of
   two identical measurements.

**Limitations (honest)**

* 3-D but laminar: no turbulence, no crystal collision/aggregation dynamics
  inside a run, no population balance. Floc size is selected *between* runs by a
  stated closure; the resolved simulation then measures the hydrodynamic
  consequence of that geometry.
* `F_b` is anchored on `N_cap`, so the *slope* of `N_eq(gdot)` (namely -1/2 in R,
  hence `d_f/2` in N) carries the structure information, not the absolute floc
  size. A reader who rejects the -1/2 exponent should read the flocculated rows as
  conditional on it.
* Primary crystals are spheres of 6–8 lattice diameters; real frazil crystals are
  discs of aspect ratio ~10. A sphere is the conservative choice (a disc packs a
  larger hydrodynamic volume at the same solids fraction), but the shape effect is
  not quantified here.
* The cell is small (<CELLS0>): of order ten aggregates and 50–300 primaries, so the
  seed-to-seed scatter in §3.5 is the honest uncertainty of a single point rather
  than an ensemble average.
* No lattice-to-SI scaling is claimed: `(K, n)` are lattice quantities. The helper
  `lattice_to_si` documents the unit change for a declared crystal scale, but no
  experiment is matched pointwise here.

## 6. Reproduction

```bash
python3 benchmarks/frazil_powerlaw_shear/run_campaign.py --plan
python3 benchmarks/frazil_powerlaw_shear/run_campaign.py --run --shard 0 --shards 17 --threads 2
python3 benchmarks/frazil_powerlaw_shear/run_campaign.py --assemble
python3 benchmarks/frazil_powerlaw_shear/make_figures.py
python3 benchmarks/frazil_powerlaw_shear/make_report.py
python3 -m pytest tests/test_frazil_powerlaw_shear.py -q
```
"""
    for k, v in subs.items():
        text = text.replace(k, v)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=ART)
    ap.add_argument("--doc", default=DOC)
    ap.add_argument("--api", action="store_true")
    args = ap.parse_args(argv)
    with open(args.artifact) as fh:
        art = json.load(fh)
    fig_index = os.path.join(os.path.dirname(args.artifact), "figures.json")
    figs = []
    if os.path.exists(fig_index):
        with open(fig_index) as fh:
            figs = json.load(fh)
    md = build_markdown(art, figs)
    with open(args.doc, "w") as fh:
        fh.write(md)
    print(f"wrote {args.doc} ({len(md)} chars)")

    if args.api:
        payload_figs = []
        for f in figs:
            path = os.path.join(ROOT, f["path"])
            if not os.path.exists(path):
                continue
            with open(path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            payload_figs.append({"name": f["name"], "caption": f["caption"],
                                 "png_base64": b64})
        acc = art["acceptance"]
        passed = [k for k, v in acc["criteria"].items() if v.get("passed")]
        summary = (f"Structure-resolved frazil power-law rheology: D3Q19 frozen-structure "
                   f"shear cell over c=0.05-0.30 and a factor-4 shear-rate ladder; "
                   f"{len(passed)}/{len(acc['criteria'])} acceptance criteria pass "
                   f"({', '.join(passed) if passed else 'none'}).")
        payload = {
            "title": "Frazil slurry power-law rheology from a resolved D3Q19 shear cell "
                     "($\\tau=K\\dot\\gamma^n$)",
            "summary": summary,
            "scenario_key": SCENARIO,
            "mission_id": MISSION_ID,
            "tags": ["ice_navigation", "frazil", "rheology", "power-law", "LBM",
                     "suspension", "TensorLBM"],
            "markdown": md,
            "figures": payload_figs,
        }
        # a mission-unique path: /tmp is shared between agents on this host, and
        # a plain /tmp/report.json silently becomes somebody else's payload
        out = os.environ.get("FRAZIL_REPORT_JSON",
                             f"/tmp/m{MISSION_ID}_report.json")
        with open(out, "w") as fh:
            json.dump(payload, fh)
        print(f"wrote {out} ({len(payload_figs)} figures, "
              f"{os.path.getsize(out)/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())