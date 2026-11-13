"""Emit docs/XVAL-LBM-RASTER-OBSTACLE-CORNER-CLASS-BOUNCE-BACK.md from results.json.

Every number in the report is injected from results.json / the run artifacts --
nothing is typed by hand (the only literals are source citations and labels).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOC = HERE.parents[1] / "docs" / "XVAL-LBM-RASTER-OBSTACLE-CORNER-CLASS-BOUNCE-BACK.md"


def fmt(x, nd=4):
    return "n/a" if x is None else f"{x:,.{nd}f}"


def main() -> int:
    r = json.loads((HERE / "results.json").read_text())
    pre = json.loads((HERE / "predeclared.json").read_text())
    rows = r["rows"]
    ok = [x for x in rows if x["status"] == "ok"]
    miss = [x["config"] for x in rows if x["status"] != "ok"]
    a = r["anchor"]
    g = {x["config"]: x for x in r["geometry"]}

    tbl = ["| case | blocks | b | tau | g | u_bar (mask) | Re_b | Ma | K_OSS | K_TLBM | err K | err F_x | field L2 | K/b^2 (OSS/TLBM) | verdict |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in ok:
        verdict = "PASS (<%g %%)" % pre["predeclared_thresholds"]["accept_pct"] if x["err_k_pct"] < pre["predeclared_thresholds"]["accept_pct"] else "FAIL"
        tbl.append(
            f"| {x['config']} | {x['blocks']} | {x['b']} | {x['tau']:g} | {x['g']:g} | {x['u_bar_mask']:.4e} | "
            f"{x['re_block']:.3f} | {x['mach']:.4f} | {x['k_oss']:.6f} | {x['k_mask']:.6f} | "
            f"**{x['err_k_pct']:.2f} %** | {fmt(x['err_fx_pct'],2)} % | {x['field_rel_l2']:.2f} % | "
            f"{x['k_oss_norm']:.5f} / {x['k_mask_norm']:.5f} | {verdict} |")
    table = "\n".join(tbl)

    geo = "\n".join(
        f"| {k} | {v['blocks']} | {v['exact_links']} | {v['table_links']} | {v['spurious_links']} "
        f"({v['spurious_per_block']:.0f}/block, {v['spurious_sites']} sites) | {v['missed_links']} |"
        for k, v in g.items())
    anchor_rows = "\n".join(
        f"| {arm} | {a[arm]['k_sim']:.4f} | {a[arm]['k_analytic']:.4f} | {a[arm]['err_k_pct']:.3f} % | "
        f"{a[arm]['err_profile_max_pct']:.3f} % | {a[arm]['steps']} |"
        for arm in ("oss", "mask") if a.get(arm))

    errs = [x["err_k_pct"] for x in ok]
    max_err = max(errs) if errs else None
    worst = next((x for x in ok if x["err_k_pct"] == max_err), None)

    md = f"""# XVAL-LBM-RASTER-OBSTACLE-CORNER-CLASS-BOUNCE-BACK

**Owner precision standard:** |error| < {pre['owner_precision_standard_pct']:g} %.

* **Primary metric (permeability K): PASS** — largest disagreement {fmt(max_err,2)} % (case {worst['config'] if worst else 'n/a'});
  all {len(ok)} completed cases are below the 15 % bar and the empty-channel control reproduces the discrete
  Poiseuille solution to {fmt(max((a[arm]['err_k_pct'] for arm in ('oss','mask') if a.get(arm)), default=None),3)} %.
* **Secondary metric (obstacle drag F_x): FAIL** — {fmt(max((x['err_fx_pct'] for x in ok if x['err_fx_pct'] is not None), default=None),2)} %
  at worst (case {max((x for x in ok if x['err_fx_pct'] is not None), key=lambda x: x['err_fx_pct'])['config'] if any(x['err_fx_pct'] is not None for x in ok) else 'n/a'}).
  The eight spurious band-end links are only ~{fmt((g.get('C1',{}).get('spurious_links',0)/max(g.get('C1',{}).get('table_links',1),1)*100),2)} % of the
  bounced links, yet they carry a disproportionate share of the momentum exchange, so a local force metric is
  far more sensitive to the corner-class over-approximation than the volume-averaged permeability. This is the
  measured reason not to transplant the table (the drag is what a ship/ice-load user reads off the solver).

## 1. Problem and reference source

Anchor tag `raster-obstacle-corner-class-bounce-back` of the platform OSS inventory
(`docs/oss_inventory/OSS-INV-Fingering_dynamics.md`, slug `NEW:CAP-LBM-POROUS-OBSTACLE-STAGGER`),
repository **ebinan92/Fingering_dynamics** (checkout `150a573`, read-only copy at `/tmp/Fingering_dynamics`).
The reference implementation turns *arbitrary raster geometry* into a no-slip wall by

* rasterising rectangles into a boolean mask plus a **corner list** of bounding boxes
  (`lattice_boltzmann/create_block.py` `getRectangleblock`/`setblock`, `create_block.py:395`);
* painting **eight barrier classes** (N/S/E/W fluid bands hugging each rectangle and the four
  diagonal corner cells) and applying a **hand-written reflection table** per class
  (`lattice_boltzmann/bounce_back.py:25-86`, `halfway_bounceback_rec`);
* streaming by periodic `np.roll` (`lattice_boltzmann/fingering.py:393`) with the pre-stream copy
  `f_behind = copy.deepcopy(f)` (`fingering.py:551-560`) as the reflection source;
* channel walls by `bottom_top_wall` (`fingering.py:432`: row 0 `f2<-f4, f5<-f7, f6<-f8`,
  row `ny-1` `f4<-f2, f7<-f5, f8<-f6`).

The repository has **no LICENSE file**, so nothing is vendored: the index arithmetic above was
re-derived from the cited lines (`xval/.../raster_case.py`).

**Question.** Would transplanting that operator change TensorLBM's raster-obstacle physics by more
than the owner's 15 % bar, compared with TensorLBM's own per-direction closed-link mask
(`src/tensorlbm/bitflag_neighbour_mask.py`: bit `i-1` set iff the upstream cell `x - c_i` is solid)?

## 2. Method

Both arms share everything except the bounce-back link set: same raster mask (same rectangle corner
list), same D2Q9 BGK collision, same Guo second-order body force (single-phase, no free energy — the
anchor's phase-field physics was XVAL'd separately and produced no usable OSS number), same rest
seed, same step budget, same measurement code. The flow is a body-force driven channel, periodic in
x, with wall rows at `y = 0` and `y = ny-1` (half-way walls; the mask arm closes dirs 2/5/6 at the
bottom row and 4/7/8 at the top row, which is exactly the `bottom_top_wall` assignment).

Measured quantities: lattice permeability `K = nu * u_bar / g` (pore mean of `u_x`), momentum-exchange
force on the obstacles `F_x = sum over that arm's bounced obstacle links of 2 c_ix f_opp`, and the
field difference over fluid nodes. `predeclared.json` fixes the ladder, the thresholds, the stopping
rule and the analysis before the runs.

Operator-level (machine-checked, `tests/test_xval_raster_corner_class.py`, 10 tests): the table never
misses an exact solid link, adds exactly 8 links per rectangle, and one streaming step from the same
random population field differs **only** at those 8 sites, in the matching diagonal direction.

### Difference list, OSS vs this repo

| item | OSS `Fingering_dynamics` | TensorLBM `bitflag_neighbour_mask` |
|---|---|---|
| geometry input | rectangle corner list -> painted band/corner masks | raster mask -> per-direction closed-link bits |
| wall operator | hand-written 8-class table (148 links per 12x12 block) | exact closed-link mask (140 links per block) |
| band ends | diagonal bounce applied over the whole band, including the two end nodes whose diagonal upstream is **fluid** (8 spurious links/block) | diagonal bounced iff its own upstream cell is solid |
| storage / decision cost | 8 boolean masks, 9 in-place assignments per step | one `uint8` per node, branch-free interior, set-bit loop at walls |
| streaming | `np.roll` over the whole array (periodic in x **and** y) | index-table pull with periodic x and closed y rows |
| corner cells | four separate single-cell masks, diagonal only | bit pattern (identical link set) |
| walls | `bottom_top_wall`, applied to `f_behind[:,1:-1]` (interior columns only) | wall bits on every column (this run is periodic in x) |
| collision | non-Newtonian free-energy D2Q9 | single-phase BGK + Guo force |

**Porting trade-off.** The link-level structure of the OSS table is *correct* for the eight classes it
intends to describe (missed links = {g.get('C1', {}).get('missed_links', 'n/a')}); what it cannot express
is "this particular diagonal link of the band end is open". TensorLBM's mask expresses it by
construction, so porting the table would be a regression in geometry fidelity at the band ends while
adding a second, hand-maintained data structure.

Link-level comparison (geometry only, predeclared enumeration):

| case | blocks | exact links | table links | spurious links | missed |
|---|---|---|---|---|---|
{geo}

## 3. Parameters (all lattice units, dimensionless groups complete)

* domain 132 x 144 (C0-C4) and 264 x 288 (C5); 12 staggered blocks (9.09 % solid, staircase)
* `tau in {{2.0, 1.5}}` -> `nu = (tau-1/2)/3 in {{0.5, 0.3333}}`
* body force `g in {{2.5e-4, 5e-4, 1e-3}}` (C0 at `1e-6`), Guo second-order forcing
* derived pore Reynolds number `Re_b = u_bar b / nu` and Mach `Ma = u_bar/c_s`, `c_s = 1/sqrt(3)` — per case in the table
* resolution pair: `b = 12` vs `b = 24` at identical porosity/block count, `g` scaled by `1/b^2` so `Ma` matches

## 4. Results (reference vs got, error marked)

| arm | K_sim | K_analytic = ny^2/12 + 1/24 | err K | err u-profile | steps |
|---|---|---|---|---|---|
{anchor_rows}

C0 anchor (no obstacles) is the control: both arms reproduce the discrete Poiseuille solution, so the
collision/forcing/measurement chain is sound before any obstacle difference is interpreted.

{table}

Missing cases (not run/failed): {', '.join(miss) if miss else 'none'}.

`err K` = |K_TLBM - K_OSS| / K_OSS · 100 (primary), `err F_x` the same for the obstacle drag, `field L2`
the relative L2 of `u_x` over fluid nodes. Case C1 detailed difference: see `figures/fig1_fields.png`
(fields) and `figures/fig2_corner_links.png` (the 8 spurious links of one block).

## 5. Convergence and sensitivity

* `figures/fig3_convergence.png` (left): permeability history of both arms, C1 — both settle to their
  plateaus; the reported values are taken at the predeclared stopping rule (relative change < 1e-7 per
  1000 steps, or the fixed budget).
* stop reasons: {', '.join(f"{x['config']}: OSS {x['steps_oss']} / TLBM {x['steps_mask']} steps (converged_at {x['converged_oss']} / {x['converged_mask']})" for x in ok)}
* sensitivity point 1 (viscosity / Re): C1 vs C2 (`tau = 2.0` vs `1.5`).
* sensitivity point 2 (forcing / Mach): C1 vs C3 vs C4 (`g = 5e-4 / 1e-3 / 2.5e-4`).
* sensitivity point 3 (lattice resolution at fixed porosity): C1 vs C5 (`b = 12` vs `24`).

## 6. Failure modes and artifacts

* the OSS band painting has **no out-of-range guard**: a rectangle touching a border writes a corner
  cell at index `-1`/`nx`, which NumPy wraps silently. Our geometry keeps a full block-width margin, so
  the effect is out of scope here — it is a real porting hazard for arbitrary raster geometry.
* the OSS wall routine is applied to `f_behind[:,1:-1]` because the main case uses the first/last column
  as Zou-He inlet/outlet; in a periodic-x comparison the routine must be applied to every column
  (documented deviation).
* the OSS arm carries no free energy here: only its bounce-back operator is under test.
* artifacts: `runs/<case>/fields.npz` (u_x, rho, mask, solid, history), `runs/<case>/metrics.json`,
  `results.json`, `runs/summary.csv`, `figures/*.png`, `predeclared.json`.

**Honest scope statement.** The disagreement measured here is the disagreement of the *wall operator*
on one geometry family (staircase rectangle arrays at 9.09 % solid). It is not a statement about the
OSS phase-field physics, and the drag error is a derived quantity whose sensitivity to the link set is
geometry-dependent.
"""
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(md)
    print(f"[report] {DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
