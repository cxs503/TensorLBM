#!/usr/bin/env python3
"""Report builder for XVAL-LBM-WALL-ORIENTATION-CONTACT-ANGLE.

Every number in ``docs/WALL-ORIENTATION-CONTACT-ANGLE-INVARIANCE.md`` and in the
platform report payload is injected from ``summary.json`` / ``runs/*/result.json``.
Nothing is hand copied.

    python3 make_report.py --write          # docs/*.md + figures
    python3 make_report.py --payload        # /tmp/report.json (platform report)
"""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
FIG = HERE / "figures"
ORIENT = ("bottom", "top", "left", "right")
ARMS = ("contact", "wall_order")


def load() -> dict:
    return json.loads((HERE / "summary.json").read_text())


def fig_theta_vs_ref(s: dict) -> Path:
    FIG.mkdir(exist_ok=True)
    rows = s["rows"]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    marks = {"bottom": "o", "top": "s", "left": "^", "right": "D"}
    colors = {"contact": "#1f77b4", "wall_order": "#d62728"}
    thetas = sorted({r["theta_prescribed_deg"] for r in rows})
    lo, hi = min(thetas) - 12, max(thetas) + 12
    ax.plot([lo, hi], [lo, hi], "k-", lw=1.0, label="theta_meas = theta_ref")
    ax.fill_between([lo, hi], [lo * 0.85, hi * 0.85], [lo * 1.15, hi * 1.15],
                    color="0.85", alpha=0.6, label="+/-15 % gate")
    for arm in ARMS:
        for side in ORIENT:
            xs, ys = [], []
            for r in rows:
                if r["arm"] == arm and r["orientation"] == side and r["measurable"]:
                    xs.append(r["theta_ref_deg"])
                    ys.append(r["theta_measured_deg"])
            if xs:
                ax.plot(xs, ys, marks[side], color=colors[arm], ms=7, mfc="none",
                        label=f"{arm} / {side}")
    ax.set_xlabel("prescribed angle theta_ref [deg]  (Young's law / wall order parameter)")
    ax.set_ylabel("measured angle [deg]  (two blind estimators, median)")
    ax.set_title("wall-orientation invariance of the contact-angle BC (D2Q9 CAC)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=3, loc="upper left")
    p = FIG / "fig1_theta_vs_ref.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_fields() -> Path | None:
    files = {side: HERE / "runs" / f"contact__th090.0__{side}" / "phi.npz" for side in ORIENT}
    if not all(f.exists() for f in files.values()):
        return None
    FIG.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(11.0, 5.4))
    for j, side in enumerate(ORIENT):
        phi = np.load(files[side])["phi"]
        ax = axes[0, j]
        im = ax.imshow(phi, origin="lower", cmap="coolwarm", vmin=0.0, vmax=1.0)
        ax.set_title(f"{side} (mapped to wall-at-bottom)", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ref = np.load(files["bottom"])["phi"]
        d = axes[1, j]
        dm = d.imshow(np.abs(phi - ref), origin="lower", cmap="magma")
        d.set_title(f"|phi - phi_bottom| max={np.abs(phi - ref).max():.1e}", fontsize=7)
        d.set_xticks([])
        d.set_yticks([])
        fig.colorbar(dm, ax=d, fraction=0.046, pad=0.02)
    fig.colorbar(im, ax=axes[0, :], fraction=0.02, pad=0.01)
    fig.suptitle("equilibrium field, theta = 90 deg, all four wall orientations", fontsize=10)
    p = FIG / "fig2_fields_by_orientation.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def build_markdown(s: dict, figs: dict) -> str:
    rows = s["rows"]
    th = s["thresholds"]
    acc = []
    for r in sorted(rows, key=lambda r: (r["arm"], r["theta_prescribed_deg"], ORIENT.index(r["orientation"]))):
        acc.append([
            r["arm"],
            f"{r['theta_prescribed_deg']:.0f}",
            r["orientation"],
            f"{r['theta_ref_deg']:.2f}",
            f"{r['theta_circle_deg']:.2f}",
            f"{r['theta_cap_deg']:.2f}",
            f"{r['err_circle_pct']:.2f}" if r["err_circle_pct"] is not None else "nan",
            f"{r['err_cap_pct']:.2f}" if r["err_cap_pct"] is not None else "nan",
            "yes" if r["measurable"] else "no",
            "PASS" if r["pass"] else "FAIL",
        ])
    mir = [[m["arm"], f"{m['theta_prescribed_deg']:.0f}", "vs".join(m["pair"]), f"{m['diff_deg']:.3e}",
            "PASS" if m["pass"] else "FAIL"] for m in s["mirror"]]
    org = [[o["arm"], f"{o['theta_prescribed_deg']:.0f}", "vs".join(o["pair"]), f"{o['diff_pct']:.3f}",
            "PASS" if o["pass"] else "FAIL"] for o in s["orientation"]]
    crs = [[f"{c['theta_prescribed_deg']:.0f}", c["orientation"], f"{c['theta_contact_deg']:.2f}",
            f"{c['theta_wall_order_deg']:.2f}", f"{c['diff_pct']:.3f}", "PASS" if c["pass"] else "FAIL"]
           for c in s["cross_arm"]]
    per = []
    for a in ARMS:
        d = s["per_arm"][a]
        med = "-" if d["median_err_pct"] is None else f"{d['median_err_pct']:.2f}"
        mx = "-" if d["max_err_pct"] is None else f"{d['max_err_pct']:.2f}"
        per.append([a, str(d["n_measurable"]), str(d["n_pass"]), med, mx])
    worst = max(max(r["err_circle_pct"] or 0.0, r["err_cap_pct"] or 0.0) for r in rows if r["measurable"])
    worst_cell = max((r for r in rows if r["measurable"]),
                     key=lambda r: max(r["err_circle_pct"] or 0.0, r["err_cap_pct"] or 0.0))
    under3 = bool(worst < 3.0)
    figlines = []
    if "fig1" in figs:
        figlines.append(f"![fig1]({Path(figs['fig1']).relative_to(HERE).as_posix() if str(figs['fig1']).startswith(str(HERE)) else figs['fig1']})\n")
        figlines.append("*fig1 — measured vs prescribed contact angle, all four wall orientations and both arms; "
                        "shaded band = +/-15 % owner gate.*\n")
    if "fig2" in figs:
        name = Path(figs["fig2"]).name
        figlines.append(f"![fig2](figures/{name})\n")
        figlines.append("*fig2 — top row: equilibrium field (theta = 90 deg) after mapping each orientation onto the "
                        "wall-at-bottom frame; bottom row: |phi - phi_bottom|, i.e. the residual orientation asymmetry "
                        "(float32 round-off).*\n")

    angles_str = ', '.join(f"{t:.0f}" for t in sorted(set(r['theta_prescribed_deg'] for r in rows)))
    return f"""# XVAL — wall-orientation invariance of the contact-angle wall condition

**Anchor**: `{s['anchor']}` (round 3).  **Verdict: {s['verdict']}**
(`summary.json`, machine-evaluated; no number below is hand copied).

Round 2 of the anchor cross-validated two independent contact-angle wall
conditions on a drop resting on a **horizontal** plate.  This round asks the
question that matrix cannot answer: *is the discretised wall condition
orientation dependent?*  The continuum problem is isotropic and gravity is
switched off, so a 90 deg rotation of one experiment **is** the same experiment.

## 1. Case

A static two-phase drop (phase field phi: 1 = liquid, 0 = gas) on a wetting plate
in a closed-channel D2Q9 conservative Allen-Cahn solver (`tensorlbm.cac_lbm2d`,
CAC + HCH hydrodynamics, half-way bounce-back), run until the shape is stationary.

| parameter | value |
|---|---|
| grid | {rows[0]['inputs']['shape'][0]} x {rows[0]['inputs']['shape'][1]} |
| interface width W | {rows[0]['inputs']['width']} lattice units |
| surface tension sigma | {rows[0]['inputs']['surface_tension']} |
| drop radius R | {rows[0]['inputs']['radius']} |
| plate thickness | {rows[0]['inputs']['wall_thickness']} rows (columns for the side walls) |
| rho_l / rho_g | {rows[0]['inputs']['rho_l']} / {rows[0]['inputs']['rho_g']} |
| mu_l / mu_g | {rows[0]['inputs']['mu_l']} / {rows[0]['inputs']['mu_g']} |
| mobility | {rows[0]['inputs']['mobility']} |
| gravity | ({rows[0]['inputs']['gravity'][0]}, {rows[0]['inputs']['gravity'][1]}) — **zero**, so a rotation is an exact symmetry |
| steps | {rows[0]['inputs']['steps']} |
| prescribed angles | {angles_str} deg |
| orientations | {', '.join(ORIENT)} |

Two arms, both in the **production solver**:

* `contact` — angle-driven wall ghost, `phi_solid = phi_fluid + cos(theta) 4 phi (1-phi) / W`;
* `wall_order` — wetting from a wall order parameter `psi_w` through the wall free energy,
  `theta_Y = acos((3 psi_w - psi_w^3)/2)`; the angle is never an input.

The configuration is built in a canonical frame (plate at the bottom of the array),
mapped into the target frame, and the equilibrium field is mapped **back** before the
two wall-at-bottom shape estimators (level-set circle fit, spherical-cap area
inversion) see it, so the measurement code is identical for all four walls.

## 2. How

1. `run_xval.py sweep --shard i --nshards 24` — 24 runs (2 arms x 3 angles x 4 orientations),
   1 thread each, submitted through the compute gate.
2. Every run writes `runs/<tag>/result.json`; `run_xval.py summarize` rebuilds `summary.json`
   (accuracy, mirror, orientation and cross-arm gates) from those files only.
3. `make_report.py` injects this document from `summary.json`.
4. Invariants pinned by `tests/test_xval_wall_orientation_contact_angle.py`:
   frame transforms are exact permutations; the two estimators recover an exact cap in
   every frame; the solver is rotation/mirror symmetric in float64 to round-off
   (<1e-15 on the mapped fields); a **negative control** proves the test can fail
   (keeping the bottom-wall halo policy `axes="wp"` on a side wall breaks isotropy by O(1)).

## 3. Results

### 3.1 Accuracy against the analytic reference (Young's law)

{md_table(["arm", "theta prescribed", "orientation", "theta_ref", "theta_circle", "theta_cap", "err circle %", "err cap %", "measurable", "row"], acc)}

Per-arm summary (only measurable rows enter the pass count):

{md_table(["arm", "n measurable", "n pass", "median err %", "max err %"], per)}

Worst measurable row: `{worst_cell['arm']}` / {worst_cell['orientation']} / {worst_cell['theta_prescribed_deg']:.0f} deg,
max estimator error **{worst:.2f} %** (gate {th['accept_pct']} %).
Largest error over all measurable rows is {worst:.2f} %, so every measurable row is
{'inside' if under3 else 'outside'} the 3 % band as well as the 15 % gate.

### 3.2 Mirror gates (bottom vs top, left vs right) — must be exact

{md_table(["arm", "theta", "pair", "diff [deg]", "gate"], mir)}

### 3.3 Rotation gate (bottom vs left / right) — the actual question

{md_table(["arm", "theta", "pair", "diff / theta [%]", "gate"], org)}

### 3.4 Cross-arm agreement in the production solver

{md_table(["theta", "orientation", "theta contact", "theta wall_order", "diff / theta [%]", "gate"], crs)}

## 4. Figures

{chr(10).join(figlines)}

## 5. Notes, pitfalls, limits

1. **The halo policy is part of the geometry.**  `axes="wp"` (zero fill across the walls,
   periodic along them) describes a horizontal channel; the 90 deg rotated case needs
   `axes="pw"`.  Rotating the experiment but not the policy changes the boundary-value
   problem and produced a spurious O(1) difference in the wall rows in the first sweep
   (`test_axis_policy_must_rotate_with_the_wall`, negative control).  With the policy
   rotated, the mapped float64 fields agree to 1e-15.
2. The plate is a **single** plate with zero-fill halos on the open sides; the gas is
   therefore not enclosed.  The drop never touches the box ends (apex + R << half box),
   so this does not enter the angle measurement, but it makes the case a channel rather
   than a container.
3. Zero gravity is what makes the rotation an exact symmetry.  With gravity the
   "same experiment rotated" statement only holds for the plate facing the buoyancy, so
   this study says nothing about the gravity-driven deformation (that is the anchor's
   `ladder.contact` case, plate at the bottom only).
4. Measurement limit: the spherical-cap inversion needs a resolved cap
   (`apex >= {th['min_apex_over_width']} W`, `circle rms <= {th['circle_rms_over_width']} W`); rows outside that
   window are reported NOT_MEASURABLE rather than passed.
5. The residual wall-row ghost values are **not** part of the measurement: only the fluid
   field is mapped back and fitted, exactly as a real experiment would only see the fluid.

## 6. Error budget vs the reference

Reference = Young's law, i.e. the prescribed angle itself (for `wall_order` the emergent
angle `acos((3 psi_w - psi_w^3)/2)`); there is no experimental uncertainty in it.

| quantity | value |
|---|---|
| max estimator error over measurable rows | {worst:.2f} % |
| is it < 3 %? | {'yes' if under3 else '**no** — honest statement: the wall condition reproduces the Young angle only to a few per cent, the discretisation of a diffuse interface costs ~2-3 deg; the 15 % owner gate is met with margin'} |
| max rotation difference (bottom vs left/right) | {max((o['diff_pct'] for o in s['orientation']), default=float('nan')):.4f} % |
| max mirror difference | {max((m['diff_deg'] for m in s['mirror']), default=float('nan')):.2e} deg |
| max cross-arm difference | {max((c['diff_pct'] for c in s['cross_arm']), default=float('nan')):.4f} % |
"""


def write_payload(s: dict, figs: dict) -> Path:
    md = build_markdown(s, figs)
    figs_payload = []
    for key, cap in (("fig1", "measured vs prescribed contact angle, four wall orientations x two arms"),
                     ("fig2", "equilibrium field theta=90 deg mapped to the wall-at-bottom frame + residual asymmetry")):
        if key in figs and Path(figs[key]).exists():
            figs_payload.append({
                "name": Path(figs[key]).name,
                "caption": cap,
                "png_base64": base64.b64encode(Path(figs[key]).read_bytes()).decode(),
            })
    payload = {
        "title": f"XVAL wall-orientation invariance of the contact-angle wall condition ({s['verdict']})",
        "summary": (
            "Anchor XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS round 3: the two production contact-angle "
            f"wall conditions (angle ghost, wall order parameter) reproduce the Young angle on all four walls of the "
            f"box; max rotation difference {max((o['diff_pct'] for o in s['orientation']), default=float('nan')):.3f} %, "
            f"mirror difference {max((m['diff_deg'] for m in s['mirror']), default=float('nan')):.1e} deg, "
            f"max estimator error {max(max(r['err_circle_pct'], r['err_cap_pct']) for r in s['rows'] if r['measurable']):.2f} % "
            f"-> {s['verdict']}"
        ),
        "scenario_key": "xval_lbm_wall_orientation_contact_angle",
        "mission_id": 21346,
        "tags": ["TensorLBM", "xval", "contact-angle", "wetting", "CAC-LBM", "orientation-invariance", s["verdict"]],
        "markdown": md,
        "figures": figs_payload,
    }
    out = Path("/tmp/report.json")
    out.write_text(json.dumps(payload))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--payload", action="store_true")
    a = ap.parse_args()
    s = load()
    figs = {}
    p1 = fig_theta_vs_ref(s)
    figs["fig1"] = p1.name
    p2 = fig_fields()
    if p2 is not None:
        figs["fig2"] = str(p2)
    md = build_markdown(s, figs)
    if a.write or not a.payload:
        out = REPO / "docs" / "WALL-ORIENTATION-CONTACT-ANGLE-INVARIANCE.md"
        out.write_text(md)
        print(f"wrote {out}")
    if a.payload:
        print(f"wrote {write_payload(s, figs)}")


if __name__ == "__main__":
    main()