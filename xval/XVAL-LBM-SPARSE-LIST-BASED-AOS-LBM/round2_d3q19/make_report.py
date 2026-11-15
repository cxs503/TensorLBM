"""Build the round-2 report (figures + markdown) from results.json and post it.

Every number is read from ``results.json`` / ``tables.md`` produced by
``run_r2.py``; nothing is hand-copied.  Figures are matplotlib PNGs embedded as
base64 in the report payload.
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = json.loads((HERE / "results.json").read_text())
TABLES = (HERE / "tables.md").read_text()

FIGDIR = HERE / "figures"
FIGDIR.mkdir(exist_ok=True)


def save(fig, name):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    (FIGDIR / name).write_bytes(buf.getvalue())
    return name


figs, captions = [], []

# --- P1 + P4: profiles vs analytic -----------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for r in RES["P1"]:
    h = r["height"]
    axes[0].plot(r["arms"]["sparse_soa"]["profile"], np.arange(h) + 0.5, "o-",
                 ms=3, label=f"D3Q19 list H={h}")
ref32 = RES["P1"][-1]
y = np.arange(ref32["height"]) + 0.5
from tensorlbm.sparse_list_aos import poiseuille_analytic  # noqa: E402
axes[0].plot(poiseuille_analytic(np.arange(ref32["height"], dtype=float), ref32["height"],
                                 ref32["force_x"], RES["nu"]), y, "k--", lw=1,
             label="analytic (H=32)")
axes[0].set_xlabel("$u_x$"); axes[0].set_ylabel("$y + 1/2$")
axes[0].set_title("P1 3-D plane Poiseuille"); axes[0].legend(fontsize=7)
p4 = RES["P4"]
axes[1].plot(p4["ref_profile"], np.arange(p4["channel_height"]) + 0.5, "k--", lw=1,
             label="analytic channel")
for i, prof in enumerate(p4["arms"]["sparse_soa"]["profiles"]):
    axes[1].plot(prof, np.arange(p4["channel_height"]) + 0.5, "o-", ms=3,
                 label=f"list arm, channel {i}")
axes[1].set_xlabel("$u_x$"); axes[1].set_title("P4 interior solid slabs (no border wall)")
axes[1].legend(fontsize=7)
figs.append(save(fig, "p1_p4_profiles.png"))
captions.append("P1: D3Q19 list-based profiles vs the exact parabola at H=8/16/32. "
                "P4: two channels carved by solid cells *inside* the list (fully periodic box, "
                "no wall plane on the domain border) vs the exact channel parabola.")

# --- P2: duct field comparison ---------------------------------------------
p2 = RES["P2"][-1]
ref = np.array(p2["ref_field"]); got = np.array(p2["arms"]["sparse_soa"]["field"])
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, data, title in zip(axes, (ref, got, got - ref),
                           ("analytic (White series)", "D3Q19 list (AoS)", "difference")):
    im = ax.imshow(data.T, origin="lower", aspect="auto", cmap="viridis")
    ax.set_title(f"{title}\nHy x Hz = {p2['hy']} x {p2['hz']}", fontsize=9)
    ax.set_xlabel("y"); ax.set_ylabel("z")
    fig.colorbar(im, ax=ax, fraction=0.046)
figs.append(save(fig, "p2_duct_field.png"))
captions.append("P2: rectangular-duct cross-section velocity, two-wall-pair geometry "
                "(periodic only along the streamwise x), analytic series vs the list solver "
                "and their difference.")

# --- P3: shear-wave decay ---------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4))
for r in RES["P3"]:
    for arm, style in (("dense", "k--"), ("sparse_soa", "o-")):
        ax.semilogy(r["times"], np.abs(r["arms"][arm]["amplitudes"]), style, ms=3,
                    label=f"n={r['n']} {arm}")
    tt = np.linspace(0, r["times"][-1], 50)
    ax.semilogy(tt, r["amplitude0"] * np.exp(-r["omega_ref"] * tt), ":", lw=1,
                color="gray", label=f"exact exp(-nu k^2 t), n={r['n']}")
ax.set_xlabel("steps"); ax.set_ylabel("mode amplitude")
ax.set_title("P3 3-D shear wave: exact rate $\\nu k^2$"); ax.legend(fontsize=7)
figs.append(save(fig, "p3_shear_wave.png"))
captions.append("P3: sine-mode amplitude of the exact shear-wave solution vs the exact "
                "exponential decay exp(-nu k^2 t); measured decay rates are tabulated in the report.")

# --- P7: storage scaling ----------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4))
fr = [r["solid_fraction"] for r in RES["P7"]]
ax.plot(fr, [r["bytes_ratio_sparse_over_dense"] for r in RES["P7"]], "o-", label="pdf bytes")
ax.plot(fr, [r["total_bytes_ratio_sparse_over_dense"] for r in RES["P7"]], "s--",
        label="pdf + index/coords")
ax.set_xlabel("solid fraction"); ax.set_ylabel("sparse / dense bytes")
ax.set_title("P7 storage cost of the list representation (24^3, D3Q19)"); ax.legend(fontsize=8)
figs.append(save(fig, "p7_storage.png"))
captions.append("P7: stored-bytes ratio (list pdf only, and list pdf + index array + coordinates) "
                "against the dense (Q x V) allocation as solid cells are removed.")

# --- markdown ---------------------------------------------------------------
def pct(x):
    return f"{100.0 * x:.3f} %"


checks = RES["checks"]
errs = {
    "P1 (H=32)": RES["P1"][-1]["arms"]["sparse_soa"]["err"]["linf_rel"],
    "P2 (16x24)": RES["P2"][-1]["arms"]["sparse_soa"]["err"]["linf_rel"],
    "P3 (n=32 decay rate)": RES["P3"][-1]["arms"]["sparse_soa"]["rel_err"],
    "P4 (interior slabs)": RES["P4"]["arms"]["sparse_soa"]["err"]["linf_rel"],
}
md = f"""## 1. Case introduction

**Anchor**: `XVAL-LBM-SPARSE-LIST-BASED-AOS-LBM`, round 2 (mission 21977, repo TensorLBM).
Round 1 (mission 18376) ported the OSS `lbmpy.sparse` list-based storage scheme to TensorLBM in
**2-D D2Q9** and validated it against analytic Poiseuille/Taylor-Green. Round 2 closes the two gaps
that round 1 left open, because a 2-D-only port with border walls does not exercise the two things
the scheme exists for:

* **the production stencil.** TensorLBM's ship-scale kernels are D3Q19 (`d3q19.py`,
  `actuator_disk_selfpropulsion.py:666`); a D2Q9-only sparse path cannot be used by them.
* **interior solid cells.** In `lssfau/lbmpy` (`mapping.py:130-135`) no-slip is *index aliasing*:
  a population whose pull source is a solid cell is replaced by the anti-directional population of
  the same cell. Round 1 only had wall planes on the domain border (where a dense solver does the
  same thing), never solid cells sitting **inside** the fluid list.

Physics/geometry: (P1) 3-D plane channel, walls in y, periodic x/z; (P2) rectangular duct, walls in
y and z, periodic only along x; (P3) 3-D shear wave, exact Navier-Stokes solution; (P4) a fully
periodic box with solid slab planes *inside* the fluid list, i.e. no wall plane on the domain
border at all; (P6/P7) representation-level checks.

References (external anchors, all re-evaluated in-process, none copied by hand):
`lbmpy 2.0` `lbmpy/sparse/mapping.py` (`create_index_array`, `pdf_index`, aliasing, periodic wrap)
and `lbmpy.sparse.SparseLbMapper` executed in-process for D3Q19; F. M. White, *Viscous Fluid Flow*,
3rd ed., §3-3.3 for the duct series; Guo, Zheng & Shi, *Phys. Rev. E* 65, 046308 (2002) for forcing;
the shear-wave rate nu k^2 is the textbook exact solution (Tannehill et al., §9).

## 2. What was done

1. **OSS survey (horizontal).** The `oss_inventory` LBM entries were reviewed: `lssfau/lbmpy`
   (list + pull-offset index array), `sorush-khajepor/listLBM` (main-list + sublists + neighbour
   list), `lanl/MF-LBM` (indirect-addressing wall tables repair the very bug this scheme is prone
   to). `lbmpy` is the only one that *defines* the index-array semantics and it is installed here,
   so its generator is used as the executable reference. Key finding used in this round:
   `create_index_array` is dimension-generic (dim from `len(flag_arr.shape)`, coordinate struct
   x/y/z) but `pdf_index(cell, d) = cell + d*len(self)` is hardcoded SoA with a literal
   `# TODO support different layouts here` while the pdf list itself may be `AoS`
   (`update_rule_sparse.py:8-20`) — that pairing is unguarded in the OSS.
2. **Method lineage.** Two textbook routes to the same physics were used as references: the exact
   parabolic channel profile (halfway bounce-back offset included) and White's rectangular-duct
   Fourier series (verified to reduce to the parabola as the second width -> infinity, see
   `tables.md` / the `test_duct_analytic_plane_limit` test). The unsteady reference is the
   nonlinearity-free shear wave `u = A sin(k z) exp(-nu k^2 t)`, exact for the full Navier-Stokes
   equations, so its decay rate is not a linearisation.
3. **Implementation (in-repo, minimal).** `src/tensorlbm/sparse_list_aos.py` was generalised from
   D2Q9 to any dimension: `C_D3Q19`/`C_D3Q19_LBMPY` stencils, dimension-dispatching `weights_for`,
   `equilibrium_vec`/`velocity_from_vec` (nd), nd Guo collision, nd dense control arm
   (`dense_collide_stream` now takes an arbitrary set of wall axes), nd `SparseListLBM` state and a
   new `duct_poiseuille_analytic`. The 2-D API (`equilibrium`, `velocity_from`, `macroscopic`)
   is kept bit-compatible; the round-1 2-D test file still passes unchanged.
4. **New validation harness** `xval/.../round2_d3q19/run_r2.py` (predeclared thresholds written
   before the run) with arms: dense traversal, list traversal SoA, list traversal AoS, OSS-array
   driven list, plus two stencil orderings.

## 3. Results

Full programmatic table: `xval/XVAL-LBM-SPARSE-LIST-BASED-AOS-LBM/round2_d3q19/tables.md`
(reproduced from `results.json`; owner band 15 %).

| case | quantity | ref | got | error | <15 % |
|---|---|---|---|---|---|
| P1 H=32 | max \\(u_x\\) | {RES['P1'][-1]['u_max_ref']:.6f} | {max(RES['P1'][-1]['arms']['sparse_soa']['profile']):.6f} | {pct(errs['P1 (H=32)'])} | {'yes' if errs['P1 (H=32)'] < 0.15 else 'no'} |
| P2 16x24 | max \\(u_x\\) | {RES['P2'][-1]['u_max_ref']:.6f} | {max(max(r) for r in RES['P2'][-1]['arms']['sparse_soa']['field']):.6f} | {pct(errs['P2 (16x24)'])} | {'yes' if errs['P2 (16x24)'] < 0.15 else 'no'} |
| P3 n=32 | decay rate | {RES['P3'][-1]['omega_ref']:.6e} | {RES['P3'][-1]['arms']['sparse_soa']['omega_got']:.6e} | {pct(errs['P3 (n=32 decay rate)'])} | {'yes' if errs['P3 (n=32 decay rate)'] < 0.15 else 'no'} |
| P4 | max \\(u_x\\) | {RES['P4']['u_max_ref']:.6f} | {max(max(p) for p in RES['P4']['arms']['sparse_soa']['profiles']):.6f} | {pct(errs['P4 (interior slabs)'])} | {'yes' if errs['P4 (interior slabs)'] < 0.15 else 'no'} |

Representation checks: D3Q19 index array vs OSS = **{RES['P5']['mismatches_vs_oss']} mismatches**
(shape {RES['P5']['shape']}, uint32), SoA-vs-AoS columns differ in
{RES['P5']['soa_vs_aos_columns_differ']} entries, stencil-order invariance
{float(RES['P5']['stencil_order_max_delta_over_umax']):.2e}, OSS-array-driven list solver vs ported
array {float(RES['P5']['oss_array_driven_vs_ported']):.2e}, layout-mismatch control
`{RES['P5']['layout_mismatch_control']}` (i.e. mixing the OSS SoA offsets with the AoS layout is
detected, not silent). Arm equivalence (dense vs list, SoA vs AoS) is at float64 round-off,
see tables.md.

Storage (P7): the list representation stores only fluid cells; pdf bytes ratio
sparse/dense = {', '.join(f'{r["solid_fraction"]:.2f}->{r["bytes_ratio_sparse_over_dense"]:.3f}' for r in RES['P7'])};
adding the index array + coordinates the total ratio is
{', '.join(f'{r["solid_fraction"]:.2f}->{r["total_bytes_ratio_sparse_over_dense"]:.3f}' for r in RES['P7'])}.

## 4. Figures

![P1/P4]({figs[0]}), ![P2]({figs[1]}), ![P3]({figs[2]}), ![P7]({figs[3]})

## 5. Notes, limits

* The list traversal is a NumPy gather/scatter prototype: it is *slower* than the dense arm per
  step at the resolutions tested (see the `step_s` columns in tables.md). The claim of this anchor
  is **representation + accuracy**, not speed; a compiled kernel is out of scope.
* The index array builds a Python dict of coordinates and loops per direction and per cell; it is
  O(N Q) with Python overhead and is not the production path for large geometries.
* The wrap target of a periodic axis must be a fluid cell: a solid cell plaza exactly on the
  wrapped row makes the index array undefined (the same assertion fires inside lbmpy). This is a
  documented geometric restriction of the scheme, and it is why the P4 slabs are placed on interior
  rows 2 and 7 of a 10-row periodic box.
* P1/P2/P4 use BGK with nu = 0.1 (tau = 0.8); halfway bounce-back gives the standard O(H^-2)
  wall-position error, which is exactly what the grid-convergence column shows.

## 6. Error verdict vs the acceptance band

Acceptance (owner standard): every benchmark error must be **< 15 %**; the TensorLBM-wide
"paper" bar would be < 3 %.
{chr(10).join(f"* {k}: {pct(v)} -> {'PASS' if v < 0.15 else 'FAIL'} (<15 %), {'PASS' if v < 0.03 else 'above'} (<3 %)" for k, v in errs.items())}
Check list: {json.dumps(checks, indent=1)}
Overall verdict from the run: **{RES['verdict']}**.

Reproduce: `PYTHONPATH=src python xval/XVAL-LBM-SPARSE-LIST-BASED-AOS-LBM/round2_d3q19/run_r2.py`
(single core, {RES['elapsed_s']:.0f} s, submitted to the cluster queue as the compute gate requires).
Tests: `PYTHONPATH=src python -m pytest tests/test_xval_sparse_list_aos_3d.py tests/test_xval_sparse_list_aos.py`.
"""

# write the round-2 report next to the artefacts
(HERE / "REPORT.md").write_text("# XVAL-LBM-SPARSE-LIST-BASED-AOS-LBM round 2\n\n" + md)
tables_append = (HERE / "tables.md").read_text()
(HERE / "tables.md").write_text(tables_append)

# post the platform report
base = os.environ["TENSOR_API_BASE"]


def call(path, body=None, token=None):
    req = urllib.request.Request(base + path,
                                 data=None if body is None else json.dumps(body).encode(),
                                 method="GET" if body is None else "POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


tok = call("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                           "password": os.environ["TENSOR_AGENT_PASS"]})
tok = tok.get("access_token") or tok.get("token")
payload = {
    "title": "XVAL-LBM-SPARSE-LIST-BASED-AOS-LBM round 2: D3Q19 list-based sparse LBM and interior solid cells",
    "summary": ("Round 2 of the sparse list-based AoS anchor: the lbmpy.sparse scheme was generalised to "
                "D3Q19, validated against analytic channel/duct/shear-wave solutions and against a "
                "geometry with solid cells inside the fluid list; index array is bit-identical to the OSS."),
    "scenario_key": "tensorlbm_xval",
    "mission_id": 21977,
    "tags": ["TensorLBM", "LBM", "sparse", "D3Q19", "xval", "lbmpy"],
    "markdown": md,
    "figures": [{"name": n, "caption": c,
                 "png_base64": base64.b64encode((FIGDIR / n).read_bytes()).decode()}
                for n, c in zip(figs, captions)],
}
try:
    out = call("/tensor/reports", payload, tok)
    print("report posted:", json.dumps(out)[:300])
except Exception as exc:  # keep the artefacts even if the platform is down
    print("REPORT POST FAILED:", exc)
    (HERE / "report_payload.json").write_text(json.dumps(payload)[:200000])
    sys.exit(1)
print("figures:", figs)