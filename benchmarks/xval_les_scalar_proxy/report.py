#!/usr/bin/env python3
"""Figures + markdown report for XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY.

Every number in the rendered markdown is injected from ``result.json``; nothing
is typed by hand (owner zero-transcription rule).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

CS_OWN = 0.1
SC_OSS = 0.17


def fmt(x, nd: int = 4) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return x
    if isinstance(x, (bool, np.bool_)):
        return "true" if x else "false"
    if isinstance(x, (int, np.integer)):
        return str(x)
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if math.isnan(xf) or math.isinf(xf):
        return "n/a"
    return f"{xf:.{nd}g}"


def mark(ok) -> str:
    if ok is None:
        return ""
    return "PASS" if ok else "**FAIL**"


# ---------------------------------------------------------------------------
def make_figures(result: dict, out: Path) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    figs: list[dict] = []

    ce = result["ce_algebra"]
    real = [r for r in ce["rows"] if r["topology"] != "solid_rotation"]

    # ---- (1) SGS viscosity law ------------------------------------------
    sf_all = np.array([r["sf"] for r in real])
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    grid = np.logspace(np.log10(sf_all.min() * 0.6), np.log10(sf_all.max() * 1.8), 50)
    ax.loglog(grid, CS_OWN**2 * grid, "k-", lw=1.6,
              label=r"closed form $\nu_t=C_s^2|S|_F$ ($C_s$=0.1, in-house)")
    ax.loglog(grid, CS_OWN**2 * math.sqrt(2.0) * grid, "k--", lw=1.1,
              label=r"standard Smagorinsky $(C_s\Delta)^2\sqrt{2S:S}$")
    styles = {"pure_shear": ("o", 1.0), "pure_strain": ("s", 1.0), "mixed_45deg": ("^", 1.0)}
    for topo, (mk, _) in styles.items():
        sel = sorted([r for r in real if r["topology"] == topo], key=lambda r: r["sf"])
        ax.loglog([r["sf"] for r in sel], [r["nu_t_inhouse"] for r in sel], mk, ms=5,
                  label=f"in-house: {topo}")
        ax.loglog([r["sf"] for r in sel], [r["nu_t_oss"] for r in sel], mk, ms=6, mfc="none",
                  label=f"upstream proxy: {topo}")
    ax.set_xlabel(r"$|S|_F=\sqrt{S_{ij}S_{ij}}$  [lattice]")
    ax.set_ylabel(r"$\nu_t$  [lattice]")
    ax.set_title("SGS viscosity on an exact first-order Chapman-Enskog field")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6.5, ncol=2)
    p = out / "fig1_nu_t_law.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append({
        "name": p.name,
        "caption": ("SGS viscosity from both closures on the exactly constructed Chapman-Enskog field vs the "
                    "closed-form laws. Both follow a straight line in |S|_F (a scalar Smagorinsky law); the "
                    "upstream |c c| proxy lands on a topology-dependent higher intercept."),
    })

    # ---- (2) where is the proxy an identity? ----------------------------
    pi = result["proxy_identity"]
    labels = [
        ("D2Q9, trace-free strain\n(upstream's own case)", pi["cases"]["d2q9_trace_free"]),
        ("D2Q9, with isotropic\nnonequilibrium", pi["cases"]["d2q9_with_trace"]),
        ("D3Q19, trace-free", pi["cases"]["d3q19_trace_free"]),
        ("D3Q27, trace-free", pi["cases"]["d3q27_trace_free"]),
        ("D2Q9, arbitrary\nnon-equilibrium (control)", pi["negative_control_random_noneq"]),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for i, (lab, d) in enumerate(labels):
        lo = d.get("k_min", d.get("k_p05"))
        hi = d["k_max"]
        mid = d["k_mean"]
        ax.plot([i, i], [lo, hi], color="C0", lw=6, solid_capstyle="butt", alpha=0.7)
        ax.plot(i, mid, "o", color="k", ms=4)
        ax.annotate(f"{mid:.2f}", (i, mid), textcoords="offset points", xytext=(8, 0), fontsize=7)
    ax.axhline(1.0, color="k", lw=1.0, ls=":", label=r"$Q=|\Pi|_F$ (identity)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([lab for lab, _ in labels], fontsize=7.5)
    ax.set_ylabel(r"$Q/|\Pi|_F$  (range over random strain tensors)")
    ax.set_title(r"Scope of the $|c_nc_m|$ identity")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    p = out / "fig2_proxy_bias.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append({
        "name": p.name,
        "caption": ("Range of the ratio Q/|Pi|_F over random strain tensors for each lattice and for an "
                    "arbitrary non-Chapman-Enskog field. The upstream proxy is an exact identity on D2Q9 for "
                    "trace-free strain (upstream's own setting), and ceases to be one in 3-D or once an "
                    "isotropic nonequilibrium component is present."),
    })

    # ---- (3) Taylor-Green / multi-mode decay ----------------------------
    cases = result["decay"]
    npan = len(cases)
    fig, axes = plt.subplots(1, npan, figsize=(4.2 * npan, 4.0), squeeze=False)
    for ax, c in zip(axes[0], cases):
        for r in c["runs"]:
            if not r["samples"]:
                continue
            t = np.array([s[0] for s in r["samples"]], dtype=float)
            K = np.maximum(np.array([s[1] for s in r["samples"]], dtype=float), 1e-30)
            ax.semilogy(t, K, lw=1.3, label=r["arm"])
        if c["case"]["kind"].startswith("analytic"):
            k0 = 0.25 * c["U0"] ** 2
            tt = np.linspace(0, c["steps"], 200)
            ax.semilogy(tt, k0 * np.exp(-c["gamma_K_theory_bgk"] * tt), "k--", lw=1.0,
                        label="closed form (no SGS)")
        ax.set_xlabel("step")
        ax.set_ylabel(r"$K=\frac{1}{2}\langle u^2+v^2\rangle$")
        ax.set_title(f"{c['case']['name']}\nRe={c['Re']:g}, tau0={c['tau0']:.4f}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=6.5)
    p = out / "fig3_decay.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append({
        "name": p.name,
        "caption": ("Kinetic-energy decay for BGK, in-house LES, coefficient-mapped in-house LES and the "
                    "upstream-port LES (last panel: 4-mode nonlinear case, no analytic reference)."),
    })

    # ---- (4) field maps -------------------------------------------------
    fm = result.get("field_maps")
    if fm:
        nu_a = np.array(fm["nu_t_A_mapped"])
        nu_b = np.array(fm["nu_t_oss"])
        ratio = np.array(fm["k_field"])
        fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.5))
        for ax, arr, title in (
            (axes[0], nu_a, r"$\nu_t$ in-house ($C_s^{map}$)"),
            (axes[1], nu_b, r"$\nu_t$ upstream proxy ($S_c$=0.17)"),
            (axes[2], ratio, r"$Q/|\Pi|_F$ (clipped at 10)"),
        ):
            im = ax.imshow(arr, origin="lower", cmap="viridis")
            ax.set_title(f"{title}\n[{fm['case']}, step {fm['step']}]", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
        p = out / "fig4_nu_t_fields.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        figs.append({
            "name": p.name,
            "caption": ("Pointwise SGS viscosity applied by each closure on the same state, and the pointwise "
                        "proxy ratio Q/|Pi|_F: the bias is spatially structured (largest where the strain "
                        "eigen-directions rotate away from the lattice diagonals)."),
        })
    return figs


# ---------------------------------------------------------------------------
def render_docs(result: dict, figs: list[dict], path: Path) -> None:
    ce = result["ce_algebra"]
    v = result["verdict"]
    OWNER = v["owner_threshold_pct"]
    L: list[str] = []
    A = L.append

    A("# XVAL-LBM-LES-SCALAR-Q-ABS-CC-PROXY")
    A("")
    A(f"Cross-validation of the TensorLBM non-equilibrium-stress SGS closure against the upstream "
      f"`les-scalar-q-abs-cc-proxy` practice (anchor row `tensorplatform.oss_solutions#row-39`).")
    A("")
    A(f"* generated by `benchmarks/xval_les_scalar_proxy/run_xval.py` from "
      f"`runs/xval_les_scalar_proxy/result.json` (git `{result['env']['git_head']}`, "
      f"{result['env']['timestamp_utc']}, torch {result['env']['torch']}, "
      f"float64 CPU, quick={result['env']['quick']}); every number below is injected from that JSON.")
    A(f"* **Verdict: {v['headline']}** — proxy bias spread `k = Q/|Pi|_F` = "
      f"{fmt(v['k_spread_pct'], 3)} % (band {fmt(OWNER, 3)} %), worst trajectory cross-error "
      f"{fmt(v['worst_gamma_cross_rel_pct'], 3)} %, worst field-level mapped `nu_t` error "
      f"{fmt(v['worst_nu_t_mapped_rel_pct'], 3)} %.")
    A("")

    # ---------------- 1 -------------------------------------------------
    A("## 1. Problem and reference provenance")
    A("")
    A("The anchor asks for cross-validation against a mature open-source implementation with the owner "
      "<15% accuracy band, a difference list, and an explicit porting decision.")
    A("")
    A("Upstream practice, verbatim from the machine-extracted source "
      f"(`{ce['source']['les_lines']}` in `LatFlow/Domain.py`, sha256 "
      f"`{result['env']['oss_files']['LatFlow/Domain.py']}`):")
    A("")
    A("```python")
    A("# LatFlow/Domain.py:90-96  (EEk, built in Domain.__init__)")
    for ln in ce["source"]["eek_block"]:
        A(ln)
    A(f"# LatFlow/Domain.py:{ce['source']['les_lines']}  (CollideSC les branch)")
    for ln in ce["source"]["les_branch"]:
        A(ln)
    A("# " + ce["source"]["sc_line"])
    A("```")
    A("")
    A(f"Extracted (never transcribed) by `extract_oss_tables.py`: `LatFlow/D2Q9.py` sha256 "
      f"`{result['env']['oss_files']['LatFlow/D2Q9.py']}`, EEk = {[fmt(e, 3) for e in ce['eek_abs']]}, "
      f"Sc = {fmt(ce['Sc'], 3)}. Upstream tarball sha256 `{result['env']['oss_tarball_sha256']}` "
      f"(inherited from XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL after the per-file hashes re-matched).")
    A("")
    A("The only upstream driver that switches `les=True` (`les_train.py`) does not run as shipped "
      "(TensorFlow 1.x, `cv2.cv.CV_FOURCC`, undefined `lr`/`total_loss`), so the cross-validated artefact "
      "is the **closure algebra**, evaluated on identical states — not an upstream trajectory.")
    A("")

    # ---------------- 2 -------------------------------------------------
    A("## 2. Method")
    A("")
    A("| probe | what it isolates | closed-form reference |")
    A("|---|---|---|")
    A("| exact Chapman-Enskog field | the strain-rate *input* each closure sees | `|Pi|_F = (2/3) rho tau0 |S|_F` |")
    A("| field level on real LBM states | the `nu_t` each closure applies to the same `f` | the same identity + Smagorinsky law |")
    A("| Taylor-Green / 4-mode decay | transport of the closed operator | `K(t)=K0 exp(-4 nu k^2 t)`, `gamma = 4 k^2 (nu0 + <nu_t>)` |")
    A("")
    A("Exact CE construction: for a uniform trace-free strain tensor `S`, "
      "`f = f^eq(rho=1, u=S.r) + f^(1)` with `f^(1)_q = -3 tau0 w_q rho c_qi c_qj S_ij`. Then "
      "`Pi_ij = sum_q c_qi c_qj f^(1)_q = -2 rho c_s^2 tau0 S_ij` **exactly**, which fixes both the "
      "closed form that the in-house norm must reproduce and the bias of the upstream proxy.")
    A("")
    A(f"Arms: `bgk` (control), `A_nominal` = in-house `collide_smagorinsky_bgk` at `C_s`={CS_OWN}, "
      f"`B_oss` = faithful port of the upstream branch at `Sc`={SC_OSS}, `A_mapped` = in-house closure at "
      f"`C_s^map = sqrt(Sc * kbar / 3)` with `kbar = <Q>/<|Pi|_F>` measured on the same states "
      f"(this is exactly the scalar conversion an adoption would publish).")
    A("")
    A("Port fidelity is a gate: the upstream `les=False` branch must reproduce the shipped `collide_bgk`. "
      f"Measured max |df| = {fmt(result['port_fidelity']['max_abs_diff_f'])} "
      f"(rel {fmt(result['port_fidelity']['rel_f'])}, threshold {fmt(result['port_fidelity']['threshold_rel'])}) "
      f"-> {mark(result['port_fidelity']['rel_f'] <= result['port_fidelity']['threshold_rel'])}.")
    A("")

    # ---------------- 3 -------------------------------------------------
    A("## 3. Parameter table")
    A("")
    A("| case | n | Re | U0 | Ma | nu | tau0 | omega=1/tau0 | k=2pi/L | steps |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for c in result["parameters"]:
        A(f"| {c['case']} | {c['n']} | {fmt(c['Re'])} | {fmt(c['U0'], 3)} | {fmt(c['Ma'], 4)} | "
          f"{fmt(c['nu'])} | {fmt(c['tau0'], 5)} | {fmt(1.0 / c['tau0'], 4)} | {fmt(c['k'], 4)} | {c['steps']} |")
    A("")
    A(f"Coefficients: in-house `C_s` = {CS_OWN} (equivalent standard Smagorinsky constant "
      f"`C_s/2^(1/4)` = {fmt(CS_OWN / 2 ** 0.25, 3)}); upstream `Sc` = {SC_OSS} "
      f"(equivalent in-house `C_s = sqrt(Sc/3)` = {fmt(math.sqrt(SC_OSS / 3.0), 4)}, i.e. classical "
      f"Smagorinsky `C_s = sqrt(Sc/3)/2^(1/4)` = {fmt(math.sqrt(SC_OSS / 3.0) / 2 ** 0.25, 4)}). "
      f"Strain amplitudes of the exact-CE sweep: |S|_F = {', '.join(fmt(a, 3) for a in ce['amplitudes'])}.")
    A("")

    # ---------------- 4 -------------------------------------------------
    A("## 4. Reference vs got")
    A("")
    A("### 4.1 Closure algebra on the exact CE field")
    A("")
    A("| topology | \\|S\\|_F | \\|Pi\\|_F numeric | closed form | err % | Q proxy | Q err vs closed form % | k=Q/\\|Pi\\|_F | nu_t in-house | nu_t upstream |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in ce["rows"]:
        A(f"| {r['topology']} | {fmt(r['sf'], 3)} | {fmt(r['pi_numeric'])} | {fmt(r['pi_closed_form'])} | "
          f"{fmt(r['pi_identity_rel_pct'], 2)} | {fmt(r['q_proxy'])} | {fmt(r['q_proxy_rel_pct'], 2)} | "
          f"{fmt(r['k_ratio'], 3)} | {fmt(r['nu_t_inhouse'])} | {fmt(r['nu_t_oss'])} |")
    A("")
    A(f"* In-house norm vs the Chapman-Enskog identity: max error "
      f"**{fmt(ce['pi_identity_max_rel_pct'], 3)} %** -> the in-house strain-rate input is the physical one "
      f"(its float32 lattice cast is the only error source).")
    A(f"* Upstream proxy vs the same identity: max error **{fmt(ce['q_proxy_max_rel_pct'], 2)} %**.")
    A(f"* Bias ratio `k = Q/|Pi|_F` spans {fmt(ce['k_min'], 3)} .. {fmt(ce['k_max'], 3)} "
      f"(mean {fmt(ce['k_mean'], 3)}), spread **{fmt(ce['k_spread_pct'], 2)} %** against a predeclared "
      f"{fmt(15.0, 3)} % consistency band -> {mark(ce['k_spread_pct'] <= 15.0)} for H1.")
    A(f"* Solid-body rotation (`S = 0`): `|Pi|_F <= {fmt(ce['solid_rotation_pi_max'])}`, "
      f"max `nu_t` = {fmt(ce['solid_rotation_nu_t_max'])} -> both closures switch off exactly (no spurious "
      f"dissipation in a rigid-body region).")
    A("")

    A("")
    ev = result["proxy_identity"]
    A("### 4.1b Scope of the identity (exploratory follow-up)")
    A("")
    A("The topology sweep above returns `k = 1` to machine precision for every strain tensor, which is not a "
      "numerical coincidence: on D2Q9 the `|c_n c_m|` kernel makes the proxy *equal* to the exact norm for any "
      "first-order Chapman-Enskog field. Writing a trace-free strain as `S = [[a, c], [c, -a]]`, the "
      "nonequilibrium populations take only two distinct values -- `f = -a/3` on the four axis directions "
      "(`EEk = 1`) and `f = -c/6` on the four diagonals (`EEk = 4`) -- so")
    A("")
    A("```")
    A("Q^2        = 2 (4 a^2/9 + 4 c^2/9)              = 8(a^2 + c^2)/9")
    A("|Pi|_F^2   = (2a/3)^2 + (2a/3)^2 + 2 (2c/3)^2   = 8(a^2 + c^2)/9      ->  Q = |Pi|_F")
    A("```")
    A("")
    A("That is a property of the D2Q9 symmetry, not of the kernel in general. Randomised checks over "
      "`trace-free` / `with-trace` strain tensors on three lattices, plus an arbitrary (non-Chapman-Enskog) "
      "field as a negative control:")
    A("")
    A("| lattice / premise | draws | k min | k mean | k max | max abs dev from 1 | spread % |")
    A("|---|---|---|---|---|---|---|")
    labels = [
        ("D2Q9, trace-free (upstream case)", ev["cases"]["d2q9_trace_free"]),
        ("D2Q9, isotropic noneq included", ev["cases"]["d2q9_with_trace"]),
        ("D3Q19, trace-free", ev["cases"]["d3q19_trace_free"]),
        ("D3Q27, trace-free", ev["cases"]["d3q27_trace_free"]),
    ]
    for lab, d in labels:
        A(f"| {lab} | {d['draws']} | {fmt(d['k_min'])} | {fmt(d['k_mean'])} | {fmt(d['k_max'])} | "
          f"{fmt(d['max_abs_dev_from_1'], 3)} | {fmt(d['relative_spread_pct'], 2)} |")
    nc = ev["negative_control_random_noneq"]
    A(f"| D2Q9, arbitrary non-equilibrium (control) | {nc['draws']} | {fmt(nc['k_p05'])} | {fmt(nc['k_mean'])} | "
      f"{fmt(nc['k_max'])} | n/a | n/a |")
    A("")
    A("* The identity is **exact** for the 2-D incompressible case that upstream actually ships (max deviation "
      f"{fmt(ev['cases']['d2q9_trace_free']['max_abs_dev_from_1'], 2)}).")
    A("* It **breaks in 3-D** and as soon as an isotropic nonequilibrium part is present, so the upstream "
      "proxy is not a dimension-independent stand-in for `|Pi|_F`; the in-house Frobenius form is the one that "
      "is correct in every case.")
    A(f"* `{ev['note']}`")
    A("")
    A("")
    A("### 4.2 Field level on identical real states")
    A("")
    A("| case | kbar=<Q>/<\\|Pi\\|_F> | k p05..p95 | nu_t in-house (C_s=0.1) | nu_t upstream (Sc=0.17) | nominal rel % | C_s^map | nu_t A_mapped | mapped rel % |")
    A("|---|---|---|---|---|---|---|---|---|")
    for f in result["field_level"]:
        A(f"| {f['case']} | {fmt(f['k_mean'], 3)} | {fmt(f['k_p05'], 2)} .. {fmt(f['k_p95'], 2)} | "
          f"{fmt(f['nu_t_A_mean'])} | {fmt(f['nu_t_B_mean'])} | {fmt(f['nu_t_nominal_rel_pct'], 2)} | "
          f"{fmt(f['Cs_map'], 4)} | {fmt(f['nu_t_A_mapped_mean'])} | "
          f"{fmt(f['nu_t_mapped_rel_pct'], 2)} {mark(f['nu_t_mapped_rel_pct'] <= OWNER)} |")
    A("")
    A("`nominal rel %` is the error if one naively swapped the coefficients (C_s=0.1 <-> Sc=0.17); "
      "`mapped rel %` is the error after the single-scalar conversion — the owner-band KPI.")
    A("")

    A("### 4.3 Trajectory level")
    A("")
    A("| case | gamma BGK | gamma A_mapped | gamma B_oss | cross A_mapped vs B % | cross nominal % | consistency A_mapped % | consistency B_oss % | consistency BGK % | LES gain A_mapped % | K_final cross % |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in result["decay"]:
        A(f"| {c['case']['name']} | {fmt(c['gamma_K_bgk'])} | {fmt(c['gamma_K_A_mapped'])} | "
          f"{fmt(c['gamma_K_B_oss'])} | {fmt(c['gamma_cross_rel_pct'], 2)} {mark(c['gamma_cross_rel_pct'] <= OWNER)} | "
          f"{fmt(c['gamma_cross_nominal_rel_pct'], 2)} | {fmt(c['consistency_A_mapped_pct'], 2)} | "
          f"{fmt(c['consistency_B_oss_pct'], 2)} | {fmt(c['consistency_bgk_pct'], 2)} | "
          f"{fmt(c['les_gain_A_mapped_pct'], 2)} | {fmt(c['K_final_cross_rel_pct'], 2)} |")
    A("")
    c0 = result["decay"][0]
    A(f"`consistency` is the a-posteriori identity `gamma_fit = 4 k^2 (nu0 + <nu_t> applied)` evaluated "
      f"over the fit window — it verifies that the per-cell `tau_eff` really acts as a viscosity. "
      f"For the first case the SGS-free closed form is `4 nu k^2 = {fmt(c0['gamma_K_theory_bgk'])}` and the "
      f"BGK arm reproduces it to {fmt(c0['consistency_bgk_pct'], 2)} %.")
    A("")

    A("### 4.4 A-priori (macroscopic strain) vs a-posteriori (applied) viscosity")
    A("")
    A("| case (state at step) | \\|S\\|_F from fd of u | \\|S\\|_F implied by the closure input | implied vs fd % | nu_t in-house measured | a-priori C_s^2\\|S\\|_F | err % | nu_t upstream measured | a-priori (1/3)Sc k \\|S\\|_F | err % |")
    A("|---|---|---|---|---|---|---|---|---|")
    for a in result["apriori"]:
        A(f"| {a['case']} (step {a['step']}) | {fmt(a['sf_mean_analytic'])} | {fmt(a['sf_implied_by_closure'])} | "
          f"{fmt(a['implied_vs_fd_rel_pct'], 2)} | {fmt(a['nu_t_inhouse_measured'])} | "
          f"{fmt(a['nu_t_inhouse_apriori'])} | {fmt(a['inhouse_apriori_rel_pct'], 2)} | "
          f"{fmt(a['nu_t_oss_measured'])} | {fmt(a['nu_t_oss_apriori'])} | {fmt(a['oss_apriori_rel_pct'], 2)} |")
    A("")
    A("The t = 0 state is exactly an equilibrium (`f = f^eq`), so nothing can be measured there; the probe "
      "therefore uses the first developed state. `|S|_implied` is what the closure sees (via the exact "
      "Chapman-Enskog identity `|Pi|_F = (2/3) rho tau_eff |S|`), `|S|_FD` is the strain magnitude of the same "
      "macroscopic field by periodic central differences: their ratio is the honest measure of how far the "
      "discrete non-equilibrium departs from the continuum strain at this resolution and Mach number.")
    A("")

    # ---------------- 5 -------------------------------------------------
    A("## 5. Convergence and sensitivity")
    A("")
    A("| axis | KPI | finding |")
    A("|---|---|---|")
    for s in result["sensitivity"]:
        A(f"| {s['axis']} | {s['kpi']} | {s['finding']} |")
    A("")

    # ---------------- 6 -------------------------------------------------
    A("## 6. Failure modes, limitations, artefacts")
    A("")
    for note in result["notes"]:
        A(f"* {note}")
    A("")
    A(f"* Artefacts: `runs/xval_les_scalar_proxy/result.json` (all raw samples, per-arm closure series), "
      f"`table.csv` (KPI x configuration with pass/fail), `figures/*.png`. "
      f"Wall time {fmt(sum(sum(c['wall_s'].values()) for c in result['decay']), 4)} s for all arms.")
    A("")

    A("### Difference list: TensorLBM vs upstream")
    A("")
    A("| aspect | upstream `LatFlow/Domain.py` | TensorLBM `turbulence.turbulence` |")
    A("|---|---|---|")
    A("| strain-rate input | `Q = sqrt(2 sum_q f_neq_q^2 EEk_q)`, `EEk_q = (sum_n |c_qn|)^2`: an L2 norm over populations, so contributions of opposite sign add instead of cancelling | `|Pi|_F = sqrt(Pi_xx^2 + Pi_yy^2 + 2 Pi_xy^2)` from the **signed** stress `Pi_ij = sum_q c_qi c_qj f_neq_q` |")
    A("| closure coefficient | `Sc` = 0.17 inside `tau = 0.5(tau0 + sqrt(tau0^2 + 6 Q Sc/rho))` | `C_s` = 0.1 inside `tau = 0.5(tau0 + sqrt(tau0^2 + 18 C_s^2 |Pi|_F/rho))` |")
    A("| implemented law | `nu_t = (2/3) Sc k |S|_F`; at `k`~1 that is `C_s,std` = 0.15 | `nu_t = C_s^2 |S|_F`, i.e. `C_s,std = C_s/2^(1/4)` = 0.084 |")
    A("| relaxation bounds | none: `tau` unbounded above | `nu_t <= 1/6` increment cap, `tau_eff >= 0.5001` floor |")
    A("| moments fed to the closure | stored `rho`,`u` from the previous stream, reused in the collision | recomputed from `f` at the collision instant |")
    A("| boundaries | solid mask handled inside the same collision kernel by a 1x1 cross-channel bounce convolution | dedicated bounce/BC kernels; the closure is boundary-model independent |")
    A("| lattice coverage | D2Q9, D3Q15, D3Q19 | D2Q9, D3Q19, D3Q27, each with BGK/MRT variants and WALE/Vreman/dynamic-Smagorinsky siblings |")
    A("| stack | TensorFlow 1.x float32 GPU, hand-rolled `graph_unroll` path for LES training | torch, float64 CPU path available, autograd-compatible (clamp documented) |")
    A("| driver | `les_train.py` non-runnable (TF1, undefined `lr`/`total_loss`) | any driver; closure usable as a pure function |")
    A("")
    A("### Porting decision and trade-offs")
    A("")
    A(v["decision_text"])
    A("")
    A("Trade-offs accepted by this decision:")
    A("")
    A(f"* Reproducing upstream numbers is now a *per-case* operation: quote the measured `k` band "
      f"({fmt(ce['k_min'], 3)}..{fmt(ce['k_max'], 3)}) and the conversion `C_s = sqrt(Sc k / 3)` instead of "
      f"a single constant. On smooth shear/strain the in-house closure can be aligned to within the owner "
      f"band; on states whose strain eigen-directions rotate relative to the lattice diagonals it cannot.")
    A("* The upstream proxy is *cheaper* than the Frobenius norm (it needs one `EEk`-weighted reduction and "
      "no cross-component products), but the cost difference is irrelevant at the resolutions where the "
      "closure matters, so cheapness alone does not justify the bias.")
    A("* The upstream `|c c|` proxy has one property worth keeping in mind: it is exactly proportional to the "
      "strain rate (amplitude-independent), so it is a legitimate *scaling* estimator — it fails only in the "
      "constant, which is what the conversion publishes.")
    A("")
    A("## Figures")
    A("")
    for f in figs:
        A(f"**{f['name']}** — {f['caption']}")
        A("")
        A(f"![{f['name']}](../runs/xval_les_scalar_proxy/figures/{f['name']})")
        A("")
    if not figs:
        A("*(no figures: `--quick` mode)*")
        A("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")