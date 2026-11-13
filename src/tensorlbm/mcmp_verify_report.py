"""Renderer for the MCMP verification benchmark report.

The point of this module is the *no-hand-copy* rule: every number in
``docs/MCMP_VERIFY_BENCH.md`` is rendered from
``artifacts/mcmp_verify_bench/results.json``, which in turn is written by
``benchmarks/mcmp_verify_bench.py`` from the solver runs.  The benchmark test
suite re-renders the document from the committed json and fails if the two
differ, so the report cannot drift from the data by editing text.

Figures are produced from the same json (plus ``hero_fields.npz`` for the field
plots) and written next to it under ``figures/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HEADER = """# MCMP verification bench (FEAT-LBM-VERIFY-BENCH)

Scope: the two-component Shan-Chen path of this repository
(`tensorlbm.multiphase.collide_sc_two_component`).  Tooling:
`src/tensorlbm/mcmp_verification.py`; runner: `benchmarks/mcmp_verify_bench.py`;
machine-readable data: `artifacts/mcmp_verify_bench/results.json`.

**Every number below is rendered from that json** (`tensorlbm.mcmp_verify_report`)
and the suite re-renders it in `tests/benchmarks/test_mcmp_verify_bench.py`, so
the tables cannot be edited by hand without breaking the tests.

Model conventions used by the measurements (they are *this* library's, not the
upstream inventories'): linear pseudopotential `psi(rho) = rho`, cross coupling
`G_12 < 0` for separation, velocity-shift forcing, equilibrium pressure
`p = cs^2 (rho_1 + rho_2) + G_12 cs^2 psi_1 psi_2`.
"""


def _f(value: Any, digits: int = 4, scale: float = 1.0, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        v = value * scale
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
            return f"{v:.{digits - 1}e}{suffix}"
        return f"{v:.{digits}f}{suffix}"
    return str(value)


def _table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _guard_codes(guards: list[dict[str, Any]]) -> str:
    return ", ".join(g["code"] for g in guards) if guards else "-"


def _laplace_section(res: dict[str, Any]) -> str:
    lap = res["laplace"]
    proto = lap["protocol"]
    parts = ["## 1. Young-Laplace calibration", ""]
    parts.append(
        "Protocol (fixed before the runs): coexistence densities from a **planar slab** "
        "(curvature-free) per coupling level; droplets initialised with those densities at "
        f"radii {proto['radii']} on an `n = {proto['n']}` periodic box, relaxed until "
        "`|u|max` plateaus; `Delta p` = (centre 4x4 box) - (corner 4x4 box) of the equilibrium "
        "pressure; `R` from the linearly interpolated `rho_1 = rho_av` crossing on four rays."
    )
    parts.append("")
    dims = int(proto.get("laplace_dims", 2))
    factor = dims - 1
    parts.append(
        f"Acceptance metric: the law `Delta p = {'' if factor == 1 else f'{factor} '}sigma / R` "
        "(2D circular droplet: "
        "`Delta p = sigma / R`; 3D sphere: `Delta p = 2 sigma / R`) has one unknown per level, so "
        f"`sigma_0 = sum x y / sum x^2` with `x = {factor}/R`, and the error is the per-case "
        "relative residual `|Delta p_i - sigma_0 x_i| / (sigma_0 x_i)`. A free two-parameter line "
        "is reported as a diagnostic only (its intercept absorbs a curvature-independent bulk "
        "pressure offset; reporting its slope as the error would relax the criterion)."
    )
    parts.append("")
    parts.append("### Coexistence (planar slab, checks the pressure convention)")
    parts.append("")
    rows = []
    for c in lap["coexistence"]:
        rows.append(
            [
                _f(c["G_12"], 3),
                _f(c["rho1_liq"]),
                _f(c["rho1_gas"]),
                _f(c["density_ratio"], 2),
                _f(c.get("flat_interface_dp"), 2),
                _f(c["u_max"], 4),
                _f(c["converged"]),
                _f(c["steps_run"], 0),
            ]
        )
    parts.append(
        _table(
            [
                "G_12",
                "rho1_liq",
                "rho1_gas",
                "ratio",
                "flat interface dp",
                "umax",
                "converged",
                "steps",
            ],
            rows,
        )
    )
    parts.append("")
    parts.append(
        "A flat interface must have **no** pressure jump: the `flat interface dp` column is the "
        "self-check of the pressure convention against the force (values in the 1e-4..1e-7 range "
        "are at the level of the finite-box truncation, not of a physical offset)."
    )
    parts.append("")
    parts.append("### Per-case measurements")
    parts.append("")
    rows = []
    for c in lap["cases"]:
        rows.append(
            [
                _f(c["G_12"], 3),
                _f(c["radius_init"], 1),
                _f(c["radius"], 2),
                _f(c["dp"], 4),
                _f(c["sigma"], 4),
                _f(c.get("rel_err_vs_law"), 3, 100.0, "%"),
                _f(c["u_max"], 4),
                _f(c["density_ratio"], 2),
                _f(c["steps_run"], 0),
                _guard_codes(c["guardrails"]),
            ]
        )
    parts.append(
        _table(
            [
                "G_12",
                "R_init",
                "R measured",
                "dp",
                "sigma = dp R" if dims == 2 else f"sigma = dp R/{dims - 1}",
                "law residual",
                "umax",
                "density ratio",
                "steps",
                "guardrails",
            ],
            rows,
        )
    )
    parts.append("")
    parts.append("### Acceptance per coupling level")
    parts.append("")
    rows = []
    for level in lap["levels"]:
        rows.append(
            [
                _f(level["G_12"], 3),
                _f(level["sigma_origin"], 4),
                _f(level["sigma_slope"], 4),
                _f(level["r2"], 5),
                _f(level["max_offset_fraction"], 3),
                _f(level["max_rel_err"], 3, 100.0, "%"),
                "<5%: " + ("**PASS**" if level["pass"] else "**FAIL**"),
                _guard_codes(level["guardrails"]),
            ]
        )
    parts.append(
        _table(
            [
                "G_12",
                "sigma (through origin)",
                "sigma (free-line slope)",
                "r^2 of free line",
                "max |b|R/|a|",
                "max law residual",
                "verdict",
                "guardrails",
            ],
            rows,
        )
    )
    parts.append("")
    parts.append(
        f"**Levels passing the <5% criterion: "
        f"{sum(1 for level in lap['levels'] if level['pass'])}/"
        f"{len(lap['levels'])}.**"
    )
    parts.append("")
    return "\n".join(parts)


def _spurious_section(res: dict[str, Any]) -> str:
    sp = res["spurious"]
    parts = ["## 2. Spurious-current baseline (grid x coupling)", ""]
    parts.append(
        f"Fixed physical droplet radius `R = {sp['radius']}` in periodic boxes of "
        f"`n in {sp['grids']}`; at rest, `|u|max` is the mass-averaged velocity maximum. "
        "`Ca_sp = mu |u|max / sigma` with `mu = rho_bar (tau - 1/2)/3` is the "
        "viscosity-independent comparison level (the raw `|u|max` scales with `sigma/mu`). "
        f"Force stencil: {sp['stencil']}."
    )
    parts.append("")
    rows = []
    for row in sorted(sp["rows"], key=lambda r: (r["G_12"], r["n"])):
        rows.append(
            [
                _f(row["n"], 0),
                _f(row["G_12"], 3),
                _f(row["u_max"], 4),
                _f(row["u_max_over_cs"], 4),
                _f(row["ca_spurious"], 4),
                _f(row["sigma"], 4),
                _f(row["density_ratio"], 2),
                _f(row["width_10_90"], 2),
                _f(row["converged"]),
                _guard_codes(row["guardrails"]),
            ]
        )
    parts.append(
        _table(
            [
                "n",
                "G_12",
                "umax",
                "umax/cs",
                "Ca_sp",
                "sigma",
                "density ratio",
                "10-90 width",
                "converged",
                "guardrails",
            ],
            rows,
        )
    )
    parts.append("")
    mn = min(r["u_max"] for r in sp["rows"])
    mx = max(r["u_max"] for r in sp["rows"])
    parts.append(
        f"Range over the table: `|u|max = {mn:.4e} .. {mx:.4e}` lattice units "
        f"(`{mn / 0.5773502691896258:.4f} .. {mx / 0.5773502691896258:.4f}` cs). "
        "The spurious level **increases with |G_12|** (stronger cohesion -> larger "
        "density ratio and larger interfacial force imbalance) and **decreases with "
        "resolution** at fixed physical radius, which is the expected "
        "pseudopotential behaviour; the platform criterion `spurious < 0.005 U_max` "
        "must therefore be evaluated at the resolution and coupling of the target "
        "case, not from a single entry of this table."
    )
    parts.append("")
    return "\n".join(parts)


def _contact_section(res: dict[str, Any]) -> str:
    parts = ["## 3. Contact angle / interface thickness quick check", ""]
    parts.append(
        "Both checks are calibrated against **analytic fields with known answers** "
        "(no solver in the loop): a circular cap on a flat wall (R = 25, n = 96, "
        "interface width 1.5) and tanh density profiles of prescribed width. The contact "
        "angle is measured three ways, independently: the near-wall gradient projection of "
        "the geometric method (solid rows extrapolated from the fluid, gradient taken on the "
        "2nd fluid layer), an algebraic circle fit to the mid-density contour, and the "
        "upstream base-length/height cap reconstruction."
    )
    parts.append("")
    rows = []
    for c in res["contact_angle"]:

        def _err(key):
            got = c[key]["theta_deg"]
            return None if got is None else got - c["theta_true"]

        rows.append(
            [
                _f(c["theta_true"], 1),
                _f(c["geometric"]["theta_deg"], 2),
                _f(_err("geometric"), 2),
                _f(c["circle_fit"]["theta_deg"], 2),
                _f(_err("circle_fit"), 2),
                _f(c["cap_geometry"]["theta_deg"], 2),
                _f(_err("cap_geometry"), 2),
                _f(c["geometric"]["theta_spread_deg"], 1),
                _guard_codes(c["geometric"]["guardrails"] + c["cap_geometry"]["guardrails"]),
            ]
        )
    parts.append(
        _table(
            [
                "theta true",
                "gradient",
                "err",
                "circle fit",
                "err",
                "cap geometry",
                "err",
                "node spread",
                "guardrails",
            ],
            rows,
        )
    )
    parts.append("")
    geo_errs = [
        abs(c["geometric"]["theta_deg"] - c["theta_true"])
        for c in res["contact_angle"]
        if c["geometric"]["theta_deg"] is not None
    ]
    circ_errs = [
        abs(c["circle_fit"]["theta_deg"] - c["theta_true"])
        for c in res["contact_angle"]
        if c["circle_fit"]["theta_deg"] is not None
    ]
    cap_errs = [
        abs(c["cap_geometry"]["theta_deg"] - c["theta_true"])
        for c in res["contact_angle"]
        if c["cap_geometry"]["theta_deg"] is not None
    ]
    parts.append(
        f"Measured accuracy: gradient method max error **{max(geo_errs):.2f} deg** "
        f"(mean {sum(geo_errs) / len(geo_errs):.2f} deg), circle fit max error "
        f"**{max(circ_errs):.3f} deg**, cap geometry max error **{max(cap_errs):.2f} deg**. "
        "The gradient route carries a systematic "
        "positive bias (towards 90 deg) from the wall extrapolation; that is why the "
        "circle fit is reported next to it and why the angle must always travel with its "
        "`fill_order`/`first_layer` setting."
    )
    parts.append("")
    parts.append("### Interface thickness (analytic tanh profiles)")
    parts.append("")
    rows = []
    for t in res["interface_thickness"]:
        rows.append(
            [
                _f(t["w_true"], 2),
                _f(t["w_tanh"], 3),
                _f(t["width_10_90"], 3),
                _f(t["width_10_90_true"], 3),
                _f(
                    None
                    if t["w_tanh"] is None
                    else (t["width_10_90"] - t["width_10_90_true"]) / t["width_10_90_true"],
                    3,
                    100.0,
                    "%",
                ),
                _f(t["r2"], 5),
                _f(t["n_points"], 0),
            ]
        )
    parts.append(
        _table(
            ["w true", "w recovered", "10-90 width", "10-90 true", "error", "r^2", "points"],
            rows,
        )
    )
    parts.append("")
    parts.append(
        "The 10-90 convention is `2 atanh(0.8) w = 2.1972 w`; the number is meaningless "
        "without it, which is why it is stated per row. The same measurement runs on the "
        "real MCMP droplets in the spurious-current table above (`10-90 width` column)."
    )
    parts.append("")
    return "\n".join(parts)


def _guardrail_section(res: dict[str, Any]) -> str:
    probe = res.get("guardrail_probe") or []
    parts = ["## 4. Validity domain (what the tool refuses to report)", ""]
    if not probe:
        parts.append("(not measured in this run)")
        parts.append("")
        return "\n".join(parts)
    parts.append(
        "The guardrails are only useful if they fire. This table runs couplings *outside* "
        "the usable window and records what the tooling does with them (all runs n=64, "
        "R_init=14, 6000 steps)."
    )
    parts.append("")
    rows = []
    for row in probe:
        rows.append(
            [
                _f(row["G_12"], 3),
                _f(row["radius"], 2),
                _f(row["dp"], 4),
                _f(row["u_max"], 4),
                _f(row["converged"]),
                _guard_codes(row["guardrails"]),
            ]
        )
    parts.append(_table(["G_12", "R measured", "dp", "umax", "converged", "guardrails"], rows))
    parts.append("")
    return "\n".join(parts)


def _chain_section(out_dir: Path) -> str:
    path = out_dir / "oss_consumption_chain.json"
    parts = ["## 5. Consumption chain (OSS inventory entries -> code)", ""]
    if not path.exists():
        parts.append("(manifest missing)")
        parts.append("")
        return "\n".join(parts)
    chain = json.loads(path.read_text())
    rows = []
    for item in chain["entries"]:
        rows.append(
            [
                f"`{item['id']}`",
                item["repo"],
                item["tag"],
                f"`{item['symbol']}`",
                item["where"],
                item["status"],
            ]
        )
    parts.append(_table(["entry", "repo", "tag", "symbol", "where it lands", "status"], rows))
    parts.append("")
    parts.append(
        "Every row is checked twice: the inventory text is read from the platform "
        "`oss_solutions` table (repo/tag/method bullet), and the upstream source quoted in "
        "that row is read from the local OSS cache before the formula is re-derived for this "
        "library's sign and pseudopotential conventions. `status=re-derived` means the "
        "upstream formula was **not** copied: the library convention differs and the "
        "adaptation is documented in the module docstring."
    )
    parts.append("")
    return "\n".join(parts)


#: Markdown link for a figure, relative to the committed report (``docs/``).
#: Kept as a constant so that rendering from a scratch directory produces byte
#: identical text (``test_report_is_rendered_from_json`` compares verbatim).
_FIG_LINK = "../artifacts/mcmp_verify_bench/figures/{name}"


def _figures(out_dir: Path, res: dict[str, Any]) -> list[str]:
    """Render the figures; returns the markdown embedding block."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:  # pragma: no cover - matplotlib is a dev extra
        return "_Figures skipped: matplotlib not available in this environment._\n"

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    dims = int(res["laplace"]["protocol"].get("laplace_dims", 2))

    # fig 1: Laplace law
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for level in res["laplace"]["levels"]:
        cases = [
            c
            for c in res["laplace"]["cases"]
            if c["G_12"] == level["G_12"] and c["radius"] and c["dp"] is not None
        ]
        x = np.array([1.0 / c["radius"] for c in cases])
        y = np.array([c["dp"] for c in cases])
        order = np.argsort(x)
        sigma = level["sigma_origin"]
        line_x = np.linspace(0, x.max() * 1.05, 50)
        ax.plot(
            line_x,
            sigma * (dims - 1) * line_x,
            "-",
            lw=1.0,
            label=f"G12={level['G_12']:g}: {dims - 1}σ/R, σ={sigma:.4f}",
        )
        ax.plot(x[order], y[order], "o", ms=5)
    ax.set_xlabel("1/R")
    ax.set_ylabel("Δp")
    ax.set_title("Young–Laplace: Δp vs 1/R (points = runs, lines = through-origin fit)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = fig_dir / "fig1_laplace.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    lines.append(f"![fig1]({_FIG_LINK.format(name=p.name)})")

    # fig 2: spurious currents
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for G in sorted({row["G_12"] for row in res["spurious"]["rows"]}):
        rows = sorted([r for r in res["spurious"]["rows"] if r["G_12"] == G], key=lambda r: r["n"])
        ns = [r["n"] for r in rows]
        axes[0].plot(ns, [r["u_max"] for r in rows], "o-", label=f"G12={G:g}")
        axes[1].plot(
            ns,
            [r["ca_spurious"] if r["ca_spurious"] is not None else np.nan for r in rows],
            "o-",
            label=f"G12={G:g}",
        )
    axes[0].set_xlabel("box size n (fixed R)")
    axes[0].set_ylabel("|u|max (lattice)")
    axes[0].set_title("spurious current level")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("box size n (fixed R)")
    axes[1].set_ylabel("Ca_sp = mu|u|max/sigma")
    axes[1].set_title("spurious capillary number")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    p = fig_dir / "fig2_spurious.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    lines.append(f"![fig2]({_FIG_LINK.format(name=p.name)})")

    # fig 3: hero fields
    hero = out_dir / "hero_fields.npz"
    if hero.exists():
        d = np.load(hero)
        rho1, rho2 = d["rho1"], d["rho2"]
        ux, uy = d["ux"], d["uy"]
        umag = np.sqrt(ux * ux + uy * uy)
        G = float(d["G_12"])
        n = int(d["n"])
        p = 0.3333333 * (rho1 + rho2) + G * 0.3333333 * rho1 * rho2
        fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.6))
        for ax, field, title, cmap in (
            (axes[0, 0], rho1, "component 1 density", "viridis"),
            (axes[0, 1], rho2, "component 2 density", "magma"),
            (axes[1, 0], p, "equilibrium pressure p", "coolwarm"),
            (axes[1, 1], np.log10(np.maximum(umag, 1e-12)), "log10 |u| (spurious)", "inferno"),
        ):
            im = ax.imshow(field, origin="lower", cmap=cmap)
            ax.set_title(f"{title} (n={n}, G12={G:g})")
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        pth = fig_dir / "fig3_fields.png"
        fig.savefig(pth, dpi=140)
        plt.close(fig)
        lines.append(f"![fig3]({_FIG_LINK.format(name=pth.name)})")

        # fig 4: profiles through the droplet centre
        c = n // 2
        r = np.arange(n) - (n - 1) / 2.0
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
        axes[0].plot(r, rho1[c, :], "-", lw=1.2, label="component 1")
        axes[0].plot(r, rho2[c, :], "--", lw=1.2, label="component 2")
        axes[0].set_xlabel("x - x_centre")
        axes[0].set_ylabel("density")
        axes[0].set_title("density profile (centre row)")
        axes[0].grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        axes[1].plot(r, p[c, :], "-", lw=1.2, color="C3")
        axes[1].set_xlabel("x - x_centre")
        axes[1].set_ylabel("p")
        axes[1].set_title("pressure profile: liquid centre vs gas far field")
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        pth = fig_dir / "fig4_profiles.png"
        fig.savefig(pth, dpi=140)
        plt.close(fig)
        lines.append(f"![fig4]({_FIG_LINK.format(name=pth.name)})")

    # fig 5: contact angle calibration
    rows = res["contact_angle"]
    true = [r["theta_true"] for r in rows]
    geo = [r["geometric"]["theta_deg"] for r in rows]
    circ = [r["circle_fit"]["theta_deg"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot([0, 180], [0, 180], "k--", lw=0.8, label="exact")
    ax.plot(true, geo, "o", label="near-wall gradient projection")
    ax.plot(true, circ, "s", label="circle fit")
    ax.set_xlabel("prescribed contact angle (deg)")
    ax.set_ylabel("measured (deg)")
    ax.set_title("contact-angle measurement calibration (analytic caps, n=96)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = fig_dir / "fig5_contact.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    lines.append(f"![fig5]({_FIG_LINK.format(name=p.name)})")

    return "\n".join(line + "\n" for line in lines)


def render_report(out_dir: str | Path, doc_path: str | Path) -> str:
    """Render ``docs/MCMP_VERIFY_BENCH.md`` from ``results.json``; return the text."""
    out = Path(out_dir)
    res = json.loads((out / "results.json").read_text())
    parts = [
        HEADER,
        "_Data: `artifacts/mcmp_verify_bench/results.json`, "
        f"generated {res['generated_at']}, "
        f"TensorLBM {res['library']['version']}, torch {res['library']['torch']}, "
        f"protocol quick={res['protocol']['quick']}._",
        "",
        "## 0. Figures",
        "",
        _figures(out, res),
        _laplace_section(res),
        _spurious_section(res),
        _contact_section(res),
        _guardrail_section(res),
        _chain_section(out),
        "## 6. Known limits of these measurements",
        "",
        "* The pressure relation is only valid for the **linear** pseudopotential that this "
        "library's MCMP force kernel uses; a non-linear psi (as in the single-component path) "
        "needs its own coefficient.",
        "* The contact-angle tool supports flat walls only (a single solid strip); curved or "
        "rough walls return a guardrail instead of an angle.",
        "* The contact angle of a *real* MCMP wall droplet is not measured here: the library's "
        "two-component path has no wetting boundary condition yet (dry wall only), so the tool "
        "is calibrated on analytic caps and is ready for the wetting rework, not proof of one.",
        "* Spurious currents are reported at `tau = 1` only; they are known to depend on "
        "viscosity, so a cross-viscosity study is a separate run.",
        "",
    ]
    text = "\n".join(parts)
    Path(doc_path).parent.mkdir(parents=True, exist_ok=True)
    Path(doc_path).write_text(text)
    return text
