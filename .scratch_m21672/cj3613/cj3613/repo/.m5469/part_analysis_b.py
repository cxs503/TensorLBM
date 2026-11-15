# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.grid": True,
                         "grid.alpha": 0.3, "figure.dpi": 90})
    return plt


def figures(dbs: dict, emp: dict, gcis: dict, annual: dict, cube: dict,
            loo: dict) -> list[str]:
    plt = _mpl()
    os.makedirs(FIG, exist_ok=True)
    paths = []
    ships = list(dbs)

    # fig 1: C_X and C_Y polars --------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.0), sharex=True)
    for j, ship in enumerate(ships):
        db = dbs[ship]
        ang = [d["angle_deg"] for d in db]
        a_full = np.arange(0, 361, 5)
        for i, key in enumerate(("C_X", "C_Y")):
            ax = axes[i, j]
            vals = [d[key] for d in db]
            errs = [d.get(key + "_std", 0.0) for d in db]
            ax.errorbar(ang, vals, yerr=errs, fmt="o-", ms=3, lw=1.0,
                        capsize=2, label="LBM (dx=3.0 m)")
            b = emp[ship]["rows"]
            ax.plot([r["angle_deg"] for r in b],
                    [r[f"{key}_buildup"] for r in b], "s--", ms=2.5, lw=0.9,
                    label="component build-up (a priori)")
            fit = emp[ship]["fit"]["params"]
            ax.plot(a_full, [blendermann_coefficients(fit, a)[key] for a_full],
                    ":", lw=1.0, label="Blendermann-style fit")
            ax.set_title(f"{ship}  $C_{{{key[-1]}}}$", fontsize=7)
            ax.set_xlabel(r"wind angle $\theta$ [deg]")
            ax.set_xticks([0, 90, 180, 270, 360])
            if i == 0 and j == 0:
                ax.legend(fontsize=5.2, loc="best")
    fig.tight_layout(pad=0.4)
    p = os.path.join(FIG, "fig1_polars.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # fig 2: grid convergence ---------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, ship in zip(axes, ships):
        g = gcis.get(ship, {})
        for key, mk in (("C_X", "o"), ("C_Y", "s")):
            lev = g.get(90.0, {}).get(key, {}).get("levels") if g else None
            if not lev:
                continue
            dxs = [t[0] for t in lev]
            vs = [t[1] for t in lev]
            ax.plot(dxs, vs, mk + "-", ms=3, lw=1.0, label=key)
            ext = g[90.0][key].get("extrapolated")
            if ext is not None:
                ax.axhline(ext, ls=":", lw=0.8, color="0.4")
        ax.set_xlabel(r"$\Delta x$ [m]"); ax.set_title(f"{ship}, $\\theta=90^\\circ$",
                                                       fontsize=7)
        ax.invert_xaxis(); ax.legend(fontsize=5.5)
    axes[0].set_ylabel("coefficient")
    fig.tight_layout(pad=0.4)
    p = os.path.join(FIG, "fig2_grid_convergence.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # fig 3: wake field ----------------------------------------------------
    fld = sorted(glob.glob(os.path.join(RAW, "*_a090_dx3.0_fields.npz")))
    if fld:
        fig, axes = plt.subplots(1, len(fld), figsize=(2.6 * len(fld), 2.4))
        if len(fld) == 1:
            axes = [axes]
        for ax, path in zip(axes, fld):
            z = np.load(path)
            dx = float(z["dx"])
            sl = z["speed_mid"] / float(z["u_lat"])
            solid = z["solid_mid"]
            img = np.where(solid > 0, np.nan, sl)
            im = ax.imshow(img, origin="lower", cmap="turbo", vmin=0.0,
                           vmax=float(np.nanpercentile(img, 99)))
            ax.set_title(os.path.basename(path).split("_a090")[0] + "  |u|/U",
                         fontsize=6)
            ax.set_xlabel("x [cells]"); ax.set_ylabel("y [cells]")
            ax.grid(False)
            fig.colorbar(im, ax=ax, fraction=0.045)
        fig.tight_layout(pad=0.4)
        p = os.path.join(FIG, "fig3_wake_field.png")
        fig.savefig(p, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # fig 4: annual assessment --------------------------------------------
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    width = 0.35
    xs = np.arange(len(ships))
    a0 = [annual[s]["cases"][0]["heading_mean_pct_superstructure"] for s in ships]
    a1 = [annual[s]["cases"][0]["heading_mean_pct_with_freeboard"] for s in ships]
    ax.bar(xs - width / 2, a0, width, label="superstructure only")
    ax.bar(xs + width / 2, a1, width, label="incl. hull freeboard estimate")
    ax.set_xticks(xs); ax.set_xticklabels(ships)
    ax.set_ylabel("annual mean wind resistance\n[% of calm-water resistance]")
    ax.legend(fontsize=5.5)
    fig.tight_layout()
    p = os.path.join(FIG, "fig4_annual_wind.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # fig 5: validation / cross-validation --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    ax = axes[0]
    if cube.get("runs"):
        names = [r["tag"] for r in cube["runs"]]
        cds = [r["C_D"] for r in cube["runs"]]
        ax.bar(names, cds, color="tab:gray")
        ax.axhline(CUBE_REF_CD, color="tab:red", ls="--", lw=1.0,
                   label=f"literature {CUBE_REF_CD}")
        ax.set_ylabel("$C_D$ (cube)"); ax.tick_params(axis="x", rotation=20,
                                                      labelsize=5.5)
        ax.legend(fontsize=5.5)
    ax = axes[1]
    for ship in ships:
        rows = loo[ship]["folds"]
        ax.plot([r["angle_deg"] for r in rows],
                [100 * r["err_C_X_rel"] for r in rows], "o-", ms=3, lw=1.0,
                label=f"{ship} $C_X$")
    ax.axhline(10.0, color="tab:red", ls="--", lw=1.0, label="10% target")
    ax.set_ylabel("LOO error [%]"); ax.set_xlabel(r"held-out $\theta$ [deg]")
    ax.legend(fontsize=5.0)
    fig.tight_layout(pad=0.4)
    p = os.path.join(FIG, "fig5_validation.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=DX_POLAR)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    os.makedirs(ART, exist_ok=True)

    recs = load_raw()
    ships = catalogue()
    dbs = {s: ship_records(s, args.dx, recs) for s in ships}
    coverage = sum(len(v) for v in dbs.values())

    # -- geometry validation ---------------------------------------------
    geom = {}
    for key, ship in ships.items():
        db = dbs[key]
        voxel = None
        if db:
            a0 = min(db, key=lambda d: abs(d["angle_deg"]))
            voxel = {"area_frontal_m2": a0["voxel_area_frontal_m2"],
                     "area_lateral_m2": a0["voxel_area_lateral_m2"]}
        geom[key] = validate_geometry(ship, voxel_areas=voxel)
        geom[key]["complexity"] = complexity_index(ship)
        geom[key]["analytic_projection"] = projected_areas(ship)

    # -- symmetry self-check ---------------------------------------------
    sym = {k: sign_conventions(v) for k, v in dbs.items() if v}

    # -- grid convergence -------------------------------------------------
    gcis = gci_table(recs)

    # -- engines, empirical models, cross-validation ----------------------
    emp, loo_auto, loo_emp, loo_lbm, engines = {}, {}, {}, {}, {}
    for key, ship in ships.items():
        db = dbs[key]
        if not db:
            continue
        engines[key] = HybridWindEngine(ship, db)
        emp[key] = empirical_vs_lbm(ship, db)
        loo_auto[key] = loo_validation(ship, db, route="auto")
        loo_emp[key] = loo_validation(ship, db, route="empirical")
        loo_lbm[key] = loo_validation(ship, db, route="lbm")

    # -- cube anchor ------------------------------------------------------
    cube_runs = []
    for r in recs:
        if r.get("case", {}).get("case") != "cube":
            continue
        cd = float(r["coefficients"]["C_D"])
        cube_runs.append({"tag": os.path.basename(r["_path"])[:-5],
                          "C_D": cd, "C_D_std": float(r["coefficients"]["C_D_std"]),
                          "re": float(r["case"]["re_side"]),
                          "cells_per_side": int(r["case"]["cells_per_side"]),
                          "rel_error_vs_literature": abs(cd - CUBE_REF_CD) / CUBE_REF_CD,
                          "source": r["_path"]})
    cube = {"literature_C_D": CUBE_REF_CD,
            "literature_range": [1.00, 1.10], "runs": cube_runs}

    # -- annual wind assessment ------------------------------------------
    annual = {k: annual_study(ships[k], engines[k]) for k in engines}

    # -- figures ----------------------------------------------------------
    figs = [] if args.no_figures else figures(dbs, emp, gcis, annual, cube,
                                              loo_auto)

    # -- acceptance -------------------------------------------------------
    def _max_gci(key: str, angle: float = 90.0):
        vals = []
        for s in ships:
            g = gcis.get(s, {}).get(angle, {}).get(key, {})
            if "gci_rel" in g:
                vals.append(g["gci_rel"])
        return max(vals) if vals else None

    geom_ok = sum(1 for g in geom.values() if g["ok"])
    mirror = max((max((p["rel_dev_C_X"] for p in s["mirror_pairs"]), default=0.0)
                  for s in sym.values()), default=None)
    mirror_y = max((max((p["rel_dev_C_Y"] for p in s["mirror_pairs"]), default=0.0)
                    for s in sym.values()), default=None)
    cube_best = (min(cube_runs, key=lambda r: r["rel_error_vs_literature"])
                 if cube_runs else None)
    loo_cx = max((v["max_err_C_X"] for v in loo_auto.values()), default=None)
    loo_cy = max((v["max_err_C_Y"] for v in loo_auto.values()), default=None)
    loo_emp_cx = max((v["max_err_C_X"] for v in loo_emp.values()), default=None)
    emp_buildup = max((v["max_err_C_X_buildup"] for v in emp.values()), default=None)

    criteria = [
        {"id": "G1", "description": "geometry validation (hull Cb/L/B, box "
         "overlap, port/starboard symmetry, voxel vs analytic projection areas "
         "within 2%) passes for all three catalogues",
         "value": geom_ok, "threshold": 3, "operator": ">=", "unit": "ships",
         "hard": True},
        {"id": "D1", "description": "LBM database coverage (ship x angle cases)",
         "value": coverage, "threshold": 21, "operator": ">=", "unit": "cases",
         "hard": True},
        {"id": "V1", "description": "port/starboard mirror self-check, max "
         "relative deviation of C_X over the 0/180, 30/150, 60/120 pairs",
         "value": mirror, "threshold": 0.10, "operator": "<=", "unit": "1",
         "hard": False},
        {"id": "V2", "description": "same mirror self-check for C_Y",
         "value": mirror_y, "threshold": 0.10, "operator": "<=", "unit": "1",
         "hard": False},
        {"id": "V3", "description": "bluff-body anchor: cube C_D versus the "
         "literature value 1.05 (best resolution run)",
         "value": None if cube_best is None else cube_best["rel_error_vs_literature"],
         "threshold": 0.10, "operator": "<=", "unit": "1", "hard": True},
        {"id": "C1", "description": "grid convergence: max GCI(C_X) at "
         "theta = 90 deg over the three ships", "value": _max_gci("C_X"),
         "threshold": 0.10, "operator": "<=", "unit": "1", "hard": True},
        {"id": "C2", "description": "grid convergence: max GCI(C_Y) at "
         "theta = 90 deg over the three ships", "value": _max_gci("C_Y"),
         "threshold": 0.10, "operator": "<=", "unit": "1", "hard": True},
        {"id": "E1", "description": "hybrid engine leave-one-angle-out error on "
         "C_X (all three ship types)", "value": loo_cx, "threshold": 0.10,
         "operator": "<=", "unit": "1", "hard": True},
        {"id": "E2", "description": "hybrid engine leave-one-angle-out error on "
         "C_Y (all three ship types)", "value": loo_cy, "threshold": 0.10,
         "operator": "<=", "unit": "1", "hard": True},
        {"id": "E3", "description": "empirical-only branch (Blendermann-style "
         "fit) leave-one-angle-out error on C_X", "value": loo_emp_cx,
         "threshold": 0.10, "operator": "<=", "unit": "1", "hard": False},
        {"id": "E4", "description": "a-priori component build-up versus LBM, max "
         "relative error on C_X (plausibility screen, not fitted)",
         "value": emp_buildup, "threshold": 0.25, "operator": "<=", "unit": "1",
         "hard": False},
        {"id": "A1", "description": "wind-rose annual assessment produced for "
         "all three ship types", "value": len(annual), "threshold": 3,
         "operator": ">=", "unit": "ships", "hard": False},
    ]
    acc = evaluate_acceptance(criteria)

    db_out = {
        "schema": "tensorlbm.ship_wind_database/v1",
        "node": "IN38-N04",
        "dx_m": args.dx,
        "conventions": {
            "frame": "ship frame, +x bow, +y starboard, +z up from the deck",
            "theta_deg": "wind direction from the bow (0 = head wind)",
            "C_X": "X_wind/(q A_F)", "C_Y": "Y_wind/(q A_L)",
            "C_K": "K_wind/(q A_L H)", "C_N": "N_wind/(q A_L B)",
            "rho_air": RHO_AIR,
        },
        "geometry_validation": geom,
        "symmetry": sym,
        "database": {k: v for k, v in dbs.items()},
        "grid_convergence": gcis,
        "cube_anchor": cube,
        "empirical_models": emp,
        "engines": {k: v.to_dict() for k, v in engines.items()},
        "cross_validation": {"hybrid_auto": loo_auto, "empirical_only": loo_emp,
                             "lbm_table_only": loo_lbm},
        "annual_wind": annual,
        "figures": [os.path.relpath(p, HERE) for p in figs],
        "raw_records": [r["_path"] for r in recs],
    }
    with open(os.path.join(ART, "wind_database.json"), "w",
              encoding="utf-8") as fh:
        json.dump(db_out, fh, indent=2, sort_keys=True, default=_default)
    with open(os.path.join(ART, "acceptance.json"), "w", encoding="utf-8") as fh:
        json.dump({"node": "IN38-N04", "schema":
                   "tensorlbm.wind_acceptance/v1", **acc,
                   "inputs": {"dx_m": args.dx, "coverage": coverage,
                              "cube_best": cube_best}},
                  fh, indent=2, sort_keys=True, default=_default)

    print(json.dumps({"coverage": coverage, "geometry_ok": geom_ok,
                      "hard_passed": f"{acc['n_hard_passed']}/{acc['n_hard']}",
                      "all_hard_passed": acc["all_hard_passed"],
                      "criteria": {c["id"]: (None if c["value"] is None
                                             else round(c["value"], 4),
                                             c["verdict"]) for c in acc["criteria"]}},
                     indent=1, default=_default))
    return 0


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


if __name__ == "__main__":
    raise SystemExit(main())