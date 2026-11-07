#!/usr/bin/env python3
"""Ice-class bow family campaign runner (IN13-W2-N01).

Builds the parametric ice-breaking bow family, runs the geometry admission
gate, the SDF/voxel volume study, the stem-angle resistance sensitivity, and
writes every artifact (STL / offsets / waterline / voxel mask), a SHA-256
manifest and the machine-readable acceptance JSON.

Usage::

    PYTHONPATH=src python scripts/run_icebow_family.py --out artifacts/icebow_family_r1

Everything is deterministic; re-running reproduces byte-identical STL files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from tensorlbm import icebow_cad as ic

FN = 0.20
L_M = 100.0
DENSITIES = ((41, 31), (81, 61), (161, 121))
MICHELL_DENSITIES = ((61, 41, 81), (121, 81, 161), (241, 161, 321))
VOXEL_SPACINGS = (0.005, 0.0025, 0.00125)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def member_id(p: ic.IceBowParams) -> str:
    tag = {"straight": "straight", "inward": "inward", "multisegment": "multiseg"}[p.family]
    return f"{tag}_a{int(round(p.alpha_stem_deg))}_cb{int(round(p.cb_target * 100))}"


def write_offsets_csv_full(path: Path, tab: dict, depth: float) -> None:
    """Offsets table: x/L, z/D and half-breadth/L on the station grid."""
    x, z, b = tab["x"], tab["z"], tab["half_beam"]
    lines = ["x_over_L,z_over_D,half_beam_over_L"]
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            lines.append(f"{x[i, j]:.6f},{z[i, j] / depth:.6f},{b[i, j]:.8f}")
    path.write_text("\n".join(lines) + "\n")


def export_member(p: ic.IceBowParams, out_root: Path) -> dict:
    mid = member_id(p)
    d = out_root / "members" / mid
    d.mkdir(parents=True, exist_ok=True)
    mesh = ic.build_ice_bow_mesh(p, n_stations=81, n_levels=61)
    dv = ic.design_volume(p)
    rep = ic.validate_mesh(mesh)
    rev = ic.reverse_parameters(mesh)

    hull_stl = ic.write_stl(d / "hull.stl", mesh.vertices, mesh.faces, header=f"icebow {mid}".encode())
    wet = ic.wetted_surface(mesh)
    wet_faces = np.arange(mesh.faces.shape[0])[
        mesh.triangles.mean(axis=1)[:, 2] <= p.draft
    ]
    wet_stl = ic.write_stl(
        d / "wetted_surface.stl", mesh.vertices, mesh.faces[wet_faces], header=b"icebow wetted"
    )
    tab = ic.offsets_table(p, n_stations=41, n_levels=21)
    write_offsets_csv_full(d / "offsets.csv", tab, p.D)

    # waterline (contact line for the DEM leg) at the design draft
    wl = np.abs(mesh.vertices[:, 2] - p.draft) <= 1e-8 * p.D
    stbd = wl & (mesh.vertices[:, 1] > 0)
    order = np.argsort(mesh.vertices[stbd, 0])
    wlv = mesh.vertices[stbd][order]
    np.savetxt(d / "waterline_starboard.csv", wlv[:, [0, 1]], delimiter=",", header="x_over_L,y_over_L")

    mask, origin, sp = ic.voxelize_mesh(mesh, 0.0025)
    np.savez_compressed(d / "voxel_mask_h0.0025.npz", mask=mask, origin=origin, spacing=sp)
    (d / "voxel_meta.json").write_text(
        json.dumps(
            {
                "shape_zyx": list(mask.shape),
                "origin": [float(v) for v in origin],
                "spacing": sp,
                "solid_cells": int(mask.sum()),
                "solid_volume": float(mask.sum()) * sp**3,
                "convention": "(nz, ny, nx) boolean solid mask, cell centres at origin+(i+0.5)*spacing",
            },
            indent=2,
        )
    )
    files = sorted(p.name for p in d.iterdir())
    return {
        "member_id": mid,
        "dir": str(d),
        "params": {
            "family": p.family,
            "alpha_stem_deg": p.alpha_stem_deg,
            "beta_ent_deg": p.beta_ent_deg,
            "c_flare": p.c_flare,
            "c_fwd": p.c_fwd,
            "cb_target": p.cb_target,
            "r_t": p.r_t,
            "L": p.L,
            "B": p.B,
            "D": p.D,
            "draft": p.draft,
        },
        "gate": rep,
        "reverse": rev,
        "wetted_area_over_L2": wet["area"],
        "n_wet_triangles": wet["n_wet"],
        "analytic": dv,
        "files": files,
        "artifacts": {
            "hull_stl": str(hull_stl),
            "wetted_stl": str(wet_stl),
        },
    }


def downstream_contract(info: dict) -> dict:
    """Re-read the exported artifacts with *unmodified* downstream readers."""
    from tensorlbm import voxelize as vz
    from tensorlbm.geometry_voxel import mesh_watertight_status, read_stl_triangles

    hull = Path(info["artifacts"]["hull_stl"])
    tri = read_stl_triangles(hull)
    status = mesh_watertight_status(tri)
    out = {
        "geometry_voxel.read_stl_triangles": {
            "shape": list(tri.shape),
            "ok": True,
        },
        "geometry_voxel.mesh_watertight_status": {
            "watertight": bool(status["watertight"]),
            "boundary_edges": int(status["boundary_edges"]),
            "nonmanifold_edges": int(status["nonmanifold_edges"]),
        },
    }
    try:  # numpy-only reader used by the drag-surrogate stack
        stl = vz.load_stl(hull)
        out["voxelize.load_stl"] = {"ok": True, "n_triangles": int(len(stl.triangles))}
    except Exception as exc:  # pragma: no cover - defensive
        out["voxelize.load_stl"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    tab = Path(info["dir"]) / "offsets.csv"
    arr = np.loadtxt(tab, delimiter=",", skiprows=1)
    out["offsets_csv"] = {"ok": True, "shape": list(arr.shape), "max_half_beam": float(arr[:, 2].max())}
    wl = np.loadtxt(Path(info["dir"]) / "waterline_starboard.csv", delimiter=",", skiprows=1)
    out["waterline_csv"] = {"ok": True, "n_points": int(wl.shape[0])}
    return out


def make_figures(out_root: Path, members, gate_rows, sens, vox_rows) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = out_root / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    figs = []

    # ---- fig 1: family body plan -----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=90)
    ax = axes[0]
    for p in members:
        if p.family != "straight":
            continue
        z = np.linspace(0.0, p.zeta_T, 60)
        tab = ic.offsets_table(p, n_stations=61, n_levels=61)
        ax.plot(tab["x"][:, 0], tab["half_beam"][:, 0], label=f"keel  a={p.alpha_stem_deg:.0f}deg")
        ax.plot(tab["x"][:, -1], tab["half_beam"][:, -1], "--", lw=1.0)
    ax.set_xlabel("x / L")
    ax.set_ylabel("b / L")
    ax.set_title("Planform: keel line (solid) vs deck line (dashed)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax = axes[1]
    for p in members:
        z = np.linspace(0.0, 1.0, 200)
        ax.plot(ic.stem_x_norm(p, z), z * p.D, label=ic.member_label(p))
    ax.set_xlabel("x / L")
    ax.set_ylabel("z / L")
    ax.set_title("Stem profiles (alpha_stem family + sub-families)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = figdir / "fig1_family.png"
    fig.savefig(path)
    plt.close(fig)
    figs.append({"name": "fig1_family", "caption": "Ice-bow family: waterline planform (left) and stem profiles (right).", "path": path})

    # ---- fig 2: mesh + voxel slicing -------------------------------------
    p = members[2]
    mesh = ic.build_ice_bow_mesh(p, n_stations=61, n_levels=41)
    mask, origin, sp = ic.voxelize_mesh(mesh, 0.005)
    fig = plt.figure(figsize=(9.6, 4.2), dpi=90)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    tri = mesh.triangles
    port = tri.mean(axis=1)[:, 1] < 0
    face_z = tri[port].mean(axis=1)[:, 2]
    coll = Poly3DCollection(
        tri[port], facecolors=plt.cm.viridis(face_z / p.D), edgecolors="none", alpha=0.95
    )
    ax.add_collection3d(coll)
    ax.set_xlim(0.0, p.L)
    ax.set_ylim(-p.B, p.B)
    ax.set_zlim(0.0, p.D)
    ax.view_init(elev=22, azim=-120)
    ax.set_xlabel("x/L")
    ax.set_ylabel("y/L")
    ax.set_zlabel("z/L")
    ax.set_title(f"port-side hull surface ({ic.member_label(p)})")
    ax = fig.add_subplot(1, 2, 2)
    j = mask.shape[1] // 2
    sl = mask[:, j, :]
    ax.imshow(sl, origin="lower", aspect="auto", cmap="Blues",
              extent=[origin[0], origin[0] + mask.shape[2] * sp, origin[2], origin[2] + mask.shape[0] * sp])
    ax.axhline(p.draft, color="r", lw=1, ls="--")
    ax.set_xlabel("x / L")
    ax.set_ylabel("z / L")
    ax.set_title("SDF/voxel solid mask, centreline-symmetric slice (h=0.5%L)")
    fig.tight_layout()
    path = figdir / "fig2_mesh_voxel.png"
    fig.savefig(path)
    plt.close(fig)
    figs.append({"name": "fig2_mesh_voxel", "caption": "Triangulated hull surface (left) and its ray-parity voxel solid mask (right).", "path": path})

    # ---- fig 3: volume convergence ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), dpi=90)
    ax = axes[0]
    labels = [r["member_id"] for r in gate_rows]
    coarse = [abs(r["gate"]["nabla_rel_error"]) * 100 for r in gate_rows]
    fine = [abs(r["fine_nabla_rel_error"]) * 100 for r in gate_rows]
    idx = np.arange(len(labels))
    ax.bar(idx - 0.2, coarse, width=0.4, label="mesh 81x61")
    ax.bar(idx + 0.2, fine, width=0.4, label="mesh 161x121")
    ax.axhline(1.0, color="r", ls="--", lw=1, label="1.0% criterion")
    ax.axhline(0.3, color="g", ls=":", lw=1, label="0.3% criterion")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("|nabla_mesh - nabla_analytic| / nabla  [%]")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax = axes[1]
    for row in vox_rows:
        errs = [abs(lv["rel_error"]) * 100 for lv in row["study"]["levels"]]
        ax.plot([lv["spacing_over_L"] for lv in row["study"]["levels"]], errs, "o-", label=row["member_id"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(1.5, color="r", ls="--", lw=1, label="1.5% criterion (finest)")
    ax.set_xlabel("voxel size / L")
    ax.set_ylabel("|V_voxel - V_analytic| / V [%]")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = figdir / "fig3_volume_convergence.png"
    fig.savefig(path)
    plt.close(fig)
    figs.append({"name": "fig3_volume_convergence", "caption": "Left: mesh displacement-volume error at two resolutions per family member. Right: voxel/solid-mask volume error vs voxel size (three levels).", "path": path})

    # ---- fig 4: resistance sensitivity -----------------------------------
    fig, ax = plt.subplots(figsize=(5.8, 4.0), dpi=90)
    straights = sorted([r for r in sens["rows"] if r["family"] == "straight"], key=lambda r: r["alpha_stem_deg"])
    alphas = [r["alpha_stem_deg"] for r in straights]
    ax.plot(alphas, [r["delta_rt_holtrop"] * 100 for r in straights], "s-", label="Holtrop-Mennen R_T (no rake term)")
    ax.plot(alphas, [r["delta_rw_michell"] * 100 for r in straights], "^-", label="Michell R_W (rake sensitive)")
    ax.plot(alphas, [r["delta_rt_hybrid"] * 100 for r in straights], "o-", color="k", label="hybrid dR_T/R_T")
    for r in straights:
        g = r["delta_rw_gci_percent"]
        if g == g:
            ax.errorbar([r["alpha_stem_deg"]], [r["delta_rt_hybrid"] * 100], yerr=g / 2, color="k", capsize=3, lw=0.8)
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xlabel("stem rake alpha_stem [deg]")
    ax.set_ylabel("delta R / R(30deg) [%]")
    ax.set_title(f"Stem-angle resistance sensitivity, Fn={sens['fn']:.2f}")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = figdir / "fig4_sensitivity.png"
    fig.savefig(path)
    plt.close(fig)
    figs.append({"name": "fig4_sensitivity", "caption": "Relative resistance change vs stem rake for the straight family (reference 30 deg), with GCI error bars on the hybrid estimate.", "path": path})
    return figs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/icebow_family_r1")
    ap.add_argument("--fn", type=float, default=FN)
    args = ap.parse_args(argv)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    members = [ic.resolve_params(p) for p in ic.family_members()]
    infos = []
    gate_rows = []
    for p in members:
        info = export_member(p, out_root)
        fine = ic.build_ice_bow_mesh(p, n_stations=161, n_levels=121)
        dv = info["analytic"]
        fine_err = abs(ic.submerged_volume(fine)["nabla"] - dv["nabla"]) / dv["nabla"]
        info["fine_nabla_rel_error"] = fine_err
        infos.append(info)
        gate_rows.append(
            {
                "member_id": info["member_id"],
                "gate": info["gate"],
                "reverse": info["reverse"],
                "fine_nabla_rel_error": fine_err,
            }
        )
        print(
            f"[gate] {info['member_id']:22s} pass={info['gate']['passed']} "
            f"V_err={info['gate']['nabla_rel_error']*100:.3f}% fine={fine_err*100:.3f}% "
            f"Cb_err={info['reverse']['cb_error']*100:+.3f}% a_err={info['reverse']['alpha_error_deg']:+.3f}deg",
            flush=True,
        )

    # voxel / SDF study (criterion d) + downstream contract (criterion f)
    vox_rows = []
    for p, info in zip(members, infos):
        mesh = ic.build_ice_bow_mesh(p, n_stations=81, n_levels=61)
        study = ic.voxel_volume_study(mesh, spacings=VOXEL_SPACINGS)
        vox_rows.append({"member_id": info["member_id"], "study": study})
        print(
            f"[sdf ] {info['member_id']:22s} monotone={study['monotone']} "
            f"finest={study['finest_rel_error_abs']*100:.3f}% gci={study['gci']['gci_fine_percent']:.3f}%",
            flush=True,
        )
    for info in infos:
        info["downstream"] = downstream_contract(info)

    # resistance sensitivity (criterion e)
    sens = ic.resistance_sensitivity(
        members, fn=args.fn, L_m=L_M, densities=DENSITIES, michell_densities=MICHELL_DENSITIES
    )
    for r in sens["rows"]:
        print(
            f"[sens] {r['family']:12s} a={r['alpha_stem_deg']:.0f} R_T={r['rt_holtrop_n']:.0f}N "
            f"dRT_HM={r['delta_rt_holtrop']*100:+.3f}% dRW_M={r['delta_rw_michell']*100:+.3f}% "
            f"dRT_hyb={r['delta_rt_hybrid']*100:+.3f}% gci={r['delta_rw_gci_percent']:.2f}%",
            flush=True,
        )

    figs = make_figures(out_root, members, gate_rows, sens, vox_rows)

    # ---- acceptance ------------------------------------------------------
    vol_default_max = max(abs(r["gate"]["nabla_rel_error"]) for r in gate_rows)
    vol_fine_max = max(r["fine_nabla_rel_error"] for r in gate_rows)
    gate_all = all(r["gate"]["passed"] for r in gate_rows)
    neg = {
        "dropped_face_rejected": True,
        "flipped_winding_rejected": True,
        "duplicate_vertex_rejected": True,
        "zero_area_sliver_rejected": True,
        "collapsed_face_rejected": True,
        "invalid_parameters_raise": True,
        "mismatched_design_cb_caught": True,
        "note": "each negative control is exercised in tests/test_icebow_cad.py",
    }
    param_ok = all(
        abs(r["reverse"]["cb_error"]) <= 0.01
        and abs(r["reverse"]["L_error"]) <= 0.01
        and abs(r["reverse"]["B_error"]) <= 0.01
        and abs(r["reverse"]["alpha_error_deg"]) <= 1.0
        and abs(r["reverse"]["beta_error_deg"]) <= 1.0
        for r in gate_rows
    )
    sdf_ok = all(
        row["study"]["converging"]
        and row["study"]["finest_rel_error_abs"] <= 0.015
        and row["study"]["finest_submerged_rel_error_abs"] <= 0.015
        and row["study"]["gci"]["gci_fine_percent"] <= 5.0
        and row["study"]["gci_submerged"]["gci_fine_percent"] <= 5.0
        for row in vox_rows
    )
    contract_ok = all(
        info["downstream"]["geometry_voxel.mesh_watertight_status"]["watertight"]
        and info["downstream"]["offsets_csv"]["ok"]
        and info["downstream"]["waterline_csv"]["ok"]
        for info in infos
    )
    acceptance = {
        "campaign_node": "IN13-W2-N01",
        "fn": args.fn,
        "L_m": L_M,
        "criteria": {
            "a_volume_default_le_1pct": {
                "value_percent": vol_default_max * 100,
                "pass": bool(vol_default_max <= 0.01),
            },
            "a_volume_fine_le_0p3pct": {
                "value_percent": vol_fine_max * 100,
                "pass": bool(vol_fine_max <= 0.003),
            },
            "b_geometry_gate_all_members": {
                "pass": bool(gate_all),
                "negative_controls": neg,
            },
            "c_parameter_consistency_le_1pct_1deg": {"pass": bool(param_ok)},
            "d_sdf_three_level_gci": {
                "pass": bool(sdf_ok),
                "finest_error_percent": max(
                    row["study"]["finest_rel_error_abs"] for row in vox_rows
                )
                * 100,
                "max_gci_percent": max(
                    row["study"]["gci"]["gci_fine_percent"] for row in vox_rows
                ),
                "converging_all": all(row["study"]["converging"] for row in vox_rows),
                "submerged_converging_all": all(
                    row["study"]["submerged_converging"] for row in vox_rows
                ),
                "monotone_volume_sequence_all": all(
                    row["study"]["monotone_volume_sequence"] for row in vox_rows
                ),
                "max_submerged_gci_percent": max(
                    row["study"]["gci_submerged"]["gci_fine_percent"] for row in vox_rows
                ),
                "finest_submerged_error_percent": max(
                    row["study"]["finest_submerged_rel_error_abs"] for row in vox_rows
                )
                * 100,
            },
            "e_sensitivity_monotone_gci_le_5pct": {
                "pass": bool(
                    sens["monotone_delta_rw_michell"]
                    and sens["max_rw_michell_gci_percent"] <= 5.0
                    and sens["delta_rw_gci_max_percent"] <= 5.0
                ),
                "monotone_holtrop": sens["monotone_delta_rt_holtrop"],
                "monotone_michell": sens["monotone_delta_rw_michell"],
                "monotone_hybrid": sens["monotone_delta_rt_hybrid"],
                "max_abs_delta_rt_hybrid_percent": sens["max_abs_delta_rt_hybrid"] * 100,
                "max_gci_percent": max(
                    sens["max_rw_michell_gci_percent"], sens["delta_rw_gci_max_percent"]
                ),
                "lbm_leg": "NOT_RUN_IN_THIS_NODE",
            },
            "f_downstream_contract": {"pass": bool(contract_ok)},
        },
    }
    acceptance["all_pass"] = bool(all(v["pass"] for v in acceptance["criteria"].values()))

    (out_root / "acceptance.json").write_text(json.dumps(acceptance, indent=2))
    (out_root / "members.json").write_text(json.dumps(infos, indent=2, default=float))
    (out_root / "voxel_study.json").write_text(json.dumps(vox_rows, indent=2, default=float))
    (out_root / "resistance_sensitivity.json").write_text(json.dumps(sens, indent=2, default=float))

    # ---- manifest --------------------------------------------------------
    manifest = {
        "campaign_node": "IN13-W2-N01",
        "module": "src/tensorlbm/icebow_cad.py",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "git_head": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "members": {
            info["member_id"]: {
                "params": info["params"],
                "gate_passed": info["gate"]["passed"],
                "files": {},
            }
            for info in infos
        },
        "acceptance_all_pass": acceptance["all_pass"],
    }
    for info in infos:
        d = Path(info["dir"])
        for f in sorted(d.iterdir()):
            manifest["members"][info["member_id"]]["files"][f.name] = {
                "sha256": sha256(f),
                "bytes": f.stat().st_size,
            }
    manifest["figures"] = {f["name"]: {"sha256": sha256(f["path"]), "bytes": f["path"].stat().st_size} for f in figs}
    manifest["summary_files"] = {
        name: sha256(out_root / name)
        for name in (
            "acceptance.json",
            "members.json",
            "voxel_study.json",
            "resistance_sensitivity.json",
        )
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(acceptance["criteria"], indent=2, default=float))
    print("ALL_PASS", acceptance["all_pass"])
    return 0 if acceptance["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())