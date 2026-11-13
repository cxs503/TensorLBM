#!/usr/bin/env python3
"""Board audit of m17865 (XVAL-LBM-ISO8-WALL-NORMAL-TABLE) -- independent re-derivation.

Reads *only* the delivered tree /root/tp-agent/work/m17865 (read-only) and the
platform DB, and re-derives the headline numbers from raw delivered artifacts
(binary OSS tables, fixture CSV, vendored Fortran source) with its own parsers.
The delivery's own harness/module is never imported.

Outputs
-------
artifacts/m17865_audit/audit_m17865.json   machine-readable readings
docs/board_audit/m17969/figures/*.png      audit figures
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

DEL = Path("/root/tp-agent/work/m17865")
ART = DEL / "artifacts" / "xval_iso8_wall_normal_table"
XV = DEL / "xval" / "XVAL-LBM-ISO8-WALL-NORMAL-TABLE"
HERE = Path(__file__).resolve().parent
OUT = Path("/root/tp-agent/work/m17969/artifacts/m17865_audit")
OUT.mkdir(parents=True, exist_ok=True)
(HERE / "figures").mkdir(parents=True, exist_ok=True)

res: dict = {}


# --------------------------------------------------------------------------- #
# 1. provenance of the vendored upstream source
# --------------------------------------------------------------------------- #
src_path = XV / "oss_src" / "wall_boundary_process.f90"
raw = src_path.read_bytes()
text = src_path.read_text(errors="replace")
declared = json.load(open(ART / "results.json"))["oss"]["vendored_sha256"]
res["provenance"] = {
    "file": str(src_path),
    "bytes": len(raw),
    "sha256_of_bytes": hashlib.sha256(raw).hexdigest(),
    "sha256_of_text_read_encode": hashlib.sha256(text.encode()).hexdigest(),
    "delivery_declared_sha256": declared,
    "crlf_lines": raw.count(b"\r\n"),
    "lf_lines": raw.count(b"\n"),
}
res["provenance"]["declared_matches_bytes"] = (
    res["provenance"]["sha256_of_bytes"] == declared
)
res["provenance"]["declared_matches_normalised_text"] = (
    res["provenance"]["sha256_of_text_read_encode"] == declared
)


# --------------------------------------------------------------------------- #
# 2. independent parse of the ISO8 stencil out of the vendored Fortran
#    (own regexes -- not the delivery's extractor)
# --------------------------------------------------------------------------- #
def frac(tok: str) -> float:
    tok = tok.strip().replace("d0", "")
    m = re.fullmatch(r"([-\d.]+)/([-\d.]+)", tok)
    return float(m.group(1)) / float(m.group(2)) if m else float(tok)


m_iso = re.search(r"ISO8\(1:7\)\s*=\s*\(/(.*?)/\)", text, re.S)
iso8_w = [frac(t) for t in m_iso.group(1).split(",")]
m_w = re.search(r"w\(0:3\)\s*=\s*\(/(.*?)/\)", text, re.S)
dkq27_w = [frac(t) for t in m_w.group(1).split(",")]
m_ex = re.search(r"ex\(0:26\)\s*=\s*\(/(.*?)/\)", text, re.S)
m_ey = re.search(r"ey\(0:26\)\s*=\s*\(/(.*?)/\)", text, re.S)
m_ez = re.search(r"ez\(0:26\)\s*=\s*\(/(.*?)/\)", text, re.S)
latt = [tuple(float(v) for v in (re.sub(r"d0", "", a).strip(),
                                 re.sub(r"d0", "", b).strip(),
                                 re.sub(r"d0", "", c).strip()))
        for a, b, c in zip(m_ex.group(1).split(","), m_ey.group(1).split(","),
                           m_ez.group(1).split(","))]

# 34 gradient pairs per gradient component (own parse): the upstream source
# writes "nwx = ISO8(1)*( A - B ) + ISO8(2)*( A - B + A - B ... ) + ..."; each
# ISO8 weight owns a contiguous chunk of (i+a,j+b,k+c) terms taken two at a time.
TERM = (r"walls_smooth_2\(\s*i\s*([+-]\s*\d+)?\s*,\s*j\s*([+-]\s*\d+)?\s*,"
        r"\s*k\s*([+-]\s*\d+)?\s*\)")


def component_pairs(name: str, stop: str):
    m = re.search(rf"{name}\s*=\s*&(.*?)\n\s*{stop}\s*=", text, re.S)
    chunks = re.split(r"ISO8\((\d)\)\s*\*", m.group(1))
    groups = []
    for wi, chunk in zip(chunks[1::2], chunks[2::2]):
        trip = [tuple(int(re.sub(r"\s", "", t)) if t else 0 for t in t3)
                for t3 in re.findall(TERM, chunk)]
        groups.append((int(wi), list(zip(trip[0::2], trip[1::2]))))
    pairs = [pr for _, g in groups for pr in g]
    weights = [iso8_w[wi - 1] for wi, g in groups for _ in g]
    return groups, pairs, weights


groups, pairs, pair_weight = component_pairs("nwx", "nwy")
_, pairs_y, weight_y = component_pairs("nwy", "nwz")
_, pairs_z, weight_z = component_pairs("nwz", "tmp")
# convention of the module: offsets are the A-B differences, the symmetric
# half-offset is d//2 -> gain on a linear field = sum w * (A-B)/2
gain_x = sum(w * (a[0] - b[0]) / 2.0 for w, (a, b) in zip(pair_weight, pairs))
gain_y = sum(w * (a[1] - b[1]) / 2.0 for w, (a, b) in zip(weight_y, pairs_y))
gain_z = sum(w * (a[2] - b[2]) / 2.0 for w, (a, b) in zip(weight_z, pairs_z))
gain_raw = {ax: 2.0 * g for ax, g in (("x", gain_x), ("y", gain_y), ("z", gain_z))}

# y/z pair sets must be axis permutations of the x set (isotropy, own check)
px = sorted((a[0] - b[0], a[1] - b[1], a[2] - b[2]) for a, b in pairs)
dy = sorted((a[1] - b[1], a[0] - b[0], a[2] - b[2]) for a, b in pairs_y)  # swap x<->y
dz = sorted((a[2] - b[2], a[1] - b[1], a[0] - b[0]) for a, b in pairs_z)  # swap x<->z

shell_counts = [sum(1 for v in latt if sum(abs(c) for c in v) == lvl)
                for lvl in range(4)]
res["stencil_independent"] = {
    "iso8_weights": iso8_w,
    "iso8_weight_count": len(iso8_w),
    "iso8_weights_sum": float(sum(iso8_w)),
    "n_gradient_pairs": len(pairs),
    "pairs_per_weight": [len(g) for _, g in groups],
    "gradient_pairs_identical_to_report_34": len(pairs) == 34,
    "d3q27_kernel_weights": dkq27_w,
    "d3q27_shell_counts": [int(n) for n in shell_counts],
    "d3q27_kernel_weight_sum": float(sum(w * n for w, n in zip(dkq27_w, shell_counts))),
    "d3q27_kernel_mass_conserving": bool(
        abs(sum(w * n for w, n in zip(dkq27_w, shell_counts)) - 1.0) < 1e-12),
    "lattice_vectors_parsed": len(latt),
    "operator_gain_linear_halfoffset": {"x": gain_x, "y": gain_y, "z": gain_z},
    "operator_gain_raw_difference_convention": gain_raw,
    "operator_gain_anisotropy": max(gain_raw.values()) / min(gain_raw.values()),
    "delivery_claim_gain": 0.8809523809523804,
    "pair_set_permutation_symmetry_holds": bool(px == dy == dz),
    "n_pairs_per_component": [len(pairs), len(pairs_y), len(pairs_z)],
    "mirror_bc_mentions": len(re.findall(r"mirror_bc", text)),
}


# --------------------------------------------------------------------------- #
# 3. independent parse of the delivered OSS binary tables + analytic normals
# --------------------------------------------------------------------------- #
def parse_dat(p: Path):
    b = p.read_bytes()
    ns, nf = np.frombuffer(b[:8], "<i4")
    off = 8
    solid_dt = np.dtype([("ix", "<i4"), ("iy", "<i4"), ("iz", "<i4"),
                         ("i_fluid_num", "<i4"), ("neighbor_list", "<i4", 18),
                         ("la_weight", "<f8")])
    fluid_dt = np.dtype([("ix", "<i4"), ("iy", "<i4"), ("iz", "<i4"),
                         ("snx", "<f8"), ("sny", "<f8"), ("snz", "<f8")])
    solid = np.frombuffer(b[off:off + ns * solid_dt.itemsize], dtype=solid_dt)
    off += ns * solid_dt.itemsize
    fluid = np.frombuffer(b[off:off + nf * fluid_dt.itemsize], dtype=fluid_dt)
    off += nf * fluid_dt.itemsize
    assert off == len(b), (p, off, len(b))
    return int(ns), int(nf), solid, fluid


def norm_stats(n: np.ndarray, ref: np.ndarray) -> dict:
    nrm = np.linalg.norm(n, axis=1)
    valid = nrm > 0
    rn = np.linalg.norm(ref, axis=1)
    ok = valid & (rn > 0)
    valid = ok
    rhat = ref[valid] / rn[valid, None]
    c = np.abs((n[valid] / nrm[valid, None] * rhat).sum(1))
    c = np.clip(c, -1.0, 1.0)
    th = np.degrees(np.arccos(c))
    rel = 2 * np.sin(np.radians(th) / 2)
    return {"n_nodes": int(valid.sum()), "n_degenerate": int((~valid).sum()),
            "mean_rel": float(rel.mean()), "max_rel": float(rel.max()),
            "mean_deg": float(th.mean()), "max_deg": float(th.max()),
            "p95_deg": float(np.percentile(th, 95))}


CASES = {
    "sphere_n48_R10_offcenter": dict(dir="sphere_n48_R10_offcenter", n=48, kind="sphere",
                                     radius=10.0, center=(24.3, 23.6, 24.1)),
    "sphere_n64_R14_offcenter": dict(dir="sphere_n64_R14_offcenter", n=64, kind="sphere",
                                     radius=14.0, center=(32.2, 31.5, 32.4)),
    "plane_n40_alpha26.565": dict(dir="plane_n40_alpha26.565", n=40, kind="plane",
                                  normal=(2.0, 1.0, 0.0)),
    "plane_n40_alpha45": dict(dir="plane_n40_alpha45", n=40, kind="plane",
                              normal=(1.0, 1.0, 0.0)),
}
delivery = {c["case"]: c for c in json.load(open(ART / "results.json"))["cases"]}
oss_arm = {}
for name, c in CASES.items():
    f = ART / "oss_runs" / c["dir"] / "output" / "indirect_addressing_walls_boundary.dat"
    ns, nf, solid, fluid = parse_dat(f)
    pos = np.column_stack([fluid["ix"], fluid["iy"], fluid["iz"]]).astype(float)
    if c["kind"] == "sphere":
        ref = pos - np.asarray(c["center"], float)          # radial
    else:
        ref = np.tile(np.asarray(c["normal"], float), (len(pos), 1))
    n = np.column_stack([fluid["snx"], fluid["sny"], fluid["snz"]])
    inside = np.all((pos >= 1) & (pos <= c["n"]), axis=1)
    allstat, physstat = norm_stats(n, ref), norm_stats(n[inside], ref[inside])
    d = delivery[name]
    oss_arm[name] = {
        "dat": str(f.relative_to(DEL)),
        "n_solid_boundary_parsed": ns,
        "n_fluid_boundary_parsed": nf,
        "n_physical_nodes_parsed": int(inside.sum()),
        "delivery_n_fluid_boundary": d["n_fluid_boundary"],
        "delivery_n_nodes_physical": d["n_nodes_physical"],
        "my_full_table_stats": allstat,
        "my_physical_only_stats": physstat,
        "delivery_k2": {k: d["iso8_smooth4"][k] for k in
                        ("n_nodes", "mean_rel", "max_rel", "mean_deg", "max_deg",
                         "p95_deg", "n_degenerate")},
        "mean_rel_abs_diff_vs_delivery": abs(physstat["mean_rel"]
                                             - d["iso8_smooth4"]["mean_rel"]),
        "max_rel_abs_diff_vs_delivery": abs(physstat["max_rel"]
                                            - d["iso8_smooth4"]["max_rel"]),
    }
res["oss_table_independent_recompute"] = oss_arm


# --------------------------------------------------------------------------- #
# 4. K1 independent: shipped fixture CSV (TensorLBM nodes) vs the OSS binary
# --------------------------------------------------------------------------- #
fix = np.genfromtxt(ART / "oss_fixture_normal_table.csv", delimiter=",", names=True)
ns, nf, solid, fluid = parse_dat(
    ART / "oss_runs" / "sphere_n48_R10_offcenter_fixture" / "output"
    / "indirect_addressing_walls_boundary.dat")
key_fix = {(int(a), int(b), int(c)) for a, b, c in
           zip(fix["i"], fix["j"], fix["k"])}
key_oss = {(int(a), int(b), int(c)) for a, b, c in
           zip(fluid["ix"], fluid["iy"], fluid["iz"])}
order = sorted(key_oss)
oss_map = {(int(a), int(b), int(c)): (x, y, z) for a, b, c, x, y, z in
           zip(fluid["ix"], fluid["iy"], fluid["iz"], fluid["snx"], fluid["sny"],
               fluid["snz"])}
common = sorted(key_fix & key_oss)
n_fix = np.array([(float(r["snx"]), float(r["sny"]), float(r["snz"]))
                  for r in fix])
fix_map = {(int(r["i"]), int(r["j"]), int(r["k"])):
           (float(r["snx"]), float(r["sny"]), float(r["snz"])) for r in fix}
A = np.array([fix_map[k] for k in common])
B = np.array([oss_map[k] for k in common])
res["k1_independent_from_artifacts"] = {
    "fixture_rows": len(fix), "oss_fluid_rows": int(nf),
    "node_sets_identical": key_fix == key_oss,
    "n_common_nodes": len(common),
    "max_abs_dn_recomputed": float(np.abs(A - B).max()),
    "max_angle_deg_recomputed": float(np.degrees(np.arccos(
        np.clip((A / np.linalg.norm(A, axis=1)[:, None]
                 * B / np.linalg.norm(B, axis=1)[:, None]).sum(1), -1, 1))).max()),
    "delivery_claim_max_abs_dn": delivery["sphere_n48_R10_offcenter"][
        "oss_vs_ours_max_abs_dn"],
    "predeclared_k1_threshold": 1e-12,
}


# --------------------------------------------------------------------------- #
# 5. report tables / figures / six sections
# --------------------------------------------------------------------------- #
md = (DEL / "docs" / "XVAL-LBM-ISO8-WALL-NORMAL-TABLE.md").read_text()
heads = re.findall(r"^##\s+(.*)$", md, re.M)
sections = {
    "①案例介绍": bool(re.search(r"^## ①", md, re.M)),
    "②方法": bool(re.search(r"^## ②", md, re.M)),
    "③参数表": bool(re.search(r"^## ③", md, re.M)),
    "④对照表": bool(re.search(r"^## ④", md, re.M)),
    "⑤收敛/敏感性": bool(re.search(r"^## ⑤", md, re.M)),
    "⑥失败模式/工件": bool(re.search(r"^## ⑥", md, re.M)),
    "差异清单": "Difference list" in md,
    "repro_cmd": "Reproduce:" in md,
}
figs = sorted((ART / "figures").glob("*.png"))
res["report"] = {
    "path": "docs/XVAL-LBM-ISO8-WALL-NORMAL-TABLE.md",
    "headings": heads,
    "six_sections": sections,
    "figures": [{"name": f.name, "bytes": f.stat().st_size} for f in figs],
    "figures_are_png": all(f.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" for f in figs),
    "claims_k1_pass": "K1 **PASS**" in md,
    "claims_k2_honest_fail": "honest FAIL" in md,
    "claims_k4_fail": "**FAIL** (16" in md or "K4 **FAIL**" in md,
}

# predeclared criteria vs delivered numbers
pre = json.load(open(XV / "predeclared.json"))
th = pre["predeclared_thresholds"]
gauges = {}
for name, d in delivery.items():
    iso = d["iso8_smooth4"]
    curv = d["curvature_iso8"]
    gauges[name] = {
        "K1_max_abs_dn": d["oss_vs_ours_max_abs_dn"],
        "K1_pass": d["oss_vs_ours_max_abs_dn"] <= 1e-12,
        "K2_mean_rel": iso["mean_rel"],
        "K2_mean_pass_15pct": iso["mean_rel"] <= th["K2_mean_rel"],
        "K2_max_rel": iso["max_rel"],
        "K2_max_pass_15pct": iso["max_rel"] <= th["K2_max_rel"],
        "K4_magnitude_rel_err": (abs(abs(curv["kappa_mean"]) - abs(curv["kappa_ref"]))
                                 / abs(curv["kappa_ref"])) if curv["kappa_ref"] else None,
        "K4_raw": curv,
    }
isot = delivery["sphere_n48_R10_offcenter"]["operator_isotropy"]
res["predeclared_gauges"] = {
    "thresholds": th,
    "per_case": gauges,
    "K3_isotropy": isot,
    "n_cases": len(gauges),
    "K1_pass_count": sum(g["K1_pass"] for g in gauges.values()),
    "K2_mean_pass_count": sum(g["K2_mean_pass_15pct"] for g in gauges.values()),
    "K2_max_pass_count": sum(g["K2_max_pass_15pct"] for g in gauges.values()),
}

# benchmark judge (the only authority): tensor_benchmark_cases rows
res["benchmark_judge"] = {
    "sql": "SELECT case_name,metric,tolerance FROM tensor_benchmark_cases "
           "WHERE anchor_slug='XVAL-LBM-ISO8-WALL-NORMAL-TABLE'",
    "rows": 0,
    "note": "0 rows (psql as owner); TensorLBM anchor has no registered case/tolerance",
}

# integration / durability
res["integration"] = {
    "callers_in_delivery_tree": 0,
    "callers_in_main_clone": 0,
    "commit": "a43fd917",
    "commit_on_main_clone": False,
    "note": "grep wall_normal_iso8 outside module/tests/xval harness -> 0; "
            "/root/tensor-ecosystem/TensorLBM HEAD be78a87b has no such file",
}

# --------------------------------------------------------------------------- #
# 6. benchmark judge via the platform DB (owner role) + integration grep
# --------------------------------------------------------------------------- #
import subprocess

def sql(q: str) -> str:
    return subprocess.run(["psql", "-U", "postgres", "tensorplatform", "-A", "-t",
                           "-F", "|", "-c", q], capture_output=True, text=True,
                          check=True).stdout.strip()

res["benchmark_judge"]["rows_raw"] = sql(
    "SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug="
    "'XVAL-LBM-ISO8-WALL-NORMAL-TABLE'")
res["benchmark_judge"]["rows"] = int(res["benchmark_judge"]["rows_raw"] or 0)
res["benchmark_judge"]["tensorbm_total_rows"] = int(sql(
    "SELECT count(*) FROM tensor_benchmark_cases") or 0)
res["benchmark_judge"]["same_anchor_family_rows"] = sql(
    "SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug ILIKE '%ISO8%'"
    " OR case_name ILIKE '%WALL%NORMAL%'")
res["verdicts_already_on_scope"] = sql(
    "SELECT count(*) FROM tensor_maturity_verdicts WHERE scope LIKE 'm17865%'")

grep = subprocess.run(["grep", "-rn", "wall_normal_iso8", str(DEL),
                       "--include=*.py"], capture_output=True, text=True).stdout
outside = [ln for ln in grep.splitlines()
           if "src/tensorlbm/boundary/wall_normal_iso8.py" not in ln
           and "tests/test_wall_normal_iso8.py" not in ln
           and "scripts/xval_iso8_wall_normal_table.py" not in ln]
res["integration"]["grep_total_hits"] = len(grep.splitlines())
res["integration"]["grep_hits_outside_module_tests_harness"] = len(outside)
res["integration"]["main_clone_has_module"] = Path(
    "/root/tensor-ecosystem/TensorLBM/src/tensorlbm/boundary/wall_normal_iso8.py").exists()
res["repro_rerun"] = json.loads((Path(__file__).resolve().parent
                                 / "repro_rerun.json").read_text()) \
    if (Path(__file__).resolve().parent / "repro_rerun.json").exists() else None

(OUT / "audit_m17865.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
print(json.dumps(res, indent=2, ensure_ascii=False)[:4000])