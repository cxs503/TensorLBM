"""Assemble ``result.json`` for the mcmp_stratified_poiseuille benchmark.

Reads every JSON record produced by
``cases/mcmp_stratified_poiseuille/run_matrix.py`` (cluster or local; the
records are the only source of numbers — nothing is retyped), applies the
thresholds declared in ``spec.json`` and writes ``result.json`` next to this
file together with a sha256 of each input record.

Usage::

    python benchmarks/verified/mcmp_stratified_poiseuille/assemble.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RECORDS = os.path.join(ROOT, "artifacts", "mcmp_stratified_poiseuille", "records")
SPEC = os.path.join(HERE, "spec.json")


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    spec = json.load(open(SPEC))
    thr = spec["thresholds"]
    files = sorted(f for f in os.listdir(RECORDS) if f.endswith(".json"))
    if not files:
        print(f"no records in {RECORDS}", file=sys.stderr)
        return 1

    cells, checksums, wall = [], {}, []
    for name in files:
        path = os.path.join(RECORDS, name)
        rec = json.load(open(path))
        checksums[name] = sha256_of(path)
        if rec["mode"] == "walls":
            wall.append(dict(
                tag=name[:-5], ny=rec["config"]["ny"], tau=rec["config"]["tau_lower"],
                **{k: v for k, v in rec["wall_calibration"].items()},
            ))
            continue
        if not rec.get("finite", False):
            cells.append(dict(tag=name[:-5], ny=rec["config"]["ny"],
                              G_12=rec["config"]["G_12"],
                              u_eq=rec["config"]["u_eq"],
                              force_scheme=rec["config"]["force_scheme"],
                              steps=rec["config"]["steps"],
                              finite=False, steady=False,
                              verdict=dict(gate_ok=False, flow_rate_ok=False,
                                           layer_ok=False, interface_ok=False,
                                           spurious_ok=False,
                                           flow_rate_ok_vs_component_rule=False)))
            continue
        ref = rec["analytical"]["halfway/mixture"]
        ref_comp = rec["analytical"]["halfway/component"]
        cell = dict(
            tag=name[:-5],
            ny=rec["config"]["ny"], G_12=rec["config"]["G_12"],
            tau_upper=rec["config"]["tau_upper"],
            force_scheme=rec["config"]["force_scheme"], u_eq=rec["config"]["u_eq"],
            steps=rec.get("steps_total")
            or (rec["history"][-1]["step"] if rec.get("history") else rec["config"]["steps"]),
            gates=dict(
                finite=rec["finite"], steady=rec["steady"],
                phi_contrast=rec["phi_contrast"], stratified=rec["stratified"],
            ),
            Q1=rec["Q1"], Q2=rec["Q2"], Q_mix=rec["Q_mix"],
            Q_mix_lower=rec["layer_flux"]["Q_mix_lower"],
            Q_mix_upper=rec["layer_flux"]["Q_mix_upper"],
            u_max_mix=rec["u_max_mix"], spurious_mix=rec["spurious_mix"],
            spurious_ratio=rec["spurious_mix"] / max(rec["u_max_mix"], 1e-30),
            trace_velocity_max=rec["trace_velocity_max"],
            interface_crossing=rec["interface_crossing"],
            wall_convention="halfway (H = ny-2), fixed by the wall_calibration control",
            errors_mixture={k: ref[k] for k in
                            ("err_Q_mix", "err_Q_mix_lower", "err_Q_mix_upper",
                             "err_profile_L2", "err_interface_rel")},
            errors_component={k: ref_comp[k] for k in
                              ("err_Q_mix", "err_Q_mix_lower", "err_Q_mix_upper",
                               "err_profile_L2", "err_interface_rel")},
            reference=dict(nu_lower=ref["nu_lower"], nu_upper=ref["nu_upper"],
                           Q_lower=ref["Q_lower"], Q_upper=ref["Q_upper"],
                           Q_total=ref["Q_total"]),
            runtime=rec["runtime"],
        )
        cell["verdict"] = dict(
            gate_ok=bool(cell["gates"]["finite"] and cell["gates"]["steady"]
                         and cell["gates"]["stratified"]),
            flow_rate_ok=bool(abs(cell["errors_mixture"]["err_Q_mix"]) < thr["err_Q_mix_abs"]),
            layer_ok=bool(abs(cell["errors_mixture"]["err_Q_mix_lower"]) < thr["err_Q_mix_layer_abs"]
                          and abs(cell["errors_mixture"]["err_Q_mix_upper"]) < thr["err_Q_mix_layer_abs"]),
            interface_ok=bool(abs(cell["errors_mixture"]["err_interface_rel"]) < thr["err_interface_rel_abs"]),
            spurious_ok=bool(cell["spurious_ratio"] < thr["spurious_ratio"]),
            flow_rate_ok_vs_component_rule=bool(
                abs(cell["errors_component"]["err_Q_mix"]) < thr["err_Q_mix_abs"]),
        )
        cells.append(cell)

    acc = spec["parameters"]["acceptance_cells"]
    repair_tags = {c["tag"] for c in acc if c["role"].startswith("repair")}
    holdout_tags = {c["tag"] for c in acc if c["role"].startswith("hold-out")}
    primary = [c for c in cells if c["tag"] in repair_tags]
    holdout = [c for c in cells if c["tag"] in holdout_tags]
    overall = bool(primary) and all(
        c["verdict"]["gate_ok"] and c["verdict"]["flow_rate_ok"] for c in primary
    ) and len(primary) >= len(repair_tags)

    out = dict(
        benchmark="mcmp_stratified_poiseuille",
        anchor=spec["anchor"],
        thresholds=thr,
        overall_verdict=dict(
            acceptance_cells_ok=overall,
            n_acceptance_cells_found=len(primary),
            n_acceptance_cells_declared=len(repair_tags),
            hold_out=[dict(tag=c["tag"], ny=c["ny"], steady=c.get("gates", {}).get("steady", False),
                           err_Q_mix_mixture_rule=c["errors_mixture"]["err_Q_mix"],
                           err_Q_mix_component_rule=c["errors_component"]["err_Q_mix"],
                           counts_as_acceptance=bool(c["verdict"]["gate_ok"]))
                      for c in holdout],
            note=("acceptance cell = u_eq=viscous + force_scheme=guo at G_12=-2.5, ny=25 and ny=33; "
                  "a cell counts only if finite & steady & stratified.  The hold-out cell is "
                  "reported with its own gate status and does not enter the acceptance flag."),
        ),
        cells=cells,
        wall_calibration=wall,
        record_sha256=checksums,
        environment=sorted({(c["runtime"]["python"], c["runtime"]["torch"], c["runtime"]["platform"])
                            for c in cells if "runtime" in c}),
    )
    dst = os.path.join(HERE, "result.json")
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {dst}: {len(cells)} flow cells, {len(wall)} wall-calibration cells, "
          f"acceptance_ok={overall}")
    for c in cells:
        if "errors_mixture" not in c:
            print(f"  {c['tag']:34s} ny={c['ny']:2d} g={c['G_12']:+.1f} {c['u_eq']:8s} "
                  f"{c['force_scheme']:5s} DIVERGED (not finite)")
            continue
        e = c["errors_mixture"]["err_Q_mix"]
        print(f"  {c['tag']:34s} ny={c['ny']:2d} g={c['G_12']:+.1f} {c['force_scheme']:5s} "
              f"phi={c['gates']['phi_contrast']:.2f} steady={c['gates']['steady']} "
              f"err_Qmix(mix)={e*100:+.1f}% err_Qmix(comp)="
              f"{c['errors_component']['err_Q_mix']*100:+.1f}% "
              f"S/Umax={c['spurious_ratio']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())