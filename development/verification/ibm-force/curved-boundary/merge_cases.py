"""Merge per-case JSONs into the stage files and the acceptance report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_validation import ROOT, summarise  # noqa: E402

OUT = ROOT / "artifacts/p399w2_n04"


def _fix_strouhal(case: dict) -> None:
    """Recompute St from the stored series (older runs used per-sample units)."""
    se = int(case["params"].get("sample_every", 20))
    st = case["metrics"].get("strouhal")
    if st is not None:
        case["metrics"]["strouhal"] = st / max(1, se)
        case["metrics"]["strouhal_note"] = (
            f"recomputed from the {se}-step sampled lift series"
        )


def main() -> None:
    cases = []
    for p in sorted(OUT.glob("case_*.json")):
        try:
            cases.append(json.loads(p.read_text()))
        except json.JSONDecodeError:  # pragma: no cover - partial file
            print(f"[skip] unreadable {p.name}")
    if not cases:
        raise SystemExit("no case_*.json found")

    cyl = [
        c
        for c in cases
        if c["label"].startswith("re100") and "fields" in c
    ]
    stk = [c for c in cases if c["label"].startswith("hasimoto")]
    for c in cases:
        _fix_strouhal(c)
    cyl.sort(key=lambda c: (c["params"]["scheme"], c["params"]["n_markers"]))
    stk.sort(key=lambda c: c["params"]["box_in_diameters"][0])

    if cyl:
        (OUT / "cylinder_re100.json").write_text(
            json.dumps({"stage": "cylinder", "cases": cyl}, indent=2)
        )
    if stk:
        (OUT / "stokes_lowre.json").write_text(
            json.dumps({"stage": "stokes", "cases": stk}, indent=2)
        )
    quad = (
        json.loads((OUT / "quadrature_audit.json").read_text())
        if (OUT / "quadrature_audit.json").exists()
        else None
    )
    report = summarise(
        quad,
        {"stage": "cylinder", "cases": cyl} if cyl else None,
        {"stage": "stokes", "cases": stk} if stk else None,
    )
    couette = (
        json.loads((OUT / "case_couette_torque.json").read_text())
        if (OUT / "case_couette_torque.json").exists()
        else None
    )
    if couette is not None:
        cm = couette["metrics"]
        for name, value, thr, src in (
            ("couette_torque_rel_error", cm.get("torque_rel_error"), 0.01,
             "|T_LBM| vs exact 4πμΩR1²R2²/(R2²−R1²) (Taylor–Couette, ReΩ≈"
             f"{couette['params']['re_omega']:.3f}), case_couette_torque.json"),
            ("couette_traction_pointwise_rel_error",
             cm.get("traction_pointwise_rel_error"), 0.05,
             "surface traction τ_k=F_k/Δs_k vs exact constant wall shear −2μB/R1², "
             "case_couette_torque.json"),
            ("couette_profile_rel_error", cm.get("profile_max_rel_error"), 0.05,
             "max |u_θ − (Ar+B/r)|/(ΩR1) in the gap, case_couette_torque.json"),
        ):
            report["criteria"][name] = {
                "value": float(value) if value is not None else None,
                "threshold": thr,
                "unit": "-",
                "pass": bool(value is not None and float(value) <= thr),
                "source": src,
            }
        report["all_pass"] = all(c["pass"] for c in report["criteria"].values())
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: (v["value"], v["pass"]) for k, v in report["criteria"].items()}, indent=2))


if __name__ == "__main__":
    main()