#!/usr/bin/env python3
"""Audit Triton codegen pinning in TensorLBM; write a machine-readable manifest.

Pin-the-codegen-before-you-tune (upstream practice: ``AminZar69/CudaPhaseField``
pins ``-maxrregcount`` before tuning a CUDA kernel).  This tool re-derives, from
the *source*, every ``@triton.jit`` kernel, every launch site and the codegen
knobs each site passes, then diffs that against
``tensorlbm.codegen_pin.DECLARED_PINS``.

Deliberately static (AST + module-constant resolution): it must run on CPU-only
build nodes where no kernel can be launched, and on CI where importing a CUDA
module is undesirable.  No GPU means no measured numbers -- those stay absent
rather than being invented.

Outputs
-------
``artifacts/codegen_pin_manifest.json``
    Full report: kernels, launch sites, wrapper defaults, declared pins,
    drift findings, environment snapshot.
``artifacts/codegen_pin_audit.md``
    Human-readable summary with the drift table.

Exit status
-----------
0 when declared pins agree with the sources, 1 when any drift is found
(so CI can gate on "a codegen knob changed without re-declaring the pin").

Usage
-----
    python tools/codegen_pin_audit.py [--out-dir artifacts] [--json-only]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.codegen_pin import (  # noqa: E402
    DECLARED_PINS,
    audit_codegen,
    environment_snapshot,
)


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    return f"`{value}`"


def render_markdown(report: dict) -> str:
    summary = report["summary"]
    env = report["environment"]
    lines: list[str] = []
    lines.append("# Triton codegen pin audit (TensorLBM)")
    lines.append("")
    lines.append(
        "Pin the codegen knobs (`num_warps`, `num_stages`, `maxnreg`) **before** "
        "tuning: an unpinned register budget lets ptxas re-allocate registers on "
        "every kernel edit, so throughput drifts for reasons unrelated to the "
        "optimisation under test (upstream practice: `AminZar69/CudaPhaseField`, "
        "`--maxrregcount` pinned before tuning)."
    )
    lines.append("")
    lines.append(f"* audited modules: {len(report['modules'])}")
    lines.append(f"* `@triton.jit` kernels: {summary['kernels']}")
    lines.append(f"* launch sites: {summary['launch_sites']}")
    lines.append(
        f"* launch sites routed through `codegen_pin.launch_kwargs`: "
        f"{summary['launch_sites_pinned_via_codegen_pin']}"
    )
    lines.append(
        "* launch sites forwarding a `maxnreg` argument: "
        f"{summary['launch_sites_forwarding_maxnreg']}"
    )
    lines.append(
        f"* declared pins that fix a `maxnreg` value: "
        f"{summary['declared_pins_with_maxnreg']} of {len(report['declared_pins'])}"
    )
    lines.append(f"* distinct declared pin fingerprints: {summary['distinct_pin_fingerprints']}")
    lines.append(f"* drift findings: **{summary['drift_findings']}**")
    lines.append("")
    lines.append("## Declared pins")
    lines.append("")
    lines.append("| kernel | source | num_warps | num_stages | maxnreg | fingerprint |")
    lines.append("|---|---|---|---|---|---|")
    for name, pin in report["declared_pins"].items():
        knobs = pin["knobs"]
        lines.append(
            "| {name} | `{source}` | {nw} | {ns} | {mn} | `{fp}` |".format(
                name=name,
                source=pin["source"],
                nw=_fmt(knobs.get("num_warps")),
                ns=_fmt(knobs.get("num_stages")),
                mn=_fmt(knobs.get("maxnreg")),
                fp=pin["fingerprint"],
            )
        )
    lines.append("")
    lines.append("## Launch sites")
    lines.append("")
    lines.append("| module | line | kernel | num_warps | num_stages | maxnreg | routed via |")
    lines.append("|---|---|---|---|---|---|---|")
    for site in report["launch_sites"]:
        codegen = site["codegen"]
        lines.append(
            "| `{module}` | {line} | `{kernel}` | {nw} | {ns} | {mn} | {via} |".format(
                module=site["module"],
                line=site["line"],
                kernel=site["kernel"],
                nw=_fmt(codegen.get("num_warps")),
                ns=_fmt(codegen.get("num_stages")),
                mn=_fmt(codegen.get("maxnreg")),
                via=site["pinned_via"] or "inline",
            )
        )
    lines.append("")
    lines.append("## Drift")
    lines.append("")
    if not report["drift"]:
        lines.append("No drift: every declared pin matches the audited source defaults.")
    else:
        lines.append("| kernel | knob | declared | audited | issue |")
        lines.append("|---|---|---|---|---|")
        for finding in report["drift"]:
            lines.append(
                "| {kernel} | {knob} | {declared} | {audited} | {issue} |".format(
                    kernel=finding.get("kernel"),
                    knob=finding.get("knob", "—"),
                    declared=_fmt(finding.get("declared")),
                    audited=_fmt(finding.get("audited")),
                    issue=finding.get("issue"),
                )
            )
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    for key in (
        "timestamp",
        "platform",
        "machine",
        "python",
        "triton_version",
        "torch_version",
        "cuda_available",
        "device_name",
        "nvcc_version",
        "maxnreg_supported",
    ):
        lines.append(f"* {key}: `{env.get(key)}`")
    lines.append("")
    lines.append(
        "`maxnreg_supported: false` on this host means no CUDA backend is installed, "
        "so a register cap cannot be exercised here. The pin still declares the "
        "intent; `launch_kwargs` forwards `maxnreg` only where the backend accepts it "
        "(never silently dropping it without the audit recording it)."
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts"))
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args(argv)

    report = audit_codegen()
    report["environment"] = environment_snapshot()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "codegen_pin_manifest.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {json_path}")
    if not args.json_only:
        md_path = out_dir / "codegen_pin_audit.md"
        md_path.write_text(render_markdown(report), encoding="utf-8")
        print(f"wrote {md_path}")
    summary = report["summary"]
    print(json.dumps(summary, indent=2, sort_keys=True))
    if report["drift"]:
        print(f"DRIFT: {len(report['drift'])} finding(s)", file=sys.stderr)
        return 1
    pinned = sum(1 for pin in DECLARED_PINS.values() if pin.maxnreg is not None)
    print(f"declared pins: {len(DECLARED_PINS)} ({pinned} with maxnreg)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
