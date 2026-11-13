#!/usr/bin/env python3
"""Entry point for XVAL-LBM-BITWISE-REFDATA-REGRESSION-PLUS-BENCHMAR.

Usage
-----
    PYTHONPATH=src python3 benchmarks/xval_bitwise_refdata/run.py gen      # (re)freeze golden refdata
    PYTHONPATH=src python3 benchmarks/xval_bitwise_refdata/run.py verify    # bitwise regression gate
    PYTHONPATH=src python3 benchmarks/xval_bitwise_refdata/run.py bench     # analytic TGV benchmark
    PYTHONPATH=src python3 benchmarks/xval_bitwise_refdata/run.py check     # verify + negative controls
                                                                            # + benchmark -> result.json + figure
Exit code 0 iff every gate passes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent.parent) not in sys.path:  # allow `python run.py` without PYTHONPATH
    sys.path.insert(0, str(HERE.parent.parent))

from harness import (  # noqa: E402
    benchmark,
    full_check,
    generate,
    load_golden,
    negative_controls,
    refdata_path,
    verify,
)


def _print_regression(reg: dict) -> None:
    print(
        f"[regression] env_match={reg['env_match']} ulp_budget={reg['ulp_budget']} "
        f"cases={reg['n_cases']} bitwise_ok={reg['n_bitwise_ok']} "
        f"ulp_drift={reg['n_ulp_drift']} regression={reg['n_regression']} "
        f"passed={reg['passed']}"
    )
    for name, r in reg["cases"].items():
        print(f"  {'OK ' if r['verdict'] == 'BITWISE_OK' else '!! '} {name:24s} {r['verdict']}")
    if reg["env_diffs"]:
        print("  env/code diffs vs golden:")
        for d in reg["env_diffs"]:
            print(f"    - {d}")


def _print_bench(bench: dict) -> None:
    print(f"[benchmark] ref = {bench['reference']}")
    for row in bench["ref_vs_got"]:
        print(
            f"  {row['item']:34s} ref={row['ref']:.8e} got={row['got']:.8e} "
            f"err={row['err_pct']:.4f}% pass={row['pass']}"
        )
    print(
        f"  worst_err_pct={bench['worst_err_pct']:.4f}% tol={bench['tolerance_pct']}% "
        f"passed={bench['passed']}"
    )


def make_figure(bench: dict, out_dir: Path) -> dict:
    """Two panels: amplitude decay vs analytic (log) and per-point error."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for lv in bench["levels"]:
        t = [s["t"] for s in lv["series"]]
        ax.semilogy(
            t, [s["amp_ref"] for s in lv["series"]], "-", lw=1.2, label=f"analytic {lv['level']}"
        )
        ax.semilogy(
            t,
            [s["amp_got"] for s in lv["series"]],
            "o",
            ms=3.0,
            mfc="none",
            label=f"LBM {lv['level']}",
        )
    ax.set_xlabel("step")
    ax.set_ylabel(r"$A(t)=\sqrt{\langle |u|^2\rangle}$")
    ax.set_title("Taylor-Green decay: LBM vs analytic")
    ax.legend(fontsize=6)
    ax.grid(True, which="both", alpha=0.25)

    ax = axes[1]
    for lv in bench["levels"] + bench["tau_points"]:
        t = [s["t"] for s in lv["series"]]
        ax.plot(t, [s["amp_err_pct"] for s in lv["series"]], "-", lw=1.2, label=lv["level"])
    ax.axhline(bench["tolerance_pct"], color="r", ls="--", lw=1.0, label="15% acceptance")
    ax.set_xlabel("step")
    ax.set_ylabel("amplitude error vs analytic [%]")
    ax.set_title("Benchmark error (all four parameter points)")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "tgv_bitwise_bench.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    return {"name": png.name, "path": str(png.relative_to(HERE.parent.parent))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["gen", "verify", "bench", "check", "controls"])
    ap.add_argument("--refdata", default=None, help="golden refdata path")
    ap.add_argument("--out", default=None, help="result.json output path (check)")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)

    refdata = Path(args.refdata) if args.refdata else refdata_path()

    if args.cmd == "gen":
        doc = generate()
        refdata.parent.mkdir(parents=True, exist_ok=True)
        refdata.write_text(json.dumps(doc, indent=1))
        print(f"[gen] wrote {refdata} ({len(doc['cases'])} cases, git={doc['git_sha']})")
        for name, c in doc["cases"].items():
            print(f"  {name:24s} sha256={c['digest'][:16]}… inv={c['invariants']}")
        return 0

    golden = load_golden(refdata)

    if args.cmd == "verify":
        reg = verify(golden)
        _print_regression(reg)
        return 0 if reg["passed"] else 1

    if args.cmd == "controls":
        neg = negative_controls(golden)
        print(json.dumps(neg, indent=1))
        return 0 if neg["both_detected"] else 1

    if args.cmd == "bench":
        bench = benchmark()
        _print_bench(bench)
        return 0 if bench["passed"] else 1

    result = full_check(golden)
    _print_regression(result["regression"])
    print(f"[negative_controls] both_detected={result['negative_controls']['both_detected']}")
    _print_bench(result["benchmark"])

    out = Path(args.out) if args.out else HERE / "result.json"
    if not args.no_figure:
        fig = make_figure(result["benchmark"], HERE / "figures")
        result["figures"] = [fig]
    result["harness_sha256"] = (
        __import__("hashlib").sha256((HERE / "harness.py").read_bytes()).hexdigest()
    )
    out.write_text(json.dumps(result, indent=1))
    print(f"[check] wrote {out}")
    print(f"[check] PASSED={result['passed']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
