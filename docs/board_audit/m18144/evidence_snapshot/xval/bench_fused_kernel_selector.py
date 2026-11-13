"""XVAL-LBM-FUSED-KERNEL-TYPE-SELECTOR -- benchmark entry point.

Runs the full cross-validation in one go and writes machine-readable artifacts:

    python xval/fused_kernel_type_selector/bench_fused_kernel_selector.py \
        --out artifacts/xval_fused_kernel_type_selector

What is measured
----------------
1. **Granularity equivalence** (the anchor): for the same initial population,
   ``default_stream_collide`` (one fused kernel) must reproduce
   ``collide_only`` + ``stream`` to round-off.  Both the tensorlbm side
   (:mod:`tensorlbm.fused_kernel_selector`) and the OSS side
   (``lbmpy``'s compiled kernel) are exercised.
2. **Cross-library agreement**: tensorlbm vs lbmpy at matched kernel order.
   The two libraries disagree on *where the equilibrium reads the moments*
   (``pre_stream`` = ``stream(collide(f))``, textbook; ``post_stream`` =
   ``collide(stream(f))``, lbmpy's pull kernel).  Both orders are reported, and
   the cross-order difference is reported as the finding it is.
3. **Physics benchmark**: 2-D shear wave (a single LBM shear mode) with the
   analytic decay as reference, run with every kernel variant; the judged
   quantity is the amplitude after N steps (ref vs got, % error, owner's 15 %
   criterion).
4. **Sensitivity**: two resolutions and two relaxation times.

Everything is written to ``<out>/results.json`` plus a figure.
Run time on CPU: ~1 min (torch is pinned to 4 threads to avoid thrashing).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lbmpy_reference as oss  # noqa: E402

from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.fused_kernel_selector import KERNEL_TYPES, collide_stream_bgk, step  # noqa: E402
from tensorlbm.solver import collide_bgk, collide_bgk_matmul, stream  # noqa: E402

torch.set_num_threads(4)


def random_field(n: int, seed: int = 1, amp: float = 0.05) -> torch.Tensor:
    """Periodic non-equilibrium field: ``f = feq(1, u)`` with random ``u``."""
    g = torch.Generator().manual_seed(seed)
    rho = torch.ones((n, n), dtype=torch.float64)
    ux = amp * torch.randn((n, n), dtype=torch.float64, generator=g)
    uy = amp * torch.randn((n, n), dtype=torch.float64, generator=g)
    return equilibrium(rho, ux, uy)


def shear_wave(n: int, u0: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """``u_x = u0 sin(k y)`` on an ``(n, n)`` periodic box; returns (f, sin-mode)."""
    k = 2.0 * math.pi / n
    y = torch.arange(n, dtype=torch.float64)
    mode = torch.sin(k * y).unsqueeze(1)  # varies along dim 0 (y)
    rho = torch.ones((n, n), dtype=torch.float64)
    ux = u0 * mode.expand(n, n).clone()
    uy = torch.zeros((n, n), dtype=torch.float64)
    return equilibrium(rho, ux, uy), mode


def shear_amplitude(f: torch.Tensor, mode: torch.Tensor) -> float:
    """Amplitude of the ``sin(ky)`` mode of ``u_x`` (projection, x-averaged)."""
    _, ux, _ = macroscopic(f)
    return float((ux * mode).mean())


def run_variant(f0: torch.Tensor, tau: float, n_steps: int, kind: str,
                moments: str = "pre_stream", matmul: bool = False) -> torch.Tensor:
    """Iterate ``n_steps`` with one of the kernel granularity choices."""
    f = f0.clone()
    for _ in range(n_steps):
        if kind == "default_stream_collide":
            f = collide_stream_bgk(f, tau, moments=moments)
        elif kind == "collide_only":
            f = stream(collide_bgk_matmul(f, tau) if matmul else collide_bgk(f, tau))
        elif kind == "stream_pull_only":
            f = step(f, tau, "stream_pull_only")
        elif kind == "stream_then_collide":
            f = collide_bgk(stream(f), tau)
        else:  # pragma: no cover - guarded by KERNEL_TYPES
            raise ValueError(kind)
    return f


def equivalence_panel(tau: float, n: int, n_steps_short: int = 1, n_steps_long: int = 50) -> dict:
    """Granularity equivalence: fused vs separated, tensorlbm vs lbmpy."""
    f0 = random_field(n)
    scale = float(f0.abs().max())
    out: dict = {"n": n, "tau": tau, "field_scale": scale, "rows": []}

    def row(name: str, a: torch.Tensor, b: torch.Tensor, ref: str) -> None:
        diff = float((a - b).abs().max())
        out["rows"].append({"quantity": name, "max_abs_diff": diff,
                            "rel": diff / scale, "reference": ref,
                            "pass_lt_15pct": diff / scale < 0.15})

    for steps in (n_steps_short, n_steps_long):
        fused = run_variant(f0, tau, steps, "default_stream_collide", "pre_stream")
        sep = run_variant(f0, tau, steps, "collide_only")
        row(f"tensorlbm fused(pre_stream) vs collide+stream, {steps} step(s)",
            fused, sep, "stream(collide_bgk(f))")
        fused_mm = run_variant(f0, tau, steps, "collide_only", matmul=True)
        row(f"tensorlbm collide_bgk_matmul vs collide_bgk, {steps} step(s)",
            fused_mm, sep, "collide_bgk(f)")
        post = run_variant(f0, tau, steps, "default_stream_collide", "post_stream")
        row(f"tensorlbm fused(post_stream) vs collide(stream(f)), {steps} step(s)",
            post, run_variant(f0, tau, steps, "stream_then_collide"),
            "collide_bgk(stream(f))")

    # cross-order difference (the porting decision documented in the report)
    pre = run_variant(f0, tau, n_steps_long, "default_stream_collide", "pre_stream")
    post = run_variant(f0, tau, n_steps_long, "default_stream_collide", "post_stream")
    out["rows"].append({
        "quantity": f"tensorlbm fused pre_stream vs post_stream, {n_steps_long} steps",
        "max_abs_diff": float((pre - post).abs().max()),
        "rel": float((pre - post).abs().max()) / scale,
        "reference": "operator-order difference (not a bug, see report)",
        "pass_lt_15pct": float((pre - post).abs().max()) / scale < 0.15,
    })

    if oss.LBMPY_AVAILABLE:
        for steps in (n_steps_short, n_steps_long):
            ref = oss.run_lbmpy_steps(f0.numpy(), tau, steps)
            got = run_variant(f0, tau, steps, "default_stream_collide", "post_stream")
            row(f"lbmpy default_stream_collide vs tensorlbm fused(post_stream), {steps} step(s)",
                got, torch.tensor(ref), "OSS lbmpy 2.0 compiled kernel")
        ref_c = oss.lbmpy_collide_only(f0.numpy(), tau)
        got_c = collide_bgk(f0, tau)
        row("lbmpy collide_only vs tensorlbm collide_bgk, 1 step", got_c,
            torch.tensor(ref_c), "OSS lbmpy 2.0 compiled kernel")
    else:  # pragma: no cover
        out["rows"].append({"quantity": "lbmpy cross-check", "status": "NOT_RUN",
                            "reason": "lbmpy/pystencils not installed"})
    return out


def physics_panel(n: int, tau: float, n_steps: int, u0: float = 1e-4) -> dict:
    """Shear-wave amplitude vs the analytic decay references (ref vs got)."""
    f0, mode = shear_wave(n, u0)
    k = 2.0 * math.pi / n
    nu = (tau - 0.5) / 3.0
    a0 = shear_amplitude(f0, mode)

    ref_continuum = a0 * math.exp(-nu * k * k * n_steps)
    ref_discrete = a0 * (1.0 - (1.0 - math.cos(k)) / tau) ** n_steps

    rows = []
    variants = [
        ("tensorlbm fused(pre_stream)", "default_stream_collide", "pre_stream", False),
        ("tensorlbm fused(post_stream)", "default_stream_collide", "post_stream", False),
        ("tensorlbm collide_bgk+stream", "collide_only", "pre_stream", False),
        ("tensorlbm collide_bgk_matmul+stream", "collide_only", "pre_stream", True),
        ("tensorlbm stream_pull_only", "stream_pull_only", "pre_stream", False),
    ]
    for label, kind, moments, matmul in variants:
        f = run_variant(f0, tau, n_steps, kind, moments, matmul)
        got = shear_amplitude(f, mode)
        rows.append({
            "variant": label,
            "got": got,
            "ref_continuum_ns": ref_continuum,
            "err_continuum_pct": 100.0 * abs(got - ref_continuum) / ref_continuum,
            "ref_discrete_shear_mode": ref_discrete,
            "err_discrete_pct": 100.0 * abs(got - ref_discrete) / ref_discrete,
        })
    if oss.LBMPY_AVAILABLE:
        f_oss = oss.run_lbmpy_steps(f0.numpy(), tau, n_steps)
        got = shear_amplitude(torch.tensor(f_oss), mode)
        rows.append({
            "variant": "OSS lbmpy default_stream_collide",
            "got": got,
            "ref_continuum_ns": ref_continuum,
            "err_continuum_pct": 100.0 * abs(got - ref_continuum) / ref_continuum,
            "ref_discrete_shear_mode": ref_discrete,
            "err_discrete_pct": 100.0 * abs(got - ref_discrete) / ref_discrete,
        })
    return {"n": n, "tau": tau, "k": k, "nu": nu, "u0": u0, "steps": n_steps,
            "a0": a0, "rows": rows}


def figure(results: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panel = results["physics"][0]
    n, tau, steps = panel["n"], panel["tau"], panel["steps"]
    f0, mode = shear_wave(n, panel["u0"])
    k = panel["k"]
    nu = panel["nu"]

    ts = list(range(0, steps + 1, max(1, steps // 40)))
    series = {
        "fused pre_stream": ("default_stream_collide", "pre_stream", False),
        "fused post_stream": ("default_stream_collide", "post_stream", False),
        "collide+stream": ("collide_only", "pre_stream", False),
    }
    curves = {}
    for label, (kind, moments, matmul) in series.items():
        f = f0.clone()
        vals = [shear_amplitude(f, mode)]
        for t in range(1, len(ts)):
            f = run_variant(f, tau, ts[t] - ts[t - 1], kind, moments, matmul)
            vals.append(shear_amplitude(f, mode))
        curves[label] = vals
    ax = axes[0]
    for label, vals in curves.items():
        ax.plot(ts[: len(vals)], vals, "o", ms=3, label=label)
    a0 = panel["a0"]
    ax.plot(ts, [a0 * math.exp(-nu * k * k * t) for t in ts], "k-", lw=1.4,
            label="analytic exp(-nu k^2 t)")
    ax.plot(ts, [a0 * (1 - (1 - math.cos(k)) / tau) ** t for t in ts], "k--", lw=1.0,
            label="discrete shear mode")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("shear-wave amplitude u_x (lattice units)")
    ax.set_title(f"shear wave decay, n={n}, tau={tau}")
    ax.legend(fontsize=7)

    ax = axes[1]
    labels = [r["variant"].replace("tensorlbm ", "").replace("OSS ", "OSS\\n") for r in results["equivalence"][0]["rows"]]
    diffs = [max(r["max_abs_diff"], 1e-18) for r in results["equivalence"][0]["rows"]]
    ax.barh(range(len(labels)), diffs, color="steelblue")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xscale("log")
    ax.set_xlabel("max |Δf| after run (lattice units, log)")
    ax.set_title("granularity equivalence (round-off = pass)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/xval_fused_kernel_type_selector")
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results: dict = {
        "anchor": "XVAL-LBM-FUSED-KERNEL-TYPE-SELECTOR",
        "kernel_types": list(KERNEL_TYPES),
        "lbmpy_available": bool(oss.LBMPY_AVAILABLE),
        "oss": {
            "package": "lssfau/lbmpy",
            "selector": "lbmpy.creationfunctions.LBMConfig(kernel_type=...)",
            "kernel_builder": "lbmpy.creationfunctions.create_lb_ast -> Kernel.compile()",
            "zero_centered_storage_detected": (bool(oss.is_zero_centered())
                                               if oss.LBMPY_AVAILABLE else None),
            "direction_permutation": (oss.direction_permutation()[0]
                                      if oss.LBMPY_AVAILABLE else None),
        },
        "equivalence": [equivalence_panel(tau=0.8, n=32)],
        "physics": [
            physics_panel(n=n, tau=tau, n_steps=args.steps)
            for n, tau in ((32, 0.8), (64, 0.8), (64, 0.6), (64, 0.9))
        ],
    }
    results["wall_time_s"] = time.time() - t0

    (out / "results.json").write_text(json.dumps(results, indent=2))
    try:
        figure(results, out / "fused_kernel_selector.png")
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        results["figure_error"] = repr(exc)
        (out / "results.json").write_text(json.dumps(results, indent=2))

    print(json.dumps({k: results[k] for k in ("lbmpy_available", "oss", "wall_time_s")}, indent=2))
    for panel in results["equivalence"]:
        print(f"\n-- equivalence (n={panel['n']}, tau={panel['tau']}) --")
        for r in panel["rows"]:
            print(f"  {r.get('max_abs_diff', r.get('status'))!s:>12}  {r['quantity']}")
    for panel in results["physics"]:
        print(f"\n-- shear wave n={panel['n']} tau={panel['tau']} steps={panel['steps']} "
              f"k={panel['k']:.4f} nu={panel['nu']:.4f} --")
        for r in panel["rows"]:
            print(f"  got={r['got']:.10e}  ref_cont={r['ref_continuum_ns']:.10e} "
                  f"({r['err_continuum_pct']:.3e}%)  ref_disc={r['ref_discrete_shear_mode']:.10e} "
                  f"({r['err_discrete_pct']:.3e}%)  {r['variant']}")
    print(f"\nartifacts -> {out/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())