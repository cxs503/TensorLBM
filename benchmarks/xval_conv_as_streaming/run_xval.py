#!/usr/bin/env python3
"""XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL -- cross-validation benchmark.

Reference (OSS, inventoried): ``loliverhennigh/Lattice-Boltzmann-fluid-flow-in-
Tensorflow``.  Its transport step is a *fixed-kernel convolution*: populations
are held channel-last ``[1, *Ndim, q]`` and streaming is one constant 3x3 (3D:
3x3x3) cross-correlation whose tail block ``[q_in, q_out]`` is diagonal
(``LatFlow/D2Q9.py`` STREAM table, ``LatFlow/Domain.py::StreamSC``), wall bounce
is a 1x1 cross-channel convolution with the direction-reversal matrix blended
through the solid mask (``Domain.py::CollideSC``), and periodicity is a manual
"mobius" ghost-row concat (``LatFlow/utils.py::pad_mobius``).

This benchmark asks the two questions an adoption decision needs:

1. **Implementation** -- is the ported operator the *same operator*?  The
   in-house gather transport (``tensorlbm.solver.stream``) and the ported
   fixed-kernel transport (``tensorlbm.streaming_conv``) are compared on the
   same state: 2D bit-wise, 3D against the canonical roll traversal.
2. **Physics** -- does the transported solution still hit the closed forms?
   2D Taylor-Green (exact Navier-Stokes, periodic) and plane Poiseuille
   (exact, wall-bounded, exercises the 1x1 bounce convolution).

The upstream tables are machine-extracted into ``oss_ref/`` (see
``extract_oss_tables.py``, provenance with sha256), never transcribed; the
comparison of the derived kernel against upstream's table is itself part of the
evidence, and it exposes one shipped upstream defect (D3Q19 ``q=3`` tap).

Run:  python benchmarks/xval_conv_as_streaming/run_xval.py
      python benchmarks/xval_conv_as_streaming/run_xval.py --quick     (CI-sized)
Docs: the markdown report is rendered from ``result.json`` by the same command
      (``render_docs``), so no number in the write-up is typed by hand.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm import streaming_conv as SC  # noqa: E402
from tensorlbm.d2q9 import C as C2D, OPPOSITE, W, equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream as stream_gather  # noqa: E402
from tensorlbm.xval.case_spec import TG2DSpec  # noqa: E402

HERE = Path(__file__).resolve().parent
REF_DIR = HERE / "oss_ref"
OUT_DIR = ROOT / "runs" / "xval_conv_as_streaming"
DOC_PATH = ROOT / "docs" / "XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL.md"
ANCHOR = "XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL"


# --------------------------------------------------------------- OSS tables
def load_oss(name: str) -> dict:
    path = REF_DIR / f"{name}.json"
    if not path.exists():
        raise SystemExit(
            f"missing {path}; run extract_oss_tables.py --download (or --src <checkout>)"
        )
    return json.loads(path.read_text())


def oss_provenance() -> dict:
    path = REF_DIR / "provenance.json"
    return json.loads(path.read_text()) if path.exists() else {}


# ---------------------------------------------------------------- operators
def make_transport(arm: str, oss: dict | None = None):
    """Return ``f -> streamed f`` for one transport arm (identical signature)."""
    if arm == "A_native_gather":
        return stream_gather
    if arm == "B_conv_fixed_kernel":
        return SC.stream_conv_2d
    if arm == "B2_conv_upstream_table":
        # verbatim upstream path: dense cross-channel kernel from the shipped
        # STREAM table + channel-last fields + cat-based mobius padding
        table = np.asarray(oss["stream"], dtype=np.float64)

        def transport(f: torch.Tensor) -> torch.Tensor:
            # upstream pad_mobius == one wrapped ghost row/column per side
            x = f.unsqueeze(0)  # [1, q, ny, nx]
            x = torch.cat([x[:, :, -1:], x, x[:, :, :1]], dim=2)
            x = torch.cat([x[:, :, :, -1:], x, x[:, :, :, :1]], dim=3)
            # TF kernel layout [kH, kW, in, out] -> torch [out, in, kH, kW]
            weight = torch.as_tensor(table, dtype=f.dtype, device=f.device).permute(3, 2, 0, 1)
            return torch.nn.functional.conv2d(x, weight).squeeze(0)

        return transport
    raise ValueError(f"unknown arm {arm}")


def f_digest(f: torch.Tensor) -> str:
    return hashlib.sha256(np.ascontiguousarray(f.detach().numpy()).tobytes()).hexdigest()


# ------------------------------------------------------- 2D Taylor-Green case
def run_tg(spec: TG2DSpec, arm: str, oss: dict, threads: int = 4) -> dict:
    torch.set_num_threads(max(1, threads))
    transport = make_transport(arm, oss)
    n, dt = spec.n, torch.float64
    jj, ii = torch.meshgrid(
        torch.arange(n, dtype=dt), torch.arange(n, dtype=dt), indexing="ij"
    )
    x, y = ii + 0.5, jj + 0.5
    ux0 = -spec.u0 * torch.cos(spec.k * x) * torch.sin(spec.k * y)
    uy0 = spec.u0 * torch.sin(spec.k * x) * torch.cos(spec.k * y)
    f = equilibrium(torch.ones_like(ux0), ux0, uy0)
    tau = spec.tau

    samples: list[tuple[int, float, float]] = []
    wall0 = time.perf_counter()
    with torch.no_grad():
        for step in range(spec.steps + 1):
            if step % spec.sample_every == 0 or step == spec.steps:
                _, uxc, uyc = macroscopic(f)
                samples.append(
                    (
                        step,
                        0.5 * float((uxc * uxc + uyc * uyc).mean()),
                        float(torch.sqrt(uxc * uxc + uyc * uyc).max()),
                    )
                )
            if step == spec.steps:
                break
            f = collide_bgk(transport(f), tau)
    wall = time.perf_counter() - wall0

    _, ux, uy = macroscopic(f)
    ex, ey = exact_tg_velocity(spec, spec.steps)
    num = np.concatenate([ux.numpy().ravel(), uy.numpy().ravel()])
    ref = np.concatenate([ex.ravel(), ey.ravel()])
    field_l2 = 100.0 * float(np.linalg.norm(num - ref) / np.linalg.norm(ref))
    return {
        "arm": arm,
        "n": n,
        "re": spec.re,
        "u0": spec.u0,
        "nu": spec.nu,
        "tau": tau,
        "mach": spec.mach,
        "steps": spec.steps,
        "samples": samples,
        "gamma_K_fit": fit_decay_rate(samples, spec.steps),
        "gamma_K_theory": spec.gamma_k_theory,
        "gamma_vel_fit": fit_decay_rate(samples, spec.steps, index=2),
        "gamma_vel_theory": spec.gamma_vel_theory,
        "field_l2_pct": field_l2,
        "umax_final": float(np.sqrt(ux.numpy() ** 2 + uy.numpy() ** 2).max()),
        "finite": bool(torch.isfinite(f).all()),
        "f_sha256": f_digest(f),
        "wall_s": wall,
        "_ux": ux.numpy(),
        "_uy": uy.numpy(),
    }


def exact_tg_velocity(spec: TG2DSpec, t: float) -> tuple[np.ndarray, np.ndarray]:
    n = spec.n
    jj, ii = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    x, y = ii + 0.5, jj + 0.5
    amp = spec.u0 * math.exp(-spec.gamma_vel_theory * t)
    return (
        -amp * np.cos(spec.k * x) * np.sin(spec.k * y),
        amp * np.sin(spec.k * x) * np.cos(spec.k * y),
    )


def fit_decay_rate(samples: list[tuple[int, float, float]], steps: int, index: int = 1) -> float:
    """Least-squares slope of ln(quantity) over the second half of the run."""
    pts = [(s[0], s[index]) for s in samples if s[0] >= steps // 2 and s[index] > 0.0]
    if len(pts) < 3:
        return float("nan")
    t = np.array([p[0] for p in pts], dtype=float)
    ln = np.log(np.array([p[1] for p in pts], dtype=float))
    slope = np.polyfit(t, ln, 1)[0]
    return float(-slope)


# --------------------------------------------------- 2D plane Poiseuille case
def guo_term(f: torch.Tensor, force: tuple[float, float], tau: float) -> torch.Tensor:
    """Body-force source term added to the post-collision state (both arms).

    Plain momentum injection ``dF_q = w_q * 3 * (c_q . F)``: summing ``dF_q *
    c_q`` gives exactly ``F`` per step, which is what the steady Hagen-Poiseuille
    balance ``g H^2 / (8 nu)`` assumes.

    Trap recorded by this benchmark: the textbook Guo prefactor
    ``(1 - 1/(2 tau))`` only pairs with the *shifted-equilibrium* collision
    (``f_eq(rho, u + F/(2 rho))``).  Added to the unmodified library BGK
    collision it under-injects momentum by exactly ``1 - 1/(2 tau)`` and the
    steady channel comes out at ``0.375 x`` the analytic peak at tau = 0.8
    (measured: 62 % profile error).  Keeping the library collision untouched
    and using the plain source gives 0.16 % at this configuration.
    """
    rho, ux, uy = macroscopic(f)
    c = C2D.to(dtype=f.dtype)
    cf = c[:, 0].view(9, 1, 1) * force[0] + c[:, 1].view(9, 1, 1) * force[1]
    w = W.to(dtype=f.dtype).view(9, 1, 1)
    return 3.0 * w * cf


def velocity_with_force(f: torch.Tensor, force: tuple[float, float]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rho, ux, uy = macroscopic(f)
    return rho, ux + 0.5 * force[0] / rho, uy + 0.5 * force[1] / rho


def run_poiseuille(
    ny: int, nx: int, tau: float, umax_target: float, steps: int, arm: str, oss: dict
) -> dict:
    dt = torch.float64
    torch.set_num_threads(max(1, 4))
    transport = make_transport(arm if arm == "A_native_gather" else "B_conv_fixed_kernel", oss)
    nu = (tau - 0.5) / 3.0
    height = float(ny - 2)  # half-way walls at y = 0.5 and y = ny - 1.5
    g = 8.0 * nu * umax_target / height**2
    force = (g, 0.0)
    mask = torch.zeros(ny, nx, dtype=dt)
    mask[0, :] = 1.0
    mask[-1, :] = 1.0
    opposite = OPPOSITE.to(torch.int64)

    f = equilibrium(
        torch.ones(ny, nx, dtype=dt), torch.zeros(ny, nx, dtype=dt), torch.zeros(ny, nx, dtype=dt)
    )
    wall0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            f_streamed = transport(f)
            f_collided = collide_bgk(f_streamed, tau) + guo_term(f_streamed, force, tau)
            if arm == "A_native_gather":
                # explicit direction reversal at solid nodes (index form)
                f_new = f_collided.clone()
                f_new[:, mask > 0] = f_streamed[opposite][:, mask > 0]
            else:
                # upstream CollideSC: 1x1 cross-channel convolution + mask blend
                f_new = SC.bounce_blend(f_collided, f_streamed, mask, opposite=opposite)
            f = f_new
    wall = time.perf_counter() - wall0

    rho, ux, uy = velocity_with_force(f, force)
    prof = ux.numpy()[1:-1, nx // 2]
    y_index = np.arange(1, ny - 1)
    y_pos = y_index - 0.5  # distance from the lower wall
    exact = g / (2.0 * nu) * y_pos * (height - y_pos)
    profile_l2 = 100.0 * float(np.linalg.norm(prof - exact) / np.linalg.norm(exact))
    peak_err = 100.0 * float(abs(prof.max() - exact.max()) / exact.max())
    return {
        "arm": arm,
        "ny": ny,
        "nx": nx,
        "tau": tau,
        "nu": nu,
        "height": height,
        "g": g,
        "umax_target": umax_target,
        "umax_analytic": g * height**2 / (8.0 * nu),
        "umax_got": float(prof.max()),
        "re_channel": umax_target * height / nu,
        "steps": steps,
        "profile_l2_pct": profile_l2,
        "peak_rel_err_pct": peak_err,
        "profile": prof.tolist(),
        "exact": exact.tolist(),
        "finite": bool(torch.isfinite(f).all()),
        "f_sha256": f_digest(f),
        "wall_s": wall,
    }


# ------------------------------------------------------- equivalence evidence
def equivalence_checks(oss: dict) -> dict:
    checks: dict = {}
    torch.manual_seed(20261113)
    # 1. 2D bit-wise: gather transport vs fixed-kernel transport
    f = torch.rand(9, 20, 24, dtype=torch.float64)
    a, b = stream_gather(f), SC.stream_conv_2d(f)
    checks["2d_gather_vs_conv"] = {
        "bitwise_equal": bool(torch.equal(a, b)),
        "max_abs_diff": float((a - b).abs().max()),
        "conserved": float(f.sum() - b.sum()),
    }

    # 2. 2D: upstream shipped STREAM table (verbatim) vs our depthwise kernel
    f = torch.rand(9, 16, 18, dtype=torch.float64)
    ours = SC.stream_conv_2d(f)
    theirs = make_transport("B2_conv_upstream_table", oss)(f)
    checks["2d_derived_vs_upstream_table"] = {
        "bitwise_equal": bool(torch.equal(ours, theirs)),
        "max_abs_diff": float((ours - theirs).abs().max()),
    }

    # 3. upstream table audits (the self-check upstream lacks)
    from tensorlbm.streaming_conv import describe_stream_table

    audit = {}
    for name, axes in (("D2Q9", "yx"), ("D3Q15", "zyx"), ("D3Q19", "zyx")):
        try:
            tab = load_oss(name)
        except SystemExit:
            continue
        rows = describe_stream_table(tab["lveloc"], np.asarray(tab["stream"]), axes)
        bad = [r for r in rows if not r["ok"]]
        audit[name] = {"q": len(rows), "n_mismatch": len(bad), "mismatches": bad}
    checks["upstream_table_audit"] = audit

    # 4. 3D: fixed-kernel conv vs the canonical D3Q19 roll traversal
    try:
        from tensorlbm.core.d3q19_stencil import roll_from_pull_source
        from tensorlbm.d3q19 import C as C3D

        f3 = torch.rand(19, 6, 7, 8, dtype=torch.float64)
        ref = torch.empty_like(f3)
        ref[0] = f3[0]
        for q in range(1, 19):
            ref[q] = roll_from_pull_source(f3[q], q)
        conv3 = SC.stream_conv_3d(f3)
        checks["3d_conv_vs_canonical_roll"] = {
            "bitwise_equal": bool(torch.equal(ref, conv3)),
            "max_abs_diff": float((ref - conv3).abs().max()),
            "q": int(C3D.shape[0]),
        }
    except Exception as exc:  # pragma: no cover - reported, not hidden
        checks["3d_conv_vs_canonical_roll"] = {"error": f"{type(exc).__name__}: {exc}"}

    # 5. bounce: 1x1 convolution reversal vs explicit index reversal
    mask = torch.zeros(12, 12, dtype=torch.float64)
    mask[0, :] = 1.0
    mask[-1, :] = 1.0
    pre = torch.rand(9, 12, 12, dtype=torch.float64)
    post = torch.rand(9, 12, 12, dtype=torch.float64)
    conv_b = SC.bounce_blend(post, pre, mask, opposite=OPPOSITE)
    ref_b = post.clone()
    ref_b[:, mask > 0] = pre[OPPOSITE][:, mask > 0]
    checks["bounce_conv_vs_index"] = {
        "bitwise_equal": bool(torch.equal(conv_b, ref_b)),
        "max_abs_diff": float((conv_b - ref_b).abs().max()),
    }
    return checks


def timing(oss: dict, n: int = 128, steps: int = 60, threads: int = 4) -> dict:
    torch.set_num_threads(max(1, threads))
    torch.manual_seed(7)
    f = torch.rand(9, n, n, dtype=torch.float64) * 0.01 + 0.1
    out = {}
    for arm in ("A_native_gather", "B_conv_fixed_kernel", "B2_conv_upstream_table"):
        transport = make_transport(arm, oss)
        for _ in range(5):
            transport(f)  # warm-up
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(steps):
                transport(f)
        out[arm] = (time.perf_counter() - t0) / steps
    return {"n": n, "steps": steps, "threads": threads, "seconds_per_step": out}


# ------------------------------------------------------------------- figures
def make_figures(result: dict, out: Path) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    figs: list[dict] = []
    tg = result["tg"]
    ref_label = tg["configs"][0]["label"]
    baseline = [c for c in tg["configs"] if c["label"] == ref_label]
    ref_case = baseline[0]
    spec = TG2DSpec(n=ref_case["n"], re=ref_case["re"], steps=ref_case["steps"])

    # 1. kinetic energy decay: both arms + closed form
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for case in baseline:
        s = np.array(case["samples"], dtype=float)
        ax.semilogy(s[:, 0], s[:, 1], "-", lw=2, label=f"{case['arm']} (KE)")
    t = np.linspace(0, spec.steps, 50)
    ax.semilogy(t, spec.k0_theory * np.exp(-spec.gamma_k_theory * t), "k--", lw=1.4,
                label="closed form $K_0 e^{-4\\nu k^2 t}$")
    ax.set_xlabel("step")
    ax.set_ylabel("K = 0.5 <u$^2$+v$^2$>")
    ax.set_title("2D Taylor-Green decay: transport arms vs exact solution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p = out / "fig1_tg_energy_decay.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append({"name": p.name, "caption": "Taylor-Green kinetic energy vs the closed-form decay for both transports (curves overlap)."})

    # 2. field snapshot + arm-to-arm deviation
    base_B = [c for c in baseline if c["arm"].startswith("B")][0]
    ux = np.array(base_B["field_ux"])
    ex, _ = exact_tg_velocity(spec, spec.steps)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    im = axes[0].imshow(ux, origin="lower", cmap="RdBu_r")
    axes[0].set_title(f"u$_x$ (conv transport), step {spec.steps}")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    dev = np.abs(np.array(base_B["field_ux"]) - np.array(baseline[0]["field_ux"]))
    im2 = axes[1].imshow(dev / np.abs(ux).max(), origin="lower", cmap="inferno")
    axes[1].set_title("|u$_x$(B) - u$_x$(A)| / max|u$_x$|")
    fig.colorbar(im2, ax=axes[1], fraction=0.046)
    p = out / "fig2_tg_field_and_arm_deviation.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append({"name": p.name, "caption": "Left: velocity field from the fixed-kernel transport. Right: pointwise difference to the gather transport (round-off only)."})

    # 3. sensitivity: decay-rate error per configuration
    labels = [c["label"] for c in tg["configs"] if c["arm"] == "A_native_gather"]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for k, arm in enumerate(("A_native_gather", "B_conv_fixed_kernel")):
        errs = [
            c["gamma_K_rel_err_pct"] for c in tg["configs"] if c["arm"] == arm
        ]
        ax.bar(x + (k - 0.5) * width, errs, width, label=arm)
    ax.axhline(3.0, color="g", ls=":", label="internal target 3%")
    ax.axhline(15.0, color="r", ls="--", label="owner gate 15%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("|$\\gamma_K$(fit) - $\\gamma_K$(theory)| / $\\gamma_K$(theory)  [%]")
    ax.set_title("sensitivity: resolution / Reynolds number")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    p = out / "fig3_tg_sensitivity.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append({"name": p.name, "caption": "Taylor-Green decay-rate error for both transports at each predeclared configuration."})

    # 4. Poiseuille profiles
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    styles = {"A_native_gather": "--", "B_conv_fixed_kernel": "-"}
    for case in result["poiseuille"]["cases"]:
        prof = np.array(case["profile"])
        ax.plot(prof, np.arange(1, case["ny"] - 1), styles.get(case["arm"], "-"), lw=1.4,
                label=f"{case['arm']} Ny={case['ny']}")
    ref = result["poiseuille"]["cases"][0]
    ax.plot(np.array(ref["exact"]), np.arange(1, ref["ny"] - 1), "k:", lw=2,
            label="exact parabola (Ny=%d)" % ref["ny"])
    ax.set_xlabel("u$_x$")
    ax.set_ylabel("y (lattice)")
    ax.set_title("plane Poiseuille: wall-bounded arm (1x1 bounce convolution)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p = out / "fig4_poiseuille_profile.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append({"name": p.name, "caption": "Steady channel profile for both transports against the closed-form parabola."})

    # 5. timing
    sec = result["timing"]["seconds_per_step"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    names = list(sec)
    ax.barh(names, [sec[k] * 1e3 for k in names])
    ax.set_xlabel("ms per streaming call (CPU, %d^2, float64)" % result["timing"]["n"])
    ax.set_title("transport cost (no GPU in this environment)")
    ax.grid(axis="x", alpha=0.3)
    p = out / "fig5_transport_cost.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append({"name": p.name, "caption": "Wall-clock cost of the three transport formulations on CPU."})
    return figs


# ------------------------------------------------------------------ plumbing
def config_list(quick: bool) -> list[dict]:
    if quick:
        return [
            {"label": "n32_Re100", "n": 32, "re": 100.0, "steps": 600, "field": True},
            {"label": "n32_Re50", "n": 32, "re": 50.0, "steps": 600, "field": False},
        ]
    return [
        {"label": "n64_Re100", "n": 64, "re": 100.0, "steps": 2400, "field": True},
        {"label": "n32_Re100", "n": 32, "re": 100.0, "steps": 2400, "field": False},
        {"label": "n64_Re50", "n": 64, "re": 50.0, "steps": 1200, "field": False},
    ]


def kpi_row(case: dict) -> dict:
    return {
        "config": case["label"],
        "arm": case["arm"],
        "n": case["n"],
        "Re": case["re"],
        "U0": case["u0"],
        "Ma": round(case["mach"], 5),
        "nu": round(case["nu"], 6),
        "tau": round(case["tau"], 6),
        "gamma_K_fit": case["gamma_K_fit"],
        "gamma_K_theory": case["gamma_K_theory"],
        "gamma_K_rel_err_pct": case["gamma_K_rel_err_pct"],
        "gamma_vel_rel_err_pct": case["gamma_vel_rel_err_pct"],
        "field_l2_pct": case["field_l2_pct"],
        "pass_15pct": case["pass_15pct"],
        "pass_3pct": case["pass_3pct"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="CI-sized configuration")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--no-doc", action="store_true", help="skip markdown rendering")
    args = ap.parse_args(argv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    predeclared = json.loads((HERE / "predeclared.json").read_text())
    oss = load_oss("D2Q9")

    t_start = time.time()
    checks = equivalence_checks(oss)

    configs: list[dict] = []
    for cfg in config_list(args.quick):
        spec = TG2DSpec(
            n=cfg["n"], re=cfg["re"], steps=cfg["steps"], sample_every=max(10, cfg["steps"] // 40)
        )
        for arm in ("A_native_gather", "B_conv_fixed_kernel"):
            case = run_tg(spec, arm, oss, threads=args.threads)
            case["label"] = cfg["label"]
            case["gamma_K_rel_err_pct"] = (
                100.0 * abs(case["gamma_K_fit"] - case["gamma_K_theory"]) / case["gamma_K_theory"]
            )
            case["gamma_vel_rel_err_pct"] = (
                100.0 * abs(case["gamma_vel_fit"] - case["gamma_vel_theory"]) / case["gamma_vel_theory"]
            )
            case["pass_15pct"] = bool(
                max(case["gamma_K_rel_err_pct"], case["gamma_vel_rel_err_pct"], case["field_l2_pct"])
                < predeclared["predeclared_thresholds"]["owner_acceptance_pct"]
            )
            case["pass_3pct"] = bool(
                max(case["gamma_K_rel_err_pct"], case["gamma_vel_rel_err_pct"], case["field_l2_pct"])
                < predeclared["predeclared_thresholds"]["internal_target_pct"]
            )
            case["field_ux"] = case["_ux"].tolist() if cfg["field"] else None
            case["field_uy"] = case["_uy"].tolist() if cfg["field"] else None
            case.pop("_ux", None)
            case.pop("_uy", None)
            configs.append(case)

    # arm-to-arm deviation per configuration
    pair: list[dict] = []
    for cfg in config_list(args.quick):
        arms = [c for c in configs if c["label"] == cfg["label"]]
        a = next(c for c in arms if c["arm"] == "A_native_gather")
        b = next(c for c in arms if c["arm"].startswith("B"))
        pair.append(
            {
                "config": cfg["label"],
                "f_sha256_A": a["f_sha256"],
                "f_sha256_B": b["f_sha256"],
                "identical_populations": a["f_sha256"] == b["f_sha256"],
                "gamma_K_abs_diff": abs(a["gamma_K_fit"] - b["gamma_K_fit"]),
                "umax_abs_diff": abs(a["umax_final"] - b["umax_final"]),
            }
        )

    # wall-bounded arm
    heights = [12] if args.quick else [16, 24]
    poiseuille = {"cases": [], "pair": []}
    for ny in heights:
        for arm in ("A_native_gather", "B_conv_fixed_kernel"):
            case = run_poiseuille(
                ny=ny,
                nx=max(16, ny),
                tau=0.8,
                umax_target=0.02,
                steps=12000 if args.quick else 30000,
                arm=arm,
                oss=oss,
            )
            poiseuille["cases"].append(case)
        cases = [c for c in poiseuille["cases"] if c["ny"] == ny]
        poiseuille["pair"].append(
            {
                "ny": ny,
                "identical_populations": cases[0]["f_sha256"] == cases[1]["f_sha256"],
                "profile_l2_pct_max": max(c["profile_l2_pct"] for c in cases),
                "peak_rel_err_pct_max": max(c["peak_rel_err_pct"] for c in cases),
            }
        )

    tim = timing(oss, n=64 if args.quick else 128, steps=20 if args.quick else 60, threads=args.threads)

    eq_rel = predeclared["predeclared_thresholds"]["implementation_equivalence_rel"]
    eq_ok = all(
        [
            checks["2d_gather_vs_conv"]["max_abs_diff"] < eq_rel
            or checks["2d_gather_vs_conv"]["bitwise_equal"],
            checks["2d_derived_vs_upstream_table"]["max_abs_diff"] < eq_rel
            or checks["2d_derived_vs_upstream_table"]["bitwise_equal"],
            checks["bounce_conv_vs_index"]["bitwise_equal"],
            checks.get("3d_conv_vs_canonical_roll", {}).get("bitwise_equal", False),
        ]
    )
    gate_pct = predeclared["predeclared_thresholds"]["owner_acceptance_pct"]
    physics_ok = all(c["pass_15pct"] for c in configs) and all(
        c["peak_rel_err_pct"] < gate_pct and c["profile_l2_pct"] < gate_pct
        for c in poiseuille["cases"]
    )
    target_ok = all(c["pass_3pct"] for c in configs) and all(
        c["peak_rel_err_pct"] < 3.0 and c["profile_l2_pct"] < 3.0 for c in poiseuille["cases"]
    )

    result = {
        "anchor": ANCHOR,
        "mission": 17680,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "quick_mode": bool(args.quick),
        "reference": {
            "repo": "loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow",
            "feature_slug": "conv-as-streaming-fixed-kernel",
            "oss_solutions_row": 36,
            "files_cited": ["LatFlow/D2Q9.py (STREAM/BOUNCE/LVELOC)", "LatFlow/Domain.py:99-101,178-185 (StreamSC/CollideSC)", "LatFlow/utils.py:4-22 (simple_conv, pad_mobius)"],
            "provenance": oss_provenance(),
            "runnable_here": False,
            "not_runnable_reason": "TensorFlow 1.x graph/session API + Python 2 `xrange`; formulation transplanted verbatim onto torch instead",
        },
        "predeclared": predeclared,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cuda": bool(torch.cuda.is_available()),
            "threads": args.threads,
            "wall_s": round(time.time() - t_start, 1),
        },
        "equivalence": checks,
        "tg": {
            "configs": configs,
            "pair": pair,
        },
        "poiseuille": poiseuille,
        "timing": tim,
        "verdict": {
            "implementation_equivalent": bool(eq_ok),
            "all_kpi_under_15pct": bool(physics_ok),
            "all_kpi_under_3pct": bool(target_ok),
            "verdict": "PASS" if (eq_ok and physics_ok) else "FAIL",
        },
    }
    figures = make_figures(result, out / "figures")
    result["figures"] = figures
    (out / "result.json").write_text(json.dumps(result, indent=1) + "\n")

    with open(out / "table.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(kpi_row(configs[0])))
        writer.writeheader()
        for case in configs:
            writer.writerow(kpi_row(case))

    if not args.no_doc:
        render_docs(result, DOC_PATH)

    print(json.dumps(result["verdict"], indent=1))
    print("artefacts ->", out)
    return 0 if result["verdict"]["verdict"] == "PASS" else 1


def render_docs(result: dict, path: Path) -> None:
    sys.path.insert(0, str(HERE))
    from render_report import build_markdown  # local module, keeps main() readable

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_markdown(result))


if __name__ == "__main__":
    raise SystemExit(main())