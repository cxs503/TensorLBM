#!/usr/bin/env python3
"""XVAL-LBM-GRAPH-UNROLL-BPTT-LES-PARAMETER - cross-validation driver.

Reference implementation
------------------------
``loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow``
(``LatFlow/Domain.py`` graph-unroll + BPTT LES parameter, ``les_train.py``).
Vendored verbatim under ``oss_src/``; the lattice tables are **executed** from
that file behind a 3-line TensorFlow shim (:mod:`oss_arm`), and the collision /
streaming / unroll operations are transcribed 1:1 with line references.

Arms (all thresholds pre-declared in ``predeclared.json``)
---------------------------------------------------------
A1 forward parity  TensorLBM ``les_bptt`` vs OSS arm  (rel L2 after 40 steps)
A2 BPTT twin recovery of ``sc_true``                  (TensorLBM reference data)
A3 BPTT cross-implementation recovery                 (OSS-arm reference data)
A4 autograd ``dL/dSc`` vs central finite difference
A5 Taylor-Green linear decay vs analytic ``exp(-2 nu k^2 t)``
A6 sensitivity: unroll length / ``sc_init`` / ``tau0``

Usage
-----
    python3 run_xval.py            # full run, writes artifacts/ + table.csv
    python3 run_xval.py --quick    # reduced iteration counts (smoke test)
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import oss_arm  # noqa: E402
from tensorlbm import les_bptt as L  # noqa: E402

torch.set_num_threads(1)  # tiny tensors: avoids thread thrash (40 steps ~0.2 s)

ART = HERE / "artifacts"
ART.mkdir(exist_ok=True)
PREDECL = json.loads((HERE / "predeclared.json").read_text())
PRECL_THRESH = PREDECL["predeclared_thresholds"]


def rel_err(got: float, ref: float) -> float:
    return abs(got - ref) / abs(ref)


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().double()
    b = b.detach().double()
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(b))


# --------------------------------------------------------------------------
# Case builders
# --------------------------------------------------------------------------
def tg_case(n: int, u0: float, k_mode: int) -> tuple[torch.Tensor, torch.Tensor]:
    return L.taylor_green_state(n=n, u0=u0, k=k_mode)


def tg_case_with_disc(n: int, u0: float, k_mode: int, radius: int):
    f0, _ = L.taylor_green_state(n=n, u0=u0, k=k_mode)
    yy, xx = np.mgrid[0:n, 0:n]
    c = n // 2
    inside = (((yy - c) ** 2 + (xx - c) ** 2) <= radius ** 2).astype(np.float32)
    boundary = torch.from_numpy(inside).view(1, 1, n, n).to(f0.dtype)
    # solid cells hold the rest state (upstream convention: boundary mask only,
    # the populations inside solids are reset by the bounce blend every step)
    return f0, boundary


def to_oss_layout(f_t: torch.Tensor) -> np.ndarray:
    """[1,9,H,W] (TensorLBM) -> [1,H,W,9] (upstream)."""
    return f_t.detach().numpy().transpose(0, 2, 3, 1).astype(np.float32)


def from_oss_vel(vel: np.ndarray) -> torch.Tensor:
    """[1,H,W,3] (upstream) -> [1,2,H,W] (TensorLBM)."""
    return torch.from_numpy(np.ascontiguousarray(vel[..., :2].transpose(0, 3, 1, 2))).double()


# --------------------------------------------------------------------------
# A1: forward parity, TensorLBM vs OSS arm
# --------------------------------------------------------------------------
def arm_a1(case: dict, boundary_kind: str) -> dict:
    n, u0, k_mode = case["n"], case["u0"], case["k_mode"]
    tau0, sc, steps = case["tau0"], case["sc_true"], case["steps_parity"]
    if boundary_kind == "none":
        f0_t, b_t = tg_case(n, u0, k_mode)
        f0_o = to_oss_layout(f0_t)
        b_o = np.zeros((1, n, n, 1), dtype=np.float32)
    else:
        f0_t, b_t = tg_case_with_disc(n, u0, k_mode, case["radius"])
        f0_o = to_oss_layout(f0_t)
        b_o = b_t.numpy().transpose(0, 2, 3, 1).astype(np.float32)

    t0 = time.time()
    hist_o = oss_arm.unroll_sc(f0_o, steps, tau0, sc, b_o, les=True)
    hist_t = L.unroll(f0_t, steps, tau0=tau0, sc=sc, boundary=b_t, les=True)
    dt = time.time() - t0

    per_step = []
    for s in (0, steps // 4, steps // 2, steps - 1):
        fo = hist_o[s]["f"][0]
        ft = hist_t[s].f[0].permute(1, 2, 0).numpy()
        vo = from_oss_vel(hist_o[s]["vel"])
        vt = hist_t[s].vel
        per_step.append(
            {
                "step": s,
                "f_rel_l2": float(np.linalg.norm(fo - ft) / np.linalg.norm(fo)),
                "f_max_abs": float(np.abs(fo - ft).max()),
                "vel_rel_l2": rel_l2(vt, vo),
            }
        )
    worst = max(p["vel_rel_l2"] for p in per_step)
    worst_f = max(p["f_rel_l2"] for p in per_step)
    thr = PRECL_THRESH["A1_forward_parity_rel_L2"]
    return {
        "arm": "A1",
        "boundary": boundary_kind,
        "steps": steps,
        "worst_vel_rel_l2": worst,
        "worst_f_rel_l2": worst_f,
        "threshold": thr,
        "pass": bool(worst < thr and worst_f < thr),
        "per_step": per_step,
        "wall_time_s": dt,
    }


# --------------------------------------------------------------------------
# A2/A3: BPTT recovery of sc from a reference history
# --------------------------------------------------------------------------
def arm_recovery(case: dict, source: str, sc_init: float, iters: int,
                 method: str = "gradient_root") -> dict:
    n, u0, k_mode = case["n"], case["u0"], case["k_mode"]
    tau0, sc_true, steps = case["tau0"], case["sc_true"], case["steps_calibration"]
    f0, b = tg_case(n, u0, k_mode)

    if source == "tensorlbm":
        ref = L.unroll(f0, steps, tau0=tau0, sc=sc_true, boundary=b)
        target = [s.vel.detach() for s in ref]
    elif source == "oss":
        b_o = np.zeros((1, n, n, 1), dtype=np.float32)
        hist_o = oss_arm.unroll_sc(to_oss_layout(f0), steps, tau0, sc_true, b_o)
        target = [from_oss_vel(s["vel"]) for s in hist_o]
    else:  # pragma: no cover
        raise ValueError(source)

    t0 = time.time()
    hist = L.calibrate_les_parameter(
        target, f0, tau0=tau0, sc_init=sc_init, boundary=b, iters=iters, lr=0.2,
        method=method,
    )
    dt = time.time() - t0
    err = rel_err(hist["sc"], sc_true)
    thr = PRECL_THRESH[
        "A2_twin_recovery_rel_error" if source == "tensorlbm" else "A3_oss_data_recovery_rel_error"
    ]
    return {
        "arm": ("A2" if method == "gradient_root" else "A2b_adam") if source == "tensorlbm" else "A3",
        "reference_source": source,
        "sc_true": sc_true,
        "sc_init": sc_init,
        "sc_got": hist["sc"],
        "rel_error": err,
        "loss_final": hist["loss"],
        "loss_initial": hist["loss_history"][0],
        "iterations": hist["iterations"],
        "threshold": thr,
        "pass": bool(err < thr),
        "history": {
            "sc": hist["sc_history"],
            "loss": hist["loss_history"],
            "grad": hist["grad_history"],
        },
        "wall_time_s": dt,
    }


# --------------------------------------------------------------------------
# A4: autograd vs central finite difference
# --------------------------------------------------------------------------
def arm_gradient(case: dict, sc_star: float = 0.17, eps_list=(1e-2, 1e-3)) -> dict:
    n, u0, k_mode = case["n"], case["u0"], case["k_mode"]
    tau0, steps = case["tau0"], case["steps_calibration"]
    f0, b = tg_case(n, u0, k_mode)
    target = [s.vel.detach() for s in L.unroll(f0, steps, tau0=tau0, sc=0.51, boundary=b)]

    sc = torch.tensor(sc_star, dtype=f0.dtype, requires_grad=True)
    loss = L.velocity_mse_loss(L.unroll(f0, steps, tau0=tau0, sc=sc, boundary=b), target)
    loss.backward()
    grad_ad = float(sc.grad)

    rows = []
    for eps in eps_list:
        lp = float(
            L.velocity_mse_loss(
                L.unroll(f0, steps, tau0=tau0, sc=torch.tensor(sc_star + eps, dtype=f0.dtype), boundary=b),
                target,
            )
        )
        lm = float(
            L.velocity_mse_loss(
                L.unroll(f0, steps, tau0=tau0, sc=torch.tensor(sc_star - eps, dtype=f0.dtype), boundary=b),
                target,
            )
        )
        fd = (lp - lm) / (2 * eps)
        rows.append({"eps": eps, "grad_fd": fd, "rel_error": rel_err(grad_ad, fd)})
    worst = max(r["rel_error"] for r in rows)
    thr = PRECL_THRESH["A4_gradient_rel_error"]
    return {
        "arm": "A4",
        "sc_star": sc_star,
        "loss": float(loss),
        "grad_autograd": grad_ad,
        "fd": rows,
        "worst_rel_error": worst,
        "threshold": thr,
        "pass": bool(worst < thr),
    }


# --------------------------------------------------------------------------
# A5: linear-regime Taylor-Green decay vs analytic
# --------------------------------------------------------------------------
def arm_analytic(case: dict, steps: int = 200) -> dict:
    """Linear-regime Taylor-Green decay vs the analytic laminar reference.

    Gate (predeclared): the ``les=False`` arm must be within 3% of
    ``u0 exp(-2 nu0 k^2 t)``.  The ``les=True`` arm is *reported* rather than
    gated: the analytic reference carries no sub-grid viscosity, so its
    deviation is the physics signal, and the implied eddy viscosity
    ``nu_eff = -ln(A/A0) / (2 k^2 t)`` is derived from the same run.
    """
    n, u0, k_mode = case["n"], case["u0"], case["k_mode"]
    tau0 = case["tau0"]
    nu0 = (tau0 - 0.5) / 3.0
    k_phys = 2.0 * math.pi * k_mode / n
    f0, b = tg_case(n, u0, k_mode)
    thr = PRECL_THRESH["A5_analytic_tg_decay_rel_error"]

    rows = []
    for les in (False, True):
        hist = L.unroll(f0, steps, tau0=tau0, sc=case["sc_true"], boundary=b, les=les)
        amp = float(L.tg_amplitude(hist[-1].f, k=k_mode)[0])
        ref = L.analytic_taylor_green_decay(u0, nu0, k_phys, steps)
        err = rel_err(amp, ref)
        nu_eff = -math.log(max(amp, 1e-300) / u0) / (2.0 * k_phys * k_phys * steps)
        rows.append(
            {
                "les": les,
                "amplitude_got": amp,
                "amplitude_ref": ref,
                "rel_error": err,
                "nu_eff": nu_eff,
                "nu0": nu0,
                "pass": bool(err < thr) if not les else None,
            }
        )
    les_faster = rows[1]["amplitude_got"] < rows[0]["amplitude_got"]
    return {
        "arm": "A5",
        "steps": steps,
        "nu0": nu0,
        "k_phys": k_phys,
        "rows": rows,
        "les_decays_faster": bool(les_faster),
        "threshold": thr,
        "gate_applies_to": "les=False (analytic reference is laminar); les=True is reported",
        "pass": bool(rows[0]["pass"] and les_faster),
        "note": "the les=True deviation is the sub-grid viscosity contribution, quantified by nu_eff",
    }


# --------------------------------------------------------------------------
# A6: sensitivity / convergence
# --------------------------------------------------------------------------
def arm_sensitivity(case: dict, iters: int) -> dict:
    rows = {"unroll_length": [], "sc_init": [], "tau0": []}
    thr = PRECL_THRESH["A6_sensitivity_max_rel_error"]
    for steps in (10, 20, 40):
        c = dict(case)
        c["steps_calibration"] = steps
        r = arm_recovery(c, "tensorlbm", 0.05, iters)
        rows["unroll_length"].append(
            {"steps": steps, "sc_got": r["sc_got"], "rel_error": r["rel_error"], "pass": r["pass"]}
        )
    for sc_init in (0.02, 0.05, 0.10, 0.30):
        r = arm_recovery(case, "tensorlbm", sc_init, iters)
        rows["sc_init"].append(
            {"sc_init": sc_init, "sc_got": r["sc_got"], "rel_error": r["rel_error"], "pass": r["pass"]}
        )
    for tau0 in (0.51, 0.56, 0.71):
        c = dict(case)
        c["tau0"] = tau0
        r = arm_recovery(c, "tensorlbm", 0.05, iters)
        rows["tau0"].append(
            {"tau0": tau0, "sc_got": r["sc_got"], "rel_error": r["rel_error"], "pass": r["pass"]}
        )
    worst = max(v["rel_error"] for group in rows.values() for v in group)
    return {
        "arm": "A6",
        "groups": rows,
        "worst_rel_error": worst,
        "threshold": thr,
        "pass": bool(worst < thr),
    }


# --------------------------------------------------------------------------
def env_probe() -> dict:
    import platform
    import subprocess

    probe = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "threads": torch.get_num_threads(),
        "tensorflow": None,
    }
    try:  # documented as NOT_RUN in predeclared.json
        import tensorflow as tf  # noqa: F401

        probe["tensorflow"] = tf.__version__
    except Exception as exc:  # pragma: no cover
        probe["tensorflow"] = f"unavailable: {type(exc).__name__}: {exc}"
    try:
        probe["pip_tensorflow_cpu"] = subprocess.run(
            ["pip", "download", "--no-deps", "-d", "/tmp/xval_pip_probe", "tensorflow-cpu"],
            capture_output=True, text=True, timeout=60,
        ).stdout.splitlines()[-1]
    except Exception as exc:  # pragma: no cover
        probe["pip_tensorflow_cpu"] = f"probe failed: {exc}"
    return probe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="reduced iterations (smoke test)")
    ap.add_argument("--out", default=str(ART / "results.json"))
    args = ap.parse_args()

    iters = 30 if args.quick else 200
    case = dict(PREDECL["cases"]["tg_k2_u02"])
    disc = dict(PREDECL["cases"]["disc_k2_u02"])

    results = {
        "anchor": PREDECL["anchor"],
        "mission": PREDECL["mission"],
        "repair_of": PREDECL["repair_of"],
        "quick": args.quick,
        "iterations_budget": iters,
        "env": env_probe(),
        "source_of_record": PREDECL["source_of_record"],
        "oss_table_sha256": oss_arm.TABLES["sha256"],
        "oss_tables": {
            "weights_sum": float(oss_arm.W.sum()),
            "EEK": oss_arm._EEK.tolist(),
            "stream_nonzeros": int((oss_arm.STREAM != 0).sum()),
        },
    }

    t0 = time.time()
    results["A1_tg"] = arm_a1(case, "none")
    results["A1_disc"] = arm_a1(disc, "disc")
    results["A2"] = arm_recovery(case, "tensorlbm", 0.05, iters)
    # short Adam/SGD trace, kept as evidence that plain gradient descent also
    # contracts on the same unrolled objective (not only the root solver)
    results["A2b_adam"] = arm_recovery(case, "tensorlbm", 0.05, 40, method="adam")
    results["A3"] = arm_recovery(case, "oss", 0.05, iters)
    results["A4"] = arm_gradient(case)
    results["A5"] = arm_analytic(case)
    if not args.quick:
        results["A6"] = arm_sensitivity(case, iters=100)
    results["wall_time_total_s"] = time.time() - t0

    # verdict
    checks = {k: results[k]["pass"] for k in ("A1_tg", "A1_disc", "A2", "A2b_adam", "A3", "A4", "A5")}
    if "A6" in results:
        checks["A6"] = results["A6"]["pass"]
    results["verdict"] = {
        "all_pass": bool(all(checks.values())),
        "checks": checks,
        "owner_precision_standard_15pct": "applied to A2/A3/A4/A6; A1 uses a tighter 1e-4 parity gate",
    }

    Path(args.out).write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps({k: results[k] for k in ("A1_tg", "A1_disc")}, indent=2, default=float)[:1500])
    for k in ("A2", "A3", "A4", "A5", "A6"):
        if k in results:
            r = results[k]
            keep = {kk: vv for kk, vv in r.items() if kk not in ("history", "groups", "per_step")}
            print(k, json.dumps(keep, default=float))
    print("verdict", results["verdict"])
    print("wrote", args.out)
    return 0 if results["verdict"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())