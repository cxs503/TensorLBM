"""TensorLBM arm of XVAL-LBM-SCMP-TWO-COMPONENT-PLANAR-INTERFACE.

Replicates the ``nfluids = 2`` branch of the OSS reference
(``lbm-principles-practice/code`` chapter9/shanchen.cpp) with the in-repo
TensorLBM kernels:

* initial condition: sharp planar step, ``rho_A = 0.1 / rho_B = 0.9`` for
  ``y < ny/2`` and the reverse above, ``f = feq(rho, 0)`` (identical to OSS
  ``initialisation()``);
* force: ``F_s = -g_ss psi_s sum_i w_i c_i psi_ss(x+c_i)`` with
  ``psi = rho0 (1 - exp(-rho/rho0))``.  TensorLBM samples the backward
  neighbours, so the exact parameter mapping is ``G = -g`` (see the note in
  :func:`tensorlbm.multiphase.sc_two_component_force`);
* collision: BGK + Guo (2002) forcing, equilibrium on the mass-averaged
  (mixture) force-corrected velocity, i.e. the OSS ``computeVelocity``/``push``
  pair;
* periodic streaming in both directions.

Two variants are provided so that a difference between the codes can be
attributed:

``faithful``
    the OSS algorithm assembled statement-by-statement from TensorLBM's public
    kernels (``sc_two_component_force`` + ``equilibrium`` + ``guo_force_term`` +
    ``stream``): the Guo source term is evaluated with the *mixture* velocity,
    exactly as OSS does.
``native``
    the same physics through ``collide_sc_two_component(u_eq="mixture",
    force_scheme="guo")``, i.e. the idiom a TensorLBM user would write.  It
    differs from ``faithful`` only in the velocity fed to the Guo source term
    (per-component corrected velocity instead of the mixture velocity).
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# small grids: one BLAS/OpenMP thread is ~12x faster than the default
# thread fan-out on a 64x64 tensor (measured 13.8 ms/step vs 175 ms/step)
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.multiphase import (  # noqa: E402
    collide_sc_two_component,
    guo_force_term,
    psi_exp,
    psi_linear,
    sc_two_component_force,
)
from tensorlbm.solver import stream  # noqa: E402

PSI = {"exp": psi_exp, "linear": psi_linear}


def init_sharp(n: int, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """OSS ``initialisation()`` for ``nfluids = 2``: sharp planar step, rest fluid."""
    y = torch.arange(n, dtype=dtype).view(n, 1).expand(n, n)
    hi = y < n // 2  # OSS:  if (y < ny/2) -> (rho_A, rho_B) = (0.1, 0.9)
    one = torch.ones((n, n), dtype=dtype)
    rho1 = torch.where(hi, 0.1 * one, 0.9 * one)
    rho2 = torch.where(hi, 0.9 * one, 0.1 * one)
    zero = torch.zeros((n, n), dtype=dtype)
    return equilibrium(rho1, zero, zero), equilibrium(rho2, zero, zero)


def corrected_mixture_velocity(
    f1: torch.Tensor, f2: torch.Tensor, Fx1, Fy1, Fx2, Fy2
) -> tuple[torch.Tensor, torch.Tensor]:
    """``u = (sum_s [sum_i f_i c_i + F_s/2]) / sum_s rho_s`` (OSS computeVelocity)."""
    rho1, ux1, uy1 = macroscopic(f1)
    rho2, ux2, uy2 = macroscopic(f2)
    r1 = rho1.clamp(min=1e-12)
    r2 = rho2.clamp(min=1e-12)
    rt = r1 + r2
    ux = (rho1 * ux1 + 0.5 * Fx1 + rho2 * ux2 + 0.5 * Fx2) / rt
    uy = (rho1 * uy1 + 0.5 * Fy1 + rho2 * uy2 + 0.5 * Fy2) / rt
    return ux, uy


def run_arm(
    n: int = 64,
    g_ab: float = 6.0,
    g_self: tuple[float, float] = (0.0, 0.0),
    psi: str = "exp",
    steps: int = 20000,
    variant: str = "faithful",
    tau: float = 1.0,
    dtype: torch.dtype = torch.float32,
    sample_every: int = 0,
    label: str = "",
) -> dict[str, Any]:
    """Run one TensorLBM arm; returns the final fields and diagnostics."""
    if variant not in ("faithful", "native"):
        raise ValueError(f"variant must be 'faithful' or 'native', got {variant!r}")
    psi_fn = PSI[psi]
    G = -g_ab                      # exact OSS -> TensorLBM mapping (G = -g)
    gs = (-float(g_self[0]), -float(g_self[1]))  # same mapping for the self terms
    f1, f2 = init_sharp(n, dtype)
    hist: list[dict[str, float]] = []
    t0 = time.time()
    finite = True
    Fx1 = Fy1 = Fx2 = Fy2 = None
    step = 0
    for step in range(1, steps + 1):
        if variant == "native":
            rho1, ux1, uy1 = macroscopic(f1)
            rho2, ux2, uy2 = macroscopic(f2)
            Fx1, Fy1, Fx2, Fy2 = sc_two_component_force(
                rho1, rho2, G, psi_fn=psi_fn, g_self=gs
            )
            f1, f2 = collide_sc_two_component(
                f1, f2, G_12=G, tau1=tau, tau2=tau, u_eq="mixture",
                force_scheme="guo", psi_fn=psi_fn, g_self=gs,
            )
        else:
            rho1, ux1, uy1 = macroscopic(f1)
            rho2, ux2, uy2 = macroscopic(f2)
            Fx1, Fy1, Fx2, Fy2 = sc_two_component_force(
                rho1, rho2, G, psi_fn=psi_fn, g_self=gs
            )
            r1 = rho1.clamp(min=1e-12)
            r2 = rho2.clamp(min=1e-12)
            c1x, c1y = ux1 + 0.5 * Fx1 / r1, uy1 + 0.5 * Fy1 / r1
            c2x, c2y = ux2 + 0.5 * Fx2 / r2, uy2 + 0.5 * Fy2 / r2
            rt = r1 + r2
            umx = (r1 * c1x + r2 * c2x) / rt
            umy = (r1 * c1y + r2 * c2y) / rt
            fe1 = equilibrium(rho1, umx, umy)
            fe2 = equilibrium(rho2, umx, umy)
            f1 = f1 - (f1 - fe1) / tau + guo_force_term(rho1, umx, umy, Fx1, Fy1, tau)
            f2 = f2 - (f2 - fe2) / tau + guo_force_term(rho2, umx, umy, Fx2, Fy2, tau)
        f1, f2 = stream(f1), stream(f2)
        if sample_every and step % sample_every == 0:
            rho1, _, _ = macroscopic(f1)
            finite = bool(torch.isfinite(f1).all() and torch.isfinite(rho1).all())
            hist.append({
                "step": float(step),
                "rho1_min": float(rho1.min()),
                "rho1_max": float(rho1.max()),
                "finite": float(finite),
            })
            if not finite:
                break
    wall = time.time() - t0
    rho1, ux1, uy1 = macroscopic(f1)
    rho2, ux2, uy2 = macroscopic(f2)
    finite = bool(torch.isfinite(f1).all() and torch.isfinite(f2).all())
    if finite:
        Fx1, Fy1, Fx2, Fy2 = sc_two_component_force(rho1, rho2, G, psi_fn=psi_fn, g_self=gs)
        umx, umy = corrected_mixture_velocity(f1, f2, Fx1, Fy1, Fx2, Fy2)
    else:
        umx = torch.full_like(rho1, float("nan"))
        umy = torch.full_like(rho1, float("nan"))
    return {
        "label": label,
        "kind": "tensorlbm",
        "params": {
            "n": n, "g_ab": g_ab, "g_self": list(g_self), "psi": psi,
            "steps": steps, "variant": variant, "tau": tau,
            "dtype": str(dtype).replace("torch.", ""), "G_12": G, "g_self_lib": list(gs),
        },
        "steps_run": step,
        "finite": finite,
        "wall_seconds": wall,
        "history": hist,
        "fields": {
            "density_comp0": rho1.cpu().numpy().astype(float),
            "density_comp1": rho2.cpu().numpy().astype(float),
            "velx": umx.cpu().numpy().astype(float),
            "vely": umy.cpu().numpy().astype(float),
        },
    }


def save(res: dict[str, Any], outdir: Path) -> Path:
    import json

    outdir.mkdir(parents=True, exist_ok=True)
    payload = dict(res)
    fields = payload.pop("fields", {})
    payload["fields_shape"] = {k: list(v.shape) for k, v in fields.items()}
    payload["fields"] = {k: np.asarray(v).tolist() for k, v in fields.items()}
    path = outdir / f"tl_{res['label']}.json"
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)
    return path


def load(outdir: Path, label: str) -> dict[str, Any]:
    import json

    with open(outdir / f"tl_{label}.json") as fh:
        res = json.load(fh)
    res["fields"] = {k: np.asarray(v, dtype=float) for k, v in res["fields"].items()}
    return res


if __name__ == "__main__":  # smoke: 200 steps at 32^2
    r = run_arm(n=32, steps=200, label="smoke")
    rho = r["fields"]["density_comp0"]
    print(f"steps={r['steps_run']} finite={r['finite']} {r['wall_seconds']:.2f}s "
          f"rhoA[{rho.min():.3f},{rho.max():.3f}] u_max={math.hypot(*[float(np.abs(r['fields'][k]).max()) for k in ('velx','vely')]):.4f}")