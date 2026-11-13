#!/usr/bin/env python3
"""Board-audit probes for m18289 delivery (XVAL-LBM-MCMP-CROSS-FORCE-OPERATOR).

Purpose: bind every claim in /root/tp-agent/work/m18289/xval/mcmp_cross_force
(preserved byte-identically under refs/, sha256 in ../MANIFEST.sha256) to a
number produced by executing the delivered code against the repo kernels in
this worktree (src/tensorlbm).

Probes
  P1  delivered package imports (sha256-pinned copy, not the live work tree)
  P2  repo_cross_force == tensorlbm.multiphase.sc_two_component_force (identity)
  P3  OSS forward-gather operator vs repo backward-gather kernel, linear psi:
      max relative difference over all four force components
  P4  stencil coefficient: forward gather of a linear field reproduces
      cs^2 * k (theory 1/3) and equals minus the backward gather
  P5  uniform field -> exactly zero cross force (both operators)
  P6  psi_mode='exp' composition path (repo gather on OSS pseudopotential)
      vs the delivered OSS operator, g = -G
  P7  stream_periodic: mass conserving, and equals a hand-written pull shift
  P8  pseudopotential conventions psi_exp(rho) vs analytic 1-exp(-rho)

Output: probe_delivery.json (all numbers machine-written; nothing hand-typed).
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
REF = HERE / "refs"
REPO = HERE.parent.parent          # worktree root
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.d2q9 import C, W                      # noqa: E402
from tensorlbm.multiphase import sc_two_component_force  # noqa: E402

DT = torch.float64
torch.manual_seed(20261113)


def load_delivery():
    """Import the sha256-pinned copy of the delivered operators module.

    The delivered ``__init__.py`` is a bare docstring (no re-exports), so the
    code under test is ``operators.py``; it is loaded as a standalone module
    exactly as a user following the package docstring would.
    """
    spec = importlib.util.spec_from_file_location(
        "delivery_operators", REF / "operators.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["delivery_operators"] = mod
    spec.loader.exec_module(mod)
    return mod


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """max|a-b| / max|a| (a = reference magnitude)."""
    den = float(a.abs().max())
    if den == 0.0:
        return float((a - b).abs().max())
    return float((a - b).abs().max()) / den


def smooth_field(ny=32, nx=32):
    """Deterministic smooth two-phase-ish field, distinct per component."""
    y = torch.arange(ny, dtype=DT).view(-1, 1) / ny * 2 * torch.pi
    x = torch.arange(nx, dtype=DT).view(1, -1) / nx * 2 * torch.pi
    r1 = 1.0 + 0.30 * torch.sin(x) * torch.cos(0.7 * y) + 0.10 * torch.cos(3 * x + y)
    r2 = 0.8 + 0.25 * torch.cos(0.5 * x + 1.3 * y) + 0.07 * torch.sin(2 * y - x)
    return r1.contiguous(), r2.contiguous()


def main():
    out = {"delivery_files": sorted(p.name for p in REF.glob("*.py"))}
    m = load_delivery()
    init_txt0 = (REF / "__init__.py").read_text()
    def _const(name):
        for line in init_txt0.splitlines():
            if line.strip().startswith(name + " ="):
                return line.split("=", 1)[1].strip().strip('"')
        return None
    out["P1_import"] = {"ok": True, "module": "operators.py",
                        "package_version": _const("__version__"),
                        "anchor": _const("ANCHOR"),
                        "criterion_pct": _const("CRITERION_PCT")}

    r1, r2 = smooth_field()
    G = -6.0                      # repo convention: negative = repulsive
    g = 6.0                       # OSS convention: positive = repulsive

    # P2 -- the delivered wrapper must be the library kernel, not a re-implementation
    ref = sc_two_component_force(r1, r2, G)
    got = m.repo_cross_force(r1, r2, G)
    out["P2_wrapper_identity"] = {
        "components": len(got),
        "max_abs_diff": max(float((a - b).abs().max()) for a, b in zip(ref, got)),
        "bitwise_equal": all(torch.equal(a, b) for a, b in zip(ref, got)),
    }

    # P3 -- OSS forward gather vs repo backward gather (linear psi, symmetric stencil)
    oss = m.oss_cross_force(r1, r2, g, psi_mode="linear")
    pairs = {"Fx1": (ref[0], oss[0]), "Fy1": (ref[1], oss[1]),
             "Fx2": (ref[2], oss[2]), "Fy2": (ref[3], oss[3])}
    out["P3_operator_identity_linear"] = {
        k: {"max_abs_diff": float((a - b).abs().max()), "rel": rel(a, b)}
        for k, (a, b) in pairs.items()}
    out["P3_operator_identity_linear"]["worst_rel"] = max(
        out["P3_operator_identity_linear"][k]["rel"] for k in pairs)

    # P4 -- stencil coefficient and forward/backward antisymmetry
    ny, nx = 16, 16
    yy = torch.arange(ny, dtype=DT).view(-1, 1).expand(ny, nx)
    xx = torch.arange(nx, dtype=DT).view(1, -1).expand(ny, nx)
    kx, ky = 0.37, 0.21
    psi = kx * xx + ky * yy
    c = C.to(torch.long)
    w = W.to(DT)
    fwd = torch.zeros_like(psi)
    bwd = torch.zeros_like(psi)
    for i in range(9):
        cx, cy = int(c[i, 0]), int(c[i, 1])
        if cx == 0 and cy == 0:
            continue
        fwd = fwd + w[i] * cx * torch.roll(psi, shifts=(-cy, -cx), dims=(0, 1))
        bwd = bwd + w[i] * cx * torch.roll(psi, shifts=(cy, cx), dims=(0, 1))
    interior = (slice(1, -1), slice(2, -2))
    out["P4_stencil"] = {
        "forward_over_kx": float(fwd[interior].mean()) / kx,
        "backward_over_kx": float(bwd[interior].mean()) / kx,
        "cs2_theory": 1.0 / 3.0,
        "forward_plus_backward_max_abs": float((fwd + bwd).abs().max()),
    }

    # P5 -- uniform field: cross force must vanish identically
    u1 = torch.full((8, 8), 1.3, dtype=DT)
    u2 = torch.full((8, 8), 0.7, dtype=DT)
    ru = sc_two_component_force(u1, u2, G)
    ou = m.oss_cross_force(u1, u2, g, psi_mode="linear")
    out["P5_uniform_zero"] = {
        "repo_max_abs": max(float(t.abs().max()) for t in ru),
        "oss_max_abs": max(float(t.abs().max()) for t in ou),
    }

    # P6 -- exp pseudopotential composition path
    re_ = m.repo_cross_force_psi(r1, r2, G, psi_mode="exp")
    oe = m.oss_cross_force(r1, r2, g, psi_mode="exp")
    out["P6_exp_psi_identity"] = {
        "worst_rel": max(rel(a, b) for a, b in zip(re_, oe)),
        "psi_exp_at_rho1": float(m.psi_exp(torch.tensor(1.0, dtype=DT))),
        "analytic_1_minus_exp": float(1.0 - torch.exp(torch.tensor(-1.0, dtype=DT))),
    }

    # P7 -- streaming kernel
    f = torch.rand(9, 12, 10, dtype=DT)
    fs = m.stream_periodic(f)
    hand = torch.stack([torch.roll(f[i], shifts=(-int(c[i, 1]), -int(c[i, 0])), dims=(0, 1))
                        for i in range(9)])
    out["P7_stream"] = {
        "handwritten_max_abs": float((fs - hand).abs().max()),
        "mass_before": float(f.sum()), "mass_after": float(fs.sum()),
        "mass_rel_drift": float((fs.sum() - f.sum()).abs() / f.sum().abs()),
    }

    # P8 -- declared entry / test surface vs what is on disk
    declared = {}
    init_txt = (REF / "__init__.py").read_text()
    for relpath in ("run.py", "tests", "tl_arm.py", "oss_arm.py", "compare.py"):
        declared[relpath] = (REF.parent / relpath).exists()
    out["P8_declared_surface"] = {
        "docstring_mentions_run_py": "run.py" in init_txt,
        "docstring_mentions_arms": all(k in init_txt for k in ("tl_arm.py", "oss_arm.py", "compare.py")),
        "docstring_mentions_tests_dir": "tests/" in init_txt,
        "on_disk": declared,
        "files_present_in_refs": sorted(p.name for p in REF.glob("*")),
        "files_present_in_delivery_root": sorted(p.name for p in REF.parent.glob("*")
                                                 if p.name not in ("probe_delivery.py",)),
    }
    del m  # keep the reference explicit

    print(json.dumps(out, indent=1, ensure_ascii=False))
    (HERE / "probe_delivery.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


if __name__ == "__main__":
    main()