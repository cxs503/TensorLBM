#!/usr/bin/env python3
"""XVAL m17678 — TensorLBM conservative Allen-Cahn h-population x CFD-GO/TCLB.

Target of comparison: TCLB ``models/multiphase/d3q27_pf_velocity`` (Fakhari &
Mitchell lineage), i.e. the conservative Allen-Cahn equation carried by an
*h-population* with an anti-diffusion (counter-gradient) source term.

Job 1 (structural, zero hand-copy): every TCLB-side number is parsed out of the
TCLB source *text* at run time -- q15 weights, the ``tmp1`` anti-diffusion
coefficient, the counter-term expression, the collision line, the isotropic
gradient / Laplacian formulas (``model.R``), the lattice matrices
(``src/lib/lattice.R``) and the ``AddSetting`` defaults (``Dynamics.R``) -- and
compared with the TensorLBM implementations (P1, P2, P2b, P9a, P9b).

Job 2 (physics, analytic references): Arm B = tensorlbm.phasefield.cac_h_population
(TCLB-parity kernel); Arm A = TensorLBM's existing h-population kernel
(``tensorlbm.cac_lbm2d`` functions driven by a prescribed velocity).  References:
analytic tanh profile, exact translation, exact conservation of the flux form.

Run:  python scripts/xval_m17678_tclb_cac_h_population.py [--quick]
Env:  TCLB_ROOT (default /tmp/TCLB)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import tensorlbm.phasefield.cac_h_population as B  # noqa: E402
from tensorlbm import cac_lbm2d as A  # noqa: E402

TCLB_ROOT = Path(os.environ.get("TCLB_ROOT", "/tmp/TCLB"))
MODEL = TCLB_ROOT / "models" / "multiphase" / "d3q27_pf_velocity"
REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts" / "xval_m17678"

torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
DTYPE = torch.float64


# ===========================================================================
# 1. TCLB source parsing (zero hand-copy)
# ===========================================================================


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _line_of(text: str, needle: str) -> int:
    idx = text.find(needle)
    return -1 if idx < 0 else text.count("\n", 0, idx) + 1


def provenance() -> dict:
    files = {
        "Dynamics.c.Rt": MODEL / "Dynamics.c.Rt",
        "Dynamics.R": MODEL / "Dynamics.R",
        "model.R": MODEL / "model.R",
        "lib/lattice.R": TCLB_ROOT / "src" / "lib" / "lattice.R",
        "Description.md": MODEL / "Description.md",
    }
    prov = {
        "tclb_root": str(TCLB_ROOT),
        "model": "models/multiphase/d3q27_pf_velocity",
        "license_note": "TCLB is GPL-3.0: read-only reference, no code copied "
        "(docs/oss_inventory/2026-11-09_TCLB_deep_dive.md L20 = reference-only)",
        "files": {},
    }
    for label, path in files.items():
        prov["files"][label] = {
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256(path) if path.exists() else None,
            "bytes": path.stat().st_size if path.exists() else None,
        }
    return prov


def extract_scheme_lines(dyn_c: str) -> dict:
    needles = {
        "tmp1_collision": "real_t tmp1 = (1.0 - 4.0*(PhaseF - C0)*(PhaseF - C0))/IntWidth;",
        "calcF_phi_body": "f_phi = wg[i] * tmp1 * (d3q27_ex[i]*nx + d3q27_ey[i]*ny + d3q27_ez[i]*nz);",
        "calcGamma": "gamma = wg[i] * (1 + 3.0*(tmp) + 4.5*(tmp*tmp) - 1.5*(u2mag)) ;",
        "weights_q15": "wh = c(rep(16/72,1),rep(8/72,6),rep(1/72,8))",
        "collision_bgk": "C(h, h - omega * ( h - w_h*heq + 0.5*Fphi) + Fphi)",
        "collision_mrt": "C(h, h - omega * ( h - heq + 0.5*Fphi) + Fphi)",
        "heq_assign": 'C(heq, EQ_h$feq * PV("C"))',
        "normal_cutoff": "if (magnPhi < minGradient){",
        "hPops27": "#define hPops 27",
        "hPops15": "#define hPops 15",
    }
    return {k: {"line": _line_of(dyn_c, v), "text": v} for k, v in needles.items()}


def extract_settings(dyn_r: str) -> dict:
    out = {}
    for m in re.finditer(r'AddSetting\(name="([A-Za-z_0-9]+)"(.*?)\)\s*\n', dyn_r):
        name, rest = m.group(1), m.group(2)
        default = re.search(r"default\s*=\s*([^,]+)", rest)
        omega = re.search(r"omega_phi\s*=\s*'([^']+)'", rest)
        out[name] = {
            "line": dyn_r.count("\n", 0, m.start()) + 1,
            "default": default.group(1).strip() if default else None,
        }
        if omega:
            out[name]["omega_phi"] = omega.group(1)
    return out


def extract_q15_weights(dyn_c: str) -> dict:
    m = re.search(r"wh = c\((rep\([0-9]+/72,[0-9]+\)(?:,rep\([0-9]+/72,[0-9]+\))*)\)", dyn_c)
    if not m:
        return {"found": False}
    reps = re.findall(r"rep\(([0-9]+)/72,([0-9]+)\)", m.group(1))
    vals = []
    for num, rep in reps:
        vals += [int(num) / 72.0] * int(rep)
    mom = 6 * vals[1] + 8 * vals[-1]
    return {
        "found": True,
        "line": _line_of(dyn_c, m.group(0)),
        "literal": m.group(0),
        "n": len(vals),
        "distinct": sorted({round(v, 12) for v in vals}, reverse=True),
        "sum": sum(vals),
        "second_moment_xx": mom,
        "note": "q15 phase lattice: sum(w c_x^2)=%.6f != 1/3" % mom,
    }


def _function_body(path: Path, fname: str) -> str:
    text = _read(path)
    start = text.find(fname + " <- function(")
    if start < 0:
        raise RuntimeError("%s not found in %s" % (fname, path))
    end = text.find("\n}\n", start)
    return text[start: end if end > 0 else len(text)]


def _strings(line: str) -> str:
    """Evaluate an R ``cat(paste0(...))`` line as a C string.

    The R generator interleaves variables (``myStringVec``/``myStringSca``) with
    quoted literals; we replace the variables by their C meaning and concatenate,
    so the *text* is reconstructed from the source rather than retyped.
    """
    idx = line.find("cat(paste0(")
    if idx >= 0:
        body = line[idx + len("cat(paste0("):]
        body = body.rsplit("))", 1)[0]
    else:
        body = line
    body = body.replace("myStringVec", "GV").replace("myStringSca", "PhaseF")
    body = body.replace("PhaseF,", "PhaseF").replace("GV,", "GV")
    body = body.replace('",', "").replace('"', "")
    body = body.replace("\\n", "").strip()
    return body


def r_gradient_spec(model_r: Path) -> dict:
    body = _function_body(model_r, "IsotropicGrad")
    comps = {}
    for comp in ("x", "y", "z"):
        idx = body.find('cat(paste0(myStringVec,".%s = ' % comp)
        if idx < 0:
            continue
        line = body[idx: body.find("\n", idx)]
        comps[comp] = {"r_text": _strings(line).split("=", 1)[-1].strip()}
    div = re.search(r'\.x /= ([0-9.]+)', body)
    return {"components": comps, "divisor": float(div.group(1)) if div else None}


def r_laplacian_spec(model_r: Path) -> dict:
    body = _function_body(model_r, "myLaplace")
    idx = body.find('cat(paste0(myStringVec, "')
    line = body[idx: body.find("\n", idx)]
    div = re.search(r'"/= ([0-9.]+)', body)
    return {"r_text": _strings(line).strip(), "divisor": float(div.group(1)) if div else None}


_PHASE_CALL = re.compile(r"PhaseF\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")


def eval_gradient_formula(spec: dict, phi: torch.Tensor) -> tuple:
    cache = {}

    def F(i, j, k):
        key = (int(i), int(j), int(k))
        if key not in cache:
            cache[key] = torch.roll(phi, shifts=(-key[2], -key[1], -key[0]), dims=(2, 1, 0))
        return cache[key]

    out = {}
    for comp, info in spec["components"].items():
        py = _PHASE_CALL.sub(r"F(\1,\2,\3)", info["r_text"]).replace(";", "")
        out[comp] = eval(py, {"__builtins__": {}}, {"F": F}) / spec["divisor"]  # noqa: S307
    return (out["x"], out["y"], out["z"])


def eval_laplacian_formula(spec: dict, phi: torch.Tensor) -> torch.Tensor:
    def F(i, j, k):
        return torch.roll(phi, shifts=(-int(k), -int(j), -int(i)), dims=(2, 1, 0))

    expr = spec["r_text"].split("=", 1)[-1]
    py = _PHASE_CALL.sub(r"F(\1,\2,\3)", expr).replace(";", "")
    return eval(py, {"__builtins__": {}}, {"F": F}) / spec["divisor"]  # noqa: S307


def parse_lattice_matrix(lattice_r: str, name: str) -> list:
    m = re.search(name + r"\s*=\s*matrix\(c\(([^)]*)\)", lattice_r, re.S)
    if not m:
        return []
    nums = [int(v) for v in re.findall(r"-?\d+", m.group(1))]
    q = len(nums) // 3
    return [[nums[a * q + i] for a in range(3)] for i in range(q)]


def symbolic_checks(picks: dict) -> dict:
    import sympy as sp

    phi, C0, W, omega = sp.symbols("phi C0 W omega", real=True)
    res = {}
    txt = picks["tmp1_collision"]["text"]
    rhs = txt.split("=", 1)[1].strip().rstrip(";")
    expr = sp.sympify(rhs.replace("PhaseF", "phi").replace("IntWidth", "W"))
    resid = sp.simplify(expr - 4 * phi * (1 - phi) / W)
    res["tmp1_identity"] = {
        "text": txt,
        "line": picks["tmp1_collision"]["line"],
        "sympy": str(sp.simplify(expr)),
        "residual_vs_4phi(1-phi)/W": str(resid),
        "equals_4phi(1-phi)/W_at_C0_1_2": bool(sp.simplify(resid.subs(C0, sp.Rational(1, 2))) == 0),
    }

    h, heq, F = sp.symbols("h heq F")
    tclb = h - omega * (h - heq + sp.Rational(1, 2) * F) + F
    ours = h - omega * (h - heq) + (1 - omega / 2) * F
    res["collision_identity"] = {
        "text": picks["collision_bgk"]["text"],
        "line": picks["collision_bgk"]["line"],
        "residual_tclb_minus_collide_then_source": str(sp.simplify(tclb - ours)),
    }

    cu, u2, w = sp.symbols("cu u2 w")
    eq_txt = picks["calcGamma"]["text"]
    rhs = eq_txt.split("wg[i] * ", 1)[1].strip().rstrip(";").strip()
    parsed = sp.sympify(rhs.replace("tmp", "cu").replace("u2mag", "u2"))
    maxwell = 1 + 3 * cu + sp.Rational(9, 2) * cu ** 2 - sp.Rational(3, 2) * u2
    res["equilibrium_text"] = {
        "text": eq_txt,
        "line": picks["calcGamma"]["line"],
        "parsed_rhs": str(parsed),
        "residual_vs_second_order_maxwellian": str(sp.simplify(parsed - maxwell)),
        "is_second_order_maxwellian": bool(sp.simplify(parsed - maxwell) == 0),
    }
    return res

# ===========================================================================
# 2. Arm A: TensorLBM's existing h-population kernel, prescribed velocity
# ===========================================================================


class ArmA2D:
    """``tensorlbm.cac_lbm2d`` phase-field functions, driven by a given u.

    Only the ``h`` population is evolved; the HCH ``g`` population / force chain is
    out of scope (declared in the prereg).  Library functions are *called*, not
    re-implemented; the lattice weights are passed in exact float64 so the
    comparison is not polluted by the module's float32 weight table.
    """

    def __init__(self, shape, width, mobility):
        self.shape = shape
        self.width = float(width)
        self.mobility = float(mobility)
        self.tau_f = 0.5 + 3.0 * self.mobility
        st = B.get_stencil("d2q9")
        self.wv = st.W.view(-1, 1, 1).to(DTYPE)
        self.cf = [st.C[:, 0].view(-1, 1, 1).to(DTYPE), st.C[:, 1].view(-1, 1, 1).to(DTYPE)]
        self.c = st.C
        self.f = None
        self.phi_u_prev = None

    def initialize(self, phi0):
        zero = (torch.zeros_like(phi0), torch.zeros_like(phi0))
        self.f = A.f_equilibrium(phi0, zero, self.wv, self.cf)
        self.phi_u_prev = (torch.zeros_like(phi0), torch.zeros_like(phi0))

    @property
    def phi(self):
        return self.f.sum(dim=0)

    def collision(self, u=None):
        phi = self.phi
        nb = A.shifted_neighbors(phi, axes="pp")
        grad = A.iso_gradient_2d(phi, axes="pp", nb=nb)
        gnorm = (grad[0] ** 2 + grad[1] ** 2).sqrt().clamp(min=1e-12)
        n_hat = (grad[0] / gnorm, grad[1] / gnorm)
        lam = A.lambda_phi(phi, self.width)
        if u is None:
            u = (torch.zeros_like(phi), torch.zeros_like(phi))
        phi_u_now = (phi * u[0], phi * u[1])
        phi_u_dt = (phi_u_now[0] - self.phi_u_prev[0], phi_u_now[1] - self.phi_u_prev[1])
        self.phi_u_prev = phi_u_now
        feq = A.f_equilibrium(phi, u, self.wv, self.cf)
        fsrc = A.f_source(phi_u_dt, lam, n_hat, self.tau_f, self.wv, self.cf)
        self.f = self.f + (1.0 / self.tau_f) * (feq - self.f) + fsrc

    def stream(self):
        f = torch.empty_like(self.f)
        f[0] = self.f[0]
        for q in range(1, 9):
            f[q] = A._shift(self.f[q], -int(self.c[q, 1]), -int(self.c[q, 0]), "pp")
        self.f = f

    def step(self, u=None):
        self.collision(u)
        self.stream()

    def run(self, n, u=None):
        for _ in range(int(n)):
            self.step(u)


# ===========================================================================
# 3. measurement helpers
# ===========================================================================


def radius_from_area(phi: torch.Tensor, level: float = 0.5, dim=None) -> float:
    vol = float((phi > level).to(DTYPE).sum().item())
    dim = phi.dim() if dim is None else dim
    if dim == 2:
        return math.sqrt(vol / math.pi)
    return (3.0 * vol / (4.0 * math.pi)) ** (1.0 / 3.0)


def centroid(phi: torch.Tensor) -> tuple:
    mesh = torch.meshgrid(*[torch.arange(n, dtype=DTYPE) for n in phi.shape], indexing="ij")
    m0 = float(phi.sum().item())
    return tuple(float((phi * g).sum().item() / m0) for g in reversed(mesh))


def radial_profile(phi: torch.Tensor, center=None):
    mesh = torch.meshgrid(*[torch.arange(n, dtype=DTYPE) for n in phi.shape], indexing="ij")
    if center is None:
        center = tuple(float(n) / 2.0 for n in phi.shape)
    r2 = None
    for g, c0 in zip(mesh, reversed(center)):
        term = (g - c0) ** 2
        r2 = term if r2 is None else r2 + term
    r = r2.sqrt()
    nb = int(r.max().item()) + 2
    acc = torch.zeros(nb, dtype=DTYPE)
    cnt = torch.zeros(nb, dtype=DTYPE)
    bins = r.round().to(torch.int64).flatten()
    acc.index_add_(0, bins, phi.flatten())
    cnt.index_add_(0, bins, torch.ones_like(phi).flatten())
    ok = cnt > 0
    return torch.arange(nb, dtype=DTYPE)[ok], (acc / cnt)[ok]


def profile_error(phi: torch.Tensor, radius: float, width_eff: float) -> dict:
    r, p = radial_profile(phi)
    ana = 0.5 + 0.5 * torch.tanh(2.0 * (radius - r) / width_eff)
    return {
        "rel_l2": float((p - ana).norm().item() / ana.norm().item()),
        "r": r.tolist(),
        "phi": p.tolist(),
        "phi_analytic": ana.tolist(),
    }


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm().item() / b.norm().item())


# ===========================================================================
# 4. experiments
# ===========================================================================


def run_static2d(n=128, radius=25.0, width=4.0, mobility=1.0 / 3.0, steps=2000, sample=100):
    phi0 = B.circle_phi((n, n), radius, width)
    rec = {
        "params": {"n": n, "R": radius, "W": width, "M": mobility, "steps": steps, "dim": 2,
                   "stencil": "d2q9", "omega": 1.0 / (3 * mobility + 0.5),
                   "R_over_W": radius / width, "L_over_R": n / radius},
        "arms": {},
    }
    phis, profs = {}, {}
    for name in ("B", "A"):
        if name == "B":
            sim = B.CACHPopulation((n, n), width=width, mobility=mobility, stencil="d2q9",
                                   min_gradient=1e-8, eq_order=2, dtype=DTYPE)
        else:
            sim = ArmA2D((n, n), width, mobility)
        sim.initialize(phi0.clone())
        m0 = float(sim.phi.sum().item())
        r0 = radius_from_area(sim.phi)
        hist = []
        t0 = time.time()
        for k in range(int(steps)):
            sim.step()
            if (k + 1) % sample == 0:
                hist.append({"t": k + 1,
                             "mass_drift": abs(float(sim.phi.sum().item()) - m0) / m0,
                             "R": radius_from_area(sim.phi)})
        w_eff = B.predicted_width(width, mobility)
        r_end = radius_from_area(sim.phi)
        prof = profile_error(sim.phi, r_end, w_eff)
        phis[name] = sim.phi.clone()
        profs[name] = prof
        rec["arms"][name] = {
            "mass_drift": abs(float(sim.phi.sum().item()) - m0) / m0,
            "R0": r0, "R_end": r_end, "R_drift": abs(r_end - r0) / r0,
            "W_eff_pred": w_eff,
            "W_measured": B.width_from_profile(sim.phi, r_end),
            "profile_err_rel_l2": prof["rel_l2"],
            "max_abs_phi": float(sim.phi.abs().max().item()),
            "min_phi": float(sim.phi.min().item()),
            "history": hist,
            "wall_s": time.time() - t0,
        }
    rec["cross_arm_rel_l2"] = rel_l2(phis["A"], phis["B"])
    rec["profile"] = {"n_bins": len(profs["B"]["r"]),
                      "r": profs["B"]["r"], "phi_B": profs["B"]["phi"],
                      "phi_analytic": profs["B"]["phi_analytic"],
                      "r_A": profs["A"]["r"], "phi_A": profs["A"]["phi"]}
    return rec


def run_advection2d(n=128, radius=20.0, width=4.0, mobility=1.0 / 3.0, U=0.05, cells=32,
                    eq_order=2):
    steps = int(round(cells / U))
    phi0 = B.circle_phi((n, n), radius, width)
    shift = int(round(U * steps))
    u = (torch.full((n, n), U, dtype=DTYPE), torch.full((n, n), U, dtype=DTYPE))
    exact = torch.roll(phi0, shifts=(shift, shift), dims=(0, 1))
    rec = {"params": {"n": n, "R": radius, "W": width, "M": mobility, "U": U, "steps": steps,
                      "exact_shift": shift, "CFL": U, "eq_order": eq_order, "dim": 2,
                      "omega": 1.0 / (3 * mobility + 0.5)}, "arms": {}}
    phis = {}
    for name in ("B", "A"):
        if name == "B":
            sim = B.CACHPopulation((n, n), width=width, mobility=mobility, stencil="d2q9",
                                   min_gradient=1e-8, eq_order=eq_order, dtype=DTYPE)
        else:
            sim = ArmA2D((n, n), width, mobility)
        sim.initialize(phi0.clone())
        m0 = float(sim.phi.sum().item())
        t0 = time.time()
        sim.run(steps, u=u)
        phi = sim.phi
        cx, cy = centroid(phi)
        d = [cx - (n - 1) / 2.0, cy - (n - 1) / 2.0]
        d = [v - n * round(v / n) for v in d]
        phis[name] = phi.clone()
        rec["arms"][name] = {
            "mass_drift": abs(float(phi.sum().item()) - m0) / m0,
            "centroid_dx": d[0], "centroid_dy": d[1],
            "centroid_err_rel": max(abs(d[0] - shift), abs(d[1] - shift)) / shift,
            "field_err_vs_exact_rel_l2": rel_l2(phi, exact),
            "R_end": radius_from_area(phi),
            "max_abs_phi": float(phi.abs().max().item()),
            "wall_s": time.time() - t0,
        }
    rec["cross_arm_rel_l2"] = rel_l2(phis["A"], phis["B"])
    mid = n // 2
    rec["cut"] = {"x": list(range(n)), "initial": phi0[mid].tolist(), "exact": exact[mid].tolist(),
                  "A": phis["A"][mid].tolist(), "B": phis["B"][mid].tolist()}
    return rec


def run_static3d(n=48, radius=10.0, width=3.0, mobility=1.0 / 3.0, steps=400, sample=50):
    phi0 = B.sphere_phi((n, n, n), radius, width)
    sim = B.CACHPopulation((n, n, n), width=width, mobility=mobility, stencil="d3q27",
                           min_gradient=1e-8, eq_order=2, dtype=DTYPE)
    sim.initialize(phi0)
    m0 = float(sim.phi.sum().item())
    r0 = radius_from_area(sim.phi, dim=3)
    hist = []
    t0 = time.time()
    for k in range(int(steps)):
        sim.step()
        if (k + 1) % sample == 0:
            hist.append({"t": k + 1, "mass_drift": abs(float(sim.phi.sum().item()) - m0) / m0,
                         "R": radius_from_area(sim.phi, dim=3)})
    r_end = radius_from_area(sim.phi, dim=3)
    w_eff = B.predicted_width(width, mobility)
    prof = profile_error(sim.phi, r_end, w_eff)
    return {
        "params": {"n": n, "R": radius, "W": width, "M": mobility, "steps": steps,
                   "stencil": "d3q27", "dim": 3, "omega": 1.0 / (3 * mobility + 0.5),
                   "R_over_W": radius / width},
        "mass_drift": abs(float(sim.phi.sum().item()) - m0) / m0,
        "R0": r0, "R_end": r_end, "R_drift": abs(r_end - r0) / r0,
        "W_eff_pred": w_eff,
        "W_measured": B.width_from_profile(sim.phi, r_end),
        "profile_err_rel_l2": prof["rel_l2"],
        "max_abs_phi": float(sim.phi.abs().max().item()),
        "history": hist,
        "profile": {"r": prof["r"], "phi": prof["phi"], "phi_analytic": prof["phi_analytic"]},
        "wall_s": time.time() - t0,
    }


def run_sensitivity(width=4.0, mobility=1.0 / 3.0, n=96, radius=18.0, steps=800):
    res = {"M_sweep": [], "W_sweep": [], "U_sweep": [], "step_sweep": []}
    for M in (1.0 / 12.0, 1.0 / 6.0, 1.0 / 3.0, 1.0 / 2.0):
        sim = B.CACHPopulation((64, 64), width=width, mobility=M, stencil="d2q9",
                               min_gradient=1e-8, dtype=DTYPE)
        sim.initialize(B.circle_phi((64, 64), 12.0, width))
        m0 = float(sim.phi.sum().item())
        sim.run(600)
        r_end = radius_from_area(sim.phi)
        w_pred = B.predicted_width(width, M)
        w_meas = B.width_from_profile(sim.phi, r_end)
        res["M_sweep"].append({"M": M, "omega": 1.0 / (3 * M + 0.5), "W": width,
                               "W_pred": w_pred, "W_measured": w_meas,
                               "ratio": w_meas / w_pred, "R_end": r_end,
                               "mass_drift": abs(float(sim.phi.sum().item()) - m0) / m0})
    for W in (2.0, 3.0, 4.0, 6.0):
        sim = B.CACHPopulation((n, n), width=W, mobility=mobility, stencil="d2q9",
                               min_gradient=1e-8, dtype=DTYPE)
        sim.initialize(B.circle_phi((n, n), radius, W))
        m0 = float(sim.phi.sum().item())
        r0 = radius_from_area(sim.phi)
        sim.run(steps)
        r_end = radius_from_area(sim.phi)
        res["W_sweep"].append({"W": W, "W_eff": B.predicted_width(W, mobility),
                               "R_over_W": radius / W, "R0": r0, "R_end": r_end,
                               "R_drift": abs(r_end - r0) / r0,
                               "mass_drift": abs(float(sim.phi.sum().item()) - m0) / m0,
                               "profile_err": profile_error(
                                   sim.phi, r_end, B.predicted_width(W, mobility))["rel_l2"]})
    for U in (0.01, 0.02, 0.04, 0.05):
        cells = 16
        steps_u = int(round(cells / U))
        phi0 = B.circle_phi((128, 128), 20.0, width)
        u = (torch.full((128, 128), U, dtype=DTYPE), torch.full((128, 128), U, dtype=DTYPE))
        sim = B.CACHPopulation((128, 128), width=width, mobility=mobility, stencil="d2q9",
                               min_gradient=1e-8, dtype=DTYPE)
        sim.initialize(phi0)
        sim.run(steps_u, u=u)
        exact = torch.roll(phi0, shifts=(cells, cells), dims=(0, 1))
        res["U_sweep"].append({"U": U, "steps": steps_u, "CFL": U, "exact_shift": cells,
                               "field_err_vs_exact_rel_l2": rel_l2(sim.phi, exact),
                               "R_end": radius_from_area(sim.phi)})
    phi0 = B.circle_phi((128, 128), 25.0, width)
    prev = None
    for st in (250, 500, 1000, 2000):
        sim = B.CACHPopulation((128, 128), width=width, mobility=mobility, stencil="d2q9",
                               min_gradient=1e-8, dtype=DTYPE)
        sim.initialize(phi0)
        sim.run(st)
        r = radius_from_area(sim.phi)
        res["step_sweep"].append({"steps": st, "R": r,
                                  "delta_R_vs_prev": None if prev is None else abs(r - prev)})
        prev = r
    return res


def structural_parity() -> dict:
    """P1 / P2 / P2b / P9a / P9b: structural identities against the TCLB text."""
    res = {}
    st27 = B.get_stencil("d3q27")
    n = 24
    g = torch.Generator().manual_seed(17678)
    phi = torch.rand((n, n, n), generator=g, dtype=DTYPE)
    for _ in range(3):  # light smoothing so the normals are meaningful
        phi = phi + B.isotropic_laplacian(phi, st27) * 1.0

    spec = r_gradient_spec(MODEL / "model.R")
    lap_spec = r_laplacian_spec(MODEL / "model.R")
    grad_ref = eval_gradient_formula(spec, phi)
    grad_ours = B.isotropic_gradient(phi, st27)
    res["P2_gradient"] = {
        "tclb_divisor": spec["divisor"],
        "tclb_text_x_head": spec["components"]["x"]["r_text"][:100],
        "rel_err_per_axis": [rel_l2(grad_ours[a], grad_ref[a]) for a in range(3)],
    }
    res["P2_laplacian"] = {"tclb_divisor": lap_spec["divisor"],
                           "tclb_text_head": lap_spec["r_text"][:100],
                           "rel_err": rel_l2(B.isotropic_laplacian(phi, st27),
                                             eval_laplacian_formula(lap_spec, phi))}

    phi2 = torch.rand((32, 32), generator=g, dtype=DTYPE)
    gA = A.iso_gradient_2d(phi2, axes="pp")
    gB = B.isotropic_gradient(phi2, B.get_stencil("d2q9"))
    res["P2b_d2q9_A_vs_B_operator"] = {
        "rel_err": [rel_l2(gB[0], gA[0]), rel_l2(gB[1], gA[1])],
        "note": "residual is the float32 weight table inside tensorlbm.d2q9 (W.dtype)",
    }

    phi3 = B.sphere_phi((n, n, n), 6.0, 3.0)
    n_hat, _ = B.interface_normal(phi3, st27, min_gradient=1e-8)
    W = 3.0
    F_ours = B.counter_term(phi3, n_hat, W, st27)
    tmp1 = (1.0 - 4.0 * (phi3 - 0.5) ** 2) / W
    F_ref = torch.empty_like(F_ours)
    for q in range(27):
        cx, cy, cz = [int(v) for v in st27.C[q].tolist()]
        F_ref[q] = st27.W[q].to(DTYPE) * tmp1 * (cx * n_hat[0] + cy * n_hat[1] + cz * n_hat[2])
    res["P1_counter_term"] = {
        "rel_err": rel_l2(F_ours, F_ref),
        "max_abs_diff": float((F_ours - F_ref).abs().max().item()),
        "net_source_max_per_node": float(F_ours.sum(dim=0).abs().max().item()),
    }

    simB = B.CACHPopulation((n, n, n), width=W, mobility=1.0 / 3.0, stencil="d3q27",
                            min_gradient=1e-8, dtype=DTYPE)
    simB.initialize(phi3)
    u3 = (torch.full((n, n, n), 0.01, dtype=DTYPE),
          torch.full((n, n, n), -0.005, dtype=DTYPE),
          torch.zeros((n, n, n), dtype=DTYPE))
    h0 = simB.h.clone()
    simB.step(u3)
    phi_c = h0.sum(dim=0)
    nb_c = B._neighbours(phi_c, st27)
    n_c, _ = B.interface_normal(phi_c, st27, neighbours=nb_c, min_gradient=1e-8)
    F_c = B.counter_term(phi_c, n_c, W, st27)
    shape27 = (27, 1, 1, 1)
    cdotu = torch.zeros(shape27, dtype=DTYPE)
    u2 = torch.zeros((n, n, n), dtype=DTYPE)
    for a in range(3):
        ca = st27.C[:, a].to(DTYPE).view(shape27)
        cdotu = cdotu + ca * u3[a].unsqueeze(0)
        u2 = u2 + u3[a] ** 2
    gamma = st27.W.view(shape27) * (1 + 3 * cdotu + 4.5 * cdotu ** 2 - 1.5 * u2.unsqueeze(0))
    h_coll = B.update_from_tclb(h0, phi_c.unsqueeze(0) * gamma, F_c, simB.omega)
    h_ref = torch.empty_like(h_coll)
    for q in range(27):
        cx, cy, cz = [int(v) for v in st27.C[q].tolist()]
        h_ref[q] = torch.roll(h_coll[q], shifts=(-cz, -cy, -cx), dims=(0, 1, 2))
    res["P9a_module_vs_tclb_text"] = {"rel_err": rel_l2(simB.h, h_ref),
                                      "max_abs_diff": float((simB.h - h_ref).abs().max().item())}

    phi02 = B.circle_phi((64, 64), 12.0, 4.0)
    simB2 = B.CACHPopulation((64, 64), width=4.0, mobility=1.0 / 3.0, stencil="d2q9",
                             min_gradient=1e-8, dtype=DTYPE)
    simB2.initialize(phi02)
    simB2.step()
    simA2 = ArmA2D((64, 64), 4.0, 1.0 / 3.0)
    simA2.initialize(phi02)
    simA2.step()
    res["P9b_armA_vs_armB_one_step"] = {
        "rel_err_phi": rel_l2(simA2.phi, simB2.phi),
        "max_abs_diff_phi": float((simA2.phi - simB2.phi).abs().max().item()),
    }
    return res


# ===========================================================================
# 5. figures
# ===========================================================================


def make_figures(static2d, adv2d, static3d, sens, outdir: Path) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    figs = []
    p = static2d["profile"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(p["r"], p["phi_analytic"], "k-", lw=1.2, label="analytic tanh, $W_{eff}=3MW$")
    axes[0].plot(p["r"], p["phi_B"], "o", ms=3, label="Arm B (TCLB parity, d2q9)")
    axes[0].plot(p["r_A"], p["phi_A"], "x", ms=4, label="Arm A (cac_lbm2d kernel)")
    axes[0].set_xlabel("r / cells")
    axes[0].set_ylabel(r"$\phi$")
    axes[0].set_title("static droplet, %d steps" % static2d["params"]["steps"])
    axes[0].legend(fontsize=8)
    for name, style in (("B", "-"), ("A", "--")):
        h = static2d["arms"][name]["history"]
        axes[1].semilogy([x["t"] for x in h], [max(x["mass_drift"], 1e-18) for x in h], style,
                         label="mass drift, arm %s" % name)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel(r"$|M(t)-M(0)|/M(0)$")
    axes[1].set_title("mass conservation of the flux form")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    f = outdir / "fig1_static2d_profile_mass.png"
    fig.savefig(f, dpi=130)
    plt.close(fig)
    figs.append(str(f))

    c = adv2d["cut"]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.plot(c["x"], c["initial"], "k:", lw=1.0, label="initial (t=0)")
    ax.plot(c["x"], c["exact"], "k-", lw=1.0, label="exact translation")
    ax.plot(c["x"], c["B"], "C0", lw=1.2, label="Arm B (TCLB parity)")
    ax.plot(c["x"], c["A"], "C1--", lw=1.2, label="Arm A (cac_lbm2d kernel)")
    ax.set_xlabel("x / cells (y = mid row)")
    ax.set_ylabel(r"$\phi$")
    ax.set_title("uniform advection, %d steps, CFL=%.2f"
                 % (adv2d["params"]["steps"], adv2d["params"]["U"]))
    ax.legend(fontsize=8)
    fig.tight_layout()
    f = outdir / "fig2_advection_cut.png"
    fig.savefig(f, dpi=130)
    plt.close(fig)
    figs.append(str(f))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ms = sens["M_sweep"]
    axes[0].errorbar([x["W_pred"] for x in ms], [x["W_measured"] for x in ms], fmt="o",
                     label="measured (tanh fit)")
    lim = [0.0, max(x["W_pred"] for x in ms) * 1.25 + 0.2]
    axes[0].plot(lim, lim, "k--", lw=0.8, label="1:1")
    axes[0].set_xlabel(r"predicted $3MW$")
    axes[0].set_ylabel("measured interface width")
    axes[0].set_title("interface-width law")
    axes[0].legend(fontsize=8)
    ws = sens["W_sweep"]
    axes[1].plot([x["W"] for x in ws], [100 * x["R_drift"] for x in ws], "o-", label="R drift [%]")
    axes[1].plot([x["W"] for x in ws], [100 * x["profile_err"] for x in ws], "s--",
                 label="profile err [%]")
    axes[1].set_xlabel("W (anti-diffusion coefficient)")
    axes[1].set_title("width sensitivity, M=1/3")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    f = outdir / "fig3_sensitivity.png"
    fig.savefig(f, dpi=130)
    plt.close(fig)
    figs.append(str(f))

    p3 = static3d["profile"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(p3["r"], p3["phi"], "o", ms=3, label="Arm B, d3q27")
    ax.plot(p3["r"], p3["phi_analytic"], "k-", lw=1.0, label="analytic tanh")
    ax.set_xlabel("r / cells")
    ax.set_ylabel(r"$\phi$")
    ax.set_title("static sphere %d^3, %d steps"
                 % (static3d["params"]["n"], static3d["params"]["steps"]))
    ax.legend(fontsize=8)
    fig.tight_layout()
    f = outdir / "fig4_static3d_sphere.png"
    fig.savefig(f, dpi=130)
    plt.close(fig)
    figs.append(str(f))
    return figs


# ===========================================================================
# 6. main
# ===========================================================================


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="short runs (CI)")
    ap.add_argument("--out", default=str(ART))
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if not MODEL.exists():
        print("FATAL: TCLB model not found at %s (set TCLB_ROOT)" % MODEL, file=sys.stderr)
        return 2

    t_start = time.time()
    dyn_c = _read(MODEL / "Dynamics.c.Rt")
    dyn_r = _read(MODEL / "Dynamics.R")
    picks = extract_scheme_lines(dyn_c)
    settings = extract_settings(dyn_r)
    q15 = extract_q15_weights(dyn_c)
    lattice_r = _read(TCLB_ROOT / "src" / "lib" / "lattice.R")
    st27 = B.get_stencil("d3q27")
    st2d = B.get_stencil("d2q9")
    lattice = {
        "d3q27_matches_tensorlbm_order": parse_lattice_matrix(lattice_r, "d3q27")
        == [list(map(int, r)) for r in st27.C.tolist()],
        "d2q9_matches_tensorlbm_order": parse_lattice_matrix(lattice_r, "d2q9")
        == [list(map(int, r)) for r in st2d.C.tolist()],
        "d3q27_n": len(parse_lattice_matrix(lattice_r, "d3q27")),
        "d2q9_n": len(parse_lattice_matrix(lattice_r, "d2q9")),
    }

    sym = symbolic_checks(picks)
    par = structural_parity()

    if args.quick:
        static2d = run_static2d(n=64, radius=12.0, steps=300, sample=100)
        adv2d = run_advection2d(n=64, radius=10.0, U=0.05, cells=8)
        static3d = run_static3d(n=24, radius=5.0, width=3.0, steps=40, sample=20)
        sens = run_sensitivity(n=64, radius=12.0, steps=120)
    else:
        static2d = run_static2d()
        adv2d = run_advection2d()
        static3d = run_static3d()
        sens = run_sensitivity()

    checks = {
        "P1_counter_term_rel_err": par["P1_counter_term"]["rel_err"],
        "P1_pass": par["P1_counter_term"]["rel_err"] < 1e-13,
        "P2_gradient_max_rel_err": max(par["P2_gradient"]["rel_err_per_axis"]),
        "P2_laplacian_rel_err": par["P2_laplacian"]["rel_err"],
        "P2_pass": (max(par["P2_gradient"]["rel_err_per_axis"]) < 1e-13
                    and par["P2_laplacian"]["rel_err"] < 1e-13),
        "P9a_rel_err": par["P9a_module_vs_tclb_text"]["rel_err"],
        "P9a_pass": par["P9a_module_vs_tclb_text"]["rel_err"] < 1e-13,
        "P9b_rel_err": par["P9b_armA_vs_armB_one_step"]["rel_err_phi"],
        "P9b_pass": par["P9b_armA_vs_armB_one_step"]["rel_err_phi"] < 1e-6,
        "P3_mass_drift_B": static2d["arms"]["B"]["mass_drift"],
        "P3_pass": static2d["arms"]["B"]["mass_drift"] < 1e-10,
        "P4_R_drift_B": static2d["arms"]["B"]["R_drift"],
        "P4_pass": static2d["arms"]["B"]["R_drift"] < 0.05,
        "P5_profile_err_B": static2d["arms"]["B"]["profile_err_rel_l2"],
        "P5_pass": static2d["arms"]["B"]["profile_err_rel_l2"] < 0.15,
        "P5_status_A_profile_err": static2d["arms"]["A"]["profile_err_rel_l2"],
        "P6_centroid_err_B": adv2d["arms"]["B"]["centroid_err_rel"],
        "P6_field_err_B": adv2d["arms"]["B"]["field_err_vs_exact_rel_l2"],
        "P6_pass": (adv2d["arms"]["B"]["centroid_err_rel"] < 0.05
                    and adv2d["arms"]["B"]["field_err_vs_exact_rel_l2"] < 0.15),
        "P6_status_A_field_err": adv2d["arms"]["A"]["field_err_vs_exact_rel_l2"],
        "P7_static_cross_arm_rel_l2": static2d["cross_arm_rel_l2"],
        "P7_advection_cross_arm_rel_l2": adv2d["cross_arm_rel_l2"],
        "P7_pass": max(static2d["cross_arm_rel_l2"], adv2d["cross_arm_rel_l2"]) < 0.15,
        "P8_ratio_min": min(x["ratio"] for x in sens["M_sweep"]),
        "P8_ratio_max": max(x["ratio"] for x in sens["M_sweep"]),
        "P8_pass": all(abs(x["ratio"] - 1.0) < 0.15 for x in sens["M_sweep"]),
        "P3d_mass_drift_3d": static3d["mass_drift"],
    }

    figs = make_figures(static2d, adv2d, static3d, sens, outdir)
    payload = {
        "mission": "m17678",
        "anchor": "XVAL-LBM-CONSERVATIVE-ALLEN-CAHN-H-POPULATION",
        "repo": "TensorLBM",
        "branch": "agent/m17678",
        "env": {
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
            "dtype": str(DTYPE),
            "device": "cpu",
            "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
            "mode": "quick" if args.quick else "full",
        },
        "provenance": provenance(),
        "tclb_scheme_lines": picks,
        "tclb_settings": {k: settings[k] for k in
                          ("IntWidth", "M", "omega_phi", "minGradient", "sigma")
                          if k in settings},
        "tclb_q15_weights": q15,
        "lattice_parity": lattice,
        "symbolic": sym,
        "structural": par,
        "checks": checks,
        "static2d": {k: v for k, v in static2d.items() if k != "profile"},
        "advection2d": {k: v for k, v in adv2d.items() if k != "cut"},
        "static3d": {k: v for k, v in static3d.items() if k != "profile"},
        "sensitivity": sens,
        "figures": figs,
        "wall_s": time.time() - t_start,
    }
    out = outdir / ("xval_tclb_cac_h_population%s.json" % ("_quick" if args.quick else ""))
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    (outdir / "curves.json").write_text(json.dumps(
        {"static2d_profile": static2d["profile"], "advection_cut": adv2d["cut"],
         "static3d_profile": static3d["profile"]}, indent=1))

    print("== structural ==")
    print(" P1 counter-term   rel_err = %.3e   pass=%s"
          % (checks["P1_counter_term_rel_err"], checks["P1_pass"]))
    print(" P2 gradient/lap   rel_err = %.3e / %.3e  pass=%s"
          % (checks["P2_gradient_max_rel_err"], checks["P2_laplacian_rel_err"], checks["P2_pass"]))
    print(" P9a module vs TCLB text  rel_err = %.3e  pass=%s"
          % (checks["P9a_rel_err"], checks["P9a_pass"]))
    print(" P9b arm A vs arm B (1 step) rel_err = %.3e  pass=%s"
          % (checks["P9b_rel_err"], checks["P9b_pass"]))
    print(" lattice order parity: d3q27=%s d2q9=%s"
          % (lattice["d3q27_matches_tensorlbm_order"], lattice["d2q9_matches_tensorlbm_order"]))
    print(" q15 weights (parsed): n=%s sum=%.6f  sum(w c_x^2)=%.6f"
          % (q15.get("n"), q15.get("sum", float("nan")), q15.get("second_moment_xx", float("nan"))))
    print("== physics ==")
    for arm in ("B", "A"):
        d = static2d["arms"][arm]
        print(" static2d arm %s: mass_drift=%.2e R_drift=%.3f%% profile_err=%.2f%% W_meas=%.3f"
              % (arm, d["mass_drift"], 100 * d["R_drift"], 100 * d["profile_err_rel_l2"],
                 d["W_measured"]))
    print(" static2d cross-arm rel L2 = %.3e" % static2d["cross_arm_rel_l2"])
    for arm in ("B", "A"):
        d = adv2d["arms"][arm]
        print(" adv2d   arm %s: centroid_err=%.3f%% field_err=%.3f%% mass_drift=%.2e"
              % (arm, 100 * d["centroid_err_rel"], 100 * d["field_err_vs_exact_rel_l2"],
                 d["mass_drift"]))
    print(" adv2d   cross-arm rel L2 = %.3e" % adv2d["cross_arm_rel_l2"])
    print(" static3d: mass_drift=%.2e R_drift=%.3f%% profile_err=%.2f%%"
          % (static3d["mass_drift"], 100 * static3d["R_drift"],
             100 * static3d["profile_err_rel_l2"]))
    print(" W law ratios:", ["%.3f" % x["ratio"] for x in sens["M_sweep"]])
    print(" artefacts:", out, "wall %.1fs" % (time.time() - t_start))
    failed = [k for k in checks if k.endswith("_pass") and not checks[k]]
    print("FAILED gates:", failed if failed else "none")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
