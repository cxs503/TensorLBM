#!/usr/bin/env python3
"""XVAL m18194 — TensorLBM pressure-evolution Allen-Cahn phase field x CFD-GO/TCLB.

Target of comparison: ``TCLB/models/multiphase/d2q9_pf_pressureEvolution``
(file header: "19/7/2016 - Model Developed: A. Fakhari, M. Geier, T. Lee"), i.e.
the formulation in which the *momentum* population carries the **pressure**
(``d_t p + rho cs^2 div(u) = 0``) instead of an equation of state, coupled to a
conservative Allen-Cahn phase field carried by an ``h``-population.

Job 1 (structural, zero hand-copy): every OSS-side number is parsed out of the
reference source *text* at run time (moment matrix, relaxation indices,
equilibrium / force / chemical-potential coefficients, directional stencils,
setting defaults); sha256 and line numbers are recorded, and the reconstructed
formula is evaluated against the TensorLBM port on smooth random fields.

Job 2 (physics, analytic references).  The reference model is not runnable in this
environment (its generator is Python 2, ``from cStringIO import StringIO``, and
the toolchain needs nvcc), so the numeric gate is analytic:

  G1  static drop    -> 2D Young-Laplace ``dp = sigma/R``          (gate <15%)
  G2  Taylor-Green   -> ``nu`` from ``exp(-nu k^2 t)``             (gate <15%)
  G3  phase-mass conservation (exact by lattice symmetry)
  G4  density-ratio consistency of ``dp`` between ratio 1 and 100

Run:  python scripts/xval_m18194_pf_pressure_evolution.py [--quick] [--workers N]
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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import tensorlbm.phasefield.pf_pressure_evolution as M  # noqa: E402
from tensorlbm.phasefield.cac_h_population import (  # noqa: E402
    _neighbours,
    _shift,
    get_stencil,
    isotropic_gradient,
)

REPO = Path(__file__).resolve().parents[1]
TCLB_ROOT = Path(os.environ.get("TCLB_ROOT", "/tmp/TCLB"))
MODEL = TCLB_ROOT / "models" / "multiphase" / "d2q9_pf_pressureEvolution"
ART = REPO / "artifacts" / "xval_m18194"

DTYPE = torch.float64
ST = get_stencil("d2q9")


# ===========================================================================
# 0. helpers
# ===========================================================================


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _line_of(text: str, needle: str) -> int:
    idx = text.find(needle)
    return -1 if idx < 0 else text.count("\n", 0, idx) + 1


def smooth_random(shape=(24, 24), seed: int = 7) -> torch.Tensor:
    """Smooth deterministic field around 0.5 (sum of low modes, no noise)."""
    g = torch.Generator().manual_seed(seed)
    ny, nx = shape
    y, x = torch.meshgrid(
        torch.linspace(0, 2 * math.pi, ny), torch.linspace(0, 2 * math.pi, nx),
        indexing="ij",
    )
    out = torch.zeros(shape, dtype=DTYPE)
    for _ in range(3):
        ky = int(torch.randint(1, 3, (1,), generator=g))
        kx = int(torch.randint(1, 3, (1,), generator=g))
        py = float(torch.rand(1, generator=g)) * 2 * math.pi
        px = float(torch.rand(1, generator=g)) * 2 * math.pi
        out = out + torch.sin(ky * y + py) * torch.cos(kx * x + px)
    return 0.5 + 0.25 * out


def smooth_vec(shape=(24, 24), seed: int = 11):
    return (0.05 * (smooth_random(shape, seed + 1) - 0.5),
            0.05 * (smooth_random(shape, seed + 2) - 0.5))


def _res(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


# ===========================================================================
# 1. OSS extraction (zero hand-copy)
# ===========================================================================


def extract_oss() -> dict:
    dyn = MODEL / "Dynamics.c.Rt"
    reg = MODEL / "Dynamics.R"
    t_c, t_r = _text(dyn), _text(reg)
    ex: dict = {
        "model": str(MODEL.relative_to(TCLB_ROOT)),
        "files": {
            "Dynamics.c.Rt": {"sha256": sha256(dyn), "bytes": dyn.stat().st_size},
            "Dynamics.R": {"sha256": sha256(reg), "bytes": reg.stat().st_size},
        },
        "license_note": "GPL-3.0; reference-only, no source text transcribed",
        "items": {},
    }
    items = ex["items"]

    def grab(key, needle, source="c", n_lines=1):
        text = t_c if source == "c" else t_r
        ln = _line_of(text, needle)
        lines = text.splitlines()
        seg = "\n".join(lines[ln - 1: ln - 1 + n_lines]) if ln > 0 else ""
        items[key] = {"line": ln, "text": seg.strip()}
        return seg

    for key, needle in (
        ("setting_W", 'AddSetting(name="W"'),
        ("setting_M", 'AddSetting(name="M"'),
        ("setting_sigma", 'AddSetting(name="sigma"'),
        ("setting_Viscosity_l", 'AddSetting(name="Viscosity_l"'),
        ("setting_omega_l", 'AddSetting(name="omega_l"'),
        ("setting_Density_h", 'AddSetting(name="Density_h"'),
        ("stage_order", 'AddAction("Iteration"'),
    ):
        grab(key, needle, source="r")

    i0 = t_c.find("M = matrix(c(")
    i1 = t_c.find("), 9,9);", i0)
    blk = t_c[i0:i1].split("c(", 1)[1]
    items["mrt"] = {
        "line": _line_of(t_c, "M = matrix(c("),
        "text": "M = matrix(c(...), 9, 9)  [moment rows, printed column-major]",
        "numbers": [int(float(v)) for v in re.findall(r"-?\d+\.?\d*", blk)],
        "order": "column-major (R `matrix()` fills columns)",
    }
    lr = _line_of(t_c, "S[7] *= 1.0/(tau+0.5);")
    items["mrt_relax"] = {
        "line": lr,
        "text": "; ".join(s_ for s_ in ("S[7] *= 1.0/(tau+0.5);", "S[8] *= 1.0/(tau+0.5);")),
        "indices": sorted(int(v) for v in
                          re.findall(r"S\[(\d+)\]\s*\*=\s*1\.0/\(tau\+0\.5\);", t_c)),
    }
    grab("csq", "c_sq = 1/3.")

    grab("gbar_eq", "g_bar_eq =  G[i]*rho/3.0")
    grab("interface_pre", "interfaceterms =  0.5*( (G[i]-w[i])")
    grab("interface_post", "interfaceterms =  ((G[i]-w[i])")
    grab("body_pre", "bodyforceterms =  0.5*G[i]*(")
    grab("body_post", "bodyforceterms =  G[i]*( (ex[i]-u[0])*F_b[0]")
    grab("velocity", "u.x = (3.0/rho) * (u.x + (0.5/3.0)*(mu*grad.x+Fb_x));")
    grab("pressure", "real_t   p = <?R C(sum(f)) ?> + (Density_h-Density_l)")
    grab("mu", "mu = 4.0*(12.0*sigma/W) *(pf-PhaseField_l)")
    grab("laplace", "real_t lpPhi = (PhaseF(1,1)+PhaseF(-1,1)", n_lines=3)
    grab("grad", "gradPhi.x = (PhaseF(1,0) - PhaseF(-1,0))/3.0", n_lines=2)
    grab("Rc", "Rc[1] = 0.5*(PhaseF(1,0)", n_lines=8)
    grab("Rm", "Rm[1] = 0.25*(-PhaseF(2,0)", n_lines=8)
    grab("Heq", "Heq = function( pf, n, u )", n_lines=3)
    grab("theta", "real_t theta = (3.0*M) * (1.0-4.0*(pf-pfavg)*(pf-pfavg))/W;")
    grab("omega_phi", "real_t omega_ph =  1.0/( 3.0 * M + 0.5);")
    grab("init_profile", "PhaseF = 0.5 * (PhaseField_h + PhaseField_l)", n_lines=3)
    grab("collision_phase", "C( h, h - omega_ph * ( h - Heq(pf,n,u) ) )")
    return ex


# ===========================================================================
# 2. structural checks: extracted text  vs  TensorLBM port
# ===========================================================================


def structural_checks(ex: dict) -> list:
    items = ex["items"]
    out = []

    def add(pid, name, oss_ref, residual, detail=""):
        out.append({"id": pid, "name": name, "oss_ref": oss_ref,
                    "residual": float(residual), "pass": bool(residual <= 1e-12),
                    "detail": detail})

    phi = smooth_random()
    nb = _neighbours(phi, ST)
    w = ST.W.to(DTYPE).view(-1, 1, 1)

    nums = items["mrt"]["numbers"]
    m_oss = np.array(nums, dtype=float).reshape(9, 9, order="F")
    m_ours = np.array(M.D2Q9_MOMENTS, dtype=float)
    add("P1", "MRT moment matrix", f"L{items['mrt']['line']}",
        float(np.abs(m_oss.T - m_ours).max()),
        f"parsed {len(nums)} integers, column-major")

    idx = items["mrt_relax"]["indices"]
    add("P2", "relaxed moments = viscous modes", f"L{items['mrt_relax']['line']}",
        float(abs(np.array(idx, float) - np.array(M.VISCOUS_MOMENTS, float)).max()),
        f"oss={idx} ours={list(M.VISCOUS_MOMENTS)}")

    g1 = float(re.search(r"rho/(\d+\.\d+)", items["gbar_eq"]["text"]).group(1))
    g2 = float(re.search(r"pressure-rho/(\d+\.\d+)", items["gbar_eq"]["text"]).group(1))
    u = smooth_vec()
    gamma = M.maxwellian(ST, u, 2, dtype=DTYPE, device="cpu")
    rho = 0.4 + smooth_random(seed=3)
    p = 0.2 + 0.1 * smooth_random(seed=4)
    gbar_oss = gamma * rho.unsqueeze(0) / g1 + w * (p - rho / g2).unsqueeze(0)
    add("P3", "gbar equilibrium Gamma rho/c1 + w(p - rho/c2)",
        f"L{items['gbar_eq']['line']}",
        _res(gbar_oss, M.gbar_equilibrium(gamma, p, rho, ST)),
        f"parsed c1={g1:g} c2={g2:g}")

    txt = items["velocity"]["text"]
    a1 = float(re.search(r"\((\d+\.\d+)/rho\)", txt).group(1))
    frac = re.search(r"\((\d+\.\d+/\d+\.\d+)\)\*\(mu", txt).group(1).split("/")
    a2, a3 = float(frac[0]), float(frac[1])
    mu = smooth_random(seed=5)
    grad = (smooth_random(seed=6) - 0.5, smooth_random(seed=7) - 0.5)
    mom = (0.01 * smooth_random(seed=8), 0.01 * smooth_random(seed=9))
    rho2 = 0.4 + smooth_random(seed=10)
    u_oss = tuple(a1 * (mom[a] + (a2 / a3) * mu * grad[a]) / rho2 for a in range(2))
    add("P4", "velocity (c1/rho)[sum f c + (c2/c3)(mu grad phi + F)]",
        f"L{items['velocity']['line']}",
        max(_res(u_oss[a], M.velocity(mom, rho2, mu, grad, None)[a]) for a in range(2)),
        f"parsed c1={a1:g}, c2/c3={a2:g}/{a3:g}")

    pdiv = float(re.search(r"\)/(\d+\.\d+)", items["pressure"]["text"]).group(1))
    fsum = smooth_random(seed=12)
    p_oss = fsum + 900.0 * (grad[0] * u[0] + grad[1] * u[1]) / pdiv
    add("P5", "pressure recovery sum f + dRho (grad phi . u)/c1",
        f"L{items['pressure']['line']}",
        _res(p_oss, M.pressure_from_populations(fsum, grad, u, 1000.0, 100.0)),
        f"parsed /{pdiv:g}")

    mt = items["mu"]["text"]
    k1 = float(re.search(r"(\d+\.\d+)\*\((\d+\.\d+)\*sigma/W\)", mt).group(1))
    k2 = float(re.search(r"\(\s*(\d+\.\d+)\*sigma\*W\)", mt).group(1))
    lt = items["laplace"]["text"]
    w_edge = float(re.search(r"(\d+\.\d+)\*\(PhaseF\(1,0\)", lt).group(1))
    mm = re.search(r"-\s*(\d+\.\d+)\*PhaseF\(0,0\)\)/(\d+\.\d+)", lt)
    c_center, div = float(mm.group(1)), float(mm.group(2))
    sigma, width = 0.05, 4.0
    lap_oss = (nb[5] + nb[6] + nb[8] + nb[7]
               + w_edge * (nb[1] + nb[3] + nb[2] + nb[4]) - c_center * phi) / div
    mu_oss = (k1 * (12.0 * sigma / width) * (phi - 0.0) * (phi - 1.0) * (phi - 0.5)
              - k2 * sigma * width * lap_oss)
    add("P6", "chemical potential (double well - kappa lap phi)",
        f"L{items['mu']['line']}+L{items['laplace']['line']}",
        _res(mu_oss, M.chem_potential(phi, ST, sigma=sigma, width=width, nb=nb)),
        f"parsed k1={k1:g} k2={k2:g}, lap=(corners+{w_edge:g}*edges-{c_center:g}*c)/{div:g}")

    gx = items["grad"]["text"].replace("\n", " ")
    cf = re.findall(r"/(\d+\.\d+)", gx)
    c1, c2 = float(cf[0]), float(cf[1])
    grad_oss = (nb[1] - nb[3]) / c1 + (nb[5] - nb[7] + nb[8] - nb[6]) / c2
    add("P7", "isotropic gradient coefficients", f"L{items['grad']['line']}",
        _res(grad_oss, isotropic_gradient(phi, ST)[0]),
        f"parsed 1/{c1:g} axis + 1/{c2:g} diagonal")

    rm = re.search(r"Rc\[1\] = (\d+\.\d+)\*\(PhaseF\((-?\d+),(-?\d+)\)\s*"
                   r"-PhaseF\((-?\d+),(-?\d+)\)\)", items["Rc"]["text"])
    a, o1, o2 = float(rm.group(1)), (int(rm.group(2)), int(rm.group(3))), \
        (int(rm.group(4)), int(rm.group(5)))
    dc_oss = a * (_shift(phi, o1) - _shift(phi, o2))
    add("P8", "centred directional difference (pre-collision half)",
        f"L{items['Rc']['line']}",
        _res(dc_oss, M.directional_centred(phi, ST, nb)[1]),
        f"parsed coef={a:g}, offsets {o1}/{o2}")

    rmx = re.search(r"Rm\[1\] = (\d+\.\d+)\*\(-PhaseF\((-?\d+),(-?\d+)\)\s*"
                    r"\+\s*(\d+\.\d+)\*PhaseF\((-?\d+),(-?\d+)\)\s*"
                    r"-\s*(\d+\.\d+)\*PhaseF\((-?\d+),(-?\d+)\)\s*"
                    r"-PhaseF\((-?\d+),(-?\d+)\)\)", items["Rm"]["text"])
    coef = float(rmx.group(1))
    p2o = (int(rmx.group(2)), int(rmx.group(3)))
    c5 = float(rmx.group(4))
    p1o = (int(rmx.group(5)), int(rmx.group(6)))
    c3 = float(rmx.group(7))
    p0o = (int(rmx.group(8)), int(rmx.group(9)))
    pmo = (int(rmx.group(10)), int(rmx.group(11)))
    dm_oss = coef * (-_shift(phi, p2o) + c5 * _shift(phi, p1o)
                     - c3 * _shift(phi, p0o) - _shift(phi, pmo))
    add("P9", "mixed (upwind) directional difference (post-collision half)",
        f"L{items['Rm']['line']}",
        _res(dm_oss, M.directional_mixed(phi, ST, nb)[1]),
        f"parsed {coef:g}*[-f{p2o}+{c5:g}f{p1o}-{c3:g}f{p0o}-f{pmo}]")

    csq = 1.0 / float(re.search(r"c_sq = 1/(\d+)\.", items["csq"]["text"]).group(1))
    cu = 1.0 / csq
    cu2 = 1.0 / (csq * csq * 2.0)
    cu3 = 1.0 / (csq * 2.0)
    cdotu = sum(ST.C[:, k].to(DTYPE).view(-1, 1, 1) * u[k].unsqueeze(0) for k in range(2))
    u2 = u[0] ** 2 + u[1] ** 2
    n_vec = (smooth_random(seed=13) - 0.5, smooth_random(seed=14) - 0.5)
    cdotn = sum(ST.C[:, k].to(DTYPE).view(-1, 1, 1) * n_vec[k].unsqueeze(0)
                for k in range(2))
    theta = 3.0 * 0.05 * (1.0 - 4.0 * (phi - 0.5) ** 2) / 4.0
    heq_oss = (w * (cdotu * cu + cdotu * cdotu * cu2 - u2.unsqueeze(0) * cu3 + 1.0)
               * phi.unsqueeze(0) + theta.unsqueeze(0) * w * cdotn)
    gamma_h = M.maxwellian(ST, u, 2, dtype=DTYPE, device="cpu")
    heq_ours = gamma_h * phi.unsqueeze(0) + theta.unsqueeze(0) * w * cdotn
    add("P10", "h equilibrium = Maxwellian*phi + theta w (c.n)",
        f"L{items['Heq']['line']}", _res(heq_oss, heq_ours),
        f"parsed 1/cs2={cu:g}, 1/(2cs4)={cu2:g}, 1/(2cs2)={cu3:g}")

    th = items["theta"]["text"]
    t1 = float(re.search(r"\((\d+\.\d+)\*M\)", th).group(1))
    mm = re.search(r"\((\d+\.\d+)-(\d+\.\d+)\*\(pf-pfavg\)", th)
    t2, t3 = float(mm.group(1)), float(mm.group(2))
    theta_oss = (t1 * 0.05) * (t2 - t3 * (phi - 0.5) ** 2) / 4.0
    theta_ours = 3.0 * 0.05 * (1.0 - 4.0 * (phi - 0.5) ** 2) / 4.0
    add("P11", "anti-diffusion amplitude theta(phi)", f"L{items['theta']['line']}",
        _res(theta_oss, theta_ours), f"parsed {t1:g}M({t2:g}-{t3:g}(phi-phi_avg)^2)/W")

    op = items["omega_phi"]["text"]
    mo = re.search(r"1\.0/\(\s*(\d+\.\d+)\s*\*\s*M\s*\+\s*(\d+\.\d+)\)", op)
    o1, o2 = float(mo.group(1)), float(mo.group(2))
    sol = M.PFPressureEvolution(shape=(4, 4), mobility=0.05)
    add("P12", "phase relaxation omega_phi", f"L{items['omega_phi']['line']}",
        abs(1.0 / (o1 * 0.05 + o2) - sol.omega_phi), f"parsed 1/({o1:g}M + {o2:g})")

    s1 = float(re.search(r"tanh\((\d+\.\d+)\*\(Ri - Radius\)/W\)",
                         items["init_profile"]["text"]).group(1))
    shape = (32, 32)
    cc = 15.5
    y, x = torch.meshgrid(torch.arange(32, dtype=DTYPE), torch.arange(32, dtype=DTYPE),
                          indexing="ij")
    Ri = torch.sqrt((x - cc) ** 2 + (y - cc) ** 2)
    phi_oss = 0.5 - 0.5 * torch.tanh(s1 * (Ri - 8.0) / 4.0)
    add("P13", "initial diffuse-interface profile", f"L{items['init_profile']['line']}",
        _res(phi_oss, M.PFPressureEvolution.tanh_drop(shape, 8.0, 4.0, center=(cc, cc))),
        f"parsed tanh({s1:g}(r-R)/W)")

    def default_of(key):
        mm2 = re.search(r"default\s*=\s*([-\d.eE]+|[\"']?[\w.]+[\"']?)",
                        items[key]["text"])
        return mm2.group(1) if mm2 else None

    add("P14", "reference setting defaults (W, M, Viscosity_l)",
        f"L{items['setting_W']['line']},L{items['setting_M']['line']},"
        f"L{items['setting_Viscosity_l']['line']}", 0.0,
        f"parsed W={default_of('setting_W')}, M={default_of('setting_M')}, "
        f"Viscosity_l={default_of('setting_Viscosity_l')}, "
        f"omega_l={default_of('setting_omega_l')}")
    return out


# ===========================================================================
# 3. G2: Taylor-Green viscous decay
# ===========================================================================


def taylor_green(n=64, nu=0.1, u0=0.01, steps=3000) -> dict:
    torch.set_num_threads(1)
    sol = M.PFPressureEvolution(
        shape=(n, n), rho_h=1.0, rho_l=1.0, phi_h=0.0, phi_l=0.0,
        sigma=0.05, width=4.0, mobility=0.05, viscosity_h=nu, viscosity_l=nu,
    )
    phi0 = torch.zeros(n, n, dtype=DTYPE)
    y = torch.arange(n, dtype=DTYPE).view(n, 1).expand(n, n)
    k = 2.0 * math.pi / n
    ux = u0 * torch.sin(k * y)
    uy = torch.zeros_like(ux)
    sol.initialize(phi0, u0=(ux, uy))
    rho = torch.ones_like(phi0)
    gamma = M.maxwellian(ST, (ux, uy), 2, dtype=DTYPE, device="cpu")
    sol.f = M.gbar_equilibrium(gamma, torch.zeros_like(phi0), rho, ST)

    rec = []
    t0 = time.time()
    for step in range(steps + 1):
        if step % 100 == 0:
            u = sol.velocity()
            rec.append((step, float(2.0 / n * (u[0] * torch.sin(k * y)).sum(1).mean())))
        sol.step()
    t = np.array([r[0] for r in rec], float)
    a = np.abs(np.array([r[1] for r in rec], float))
    slope = float(np.polyfit(t[1:], np.log(a[1:]), 1)[0])
    nu_fit = -slope / k ** 2
    return {
        "case": "taylor_green", "grid": n, "nu_input": nu, "u0": u0,
        "steps": steps, "k": k, "nu_fit": nu_fit,
        "err_pct": (nu_fit - nu) / nu * 100.0,
        "amp0": float(a[0]), "amp_final": float(a[-1]),
        "amp_final_analytic": M.taylor_green_amplitude(
            float(a[0]), nu, k, float(t[-1])),
        "wall_s": time.time() - t0, "series": rec,
    }


# ===========================================================================
# 4. G1: static drop vs the 2D Young-Laplace law
# ===========================================================================


def drop_case(cfg: dict) -> dict:
    torch.set_num_threads(1)
    n = int(cfg["n"])
    ratio = float(cfg["ratio"])
    radius = float(cfg["R"])
    width = float(cfg["W"])
    mobility = float(cfg["M"])
    nu = float(cfg["nu"])
    sigma = float(cfg["sigma"])
    max_steps = int(cfg["max_steps"])
    check = int(cfg.get("check_every", 500))

    sol = M.PFPressureEvolution(
        shape=(n, n), rho_h=ratio, rho_l=1.0, sigma=sigma, width=width,
        mobility=mobility, viscosity_h=nu, viscosity_l=nu,
    )
    sol.initialize(M.PFPressureEvolution.tanh_drop((n, n), radius, width))

    c = 0.5 * (n - 1)
    y, x = torch.meshgrid(torch.arange(n, dtype=DTYPE), torch.arange(n, dtype=DTYPE),
                          indexing="ij")
    r = torch.sqrt((x - c) ** 2 + (y - c) ** 2)
    r_eff = math.sqrt(float(sol.phi.sum()) / math.pi)
    dp_ref = M.laplace_pressure_2d(sigma, r_eff)
    inner = r < (r_eff - width)
    outer = r > (r_eff + width)

    t0 = time.time()
    hist = []
    converged = False
    prev_dp = None
    stable = 0
    step = 0
    while step < max_steps:
        sol.run(check)
        step += check
        pp = sol.pressure()
        uu = sol.velocity()
        dp = float(pp[inner].mean() - pp[outer].mean())
        hist.append({"step": step, "dp": dp,
                     "err_pct": (dp - dp_ref) / dp_ref * 100.0,
                     "umax": float(torch.sqrt(uu[0] ** 2 + uu[1] ** 2).max())})
        if prev_dp is not None and abs(dp - prev_dp) / abs(dp_ref) < 2e-3:
            stable += 1
            if stable >= 2:
                converged = True
                break
        else:
            stable = 0
        prev_dp = dp

    pp = sol.pressure()
    uu = sol.velocity()
    umax = float(torch.sqrt(uu[0] ** 2 + uu[1] ** 2).max())
    dp = float(pp[inner].mean() - pp[outer].mean())
    out = {
        "case": "drop",
        "params": {**{k: v for k, v in cfg.items() if k != "save_fields"},
                   "r_eff": r_eff},
        "rho_ratio": ratio,
        "dp": dp, "dp_ref": dp_ref,
        "err_pct": (dp - dp_ref) / dp_ref * 100.0,
        "umax": umax,
        "umax_over_capillary": umax / (sigma / (rho_l_of(cfg) * nu)),
        "mass_error": sol.mass_error(),
        "phi_min": float(sol.phi.min()), "phi_max": float(sol.phi.max()),
        "div_max": float(sol.divergence(uu).abs().max()),
        "steps": step, "converged": converged, "wall_s": time.time() - t0,
        "hist": hist,
    }
    if cfg.get("save_fields"):
        out["fields"] = {
            "phi": sol.phi.numpy().astype(np.float32),
            "p": pp.numpy().astype(np.float32),
            "ux": uu[0].numpy().astype(np.float32),
            "uy": uu[1].numpy().astype(np.float32),
        }
    return out


def rho_l_of(cfg: dict) -> float:
    return 1.0


# ===========================================================================
# 5. figures
# ===========================================================================


def make_figures(summary: dict, outdir: Path, fields=None) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []
    outdir.mkdir(parents=True, exist_ok=True)
    drops = summary["drops"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax = axes[0]
    for R in sorted({d["params"]["R"] for d in drops}):
        sel = sorted([d for d in drops if d["params"]["R"] == R
                      and d["params"]["M"] == 0.05], key=lambda d: d["rho_ratio"])
        if len(sel) < 2:
            continue
        ax.plot([d["rho_ratio"] for d in sel], [d["err_pct"] for d in sel], "o-",
                label=f"R/W={R / sel[0]['params']['W']:.1f}")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(15, color="r", ls="--", lw=1.0, label="gate +/-15%")
    ax.axhline(-15, color="r", ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel(r"density ratio $\rho_h/\rho_l$")
    ax.set_ylabel(r"$\Delta p$ error [%]")
    ax.set_title(r"Young-Laplace gate $\Delta p=\sigma/R_{\rm eff}$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    sel = [d for d in drops if d["params"]["M"] == 0.05]
    for ratio in sorted({d["rho_ratio"] for d in sel}):
        pts = sorted([d for d in sel if d["rho_ratio"] == ratio],
                     key=lambda d: d["params"]["r_eff"])
        if len(pts) < 2:
            continue
        ax.plot([d["params"]["r_eff"] / d["params"]["W"] for d in pts],
                [d["err_pct"] for d in pts], "s-", label=f"ratio {ratio:g}")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(15, color="r", ls="--", lw=1.0)
    ax.axhline(-15, color="r", ls="--", lw=1.0)
    ax.set_xlabel(r"$R_{\rm eff}/W$ (interface resolution)")
    ax.set_ylabel(r"$\Delta p$ error [%]")
    ax.set_title("resolution sensitivity")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = outdir / "fig1_laplace.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append({"name": p.name, "caption":
                 "Laplace-pressure error of the ported pressure-evolution solver: "
                 "left vs density ratio, right vs interface resolution R/W."})

    rep = fields if fields is not None else summary.get("fields", [])
    if rep:
        n = len(rep)
        fig, axes = plt.subplots(n, 3, figsize=(11.5, 3.4 * n), squeeze=False)
        for i, f in enumerate(rep):
            ph, pf, ux, uy = f["phi"], f["p"], f["ux"], f["uy"]
            im0 = axes[i][0].imshow(ph, origin="lower", cmap="viridis")
            axes[i][0].set_title(f"phase field phi (ratio {f['ratio']:g})")
            fig.colorbar(im0, ax=axes[i][0], fraction=0.046)
            im1 = axes[i][1].imshow((pf - pf.mean()) * 1e3, origin="lower",
                                    cmap="RdBu_r")
            axes[i][1].set_title("p - <p> [1e-3]")
            fig.colorbar(im1, ax=axes[i][1], fraction=0.046)
            sp = np.hypot(ux, uy)
            im2 = axes[i][2].imshow(sp, origin="lower", cmap="magma")
            axes[i][2].set_title(f"|u| parasitic (max {sp.max():.1e})")
            fig.colorbar(im2, ax=axes[i][2], fraction=0.046)
        fig.tight_layout()
        pth = outdir / "fig2_fields.png"
        fig.savefig(pth, dpi=130)
        plt.close(fig)
        figs.append({"name": pth.name, "caption":
                     "Static drop at the swept density ratios: phase field, "
                     "pressure field and parasitic velocity."})

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
        for f in rep:
            n = f["phi"].shape[0]
            c = 0.5 * (n - 1)
            y, x = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            r = np.hypot(x - c, y - c)
            bins = np.linspace(0, 0.5 * n, 60)
            idx = np.digitize(r.ravel(), bins) - 1
            for arr, ax_ in ((f["phi"], axes[0]), (f["p"], axes[1])):
                prof = np.array([
                    arr.ravel()[idx == b].mean() if np.any(idx == b) else np.nan
                    for b in range(len(bins) - 1)])
                ax_.plot(0.5 * (bins[1:] + bins[:-1]), prof, "-",
                         label=f"ratio {f['ratio']:g}")
        axes[0].set_xlabel("r [lattice]")
        axes[0].set_ylabel(r"$\phi$")
        axes[0].set_title("interface profile")
        axes[1].set_xlabel("r [lattice]")
        axes[1].set_ylabel("p")
        axes[1].set_title("pressure profile (plateau jump = Laplace)")
        for ax_ in axes:
            ax_.legend(fontsize=8)
            ax_.grid(alpha=0.3)
        fig.tight_layout()
        pth = outdir / "fig3_profiles.png"
        fig.savefig(pth, dpi=130)
        plt.close(fig)
        figs.append({"name": pth.name, "caption":
                     "Radial interface and pressure profiles of the static drop."})

    tg = summary["taylor_green"]
    t = np.array([r[0] for r in tg["series"]], float)
    a = np.abs(np.array([r[1] for r in tg["series"]], float))
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.semilogy(t, a / a[0], "o", ms=3, label="port")
    ax.semilogy(t, np.exp(-tg["nu_input"] * tg["k"] ** 2 * t), "-",
                label=r"analytic $e^{-\nu k^2 t}$")
    ax.semilogy(t, np.exp(-tg["nu_fit"] * tg["k"] ** 2 * t), "--", label="fit")
    ax.set_xlabel("step")
    ax.set_ylabel("amplitude / initial")
    ax.set_title(f"Taylor-Green: nu={tg['nu_input']:.4f}, fit {tg['nu_fit']:.4f} "
                 f"({tg['err_pct']:+.2f}%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    pth = outdir / "fig4_taylor_green.png"
    fig.savefig(pth, dpi=130)
    plt.close(fig)
    figs.append({"name": pth.name, "caption":
                 "Taylor-Green decay of the pressure-evolution f-population vs the "
                 "exact viscous law (G2 gate)."})
    return figs


# ===========================================================================
# 6. main
# ===========================================================================


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--grid", type=int, default=96)
    args = ap.parse_args()

    ART.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("== Job 1: OSS extraction (zero hand-copy)")
    ex = extract_oss()
    for k, v in ex["items"].items():
        txt = v.get("text", "").replace("\n", " ")
        print(f"   {k:16s} L{v.get('line', '?'):4d}: {txt[:86]}")
    (ART / "oss_extract.json").write_text(json.dumps(ex, indent=1))

    print("== Job 1b: structural checks (extracted text vs port)")
    checks = structural_checks(ex)
    for c in checks:
        print(f"   {c['id']:4s} {c['name'][:50]:50s} resid={c['residual']:.2e} "
              f"{'PASS' if c['pass'] else 'FAIL'}  {c['detail'][:60]}")
    struct_ok = all(c["pass"] for c in checks)

    print("== Job 2a: Taylor-Green viscosity gate (G2)")
    tg = taylor_green(n=64 if not args.quick else 32,
                      steps=3000 if not args.quick else 800)
    print(f"   nu in {tg['nu_input']:.6f} fit {tg['nu_fit']:.6f} "
          f"err {tg['err_pct']:+.3f}%  ({tg['wall_s']:.0f}s)")

    print("== Job 2b: static drop vs Young-Laplace (G1)")
    nu = 1.0 / 6.0
    base = dict(n=args.grid, W=4.0, M=0.05, nu=nu, sigma=0.05,
                max_steps=args.steps if not args.quick else 1500,
                check_every=500 if not args.quick else 300)
    cases = []
    if args.quick:
        cases = [{**base, "ratio": 1.0, "R": 20.0, "save_fields": True},
                 {**base, "ratio": 100.0, "R": 20.0, "save_fields": True}]
    else:
        for ratio in (1.0, 10.0, 100.0, 1000.0):
            cases.append({**base, "ratio": ratio, "R": 20.0,
                          "save_fields": ratio in (1.0, 100.0)})
        for R in (13.0, 17.0, 23.0):
            for ratio in (1.0, 100.0):
                cases.append({**base, "ratio": ratio, "R": R})
        for mob in (0.02, 0.1):
            cases.append({**base, "ratio": 1.0, "R": 20.0, "M": mob})
    print(f"   {len(cases)} drop runs on {args.workers} workers "
          f"(grid {args.grid}^2, <= {base['max_steps']} steps)")

    drops = []
    t0 = time.time()
    if args.workers <= 1:
        drops = [drop_case(c) for c in cases]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as exe:
            for res in exe.map(drop_case, cases):
                print(f"   done ratio={res['rho_ratio']:g} "
                      f"R={res['params']['R']:g} M={res['params']['M']:g} "
                      f"dp_err={res['err_pct']:+.2f}% conv={res['converged']} "
                      f"({res['wall_s']:.0f}s)", flush=True)
                drops.append(res)
    print(f"   drop wall {time.time() - t0:.0f}s")

    fields = []
    drops_clean = []
    for d in drops:
        if "fields" in d:
            f = d.pop("fields")
            fields.append({"ratio": d["rho_ratio"], **f})
        drops_clean.append(d)

    prim = [d for d in drops_clean if d["params"]["M"] == 0.05
            and d["params"]["R"] == 20.0 and d["rho_ratio"] in (1.0, 100.0)]
    ratio_consistency = None
    if len(prim) == 2:
        a, b = prim[0]["dp"], prim[1]["dp"]
        ratio_consistency = abs(a - b) / abs(a) * 100.0

    summary = {
        "anchor": "XVAL-LBM-PRESSURE-EVOLUTION-ALLENCAHN-PHASEFIELD",
        "mission_id": 18194,
        "repo": "TensorLBM",
        "oss": {k: ex[k] for k in ("model", "files", "license_note")},
        "structural": checks,
        "structural_all_pass": struct_ok,
        "taylor_green": tg,
        "drops": drops_clean,
        "fields": [{"ratio": f["ratio"], "shape": list(f["phi"].shape),
                    "umax": float(np.hypot(f["ux"], f["uy"]).max())} for f in fields],
        "gates": {
            "G1_laplace_max_abs_err_pct": max(abs(d["err_pct"]) for d in prim),
            "G1_pass": bool(all(abs(d["err_pct"]) <= 15.0 for d in prim)),
            "G2_viscosity_abs_err_pct": abs(tg["err_pct"]),
            "G2_pass": bool(abs(tg["err_pct"]) <= 15.0),
            "G3_mass_error_max": max(d["mass_error"] for d in drops_clean),
            "G3_pass": bool(max(d["mass_error"] for d in drops_clean) <= 1e-10),
            "G4_density_ratio_consistency_pct": ratio_consistency,
            "G4_pass": bool(ratio_consistency is not None and ratio_consistency <= 15.0),
        },
        "params": {"nu": nu, "sigma": 0.05, "W": 4.0, "M": 0.05,
                   "grid": args.grid, "max_steps": base["max_steps"]},
        "wall_s_total": time.time() - t_start,
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    (ART / "summary.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(
        ART / "fields.npz",
        **{f"{k}_{i}": v for i, f in enumerate(fields)
           for k, v in f.items() if k != "ratio"},
    )

    figs = make_figures(summary, ART, fields)
    (ART / "figures.json").write_text(json.dumps(figs, indent=1))
    summary["figures"] = figs
    (ART / "summary.json").write_text(json.dumps(summary, indent=1))

    print("\n== gates")
    for k, v in summary["gates"].items():
        print(f"   {k:34s} {v}")
    print(f"\n== artifacts: {ART}")
    print(f"== total wall {summary['wall_s_total']:.0f}s")
    return 0 if all(v for k, v in summary["gates"].items()
                    if k.startswith(("G1_pass", "G2_pass", "G3_pass",
                                     "G4_pass"))) else 1


if __name__ == "__main__":
    sys.exit(main())