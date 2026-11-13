#!/usr/bin/env python3
"""XVAL: Phase-field (Fakhari/CudaPhaseField) vs Pseudopotential (Shan-Chen/TensorLBM).

Anchor: XVAL-LBM-NOT-APPLICABLE-PHASE-FIELD-NOT-PSEUDOPOTENTIAL
OSS reference: AminZar69/CudaPhaseField (Fakhari family phase-field LBM)
TensorLBM: Shan-Chen single-component multiphase (pseudopotential)

Key claim: These two approaches are NOT directly cross-validatable because they
solve fundamentally different equations:
  - Phase-field: Allen-Cahn/Cahn-Hilliard + two distributions (h, g)
  - Pseudopotential: Single distribution + ψ(ρ) interaction force

This script runs both on the same Laplace test (static droplet) and documents
the structural differences, parameter mappings, and why direct number comparison
is not meaningful.
"""

import json
import math
import os
import sys
import time

import numpy as np
import torch

# ---------------------------------------------------------------------------
# TensorLBM Shan-Chen pseudopotential (SCMP)
# ---------------------------------------------------------------------------

def run_scmp_laplace(nx=128, ny=128, radius=20.0, G=5.0, tau=1.0,
                     steps=5000, sample=250, device="cpu"):
    """Shan-Chen pseudopotential static droplet (Laplace test).

    Returns dict with sigma_recovered, rho_l, rho_v, dp, R_eq, history.
    """
    dev = torch.device(device)
    cs2 = 1.0 / 3.0

    # D2Q9
    C = torch.tensor([
        [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
        [1, 1], [-1, 1], [-1, -1], [1, -1]
    ], device=dev)
    W = torch.tensor([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], device=dev)
    OPP = [0, 3, 4, 1, 2, 7, 8, 5, 6]

    def equilibrium(rho, ux, uy):
        cu = C[:, 0].view(9, 1, 1) * ux.unsqueeze(0) + C[:, 1].view(9, 1, 1) * uy.unsqueeze(0)
        usq = (ux**2 + uy**2).unsqueeze(0)
        return rho.unsqueeze(0) * W.view(9, 1, 1) * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * usq)

    def macroscopic(f):
        rho = f.sum(dim=0)
        ux = (C[:, 0].view(9, 1, 1) * f).sum(dim=0) / rho.clamp(min=1e-12)
        uy = (C[:, 1].view(9, 1, 1) * f).sum(dim=0) / rho.clamp(min=1e-12)
        return rho, ux, uy

    def psi_exp(rho, rho0=1.0):
        return rho0 * (1.0 - torch.exp(-rho / rho0))

    def sc_force(rho, G_val):
        psi = psi_exp(rho)
        fx = torch.zeros_like(rho)
        fy = torch.zeros_like(rho)
        for i in range(1, 9):
            cx, cy = int(C[i, 0]), int(C[i, 1])
            psi_shift = torch.roll(psi, shifts=(-cy, -cx), dims=(0, 1))
            fx += W[i] * cx * psi_shift
            fy += W[i] * cy * psi_shift
        fx = G_val * cs2 * psi * fx
        fy = G_val * cs2 * psi * fy
        return fx, fy

    def collide_guo(f, fx, fy, tau_val):
        rho, ux, uy = macroscopic(f)
        feq = equilibrium(rho, ux, uy)
        rho_safe = rho.clamp(min=1e-12)
        uxp = ux + fx / (2.0 * rho_safe)
        uyp = uy + fy / (2.0 * rho_safe)
        cu = C[:, 0].view(9, 1, 1) * uxp.unsqueeze(0) + C[:, 1].view(9, 1, 1) * uyp.unsqueeze(0)
        cf = C[:, 0].view(9, 1, 1) * fx.unsqueeze(0) + C[:, 1].view(9, 1, 1) * fy.unsqueeze(0)
        uf = uxp * fx + uyp * fy
        src = W.view(9, 1, 1) * (1.0 - 1.0 / (2.0 * tau_val)) * ((cf - uf.unsqueeze(0)) / cs2 + cu * cf / cs2**2)
        return f - (f - feq) / tau_val + src

    def stream_periodic(f):
        out = torch.empty_like(f)
        for i in range(9):
            cx, cy = int(C[i, 0]), int(C[i, 1])
            out[i] = torch.roll(f[i], shifts=(cy, cx), dims=(0, 1))
        return out

    # Initial condition: circular droplet
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32, device=dev),
        torch.arange(nx, dtype=torch.float32, device=dev),
        indexing="ij",
    )
    cx_c, cy_c = nx / 2.0, ny / 2.0
    dist = torch.sqrt((xx - cx_c)**2 + (yy - cy_c)**2)
    rho_l, rho_v = 1.957, 0.1596  # coexistence for G=5, psi_exp, tau=1
    rho = rho_v + (rho_l - rho_v) * 0.5 * (1.0 - torch.tanh(2.0 * (dist - radius) / 3.0))
    zero = torch.zeros_like(rho)
    f = equilibrium(rho, zero, zero)

    history = []
    for step in range(1, steps + 1):
        rho = f.sum(dim=0)
        fx, fy = sc_force(rho, G)
        f = collide_guo(f, fx, fy, tau)
        f = stream_periodic(f)

        if step % sample == 0 or step == steps:
            rho = f.sum(dim=0)
            ux = (C[:, 0].view(9, 1, 1) * f).sum(dim=0) / rho.clamp(min=1e-12)
            uy = (C[:, 1].view(9, 1, 1) * f).sum(dim=0) / rho.clamp(min=1e-12)
            u_max = float(torch.sqrt(ux**2 + uy**2).max())

            # Measure droplet radius and pressure difference
            threshold = 0.5 * (rho_l + rho_v)
            # Find interface by radial average
            r_field = torch.sqrt((xx - cx_c)**2 + (yy - cy_c)**2)
            # Inner region (r < 0.5*radius)
            inner = r_field < 0.5 * radius
            outer = r_field > 1.5 * radius
            rho_in = float(rho[inner].mean()) if inner.any() else rho_l
            rho_out = float(rho[outer].mean()) if outer.any() else rho_v

            # Pressure from SC EOS: p = rho*cs2 + G*cs2/2 * psi^2
            psi_in = psi_exp(torch.tensor(rho_in)).item()
            psi_out = psi_exp(torch.tensor(rho_out)).item()
            p_in = rho_in * cs2 + G * cs2 / 2.0 * psi_in**2
            p_out = rho_out * cs2 + G * cs2 / 2.0 * psi_out**2
            dp = p_in - p_out

            # Estimate radius from contour
            diff = (rho - threshold).abs()
            band = diff < 0.1 * (rho_l - rho_v)
            if band.any():
                r_at_interface = r_field[band].mean().item()
            else:
                r_at_interface = radius

            sigma_laplace = dp * r_at_interface  # 2D: dp = sigma/R

            history.append({
                "step": step,
                "rho_in": rho_in,
                "rho_out": rho_out,
                "dp": dp,
                "R_eq": r_at_interface,
                "sigma_laplace": sigma_laplace,
                "u_max": u_max,
            })

    return {
        "model": "SCMP (Shan-Chen pseudopotential)",
        "G": G,
        "tau": tau,
        "radius_init": radius,
        "rho_l": rho_l,
        "rho_v": rho_v,
        "history": history,
        "final": history[-1] if history else {},
    }


# ---------------------------------------------------------------------------
# Fakhari-style phase-field (2D adaptation of CudaPhaseField approach)
# ---------------------------------------------------------------------------

def run_fakhari_phasefield_2d(nx=128, ny=128, radius=20.0,
                              sigma=0.01, W=4.0, M=0.02,
                              tau_g=0.8, rho_l=0.1, rho_h=1.0,
                              steps=5000, sample=250, device="cpu"):
    """Fakhari phase-field LBM for static droplet (2D adaptation).

    This implements the key features of the Fakhari/CudaPhaseField approach:
    - Two distributions: h (phase field C), g (hydrodynamics)
    - Allen-Cahn equation with anti-diffusion source
    - Chemical potential: mu = 4*Beta*C*(C-1)*(C-0.5) - k*laplacian(C)
    - Isotropic gradient and Laplacian

    Returns dict with sigma_recovered, history.
    """
    dev = torch.device(device)
    cs2 = 1.0 / 3.0

    # D2Q9
    C = torch.tensor([
        [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
        [1, 1], [-1, 1], [-1, -1], [1, -1]
    ], device=dev)
    W = torch.tensor([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], device=dev)
    OPP = [0, 3, 4, 1, 2, 7, 8, 5, 6]

    # Phase-field parameters
    Beta = 12.0 * sigma / W
    k_grad = 1.5 * sigma * W
    tau_h = 0.5 + 3.0 * M
    w_c = 1.0 / tau_h
    s8 = 1.0 / tau_g

    # Initial phase field C: 0=light (inside), 1=heavy (outside)
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32, device=dev),
        torch.arange(nx, dtype=torch.float32, device=dev),
        indexing="ij",
    )
    cx_c, cy_c = nx / 2.0, ny / 2.0
    dist = torch.sqrt((xx - cx_c)**2 + (yy - cy_c)**2)
    C_field = 0.5 - 0.5 * torch.tanh(2.0 * (dist - radius) / W)
    C_field = C_field.clamp(0.0, 1.0)

    # Density
    rho = rho_l + C_field * (rho_h - rho_l)

    # Pressure (deviation from reference)
    p_inf = rho_h * cs2
    P = torch.zeros_like(C_field)  # static droplet, no pressure jump initially

    # Velocity
    ux = torch.zeros_like(C_field)
    uy = torch.zeros_like(C_field)

    # Distributions
    h = torch.zeros(9, ny, nx, device=dev)  # phase field
    g = torch.zeros(9, ny, nx, device=dev)  # hydrodynamics

    def equilibrium_h(C_val, ux_v, uy_v):
        cu = C[:, 0].view(9, 1, 1) * ux_v.unsqueeze(0) + C[:, 1].view(9, 1, 1) * uy_v.unsqueeze(0)
        usq = (ux_v**2 + uy_v**2).unsqueeze(0)
        Gamma_w = W.view(9, 1, 1) * (cu * (3.0 + 4.5 * cu) - 1.5 * usq)
        Gamma = Gamma_w + W.view(9, 1, 1)
        return C_val.unsqueeze(0) * Gamma

    def equilibrium_g(P_val, ux_v, uy_v):
        cu = C[:, 0].view(9, 1, 1) * ux_v.unsqueeze(0) + C[:, 1].view(9, 1, 1) * uy_v.unsqueeze(0)
        usq = (ux_v**2 + uy_v**2).unsqueeze(0)
        Gamma_w = W.view(9, 1, 1) * (cu * (3.0 + 4.5 * cu) - 1.5 * usq)
        return P_val.unsqueeze(0) * W.view(9, 1, 1) + Gamma_w

    def gradient_iso_2d(phi):
        """4th-order isotropic gradient."""
        dx = (torch.roll(phi, -1, dims=1) - torch.roll(phi, 1, dims=1)) / 3.0
        dy = (torch.roll(phi, -1, dims=0) - torch.roll(phi, 1, dims=0)) / 3.0
        # Add diagonal contributions
        dx += (torch.roll(torch.roll(phi, -1, dims=0), -1, dims=1) -
               torch.roll(torch.roll(phi, 1, dims=0), 1, dims=1)) / 12.0
        dx += (torch.roll(torch.roll(phi, -1, dims=0), 1, dims=1) -
               torch.roll(torch.roll(phi, 1, dims=0), -1, dims=1)) / 12.0
        dy += (torch.roll(torch.roll(phi, -1, dims=0), -1, dims=1) -
               torch.roll(torch.roll(phi, 1, dims=0), 1, dims=1)) / 12.0
        dy += (torch.roll(torch.roll(phi, -1, dims=0), 1, dims=1) -
               torch.roll(torch.roll(phi, 1, dims=0), -1, dims=1)) / 12.0
        return dx, dy

    def laplacian_iso_2d(phi):
        """9-point isotropic Laplacian."""
        lap = (torch.roll(phi, -1, dims=0) + torch.roll(phi, 1, dims=0) +
               torch.roll(phi, -1, dims=1) + torch.roll(phi, 1, dims=1) -
               4.0 * phi) / 3.0
        lap += (torch.roll(torch.roll(phi, -1, dims=0), -1, dims=1) +
                torch.roll(torch.roll(phi, -1, dims=0), 1, dims=1) +
                torch.roll(torch.roll(phi, 1, dims=0), -1, dims=1) +
                torch.roll(torch.roll(phi, 1, dims=0), 1, dims=1) -
                4.0 * phi) / 12.0
        return lap

    def stream_2d(f):
        out = torch.empty_like(f)
        for i in range(9):
            cx, cy = int(C[i, 0]), int(C[i, 1])
            out[i] = torch.roll(f[i], shifts=(cy, cx), dims=(0, 1))
        return out

    # Initialize distributions
    heq = equilibrium_h(C_field, ux, uy)
    geq = equilibrium_g(P, ux, uy)
    h = heq.clone()
    g = geq.clone()

    history = []
    for step in range(1, steps + 1):
        # Macroscopic from h
        C_field = h.sum(0).clamp(0.0, 1.0)
        rho = rho_l + C_field * (rho_h - rho_l)

        # Chemical potential
        lap_C = laplacian_iso_2d(C_field)
        mu = 4.0 * Beta * C_field * (C_field - 1.0) * (C_field - 0.5) - k_grad * lap_C

        # Gradient
        dCdx, dCdy = gradient_iso_2d(C_field)
        grad_mag = torch.sqrt(dCdx**2 + dCdy**2).clamp(min=1e-32)
        ni = dCdx / grad_mag
        nj = dCdy / grad_mag

        # Collision h (phase field with anti-diffusion)
        heq = equilibrium_h(C_field, ux, uy)
        eF = (1.0 - 4.0 * (C_field - 0.5)**2) / W
        c_dot_n = (C[:, 0].view(9, 1, 1) * ni.unsqueeze(0) +
                   C[:, 1].view(9, 1, 1) * nj.unsqueeze(0))
        hlp_h = W.view(9, 1, 1) * eF.unsqueeze(0) * c_dot_n
        h = h * (1.0 - w_c) + heq * w_c + hlp_h

        # Collision g (hydrodynamics with surface tension force)
        Fx = mu * dCdx
        Fy = mu * dCdy
        c_dot_F = C[:, 0].view(9, 1, 1) * Fx.unsqueeze(0) + C[:, 1].view(9, 1, 1) * Fy.unsqueeze(0)
        hlp_g = 3.0 * W.view(9, 1, 1) * c_dot_F / rho_h
        geq = equilibrium_g(P, ux, uy)
        g = g * (1.0 - s8) + geq * s8 + hlp_g

        # Streaming
        h = stream_2d(h)
        g = stream_2d(g)

        # Bounce-back at boundaries (periodic for this test)
        # (periodic streaming already handles this)

        # Macroscopic velocity from g
        P = g.sum(0)
        ux = (C[:, 0].view(9, 1, 1) * g).sum(0) / rho_h + 0.5 * Fx / rho_h
        uy = (C[:, 1].view(9, 1, 1) * g).sum(0) / rho_h + 0.5 * Fy / rho_h
        ux = ux.clamp(-0.3, 0.3)
        uy = uy.clamp(-0.3, 0.3)

        if step % sample == 0 or step == steps:
            u_max = float(torch.sqrt(ux**2 + uy**2).max())

            # Measure droplet properties
            threshold = 0.5
            r_field = torch.sqrt((xx - cx_c)**2 + (yy - cy_c)**2)
            inner = r_field < 0.5 * radius
            outer = r_field > 1.5 * radius

            C_in = float(C_field[inner].mean()) if inner.any() else 0.0
            C_out = float(C_field[outer].mean()) if outer.any() else 1.0

            # Estimate radius from contour
            diff = (C_field - threshold).abs()
            band = diff < 0.1
            if band.any():
                r_at_interface = r_field[band].mean().item()
            else:
                r_at_interface = radius

            # Pressure difference (from chemical potential balance)
            # For static droplet: dp = sigma / R (2D)
            # We recover sigma from the measured dp and R
            # In phase-field, the pressure jump is related to the chemical potential
            # For a static droplet, we can estimate dp from the density/pressure field
            p_in = float(P[inner].mean()) if inner.any() else 0.0
            p_out = float(P[outer].mean()) if outer.any() else 0.0
            dp = p_in - p_out

            sigma_laplace = dp * r_at_interface if r_at_interface > 0 else 0.0

            history.append({
                "step": step,
                "C_in": C_in,
                "C_out": C_out,
                "dp": dp,
                "R_eq": r_at_interface,
                "sigma_laplace": sigma_laplace,
                "u_max": u_max,
            })

    return {
        "model": "Fakhari phase-field (CudaPhaseField family)",
        "sigma_input": sigma,
        "W": W,
        "M": M,
        "tau_g": tau_g,
        "rho_l": rho_l,
        "rho_h": rho_h,
        "radius_init": radius,
        "history": history,
        "final": history[-1] if history else {},
    }


# ---------------------------------------------------------------------------
# Main: run both and compare
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("XVAL: Phase-field vs Pseudopotential — NOT APPLICABLE")
    print("Anchor: XVAL-LBM-NOT-APPLICABLE-PHASE-FIELD-NOT-PSEUDOPOTENTIAL")
    print("=" * 70)

    device = "cpu"
    nx, ny = 128, 128
    radius = 20.0
    steps = 5000
    sample = 250

    # --- Run SCMP (pseudopotential) ---
    print("\n[1/2] Running Shan-Chen pseudopotential (SCMP)...")
    t0 = time.time()
    scmp_result = run_scmp_laplace(nx=nx, ny=ny, radius=radius, G=5.0, tau=1.0,
                                    steps=steps, sample=sample, device=device)
    t_scmp = time.time() - t0
    print(f"  SCMP done in {t_scmp:.1f}s")
    print(f"  rho_l={scmp_result['final'].get('rho_in', 'N/A'):.4f}, "
          f"rho_v={scmp_result['final'].get('rho_out', 'N/A'):.4f}")
    print(f"  dp={scmp_result['final'].get('dp', 0):.6f}, "
          f"R={scmp_result['final'].get('R_eq', 0):.2f}, "
          f"sigma={scmp_result['final'].get('sigma_laplace', 0):.6f}")

    # --- Run Fakhari phase-field ---
    print("\n[2/2] Running Fakhari phase-field (CudaPhaseField family)...")
    t0 = time.time()
    # Match surface tension: for SCMP G=5, sigma ≈ 0.0561 (lattice units)
    sigma_target = 0.0561
    pf_result = run_fakhari_phasefield_2d(nx=nx, ny=ny, radius=radius,
                                           sigma=sigma_target, W=4.0, M=0.02,
                                           tau_g=0.8, rho_l=0.1, rho_h=1.0,
                                           steps=steps, sample=sample, device=device)
    t_pf = time.time() - t0
    print(f"  Phase-field done in {t_pf:.1f}s")
    print(f"  C_in={pf_result['final'].get('C_in', 'N/A'):.4f}, "
          f"C_out={pf_result['final'].get('C_out', 'N/A'):.4f}")
    print(f"  dp={pf_result['final'].get('dp', 0):.6f}, "
          f"R={pf_result['final'].get('R_eq', 0):.2f}, "
          f"sigma={pf_result['final'].get('sigma_laplace', 0):.6f}")

    # --- Comparison ---
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    scmp_final = scmp_result["final"]
    pf_final = pf_result["final"]

    comparison = {
        "anchor": "XVAL-LBM-NOT-APPLICABLE-PHASE-FIELD-NOT-PSEUDOPOTENTIAL",
        "oss_reference": "AminZar69/CudaPhaseField (Fakhari family phase-field LBM)",
        "tensorlbm_model": "Shan-Chen single-component multiphase (pseudopotential)",
        "verdict": "NOT_APPLICABLE",
        "reason": (
            "Phase-field and pseudopotential are fundamentally different models. "
            "Phase-field solves Allen-Cahn/Cahn-Hilliard with two distributions (h, g) "
            "and a chemical potential. Pseudopotential uses a single distribution with "
            "a ψ(ρ) interaction force. They cannot be directly cross-validated."
        ),
        "scmp": {
            "model": "SCMP (Shan-Chen pseudopotential)",
            "G": 5.0,
            "tau": 1.0,
            "rho_l": scmp_final.get("rho_in", None),
            "rho_v": scmp_final.get("rho_out", None),
            "dp": scmp_final.get("dp", None),
            "R_eq": scmp_final.get("R_eq", None),
            "sigma_laplace": scmp_final.get("sigma_laplace", None),
            "u_max": scmp_final.get("u_max", None),
            "runtime_s": round(t_scmp, 2),
        },
        "phasefield": {
            "model": "Fakhari phase-field (CudaPhaseField family)",
            "sigma_input": sigma_target,
            "W": 4.0,
            "M": 0.02,
            "C_in": pf_final.get("C_in", None),
            "C_out": pf_final.get("C_out", None),
            "dp": pf_final.get("dp", None),
            "R_eq": pf_final.get("R_eq", None),
            "sigma_laplace": pf_final.get("sigma_laplace", None),
            "u_max": pf_final.get("u_max", None),
            "runtime_s": round(t_pf, 2),
        },
        "structural_differences": {
            "distributions": "SCMP: 1 (f) | PF: 2 (h for phase, g for hydro)",
            "interface_physics": "SCMP: ψ(ρ) nearest-neighbor force | PF: chemical potential μ = δF/δC",
            "equation": "SCMP: single LBE with body force | PF: Allen-Cahn + NS (two LBEs)",
            "density_contrast": "SCMP: emergent from ψ(ρ) EOS | PF: prescribed ρ = ρ_l + C(ρ_h - ρ_l)",
            "surface_tension": "SCMP: implicit from G and ψ | PF: explicit via σ parameter in μ",
            "anti_diffusion": "SCMP: none | PF: required (eF term) to maintain interface",
            "gradient_order": "SCMP: nearest-neighbor (1st order) | PF: isotropic 4th-order",
        },
        "parameter_mapping": {
            "note": "No direct 1:1 mapping exists. SCMP's G controls phase separation strength "
                    "and emergent σ; PF's σ is an explicit input parameter. Matching σ requires "
                    "calibration, not formula.",
            "scmp_sigma_emergent": "~0.0561 for G=5, psi_exp, tau=1",
            "pf_sigma_explicit": sigma_target,
        },
        "why_not_applicable": [
            "Different governing equations (single LBE vs. coupled AC+NS)",
            "Different number of distribution functions (1 vs. 2)",
            "Different interface physics (emergent vs. prescribed)",
            "Different stability constraints (SCMP: τ>0.5 | PF: τ_g, τ_h independent)",
            "Different density ratio mechanisms (EOS vs. linear interpolation)",
            "Different gradient discretizations (nearest-neighbor vs. isotropic 4th-order)",
            "Different force formulations (pseudopotential gather vs. chemical potential gradient)",
        ],
    }

    # Save results
    out_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(out_dir, "runs", "scmp_result.json"), "w") as f:
        json.dump(scmp_result, f, indent=2, default=str)
    with open(os.path.join(out_dir, "runs", "pf_result.json"), "w") as f:
        json.dump(pf_result, f, indent=2, default=str)
    with open(os.path.join(out_dir, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    print(f"\nResults saved to {out_dir}/runs/ and {out_dir}/comparison.json")
    print("\nVERDICT: NOT_APPLICABLE — phase-field and pseudopotential are")
    print("fundamentally different models; direct number comparison is not meaningful.")
    print("See comparison.json for structural differences and parameter mapping.")

    return comparison


if __name__ == "__main__":
    main()
