#!/usr/bin/env python3
"""
SDAA sweep for wigley_hydrostatics benchmark.

Runs the Wigley parabolic hull hydrostatics computation on SDAA device,
checks all verification criteria from the verified case.

Criteria (from benchmarks/verified/wigley_hydrostatics/result.json):
  - All form coefficients within 3% of analytic at L=640 and L=960
  - Inter-grid span (|val(L=640) - val(L=960)| / analytic * 100) within 3%
  - max |err| <= 3%, max span <= 3%
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import torch

# ── Device setup ──────────────────────────────────────────────────────────
DEVICE = "sdaa"
if not torch.sdaa.is_available():
    print("FATAL: SDAA not available", file=sys.stderr)
    sys.exit(1)

device_name = torch.sdaa.get_device_name(0)
print(f"[sdaa] device: {device_name}")

# ── Analytic reference values ─────────────────────────────────────────────
ANALYTIC = {
    "C_B": 4 / 9,
    "C_wp": 2 / 3,
    "C_M": 2 / 3,
    "C_P": 2 / 3,
    "KB_over_T": 5 / 8,
    "IT_over_B3L": 4 / 105,
    "IL_over_L3B": 1 / 30,
}

# ── Wigley hull voxel mask (from the verified benchmark) ──────────────────

def full_mask(nx, ny, nz, cx, cy, cz_keel, L, B, T, chunk=32, device="cpu"):
    zz = torch.arange(nz, device=device, dtype=torch.float32)
    yy = torch.arange(ny, device=device, dtype=torch.float32)
    xx = torch.arange(nx, device=device, dtype=torch.float32)
    x_norm = (xx - cx) / (L / 2.0)
    z_norm = (zz - (cz_keel + T)) / T
    in_length = x_norm.abs() <= 1.0
    in_draft = (z_norm >= -1.0) & (z_norm <= 0.0)
    rows = []
    for k0 in range(0, nz, chunk):
        k1 = min(k0 + chunk, nz)
        zc = z_norm[k0:k1].view(-1, 1, 1)
        hb = (B / 2.0) * (1.0 - x_norm.view(1, 1, nx) ** 2) * (1.0 - zc**2)
        valid = in_length.view(1, 1, nx) & in_draft[k0:k1].view(-1, 1, 1)
        hb = torch.where(valid, hb, torch.full_like(hb, -1.0))
        rows.append((yy.view(1, ny, 1) - cy).abs() <= hb)
    return torch.cat(rows, dim=0)


def measure(L, B, T, device="cpu"):
    margin = 1.0
    nx = int(math.ceil(L)) + 2 * int(math.ceil(margin * L)) + 2
    ny = int(math.ceil(B)) + 2 * int(math.ceil(margin * B)) + 3
    nz = int(math.ceil(T)) + 2 * int(math.ceil(margin * T)) + 2
    cx = nx / 2.0
    cy = float(int(round(ny / 2.0)))
    cz_keel = float(int(margin * T))
    m = full_mask(nx, ny, nz, cx, cy, cz_keel, L, B, T, device=device)
    mf = m.to(torch.float64)
    V = float(mf.sum())
    zz = torch.arange(nz, device=device, dtype=torch.float64).view(-1, 1, 1)
    yy = torch.arange(ny, device=device, dtype=torch.float64).view(1, -1, 1)
    xx = torch.arange(nx, device=device, dtype=torch.float64).view(1, 1, -1)
    sz = float((mf * zz).sum())
    sx = float((mf * xx).sum())
    # exposed top face
    exp = m.clone()
    exp[:-1] = m[:-1] & ~m[1:]
    exp[-1] = False
    ef = exp.to(torch.float64)
    A_wp = float(ef.sum())
    IT = float((ef * (yy - cy) ** 2).sum())
    IL = float((ef * (xx - cx) ** 2).sum())
    # midship section
    ix = int(round(cx))
    x_norm_i = (ix - cx) / (L / 2.0)
    z_norm = (torch.arange(nz, device=device, dtype=torch.float64) - (cz_keel + T)) / T
    hb_i = (B / 2.0) * (1.0 - x_norm_i**2) * (1.0 - z_norm**2)
    valid_i = (abs(x_norm_i) <= 1.0) & (z_norm >= -1.0) & (z_norm <= 0.0)
    hb_i = torch.where(valid_i, hb_i, torch.full_like(hb_i, -1.0))
    A_m = float(((yy.view(1, -1, 1) - cy).abs() <= hb_i.view(-1, 1, 1)).sum())
    r = dict(
        L=L,
        n=(nx, ny, nz),
        V=V,
        A_wp=A_wp,
        A_m=A_m,
        C_B=V / (L * B * T),
        C_wp=A_wp / (L * B),
        C_M=A_m / (B * T),
        C_P=V / (A_m * L),
        KB_over_T=(sz / V - cz_keel) / T,
        IT_over_B3L=IT / (B**3 * L),
        IL_over_L3B=IL / (L**3 * B),
        LCB_over_L=(sx / V - cx) / L,
    )
    r["err_pct"] = {k: (r[k] - ANALYTIC[k]) / ANALYTIC[k] * 100 for k in ANALYTIC}
    return r


# ── Main sweep ────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    grids = [640, 960]
    results = {}

    for L in grids:
        B = L / 10.0
        T = L / 16.0
        print(f"\n[sdaa] Running L={L} (grid ~{L}x{L/10}x{L/16}) ...", flush=True)
        tg = time.time()
        r = measure(float(L), B, T, device=DEVICE)
        elapsed = time.time() - tg
        r["elapsed_s"] = round(elapsed, 2)
        results[L] = r
        print(f"  done in {elapsed:.1f}s")
        for k in ANALYTIC:
            print(f"  {k} = {r[k]:.6f}  err = {r['err_pct'][k]:+.3f}%")

    # ── Compute spans ─────────────────────────────────────────────────────
    spans = {}
    for k in ANALYTIC:
        v640 = results[640][k]
        v960 = results[960][k]
        span_pct = abs(v640 - v960) / ANALYTIC[k] * 100
        spans[k] = round(span_pct, 4)

    total_elapsed = time.time() - t0

    # ── Gate checks ───────────────────────────────────────────────────────
    max_err_pct = 0.0
    max_span_pct = 0.0
    gates = {}

    for k in ANALYTIC:
        err640 = abs(results[640]["err_pct"][k])
        err960 = abs(results[960]["err_pct"][k])
        span = spans[k]
        max_err_pct = max(max_err_pct, err640, err960)
        max_span_pct = max(max_span_pct, span)
        gates[k] = {
            "err_L640_pct": round(err640, 4),
            "err_L960_pct": round(err960, 4),
            "span_pct": span,
            "err_within_3pct": err640 <= 3.0 and err960 <= 3.0,
            "span_within_3pct": span <= 3.0,
            "PASS": err640 <= 3.0 and err960 <= 3.0 and span <= 3.0,
        }

    overall_pass = all(g["PASS"] for g in gates.values())

    # ── Build output ──────────────────────────────────────────────────────
    output = {
        "case": "wigley_hydrostatics",
        "device": DEVICE,
        "device_name": device_name,
        "device_evidence": f"torch.sdaa.get_device_name(0) = {device_name!r}",
        "triton_interpret": False,
        "triton_note": "No triton dependency; pure PyTorch tensor ops on SDAA",
        "grids_run": grids,
        "analytic": {k: f"{v:.10g}" for k, v in ANALYTIC.items()},
        "results": {},
        "spans": spans,
        "gates": gates,
        "max_abs_err_pct": round(max_err_pct, 4),
        "max_span_pct": round(max_span_pct, 4),
        "overall_PASS": overall_pass,
        "wall_clock_s": round(total_elapsed, 2),
    }

    for L in grids:
        r = results[L]
        output["results"][str(L)] = {
            "grid": list(r["n"]),
            "V": r["V"],
            "A_wp": r["A_wp"],
            "A_m": r["A_m"],
            "C_B": r["C_B"],
            "C_wp": r["C_wp"],
            "C_M": r["C_M"],
            "C_P": r["C_P"],
            "KB_over_T": r["KB_over_T"],
            "IT_over_B3L": r["IT_over_B3L"],
            "IL_over_L3B": r["IL_over_L3B"],
            "LCB_over_L": r["LCB_over_L"],
            "err_pct": {k: round(v, 4) for k, v in r["err_pct"].items()},
            "elapsed_s": r["elapsed_s"],
        }

    # ── Write artifact ────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(__file__), "..", "artifacts", "sdaa_wigley_hydrostatics.json")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[artifact] written to {out_path}")
    print(f"[result] overall_PASS = {overall_pass}")
    print(f"[result] max_abs_err_pct = {max_err_pct:.4f}%")
    print(f"[result] max_span_pct = {max_span_pct:.4f}%")
    print(f"[result] wall_clock = {total_elapsed:.1f}s")

    for k, g in gates.items():
        status = "PASS" if g["PASS"] else "FAIL"
        print(f"  [{status}] {k}: err640={g['err_L640_pct']:.3f}% err960={g['err_L960_pct']:.3f}% span={g['span_pct']:.3f}%")


if __name__ == "__main__":
    main()
