# sphere_re100_d3q27 — D3Q27 twin of the verified sphere Re=100 case  ✅ VERIFIED

Status: **verified** (2026-10-09, Wave-11 track D). Stencil-robustness twin of
`verified/sphere_re100` (D3Q19 + MRT): same problem, same gate structure,
reference sha256-pinned to the promoted archive **before any gate run**.
G1/G2/G3 all PASS; G4 cross-stencil rows disclosed. `verify.py` re-derives the
whole verdict from the tier histories and the pinned archive.

## Physics

Uniform flow past a sphere at Re = 100; drag coefficient Cd over the last-20%
sample window. Twin differences vs the verified D3Q19 case (frozen in
prereg.md before the runs): lattice D3Q27, collision BGK27 (the library has no
low-memory MRT27 — disclosed), streaming `stream27_roll`, BFL sparse link
boundary with exact ray-sphere q. The five hot kernels use in-place mirrored
implementations proven `torch.equal` to the library originals (`diag/ab27.json`),
capping the live set at 3 full (27,N) fields so the 84.7M-cell D60 twin fits a
32 GB card.

## Reference (locked pre-run, sha256-pinned)

cd_ref = 1.0917310910948732 — Schiller-Naumann 24/Re*(1+0.15*Re^0.687) = 1.0917311, read
from the machine-pulled `verified/sphere_re100` archive (remote promotion
commit 20cd23472) and pinned in `reflock.json` (result.json / t1 / t2 /
manifest sha256) before any gate run. Correlation band
[1.069, 1.109] (Schiller-Naumann 0.681 variant …
Clift-Gauvin) brackets it.

## Results (machine values from result.json / gate/verdict27.json)

| tier | domain (cells) | τ | steps | Cd window | err |
|---|---|---|---|---|---|
| T1 D40 | [280, 280, 320] | 0.56 | 12,000 | 1.1222091370 | +2.7917% |
| T2 D60 | [420, 420, 480] | 0.59 | 24,000 | 1.1197800350 | +2.5692% |

* G1 accuracy: |err| ≤ 3% on both tiers (2.7917% / 2.5692%)
* G2 monotone: err strictly decreasing with resolution (2.7917 → 2.5692%)
* G3 health: window drift 0.1440/0.0418% (gate 0.3%),
  CV closure 2.86e-03/3.83e-04 (1e-2),
  mass 0.00/-0.19 ppm, no divergence;
  archive-twin parameter checks all green (domain/τ/steps/u_lb/Re/lat)
* G4 disclosures: 27-vs-19 same-domain Δ = 0.0204% (D40) /
  0.0284% (D60); 27-twin span 0.2165% vs
  archived 19-twin span 0.231%; same-domain BGK19-vs-MRT19
  Δ = 0.0176% — the D3Q27 twin sits inside the verified
  19-velocity error band and the collision choice is a sub-0.02% effect at this Re

## Reproduce

    python verify.py          # re-derive gate/verdict27.json from the archives
    python run.py out.json --D 40 --steps 12000   # rerun a tier (CUDA, ~24 min D40)

Set `W27_ARCHIVE_DIR=$PWD/diag/remote_pull/verified__sphere_re100` to arm the
driver's archive-twin parameter guard.

## History

The pending skeleton (an earlier wxsc-era run.py design sketch on the
cumulant/MRT27 route) is retired with this promotion; the preregistered BFL
sparse-route twin shipped here superseded it. Generator scripts for the audit
evidence and the full remote pull remain in the campaign staging (paths in
result.json:provenance).

Artifacts: `run.py` / `verify.py` (gate-run scripts md5 8460d639 / d69e36c2,
lint-normalized here — import order + one unused import, semantics unchanged),
`prereg.md` + md5 sidecar (frozen-append-only chain), `reflock.json`,
`gate/` (tier histories + verdict + run logs), `diag/` (A/B bitwise evidence,
lattice audit, mini/smoke checks, pinned remote-pull subset), `result.json`
(aggregate + provenance).
