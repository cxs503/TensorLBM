"""Bitwise baseline artefacts for the soft-geometry (inverse design) feature.

The soft-solid extension of ``tensorlbm.autograd_path`` is strictly opt-in:
``soft=None`` (the default) must keep the packaged step chain bit-for-bit
identical to the pre-feature commit.  This script pins that contract across a
spread of rollout configurations by writing the final states, the per-step
force probes and the accumulated drag of a fixed, seed-deterministic set of
rollouts to a single ``baseline.pt`` payload:

    CUDA_VISIBLE_DEVICES= PYTHONPATH=src python \
        scripts/gen_autograd_inverse_baseline.py --out /nfs/wangxi/tmp/inv_base

Run it at the base commit (pre-feature) to produce the reference payload, then
at any later revision and compare with ``--check`` (or load both payloads and
``torch.equal`` them).  ``tests/test_autograd_inverse.py`` performs the same
comparison automatically when ``TENSORLBM_INVERSE_BASELINE_DIR`` points at the
directory holding ``baseline.pt``.

Determinism: CPU-only rollouts (pass ``CUDA_VISIBLE_DEVICES=``), initial
conditions built from CPU-seeded generators, no data-dependent host syncs in
the chain.  The payload is a plain dict ``{config_name: {"f": tensor,
"drag": tensor}}`` plus a small header with the git commit it was produced at.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import pathlib
import subprocess

import torch

from tensorlbm.autograd_path import (
    InletSpec,
    OutletSpec,
    WallSpec,
    obstacle_force,
    rollout,
)
from tensorlbm.boundaries3d import sphere_mask
from tensorlbm.d3q19 import equilibrium3d

# --- the fixed configuration set (mirrors tests/test_autograd_path.py cases)
PERIODIC = dict(nz=10, ny=12, nx=16, cx=8.0, cy=6.0, cz=5.0, radius=2.5)
BOUNDED = dict(nz=8, ny=10, nx=20, cx=6.0, cy=5.0, cz=4.0, radius=2.0)
U_IN = 0.08


def shear_flow_f0(amplitude: float, dtype: torch.dtype) -> torch.Tensor:
    """Equilibrium shear flow u = a*(sin(2 pi y/ny) + 0.3*cos(2 pi z/nz)) x-hat."""
    nz, ny, nx = PERIODIC["nz"], PERIODIC["ny"], PERIODIC["nx"]
    zz, yy, _xx = torch.meshgrid(
        torch.arange(nz, dtype=dtype),
        torch.arange(ny, dtype=dtype),
        torch.arange(nx, dtype=dtype),
        indexing="ij",
    )
    ux = amplitude * (
        torch.sin(2.0 * math.pi * yy / ny) + 0.3 * torch.cos(2.0 * math.pi * zz / nz)
    )
    zeros = torch.zeros_like(ux)
    return equilibrium3d(torch.ones_like(ux), ux, zeros, zeros)


def noisy_shear_f0(dtype: torch.dtype, seed: int = 11) -> torch.Tensor:
    """Shear flow plus deterministic off-equilibrium noise."""
    nz, ny, nx = PERIODIC["nz"], PERIODIC["ny"], PERIODIC["nx"]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.rand((19, nz, ny, nx), generator=gen, dtype=torch.float64).to(dtype) - 0.5
    return shear_flow_f0(0.05, dtype) + 0.05 * noise


def uniform_flow_f0(u: float, box: dict, dtype: torch.dtype, seed: int = 23) -> torch.Tensor:
    """Uniform equilibrium inflow plus deterministic off-equilibrium noise."""
    nz, ny, nx = box["nz"], box["ny"], box["nx"]
    ones = torch.ones(nz, ny, nx, dtype=dtype)
    zeros = torch.zeros_like(ones)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.rand((19, nz, ny, nx), generator=gen, dtype=torch.float64).to(dtype) - 0.5
    return equilibrium3d(ones, u * ones, zeros, zeros) + 0.02 * noise


def _config_rollouts() -> dict[str, dict]:
    """The fixed rollout set: {name: kwargs for rollout} plus f0 builders."""
    return {
        "periodic_masked_f64": dict(
            f0=lambda: noisy_shear_f0(torch.float64),
            steps=12,
            tau=0.8,
            mask=sphere_mask(
                PERIODIC["nx"], PERIODIC["ny"], PERIODIC["nz"],
                PERIODIC["cx"], PERIODIC["cy"], PERIODIC["cz"], PERIODIC["radius"],
                device=torch.device("cpu"),
            ),
        ),
        "periodic_masked_f32": dict(
            f0=lambda: noisy_shear_f0(torch.float32),
            steps=12,
            tau=0.8,
            mask=sphere_mask(
                PERIODIC["nx"], PERIODIC["ny"], PERIODIC["nz"],
                PERIODIC["cx"], PERIODIC["cy"], PERIODIC["cz"], PERIODIC["radius"],
                device=torch.device("cpu"),
            ),
        ),
        "bounded_eqcopy_f64": dict(
            f0=lambda: uniform_flow_f0(U_IN, BOUNDED, torch.float64),
            steps=10,
            tau=0.7,
            mask=sphere_mask(
                BOUNDED["nx"], BOUNDED["ny"], BOUNDED["nz"],
                BOUNDED["cx"], BOUNDED["cy"], BOUNDED["cz"], BOUNDED["radius"],
                device=torch.device("cpu"),
            ),
            inlet=InletSpec(ux=U_IN),
            outlet=OutletSpec(),
        ),
        "bounded_zouhe_conv_slip_f64": dict(
            f0=lambda: uniform_flow_f0(U_IN, BOUNDED, torch.float64),
            steps=10,
            tau=0.7,
            mask=sphere_mask(
                BOUNDED["nx"], BOUNDED["ny"], BOUNDED["nz"],
                BOUNDED["cx"], BOUNDED["cy"], BOUNDED["cz"], BOUNDED["radius"],
                device=torch.device("cpu"),
            ),
            inlet=InletSpec(ux=U_IN, method="zouhe"),
            outlet=OutletSpec(method="convective"),
            walls=WallSpec(method="free-slip"),
        ),
        "bounded_perface_f64": dict(
            f0=lambda: uniform_flow_f0(U_IN, BOUNDED, torch.float64),
            steps=10,
            tau=0.7,
            mask=sphere_mask(
                BOUNDED["nx"], BOUNDED["ny"], BOUNDED["nz"],
                BOUNDED["cx"], BOUNDED["cy"], BOUNDED["cz"], BOUNDED["radius"],
                device=torch.device("cpu"),
            ),
            inlet=InletSpec(ux=U_IN),
            outlet=OutletSpec(method="convective"),
            walls=WallSpec(
                method="free-slip",
                overrides={
                    "-z": WallSpec(method="periodic"),
                    "+z": WallSpec(method="freestream", rho0=1.03, ux=0.06, uy=-0.02),
                },
            ),
        ),
    }


def run_rollouts() -> dict[str, dict[str, torch.Tensor]]:
    """Run every configuration on the CPU and collect f_final + accumulated drag."""
    payload: dict[str, dict[str, torch.Tensor]] = {}
    for name, cfg in _config_rollouts().items():
        torch.manual_seed(0)
        f0 = cfg["f0"]()
        f, probes = rollout(
            f0,
            cfg["steps"],
            cfg["tau"],
            cfg["mask"],
            inlet=cfg.get("inlet"),
            outlet=cfg.get("outlet"),
            walls=cfg.get("walls"),
            return_probes=True,
        )
        drag = sum(obstacle_force(p, cfg["mask"])[0] for p in probes)
        payload[name] = {"f": f.detach().clone(), "drag": drag.detach().clone()}
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None, help="output directory for baseline.pt")
    parser.add_argument(
        "--check",
        metavar="DIR",
        help="compare a freshly produced payload against the baseline.pt in DIR",
    )
    args = parser.parse_args()

    payload = run_rollouts()
    if args.check is None:
        out_dir = pathlib.Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        header = {"format": 1, "commit": _git_commit()}
        torch.save({"header": header, "configs": payload}, out_dir / "baseline.pt")
        digest = _digest(out_dir / "baseline.pt")
        print(f"wrote {out_dir / 'baseline.pt'} (sha256 {digest[:16]}..., commit {header['commit']})")
        for name, entry in payload.items():
            print(
                f"  {name:32s} f[{entry['f'].shape}] drag={float(entry['drag']):.12e}"
            )
        return

    reference = torch.load(pathlib.Path(args.check) / "baseline.pt", weights_only=True)
    ref_commit = reference["header"]["commit"]
    failures = []
    for name, entry in payload.items():
        ref = reference["configs"].get(name)
        if ref is None:
            failures.append(f"{name}: missing from reference payload")
            continue
        same_f = torch.equal(entry["f"].cpu(), ref["f"].cpu())
        same_drag = torch.equal(entry["drag"].cpu(), ref["drag"].cpu())
        status = "OK" if (same_f and same_drag) else "MISMATCH"
        print(
            f"  {name:32s} f={'equal' if same_f else 'DIFFER'} "
            f"drag={'equal' if same_drag else 'DIFFER'} -> {status}"
        )
        if not (same_f and same_drag):
            failures.append(name)
    extra = sorted(set(reference["configs"]) - set(payload))
    for name in extra:
        print(f"  {name:32s} extra in reference")
    if failures or extra:
        raise SystemExit(f"bitwise baseline check FAILED for {failures + extra}")
    print(f"bitwise baseline check passed against {args.check} (commit {ref_commit})")


def _git_commit() -> str:
    try:
        return (
            subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
            .stdout.strip()
        )
    except OSError:
        return "unknown"


def _digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
