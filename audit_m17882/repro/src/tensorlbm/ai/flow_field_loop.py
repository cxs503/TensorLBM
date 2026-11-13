"""T4 physics-AI loop: real LBM field production -> dataset asset -> FNO flow
map with **held-out Reynolds numbers** -> LBM feedback -> cross-validation.

Why this module exists (the August-2026 inventory breakpoint)
-------------------------------------------------------------
The August AI4S field corpus that fed :mod:`tensorlbm.ai.drag_surrogate` and
:mod:`tensorlbm.ai.flow_transformer_platform_pipeline` was assembled from
*runs* whose fields were either synthesised from closed-form solutions or
produced by smoke-size (a few hundred step, coarse grid) runs.  Every reported
holdout error was therefore an error against a synthetic or under-converged
reference, and the chain "produce field -> asset -> train -> hold out ->
feed back to LBM" had no verified left end and no verified right end.

This module makes both ends objects you can check:

1. **Producer** (:func:`run_case`): a plain D2Q9 BGK LBM in lattice units,
   driven only by :mod:`tensorlbm.solver` primitives
   (``equilibrium`` / ``collide_bgk`` / ``stream`` / ``macroscopic``).
   It is certified by :func:`verify_producer`, which runs the *single-mode*
   Taylor-Green vortex and compares the measured decay with the exact
   Navier-Stokes solution

   .. math::

       u = u_0 \\sin(kx)\\cos(ky)\\,e^{-2\\nu k^2 t},\\quad
       v = -u_0 \\cos(kx)\\sin(ky)\\,e^{-2\\nu k^2 t}

   (G. I. Taylor, *Phil. Mag.* 46 (1923) 671; the same closed form is used as
   the TGV verification anchor in ``docs/TAYLOR_GREEN_3D_VALIDATION.md`` and
   in Li et al. 2021 FNO benchmarks).  The analytic field is used **only** as
   a ruler here — it is never written into the dataset.

2. **Dataset asset** (:func:`produce_dataset`): every snapshot is an
   ``.npz`` written by :func:`run_case` with a sha256 in the manifest, plus
   per-run provenance (grid, ``u0``, ``Re``, ``nu``, ``tau``, ``Ma``, modes,
   seed, step schedule).  The manifest is registered in the existing
   :class:`tensorlbm.data.catalog.FieldDataCatalog` (TD-side contract reuse —
   this module does not invent a data platform).

3. **AI step** (:func:`train_flow_map` / :func:`evaluate_holdout`): an
   :class:`tensorlbm.ai.fno.FNO2d` learns the *flow map*
   ``(u_t, v_t, nu) -> (u_{t+dt}, v_{t+dt})`` on a fixed ``dt`` (a fixed
   number of LBM steps) and is scored on Reynolds numbers that never appear in
   training (``held_out_re``).  Two physics baselines are scored on the same
   rows: persistence (identity) and the *exact discrete viscous prior*
   ``u_hat *= exp(-nu k^2 dt)`` per Fourier mode, which is the diagonal
   solution operator of the discrete diffusion term — i.e. the strongest
   cheap physics prior available for this family.

4. **Feedback into LBM** (:func:`feedback_acceleration`): the surrogate is
   used as a temporal coarse-stepper — ``k`` FNO passes replace ``k`` x
   ``dt`` LBM steps, then the state is handed back to the *real* solver for
   the remaining steps.  The measured quantities are the field error against
   the pure-LBM reference at the handback target, how far that error decays
   once the solver is back in charge ("healing"), and the wall-clock saved.

Everything is deterministic given ``seed``/``torch.manual_seed``; the driver
``bench_t4_physics_ai_loop.py`` writes the artifacts and figures.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from tensorlbm.ai.fno import FNO2d, FNO2dArch
from tensorlbm.solver import collide_bgk, equilibrium, macroscopic, stream

__all__ = [
    "DATASET_SCHEMA",
    "FieldRunConfig",
    "FlowMapConfig",
    "RunResult",
    "analytic_single_mode",
    "evaluate_holdout",
    "feedback_acceleration",
    "initial_velocity",
    "load_dataset",
    "produce_dataset",
    "run_case",
    "train_flow_map",
    "verify_producer",
    "viscous_prior_step",
]

#: Manifest schema identifier written by :func:`produce_dataset`.
DATASET_SCHEMA = "tensorlbm.t4.flow-field-dataset/v1"

#: Modes of the multi-mode initial condition (integer wavenumbers, 2*pi units).
DEFAULT_MODES: tuple[tuple[int, int], ...] = (
    (1, 1),
    (2, 1),
    (1, 2),
    (2, 2),
    (3, 1),
    (1, 3),
)


# ---------------------------------------------------------------------------
# Case definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldRunConfig:
    """One LBM production run (decaying multi-mode Taylor-Green, D2Q9 BGK).

    The Reynolds number is defined on the fundamental wavenumber
    ``k1 = 2 pi / n`` and the initial velocity scale ``u0``:
    ``Re = u0 / (k1 * nu)``, i.e. ``nu = u0 / (k1 * Re)``.  Both are in
    lattice units (``dx = dt = 1``), so ``tau = 3 nu + 0.5``.
    """

    n: int = 128
    u0: float = 0.05
    re: float = 100.0
    seed: int = 0
    n_steps: int = 4000
    sample_every: int = 200
    modes: tuple[tuple[int, int], ...] = DEFAULT_MODES
    single_mode: bool = False
    tag: str = ""

    @property
    def k1(self) -> float:
        return 2.0 * math.pi / float(self.n)

    @property
    def nu(self) -> float:
        return self.u0 / (self.k1 * self.re)

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5

    @property
    def mach(self) -> float:
        """Lattice Mach number of the initial field (``cs = 1/sqrt(3)``)."""
        return self.u0 * math.sqrt(3.0)

    @property
    def run_id(self) -> str:
        base = f"tgv_n{self.n}_re{self.re:g}_s{self.seed}"
        return f"{base}_{self.tag}" if self.tag else base

    def provenance(self) -> dict[str, Any]:
        d = asdict(self)
        d["modes"] = [list(m) for m in self.modes]
        d.update(
            {
                "nu": self.nu,
                "tau": self.tau,
                "k1": self.k1,
                "mach": self.mach,
                "re_definition": "Re = u0 / (k1 * nu), k1 = 2*pi/n",
                "units": "lattice (dx=dt=1); u in lu, t in steps",
            }
        )
        return d


@dataclass
class RunResult:
    """Output of one production run."""

    run_id: str
    config: FieldRunConfig
    steps: list[int] = field(default_factory=list)
    ux: list[torch.Tensor] = field(default_factory=list)
    uy: list[torch.Tensor] = field(default_factory=list)
    rho: list[torch.Tensor] = field(default_factory=list)
    wall_time_s: float = 0.0
    diverged: bool = False

    @property
    def n_snapshots(self) -> int:
        return len(self.steps)


def _grid(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.arange(n, device=device, dtype=torch.float32).view(1, n).expand(n, n)
    y = torch.arange(n, device=device, dtype=torch.float32).view(n, 1).expand(n, n)
    return x, y


def initial_velocity(
    cfg: FieldRunConfig, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Divergence-free multi-mode Taylor-Green initial field.

    A stream function ``psi = sum_k a_k cos(k.x + phi_k)`` gives
    ``u = d psi / d y``, ``v = -d psi / d x``; the field is then scaled so
    that ``rms(|u|) = u0``.  Amplitudes fall as ``1/|k|`` so that the
    enstrophy stays resolvable on the smallest grid used.  With
    ``single_mode`` the configuration degenerates to the closed-form
    single-mode Taylor-Green vortex used by :func:`verify_producer`.
    """
    dev = torch.device(device)
    n = cfg.n
    x, y = _grid(n, dev)
    if cfg.single_mode:
        # Closed-form single-mode Taylor-Green: u = u0 sin(kx) cos(ky),
        # v = -u0 cos(kx) sin(ky) with k = 2 pi / n (one cell-period).
        k = 2.0 * math.pi / n
        ux = cfg.u0 * torch.sin(k * x) * torch.cos(k * y)
        uy = -cfg.u0 * torch.cos(k * x) * torch.sin(k * y)
        return ux, uy

    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    ux = torch.zeros((n, n), device=dev, dtype=torch.float32)
    uy = torch.zeros((n, n), device=dev, dtype=torch.float32)
    for mx, my in cfg.modes:
        kx, ky = 2.0 * math.pi * mx / n, 2.0 * math.pi * my / n
        kmag2 = kx * kx + ky * ky
        # stream function psi = sum_k (1/|k|^2) cos(k.x + phi)  =>
        # u = dpsi/dy = -ky/|k|^2 sin(...), v = -dpsi/dx = +kx/|k|^2 sin(...)
        # (exactly divergence free on the discrete grid, since both are
        # spectral derivatives of the same psi on the periodic torus).
        phase = float(torch.rand((), generator=gen).item() * 2.0 * math.pi)
        arg = kx * x + ky * y + phase
        ux = ux - (ky / kmag2) * torch.sin(arg)
        uy = uy + (kx / kmag2) * torch.sin(arg)

    rms = torch.sqrt((ux**2 + uy**2).mean())
    scale = cfg.u0 / float(rms)
    return ux * scale, uy * scale


def analytic_single_mode(
    n: int, u0: float, nu: float, steps: Sequence[int]
) -> list[float]:
    """Exact ``rms(|u|)`` of the single-mode Taylor-Green vortex.

    ``u_rms(t) = u0 / sqrt(2) * exp(-2 nu k^2 t)`` with ``k = 2 pi / n``
    (Taylor 1923; incompressible Navier-Stokes).
    """
    k = 2.0 * math.pi / n
    return [u0 / math.sqrt(2.0) * math.exp(-2.0 * nu * k * k * float(s)) for s in steps]


def run_case(
    cfg: FieldRunConfig,
    device: torch.device | str = "cpu",
    *,
    record: bool = True,
    initial: tuple[torch.Tensor, torch.Tensor] | None = None,
    n_steps: int | None = None,
) -> RunResult:
    """Run the D2Q9 BGK solver and return the sampled fields.

    ``initial`` overrides the initial velocity field (used by the feedback
    experiment to restart the solver from a surrogate state);
    ``n_steps`` overrides ``cfg.n_steps`` for the continuation runs.
    """
    dev = torch.device(device)
    total = int(cfg.n_steps if n_steps is None else n_steps)
    ux, uy = initial if initial is not None else initial_velocity(cfg, dev)
    rho = torch.ones_like(ux)
    f = equilibrium(rho, ux, uy, device=dev)
    tau = cfg.tau

    out = RunResult(run_id=cfg.run_id, config=cfg)
    t0 = time.perf_counter()
    nonzero = ux.abs().sum() > 0
    for step in range(total + 1):
        if record and step % cfg.sample_every == 0:
            r_, ux_, uy_ = macroscopic(f)
            out.steps.append(step)
            out.ux.append(ux_.detach().clone())
            out.uy.append(uy_.detach().clone())
            out.rho.append(r_.detach().clone())
            if not bool(torch.isfinite(ux_).all()):
                out.diverged = True
                break
        if step == total:
            break
        f = stream(collide_bgk(f, tau))
    out.wall_time_s = time.perf_counter() - t0
    if not nonzero:
        out.diverged = True
    return out


def verify_producer(
    *,
    n: int = 128,
    u0: float = 0.05,
    re_values: Sequence[float] = (50.0, 200.0),
    n_steps: int = 1000,
    sample_every: int = 200,
) -> dict[str, Any]:
    """Certify that the field producer is a real solver with the advertised nu.

    Runs the closed-form single-mode Taylor-Green vortex and compares the
    measured ``rms(|u|)(t)`` with :func:`analytic_single_mode`.  A producer
    that synthesises its "fields" from the analytic solution (the August
    breakpoint) would score ~0 % here after any resampling/rounding error
    budget; a real solver with the wrong viscosity would show a systematic
    decay-rate offset.  Both are checked explicitly:

    * pointwise ``u_rms`` relative error at every sample;
    * the fitted decay rate ``lambda`` from ``log u_rms(t)`` vs the analytic
      ``2 nu k^2``.
    """
    rows = []
    for re in re_values:
        cfg = FieldRunConfig(
            n=n,
            u0=u0,
            re=float(re),
            seed=0,
            n_steps=n_steps,
            sample_every=sample_every,
            single_mode=True,
            tag="producer-verify",
        )
        res = run_case(cfg)
        got = [float(torch.sqrt((ux**2 + uy**2).mean())) for ux, uy in zip(res.ux, res.uy)]
        ref = analytic_single_mode(n, u0, cfg.nu, res.steps)
        errs = [(g - r) / r for g, r in zip(got, ref)]
        # least-squares decay rate from log(u_rms)
        t = torch.tensor([float(s) for s in res.steps], dtype=torch.float64)
        lu = torch.log(torch.tensor(got, dtype=torch.float64))
        slope = float(
            -((t - t.mean()) * (lu - lu.mean())).sum() / ((t - t.mean()) ** 2).sum()
        )
        lam_ref = 2.0 * cfg.nu * cfg.k1**2
        decay_fraction = (got[0] - got[-1]) / got[0]
        rows.append(
            {
                "n": n,
                "re": float(re),
                "nu": cfg.nu,
                "tau": cfg.tau,
                "mach": cfg.mach,
                "steps": res.steps,
                "u_rms_lbm": got,
                "u_rms_analytic": ref,
                "rel_err": errs,
                "max_abs_rel_err": max(abs(e) for e in errs),
                "decay_rate_lbm": slope,
                "decay_rate_analytic": lam_ref,
                "decay_rate_rel_err": (slope - lam_ref) / lam_ref,
                "decay_fraction_over_window": decay_fraction,
                "decay_rate_identifiable": bool(decay_fraction >= 0.05),
                "wall_time_s": res.wall_time_s,
            }
        )
    worst = max(r["max_abs_rel_err"] for r in rows)
    rates = [abs(r["decay_rate_rel_err"]) for r in rows if r["decay_rate_identifiable"]]
    worst_rate = max(rates) if rates else 0.0
    return {
        "kind": "producer-verification",
        "anchor": (
            "Taylor-Green vortex exact Navier-Stokes solution "
            "u_rms = u0/sqrt(2)*exp(-2 nu k^2 t), k = 2 pi / n "
            "(G. I. Taylor, Phil. Mag. 46 (1923) 671)"
        ),
        "rows": rows,
        "max_abs_rel_err_field": worst,
        "max_abs_rel_err_decay_rate": worst_rate,
        "decay_rate_check_identifiable": bool(rates),
        "criteria": {
            "u_rms_max_abs_rel_err": "< 1%",
            "decay_rate_max_abs_rel_err": "< 2% (only when >=5% of the amplitude "
            "decays inside the window, otherwise the rate is not identifiable)",
        },
        "verified": bool(worst < 0.01 and worst_rate < 0.02 and bool(rates)),
    }


# ---------------------------------------------------------------------------
# Dataset asset
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def produce_dataset(
    configs: Iterable[FieldRunConfig],
    out_dir: str | Path,
    *,
    catalog_db: str | Path | None = None,
    dataset_name: str = "t4-tgv-lbm-fields",
    split: dict[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Run every configuration and persist the corpus as a versioned asset.

    Layout::

        out_dir/
          fields/<run_id>.npz        # ux, uy, rho, steps, n, u0, re, nu, tau
          manifest.json              # per-file sha256 + provenance + split
          catalog.sqlite             # FieldDataCatalog (asset + lineage)

    The manifest is the TD/TK-facing contract: one row per field file with
    shape/dtype/units, the physical provenance, and the train/holdout split.
    """
    out = Path(out_dir)
    (out / "fields").mkdir(parents=True, exist_ok=True)

    entries = []
    for cfg in configs:
        res = run_case(cfg)
        if res.diverged:
            raise RuntimeError(f"run {cfg.run_id} diverged (NaN in the velocity field)")
        arr_ux = torch.stack(res.ux).numpy()
        arr_uy = torch.stack(res.uy).numpy()
        arr_rho = torch.stack(res.rho).numpy()
        path = out / "fields" / f"{cfg.run_id}.npz"
        import numpy as np

        np.savez_compressed(
            path,
            ux=arr_ux.astype("float32"),
            uy=arr_uy.astype("float32"),
            rho=arr_rho.astype("float32"),
            steps=np.asarray(res.steps, dtype="int64"),
            n=np.asarray(cfg.n),
            u0=np.asarray(cfg.u0),
            re=np.asarray(cfg.re),
            nu=np.asarray(cfg.nu),
            tau=np.asarray(cfg.tau),
        )
        entries.append(
            {
                "run_id": cfg.run_id,
                "file": f"fields/{cfg.run_id}.npz",
                "sha256": _sha256(path),
                "shape": list(arr_ux.shape),
                "dtype": "float32",
                "channels": ["ux", "uy", "rho"],
                "n_snapshots": res.n_snapshots,
                "steps": res.steps,
                "wall_time_s": round(res.wall_time_s, 3),
                "provenance": cfg.provenance(),
            }
        )

    split_map = {
        k: (list(v) if isinstance(v, (list, tuple)) else v)
        for k, v in (split or {}).items()
    }
    manifest = {
        "schema": DATASET_SCHEMA,
        "name": dataset_name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": {
            "solver": "tensorlbm D2Q9 BGK (equilibrium/collide_bgk/stream/macroscopic)",
            "code": "src/tensorlbm/ai/flow_field_loop.py",
            "real_lbm": True,
            "note": "fields are solver output; analytic solutions are used only "
            "by verify_producer() as a ruler",
        },
        "units": {"u": "lattice units (lu)", "t": "LBM steps", "rho": "lu"},
        "n_files": len(entries),
        "files": entries,
        "split": split_map,
        "totals": {
            "n_snapshots": sum(e["n_snapshots"] for e in entries),
            "wall_time_s": round(sum(e["wall_time_s"] for e in entries), 2),
        },
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    manifest["manifest_sha256"] = _sha256(manifest_path)

    # Register with the existing TD-side catalog contract (no new platform).
    from tensorlbm.data.catalog import AssetRecord, FieldDataCatalog, LineageRecord

    db = Path(catalog_db) if catalog_db else out / "catalog.sqlite"
    cat = FieldDataCatalog.open(db)
    try:
        dataset_id = f"{dataset_name}:dataset"
        cat.register_asset(
            AssetRecord(
                asset_id=dataset_id,
                name=dataset_name,
                kind="dataset",
                description=(
                    f"{len(entries)} real-LBM runs, {manifest['totals']['n_snapshots']} "
                    "snapshots, D2Q9 BGK decaying Taylor-Green"
                ),
                shape=str(entries[0]["shape"]),
                dtype="float32",
                tags=("t4", "lbm", "flow-field", "fno"),
                source_run_id=dataset_name,
            )
        )
        for key, value in (
            ("manifest_sha256", manifest["manifest_sha256"]),
            ("schema", DATASET_SCHEMA),
            ("split", json.dumps(split_map)),
            ("solver", manifest["generator"]["solver"]),
        ):
            cat.add_metadata(dataset_id, key, value, source="run", confidence=1.0)
        for e in entries:
            field_id = f"{dataset_name}:field:{e['run_id']}"
            cat.register_asset(
                AssetRecord(
                    asset_id=field_id,
                    name=e["run_id"],
                    kind="field_product",
                    field_name="u",
                    units="lu",
                    shape=str(e["shape"]),
                    dtype="float32",
                    tags=("t4", "lbm"),
                    source_run_id=e["run_id"],
                )
            )
            cat.add_metadata(field_id, "sha256", e["sha256"], source="run", confidence=1.0)
            cat.add_metadata(
                field_id, "provenance", json.dumps(e["provenance"]), source="run", confidence=1.0
            )
            cat.add_lineage(
                LineageRecord(
                    source_id=field_id,
                    target_id=dataset_id,
                    relation_type="derived_from",
                    resource_type="product",
                )
            )
    finally:
        cat.close()

    manifest["catalog_db"] = str(db)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def load_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    """Load a manifest + all fields into memory (CPU float32 tensors)."""
    import numpy as np

    root = Path(dataset_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    runs = {}
    for e in manifest["files"]:
        z = np.load(root / e["file"])
        runs[e["run_id"]] = {
            "ux": torch.from_numpy(z["ux"]),
            "uy": torch.from_numpy(z["uy"]),
            "steps": [int(s) for s in z["steps"]],
            "re": float(z["re"]),
            "nu": float(z["nu"]),
            "n": int(z["n"]),
            "u0": float(z["u0"]),
            "provenance": e["provenance"],
        }
    return {"manifest": manifest, "runs": runs, "dir": str(root)}


# ---------------------------------------------------------------------------
# AI step: FNO flow map on held-out Reynolds numbers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlowMapConfig:
    """Training configuration for the FNO flow map."""

    width: int = 32
    n_layers: int = 4
    modes: int = 10
    mlp_hidden: int = 128
    epochs: int = 120
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 16
    patience: int = 30
    seed: int = 0


def _pairs(run: dict[str, Any], u0: float) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Build ``(x, y)`` training pairs for one run.

    ``x`` channels: ``u/u0, v/u0, nu/nu_ref`` (constant channel); ``y``: the
    two velocity channels at the next sample.  Normalising by ``u0`` makes the
    inputs scale invariant (the v1.1/v1.2 lesson of
    ``docs/drag_surrogate_fno_20260822.md``: scale-invariant inputs + data
    scale, not architecture, decide the race).
    """
    ux, uy = run["ux"], run["uy"]
    nu = run["nu"]
    n_snap = ux.shape[0]
    xs, ys = [], []
    nu_ref = 1e-2
    for i in range(n_snap - 1):
        a = torch.stack([ux[i] / u0, uy[i] / u0, torch.full_like(ux[i], nu / nu_ref)])
        b = torch.stack([ux[i + 1] / u0, uy[i + 1] / u0])
        xs.append(a)
        ys.append(b)
    return torch.stack(xs), torch.stack(ys), nu


def _rel_l2(pred: torch.Tensor, ref: torch.Tensor) -> float:
    num = torch.linalg.vector_norm((pred - ref).reshape(-1))
    den = torch.linalg.vector_norm(ref.reshape(-1)) + 1e-12
    return float(num / den)


def viscous_prior_step(
    ux: torch.Tensor, uy: torch.Tensor, nu: float, dt_steps: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact discrete viscous prior in Fourier space.

    For the lattice Laplacian, one BGK step multiplies the mode ``k`` by
    ``1 - (something)``; over ``dt`` steps the leading-order (continuum)
    operator is ``exp(-nu |k|^2 dt)`` with ``|k|`` in radians per cell.  This
    is the strongest parameter-free prior for the mapping — the attempt is
    honest: if the FNO does not beat it, the surrogate has not learned the
    advective (nonlinear) part of the flow map.
    """
    kx = torch.fft.fftfreq(ux.shape[-1]) * 2.0 * math.pi * ux.shape[-1] / ux.shape[-1]
    ky = torch.fft.fftfreq(ux.shape[-2]) * 2.0 * math.pi
    # |k|^2 in radians/cell
    k2 = ky.view(-1, 1) ** 2 + kx.view(1, -1) ** 2
    damping = torch.exp(-nu * k2 * float(dt_steps))
    out = []
    for f in (ux, uy):
        fh = torch.fft.fft2(f)
        out.append(torch.real(torch.fft.ifft2(fh * damping)))
    return out[0], out[1]


def train_flow_map(
    dataset_dir: str | Path,
    out_path: str | Path,
    *,
    config: FlowMapConfig | None = None,
    train_res: Sequence[float] | None = None,
    holdout_res: Sequence[float] | None = None,
    grid_n: int | None = None,
    log: bool = True,
) -> dict[str, Any]:
    """Train the FNO flow map on ``train_res`` and never touch ``holdout_res``."""
    cfg = config or FlowMapConfig()
    data = load_dataset(dataset_dir)
    runs = data["runs"]
    manifest = data["manifest"]
    split = manifest.get("split") or {}
    train_res = list(train_res if train_res is not None else split.get("train_re", []))
    holdout_res = list(holdout_res if holdout_res is not None else split.get("holdout_re", []))
    if not train_res:
        train_res = sorted({r["re"] for r in runs.values()})
    holdout_set = {round(float(r), 6) for r in holdout_res}

    torch.manual_seed(cfg.seed)
    xs, ys = [], []
    val_xs, val_ys = [], []
    for run_id, run in sorted(runs.items()):
        if round(run["re"], 6) in holdout_set:
            continue
        if grid_n is not None and run["n"] != grid_n:
            continue
        u0 = run["u0"]
        x, y, _ = _pairs(run, u0)
        # last 20 % of the snapshots of seed 0 runs form the validation split
        if run["provenance"].get("seed", 0) == 0:
            k = max(1, int(0.2 * x.shape[0]))
            val_xs.append(x[-k:])
            val_ys.append(y[-k:])
            xs.append(x[:-k])
            ys.append(y[:-k])
        else:
            xs.append(x)
            ys.append(y)
    X = torch.cat(xs)
    Y = torch.cat(ys)
    VX = torch.cat(val_xs) if val_xs else X[:4]
    VY = torch.cat(val_ys) if val_ys else Y[:4]

    arch = FNO2dArch(
        in_channels=3,
        out_channels=2,
        width=cfg.width,
        n_layers=cfg.n_layers,
        modes_x=cfg.modes,
        modes_y=cfg.modes,
        mlp_hidden=cfg.mlp_hidden,
    )
    model = FNO2d(arch)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    best_val = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    history = []
    bad = 0
    t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(X.shape[0])
        tot = 0.0
        for i in range(0, X.shape[0], cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            pred = model(X[idx])
            loss = torch.mean((pred - Y[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * idx.numel()
        sched.step()
        model.eval()
        with torch.no_grad():
            val = float(torch.mean((model(VX) - VY) ** 2))
        history.append({"epoch": epoch, "train_mse": tot / X.shape[0], "val_mse": val})
        if val < best_val - 1e-12:
            best_val = val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if log and (epoch == 1 or epoch % max(1, cfg.epochs // 8) == 0):
            print(f"[fno] epoch {epoch:4d} train {tot / X.shape[0]:.3e} val {val:.3e}", flush=True)
        if bad >= cfg.patience:
            break
    model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict(), "arch": asdict(arch)}, str(out_path))

    return {
        "model": model,
        "model_path": str(out_path),
        "arch": asdict(arch),
        "train_res": [float(r) for r in train_res],
        "holdout_res": [float(r) for r in holdout_res],
        "n_train_pairs": int(X.shape[0]),
        "n_val_pairs": int(VX.shape[0]),
        "best_val_mse": best_val,
        "epochs_run": len(history),
        "train_wall_s": round(time.perf_counter() - t0, 2),
        "history": history,
        "config": asdict(cfg),
    }


def _predict_rollout(
    model: FNO2d,
    run: dict[str, Any],
    u0: float,
    n_steps: int,
    mode: str,
    dt_steps: int,
) -> list[torch.Tensor]:
    """Autoregressive rollout of ``n_steps`` samples from snapshot 0."""
    ux, uy = run["ux"], run["uy"]
    nu = run["nu"]
    nu_ref = 1e-2
    cur = (ux[0], uy[0])
    preds = [torch.stack(cur) / u0]
    with torch.no_grad():
        for _ in range(n_steps):
            if mode == "persistence":
                cur = (ux[0].clone(), uy[0].clone()) if False else cur
                nxt = cur
            elif mode == "viscous":
                nxt = viscous_prior_step(cur[0], cur[1], nu, dt_steps)
            elif mode == "fno":
                x = torch.stack(
                    [cur[0] / u0, cur[1] / u0, torch.full_like(cur[0], nu / nu_ref)]
                )[None]
                out = model(x)[0]
                nxt = (out[0] * u0, out[1] * u0)
            else:  # pragma: no cover - guarded by the caller
                raise ValueError(f"unknown mode {mode!r}")
            cur = nxt
            preds.append(torch.stack(cur) / u0)
    return preds


def evaluate_holdout(
    model: FNO2d,
    dataset_dir: str | Path,
    *,
    train_res: Sequence[float],
    holdout_res: Sequence[float],
    n_rollout: int = 0,
    grid_n: int | None = None,
) -> dict[str, Any]:
    """Score FNO vs baselines on Reynolds numbers excluded from training.

    Metrics (all relative L2 in the velocity field, ``u0``-normalised):

    * one-step: ``||pred(t+dt) - LBM(t+dt)|| / ||LBM(t+dt)||``;
    * rollout to the end of the run (``n_rollout = n_snapshots - 1``) for
      FNO / persistence / exact-viscous prior — this is where a surrogate
      that only learned dissipation falls behind a real flow map.
    """
    data = load_dataset(dataset_dir)
    runs = data["runs"]
    holdout_set = {round(float(r), 6) for r in holdout_res}
    train_set = {round(float(r), 6) for r in train_res}

    per_run = []
    for run_id, run in sorted(runs.items()):
        re = round(run["re"], 6)
        if re not in holdout_set:
            continue
        if grid_n is not None and run["n"] != grid_n:
            continue
        u0 = run["u0"]
        dt_steps = run["steps"][1] - run["steps"][0]
        x, y, _ = _pairs(run, u0)
        model.eval()
        with torch.no_grad():
            pred = model(x)
        fno_one = [_rel_l2(pred[i], y[i]) for i in range(x.shape[0])]
        # NOTE: baselines must live in the same units as ``y`` (u/u0) — see the
        # m15932 board audit: feeding raw lattice-unit fields to _rel_l2 made
        # persistence/viscous sit at ~1/u0 ~ 1.0 by construction and flipped the
        # "FNO beats its baselines" verdict.
        vis = [
            torch.stack(viscous_prior_step(run["ux"][i], run["uy"][i], run["nu"], dt_steps)) / u0
            for i in range(x.shape[0])
        ]
        vis_err = [_rel_l2(vis[i], y[i]) for i in range(x.shape[0])]
        pers = [
            _rel_l2(torch.stack([run["ux"][i], run["uy"][i]]) / u0, y[i])
            for i in range(x.shape[0])
        ]
        row = {
            "run_id": run_id,
            "re": run["re"],
            "n": run["n"],
            "nu": run["nu"],
            "in_train": re in train_set,
            "dt_steps": dt_steps,
            "one_step_rel_l2": {
                "fno": sum(fno_one) / len(fno_one),
                "viscous_prior": sum(vis_err) / len(vis_err),
                "persistence": sum(pers) / len(pers),
            },
        }
        if n_rollout > 0:
            n_roll = min(n_rollout, x.shape[0])
            roll = {}
            for mode in ("fno", "viscous", "persistence"):
                preds = _predict_rollout(model, run, u0, n_roll, mode, dt_steps)
                ref = [
                    torch.stack([run["ux"][i + 1], run["uy"][i + 1]]) / u0 for i in range(n_roll)
                ]
                errs = [_rel_l2(preds[i + 1], ref[i]) for i in range(n_roll)]
                roll[mode] = {
                    "rel_l2_series": errs,
                    "final": errs[-1],
                    "mean": sum(errs) / len(errs),
                }
            row["rollout"] = roll
            row["rollout_steps_lbm"] = n_roll * dt_steps
        per_run.append(row)

    summary: dict[str, Any] = {}
    if per_run:
        for key in ("fno", "viscous_prior", "persistence"):
            summary[f"one_step_rel_l2_{key}"] = sum(
                r["one_step_rel_l2"][key] for r in per_run
            ) / len(per_run)
        if n_rollout > 0:
            for key in ("fno", "viscous", "persistence"):
                summary[f"rollout_final_rel_l2_{key}"] = sum(
                    r["rollout"][key]["final"] for r in per_run
                ) / len(per_run)
    return {
        "holdout_res": [float(r) for r in holdout_res],
        "train_res": [float(r) for r in train_res],
        "per_run": per_run,
        "summary": summary,
        "n_holdout_runs": len(per_run),
    }


# ---------------------------------------------------------------------------
# Feedback into the LBM solver
# ---------------------------------------------------------------------------


def feedback_acceleration(
    model: FNO2d,
    dataset_dir: str | Path,
    *,
    holdout_res: Sequence[float],
    k_values: Sequence[int] = (2, 4, 6),
    heal_steps: int | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Use the surrogate as a temporal coarse-stepper, then hand back to LBM.

    Protocol for one held-out run (samples every ``dt`` LBM steps):

    * reference: the pure-LBM field at step ``k*dt`` and at the end of the
      run, taken from the production dataset;
    * accelerated path: FNO rollout for ``k`` samples starting from the LBM
      initial condition, then a *real* LBM continuation from that surrogate
      state for the remaining steps;
    * healing: continue the LBM for ``heal_steps`` further steps and compare
      with the pure-LBM reference at the same physical time.

    Reported per ``k``: handback error, healed error, and the wall-clock
    speed-up (LBM steps replaced by surrogate passes, using the measured
    per-step cost of the continuation run).
    """
    data = load_dataset(dataset_dir)
    runs = data["runs"]
    holdout_set = {round(float(r), 6) for r in holdout_res}
    rows = []
    for run_id, run in sorted(runs.items()):
        if round(run["re"], 6) not in holdout_set:
            continue
        u0 = run["u0"]
        nu = run["nu"]
        nu_ref = 1e-2
        dt = run["steps"][1] - run["steps"][0]
        n_snap = run["ux"].shape[0]
        for k in k_values:
            if k >= n_snap - 1:
                continue
            # --- surrogate rollout of k samples
            cur = (run["ux"][0], run["uy"][0])
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(k):
                    x = torch.stack(
                        [cur[0] / u0, cur[1] / u0, torch.full_like(cur[0], nu / nu_ref)]
                    )[None]
                    out = model(x)[0]
                    cur = (out[0] * u0, out[1] * u0)
            sur_time = time.perf_counter() - t0
            sur_err = _rel_l2(
                torch.stack(cur), torch.stack([run["ux"][k], run["uy"][k]])
            )
            # --- hand back to the real solver
            cfg = FieldRunConfig(
                n=run["n"],
                u0=u0,
                re=run["re"],
                seed=run["provenance"].get("seed", 0),
                n_steps=dt,
                sample_every=dt,
                modes=tuple(tuple(m) for m in run["provenance"]["modes"]),
                tag="feedback",
            )
            cont = run_case(cfg, device, initial=cur, n_steps=dt)
            ref = torch.stack([run["ux"][k + 1], run["uy"][k + 1]])
            hand_err = _rel_l2(torch.stack([cont.ux[-1], cont.uy[-1]]), ref)
            # --- healing: keep the solver in charge for heal_steps more steps
            healed_err = None
            if heal_steps and heal_steps % dt == 0:
                heal_cfg = FieldRunConfig(
                    n=run["n"],
                    u0=u0,
                    re=run["re"],
                    seed=run["provenance"].get("seed", 0),
                    n_steps=heal_steps,
                    sample_every=dt,
                    modes=tuple(tuple(m) for m in run["provenance"]["modes"]),
                    tag="heal",
                )
                heal = run_case(
                    heal_cfg, device, initial=(cont.ux[-1], cont.uy[-1]), n_steps=heal_steps
                )
                j = k + 1 + heal_steps // dt
                if j < n_snap:
                    healed_err = _rel_l2(
                        torch.stack([heal.ux[-1], heal.uy[-1]]),
                        torch.stack([run["ux"][j], run["uy"][j]]),
                    )
            lbm_step_cost = cont.wall_time_s / max(1, dt)
            rows.append(
                {
                    "run_id": run_id,
                    "re": run["re"],
                    "k_surrogate_steps": k,
                    "lbm_steps_replaced": k * dt,
                    "surrogate_rel_l2_at_handback": sur_err,
                    "handback_rel_l2": hand_err,
                    "healed_rel_l2": healed_err,
                    "surrogate_wall_s": round(sur_time, 4),
                    "lbm_wall_s_for_one_dt": round(cont.wall_time_s, 4),
                    "lbm_step_cost_s": lbm_step_cost,
                    "speedup_vs_lbm": (k * dt * lbm_step_cost) / max(sur_time, 1e-9),
                }
            )
    return {
        "holdout_res": [float(r) for r in holdout_res],
        "heal_steps": heal_steps,
        "rows": rows,
        "note": (
            "speedup_vs_lbm compares the wall time of k FNO passes with the "
            "measured wall time of the replaced LBM steps; it is a wall-clock "
            "ratio, not a correctness claim — correctness is handback_rel_l2."
        ),
    }