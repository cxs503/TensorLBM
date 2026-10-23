"""GPU smoke / verification suite: one command, pass/fail JSON report.

Formalises the ad-hoc GPU verification battery that runs before every
PR, after a driver update or a torch upgrade.  Nine checks, each with an
explicit threshold whose provenance is recorded in
``docs/gpu_smoke_suite_20260825.md``:

1.  ``env_sanity`` -- torch/CUDA/driver identity, venv python, /nfs
    writable, TMPDIR on /nfs (ops rule: root partition is small).
2.  ``parity_cpu_cuda`` -- one real serving checkpoint through
    ``load_checkpoint().to_model()`` on a FIXED batch (corpus field row
    0 + first 32 condition rows): log10-space max diff within the
    float32 kernel-noise class (bitwise is NOT expected across
    devices).
3.  ``determinism_gpu_repeat`` -- two GPU runs of the same batch must
    be bitwise equal.
4.  ``serve_latency`` -- ``DragSurrogateService`` +
    ``ModelEnsembleBackend`` (5 members), 64-Re query (the
    ``b4_serve_benchmark.py`` pattern): p50 < 50 ms.
5.  ``echo_latency`` -- minimal inline twin of the (unmerged-here) CAD
    slider echo path: rotating cold design -> condition rows -> guard ->
    corpus-cache field row -> 5-member forward: p50 < 150 ms.
6.  ``onnx_parity`` -- private-ORT spot check of the stacked ensemble
    ONNX artifact vs the torch backend (subprocess, own PYTHONPATH):
    log10 max diff < 1e-5.
7.  ``voxelizer_cross_impl`` -- icosphere (r=5, subdiv 3) through
    ``voxelize.mask_from_stl`` vs ``geometry_voxel.voxelize_stl_reference``:
    100% identical cells required.
8.  ``train_smoke`` -- tiny CondFNODrag trained 30 steps on a synthetic
    slice: loss decreases, no NaN, backward works.
9.  ``suboff_mask_cpu_cuda`` -- production-grid SUBOFF mask (mother
    config) built on CPU and GPU: bitwise-identical bool masks.

Checks whose /nfs run artifacts are absent (fresh checkout) skip with a
reason instead of failing.  ``--cpu-only`` (CI host) skips the GPU
checks the same way and keeps the CPU-runnable ones.  ``--quick`` drops
latency reps to 1.

Usage::

    TMPDIR=/nfs/wangxi/tmp PYTHONPATH=src python benchmarks/gpu_smoke_suite.py \\
        --gpu 4 --out /nfs/wangxi/tmp/gpu_smoke_report.json

Exit code 0 iff no check failed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

# Thread pinning for --cpu-only must happen before torch grabs the thread
# pools.  argv is only scanned when the suite is invoked as a script with
# the flag present; a plain `import gpu_smoke_suite` is a no-op.
if "--cpu-only" in sys.argv[1:]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

import tensorlbm
from tensorlbm.ai.drag_cond import CondFNODrag
from tensorlbm.ai.inference_service import (
    CondDragCheckpoint,
    CorpusIndex,
    DragSurrogateService,
    EnvelopeMahalanobisGuardrail,
    ModelEnsembleBackend,
    load_checkpoint,
    load_corpus_index,
)
from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask
from tensorlbm.voxelize import mask_from_stl

# ---------------------------------------------------------------------------
# Recorded reference values (docs/gpu_smoke_suite_20260825.md has the citations)
# ---------------------------------------------------------------------------

DEFAULT_CKPT = Path("/nfs/wangxi/runs/b4_serve_20260824/ckpts/serve_cfull_s0.pt")
DEFAULT_CKPT_DIR = Path("/nfs/wangxi/runs/b4_serve_20260824/ckpts")
DEFAULT_RUN_DIR = Path("/nfs/wangxi/runs/b4_v4_20260824")
DEFAULT_PYDEPS = Path("/nfs/wangxi/runs/b4_serve_20260824/pydeps")
DEFAULT_ONNX = Path("/nfs/wangxi/runs/b4_onnx_20260825/ensemble_cfull_stacked.onnx")

#: production serving architecture (train_fno_v4.py ARCH_BASE; the trained
#: checkpoints carry their own arch -- this is only the random-weight
#: latency stand-in when /nfs run artifacts are absent).
ARCH_BASE = dict(
    in_ch=5, width=32, n_layers=4, modes=(16, 32), mlp_hidden=128, film_hidden=64, cond_dim=8
)

#: thresholds; `ref` = the recorded measurement each gate is derived from.
THRESHOLDS: dict[str, dict[str, Any]] = {
    "parity_cpu_cuda": {
        "threshold": 1e-5,
        "unit": "max |log10 cd_cuda - log10 cd_cpu|",
        "ref": "spec draft 1e-6; measured reference value 2.15e-6 (member s0, "
        "median 2e-7, 6/32 rows above 1e-6 -- see doc for the calibration run)",
    },
    "determinism_gpu_repeat": {"threshold": "bitwise equal", "ref": "bitwise on every run"},
    "serve_latency": {
        "threshold": 50.0,
        "unit": "ms p50, 64-Re guarded predict, 5 real members",
        "ref": "17.67 ms p50 recorded in docs/inference_service_20260824.md (17.7 ms table)",
    },
    "echo_latency": {
        "threshold": 150.0,
        "unit": "ms p50, cold-design service call, fixed corpus field row",
        "ref": "38.2 ms recorded end-to-end slider echo incl. geometry front-end; "
        "42.2 ms p50 in runs/b4_echo_20260825/bench_echo.jsonl (cuda slider_move)",
    },
    "onnx_parity": {
        "threshold": 1e-5,
        "unit": "max |log10 cd_ort - log10 cd_torch| over 5 members x 16 rows",
        "ref": "4.65e-7 max on 274 real rows (runs/b4_onnx_20260825/parity_report_stacked.json)",
    },
    "voxelizer_cross_impl": {
        "threshold": 0,
        "unit": "mismatched cells (of 64^3)",
        "ref": "0 mismatches, 33208 solid cells (recorded cross-implementation result)",
    },
    "train_smoke": {
        "threshold": "last5_mean < first5_mean and all finite",
        "ref": "fresh tiny model always satisfies this on a working autograd stack",
    },
    "suboff_mask_cpu_cuda": {
        "threshold": "bitwise equal masks",
        "ref": "4157 solid cells, identical on both devices (production grid)",
    },
}

DEFAULT_REPS = 20
QUICK_REPS = 1
N_PARITY_ROWS = 32
N_ONNX_ROWS = 16
TRAIN_STEPS = 30


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def res(
    name: str,
    status: str,
    measured: Any,
    threshold: Any,
    seconds: float,
    **detail: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "measured": measured,
        "threshold": threshold,
        "seconds": round(seconds, 3),
        "detail": detail,
    }


def _worst(statuses: list[str]) -> str:
    for bad in ("fail", "skip", "pass"):
        if bad in statuses:
            return bad
    return "pass"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_calls(
    fn: Callable[[], Any], reps: int, device: torch.device, warmup: int
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    _sync(device)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        ts.append((time.perf_counter() - t0) * 1e3)
    a = np.asarray(ts, dtype=np.float64)
    return {
        "p50_ms": float(np.quantile(a, 0.50)),
        "mean_ms": float(a.mean()),
        "p95_ms": float(np.quantile(a, 0.95)),
        "reps": int(reps),
    }


def syn_checkpoint(seed: int, arch: dict[str, Any]) -> CondDragCheckpoint:
    """Random-weight member (latency/parity-stand-in; never a quality claim)."""
    torch.manual_seed(seed)
    model = CondFNODrag(**arch)
    return CondDragCheckpoint(
        arch=dict(arch),
        state_dict=model.state_dict(),
        norm=dict(
            ch_mean=np.zeros(arch["in_ch"], dtype=np.float64),
            ch_std=np.ones(arch["in_ch"], dtype=np.float64),
            p_mean=np.zeros(8, dtype=np.float64),
            p_std=np.ones(8, dtype=np.float64),
            y_mean=0.0,
            y_std=1.0,
        ),
        meta=dict(seed=seed, synthetic="random-weights stand-in"),
    )


def synthetic_index() -> CorpusIndex:
    """Latency-only corpus twin (b4_serve_benchmark.py synthetic fallback)."""
    from tensorlbm.ai.drag_cond import condition_v3, geometry_channels, suboff_geometry_features

    rng = np.random.default_rng(0)
    fields = rng.standard_normal((16, 5, 64, 128)).astype(np.float32)
    re_arr = np.geomspace(50.0, 100.0, 16)
    geo = geometry_channels(suboff_geometry_features("full", 1.0, 1.0))
    cond = condition_v3(
        re_arr, np.full(16, 0.1), np.ones(16), np.ones(16), np.broadcast_to(geo, (16, 4))
    )
    return CorpusIndex(
        fields=fields, re=re_arr, designs=tuple([("full", 1.0, 1.0, 0.1)] * 16), cond=cond
    )


def icosphere(subdiv: int) -> np.ndarray:
    """Unit icosphere (20-face icosahedron + midpoint subdivision).

    Same generator as ``tests/test_voxelize.py`` / ``b4_voxelize_bench.py``;
    duplicated inline so the suite has no test-tree dependency.
    """
    t = (1.0 + 5.0**0.5) / 2.0
    verts = [
        (-1, t, 0),
        (1, t, 0),
        (-1, -t, 0),
        (1, -t, 0),
        (0, -1, t),
        (0, 1, t),
        (0, -1, -t),
        (0, 1, -t),
        (t, 0, -1),
        (t, 0, 1),
        (-t, 0, -1),
        (-t, 0, 1),
    ]
    verts = [v / np.linalg.norm(v) for v in verts]
    faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]
    vlist: list[list[float]] = [list(map(float, v)) for v in verts]
    for _ in range(subdiv):
        cache: dict[tuple[int, int], int] = {}
        new_faces: list[tuple[int, int, int]] = []

        def midpoint(i: int, j: int) -> int:
            key = (i, j) if i < j else (j, i)
            if key not in cache:
                m = (np.asarray(vlist[i]) + np.asarray(vlist[j])) / 2.0
                vlist.append(list(m / np.linalg.norm(m)))
                cache[key] = len(vlist) - 1
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces.extend([(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)])
        faces = new_faces
    pts = np.asarray(vlist, dtype=np.float64)
    return pts[np.asarray(faces, dtype=np.int64)]


# The ORT side of the ONNX check runs in a subprocess: pydeps carries its
# own numpy 2.x and mixing it into the live process would shadow the venv
# numpy mid-flight.  argv: onnx_path inputs.npz outputs.npy
_ORT_RUNNER = """
import sys
import numpy as np
import onnxruntime as ort
onnx_path, in_npz, out_npy = sys.argv[1], sys.argv[2], sys.argv[3]
z = np.load(in_npz)
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
names = [i.name for i in sess.get_inputs()]
out_names = [o.name for o in sess.get_outputs()]
target = "member_cd" if "member_cd" in out_names else out_names[0]
member_cd = sess.run([target], {names[0]: z["field"], names[1]: z["cond"]})[0]
np.save(out_npy, np.asarray(member_cd, dtype=np.float64))
print("onnxruntime", ort.__version__, "numpy", np.__version__)
"""


# ---------------------------------------------------------------------------
# the suite
# ---------------------------------------------------------------------------


class SmokeSuite:
    """Holds configuration + lazily loaded shared artifacts."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.cpu_only: bool = bool(args.cpu_only)
        self.reps: int = QUICK_REPS if args.quick else DEFAULT_REPS
        self.warmup: int = 1 if args.quick else 3
        if not self.cpu_only:
            # Select the GPU before the first CUDA call; --gpu wins over an
            # inherited CUDA_VISIBLE_DEVICES.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        self.device = torch.device("cpu" if self.cpu_only else "cuda")
        self.cuda_ok: bool = False
        if not self.cpu_only:
            self.cuda_ok = torch.cuda.is_available()
        # lazily loaded shared artifacts
        self._index: CorpusIndex | None = None
        self._index_source: str = ""
        self._ckpts: list[CondDragCheckpoint] | None = None
        self._ckpts_real: bool | None = None

    # -- shared artifact loaders ------------------------------------------------

    def corpus(self) -> tuple[CorpusIndex | None, str]:
        """Real corpus index when the run dir exists, else None."""
        if self._index is not None or self._index_source:
            return self._index, self._index_source
        run_dir = Path(self.args.run_dir)
        if (run_dir / "cache_v4.npz").is_file() or (run_dir / "cache.npz").is_file():
            self._index = load_corpus_index(run_dir)
            self._index_source = str(run_dir)
        else:
            self._index_source = f"absent: {run_dir}"
        return self._index, self._index_source

    def members(self) -> tuple[list[CondDragCheckpoint], bool]:
        """The 5 serving members, or random-weight twins when ckpts absent."""
        if self._ckpts is not None:
            return self._ckpts, bool(self._ckpts_real)
        ckpt_dir = Path(self.args.ckpt_dir)
        paths = sorted(ckpt_dir.glob("*.pt"))
        if paths:
            self._ckpts = [load_checkpoint(p) for p in paths]
            self._ckpts_real = True
        else:
            self._ckpts = [syn_checkpoint(s, ARCH_BASE) for s in range(5)]
            self._ckpts_real = False
        return self._ckpts, bool(self._ckpts_real)

    def service(self) -> tuple[DragSurrogateService, CorpusIndex, str]:
        """Guarded service on the suite device (real or synthetic corpus)."""
        index, source = self.corpus()
        if index is None:
            index = synthetic_index()
            source = "synthetic (run dir absent -- latencies valid, quality not)"
        ckpts, real = self.members()
        guard = EnvelopeMahalanobisGuardrail(index.cond)
        backend = ModelEnsembleBackend(ckpts, device=self.device)
        svc = DragSurrogateService(
            backend,
            guard,
            corpus_cache=index.fields,
            cache_re=index.re,
            cache_designs=list(index.designs),
        )
        svc.info_weights = "checkpoints" if real else "synthetic random-weight stand-in"  # type: ignore[attr-defined]
        return svc, index, source

    # -- checks -----------------------------------------------------------------


def check_env(suite: SmokeSuite) -> dict[str, Any]:
    t0 = time.perf_counter()
    sub: dict[str, dict[str, Any]] = {}
    sub["torch"] = {"measured": torch.__version__, "threshold": ">=2.0", "status": "pass"}
    sub["tensorlbm"] = {"measured": str(tensorlbm.__file__), "threshold": "info", "status": "pass"}
    sub["python"] = {"measured": sys.executable, "threshold": "info", "status": "pass"}
    cuda = torch.cuda.is_available() if not suite.cpu_only else torch.cuda.is_available()
    want_cuda = not suite.cpu_only
    sub["cuda_available"] = {
        "measured": bool(cuda),
        "threshold": "required unless --cpu-only",
        "status": "pass" if (cuda or not want_cuda) else "fail",
    }
    device_name = driver = None
    if cuda:
        device_name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        sub["device"] = {
            "measured": {"name": device_name, "capability": f"{cap[0]}.{cap[1]}"},
            "threshold": "info",
            "status": "pass",
        }
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--id={suite.args.gpu}",
                "--query-gpu=driver_version,name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        driver = out.stdout.strip()
        sub["driver"] = {"measured": driver, "threshold": "info", "status": "pass"}
    except (OSError, subprocess.SubprocessError) as exc:
        sub["driver"] = {
            "measured": None,
            "threshold": "info",
            "status": "skip",
            "reason": f"nvidia-smi unavailable: {exc}",
        }
    nfs = Path("/nfs")
    if not nfs.is_dir():
        sub["nfs_writable"] = {
            "measured": None,
            "threshold": "probe write",
            "status": "skip",
            "reason": "/nfs not present (fresh checkout)",
        }
        sub["tmpdir_on_nfs"] = {
            "measured": None,
            "threshold": "effective temp dir under /nfs",
            "status": "skip",
            "reason": "/nfs not present (fresh checkout)",
        }
    else:
        probe_dir = Path(tempfile.gettempdir())
        if str(probe_dir).startswith("/nfs"):
            probe_dir.mkdir(parents=True, exist_ok=True)
        else:
            probe_dir = Path("/nfs/wangxi/tmp") if Path("/nfs/wangxi/tmp").is_dir() else nfs
        try:
            probe = probe_dir / f".gpu_smoke_probe_{os.getpid()}"
            probe.write_text("probe")
            probe.unlink()
            sub["nfs_writable"] = {
                "measured": str(probe_dir),
                "threshold": "probe write",
                "status": "pass",
            }
        except OSError as exc:
            sub["nfs_writable"] = {
                "measured": str(probe_dir),
                "threshold": "probe write",
                "status": "fail",
                "reason": str(exc),
            }
        effective = str(Path(tempfile.gettempdir()).resolve())
        tmp_env = os.environ.get("TMPDIR")
        ok = effective.startswith("/nfs")
        sub["tmpdir_on_nfs"] = {
            "measured": {"TMPDIR": tmp_env, "effective": effective},
            "threshold": "effective temp dir under /nfs (root partition is small)",
            "status": "pass" if ok else "fail",
            "reason": None if ok else "export TMPDIR=/nfs/wangxi/tmp before running",
        }
    statuses = [v["status"] for v in sub.values()]
    return res(
        "env_sanity",
        _worst(statuses),
        {k: v["measured"] for k, v in sub.items()},
        "cuda in GPU mode; /nfs + TMPDIR on /nfs when /nfs exists",
        time.perf_counter() - t0,
        sub_checks=sub,
    )


def _fixed_batch(suite: SmokeSuite) -> tuple[np.ndarray, np.ndarray] | None:
    """Field row 0 + first N_PARITY_ROWS condition rows of the real corpus."""
    index, _ = suite.corpus()
    if index is None:
        return None
    return index.fields[0], np.asarray(index.cond[:N_PARITY_ROWS])


def _forward_log10(
    ckpt: CondDragCheckpoint, field: np.ndarray, cond: np.ndarray, device: str
) -> np.ndarray:
    """One member forward via load_checkpoint().to_model(), log10 C_D out.

    Mirrors ``ModelEnsembleBackend.predict`` normalisation exactly
    (z-scored field + condition, z-scored log10 C_D target).
    """
    model = ckpt.to_model(device)
    norm = ckpt.norm
    x = torch.from_numpy(np.asarray(field, dtype=np.float32)).to(device)
    p = torch.from_numpy(np.asarray(cond, dtype=np.float64).astype(np.float32)).to(device)
    ch_m = torch.as_tensor(norm["ch_mean"], dtype=torch.float32, device=device).view(1, -1, 1, 1)
    ch_s = torch.as_tensor(norm["ch_std"], dtype=torch.float32, device=device).view(1, -1, 1, 1)
    p_m = torch.as_tensor(norm["p_mean"], dtype=torch.float32, device=device)
    p_s = torch.as_tensor(norm["p_std"], dtype=torch.float32, device=device)
    xn = (x.unsqueeze(0).expand(p.shape[0], -1, -1, -1) - ch_m) / ch_s
    with torch.no_grad():
        z = model(xn, (p - p_m) / p_s)
    z32 = z.float().cpu().double().numpy()
    return z32 * float(norm["y_std"]) + float(norm["y_mean"])


def check_parity_cpu_cuda(suite: SmokeSuite) -> dict[str, Any]:
    t0 = time.perf_counter()
    spec = THRESHOLDS["parity_cpu_cuda"]
    ckpt_path = Path(suite.args.ckpt)
    if not ckpt_path.is_file():
        return res(
            "parity_cpu_cuda",
            "skip",
            None,
            spec["threshold"],
            time.perf_counter() - t0,
            reason=f"serving checkpoint absent: {ckpt_path}",
            ref=spec["ref"],
        )
    batch = _fixed_batch(suite)
    if batch is None:
        return res(
            "parity_cpu_cuda",
            "skip",
            None,
            spec["threshold"],
            time.perf_counter() - t0,
            reason=f"corpus run dir absent: {suite.args.run_dir}",
            ref=spec["ref"],
        )
    if not suite.cuda_ok:
        return res(
            "parity_cpu_cuda",
            "skip",
            None,
            spec["threshold"],
            time.perf_counter() - t0,
            reason="--cpu-only / CUDA unavailable (cross-device parity needs both)",
            ref=spec["ref"],
        )
    field, cond = batch
    ckpt = load_checkpoint(ckpt_path)
    z_cpu = _forward_log10(ckpt, field, cond, "cpu")
    z_cuda = _forward_log10(ckpt, field, cond, "cuda")
    diff = np.abs(z_cuda - z_cpu)
    max_diff = float(diff.max())
    ok = max_diff <= float(spec["threshold"])
    return res(
        "parity_cpu_cuda",
        "pass" if ok else "fail",
        {
            "max_log10_diff": max_diff,
            "median_log10_diff": float(np.median(diff)),
            "rows_over_1e-6": int((diff > 1e-6).sum()),
            "rows": int(diff.size),
        },
        spec["threshold"],
        time.perf_counter() - t0,
        checkpoint=str(ckpt_path),
        batch={"field_row": 0, "cond_rows": N_PARITY_ROWS, "member_meta": ckpt.meta},
        ref=spec["ref"],
    )


def check_determinism_gpu_repeat(suite: SmokeSuite) -> dict[str, Any]:
    t0 = time.perf_counter()
    spec = THRESHOLDS["determinism_gpu_repeat"]
    if not suite.cuda_ok:
        return res(
            "determinism_gpu_repeat",
            "skip",
            None,
            spec["threshold"],
            time.perf_counter() - t0,
            reason="--cpu-only / CUDA unavailable",
            ref=spec["ref"],
        )
    batch = _fixed_batch(suite)
    ckpt_path = Path(suite.args.ckpt)
    if batch is not None and ckpt_path.is_file():
        field, cond = batch
        ckpt = load_checkpoint(ckpt_path)
        weights = str(ckpt_path)
    else:
        index, source = suite.corpus()
        if index is None:
            synth = synthetic_index()
            field, cond = synth.fields[0], synth.cond[:N_PARITY_ROWS]
        else:
            field, cond = index.fields[0], np.asarray(index.cond[:N_PARITY_ROWS])
        ckpt = syn_checkpoint(0, ARCH_BASE)
        weights = f"synthetic random-weight stand-in (real ckpt absent: {ckpt_path})"
    z1 = _forward_log10(ckpt, field, cond, "cuda")
    z2 = _forward_log10(ckpt, field, cond, "cuda")
    bitwise = bool(np.array_equal(z1.view(np.float64), z2.view(np.float64)))
    return res(
        "determinism_gpu_repeat",
        "pass" if bitwise else "fail",
        {"bitwise_equal": bitwise},
        spec["threshold"],
        time.perf_counter() - t0,
        weights=weights,
        ref=spec["ref"],
    )


def check_serve_latency(suite: SmokeSuite) -> dict[str, Any]:
    t0 = time.perf_counter()
    spec = THRESHOLDS["serve_latency"]
    if not suite.cuda_ok:
        return res(
            "serve_latency",
            "skip",
            None,
            spec["threshold"],
            time.perf_counter() - t0,
            reason="GPU latency gate (recorded CPU p50 is 1297 ms -- different budget)",
            ref=spec["ref"],
        )
    svc, index, source = suite.service()
    re_grid = np.geomspace(float(index.re.min()), float(index.re.max()), 64)
    query = dict(hull_type="full", sail_scale=1.0, fin_scale=1.0, u_in=0.1)
    stats = time_calls(
        lambda: svc.predict(**query, re_grid=re_grid), suite.reps, suite.device, suite.warmup
    )
    ok = stats["p50_ms"] < float(spec["threshold"])
    return res(
        "serve_latency",
        "pass" if ok else "fail",
        stats,
        spec["threshold"],
        time.perf_counter() - t0,
        corpus=source,
        weights=svc.info_weights,  # type: ignore[attr-defined]
        query=dict(query, n_re=64),
        ref=spec["ref"],
    )


def check_echo_latency(suite: SmokeSuite) -> dict[str, Any]:
    """Inline twin of the CAD-slider echo hot path (no unmerged deps).

    Measures what the echo pipeline spends per cold slider move once the
    corpus field cache is attached: NEW design every call (cold geometry
    feature computation) -> condition rows -> guard -> fixed corpus-cache
    field row -> 5-member forward.  The CAD mask/SDF front-end
    (GeometryEchoPipeline, recorded 16.3 ms GPU / 20.6 ms CPU p50) is NOT
    in this number -- it is not part of the service call.
    """
    t0 = time.perf_counter()
    spec = THRESHOLDS["echo_latency"]
    if not suite.cuda_ok:
        return res(
            "echo_latency",
            "skip",
            None,
            spec["threshold"],
            time.perf_counter() - t0,
            reason="GPU latency gate (recorded CPU slider p50 is 1298 ms -- different budget)",
            ref=spec["ref"],
        )
    svc, index, source = suite.service()
    re_grid = np.geomspace(float(index.re.min()), float(index.re.max()), 64)
    sails = np.geomspace(0.7, 1.6, suite.reps + suite.warmup + 1)

    def one_call(i: int) -> Any:
        return svc.predict(
            hull_type="full",
            sail_scale=float(sails[i % sails.size]),  # distinct value: LRU never hits
            fin_scale=1.0,
            re_grid=re_grid,
            u_in=0.1,
            field_point=0,  # corpus-cache field row
        )

    counter = {"i": 0}

    def call() -> Any:
        counter["i"] += 1
        return one_call(counter["i"])

    stats = time_calls(call, suite.reps, suite.device, suite.warmup)
    ok = stats["p50_ms"] < float(spec["threshold"])
    return res(
        "echo_latency",
        "pass" if ok else "fail",
        stats,
        spec["threshold"],
        time.perf_counter() - t0,
        corpus=source,
        weights=svc.info_weights,  # type: ignore[attr-defined]
        path="cold-design condition+guard+ensemble with fixed corpus field row 0; "
        "CAD mask/SDF front-end excluded (recorded 16.3 ms GPU p50 on top)",
        ref=spec["ref"],
    )


def check_onnx_parity(suite: SmokeSuite) -> dict[str, Any]:
    t0 = time.perf_counter()
    spec = THRESHOLDS["onnx_parity"]
    pydeps = Path(suite.args.pydeps)
    onnx_path = Path(suite.args.onnx)
    if not pydeps.is_dir():
        return res(
            "onnx_parity",
            "skip",
            None,
            spec["threshold"],
            time.perf_counter() - t0,
            reason=f"private ORT pydeps absent: {pydeps}",
            ref=spec["ref"],
        )
    if not onnx_path.is_file():
        return res(
            "onnx_parity",
            "skip",
            None,
            spec["threshold"],
            time.perf_counter() - t0,
            reason=f"ONNX artifact absent: {onnx_path}",
            ref=spec["ref"],
        )
    batch = _fixed_batch(suite)
    if batch is None:
        return res(
            "onnx_parity",
            "skip",
            None,
            spec["threshold"],
            time.perf_counter() - t0,
            reason=f"corpus run dir absent: {suite.args.run_dir}",
            ref=spec["ref"],
        )
    ckpts, real = suite.members()
    if not real:
        return res(
            "onnx_parity",
            "skip",
            None,
            spec["threshold"],
            time.perf_counter() - t0,
            reason="serving checkpoints absent (ONNX artifact was exported from the "
            "trained members; a random-weight torch reference would not parity)",
            ref=spec["ref"],
        )
    field, cond = batch
    field16 = np.asarray(field[None, ...], dtype=np.float32)  # (1, 5, ny, nx)
    cond16 = np.asarray(cond[:N_ONNX_ROWS], dtype=np.float64).astype(np.float32)
    work = Path(tempfile.gettempdir())
    in_npz = work / "gpu_smoke_onnx_inputs.npz"
    out_npy = work / "gpu_smoke_onnx_ort.npy"
    np.savez(in_npz, field=field16, cond=cond16)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(pydeps) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    t1 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-c", _ORT_RUNNER, str(onnx_path), str(in_npz), str(out_npy)],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    ort_s = time.perf_counter() - t1
    if proc.returncode != 0 or not out_npy.is_file():
        return res(
            "onnx_parity",
            "fail",
            {"returncode": proc.returncode, "stderr": proc.stderr[-800:]},
            spec["threshold"],
            time.perf_counter() - t0,
            reason="ORT subprocess failed",
            ref=spec["ref"],
        )
    ort_out = np.load(out_npy)  # (M, N) linear C_D
    backend = ModelEnsembleBackend(ckpts, device="cpu")  # recorded reference device = cpu
    torch_out = backend.predict(field, np.asarray(cond[:N_ONNX_ROWS], dtype=np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.abs(np.log10(ort_out) - np.log10(torch_out))
    max_diff = float(d.max())
    ok = max_diff < float(spec["threshold"])
    return res(
        "onnx_parity",
        "pass" if ok else "fail",
        {"max_log10_diff": max_diff, "shape": list(ort_out.shape)},
        spec["threshold"],
        time.perf_counter() - t0,
        onnx=str(onnx_path),
        ort_info=proc.stdout.strip(),
        ort_subprocess_s=round(ort_s, 3),
        rows=N_ONNX_ROWS,
        ref=spec["ref"],
    )


def check_voxelizer_cross_impl(suite: SmokeSuite) -> dict[str, Any]:
    """mask_from_stl vs voxelize_stl_reference on the canonical test mesh.

    Both implementations sample cell centres at origin + (i+0.5)*spacing
    (origin = lower corner), so identical arguments are directly
    comparable.  CPU-only by design (the reference is the pure-torch
    float64 path).
    """
    t0 = time.perf_counter()
    spec = THRESHOLDS["voxelizer_cross_impl"]
    from tensorlbm.geometry_voxel import voxelize_stl_reference

    tris = icosphere(3) * 5.0  # radius 5, subdiv 3 (1280 triangles)
    shape = (64, 64, 64)
    origin = (-8.0, -8.0, -8.0)
    spacing = 0.25
    t1 = time.perf_counter()
    m_vox = mask_from_stl(tris, shape, origin=origin, spacing=spacing)
    t_vox = time.perf_counter() - t1
    t1 = time.perf_counter()
    m_ref, _boundary, _q = voxelize_stl_reference(
        tris, shape, origin=origin, spacing=(spacing, spacing, spacing)
    )
    t_ref = time.perf_counter() - t1
    m_ref_np = m_ref.numpy() if isinstance(m_ref, torch.Tensor) else np.asarray(m_ref)
    mism = int(np.logical_xor(m_vox, m_ref_np).sum())
    ok = mism == int(spec["threshold"])
    return res(
        "voxelizer_cross_impl",
        "pass" if ok else "fail",
        {
            "mismatched_cells": mism,
            "solid_voxelize": int(m_vox.sum()),
            "solid_reference": int(m_ref_np.sum()),
        },
        spec["threshold"],
        time.perf_counter() - t0,
        mesh={"icosphere": "radius 5, subdiv 3", "triangles": int(tris.shape[0])},
        grid={"shape": list(shape), "origin": list(origin), "spacing": spacing},
        mask_from_stl_s=round(t_vox, 3),
        reference_s=round(t_ref, 3),
        ref=spec["ref"],
    )


def check_train_smoke(suite: SmokeSuite) -> dict[str, Any]:
    t0 = time.perf_counter()
    spec = THRESHOLDS["train_smoke"]
    dev = suite.device
    torch.manual_seed(0)
    model = CondFNODrag(
        in_ch=5, width=8, n_layers=2, modes=(4, 8), mlp_hidden=32, film_hidden=32, cond_dim=8
    ).to(dev)
    g = torch.Generator(device="cpu").manual_seed(7)
    fields = torch.randn(8, 5, 16, 32, generator=g).to(dev)
    cond = torch.randn(8, 8, generator=g).to(dev)
    # fixed synthetic teacher: smooth function of the inputs, learnable in 30 steps
    target = (
        0.5 * fields[:, :2].mean(dim=(1, 2, 3)) - 0.3 * cond[:, 0] + 0.2 * cond[:, 1]
    ).detach()
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    losses: list[float] = []
    grad_finite = False
    for _step in range(TRAIN_STEPS):
        opt.zero_grad()
        z = model(fields, cond).squeeze(-1)
        loss = torch.nn.functional.mse_loss(z, target)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        grad_finite = all(torch.isfinite(gr).all().item() for gr in grads)
        if not grad_finite or not torch.isfinite(loss).item():
            break
        opt.step()
        losses.append(float(loss.item()))
    if not losses:
        return res(
            "train_smoke",
            "fail",
            {"losses": losses, "grad_finite": grad_finite},
            spec["threshold"],
            time.perf_counter() - t0,
            reason="non-finite loss/grad before step 1",
            ref=spec["ref"],
        )
    n5 = min(5, len(losses) // 2 or 1)
    first5 = float(np.mean(losses[:n5]))
    last5 = float(np.mean(losses[-n5:]))
    all_finite = bool(np.isfinite(losses).all()) and grad_finite
    ok = all_finite and last5 < first5
    return res(
        "train_smoke",
        "pass" if ok else "fail",
        {
            "first5_mean_loss": first5,
            "last5_mean_loss": last5,
            "all_finite": all_finite,
            "backward_works": grad_finite,
            "steps": len(losses),
        },
        spec["threshold"],
        time.perf_counter() - t0,
        arch={"width": 8, "n_layers": 2, "modes": [4, 8]},
        device=str(dev),
        ref=spec["ref"],
    )


def check_suboff_mask_cpu_cuda(suite: SmokeSuite) -> dict[str, Any]:
    """Production-grid SUBOFF mask, mother config, CPU vs GPU."""
    t0 = time.perf_counter()
    spec = THRESHOLDS["suboff_mask_cpu_cuda"]
    if not suite.cuda_ok:
        return res(
            "suboff_mask_cpu_cuda",
            "skip",
            None,
            spec["threshold"],
            time.perf_counter() - t0,
            reason="--cpu-only / CUDA unavailable (cross-device mask identity needs both)",
            ref=spec["ref"],
        )
    # mother config (all mults 1.0) on the production 128-resolution grid
    # (PRODUCTION_GRID: nx=128, ny=nz=64, cx=0.35*nx, length=0.6*nx)
    kwargs = dict(
        hull_type="full",
        nx=128,
        ny=64,
        nz=64,
        cx=128 * 0.35,
        cy=32.0,
        cz=32.0,
        length=0.6 * 128,
        config=SuboffConfig(),
    )
    mask_cpu, stats = build_suboff_mask(device="cpu", **kwargs)
    mask_cuda, _ = build_suboff_mask(device="cuda", **kwargs)
    bitwise = bool(torch.equal(mask_cpu, mask_cuda.cpu()))
    shape_ok = tuple(mask_cpu.shape) == (64, 64, 128) and mask_cpu.dtype == torch.bool
    solid = int(mask_cpu.sum())
    force_layout_ok = shape_ok and solid == int(stats["solid_cells"])
    ok = bitwise and shape_ok and force_layout_ok
    return res(
        "suboff_mask_cpu_cuda",
        "pass" if ok else "fail",
        {
            "bitwise_equal": bitwise,
            "solid_cells": solid,
            "shape": list(mask_cpu.shape),
            "dtype": str(mask_cpu.dtype),
            "stats_consistent": force_layout_ok,
        },
        spec["threshold"],
        time.perf_counter() - t0,
        grid="production 128-resolution (64, 64, 128) streamwise-x layout",
        config="SuboffConfig() mother defaults (all mults 1.0)",
        ref=spec["ref"],
    )


CHECKS: list[Callable[[SmokeSuite], dict[str, Any]]] = [
    check_env,
    check_parity_cpu_cuda,
    check_determinism_gpu_repeat,
    check_serve_latency,
    check_echo_latency,
    check_onnx_parity,
    check_voxelizer_cross_impl,
    check_train_smoke,
    check_suboff_mask_cpu_cuda,
]


def summary_table(checks: list[dict[str, Any]]) -> str:
    name_w = max(len(c["name"]) for c in checks) + 2
    head = f"{'CHECK'.ljust(name_w)}{'STATUS':<8}{'MEASURED':<34}{'THRESHOLD':<28}SECONDS"
    lines = [head, "-" * len(head)]
    for c in checks:
        m = c["measured"]
        if isinstance(m, dict):
            key = (
                "max_log10_diff"
                if "max_log10_diff" in m
                else "p50_ms"
                if "p50_ms" in m
                else "mismatched_cells"
                if "mismatched_cells" in m
                else "last5_mean_loss"
                if "last5_mean_loss" in m
                else "bitwise_equal"
                if "bitwise_equal" in m
                else "sub_checks"
            )
            m_txt = json.dumps(m.get(key, m), default=str)
        else:
            m_txt = "-" if m is None else str(m)
        if len(m_txt) > 33:
            m_txt = m_txt[:30] + "..."
        th = c["threshold"]
        th_txt = th if isinstance(th, str) else json.dumps(th)
        if len(th_txt) > 27:
            th_txt = th_txt[:24] + "..."
        lines.append(
            f"{c['name'].ljust(name_w)}{c['status']:<8}{m_txt:<34}{th_txt:<28}{c['seconds']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TensorLBM GPU smoke / verification suite")
    ap.add_argument("--gpu", type=int, default=4, help="GPU index (default 4)")
    ap.add_argument("--out", default="gpu_smoke_report.json", help="report JSON path")
    ap.add_argument("--quick", action="store_true", help="1 latency rep instead of 20")
    ap.add_argument("--cpu-only", action="store_true", help="CI mode: GPU checks skip cleanly")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="serving checkpoint (parity)")
    ap.add_argument("--ckpt-dir", default=str(DEFAULT_CKPT_DIR), help="5-member ckpt dir")
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR), help="B4 corpus run dir")
    ap.add_argument("--pydeps", default=str(DEFAULT_PYDEPS), help="private ORT pydeps dir")
    ap.add_argument("--onnx", default=str(DEFAULT_ONNX), help="stacked ensemble ONNX artifact")
    args = ap.parse_args(argv)

    suite = SmokeSuite(args)
    print(f"tensorlbm : {tensorlbm.__file__}")
    print(
        f"torch     : {torch.__version__}  device: {suite.device}"
        f"{' (' + torch.cuda.get_device_name(0) + ')' if suite.cuda_ok else ''}"
    )
    print(f"python    : {sys.executable}")
    print(f"reps      : {suite.reps} (quick={args.quick}, cpu-only={args.cpu_only})")
    print()

    checks = []
    for fn in CHECKS:
        t0 = time.perf_counter()
        try:
            r = fn(suite)
        except Exception as exc:  # noqa: BLE001 -- a smoke suite reports, never dies
            r = res(
                fn.__name__.removeprefix("check_"),
                "fail",
                {"error": f"{type(exc).__name__}: {exc}"},
                "see detail",
                time.perf_counter() - t0,
            )
        checks.append(r)
        print(
            f"[{r['status'].upper():<5}] {r['name']} ({r['seconds']} s)"
            + (f"  -- {r['detail'].get('reason', '')}" if r["status"] == "skip" else "")
        )

    counts = {s: sum(1 for c in checks if c["status"] == s) for s in ("pass", "fail", "skip")}
    report: dict[str, Any] = {
        "suite": "gpu_smoke_suite",
        "version": 1,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": {
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": (torch.cuda.get_device_name(0) if suite.cuda_ok else None),
            "python": sys.executable,
            "tensorlbm": str(tensorlbm.__file__),
            "argv": sys.argv[1:],
        },
        "thresholds": {k: v["threshold"] for k, v in THRESHOLDS.items()},
        "summary": counts,
        "checks": checks,
    }
    out_path = Path(args.out)
    if out_path.parent != Path(""):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1, default=str))
    print()
    print(summary_table(checks))
    print()
    print(
        f"pass {counts['pass']}  fail {counts['fail']}  skip {counts['skip']}"
        f"   -> report: {out_path}"
    )
    return 1 if counts["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
