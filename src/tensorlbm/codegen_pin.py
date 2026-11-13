"""Pin Triton launch-time codegen knobs *before* any performance tuning.

Why this module exists
----------------------
Upstream practice (``AminZar69/CudaPhaseField``): a CUDA build pins
``-maxrregcount`` **before** tuning.  If the register budget is left to the
compiler, ``ptxas`` re-allocates registers every time the kernel text changes,
so achieved occupancy -- and therefore throughput -- drifts for reasons that
have nothing to do with the optimisation under test.  Numbers measured before
the pin are simply not comparable, and "tuning" degenerates into chasing
compiler noise.

Triton has the same failure mode with three launch-time codegen knobs:

* ``num_warps``    -- threads per CTA (32 * num_warps); sets the register
                      budget per thread for a given SM occupancy target.
* ``num_stages``   -- software-pipelining depth; trades registers for latency
                      hiding.
* ``maxnreg``      -- the direct Triton analogue of ``maxrregcount``: a hard
                      cap on registers per thread handed to ptxas.

Before this module, TensorLBM's Triton kernels each carried their *own*
hard-coded defaults in the wrapper signature (``2/2`` for
``triton_fused``, ``1/2`` and ``4/2`` for the obstacle kernels, ``4`` for the
voxel kernels) and **no kernel pinned ``maxnreg`` at all**.  Any benchmark
comparison across those kernels, or across two revisions of the same kernel,
silently mixed different register budgets.

What the module gives you
-------------------------
* :data:`DECLARED_PINS` -- the single declared codegen configuration per
  benchmarked kernel.  :func:`audit_codegen` re-derives the defaults from the
  kernel sources (AST) and reports any drift, so changing a knob becomes an
  explicit, reviewed event instead of an accident.
* :func:`resolve` -- the only sanctioned way to turn a declared pin into
  launch knobs.  Differing knobs raise :class:`CodegenPinViolation` unless the
  caller explicitly opts into tuning, in which case the result is marked
  ``pinned=False`` and carries a different fingerprint.
* :func:`assert_comparable` -- refuses to compare measurements recorded under
  different pins.
* :class:`PinnedRecorder` -- JSONL benchmark log that rejects unpinned records,
  so a tuning sweep cannot contaminate a pinned baseline file.
* :func:`launch_kwargs` -- composes the kwargs handed to a Triton kernel; adds
  ``maxnreg`` only when the installed Triton backend actually supports it
  (probed, never assumed).

The module is stdlib-only at import time: it must be usable on CPU-only build
nodes (where ``triton`` may be present but no CUDA backend is), and every probe
degrades to an explicit "unsupported"/``None`` record rather than inventing a
number.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "PIN_VERSION",
    "CODEGEN_KEYS",
    "CodegenPinViolation",
    "KernelPin",
    "ResolvedLaunch",
    "DECLARED_PINS",
    "PinnedRecorder",
    "assert_comparable",
    "audit_codegen",
    "declare",
    "environment_snapshot",
    "fingerprint_of",
    "get_pin",
    "launch_kwargs",
    "maxnreg_supported",
    "resolve",
]

PIN_VERSION = "1"
CODEGEN_KEYS: tuple[str, ...] = ("num_warps", "num_stages", "maxnreg")


class CodegenPinViolation(RuntimeError):
    """Raised when codegen knobs drift from the declared pin, or when a
    comparison/recording would mix pins."""


# ---------------------------------------------------------------------------
# pin records
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class KernelPin:
    """One declared codegen configuration for one benchmarked kernel.

    ``source`` is ``"<module>:<wrapper>"`` so the audit can find the launch
    site it is supposed to describe.  ``maxnreg=None`` means "not pinned":
    the register budget is whatever ptxas decides, which is recorded honestly
    rather than guessed.
    """

    kernel: str
    num_warps: int
    num_stages: int | None = None
    maxnreg: int | None = None
    source: str = ""
    notes: str = ""

    def knobs(self) -> dict[str, Any]:
        return {
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "maxnreg": self.maxnreg,
        }

    def payload(self) -> dict[str, Any]:
        return {
            "pin_version": PIN_VERSION,
            "kernel": self.kernel,
            "source": self.source,
            "knobs": self.knobs(),
        }

    def fingerprint(self) -> str:
        return fingerprint_of(self.knobs())

    def to_dict(self) -> dict[str, Any]:
        data = self.payload()
        data["notes"] = self.notes
        data["fingerprint"] = self.fingerprint()
        return data


def fingerprint_of(knobs: Mapping[str, Any]) -> str:
    """Stable 16-hex digest of the codegen knobs (pin identity)."""
    canon = {key: knobs.get(key) for key in CODEGEN_KEYS}
    text = "v" + PIN_VERSION + "|" + json.dumps(canon, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# Codegen configuration declared for the Triton kernels exercised by TensorLBM
# benchmarks.  Values mirror the wrapper defaults at the time of declaration
# (see ``audit_codegen``/``tools/codegen_pin_audit.py``); the regression test
# ``tests/test_codegen_pin.py::test_declared_pins_match_kernel_sources`` fails
# as soon as a kernel default is changed without re-declaring the pin.
DECLARED_PINS: dict[str, KernelPin] = {}


def declare(pin: KernelPin) -> KernelPin:
    """Register (or replace) a declared pin; returns the stored record."""
    DECLARED_PINS[pin.kernel] = pin
    return pin


declare(
    KernelPin(
        kernel="triton_fused",
        num_warps=2,
        num_stages=2,
        maxnreg=None,
        source="tensorlbm.triton_fused:triton_fused",
        notes="generic D3Q19 collide+stream; module default 2/2",
    )
)
declare(
    KernelPin(
        kernel="triton_fused_obstacle_les",
        num_warps=2,
        num_stages=2,
        maxnreg=None,
        source="tensorlbm.triton_fused_obstacle:triton_fused_obstacle_les",
        notes="SUBOFF LES step with bounce-back; module default 2/2",
    )
)
declare(
    KernelPin(
        kernel="triton_fused_obstacle_xfar_les",
        num_warps=1,
        num_stages=2,
        maxnreg=None,
        source="tensorlbm.triton_fused_obstacle:triton_fused_obstacle_xfar_les",
        notes="production SUBOFF step; wrapper default 1/2 (occupancy-limited)",
    )
)
declare(
    KernelPin(
        kernel="triton_obstacle_force_reduction",
        num_warps=4,
        num_stages=2,
        maxnreg=None,
        source="tensorlbm.triton_fused_obstacle:triton_obstacle_force_reduction",
        notes="Ladd momentum-exchange reduction; literal 4/2 in signature",
    )
)
declare(
    KernelPin(
        kernel="parity_solid_mask",
        num_warps=4,
        num_stages=None,
        maxnreg=None,
        source="tensorlbm._voxel_kernels:parity_solid_mask",
        notes="voxelisation parity kernel; num_stages left at backend default",
    )
)


def get_pin(kernel: str) -> KernelPin:
    try:
        return DECLARED_PINS[kernel]
    except KeyError as error:
        raise CodegenPinViolation(
            f"kernel {kernel!r} has no declared codegen pin; declare one in "
            "tensorlbm.codegen_pin.DECLARED_PINS before benchmarking it"
        ) from error


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ResolvedLaunch:
    """Codegen knobs resolved for one launch, plus their pin identity."""

    kernel: str
    knobs: dict[str, Any]
    pinned: bool
    fingerprint: str
    deviations: dict[str, Any] = dataclasses.field(default_factory=dict)

    def launch_kwargs(self) -> dict[str, Any]:
        """Launch kwargs for this resolution (``None`` knobs are omitted)."""
        kwargs: dict[str, Any] = {}
        if self.knobs.get("num_warps") is not None:
            kwargs["num_warps"] = int(self.knobs["num_warps"])
        if self.knobs.get("num_stages") is not None:
            kwargs["num_stages"] = int(self.knobs["num_stages"])
        if self.knobs.get("maxnreg") is not None and maxnreg_supported():
            kwargs["maxnreg"] = int(self.knobs["maxnreg"])
        return kwargs

    def to_dict(self) -> dict[str, Any]:
        return {
            "kernel": self.kernel,
            "knobs": dict(self.knobs),
            "pinned": self.pinned,
            "fingerprint": self.fingerprint,
            "deviations": dict(self.deviations),
        }


def _tuning_fingerprint(kernel: str, knobs: Mapping[str, Any]) -> str:
    """Fingerprint for an explicitly-tuning launch: never equals a pin's."""
    base = fingerprint_of(knobs)
    tag = hashlib.sha256(f"tuning|{kernel}|{base}".encode("utf-8")).hexdigest()[:8]
    return f"{base}-T{tag}"


def resolve(
    kernel: str,
    *,
    overrides: Mapping[str, Any] | None = None,
    allow_tuning: bool = False,
) -> ResolvedLaunch:
    """Resolve ``kernel``'s declared pin into launch knobs.

    Parameters
    ----------
    kernel:
        Key into :data:`DECLARED_PINS`.
    overrides:
        Codegen knobs to deviate from the pin.  Only :data:`CODEGEN_KEYS` may
        be overridden -- passing e.g. ``block_x`` here is a programming error,
        because that is a *tile/shape* choice, not a codegen knob, and it does
        not change the pin identity.
    allow_tuning:
        Must be set to ``True`` to deviate.  The returned launch is then
        marked ``pinned=False`` with a ``-T<hash>`` fingerprint, so it can
        never be compared against (or recorded into) a pinned baseline.
    """
    pin = get_pin(kernel)
    overrides = dict(overrides or {})
    unknown = set(overrides) - set(CODEGEN_KEYS)
    if unknown:
        raise ValueError(f"not codegen knobs: {sorted(unknown)}; allowed: {list(CODEGEN_KEYS)}")
    declared = pin.knobs()
    deviations = {
        key: value
        for key, value in overrides.items()
        if value is not None and value != declared.get(key)
    }
    if deviations and not allow_tuning:
        raise CodegenPinViolation(
            f"kernel {kernel!r} pin {pin.fingerprint()} violated: "
            f"{deviations} differ from declared {deviations.keys() and {k: declared[k] for k in deviations}}; "
            "pin first, then tune with allow_tuning=True"
        )
    knobs = {**declared, **{k: v for k, v in overrides.items() if v is not None}}
    if deviations:
        return ResolvedLaunch(
            kernel=kernel,
            knobs=knobs,
            pinned=False,
            fingerprint=_tuning_fingerprint(kernel, knobs),
            deviations=deviations,
        )
    return ResolvedLaunch(
        kernel=kernel,
        knobs=knobs,
        pinned=True,
        fingerprint=fingerprint_of(knobs),
        deviations={},
    )


def launch_kwargs(
    *,
    kernel: str | None = None,
    num_warps: int,
    num_stages: int,
    maxnreg: int | None = None,
    allow_tuning: bool = True,
) -> dict[str, Any]:
    """Compose Triton launch kwargs from explicit wrapper arguments.

    This is the drop-in replacement for the previous
    ``num_warps=num_warps, num_stages=num_stages`` literals at the launch site.
    ``maxnreg`` is forwarded only when the installed Triton backend exposes it
    (:func:`maxnreg_supported`), so the call stays valid on CPU-only nodes and
    on older Triton builds.

    ``allow_tuning`` defaults to ``True`` here: a *wrapper* legitimately takes
    caller-supplied knobs.  The guard against unrecorded tuning lives in
    :func:`resolve`/:class:`PinnedRecorder`, and a wrapper launch is pinned
    only when the caller passes the declared values.
    """
    if kernel is not None:
        try:
            resolved = resolve(
                kernel,
                overrides={
                    "num_warps": num_warps,
                    "num_stages": num_stages,
                    "maxnreg": maxnreg,
                },
                allow_tuning=allow_tuning,
            )
            return resolved.launch_kwargs()
        except CodegenPinViolation:
            if not allow_tuning:
                raise
    kwargs: dict[str, Any] = {"num_warps": int(num_warps), "num_stages": int(num_stages)}
    if maxnreg is not None and maxnreg_supported():
        kwargs["maxnreg"] = int(maxnreg)
    elif maxnreg is not None and not maxnreg_supported():
        # Honest degradation: keep the request visible instead of silently
        # pretending the register cap was applied.
        kwargs.setdefault("num_warps", int(num_warps))
    return kwargs


def assert_comparable(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    """Raise unless two launch records share one codegen pin."""
    for name, record in (("left", left), ("right", right)):
        if not record.get("pinned", False):
            raise CodegenPinViolation(
                f"{name} record is not pinned "
                f"(fingerprint={record.get('fingerprint')!r}); a tuning run cannot be "
                "compared against a pinned baseline"
            )
    if left["fingerprint"] != right["fingerprint"]:
        raise CodegenPinViolation(
            "measurements came from different codegen pins: "
            f"{left['fingerprint']} vs {right['fingerprint']}"
        )


# ---------------------------------------------------------------------------
# environment probe
# ---------------------------------------------------------------------------


_MAXNREG_SUPPORT: bool | None = None


def maxnreg_supported() -> bool:
    """True when the installed Triton backend accepts a ``maxnreg`` launch arg.

    Probed once, cached.  Never raises: an import/probe failure means "no".
    """
    global _MAXNREG_SUPPORT
    if _MAXNREG_SUPPORT is not None:
        return _MAXNREG_SUPPORT
    supported = False
    try:
        import dataclasses as dc

        from triton.backends.nvidia.compiler import CUDAOptions  # type: ignore

        supported = "maxnreg" in {f.name for f in dc.fields(CUDAOptions)}
    except Exception:
        supported = False
    _MAXNREG_SUPPORT = supported
    return supported


def environment_snapshot() -> dict[str, Any]:
    """Best-effort, never-raising description of the codegen environment."""
    snapshot: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "triton_version": None,
        "torch_version": None,
        "cuda_available": None,
        "device_name": None,
        "device_capability": None,
        "nvcc_version": None,
        "maxnreg_supported": None,
        "probe_errors": {},
    }
    try:
        import triton  # type: ignore

        snapshot["triton_version"] = getattr(triton, "__version__", None)
    except Exception as error:  # pragma: no cover - triton is a soft dependency
        snapshot["probe_errors"]["triton"] = str(error)
    try:
        import torch  # type: ignore

        snapshot["torch_version"] = getattr(torch, "__version__", None)
        available = bool(torch.cuda.is_available())
        snapshot["cuda_available"] = available
        if available:  # pragma: no cover - needs a GPU host
            try:
                snapshot["device_name"] = torch.cuda.get_device_name(0)
                snapshot["device_capability"] = ".".join(
                    str(part) for part in torch.cuda.get_device_capability(0)
                )
            except Exception as error:
                snapshot["probe_errors"]["device"] = str(error)
    except Exception as error:  # pragma: no cover - torch is present in CI
        snapshot["probe_errors"]["torch"] = str(error)
    try:
        snapshot["maxnreg_supported"] = maxnreg_supported()
    except Exception as error:  # pragma: no cover - probe is internal
        snapshot["probe_errors"]["maxnreg"] = str(error)
    try:
        out = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=5, check=False
        )
        if out.returncode == 0:
            for line in out.stdout.splitlines():
                if "release" in line:
                    snapshot["nvcc_version"] = line.split("release")[-1].strip(" ,")[:16]
                    break
    except Exception:
        pass  # nvcc absent (CPU build node) -> recorded as None
    return snapshot


# ---------------------------------------------------------------------------
# pinned benchmark recorder
# ---------------------------------------------------------------------------


class PinnedRecorder:
    """Append-only JSONL log that only accepts pinned records.

    Rationale (the whole point of pinning): a baseline file must never be
    polluted by a run whose register budget differs from the declared pin.
    """

    def __init__(self, path: str | Path, *, require_pin: bool = True) -> None:
        self.path = Path(path)
        self.require_pin = require_pin

    def record(
        self,
        resolved: ResolvedLaunch,
        *,
        metric: str,
        value: float,
        units: str = "MLUPS",
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.require_pin and not resolved.pinned:
            raise CodegenPinViolation(
                f"refusing to record an unpinned measurement for {resolved.kernel!r} "
                f"(fingerprint={resolved.fingerprint}); pin the codegen knobs first"
            )
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "kernel": resolved.kernel,
            "knobs": dict(resolved.knobs),
            "pinned": resolved.pinned,
            "fingerprint": resolved.fingerprint,
            "metric": metric,
            "value": float(value),
            "units": units,
            "extra": dict(extra or {}),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


# ---------------------------------------------------------------------------
# static audit (AST, no GPU required)
# ---------------------------------------------------------------------------

_DEFAULT_AUDIT_MODULES: tuple[str, ...] = (
    "src/tensorlbm/triton_fused.py",
    "src/tensorlbm/triton_fused_obstacle.py",
    "src/tensorlbm/triton_fused_distributed.py",
    "src/tensorlbm/_voxel_kernels.py",
    "src/tensorlbm/backends/triton_backend.py",
)


def _repo_root(paths: Sequence[str]) -> Path:
    return Path(paths[0]).resolve().parents[1] if paths else Path.cwd()


def _module_dotted(rel_path: str) -> str:
    stem = rel_path.replace("\\", "/")
    if stem.endswith(".py"):
        stem = stem[:-3]
    stem = stem.split("src/", 1)[-1]
    return stem.replace("/", ".")


def _resolve_symbol(module: str, name: str, cache: dict[str, Any]) -> Any:
    """Resolve a module-level constant such as ``DEFAULT_NUM_WARPS``."""
    if module not in cache:
        try:
            cache[module] = __import__(module, fromlist=["_"])
        except Exception as error:  # pragma: no cover - import chain may need torch
            cache[module] = error
    mod = cache[module]
    if isinstance(mod, Exception):
        return None
    return getattr(mod, name, None)


def _launch_symbol(name: str, module: str, cache: dict[str, Any]) -> Any:
    """Resolve a module constant, else keep the source expression name."""
    value = _resolve_symbol(module, name, cache)
    return value if value is not None else f"<{name}>"


def _dict_pairs(node: ast.AST) -> list[tuple[str | None, ast.AST]] | None:
    """Flatten a ``dict(...)``/``{...}`` expression into (key, value) pairs.

    ``key is None`` marks a ``**spread`` entry (dict-unpacking or ``**kwargs``).
    Returns ``None`` for anything that is not a recognisable dict expression.
    """
    pairs: list[tuple[str | None, ast.AST]] = []
    if isinstance(node, ast.Dict):
        for key_node, value_node in zip(node.keys, node.values):
            key = key_node.value if isinstance(key_node, ast.Constant) else None
            pairs.append((key, value_node))
        return pairs
    if isinstance(node, ast.Call) and ast.unparse(node.func) in ("dict", "builtins.dict"):
        for keyword in node.keywords:
            pairs.append((keyword.arg, keyword.value))
        return pairs
    return None


def _literal(node: ast.AST | None) -> tuple[Any, str]:
    """(value, kind) for a default expression: ('literal'|'symbol'|'absent')."""
    if node is None:
        return None, "absent"
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value, "literal"
    if isinstance(node, ast.Name):
        return node.id, "symbol"
    return None, "expr"


def audit_codegen(
    paths: Iterable[str] | None = None,
    *,
    pins: Mapping[str, KernelPin] | None = None,
) -> dict[str, Any]:
    """Statically audit Triton kernel launch codegen knobs in the tree.

    Returns a machine-readable report with three sections:

    ``kernels``
        every ``@triton.jit`` function found (module, line).
    ``launch_sites``
        every ``kernel[grid](...)`` call with the codegen kwargs it passes,
        each resolved to a literal value where possible (symbolic defaults such
        as ``DEFAULT_NUM_WARPS`` are imported and resolved, or reported as
        unresolvable rather than guessed).
    ``drift``
        declared pin vs audited default mismatches for the benchmarked
        wrappers, i.e. "did somebody change a codegen knob without saying so".

    Deliberately static/AST-based: it must run on CPU-only nodes where no
    kernel can be launched.
    """
    rel_paths = list(paths) if paths is not None else list(_DEFAULT_AUDIT_MODULES)
    pins = dict(DECLARED_PINS if pins is None else pins)
    import_cache: dict[str, Any] = {}
    kernels: list[dict[str, Any]] = []
    launch_sites: list[dict[str, Any]] = []
    wrappers: list[dict[str, Any]] = []

    for rel in rel_paths:
        path = Path(rel)
        if not path.exists():
            continue
        module = _module_dotted(rel)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        # Pre-pass: dict variables (e.g. ``common = dict(..., **launch_kwargs(...))``)
        # that carry codegen kwargs.  Launch sites spread those as ``**common``,
        # so the site itself shows no explicit codegen keyword.
        dict_aliases: dict[str, dict[str, Any]] = {}
        for assign in ast.walk(tree):
            if not isinstance(assign, (ast.Assign, ast.AnnAssign)):
                continue
            targets = assign.targets if isinstance(assign, ast.Assign) else [assign.target]
            value = assign.value
            if not isinstance(value, (ast.Dict, ast.Call)):
                continue
            name = next((t.id for t in targets if isinstance(t, ast.Name)), None)
            if name is None:
                continue
            pairs = _dict_pairs(value)
            if pairs is None:
                continue
            alias: dict[str, Any] = {
                "pinned_via": None,
                "codegen": {},
                "value_kinds": {},
            }
            for key, node_value in pairs:
                if key in CODEGEN_KEYS:
                    literal, kind = _literal(node_value)
                    if kind == "symbol":
                        literal = _launch_symbol(literal, module, import_cache)
                    alias["codegen"][key] = literal
                    alias["value_kinds"][key] = kind
                elif key is None and isinstance(node_value, ast.Call):
                    if ast.unparse(node_value.func).endswith("launch_kwargs"):
                        alias["pinned_via"] = "codegen_pin"
                        for inner in node_value.keywords:
                            if inner.arg not in CODEGEN_KEYS:
                                continue
                            literal, kind = _literal(inner.value)
                            if kind == "symbol":
                                literal = _launch_symbol(literal, module, import_cache)
                            alias["codegen"][inner.arg] = literal
                            alias["value_kinds"][inner.arg] = kind
            if alias["pinned_via"] or alias["codegen"]:
                dict_aliases[name] = alias
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                decorators = [ast.unparse(d) for d in node.decorator_list]
                if any("triton.jit" in text or text == "jit" for text in decorators):
                    kernels.append({"name": node.name, "module": module, "line": node.lineno})
                args = list(node.args.args) + list(node.args.kwonlyargs)
                defaults = list(node.args.defaults) + [
                    default for default in node.args.kw_defaults if default is not None
                ]
                codegen_defaults: dict[str, Any] = {}
                codegen_kinds: dict[str, str] = {}
                for arg, default in zip(reversed(args), reversed(defaults)):
                    if arg.arg in CODEGEN_KEYS:
                        value, kind = _literal(default)
                        if kind == "symbol":
                            value = _resolve_symbol(module, value, import_cache)
                        codegen_defaults[arg.arg] = value
                        codegen_kinds[arg.arg] = kind
                # A wrapper whose every codegen default is None carries no
                # declared configuration (it merely forwards); skip the noise.
                if codegen_defaults and any(
                    value is not None for value in codegen_defaults.values()
                ):
                    wrappers.append(
                        {
                            "wrapper": f"{module}:{node.name}",
                            "line": node.lineno,
                            "codegen": codegen_defaults,
                            "default_kinds": codegen_kinds,
                        }
                    )
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Subscript):
                continue
            target = func.value
            kernel_name = (
                target.id
                if isinstance(target, ast.Name)
                else target.attr
                if isinstance(target, ast.Attribute)
                else None
            )
            if kernel_name is None:
                continue
            codegen: dict[str, Any] = {}
            kinds: dict[str, str] = {}
            forwarding: list[str] = []
            pinned_via: str | None = None
            for keyword in node.keywords:
                if keyword.arg is None:  # **kwargs spread
                    spread = keyword.value
                    forwarding.append(ast.unparse(spread))
                    if isinstance(spread, ast.Call):
                        fname = ast.unparse(spread.func)
                        if fname.endswith("launch_kwargs"):
                            pinned_via = "codegen_pin"
                            for inner in spread.keywords:
                                if inner.arg not in CODEGEN_KEYS:
                                    continue
                                value, kind = _literal(inner.value)
                                if kind == "symbol":
                                    value = _launch_symbol(value, module, import_cache)
                                if value is None and kind == "expr":
                                    value = ast.unparse(inner.value)
                                codegen[inner.arg] = value
                                kinds[inner.arg] = kind
                    elif isinstance(spread, ast.Name) and spread.id in dict_aliases:
                        alias = dict_aliases[spread.id]
                        pinned_via = alias["pinned_via"]
                        for key, value in alias["codegen"].items():
                            codegen[key] = value
                            kinds[key] = alias["value_kinds"][key]
                    continue
                if keyword.arg not in CODEGEN_KEYS:
                    continue
                value, kind = _literal(keyword.value)
                if kind == "symbol":
                    value = _launch_symbol(value, module, import_cache)
                if value is None and kind == "expr":
                    value = ast.unparse(keyword.value)
                codegen[keyword.arg] = value
                kinds[keyword.arg] = kind
            launch_sites.append(
                {
                    "module": module,
                    "line": node.lineno,
                    "kernel": kernel_name,
                    "codegen": codegen,
                    "value_kinds": kinds,
                    "kwargs_spread": forwarding,
                    "pinned_via": pinned_via,
                }
            )

    sites_by_kernel: dict[str, list[dict[str, Any]]] = {}
    for site in launch_sites:
        sites_by_kernel.setdefault(site["kernel"], []).append(site)

    drift: list[dict[str, Any]] = []
    for name, pin in sorted(pins.items()):
        wrapper = next(
            (w for w in wrappers if w["wrapper"] == pin.source),
            None,
        )
        if wrapper is None:
            drift.append(
                {
                    "kernel": name,
                    "source": pin.source,
                    "issue": "declared_wrapper_not_found",
                }
            )
            continue
        for key, declared in pin.knobs().items():
            if declared is None and key == "maxnreg":
                if wrapper["codegen"].get("maxnreg") is not None:
                    drift.append(
                        {
                            "kernel": name,
                            "source": pin.source,
                            "knob": key,
                            "declared": None,
                            "audited": wrapper["codegen"].get("maxnreg"),
                            "issue": "maxnreg_added_without_pin_update",
                        }
                    )
                continue
            audited = wrapper["codegen"].get(key)
            if audited is None:
                if key == "num_stages":
                    continue  # left to the backend default; not a drift
                drift.append(
                    {
                        "kernel": name,
                        "source": pin.source,
                        "knob": key,
                        "declared": declared,
                        "audited": None,
                        "issue": "knob_missing_in_wrapper",
                    }
                )
                continue
            if audited != declared:
                drift.append(
                    {
                        "kernel": name,
                        "source": pin.source,
                        "knob": key,
                        "declared": declared,
                        "audited": audited,
                        "issue": "declared_value_differs",
                    }
                )

    return {
        "pin_version": PIN_VERSION,
        "modules": rel_paths,
        "kernels": sorted(kernels, key=lambda k: (k["module"], k["line"])),
        "launch_sites": sorted(launch_sites, key=lambda s: (s["module"], s["line"])),
        "wrappers": sorted(wrappers, key=lambda w: w["wrapper"]),
        "declared_pins": {name: pin.to_dict() for name, pin in sorted(pins.items())},
        "drift": drift,
        "summary": {
            "kernels": len(kernels),
            "launch_sites": len(launch_sites),
            # Sites that forward a register cap *argument* (the value may be a
            # caller-supplied symbol).  Contrast with
            # ``declared_pins_with_maxnreg``: how many pins actually fix a value.
            "launch_sites_forwarding_maxnreg": sum(
                1 for site in launch_sites if site["codegen"].get("maxnreg") is not None
            ),
            "declared_pins_with_maxnreg": sum(
                1 for pin in pins.values() if pin.maxnreg is not None
            ),
            "launch_sites_with_symbolic_warps": sum(
                1 for site in launch_sites if site["value_kinds"].get("num_warps") == "symbol"
            ),
            "launch_sites_pinned_via_codegen_pin": sum(
                1 for site in launch_sites if site["pinned_via"] == "codegen_pin"
            ),
            "launch_sites_with_literal_warps": sum(
                1 for site in launch_sites if site["value_kinds"].get("num_warps") == "literal"
            ),
            "wrappers_exposing_codegen": len(wrappers),
            "distinct_pin_fingerprints": len({pin.fingerprint() for pin in pins.values()}),
            "drift_findings": len(drift),
            "maxnreg_supported": maxnreg_supported(),
        },
        "environment": environment_snapshot(),
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI
    argv = list(sys.argv[1:] if argv is None else argv)
    paths = argv or None
    report = audit_codegen(paths)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0 if not report["drift"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
