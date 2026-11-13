"""Baseline-reference harness — port of ``pylbm/pylbm`` pytest practice.

What is ported (and from where)
-------------------------------
The upstream project ``pylbm`` keeps one pytest plugin inside
``tests/conftest.py`` (lines are for the checkout in ``/root/oss-cache/pylbm``):

* ``generate_test_name`` (conftest.py:318-335) — a *deterministic*
  test-id -> artifact-name mapping, so a baseline file can be found without a
  registry: brackets become ``_`` and a trailing ``_.h5`` collapses to ``.h5``.
* ``H5Comparison.pytest_runtest_call`` (conftest.py:337-410) — the test
  function *returns* the simulation object, the plugin serialises the
  monitored conserved moments to a temporary HDF5 file, then compares every
  dataset against ``tests/reference/<name>.h5`` with
  ``pytest.approx(rel=rtol, abs=atol)``; defaults ``atol=1e-7, rtol=1e-14``
  (conftest.py:345-346).  A missing baseline is a hard error with the message
  "File not found for comparison test ... This is expected for new tests",
  never a silent skip.
* ``--h5diff-generate-path`` (conftest.py:191-198, 412-418) — a *generate
  mode* that writes the reference file and then ``pytest.skip``s the test.
  ``--h5diff-reference-path`` overrides the reference directory (may also be
  an http URL upstream).
* ``tests/baseline/*.png`` with ``@pytest.mark.mpl_image_compare`` is the
  sibling *image* baseline mechanism (``tests/test_domain.py:132``).

Deliberate differences (documented in the XVAL report)
-----------------------------------------------------
* **NPZ instead of HDF5.** ``h5py`` is an optional extra of this repository
  (``pyproject.toml`` ``[project.optional-dependencies].io``) while baseline
  checks must run on the default install; ``numpy`` is a hard dependency.  The
  *semantics* (named datasets + atol/rtol comparison + generate mode) are kept
  identical, and the on-disk block is readable by ``numpy`` alone.
* **Per-field diagnostics.** Upstream returns only pytest's failure message;
  here ``compare`` reports ``max_abs``/``max_rel``/``n_exceeded`` per field so
  a report can quote "ref vs got" numbers directly.
* **Provenance.** Upstream stores arrays only.  This port stamps a sha256 of
  the array payload plus the environment (torch/numpy versions, dtype, device)
  into the block metadata, so a baseline cannot be silently regenerated on a
  different stack and still look "the same".

Usage
-----
::

    # entry point (CLI)
    python -m tensorlbm.verification.baseline_reference list
    python -m tensorlbm.verification.baseline_reference check  <name> <candidate.npz>
    python -m tensorlbm.verification.baseline_reference register <name> <candidate.npz>

Environment variables (the pytest-facing equivalents of the upstream flags):

``TLBM_BASELINE_REFERENCE_DIR``
    Reference directory (default: ``tests/reference`` next to the checkout).
``TLBM_BASELINE_GENERATE``
    Generate mode: write baselines into that directory, do not compare.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tensorlbm.baseline_io import (
    as_field_dict,
    payload_digest,
    read_block,
    write_block,
)

#: Upstream defaults, `tests/conftest.py:345-346` (``pytest.approx`` kwargs).
DEFAULT_ATOL = 1e-7
DEFAULT_RTOL = 1e-14

ENV_REFERENCE_DIR = "TLBM_BASELINE_REFERENCE_DIR"
ENV_GENERATE = "TLBM_BASELINE_GENERATE"


class ReferenceNotFound(RuntimeError):
    """Raised when a comparison test has no baseline yet.

    Mirrors upstream ``conftest.py:383-391``: a missing baseline is an explicit
    error whose text says it is *expected for new tests* (generate one), not a
    silent pass or skip.
    """


@dataclasses.dataclass(frozen=True)
class FieldDiff:
    """Elementwise agreement of one array field."""

    name: str
    shape: tuple[int, ...]
    max_abs: float
    max_rel: float
    worst_index: tuple[int, ...]
    n_exceeded: int
    size: int

    @property
    def ok(self) -> bool:
        return self.n_exceeded == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "max_abs": self.max_abs,
            "max_rel": self.max_rel,
            "worst_index": list(self.worst_index),
            "n_exceeded": self.n_exceeded,
            "size": self.size,
            "ok": self.ok,
        }


@dataclasses.dataclass(frozen=True)
class CompareResult:
    """Outcome of one baseline comparison (kept JSON-serialisable)."""

    name: str
    ok: bool
    atol: float
    rtol: float
    fields: dict[str, FieldDiff]
    reference_path: str
    message: str = ""
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "atol": self.atol,
            "rtol": self.rtol,
            "reference_path": self.reference_path,
            "message": self.message,
            "fields": {k: v.as_dict() for k, v in self.fields.items()},
            "meta": self.meta,
        }


# ---------------------------------------------------------------------------
# Naming (port of conftest.py:318-335)
# ---------------------------------------------------------------------------


def baseline_name(
    test_id: str,
    *,
    single_reference: bool = False,
    filename: str | None = None,
    ext: str = ".npz",
) -> str:
    """Map a pytest node id to a baseline file name.

    Direct port of upstream ``H5Comparison.generate_test_name``: an explicit
    ``filename`` wins; with ``single_reference`` the *function* name is used
    (one baseline per test function, shared by all parametrisations); otherwise
    the full node id is used with ``[``/``]`` replaced by ``_`` and the trailing
    ``_.<ext>`` collapsed (upstream collapses ``_.h5``).

    >>> baseline_name("tests/test_x.py::test_cavity[33-100]")
    'test_cavity_33-100.npz'
    >>> baseline_name("tests/test_x.py::test_cavity[33-100]", single_reference=True)
    'test_cavity.npz'
    """
    if filename is not None:
        return filename
    if single_reference:
        function = test_id.split("::")[-1].split("[")[0]
        return f"{function}{ext}"
    name = test_id.split("::")[-1] if "::" in test_id else test_id
    name = name.replace("[", "_").replace("]", "_")
    name = f"{name}{ext}" if not name.endswith(ext) else name
    return name.replace(f"_{ext}", ext)


def resolve_reference_dir(reference_dir: str | Path | None = None) -> Path:
    """Reference directory, honouring ``TLBM_BASELINE_REFERENCE_DIR``.

    Default is ``tests/reference`` at the checkout root (upstream keeps its
    baselines in ``tests/reference`` relative to the test file); when the
    checkout layout is not present the current working directory is used.
    """
    if reference_dir is not None:
        return Path(reference_dir)
    env = os.environ.get(ENV_REFERENCE_DIR)
    if env:
        return Path(env)
    root = Path(__file__).resolve().parents[3]
    candidate = root / "tests" / "reference"
    if candidate.is_dir():
        return candidate
    return Path("tests/reference")


def generate_mode_dir() -> Path | None:
    """Directory requested by ``TLBM_BASELINE_GENERATE`` (None if inactive)."""
    env = os.environ.get(ENV_GENERATE)
    if not env:
        return None
    return Path(env)


# ---------------------------------------------------------------------------
# Block IO
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Comparison (port of conftest.py:396-405, pytest.approx semantics)
# ---------------------------------------------------------------------------


def compare_fields(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> dict[str, FieldDiff]:
    """Elementwise ``pytest.approx`` agreement per field.

    Upstream compares with ``pytest.approx(actual, rel=rtol, abs=atol)``, whose
    acceptance rule is ``abs(actual - expected) <= max(atol, rtol*abs(expected))``;
    that rule is implemented here verbatim, over all fields of the baseline.
    """
    missing = sorted(set(reference) - set(candidate))
    extra = sorted(set(candidate) - set(reference))
    if missing or extra:
        msg = f"field set mismatch: missing={missing} extra={extra}"
        raise ValueError(msg)

    diffs: dict[str, FieldDiff] = {}
    for name, ref in reference.items():
        ref_arr = np.asarray(ref, dtype=np.float64)
        got_arr = np.asarray(candidate[name], dtype=np.float64)
        if ref_arr.shape != got_arr.shape:
            msg = f"field {name!r}: reference shape {ref_arr.shape} != candidate {got_arr.shape}"
            raise ValueError(msg)
        delta = np.abs(got_arr - ref_arr)
        tol = np.maximum(atol, rtol * np.abs(ref_arr))
        finite_ref = np.isfinite(ref_arr)
        exceed = (delta > tol) | ~finite_ref & np.isfinite(got_arr) | np.isfinite(ref_arr) & ~np.isfinite(got_arr)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(np.abs(ref_arr) > 0, delta / np.abs(ref_arr), np.inf)
        flat = int(np.argmax(delta)) if delta.size else 0
        worst = tuple(int(i) for i in np.unravel_index(flat, ref_arr.shape)) if delta.size else ()
        diffs[name] = FieldDiff(
            name=name,
            shape=tuple(int(s) for s in ref_arr.shape),
            max_abs=float(delta.max()) if delta.size else 0.0,
            max_rel=float(np.nanmax(rel[np.isfinite(rel)])) if np.isfinite(rel).any() else float("inf"),
            worst_index=worst,
            n_exceeded=int(np.count_nonzero(exceed)),
            size=int(ref_arr.size),
        )
    return diffs


class BaselineStore:
    """A directory of baseline blocks, with pylbm's generate-mode switch."""

    def __init__(
        self,
        reference_dir: str | Path | None = None,
        *,
        generate_dir: str | Path | None = None,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
    ) -> None:
        self.reference_dir = resolve_reference_dir(reference_dir)
        self.generate_dir = Path(generate_dir) if generate_dir is not None else generate_mode_dir()
        self.atol = float(atol)
        self.rtol = float(rtol)

    # -- introspection ---------------------------------------------------
    @property
    def generating(self) -> bool:
        return self.generate_dir is not None

    def path_for(self, name: str) -> Path:
        return self.reference_dir / name

    def list(self) -> list[str]:
        if not self.reference_dir.is_dir():
            return []
        return sorted(p.name for p in self.reference_dir.glob("*.npz"))

    # -- writing ---------------------------------------------------------
    def write(
        self,
        name: str,
        fields: Mapping[str, Any],
        meta: Mapping[str, Any] | None = None,
    ) -> Path:
        target_dir = self.generate_dir if self.generating else self.reference_dir
        return write_block(Path(target_dir) / name, fields, meta)

    def register(self, name: str, block_path: str | Path) -> Path:
        """Copy an existing block file into the reference directory."""
        fields, meta = read_block(block_path)
        meta = dict(meta)
        meta["registered_from"] = str(Path(block_path))
        return write_block(self.path_for(name), fields, {k: v for k, v in meta.items() if k != "sha256"})

    # -- comparing -------------------------------------------------------
    def compare(
        self,
        name: str,
        candidate: str | Path | Mapping[str, Any],
        *,
        atol: float | None = None,
        rtol: float | None = None,
    ) -> CompareResult:
        atol = self.atol if atol is None else float(atol)
        rtol = self.rtol if rtol is None else float(rtol)
        ref_path = self.path_for(name)
        if not ref_path.exists():
            msg = (
                f"File not found for comparison test:\n\t{ref_path}\n"
                "This is expected for new tests (generate it with "
                f"{ENV_GENERATE}=<dir> or `register`)."
            )
            raise ReferenceNotFound(msg)
        ref_fields, ref_meta = read_block(ref_path)
        if isinstance(candidate, (str, Path)):
            got_fields, _ = read_block(candidate)
        else:
            got_fields = as_field_dict(candidate)

        diffs = compare_fields(ref_fields, got_fields, atol=atol, rtol=rtol)
        ok = all(d.ok for d in diffs.values())
        if ok:
            message = f"{name}: {len(diffs)} field(s) within atol={atol:g}, rtol={rtol:g}"
        else:
            bad = ", ".join(
                f"{d.name}: {d.n_exceeded}/{d.size} exceeded (max_abs={d.max_abs:.3e})"
                for d in diffs.values()
                if not d.ok
            )
            message = f"{name}: FAIL -> {bad}"
        # Upstream stores arrays only; the digest check is this port's addition.
        digest_ok = ref_meta.get("sha256") in (None, payload_digest(ref_fields))
        if not digest_ok:
            ok = False
            message = f"{name}: reference payload digest mismatch (file edited by hand?)"
        return CompareResult(
            name=name,
            ok=ok,
            atol=atol,
            rtol=rtol,
            fields=diffs,
            reference_path=str(ref_path),
            message=message,
            meta=ref_meta,
        )


# ---------------------------------------------------------------------------
# CLI (the entry point referenced by the XVAL report)
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tensorlbm.verification.baseline_reference",
        description="Manage and check numerical baselines (pylbm-style pytest reference practice).",
    )
    parser.add_argument("--reference-dir", default=None, help=f"override {ENV_REFERENCE_DIR}")
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list baselines in the reference directory")
    check = sub.add_parser("check", help="compare a candidate block against a baseline")
    check.add_argument("name")
    check.add_argument("candidate")
    check.add_argument("--json-out", default=None, help="write the CompareResult JSON here")
    reg = sub.add_parser("register", help="register a candidate block as the baseline")
    reg.add_argument("name")
    reg.add_argument("candidate")

    args = parser.parse_args(argv)
    store = BaselineStore(args.reference_dir, atol=args.atol, rtol=args.rtol)

    if args.cmd == "list":
        for name in store.list():
            print(name)
        return 0

    if args.cmd == "register":
        path = store.register(args.name, args.candidate)
        print(f"registered {args.name} -> {path}")
        return 0

    result = store.compare(args.name, args.candidate)
    payload = result.as_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(_cli())