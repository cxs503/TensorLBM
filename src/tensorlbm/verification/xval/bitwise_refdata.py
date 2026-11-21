"""Bitwise reproducibility and reference-data regression for LBM solvers.

Key idea: run the solver twice with identical seeds and configuration; hash the
final distribution-function array.  If the hashes match, the solver is
bitwise reproducible on this platform.  Reference data (scalar metrics + field
snapshots) are stored as compressed .npz files; regression tests compare new
runs against the stored golden data.

Bitwise reference-data regression (joconnor22/LIFE pattern)
-----------------------------------------------------------
A *bitwise* regression test does not stop at a relative-error threshold.  It
pins the exact byte-level output of a verified run: the SHA-256 of every stored
array is recorded in the manifest, and a re-run is accepted only if its arrays
reproduce those digests *exactly* (``np.array_equal`` / hash match).  This
catches any change that would still pass a ``<15 %`` tolerance test but
represents a real behavioural shift -- e.g. a reordering of the collision/
streaming kernels, a fused-kernel rewrite, a dtype or byte-order change, or a
different reduction order in the macroscopic moment computation.

References
----------
* Krüger, T. et al. (2017). *The Lattice Boltzmann Method*. Springer, §4.4.
* Latt, J. (2007). "Choice of units in lattice Boltzmann simulations".
  http://lbmethod.org/numerics:units
* joconnor22/LIFE: bitwise reference-data regression + benchmark cases
  (golden-snapshot pinning of solver output).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Bitwise reproducibility
# ---------------------------------------------------------------------------

@dataclass
class BitwiseReproConfig:
    """Configuration for bitwise reproducibility testing.

    Parameters
    ----------
    seed : int
        Random seed for initialisation (must be set *before* any RNG call).
    dtype : str
        Floating-point dtype: 'float32' or 'float64'.
    device : str
        Device string: 'cpu' or 'cuda'.
    n_runs : int
        Number of identical runs to compare (minimum 2).
    """
    seed: int = 42
    dtype: str = "float64"
    device: str = "cpu"
    n_runs: int = 2


def compute_state_hash(arr: np.ndarray) -> str:
    """Compute SHA-256 hash of a numpy array's raw bytes.

    The hash is computed over the array's contiguous byte representation,
    which is deterministic for a given dtype, shape, and byte order.

    Parameters
    ----------
    arr : np.ndarray
        Array to hash.  Will be made contiguous if not already.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest.
    """
    arr = np.ascontiguousarray(arr)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def verify_bitwise_reproducibility(
    run_fn,
    config: BitwiseReproConfig,
    **run_kwargs,
) -> Dict[str, Any]:
    """Verify that ``run_fn`` produces bitwise-identical output across runs.

    Parameters
    ----------
    run_fn : callable
        A function that accepts ``(config, **kwargs)`` and returns a dict
        with at least a ``'f'`` key containing the final distribution array.
    config : BitwiseReproConfig
        Reproducibility configuration.
    **run_kwargs
        Additional keyword arguments forwarded to ``run_fn``.

    Returns
    -------
    dict
        Keys:
        - ``'reproducible'``: bool, True if all runs produced identical hashes.
        - ``'hashes'``: list of SHA-256 hex digests, one per run.
        - ``'config'``: the BitwiseReproConfig used.
    """
    hashes = []
    results = []
    for _ in range(config.n_runs):
        result = run_fn(config, **run_kwargs)
        f_arr = result.get("f")
        if f_arr is None:
            raise ValueError("run_fn must return a dict with key 'f' (distribution array)")
        if isinstance(f_arr, np.ndarray):
            arr = f_arr
        else:
            # Assume torch tensor
            arr = f_arr.detach().cpu().numpy()
        h = compute_state_hash(arr)
        hashes.append(h)
        results.append(result)

    reproducible = all(h == hashes[0] for h in hashes[1:])
    return {
        "reproducible": reproducible,
        "hashes": hashes,
        "config": asdict(config),
    }


# ---------------------------------------------------------------------------
# Reference-data store (golden snapshots)
# ---------------------------------------------------------------------------

_REFDATA_DIR = Path(__file__).parent / "refdata"


@dataclass
class ReferenceDataStore:
    """Store and retrieve golden reference data for regression testing.

    Data are stored as compressed .npz files in the ``refdata/`` directory
    adjacent to this module.  Each case has a unique slug (e.g.
    ``poiseuille_2d_H16``) and an optional version tag.

    The manifest (``refdata/manifest.json``) records metadata: case slug,
    version, creation timestamp, SHA-256 of *every* stored array (for bitwise
    regression), and the acceptance thresholds.
    """
    base_dir: Path = field(default_factory=lambda: _REFDATA_DIR)

    def _npz_path(self, slug: str, version: str = "v1") -> Path:
        return self.base_dir / f"{slug}_{version}.npz"

    def _manifest_path(self) -> Path:
        return self.base_dir / "manifest.json"

    def _load_manifest(self) -> Dict:
        p = self._manifest_path()
        if p.exists():
            return json.loads(p.read_text())
        return {"cases": {}}

    def _save_manifest(self, manifest: Dict) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path().write_text(json.dumps(manifest, indent=2))

    def save(
        self,
        slug: str,
        arrays: Dict[str, np.ndarray],
        scalars: Dict[str, float],
        thresholds: Dict[str, float],
        version: str = "v1",
    ) -> Path:
        """Save reference data for a case.

        Parameters
        ----------
        slug : str
            Case identifier (e.g. ``poiseuille_2d_H16``).
        arrays : dict
            Named numpy arrays to store (e.g. velocity profiles).
        scalars : dict
            Named scalar metrics (e.g. L2 error, max error).
        thresholds : dict
            Named acceptance thresholds (e.g. ``{'eps_inf': 0.15}``).
        version : str
            Version tag.

        Returns
        -------
        Path
            Path to the saved .npz file.
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)
        npz_path = self._npz_path(slug, version)

        # Store arrays + scalars (scalars as 0-d arrays)
        save_dict = {}
        for k, v in arrays.items():
            save_dict[f"arr_{k}"] = np.asarray(v)
        for k, v in scalars.items():
            save_dict[f"scl_{k}"] = np.asarray(v)
        for k, v in thresholds.items():
            save_dict[f"thr_{k}"] = np.asarray(v)

        np.savez_compressed(npz_path, **save_dict)

        # Update manifest: record the SHA-256 of *every* stored array so a
        # re-run can be pinned bitwise (not just within a tolerance band).
        manifest = self._load_manifest()
        loaded = np.load(npz_path, allow_pickle=False)
        array_hashes = {}
        for k in arrays.keys():
            array_hashes[k] = compute_state_hash(loaded[f"arr_{k}"])
        first_key = list(arrays.keys())[0]
        manifest["cases"][slug] = {
            "version": version,
            "npz_file": npz_path.name,
            "sha256_first_array": array_hashes[first_key],
            "sha256_arrays": array_hashes,
            "scalars": scalars,
            "thresholds": thresholds,
        }
        self._save_manifest(manifest)
        return npz_path

    def load(
        self,
        slug: str,
        version: str = "v1",
    ) -> Dict[str, Any]:
        """Load reference data for a case.

        Returns
        -------
        dict
            Keys: ``'arrays'``, ``'scalars'``, ``'thresholds'``.
        """
        npz_path = self._npz_path(slug, version)
        if not npz_path.exists():
            raise FileNotFoundError(f"No reference data for {slug} v{version} at {npz_path}")

        data = np.load(npz_path, allow_pickle=False)
        arrays = {}
        scalars = {}
        thresholds = {}
        for k in data.files:
            if k.startswith("arr_"):
                arrays[k[4:]] = data[k]
            elif k.startswith("scl_"):
                scalars[k[4:]] = float(data[k])
            elif k.startswith("thr_"):
                thresholds[k[4:]] = float(data[k])

        return {"arrays": arrays, "scalars": scalars, "thresholds": thresholds}

    def compare(
        self,
        slug: str,
        new_arrays: Dict[str, np.ndarray],
        new_scalars: Dict[str, float],
        version: str = "v1",
    ) -> Dict[str, Any]:
        """Compare new results against stored reference data.

        Returns
        -------
        dict
            Keys:
            - ``'scalar_errors'``: dict of relative errors for each scalar.
            - ``'array_errors'``: dict of L2/Linf errors for each array.
            - ``'passed'``: bool, True if all scalar errors are within thresholds.
            - ``'details'``: per-metric pass/fail.
        """
        ref = self.load(slug, version)
        ref_scalars = ref["scalars"]
        ref_thresholds = ref["thresholds"]
        ref_arrays = ref["arrays"]

        scalar_errors = {}
        array_errors = {}
        details = {}

        # Compare scalars
        for k, new_val in new_scalars.items():
            if k in ref_scalars:
                ref_val = ref_scalars[k]
                if abs(ref_val) > 1e-15:
                    rel_err = abs(new_val - ref_val) / abs(ref_val)
                else:
                    rel_err = abs(new_val - ref_val)
                scalar_errors[k] = rel_err
                threshold = ref_thresholds.get(k, 0.15)  # default 15%
                details[k] = {
                    "ref": ref_val,
                    "new": new_val,
                    "rel_err": rel_err,
                    "threshold": threshold,
                    "passed": rel_err <= threshold,
                }

        # Compare arrays
        for k, new_arr in new_arrays.items():
            if k in ref_arrays:
                ref_arr = ref_arrays[k]
                diff = np.asarray(new_arr) - ref_arr
                l2 = float(np.sqrt(np.mean(diff**2)))
                linf = float(np.max(np.abs(diff)))
                scale = float(np.max(np.abs(ref_arr))) if np.max(np.abs(ref_arr)) > 1e-15 else 1.0
                array_errors[k] = {
                    "l2": l2,
                    "l2_rel": l2 / scale,
                    "linf": linf,
                    "linf_rel": linf / scale,
                }

        passed = all(d["passed"] for d in details.values())

        return {
            "scalar_errors": scalar_errors,
            "array_errors": array_errors,
            "passed": passed,
            "details": details,
        }

    def bitwise_compare(
        self,
        slug: str,
        new_arrays: Dict[str, np.ndarray],
        version: str = "v1",
    ) -> Dict[str, Any]:
        """Bitwise reference-data regression against the stored golden snapshot.

        Unlike :meth:`compare` (which accepts results within a relative-error
        tolerance band), this method pins the *exact* byte-level output of the
        solver.  Every array in ``new_arrays`` must reproduce the SHA-256 digest
        recorded in the manifest for the same key -- i.e. ``np.array_equal``
        with the stored golden array -- or the regression is reported as failed.

        This is the joconnor22/LIFE "bitwise refdata regression" pattern: it
        catches a reordering of the collision/streaming kernels, a fused-kernel
        rewrite, a dtype/byte-order change, or a different reduction order in
        the macroscopic moment computation -- all of which would still pass a
        ``<15 %`` tolerance test but represent a real behavioural shift.

        Parameters
        ----------
        slug : str
            Case identifier (e.g. ``poiseuille_2d_H16``).
        new_arrays : dict
            Named numpy arrays produced by a fresh run of the solver.
        version : str
            Version tag.

        Returns
        -------
        dict
            Keys:
            - ``'bitwise_passed'``: bool, True only if *every* array matches
              its stored golden digest exactly.
            - ``'array_hashes'``: dict of SHA-256 digests of ``new_arrays``.
            - ``'golden_hashes'``: dict of SHA-256 digests recorded in the
              manifest for the stored golden snapshot.
            - ``'details'``: per-array ``{'match': bool, 'golden': str,
              'new': str}``.
            - ``'n_arrays'`` / ``'n_matched'``: counts.
        """
        ref = self.load(slug, version)
        ref_arrays = ref["arrays"]

        # Golden digests recorded at save time (authoritative for bitwise pin).
        manifest = self._load_manifest()
        case_meta = manifest.get("cases", {}).get(slug, {})
        golden_hashes = case_meta.get("sha256_arrays", {})

        # If the manifest predates per-array hashing, fall back to hashing the
        # loaded golden arrays directly (same bytes, same digest).
        if not golden_hashes:
            golden_hashes = {k: compute_state_hash(v) for k, v in ref_arrays.items()}

        array_hashes = {}
        details = {}
        n_matched = 0
        # A bitwise regression must reproduce the *complete* golden snapshot:
        # every golden array key must be present in the re-run and match its
        # recorded digest.  Iterate over the union of golden and new keys so a
        # dropped key (or an extra key) is reported, not silently ignored.
        all_keys = sorted(set(ref_arrays.keys()) | set(new_arrays.keys()))
        for k in all_keys:
            if k not in new_arrays:
                # Golden key missing from the re-run: regression.
                details[k] = {"match": False, "golden": golden_hashes.get(k),
                              "new": None, "reason": "key_missing_in_rerun"}
                continue
            if k not in ref_arrays:
                # New key not present in golden snapshot: cannot pin bitwise.
                details[k] = {"match": False, "golden": None, "new": None,
                              "reason": "key_absent_in_golden"}
                continue
            new_h = compute_state_hash(np.asarray(new_arrays[k]))
            array_hashes[k] = new_h
            golden_h = golden_hashes.get(k)
            match = (golden_h is not None) and (new_h == golden_h)
            if match:
                n_matched += 1
            details[k] = {
                "match": match,
                "golden": golden_h,
                "new": new_h,
            }

        n_arrays = len(all_keys)
        bitwise_passed = (n_arrays > 0) and (n_matched == n_arrays)

        return {
            "bitwise_passed": bitwise_passed,
            "array_hashes": array_hashes,
            "golden_hashes": golden_hashes,
            "details": details,
            "n_arrays": n_arrays,
            "n_matched": n_matched,
        }

    def has_case(self, slug: str, version: str = "v1") -> bool:
        """Check if reference data exists for a case."""
        return self._npz_path(slug, version).exists()

    def list_cases(self) -> list:
        """List all cases in the manifest."""
        manifest = self._load_manifest()
        return list(manifest.get("cases", {}).keys())
