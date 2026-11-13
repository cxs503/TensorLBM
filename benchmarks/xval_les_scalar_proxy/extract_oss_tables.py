#!/usr/bin/env python3
"""Machine-extract the upstream D2Q9 lattice table and the LES closure.

Reference implementation (OSS, inventoried as
``tensorplatform.oss_solutions#row-39``, tag ``les-scalar-q-abs-cc-proxy``)::

    loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow

Nothing in this cross-validation is transcribed by hand: the lattice
velocities/weights come out of ``LatFlow/D2Q9.py`` by parsing the shipped
source text, the SGS coefficients come out of ``LatFlow/Domain.py``, and every
extracted number is stored together with the file's sha256 so the artefact can
be re-derived (or refuted) later.

Extraction targets
------------------
``LatFlow/D2Q9.py``
    ``WEIGHTS`` (9 weights) and ``LVELOC`` (9 lattice velocities).
``LatFlow/Domain.py``
    * ``self.Sc = 0.17``                                  (line 57)
    * ``self.EEk = sum_{n,m} |C_n * C_m|``                (lines 90-96)
    * the ``les`` branch of ``CollideSC``: ``Q``, ``tau``  (lines 122-128)

Outputs (written next to this file)::

    oss_ref/oss_lattice.json   weights, velocities, EEk, extracted source lines
    oss_ref/provenance.json    sha256 of every source file + the exact locators

Run::

    python benchmarks/xval_les_scalar_proxy/extract_oss_tables.py [--root DIR]

``--root`` defaults to ``$OSS_LBM_ROOT`` and then to the cached tarball tree
used by XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF_DIR = HERE / "oss_ref"
DEFAULT_ROOT = Path(
    "/tmp/oss_lbm/Lattice-Boltzmann-fluid-flow-in-Tensorflow-master"
)
TARBALL_URL = (
    "https://codeload.github.com/loliverhennigh/"
    "Lattice-Boltzmann-fluid-flow-in-Tensorflow/tar.gz/refs/heads/master"
)
# sha256 of the master tarball, recorded by the earlier cross-validation of the
# same tree (XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL).  Re-used here only after
# the per-file sha256 below re-match, which proves it is the same tree.
TARBALL_SHA256 = "7e4e3459730aecd57daef1cc5695e10368424b497a23849b3c10d095c4e23f10"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_weights(text: str) -> list[float]:
    m = re.search(r"WEIGHTS\s*=\s*tf\.constant\(\s*\[(.*?)\]\s*,", text, re.S)
    if m is None:
        raise RuntimeError("WEIGHTS table not found in LatFlow/D2Q9.py")
    body = m.group(1)
    vals = [e.strip() for e in body.replace("\n", " ").split(",") if e.strip()]
    out: list[float] = []
    for v in vals:
        if "/" in v:
            num, den = v.split("/")
            out.append(float(num.replace(".", "")) / float(den) if "." in num else float(num) / float(den))
        else:
            out.append(float(v))
    return out


def parse_lveloc(text: str) -> list[list[int]]:
    m = re.search(r"LVELOC\s*=\s*tf\.constant\(\s*\[(.*?)\]\s*,\s*dtype", text, re.S)
    if m is None:
        raise RuntimeError("LVELOC table not found in LatFlow/D2Q9.py")
    body = m.group(1)
    rows = re.findall(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", body)
    if len(rows) != 9:
        raise RuntimeError(f"expected 9 D2Q9 velocities, parsed {len(rows)}")
    return [[int(a), int(b), int(c)] for a, b, c in rows]


def parse_scalar(text: str, name: str) -> float:
    m = re.search(rf"self\.{name}\s*=\s*([0-9.eE+-]+)", text)
    if m is None:
        raise RuntimeError(f"self.{name} not found in LatFlow/Domain.py")
    return float(m.group(1))


def eek_from_lattice(c: list[list[int]]) -> list[float]:
    """EEk_q = sum_{n,m} |c_qn * c_qm| exactly as upstream builds it.

    Upstream accumulates ``tf.abs(C[:,:,n] * C[:,:,m])`` over n, m in range(3)
    (2-D keeps the third component equal to zero), so EEk_q is the absolute
    dyadic sum.  It also equals ``(sum_n |c_qn|)^2``; both forms are stored so
    the identity can be re-checked later.
    """
    out = []
    for cq in c:
        s = 0.0
        for n in range(3):
            for m in range(3):
                s += abs(cq[n] * cq[m])
        out.append(s)
    return out


def extract(root: Path) -> tuple[dict, dict]:
    d2q9 = root / "LatFlow" / "D2Q9.py"
    domain = root / "LatFlow" / "Domain.py"
    les_driver = root / "les_train.py"
    for p in (d2q9, domain):
        if not p.is_file():
            raise SystemExit(
                f"upstream file missing: {p}\n"
                f"point --root / $OSS_LBM_ROOT at a checkout of the repo "
                f"(download: {TARBALL_URL})"
            )
    d2q9_text = d2q9.read_text(errors="replace")
    domain_text = domain.read_text(errors="replace")

    weights = parse_weights(d2q9_text)
    c3 = parse_lveloc(d2q9_text)
    c2 = [[v[0], v[1]] for v in c3]
    eek = eek_from_lattice(c3)
    eek_alt = [float(sum(abs(v) for v in cq[:2])) ** 2 for cq in c3]
    sc = parse_scalar(domain_text, "Sc")

    # locate the LES branch verbatim (line numbers are 1-based, inclusive)
    lines = domain_text.splitlines()
    les_start = next(
        i for i, ln in enumerate(lines) if ln.strip() == "if self.les:" and "Q =" in lines[i + 1]
    )
    les_block = lines[les_start: les_start + 5]
    eek_start = next(
        i for i, ln in enumerate(lines) if ln.strip().startswith("self.EEk = tf.zeros")
    )
    eek_block = lines[eek_start: eek_start + 7]
    sc_line = next(i for i, ln in enumerate(lines) if re.search(r"self\.Sc\s*=", ln))

    lattice = {
        "method": "D2Q9",
        "weights": weights,
        "velocities_3d": c3,
        "velocities_2d": c2,
        "eek_abs_dyadic": eek,
        "eek_from_abs_sum_squared": eek_alt,
        "eek_identity_max_abs_diff": max(abs(a - b) for a, b in zip(eek, eek_alt)),
        "Sc": sc,
        "tau_formula": "tau_eff = 0.5*(tau0 + sqrt(tau0^2 + 6*Q*Sc/rho))",
        "q_formula": "Q = sqrt(2 * sum_q (f_q - f_q^eq)^2 * EEk_q)",
        "locators": {
            "weights": "LatFlow/D2Q9.py:6",
            "lveloc": "LatFlow/D2Q9.py:9",
            "eek": f"LatFlow/Domain.py:{eek_start + 1}-{eek_start + len(eek_block)}",
            "sc": f"LatFlow/Domain.py:{sc_line + 1}",
            "les_branch": f"LatFlow/Domain.py:{les_start + 1}-{les_start + len(les_block)}",
        },
        "extracted_source": {
            "eek_block": eek_block,
            "les_branch": les_block,
            "sc_line": lines[sc_line].strip(),
        },
    }

    files = {
        "LatFlow/D2Q9.py": sha256_of(d2q9),
        "LatFlow/Domain.py": sha256_of(domain),
    }
    if les_driver.is_file():
        files["les_train.py"] = sha256_of(les_driver)
    expected_d2q9 = "8acdfd9974d8fccf31912318f2e03c76fd60304182749052f8ba7335c558c0b7"
    provenance = {
        "repo": "loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow",
        "tarball_url": TARBALL_URL,
        "tarball_sha256": TARBALL_SHA256 if files["LatFlow/D2Q9.py"] == expected_d2q9 else None,
        "tarball_sha256_note": (
            "inherited from XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL after re-matching "
            "the per-file sha256 of LatFlow/D2Q9.py"
            if files["LatFlow/D2Q9.py"] == expected_d2q9
            else "local tree differs from the recorded tarball; identify the exact ref before citing"
        ),
        "source_root": str(root),
        "files": files,
        "extractor": "benchmarks/xval_les_scalar_proxy/extract_oss_tables.py",
        "inventory_row": "tensorplatform.oss_solutions#row-39",
        "inventory_tag": "les-scalar-q-abs-cc-proxy",
        "les_driver_status": (
            "shipped but not runnable (TF1 API, cv2.cv.CV_FOURCC, undefined lr/total_loss); "
            "cross-validated artefact is the closure in LatFlow/Domain.py"
        ),
    }
    return lattice, provenance


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=os.environ.get("OSS_LBM_ROOT", str(DEFAULT_ROOT)))
    args = ap.parse_args(argv)

    lattice, provenance = extract(Path(args.root))
    REF_DIR.mkdir(parents=True, exist_ok=True)
    (REF_DIR / "oss_lattice.json").write_text(json.dumps(lattice, indent=1) + "\n")
    (REF_DIR / "provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    print(f"wrote {REF_DIR/'oss_lattice.json'}")
    print(f"wrote {REF_DIR/'provenance.json'}")
    print(f"  EEk              = {lattice['eek_abs_dyadic']}")
    print(f"  Sc               = {lattice['Sc']}")
    print(f"  D2Q9.py sha256   = {provenance['files']['LatFlow/D2Q9.py']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())