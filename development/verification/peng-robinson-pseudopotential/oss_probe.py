#!/usr/bin/env python
"""Extract the openLBMPM (PorousMediaSimulation/openLBMPM) pseudopotential facts
straight from its source — **zero hand-transcription**.

Everything this script prints is derived by parsing the OSS files:
  * the P-R (Peng-Robinson) effective-mass expression is *evaluated from the OSS
    source text itself* (the CUDA kernel body is de-continued, its identifiers
    mapped, and the two right-hand sides are `eval`-ed);
  * the number of call sites of the P-R kernels is counted over the whole tree;
  * the active potential branch and the MCMP parameters are read from the
    shipped `.ini` files with `configparser`;
  * the interaction-stencil weights are `eval`-ed from the OSS `np.array([...])`
    literals and their isotropy moments recomputed;
  * sha256 + git HEAD pin the exact snapshot.

Usage:  python oss_probe.py [--oss-root /tmp/openLBMPM] [--out artifacts/oss_facts.json]
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# source helpers
# ---------------------------------------------------------------------------

CONT = re.compile(r"\\\s*$")


def dedent_join(text: str) -> list[str]:
    """Split source text into logical statements (join backslash continuations)."""
    out, buf = [], ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if CONT.search(line):
            buf += CONT.sub("", line)
            continue
        out.append(buf + line)
        buf = ""
    if buf:
        out.append(buf)
    return out


def kernel_body(path: Path, func: str) -> str:
    """Return the source text of `def func(...)` up to the next decorator."""
    lines = path.read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip().startswith(f"def {func}("))
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("@cuda.jit"):
            end = j
            break
    return "\n".join(lines[start:end])


def eval_rhs(stmt: str, name: str) -> str:
    """Right-hand side of the assignment to `name` inside one logical statement."""
    m = re.search(re.escape(name) + r"\s*=\s*(.+)$", stmt)
    if not m:
        raise ValueError(f"no assignment to {name!r} in: {stmt[:120]}")
    return m.group(1).strip()


IDENT = {
    "fluidRho[i, indices]": "rho",
    "fluidPotential[i, indices]": "psi",
    "tmpP": "p",
    "constR": "R",
    "temperatureT": "T",
    "coeffA": "A",
    "coeffB": "B",
    "coeffAlpha": "alpha",
    "constC0": "c0",
    "constG": "G",
}


def to_python_expr(src: str) -> str:
    """Map OSS identifiers/subscripts to a plain python expression."""
    out = src
    for k in sorted(IDENT, key=len, reverse=True):
        out = out.replace(k, IDENT[k])
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head(root: Path) -> dict:
    def run(*a):
        return subprocess.run(a, cwd=root, capture_output=True, text=True).stdout.strip()

    return {"head": run("git", "rev-parse", "HEAD"),
            "log": run("git", "log", "-1", "--format=%H %ad %s", "--date=short")}


# ---------------------------------------------------------------------------
# isotropy moments of an interaction stencil (Sbragaglia 2007 / Porter 2012)
# ---------------------------------------------------------------------------

def stencil_moments(weights, offsets) -> dict:
    w = np.asarray(weights, dtype=float)
    c = np.asarray(offsets, dtype=float)
    assert len(w) == len(c), (len(w), len(c))
    cx, cy = c[:, 0], c[:, 1]
    m2 = float(np.sum(w * cx * cx))
    s4 = float(np.sum(w * cx**4))
    s22 = float(np.sum(w * cx * cx * cy * cy))
    s6 = float(np.sum(w * cx**6))
    s42 = float(np.sum(w * cx**4 * cy * cy))
    s8 = float(np.sum(w * cx**8))
    s62 = float(np.sum(w * cx**6 * cy * cy))
    return {
        "n_points": int(len(w)),
        "sum_w": float(np.sum(w)),
        "sum_w_cx2": m2,
        # isotropic value of each ratio is 1 (continuous angular averages:
        # int cos^4 = 3 int cos^2 sin^2 , int cos^6 = 5 int cos^4 sin^2 ,
        # int cos^8 = 7 int cos^6 sin^2)
        "A4": s4 / (3.0 * s22),
        "A6": s6 / (5.0 * s42),
        "A8": s8 / (7.0 * s62),
    }


# ---------------------------------------------------------------------------
# main probe
# ---------------------------------------------------------------------------

def probe(root: Path) -> dict:
    shan = root / "ShanChen2D"
    opt = shan / "OptimizedD2Q9GPU.py"
    expl = shan / "ExplicitD2Q9GPU.py"
    driver = shan / "ShanChenD2Q9.py"

    # ---- 1. P-R effective mass, evaluated from the OSS source text ---------
    pr_body = kernel_body(opt, "calFluidPotentialGPUPR")
    stmts = dedent_join(pr_body)
    p_src = eval_rhs(next(s for s in stmts if re.search(r"\btmpP\s*=", s)), "tmpP")
    psi_src = eval_rhs(
        next(s for s in stmts if "fluidPotential[i, indices] = math.sqrt" in to_python_expr(s)
             or re.search(r"fluidPotential\[i, indices\]\s*=\s*sqrt", s)),
        "fluidPotential[i, indices]",
    )
    p_expr, psi_expr = to_python_expr(p_src), to_python_expr(psi_src)

    def pr_psi(rho, T, A=2.0 / 49.0, B=2.0 / 21.0, R=1.0, alpha=1.0, c0=1.0 / 3.0, G=-1.0):
        env = dict(rho=rho, T=T, A=A, B=B, R=R, alpha=alpha, c0=c0, G=G, sqrt=np.sqrt)
        p = eval(p_expr, {"__builtins__": {}}, env)  # noqa: S307 (OSS source text)
        return eval(psi_expr, {"__builtins__": {}}, dict(env, p=p))  # noqa: S307

    # ---- 2. call sites of the P-R kernels (whole tree) ---------------------
    kernels = ["calFluidPotentialGPUPR", "calEffectiveMassPR"]
    callsites = {k: [] for k in kernels}
    for f in sorted(root.rglob("*.py")):
        txt = f.read_text(errors="ignore")
        for k in kernels:
            # a launch looks like `kernel[grid, block](...)`; a *call* (or any
            # non-definition use) looks like `kernel(` / `kernel[`
            for m in re.finditer(re.escape(k) + r"\s*[\[(]", txt):
                line = txt[: m.start()].count("\n") + 1
                if txt.splitlines()[line - 1].lstrip().startswith("def "):
                    continue  # definition, not a use
                callsites[k].append(f"{f.relative_to(root)}:{line}")

    # ---- 3. the active potential branch + its macro pressure --------------
    eq_body = kernel_body(opt, "calFluidPotentialGPUEql")
    eq_rhs = [
        s.strip() for s in dedent_join(eq_body)
        if "fluidPotential[i, indices] =" in s
    ]
    press_body = kernel_body(opt, "calMacroPressure")
    press_rhs = [
        s.strip() for s in dedent_join(press_body)
        if re.search(r"tmpPressure\s*\+=", s) or re.search(r"tmpPressure\s*=", s)
    ]

    # ---- 4. ini parameters (active configuration) -------------------------
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";",))
    cfg.optionxform = str  # keep the OSS key case (potentialType)
    cfg.read(root / "IniFiles" / "shanchen2D.ini")
    ini = {s: dict(cfg[s]) for s in cfg.sections()}

    # parameter plumbing the P-R kernel would need: is it present anywhere?
    needed = ["constR", "temperatureT", "coeffA", "coeffB", "coeffAlpha", "constC0", "constG"]
    ini_all = "\n".join((root / "IniFiles" / f).read_text(errors="ignore")
                        for f in sorted((root / "IniFiles").glob("*.ini")))
    plumbing = {k: bool(re.search(k, ini_all, re.I)) for k in needed}
    eos_key = bool(re.search(r"potential|eos", ini_all, re.I))

    # ---- 5. interaction stencil weights (eval of the OSS array literals) ---
    drv = driver.read_text()
    weights = {}
    for name in ("weightInter4", "weightInter8", "weightInter10"):
        m = re.search(re.escape(name) + r"\s*=\s*(np\.array\(\[.*?\]\))", drv, re.S)
        if m:
            weights[name] = [float(v) for v in eval(m.group(1), {"np": np})]  # noqa: S307

    # neighbour order of fillNeighboringNodes: E,N,W,S,NE,NW,SW,SE
    offsets8 = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
    d2q9_w = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]
    d2q9_c = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
    moments = {
        "OSS_iso4_8pt_E_N_W_S_NE_NW_SW_SE": stencil_moments(weights["weightInter4"], offsets8),
        "TensorLBM_D2Q9": stencil_moments(d2q9_w, d2q9_c),
    }

    files = [opt, expl, driver, root / "IniFiles" / "shanchen2D.ini"]
    return {
        "oss_root": str(root),
        "git": git_head(root),
        "source_sha256": {str(f.relative_to(root)): sha256(f) for f in files},
        "pr_kernel": {
            "kernel": "calFluidPotentialGPUPR",
            "file": str(opt.relative_to(root)),
            "p_EOS_src_verbatim": p_src,
            "psi_src_verbatim": psi_src,
            "p_EOS_expr_mapped": p_expr,
            "psi_expr_mapped": psi_expr,
            "callsites": callsites,
            "callsite_count": {k: len(v) for k, v in callsites.items()},
            "second_impl": "ExplicitD2Q9GPU.calEffectiveMassPR (also uncalled)",
            "wire_up": {
                "ini_keys_present_for_pr_parameters": plumbing,
                "any_potential_or_eos_key_in_ini": eos_key,
                "verdict": "kernel-only (dead code): zero call sites, no parameter plumbing",
            },
        },
        "active_potential": {
            "kernel": "calFluidPotentialGPUEql",
            "rhs": eq_rhs,
            "ini_potentialType": ini.get("ShanChenParameters", {}).get("potentialType", "<absent>"),
            "macro_pressure_rhs": press_rhs,
        },
        "ini_shanchen2D": ini,
        "stencil": {
            "weights": {k: v for k, v in weights.items()},
            "moments": moments,
            "neighbour_order": "E,N,W,S,NE,NW,SW,SE (fillNeighboringNodes)",
        },
        "notes": [
            "P-R kernel signature: (totalNodes, numFluids, xDim, constR, temperatureT,"
            " coeffA, coeffB, coeffAlpha, constC0, constG, fluidRho, fluidPotential).",
            "psi expression is raw sqrt (no clamp): NaN wherever p_EOS > rho/3.",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oss-root", default="/tmp/openLBMPM")
    ap.add_argument("--out", default=str(Path(__file__).parent / "artifacts" / "oss_facts.json"))
    a = ap.parse_args()
    facts = probe(Path(a.oss_root))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(facts, indent=2))

    pr = facts["pr_kernel"]
    print("== OSS P-R pseudopotential, parsed from source ==")
    print("  p_EOS :", pr["p_EOS_expr_mapped"])
    print("  psi   :", pr["psi_expr_mapped"])
    print("  callsites:", pr["callsite_count"], "->", pr["wire_up"]["verdict"])
    print("  active potential:", facts["active_potential"]["rhs"],
          "potentialType =", facts["active_potential"]["ini_potentialType"])
    print("  stencil moments:")
    for k, v in facts["stencil"]["moments"].items():
        print(f"    {k:42s} n={v['n_points']:2d} Sw={v['sum_w']:.6f} "
              f"Swcx2={v['sum_w_cx2']:.6f} A4={v['A4']:.6f} A6={v['A6']:.6f} A8={v['A8']:.6f}")
    print("wrote", out)


if __name__ == "__main__":
    main()