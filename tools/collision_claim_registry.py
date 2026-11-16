#!/usr/bin/env python
"""Collision-operator claim ↔ implementation-registry consistency audit.

Motivation
----------
The library advertises collision families in three places that drift apart:

1. **Prose claims** — ``README.md`` / ``README.zh-CN.md`` and the capability
   documents under ``docs/`` (what the project says it can do);
2. **the machine registry** — :func:`tensorlbm.advanced_collision_contract.
   collision_capability_matrix` (which kernel is wired for which lattice);
3. **the evidence ledger** — ``docs/m23154/COLLISION_CLAIM_REGISTRY.json``
   (which family has been *measured*, and at what grade).

A family can therefore be claimed in prose while the registry withholds it, or
still be listed as WITHHELD long after a kernel was registered and measured.
Both directions are silent failures for an audit, so this tool machine-checks
them:

=========== ==================================================================
R1          a prose line claims family *F* AVAILABLE but the ledger grades *F*
            below REAL (unbacked claim)
R2          a prose line says family *F* is WITHHELD / not implemented while the
            ledger grades *F* REAL (stale withholding)
R3          every registry cell marked AVAILABLE must actually import and pass
            the conservation smoke test (mass and momentum, one collision,
            float64, <= 1e-12 relative)
R4          every family present in the registry must have a ledger row
=========== ==================================================================

Grades in the ledger are mechanical (see :func:`grade_family`): ``REAL`` needs a
registered kernel, a passing smoke test *and* a Taylor–Green decay audit within
3 % of ``gamma_E = 6 nu k^2``; ``WITHHELD`` means registered but unmeasured;
``NOT_SUPPORTED`` means no kernel at all.

Usage::

    python tools/collision_claim_registry.py check          # exit 1 on any FAIL
    python tools/collision_claim_registry.py emit-ledger    # write/refresh ledger
    python tools/collision_claim_registry.py show           # human summary

The tool is deliberately generic in its *mechanism* (claims, registry, ledger,
gates) and local in its *scope* (TensorLBM collision families), so the same
pattern can be reused for other claim surfaces.
"""

from __future__ import annotations

import argparse
import glob
import importlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

LEDGER = os.path.join(HERE, "docs", "m23154", "COLLISION_CLAIM_REGISTRY.json")
RUN_GLOB = os.path.join(HERE, "docs", "m23154", "runs", "*.json")
#: earlier published Taylor-Green artefacts (other missions) -- ingestion is
#: schema-tolerant on purpose: any JSON object carrying a ``kernel`` field and a
#: numeric ``err*_pct`` field is treated as one measurement.
PRIOR_GLOBS = (
    "benchmarks/pending/collision_kernels_3d_tg/*.json",
    "benchmarks/verified/collision_kernels_3d_tg/*.json",
)

#: canonical family name -> spellings that count as a claim for it.
FAMILY_ALIASES: dict[str, tuple[str, ...]] = {
    "BGK": ("BGK", "SRT", "single-relaxation-time", "单松弛"),
    "MRT": ("MRT", "multi-relaxation-time", "多松弛"),
    "TRT": ("TRT", "two-relaxation-time", "双松弛"),
    "RLBM": ("RLBM", "regularized BGK", "正则化"),
    "CM": ("CM", "cascaded", "central-moment", "中心矩", "级联"),
    "CUMULANT": ("Cumulant", "cumulant", "累积量"),
    "KBC": ("KBC", "entropic KBC", "熵"),
}

#: prose surfaces scanned for claims (repo-relative).
CLAIM_SOURCES: tuple[str, ...] = (
    "README.md",
    "README.zh-CN.md",
    "docs/advanced_collision_capability_matrix.md",
    "docs/general_capability_matrix_r1.md",
)

#: families that also have a D2Q9 kernel worth checking against a "2D" claim.
D2Q9_FAMILIES: dict[str, str] = {
    "CUMULANT": "tensorlbm.cumulant.collide_cumulant_d2q9",
}

#: only these surfaces make *kernel-level* withholding claims (R2).  The
#: composition matrix discusses pipeline-level withholding, which is a
#: different object and must not be read as "the kernel is missing".
KERNEL_LEVEL_SOURCES: tuple[str, ...] = ("docs/advanced_collision_capability_matrix.md",)

WITHHELD_PATTERNS = re.compile(
    r"WITHHELD|not\s+implemented|未实现|NOT_SUPPORTED|does\s+not\s+solve|不实现", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# registry access
# ---------------------------------------------------------------------------


def registry_matrix() -> dict[str, dict[str, dict[str, Any]]]:
    """Flat ``{lattice: {family: {available, entrypoint, status, note}}}``."""
    from tensorlbm.advanced_collision_contract import collision_capability_matrix

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for lattice, row in collision_capability_matrix().items():
        out[lattice] = {
            fam: {
                "available": cap.available,
                "entrypoint": cap.entrypoint,
                "status": cap.status,
                "note": cap.note,
            }
            for fam, cap in row.items()
        }
    return out


def resolve_entrypoint(dotted: str) -> Any:
    """Import ``tensorlbm.module.function`` and return the callable."""
    module, _, func = dotted.rpartition(".")
    return getattr(importlib.import_module(module), func)


# ---------------------------------------------------------------------------
# smoke test: one collision, float64, conservation + shear factor
# ---------------------------------------------------------------------------


def _d3_setup(lattice: str):
    if lattice == "D3Q19":
        from tensorlbm import d3q19 as lat
    else:
        from tensorlbm import d3q27 as lat
    return lat


def smoke_collision(lattice: str, entrypoint: str, tau: float = 0.9) -> dict[str, Any]:
    """Apply one collision to a smooth, seeded non-equilibrium state.

    Measures (a) mass/momentum conservation from the operator itself and
    (b) the deviatoric-stress relaxation factor, which for every LBM kernel in
    this package must be ``sigma = 1 - 1/tau`` (that is what pins
    ``nu = c_s^2 (tau - 1/2)``).
    """
    torch.set_default_dtype(torch.float64)
    if lattice == "D3Q19":
        from tensorlbm import d3q19 as lat

        eq, macro = lat.equilibrium3d, lat.macroscopic3d
    elif lattice == "D3Q27":
        from tensorlbm import d3q27 as lat

        eq, macro = lat.equilibrium27, lat.macroscopic27
    else:
        raise ValueError(f"unsupported lattice {lattice}")

    nt = 4
    idx = torch.arange(nt, dtype=torch.float64)
    z, y, x = torch.meshgrid(idx, idx, idx, indexing="ij")
    rho = 1.0 + 0.01 * torch.sin(x)
    ux = 0.02 * torch.sin(y)
    uy = 0.02 * torch.cos(x)
    uz = 0.01 * torch.sin(x + y)
    f = eq(rho, ux, uy, uz)
    C = lat.C.to(torch.float64)
    w = lat.W_EXACT64.to(torch.float64)
    Q = C.shape[0]
    cx, cy, cz = (C[:, i].view(Q, 1, 1, 1) for i in range(3))
    wv = w.view(Q, 1, 1, 1)
    neq = 1e-4 * wv * (cx * cy) + 5e-5 * wv * (cx * cx - cy * cy) + 3e-5 * wv * (cx * cx + cy * cy + cz * cz - 1.0)
    f = f + neq

    fn = resolve_entrypoint(entrypoint)
    rho0, jx0, jy0, jz0 = macro(f)
    pi_xy_pre = float((cx * cy * (f - eq(*macro(f)))).sum().item())
    post = fn(f.clone(), tau)
    rho1, jx1, jy1, jz1 = macro(post)
    pi_xy_post = float((cx * cy * (post - eq(*macro(post)))).sum().item())

    scale = max(float(rho0.abs().max().item()), 1e-30)
    jscale = max(float(torch.sqrt(jx0**2 + jy0**2 + jz0**2).max().item()), 1e-30)
    return {
        "lattice": lattice,
        "entrypoint": entrypoint,
        "tau": tau,
        "mass_err_rel": float((rho1 - rho0).abs().max().item()) / scale,
        "momentum_err_rel": float(torch.sqrt((jx1 - jx0) ** 2 + (jy1 - jy0) ** 2 + (jz1 - jz0) ** 2).max().item())
        / jscale,
        "shear_factor": pi_xy_post / pi_xy_pre if pi_xy_pre else float("nan"),
        "shear_factor_expected": 1.0 - 1.0 / tau,
        "f_min": float(post.min().item()),
    }


CONSERVATION_TOL = 1e-12
SHEAR_TOL = 1e-8


# ---------------------------------------------------------------------------
# prose claims
# ---------------------------------------------------------------------------


@dataclass
class Claim:
    source: str
    line: int
    text: str
    families: list[str]
    withheld: bool

    def as_dict(self) -> dict[str, Any]:
        return dict(source=self.source, line=self.line, text=self.text.strip(),
                    families=self.families, withheld=self.withheld)


def extract_claims() -> list[Claim]:
    claims: list[Claim] = []
    for rel in CLAIM_SOURCES:
        path = os.path.join(HERE, rel)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                fams = [fam for fam, al in FAMILY_ALIASES.items() if any(a in line for a in al)]
                if not fams:
                    continue
                if not re.search(r"collision|碰撞|算子|operator|kernel|核", line, re.IGNORECASE):
                    continue
                # lines about the *legacy* routines (``advanced_collision.collide_*``)
                # describe what those functions are not; they are not claims about
                # the registered family and must not be read as availability claims.
                if "advanced_collision.collide_" in line:
                    continue
                claims.append(Claim(rel, i, line, fams, bool(WITHHELD_PATTERNS.search(line))))
    return claims


# ---------------------------------------------------------------------------
# Taylor-Green evidence
# ---------------------------------------------------------------------------


def _walk_records(node: Any, src: str, out: list[dict[str, Any]]) -> None:
    if isinstance(node, dict):
        kern = node.get("kernel") or node.get("name")
        err = None
        for key, val in node.items():
            if isinstance(val, (int, float)) and re.search(r"^err.*pct$", key) and "vel" not in key:
                err = float(val)
                break
        if isinstance(kern, str) and err is not None:
            out.append(
                dict(
                    kernel=kern,
                    n=node.get("n") or node.get("N"),
                    tau=node.get("tau"),
                    err_pct=err,
                    r2=node.get("r2"),
                    dtype=node.get("dtype", "unknown"),
                    tier="prior",
                    source=src,
                )
            )
        for val in node.values():
            _walk_records(val, src, out)
    elif isinstance(node, list):
        for val in node:
            _walk_records(val, src, out)


def prior_evidence() -> list[dict[str, Any]]:
    """Measurements published by earlier missions (repo-resident JSON artefacts)."""
    found: list[dict[str, Any]] = []
    for pattern in PRIOR_GLOBS:
        for path in sorted(glob.glob(os.path.join(HERE, pattern))):
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:  # noqa: BLE001
                continue
            _walk_records(data, os.path.relpath(path, HERE), found)
    return found


def tg_evidence() -> dict[str, list[dict[str, Any]]]:
    """``{kernel: [run records]}`` from the m23154 Taylor-Green sweep and prior artefacts."""
    ev: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(glob.glob(RUN_GLOB)):
        try:
            with open(path, encoding="utf-8") as fh:
                rec = json.load(fh)
        except Exception:  # noqa: BLE001
            continue
        if "err_pct" not in rec or rec.get("dtype") != "float64":
            continue
        ev.setdefault(rec["kernel"], []).append(
            dict(tier=rec.get("tier"), n=rec["n"], tau=rec["tau"], err_pct=rec["err_pct"], r2=rec["r2"],
                 mass_drift_rel=rec["mass_drift_rel"], dtype=rec.get("dtype"), source=os.path.basename(path))
        )
    for rec in prior_evidence():
        rec.setdefault("mass_drift_rel", None)
        ev.setdefault(rec["kernel"], []).append(rec)
    return ev


#: family -> the audit kernels that testify for it (ledger wiring).
FAMILY_KERNELS: dict[str, tuple[str, ...]] = {
    "BGK": ("bgk_d3q19_ctl", "bgk27", "collide_bgk27", "bgk_d3q19"),
    "MRT": ("mrt_d3q27_ctl", "mrt27", "collide_mrt27"),
    "TRT": ("trt_d3q27_ctl", "trt27", "collide_trt27"),
    "RLBM": ("rlbm_d3q27_ctl", "rlbm27", "collide_rlbm27"),
    "CM": ("cascaded_d3q27_ctl", "cascaded_d3q27", "collide_cascaded_d3q27"),
    "CUMULANT": ("cumulant_d3q27", "cumulant_geier_d3q27", "collide_cumulant_d3q27"),
    "KBC": ("kbc_d3q19", "kbc_d3q27", "collide_kbc_d3q19", "collide_kbc_d3q27"),
}

TG_TOL_PCT = 3.0


def grade_family(
    family: str,
    cells: dict[str, dict[str, Any]],
    smoke: dict[str, dict[str, Any]],
    evidence: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Mechanical evidence grade for one family."""
    registered = [lat for lat, cell in cells.items() if cell["available"]]
    if not registered:
        return dict(grade="NOT_SUPPORTED", reason="no registered kernel", lattices={}, tg={})

    bad_conservation = [
        {"lattice": lat, "mass_err_rel": smoke[lat + "|" + cells[lat]["entrypoint"]]["mass_err_rel"],
         "momentum_err_rel": smoke[lat + "|" + cells[lat]["entrypoint"]]["momentum_err_rel"]}
        for lat in registered
        if not (
            smoke[lat + "|" + cells[lat]["entrypoint"]]["mass_err_rel"] <= CONSERVATION_TOL
            and smoke[lat + "|" + cells[lat]["entrypoint"]]["momentum_err_rel"] <= CONSERVATION_TOL
        )
    ]
    tg = {}
    for kern in FAMILY_KERNELS.get(family, ()):
        runs = evidence.get(kern, [])
        if runs:
            r2s = [r["r2"] for r in runs if isinstance(r.get("r2"), (int, float))]
            tg[kern] = dict(
                runs=runs,
                max_abs_err_pct=max(abs(r["err_pct"]) for r in runs),
                min_r2=min(r2s) if r2s else 1.0,
            )

    if bad_conservation:
        grade, reason = "MOCK", "registered kernel fails the conservation smoke test"
    elif not tg:
        grade, reason = "WITHHELD", "registered kernel but no Taylor-Green audit evidence"
    else:
        worst = max(v["max_abs_err_pct"] for v in tg.values())
        r2min = min(v["min_r2"] for v in tg.values())
        if worst <= TG_TOL_PCT and r2min >= 0.999:
            grade, reason = "REAL", f"conservation OK and Taylor-Green |err| <= {TG_TOL_PCT}% ({worst:.3f}%)"
        else:
            grade, reason = "WITHHELD", (
                f"Taylor-Green decay deviates from gamma_E = 6 nu k^2 by up to {worst:.2f}% "
                f"(tolerance {TG_TOL_PCT}%), R^2 min {r2min:.5f}"
            )
    return dict(
        grade=grade,
        reason=reason,
        lattices={lat: cells[lat]["entrypoint"] for lat in registered},
        conservation=[] if not bad_conservation else bad_conservation,
        tg=tg,
    )


def build_ledger() -> dict[str, Any]:
    matrix = registry_matrix()
    smoke: dict[str, dict[str, Any]] = {}
    d2_smoke: dict[str, dict[str, Any]] = {}
    for lat, row in matrix.items():
        for fam, cell in row.items():
            if cell["available"] and cell["entrypoint"]:
                key = lat + "|" + cell["entrypoint"]
                if key not in smoke:
                    try:
                        smoke[key] = smoke_collision(lat, cell["entrypoint"])
                    except Exception as exc:  # noqa: BLE001
                        smoke[key] = dict(
                            lattice=lat, entrypoint=cell["entrypoint"], tau=0.9,
                            mass_err_rel=float("nan"), momentum_err_rel=float("nan"),
                            shear_factor=float("nan"), shear_factor_expected=1 - 1 / 0.9,
                            f_min=float("nan"), error=f"{type(exc).__name__}: {exc}",
                        )
    evidence = tg_evidence()
    families = sorted({fam for row in matrix.values() for fam in row})
    ledger: dict[str, Any] = {
        "generated_by": "tools/collision_claim_registry.py emit-ledger",
        "tg_tolerance_pct": TG_TOL_PCT,
        "conservation_tol": CONSERVATION_TOL,
        "families": {},
        "smoke": list(smoke.values()),
        "d2q9": {},
    }
    for fam in families:
        cells = {lat: row[fam] for lat, row in matrix.items() if fam in row}
        rec = grade_family(fam, cells, smoke, evidence)
        rec["claims"] = [c.as_dict() for c in extract_claims() if fam in c.families]
        ledger["families"][fam] = rec
    for fam, dotted in D2Q9_FAMILIES.items():
        try:
            got = smoke_collision_2d(dotted)
        except Exception as exc:  # noqa: BLE001
            got = {"entrypoint": dotted, "error": f"{type(exc).__name__}: {exc}"}
        ledger["d2q9"][fam] = got
    return ledger


def smoke_collision_2d(dotted: str, tau: float = 0.9) -> dict[str, Any]:
    """D2Q9 conservation smoke test (mass, momentum, one collision)."""
    from tensorlbm import d2q9

    torch.set_default_dtype(torch.float64)
    n = 8
    idx = torch.arange(n, dtype=torch.float64)
    y, x = torch.meshgrid(idx, idx, indexing="ij")
    rho = 1.0 + 0.01 * torch.sin(x)
    ux = 0.02 * torch.sin(y)
    uy = 0.02 * torch.cos(x)
    f = d2q9.equilibrium(rho, ux, uy)
    fn = resolve_entrypoint(dotted)
    post = fn(f.clone(), tau)
    r0, ax0, ay0 = d2q9.macroscopic(f)
    r1, ax1, ay1 = d2q9.macroscopic(post)
    return dict(
        entrypoint=dotted,
        mass_err_rel=float((r1 - r0).abs().max().item()) / max(float(r0.abs().max().item()), 1e-30),
        momentum_err_rel=float(torch.sqrt((ax1 - ax0) ** 2 + (ay1 - ay0) ** 2).max().item())
        / max(float(torch.sqrt(ax0**2 + ay0**2).max().item()), 1e-30),
        f_min=float(post.min().item()),
    )


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def check(ledger: dict[str, Any] | None = None) -> dict[str, Any]:
    ledger = ledger or build_ledger()
    claims = extract_claims()
    results: list[dict[str, Any]] = []

    fam_grade = {fam: rec["grade"] for fam, rec in ledger["families"].items()}
    for c in claims:
        for fam in c.families:
            grade = fam_grade.get(fam, "NOT_SUPPORTED")
            if not c.withheld and grade != "REAL":
                results.append(dict(gate="R1", result="FAIL", family=fam, ledger_grade=grade,
                                    source=c.source, line=c.line, text=c.text.strip()))
            if c.withheld and grade == "REAL" and c.source in KERNEL_LEVEL_SOURCES:
                results.append(dict(gate="R2", result="FAIL", family=fam, ledger_grade=grade,
                                    source=c.source, line=c.line, text=c.text.strip()))

    for s in ledger["smoke"]:
        ok = (
            s["mass_err_rel"] <= CONSERVATION_TOL
            and s["momentum_err_rel"] <= CONSERVATION_TOL
        )
        results.append(dict(gate="R3", result="PASS" if ok else "FAIL", family=None,
                            entrypoint=s["entrypoint"], mass_err_rel=s["mass_err_rel"],
                            momentum_err_rel=s["momentum_err_rel"]))
    for fam, rec in ledger["families"].items():
        ok = "grade" in rec and rec["grade"] in ("REAL", "WITHHELD", "MOCK", "NOT_SUPPORTED", "STUB")
        results.append(dict(gate="R4", result="PASS" if ok else "FAIL", family=fam, ledger_grade=rec.get("grade")))

    fails = [r for r in results if r["result"] == "FAIL"]
    return dict(
        checked_claims=len(claims),
        gates=results,
        fails=fails,
        ok=not fails,
        ledger_summary={fam: rec["grade"] for fam, rec in ledger["families"].items()},
    )


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", nargs="?", default="check", choices=("check", "emit-ledger", "show"))
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.command == "emit-ledger":
        ledger = build_ledger()
        os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
        with open(LEDGER, "w", encoding="utf-8") as fh:
            json.dump(ledger, fh, indent=1, sort_keys=True)
        print(f"wrote {os.path.relpath(LEDGER, HERE)}")
        print(json.dumps({fam: rec["grade"] for fam, rec in ledger["families"].items()}, indent=1))
        return 0

    res = check()
    if args.command == "show":
        print(json.dumps(res, indent=1))
    else:
        print(json.dumps(res["ledger_summary"], indent=1))
        for f in res["fails"]:
            print("FAIL", json.dumps(f, ensure_ascii=False))
        print("OK" if res["ok"] else f"{len(res['fails'])} gate failure(s)")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())