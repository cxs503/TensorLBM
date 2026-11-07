#!/usr/bin/env python3
"""Apply the G4 fix: SC single-component default coupling -4.0 -> +5.0
(library convention: G > 0 = attractive) + stale sign docstrings.

Run from repo root:  python3 /tmp/g4_apply.py
"""
from pathlib import Path

ROOT = Path("/root/tp-agent/work/m5051")

EDITS = {
    "src/tensorlbm/multiphase.py": [
        (
            "    The self-interaction (G < 0, attractive) generates liquid–gas coexistence\n"
            "    via a density-dependent pseudopotential.\n",
            "    The self-interaction (G > 0, attractive in this library's backward-gather\n"
            "    convention — see the CONVENTION NOTE above) generates liquid–gas\n"
            "    coexistence via a density-dependent pseudopotential.\n",
        ),
        (
            "        G:           SC self-coupling constant (< 0 for attraction / phase separation).\n",
            "        G:           SC self-coupling constant (> 0 for attraction / phase\n"
            "                     separation, library convention).\n",
        ),
        (
            "def collide_sc_single_component(\n"
            "    f: torch.Tensor,\n"
            "    G: float = -4.0,\n",
            "def collide_sc_single_component(\n"
            "    f: torch.Tensor,\n"
            "    G: float = 5.0,\n",
        ),
        (
            "    Use G < 0 to generate attractive self-interaction and spontaneous\n"
            "    liquid–gas phase separation.\n",
            "    Use G > 0 (this library's backward-gather convention, see the\n"
            "    CONVENTION NOTE above) to generate attractive self-interaction and\n"
            "    spontaneous liquid–gas phase separation.  For the classic psi_exp the\n"
            "    critical coupling is exactly G_c = +4.0 (classic −4.0): at G_c the EOS\n"
            "    spinodal degenerates to the single point ρ = ln 2 and no phase\n"
            "    separation is possible; the default +5.0 sits inside the coexistence\n"
            "    window (discrete density ratio ≈ 12, see ``tools/g4_ledger_audit.py``).\n",
        ),
        (
            "        G:           SC self-coupling constant (< 0 for phase separation with\n"
            "                     the classic psi_exp; +1.0 with the corrected-sign\n"
            "                     :func:`make_psi_eos` psi, see the convention note).\n",
            "        G:           SC self-coupling constant (> 0 for phase separation with\n"
            "                     the classic psi_exp, library convention; +1.0 with the\n"
            "                     corrected-sign :func:`make_psi_eos` psi, see the\n"
            "                     convention note).\n",
        ),
    ],
    "src/tensorlbm/multiphase3d.py": [
        (
            "def collide_sc_single_component_3d(\n"
            "    f: torch.Tensor,\n"
            "    G: float = -4.0,\n",
            "def collide_sc_single_component_3d(\n"
            "    f: torch.Tensor,\n"
            "    G: float = 5.0,\n",
        ),
        (
            "        G:           SC self-coupling constant (< 0 → attractive → phase sep.).\n",
            "        G:           SC self-coupling constant (> 0 → attractive → phase sep.,\n"
            "                     library convention; classic −4.0 is exactly critical).\n",
        ),
    ],
    "src/tensorlbm/multiphase3d_d3q27.py": [
        (
            "def collide_sc_single_component_27(\n"
            "    f: torch.Tensor,\n"
            "    G: float = -4.0,\n",
            "def collide_sc_single_component_27(\n"
            "    f: torch.Tensor,\n"
            "    G: float = 5.0,\n",
        ),
        (
            "        G:           SC self-coupling constant (< 0 → attractive → phase sep.).\n",
            "        G:           SC self-coupling constant (> 0 → attractive → phase sep.,\n"
            "                     library convention; classic −4.0 is exactly critical).\n",
        ),
    ],
    "src/tensorlbm/multiphase_benchmarks.py": [
        (
            "    The Shan-Chen single-component model with a self-attractive pseudopotential\n"
            "    (G < 0) is initialised from a uniform density field with small random\n"
            "    noise.\n",
            "    The Shan-Chen single-component model with a self-attractive pseudopotential\n"
            "    (G > 0, library convention) is initialised from a uniform density field\n"
            "    with small random noise.\n",
        ),
        (
            "    G:               SC self-coupling constant (< 0 for phase separation).\n",
            "    G:               SC self-coupling constant (> 0 for phase separation,\n"
            "                    library convention; classic −4.0 is exactly critical).\n",
        ),
        (
            "    nx: int = 64\n"
            "    ny: int = 64\n"
            "    G: float = -4.0\n",
            "    nx: int = 64\n"
            "    ny: int = 64\n"
            "    G: float = 5.0\n",
        ),
        (
            "    nx: int = 40\n"
            "    ny: int = 40\n"
            "    nz: int = 40\n"
            "    G: float = -4.0\n",
            "    nx: int = 40\n"
            "    ny: int = 40\n"
            "    nz: int = 40\n"
            "    G: float = 5.0\n",
        ),
    ],
}

for rel, pairs in EDITS.items():
    path = ROOT / rel
    text = path.read_text()
    for old, new in pairs:
        if old not in text:
            raise SystemExit(f"NOT FOUND in {rel}:\n{old!r}")
        if text.count(old) != 1:
            raise SystemExit(f"AMBIGUOUS ({text.count(old)}x) in {rel}:\n{old!r}")
        text = text.replace(old, new)
    path.write_text(text)
    print(f"patched {rel}: {len(pairs)} edits")
print("OK")
