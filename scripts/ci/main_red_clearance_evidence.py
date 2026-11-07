#!/usr/bin/env python3
"""Evidence collector for the main-red clearance of 2026-11-06.

The 2026-11-06 clearance fixed two families of red tests on the self-hosted
SDAA runner:

* ``CutCellBGK2D.Q`` indexed the ``(ny, nx)`` cut-cell volume with
  ``volume_m3[mask, None]``, an illegal boolean-index form, so every test that
  built or stepped the model died with ``IndexError``;
* ``multi_gpu.auto_decompose`` only probed CUDA and silently degraded to a
  single ``["cpu"]`` domain on an SDAA-only node (and had no way to request a
  device family explicitly).

This script turns the before/after pytest JUnit XMLs into one machine-readable
summary plus the figure embedded in ``docs/ci/main_red_clearance_20261106.md``.
Nothing in the summary is typed by hand.

Usage (from the repository root)::

    python scripts/ci/main_red_clearance_evidence.py \
        --before artifacts/ci/main_red_clearance/before.xml \
        --after  artifacts/ci/main_red_clearance/after.xml \
        --outdir artifacts/ci/main_red_clearance

The two XML files are produced by::

    git stash push -- src/tensorlbm/cut_cell_bgk_2d.py src/tensorlbm/multi_gpu.py
    python -m pytest tests/test_cut_cell_bgk_2d.py \
        tests/test_multi_device_sdaa_common.py -q --tb=no --junit-xml=.../before.xml
    git stash pop
    python -m pytest tests/test_cut_cell_bgk_2d.py \
        tests/test_multi_device_sdaa_common.py -q --tb=no --junit-xml=.../after.xml
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.cut_cell_bgk_2d import CutCellBGK2D  # noqa: E402
from tensorlbm.multi_gpu import auto_decompose  # noqa: E402


# --------------------------------------------------------------------------- #
# JUnit XML -> per-file outcome counts
# --------------------------------------------------------------------------- #
def _outcome(case: ET.Element) -> str:
    for child in case:
        tag = child.tag
        if tag in {"failure", "error"}:
            return "fail"
        if tag == "skipped":
            return "skip"
    return "pass"


def parse_junit(path: Path) -> dict[str, dict[str, int]]:
    """Return ``{test_file: {outcome: count}}`` for one JUnit XML run."""
    tree = ET.parse(path)
    per_file: dict[str, dict[str, int]] = {}
    for case in tree.iter("testcase"):
        # classname is e.g. "tests.test_cut_cell_bgk_2d" (module) or
        # "tests.test_x.TestClass"; derive "<file>::<rest>" as the test id.
        classname = case.get("classname", "")
        parts = classname.split(".")
        module = "tests/" + parts[1] + ".py" if len(parts) > 1 else classname
        counts = per_file.setdefault(module, {"pass": 0, "fail": 0, "skip": 0})
        counts[_outcome(case)] += 1
    return per_file


# --------------------------------------------------------------------------- #
# Physics/numerics probe: cut-cell volume expansion fidelity
# --------------------------------------------------------------------------- #
def _nonuniform_model() -> CutCellBGK2D:
    """Same non-uniform cut-cell case as ``tests/test_cut_cell_bgk_2d.py``."""
    nx, ny = 12, 10
    y, x = torch.meshgrid(
        torch.arange(ny, dtype=torch.float64),
        torch.arange(nx, dtype=torch.float64),
        indexing="ij",
    )
    rho = 1000 * (1 + 0.001 * torch.sin(2 * math.pi * x / nx) * torch.cos(2 * math.pi * y / ny))
    velocity = torch.stack(
        (0.01 * torch.sin(2 * math.pi * y / ny), 0.004 * torch.cos(2 * math.pi * x / nx)), dim=-1
    )
    return CutCellBGK2D(
        geometry={"center_m": (0.56, 0.47)}, density_kg_m3=rho, velocity_m_s=velocity
    )


def cut_cell_probe() -> dict[str, object]:
    """Reconstruct the per-cell mass field through ``Q`` and measure the error."""
    model = _nonuniform_model()
    mask = model.group_map >= 0
    Q = model.Q
    density = (model.group_Q.sum(1) / model.group_volume_m3)[model.group_map[mask]]
    expected = density * model.volume_m3[mask]
    residual = (Q.sum(0)[mask] - expected).abs()
    return {
        "grid_ny_nx": [int(model.geometry.ny), int(model.geometry.nx)],
        "fluid_cells": int(mask.sum()),
        "merged_groups": int(model.group_count),
        "field_shape": list(Q.shape),
        "residual_max_kg_per_m": float(residual.max()),
        "residual_mean_kg_per_m": float(residual.mean()),
        "mass_total_kg": float(Q.sum()),
        "mass_total_group_kg": float(model.group_Q.sum()),
        "mass_relative_error": float(abs(Q.sum() - model.group_Q.sum()) / model.group_Q.sum()),
        "residuals_kg_per_m": [float(v) for v in residual],
    }


def device_probe() -> dict[str, object]:
    """What ``auto_decompose`` now selects on this host."""
    f = torch.zeros(19, 2, 2, 8)
    probe = {
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_count": int(torch.cuda.device_count()),
        "torch_sdaa_present": hasattr(torch, "sdaa"),
        "torch_sdaa_available": bool(torch.sdaa.is_available())
        if hasattr(torch, "sdaa")
        else False,
        "torch_sdaa_count": int(torch.sdaa.device_count()) if hasattr(torch, "sdaa") else 0,
    }
    probe["auto_devices"] = auto_decompose(f).devices
    probe["explicit_sdaa_2"] = auto_decompose(f, device_type="sdaa", n_devices=2).devices
    probe["explicit_cpu"] = auto_decompose(f, device_type="cpu", n_devices=0).devices
    return probe


# --------------------------------------------------------------------------- #
# Report artefacts
# --------------------------------------------------------------------------- #
def build_summary(before: Path, after: Path) -> dict[str, object]:
    per_file_before = parse_junit(before)
    per_file_after = parse_junit(after)
    files = sorted(set(per_file_before) | set(per_file_after))
    table = []
    for name in files:
        b = per_file_before.get(name, {"pass": 0, "fail": 0, "skip": 0})
        a = per_file_after.get(name, {"pass": 0, "fail": 0, "skip": 0})
        table.append(
            {
                "file": name,
                "before_fail": b["fail"],
                "before_pass": b["pass"],
                "after_fail": a["fail"],
                "after_pass": a["pass"],
                "after_skip": a["skip"],
            }
        )
    totals = {
        "before_fail": sum(r["before_fail"] for r in table),
        "before_pass": sum(r["before_pass"] for r in table),
        "after_fail": sum(r["after_fail"] for r in table),
        "after_pass": sum(r["after_pass"] for r in table),
        "after_skip": sum(r["after_skip"] for r in table),
    }
    return {
        "generated_utc": __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "per_file": table,
        "totals": totals,
        "cut_cell_probe": cut_cell_probe(),
        "device_probe": device_probe(),
    }


def draw_figure(summary: dict[str, object], out_png: Path) -> None:
    per_file = summary["per_file"]
    probe = summary["cut_cell_probe"]
    dev = summary["device_probe"]
    totals = summary["totals"]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), gridspec_kw={"width_ratios": [1.0, 1.15]})

    # -- panel A: red tests per file, before vs after ------------------------
    ax = axes[0]
    labels = [Path(r["file"]).name.replace("test_", "").replace(".py", "") for r in per_file]
    before = [r["before_fail"] for r in per_file]
    after = [r["after_fail"] for r in per_file]
    y = range(len(labels))
    h = 0.36
    ax.barh([i + h / 2 for i in y], before, height=h, color="#c0392b", label="before fix")
    ax.barh([i - h / 2 for i in y], after, height=h, color="#27ae60", label="after fix")
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(b + 0.2, i + h / 2, str(b), va="center", fontsize=9, color="#c0392b")
        ax.text(a + 0.2, i - h / 2, str(a), va="center", fontsize=9, color="#27ae60")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("failing tests (SDAA runner)")
    ax.set_xlim(0, max(before + [1]) * 1.25)
    ax.set_title(
        f"(a) main red clearance: {totals['before_fail']} -> {totals['after_fail']} failures",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    # -- panel B: cut-cell volume expansion residual ------------------------
    ax = axes[1]
    residuals = probe["residuals_kg_per_m"]
    ax.hist(
        residuals,
        bins=24,
        color="#2c7fb8",
        edgecolor="white",
        label=f"n={probe['fluid_cells']} fluid cells",
    )
    ax.set_xlabel("|Q.sum(0) - rho_cell * V_cut|  [kg/m]")
    ax.set_ylabel("fluid cells")
    ax.set_title(
        "(b) cut-cell field reconstruction through Q\n"
        f"max residual {probe['residual_max_kg_per_m']:.2e} kg/m, "
        f"relative mass error {probe['mass_relative_error']:.2e}",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    txt = (
        "auto_decompose on this host\n"
        f"  cuda avail {dev['torch_cuda_available']} (n={dev['torch_cuda_count']})   "
        f"sdaa avail {dev['torch_sdaa_available']} (n={dev['torch_sdaa_count']})\n"
        f"  auto          -> {dev['auto_devices']}\n"
        f"  sdaa, 2 cards -> {dev['explicit_sdaa_2']}\n"
        f"  cpu           -> {dev['explicit_cpu']}"
    )
    ax.text(
        0.02,
        0.62,
        txt,
        transform=ax.transAxes,
        fontsize=7.6,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "#f7f7f7", "edgecolor": "#999999"},
    )
    fig.suptitle(
        "TensorLBM main-red clearance 2026-11-06 (cut-cell Q index + accelerator auto-detect)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before", type=Path, required=True)
    ap.add_argument("--after", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(args.before, args.after)
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    draw_figure(summary, args.outdir / "main_red_clearance.png")
    print(json.dumps({k: summary[k] for k in ("totals", "per_file")}, indent=2))


if __name__ == "__main__":
    main()
