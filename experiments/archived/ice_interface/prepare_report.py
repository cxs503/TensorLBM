#!/usr/bin/env python3
"""Prepare and submit the AG27W2-N08 report to the reports API."""

import json
import os
import base64
import sys

RUNS_DIR = os.path.join("runs", "ag27w2_n08")

# Load the report markdown
with open(os.path.join(RUNS_DIR, "REPORT.md")) as f:
    markdown = f.read()

# Load figures and encode as base64
figures = []
fig_files = [
    ("fig1_floe_field.png", "Broken-ice floe field: concentration and size distribution (C1 PASS)"),
    ("fig2_lbm_gci.png", "Water-phase residual decay and three-grid study (C2/C6)"),
    ("fig3_resistance_energy.png", "Ice resistance vs references and floe-phase power terms (C4/C5)"),
    ("fig4_dem_series.png", "Instantaneous DEM ice resistance on the hull"),
]

for fname, caption in fig_files:
    fpath = os.path.join(RUNS_DIR, fname)
    if os.path.exists(fpath):
        with open(fpath, "rb") as f:
            png_b64 = base64.b64encode(f.read()).decode("utf-8")
        figures.append({
            "name": fname,
            "caption": caption,
            "png_base64": png_b64,
        })

# Build report JSON
report = {
    "title": "AG27W2-N08: Full-Ship Ice Resistance Physics-Based Prediction Chain",
    "summary": "Implemented and partially verified a first-principles prediction chain for full-ship ice resistance combining DEM floe field + D3Q19 LBM water phase + two-way FSI coupling. C1 (floe field statistics) PASSES. C2 mass conservation PASSES. C3 FSI implementation consistency PASSES. C4/C5/C6 FAIL at verification scale due to physics limitations (DEM not statistically converged, sub-cell floes, coarse grids). Implementation is complete (~2058 lines) with clear documentation of limitations.",
    "scenario_key": "ag27w2_n08_ice_fullship",
    "mission_id": 7926,
    "tags": ["ice-resistance", "DEM", "LBM", "FSI", "KVLCC2", "broken-ice", "full-ship"],
    "markdown": markdown,
    "figures": figures,
}

with open("/tmp/report.json", "w") as f:
    json.dump(report, f)

print(f"Report prepared: {len(markdown)} chars, {len(figures)} figures")
print(f"Saved to /tmp/report.json")
