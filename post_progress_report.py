#!/usr/bin/env python3
"""Generate and post the m728 progress report (QA batch review)."""
import requests
import json
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = "http://127.0.0.1:8010/api/v1"

# Login
resp = requests.post(f"{BASE}/auth/login", json={
    "username": "eco_admin",
    "password": "EcoTest!2026pass"
})
token = resp.json()["access_token"]
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# Generate summary figure: QA batch results bar chart
fig, ax = plt.subplots(figsize=(8, 4))
reports = ['613\n(m720)', '614\n(m711)', '615\n(m706)']
scores = [0.7, 1.0, 0.7]
colors = ['#f0ad4e', '#5cb85c', '#f0ad4e']
labels = ['MINOR', 'PASS', 'MINOR']

bars = ax.bar(reports, scores, color=colors, edgecolor='white', linewidth=1.5)
ax.set_ylim(0, 1.2)
ax.set_ylabel('Quality Score', fontsize=11)
ax.set_title('QA Batch 613,614,615 - Quality Assessment', fontsize=12, fontweight='bold')
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['RESUBMIT', '', '', 'MINOR', 'PASS'])

for bar, label in zip(bars, labels):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
            label, ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.axhline(y=0.7, color='#f0ad4e', linestyle='--', alpha=0.5, label='MINOR threshold')
ax.axhline(y=1.0, color='#5cb85c', linestyle='--', alpha=0.5, label='PASS threshold')
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()

buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=150)
buf.seek(0)
png_b64 = base64.b64encode(buf.read()).decode('ascii')
plt.close()

# Generate number traceability figure
fig2, ax2 = plt.subplots(figsize=(8, 3.5))
categories = ['613: mass_err', '613: mom_err', '613: subdiv',
              '614: sail_force', '614: keel_drag', '614: total',
              '615: momentum', '615: settling', '615: pytest']
norm_values = [0, 0, 0.001, 0.89, 4.8, 5.69, 0, 0.07, 1.0]
colors2 = ['#5cb85c'] * 9

bars2 = ax2.barh(categories, norm_values, color=colors2, edgecolor='white')
ax2.set_xlabel('Normalized Value (for display)', fontsize=10)
ax2.set_title('Number Traceability: All 9抽查数字 Matched (3 per report)', fontsize=11, fontweight='bold')
actual_vals = ['0.0', '0.0', '4.26e-14', '889662', '4799974', '5689637', '0.0', '0.07%', '100%']
for bar, val in zip(bars2, actual_vals):
    ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
             val, ha='left', va='center', fontsize=8)
plt.tight_layout()

buf2 = io.BytesIO()
plt.savefig(buf2, format='png', dpi=150)
buf2.seek(0)
png_b64_2 = base64.b64encode(buf2.read()).decode('ascii')
plt.close()

# Build markdown report using string concatenation to avoid format issues
markdown_lines = []
markdown_lines.append("# QA Batch Review: Reports 613, 614, 615 - Progress Report")
markdown_lines.append("")
markdown_lines.append("## 1. Case Introduction")
markdown_lines.append("")
markdown_lines.append("**Physical Background**: The ship performance platform regularly produces technical reports that require dedicated QA to ensure quality. This task reviews batch 613/614/615 (corresponding to missions m720/m711/m706).")
markdown_lines.append("")
markdown_lines.append("**QA Dimensions**:")
markdown_lines.append("| Dimension | Weight | Description |")
markdown_lines.append("|-----------|--------|-------------|")
markdown_lines.append("| Structure (5 sections) | 20% | Purpose/Method/Results/Audit/Limitations |")
markdown_lines.append("| Number traceability | 30% | Verify 3 key numbers against artifacts |")
markdown_lines.append("| honest-FAIL | 20% | Original FAIL must not be disguised as PASS |")
markdown_lines.append("| Figures | 15% | Field contour + comparison curves |")
markdown_lines.append("| Error assessment | 15% | Key quantities vs baseline <3% |")
markdown_lines.append("")
markdown_lines.append("**Reports Under Review**:")
markdown_lines.append("| Report ID | Mission | Topic | Domain |")
markdown_lines.append("|-----------|---------|-------|--------|")
markdown_lines.append("| 613 | 720 | FEM->DEM Fracture Fragment Handoff | Ice/Structure |")
markdown_lines.append("| 614 | 711 | Ice Ridge Mechanics | Ice Navigation |")
markdown_lines.append("| 615 | 706 | DEM<->Fluid Bidirectional Coupling | Ice/CFD |")
markdown_lines.append("")
markdown_lines.append("## 2. Methodology")
markdown_lines.append("")
markdown_lines.append("**QA Process**:")
markdown_lines.append("1. `curl` login to platform -> obtain JWT token")
markdown_lines.append("2. `GET /tensor/reports/{id}` to fetch all three report markdown")
markdown_lines.append("3. Review each report against 6 QA criteria")
markdown_lines.append("4. Number traceability: grep `work/m{mission_id}/docs/*.md` and `artifacts/*.json`")
markdown_lines.append("5. Generate judgment table and defect list -> write to `docs/qa_batch_613_614_615.md`")
markdown_lines.append("6. `POST /tensor/reports` to publish QA summary report")
markdown_lines.append("")
markdown_lines.append("**Toolchain**: Python 3.11 + requests + matplotlib")
markdown_lines.append("")
markdown_lines.append("## 3. Results")
markdown_lines.append("")
markdown_lines.append("### QA Judgment Summary")
markdown_lines.append("")
markdown_lines.append("| Report | Structure | Numbers | honest-FAIL | Figures | Error | Conclusion | Final |")
markdown_lines.append("|--------|-----------|---------|-------------|---------|-------|------------|-------|")
markdown_lines.append("| 613 | Missing audit | 3/3 OK | OK | Missing contour | OK | OK | **MINOR** |")
markdown_lines.append("| 614 | Complete | 3/3 OK | OK | Complete | OK | OK | **PASS** |")
markdown_lines.append("| 615 | Missing audit | 3/3 OK | OK | Missing contour | OK | OK | **MINOR** |")
markdown_lines.append("")
markdown_lines.append("### Number Traceability Detail")
markdown_lines.append("")
markdown_lines.append("| Report | Number | Report Value | Artifact Path | Artifact Value | Match |")
markdown_lines.append("|--------|--------|--------------|---------------|----------------|-------|")
markdown_lines.append("| 613 | mass_transfer_error | 0.0 kg | docs/m720/fracture_coupling_validation.md | 0.0 kg | OK |")
markdown_lines.append("| 613 | momentum_transfer_error | 0.0 kg*m/s | docs/m720/fracture_coupling_validation.md | 0.0 kg*m/s | OK |")
markdown_lines.append("| 613 | subdiv 1:32 mass error | 4.26e-14 kg | docs/m720/fracture_coupling_validation.md | 4.26e-14 kg | OK |")
markdown_lines.append("| 614 | Kerr sailing force | 889,662 N | artifacts/ridge_mechanics_results.json | 889662.45 N | OK |")
markdown_lines.append("| 614 | Timco keel drag | 4,799,974 N | artifacts/ridge_mechanics_results.json | 4799974.14 N | OK |")
markdown_lines.append("| 614 | Total resistance | 5,689,637 N | artifacts/ridge_mechanics_results.json | 5689636.59 N | OK |")
markdown_lines.append("| 615 | momentum conservation | 0.0 kg*m/s | artifacts/verification_m706.json | 0.0 | OK |")
markdown_lines.append("| 615 | settling error | 0.000709 | artifacts/verification_m706.json | 0.0007094 | OK |")
markdown_lines.append("| 615 | pytest pass rate | 12/12 | artifacts/verification_m706.json | 12/12 | OK |")
markdown_lines.append("")
markdown_lines.append("**Number traceability hit rate: 9/9 = 100%**")
markdown_lines.append("")
markdown_lines.append("## 4. Figures")
markdown_lines.append("")
markdown_lines.append("### Figure 1: QA Batch Quality Scores")
markdown_lines.append("")
markdown_lines.append("![QA Quality Scores](data:image/png;base64," + png_b64 + ")")
markdown_lines.append("")
markdown_lines.append("*Figure 1: Quality scores for three reports. 614(m711) passes all criteria; 613(m720) and 615(m706) downgraded to MINOR due to missing audit section and figure exemption statement.*")
markdown_lines.append("")
markdown_lines.append("### Figure 2: Number Traceability")
markdown_lines.append("")
markdown_lines.append("![Number Traceability](data:image/png;base64," + png_b64_2 + ")")
markdown_lines.append("")
markdown_lines.append("*Figure 2: All 9 spot-checked numbers matched artifact evidence, 100% traceability.*")
markdown_lines.append("")
markdown_lines.append("## 5. Notes & Limitations")
markdown_lines.append("")
markdown_lines.append("1. **Missing audit section is a common issue**: Both 613 and 615 lack independent audit sections. Recommend platform enforce this in report templates.")
markdown_lines.append("2. **Figure exemption statements**: Pure data transfer/backend tasks may exempt field contours, but must explicitly state the reason in the report.")
markdown_lines.append("3. **Number traceability reliability**: All three reports have complete artifacts with good number consistency.")
markdown_lines.append("4. **No RESUBMIT**: No reports with critical defects in this batch; no rework missions needed.")
markdown_lines.append("5. **Limitation**: QA covers report text and artifact numbers only; no independent recalculation of underlying code/algorithms.")
markdown_lines.append("")
markdown_lines.append("## 6. Error Assessment")
markdown_lines.append("")
markdown_lines.append("| QA Metric | Baseline | Actual | Error | <3%? |")
markdown_lines.append("|-----------|----------|--------|-------|------|")
markdown_lines.append("| Number traceability hit rate | 100% | 100% (9/9) | 0% | YES |")
markdown_lines.append("| Structure completeness | 100% | 66.7% (2/3 missing audit) | 33.3% | NO (MINOR not RESUBMIT) |")
markdown_lines.append("| Figure completeness | 100% | 33.3% (1/3 complete) | 66.7% | NO (MINOR not RESUBMIT) |")
markdown_lines.append("| honest-FAIL compliance | 100% | 100% | 0% | YES |")
markdown_lines.append("| Error assessment completeness | 100% | 100% (3/3) | 0% | YES |")
markdown_lines.append("")
markdown_lines.append("**Conclusion**: Core QA metrics (number traceability, honest-FAIL, error assessment) all pass. Structure and figure deficiencies are MINOR level, not triggering rework.")

markdown = "\n".join(markdown_lines)

# Post report
report_payload = {
    "title": "QA Batch Review 613,614,615 - Progress Report",
    "summary": "Completed QA review of batch 613/614/615. Judgments: 613=MINOR, 614=PASS, 615=MINOR. Number traceability 9/9=100% hit rate, no honest-FAIL violations. Generated docs/qa_batch_613_614_615.md.",
    "scenario_key": "report-qa",
    "mission_id": 728,
    "tags": ["QA", "progress-report"],
    "markdown": markdown,
    "figures": [
        {
            "name": "qa_quality_scores.png",
            "caption": "QA Batch Quality Scores: 613=MINOR, 614=PASS, 615=MINOR",
            "png_base64": png_b64
        },
        {
            "name": "number_traceability.png",
            "caption": "Number Traceability: 9/9 spot-checked numbers all matched artifact evidence",
            "png_base64": png_b64_2
        }
    ]
}

resp = requests.post(f"{BASE}/tensor/reports", headers=headers, json=report_payload)
print(f"POST status: {resp.status_code}")
data = resp.json()
print(f"Report ID: {data.get('id', 'N/A')}")
print(f"Title: {data.get('title', 'N/A')}")
