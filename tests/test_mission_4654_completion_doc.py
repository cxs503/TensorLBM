"""Guard ``docs/MISSION_4654_COMPLETION.md`` against silent staleness.

Mission m4654 relocated eight repository-root paths for PR #668 and reported the
result.  That first delivery (PR #672) was closed because its claims could not be
re-checked: five of the eight targets had already been re-homed by a later
mission, and one listed "offender" (``AGENTS.md``) was in fact a canonical root
file.  See the report's §1 and §5.

This module parses the machine-readable relocation table in the report and
verifies every path claim against the current working tree, so a second
relocation fails CI instead of quietly invalidating the document.  Dependency
free on purpose: stdlib + the repository's own layout policy, no solver imports.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "docs" / "MISSION_4654_COMPLETION.md"
POLICY = REPO / "docs" / "architecture" / "repository_layout_policy.json"
LAYOUT_GATE = REPO / "tools" / "check_repository_layout.py"

BEGIN = "<!-- m4654-relocation-table:begin -->"
END = "<!-- m4654-relocation-table:end -->"
COLUMNS = ("root_file", "m4654_target", "current_path", "status")
VALID_STATUS = frozenset({"relocated", "canonical-root"})

# Sections the report must keep: the four claims a reader is asked to trust.
REQUIRED_HEADINGS = (
    "## 1. Why this report exists twice",
    "## 3. Relocation table",
    "## 4. Verification",
    "## 7. Limitations",
)


def read_report() -> str:
    if not REPORT.is_file():
        raise AssertionError(f"missing completion report: {REPORT.relative_to(REPO)}")
    return REPORT.read_text(encoding="utf-8")


def parse_table(text: str) -> list[dict[str, str]]:
    """Return the relocation rows as dicts keyed by ``COLUMNS``."""
    if BEGIN not in text or END not in text:
        raise AssertionError("relocation table markers missing; the guard cannot verify the report")
    block = text.split(BEGIN, 1)[1].split(END, 1)[0]
    lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        raise AssertionError("relocation table has no data rows")
    header = tuple(cell.strip() for cell in lines[0].strip("|").split("|"))
    if header != COLUMNS:
        raise AssertionError(f"unexpected table header {header}, expected {COLUMNS}")
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        if set(line) <= set("|-: "):  # markdown separator row
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(COLUMNS):
            raise AssertionError(f"malformed relocation row: {line!r}")
        rows.append(dict(zip(COLUMNS, cells)))
    if not rows:
        raise AssertionError("relocation table parsed to zero rows")
    return rows


def load_policy() -> dict:
    return json.loads(POLICY.read_text(encoding="utf-8"))


class Mission4654CompletionDocTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = read_report()
        cls.rows = parse_table(cls.text)
        cls.policy = load_policy()

    # -- document shape ---------------------------------------------------
    def test_report_keeps_required_sections(self):
        for heading in REQUIRED_HEADINGS:
            self.assertIn(heading, self.text)

    def test_report_covers_the_eight_relocated_paths(self):
        self.assertEqual(len(self.rows), 8)

    def test_status_column_is_known(self):
        for row in self.rows:
            self.assertIn(row["status"], VALID_STATUS, row["root_file"])

    # -- path claims vs. the tree -----------------------------------------
    def test_relocated_rows_are_absent_from_the_root(self):
        for row in self.rows:
            if row["status"] != "relocated":
                continue
            name = row["root_file"]
            self.assertNotIn(name, self.policy["legacy_root_files"], name)
            self.assertFalse(
                (REPO / name).exists(),
                f"{name} is back in the repository root; the layout gate would fail",
            )

    def test_relocated_rows_exist_at_the_recorded_path(self):
        for row in self.rows:
            if row["status"] != "relocated":
                continue
            target = row["current_path"]
            self.assertTrue(
                (REPO / target).is_file(),
                f"{row['root_file']} is not at the recorded path {target}; "
                "update the report and this guard together",
            )

    def test_canonical_root_rows_are_allowed_by_policy(self):
        canonical = [row for row in self.rows if row["status"] == "canonical-root"]
        self.assertTrue(canonical, "the AGENTS.md correction must stay recorded")
        for row in canonical:
            self.assertIn(
                row["root_file"],
                self.policy["root_files"],
                f"{row['root_file']} is not a canonical root file",
            )
            self.assertTrue((REPO / row["root_file"]).is_file(), row["root_file"])

    def test_current_paths_are_unique_and_non_root(self):
        seen: set[str] = set()
        for row in self.rows:
            target = row["current_path"]
            self.assertNotIn(target, seen, f"duplicate current_path {target}")
            seen.add(target)
            if row["status"] == "relocated":
                self.assertIn("/", target, f"{target} is still a root path")
                # owned directory = a real top-level directory of the repo
                self.assertTrue((REPO / target.split("/")[0]).is_dir(), target)

    def test_relocation_kept_the_smoke_marker_importable(self):
        """`tests/test_smoke_m4630.py` imports the marker from its own directory."""
        marker = next(
            (row for row in self.rows if row["root_file"] == "smoke_marker_m4630.py"),
            None,
        )
        self.assertIsNotNone(marker)
        self.assertEqual(marker["current_path"], "tests/smoke_marker_m4630.py")
        self.assertTrue((REPO / "tests" / "test_smoke_m4630.py").is_file())

    # -- the gate claim in §4 stays checked ---------------------------------
    @unittest.skipUnless(
        (REPO / ".git").exists(), "layout gate needs a git work tree (staged exports lack one)"
    )
    def test_layout_gate_still_passes(self):
        completed = subprocess.run(
            [sys.executable, str(LAYOUT_GATE)],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"layout gate red, the report's V1 claim no longer holds:\n"
            f"{completed.stdout}{completed.stderr}",
        )
        self.assertRegex(completed.stdout, re.compile(r"Repository layout OK"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
