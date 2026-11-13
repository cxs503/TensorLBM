"""Documentation ↔ code consistency gate for the READMEs (agent/m17139).

Motivation
----------
The English/Chinese READMEs are the contract a new user copies first, yet they
are edited by hand and drift away from ``argparse`` choices, module-level
callable names and the audited benchmark artifacts.  GitHub issue class seen in
the m17139 audit: a documented ``--device mps`` that the example CLI rejects,
D3Q27 primitives described with the D2Q9/D3Q19 symbol names, and a quantitative
summary row whose numbers were untraceable to any in-repo evidence document.

This module is the *automated* part of that audit: every assertion below encodes
one README claim and re-derives the truth from the code/evidence tree, so the
claim cannot silently rot again.  Companion of ``tests/test_readme_quickstart.py``
(which pins the Python snippet by execution).

Checks
------
1. ``test_local_markdown_links_resolve`` — every relative link in
   ``README.md`` / ``README.zh-CN.md`` resolves to a file in the tree.
2. ``test_documented_scripts_exist`` — every ``examples/…``/``scripts/…``/
   ``benchmarks/…`` runnable named in a documented shell command exists.
3. ``test_documented_cli_flags_are_accepted`` — every ``--flag`` used in a
   documented ``PYTHONPATH=src python <script> …`` command is accepted by that
   script's ``argparse`` (inspected statically via :mod:`ast`, no torch import).
4. ``test_documented_device_values_are_accepted`` — the backticked device values
   listed for ``examples/cylinder_flow.py`` are a subset of its ``--device``
   choices (regression guard for the ``mps`` drift).
5. ``test_lattice_primitive_claims_resolve`` — the lattice callables the README
   attributes to ``tensorlbm.api`` / ``tensorlbm.d3q27`` really exist there.
6. ``test_quantitative_summary_numbers_are_traceable`` — the flagship 2-D
   cylinder row of the "Quantitative validation summary" table cites an in-repo
   evidence document and every number in that row appears in it (zero hand-copy).

All checks are offline and CPU-only; no solver runs, expected runtime < 2 s.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
READMES = ("README.md", "README.zh-CN.md")
# Documents that carry copy-pastable ``PYTHONPATH=src python …`` commands and are
# therefore part of the same contract as the READMEs.
COMMAND_DOCS = READMES + ("docs/software_manual.md", "docs/suboff_platform_manual.md")

# Device spellings that users can legitimately meet in this library.  The
# documented list must be covered by the example CLI's ``--device`` choices.
DEVICE_VOCAB = {"cpu", "cuda", "sdaa", "mps", "npu", "mlu", "musa"}

_CMD_RE = re.compile(r"^\s*PYTHONPATH=src python\s+(?P<script>[\w./-]+\.py)(?P<args>.*)$")
_FLAG_RE = re.compile(r"--[A-Za-z][\w-]*")
_TICK_RE = re.compile(r"`([^`]+)`")
_NUM_RE = re.compile(r"\d+\.\d+")


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _documented_commands(text: str) -> list[tuple[str, str]]:
    """Return ``(script, argument_string)`` for every documented run command.

    Shell line continuations are joined first so multi-line commands such as the
    ``examples/param_scan.py`` block are captured as a single command.
    """
    commands: list[tuple[str, str]] = []
    buffer: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.endswith("\\"):
            buffer.append(line[:-1].strip())
            continue
        buffer.append(line.strip())
        joined = " ".join(part for part in buffer if part)
        buffer = []
        match = _CMD_RE.match(joined)
        if match:
            commands.append((match.group("script"), match.group("args")))
    return commands


def _argparse_options(script: Path) -> dict[str, set[str] | None]:
    """Statically collect ``{flag: choices}`` from a script's ``argparse`` setup.

    Static (AST) inspection keeps the test fast: importing the example modules
    would pull in torch.  Parsers assembled through helper functions that are
    not visible in the file simply yield fewer flags; the flag assertions skip
    scripts for which nothing was found.
    """
    try:
        tree = ast.parse(script.read_text(encoding="utf-8"))
    except SyntaxError:  # pragma: no cover - defensive
        return {}
    options: dict[str, set[str] | None] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            continue
        choices: set[str] | None = None
        for keyword in node.keywords:
            if keyword.arg == "choices" and isinstance(
                keyword.value, (ast.List, ast.Tuple, ast.Set)
            ):
                values = {
                    element.value
                    for element in keyword.value.elts
                    if isinstance(element, ast.Constant) and isinstance(element.value, str)
                }
                choices = values
        names = [first.value]
        names += [
            extra.value
            for extra in node.args[1:]
            if isinstance(extra, ast.Constant) and isinstance(extra.value, str)
        ]
        for name in names:
            if name.startswith("-"):
                options[name] = choices
    return options


def _all_documented_commands() -> list[tuple[str, str]]:
    commands: list[tuple[str, str]] = []
    for name in COMMAND_DOCS:
        commands.extend(_documented_commands(_read(name)))
    return commands


def test_local_markdown_links_resolve() -> None:
    """Documentation links must point at files that exist in the checkout."""
    missing: list[str] = []
    for name in READMES:
        text = _read(name)
        for target in re.findall(r"\]\((?!https?://|mailto:)([^)\s]+)\)", text):
            path = target.split("#", 1)[0].strip()
            if path and not (ROOT / path).exists():
                missing.append(f"{name}: {path}")
    assert not missing, "README links point to missing files: " + ", ".join(missing)


def test_documented_scripts_exist() -> None:
    """Every script a documented command invokes must exist."""
    missing = [script for script, _ in _all_documented_commands() if not (ROOT / script).exists()]
    assert not missing, f"documented commands reference missing scripts: {sorted(set(missing))}"


def test_documented_cli_flags_are_accepted() -> None:
    """Documented CLI flags must be accepted by the script they are written for."""
    problems: list[str] = []
    for name in COMMAND_DOCS:
        for script, arguments in _documented_commands(_read(name)):
            path = ROOT / script
            if not path.exists():
                continue  # covered by test_documented_scripts_exist
            flags = _FLAG_RE.findall(arguments)
            options = _argparse_options(path)
            if not options:
                # Scripts without a static argparse (e.g. the DG demos, which take
                # no CLI arguments) are only acceptable when the documented
                # command passes no flags either.
                assert not flags, (
                    f"{name}: {script} has no argparse yet the documented command passes {flags}"
                )
                continue
            for flag in flags:
                if flag not in options:
                    problems.append(f"{name}: {script} does not accept {flag}")
            # Ordinal flags with a fixed choice set must also get a legal value
            # (this is what caught the documented ``--device mps``).
            tokens = arguments.split()
            for index, token in enumerate(tokens):
                choices = options.get(token)
                if not choices or index + 1 >= len(tokens):
                    continue
                value = tokens[index + 1]
                if value.startswith("-"):
                    continue
                if value not in choices:
                    problems.append(f"{name}: {script} {token} {value} not in {sorted(choices)}")
    assert not problems, "documented CLI usage rejected by the CLI: " + "; ".join(
        sorted(set(problems))
    )


def test_documented_device_values_are_accepted() -> None:
    """Values documented for ``--device`` must be in the example CLI's choices."""
    text = _read("README.md")
    documented: set[str] = set()
    for line in text.splitlines():
        if "`--device`" not in line:
            continue
        documented |= {token for token in _TICK_RE.findall(line) if token in DEVICE_VOCAB}
    assert documented, "README no longer documents any --device value (audit this test)"

    choices = _argparse_options(ROOT / "examples" / "cylinder_flow.py").get("--device")
    assert choices, "examples/cylinder_flow.py has no static --device choices to compare against"
    unsupported = documented - choices
    assert not unsupported, (
        f"README documents --device values {sorted(unsupported)} that "
        f"examples/cylinder_flow.py rejects (choices={sorted(choices)})"
    )


def test_lattice_primitive_claims_resolve() -> None:
    """Callables the README attributes to the lattice modules must exist there."""
    import tensorlbm.api as api
    from tensorlbm import d3q27

    text = _read("README.md")
    lines = text.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if "lattice primitives" in line),
        None,
    )
    assert start is not None, "README lost the lattice-primitive bullet (audit this test)"
    # The bullet wraps over indented continuation lines; take the whole block.
    block = [lines[start]]
    for line in lines[start + 1 :]:
        if line.startswith("- ") or not line.strip():
            break
        block.append(line)
    bullet = "\n".join(block)

    # The stable API must keep the D2Q9 / D3Q19 primitives it advertises.
    for symbol in ("equilibrium", "macroscopic", "equilibrium3d", "macroscopic3d"):
        assert symbol in api.__all__, f"tensorlbm.api lost documented export {symbol}"
        assert hasattr(api, symbol)

    # D3Q27 primitives are documented as living in tensorlbm.d3q27 with the
    # *27-suffixed names; the audit found the README using the bare D2Q9 names.
    documented_27 = [
        symbol for symbol in ("equilibrium27", "macroscopic27") if f"`{symbol}`" in bullet
    ]
    assert documented_27, (
        "README must document the D3Q27 primitives by their real names "
        "(equilibrium27 / macroscopic27) so the claim is copy-pastable"
    )
    for symbol in documented_27:
        assert hasattr(d3q27, symbol), f"tensorlbm.d3q27 lost documented symbol {symbol}"


def test_quantitative_summary_numbers_are_traceable() -> None:
    """The flagship benchmark row must cite evidence and match its numbers."""
    text = _read("README.md")
    start = text.find("## Quantitative validation summary")
    assert start != -1, "README lost the quantitative validation summary"
    section = text[start:]
    end = section.find("\n## ", 1)
    section = section if end == -1 else section[:end]

    row = next(
        (line for line in section.splitlines() if "cylinder flow" in line.lower()),
        None,
    )
    assert row is not None, "README quantitative summary lost its 2-D cylinder row"

    cited = [path for path in _TICK_RE.findall(row) if path.endswith(".md")]
    cited += re.findall(r"(docs/[\w./-]+\.md)", row)
    assert cited, (
        "2-D cylinder row must cite an in-repo evidence document (e.g. docs/CASE-KARMAN-RE100.md)"
    )

    numbers = _NUM_RE.findall(row)
    assert numbers, "2-D cylinder row must quote the measured and reference values"
    evidence_text = ""
    for path in cited:
        evidence = ROOT / path
        assert evidence.exists(), f"cited evidence document missing: {path}"
        evidence_text += evidence.read_text(encoding="utf-8")
    untraceable = [value for value in numbers if value not in evidence_text]
    assert not untraceable, (
        "numbers in the 2-D cylinder row are not traceable to the cited evidence "
        f"({cited}): {untraceable}"
    )


@pytest.mark.parametrize("name", READMES)
def test_readme_has_no_placeholder_markers(name: str) -> None:
    """Cheap hygiene guard: READMEs must not ship merge/placeholder markers."""
    text = _read(name)
    for marker in ("<<<<<<<", ">>>>>>>", "TODO(README)", "FIXME(README)"):
        assert marker not in text, f"{name} contains placeholder marker {marker!r}"
