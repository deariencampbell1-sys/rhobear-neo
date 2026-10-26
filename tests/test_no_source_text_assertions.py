"""Guard: no test module under tests/ may read or slice the source it tests.

[Defect 6 — Apollo PR #11 round 1]  A test that asserts on source text
(`inspect.getsource` + regex, `open(module.__file__).read()` + `in`) breaks
on any reformat, passes on a wrong behaviour, and cannot catch the bug it
names.  This guard fails while any test module reads its target's source,
so the pattern cannot creep back.

Scope: every `tests/*.py` except this file (whose patterns are the very
text it greps — the acceptance grep allows this file alone).

ONE documented exception, enforced as a *shrink-only* baseline:

    PRE_EXISTING_BASELINE — three test modules that already read source on
    `main`, written by earlier lanes and NOT owned by this task
    (`test_job_recovery.py` and `test_builder_evidence.py` are inside
    W1-13's open PR #12; `test_github_app.py` belongs to no open task this
    wave).  Editing them here would be drift (l0/drift pins changed paths
    to the owning brief), so their current hit counts are frozen:

      * a baseline file may only ever contain FEWER hits than recorded —
        converting one is a pass, adding one fails the suite;
      * a file not on the baseline fails on its FIRST hit;
      * delete a baseline entry as soon as its file is converted.

    QUESTION for L0: a cross-lane cleanup task would let this list reach
    empty; until then this is the honest maximum one lane can enforce.
"""
from __future__ import annotations

import re
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SELF = Path(__file__).name

# file -> maximum number of source-reading hits allowed (shrink only).
PRE_EXISTING_BASELINE: dict[str, int] = {
    "test_job_recovery.py": 7,
    "test_builder_evidence.py": 1,
    "test_github_app.py": 1,
}
_BASELINE_FILES = frozenset(PRE_EXISTING_BASELINE)

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("inspect.getsource", re.compile(r"inspect\.getsource\s*\(")),
    ("source segment", re.compile(r"get_source_segment\s*\(")),
    ("linecache", re.compile(r"\blinecache\.")),
    ("module __file__ read",
     re.compile(r"__file__.*(?:open\s*\(|\.read\s*\(|\.read_text\s*\()")),
    ("read of a .py path", re.compile(r"open\s*\([^)\n]*\.py['\"]")),
)
_WRITE_MODE = re.compile(r",\s*['\"][wax+]['\"]")   # open(p, "w") = fixture


def scan_text(text: str) -> list[str]:
    """Source-reading hits in one text: `line N: [idiom] <line>` entries."""
    found: list[str] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        non_py_read = any(p.search(line) for _l, p in _PATTERNS[:4])
        if _WRITE_MODE.search(line) and not non_py_read:
            continue                      # writing a fixture, not reading
        for label, pattern in _PATTERNS:
            if pattern.search(line):
                found.append(f"line {lineno}: [{label}] {line.strip()[:96]}")
                break
    return found


def source_reading_hits() -> dict[str, list[str]]:
    """{module: [hit, ...]} — every source-reading line in tests/, self
    excluded."""
    hits: dict[str, list[str]] = {}
    for path in sorted(TESTS_DIR.glob("*.py")):
        if path.name == SELF:
            continue
        found = scan_text(path.read_text(encoding="utf-8", errors="replace"))
        if found:
            hits[path.name] = found
    return hits


def test_no_test_module_reads_or_slices_its_source() -> None:
    """Behavioural guard over the test tree: read every test module and
    fail on any source-reading idiom outside the shrink-only baseline."""
    hits = source_reading_hits()
    problems: list[str] = []

    for name in sorted(n for n in hits if n not in _BASELINE_FILES):
        problems.append(f"NEW OFFENDER: {name}\n    " + "\n    ".join(hits[name]))
    for name in sorted(hits):
        limit = PRE_EXISTING_BASELINE.get(name)
        if limit is not None and len(hits[name]) > limit:
            problems.append(
                f"BASELINE GREW: {name} ({len(hits[name])} > {limit})\n    "
                + "\n    ".join(hits[name]))

    assert not problems, (
        "test modules must assert on BEHAVIOUR, not on source text "
        "(convert the test; see this file's module docstring):\n  "
        + "\n  ".join(problems)
    )


def test_the_baseline_is_frozen() -> None:
    """The exception itself is guarded: exactly the three named modules,
    each at or below its frozen ceiling — growing it, or adding a new
    entry, is an L0 decision, never a test's."""
    assert set(PRE_EXISTING_BASELINE) == {
        "test_job_recovery.py", "test_builder_evidence.py",
        "test_github_app.py",
    }
    hits = source_reading_hits()
    for name, limit in PRE_EXISTING_BASELINE.items():
        assert len(hits.get(name, [])) <= limit, (
            f"{name} grew past its baseline of {limit}")


def test_the_scanner_catches_every_idiom_it_names() -> None:
    """The guard cannot be vacuous: the SAME scanner used on the tree must
    fire on every source-reading idiom, and must stay silent on benign
    lines (a scanner that never fires would make this whole file theatre)."""
    sample = "\n".join([
        "import inspect",
        "a = inspect.getsource(m.validate)",
        "b = open(m.__file__).read()",
        "c = open('/x/y/neo_protocol.py').read()",
        "d = linecache.getlines(m.__file__)",
        "e = ast.get_source_segment(src, node)",
    ])
    fired = {label for label, _p in _PATTERNS
             for entry in scan_text(sample)
             if f"[{label}]" in entry}
    assert fired == {label for label, _p in _PATTERNS}, fired

    benign = "\n".join([
        "assert value == expected",
        "path = tmp_path / 'work.py'",
        "with open(path, 'w') as fh: fh.write(text)",
        "import inspect  # not called",
    ])
    assert scan_text(benign) == [], scan_text(benign)